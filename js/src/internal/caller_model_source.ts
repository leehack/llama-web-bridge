import {
  blobModelReader,
  opfsModelReader,
} from './external_model_reader.ts';
import type { ExternalModelReader, SyncModelFile } from './external_model_reader.ts';
import { ModelStagingError } from './model_staging.ts';

// The placeholder model URL a caller-supplied source travels under. It has no
// shard pattern, so the loader never expands it.
export const CALLER_MODEL_SOURCE_URL = 'caller-source://model/model.gguf';

interface OpfsFileHandle {
  readonly kind: 'file';
  readonly name?: string;
  getFile(): Promise<File>;
  createSyncAccessHandle?(): Promise<SyncModelFile & { getSize(): number }>;
}

export type CallerModelSourceInput = Blob | OpfsFileHandle;

// What a source looked like when the caller handed it over. Every later open of
// the same source, including a replay after a worker restart, is checked
// against it. A file handle also carries a sampled content fingerprint, because
// some browsers keep a file's size and modification time across an in-place edit.
export interface CallerModelSource {
  source: CallerModelSourceInput;
  size: number;
  lastModified: number | null;
  fingerprint: string | null;
}

export interface OpenedCallerModelSource {
  reader: ExternalModelReader;
  kind: CallerReaderKind;
  // Rechecks the source against its description. The loader calls it once the
  // native load returned, because a source nobody locks can change during a load.
  verify(): Promise<void>;
}

export type CallerReaderKind = 'caller_blob' | 'caller_file' | 'caller_opfs_sync' | 'caller_opfs_blob';

const GGUF_MAGIC = [0x47, 0x47, 0x55, 0x46];
const SPLIT_COUNT_KEY = 'split.count';
const METADATA_SCAN_LIMIT_BYTES = 512 * 1024 * 1024;
const SCAN_WINDOW_BYTES = 4 * 1024 * 1024;
const FINGERPRINT_STRIPE_BYTES = 64 * 1024;
const FINGERPRINT_INTERIOR_STRIPES = 16;

function tag(value: unknown): string {
  return Object.prototype.toString.call(value);
}

export function isBlobSource(value: unknown): value is Blob {
  return value instanceof Blob || tag(value) === '[object Blob]' || tag(value) === '[object File]';
}

export function isFileHandleSource(value: unknown): value is OpfsFileHandle {
  const candidate = value as Partial<OpfsFileHandle> | null;
  return candidate != null
    && typeof candidate === 'object'
    && candidate.kind === 'file'
    && typeof candidate.getFile === 'function';
}

export function isCallerModelSource(value: unknown): value is CallerModelSource {
  const candidate = value as Partial<CallerModelSource> | null;
  return candidate != null
    && typeof candidate === 'object'
    && (isBlobSource(candidate.source) || isFileHandleSource(candidate.source))
    && Number.isSafeInteger(candidate.size)
    && (candidate.lastModified === null || Number.isFinite(candidate.lastModified))
    && (candidate.fingerprint === null || typeof candidate.fingerprint === 'string');
}

function invalid(message: string): ModelStagingError {
  return new ModelStagingError('MODEL_SOURCE_INVALID', message);
}

function changed(message: string): ModelStagingError {
  return new ModelStagingError('MODEL_SOURCE_CHANGED', message);
}

async function snapshotHandle(handle: OpfsFileHandle): Promise<File> {
  try {
    return await handle.getFile();
  } catch (error) {
    const name = (error as { name?: string } | null)?.name;
    if (name === 'NotFoundError') {
      throw changed('The model file is no longer there.');
    }
    throw invalid(`The model file handle cannot be read: ${(error as Error)?.message ?? error}`);
  }
}

type ReadAt = (offset: number, length: number) => Promise<Uint8Array>;

// The byte ranges a fingerprint samples: the head, the tail and evenly spaced
// stripes between them. A file small enough for every stripe to touch the next is
// covered whole. Bytes between stripes of a larger file are not sampled.
export function fingerprintRanges(size: number): Array<[number, number]> {
  const stripe = FINGERPRINT_STRIPE_BYTES;
  const ranges: Array<[number, number]> = [[0, Math.min(stripe, size)]];
  for (let index = 1; index <= FINGERPRINT_INTERIOR_STRIPES; index += 1) {
    const start = Math.floor((index * Math.max(size - stripe, 0)) / (FINGERPRINT_INTERIOR_STRIPES + 1));
    ranges.push([start, Math.min(start + stripe, size)]);
  }
  ranges.push([Math.max(size - stripe, 0), size]);
  const merged: Array<[number, number]> = [];
  for (const range of ranges) {
    const last = merged[merged.length - 1];
    if (last && range[0] <= last[1]) {
      last[1] = Math.max(last[1], range[1]);
    } else {
      merged.push([range[0], range[1]]);
    }
  }
  return merged;
}

// Cost is bounded by the stripe count whatever the file size, so a multi-gigabyte
// model is never read whole. Not a cryptographic digest: it detects accidental or
// ordinary edits, not an adversary.
export async function fingerprintSource(size: number, readAt: ReadAt): Promise<string> {
  let high = 0x811c9dc5 ^ size;
  let low = 0x9e3779b1 + (size >>> 0);
  for (const [start, end] of fingerprintRanges(size)) {
    high = Math.imul(high ^ start, 0x01000193) >>> 0;
    for (let offset = start; offset < end; offset += FINGERPRINT_STRIPE_BYTES * 4) {
      const length = Math.min(FINGERPRINT_STRIPE_BYTES * 4, end - offset);
      const bytes = await readAt(offset, length);
      if (bytes.length !== length) throw new Error('The model source returned a short read.');
      for (let index = 0; index < length; index += 1) {
        const byte = bytes[index]!;
        high = Math.imul(high ^ byte, 0x01000193) >>> 0;
        low = Math.imul(low ^ byte, 0x9e3779b1);
        low ^= low >>> 15;
        low >>>= 0;
      }
    }
  }
  return `${high.toString(16).padStart(8, '0')}${low.toString(16).padStart(8, '0')}`;
}

function blobReadAt(blob: Blob): ReadAt {
  return async (offset, length) => new Uint8Array(await blob.slice(offset, offset + length).arrayBuffer());
}

function readerReadAt(reader: ExternalModelReader): ReadAt {
  return async (offset, length) => {
    const bytes = new Uint8Array(length);
    await reader.read(offset, length, (chunk) => bytes.set(chunk));
    if (reader.failure) throw reader.failure.error;
    return bytes;
  };
}

async function verifyFingerprint(description: CallerModelSource, readAt: ReadAt): Promise<void> {
  if (description.fingerprint === null) return;
  let current: string;
  try {
    current = await fingerprintSource(description.size, readAt);
  } catch (error) {
    if ((error as { name?: string } | null)?.name === 'AbortError') throw error;
    throw changed(`The model source could not be sampled and may have changed: ${(error as Error)?.message ?? error}`);
  }
  if (current !== description.fingerprint) {
    throw changed('The model source changed: its content differs from when it was loaded.');
  }
}

// Describes a caller's source as it is now. The facade calls this before it
// forwards the source, so a worker restart or a CPU fallback replays the
// caller's own reference and compares it with this description.
export async function describeCallerModelSource(source: unknown): Promise<CallerModelSource> {
  if (isBlobSource(source)) {
    if (source.size <= 0) throw invalid('The model Blob is empty.');
    const lastModified = tag(source) === '[object File]' && Number.isFinite((source as File).lastModified)
      ? (source as File).lastModified
      : null;
    return { source, size: source.size, lastModified, fingerprint: null };
  }
  if (isFileHandleSource(source)) {
    const file = await snapshotHandle(source);
    if (file.size <= 0) throw invalid('The model file is empty.');
    return {
      source,
      size: file.size,
      lastModified: file.lastModified,
      fingerprint: await fingerprintFile(file),
    };
  }
  throw invalid('A model source must be a Blob, a File or a FileSystemFileHandle.');
}

async function fingerprintFile(file: Blob): Promise<string> {
  try {
    return await fingerprintSource(file.size, blobReadAt(file));
  } catch (error) {
    throw invalid(`The model file cannot be read: ${(error as Error)?.message ?? error}`);
  }
}

function checkSnapshot(description: CallerModelSource, file: Blob & { lastModified?: number }): void {
  if (file.size !== description.size) {
    throw changed(
      `The model source changed: it was ${description.size} bytes and is now ${file.size} bytes.`,
    );
  }
  if (description.lastModified !== null
    && Number.isFinite(file.lastModified)
    && file.lastModified !== description.lastModified) {
    throw changed('The model source changed: its modification time differs from when it was loaded.');
  }
}

async function isOpfsHandle(handle: OpfsFileHandle): Promise<boolean> {
  try {
    const root = await navigator.storage.getDirectory();
    return (await root.resolve(handle as unknown as FileSystemHandle)) !== null;
  } catch (_) {
    return false;
  }
}

// Opens a reader the core can attach to a staging file. The bridge owns the
// reader and the sync access handle it creates; the caller's own handle and
// Blob are only read.
//
// A Blob or File cannot change. A file handle can, and only the exclusive sync
// access handle a worker holds keeps it still; the loader keeps that handle open
// until the model unloads. Without it (the main thread, or a handle outside OPFS)
// the bridge reads the live file and relies on the size, the modification time and
// the sampled fingerprint to notice an edit.
export async function openCallerModelSource(
  description: CallerModelSource,
  options: { worker: boolean; signal: AbortSignal | null },
): Promise<OpenedCallerModelSource> {
  const { source } = description;
  if (isBlobSource(source)) {
    checkSnapshot(description, source);
    return {
      reader: blobModelReader(source, options.signal),
      kind: tag(source) === '[object File]' ? 'caller_file' : 'caller_blob',
      verify: async () => checkSnapshot(description, source),
    };
  }

  const handle = source as OpfsFileHandle;
  let sync: (SyncModelFile & { getSize(): number }) | null = null;
  if (options.worker && typeof handle.createSyncAccessHandle === 'function') {
    try {
      sync = await handle.createSyncAccessHandle();
    } catch (error) {
      const name = (error as { name?: string } | null)?.name ?? 'Error';
      // Browsers disagree on the name of the refusal: Chromium and Firefox throw
      // NoModificationAllowedError and WebKit throws InvalidStateError. Inside OPFS
      // any refusal means the file is held elsewhere, and a live read would not be safe.
      if (name === 'NoModificationAllowedError' || await isOpfsHandle(handle)) {
        throw new ModelStagingError(
          'MODEL_SOURCE_BUSY',
          `The model file cannot be locked for reading (${name}). Close any other sync access handle on it and load the model again.`,
        );
      }
      // A handle outside OPFS, such as a file picker handle, has no sync access handle.
    }
  }
  try {
    const file = await snapshotHandle(handle);
    checkSnapshot(description, file);
    if (sync) {
      if (sync.getSize() !== description.size) {
        throw changed('The model source changed while it was being opened.');
      }
      const reader = opfsModelReader(sync, description.size, options.signal);
      sync = null;
      try {
        await verifyFingerprint(description, readerReadAt(reader));
      } catch (error) {
        reader.close();
        throw error;
      }
      return {
        reader,
        kind: 'caller_opfs_sync',
        verify: async () => {
          checkSnapshot(description, await snapshotHandle(handle));
          await verifyFingerprint(description, readerReadAt(reader));
        },
      };
    }
    await verifyFingerprint(description, blobReadAt(file));
    return {
      reader: blobModelReader(file, options.signal),
      kind: 'caller_opfs_blob',
      verify: async () => {
        const current = await snapshotHandle(handle);
        checkSnapshot(description, current);
        await verifyFingerprint(description, blobReadAt(current));
      },
    };
  } catch (error) {
    try {
      sync?.close();
    } catch (_) {
      // The handle is already unusable.
    }
    throw error;
  }
}

class WindowReader {
  private window = new Uint8Array(0);
  private windowStart = 0;
  position = 0;
  declare private readonly reader: ExternalModelReader;
  declare private readonly size: number;

  constructor(reader: ExternalModelReader, size: number) {
    this.reader = reader;
    this.size = size;
  }

  private async fill(at: number, length: number): Promise<void> {
    const wanted = Math.min(this.size - at, Math.max(length, SCAN_WINDOW_BYTES));
    if (wanted < length) throw invalid('The model file ends inside its GGUF metadata.');
    const bytes = new Uint8Array(wanted);
    await this.reader.read(at, wanted, (chunk) => bytes.set(chunk));
    if (this.reader.failure) throw this.reader.failure.error;
    this.window = bytes;
    this.windowStart = at;
  }

  async bytes(length: number): Promise<Uint8Array> {
    const from = this.position - this.windowStart;
    if (from < 0 || from + length > this.window.length) await this.fill(this.position, length);
    const start = this.position - this.windowStart;
    this.position += length;
    return this.window.subarray(start, start + length);
  }

  skip(length: number): void {
    this.position += length;
    if (this.position > this.size) throw invalid('The model file ends inside its GGUF metadata.');
  }

  async u32(): Promise<number> {
    const view = await this.bytes(4);
    return new DataView(view.buffer, view.byteOffset, 4).getUint32(0, true);
  }

  async u64(): Promise<number> {
    const view = await this.bytes(8);
    const value = new DataView(view.buffer, view.byteOffset, 8).getBigUint64(0, true);
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw invalid('The GGUF metadata is corrupt.');
    return Number(value);
  }

  async string(): Promise<string> {
    const length = await this.u64();
    if (length > 1024 * 1024) throw invalid('The GGUF metadata is corrupt.');
    return new TextDecoder().decode(await this.bytes(length));
  }
}

const FIXED_VALUE_BYTES: Record<number, number> = {
  0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
};

async function skipValue(cursor: WindowReader, type: number): Promise<void> {
  const fixed = FIXED_VALUE_BYTES[type];
  if (fixed !== undefined) {
    cursor.skip(fixed);
  } else if (type === 8) {
    cursor.skip(await cursor.u64());
  } else if (type === 9) {
    const elementType = await cursor.u32();
    const count = await cursor.u64();
    const elementBytes = FIXED_VALUE_BYTES[elementType];
    if (elementBytes !== undefined) {
      cursor.skip(elementBytes * count);
    } else {
      for (let index = 0; index < count; index += 1) await skipValue(cursor, elementType);
    }
  } else {
    throw invalid(`The GGUF metadata has an unknown value type (${type}).`);
  }
}

async function readSplitCount(cursor: WindowReader, type: number): Promise<number> {
  const view = await cursor.bytes(FIXED_VALUE_BYTES[type] ?? 0);
  const data = new DataView(view.buffer, view.byteOffset, view.byteLength);
  switch (type) {
    case 0: return data.getUint8(0);
    case 2: return data.getUint16(0, true);
    case 4: return data.getUint32(0, true);
    case 5: return data.getInt32(0, true);
    case 10: return Number(data.getBigUint64(0, true));
    default: return 0;
  }
}

// Checks the GGUF magic and the `split.count` metadata of a model the core is
// about to load. A split model needs its sibling files, which a single source
// cannot supply.
export async function inspectCallerModel(reader: ExternalModelReader): Promise<void> {
  const cursor = new WindowReader(reader, reader.size);
  try {
    const magic = await cursor.bytes(4);
    if (!GGUF_MAGIC.every((byte, index) => magic[index] === byte)) {
      throw invalid('The model source is not a GGUF file.');
    }
    await cursor.u32();
    await cursor.u64();
    const keyCount = await cursor.u64();
    for (let index = 0; index < keyCount; index += 1) {
      if (cursor.position > METADATA_SCAN_LIMIT_BYTES) return;
      const key = await cursor.string();
      const type = await cursor.u32();
      if (key === SPLIT_COUNT_KEY) {
        if (await readSplitCount(cursor, type) > 1) {
          throw new ModelStagingError(
            'MODEL_SOURCE_SPLIT_UNSUPPORTED',
            'A caller model source cannot be a split GGUF. Load the shards by URL or merge them into one file.',
          );
        }
        return;
      }
      await skipValue(cursor, type);
    }
  } catch (error) {
    if (error instanceof ModelStagingError || (error as { name?: string } | null)?.name === 'AbortError') throw error;
    const message = error instanceof Error ? error.message : String(error);
    throw new ModelStagingError('MODEL_READ_FAILED', `The model source could not be read: ${message}`);
  }
}
