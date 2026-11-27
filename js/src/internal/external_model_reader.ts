import { throwIfAborted } from './abort.ts';
import type { CachedModelParts } from './download.ts';

export const EXTERNAL_MODEL_READ_BYTES = 16 * 1024 * 1024;
const BLOB_WINDOW_BYTES = 4 * 1024 * 1024;

type CopyReaderBytes = (bytes: Uint8Array) => void;

export interface ExternalModelReader {
  readonly mode: 0 | 1;
  readonly size: number;
  readonly signal: AbortSignal | null;
  read(offset: number, length: number, copy: CopyReaderBytes): void | Promise<void>;
  close(): void;
  failure?: { code: number; error: unknown };
}

function releaseBuffer(buffer?: ArrayBuffer): void {
  try {
    (buffer as (ArrayBuffer & { transfer?: (size: number) => ArrayBuffer }) | undefined)?.transfer?.(0);
  } catch (_) {
    // The browser can still reclaim a buffer after its last reference is dropped.
  }
}

export function blobModelReader(blob: Blob, signal: AbortSignal | null): ExternalModelReader {
  let source: Blob | null = blob;
  let window: Uint8Array<ArrayBuffer> | null = null;
  let windowStart = 0;
  return {
    mode: 1,
    size: blob.size,
    signal,
    read(offset, length, copy) {
      if (!source) throw new Error('The model Blob reader is closed.');
      throwIfAborted(signal, 'Model read was cancelled.');
      const from = offset - windowStart;
      if (window && from >= 0 && from + length <= window.byteLength) {
        copy(window.subarray(from, from + length));
        return;
      }

      const keep = length < BLOB_WINDOW_BYTES;
      if (keep) {
        releaseBuffer(window?.buffer);
        window = null;
      }
      const readBytes = keep ? Math.min(BLOB_WINDOW_BYTES, source.size - offset) : length;
      return source.slice(offset, offset + readBytes).arrayBuffer().then((buffer) => {
        try {
          if (!source) throw new Error('The model Blob reader is closed.');
          throwIfAborted(signal, 'Model read was cancelled.');
          if (buffer.byteLength !== readBytes) throw new Error('Model Blob returned a short read.');
          const bytes = new Uint8Array(buffer);
          copy(bytes.subarray(0, length));
          if (keep) {
            window = bytes;
            windowStart = offset;
          }
        } finally {
          if (buffer !== window?.buffer) releaseBuffer(buffer);
        }
      });
    },
    close() {
      source = null;
      releaseBuffer(window?.buffer);
      window = null;
    },
  };
}

export interface SyncModelFile {
  read(buffer: Uint8Array<ArrayBuffer>, options: { at: number }): number;
  close(): void;
}

export function opfsModelReader(
  syncHandle: SyncModelFile,
  size: number,
  signal: AbortSignal | null,
): ExternalModelReader {
  let bounce: Uint8Array<ArrayBuffer> | null = null;
  let handle: typeof syncHandle | null = syncHandle;
  return {
    mode: 0,
    size,
    signal,
    read(offset, length, copy) {
      if (!handle) throw new Error('The OPFS model reader is closed.');
      throwIfAborted(signal, 'Model read was cancelled.');
      // Browser implementations do not uniformly accept SharedArrayBuffer views.
      bounce ??= new Uint8Array(Math.min(size, EXTERNAL_MODEL_READ_BYTES));
      const chunk = bounce.subarray(0, length);
      if (handle.read(chunk, { at: offset }) !== length) throw new Error('OPFS model returned a short read.');
      copy(chunk);
    },
    close() {
      const ownedHandle = handle;
      handle = null;
      releaseBuffer(bounce?.buffer);
      bounce = null;
      ownedHandle?.close();
    },
  };
}

export function cachePartsModelReader(parts: CachedModelParts, signal: AbortSignal | null): ExternalModelReader {
  let source: CachedModelParts | null = parts;
  let index = -1;
  let cached: Uint8Array<ArrayBuffer> | null = null;
  // WasmFS supplies one destination per bounded read. Reuse a private buffer
  // only for requests that straddle a Cache Storage part.
  let crossing: Uint8Array<ArrayBuffer> | null = null;

  const getPart = async (wanted: number) => {
    if (!source) throw new Error('The model cache reader is closed.');
    throwIfAborted(signal, 'Model read was cancelled.');
    releaseBuffer(cached?.buffer);
    cached = null;
    const buffer = await source.readPart(wanted);
    try {
      if (!source) throw new Error('The model cache reader is closed.');
      throwIfAborted(signal, 'Model read was cancelled.');
      cached = new Uint8Array(buffer);
      index = wanted;
      return cached;
    } finally {
      if (buffer !== cached?.buffer) releaseBuffer(buffer);
    }
  };

  return {
    mode: 1,
    size: parts.decodedBytes,
    signal,
    read(offset, length, copy) {
      const first = Math.floor(offset / parts.partBytes);
      const from = offset % parts.partBytes;
      if (cached && index === first && from + length <= cached.byteLength) {
        throwIfAborted(signal, 'Model read was cancelled.');
        copy(cached.subarray(from, from + length));
        return;
      }
      return (async () => {
        const start = cached && index === first ? cached : await getPart(first);
        if (from + length <= start.byteLength) {
          copy(start.subarray(from, from + length));
          return;
        }
        crossing ??= new Uint8Array(EXTERNAL_MODEL_READ_BYTES);
        const output = crossing;
        let copied = 0;
        let part = start;
        let partOffset = from;
        for (let next = first + 1; copied < length; ++next) {
          const count = Math.min(part.byteLength - partOffset, length - copied);
          output.set(part.subarray(partOffset, partOffset + count), copied);
          copied += count;
          if (copied < length) part = await getPart(next);
          partOffset = 0;
        }
        copy(output.subarray(0, length));
      })();
    },
    close() {
      source = null;
      releaseBuffer(cached?.buffer);
      cached = null;
      releaseBuffer(crossing?.buffer);
      crossing = null;
    },
  };
}
