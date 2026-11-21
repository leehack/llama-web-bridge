// Streaming model downloads with progress into the Emscripten filesystem.

import type { BridgeProgressEvent } from '../llama_webgpu_bridge.d.ts';
import { createAbortError, throwIfAborted } from './abort.ts';
import { parsePositiveInteger } from './parse.ts';

// The subset of the Emscripten WASMFS `FS` API these helpers use. Stream
// handles are opaque to JS.
export interface EmscriptenFs {
  readdir(path: string): unknown;
  mkdir(path: string): void;
  unlink(path: string): void;
  open(path: string, flags: string): unknown;
  write(
    stream: unknown,
    buffer: Uint8Array,
    offset: number,
    length: number,
    position: number | bigint,
  ): number;
  close(stream: unknown): void;
  writeFile(path: string, data: Uint8Array): void;
}

export type ProgressCallback = (event: BridgeProgressEvent) => void;

export interface DrainOptions {
  chunkTimeoutMs?: unknown;
  releaseConsumedChunks?: boolean;
}

export interface DirectFileChunkWriter {
  writeChunk(chunk: Uint8Array, position: number): number | void;
  close?: () => void;
  assertMountOwned?: () => void;
  heapResident?: boolean;
  heapUnavailableReason?: string;
}

export interface WriteResponseOptions {
  totalBytes?: unknown;
  useBigIntPosition?: boolean;
  skipMemoryReservation?: boolean;
  directWriter?: DirectFileChunkWriter | null;
  // Free each network chunk once it is copied. Only for a response nothing
  // else reads: a cache tee shares these buffers.
  releaseConsumedChunks?: boolean;
  startOffset?: unknown;
  heapFileBytes?: number | null;
  preservePartialOnError?: boolean;
  chunkTimeoutMs?: unknown;
  signal?: AbortSignal | null;
  abortMessage?: string;
}

// A transfer failure annotated with how far the file got, for resumable loads.
type TransferError = Error & { llamadartLoadedBytes?: number; llamadartFilePath?: string };

export function hasReadableResponseStream(response: Response | null | undefined): boolean {
  return !!(
    response
    && response.body
    && typeof response.body.getReader === 'function'
  );
}

export function sumProgressValues(values: Iterable<unknown> | null | undefined): number {
  let total = 0;
  for (const value of values || []) {
    const numeric = Number(value);
    if (Number.isFinite(numeric) && numeric > 0) {
      total += numeric;
    }
  }
  return total;
}

export function createModelTransferLengthError(message: string): Error {
  const error = new Error(message);
  error.name = 'ModelTransferLengthError';
  return error;
}

function writeDirectChunk(
  directWriter: DirectFileChunkWriter,
  chunk: Uint8Array,
  position: number,
  onBytesWritten?: (written: number) => void,
): number {
  let offset = 0;
  while (offset < chunk.byteLength) {
    const remaining = offset === 0 ? chunk : chunk.subarray(offset);
    const result = directWriter.writeChunk(remaining, position + offset);
    const written = result === undefined ? remaining.byteLength : Number(result);
    if (!Number.isInteger(written) || written <= 0 || written > remaining.byteLength) {
      throw new Error(
        `Direct file writer returned invalid byte count (${String(result)} for ${remaining.byteLength} bytes).`,
      );
    }
    offset += written;
    onBytesWritten?.(written);
  }
  return offset;
}

// Safari keeps consumed stream chunks alive long after they are dropped;
// detaching returns the memory at once.
function releaseChunkBuffer(chunk: Uint8Array): void {
  const buffer = chunk.buffer as ArrayBuffer & { transfer?: (length: number) => ArrayBuffer };
  if (
    chunk.byteOffset !== 0
    || chunk.byteLength !== buffer.byteLength
    || typeof buffer.transfer !== 'function'
  ) {
    return;
  }
  try {
    buffer.transfer(0);
  } catch (_) {
    // A shared or non-detachable buffer stays as it is.
  }
}

function parseTotalFromContentRangeHeader(contentRangeHeader: string | null): number {
  if (typeof contentRangeHeader !== 'string' || contentRangeHeader.length === 0) {
    return 0;
  }

  const slash = contentRangeHeader.lastIndexOf('/');
  if (slash < 0 || slash + 1 >= contentRangeHeader.length) {
    return 0;
  }

  return parsePositiveInteger(contentRangeHeader.slice(slash + 1));
}

export function inferResponseTotalBytes(
  response: Response | null | undefined,
  loadedFallback: unknown = 0,
): number {
  if (!response || !response.headers) {
    return parsePositiveInteger(loadedFallback);
  }

  const linkedSize = parsePositiveInteger(response.headers.get('x-linked-size'));
  if (linkedSize > 0) {
    return linkedSize;
  }

  const contentRangeTotal = parseTotalFromContentRangeHeader(
    response.headers.get('content-range'),
  );
  if (contentRangeTotal > 0) {
    return contentRangeTotal;
  }

  const contentLength = parsePositiveInteger(response.headers.get('content-length'));
  if (contentLength > 0) {
    return contentLength;
  }

  return parsePositiveInteger(loadedFallback);
}

export interface ResumeIdentity {
  url: string;
  strongEtag: string | null;
  encoding: string | null;
  total: number | null;
  initialSpanBytes: number | null;
  resumedStrongEtags: Map<string, string>;
}

type ResumeVerdict =
  | { verdict: 'changed'; reason: 'url' | 'range' | 'etag' }
  | {
    verdict: 'validated' | 'unvalidated';
    total: number | null;
    expectedBytes: number | null;
    validator: { encoding: string; etag: string } | null;
  };

export function strongEtagOf(value: string | null): string | null {
  return value !== null && /^"[\x21\x23-\x7E\x80-\xFF]*"$/.test(value) ? value : null;
}

function contentEncodingOf(response: Response): string | null {
  const encoding = response.headers.get('content-encoding');
  if (encoding !== null) {
    const codings = encoding.split(',')
      .map((coding) => coding.replace(/^[ \t]+|[ \t]+$/g, '').toLowerCase())
      .filter(Boolean);
    return codings.join(',') || 'identity';
  }
  if (response.type === 'basic') return 'identity';

  // A CORS response can expose Content-Length while hiding Content-Encoding.
  if (response.type === 'cors') {
    const exposed = response.headers.get('access-control-expose-headers')?.toLowerCase().split(',');
    if (exposed?.some((name) => ['*', 'content-encoding'].includes(name.trim()))) return 'identity';
  }
  return null;
}

function parseByteLength(value: string | null): number | null {
  if (value === null || !/^\d+$/.test(value)) return null;
  const length = Number(value);
  return Number.isSafeInteger(length) ? length : null;
}

function parseByteRange(value: string | null): { start: number; end: number; total: number | null } | null {
  const match = value === null ? null : /^bytes (\d+)-(\d+)\/(\d+|\*)$/i.exec(value);
  if (!match) return null;
  const start = parseByteLength(match[1]);
  const end = parseByteLength(match[2]);
  const total = parseByteLength(match[3]);
  if (start === null || end === null || (total === null && match[3] !== '*')) return null;
  return { start, end, total };
}

export function captureResumeIdentity(
  response: Response,
  finalUrl: string,
): ResumeIdentity | null {
  const encoding = contentEncodingOf(response);
  const range = response.status === 206 && encoding === 'identity'
    ? parseByteRange(response.headers.get('content-range'))
    : null;
  const rangeTotal = range?.total ?? null;
  const linkedSize = parseByteLength(response.headers.get('x-linked-size'));
  if (rangeTotal !== null && linkedSize !== null && rangeTotal !== linkedSize) return null;

  let total = linkedSize;
  if (total === null && encoding === 'identity') {
    total = response.status === 206
      ? rangeTotal
      : parseByteLength(response.headers.get('content-length'));
  }
  return {
    url: finalUrl,
    strongEtag: strongEtagOf(response.headers.get('etag')),
    encoding,
    total,
    initialSpanBytes: range?.start === 0 ? range.end + 1 : null,
    resumedStrongEtags: new Map(),
  };
}

export function classifyResumeResponse(
  identity: ResumeIdentity,
  response: Response,
  finalUrl: string,
  offset: number,
  minimumBytes = 0,
): ResumeVerdict {
  if (finalUrl !== identity.url) return { verdict: 'changed', reason: 'url' };

  const encoding = contentEncodingOf(response);
  const contentRange = response.headers.get('content-range');
  const range = encoding === 'identity' ? parseByteRange(contentRange) : null;
  const length = encoding === 'identity' ? parseByteLength(response.headers.get('content-length')) : null;
  // X-Linked-Size describes the model artifact, independent of HTTP encoding.
  const linkedSize = parseByteLength(response.headers.get('x-linked-size'));
  const total = range?.total ?? linkedSize ?? identity.total;
  if (total !== null && (total < minimumBytes
    || (identity.total !== null && total !== identity.total)
    || (linkedSize !== null && total !== linkedSize))) {
    return { verdict: 'changed', reason: 'range' };
  }

  let rangeVerified = false;
  if (range) {
    if (range.start !== offset || range.end < range.start
      || (total !== null && range.end !== total - 1)
      || (length !== null && length !== range.end - range.start + 1)) {
      return { verdict: 'changed', reason: 'range' };
    }
    rangeVerified = total !== null;
  } else if (length !== null && total !== null) {
    if (length !== total - offset) return { verdict: 'changed', reason: 'range' };
    rangeVerified = contentRange === null;
  }

  const etag = strongEtagOf(response.headers.get('etag'));
  // Strong validators identify the encoded representation.
  const originalTag = encoding !== null && identity.encoding === encoding ? identity.strongEtag : null;
  const validator = etag !== null && encoding !== null ? { etag, encoding } : null;
  if (validator) {
    const earlierTag = originalTag ?? identity.resumedStrongEtags.get(validator.encoding);
    if (earlierTag !== undefined && earlierTag !== validator.etag) {
      return { verdict: 'changed', reason: 'etag' };
    }
  }

  const tailBytes = range ? range.end + 1 : (length === null ? null : offset + length);
  return {
    // A tag first seen on a range can check later ranges but cannot verify the original prefix.
    verdict: validator && originalTag !== null && rangeVerified ? 'validated' : 'unvalidated',
    total,
    expectedBytes: total ?? tailBytes,
    validator,
  };
}

export function isRetryableStreamNetworkError(error: unknown): boolean {
  const text = String(error || '').toLowerCase();
  return text.includes('network error')
    || text.includes('failed to fetch')
    || text.includes('networkerror')
    || text.includes('err_network_io_suspended')
    || text.includes('the network connection was lost')
    || text.includes('connection reset')
    || text.includes('timeout')
    || text.includes('timed out');
}

async function readStreamChunkWithTimeout(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  timeoutMs: unknown,
  label = 'stream read',
): Promise<ReadableStreamReadResult<Uint8Array>> {
  const resolvedTimeout = Number(timeoutMs);
  if (!Number.isFinite(resolvedTimeout) || resolvedTimeout <= 0) {
    return reader.read();
  }

  let timeoutHandle: ReturnType<typeof globalThis.setTimeout> | null = null;
  try {
    return await Promise.race([
      reader.read(),
      new Promise<never>((_, reject) => {
        timeoutHandle = globalThis.setTimeout(() => {
          reject(new Error(`${label} timeout (${resolvedTimeout}ms)`));
        }, resolvedTimeout);
      }),
    ]);
  } finally {
    if (timeoutHandle != null) {
      globalThis.clearTimeout(timeoutHandle);
    }
  }
}

export async function drainResponseWithProgress(
  response: Response,
  progressCallback?: ProgressCallback | null,
  options: DrainOptions = {},
): Promise<number> {
  const total = Number(response.headers.get('content-length')) || 0;
  const chunkTimeoutMs = parsePositiveInteger(options.chunkTimeoutMs);

  if (!response.body || typeof response.body.getReader !== 'function') {
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (typeof progressCallback === 'function') {
      progressCallback({ loaded: bytes.byteLength, total: total || bytes.byteLength });
    }
    return bytes.byteLength;
  }

  const reader = response.body.getReader();
  let loaded = 0;
  let lastBucket = -1;

  try {
    while (true) {
      const { done, value } = await readStreamChunkWithTimeout(
        reader,
        chunkTimeoutMs,
        'response drain read',
      );
      if (done) {
        break;
      }

      if (!value || value.length === 0) {
        continue;
      }

      loaded += value.length;
      if (options.releaseConsumedChunks === true) releaseChunkBuffer(value);
      if (typeof progressCallback === 'function') {
        const effectiveTotal = total || loaded;
        const bucket = effectiveTotal > 0
          ? Math.floor((loaded / effectiveTotal) * 100)
          : -1;
        if (bucket > lastBucket) {
          lastBucket = bucket;
          progressCallback({ loaded, total: effectiveTotal });
        }
      }
    }

    if (typeof progressCallback === 'function') {
      progressCallback({ loaded, total: total || loaded });
    }
  } catch (error) {
    void reader.cancel(error).catch(() => {});
    throw error;
  }

  return loaded;
}

export const DEFAULT_CACHE_TEE_LAG_BYTES = 8 * 1024 * 1024;

export interface CacheTee {
  // The copy the caller reads; it keeps the source's status, headers and url.
  response: Response;
  // The copy to hand to `cache.put`.
  cacheResponse: Response;
  // Stops feeding the cache copy and errors it, so the cache commits no entry.
  detachCache(reason: unknown): void;
  // Whether the caller cancelled its copy before the body ended.
  readonly cancelled: boolean;
  readonly decodedBytes: number;
}

/**
 * Splits a response body between the caller and `cache.put`.
 *
 * `Response.clone()` buffers whatever one branch has not read yet, so a caller
 * that waits for the cache, or a cache writer slower than the caller, holds up
 * to the whole body in memory. Here the source is read only when the caller
 * reads, and only while the cache copy is less than `maxCacheLagBytes` behind,
 * so a slow cache writer slows the transfer instead. Cancelling the caller's
 * copy errors the cache copy, so an abandoned transfer never commits an entry.
 */
export function teeResponseForCache(
  source: Response,
  maxCacheLagBytes: number = DEFAULT_CACHE_TEE_LAG_BYTES,
  recordId?: string,
): CacheTee {
  const reader = source.body!.getReader();
  let cacheController: ReadableStreamDefaultController<Uint8Array> | null = null;
  let wakeReader: (() => void) | null = null;
  let finished = false;
  let cancelled = false;
  let decodedBytes = 0;

  const wake = () => {
    const resolve = wakeReader;
    wakeReader = null;
    resolve?.();
  };
  const detachCache = (reason: unknown) => {
    const controller = cacheController;
    cacheController = null;
    if (controller) {
      try {
        controller.error(reason);
      } catch (_) {
        // already closed or errored
      }
    }
    wake();
  };
  const cacheIsFull = () => {
    const desiredSize = cacheController?.desiredSize;
    return desiredSize != null && desiredSize <= 0;
  };

  const cacheStream = new ReadableStream<Uint8Array>(
    {
      start(controller) {
        cacheController = controller;
      },
      pull() {
        wake();
      },
      cancel() {
        cacheController = null;
        wake();
      },
    },
    new ByteLengthQueuingStrategy({ highWaterMark: Math.max(1, maxCacheLagBytes) }),
  );

  const bodyStream = new ReadableStream<Uint8Array>(
    {
      async pull(controller) {
        while (!finished && cacheIsFull()) {
          await new Promise<void>((resolve) => {
            wakeReader = resolve;
          });
        }
        if (finished) {
          return;
        }
        let result: ReadableStreamReadResult<Uint8Array>;
        try {
          result = await reader.read();
        } catch (error) {
          finished = true;
          detachCache(error);
          throw error;
        }
        if (finished) {
          return;
        }
        if (result.done) {
          finished = true;
          controller.close();
          try {
            cacheController?.close();
          } catch (_) {
            // cancelled by the cache writer
          }
          cacheController = null;
          return;
        }
        decodedBytes += result.value.byteLength;
        controller.enqueue(result.value);
        try {
          cacheController?.enqueue(result.value);
        } catch (_) {
          cacheController = null;
        }
      },
      cancel(reason) {
        finished = true;
        cancelled = true;
        detachCache(reason ?? new Error('The response was cancelled before the cache copy completed.'));
        return reader.cancel(reason);
      },
    },
    { highWaterMark: 0 },
  );

  const init = { status: source.status, statusText: source.statusText, headers: source.headers };
  const response = new Response(bodyStream, init);
  // Redirect and encoding checks need the network response's URL and filter type.
  Object.defineProperties(response, {
    url: { value: source.url },
    type: { value: source.type },
  });
  const cacheHeaders = new Headers(source.headers);
  cacheHeaders.delete(MODEL_CACHE_RECORD_HEADER);
  if (recordId) cacheHeaders.set(MODEL_CACHE_RECORD_HEADER, recordId);
  return {
    response,
    cacheResponse: new Response(cacheStream, { ...init, headers: cacheHeaders }),
    detachCache,
    get cancelled() {
      return cancelled;
    },
    get decodedBytes() {
      return decodedBytes;
    },
  };
}

const MODEL_CACHE_RECORD_HEADER = 'x-llamadart-cache-record';
const MODEL_CACHE_SIZE_HEADER = 'x-llamadart-decoded-size';

async function modelCacheKeyDigest(cacheKey: string): Promise<string> {
  const key = new URL(cacheKey);
  key.hash = '';
  const hash = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(key.href));
  return Array.from(new Uint8Array(hash), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function modelCacheMetadataUrl(cacheKey: string): Promise<string> {
  return 'https://llamadart-cache.invalid/decoded-size/' + await modelCacheKeyDigest(cacheKey);
}

function modelCacheRecord(response: Response | undefined): string | null {
  const record = response?.headers.get(MODEL_CACHE_RECORD_HEADER) ?? '';
  return /^[0-9a-f-]{36}$/.test(record) ? record : null;
}

export async function readCachedModelSize(cache: Cache, cacheKey: string, response: Response): Promise<number | null> {
  const record = modelCacheRecord(response);
  if (!record) return null;
  try {
    const metadata = await cache.match((await modelCacheMetadataUrl(cacheKey)) + '?record=' + record);
    const bytes = parseByteLength(metadata?.headers.get(MODEL_CACHE_SIZE_HEADER) ?? null);
    return bytes !== null && bytes > 0 ? bytes : null;
  } catch (_) {
    return null;
  }
}

export async function pruneCachedModelSizes(cache: Cache, cacheKey: string): Promise<void> {
  try {
    const url = await modelCacheMetadataUrl(cacheKey);
    const entries = await cache.keys(url, { ignoreSearch: true });
    if (entries.length === 0) return;
    // Metadata is written after its body commits. Only prune records listed
    // before checking which body is current, so a concurrent store is safe.
    const cached = await cache.match(cacheKey);
    const record = modelCacheRecord(cached);
    // A clone-backed body may wait for its other branch on cancellation.
    void cached?.body?.cancel().catch(() => {});
    const keep = url + '?record=' + record;
    await Promise.all(entries.filter((entry) => entry.url !== keep).map((entry) => cache.delete(entry)));
  } catch (_) {
    // A metadata cleanup failure does not affect the model body.
  }
}

export function startModelCacheStore(cache: Cache, cacheKey: string, response: Response) {
  const record = globalThis.crypto?.randomUUID?.();
  const tee = teeResponseForCache(response, DEFAULT_CACHE_TEE_LAG_BYTES, record);
  const stored = cache.put(cacheKey, tee.cacheResponse).then(async () => {
    if (!record || !Number.isSafeInteger(tee.decodedBytes) || tee.decodedBytes <= 0) return;
    try {
      const url = (await modelCacheMetadataUrl(cacheKey)) + '?record=' + record;
      await cache.put(url, new Response(null, {
        headers: { [MODEL_CACHE_SIZE_HEADER]: String(tee.decodedBytes) },
      }));
      await pruneCachedModelSizes(cache, cacheKey);
    } catch (_) {
      // The model remains usable as a legacy cache entry.
    }
  }, (error: unknown) => {
    tee.detachCache(error);
    throw error;
  });
  return { tee, stored };
}

// WebKit holds a whole Cache Storage body in the page's process, several times
// over while it commits and again when it is matched. A model is stored there
// as fixed-size part entries under a manifest that is written last, so only
// one part is in memory at a time and an interrupted store is never a hit.
export const MODEL_CACHE_PART_BYTES = 16 * 1024 * 1024;
const MODEL_CACHE_PART_COUNT_HEADER = 'x-llamadart-part-count';
const MODEL_CACHE_PART_SIZE_HEADER = 'x-llamadart-part-size';
const MODEL_CACHE_INTERNAL_HEADERS = [
  MODEL_CACHE_RECORD_HEADER,
  MODEL_CACHE_SIZE_HEADER,
  MODEL_CACHE_PART_COUNT_HEADER,
  MODEL_CACHE_PART_SIZE_HEADER,
];

const modelCacheManifestUrl = (digest: string) => 'https://llamadart-cache.invalid/model-parts/' + digest;
const modelCachePartsUrl = (digest: string) => 'https://llamadart-cache.invalid/model-part/' + digest;
const modelCachePartUrl = (digest: string, record: string, index: number) =>
  modelCachePartsUrl(digest) + '?record=' + record + '&part=' + index;

export class ModelCacheEntryError extends Error {
  declare readonly retryWithoutCache: boolean;

  constructor(message: string, retryWithoutCache: boolean) {
    super(message);
    this.name = 'ModelCacheEntryError';
    this.retryWithoutCache = retryWithoutCache;
  }
}

interface ModelCacheManifest {
  record: string;
  decodedBytes: number;
  partCount: number;
  partBytes: number;
  headers: Headers;
}

async function readModelCacheManifest(cache: Cache, digest: string): Promise<ModelCacheManifest | null> {
  const manifest = await cache.match(modelCacheManifestUrl(digest));
  const record = modelCacheRecord(manifest);
  if (!manifest || !record) return null;
  const decodedBytes = parseByteLength(manifest.headers.get(MODEL_CACHE_SIZE_HEADER));
  const partCount = parseByteLength(manifest.headers.get(MODEL_CACHE_PART_COUNT_HEADER));
  const partBytes = parseByteLength(manifest.headers.get(MODEL_CACHE_PART_SIZE_HEADER));
  if (!decodedBytes || !partCount || !partBytes || partCount !== Math.ceil(decodedBytes / partBytes)) {
    return null;
  }
  return { record, decodedBytes, partCount, partBytes, headers: manifest.headers };
}

async function deleteModelCachePartEntries(
  cache: Cache,
  digest: string,
  shouldDelete: (record: string | null) => boolean,
): Promise<void> {
  const entries = await cache.keys(modelCachePartsUrl(digest), { ignoreSearch: true });
  const stale = entries.filter((entry) => shouldDelete(new URL(entry.url).searchParams.get('record')));
  await Promise.all(stale.map((entry) => cache.delete(entry)));
}

// null means a writer holds the lock and the caller should skip eviction.
export async function deleteModelCacheParts(cache: Cache, cacheKey: string, lockScope: string): Promise<boolean | null> {
  if (!getModelCacheLocks()) return false;
  const digest = await modelCacheKeyDigest(cacheKey);
  let removed: boolean | null = null;
  await withModelCacheLock(lockScope, digest, async (lock) => {
    if (!lock) return;
    removed = await cache.delete(modelCacheManifestUrl(digest));
    await deleteModelCachePartEntries(cache, digest, () => true);
  });
  return removed;
}

export interface CachedModelParts {
  response: Response;
  decodedBytes: number;
}

// The part-stored model for a key as one response, or null when there is none.
// A part that is missing or has the wrong length drops the entry and errors
// the body with a ModelCacheEntryError.
export async function openModelCacheParts(
  cache: Cache,
  cacheKey: string,
  lockScope: string,
): Promise<CachedModelParts | null> {
  const digest = await modelCacheKeyDigest(cacheKey);
  const manifest = await readModelCacheManifest(cache, digest);
  if (!manifest) return null;
  const { record, decodedBytes, partCount, partBytes } = manifest;
  let index = 0;
  const body = new ReadableStream<Uint8Array>(
    {
      async pull(controller) {
        if (index === partCount) {
          controller.close();
          return;
        }
        const expected = index === partCount - 1 ? decodedBytes - partBytes * index : partBytes;
        let bytes: Uint8Array | null = null;
        try {
          const part = await cache.match(modelCachePartUrl(digest, record, index));
          // A stream would keep each part alive on WebKit after it is read.
          bytes = part ? new Uint8Array(await part.arrayBuffer()) : null;
        } catch (_) {
          bytes = null;
        }
        if (!bytes || bytes.byteLength !== expected) {
          let retryWithoutCache = true;
          try {
            await withModelCacheLock(lockScope, digest, async (lock) => {
              if (!lock) return;
              if ((await readModelCacheManifest(cache, digest))?.record === record) {
                await cache.delete(modelCacheManifestUrl(digest));
                await deleteModelCachePartEntries(cache, digest, (partRecord) => partRecord === record);
              }
              retryWithoutCache = false;
            });
          } catch (_) {
            // An uncached retry can proceed even if the damaged entry could not be removed.
          }
          controller.error(new ModelCacheEntryError(
            'Cached model part ' + index + ' is missing or damaged.',
            retryWithoutCache,
          ));
          return;
        }
        index += 1;
        controller.enqueue(bytes);
      },
    },
    { highWaterMark: 0 },
  );
  const headers = new Headers(manifest.headers);
  for (const name of MODEL_CACHE_INTERNAL_HEADERS) headers.delete(name);
  // The parts hold decoded bytes.
  headers.delete('content-encoding');
  headers.set('content-length', String(decodedBytes));
  return { response: new Response(body, { status: 200, headers }), decodedBytes };
}

export async function readCachedModelPartsSize(cache: Cache, cacheKey: string): Promise<number | null> {
  return (await readModelCacheManifest(cache, await modelCacheKeyDigest(cacheKey)))?.decodedBytes ?? null;
}

export interface ModelCachePartStore {
  // The copy the caller reads; it keeps the source's status, headers and url.
  response: Response;
  stored: Promise<void>;
  readonly cancelled: boolean;
}

type LockManagerLike = {
  request(name: string, options: { ifAvailable: boolean }, callback: (lock: unknown) => unknown): Promise<unknown>;
};

function getModelCacheLocks(): LockManagerLike | null {
  const locks = (globalThis.navigator as { locks?: LockManagerLike } | undefined)?.locks;
  return typeof locks?.request === 'function' ? locks : null;
}

async function withModelCacheLock(
  lockScope: string,
  digest: string,
  callback: (lock: unknown) => unknown,
): Promise<void> {
  const locks = getModelCacheLocks();
  if (!locks) {
    throw new Error('Web Locks are unavailable, so the model cannot be stored in parts.');
  }
  await locks.request('llamadart-model-cache:' + lockScope + ':' + digest, { ifAvailable: true }, callback);
}

/**
 * Stores a response body as part entries while the caller reads it.
 *
 * Each full part is committed before the next chunk is delivered, so the store
 * holds one part and slows the transfer instead of buffering it. Unlike a tee,
 * the caller's chunks are not shared with the cache and may be released. A
 * store failure leaves the transfer intact and rejects `stored`.
 */
export function startModelCachePartStore(
  cache: Cache,
  cacheKey: string,
  source: Response,
  lockScope: string,
  partBytes: number = MODEL_CACHE_PART_BYTES,
): ModelCachePartStore {
  if (source.status === 206) {
    return {
      response: source,
      stored: Promise.reject(new TypeError('A partial response cannot be stored in the model cache.')),
      cancelled: false,
    };
  }
  const reader = source.body!.getReader();
  let cancelled = false;
  let storing = true;
  let record = '';
  let digest = '';
  let part: Uint8Array | null = null;
  let filled = 0;
  let partCount = 0;
  let decodedBytes = 0;
  let ready: Promise<void> | null = null;
  let pendingPut: Promise<void> | null = null;
  let releaseLock: (() => void) | null = null;
  let resolveStored!: () => void;
  let rejectStored!: (error: unknown) => void;
  const stored = new Promise<void>((resolve, reject) => {
    resolveStored = resolve;
    rejectStored = reject;
  });

  const pruneUncommitted = async () => {
    const current = await readModelCacheManifest(cache, digest);
    await deleteModelCachePartEntries(cache, digest, (partRecord) => !current || partRecord !== current.record);
  };

  // Acquire the lock only once the consumer reads, so a failed staging setup
  // cannot leave a lock held by a response that never gets consumed.
  const prepare = async () => {
    if (typeof globalThis.crypto?.randomUUID !== 'function') {
      throw new Error('A UUID generator is unavailable, so the model cannot be stored in parts.');
    }
    record = globalThis.crypto.randomUUID();
    digest = await modelCacheKeyDigest(cacheKey);
    const held = await new Promise<boolean>((resolve, reject) => {
      withModelCacheLock(lockScope, digest, (lock) => {
        if (!lock) {
          resolve(false);
          return undefined;
        }
        return new Promise<void>((release) => {
          releaseLock = release;
          resolve(true);
        });
      }).catch(reject);
    });
    if (!held) {
      throw new Error('Another context is storing this model.');
    }
    await pruneUncommitted();
  };

  const unlock = () => {
    const release = releaseLock;
    releaseLock = null;
    release?.();
  };
  const abandon = (error: unknown) => {
    if (!storing) return;
    storing = false;
    part = null;
    rejectStored(error);
    void (async () => {
      // A pending put can still commit after the caller cancels.
      await Promise.allSettled([ready, pendingPut]);
      if (releaseLock) {
        await deleteModelCachePartEntries(cache, digest, (partRecord) => partRecord === record);
      }
    })().catch(() => {}).finally(unlock);
  };
  const put = async (url: string, response: Response) => {
    pendingPut = cache.put(url, response);
    try {
      await pendingPut;
    } finally {
      pendingPut = null;
    }
  };
  const flush = async () => {
    const body = filled === part!.byteLength ? part! : part!.subarray(0, filled);
    await put(modelCachePartUrl(digest, record, partCount), new Response(body as BodyInit));
    partCount += 1;
    filled = 0;
  };
  const append = async (chunk: Uint8Array) => {
    part ??= new Uint8Array(partBytes);
    let offset = 0;
    while (storing && offset < chunk.byteLength) {
      const length = Math.min(part.byteLength - filled, chunk.byteLength - offset);
      part.set(chunk.subarray(offset, offset + length), filled);
      filled += length;
      offset += length;
      if (filled === part.byteLength) await flush();
    }
  };
  const commit = async () => {
    if (filled > 0) await flush();
    if (!Number.isSafeInteger(decodedBytes) || decodedBytes <= 0) {
      throw new Error('The model response was empty.');
    }
    const expectedBytes = captureResumeIdentity(source, source.url)?.total;
    if (expectedBytes != null && decodedBytes !== expectedBytes) {
      throw createModelTransferLengthError('Model response did not match its declared decoded size.');
    }
    const headers = new Headers(source.headers);
    headers.set(MODEL_CACHE_RECORD_HEADER, record);
    headers.set(MODEL_CACHE_SIZE_HEADER, String(decodedBytes));
    headers.set(MODEL_CACHE_PART_COUNT_HEADER, String(partCount));
    headers.set(MODEL_CACHE_PART_SIZE_HEADER, String(partBytes));
    await put(modelCacheManifestUrl(digest), new Response(null, { headers }));
    storing = false;
    part = null;
    try {
      await deleteModelCachePartEntries(cache, digest, (partRecord) => partRecord !== record);
      // A whole-body entry from an earlier version is now a stale duplicate.
      if (await cache.delete(cacheKey)) await pruneCachedModelSizes(cache, cacheKey);
    } catch (_) {
      // Leftovers are pruned by the next store.
    }
    unlock();
    resolveStored();
  };

  const bodyStream = new ReadableStream<Uint8Array>(
    {
      async pull(controller) {
        let result: ReadableStreamReadResult<Uint8Array>;
        try {
          result = await reader.read();
        } catch (error) {
          abandon(error);
          throw error;
        }
        if (cancelled) return;
        if (result.done) {
          controller.close();
          if (storing) {
            try {
              await (ready ??= prepare());
              if (storing) await commit();
            } catch (error) {
              abandon(error);
            }
          }
          return;
        }
        decodedBytes += result.value.byteLength;
        if (storing) {
          try {
            await (ready ??= prepare());
            if (storing) await append(result.value);
          } catch (error) {
            abandon(error);
          }
        }
        if (cancelled) return;
        controller.enqueue(result.value);
      },
      cancel(reason) {
        cancelled = true;
        abandon(reason ?? new Error('The response was cancelled before the cache copy completed.'));
        return reader.cancel(reason);
      },
    },
    { highWaterMark: 0 },
  );

  const response = new Response(bodyStream, {
    status: source.status,
    statusText: source.statusText,
    headers: source.headers,
  });
  Object.defineProperties(response, {
    url: { value: source.url },
    type: { value: source.type },
  });
  return {
    response,
    stored,
    get cancelled() {
      return cancelled;
    },
  };
}

// Emscripten's WASMFS FS.analyzePath() reads the whole entry to fill
// `object.contents`: it throws ErrnoError ("FS error") for an existing directory
// and copies an existing file into JS memory. Probe paths with mkdir/unlink and
// a directory listing instead.
export function ensureFsDirectory(fs: EmscriptenFs, dirPath: string): void {
  const slash = dirPath.lastIndexOf('/');
  const parent = slash > 0 ? dirPath.slice(0, slash) : '/';
  const name = dirPath.slice(slash + 1);
  const exists = () => {
    try {
      const entries = fs.readdir(parent);
      return Array.isArray(entries) && entries.includes(name);
    } catch (_) {
      return false;
    }
  };

  if (exists()) {
    return;
  }
  try {
    fs.mkdir(dirPath);
  } catch (error) {
    if (!exists()) {
      throw error;
    }
  }
}

export function unlinkFsFile(fs: EmscriptenFs, filePath: string): boolean {
  try {
    fs.unlink(filePath);
    return true;
  } catch (_) {
    return false;
  }
}

// WASMFS grows its vector geometrically; crossing 1 GiB can request a
// near-2 GiB allocation. Reserve a bounded size hint, then truncate the
// logical file while retaining its capacity. Small, visibly encoded,
// resumed and unknown-length responses retain the incremental writer.
function parseIdentityLength(response: Response, startOffset: number): { length: number; encoding: string | undefined } {
  if (startOffset !== 0 || response.status !== 200 || response.headers.has('content-range')) {
    return { length: 0, encoding: undefined };
  }
  const encoding = response.headers.get('content-encoding')?.trim().toLowerCase();
  if (encoding && encoding !== 'identity') return { length: 0, encoding };
  const header = response.headers.get('content-length')?.trim() || '';
  if (!/^[0-9]+$/.test(header)) return { length: 0, encoding };
  const length = Number(header);
  if (!Number.isSafeInteger(length) || length < 1073741824) {
    return { length: 0, encoding };
  }
  return { length, encoding };
}

function reservableResponseLength(response: Response, startOffset: number, bigInt: boolean): number {
  if (bigInt) return 0;
  const { length } = parseIdentityLength(response, startOffset);
  return length <= 2147483647 ? length : 0;
}

function exactExpectedTransferLength(response: Response, startOffset: number): number {
  const { length, encoding } = parseIdentityLength(response, startOffset);
  // CORS can hide Content-Encoding while exposing compressed Content-Length.
  // Constructed/cache responses can also lose the original response type.
  return length > 0 && (response.type === 'basic' || encoding === 'identity') ? length : 0;
}

export async function writeResponseToFsFileWithProgress(
  response: Response,
  fs: EmscriptenFs,
  filePath: string,
  progressCallback?: ProgressCallback | null,
  writeOptions: WriteResponseOptions = {},
): Promise<number> {
  const total = parsePositiveInteger(writeOptions.totalBytes)
    || inferResponseTotalBytes(response, 0);
  const useBigIntPosition = writeOptions.useBigIntPosition === true;
  const directWriter = writeOptions.directWriter || null;
  const releaseConsumedChunks = directWriter != null && writeOptions.releaseConsumedChunks === true;
  const startOffset = parsePositiveInteger(writeOptions.startOffset);
  const heapFileBytes = writeOptions.heapFileBytes ?? null;
  const preservePartialOnError = writeOptions.preservePartialOnError === true;
  const appendMode = startOffset > 0;
  const chunkTimeoutMs = parsePositiveInteger(writeOptions.chunkTimeoutMs);
  const signal = writeOptions.signal || null;
  const abortMessage = writeOptions.abortMessage || 'Model transfer was cancelled.';

  throwIfAborted(signal, abortMessage);

  if (!appendMode && !directWriter) {
    // best-effort replacement of stale temp files
    unlinkFsFile(fs, filePath);
  }

  if (!response.body || typeof response.body.getReader !== 'function') {
    const skipReservation = directWriter != null || writeOptions.skipMemoryReservation === true;
    const exactLength = skipReservation ? exactExpectedTransferLength(response, startOffset) : 0;
    try {
      throwIfAborted(signal, abortMessage);
      const bytes = new Uint8Array(await response.arrayBuffer());
      throwIfAborted(signal, abortMessage);
      if (exactLength > 0 && bytes.byteLength !== exactLength) {
        throw createModelTransferLengthError(
          bytes.byteLength < exactLength
            ? 'Model response ended before its declared byte length.'
            : 'Model response exceeded its declared byte length.',
        );
      }
      if (heapFileBytes !== null && startOffset + bytes.byteLength > heapFileBytes) {
        throw createModelTransferLengthError('Model response exceeded its declared byte length.');
      }
      let directLoaded = 0;
      if (directWriter) {
        try {
          writeDirectChunk(directWriter, bytes, startOffset, (written) => {
            directLoaded += written;
          });
        } catch (error) {
          if (error && typeof error === 'object') {
            (error as TransferError).llamadartLoadedBytes = startOffset + directLoaded;
            (error as TransferError).llamadartFilePath = filePath;
          }
          throw error;
        }
      } else if (appendMode) {
        const stream = fs.open(filePath, 'r+');
        try {
          throwIfAborted(signal, abortMessage);
          const position = useBigIntPosition ? BigInt(startOffset) : startOffset;
          fs.write(stream, bytes, 0, bytes.length, position);
        } finally {
          fs.close(stream);
        }
      } else {
        fs.writeFile(filePath, bytes);
      }

      const finalLoaded = startOffset + (directWriter ? directLoaded : bytes.byteLength);
      if (typeof progressCallback === 'function') {
        progressCallback({ loaded: finalLoaded, total: total || finalLoaded });
      }
      return finalLoaded;
    } finally {
      directWriter?.close?.();
    }
  }

  const skipReservation = directWriter != null || writeOptions.skipMemoryReservation === true;
  const reservedLength = skipReservation
    ? 0
    : reservableResponseLength(response, startOffset, useBigIntPosition);
  // CORS can hide Content-Encoding while exposing compressed Content-Length.
  // Constructed/cache responses can also lose the original response type.
  const encoding = response.headers.get('content-encoding')?.trim().toLowerCase();
  const exactLength = skipReservation
    ? exactExpectedTransferLength(response, startOffset)
    : (reservedLength > 0 && (response.type === 'basic' || encoding === 'identity') ? reservedLength : 0);
  const reader = response.body.getReader();
  const openMode = appendMode ? 'r+' : 'w';
  const stream = directWriter ? null : fs.open(filePath, openMode);
  let loaded = 0;
  let lastBucket = -1;
  let writePosition = startOffset;
  let writePositionBigInt: bigint | null = null;
  const emitProgressIfNeeded = () => {
    if (typeof progressCallback !== 'function') {
      return;
    }
    const effectiveLoaded = startOffset + loaded;
    const effectiveTotal = total || effectiveLoaded;
    const bucket = effectiveTotal > 0
      ? Math.floor((effectiveLoaded / effectiveTotal) * 100)
      : -1;
    if (bucket > lastBucket) {
      lastBucket = bucket;
      progressCallback({ loaded: effectiveLoaded, total: effectiveTotal });
    }
  };

  try {
    if (reservedLength > 0 && stream != null) {
      try {
        const written = fs.write(stream, new Uint8Array(1), 0, 1, reservedLength - 1);
        if (written !== 1) throw new Error('Filesystem reservation returned a short write.');
      } catch (_) {
        // A size hint must not reject a body that incremental writes can load.
      }
      // A reservation (including a partial one) must never pad the file body.
      const reset = fs.open(filePath, 'w');
      fs.close(reset);
      throwIfAborted(signal, abortMessage);
    }
    while (true) {
      throwIfAborted(signal, abortMessage);
      const { done, value } = await readStreamChunkWithTimeout(
        reader,
        chunkTimeoutMs,
        'response file read',
      );
      if (done) {
        throwIfAborted(signal, abortMessage);
        if (heapFileBytes !== null && startOffset + loaded > heapFileBytes) {
          throw createModelTransferLengthError('Model response exceeded its declared byte length.');
        }
        if (exactLength > 0 && loaded !== exactLength) {
          throw createModelTransferLengthError('Model response ended before its declared byte length.');
        }
        break;
      }

      if (!value || value.length === 0) {
        continue;
      }

      if (exactLength > 0 && value.length > exactLength - loaded) {
        throw createModelTransferLengthError('Model response exceeded its declared byte length.');
      }

      if (heapFileBytes !== null && startOffset + loaded + value.length > heapFileBytes) {
        if (response.status !== 206) {
          throw createModelTransferLengthError('Model response exceeded its declared byte length.');
        }
        // Count the invalid range tail so a later transport error still
        // triggers a restart, without writing outside the fixed heap block.
        loaded += value.length;
        emitProgressIfNeeded();
        if (releaseConsumedChunks) releaseChunkBuffer(value);
        continue;
      }

      if (directWriter) {
        throwIfAborted(signal, abortMessage);
        // Copy synchronously from the network chunk directly into the JS-backed
        // storage chunks, avoiding value.slice() and any wasm linear memory copy.
        writeDirectChunk(directWriter, value, writePosition, (written) => {
          writePosition += written;
          loaded += written;
          emitProgressIfNeeded();
        });
        if (releaseConsumedChunks) {
          releaseChunkBuffer(value);
        }
      } else {
        // Some browsers may reuse the same Uint8Array backing store across reads.
        // Clone each chunk before writing to avoid transient buffer aliasing.
        const chunk = value.slice ? value.slice() : new Uint8Array(value);
        throwIfAborted(signal, abortMessage);
        if (useBigIntPosition) {
          if (writePositionBigInt == null) {
            writePositionBigInt = BigInt(startOffset);
          }
          fs.write(stream, chunk, 0, chunk.length, writePositionBigInt);
          writePositionBigInt += BigInt(chunk.length);
        } else {
          fs.write(stream, chunk, 0, chunk.length, writePosition);
          writePosition += chunk.length;
        }
        loaded += chunk.length;
        emitProgressIfNeeded();
      }
    }
  } catch (error) {
    if (signal?.aborted && (error as Error | null)?.name !== 'AbortError') {
      error = createAbortError(abortMessage);
    }

    try {
      if (error && typeof error === 'object') {
        (error as TransferError).llamadartLoadedBytes = startOffset + loaded;
        (error as TransferError).llamadartFilePath = filePath;
      }
    } catch (_) {
      // ignore metadata attachment failures
    }

    if (directWriter) {
      try {
        directWriter.close?.();
      } catch (_) {
        // ignore close failures during abort/error
      }
    } else if (stream != null) {
      try {
        fs.close(stream);
      } catch (_) {
        // ignore close failures during abort/error
      }
    }

    try {
      await reader.cancel?.();
    } catch (_) {
      // ignore best-effort reader cancellation failures
    }

    if (!preservePartialOnError || startOffset + loaded === 0 || signal?.aborted
      || (error as Error | null)?.name === 'AbortError'
      || (error as Error | null)?.name === 'ModelTransferLengthError') {
      try {
        directWriter?.assertMountOwned?.();
        fs.unlink(filePath);
      } catch (_) {
        // ignore best-effort cleanup failures
      }
    }

    throw error;
  }

  if (directWriter) {
    directWriter.close?.();
  } else if (stream != null) {
    fs.close(stream);
  }

  const finalLoaded = startOffset + loaded;
  if (typeof progressCallback === 'function') {
    progressCallback({ loaded: finalLoaded, total: total || finalLoaded });
  }

  return finalLoaded;
}
