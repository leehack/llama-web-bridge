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
}

export interface DirectFileChunkWriter {
  writeChunk(chunk: Uint8Array, position: number): number | void;
  close?: () => void;
}

export interface WriteResponseOptions {
  totalBytes?: unknown;
  exactTransferLength?: unknown;
  useBigIntPosition?: boolean;
  skipMemoryReservation?: boolean;
  directWriter?: DirectFileChunkWriter | null;
  startOffset?: unknown;
  preservePartialOnError?: boolean;
  chunkTimeoutMs?: unknown;
  signal?: AbortSignal | null;
  abortMessage?: string;
}

export interface ParsedContentRange {
  start: number;
  end: number;
  total: number;
}

export interface ValidatedPartialRange extends ParsedContentRange {
  expectedTailLength: number;
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

function createModelTransferLengthError(message: string): Error {
  const error = new Error(message);
  error.name = 'ModelTransferLengthError';
  return error;
}

export function parseContentRangeHeader(
  contentRangeHeader: string | null | undefined,
): ParsedContentRange | null {
  if (typeof contentRangeHeader !== 'string') {
    return null;
  }

  const match = /^\s*bytes\s+(\d+)-(\d+)\/(\d+|\*)\s*$/i.exec(contentRangeHeader);
  if (!match) {
    return null;
  }

  const start = Number(match[1]);
  const end = Number(match[2]);
  const total = match[3] === '*' ? 0 : Number(match[3]);
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start) {
    return null;
  }
  if (match[3] !== '*' && (!Number.isSafeInteger(total) || total <= end)) {
    return null;
  }

  return { start, end, total };
}

export function validateResumedPartialResponse(
  response: Response,
  expectedOffset: number,
  expectedTotalBytes = 0,
): ValidatedPartialRange {
  const range = parseContentRangeHeader(response.headers?.get('content-range'));
  if (!range) {
    throw createModelTransferLengthError(
      'Resumed 206 response is missing a valid Content-Range header.',
    );
  }
  if (range.start !== expectedOffset) {
    throw createModelTransferLengthError(
      `Resumed Content-Range start ${range.start} does not match resume offset ${expectedOffset}.`,
    );
  }
  const knownTotal = parsePositiveInteger(expectedTotalBytes);
  if (knownTotal > 0 && range.total !== knownTotal) {
    throw createModelTransferLengthError(
      `Resumed Content-Range total ${range.total || '*'} does not match expected size ${knownTotal}.`,
    );
  }
  const resolvedTotal = range.total > 0 ? range.total : knownTotal;
  if (resolvedTotal <= 0) {
    throw createModelTransferLengthError(
      'Resumed 206 Content-Range must specify a finite total byte length.',
    );
  }
  if (range.end !== resolvedTotal - 1) {
    throw createModelTransferLengthError(
      `Resumed Content-Range end ${range.end} does not reach expected end ${resolvedTotal - 1}.`,
    );
  }

  const expectedTailLength = range.end - range.start + 1;
  const rawContentLength = response.headers?.get('content-length');
  if (typeof rawContentLength === 'string' && rawContentLength.trim().length > 0) {
    const trimmed = rawContentLength.trim();
    const declaredLength = /^[0-9]+$/.test(trimmed) ? Number(trimmed) : Number.NaN;
    if (!Number.isSafeInteger(declaredLength) || declaredLength !== expectedTailLength) {
      throw createModelTransferLengthError(
        `Resumed Content-Length ${trimmed} does not match Content-Range length ${expectedTailLength}.`,
      );
    }
  }

  return {
    ...range,
    total: resolvedTotal,
    expectedTailLength,
  };
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
): CacheTee {
  const reader = source.body!.getReader();
  let cacheController: ReadableStreamDefaultController<Uint8Array> | null = null;
  let wakeReader: (() => void) | null = null;
  let finished = false;
  let cancelled = false;

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
  // A constructed Response has an empty url; callers resolve redirects from it.
  Object.defineProperty(response, 'url', { value: source.url });
  return {
    response,
    cacheResponse: new Response(cacheStream, init),
    detachCache,
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
function parseIdentityLength(response: Response, startOffset: number): { length: number; exact: boolean } {
  if (startOffset !== 0 || response.status !== 200 || response.headers.has('content-range')) {
    return { length: 0, exact: false };
  }
  const encoding = response.headers.get('content-encoding')?.trim().toLowerCase();
  if (encoding && encoding !== 'identity') return { length: 0, exact: false };
  const header = response.headers.get('content-length')?.trim() || '';
  if (!/^[0-9]+$/.test(header)) return { length: 0, exact: false };
  const length = Number(header);
  if (!Number.isSafeInteger(length) || length < 1073741824) {
    return { length: 0, exact: false };
  }
  // CORS can hide Content-Encoding while exposing compressed Content-Length.
  // Constructed/cache responses can also lose the original response type.
  const exact = response.type === 'basic' || encoding === 'identity';
  return { length, exact };
}

function reservableResponseLength(response: Response, startOffset: number, bigInt: boolean): number {
  if (bigInt) return 0;
  const { length } = parseIdentityLength(response, startOffset);
  return length <= 2147483647 ? length : 0;
}

function exactExpectedTransferLength(response: Response, startOffset: number): number {
  const { length, exact } = parseIdentityLength(response, startOffset);
  return exact ? length : 0;
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
  const startOffset = parsePositiveInteger(writeOptions.startOffset);
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

  const explicitExactLength = parsePositiveInteger(writeOptions.exactTransferLength);

  if (!response.body || typeof response.body.getReader !== 'function') {
    throwIfAborted(signal, abortMessage);
    const bytes = new Uint8Array(await response.arrayBuffer());
    throwIfAborted(signal, abortMessage);
    if (explicitExactLength > 0 && bytes.byteLength !== explicitExactLength) {
      throw createModelTransferLengthError('Model response did not match its declared byte length.');
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
      } finally {
        directWriter.close?.();
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

    const finalLoaded = startOffset + bytes.byteLength;
    if (typeof progressCallback === 'function') {
      progressCallback({ loaded: finalLoaded, total: total || finalLoaded });
    }
    return finalLoaded;
  }

  const encoding = response.headers.get('content-encoding')?.trim().toLowerCase();
  const isIdentity = !encoding || encoding === 'identity';
  const reservedLength = (directWriter || writeOptions.skipMemoryReservation === true)
    ? 0
    : reservableResponseLength(response, startOffset, useBigIntPosition);
  const headerExactLength = exactExpectedTransferLength(response, startOffset);
  const exactLength = headerExactLength > 0
    ? headerExactLength
    : (isIdentity ? explicitExactLength : 0);
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

      if (directWriter) {
        throwIfAborted(signal, abortMessage);
        // Copy synchronously from the network chunk directly into the JS-backed
        // storage chunks, avoiding value.slice() and any wasm linear memory copy.
        writeDirectChunk(directWriter, value, writePosition, (written) => {
          writePosition += written;
          loaded += written;
          emitProgressIfNeeded();
        });
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
