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

export interface WriteResponseOptions {
  totalBytes?: unknown;
  useBigIntPosition?: boolean;
  startOffset?: unknown;
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
  const startOffset = parsePositiveInteger(writeOptions.startOffset);
  const preservePartialOnError = writeOptions.preservePartialOnError === true;
  const appendMode = startOffset > 0;
  const chunkTimeoutMs = parsePositiveInteger(writeOptions.chunkTimeoutMs);
  const signal = writeOptions.signal || null;
  const abortMessage = writeOptions.abortMessage || 'Model transfer was cancelled.';

  throwIfAborted(signal, abortMessage);

  if (!appendMode) {
    // best-effort replacement of stale temp files
    unlinkFsFile(fs, filePath);
  }

  if (!response.body || typeof response.body.getReader !== 'function') {
    throwIfAborted(signal, abortMessage);
    const bytes = new Uint8Array(await response.arrayBuffer());
    throwIfAborted(signal, abortMessage);
    if (appendMode) {
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

  const reader = response.body.getReader();
  const openMode = appendMode ? 'r+' : 'w';
  const stream = fs.open(filePath, openMode);
  let loaded = 0;
  let lastBucket = -1;
  let writePosition = startOffset;
  let writePositionBigInt: bigint | null = null;

  try {
    while (true) {
      throwIfAborted(signal, abortMessage);
      const { done, value } = await readStreamChunkWithTimeout(
        reader,
        chunkTimeoutMs,
        'response file read',
      );
      if (done) {
        throwIfAborted(signal, abortMessage);
        break;
      }

      if (!value || value.length === 0) {
        continue;
      }

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

      if (typeof progressCallback === 'function') {
        const effectiveLoaded = startOffset + loaded;
        const effectiveTotal = total || effectiveLoaded;
        const bucket = effectiveTotal > 0
          ? Math.floor((effectiveLoaded / effectiveTotal) * 100)
          : -1;
        if (bucket > lastBucket) {
          lastBucket = bucket;
          progressCallback({ loaded: effectiveLoaded, total: effectiveTotal });
        }
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

    try {
      fs.close(stream);
    } catch (_) {
      // ignore close failures during abort/error
    }

    try {
      await reader.cancel?.();
    } catch (_) {
      // ignore best-effort reader cancellation failures
    }

    if (!preservePartialOnError || signal?.aborted || (error as Error | null)?.name === 'AbortError') {
      try {
        fs.unlink(filePath);
      } catch (_) {
        // ignore best-effort cleanup failures
      }
    }

    throw error;
  }

  fs.close(stream);

  const finalLoaded = startOffset + loaded;
  if (typeof progressCallback === 'function') {
    progressCallback({ loaded: finalLoaded, total: total || finalLoaded });
  }

  return finalLoaded;
}
