// Streaming model downloads with progress into the Emscripten filesystem.

import { createAbortError, throwIfAborted } from './abort.js';
import { parsePositiveInteger } from './parse.js';

export function hasReadableResponseStream(response) {
  return !!(
    response
    && response.body
    && typeof response.body.getReader === 'function'
  );
}

export function sumProgressValues(values) {
  let total = 0;
  for (const value of values || []) {
    const numeric = Number(value);
    if (Number.isFinite(numeric) && numeric > 0) {
      total += numeric;
    }
  }
  return total;
}

function parseTotalFromContentRangeHeader(contentRangeHeader) {
  if (typeof contentRangeHeader !== 'string' || contentRangeHeader.length === 0) {
    return 0;
  }

  const slash = contentRangeHeader.lastIndexOf('/');
  if (slash < 0 || slash + 1 >= contentRangeHeader.length) {
    return 0;
  }

  return parsePositiveInteger(contentRangeHeader.slice(slash + 1));
}

export function inferResponseTotalBytes(response, loadedFallback = 0) {
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

export function isRetryableStreamNetworkError(error) {
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

async function readStreamChunkWithTimeout(reader, timeoutMs, label = 'stream read') {
  const resolvedTimeout = Number(timeoutMs);
  if (!Number.isFinite(resolvedTimeout) || resolvedTimeout <= 0) {
    return reader.read();
  }

  let timeoutHandle = null;
  try {
    return await Promise.race([
      reader.read(),
      new Promise((_, reject) => {
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

export async function drainResponseWithProgress(response, progressCallback, options = {}) {
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

// Emscripten's WASMFS FS.analyzePath() reads the whole entry to fill
// `object.contents`: it throws ErrnoError ("FS error") for an existing directory
// and copies an existing file into JS memory. Probe paths with mkdir/unlink and
// a directory listing instead.
export function ensureFsDirectory(fs, dirPath) {
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

export function unlinkFsFile(fs, filePath) {
  try {
    fs.unlink(filePath);
    return true;
  } catch (_) {
    return false;
  }
}

export async function writeResponseToFsFileWithProgress(
  response,
  fs,
  filePath,
  progressCallback,
  writeOptions = {},
) {
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
  /** @type {bigint | null} */
  let writePositionBigInt = null;

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
    if (signal?.aborted && error?.name !== 'AbortError') {
      error = createAbortError(abortMessage);
    }

    try {
      if (error && typeof error === 'object') {
        error.llamadartLoadedBytes = startOffset + loaded;
        error.llamadartFilePath = filePath;
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

    if (!preservePartialOnError || signal?.aborted || error?.name === 'AbortError') {
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
