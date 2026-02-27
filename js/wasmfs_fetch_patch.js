/**
 * Fetch backend patch for large cross-origin model files.
 *
 * The stock Emscripten wasmfs fetch backend may fallback to downloading the
 * whole file when `Content-Length` is missing on HEAD, even when ranged reads
 * are supported and size metadata is available via other exposed headers.
 *
 * This override keeps chunked range mode when size can be inferred from
 * `X-Linked-Size` or `Content-Range`.
 */

addToLibrary({
  _wasmfs_create_fetch_backend_js__deps: [
    '$wasmFS$backends',
    '$wasmFS$JSMemoryRanges',
    '_wasmfs_fetch_get_file_url',
    '_wasmfs_fetch_get_chunk_size',
  ],

  _wasmfs_create_fetch_backend_js: async function(backend) {
    function parsePositiveInt(value) {
      if (!value) {
        return 0;
      }

      var parsed = parseInt(value, 10);
      return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
    }

    function parseTotalFromContentRange(contentRange) {
      if (!contentRange) {
        return 0;
      }

      var slash = contentRange.lastIndexOf('/');
      if (slash < 0 || slash + 1 >= contentRange.length) {
        return 0;
      }

      return parsePositiveInt(contentRange.substring(slash + 1));
    }

    function inferRemoteSize(headers) {
      if (!headers) {
        return 0;
      }

      var size = parsePositiveInt(headers.get('Content-Length'));
      if (size > 0) {
        return size;
      }

      size = parsePositiveInt(headers.get('X-Linked-Size'));
      if (size > 0) {
        return size;
      }

      size = parseTotalFromContentRange(headers.get('Content-Range'));
      if (size > 0) {
        return size;
      }

      return 0;
    }

    function supportsRangeRequests(headers) {
      if (!headers) {
        return false;
      }

      var acceptRanges = (headers.get('Accept-Ranges') || '').toLowerCase();
      if (acceptRanges === 'bytes') {
        return true;
      }

      // Some CDNs omit/strip Accept-Ranges but still expose a byte-range total.
      return parseTotalFromContentRange(headers.get('Content-Range')) > 0;
    }

    function asNumber(value) {
      return typeof value === 'bigint' ? Number(value) : value;
    }

    function asWasmSize(value) {
#if MEMORY64
      if (typeof value === 'bigint') {
        return value;
      }

      var numeric = Number(value);
      if (!Number.isFinite(numeric)) {
        return 0n;
      }

      return BigInt(Math.trunc(numeric));
#else
      return Number(value);
#endif
    }

    function setLastFetchError(tag, detail) {
      var stats = globalThis.__llamadartFetchBackendStats;
      if (stats && typeof stats === 'object') {
        stats.errors = (Number(stats.errors) || 0) + 1;
      }

      try {
        var text = String(detail || '').trim();
        if (text.length > 200) {
          text = text.slice(0, 200);
        }
        if (stats && typeof stats === 'object') {
          stats.lastError = text;
        }
        globalThis.__llamadartFetchBackendLastError = text.length > 0
          ? tag + ':' + text
          : tag;
      } catch (_) {
        if (stats && typeof stats === 'object') {
          stats.lastError = String(detail || '');
        }
        globalThis.__llamadartFetchBackendLastError = String(tag || 'fetch_error');
      }
    }

    function getFetchStats() {
      var stats = globalThis.__llamadartFetchBackendStats;
      if (!stats || typeof stats !== 'object') {
        stats = {
          reads: 0,
          getSize: 0,
          ranges: 0,
          wholeFileFallbacks: 0,
          errors: 0,
          lastError: '',
        };
        globalThis.__llamadartFetchBackendStats = stats;
      }
      return stats;
    }

    async function getFileRange(file, offset, len) {
      var stats = getFetchStats();
      stats.ranges = (Number(stats.ranges) || 0) + 1;

      var url = '';
      var fileUrl_p = __wasmfs_fetch_get_file_url(file);
      var fileUrl = UTF8ToString(fileUrl_p);
      var isAbs = fileUrl.indexOf('://') !== -1;
      if (isAbs) {
        url = fileUrl;
      } else {
        try {
          var u = new URL(fileUrl, self.location.origin);
          url = u.toString();
        } catch (_e) {
          throw { status: 404 };
        }
      }

      var chunkSize = __wasmfs_fetch_get_chunk_size(file);
      offset ??= 0;
      len ??= chunkSize;

      if (!(file in wasmFS$JSMemoryRanges)) {
        var fileInfo = await fetch(url, {
          method: 'HEAD',
          headers: { 'Range': 'bytes=0-' },
          cache: 'no-store',
        });

        var size = fileInfo.ok ? inferRemoteSize(fileInfo.headers) : 0;
        var ranged = fileInfo.ok ? supportsRangeRequests(fileInfo.headers) : false;

        if (!fileInfo.ok || size <= 0 || !ranged) {
          // Some hosts either block/strip useful HEAD headers or omit range
          // capability hints there. Probe with a tiny ranged GET and infer from
          // 206 + Content-Range.
          try {
            var probe = await fetch(url, {
              headers: { 'Range': 'bytes=0-0' },
              cache: 'no-store',
            });
            if (probe.ok || probe.status === 206) {
              var probeSize = inferRemoteSize(probe.headers);
              if (probeSize > 0) {
                size = probeSize;
              }

              ranged =
                ranged ||
                probe.status === 206 ||
                supportsRangeRequests(probe.headers);
            }
          } catch (_) {
            // Keep best-effort values.
          }
        }

        var canFallbackWholeFile = size > 0 && size <= 256 * 1024 * 1024;

        if (ranged && size > chunkSize * 2) {
          wasmFS$JSMemoryRanges[file] = {
            size,
            chunks: [],
            chunkSize: chunkSize,
          };
          len = Math.min(len, size - offset);
        } else if (canFallbackWholeFile) {
          // Fallback: download once and serve from in-memory chunks.
          stats.wholeFileFallbacks = (Number(stats.wholeFileFallbacks) || 0) + 1;
          var wholeFileReq = await fetch(url, { cache: 'no-store' });
          if (!wholeFileReq.ok) {
            throw wholeFileReq;
          }

          var wholeFileData = new Uint8Array(await wholeFileReq.arrayBuffer());
          wasmFS$JSMemoryRanges[file] = {
            size: wholeFileData.byteLength,
            chunks: [wholeFileData],
            chunkSize: wholeFileData.byteLength,
          };
          return;
        } else {
          // Refuse whole-file fallback when size is unknown or too large,
          // because it can trigger giant ArrayBuffer allocations.
          setLastFetchError('whole_file_fallback_refused', 'size=' + size);
          throw { status: 413 };
        }
      }

      var firstChunk = (offset / chunkSize) | 0;
      var lastChunk = ((offset + len - 1) / chunkSize) | 0;
      var allPresent = true;
      var i;

      for (i = firstChunk; i <= lastChunk; i++) {
        if (!wasmFS$JSMemoryRanges[file].chunks[i]) {
          allPresent = false;
          break;
        }
      }

      if (allPresent) {
        return;
      }

      var start = firstChunk * chunkSize;
      var end = (lastChunk + 1) * chunkSize;
      var response = await fetch(url, {
        headers: { 'Range': `bytes=${start}-${end - 1}` },
        cache: 'no-store',
      });
      if (!response.ok) {
        throw response;
      }

#if MIN_FIREFOX_VERSION < 128 || MIN_CHROME_VERSION < 132 || MIN_SAFARI_VERSION < 180000 || MIN_NODE_VERSION < 220300
      var bytes = new Uint8Array(await response['arrayBuffer']());
#else
      var bytes = await response['bytes']();
#endif

      for (i = firstChunk; i <= lastChunk; i++) {
        wasmFS$JSMemoryRanges[file].chunks[i] =
          bytes.slice(i * chunkSize - start, (i + 1) * chunkSize - start);
      }
    }

    wasmFS$backends[backend] = {
      allocFile: async (_file) => { /* nop */ },

      freeFile: async (file) => {
        wasmFS$JSMemoryRanges[file] = undefined;
      },

      write: async (_file, _buffer, _length, _offset) => {
        // Fetch backend is read-only.
        return asWasmSize(-{{{ cDefs.EBADF }}});
      },

      read: async (file, buffer, length, offset) => {
        try {
        var stats = getFetchStats();
        stats.reads = (Number(stats.reads) || 0) + 1;

        var offsetNum = asNumber(offset || 0);
        var lengthNum = asNumber(length || 0);
        var bufferNum = asNumber(buffer || 0);

        if (!Number.isFinite(offsetNum) || !Number.isFinite(lengthNum) || !Number.isFinite(bufferNum)) {
          return asWasmSize(-{{{ cDefs.EBADF }}});
        }

        if (offsetNum < 0 || lengthNum <= 0) {
          return asWasmSize(0);
        }

        try {
          await getFileRange(file, offsetNum, lengthNum);
        } catch (failedResponse) {
          setLastFetchError('range_fetch_failed', failedResponse?.status || failedResponse);
          return asWasmSize(
            failedResponse.status === 404
              ? -{{{ cDefs.ENOENT }}}
              : -{{{ cDefs.EBADF }}},
          );
        }

        var fileInfo = wasmFS$JSMemoryRanges[file];
        lengthNum = Math.min(lengthNum, fileInfo.size - offsetNum);
        if (lengthNum <= 0) {
          return asWasmSize(0);
        }

        var chunks = fileInfo.chunks;
        var chunkSize = fileInfo.chunkSize;
        var firstChunk = (offsetNum / chunkSize) | 0;
        var lastChunk = ((offsetNum + lengthNum - 1) / chunkSize) | 0;
        var readLength = 0;

        for (var i = firstChunk; i <= lastChunk; i++) {
          var chunk = chunks[i];
          if (!chunk) {
            setLastFetchError('missing_chunk', 'chunk=' + i);
            return asWasmSize(-{{{ cDefs.EBADF }}});
          }
          var start = Math.max(i * chunkSize, offsetNum);
          var chunkStart = i * chunkSize;
          var end = Math.min(chunkStart + chunkSize, offsetNum + lengthNum);
          HEAPU8.set(
            chunk.subarray(start - chunkStart, end - chunkStart),
            bufferNum + (start - offsetNum),
          );
          readLength = end - offsetNum;
        }

        return asWasmSize(readLength);
        } catch (error) {
          setLastFetchError('read_exception', error);
          return asWasmSize(-{{{ cDefs.EBADF }}});
        }
      },

      getSize: async (file) => {
        var stats = getFetchStats();
        stats.getSize = (Number(stats.getSize) || 0) + 1;
        try {
          await getFileRange(file, 0, 0);
        } catch (_failedResponse) {
          return asWasmSize(0);
        }
        return asWasmSize(wasmFS$JSMemoryRanges[file].size);
      },
    };
  },
});
