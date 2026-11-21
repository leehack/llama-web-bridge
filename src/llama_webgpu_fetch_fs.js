#if PTHREADS
addToLibrary({
  $llamadartFetchFsFiles: () => (llamadartFetchFsFiles.files ??= new Map()),

  $llamadartFetchFsEnsureRange: async (entry, offset, length) => {
    if (entry.size === undefined) {
      if (!entry.url.includes('://')) {
        try {
          entry.url = new URL(entry.url, self.location.origin).toString();
        } catch (_) {
          throw { status: 404 };
        }
      }
      const head = await fetch(entry.url, { method: 'HEAD', headers: { Range: 'bytes=0-' } });
      const contentLength = head.headers.get('Content-Length');
      const size = /^\d+$/.test(contentLength) ? Number(contentLength) : NaN;
      if (head.ok && head.headers.get('Accept-Ranges') === 'bytes' &&
          Number.isSafeInteger(size) && size > entry.chunkSize * 2) {
        entry.size = size;
      } else {
        const response = await fetch(entry.url);
        if (!response.ok) throw response;
        const bytes = new Uint8Array(await response.arrayBuffer());
        entry.size = bytes.length;
        entry.chunkSize = bytes.length || 1;
        entry.chunks[0] = bytes;
      }
    }

    if (length <= 0 || offset >= entry.size) return;
    const chunkSize = entry.chunkSize;
    const firstChunk = Math.floor(offset / chunkSize);
    const lastChunk = Math.floor((offset + Math.min(length, entry.size - offset) - 1) / chunkSize);
    const maxBatchChunks = Math.max(1, Math.floor((64 * 1024 * 1024) / chunkSize));
    for (let index = firstChunk; index <= lastChunk;) {
      if (entry.chunks[index]) {
        ++index;
        continue;
      }
      const firstMissing = index++;
      const batchEnd = Math.min(firstMissing + maxBatchChunks, lastChunk + 1);
      while (index < batchEnd && !entry.chunks[index]) ++index;

      const start = firstMissing * chunkSize;
      const end = Math.min(index * chunkSize, entry.size);
      const response = await fetch(entry.url, { headers: { Range: `bytes=${start}-${end - 1}` } });
      if (!response.ok) throw response;
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (response.status === 200 && bytes.length === entry.size) {
        entry.chunkSize = bytes.length || 1;
        entry.chunks = [bytes];
        return;
      }
      if (response.status !== 206 || bytes.length !== end - start) {
        throw new Error('Invalid model range response');
      }
      for (let chunk = firstMissing; chunk < index; ++chunk) {
        const chunkOffset = (chunk - firstMissing) * chunkSize;
        entry.chunks[chunk] = bytes.subarray(chunkOffset, chunkOffset + chunkSize);
      }
    }
  },

  _llamadart_fetch_fs_alloc_file__sig: 'vpipi',
  _llamadart_fetch_fs_alloc_file__deps: ['$llamadartFetchFsFiles', '$UTF8ToString', 'emscripten_proxy_finish'],
  _llamadart_fetch_fs_alloc_file: (ctx, fileId, url, chunkSize) => {
    try {
      llamadartFetchFsFiles().set(fileId >>> 0, {
        url: UTF8ToString(url),
        chunkSize: (chunkSize >>> 0) || (16 * 1024 * 1024),
        chunks: [],
      });
    } catch (_) {
      // An allocation failure leaves no entry for subsequent reads.
    } finally {
      _emscripten_proxy_finish(ctx);
    }
  },

  _llamadart_fetch_fs_free_file__sig: 'vpi',
  _llamadart_fetch_fs_free_file__deps: ['$llamadartFetchFsFiles', 'emscripten_proxy_finish'],
  _llamadart_fetch_fs_free_file: (ctx, fileId) => {
    try {
      const files = llamadartFetchFsFiles();
      const key = fileId >>> 0;
      const entry = files.get(key);
      if (entry) entry.chunks.length = 0;
      files.delete(key);
    } catch (_) {
    } finally {
      _emscripten_proxy_finish(ctx);
    }
  },

  _llamadart_fetch_fs_read__sig: 'vpipijp',
  _llamadart_fetch_fs_read__i53abi: true,
  _llamadart_fetch_fs_read__deps: ['$llamadartFetchFsFiles', '$llamadartFetchFsEnsureRange', 'emscripten_proxy_finish'],
  _llamadart_fetch_fs_read: async (ctx, fileId, buffer, length, offset, resultPtr) => {
    let result = 0;
    try {
      if (!Number.isSafeInteger(offset) || offset < 0) {
        result = -{{{ cDefs.EINVAL }}};
        return;
      }
      const requested = length >>> 0;
      if (requested === 0) return;
      const entry = llamadartFetchFsFiles().get(fileId >>> 0);
      if (!entry) {
        result = -{{{ cDefs.EBADF }}};
        return;
      }
      await llamadartFetchFsEnsureRange(entry, offset, requested);
      const end = offset + Math.min(requested, entry.size - offset);
      const chunkSize = entry.chunkSize;
      for (let start = offset; start < end;) {
        const index = Math.floor(start / chunkSize);
        const base = index * chunkSize;
        const chunkEnd = Math.min(end, base + chunkSize);
        HEAPU8.set(entry.chunks[index].subarray(start - base, chunkEnd - base), Number(buffer) + (start - offset));
        start = chunkEnd;
        result = start - offset;
      }
    } catch (error) {
      result = error?.status === 404 ? -{{{ cDefs.ENOENT }}} : -{{{ cDefs.EBADF }}};
    } finally {
      try {
        {{{ makeSetValue('resultPtr', 0, 'result', 'i32') }}};
      } finally {
        _emscripten_proxy_finish(ctx);
      }
    }
  },

  _llamadart_fetch_fs_get_size__sig: 'vpip',
  _llamadart_fetch_fs_get_size__deps: ['$llamadartFetchFsFiles', '$llamadartFetchFsEnsureRange', 'emscripten_proxy_finish'],
  _llamadart_fetch_fs_get_size: async (ctx, fileId, outSizePtr) => {
    let size = 0;
    try {
      const entry = llamadartFetchFsFiles().get(fileId >>> 0);
      if (!entry) return;
      await llamadartFetchFsEnsureRange(entry, 0, 0);
      size = entry.size;
    } catch (_) {
    } finally {
      try {
        {{{ makeSetValue('outSizePtr', 0, 'size', 'i64') }}};
      } finally {
        _emscripten_proxy_finish(ctx);
      }
    }
  },
});
#endif
