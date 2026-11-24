// Chunked JS-backed WASMFS file storage for staging large models outside wasm linear memory.

addToLibrary({
  $llamadartJsModelFsState: {
    files: null,
    handles: null,
    nextHandle: 1,
    nextPreallocBytes: 0,
    lastOomBytes: 0,
    chunkSize: 67108864, // 64 MiB
  },

  $llamadartJsModelFsFiles__deps: ['$llamadartJsModelFsState'],
  $llamadartJsModelFsFiles: () => {
    if (!llamadartJsModelFsState.files) {
      llamadartJsModelFsState.files = new Map();
    }
    return llamadartJsModelFsState.files;
  },

  $llamadartJsModelFsHandles__deps: ['$llamadartJsModelFsState'],
  $llamadartJsModelFsHandles: () => {
    if (!llamadartJsModelFsState.handles) {
      llamadartJsModelFsState.handles = new Map();
    }
    return llamadartJsModelFsState.handles;
  },

  $llamadartJsModelFsDiscardChunks: (chunks, keep = 0) => {
    // Reads copy into WASM synchronously, and direct writes copy into these
    // private chunks. Heap mappings use a separate native block.
    for (let i = keep; i < chunks.length; i += 1) {
      const buffer = chunks[i]?.buffer;
      try {
        buffer?.transfer?.(0);
      } catch (_) {
        // Dropping the reference still lets GC reclaim the buffer.
      }
    }
    chunks.length = keep;
  },

  $llamadartJsModelFsWriteBytes__deps: [
    '$llamadartJsModelFsState',
    '$llamadartJsModelFsDiscardChunks',
  ],
  $llamadartJsModelFsWriteBytes: (file, copySlice, length, offset) => {
    const chunkSize = file.chunkSize;
    const end = offset + length;
    if (file.heapPtr) {
      if (end > file.heapCapacity) {
        return -{{{ cDefs.EFBIG }}};
      }
      // The block is not zeroed up front; zero only the gap a write skips, so
      // a sequential download never pays a second pass.
      if (offset > file.size) {
        HEAPU8.fill(0, file.heapPtr + file.size, file.heapPtr + offset);
      }
      copySlice(null, file.heapPtr + offset, 0, length);
      if (end > file.size) {
        file.size = end;
      }
      return length;
    }
    const firstChunk = Math.floor(offset / chunkSize);
    const lastChunk = Math.floor((end - 1) / chunkSize);

    try {
      for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
        const chunkBase = idx * chunkSize;
        const writeStart = Math.max(offset, chunkBase);
        const writeEnd = Math.min(end, chunkBase + chunkSize);
        const copyLen = writeEnd - writeStart;
        const dstOff = writeStart - chunkBase;
        const srcOffset = writeStart - offset;

        let chunk = file.chunks[idx];
        if (!chunk) {
          chunk = new Uint8Array(chunkSize);
          file.chunks[idx] = chunk;
        } else if (dstOff + copyLen > chunk.length) {
          // Only resize this single chunk if the file grew past its preallocated tail.
          const expanded = new Uint8Array(chunkSize);
          expanded.set(chunk);
          chunk = expanded;
          file.chunks[idx] = chunk;
        }

        copySlice(chunk, dstOff, srcOffset, copyLen);
      }
      if (end > file.size) {
        file.size = end;
      }
      return length;
    } catch (_) {
      file.oomError = true;
      llamadartJsModelFsDiscardChunks(file.chunks);
      file.size = 0;
      llamadartJsModelFsState.lastOomBytes = end;
      return -{{{ cDefs.ENOMEM }}};
    }
  },

  _llamadart_js_model_fs_set_next_prealloc_bytes__sig: 'vj',
  _llamadart_js_model_fs_set_next_prealloc_bytes__i53abi: true,
  _llamadart_js_model_fs_set_next_prealloc_bytes__deps: ['$llamadartJsModelFsState'],
  _llamadart_js_model_fs_set_next_prealloc_bytes: (bytes) => {
    llamadartJsModelFsState.nextPreallocBytes = bytes > 0 ? bytes : 0;
  },

  _llamadart_js_model_fs_alloc_file__sig: 'vpipp',
  _llamadart_js_model_fs_alloc_file__deps: [
    '$llamadartJsModelFsState',
    '$llamadartJsModelFsFiles',
    '$llamadartJsModelFsDiscardChunks',
  ],
  _llamadart_js_model_fs_alloc_file: (fileId, mountId, heapPtr, heapCapacity) => {
    const chunkSize = llamadartJsModelFsState.chunkSize;
    const preallocBytes = llamadartJsModelFsState.nextPreallocBytes;
    llamadartJsModelFsState.nextPreallocBytes = 0;

    if (Number(heapPtr) > 0) {
      // Heap-resident: the bytes live in one wasm heap block the native side owns.
      llamadartJsModelFsFiles().set(fileId, {
        mountId: Number(mountId) | 0,
        size: 0,
        chunkSize,
        chunks: [],
        oomError: false,
        heapPtr: Number(heapPtr),
        heapCapacity: Number(heapCapacity),
      });
      return;
    }

    const chunks = [];
    let oomError = false;
    if (preallocBytes > 0) {
      const numChunks = Math.ceil(preallocBytes / chunkSize);
      try {
        for (let i = 0; i < numChunks; i += 1) {
          const isLast = i === numChunks - 1;
          const len = isLast ? (preallocBytes - i * chunkSize) : chunkSize;
          chunks.push(new Uint8Array(len));
        }
      } catch (_) {
        llamadartJsModelFsDiscardChunks(chunks);
        oomError = true;
        llamadartJsModelFsState.lastOomBytes = preallocBytes;
      }
    }

    llamadartJsModelFsFiles().set(fileId, {
      mountId: Number(mountId) | 0,
      size: 0,
      chunkSize,
      chunks,
      oomError,
    });
  },

  _llamadart_js_model_fs_free_file__sig: 'vp',
  _llamadart_js_model_fs_free_file__deps: [
    '$llamadartJsModelFsFiles',
    '$llamadartJsModelFsHandles',
    '$llamadartJsModelFsDiscardChunks',
  ],
  _llamadart_js_model_fs_free_file: (fileId) => {
    const files = llamadartJsModelFsFiles();
    const file = files.get(fileId);
    if (file) {
      llamadartJsModelFsDiscardChunks(file.chunks);
      file.size = 0;
      files.delete(fileId);
    }
    const handles = llamadartJsModelFsHandles();
    for (const [handle, mappedFileId] of handles.entries()) {
      if (mappedFileId === fileId) {
        handles.delete(handle);
      }
    }
  },

  _llamadart_js_model_fs_bind_direct__sig: 'ip',
  _llamadart_js_model_fs_bind_direct__deps: [
    '$llamadartJsModelFsState',
    '$llamadartJsModelFsFiles',
    '$llamadartJsModelFsHandles',
  ],
  _llamadart_js_model_fs_bind_direct: (fileId) => {
    const file = llamadartJsModelFsFiles().get(fileId);
    if (!file) {
      return -{{{ cDefs.EBADF }}};
    }
    if (file.oomError) {
      return -{{{ cDefs.ENOMEM }}};
    }
    const handle = llamadartJsModelFsState.nextHandle;
    llamadartJsModelFsState.nextHandle += 1;
    llamadartJsModelFsHandles().set(handle, fileId);
    return handle;
  },

  _llamadart_js_model_fs_unbind_direct__sig: 'vi',
  _llamadart_js_model_fs_unbind_direct__deps: ['$llamadartJsModelFsHandles'],
  _llamadart_js_model_fs_unbind_direct: (handle) => {
    llamadartJsModelFsHandles().delete(handle);
  },

  $llamadartJsModelFsWriteDirect__deps: [
    '$llamadartJsModelFsFiles',
    '$llamadartJsModelFsHandles',
    '$llamadartJsModelFsWriteBytes',
  ],
  $llamadartJsModelFsWriteDirect: (handle, source, offset) => {
    const fileId = llamadartJsModelFsHandles().get(handle);
    const file = fileId !== undefined ? llamadartJsModelFsFiles().get(fileId) : undefined;
    if (!file) {
      return -{{{ cDefs.EBADF }}};
    }
    if (file.oomError) {
      return -{{{ cDefs.ENOMEM }}};
    }
    const typedArrayProto = typeof Uint8Array === 'function'
      ? Object.getPrototypeOf(Uint8Array.prototype)
      : null;
    const typedArrayNameGetter = typedArrayProto
      ? Object.getOwnPropertyDescriptor(typedArrayProto, Symbol.toStringTag)?.get
      : null;
    const isUint8Array = source instanceof Uint8Array
      || (ArrayBuffer.isView(source)
        && typeof typedArrayNameGetter === 'function'
        && typedArrayNameGetter.call(source) === 'Uint8Array');
    if (!isUint8Array || !Number.isSafeInteger(offset) || offset < 0) {
      return -{{{ cDefs.EINVAL }}};
    }
    const length = source.byteLength;
    if (length === 0) {
      return 0;
    }
    return llamadartJsModelFsWriteBytes(
      file,
      (chunk, dstOff, srcOffset, copyLen) => {
        (chunk || HEAPU8).set(source.subarray(srcOffset, srcOffset + copyLen), dstOff);
      },
      length,
      offset,
    );
  },

  _llamadart_js_model_fs_write__sig: 'ippij',
  _llamadart_js_model_fs_write__i53abi: true,
  _llamadart_js_model_fs_write__deps: [
    '$llamadartJsModelFsFiles',
    '$llamadartJsModelFsWriteBytes',
  ],
  _llamadart_js_model_fs_write: (fileId, buffer, length, offset) => {
    const file = llamadartJsModelFsFiles().get(fileId);
    if (!file) {
      return -{{{ cDefs.EBADF }}};
    }
    if (file.oomError) {
      return -{{{ cDefs.ENOMEM }}};
    }
    if (offset < 0) {
      return -{{{ cDefs.EINVAL }}};
    }
    length >>>= 0;
    if (length === 0) {
      return 0;
    }

    const srcBase = Number(buffer);
    return llamadartJsModelFsWriteBytes(
      file,
      (chunk, dstOff, srcOffset, copyLen) => {
        const start = srcBase + srcOffset;
        if (chunk) {
          chunk.set(HEAPU8.subarray(start, start + copyLen), dstOff);
        } else {
          HEAPU8.copyWithin(dstOff, start, start + copyLen);
        }
      },
      length,
      offset,
    );
  },

  _llamadart_js_model_fs_read__sig: 'ippij',
  _llamadart_js_model_fs_read__i53abi: true,
  _llamadart_js_model_fs_read__deps: ['$llamadartJsModelFsFiles'],
  _llamadart_js_model_fs_read: (fileId, buffer, length, offset) => {
    const file = llamadartJsModelFsFiles().get(fileId);
    if (!file) {
      return -{{{ cDefs.EBADF }}};
    }
    if (offset < 0) {
      return -{{{ cDefs.EINVAL }}};
    }
    length >>>= 0;
    if (length === 0 || offset >= file.size) {
      return 0;
    }

    const available = file.size - offset;
    const toRead = Math.min(length, available);
    const end = offset + toRead;
    if (file.heapPtr) {
      HEAPU8.copyWithin(Number(buffer), file.heapPtr + offset, file.heapPtr + end);
      return toRead;
    }
    const chunkSize = file.chunkSize;
    const firstChunk = Math.floor(offset / chunkSize);
    const lastChunk = Math.floor((end - 1) / chunkSize);
    const dstBase = Number(buffer);

    for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
      const chunkBase = idx * chunkSize;
      const readStart = Math.max(offset, chunkBase);
      const readEnd = Math.min(end, chunkBase + chunkSize);
      const copyLen = readEnd - readStart;
      const srcOff = readStart - chunkBase;
      const dstOff = dstBase + (readStart - offset);

      const chunk = file.chunks[idx];
      if (!chunk || srcOff >= chunk.length) {
        HEAPU8.fill(0, dstOff, dstOff + copyLen);
      } else if (srcOff + copyLen <= chunk.length) {
        HEAPU8.set(chunk.subarray(srcOff, srcOff + copyLen), dstOff);
      } else {
        const validLen = chunk.length - srcOff;
        HEAPU8.set(chunk.subarray(srcOff, chunk.length), dstOff);
        HEAPU8.fill(0, dstOff + validLen, dstOff + copyLen);
      }
    }

    return toRead;
  },

  _llamadart_js_model_fs_get_size__sig: 'vpp',
  _llamadart_js_model_fs_get_size__deps: ['$llamadartJsModelFsFiles'],
  _llamadart_js_model_fs_get_size: (fileId, outSizePtr) => {
    const file = llamadartJsModelFsFiles().get(fileId);
    const size = file ? file.size : 0;
    {{{ makeSetValue('outSizePtr', 0, 'size', 'i64') }}};
  },

  _llamadart_js_model_fs_set_size__sig: 'ipj',
  _llamadart_js_model_fs_set_size__i53abi: true,
  _llamadart_js_model_fs_set_size__deps: [
    '$llamadartJsModelFsFiles',
    '$llamadartJsModelFsDiscardChunks',
  ],
  _llamadart_js_model_fs_set_size: (fileId, size) => {
    const file = llamadartJsModelFsFiles().get(fileId);
    if (!file) {
      return -{{{ cDefs.EBADF }}};
    }
    if (size < 0) {
      return -{{{ cDefs.EINVAL }}};
    }
    if (file.heapPtr) {
      if (size > file.heapCapacity) {
        return -{{{ cDefs.EFBIG }}};
      }
      // Bytes past the old size are whatever the block held before.
      if (size > file.size) {
        HEAPU8.fill(0, file.heapPtr + file.size, file.heapPtr + size);
      }
      file.size = size;
      return 0;
    }
    if (size === 0) {
      if (file.size > 0) {
        llamadartJsModelFsDiscardChunks(file.chunks);
        file.size = 0;
      }
      return 0;
    }

    const chunkSize = file.chunkSize;
    if (size < file.size) {
      const keepChunks = Math.ceil(size / chunkSize);
      llamadartJsModelFsDiscardChunks(file.chunks, keepChunks);
      const lastIdx = keepChunks - 1;
      const tailLen = size - lastIdx * chunkSize;
      const lastChunk = file.chunks[lastIdx];
      if (lastChunk && tailLen < lastChunk.length) {
        lastChunk.fill(0, tailLen);
      }
    }
    file.size = size;
    return 0;
  },

  _llamadart_js_model_fs_release_mount__sig: 'vi',
  _llamadart_js_model_fs_release_mount__deps: [
    '$llamadartJsModelFsFiles',
    '$llamadartJsModelFsHandles',
    '$llamadartJsModelFsDiscardChunks',
  ],
  _llamadart_js_model_fs_release_mount: (mountId) => {
    const files = llamadartJsModelFsFiles();
    const handles = llamadartJsModelFsHandles();
    const targetMountId = Number(mountId) | 0;
    for (const [fileId, file] of files.entries()) {
      if ((file.mountId | 0) !== targetMountId) {
        continue;
      }
      llamadartJsModelFsDiscardChunks(file.chunks);
      file.size = 0;
      files.delete(fileId);
    }
    for (const [handle, fileId] of handles.entries()) {
      if (!files.has(fileId)) {
        handles.delete(handle);
      }
    }
  },

  _llamadart_js_model_fs_consume_oom_bytes__sig: 'd',
  _llamadart_js_model_fs_consume_oom_bytes__deps: ['$llamadartJsModelFsState'],
  _llamadart_js_model_fs_consume_oom_bytes: () => {
    const bytes = llamadartJsModelFsState.lastOomBytes;
    llamadartJsModelFsState.lastOomBytes = 0;
    return bytes;
  },

  _llamadart_js_model_fs_allocated_bytes__sig: 'd',
  _llamadart_js_model_fs_allocated_bytes__deps: ['$llamadartJsModelFsFiles'],
  _llamadart_js_model_fs_allocated_bytes: () => {
    let total = 0;
    for (const file of llamadartJsModelFsFiles().values()) {
      for (let i = 0; i < file.chunks.length; i += 1) {
        const chunk = file.chunks[i];
        if (chunk) {
          total += chunk.byteLength;
        }
      }
    }
    return total;
  },
});
