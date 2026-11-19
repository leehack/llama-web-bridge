import assert from 'node:assert/strict';
import test from 'node:test';

import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';
import { writeResponseToFsFileWithProgress } from '../../js/src/internal/download.ts';

function createStubFs() {
  const dirs = new Set(['/']);
  const files = new Map();
  return {
    dirs,
    files,
    writes: [],
    readdir(dir) {
      if (!dirs.has(dir)) {
        throw new Error(`No such directory: ${dir}`);
      }
      return ['.', '..'];
    },
    mkdir(dir) {
      dirs.add(dir);
    },
    unlink(filePath) {
      if (!files.has(filePath)) {
        throw new Error(`ENOENT: ${filePath}`);
      }
      files.delete(filePath);
    },
    open(filePath, mode) {
      if (mode === 'w') {
        files.set(filePath, 0);
      } else if (!files.has(filePath)) {
        throw new Error(`ENOENT: ${filePath}`);
      }
      return { filePath };
    },
    write(stream, _bytes, _offset, length, position) {
      this.writes.push({ filePath: stream.filePath, length, position: Number(position) });
      files.set(stream.filePath, Math.max(files.get(stream.filePath) || 0, Number(position) + length));
      return length;
    },
    close() {},
    writeFile(filePath, bytes) {
      files.set(filePath, bytes.length);
    },
    readFile() {
      return new Uint8Array();
    },
  };
}

test('jsBackedModelStaging mounts /jsmodels, preallocates, and skips memory reservation', async () => {
  const fs = createStubFs();
  const ccalls = [];
  const core = {
    FS: fs,
    ccall(name, _returnType, _argTypes, args = []) {
      ccalls.push([name, ...args]);
      if (name === 'llamadart_webgpu_mount_js_model_fs') {
        fs.dirs.add(args[0]);
        return 0;
      }
      if (name === 'llamadart_webgpu_get_context_size') {
        return 2048;
      }
      return 0;
    },
  };

  const bridge = new LlamaWebGpuBridge({
    disableWorker: true,
    allowSingleThreadCore: true,
    jsBackedModelStaging: true,
  });
  assert.equal(bridge._workerConfig().jsBackedModelStaging, true);
  assert.equal(new LlamaWebGpuBridge({ disableWorker: true })._workerConfig().jsBackedModelStaging, undefined);

  const runtime = bridge._runtime;
  runtime._core = core;
  runtime._coreVariant = 'wasm32';
  runtime._probeBackends = async () => false;
  runtime._ensureCore = async () => core;
  runtime._coreSupportsPthreads = () => false;
  runtime._syncThreadPoolSizeHintFromCore = () => {};
  runtime._resolveNativeLoadOptions = () => {};
  runtime._emitSuppressedWarmupWarningSummaryIfNeeded = () => {};
  runtime._getCachedModelResponse = async () => new Response(new Uint8Array([1, 2, 3, 4]), {
    status: 200,
    headers: { 'content-length': '4', 'content-encoding': 'identity' },
  });

  await runtime.loadModelFromUrl('https://models.test/demo.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });

  assert.deepEqual(ccalls[0], ['llamadart_webgpu_mount_js_model_fs', '/jsmodels']);
  assert.deepEqual(ccalls[1], ['llamadart_webgpu_js_model_fs_prealloc_next', 4]);
  assert.deepEqual(ccalls[2].slice(0, 3), ['llamadart_webgpu_load_model', '/jsmodels/demo.gguf', 2048]);
  assert.ok(runtime._runtimeNotes.includes('model_staging_js_backed'));
  assert.equal(fs.files.size, 0);
});

test('skipMemoryReservation skips the 1-byte seek write even for 1 GiB+ Content-Length', async () => {
  const fs = createStubFs();
  const gib = 1073741824;
  const response = new Response(new Uint8Array([7, 8, 9]), {
    status: 200,
    headers: { 'content-length': String(gib), 'content-encoding': 'identity' },
  });
  const written = await writeResponseToFsFileWithProgress(
    response,
    fs,
    '/jsmodels/big.gguf',
    null,
    { skipMemoryReservation: true },
  );
  assert.equal(written, 3);
  assert.deepEqual(fs.writes, [{ filePath: '/jsmodels/big.gguf', length: 3, position: 0 }]);
});
