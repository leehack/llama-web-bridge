import assert from 'node:assert/strict';
import test from 'node:test';

import { LlamaWebGpuBridge, ModelStagingError } from '../../js/src/llama_webgpu_bridge.js';
import { writeResponseToFsFileWithProgress } from '../../js/src/internal/download.ts';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';
import {
  createRealWorkerBridge,
  withStubWorkerEnvironment,
  workerDriver,
} from './bridge_operation_queue_fixtures.mjs';
import { readNativeCoreSource } from './native_core_source.mjs';

function createStubFs({ failWriteWith = null } = {}) {
  const dirs = new Set(['/']);
  const files = new Map();
  let nextStreamId = 1;
  return {
    dirs,
    files,
    writes: [],
    unmountedPaths: [],
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
      return { id: nextStreamId++, filePath };
    },
    write(stream, _bytes, _offset, length, position) {
      if (typeof failWriteWith === 'function') {
        const err = failWriteWith({ filePath: stream.filePath, length, position: Number(position) });
        if (err) {
          throw err;
        }
      }
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

function createStagingHarness({
  pthreads = false,
  jsBackedModelStaging = true,
  failWriteWith = null,
  onLoadModel = null,
} = {}) {
  const fs = createStubFs({ failWriteWith });
  const ccalls = [];
  let preallocNextBytes = 0;
  let oomBytes = 0;
  let lastErrorText = '';
  let runtime = null;

  const core = {
    FS: fs,
    UTF8ToString: () => lastErrorText,
    ccall(name, _returnType, _argTypes, args = []) {
      ccalls.push([name, ...args]);
      if (name === 'llamadart_webgpu_mount_js_model_fs') {
        fs.dirs.add(args[0]);
        return 0;
      }
      if (name === 'llamadart_webgpu_unmount_js_model_fs') {
        fs.unmountedPaths.push(args[0]);
        for (const p of [...fs.files.keys()]) {
          if (p.startsWith(`${args[0]}/`)) {
            fs.files.delete(p);
          }
        }
        fs.dirs.delete(args[0]);
        preallocNextBytes = 0;
        return 0;
      }
      if (name === 'llamadart_webgpu_js_model_fs_prealloc_next') {
        preallocNextBytes = Number(args[0]) || 0;
        return 0;
      }
      if (name === 'llamadart_webgpu_js_model_fs_consume_oom_bytes') {
        const value = oomBytes;
        oomBytes = 0;
        return value;
      }
      if (name === 'llamadart_webgpu_last_error') {
        return lastErrorText;
      }
      if (name === 'llamadart_webgpu_load_model' && typeof onLoadModel === 'function') {
        return onLoadModel({
          args,
          setLastError: (msg) => { lastErrorText = msg; },
          emitLog: (msg) => {
            runtime._lastCoreErrorText = msg;
            runtime._lastCoreErrorHint = msg;
          },
        });
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
    jsBackedModelStaging,
  });
  runtime = bridge._runtime;
  runtime._core = core;
  runtime._coreVariant = 'wasm32';
  runtime._probeBackends = async () => false;
  runtime._ensureCore = async () => {
    runtime._core = core;
    runtime._coreVariant = 'wasm32';
    return core;
  };
  runtime._coreSupportsPthreads = () => pthreads;
  runtime._syncThreadPoolSizeHintFromCore = () => {};
  runtime._resolveNativeLoadOptions = () => {};
  runtime._emitSuppressedWarmupWarningSummaryIfNeeded = () => {};
  runtime._getCachedModelResponse = async () => new Response(new Uint8Array([1, 2, 3, 4]), {
    status: 200,
    headers: { 'content-length': '4', 'content-encoding': 'identity' },
  });

  return {
    bridge,
    runtime,
    core,
    fs,
    ccalls,
    get preallocNextBytes() { return preallocNextBytes; },
    setOomBytes: (bytes) => { oomBytes = bytes; },
    setLastError: (msg) => { lastErrorText = msg; },
  };
}

test('jsBackedModelStaging defaults off and works opt-in for threaded and non-threaded cores', async () => {
  const defaultBridge = new LlamaWebGpuBridge({ disableWorker: true });
  assert.equal(defaultBridge._workerConfig().jsBackedModelStaging, undefined);

  const defaultHarness = createStagingHarness({ jsBackedModelStaging: false, pthreads: false });
  await defaultHarness.runtime.loadModelFromUrl('https://models.test/default.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(defaultHarness.fs.dirs.has('/jsmodels'), false);
  assert.equal(defaultHarness.fs.dirs.has('/models'), true);
  assert.ok(!defaultHarness.runtime._runtimeNotes.includes('model_staging_js_backed'));

  for (const pthreads of [false, true]) {
    const harness = createStagingHarness({ jsBackedModelStaging: true, pthreads });
    assert.equal(harness.bridge._workerConfig().jsBackedModelStaging, true);

    await harness.runtime.loadModelFromUrl('https://models.test/demo.gguf', {
      nCtx: 2048,
      nGpuLayers: 0,
      useCache: false,
      forceRemoteFetchBackend: false,
    });

    assert.deepEqual(harness.ccalls[0], ['llamadart_webgpu_mount_js_model_fs', '/jsmodels']);
    assert.deepEqual(harness.ccalls[1], ['llamadart_webgpu_js_model_fs_prealloc_next', 4]);
    assert.deepEqual(harness.ccalls[2].slice(0, 3), ['llamadart_webgpu_load_model', '/jsmodels/demo.gguf', 2048]);
    assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_unmount_js_model_fs' && c[1] === '/jsmodels'));
    assert.ok(harness.runtime._runtimeNotes.includes('model_staging_js_backed'));
    assert.equal(harness.fs.files.size, 0);
    assert.deepEqual(harness.fs.unmountedPaths, ['/jsmodels']);
  }
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

test('JS allocation failure during staging raises ModelStagingError, unmounts chunks, and retries cleanly', async () => {
  let failAlloc = true;
  const harness = createStagingHarness({
    jsBackedModelStaging: true,
    failWriteWith: () => {
      if (!failAlloc) return null;
      harness.setOomBytes(2_600_000_000);
      const err = new Error('FS error');
      err.name = 'ErrnoError';
      err.errno = 12;
      return err;
    },
  });

  const err = await harness.runtime.loadModelFromUrl('https://models.test/huge.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject'), (e) => e);

  assert.ok(err instanceof ModelStagingError, String(err));
  assert.equal(err.code, 'MODEL_STAGING_ALLOCATION_FAILED');
  assert.match(err.message, /JavaScript memory while staging/);
  assert.equal(harness.fs.files.size, 0, 'failed staging must unlink partial file');
  assert.deepEqual(harness.fs.unmountedPaths, ['/jsmodels'], 'failed staging must unmount/release chunks');

  // Retry succeeds on the same runtime with zero leaked chunks.
  failAlloc = false;
  await harness.runtime.loadModelFromUrl('https://models.test/small.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(harness.fs.files.size, 0);
  assert.deepEqual(harness.fs.unmountedPaths, ['/jsmodels', '/jsmodels']);
});

test('wasm32 4 GiB tensor-size limit raises ModelStagingError instead of raw abort and cleans up for retry', async () => {
  // Case 1: clean native preflight return code (-4).
  let blockOversized = true;
  const preflightHarness = createStagingHarness({
    jsBackedModelStaging: true,
    onLoadModel: ({ setLastError }) => {
      if (blockOversized) {
        setLastError('GGUF tensor payload exceeds the wasm32 4 GiB address space limit (4297927168 > 4294967295 bytes).');
        return -4;
      }
      return 0;
    },
  });

  const preflightErr = await preflightHarness.runtime.loadModelFromUrl('https://models.test/8b.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject'), (e) => e);

  assert.ok(preflightErr instanceof ModelStagingError, String(preflightErr));
  assert.equal(preflightErr.code, 'MODEL_TENSOR_SIZE_LIMIT_EXCEEDED');
  assert.match(preflightErr.message, /wasm32 4 GiB/);
  assert.equal(preflightHarness.fs.files.size, 0);
  assert.deepEqual(preflightHarness.fs.unmountedPaths, ['/jsmodels']);

  // Subsequent retry works cleanly on the still-live core.
  blockOversized = false;
  await preflightHarness.runtime.loadModelFromUrl('https://models.test/4b.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(preflightHarness.fs.files.size, 0);

  // Case 2: raw core abort with gguf_init_from_reader size overflow log is translated too.
  let shouldAbort = true;
  const abortHarness = createStagingHarness({
    jsBackedModelStaging: true,
    onLoadModel: ({ emitLog }) => {
      if (shouldAbort) {
        emitLog("gguf_init_from_reader: tensor 'blk.30.attn_output.weight' size overflow, cannot accumulate size 4288489984 + 9437184");
        throw new Error('RuntimeError: Aborted(undefined)');
      }
      return 0;
    },
  });

  const abortErr = await abortHarness.runtime.loadModelFromUrl('https://models.test/8b.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject'), (e) => e);

  assert.ok(abortErr instanceof ModelStagingError, String(abortErr));
  assert.equal(abortErr.code, 'MODEL_TENSOR_SIZE_LIMIT_EXCEEDED');
  assert.equal(abortHarness.fs.files.size, 0);
  assert.deepEqual(abortHarness.fs.unmountedPaths, ['/jsmodels']);

  // Core was reset cleanly so retry re-initializes instead of reusing a dead aborted core.
  shouldAbort = false;
  await abortHarness.runtime.loadModelFromUrl('https://models.test/0.6b.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(abortHarness.fs.files.size, 0);
});

test('ModelStagingError preserves class and code across worker host and proxy without main-thread fallback', async () => {
  await withStubWorkerEnvironment(async ({ workers }) => {
    const bridge = createRealWorkerBridge({ jsBackedModelStaging: true });
    assert.equal(bridge._shouldFallbackToMainThread(
      new ModelStagingError('MODEL_STAGING_ALLOCATION_FAILED', 'Array buffer allocation failed'),
    ), false);

    const proxy = bridge._workerProxy;
    const driver = workerDriver(proxy, workers[0]);
    driver.ready();

    for (const code of ['MODEL_STAGING_ALLOCATION_FAILED', 'MODEL_TENSOR_SIZE_LIMIT_EXCEEDED']) {
      const call = proxy.call('loadModelFromUrl', ['https://models.test/model.gguf', {}]);
      await driver.settle();
      const pending = driver.calls('loadModelFromUrl').at(-1);
      proxy._worker.onmessage({
        data: {
          type: 'error',
          id: pending.id,
          message: `staging error (${code})`,
          modelStagingErrorCode: code,
        },
      });
      const err = await call.then(() => assert.fail('must reject'), (e) => e);
      assert.ok(err instanceof ModelStagingError, String(err));
      assert.equal(err.code, code);
    }

    proxy._worker.onmessage({ data: { type: 'result', id: 0 } });
    await bridge.dispose().catch(() => {});
  });

  // Verify worker_host serializes ModelStagingError cleanly over postMessage too.
  const replies = [];
  const oldSelf = globalThis.self;
  globalThis.self = { postMessage: (msg) => replies.push(msg) };
  try {
    installBridgeWorkerHost();
    await self.onmessage({ data: { type: 'init', config: { jsBackedModelStaging: true } } });
    assert.ok(replies.some((m) => m.type === 'ready'));
  } finally {
    if (oldSelf === undefined) delete globalThis.self;
    else globalThis.self = oldSelf;
  }
});

test('native model_load.inc checks wasm32 GGUF tensor size limit before llama_model_load_from_file', () => {
  const source = readNativeCoreSource();
  const loader = source.slice(source.indexOf('int32_t load_model_internal('));
  const preLoad = loader.slice(0, loader.indexOf('llama_model_load_from_file('));
  assert.match(preLoad, /if \(!check_wasm32_gguf_tensor_size_limit\(model_path\)\)\s*\{\s*return -4;\s*\}/);
  assert.match(source, /SIZE_MAX <= 0xffffffffULL/);
});
