import assert from 'node:assert/strict';
import { test } from 'node:test';

import * as bridgeModule from '../../js/src/llama_webgpu_bridge.js';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';
import {
  createDirectBridge,
  createRealWorkerBridge,
  withStubWorkerEnvironment,
  workerDriver,
} from './bridge_operation_queue_fixtures.mjs';
import { readNativeCoreSource } from './native_core_source.mjs';

const MODEL_URL = 'https://models.test/m.gguf';
const OPTIONS = { modelBytesHint: 3, nGpuLayers: 0, useCache: false };
const FETCH = 'llamadart_webgpu_load_model_from_url';
const STAGE = 'llamadart_webgpu_load_model';
const ERROR_CODE = 'SINGLE_THREAD_CORE_UNSUPPORTED';
const NATIVE_ERROR = 'Fetch-backed model loading requires a core built with pthreads';

function stubCore(core, { pthreads = false, fetchResult = 0, nativeError = '' } = {}) {
  const calls = [];
  const ccall = core.ccall.bind(core);
  core.ccall = (name, ...args) => {
    calls.push(name);
    if (name === 'llamadart_webgpu_supports_pthreads') return Number(pthreads);
    if (name === 'llamadart_webgpu_pthread_pool_size') return 4;
    if (name === 'llamadart_webgpu_probe') return 0;
    if (name === 'llamadart_webgpu_backends_json') return '[]';
    if (name === FETCH) return fetchResult;
    if (name === STAGE || name === 'llamadart_webgpu_free_model') return 0;
    if (name === 'llamadart_webgpu_last_error' && nativeError) return nativeError;
    return ccall(name, ...args);
  };
  return calls;
}

function setup(t, { worker = true, ...coreOptions } = {}) {
  const { bridge, core } = createDirectBridge();
  t.after(() => bridge.dispose());
  const runtime = bridge._runtime;
  Object.assign(runtime, {
    _modelBytes: 0,
    _isWorkerRuntime: () => worker,
    _resolveRemoteFetchUrl: async (url) => ({ resolvedUrl: url, sizeBytes: 3 }),
    _getCachedModelResponse: async () => new Response(new Uint8Array([1, 2, 3])),
  });
  runtime._config.allowAutoRemoteFetchBackend = true;
  const calls = stubCore(core, coreOptions);
  const notes = () => bridge.getModelMetadata()['llamadart.webgpu.runtime_notes'].split(';');
  return { bridge, calls, notes };
}

function assertSingleThreadError(error) {
  assert.equal(error?.code, ERROR_CODE);
  assert.ok(error instanceof bridgeModule.SingleThreadCoreError);
  assert.equal(error.name, 'SingleThreadCoreError');
  assert.equal(error.feature, 'remote_fetch_backend');
  return true;
}

test('forced worker fetch rejects before entering the fetch-backed native loader', async (t) => {
  const { bridge, calls } = setup(t);
  const error = await bridge.loadModelFromUrl(MODEL_URL, { ...OPTIONS, forceRemoteFetchBackend: true })
    .catch((reason) => reason);
  assert.equal(calls.includes(FETCH), false, 'forced fetch reached the native URL loader');
  assert.equal(calls.includes(STAGE), false);
  assertSingleThreadError(error);
});

test('single-thread auto fetch falls back to streamed staging and reports both notes', async (t) => {
  const { bridge, calls, notes } = setup(t);
  await bridge.loadModelFromUrl(MODEL_URL, OPTIONS);
  assert.equal(calls.includes(FETCH), false);
  assert.equal(calls.includes(STAGE), true);
  assert.ok(notes().includes('core_pthreads:0'));
  assert.ok(notes().includes('core_single_thread'));
  assert.ok(notes().includes('model_fetch_backend_requires_pthreads'));
});

test('threaded worker fetch still reaches the native URL loader', async (t) => {
  const { bridge, calls, notes } = setup(t, { pthreads: true });
  await bridge.loadModelFromUrl(MODEL_URL, { ...OPTIONS, forceRemoteFetchBackend: true });
  assert.equal(calls.includes(FETCH), true);
  assert.equal(calls.includes(STAGE), false);
  assert.ok(notes().includes('core_pthreads:1'));
  assert.ok(notes().includes('model_source_fetch_backend'));
  assert.ok(!notes().includes('core_single_thread'));
  assert.ok(!notes().includes('model_fetch_backend_requires_pthreads'));
});

test('direct single-thread loading still ignores a forced fetch backend', async (t) => {
  const { bridge, calls, notes } = setup(t, { worker: false });
  await bridge.loadModelFromUrl(MODEL_URL, { ...OPTIONS, forceRemoteFetchBackend: true });
  assert.equal(calls.includes(FETCH), false);
  assert.equal(calls.includes(STAGE), true);
  assert.ok(notes().includes('core_single_thread'));
  assert.ok(!notes().includes('model_fetch_backend_requires_pthreads'));
});

test('the native no-pthreads result rejects forced fetch and gives auto fetch a fallback', async (t) => {
  const { bridge, calls, notes } = setup(t, {
    pthreads: true,
    fetchResult: -7,
    nativeError: NATIVE_ERROR,
  });
  await assert.rejects(
    bridge.loadModelFromUrl(MODEL_URL, { ...OPTIONS, forceRemoteFetchBackend: true }),
    (error) => assertSingleThreadError(error) && error.message.includes(NATIVE_ERROR),
  );
  await bridge.loadModelFromUrl(MODEL_URL, { ...OPTIONS, remoteFetchThresholdBytes: 1 });
  assert.equal(calls.filter((name) => name === FETCH).length, 2);
  assert.equal(calls.filter((name) => name === STAGE).length, 1);
  assert.ok(notes().includes('model_fetch_backend_requires_pthreads'));
});

test('the unrelated native -4 result retains the existing worker-only fallback', async (t) => {
  const { bridge, calls, notes } = setup(t, {
    pthreads: true,
    fetchResult: -4,
    nativeError: 'Fetch backend requires a worker-thread bridge runtime on this build',
  });
  await bridge.loadModelFromUrl(MODEL_URL, { ...OPTIONS, forceRemoteFetchBackend: true });
  assert.ok(calls.includes(FETCH));
  assert.ok(calls.includes(STAGE));
  assert.ok(!notes().includes('model_fetch_backend_requires_pthreads'));
});

test('single-thread error round-trips through the host and worker proxy', async (t) => {
  t.mock.method(LlamaWebGpuBridgeRuntime.prototype, '_resolveCoreInitTimeoutMs', () => 50);
  t.mock.method(LlamaWebGpuBridgeRuntime.prototype, '_resolveRemoteFetchUrl', async (url) => ({ resolvedUrl: url }));
  const { core } = createDirectBridge();
  const calls = stubCore(core);
  const oldSelf = globalThis.self;
  const oldScope = globalThis.WorkerGlobalScope;
  const replies = [];
  try {
    globalThis.WorkerGlobalScope = class {
      static [Symbol.hasInstance](value) { return value === globalThis; }
    };
    globalThis.self = { postMessage: (message) => replies.push(message) };
    installBridgeWorkerHost();
    await self.onmessage({ data: {
      type: 'init', config: { coreModuleFactory: async () => core },
    } });
    await self.onmessage({ data: {
      type: 'call', id: 1, method: 'loadModelFromUrl',
      args: [MODEL_URL, { ...OPTIONS, forceRemoteFetchBackend: true }],
    } });
    await self.onmessage({ data: { type: 'call', id: 2, method: 'dispose' } });
  } finally {
    if (oldSelf === undefined) delete globalThis.self;
    else globalThis.self = oldSelf;
    if (oldScope === undefined) delete globalThis.WorkerGlobalScope;
    else globalThis.WorkerGlobalScope = oldScope;
  }
  const reply = replies.find(({ id, type }) => id === 1 && type === 'error');
  assert.equal(reply?.errorCode, ERROR_CODE);
  assert.equal(calls.includes(FETCH), false);

  await withStubWorkerEnvironment(async ({ workers }) => {
    const bridge = createRealWorkerBridge();
    const proxy = bridge._workerProxy;
    const driver = workerDriver(proxy, workers[0]);
    driver.ready();
    const pending = proxy.call('loadModelFromUrl', [MODEL_URL, OPTIONS]);
    const rejected = assert.rejects(pending, assertSingleThreadError);
    await driver.settle();
    proxy._worker.onmessage({ data: { ...reply, id: driver.calls('loadModelFromUrl')[0].id } });
    await rejected;
    proxy._worker.onmessage({ data: { type: 'result', id: 0 } });
    await bridge.dispose();
  });
});

test('the native URL loader rejects no-pthreads builds before creating a fetch backend', () => {
  const source = readNativeCoreSource();
  const loader = source.slice(source.indexOf('EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_load_model_from_url('));
  const backend = loader.indexOf('wasmfs_create_fetch_backend(');
  assert.ok(backend > 0, 'missing native fetch backend');
  assert.match(
    loader.slice(0, backend),
    /#if !defined\(__EMSCRIPTEN_PTHREADS__\)\s+set_error\("Fetch-backed model loading requires a core built with pthreads"\);\s+return -7;\s+#endif/,
  );
});
