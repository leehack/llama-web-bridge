import assert from 'node:assert/strict';

import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';
import { CrossOriginIsolationError } from '../../js/src/llama_webgpu_bridge.js';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';
import {
  createRealWorkerBridge,
  withStubWorkerEnvironment,
  workerDriver,
} from './bridge_operation_queue_fixtures.mjs';

// A core module whose factory never settles, as the pthread core does when
// posting its shared memory to a worker throws DataCloneError.
const HANGING_CORE = `data:text/javascript,export default () => new Promise(() => {});`;
const REACHED_CORE = `data:text/javascript,export default async () => { throw new Error('core factory reached'); };`;

function withUnpostableSharedMemory(run) {
  const original = globalThis.MessageChannel;
  globalThis.MessageChannel = class {
    constructor() {
      this.port1 = {
        postMessage() {
          throw new DOMException('The object can not be cloned.', 'DataCloneError');
        },
        close() {},
      };
      this.port2 = { close() {} };
    }
  };
  return Promise.resolve(run()).finally(() => {
    globalThis.MessageChannel = original;
  });
}

function rejectionWithin(promise, ms) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`did not settle within ${ms}ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

function directBridge(coreModuleUrl, extra = {}) {
  return new LlamaWebGpuBridge({ disableWorker: true, coreModuleUrl, ...extra });
}

// Non-isolated page: shared memory cannot be posted, so core init fails fast.
await withUnpostableSharedMemory(async () => {
  for (const extra of [{}, { coreModuleUrlMem64: HANGING_CORE, preferMemory64: true }]) {
    const bridge = directBridge(HANGING_CORE, extra);
    const started = Date.now();
    const error = await rejectionWithin(
      bridge.loadModelFromUrl('https://models.test/m.gguf', { nGpuLayers: 0 }).then(
        () => assert.fail('load must reject'),
        (rejected) => rejected,
      ),
      2000,
    );
    assert.ok(Date.now() - started < 2000);
    assert.ok(error instanceof CrossOriginIsolationError, String(error));
    assert.equal(error.name, 'CrossOriginIsolationError');
    assert.equal(error.code, 'CROSS_ORIGIN_ISOLATION_REQUIRED');
    assert.match(error.message, /cross-origin isolat/i);
    assert.match(error.message, /Cross-Origin-Embedder-Policy: require-corp/);
    assert.match(error.message, /credentialless/);
    assert.match(error.message, /Safari/);
    await bridge.dispose();
  }
});

// Isolated page: the same configuration reaches the core factory.
{
  const bridge = directBridge(REACHED_CORE);
  await assert.rejects(
    bridge.loadModelFromUrl('https://models.test/m.gguf', { nGpuLayers: 0 }),
    /core factory reached/,
  );
  await bridge.dispose();
}

// A custom core factory is not the pthread core and is never gated.
await withUnpostableSharedMemory(async () => {
  const bridge = new LlamaWebGpuBridge({
    disableWorker: true,
    coreModuleFactory: async () => { throw new Error('custom factory reached'); },
  });
  await assert.rejects(
    bridge.loadModelFromUrl('https://models.test/m.gguf', { nGpuLayers: 0 }),
    /custom factory reached/,
  );
  await bridge.dispose();
});

// The worker host reports the typed error and the proxy restores it.
await withUnpostableSharedMemory(async () => {
  const replies = [];
  const oldSelf = globalThis.self;
  globalThis.self = { postMessage: (message) => replies.push(message) };
  try {
    installBridgeWorkerHost();
    await self.onmessage({ data: { type: 'init', config: { coreModuleUrl: HANGING_CORE } } });
    await self.onmessage({ data: { type: 'call', id: 1, method: 'loadModelFromUrl', args: [
      'https://models.test/m.gguf', { nGpuLayers: 0 },
    ] } });
    const reply = replies.find((message) => message.id === 1);
    assert.equal(reply.type, 'error');
    assert.equal(reply.errorCode, 'CROSS_ORIGIN_ISOLATION_REQUIRED');
  } finally {
    if (oldSelf === undefined) delete globalThis.self;
    else globalThis.self = oldSelf;
  }
});

await withStubWorkerEnvironment(async ({ workers }) => {
  const bridge = createRealWorkerBridge();
  const proxy = bridge._workerProxy;
  const driver = workerDriver(proxy, workers[0]);
  driver.ready();
  const call = proxy.call('loadModelFromUrl', ['https://models.test/m.gguf', {}]);
  await driver.settle();
  proxy._worker.onmessage({ data: {
    type: 'error',
    id: driver.calls('loadModelFromUrl')[0].id,
    message: 'cross-origin isolation required',
    errorCode: 'CROSS_ORIGIN_ISOLATION_REQUIRED',
  } });
  const error = await call.catch((rejected) => rejected);
  assert.ok(error instanceof CrossOriginIsolationError, String(error));
  proxy._worker.onmessage({ data: { type: 'result', id: 0 } });
  await bridge.dispose().catch(() => {});
});

// A core built without pthreads never posts shared memory, so a page that
// opts out loads it on a non-isolated page, capped to one thread.
await withUnpostableSharedMemory(async () => {
  const bridge = directBridge(REACHED_CORE, { allowSingleThreadCore: true });
  await assert.rejects(
    bridge.loadModelFromUrl('https://models.test/m.gguf', { nGpuLayers: 0 }),
    /core factory reached/,
  );
  await bridge.dispose();

  const { createDirectBridge } = await import('./bridge_operation_queue_fixtures.mjs');
  const { bridge: loaded, core } = createDirectBridge();
  const runtime = loaded._runtime;
  runtime._modelBytes = 0;
  const loads = [];
  const ccall = core.ccall.bind(core);
  core.ccall = (name, type, types, args = []) => {
    if (name === 'llamadart_webgpu_load_model') { loads.push([...args]); return 0; }
    if (name === 'llamadart_webgpu_free_model') return 0;
    if (name === 'llamadart_webgpu_get_context_size') return 128;
    return ccall(name, type, types, args);
  };
  runtime._probeBackends = async () => {};
  runtime._ensureCore = async () => core;
  runtime._coreSupportsPthreads = () => true;
  runtime._syncThreadPoolSizeHintFromCore = () => { runtime._threadPoolSizeHint = 4; };
  runtime._getCachedModelResponse = async () => new Response(new Uint8Array([1, 2, 3]));
  runtime._emitLogger = () => {};
  await loaded.loadModelFromUrl('https://models.test/m.gguf', {
    nCtx: 128, nThreads: 4, nThreadsBatch: 4, nGpuLayers: 0, forceRemoteFetchBackend: false, useCache: false,
  });
  assert.equal(loads[0][2], 1);
  assert.equal(loads[0][3], 1);
  assert.match(loaded.getModelMetadata()['llamadart.webgpu.runtime_notes'], /threads_capped_no_coi/);
});

// The opt-out reaches the worker host.
{
  const bridge = directBridge(REACHED_CORE, { allowSingleThreadCore: true });
  assert.equal(bridge._workerConfig().allowSingleThreadCore, true);
  assert.equal(directBridge(REACHED_CORE)._workerConfig().allowSingleThreadCore, undefined);
}
