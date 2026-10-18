import assert from 'node:assert/strict';
import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';
import { withStubWorkerEnvironment, createRealWorkerBridge, workerDriver } from './bridge_operation_queue_fixtures.mjs';

const timeoutKeys = [
  'workerRequestTimeoutMs', 'workerInitTimeoutMs', 'workerModelLoadTimeoutMs',
  'workerMmprojLoadTimeoutMs', 'workerCompletionTimeoutMs', 'workerTextToSpeechTimeoutMs',
  'coreInitTimeoutMs',
];
const methods = ['tokenize', 'loadModelFromUrl', 'loadDraftModel', 'loadMultimodalProjector', 'createCompletion', 'synthesizeSpeech'];

await withStubWorkerEnvironment(async () => {
  for (const value of [undefined, 0, -1, NaN, Infinity, 1, 23456.9, 99999999]) {
    const config = Object.fromEntries(timeoutKeys.map((key) => [key, value]));
    // Exercise construction's real facade -> proxy -> init message route.
    const bridge = createRealWorkerBridge(config);
    const proxy = bridge._workerProxy;
    assert.ok(proxy);
    const init = proxy._worker.posted.find((message) => message.type === 'init');
    assert.ok(init);
    for (const key of timeoutKeys) {
      assert.equal(proxy._config[key], value, `${key} must reach the proxy unchanged`);
      assert.equal(init.config[key], value, `${key} must reach the worker host unchanged`);
    }
    const expectedRequest = Number.isFinite(value) && value > 0
      ? Math.max(5000, Math.min(3600000, Math.trunc(value))) : null;
    const defaults = [120000, 180000, 180000, 480000, 360000, 1200000];
    methods.forEach((method, i) => assert.equal(proxy._resolveRequestTimeoutMs(method), expectedRequest ?? defaults[i], method));
    assert.equal(proxy._resolveWorkerReadyTimeoutMs(), Number.isFinite(value) && value > 0
      ? Math.max(3000, Math.min(120000, Math.trunc(value))) : 20000);
    // The host spreads init.config into its direct facade; exercise that
    // runtime configuration path without replacing its timeout resolver.
    const host = new LlamaWebGpuBridge({ ...structuredClone(init.config), disableWorker: true });
    assert.equal(host._runtime._resolveCoreInitTimeoutMs(), Number.isFinite(value) && value > 0
      ? Math.max(10000, Math.min(600000, Math.trunc(value))) : 90000);
    await host.dispose();
    const driver = workerDriver(proxy, proxy._worker);
    driver.ready();
    const disposed = bridge.dispose();
    await driver.settle();
    const call = driver.calls('dispose').at(-1);
    driver.reply(call.id, undefined);
    await disposed;
  }
  // Per-method overrides win over the global timeout, including after restart.
  const bridge = createRealWorkerBridge({
    workerRequestTimeoutMs: 17000,
    workerModelLoadTimeoutMs: 27000,
    workerMmprojLoadTimeoutMs: 37000,
    workerCompletionTimeoutMs: 47000,
    workerTextToSpeechTimeoutMs: 57000,
    workerInitTimeoutMs: 7000,
    coreInitTimeoutMs: 67000,
  });
  const replacement = bridge._createWorkerProxy();
  for (const proxy of [bridge._workerProxy, replacement]) {
    assert.deepEqual(methods.map((method) => proxy._resolveRequestTimeoutMs(method)), [17000, 27000, 27000, 37000, 47000, 57000]);
    assert.equal(proxy._resolveWorkerReadyTimeoutMs(), 7000);
    assert.equal(new LlamaWebGpuBridgeRuntime(proxy._config)._resolveCoreInitTimeoutMs(), 67000);
    proxy._clearReadyTimeout();
    proxy._worker.terminate();
  }
  for (const invalid of [0, -1, NaN, Infinity]) {
    const fallback = createRealWorkerBridge({
      workerRequestTimeoutMs: 17000,
      workerModelLoadTimeoutMs: invalid,
      workerMmprojLoadTimeoutMs: invalid,
      workerCompletionTimeoutMs: invalid,
      workerTextToSpeechTimeoutMs: invalid,
    })._workerProxy;
    assert.deepEqual(methods.map((method) => fallback._resolveRequestTimeoutMs(method)), methods.map(() => 17000));
    fallback._clearReadyTimeout();
    fallback._worker.terminate();
  }
});

// Exercise the host's real init-message handler and runtime construction.
// Stop at the backend probe, before downloading or compiling a core.
const originalSelf = globalThis.self;
const originalProbe = LlamaWebGpuBridgeRuntime.prototype._probeBackends;
let hostTimeout;
const replies = [];
globalThis.self = { postMessage: (message) => replies.push(message) };
try {
  LlamaWebGpuBridgeRuntime.prototype._probeBackends = async function () {
    hostTimeout = this._resolveCoreInitTimeoutMs();
    throw new Error('test stopped before core initialization');
  };
  installBridgeWorkerHost();
  const config = new LlamaWebGpuBridge({ disableWorker: true, coreInitTimeoutMs: 67000 })._workerConfig();
  await self.onmessage({ data: { type: 'init', config } });
  await self.onmessage({ data: { type: 'call', id: 1, method: 'loadModelFromUrl', args: ['model.gguf'] } });
  assert.equal(hostTimeout, 67000, 'core timeout must survive worker-host initialization');
  assert.ok(replies.some((message) => message.type === 'error' && message.id === 1));
} finally {
  LlamaWebGpuBridgeRuntime.prototype._probeBackends = originalProbe;
  if (originalSelf === undefined) delete globalThis.self;
  else globalThis.self = originalSelf;
}

console.log('Worker timeout configuration tests passed');
