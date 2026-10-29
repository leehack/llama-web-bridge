import assert from 'node:assert/strict';
import { test } from 'node:test';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';
import { withStubWorkerEnvironment, createRealWorkerBridge, workerDriver } from './bridge_operation_queue_fixtures.mjs';

const MINUTE = 60 * 1000;
const image = [{ type: 'image', bytes: new Uint8Array([1]) }];
const audio = [{ type: 'audio', samples: new Float32Array([0]) }];
const mixed = [{ type: 'text', text: 'caption' }, ...image, ...audio];
const caps = { apiVersion: 1, text: true, image: true, audio: true };

const flush = () => new Promise((resolve) => setImmediate(resolve));

async function waitForCall(driver, method) {
  for (let i = 0; i < 50; i++) {
    const call = driver.calls(method).at(-1);
    if (call) return call;
    await flush();
  }
  throw new Error(`no ${method} call reached the worker`);
}

function progress(proxy, id, completed, total) {
  proxy._worker.onmessage({ data: { type: 'event', id, event: 'progress', payload: { completed, total } } });
}

function withProxy(config, run) {
  return withStubWorkerEnvironment(async () => {
    const proxy = createRealWorkerBridge(config)._workerProxy;
    try {
      await run(proxy);
    } finally {
      proxy._clearReadyTimeout();
      proxy._worker.terminate();
    }
  });
}

test('media embeddings get a media budget; text keeps the generic request budget', () => withProxy({}, (proxy) => {
  const budget = (method, input) => proxy._resolveRequestTimeoutMs(method, [input, {}]);
  assert.equal(budget('embed', 'text'), 120000);
  assert.equal(budget('embed', [{ type: 'text', text: 'only text parts' }]), 120000);
  assert.equal(budget('embedBatch', ['a', 'b', [{ type: 'text', text: 'c' }]]), 120000);
  assert.equal(budget('embedBatch', []), 120000);
  for (const media of [image, audio, mixed]) {
    assert.equal(budget('embed', media), 10 * MINUTE);
    assert.equal(budget('embedBatch', ['text', media]), 10 * MINUTE);
  }
}));

test('an explicit workerRequestTimeoutMs selects the embedding budget', async () => {
  for (const value of [17000, 3600000]) {
    await withProxy({ workerRequestTimeoutMs: value }, (proxy) => {
      for (const [method, input] of [['embed', 'text'], ['embed', mixed], ['embedBatch', ['text', image]]]) {
        assert.equal(proxy._resolveRequestTimeoutMs(method, [input, {}]), value, method);
      }
    });
  }
  for (const value of [0, -1, NaN, Infinity]) {
    await withProxy({ workerRequestTimeoutMs: value }, (proxy) => {
      assert.equal(proxy._resolveRequestTimeoutMs('embedBatch', [[image], {}]), 10 * MINUTE);
      assert.equal(proxy._resolveRequestTimeoutMs('embedBatch', [['text'], {}]), 120000);
    });
  }
});

async function startWorkerBatch(inputs) {
  const bridge = createRealWorkerBridge();
  const proxy = bridge._workerProxy;
  const driver = workerDriver(proxy, proxy._worker);
  driver.ready();
  const warnings = [];
  bridge._emitBridgeWarn = (message) => warnings.push(message);
  const result = bridge.embedBatch(inputs);
  if (inputs.some(Array.isArray)) {
    const probe = await waitForCall(driver, 'getEmbeddingCapabilities');
    driver.reply(probe.id, caps);
  }
  const call = await waitForCall(driver, 'embedBatch');
  return { bridge, proxy, driver, result, call, warnings };
}

// Each input runs past the generic 2-minute budget for the whole batch, but
// within its own per-input budget.
for (const [label, inputs, perInputMs] of [
  ['a CPU media batch', ['text', image, audio, mixed], 9 * MINUTE],
  ['a long text batch', ['a', 'b', 'c', 'd'], 100 * 1000],
]) test(`${label} slower than the generic deadline stays on the worker while inputs finish`, async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  await withStubWorkerEnvironment(async () => {
    const { bridge, proxy, result, call, warnings } = await startWorkerBatch(inputs);
    for (let completed = 1; completed <= inputs.length; completed++) {
      t.mock.timers.tick(perInputMs);
      await flush();
      if (completed < inputs.length) progress(proxy, call.id, completed, inputs.length);
    }
    const vectors = inputs.map((_, i) => [i, 1]);
    proxy._worker.onmessage({ data: { type: 'result', id: call.id, value: vectors } });
    assert.deepEqual(await result, vectors);
    assert.equal(bridge._workerProxy, proxy, 'the batch must not retire the worker');
    assert.equal(bridge.getModelMetadata()['llamadart.webgpu.execution'], 'worker');
    assert.equal(bridge.getModelMetadata()['llamadart.webgpu.worker_fallback_reason'], undefined);
    assert.deepEqual(warnings, []);
    proxy._clearReadyTimeout();
  });
});

test('a media batch input that stalls past its budget falls back visibly', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  await withStubWorkerEnvironment(async () => {
    const { bridge, proxy, result, call, warnings } = await startWorkerBatch(['text', image]);
    const direct = [];
    bridge._createRuntime = () => ({
      _runtimeNotes: [],
      async embedBatch(inputs) {
        direct.push(inputs);
        return [[0, 1], [1, 0]];
      },
      async dispose() {},
    });
    bridge._ensureRuntimeReadyAfterWorkerFallback = async () => {};
    bridge._ensureRememberedEmbeddingProjector = async () => {};
    bridge._waitForWorkerDisposal = async () => {};
    t.mock.timers.tick(5 * MINUTE);
    await flush();
    progress(proxy, call.id, 1, 2);
    t.mock.timers.tick(10 * MINUTE - 1);
    await flush();
    assert.equal(bridge._workerProxy, proxy, 'progress must re-arm the per-input budget');
    t.mock.timers.tick(1);
    assert.deepEqual(await result, [[0, 1], [1, 0]]);
    assert.equal(direct.length, 1);
    assert.equal(bridge._workerProxy, null);
    assert.deepEqual(bridge._runtime._runtimeNotes, ['worker_fallback:Worker request timeout (embedBatch, 600000ms)']);
    const metadata = bridge.getModelMetadata();
    assert.equal(metadata['llamadart.webgpu.execution'], 'main-thread');
    assert.equal(metadata['llamadart.webgpu.worker_fallback_reason'], 'Worker request timeout (embedBatch, 600000ms)');
    assert.ok(warnings.some((message) => message.includes('falling back to main thread (Worker request timeout (embedBatch, 600000ms))')));
  });
});

test('the runtime reports each finished batch input through the worker host', async () => {
  const originalSelf = globalThis.self;
  const prototype = LlamaWebGpuBridgeRuntime.prototype;
  const originals = { embed: prototype.embed, parts: prototype._embedParts, preflight: prototype._preflightEmbeddingInputs };
  const messages = [];
  const order = [];
  globalThis.self = { postMessage: (message) => { messages.push(message); order.push(`post:${message.type}`); } };
  try {
    prototype._preflightEmbeddingInputs = async () => {};
    prototype.embed = async (text) => { order.push(`embed:${text}`); return [1]; };
    prototype._embedParts = async () => { order.push('parts'); return [2]; };
    installBridgeWorkerHost();
    await self.onmessage({ data: { type: 'init', config: {} } });
    await self.onmessage({ data: { type: 'call', id: 7, method: 'embedBatch', args: [['a', image, 'b'], { normalize: false }] } });
    assert.deepEqual(order, [
      'post:ready', 'embed:a', 'post:event', 'parts', 'post:event', 'embed:b', 'post:event', 'post:result',
    ]);
    assert.deepEqual(messages.filter((message) => message.type === 'event').map(({ id, event, payload }) => ({ id, event, payload })), [
      { id: 7, event: 'progress', payload: { completed: 1, total: 3 } },
      { id: 7, event: 'progress', payload: { completed: 2, total: 3 } },
      { id: 7, event: 'progress', payload: { completed: 3, total: 3 } },
    ]);
    assert.deepEqual(messages.at(-1), { type: 'result', id: 7, value: [[1], [2], [1]] });
  } finally {
    Object.assign(prototype, { embed: originals.embed, _embedParts: originals.parts, _preflightEmbeddingInputs: originals.preflight });
    if (originalSelf === undefined) delete globalThis.self;
    else globalThis.self = originalSelf;
  }
});
