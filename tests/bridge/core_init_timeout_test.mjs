import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { test } from 'node:test';

import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';

function observeInitTimers(t) {
  const pending = new Set();
  const started = t.mock.method(globalThis, 'setTimeout', (_callback, delay) => {
    assert.equal(delay, 90000);
    const timer = Symbol('core init timer');
    pending.add(timer);
    return timer;
  });
  t.mock.method(globalThis, 'clearTimeout', (timer) => pending.delete(timer));
  return { started, pending };
}

test('core initialization clears its timeout after success', async (t) => {
  const { started, pending } = observeInitTimers(t);
  const runtime = new LlamaWebGpuBridgeRuntime({ coreModuleFactory: async () => ({ ccall: () => 0 }) });

  assert.ok(await runtime._ensureCore());

  assert.equal(started.mock.callCount(), 1);
  assert.equal(pending.size, 0);
});

test('core initialization clears its timeout after factory rejection', async (t) => {
  const { started, pending } = observeInitTimers(t);
  const failure = new Error('core initialization failed');
  const runtime = new LlamaWebGpuBridgeRuntime({ coreModuleFactory: async () => { throw failure; } });

  await assert.rejects(runtime._ensureCore(), (error) => error === failure);

  assert.equal(started.mock.callCount(), 1);
  assert.equal(pending.size, 0);
});

test('memory64 fallback clears both core initialization timers', async (t) => {
  const { started, pending } = observeInitTimers(t);
  const runtime = new LlamaWebGpuBridgeRuntime({
    allowSingleThreadCore: true,
    logLevel: 0,
    coreModuleUrlMem64: 'data:text/javascript,export default async () => { throw new Error("memory64 init failed"); };',
    coreModuleUrl: 'data:text/javascript,export default async () => ({ ccall: () => 0 });',
  });

  await runtime._ensureCore();

  assert.equal(runtime._coreVariant, 'wasm32');
  assert.equal(started.mock.callCount(), 2);
  assert.equal(pending.size, 0);
});

test('a hung core still rejects at the configured initialization timeout', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const runtime = new LlamaWebGpuBridgeRuntime({
    coreInitTimeoutMs: 12000,
    coreModuleFactory: () => new Promise(() => {}),
  });
  let settled = false;
  const rejected = assert.rejects(runtime._ensureCore().finally(() => { settled = true; }), {
    name: 'Error',
    message: 'Bridge core init timeout (12000ms)',
  });

  await setImmediate();
  t.mock.timers.tick(11999);
  await setImmediate();
  assert.equal(settled, false);

  t.mock.timers.tick(1);
  await rejected;
});
