// New behaviour: how the runtime offers the hook and reports the start.
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import { setImmediate } from 'node:timers/promises';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';
import { BlobWorkerBlockedError } from '../../js/src/internal/worker_start.ts';
const HOOK = Symbol.for('llamadart.pthreadWorker');
const CORE = 'https://cdn.test/core.js';

// A fake core in the shape of the Emscripten output: it starts its pool through
// the hook when given one, as src/llama_webgpu_core_pthread_pre.js does.
let coreSerial = 0;
function fakeCore(t, { scriptUrl, pool = 4, ready = 'on-load', wrap = '(options) => options', check = 'message' } = {}) {
  const state = { calls: 0, workers: [], checks: [], blobs: [] };
  const moduleUrl = `fake-core:${coreSerial += 1}`;
  globalThis.__fakeCore = { state, scriptUrl, pool, ready };
  const source = `
    const wrap = ${wrap};
    export default async (given) => {
      const options = wrap(given);
      const { state, scriptUrl, pool, ready } = globalThis.__fakeCore;
      state.calls += 1;
      const Worker = options[Symbol.for('llamadart.pthreadWorker')] || globalThis.Worker;
      state.Worker = Worker;
      if (scriptUrl) {
        const workers = Array.from({ length: pool }, () => new Worker(new URL(scriptUrl), { type: 'module', name: 'em-pthread' }));
        if (ready === 'never') await new Promise(() => {});
        if (ready === 'manual') await new Promise((_, reject) => { state.rejectFactory = reject; });
        await Promise.all(workers.map((worker) => worker.loaded));
      }
      return { ccall: () => 0 };
    };`;
  const hook = registerHooks({
    resolve: (specifier, context, next) => (specifier === CORE ? { url: moduleUrl, shortCircuit: true } : next(specifier, context)),
    load: (url, context, next) => (url === moduleUrl ? { format: 'module', source, shortCircuit: true } : next(url, context)),
  });
  const globals = {
    origin: 'https://app.test',
    Worker: class extends EventTarget {
      constructor(url, options) {
        super();
        Object.assign(this, { url, options, terminated: false });
        if (options?.name !== 'em-pthread') {
          state.checks.push(this);
          if (check === 'message') queueMicrotask(() => this.dispatchEvent(new Event('message')));
          return;
        }
        this.loaded = new Promise((resolve) => { this.load = resolve; });
        state.workers.push(this);
        if (ready === 'throw' && String(url).startsWith('blob:')) throw new TypeError('This document requires TrustedScriptURL assignment');
        if (ready === 'on-load') this.load();
        if (ready === 'error') queueMicrotask(() => { this.dispatchEvent(new Event('error')); this.onerror?.({}); });
      }
      terminate() { this.terminated = true; }
    },
  };
  const originals = Object.fromEntries(Object.keys(globals).map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
  t.mock.method(URL, 'createObjectURL', (blob) => { state.blobs.push(blob); return `blob:https://app.test/${state.blobs.length}`; });
  t.mock.method(URL, 'revokeObjectURL', () => {});
  t.after(() => {
    hook.deregister();
    delete globalThis.__fakeCore;
    for (const [key, descriptor] of Object.entries(originals)) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
    }
  });
  return state;
}
const runtimeFor = (config = {}) => new LlamaWebGpuBridgeRuntime({ allowSingleThreadCore: true, coreInitTimeoutMs: 10_000, coreModuleUrl: CORE, ...config });

test('a core that starts no worker: one factory call, no worker, no blob', async (t) => {
  const state = fakeCore(t, {});
  const runtime = runtimeFor();
  await runtime._ensureCore();
  assert.deepEqual([state.calls, state.workers.length, state.blobs.length], [1, 0, 0]);
  assert.equal(runtime._pthreadStart, 'direct');
});

test('a same-origin pthread script: the pool is constructed from the core\'s own URL', async (t) => {
  const state = fakeCore(t, { scriptUrl: 'https://app.test/assets/core.js' });
  const runtime = runtimeFor();
  await runtime._ensureCore();
  assert.deepEqual([state.calls, state.workers.length, state.blobs.length], [1, 4, 0]);
  assert.ok(state.workers.every((worker) => worker.url instanceof URL && worker.url.href === 'https://app.test/assets/core.js'));
  assert.equal(runtime._pthreadStart, 'direct');
});

test('a cross-origin pthread script: the pool starts from the blob wrapper of the URL the core used', async (t) => {
  // The factory was imported from CORE, but its worker script is a sibling file: the hook sees the real one.
  const state = fakeCore(t, { scriptUrl: 'https://cdn.test/other/llama_webgpu_core.js' });
  const runtime = runtimeFor();
  await runtime._ensureCore();
  assert.deepEqual([state.calls, state.workers.length, state.blobs.length], [1, 4, 1]);
  assert.equal(await state.blobs[0].text(), 'import "https://cdn.test/other/llama_webgpu_core.js";\n');
  assert.ok(state.workers.every((worker) => worker.url === 'blob:https://app.test/1' && worker.options.name === 'em-pthread'));
  assert.equal(runtime._pthreadStart, 'blob');
});

test('a wrapper that copies its argument keeps the hook', async (t) => {
  const state = fakeCore(t, { scriptUrl: 'https://cdn.test/copy/core.js', wrap: '(options) => ({ ...options })' });
  const runtime = runtimeFor();
  await runtime._ensureCore();
  assert.equal(state.blobs.length, 1);
  assert.equal(runtime._pthreadStart, 'blob');
});

test('a wrapper that drops the hook behaves as on main', async (t) => {
  const state = fakeCore(t, { scriptUrl: 'https://cdn.test/drop/core.js', wrap: '({ locateFile, print, printErr, onAbort }) => ({ locateFile, print, printErr, onAbort })' });
  const runtime = runtimeFor();
  await runtime._ensureCore();
  assert.equal(state.blobs.length, 0);
  assert.ok(state.workers.every((worker) => worker.url.href === 'https://cdn.test/drop/core.js'));
  assert.equal(runtime._pthreadStart, 'direct');
});

test('a refused blob start rejects the load once with the typed error, well before the init timeout', async (t) => {
  const state = fakeCore(t, { scriptUrl: 'https://cdn.test/blocked/core.js', ready: 'error' });
  const runtime = runtimeFor();
  const started = performance.now();
  await assert.rejects(runtime.loadModelFromUrl('https://models.test/m.gguf'), (error) => error instanceof BlobWorkerBlockedError
    && error.target === 'pthread' && error.reason === 'script');
  assert.ok(performance.now() - started < 3000);
  assert.equal(state.calls, 1);
  assert.equal(runtime._core, null);
  assert.equal(state.workers.length, 4);
  assert.ok(state.workers.every((worker) => worker.terminated));
});

test('a blob start that never answers ends in the caller\'s own init timeout, untyped', async (t) => {
  const state = fakeCore(t, { scriptUrl: 'https://cdn.test/slow/core.js', ready: 'never' });
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const runtime = runtimeFor({ coreInitTimeoutMs: 12_000 });
  const start = assert.rejects(runtime._ensureCore(), /Bridge core init timeout \(12000ms\)/);
  for (let i = 0; i < 20; i += 1) await Promise.resolve();
  t.mock.timers.tick(12_000);
  await start;
  assert.equal(state.calls, 1);
});

test('a pending classifier is stopped at the 10-second core-init deadline', async (t) => {
  const state = fakeCore(t, { scriptUrl: 'https://cdn.test/deadline/core.js', ready: 'never', check: 'silent' });
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const startTimer = t.mock.method(globalThis, 'setTimeout');
  const clearTimer = t.mock.method(globalThis, 'clearTimeout');
  const runtime = runtimeFor();
  let error;
  const started = runtime._ensureCore().catch((caught) => { error = caught; });
  await setImmediate();

  t.mock.timers.tick(9900);
  state.workers[0].dispatchEvent(new Event('error'));
  t.mock.timers.tick(99);
  await setImmediate();
  const [classifier] = state.checks;
  assert.ok(classifier);
  assert.equal(classifier.terminated, false);
  assert.equal(error, undefined);

  t.mock.timers.tick(1);
  await setImmediate();
  assert.equal(error?.name, 'Error');
  assert.equal(error.message, 'Bridge core init timeout (10000ms)');
  const initTimer = startTimer.mock.calls.find(({ arguments: args }) => args[1] === 10_000).result;
  assert.ok(clearTimer.mock.calls.some(({ arguments: args }) => args[0] === initTimer));
  assert.equal(classifier.terminated, true);
  await started;
});

test('a pending classifier cannot delay or replace a factory error', async (t) => {
  const state = fakeCore(t, { scriptUrl: 'https://cdn.test/factory-error/core.js', ready: 'manual', check: 'silent' });
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const runtime = runtimeFor();
  let error;
  const started = runtime._ensureCore().catch((caught) => { error = caught; });
  await setImmediate();

  state.workers[0].dispatchEvent(new Event('error'));
  t.mock.timers.tick(1);
  await setImmediate();
  const [classifier] = state.checks;
  assert.ok(classifier);
  assert.equal(classifier.terminated, false);

  const factoryError = new Error('factory failed independently');
  state.rejectFactory(factoryError);
  await setImmediate();
  assert.equal(error, factoryError);
  assert.equal(classifier.terminated, true);
  classifier.dispatchEvent(new Event('message'));
  await started;
});

test('a completed classification wins against a pending factory and init timer', async (t) => {
  const state = fakeCore(t, { scriptUrl: 'https://cdn.test/classified/core.js', ready: 'never', check: 'silent' });
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const runtime = runtimeFor();
  let error;
  const started = runtime._ensureCore().catch((caught) => { error = caught; });
  await setImmediate();

  state.workers[0].dispatchEvent(new Event('error'));
  t.mock.timers.tick(1);
  await setImmediate();
  state.checks[0].dispatchEvent(new Event('message'));
  await setImmediate();
  assert.ok(error instanceof BlobWorkerBlockedError);
  assert.equal(error.reason, 'script');
  await started;
});

test('disposing a runtime abandons a pending blob pool and prevents late worker creation', async (t) => {
  const scriptUrl = 'https://cdn.test/disposed-pool/core.js';
  const state = fakeCore(t, { scriptUrl, ready: 'manual' });
  const runtime = runtimeFor();
  const failure = new Error('factory stopped');
  const start = assert.rejects(runtime._ensureCore(), (error) => error === failure);
  await setImmediate();
  assert.equal(state.workers.length, 4);
  assert.ok(state.workers.every((worker) => !worker.terminated));
  await runtime.dispose();

  assert.ok(state.workers.every((worker) => worker.terminated));
  assert.throws(() => new state.Worker(new URL(scriptUrl), { type: 'module', name: 'em-pthread' }),
    /core initialization was abandoned/);
  assert.equal(state.workers.length, 4);
  state.rejectFactory(failure);
  await start;
});

test('a refused memory64 start falls back to the wasm32 core like any other init error', async (t) => {
  const state = fakeCore(t, { scriptUrl: 'https://cdn.test/mem64/core.js', ready: 'error' });
  const runtime = new LlamaWebGpuBridgeRuntime({
    allowSingleThreadCore: true, coreInitTimeoutMs: 10_000, preferMemory64: true, coreModuleUrlMem64: CORE,
    coreModuleUrl: 'data:text/javascript,export default async () => ({ ccall: () => 0 });',
  });
  runtime._preferMemory64 = true;
  await runtime._ensureCore();
  assert.equal(runtime._coreVariant, 'wasm32');
  assert.ok(runtime._runtimeNotes.includes('core_mem64_unavailable:init_error'));
  assert.equal(runtime._pthreadStart, 'direct');
  assert.equal(state.calls, 1);
});

test('a blob constructor that throws inside the core rejects the load with the typed error, not the browser one', async (t) => {
  const state = fakeCore(t, { scriptUrl: 'https://cdn.test/tt/core.js', ready: 'throw' });
  const runtime = runtimeFor();
  await assert.rejects(runtime.loadModelFromUrl('https://models.test/m.gguf'), (error) => error instanceof BlobWorkerBlockedError && error.reason === 'unavailable');
  assert.equal(state.calls, 1);
});
