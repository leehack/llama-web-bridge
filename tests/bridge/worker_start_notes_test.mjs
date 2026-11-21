// New behaviour: start notes, and what a refused blob bridge-worker start reports.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { LlamaWebGpuBridge, BlobWorkerBlockedError } from '../../js/src/llama_webgpu_bridge.js';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';
import { BridgeWorkerProxy } from '../../js/src/worker_proxy.ts';
import { blobStartBlockedReasonOf } from '../../js/src/internal/worker_start.ts';

const reachedCore = 'data:text/javascript,export default async () => { throw new Error("core factory reached"); };';
const coreConfig = { coreModuleUrl: reachedCore, allowSingleThreadCore: true, coreInitTimeoutMs: 10_000 };
const CDN_WORKER = 'https://cdn.test/llama_webgpu_bridge_worker.js';
const modelUrl = 'https://models.test/model.gguf';

// Chromium's rule: a cross-origin script URL throws, a blob: URL constructs.
function stub(t, { blob = 'constructs' } = {}) {
  const state = { workers: [], blobs: [], revoked: [], events: new EventTarget(), listeners: 0 };
  const globals = {
    origin: 'https://app.test',
    window: { location: { href: 'https://app.test/page/' } },
    addEventListener(type, listener) { state.listeners += 1; state.events.addEventListener(type, listener); },
    removeEventListener(type, listener) { state.listeners -= 1; state.events.removeEventListener(type, listener); },
    Worker: class {
      constructor(url, options) {
        const href = String(url);
        if (!href.startsWith('blob:') && new URL(href).origin !== 'https://app.test') throw new DOMException('cross-origin', 'SecurityError');
        if (href.startsWith('blob:') && blob === 'throws') throw new TypeError('This document requires TrustedScriptURL assignment');
        Object.assign(this, { url: href, options, posted: [], terminated: false });
        state.workers.push(this);
      }
      postMessage(message) {
        this.posted.push(message);
        if (message.method === 'dispose') this.onmessage({ data: { type: 'result', id: message.id } });
      }
      terminate() { this.terminated = true; }
    },
  };
  const originals = Object.fromEntries(Object.keys(globals).map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
  t.after(() => {
    for (const [key, descriptor] of Object.entries(originals)) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
    }
  });
  t.mock.method(URL, 'createObjectURL', (b) => { state.blobs.push(b); return `blob:https://app.test/${state.blobs.length}`; });
  t.mock.method(URL, 'revokeObjectURL', (url) => state.revoked.push(url));
  state.violation = () => state.events.dispatchEvent(Object.assign(new Event('securitypolicyviolation'), { blockedURI: 'blob', effectiveDirective: 'worker-src', disposition: 'enforce' }));
  return state;
}

test('worker_start:<mode> is re-added on every load, exactly once, and only for a known mode', async () => {
  for (const workerStart of ['blob', 'direct', 'invalid', undefined]) {
    const runtime = new LlamaWebGpuBridgeRuntime({ ...coreConfig, workerStart });
    for (let attempt = 0; attempt < 2; attempt += 1) {
      await assert.rejects(runtime.loadModelFromUrl(modelUrl), /core factory reached/);
      assert.deepEqual(runtime._runtimeNotes.filter((note) => note.startsWith('worker_start')), workerStart === 'blob' || workerStart === 'direct' ? [`worker_start:${workerStart}`] : []);
    }
    await runtime.dispose();
  }
});

test('the proxy tells its worker how it was started', (t) => {
  const state = stub(t);
  const direct = new BridgeWorkerProxy({ moduleUrl: 'https://app.test/llama_webgpu_bridge_worker.js', config: { ...coreConfig } });
  const blob = new BridgeWorkerProxy({ moduleUrl: CDN_WORKER, config: { ...coreConfig } });
  t.after(() => { direct._clearReadyTimeout(); blob._clearReadyTimeout(); blob._endCspWatch(); });
  assert.equal(direct.startMode, 'direct');
  assert.equal(blob.startMode, 'blob');
  assert.deepEqual(state.workers.map((w) => w.posted[0].config.workerStart), ['direct', 'blob']);
  assert.equal(state.listeners, 1);   // only the blob start watches for a CSP refusal
});

for (const [name, setup, reason] of [
  ['a CSP refusal seen before the error', (state) => { state.violation(); state.workers[0].onerror({}); }, 'csp'],
  ['a CSP refusal seen after the error', (state) => { state.workers[0].onerror({}); state.violation(); }, 'csp'],
  ['no violation in an opaque origin', (state) => { globalThis.origin = 'null'; state.workers[0].onerror({}); }, 'unavailable'],
  ['no violation', (state) => { state.workers[0].onerror({}); }, 'script'],
]) {
  test(`a refused blob start keeps main's crash error and adds the reason: ${name}`, async (t) => {
    const state = stub(t);
    const proxy = new BridgeWorkerProxy({ moduleUrl: CDN_WORKER, config: { ...coreConfig } });
    setup(state);
    const error = await proxy._ready.then(() => null, (e) => e);
    assert.equal(error.message, 'Bridge worker crashed');
    assert.equal(error.llamadartWorkerCrash, true);
    assert.ok(!(error instanceof BlobWorkerBlockedError));
    assert.equal(await blobStartBlockedReasonOf(error), reason);
    assert.equal(state.workers.length, 1);   // nothing is started to find out why
    assert.equal(state.listeners, 0);
  });
}

test('a direct worker crash is not a blocked blob start', async (t) => {
  const state = stub(t);
  const proxy = new BridgeWorkerProxy({ moduleUrl: 'https://app.test/llama_webgpu_bridge_worker.js', config: { ...coreConfig } });
  state.workers[0].onerror({ message: 'boom' });
  const error = await proxy._ready.then(() => null, (e) => e);
  assert.equal(error.message, 'boom');
  assert.equal(blobStartBlockedReasonOf(error), undefined);
  assert.equal(state.workers.length, 1);
});

test('bridge worker ready cannot reverse an earlier startup error', async (t) => {
  const state = stub(t);
  const proxy = new BridgeWorkerProxy({ moduleUrl: CDN_WORKER, config: { ...coreConfig } });
  const [worker] = state.workers;
  worker.onerror({ message: 'initialization failed' });
  worker.onmessage({ data: { type: 'ready' } });

  const error = await proxy._ready.then(() => null, (caught) => caught);
  assert.equal(error?.message, 'initialization failed');
  assert.equal(await blobStartBlockedReasonOf(error), 'script');
});

test('the watch ends when the worker is ready, times out, or is disposed', async (t) => {
  const state = stub(t);
  const ready = new BridgeWorkerProxy({ moduleUrl: CDN_WORKER, config: { ...coreConfig } });
  state.workers[0].onmessage({ data: { type: 'ready' } });
  await ready._ready;
  assert.equal(state.listeners, 0);
  const disposed = new BridgeWorkerProxy({ moduleUrl: CDN_WORKER, config: { ...coreConfig } });
  assert.equal(state.listeners, 1);
  state.workers[1].onmessage({ data: { type: 'ready' } });
  await disposed.dispose();
  assert.equal(state.listeners, 0);
});

test('the facade falls back as on main and keeps worker_start_blocked across a replacement runtime', { timeout: 5000 }, async (t) => {
  const state = stub(t);
  const bridge = new LlamaWebGpuBridge({ ...coreConfig, workerUrl: CDN_WORKER });
  const load = bridge.loadModelFromUrl(modelUrl);
  state.workers[0].onerror({});
  state.violation();
  await assert.rejects(load, /core factory reached/);
  await new Promise((resolve) => setTimeout(resolve, 0));
  const metadata = bridge.getModelMetadata();
  assert.equal(metadata['llamadart.webgpu.execution'], 'main-thread');
  assert.equal(metadata['llamadart.webgpu.worker_fallback_reason'], 'Bridge worker crashed');
  assert.ok(metadata['llamadart.webgpu.runtime_notes'].split(';').includes('worker_start_blocked:csp'));
  assert.ok(bridge._runtime._runtimeNotes.includes('worker_start_blocked:csp'));
  const previous = bridge._runtime;
  bridge._handleRuntimeCoreRetired(previous);
  assert.notEqual(bridge._runtime, previous);
  await assert.rejects(bridge.loadModelFromUrl(modelUrl), /core factory reached/);
  assert.ok(bridge._runtime._runtimeNotes.includes('worker_start_blocked:csp'));
  await bridge.dispose();
});

test('a blob constructor that throws still falls back, with worker_start_blocked:unavailable', { timeout: 5000 }, async (t) => {
  stub(t, { blob: 'throws' });
  const bridge = new LlamaWebGpuBridge({ ...coreConfig, workerUrl: CDN_WORKER });
  await assert.rejects(bridge.loadModelFromUrl(modelUrl), /core factory reached/);
  assert.equal(bridge.getModelMetadata()['llamadart.webgpu.worker_fallback_reason'], 'This document requires TrustedScriptURL assignment');
  assert.ok(bridge._runtime._runtimeNotes.includes('worker_start_blocked:unavailable'));
  await bridge.dispose();
});

test('a pthread block reported by the worker is the typed error on the page and never a fallback', async (t) => {
  const state = stub(t);
  const bridge = new LlamaWebGpuBridge({ ...coreConfig, workerUrl: 'https://app.test/llama_webgpu_bridge_worker.js' });
  state.workers[0].onmessage({ data: { type: 'ready' } });
  const load = bridge.loadModelFromUrl(modelUrl);
  await new Promise((resolve) => setImmediate(resolve));
  const call = state.workers[0].posted.find((m) => m.type === 'call');
  state.workers[0].onmessage({ data: { type: 'error', id: call.id, message: 'blocked', errorCode: 'BLOB_WORKER_BLOCKED', blobWorkerBlockReason: 'csp' } });
  await assert.rejects(load, (error) => error instanceof BlobWorkerBlockedError && error.reason === 'csp' && error.target === 'pthread' && error.message === 'blocked');
  assert.equal(bridge.getModelMetadata()['llamadart.webgpu.execution'], 'worker');
  // Whatever its text says, a pthread block would fail the same way on the page.
  assert.equal(bridge._shouldFallbackToMainThread(new BlobWorkerBlockedError('script', 'Bridge worker request timed out')), false);
  await bridge.dispose();
});
