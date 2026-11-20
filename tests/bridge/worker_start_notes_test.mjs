import assert from 'node:assert/strict';
import { test } from 'node:test';

import { BlobWorkerBlockedError, LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';
import { workerDriver } from './bridge_operation_queue_fixtures.mjs';
import { stubWorkerStart } from './worker_blob_start_fixture.mjs';

const reachedCore = 'data:text/javascript,export default async () => { throw new Error("core factory reached"); };';
const coreConfig = { coreModuleUrl: reachedCore, allowSingleThreadCore: true, coreInitTimeoutMs: 10_000 };
const workerUrl = 'https://cdn.test/llama_webgpu_bridge_worker.js';
const modelUrl = 'https://models.test/model.gguf';
const workerNotePrefix = 'worker_start';

test('the worker start mode survives the per-load note reset exactly once', async () => {
  for (const workerStart of ['blob', 'direct', 'invalid']) {
    const runtime = new LlamaWebGpuBridgeRuntime({ ...coreConfig, workerStart });
    try {
      for (let attempt = 0; attempt < 2; attempt++) {
        await assert.rejects(runtime.loadModelFromUrl(modelUrl), /core factory reached/);
        assert.deepEqual(
          runtime._runtimeNotes.filter((note) => note.startsWith(workerNotePrefix)),
          workerStart === 'invalid' ? [] : [`worker_start:${workerStart}`],
        );
      }
    } finally {
      await runtime.dispose();
    }
  }
});

test('the blocked worker note survives a direct load and runtime retirement', { timeout: 2500 }, async (t) => {
  const state = stubWorkerStart(t);
  const bridge = new LlamaWebGpuBridge({ ...coreConfig, workerUrl });
  try {
    const load = bridge.loadModelFromUrl(modelUrl);
    state.violation();
    state.workers[0].onerror({});
    await assert.rejects(load, /core factory reached/);
    const metadata = bridge.getModelMetadata();
    assert.equal(metadata['llamadart.webgpu.execution'], 'main-thread');
    assert.match(metadata['llamadart.webgpu.worker_fallback_reason'], /worker-src blob:/);
    assert.ok(bridge._runtime._runtimeNotes.includes('worker_start_blocked:csp'));

    const previous = bridge._runtime;
    bridge._handleRuntimeCoreRetired(previous);
    assert.notEqual(bridge._runtime, previous);
    assert.deepEqual(bridge._runtime._startNotes, ['worker_start_blocked:csp']);
    await assert.rejects(bridge.loadModelFromUrl(modelUrl), /core factory reached/);
    assert.ok(bridge._runtime._runtimeNotes.includes('worker_start_blocked:csp'));
  } finally {
    await bridge.dispose();
  }
});

test('a synchronous blob construction failure retains its fallback note', async (t) => {
  const state = stubWorkerStart(t, {
    construct() { throw new TypeError('This document requires TrustedScriptURL assignment'); },
  });
  const bridge = new LlamaWebGpuBridge({ ...coreConfig, workerUrl });
  try {
    assert.equal(state.workers.length, 1);
    assert.equal(state.listeners.size, 0);
    assert.deepEqual(state.revoked, [state.workers[0].url]);
    await assert.rejects(bridge.loadModelFromUrl(modelUrl), /core factory reached/);
    assert.ok(bridge._runtime._runtimeNotes.includes('worker_start_blocked:unavailable'));
  } finally {
    await bridge.dispose();
  }
});

test('only a bridge-worker block is recoverable through direct mode', async () => {
  const bridge = new LlamaWebGpuBridge({ disableWorker: true });
  assert.equal(bridge._shouldFallbackToMainThread(new BlobWorkerBlockedError('csp', 'pthread')), false);
  assert.equal(bridge._shouldFallbackToMainThread(new BlobWorkerBlockedError('csp', 'bridge_worker')), true);
  await bridge.dispose();
});

test('the pthread error survives the worker protocol without a direct retry', async (t) => {
  const state = stubWorkerStart(t);
  const bridge = new LlamaWebGpuBridge({ ...coreConfig, workerUrl });
  try {
    const [worker] = state.workers;
    const driver = workerDriver(bridge._workerProxy, worker);
    driver.ready();
    const load = bridge.loadModelFromUrl(modelUrl);
    await driver.settle();
    const [call] = driver.calls('loadModelFromUrl');
    assert.ok(call);
    worker.onmessage({
      data: {
        type: 'error', id: call.id, message: 'pthread start is blocked',
        errorCode: 'BLOB_WORKER_BLOCKED', blobWorkerBlockReason: 'script',
      },
    });
    await assert.rejects(load, (error) => error instanceof BlobWorkerBlockedError
      && error.code === 'BLOB_WORKER_BLOCKED' && error.reason === 'script' && error.target === 'pthread');
    assert.equal(bridge._runtime, null);
    assert.ok(bridge._workerProxy);
  } finally {
    await bridge.dispose();
  }
});

for (const [mode, assetOrigin] of [['blob', 'https://cdn.test'], ['direct', 'https://app.test']]) {
  test(`worker init resolves asset URLs against the page base in ${mode} mode`, async (t) => {
    const state = stubWorkerStart(t);
    const originalDocument = Object.getOwnPropertyDescriptor(globalThis, 'document');
    Object.defineProperty(globalThis, 'document', {
      configurable: true, value: { baseURI: 'https://app.test/asset-base/' },
    });
    t.after(() => {
      if (originalDocument) Object.defineProperty(globalThis, 'document', originalDocument);
      else delete globalThis.document;
    });
    const bridge = new LlamaWebGpuBridge({
      workerUrl: assetOrigin + '/off-origin-path/llama_webgpu_bridge_worker.js',
      wasmUrl: '/models/core.wasm',
      wasmUrlMem64: './core64.wasm',
      coreModuleUrl: './models/core.js',
      coreModuleUrlMem64: '/models/core64.js',
    });
    const [worker] = state.workers;
    const driver = workerDriver(bridge._workerProxy, worker);
    driver.ready();
    try {
      const { config } = worker.posted[0];
      assert.equal(config.workerStart, mode);
      assert.equal(config.wasmUrl, 'https://app.test/models/core.wasm');
      assert.equal(config.wasmUrlMem64, 'https://app.test/asset-base/core64.wasm');
      assert.equal(config.coreModuleUrl, 'https://app.test/asset-base/models/core.js');
      assert.equal(config.coreModuleUrlMem64, 'https://app.test/models/core64.js');
    } finally {
      await bridge.dispose();
    }
  });
}
