import assert from 'node:assert/strict';
import { createDirectBridge } from './bridge_operation_queue_fixtures.mjs';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';

// Keep the real facade, load parser, native ccall argument construction,
// generation loop and recovery. Only the core/network/environment are doubles.
function fixture() {
  const { bridge, core } = createDirectBridge({ hello: [new Error('failed to decode')] });
  const runtime = bridge._runtime;
  const loads = [];
  const ccall = core.ccall.bind(core);
  core.ccall = (name, type, types, args = []) => {
    if (name === 'llamadart_webgpu_load_model') {
      loads.push([...args]);
      core.activePrompt = null;
      if (loads.length > 1) core.scripts.hello = ['O', 'K'];
      return 0;
    }
    if (name === 'llamadart_webgpu_free_model') return 0;
    if (name === 'llamadart_webgpu_get_context_size') return 128;
    return ccall(name, type, types, args);
  };
  runtime._modelBytes = 0;
  runtime._coreVariant = 'wasm64';
  runtime._probeBackends = async () => {};
  runtime._ensureCore = async () => core;
  runtime._coreSupportsPthreads = () => true;
  runtime._syncThreadPoolSizeHintFromCore = () => {};
  runtime._getCachedModelResponse = async () => new Response(new Uint8Array([1, 2, 3]));
  runtime._emitLogger = () => {};
  return { bridge, runtime, core, loads };
}

const oldIsolation = Object.getOwnPropertyDescriptor(globalThis, 'crossOriginIsolated');
Object.defineProperty(globalThis, 'crossOriginIsolated', { configurable: true, value: true });
try {
  for (const custom of [
    {},
    { nSeqMax: 3, useMmap: true, useMlock: true, flashAttention: 1, cacheTypeK: 8, cacheTypeV: 2, kvUnified: false,
      ropeFrequencyBase: 1000000, ropeFrequencyScale: 0.5, splitMode: 2, mainGpu: 1, loadMtp: true,
      speculativeRollbackTokenMax: 17, nThreadsBatch: 2, nBatch: 96, nUbatch: 64 },
    { nSeqMax: 2, flashAttention: -1, cacheTypeK: 8, kvUnified: undefined, useMmap: false, useMlock: false,
      ropeFrequencyBase: 0, ropeFrequencyScale: 0, splitMode: -1, mainGpu: -1, nBatch: 999, nUbatch: 999 },
    { nSeqMax: 0, kvUnified: false, nThreads: 0, nThreadsBatch: 0, nBatch: 0, nUbatch: 0 },
    { nSeqMax: -1, flashAttention: 99, cacheTypeK: 99, cacheTypeV: -1, kvUnified: NaN, ropeFrequencyBase: NaN,
      ropeFrequencyScale: -1, splitMode: 99, mainGpu: -99, speculativeRollbackTokenMax: -1 },
  ]) {
    const { bridge, runtime, loads } = fixture();
    const options = { nCtx: 128, nThreads: 1, nGpuLayers: 99, forceRemoteFetchBackend: false, useCache: false, ...custom };
    await bridge.loadModelFromUrl('https://models.test/model.gguf', options);
    const before = loads[0];
    options.ropeFrequencyScale = 42; // Mutating the caller's bag must not alter recovery.
    const preferMemory64 = runtime._preferMemory64;
    assert.equal(await bridge.createCompletion('hello', { nPredict: 2 }), 'OK');
    assert.equal(loads.length, 2, 'the production failure branch must reload exactly once');
    const expected = [...before];
    expected[6] = 0; // Only GPU placement changes in native load arguments.
    assert.deepEqual(loads[1], expected, 'recovery must preserve every resolved native load argument');
    assert.equal(runtime._preferMemory64, preferMemory64);
  }
  for (const loseIsolation of [false, true]) {
    const { bridge, runtime, loads } = fixture();
    runtime._syncThreadPoolSizeHintFromCore = () => {
      runtime._threadPoolSizeHint = loads.length === 0 ? 4 : 2;
      if (loseIsolation && loads.length > 0) {
        Object.defineProperty(globalThis, 'crossOriginIsolated', { configurable: true, value: false });
      }
    };
    try {
      await bridge.loadModelFromUrl('https://models.test/pool.gguf', {
        nCtx: 128, nGpuLayers: 99, nThreads: 4, nThreadsBatch: 3,
        nBatch: 96, nUbatch: 64, ropeFrequencyScale: 0.5, forceRemoteFetchBackend: false,
      });
      assert.equal(await bridge.createCompletion('hello', { nPredict: 2 }), 'OK');
      assert.equal(loads.length, 2);
      const expected = [...loads[0]];
      expected[2] = loseIsolation ? 1 : 2;
      expected[3] = loseIsolation ? 1 : 2;
      expected[6] = 0;
      assert.deepEqual(loads[1], expected, 'recovery must respect replacement pool and isolation limits');
    } finally {
      Object.defineProperty(globalThis, 'crossOriginIsolated', { configurable: true, value: true });
    }
  }
  const { bridge, runtime, loads } = fixture();
  await bridge.loadModelFromUrl('https://models.test/model.gguf', { nGpuLayers: 99, forceRemoteFetchBackend: false });
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(bridge.createCompletion('hello', { signal: controller.signal }), { name: 'AbortError' });
  assert.equal(loads.length, 1);
  runtime._loraAdaptersLoaded = true;
  assert.equal(await runtime._recoverGenerationWithCpuFallback(), false);
  assert.equal(loads.length, 1, 'LoRA recovery refusal is preserved');
  const partial = fixture();
  await partial.bridge.loadModelFromUrl('https://models.test/partial.gguf', { nGpuLayers: 99, forceRemoteFetchBackend: false });
  partial.core.scripts.hello = ['A', new Error('failed to decode')];
  await assert.rejects(partial.bridge.createCompletion('hello'), /failed to decode/);
  assert.equal(partial.loads.length, 1, 'partial generation must never trigger a recovery reload');

  // Real worker-host message dispatch constructs its own facade and runtime.
  // Reuse only the same core/network/environment doubles, with no recovery
  // helper or load parser replaced on this route either.
  const worker = fixture();
  const prototype = LlamaWebGpuBridgeRuntime.prototype;
  const methods = ['_probeBackends', '_ensureCore', '_coreSupportsPthreads', '_syncThreadPoolSizeHintFromCore', '_getCachedModelResponse', '_emitLogger'];
  const originals = new Map(methods.map((method) => [method, prototype[method]]));
  const oldSelf = globalThis.self;
  const replies = [];
  globalThis.self = { postMessage: (message) => replies.push(message) };
  try {
    for (const method of methods) prototype[method] = worker.runtime[method];
    prototype._ensureCore = async function () {
      this._core = worker.core;
      this._coreVariant = 'wasm64';
      return worker.core;
    };
    installBridgeWorkerHost();
    await self.onmessage({ data: { type: 'init', config: {} } });
    await self.onmessage({ data: { type: 'call', id: 1, method: 'loadModelFromUrl', args: [
      'https://models.test/worker.gguf', { nCtx: 128, nGpuLayers: 99, nSeqMax: 2, ropeFrequencyScale: 0.5, cacheTypeK: 8, forceRemoteFetchBackend: false },
    ] } });
    await self.onmessage({ data: { type: 'call', id: 2, method: 'createCompletion', args: ['hello', { nPredict: 2 }] } });
    const result = replies.find((message) => message.id === 2 && message.type === 'result');
    assert.ok(result, JSON.stringify(replies));
    assert.equal(result.value, 'OK');
    assert.equal(worker.loads.length, 2);
    const expected = [...worker.loads[0]];
    expected[6] = 0;
    assert.deepEqual(worker.loads[1], expected, 'worker-host recovery preserves native settings');
  } finally {
    for (const [method, original] of originals) prototype[method] = original;
    if (oldSelf === undefined) delete globalThis.self;
    else globalThis.self = oldSelf;
  }
} finally {
  if (oldIsolation) Object.defineProperty(globalThis, 'crossOriginIsolated', oldIsolation);
  else delete globalThis.crossOriginIsolated;
}

console.log('Generation recovery load option tests passed');
