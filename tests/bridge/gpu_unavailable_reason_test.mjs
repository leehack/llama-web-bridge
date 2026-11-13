import assert from 'node:assert/strict';
import { createDirectBridge } from './bridge_operation_queue_fixtures.mjs';

// Real facade, load path and backend probe. Only the core, the network and
// navigator.gpu are doubles.
async function load({ gpu, coreHasWebGpu = false, nGpuLayers = 99 }) {
  const { bridge, core } = createDirectBridge();
  const runtime = bridge._runtime;
  const ccall = core.ccall.bind(core);
  core.ccall = (name, type, types, args = []) => {
    if (name === 'llamadart_webgpu_probe') return coreHasWebGpu ? 1 : 0;
    if (name === 'llamadart_webgpu_backends_json') return coreHasWebGpu ? '["WebGPU","CPU"]' : '["CPU"]';
    if (name === 'llamadart_webgpu_load_model' || name === 'llamadart_webgpu_free_model') return 0;
    if (name === 'llamadart_webgpu_get_context_size') return 128;
    return ccall(name, type, types, args);
  };
  runtime._modelBytes = 0;
  runtime._ensureCore = async () => core;
  runtime._coreSupportsPthreads = () => true;
  runtime._syncThreadPoolSizeHintFromCore = () => {};
  runtime._getCachedModelResponse = async () => new Response(new Uint8Array([1, 2, 3]));
  const warnings = [];
  runtime._emitLogger = (level, message) => {
    if (level === 'warn') warnings.push(String(message));
  };

  const oldNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { userAgent: 'Chrome', gpu } });
  try {
    await bridge.loadModelFromUrl('https://models.test/model.gguf', {
      nCtx: 128, nThreads: 1, nGpuLayers, forceRemoteFetchBackend: false, useCache: false,
    });
  } finally {
    Object.defineProperty(globalThis, 'navigator', oldNavigator);
  }
  const notes = bridge.getModelMetadata()['llamadart.webgpu.runtime_notes'].split(';');
  return { bridge, warnings, reasons: notes.filter((note) => note.startsWith('gpu_unavailable:')) };
}

function fakeGpu(features) {
  const gpu = { requests: 0 };
  gpu.requestAdapter = async () => {
    gpu.requests += 1;
    return features && { features: new Set(features), info: { vendor: 'nvidia', architecture: 'lovelace' } };
  };
  return gpu;
}

// Chrome on Linux hides shader-f16 on NVIDIA drivers older than 615.71, and
// ggml-webgpu then registers no device: the load must say why it is on CPU.
const hidden = await load({ gpu: fakeGpu(['subgroups', 'timestamp-query']) });
assert.equal(hidden.bridge.isGpuActive(), false);
assert.equal(hidden.bridge.getBackendName(), 'CPU');
assert.deepEqual(hidden.reasons, ['gpu_unavailable:adapter_missing_shader_f16']);
assert.equal(hidden.warnings.length, 1);
assert.match(hidden.warnings[0], /shader-f16/);
assert.match(hidden.warnings[0], /nvidia/);

assert.deepEqual((await load({ gpu: fakeGpu(null) })).reasons, ['gpu_unavailable:no_adapter']);
assert.deepEqual((await load({ gpu: undefined })).reasons, ['gpu_unavailable:no_webgpu']);
const throwing = { requestAdapter: async () => { throw new Error('lost'); } };
assert.deepEqual((await load({ gpu: throwing })).reasons, ['gpu_unavailable:no_adapter']);
// An adapter that qualifies while the core has no WebGPU device is not explained.
assert.deepEqual((await load({ gpu: fakeGpu(['shader-f16']) })).reasons, ['gpu_unavailable:unknown']);

// A working WebGPU load and a CPU-only request stay silent and never ask for an adapter.
for (const quiet of [{ coreHasWebGpu: true }, { nGpuLayers: 0 }]) {
  const gpu = fakeGpu(['subgroups']);
  const loaded = await load({ gpu, ...quiet });
  assert.deepEqual(loaded.reasons, []);
  assert.deepEqual(loaded.warnings, []);
  assert.equal(gpu.requests, 0);
  assert.equal(loaded.bridge.isGpuActive(), quiet.coreHasWebGpu === true);
}
