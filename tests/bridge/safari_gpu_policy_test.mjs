import assert from 'node:assert/strict';
import test from 'node:test';

import * as environment from '../../js/src/internal/environment.ts';
import { createDirectBridge } from './bridge_operation_queue_fixtures.mjs';

const MAC = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)';
const safari = (version) => `${MAC} Version/${version} Safari/605.1.15`;
const IPHONE_27 = 'Mozilla/5.0 (iPhone; CPU iPhone OS 27_0 like Mac OS X) AppleWebKit/605.1.15 '
  + '(KHTML, like Gecko) Version/27.0 Mobile/15E148 Safari/604.1';
const CHROME = `${MAC.replace('605.1.15', '537.36')} Chrome/157.0.0.0 Safari/537.36`;
const PADS = '<pad>'.repeat(48);
const UNIT = [0.6, 0.8];

test('the Safari major version comes from the Version token', () => {
  assert.equal(environment.safariMajorVersion(safari('27.0')), 27);
  assert.equal(environment.safariMajorVersion(safari('26.2')), 26);
  assert.equal(environment.safariMajorVersion(safari('100.1')), 100);
  assert.equal(environment.safariMajorVersion(IPHONE_27), 27);
  assert.equal(environment.safariMajorVersion(`${MAC} Safari/605.1.15`), null);
  assert.equal(environment.safariMajorVersion(CHROME), null);
  assert.equal(environment.safariMajorVersion(undefined), null);
});

test('only a verified Safari version drops the default layer cap', () => {
  assert.equal(environment.safariDefaultMaxGpuLayers(safari('27.0')), Infinity);
  assert.equal(environment.safariDefaultMaxGpuLayers(safari('28.1')), Infinity);
  assert.equal(environment.safariDefaultMaxGpuLayers(IPHONE_27), Infinity);
  assert.equal(environment.safariDefaultMaxGpuLayers(safari('26.4')), 1);
  assert.equal(environment.safariDefaultMaxGpuLayers(`${MAC} Safari/605.1.15`), 1);
});

// Real facade and load path. Only the core, the network, the user agent and
// the probed runtime calls are doubles.
async function load({
  userAgent,
  options = {},
  embeddingModel = false,
  capabilitiesError = null,
  completion = async () => 'Hello there.',
  embed = async () => UNIT,
  acceptLayers = () => true,
}) {
  const oldNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { userAgent } });
  try {
    const { bridge, core } = createDirectBridge();
    const runtime = bridge._runtime;
    const ccall = core.ccall.bind(core);
    const loadedLayers = [];
    core.ccall = (name, type, types, args = []) => {
      if (name === 'llamadart_webgpu_probe') return 1;
      if (name === 'llamadart_webgpu_backends_json') return '["WebGPU","CPU"]';
      if (name === 'llamadart_webgpu_load_model') {
        loadedLayers.push(args[6]);
        return acceptLayers(args[6]) ? 0 : -1;
      }
      if (name === 'llamadart_webgpu_free_model' || name === 'llamadart_webgpu_shutdown') return 0;
      if (name === 'llamadart_webgpu_get_context_size') return 128;
      return ccall(name, type, types, args);
    };
    runtime._modelBytes = 0;
    runtime._ensureCore = async () => core;
    runtime._coreSupportsPthreads = () => true;
    runtime._syncThreadPoolSizeHintFromCore = () => {};
    runtime._getCachedModelResponse = async () => new Response(new Uint8Array([1, 2, 3]));
    runtime._emitLogger = () => {};
    const calls = { completion: 0, embed: 0 };
    runtime.createCompletion = async (...args) => {
      calls.completion += 1;
      return completion(...args);
    };
    runtime.embed = async (...args) => {
      calls.embed += 1;
      return embed(...args);
    };
    runtime.getEmbeddingCapabilities = async () => {
      if (capabilitiesError) throw capabilitiesError;
      return { apiVersion: 1, text: embeddingModel, image: false, audio: false };
    };

    await bridge.loadModelFromUrl('https://models.test/model.gguf', {
      nCtx: 128, nThreads: 1, nGpuLayers: 99, forceRemoteFetchBackend: false, useCache: false, ...options,
    });
    const metadata = bridge.getModelMetadata();
    return {
      calls,
      loadedLayers,
      layers: metadata['llamadart.webgpu.n_gpu_layers'],
      notes: metadata['llamadart.webgpu.runtime_notes'].split(';').filter((note) => note.startsWith('safari_')),
    };
  } finally {
    Object.defineProperty(globalThis, 'navigator', oldNavigator);
  }
}

test('Safari 27 offloads every requested layer by default', async () => {
  for (const userAgent of [safari('27.0'), IPHONE_27]) {
    const result = await load({ userAgent });
    assert.deepEqual(result.loadedLayers, [99]);
    assert.equal(result.layers, '99');
    assert.deepEqual(result.notes, ['safari_gpu_probe_passed']);
    assert.equal(result.calls.completion, 2);
  }
});

test('an older or unknown Safari keeps the one-layer cap', async () => {
  for (const userAgent of [safari('26.2'), `${MAC} Safari/605.1.15`]) {
    const result = await load({ userAgent });
    assert.deepEqual(result.loadedLayers, [1]);
    assert.deepEqual(result.notes, ['safari_gpu_layers_capped:1', 'safari_gpu_probe_passed']);
  }
});

test('safariMaxGpuLayers overrides the default in both directions', async () => {
  const lowered = await load({ userAgent: safari('27.0'), options: { safariMaxGpuLayers: 4 } });
  assert.deepEqual(lowered.loadedLayers, [4]);
  assert.deepEqual(lowered.notes, ['safari_gpu_layers_capped:4', 'safari_gpu_probe_passed']);

  const raised = await load({ userAgent: safari('26.2'), options: { safariMaxGpuLayers: 99 } });
  assert.deepEqual(raised.loadedLayers, [99]);
  assert.deepEqual(raised.notes, ['safari_gpu_probe_passed']);
});

test('another browser is neither capped nor probed', async () => {
  const result = await load({ userAgent: CHROME });
  assert.deepEqual(result.loadedLayers, [99]);
  assert.deepEqual(result.notes, []);
  assert.deepEqual(result.calls, { completion: 0, embed: 0 });
});

// EmbeddingGemma-2 has no language-model head: on the CPU and on WebGPU its
// completion is a run of `<pad>`, which the corruption heuristic rejects.
test('an embedding model is probed with an embedding and stays on the GPU', async () => {
  for (const norm of [0.995, 1, 1.005]) {
    const result = await load({
      userAgent: safari('27.0'),
      embeddingModel: true,
      completion: async () => PADS,
      embed: async () => UNIT.map((value) => value * norm),
    });
    assert.deepEqual(result.loadedLayers, [99]);
    assert.equal(result.layers, '99');
    assert.deepEqual(result.notes, ['safari_gpu_probe_embedding', 'safari_gpu_probe_passed']);
    assert.equal(result.calls.completion, 0);
    assert.equal(result.calls.embed, 2);
  }
});

test('a degenerate or failing embedding probe walks the fallback ladder', async () => {
  // embed() reads a non-finite value as 0, so a NaN vector arrives as zeros.
  const failures = [
    async () => [0, 0],
    async () => [],
    async () => UNIT.map((value) => value * 0.98),
    async () => UNIT.map((value) => value * 1.02),
    async () => { throw new Error('device lost'); },
  ];
  for (const embed of failures) {
    const result = await load({ userAgent: safari('27.0'), embeddingModel: true, embed });
    assert.deepEqual(result.loadedLayers, [99, 1, 0]);
    assert.equal(result.layers, '0');
    assert.deepEqual(result.notes, ['safari_gpu_probe_embedding', 'safari_gpu_probe_failed', 'safari_fallback_cpu']);
  }
});

test('a text model keeps the completion probe and its corruption check', async () => {
  const corrupted = await load({ userAgent: safari('27.0'), completion: async () => PADS });
  assert.deepEqual(corrupted.loadedLayers, [99, 1, 0]);
  assert.deepEqual(corrupted.notes, ['safari_gpu_probe_failed', 'safari_fallback_cpu']);
  assert.equal(corrupted.calls.embed, 0);
});

test('a core without embedding capabilities keeps the completion probe', async () => {
  const result = await load({ userAgent: safari('27.0'), capabilitiesError: new Error('missing export') });
  assert.deepEqual(result.notes, ['safari_gpu_probe_passed']);
  assert.deepEqual(result.calls, { completion: 2, embed: 0 });
});
