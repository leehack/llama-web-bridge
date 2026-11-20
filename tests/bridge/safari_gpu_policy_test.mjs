import assert from 'node:assert/strict';
import test from 'node:test';

import * as environment from '../../js/src/internal/environment.ts';
import { createDirectBridge } from './bridge_operation_queue_fixtures.mjs';

const MAC = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)';
const safari = (version) => `${MAC} Version/${version} Safari/605.1.15`;
const IPHONE_27 = 'Mozilla/5.0 (iPhone; CPU iPhone OS 27_0 like Mac OS X) AppleWebKit/605.1.15 '
  + '(KHTML, like Gecko) Version/27.0 Mobile/15E148 Safari/604.1';
const IPAD_27 = IPHONE_27.replace('iPhone; CPU iPhone OS', 'iPad; CPU OS');
const CHROME = `${MAC.replace('605.1.15', '537.36')} Chrome/157.0.0.0 Safari/537.36`;
const PADS = '<pad>'.repeat(48);
const SAFARI_LIKE_UAS = {
  'GNOME Web / WebKitGTK': safari('60.5').replace('Macintosh; Intel Mac OS X 10_15_7', 'X11; Linux x86_64'),
  'Linux with a Safari version token': safari('27.0').replace('Macintosh; Intel Mac OS X 10_15_7', 'X11; Linux x86_64'),
  'WebKitGTK with the Macintosh compatibility quirk': safari('60.5'),
  'Edge iOS': IPHONE_27.replace('Version/27.0', 'EdgiOS/157.0 Version/27.0'),
  'Brave iOS': `${IPHONE_27} Brave`,
  'Opera iOS': IPHONE_27.replace('Version/27.0', 'OPiOS/6.0 Version/27.0'),
};
const INVALID_VERSION_UAS = [
  safari('27garbage'),
  safari('27.0.'),
  safari('27..0'),
  `${MAC} App-Version/27.0 Safari/605.1.15`,
  `${MAC} App-Version/27.0 Version/26.2 Safari/605.1.15`,
  `${MAC} Version/27.0 Version/26.2 Safari/605.1.15`,
];

test('the Safari major version comes from the Version token', () => {
  assert.equal(environment.safariMajorVersion(safari('27.0')), 27);
  assert.equal(environment.safariMajorVersion(safari('26.2')), 26);
  assert.equal(environment.safariMajorVersion(safari('100.1')), 100);
  assert.equal(environment.safariMajorVersion(safari('27.0.2')), 27);
  assert.equal(environment.safariMajorVersion(IPHONE_27), 27);
  assert.equal(environment.safariMajorVersion(IPAD_27), 27);
  assert.equal(environment.safariMajorVersion(`${MAC} Safari/605.1.15`), null);
  assert.equal(environment.safariMajorVersion(CHROME), null);
  assert.equal(environment.safariMajorVersion(undefined), null);
});

test('the default layer cap is lifted starting at Safari 27', () => {
  assert.equal(environment.safariDefaultMaxGpuLayers(safari('27.0')), Infinity);
  assert.equal(environment.safariDefaultMaxGpuLayers(safari('28.1')), Infinity);
  assert.equal(environment.safariDefaultMaxGpuLayers(IPHONE_27), Infinity);
  assert.equal(environment.safariDefaultMaxGpuLayers(safari('26.4')), 1);
  assert.equal(environment.safariDefaultMaxGpuLayers(`${MAC} Safari/605.1.15`), 1);
});

// Mirror the float inputs, double-precision accumulation and float output in
// normalize_embedding_inplace before the bridge reads the core's JSON.
function normalizeEmbedding(values) {
  const floats = values.map(Math.fround);
  const normSquared = floats.reduce((sum, value) => sum + value * value, 0);
  if (normSquared <= 0) return floats;
  const scale = 1 / Math.sqrt(normSquared);
  return floats.map((value) => Math.fround(value * scale));
}

async function load({
  userAgent,
  options = {},
  embeddingModel = false,
  legacyCore = false,
  completion = async () => 'Hello there.',
  nativeEmbedding,
  embeddingJson,
}) {
  const oldNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { userAgent } });
  try {
    const { bridge, core } = createDirectBridge();
    const runtime = bridge._runtime;
    const ccall = core.ccall.bind(core);
    const loadedLayers = [];
    const nativeEmbedCalls = [];
    let lastEmbeddingJson = '[]';
    core._llamadart_webgpu_embed_bytes_to_json = () => {};
    if (!legacyCore) {
      core._llamadart_webgpu_embedding_capabilities_json = () => {};
      core._llamadart_webgpu_embed_parts_to_json = () => {};
    }
    core.ccall = (name, type, types, args = []) => {
      if (name === 'llamadart_webgpu_probe') return 1;
      if (name === 'llamadart_webgpu_backends_json') return '["WebGPU","CPU"]';
      if (name === 'llamadart_webgpu_load_model') {
        loadedLayers.push(args[6]);
        return 0;
      }
      if (name === 'llamadart_webgpu_free_model' || name === 'llamadart_webgpu_shutdown') return 0;
      if (name === 'llamadart_webgpu_get_context_size') return 128;
      if (name === 'llamadart_webgpu_embedding_capabilities_json') {
        return JSON.stringify({ apiVersion: 1, text: embeddingModel, image: false, audio: false });
      }
      if (name === 'llamadart_webgpu_embed_bytes_to_json') {
        const [bytes, , normalize] = args;
        const prompt = new TextDecoder().decode(bytes);
        nativeEmbedCalls.push({ prompt, normalize });
        const firstPrompt = nativeEmbedCalls[0].prompt;
        const values = nativeEmbedding?.(prompt, loadedLayers.at(-1), firstPrompt)
          ?? (prompt === firstPrompt ? [3, 4] : [4, -3]);
        const embedding = normalize ? normalizeEmbedding(values) : values;
        lastEmbeddingJson = embeddingJson?.(embedding)
          ?? JSON.stringify(embedding.map((value) => Number.isFinite(value) ? Number(value.toPrecision(9)) : 0));
        return embedding.length;
      }
      if (name === 'llamadart_webgpu_last_embedding_json') return lastEmbeddingJson;
      return ccall(name, type, types, args);
    };
    runtime._modelBytes = 0;
    runtime._ensureCore = async () => core;
    runtime._coreSupportsPthreads = () => true;
    runtime._syncThreadPoolSizeHintFromCore = () => {};
    runtime._getCachedModelResponse = async () => new Response(new Uint8Array([1, 2, 3]));
    runtime._emitLogger = () => {};
    let completionCalls = 0;
    runtime.createCompletion = async (...args) => {
      completionCalls += 1;
      return completion(...args);
    };

    await bridge.loadModelFromUrl('https://models.test/model.gguf', {
      nCtx: 128, nThreads: 1, nGpuLayers: 99, forceRemoteFetchBackend: false, useCache: false, ...options,
    });
    const metadata = bridge.getModelMetadata();
    return {
      calls: { completion: completionCalls, embed: nativeEmbedCalls.length },
      nativeEmbedCalls,
      lastEmbeddingJson,
      loadedLayers,
      layers: metadata['llamadart.webgpu.n_gpu_layers'],
      notes: metadata['llamadart.webgpu.runtime_notes'].split(';').filter((note) => note.startsWith('safari_')),
    };
  } finally {
    Object.defineProperty(globalThis, 'navigator', oldNavigator);
  }
}

test('Safari 27 offloads every requested layer by default', async () => {
  for (const userAgent of [safari('27.0'), IPHONE_27, IPAD_27]) {
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

for (const [browser, userAgent] of Object.entries(SAFARI_LIKE_UAS)) {
  test(`${browser} retains the legacy cap and completion probe`, async () => {
    const result = await load({ userAgent, embeddingModel: true });
    assert.deepEqual(result.loadedLayers, [1]);
    assert.deepEqual(result.notes, ['safari_gpu_layers_capped:1', 'safari_gpu_probe_passed']);
    assert.deepEqual(result.calls, { completion: 2, embed: 0 });
  });
}

for (const userAgent of INVALID_VERSION_UAS) {
  const products = userAgent.slice(MAC.length + 1).replace(' Safari/605.1.15', '');
  test(`invalid Version products retain the legacy cap and completion probe (${products})`, async () => {
    const result = await load({ userAgent, embeddingModel: true });
    assert.deepEqual(result.loadedLayers, [1]);
    assert.deepEqual(result.notes, ['safari_gpu_layers_capped:1', 'safari_gpu_probe_passed']);
    assert.deepEqual(result.calls, { completion: 2, embed: 0 });
  });
}

test('browsers outside the legacy Safari detector retain unprobed loads', async () => {
  for (const userAgent of [
    CHROME,
    `${MAC} Edg/157.0 Version/27.0 Safari/605.1.15`,
    `${MAC} Firefox/160.0 Version/27.0 Safari/605.1.15`,
    `${MAC} Chromium/157.0 Version/27.0 Safari/605.1.15`,
    `${MAC} OPR/157.0 Version/27.0 Safari/605.1.15`,
    IPHONE_27.replace('Version/27.0', 'CriOS/157.0 Version/27.0'),
    IPHONE_27.replace('Version/27.0', 'FxiOS/160.0 Version/27.0'),
    IPHONE_27.replace('Version/27.0 ', '').replace(' Safari/604.1', ''),
  ]) {
    const result = await load({ userAgent, embeddingModel: true });
    assert.deepEqual(result.loadedLayers, [99], userAgent);
    assert.deepEqual(result.notes, [], userAgent);
    assert.deepEqual(result.calls, { completion: 0, embed: 0 }, userAgent);
  }
});

// EmbeddingGemma-2 has no language-model head: on the CPU and on WebGPU its
// completion is a run of `<pad>`, which the corruption heuristic rejects.
test('an embedding model is probed with an embedding and stays on the GPU', async () => {
  for (const norm of [0.995, 1, 1.005]) {
    const result = await load({
      userAgent: safari('27.0'),
      embeddingModel: true,
      completion: async () => PADS,
      embeddingJson: (values) => JSON.stringify(values.map((value) => value * norm)),
    });
    assert.deepEqual(result.loadedLayers, [99]);
    assert.equal(result.layers, '99');
    assert.deepEqual(result.notes, ['safari_gpu_probe_embedding', 'safari_gpu_probe_passed']);
    assert.equal(result.calls.completion, 0);
    assert.equal(result.calls.embed, 2);
  }
});

test('a degenerate or failing embedding probe walks the fallback ladder', async () => {
  const failures = [
    { nativeEmbedding: () => [0, 0] },
    { nativeEmbedding: () => [] },
    { nativeEmbedding: () => [Infinity, 3, 4] },
    { nativeEmbedding: () => { throw new Error('device lost'); } },
    { embeddingJson: (values) => JSON.stringify(values.map((value) => value * 0.98)) },
    { embeddingJson: (values) => JSON.stringify(values.map((value) => value * 1.02)) },
  ];
  for (const failure of failures) {
    const result = await load({ userAgent: safari('27.0'), embeddingModel: true, ...failure });
    assert.deepEqual(result.loadedLayers, [99, 1, 0]);
    assert.equal(result.layers, '0');
    assert.deepEqual(result.notes, ['safari_gpu_probe_embedding', 'safari_gpu_probe_failed', 'safari_fallback_cpu']);
  }
});

test('a constant native embedding still fails after normalization', async () => {
  const result = await load({
    userAgent: safari('27.0'),
    embeddingModel: true,
    completion: async () => PADS,
    nativeEmbedding: () => new Array(768).fill(123),
  });
  assert.ok(Math.abs(Math.hypot(...JSON.parse(result.lastEmbeddingJson)) - 1) < 0.01);
  assert.equal(new Set(result.nativeEmbedCalls.map(({ prompt }) => prompt)).size, 2);
  assert.ok(result.nativeEmbedCalls.every(({ normalize }) => normalize === 1));
  assert.deepEqual(result.loadedLayers, [99, 1, 0]);
  assert.deepEqual(result.notes, ['safari_gpu_probe_embedding', 'safari_gpu_probe_failed', 'safari_fallback_cpu']);
});

test('the embedding probe rejects non-finite values before public embedding sanitization', async () => {
  const result = await load({
    userAgent: safari('27.0'),
    embeddingModel: true,
    embeddingJson: (values) => `[${values.join(',')},1e999]`,
  });
  assert.deepEqual(result.loadedLayers, [99, 1, 0]);
});

test('the embedding probe applies the cosine cutoff within the unit norm tolerance', async () => {
  for (const cosine of [0.9985, 0.9995]) {
    const result = await load({
      userAgent: safari('27.0'),
      embeddingModel: true,
      nativeEmbedding: (prompt, _layers, firstPrompt) => (
        prompt === firstPrompt ? [1, 0] : [cosine, Math.sqrt(1 - cosine * cosine)]
      ),
      embeddingJson: (values) => JSON.stringify(values.map((value) => value * 0.995)),
    });
    assert.deepEqual(result.loadedLayers, cosine < 0.999 ? [99] : [99, 1, 0]);
  }
});

test('the embedding probe compares both prompts again after reducing layers', async () => {
  const result = await load({
    userAgent: safari('27.0'),
    embeddingModel: true,
    options: { safariProbePrompts: ['one custom generation prompt'] },
    nativeEmbedding: (prompt, layers, firstPrompt) => layers > 1 || prompt === firstPrompt ? [3, 4] : [4, -3],
  });
  assert.deepEqual(result.loadedLayers, [99, 1]);
  assert.equal(result.nativeEmbedCalls.length, 4);
  assert.equal(new Set(result.nativeEmbedCalls.map(({ prompt }) => prompt)).size, 2);
  assert.deepEqual(result.notes, ['safari_gpu_probe_embedding', 'safari_gpu_probe_failed', 'safari_gpu_layers_capped:1']);
});

test('a text model keeps the completion probe and its corruption check', async () => {
  const corrupted = await load({ userAgent: safari('27.0'), completion: async () => PADS });
  assert.deepEqual(corrupted.loadedLayers, [99, 1, 0]);
  assert.deepEqual(corrupted.notes, ['safari_gpu_probe_failed', 'safari_fallback_cpu']);
  assert.equal(corrupted.calls.embed, 0);
});

test('a core without embedding capabilities keeps the completion probe', async () => {
  const result = await load({ userAgent: safari('27.0'), legacyCore: true });
  assert.deepEqual(result.notes, ['safari_gpu_probe_passed']);
  assert.deepEqual(result.calls, { completion: 2, embed: 0 });
});
