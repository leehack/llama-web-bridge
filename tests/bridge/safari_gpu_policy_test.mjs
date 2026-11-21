import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test, { after } from 'node:test';
import { fileURLToPath } from 'node:url';

import * as environment from '../../js/src/internal/environment.ts';
import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';
import { createStubCore } from './bridge_operation_queue_fixtures.mjs';
import { readNativeCoreSource } from './native_core_source.mjs';

const MAC = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)';
const safari = (version) => `${MAC} Version/${version} Safari/605.1.15`;
const IPHONE_26 = 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 '
  + '(KHTML, like Gecko) Version/26.5 Mobile/15E148 Safari/604.1';
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

let normalizationDir;
after(() => {
  if (normalizationDir) rmSync(normalizationDir, { recursive: true, force: true });
});

function normalizeEmbedding(values) {
  if (!normalizationDir) {
    const source = readNativeCoreSource();
    const start = source.indexOf('void normalize_embedding_inplace(');
    assert.ok(start >= 0);
    const normalization = source.slice(start, source.indexOf('\n}\n', start) + 2);
    const coreIncludes = fileURLToPath(new URL('../../src/', import.meta.url));
    normalizationDir = mkdtempSync(path.join(tmpdir(), 'lane245-normalization-'));
    const cpp = path.join(normalizationDir, 'normalize.cpp');
    writeFileSync(cpp, `
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "llama_webgpu_embedding_json.h"
${normalization}
int main(int argc, char **argv) {
  std::vector<float> values;
  for (int i = 1; i < argc; ++i) values.push_back(std::strtof(argv[i], nullptr));
  normalize_embedding_inplace(values);
  std::cout << llamadart_webgpu_detail::serialize_embedding_json(values);
}
`);
    execFileSync(process.env.CXX || 'c++', ['-std=c++17', '-I', coreIncludes, cpp, '-o', path.join(normalizationDir, 'normalize')]);
  }
  return JSON.parse(execFileSync(path.join(normalizationDir, 'normalize'), values.map(String), { encoding: 'utf8' }));
}

async function load({
  userAgent,
  config = { nGpuLayers: 99 },
  options = {},
  reloadOptions = [],
  embeddingModel = false,
  legacyCore = false,
  completion = async () => 'Hello there.',
  nativeEmbedding,
  embeddingJson,
  embeddingDimension,
  promptTokens,
}) {
  const oldNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { userAgent } });
  try {
    const bridge = new LlamaWebGpuBridge({ ...config, disableWorker: true });
    const core = createStubCore();
    const runtime = bridge._runtime;
    runtime._core = core;
    const ccall = core.ccall.bind(core);
    const loadedLayers = [];
    const nativeEmbedCalls = [];
    let effectiveContext;
    let effectiveMicrobatch;
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
        effectiveContext = args[1];
        effectiveMicrobatch = Math.min(effectiveContext, args[4] || 2048, args[5] || 512);
        return 0;
      }
      if (name === 'llamadart_webgpu_free_model' || name === 'llamadart_webgpu_shutdown') return 0;
      if (name === 'llamadart_webgpu_get_context_size') return effectiveContext;
      if (name === 'llamadart_webgpu_embedding_capabilities_json') {
        return JSON.stringify({ apiVersion: 1, text: embeddingModel, image: false, audio: false });
      }
      if (name === 'llamadart_webgpu_embed_bytes_to_json') {
        const [bytes, , normalize] = args;
        const prompt = new TextDecoder().decode(bytes);
        nativeEmbedCalls.push({ prompt, normalize });
        if (promptTokens?.(prompt) > effectiveMicrobatch) return -12;
        const firstPrompt = nativeEmbedCalls[0].prompt;
        const values = nativeEmbedding?.(prompt, loadedLayers.at(-1), firstPrompt)
          ?? (prompt === firstPrompt ? [3, 4] : [4, -3]);
        const embedding = normalize ? normalizeEmbedding(values) : values;
        lastEmbeddingJson = embeddingJson?.(embedding) ?? JSON.stringify(embedding);
        return embeddingDimension ?? embedding.length;
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

    const metadataLayers = await loadModels(bridge, [options, ...reloadOptions]);
    const metadata = bridge.getModelMetadata();
    return {
      calls: { completion: completionCalls, embed: nativeEmbedCalls.length },
      nativeEmbedCalls,
      lastEmbeddingJson,
      loadedLayers,
      metadataLayers,
      gpuActive: bridge.isGpuActive(),
      layers: metadata['llamadart.webgpu.n_gpu_layers'],
      notes: metadata['llamadart.webgpu.runtime_notes'].split(';').filter((note) => note.startsWith('safari_')),
    };
  } finally {
    Object.defineProperty(globalThis, 'navigator', oldNavigator);
  }
}

async function loadModels(bridge, options) {
  const metadataLayers = [];
  for (const [index, loadOptions] of options.entries()) {
    await bridge.loadModelFromUrl(`https://models.test/model-${index}.gguf`, {
      nCtx: 128, nThreads: 1, forceRemoteFetchBackend: false, useCache: false, ...loadOptions,
    });
    metadataLayers.push(bridge.getModelMetadata()['llamadart.webgpu.n_gpu_layers']);
  }
  return metadataLayers;
}

const RELOAD_OPTIONS = [
  {},
  { safariMaxGpuLayers: Infinity },
  {},
  { nGpuLayers: -7 },
  { safariMaxGpuLayers: Infinity },
  { nGpuLayers: 0 },
  { safariMaxGpuLayers: Infinity },
];

test('direct Safari applies each cap to the requested count across model loads', async (t) => {
  for (const [config, originalLayers] of [[{}, -1], [{ nGpuLayers: -5 }, -5], [{ nGpuLayers: 42 }, 42]]) {
    await t.test(`original request ${originalLayers}`, async () => {
      const result = await load({ userAgent: IPHONE_26, config, reloadOptions: RELOAD_OPTIONS.slice(1) });
      const expectedLayers = [1, originalLayers, 1, 1, -7, 0, 0];
      assert.deepEqual(result.loadedLayers, expectedLayers);
      assert.deepEqual(result.metadataLayers, expectedLayers.map(String));
    });
  }
});

test('the Safari worker applies each cap to the requested count across model loads', async (t) => {
  const core = createStubCore();
  const loadedLayers = [];
  const ccall = core.ccall.bind(core);
  t.mock.method(core, 'ccall', (name, type, types, args = []) => {
    if (name === 'llamadart_webgpu_probe') return 1;
    if (name === 'llamadart_webgpu_load_model') {
      loadedLayers.push(args[6]);
      return 0;
    }
    if (name === 'llamadart_webgpu_free_model') return 0;
    return ccall(name, type, types, args);
  });
  const prototype = LlamaWebGpuBridgeRuntime.prototype;
  t.mock.method(prototype, '_ensureCore', async function () {
    this._core = core;
    return core;
  });
  t.mock.method(prototype, '_getCachedModelResponse', async () => new Response(Uint8Array.of(1, 2, 3)));
  t.mock.method(prototype, 'createCompletion', async () => 'Hello there.');

  const { port1, port2 } = new MessageChannel();
  const oldSelf = globalThis.self;
  const oldWorker = globalThis.Worker;
  const host = { postMessage: (message) => port2.postMessage(message) };
  let bridge;
  try {
    globalThis.self = host;
    installBridgeWorkerHost();
    port2.onmessage = (event) => host.onmessage(event);
    globalThis.Worker = class {
      constructor() { port1.onmessage = (event) => this.onmessage(event); }
      postMessage(message, transfers) { port1.postMessage(message, transfers); }
      terminate() { port1.close(); port2.close(); }
    };
    bridge = new LlamaWebGpuBridge({ userAgent: IPHONE_26, nGpuLayers: -5 });
    const metadataLayers = await loadModels(bridge, RELOAD_OPTIONS);
    assert.equal(bridge.getModelMetadata()['llamadart.webgpu.execution'], 'worker');
    assert.deepEqual(loadedLayers, [1, -5, 1, 1, -7, 0, 0]);
    assert.deepEqual(metadataLayers, loadedLayers.map(String));
  } finally {
    try {
      await bridge?.dispose();
    } finally {
      port1.close();
      port2.close();
      globalThis.self = oldSelf;
      globalThis.Worker = oldWorker;
    }
  }
});

test('Safari 27 offloads every requested layer by default', async () => {
  for (const userAgent of [safari('27.0'), IPHONE_27, IPAD_27]) {
    const result = await load({ userAgent });
    assert.deepEqual(result.loadedLayers, [99]);
    assert.equal(result.layers, '99');
    assert.deepEqual(result.notes, ['safari_gpu_probe_passed']);
    assert.equal(result.calls.completion, 2);
  }
});

test('Safari below 27 applies its layer cap before loading and probing', async (t) => {
  const requests = [
    { name: 'macOS Safari, omitted', userAgent: safari('26.2'), config: {}, capped: true },
    { name: 'iPhone Safari, omitted', config: {}, capped: true },
    { name: 'unknown Safari, omitted', userAgent: `${MAC} Safari/605.1.15`, config: {}, capped: true },
    { name: 'configured -2', config: { nGpuLayers: -2 }, capped: true },
    ...[-1, -5, 0, 1, 99].map((nGpuLayers) => ({
      name: `requested ${nGpuLayers}`,
      options: { nGpuLayers },
      layers: nGpuLayers === 0 ? 0 : 1,
      capped: nGpuLayers !== 0 && nGpuLayers !== 1,
    })),
  ];
  for (const { name, userAgent = IPHONE_26, config, options, layers = 1, capped } of requests) {
    await t.test(name, async () => {
      const result = await load({ userAgent, config, options });
      assert.deepEqual(result.loadedLayers, [layers]);
      assert.equal(result.layers, String(layers));
      assert.equal(result.gpuActive, layers > 0);
      assert.deepEqual(result.notes, [
        ...(capped ? ['safari_gpu_layers_capped:1'] : []),
        ...(layers > 0 ? ['safari_gpu_probe_passed'] : []),
      ]);
      assert.deepEqual(result.calls, { completion: layers > 0 ? 2 : 0, embed: 0 });
    });
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

test('Safari applies explicit caps to full-offload requests', async (t) => {
  for (const [cap, layers] of [[4.5, 4], [0, 1], [-1, 1], [-Infinity, 1], [NaN, 1]]) {
    await t.test(`cap ${cap}`, async () => {
      const result = await load({ userAgent: IPHONE_26, config: {}, options: { safariMaxGpuLayers: cap } });
      assert.deepEqual(result.loadedLayers, [layers]);
      assert.deepEqual(result.notes, [`safari_gpu_layers_capped:${layers}`, 'safari_gpu_probe_passed']);
    });
  }
  const result = await load({ userAgent: safari('27.0'), options: { nGpuLayers: -5, safariMaxGpuLayers: 4 } });
  assert.deepEqual(result.loadedLayers, [4]);
  assert.deepEqual(result.notes, ['safari_gpu_layers_capped:4', 'safari_gpu_probe_passed']);
});

test('Safari below 27 accepts Infinity to skip the GPU layer cap', async (t) => {
  for (const [options, layers] of [[{}, -1], [{ nGpuLayers: -5 }, -5], [{ nGpuLayers: 99 }, 99]]) {
    await t.test(`native count ${layers}`, async () => {
      const result = await load({ userAgent: IPHONE_26, config: {}, options: { ...options, safariMaxGpuLayers: Infinity } });
      assert.deepEqual(result.loadedLayers, [layers]);
      assert.equal(result.notes.some((note) => note.startsWith('safari_gpu_layers_capped:')), false);
    });
  }
});

test('Safari 27 and non-WebKit browsers pass through uncapped counts', async () => {
  for (const userAgent of [safari('27.0'), IPHONE_27, IPAD_27, CHROME]) {
    for (const [options, layers] of [[{}, -1], [{ nGpuLayers: -5 }, -5], [{ nGpuLayers: 99 }, 99]]) {
      const result = await load({ userAgent, config: {}, options });
      assert.deepEqual(result.loadedLayers, [layers], userAgent);
    }
  }
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

  test(`${browser} passes through full-offload requests`, async () => {
    for (const [options, layers] of [[{}, -1], [{ nGpuLayers: -5 }, -5]]) {
      const result = await load({ userAgent, config: {}, options });
      assert.deepEqual(result.loadedLayers, [layers]);
      assert.deepEqual(result.notes, []);
      assert.deepEqual(result.calls, { completion: 0, embed: 0 });
    }
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

test('small contexts use shorter embedding probes and keep GPU layers', async () => {
  for (const nCtx of [8, 10, 12]) {
    const tokenCount = (prompt) => {
      if (prompt === 'The Eiffel Tower is in Paris.') return 11;
      if (prompt === 'WebGPU runs matrix math on the GPU.') return 13;
      return 3;
    };
    const result = await load({
      userAgent: safari('27.0'),
      embeddingModel: true,
      options: { nCtx, nBatch: 32, nUbatch: 32 },
      promptTokens: tokenCount,
      nativeEmbedding: (prompt) => prompt === 'cat' ? [3, 4] : [4, -3],
    });
    assert.deepEqual(result.loadedLayers, [99], `nCtx=${nCtx}`);
    assert.deepEqual(result.notes, ['safari_gpu_probe_embedding', 'safari_gpu_probe_passed']);
    assert.deepEqual(result.nativeEmbedCalls.slice(-2).map(({ prompt }) => prompt), ['cat', 'dog']);
  }
});

test('an embedding probe that cannot fit is skipped without dropping GPU layers', async () => {
  const result = await load({
    userAgent: safari('27.0'),
    embeddingModel: true,
    options: { nCtx: 2, nBatch: 32, nUbatch: 32 },
    promptTokens: () => 3,
  });
  assert.deepEqual(result.loadedLayers, [99]);
  assert.deepEqual(result.notes, ['safari_gpu_probe_embedding', 'safari_gpu_probe_skipped_input_too_long']);
});

test('an oversized second prompt does not hide corruption in the first embedding', async () => {
  const result = await load({
    userAgent: safari('27.0'),
    embeddingModel: true,
    options: { nCtx: 12, nBatch: 32, nUbatch: 32 },
    promptTokens: (prompt) => prompt === 'WebGPU runs matrix math on the GPU.' ? 13 : 3,
    nativeEmbedding: (prompt, _layers, firstPrompt) => {
      if (prompt === firstPrompt) return [4, 0];
      return prompt === 'cat' ? [3, 4] : [4, -3];
    },
  });
  assert.deepEqual(result.loadedLayers, [99, 1, 0]);
});

test('the embedding probe rejects opposite single-axis vectors after native normalization', async () => {
  const result = await load({
    userAgent: safari('27.0'),
    embeddingModel: true,
    nativeEmbedding: (prompt, _layers, firstPrompt) => [prompt === firstPrompt ? 4 : -4, 1e-8],
  });
  assert.ok(Math.abs(Math.hypot(...JSON.parse(result.lastEmbeddingJson)) - 1) < 0.01);
  assert.deepEqual(result.loadedLayers, [99, 1, 0]);
});

test('the embedding probe rejects vectors shorter than the native reported dimension', async () => {
  const result = await load({
    userAgent: safari('27.0'),
    embeddingModel: true,
    embeddingDimension: 768,
  });
  assert.deepEqual(result.loadedLayers, [99, 1, 0]);
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
    embeddingJson: (values) => `[${values[0]},1e999]`,
  });
  assert.deepEqual(result.loadedLayers, [99, 1, 0]);
});

test('the embedding probe applies the cosine cutoff within the unit norm tolerance', async () => {
  for (const cosine of [0.9985, 0.9995]) {
    const result = await load({
      userAgent: safari('27.0'),
      embeddingModel: true,
      nativeEmbedding: (prompt, _layers, firstPrompt) => (
        prompt === firstPrompt
          ? [1, 1]
          : [cosine + Math.sqrt(1 - cosine * cosine), cosine - Math.sqrt(1 - cosine * cosine)]
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
