import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test, { after } from 'node:test';
import { fileURLToPath } from 'node:url';

import * as environment from '../../js/src/internal/environment.ts';
import { createDirectBridge } from './bridge_operation_queue_fixtures.mjs';
import { readNativeCoreSource } from './native_core_source.mjs';
import { COMPLETION, PROMPT, withSafariBridge } from './safari_gpu_policy_worker_fixture.mjs';

const MAC = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)';
const safari = (version) => `${MAC} Version/${version} Safari/605.1.15`;
const IPHONE_27 = 'Mozilla/5.0 (iPhone; CPU iPhone OS 27_0 like Mac OS X) AppleWebKit/605.1.15 '
  + '(KHTML, like Gecko) Version/27.0 Mobile/15E148 Safari/604.1';
const IPAD_27 = IPHONE_27.replace('iPhone; CPU iPhone OS', 'iPad; CPU OS');
const IPHONE_26 = IPHONE_27.replace('27_0', '18_7').replace('Version/27.0', 'Version/26.5');
const CHROME = `${MAC.replace('605.1.15', '537.36')} Chrome/157.0.0.0 Safari/537.36`;
const FIREFOX = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:157.0) Gecko/20100101 Firefox/157.0';
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

test('the Safari cap compares full-offload requests against the per-load maximum', () => {
  const cases = [
    ...[0, -0].flatMap((zero) => [undefined, 4, Infinity].map((max) => [zero, 1, max, 0, false])),
    ...[-1, -5].flatMap((layers) => [
      [layers, 1, undefined, 1, true],
      [layers, 1, Infinity, layers, false],
      [layers, 1, 4.5, 4, true],
      [layers, 1, 0, 1, true],
      [layers, 1, -1, 1, true],
      [layers, 1, NaN, 1, true],
      [layers, 1, -Infinity, 1, true],
      [layers, Infinity, NaN, layers, false],
      [layers, Infinity, -Infinity, layers, false],
    ]),
    [99, 1, undefined, 1, true],
    [1, 1, undefined, 1, false],
    [99, 1, 99, 99, false],
    [99, 1, Infinity, 99, false],
    [1.5, 1, undefined, 1, true],
    [0.5, 1, undefined, 0.5, false],
  ];
  for (const [requested, defaultMax, override, effective, capped] of cases) {
    assert.deepEqual(environment.capSafariGpuLayers(requested, defaultMax, override), { effective, capped });
  }
});

const loadModel = (bridge, options = {}, file = 'model') => bridge.loadModelFromUrl(`https://models.test/${file}.gguf`, {
  useCache: false, forceRemoteFetchBackend: false, ...options,
});
const metadata = (bridge, key) => bridge.getModelMetadata()[`llamadart.webgpu.${key}`];
const safariNotes = (bridge) => metadata(bridge, 'runtime_notes').split(';').filter((note) => note.startsWith('safari_'));
const withoutProbe = { safariGpuProbe: false };

test('Apple Safari before 27 uses the one-layer cap for omitted and explicit offload requests', async (t) => {
  for (const [name, userAgent, nGpuLayers, options, layers, capped] of [
    ['macOS default', safari('26.2'), undefined, {}, 1, true],
    ['unknown version', `${MAC} Safari/605.1.15`, undefined, {}, 1, true],
    ['configured negative', safari('26.2'), -2, {}, 1, true],
    ['explicit negative', safari('26.2'), undefined, { nGpuLayers: -5 }, 1, true],
    ['one layer', safari('26.2'), undefined, { nGpuLayers: 1 }, 1, false],
    ['99 layers', safari('26.2'), undefined, { nGpuLayers: 99 }, 1, true],
    ...[4.5, 0, -1, -Infinity, NaN].map((max) => [
      `cap ${max}`, safari('26.2'), undefined, { safariMaxGpuLayers: max }, max === 4.5 ? 4 : 1, true,
    ]),
    ['Safari 27 with a finite cap', safari('27.0'), undefined, { nGpuLayers: -5, safariMaxGpuLayers: 4 }, 4, true],
  ]) {
    await t.test(name, () => withSafariBridge({ userAgent, nGpuLayers, disableWorker: true }, async ({ bridge, trace, calls }) => {
      await loadModel(bridge, options);
      assert.deepEqual(trace, [layers]);
      assert.equal(metadata(bridge, 'n_gpu_layers'), String(layers));
      assert.equal(bridge.isGpuActive(), true);
      assert.deepEqual(safariNotes(bridge), [
        ...(capped ? [`safari_gpu_layers_capped:${layers}`] : []), 'safari_gpu_probe_passed',
      ]);
      assert.equal(calls.completion, 2);
    }));
  }
});

test('fresh iPhone Safari 26 loads retain the device-qualified 1/1/1/0 native counts', async (t) => {
  for (const [name, disableWorker, options, layers, capped] of [
    ['omitted direct', true, {}, 1, true],
    ['omitted worker', false, {}, 1, true],
    ['negative direct', true, { nGpuLayers: -1 }, 1, true],
    ['zero direct', true, { nGpuLayers: 0 }, 0, false],
  ]) {
    await t.test(name, () => withSafariBridge({ userAgent: IPHONE_26, disableWorker }, async ({ bridge, trace, calls }) => {
      await loadModel(bridge, options);
      assert.deepEqual(trace, [layers]);
      assert.equal(metadata(bridge, 'n_gpu_layers'), String(layers));
      assert.equal(metadata(bridge, 'execution'), disableWorker ? 'main-thread' : 'worker');
      assert.equal(bridge.isGpuActive(), layers > 0);
      assert.deepEqual(safariNotes(bridge), capped ? ['safari_gpu_layers_capped:1', 'safari_gpu_probe_passed'] : []);
      if (disableWorker) assert.equal(calls.completion, layers > 0 ? 2 : 0);
    }));
  }
});

test('the Apple Safari cap fits the signed Wasm load parameter', async (t) => {
  const maxInt32 = 2147483647;
  for (const disableWorker of [true, false]) {
    const execution = disableWorker ? 'main-thread' : 'worker';
    await t.test(execution, () => withSafariBridge({ userAgent: IPHONE_26, disableWorker }, async ({ bridge, trace }) => {
      for (const [nGpuLayers, requested] of [[undefined, -1], [-5, -5]]) {
        for (const safariMaxGpuLayers of [2 ** 32, Number.MAX_SAFE_INTEGER, maxInt32 + 1]) {
          await loadModel(bridge, { ...withoutProbe, nGpuLayers, safariMaxGpuLayers });
          assert.equal(trace.at(-1), requested, `native layers for cap ${safariMaxGpuLayers}`);
          assert.equal(metadata(bridge, 'n_gpu_layers'), String(requested));
          assert.deepEqual(safariNotes(bridge), []);
          assert.equal(bridge.getBackendName(), 'WebGPU, CPU');
          assert.equal(bridge.isGpuActive(), true);
        }
      }
      await loadModel(bridge, { ...withoutProbe, safariMaxGpuLayers: maxInt32 });
      assert.equal(trace.at(-1), maxInt32);
      assert.equal(metadata(bridge, 'n_gpu_layers'), String(maxInt32));
      assert.deepEqual(safariNotes(bridge), [`safari_gpu_layers_capped:${maxInt32}`]);
      assert.equal(bridge.getBackendName(), 'WebGPU, CPU');
      assert.equal(bridge.isGpuActive(), true);
      assert.equal(metadata(bridge, 'execution'), execution);
    }));
  }
});

test('uncapped default and explicit GPU requests report offload and run the Safari probe', async (t) => {
  for (const [name, userAgent, options, probes] of [
    ['Safari 26 with cap lifted', safari('26.2'), { safariMaxGpuLayers: Infinity }, true],
    ['Safari 27', safari('27.0'), {}, true],
    ['iPhone Safari 27', IPHONE_27, {}, true],
    ['iPad Safari 27', IPAD_27, {}, true],
    ['Chrome', CHROME, {}, false],
    ['Firefox', FIREFOX, {}, false],
  ]) {
    for (const disableWorker of [true, false]) {
      await t.test(`${name}, ${disableWorker ? 'direct' : 'worker'}`, () => withSafariBridge({
        userAgent, disableWorker,
      }, async ({ bridge, trace }) => {
        for (const nGpuLayers of [undefined, -1, 1, 0]) {
          await loadModel(bridge, { ...options, nGpuLayers });
          assert.equal(bridge.isGpuActive(), nGpuLayers !== 0);
          assert.deepEqual(safariNotes(bridge), probes && nGpuLayers !== 0 ? ['safari_gpu_probe_passed'] : []);
        }
        assert.deepEqual(trace, [-1, -1, 1, 0]);
      }));
    }
  }
});

test('uncapped Safari probes full offload, then tries one layer before CPU', async () => {
  for (const disableWorker of [true, false]) {
    await withSafariBridge({ userAgent: safari('27.0'), disableWorker }, async ({ bridge, trace }) => {
      await loadModel(bridge);
      assert.deepEqual(trace, [-1, 1, 0]);
      assert.equal(bridge.isGpuActive(), false);
      assert.ok(safariNotes(bridge).includes('safari_fallback_cpu'));
    }, { corruptProbe: true });
  }
});

test('full-offload requests report unavailable WebGPU in direct and worker modes', async () => {
  for (const disableWorker of [true, false]) {
    await withSafariBridge({ userAgent: CHROME, disableWorker }, async ({ bridge }) => {
      await loadModel(bridge);
      assert.equal(bridge.isGpuActive(), false);
      assert.match(metadata(bridge, 'runtime_notes'), /gpu_unavailable:/);
      await loadModel(bridge, { nGpuLayers: 0 });
      assert.doesNotMatch(metadata(bridge, 'runtime_notes'), /gpu_unavailable:/);
    }, { gpuAvailable: false });
  }
});

test('GPU layer requests must fit the signed Wasm load parameter', async () => {
  for (const disableWorker of [true, false]) {
    for (const nGpuLayers of [2 ** 32, Number.MAX_SAFE_INTEGER, -(2 ** 31) - 1, 0.5, Infinity, NaN]) {
      const config = { userAgent: CHROME, disableWorker };
      await assert.rejects(withSafariBridge({ ...config, nGpuLayers }, async () => {}), RangeError);
      await withSafariBridge(config, async ({ bridge, trace }) => {
        await loadModel(bridge, { nGpuLayers: 1 });
        await assert.rejects(loadModel(bridge, { nGpuLayers }), RangeError);
        assert.deepEqual(trace, [1]);
        assert.equal(bridge.isGpuActive(), true);
      });
    }
    await withSafariBridge({ userAgent: CHROME, disableWorker, nGpuLayers: -(2 ** 31) }, async ({ bridge, trace }) => {
      await loadModel(bridge);
      await loadModel(bridge, { nGpuLayers: 2 ** 31 - 1 });
      assert.deepEqual(trace, [-(2 ** 31), 2 ** 31 - 1]);
    });
  }
});

test('an uncapped load uses the layer request after native options are resolved', () => withSafariBridge({
  userAgent: CHROME, disableWorker: true,
}, async ({ bridge, trace }) => {
  const runtime = bridge._runtime;
  const resolveNativeLoadOptions = runtime._resolveNativeLoadOptions.bind(runtime);
  runtime._resolveNativeLoadOptions = (options) => {
    resolveNativeLoadOptions(options);
    runtime._nGpuLayers = 3;
  };
  await loadModel(bridge, { nGpuLayers: 2 });
  assert.deepEqual(trace, [3]);
  assert.equal(metadata(bridge, 'n_gpu_layers'), '3');
}));

test('Apple Safari computes a fresh cap without changing the inherited request', async (t) => {
  for (const disableWorker of [true, false]) {
    for (const nGpuLayers of [undefined, -5, 42]) {
      const execution = disableWorker ? 'main-thread' : 'worker';
      await t.test(`${execution}, configured ${nGpuLayers}`, () => withSafariBridge({
        userAgent: safari('26.2'), nGpuLayers, disableWorker,
      }, async ({ bridge, trace }) => {
        const options = [
          {}, { safariMaxGpuLayers: Infinity }, {}, { nGpuLayers: -7 },
          { safariMaxGpuLayers: Infinity }, { nGpuLayers: 0 }, { safariMaxGpuLayers: Infinity },
        ];
        const counts = [];
        for (const option of options) {
          await loadModel(bridge, { ...withoutProbe, ...option });
          counts.push(Number(metadata(bridge, 'n_gpu_layers')));
          assert.equal(metadata(bridge, 'execution'), execution);
        }
        const expected = [1, nGpuLayers ?? -1, 1, 1, -7, 0, 0];
        assert.deepEqual(trace, expected);
        assert.deepEqual(counts, expected);
      }));
    }
  }
});

test('Safari lookalikes keep their sticky positive caps and pass through full offload', async (t) => {
  for (const [name, userAgent] of [
    ...Object.entries(SAFARI_LIKE_UAS), ...INVALID_VERSION_UAS.map((ua) => [ua.slice(MAC.length + 1), ua]),
  ]) {
    for (const disableWorker of [true, false]) {
      const execution = disableWorker ? 'main-thread' : 'worker';
      await t.test(`${name}, ${execution}`, async () => {
        await withSafariBridge({ userAgent, nGpuLayers: 7, disableWorker }, async ({ bridge, trace }) => {
          for (const options of [
            {}, { safariMaxGpuLayers: 4 }, { nGpuLayers: 7, safariMaxGpuLayers: Infinity },
            { nGpuLayers: 7, safariMaxGpuLayers: 4 },
          ]) await loadModel(bridge, { ...withoutProbe, ...options });
          assert.deepEqual(trace, [1, 1, 1, 4]);
          assert.equal(metadata(bridge, 'execution'), execution);
        });
        await withSafariBridge({ userAgent, disableWorker }, async ({ bridge, trace, calls }) => {
          for (const options of [{}, { nGpuLayers: -5 }]) {
            await loadModel(bridge, options);
            assert.deepEqual(safariNotes(bridge), ['safari_gpu_probe_passed']);
          }
          assert.deepEqual(trace, [-1, -5]);
          if (disableWorker) assert.equal(calls.completion, 4);
          assert.equal(metadata(bridge, 'execution'), execution);
        });
      });
    }
  }
});

async function restartWorker(bridge, [worker]) {
  worker.postMessage({ type: 'trapTokenize' });
  await assert.rejects(bridge.tokenize('hi'), /unreachable/);
  assert.deepEqual(await bridge.tokenize('hi'), [104, 105]);
}

test('worker replay uses the bridge configuration and the remembered options', async (t) => {
  for (const [name, nGpuLayers, options, expected] of [
    ['default', undefined, [{}], [1, 1]],
    ['opt-out', undefined, [{ safariMaxGpuLayers: Infinity }], [-1, -1]],
    ['explicit layers and cap', undefined, [{ nGpuLayers: -5, safariMaxGpuLayers: 4 }], [4, 4]],
    ['CPU only', undefined, [{ nGpuLayers: 0 }], [0, 0]],
    ['omitted count on the second load', 42, [{ nGpuLayers: 7 }, { safariMaxGpuLayers: Infinity }], [1, 7, 42]],
  ]) {
    await t.test(name, () => withSafariBridge({ userAgent: safari('26.2'), nGpuLayers }, async ({ bridge, trace, workers }) => {
      for (const [index, option] of options.entries()) await loadModel(bridge, { ...withoutProbe, ...option }, `model-${index}`);
      await restartWorker(bridge, workers);
      assert.deepEqual(trace, expected);
      assert.equal(workers.length, 2);
      assert.equal(metadata(bridge, 'execution'), 'worker');
      assert.equal(metadata(bridge, 'n_gpu_layers'), String(expected.at(-1)));
    }));
  }
});

test('worker replay and main-thread fallback keep CPU after generation recovery', async (t) => {
  for (const execution of ['worker', 'main-thread']) {
    await t.test(execution, () => withSafariBridge({
      userAgent: safari('26.2'), nGpuLayers: 0, preferMemory64: true,
    }, async ({ bridge, trace, workers }) => {
      await loadModel(bridge, { ...withoutProbe, nGpuLayers: 7 }, 'a');
      await loadModel(bridge, withoutProbe, 'b');
      assert.equal(metadata(bridge, 'core_variant'), 'wasm64');
      assert.equal(await bridge.createCompletion(PROMPT, { nPredict: 5 }), COMPLETION);
      assert.equal(metadata(bridge, 'n_gpu_layers'), '0');
      assert.match(metadata(bridge, 'runtime_notes'), /generation_recovery_cpu_applied/);
      if (execution === 'worker') {
        await restartWorker(bridge, workers);
      } else {
        workers[0].crashOnResult = true;
        await bridge.getCompletionCapabilities();
      }
      assert.deepEqual(trace, [1, 1, 0, 0]);
      assert.equal(workers.length, execution === 'worker' ? 2 : 1);
      assert.equal(metadata(bridge, 'execution'), execution);
      assert.equal(metadata(bridge, 'n_gpu_layers'), '0');
    }, { failGpuDecodeAt: 1 }));
  }
});

test('completion recovers to CPU again after replay of an omitted Safari count', async (t) => {
  for (const execution of ['worker', 'main-thread']) {
    await t.test(execution, () => withSafariBridge({
      userAgent: safari('26.2'), preferMemory64: true,
    }, async ({ bridge, trace, workers }) => {
      await loadModel(bridge, withoutProbe);
      assert.equal(await bridge.createCompletion(PROMPT, { nPredict: 5 }), COMPLETION);
      assert.deepEqual(trace, [1, 0]);
      assert.equal(metadata(bridge, 'n_gpu_layers'), '0');
      if (execution === 'worker') {
        await restartWorker(bridge, workers);
      } else {
        workers[0].crashOnResult = true;
        await bridge.getCompletionCapabilities();
      }
      assert.equal(metadata(bridge, 'execution'), execution);
      assert.deepEqual(trace, [1, 0, 1]);
      assert.equal(await bridge.createCompletion(PROMPT, { nPredict: 5 }), COMPLETION);
      assert.deepEqual(trace, [1, 0, 1, 0]);
      assert.equal(metadata(bridge, 'n_gpu_layers'), '0');
      assert.match(metadata(bridge, 'runtime_notes'), /generation_recovery_cpu_applied/);
    }, { failGpuDecodeAt: 1 }));
  }
});

test('an omitted Safari count enables wasm64 CPU recovery and subsequent omitted loads inherit CPU', () => withSafariBridge({
  userAgent: safari('26.2'), disableWorker: true, preferMemory64: true,
}, async ({ bridge, trace }) => {
  await loadModel(bridge, withoutProbe, 'a');
  assert.equal(metadata(bridge, 'core_variant'), 'wasm64');
  assert.equal(await bridge.createCompletion(PROMPT, { nPredict: 5 }), COMPLETION);
  assert.equal(metadata(bridge, 'n_gpu_layers'), '0');
  assert.match(metadata(bridge, 'runtime_notes'), /generation_recovery_cpu_applied/);
  await loadModel(bridge, withoutProbe, 'b');
  assert.deepEqual(trace, [1, 0, 0]);
}, { failGpuDecodeAt: 1 }));

test('the Safari probe keeps wasm64 generation recovery outside Apple Safari JS staging', async (t) => {
  const appleSafari = safari('26.2');
  const lookalike = appleSafari.replace('Macintosh; Intel Mac OS X 10_15_7', 'X11; Linux x86_64');
  for (const [name, userAgent, remote] of [
    ['lookalike staged', lookalike, false],
    ['Apple Safari native fetch', appleSafari, true],
    ['lookalike native fetch', lookalike, true],
    ['Apple Safari standard staging', appleSafari, false],
  ]) {
    await t.test(name, () => withSafariBridge({
      userAgent, preferMemory64: true, jsBackedModelStaging: false,
    }, async ({ bridge, trace, loadSources }) => {
      await loadModel(bridge, {
        ...(remote ? { nGpuLayers: 1 } : { nGpuLayers: 8, safariMaxGpuLayers: 4 }),
        forceRemoteFetchBackend: remote,
        modelBytesHint: 3,
      });
      assert.deepEqual(trace, [remote ? 1 : 4, 0]);
      assert.deepEqual(loadSources, [remote ? 'native-fetch' : 'staged', 'staged']);
      assert.equal(metadata(bridge, 'execution'), 'worker');
      assert.equal(metadata(bridge, 'n_gpu_layers'), '0');
      assert.match(metadata(bridge, 'runtime_notes'), /generation_recovery_cpu_applied/);
      assert.ok(safariNotes(bridge).includes('safari_gpu_probe_passed'));
      assert.equal(await bridge.createCompletion(PROMPT, { nPredict: 5 }), COMPLETION);
      assert.equal(trace.length, 2);
    }, { failGpuDecodeAt: remote ? 1 : 2, failNativeFetchRetry: remote }));
  }
});

test('a failed probe of the omitted count stabilizes on CPU and subsequent loads inherit it', () => withSafariBridge({
  userAgent: safari('26.2'), disableWorker: true,
}, async ({ bridge, trace }) => {
  await loadModel(bridge);
  assert.deepEqual(trace, [1, 0]);
  assert.equal(metadata(bridge, 'n_gpu_layers'), '0');
  assert.equal(bridge.isGpuActive(), false);
  assert.equal(bridge.getBackendName(), 'WASM (Prototype bridge)');
  assert.deepEqual(safariNotes(bridge), ['safari_gpu_layers_capped:1', 'safari_gpu_probe_failed', 'safari_fallback_cpu']);
  await loadModel(bridge);
  assert.deepEqual(trace, [1, 0, 0]);
  assert.deepEqual(safariNotes(bridge), []);
}, { corruptProbe: true }));

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
  options = {},
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
    const { bridge, core } = createDirectBridge();
    const runtime = bridge._runtime;
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

test('a failed Safari probe tries one layer after capping full offload to four', async () => {
  let firstProbe = true;
  const result = await load({
    userAgent: safari('26.2'),
    options: { nGpuLayers: -5, safariMaxGpuLayers: 4 },
    completion: async () => {
      if (!firstProbe) return COMPLETION;
      firstProbe = false;
      return PADS;
    },
  });
  assert.deepEqual(result.loadedLayers, [4, 1]);
  assert.equal(result.layers, '1');
  assert.deepEqual(result.notes, ['safari_gpu_layers_capped:4', 'safari_gpu_probe_failed', 'safari_gpu_layers_capped:1']);
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
