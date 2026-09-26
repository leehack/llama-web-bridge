#!/usr/bin/env node
// Browser smoke for speculative decoding.

import fsp from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import {
  copyBridgeArtifacts,
  copyMemory64Artifacts,
  ensure,
  env,
  expandHome,
  expectedModes,
  isDirectory,
  modeRuntimeFailures,
  parseSmokeArgs,
  pyJson,
  resolvePath,
  resolvePinnedFile,
  runMain,
  runPlaywright,
  runPollingPlaywright,
  withServer,
  withTempDir,
  writeStdout,
} from './browser_smoke_support.mjs';

const DESCRIPTION = `Browser smoke for speculative decoding.

For each group, loads a checksum-pinned target GGUF in the direct and worker
runtimes of the wasm32 and wasm64 cores, runs a greedy completion without
speculative decoding, then the same completion with each strategy. Every
speculative output must equal the baseline text and token count, report
speculative usage, and draft tokens where the group expects them. Rejected
requests must leave the runtime usable.`;

export const MEMORY_MODES = Object.freeze(['wasm32', 'wasm64']);
export const RUNTIME_MODES = Object.freeze(['direct', 'worker']);
export const DEFAULT_MODELS_DIR = '~/.cache/llama-web-bridge/speculative-smoke-models';

const NGRAM_PROMPT = 'Count: one, two, three, four, five. Again: one, two, three, four, five. Again:';
const DRAFT_PROMPT = 'Explain in a few sentences why the sky is blue.\n\nAnswer:';
const N_PREDICT = 48;
// Small n-gram sizes: llama.cpp's defaults (12/48, and a 24-token match for
// ngram-mod) draft nothing on a prompt this short.
const NGRAM_SIZES = Object.freeze({ ngramSizeN: 3, ngramSizeM: 8 });
const NGRAM_MOD = Object.freeze({ ngramMatch: 3, ngramTokenMin: 1, ngramTokenMax: 16 });

const hf = (repo, revision, file) => `https://huggingface.co/${repo}/resolve/${revision}/${file}`;

// Pinned inputs. `url` is null for drafts converted locally from safetensors,
// which must be passed through --models-dir.
export const FILES = Object.freeze({
  smollm2_360m: {
    path: 'smollm2/SmolLM2-360M-Instruct-Q8_0.gguf',
    url: hf('bartowski/SmolLM2-360M-Instruct-GGUF', '7be6f65f1db715fe5dc5a4634c0d459b4eed42ec', 'SmolLM2-360M-Instruct-Q8_0.gguf'),
    sha256: 'c004ac34cce45e03e4452bca472535bc371fcb0d07ae560bba2c57ce4a284b7b',
  },
  smollm2_135m: {
    path: 'smollm2/SmolLM2-135M-Instruct-Q8_0.gguf',
    url: hf('bartowski/SmolLM2-135M-Instruct-GGUF', '09816acd5d99df7be770d85ea30822623dab342c', 'SmolLM2-135M-Instruct-Q8_0.gguf'),
    sha256: '5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83',
  },
  smollm2_360m_cache: {
    path: 'ngram-cache/smollm2-360m-static.lcs',
    url: null,
    sha256: 'e07dbe3c572b1220c0d1712970d4aac30577bd5d5a8067ca1c3da6f56184efea',
  },
  qwen3_0_6b: {
    path: 'qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf',
    url: hf('ggml-org/Qwen3-0.6B-GGUF', 'b5f37287796e5be0ea3dab2e7430873fb3f73e49', 'Qwen3-0.6B-Q8_0.gguf'),
    sha256: '361cc68159042c36ebff7715dc5a2e4612153e88f3e9c9c234820849d6dc9e1d',
  },
  eagle3_qwen3_0_6b: {
    path: 'eagle3/SGLang-EAGLE3-Qwen3-0.6B-SpecForge-F16.gguf',
    url: null,
    sha256: '1aa5c21b7386b5f9d9503c34607d2d59ef2cd062d9d4ec92573a478d7d881cda',
  },
  qwen3_5_0_8b: {
    path: 'qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf',
    url: hf('unsloth/Qwen3.5-0.8B-GGUF', '6ab461498e2023f6e3c1baea90a8f0fe38ab64d0', 'Qwen3.5-0.8B-Q4_K_M.gguf'),
    sha256: 'bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517',
  },
  qwen3_5_0_8b_mtp: {
    path: 'mtp/Qwen3.5-0.8B-Q4_K_M.gguf',
    url: hf('unsloth/Qwen3.5-0.8B-MTP-GGUF', 'cf8a611f6ed2c2060046219a19f12cd3d5ecd67c', 'Qwen3.5-0.8B-Q4_K_M.gguf'),
    sha256: 'ac7c9d7a1b3e3695bb3bd50f8ceaa97f9c93e99ccc3d3d1a620301b6dd6d3d86',
  },
  dflash_qwen3_5_0_8b: {
    path: 'dflash/Qwen3.5-0.8B-DFlash-Moonlight556-F16.gguf',
    url: null,
    sha256: 'b345ff5ace804a11fa8175b57c4b3b23f03059d70f26970225b735fe041d236f',
  },
  dspark_qwen3_5_0_8b: {
    path: 'dspark/Qwen3.5-0.8B-DSpark.gguf',
    url: hf('r3lax/Qwen3.5-0.8B-DSpark', 'ab91a3ebcdf7bcfaeec8b04e479065ede4132437', 'Qwen3.5-0.8B-DSpark.gguf'),
    sha256: '2316efef71399b6a404846fbbe4efd0f0d10e52517202a14454c11392c6c7f82',
  },
});

// A group loads `target` once per mode. Each run names the strategies, the
// draft model it needs, and what its usage must show.
export const GROUPS = Object.freeze({
  // The CI group: the state-persistence smoke model (tiny, random weights),
  // with itself as the draft model so every draft token is accepted.
  tiny: {
    target: 'smoke_model',
    runs: [
      { prompt: 'ngram', strategies: ['ngram-simple'], options: NGRAM_SIZES, expectDrafts: true },
      { prompt: 'ngram', strategies: ['ngram-map-k'], options: NGRAM_SIZES, expectDrafts: true },
      { prompt: 'ngram', strategies: ['ngram-map-k4v'], options: NGRAM_SIZES, expectDrafts: true },
      // llama-server drafts nothing here either.
      { prompt: 'ngram', strategies: ['ngram-mod'], options: NGRAM_MOD },
      { prompt: 'ngram', strategies: ['ngram-cache'], cache: 'generated' },
      { prompt: 'draft', strategies: ['draft-simple'], draft: 'smoke_model', expectDrafts: true, expectAccepted: true },
      {
        prompt: 'draft',
        strategies: ['ngram-mod', 'draft-simple'],
        options: NGRAM_MOD,
        draft: 'smoke_model',
        expectDrafts: true,
        expectAccepted: true,
      },
    ],
    rejections: [
      { label: 'malformed n-gram cache', strategies: ['ngram-cache'], cache: 'malformed', error: 'n-gram cache is malformed' },
      { label: 'draft-mtp without MTP layers', strategies: ['draft-mtp'], error: "draft-mtp needs the model's MTP head" },
    ],
  },
  smollm2: {
    target: 'smollm2_360m',
    runs: [
      { prompt: 'ngram', strategies: ['ngram-simple'], options: NGRAM_SIZES, expectDrafts: true, expectAccepted: true },
      { prompt: 'ngram', strategies: ['ngram-map-k'], options: NGRAM_SIZES, expectDrafts: true, expectAccepted: true },
      { prompt: 'ngram', strategies: ['ngram-map-k4v'], options: NGRAM_SIZES, expectDrafts: true, expectAccepted: true },
      { prompt: 'ngram', strategies: ['ngram-mod'], options: NGRAM_MOD, expectDrafts: true, expectAccepted: true },
      { prompt: 'ngram', strategies: ['ngram-cache'], cache: 'smollm2_360m_cache', expectDrafts: true, expectAccepted: true },
      { prompt: 'draft', strategies: ['draft-simple'], draft: 'smollm2_135m', expectDrafts: true, expectAccepted: true },
      {
        prompt: 'draft',
        strategies: ['ngram-mod', 'draft-simple'],
        options: NGRAM_MOD,
        draft: 'smollm2_135m',
        expectDrafts: true,
        expectAccepted: true,
      },
    ],
    rejections: [
      {
        label: 'a DSpark draft for a wider target',
        draft: 'dspark_qwen3_5_0_8b',
        error: "reads target hidden states of size 1024, but the loaded model's hidden size is 960",
      },
    ],
  },
  eagle3: {
    target: 'qwen3_0_6b',
    runs: [
      { prompt: 'draft', strategies: ['draft-eagle3'], draft: 'eagle3_qwen3_0_6b', expectDrafts: true, expectAccepted: true },
    ],
    rejections: [
      { label: 'draft-simple with an EAGLE3 draft', strategies: ['draft-simple'], error: 'draft-simple needs a standalone language model' },
    ],
  },
  mtp: {
    target: 'qwen3_5_0_8b_mtp',
    loadOptions: { loadMtp: true, speculativeRollbackTokenMax: 16 },
    runs: [
      { prompt: 'draft', strategies: ['draft-mtp'], expectDrafts: true, expectAccepted: true },
    ],
    rejections: [],
  },
  // A recurrent target with rollback snapshots runs the block-diffusion drafts.
  block: {
    target: 'qwen3_5_0_8b',
    loadOptions: { speculativeRollbackTokenMax: 16 },
    runs: [
      { prompt: 'draft', strategies: ['draft-dflash'], draft: 'dflash_qwen3_5_0_8b', expectDrafts: true, expectAccepted: true },
      {
        prompt: 'draft',
        strategies: ['draft-dspark'],
        options: { draftTokenMax: 7 },
        draft: 'dspark_qwen3_5_0_8b',
        expectDrafts: true,
        expectAccepted: true,
      },
    ],
    rejections: [
      { label: 'draft-dflash with a DSpark draft', strategies: ['draft-dflash'], error: 'draft-dflash needs a dflash draft model' },
    ],
  },
  // A recurrent target without rollback snapshots: n-gram drafts are verified
  // from a checkpoint and the accepted tokens replayed.
  recurrent: {
    target: 'qwen3_5_0_8b',
    runs: [
      { prompt: 'ngram', strategies: ['ngram-simple'], options: NGRAM_SIZES, expectDrafts: true, expectAccepted: true, expectReplay: true },
    ],
    rejections: [
      { label: 'draft-mtp without loadMtp', strategies: ['draft-mtp'], error: "draft-mtp needs the model's MTP head" },
    ],
  },
});

// The files a group serves, by FILES key or 'smoke_model'.
export function groupFiles(group) {
  const keys = new Set([group.target]);
  for (const run of group.runs) {
    if (run.draft) keys.add(run.draft);
    if (run.cache && FILES[run.cache]) keys.add(run.cache);
  }
  for (const rejection of group.rejections) {
    if (rejection.draft) keys.add(rejection.draft);
  }
  return [...keys];
}

export function renderHarness(groups, nCtx, memoryModes, runtimeModes, gpuLayers = 0) {
  const config = JSON.stringify({
    groups,
    prompts: { ngram: NGRAM_PROMPT, draft: DRAFT_PROMPT },
    nPredict: N_PREDICT,
    nCtx,
    gpuLayers,
    memoryModes,
    runtimeModes,
  });
  return `
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge speculative decoding smoke</title>
<pre id="result">pending</pre>
<script type="module">
(async () => {
  const result = document.getElementById('result');
  const finish = (payload) => {
    result.textContent = JSON.stringify(payload);
    window.__smokeResult = payload;
  };
  const errorText = (error) => String(error && error.message ? error.message : error);
  try {
    if (!window.crossOriginIsolated) {
      throw new Error('test page is not cross-origin isolated');
    }
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    const config = ${config};

    // llama-lookup-create's static cache: every token counted after the two
    // tokens before it, in common_ngram_cache_save's layout.
    const staticCacheBytes = (tokens) => {
      const counts = new Map();
      for (let i = 2; i < tokens.length; i += 1) {
        const key = tokens[i - 2] + ',' + tokens[i - 1];
        const next = counts.get(key) || new Map();
        next.set(tokens[i], (next.get(tokens[i]) || 0) + 1);
        counts.set(key, next);
      }
      const words = [];
      for (const [key, next] of counts) {
        const [a, b] = key.split(',').map(Number);
        words.push(a, b, -1, -1, next.size);
        for (const [token, count] of next) words.push(token, count);
      }
      return new Int32Array(words);
    };

    const greedy = { nPredict: config.nPredict, temp: 0, topK: 1, topP: 1, penalty: 1, seed: 42 };
    const complete = async (bridge, prompt, speculativeDecoding) => {
      let usage = null;
      const text = await bridge.createCompletion(prompt, {
        ...greedy,
        ...(speculativeDecoding ? { speculativeDecoding } : {}),
        onUsage: (value) => { usage = value; },
      });
      return { text, usage };
    };

    const runGroup = async (group, memoryMode, runtimeMode) => {
      const mode = memoryMode + ' ' + runtimeMode;
      const useMemory64 = memoryMode === 'wasm64';
      const bridge = new LlamaWebGpuBridge({
        disableWorker: runtimeMode === 'direct',
        preferMemory64: useMemory64,
        coreModuleUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.js' : undefined,
        wasmUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.wasm' : undefined,
      });
      const failures = [];
      const check = (condition, message) => {
        if (!condition) failures.push(message);
      };
      const runs = [];
      const rejections = [];
      try {
        await bridge.loadModelFromUrl('/' + group.target, {
          nCtx: config.nCtx,
          nThreads: 2,
          nGpuLayers: config.gpuLayers,
          useCache: false,
          forceRemoteFetchBackend: false,
          ...(group.loadOptions || {}),
        });
        const baselines = {};
        let loadedDraft = null;
        for (const run of group.runs) {
          const label = run.strategies.join('+') + (run.cache ? ' (' + run.cache + ' cache)' : '');
          const prompt = config.prompts[run.prompt];
          baselines[run.prompt] ??= await complete(bridge, prompt, null);
          const baseline = baselines[run.prompt];
          if (run.draft && run.draft !== loadedDraft) {
            let progressEvents = 0;
            const info = await bridge.loadDraftModel('/' + run.draft, {
              useCache: false,
              progressCallback: () => { progressEvents += 1; },
            });
            check(typeof info.architecture === 'string' && info.architecture.length > 0, label + ': draft has no architecture');
            check(progressEvents > 0, label + ': draft load reported no progress');
            loadedDraft = run.draft;
          }
          const capabilities = await bridge.getCompletionCapabilities();
          for (const strategy of run.strategies) {
            check(capabilities.speculativeDecoding[strategy] === true, label + ': ' + strategy + ' is not reported as supported');
          }
          const speculativeDecoding = { strategies: run.strategies, ...(run.options || {}) };
          if (run.cache === 'generated') {
            speculativeDecoding.ngramCacheStatic = staticCacheBytes(await bridge.tokenize(prompt + prompt));
          } else if (run.cache) {
            speculativeDecoding.ngramCacheStatic = '/' + run.cache;
          }
          const speculative = await complete(bridge, prompt, speculativeDecoding);
          const counts = speculative.usage && speculative.usage.speculative;
          check(speculative.text === baseline.text, label + ': output differs from the baseline: '
            + JSON.stringify(speculative.text) + ' vs ' + JSON.stringify(baseline.text));
          check(
            speculative.usage && baseline.usage
              && speculative.usage.completionTokens === baseline.usage.completionTokens,
            label + ': completion token counts differ',
          );
          check(!('speculative' in (baseline.usage || {})), label + ': the baseline reports speculative usage');
          check(counts && counts.draftAttempts > 0, label + ': no speculative usage');
          if (counts) {
            check(counts.acceptedDraftTokens <= counts.draftTokens, label + ': more accepted than drafted');
            if (run.expectDrafts) check(counts.draftTokens > 0, label + ': drafted nothing');
            if (run.expectAccepted) check(counts.acceptedDraftTokens > 0, label + ': accepted nothing');
            if (run.expectReplay) check(counts.replayTokens > 0, label + ': replayed nothing');
            else check(counts.replayTokens === 0, label + ': replayed tokens without a checkpoint');
          }
          runs.push({ label, text: speculative.text, baseline: baseline.text, usage: counts || null });
        }
        for (const rejection of group.rejections) {
          const speculativeDecoding = { strategies: rejection.strategies };
          if (rejection.cache === 'malformed') {
            speculativeDecoding.ngramCacheStatic = new Int32Array([1, 2, -1, -1, 5]);
          }
          let error = null;
          try {
            if (rejection.draft) {
              await bridge.loadDraftModel('/' + rejection.draft, { useCache: false });
            } else {
              await complete(bridge, config.prompts.draft, speculativeDecoding);
            }
          } catch (caught) {
            error = errorText(caught);
          }
          check(error !== null && error.includes(rejection.error),
            rejection.label + ': expected ' + JSON.stringify(rejection.error) + ', got ' + JSON.stringify(error));
          rejections.push({ label: rejection.label, error });
        }
        const after = await complete(bridge, config.prompts.draft, null);
        check(after.text === (baselines.draft || after).text, 'the runtime changed its output after the speculative runs');
        const metadata = bridge.getModelMetadata();
        const backend = bridge.getBackendName();
        check(config.gpuLayers === 0 || backend.includes('WebGPU'), 'expected a WebGPU backend, got ' + backend);
        return {
          mode,
          group: group.name,
          failures,
          runs,
          rejections,
          backend,
          execution: metadata['llamadart.webgpu.execution'] || null,
          coreVariant: metadata['llamadart.webgpu.core_variant'] || null,
          workerFallbackReason: metadata['llamadart.webgpu.worker_fallback_reason'] || null,
        };
      } catch (error) {
        return { mode, group: group.name, failures: [...failures, 'threw: ' + errorText(error)], runs, rejections };
      } finally {
        await bridge.dispose();
      }
    };

    const modeResults = [];
    for (const group of config.groups) {
      for (const memoryMode of config.memoryModes) {
        for (const runtimeMode of config.runtimeModes) {
          modeResults.push(await runGroup(group, memoryMode, runtimeMode));
        }
      }
    }
    finish({
      ok: true,
      modeResults,
      globalWorkerFallbackReason: globalThis.__llamadartBridgeWorkerFallbackReason || null,
    });
  } catch (error) {
    finish({ ok: false, error: String(error && error.stack ? error.stack : error) });
  }
})();
</script>
`;
}

export function validatePayload(payload, groupNames, memoryModes, runtimeModes) {
  if (payload.ok !== true) return [`harness failed: ${String(payload.error)}`];
  const failures = [];
  const modeResults = Array.isArray(payload.modeResults) ? payload.modeResults : [];
  const expected = groupNames.flatMap((name) => expectedModes(memoryModes, runtimeModes).map((mode) => `${name} ${mode}`));
  const actual = modeResults.map((entry) => `${entry?.group} ${entry?.mode}`);
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    return [`mode results missing: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`];
  }
  for (const entry of modeResults) {
    const label = `${entry.group} ${entry.mode}`;
    if (!Array.isArray(entry.failures)) {
      failures.push(`${label}: failures missing`);
      continue;
    }
    failures.push(...entry.failures.map((failure) => `${label}: ${failure}`));
    if (entry.failures.length === 0) {
      failures.push(...modeRuntimeFailures(entry.mode, entry).map((failure) => `${entry.group} ${failure}`));
    }
  }
  if (payload.globalWorkerFallbackReason != null) {
    failures.push(`worker fell back to the main thread: ${payload.globalWorkerFallbackReason}`);
  }
  return failures;
}

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'speculative_browser_smoke.mjs',
    description: DESCRIPTION,
    options: [
      {
        flag: '--dist-dir',
        type: 'path',
        default: () => env.path('BRIDGE_DIST_DIR', 'dist'),
        help: 'Directory containing built bridge artifacts.',
      },
      {
        flag: '--group',
        type: 'string',
        choices: ['all', ...Object.keys(GROUPS)],
        default: () => 'tiny',
        help: 'Model group to run; all runs every group.',
      },
      {
        flag: '--models-dir',
        type: 'path',
        default: () => env.path('LLAMA_WEBGPU_SPECULATIVE_MODELS_DIR', DEFAULT_MODELS_DIR),
        help: 'Directory holding the pinned files at their FILES paths; missing downloadable files are fetched into it.',
      },
      {
        flag: '--model-url',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_URL'),
        help: 'GGUF URL of the tiny group; defaults to the state-persistence smoke model.',
      },
      {
        flag: '--model-path',
        type: 'path',
        default: () => env.optionalPath('LLAMA_WEBGPU_SMOKE_MODEL_PATH'),
        help: 'Local GGUF path of the tiny group.',
      },
      {
        flag: '--model-sha256',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_SHA256'),
        help: 'Expected SHA-256 of the tiny group model.',
      },
      {
        flag: '--model-cache-dir',
        type: 'path',
        default: () => env.path('LLAMA_WEBGPU_SMOKE_MODEL_CACHE', '~/.cache/llama-web-bridge/state-smoke-models'),
        help: 'Cache directory used with --model-url.',
      },
      {
        flag: '--memory-mode',
        type: 'string',
        choices: ['all', ...MEMORY_MODES],
        default: () => 'all',
        help: 'Core memory mode to run; wasm64 needs the mem64 artifacts.',
      },
      {
        flag: '--runtime-mode',
        type: 'string',
        choices: ['all', ...RUNTIME_MODES],
        default: () => 'all',
        help: 'Runtime mode to run.',
      },
      {
        flag: '--gpu-layers',
        type: 'int',
        default: () => 0,
        help: 'nGpuLayers for the target and draft models; above 0 launches Chromium with WebGPU enabled.',
      },
      {
        flag: '--n-ctx',
        type: 'int',
        default: () => 2048,
        help: 'Context size passed to loadModelFromUrl.',
      },
      {
        flag: '--timeout-ms',
        type: 'int',
        default: () => env.int('LLAMA_WEBGPU_SPECULATIVE_TIMEOUT_MS', '1800000'),
        help: 'Browser operation timeout in milliseconds.',
      },
      {
        flag: '--artifacts-dir',
        type: 'path',
        default: () => env.optionalPath('LLAMA_WEBGPU_SPECULATIVE_ARTIFACTS_DIR'),
        help: 'Directory for JSON/console/screenshot diagnostics.',
      },
    ],
  });
}

async function resolveFile(key, args) {
  if (key === 'smoke_model') {
    return resolvePinnedFile({
      filePath: args.modelPath,
      url: args.modelUrl,
      expectedSha256: args.modelSha256,
      cacheDir: args.modelCacheDir,
      label: 'tiny group model',
    });
  }
  const file = FILES[key];
  const modelsDir = resolvePath(expandHome(args.modelsDir));
  const local = path.join(modelsDir, file.path);
  const exists = await fsp.stat(local).then((stat) => stat.isFile(), () => false);
  return resolvePinnedFile({
    filePath: exists || file.url === null ? local : null,
    url: file.url,
    expectedSha256: file.sha256,
    cacheDir: path.dirname(local),
    label: key,
  });
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const distDir = resolvePath(args.distDir);
  ensure(isDirectory(distDir), `dist directory does not exist: ${distDir}`);
  const artifactsDir = args.artifactsDir !== null ? resolvePath(args.artifactsDir) : null;
  const groupNames = args.group === 'all' ? Object.keys(GROUPS) : [args.group];
  const memoryModes = args.memoryMode === 'all' ? MEMORY_MODES : [args.memoryMode];
  const runtimeModes = args.runtimeMode === 'all' ? RUNTIME_MODES : [args.runtimeMode];

  const payload = await withTempDir('llama-web-bridge-speculative-smoke-', async (webRoot) => {
    await copyBridgeArtifacts(distDir, webRoot);
    if (memoryModes.includes('wasm64')) await copyMemory64Artifacts(distDir, webRoot);
    const staged = new Set();
    for (const name of groupNames) {
      for (const key of groupFiles(GROUPS[name])) {
        if (staged.has(key)) continue;
        await fsp.copyFile(await resolveFile(key, args), path.join(webRoot, key));
        staged.add(key);
      }
    }
    const groups = groupNames.map((name) => ({ name, ...GROUPS[name] }));
    await fsp.writeFile(
      path.join(webRoot, 'index.html'),
      renderHarness(groups, args.nCtx, memoryModes, runtimeModes, args.gpuLayers),
      'utf8',
    );
    return withServer(
      webRoot,
      (url) => (args.gpuLayers > 0
        ? runPollingPlaywright(url, args.timeoutMs, artifactsDir, {
          artifactPrefix: 'speculative-smoke',
          stageMarker: '[speculative-smoke]',
        })
        : runPlaywright(url, args.timeoutMs, artifactsDir, 'speculative-smoke')),
    );
  });

  writeStdout(`${pyJson(payload, { indent: 2, sortKeys: true })}\n`);
  const failures = validatePayload(payload, groupNames, memoryModes, runtimeModes);
  for (const failure of failures) process.stderr.write(`speculative browser smoke: ${failure}\n`);
  return failures.length ? 1 : 0;
}

if (import.meta.main) {
  await runMain('speculative', main);
}
