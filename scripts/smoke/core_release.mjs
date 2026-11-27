#!/usr/bin/env node

import assert from 'node:assert/strict';
import fsp from 'node:fs/promises';
import path from 'node:path';

import {
  DEFAULT_MODEL_CACHE, copyBridgeArtifacts, copyMemory64Artifacts, copySingleThreadArtifacts,
  env, parseSmokeArgs, resolvePinnedModel, runMain, runPollingPlaywright,
  stageFile, webGpuLaunchArgs, withServer, withTempDir,
} from './support.mjs';

const CASES = ['dispose', 'abort'];
const DECISION_CASE = 'decision-abort';
const MODEL = 'release-smoke.gguf';
const DECISION_MODEL = 'decision-encoder.gguf';
const DECISION_HEAD = 'decision-head.safetensors';

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'core_release.mjs',
    description: 'Check that Chromium can collect a disposed or retired direct-mode core.',
    options: [
      { flag: '--dist-dir', type: 'path', default: () => env.path('BRIDGE_DIST_DIR', 'dist') },
      { flag: '--threading', type: 'string', choices: ['default', 'single'], default: () => 'default' },
      { flag: '--memory-mode', type: 'string', choices: ['wasm32', 'wasm64'], default: () => 'wasm32' },
      { flag: '--case', type: 'string', choices: [...CASES, DECISION_CASE] },
      { flag: '--model-path', type: 'path', default: () => env.optionalPath('LLAMA_WEBGPU_SMOKE_MODEL_PATH') },
      { flag: '--model-url', type: 'string', default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_URL') },
      { flag: '--model-sha256', type: 'string', default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_SHA256') },
      { flag: '--model-cache-dir', type: 'path', default: () => env.path('LLAMA_WEBGPU_SMOKE_MODEL_CACHE', DEFAULT_MODEL_CACHE) },
      { flag: '--decision-model-path', type: 'path' },
      { flag: '--decision-head-path', type: 'path' },
      { flag: '--artifacts-dir', type: 'path' },
      { flag: '--timeout-ms', type: 'int', default: () => 90000 },
    ],
  });
}

// The page holds each core only weakly, so the bridge and the core's own
// runtime are all that can keep one reachable.
export function renderTrackedCore(coreName) {
  return `import createCore from '/${coreName}.js';
globalThis.__trackedCores ??= [];
export default async (options) => {
  const core = await createCore(options);
  globalThis.__trackedCores.push(new WeakRef(core));
  return core;
};
`;
}

async function runBrowserCase(settings) {
  const { LlamaWebGpuBridge } = await import('/llama_webgpu_bridge.js');
  const { scenario, modelUrl, decisionModelUrl, decisionHeadUrl, single, memory64, coreName } = settings;
  const assert = (condition, message) => {
    if (!condition) throw new Error(message);
  };
  const cores = () => globalThis.__trackedCores || [];
  const collected = async (ref) => {
    for (let attempt = 0; attempt < 50; attempt += 1) {
      await globalThis.gc({ type: 'major', execution: 'async' });
      // deref() keeps its target alive until this task ends.
      await new Promise((resolve) => setTimeout(resolve, 100));
      if (ref.deref() === undefined) return true;
    }
    return false;
  };
  let bridge;
  const result = { ok: false, scenario, crossOriginIsolated };
  try {
    assert(typeof globalThis.gc === 'function', 'Chromium was not started with --js-flags=--expose-gc');
    assert(crossOriginIsolated === !single, 'Unexpected page isolation');
    const config = {
      disableWorker: true,
      allowSingleThreadCore: single,
      preferMemory64: memory64,
      coreModuleUrl: '/tracked-core.js',
      wasmUrl: `/${coreName}.wasm`,
    };
    if (memory64) {
      config.coreModuleUrlMem64 = config.coreModuleUrl;
      config.wasmUrlMem64 = config.wasmUrl;
    }
    const nThreads = single ? 1 : 4;
    const options = { nCtx: 128, nThreads, nThreadsBatch: nThreads, nGpuLayers: 0, useCache: false };
    const load = (url = modelUrl) => bridge.loadModelFromUrl(url, options);
    const complete = () => bridge.createCompletion('hello', { nPredict: 8, temp: 0, seed: 1 });

    window.__smokeStage = scenario + ':first-core';
    bridge = new LlamaWebGpuBridge(config);
    let decisionHandle;
    if (decisionModelUrl) {
      await bridge.loadModelFromUrl(decisionModelUrl, { ...options, nCtx: 512, nBatch: 512, nUbatch: 512 });
      const head = await bridge.loadDecisionHead(decisionHeadUrl);
      decisionHandle = head.handle;
      assert(Number.isInteger(decisionHandle) && decisionHandle > 0, 'The first core did not load a decision head');
    } else {
      await load();
      assert((await complete()).length > 0, 'The first core generated nothing');
    }
    assert(cores().length === 1, `Expected one core, found ${cores().length}`);
    assert(bridge.getModelMetadata()['llamadart.webgpu.core_variant'] === (memory64 ? 'wasm64' : 'wasm32'),
      'The selected Wasm core did not match the requested memory mode');

    window.__smokeStage = scenario + ':replace';
    if (scenario === 'dispose') {
      await bridge.dispose();
      bridge = new LlamaWebGpuBridge(config);
    } else {
      let aborted = null;
      cores()[0].deref().ccall('llamadart_webgpu_test_abort_next_model_load', null, [], []);
      await load().catch((error) => { aborted = error; });
      assert(aborted && /^Aborted\(/.test(aborted.message), `The injected native abort did not retire the core: ${aborted}`);
      assert(bridge._coreRetirements === 1, 'Expected exactly one retirement');
    }
    await load();
    assert(cores().length === 2, `Expected a replacement core, found ${cores().length}`);

    window.__smokeStage = scenario + ':collect';
    result.firstCoreCollected = await collected(cores()[0]);
    result.liveCoreCollected = cores()[1].deref() === undefined;
    assert(!result.liveCoreCollected, 'The core in use was collected, so the probe proves nothing');
    assert(result.firstCoreCollected,
      `The ${scenario === 'dispose' ? 'disposed' : 'retired'} core is still reachable after garbage collection`);
    if (decisionHandle !== undefined) {
      const stale = await bridge.runDecision(decisionHandle, [{ tokens: [1] }]).then(() => null, (error) => error);
      assert(stale?.code === 'head-lost', `Expected head-lost for the stale decision handle: ${stale}`);
      assert(await bridge.freeDecisionHead(decisionHandle) === undefined, 'Freeing a stale decision handle must be a no-op');
    }
    assert((await complete()).length > 0, 'The replacement core generated nothing');
    result.ok = true;
  } catch (error) {
    result.error = String(error?.stack || error);
  } finally {
    try {
      await bridge?.dispose();
    } catch (error) {
      result.ok = false;
      result.disposeError = String(error?.stack || error);
    }
    window.__smokeResult = result;
  }
}

export function renderHarness(settings) {
  return `<!doctype html>
<meta charset="utf-8">
<title>Core release</title>
<body>
<script type="module">
const runBrowserCase = ${runBrowserCase.toString()};
await runBrowserCase(${JSON.stringify(settings)});
</script>
`;
}

export function launchArgs(platform = process.platform, { single = false } = {}) {
  return [...webGpuLaunchArgs(platform, { sharedArrayBuffer: !single }), '--js-flags=--expose-gc'];
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const decision = args.case === DECISION_CASE;
  if (decision) {
    assert(args.decisionModelPath && args.decisionHeadPath,
      '--case decision-abort requires --decision-model-path and --decision-head-path');
  }
  const scenarios = args.case === null ? CASES : [args.case];
  const modelPath = await resolvePinnedModel(args);
  const single = args.threading === 'single';
  const memory64 = args.memoryMode === 'wasm64';
  const coreName = 'llama_webgpu_core' + (memory64 ? '_mem64' : '') + (single ? '_st' : '');
  const results = await withTempDir('core-release-smoke-', async (webRoot) => {
    await copyBridgeArtifacts(args.distDir, webRoot);
    if (single) await copySingleThreadArtifacts(args.distDir, webRoot, { memory64 });
    else if (memory64) await copyMemory64Artifacts(args.distDir, webRoot);
    await stageFile(modelPath, path.join(webRoot, MODEL));
    if (decision) {
      await stageFile(args.decisionModelPath, path.join(webRoot, DECISION_MODEL));
      await stageFile(args.decisionHeadPath, path.join(webRoot, DECISION_HEAD));
    }
    await fsp.writeFile(path.join(webRoot, 'tracked-core.js'), renderTrackedCore(coreName));
    return withServer(webRoot, async (url) => {
      const collected = {};
      for (const scenario of scenarios) {
        const key = [args.threading, args.memoryMode, scenario].join('-');
        await fsp.writeFile(path.join(webRoot, 'index.html'), renderHarness({
          scenario, single, memory64, coreName, modelUrl: '/' + MODEL,
          decisionModelUrl: decision ? '/' + DECISION_MODEL : null,
          decisionHeadUrl: decision ? '/' + DECISION_HEAD : null,
        }));
        const payload = await runPollingPlaywright(url, args.timeoutMs, args.artifactsDir, {
          artifactPrefix: key,
          launchArgs: launchArgs(process.platform, { single }),
          stageMarker: 'CORE_RELEASE_SMOKE',
          stagePrefix: key,
        });
        assert.equal(payload.ok, true, `${key}: ${payload.error || payload.disposeError}`);
        collected[key] = payload.firstCoreCollected;
      }
      return collected;
    }, { isolated: !single });
  });
  console.log(JSON.stringify({ ok: true, results }, null, 2));
}

if (import.meta.main) await runMain('core release', main);
