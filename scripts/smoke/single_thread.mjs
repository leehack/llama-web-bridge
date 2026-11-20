#!/usr/bin/env node

import fsp from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import {
  DEFAULT_MODEL_CACHE,
  copyBridgeArtifacts,
  copySingleThreadArtifacts,
  ensure,
  isDirectory,
  parseSmokeArgs,
  resolvePath,
  resolvePinnedModel,
  runMain,
  runPollingPlaywright,
  stageFile,
  webGpuLaunchArgs,
  withServer,
  withTempDir,
  writeTextArtifact,
} from './support.mjs';

const ARTIFACT_PREFIX = 'single-thread-smoke';
const HEAP_CHANNEL = 'single-thread-smoke-heap';
const MODEL_FILENAME = 'single-thread-model.gguf';

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'single_thread.mjs',
    description: 'Local real-model acceptance smoke for a single-thread core on a non-isolated page.',
    options: [
      { flag: '--dist-dir', type: 'path', required: true },
      { flag: '--model-path', type: 'path' },
      { flag: '--model-url', type: 'string' },
      { flag: '--model-sha256', type: 'string', required: true },
      { flag: '--model-cache-dir', type: 'path', default: () => DEFAULT_MODEL_CACHE },
      { flag: '--memory-mode', type: 'string', choices: ['wasm32', 'wasm64'], default: () => 'wasm32' },
      { flag: '--runtime-mode', type: 'string', choices: ['direct', 'worker'], default: () => 'worker' },
      { flag: '--gpu-layers', type: 'int', default: () => 999 },
      { flag: '--js-backed-model-staging', type: 'flag', default: () => false },
      { flag: '--timeout-ms', type: 'int', default: () => 180000 },
      { flag: '--artifacts-dir', type: 'path' },
    ],
  });
}

// WebAssembly linear memory cannot shrink, so its size after inference is its peak.
function renderHeapProbe(coreName) {
  return `
import createCore from '/${coreName}.js';
export default async (options) => {
  let memory;
  const { instantiate, instantiateStreaming } = WebAssembly;
  const captureMemory = (load) => async (...args) => {
    const result = await load(...args);
    memory = Object.values((result.instance ?? result).exports)
      .find((value) => value instanceof WebAssembly.Memory);
    return result;
  };

  let core;
  try {
    WebAssembly.instantiate = captureMemory(instantiate);
    WebAssembly.instantiateStreaming = captureMemory(instantiateStreaming);
    core = await createCore(options);
  } finally {
    WebAssembly.instantiate = instantiate;
    WebAssembly.instantiateStreaming = instantiateStreaming;
  }
  if (!memory) throw new Error('single-thread core did not export its wasm memory');

  const channel = new BroadcastChannel(${JSON.stringify(HEAP_CHANNEL)});
  channel.onmessage = () => channel.postMessage(memory.buffer.byteLength);
  return core;
};
`;
}

function renderNativeFetchProbe(coreName) {
  return `
import createCore from '/${coreName}.js';
self.onmessage = async ({ data: modelUrl }) => {
  try {
    const core = await createCore();
    const rc = await core.ccall(
      'llamadart_webgpu_load_model_from_url', 'number',
      ['string', ...Array(20).fill('number')],
      [modelUrl, 64, 1, 1, 32, 32, 0, ...Array(14).fill(0)], { async: true },
    );
    const message = core.ccall('llamadart_webgpu_last_error', 'string', [], []);
    self.postMessage({ rc, message });
  } catch (error) {
    self.postMessage({ error: String(error) });
  }
};
`;
}

export function renderHarness({ memoryMode, runtimeMode, gpuLayers, jsBackedModelStaging }, modelBytes) {
  const worker = runtimeMode === 'worker';
  const coreOptions = {
    allowSingleThreadCore: true,
    allowAutoRemoteFetchBackend: true,
    disableWorker: !worker,
    jsBackedModelStaging,
    preferMemory64: memoryMode === 'wasm64',
    remoteFetchThresholdBytes: 1,
    coreModuleUrl: '/measure-core_st.js',
    wasmUrl: '/llama_webgpu_core_st.wasm',
    ...(memoryMode === 'wasm64' ? {
      coreModuleUrlMem64: '/measure-core_mem64_st.js',
      wasmUrlMem64: '/llama_webgpu_core_mem64_st.wasm',
    } : {}),
  };

  return `
<!doctype html>
<meta charset="utf-8">
<title>Single-thread GGUF acceptance</title>
<script type="module">
import * as bridgeModule from '/llama_webgpu_bridge.js';

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const options = { nCtx: 2048, nThreads: 4, nGpuLayers: ${gpuLayers}, useCache: false };
const modelUrl = '/${MODEL_FILENAME}';
let bridge;
let stagedBytes = 0;
let nativeFetchGuard;
try {
  assert(crossOriginIsolated === false, 'test page must be non-isolated');
  assert(typeof SharedArrayBuffer === 'undefined', 'test page must not expose SharedArrayBuffer');
  bridge = new bridgeModule.LlamaWebGpuBridge(${JSON.stringify(coreOptions)});

  if (${worker}) {
    window.__smokeStage = 'rejecting forced fetch';
    let rejected = false;
    try {
      await bridge.loadModelFromUrl(modelUrl, { ...options, forceRemoteFetchBackend: true });
    } catch (error) {
      if (error?.code !== 'SINGLE_THREAD_CORE_UNSUPPORTED') throw error;
      assert(error instanceof bridgeModule.SingleThreadCoreError && error.feature === 'remote_fetch_backend',
        'forced fetch did not preserve its typed error');
      rejected = true;
    }
    assert(rejected, 'forced fetch unexpectedly succeeded');

    window.__smokeStage = 'calling the native URL loader directly';
    const nativeWorker = new Worker('/native-fetch-probe.js', { type: 'module' });
    try {
      nativeFetchGuard = await new Promise((resolve, reject) => {
        nativeWorker.onmessage = ({ data }) => resolve(data);
        nativeWorker.onerror = (event) => reject(new Error(event.message));
        nativeWorker.postMessage(new URL(modelUrl, location.href).href);
      });
      assert(nativeFetchGuard.rc === -7
        && nativeFetchGuard.message === 'Fetch-backed model loading requires a core built with pthreads',
        'unexpected direct native fetch result: ' + JSON.stringify(nativeFetchGuard));
    } finally {
      nativeWorker.terminate();
    }
  }

  window.__smokeStage = 'staging model';
  await bridge.loadModelFromUrl(modelUrl, {
    ...options,
    progressCallback: ({ loaded }) => { stagedBytes = Math.max(stagedBytes, loaded); },
  });
  const verifyRuntime = (phase) => {
    const metadata = bridge.getModelMetadata();
    const notes = metadata['llamadart.webgpu.runtime_notes'].split(';');
    const gpuActive = bridge.isGpuActive();
    assert(gpuActive === (options.nGpuLayers > 0), 'GPU state did not match requested offload ' + phase);
    assert(metadata['llamadart.webgpu.core_variant'] === ${JSON.stringify(memoryMode)}, 'unexpected memory mode ' + phase);
    assert(metadata['llamadart.webgpu.execution'] === ${JSON.stringify(worker ? 'worker' : 'main-thread')},
      'unexpected execution mode ' + phase);
    assert(!notes.includes('generation_recovery_cpu_applied'), 'generation used CPU fallback');
    return { metadata, notes, gpuActive };
  };
  const { metadata, notes: loadNotes } = verifyRuntime('after loading');
  assert(metadata['llamadart.webgpu.n_threads'] === '1', 'expected one CPU thread');
  for (const note of ['core_pthreads:0', 'core_single_thread', 'threads_capped_no_coi']) {
    assert(loadNotes.includes(note), 'missing runtime note ' + note);
  }
  assert(loadNotes.includes('model_fetch_backend_requires_pthreads') === ${worker}, 'unexpected fetch fallback note');
  assert(metadata['llamadart.webgpu.model_source'] === 'network', 'model did not use streamed staging');
  assert(loadNotes.includes('model_staging_js_backed') === ${jsBackedModelStaging}, 'unexpected staging mode');
  assert(stagedBytes === ${modelBytes} && Number(metadata['llamadart.webgpu.model_bytes']) === stagedBytes,
    'staged byte count differs from the verified model size');

  window.__smokeStage = 'generating';
  let usage;
  await bridge.createCompletion('The capital of France is', {
    nPredict: 12,
    temp: 0,
    seed: 1,
    onUsage: (entry) => { usage = entry; },
  });
  assert(usage?.completionTokens > 0, 'model did not generate');
  const { notes, gpuActive } = verifyRuntime('after generation');
  const channel = new BroadcastChannel(${JSON.stringify(HEAP_CHANNEL)});
  const peakWasmHeapBytes = await new Promise((resolve) => {
    channel.onmessage = (event) => resolve(event.data);
    channel.postMessage('peak heap');
  });
  channel.close();
  assert(Number.isSafeInteger(peakWasmHeapBytes) && peakWasmHeapBytes > 0, 'missing wasm heap measurement');

  window.__smokeResult = {
    ok: true,
    browser: navigator.userAgent,
    core: ${JSON.stringify(memoryMode)},
    runtime: ${JSON.stringify(runtimeMode)},
    jsBackedModelStaging: ${jsBackedModelStaging},
    gpuLayers: options.nGpuLayers,
    gpuActive,
    backend: bridge.getBackendName(),
    crossOriginIsolated,
    sharedArrayBuffer: typeof SharedArrayBuffer,
    stagedBytes,
    peakWasmHeapBytes,
    usage,
    notes,
    nativeFetchGuard,
  };
} catch (error) {
  window.__smokeResult = {
    ok: false,
    stage: window.__smokeStage,
    stagedBytes,
    code: error?.code,
    error: String(error?.stack || error?.message || error),
    notes: bridge?.getModelMetadata()['llamadart.webgpu.runtime_notes']?.split(';'),
  };
} finally {
  await bridge?.dispose();
}
</script>
`;
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const distDir = resolvePath(args.distDir);
  const artifactsDir = args.artifactsDir === null ? null : resolvePath(args.artifactsDir);
  ensure(isDirectory(distDir), `dist directory does not exist: ${distDir}`);
  ensure(args.gpuLayers >= 0, '--gpu-layers must be non-negative');
  ensure(args.timeoutMs > 0, '--timeout-ms must be positive');
  const model = await resolvePinnedModel(args);
  const { size } = await fsp.stat(model);
  const memory64 = args.memoryMode === 'wasm64';
  const page = renderHarness(args, size);
  const payload = await withTempDir('single-thread-smoke-', async (webRoot) => {
    await copyBridgeArtifacts(distDir, webRoot);
    await copySingleThreadArtifacts(distDir, webRoot, { memory64 });
    await stageFile(model, path.join(webRoot, MODEL_FILENAME));
    await fsp.writeFile(path.join(webRoot, 'measure-core_st.js'), renderHeapProbe('llama_webgpu_core_st'));
    if (memory64) {
      await fsp.writeFile(path.join(webRoot, 'measure-core_mem64_st.js'), renderHeapProbe('llama_webgpu_core_mem64_st'));
    }
    if (args.runtimeMode === 'worker') {
      const coreName = memory64 ? 'llama_webgpu_core_mem64_st' : 'llama_webgpu_core_st';
      await fsp.writeFile(path.join(webRoot, 'native-fetch-probe.js'), renderNativeFetchProbe(coreName));
    }
    await fsp.writeFile(path.join(webRoot, 'index.html'), page);
    await writeTextArtifact(artifactsDir, 'index.html', page);
    return withServer(webRoot, (url) => runPollingPlaywright(url, args.timeoutMs, artifactsDir, {
      artifactPrefix: ARTIFACT_PREFIX,
      stagePrefix: ARTIFACT_PREFIX,
      launchArgs: webGpuLaunchArgs(process.platform, { sharedArrayBuffer: false }),
    }), { isolated: false });
  });
  process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
  return payload.ok ? 0 : 1;
}

if (import.meta.main) await runMain('single thread', main);
