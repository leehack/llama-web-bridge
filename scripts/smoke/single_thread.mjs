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
const CORE_PROBE_CHANNEL = 'single-thread-smoke-core';
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
      { flag: '--heap-staging-lifecycle', type: 'flag', default: () => false },
      { flag: '--heap-staging-reuse', type: 'flag', default: () => false },
      { flag: '--expect-heap-fallback', type: 'string', choices: ['unsupported', 'allocation'] },
      { flag: '--timeout-ms', type: 'int', default: () => 180000 },
      { flag: '--artifacts-dir', type: 'path' },
    ],
  });
}

export function parseGpuOffload(line) {
  const match = /^(?:load_tensors: )?offloaded (\d+)\/(\d+) layers to GPU$/.exec(String(line).trim());
  if (!match) return null;
  const offloadedLayers = Number(match[1]);
  const modelLayers = Number(match[2]);
  if (!Number.isSafeInteger(modelLayers) || modelLayers === 0 || offloadedLayers > modelLayers) return null;
  return { offloadedLayers, modelLayers };
}

export function verifyGpuOffload(requestedLayers, gpuOffload) {
  if (requestedLayers <= 0) return;
  if (!gpuOffload) throw new Error('missing native GPU layer offload count');
  const { offloadedLayers, modelLayers } = gpuOffload;
  const expectedLayers = Math.min(requestedLayers, modelLayers);
  if (offloadedLayers < expectedLayers) {
    throw new Error(`offloaded ${offloadedLayers}/${modelLayers} layers to GPU, expected at least ${expectedLayers}`);
  }
}

function renderCoreProbe(coreName) {
  return `
import createCore from '/${coreName}.js';
const parseGpuOffload = ${parseGpuOffload.toString()};
export default async (options) => {
  let memory;
  let gpuOffload = null;
  let jsStagedBytesAtLoad = 0;
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
    core = await createCore({
      ...options,
      printErr: (line) => {
        gpuOffload = parseGpuOffload(line) ?? gpuOffload;
        options.printErr(line);
      },
    });
  } finally {
    WebAssembly.instantiate = instantiate;
    WebAssembly.instantiateStreaming = instantiateStreaming;
  }
  if (!memory) throw new Error('single-thread core did not export its wasm memory');

  const channel = new BroadcastChannel(${JSON.stringify(CORE_PROBE_CHANNEL)});
  const id = crypto.randomUUID();
  channel.onmessage = ({ data }) => {
    if (data === 'arm load failure') {
      core._llamadart_webgpu_test_fail_next_model_load();
    }
    if (data === 'heap staging' || data === 'arm load failure') {
      channel.postMessage({
        id,
        bytes: core._llamadart_webgpu_js_model_fs_heap_resident_bytes(),
        heapBytes: memory.buffer.byteLength,
        jsStagedBytesAtLoad,
      });
      return;
    }
    channel.postMessage(data === 'gpu offload' ? gpuOffload : memory.buffer.byteLength);
  };
  const ccall = core.ccall.bind(core);
  core.ccall = (name, ...args) => {
    if (name === 'llamadart_webgpu_load_model') {
      jsStagedBytesAtLoad = core._llamadart_webgpu_js_model_fs_allocated_bytes?.();
    }
    if (name === 'llamadart_webgpu_shutdown') channel.close();
    return ccall(name, ...args);
  };
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

async function verifyHeapStagingLifecycle({
  bridge, createBridge, modelUrl, options, readCoreProbe, modelBytes, loadNotes, reuseOnly, assert,
}) {
  const hasHeapNote = (notes) => notes.includes('model_staging_heap_resident');
  assert(hasHeapNote(loadNotes), 'missing heap staging note');
  const expectedBytes = Math.ceil(modelBytes / 65536) * 65536;
  const first = await readCoreProbe('heap staging');
  const checkBytes = async (phase, bytes, coreId) => {
    const current = await readCoreProbe('heap staging');
    assert(current.bytes === bytes, phase + ': heap-resident bytes ' + current.bytes + ', expected ' + bytes);
    if (coreId) assert(current.id === coreId, phase + ': core was replaced');
    return current;
  };

  assert(first.bytes === expectedBytes, 'initial load did not retain exactly one heap block');
  assert(first.jsStagedBytesAtLoad === 0, 'initial load allocated JS staging chunks');
  await bridge.loadModelFromUrl(modelUrl, options);
  const second = await checkBytes('same-core replacement', expectedBytes, first.id);
  assert(second.jsStagedBytesAtLoad === 0, 'same-core replacement allocated JS staging chunks');
  assert(second.heapBytes === first.heapBytes, 'same-core replacement grew the wasm heap');
  const reuse = {
    bytes: expectedBytes,
    sameCoreHeapBytes: second.heapBytes,
    sameCoreJsStagedBytes: second.jsStagedBytesAtLoad,
  };
  const completion = (instance) => instance.createCompletion('The capital of France is', { nPredict: 12, temp: 0, seed: 1 });
  const reloadedOutput = await completion(bridge);
  if (reuseOnly) return { ...reuse, reloadedOutput };

  await readCoreProbe('arm load failure');
  let rejection;
  try {
    await bridge.loadModelFromUrl(modelUrl, options);
  } catch (error) {
    rejection = String(error.message);
  }
  assert(rejection?.includes('llama_model_load_from_file failed'), 'expected native load failure: ' + rejection);
  await checkBytes('after failed load', 0, first.id);
  await bridge.loadModelFromUrl(modelUrl, options);
  await checkBytes('same-core reload after failure', expectedBytes, first.id);
  const recoveredOutput = await completion(bridge);
  assert(recoveredOutput === reloadedOutput, 'generation changed after failed load');
  await bridge.dispose();

  const replacement = createBridge();
  try {
    const { nGpuLayers, ...omitted } = options;
    for (const loadOptions of [omitted, { ...omitted, nGpuLayers: -1 }]) {
      await replacement.loadModelFromUrl(modelUrl, loadOptions);
      await checkBytes('full offload', 0);
      const notes = replacement.getModelMetadata()['llamadart.webgpu.runtime_notes'].split(';');
      assert(!hasHeapNote(notes), 'full offload used heap staging');
    }
    await replacement.loadModelFromUrl(modelUrl, options);
    await checkBytes('reload after dispose', expectedBytes);
    const replacementOutput = await completion(replacement);
    assert(replacementOutput === reloadedOutput, 'generation changed on reload after dispose');
    return { ...reuse, reloadedOutput, replacementOutput };
  } finally {
    await replacement.dispose();
  }
}

export function renderHarness({
  memoryMode, runtimeMode, gpuLayers, jsBackedModelStaging, heapStagingLifecycle = false,
  heapStagingReuse = false, expectHeapFallback = null,
}, modelBytes, contextOptions = {}) {
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
const verifyGpuOffload = ${verifyGpuOffload.toString()};
const verifyHeapStagingLifecycle = ${verifyHeapStagingLifecycle.toString()};
const readCoreProbe = async (request) => {
  const channel = new BroadcastChannel(${JSON.stringify(CORE_PROBE_CHANNEL)});
  try {
    return await new Promise((resolve) => {
      channel.onmessage = ({ data }) => resolve(data);
      channel.postMessage(request);
    });
  } finally {
    channel.close();
  }
};
const options = ${JSON.stringify({ nCtx: 2048, nThreads: 4, nGpuLayers: gpuLayers, useCache: false, ...contextOptions })};
const modelUrl = '/${MODEL_FILENAME}';
const createBridge = () => new bridgeModule.LlamaWebGpuBridge(${JSON.stringify(coreOptions)});
let bridge;
let stagedBytes = 0;
let nativeFetchGuard;
try {
  assert(crossOriginIsolated === false, 'test page must be non-isolated');
  assert(typeof SharedArrayBuffer === 'undefined', 'test page must not expose SharedArrayBuffer');
  bridge = createBridge();

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
  window.__smokeStage = 'checking GPU offload';
  const gpuOffload = options.nGpuLayers > 0 ? await readCoreProbe('gpu offload') : null;
  verifyGpuOffload(options.nGpuLayers, gpuOffload);
  assert(metadata['llamadart.webgpu.n_threads'] === '1', 'expected one CPU thread');
  for (const note of ['core_pthreads:0', 'core_single_thread', 'threads_capped_no_coi']) {
    assert(loadNotes.includes(note), 'missing runtime note ' + note);
  }
  assert(loadNotes.includes('model_fetch_backend_requires_pthreads') === ${worker}, 'unexpected fetch fallback note');
  assert(metadata['llamadart.webgpu.model_source'] === 'network', 'model did not use streamed staging');
  assert(loadNotes.includes('model_staging_js_backed') === ${jsBackedModelStaging}, 'unexpected staging mode');
  const expectedFallback = ${JSON.stringify(expectHeapFallback)};
  if (expectedFallback) {
    const note = 'model_staging_heap_resident_unavailable:' + (expectedFallback === 'unsupported' ? 'unsupported' : '-48');
    assert(loadNotes.includes(note), 'missing heap staging fallback note ' + note);
    assert(!loadNotes.includes('model_staging_heap_resident'), 'fallback retained a heap block');
    const snapshot = await readCoreProbe('heap staging');
    assert(snapshot.bytes === 0, 'fallback retained heap-resident bytes');
    assert(snapshot.jsStagedBytesAtLoad >= ${modelBytes}, 'fallback did not stage the file in JS chunks');
  }
  assert(stagedBytes === ${modelBytes} && Number(metadata['llamadart.webgpu.model_bytes']) === stagedBytes,
    'staged byte count differs from the verified model size');

  window.__smokeStage = 'generating';
  let usage;
  const output = await bridge.createCompletion('The capital of France is', {
    nPredict: 12,
    temp: 0,
    seed: 1,
    onUsage: (entry) => { usage = entry; },
  });
  assert(usage?.completionTokens > 0, 'model did not generate');
  const { notes, gpuActive } = verifyRuntime('after generation');
  // WebAssembly linear memory cannot shrink, so its size after inference is its peak.
  const peakWasmHeapBytes = await readCoreProbe('peak heap');
  assert(Number.isSafeInteger(peakWasmHeapBytes) && peakWasmHeapBytes > 0, 'missing wasm heap measurement');

  let heapStagingLifecycle;
  if (${heapStagingLifecycle || heapStagingReuse}) {
    window.__smokeStage = 'checking heap staging lifecycle';
    heapStagingLifecycle = await verifyHeapStagingLifecycle({
      bridge, createBridge, modelUrl, options, readCoreProbe, modelBytes: ${modelBytes}, loadNotes,
      reuseOnly: ${heapStagingReuse}, assert,
    });
    assert(heapStagingLifecycle.reloadedOutput === output, 'generation changed after same-core replacement');
  }
  window.__smokeResult = {
    ok: true,
    browser: navigator.userAgent,
    core: ${JSON.stringify(memoryMode)},
    runtime: ${JSON.stringify(runtimeMode)},
    jsBackedModelStaging: ${jsBackedModelStaging},
    gpuLayers: options.nGpuLayers,
    gpuActive,
    gpuOffload,
    backend: bridge.getBackendName(),
    crossOriginIsolated,
    sharedArrayBuffer: typeof SharedArrayBuffer,
    stagedBytes,
    peakWasmHeapBytes,
    usage,
    output,
    heapStagingLifecycle,
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
  const heapStagingCheck = args.heapStagingLifecycle || args.heapStagingReuse;
  if (heapStagingCheck || args.expectHeapFallback) {
    ensure(args.jsBackedModelStaging && args.gpuLayers === 0,
      'heap staging checks require --js-backed-model-staging and --gpu-layers 0');
    ensure(!heapStagingCheck || !args.expectHeapFallback,
      'heap staging and fallback checks require separate runs');
    ensure(!args.heapStagingLifecycle || !args.heapStagingReuse,
      'select one heap staging check');
  }
  const model = await resolvePinnedModel(args);
  const { size } = await fsp.stat(model);
  const memory64 = args.memoryMode === 'wasm64';
  const contextOptions = heapStagingCheck ? { nCtx: 128, nBatch: 16, nUbatch: 16 } : {};
  const page = renderHarness(args, size, contextOptions);
  const payload = await withTempDir('single-thread-smoke-', async (webRoot) => {
    await copyBridgeArtifacts(distDir, webRoot);
    await copySingleThreadArtifacts(distDir, webRoot, { memory64 });
    await stageFile(model, path.join(webRoot, MODEL_FILENAME));
    await fsp.writeFile(path.join(webRoot, 'measure-core_st.js'), renderCoreProbe('llama_webgpu_core_st'));
    if (memory64) {
      await fsp.writeFile(path.join(webRoot, 'measure-core_mem64_st.js'), renderCoreProbe('llama_webgpu_core_mem64_st'));
    }
    if (args.runtimeMode === 'worker') {
      const coreName = memory64 ? 'llama_webgpu_core_mem64_st' : 'llama_webgpu_core_st';
      await fsp.writeFile(path.join(webRoot, 'native-fetch-probe.js'), renderNativeFetchProbe(coreName));
    }
    await fsp.writeFile(path.join(webRoot, 'index.html'), page);
    await writeTextArtifact(artifactsDir, 'index.html', page);
    return withServer(webRoot, async (url) => {
      const run = (pageUrl, artifactPrefix) => runPollingPlaywright(pageUrl, args.timeoutMs, artifactsDir, {
        artifactPrefix,
        stagePrefix: artifactPrefix,
        launchArgs: webGpuLaunchArgs(process.platform, { sharedArrayBuffer: false }),
      });
      const staged = await run(url, ARTIFACT_PREFIX);
      if (staged.ok && args.heapStagingLifecycle) {
        await fsp.writeFile(path.join(webRoot, 'baseline.html'), renderHarness({
          ...args, jsBackedModelStaging: false, heapStagingLifecycle: false,
        }, size, contextOptions));
        const baseline = await run(new URL('/baseline.html', url).href, ARTIFACT_PREFIX + '-baseline');
        ensure(baseline.ok, baseline.error || 'staging-off baseline failed');
        ensure(staged.output === baseline.output, 'heap staging changed generation relative to staging off');
      }
      return staged;
    }, { isolated: false });
  });
  process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
  return payload.ok ? 0 : 1;
}

if (import.meta.main) await runMain('single thread', main);
