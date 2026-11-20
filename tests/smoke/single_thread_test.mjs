import assert from 'node:assert/strict';
import { test } from 'node:test';
import vm from 'node:vm';

import { parseArgs, parseGpuOffload, renderHarness, verifyGpuOffload } from '../../scripts/smoke/single_thread.mjs';
import { webGpuLaunchArgs } from '../../scripts/smoke/support.mjs';

function renderPage(options) {
  const page = renderHarness(options, 42);
  const script = page.match(/<script type="module">([\s\S]+)<\/script>/)[1];
  return script.replace(/^import \* as bridgeModule from .*;$/m, '');
}

async function runPage(options, { fallBackDuringGeneration = false, gpuOffload = { offloadedLayers: 29, modelLayers: 29 }, ...overrides } = {}) {
  const calls = [];
  class SingleThreadCoreError extends Error {
    code = 'SINGLE_THREAD_CORE_UNSUPPORTED';
    feature = 'remote_fetch_backend';
  }
  class LlamaWebGpuBridge {
    constructor(config) {
      calls.push(config);
    }
    async loadModelFromUrl(_url, opts) {
      calls.push(opts.forceRemoteFetchBackend);
      if (opts.forceRemoteFetchBackend) throw new SingleThreadCoreError();
      opts.progressCallback({ loaded: 42 });
    }
    getModelMetadata() {
      return {
        'llamadart.webgpu.runtime_notes': [
          'core_pthreads:0', 'core_single_thread', 'threads_capped_no_coi',
          ...(options.runtimeMode === 'worker' ? ['model_fetch_backend_requires_pthreads'] : []),
          ...(options.jsBackedModelStaging ? ['model_staging_js_backed'] : []),
        ].join(';'),
        'llamadart.webgpu.core_variant': options.memoryMode,
        'llamadart.webgpu.execution': options.runtimeMode === 'worker' ? 'worker' : 'main-thread',
        'llamadart.webgpu.n_threads': '1',
        'llamadart.webgpu.model_source': 'network',
        'llamadart.webgpu.model_bytes': '42',
      };
    }
    isGpuActive() { return options.gpuLayers > 0 && !this.fellBack; }
    getBackendName() { return 'Test backend'; }
    async createCompletion(_prompt, { onUsage }) {
      this.fellBack = fallBackDuringGeneration;
      onUsage({ completionTokens: 1 });
    }
    async dispose() {}
  }
  class BroadcastChannel {
    postMessage(request) {
      this.onmessage({ data: request === 'gpu offload' ? gpuOffload : 100 });
    }
    close() {}
  }
  class Worker {
    postMessage() {
      this.onmessage({ data: { rc: -7, message: 'Fetch-backed model loading requires a core built with pthreads' } });
    }
    terminate() {}
  }
  const window = {};
  await vm.runInNewContext(`(async () => { ${renderPage(options)} })()`, {
    BroadcastChannel, bridgeModule: { LlamaWebGpuBridge, SingleThreadCoreError },
    crossOriginIsolated: false, SharedArrayBuffer: undefined, navigator: { userAgent: 'Test Chromium' },
    performance, window, Worker, URL, location: { href: 'http://localhost/index.html' }, ...overrides,
  });
  return { result: window.__smokeResult, calls };
}

test('single-thread acceptance uses WebGPU flags without exposing shared memory', () => {
  assert.ok(webGpuLaunchArgs('darwin').includes('--enable-features=SharedArrayBuffer'));
  for (const platform of ['darwin', 'linux']) {
    const args = webGpuLaunchArgs(platform, { sharedArrayBuffer: false });
    assert.ok(args.includes('--enable-unsafe-webgpu'));
    assert.ok(args.every((arg) => !arg.includes('SharedArrayBuffer')));
  }
});

test('single-thread acceptance exercises worker fallback and reports wasm64 peak heap', async () => {
  const options = { memoryMode: 'wasm64', runtimeMode: 'worker', gpuLayers: 999, jsBackedModelStaging: true };
  const { result, calls } = await runPage(options);
  assert.equal(result.ok, true, result.error);
  assert.equal(result.gpuActive, true);
  assert.deepEqual(result.gpuOffload, { offloadedLayers: 29, modelLayers: 29 });
  assert.equal(result.stagedBytes, 42);
  assert.equal(result.peakWasmHeapBytes, 100);
  assert.equal(result.nativeFetchGuard.rc, -7);
  assert.equal(calls[0].coreModuleUrlMem64, '/measure-core_mem64_st.js');
  assert.equal(calls[0].wasmUrlMem64, '/llama_webgpu_core_mem64_st.wasm');
  assert.deepEqual(calls.slice(1), [true, undefined]);
});

test('single-thread acceptance rejects isolation and a visible SharedArrayBuffer', async () => {
  const options = { memoryMode: 'wasm32', runtimeMode: 'direct', gpuLayers: 0, jsBackedModelStaging: false };
  const direct = await runPage(options);
  assert.equal(direct.result.ok, true, direct.result.error);
  assert.equal(direct.result.gpuActive, false);
  assert.deepEqual(direct.calls.slice(1), [undefined]);
  const isolated = await runPage(options, { crossOriginIsolated: true });
  assert.equal(isolated.result.ok, false);
  assert.match(isolated.result.error, /must be non-isolated/);
  const shared = await runPage(options, { SharedArrayBuffer });
  assert.equal(shared.result.ok, false);
  assert.match(shared.result.error, /must not expose SharedArrayBuffer/);
});

test('single-thread acceptance rejects zero offloaded layers while WebGPU stays active', async () => {
  const options = { memoryMode: 'wasm32', runtimeMode: 'direct', gpuLayers: 999, jsBackedModelStaging: false };
  const { result } = await runPage(options, { gpuOffload: parseGpuOffload('load_tensors: offloaded 0/29 layers to GPU') });
  assert.equal(result.ok, false);
  assert.equal(result.stage, 'checking GPU offload');
  assert.match(result.error, /offloaded 0\/29 layers to GPU, expected at least 29/);
});

test('GPU offload requires a native count that meets the request capped at model layers', () => {
  const nativeLog = (count) => parseGpuOffload(`load_tensors: offloaded ${count} layers to GPU\n`);
  assert.deepEqual(nativeLog('0/29'), { offloadedLayers: 0, modelLayers: 29 });
  assert.throws(() => verifyGpuOffload(999, nativeLog('0/29')), /expected at least 29/);
  assert.throws(() => verifyGpuOffload(999, nativeLog('28/29')), /expected at least 29/);
  assert.throws(() => verifyGpuOffload(4, nativeLog('3/29')), /expected at least 4/);
  verifyGpuOffload(999, nativeLog('29/29'));
  verifyGpuOffload(4, nativeLog('4/29'));
  assert.equal(nativeLog('30/29'), null);
  const missing = parseGpuOffload('load_tensors: offloading 28 repeating layers to GPU');
  assert.equal(missing, null);
  assert.throws(() => verifyGpuOffload(999, missing), /missing native GPU layer offload count/);
  verifyGpuOffload(0, missing);
});

test('single-thread acceptance fails if generation switches from WebGPU to CPU', async () => {
  const options = { memoryMode: 'wasm64', runtimeMode: 'worker', gpuLayers: 999, jsBackedModelStaging: true };
  const { result } = await runPage(options, { fallBackDuringGeneration: true });
  assert.equal(result.ok, false);
  assert.match(result.error, /GPU state did not match requested offload after generation/);
});

test('single-thread acceptance accepts pinned local input and explicit runtime options', () => {
  const args = parseArgs([
    '--dist-dir', 'dist', '--model-path', 'model.gguf', '--model-sha256', 'sha',
    '--memory-mode', 'wasm64', '--runtime-mode', 'direct', '--gpu-layers', '0',
    '--js-backed-model-staging',
  ]);
  assert.equal(args.modelPath, 'model.gguf');
  assert.equal(args.modelSha256, 'sha');
  assert.equal(args.memoryMode, 'wasm64');
  assert.equal(args.runtimeMode, 'direct');
  assert.equal(args.gpuLayers, 0);
  assert.equal(args.jsBackedModelStaging, true);
});
