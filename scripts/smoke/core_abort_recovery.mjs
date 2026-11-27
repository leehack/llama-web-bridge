#!/usr/bin/env node

import assert from 'node:assert/strict';
import fsp from 'node:fs/promises';
import path from 'node:path';

import {
  DEFAULT_MODEL_CACHE, copyBridgeArtifacts, copyMemory64Artifacts, copySingleThreadArtifacts,
  env, parseSmokeArgs, resolvePinnedModel, runMain, runPollingPlaywright,
  stageFile, webGpuLaunchArgs, withServer, withTempDir,
} from './support.mjs';

const STAGED_CASES = ['staged-load', 'staged-load-no-mmap'];
const DEFAULT_CASES = ['load', 'cpp-throw', 'next-token', 'cancel-cleanup', 'state-write', ...STAGED_CASES, 'healthy'];
const IFRAME_CASES = ['iframe', 'older-iframe'];
const MODEL = 'abort-smoke.gguf';
const TRUNCATED = 'abort-smoke-truncated.gguf';

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'core_abort_recovery.mjs',
    description: 'Exercise real native abort and Wasm trap recovery in Chromium.',
    options: [
      { flag: '--dist-dir', type: 'path', default: () => env.path('BRIDGE_DIST_DIR', 'dist') },
      { flag: '--baseline-dist-dir', type: 'path' },
      { flag: '--threading', type: 'string', choices: ['default', 'single'], default: () => 'default' },
      { flag: '--memory-mode', type: 'string', choices: ['wasm32', 'wasm64'], default: () => 'wasm32' },
      { flag: '--runtime-mode', type: 'string', choices: ['all', 'direct', 'worker'], default: () => 'all' },
      { flag: '--case', type: 'string', choices: [...DEFAULT_CASES, ...IFRAME_CASES] },
      { flag: '--model-path', type: 'path', default: () => env.optionalPath('LLAMA_WEBGPU_SMOKE_MODEL_PATH') },
      { flag: '--model-url', type: 'string', default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_URL') },
      { flag: '--model-sha256', type: 'string', default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_SHA256') },
      { flag: '--model-cache-dir', type: 'path', default: () => env.path('LLAMA_WEBGPU_SMOKE_MODEL_CACHE', DEFAULT_MODEL_CACHE) },
      { flag: '--artifacts-dir', type: 'path' },
      { flag: '--timeout-ms', type: 'int', default: () => 90000 },
    ],
  });
}

async function instrumentCore(createCore, settings, options) {
  const channel = new BroadcastChannel('core-abort-smoke');
  const id = crypto.randomUUID();
  const RealWorker = globalThis.Worker;
  let created = 0;
  let terminated = 0;
  let retired = false;
  let armed;
  const report = (type, payload = {}) => channel.postMessage({ id, type, ...payload });
  globalThis.Worker = class extends RealWorker {
    constructor(url, options) {
      super(url, options);
      created += 1;
      const terminate = this.terminate.bind(this);
      this.terminate = () => {
        terminated += 1;
        terminate();
      };
    }
  };
  let core;
  try {
    core = await createCore({
      ...options,
      onAbort(reason) {
        report('abort', { reason: String(reason) });
        options.onAbort?.(reason);
      },
    });
  } finally {
    globalThis.Worker = RealWorker;
  }
  const originalCcall = core.ccall;
  const enter = (name) => {
    report('call', { name, retired });
    if (armed?.point === name) {
      const { fault } = armed;
      armed = null;
      report('injected', { name, fault });
      if (fault === 'plain') throw new Error('operation aborted(foo)');
      if (fault === 'native-abort' || fault === 'cpp-throw') {
        originalCcall(`llamadart_webgpu_test_${fault === 'cpp-throw' ? 'throw' : 'abort'}_next_model_load`, null, [], []);
        return;
      }
      originalCcall('llamadart_webgpu_tokenize_bytes_to_json', 'number',
        ['pointer', 'number', 'number'], [0x7ffffff0, 64, 0]);
      throw new Error('The native trap unexpectedly returned');
    }
  };
  core.ccall = (name, ...args) => {
    enter(name);
    return originalCcall(name, ...args);
  };
  core.FS = new Proxy(core.FS, {
    get(target, name) {
      const value = Reflect.get(target, name);
      if (typeof value !== 'function') return value;
      return (...args) => {
        enter('FS.' + String(name));
        return Reflect.apply(value, target, args);
      };
    },
  });
  if (settings.hideRuntimeError) delete core.llamadartWasmRuntimeError;
  const retire = core.llamadartRetire;
  if (retire) {
    core.llamadartRetire = () => {
      retired = true;
      const stats = retire();
      report('retired', { stats, created, terminated });
      return stats;
    };
  }
  channel.onmessage = ({ data }) => {
    if (data.command === 'arm' && data.id === id) {
      armed = { point: data.point, fault: data.fault };
      report('armed');
    }
    if (data.command === 'snapshot') {
      report('snapshot', { request: data.request, created, terminated });
    }
  };
  report('created', { created, terminated });
  return core;
}

export function renderInstrumentedCore(coreName, { hideRuntimeError = false } = {}) {
  return `import createCore from '/${coreName}.js';
const instrumentCore = ${instrumentCore.toString()};
export default (options) => instrumentCore(createCore, ${JSON.stringify({ hideRuntimeError })}, options);
`;
}

async function runBrowserCase(settings) {
  const { LlamaWebGpuBridge } = await import('/llama_webgpu_bridge.js');
  const { scenario, modelUrl, truncatedUrl, worker, single, memory64, staging, coreName } = settings;
  const channel = new BroadcastChannel('core-abort-smoke');
  const events = [];
  const workers = [];
  const assert = (condition, message) => {
    if (!condition) throw new Error(message);
  };
  const waitFor = async (predicate, message) => {
    for (let i = 0; i < 200; i += 1) {
      const result = predicate();
      if (result) return result;
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    throw new Error(message);
  };
  const expectError = async (operation, predicate) => {
    try {
      await operation();
    } catch (error) {
      assert(predicate(error), `Unexpected error: ${error?.name} ${error?.code || ''} ${error?.message}`);
      return error;
    }
    throw new Error('The operation unexpectedly resolved');
  };
  const snapshot = (id) => {
    const request = crypto.randomUUID();
    channel.postMessage({ command: 'snapshot', request });
    return waitFor(() => events.find((event) => event.type === 'snapshot'
      && event.request === request && event.id === id), 'No snapshot from the active core');
  };
  const arm = async (id, point, fault = 'trap') => {
    const from = events.length;
    channel.postMessage({ command: 'arm', id, point, fault });
    await waitFor(() => events.slice(from).find((event) => event.type === 'armed' && event.id === id),
      `Core did not arm for ${point}`);
  };

  channel.onmessage = ({ data }) => { if (data.type) events.push(data); };
  const RealWorker = window.Worker;
  window.Worker = class extends RealWorker {
    constructor(url, options) {
      super(url, options);
      if (options?.name === 'em-pthread') return;
      const state = { terminated: false };
      workers.push(state);
      const terminate = this.terminate.bind(this);
      this.terminate = () => {
        state.terminated = true;
        terminate();
      };
    }
  };
  let bridge;
  const result = { ok: false, scenario, worker, staging, crossOriginIsolated,
    sharedArrayBuffer: typeof SharedArrayBuffer };
  try {
    assert(crossOriginIsolated === !single, 'Unexpected page isolation');
    assert(typeof SharedArrayBuffer === (single ? 'undefined' : 'function'), 'Unexpected SharedArrayBuffer support');
    const config = {
      disableWorker: !worker,
      allowSingleThreadCore: single,
      jsBackedModelStaging: staging,
      preferMemory64: memory64,
      coreModuleUrl: '/instrumented-core.js',
      wasmUrl: `/${coreName}.wasm`,
      workerUrl: '/llama_webgpu_bridge_worker.js',
    };
    if (memory64) {
      config.coreModuleUrlMem64 = config.coreModuleUrl;
      config.wasmUrlMem64 = config.wasmUrl;
    }
    const iframe = scenario === 'iframe' || scenario === 'older-iframe';
    let iframeWindow;
    if (iframe) {
      const frame = document.createElement('iframe');
      frame.src = '/core-frame.html';
      const loaded = new Promise((resolve) => frame.addEventListener('load', resolve, { once: true }));
      document.body.append(frame);
      await loaded;
      iframeWindow = frame.contentWindow;
      config.coreModuleFactory = await iframeWindow.loadCoreFactory;
    }
    bridge = new LlamaWebGpuBridge(config);
    const nThreads = single ? 1 : 4;
    const noMmap = scenario === 'staged-load-no-mmap';
    const options = {
      nCtx: 128, nThreads, nThreadsBatch: nThreads, nGpuLayers: 0, useCache: false,
      ...(noMmap ? { useMmap: false } : {}),
    };
    const load = (url = modelUrl) => bridge.loadModelFromUrl(url, options);
    const complete = (extra = {}) => bridge.createCompletion('hello', {
      nPredict: 8, temp: 0, seed: 1, ...extra,
    });
    window.__smokeStage = scenario + ':load';
    await load();
    const first = await waitFor(() => events.find((event) => event.type === 'created'), 'No core creation event');
    const initial = await snapshot(first.id);
    const metadata = bridge.getModelMetadata();
    const note = (value) => (value['llamadart.webgpu.runtime_notes'] || '').split(';');
    if (noMmap) {
      assert(!note(metadata).includes('model_staging_heap_resident'), 'useMmap:false staged into the heap');
    }
    assert(iframe || metadata['llamadart.webgpu.core_variant'] === (memory64 ? 'wasm64' : 'wasm32'),
      'The selected Wasm core did not match the requested memory mode');
    assert(Number(metadata['llamadart.webgpu.n_threads']) === nThreads, 'Unexpected model thread count');
    assert(initial.created === (single ? 0 : 4), `Unexpected pthread pool size: ${initial.created}`);
    assert(Boolean(bridge._workerProxy) === worker, 'Unexpected runtime fallback');

    if (scenario === 'parity') {
      const tokens = Array.from(await bridge.tokenize('hello'));
      const text = await complete();
      const bytes = await bridge.stateSaveBytes(tokens);
      await bridge.stateLoadBytes(bytes);
      await load();
      const updated = await snapshot(first.id);
      const parity = {
        tokens, text,
        metadata: Object.fromEntries(Object.entries(metadata).filter(([key]) =>
          key.startsWith('llamadart.webgpu.core_') || key.startsWith('llamadart.webgpu.n_threads'))),
        notes: note(metadata).filter((value) => value.startsWith('core_')),
        pthreads: updated.created,
        retirements: bridge._coreRetirements || 0,
      };
      await bridge.dispose();
      bridge = null;
      await waitFor(() => events.some(({ name }) => name === 'llamadart_webgpu_shutdown'),
        'No native shutdown during disposal');
      // The number of streamed download chunks varies between runs, and each chunk repeats its calls.
      parity.calls = events.filter(({ type }) => type === 'call').map(({ name }) => name)
        .filter((name, index, names) => name !== names[index - 1]);
      assert(parity.retirements === 0, 'The parity scenario retired a core');
      Object.assign(result, { parity, ok: true });
      return;
    }

    window.__smokeStage = scenario + ':fault';
    if (scenario === 'healthy') {
      await expectError(() => complete({ grammar: 'root ::= answer' }), (error) => error.message.includes('(invalid grammar)'));
      const controller = new AbortController();
      controller.abort();
      await expectError(() => complete({ signal: controller.signal }), (error) => error.name === 'AbortError');
      await arm(first.id, 'llamadart_webgpu_next_token', 'plain');
      await expectError(() => complete(), (error) => error.message === 'operation aborted(foo)');
      await expectError(() => load(truncatedUrl), (error) => error.message.startsWith('Failed to load GGUF model'));
      await load();
      const text = await complete();
      assert(typeof text === 'string' && text.length > 0, 'Generation failed after reloading on the same core');
      await snapshot(first.id);
      assert(!events.some(({ type }) => type === 'abort'), 'A healthy error called onAbort');
      assert(!bridge._coreRetirements && !events.some(({ type }) => type === 'retired'), 'A healthy error retired the core');
      assert(events.filter(({ type }) => type === 'created').length === 1, 'A healthy error replaced the core');
      assert(workers.filter(({ terminated }) => !terminated).length === Number(worker),
        'A healthy error changed the worker topology');
      result.ok = true;
      return;
    }

    const abortedLoad = scenario === 'load' || scenario === 'cpp-throw' || scenario === 'staged-load' || noMmap;
    const wasmTrap = (error) => error.message.includes('memory access out of bounds');
    const nativeAbort = (error) => /^Aborted\(/.test(error.message);
    const abortDuringLoad = async (fault = 'native-abort') => {
      await arm(first.id, 'llamadart_webgpu_load_model', fault);
      await expectError(() => load(), nativeAbort);
    };
    if (abortedLoad) {
      await abortDuringLoad(scenario === 'cpp-throw' ? 'cpp-throw' : 'native-abort');
    } else if (scenario === 'state-write') {
      await arm(first.id, 'FS.writeFile');
      await expectError(() => bridge.stateLoadBytes(new Uint8Array(64)), wasmTrap);
    } else if (scenario === 'cancel-cleanup') {
      await arm(first.id, 'llamadart_webgpu_end_generation');
      const controller = new AbortController();
      await complete({ nPredict: 64, signal: controller.signal, onToken: () => controller.abort() })
        .catch((error) => {
          assert(worker ? error.name === 'AbortError' : wasmTrap(error), `Unexpected cleanup error: ${error}`);
        });
      assert(controller.signal.aborted, 'Generation never started streaming');
    } else {
      await arm(first.id, 'llamadart_webgpu_next_token');
      const error = await expectError(() => complete(), wasmTrap);
      if (iframe) {
        assert(!(error instanceof Error) && error instanceof iframeWindow.WebAssembly.RuntimeError,
          'Trap did not come from the iframe Wasm realm');
      }
      if (scenario === 'older-iframe') {
        assert(!bridge._coreRetirements, 'An older iframe core retired on an unrecognized bare trap');
        await snapshot(first.id);
        await abortDuringLoad();
      }
    }

    const retired = await waitFor(() => events.find((event) => event.type === 'retired' && event.id === first.id),
      'The fault did not run the real core retirement hook');
    if (abortedLoad || scenario === 'older-iframe') {
      assert(events.some((event) => event.type === 'abort' && event.id === first.id), 'No real onAbort callback');
    }
    assert(retired.terminated === initial.created, 'The abort did not terminate the full pthread pool');
    assert(retired.stats.threads === (single ? -1 : initial.created), 'Unexpected retirement thread count');
    if (staging) assert(retired.stats.stagedBytes > 0, 'The abort did not report staged bytes released');
    window.__smokeStage = scenario + ':recover';
    if (!worker) {
      const notes = note(bridge.getModelMetadata());
      assert(notes.includes('core_retired'), 'Direct replacement lost the retirement note');
      assert(notes.includes(`core_retired_staged_bytes:${retired.stats.stagedBytes}`), 'Missing staged-byte note');
      if (!single) assert(notes.includes(`core_retired_threads:${initial.created}`), 'Missing pthread note');
      await expectError(() => bridge.tokenize('hello'),
        (error) => error.name === 'ModelUnloadedError' && error.code === 'MODEL_UNLOADED_AFTER_CORE_ABORT');
      await load();
    }
    const text = await complete();
    assert(typeof text === 'string' && text.length > 0, 'Generation did not recover');
    const active = await waitFor(() => events.find((event) => event.type === 'created' && event.id !== first.id),
      'No replacement core');
    const after = await snapshot(active.id);
    assert(after.created === initial.created && after.terminated === 0, 'Unexpected replacement pthread pool');
    assert(events.filter(({ type }) => type === 'created').length === 2, 'Expected exactly one replacement core');
    assert(bridge._coreRetirements === 1, 'Expected exactly one retirement');
    assert(!events.some(({ type, retired }) => type === 'call' && retired), 'A call reached the retired core');
    assert(workers.length === (worker ? 2 : 0)
      && workers.filter(({ terminated }) => !terminated).length === Number(worker),
    'Expected one live replacement bridge worker');
    Object.assign(result, {
      ok: true,
      pthreadsCreated: initial.created + after.created,
      pthreadsTerminated: retired.terminated,
      stagedBytes: retired.stats.stagedBytes,
    });
  } catch (error) {
    result.error = String(error?.stack || error);
  } finally {
    try {
      await bridge?.dispose();
    } catch (error) {
      result.ok = false;
      result.disposeError = String(error?.stack || error);
    }
    window.Worker = RealWorker;
    channel.close();
    if (!result.ok) result.events = events;
    window.__smokeResult = result;
  }
}

export function renderHarness(settings) {
  return `<!doctype html>
<meta charset="utf-8">
<title>Core abort recovery</title>
<body>
<script type="module">
const runBrowserCase = ${runBrowserCase.toString()};
await runBrowserCase(${JSON.stringify(settings)});
</script>
`;
}

async function runDist(args, distDir, modelPath, cases) {
  const single = args.threading === 'single';
  const memory64 = args.memoryMode === 'wasm64';
  const coreName = 'llama_webgpu_core' + (memory64 ? '_mem64' : '') + (single ? '_st' : '');
  return withTempDir('core-abort-smoke-', async (webRoot) => {
    await copyBridgeArtifacts(distDir, webRoot);
    if (single) await copySingleThreadArtifacts(distDir, webRoot, { memory64 });
    else if (memory64) await copyMemory64Artifacts(distDir, webRoot);
    await stageFile(modelPath, path.join(webRoot, MODEL));
    const model = await fsp.readFile(modelPath);
    await fsp.writeFile(path.join(webRoot, TRUNCATED), model.subarray(0, 1_000_000));
    await fsp.writeFile(path.join(webRoot, 'core-frame.html'),
      '<script>window.loadCoreFactory = import("/instrumented-core.js").then(module => module.default)</script>');
    return withServer(webRoot, async (url) => {
      const { chromium } = await import('playwright');
      const browser = await chromium.launch({ args: webGpuLaunchArgs(process.platform, { sharedArrayBuffer: !single }) });
      try {
        const results = {};
        for (const { scenario, worker, staging } of cases) {
          const key = [args.threading, args.memoryMode, worker ? 'worker' : 'direct', scenario, ...(staging ? ['staging'] : [])].join('-');
          await fsp.writeFile(path.join(webRoot, 'instrumented-core.js'),
            renderInstrumentedCore(coreName, { hideRuntimeError: scenario === 'older-iframe' }));
          await fsp.writeFile(path.join(webRoot, 'index.html'), renderHarness({
            scenario, worker, staging, single, memory64, coreName,
            modelUrl: '/' + MODEL, truncatedUrl: '/' + TRUNCATED,
          }));
          const payload = await runPollingPlaywright(url, args.timeoutMs, args.artifactsDir, {
            artifactPrefix: key,
            // The polling helper owns the page and closes its context after each case.
            launchBrowser: () => browser.newContext(),
            stageMarker: 'CORE_ABORT_SMOKE',
            stagePrefix: key,
          });
          assert.equal(payload.ok, true, `${key}: ${payload.error || payload.disposeError}\n${JSON.stringify(payload.events)}`);
          results[key] = scenario === 'parity' ? payload.parity : {
            pthreadsCreated: payload.pthreadsCreated,
            pthreadsTerminated: payload.pthreadsTerminated,
            stagedBytes: payload.stagedBytes,
          };
        }
        return results;
      } finally {
        await browser.close();
      }
    }, { isolated: !single });
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const modelPath = await resolvePinnedModel(args);
  const modes = args.runtimeMode === 'all' ? ['direct', 'worker'] : [args.runtimeMode];
  const cases = modes.flatMap((mode) => {
    const worker = mode === 'worker';
    if (args.baselineDistDir) return [false, true].map((staging) => ({ scenario: 'parity', worker, staging }));
    return [...DEFAULT_CASES, ...(worker ? [] : IFRAME_CASES)]
      .filter((scenario) => args.case == null || args.case === scenario)
      .map((scenario) => ({ scenario, worker, staging: STAGED_CASES.includes(scenario) }));
  });
  assert.ok(cases.length > 0, 'The selected case requires direct runtime');
  const results = await runDist(args, args.distDir, modelPath, cases);
  if (args.baselineDistDir) {
    const baselineArgs = { ...args, artifactsDir: args.artifactsDir && path.join(args.artifactsDir, 'baseline') };
    const baseline = await runDist(baselineArgs, args.baselineDistDir, modelPath, cases);
    if (args.artifactsDir) {
      await fsp.mkdir(args.artifactsDir, { recursive: true });
      await fsp.writeFile(path.join(args.artifactsDir, 'parity-branch.json'), JSON.stringify(results, null, 2));
      await fsp.writeFile(path.join(args.artifactsDir, 'parity-baseline.json'), JSON.stringify(baseline, null, 2));
    }
    assert.deepEqual(results, baseline);
  }
  console.log(JSON.stringify({ ok: true, results }, null, 2));
}

if (import.meta.main) await runMain('core abort recovery', main);
