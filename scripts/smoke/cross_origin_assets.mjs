#!/usr/bin/env node

import assert from 'node:assert/strict';
import fsp from 'node:fs/promises';
import path from 'node:path';

import { chromium, firefox } from 'playwright';

import {
  DEFAULT_MODEL_CACHE, copyBridgeArtifacts, copyMemory64Artifacts, copySingleThreadArtifacts,
  env, parseSmokeArgs, resolvePinnedModel, runMain, stageFile, webGpuLaunchArgs, withServer,
  withTempDir, writeJsonArtifact,
} from './support.mjs';

const MODEL = 'cross-origin-model.gguf';
const WORKER_HOOK = 'var Worker=Module[Symbol.for("llamadart.pthreadWorker")]||globalThis.Worker;';
const POOL_SIZE = 4;

const scenarios = {
  'plain-st-direct': { single: true, direct: true, workers: 0 },
  'csp-st-direct': { single: true, direct: true, csp: 'self', workers: 0 },
  'plain-st': { single: true, bridgeModuleWorker: true, workers: 1 },
  'plain-st-default-worker': { single: true, workers: 1 },
  'relative-st': { single: true, relativeCore: true, workers: 1 },
  'error-then-ready': { single: true, isolated: true, errorThenReady: true, fallback: 'script' },
  'csp-st': { single: true, csp: 'self', workers: 0, violations: 1, fallback: 'csp' },
  'isolated-threaded': {},
  'isolated-threaded-direct': { direct: true },
  'bundled-loader': { direct: true, bundledLoader: true },
  'csp-threaded': { csp: 'self', direct: true, blocked: 'csp' },
  'csp-threaded-worker': { csp: 'self', blocked: 'csp' },
  'csp-threaded-worker-response': { workerResponseCsp: true, blocked: 'csp' },
  'csp-blob-only-threaded': { csp: 'blob', direct: true, blocked: 'script' },
  'csp-blob-only-threaded-worker': { csp: 'blob', blocked: 'script' },
  'same-origin': { sameOrigin: true, csp: 'self' },
  'failed-wasm': { direct: true, failedWasm: true },
  'old-core': { direct: true, oldCore: true, workers: 0 },
};

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'cross_origin_assets.mjs',
    description: 'Exercise CDN-only worker and pthread startup in a real browser.',
    options: [
      { flag: '--dist-dir', type: 'path', default: () => env.path('BRIDGE_DIST_DIR', 'dist') },
      { flag: '--model-path', type: 'path' },
      { flag: '--model-url', type: 'string', default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_URL') },
      { flag: '--model-sha256', type: 'string', default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_SHA256') },
      { flag: '--model-cache-dir', type: 'path', default: () => env.path('LLAMA_WEBGPU_SMOKE_MODEL_CACHE', DEFAULT_MODEL_CACHE) },
      { flag: '--memory-mode', type: 'string', choices: ['wasm32', 'wasm64'], default: () => 'wasm32' },
      { flag: '--case', type: 'string', choices: Object.keys(scenarios) },
      { flag: '--browser', type: 'string', choices: ['chromium', 'firefox'], default: () => 'chromium' },
      { flag: '--artifacts-dir', type: 'path' },
      { flag: '--timeout-ms', type: 'int', default: () => 90_000 },
    ],
  });
}

export function coreFileStem(memoryMode, single) {
  return `llama_webgpu_core${memoryMode === 'wasm64' ? '_mem64' : ''}${single ? '_st' : ''}`;
}

export function renderHarness(name, assetOrigin, memoryMode) {
  const scenario = scenarios[name];
  const core = coreFileStem(memoryMode, scenario.single);
  const bridgeOrigin = scenario.workerResponseCsp ? '' : assetOrigin;
  const bridgeUrl = `${bridgeOrigin}/llama_webgpu_bridge.js`;
  const coreFile = scenario.bundledLoader ? 'bundled-loader.js' : `${scenario.oldCore ? 'old/' : ''}${core}.js`;
  const coreUrl = `${scenario.relativeCore ? '.' : assetOrigin}/${coreFile}`;
  const wasmUrl = `${assetOrigin}/${scenario.failedWasm ? 'missing' : core}.wasm`;
  return `
const { LlamaWebGpuBridge, BlobWorkerBlockedError } = await import(${JSON.stringify(bridgeUrl)});
const config = ${JSON.stringify({
    disableWorker: scenario.direct === true,
    allowSingleThreadCore: scenario.single === true,
    coreModuleUrl: coreUrl,
    wasmUrl,
    ...(memoryMode === 'wasm64' ? { preferMemory64: true, coreModuleUrlMem64: coreUrl, wasmUrlMem64: wasmUrl } : {}),
    ...(scenario.bridgeModuleWorker ? { workerUrl: bridgeUrl } : {}),
    ...(scenario.errorThenReady ? { workerUrl: `${assetOrigin}/error-then-ready.js` } : {}),
    coreInitTimeoutMs: 10_000,
    workerInitTimeoutMs: 3_000,
  })};
const violations = [];
addEventListener('securitypolicyviolation', (event) => {
  violations.push({ directive: event.effectiveDirective, blocked: event.blockedURI });
});
const result = { violations, isolated: crossOriginIsolated, sharedMemory: typeof SharedArrayBuffer };
${scenario.errorThenReady ? `
const NativeWorker = globalThis.Worker;
result.workerEvents = [];
globalThis.Worker = class extends NativeWorker {
  constructor(...args) {
    super(...args);
    this.addEventListener('error', () => {
      result.workerEvents.push('error');
      // Let the worker enqueue ready before the page handles the next task.
      const until = performance.now() + 30;
      while (performance.now() < until) {}
    });
    this.addEventListener('message', ({ data }) => result.workerEvents.push(data.type));
  }
};
` : ''}
const bridge = new LlamaWebGpuBridge(config);
const fallbackKey = 'llamadart.webgpu.worker_fallback_reason';
const started = performance.now();
const observeFallback = () => {
  if (result.fallbackMs === undefined && bridge.getModelMetadata()[fallbackKey]) {
    result.fallbackMs = performance.now() - started;
  }
};
const timer = setInterval(observeFallback, 20);
try {
  const load = () => bridge.loadModelFromUrl(${JSON.stringify(`${assetOrigin}/${MODEL}`)}, {
    nCtx: 512, nThreads: 4, nGpuLayers: 0, useCache: false,
    forceRemoteFetchBackend: ${Boolean(scenario.errorThenReady)},
  });
  ${scenario.failedWasm ? `
  result.rejections = [];
  for (let attempt = 0; attempt < 2; attempt++) {
    try { await load(); } catch (error) { result.rejections.push(String(error)); }
  }
  ` : `
  await load();
  await bridge.createCompletion('The capital of France is', {
    nPredict: 6, temp: 0, seed: 1,
    onUsage: (usage) => { result.completionTokens = usage.completionTokens; },
  });
  observeFallback();
  result.metadata = bridge.getModelMetadata();
  `}
} catch (error) {
  result.error = {
    message: error.message, name: error.name, code: error.code,
    reason: error.reason, target: error.target,
    typed: typeof BlobWorkerBlockedError === 'function' && error instanceof BlobWorkerBlockedError,
    elapsedMs: performance.now() - started,
  };
} finally {
  clearInterval(timer);
  await bridge.dispose();
  window.__crossOriginSmoke = result;
}
`;
}

export function verifyScenario(name, memoryMode, result, workers, wasmRequests, browser = 'chromium') {
  const scenario = scenarios[name];
  const firefoxFallback = browser === 'firefox' && !scenario.direct && !scenario.sameOrigin && !scenario.workerResponseCsp;
  const fallback = firefoxFallback || Boolean(scenario.fallback);
  const direct = scenario.direct || fallback;
  const blobWorkers = workers.filter((url) => url.startsWith('blob:')).length;
  const urlWorkers = workers.length - blobWorkers;
  const isolated = scenario.isolated ?? !scenario.single;
  assert.equal(result.isolated, isolated, 'page isolation');
  assert.equal(result.sharedMemory, isolated ? 'function' : 'undefined', 'SharedArrayBuffer');
  if (scenario.workers !== undefined) {
    assert.equal(workers.length, firefoxFallback ? 1 : scenario.workers, 'worker count');
  }
  const violations = result.violations.length;
  assert.ok(result.violations.every(({ directive, blocked }) => directive === 'worker-src' && blocked === 'blob'),
    'unexpected CSP violation');
  if (scenario.blocked === 'csp' && !scenario.workerResponseCsp) {
    const bridgeViolations = browser === 'chromium' && !scenario.direct ? 1 : 0;
    const attemptedPools = memoryMode === 'wasm64' ? 2 : 1;
    assert.ok(violations >= bridgeViolations + 1 && violations <= bridgeViolations + attemptedPools * POOL_SIZE,
      'worker-src CSP violations');
    assert.equal(urlWorkers, firefoxFallback ? 1 : 0, 'direct bridge worker attempts');
    assert.ok(blobWorkers <= (browser === 'firefox' ? violations : 0), 'unexpected blob worker during CSP rejection');
  } else {
    assert.equal(violations, firefoxFallback ? 0 : scenario.violations ?? 0, 'worker-src CSP violations');
  }

  if (scenario.blocked) {
    assert.equal(result.error?.name, 'BlobWorkerBlockedError', result.error?.message);
    assert.equal(result.error.code, 'BLOB_WORKER_BLOCKED');
    const reason = scenario.workerResponseCsp && browser === 'firefox' ? 'unavailable' : scenario.blocked;
    assert.equal(result.error.reason, reason);
    assert.equal(result.error.target, 'pthread');
    assert.equal(result.error.typed, true);
    assert.ok(result.error.elapsedMs < 2_000, 'blocked pthread startup did not reject promptly');
    return;
  }
  if (scenario.oldCore) {
    if (browser === 'firefox') {
      assert.match(result.error?.message, /Bridge core init timeout/);
    } else {
      assert.equal(result.error?.name, 'SecurityError', result.error?.message);
    }
    return;
  }

  assert.equal(result.error, undefined, result.error?.message);
  if (scenario.failedWasm) {
    assert.equal(result.rejections?.length, 2, 'load and retry must both reject');
    for (const error of result.rejections) assert.match(error, /fetching of the wasm failed/);
    assert.equal(result.liveWorkers, 0, 'failed core retained pthread workers');
    return;
  }
  if (scenario.errorThenReady) assert.equal(result.workerEvents[0], 'error');
  assert.ok(result.completionTokens > 0, 'no completion tokens');
  const meta = (key) => result.metadata[`llamadart.webgpu.${key}`];
  const startMode = scenario.sameOrigin ? 'direct' : 'blob';
  assert.equal(meta('execution'), direct ? 'main-thread' : 'worker');
  assert.equal(meta('n_threads'), scenario.single ? '1' : '4');
  assert.equal(meta('core_variant'), memoryMode);

  const startNotes = meta('runtime_notes').split(';');
  const expected = [
    ...(direct ? [] : [`worker_start:${startMode}`]),
    ...(scenario.fallback && !firefoxFallback ? [`worker_start_blocked:${scenario.fallback}`] : []),
    ...(scenario.single ? [] : [`pthread_start:${startMode}`]),
  ];
  assert.deepEqual(
    startNotes.filter((note) => /^(worker_start|worker_start_blocked|pthread_start):/.test(note)).sort(),
    expected.sort(),
  );
  assert.equal(startNotes.includes('core_single_thread'), Boolean(scenario.single));
  if (!scenario.single) {
    assert.ok(workers.length >= POOL_SIZE + (direct && !firefoxFallback ? 0 : 1), 'missing thread pool or worker');
    assert.equal(wasmRequests, 1, 'Wasm requests per core start');
  }
  assert.equal(urlWorkers, scenario.sameOrigin ? workers.length : firefoxFallback ? 1 : 0, 'worker URL origins');
  if (fallback) {
    const reason = scenario.errorThenReady ? /initialization callback failed/ : /^Bridge worker crashed$/;
    assert.match(meta('worker_fallback_reason'), reason);
    assert.ok(result.fallbackMs < 2_000, 'bridge worker fallback was too slow');
  }
}

export function coreWithoutWorkerHook(source) {
  assert.ok(source.includes(WORKER_HOOK), 'could not find the generated pthread hook');
  return source.replace(WORKER_HOOK, '');
}

async function runScenario(browser, pageDir, assetOrigin, name, options, wasmRequests) {
  const scenario = scenarios[name];
  const policy = scenario.csp
    ? [['Content-Security-Policy', [
      "default-src 'self'",
      `script-src 'self' 'wasm-unsafe-eval' ${assetOrigin}`,
      `connect-src 'self' ${assetOrigin}`,
      `worker-src 'self'${scenario.csp === 'blob' ? ' blob:' : ''}`,
    ].join('; ')]]
    : [];
  await fsp.writeFile(path.join(pageDir, 'harness.js'), renderHarness(name, scenario.sameOrigin ? '' : assetOrigin, options.memoryMode));

  return withServer(pageDir, async (url) => {
    const workers = [];
    const console = [];
    const page = await browser.newPage();
    page.on('worker', (worker) => workers.push(worker.url()));
    page.on('console', (line) => console.push(`${line.type()}: ${line.text()}`));
    page.on('pageerror', (error) => console.push(`pageerror: ${error.message}`));
    const corePath = `/${coreFileStem(options.memoryMode, scenario.single)}.wasm`;
    const initialRequests = wasmRequests.get(corePath) ?? 0;
    try {
      await page.goto(url, { timeout: options.timeoutMs });
      await page.waitForFunction(() => window.__crossOriginSmoke, null, { timeout: options.timeoutMs });
      const result = await page.evaluate(() => window.__crossOriginSmoke);
      if (scenario.failedWasm) {
        await page.waitForTimeout(100);
        result.liveWorkers = page.workers().length;
      }
      const fetched = (wasmRequests.get(corePath) ?? 0) - initialRequests;
      await writeJsonArtifact(options.artifactsDir, `${name}.json`, { ...result, workers, wasmRequests: fetched, console });
      verifyScenario(name, options.memoryMode, result, workers, fetched, options.browser);
    } catch (error) {
      throw new Error(`${name}: ${error.message}\n${console.slice(-20).join('\n')}`, { cause: error });
    } finally {
      await page.close();
    }
  }, {
    isolated: scenario.isolated ?? !scenario.single,
    extraHeaders: (req) => scenario.workerResponseCsp && req.url === '/llama_webgpu_bridge_worker.js'
      ? [['Content-Security-Policy', "worker-src 'self'"]]
      : policy,
    onRequest: wasmRequests.record,
  });
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const model = await resolvePinnedModel(options);
  await withTempDir('cross-origin-assets-', async (tmp) => {
    const pageDir = path.join(tmp, 'page');
    const assetDir = path.join(tmp, 'assets');
    await Promise.all([fsp.mkdir(pageDir), fsp.mkdir(path.join(assetDir, 'old'), { recursive: true })]);
    await fsp.writeFile(path.join(pageDir, 'index.html'),
      '<!doctype html><meta charset="utf-8"><script type="module" src="/harness.js"></script>');

    async function stageAssets(dir) {
      await copyBridgeArtifacts(options.distDir, dir);
      if (options.memoryMode === 'wasm64') await copyMemory64Artifacts(options.distDir, dir);
      await copySingleThreadArtifacts(options.distDir, dir, { memory64: options.memoryMode === 'wasm64' });
      await stageFile(model, path.join(dir, MODEL));
    }
    await stageAssets(assetDir);
    const threadedFile = `${coreFileStem(options.memoryMode, false)}.js`;
    const singleFile = `${coreFileStem(options.memoryMode, true)}.js`;
    const [threaded, single] = await Promise.all(
      [threadedFile, singleFile].map((file) => fsp.readFile(path.join(assetDir, file), 'utf8')),
    );
    assert.ok(threaded.includes(WORKER_HOOK), 'threaded core lacks the worker hook');
    assert.ok(!single.includes(WORKER_HOOK), 'single-thread core includes the worker hook');
    await fsp.writeFile(path.join(assetDir, 'old', threadedFile), coreWithoutWorkerHook(threaded));
    await fsp.writeFile(path.join(assetDir, 'bundled-loader.js'), `
if (typeof document === 'undefined') throw new Error('Application code requires the page');
${threaded}
`);
    await fsp.writeFile(path.join(assetDir, 'error-then-ready.js'), `
import { enableBridgeWorkerHost } from './llama_webgpu_bridge.js';
enableBridgeWorkerHost();
const handleMessage = self.onmessage;
self.onmessage = (event) => {
  if (event.data.type === 'init') reportError(new Error('Bridge worker initialization callback failed'));
  handleMessage(event);
};
`);

    const requests = new Map();
    const recordRequest = (req) => {
      if (req.method === 'GET' && req.url.endsWith('.wasm')) {
        requests.set(req.url, (requests.get(req.url) ?? 0) + 1);
      }
    };
    const chromiumOnly = ['old-core', 'error-then-ready', 'failed-wasm'];
    const browserScenarios = Object.keys(scenarios).filter((name) => options.browser !== 'firefox' || !chromiumOnly.includes(name));
    const cases = options.case ? [options.case] : browserScenarios;
    await withServer(assetDir, async (url) => {
      const assetOrigin = new URL(url).origin;
      for (const single of [true, false]) {
        const selected = cases.filter((name) => Boolean(scenarios[name].single) === single);
        if (!selected.length) continue;
        const browser = options.browser === 'firefox'
          ? await firefox.launch()
          : await chromium.launch({ args: webGpuLaunchArgs(process.platform, { sharedArrayBuffer: !single }) });
        try {
          for (const name of selected) {
            if (scenarios[name].sameOrigin || scenarios[name].workerResponseCsp) await stageAssets(pageDir);
            await runScenario(browser, pageDir, assetOrigin, name, options, {
              get: (file) => requests.get(file), record: recordRequest,
            });
            process.stdout.write(`PASS ${name}\n`);
          }
        } finally {
          await browser.close();
        }
      }
    }, { isolated: false, extraHeaders: [['Cross-Origin-Resource-Policy', 'cross-origin']], onRequest: recordRequest });
  });
  return 0;
}

if (import.meta.main) await runMain('Cross-origin assets', main);
