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
const WORKER_HOOK = 'Module["mainScriptUrlOrBlob"]';

const scenarios = {
  'plain-st-direct': { single: true, direct: true, workers: 0 },
  'csp-st-direct': { single: true, direct: true, csp: 'self', workers: 0 },
  'plain-st': { single: true, bridgeModuleWorker: true, workers: 1 },
  'plain-st-default-worker': { single: true, workers: 1 },
  'csp-st': { single: true, csp: 'self', workers: 0, violations: 1, fallback: 'csp' },
  'isolated-threaded': {},
  'isolated-threaded-direct': { direct: true },
  'bundled-loader': { direct: true, bundledLoader: true },
  'csp-threaded': { csp: 'self', direct: true, blocked: 'csp', workers: 0, violations: 1 },
  'csp-threaded-worker': { csp: 'self', blocked: 'csp', workers: 0, violations: 1 },
  'csp-threaded-worker-response': { workerResponseCsp: true, blocked: 'csp' },
  'csp-blob-only-threaded': { csp: 'blob', direct: true, blocked: 'script' },
  'csp-blob-only-threaded-worker': { csp: 'blob', blocked: 'script' },
  'same-origin': { sameOrigin: true },
  'old-core': { direct: true, oldCore: true, workers: 0 },
};

const firefoxScenarios = [
  'plain-st-direct', 'csp-st-direct', 'plain-st', 'csp-st',
  'isolated-threaded', 'isolated-threaded-direct',
  'csp-threaded', 'csp-threaded-worker', 'csp-threaded-worker-response',
  'csp-blob-only-threaded',
];

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
  const coreUrl = scenario.bundledLoader
    ? `${assetOrigin}/bundled-loader.js`
    : `${assetOrigin}/${scenario.oldCore ? 'old/' : ''}${core}.js`;
  const wasmUrl = `${assetOrigin}/${core}.wasm`;
  return `
const { LlamaWebGpuBridge, BlobWorkerBlockedError } = await import(${JSON.stringify(bridgeUrl)});
const config = ${JSON.stringify({
    disableWorker: scenario.direct === true,
    allowSingleThreadCore: scenario.single === true,
    coreModuleUrl: coreUrl,
    wasmUrl,
    ...(memoryMode === 'wasm64' ? { preferMemory64: true, coreModuleUrlMem64: coreUrl, wasmUrlMem64: wasmUrl } : {}),
    ...(scenario.bridgeModuleWorker ? { workerUrl: bridgeUrl } : {}),
    coreInitTimeoutMs: 10_000,
    workerInitTimeoutMs: 3_000,
  })};
const violations = [];
addEventListener('securitypolicyviolation', (event) => {
  violations.push({ directive: event.effectiveDirective, blocked: event.blockedURI });
});
const result = { violations, isolated: crossOriginIsolated, sharedMemory: typeof SharedArrayBuffer };
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
  await bridge.loadModelFromUrl(${JSON.stringify(`${assetOrigin}/${MODEL}`)}, {
    nCtx: 512, nThreads: 4, nGpuLayers: 0, useCache: false,
  });
  await bridge.createCompletion('The capital of France is', {
    nPredict: 6, temp: 0, seed: 1,
    onUsage: (usage) => { result.completionTokens = usage.completionTokens; },
  });
  observeFallback();
  result.metadata = bridge.getModelMetadata();
} catch (error) {
  result.error = {
    message: error.message, name: error.name, code: error.code,
    reason: error.reason, target: error.target, typed: error instanceof BlobWorkerBlockedError,
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
  assert.equal(result.isolated, !scenario.single, 'page isolation');
  assert.equal(result.sharedMemory, scenario.single ? 'undefined' : 'function', 'SharedArrayBuffer');
  if (scenario.workers !== undefined) {
    if (browser === 'firefox' && scenario.csp === 'self' && scenario.workers === 0) {
      // Playwright can report a Firefox worker event for a CSP-rejected attempt.
      assert.ok(workers.length <= result.violations.length && workers.every((url) => url.startsWith('blob:')),
        'unexpected worker during CSP rejection');
    } else {
      assert.equal(workers.length, scenario.workers, 'worker count');
    }
  }
  if (scenario.violations !== undefined) {
    const count = result.violations.length;
    const firefoxDuplicate = browser === 'firefox' && name === 'csp-st' && count === 2;
    assert.ok(count === scenario.violations || firefoxDuplicate, 'worker-src CSP violations');
    assert.ok(result.violations.every(({ directive, blocked }) => directive === 'worker-src' && blocked === 'blob'),
      'unexpected CSP violation');
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
    assert.equal(result.error?.name, 'SecurityError', result.error?.message);
    assert.equal(result.violations.length, 0);
    return;
  }

  assert.equal(result.error, undefined, result.error?.message);
  if (!scenario.violations) assert.deepEqual(result.violations, []);
  assert.ok(result.completionTokens > 0, 'no completion tokens');
  const meta = (key) => result.metadata[`llamadart.webgpu.${key}`];
  const direct = scenario.direct || Boolean(scenario.fallback);
  const startMode = scenario.sameOrigin ? 'direct' : 'blob';
  assert.equal(meta('execution'), direct ? 'main-thread' : 'worker');
  assert.equal(meta('n_threads'), scenario.single ? '1' : '4');
  assert.equal(meta('core_variant'), memoryMode);

  const startNotes = meta('runtime_notes').split(';');
  const expected = [
    ...(direct ? [] : [`worker_start:${startMode}`]),
    ...(scenario.fallback ? [`worker_start_blocked:${scenario.fallback}`] : []),
    ...(scenario.single ? [] : [`pthread_start:${startMode}`]),
  ];
  assert.deepEqual(
    startNotes.filter((note) => /^(worker_start|worker_start_blocked|pthread_start):/.test(note)).sort(),
    expected.sort(),
  );
  assert.equal(startNotes.includes('core_single_thread'), Boolean(scenario.single));
  if (!scenario.single) {
    assert.ok(workers.length >= (direct ? 4 : 5), 'missing thread pool or worker');
    assert.equal(wasmRequests, 1, 'Wasm requests per core start');
  }
  for (const url of workers) assert.equal(url.startsWith('blob:'), !scenario.sameOrigin, url);
  if (scenario.fallback) {
    assert.match(meta('worker_fallback_reason'), /worker-src blob:/);
    assert.ok(result.fallbackMs < 2_000, 'bridge worker fallback was too slow');
  }
}

export function coreWithoutWorkerHook(source) {
  const start = source.indexOf(`if(${WORKER_HOOK}){`);
  const end = source.indexOf('else worker=new Worker(new URL(', start);
  assert.ok(start >= 0 && end > start, 'could not find the generated pthread hook');
  return source.slice(0, start) + source.slice(end + 'else '.length);
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
      const fetched = (wasmRequests.get(corePath) ?? 0) - initialRequests;
      await writeJsonArtifact(options.artifactsDir, `${name}.json`, { ...result, workers, wasmRequests: fetched, console });
      verifyScenario(name, options.memoryMode, result, workers, fetched, options.browser);
    } catch (error) {
      throw new Error(`${name}: ${error.message}\n${console.slice(-20).join('\n')}`, { cause: error });
    } finally {
      await page.close();
    }
  }, {
    isolated: !scenario.single,
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

    const requests = new Map();
    const recordRequest = (req) => {
      if (req.method === 'GET' && req.url.endsWith('.wasm')) {
        requests.set(req.url, (requests.get(req.url) ?? 0) + 1);
      }
    };
    const browserScenarios = options.browser === 'firefox' ? firefoxScenarios : Object.keys(scenarios);
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
