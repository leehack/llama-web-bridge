#!/usr/bin/env node

import assert from 'node:assert/strict';
import fsp from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';

import {
  DEFAULT_MODEL_CACHE, ISOLATION_HEADERS, contentType, copyBridgeArtifacts, copyMemory64Artifacts,
  env, isFile, parseSmokeArgs, resolvePinnedModel, runMain, runPollingPlaywright, translatePath,
  withTempDir,
} from './support.mjs';

const MODEL_PATH = '/mm/tiny.gguf';
const SIGNED_MODEL_PATH = `${MODEL_PATH}?sig=a/b&token=${'x'.repeat(280)}`;
const CHUNK_SIZE = 64 * 1024;

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'remote_fetch_backend.mjs',
    description: 'Load a GGUF through the WasmFS range backend in both Chromium worker modes.',
    options: [
      { flag: '--dist-dir', type: 'path', default: () => env.path('BRIDGE_DIST_DIR', 'dist') },
      { flag: '--memory-mode', type: 'string', choices: ['wasm32', 'wasm64'], default: () => 'wasm32' },
      { flag: '--model-path', type: 'path', default: () => env.optionalPath('LLAMA_WEBGPU_SMOKE_MODEL_PATH') },
      { flag: '--model-url', type: 'string', default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_URL') },
      { flag: '--model-sha256', type: 'string', default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_SHA256') },
      { flag: '--model-cache-dir', type: 'path', default: () => env.path('LLAMA_WEBGPU_SMOKE_MODEL_CACHE', DEFAULT_MODEL_CACHE) },
      { flag: '--artifacts-dir', type: 'path' },
      { flag: '--timeout-ms', type: 'int', default: () => 90_000 },
    ],
  });
}

function sendModel(req, res, model) {
  const headers = { 'Accept-Ranges': 'bytes', 'Content-Type': 'application/octet-stream' };
  const { range } = req.headers;
  if (req.method === 'HEAD' || range === undefined) {
    res.writeHead(200, { ...headers, 'Content-Length': model.length });
    res.end(req.method === 'HEAD' ? undefined : model);
    return;
  }

  const match = /^bytes=(\d+)-(\d*)$/.exec(range);
  const start = Number(match?.[1]);
  const end = Math.min(match?.[2] ? Number(match[2]) : model.length - 1, model.length - 1);
  if (!match || start > end) {
    res.writeHead(416, { ...headers, 'Content-Range': `bytes */${model.length}` });
    res.end();
    return;
  }

  res.writeHead(206, {
    ...headers,
    'Content-Range': `bytes ${start}-${end}/${model.length}`,
    'Content-Length': end - start + 1,
  });
  res.end(model.subarray(start, end + 1));
}

export async function withModelRangeServer(webRoot, model, run) {
  const requests = [];
  const server = http.createServer(async (req, res) => {
    for (const [name, value] of ISOLATION_HEADERS) res.setHeader(name, value);
    try {
      const { method, url } = req;
      const { pathname } = new URL(url, 'http://localhost');
      if (pathname.startsWith('/mm')) requests.push({ method, url, pathname, range: req.headers.range });
      if (method !== 'GET' && method !== 'HEAD') {
        res.writeHead(405, { Allow: 'GET, HEAD' }).end();
      } else if (pathname === MODEL_PATH) {
        sendModel(req, res, model);
      } else {
        const file = translatePath(webRoot, pathname);
        if (pathname.startsWith('/mm') || !isFile(file)) {
          res.writeHead(404).end();
          return;
        }
        const body = await fsp.readFile(file);
        res.writeHead(200, { 'Content-Type': contentType(file), 'Content-Length': body.length });
        res.end(method === 'HEAD' ? undefined : body);
      }
    } catch (error) {
      res.destroy(error);
    }
  });

  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  try {
    return await run(`http://127.0.0.1:${server.address().port}/index.html`, requests);
  } finally {
    await new Promise((resolve) => {
      server.close(resolve);
      server.closeAllConnections();
    });
  }
}

export function assertModelRequests(requests, modelPath) {
  const unexpected = requests.filter(({ url }) => url !== modelPath);
  assert.equal(unexpected.length, 0, `unexpected model URL in ${JSON.stringify(unexpected)}`);
  assert.ok(requests.some(({ method, range }) => method === 'HEAD' && range === 'bytes=0-'),
    'missing WasmFS HEAD with Range: bytes=0-');
  assert.ok(requests.some(({ method, range }) => method === 'GET' && range === `bytes=0-${CHUNK_SIZE - 1}`),
    'missing WasmFS chunk GET');
}

async function runBrowserCase(settings, disableWorker) {
  const result = { ok: false, crossOriginIsolated };
  let bridge;
  try {
    if (!crossOriginIsolated) throw new Error('Expected cross-origin isolation');
    const { LlamaWebGpuBridge } = await import('/llama_webgpu_bridge.js');
    bridge = new LlamaWebGpuBridge({
      disableWorker,
      preferMemory64: settings.memoryMode === 'wasm64',
      coreModuleUrl: '/llama_webgpu_core.js',
      wasmUrl: '/llama_webgpu_core.wasm',
      coreModuleUrlMem64: '/llama_webgpu_core_mem64.js',
      wasmUrlMem64: '/llama_webgpu_core_mem64.wasm',
    });
    await bridge.loadModelFromUrl(location.origin + settings.modelPath, {
      nCtx: 64, nThreads: 1, nGpuLayers: 0, useCache: false,
      forceRemoteFetchBackend: true, remoteFetchChunkBytes: settings.chunkSize,
    });
    const metadata = bridge.getModelMetadata();
    result.variant = metadata['llamadart.webgpu.core_variant'];
    result.execution = metadata['llamadart.webgpu.execution'];
    result.source = metadata['llamadart.webgpu.model_source'];
    result.notes = (metadata['llamadart.webgpu.runtime_notes'] || '').split(';');
    result.tokens = Array.from(await bridge.tokenize('Hello', true));
    if (disableWorker) {
      result.completion = await bridge.createCompletion('Hello', { nPredict: 1, temp: 0, seed: 1 });
    }
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
  }
  return result;
}

export function renderAppWorker() {
  return `const runBrowserCase = ${runBrowserCase.toString()};
self.onmessage = async ({ data: settings }) => self.postMessage(await runBrowserCase(settings, true));
`;
}

export function renderHarness(settings) {
  const run = settings.scenario === 'app-worker' ? `
const worker = new Worker('/app_worker.mjs', { type: 'module' });
try {
  window.__smokeResult = await new Promise((resolve) => {
    worker.onmessage = ({ data }) => resolve(data);
    worker.onerror = (event) => resolve({ ok: false, error: event.message });
    worker.postMessage(settings);
  });
} finally {
  worker.terminate();
}` : `
const runBrowserCase = ${runBrowserCase.toString()};
window.__smokeResult = await runBrowserCase(settings, false);`;
  return `<!doctype html>
<meta charset="utf-8">
<title>Remote fetch backend</title>
<script type="module">
const settings = ${JSON.stringify(settings)};
window.__smokeStage = settings.scenario;
${run}
</script>
`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const model = await fsp.readFile(await resolvePinnedModel(args));
  assert.ok(model.length > 2 * CHUNK_SIZE, 'Use a model larger than two chunks to trigger WasmFS range fetches');
  const results = await withTempDir('remote-fetch-backend-', async (webRoot) => {
    await copyBridgeArtifacts(args.distDir, webRoot);
    if (args.memoryMode === 'wasm64') await copyMemory64Artifacts(args.distDir, webRoot);
    await fsp.writeFile(path.join(webRoot, 'app_worker.mjs'), renderAppWorker());
    return withModelRangeServer(webRoot, model, async (url, requests) => {
      const cases = {};
      for (const [scenario, modelPath] of [['app-worker', MODEL_PATH], ['bridge-worker', SIGNED_MODEL_PATH]]) {
        requests.length = 0;
        const settings = { scenario, modelPath, memoryMode: args.memoryMode, chunkSize: CHUNK_SIZE };
        await fsp.writeFile(path.join(webRoot, 'index.html'), renderHarness(settings));
        const result = await runPollingPlaywright(url, args.timeoutMs, args.artifactsDir, {
          artifactPrefix: `remote-fetch-${args.memoryMode}-${scenario}`,
          stagePrefix: scenario,
          transform(payload) { payload.requests = requests.slice(); },
        }).catch((error) => ({ ok: false, error: String(error?.stack || error), requests: requests.slice() }));
        cases[scenario] = result;
      }
      return cases;
    });
  });

  const failures = [];
  for (const [scenario, result] of Object.entries(results)) {
    const modelPath = scenario === 'app-worker' ? MODEL_PATH : SIGNED_MODEL_PATH;
    try {
      assert.equal(result.ok, true, [result.error, result.disposeError].filter(Boolean).join('\n'));
      assert.equal(result.crossOriginIsolated, true);
      assert.equal(result.variant, args.memoryMode);
      if (scenario === 'bridge-worker') assert.equal(result.execution, 'worker');
      assert.equal(result.source, 'network-fetch');
      assert.ok(result.notes.includes('model_source_fetch_backend'));
      assert.ok(!result.notes.some((note) => note.startsWith('model_fetch_backend_failed')));
      assert.ok(result.tokens.length > 0, 'tokenize returned no tokens');
      if (scenario === 'app-worker') assert.equal(typeof result.completion, 'string');
      assertModelRequests(result.requests, modelPath);
    } catch (error) {
      failures.push(`${scenario}: ${error.message}\nRequests: ${JSON.stringify(result.requests)}`);
    }
  }
  assert.equal(failures.length, 0, failures.join('\n'));
  assert.deepEqual(results['bridge-worker'].tokens, results['app-worker'].tokens);
  console.log(JSON.stringify({ ok: true, memoryMode: args.memoryMode, scenarios: Object.keys(results) }));
}

if (import.meta.main) await runMain('remote fetch backend', main);
