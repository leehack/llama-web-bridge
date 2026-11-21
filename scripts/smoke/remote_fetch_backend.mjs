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
const TRUNCATED_MODEL_PATH = '/mm/bad.gguf';
const MOUNT_PROBE_PATH = '/fetch-mount-probe';
const CHUNK_SIZE = 64 * 1024;
const LARGE_CHUNK_SIZE = 2 * CHUNK_SIZE;

export function repeatedLoads(modelPath) {
  return [
    { modelPath, chunkSize: CHUNK_SIZE },
    ...Array.from({ length: 5 }, (_, index) => {
      const signed = index % 2 === 0;
      const url = signed ? SIGNED_MODEL_PATH : MODEL_PATH;
      return {
        modelPath: `${url}${signed ? '&' : '?'}load=${index + 2}`,
        chunkSize: signed ? LARGE_CHUNK_SIZE : CHUNK_SIZE,
      };
    }),
  ];
}

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
  const mounts = [];
  const truncated = model.subarray(0, Math.min(1_000_000, Math.floor(model.length * 0.75)));
  const server = http.createServer(async (req, res) => {
    for (const [name, value] of ISOLATION_HEADERS) res.setHeader(name, value);
    try {
      const { method, url } = req;
      const { pathname, searchParams } = new URL(url, 'http://localhost');
      if (pathname.startsWith('/mm')) requests.push({ method, url, pathname, range: req.headers.range });
      if (method !== 'GET' && method !== 'HEAD') {
        res.writeHead(405, { Allow: 'GET, HEAD' }).end();
      } else if (pathname === MODEL_PATH) {
        sendModel(req, res, model);
      } else if (pathname === TRUNCATED_MODEL_PATH) {
        sendModel(req, res, truncated);
      } else if (pathname === MOUNT_PROBE_PATH) {
        mounts.push({
          core: searchParams.get('core'),
          url: searchParams.get('url'),
          files: searchParams.getAll('file'),
        });
        res.writeHead(204).end();
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
    return await run(`http://127.0.0.1:${server.address().port}/index.html`, requests, mounts);
  } finally {
    await new Promise((resolve) => {
      server.close(resolve);
      server.closeAllConnections();
    });
  }
}

export function assertModelRequests(requests, modelPath, chunkSize = CHUNK_SIZE) {
  const unexpected = requests.filter(({ url }) => url !== modelPath);
  assert.equal(unexpected.length, 0, `unexpected model URL in ${JSON.stringify(unexpected)}`);
  assert.ok(requests.some(({ method, range }) => method === 'HEAD' && range === 'bytes=0-'),
    'missing WasmFS HEAD with Range: bytes=0-');
  assert.ok(requests.some(({ method, range }) => method === 'GET' && range === `bytes=0-${chunkSize - 1}`),
    'missing WasmFS chunk GET');
}

export function renderTrackedCore(coreName) {
  return `import createCore from '/${coreName}.js';
export default async (options) => {
  const core = await createCore(options);
  const coreId = crypto.randomUUID();
  const ccall = core.ccall.bind(core);
  core.ccall = (name, ...args) => {
    const result = ccall(name, ...args);
    if (name !== 'llamadart_webgpu_load_model_from_url') return result;
    return Promise.resolve(result).then(async (rc) => {
      const url = new URL(args[2][0]);
      const query = new URLSearchParams({ core: coreId, url: url.pathname + url.search });
      for (const file of core.FS.readdir('/')) {
        if (/^remote_model_\\d+\\.gguf$/.test(file)) query.append('file', file);
      }
      await fetch('${MOUNT_PROBE_PATH}?' + query);
      return rc;
    });
  };
  return core;
};
`;
}

async function runBrowserCase(settings, disableWorker) {
  const result = { ok: false, crossOriginIsolated, loads: [] };
  let bridge;
  try {
    if (!crossOriginIsolated) throw new Error('Expected cross-origin isolation');
    const { LlamaWebGpuBridge } = await import('/llama_webgpu_bridge.js');
    const { occupiedPaths, loads, abortLoad, recoveryLoad, nThreads } = settings;
    const config = {
      disableWorker,
      preferMemory64: settings.memoryMode === 'wasm64',
      coreModuleUrl: occupiedPaths ? '/llama_webgpu_core.js' : '/tracked-core.js',
      wasmUrl: '/llama_webgpu_core.wasm',
      coreModuleUrlMem64: occupiedPaths ? '/llama_webgpu_core_mem64.js' : '/tracked-core-mem64.js',
      wasmUrlMem64: '/llama_webgpu_core_mem64.wasm',
    };
    let appCore;
    if (occupiedPaths) {
      const moduleUrl = config.preferMemory64 ? config.coreModuleUrlMem64 : config.coreModuleUrl;
      if (config.preferMemory64) config.wasmUrl = config.wasmUrlMem64;
      config.coreModuleFactory = async (options) => {
        const { default: createCore } = await import(moduleUrl);
        appCore = await createCore(options);
        for (const [index, file] of occupiedPaths.entries()) {
          appCore.FS.writeFile(file, Uint8Array.of(index + 1));
        }
        return appCore;
      };
    }
    bridge = new LlamaWebGpuBridge(config);
    const nativeLoad = ({ modelPath, chunkSize }) => bridge.loadModelFromUrl(location.origin + modelPath, {
      nCtx: 64, nThreads, nGpuLayers: 0, useCache: false,
      forceRemoteFetchBackend: true, remoteFetchChunkBytes: chunkSize,
    });
    const snapshot = async () => {
      const metadata = bridge.getModelMetadata();
      return {
        variant: metadata['llamadart.webgpu.core_variant'],
        execution: metadata['llamadart.webgpu.execution'],
        source: metadata['llamadart.webgpu.model_source'],
        notes: (metadata['llamadart.webgpu.runtime_notes'] || '').split(';'),
        nThreads: Number(metadata['llamadart.webgpu.n_threads']),
        retirements: bridge._coreRetirements,
        tokens: Array.from(await bridge.tokenize('Hello', true)),
      };
    };
    const complete = () => bridge.createCompletion('Hello', { nPredict: 1, temp: 0, seed: 1 });
    for (const [index, load] of loads.entries()) {
      console.log(`REMOTE_FETCH_SMOKE ${settings.scenario} load ${index + 1}/${loads.length}`);
      await nativeLoad(load);
      result.loads.push(await snapshot());
    }
    if (appCore) {
      result.occupiedContents = occupiedPaths.map((file) => Array.from(appCore.FS.readFile(file)));
      result.remoteModelFiles = appCore.FS.readdir('/').filter((file) => file.startsWith('remote_model_')).sort();
    }
    result.completion = await complete();
    if (abortLoad) {
      console.log(`REMOTE_FETCH_SMOKE ${settings.scenario} abort`);
      try {
        await nativeLoad(abortLoad);
      } catch (error) {
        result.abortMessage = error.message;
      }
      result.afterAbortRetirements = bridge._coreRetirements;
      if (!result.abortMessage || result.afterAbortRetirements !== 1) {
        throw new Error(`Expected a native core abort and one retirement, got ${result.abortMessage} and ${result.afterAbortRetirements}`);
      }
      await nativeLoad(recoveryLoad);
      result.recovery = await snapshot();
      result.recoveryCompletion = await complete();
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
  const run = settings.scenario === 'bridge-worker' ? `
const runBrowserCase = ${runBrowserCase.toString()};
window.__smokeResult = await runBrowserCase(settings, false);` : `
const worker = new Worker('/app_worker.mjs', { type: 'module' });
try {
  window.__smokeResult = await new Promise((resolve) => {
    worker.onmessage = ({ data }) => resolve(data);
    worker.onerror = (event) => resolve({ ok: false, error: event.message });
    worker.postMessage(settings);
  });
} finally {
  worker.terminate();
}`;
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
  assert.ok(model.length > 2 * LARGE_CHUNK_SIZE, 'Use a model larger than two chunks to trigger WasmFS range fetches');
  const abortLoad = { modelPath: TRUNCATED_MODEL_PATH, chunkSize: CHUNK_SIZE };
  const recoveryLoad = { modelPath: `${MODEL_PATH}?load=recovery`, chunkSize: LARGE_CHUNK_SIZE };
  const scenarios = [
    { scenario: 'app-worker', loads: repeatedLoads(MODEL_PATH), nThreads: 4, abortLoad, recoveryLoad },
    { scenario: 'bridge-worker', loads: repeatedLoads(SIGNED_MODEL_PATH), nThreads: 4, abortLoad, recoveryLoad },
    {
      scenario: 'app-worker-collision',
      loads: [{ modelPath: MODEL_PATH, chunkSize: CHUNK_SIZE }],
      nThreads: 1,
      occupiedPaths: ['/remote_model_1.gguf', '/remote_model_2.gguf'],
    },
  ];
  const results = await withTempDir('remote-fetch-backend-', async (webRoot) => {
    await copyBridgeArtifacts(args.distDir, webRoot);
    if (args.memoryMode === 'wasm64') await copyMemory64Artifacts(args.distDir, webRoot);
    await fsp.writeFile(path.join(webRoot, 'app_worker.mjs'), renderAppWorker());
    await fsp.writeFile(path.join(webRoot, 'tracked-core.js'), renderTrackedCore('llama_webgpu_core'));
    if (args.memoryMode === 'wasm64') {
      await fsp.writeFile(path.join(webRoot, 'tracked-core-mem64.js'), renderTrackedCore('llama_webgpu_core_mem64'));
    }
    return withModelRangeServer(webRoot, model, async (url, requests, mounts) => {
      const cases = {};
      for (const entry of scenarios) {
        requests.length = 0;
        mounts.length = 0;
        const { scenario } = entry;
        const settings = { ...entry, memoryMode: args.memoryMode };
        await fsp.writeFile(path.join(webRoot, 'index.html'), renderHarness(settings));
        const result = await runPollingPlaywright(url, args.timeoutMs, args.artifactsDir, {
          artifactPrefix: `remote-fetch-${args.memoryMode}-${scenario}`,
          stageMarker: 'REMOTE_FETCH_SMOKE',
          stagePrefix: scenario,
          transform(payload) {
            payload.requests = requests.slice();
            payload.mounts = mounts.slice();
          },
        }).catch((error) => ({ ok: false, error: String(error?.stack || error), requests: requests.slice() }));
        cases[scenario] = result;
      }
      return cases;
    });
  });

  const failures = [];
  const assertLoad = (load, { scenario, nThreads, occupiedPaths }, retirements, tokens) => {
    // The custom factory selects its core directly, outside variant detection.
    if (!occupiedPaths) assert.equal(load.variant, args.memoryMode);
    if (scenario === 'bridge-worker') assert.equal(load.execution, 'worker');
    assert.equal(load.source, 'network-fetch');
    assert.ok(load.notes.includes('model_source_fetch_backend'));
    assert.ok(!load.notes.some((note) => note.startsWith('model_fetch_backend_failed')));
    assert.equal(load.nThreads, nThreads);
    assert.equal(load.retirements, retirements);
    assert.ok(load.tokens.length > 0, 'tokenize returned no tokens');
    assert.deepEqual(load.tokens, tokens);
  };
  for (const entry of scenarios) {
    const { scenario, loads, occupiedPaths, abortLoad, recoveryLoad } = entry;
    const result = results[scenario];
    try {
      assert.equal(result.ok, true, [result.error, result.disposeError].filter(Boolean).join('\n'));
      assert.equal(result.crossOriginIsolated, true);
      assert.equal(result.loads.length, loads.length);
      const tokens = result.loads[0].tokens;
      for (const load of result.loads) assertLoad(load, entry, 0, tokens);
      assert.ok(result.completion?.length > 0, 'completion returned no text');
      if (occupiedPaths) {
        assert.deepEqual(result.occupiedContents, occupiedPaths.map((_, index) => [index + 1]));
        assert.deepEqual(result.remoteModelFiles, occupiedPaths.map((file) => path.basename(file)));
      }
      const expectedRequests = [...loads];
      if (abortLoad) {
        assert.match(result.abortMessage, /^Aborted\(/);
        assert.equal(result.afterAbortRetirements, 1);
        assertLoad(result.recovery, entry, 1, tokens);
        assert.equal(result.recoveryCompletion, result.completion);
        assert.deepEqual(result.mounts.map(({ url }) => url),
          [...loads, recoveryLoad].map(({ modelPath }) => modelPath));
        const originalCore = result.mounts[0].core;
        assert.ok(originalCore);
        for (const mount of result.mounts.slice(0, -1)) assert.equal(mount.core, originalCore);
        assert.ok(result.mounts.at(-1).core);
        assert.notEqual(result.mounts.at(-1).core, originalCore);
        for (const mount of result.mounts) assert.deepEqual(mount.files, []);
        expectedRequests.push(abortLoad, recoveryLoad);
      }
      const expectedPaths = new Set(expectedRequests.map(({ modelPath }) => modelPath));
      assert.ok(result.requests.every(({ url }) => expectedPaths.has(url)), 'unexpected model URL');
      for (const { modelPath, chunkSize } of expectedRequests) {
        assertModelRequests(result.requests.filter(({ url }) => url === modelPath), modelPath, chunkSize);
      }
    } catch (error) {
      failures.push(`${scenario}: ${error.message}\nRequests: ${JSON.stringify(result.requests)}`);
    }
  }
  assert.equal(failures.length, 0, failures.join('\n'));
  assert.deepEqual(results['bridge-worker'].loads[0].tokens, results['app-worker'].loads[0].tokens);
  assert.deepEqual(results['app-worker-collision'].loads[0].tokens, results['app-worker'].loads[0].tokens);
  console.log(JSON.stringify({ ok: true, memoryMode: args.memoryMode, scenarios: Object.keys(results) }));
}

if (import.meta.main) await runMain('remote fetch backend', main);
