#!/usr/bin/env node

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import process from 'node:process';
import { gzipSync } from 'node:zlib';
import { chromium } from 'playwright';

import {
  DEFAULT_MODEL_CACHE, cachedModelName, env, isDirectory, parseSmokeArgs, resolvePath, resolvePinnedModel,
  runMain, webGpuLaunchArgs,
} from './support.mjs';

const CUT = 131072;
const RANGE = `bytes=${CUT}-`;
const MODEL = '/model.gguf';
const OTHER = '/other.gguf';
const RECOVERY = '/recovery.gguf';
const LAST_MODIFIED = 'Wed, 21 Oct 2015 07:28:00 GMT';
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'resume_integrity.mjs',
    description: 'Exercise streamed model resume with real cores and an HTTP Range server.',
    options: [
      { flag: '--dist-dir', type: 'path', default: () => env.path('BRIDGE_DIST_DIR', 'dist') },
      { flag: '--baseline-dist-dir', type: 'path' },
      { flag: '--threading', type: 'string', choices: ['default', 'single'], default: () => 'default' },
      { flag: '--js-backed-model-staging', type: 'flag', default: () => false },
      { flag: '--no-mmap', type: 'flag', default: () => false },
      { flag: '--model-path', type: 'path', default: () => env.optionalPath('LLAMA_WEBGPU_SMOKE_MODEL_PATH') },
      { flag: '--model-sha256', type: 'string', default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_SHA256') },
      { flag: '--model-cache-dir', type: 'path', default: () => env.path('LLAMA_WEBGPU_SMOKE_MODEL_CACHE', DEFAULT_MODEL_CACHE) },
      { flag: '--timeout-ms', type: 'int', default: () => 60000 },
    ],
  });
}

function makeRevisions(revA) {
  // The pinned fixture has equal-length pieces on opposite sides of the cut.
  const nameOffset = 101;
  const helloOffset = 323836;
  const capitalHelloOffset = 214286;
  const nameA = 'LLaMA v2';
  const nameB = 'BBBBBBBB';
  const hello = Buffer.from('▁hello');
  const capitalHello = Buffer.from('▁Hello');
  assert.equal(revA.toString('ascii', 0, 4), 'GGUF');
  assert.equal(revA.toString('utf8', nameOffset, nameOffset + nameA.length), nameA);
  assert.deepEqual(revA.subarray(helloOffset, helloOffset + hello.length), hello);
  assert.deepEqual(revA.subarray(capitalHelloOffset, capitalHelloOffset + capitalHello.length), capitalHello);
  const revB = Buffer.from(revA);
  revB.write(nameB, nameOffset);
  capitalHello.copy(revB, helloOffset);
  hello.copy(revB, capitalHelloOffset);
  const hybrid = Buffer.concat([revA.subarray(0, CUT), revB.subarray(CUT)]);
  return {
    revA, revB,
    expected: {
      A: { sha256: sha256(revA), name: nameA, token: 22172 },
      B: { sha256: sha256(revB), name: nameB, token: 15043 },
      hybrid: { sha256: sha256(hybrid), name: nameA, token: 15043 },
    },
  };
}

export function renderHarness({ threading, jsBackedModelStaging, noMmap }, { modelUrl, worker, recover, cacheReload = false }) {
  const single = threading === 'single';
  const config = {
    disableWorker: !worker,
    allowSingleThreadCore: single,
    jsBackedModelStaging,
    preferMemory64: false,
    coreModuleUrl: '/instrumented-core.js',
    wasmUrl: single ? '/llama_webgpu_core_st.wasm' : '/llama_webgpu_core.wasm',
  };
  return `<!doctype html>
<meta charset="utf-8">
<title>Stream resume integrity</title>
<script type="module">
import { LlamaWebGpuBridge } from '/llama_webgpu_bridge.js';

const channel = new BroadcastChannel('resume-integrity');
const staged = [];
channel.onmessage = ({ data }) => {
  staged.push(data);
  channel.postMessage('recorded');
};
const bridge = new LlamaWebGpuBridge(${JSON.stringify(config)});
const progress = [];
const options = { nCtx: 128, nThreads: 1, nGpuLayers: 0, useCache: ${cacheReload}${noMmap ? ', useMmap: false' : ''} };
const result = { crossOriginIsolated, sharedArrayBuffer: typeof SharedArrayBuffer, progress, staged };
try {
  let originalCore;
  if (${recover}) originalCore = await bridge._runtime._ensureCore();
  try {
    await bridge.loadModelFromUrl(${JSON.stringify(modelUrl)}, {
      ...options,
      progressCallback: (event) => progress.push(event),
    });
    if (${cacheReload}) {
      result.initialCache = bridge.getModelMetadata()['llamadart.webgpu.model_cache_state'];
      await bridge.loadModelFromUrl(${JSON.stringify(modelUrl)}, options);
    }
    const metadata = bridge.getModelMetadata();
    result.cache = metadata['llamadart.webgpu.model_cache_state'];
    result.name = metadata['general.name'];
    result.notes = (metadata['llamadart.webgpu.runtime_notes'] || '').split(';');
    result.execution = metadata['llamadart.webgpu.execution'];
    result.tokens = Array.from(await bridge.tokenize(' hello', false));
    result.ok = true;
  } catch (error) {
    result.ok = false;
    result.error = { name: error.name, message: error.message };
    result.notes = bridge._runtime?._runtimeNotes.slice() || [];
    if (${recover} && error.name === 'ModelTransferLengthError') {
      await bridge.loadModelFromUrl(${JSON.stringify(RECOVERY)}, options);
      result.recoveredSameCore = originalCore === await bridge._runtime._ensureCore()
        && (await bridge.tokenize(' hello', false)).includes(22172);
    }
  }
} catch (error) {
  result.harnessError = String(error.stack || error);
} finally {
  await bridge.dispose();
  channel.close();
  window.__smokeResult = result;
}
</script>
`;
}

function renderInstrumentedCore(coreName, deterministicProgress) {
  return `
import createCore from '/${coreName}.js';

if (${deterministicProgress}) {
  // Browser receipt boundaries vary. Use fixed-size reads for the exact
  // progress comparison while each request still goes to the real server.
  const fetchResponse = globalThis.fetch;
  globalThis.fetch = async (...args) => {
    const response = await fetchResponse(...args);
    if (!response.url.endsWith('/model.gguf') || !response.body) return response;
    const reader = response.body.getReader();
    const size = 65536;
    let pending = new Uint8Array();
    const body = new ReadableStream({
      async pull(controller) {
        while (pending.length < size) {
          const { value, done } = await reader.read();
          if (done) {
            if (pending.length) controller.enqueue(pending);
            controller.close();
            return;
          }
          const combined = new Uint8Array(pending.length + value.length);
          combined.set(pending);
          combined.set(value, pending.length);
          pending = combined;
        }
        controller.enqueue(pending.slice(0, size));
        pending = pending.subarray(size);
      },
      cancel(reason) { return reader.cancel(reason); },
    }, { highWaterMark: 0 });
    const wrapped = new Response(body, { status: response.status, headers: response.headers });
    Object.defineProperties(wrapped, {
      url: { value: response.url },
      type: { value: response.type },
    });
    return wrapped;
  };
}

export default async (options) => {
  const core = await createCore(options);
  const ccall = core.ccall.bind(core);
  const channel = new BroadcastChannel('resume-integrity');
  core.ccall = (name, type, types, args, opts) => {
    if (name !== 'llamadart_webgpu_load_model') return ccall(name, type, types, args, opts);
    const bytes = core.FS.readFile(args[0]);
    return crypto.subtle.digest('SHA-256', bytes).then(async (hash) => {
      const sha256 = Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, '0')).join('');
      const recorded = new Promise((resolve) => channel.addEventListener('message', resolve, { once: true }));
      channel.postMessage({ sha256, bytes: bytes.byteLength });
      await recorded;
      return ccall(name, type, types, args, opts);
    });
  };
  return core;
};
`;
}

const MIME = { '.js': 'text/javascript', '.wasm': 'application/wasm' };
const FULL = [[MODEL, null]];
const RESUME = [...FULL, [MODEL, RANGE]];
const RESTART = [...RESUME, ...FULL];
const REDIRECT = [...RESUME, [OTHER, RANGE], ...FULL];
const VALIDATED = 'model_stream_resume_validated';
const UNVALIDATED = 'model_stream_resume_unvalidated';

const CASES = {
  uninterrupted: { expected: 'A', requests: FULL },
  'no-validator': { cut: true, expected: 'A', requests: RESUME, note: UNVALIDATED },
  'always-cut-full-get': { cut: true, alwaysCut: true, expected: 'A', requests: RESUME, note: UNVALIDATED },
  'strong-etag-changed': { cut: true, initialTag: '"a"', laterTag: '"b"', newRevision: true,
    expected: 'B', requests: RESTART, note: 'model_stream_restart:etag' },
  'etag-restart-without-size': { cut: true, initialTag: '"a"', laterTag: '"b"', newRevision: true,
    chunkedReplacement: true, expected: 'B', requests: RESTART, note: 'model_stream_restart:etag' },
  'interrupted-range-overrun': { cut: true, initialTag: '"a"', laterTag: '"a"', overlongTail: true,
    expected: 'A', requests: RESTART, note: 'model_stream_restart:range' },
  'decoded-cache-reload': { cacheReload: true, expected: 'A', requests: FULL },
  'weak-etag-changed': { cut: true, initialTag: 'W/"a"', laterTag: 'W/"b"', newRevision: true,
    expected: 'hybrid', requests: RESUME, note: UNVALIDATED },
  'strong-etag-unchanged': { cut: true, initialTag: '"same"', laterTag: '"same"',
    expected: 'A', requests: RESUME, note: VALIDATED },
  redirect: { cut: true, redirect: true, initialTag: '"same"', laterTag: '"same"', newRevision: true,
    expected: 'B', requests: REDIRECT, note: 'model_stream_restart:url' },
  'wrong-start': { cut: true, malformed: 'start', expected: 'A', requests: RESTART, note: 'model_stream_restart:range' },
  'wrong-total': { cut: true, malformed: 'total', expected: 'A', requests: RESTART, note: 'model_stream_restart:range' },
  'contradictory-tail-length': { cut: true, shortTail: true, declareTailLength: true, initialTag: '"a"', laterTag: '"a"',
    expected: 'A', requests: RESTART, note: 'model_stream_restart:range' },
  'short-tail': { cut: true, shortTail: true, recover: true,
    requests: [...RESUME, [RECOVERY, null]], note: UNVALIDATED },
  'wrong-replacement-size': { cut: true, wrongReplacementSize: true, recover: true, initialTag: '"a"', laterTag: '"b"',
    requests: [...RESTART, [RECOVERY, null]], note: 'model_stream_restart:etag' },
  'mixed-case-range-unit': { cut: true, alwaysCut: true, rangeUnit: 'Bytes',
    expected: 'A', requests: RESUME, note: UNVALIDATED },
  'wildcard-range-total': { cut: true, alwaysCut: true, wildcardTotal: true,
    expected: 'A', requests: RESUME, note: UNVALIDATED },
  'cross-origin-etag-exposed-on-resume': { crossOrigin: true, exposeOnResume: true, cut: true, alwaysCut: true,
    initialTag: '"a"', laterTag: '"a"', expected: 'A', requests: RESUME, note: UNVALIDATED },
  'negotiated-gzip': { cut: true, alwaysCut: true, gzip: true,
    expected: 'A', requests: RESUME, note: UNVALIDATED },
  'cross-origin-negotiated-gzip': { crossOrigin: true, expose: true, cut: true, alwaysCut: true, gzip: true,
    initialTag: '"gzip"', laterTag: '"identity"', expected: 'A', requests: RESUME, note: UNVALIDATED },
  'cross-origin-exposed': { crossOrigin: true, exposeAll: true,
    cut: true, initialTag: '"a"', laterTag: '"a"',
    expected: 'A', requests: RESUME, note: VALIDATED },
  'cross-origin-strong-etag-changed': { crossOrigin: true, exposeAll: true, cut: true,
    initialTag: '"a"', laterTag: '"b"', newRevision: true,
    expected: 'B', requests: RESTART, note: 'model_stream_restart:etag' },
  'cross-origin-hidden': { crossOrigin: true, cut: true, initialTag: '"a"', laterTag: '"a"',
    expected: 'A', requests: RESUME, note: UNVALIDATED },
  'weak-etag-unchanged': { cut: true, initialTag: 'W/"same"', laterTag: 'W/"same"', newRevision: true,
    expected: 'hybrid', requests: RESUME, note: UNVALIDATED },
  'last-modified-unchanged': { cut: true, lastModified: true, newRevision: true,
    expected: 'hybrid', requests: RESUME, note: UNVALIDATED },
  'unquoted-etag-unchanged': { cut: true, initialTag: 'same', laterTag: 'same', newRevision: true,
    expected: 'hybrid', requests: RESUME, note: UNVALIDATED },
};
const CONTROLS = ['uninterrupted', 'no-validator', 'always-cut-full-get'];

function serveModel(req, res, requestPath, index, scenario, { revA, revB }) {
  const headers = { 'Content-Type': 'application/octet-stream', 'Cache-Control': 'no-store' };
  if (scenario.crossOrigin) {
    headers['Access-Control-Allow-Origin'] = '*';
    if (scenario.exposeAll) {
      headers['Access-Control-Expose-Headers'] = '*';
    } else if (scenario.expose || (scenario.exposeOnResume && req.headers.range)) {
      headers['Access-Control-Expose-Headers'] = 'ETag, Content-Range, Content-Encoding';
    }
  }
  if (req.method !== 'GET') {
    res.writeHead(403, headers).end();
    return;
  }
  if (requestPath === RECOVERY) {
    res.writeHead(200, { ...headers, 'Content-Length': revA.length }).end(revA);
    return;
  }

  const range = /^bytes=(\d+)-$/.exec(req.headers.range || '');
  if (scenario.redirect && range && requestPath === MODEL) {
    res.writeHead(302, { ...headers, Location: OTHER }).end();
    return;
  }

  const tag = index === 0 ? scenario.initialTag : scenario.laterTag;
  if (tag) headers.ETag = tag;
  if (scenario.lastModified) headers['Last-Modified'] = LAST_MODIFIED;
  let revision = index > 0 && scenario.newRevision ? revB : revA;
  if (scenario.redirect && requestPath === OTHER && !range) revision = revA;
  if (range && scenario.malformed === 'total') revision = Buffer.concat([revision, Buffer.alloc(4096)]);

  const start = range && scenario.malformed !== 'start' ? Number(range[1]) : 0;
  const shortTail = range && scenario.shortTail;
  const body = revision.subarray(start, shortTail ? start + 400000 : undefined);
  if (range) {
    const unit = scenario.rangeUnit ?? 'bytes';
    const total = scenario.wildcardTotal ? '*' : revision.length;
    headers['Content-Range'] = `${unit} ${start}-${revision.length - 1}/${total}`;
  }
  const chunkedReplacement = scenario.chunkedReplacement && !range && index > 0;
  if ((!shortTail || scenario.declareTailLength) && !chunkedReplacement && !(range && scenario.overlongTail)) {
    headers['Content-Length'] = body.length;
  }
  if (scenario.cacheReload) {
    const encoded = gzipSync(body);
    res.writeHead(200, { ...headers, 'Content-Encoding': 'gzip', 'Content-Length': encoded.length }).end(encoded);
    return;
  }
  if (scenario.wrongReplacementSize && !range && index > 0) headers['X-Linked-Size'] = revision.length - 1;
  if (scenario.gzip) {
    headers.Vary = 'Accept-Encoding';
    if (!range && /\bgzip\b/.test(req.headers['accept-encoding'] ?? '')) {
      const prefix = gzipSync(revision.subarray(0, CUT));
      headers['Content-Length'] = prefix.length + gzipSync(revision.subarray(CUT)).length;
      headers['Content-Encoding'] = 'gzip';
      res.writeHead(200, headers);
      res.write(prefix, () => setTimeout(() => res.destroy(), 150));
      return;
    }
  }
  res.writeHead(range ? 206 : 200, headers);
  if (range && scenario.overlongTail) {
    res.write(Buffer.concat([body, Buffer.alloc(CUT)]), () => setTimeout(() => res.destroy(), 150));
  } else if (!range && scenario.cut && (index === 0 || scenario.alwaysCut)) {
    res.write(revision.subarray(0, CUT), () => setTimeout(() => res.destroy(), 150));
  } else {
    res.end(body);
  }
}

async function withHttpServer(handler, run) {
  const server = http.createServer(handler);
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  try {
    return await run(`http://127.0.0.1:${server.address().port}`);
  } finally {
    await new Promise((resolve) => {
      server.close(resolve);
      server.closeAllConnections();
    });
  }
}

async function runCase(browser, args, revisions, name, { worker = false, parity = false } = {}) {
  const scenario = CASES[name];
  const single = args.threading === 'single';
  const coreName = single ? 'llama_webgpu_core_st' : 'llama_webgpu_core';
  const isolationHeaders = single ? {} : {
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  };
  const files = new Set([
    'llama_webgpu_bridge.js', 'llama_webgpu_bridge_worker.js', `${coreName}.js`, `${coreName}.wasm`,
  ]);
  const requests = [];
  const runModel = (req, res) => {
    const headers = { ...req.headers };
    delete headers.host;
    delete headers.referer;
    if (headers.origin) headers.origin = 'page';
    const index = requests.length;
    requests.push({ method: req.method, path: req.url, headers });
    serveModel(req, res, req.url, index, scenario, revisions);
  };

  const runPage = (modelOrigin) => withHttpServer((req, res) => {
    if (req.url === MODEL || req.url === OTHER || req.url === RECOVERY) {
      runModel(req, res);
      return;
    }
    if (req.url === '/index.html') {
      res.writeHead(200, { ...isolationHeaders, 'Content-Type': 'text/html' });
      res.end(renderHarness(args, { modelUrl: modelOrigin + MODEL, worker, recover: scenario.recover ?? false, cacheReload: scenario.cacheReload }));
      return;
    }
    if (req.url === '/instrumented-core.js') {
      res.writeHead(200, { ...isolationHeaders, 'Content-Type': 'text/javascript' });
      res.end(renderInstrumentedCore(coreName, parity));
      return;
    }
    const fileName = req.url.slice(1);
    if (!files.has(fileName)) {
      res.writeHead(404).end();
      return;
    }
    const file = fs.createReadStream(path.join(args.distDir, fileName));
    file.once('open', () => {
      res.writeHead(200, { ...isolationHeaders, 'Content-Type': MIME[path.extname(fileName)] });
      file.pipe(res);
    });
    file.once('error', () => res.destroy());
  }, async (origin) => {
    const page = await browser.newPage();
    try {
      await page.goto(origin + '/index.html', { timeout: args.timeoutMs });
      const result = await page.waitForFunction(() => window.__smokeResult, null, { timeout: args.timeoutMs });
      return { ...await result.jsonValue(), requests };
    } finally {
      await page.close();
    }
  });
  return scenario.crossOrigin ? withHttpServer(runModel, runPage) : runPage('');
}

function verifyCase(args, revisions, name, result, worker = false) {
  if (args.noMmap) {
    assert.ok(!result.notes?.includes('model_staging_heap_resident'), `${name}: useMmap:false staged into the heap`);
  }
  const scenario = CASES[name];
  const single = args.threading === 'single';
  assert.equal(result.harnessError, undefined);
  assert.equal(result.crossOriginIsolated, !single);
  assert.equal(result.sharedArrayBuffer, single ? 'undefined' : 'function');
  assert.deepEqual(result.requests.map(({ method, path, headers }) =>
    [method, path, headers.range ?? null, headers['if-range'] ?? null]),
  scenario.requests.map(([path, range]) => ['GET', path, range, null]));
  const notes = result.notes.filter((note) => note === VALIDATED || note === UNVALIDATED
    || note.startsWith('model_stream_restart:'));
  const expectedNotes = scenario.overlongTail ? [VALIDATED, scenario.note] : scenario.note ? [scenario.note] : [];
  assert.deepEqual(notes, expectedNotes);
  const rewinds = result.progress.filter(({ loaded }, index) => index > 0 && loaded < result.progress[index - 1].loaded);
  assert.deepEqual(rewinds.map(({ loaded }) => loaded), scenario.note?.startsWith('model_stream_restart:') ? [0] : []);
  if (scenario.chunkedReplacement && args.jsBackedModelStaging && !args.noMmap) {
    assert.ok(!result.notes.includes('model_staging_heap_resident'), 'the discarded heap file leaked its staging note');
    assert.ok(result.notes.includes('model_staging_heap_resident_unavailable:unknown_size'));
  }
  if (scenario.recover) {
    assert.equal(result.ok, false);
    assert.equal(result.error?.name, 'ModelTransferLengthError');
    assert.equal(result.recoveredSameCore, true);
    assert.deepEqual(result.staged, [{ sha256: revisions.expected.A.sha256, bytes: revisions.revA.length }]);
  } else {
    assert.equal(result.ok, true, JSON.stringify(result.error));
    assert.equal(result.execution, worker ? 'worker' : 'main-thread');
    assert.equal(result.notes.includes('model_staging_js_backed'), args.jsBackedModelStaging);
    const expected = revisions.expected[scenario.expected];
    const staged = { sha256: expected.sha256, bytes: revisions.revA.length };
    assert.deepEqual(result.staged, scenario.cacheReload ? [staged, staged] : [staged]);
    if (scenario.cacheReload) {
      assert.equal(result.initialCache, 'stored');
      assert.equal(result.cache, 'hit');
      assert.equal(result.notes.includes('model_staging_heap_resident'), args.jsBackedModelStaging && !args.noMmap);
    }
    assert.equal(result.name, expected.name);
    assert.deepEqual(result.tokens, [29871, expected.token]);
  }
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  args.distDir = resolvePath(args.distDir);
  assert.ok(isDirectory(args.distDir), `dist directory does not exist: ${args.distDir}`);
  const baseline = args.baselineDistDir && resolvePath(args.baselineDistDir);
  if (baseline) assert.ok(isDirectory(baseline), `baseline dist directory does not exist: ${baseline}`);
  const stateModelUrl = process.env.LLAMA_WEBGPU_SMOKE_MODEL_URL;
  const modelPath = args.modelPath ?? (stateModelUrl
    ? path.join(args.modelCacheDir, cachedModelName(stateModelUrl))
    : null);
  assert.ok(modelPath, '--model-path or the state smoke model cache is required');
  const model = await resolvePinnedModel({ ...args, modelPath });
  const revisions = makeRevisions(await fsp.readFile(model));
  const browser = await chromium.launch({
    args: webGpuLaunchArgs(process.platform, { sharedArrayBuffer: args.threading === 'default' }),
  });
  const results = {};
  let failed = false;
  try {
    for (const name of Object.keys(CASES)) {
      const result = await runCase(browser, args, revisions, name);
      results[name] = result;
      try {
        verifyCase(args, revisions, name, result);
      } catch (error) {
        failed = true;
        process.stderr.write(`${name}: ${error.message}\n`);
      }
    }
    if (baseline) {
      const comparable = (result) => ({
        ...result, notes: result.notes.filter((note) => note !== UNVALIDATED),
      });
      results.parity = Object.fromEntries(await Promise.all([false, true].map(async (worker) => {
        const parityCases = {};
        for (const name of CONTROLS) {
          const current = await runCase(browser, args, revisions, name, { worker, parity: true });
          verifyCase(args, revisions, name, current, worker);
          const prior = await runCase(browser, { ...args, distDir: baseline }, revisions, name, { worker, parity: true });
          assert.deepEqual(comparable(current), comparable(prior),
            `${name} request, bytes, notes or progress differs from baseline`);
          parityCases[name] = current;
        }
        return [worker ? 'worker' : 'direct', parityCases];
      })));
    }
  } finally {
    await browser.close();
  }
  process.stdout.write(`${JSON.stringify(results, null, 2)}\n`);
  return failed ? 1 : 0;
}

if (import.meta.main) await runMain('resume integrity', main);
