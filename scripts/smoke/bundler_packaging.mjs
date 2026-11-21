#!/usr/bin/env node

import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import fsp from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { promisify } from 'node:util';

import { build as esbuild } from 'esbuild';
import { chromium, firefox } from 'playwright';

import { ARTIFACTS } from '../release/manifest.mjs';
import { FIXTURE_RELEASE_TAG, verifyTarball } from '../release/npm_package.mjs';
import {
  DEFAULT_MODEL_CACHE, ISOLATION_HEADERS, env, isFile, parseSmokeArgs, resolvePinnedModel, runMain,
  serveIsolated, sha256File, stageFile, translatePath, withTempDir,
} from './support.mjs';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PACKAGE = 'llama-web-bridge';
const VERSION = FIXTURE_RELEASE_TAG.slice(1);
const CDN_PATH = `/npm/${PACKAGE}@${VERSION}/dist`;
const BRIDGE = 'llama_webgpu_bridge.js';
const WORKER = 'llama_webgpu_bridge_worker.js';
const SUITES = ['vite', 'webpack', 'esbuild', 'cdn'];
const DEPENDENCIES = createRequire(new URL('./bundler_deps/package.json', import.meta.url));
const run = promisify(execFile);

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'bundler_packaging.mjs',
    description: 'Pack the built bridge and exercise it in Vite, webpack, esbuild and native browser ESM.',
    options: [
      { flag: '--dist-dir', type: 'path', default: () => env.path('BRIDGE_DIST_DIR', 'dist') },
      { flag: '--model-path', type: 'path' },
      { flag: '--model-url', type: 'string', default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_URL') },
      { flag: '--model-sha256', type: 'string', default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_SHA256') },
      { flag: '--model-cache-dir', type: 'path', default: () => env.path('LLAMA_WEBGPU_SMOKE_MODEL_CACHE', DEFAULT_MODEL_CACHE) },
      { flag: '--suite', type: 'string', choices: SUITES },
      { flag: '--browser', type: 'string', choices: ['chromium', 'firefox'], default: () => 'chromium' },
      { flag: '--timeout-ms', type: 'int', default: () => 45_000 },
    ],
  });
}

export function renderHarness(specifier) {
  return `
import { LlamaWebGpuBridge } from ${JSON.stringify(specifier)};

const { options, modelUrl, documentBase } = globalThis.smokeInput;
if (documentBase) {
  const base = document.createElement('base');
  base.href = documentBase;
  document.head.prepend(base);
}
const result = { isolated: crossOriginIsolated };
const bridge = new LlamaWebGpuBridge({ logLevel: 0, ...options });
try {
  await bridge.loadModelFromUrl(modelUrl, { nCtx: 128, nThreads: 4, nGpuLayers: 0, useCache: false });
  await bridge.createCompletion('The capital of France is', {
    nPredict: 4, temp: 0, seed: 1,
    onUsage: ({ completionTokens }) => { result.tokens = completionTokens; },
  });
  const meta = bridge.getModelMetadata();
  result.execution = meta['llamadart.webgpu.execution'];
  result.threads = meta['llamadart.webgpu.n_threads'];
  result.memory = meta['llamadart.webgpu.core_variant'];
  result.notes = meta['llamadart.webgpu.runtime_notes'];
  result.fallback = meta['llamadart.webgpu.worker_fallback_reason'];
} catch (error) {
  result.error = String(error);
} finally {
  await bridge.dispose();
  globalThis.smokeResult = result;
}
`;
}

export function verifyScenario(scenario, result, requests, errors) {
  assert.ok(errors.length === 0, [...new Set(errors)].join('\n'));
  assert.equal(result.error, undefined, result.error);
  assert.equal(result.isolated, scenario.isolated, 'cross-origin isolation');
  assert.ok(result.tokens > 0, 'no completion tokens');
  assert.equal(result.execution, scenario.execution, 'bridge execution');
  assert.equal(result.threads, scenario.isolated ? '4' : '1', 'model thread count');
  assert.equal(result.memory, scenario.memory, 'core memory mode');
  if (scenario.fallback) assert.match(result.fallback ?? '', /^Bridge worker crashed$/);
  else assert.equal(result.fallback, undefined, 'unexpected worker fallback');
  const notes = (result.notes ?? '').split(';');
  assert.equal(notes.find((note) => note.startsWith('worker_start:')), scenario.workerStart && `worker_start:${scenario.workerStart}`);
  assert.equal(notes.find((note) => note.startsWith('pthread_start:')), scenario.pthreadStart && `pthread_start:${scenario.pthreadStart}`);
  assert.equal(notes.includes('core_single_thread'), !scenario.isolated, 'single-thread core selection');
  if (scenario.workerStart) {
    const expected = new URL(scenario.workerUrl);
    assert.ok(requests.some(({ url, dest }) => {
      if (scenario.workerStart === 'direct' && dest !== 'worker') return false;
      const actual = new URL(url);
      return actual.origin === expected.origin && actual.pathname === expected.pathname
        && (!expected.search || actual.search === expected.search);
    }), `expected worker entry ${expected.href}`);
  }
  if (scenario.coreUrl) {
    assert.ok(requests.some(({ url }) => url === scenario.coreUrl), `expected explicit core URL ${scenario.coreUrl}`);
  }
  if (scenario.wasmUrl) {
    assert.ok(requests.some(({ url }) => url === scenario.wasmUrl), `expected explicit Wasm URL ${scenario.wasmUrl}`);
  }

  const core = `llama_webgpu_core${scenario.memory === 'wasm64' ? '_mem64' : ''}${scenario.isolated ? '' : '_st'}`;
  const asset = (extension) => scenario.hashed
    ? new RegExp(`^(?:${core}-[\\w-]+|[a-f0-9]{8,})\\.${extension}$`)
    : new RegExp(`^${core}\\.${extension}$`);
  const inBase = requests.filter(({ url }) => url.startsWith(`${scenario.expectedBase}/`));
  const names = inBase.map(({ url, dest }) => ({ url, dest, file: path.posix.basename(new URL(url).pathname) }));
  assert.ok(names.some(({ file }) => asset('wasm').test(file)), `expected Wasm under ${scenario.expectedBase}`);
  assert.ok(names.some(({ file }) => asset('js').test(file)), `expected core script under ${scenario.expectedBase}`);
  if (scenario.hashed && scenario.isolated) {
    assert.ok(names.some(({ url, dest, file }) => dest === 'worker' && asset('js').test(file)
      && names.some((entry) => entry.url === url && entry.dest !== 'worker')), 'expected the same hashed core URL for pthread startup');
  }
}

async function preparePackage(temp, distDir) {
  const fixtureDist = path.join(temp, 'fixture');
  await fsp.mkdir(fixtureDist);
  const records = await Promise.all(ARTIFACTS.map(async (name) => {
    const target = path.join(fixtureDist, name);
    await fsp.copyFile(path.join(distDir, name), target);
    return [name, { sha256: await sha256File(target), size_bytes: (await fsp.stat(target)).size }];
  }));
  const manifest = {
    test_fixture: true,
    release_tag: FIXTURE_RELEASE_TAG,
    release_channel: 'stable',
    release_rebuild: 0,
    artifacts: Object.fromEntries(records),
  };
  await Promise.all([
    fsp.writeFile(path.join(fixtureDist, 'manifest.json'), JSON.stringify(manifest)),
    fsp.writeFile(path.join(fixtureDist, 'sha256sums.txt'), records.map(([name, { sha256 }]) => `${sha256}  ${name}\n`).join('')),
  ]);
  const staged = path.join(temp, 'package');
  await run(process.execPath, [path.join(ROOT, 'scripts/release/npm_package.mjs'), 'stage',
    '--dist-dir', fixtureDist, '--out-dir', staged, '--release-tag', FIXTURE_RELEASE_TAG, '--release-rebuild', '0',
    '--license', path.join(ROOT, 'LICENSE'), '--readme', path.join(ROOT, 'README.md'), '--allow-test-fixture']);
  const { stdout } = await run('npm', ['pack', '--ignore-scripts', '--json', staged], { cwd: temp });
  const [{ filename }] = JSON.parse(stdout);
  const tarball = path.join(temp, filename);
  const proof = verifyTarball({ tarball, distDir: fixtureDist, expectedVersion: VERSION });
  console.log(`Packed ${proof.file_count} files, ${(proof.packed_bytes / 2 ** 20).toFixed(2)} MiB`);

  const app = path.join(temp, 'app');
  await fsp.mkdir(app);
  await run('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund', '--offline',
    '--package-lock=false', '--no-save', tarball], { cwd: app });
  return { app, installed: path.join(app, 'node_modules', PACKAGE, 'dist') };
}

function html(script) {
  return `<!doctype html><meta charset="utf-8"><title>Bridge bundle smoke</title>
<link rel="icon" href="data:,"><script type="module" src="${script}"></script>\n`;
}

async function exposeDist(root, relativePath, installed) {
  const target = path.join(root, relativePath);
  await fsp.mkdir(path.dirname(target), { recursive: true });
  await fsp.symlink(installed, target, 'dir');
}

async function emittedWorker(output, installed) {
  const source = await fsp.readFile(path.join(installed, WORKER));
  const scripts = (await fsp.readdir(output, { recursive: true })).filter((name) => name.endsWith('.js'));
  for (const script of scripts) {
    if (source.equals(await fsp.readFile(path.join(output, script)))) return `/${script}`;
  }
  throw new Error('bundler did not emit the standalone worker');
}

async function buildVite(app) {
  const vite = await import(pathToFileURL(DEPENDENCIES.resolve('vite')));
  const config = { root: app, configFile: false, logLevel: 'warn' };
  const output = path.join(app, 'vite-build');
  await vite.build({ ...config, build: { outDir: 'vite-build' } });
  return { vite, config, output };
}

async function buildWebpack(app, installed) {
  const output = path.join(app, 'webpack-build');
  const config = path.join(app, 'webpack.config.cjs');
  await fsp.writeFile(config, `module.exports = ${JSON.stringify({
    mode: 'production',
    entry: './app.js',
    output: { path: output, filename: 'app.js', publicPath: 'auto' },
    stats: { all: false, assets: true, errors: true },
  })};\n`);
  const { stdout } = await run(process.execPath, [DEPENDENCIES.resolve('webpack-cli/bin/cli.js'),
    '--config', config, '--json'], {
    cwd: app,
    env: { ...process.env, WEBPACK_PACKAGE: DEPENDENCIES.resolve('webpack') },
  }).catch((error) => { throw new Error(`${error.message}${error.stdout}`, { cause: error }); });
  const workerAsset = JSON.parse(stdout).assets.find(({ info }) => info?.sourceFilename?.endsWith(`/${WORKER}`));
  assert.ok(workerAsset, 'webpack did not emit the standalone worker');
  await Promise.all([
    fsp.writeFile(path.join(output, 'index.html'), html('/app.js')),
    exposeDist(output, 'stock', installed),
    exposeDist(output, 'document/stock', installed),
  ]);
  return { output, workerPath: `/${workerAsset.name}` };
}

async function buildEsbuild(app, installed, copyAssets) {
  const output = path.join(app, copyAssets ? 'esbuild-copy' : 'esbuild-base');
  const plugins = copyAssets ? [{
    name: 'copy-bridge-siblings',
    setup(build) {
      build.onEnd(async ({ errors }) => {
        if (errors.length > 0) return;
        await Promise.all(ARTIFACTS.filter((name) => name !== BRIDGE && !name.endsWith('.d.ts'))
          .map((name) => fsp.copyFile(path.join(installed, name), path.join(output, name))));
      });
    },
  }] : [];
  await esbuild({
    absWorkingDir: app, entryPoints: ['app.js'], bundle: true, format: 'esm',
    outdir: output, plugins,
  });
  await fsp.writeFile(path.join(output, 'index.html'), html('/app.js'));
  return output;
}

function collectRequest(requests, req, origin, status) {
  const url = new URL(req.url, origin).href;
  requests.push({ url, dest: req.headers['sec-fetch-dest'], status });
}

async function staticServer(root, requests, options) {
  const server = await serveIsolated(root, {
    ...options,
    onRequest(req) {
      const file = translatePath(root, req.url);
      const found = isFile(file) || isFile(path.join(file, 'index.html'));
      collectRequest(requests, req, `http://${req.headers.host}`, found ? 200 : 404);
    },
  });
  return { ...server, origin: new URL(server.url).origin };
}

async function exercise(browser, requests, server, modelUrl, timeout, name, config = {}) {
  const {
    isolated = true, memory = 'wasm64', options = {}, expectedBase = server.origin,
    hashed = false, crossOriginWorker = false, crossOriginCore = false, browserName,
    documentBase, importUrl, workerPath,
  } = config;
  const fallback = crossOriginWorker && browserName === 'firefox';
  const execution = options.disableWorker || fallback ? 'main-thread' : 'worker';
  const workerStart = execution === 'worker' ? (crossOriginWorker ? 'blob' : 'direct') : undefined;
  const pthreadStart = isolated ? (crossOriginCore ? 'blob' : 'direct') : undefined;
  const workerUrl = options.workerUrl ?? (workerPath ? `${server.origin}${workerPath}` : `${expectedBase}/${WORKER}`);
  const coreUrl = memory === 'wasm64' ? options.coreModuleUrlMem64 : options.coreModuleUrl;
  const wasmUrl = memory === 'wasm64' ? options.wasmUrlMem64 : options.wasmUrl;
  const cursor = requests.length;
  const errors = [];
  const page = await browser.newPage();
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('response', (response) => {
    if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`);
  });
  const started = performance.now();
  try {
    await page.addInitScript((input) => { globalThis.smokeInput = input; }, { options, modelUrl, documentBase });
    await page.goto(server.url, { timeout });
    const result = await (await page.waitForFunction(() => globalThis.smokeResult, undefined, { timeout })).jsonValue();
    const observed = requests.slice(cursor);
    for (const { status, url } of observed) if (status >= 400) errors.push(`${status} ${url}`);
    if (importUrl) assert.ok(observed.some(({ url }) => url === importUrl), 'expected a cross-origin module import');
    verifyScenario({
      isolated, memory, execution, expectedBase, hashed, pthreadStart, workerStart, workerUrl, coreUrl, wasmUrl, fallback,
    }, result, observed, errors);
    console.log(`PASS ${name} (${execution}, ${result.threads} threads, ${Math.round(performance.now() - started)} ms)`);
  } catch (error) {
    throw new Error(`${name}: ${error.message}\n${JSON.stringify(requests.slice(cursor))}`, { cause: error });
  } finally {
    await page.close();
  }
}

async function withStatic(root, requests, options, callback) {
  const server = await staticServer(root, requests, options);
  try {
    await callback(server);
  } finally {
    await server.close();
  }
}

async function runVite(app, installed, requests, testCase) {
  const { vite, config, output } = await buildVite(app);
  const workerPath = await emittedWorker(output, installed);
  await withStatic(output, requests, {}, async (server) => {
    await testCase(server, 'vite production worker', { hashed: true, workerPath });
    await testCase(server, 'vite production direct', { hashed: true, options: { disableWorker: true } });
  });
  await withStatic(output, requests, { isolated: false }, (server) =>
    testCase(server, 'vite production single-thread', {
      hashed: true, workerPath, isolated: false, options: { allowSingleThreadCore: true },
    }));

  const dev = await vite.createServer({
    ...config,
    optimizeDeps: { exclude: [PACKAGE] },
    server: { host: '127.0.0.1', port: 0, headers: Object.fromEntries(ISOLATION_HEADERS) },
    plugins: [{
      name: 'record-smoke-requests',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          res.once('finish', () => collectRequest(requests, req, `http://${req.headers.host}`, res.statusCode));
          next();
        });
      },
    }],
  });
  try {
    await dev.listen();
    const origin = new URL(dev.resolvedUrls.local[0]).origin;
    await testCase({ origin, url: `${origin}/index.html` }, 'vite dev worker', {
      workerPath: `/node_modules/${PACKAGE}/dist/${WORKER}`,
    });
  } finally {
    await dev.close();
  }
}

async function runWebpack(app, installed, cdnBase, requests, testCase) {
  const { output, workerPath } = await buildWebpack(app, installed);
  await withStatic(output, requests, {}, async (server) => {
    for (const disableWorker of [false, true]) {
      const route = disableWorker ? 'direct' : 'worker';
      await testCase(server, `webpack zero-config ${route}`, { hashed: true, workerPath, options: { disableWorker } });
      await testCase(server, `webpack relative assetBaseUrl ${route}`, {
        documentBase: '/document/', expectedBase: `${server.origin}/document/stock`,
        options: { assetBaseUrl: './stock', disableWorker },
      });
      await testCase(server, `webpack absolute assetBaseUrl ${route}`, {
        expectedBase: `${server.origin}/stock`,
        options: { assetBaseUrl: `${server.origin}/stock/`, disableWorker },
      });
      await testCase(server, `webpack cross-origin assetBaseUrl ${route}`, {
        expectedBase: cdnBase, crossOriginCore: true, crossOriginWorker: !disableWorker,
        options: { assetBaseUrl: cdnBase, disableWorker },
      });
    }
    await testCase(server, 'webpack cross-origin core with same-origin worker', {
      expectedBase: cdnBase, crossOriginCore: true,
      options: { assetBaseUrl: cdnBase, workerUrl: `${server.origin}/stock/${WORKER}` },
    });
    const stock = `${server.origin}/stock`;
    await testCase(server, 'webpack explicit coreModuleUrl with implicit Wasm', {
      memory: 'wasm32', expectedBase: stock,
      options: { coreModuleUrl: `${stock}/llama_webgpu_core.js?v=${VERSION}`, disableWorker: true },
    });
    await testCase(server, 'webpack explicit llamadart worker and core URLs', {
      expectedBase: stock,
      options: {
        workerUrl: `${stock}/${BRIDGE}?__llamadartBridgeWorker=1`,
        coreModuleUrl: `${stock}/llama_webgpu_core.js?v=${VERSION}`,
        wasmUrl: `${stock}/llama_webgpu_core.wasm?v=${VERSION}`,
        coreModuleUrlMem64: `${stock}/llama_webgpu_core_mem64.js?v=${VERSION}`,
        wasmUrlMem64: `${stock}/llama_webgpu_core_mem64.wasm?v=${VERSION}`,
      },
    });
  });
  await withStatic(output, requests, { isolated: false }, (server) =>
    testCase(server, 'webpack production single-thread', {
      hashed: true, workerPath, isolated: false, options: { allowSingleThreadCore: true },
    }));
}

async function runEsbuild(app, installed, cdnBase, requests, testCase) {
  const baseOutput = await buildEsbuild(app, installed, false);
  await withStatic(baseOutput, requests, {}, (server) => testCase(server, 'esbuild cross-origin assetBaseUrl', {
    expectedBase: cdnBase, crossOriginCore: true, crossOriginWorker: true, options: { assetBaseUrl: cdnBase },
  }));
  const copyOutput = await buildEsbuild(app, installed, true);
  await withStatic(copyOutput, requests, {}, (server) => testCase(server, 'esbuild copied sibling assets'));
}

async function runCdn(app, cdnBase, requests, testCase) {
  const root = path.join(app, 'cdn-consumer');
  const importUrl = `${cdnBase}/${BRIDGE}`;
  await fsp.mkdir(root);
  await Promise.all([
    fsp.writeFile(path.join(root, 'index.html'), html('/app.js')),
    fsp.writeFile(path.join(root, 'app.js'), renderHarness(importUrl)),
  ]);
  await withStatic(root, requests, {}, (server) => testCase(server, 'CDN threaded ESM', {
    expectedBase: cdnBase, crossOriginWorker: true, crossOriginCore: true, importUrl,
  }));
  await withStatic(root, requests, { isolated: false }, (server) => testCase(server, 'CDN single-thread ESM', {
    expectedBase: cdnBase, crossOriginWorker: true, isolated: false, importUrl, options: { allowSingleThreadCore: true },
  }));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const started = performance.now();
  const model = await resolvePinnedModel(args);
  await withTempDir('lane-npm2-smoke-', async (temp) => {
    const { app, installed } = await preparePackage(temp, args.distDir);
    await Promise.all([
      fsp.writeFile(path.join(app, 'index.html'), html('/app.js')),
      fsp.writeFile(path.join(app, 'app.js'), renderHarness(PACKAGE)),
    ]);
    const cdnRoot = path.join(temp, 'cdn');
    await exposeDist(cdnRoot, CDN_PATH.slice(1), installed);
    await stageFile(model, path.join(cdnRoot, 'model.gguf'));
    const requests = [];
    const browser = await { chromium, firefox }[args.browser].launch();
    try {
      await withStatic(cdnRoot, requests, {
        isolated: false, extraHeaders: [['Cross-Origin-Resource-Policy', 'cross-origin']],
      }, async (cdn) => {
        const cdnBase = `${cdn.origin}${CDN_PATH}`;
        const modelUrl = `${cdn.origin}/model.gguf`;
        const testCase = (server, name, config) => exercise(browser, requests, server, modelUrl, args.timeoutMs, name,
          { ...config, browserName: args.browser });
        if (!args.suite || args.suite === 'vite') await runVite(app, installed, requests, testCase);
        if (!args.suite || args.suite === 'webpack') await runWebpack(app, installed, cdnBase, requests, testCase);
        if (!args.suite || args.suite === 'esbuild') await runEsbuild(app, installed, cdnBase, requests, testCase);
        if (!args.suite || args.suite === 'cdn') await runCdn(app, cdnBase, requests, testCase);
      });
    } finally {
      await browser.close();
    }
  });
  console.log(`Bundler packaging smoke (${args.browser}) passed in ${((performance.now() - started) / 1000).toFixed(1)} s`);
}

if (import.meta.main) await runMain('bundler packaging', main);
