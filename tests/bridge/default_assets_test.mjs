import assert from 'node:assert/strict';
import { copyFile, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { registerHooks } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { pathToFileURL } from 'node:url';
import { LlamaWebGpuBridge } from '../../js/src/bridge.ts';
import { createPthreadWorkerStart, PTHREAD_WORKER_HOOK } from '../../js/src/internal/worker_start.ts';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';

const bundle = new URL('../../js/llama_webgpu_bridge.js', import.meta.url);
const asset = (file) => new URL(`../../js/src/${file}`, import.meta.url).href;
const keys = ['coreModuleUrl', 'wasmUrl', 'coreModuleUrlMem64', 'wasmUrlMem64'];
const urls = (config) => Object.fromEntries(keys.map((key) => [key, config[key]]));

test('the bridge-owned worker receives both single-thread defaults from the page', () => {
  const bridge = new LlamaWebGpuBridge({ disableWorker: true, allowSingleThreadCore: true });
  assert.equal(bridge._workerModuleUrl(), asset('llama_webgpu_bridge_worker.js'));
  assert.deepEqual(urls(bridge._workerConfig()), {
    coreModuleUrl: asset('llama_webgpu_core_st.js'),
    wasmUrl: asset('llama_webgpu_core_st.wasm'),
    coreModuleUrlMem64: asset('llama_webgpu_core_mem64_st.js'),
    wasmUrlMem64: asset('llama_webgpu_core_mem64_st.wasm'),
  });
  const wasm32 = new LlamaWebGpuBridge({ disableWorker: true, preferMemory64: false });
  assert.deepEqual(urls(wasm32._workerConfig()), {
    coreModuleUrl: asset('llama_webgpu_core.js'),
    wasmUrl: asset('llama_webgpu_core.wasm'),
    coreModuleUrlMem64: undefined,
    wasmUrlMem64: undefined,
  });
});

test('a relative assetBaseUrl resolves against the calling document and keeps per-file overrides', (t) => {
  const document = Object.getOwnPropertyDescriptor(globalThis, 'document');
  Object.defineProperty(globalThis, 'document', {
    configurable: true, value: { baseURI: 'https://app.test/root/index.html' },
  });
  t.after(() => {
    if (document) Object.defineProperty(globalThis, 'document', document);
    else delete globalThis.document;
  });

  const bridge = new LlamaWebGpuBridge({
    disableWorker: true, assetBaseUrl: './runtime', coreModuleUrl: './custom.js',
  });
  assert.equal(bridge._workerModuleUrl(), 'https://app.test/root/runtime/llama_webgpu_bridge_worker.js');
  assert.deepEqual(urls(bridge._workerConfig()), {
    coreModuleUrl: './custom.js',
    wasmUrl: 'https://app.test/root/runtime/llama_webgpu_core.wasm',
    coreModuleUrlMem64: 'https://app.test/root/runtime/llama_webgpu_core_mem64.js',
    wasmUrlMem64: 'https://app.test/root/runtime/llama_webgpu_core_mem64.wasm',
  });
  assert.deepEqual(bridge._workerConfig()._coreAssetSources, { wasm32: 'explicit', wasm64: 'base' });
});

test('an app-owned worker gets base URLs only when the caller requests them', async (t) => {
  const scratch = await mkdtemp(join(tmpdir(), 'lane-npm1-app-worker-test-'));
  t.after(() => rm(scratch, { recursive: true, force: true }));
  const workerBundle = pathToFileURL(join(scratch, 'app_bridge.mjs'));
  await copyFile(bundle, workerBundle);
  await writeFile(join(scratch, 'llama_webgpu_core.js'),
    "export default async () => { throw new Error('worker-local core selected'); };");
  const replies = [];
  const previousSelf = Object.getOwnPropertyDescriptor(globalThis, 'self');
  Object.defineProperty(globalThis, 'self', { configurable: true, value: {
    postMessage(message) { replies.push(message); },
  } });
  t.after(() => {
    if (previousSelf) Object.defineProperty(globalThis, 'self', previousSelf);
    else delete globalThis.self;
  });

  const { enableBridgeWorkerHost } = await import(workerBundle);
  enableBridgeWorkerHost();
  const config = { disableWorker: true, workerUrl: 'https://app.test/custom.mjs' };
  const bridge = new LlamaWebGpuBridge({ ...config, logLevel: 0 });
  assert.equal(bridge._workerModuleUrl(), config.workerUrl);
  const wire = bridge._workerConfig();
  assert.deepEqual(Object.values(urls(wire)), [undefined, undefined, undefined, undefined]);
  assert.equal(Object.hasOwn(wire, '_coreAssetSources'), false);
  await self.onmessage({ data: { type: 'init', config: wire } });
  await self.onmessage({ data: { type: 'call', id: 1, method: 'loadModelFromUrl', args: ['https://app.test/model.gguf'] } });
  assert.equal(replies.at(-1).message, 'worker-local core selected');

  const withBase = new LlamaWebGpuBridge({ ...config, assetBaseUrl: 'https://cdn.test/runtime' });
  assert.equal(withBase._workerModuleUrl(), config.workerUrl);
  assert.deepEqual(urls(withBase._workerConfig()), {
    coreModuleUrl: 'https://cdn.test/runtime/llama_webgpu_core.js',
    wasmUrl: 'https://cdn.test/runtime/llama_webgpu_core.wasm',
    coreModuleUrlMem64: 'https://cdn.test/runtime/llama_webgpu_core_mem64.js',
    wasmUrlMem64: 'https://cdn.test/runtime/llama_webgpu_core_mem64.wasm',
  });
  assert.deepEqual(withBase._workerConfig()._coreAssetSources, { wasm32: 'base', wasm64: 'base' });
});

test('explicit URLs and the llamadart worker bundle route retain their original wire configuration', () => {
  const config = {
    workerUrl: 'https://cdn.test/v/llama_webgpu_bridge.js?__llamadartBridgeWorker=1',
    coreModuleUrl: './loader.js',
    wasmUrl: '/assets/core.wasm',
    preferMemory64: true,
    allowSingleThreadCore: true,
  };
  const bridge = new LlamaWebGpuBridge({ ...config, disableWorker: true });
  assert.equal(bridge._workerModuleUrl(), config.workerUrl);
  assert.equal(Object.hasOwn(bridge._workerConfig(), '_coreAssetSources'), false);
  assert.deepEqual(urls(bridge._workerConfig()), {
    coreModuleUrl: config.coreModuleUrl,
    wasmUrl: config.wasmUrl,
    coreModuleUrlMem64: undefined,
    wasmUrlMem64: undefined,
  });
  const legacyMemory64 = new LlamaWebGpuBridge({
    disableWorker: true, coreModuleUrlMem64: './core64.js',
  });
  assert.deepEqual(Object.values(urls(legacyMemory64._workerConfig())), [
    undefined, undefined, './core64.js', undefined,
  ]);
});

test('an inlined bridge module can still serve as its own worker entry', async () => {
  const dataUrl = `data:text/javascript;base64,${await readFile(bundle, 'base64')}`;
  const { LlamaWebGpuBridge: InlinedBridge } = await import(dataUrl);
  const bridge = new InlinedBridge({ disableWorker: true, coreModuleUrl: 'https://app.test/core.js' });
  assert.ok(bridge._workerModuleUrl() === dataUrl, 'expected the inline module URL as the worker entry');
});

test('both generated entries share the installed worker host in the same worker', async (t) => {
  const replies = [];
  const previousSelf = Object.getOwnPropertyDescriptor(globalThis, 'self');
  Object.defineProperty(globalThis, 'self', { configurable: true, value: {
    postMessage(message) { replies.push(message); },
  } });
  t.after(() => {
    if (previousSelf) Object.defineProperty(globalThis, 'self', previousSelf);
    else delete globalThis.self;
  });

  await import(new URL('./llama_webgpu_bridge_worker.js', bundle));
  await self.onmessage({ data: { type: 'init', config: {} } });
  const installed = self.onmessage;
  const { enableBridgeWorkerHost } = await import(bundle);
  enableBridgeWorkerHost();
  assert.equal(self.onmessage, installed);
  await self.onmessage({ data: { type: 'call', id: 1, method: 'getContextSize' } });
  assert.equal(replies.at(-1).type, 'result');
});

test('the pthread hook redirects only a bridge-resolved core’s stock sibling request', (t) => {
  const attempts = [];
  const previousWorker = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  Object.defineProperty(globalThis, 'Worker', { configurable: true, value: class {
    constructor(url) { attempts.push(url); }
  } });
  t.after(() => {
    if (previousWorker) Object.defineProperty(globalThis, 'Worker', previousWorker);
    else delete globalThis.Worker;
  });

  const hashed = 'https://app.test/assets/llama_webgpu_core-abc123.js';
  const requested = new URL('llama_webgpu_core.js', hashed);
  const unrelated = new URL('helper.js', hashed);
  const otherOrigin = new URL('https://cdn.test/assets/llama_webgpu_core.js');
  const resolved = createPthreadWorkerStart({ pthreadScriptUrl: hashed });
  new resolved.Worker(requested);
  new resolved.Worker(unrelated);
  new resolved.Worker(otherOrigin);
  const explicit = createPthreadWorkerStart();
  new explicit.Worker(requested);
  const unhashed = createPthreadWorkerStart({ pthreadScriptUrl: requested.href });
  new unhashed.Worker(requested);
  assert.deepEqual(attempts, [hashed, unrelated, otherOrigin, requested, requested]);
});

test('direct runtime resolves the core and wasm and enables the rewrite only for base or default candidates', async (t) => {
  const hashed = 'https://app.test/assets/llama_webgpu_core-abc.js';
  const base = 'https://cdn.test/bridge/';
  const baseMem64 = `${base}llama_webgpu_core_mem64.js`;
  const defaultCore = asset('llama_webgpu_core_st.js');
  const imports = [];
  const calls = [];
  const previousWorker = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  Object.defineProperty(globalThis, 'Worker', { configurable: true, value: class {
    constructor(url) { calls.push(String(url)); }
  } });
  globalThis.__laneNpm1CoreOptions = [];
  const hook = registerHooks({
    resolve(specifier, context, nextResolve) {
      if (![hashed, baseMem64, defaultCore].includes(specifier)) return nextResolve(specifier, context);
      imports.push(specifier);
      return { url: `lane-npm1:${specifier}`, shortCircuit: true };
    },
    load(url, context, nextLoad) {
      if (!url.startsWith('lane-npm1:')) return nextLoad(url, context);
      return { format: 'module', shortCircuit: true, source: `
        export default async (options) => {
          if (${url.endsWith('llama_webgpu_core_mem64.js')} && globalThis.__laneNpm1FailMem64) {
            throw new Error('wasm64 unavailable');
          }
          globalThis.__laneNpm1CoreOptions.push(options);
          return { ccall: () => 0 };
        };
      ` };
    },
  });
  t.after(() => {
    hook.deregister();
    delete globalThis.__laneNpm1CoreOptions;
    delete globalThis.__laneNpm1FailMem64;
    if (previousWorker) Object.defineProperty(globalThis, 'Worker', previousWorker);
    else delete globalThis.Worker;
  });

  const direct = new LlamaWebGpuBridgeRuntime({ preferMemory64: false, allowSingleThreadCore: true });
  await direct._ensureCore();
  const explicit = new LlamaWebGpuBridgeRuntime({
    assetBaseUrl: base, coreModuleUrl: hashed, preferMemory64: false,
  });
  await explicit._ensureCore();
  const forwarded = {
    coreModuleUrl: hashed, coreModuleUrlMem64: baseMem64,
    wasmUrlMem64: `${base}llama_webgpu_core_mem64.wasm`,
    _coreAssetSources: { wasm32: 'default', wasm64: 'base' },
  };
  const worker64 = new LlamaWebGpuBridgeRuntime(forwarded);
  await worker64._ensureCore();
  assert.deepEqual(imports, [defaultCore, hashed, baseMem64]);
  assert.equal(worker64._coreVariant, 'wasm64');
  const mixed = new LlamaWebGpuBridgeRuntime({ assetBaseUrl: base, coreModuleUrl: hashed });
  await mixed._ensureCore();
  assert.deepEqual(imports, [defaultCore, hashed, baseMem64, baseMem64]);

  const worker32 = new LlamaWebGpuBridgeRuntime({ ...forwarded, preferMemory64: false });
  await worker32._ensureCore();
  const untagged = new LlamaWebGpuBridgeRuntime({
    coreModuleUrl: hashed, _coreAssetSources: {}, preferMemory64: false,
  });
  await untagged._ensureCore();
  globalThis.__laneNpm1FailMem64 = true;
  const fallback = new LlamaWebGpuBridgeRuntime({ ...forwarded, logLevel: 0 });
  await fallback._ensureCore();

  assert.deepEqual(imports, [defaultCore, hashed, baseMem64, baseMem64, hashed, hashed, baseMem64, hashed]);
  assert.equal(fallback._coreVariant, 'wasm32');
  const [directOpts, explicitOpts, worker64Opts, mixedOpts, worker32Opts, untaggedOpts] = globalThis.__laneNpm1CoreOptions;
  assert.equal(directOpts.locateFile('core.wasm', '/'), asset('llama_webgpu_core_st.wasm'));
  assert.equal(explicitOpts.locateFile('core.wasm', '/'), `${base}llama_webgpu_core.wasm`);
  assert.equal(worker64Opts.locateFile('core.wasm', '/'), `${base}llama_webgpu_core_mem64.wasm`);
  assert.equal(mixedOpts.locateFile('core.wasm', '/'), `${base}llama_webgpu_core_mem64.wasm`);
  const requested = new URL('llama_webgpu_core.js', hashed);
  new explicitOpts[PTHREAD_WORKER_HOOK](requested);
  new worker32Opts[PTHREAD_WORKER_HOOK](requested);
  new untaggedOpts[PTHREAD_WORKER_HOOK](requested);
  assert.deepEqual(calls, [requested.href, hashed, requested.href]);
});

test('a custom factory retains its options and disables bridge-owned worker startup', async () => {
  let options;
  const bridge = new LlamaWebGpuBridge({
    assetBaseUrl: 'https://cdn.test/runtime/',
    wasmUrl: './explicit.wasm',
    coreModuleFactory: async (arg) => { options = arg; return { ccall: () => 0 }; },
  });
  assert.equal(bridge._shouldUseWorker(), false);
  await bridge._runtime._ensureCore();
  assert.deepEqual(Reflect.ownKeys(options), ['locateFile', 'print', 'printErr', 'onAbort']);
  assert.equal(options.locateFile('core.wasm', '/'), './explicit.wasm');
});

test('the generated bridge exposes literal assets and the worker has no relative import', async () => {
  const main = await readFile(bundle, 'utf8');
  for (const suffix of ['', '_mem64', '_st', '_mem64_st']) {
    for (const ext of ['js', 'wasm']) {
      assert.ok(main.includes(`new URL("./llama_webgpu_core${suffix}.${ext}", import.meta.url)`));
    }
  }
  assert.ok(main.includes('new URL("./llama_webgpu_bridge_worker.js", import.meta.url)'));
  assert.match(main, /import\(\s*\/\* @vite-ignore \*\/\s*\/\* webpackIgnore: true \*\/\s*moduleUrl\s*\)/);
  const worker = await readFile(new URL('./llama_webgpu_bridge_worker.js', bundle), 'utf8');
  assert.doesNotMatch(worker, /^\s*import\b[^\n]*["']\.{1,2}\//m);
});
