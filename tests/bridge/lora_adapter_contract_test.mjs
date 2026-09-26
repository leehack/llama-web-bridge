import assert from 'node:assert/strict';

import { LlamaWebGpuBridge, enableBridgeWorkerHost } from '../../js/src/llama_webgpu_bridge.js';
import { readRepoText } from './bridge_js_source.mjs';
import {
  createRealWorkerBridge,
  createWorkerBridge,
  withStubWorkerEnvironment,
  workerDriver,
} from './bridge_operation_queue_fixtures.mjs';
import { readNativeCoreSource } from './native_core_source.mjs';
import * as smoke from '../../scripts/smoke/lora_adapter.mjs';

// A core that records LoRA calls and applies them as the C++ core does.
function createLoraCore({ version = 1, exported = true, loadResult = null, loadError = '' } = {}) {
  const files = new Map();
  const core = {
    calls: [],
    files,
    staged: [],
    active: [],
    lastError: '',
    nextHandle: 7,
    FS: {
      mkdir: () => {},
      readdir: () => ['.', '..', 'lora'],
      writeFile: (path, bytes) => files.set(path, new Uint8Array(bytes)),
      open: (path) => ({ path, chunks: [] }),
      write: (stream, buffer, offset, length) => {
        stream.chunks.push(buffer.slice(offset, offset + length));
        return length;
      },
      close: (stream) => {
        const bytes = new Uint8Array(stream.chunks.reduce((sum, chunk) => sum + chunk.length, 0));
        let at = 0;
        for (const chunk of stream.chunks) {
          bytes.set(chunk, at);
          at += chunk.length;
        }
        files.set(stream.path, bytes);
      },
      unlink: (path) => files.delete(path),
    },
    ccall(name, _returnType, _argTypes, args = []) {
      core.calls.push([name, ...args]);
      switch (name) {
        case 'llamadart_webgpu_lora_api_version':
          return version;
        case 'llamadart_webgpu_lora_load': {
          core.staged.push({ path: args[0], bytes: files.get(args[0]) });
          core.lastError = loadError;
          return Promise.resolve(loadResult ?? core.nextHandle++);
        }
        case 'llamadart_webgpu_lora_set': {
          const entry = core.active.find(([handle]) => handle === args[0]);
          if (entry) {
            entry[1] = args[1];
          } else {
            core.active.push([args[0], args[1]]);
          }
          return Promise.resolve(0);
        }
        case 'llamadart_webgpu_lora_remove':
          core.active = core.active.filter(([handle]) => handle !== args[0]);
          return Promise.resolve(0);
        case 'llamadart_webgpu_lora_clear':
          core.active = [];
          return Promise.resolve(0);
        case 'llamadart_webgpu_last_error':
          return core.lastError;
        case 'llamadart_webgpu_free_model':
        case 'llamadart_webgpu_media_clear_pending':
        case 'llamadart_webgpu_mmproj_free':
        case 'llamadart_webgpu_shutdown':
          return 0;
        default:
          throw new Error(`Unexpected ccall: ${name}`);
      }
    },
  };
  if (exported) {
    core._llamadart_webgpu_lora_api_version = () => version;
  }
  return core;
}

function createDirectLoraBridge(coreOptions) {
  const bridge = new LlamaWebGpuBridge({ disableWorker: true });
  const core = createLoraCore(coreOptions);
  bridge._runtime._core = core;
  bridge._runtime._modelBytes = 1;
  return { bridge, core };
}

const coreCalls = (core, name) => core.calls.filter((call) => call[0] === name);
const loraCalls = (core) => core.calls.filter((call) => call[0].startsWith('llamadart_webgpu_lora_'));

function response(bytes, status = 200) {
  return new Response(new Uint8Array(bytes), {
    status,
    statusText: status === 200 ? 'OK' : 'Not Found',
    headers: { 'content-length': String(bytes.length) },
  });
}

// An in-memory Cache API.
function installCaches() {
  const stores = new Map();
  const original = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  Object.defineProperty(globalThis, 'caches', {
    configurable: true,
    value: {
      async open(name) {
        if (!stores.has(name)) {
          const entries = new Map();
          stores.set(name, {
            entries,
            async match(key) {
              const stored = entries.get(key);
              return stored ? response(stored) : undefined;
            },
            async put(key, value) {
              entries.set(key, new Uint8Array(await value.arrayBuffer()));
            },
          });
        }
        return stores.get(name);
      },
    },
  });
  return {
    stores,
    restore() {
      if (original) {
        Object.defineProperty(globalThis, 'caches', original);
      } else {
        delete globalThis.caches;
      }
    },
  };
}

// Direct runtime: bytes staging, facade handles, and the active set.
{
  const { bridge, core } = createDirectLoraBridge();
  const capabilities = await bridge.getLoraAdapterCapabilities();
  assert.deepEqual(capabilities, { apiVersion: 1, supported: true });

  const info = await bridge.loadLoraAdapter(new Uint8Array([1, 2, 3]));
  assert.deepEqual(info, { handle: 1 }, 'the facade returns its own handle');
  assert.match(core.staged[0].path, /^\/lora\/adapter_\d+\.gguf$/);
  assert.deepEqual(Array.from(core.staged[0].bytes), [1, 2, 3]);
  assert.equal(core.files.size, 0, 'the staged adapter is deleted after loading');
  assert.equal(bridge._runtime._loraAdaptersLoaded, true);

  const second = await bridge.loadLoraAdapter(new Uint16Array([0x0201]).buffer);
  assert.equal(second.handle, 2);
  assert.deepEqual(Array.from(core.staged[1].bytes), [1, 2]);

  await bridge.setLoraAdapter(1);
  await bridge.setLoraAdapter(2, 0.5);
  await bridge.setLoraAdapter(1, -0.25);
  assert.deepEqual(core.active, [[7, -0.25], [8, 0.5]], 'facade handles resolve to core handles; an update keeps its place');
  assert.deepEqual([...bridge._activeLoraScales], [[1, -0.25], [2, 0.5]]);
  await bridge.removeLoraAdapter(1);
  assert.deepEqual(core.active, [[8, 0.5]]);
  await bridge.removeLoraAdapter(1);
  assert.equal(coreCalls(core, 'llamadart_webgpu_lora_remove').length, 2, 'removing an inactive adapter still reaches the core');
  await bridge.clearLoraAdapters();
  assert.deepEqual(core.active, []);
  assert.equal(bridge._activeLoraScales.size, 0);

  const before = loraCalls(core).length;
  await assert.rejects(bridge.setLoraAdapter(0), /^TypeError: LoRA adapter handle must be a positive integer, got 0\./);
  await assert.rejects(bridge.setLoraAdapter(1.5), /handle must be a positive integer/);
  await assert.rejects(bridge.setLoraAdapter(1, Number.NaN), /^TypeError: LoRA adapter scale must be a finite number, got NaN\./);
  await assert.rejects(bridge.setLoraAdapter(1, '1'), /scale must be a finite number/);
  await assert.rejects(bridge.setLoraAdapter(1, 1e39), /^RangeError: LoRA adapter scale 1e\+39 is outside the 32-bit float range\./);
  await assert.rejects(bridge.removeLoraAdapter(-1), /handle must be a positive integer/);
  await assert.rejects(bridge.setLoraAdapter(3), /LoRA adapter 3 is not loaded; its model was unloaded or replaced/);
  await assert.rejects(bridge.removeLoraAdapter(3), /LoRA adapter 3 is not loaded/);
  await assert.rejects(bridge.loadLoraAdapter([1, 2]), /must be a URL string, an ArrayBuffer or a typed array/);
  await assert.rejects(bridge.loadLoraAdapter(new Uint8Array()), /LoRA adapter bytes are empty/);
  await assert.rejects(bridge.loadLoraAdapter(''), /LoRA adapter URL is empty/);
  const aborted = new AbortController();
  aborted.abort();
  await assert.rejects(
    bridge.loadLoraAdapter(new Uint8Array([1]), { signal: aborted.signal }),
    (error) => error.name === 'AbortError' && /LoRA adapter load was cancelled/.test(error.message),
  );
  assert.equal(loraCalls(core).length, before, 'rejected arguments never reach the core');

  // A new model drops the facade's adapters, and a runtime that holds adapters
  // does not reload its model to recover a failed generation.
  bridge._rememberLoadedModel('other.gguf');
  await assert.rejects(bridge.setLoraAdapter(1), /LoRA adapter 1 is not loaded/);
  const third = await bridge.loadLoraAdapter(new Uint8Array([9]));
  assert.equal(third.handle, 3, 'facade handles are never reused');
  const runtime = bridge._runtime;
  runtime._loadedModelUrl = 'model.gguf';
  let reloads = 0;
  runtime.loadModelFromUrl = async () => {
    reloads += 1;
  };
  assert.equal(await runtime._recoverGenerationWithCpuFallback(), false);
  assert.equal(reloads, 0);
  assert.ok(runtime._runtimeNotes.includes('generation_recovery_cpu_skipped_lora'));
  runtime._modelBytes = 1;
  runtime._releaseLoadedModel(core);
  assert.equal(runtime._loraAdaptersLoaded, false, 'releasing the model forgets its adapters');

  await bridge.dispose();
  assert.equal(bridge._loraAdapters.size, 0);
}

// Direct runtime: load failures, capability probes and a missing model.
{
  const { bridge, core } = createDirectLoraBridge({
    loadResult: -5,
    loadError: 'the adapter is an aLoRA adapter (2 invocation token(s)).',
  });
  await assert.rejects(
    bridge.loadLoraAdapter(new Uint8Array([1])),
    /^Error: Failed to load LoRA adapter: the adapter is an aLoRA adapter \(2 invocation token\(s\)\)\.$/,
  );
  assert.equal(core.files.size, 0, 'a failed load still deletes the staged adapter');
  assert.equal(bridge._loraAdapters.size, 0);
  assert.equal(bridge._runtime._loraAdaptersLoaded, false);
}

for (const [coreOptions, reason] of [
  [{ exported: false }, /does not include LoRA adapters/],
  [{ version: 2 }, /LoRA API version 2; this bridge needs version 1/],
]) {
  const { bridge, core } = createDirectLoraBridge(coreOptions);
  const capabilities = await bridge.getLoraAdapterCapabilities();
  assert.equal(capabilities.supported, false);
  assert.match(capabilities.reason, reason);
  await assert.rejects(bridge.loadLoraAdapter(new Uint8Array([1])), reason);
  await assert.rejects(bridge.clearLoraAdapters(), reason);
  assert.equal(coreCalls(core, 'llamadart_webgpu_lora_load').length, 0);
}

{
  const bridge = new LlamaWebGpuBridge({ disableWorker: true });
  const capabilities = await bridge.getLoraAdapterCapabilities();
  assert.deepEqual(capabilities, { apiVersion: 1, supported: false, reason: 'WebGPU core is not initialized' });
  await assert.rejects(bridge.loadLoraAdapter(new Uint8Array([1])), /No model loaded/);
  await assert.rejects(bridge.clearLoraAdapters(), /No model loaded/);
}

// Direct runtime: a URL adapter goes through the Cache API with progress.
{
  const caches = installCaches();
  try {
    const { bridge, core } = createDirectLoraBridge();
    bridge._runtime._config.cacheName = 'lora-test-cache';
    const fetched = [];
    let failNext = 0;
    bridge._runtime._fetchWithTimeout = async (url, init) => {
      fetched.push({ url, init });
      if (failNext > 0) {
        failNext -= 1;
        throw new TypeError('Failed to fetch');
      }
      return url.includes('missing') ? response([], 404) : response([4, 5, 6]);
    };
    const url = 'https://example.test/adapters/a.gguf';
    const progress = [];
    await bridge.loadLoraAdapter(url, { progressCallback: (event) => progress.push(event) });
    assert.deepEqual(progress.at(-1), { loaded: 3, total: 3 });
    assert.deepEqual(Array.from(core.staged[0].bytes), [4, 5, 6]);
    assert.equal(fetched.length, 1);
    assert.equal(fetched[0].init.cache, undefined, 'a cached load leaves the HTTP cache alone, as models do');
    assert.ok(caches.stores.get('lora-test-cache').entries.has(url), 'the bridge cacheName holds the adapter');

    await bridge.loadLoraAdapter(url);
    assert.equal(fetched.length, 1, 'a second load reads the Cache API');
    assert.deepEqual(Array.from(core.staged[1].bytes), [4, 5, 6]);

    await bridge.loadLoraAdapter(url, { useCache: false });
    assert.equal(fetched.length, 2);
    assert.equal(fetched[1].init.cache, 'no-store');

    await bridge.loadLoraAdapter('https://example.test/b.gguf', { cacheName: 'other-cache' });
    assert.ok(caches.stores.get('other-cache').entries.has('https://example.test/b.gguf'));

    failNext = 1;
    await bridge.loadLoraAdapter('https://example.test/c.gguf', { useCache: false });
    assert.ok(bridge._runtime._runtimeNotes.includes('lora_fetch_retry:1'), 'one network failure is retried');

    const loads = coreCalls(core, 'llamadart_webgpu_lora_load').length;
    await assert.rejects(
      bridge.loadLoraAdapter('https://user:pass@example.test/missing.gguf?sig=secret#frag'),
      (error) => error.message === 'Failed to fetch LoRA adapter: 404 Not Found',
    );
    assert.equal(coreCalls(core, 'llamadart_webgpu_lora_load').length, loads);
    assert.equal(core.files.size, 0);
  } finally {
    caches.restore();
  }
}

// Worker path: copies, handles, progress and model reloads.
{
  const workerCalls = [];
  const proxy = {};
  const bridge = createWorkerBridge({
    _workerProxy: proxy,
    _callWorker: async (method, args, onEvent, transferList = []) => {
      workerCalls.push({ method, args, transferList });
      if (method === 'loadLoraAdapter') {
        onEvent?.({ event: 'progress', payload: { loaded: 1, total: 2 } });
        return { handle: 40 + workerCalls.filter((call) => call.method === method).length };
      }
      if (method === 'getLoraAdapterCapabilities') {
        return { apiVersion: 1, supported: true };
      }
      return undefined;
    },
  });

  assert.deepEqual(await bridge.getLoraAdapterCapabilities(), { apiVersion: 1, supported: true });
  const caller = new Uint8Array([9, 8, 7]);
  const progress = [];
  const signal = new AbortController().signal;
  const info = await bridge.loadLoraAdapter(caller, {
    signal,
    useCache: false,
    progressCallback: (event) => progress.push(event),
  });
  assert.deepEqual(info, { handle: 1 });
  const load = workerCalls.at(-1);
  assert.notEqual(load.args[0].buffer, caller.buffer, 'the worker receives a copy');
  assert.deepEqual(load.transferList, [load.args[0].buffer]);
  assert.deepEqual(load.args[1], { useCache: false }, 'the signal and callback stay on the main thread');
  assert.equal(caller.byteLength, 3, "the caller's buffer is not detached");
  assert.deepEqual(progress, [{ loaded: 1, total: 2 }]);
  caller[0] = 0;
  assert.deepEqual(Array.from(bridge._loraAdapters.get(1).source), [9, 8, 7], 'the retained source is a copy');
  assert.notEqual(load.transferList[0], bridge._loraAdapters.get(1).source.buffer, 'the retained source is never transferred');

  globalThis.window = { location: { href: 'https://example.test/app/index.html' } };
  try {
    await bridge.loadLoraAdapter('adapters/a.gguf');
  } finally {
    delete globalThis.window;
  }
  assert.equal(workerCalls.at(-1).args[0], 'https://example.test/app/adapters/a.gguf', 'relative URLs resolve against the page');

  await bridge.setLoraAdapter(2, 0.75);
  assert.deepEqual(workerCalls.at(-1), { method: 'setLoraAdapter', args: [42, 0.75], transferList: [] });
  await bridge.removeLoraAdapter(2);
  assert.deepEqual(workerCalls.at(-1).args, [42]);
  await bridge.clearLoraAdapters();
  assert.equal(workerCalls.at(-1).method, 'clearLoraAdapters');

  bridge._callWorker = async (method, args) => {
    workerCalls.push({ method, args });
    return method === 'loadModelFromUrl' ? 1 : undefined;
  };
  await bridge.loadModelFromUrl('next.gguf');
  await assert.rejects(bridge.setLoraAdapter(1), /LoRA adapter 1 is not loaded/);
}

// Worker path: a worker failure moves the adapters to the main-thread runtime.
function createFallbackRuntime() {
  return {
    _modelBytes: 0,
    _runtimeNotes: [],
    calls: [],
    nextHandle: 90,
    async loadModelFromUrl(url) {
      this.calls.push(['loadModelFromUrl', url]);
      this._modelBytes = 1;
    },
    getLoraAdapterCapabilities() {
      return { apiVersion: 1, supported: true, from: 'runtime' };
    },
    async loadLoraAdapter(source, options) {
      this.calls.push(['loadLoraAdapter', typeof source === 'string' ? source : Array.from(source), options.useCache]);
      return { handle: this.nextHandle++ };
    },
    async setLoraAdapter(handle, scale) {
      this.calls.push(['setLoraAdapter', handle, scale]);
    },
    async removeLoraAdapter(handle) {
      this.calls.push(['removeLoraAdapter', handle]);
    },
    async clearLoraAdapters() {
      this.calls.push(['clearLoraAdapters']);
    },
  };
}

const workerFailure = () => new Error('Bridge worker request failed: worker terminated');

{
  const runtime = createFallbackRuntime();
  let failRemove = false;
  const bridge = createWorkerBridge({
    _workerProxy: { dispose: async () => {} },
    _createRuntime: () => runtime,
    _callWorker: async (method) => {
      if (method === 'loadLoraAdapter') {
        return { handle: 5 };
      }
      if (failRemove && method === 'removeLoraAdapter') {
        throw workerFailure();
      }
      return undefined;
    },
  });
  const a = await bridge.loadLoraAdapter('https://example.test/a.gguf', { useCache: false });
  const b = await bridge.loadLoraAdapter(new Uint8Array([3, 4]));
  const idle = await bridge.loadLoraAdapter('https://example.test/idle.gguf');
  await bridge.setLoraAdapter(a.handle, 0.5);
  await bridge.setLoraAdapter(b.handle);
  failRemove = true;
  await bridge.removeLoraAdapter(b.handle);

  assert.equal(bridge._workerProxy, null);
  assert.deepEqual(runtime.calls, [
    ['loadModelFromUrl', 'model.gguf'],
    ['clearLoraAdapters'],
    ['loadLoraAdapter', 'https://example.test/a.gguf', false],
    ['setLoraAdapter', 90, 0.5],
    ['loadLoraAdapter', [3, 4], undefined],
    ['setLoraAdapter', 91, 1],
    ['removeLoraAdapter', 91],
  ], 'the runtime reloads the model, restores the active set in order, then retries the call');
  assert.deepEqual([...bridge._activeLoraScales], [[a.handle, 0.5]]);

  runtime.calls.length = 0;
  await bridge.setLoraAdapter(idle.handle, 2);
  assert.deepEqual(runtime.calls, [
    ['loadLoraAdapter', 'https://example.test/idle.gguf', undefined],
    ['setLoraAdapter', 92, 2],
  ], 'an adapter that was not applied reloads when next set');
  runtime.calls.length = 0;
  await bridge.removeLoraAdapter(b.handle);
  assert.deepEqual(runtime.calls, [['removeLoraAdapter', 91]]);
  assert.deepEqual((await bridge.getLoraAdapterCapabilities()).from, 'runtime');

  runtime.calls.length = 0;
  await bridge._ensureRuntimeReadyAfterWorkerFallback({ _llamadartForceRuntimeReload: true }, null);
  assert.deepEqual(runtime.calls, [
    ['loadModelFromUrl', 'model.gguf'],
    ['clearLoraAdapters'],
    ['loadLoraAdapter', 'https://example.test/a.gguf', false],
    ['setLoraAdapter', 93, 0.5],
    ['loadLoraAdapter', 'https://example.test/idle.gguf', undefined],
    ['setLoraAdapter', 94, 2],
  ], 'a model reload on the runtime reloads the adapters it freed');
}

{
  const runtime = createFallbackRuntime();
  const bridge = createWorkerBridge({
    _workerProxy: { dispose: async () => {} },
    _createRuntime: () => runtime,
    _callWorker: async () => {
      throw workerFailure();
    },
  });
  const info = await bridge.loadLoraAdapter('https://example.test/a.gguf');
  assert.deepEqual(runtime.calls, [
    ['loadModelFromUrl', 'model.gguf'],
    ['loadLoraAdapter', 'https://example.test/a.gguf', undefined],
  ], 'a load retried on the main thread');
  await bridge.setLoraAdapter(info.handle, 1);
  assert.deepEqual(runtime.calls.at(-1), ['setLoraAdapter', 90, 1]);
}

// Worker path: a replacement worker gets the model, then the applied adapters.
{
  const workerCalls = [];
  const bridge = createWorkerBridge({
    _loadedMmProjUrl: null,
    _callWorker: async (method, args) => {
      workerCalls.push([method, ...args]);
      return method === 'loadLoraAdapter' ? { handle: 60 + workerCalls.length } : undefined;
    },
  });
  const info = await bridge.loadLoraAdapter(new Uint8Array([1]));
  await bridge.setLoraAdapter(info.handle, 0.25);
  bridge._workerProxy = {};
  bridge._workerModelMissing = true;
  workerCalls.length = 0;
  await bridge.setLoraAdapter(info.handle, 0.5);
  assert.deepEqual(workerCalls.map(([method]) => method), [
    'loadModelFromUrl',
    'clearLoraAdapters',
    'loadLoraAdapter',
    'setLoraAdapter',
    'setLoraAdapter',
  ]);
  assert.deepEqual(workerCalls[3].slice(1), [63, 0.25]);
  assert.deepEqual(workerCalls[4].slice(1), [63, 0.5]);
}

// Worker path: a failed restore is retried on the next call, and a model
// reload into the same worker reloads the adapters it freed.
{
  const workerCalls = [];
  let failAdapterLoad = false;
  const bridge = createWorkerBridge({
    _loadedMmProjUrl: null,
    _callWorker: async (method, args) => {
      workerCalls.push([method, ...args]);
      if (method === 'loadLoraAdapter') {
        if (failAdapterLoad) {
          throw new Error('Failed to fetch LoRA adapter: 503 Service Unavailable');
        }
        return { handle: 70 + workerCalls.length };
      }
      return undefined;
    },
  });
  const a = await bridge.loadLoraAdapter('https://example.test/a.gguf', { useCache: false });
  const b = await bridge.loadLoraAdapter('https://example.test/b.gguf', { useCache: false });
  await bridge.setLoraAdapter(a.handle, 1);
  await bridge.setLoraAdapter(b.handle, 0.5);

  bridge._workerProxy = {};
  bridge._workerModelMissing = true;
  failAdapterLoad = true;
  await assert.rejects(bridge.tokenize('x'), /503/);
  assert.equal(bridge._workerModelMissing, true, 'a failed adapter restore leaves the worker marked for restore');

  failAdapterLoad = false;
  workerCalls.length = 0;
  await bridge.tokenize('y');
  assert.deepEqual(workerCalls.map(([method]) => method), [
    'loadModelFromUrl',
    'clearLoraAdapters',
    'loadLoraAdapter',
    'setLoraAdapter',
    'loadLoraAdapter',
    'setLoraAdapter',
    'tokenize',
  ], 'the next call reloads the model and every applied adapter');
  assert.equal(bridge._workerModelMissing, false);

  workerCalls.length = 0;
  await bridge._loadRememberedModelIntoWorker({});
  assert.deepEqual(workerCalls.map(([method]) => method), [
    'loadModelFromUrl',
    'clearLoraAdapters',
    'loadLoraAdapter',
    'setLoraAdapter',
    'loadLoraAdapter',
    'setLoraAdapter',
  ], 'a model reload into the same worker reloads its adapters');
  await bridge.setLoraAdapter(a.handle, 2);
  assert.deepEqual(workerCalls.at(-1), ['setLoraAdapter', 73, 2], 'set uses the reloaded worker handle');
}

// Worker host: adapter loads post progress events and keep the signal local.
{
  const originals = { self: globalThis.self, load: LlamaWebGpuBridge.prototype.loadLoraAdapter };
  const posted = [];
  globalThis.self = {
    postMessage(message) {
      posted.push(message);
    },
  };
  let loadOptions = null;
  LlamaWebGpuBridge.prototype.loadLoraAdapter = async function (_source, options) {
    loadOptions = options;
    options.progressCallback({ loaded: 3, total: 4 });
    return { handle: 2 };
  };
  try {
    enableBridgeWorkerHost();
    await globalThis.self.onmessage({ data: { type: 'init', config: {} } });
    await globalThis.self.onmessage({
      data: { type: 'call', id: 1, method: 'loadLoraAdapter', args: ['a.gguf', { useCache: false, signal: 'x' }] },
    });
    assert.deepEqual(posted.filter((message) => message.id === 1), [
      { type: 'event', id: 1, event: 'progress', payload: { loaded: 3, total: 4 } },
      { type: 'result', id: 1, value: { handle: 2 } },
    ]);
    assert.equal(loadOptions.useCache, false);
    assert.equal('signal' in loadOptions, false);
  } finally {
    LlamaWebGpuBridge.prototype.loadLoraAdapter = originals.load;
    globalThis.self = originals.self;
  }
}

// Worker proxy: an adapter load is bounded by stalls between progress events.
await withStubWorkerEnvironment(async ({ workers }) => {
  const bridge = createRealWorkerBridge();
  const proxy = bridge._workerProxy;
  workerDriver(proxy, workers[0]).ready();
  assert.equal(proxy._resolveRequestTimeoutMs('loadLoraAdapter', ['a.gguf', {}]), 10 * 60 * 1000);
  assert.equal(proxy._resolveRequestTimeoutMs('setLoraAdapter', [1, 1]), 120000);
});

// Native core: the rules the fake core above stands in for.
{
  const CORE = readNativeCoreSource();
  const LORA = readRepoText('src/llama_webgpu_lora.cpp');
  const HEADER = readRepoText('src/llama_webgpu_lora.h');
  const CMAKE = readRepoText('CMakeLists.txt');
  const LORA_JS = readRepoText('js/src/internal/lora.ts');
  const body = (signature) => {
    const start = CORE.indexOf(signature);
    assert.ok(start >= 0, `missing ${signature}`);
    return CORE.slice(start, CORE.indexOf('\n}\n', start));
  };

  assert.ok(HEADER.includes('LLAMADART_WEBGPU_LORA_API_VERSION = 1') && LORA_JS.includes('LORA_API_VERSION = 1'));
  for (const symbol of ['api_version', 'load', 'set', 'remove', 'clear']) {
    const name = `llamadart_webgpu_lora_${symbol}`;
    assert.match(CORE, new RegExp(String.raw`EMSCRIPTEN_KEEPALIVE [^\n]*\b${name}\(`));
    assert.ok(CMAKE.includes(`'_${name}'`), `CMake must export _${name}`);
  }
  assert.ok(
    CMAKE.includes("'llamadart_webgpu_lora_adapter_init'")
      && /set_source_files_properties\([^)]*src\/llama_webgpu_lora\.cpp/.test(CMAKE),
    'the adapter loader must be compiled and linked with exception catching',
  );
  assert.ok(
    /extern "C" __attribute__\(\(noinline\)\) llama_adapter_lora \*\s*llamadart_webgpu_lora_adapter_init\(/.test(LORA)
      && LORA.indexOf('catch (const std::exception & error)') < LORA.indexOf('std::fclose(file);'),
    'the loader must catch llama.cpp throws by name and close the file on both paths',
  );

  const freeRuntime = body('void free_runtime() {');
  const contextFreed = freeRuntime.indexOf('llama_free(g_state.ctx)');
  const adaptersFreed = freeRuntime.indexOf('free_lora_adapters();');
  assert.ok(
    contextFreed >= 0 && contextFreed < adaptersFreed
      && adaptersFreed < freeRuntime.indexOf('llama_model_free(g_state.model)'),
    'adapters are freed after the context that applies them and before their model',
  );
  assert.ok(
    CORE.includes('g_next_lora_handle++') && !/free_runtime[^]*g_next_lora_handle = /.test(freeRuntime),
    'core adapter handles are monotonic within a runtime',
  );
  const load = body('int32_t llamadart_webgpu_lora_load(');
  assert.ok(
    load.indexOf('llama_adapter_get_alora_n_invocation_tokens') < load.indexOf('llama_adapter_lora_free(adapter)')
      && load.indexOf('llama_adapter_lora_free(adapter)') < load.indexOf('g_lora_adapters[handle] = adapter'),
    'an aLoRA adapter is freed and never registered',
  );
  for (const signature of [
    'int32_t llamadart_webgpu_lora_load(',
    'int32_t llamadart_webgpu_lora_set(',
    'int32_t llamadart_webgpu_lora_remove(',
    'int32_t llamadart_webgpu_lora_clear(',
  ]) {
    assert.ok(body(signature).includes('check_lora_call('), `${signature} must refuse to run during generation`);
  }
  assert.ok(
    body('int32_t check_lora_call(').includes('g_generation_active || g_tts_active'),
    'LoRA calls must refuse to run during generation or text-to-speech',
  );
  assert.ok(
    body('void apply_active_loras() {').includes('g_cached_prompt_tokens.clear();'),
    'changing adapters must drop the cached prompt',
  );
}

// Smoke: the aLoRA adapter it builds and the payload checks.
{
  const u32 = (value) => { const b = Buffer.alloc(4); b.writeUInt32LE(value); return b; };
  const u64 = (value) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(value)); return b; };
  const str = (text) => Buffer.concat([u64(Buffer.byteLength(text)), Buffer.from(text)]);
  // GGUF v3: an alignment of 16, a string array and one 1-D tensor of 4 bytes.
  const header = Buffer.concat([
    Buffer.from('GGUF'), u32(3), u64(1), u64(2),
    str('general.alignment'), u32(4), u32(16),
    str('general.tags'), u32(9), u32(8), u64(2), str('a'), str('bc'),
    str('t'), u32(1), u64(4), u32(0), u64(0),
  ]);
  const tensorData = Buffer.from([1, 2, 3, 4]);
  const original = Buffer.concat([header, Buffer.alloc((16 - (header.length % 16)) % 16), tensorData]);
  const edited = smoke.withAloraInvocationTokens(original, [7, 8, 9]);
  const key = str('adapter.alora.invocation_tokens');
  const kvEnd = original.indexOf(str('t'), 24);
  assert.equal(edited.readBigUInt64LE(16), 3n, 'one metadata entry is added');
  assert.deepEqual(edited.subarray(24, kvEnd), original.subarray(24, kvEnd), 'existing metadata is kept');
  assert.deepEqual(
    edited.subarray(kvEnd, kvEnd + key.length + 28),
    Buffer.concat([key, u32(9), u32(4), u64(3), u32(7), u32(8), u32(9)]),
  );
  assert.equal(edited.length % 16, 4, 'tensor data stays aligned to general.alignment');
  assert.deepEqual(edited.subarray(-4), tensorData);
  assert.throws(() => smoke.withAloraInvocationTokens(Buffer.from('GGML0000'), [1]), /not a GGUF file/);

  const entry = (mode, extra = {}) => ({
    mode,
    failures: [],
    gpuActive: false,
    coreVariant: mode.split(' ')[0],
    execution: mode.endsWith('worker') ? 'worker' : 'main-thread',
    ...extra,
  });
  const modes = ['wasm32 direct', 'wasm32 worker'];
  const payload = (results, extra = {}) => ({ ok: true, modeResults: results, globalWorkerFallbackReason: null, ...extra });
  assert.deepEqual(smoke.validatePayload(payload(modes.map((mode) => entry(mode))), ['wasm32'], 0), []);
  assert.deepEqual(smoke.validatePayload({ ok: false, error: 'boom' }, ['wasm32'], 0), ['harness failed: boom']);
  assert.deepEqual(smoke.validatePayload(payload([entry(modes[0])]), ['wasm32'], 0), ['mode results missing']);
  assert.deepEqual(
    smoke.validatePayload(payload([entry(modes[0], { failures: ['scale 0: differs'] }), entry(modes[1])]), ['wasm32'], 0),
    ['wasm32 direct: scale 0: differs'],
  );
  assert.deepEqual(
    smoke.validatePayload(payload([entry(modes[0], { coreVariant: 'wasm64' }), entry(modes[1], { gpuActive: true })]), ['wasm32'], 0),
    ["wasm32 direct: expected the wasm32 core, got 'wasm64'", 'wasm32 worker: GPU active is True with 0 GPU layers'],
  );
  assert.deepEqual(
    smoke.validatePayload(payload(modes.map((mode) => entry(mode, { gpuActive: true }))), ['wasm32'], 99),
    [],
  );
  assert.deepEqual(
    smoke.validatePayload(payload(modes.map((mode) => entry(mode)), { globalWorkerFallbackReason: 'x' }), ['wasm32'], 0),
    ["worker fell back to the main thread: 'x'"],
  );

  const args = smoke.parseArgs(['--dist-dir', '/d']);
  assert.equal(args.modelUrl, smoke.DEFAULT_MODEL_URL);
  assert.equal(args.adapterSha256, smoke.DEFAULT_ADAPTER_SHA256);
  assert.equal(args.gpuLayers, 0);
  assert.match(smoke.DEFAULT_MODEL_URL, /\/resolve\/[0-9a-f]{40}\//, 'the default model is pinned to a revision');
  assert.match(smoke.DEFAULT_ADAPTER_URL, /\/resolve\/[0-9a-f]{40}\//, 'the default adapter is pinned to a revision');
  const harness = smoke.renderHarness({ memoryModes: ['wasm32'], runtimeModes: smoke.RUNTIME_MODES });
  for (const method of ['setLoraAdapter(adapter.handle, 0)', 'removeLoraAdapter', 'clearLoraAdapters', 'aLoRA adapter']) {
    assert.ok(harness.includes(method), `the smoke harness must exercise ${method}`);
  }
}

console.log('LoRA adapter contract tests passed');
