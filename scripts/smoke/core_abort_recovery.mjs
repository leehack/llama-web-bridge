#!/usr/bin/env node

import assert from 'node:assert/strict';
import fsp from 'node:fs/promises';
import path from 'node:path';

import { buildLargeOutputGguf } from './synthetic_gguf.mjs';
import {
  DEFAULT_MODEL_CACHE, copyBridgeArtifacts, copyMemory64Artifacts, copySingleThreadArtifacts,
  env, parseSmokeArgs, resolvePinnedModel, runMain, runPollingPlaywright,
  stageFile, webGpuLaunchArgs, withServer, withTempDir,
} from './support.mjs';

const STAGED_CASES = ['staged-load', 'staged-load-no-mmap'];
const LATE_READER_CASES = ['reader-late-mmap', 'reader-late-no-mmap'];
const READER_CASES = ['reader-header', 'reader-tensor', 'reader-tensor-no-mmap', 'reader-cancel', 'reader-cold', ...LATE_READER_CASES];
const SOURCE_CASES = ['source-blob', 'source-opfs', 'source-large', 'source-fault', 'source-cancel', 'source-restart', 'source-busy', 'source-lock', 'source-rejected'];
const LEAK_CASE = 'failed-load-leak';
const DEFAULT_CASES = ['load', 'cpp-throw', 'next-token', 'cancel-cleanup', 'state-write', ...STAGED_CASES, 'healthy', LEAK_CASE, ...READER_CASES, ...SOURCE_CASES];
const LARGE_MODEL_CASES = [...LATE_READER_CASES, 'source-large', 'source-fault', 'source-cancel'];
const IFRAME_CASES = ['iframe', 'older-iframe'];
const MODEL = 'abort-smoke.gguf';
const TRUNCATED = 'abort-smoke-truncated.gguf';
const LARGE = 'abort-smoke-large.gguf';

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
    const result = originalCcall(name, ...args);
    if (name !== 'llamadart_webgpu_load_model') return result;
    return Promise.resolve(result).then((rc) => {
      report('load-rc', { rc });
      return rc;
    });
  };
  // Wraps the next bound model reader so a read fails or waits at a chosen point.
  let armedReader = null;
  const originalBind = core.llamadartJsModelFsBindExternal;
  if (originalBind) {
    core.llamadartJsModelFsBindExternal = (handle, mountId, reader) => {
      report('bind', { mode: reader.mode, size: reader.size });
      const fault = armedReader;
      armedReader = null;
      if (!fault) return originalBind(handle, mountId, reader);
      let injected = false;
      const wrapped = {
        get mode() { return reader.mode; },
        get size() { return reader.size; },
        get signal() { return reader.signal; },
        get failure() { return reader.failure; },
        set failure(value) { reader.failure = value; },
        close: () => reader.close(),
        read: (offset, length, copy) => {
          // The header is read in small pieces. A whole-file mapping or a tensor
          // read is a large read, or one that starts in the second half.
          const header = length <= 65536 && offset < reader.size / 2;
          // A late fault is damage past the first 16 MiB step of a larger read.
          const hit = fault === 'late' ? offset >= 16 * 1024 * 1024
            : fault === 'header' ? header : !injected && !header;
          if (!hit) return reader.read(offset, length, copy);
          if (!injected) report('reader-fault', { fault, offset, length });
          injected = true;
          if (fault !== 'cancel') throw new Error('injected model read failure');
          return (async () => {
            for (let i = 0; i < 500 && !reader.signal?.aborted; i += 1) {
              await new Promise((resolve) => setTimeout(resolve, 10));
            }
            return reader.read(offset, length, copy);
          })();
        },
      };
      return originalBind(handle, mountId, wrapped);
    };
  }
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
      if (data.point === 'reader') {
        armedReader = data.fault;
        // A synchronous OPFS read cannot wait for a cancel, so a cancel reads the
        // Blob that browsers without exclusive sync handles fall back to.
        if (data.fault === 'cancel' && typeof FileSystemFileHandle !== 'undefined') {
          Object.defineProperty(FileSystemFileHandle.prototype, 'createSyncAccessHandle',
            { value: undefined, configurable: true });
        }
      } else armed = { point: data.point, fault: data.fault };
      report('armed');
    }
    if (data.command === 'snapshot') {
      const headroom = Number(originalCcall('llamadart_webgpu_heap_headroom_bytes', 'number', [], []));
      report('snapshot', { request: data.request, created, terminated, headroom });
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
  const { scenario, modelUrl: smallModelUrl, largeUrl, truncatedUrl, worker, single, memory64, staging, coreName } = settings;
  const modelUrl = scenario.startsWith('reader-late-') ? largeUrl : smallModelUrl;
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
    if (scenario.startsWith('reader-')) {
      // A model that llama.cpp cannot read through an external reader fails the
      // native load with -2. The core stays usable, a damaged cache entry is
      // replaced once, and a cancel leaves the next load unaffected.
      const cacheName = 'reader-smoke-' + crypto.randomUUID();
      const readerOptions = {
        nCtx: 128, nThreads, nThreadsBatch: nThreads, nGpuLayers: 0, useCache: true, cacheName,
        ...(scenario.endsWith('-no-mmap') ? { useMmap: false } : {}),
        ...(scenario === 'reader-late-mmap' ? { useMmap: true } : {}),
      };
      const loadCached = (extra = {}) => bridge.loadModelFromUrl(modelUrl, { ...readerOptions, ...extra });
      // The synthetic model has no byte-fallback tokens, so its prompt uses only its own pieces.
      const completeText = () => bridge.createCompletion(modelUrl === largeUrl ? 'a b c' : 'hello', { nPredict: 8, temp: 0, seed: 1 });
      const notes = () => (bridge.getModelMetadata()['llamadart.webgpu.runtime_notes'] || '').split(';');
      if (scenario === 'reader-cold') {
        // A download that was just streamed into OPFS is not a cache entry anyone
        // could have damaged, so a failed read there is reported, not retried.
        await bridge.loadModelFromUrl(modelUrl, { ...readerOptions, useCache: false });
        const core = await waitFor(() => events.find((event) => event.type === 'created'), 'No core creation event');
        const mark = events.length;
        await arm(core.id, 'reader', 'header');
        await expectError(() => loadCached(),
          (error) => error.name === 'ModelStagingError' && error.code === 'MODEL_READ_FAILED');
        assert(events.slice(mark).some((event) => event.type === 'bind' && event.mode === 0), 'The cold load did not read through OPFS');
        assert(!events.some((event) => event.type === 'abort' || event.type === 'retired'), 'A read failure aborted or retired the core');
        await loadCached();
        const recovered = await completeText();
        assert(typeof recovered === 'string' && recovered.length > 0, 'Generation failed after the read failure');
        assert(events.filter((event) => event.type === 'created').length === 1, 'A read failure replaced the core');
        await bridge.evictModelFromCache(modelUrl, { cacheName });
        result.ok = true;
        return;
      }
      window.__smokeStage = scenario + ':cold';
      await loadCached();
      const created = await waitFor(() => events.find((event) => event.type === 'created'), 'No core creation event');
      assert(notes().includes(worker ? 'model_cache_opfs' : 'model_cache_stored'), 'The cold load did not store the model');
      assert(await bridge.isModelCached(modelUrl, { cacheName }), 'The cold load did not leave a cached model');
      const fault = scenario.startsWith('reader-late-') ? 'late' : scenario.slice('reader-'.length).replace('-no-mmap', '');
      const clean = fault === 'late' ? await completeText() : null;
      const from = events.length;
      window.__smokeStage = scenario + ':fault';
      await arm(created.id, 'reader', fault);
      if (fault === 'cancel') {
        const controller = new AbortController();
        const aborting = waitFor(() => events.slice(from).find((event) => event.type === 'reader-fault'),
          'The load never reached a tensor read').then(() => controller.abort());
        await expectError(() => loadCached({ signal: controller.signal }), (error) => error.name === 'AbortError');
        await aborting;
        assert(await bridge.isModelCached(modelUrl, { cacheName }), 'A cancelled load evicted the cache entry');
      } else {
        await loadCached();
        if (clean !== null) {
          const reloaded = await completeText();
          assert(reloaded === clean, `A read that failed late changed the output: ${JSON.stringify(clean)} then ${JSON.stringify(reloaded)}`);
        }
        assert(notes().includes('model_cache_entry_invalid') && notes().includes('model_stream_restart:cache'),
          'The damaged cache entry was not replaced: ' + notes().join(','));
        assert(await bridge.isModelCached(modelUrl, { cacheName }), 'The replaced model is not cached');
      }
      const faultEvents = events.slice(from);
      const rcs = faultEvents.filter((event) => event.type === 'load-rc').map((event) => event.rc);
      const bound = faultEvents.find((event) => event.type === 'bind');
      assert(bound, 'The load did not use an external reader');
      assert(bound.mode === (worker && fault !== 'cancel' ? 0 : 1), `Unexpected reader mode ${bound.mode}`);
      // A mapped load may finish with rc 0 over zeroed bytes; the bridge must then reject it.
      assert(rcs[0] === -2 || (fault === 'late' && rcs[0] === 0), `The failed read did not fail the load: ${rcs}`);
      assert(!events.some((event) => event.type === 'abort'), 'A read failure called onAbort');
      assert(!bridge._coreRetirements && !events.some((event) => event.type === 'retired'), 'A read failure retired the core');
      window.__smokeStage = scenario + ':reload';
      if (fault === 'cancel') await loadCached();
      const text = await completeText();
      assert(typeof text === 'string' && text.length > 0, 'Generation failed after the read failure');
      assert(events.filter((event) => event.type === 'created').length === 1, 'A read failure replaced the core');
      await bridge.evictModelFromCache(modelUrl, { cacheName });
      Object.assign(result, { ok: true, rcs });
      return;
    }

    if (scenario.startsWith('source-')) {
      // A caller's Blob, File or OPFS handle is read on demand and never copied.
      // A damaged or cancelled source leaves the core usable, and a restart
      // reloads the caller's own reference or refuses a changed one.
      const opfsRoot = await navigator.storage.getDirectory();
      const opfsNames = [];
      const sourceOptions = { nCtx: 128, nThreads, nThreadsBatch: nThreads, nGpuLayers: 0 };
      const fetchBytes = async (url) => new Uint8Array(await (await fetch(url)).arrayBuffer());
      // A worker the bridge replaced or disposed lets go of the file a moment later, so a
      // caller waiting for that passes `attempts`. A write that must work at once passes 1.
      const rewrite = async (handle, bytes, attempts = 100) => {
        for (let attempt = 0; ; attempt += 1) {
          try {
            const writable = await handle.createWritable();
            await writable.write(bytes);
            await writable.close();
            return handle;
          } catch (error) {
            if (attempt + 1 >= attempts) throw error;
            await new Promise((resolve) => setTimeout(resolve, 30));
          }
        }
      };
      // Writes that keep the size, and on Chromium's in-memory OPFS the modification
      // time too, so only the sampled content fingerprint can tell.
      const inPlaceWrite = async (handle, position, bytes) => {
        for (let attempt = 0; ; attempt += 1) {
          try {
            const writable = await handle.createWritable({ keepExistingData: true });
            await writable.write({ type: 'write', position, data: bytes });
            await writable.close();
            return;
          } catch (error) {
            if (attempt >= 100) throw error;
            await new Promise((resolve) => setTimeout(resolve, 30));
          }
        }
      };
      const lockerSource = `
        self.onmessage = async ({ data }) => {
          try {
            const sync = await data.handle.createSyncAccessHandle();
            if (data.op === 'write') {
              sync.write(data.bytes, { at: data.position });
              sync.flush();
              sync.close();
              self.postMessage({ done: true });
              return;
            }
            self.postMessage({ held: true });
            self.onmessage = () => { sync.close(); self.postMessage({ released: true }); };
          } catch (error) {
            self.postMessage({ error: error.name });
          }
        };`;
      const lockerWorker = (message) => new Promise((resolve) => {
        const url = URL.createObjectURL(new Blob([lockerSource], { type: 'text/javascript' }));
        const worker = new Worker(url);
        worker.onmessage = ({ data }) => resolve({ worker, data, url });
        worker.postMessage(message);
      });
      const syncWrite = async (handle, position, bytes) => {
        for (let attempt = 0; ; attempt += 1) {
          const { worker, data, url } = await lockerWorker({ op: 'write', handle, position, bytes });
          worker.terminate();
          URL.revokeObjectURL(url);
          if (data.done) return;
          if (attempt >= 100) throw new Error('A sync access handle write failed: ' + data.error);
          await new Promise((resolve) => setTimeout(resolve, 30));
        }
      };
      const holdSync = async (handle) => {
        const { worker, data, url } = await lockerWorker({ op: 'hold', handle });
        assert(data.held, 'The test could not hold a sync access handle: ' + data.error);
        return async () => {
          await new Promise((resolve) => { worker.onmessage = resolve; worker.postMessage('release'); });
          worker.terminate();
          URL.revokeObjectURL(url);
        };
      };
      // One attempt, no retry: what a caller sees right now.
      const tryWrite = async (handle) => {
        try {
          const writable = await handle.createWritable({ keepExistingData: true });
          await writable.close();
          return 'allowed';
        } catch (error) {
          return error.name;
        }
      };
      const trySyncOpen = async (handle) => {
        const { worker, data, url } = await lockerWorker({ op: 'hold', handle });
        if (data.held) {
          await new Promise((resolve) => { worker.onmessage = resolve; worker.postMessage('release'); });
        }
        worker.terminate();
        URL.revokeObjectURL(url);
        return data.held ? 'allowed' : data.error;
      };
      const expectLocked = async (handle, what) => {
        const writer = await tryWrite(handle);
        const sync = await trySyncOpen(handle);
        assert(writer === 'NoModificationAllowedError', `${what}: createWritable was ${writer}, expected the file to be locked`);
        assert(sync === 'NoModificationAllowedError', `${what}: a sync access handle was ${sync}, expected the file to be locked`);
      };
      const expectUnlocked = async (handle, what) => {
        const writer = await tryWrite(handle);
        const sync = await trySyncOpen(handle);
        assert(writer === 'allowed' && sync === 'allowed', `${what}: file still locked (${writer}, ${sync})`);
      };
      const writeOpfs = async (bytes) => {
        const name = `source-smoke-${crypto.randomUUID()}.gguf`;
        opfsNames.push(name);
        return rewrite(await opfsRoot.getFileHandle(name, { create: true }), bytes);
      };
      const smallBytes = await fetchBytes(smallModelUrl);
      const notes = () => (bridge.getModelMetadata()['llamadart.webgpu.runtime_notes'] || '').split(';');
      const generate = (prompt) => bridge.createCompletion(prompt, { nPredict: 8, temp: 0, seed: 1 });
      const smallPrompt = 'hello';
      const largePrompt = 'a b c';
      const created = () => events.filter((event) => event.type === 'created');
      // Core events arrive on a BroadcastChannel, one task after the call that sent them.
      const settle = () => new Promise((resolve) => setTimeout(resolve, 100));
      const countOf = (type) => events.filter((event) => event.type === type).length;
      const noAbort = async ({ replacedWorkers = false } = {}) => {
        await settle();
        assert(!events.some((event) => event.type === 'abort' || (!replacedWorkers && event.type === 'retired')),
          'A source error aborted or retired the core');
        assert(!bridge._coreRetirements, 'A source error retired the core');
      };
      const expectSource = async (kind, size) => {
        await settle();
        assert(notes().includes('model_source:caller'), 'The load did not report a caller source: ' + notes().join(','));
        assert(notes().includes('model_reader:' + kind), `Expected reader ${kind}: ` + notes().join(','));
        assert(!notes().some((note) => note.startsWith('model_cache_')), 'A caller source touched the model cache');
        const metadata = bridge.getModelMetadata();
        assert(metadata['llamadart.webgpu.model_source'] === 'caller', 'Unexpected model source ' + metadata['llamadart.webgpu.model_source']);
        assert(metadata['llamadart.webgpu.model_cache_state'] === 'disabled', 'Unexpected cache state');
        const bind = events.filter((event) => event.type === 'bind').at(-1);
        assert(bind && bind.size === size, `The reader did not cover the file: ${bind?.size} of ${size}`);
      };
      const referenceFor = async (url, prompt) => {
        await bridge.loadModelFromUrl(url, { ...sourceOptions, useCache: false });
        const text = await generate(prompt);
        assert(typeof text === 'string' && text.length > 0, 'The URL load generated nothing');
        return text;
      };
      const firstCore = () => waitFor(() => created()[0], 'No core creation event');
      const syncKind = worker ? 'caller_opfs_sync' : 'caller_opfs_blob';
      try {
        if (scenario === 'source-blob') {
          const reference = await referenceFor(smallModelUrl, smallPrompt);
          const core = await firstCore();
          await bridge.loadModelFromSource(new Blob([smallBytes]), sourceOptions);
          await expectSource('caller_blob', smallBytes.length);
          assert(await generate(smallPrompt) === reference, 'A Blob source generated different text');
          await bridge.loadModelFromSource(new File([smallBytes], 'weights.gguf', { lastModified: 12345 }), sourceOptions);
          await expectSource('caller_file', smallBytes.length);
          assert(await generate(smallPrompt) === reference, 'A File source generated different text');
          const preflight = await bridge.preflightCore();
          assert(preflight.externalModelReader === true, 'The core did not report externalModelReader');
          assert(LlamaWebGpuBridge.supportsCallerModelSource === true, 'The bridge did not report supportsCallerModelSource');
          await bridge.loadModelFromUrl(smallModelUrl, { ...sourceOptions, useCache: false });
          assert(bridge.getModelMetadata()['llamadart.webgpu.model_source'] !== 'caller', 'A URL load kept the caller source label');
          assert(await generate(smallPrompt) === reference, 'A URL load after a source load generated different text');
          await noAbort();
          assert(created().length === 1 && created()[0].id === core.id, 'A source load replaced the core');
        } else if (scenario === 'source-opfs') {
          const reference = await referenceFor(smallModelUrl, smallPrompt);
          const handle = await writeOpfs(smallBytes);
          const before = await handle.getFile();
          await bridge.loadModelFromSource(handle, sourceOptions);
          await expectSource(syncKind, smallBytes.length);
          assert(await generate(smallPrompt) === reference, 'An OPFS source generated different text');
          const after = await handle.getFile();
          assert(after.size === before.size && after.lastModified === before.lastModified, 'The load changed the caller file');
          if (worker) await expectLocked(handle, 'a loaded OPFS source');
          await noAbort();
          // Disposal closes the bridge's own sync access handle, so the caller can write again.
          await bridge.dispose();
          bridge = null;
          await rewrite(handle, smallBytes);
        } else if (scenario === 'source-large') {
          const largeBytes = await fetchBytes(largeUrl);
          assert(largeBytes.length > 32 * 1024 * 1024, 'The large fixture is not larger than two read steps');
          const reference = await referenceFor(largeUrl, largePrompt);
          // A mapping would copy the whole file into the heap, so it is ignored.
          await bridge.loadModelFromSource(new Blob([largeBytes]), { ...sourceOptions, useMmap: true });
          await expectSource('caller_blob', largeBytes.length);
          assert(notes().includes('model_mmap_disabled:external_reader'), 'A caller source was mapped: ' + notes().join(','));
          assert(await generate(largePrompt) === reference, 'A large Blob generated different text');
          await bridge.loadModelFromSource(await writeOpfs(largeBytes), sourceOptions);
          await expectSource(syncKind, largeBytes.length);
          assert(await generate(largePrompt) === reference, 'A large OPFS file generated different text');
          await noAbort();
          assert(created().length === 1, 'A source load replaced the core');
        } else if (scenario === 'source-fault') {
          const largeBytes = await fetchBytes(largeUrl);
          const reference = await referenceFor(largeUrl, largePrompt);
          const core = await firstCore();
          for (const [fault, makeSource] of [
            ['late', async () => new Blob([largeBytes])],
            ['tensor', async () => new Blob([largeBytes])],
            ['late', () => writeOpfs(largeBytes)],
          ]) {
            const from = events.length;
            await arm(core.id, 'reader', fault);
            await expectError(async () => bridge.loadModelFromSource(await makeSource(), sourceOptions),
              (error) => error.name === 'ModelStagingError' && error.code === 'MODEL_READ_FAILED');
            await settle();
            assert(events.slice(from).some((event) => event.type === 'reader-fault'), `The ${fault} fault never fired`);
            assert(!events.slice(from).some((event) => event.type === 'load-rc' && event.rc === 0), 'A damaged source reported a loaded model');
            await noAbort();
          }
          await bridge.loadModelFromSource(new Blob([largeBytes]), sourceOptions);
          assert(await generate(largePrompt) === reference, 'The core generated different text after a damaged source');
          assert(created().length === 1, 'A damaged source replaced the core');
        } else if (scenario === 'source-cancel') {
          const largeBytes = await fetchBytes(largeUrl);
          const reference = await referenceFor(largeUrl, largePrompt);
          const core = await firstCore();
          const from = events.length;
          await arm(core.id, 'reader', 'cancel');
          const controller = new AbortController();
          const aborting = waitFor(() => events.slice(from).find((event) => event.type === 'reader-fault'),
            'The load never reached a source read').then(() => controller.abort());
          await expectError(() => bridge.loadModelFromSource(new Blob([largeBytes]), { ...sourceOptions, signal: controller.signal }),
            (error) => error.name === 'AbortError');
          await aborting;
          await bridge.loadModelFromSource(new Blob([largeBytes]), sourceOptions);
          assert(await generate(largePrompt) === reference, 'The core generated different text after a cancelled source load');
          await noAbort();
          assert(created().length === 1, 'A cancelled source load replaced the core');
        } else if (scenario === 'source-restart') {
          const reference = await referenceFor(smallModelUrl, smallPrompt);
          const handle = await writeOpfs(smallBytes);
          const restart = async () => {
            if (worker) {
              await bridge._replaceWorkerProxyForMultimodalCpuMode();
              await bridge._loadRememberedModelIntoWorker(bridge._sanitizeModelLoadOptions(bridge._loadedModelOptions));
            } else {
              assert(await bridge._runtime._recoverGenerationWithCpuFallback({}) === true, 'CPU recovery did not reload the source');
            }
          };
          for (const [source, kind] of [[new Blob([smallBytes]), 'caller_blob'], [handle, syncKind]]) {
            await bridge.loadModelFromSource(source, sourceOptions);
            await settle();
            const binds = countOf('bind');
            await restart();
            await settle();
            assert(countOf('bind') === binds + 1, 'The restart did not reopen the source');
            await expectSource(kind, smallBytes.length);
            assert(await generate(smallPrompt) === reference, 'A restarted source generated different text');
          }
          // A file that changed since it was handed over is refused, never loaded: a
          // rewrite with other bytes, a shorter file, and two writes in place that keep
          // the size (and on an in-memory OPFS the modification time as well).
          const flipped = smallBytes.slice();
          flipped[1000] ^= 1;
          const tail = smallBytes.length - 4096;
          const patch = new Uint8Array(4096).fill(0x7f);
          const changes = [
            ['rewrite', (target) => rewrite(target, flipped)],
            ['shorter', (target) => rewrite(target, smallBytes.subarray(0, smallBytes.length - 1024))],
            ['in-place-writable', (target) => inPlaceWrite(target, tail, patch)],
            ['in-place-sync', (target) => syncWrite(target, tail, patch)],
          ];
          for (const [label, change] of changes) {
            await bridge.loadModelFromSource(handle, sourceOptions);
            const described = await handle.getFile();
            // Replacing the worker releases its sync access handle, so the file can be written.
            if (worker) await bridge._replaceWorkerProxyForMultimodalCpuMode();
            await change(handle);
            const edited = await handle.getFile();
            result[`${worker ? 'worker' : 'direct'}-${label}-kept-metadata`] =
              edited.size === described.size && edited.lastModified === described.lastModified;
            await settle();
            const loads = countOf('load-rc');
            const binds = countOf('bind');
            if (worker) {
              await expectError(
                () => bridge._loadRememberedModelIntoWorker(bridge._sanitizeModelLoadOptions(bridge._loadedModelOptions)),
                (error) => error.name === 'ModelStagingError' && error.code === 'MODEL_SOURCE_CHANGED');
            } else {
              assert(await bridge._runtime._recoverGenerationWithCpuFallback({}) === false, `CPU recovery loaded a changed source (${label})`);
            }
            await settle();
            assert(countOf('load-rc') === loads, `A changed source reached the native loader (${label})`);
            assert(countOf('bind') === binds, `A changed source was attached to the core (${label})`);
            await rewrite(handle, smallBytes);
          }
          if (worker) {
            await bridge.loadModelFromSource(new Blob([smallBytes]), sourceOptions);
            await bridge._disableWorkerFallback(new Error('simulated worker loss'));
            await bridge._ensureRuntimeReadyAfterWorkerFallback({ _llamadartForceRuntimeReload: true });
            assert(await generate(smallPrompt) === reference, 'The main-thread fallback generated different text');
            await expectSource('caller_blob', smallBytes.length);
          }
          await noAbort({ replacedWorkers: worker });
        } else if (scenario === 'source-busy') {
          const reference = await referenceFor(smallModelUrl, smallPrompt);
          const core = await firstCore();
          const handle = await writeOpfs(smallBytes);
          const release = await holdSync(handle);
          try {
            if (worker) {
              // Another context holds the file, so the worker refuses it rather than read a live file.
              await settle();
              const loads = countOf('load-rc');
              const binds = countOf('bind');
              await expectError(() => bridge.loadModelFromSource(handle, sourceOptions),
                (error) => error.name === 'ModelStagingError' && error.code === 'MODEL_SOURCE_BUSY');
              await settle();
              assert(countOf('load-rc') === loads && countOf('bind') === binds, 'A busy source reached the core');
              await bridge.loadModelFromSource(new Blob([smallBytes]), sourceOptions);
              assert(await generate(smallPrompt) === reference, 'The core generated different text after a busy source');
            } else {
              // The main thread holds no lock, so it reads the live file and rechecks it after the load.
              await bridge.loadModelFromSource(handle, sourceOptions);
              await expectSource('caller_opfs_blob', smallBytes.length);
              assert(await generate(smallPrompt) === reference, 'A live file source generated different text');
            }
          } finally {
            await release();
          }
          if (worker) {
            await bridge.loadModelFromSource(handle, sourceOptions);
            await expectSource('caller_opfs_sync', smallBytes.length);
            assert(await generate(smallPrompt) === reference, 'A source generated different text once it was free');
          }
          await noAbort();
          assert(created().length === 1 && created()[0].id === core.id, 'A busy source replaced the core');
        } else if (scenario === 'source-lock') {
          const reference = await referenceFor(smallModelUrl, smallPrompt);
          const core = await firstCore();
          const n = smallBytes.length;
          const stripe = 64 * 1024;
          const starts = [];
          for (let index = 1; index <= 16; index += 1) starts.push(Math.floor((index * (n - stripe)) / 17));
          assert(starts[8] - starts[7] > stripe + 8192, 'The fixture has no unsampled gap');
          const gapAt = starts[7] + stripe + 2048;
          const gapPatch = new Uint8Array(4096).fill(0x7f);
          const handle = await writeOpfs(smallBytes);
          const replay = async () => {
            if (worker) {
              await bridge._loadRememberedModelIntoWorker(bridge._sanitizeModelLoadOptions(bridge._loadedModelOptions));
              return true;
            }
            return bridge._runtime._recoverGenerationWithCpuFallback({});
          };
          await bridge.loadModelFromSource(handle, sourceOptions);
          await expectSource(syncKind, n);
          if (worker) {
            // The worker keeps its exclusive sync access handle for as long as the model is loaded.
            await expectLocked(handle, 'after the load');
            assert(await generate(smallPrompt) === reference, 'A locked source generated different text');
            await replay();
            await expectSource('caller_opfs_sync', n);
            await expectLocked(handle, 'after an in-worker reload');
            assert(await generate(smallPrompt) === reference, 'A reloaded locked source generated different text');
            // Another source replaces the model and lets go of the file at once.
            await bridge.loadModelFromSource(new Blob([smallBytes]), sourceOptions);
            await expectUnlocked(handle, 'after another source replaced the model');
            // A load that fails after the file was attached lets go of it too.
            const damaged = await writeOpfs(smallBytes.subarray(0, n - 200000));
            await expectError(() => bridge.loadModelFromSource(damaged, sourceOptions), () => true);
            await expectUnlocked(damaged, 'after a failed load');
            await bridge.loadModelFromSource(handle, sourceOptions);
            await expectLocked(handle, 'after a reload');
            // Disposing the bridge ends the worker, which releases the lock.
            await bridge.dispose();
            bridge = null;
            await rewrite(handle, smallBytes);
          } else {
            // The main thread holds no lock: an edit between the sampled stripes goes
            // unnoticed (documented), and one inside a stripe is refused.
            assert(await tryWrite(handle) === 'allowed', 'The main thread should not lock the file');
            await rewrite(handle, smallBytes, 1);
            await bridge.loadModelFromSource(handle, sourceOptions);
            await inPlaceWrite(handle, gapAt, gapPatch);
            assert(await replay() === true, 'An edit between the sampled stripes was expected to reload');
            await rewrite(handle, smallBytes, 1);
            await bridge.loadModelFromSource(handle, sourceOptions);
            await inPlaceWrite(handle, n - 4096, gapPatch);
            assert(await replay() === false, 'An edit inside a sampled stripe was reloaded');
            await rewrite(handle, smallBytes, 1);
          }
          if (bridge) {
            await noAbort();
            assert(created().length === 1 && created()[0].id === core.id, 'A locked source replaced the core');
          }
        } else if (scenario === 'source-rejected') {
          const reference = await referenceFor(smallModelUrl, smallPrompt);
          const core = await firstCore();
          const u32 = (value) => new Uint8Array(new Uint32Array([value]).buffer);
          const u64 = (value) => new Uint8Array(new BigUint64Array([BigInt(value)]).buffer);
          const text = (value) => { const bytes = new TextEncoder().encode(value); return [u64(bytes.length), bytes]; };
          const join = (parts) => {
            const flat = parts.flat();
            const out = new Uint8Array(flat.reduce((total, part) => total + part.length, 0));
            let at = 0;
            for (const part of flat) { out.set(part, at); at += part.length; }
            return out;
          };
          const splitGguf = join([
            [new TextEncoder().encode('GGUF'), u32(3), u64(0), u64(2)],
            text('general.architecture'), [u32(8)], text('llama'),
            text('split.count'), [u32(2), new Uint8Array([3, 0])],
            [new Uint8Array(64)],
          ]);
          const directory = await opfsRoot.getDirectoryHandle('source-smoke-dir', { create: true });
          try {
            for (const [code, source] of [
              ['MODEL_SOURCE_SPLIT_UNSUPPORTED', new Blob([splitGguf])],
              ['MODEL_SOURCE_INVALID', new Blob([new TextEncoder().encode('this is not a model file')])],
              ['MODEL_SOURCE_INVALID', new Blob([])],
              ['MODEL_SOURCE_INVALID', 'https://example.test/model.gguf'],
              ['MODEL_SOURCE_INVALID', directory],
            ]) {
              await expectError(() => bridge.loadModelFromSource(source, sourceOptions),
                (error) => error.name === 'ModelStagingError' && error.code === code);
            }
          } finally {
            await opfsRoot.removeEntry('source-smoke-dir');
          }
          await bridge.loadModelFromSource(new Blob([smallBytes]), sourceOptions);
          assert(await generate(smallPrompt) === reference, 'The core generated different text after rejected sources');
          await noAbort();
          assert(created().length === 1 && created()[0].id === core.id, 'A rejected source replaced the core');
        }
        result.ok = true;
      } finally {
        await bridge?.dispose().catch(() => {});
        bridge = null;
        for (const name of opfsNames) {
          for (let attempt = 0; attempt < 100; attempt += 1) {
            try {
              await opfsRoot.removeEntry(name);
              break;
            } catch (_) {
              await new Promise((resolve) => setTimeout(resolve, 30));
            }
          }
        }
      }
      return;
    }
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
    if (scenario === 'failed-load-leak') {
      // llama.cpp frees what a failed load allocated only when its own unwind runs.
      // Without it every failure keeps its loader's buffers in the Wasm heap for good.
      const failLoad = () => expectError(() => load(truncatedUrl), (error) => error.message.startsWith('Failed to load GGUF model'));
      for (let i = 0; i < 3; i += 1) await failLoad();
      const warm = await snapshot(first.id);
      assert(warm.headroom > 0, 'The core did not report its heap headroom');
      for (let i = 0; i < 60; i += 1) await failLoad();
      const settled = await snapshot(first.id);
      const growth = warm.headroom - settled.headroom;
      assert(growth <= 2 * 1024 * 1024, `60 failed loads used ${growth} bytes of heap headroom`);
      await load();
      const text = await complete();
      assert(typeof text === 'string' && text.length > 0, 'Generation failed after the failed loads');
      assert(!events.some(({ type }) => type === 'abort' || type === 'retired'), 'A failed load aborted or retired the core');
      Object.assign(result, { ok: true, growth });
      return;
    }
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
    if (cases.some(({ scenario }) => LARGE_MODEL_CASES.includes(scenario))) {
      await fsp.writeFile(path.join(webRoot, LARGE), buildLargeOutputGguf());
    }
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
            modelUrl: '/' + MODEL, largeUrl: '/' + LARGE, truncatedUrl: '/' + TRUNCATED,
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
      .filter((scenario) => worker || scenario !== 'reader-cold')
      .map((scenario) => ({ scenario, worker, staging: STAGED_CASES.includes(scenario) || READER_CASES.includes(scenario) || SOURCE_CASES.includes(scenario) }));
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
