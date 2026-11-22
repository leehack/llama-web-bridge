import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import { setImmediate } from 'node:timers/promises';

import * as api from '../../js/src/llama_webgpu_bridge.js';
import { retireCore } from '../../js/src/internal/core_guard.ts';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';
import {
  createRealWorkerBridge,
  createStubCore,
  withStubWorkerEnvironment,
  workerDriver,
} from './bridge_operation_queue_fixtures.mjs';

const base = 'https://assets.test/bridge/';
const loader = `${base}llama_webgpu_core_st.js`;
const wasm = `${base}llama_webgpu_core_st.wasm`;
const loader64 = `${base}llama_webgpu_core_mem64_st.js`;
const wasm64 = `${base}llama_webgpu_core_mem64_st.wasm`;
const model = 'https://models.test/model.gguf';
const loadOptions = { nGpuLayers: 0, nThreads: 1, useCache: false };
const config32 = { coreModuleUrl: loader, allowSingleThreadCore: true, preferMemory64: false };
const config64 = { ...config32, coreModuleUrlMem64: loader64, preferMemory64: true };
const signedLoader = 'https://user:pass@assets.test/bridge/llama_webgpu_core_st.js?v=0.1.92&token=secret#frag';
let fixtureSerial = 0;

function setGlobal(t, key, value) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, key);
  Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  t.after(() => {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  });
}

function withImports(t, entries) {
  const id = `pf-fixture-${++fixtureSerial}`;
  const imports = [];
  const modules = new Map(Object.entries(entries).map(([specifier, entry], index) => [
    specifier, { entry, url: `pf-core:${id}/${index}` },
  ]));
  const hook = registerHooks({
    resolve(specifier, context, next) {
      const module = modules.get(specifier);
      if (!module) return next(specifier, context);
      imports.push(specifier);
      if (module.entry instanceof Error) throw module.entry;
      return { shortCircuit: true, url: module.url };
    },
    load(url, context, next) {
      if (!url.startsWith(`pf-core:${id}/`)) return next(url, context);
      const { entry } = [...modules.values()].find((module) => module.url === url);
      const source = typeof entry === 'function'
        ? `export default globalThis[${JSON.stringify(id)}].find(({url}) => url === ${JSON.stringify(url)}).entry;`
        : entry;
      return { shortCircuit: true, format: 'module', source };
    },
  });
  setGlobal(t, id, [...modules.values()]);
  t.after(() => hook.deregister());
  return imports;
}

function makeCore({ threaded = false, trace = [] } = {}) {
  const core = createStubCore();
  const ccall = core.ccall.bind(core);
  core.ccall = (name, ...args) => {
    if (name === 'llamadart_webgpu_supports_pthreads') return Number(threaded);
    if (name === 'llamadart_webgpu_backends_json') return '["CPU"]';
    if (name === 'llamadart_webgpu_probe') trace.push('probe');
    if (['llamadart_webgpu_probe', 'llamadart_webgpu_load_model', 'llamadart_webgpu_free_model'].includes(name)) {
      return 0;
    }
    return ccall(name, ...args);
  };
  return core;
}

function wasmFactory({ file = wasm, fetchBinary = false, trace = [], threaded = false } = {}) {
  return async ({ locateFile }) => {
    trace.push('factory');
    const url = locateFile(new URL(file).pathname.split('/').at(-1), base);
    if (fetchBinary) {
      const response = await fetch(url);
      if (!response.ok) throw new WebAssembly.CompileError('wasm startup failed');
    }
    return makeCore({ threaded, trace });
  };
}

function assertApi() {
  assert.equal(api.LlamaWebGpuBridge.supportsCorePreflight, true);
  assert.equal(typeof api.CoreAssetError, 'function');
}

async function withBridge(mode, config, run, { startupFailure } = {}) {
  assertApi();
  if (mode === 'direct') {
    const bridge = new api.LlamaWebGpuBridge({ ...config, disableWorker: true });
    try {
      return await run(bridge, []);
    } finally {
      await bridge.dispose();
    }
  }
  return withStubWorkerEnvironment(async ({ workers }) => {
    const bridge = createRealWorkerBridge(config);
    const worker = workers[0];
    const previousSelf = globalThis.self;
    const replies = [];
    globalThis.self = {
      postMessage(message) {
        replies.push(message);
        worker.onmessage({ data: structuredClone(message) });
      },
    };
    try {
      if (startupFailure) {
        startupFailure(worker);
      } else {
        installBridgeWorkerHost();
        worker.postMessage = (message) => self.onmessage({ data: structuredClone(message) });
        await worker.postMessage(worker.posted[0]);
      }
      return await run(bridge, replies);
    } finally {
      await bridge.dispose();
      if (previousSelf === undefined) delete globalThis.self;
      else globalThis.self = previousSelf;
    }
  });
}

function assertResult(result, {
  variant = 'wasm32', threading = 'single-thread', mode = 'direct',
  loaderUrl = loader, wasmUrl = wasm, fallbackReason = null,
} = {}) {
  assert.deepEqual(result, {
    variant, threading, execution: mode === 'direct' ? 'main-thread' : 'worker',
    fallback: fallbackReason !== null, fallbackReason, loaderUrl, wasmUrl,
  });
}

function assetFailure({ asset, url, httpStatus, code = 'CORE_ASSET_UNAVAILABLE', variant = 'wasm32' }) {
  return (error) => {
    assert.ok(error instanceof api.CoreAssetError, String(error));
    assert.equal(error.name, 'CoreAssetError');
    assert.equal(error.code, code);
    assert.equal(error.asset, asset);
    assert.equal(error.url, url);
    assert.equal(error.httpStatus, httpStatus);
    assert.equal(error.variant, variant);
    assert.doesNotMatch(error.message, /user:pass|token=|secret|#frag/);
    return true;
  };
}

function mockDiagnostic(t, { status = 404, type = 'text/plain', networkError = false } = {}) {
  const urls = [];
  let cancelled = 0;
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    urls.push(url);
    assert.equal(options.method, 'GET');
    assert.equal(options.credentials, 'same-origin');
    assert.ok(options.signal instanceof AbortSignal);
    if (networkError) throw new TypeError('network error with token=secret');
    const response = new Response(new ReadableStream({ cancel() { cancelled += 1; } }), {
      status, headers: { 'Content-Type': type },
    });
    return response;
  });
  return { urls, get cancelled() { return cancelled; } };
}

test('the generated bundle exposes CoreAssetError with the bridge global', async (t) => {
  setGlobal(t, 'window', {});
  const bundle = await import('../../js/llama_webgpu_bridge.js');
  assert.equal(window.LlamaWebGpuBridge, bundle.LlamaWebGpuBridge);
  assert.equal(bundle.LlamaWebGpuBridge.supportsCorePreflight, true);
  assert.equal(typeof bundle.CoreAssetError, 'function');
  assert.equal(window.CoreAssetError, bundle.CoreAssetError);
  assert.equal(new bundle.CoreAssetError('CORE_PREFLIGHT_UNSUPPORTED', 'worker', { asset: 'worker' }).name, 'CoreAssetError');
});

for (const mode of ['direct', 'worker']) {
  test(`${mode}: single-thread preflight reuses the core and downloads Wasm once`, async (t) => {
    setGlobal(t, 'crossOriginIsolated', false);
    setGlobal(t, 'MessageChannel', class {
      port1 = { postMessage() { throw new DOMException('not isolated', 'DataCloneError'); }, close() {} };
      port2 = { close() {} };
    });
    const trace = [];
    const imports = withImports(t, { [loader]: wasmFactory({ fetchBinary: true, trace }) });
    const requests = [];
    t.mock.method(globalThis, 'fetch', async (url, options) => {
      assert.ok(url === wasm || url === model);
      if (url === wasm) assert.equal(options, undefined, 'no diagnostic fetch on success');
      requests.push(url);
      return new Response(Uint8Array.of(1, 2, 3));
    });
    await withBridge(mode, config32, async (bridge, replies) => {
      if (mode === 'worker') assert.equal(replies[0].supportsCorePreflight, true);
      const result = await bridge.preflightCore();
      assertResult(result, { mode });
      assert.equal(bridge.getModelMetadata()['llamadart.webgpu.core_variant'], 'wasm32');
      result.wasmUrl = 'changed';
      assertResult(await bridge.preflightCore(), { mode });
      assert.deepEqual(trace, ['factory']);
      await bridge.loadModelFromUrl(model, loadOptions);
      assertResult(await bridge.preflightCore(), { mode });
      assert.deepEqual(trace, ['factory', 'probe']);
      assert.deepEqual(requests, [wasm, model]);
      assert.deepEqual(imports, [loader]);
    });
  });

  test(`${mode}: preflight preserves the load isolation error without the single-thread opt-in`, async (t) => {
    setGlobal(t, 'crossOriginIsolated', false);
    setGlobal(t, 'MessageChannel', class {
      port1 = { postMessage() { throw new DOMException('not isolated', 'DataCloneError'); }, close() {} };
      port2 = { close() {} };
    });
    await withBridge(mode, { ...config32, allowSingleThreadCore: false }, async (bridge) => {
      const isIsolationError = (error) => error instanceof api.CrossOriginIsolationError;
      const ensureCore = t.mock.method(LlamaWebGpuBridgeRuntime.prototype, '_ensureCore');
      await assert.rejects(bridge.preflightCore(), isIsolationError);
      await assert.rejects(bridge.loadModelFromUrl(model, loadOptions), isIsolationError);
      assert.equal(ensureCore.mock.callCount(), 2);
    });
  });

  for (const assetBaseUrl of [base, './bridge/']) {
    test(`${mode}: assetBaseUrl ${assetBaseUrl} resolves the default single-thread pair`, async (t) => {
      setGlobal(t, 'document', { baseURI: 'https://assets.test/' });
      withImports(t, { [loader]: wasmFactory() });
      await withBridge(mode, { assetBaseUrl, allowSingleThreadCore: true, preferMemory64: false }, async (bridge) => {
        assertResult(await bridge.preflightCore(), { mode });
      });
    });
  }

  for (const reason of ['unsupported', 'init_error', 'timeout']) {
    test(`${mode}: ${reason} memory64 fallback never probes the unselected binary and survives model load`, async (t) => {
      t.mock.method(WebAssembly, 'validate', () => reason !== 'unsupported');
      t.mock.method(LlamaWebGpuBridgeRuntime.prototype, '_resolveCoreInitTimeoutMs', () => 20);
      const imports = withImports(t, {
        [loader64]: ({ locateFile }) => {
          locateFile('llama_webgpu_core_mem64_st.wasm', base);
          if (reason === 'timeout') return new Promise(() => {});
          throw new WebAssembly.CompileError('memory64 could not initialize');
        },
        [loader]: wasmFactory(),
      });
      t.mock.method(globalThis, 'fetch', async (url) => {
        assert.equal(url, model, 'only the model is fetched');
        return new Response(Uint8Array.of(1, 2, 3));
      });
      await withBridge(mode, config64, async (bridge) => {
        assertResult(await bridge.preflightCore(), { mode, fallbackReason: reason });
        await bridge.loadModelFromUrl(model, loadOptions);
        assert.match(bridge.getModelMetadata()['llamadart.webgpu.runtime_notes'], new RegExp(`core_mem64_unavailable:${reason}`));
        assert.deepEqual(imports, [loader64, loader]);
      });
    });
  }

  for (const options of [
    { status: 404 }, { status: 410 }, { status: 503 }, { status: 200, type: 'text/html' }, { networkError: true },
  ]) {
    test(`${mode}: loader ${JSON.stringify(options)} preserves typed error and redacts URL`, async (t) => {
      withImports(t, { [signedLoader]: new TypeError('Import failed at token=secret') });
      const requests = mockDiagnostic(t, options);
      await withBridge(mode, { ...config32, coreModuleUrl: signedLoader }, async (bridge, replies) => {
        const proxy = bridge._workerProxy;
        const expected = assetFailure({
          asset: 'loader', url: loader, httpStatus: options.networkError ? null : options.status,
        });
        await assert.rejects(bridge.preflightCore(), expected);
        assert.equal(bridge._workerProxy, proxy);
        assert.deepEqual(requests.urls, [signedLoader]);
        assert.equal(requests.cancelled, options.networkError ? 0 : 1);
        await assert.rejects(bridge.loadModelFromUrl(model, loadOptions), expected);
        assert.equal(bridge._workerProxy, proxy);
        assert.deepEqual(requests.urls, [signedLoader, signedLoader]);
        if (mode === 'worker') {
          const response = replies.find((message) => message.type === 'error');
          assert.doesNotMatch(JSON.stringify(response), /user:pass|token=|secret|#frag/);
        }
      });
    });
  }
}

for (const derived of [false, true]) {
  for (const options of [{ status: 404 }, { status: 503 }, { networkError: true }, { status: 200, type: 'application/wasm' }]) {
    test(`direct: ${derived ? 'asset-base' : 'explicit'} Wasm ${JSON.stringify(options)} reports only asset failures`, async (t) => {
      const original = new WebAssembly.CompileError('invalid Wasm bytes');
      withImports(t, { [loader]: ({ locateFile }) => {
        assert.equal(locateFile('llama_webgpu_core_st.wasm', 'https://wrong.test/'), wasm);
        throw original;
      } });
      const requests = mockDiagnostic(t, options);
      const config = derived ? { assetBaseUrl: base } : { ...config32, wasmUrl: wasm };
      await withBridge('direct', { ...config32, ...config }, async (bridge) => {
        const expected = options.status === 200 ? (error) => error === original : assetFailure({
          asset: 'wasm', url: wasm, httpStatus: options.networkError ? null : options.status,
        });
        await assert.rejects(bridge.preflightCore(), expected);
        assert.deepEqual(requests.urls, [wasm]);
        assert.equal(requests.cancelled, options.networkError ? 0 : 1);
      });
    });
  }
}

test('adjacent Wasm is diagnosed and a later successful preflight discovers its URL', async (t) => {
  let attempts = 0;
  withImports(t, { [signedLoader]: (options) => {
    const url = options.locateFile('llama_webgpu_core_st.wasm', base);
    assert.equal(url, wasm);
    if (++attempts === 1) throw new WebAssembly.CompileError('adjacent binary missing');
    return makeCore();
  } });
  const requests = mockDiagnostic(t);
  await withBridge('direct', { ...config32, coreModuleUrl: signedLoader }, async (bridge) => {
    await assert.rejects(bridge.preflightCore(), assetFailure({ asset: 'wasm', url: wasm, httpStatus: 404 }));
    assertResult(await bridge.preflightCore(), { loaderUrl: signedLoader });
    assert.deepEqual(requests.urls, [wasm]);
  });
});

test('loader errors that cannot be classified are returned unchanged', async (t) => {
  const syntax = new SyntaxError('invalid loader');
  const reference = new ReferenceError('invalid loader reference');
  const imports = {
    [loader]: syntax,
    [`${loader}?reference`]: reference,
    [`${loader}?bad-export`]: 'export const notAFactory = 1;',
    'unmapped-core': new TypeError('bare module resolution failed'),
  };
  withImports(t, imports);
  const requests = mockDiagnostic(t);
  for (const [coreModuleUrl, error] of Object.entries(imports)) {
    await withBridge('direct', { ...config32, coreModuleUrl }, async (bridge) => {
      await assert.rejects(bridge.preflightCore(), typeof error === 'string'
        ? /Unable to resolve llama_webgpu_core factory function/ : (rejection) => rejection === error);
    });
  }
  assert.deepEqual(requests.urls, []);
});

test('Wasm initialization timeout and pthread/isolation errors bypass diagnostics', async (t) => {
  const blocked = new api.BlobWorkerBlockedError('csp');
  const isolation = new api.CrossOriginIsolationError();
  t.mock.method(LlamaWebGpuBridgeRuntime.prototype, '_resolveCoreInitTimeoutMs', () => 20);
  let failure;
  withImports(t, { [loader]: ({ locateFile }) => {
    locateFile('llama_webgpu_core_st.wasm', base);
    if (failure) throw failure;
    return new Promise(() => {});
  } });
  const requests = mockDiagnostic(t);
  await withBridge('direct', config32, async (bridge) => {
    for (failure of [blocked, isolation]) {
      await assert.rejects(bridge.preflightCore(), (error) => error === failure);
    }
    failure = null;
    await assert.rejects(bridge.preflightCore(), /Bridge core init timeout/);
  });
  assert.deepEqual(requests.urls, []);
});

for (const preferMemory64 of [true, false]) {
  test(`unselected core is neither imported nor probed (preferMemory64: ${preferMemory64})`, async (t) => {
    const imports = withImports(t, {
      [loader]: preferMemory64 ? wasmFactory({ fetchBinary: true }) : wasmFactory(),
      [loader64]: preferMemory64 ? wasmFactory({ file: wasm64 }) : wasmFactory({ file: wasm64, fetchBinary: true }),
    });
    const requests = mockDiagnostic(t);
    await withBridge('direct', { ...config64, preferMemory64 }, async (bridge) => {
      assertResult(await bridge.preflightCore(), preferMemory64
        ? { variant: 'memory64', loaderUrl: loader64, wasmUrl: wasm64 }
        : {});
      assert.deepEqual(imports, [preferMemory64 ? loader64 : loader]);
      assert.deepEqual(requests.urls, []);
    });
  });
}

test('a custom factory exposes no loader, preserves its options and never constructs a worker', async () => {
  assertApi();
  await withStubWorkerEnvironment(async ({ workers }) => {
    const bridge = createRealWorkerBridge({
      wasmUrl: wasm,
      coreModuleFactory(options) {
        assert.deepEqual(Reflect.ownKeys(options), ['locateFile', 'print', 'printErr', 'onAbort']);
        return makeCore({ threaded: true });
      },
    });
    try {
      assertResult(await bridge.preflightCore(), { loaderUrl: null, threading: 'threaded' });
      assert.deepEqual(workers, []);
    } finally {
      await bridge.dispose();
    }
    await assert.rejects(bridge.preflightCore(), /Bridge has been disposed\./);
  });
});

test('legacy workers reject before dispatch and keep the worker proxy', async () => {
  assertApi();
  await withStubWorkerEnvironment(async ({ workers }) => {
    const bridge = createRealWorkerBridge();
    const proxy = bridge._workerProxy;
    const driver = workerDriver(proxy, workers[0]);
    driver.ready();
    workers[0].postMessage = (message) => {
      workers[0].posted.push(message);
      if (message.method === 'preflightCore') driver.error(message.id, 'Unknown method preflightCore');
    };
    try {
      await assert.rejects(bridge.preflightCore(), assetFailure({
        code: 'CORE_PREFLIGHT_UNSUPPORTED', asset: 'worker', url: null, httpStatus: null, variant: null,
      }));
      assert.deepEqual(driver.calls(), []);
      assert.equal(bridge._workerProxy, proxy);
      assert.equal(bridge._runtime, null);
    } finally {
      const disposal = bridge.dispose();
      await driver.settle();
      for (const call of driver.calls('dispose')) driver.reply(call.id, undefined);
      await disposal;
    }
  });
});

for (const startup of ['crash', 'init timeout']) {
  test(`worker ${startup} falls back to direct preflight`, async (t) => {
    if (startup === 'init timeout') t.mock.timers.enable({ apis: ['setTimeout'] });
    withImports(t, { [loader]: wasmFactory() });
    await withBridge('worker', config32, async (bridge) => {
      assertResult(await bridge.preflightCore());
    }, {
      startupFailure(worker) {
        if (startup === 'crash') worker.onerror({ message: 'Uncaught exception in the worker entry' });
        else {
          // Fire the ready timeout without waiting for the default deadline.
          t.mock.timers.tick(20_000);
        }
      },
    });
  });
}

test('direct retirement discards the cached preflight but retains the unloaded-model guard', async (t) => {
  const trace = [];
  withImports(t, { [loader]: wasmFactory({ trace }) });
  await withBridge('direct', config32, async (bridge) => {
    assertResult(await bridge.preflightCore());
    const retired = bridge._runtime;
    retireCore(retired._core, new Error('core aborted'));
    assertResult(await bridge.preflightCore());
    assert.notEqual(bridge._runtime, retired);
    assert.deepEqual(trace, ['factory', 'factory']);
    await assert.rejects(bridge.tokenize('still unloaded'), api.ModelUnloadedError);
  });
});

test('preflight replaces a retired worker without restoring its previous model', async () => {
  assertApi();
  await withStubWorkerEnvironment(async ({ workers }) => {
    const bridge = createRealWorkerBridge(config32);
    const original = bridge._workerProxy;
    const originalDriver = workerDriver(original, workers[0]);
    originalDriver.ready({ supportsCorePreflight: true });
    bridge._rememberLoadedModel(model, loadOptions);
    original._worker.onmessage({ data: { type: 'core-retired' } });
    const pending = bridge.preflightCore();
    await originalDriver.settle();
    for (const call of originalDriver.calls('dispose')) originalDriver.reply(call.id, undefined);
    await originalDriver.settle();
    const replacement = bridge._workerProxy;
    assert.notEqual(replacement, original);
    const driver = workerDriver(replacement, workers[1]);
    driver.ready({ supportsCorePreflight: true });
    await driver.settle();
    assert.deepEqual(driver.calls().map((call) => call.method), ['preflightCore']);
    const result = {
      variant: 'wasm32', threading: 'single-thread', execution: 'main-thread',
      fallback: false, fallbackReason: null, loaderUrl: loader, wasmUrl: wasm,
    };
    driver.reply(driver.calls()[0].id, result);
    assertResult(await pending, { mode: 'worker' });
    const disposal = bridge.dispose();
    await driver.settle();
    for (const call of originalDriver.calls('dispose')) originalDriver.reply(call.id, undefined);
    for (const call of driver.calls('dispose')) driver.reply(call.id, undefined);
    await disposal;
  });
});

test('a slow diagnostic returns its worker CoreAssetError after both candidate budgets', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let startMemory64, startWasm32, startDiagnostic;
  const memory64Started = new Promise((resolve) => { startMemory64 = resolve; });
  const wasm32Started = new Promise((resolve) => { startWasm32 = resolve; });
  const diagnosticStarted = new Promise((resolve) => { startDiagnostic = resolve; });
  withImports(t, {
    [loader64]: () => {
      startMemory64();
      return new Promise(() => {});
    },
    [loader]: ({ locateFile }) => {
      locateFile('llama_webgpu_core_st.wasm', base);
      startWasm32();
      return new Promise((_, reject) => setTimeout(() => reject(new Error('Wasm missing')), 89_000));
    },
  });
  t.mock.method(globalThis, 'fetch', (url, { signal }) => {
    assert.equal(url, wasm);
    startDiagnostic();
    return new Promise((_, reject) => signal.addEventListener('abort', reject, { once: true }));
  });
  await withBridge('worker', config64, async (bridge) => {
    const proxy = bridge._workerProxy;
    assert.equal(proxy._resolveRequestTimeoutMs('preflightCore'), 220_000);
    const expected = assert.rejects(bridge.preflightCore(), assetFailure({ asset: 'wasm', url: wasm, httpStatus: null }));
    await memory64Started;
    t.mock.timers.tick(90_000);
    await wasm32Started;
    t.mock.timers.tick(89_000);
    await diagnosticStarted;
    await setImmediate();
    t.mock.timers.tick(10_000);
    await expected;
    assert.equal(bridge._workerProxy, proxy);
  });
});

test('an explicit global worker deadline wins unchanged for preflight', async () => {
  await withStubWorkerEnvironment(async ({ workers }) => {
    const bridge = createRealWorkerBridge({ ...config64, workerRequestTimeoutMs: 1_234 });
    const driver = workerDriver(bridge._workerProxy, workers[0]);
    driver.ready();
    try {
      assert.equal(bridge._workerProxy._resolveRequestTimeoutMs('preflightCore'), 1_234);
    } finally {
      const disposal = bridge.dispose();
      await driver.settle();
      for (const call of driver.calls('dispose')) driver.reply(call.id, undefined);
      await disposal;
    }
  });
});

test('worker CoreAssetError bypasses fallback and filesystem recovery even with matching text', async (t) => {
  assertApi();
  const failure = new api.CoreAssetError('CORE_ASSET_UNAVAILABLE', 'Bridge worker timed out: asset not found', {
    asset: 'wasm', url: wasm, httpStatus: 404, variant: 'wasm32',
  });
  withImports(t, { [loader]: () => { throw failure; } });
  await withBridge('worker', config32, async (bridge) => {
    const proxy = bridge._workerProxy;
    const expected = assetFailure({ asset: 'wasm', url: wasm, httpStatus: 404 });
    await assert.rejects(bridge.preflightCore(), (error) => {
      assert.ok(expected(error));
      assert.equal(bridge._shouldFallbackToMainThread(error), false);
      assert.equal(bridge._isRecoverableWorkerFsError(error), false);
      return true;
    });
    assert.equal(bridge._workerProxy, proxy);
  });
});
