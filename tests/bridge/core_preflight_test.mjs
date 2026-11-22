import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
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
const signedWasm = `${wasm.replace('https://', 'https://user:pass@')}?token=wasm(secret)#wasm-frag`;
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

for (const mode of ['direct', 'worker']) {
  for (const stage of ['loader', 'wasm']) {
    test(`${mode}: signed ${stage} URLs are redacted in core init logs and passthrough errors`, async (t) => {
      const logs = [];
      const log = (line) => logs.push(line);
      for (const level of ['log', 'warn', 'error']) t.mock.method(console, level, log);
      const expected = stage === 'loader' ? `Import failed (${loader}).` : `Compile failed (${wasm}).`;
      const failure = stage === 'loader'
        ? new TypeError(`Import failed (${signedLoader}).`)
        : new WebAssembly.CompileError(`Compile failed (${signedWasm.split('#')[0]}).`);
      withImports(t, {
        [signedLoader]: stage === 'loader' ? failure : ({ locateFile, print, printErr, onAbort }) => {
          const binary = locateFile('llama_webgpu_core_st.wasm', base);
          print(`initializing '${signedLoader}'`);
          printErr(`wasm streaming compile failed for (${binary})`);
          printErr('redirect failed at “https://mirror.test/core.wasm?token=secret#mirror-frag”');
          onAbort(`wasm startup aborted at ${binary}`);
          throw failure;
        },
      });
      const requests = mockDiagnostic(t, { status: 200, type: stage === 'loader' ? 'text/javascript' : 'application/wasm' });
      await withBridge(mode, { ...config32, coreModuleUrl: signedLoader, wasmUrl: signedWasm }, async (bridge, replies) => {
        for (const invoke of [() => bridge.preflightCore(), () => bridge.loadModelFromUrl(model, loadOptions)]) {
          await assert.rejects(invoke, (error) => {
            if (mode === 'direct') assert.ok(error instanceof failure.constructor);
            assert.equal(error.message, expected);
            assert.doesNotMatch(error.stack, /user:pass|token=|secret|#.*frag/);
            return true;
          });
        }
        const wireErrors = replies.filter((message) => message.type === 'error');
        if (mode === 'worker') assert.deepEqual(wireErrors.map((message) => message.message), [expected, expected]);
        assert.doesNotMatch(JSON.stringify(wireErrors), /user:pass|token=|secret|#.*frag/);
        assert.ok(requests.urls.every((url) => url === (stage === 'loader' ? signedLoader : signedWasm)));
      });
      assert.doesNotMatch(logs.join('\n'), /user:pass|token=|secret|#.*frag/);
      if (stage === 'wasm') {
        assert.ok(logs.includes(`initializing '${loader}'`));
        assert.ok(logs.includes(`wasm streaming compile failed for (${wasm})`));
        assert.ok(logs.includes('redirect failed at “https://mirror.test/core.wasm”'));
        assert.ok(logs.includes(`core abort: wasm startup aborted at ${wasm}`));
      }
    });
  }
}

for (const exportName of ['llamadart_webgpu_supports_pthreads', 'llamadart_webgpu_set_log_level']) {
  test(`direct preflight rejects retirement during ${exportName} before caching`, async (t) => {
    let attempts = 0;
    const trap = new WebAssembly.RuntimeError('Core initialization trapped');
    const imports = withImports(t, {
      [loader64]: (options) => {
        options.locateFile('llama_webgpu_core_mem64_st.wasm', base);
        const core = makeCore();
        if (++attempts === 1) {
          const ccall = core.ccall;
          core.ccall = (name, ...args) => {
            if (name === exportName) throw trap;
            return ccall(name, ...args);
          };
        }
        return core;
      },
      [loader]: wasmFactory(),
    });
    await withBridge('direct', config64, async (bridge) => {
      const retired = bridge._runtime;
      await assert.rejects(bridge.preflightCore(), (error) => error === trap || error instanceof api.ModelUnloadedError);
      assert.equal(retired._corePreflight, null);
      assert.notEqual(bridge._runtime, retired);
      assert.equal(bridge._runtime._core, null);
      assert.deepEqual(imports, [loader64]);
      assertResult(await bridge.preflightCore(), { variant: 'memory64', loaderUrl: loader64, wasmUrl: wasm64 });
      assert.equal(attempts, 2);
    });
  });
}

test('core init redaction preserves DOMException identity fields', async (t) => {
  withImports(t, { [signedLoader]: new DOMException(`Import failed (${signedLoader})`, 'NetworkError') });
  mockDiagnostic(t, { status: 200, type: 'text/javascript' });
  await withBridge('direct', { ...config32, coreModuleUrl: signedLoader }, async (bridge) => {
    await assert.rejects(bridge.preflightCore(), (error) => {
      assert.ok(error instanceof DOMException);
      assert.equal(error.name, 'NetworkError');
      assert.equal(error.code, DOMException.NETWORK_ERR);
      assert.equal(error.message, `Import failed (${loader})`);
      return true;
    });
  });
});

for (const source of ['loader', 'custom factory']) {
  test(`direct ${source} redacts an error cause even when the outer error has no URL`, async (t) => {
    const failure = new TypeError('x', { cause: new Error('fetch https://u:p@h/a.wasm?sig=1 failed') });
    let config;
    if (source === 'loader') {
      withImports(t, { [loader]: failure });
      mockDiagnostic(t, { status: 200, type: 'text/javascript' });
      config = config32;
    } else {
      config = { coreModuleFactory() { throw failure; } };
    }
    await withBridge('direct', config, async (bridge) => {
      await assert.rejects(bridge.preflightCore(), (error) => {
        assert.ok(error instanceof TypeError);
        assert.equal(error.message, 'x');
        assert.notEqual(error.cause, failure.cause);
        assert.ok(error.cause instanceof Error);
        assert.equal(error.cause.message, 'fetch https://h/a.wasm failed');
        assert.doesNotMatch(error.cause.stack, /u:p|sig=1/);
        assert.match(failure.cause.message, /sig=1/);
        return true;
      });
    });
  });
}

test('direct preflight redacts nested and string causes and bounds cyclic and deep causes', async () => {
  const nested = new Error('outer', { cause: new TypeError('middle', {
    cause: new WebAssembly.CompileError('inner', { cause: 'fetch https://u:p@h/a.wasm?sig=1 failed' }),
  }) });
  const cyclic = new TypeError('cycle');
  cyclic.cause = new Error('fetch https://u:p@h/a.wasm?sig=1 failed', { cause: cyclic });
  let deep = new Error('fetch https://u:p@h/a.wasm?sig=1 failed');
  for (let index = 0; index < 10; index++) deep = new Error(`level ${index}`, { cause: deep });
  for (const failure of [nested, cyclic, deep]) {
    await withBridge('direct', { coreModuleFactory() { throw failure; } }, async (bridge) => {
      await assert.rejects(bridge.preflightCore(), (error) => {
        const visited = new Set();
        let current = error;
        while (current instanceof Error && !visited.has(current)) {
          visited.add(current);
          assert.doesNotMatch(`${current.message}\n${current.stack}`, /u:p|sig=1/);
          current = current.cause;
        }
        assert.ok(visited.size <= 6);
        if (failure === nested) {
          assert.ok(error.cause instanceof TypeError);
          assert.ok(error.cause.cause instanceof WebAssembly.CompileError);
          assert.equal(current, 'fetch https://h/a.wasm failed');
        }
        if (failure === cyclic) {
          assert.equal(error.cause.message, 'fetch https://h/a.wasm failed');
        }
        return true;
      });
    });
  }
});

for (const mode of ['direct', 'worker']) {
  test(`${mode}: unsigned relative core URLs preserve initialization log and stack text`, async (t) => {
    setGlobal(t, 'document', { baseURI: 'http://127.0.0.1:52208/' });
    const logLine = 'wasm streaming compile failed: CompileError: WebAssembly.instantiateStreaming(): expected magic word';
    const message = 'Aborted(CompileError: WebAssembly.instantiate(): expected magic word)';
    const stack = `RuntimeError: ${message}\n    at abort (http://127.0.0.1:52208/loader.js:1:2274)`;
    const lines = [];
    for (const level of ['log', 'warn', 'error']) t.mock.method(console, level, (line) => lines.push(line));
    withImports(t, { '/loader.js': ({ locateFile, print, printErr }) => {
      assert.equal(locateFile('lib.wasm', base), 'wasm');
      printErr(logLine);
      print(stack);
      const error = new WebAssembly.RuntimeError(message);
      error.stack = stack;
      throw error;
    } });
    mockDiagnostic(t, { status: 200, type: 'application/wasm' });
    await withBridge(mode, { ...config32, coreModuleUrl: '/loader.js', wasmUrl: 'wasm' }, async (bridge) => {
      await assert.rejects(bridge.loadModelFromUrl(model, loadOptions), (error) => {
        assert.equal(error.message, message);
        if (mode === 'direct') assert.equal(error.stack, stack);
        return true;
      });
    });
    assert.deepEqual(lines, [logLine, stack, logLine, stack]);
  });

  test(`${mode}: a signed relative loader URL is redacted without changing other text`, async (t) => {
    const coreModuleUrl = 'loader.js?sig=1';
    withImports(t, { [coreModuleUrl]: new TypeError(`Import of ${coreModuleUrl} failed`) });
    await withBridge(mode, { ...config32, coreModuleUrl }, async (bridge) => {
      await assert.rejects(bridge.preflightCore(), { message: 'Import of loader.js failed' });
    });
  });
}

test('redirected core URLs lose their full query and keep stack positions during preflight and load', async (t) => {
  const requested = 'http://127.0.0.1:52208/loader.js?v=42';
  const redirected = `${requested}&fixtureLabel=BLUE_TEST_VALUE`;
  const location = 'http://127.0.0.1:52208/loader.js';
  const originalMessage = `Wasm initialization failed at ${redirected}`;
  const stack = `CompileError: ${originalMessage}\n    at abort (${redirected}:10:5)\n    at start (${redirected}:11)`;
  const expectedMessage = `Wasm initialization failed at ${location}`;
  const expectedStack = `CompileError: ${expectedMessage}\n    at abort (${location}:10:5)\n    at start (${location}:11)`;
  withImports(t, { [requested]: ({ locateFile }) => {
    locateFile('lib.wasm', base);
    const error = new WebAssembly.CompileError(originalMessage);
    error.stack = stack;
    throw error;
  } });
  mockDiagnostic(t, { status: 200, type: 'application/wasm' });
  await withBridge('direct', { ...config32, coreModuleUrl: requested }, async (bridge) => {
    for (const invoke of [() => bridge.preflightCore(), () => bridge.loadModelFromUrl(model, loadOptions)]) {
      await assert.rejects(invoke, (error) => {
        assert.ok(error instanceof WebAssembly.CompileError);
        assert.equal(error.message, expectedMessage);
        assert.equal(error.stack, expectedStack);
        return true;
      });
    }
  });
});

for (const mode of ['direct', 'worker']) {
  test(`${mode}: a signed root-relative URL never rewrites inside an absolute stack frame`, async (t) => {
    const coreModuleUrl = '/loader.js?v=42';
    const origin = 'http://127.0.0.1:52208';
    const absolute = `${origin}${coreModuleUrl}`;
    const baseFrame = `    at abort (${origin}/loader.js:1:2274)`;
    const notes = [];
    t.mock.method(console, 'log', (line) => notes.push(line));
    withImports(t, { [coreModuleUrl]: ({ locateFile, print }) => {
      locateFile('lib.wasm', base);
      const error = new WebAssembly.CompileError(`Initialization failed for ${coreModuleUrl}`);
      error.stack = `CompileError: ${error.message}\n    at abort (${absolute}:1:2274)`;
      print(error.stack);
      throw error;
    } });
    mockDiagnostic(t, { status: 200, type: 'application/wasm' });
    await withBridge(mode, { ...config32, coreModuleUrl }, async (bridge) => {
      await assert.rejects(bridge.preflightCore(), (error) => {
        assert.equal(error.message, 'Initialization failed for file://[redacted]');
        if (mode === 'direct') assert.equal(error.stack.split('\n')[1], baseFrame);
        return true;
      });
    });
    assert.equal(notes.length, 1);
    assert.equal(notes[0].split('\n')[1], baseFrame);
  });
}

test('mirror error redaction cuts aggregate cycles and preserves shared children', async () => {
  const child = new TypeError('Failed to import https://cdn.test/mirror.js?sig=1');
  const failure = new AggregateError([child, child], 'All mirrors failed');
  failure.errors.push(failure);
  await withBridge('direct', { coreModuleFactory() { throw failure; } }, async (bridge) => {
    await assert.rejects(bridge.preflightCore(), (error) => {
      assert.ok(error instanceof AggregateError);
      assert.equal(error.errors.length, 3);
      for (const mirrored of error.errors.slice(0, 2)) {
        assert.ok(mirrored instanceof TypeError);
        assert.equal(mirrored.message, 'Failed to import https://cdn.test/mirror.js');
      }
      assert.equal(error.errors[2], undefined);
      return true;
    });
  });
});

for (const wrapped of [false, true]) {
  test(`mirror loader redacts Promise.any errors${wrapped ? ' under cause' : ''}`, async (t) => {
    const mirrors = ['https://a%40b:pw@cdn.test/first.js?v=BLUE_TEST_VALUE', 'https://cdn.test/second.js#GREEN_TEST_VALUE'];
    const paths = ['https://cdn.test/first.js', 'https://cdn.test/second.js'];
    const rejections = mirrors.map((url) => {
      const error = new TypeError(`Failed to import ${url}`);
      error.stack = `TypeError: ${error.message}\n    at mirror (${url}:8:2)`;
      return error;
    });
    withImports(t, {
      ...Object.fromEntries(mirrors.map((url, index) => [url, rejections[index]])),
      [loader]: async () => {
        try {
          return await Promise.any(mirrors.map((url) => import(url)));
        } catch (error) {
          if (wrapped) throw new Error('The mirror loader failed', { cause: error });
          throw error;
        }
      },
    });
    await withBridge('direct', config32, async (bridge) => {
      await assert.rejects(bridge.preflightCore(), (error) => {
        const aggregate = wrapped ? error.cause : error;
        assert.ok(aggregate instanceof AggregateError);
        assert.equal(aggregate.name, 'AggregateError');
        assert.equal(aggregate.message, 'All promises were rejected');
        assert.equal(aggregate.errors.length, 2);
        for (const [index, child] of aggregate.errors.entries()) {
          assert.ok(child instanceof TypeError);
          assert.equal(child.message, `Failed to import ${paths[index]}`);
          assert.equal(child.stack, `TypeError: ${child.message}\n    at mirror (${paths[index]}:8:2)`);
        }
        return true;
      });
    });
  });
}

test('core redaction preserves unsigned URLs and handles mixed URLs in an object cause', async () => {
  const unsigned = 'https://Assets.test/with/@/loader.js';
  const failure = new Error('Import failed', { cause: {
    message: `Opening ${unsigned}, /loader.js and https://a%40b:secret@assets.test/loader.js?sig=1; then ${unsigned}.`,
  } });
  await withBridge('direct', { coreModuleFactory() { throw failure; } }, async (bridge) => {
    await assert.rejects(bridge.preflightCore(), (error) => {
      assert.equal(error.cause.message, `Opening ${unsigned}, /loader.js and https://assets.test/loader.js; then ${unsigned}.`);
      return true;
    });
  });
});

for (const mode of ['direct', 'worker']) {
  const signedMirror = 'https://cdn.example.test/core.wasm?v=42';
  const mirror = 'https://cdn.example.test/core.wasm';
  const backup = 'https://backup.example.test/core.wasm';
  const scrubbedList = [signedMirror, backup].map((candidate) => {
    const { origin, pathname } = new URL(candidate);
    return origin + pathname;
  }).join(',');
  for (const [format, input, expected] of [
    [
      'comma-joined mirror list',
      `Tried [${signedMirror},${backup}]`,
      `Tried [${mirror}]`,
    ],
    ['single bracketed URL', '[https://x/core.wasm?v=42]', '[https://x/core.wasm]'],
    [
      'signed URL followed by an unsigned URL',
      'Tried https://x/core.wasm#preview,https://[::1]/core.wasm',
      'Tried https://x/core.wasm',
    ],
    ['comma-and-space mirror list', `Tried [${signedMirror}, ${backup}]`, `Tried [${mirror}, ${backup}]`],
    ['space-separated mirror list', `Tried ${signedMirror} ${backup}`, `Tried ${mirror} ${backup}`],
    ['JSON mirror list', JSON.stringify([signedMirror, backup]), JSON.stringify([mirror, backup])],
    ['pre-scrubbed mirror list', `Tried [${scrubbedList}]`, `Tried [${mirror},${backup}]`],
    ['path-only comma-joined mirror list', `Tried [${mirror},${backup}]`, `Tried [${mirror},${backup}]`],
  ]) {
    test(`${mode}: core diagnostic punctuation survives redaction for a ${format}`, async (t) => {
      const lines = [];
      for (const level of ['log', 'warn', 'error']) t.mock.method(console, level, (line) => lines.push(line));
      withImports(t, { [loader]: ({ print, printErr }) => {
        print(input);
        printErr(input);
        throw new Error(input);
      } });
      await withBridge(mode, config32, async (bridge) => {
        await assert.rejects(bridge.loadModelFromUrl(model, loadOptions), { message: expected });
      });
      assert.deepEqual(lines, [expected, expected, expected, expected]);
    });
  }
}

for (const mode of ['direct', 'worker']) {
  for (const [format, input, expected] of [
    ['closing bracket', 'https://cdn/x.wasm?a=]&sig=SECRET', 'https://cdn/x.wasm'],
    ['closing parenthesis', 'https://cdn/x.wasm?a=)&sig=SECRET', 'https://cdn/x.wasm'],
    ['comma and scheme', 'https://cdn/x.wasm?next=,https://y/z&sig=SECRET', 'https://cdn/x.wasm'],
    ['percent-decoded comma', decodeURIComponent('https://cdn/x.wasm?next=%2Chttps://decoded/z&sig=SECRET'), 'https://cdn/x.wasm'],
    ['parenthesized frame', '(https://x/a.js?sig=S:10:5)', '(https://x/a.js:10:5)'],
    ['Firefox frame', '@https://x/a.js?sig=S:3:7', '@https://x/a.js:3:7'],
  ]) {
    test(`${mode}: core URL hard terminators preserve ${format} formatting safely`, async (t) => {
      const lines = [];
      for (const level of ['log', 'warn', 'error']) t.mock.method(console, level, (line) => lines.push(line));
      withImports(t, { [loader]: ({ print, printErr }) => {
        print(input);
        printErr(input);
        throw new Error(input);
      } });
      await withBridge(mode, config32, async (bridge) => {
        await assert.rejects(bridge.loadModelFromUrl(model, loadOptions), { message: expected });
      });
      assert.deepEqual(lines, [expected, expected, expected, expected]);
    });
  }
}

for (const mode of ['direct', 'worker']) {
  for (const [format, path] of [
    ['audit filename with a closing parenthesis', '/volume-2).wasm'],
    ['closing square bracket', '/volume-3].wasm'],
    ['comma', '/volume,3).wasm'],
    ['semicolon', '/volume;3].wasm'],
    ['quote-free punctuation', '/volume!~|3).wasm'],
    ['comma and another scheme', '/volume,https://nested.example.test/part].wasm'],
  ]) {
    test(`${mode}: signed asset path keeps ${format} without retaining its query`, async (t) => {
      const location = new URL(path, 'https://assets.example.test').href;
      const input = `Loading ${location}?sig=SECRET`;
      const expected = `Loading ${location}`;
      const lines = [];
      for (const level of ['log', 'warn', 'error']) t.mock.method(console, level, (line) => lines.push(line));
      withImports(t, { [loader]: ({ print, printErr }) => {
        print(input);
        printErr(input);
        throw new Error(input);
      } });
      await withBridge(mode, config32, async (bridge) => {
        await assert.rejects(bridge.loadModelFromUrl(model, loadOptions), { message: expected });
      });
      assert.deepEqual(lines, [expected, expected, expected, expected]);
    });
  }
}

test('stripping path-only userinfo preserves the other URL bytes', async () => {
  const input = 'Loading HTTPS://u:p@ASSETS.example.test:443/volume-2).wasm';
  await withBridge('direct', { coreModuleFactory() { throw new Error(input); } }, async (bridge) => {
    await assert.rejects(bridge.preflightCore(), { message: 'Loading HTTPS://ASSETS.example.test:443/volume-2).wasm' });
  });
});

for (const mode of ['direct', 'worker']) {
  for (const [input, expected = input] of [
    ['https://cdn.example.test,ops@example.test,ready'],
    ['(https://cdn.example.test),owner=ops@example.test'],
    ['https://cdn.example.test)contact=ops@example.test'],
    ['https://user:pass@host/core.wasm', 'https://host/core.wasm'],
    ['https://TOKEN@host/core.wasm', 'https://host/core.wasm'],
  ]) {
    test(`${mode}: strip only plain URL userinfo in ${input}`, async (t) => {
      const lines = [];
      for (const level of ['log', 'warn', 'error']) t.mock.method(console, level, (line) => lines.push(line));
      withImports(t, { [loader]: ({ print, printErr }) => {
        print(input);
        printErr(input);
        throw new Error(input);
      } });
      await withBridge(mode, config32, async (bridge) => {
        await assert.rejects(bridge.loadModelFromUrl(model, loadOptions), { message: expected });
      });
      assert.deepEqual(lines, [expected, expected, expected, expected]);
    });
  }

  for (const asset of ['wasmUrl', 'coreModuleUrl', 'workerUrl']) {
    test(`${mode}: redact a configured ${asset} with an apostrophe in its path`, async (t) => {
      const configured = `HTTPS://Assets.Test/it's/${asset}.js?sig=SECRET`;
      const serialized = new URL(configured).href;
      const expected = `Loading ${serialized.split('?')[0]}`;
      const lines = [];
      for (const level of ['log', 'warn', 'error']) t.mock.method(console, level, (line) => lines.push(line));
      const config = { ...config32, [asset]: configured };
      withImports(t, { [config.coreModuleUrl]: ({ print, printErr }) => {
        const message = `Loading ${serialized}&extra=SECRET`;
        print(message);
        printErr(message);
        throw new Error(message);
      } });
      await withBridge(mode, config, async (bridge) => {
        await assert.rejects(bridge.loadModelFromUrl(model, loadOptions), { message: expected });
      });
      assert.deepEqual(lines, [expected, expected, expected, expected]);
    });
  }

  test(`${mode}: redact an observed apostrophe URL after a diagnostic redirect`, async (t) => {
    const redirectUrl = `https://mirror.test/it's/core.wasm?sig=SECRET`;
    const expected = `Wasm failure at ${redirectUrl.split('?')[0]}`;
    const lines = [];
    for (const level of ['log', 'warn', 'error']) t.mock.method(console, level, (line) => lines.push(line));
    let attempts = 0;
    withImports(t, { [loader]: ({ locateFile, printErr }) => {
      locateFile('core.wasm', base);
      const message = `Wasm failure at ${redirectUrl}`;
      if (++attempts === 2) printErr(message);
      throw new WebAssembly.CompileError(message);
    } });
    const fetch = t.mock.method(globalThis, 'fetch', async () => {
      const response = new Response('', { headers: { 'Content-Type': 'application/wasm' } });
      Object.defineProperty(response, 'url', { value: redirectUrl });
      return response;
    });
    await withBridge(mode, { ...config32, wasmUrl: wasm }, async (bridge) => {
      await assert.rejects(bridge.loadModelFromUrl(model, loadOptions), { message: expected });
    });
    assert.deepEqual(lines, [expected]);
    assert.equal(fetch.mock.callCount(), 1);
  });
}

test('direct preflight rejects a cached core that retires while applying its log level', async (t) => {
  let trap = false;
  withImports(t, { [loader]: (options) => {
    options.locateFile('llama_webgpu_core_st.wasm', base);
    const core = makeCore();
    const ccall = core.ccall;
    core.ccall = (name, ...args) => {
      if (trap && name === 'llamadart_webgpu_set_log_level') throw new WebAssembly.RuntimeError('Core trapped');
      return ccall(name, ...args);
    };
    return core;
  } });
  await withBridge('direct', config32, async (bridge) => {
    const retired = bridge._runtime;
    assertResult(await bridge.preflightCore());
    trap = true;
    await assert.rejects(bridge.preflightCore(), api.ModelUnloadedError);
    assert.equal(retired._corePreflight, null);
  });
});

test('preflight retry documentation describes the browser module map in both public references', () => {
  const docs = readFileSync(new URL('../../docs/api.md', import.meta.url), 'utf8');
  const types = readFileSync(new URL('../../js/src/llama_webgpu_bridge.d.ts', import.meta.url), 'utf8');
  for (const text of [docs, types]) {
    assert.match(text, /browser caches a failed loader import by URL/);
    assert.match(text, /reload the page or use a different loader URL/);
  }
});

test('a corrupt reachable Wasm gets one diagnostic GET per load operation', async (t) => {
  let attempts = 0;
  withImports(t, { [loader]: ({ locateFile }) => {
    attempts += 1;
    locateFile('llama_webgpu_core_st.wasm', base);
    throw new WebAssembly.CompileError('invalid Wasm bytes');
  } });
  const requests = mockDiagnostic(t, { status: 200, type: 'application/wasm' });
  await withBridge('direct', config32, async (bridge) => {
    for (const count of [1, 2]) {
      await assert.rejects(bridge.loadModelFromUrl(model, loadOptions), WebAssembly.CompileError);
      assert.equal(attempts, 2 * count);
      assert.equal(requests.urls.length, count);
      assert.equal(requests.cancelled, count);
    }
  });
});
