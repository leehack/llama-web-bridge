import assert from 'node:assert/strict';
import test from 'node:test';

import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';

const TIMEOUT = 'Bridge core init timeout (10000ms)';
const HINT = TIMEOUT + '. If this core was built with pthreads,';

function onPage(t, { isolated = false, isolationThrows = false, onIsolationRead } = {}) {
  const saved = Object.getOwnPropertyDescriptor(globalThis, 'crossOriginIsolated');
  Object.defineProperty(globalThis, 'crossOriginIsolated', {
    configurable: true,
    get() {
      onIsolationRead?.();
      if (isolationThrows) throw new Error('isolation unavailable');
      return isolated;
    },
  });
  t.after(() => {
    if (saved) Object.defineProperty(globalThis, 'crossOriginIsolated', saved);
    else delete globalThis.crossOriginIsolated;
  });

  const delays = [];
  const originalSetTimeout = globalThis.setTimeout;
  t.mock.method(globalThis, 'setTimeout', (callback, ms, ...args) => {
    delays.push(ms);
    return originalSetTimeout(callback, ms === 10000 ? 40 : ms, ...args);
  });
  let channelProbes = 0;
  t.mock.method(globalThis, 'MessageChannel', class {
    constructor() {
      channelProbes += 1;
      throw new Error('unexpected MessageChannel');
    }
  });
  t.after(() => assert.equal(channelProbes, 0));
  return delays;
}

function directBridge(config) {
  return new LlamaWebGpuBridge({
    disableWorker: true, allowSingleThreadCore: true, coreInitTimeoutMs: 10000, ...config,
  });
}

function coreUrl(body = 'await new Promise(() => {});') {
  return 'data:text/javascript,' + encodeURIComponent(`
    export default async (options) => {
      ${body}
      return { ...options, kind: 'wasm32' };
    };
  `);
}

function captureInitError(bridge) {
  return bridge._runtime._ensureCore().then(() => assert.fail('must time out'), (error) => error);
}

test('a timeout adds conditional guidance without calling the unfinished core', async (t) => {
  const delays = onPage(t);
  let exportCalls = 0;
  const bridge = directBridge({
    coreModuleFactory(options) {
      const module = {
        ...options,
        _llamadart_webgpu_supports_pthreads() {
          exportCalls += 1;
          return 1;
        },
      };
      module.locateFile('core.wasm', '/');
      return new Promise(() => {});
    },
  });
  const error = await captureInitError(bridge);
  assert.deepEqual(delays, [10000]);
  assert.equal(exportCalls, 0);
  assert.equal(error.constructor, Error);
  assert.equal(error.code, undefined);
  assert.ok(error.message.startsWith(HINT), error.message);
  assert.match(error.message, /"_st"/);
  assert.match(error.message, /allowSingleThreadCore/);
  assert.match(error.message, /"Cross-Origin-Opener-Policy: same-origin"/);
  assert.match(error.message, /"Cross-Origin-Embedder-Policy: require-corp"/);
  const plain = new Error(TIMEOUT);
  assert.equal(bridge._shouldFallbackToMainThread(error), bridge._shouldFallbackToMainThread(plain));
  assert.equal(bridge._isRecoverableWorkerFsError(error), bridge._isRecoverableWorkerFsError(plain));
});

test('only the final model-load deadline reads the isolation flag', async (t) => {
  const controller = new AbortController();
  const reads = [];
  let starts = 0;
  const delays = onPage(t, { onIsolationRead() {
    reads.push(starts);
    controller.abort();
  } });
  const bridge = directBridge({ coreModuleFactory() {
    starts += 1;
    return new Promise(() => {});
  } });
  await assert.rejects(
    bridge.loadModelFromUrl('https://models.test/m.gguf', { signal: controller.signal }),
    (error) => error.constructor === Error && error.message.startsWith(HINT),
  );
  assert.deepEqual(delays, [10000, 10000]);
  assert.equal(starts, 2);
  assert.deepEqual(reads, [2]);
});

test('a throwing inherited loader property cannot alter the timeout', async (t) => {
  onPage(t);
  const saved = Object.getOwnPropertyDescriptor(Object.prototype, 'moduleUrl');
  let accesses = 0;
  const bridge = directBridge({ coreModuleFactory: () => new Promise(() => {}) });
  Object.defineProperty(Object.prototype, 'moduleUrl', {
    configurable: true,
    get() {
      accesses += 1;
      throw new TypeError('unrelated moduleUrl accessor');
    },
  });
  try {
    const error = await captureInitError(bridge);
    assert.equal(error.constructor, Error);
    assert.ok(error.message.startsWith(HINT), error.message);
    assert.equal(accesses, 0);
  } finally {
    if (saved) Object.defineProperty(Object.prototype, 'moduleUrl', saved);
    else delete Object.prototype.moduleUrl;
  }
});

test('only the bridge deadline adds a hint', async (t) => {
  onPage(t);
  const factoryError = directBridge({
    coreModuleUrl: coreUrl(`throw new Error(${JSON.stringify(TIMEOUT)});`),
  });
  assert.equal((await captureInitError(factoryError)).message, TIMEOUT);
  assert.ok((await captureInitError(directBridge({ coreModuleUrl: coreUrl() }))).message.startsWith(HINT));
});

test('only explicitly non-isolated pages get a hint', async (t) => {
  for (const isolated of [true, null]) {
    await t.test(String(isolated), async (page) => {
      onPage(page, { isolated });
      const bridge = directBridge({ coreModuleUrl: coreUrl() });
      assert.equal((await captureInitError(bridge)).message, TIMEOUT);
    });
  }
});

test('a diagnostic exception preserves the timeout and wasm32 fallback', async (t) => {
  let reads = 0;
  onPage(t, { isolationThrows: true, onIsolationRead: () => { reads += 1; } });
  const error = await captureInitError(directBridge({ coreModuleUrl: coreUrl() }));
  assert.equal(error.constructor, Error);
  assert.equal(error.message, TIMEOUT);
  assert.equal(reads, 1);

  const bridge = directBridge({
    preferMemory64: true,
    coreModuleUrlMem64: coreUrl(),
    coreModuleUrl: coreUrl(''),
  });
  assert.equal((await bridge._runtime._ensureCore()).kind, 'wasm32');
  assert.ok(bridge._runtime._runtimeNotes.includes('core_mem64_unavailable:timeout'));
  assert.equal(reads, 1);
});

test('an opaque factory rejection still falls back to wasm32', async (t) => {
  onPage(t);
  const bridge = directBridge({
    preferMemory64: true,
    coreModuleUrlMem64: coreUrl(`
      const { proxy, revoke } = Proxy.revocable({}, {});
      revoke();
      throw proxy;
    `),
    coreModuleUrl: coreUrl(''),
  });
  assert.equal((await bridge._runtime._ensureCore()).kind, 'wasm32');
  assert.ok(bridge._runtime._runtimeNotes.includes('core_mem64_unavailable:init_error'));
});
