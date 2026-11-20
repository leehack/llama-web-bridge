import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { pathToFileURL } from 'node:url';

import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';

const TIMEOUT = 'Bridge core init timeout (10000ms)';

function onPage(t, { isolated = false, closeThrows = false, isolationThrows = false } = {}) {
  const originalIsolation = Object.getOwnPropertyDescriptor(globalThis, 'crossOriginIsolated');
  Object.defineProperty(globalThis, 'crossOriginIsolated', {
    configurable: true,
    get() {
      if (isolationThrows) throw new Error('isolation unavailable');
      return isolated;
    },
  });
  t.after(() => {
    if (originalIsolation) Object.defineProperty(globalThis, 'crossOriginIsolated', originalIsolation);
    else delete globalThis.crossOriginIsolated;
  });

  const originalSetTimeout = globalThis.setTimeout;
  t.mock.method(globalThis, 'setTimeout', (fn, ms, ...args) => originalSetTimeout(fn, ms === 10000 ? 40 : ms, ...args));
  if (!isolated) {
    t.mock.method(globalThis, 'MessageChannel', class {
      port1 = {
        postMessage() { throw new DOMException('refused', 'DataCloneError'); },
        close() {
          if (closeThrows) throw new TypeError('MessagePort close wrapper failed');
        },
      };
      port2 = { close() {} };
    });
  }
}

function directBridge(config) {
  return new LlamaWebGpuBridge({
    disableWorker: true, allowSingleThreadCore: true, coreInitTimeoutMs: 10000, ...config,
  });
}

async function writeCore(t, filename, setup = 'await new Promise(() => {});') {
  const dir = await mkdtemp(path.join(tmpdir(), 'pthread-timeout-'));
  t.after(() => rm(dir, { recursive: true }));
  const file = path.join(dir, filename);
  await writeFile(file, `
    export default async (options) => {
      const core = { ...options, kind: ${JSON.stringify(filename)} };
      core.locateFile('core.wasm', '/');
      ${setup}
      return core;
    };
  `);
  return pathToFileURL(file).href;
}

function captureInitError(bridge) {
  return bridge._runtime._ensureCore().then(() => assert.fail('must time out'), (error) => error);
}

test('an unfinished custom factory times out without calling its pthread export', async (t) => {
  onPage(t);
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
  assert.equal(exportCalls, 0);
  assert.equal(error.constructor, Error);
  assert.equal(error.message, TIMEOUT);
});

test('an auxiliary pthread core cannot label a custom single-thread core timeout', async (t) => {
  onPage(t);
  const bridge = directBridge({
    coreModuleFactory(options) {
      const primary = { ...options, _llamadart_webgpu_supports_pthreads: () => 0 };
      primary.locateFile('core_st.wasm', '/');
      const auxiliary = { ...options, _llamadart_webgpu_supports_pthreads: () => 1 };
      auxiliary.locateFile('core.wasm', '/');
      return new Promise(() => {});
    },
  });
  assert.equal((await captureInitError(bridge)).message, TIMEOUT);
});

test('a throwing MessagePort close cannot interrupt wasm64 timeout fallback', async (t) => {
  onPage(t, { closeThrows: true });
  const bridge = directBridge({
    preferMemory64: true,
    coreModuleUrlMem64: await writeCore(t, 'llama_webgpu_core_mem64.js'),
    coreModuleUrl: await writeCore(t, 'llama_webgpu_core_st.js', ''),
  });
  assert.equal((await bridge._runtime._ensureCore()).kind, 'llama_webgpu_core_st.js');
  assert.ok(bridge._runtime._runtimeNotes.includes('core_mem64_unavailable:timeout'));
});

test('a diagnostic exception preserves the plain timeout', async (t) => {
  onPage(t, { isolationThrows: true });
  const bridge = directBridge({ coreModuleUrl: await writeCore(t, 'llama_webgpu_core.js') });
  const error = await captureInitError(bridge);
  assert.equal(error.constructor, Error);
  assert.equal(error.message, TIMEOUT);
});

test('only recognized threaded loader URLs on non-isolated pages get a timeout hint', async (t) => {
  onPage(t);
  for (const name of ['llama_webgpu_core.js', 'llama_webgpu_core_mem64.js']) {
    const url = await writeCore(t, name);
    const bridge = directBridge({ coreModuleUrl: url + '?version=1#core' });
    const error = await captureInitError(bridge);
    assert.equal(error.constructor, Error);
    assert.equal(error.code, undefined);
    assert.ok(error.message.startsWith(TIMEOUT + '.'), error.message);
    assert.match(error.message, /if (the|this) URL serves a pthread core/i);
    assert.match(error.message, /"_st"/);
    assert.match(error.message, /allowSingleThreadCore/);
    assert.match(error.message, /Cross-Origin-Embedder-Policy: require-corp/);
    const plain = new Error(TIMEOUT);
    assert.equal(bridge._shouldFallbackToMainThread(error), bridge._shouldFallbackToMainThread(plain));
    assert.equal(bridge._isRecoverableWorkerFsError(error), bridge._isRecoverableWorkerFsError(plain));
  }
  for (const name of ['llama_webgpu_core_st.js', 'llama_webgpu_core_mem64_st.js', 'wrapper.js']) {
    const bridge = directBridge({ coreModuleUrl: await writeCore(t, name) });
    assert.equal((await captureInitError(bridge)).message, TIMEOUT, name);
  }
  const inline = 'data:text/javascript,export default () => new Promise(() => {}); // /llama_webgpu_core.js';
  assert.equal((await captureInitError(directBridge({ coreModuleUrl: inline }))).message, TIMEOUT);
});

test('an isolated page keeps the plain timeout for a threaded loader URL', async (t) => {
  onPage(t, { isolated: true });
  const bridge = directBridge({ coreModuleUrl: await writeCore(t, 'llama_webgpu_core.js') });
  assert.equal((await captureInitError(bridge)).message, TIMEOUT);
});

test('only the bridge deadline gets a hint, and a fallback uses its own URL', async (t) => {
  onPage(t);
  const factoryError = directBridge({
    coreModuleUrl: await writeCore(t, 'llama_webgpu_core.js', `throw new Error(${JSON.stringify(TIMEOUT)});`),
  });
  assert.equal((await captureInitError(factoryError)).message, TIMEOUT);

  const fallback = directBridge({
    preferMemory64: true,
    coreModuleUrlMem64: await writeCore(t, 'llama_webgpu_core_mem64.js'),
    coreModuleUrl: await writeCore(t, 'llama_webgpu_core_st.js'),
  });
  assert.equal((await captureInitError(fallback)).message, TIMEOUT);
  assert.ok(fallback._runtime._runtimeNotes.includes('core_mem64_unavailable:timeout'));
});

test('an opaque factory rejection still falls back to wasm32', async (t) => {
  onPage(t);
  const bridge = directBridge({
    preferMemory64: true,
    coreModuleUrlMem64: await writeCore(t, 'llama_webgpu_core_mem64.js', `
      const { proxy, revoke } = Proxy.revocable({}, {});
      revoke();
      throw proxy;
    `),
    coreModuleUrl: await writeCore(t, 'llama_webgpu_core_st.js', ''),
  });
  assert.equal((await bridge._runtime._ensureCore()).kind, 'llama_webgpu_core_st.js');
  assert.ok(bridge._runtime._runtimeNotes.includes('core_mem64_unavailable:init_error'));
});
