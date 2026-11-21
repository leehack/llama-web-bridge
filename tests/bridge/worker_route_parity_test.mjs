// Characterization of main's bridge-worker start route. Passes on main; must keep passing.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { BridgeWorkerProxy } from '../../js/src/worker_proxy.ts';

const PAGE = 'https://app.test/page/index.html';
const MODE = '__llamadartBridgeWorker=1';
const wrapper = (url) => `import * as workerModule from ${JSON.stringify(url)};\nif (workerModule && typeof workerModule.enableBridgeWorkerHost === 'function') { workerModule.enableBridgeWorkerHost(); }\n`;

// engine: how `new Worker(crossOriginUrl)` fails. Chromium throws; Firefox constructs and errors later.
function stub(t, { engine, blobThrows = false }) {
  const state = { attempts: [], blobs: [], revoked: [], workers: [] };
  const globals = {
    origin: 'https://app.test',
    window: { location: { href: PAGE } },
    addEventListener() {},
    removeEventListener() {},
    Worker: class {
      constructor(url, options) {
        const href = String(url);
        state.attempts.push({ url: href, options });
        const isBlob = href.startsWith('blob:');
        if (isBlob && blobThrows) throw new TypeError('blob workers refused');
        if (!isBlob && engine === 'chromium' && new URL(href).origin !== 'https://app.test') {
          throw new DOMException(`Script at '${href}' cannot be accessed from origin 'https://app.test'.`, 'SecurityError');
        }
        this.url = href;
        this.posted = [];
        state.workers.push(this);
      }
      postMessage(message) { this.posted.push(message); }
      terminate() {}
    },
  };
  const originals = Object.fromEntries(Object.keys(globals).map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
  t.after(() => {
    for (const [key, descriptor] of Object.entries(originals)) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
    }
  });
  t.mock.method(URL, 'createObjectURL', (blob) => { state.blobs.push(blob); return `blob:https://app.test/${state.blobs.length}`; });
  t.mock.method(URL, 'revokeObjectURL', (url) => state.revoked.push(url));
  return state;
}

const CDN_ENTRY = 'https://cdn.test/v/llama_webgpu_bridge_worker.js';
const CDN_BUNDLE = 'https://cdn.test/v/llama_webgpu_bridge.js';
const routes = [
  // [workerUrl, engine, expected Worker attempts in order, expected blob wrapper imports]
  ['./assets/llama_webgpu_bridge_worker.js', 'chromium', ['https://app.test/page/assets/llama_webgpu_bridge_worker.js'], []],
  ['./assets/llama_webgpu_bridge_worker.js', 'firefox', ['https://app.test/page/assets/llama_webgpu_bridge_worker.js'], []],
  ['/assets/llama_webgpu_bridge_worker.js', 'chromium', ['https://app.test/assets/llama_webgpu_bridge_worker.js'], []],
  ['/assets/llama_webgpu_bridge_worker.js', 'firefox', ['https://app.test/assets/llama_webgpu_bridge_worker.js'], []],
  ['https://app.test/x/llama_webgpu_bridge_worker.js', 'chromium', ['https://app.test/x/llama_webgpu_bridge_worker.js'], []],
  ['https://app.test/x/llama_webgpu_bridge.js', 'chromium', [`https://app.test/x/llama_webgpu_bridge.js?${MODE}`], []],
  ['https://app.test/x/llama_webgpu_bridge.js', 'firefox', [`https://app.test/x/llama_webgpu_bridge.js?${MODE}`], []],
  [CDN_ENTRY, 'chromium', [CDN_ENTRY, `${CDN_BUNDLE}?${MODE}`, 'blob:https://app.test/1'], [CDN_ENTRY]],
  [CDN_ENTRY, 'firefox', [CDN_ENTRY], []],
  [CDN_BUNDLE, 'chromium', [`${CDN_BUNDLE}?${MODE}`, 'blob:https://app.test/1'], [CDN_BUNDLE]],
  [CDN_BUNDLE, 'firefox', [`${CDN_BUNDLE}?${MODE}`], []],
  ['../llama_webgpu_bridge_worker.js', 'chromium', ['https://app.test/llama_webgpu_bridge_worker.js'], []],
];

for (const [workerUrl, engine, attempts, imports] of routes) {
  test(`worker route: ${workerUrl} on ${engine}`, async (t) => {
    const state = stub(t, { engine });
    const config = { coreModuleUrl: './core.js', wasmUrl: './core.wasm' };
    const proxy = new BridgeWorkerProxy({ moduleUrl: workerUrl, config });
    t.after(() => { proxy._clearReadyTimeout(); proxy._endCspWatch?.(); });
    assert.deepEqual(state.attempts.map((a) => a.url), attempts);
    for (const attempt of state.attempts) assert.deepEqual(attempt.options, { type: 'module' });
    assert.deepEqual(await Promise.all(state.blobs.map((b) => b.text())), imports.map(wrapper));
    assert.equal(state.workers.length, 1);
    assert.equal(state.workers[0].url, attempts.at(-1));
    assert.equal(state.workers[0].posted.length, 1);
    assert.equal(state.workers[0].posted[0].type, 'init');
  });
}

test('worker route: every blob attempt refused rethrows the last error and revokes its URLs', async (t) => {
  const state = stub(t, { engine: 'chromium', blobThrows: true });
  assert.throws(() => new BridgeWorkerProxy({ moduleUrl: CDN_ENTRY, config: {} }), (error) => error instanceof TypeError && error.message === 'blob workers refused');
  assert.deepEqual(state.attempts.map((a) => a.url), [CDN_ENTRY, `${CDN_BUNDLE}?${MODE}`, 'blob:https://app.test/1', 'blob:https://app.test/2']);
  assert.deepEqual(await Promise.all(state.blobs.map((b) => b.text())), [CDN_ENTRY, CDN_BUNDLE].map(wrapper));
  assert.deepEqual(state.revoked, ['blob:https://app.test/1', 'blob:https://app.test/2']);
});

// Every URL option reaches the worker exactly as the caller wrote it, in every start mode.
const forms = { relative: './dir/file', 'root-relative': '/dir/file', 'absolute same-origin': 'https://app.test/dir/file', 'absolute CDN': 'https://cdn.test/dir/file', bare: 'llama-core' };
const starts = { 'direct same-origin worker': ['https://app.test/x/llama_webgpu_bridge_worker.js', 'chromium'], 'CDN worker, blob start (Chromium)': [CDN_ENTRY, 'chromium'], 'CDN worker, direct attempt (Firefox)': [CDN_ENTRY, 'firefox'] };
for (const [start, [workerUrl, engine]] of Object.entries(starts)) {
  for (const option of ['coreModuleUrl', 'coreModuleUrlMem64', 'wasmUrl', 'wasmUrlMem64']) {
    for (const [form, value] of Object.entries(forms)) {
      test(`config passthrough: ${option} ${form} to a ${start}`, (t) => {
        const state = stub(t, { engine });
        const config = { coreModuleUrl: 'https://app.test/core.js', preferMemory64: true, [option]: `${value}.${option.startsWith('wasm') ? 'wasm' : 'js'}` };
        const proxy = new BridgeWorkerProxy({ moduleUrl: workerUrl, config });
        t.after(() => { proxy._clearReadyTimeout(); proxy._endCspWatch?.(); });
        const { workerStart, ...posted } = state.workers[0].posted[0].config;
        assert.deepEqual(posted, config);
      });
    }
  }
}
