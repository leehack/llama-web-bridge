import assert from 'node:assert/strict';
import { test } from 'node:test';

import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';
import {
  createRealWorkerBridge,
  withStubWorkerEnvironment,
} from './bridge_operation_queue_fixtures.mjs';

const MODEL_URL = 'https://models.test/model.gguf';
const LOAD_OPTIONS = { nGpuLayers: 0, nThreads: 1, useCache: false };
const TOKEN = 'lane216-fixture-token';
const FIXTURES_URL = new URL('./bridge_operation_queue_fixtures.mjs', import.meta.url).href;

function moduleUrl(source) {
  return `data:text/javascript,${encodeURIComponent(source)}`;
}

function setup(t, wasm64Factory) {
  const warnings = [];
  t.mock.method(LlamaWebGpuBridgeRuntime.prototype, '_resolveCoreInitTimeoutMs', () => 30);
  t.mock.method(globalThis, 'fetch', async (url) => {
    assert.equal(url, MODEL_URL);
    return new Response(Uint8Array.of(1, 2, 3));
  });
  const factory = `async () => {
    const { createStubCore } = await import(${JSON.stringify(FIXTURES_URL)});
    const core = createStubCore();
    const ccall = core.ccall.bind(core);
    core.ccall = (name, ...args) => {
      if (name === 'llamadart_webgpu_backends_json') return '["CPU"]';
      if (['llamadart_webgpu_probe', 'llamadart_webgpu_supports_pthreads',
           'llamadart_webgpu_load_model', 'llamadart_webgpu_free_model'].includes(name)) return 0;
      return ccall(name, ...args);
    };
    return core;
  }`;
  const coreModuleUrl = moduleUrl(`export default ${factory}`);
  return {
    warnings,
    config: {
      allowSingleThreadCore: true,
      coreModuleUrl,
      coreModuleUrlMem64: wasm64Factory ? moduleUrl(`export default ${wasm64Factory}`) : coreModuleUrl,
      wasmUrlMem64: `https://assets.test/core.wasm?token=${TOKEN}`,
    },
  };
}

function assertFallback(bridge, warnings, reason, { reused = false } = {}) {
  const metadata = bridge.getModelMetadata();
  const notes = metadata['llamadart.webgpu.runtime_notes'];
  assert.equal(metadata['llamadart.webgpu.core_variant'], 'wasm32');
  assert.deepEqual(notes.split(';').filter((note) => /^core_(mem64|wasm32)/.test(note)), [
    ...(reused ? [] : ['core_mem64_attempt']),
    'core_mem64_unavailable',
    `core_mem64_unavailable:${reason}`,
    ...(reused ? [] : ['core_wasm32_active']),
  ]);
  assert.doesNotMatch(notes, /assets\.test|fixture-token/);
  assert.equal(warnings.length, 1);
  assert.ok(warnings[0].includes(reason));
  assert.doesNotMatch(warnings[0], /assets\.test|fixture-token/);
}

const failWithSignedUrl = `async ({ locateFile }) => {
  throw new Error('Could not initialize ' + locateFile('core.wasm', ''));
}`;

for (const { reason, supported, factory } of [
  { reason: 'unsupported', supported: false, factory: `async () => { throw new WebAssembly.CompileError('invalid memory limits flags'); }` },
  { reason: 'init_error', supported: true, factory: failWithSignedUrl },
  { reason: 'timeout', supported: true, factory: '() => new Promise(() => {})' },
]) {
  test(`memory64 ${reason} is reported when the load falls back to wasm32`, async (t) => {
    t.mock.method(WebAssembly, 'validate', () => supported);
    const { config, warnings } = setup(t, factory);
    const bridge = new LlamaWebGpuBridge({
      ...config, disableWorker: true, logger: { warn: (message) => warnings.push(message) },
    });
    t.after(() => bridge.dispose());
    await bridge.loadModelFromUrl(MODEL_URL, LOAD_OPTIONS);
    assertFallback(bridge, warnings, reason);
    await bridge.loadModelFromUrl(MODEL_URL, LOAD_OPTIONS);
    assertFallback(bridge, warnings, reason, { reused: true });
  });
}

test('successful memory64 loads have no fallback diagnostic', async (t) => {
  const { config, warnings } = setup(t);
  const bridge = new LlamaWebGpuBridge({
    ...config, disableWorker: true, logger: { warn: (message) => warnings.push(message) },
  });
  t.after(() => bridge.dispose());
  await bridge.loadModelFromUrl(MODEL_URL, LOAD_OPTIONS);
  const metadata = bridge.getModelMetadata();
  assert.equal(metadata['llamadart.webgpu.core_variant'], 'wasm64');
  assert.doesNotMatch(metadata['llamadart.webgpu.runtime_notes'], /core_mem64_unavailable/);
  assert.deepEqual(warnings, []);
});

test('the memory64 reason survives the worker host and proxy round trip', async (t) => {
  t.mock.method(WebAssembly, 'validate', () => true);
  const { config, warnings } = setup(t, failWithSignedUrl);
  t.mock.method(console, 'warn', (message) => warnings.push(message));
  await withStubWorkerEnvironment(async ({ workers }) => {
    const bridge = createRealWorkerBridge(config);
    const worker = workers[0];
    const oldSelf = globalThis.self;
    globalThis.self = { postMessage: (message) => worker.onmessage({ data: structuredClone(message) }) };
    try {
      installBridgeWorkerHost();
      worker.postMessage = (message) => self.onmessage({ data: structuredClone(message) });
      await worker.postMessage(worker.posted[0]);
      await bridge.loadModelFromUrl(MODEL_URL, LOAD_OPTIONS);
      assert.equal(bridge.getModelMetadata()['llamadart.webgpu.execution'], 'worker');
      assertFallback(bridge, warnings, 'init_error');
      await bridge.loadModelFromUrl(MODEL_URL, LOAD_OPTIONS);
      assert.equal(bridge.getModelMetadata()['llamadart.webgpu.execution'], 'worker');
      assertFallback(bridge, warnings, 'init_error', { reused: true });
    } finally {
      await bridge.dispose();
      if (oldSelf === undefined) delete globalThis.self;
      else globalThis.self = oldSelf;
    }
  });
});
