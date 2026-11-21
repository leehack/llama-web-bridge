import assert from 'node:assert/strict';
import { test } from 'node:test';

import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';
import { createDirectBridge, createStubCore } from './bridge_operation_queue_fixtures.mjs';
import { readNativeCoreSource } from './native_core_source.mjs';

const MODEL = 'https://models.test/mm/tiny.gguf';
const SIGNED_MODEL = `${MODEL}?sig=a/b&token=${'x'.repeat(280)}`;
const OPTIONS = { nGpuLayers: 0, useCache: false, forceRemoteFetchBackend: true, modelBytesHint: 3 };
const FETCH = 'llamadart_webgpu_load_model_from_url';
const STAGE = 'llamadart_webgpu_load_model';

function trackLoads(t, core) {
  const loads = [];
  const ccall = core.ccall.bind(core);
  t.mock.method(core, 'ccall', (name, result, types, args, ...rest) => {
    if (name === 'llamadart_webgpu_supports_pthreads') return 1;
    if (name === 'llamadart_webgpu_pthread_pool_size') return 4;
    if (name === 'llamadart_webgpu_probe' || name === 'llamadart_webgpu_free_model') return 0;
    if (name === 'llamadart_webgpu_backends_json') return '[]';
    if (name === FETCH || name === STAGE) {
      loads.push({ name, path: args[0] });
      return 0;
    }
    return ccall(name, result, types, args, ...rest);
  });
  return loads;
}

function stubWorkerScope(t, self) {
  const descriptors = Object.fromEntries(['WorkerGlobalScope', 'self'].map((key) => [
    key, Object.getOwnPropertyDescriptor(globalThis, key),
  ]));
  t.after(() => {
    for (const [key, descriptor] of Object.entries(descriptors)) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
  globalThis.WorkerGlobalScope = class {
    static [Symbol.hasInstance](value) { return value === globalThis; }
  };
  globalThis.self = self;
}

function stubModelRequests(t) {
  t.mock.method(LlamaWebGpuBridgeRuntime.prototype, '_resolveRemoteFetchUrl',
    async (url) => ({ resolvedUrl: url, sizeBytes: 3 }));
  t.mock.method(LlamaWebGpuBridgeRuntime.prototype, '_getCachedModelResponse',
    async () => new Response(new Uint8Array([1, 2, 3])));
}

test('native WasmFS fetch uses the complete model URL and a synthetic local mount', () => {
  const source = readNativeCoreSource();
  const loader = source.match(/EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_load_model_from_url\([\s\S]*?^}/m)?.[0];
  assert.ok(loader, 'missing native model URL loader');
  assert.doesNotMatch(loader, /wasmfs_create_fetch_backend\(/);
  assert.doesNotMatch(loader, /\.(?:rfind|substr)\(/);
  const retry = loader.match(/\bdo\s*\{([^}]*)\}\s*while\s*\(fd\s*==\s*-EEXIST\)/);
  assert.ok(retry, 'the synthetic mount must retry on EEXIST');
  assert.equal([...loader.matchAll(/create_shared_fetch_file\(/g)].length, 1);
  assert.match(retry[1], /fetch_file_path\s*=\s*"\/remote_model_"\s*\+\s*std::to_string\(\+\+\w+\)\s*\+\s*"\.gguf"/);
  assert.match(retry[1], /fd\s*=\s*llamadart_webgpu::create_shared_fetch_file\(\s*fetch_file_path\.c_str\(\),\s*model_url,\s*effective_chunk_size\)/);
  assert.match(loader, /unlink\(fetch_file_path\.c_str\(\)\)/);
});

test('the app-owned worker passes plain and signed model URLs intact to native fetch', async (t) => {
  stubWorkerScope(t, globalThis);
  stubModelRequests(t);
  const { bridge, core } = createDirectBridge();
  const loads = trackLoads(t, core);
  try {
    await bridge.loadModelFromUrl(MODEL, OPTIONS);
    await bridge.loadModelFromUrl(SIGNED_MODEL, OPTIONS);
    assert.deepEqual(loads, [MODEL, SIGNED_MODEL].map((path) => ({ name: FETCH, path })));
  } finally {
    await bridge.dispose();
  }
});

test('the bridge-owned worker host passes the complete signed URL to native fetch', async (t) => {
  const replies = [];
  const scope = { postMessage: (message) => replies.push(message) };
  stubWorkerScope(t, scope);
  stubModelRequests(t);
  const core = createStubCore();
  const loads = trackLoads(t, core);
  installBridgeWorkerHost();
  const send = (data) => scope.onmessage({ data });
  await send({ type: 'init', config: { coreModuleFactory: async () => core } });
  try {
    await send({ type: 'call', id: 1, method: 'loadModelFromUrl', args: [SIGNED_MODEL, OPTIONS] });
    const reply = replies.find(({ id, type }) => id === 1 && type !== 'event');
    assert.equal(reply?.type, 'result', reply?.message);
    assert.deepEqual(loads, [{ name: FETCH, path: SIGNED_MODEL }]);
  } finally {
    await send({ type: 'call', id: 2, method: 'dispose' });
  }
});

test('page-thread loading stages the model even when remote fetch is forced', async (t) => {
  stubModelRequests(t);
  const { bridge, core } = createDirectBridge();
  const loads = trackLoads(t, core);
  try {
    await bridge.loadModelFromUrl(MODEL, OPTIONS);
    assert.deepEqual(loads.map(({ name }) => name), [STAGE]);
  } finally {
    await bridge.dispose();
  }
});

test('the worker stages both inferred and explicit split GGUFs with forced remote fetch', async (t) => {
  stubWorkerScope(t, globalThis);
  stubModelRequests(t);
  const { bridge, core } = createDirectBridge();
  const loads = trackLoads(t, core);
  try {
    for (const model of [
      'https://models.test/mm/m-00001-of-00002.gguf',
      ['https://models.test/mm/a.gguf', 'https://models.test/mm/b.gguf'],
    ]) {
      await bridge.loadModelFromUrl(model, OPTIONS);
      assert.ok(bridge.getModelMetadata()['llamadart.webgpu.runtime_notes'].split(';')
        .includes('model_fetch_backend_skipped_split'));
    }
    assert.deepEqual(loads.map(({ name }) => name), [STAGE, STAGE]);
  } finally {
    await bridge.dispose();
  }
});
