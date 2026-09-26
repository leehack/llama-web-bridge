import assert from 'node:assert/strict';

import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';
import {
  createRealWorkerBridge,
  settleQuietly,
  withStubWorkerEnvironment,
  workerDriver,
} from './bridge_operation_queue_fixtures.mjs';

function createProxy(name, callImpl) {
  const proxy = {
    name,
    calls: [],
    disposeCalls: 0,
    async call(method, args, onEvent) {
      proxy.calls.push({ method, args });
      if (method === 'cancel' || method === 'setLogLevel') {
        return { value: undefined };
      }
      return callImpl(method, args, onEvent, proxy);
    },
    async dispose() {
      proxy.disposeCalls += 1;
    },
  };
  return proxy;
}

function createRuntime(overrides = {}) {
  const runtime = {
    _modelBytes: 0,
    _runtimeNotes: [],
    loadCalls: [],
    projectorCalls: [],
    async loadModelFromUrl(url, options) {
      runtime.loadCalls.push({ url, options: { ...options } });
      runtime._modelBytes = 1;
    },
    async loadMultimodalProjector(url) {
      runtime.projectorCalls.push(url);
    },
    async dispose() {},
    getModelMetadata: () => ({}),
    getContextSize: () => 128,
    isGpuActive: () => false,
    getBackendName: () => 'WASM (Prototype bridge)',
    supportsVision: () => false,
    supportsAudio: () => false,
    ...overrides,
  };
  return runtime;
}

function createBridge(overrides = {}) {
  const warnings = [];
  const bridge = Object.create(LlamaWebGpuBridge.prototype);
  Object.assign(bridge, {
    _config: {},
    _runtime: null,
    _workerProxy: null,
    _workerGeneration: 1,
    _workerDisposePromise: null,
    _retiringWorkerDisposals: new Set(),
    _retiredWorkerProxies: new WeakSet(),
    _workerFallbackReason: null,
    _metadata: {},
    _contextSize: 0,
    _gpuActive: false,
    _backendName: 'WASM (Prototype bridge)',
    _supportsVision: false,
    _supportsAudio: false,
    _loadedModelUrl: null,
    _loadedModelOptions: null,
    _loadedMmProjUrl: null,
    _multimodalWorkerCpuMode: false,
    _bridgeWarnRecent: new Map(),
    _operationQueueTail: null,
    _activeOperation: null,
    _nextOperationId: 0,
    _lifecycleState: 'open',
    _shadowStateTransactionDepth: 0,
    _deferredShadowState: null,
    _disposed: false,
    _disposePromise: null,
    _disposalWaiters: new Set(),
    _emitBridgeWarn: (message) => warnings.push(message),
    _createRuntime: () => createRuntime(),
    ...overrides,
  });
  return { bridge, warnings };
}

// Every `new Worker` throws synchronously, as under a CSP that forbids both the
// direct and the blob worker URL, so constructing a BridgeWorkerProxy throws.
async function withBlockedWorkers(run) {
  const originals = {
    Worker: globalThis.Worker,
    createObjectURL: globalThis.URL.createObjectURL,
    revokeObjectURL: globalThis.URL.revokeObjectURL,
  };
  let constructions = 0;
  globalThis.Worker = class BlockedWorker {
    constructor() {
      constructions += 1;
      const error = new Error('Worker construction violates the Content Security Policy');
      error.name = 'SecurityError';
      throw error;
    }
  };
  globalThis.URL.createObjectURL = () => 'blob:blocked-worker';
  globalThis.URL.revokeObjectURL = () => {};
  try {
    return await run({ constructions: () => constructions });
  } finally {
    globalThis.Worker = originals.Worker;
    globalThis.URL.createObjectURL = originals.createObjectURL;
    globalThis.URL.revokeObjectURL = originals.revokeObjectURL;
  }
}

// A CPU-mode worker bridge that never created a direct runtime; its worker
// fails every model call so the facade tries to replace it.
function createCpuWorkerBridgeWithBrokenWorker(overrides = {}) {
  const proxy = createProxy('broken', async (method) => {
    if (method === 'dispose') {
      return { value: undefined };
    }
    throw new Error('Bridge worker crashed');
  });
  const runtimes = [];
  const { bridge, warnings } = createBridge({
    _workerProxy: proxy,
    _loadedModelUrl: 'model.gguf',
    _loadedModelOptions: { nGpuLayers: 0 },
    _createRuntime: () => {
      const runtime = createRuntime({
        disposeCalls: 0,
        async tokenize(text) {
          return [...text].map((character) => character.codePointAt(0));
        },
        async dispose() {
          runtime.disposeCalls += 1;
        },
      });
      runtimes.push(runtime);
      return runtime;
    },
    ...overrides,
  });
  return { bridge, warnings, proxy, runtimes };
}

function assertDirectRuntimeAfterBlockedReplacement(bridge, proxy, runtimes) {
  assert.ok(
    bridge._workerProxy != null || bridge._runtime != null,
    'an open bridge must keep a worker proxy or a direct runtime',
  );
  assert.equal(bridge._workerProxy, null);
  assert.equal(proxy.disposeCalls, 1, 'the broken worker is retired');
  assert.equal(runtimes.length, 1);
  assert.equal(bridge._runtime, runtimes[0], 'the open bridge keeps a direct runtime');
  assert.match(bridge._workerFallbackReason, /Content Security Policy/);
  assert.equal(
    runtimes[0]._runtimeNotes.filter((note) => note.startsWith('worker_fallback:')).length,
    1,
  );
}

// Deterministic errors a healthy worker posts for each facade method.
const FACADE_WORKER_METHODS = [
  {
    method: 'tokenize',
    coreError: 'Tokenization failed: Prompt tokenization failed',
    invoke: (bridge) => bridge.tokenize('hello', true),
    mainThreadResult: [1, 2],
  },
  {
    method: 'detokenize',
    coreError: 'Detokenization failed (code=-1)',
    invoke: (bridge) => bridge.detokenize([1, 2], false),
    mainThreadResult: 'hello',
  },
  {
    method: 'embed',
    coreError: 'Embedding generation failed: Embedding input tokenized to an empty sequence',
    invoke: (bridge) => bridge.embed('hello', {}),
    mainThreadResult: [0.5, 0.25],
  },
  {
    method: 'embedBatch',
    coreError: 'Embedding generation failed: Embeddings cannot be generated during active generation or text-to-speech synthesis',
    invoke: (bridge) => bridge.embedBatch(['a', 'b'], {}),
    mainThreadResult: [[0.5], [0.25]],
  },
  {
    method: 'scoreNextToken',
    coreError: 'Next-token scoring failed: Token id 99 is outside the vocabulary of 32 tokens',
    invoke: (bridge) => bridge.scoreNextToken('hello', { candidates: [99] }),
    mainThreadResult: { candidates: [], top: [], promptTokens: 1 },
  },
  {
    method: 'applyChatTemplate',
    coreError: 'messages is not iterable',
    invoke: (bridge) => bridge.applyChatTemplate([{ role: 'user', content: 'hi' }], true),
    mainThreadResult: 'user: hi\nassistant: ',
  },
];

const NO_MODEL = 'No model loaded. Call loadModelFromUrl first.';

// The shadow state a worker posts. Its runtime reports model_bytes 0 once a
// load has dropped the previous model.
function workerModelState(model) {
  return {
    metadata: {
      'general.name': model ?? '',
      'llamadart.webgpu.model_bytes': model ? '4' : '0',
    },
    contextSize: model ? 64 : 0,
    gpuActive: false,
    backendName: 'WASM (Prototype bridge)',
    supportsVision: false,
    supportsAudio: false,
  };
}

// A proxy that holds at most one model, as a real worker does. `failLoad`
// returns { message, keepsModel, stateless } to fail a load, or null to let
// it succeed; `stateless` rejects without a posted state, as a crash does.
function createModelProxy(name, { failLoad = () => null, failProjector = null } = {}) {
  const proxy = createProxy(name, async (method, args) => {
    if (method === 'loadModelFromUrl') {
      const failure = failLoad(args[0]);
      if (failure) {
        if (!failure.keepsModel) {
          proxy.model = null;
        }
        const error = new Error(failure.message);
        throw failure.stateless ? error : Object.assign(error, { state: workerModelState(proxy.model) });
      }
      proxy.model = args[0];
      return { value: 1, state: workerModelState(proxy.model) };
    }
    if (method === 'loadMultimodalProjector' && failProjector) {
      throw new Error(failProjector);
    }
    if (proxy.model == null) {
      throw Object.assign(new Error(NO_MODEL), { state: workerModelState(null) });
    }
    return { value: { method, model: proxy.model } };
  });
  proxy.model = null;
  return proxy;
}

function modelLoads(proxy) {
  return proxy.calls
    .filter((call) => call.method === 'loadModelFromUrl')
    .map((call) => call.args[0]);
}

// Worker requests that need the loaded model. The capability probe still
// falls back to the main thread on any worker error.
const MODEL_WORKER_METHODS = [
  ['getTextToSpeechCapabilities', (bridge) => bridge.getTextToSpeechCapabilities(), { anyErrorFallsBack: true }],
  ['synthesizeSpeech', (bridge) => bridge.synthesizeSpeech({ text: 'hi' })],
  ['createCompletion', (bridge) => bridge.createCompletion('prompt', { nPredict: 4 })],
  ['tokenize', (bridge) => bridge.tokenize('hello', true)],
  ['detokenize', (bridge) => bridge.detokenize([1, 2], false)],
  ['embed', (bridge) => bridge.embed('hello', {})],
  ['embedBatch', (bridge) => bridge.embedBatch(['a', 'b'], {})],
  ['scoreNextToken', (bridge) => bridge.scoreNextToken('hello', { topK: 1 })],
  ['stateSaveFile', (bridge) => bridge.stateSaveFile('/states/a.bin', [1])],
  ['stateLoadFile', (bridge) => bridge.stateLoadFile('/states/a.bin', 16)],
  ['stateSaveBytes', (bridge) => bridge.stateSaveBytes([1])],
  ['stateLoadBytes', (bridge) => bridge.stateLoadBytes(new Uint8Array([1, 2]), 16)],
  ['loadDecisionHead', (bridge) => bridge.loadDecisionHead(new Uint8Array([1, 2]), {})],
];

const CASES = [
  ['CPU projector setup keeps a direct runtime when the worker cannot be replaced', async () => {
    await withBlockedWorkers(async ({ constructions }) => {
      const { bridge, proxy, runtimes } = createCpuWorkerBridgeWithBrokenWorker({
        _multimodalWorkerCpuMode: true,
      });

      await assert.rejects(
        bridge.loadMultimodalProjector('mmproj.gguf'),
        /CPU multimodal projector setup failed \(Worker construction violates the Content Security Policy\)/,
      );

      assert.ok(constructions() > 0, 'the replacement worker was attempted');
      assertDirectRuntimeAfterBlockedReplacement(bridge, proxy, runtimes);
      assert.deepEqual(await bridge.tokenize('hi'), [104, 105]);
    });
  }],

  ['CPU media completion keeps a direct runtime when the worker cannot be replaced', async () => {
    await withBlockedWorkers(async ({ constructions }) => {
      const { bridge, proxy, runtimes } = createCpuWorkerBridgeWithBrokenWorker();

      await assert.rejects(
        bridge.createCompletion('describe', {
          parts: [{ type: 'image', bytes: new Uint8Array(1) }],
        }),
        /CPU multimodal worker setup failed \(Worker construction violates the Content Security Policy\)/,
      );

      assert.ok(constructions() > 0, 'the replacement worker was attempted');
      assertDirectRuntimeAfterBlockedReplacement(bridge, proxy, runtimes);
      assert.deepEqual(await bridge.tokenize('hi'), [104, 105]);
    });
  }],

  ['GPU media completion falls back once when the worker cannot be replaced', async () => {
    await withBlockedWorkers(async () => {
      const { bridge, proxy, runtimes } = createCpuWorkerBridgeWithBrokenWorker({
        _loadedModelOptions: { nGpuLayers: 99 },
      });
      bridge._createRuntime = ((createRuntimeImpl) => () => {
        const runtime = createRuntimeImpl();
        runtime.createCompletion = async (prompt) => `main:${prompt}`;
        return runtime;
      })(bridge._createRuntime);

      assert.equal(
        await bridge.createCompletion('describe', {
          parts: [{ type: 'image', bytes: new Uint8Array(1) }],
        }),
        'main:describe',
      );

      // Both the failed replacement and the caller fall back for the same
      // error; the runtime records that fallback once.
      assertDirectRuntimeAfterBlockedReplacement(bridge, proxy, runtimes);
      assert.equal(runtimes[0].loadCalls.length, 1, 'the model is reloaded on the main thread');
    });
  }],

  ['disposal owns teardown of a runtime created by a blocked worker replacement', async () => {
    await withBlockedWorkers(async () => {
      let failProjector;
      const { bridge, proxy, runtimes } = createCpuWorkerBridgeWithBrokenWorker({
        _multimodalWorkerCpuMode: true,
      });
      const brokenCall = proxy.call;
      proxy.call = (method, args, onEvent) => {
        if (method !== 'loadMultimodalProjector') {
          return brokenCall(method, args, onEvent);
        }
        proxy.calls.push({ method, args });
        return new Promise((_resolve, reject) => {
          failProjector = () => reject(new Error('Bridge worker crashed'));
        });
      };

      const projector = settleQuietly(bridge.loadMultimodalProjector('mmproj.gguf'));
      for (let i = 0; i < 20 && !failProjector; i += 1) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      assert.ok(failProjector, 'the projector request reached the worker');
      const disposal = bridge.dispose();
      failProjector();

      await assert.rejects(projector, /CPU multimodal projector setup failed/);
      await disposal;

      assert.equal(bridge._lifecycleState, 'disposed');
      assert.equal(bridge._workerProxy, null);
      assert.equal(bridge._runtime, null);
      // The settle-first owner may still fall back once; disposal then tears
      // that runtime down and nothing recreates one afterwards.
      assert.equal(runtimes.length, 1);
      assert.equal(runtimes[0].disposeCalls, 1);
      await assert.rejects(bridge.tokenize('hi'), /disposed/i);
      await assert.rejects(bridge.loadMultimodalProjector('mmproj.gguf'), /disposed/i);
      assert.equal(runtimes.length, 1, 'no call after disposal creates a runtime');
      assert.equal(bridge._runtime, null);
    });
  }],

  ['main-thread fallback classification is decided by the serialized error text', () => {
    const { bridge } = createBridge();
    const fallsBack = (error) => bridge._shouldFallbackToMainThread(error);

    assert.equal(fallsBack(new Error('Worker request timeout')), true);
    assert.equal(fallsBack(new Error('Worker init timeout')), true);
    assert.equal(fallsBack(new Error('Bridge worker crashed')), true);
    assert.equal(fallsBack(new Error('response drain read timed out')), true);
    assert.equal(fallsBack('Worker proxy is not available'), true);
    assert.equal(fallsBack(new Error('Aborted(native code called abort())')), false);
    assert.equal(fallsBack(new Error('std::bad_alloc')), false);
    assert.equal(fallsBack(new Error('Out of memory')), false);
    assert.equal(fallsBack(new Error('unrelated model failure')), false);
  }],

  ['request-timeout and recoverable-FS classifiers agree on the exact timeout text', () => {
    const { bridge } = createBridge();

    for (const text of ['Worker request timeout', 'Worker init timeout', 'worker timed out']) {
      assert.equal(bridge._isWorkerRequestTimeoutError(new Error(text)), true, text);
    }
    assert.equal(bridge._isWorkerRequestTimeoutError(new Error('FS error: not found')), false);

    for (const text of ['FS error', 'No such file', 'not found', 'Invalid argument', 'read timed out']) {
      assert.equal(bridge._isRecoverableWorkerFsError(new Error(text)), true, text);
    }
    assert.equal(bridge._isRecoverableWorkerFsError(new Error('Bridge worker crashed')), false);

    const timeout = new Error('Worker request timeout');
    assert.equal(
      bridge._isRecoverableWorkerFsError(timeout) && !bridge._isWorkerRequestTimeoutError(timeout),
      false,
      'a request timeout never selects the FS restart-and-retry path',
    );
  }],

  ['worker model-load FS error restarts the worker once and retries there', async () => {
    let attempts = 0;
    const failing = createProxy('fs-failing', async (method) => {
      if (method !== 'loadModelFromUrl') {
        return { value: undefined };
      }
      attempts += 1;
      throw new Error('FS error: no such file or directory');
    });
    const replacement = createProxy('fs-replacement', async (method) => {
      if (method !== 'loadModelFromUrl') {
        return { value: undefined };
      }
      attempts += 1;
      return { value: 'loaded-on-replacement' };
    });
    const replaceCalls = [];
    const { bridge, warnings } = createBridge({ _workerProxy: failing });
    bridge._replaceWorkerProxyForMultimodalCpuMode = async () => {
      replaceCalls.push(bridge._workerProxy?.name);
      bridge._workerProxy = replacement;
    };

    const result = await bridge.loadModelFromUrl('model.gguf', { nGpuLayers: 99 });

    assert.equal(result, 'loaded-on-replacement');
    assert.equal(attempts, 2);
    assert.deepEqual(replaceCalls, ['fs-failing']);
    assert.equal(bridge._workerProxy, replacement);
    assert.equal(bridge._runtime, null, 'no main-thread runtime is created');
    assert.equal(bridge._workerFallbackReason, null);
    assert.ok(warnings.some((message) => message.includes('worker model-load FS error detected')));
  }],

  ['a model-host HTTP error is not a recoverable FS error and keeps the worker', async () => {
    const { bridge: classifier } = createBridge();
    for (const text of [
      'Failed to fetch model shard: 404 Not Found',
      'Failed to fetch model shard: 403 Forbidden',
      'Failed to fetch model shard: 500 Internal Server Error',
      'Range resume not honored for model shard: 404 Not Found',
    ]) {
      assert.equal(classifier._isRecoverableWorkerFsError(new Error(text)), false, text);
    }

    const worker = createModelProxy('holds-a', {
      failLoad: (url) => (url === 'b.gguf'
        ? { message: 'Failed to fetch model shard: 404 Not Found' }
        : null),
    });
    const { bridge, warnings } = createBridge({
      _workerProxy: worker,
      _createWorkerProxy: () => {
        throw new Error('the worker must not be restarted for an HTTP status');
      },
      _createRuntime: () => {
        throw new Error('the main-thread runtime must not be created');
      },
    });
    await bridge.loadModelFromUrl('a.gguf', { nGpuLayers: 0 });

    await assert.rejects(
      bridge.loadModelFromUrl('b.gguf', {}),
      (error) => error.message === 'Failed to fetch model shard: 404 Not Found',
    );

    assert.equal(bridge._workerProxy, worker);
    assert.equal(worker.disposeCalls, 0);
    assert.deepEqual(modelLoads(worker), ['a.gguf', 'b.gguf']);
    assert.ok(!warnings.some((message) => message.includes('restarting worker')));
  }],

  ['a failed load that restarted the worker forgets the previous model', async () => {
    const fsError = { message: 'FS error: no such file or directory' };
    const original = createModelProxy('holds-a', {
      failLoad: (url) => (url === 'b.gguf' ? fsError : null),
    });
    const replacement = createModelProxy('replacement', {
      failLoad: (url) => (url === 'b.gguf' ? fsError : null),
    });
    const { bridge } = createBridge({
      _workerProxy: original,
      _createWorkerProxy: () => replacement,
      _createRuntime: () => {
        throw new Error('the main-thread runtime must not be created');
      },
    });
    await bridge.loadModelFromUrl('a.gguf', { nGpuLayers: 0 });
    assert.equal(bridge.getContextSize(), 64);

    await assert.rejects(bridge.loadModelFromUrl('b.gguf', {}), /FS error/);

    assert.equal(bridge._workerProxy, replacement);
    assert.equal(original.disposeCalls, 1, 'the worker holding a.gguf was retired');
    assert.equal(bridge._loadedModelUrl, null, 'the facade no longer claims a.gguf');
    assert.equal(bridge._loadedModelOptions, null);
    assert.equal(bridge._loadedMmProjUrl, null);
    assert.equal(bridge.getContextSize(), 0);
    assert.equal(bridge.getModelMetadata()['general.name'], '');

    for (const [method, invoke, { anyErrorFallsBack = false } = {}] of MODEL_WORKER_METHODS) {
      if (!anyErrorFallsBack) {
        await assert.rejects(invoke(bridge), (error) => error.message === NO_MODEL, method);
      }
    }
    assert.deepEqual(modelLoads(replacement), ['b.gguf'], 'a.gguf is never silently reloaded');
    assert.equal(bridge._runtime, null);

    await bridge.loadModelFromUrl('a.gguf', { nGpuLayers: 0 });
    assert.deepEqual((await bridge.tokenize('hello')).model, 'a.gguf');
  }],

  ['a restart whose retry posts no state still forgets the previous model', async () => {
    // A real worker's second load fails with "FS error" before it drops the
    // model it holds, so only the restart discards a.gguf.
    const original = createModelProxy('holds-a', {
      failLoad: (url) => (url === 'b.gguf' ? { message: 'FS error', keepsModel: true } : null),
    });
    const replacement = createModelProxy('replacement', {
      failLoad: (url) => (url === 'b.gguf'
        ? { message: 'FS error: no such file or directory', stateless: true }
        : null),
    });
    const { bridge } = createBridge({
      _workerProxy: original,
      _createWorkerProxy: () => replacement,
      _createRuntime: () => {
        throw new Error('the main-thread runtime must not be created');
      },
    });
    await bridge.loadModelFromUrl('a.gguf', { nGpuLayers: 0 });

    await assert.rejects(bridge.loadModelFromUrl('b.gguf', {}), /FS error/);

    assert.equal(bridge._workerProxy, replacement);
    assert.equal(bridge._loadedModelUrl, null);
    assert.equal(bridge.getContextSize(), 0);
    assert.equal(bridge.getModelMetadata()['general.name'], undefined);
    await assert.rejects(bridge.tokenize('hi'), (error) => error.message === NO_MODEL);
    assert.deepEqual(modelLoads(replacement), ['b.gguf']);
  }],

  ['a worker load cancelled by its signal syncs from the worker reply', async () => {
    for (const workerLoadsB of [false, true]) {
      let settleLoad = null;
      const worker = createModelProxy('holds-a');
      const call = worker.call;
      worker.call = (method, args, ...rest) => {
        if (method !== 'loadModelFromUrl' || args[0] !== 'b.gguf') {
          return call(method, args, ...rest);
        }
        worker.calls.push({ method, args });
        return new Promise((resolve, reject) => {
          settleLoad = () => {
            if (workerLoadsB) {
              worker.model = 'b.gguf';
              resolve({ value: 1, state: workerModelState('b.gguf') });
              return;
            }
            // The aborted download already dropped the previous model.
            worker.model = null;
            reject(Object.assign(new Error('Model load was cancelled.'), {
              name: 'AbortError',
              state: workerModelState(null),
            }));
          };
        });
      };
      const { bridge } = createBridge({ _workerProxy: worker });
      await bridge.loadModelFromUrl('a.gguf', { nGpuLayers: 0 });
      const label = `workerLoadsB=${workerLoadsB}`;

      const controller = new AbortController();
      const load = settleQuietly(bridge.loadModelFromUrl('b.gguf', { signal: controller.signal }));
      for (let i = 0; i < 20 && !settleLoad; i += 1) {
        await Promise.resolve();
      }
      assert.ok(settleLoad, `${label}: the load reached the worker`);
      controller.abort();
      settleLoad();
      await assert.rejects(load, (error) => error.name === 'AbortError', label);
      // The next request queues behind the cancelled load's cleanup.
      const next = await bridge.tokenize('hi').then((value) => value, (error) => error);

      assert.equal(bridge._loadedModelUrl, null, `${label}: the facade no longer claims a.gguf`);
      assert.equal(bridge._workerModelMissing, false, label);
      if (workerLoadsB) {
        assert.equal(next.model, 'b.gguf', label);
      } else {
        assert.equal(next.message, NO_MODEL, label);
      }
    }
  }],

  ['a failed worker load re-syncs the model bookkeeping with the worker state', async () => {
    for (const { failure, keepsModel } of [
      { failure: 'Failed to fetch model shard: 500 Internal Server Error', keepsModel: false },
      { failure: 'Array buffer allocation failed', keepsModel: false },
      { failure: 'WebGPU backend probe failed', keepsModel: true },
    ]) {
      const worker = createModelProxy('holds-a', {
        failLoad: (url) => (url === 'b.gguf' ? { message: failure, keepsModel } : null),
      });
      const { bridge } = createBridge({ _workerProxy: worker });
      await bridge.loadModelFromUrl('a.gguf', { nGpuLayers: 0 });

      await assert.rejects(bridge.loadModelFromUrl('b.gguf', {}), (error) => error.message === failure);

      assert.equal(bridge._workerProxy, worker, failure);
      assert.equal(bridge._loadedModelUrl, keepsModel ? 'a.gguf' : null, failure);
      assert.deepEqual(bridge._loadedModelOptions, keepsModel ? { nGpuLayers: 0 } : null, failure);
      assert.equal(bridge.getContextSize(), keepsModel ? 64 : 0, failure);
    }
  }],

  ['a failed direct-runtime load forgets the model the runtime dropped', async () => {
    for (const keepsModel of [false, true]) {
      const runtime = createRuntime({
        async loadModelFromUrl(url) {
          if (url === 'b.gguf') {
            if (!keepsModel) {
              runtime._modelBytes = 0;
            }
            throw new Error('Failed to fetch model shard: 404 Not Found');
          }
          runtime._modelBytes = 4;
        },
      });
      const { bridge } = createBridge({ _runtime: runtime });
      await bridge.loadModelFromUrl('a.gguf', { nGpuLayers: 0 });
      assert.equal(bridge._loadedModelUrl, 'a.gguf');

      await assert.rejects(bridge.loadModelFromUrl('b.gguf', {}), /404 Not Found/);

      assert.equal(bridge._loadedModelUrl, keepsModel ? 'a.gguf' : null, `keepsModel=${keepsModel}`);
    }
  }],

  ['a multimodal restart that could not reload the model restores it for the next request', async () => {
    for (const [method, invoke] of MODEL_WORKER_METHODS) {
      const reloadError = { message: 'FS error: no such file or directory' };
      const original = createModelProxy('holds-a', { failProjector: 'FS error: projector' });
      // The CPU projector recovery restarts twice and cannot reload a.gguf.
      const firstReplacement = createModelProxy('first-replacement', { failLoad: () => reloadError });
      let secondReplacementLoads = 0;
      const secondReplacement = createModelProxy('second-replacement', {
        failLoad: () => {
          secondReplacementLoads += 1;
          return secondReplacementLoads === 1 ? reloadError : null;
        },
      });
      const replacements = [firstReplacement, secondReplacement];
      const { bridge } = createBridge({
        _workerProxy: original,
        _createWorkerProxy: () => replacements.shift(),
        _createRuntime: () => {
          throw new Error(`${method}: the main-thread runtime must not be created`);
        },
      });
      await bridge.loadModelFromUrl('a.gguf', { nGpuLayers: 0 });

      await assert.rejects(
        bridge.loadMultimodalProjector('mmproj.gguf'),
        /CPU multimodal projector setup failed/,
        method,
      );
      assert.equal(bridge._workerProxy, secondReplacement, method);
      assert.equal(secondReplacement.model, null, `${method}: the replacement starts empty`);
      assert.equal(bridge._loadedModelUrl, 'a.gguf', method);

      const result = await invoke(bridge);

      assert.equal(result.model, 'a.gguf', method);
      assert.equal(result.method, method, method);
      assert.deepEqual(modelLoads(secondReplacement), ['a.gguf', 'a.gguf'], method);
      assert.deepEqual(
        secondReplacement.calls.find((call) => call.method === 'loadModelFromUrl').args[1],
        { nGpuLayers: 0 },
        `${method}: the model reloads with its remembered options`,
      );
      assert.equal(bridge._workerProxy, secondReplacement, method);
      assert.equal(bridge._runtime, null, method);
      assert.equal(bridge._workerFallbackReason, null, method);

      await invoke(bridge);
      assert.deepEqual(modelLoads(secondReplacement), ['a.gguf', 'a.gguf'], `${method}: restored once`);
    }
  }],

  ['a model restore that fails with a core error is rethrown and keeps the worker', async () => {
    const original = createModelProxy('holds-a', { failProjector: 'FS error: projector' });
    const gone = { message: 'Failed to fetch model shard: 404 Not Found' };
    const firstReplacement = createModelProxy('first-replacement', { failLoad: () => gone });
    const secondReplacement = createModelProxy('second-replacement', { failLoad: () => gone });
    const replacements = [firstReplacement, secondReplacement];
    const { bridge } = createBridge({
      _workerProxy: original,
      _createWorkerProxy: () => {
        const next = replacements.shift();
        if (!next) {
          throw new Error('the restore must not restart the worker again');
        }
        return next;
      },
      _createRuntime: () => {
        throw new Error('the main-thread runtime must not be created');
      },
    });
    await bridge.loadModelFromUrl('a.gguf', { nGpuLayers: 0 });
    await assert.rejects(bridge.loadMultimodalProjector('mmproj.gguf'), /CPU multimodal/);
    const current = bridge._workerProxy;

    await assert.rejects(
      bridge.tokenize('hello'),
      (error) => error.message === 'Failed to fetch model shard: 404 Not Found',
    );

    assert.equal(bridge._workerProxy, current);
    assert.equal(bridge._runtime, null);
    assert.equal(bridge._workerFallbackReason, null);
  }],

  ['worker model-load request timeout skips the worker restart and falls back to main thread', async () => {
    let attempts = 0;
    const proxy = createProxy('timeout', async (method) => {
      if (method !== 'loadModelFromUrl') {
        return { value: undefined };
      }
      attempts += 1;
      throw new Error('Worker request timeout');
    });
    const runtime = createRuntime();
    const { bridge, warnings } = createBridge({
      _workerProxy: proxy,
      _createRuntime: () => runtime,
    });
    bridge._replaceWorkerProxyForMultimodalCpuMode = async () => {
      throw new Error('the worker must not be restarted for a request timeout');
    };

    await bridge.loadModelFromUrl('model.gguf', { nGpuLayers: 99 });

    assert.equal(attempts, 1);
    assert.equal(bridge._workerProxy, null);
    assert.equal(proxy.disposeCalls, 1);
    assert.equal(bridge._runtime, runtime);
    assert.equal(runtime.loadCalls.length, 1);
    assert.equal(bridge._workerFallbackReason, 'Worker request timeout');
    assert.ok(runtime._runtimeNotes.includes('worker_fallback:Worker request timeout'));
    assert.ok(!warnings.some((message) => message.includes('FS error detected')));
  }],

  ['worker model-load non-recoverable error is rethrown without fallback', async () => {
    const proxy = createProxy('oom', async (method) => {
      if (method !== 'loadModelFromUrl') {
        return { value: undefined };
      }
      throw new Error('Array buffer allocation failed');
    });
    const { bridge } = createBridge({ _workerProxy: proxy });

    await assert.rejects(
      bridge.loadModelFromUrl('model.gguf', {}),
      /Array buffer allocation failed/,
    );
    assert.equal(bridge._workerProxy, proxy);
    assert.equal(proxy.disposeCalls, 0);
    assert.equal(bridge._runtime, null);
    assert.equal(bridge._workerFallbackReason, null);
  }],

  ['a worker request falls back only when the worker itself is unusable', () => {
    const { bridge } = createBridge();
    const unusable = (error) => bridge._isWorkerUnusableError(error);

    // Deterministic core errors from a healthy worker.
    for (const text of [
      'Generation step failed: Grammar rejected every candidate token',
      'Generation step failed: Sampler returned LLAMA_TOKEN_NULL',
      'Generation step failed: llama_decode failed while generating tokens',
      'Failed to start generation: llama_decode failed while processing prompt',
      'Failed to start generation: Failed to initialize sampler chain (invalid grammar)',
      'Failed to start generation (code=-5)',
      'Failed to start generation: prompt exceeds context size',
      'No model loaded. Call loadModelFromUrl first.',
    ]) {
      assert.equal(unusable(new Error(text)), false, text);
    }

    // The worker or its core can no longer serve requests.
    for (const text of [
      'Aborted(undefined). Build with -sASSERTIONS for more info.',
      'RuntimeError: Aborted(undefined). Build with -sASSERTIONS for more info.',
      'Aborted(native code called abort())',
      'memory access out of bounds',
      'unreachable',
      'Worker request timeout (createCompletion, 5000ms)',
      'Bridge worker init timeout (3000ms)',
      'Bridge worker completion stalled for 5000ms.',
      'Bridge worker crashed',
      'Bridge worker disposed',
      'Failed to initialize bridge worker',
    ]) {
      assert.equal(unusable(new Error(text)), true, text);
    }
    assert.equal(
      unusable(Object.assign(new Error('Uncaught ReferenceError: x is not defined'), {
        llamadartWorkerCrash: true,
      })),
      true,
      'a worker onerror crash counts whatever its text',
    );
    assert.equal(
      unusable(Object.assign(new Error('stalled'), { llamadartWorkerTimeout: true })),
      true,
    );
  }],

  ['worker completion core error is rethrown and keeps the worker', async () => {
    const coreError = 'Generation step failed: Grammar rejected every candidate token';
    let completions = 0;
    const proxy = createProxy('grammar', async (method) => {
      if (method !== 'createCompletion') {
        return { value: undefined };
      }
      completions += 1;
      throw new Error(coreError);
    });
    const runtime = createRuntime({
      async createCompletion() {
        throw new Error('the main-thread runtime must not run the request');
      },
    });
    const { bridge, warnings } = createBridge({
      _workerProxy: proxy,
      _loadedModelUrl: 'model.gguf',
      _createRuntime: () => runtime,
    });
    let fallbackCalls = 0;
    const disableWorkerFallback = bridge._disableWorkerFallback;
    bridge._disableWorkerFallback = (error) => {
      fallbackCalls += 1;
      disableWorkerFallback.call(bridge, error);
    };

    await assert.rejects(
      bridge.createCompletion('prompt', { grammar: 'root ::= "yes" | "no"', nPredict: 4 }),
      (error) => error.message === coreError,
    );

    assert.equal(completions, 1);
    assert.equal(fallbackCalls, 0);
    assert.equal(bridge._workerProxy, proxy);
    assert.equal(proxy.disposeCalls, 0);
    assert.equal(bridge._runtime, null);
    assert.equal(bridge._workerFallbackReason, null);
    assert.equal(runtime.loadCalls.length, 0);
    assert.ok(!warnings.some((message) => message.includes('falling back to main thread')));
  }],

  ['worker invalid-grammar rejection with media parts keeps the worker', async () => {
    const coreError = 'Failed to start generation: Failed to initialize sampler chain '
      + '(invalid grammar): unexpected end of input';
    const proxy = createProxy('grammar-media', async (method) => {
      if (method !== 'createCompletion') {
        return { value: undefined };
      }
      throw new Error(coreError);
    });
    const { bridge, warnings } = createBridge({
      _workerProxy: proxy,
      _multimodalWorkerCpuMode: true,
      _loadedModelUrl: 'model.gguf',
      _loadedModelOptions: { nGpuLayers: 99 },
      _loadedMmProjUrl: 'mmproj.gguf',
      _createRuntime: () => {
        throw new Error('no main-thread runtime may be created');
      },
    });

    await assert.rejects(
      bridge.createCompletion('prompt', {
        grammar: 'root ::= "unterminated',
        parts: [{ type: 'image', bytes: new Uint8Array(1) }],
      }),
      (error) => error.message === coreError,
    );
    assert.equal(bridge._workerProxy, proxy);
    assert.equal(proxy.disposeCalls, 0);
    assert.equal(bridge._runtime, null);
    assert.equal(bridge._workerFallbackReason, null);
    assert.ok(!warnings.some((message) => message.includes('falling back')));
  }],

  ['worker completion core abort still falls back to the main thread', async () => {
    const abortText = 'Aborted(undefined). Build with -sASSERTIONS for more info.';
    let completions = 0;
    const proxy = createProxy('aborted', async (method) => {
      if (method !== 'createCompletion') {
        return { value: undefined };
      }
      completions += 1;
      throw new Error(abortText);
    });
    const runtimeCompletions = [];
    const runtime = createRuntime({
      async createCompletion(prompt) {
        runtimeCompletions.push(prompt);
        return 'main-thread';
      },
    });
    const { bridge } = createBridge({
      _workerProxy: proxy,
      _loadedModelUrl: 'model.gguf',
      _loadedModelOptions: { nGpuLayers: 0 },
      _createRuntime: () => runtime,
    });
    let fallbackCalls = 0;
    const disableWorkerFallback = bridge._disableWorkerFallback;
    bridge._disableWorkerFallback = (error) => {
      fallbackCalls += 1;
      disableWorkerFallback.call(bridge, error);
    };

    assert.equal(await bridge.createCompletion('prompt', { nPredict: 4 }), 'main-thread');

    assert.equal(completions, 1);
    assert.equal(fallbackCalls, 1);
    assert.equal(bridge._workerProxy, null);
    assert.equal(proxy.disposeCalls, 1);
    assert.equal(bridge._runtime, runtime);
    assert.equal(runtime.loadCalls.length, 1, 'the model is reloaded on the main thread');
    assert.deepEqual(runtimeCompletions, ['prompt']);
    assert.equal(bridge._workerFallbackReason, abortText);
  }],

  ['facade worker core errors are rethrown and keep the worker', async () => {
    for (const { method, coreError, invoke } of FACADE_WORKER_METHODS) {
      let dispatches = 0;
      const proxy = createProxy(method, async (called) => {
        if (called !== method) {
          return { value: undefined };
        }
        dispatches += 1;
        throw new Error(coreError);
      });
      const { bridge, warnings } = createBridge({
        _workerProxy: proxy,
        _loadedModelUrl: 'model.gguf',
        _createRuntime: () => {
          throw new Error(`${method}: the main-thread runtime must not be created`);
        },
      });

      await assert.rejects(invoke(bridge), (error) => error.message === coreError, method);

      assert.equal(dispatches, 1, method);
      assert.equal(bridge._workerProxy, proxy, method);
      assert.equal(proxy.disposeCalls, 0, method);
      assert.equal(bridge._runtime, null, method);
      assert.equal(bridge._workerFallbackReason, null, method);
      assert.deepEqual(warnings, [], method);
    }
  }],

  ['facade worker requests fall back when the worker is unusable', async () => {
    const unusableErrors = [
      () => Object.assign(new Error('Uncaught ReferenceError: x is not defined'), {
        llamadartWorkerCrash: true,
      }),
      () => new Error('Worker request timeout (tokenize, 5000ms)'),
      () => new Error('Aborted(undefined). Build with -sASSERTIONS for more info.'),
    ];
    for (const { method, invoke, mainThreadResult } of FACADE_WORKER_METHODS) {
      for (const makeError of unusableErrors) {
        const error = makeError();
        const proxy = createProxy(method, async (called) => {
          if (called !== method) {
            return { value: undefined };
          }
          throw error;
        });
        const runtimeCalls = [];
        const runtime = createRuntime({
          [method]: async (...args) => {
            runtimeCalls.push(args);
            return mainThreadResult;
          },
        });
        const { bridge } = createBridge({
          _workerProxy: proxy,
          _loadedModelUrl: 'model.gguf',
          _loadedModelOptions: { nGpuLayers: 0 },
          _createRuntime: () => runtime,
        });
        const label = `${method}: ${error.message}`;

        assert.deepEqual(await invoke(bridge), mainThreadResult, label);

        assert.equal(runtimeCalls.length, 1, label);
        assert.equal(bridge._workerProxy, null, label);
        assert.equal(proxy.disposeCalls, 1, label);
        assert.equal(bridge._runtime, runtime, label);
        assert.equal(bridge._workerFallbackReason, error.message, label);
      }
    }
  }],

  ['real worker proxy: posted facade core errors keep the worker', async () => {
    await withStubWorkerEnvironment(async ({ workers }) => {
      const bridge = createRealWorkerBridge();
      const warnings = [];
      bridge._emitBridgeWarn = (message) => warnings.push(message);
      bridge._createRuntime = () => {
        throw new Error('the main-thread runtime must not be created');
      };
      const proxy = bridge._workerProxy;
      const driver = workerDriver(proxy, workers[0]);
      driver.ready();

      for (const { method, coreError, invoke } of FACADE_WORKER_METHODS) {
        const pending = settleQuietly(invoke(bridge));
        await driver.settle();
        const calls = driver.calls(method);
        driver.error(calls[calls.length - 1].id, coreError);
        await assert.rejects(pending, (error) => error.message === coreError, method);
      }

      assert.equal(bridge._workerProxy, proxy);
      assert.equal(workers[0].terminated, 0);
      assert.equal(bridge._workerFallbackReason, null);
      assert.deepEqual(warnings, []);

      const next = settleQuietly(bridge.tokenize('ok'));
      await driver.settle();
      driver.reply(driver.calls('tokenize')[1].id, [7]);
      assert.deepEqual(await next, [7], 'the kept worker serves the next request');
    });
  }],

  ['real worker proxy: a posted core error is rethrown and the worker is kept', async () => {
    await withStubWorkerEnvironment(async ({ workers }) => {
      const bridge = createRealWorkerBridge();
      const warnings = [];
      bridge._emitBridgeWarn = (message) => warnings.push(message);
      bridge._createRuntime = () => {
        throw new Error('the main-thread runtime must not be created');
      };
      const proxy = bridge._workerProxy;
      const driver = workerDriver(proxy, workers[0]);
      driver.ready();

      const completion = settleQuietly(bridge.createCompletion('alpha', { nPredict: 4 }));
      await driver.settle();
      const [call] = driver.calls('createCompletion');
      driver.error(call.id, 'Failed to start generation: llama_decode failed while processing prompt');

      await assert.rejects(completion, /llama_decode failed while processing prompt/);
      assert.equal(bridge._workerProxy, proxy);
      assert.equal(workers[0].terminated, 0);
      assert.equal(bridge._workerFallbackReason, null);
      assert.deepEqual(warnings, []);

      const next = settleQuietly(bridge.createCompletion('beta', { nPredict: 4 }));
      await driver.settle();
      driver.reply(driver.calls('createCompletion')[1].id, 'BETA');
      assert.equal(await next, 'BETA', 'the kept worker serves the next request');
    });
  }],

  ['real worker proxy: an onerror crash falls back whatever its text', async () => {
    await withStubWorkerEnvironment(async ({ workers }) => {
      const bridge = createRealWorkerBridge();
      const runtime = createRuntime({
        async createCompletion(prompt) {
          return `main:${prompt}`;
        },
      });
      bridge._emitBridgeWarn = () => {};
      bridge._createRuntime = () => runtime;
      bridge._loadedModelUrl = 'model.gguf';
      bridge._loadedModelOptions = { nGpuLayers: 0 };
      const proxy = bridge._workerProxy;
      const driver = workerDriver(proxy, workers[0]);
      driver.ready();

      const completion = settleQuietly(bridge.createCompletion('alpha', { nPredict: 4 }));
      await driver.settle();
      proxy._worker.onerror({ message: 'Uncaught ReferenceError: x is not defined' });
      // Retiring the crashed proxy sends it a best-effort dispose; answer it.
      for (let i = 0; i < 20 && driver.calls('dispose').length === 0; i += 1) {
        await driver.settle();
      }
      const [disposeCall] = driver.calls('dispose');
      assert.ok(disposeCall, 'the crashed worker is retired');
      driver.reply(disposeCall.id, null);

      assert.equal(await completion, 'main:alpha');
      assert.equal(bridge._workerProxy, null);
      assert.equal(workers[0].terminated, 1);
      assert.equal(bridge._runtime, runtime);
      assert.equal(bridge._workerFallbackReason, 'Uncaught ReferenceError: x is not defined');
    });
  }],

  ['fallback reason is always the serialized error, including for flagged errors', () => {
    const flagged = Object.assign(new Error('Bridge worker crashed'), {
      llamadartForceCpuMultimodal: true,
    });
    for (const error of [new Error('Bridge worker crashed'), flagged, 'plain string failure']) {
      const proxy = createProxy('fallback', async () => ({ value: undefined }));
      const runtime = createRuntime();
      const { bridge, warnings } = createBridge({
        _workerProxy: proxy,
        _createRuntime: () => runtime,
      });

      bridge._disableWorkerFallback(error);

      const expected = typeof error === 'string' ? error : error.message;
      assert.equal(bridge._workerFallbackReason, expected);
      assert.equal(globalThis.__llamadartBridgeWorkerFallbackReason, expected);
      assert.equal(bridge._workerProxy, null);
      assert.equal(proxy.disposeCalls, 1);
      assert.equal(bridge._runtime, runtime);
      assert.deepEqual(runtime._runtimeNotes, [`worker_fallback:${expected}`]);
      assert.deepEqual(warnings, [
        `llamadart: bridge worker unavailable, falling back to main thread (${expected})`,
      ]);
    }
  }],

  ['multimodal recovery selects CPU-safe reload only for timeout and WebGPU classifications', async () => {
    const run = async (error) => {
      const runtime = createRuntime();
      const { bridge, warnings } = createBridge({
        _runtime: runtime,
        _loadedModelUrl: 'model.gguf',
        _loadedModelOptions: { nGpuLayers: 99, nCtx: 8192, nThreads: 8 },
        _loadedMmProjUrl: 'mmproj.gguf',
      });
      await bridge._ensureRuntimeReadyAfterWorkerFallback(
        { parts: [{ type: 'image', bytes: new Uint8Array(1) }] },
        error,
      );
      assert.equal(runtime.loadCalls.length, 1);
      assert.deepEqual(runtime.projectorCalls, ['mmproj.gguf']);
      return { options: runtime.loadCalls[0].options, notes: runtime._runtimeNotes, warnings };
    };

    const stalled = await run(new Error('worker timed out'));
    assert.equal(stalled.options.nGpuLayers, 0);
    assert.equal(stalled.options.nCtx, 4096);
    assert.ok(stalled.notes.includes('worker_fallback_timeout'));
    assert.ok(stalled.notes.includes('worker_fallback_cpu_multimodal'));
    assert.ok(stalled.warnings.some((message) => message.includes('after worker timeout')));

    const flaggedTimeout = Object.assign(new Error('stalled'), { llamadartWorkerTimeout: true });
    const flagged = await run(flaggedTimeout);
    assert.equal(flagged.options.nGpuLayers, 0);
    assert.ok(flagged.notes.includes('worker_fallback_timeout'));

    const webgpu = await run(new Error('Aborted()'));
    assert.equal(webgpu.options.nGpuLayers, 0);
    assert.ok(webgpu.notes.includes('worker_fallback_cpu_multimodal'));
    assert.ok(!webgpu.notes.includes('worker_fallback_timeout'));
    assert.ok(webgpu.warnings.some((message) => message.includes('workgroup limit failure')));

    const requestTimeout = await run(new Error('Worker request timeout'));
    assert.equal(requestTimeout.options.nGpuLayers, 99);
    assert.equal(requestTimeout.options.nCtx, 8192);
    assert.ok(!requestTimeout.notes.includes('worker_fallback_cpu_multimodal'));
    assert.ok(!requestTimeout.notes.includes('worker_fallback_timeout'));
    assert.ok(!requestTimeout.warnings.some((message) => message.includes('CPU fallback')));

    const crash = await run(new Error('Bridge worker crashed'));
    assert.equal(crash.options.nGpuLayers, 99);
    assert.ok(!crash.notes.includes('worker_fallback_cpu_multimodal'));
    assert.ok(!crash.warnings.some((message) => message.includes('CPU fallback')));

    const flaggedCrash = await run(Object.assign(new Error('Bridge worker crashed'), {
      llamadartForceCpuMultimodal: true,
    }));
    assert.equal(flaggedCrash.options.nGpuLayers, 99);
    assert.ok(!flaggedCrash.notes.includes('worker_fallback_cpu_multimodal'));
    assert.ok(!flaggedCrash.warnings.some((message) => message.includes('CPU fallback')));
  }],

  ['failed main-thread recovery leaves a usable direct runtime for the next load', async () => {
    const proxy = createProxy('crashed', async (method) => {
      if (method !== 'tokenize') {
        return { value: undefined };
      }
      throw new Error('Bridge worker crashed');
    });
    const created = [];
    const createTrackedRuntime = () => {
      const index = created.length;
      const runtime = createRuntime({
        disposeCalls: 0,
        async loadModelFromUrl(url, options) {
          runtime.loadCalls.push({ url, options: { ...options } });
          if (index === 0) {
            throw new Error('recovery reload failed');
          }
          runtime._modelBytes = 1;
        },
        async dispose() {
          runtime.disposeCalls += 1;
        },
        async tokenize(text) {
          if (runtime._modelBytes <= 0) {
            throw new Error('No model loaded. Call loadModelFromUrl first.');
          }
          return [text.length];
        },
        async createCompletion(prompt) {
          if (runtime._modelBytes <= 0) {
            throw new Error('No model loaded. Call loadModelFromUrl first.');
          }
          return `main:${prompt}`;
        },
      });
      created.push(runtime);
      return runtime;
    };
    const { bridge } = createBridge({
      _workerProxy: proxy,
      _loadedModelUrl: 'model.gguf',
      _loadedModelOptions: { nGpuLayers: 0 },
      _loadedMmProjUrl: 'mmproj.gguf',
      _createRuntime: createTrackedRuntime,
    });

    await assert.rejects(bridge.tokenize('alpha'), /recovery reload failed/);

    const [failed] = created;
    assert.equal(bridge._workerProxy, null, 'the crashed worker stays retired');
    assert.equal(proxy.disposeCalls, 1);
    assert.equal(failed.disposeCalls, 1, 'the half-recovered runtime is torn down');
    assert.notEqual(bridge._runtime, failed);
    assert.equal(bridge._loadedModelUrl, null, 'the unrecovered model is forgotten');
    assert.equal(bridge._loadedModelOptions, null);
    assert.equal(bridge._loadedMmProjUrl, null);

    await assert.rejects(
      bridge.tokenize('alpha'),
      (error) => !(error instanceof TypeError) && /No model loaded/.test(error.message),
      'a call before a new load fails clearly instead of dereferencing a null runtime',
    );
    await assert.rejects(
      bridge.createCompletion('describe', { parts: [{ type: 'image', bytes: new Uint8Array(1) }] }),
      /No model loaded/,
      'a media completion does not silently retry the forgotten model',
    );
    assert.equal(created.length, 2, 'the forgotten model is not reloaded');

    await bridge.loadModelFromUrl('good.gguf', { nGpuLayers: 0 });
    const current = bridge._runtime;
    assert.ok(current && current !== failed, 'the load runs on a fresh direct runtime');
    assert.deepEqual(current.loadCalls.map((call) => call.url), ['good.gguf']);
    assert.equal(bridge._loadedModelUrl, 'good.gguf');
    assert.deepEqual(await bridge.tokenize('beta'), [4]);

    const createdBeforeDispose = created.length;
    await bridge.dispose();
    assert.equal(bridge._runtime, null);
    assert.equal(current.disposeCalls, 1);
    await assert.rejects(bridge.tokenize('gamma'), /Bridge has been disposed/);
    await assert.rejects(bridge.loadModelFromUrl('good.gguf', {}), /Bridge has been disposed/);
    await assert.rejects(bridge.prefetchModelToCache('good.gguf'), /Bridge has been disposed/);
    await assert.rejects(
      bridge._ensureRuntimeReadyAfterWorkerFallback({}, null),
      /Bridge has been disposed/,
    );
    assert.equal(bridge._runtime, null);
    assert.equal(created.length, createdBeforeDispose, 'disposal never recreates a runtime');
  }],

  ['failed recovery does not recreate a runtime once disposal has begun', async () => {
    const proxy = createProxy('crashed', async (method) => {
      if (method !== 'tokenize') {
        return { value: undefined };
      }
      throw new Error('Bridge worker crashed');
    });
    let disposal = null;
    let created = 0;
    const { bridge } = createBridge({
      _workerProxy: proxy,
      _loadedModelUrl: 'model.gguf',
      _loadedModelOptions: { nGpuLayers: 0 },
      _createRuntime: () => {
        created += 1;
        return createRuntime({
          async loadModelFromUrl() {
            throw new Error('recovery reload failed');
          },
          async dispose() {
            // The application disposes the bridge while the failed runtime is
            // still being torn down.
            disposal ??= bridge.dispose();
          },
        });
      },
    });

    await assert.rejects(bridge.tokenize('alpha'));
    assert.ok(disposal, 'disposal began during recovery cleanup');
    await disposal;

    assert.equal(created, 1, 'no replacement runtime is created during disposal');
    assert.equal(bridge._runtime, null);
    assert.equal(bridge._lifecycleState, 'disposed');
  }],

  ['bridge and runtime log gates share one level table', () => {
    const levels = ['debug', 'log', 'info', 'warn', 'error', 'trace'];
    const expected = {
      0: { debug: false, log: false, info: false, warn: false, error: false, trace: false },
      1: { debug: true, log: true, info: true, warn: true, error: true, trace: true },
      2: { debug: false, log: true, info: true, warn: true, error: true, trace: true },
      3: { debug: false, log: false, info: false, warn: true, error: true, trace: false },
      4: { debug: false, log: false, info: false, warn: false, error: true, trace: false },
    };

    const realRuntime = (bridge) => LlamaWebGpuBridge.prototype._createRuntime.call(bridge);

    for (const configured of [0, 1, 2, 3, 4]) {
      const { bridge } = createBridge({ _config: { logLevel: configured } });
      const runtime = realRuntime(bridge);
      runtime._logLevel = configured;
      for (const level of levels) {
        assert.equal(
          bridge._shouldEmitBridgeLevel(level),
          expected[configured][level],
          `bridge logLevel=${configured} ${level}`,
        );
        assert.equal(
          runtime._shouldEmitLoggerLevel(level),
          expected[configured][level],
          `runtime logLevel=${configured} ${level}`,
        );
      }
    }

    const { bridge: unconfigured } = createBridge({ _config: {}, _runtime: null });
    assert.equal(unconfigured._shouldEmitBridgeLevel('debug'), false);
    assert.equal(unconfigured._shouldEmitBridgeLevel('info'), true);

    const runtime = realRuntime(unconfigured);
    runtime._logLevel = -1;
    assert.equal(runtime._shouldEmitLoggerLevel('debug'), true);
    runtime._logLevel = 9;
    assert.equal(runtime._shouldEmitLoggerLevel('debug'), false);
    assert.equal(runtime._shouldEmitLoggerLevel('error'), true);
  }],
];

const failures = [];
const previousGlobalReason = globalThis.__llamadartBridgeWorkerFallbackReason;
try {
  for (const [name, run] of CASES) {
    try {
      await run();
    } catch (error) {
      failures.push(`${name}: ${error?.message || error}`);
    }
  }
} finally {
  globalThis.__llamadartBridgeWorkerFallbackReason = previousGlobalReason;
}

if (failures.length > 0) {
  console.error(`${failures.length}/${CASES.length} bridge worker error classification cases failed:`);
  for (const failure of failures) {
    console.error(`  - ${failure}`);
  }
  process.exitCode = 1;
} else {
  console.log(`Bridge worker error classification tests passed (${CASES.length} cases)`);
}
