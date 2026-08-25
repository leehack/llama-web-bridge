import assert from 'node:assert/strict';

import { LlamaWebGpuBridge } from '../js/src/llama_webgpu_bridge.js';

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

function createProxy(name, { disposePromise = null, callImpl = null } = {}) {
  const proxy = {
    name,
    calls: [],
    pending: [],
    async call(method, args, onEvent) {
      proxy.calls.push({ method, args, onEvent });
      if (callImpl) {
        return callImpl(method, args, onEvent, proxy);
      }
      if (method === 'cancel' || method === 'setLogLevel') {
        return { value: undefined };
      }
      return new Promise((resolve, reject) => {
        proxy.pending.push({ method, args, onEvent, resolve, reject });
      });
    },
    dispose() {
      proxy.disposeCalls = (proxy.disposeCalls || 0) + 1;
      return disposePromise || Promise.resolve();
    },
  };
  return proxy;
}

function createBridge(overrides = {}) {
  const bridge = Object.create(LlamaWebGpuBridge.prototype);
  Object.assign(bridge, {
    _config: {},
    _runtime: null,
    _workerProxy: null,
    _workerGeneration: 0,
    _workerDisposePromise: null,
    _retiringWorkerDisposals: new Set(),
    _workerFallbackReason: null,
    _metadata: { lifecycle: 'old' },
    _contextSize: 128,
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
    _shadowStateTransactionDepth: 0,
    _deferredShadowState: null,
    _disposed: false,
    _disposePromise: null,
    _disposalWaiters: new Set(),
    _emitBridgeWarn: () => {},
    _createRuntime: () => ({
      _modelBytes: 0,
      _runtimeNotes: [],
      async loadModelFromUrl() {
        this._modelBytes = 1;
      },
      async dispose() {},
      supportsVision: () => false,
      supportsAudio: () => false,
    }),
    ...overrides,
  });
  return bridge;
}

async function expectAbort(promise) {
  await assert.rejects(promise, (error) => error?.name === 'AbortError');
}

const CASES = [
  ['operation ids are monotonic and failures reach a failed terminal state', async () => {
    const bridge = createBridge();
    let failedOperation;
    let succeedingOperation;

    await assert.rejects(
      bridge._runExclusive(async (operation) => {
        failedOperation = operation;
        throw new Error('expected lifecycle failure');
      }, { kind: 'generation' }),
      /expected lifecycle failure/,
    );

    await bridge._runExclusive(async (operation) => {
      succeedingOperation = operation;
    }, { kind: 'tokenize' });

    assert.equal(failedOperation.id, 1);
    assert.equal(failedOperation.state, 'failed');
    assert.equal(succeedingOperation.id, 2);
    assert.equal(succeedingOperation.state, 'completed');
  }],

  ['active operation owns a captured worker for cancellation', async () => {
    const oldProxy = createProxy('generation-7');
    const replacement = createProxy('generation-8');
    const bridge = createBridge({
      _workerProxy: oldProxy,
      _workerGeneration: 7,
    });
    let release;
    const operation = bridge._runExclusive(
      () => new Promise((resolve) => {
        release = resolve;
      }),
    );

    await tick();
    assert.equal(bridge._activeOperation.id, 1);
    assert.equal(bridge._activeOperation.workerGeneration, 7);
    assert.equal(bridge._activeOperation.workerProxy, oldProxy);

    bridge._workerProxy = replacement;
    bridge._workerGeneration = 8;
    bridge.cancel();
    await tick();

    assert.deepEqual(oldProxy.calls.map(({ method }) => method), ['cancel']);
    assert.deepEqual(replacement.calls, [], 'cancellation must not target only the replacement');

    release();
    await operation;
    assert.equal(bridge._activeOperation, null);
  }],

  ['stale worker events are suppressed after replacement', async () => {
    let resolveRequest;
    const oldProxy = createProxy('generation-1', {
      callImpl: (method, args, onEvent, proxy) => {
        proxy.pending.push({ method, args, onEvent, resolve: resolveRequest });
        return new Promise((resolve) => {
          resolveRequest = resolve;
          proxy.pending.at(-1).resolve = resolve;
        });
      },
    });
    const replacement = createProxy('generation-2');
    const bridge = createBridge({
      _workerProxy: oldProxy,
      _workerGeneration: 1,
      _metadata: { current: 'old' },
    });
    const events = [];

    const pending = bridge._runExclusive(() => bridge._callWorker(
      'createCompletion',
      ['hello'],
      (event) => events.push(event),
    ));
    await tick();
    bridge._workerProxy = replacement;
    bridge._workerGeneration = 2;
    bridge._metadata = { current: 'new' };

    oldProxy.calls[0].onEvent({ event: 'token', payload: { pieceText: 'stale' } });
    resolveRequest({
      value: 'response',
      state: {
        metadata: { current: 'stale' },
        contextSize: 999,
        supportsVision: true,
      },
    });

    assert.equal(await pending, 'response');
    assert.deepEqual(events, [], 'late token events must not reach the current callback');
    assert.deepEqual(bridge._metadata, { current: 'new' });
    assert.equal(bridge._contextSize, 128);
    assert.equal(bridge._supportsVision, false);
  }],

  ['stale worker state cannot enter an atomic recovery transaction', async () => {
    let resolveRequest;
    const oldProxy = createProxy('generation-1', {
      callImpl: () => new Promise((resolve) => {
        resolveRequest = resolve;
      }),
    });
    const replacement = createProxy('generation-2');
    const bridge = createBridge({
      _workerProxy: oldProxy,
      _workerGeneration: 1,
      _shadowStateTransactionDepth: 1,
      _deferredShadowState: null,
    });

    const pending = bridge._runExclusive(
      () => bridge._callWorker('loadMultimodalProjector', ['projector.gguf']),
      { kind: 'projector-load' },
    );
    await tick();
    bridge._workerProxy = replacement;
    bridge._workerGeneration = 2;
    resolveRequest({
      value: 1,
      state: { metadata: { recovery: 'stale-projector' }, supportsVision: true },
    });

    assert.equal(await pending, 1);
    assert.equal(
      bridge._deferredShadowState,
      null,
      'a retired proxy must not populate the replacement recovery transaction',
    );
  }],

  ['all superseded worker teardowns remain pending until disposal', async () => {
    let releaseFirst;
    let releaseSecond;
    const firstTeardown = new Promise((resolve) => { releaseFirst = resolve; });
    const secondTeardown = new Promise((resolve) => { releaseSecond = resolve; });
    const first = createProxy('generation-1', { disposePromise: firstTeardown });
    const second = createProxy('generation-2', { disposePromise: secondTeardown });
    const bridge = createBridge({ _workerProxy: first });

    bridge._disableWorkerFallback(new Error('first worker died'));
    bridge._workerProxy = second;
    bridge._disableWorkerFallback(new Error('second worker died'));

    const disposal = bridge.dispose();
    let resolved = false;
    disposal.then(() => { resolved = true; });
    await tick();
    assert.equal(resolved, false);
    assert.equal(bridge._retiringWorkerDisposals.size, 2);

    releaseSecond();
    await tick();
    assert.equal(resolved, false, 'a later generation must not hide an earlier retirement');

    releaseFirst();
    await disposal;
    assert.equal(resolved, true);
    assert.equal(bridge._retiringWorkerDisposals.size, 0);
  }],

  ['split-model sources are replayed as arrays during direct fallback', async () => {
    const calls = [];
    const runtime = {
      _modelBytes: 0,
      _runtimeNotes: [],
      async loadModelFromUrl(source, options) {
        calls.push({ source, options });
        this._modelBytes = 1;
      },
      supportsVision: () => false,
      supportsAudio: () => false,
      async dispose() {},
    };
    const bridge = createBridge({ _runtime: runtime, _workerProxy: null });
    bridge._rememberLoadedModel(
      ['model-00001.gguf', 'model-00002.gguf'],
      { nGpuLayers: 0, progressCallback: () => {}, signal: new AbortController().signal },
    );

    await bridge._ensureRuntimeReadyAfterWorkerFallback({}, new Error('worker request timeout'));

    assert.deepEqual(calls[0].source, ['model-00001.gguf', 'model-00002.gguf']);
    assert.equal(typeof calls[0].source, 'object');
    assert.equal(calls[0].options.nGpuLayers, 0);
    assert.equal('signal' in calls[0].options, false);
  }],

  ['worker model load strips AbortSignal and cancellation targets its owner', async () => {
    let releaseLoad;
    const proxy = createProxy('generation-3', {
      callImpl: (method, args, _onEvent, target) => {
        target.calls.at(-1).args = args;
        if (method === 'loadModelFromUrl') {
          return new Promise((resolve) => { releaseLoad = resolve; });
        }
        if (method === 'cancel') {
          return Promise.resolve({ value: undefined });
        }
        return Promise.resolve({ value: undefined });
      },
    });
    const bridge = createBridge({ _workerProxy: proxy, _workerGeneration: 3 });
    const controller = new AbortController();
    const pending = bridge._runExclusive(
      () => bridge._loadModelFromUrlUnlocked('model.gguf', { signal: controller.signal }),
      { signal: controller.signal, abortMessage: 'Model load was cancelled.' },
    );

    await tick();
    const loadCall = proxy.calls.find(({ method }) => method === 'loadModelFromUrl');
    assert.ok(loadCall);
    assert.equal('signal' in loadCall.args[1], false);

    controller.abort();
    await tick();
    assert.deepEqual(proxy.calls.map(({ method }) => method), ['loadModelFromUrl', 'cancel']);

    releaseLoad({ value: 1, state: {} });
    await expectAbort(pending);
  }],

  ['a cancelled worker model load rejects even when the worker settles successfully', async () => {
    let releaseLoad;
    const proxy = createProxy('generation-3', {
      callImpl: (method) => {
        if (method === 'loadModelFromUrl') {
          return new Promise((resolve) => { releaseLoad = resolve; });
        }
        return Promise.resolve({ value: undefined });
      },
    });
    const bridge = createBridge({ _workerProxy: proxy, _workerGeneration: 3 });
    const controller = new AbortController();
    const pending = bridge.loadModelFromUrl('model.gguf', { signal: controller.signal });

    await tick();
    controller.abort();
    await tick();

    releaseLoad({
      value: 1,
      state: { metadata: { model: 'late-success' }, contextSize: 4096, supportsVision: true },
    });

    await expectAbort(pending);
    assert.equal(
      bridge._loadedModelUrl,
      null,
      'a cancelled load must not commit a model source',
    );
    assert.equal(bridge._loadedModelOptions, null);
    assert.deepEqual(
      bridge._metadata,
      { lifecycle: 'old' },
      'a cancelled load must not publish late worker state',
    );
    assert.equal(bridge._contextSize, 128);
    assert.equal(bridge._supportsVision, false);
  }],

  ['a cancelled worker completion rejects even when the worker settles successfully', async () => {
    let releaseCompletion;
    const proxy = createProxy('generation-3', {
      callImpl: (method) => {
        if (method === 'createCompletion') {
          return new Promise((resolve) => { releaseCompletion = resolve; });
        }
        return Promise.resolve({ value: undefined });
      },
    });
    const bridge = createBridge({
      _workerProxy: proxy,
      _workerGeneration: 3,
      _disableWorkerFallback: () => {
        throw new Error('a cancelled completion must not enter worker fallback');
      },
    });
    const controller = new AbortController();
    const pending = bridge.createCompletion('hello', { signal: controller.signal });

    await tick();
    controller.abort();
    await tick();

    releaseCompletion({
      value: 'late generated text',
      state: { metadata: { model: 'late-success' } },
    });

    await expectAbort(pending);
    assert.deepEqual(
      bridge._metadata,
      { lifecycle: 'old' },
      'a cancelled completion must not publish late worker state',
    );
  }],

  ['a cancelled worker synthesis rejects even when the worker settles successfully', async () => {
    let releaseSynthesis;
    const proxy = createProxy('generation-3', {
      callImpl: (method) => {
        if (method === 'synthesizeSpeech') {
          return new Promise((resolve) => { releaseSynthesis = resolve; });
        }
        return Promise.resolve({ value: undefined });
      },
    });
    const bridge = createBridge({
      _workerProxy: proxy,
      _workerGeneration: 3,
      _disableWorkerFallback: () => {
        throw new Error('a cancelled synthesis must not enter worker fallback');
      },
    });
    const controller = new AbortController();
    const pending = bridge.synthesizeSpeech({ text: 'hello', signal: controller.signal });

    await tick();
    controller.abort();
    await tick();

    releaseSynthesis({
      value: { pcm: new Float32Array([0.5]), sampleRate: 24000 },
      state: { metadata: { model: 'late-success' } },
    });

    await expectAbort(pending);
    assert.deepEqual(
      bridge._metadata,
      { lifecycle: 'old' },
      'a cancelled synthesis must not publish late worker state',
    );
  }],

  ['direct transfer aborts before native model entry and removes partial files', async () => {
    const bridge = new LlamaWebGpuBridge({ disableWorker: true });
    const runtime = bridge._runtime;
    const files = new Map();
    let nativeLoadCalls = 0;
    let shutdownCalls = 0;
    const controller = new AbortController();
    const core = {
      FS: {
        analyzePath: (filePath) => ({ exists: files.has(filePath) }),
        mkdir: () => {},
        open: (filePath) => ({ filePath }),
        write: (stream, bytes) => { files.set(stream.filePath, new Uint8Array(bytes)); },
        close: () => {},
        unlink: (filePath) => { files.delete(filePath); },
      },
      async ccall(name) {
        if (name === 'llamadart_webgpu_load_model') {
          nativeLoadCalls += 1;
          return 0;
        }
        if (name === 'llamadart_webgpu_shutdown') {
          shutdownCalls += 1;
          return undefined;
        }
        if (name === 'llamadart_webgpu_get_context_size') {
          return 128;
        }
        if (name === 'llamadart_webgpu_supports_pthreads') {
          return 0;
        }
        return 0;
      },
    };
    let readCount = 0;
    runtime._core = core;
    runtime._coreVariant = 'wasm32';
    runtime._probeBackends = async () => false;
    runtime._ensureCore = async () => core;
    runtime._coreSupportsPthreads = () => false;
    runtime._syncThreadPoolSizeHintFromCore = () => {};
    runtime._resolveNativeLoadOptions = () => {};
    runtime._releaseModelFiles = () => 0;
    runtime._emitSuppressedWarmupWarningSummaryIfNeeded = () => {};
    runtime._getCachedModelResponse = async () => ({
      ok: true,
      status: 200,
      statusText: 'OK',
      headers: new Headers({ 'content-length': '2' }),
      body: {
        getReader() {
          return {
            async read() {
              if (readCount++ === 0) {
                controller.abort();
                return { done: false, value: new Uint8Array([1, 2]) };
              }
              return { done: true, value: undefined };
            },
            async cancel() {},
          };
        },
      },
    });

    await expectAbort(runtime.loadModelFromUrl('model.gguf', {
      signal: controller.signal,
      nGpuLayers: 0,
      useCache: false,
    }));

    assert.equal(nativeLoadCalls, 0, 'abort must win before the native model ccall');
    assert.equal(shutdownCalls, 0, 'native shutdown is unnecessary when entry was skipped');
    assert.equal(files.size, 0, 'partial model files must be removed on abort');
  }],

  ['projector recovery does not publish a partial base-model snapshot', async () => {
    const proxy = createProxy('generation-5', {
      callImpl: (method) => {
        if (method === 'loadModelFromUrl') {
          return Promise.resolve({
            value: 1,
            state: {
              metadata: { recovery: 'base-only' },
              contextSize: 4096,
              supportsVision: false,
              supportsAudio: false,
            },
          });
        }
        if (method === 'loadMultimodalProjector') {
          return Promise.reject(new Error('projector failed'));
        }
        return Promise.resolve({ value: undefined, state: {} });
      },
    });
    const bridge = createBridge({
      _workerProxy: proxy,
      _workerGeneration: 5,
      _loadedModelUrl: 'model.gguf',
      _loadedModelOptions: { nGpuLayers: 1 },
      _loadedMmProjUrl: 'projector.gguf',
      _metadata: { recovery: 'previous-atomic-state' },
      _contextSize: 2048,
    });
    bridge._replaceWorkerProxyForMultimodalCpuMode = async () => {
      bridge._workerProxy = proxy;
    };

    await assert.rejects(
      bridge._ensureWorkerMultimodalCpuMode(),
      /projector failed/,
    );
    assert.deepEqual(
      bridge._metadata,
      { recovery: 'previous-atomic-state' },
      'base-model state must not become visible before projector recovery commits',
    );
    assert.equal(bridge._contextSize, 2048);
  }],

  ['dispose during recovery prevents a new worker or runtime from being installed', async () => {
    let runtimeCreations = 0;
    let workerCreations = 0;
    const bridge = createBridge({
      _createRuntime: () => {
        runtimeCreations += 1;
        return {
          _modelBytes: 0,
          _runtimeNotes: [],
          async loadModelFromUrl() {},
          async dispose() {},
        };
      },
      _replaceWorkerProxyForMultimodalCpuMode: LlamaWebGpuBridge.prototype
        ._replaceWorkerProxyForMultimodalCpuMode,
      _workerModuleUrl: () => 'worker.js',
      _workerConfig: () => ({}),
    });
    bridge._replaceWorkerProxyForMultimodalCpuMode = async function replaceAfterDispose() {
      if (this._disposed) {
        throw new Error('Bridge has been disposed.');
      }
      workerCreations += 1;
    };

    await bridge.dispose();
    await assert.rejects(
      bridge._replaceWorkerProxyForMultimodalCpuMode(),
      /Bridge has been disposed/,
    );
    assert.equal(runtimeCreations, 0);
    assert.equal(workerCreations, 0);
    assert.equal(bridge._workerProxy, null);
    assert.equal(bridge._runtime, null);
  }],

  ['operation identity survives worker retirement and records the next generation', async () => {
    const retiring = createProxy('generation-7');
    const bridge = createBridge({
      _workerProxy: retiring,
      _workerGeneration: 7,
      _createRuntime: () => ({
        _modelBytes: 0,
        _runtimeNotes: [],
        async loadModelFromUrl() {},
        async dispose() {},
        getModelMetadata: () => ({}),
        getContextSize: () => 128,
        isGpuActive: () => false,
        getBackendName: () => 'WASM (Prototype bridge)',
        supportsVision: () => false,
        supportsAudio: () => false,
      }),
    });
    const trace = [];

    await bridge._runExclusive(async (operation) => {
      trace.push({ id: operation.id, generation: operation.workerGeneration });
      bridge._disableWorkerFallback(new Error('worker request timeout'));
      trace.push({
        id: operation.id,
        generation: operation.workerGeneration,
        currentGeneration: bridge._workerGeneration,
      });
    }, { kind: 'generation' });

    assert.equal(trace[0].id, trace[1].id);
    assert.equal(trace[0].generation, 7);
    assert.equal(trace[1].generation, 8);
    assert.equal(trace[1].currentGeneration, 8);
  }],

  ['worker timeout cancels only the owning worker and keeps recovery live', async () => {
    const proxy = createProxy('generation-4');
    const loadCalls = [];
    const runtime = {
      _modelBytes: 0,
      _runtimeNotes: [],
      cancelCalls: 0,
      cancel() {
        this.cancelCalls += 1;
      },
      async loadModelFromUrl(source, options) {
        loadCalls.push({ source, options });
        this._modelBytes = 1;
      },
      async dispose() {},
      getModelMetadata: () => ({}),
      getContextSize: () => 128,
      isGpuActive: () => false,
      getBackendName: () => 'WASM (Prototype bridge)',
      supportsVision: () => false,
      supportsAudio: () => false,
    };
    const bridge = createBridge({
      _workerProxy: proxy,
      _workerGeneration: 4,
      _runtime: runtime,
      _loadedModelUrl: 'model.gguf',
      _loadedModelOptions: { nGpuLayers: 0 },
    });
    let operation;

    await bridge._runExclusive(async (activeOperation) => {
      operation = activeOperation;
      bridge._cancelOperation(activeOperation, 'worker-timeout');
      assert.equal(activeOperation.cancelRequested, false);
      assert.equal(activeOperation.abortController.signal.aborted, false);
      assert.equal(activeOperation.state, 'recovering');

      bridge._workerProxy = null;
      const timeoutError = new Error('worker timeout');
      timeoutError.llamadartWorkerTimeout = true;
      await bridge._ensureRuntimeReadyAfterWorkerFallback({}, timeoutError);
    }, { kind: 'generation' });

    assert.deepEqual(proxy.calls.map(({ method }) => method), ['cancel']);
    assert.equal(runtime.cancelCalls, 0);
    assert.equal(loadCalls.length, 1);
    assert.equal(loadCalls[0].options.signal.aborted, false);
    assert.equal(operation.state, 'completed');
  }],

  ['user cancellation does not enter worker fallback recovery', async () => {
    let rejectCompletion;
    let fallbackCalls = 0;
    const proxy = createProxy('generation-3', {
      callImpl: (method) => {
        if (method === 'createCompletion') {
          return new Promise((_, reject) => {
            rejectCompletion = reject;
          });
        }
        return Promise.resolve({ value: undefined });
      },
    });
    const bridge = createBridge({
      _workerProxy: proxy,
      _workerGeneration: 3,
      _disableWorkerFallback: () => {
        fallbackCalls += 1;
      },
    });
    const controller = new AbortController();
    const pending = bridge.createCompletion('hello', { signal: controller.signal });

    await tick();
    controller.abort();
    await tick();
    rejectCompletion(new Error('worker request cancelled'));

    await expectAbort(pending);
    assert.equal(fallbackCalls, 0);
  }],

  ['cancel during recovery leaves the queued successor dispatchable', async () => {
    const owned = createProxy('generation-1');
    const bridge = createBridge({
      _workerProxy: owned,
      _workerGeneration: 1,
      _createRuntime: () => ({
        _modelBytes: 0,
        _runtimeNotes: [],
        async loadModelFromUrl() {},
        async dispose() {},
        getModelMetadata: () => ({}),
        getContextSize: () => 128,
        isGpuActive: () => false,
        getBackendName: () => 'WASM (Prototype bridge)',
        supportsVision: () => false,
        supportsAudio: () => false,
      }),
    });
    let releaseActive;
    let activeOperation;
    let successorOperation;
    let successorStartedWhileActiveRan = false;

    const active = bridge._runExclusive(async (operation) => {
      activeOperation = operation;
      bridge._disableWorkerFallback(new Error('worker request timeout'));
      await new Promise((resolve) => { releaseActive = resolve; });
    }, { kind: 'generation' });

    await tick();
    const successor = bridge._runExclusive(async (operation) => {
      successorOperation = operation;
      successorStartedWhileActiveRan = activeOperation.state === 'running';
    }, { kind: 'tokenize' });

    await tick();
    assert.equal(successorOperation, undefined, 'successor must wait for the active slot');

    bridge.cancel();
    await tick();

    assert.equal(activeOperation.cancelRequested, true);
    assert.equal(activeOperation.state, 'cancelling');
    assert.deepEqual(
      owned.calls.map(({ method }) => method),
      ['cancel'],
      'cancellation must reach the generation the active operation captured',
    );

    releaseActive();
    await active;
    await successor;

    assert.equal(activeOperation.state, 'cancelled');
    assert.equal(successorOperation.id, activeOperation.id + 1);
    assert.equal(successorOperation.cancelRequested, false);
    assert.equal(successorOperation.state, 'completed');
    assert.equal(
      successorStartedWhileActiveRan,
      false,
      'the successor must dispatch only after the cancelled operation reaches a terminal state',
    );
  }],

  ['split-model arrays survive worker safe-mode reload', async () => {
    const replayed = [];
    const proxy = createProxy('generation-6', {
      callImpl: (method, args) => {
        replayed.push({ method, args });
        return Promise.resolve({ value: 1 });
      },
    });
    const bridge = createBridge({
      _workerProxy: proxy,
      _workerGeneration: 6,
      _loadedModelUrl: ['shard-00001.gguf', 'shard-00002.gguf'],
      _loadedModelOptions: { nGpuLayers: 8 },
      _loadedMmProjUrl: 'projector.gguf',
    });

    assert.equal(await bridge._ensureWorkerMultimodalCpuMode(), true);

    const load = replayed.find(({ method }) => method === 'loadModelFromUrl');
    assert.ok(load, 'safe-mode reload must reissue the model load');
    assert.ok(Array.isArray(load.args[0]), 'a split model must stay an array');
    assert.deepEqual(load.args[0], ['shard-00001.gguf', 'shard-00002.gguf']);
    assert.deepEqual(
      replayed.map(({ method }) => method),
      ['loadModelFromUrl', 'loadMultimodalProjector'],
    );
  }],

  ['cancel during disposal never cancels the teardown operation', async () => {
    let releaseTeardown;
    const teardown = new Promise((resolve) => { releaseTeardown = resolve; });
    const proxy = createProxy('generation-9', { disposePromise: teardown });
    const runtime = {
      _modelBytes: 1,
      _runtimeNotes: [],
      cancelCalls: 0,
      cancel() {
        this.cancelCalls += 1;
      },
      async dispose() {},
      getModelMetadata: () => ({}),
      getContextSize: () => 128,
      isGpuActive: () => false,
      getBackendName: () => 'WASM (Prototype bridge)',
      supportsVision: () => false,
      supportsAudio: () => false,
    };
    const bridge = createBridge({ _workerProxy: proxy, _runtime: runtime });

    const disposal = bridge.dispose();
    await tick();

    const teardownOperation = bridge._activeOperation;
    assert.ok(teardownOperation, 'teardown must own the queue while disposing');
    assert.equal(teardownOperation.kind, 'dispose');

    bridge.cancel();
    await tick();

    assert.equal(
      teardownOperation.cancelRequested,
      false,
      'cancel() must not cancel the disposal teardown operation',
    );
    assert.notEqual(teardownOperation.state, 'cancelling');
    assert.equal(
      runtime.cancelCalls,
      0,
      'cancel() must not reach a runtime that disposal is tearing down',
    );
    assert.deepEqual(
      proxy.calls.map(({ method }) => method),
      [],
      'cancel() must not dispatch to a retiring worker proxy',
    );

    releaseTeardown();
    await disposal;

    bridge.cancel();
    assert.equal(runtime.cancelCalls, 0, 'cancel() after disposal must stay a no-op');
    assert.equal(bridge._lifecycleState, 'disposed');
  }],

  ['dispose racing a blocked recovery settles the owner first and leaks no resource', async () => {
    let releaseReload;
    const order = [];
    const retiring = createProxy('generation-1');
    const reloadedRuntime = {
      _modelBytes: 0,
      _runtimeNotes: [],
      disposeCalls: 0,
      async loadModelFromUrl() {
        await new Promise((resolve) => { releaseReload = resolve; });
        this._modelBytes = 1;
      },
      async dispose() {
        this.disposeCalls += 1;
        order.push('runtime-disposed');
      },
      getModelMetadata: () => ({ model: 'recovered' }),
      getContextSize: () => 4096,
      isGpuActive: () => false,
      getBackendName: () => 'WASM (Prototype bridge)',
      supportsVision: () => false,
      supportsAudio: () => false,
    };
    const bridge = createBridge({
      _workerProxy: retiring,
      _workerGeneration: 1,
      _loadedModelUrl: 'model.gguf',
      _loadedModelOptions: { nGpuLayers: 0 },
      _createRuntime: () => reloadedRuntime,
    });

    const active = bridge._runExclusive(async () => {
      const timeoutError = new Error('worker timeout');
      timeoutError.llamadartWorkerTimeout = true;
      bridge._disableWorkerFallback(timeoutError);
      await bridge._ensureRuntimeReadyAfterWorkerFallback({}, timeoutError);
      order.push('owner-settled');
    }, { kind: 'generation' });

    await tick();
    assert.ok(releaseReload, 'recovery must be blocked inside the reload');

    const disposal = bridge.dispose();
    let disposeResolved = false;
    disposal.then(() => { disposeResolved = true; }, () => { disposeResolved = true; });

    await tick();
    assert.equal(disposeResolved, false, 'dispose must wait for the blocked owner');

    releaseReload();
    await active.catch(() => {});
    await disposal;

    assert.equal(order[0], 'owner-settled', 'the active owner settles before teardown');
    assert.ok(order.includes('runtime-disposed'), 'the recovered runtime must be torn down');
    assert.equal(
      bridge._retiringWorkerDisposals.size,
      0,
      'no retiring worker may survive the teardown boundary',
    );
    assert.equal(bridge._workerProxy, null);
    assert.equal(bridge._runtime, null, 'no runtime may survive the teardown boundary');
    assert.equal(bridge._loadedModelUrl, null);
    assert.equal(bridge._lifecycleState, 'disposed');

    await assert.rejects(bridge.tokenize('after'), /Bridge has been disposed/);
    assert.throws(
      () => bridge._throwIfDisposed(),
      /Bridge has been disposed/,
      'post-disposal helpers must still reject',
    );
  }],

  ['dispose racing a blocked worker retirement still lets the owner install its replacement', async () => {
    const originalWorker = globalThis.Worker;
    const workers = [];
    class FakeWorker {
      constructor() {
        this.terminated = false;
        workers.push(this);
      }

      postMessage(message) {
        if (message?.type === 'init') {
          queueMicrotask(() => this.onmessage?.({ data: { type: 'ready' } }));
        }
      }

      terminate() {
        this.terminated = true;
      }
    }
    globalThis.Worker = FakeWorker;

    try {
      let releaseRetirement;
      const retiring = {
        disposeCalls: 0,
        dispose() {
          this.disposeCalls += 1;
          return new Promise((resolve) => { releaseRetirement = resolve; });
        },
      };

      const bridge = new LlamaWebGpuBridge({
        disableWorker: true,
        workerUrl: 'https://example.invalid/worker.js',
      });
      bridge._runtime = null;
      bridge._workerProxy = retiring;
      bridge._workerGeneration = 1;

      const active = bridge._runExclusive(
        () => bridge._replaceWorkerProxyForMultimodalCpuMode(),
        { kind: 'generation' },
      ).then(
        () => ({ outcome: 'resolved' }),
        (error) => ({ outcome: 'rejected', message: error?.message }),
      );

      await tick();
      assert.ok(releaseRetirement, 'the owner must be blocked retiring the old worker');

      const disposal = bridge.dispose();
      await tick();
      releaseRetirement();

      const activeOutcome = await active;
      assert.deepEqual(
        activeOutcome,
        { outcome: 'resolved' },
        'the already-active owner must finish its worker recovery during disposing',
      );

      await disposal;

      assert.equal(bridge._lifecycleState, 'disposed');
      assert.equal(bridge._workerProxy, null, 'no worker may survive teardown');
      assert.equal(bridge._runtime, null, 'no runtime may survive teardown');
      assert.equal(bridge._retiringWorkerDisposals.size, 0);
      assert.equal(retiring.disposeCalls, 1);
      assert.equal(workers.length, 1, 'the owner installed exactly one replacement');
      assert.equal(
        workers[0].terminated,
        true,
        'queued teardown must retire the replacement the owner installed',
      );

      await assert.rejects(bridge.tokenize('after'), /Bridge has been disposed/);
      await assert.rejects(
        bridge._replaceWorkerProxyForMultimodalCpuMode(),
        /Bridge has been disposed/,
        'post-disposal helpers must stay rejected',
      );
    } finally {
      globalThis.Worker = originalWorker;
    }
  }],
];

const failures = [];
for (const [name, run] of CASES) {
  try {
    await Promise.race([
      run(),
      new Promise((_, reject) => setTimeout(
        () => reject(new Error('timed out after 2000ms; lifecycle case deadlocked')),
        2000,
      )),
    ]);
  } catch (error) {
    failures.push(`${name}: ${error?.message || error}`);
  }
}

if (failures.length > 0) {
  console.error(`${failures.length}/${CASES.length} bridge operation lifecycle cases failed:`);
  for (const failure of failures) {
    console.error(`  - ${failure}`);
  }
  process.exitCode = 1;
} else {
  console.log(`Bridge operation lifecycle tests passed (${CASES.length} cases)`);
}
