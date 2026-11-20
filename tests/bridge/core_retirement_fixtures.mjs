import * as api from '../../js/src/llama_webgpu_bridge.js';

export { api };
export const { LlamaWebGpuBridge } = api;

export const MODEL_STATE = {
  metadata: { 'llamadart.webgpu.model_bytes': '1024', 'llamadart.webgpu.n_gpu_layers': '0' },
  contextSize: 128,
  gpuActive: false,
  backendName: 'WASM (Prototype bridge)',
  supportsVision: false,
  supportsAudio: false,
};
export const EMPTY_STATE = { ...MODEL_STATE, metadata: {}, contextSize: 0 };

// A fake BridgeWorkerProxy. `impl(method, args, proxy)` returns a value or throws.
export function createProxy(name, impl) {
  const proxy = {
    name,
    calls: [],
    disposeCalls: 0,
    hasModel: false,
    async call(method, args) {
      proxy.calls.push(method);
      if (method === 'cancel' || method === 'setLogLevel') {
        return { value: undefined };
      }
      return impl(method, args, proxy);
    },
    async dispose() {
      proxy.disposeCalls += 1;
    },
  };
  return proxy;
}

// A healthy worker: loads succeed and report a model; other calls echo.
export function healthyImpl(method, _args, proxy) {
  if (method === 'loadModelFromUrl') {
    proxy.hasModel = true;
    return { value: 1, state: MODEL_STATE };
  }
  if (method === 'createCompletion') {
    if (!proxy.hasModel) {
      throw Object.assign(new Error('No model loaded. Call loadModelFromUrl first.'), { state: EMPTY_STATE });
    }
    return { value: `text from ${proxy.name}`, state: MODEL_STATE };
  }
  if (method === 'loadLoraAdapter') {
    proxy.loraLoads = (proxy.loraLoads || 0) + 1;
    return { value: { handle: 100 + proxy.loraLoads } };
  }
  if (method === 'tokenize') {
    return { value: [1, 2, 3] };
  }
  return { value: undefined };
}

export function createRuntime(overrides = {}) {
  const runtime = {
    _modelBytes: 0,
    _runtimeNotes: [],
    calls: [],
    async loadModelFromUrl(url) {
      runtime.calls.push(['loadModelFromUrl', url]);
      runtime._modelBytes = 1;
    },
    async createCompletion() {
      runtime.calls.push(['createCompletion']);
      if (!(runtime._modelBytes > 0)) {
        throw new Error('No model loaded. Call loadModelFromUrl first.');
      }
      return 'text from main thread';
    },
    async tokenize() {
      runtime.calls.push(['tokenize']);
      return [9];
    },
    async loadMultimodalProjector() {},
    async dispose() {},
    cancel() {},
    getModelMetadata: () => (runtime._modelBytes > 0 ? MODEL_STATE.metadata : {}),
    getContextSize: () => 128,
    isGpuActive: () => false,
    getBackendName: () => 'WASM (Prototype bridge)',
    supportsVision: () => false,
    supportsAudio: () => false,
    ...overrides,
  };
  return runtime;
}

// A facade built without its constructor, as the repo's own tests do, with a
// counting worker factory. `nextProxy(index)` builds each replacement.
export function createBridge({ proxy = null, nextProxy, overrides = {} } = {}) {
  const warnings = [];
  const created = [];
  const runtimes = [];
  const bridge = Object.create(LlamaWebGpuBridge.prototype);
  Object.assign(bridge, {
    _config: {},
    _runtime: null,
    _workerProxy: proxy,
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
    _loadedDraftModel: null,
    _multimodalWorkerCpuMode: false,
    _workerModelMissing: false,
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
    _decisionHeads: new Map(),
    _nextDecisionHandle: 1,
    _loraAdapters: new Map(),
    _activeLoraScales: new Map(),
    _nextLoraHandle: 1,
    _emitBridgeWarn: (message) => warnings.push(message),
    _createRuntime: () => {
      const runtime = createRuntime();
      runtimes.push(runtime);
      return runtime;
    },
    _createWorkerProxy: () => {
      const replacement = (nextProxy || ((index) => createProxy(`replacement-${index}`, healthyImpl)))(
        created.length + 1,
      );
      created.push(replacement);
      return replacement;
    },
    ...overrides,
  });
  return { bridge, warnings, created, runtimes };
}

export async function outcome(promise) {
  try {
    return { ok: true, value: await promise };
  } catch (error) {
    return {
      ok: false,
      name: error?.name,
      code: error?.code,
      message: String(error?.message ?? error),
      error,
    };
  }
}

// Facade topology after a call: what a parity guard compares.
export function topology(bridge, original, created, runtimes) {
  return {
    worker: bridge._workerProxy === original
      ? 'kept'
      : bridge._workerProxy ? `replaced(${bridge._workerProxy.name})` : 'none',
    workersCreated: created.length,
    originalDisposed: original ? original.disposeCalls : 0,
    replacementsDisposed: created.filter((entry) => entry.disposeCalls > 0).length,
    directRuntimes: runtimes.length,
    fallbackReason: bridge._workerFallbackReason,
    rememberedSource: bridge._loadedModelUrl,
    workerModelMissing: bridge._workerModelMissing,
    modelBytes: bridge._metadata?.['llamadart.webgpu.model_bytes'] ?? null,
  };
}
