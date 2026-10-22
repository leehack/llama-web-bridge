
import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';

// Public async methods intentionally stay off the single-writer queue.
export const UNQUEUED_ASYNC_METHODS = new Map([
  ['prefetchModelToCache', 'Cache Storage and network only; never enters the wasm core'],
  ['evictModelFromCache', 'Cache Storage only; never enters the wasm core'],
]);

// dispose() marks synchronously, rejects queued/new work, and drains after the active owner.
export const DISPOSE_METHOD = 'dispose';

export const UNQUEUED_SYNC_METHODS = new Set([
  'cancel',
  'setLogLevel',
  'supportsVision',
  'supportsAudio',
  'getModelMetadata',
  'getContextSize',
  'isGpuActive',
  'getBackendName',
]);

export const SNAPSHOT_KINDS = new Set(['metaJson', 'contextSize']);

export function opTrace(core) {
  return core.trace.filter(([kind]) => !SNAPSHOT_KINDS.has(kind));
}
/** Builds a real BridgeWorkerProxy while standing in for browser worker APIs. */
export async function withStubWorkerEnvironment(run) {
  const originals = {
    Worker: globalThis.Worker,
    createObjectURL: globalThis.URL.createObjectURL,
    revokeObjectURL: globalThis.URL.revokeObjectURL,
  };
  const workers = [];
  const revoked = [];

  globalThis.Worker = class StubWorker {
    constructor(url) {
      // Mirror a browser that refuses cross-origin module workers, so the proxy
      // falls back to the blob path and a blob URL must later be revoked.
      if (!String(url).startsWith('blob:')) {
        throw new Error('cross-origin worker blocked');
      }
      this.url = url;
      this.terminated = 0;
      this.posted = [];
      workers.push(this);
    }

    postMessage(message) {
      this.posted.push(message);
    }

    terminate() {
      this.terminated += 1;
    }
  };
  globalThis.URL.createObjectURL = () => 'blob:stub-worker';
  globalThis.URL.revokeObjectURL = (url) => revoked.push(url);

  try {
    // Await before restoring globals so the async body stays inside the stub.
    return await run({ workers, revoked });
  } finally {
    globalThis.Worker = originals.Worker;
    globalThis.URL.createObjectURL = originals.createObjectURL;
    globalThis.URL.revokeObjectURL = originals.revokeObjectURL;
  }
}

export function createRealWorkerBridge(config = {}) {
  return new LlamaWebGpuBridge({
    workerUrl: 'https://example.invalid/llama_webgpu_bridge_worker.js',
    ...config,
  });
}

// Drives a real BridgeWorkerProxy and delivers replies by request id.
export function workerDriver(proxy, worker) {
  return {
    ready() {
      proxy._worker.onmessage({ data: { type: 'ready' } });
    },
    calls(method = null) {
      return worker.posted.filter(
        (message) => message.type === 'call' && (method == null || message.method === method),
      );
    },
    reply(id, value, state = null) {
      proxy._worker.onmessage({ data: { type: 'result', id, value, state } });
    },
    error(id, message = 'Worker request failed', state = null) {
      proxy._worker.onmessage({ data: { type: 'error', id, message, state } });
    },
    async settle() {
      await new Promise((resolve) => setTimeout(resolve, 0));
    },
  };
}

// Direct runtime used to make unqueued fallback state observable.
export function createSnapshotRuntime(reads = []) {
  return {
    _runtimeNotes: [],
    _modelSource: 'network',
    _modelCacheState: 'disabled',
    _modelCacheName: 'direct-cache',
    getModelMetadata() {
      reads.push('metadata');
      return {
        'llamadart.webgpu.model_bytes': '2048',
        'llamadart.webgpu.n_gpu_layers': '0',
      };
    },
    getContextSize() {
      reads.push('contextSize');
      return 512;
    },
    isGpuActive: () => false,
    getBackendName: () => 'WASM (Prototype bridge)',
    supportsVision: () => false,
    supportsAudio: () => false,
    applyChatTemplate: (messages, addAssistant) =>
      `${messages.map((m) => `${m.role}: ${m.content}`).join('\n')}${addAssistant ? '\nassistant: ' : ''}`,
    setLogLevel: () => {},
    async dispose() {},
  };
}

export function tokensFor(text) {
  return [...String(text)].map((character) => character.codePointAt(0));
}

// Keep abandoned case promises from becoming process-level unhandled rejections.
export function settleQuietly(promise) {
  promise.catch(() => {});
  return promise;
}

const utf8Encoder = new TextEncoder();

// One char per byte, like the C++ std::string the core appends token pieces to.
function byteString(piece) {
  const bytes = piece instanceof Uint8Array ? piece : utf8Encoder.encode(String(piece));
  return String.fromCharCode(...bytes);
}

function bytesOf(text, offset = 0) {
  return [...text.slice(offset)].map((character) => character.charCodeAt(0));
}

// Emscripten 6.0.8's UTF8ArrayToString as `ccall(..., 'string')` runs it: it
// decodes 16 bytes or fewer by hand, reading past the terminator (zeros here)
// to finish an incomplete sequence and never checking continuation bytes.
export function emscriptenUtf8ToString(bytes) {
  if (bytes.length > 16) {
    return new TextDecoder().decode(Uint8Array.from(bytes));
  }
  const heap = [...bytes, 0, 0, 0, 0];
  let index = 0;
  let text = '';
  while (index < bytes.length) {
    let u0 = heap[index++];
    if (!(u0 & 128)) {
      text += String.fromCharCode(u0);
      continue;
    }
    const u1 = heap[index++] & 63;
    if ((u0 & 224) === 192) {
      text += String.fromCharCode(((u0 & 31) << 6) | u1);
      continue;
    }
    const u2 = heap[index++] & 63;
    if ((u0 & 240) === 224) {
      u0 = ((u0 & 15) << 12) | (u1 << 6) | u2;
    } else {
      u0 = ((u0 & 7) << 18) | (u1 << 12) | (u2 << 6) | (heap[index++] & 63);
    }
    if (u0 < 65536) {
      text += String.fromCharCode(u0);
    } else {
      const ch = u0 - 65536;
      text += String.fromCharCode(55296 | (ch >> 10), 56320 | (ch & 1023));
    }
  }
  return text;
}

/** Mirrors singleton result buffers and the C++ re-entrant-begin guard. */
export function createStubCore(scripts = {}) {
  const files = new Map();
  const core = {
    trace: [],
    beginRejections: 0,
    activePrompt: null,
    // g_last_output's bytes, one char per byte.
    lastOutput: '',
    usage: { promptTokens: 0, cachedPromptTokens: 0, completionTokens: 0 },
    lastTokensJson: '[]',
    lastDetokenized: '',
    lastEmbeddingJson: '[]',
    lastError: '',
    pending: [],
    scripts,
    // Probed by the runtime; a core older than the export lacks it.
    _llamadart_webgpu_last_output_bytes_json: () => {},
    FS: {
      readdir: () => ['states'],
      mkdir: () => {},
      analyzePath: (path) => ({ exists: files.has(path) }),
      readFile: (path) => files.get(path) || new Uint8Array(),
      writeFile: (path, bytes) => files.set(path, bytes),
      unlink: (path) => files.delete(path),
      open: () => 1,
      write: () => {},
      close: () => {},
    },
    ccall(name, _retType, _argTypes, args = []) {
      switch (name) {
        case 'llamadart_webgpu_begin_generation': {
          if (core.activePrompt !== null) {
            // Mirrors the C++ guard: reject without touching any shared buffer.
            core.beginRejections += 1;
            core.trace.push(['begin-rejected', args[0]]);
            return -7;
          }
          core.trace.push(['begin', args[0]]);
          core.activePrompt = args[0];
          core.lastOutput = '';
          core.usage = { promptTokens: tokensFor(args[0]).length, cachedPromptTokens: 0, completionTokens: 0 };
          core.pending = [...(core.scripts[args[0]] || [])];
          return 0;
        }

        case 'llamadart_webgpu_next_token': {
          if (core.activePrompt === null) {
            core.lastError = 'Generation is not active';
            return -1;
          }
          const piece = core.pending.shift();
          if (piece === undefined) {
            return 0;
          }
          if (piece instanceof Error) {
            // A step can append its piece before llama_decode fails.
            if (piece.appended != null) {
              core.lastOutput += byteString(piece.appended);
            }
            core.lastError = piece.message;
            return -1;
          }
          core.lastOutput += byteString(piece);
          core.usage.completionTokens += 1;
          return 1;
        }

        case 'llamadart_webgpu_last_output':
          return emscriptenUtf8ToString(bytesOf(core.lastOutput));

        case 'llamadart_webgpu_last_output_bytes_json':
          if (typeof core._llamadart_webgpu_last_output_bytes_json !== 'function') {
            // Emscripten's ccall on a missing export.
            throw new TypeError('func is not a function');
          }
          return JSON.stringify(bytesOf(core.lastOutput, Math.max(0, Number(args[0]) || 0)));

        case 'llamadart_webgpu_last_generation_usage_json':
          core.trace.push(['usage', null]);
          return JSON.stringify(core.usage);

        case 'llamadart_webgpu_end_generation':
          core.trace.push(['end', core.activePrompt]);
          core.activePrompt = null;
          return undefined;

        case 'llamadart_webgpu_tokenize_to_json':
          core.trace.push(['tokenize', args[0]]);
          core.lastTokensJson = JSON.stringify(tokensFor(args[0]));
          return 0;

        case 'llamadart_webgpu_last_tokens_json':
          return core.lastTokensJson;

        case 'llamadart_webgpu_detokenize_from_json':
          core.trace.push(['detokenize', args[0]]);
          core.lastDetokenized = JSON.parse(args[0])
            .map((token) => String.fromCodePoint(token))
            .join('');
          return 0;

        case 'llamadart_webgpu_last_detokenized':
          return core.lastDetokenized;

        case 'llamadart_webgpu_embed_to_json':
          core.trace.push(['embed', args[0]]);
          core.lastEmbeddingJson = JSON.stringify(tokensFor(args[0]));
          return 0;

        case 'llamadart_webgpu_last_embedding_json':
          return core.lastEmbeddingJson;

        case 'llamadart_webgpu_score_next_token_to_json':
          core.trace.push(['score', args[0]]);
          core.lastNextTokenScoresJson = JSON.stringify({
            candidates: [],
            top: [{ token: tokensFor(args[0])[0], bytes: [], logprob: 0 }],
            promptTokens: tokensFor(args[0]).length,
          });
          return 0;

        case 'llamadart_webgpu_last_next_token_scores_json':
          return core.lastNextTokenScoresJson;

        case 'llamadart_webgpu_state_save_file':
          core.trace.push(['stateSave', args[0]]);
          files.set(args[0], new Uint8Array([1, 2, 3]));
          return 3;

        case 'llamadart_webgpu_state_load_file':
          core.trace.push(['stateLoad', args[1]]);
          return 3;

        case 'llamadart_webgpu_last_error':
          return core.lastError;

        case 'llamadart_webgpu_model_meta_json':
          core.trace.push(['metaJson', null]);
          return JSON.stringify({ 'llamadart.webgpu.stub': '1' });

        case 'llamadart_webgpu_get_context_size':
          core.trace.push(['contextSize', null]);
          return 128;

        case 'llamadart_webgpu_media_clear_pending':
          return undefined;

        case 'llamadart_webgpu_mmproj_free':
          core.trace.push(['mmprojFree', null]);
          return 0;

        case 'llamadart_webgpu_shutdown':
          core.trace.push(['shutdown', null]);
          return undefined;

        default:
          throw new Error(`Unexpected ccall: ${name}`);
      }
    },
  };

  return core;
}

export function createDirectBridge(scripts = {}) {
  const bridge = new LlamaWebGpuBridge({ disableWorker: true });
  const core = createStubCore(scripts);
  bridge._runtime._core = core;
  bridge._runtime._modelBytes = 1;
  bridge._runtime._nCtx = 128;
  bridge._captureDirectRuntimeState?.();
  core.trace.length = 0;
  return { bridge, core };
}

export function createWorkerBridge(overrides = {}) {
  const bridge = Object.create(LlamaWebGpuBridge.prototype);
  Object.assign(bridge, {
    _config: {},
    _runtime: null,
    _workerProxy: {},
    _workerDisposePromise: null,
    _workerFallbackReason: null,
    _operationQueueTail: null,
    _disposalWaiters: new Set(),
    _metadata: {},
    _contextSize: 128,
    _loadedModelUrl: 'model.gguf',
    _loadedModelOptions: { nGpuLayers: 0 },
    _loadedMmProjUrl: 'mmproj.gguf',
    _multimodalWorkerCpuMode: false,
    _bridgeWarnRecent: new Map(),
    _emitBridgeWarn: () => {},
    ...overrides,
  });
  return bridge;
}
