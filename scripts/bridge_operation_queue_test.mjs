import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { LlamaWebGpuBridge } from '../js/src/llama_webgpu_bridge.js';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

/**
 * Public async methods that intentionally stay off the single-writer queue.
 * Each entry needs a reason that holds for the whole call, not just its happy
 * path, because anything reaching the singleton core must be serialized.
 */
const UNQUEUED_ASYNC_METHODS = new Map([
  ['prefetchModelToCache', 'Cache Storage and network only; never enters the wasm core'],
  ['evictModelFromCache', 'Cache Storage only; never enters the wasm core'],
]);

/**
 * dispose() is neither a queued operation nor a plain exclusion: it marks the
 * bridge disposed synchronously and rejects queued and new work, then takes the
 * queue with allowDisposed so teardown runs only after the active owner has
 * released the slot. It does not interrupt the operation already running.
 */
const DISPOSE_METHOD = 'dispose';

const UNQUEUED_SYNC_METHODS = new Set([
  'cancel',
  'setLogLevel',
  'supportsVision',
  'supportsAudio',
  'getModelMetadata',
  'getContextSize',
  'isGpuActive',
  'getBackendName',
]);

// The in-slot snapshot refresh issues its own ccalls; ordering assertions care
// about the operations themselves.
const SNAPSHOT_KINDS = new Set(['metaJson', 'contextSize']);

function opTrace(core) {
  return core.trace.filter(([kind]) => !SNAPSHOT_KINDS.has(kind));
}


/**
 * Builds a real BridgeWorkerProxy (not a `_callWorker` stub) by standing in for
 * the browser worker APIs, so proxy-level timer, readiness and pending-request
 * behaviour is exercised directly.
 */
async function withStubWorkerEnvironment(run) {
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
    // Must await: restoring the globals in `finally` before the async body runs
    // would silently un-stub the environment mid-case.
    return await run({ workers, revoked });
  } finally {
    globalThis.Worker = originals.Worker;
    globalThis.URL.createObjectURL = originals.createObjectURL;
    globalThis.URL.revokeObjectURL = originals.revokeObjectURL;
  }
}

function createRealWorkerBridge(config = {}) {
  return new LlamaWebGpuBridge({
    workerUrl: 'https://example.invalid/llama_webgpu_bridge_worker.js',
    ...config,
  });
}

// Drives a real BridgeWorkerProxy from the test: mark it ready, read the calls
// it actually posted, and deliver worker replies by request id.
function workerDriver(proxy, worker) {
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

// A direct runtime that only answers the snapshot getters, so a fallback from an
// unqueued path has something distinguishable to capture.
function createSnapshotRuntime(reads = []) {
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

function tokensFor(text) {
  return [...String(text)].map((character) => character.codePointAt(0));
}

// A case that fails early can abandon a still-pending bridge promise; keep that
// from turning into a process-level unhandled rejection so every case reports.
function settleQuietly(promise) {
  promise.catch(() => {});
  return promise;
}

/**
 * Mirrors the singleton llama.cpp core: every result is published through one
 * global buffer, and begin_generation refuses a second entry the way the C++
 * guard does instead of clearing the active generation's output.
 */
function createStubCore(scripts = {}) {
  const files = new Map();
  const core = {
    trace: [],
    beginRejections: 0,
    activePrompt: null,
    lastOutput: '',
    lastTokensJson: '[]',
    lastDetokenized: '',
    lastEmbeddingJson: '[]',
    lastError: '',
    pending: [],
    scripts,
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
            core.lastError = piece.message;
            return -1;
          }
          core.lastOutput += piece;
          return 1;
        }

        case 'llamadart_webgpu_last_output':
          return core.lastOutput;

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

function createDirectBridge(scripts = {}) {
  const bridge = new LlamaWebGpuBridge({ disableWorker: true });
  const core = createStubCore(scripts);
  bridge._runtime._core = core;
  bridge._runtime._modelBytes = 1;
  bridge._runtime._nCtx = 128;
  // Optional so this suite can also run against a build without the queue,
  // where every failure is then behavioural rather than a missing helper.
  bridge._captureDirectRuntimeState?.();
  core.trace.length = 0;
  return { bridge, core };
}

function createWorkerBridge(overrides = {}) {
  const bridge = Object.create(LlamaWebGpuBridge.prototype);
  Object.assign(bridge, {
    _config: {},
    _runtime: null,
    _workerProxy: {},
    _workerDisposePromise: null,
    _workerFallbackReason: null,
    _operationQueueTail: null,
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

/** @type {Array<[string, () => Promise<void>]>} */
const CASES = [
  ['concurrent tokenization ownership (direct)', async () => {
    const { bridge, core } = createDirectBridge();

    const [alpha, beta] = await Promise.all([
      bridge.tokenize('a'),
      bridge.tokenize('b'),
    ]);

    assert.deepEqual(alpha, [97], 'the first tokenize must own its own result buffer');
    assert.deepEqual(beta, [98], 'the second tokenize must own its own result buffer');
    assert.deepEqual(
      opTrace(core).map(([kind, value]) => `${kind}:${value}`),
      ['tokenize:a', 'tokenize:b'],
      'tokenization must run in FIFO order',
    );
  }],

  // Concurrent completions run one at a time, in order, with intact outputs.

  ['concurrent completions FIFO and intact outputs (direct)', async () => {
    const { bridge, core } = createDirectBridge({
      alpha: ['A', 'L', 'P', 'H', 'A'],
      beta: ['B', 'E', 'T', 'A'],
      gamma: ['G'],
    });

    const results = await Promise.all([
      bridge.createCompletion('alpha'),
      bridge.createCompletion('beta'),
      bridge.createCompletion('gamma'),
    ]);

    assert.deepEqual(results, ['ALPHA', 'BETA', 'G']);
    assert.equal(core.beginRejections, 0, 'the core guard must never be reached');
    assert.deepEqual(
      opTrace(core).map(([kind, value]) => `${kind}:${value}`),
      ['begin:alpha', 'end:alpha', 'begin:beta', 'end:beta', 'begin:gamma', 'end:gamma'],
      'completions must be strictly serialized in FIFO order',
    );
  }],

  // Mixed operations keep FIFO order and never interleave against the core.

  ['cross-operation ordering (direct)', async () => {
    const { bridge, core } = createDirectBridge({ alpha: ['A'] });

    const results = await Promise.all([
      bridge.createCompletion('alpha'),
      bridge.tokenize('b'),
      bridge.embed('c'),
      bridge.detokenize([100]),
      bridge.stateSaveBytes([1]),
    ]);

    assert.equal(results[0], 'A');
    assert.deepEqual(results[1], [98]);
    assert.deepEqual(results[2], [99]);
    assert.equal(results[3], 'd');
    assert.deepEqual([...results[4]], [1, 2, 3]);
    assert.deepEqual(
      opTrace(core).map(([kind]) => kind),
      ['begin', 'end', 'tokenize', 'embed', 'detokenize', 'stateSave'],
      'cross-operation ordering must follow call order without interleaving',
    );
  }],

  // A failing operation must release the queue for the operations behind it.

  ['error recovery releases the queue (direct)', async () => {
    const { bridge, core } = createDirectBridge({ beta: ['B'] });

    const failing = bridge.createCompletion('missing-script');
    const queued = bridge.createCompletion('beta');

    assert.equal(await failing, '', 'the stub yields no tokens for an unscripted prompt');
    assert.equal(await queued, 'B');

    bridge._runtime._modelBytes = 0;
    await assert.rejects(bridge.tokenize('a'), /No model loaded/);
    bridge._runtime._modelBytes = 1;
    assert.deepEqual(await bridge.tokenize('a'), [97], 'a rejected operation must not wedge the queue');
    assert.equal(core.beginRejections, 0);
  }],

  // Cancellation is out-of-band: it must reach a running generation instead of
  // queueing behind it, and the queued follow-up must still run.

  ['cancellation stays out-of-band (direct)', async () => {
    // Keep the script under the 256-token nPredict default so an uncancelled run
    // would return every token and truncation is unambiguous.
    const scriptedTokens = 200;
    const { bridge, core } = createDirectBridge({
      alpha: Array.from({ length: scriptedTokens }, () => 'A'),
    });

    const completion = settleQuietly(bridge.createCompletion('alpha'));
    const queuedTokenize = settleQuietly(bridge.tokenize('b'));

    await new Promise((resolve) => setTimeout(resolve, 0));
    bridge.cancel();

    const text = await completion;
    assert.ok(
      text.length > 0 && text.length < scriptedTokens,
      `cancel must truncate generation, got ${text.length}/${scriptedTokens}`,
    );
    assert.deepEqual(await queuedTokenize, [98]);
    assert.deepEqual(
      opTrace(core).filter(([kind]) => kind !== 'begin' && kind !== 'end').map(([kind]) => kind),
      ['tokenize'],
      'the queued operation must run after the cancelled generation released the queue',
    );
  }],

  // Worker dispatch: the same queue serializes worker RPCs in FIFO order.

  ['worker dispatch FIFO ordering', async () => {
    const dispatched = [];
    const active = [];
    const releases = [];
    const bridge = createWorkerBridge({
      _callWorker: (method, args) => {
        dispatched.push(`${method}:${JSON.stringify(args[0])}`);
        active.push(method);
        assert.equal(active.length, 1, `worker dispatch overlapped on ${method}`);
        return new Promise((resolve) => {
          releases.push(() => {
            active.pop();
            resolve(method === 'createCompletion' ? 'WORKER-ALPHA' : [98]);
          });
        });
      },
    });

    const completion = settleQuietly(bridge.createCompletion('alpha'));
    const tokenize = settleQuietly(bridge.tokenize('b'));

    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.deepEqual(dispatched, ['createCompletion:"alpha"'], 'the queued tokenize must not dispatch yet');

    releases[0]();
    assert.equal(await completion, 'WORKER-ALPHA');
    await new Promise((resolve) => setTimeout(resolve, 0));
    releases[1]();
    assert.deepEqual(await tokenize, [98]);
    assert.deepEqual(dispatched, ['createCompletion:"alpha"', 'tokenize:"b"']);
  }],

  // Worker dispatch: cancel() bypasses the queue while a completion is in flight.

  ['worker cancellation stays out-of-band', async () => {
    const dispatched = [];
    let releaseCompletion = null;
    const bridge = createWorkerBridge({
      _callWorker: (method) => {
        dispatched.push(method);
        if (method === 'cancel') {
          return Promise.resolve(undefined);
        }
        return new Promise((resolve) => {
          releaseCompletion = () => resolve('WORKER-ALPHA');
        });
      },
    });

    const completion = settleQuietly(bridge.createCompletion('alpha'));
    settleQuietly(bridge.tokenize('b'));

    await new Promise((resolve) => setTimeout(resolve, 0));
    bridge.cancel();
    assert.deepEqual(
      dispatched,
      ['createCompletion', 'cancel'],
      'cancel must reach the worker before the in-flight completion settles',
    );

    releaseCompletion();
    assert.equal(await completion, 'WORKER-ALPHA');
  }],

  // The recursive empty-multimodal worker retry must not deadlock on the queue.

  ['recursive empty-multimodal retry does not deadlock', async () => {
    const dispatched = [];
    const bridge = createWorkerBridge({
      _ensureWorkerMultimodalCpuMode: async () => {},
      _replaceWorkerProxyForMultimodalCpuMode: async () => {},
      _callWorker: async (method, args) => {
        dispatched.push([method, args[0]]);
        return dispatched.length === 1 ? '' : 'RETRIED';
      },
    });

    const options = { parts: [{ type: 'image', bytes: new Uint8Array([1]) }] };
    const result = await Promise.race([
      bridge.createCompletion('alpha', options),
      new Promise((_, reject) => {
        setTimeout(() => reject(new Error('multimodal retry deadlocked on the queue')), 1000);
      }),
    ]);

    assert.equal(result, 'RETRIED');
    assert.equal(dispatched.length, 2, 'the empty-response retry must reach the worker twice');
    assert.equal(await bridge.createCompletion('alpha', options), 'RETRIED', 'the queue must release');
  }],

  // The runtime CPU-recovery retry recursion must still work under the queue.

  ['recursive runtime CPU recovery does not deadlock', async () => {
    const scripts = { alpha: [new Error('failed to decode')] };
    const { bridge, core } = createDirectBridge(scripts);
    const runtime = bridge._runtime;
    runtime._shouldAttemptGenerationRecovery = () => true;
    runtime._recoverGenerationWithCpuFallback = async () => {
      scripts.alpha = ['O', 'K'];
      return true;
    };

    const result = await Promise.race([
      bridge.createCompletion('alpha'),
      new Promise((_, reject) => {
        setTimeout(() => reject(new Error('CPU recovery retry deadlocked on the queue')), 1000);
      }),
    ]);

    assert.equal(result, 'OK');
    assert.deepEqual(
      opTrace(core).filter(([kind]) => kind === 'begin').map(([, prompt]) => prompt),
      ['alpha', 'alpha'],
      'the runtime recovery retry must re-enter generation',
    );
    assert.equal(core.beginRejections, 0, 'recovery must end the failed generation before retrying');
  }],

  // A pre-aborted signal must reject before the call ever takes a queue slot.

  ['pre-aborted calls never reach the core or the worker (direct)', async () => {
    const { bridge, core } = createDirectBridge({ alpha: ['A'] });
    const controller = new AbortController();
    controller.abort();

    await assert.rejects(
      bridge.createCompletion('alpha', { signal: controller.signal }),
      (error) => error?.name === 'AbortError',
    );
    assert.deepEqual(opTrace(core), [], 'a pre-aborted completion must not enter the core');

    assert.equal(await bridge.createCompletion('alpha'), 'A', 'the queue must stay usable');
  }],

  // Aborting while queued must skip the slot without cancelling the predecessor.

  ['abort while queued skips the slot and spares the predecessor (direct)', async () => {
    const scriptedTokens = 60;
    const { bridge, core } = createDirectBridge({
      first: Array.from({ length: scriptedTokens }, () => 'F'),
      second: ['S'],
    });
    const controller = new AbortController();

    const first = settleQuietly(bridge.createCompletion('first'));
    const second = settleQuietly(
      bridge.createCompletion('second', { signal: controller.signal }),
    );

    await new Promise((resolve) => setTimeout(resolve, 0));
    controller.abort();

    await assert.rejects(second, (error) => error?.name === 'AbortError');
    assert.equal(
      await first,
      'F'.repeat(scriptedTokens),
      'aborting a queued call must not cancel the unrelated active predecessor',
    );
    assert.deepEqual(
      opTrace(core).filter(([kind]) => kind === 'begin').map(([, prompt]) => prompt),
      ['first'],
      'the aborted queued completion must never call begin_generation',
    );
  }],

  // The same rule holds for worker dispatch: no RPC for a skipped slot.

  ['abort while queued skips worker dispatch', async () => {
    const dispatched = [];
    let releaseFirst = null;
    const controller = new AbortController();
    const bridge = createWorkerBridge({
      _callWorker: (method, args) => {
        dispatched.push(`${method}:${JSON.stringify(args?.[0] ?? null)}`);
        if (method === 'cancel') {
          return Promise.resolve(undefined);
        }
        return new Promise((resolve) => {
          releaseFirst = () => resolve('FIRST');
        });
      },
    });

    const first = settleQuietly(bridge.createCompletion('first'));
    const second = settleQuietly(
      bridge.createCompletion('second', { signal: controller.signal }),
    );

    await new Promise((resolve) => setTimeout(resolve, 0));
    controller.abort();
    await assert.rejects(second, (error) => error?.name === 'AbortError');

    assert.deepEqual(
      dispatched,
      ['createCompletion:"first"'],
      'the aborted queued call must not dispatch, and must not cancel the predecessor',
    );

    releaseFirst();
    assert.equal(await first, 'FIRST');
  }],

  // Queued text-to-speech and model loads honour the same signal contract.

  ['queued synthesizeSpeech and loadModelFromUrl honour abort', async () => {
    const dispatched = [];
    let releaseFirst = null;
    const bridge = createWorkerBridge({
      _callWorker: (method) => {
        dispatched.push(method);
        if (method === 'cancel') {
          return Promise.resolve(undefined);
        }
        return new Promise((resolve) => {
          releaseFirst = () => resolve('FIRST');
        });
      },
    });

    const first = settleQuietly(bridge.createCompletion('first'));
    const ttsController = new AbortController();
    const loadController = new AbortController();
    const tts = settleQuietly(bridge.synthesizeSpeech({ text: 'hi', signal: ttsController.signal }));
    const load = settleQuietly(
      bridge.loadModelFromUrl('model.gguf', { signal: loadController.signal }),
    );

    await new Promise((resolve) => setTimeout(resolve, 0));
    ttsController.abort();
    loadController.abort();

    await assert.rejects(tts, (error) => error?.name === 'AbortError');
    await assert.rejects(load, (error) => error?.name === 'AbortError');
    assert.deepEqual(dispatched, ['createCompletion'], 'no skipped slot may dispatch');

    releaseFirst();
    assert.equal(await first, 'FIRST');
  }],

  // Synchronous getters must never enter the core while async work owns it.

  ['synchronous getters do not enter the core during active work (direct)', async () => {
    const scriptedTokens = 40;
    const { bridge, core } = createDirectBridge({
      alpha: Array.from({ length: scriptedTokens }, () => 'A'),
    });

    const completion = settleQuietly(bridge.createCompletion('alpha'));
    await new Promise((resolve) => setTimeout(resolve, 0));

    const traceLengthBeforeGetters = core.trace.length;
    const metadata = bridge.getModelMetadata();
    const contextSize = bridge.getContextSize();
    const backendName = bridge.getBackendName();
    const gpuActive = bridge.isGpuActive();
    const supportsVision = bridge.supportsVision();

    assert.equal(contextSize, 128, 'context size must be served from the facade cache');
    assert.equal(metadata['llamadart.webgpu.execution'], 'main-thread');
    assert.equal(metadata['llamadart.webgpu.stub'], '1', 'cached metadata must be real');
    assert.equal(typeof backendName, 'string');
    assert.equal(typeof gpuActive, 'boolean');
    assert.equal(typeof supportsVision, 'boolean');
    assert.deepEqual(
      core.trace.slice(traceLengthBeforeGetters),
      [],
      'no synchronous getter may ccall while an async operation owns the runtime',
    );

    await completion;
  }],

  // State-load default capacities must be resolved inside the queue slot, so a
  // model load queued ahead of them is reflected in the capacity they use.

  ['state load default capacity resolves inside the slot (direct)', async () => {
    const { bridge, core } = createDirectBridge({ alpha: ['A'] });

    // A queued load that grows the context; the state load is enqueued behind it
    // before the new size is observable.
    bridge._runtime.loadModelFromUrl = async () => {
      bridge._runtime._nCtx = 4096;
      core.ccall = ((inner) => function ccall(name, retType, argTypes, args = []) {
        if (name === 'llamadart_webgpu_get_context_size') {
          core.trace.push(['contextSize', null]);
          return 4096;
        }
        return inner.call(core, name, retType, argTypes, args);
      })(core.ccall);
      return undefined;
    };

    const completion = settleQuietly(bridge.createCompletion('alpha'));
    const load = settleQuietly(bridge.loadModelFromUrl('model.gguf'));
    const stateLoad = settleQuietly(bridge.stateLoadBytes(new Uint8Array([1, 2, 3])));

    await completion;
    await load;
    await stateLoad;

    const kinds = opTrace(core).map(([kind]) => kind);
    assert.deepEqual(
      kinds.filter((kind) => kind === 'begin' || kind === 'stateLoad'),
      ['begin', 'stateLoad'],
      'the state load must run after the earlier slots released',
    );

    const stateLoadCapacity = opTrace(core).find(([kind]) => kind === 'stateLoad')?.[1];
    assert.equal(
      stateLoadCapacity,
      4096,
      'the default capacity must be resolved inside the slot, after the queued load',
    );
    assert.equal(
      bridge.getContextSize(),
      4096,
      'the facade snapshot must be refreshed by the queued load',
    );
  }],

  ['state load explicit capacities handle conversion exceptions and single coercion', async () => {
    const { bridge, core } = createDirectBridge();
    const resolved = [];
    let coercions = 0;
    const throwingPrimitive = {
      [Symbol.toPrimitive]() {
        throw new TypeError('cannot convert token capacity');
      },
    };
    const coercible = {
      valueOf() {
        coercions += 1;
        return 9.75;
      },
    };

    for (const tokenCapacity of [
      0,
      -1,
      Number.NaN,
      Number.POSITIVE_INFINITY,
      Symbol('invalid'),
      throwingPrimitive,
      coercible,
    ]) {
      await bridge.stateLoadBytes(new Uint8Array([1, 2, 3]), tokenCapacity);
      resolved.push(
        opTrace(core).filter(([kind]) => kind === 'stateLoad').at(-1)?.[1],
      );
    }

    assert.deepEqual(
      resolved,
      [128, 128, 128, Number.POSITIVE_INFINITY, 128, 128, 9],
      'invalid capacities fall back while valid numeric conversion is truncated',
    );
    assert.equal(coercions, 1, 'a capacity object must be coerced only once');
  }],

  // Disposal marks synchronously, rejects queued and new work, and tears down
  // only once the current owner has released the slot. The active operation is
  // not interrupted; it keeps the cancellation behaviour it has on the main
  // branch.
  ['dispose rejects queued work and tears down after the active owner (direct)', async () => {
    const scriptedTokens = 40;
    const { bridge, core } = createDirectBridge({
      alpha: Array.from({ length: scriptedTokens }, () => 'A'),
      beta: ['B'],
    });

    const active = settleQuietly(bridge.createCompletion('alpha'));
    const queued = settleQuietly(bridge.createCompletion('beta'));

    await new Promise((resolve) => setTimeout(resolve, 0));
    const disposal = bridge.dispose();

    await assert.rejects(queued, /Bridge has been disposed/);
    assert.equal(
      await active,
      'A'.repeat(scriptedTokens),
      'the active operation completes normally; disposal does not interrupt it',
    );
    assert.deepEqual(
      opTrace(core).filter(([kind]) => kind === 'begin').map(([, prompt]) => prompt),
      ['alpha'],
      'the queued completion must never reach the core',
    );

    await disposal;
    await bridge.dispose();
    await assert.rejects(bridge.tokenize('a'), /Bridge has been disposed/);
    assert.equal(bridge._runtime, null, 'teardown completes after the owner released');
  }],


  ['dispose rejects queued work and tears down after the active owner (worker)', async () => {
    const dispatched = [];
    let releaseActive = null;
    let disposeCalls = 0;
    const bridge = createWorkerBridge({
      _workerProxy: {
        async dispose() {
          disposeCalls += 1;
        },
      },
      _callWorker: (method) => {
        dispatched.push(method);
        return new Promise((resolve) => {
          releaseActive = () => resolve('ACTIVE');
        });
      },
    });

    const active = settleQuietly(bridge.createCompletion('alpha'));
    const queued = settleQuietly(bridge.tokenize('b'));

    await new Promise((resolve) => setTimeout(resolve, 0));
    const disposal = bridge.dispose();

    await assert.rejects(queued, /Bridge has been disposed/);
    assert.deepEqual(dispatched, ['createCompletion'], 'queued work must never dispatch');
    assert.equal(disposeCalls, 0, 'teardown must wait for the active owner');

    releaseActive();
    assert.equal(await active, 'ACTIVE', 'active work still settles normally');

    await disposal;
    await bridge.dispose();
    assert.equal(disposeCalls, 1, 'worker teardown must be idempotent');
    assert.equal(bridge._workerProxy, null);
    await assert.rejects(bridge.embed('x'), /Bridge has been disposed/);
  }],


  // Disposal listeners must be scoped to a pending operation, not retained for
  // the bridge lifetime once the operation has settled.

  ['settled operations do not accumulate lifecycle listeners', async () => {
    const { bridge } = createDirectBridge({ alpha: ['A'] });
    assert.equal(bridge._disposalWaiters.size, 0, 'a fresh bridge holds no waiters');

    for (let index = 0; index < 25; index += 1) {
      await bridge.tokenize('a');
    }
    assert.equal(
      bridge._disposalWaiters.size,
      0,
      'settled operations must detach their disposal listener',
    );

    const pending = settleQuietly(bridge.createCompletion('alpha'));
    assert.equal(
      bridge._disposalWaiters.size,
      1,
      'a pending operation registers exactly one disposal listener',
    );
    await pending;
    assert.equal(bridge._disposalWaiters.size, 0, 'settlement must detach the listener');

    const rejected = settleQuietly(bridge.stateLoadBytes(new Uint8Array()));
    await assert.rejects(rejected, /State bytes are empty/);
    assert.equal(
      bridge._disposalWaiters.size,
      0,
      'a rejected operation must detach its listener too',
    );
  }],

  // Cache-only helpers refresh their public metadata snapshot after settlement.

  ['cache metadata reflects completed prefetch and eviction', async () => {
    const { bridge } = createDirectBridge();
    const runtime = bridge._runtime;

    const baseline = bridge.getModelMetadata();
    const baselineContext = bridge.getContextSize();

    runtime.prefetchModelToCache = async function prefetch() {
      this._modelCacheState = 'ready';
      this._modelCacheName = 'custom-cache-v9';
      this._modelSource = 'cache';
      this._runtimeNotes.push('model_cache_prefetched');
      return true;
    };
    assert.equal(await bridge.prefetchModelToCache('model.gguf'), true);

    let metadata = bridge.getModelMetadata();
    assert.equal(metadata['llamadart.webgpu.model_cache_state'], 'ready');
    assert.equal(metadata['llamadart.webgpu.model_cache_name'], 'custom-cache-v9');
    assert.equal(metadata['llamadart.webgpu.model_source'], 'cache');
    assert.match(metadata['llamadart.webgpu.runtime_notes'], /model_cache_prefetched/);

    // Non-cache facade state must be untouched by the cache-only refresh.
    assert.equal(bridge.getContextSize(), baselineContext);
    assert.equal(
      metadata['llamadart.webgpu.model_bytes'],
      baseline['llamadart.webgpu.model_bytes'],
    );
    assert.equal(metadata['llamadart.webgpu.execution'], 'main-thread');

    runtime.evictModelFromCache = async function evict() {
      this._modelCacheState = 'evicted';
      this._runtimeNotes.push('model_cache_evicted');
      return true;
    };
    assert.equal(await bridge.evictModelFromCache('model.gguf'), true);

    metadata = bridge.getModelMetadata();
    assert.equal(metadata['llamadart.webgpu.model_cache_state'], 'evicted');
    assert.match(metadata['llamadart.webgpu.runtime_notes'], /model_cache_evicted/);
  }],

  // --- Static public-operation coverage and C++ guard contract ---

  ['queue covers every stateful public async method', async () => {
    const prototype = LlamaWebGpuBridge.prototype;
    const publicNames = Object.getOwnPropertyNames(prototype)
      .filter((name) => name !== 'constructor' && !name.startsWith('_'));

    const queued = [];
    for (const name of publicNames) {
      const descriptor = Object.getOwnPropertyDescriptor(prototype, name);
      assert.equal(typeof descriptor.value, 'function', `${name} must be a method`);

      const source = descriptor.value.toString();
      const isAsync = source.startsWith('async');
      const isQueued = source.includes('_runExclusive(');

      if (name === DISPOSE_METHOD) {
        assert.ok(!isAsync, 'dispose must not be async so repeated calls share one promise');
        assert.match(source, /this\._disposed = true/, 'dispose must mark disposal first');
        assert.match(
          source,
          /this\._notifyDisposalWaiters\(\)/,
          'dispose must reject queued work synchronously',
        );
        assert.match(
          source,
          /allowDisposed: true/,
          'dispose teardown must take the queue with allowDisposed',
        );
        continue;
      }

      if (!isAsync) {
        assert.ok(
          UNQUEUED_SYNC_METHODS.has(name),
          `synchronous method ${name} is not accounted for in UNQUEUED_SYNC_METHODS`,
        );
        assert.ok(!isQueued, `synchronous method ${name} must not take the queue`);
        continue;
      }

      if (UNQUEUED_ASYNC_METHODS.has(name)) {
        assert.ok(
          !isQueued,
          `${name} is documented as unqueued (${UNQUEUED_ASYNC_METHODS.get(name)}) but takes the queue`,
        );
        continue;
      }

      assert.ok(
        isQueued,
        `public async method ${name} touches the singleton runtime but is not queued; `
        + 'queue it or justify it in UNQUEUED_ASYNC_METHODS',
      );
      queued.push(name);
    }

    assert.deepEqual(
      queued,
      [
        'loadModelFromUrl',
        'createCompletion',
        'loadMultimodalProjector',
        'unloadMultimodalProjector',
        'getTextToSpeechCapabilities',
        'synthesizeSpeech',
        'tokenize',
        'stateSaveFile',
        'stateLoadFile',
        'stateSaveBytes',
        'stateLoadBytes',
        'detokenize',
        'embed',
        'embedBatch',
        'applyChatTemplate',
      ],
      'the set of queued public operations changed',
    );

    // A queued wrapper must delegate to an unlocked helper, or a nested facade
    // call inside it would wait on the queue slot it already owns.
    for (const name of queued) {
      const source = Object.getOwnPropertyDescriptor(prototype, name).value.toString();
      assert.match(
        source,
        new RegExp(`this\\._${name}Unlocked\\(`),
        `${name} must delegate to _${name}Unlocked to stay re-entrant`,
      );
      assert.equal(
        typeof prototype[`_${name}Unlocked`],
        'function',
        `_${name}Unlocked must exist`,
      );
    }
  }],

  // The C++ guard must reject a second begin before it destroys any state the
  // in-flight generation still owns.

  ['C++ begin_generation guard rejects before destroying state', async () => {
    const core = readFileSync(path.join(rootDir, 'src/llama_webgpu_core.cpp'), 'utf8');
    const start = core.indexOf('int32_t begin_generation_impl(');
    assert.ok(start > 0, 'begin_generation_impl must exist');
    const body = core.slice(start, core.indexOf('\n}\n', start));

    const guardIndex = body.indexOf('if (g_generation_active) {');
    assert.ok(guardIndex > 0, 'begin_generation_impl must reject a re-entrant begin');
    const guardBlock = body.slice(guardIndex, body.indexOf('}', guardIndex) + 1);
    assert.match(
      guardBlock,
      /if \(g_generation_active\) \{\s*\n\s*return -7;\s*\n\s*\}/,
      'the guard must return -7 and nothing else',
    );
    assert.doesNotMatch(
      guardBlock,
      /set_error|clear_error|g_last_/,
      'the guard must not touch g_last_error or any other shared buffer',
    );

    for (const destructive of [
      'clear_error();',
      'g_last_output.clear();',
      'g_last_piece.clear();',
      'end_generation_state();',
    ]) {
      const index = body.indexOf(destructive);
      assert.ok(index > 0, `begin_generation_impl must still contain ${destructive}`);
      assert.ok(
        guardIndex < index,
        `the re-entrancy guard must run before ${destructive}, `
        + `but the guard is at ${guardIndex} and ${destructive} at ${index}`,
      );
    }

    // The stub core used by the behavioural cases above encodes the same
    // contract, so those cases prove the JS queue never reaches this guard.
    const stubCore = createStubCore({ alpha: ['A'] });
    assert.equal(stubCore.ccall('llamadart_webgpu_begin_generation', 'number', [], ['alpha']), 0);
    stubCore.lastOutput = 'PARTIAL';
    stubCore.lastError = 'earlier unrelated failure';
    assert.equal(stubCore.ccall('llamadart_webgpu_begin_generation', 'number', [], ['beta']), -7);
    assert.equal(stubCore.beginRejections, 1);
    assert.equal(
      stubCore.lastOutput,
      'PARTIAL',
      'a rejected begin must leave the active generation output untouched',
    );
    assert.equal(
      stubCore.activePrompt,
      'alpha',
      'a rejected begin must leave the active generation in place',
    );
    assert.equal(
      stubCore.ccall('llamadart_webgpu_last_error', 'string', [], []),
      'earlier unrelated failure',
      'a rejected begin must not overwrite g_last_error',
    );
  }],

  // -7 must surface as a stable JS error that does not read the shared buffer.

  ['re-entrant begin maps to a stable error without reading last_error', async () => {
    const { bridge, core } = createDirectBridge({ alpha: ['A'] });
    core.lastError = 'stale unrelated error text';
    core.activePrompt = 'someone-else';

    await assert.rejects(
      bridge.createCompletion('alpha'),
      (error) => {
        assert.match(error.message, /Generation is already active on this bridge runtime/);
        assert.doesNotMatch(error.message, /stale unrelated error text/);
        return true;
      },
    );
    assert.equal(core.beginRejections, 1);
  }],

  // Unqueued helpers still have to respect the lifecycle: after disposal none of
  // them may dispatch, and none may lazily rebuild a runtime.
  ['unqueued helpers reject after dispose without resurrecting the runtime (direct)', async () => {
    const { bridge } = createDirectBridge();
    let runtimesCreated = 0;
    bridge._createRuntime = () => {
      runtimesCreated += 1;
      return {};
    };

    await bridge.dispose();
    assert.equal(bridge._runtime, null, 'teardown must leave no runtime');

    await assert.rejects(bridge.prefetchModelToCache('model.gguf'), /Bridge has been disposed/);
    await assert.rejects(bridge.evictModelFromCache('model.gguf'), /Bridge has been disposed/);
    await assert.rejects(
      bridge.applyChatTemplate([{ role: 'user', content: 'hi' }], true),
      /Bridge has been disposed/,
    );

    assert.equal(runtimesCreated, 0, 'a disposed bridge must never rebuild a runtime');
    assert.equal(bridge._runtime, null, 'no runtime may be resurrected');
  }],

  ['unqueued helpers reject after dispose without dispatching (worker)', async () => {
    let dispatches = 0;
    let runtimesCreated = 0;
    const bridge = createWorkerBridge({
      _workerProxy: { async dispose() {} },
      _callWorker: async () => {
        dispatches += 1;
        return undefined;
      },
      _createRuntime: () => {
        runtimesCreated += 1;
        return {};
      },
    });

    const disposal = bridge.dispose();
    await disposal;

    await assert.rejects(bridge.prefetchModelToCache('model.gguf'), /Bridge has been disposed/);
    await assert.rejects(bridge.evictModelFromCache('model.gguf'), /Bridge has been disposed/);
    await assert.rejects(
      bridge.applyChatTemplate([{ role: 'user', content: 'hi' }], true),
      /Bridge has been disposed/,
    );

    assert.equal(dispatches, 0, 'a disposed bridge must never dispatch to a worker');
    assert.equal(runtimesCreated, 0, 'a disposed bridge must never rebuild a runtime');
    assert.equal(bridge._workerProxy, null);
    assert.equal(bridge._runtime, null);

    // The lifecycle gate must not disturb dispose() idempotence.
    assert.equal(bridge.dispose(), disposal, 'dispose must still return the same promise');
    assert.equal(bridge._runtime, null, 'repeated dispose must not resurrect a runtime');
  }],

  // --- Real BridgeWorkerProxy, driven through the facade queue ---

  ['real worker proxy dispatches queued operations in FIFO order', async () => {
    await withStubWorkerEnvironment(async ({ workers }) => {
      const bridge = createRealWorkerBridge();
      const driver = workerDriver(bridge._workerProxy, workers[0]);
      driver.ready();

      const first = settleQuietly(bridge.tokenize('a'));
      const second = settleQuietly(bridge.tokenize('b'));
      await driver.settle();

      assert.equal(driver.calls('tokenize').length, 1, 'the second call must wait its turn');

      driver.reply(driver.calls('tokenize')[0].id, [97]);
      assert.deepEqual(await first, [97]);
      await driver.settle();

      const tokenizeCalls = driver.calls('tokenize');
      assert.equal(tokenizeCalls.length, 2, 'the queued call dispatches once the slot frees');
      assert.deepEqual(tokenizeCalls.map((call) => call.args[0]), ['a', 'b'], 'FIFO order');

      driver.reply(tokenizeCalls[1].id, [98]);
      assert.deepEqual(await second, [98]);
      assert.equal(bridge._workerProxy._pending.size, 0, 'no pending requests may remain');
    });
  }],

  ['queued abort neither dispatches nor cancels the real active owner', async () => {
    await withStubWorkerEnvironment(async ({ workers }) => {
      const bridge = createRealWorkerBridge();
      const driver = workerDriver(bridge._workerProxy, workers[0]);
      driver.ready();
      const controller = new AbortController();

      const active = settleQuietly(bridge.createCompletion('alpha'));
      const queued = settleQuietly(
        bridge.createCompletion('beta', { signal: controller.signal }),
      );
      await driver.settle();
      assert.equal(driver.calls('createCompletion').length, 1);

      controller.abort();
      await assert.rejects(queued, (error) => error?.name === 'AbortError');

      assert.equal(
        driver.calls('createCompletion').length,
        1,
        'the aborted queued call must never dispatch',
      );
      assert.equal(driver.calls('cancel').length, 0, 'the active owner must not be cancelled');

      driver.reply(driver.calls('createCompletion')[0].id, 'ALPHA');
      assert.equal(await active, 'ALPHA', 'the active owner completes normally');
    });
  }],

  ['dispose waits for the real active owner, then disposes and revokes once', async () => {
    await withStubWorkerEnvironment(async ({ workers, revoked }) => {
      const bridge = createRealWorkerBridge();
      const proxy = bridge._workerProxy;
      const driver = workerDriver(proxy, workers[0]);
      driver.ready();

      const active = settleQuietly(bridge.createCompletion('alpha'));
      const queued = settleQuietly(bridge.tokenize('b'));
      await driver.settle();

      const disposal = bridge.dispose();
      await assert.rejects(queued, /Bridge has been disposed/);
      await driver.settle();

      assert.equal(driver.calls('dispose').length, 0, 'teardown must wait for the owner');
      assert.equal(workers[0].terminated, 0);

      driver.reply(driver.calls('createCompletion')[0].id, 'ALPHA');
      assert.equal(await active, 'ALPHA');
      await driver.settle();

      const disposeCalls = driver.calls('dispose');
      assert.equal(disposeCalls.length, 1, 'teardown dispatches once the owner released');
      driver.reply(disposeCalls[0].id, null);
      await disposal;

      assert.equal(workers[0].terminated, 1, 'the worker must be terminated exactly once');
      assert.deepEqual(revoked, ['blob:stub-worker'], 'the blob URL must be revoked once');
      assert.equal(bridge._workerProxy, null);

      await bridge.dispose();
      assert.equal(workers[0].terminated, 1, 'repeated dispose must not terminate again');
      assert.deepEqual(revoked, ['blob:stub-worker'], 'repeated dispose must not revoke again');
    });
  }],

  ['a superseded real proxy response cannot overwrite the facade snapshot', async () => {
    await withStubWorkerEnvironment(async ({ workers }) => {
      const bridge = createRealWorkerBridge();
      const staleProxy = bridge._workerProxy;
      const driver = workerDriver(staleProxy, workers[0]);
      driver.ready();

      const pending = settleQuietly(bridge._callWorker('tokenize', ['a']));
      await driver.settle();
      const request = driver.calls('tokenize')[0];

      // The proxy identity changes before its reply arrives.
      bridge._workerProxy = { id: 'fresh' };
      bridge._metadata = { fresh: '1' };
      bridge._contextSize = 2048;

      driver.reply(request.id, [97], {
        metadata: { resurrected: '1' },
        contextSize: 8192,
        supportsVision: true,
      });
      assert.deepEqual(await pending, [97], 'the caller still receives its own result');

      assert.deepEqual(
        bridge.getModelMetadata(),
        { fresh: '1', 'llamadart.webgpu.execution': 'worker' },
        'a superseded proxy must not write into the current snapshot',
      );
      assert.equal(bridge.getContextSize(), 2048);
      assert.equal(bridge.supportsVision(), false);

      bridge._workerProxy = staleProxy;
      staleProxy._worker.terminate();
    });
  }],

  // A worker fallback can supersede the proxy while a public dispose() is
  // already queued behind the failing operation. Disposal must drain that
  // stale proxy as well as any proxy still installed on the facade.
  ['public dispose waits for superseded worker teardown', async () => {
    await withStubWorkerEnvironment(async ({ workers, revoked }) => {
      const bridge = createRealWorkerBridge();
      bridge._emitBridgeWarn = () => {};
      const staleProxy = bridge._workerProxy;
      const driver = workerDriver(staleProxy, workers[0]);
      driver.ready();

      const active = settleQuietly(
        bridge.applyChatTemplate([{ role: 'user', content: 'hi' }], true),
      );
      await driver.settle();
      const activeCall = driver.calls('applyChatTemplate')[0];
      assert.ok(activeCall, 'the worker-backed operation must own the queue');

      const disposal = bridge.dispose();
      let disposalResolved = false;
      disposal.then(() => {
        disposalResolved = true;
      });
      await driver.settle();
      assert.equal(driver.calls('dispose').length, 0, 'teardown waits for the active owner');
      assert.equal(disposalResolved, false, 'queued disposal must still be pending');

      driver.error(activeCall.id, 'worker died');
      assert.equal(await active, 'user: hi\nassistant: ');
      await driver.settle();

      const staleDisposeCall = driver.calls('dispose')[0];
      assert.ok(staleDisposeCall, 'fallback must start stale worker disposal');
      assert.equal(
        disposalResolved,
        false,
        'public disposal must wait while stale worker termination is held',
      );
      assert.equal(workers[0].terminated, 0, 'the stale worker must not terminate early');
      assert.deepEqual(revoked, [], 'the stale worker blob must remain live while pending');

      driver.reply(staleDisposeCall.id, null);
      await disposal;

      assert.equal(workers[0].terminated, 1, 'public disposal waits for worker termination');
      assert.deepEqual(revoked, ['blob:stub-worker'], 'public disposal waits for blob revocation');
      assert.equal(bridge._workerDisposePromise, null, 'the drained stale promise is cleared');
    });
  }],

  // A worker call issued before dispose can reject after teardown. The unqueued
  // call sites must not run worker fallback or rebuild a runtime at that point.
  ['late worker rejection after dispose does not resurrect the runtime', async () => {
    let runtimesCreated = 0;
    let fallbacks = 0;
    let rejectLogLevel = null;
    const bridge = createWorkerBridge({
      _workerProxy: { async dispose() {} },
      _callWorker: () => new Promise((_, reject) => {
        rejectLogLevel = reject;
      }),
      _createRuntime: () => {
        runtimesCreated += 1;
        return createSnapshotRuntime();
      },
    });
    const realDisable = bridge._disableWorkerFallback.bind(bridge);
    bridge._disableWorkerFallback = (error) => {
      fallbacks += 1;
      return realDisable(error);
    };

    // Starts while the bridge is still live.
    bridge.setLogLevel(1);
    await new Promise((resolve) => setTimeout(resolve, 0));

    const disposal = bridge.dispose();
    await disposal;
    assert.equal(bridge._runtime, null, 'teardown must leave no runtime');

    const unhandled = [];
    const onUnhandled = (reason) => unhandled.push(reason);
    process.on('unhandledRejection', onUnhandled);
    try {
      rejectLogLevel(new Error('worker died'));
      await new Promise((resolve) => setTimeout(resolve, 10));
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }

    assert.deepEqual(unhandled, [], 'setLogLevel must not raise an unhandled rejection');
    assert.equal(fallbacks, 0, 'no worker fallback may run after disposal');
    assert.equal(runtimesCreated, 0, 'no runtime may be rebuilt after disposal');
    assert.equal(bridge._runtime, null, 'no runtime may be resurrected');
    await assert.rejects(
      bridge.applyChatTemplate([{ role: 'user', content: 'hi' }], true),
      /Bridge has been disposed/,
    );
    assert.equal(bridge.dispose(), disposal, 'dispose stays identical and idempotent');
  }],


  // A cache-only direct runtime must never overwrite the authoritative snapshot
  // for the active worker model.
  ['worker-mode cache helpers preserve the complete worker snapshot', async () => {
    let workerDispatches = 0;
    let coreCalls = 0;
    const cacheRuntime = {
      _modelSource: 'network',
      _modelCacheState: 'unavailable',
      _modelCacheName: 'default-cache',
      _runtimeNotes: [],
      _core: { ccall: () => { coreCalls += 1; return 0; } },
      async prefetchModelToCache() {
        this._modelCacheState = 'ready';
        this._modelCacheName = 'custom-cache-v9';
        this._modelSource = 'cache';
        this._runtimeNotes.push('model_cache_prefetched');
        return true;
      },
      async evictModelFromCache() {
        this._modelCacheState = 'evicted';
        this._runtimeNotes.push('model_cache_evicted');
        return true;
      },
    };
    const bridge = createWorkerBridge({
      _runtime: cacheRuntime,
      _metadata: {
        'llamadart.webgpu.n_gpu_layers': '99',
        'llamadart.webgpu.model_bytes': '4096',
        'llamadart.webgpu.model_source': 'worker-network',
        'llamadart.webgpu.model_cache_state': 'worker-old',
        'llamadart.webgpu.model_cache_name': 'worker-cache',
        'llamadart.webgpu.runtime_notes': 'worker_threads;worker_fallback',
      },
      _contextSize: 8192,
      _gpuActive: true,
      _backendName: 'WebGPU (Prototype bridge)',
      _supportsVision: true,
      _callWorker: async () => {
        workerDispatches += 1;
        return undefined;
      },
    });

    assert.equal(await bridge.prefetchModelToCache('model.gguf'), true);
    let metadata = bridge.getModelMetadata();
    assert.equal(metadata['llamadart.webgpu.model_cache_state'], 'worker-old');
    assert.equal(metadata['llamadart.webgpu.model_cache_name'], 'worker-cache');
    assert.equal(metadata['llamadart.webgpu.model_source'], 'worker-network');
    assert.equal(metadata['llamadart.webgpu.runtime_notes'], 'worker_threads;worker_fallback');

    // Worker-owned fields must survive untouched.
    assert.equal(metadata['llamadart.webgpu.execution'], 'worker');
    assert.equal(metadata['llamadart.webgpu.n_gpu_layers'], '99');
    assert.equal(metadata['llamadart.webgpu.model_bytes'], '4096');
    assert.equal(bridge.getContextSize(), 8192);
    assert.equal(bridge.isGpuActive(), true);
    assert.equal(bridge.getBackendName(), 'WebGPU (Prototype bridge)');
    assert.equal(bridge.supportsVision(), true);

    assert.equal(await bridge.evictModelFromCache('model.gguf'), true);
    metadata = bridge.getModelMetadata();
    assert.equal(metadata['llamadart.webgpu.model_cache_state'], 'worker-old');
    assert.equal(metadata['llamadart.webgpu.model_cache_name'], 'worker-cache');
    assert.equal(metadata['llamadart.webgpu.model_source'], 'worker-network');
    assert.equal(metadata['llamadart.webgpu.runtime_notes'], 'worker_threads;worker_fallback');
    assert.equal(metadata['llamadart.webgpu.execution'], 'worker');

    assert.equal(coreCalls, 0, 'cache helpers must never call into the core');
    assert.equal(workerDispatches, 0, 'cache helpers must never dispatch to the worker');
  }],

  // An unqueued path that falls back to the direct runtime must leave the
  // synchronous snapshot describing that runtime, not the abandoned worker.
  // An unqueued applyChatTemplate could replace worker topology mid-flight, so a
  // state-bearing response from the active owner would land on a dead proxy.
  ['a state-bearing owner applies its state before applyChatTemplate can fall back', async () => {
    const events = [];
    const runtime = createSnapshotRuntime();
    let resolveLoad = null;
    const bridge = createWorkerBridge({
      _runtime: null,
      _workerProxy: { async dispose() { events.push('worker-disposed'); } },
      _createRuntime: () => runtime,
      _callWorker: (method) => {
        events.push(`dispatch:${method}`);
        if (method === 'applyChatTemplate') {
          return Promise.reject(new Error('worker died'));
        }
        return new Promise((resolve) => {
          resolveLoad = () => {
            events.push('owner-state-applied');
            bridge._contextSize = 8192;
            resolve('LOADED');
          };
        });
      },
    });

    const owner = settleQuietly(bridge.loadModelFromUrl('model.gguf'));
    const template = settleQuietly(
      bridge.applyChatTemplate([{ role: 'user', content: 'hi' }], true),
    );
    await new Promise((resolve) => setTimeout(resolve, 0));

    assert.deepEqual(
      events,
      ['dispatch:loadModelFromUrl'],
      'applyChatTemplate must not dispatch or replace topology while the owner runs',
    );
    assert.notEqual(bridge._workerProxy, null, 'the owner still has its worker');

    resolveLoad();
    assert.equal(await owner, 'LOADED');
    assert.equal(await template, 'user: hi\nassistant: ');

    assert.deepEqual(
      events,
      [
        'dispatch:loadModelFromUrl',
        'owner-state-applied',
        'dispatch:applyChatTemplate',
        'worker-disposed',
      ],
      'the owner applies its state before any fallback replaces the worker',
    );
  }],

  ['applyChatTemplate serializes with other queued work and falls back under the queue', async () => {
    const dispatched = [];
    const runtime = createSnapshotRuntime();
    const bridge = createWorkerBridge({
      _runtime: null,
      _workerProxy: { async dispose() {} },
      _createRuntime: () => runtime,
      _callWorker: async (method) => {
        dispatched.push(method);
        if (method === 'applyChatTemplate') {
          throw new Error('worker died');
        }
        return [98];
      },
    });

    const results = await Promise.all([
      bridge.tokenize('b'),
      bridge.applyChatTemplate([{ role: 'user', content: 'hi' }], true),
    ]);

    assert.deepEqual(results[0], [98]);
    assert.equal(results[1], 'user: hi\nassistant: ', 'the direct fallback result is returned');
    assert.deepEqual(
      dispatched,
      ['tokenize', 'applyChatTemplate'],
      'applyChatTemplate waits its turn in FIFO order',
    );
    assert.equal(bridge._workerProxy, null, 'the fallback ran and replaced topology');

    // Direct mode still works through the queue.
    assert.equal(
      await bridge.applyChatTemplate([{ role: 'user', content: 'again' }], false),
      'user: again',
    );
  }],

  ['setLogLevel worker failure never replaces the worker or the direct runtime', async () => {
    let rejectLogLevel = null;
    let runtimesCreated = 0;
    const bridge = createWorkerBridge({
      _runtime: null,
      _createRuntime: () => {
        runtimesCreated += 1;
        return createSnapshotRuntime();
      },
      _callWorker: () => new Promise((_, reject) => {
        rejectLogLevel = reject;
      }),
    });
    const originalProxy = bridge._workerProxy;

    const unhandled = [];
    const onUnhandled = (reason) => unhandled.push(reason);
    process.on('unhandledRejection', onUnhandled);
    try {
      bridge.setLogLevel(1);
      await new Promise((resolve) => setTimeout(resolve, 0));
      rejectLogLevel(new Error('worker died'));
      await new Promise((resolve) => setTimeout(resolve, 20));
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }

    assert.deepEqual(unhandled, [], 'log-level delivery must not raise an unhandled rejection');
    assert.equal(bridge._workerProxy, originalProxy, 'the worker must not be replaced');
    assert.equal(runtimesCreated, 0, 'no direct runtime may be created');
    assert.equal(bridge._runtime, null);
    assert.equal(bridge._workerFallbackReason, null, 'no fallback may be recorded');
  }],

  ['setLogLevel after dispose throws the stable lifecycle error', async () => {
    const { bridge } = createDirectBridge();
    await bridge.dispose();
    assert.equal(bridge._runtime, null);
    assert.throws(() => bridge.setLogLevel(1), /Bridge has been disposed/);
  }],

  ['dispose returns the identical promise for repeated calls', async () => {
    const { bridge } = createDirectBridge();
    const first = bridge.dispose();
    const second = bridge.dispose();
    assert.equal(first, second, 'dispose must return the same Promise object');
    await first;
    assert.equal(bridge.dispose(), first, 'dispose stays idempotent after teardown');
  }],

  ['re-entrant logger observes the reserved dispose promise', async () => {
    const { bridge, core } = createDirectBridge();
    bridge._runtime._mmProjPath = '/mmproj.gguf';
    const innerCcall = core.ccall.bind(core);
    core.ccall = (name, retType, argTypes, args) => {
      if (name === 'llamadart_webgpu_mmproj_free') {
        return 1;
      }
      return innerCcall(name, retType, argTypes, args);
    };

    let reentrantDispose = null;
    bridge._runtime._config.logger = {
      warn: () => {
        reentrantDispose = bridge.dispose();
      },
    };

    const first = bridge.dispose();
    assert.equal(
      reentrantDispose,
      first,
      'a synchronous logger callback must observe the stable public dispose promise',
    );
    await first;
  }],

  // A rejected predecessor must release the queue in worker mode too.
  ['a rejected predecessor releases the queue (worker)', async () => {
    const dispatched = [];
    const bridge = createWorkerBridge({
      _callWorker: async (method, args) => {
        dispatched.push(`${method}:${JSON.stringify(args?.[0] ?? null)}`);
        if (method === 'tokenize') {
          throw new Error('worker tokenize exploded');
        }
        return 'OK';
      },
      _disableWorkerFallback: () => {
        throw new Error('worker tokenize exploded');
      },
    });

    const failing = settleQuietly(bridge.tokenize('a'));
    const following = settleQuietly(bridge.createCompletion('alpha'));

    await assert.rejects(failing, /worker tokenize exploded/);
    assert.equal(await following, 'OK', 'the queue must release after a rejection');
    assert.deepEqual(dispatched, ['tokenize:"a"', 'createCompletion:"alpha"']);
  }],

  // Queues are per instance; one bridge must not block another.
  ['operation queues are per bridge instance', async () => {
    const slow = createDirectBridge({ alpha: Array.from({ length: 80 }, () => 'A') });
    const fast = createDirectBridge({ beta: ['B'] });

    const slowCompletion = settleQuietly(slow.bridge.createCompletion('alpha'));
    assert.equal(
      await fast.bridge.createCompletion('beta'),
      'B',
      'a second bridge instance must not wait on the first instance queue',
    );

    slow.bridge.cancel();
    await slowCompletion;
  }],

];

const failures = [];
for (const [name, run] of CASES) {
  try {
    let timer = null;
    await Promise.race([
      run(),
      new Promise((_, reject) => {
        timer = setTimeout(
          () => reject(new Error('timed out after 2000ms; the operation queue deadlocked')),
          2000,
        );
      }),
    ]).finally(() => {
      if (timer != null) {
        clearTimeout(timer);
      }
    });
  } catch (error) {
    failures.push(`${name}: ${error?.message || error}`);
  }
}

if (failures.length > 0) {
  console.error(`${failures.length}/${CASES.length} bridge operation queue cases failed:`);
  for (const failure of failures) {
    console.error(`  - ${failure}`);
  }
  process.exitCode = 1;
} else {
  console.log(`Bridge operation queue tests passed (${CASES.length} cases)`);
}
