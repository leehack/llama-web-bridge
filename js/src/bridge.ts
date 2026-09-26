// Public bridge facade: picks the worker or direct runtime and owns the operation queue.

import { createAbortError } from './internal/abort.ts';
import { NO_COMPLETION_CAPABILITIES, resolveCompletionSamplingOptions } from './internal/completion_options.ts';
import { BRIDGE_DISPOSED_MESSAGE, INVALID_GRAMMAR_ERROR_TEXT } from './internal/constants.ts';
import {
  DECISION_API_VERSION,
  decisionHandleFrom,
  normalizeDecisionSequences,
} from './internal/decision.ts';
import { logLevelForName, logThresholdForConfiguredLevel } from './internal/logging.ts';
import {
  LORA_API_VERSION,
  LORA_LOAD_ABORT_MESSAGE,
  loraHandleFrom,
  loraScaleFrom,
  staleLoraAdapterError,
} from './internal/lora.ts';
import { cloneModelSource, hasModelSource, normalizeAbsoluteUrl } from './internal/model_source.ts';
import { bufferSourceBytes, toUint8Array } from './internal/typed_values.ts';
import { LlamaWebGpuBridgeRuntime } from './runtime.ts';
import { emptyBridgeState, serializeWorkerError } from './worker_protocol.ts';
import { BridgeWorkerProxy } from './worker_proxy.ts';
import type { ModelSource } from './internal/model_source.ts';
import type { LogMethod } from './internal/types.ts';
import type {
  BridgeProgressEvent,
  CompletionCapabilities,
  CompletionOptions,
  CompletionUsage,
  DecisionCapabilities,
  DecisionHeadInfo,
  DecisionHeadOptions,
  DecisionOutput,
  DecisionSequence,
  EmbedOptions,
  LlamaWebGpuBridgeConfig,
  LoadModelOptions,
  LoraAdapterCapabilities,
  LoraAdapterInfo,
  LoraAdapterLoadOptions,
  ModelMetadata,
  NextTokenScoreOptions,
  NextTokenScores,
  LlamaWebGpuBridge as PublicLlamaWebGpuBridge,
  StateLoadResult,
  TextToSpeechCapabilities,
  TextToSpeechOptions,
  TextToSpeechProgress,
  TextToSpeechResult,
} from './llama_webgpu_bridge.d.ts';
import type {
  BridgeStateSnapshot,
  WorkerEventHandler,
  WorkerRequestError,
  WorkerResponse,
} from './worker_protocol.ts';

// Where a queued operation is in its lifecycle. `skipped` and `disposed` end
// an operation that never started; the rest follow one that took the slot.
type BridgeOperationState =
  | 'queued'
  | 'skipped'
  | 'disposed'
  | 'starting'
  | 'running'
  | 'recovering'
  | 'cancelling'
  | 'settling'
  | 'cancelled'
  | 'failed'
  | 'completed';

// One call's record in the operation queue. It captures every worker
// generation, proxy, and runtime the call touched, so cancellation reaches
// exactly those rather than whatever topology is current.
interface BridgeOperation {
  id: number;
  kind: string;
  state: BridgeOperationState;
  callerSignal: AbortSignal | null;
  abortController: AbortController | null;
  cancelRequested: boolean;
  cancelReason: string | null;
  workerTimeoutCancellationSent: boolean;
  workerGeneration: number;
  workerGenerations: Set<number>;
  workerProxy: BridgeWorkerProxy | null;
  // Set once the operation starts on a worker, then follows each replacement.
  currentWorkerProxy?: BridgeWorkerProxy;
  runtime: LlamaWebGpuBridgeRuntime | null;
  workerProxies: Set<BridgeWorkerProxy>;
  runtimes: Set<LlamaWebGpuBridgeRuntime>;
  modelSource: ModelSource | null;
  modelOptions: LoadModelOptions | null;
  projectorSource: string | null;
  recoveryAttempted: boolean;
  generationStarted: boolean;
  retirements: Set<Promise<void>>;
  deferredShadowState: BridgeStateSnapshot | null;
}

// How `_runExclusive` admits and labels one operation.
interface ExclusiveRunOptions {
  signal?: AbortSignal | null;
  abortMessage?: string;
  allowDisposed?: boolean;
  kind?: string;
}

// The last worker model load's proxy and reply, read after a failed load.
interface WorkerLoadAttempt {
  proxy: BridgeWorkerProxy | null;
  state: BridgeStateSnapshot | null;
  loaded: boolean;
}

// Completion options plus the recursion guard the empty-response retry sets.
interface BridgeCompletionOptions extends CompletionOptions {
  __llamadartEmptyRetryAttempted?: boolean;
}

// What worker-fallback recovery reads: a completion's options, or the flags
// text-to-speech recovery passes.
interface WorkerFallbackOptions {
  parts?: readonly unknown[];
  signal?: AbortSignal | null;
  _llamadartForceRuntimeReload?: boolean;
  _llamadartTextToSpeech?: boolean;
}

// A worker token event's payload: text, or bytes when a piece splits UTF-8.
interface WorkerTokenPayload {
  pieceText?: unknown;
  piece?: unknown;
  currentText?: unknown;
}

// A facade decision handle's target: the head's handle in the worker or
// runtime that loaded it.
interface DecisionHeadEntry {
  owner: object;
  handle: number;
}

// A facade LoRA handle's target: the adapter's handle in the worker or runtime
// that loaded it, and what reloads it into a replacement. `source` is null when
// workers are disabled, since the owner then never changes.
interface LoraAdapterEntry {
  owner: object | null;
  handle: number;
  source: string | Uint8Array | null;
  cacheOptions: Pick<LoraAdapterLoadOptions, 'useCache' | 'cacheName'>;
}

// The runtime and worker LoRA calls that take only numbers.
type LoraOwnerMethod = 'setLoraAdapter' | 'removeLoraAdapter' | 'clearLoraAdapters';

// A decision head download's progress, as `DecisionHeadOptions.onProgress` takes it.
type DecisionHeadProgress = Parameters<NonNullable<DecisionHeadOptions['onProgress']>>[0];

// The fallback reason published on globalThis for diagnostics.
interface WorkerFallbackGlobals {
  __llamadartBridgeWorkerFallbackReason?: string;
}

export class LlamaWebGpuBridge implements PublicLlamaWebGpuBridge {
  declare _config: LlamaWebGpuBridgeConfig;
  declare _runtime: LlamaWebGpuBridgeRuntime | null;
  declare _workerProxy: BridgeWorkerProxy | null;
  declare _workerGeneration: number;
  declare _workerDisposePromise: Promise<void> | null;
  declare _retiringWorkerDisposals: Set<Promise<void>>;
  declare _retiredWorkerProxies: WeakSet<object>;
  declare _workerFallbackReason: string | null;
  declare _metadata: ModelMetadata;
  declare _contextSize: number;
  declare _gpuActive: boolean;
  declare _backendName: string;
  declare _supportsVision: boolean;
  declare _supportsAudio: boolean;
  declare _loadedModelUrl: ModelSource | null;
  declare _loadedModelOptions: LoadModelOptions | null;
  declare _loadedMmProjUrl: string | null;
  declare _multimodalWorkerCpuMode: boolean;
  declare _workerModelMissing: boolean;
  declare _bridgeWarnRecent: Map<string, number>;
  declare _operationQueueTail: Promise<unknown> | null;
  declare _activeOperation: BridgeOperation | null;
  declare _nextOperationId: number;
  declare _lifecycleState: 'open' | 'disposing' | 'disposed';
  declare _shadowStateTransactionDepth: number;
  declare _deferredShadowState: BridgeStateSnapshot | null;
  declare _disposed: boolean;
  declare _disposePromise: Promise<void> | null;
  declare _disposalWaiters: Set<() => void>;
  declare _decisionHeads: Map<number, DecisionHeadEntry>;
  declare _nextDecisionHandle: number;
  declare _loraAdapters: Map<number, LoraAdapterEntry>;
  declare _activeLoraScales: Map<number, number>;
  declare _nextLoraHandle: number;

  static supportsSafariAdaptiveGpu =
    LlamaWebGpuBridgeRuntime.supportsSafariAdaptiveGpu === true;

  static supportsCompletionUsage = true;

  constructor(config: LlamaWebGpuBridgeConfig = {}) {
    this._config = config;
    this._runtime = null;
    this._workerProxy = null;
    this._workerGeneration = 0;
    this._workerDisposePromise = null;
    this._retiringWorkerDisposals = new Set();
    this._retiredWorkerProxies = new WeakSet();
    this._workerFallbackReason = null;

    this._metadata = {};
    this._contextSize = 0;
    this._gpuActive = false;
    this._backendName = 'WASM (Prototype bridge)';
    this._supportsVision = false;
    this._supportsAudio = false;
    this._loadedModelUrl = null;
    this._loadedModelOptions = null;
    this._loadedMmProjUrl = null;
    this._multimodalWorkerCpuMode = false;
    // True while the current worker is a replacement that has not yet loaded
    // _loadedModelUrl. A replacement worker always starts without a model.
    this._workerModelMissing = false;
    this._bridgeWarnRecent = new Map();
    this._operationQueueTail = null;
    this._activeOperation = null;
    this._nextOperationId = 0;
    this._lifecycleState = 'open';
    this._shadowStateTransactionDepth = 0;
    this._deferredShadowState = null;
    this._disposed = false;
    this._disposePromise = null;
    this._disposalWaiters = new Set();
    // Facade decision handles map to { owner, handle } so a head loaded by a
    // retired worker or runtime can never resolve to a newer owner's head.
    this._decisionHeads = new Map();
    this._nextDecisionHandle = 1;
    // Facade LoRA handles, and the scales of the applied ones in the order
    // first set. Both describe the current model and are cleared with it.
    this._loraAdapters = new Map();
    this._activeLoraScales = new Map();
    this._nextLoraHandle = 1;

    if (this._shouldUseWorker()) {
      try {
        this._workerProxy = this._createWorkerProxy();
      } catch (error) {
        this._disableWorkerFallback(error);
      }
    }

    if (this._workerProxy) {
      this._workerGeneration = 1;
    }

    if (!this._workerProxy) {
      this._runtime = this._createRuntime();
      this._captureDirectRuntimeState();
    }
  }

  _createRuntime() {
    return new LlamaWebGpuBridgeRuntime({
      ...this._config,
      disableWorker: true,
    });
  }

  _operationSignal(operation = this._activeOperation) {
    return operation?.abortController?.signal || null;
  }

  _recordWorkerGeneration(operation = this._activeOperation, proxy = this._workerProxy) {
    if (!operation) {
      return;
    }

    const generation = this._workerGeneration || 0;
    operation.workerGeneration = generation;
    operation.workerGenerations?.add(generation);
    if (proxy) {
      operation.workerProxy = proxy;
      operation.currentWorkerProxy = proxy;
      operation.workerProxies?.add(proxy);
    }
  }

  _advanceWorkerGeneration() {
    this._workerGeneration = (this._workerGeneration || 0) + 1;
    this._recordWorkerGeneration();
    return this._workerGeneration;
  }

  _isOperationCurrent(operation: BridgeOperation | null) {
    return !!(
      operation
      && this._activeOperation === operation
      && this._lifecycleState !== 'disposed'
      && operation.state !== 'cancelling'
      && operation.state !== 'cancelled'
      && operation.state !== 'disposed'
    );
  }

  _isOperationCallbackCurrent(operation: BridgeOperation | null, proxy: BridgeWorkerProxy) {
    return this._isOperationCurrent(operation) && proxy === this._workerProxy;
  }

  _sendWorkerCancel(proxy: Partial<BridgeWorkerProxy> | null) {
    if (!proxy) {
      return;
    }

    if (typeof proxy.call !== 'function') {
      // Test doubles and legacy embedders may provide only the facade-level
      // `_callWorker` hook. Keep cancellation out-of-band for those shims while
      // never redirecting a stale captured proxy to a replacement.
      if (proxy === this._workerProxy && typeof this._callWorker === 'function') {
        try {
          Promise.resolve(this._callWorker('cancel', [])).catch(() => {});
        } catch (_) {
          // Best-effort cancellation.
        }
      }
      return;
    }

    try {
      Promise.resolve(proxy.call('cancel', [], null)).catch(() => {});
    } catch (_) {
      // Best-effort cancellation must never throw into the caller.
    }
  }

  _cancelOperation(operation: BridgeOperation | null, reason = 'user') {
    if (!operation) {
      return;
    }
    if (reason === 'worker-timeout') {
      if (operation.cancelRequested || operation.workerTimeoutCancellationSent) {
        return;
      }
      operation.workerTimeoutCancellationSent = true;
      operation.cancelReason = reason;
      operation.state = 'recovering';
      const proxies = new Set(operation.workerProxies || []);
      if (operation.workerProxy) {
        proxies.add(operation.workerProxy);
      }
      for (const proxy of proxies) {
        this._sendWorkerCancel(proxy);
      }
      return;
    }
    if (operation.cancelRequested) {
      return;
    }

    operation.cancelRequested = true;
    operation.cancelReason = reason;
    operation.state = 'cancelling';
    try {
      operation.abortController?.abort();
    } catch (_) {
      // Ignore best-effort abort-controller failures.
    }

    const proxies = new Set(operation.workerProxies || []);
    if (operation.workerProxy) {
      proxies.add(operation.workerProxy);
    }
    for (const proxy of proxies) {
      this._sendWorkerCancel(proxy);
    }

    const runtimes = new Set(operation.runtimes || []);
    if (operation.runtime) {
      runtimes.add(operation.runtime);
    }
    for (const runtime of runtimes) {
      try {
        runtime?.cancel?.();
      } catch (_) {
        // Best-effort cancellation must not mask the active operation.
      }
    }
  }

  _isOperationCancellation(error: unknown, operation = this._activeOperation) {
    if (operation?.callerSignal?.aborted) {
      return true;
    }
    if (
      operation?.cancelRequested
      && operation.cancelReason !== 'worker-timeout'
    ) {
      return true;
    }
    return (error as Error | null)?.name === 'AbortError' && !this._isWorkerTimeoutError(error);
  }

  _throwIfOperationCancelled(
    error: unknown,
    message = 'Bridge operation was cancelled.',
    operation = this._activeOperation,
  ) {
    if (this._isOperationCancellation(error, operation)) {
      throw createAbortError(message);
    }
  }

  /**
   * Guards the success path: a worker that ignored cancellation and settled
   * with a full result must still reject once the caller's signal was aborted,
   * and must not commit that result. Only the caller's signal counts here.
   * A bare `cancel()` stays the documented best-effort control signal, so a
   * result that still arrived is delivered rather than discarded.
   */
  _throwIfCallerCancelled(
    message = 'Bridge operation was cancelled.',
    operation = this._activeOperation,
  ) {
    if (operation?.callerSignal?.aborted) {
      throw createAbortError(message);
    }
  }

  _retireWorkerProxy(proxy: BridgeWorkerProxy | null) {
    if (!proxy || typeof proxy !== 'object') {
      return Promise.resolve();
    }

    if (this._retiredWorkerProxies?.has(proxy)) {
      return Promise.resolve();
    }
    this._retiredWorkerProxies?.add(proxy);

    const operation = this._activeOperation;
    operation?.workerProxies?.add(proxy);
    let disposal: Promise<void>;
    try {
      disposal = Promise.resolve(proxy.dispose?.()).catch(() => {});
    } catch (_) {
      disposal = Promise.resolve();
    }

    this._retiringWorkerDisposals ??= new Set();
    this._retiringWorkerDisposals.add(disposal);
    this._workerDisposePromise = disposal;
    operation?.retirements?.add(disposal);

    disposal.then(
      () => {
        this._retiringWorkerDisposals.delete(disposal);
        if (this._workerDisposePromise === disposal && this._retiringWorkerDisposals.size === 0) {
          this._workerDisposePromise = null;
        }
      },
      () => {
        this._retiringWorkerDisposals.delete(disposal);
        if (this._workerDisposePromise === disposal && this._retiringWorkerDisposals.size === 0) {
          this._workerDisposePromise = null;
        }
      },
    );
    return disposal;
  }

  async _waitForWorkerDisposal() {
    this._retiringWorkerDisposals ??= new Set();
    while (this._retiringWorkerDisposals.size > 0) {
      const pending = [...this._retiringWorkerDisposals];
      await Promise.all(pending);
    }
    this._workerDisposePromise = null;
  }

  /**
   * llama.cpp exposes one model/context per bridge instance and publishes
   * results through process-global C buffers (`last_output`, `last_tokens_json`,
   * `last_detokenized`, `last_embedding_json`, ...). Every async call that reads
   * or mutates that runtime therefore has to own it exclusively for its whole
   * duration: a second caller interleaving between the work ccall and the buffer
   * read would otherwise observe the other caller's result. This FIFO queue is
   * the single-writer lock, shared by the worker and direct-runtime paths.
   *
   * A caller whose signal is already aborted, or that aborts while still queued,
   * never runs and never dispatches; its slot is skipped in turn without
   * touching the unrelated operation that currently owns the runtime. Once an
   * operation has started, its signal is routed to `_cancelOperation` so the
   * cancel reaches the worker generation and runtime that operation captured
   * rather than whatever topology happens to be current.
   */
  _runExclusive<T>(
    run: (operation: BridgeOperation) => Promise<T>,
    options: ExclusiveRunOptions = {},
  ): Promise<T> {
    const signal = options.signal || null;
    const abortMessage = options.abortMessage || 'Bridge operation was cancelled.';
    const allowDisposed = options.allowDisposed === true;
    const abortError = () => createAbortError(abortMessage);
    const lifecycleState = this._lifecycleState || (this._disposed ? 'disposed' : 'open');

    if (!allowDisposed && (this._disposed || lifecycleState !== 'open')) {
      return Promise.reject(new Error(BRIDGE_DISPOSED_MESSAGE));
    }
    if (signal?.aborted) {
      return Promise.reject(abortError());
    }

    const operation: BridgeOperation = {
      id: ++this._nextOperationId,
      kind: options.kind || 'bridge-operation',
      state: 'queued',
      callerSignal: signal,
      abortController: typeof AbortController === 'function' ? new AbortController() : null,
      cancelRequested: false,
      cancelReason: null,
      workerTimeoutCancellationSent: false,
      workerGeneration: this._workerGeneration || 0,
      workerGenerations: new Set([this._workerGeneration || 0]),
      workerProxy: this._workerProxy || null,
      runtime: this._runtime || null,
      workerProxies: new Set(this._workerProxy ? [this._workerProxy] : []),
      runtimes: new Set(this._runtime ? [this._runtime] : []),
      modelSource: null,
      modelOptions: null,
      projectorSource: null,
      recoveryAttempted: false,
      generationStarted: false,
      retirements: new Set(),
      deferredShadowState: null,
    };

    const predecessor = this._operationQueueTail;
    let releaseSuccessor: () => void = () => {};
    const queueSlot = new Promise((resolve) => {
      releaseSuccessor = () => resolve(undefined);
    });
    this._operationQueueTail = queueSlot;

    let started = false;
    let removeAbortListener: (() => void) | null = null;
    let removeDisposalWaiter: (() => void) | null = null;
    const detachWatchers = () => {
      removeAbortListener?.();
      removeDisposalWaiter?.();
      removeAbortListener = null;
      removeDisposalWaiter = null;
    };
    const watchesAbort = signal != null && typeof signal.addEventListener === 'function';
    const watchesDisposal = !allowDisposed;
    const ordered = (async () => {
      if (predecessor) {
        await predecessor;
      }
      if (signal?.aborted) {
        operation.state = 'skipped';
        throw abortError();
      }
      if (!allowDisposed && (this._disposed || this._lifecycleState === 'disposing')) {
        operation.state = 'disposed';
        throw new Error(BRIDGE_DISPOSED_MESSAGE);
      }

      started = true;
      // The watcher below assigns it after this function was created.
      (removeDisposalWaiter as (() => void) | null)?.();
      removeDisposalWaiter = null;
      operation.state = 'starting';
      operation.workerProxy = this._workerProxy || null;
      operation.workerGeneration = this._workerGeneration || 0;
      operation.workerGenerations.add(operation.workerGeneration);
      operation.runtime = this._runtime || null;
      if (operation.workerProxy) {
        operation.workerProxies.add(operation.workerProxy);
        operation.currentWorkerProxy = operation.workerProxy;
      }
      if (operation.runtime) {
        operation.runtimes.add(operation.runtime);
      }
      this._activeOperation = operation;
      operation.state = 'running';
      let operationFailed = false;
      try {
        return await run(operation);
      } catch (error) {
        operationFailed = true;
        throw error;
      } finally {
        let terminalState: BridgeOperationState = operation.cancelRequested
          ? 'cancelled'
          : (operationFailed ? 'failed' : 'completed');
        operation.state = terminalState === 'completed' ? 'settling' : terminalState;
        try {
          // Refresh the snapshot on success and failure alike, while this
          // operation still owns the slot, so synchronous getters never have to
          // read the runtime themselves.
          this._captureDirectRuntimeState();
        } catch (error) {
          terminalState = 'failed';
          throw error;
        } finally {
          if (this._activeOperation === operation) {
            this._activeOperation = null;
          }
          operation.state = terminalState;
        }
      }
    })();

    const release = () => {
      if (this._operationQueueTail === queueSlot) {
        this._operationQueueTail = null;
      }
      releaseSuccessor();
    };
    ordered.then(release, release);

    if (!watchesAbort && !watchesDisposal) {
      return ordered;
    }

    // Reject the caller as soon as it is cancelled or the bridge is disposed
    // while still waiting. The slot stays in the chain and is skipped in turn.
    const abandonedWhileQueued = new Promise<never>((_, reject) => {
      if (watchesAbort) {
        const onAbort = () => {
          if (!started) {
            operation.state = 'skipped';
            reject(abortError());
            return;
          }
          this._cancelOperation(operation, 'signal');
        };
        signal.addEventListener('abort', onAbort, { once: true });
        removeAbortListener = () => signal.removeEventListener('abort', onAbort);
        if (signal.aborted) {
          onAbort();
        }
      }

      if (watchesDisposal && !started) {
        removeDisposalWaiter = this._addDisposalWaiter(() => {
          if (!started) {
            operation.state = 'disposed';
            reject(new Error(BRIDGE_DISPOSED_MESSAGE));
          }
        });
      }
    });

    const raced = Promise.race([ordered, abandonedWhileQueued]);
    raced.then(detachWatchers, detachWatchers);
    return raced;
  }

  /**
   * Registers a disposal listener for the lifetime of one pending operation and
   * returns its remover. A listener set is used rather than a long-lived promise
   * because a promise reaction cannot be detached.
   */
  _addDisposalWaiter(onDisposed: () => void) {
    const waiters = this._disposalWaiters;
    waiters.add(onDisposed);
    return () => {
      waiters.delete(onDisposed);
    };
  }

  /**
   * True while `operation` is the non-dispose owner of the queue slot. Teardown
   * is queued behind it, so it is allowed to finish the work it already started
   * even once disposal has been requested.
   */
  _isSettleFirstOwner(operation = this._activeOperation) {
    return !!(
      operation
      && this._activeOperation === operation
      && operation.kind !== 'dispose'
      && operation.state !== 'cancelled'
      && operation.state !== 'disposed'
    );
  }

  /**
   * Lifecycle gate for the helpers that stay off the queue. They must not
   * dispatch to a torn-down worker, and must not lazily recreate the runtime
   * disposal just released. `disposing` still admits the operation that already
   * owns the slot, because disposal settles it first rather than interrupting
   * it; once teardown has run, every caller is rejected.
   */
  _throwIfDisposed() {
    if (this._lifecycleState === 'disposed') {
      throw new Error(BRIDGE_DISPOSED_MESSAGE);
    }
    if (
      (this._disposed || this._lifecycleState === 'disposing')
      && !this._isSettleFirstOwner()
    ) {
      throw new Error(BRIDGE_DISPOSED_MESSAGE);
    }
  }

  _notifyDisposalWaiters() {
    const waiters = this._disposalWaiters;
    if (waiters.size === 0) {
      return;
    }
    this._disposalWaiters = new Set();
    for (const waiter of waiters) {
      waiter();
    }
  }

  /**
   * Mirror the direct runtime's model/context state into facade fields so the
   * synchronous getters can answer from cache. Reading them straight from the
   * runtime would ccall into the singleton core outside the queue, racing
   * whichever async operation currently owns it.
   */
  _captureDirectRuntimeState() {
    // Worker mode owns the snapshot through _applyShadowState; a stale direct
    // runtime must never overwrite it.
    if (this._workerProxy || !this._runtime) {
      return;
    }

    try {
      this._metadata = this._runtime.getModelMetadata();
      this._contextSize = this._runtime.getContextSize();
      this._gpuActive = this._runtime.isGpuActive();
      this._backendName = this._runtime.getBackendName();
      this._supportsVision = this._runtime.supportsVision();
      this._supportsAudio = this._runtime.supportsAudio();
    } catch (_) {
      // Keep the previously cached snapshot when the core is unavailable.
    }
  }

  /**
   * prefetch/evict stay off the queue because they only touch Cache Storage and
   * the network. In direct mode their plain-JS cache bookkeeping is copied into
   * the facade snapshot. A worker-backed snapshot remains wholly worker-owned so
   * a separate cache-only runtime cannot misdescribe the active worker model.
   */
  _refreshCacheMetadataSnapshot() {
    const runtime = this._runtime;
    if (this._workerProxy || !runtime) {
      return;
    }

    this._metadata = {
      ...(this._metadata || {}),
      'llamadart.webgpu.model_source': String(runtime._modelSource ?? ''),
      'llamadart.webgpu.model_cache_state': String(runtime._modelCacheState ?? ''),
      'llamadart.webgpu.model_cache_name': String(runtime._modelCacheName ?? ''),
      'llamadart.webgpu.runtime_notes': Array.isArray(runtime._runtimeNotes)
        ? runtime._runtimeNotes.join(';')
        : '',
    };
  }

  _sanitizeModelLoadOptions(options: LoadModelOptions = {}) {
    const source: LoadModelOptions = options && typeof options === 'object' ? options : {};
    const sanitized = { ...source };
    delete sanitized.progressCallback;
    delete sanitized.signal;
    return sanitized;
  }

  _createCpuSafeMultimodalLoadOptions(options: LoadModelOptions = {}) {
    const sanitized = this._sanitizeModelLoadOptions(options);
    sanitized.nGpuLayers = 0;

    if (Number.isFinite(Number(sanitized.nCtx)) && Number(sanitized.nCtx) > 4096) {
      sanitized.nCtx = 4096;
    }

    if (!Number.isFinite(Number(sanitized.nThreads)) || Number(sanitized.nThreads) <= 0) {
      sanitized.nThreads = 4;
    } else {
      sanitized.nThreads = Math.min(4, Math.max(1, Math.trunc(Number(sanitized.nThreads))));
    }

    sanitized.nThreadsBatch = sanitized.nThreads;

    if (!Number.isFinite(Number(sanitized.nBatch)) || Number(sanitized.nBatch) <= 0) {
      sanitized.nBatch = 128;
    } else {
      sanitized.nBatch = Math.min(128, Math.max(32, Math.trunc(Number(sanitized.nBatch))));
    }

    if (!Number.isFinite(Number(sanitized.nUbatch)) || Number(sanitized.nUbatch) <= 0) {
      sanitized.nUbatch = Math.min(64, sanitized.nBatch as number);
    } else {
      sanitized.nUbatch = Math.min(
        sanitized.nBatch as number,
        Math.min(64, Math.max(1, Math.trunc(Number(sanitized.nUbatch)))),
      );
    }

    return sanitized;
  }

  _rememberLoadedModel(url: ModelSource, options: LoadModelOptions = {}) {
    const normalizedUrl = cloneModelSource(url);
    if (!hasModelSource(normalizedUrl)) {
      return;
    }

    this._loadedModelUrl = normalizedUrl;
    this._loadedModelOptions = this._sanitizeModelLoadOptions(options);
    this._loadedMmProjUrl = null;
    this._forgetLoraAdapters();
    this._multimodalWorkerCpuMode = this._workerProxy != null;
    this._workerModelMissing = false;
    if (this._activeOperation) {
      this._activeOperation.modelSource = cloneModelSource(normalizedUrl);
      this._activeOperation.modelOptions = { ...this._loadedModelOptions };
      this._activeOperation.projectorSource = null;
    }
  }

  _forgetLoadedModel() {
    this._loadedModelUrl = null;
    this._loadedModelOptions = null;
    this._loadedMmProjUrl = null;
    this._forgetLoraAdapters();
    this._multimodalWorkerCpuMode = false;
    this._workerModelMissing = false;
  }

  /**
   * A failed load can leave its target without any model: the runtime drops
   * the previous model once the new download starts. Keep the facade's model
   * source only while the target still reports a loaded model, so recovery
   * never reloads a model the target no longer holds.
   *
   * A worker's reply is read from `attempt`, not the facade snapshot: a
   * cancelled operation no longer accepts worker state, so the snapshot can
   * still describe a model the worker already dropped. Only the bookkeeping is
   * corrected here; a cancelled load never publishes the worker's snapshot.
   */
  _syncLoadedModelAfterFailedLoad(attempt: WorkerLoadAttempt) {
    const proxy = this._workerProxy;
    if (!proxy) {
      if (!(Number(this._runtime?._modelBytes) > 0)) {
        this._forgetLoadedModel();
      }
      return;
    }

    const fromCurrentWorker = attempt.proxy === proxy;
    // The worker loaded the new model, but the caller cancelled before the
    // facade recorded it: the worker no longer holds the remembered model.
    const replacedModel = fromCurrentWorker && attempt.loaded;
    const state = fromCurrentWorker ? attempt.state : null;
    const modelBytes = Number(
      (state ? state.metadata : this._metadata)?.['llamadart.webgpu.model_bytes'],
    );
    if (replacedModel || !(modelBytes > 0)) {
      this._forgetLoadedModel();
    }
  }

  _rememberLoadedMmProj(url: string) {
    const normalizedUrl = String(url || '').trim();
    if (normalizedUrl.length === 0) {
      return;
    }

    this._loadedMmProjUrl = normalizedUrl;
    if (this._activeOperation) {
      this._activeOperation.projectorSource = normalizedUrl;
    }
  }

  _hasMediaParts(options: WorkerFallbackOptions = {}) {
    return Array.isArray(options?.parts) && options.parts.length > 0;
  }

  async _replaceWorkerProxyForMultimodalCpuMode() {
    this._throwIfDisposed();
    const hadWorkerProxy = this._workerProxy != null;
    if (hadWorkerProxy) {
      const staleProxy = this._workerProxy;
      // Invalidate the old generation before detaching its proxy. Cancellation
      // still retains the old proxy through the operation's workerProxies set,
      // while all subsequent requests receive the new generation.
      this._advanceWorkerGeneration();
      this._workerProxy = null;
      this._retireWorkerProxy(staleProxy);
      await this._waitForWorkerDisposal();
      this._throwIfDisposed();
    }

    let replacement;
    try {
      replacement = this._createWorkerProxy();
    } catch (error) {
      // The constructor throws synchronously when no worker can be created
      // (for example a CSP SecurityError), and any previous proxy is already
      // detached. Fall back before rethrowing so an open bridge keeps a direct
      // runtime whichever catch the caller takes; the fallback itself refuses
      // to recreate a runtime once disposal owns teardown.
      this._disableWorkerFallback(error);
      throw error;
    }
    try {
      // Same gate as every other helper: the owner that already holds the slot
      // may still install its replacement while disposal is queued behind it,
      // and queued teardown retires whatever is current. Any other caller is
      // rejected and must not leak the proxy it just constructed.
      this._throwIfDisposed();
    } catch (error) {
      this._retireWorkerProxy(replacement);
      throw error;
    }
    if (!hadWorkerProxy) {
      this._advanceWorkerGeneration();
    }
    this._workerProxy = replacement;
    this._recordWorkerGeneration(this._activeOperation, replacement);
    this._multimodalWorkerCpuMode = false;
    this._workerModelMissing = true;
  }

  _isRecoverableWorkerFsError(error: unknown) {
    const text = serializeWorkerError(error).toLowerCase();
    // An HTTP status from the model host ("404 Not Found") is deterministic: a
    // fresh worker gets the same answer, so a restart only discards the model
    // the current worker holds.
    if (text.includes('model shard:')) {
      return false;
    }
    return (
      text.includes('fs error')
      || text.includes('no such file')
      || text.includes('not found')
      || text.includes('invalid argument')
      || text.includes('timed out')
    );
  }

  _isWorkerRequestTimeoutError(error: unknown) {
    const text = serializeWorkerError(error).toLowerCase();
    return (
      text.includes('worker request timeout')
      || text.includes('worker init timeout')
      || text.includes('worker timed out')
    );
  }

  async _ensureWorkerMultimodalCpuMode() {
    this._throwIfDisposed();
    if (!this._workerProxy) {
      await this._replaceWorkerProxyForMultimodalCpuMode();
    }
    this._throwIfDisposed();

    if (this._multimodalWorkerCpuMode) {
      return true;
    }

    if (!hasModelSource(this._loadedModelUrl)) {
      return false;
    }

    const selectedOptions = this._sanitizeModelLoadOptions(
      this._loadedModelOptions || {},
    );
    const operation = this._activeOperation;
    if (operation) {
      operation.recoveryAttempted = true;
      operation.state = 'recovering';
    }

    const previousDeferredState = this._deferredShadowState;
    this._shadowStateTransactionDepth += 1;
    this._deferredShadowState = null;
    let recoverySucceeded = false;

    const applyWorkerSafeMode = () => this._loadRememberedModelIntoWorker(selectedOptions);

    try {
      await applyWorkerSafeMode();
      recoverySucceeded = true;
      this._emitBridgeWarn(
        'llamadart: multimodal worker prepared in selected backend mode.',
      );
      return true;
    } catch (error) {
      this._throwIfDisposed();
      this._emitBridgeWarn(
        `llamadart: multimodal worker setup failed once; restarting worker (${serializeWorkerError(error)}).`,
      );

      await this._replaceWorkerProxyForMultimodalCpuMode();
      await applyWorkerSafeMode();
      recoverySucceeded = true;
      this._emitBridgeWarn(
        'llamadart: multimodal worker recovered after restart.',
      );
      return true;
    } finally {
      this._shadowStateTransactionDepth = Math.max(0, this._shadowStateTransactionDepth - 1);
      if (this._shadowStateTransactionDepth === 0) {
        if (recoverySucceeded && this._deferredShadowState && this._workerProxy) {
          this._applyShadowStateFrom(this._workerProxy, this._deferredShadowState);
        }
        this._deferredShadowState = previousDeferredState || null;
      }
      if (operation && this._activeOperation === operation && operation.state === 'recovering') {
        operation.state = operation.cancelRequested ? 'cancelling' : 'running';
      }
    }
  }

  async _loadRememberedModelIntoWorker(selectedOptions: LoadModelOptions) {
    this._throwIfDisposed();
    this._dropLoraAdaptersOf(this._workerProxy);
    await this._callWorker('loadModelFromUrl', [this._loadedModelUrl, selectedOptions]);
    this._throwIfDisposed();
    if (typeof this._loadedMmProjUrl === 'string' && this._loadedMmProjUrl.length > 0) {
      await this._callWorker('loadMultimodalProjector', [this._loadedMmProjUrl]);
    }
    this._throwIfDisposed();
    await this._restoreLoraAdapters();
    this._throwIfDisposed();
    this._workerModelMissing = false;
    this._loadedModelOptions = selectedOptions;
    this._multimodalWorkerCpuMode = true;
  }

  /**
   * Multimodal recovery can replace the worker and then fail to reload the
   * model, its projector or its applied LoRA adapters into it. The facade
   * still holds that model, but the fresh worker would answer every request
   * with "No model loaded" or without the adapters. Reload them first, and
   * again on the next call if this fails. A failure here reaches the caller's
   * own worker-error handling.
   */
  async _restoreWorkerModelIfMissing() {
    if (
      !this._workerProxy
      || this._workerModelMissing !== true
      || !hasModelSource(this._loadedModelUrl)
    ) {
      return;
    }

    this._emitBridgeWarn(
      'llamadart: bridge worker was restarted without its model; reloading it.',
    );
    await this._loadRememberedModelIntoWorker(
      this._sanitizeModelLoadOptions(this._loadedModelOptions || {}),
    );
  }

  _isDispatchWorkgroupLimitError(error: unknown) {
    const text = serializeWorkerError(error).toLowerCase();
    return (
      text.includes('dispatch workgroup count')
      || text.includes('max compute workgroups per dimension')
      || text.includes('invalid commandbuffer')
      || text.includes('ggml_webgpu: device error')
      || text.includes('runtimeerror: aborted')
      || text.includes('aborted()')
    );
  }

  // An aborted or trapped Wasm core stays dead. The worker host posts only
  // the message text, so a worker-side RuntimeError arrives without its name.
  _isWasmCoreAbortError(error: unknown) {
    if (typeof WebAssembly !== 'undefined' && error instanceof WebAssembly.RuntimeError) {
      return true;
    }

    const text = serializeWorkerError(error).toLowerCase();
    return (
      text.includes('aborted(')
      || text.includes('runtimeerror')
      || text.includes('unreachable')
      || text.includes('memory access out of bounds')
      || text.includes('function signature mismatch')
      || text.includes('program terminated with exit')
    );
  }

  // A worker request falls back only when the worker can no longer serve
  // requests: it crashed, stalled, timed out, never initialized, or its core
  // aborted. A core error from a healthy worker (invalid grammar, a context
  // limit, an unsupported template, a model without embeddings) is
  // deterministic; retrying on the main thread fails the same way and strands
  // the session there.
  _isWorkerUnusableError(error: unknown) {
    if (
      error
      && typeof error === 'object'
      && (error as WorkerRequestError).llamadartWorkerCrash === true
    ) {
      return true;
    }

    return (
      this._isWorkerTimeoutError(error)
      || this._isWasmCoreAbortError(error)
      || this._shouldFallbackToMainThread(error)
    );
  }

  _isWorkerTimeoutError(error: unknown) {
    if (
      error
      && typeof error === 'object'
      && (error as WorkerRequestError).llamadartWorkerTimeout === true
    ) {
      return true;
    }

    const text = serializeWorkerError(error).toLowerCase();
    return (
      text.includes('worker completion stalled')
      || text.includes('worker createcompletion stalled')
      || text.includes('worker timed out')
      || text.includes('worker timeout')
    );
  }

  _isCpuModelMode() {
    const requestedLayers = Number(this._loadedModelOptions?.nGpuLayers);
    if (Number.isFinite(requestedLayers)) {
      return requestedLayers === 0;
    }

    const metadataLayers = Number(this._metadata?.['llamadart.webgpu.n_gpu_layers']);
    if (Number.isFinite(metadataLayers)) {
      return metadataLayers === 0;
    }

    return false;
  }

  _workerCompletionStallTimeoutMs(options: WorkerFallbackOptions = {}) {
    const configured = Number(this._config?.workerGenerationStallTimeoutMs);
    if (Number.isFinite(configured) && configured > 0) {
      return Math.max(5000, Math.min(300000, Math.trunc(configured)));
    }

    if (!this._hasMediaParts(options)) {
      return 90000;
    }

    if (this._isCpuModelMode()) {
      return 0;
    }

    return 180000;
  }

  async _ensureRuntimeReadyAfterWorkerFallback(
    options: WorkerFallbackOptions = {},
    fallbackError: unknown = null,
  ) {
    this._throwIfDisposed();
    await this._waitForWorkerDisposal();
    this._throwIfDisposed();

    if (!this._runtime) {
      this._runtime = this._createRuntime();
    }
    const operation = this._activeOperation;
    operation?.runtimes?.add(this._runtime);
    if (operation && operation.state === 'running') {
      operation.state = 'recovering';
    }

    const forceReloadRequested = options?._llamadartForceRuntimeReload === true;
    const mediaPartsRequested = this._hasMediaParts(options);
    const textToSpeechRequested = options?._llamadartTextToSpeech === true;
    const multimodalRuntimeRequired = mediaPartsRequested || textToSpeechRequested;
    const shouldEnsureMultimodalInRuntime =
      multimodalRuntimeRequired
      && typeof this._loadedMmProjUrl === 'string'
      && this._loadedMmProjUrl.length > 0;
    const workerTimedOut = this._isWorkerTimeoutError(fallbackError);
    const dispatchWorkgroupFallback = this._isDispatchWorkgroupLimitError(fallbackError);
    const loadedGpuLayers = Number(this._loadedModelOptions?.nGpuLayers);
    const metadataGpuLayers = Number(this._metadata?.['llamadart.webgpu.n_gpu_layers']);
    const modelLoadedWithGpu = Number.isFinite(loadedGpuLayers)
      ? loadedGpuLayers !== 0
      : (Number.isFinite(metadataGpuLayers) ? metadataGpuLayers !== 0 : true);
    const shouldUseCpuMultimodalFallback =
      multimodalRuntimeRequired
      && modelLoadedWithGpu
      && (dispatchWorkgroupFallback || workerTimedOut);

    try {
      if (
        Number(this._runtime?._modelBytes) > 0
        && !forceReloadRequested
        && !shouldUseCpuMultimodalFallback
      ) {
        if (shouldEnsureMultimodalInRuntime) {
          const runtimeSupportsMedia =
            (typeof this._runtime.supportsVision === 'function' && this._runtime.supportsVision())
            || (typeof this._runtime.supportsAudio === 'function' && this._runtime.supportsAudio());

          if (!runtimeSupportsMedia) {
            this._throwIfDisposed();
            await this._runtime.loadMultimodalProjector(this._loadedMmProjUrl!);
            this._throwIfDisposed();
            if (Array.isArray(this._runtime._runtimeNotes)) {
              this._runtime._runtimeNotes.push('worker_fallback_reload_mmproj');
            }
          }
        }

        await this._restoreLoraAdapters();
        this._throwIfDisposed();
        return;
      }

      if (!hasModelSource(this._loadedModelUrl)) {
        return;
      }

      const loadOptions = shouldUseCpuMultimodalFallback
        ? this._createCpuSafeMultimodalLoadOptions(this._loadedModelOptions || {})
        : this._sanitizeModelLoadOptions(this._loadedModelOptions || {});
      const recoverySignal = this._operationSignal(operation) || options.signal || null;
      if (recoverySignal) {
        loadOptions.signal = recoverySignal;
      }
      if (shouldUseCpuMultimodalFallback) {
        if (textToSpeechRequested) {
          this._emitBridgeWarn(
            'llamadart: retrying text-to-speech once with CPU fallback after WebGPU failure.',
          );
        } else if (workerTimedOut) {
          this._emitBridgeWarn(
            'llamadart: retrying multimodal generation with CPU fallback after worker timeout.',
          );
        } else {
          this._emitBridgeWarn(
            'llamadart: retrying multimodal generation with CPU fallback after WebGPU workgroup limit failure.',
          );
        }
      }

      if (workerTimedOut) {
        this._emitBridgeWarn(
          'llamadart: bridge worker completion stalled; restarting generation path on main-thread runtime.',
        );
      }

      this._throwIfDisposed();
      this._dropLoraAdaptersOf(this._runtime);
      await this._runtime.loadModelFromUrl(this._loadedModelUrl!, loadOptions);
      this._throwIfDisposed();
      if (Array.isArray(this._runtime._runtimeNotes)) {
        this._runtime._runtimeNotes.push('worker_fallback_reload_model');
        if (forceReloadRequested) {
          this._runtime._runtimeNotes.push('worker_fallback_reload_forced');
        }
        if (workerTimedOut) {
          this._runtime._runtimeNotes.push('worker_fallback_timeout');
        }
        if (shouldUseCpuMultimodalFallback) {
          this._runtime._runtimeNotes.push('worker_fallback_cpu_multimodal');
          if (textToSpeechRequested) {
            this._runtime._runtimeNotes.push('worker_fallback_cpu_text_to_speech');
          }
        }
      }

      if (shouldEnsureMultimodalInRuntime) {
        this._throwIfDisposed();
        await this._runtime.loadMultimodalProjector(this._loadedMmProjUrl!);
        this._throwIfDisposed();
      }
      await this._restoreLoraAdapters();
      this._throwIfDisposed();
    } catch (error) {
      if (this._disposed || this._lifecycleState === 'disposing') {
        throw new Error(BRIDGE_DISPOSED_MESSAGE);
      }
      if ((error as Error | null)?.name === 'AbortError') {
        throw error;
      }
      // A model load followed by a projector failure is not a valid facade
      // state. Dispose the partially initialized direct runtime before the
      // queued operation publishes its snapshot.
      try {
        await this._runtime?.dispose?.();
      } catch (_) {
        // best-effort cleanup of an atomic recovery failure
      }
      this._runtime = null;
      this._metadata = {};
      this._contextSize = 0;
      this._gpuActive = false;
      this._backendName = '';
      this._supportsVision = false;
      this._supportsAudio = false;
      // The worker is already retired, so the facade is now direct-only with
      // no model: forget the model the failed recovery could not honour, and
      // keep an unloaded runtime so later calls fail with "No model loaded"
      // or load a new model instead of dereferencing null; the queue then
      // snapshots that unloaded runtime over the zeroed fields above. Disposal
      // that began during the cleanup above owns teardown and gets no runtime.
      this._loadedModelUrl = null;
      this._loadedModelOptions = null;
      this._loadedMmProjUrl = null;
      this._forgetLoraAdapters();
      if (!this._disposed && this._lifecycleState === 'open' && !this._workerProxy) {
        this._runtime = this._createRuntime();
        operation?.runtimes?.add(this._runtime);
      }
      throw error;
    } finally {
      if (operation && this._activeOperation === operation && operation.state === 'recovering') {
        operation.state = operation.cancelRequested ? 'cancelling' : 'running';
      }
    }
  }

  _createWorkerProxy() {
    return new BridgeWorkerProxy({
      moduleUrl: this._workerModuleUrl(),
      config: this._workerConfig(),
    });
  }

  _shouldUseWorker() {
    if (this._config?.disableWorker === true) {
      return false;
    }

    if (typeof Worker === 'undefined' ||
        typeof Blob === 'undefined' ||
        typeof URL === 'undefined' ||
        typeof URL.createObjectURL !== 'function') {
      return false;
    }

    if (this._config?.coreModuleFactory != null) {
      return false;
    }

    return true;
  }

  _workerModuleUrl() {
    const candidate = this._config?.workerUrl;
    if (typeof candidate === 'string' && candidate.trim().length > 0) {
      return candidate.trim();
    }

    try {
      return new URL('./llama_webgpu_bridge_worker.js', import.meta.url).toString();
    } catch (_) {
      return import.meta.url;
    }
  }

  _workerConfig() {
    const config = this._config || {};
    return {
      wasmUrl: typeof config.wasmUrl === 'string' ? config.wasmUrl : undefined,
      wasmUrlMem64: typeof config.wasmUrlMem64 === 'string'
        ? config.wasmUrlMem64
        : undefined,
      coreModuleUrl: typeof config.coreModuleUrl === 'string'
        ? config.coreModuleUrl
        : undefined,
      coreModuleUrlMem64: typeof config.coreModuleUrlMem64 === 'string'
        ? config.coreModuleUrlMem64
        : undefined,
      preferMemory64: typeof config.preferMemory64 === 'boolean'
        ? config.preferMemory64
        : undefined,
      threads: Number(config.threads) > 0 ? Number(config.threads) : undefined,
      threadsBatch: Number(config.threadsBatch) > 0
        ? Number(config.threadsBatch)
        : undefined,
      threadPoolSize: Number(config.threadPoolSize) > 0
        ? Number(config.threadPoolSize)
        : undefined,
      nBatch: Number(config.nBatch) > 0 ? Number(config.nBatch) : undefined,
      nUbatch: Number(config.nUbatch) > 0 ? Number(config.nUbatch) : undefined,
      nGpuLayers: Number.isFinite(config.nGpuLayers)
        ? Number(config.nGpuLayers)
        : undefined,
      userAgent: typeof config.userAgent === 'string' ? config.userAgent : undefined,
      cacheName: typeof config.cacheName === 'string' ? config.cacheName : undefined,
      remoteFetchThresholdBytes: Number(config.remoteFetchThresholdBytes) > 0
        ? Number(config.remoteFetchThresholdBytes)
        : undefined,
      remoteFetchChunkBytes: Number(config.remoteFetchChunkBytes) > 0
        ? Number(config.remoteFetchChunkBytes)
        : undefined,
      mediaMaxImagePixels: Number(config.mediaMaxImagePixels) > 0
        ? Number(config.mediaMaxImagePixels)
        : undefined,
      mediaMaxImageEdge: Number(config.mediaMaxImageEdge) > 0
        ? Number(config.mediaMaxImageEdge)
        : undefined,
      disableImageDownscale: config.disableImageDownscale === true,
      allowAutoRemoteFetchBackend: config.allowAutoRemoteFetchBackend === true,
      logLevel: Number.isFinite(config.logLevel) ? Number(config.logLevel) : 2,
    };
  }

  _applyShadowState(state: Partial<BridgeStateSnapshot> | null | undefined) {
    if (!state || typeof state !== 'object') {
      return;
    }

    if (state.metadata && typeof state.metadata === 'object') {
      this._metadata = state.metadata;
    }
    if (Number.isFinite(state.contextSize)) {
      this._contextSize = Number(state.contextSize);
    }
    if (typeof state.gpuActive === 'boolean') {
      this._gpuActive = state.gpuActive;
    }
    if (typeof state.backendName === 'string' && state.backendName.length > 0) {
      this._backendName = state.backendName;
    }
    if (typeof state.supportsVision === 'boolean') {
      this._supportsVision = state.supportsVision;
    }
    if (typeof state.supportsAudio === 'boolean') {
      this._supportsAudio = state.supportsAudio;
    }
  }

  _shouldFallbackToMainThread(error: unknown) {
    const text = serializeWorkerError(error).toLowerCase();

    if (text.includes('aborted(native code called abort())')) {
      return false;
    }
    if (text.includes('array buffer allocation failed')) {
      return false;
    }
    if (text.includes('bad_alloc')) {
      return false;
    }
    if (text.includes('out of memory')) {
      return false;
    }
    if (text.includes('memory access out of bounds')) {
      return false;
    }

    if (text.includes('bridge worker')) {
      return true;
    }
    if (text.includes('worker request failed')) {
      return true;
    }
    if (text.includes('worker request timeout')) {
      return true;
    }
    if (text.includes('worker init timeout')) {
      return true;
    }
    if (text.includes('timed out')) {
      return true;
    }
    if (text.includes('worker proxy is not available')) {
      return true;
    }
    if (text.includes('worker is not initialized')) {
      return true;
    }
    if (text.includes('failed to initialize bridge worker')) {
      return true;
    }
    if (text.includes('script error')) {
      return true;
    }

    return false;
  }

  _resolvedBridgeLogLevel() {
    const configured = Number(this._config?.logLevel);
    if (Number.isFinite(configured)) {
      return Math.max(0, Math.min(4, Math.trunc(configured)));
    }

    const runtimeLevel = Number(this._runtime?._logLevel);
    if (Number.isFinite(runtimeLevel)) {
      return Math.max(0, Math.min(4, Math.trunc(runtimeLevel)));
    }

    return 2;
  }

  _shouldEmitBridgeLevel(level: LogMethod) {
    const configured = this._resolvedBridgeLogLevel();
    const threshold = logThresholdForConfiguredLevel(configured);
    if (threshold > 3) {
      return false;
    }

    return logLevelForName(level) >= threshold;
  }

  _shouldSuppressBridgeWarn(message: string) {
    const text = String(message || '').trim();
    if (text.length === 0) {
      return false;
    }

    const configuredWindow = Number(this._config?.warnDedupWindowMs);
    const dedupWindowMs = Number.isFinite(configuredWindow) && configuredWindow > 0
      ? Math.max(500, Math.min(60000, Math.trunc(configuredWindow)))
      : 5000;
    const now = Date.now();
    const last = Number(this._bridgeWarnRecent.get(text) || 0);
    this._bridgeWarnRecent.set(text, now);

    if (this._bridgeWarnRecent.size > 80) {
      const staleThreshold = now - (dedupWindowMs * 2);
      for (const [key, atMs] of this._bridgeWarnRecent.entries()) {
        if (Number(atMs) < staleThreshold) {
          this._bridgeWarnRecent.delete(key);
        }
      }
    }

    return last > 0 && (now - last) < dedupWindowMs;
  }

  _emitBridgeWarn(message: string) {
    if (!this._shouldEmitBridgeLevel('warn')) {
      return;
    }

    if (this._shouldSuppressBridgeWarn(message)) {
      return;
    }

    if (this._runtime && typeof this._runtime._emitLogger === 'function') {
      this._runtime._emitLogger('warn', message);
      return;
    }

    if (typeof console !== 'undefined' && typeof console.warn === 'function') {
      console.warn(message);
    }
  }

  _disableWorkerFallback(error: unknown) {
    const reason = serializeWorkerError(error);
    this._workerFallbackReason = reason;

    if (typeof globalThis !== 'undefined') {
      (globalThis as WorkerFallbackGlobals).__llamadartBridgeWorkerFallbackReason = reason;
    }

    this._emitBridgeWarn(
      `llamadart: bridge worker unavailable, falling back to main thread (${reason})`,
    );

    if (this._workerProxy) {
      const workerProxy = this._workerProxy;
      // Invalidate the worker owner before detaching it. The active operation
      // retains the old proxy for targeted cancellation while the next phase
      // observes a distinct generation, even when it moves to direct runtime.
      this._advanceWorkerGeneration();
      this._workerProxy = null;
      this._retireWorkerProxy(workerProxy);
    }
    this._multimodalWorkerCpuMode = false;
    this._workerModelMissing = false;

    // Disposal owns the terminal transition, but an operation that already
    // owns the queue may still need one synchronous topology fallback to honor
    // the settle-first compatibility contract. A helper invoked after disposal
    // with no active owner must never recreate the runtime.
    const activeOperationMayFinish =
      this._activeOperation
      && this._activeOperation.kind !== 'dispose'
      && this._activeOperation.state !== 'cancelled'
      && this._activeOperation.state !== 'disposed';
    if (
      (this._disposed || this._lifecycleState === 'disposing')
      && !activeOperationMayFinish
    ) {
      return;
    }

    if (!this._runtime) {
      this._runtime = this._createRuntime();
      this._activeOperation?.runtimes?.add(this._runtime);
    }

    // A failed worker replacement falls back before its caller does, so the
    // caller's own fallback for the same error must not repeat the note.
    const note = `worker_fallback:${reason}`;
    if (
      this._runtime
      && Array.isArray(this._runtime._runtimeNotes)
      && typeof reason === 'string'
      && reason.length > 0
      && !this._runtime._runtimeNotes.includes(note)
    ) {
      this._runtime._runtimeNotes.push(note);
    }
  }

  async _callWorker<T = unknown>(
    method: string,
    args: unknown[],
    onEvent?: WorkerEventHandler | null,
    transferList: Transferable[] = [],
  ): Promise<T> {
    const proxy = this._workerProxy;
    if (!proxy) {
      throw new Error('Bridge worker proxy is not available');
    }

    const operation = this._activeOperation;
    if (operation) {
      this._recordWorkerGeneration(operation, proxy);
    }
    // A cancelled operation still runs until its worker call returns, so the
    // usage of its cancelled generation is accepted from the proxy that ran it.
    const guardedEvent = typeof onEvent === 'function'
      ? (event: WorkerResponse) => {
        if (
          !operation
          || this._isOperationCallbackCurrent(operation, proxy)
          || (event.event === 'usage' && proxy === this._workerProxy)
        ) {
          onEvent(event);
        }
      }
      : undefined;

    try {
      const response = await proxy.call(
        method,
        args,
        guardedEvent,
        transferList,
        operation
          ? { operationId: operation.id, workerGeneration: operation.workerGeneration }
          : undefined,
      );
      if (response?.state) {
        this._acceptWorkerState(operation, proxy, response.state);
      }
      return response?.value as T;
    } catch (error) {
      if (error && typeof error === 'object' && (error as WorkerRequestError).state) {
        this._acceptWorkerState(operation, proxy, (error as WorkerRequestError).state!);
      }
      throw error;
    }
  }

  /**
   * A worker response may only reach facade state while its proxy is still the
   * operation's current owner. Staging it for an open recovery transaction is
   * the same commit deferred, so a retired generation must be rejected before
   * the transaction branch, not after it.
   */
  _acceptWorkerState(
    operation: BridgeOperation | null,
    proxy: BridgeWorkerProxy,
    state: BridgeStateSnapshot,
  ) {
    if (operation && !this._isOperationCallbackCurrent(operation, proxy)) {
      return;
    }

    if (this._shadowStateTransactionDepth > 0) {
      this._deferredShadowState = state;
      if (operation) {
        operation.deferredShadowState = state;
      }
      return;
    }

    this._applyShadowStateFrom(proxy, state);
  }

  /**
   * The getters now answer from the facade snapshot, so a response from a proxy
   * that has since been replaced must not write into the current worker's state.
   */
  _applyShadowStateFrom(proxy: BridgeWorkerProxy, state: BridgeStateSnapshot) {
    if (proxy !== this._workerProxy) {
      return;
    }
    this._applyShadowState(state);
  }

  async loadModelFromUrl(url: string | string[], options: LoadModelOptions = {}) {
    return this._runExclusive(
      () => this._loadModelFromUrlUnlocked(url, options),
      {
        signal: options?.signal,
        abortMessage: 'Model load was cancelled.',
        kind: 'load-model',
      },
    );
  }

  async _loadModelFromUrlUnlocked(url: string | string[], options: LoadModelOptions = {}) {
    const attempt: WorkerLoadAttempt = { proxy: null, state: null, loaded: false };
    try {
      return await this._loadModelFromUrlOnTarget(url, options, attempt);
    } catch (error) {
      this._syncLoadedModelAfterFailedLoad(attempt);
      throw error;
    }
  }

  /**
   * @param attempt Records the last worker load's proxy and reply for
   *   `_syncLoadedModelAfterFailedLoad`.
   */
  async _loadModelFromUrlOnTarget(
    url: string | string[],
    options: LoadModelOptions,
    attempt: WorkerLoadAttempt,
  ) {
    if (!this._workerProxy) {
      const runtimeOptions = {
        ...options,
        signal: this._operationSignal() || options.signal || null,
      };
      const result = await this._runtime!.loadModelFromUrl(url, runtimeOptions);
      this._rememberLoadedModel(url, options);
      return result;
    }

    const invokeWorkerLoad = async () => {
      const workerOptions = { ...options };
      delete workerOptions.progressCallback;
      delete workerOptions.signal;

      const proxy = this._workerProxy;
      attempt.proxy = proxy;
      attempt.state = null;
      attempt.loaded = false;
      let result;
      try {
        result = await this._callWorker(
          'loadModelFromUrl',
          [url, workerOptions],
          (event) => {
            if (event.event !== 'progress') {
              return;
            }
            if (typeof options.progressCallback !== 'function') {
              return;
            }
            options.progressCallback((event.payload || {}) as BridgeProgressEvent);
          },
        );
      } catch (error) {
        attempt.state = error && typeof error === 'object'
          ? (error as WorkerRequestError).state || null
          : null;
        throw error;
      }
      attempt.loaded = true;
      this._throwIfCallerCancelled('Model load was cancelled.');
      this._rememberLoadedModel(url, workerOptions);
      return result;
    };

    try {
      return await invokeWorkerLoad();
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Model load was cancelled.');
      if (this._isRecoverableWorkerFsError(error) && !this._isWorkerRequestTimeoutError(error)) {
        this._emitBridgeWarn(
          `llamadart: worker model-load FS error detected; restarting worker (${serializeWorkerError(error)}).`,
        );
        try {
          await this._replaceWorkerProxyForMultimodalCpuMode();
          // The replacement starts empty, and the caller asked for the
          // previous model to be replaced, so the facade holds no model
          // until the retry succeeds.
          this._forgetLoadedModel();
          this._applyShadowState(emptyBridgeState());
          return await invokeWorkerLoad();
        } catch (retryError) {
          this._emitBridgeWarn(
            `llamadart: worker model-load retry failed (${serializeWorkerError(retryError)}).`,
          );
          error = retryError;
        }
      }

      if (!this._shouldFallbackToMainThread(error)) {
        throw error;
      }

      this._disableWorkerFallback(error);
      await this._waitForWorkerDisposal();
      this._throwIfDisposed();
      const result = await this._runtime!.loadModelFromUrl(url, {
        ...options,
        signal: this._operationSignal() || options.signal || null,
      });
      this._rememberLoadedModel(url, options);
      return result;
    }
  }

  async prefetchModelToCache(url: string | string[], options: LoadModelOptions = {}) {
    this._throwIfDisposed();
    if (!this._runtime) {
      this._runtime = this._createRuntime();
    }
    try {
      return await this._runtime.prefetchModelToCache(url, options);
    } finally {
      this._refreshCacheMetadataSnapshot();
    }
  }

  async evictModelFromCache(url: string | string[], options: Record<string, unknown> = {}) {
    this._throwIfDisposed();
    if (!this._runtime) {
      this._runtime = this._createRuntime();
    }
    try {
      return await this._runtime.evictModelFromCache(url, options);
    } finally {
      this._refreshCacheMetadataSnapshot();
    }
  }

  async createCompletion(prompt: string, options: CompletionOptions = {}) {
    resolveCompletionSamplingOptions(options);
    const onUsage = options?.onUsage;
    if (typeof onUsage !== 'function') {
      return this._runExclusive(
        () => this._createCompletionUnlocked(prompt, options),
        {
          signal: options?.signal,
          abortMessage: 'Generation was cancelled.',
          kind: 'generation',
        },
      );
    }

    let usage = null as CompletionUsage | null;
    let text: string;
    try {
      text = await this._runExclusive(
        () => this._createCompletionUnlocked(prompt, {
          ...options,
          onUsage: (value: CompletionUsage) => {
            usage = value;
          },
        }),
        {
          signal: options.signal,
          abortMessage: 'Generation was cancelled.',
          kind: 'generation',
        },
      );
    } catch (error) {
      if (usage != null && (error as Error | null)?.name === 'AbortError') {
        onUsage(usage);
      }
      throw error;
    }
    if (usage != null) {
      onUsage(usage);
    }
    return text;
  }

  async getCompletionCapabilities() {
    return this._runExclusive(
      () => this._getCompletionCapabilitiesUnlocked(),
      { kind: 'completion-capabilities' },
    );
  }

  async _getCompletionCapabilitiesUnlocked(): Promise<CompletionCapabilities> {
    if (!this._workerProxy) {
      return this._runtime?.getCompletionCapabilities() ?? { ...NO_COMPLETION_CAPABILITIES };
    }
    try {
      await this._restoreWorkerModelIfMissing();
      return await this._callWorker<CompletionCapabilities>('getCompletionCapabilities', []);
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Completion capability probe was cancelled.');
      if (!this._isWorkerUnusableError(error)) {
        throw error;
      }
      this._disableWorkerFallback(error);
      await this._waitForWorkerDisposal();
      await this._ensureRuntimeReadyAfterWorkerFallback({}, error);
      return this._runtime!.getCompletionCapabilities();
    }
  }

  async _createCompletionUnlocked(
    prompt: string,
    options: BridgeCompletionOptions = {},
  ): Promise<string> {
    const isWarmup = options?.warmup === true;
    const hasRetriedEmptyMultimodal =
      options?.__llamadartEmptyRetryAttempted === true;
    const workerAllowed = this._config?.disableWorker !== true;
    if (this._hasMediaParts(options) && workerAllowed) {
      const hasWorkerFallback =
        typeof this._workerFallbackReason === 'string'
        && this._workerFallbackReason.length > 0;
      if (hasWorkerFallback && !this._workerProxy && !this._isCpuModelMode()) {
        await this._ensureRuntimeReadyAfterWorkerFallback(options, null);
        return this._runtime!.createCompletion(prompt, options);
      }

      try {
        if (!this._workerProxy) {
          await this._replaceWorkerProxyForMultimodalCpuMode();
        }
        await this._ensureWorkerMultimodalCpuMode();
      } catch (error) {
        this._throwIfOperationCancelled(error, 'Generation was cancelled.');
        const reason = serializeWorkerError(error);
        if (isWarmup) {
          this._emitBridgeWarn(
            `llamadart: multimodal warmup skipped after worker setup issue (${reason}).`,
          );
          return '';
        }

        this._emitBridgeWarn(
          `llamadart: unable to prepare multimodal worker CPU mode (${reason}).`,
        );

        if (this._isCpuModelMode()) {
          throw new Error(
            `CPU multimodal worker setup failed (${reason}). `
            + 'Reload model and retry with a smaller image.',
          );
        }

        this._disableWorkerFallback(error);
        await this._waitForWorkerDisposal();
        await this._ensureRuntimeReadyAfterWorkerFallback(options, error);
        return this._runtime!.createCompletion(prompt, options);
      }
    }

    if (!this._workerProxy) {
      return this._runtime!.createCompletion(prompt, options);
    }

    try {
      await this._restoreWorkerModelIfMissing();
      const workerOptions = { ...options };
      delete workerOptions.onToken;
      delete workerOptions.onUsage;
      delete workerOptions.signal;
      delete workerOptions.__llamadartEmptyRetryAttempted;

      const stallTimeoutMs = this._workerCompletionStallTimeoutMs(options);
      let timeoutHandle: ReturnType<typeof setTimeout> | null = null;
      let rejectOnStall: ((reason: unknown) => void) | null = null;

      const clearStallTimer = () => {
        if (timeoutHandle != null) {
          globalThis.clearTimeout(timeoutHandle);
          timeoutHandle = null;
        }
      };

      const armStallTimer = () => {
        clearStallTimer();
        if (!Number.isFinite(stallTimeoutMs) || stallTimeoutMs <= 0) {
          return;
        }

        timeoutHandle = globalThis.setTimeout(() => {
          const timeoutError = new Error(
            `Bridge worker completion stalled for ${stallTimeoutMs}ms.`,
          ) as WorkerRequestError;
          timeoutError.llamadartWorkerTimeout = true;
          this._cancelOperation(this._activeOperation, 'worker-timeout');
          rejectOnStall?.(timeoutError);
        }, stallTimeoutMs);
      };

      const stallPromise = new Promise<never>((_, reject) => {
        rejectOnStall = reject;
      });

      armStallTimer();

      let sawWorkerTokenEvent = false;

      try {
        const workerResult = await Promise.race([
          this._callWorker<string>(
            'createCompletion',
            [prompt, workerOptions],
            (event) => {
              if (event.event === 'usage') {
                if (typeof options.onUsage === 'function') {
                  options.onUsage(event.payload as CompletionUsage);
                }
                return;
              }

              if (event.event !== 'token') {
                return;
              }

              armStallTimer();
              sawWorkerTokenEvent = true;

              if (typeof options.onToken !== 'function') {
                return;
              }

              const payload = (event.payload || {}) as WorkerTokenPayload;
              const piece = typeof payload.pieceText === 'string'
                ? payload.pieceText
                : Uint8Array.from(Array.isArray(payload.piece) ? payload.piece : []);
              options.onToken(piece, String(payload.currentText || ''));
            },
          ),
          stallPromise,
        ]);

        if (
          this._hasMediaParts(options)
          && !isWarmup
          && !sawWorkerTokenEvent
          && String(workerResult || '').trim().length == 0
        ) {
          this._emitBridgeWarn(
            'llamadart: multimodal worker produced empty response without token events.',
          );

          if (!hasRetriedEmptyMultimodal) {
            this._emitBridgeWarn(
              'llamadart: retrying multimodal worker once after empty response.',
            );
            try {
              await this._replaceWorkerProxyForMultimodalCpuMode();
              await this._ensureWorkerMultimodalCpuMode();
            } catch (retrySetupError) {
              this._emitBridgeWarn(
                `llamadart: multimodal empty-response retry setup failed (${serializeWorkerError(retrySetupError)}).`,
              );
            }

            return this._createCompletionUnlocked(prompt, {
              ...options,
              __llamadartEmptyRetryAttempted: true,
            });
          }
        }

        this._throwIfCallerCancelled('Generation was cancelled.');
        return workerResult;
      } finally {
        clearStallTimer();
      }
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Generation was cancelled.');
      if (serializeWorkerError(error).includes(INVALID_GRAMMAR_ERROR_TEXT)) {
        throw error;
      }
      if (this._hasMediaParts(options)) {
        const reason = serializeWorkerError(error);

        if (isWarmup) {
          this._emitBridgeWarn(
            `llamadart: multimodal warmup skipped after worker request issue (${reason}).`,
          );
          return '';
        }

        if (this._isCpuModelMode()) {
          this._emitBridgeWarn(
            `llamadart: CPU multimodal worker request failed (${reason}); skipping main-thread fallback.`,
          );
          throw new Error(
            `CPU multimodal request failed (${reason}). `
            + 'Reload model and retry with a smaller image.',
          );
        }

        this._emitBridgeWarn(
          `llamadart: multimodal worker request failed (${reason}); falling back to main-thread runtime.`,
        );

        this._disableWorkerFallback(error);
        await this._waitForWorkerDisposal();
        await this._ensureRuntimeReadyAfterWorkerFallback(options, error);
        return this._runtime!.createCompletion(prompt, options);
      }

      if (!this._isWorkerUnusableError(error)) {
        throw error;
      }

      this._disableWorkerFallback(error);
      await this._waitForWorkerDisposal();
      await this._ensureRuntimeReadyAfterWorkerFallback(options, error);
      return this._runtime!.createCompletion(prompt, options);
    }
  }

  async loadMultimodalProjector(url: string) {
    return this._runExclusive(
      () => this._loadMultimodalProjectorUnlocked(url),
      { kind: 'projector-load' },
    );
  }

  async _loadMultimodalProjectorUnlocked(url: string) {
    const startedWithWorker = this._workerProxy != null;
    const invokeRuntimeLoad = async () => {
      if (!this._runtime) {
        this._runtime = this._createRuntime();
      }

      await this._ensureRuntimeReadyAfterWorkerFallback({}, null);
      const result = await this._runtime.loadMultimodalProjector(url);
      this._rememberLoadedMmProj(url);
      this._supportsVision = this._runtime.supportsVision();
      this._supportsAudio = this._runtime.supportsAudio();
      return result;
    };

    try {
      if (!this._workerProxy) {
        return await invokeRuntimeLoad();
      }

      await this._ensureWorkerMultimodalCpuMode();
      const result = await this._callWorker('loadMultimodalProjector', [url]);
      this._rememberLoadedMmProj(url);
      return result;
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Projector operation was cancelled.');
      if (!startedWithWorker) {
        throw error;
      }
      const reason = serializeWorkerError(error);
      this._emitBridgeWarn(
        `llamadart: multimodal worker setup failed (${reason}).`,
      );

      if (this._isCpuModelMode()) {
        try {
          await this._replaceWorkerProxyForMultimodalCpuMode();
          await this._ensureWorkerMultimodalCpuMode();
          const retryResult = await this._callWorker('loadMultimodalProjector', [url]);
          this._rememberLoadedMmProj(url);
          this._emitBridgeWarn(
            'llamadart: CPU multimodal worker setup recovered after worker restart.',
          );
          return retryResult;
        } catch (retryError) {
          const retryReason = serializeWorkerError(retryError);
          throw new Error(
            `CPU multimodal projector setup failed (${retryReason}). `
            + 'Reload model and retry with a smaller image.',
          );
        }
      }

      this._disableWorkerFallback(error);
      await this._waitForWorkerDisposal();
      return invokeRuntimeLoad();
    }
  }

  async unloadMultimodalProjector() {
    return this._runExclusive(
      () => this._unloadMultimodalProjectorUnlocked(),
      { kind: 'projector-unload' },
    );
  }

  async _unloadMultimodalProjectorUnlocked() {
    if (!this._workerProxy) {
      const result = await this._runtime!.unloadMultimodalProjector();
      this._loadedMmProjUrl = null;
      this._supportsVision = this._runtime!.supportsVision();
      this._supportsAudio = this._runtime!.supportsAudio();
      return result;
    }

    try {
      const result = await this._callWorker('unloadMultimodalProjector', []);
      this._loadedMmProjUrl = null;
      return result;
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Projector operation was cancelled.');
      this._disableWorkerFallback(error);
      await this._waitForWorkerDisposal();
      const result = await this._runtime!.unloadMultimodalProjector();
      this._loadedMmProjUrl = null;
      this._supportsVision = this._runtime!.supportsVision();
      this._supportsAudio = this._runtime!.supportsAudio();
      return result;
    }
  }

  supportsVision() {
    return this._supportsVision;
  }

  supportsAudio() {
    return this._supportsAudio;
  }

  async getTextToSpeechCapabilities() {
    return this._runExclusive(
      () => this._getTextToSpeechCapabilitiesUnlocked(),
      { kind: 'tts-capabilities' },
    );
  }

  async _getTextToSpeechCapabilitiesUnlocked() {
    if (!this._workerProxy) {
      return this._runtime!.getTextToSpeechCapabilities();
    }
    try {
      await this._restoreWorkerModelIfMissing();
      return await this._callWorker<TextToSpeechCapabilities>('getTextToSpeechCapabilities', []);
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Text-to-speech capability probe was cancelled.');
      this._disableWorkerFallback(error);
      await this._waitForWorkerDisposal();
      await this._ensureRuntimeReadyAfterWorkerFallback({}, error);
      return this._runtime!.getTextToSpeechCapabilities();
    }
  }

  async synthesizeSpeech(options: Partial<TextToSpeechOptions> = {}) {
    return this._runExclusive(
      () => this._synthesizeSpeechUnlocked(options),
      {
        signal: options?.signal,
        abortMessage: 'Text-to-speech synthesis was cancelled.',
        kind: 'text-to-speech',
      },
    );
  }

  async _synthesizeSpeechUnlocked(options: Partial<TextToSpeechOptions> = {}) {
    if (options.signal?.aborted) {
      throw new DOMException('Text-to-speech synthesis was cancelled.', 'AbortError');
    }
    if (!this._workerProxy) {
      return this._runtime!.synthesizeSpeech(options);
    }

    try {
      await this._restoreWorkerModelIfMissing();
      const workerOptions = { ...options };
      delete workerOptions.onProgress;
      delete workerOptions.signal;
      const transferList: Transferable[] = [];
      const speakerBytes = toUint8Array(options.speakerAudio);
      if (speakerBytes && speakerBytes.length > 0) {
        const copy = new Uint8Array(speakerBytes);
        workerOptions.speakerAudio = copy;
        transferList.push(copy.buffer);
      }
      const result = await this._callWorker<TextToSpeechResult>(
        'synthesizeSpeech',
        [workerOptions],
        (event) => {
          if (event.event === 'progress') {
            options.onProgress?.((event.payload || {}) as TextToSpeechProgress);
          }
        },
        transferList,
      );
      this._throwIfCallerCancelled('Text-to-speech synthesis was cancelled.');
      return result;
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Text-to-speech synthesis was cancelled.');
      if (
        (error as Error | null)?.name === 'AbortError'
        || options.signal?.aborted
        || serializeWorkerError(error).toLowerCase().includes('cancel')
      ) {
        throw new DOMException('Text-to-speech synthesis was cancelled.', 'AbortError');
      }

      const canRetryOnCpu =
        !this._isCpuModelMode()
        && (
          this._isDispatchWorkgroupLimitError(error)
          || this._isWorkerRequestTimeoutError(error)
        );
      if (canRetryOnCpu) {
        this._disableWorkerFallback(error);
        await this._waitForWorkerDisposal();
        await this._ensureRuntimeReadyAfterWorkerFallback(
          {
            _llamadartForceRuntimeReload: true,
            _llamadartTextToSpeech: true,
          },
          error,
        );
        if (options.signal?.aborted) {
          throw new DOMException('Text-to-speech synthesis was cancelled.', 'AbortError');
        }
        return this._runtime!.synthesizeSpeech(options);
      }

      if (this._isWorkerRequestTimeoutError(error)) {
        this._disableWorkerFallback(error);
        await this._waitForWorkerDisposal();
      }
      throw error;
    }
  }

  _decisionHeadMap() {
    if (!(this._decisionHeads instanceof Map)) {
      this._decisionHeads = new Map();
      this._nextDecisionHandle = 1;
    }
    return this._decisionHeads;
  }

  _registerDecisionHead(owner: object, info: DecisionHeadInfo) {
    const heads = this._decisionHeadMap();
    const handle = this._nextDecisionHandle++;
    heads.set(handle, { owner, handle: Number(info?.handle) });
    return { ...info, handle };
  }

  _decisionOwner() {
    return this._workerProxy || this._runtime || null;
  }

  _staleDecisionHeadError(handle: number) {
    return new Error(
      `Decision head ${handle} is not loaded; it was freed, its model was unloaded, `
      + 'or the bridge runtime restarted. Load the decision head again.',
    );
  }

  _resolveDecisionHead(handle: number) {
    decisionHandleFrom(handle);
    const heads = this._decisionHeadMap();
    const entry = heads.get(handle);
    if (!entry) {
      return null;
    }
    if (entry.owner !== this._decisionOwner()) {
      heads.delete(handle);
      return null;
    }
    return entry;
  }

  _requireDecisionRuntime() {
    if (!this._runtime) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }
    return this._runtime;
  }

  /**
   * Moves a failed worker's work to the main-thread runtime. Decision heads
   * the worker held are gone with it; the model, its projector and the applied
   * LoRA adapters are restored so later calls keep working.
   */
  async _recoverWorkerOnMainThread(error: unknown) {
    this._disableWorkerFallback(error);
    await this._waitForWorkerDisposal();
    await this._ensureRuntimeReadyAfterWorkerFallback({}, error);
  }

  async getDecisionCapabilities() {
    return this._runExclusive(
      () => this._getDecisionCapabilitiesUnlocked(),
      { kind: 'decision-capabilities' },
    );
  }

  async _getDecisionCapabilitiesUnlocked() {
    if (!this._workerProxy) {
      return this._runtime
        ? this._runtime.getDecisionCapabilities()
        : { apiVersion: DECISION_API_VERSION, supported: false, reason: 'WebGPU core is not initialized' };
    }
    try {
      return await this._callWorker<DecisionCapabilities>('getDecisionCapabilities', []);
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Decision capability probe was cancelled.');
      if (!this._shouldFallbackToMainThread(error)) {
        throw error;
      }
      await this._recoverWorkerOnMainThread(error);
      return this._requireDecisionRuntime().getDecisionCapabilities();
    }
  }

  async loadDecisionHead(
    source: string | ArrayBuffer | ArrayBufferView,
    options: DecisionHeadOptions = {},
  ) {
    return this._runExclusive(
      () => this._loadDecisionHeadUnlocked(source, options),
      { kind: 'decision-head-load' },
    );
  }

  async _loadDecisionHeadUnlocked(
    source: string | ArrayBuffer | ArrayBufferView,
    options: DecisionHeadOptions = {},
  ) {
    const loadInRuntime = async () => {
      const runtime = this._requireDecisionRuntime();
      return this._registerDecisionHead(
        runtime,
        await runtime.loadDecisionHead(source, options),
      );
    };
    if (!this._workerProxy) {
      return loadInRuntime();
    }

    const workerOptions = { configJson: options?.configJson ?? null };
    let workerSource: string | Uint8Array | ArrayBuffer | ArrayBufferView = source;
    const transferList: Transferable[] = [];
    if (typeof source === 'string') {
      // The worker resolves relative URLs against its own script URL.
      workerSource = source.length > 0 ? normalizeAbsoluteUrl(source) : source;
    } else {
      const bytes = bufferSourceBytes(source);
      if (bytes) {
        // Transfer a copy so the caller's buffer stays usable.
        const copy = new Uint8Array(bytes);
        workerSource = copy;
        transferList.push(copy.buffer);
      }
    }
    try {
      await this._restoreWorkerModelIfMissing();
      const proxy = this._workerProxy;
      const info = await this._callWorker<DecisionHeadInfo>(
        'loadDecisionHead',
        [workerSource, workerOptions],
        (event) => {
          if (event.event === 'progress') {
            options?.onProgress?.((event.payload || {}) as DecisionHeadProgress);
          }
        },
        transferList,
      );
      return this._registerDecisionHead(proxy, info);
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Decision head load was cancelled.');
      if (!this._shouldFallbackToMainThread(error)) {
        throw error;
      }
      await this._recoverWorkerOnMainThread(error);
      return loadInRuntime();
    }
  }

  async runDecision(handle: number, sequences: readonly DecisionSequence[]) {
    return this._runExclusive(
      () => this._runDecisionUnlocked(handle, sequences),
      { kind: 'decision-run' },
    );
  }

  async _runDecisionUnlocked(handle: number, sequences: readonly DecisionSequence[]) {
    const entry = this._resolveDecisionHead(handle);
    if (!entry) {
      throw this._staleDecisionHeadError(handle);
    }
    const normalized = normalizeDecisionSequences(sequences);
    if (!this._workerProxy) {
      return this._requireDecisionRuntime().runDecision(entry.handle, normalized);
    }
    try {
      return await this._callWorker<DecisionOutput[]>('runDecision', [entry.handle, normalized]);
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Decision run was cancelled.');
      if (!this._shouldFallbackToMainThread(error)) {
        throw error;
      }
      this._decisionHeadMap().delete(handle);
      await this._recoverWorkerOnMainThread(error);
      throw new Error(
        `Decision head ${handle} was lost when the bridge worker failed `
        + `(${serializeWorkerError(error)}). Load the decision head again.`,
      );
    }
  }

  async freeDecisionHead(handle: number) {
    return this._runExclusive(
      () => this._freeDecisionHeadUnlocked(handle),
      { kind: 'decision-head-free' },
    );
  }

  async _freeDecisionHeadUnlocked(handle: number) {
    const entry = this._resolveDecisionHead(handle);
    if (!entry) {
      return;
    }
    this._decisionHeadMap().delete(handle);
    if (!this._workerProxy) {
      await this._requireDecisionRuntime().freeDecisionHead(entry.handle);
      return;
    }
    try {
      await this._callWorker('freeDecisionHead', [entry.handle]);
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Decision head free was cancelled.');
      if (!this._shouldFallbackToMainThread(error)) {
        throw error;
      }
      // The failed worker took the head with it.
      await this._recoverWorkerOnMainThread(error);
    }
  }

  // Test doubles skip the constructor, so the LoRA state starts on first use.
  _loraAdapterMap() {
    if (!(this._loraAdapters instanceof Map)) {
      this._loraAdapters = new Map();
      this._activeLoraScales = new Map();
      this._nextLoraHandle = 1;
    }
    return this._loraAdapters;
  }

  _activeLoraScaleMap() {
    this._loraAdapterMap();
    return this._activeLoraScales;
  }

  _forgetLoraAdapters() {
    this._loraAdapterMap().clear();
    this._activeLoraScaleMap().clear();
  }

  /**
   * Marks the adapters `owner` holds as lost before it loads the model again,
   * which frees them. An adapter with a retained source reloads on the next
   * restore or set; one without is forgotten.
   */
  _dropLoraAdaptersOf(owner: object | null) {
    for (const [handle, entry] of this._loraAdapterMap()) {
      if (owner === null || entry.owner !== owner) {
        continue;
      }
      if (entry.source === null) {
        this._loraAdapterMap().delete(handle);
        this._activeLoraScaleMap().delete(handle);
      } else {
        entry.owner = null;
      }
    }
  }

  _loraOwner() {
    return this._workerProxy || this._runtime || null;
  }

  _requireLoraRuntime() {
    if (!this._runtime) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }
    return this._runtime;
  }

  _loraAdapterEntry(handle: number) {
    const entry = this._loraAdapterMap().get(handle);
    if (!entry) {
      throw staleLoraAdapterError(handle);
    }
    return entry;
  }

  // Loads an adapter into the current owner: the worker, or the direct runtime.
  async _loadLoraAdapterIntoOwner(
    source: string | Uint8Array,
    options: LoraAdapterLoadOptions,
  ): Promise<{ owner: object; handle: number }> {
    if (!this._workerProxy) {
      const runtime = this._requireLoraRuntime();
      const info = await runtime.loadLoraAdapter(source, {
        ...options,
        signal: this._operationSignal() || options.signal || null,
      });
      return { owner: runtime, handle: info.handle };
    }
    const workerOptions: LoraAdapterLoadOptions = {};
    if (options.useCache !== undefined) {
      workerOptions.useCache = options.useCache;
    }
    if (options.cacheName !== undefined) {
      workerOptions.cacheName = options.cacheName;
    }
    let workerSource = source;
    const transferList: Transferable[] = [];
    if (typeof source !== 'string') {
      // Transfer a copy so the caller's buffer and the retained source stay usable.
      workerSource = new Uint8Array(source);
      transferList.push(workerSource.buffer);
    }
    const proxy = this._workerProxy;
    const info = await this._callWorker<LoraAdapterInfo>(
      'loadLoraAdapter',
      [workerSource, workerOptions],
      (event) => {
        if (event.event === 'progress') {
          options.progressCallback?.((event.payload || {}) as BridgeProgressEvent);
        }
      },
      transferList,
    );
    return { owner: proxy, handle: Number(info?.handle) };
  }

  // Reloads an adapter its owner lost, into the current owner.
  async _ensureLoraAdapterOnOwner(handle: number, entry: LoraAdapterEntry) {
    if (entry.owner === this._loraOwner()) {
      return;
    }
    if (entry.source === null) {
      this._loraAdapterMap().delete(handle);
      this._activeLoraScaleMap().delete(handle);
      throw staleLoraAdapterError(handle);
    }
    const loaded = await this._loadLoraAdapterIntoOwner(entry.source, entry.cacheOptions);
    entry.owner = loaded.owner;
    entry.handle = loaded.handle;
  }

  async _callLoraOwner(method: LoraOwnerMethod, args: number[]) {
    if (this._workerProxy) {
      await this._callWorker(method, args);
      return;
    }
    const runtime = this._requireLoraRuntime();
    if (method === 'setLoraAdapter') {
      await runtime.setLoraAdapter(args[0], args[1]);
    } else if (method === 'removeLoraAdapter') {
      await runtime.removeLoraAdapter(args[0]);
    } else {
      await runtime.clearLoraAdapters();
    }
  }

  /**
   * Makes the current owner apply exactly the facade's active adapters after a
   * worker restart or a fallback to the main thread. The owner's own set is
   * cleared first, since a runtime that held the model earlier may still apply
   * adapters from then. Inactive adapters reload when next set.
   */
  async _restoreLoraAdapters() {
    const owner = this._loraOwner();
    if (![...this._loraAdapterMap().values()].some((entry) => entry.owner !== owner)) {
      return;
    }
    await this._callLoraOwner('clearLoraAdapters', []);
    for (const [handle, scale] of [...this._activeLoraScaleMap()]) {
      const entry = this._loraAdapterEntry(handle);
      await this._ensureLoraAdapterOnOwner(handle, entry);
      await this._callLoraOwner('setLoraAdapter', [entry.handle, scale]);
    }
  }

  // Runs a LoRA call on the current owner, moving to the main-thread runtime
  // and running it there once if the worker fails.
  async _runLoraCall<T>(call: () => Promise<T>, cancelMessage: string): Promise<T> {
    if (!this._workerProxy) {
      return call();
    }
    try {
      await this._restoreWorkerModelIfMissing();
      return await call();
    } catch (error) {
      this._throwIfOperationCancelled(error, cancelMessage);
      if (!this._shouldFallbackToMainThread(error)) {
        throw error;
      }
      await this._recoverWorkerOnMainThread(error);
      return call();
    }
  }

  async getLoraAdapterCapabilities() {
    return this._runExclusive(
      () => this._getLoraAdapterCapabilitiesUnlocked(),
      { kind: 'lora-capabilities' },
    );
  }

  async _getLoraAdapterCapabilitiesUnlocked(): Promise<LoraAdapterCapabilities> {
    if (!this._workerProxy && !this._runtime) {
      return { apiVersion: LORA_API_VERSION, supported: false, reason: 'WebGPU core is not initialized' };
    }
    return this._runLoraCall(
      () => (this._workerProxy
        ? this._callWorker<LoraAdapterCapabilities>('getLoraAdapterCapabilities', [])
        : Promise.resolve(this._requireLoraRuntime().getLoraAdapterCapabilities())),
      'LoRA capability probe was cancelled.',
    );
  }

  async loadLoraAdapter(
    source: string | ArrayBuffer | ArrayBufferView,
    options: LoraAdapterLoadOptions = {},
  ) {
    return this._runExclusive(
      () => this._loadLoraAdapterUnlocked(source, options || {}),
      {
        signal: options?.signal,
        abortMessage: LORA_LOAD_ABORT_MESSAGE,
        kind: 'lora-adapter-load',
      },
    );
  }

  async _loadLoraAdapterUnlocked(
    source: string | ArrayBuffer | ArrayBufferView,
    options: LoraAdapterLoadOptions,
  ): Promise<LoraAdapterInfo> {
    let normalized: string | Uint8Array;
    if (typeof source === 'string') {
      if (source.length === 0) {
        throw new Error('LoRA adapter URL is empty.');
      }
      // The worker resolves relative URLs against its own script URL.
      normalized = normalizeAbsoluteUrl(source);
    } else {
      const bytes = bufferSourceBytes(source);
      if (!bytes) {
        throw new TypeError('LoRA adapter source must be a URL string, an ArrayBuffer or a typed array.');
      }
      if (bytes.byteLength === 0) {
        throw new Error('LoRA adapter bytes are empty.');
      }
      normalized = bytes;
    }
    // With workers enabled the owner can change, so keep what reloads the
    // adapter; bytes are copied because the caller may reuse its buffer.
    let retained: string | Uint8Array | null = null;
    if (this._config?.disableWorker !== true) {
      retained = typeof normalized === 'string' ? normalized : new Uint8Array(normalized);
    }
    const loaded = await this._runLoraCall(
      () => this._loadLoraAdapterIntoOwner(retained ?? normalized, options),
      LORA_LOAD_ABORT_MESSAGE,
    );
    this._loraAdapterMap();
    const handle = this._nextLoraHandle++;
    this._loraAdapterMap().set(handle, {
      owner: loaded.owner,
      handle: loaded.handle,
      source: retained,
      cacheOptions: { useCache: options.useCache, cacheName: options.cacheName },
    });
    return { handle };
  }

  async setLoraAdapter(handle: number, scale = 1) {
    const facadeHandle = loraHandleFrom(handle);
    const checkedScale = loraScaleFrom(scale);
    return this._runExclusive(
      () => this._setLoraAdapterUnlocked(facadeHandle, checkedScale),
      { kind: 'lora-adapter-set' },
    );
  }

  async _setLoraAdapterUnlocked(handle: number, scale: number) {
    const entry = this._loraAdapterEntry(handle);
    await this._runLoraCall(async () => {
      await this._ensureLoraAdapterOnOwner(handle, entry);
      await this._callLoraOwner('setLoraAdapter', [entry.handle, scale]);
    }, 'LoRA adapter update was cancelled.');
    this._activeLoraScaleMap().set(handle, scale);
  }

  async removeLoraAdapter(handle: number) {
    const facadeHandle = loraHandleFrom(handle);
    return this._runExclusive(
      () => this._removeLoraAdapterUnlocked(facadeHandle),
      { kind: 'lora-adapter-remove' },
    );
  }

  async _removeLoraAdapterUnlocked(handle: number) {
    const entry = this._loraAdapterEntry(handle);
    await this._runLoraCall(async () => {
      // An owner that never loaded this adapter does not apply it.
      if (entry.owner === this._loraOwner()) {
        await this._callLoraOwner('removeLoraAdapter', [entry.handle]);
      }
    }, 'LoRA adapter update was cancelled.');
    this._activeLoraScaleMap().delete(handle);
  }

  async clearLoraAdapters() {
    return this._runExclusive(
      () => this._clearLoraAdaptersUnlocked(),
      { kind: 'lora-adapters-clear' },
    );
  }

  async _clearLoraAdaptersUnlocked() {
    await this._runLoraCall(
      () => this._callLoraOwner('clearLoraAdapters', []),
      'LoRA adapter update was cancelled.',
    );
    this._activeLoraScaleMap().clear();
  }

  async tokenize(text: string, addSpecial = true) {
    return this._runExclusive(
      () => this._tokenizeUnlocked(text, addSpecial),
      { kind: 'tokenize' },
    );
  }

  async _tokenizeUnlocked(text: string, addSpecial = true) {
    if (!this._workerProxy) {
      return this._runtime!.tokenize(text, addSpecial);
    }

    try {
      await this._restoreWorkerModelIfMissing();
      return await this._callWorker<number[]>('tokenize', [text, addSpecial]);
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Tokenization was cancelled.');
      if (!this._isWorkerUnusableError(error)) {
        throw error;
      }
      this._disableWorkerFallback(error);
      await this._waitForWorkerDisposal();
      await this._ensureRuntimeReadyAfterWorkerFallback({}, error);
      return this._runtime!.tokenize(text, addSpecial);
    }
  }

  async stateSaveFile(path: string, tokens: number[] | ArrayLike<number> = []) {
    return this._runExclusive(
      () => this._stateSaveFileUnlocked(path, tokens),
      { kind: 'state-save-file' },
    );
  }

  async _stateSaveFileUnlocked(path: string, tokens: number[] | ArrayLike<number> = []) {
    if (!this._workerProxy) {
      return this._runtime!.stateSaveFile(path, tokens);
    }

    const normalized = Array.isArray(tokens)
      ? tokens
      : Array.from(tokens || []);
    await this._restoreWorkerModelIfMissing();
    return this._callWorker<true>('stateSaveFile', [path, normalized]);
  }

  // The capacity default is resolved inside the slot: evaluating it as a
  // parameter default would read runtime state before the queue was taken.
  async stateLoadFile(path: string, tokenCapacity: number | undefined = undefined) {
    return this._runExclusive(
      () => this._stateLoadFileUnlocked(path, tokenCapacity),
      { kind: 'state-load-file' },
    );
  }

  _resolveTokenCapacity(tokenCapacity: unknown) {
    let numericCapacity;
    try {
      numericCapacity = Number(tokenCapacity);
    } catch {
      return this.getContextSize();
    }
    return numericCapacity > 0
      ? Math.trunc(numericCapacity)
      : this.getContextSize();
  }

  async _stateLoadFileUnlocked(path: string, tokenCapacity: number | undefined) {
    tokenCapacity = this._resolveTokenCapacity(tokenCapacity);
    if (!this._workerProxy) {
      return this._runtime!.stateLoadFile(path, tokenCapacity);
    }

    await this._restoreWorkerModelIfMissing();
    return this._callWorker<StateLoadResult>('stateLoadFile', [path, tokenCapacity]);
  }

  async stateSaveBytes(tokens: number[] | ArrayLike<number> = []) {
    return this._runExclusive(
      () => this._stateSaveBytesUnlocked(tokens),
      { kind: 'state-save-bytes' },
    );
  }

  async _stateSaveBytesUnlocked(tokens: number[] | ArrayLike<number> = []) {
    if (!this._workerProxy) {
      return this._runtime!.stateSaveBytes(tokens);
    }

    const normalized = Array.isArray(tokens)
      ? tokens
      : Array.from(tokens || []);
    await this._restoreWorkerModelIfMissing();
    return this._callWorker<Uint8Array>('stateSaveBytes', [normalized]);
  }

  async stateLoadBytes(
    bytes: Uint8Array | ArrayBuffer | ArrayLike<number>,
    tokenCapacity: number | undefined = undefined,
  ) {
    return this._runExclusive(
      () => this._stateLoadBytesUnlocked(bytes, tokenCapacity),
      { kind: 'state-load-bytes' },
    );
  }

  async _stateLoadBytesUnlocked(
    bytes: Uint8Array | ArrayBuffer | ArrayLike<number>,
    tokenCapacity: number | undefined,
  ) {
    tokenCapacity = this._resolveTokenCapacity(tokenCapacity);
    if (!this._workerProxy) {
      return this._runtime!.stateLoadBytes(bytes, tokenCapacity);
    }

    const normalizedBytes = toUint8Array(bytes);
    if (!normalizedBytes || normalizedBytes.length === 0) {
      throw new Error('State bytes are empty.');
    }

    await this._restoreWorkerModelIfMissing();
    const transferableBytes = new Uint8Array(normalizedBytes);
    return this._callWorker<StateLoadResult>(
      'stateLoadBytes',
      [transferableBytes, tokenCapacity],
      null,
      [transferableBytes.buffer],
    );
  }

  async detokenize(tokens: number[] | ArrayLike<number>, special = false) {
    return this._runExclusive(
      () => this._detokenizeUnlocked(tokens, special),
      { kind: 'detokenize' },
    );
  }

  async _detokenizeUnlocked(tokens: number[] | ArrayLike<number>, special = false) {
    if (!this._workerProxy) {
      return this._runtime!.detokenize(tokens, special);
    }

    const normalized = Array.isArray(tokens)
      ? tokens
      : Array.from(tokens || []);

    try {
      await this._restoreWorkerModelIfMissing();
      return await this._callWorker<string>('detokenize', [normalized, special]);
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Detokenization was cancelled.');
      if (!this._isWorkerUnusableError(error)) {
        throw error;
      }
      this._disableWorkerFallback(error);
      await this._waitForWorkerDisposal();
      await this._ensureRuntimeReadyAfterWorkerFallback({}, error);
      return this._runtime!.detokenize(normalized, special);
    }
  }

  async embed(text: string, options: EmbedOptions = {}) {
    return this._runExclusive(
      () => this._embedUnlocked(text, options),
      { kind: 'embedding' },
    );
  }

  async _embedUnlocked(text: string, options: EmbedOptions = {}) {
    if (!this._workerProxy) {
      return this._runtime!.embed(text, options);
    }

    try {
      await this._restoreWorkerModelIfMissing();
      return await this._callWorker<number[]>('embed', [text, options]);
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Embedding was cancelled.');
      if (!this._isWorkerUnusableError(error)) {
        throw error;
      }
      this._disableWorkerFallback(error);
      await this._waitForWorkerDisposal();
      await this._ensureRuntimeReadyAfterWorkerFallback({}, error);
      return this._runtime!.embed(text, options);
    }
  }

  async embedBatch(texts: string[], options: EmbedOptions = {}) {
    return this._runExclusive(
      () => this._embedBatchUnlocked(texts, options),
      { kind: 'embedding-batch' },
    );
  }

  async _embedBatchUnlocked(texts: string[], options: EmbedOptions = {}) {
    const normalized = Array.isArray(texts)
      ? texts
      : Array.from(texts || []) as string[];
    if (!this._workerProxy) {
      return this._runtime!.embedBatch(normalized, options);
    }

    try {
      await this._restoreWorkerModelIfMissing();
      return await this._callWorker<number[][]>('embedBatch', [normalized, options]);
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Batch embedding was cancelled.');
      if (!this._isWorkerUnusableError(error)) {
        throw error;
      }
      this._disableWorkerFallback(error);
      await this._waitForWorkerDisposal();
      await this._ensureRuntimeReadyAfterWorkerFallback({}, error);
      return this._runtime!.embedBatch(normalized, options);
    }
  }

  async scoreNextToken(prompt: string, options: NextTokenScoreOptions = {}) {
    return this._runExclusive(
      () => this._scoreNextTokenUnlocked(prompt, options),
      { kind: 'next-token-scoring' },
    );
  }

  async _scoreNextTokenUnlocked(prompt: string, options: NextTokenScoreOptions = {}) {
    if (!this._workerProxy) {
      return this._runtime!.scoreNextToken(prompt, options);
    }

    try {
      await this._restoreWorkerModelIfMissing();
      return await this._callWorker<NextTokenScores>('scoreNextToken', [prompt, options]);
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Next-token scoring was cancelled.');
      if (!this._isWorkerUnusableError(error)) {
        throw error;
      }
      this._disableWorkerFallback(error);
      await this._waitForWorkerDisposal();
      await this._ensureRuntimeReadyAfterWorkerFallback({}, error);
      return this._runtime!.scoreNextToken(prompt, options);
    }
  }

  getModelMetadata() {
    if (this._workerProxy) {
      return {
        ...(this._metadata || {}),
        'llamadart.webgpu.execution': 'worker',
      };
    }

    const workerReason =
      typeof this._workerFallbackReason === 'string' && this._workerFallbackReason.length > 0
        ? this._workerFallbackReason
        : null;

    return {
      ...(this._metadata || {}),
      'llamadart.webgpu.execution': 'main-thread',
      ...(workerReason == null
        ? {}
        : { 'llamadart.webgpu.worker_fallback_reason': workerReason }),
    };
  }

  // The synchronous getters answer from the facade snapshot in both modes.
  // Reading the direct runtime here would ccall into the singleton core outside
  // the queue and race whichever async operation currently owns it; the snapshot
  // is refreshed by _captureDirectRuntimeState() while the slot is held.
  getContextSize() {
    return this._contextSize || 0;
  }

  isGpuActive() {
    return this._gpuActive;
  }

  getBackendName() {
    return this._backendName;
  }

  setLogLevel(level: string | number) {
    this._throwIfDisposed();
    if (Number.isFinite(level)) {
      this._config.logLevel = Math.max(0, Math.min(4, Math.trunc(level as number)));
    }

    if (this._workerProxy) {
      // Log-level delivery is best effort and runs outside the queue, so a
      // failure must not replace worker topology here: doing so could strand a
      // state-bearing response from the operation that currently owns the queue.
      // The next real queued operation performs the normal fallback under
      // queue ownership.
      this._callWorker('setLogLevel', [level]).catch(() => {});
      if (this._runtime) {
        this._runtime.setLogLevel(level);
      }
      return;
    }
    this._runtime!.setLogLevel(level);
  }

  cancel() {
    // Disposal owns the terminal transition. Cancelling its teardown slot, or
    // reaching into resources it already retired, would turn an out-of-band
    // control signal into a teardown race.
    if (this._lifecycleState === 'disposing' || this._lifecycleState === 'disposed') {
      return;
    }

    if (this._activeOperation) {
      this._cancelOperation(this._activeOperation);
      return;
    }

    if (this._workerProxy) {
      this._sendWorkerCancel(this._workerProxy);
    }

    if (this._runtime) {
      try {
        this._runtime.cancel();
      } catch (_) {
        // Best-effort cancellation.
      }
    }
  }

  /**
   * Marks the bridge disposed synchronously, then tears down on the queue so
   * teardown never runs underneath an operation that still owns the runtime.
   * Work queued before disposal is rejected with a stable lifecycle error rather
   * than dereferencing a torn-down runtime. Not `async`, so repeated calls hand
   * back the identical teardown promise.
   */
  dispose() {
    if (this._disposed) {
      return this._disposePromise!;
    }

    this._disposed = true;
    this._lifecycleState = 'disposing';

    // Reserve the public promise before teardown can run synchronously. A
    // runtime logger (or another teardown callback) may re-enter dispose()
    // before _runExclusive() returns; it must observe this exact promise rather
    // than the still-unassigned field.
    let resolveDispose: (value?: void | PromiseLike<void>) => void = () => {};
    let rejectDispose: (reason?: unknown) => void = () => {};
    this._disposePromise = new Promise((resolve, reject) => {
      resolveDispose = resolve;
      rejectDispose = reject;
    });
    this._notifyDisposalWaiters();

    const teardown = this._runExclusive(
      () => this._disposeUnlocked(),
      { allowDisposed: true, kind: 'dispose' },
    );
    teardown.then(
      () => resolveDispose(),
      (error) => rejectDispose(error),
    );
    return this._disposePromise;
  }

  async _disposeUnlocked() {
    const workerProxy = this._workerProxy;
    this._workerProxy = null;
    if (workerProxy) {
      this._retireWorkerProxy(workerProxy);
    }

    try {
      // Avoid introducing an unnecessary async boundary for the common direct
      // runtime path. Runtime.dispose() performs its synchronous native cleanup
      // (including logger callbacks) before returning its promise; callers that
      // re-enter dispose() during that cleanup must observe the promise reserved
      // by the public method above. Retiring workers still have to settle before
      // the direct runtime is torn down.
      if (this._retiringWorkerDisposals?.size > 0) {
        await this._waitForWorkerDisposal();
      }

      if (this._runtime) {
        const runtime = this._runtime;
        this._runtime = null;
        await runtime.dispose();
      }
    } finally {
      // The getters read this snapshot, so teardown must clear it in both modes.
      this._metadata = {};
      this._contextSize = 0;
      this._gpuActive = false;
      this._backendName = 'WASM (Prototype bridge)';
      this._supportsVision = false;
      this._supportsAudio = false;
      this._loadedModelUrl = null;
      this._loadedModelOptions = null;
      this._loadedMmProjUrl = null;
      this._workerFallbackReason = null;
      this._multimodalWorkerCpuMode = false;
      this._workerModelMissing = false;
      this._decisionHeads?.clear();
      this._forgetLoraAdapters();
      this._lifecycleState = 'disposed';
    }
  }

  async applyChatTemplate(
    messages: Array<Record<string, unknown>>,
    addAssistant = true,
    customTemplate: string | null = null,
  ) {
    return this._runExclusive(
      () => this._applyChatTemplateUnlocked(messages, addAssistant, customTemplate),
      { kind: 'template' },
    );
  }

  async _applyChatTemplateUnlocked(
    messages: Array<Record<string, unknown>>,
    addAssistant = true,
    customTemplate: string | null = null,
  ) {
    if (!this._workerProxy) {
      return this._runtime!.applyChatTemplate(messages, addAssistant, customTemplate);
    }

    try {
      return await this._callWorker<string>('applyChatTemplate', [messages, addAssistant, customTemplate]);
    } catch (error) {
      this._throwIfOperationCancelled(error, 'Chat template operation was cancelled.');
      if (!this._isWorkerUnusableError(error)) {
        throw error;
      }
      // Template fallback runs while this operation owns the queue slot, so
      // worker replacement cannot race another runtime-backed call.
      this._disableWorkerFallback(error);
      return this._runtime!.applyChatTemplate(messages, addAssistant, customTemplate);
    }
  }
}
