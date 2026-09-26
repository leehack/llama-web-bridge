// Direct (same-thread) runtime that drives the Emscripten core.

import { createAbortError, throwIfAborted } from './internal/abort.ts';
import {
  GENERATION_ALREADY_ACTIVE_MESSAGE,
  GENERATION_ALREADY_ACTIVE_RC,
  defaultModelCacheName,
} from './internal/constants.ts';
import {
  NO_COMPLETION_CAPABILITIES,
  completionCapabilitiesFrom,
  requireCompletionCapabilities,
  resolveCompletionSamplingOptions,
} from './internal/completion_options.ts';
import { importCoreFactory } from './internal/core_loader.ts';
import {
  DECISION_API_VERSION,
  decisionHandleFrom,
  decodeDecisionOutputs,
  encodeDecisionSequences,
} from './internal/decision.ts';
import {
  drainResponseWithProgress,
  ensureFsDirectory,
  hasReadableResponseStream,
  inferResponseTotalBytes,
  isRetryableStreamNetworkError,
  sumProgressValues,
  unlinkFsFile,
  writeResponseToFsFileWithProgress,
} from './internal/download.ts';
import { isCrossOriginIsolatedRuntime, isSafariUserAgent } from './internal/environment.ts';
import { decodeImageBytesToRgb } from './internal/image.ts';
import { logLevelForName, logThresholdForConfiguredLevel } from './internal/logging.ts';
import {
  LORA_API_VERSION,
  LORA_LOAD_ABORT_MESSAGE,
  loraHandleFrom,
  loraScaleFrom,
} from './internal/lora.ts';
import {
  basenameFromUrl,
  cloneModelSource,
  expandModelShardUrls,
  hasModelSource,
  normalizeAbsoluteUrl,
} from './internal/model_source.ts';
import { nextTokenCandidateIds, scoredTokensFrom } from './internal/next_token_scores.ts';
import {
  parseBooleanFlag,
  parseEnumValue,
  parseInteger,
  parseOptionalBooleanFlag,
  parsePositiveInteger,
  parsePositiveNumber,
} from './internal/parse.ts';
import {
  buildPromptFromMessages,
  looksLikeCorruptedGeneration,
  trimUnstableUtf8Tail,
} from './internal/text.ts';
import { bufferSourceBytes, isInt32, toFloat32Array, toUint8Array } from './internal/typed_values.ts';
import type { ProgressCallback } from './internal/download.ts';
import type { ModelSource } from './internal/model_source.ts';
import type { CcallArgType, LlamaCoreModule, LogMethod } from './internal/types.ts';
import type {
  CompletionCapabilities,
  CompletionFinishReason,
  CompletionOptions,
  DecisionCapabilities,
  DecisionHeadInfo,
  DecisionHeadOptions,
  EmbedOptions,
  LlamaWebGpuBridgeConfig,
  LoadModelOptions,
  LoraAdapterCapabilities,
  LoraAdapterInfo,
  LoraAdapterLoadOptions,
  NextTokenScoreOptions,
  TextToSpeechCapabilities,
  TextToSpeechOptions,
  TextToSpeechProgress,
} from './llama_webgpu_bridge.d.ts';

// The optional `logger` config entry: any subset of the console methods.
type BridgeLogger = Partial<Record<LogMethod, (message: unknown) => void>>;

// Fetch and cache knobs read from a load, prefetch or request options bag.
// Callers may pass anything, so each value is coerced where it is read.
// Load options as the bridge passes them: a null signal means none.
export type RuntimeLoadModelOptions = {
  [K in keyof LoadModelOptions as K extends 'signal' ? never : K]: LoadModelOptions[K];
} & { signal?: AbortSignal | null };

interface TransferOptions {
  cacheName?: unknown;
  fetchTimeoutMs?: unknown;
  streamChunkTimeoutMs?: unknown;
  remoteFetchThresholdBytes?: unknown;
  remoteFetchChunkBytes?: unknown;
  forceRemoteFetchBackend?: unknown;
}

// A model shard request: the load options plus the resume/stream controls
// the loader adds per attempt.
interface CachedModelResponseOptions extends RuntimeLoadModelOptions {
  requireReadableStream?: boolean;
  requestHeaders?: Record<string, string> | null;
}

// LoRA load options as the bridge passes them: a null signal means none, and
// the fetch knobs a model load reads are honoured too.
type RuntimeLoraAdapterLoadOptions = {
  [K in keyof LoraAdapterLoadOptions as K extends 'signal' ? never : K]: LoraAdapterLoadOptions[K];
} & { signal?: AbortSignal | null } & TransferOptions;

// Per-request image downscale limits; they override the configured ones.
interface MediaImageLimitOptions {
  disableImageDownscale?: unknown;
  mediaMaxImagePixels?: unknown;
  mediaMaxImageEdge?: unknown;
}

// Completion options as the runtime reads them: the public ones, the media
// limits, the chunk size a CPU-fallback reload reuses, and the recursion guard
// a recovery retry sets.
interface RuntimeCompletionOptions extends CompletionOptions, MediaImageLimitOptions {
  remoteFetchChunkBytes?: unknown;
  _llamadartGenerationRecoveryAttempted?: boolean;
}

// A multimodal part as received. Parts are not validated upstream, so every
// field is read defensively.
interface MediaPartInput {
  type?: unknown;
  bytes?: unknown;
  samples?: unknown;
  url?: unknown;
  width?: unknown;
  height?: unknown;
}

// What a ranged probe learns about a remote model before a fetch-backed load.
interface RemoteFetchProbe {
  resolvedUrl: string | null;
  sizeBytes: number | null;
}

// Debug state the core's fetch backend publishes on globalThis.
interface FetchBackendDebugGlobals {
  __llamadartFetchBackendLastError?: unknown;
  __llamadartFetchBackendStats?: {
    reads?: unknown;
    getSize?: unknown;
    ranges?: unknown;
    wholeFileFallbacks?: unknown;
    errors?: unknown;
  };
}

const textEncoder = new TextEncoder();

export class LlamaWebGpuBridgeRuntime {
  declare _config: LlamaWebGpuBridgeConfig;
  declare _core: LlamaCoreModule | null;
  declare _backendLabels: string[];
  declare _gpuActive: boolean;
  declare _modelPath: string | null;
  declare _modelPaths: string[];
  declare _modelBytes: number;
  declare _mmProjPath: string | null;
  declare _mmSupportsVision: boolean;
  declare _mmSupportsAudio: boolean;
  declare _mediaFileCounter: number;
  declare _stateFileCounter: number;
  declare _decisionFileCounter: number;
  declare _loraFileCounter: number;
  declare _loraAdaptersLoaded: boolean;
  declare _stagedMediaPaths: string[];
  declare _nCtx: number;
  declare _abortRequested: boolean;
  declare _textToSpeechActive: boolean;
  declare _textToSpeechDone: Promise<void> | null;
  declare _resolveTextToSpeechDone: (() => void) | null;
  declare _runtimeNotes: string[];
  declare _threadPoolSizeHint: number | null;
  declare _threads: number;
  declare _threadsBatch: number;
  declare _nBatch: number;
  declare _nUbatch: number;
  declare _nGpuLayers: number;
  declare _nSeqMax: number;
  declare _useMmap: boolean;
  declare _useMlock: boolean;
  declare _flashAttention: number;
  declare _cacheTypeK: number;
  declare _cacheTypeV: number;
  declare _kvUnified: number;
  declare _ropeFrequencyBase: number;
  declare _ropeFrequencyScale: number;
  declare _splitMode: number;
  declare _mainGpu: number;
  declare _isSafari: boolean;
  declare _coreVariant: string;
  declare _preferMemory64: boolean;
  declare _modelSource: string;
  declare _modelCacheState: string;
  declare _modelCacheName: string;
  declare _loadedModelUrl: ModelSource | null;
  declare _mmProjSourceUrl: string | null;
  declare _suppressedWarmupWarningCount: number;
  declare _didReportWarmupWarningSuppression: boolean;
  declare _remoteFetchThresholdBytes: number;
  declare _remoteFetchChunkBytes: number;
  declare _mediaMaxImagePixels: number;
  declare _mediaMaxImageEdge: number;
  declare _disableImageDownscale: boolean;
  declare _activeTransferAbortController: AbortController | null;
  declare _lastCoreErrorText: string;
  declare _lastCoreErrorHint: string;
  declare _logLevel: number;

  constructor(config: LlamaWebGpuBridgeConfig = {}) {
    this._config = config;
    this._core = null;
    this._backendLabels = [];
    this._gpuActive = false;
    this._modelPath = null;
    this._modelPaths = [];
    this._modelBytes = 0;
    this._mmProjPath = null;
    this._mmSupportsVision = false;
    this._mmSupportsAudio = false;
    this._mediaFileCounter = 0;
    this._stateFileCounter = 0;
    this._decisionFileCounter = 0;
    this._loraFileCounter = 0;
    this._loraAdaptersLoaded = false;
    this._stagedMediaPaths = [];
    this._nCtx = 4096;
    this._abortRequested = false;
    this._textToSpeechActive = false;
    this._textToSpeechDone = null;
    this._resolveTextToSpeechDone = null;
    this._runtimeNotes = [];
    this._threadPoolSizeHint = Number(config.threadPoolSize) > 0
      ? Math.max(1, Math.trunc(Number(config.threadPoolSize)))
      : null;
    const requestedThreads = Number(config.threads) > 0
      ? Number(config.threads)
      : this._resolveAutoThreadCount();
    this._threads = this._capThreadsToPool(requestedThreads);
    const requestedThreadsBatch = Number(config.threadsBatch) > 0
      ? Number(config.threadsBatch)
      : this._threads;
    this._threadsBatch = this._capThreadsToPool(
      requestedThreadsBatch,
      { noteTag: 'threads_batch_capped_pool' },
    );
    this._nBatch = Number(config.nBatch) > 0
      ? Math.max(32, Math.trunc(Number(config.nBatch)))
      : 0;
    this._nUbatch = Number(config.nUbatch) > 0
      ? Math.max(32, Math.trunc(Number(config.nUbatch)))
      : 0;
    this._nGpuLayers = Number.isFinite(config.nGpuLayers)
      ? Number(config.nGpuLayers)
      : -1;
    this._nSeqMax = 0;
    this._useMmap = false;
    this._useMlock = false;
    this._flashAttention = -1;
    this._cacheTypeK = 1;
    this._cacheTypeV = 1;
    this._kvUnified = -1;
    this._ropeFrequencyBase = 0;
    this._ropeFrequencyScale = 0;
    this._splitMode = -1;
    this._mainGpu = -1;
    this._isSafari = isSafariUserAgent(this._config.userAgent ?? globalThis.navigator?.userAgent ?? '');
    this._coreVariant = 'uninitialized';
    this._preferMemory64 = this._config.preferMemory64 !== false;
    this._modelSource = 'network';
    this._modelCacheState = 'disabled';
    this._modelCacheName = defaultModelCacheName;
    this._loadedModelUrl = null;
    this._mmProjSourceUrl = null;
    this._suppressedWarmupWarningCount = 0;
    this._didReportWarmupWarningSuppression = false;
    this._remoteFetchThresholdBytes = Number(config.remoteFetchThresholdBytes) > 0
      ? Number(config.remoteFetchThresholdBytes)
      : 1900 * 1024 * 1024;
    this._remoteFetchChunkBytes = Number(config.remoteFetchChunkBytes) > 0
      ? Number(config.remoteFetchChunkBytes)
      : 16 * 1024 * 1024;
    this._mediaMaxImagePixels = Number(config.mediaMaxImagePixels) > 0
      ? Math.max(65536, Math.min(33554432, Math.trunc(Number(config.mediaMaxImagePixels))))
      : (1024 * 1024);
    this._mediaMaxImageEdge = Number(config.mediaMaxImageEdge) > 0
      ? Math.max(64, Math.min(16384, Math.trunc(Number(config.mediaMaxImageEdge))))
      : 1280;
    this._disableImageDownscale = config.disableImageDownscale === true;
    this._activeTransferAbortController = null;
    this._lastCoreErrorText = '';
    this._lastCoreErrorHint = '';
    this._logLevel = Number.isFinite(config.logLevel)
      ? Math.max(0, Math.min(4, Math.trunc(config.logLevel as number)))
      : 2;
  }

  static supportsSafariAdaptiveGpu = true;

  _pushRuntimeNote(note: string) {
    if (typeof note !== 'string' || note.length === 0) {
      return;
    }

    if (!Array.isArray(this._runtimeNotes)) {
      this._runtimeNotes = [];
    }

    if (!this._runtimeNotes.includes(note)) {
      this._runtimeNotes.push(note);
    }
  }

  _detectThreadPoolSizeFromCore() {
    if (!this._coreSupportsPthreads()) {
      return 1;
    }

    const core = this._core;
    if (!core || typeof core !== 'object') {
      return null;
    }

    try {
      if (typeof core.ccall === 'function') {
        const compiledPoolSize = Number(
          core.ccall('llamadart_webgpu_pthread_pool_size', 'number', [], []),
        );
        if (Number.isFinite(compiledPoolSize) && compiledPoolSize > 0) {
          return Math.max(1, Math.trunc(compiledPoolSize));
        }
      }
    } catch (_) {
      // Ignore lookup failures and fall back to runtime heuristics.
    }

    try {
      const pThread = core.PThread;
      if (!pThread || typeof pThread !== 'object') {
        return null;
      }

      const unused = Array.isArray(pThread.unusedWorkers)
        ? pThread.unusedWorkers.length
        : 0;
      const running = Array.isArray(pThread.runningWorkers)
        ? pThread.runningWorkers.length
        : 0;
      const total = unused + running;
      return total > 0 ? total : null;
    } catch (_) {
      return null;
    }
  }

  _coreSupportsPthreads() {
    const core = this._core;
    if (!core || typeof core !== 'object') {
      return false;
    }

    try {
      if (typeof core.ccall === 'function') {
        const compiledWithPthreads = Number(
          core.ccall('llamadart_webgpu_supports_pthreads', 'number', [], []),
        );
        if (Number.isFinite(compiledWithPthreads)) {
          return compiledWithPthreads !== 0;
        }
      }
    } catch (_) {
      // Ignore lookup failures and fall back to runtime heuristics.
    }

    try {
      const wasmBuffer = core.wasmMemory?.buffer;
      if (
        typeof SharedArrayBuffer === 'function'
        && wasmBuffer instanceof SharedArrayBuffer
      ) {
        return true;
      }
    } catch (_) {
      // Ignore wasmMemory inspection failures and fall back.
    }

    try {
      const heapBuffer = core.HEAP8?.buffer || core.HEAPU8?.buffer;
      if (
        typeof SharedArrayBuffer === 'function'
        && heapBuffer instanceof SharedArrayBuffer
      ) {
        return true;
      }
    } catch (_) {
      // Ignore HEAP buffer inspection failures and fall back.
    }

    try {
      const pThread = core.PThread;
      if (!pThread || typeof pThread !== 'object') {
        return false;
      }

      return Array.isArray(pThread.unusedWorkers)
        || Array.isArray(pThread.runningWorkers)
        || typeof pThread.allocateUnusedWorker === 'function';
    } catch (_) {
      return false;
    }
  }

  _syncThreadPoolSizeHintFromCore() {
    const detected = this._detectThreadPoolSizeFromCore();
    if (!Number.isFinite(detected) || detected! <= 0) {
      return;
    }

    this._threadPoolSizeHint = Math.max(1, Math.trunc(detected!));
  }

  _resolveAutoThreadCount() {
    const hardwareThreads = Number(globalThis.navigator?.hardwareConcurrency);
    if (Number.isFinite(hardwareThreads) && hardwareThreads > 0) {
      return Math.max(1, Math.min(8, Math.trunc(hardwareThreads)));
    }

    return 4;
  }

  _capThreadsToPool(candidate: number, { noteTag = 'threads_capped_pool' }: { noteTag?: string } = {}) {
    let resolved = Number(candidate);
    if (!Number.isFinite(resolved) || resolved <= 0) {
      resolved = 1;
    }

    resolved = Math.max(1, Math.trunc(resolved));
    const poolSize = Number(this._threadPoolSizeHint);
    if (Number.isFinite(poolSize) && poolSize > 0 && resolved > poolSize) {
      if (noteTag) {
        this._pushRuntimeNote(`${noteTag}:${poolSize}`);
      }
      return poolSize;
    }

    return resolved;
  }

  _isVerboseWarmupWarning(text: string) {
    const lowered = String(text || '').toLowerCase();
    if (lowered.length === 0) {
      return false;
    }

    if (lowered.includes('warmup:')) {
      return true;
    }
    if (lowered.includes('please report this on github as an issue')) {
      return true;
    }
    if (lowered.includes('warning: ref:')) {
      return true;
    }
    if (lowered.includes('github.com/ggml-org/llama.cpp/pull/')) {
      return true;
    }
    if (lowered.includes('****************')) {
      return true;
    }

    return false;
  }

  _emitSuppressedWarmupWarningSummaryIfNeeded() {
    if (this._suppressedWarmupWarningCount <= 0) {
      return;
    }

    if (this._logLevel <= 2) {
      this._emitLogger(
        'log',
        `info: suppressed ${this._suppressedWarmupWarningCount} verbose warmup log lines (set bridge/runtime log level to Debug to inspect full warmup trace).`,
      );
    }
    this._suppressedWarmupWarningCount = 0;
  }

  _shouldAttemptGenerationRecovery(
    errorText: string,
    options: RuntimeCompletionOptions = {},
    generated = 0,
  ) {
    if (generated > 0) {
      return false;
    }

    if (this._nGpuLayers <= 0) {
      return false;
    }

    if (this._coreVariant !== 'wasm64') {
      return false;
    }

    if (!hasModelSource(this._loadedModelUrl)) {
      return false;
    }

    if (options._llamadartGenerationRecoveryAttempted === true) {
      return false;
    }

    if (Array.isArray(options.parts) && options.parts.length > 0) {
      return false;
    }

    const lowered = String(errorText || '').toLowerCase();
    if (lowered.includes('failed to decode')) {
      return true;
    }
    if (lowered.includes('failed to compute graph')) {
      return true;
    }
    if (lowered.includes('ggml_backend_sched_graph_compute_async failed')) {
      return true;
    }

    return false;
  }

  _isContextLimitGenerationError(errorText = '') {
    const lowered = [
      String(errorText || ''),
      String(this._lastCoreErrorText || ''),
      String(this._lastCoreErrorHint || ''),
    ]
      .join(' ')
      .toLowerCase();

    return (
      lowered.includes('failed to find a memory slot for batch')
      || lowered.includes('failed to prepare attention batches')
      || lowered.includes('context overflow')
      || lowered.includes('context full')
      || lowered.includes('kv cache full')
      || lowered.includes('insufficient kv')
      || lowered.includes('no kv slot')
    );
  }

  async _recoverGenerationWithCpuFallback(options: RuntimeCompletionOptions = {}) {
    const modelUrl = cloneModelSource(this._loadedModelUrl);
    if (!hasModelSource(modelUrl)) {
      return false;
    }
    // A reload frees the model's LoRA adapters, so the retry would silently
    // generate without them.
    if (this._loraAdaptersLoaded) {
      this._runtimeNotes.push('generation_recovery_cpu_skipped_lora');
      return false;
    }

    this._runtimeNotes.push('generation_recovery_cpu_attempt');
    this._emitLogger(
      'warn',
      'warning: generation failed on wasm64/WebGPU; retrying by reloading model with CPU fallback for stability.',
    );

    const previousPreferMemory64 = this._preferMemory64;
    const previousMMProjSourceUrl = this._mmProjSourceUrl;

    try {
      this._preferMemory64 = false;
      await this.loadModelFromUrl(modelUrl, {
        nCtx: this._nCtx,
        nThreads: this._threads,
        nGpuLayers: 0,
        useCache: true,
        forceRemoteFetchBackend: false,
        remoteFetchChunkBytes: this._resolveRemoteFetchChunkBytes(options),
        safariGpuProbe: false,
        signal: options.signal,
      });

      if (typeof previousMMProjSourceUrl === 'string' && previousMMProjSourceUrl.length > 0) {
        try {
          await this.loadMultimodalProjector(previousMMProjSourceUrl);
        } catch (_) {
          this._runtimeNotes.push('generation_recovery_mmproj_reload_failed');
        }
      }

      this._runtimeNotes.push('generation_recovery_cpu_applied');
      return true;
    } catch (_) {
      this._runtimeNotes.push('generation_recovery_cpu_failed');
      return false;
    } finally {
      this._preferMemory64 = previousPreferMemory64;
    }
  }

  _loggerFor(level: LogMethod) {
    const logger = this._config?.logger as BridgeLogger | undefined;
    const fallback = (typeof console !== 'undefined') ? console : null;

    if (logger && typeof logger[level] === 'function') {
      return logger[level]!.bind(logger);
    }

    if (!fallback) {
      return () => {};
    }

    if (typeof fallback[level] === 'function') {
      return fallback[level].bind(fallback);
    }

    if (typeof fallback.log === 'function') {
      return fallback.log.bind(fallback);
    }

    return () => {};
  }

  _shouldEmitLoggerLevel(level: LogMethod) {
    const current = Number(this._logLevel);
    if (!Number.isFinite(current) || current < 0) {
      return true;
    }

    const threshold = logThresholdForConfiguredLevel(
      Math.max(0, Math.min(4, Math.trunc(current))),
    );
    if (threshold > 3) {
      return false;
    }

    return logLevelForName(level) >= threshold;
  }

  _emitLogger(level: LogMethod, message: unknown) {
    if (!this._shouldEmitLoggerLevel(level)) {
      return;
    }

    try {
      this._loggerFor(level)(message);
    } catch (_) {
      // Logger callbacks are best-effort only.
    }
  }

  _classifyCoreErrorLine(text: string) {
    const trimmed = String(text ?? '').trim();
    if (trimmed.length === 0) {
      return 'ignore';
    }

    if (this._isVerboseWarmupWarning(trimmed)) {
      return 'warmup';
    }

    const lowered = trimmed.toLowerCase();
    if (
      lowered.startsWith('warning')
      || lowered.startsWith('warn:')
      || lowered.includes(' warning:')
    ) {
      return 'warn';
    }

    if (
      lowered.startsWith('error')
      || lowered.startsWith('err:')
      || lowered.includes(' error:')
      || lowered.includes('failed')
      || lowered.includes('exception')
      || lowered.includes('abort')
      || lowered.includes('fatal')
      || lowered.includes('out of memory')
      || lowered.includes('invalid')
    ) {
      return 'error';
    }

    return 'info';
  }

  _applyCoreLogLevel() {
    if (!this._core) {
      return;
    }

    try {
      this._core.ccall(
        'llamadart_webgpu_set_log_level',
        null,
        ['number'],
        [this._logLevel],
      );
    } catch (_) {
      // Older core builds may not expose log-level setter.
    }
  }

  _coreErrorMessage(prefix: string, fallbackCode = 0) {
    try {
      const err = this._core?.ccall('llamadart_webgpu_last_error', 'string', [], []);
      if (err) {
        return `${prefix}: ${err}`;
      }
    } catch (_) {
      // Ignore nested error retrieval failures.
    }
    return `${prefix} (code=${fallbackCode})`;
  }

  _resolveCacheName(options: TransferOptions = {}) {
    if (typeof options.cacheName === 'string' && options.cacheName.trim().length > 0) {
      return options.cacheName.trim();
    }

    if (typeof this._config.cacheName === 'string' && this._config.cacheName.trim().length > 0) {
      return this._config.cacheName.trim();
    }

    return defaultModelCacheName;
  }

  _isWorkerRuntime() {
    return typeof WorkerGlobalScope !== 'undefined'
      && globalThis instanceof WorkerGlobalScope;
  }

  _resolveRemoteFetchThresholdBytes(options: TransferOptions = {}) {
    const candidate = Number(options.remoteFetchThresholdBytes);
    if (Number.isFinite(candidate) && candidate > 0) {
      return Math.trunc(candidate);
    }
    return Math.trunc(this._remoteFetchThresholdBytes);
  }

  _resolveRemoteFetchChunkBytes(options: TransferOptions = {}) {
    const candidate = Number(options.remoteFetchChunkBytes);
    if (Number.isFinite(candidate) && candidate > 0) {
      return Math.max(16 * 1024, Math.trunc(candidate));
    }
    return Math.max(16 * 1024, Math.trunc(this._remoteFetchChunkBytes));
  }

  _canUseRemoteFetchBackend(options: TransferOptions = {}) {
    if (options.forceRemoteFetchBackend === false) {
      return false;
    }

    if (!this._isWorkerRuntime()) {
      return false;
    }

    if (options.forceRemoteFetchBackend === true) {
      return true;
    }

    return this._config.allowAutoRemoteFetchBackend === true;
  }

  async _tryHeadContentLength(url: string) {
    function parseSizeFromHeaders(headers: Headers | null | undefined) {
      if (!headers) {
        return 0;
      }

      const length = Number(headers.get('content-length')) || 0;
      if (length > 0) {
        return length;
      }

      const linkedSize = Number(headers.get('x-linked-size')) || 0;
      if (linkedSize > 0) {
        return linkedSize;
      }

      const contentRange = String(headers.get('content-range') || '');
      const slash = contentRange.lastIndexOf('/');
      if (slash >= 0 && slash + 1 < contentRange.length) {
        const total = Number(contentRange.slice(slash + 1)) || 0;
        if (total > 0) {
          return total;
        }
      }

      return 0;
    }

    try {
      const response = await this._fetchWithTimeout(url, {
        method: 'HEAD',
        cache: 'no-store',
      }, 45000);
      if (!response.ok) {
        throw new Error('HEAD request failed');
      }

      const size = parseSizeFromHeaders(response.headers);
      if (size > 0) {
        return size;
      }
    } catch (_) {
      // ignore best-effort HEAD failures
    }

    try {
      const probe = await this._fetchWithTimeout(url, {
        headers: {
          Range: 'bytes=0-0',
        },
        cache: 'no-store',
      }, 45000);

      if (!(probe.ok || probe.status === 206)) {
        return null;
      }

      const size = parseSizeFromHeaders(probe.headers);
      if (size > 0) {
        this._runtimeNotes.push('model_fetch_size_probe');
        return size;
      }
    } catch (_) {
      // ignore best-effort range probe failures
    }

    return null;
  }

  async _resolveRemoteFetchUrl(url: string): Promise<RemoteFetchProbe | null> {
    function parseSizeFromHeaders(headers: Headers | null | undefined) {
      if (!headers) {
        return 0;
      }

      const contentRange = String(headers.get('content-range') || '');
      const slash = contentRange.lastIndexOf('/');
      if (slash >= 0 && slash + 1 < contentRange.length) {
        const total = Number(contentRange.slice(slash + 1)) || 0;
        if (total > 0) {
          return total;
        }
      }

      const linkedSize = Number(headers.get('x-linked-size')) || 0;
      if (linkedSize > 0) {
        return linkedSize;
      }

      const length = Number(headers.get('content-length')) || 0;
      if (length > 0) {
        return length;
      }

      return 0;
    }

    try {
      const probe = await this._fetchWithTimeout(url, {
        headers: {
          Range: 'bytes=0-0',
        },
        cache: 'no-store',
      }, 45000);

      if (!(probe.ok || probe.status === 206)) {
        return null;
      }

      const resolvedUrl =
        typeof probe.url === 'string' && probe.url.length > 0
          ? probe.url
          : null;
      const sizeBytes = parseSizeFromHeaders(probe.headers);

      return {
        resolvedUrl,
        sizeBytes: sizeBytes > 0 ? sizeBytes : null,
      };
    } catch (_) {
      return null;
    }
  }

  _resolveNativeLoadOptions(options: RuntimeLoadModelOptions = {}) {
    this._nSeqMax = parsePositiveInteger(options.nSeqMax);
    this._useMmap = parseBooleanFlag(options.useMmap, false);
    this._useMlock = parseBooleanFlag(options.useMlock, false);
    this._flashAttention = parseEnumValue(options.flashAttention, [-1, 0, 1], -1);
    this._cacheTypeK = parseEnumValue(options.cacheTypeK, [1, 2, 8], 1);
    this._cacheTypeV = parseEnumValue(options.cacheTypeV, [1, 2, 8], 1);
    this._kvUnified = parseOptionalBooleanFlag(options.kvUnified);
    this._ropeFrequencyBase = parsePositiveNumber(options.ropeFrequencyBase);
    this._ropeFrequencyScale = parsePositiveNumber(options.ropeFrequencyScale);
    this._splitMode = parseEnumValue(options.splitMode, [0, 1, 2, 3], -1);
    this._mainGpu = parseInteger(options.mainGpu, -1);
    if (this._mainGpu < 0) {
      this._mainGpu = -1;
    }

    const wantsQuantizedKvCache = this._cacheTypeK !== 1 || this._cacheTypeV !== 1;
    if (this._flashAttention === 0 && wantsQuantizedKvCache) {
      throw new Error(
        'Non-F16 KV cache requires flashAttention to be auto or enabled.',
      );
    }
    if (this._flashAttention === -1 && wantsQuantizedKvCache) {
      this._flashAttention = 1;
      this._runtimeNotes.push('flash_attention:auto_enabled_for_kv_cache');
    }
    if (this._kvUnified < 0 && this._nSeqMax > 1) {
      this._kvUnified = 1;
      this._runtimeNotes.push('kv_unified:auto_enabled_for_sequences');
    }
  }

  _nativeLoadOptionValues() {
    return [
      this._nSeqMax,
      this._useMmap ? 1 : 0,
      this._useMlock ? 1 : 0,
      this._flashAttention,
      this._cacheTypeK,
      this._cacheTypeV,
      this._kvUnified,
      this._ropeFrequencyBase,
      this._ropeFrequencyScale,
      this._splitMode,
      this._mainGpu,
    ];
  }

  _nativeLoadOptionTypes() {
    return this._nativeLoadOptionValues().map(() => 'number' as CcallArgType);
  }

  async _tryLoadModelFromRemoteFetchBackend(
    core: LlamaCoreModule,
    url: string,
    options: RuntimeLoadModelOptions = {},
  ) {
    if (!this._canUseRemoteFetchBackend(options)) {
      return { loaded: false, sizeBytes: null };
    }

    const thresholdBytes = this._resolveRemoteFetchThresholdBytes(options);
    const chunkBytes = this._resolveRemoteFetchChunkBytes(options);
    const forceRemote = options.forceRemoteFetchBackend === true;

    let sizeBytes: number | null = Number(options.modelBytesHint);
    if (!Number.isFinite(sizeBytes) || sizeBytes <= 0) {
      sizeBytes = await this._tryHeadContentLength(url);
    }

    if (!forceRemote) {
      if (Number.isFinite(sizeBytes) && sizeBytes! > 0 && sizeBytes! < thresholdBytes) {
        this._runtimeNotes.push('model_fetch_backend_skipped_small');
        return { loaded: false, sizeBytes: sizeBytes };
      }

      if (!Number.isFinite(sizeBytes) || sizeBytes! <= 0) {
        this._runtimeNotes.push('model_fetch_backend_size_unknown');
        this._runtimeNotes.push('model_fetch_backend_unknown_size_attempt');
      }
    }

    this._runtimeNotes.push('model_fetch_backend_attempt');
    this._runtimeNotes.push(`model_fetch_chunk:${chunkBytes}`);
    this._lastCoreErrorHint = '';

    let remoteFetchUrl = url;
    const resolvedProbe = await this._resolveRemoteFetchUrl(url);
    if (resolvedProbe?.resolvedUrl) {
      remoteFetchUrl = resolvedProbe.resolvedUrl;
      if (remoteFetchUrl !== url) {
        this._runtimeNotes.push('model_fetch_backend_resolved_url');
      }
    }
    if ((!Number.isFinite(sizeBytes) || sizeBytes! <= 0) &&
        Number.isFinite(resolvedProbe?.sizeBytes) &&
        resolvedProbe!.sizeBytes! > 0) {
      sizeBytes = resolvedProbe!.sizeBytes;
    }

    try {
      (globalThis as FetchBackendDebugGlobals).__llamadartFetchBackendLastError = null;
    } catch (_) {
      // ignore debug-state reset failures
    }

    if (typeof options.progressCallback === 'function') {
      options.progressCallback({ loaded: 0, total: Number.isFinite(sizeBytes) ? sizeBytes! : 0 });
    }

    try {
      const rc = Number(
        await core.ccall(
          'llamadart_webgpu_load_model_from_url',
          'number',
          [
            'string',
            'number',
            'number',
            'number',
            'number',
            'number',
            'number',
            'number',
            ...this._nativeLoadOptionTypes(),
          ],
          [
            remoteFetchUrl,
            this._nCtx,
            this._threads,
            this._threadsBatch,
            this._nBatch,
            this._nUbatch,
            this._nGpuLayers,
            chunkBytes,
            ...this._nativeLoadOptionValues(),
          ],
          { async: true },
        ),
      );

      if (rc !== 0) {
        throw new Error(this._coreErrorMessage('Fetch-backed model load failed', rc));
      }

      this._modelSource = 'network-fetch';
      this._modelPath = null;
      this._modelBytes = Number.isFinite(sizeBytes) && sizeBytes! > 0 ? Math.trunc(sizeBytes!) : 1;
      this._runtimeNotes.push('model_source_fetch_backend');

      if (typeof options.progressCallback === 'function') {
        const resolved = Number.isFinite(sizeBytes) && sizeBytes! > 0 ? sizeBytes! : 1;
        options.progressCallback({ loaded: resolved, total: resolved });
      }

      return {
        loaded: true,
        sizeBytes: Number.isFinite(sizeBytes) ? sizeBytes : null,
      };
    } catch (error) {
      const text = String(error || '').toLowerCase();

      if (text.includes('aborted(native code called abort())')) {
        try {
          const stderrText = String(
            this._lastCoreErrorHint || this._lastCoreErrorText || '',
          ).trim();
          if (stderrText.length > 0) {
            const token = stderrText
              .slice(0, 120)
              .replace(/[\s;=]+/g, '_')
              .replace(/[^a-zA-Z0-9._:-]/g, '');
            if (token.length > 0) {
              this._runtimeNotes.push(`model_core_stderr:${token}`);
            }
          }
        } catch (_) {
          // ignore stderr capture failures on abort path
        }

        try {
          const fetchError = String(
            (globalThis as FetchBackendDebugGlobals).__llamadartFetchBackendLastError || '',
          ).trim();
          if (fetchError.length > 0) {
            const token = fetchError
              .slice(0, 120)
              .replace(/[\s;=]+/g, '_')
              .replace(/[^a-zA-Z0-9._:-]/g, '');
            if (token.length > 0) {
              this._runtimeNotes.push(`model_fetch_js_error:${token}`);
            }
          }
        } catch (_) {
          // ignore fetch-backend debug-state probe failures
        }

        try {
          const stats = (globalThis as FetchBackendDebugGlobals).__llamadartFetchBackendStats;
          if (stats && typeof stats === 'object') {
            const reads = Number(stats.reads) || 0;
            const getSize = Number(stats.getSize) || 0;
            const ranges = Number(stats.ranges) || 0;
            const fallbacks = Number(stats.wholeFileFallbacks) || 0;
            const errors = Number(stats.errors) || 0;
            this._runtimeNotes.push(
              `model_fetch_stats:r${reads}_s${getSize}_q${ranges}_f${fallbacks}_e${errors}`,
            );
          }
        } catch (_) {
          // ignore fetch stats probe failures
        }

        try {
          const coreError = String(
            this._core?.ccall('llamadart_webgpu_last_error', 'string', [], []) || '',
          ).trim();
          if (coreError.length > 0) {
            const token = coreError
              .slice(0, 120)
              .replace(/[\s;=]+/g, '_')
              .replace(/[^a-zA-Z0-9._:-]/g, '');
            if (token.length > 0) {
              this._runtimeNotes.push(`model_fetch_core_error:${token}`);
            }
          }
        } catch (_) {
          // ignore last-error probing failures on abort path
        }

        this._runtimeNotes.push('model_fetch_backend_abort');
        throw error;
      }

      if (text.includes('fetch-backed model load failed')) {
        this._runtimeNotes.push('model_fetch_backend_failed');
        return {
          loaded: false,
          sizeBytes: Number.isFinite(sizeBytes) ? sizeBytes : null,
        };
      }

      if (
        text.includes('load_model_from_url')
        && (text.includes('not found')
          || text.includes('undefined symbol')
          || text.includes('missing function'))
      ) {
        this._runtimeNotes.push('model_fetch_backend_unavailable');
        return {
          loaded: false,
          sizeBytes: Number.isFinite(sizeBytes) ? sizeBytes : null,
        };
      }

      if (text.includes('worker-thread bridge runtime')) {
        this._runtimeNotes.push('model_fetch_backend_requires_worker');
        return {
          loaded: false,
          sizeBytes: Number.isFinite(sizeBytes) ? sizeBytes : null,
        };
      }

      throw error;
    }
  }

  _beginTransferAbortController() {
    if (typeof AbortController !== 'function') {
      this._activeTransferAbortController = null;
      return null;
    }

    if (this._activeTransferAbortController) {
      try {
        this._activeTransferAbortController.abort();
      } catch (_) {
        // ignore abort failures on stale controllers
      }
    }

    const controller = new AbortController();
    this._activeTransferAbortController = controller;
    return controller;
  }

  _clearTransferAbortController(controller: AbortController | null) {
    if (this._activeTransferAbortController === controller) {
      this._activeTransferAbortController = null;
    }
  }

  _resolveFetchTimeoutMs(options: TransferOptions = {}, defaultTimeoutMs = 180000) {
    const configured = Number(options.fetchTimeoutMs);
    if (Number.isFinite(configured) && configured > 0) {
      return Math.max(10000, Math.min(1800000, Math.trunc(configured)));
    }

    return defaultTimeoutMs;
  }

  _resolveStreamChunkTimeoutMs(options: TransferOptions = {}, defaultTimeoutMs = 90000) {
    const configured = Number(options.streamChunkTimeoutMs);
    if (Number.isFinite(configured) && configured > 0) {
      return Math.max(5000, Math.min(300000, Math.trunc(configured)));
    }

    return defaultTimeoutMs;
  }

  _resolveCoreInitTimeoutMs() {
    const configured = Number(this._config.coreInitTimeoutMs);
    if (!Number.isFinite(configured) || configured <= 0) {
      return 90000;
    }

    return Math.max(10000, Math.min(600000, Math.trunc(configured)));
  }

  _resolveMediaImageMaxPixels(options: MediaImageLimitOptions = {}) {
    if (options.disableImageDownscale === true || this._disableImageDownscale) {
      return 0;
    }

    const configured = Number(options.mediaMaxImagePixels);
    if (Number.isFinite(configured) && configured > 0) {
      return Math.max(65536, Math.min(33554432, Math.trunc(configured)));
    }

    return this._mediaMaxImagePixels;
  }

  _resolveMediaImageMaxEdge(options: MediaImageLimitOptions = {}) {
    if (options.disableImageDownscale === true || this._disableImageDownscale) {
      return 0;
    }

    const configured = Number(options.mediaMaxImageEdge);
    if (Number.isFinite(configured) && configured > 0) {
      return Math.max(64, Math.min(16384, Math.trunc(configured)));
    }

    return this._mediaMaxImageEdge;
  }

  async _fetchWithTimeout(url: string, init: RequestInit = {}, timeoutMs = 0) {
    const resolvedTimeout = Number(timeoutMs);
    if (!Number.isFinite(resolvedTimeout) || resolvedTimeout <= 0) {
      return fetch(url, init);
    }

    if (typeof AbortController !== 'function') {
      return Promise.race([
        fetch(url, init),
        new Promise<never>((_, reject) => {
          globalThis.setTimeout(
            () => reject(new Error(`fetch timeout (${resolvedTimeout}ms)`)),
            resolvedTimeout,
          );
        }),
      ]);
    }

    const timeoutController = new AbortController();
    const externalSignal = init.signal;
    let didTimeout = false;
    let timeoutHandle = null;
    const onExternalAbort = () => {
      try {
        timeoutController.abort();
      } catch (_) {
        // ignore abort races
      }
    };

    if (externalSignal && typeof externalSignal.addEventListener === 'function') {
      externalSignal.addEventListener('abort', onExternalAbort, { once: true });
      if (externalSignal.aborted) {
        onExternalAbort();
      }
    }

    timeoutHandle = globalThis.setTimeout(() => {
      didTimeout = true;
      try {
        timeoutController.abort();
      } catch (_) {
        // ignore abort races
      }
    }, resolvedTimeout);

    try {
      return await fetch(url, {
        ...init,
        signal: timeoutController.signal,
      });
    } catch (error) {
      if (didTimeout) {
        throw new Error(`fetch timeout (${resolvedTimeout}ms)`);
      }
      throw error;
    } finally {
      if (timeoutHandle != null) {
        globalThis.clearTimeout(timeoutHandle);
      }
      if (externalSignal && typeof externalSignal.removeEventListener === 'function') {
        externalSignal.removeEventListener('abort', onExternalAbort);
      }
    }
  }

  _unlinkDecisionFile(path: string | null) {
    if (!this._core || typeof path !== 'string' || path.length === 0) {
      return;
    }
    try {
      this._core.FS.unlink(path);
    } catch (_) {
      // ignore best-effort cleanup failures
    }
  }

  _deleteFsFile(path: string | null) {
    if (!this._core || typeof path !== 'string' || path.length === 0) {
      return false;
    }

    return unlinkFsFile(this._core.FS, path);
  }

  _releaseModelFiles() {
    const paths = Array.isArray(this._modelPaths) && this._modelPaths.length > 0
      ? [...this._modelPaths]
      : (this._modelPath ? [this._modelPath] : []);

    let removed = 0;
    for (const path of paths) {
      if (this._deleteFsFile(path)) {
        removed += 1;
      }
    }

    this._modelPaths = [];
    return removed;
  }

  // A load replaces the current model. Free it, its projector and their FS files
  // before downloading the next model so both never occupy the wasm heap at once.
  // The core refuses while a generation or speech synthesis is still active,
  // which leaves the current model untouched.
  _releaseLoadedModel(core: LlamaCoreModule) {
    const projectorPath = this._mmProjPath;
    const hasModelFiles = Array.isArray(this._modelPaths) && this._modelPaths.length > 0;
    if (this._modelBytes <= 0 && !projectorPath && !hasModelFiles) {
      return false;
    }

    const rc = Number(core.ccall('llamadart_webgpu_free_model', 'number', [], []));
    if (rc !== 0) {
      throw new Error(this._coreErrorMessage('Failed to release the loaded model', rc));
    }

    this._loraAdaptersLoaded = false;
    this._clearStagedMediaFiles();
    this._deleteFsFile(projectorPath);
    this._releaseModelFiles();
    this._modelPath = null;
    this._modelBytes = 0;
    this._modelSource = 'network';
    this._mmProjPath = null;
    this._mmSupportsVision = false;
    this._mmSupportsAudio = false;
    this._runtimeNotes.push('previous_model_released');
    return true;
  }

  async _getCachedModelResponse(url: string, options: CachedModelResponseOptions = {}) {
    let useCache = options.useCache !== false;
    const forceRefresh = options.force === true;
    const requireReadableStream = options.requireReadableStream === true;
    const requestHeaders =
      options.requestHeaders && typeof options.requestHeaders === 'object'
        ? options.requestHeaders
        : null;

    if (requestHeaders && Object.keys(requestHeaders).length > 0) {
      useCache = false;
    }

    const fetchOptions = {
      ...(options.signal ? { signal: options.signal } : {}),
      ...(requestHeaders ? { headers: requestHeaders } : {}),
    };
    const fetchTimeoutMs = this._resolveFetchTimeoutMs(options, 180000);
    this._modelSource = 'network';
    this._modelCacheState = useCache ? 'unavailable' : 'disabled';
    this._modelCacheName = this._resolveCacheName(options);

    if (!useCache) {
      const response = await this._fetchWithTimeout(url, {
        cache: 'no-store',
        ...fetchOptions,
      }, fetchTimeoutMs);
      this._modelCacheState = 'disabled';
      if (requireReadableStream) {
        this._runtimeNotes.push(
          hasReadableResponseStream(response)
            ? 'model_network_stream'
            : 'model_network_no_stream',
        );
      }
      return response;
    }

    if (!globalThis.caches || typeof globalThis.caches.open !== 'function') {
      this._modelCacheState = 'unavailable';
      const response = await this._fetchWithTimeout(url, fetchOptions, fetchTimeoutMs);
      if (requireReadableStream) {
        this._runtimeNotes.push(
          hasReadableResponseStream(response)
            ? 'model_network_stream'
            : 'model_network_no_stream',
        );
      }
      return response;
    }

    const cacheKey = normalizeAbsoluteUrl(url);

    try {
      const cache = await globalThis.caches.open(this._modelCacheName);
      const cached = forceRefresh ? null : await cache.match(cacheKey);
      if (cached) {
        const cacheHasStream = hasReadableResponseStream(cached);
        if (!requireReadableStream || cacheHasStream) {
          this._modelSource = 'cache';
          this._modelCacheState = 'hit';
          this._runtimeNotes.push('model_cache_hit');
          if (requireReadableStream) {
            this._runtimeNotes.push('model_cache_stream');
          }
          return cached;
        }

        this._runtimeNotes.push('model_cache_hit_no_stream');
        this._modelSource = 'network';
        this._modelCacheState = 'refresh';
        const refreshed = await this._fetchWithTimeout(url, {
          cache: 'no-store',
          ...fetchOptions,
        }, fetchTimeoutMs);

        if (refreshed.ok) {
          try {
            await cache.put(cacheKey, refreshed.clone());
            this._modelCacheState = 'stored';
            this._runtimeNotes.push('model_cache_stored');
          } catch (_) {
            this._modelCacheState = 'store_failed';
            this._runtimeNotes.push('model_cache_store_failed');
          }
        }

        this._runtimeNotes.push(
          hasReadableResponseStream(refreshed)
            ? 'model_network_stream'
            : 'model_network_no_stream',
        );
        return refreshed;
      }

      this._modelCacheState = forceRefresh ? 'refresh' : 'miss';
      const response = await this._fetchWithTimeout(url, fetchOptions, fetchTimeoutMs);

      if (response.ok) {
        try {
          await cache.put(cacheKey, response.clone());
          this._modelCacheState = 'stored';
          this._runtimeNotes.push('model_cache_stored');
        } catch (_) {
          this._modelCacheState = 'store_failed';
          this._runtimeNotes.push('model_cache_store_failed');
        }
      }

      if (requireReadableStream) {
        this._runtimeNotes.push(
          hasReadableResponseStream(response)
            ? 'model_network_stream'
            : 'model_network_no_stream',
        );
      }

      return response;
    } catch (_) {
      this._modelCacheState = 'error';
      this._runtimeNotes.push('model_cache_error');
      const response = await this._fetchWithTimeout(url, fetchOptions, fetchTimeoutMs);
      if (requireReadableStream) {
        this._runtimeNotes.push(
          hasReadableResponseStream(response)
            ? 'model_network_stream'
            : 'model_network_no_stream',
        );
      }
      return response;
    }
  }

  async prefetchModelToCache(url: string | string[], options: LoadModelOptions = {}) {
    const useCache = options.useCache !== false;
    this._modelSource = 'network';
    this._modelCacheState = useCache ? 'unavailable' : 'disabled';
    this._modelCacheName = this._resolveCacheName(options);

    const progressCallback = typeof options.progressCallback === 'function'
      ? options.progressCallback
      : null;

    const modelUrls = expandModelShardUrls(url);
    if (modelUrls.length === 0) {
      throw new Error('Model URL is empty.');
    }

    if (modelUrls.length > 1) {
      this._runtimeNotes.push(`model_split_cache_prefetch:${modelUrls.length}`);
    }

    const shardLoaded = new Array(modelUrls.length).fill(0);
    const shardTotals = new Array(modelUrls.length).fill(0);
    const emitAggregateProgress = () => {
      if (!progressCallback) {
        return;
      }

      const loaded = sumProgressValues(shardLoaded);
      const total = sumProgressValues(shardTotals);
      progressCallback({
        loaded,
        total: total > 0 ? total : loaded,
      });
    };

    const controller = this._beginTransferAbortController();

    try {
      const fetchOptions: RequestInit = {
        cache: 'no-store',
        ...(controller?.signal ? { signal: controller.signal } : {}),
      };
      const fetchTimeoutMs = this._resolveFetchTimeoutMs(options, 180000);
      const chunkTimeoutMs = this._resolveStreamChunkTimeoutMs(options, 90000);

      const cache = (useCache && globalThis.caches && typeof globalThis.caches.open === 'function')
        ? await globalThis.caches.open(this._modelCacheName)
        : null;

      if (!useCache || !cache) {
        this._modelCacheState = useCache ? 'unavailable' : 'disabled';
      }

      for (let shardIndex = 0; shardIndex < modelUrls.length; shardIndex += 1) {
        const shardUrl = modelUrls[shardIndex];
        const cacheKey = normalizeAbsoluteUrl(shardUrl);
        let response;
        let headerTotal = 0;

        if (cache) {
          const cached = options.force === true ? null : await cache.match(cacheKey);
          if (cached) {
            this._modelSource = 'cache';
            this._modelCacheState = 'hit';
            this._runtimeNotes.push('model_cache_hit');

            headerTotal = Number(cached.headers.get('content-length')) || 0;
            const resolved = headerTotal > 0 ? headerTotal : 1;
            shardLoaded[shardIndex] = resolved;
            shardTotals[shardIndex] = resolved;
            emitAggregateProgress();
            continue;
          }

          this._modelSource = 'network';
          this._modelCacheState = options.force === true ? 'refresh' : 'miss';
        }

        response = await this._fetchWithTimeout(shardUrl, fetchOptions, fetchTimeoutMs);
        if (!response.ok) {
          throw new Error(
            `Failed to prefetch model shard: ${response.status} ${response.statusText}`,
          );
        }

        headerTotal = Number(response.headers.get('content-length')) || 0;
        if (headerTotal > 0) {
          shardTotals[shardIndex] = headerTotal;
        }

        const putPromise = cache
          ? cache.put(cacheKey, response.clone())
            .then(() => ({ ok: true, error: null }))
            .catch((error) => ({ ok: false, error }))
          : null;

        await drainResponseWithProgress(
          response,
          progressCallback
              ? (progress) => {
                  const loaded = Number(progress?.loaded) || 0;
                  const total = Number(progress?.total) || 0;
                  shardLoaded[shardIndex] = loaded;
                  if (total > 0) {
                    shardTotals[shardIndex] = total;
                  }
                  emitAggregateProgress();
                }
              : null,
          { chunkTimeoutMs },
        );

        const finalLoaded = shardLoaded[shardIndex] > 0
          ? shardLoaded[shardIndex]
          : (shardTotals[shardIndex] > 0 ? shardTotals[shardIndex] : headerTotal);
        shardLoaded[shardIndex] = finalLoaded;
        if (finalLoaded > 0 && shardTotals[shardIndex] < finalLoaded) {
          shardTotals[shardIndex] = finalLoaded;
        }
        emitAggregateProgress();

        if (putPromise) {
          const putResult = await putPromise;
          if (putResult.ok) {
            this._modelCacheState = 'stored';
            this._runtimeNotes.push('model_cache_stored');
          } else {
            const putErrorText = String(putResult.error || '').toLowerCase();
            if (putErrorText.includes('abort') || putErrorText.includes('cancel')) {
              throw putResult.error || new Error('Model cache prefetch was aborted.');
            }

            this._modelCacheState = 'store_failed';
            this._runtimeNotes.push('model_cache_store_failed');
            throw new Error('Failed to store prefetched model in browser cache.');
          }
        }
      }

      if (modelUrls.length > 1) {
        this._runtimeNotes.push('model_split_cache_prefetched');
      }
      return 1;
    } catch (error) {
      this._modelCacheState = 'error';

      const text = String(error || '').toLowerCase();
      if (text.includes('abort')) {
        this._runtimeNotes.push('model_cache_prefetch_aborted');
      } else if (text.includes('quota') || text.includes('storage')) {
        this._runtimeNotes.push('model_cache_quota_exceeded');
      } else {
        this._runtimeNotes.push('model_cache_error');
      }

      throw error;
    } finally {
      this._clearTransferAbortController(controller);
    }
  }

  async evictModelFromCache(url: string | string[], options: Record<string, unknown> = {}) {
    this._modelCacheName = this._resolveCacheName(options);

    const modelUrls = expandModelShardUrls(url);
    if (modelUrls.length === 0) {
      return false;
    }

    if (!globalThis.caches || typeof globalThis.caches.open !== 'function') {
      this._modelCacheState = 'unavailable';
      return false;
    }

    try {
      const cache = await globalThis.caches.open(this._modelCacheName);
      let removedCount = 0;
      for (const modelUrl of modelUrls) {
        const cacheKey = normalizeAbsoluteUrl(modelUrl);
        const removed = await cache.delete(cacheKey);
        if (removed) {
          removedCount += 1;
        }
      }

      const removedAny = removedCount > 0;
      this._modelCacheState = removedAny ? 'evicted' : 'miss';
      if (removedAny) {
        this._runtimeNotes.push('model_cache_evicted');
      }
      if (modelUrls.length > 1) {
        this._runtimeNotes.push(`model_split_cache_evicted:${removedCount}/${modelUrls.length}`);
      }
      return removedAny;
    } catch (_) {
      this._modelCacheState = 'error';
      this._runtimeNotes.push('model_cache_error');
      return false;
    }
  }

  async _ensureCore() {
    if (this._core) {
      this._applyCoreLogLevel();
      return this._core;
    }

    const candidates = [];
    if (this._config.coreModuleFactory) {
      candidates.push({
        variant: 'custom',
        factoryPromise: Promise.resolve(this._config.coreModuleFactory),
        wasmUrl: this._config.wasmUrl,
      });
    } else {
      if (
        this._preferMemory64
        && typeof this._config.coreModuleUrlMem64 === 'string'
        && this._config.coreModuleUrlMem64.length > 0
      ) {
        candidates.push({
          variant: 'wasm64',
          factoryPromise: importCoreFactory(this._config.coreModuleUrlMem64),
          wasmUrl: this._config.wasmUrlMem64 || this._config.wasmUrl,
        });
      }

      candidates.push({
        variant: 'wasm32',
        factoryPromise: importCoreFactory(this._config.coreModuleUrl ?? './llama_webgpu_core.js'),
        wasmUrl: this._config.wasmUrl,
      });
    }

    let lastError = null;
    for (const candidate of candidates) {
      if (candidate.variant === 'wasm64') {
        this._runtimeNotes.push('core_mem64_attempt');
      }

      try {
        const moduleFactory = await candidate.factoryPromise;
        const initTimeoutMs = this._resolveCoreInitTimeoutMs();
        this._core = await Promise.race([
          moduleFactory({
          locateFile: (path: string, prefix: string) => {
            if (path.endsWith('.wasm') && candidate.wasmUrl) {
              return candidate.wasmUrl;
            }
            return `${prefix}${path}`;
          },
          print: (msg: unknown) => {
            this._emitLogger('log', msg);
          },
          printErr: (msg: unknown) => {
            const text = String(msg ?? '');
            this._lastCoreErrorText = text;
            const trimmed = text.trim();
            if (trimmed.length > 0) {
              const loweredTrimmed = trimmed.toLowerCase();
              const isGenericAbort =
                loweredTrimmed === 'aborted(native code called abort())'
                || loweredTrimmed === 'native code called abort()'
                || loweredTrimmed === 'aborted';
              if (!isGenericAbort) {
                this._lastCoreErrorHint = trimmed;
              }
            }
            const classification = this._classifyCoreErrorLine(text);
            if (classification === 'ignore') {
              return;
            }

            if (classification === 'warmup') {
              if (this._logLevel >= 2) {
                this._suppressedWarmupWarningCount += 1;
                if (!this._didReportWarmupWarningSuppression) {
                  this._didReportWarmupWarningSuppression = true;
                  if (this._logLevel <= 2) {
                    this._emitLogger(
                      'log',
                      'info: suppressing verbose warmup op logs; set bridge/runtime log level to Debug to inspect all warmup details.',
                    );
                  }
                }
                this._pushRuntimeNote('warmup_warning_suppressed');
                return;
              }

              this._emitLogger('log', trimmed.length > 0 ? trimmed : text);
              return;
            }

            if (classification === 'warn') {
              this._emitLogger('warn', trimmed.length > 0 ? trimmed : text);
              return;
            }

            if (classification === 'info') {
              this._emitLogger('log', trimmed.length > 0 ? trimmed : text);
              return;
            }

            this._emitLogger('error', trimmed.length > 0 ? trimmed : text);
          },
          onAbort: (reason: unknown) => {
            const text = String(reason ?? '').trim();
            if (text.length > 0) {
              const token = text
                .slice(0, 120)
                .replace(/[\s;=]+/g, '_')
                .replace(/[^a-zA-Z0-9._:-]/g, '');
              if (token.length > 0) {
                this._runtimeNotes.push(`core_abort:${token}`);
              }
              this._emitLogger('error', `core abort: ${text}`);
            } else {
              this._runtimeNotes.push('core_abort');
              this._emitLogger('error', 'core abort');
            }
          },
          }),
          new Promise((_, reject) => {
            globalThis.setTimeout(() => {
              reject(new Error(`Bridge core init timeout (${initTimeoutMs}ms)`));
            }, initTimeoutMs);
          }),
        ]) as LlamaCoreModule;

        this._coreVariant = candidate.variant === 'wasm64' ? 'wasm64' : 'wasm32';
        if (candidate.variant === 'wasm64') {
          this._runtimeNotes.push('core_mem64_active');
        } else if (candidate.variant === 'wasm32') {
          this._runtimeNotes.push('core_wasm32_active');
        }

        break;
      } catch (error) {
        lastError = error;
        if (candidate.variant === 'wasm64') {
          this._runtimeNotes.push('core_mem64_unavailable');
          continue;
        }
        throw error;
      }
    }

    if (!this._core) {
      throw lastError || new Error('Failed to initialize bridge core module');
    }

    this._applyCoreLogLevel();

    return this._core;
  }

  async _probeBackends() {
    try {
      const core = await this._ensureCore();
      const probeResult = Number(
        await core.ccall('llamadart_webgpu_probe', 'number', [], [], { async: true }),
      );
      const json = core.ccall('llamadart_webgpu_backends_json', 'string', [], []);

      let parsed = [];
      try {
        parsed = JSON.parse(json || '[]');
      } catch (_) {
        parsed = [];
      }

      this._backendLabels = Array.isArray(parsed)
        ? parsed.map((v) => String(v))
        : [];
      this._gpuActive = probeResult === 1;
    } catch (_err) {
      this._backendLabels = [];
      this._gpuActive = false;
    }

    return this._gpuActive;
  }

  async loadModelFromUrl(url: ModelSource, options: RuntimeLoadModelOptions = {}) {
    const operationSignal = options?.signal || null;
    const abortMessage = 'Model load was cancelled.';
    throwIfAborted(operationSignal, abortMessage);
    this._abortRequested = false;
    this._runtimeNotes = [];
    this._suppressedWarmupWarningCount = 0;
    this._didReportWarmupWarningSuppression = false;
    await this._probeBackends();
    throwIfAborted(operationSignal, abortMessage);

    const core = await this._ensureCore();
    throwIfAborted(operationSignal, abortMessage);

    const modelUrls = expandModelShardUrls(url);
    if (modelUrls.length === 0) {
      throw new Error('Model URL is empty.');
    }
    // Release before any per-load option below changes, so a refused release
    // leaves the current model and its configuration intact.
    this._releaseLoadedModel(core);
    this._mmProjSourceUrl = null;

    const configuredPoolHint = Number(this._threadPoolSizeHint);
    this._syncThreadPoolSizeHintFromCore();
    const coreSupportsPthreads = this._coreSupportsPthreads();
    this._pushRuntimeNote(`core_pthreads:${coreSupportsPthreads ? 1 : 0}`);
    if (
      !coreSupportsPthreads
      && Number.isFinite(configuredPoolHint)
      && configuredPoolHint > 1
    ) {
      this._runtimeNotes.push('threads_capped_no_pthread');
    }

    this._nCtx = Number(options.nCtx) > 0 ? Number(options.nCtx) : this._nCtx;

    const requestedThreads = Number(options.nThreads);
    if (Number.isFinite(requestedThreads) && requestedThreads > 0) {
      this._threads = this._capThreadsToPool(requestedThreads);
    } else {
      this._threads = this._capThreadsToPool(this._resolveAutoThreadCount());
    }

    const requestedThreadsBatch = Number(options.nThreadsBatch);
    if (Number.isFinite(requestedThreadsBatch) && requestedThreadsBatch > 0) {
      this._threadsBatch = this._capThreadsToPool(
        requestedThreadsBatch,
        { noteTag: 'threads_batch_capped_pool' },
      );
    } else {
      this._threadsBatch = this._threads;
    }

    const requestedGpuLayers = Number(options.nGpuLayers);
    if (Number.isFinite(requestedGpuLayers)) {
      this._nGpuLayers = Math.trunc(requestedGpuLayers);
    }

    const isCpuModelMode = this._nGpuLayers === 0;

    const requestedBatch = Number(options.nBatch);
    this._nBatch = Number.isFinite(requestedBatch) && requestedBatch > 0
      ? Math.max(32, Math.trunc(requestedBatch))
      : (isCpuModelMode ? Math.min(this._nCtx, 512) : 0);

    const requestedUbatch = Number(options.nUbatch);
    this._nUbatch = Number.isFinite(requestedUbatch) && requestedUbatch > 0
      ? Math.max(32, Math.trunc(requestedUbatch))
      : (isCpuModelMode ? Math.min(this._nBatch || 256, 256) : 0);

    if (this._nBatch > 0 && this._nBatch > this._nCtx) {
      this._nBatch = this._nCtx;
    }
    if (this._nUbatch > 0 && this._nBatch > 0 && this._nUbatch > this._nBatch) {
      this._nUbatch = this._nBatch;
    }

    this._resolveNativeLoadOptions(options);

    if (Number.isFinite(this._threadPoolSizeHint) && this._threadPoolSizeHint! > 0) {
      this._pushRuntimeNote(`thread_pool_size:${this._threadPoolSizeHint}`);
    }

    if (!isCrossOriginIsolatedRuntime()) {
      this._runtimeNotes.push('threads_capped_no_coi');
      if (this._threads > 1) {
        this._threads = 1;
      }
      if (this._threadsBatch > 1) {
        this._threadsBatch = 1;
      }
    }

    this._pushRuntimeNote(`threads_batch:${this._threadsBatch}`);
    if (this._nBatch > 0) {
      this._pushRuntimeNote(`n_batch:${this._nBatch}`);
    }
    if (this._nUbatch > 0) {
      this._pushRuntimeNote(`n_ubatch:${this._nUbatch}`);
    }
    if (this._nSeqMax > 0) {
      this._pushRuntimeNote(`n_seq_max:${this._nSeqMax}`);
    }
    if (isCpuModelMode && !Number.isFinite(requestedBatch) && !Number.isFinite(requestedUbatch)) {
      this._runtimeNotes.push('cpu_batch_tuned_default');
    }

    if (this._isSafari && this._nGpuLayers > 0) {
      const requestedSafariMaxLayers = Number(options.safariMaxGpuLayers);
      const safariMaxGpuLayers = Number.isFinite(requestedSafariMaxLayers)
        ? Math.max(1, Math.trunc(requestedSafariMaxLayers))
        : 1;

      if (this._nGpuLayers > safariMaxGpuLayers) {
        this._nGpuLayers = safariMaxGpuLayers;
        this._runtimeNotes.push(`safari_gpu_layers_capped:${safariMaxGpuLayers}`);
      }
    }

    this._loadedModelUrl = cloneModelSource(url);
    if (modelUrls.length > 1) {
      this._runtimeNotes.push(`model_split_detected:${modelUrls.length}`);
    }

    let loadedViaRemoteFetch = false;
    let remoteFetchReloadUrl = null;
    const remoteFetchReloadChunkBytes = this._resolveRemoteFetchChunkBytes(options);
    if (modelUrls.length === 1) {
      const remoteResult = await this._tryLoadModelFromRemoteFetchBackend(
        core,
        modelUrls[0],
        options,
      );
      loadedViaRemoteFetch = remoteResult.loaded === true;
      if (loadedViaRemoteFetch) {
        remoteFetchReloadUrl = modelUrls[0];
      }
    } else {
      this._runtimeNotes.push('model_fetch_backend_skipped_split');
    }

    if (!loadedViaRemoteFetch) {
      ensureFsDirectory(core.FS, '/models');

      const progressCallback = typeof options.progressCallback === 'function'
        ? options.progressCallback
        : null;
      const shardLoaded = new Array(modelUrls.length).fill(0);
      const shardTotals = new Array(modelUrls.length).fill(0);
      const emitAggregateProgress = () => {
        if (!progressCallback) {
          return;
        }

        const loaded = sumProgressValues(shardLoaded);
        const total = sumProgressValues(shardTotals);
        progressCallback({
          loaded,
          total: total > 0 ? total : loaded,
        });
      };

      const modelPaths = [];
      let totalModelBytes = 0;
      const maxStreamResumeRetries = Number.isFinite(options.streamResumeRetries)
        ? Math.max(0, Math.trunc(options.streamResumeRetries as number))
        : 8;
      const streamChunkTimeoutMs = this._resolveStreamChunkTimeoutMs(
        options,
        90000,
      );

      try {
        for (let shardIndex = 0; shardIndex < modelUrls.length; shardIndex += 1) {
          const shardUrl = modelUrls[shardIndex];
          const fileName = basenameFromUrl(shardUrl);
          const modelPath = `/models/${fileName}`;
          modelPaths.push(modelPath);

          let shardBytes = 0;
          let resumeOffset = 0;
          let resumeAttempt = 0;
          let activeShardUrl = shardUrl;
          let knownTotalBytes = 0;

          while (true) {
            throwIfAborted(operationSignal, abortMessage);
            const requestHeaders = resumeOffset > 0
              ? { Range: `bytes=${resumeOffset}-` }
              : null;

            const response = await this._getCachedModelResponse(activeShardUrl, {
              ...options,
              requireReadableStream: true,
              useCache: requestHeaders ? false : options.useCache,
              force: requestHeaders ? true : options.force,
              requestHeaders,
            });
            if (!response.ok) {
              throw new Error(
                `Failed to fetch model shard: ${response.status} ${response.statusText}`,
              );
            }

            const responseUrl =
              typeof response.url === 'string' && response.url.length > 0
                ? response.url
                : activeShardUrl;
            if (responseUrl !== activeShardUrl) {
              activeShardUrl = responseUrl;
              if (resumeOffset > 0) {
                this._runtimeNotes.push('model_stream_resume_redirect');
              }
            }

            if (resumeOffset > 0 && response.status !== 206) {
              this._runtimeNotes.push(`model_stream_resume_status:${response.status}`);
              throw new Error(
                `Range resume not honored for model shard: ${response.status} ${response.statusText}`,
              );
            }

            if (!hasReadableResponseStream(response)) {
              this._runtimeNotes.push('model_response_nostream');
              const declaredBytes = inferResponseTotalBytes(response, 0);
              const declaredText = declaredBytes > 0 ? `${declaredBytes} bytes` : 'unknown size';
              throw new Error(
                `Model response did not expose a readable stream (${declaredText}).`,
              );
            }
            this._runtimeNotes.push('model_response_stream');

            const responseTotal = inferResponseTotalBytes(response, knownTotalBytes);
            if (responseTotal > 0) {
              knownTotalBytes = responseTotal;
              shardTotals[shardIndex] = responseTotal;
            }

            try {
              shardBytes = await writeResponseToFsFileWithProgress(
                response,
                core.FS,
                modelPath,
                progressCallback
                  ? (progress) => {
                      const loaded = Number(progress?.loaded) || 0;
                      const total = Number(progress?.total) || 0;
                      shardLoaded[shardIndex] = loaded;
                      if (total > 0) {
                        shardTotals[shardIndex] = total;
                      }
                      emitAggregateProgress();
                    }
                  : null,
                {
                  useBigIntPosition: this._coreVariant === 'wasm64',
                  startOffset: resumeOffset,
                  preservePartialOnError: true,
                  totalBytes: knownTotalBytes,
                  chunkTimeoutMs: streamChunkTimeoutMs,
                  signal: operationSignal,
                  abortMessage,
                },
              );
              break;
            } catch (error) {
              const text = String(error || '').toLowerCase();
              const loadedBytes = Number(
                (error as { llamadartLoadedBytes?: unknown } | null | undefined)?.llamadartLoadedBytes,
              );
              if (Number.isFinite(loadedBytes) && loadedBytes >= 0) {
                const normalizedLoaded = Math.trunc(loadedBytes);
                this._runtimeNotes.push(`model_fs_write_loaded:${normalizedLoaded}`);
                if (normalizedLoaded > resumeOffset) {
                  resumeOffset = normalizedLoaded;
                  shardLoaded[shardIndex] = normalizedLoaded;
                  if (knownTotalBytes > 0 && shardTotals[shardIndex] < knownTotalBytes) {
                    shardTotals[shardIndex] = knownTotalBytes;
                  }
                  emitAggregateProgress();
                }
              }

              const shouldRetryResume =
                isRetryableStreamNetworkError(error)
                && resumeOffset > 0
                && resumeAttempt < maxStreamResumeRetries;
              if (shouldRetryResume) {
                resumeAttempt += 1;
                this._runtimeNotes.push(`model_stream_resume_retry:${resumeAttempt}`);
                this._runtimeNotes.push(`model_stream_resume_offset:${resumeOffset}`);
                continue;
              }

              if (text.includes('bigint')) {
                this._runtimeNotes.push('model_fs_write_bigint_error');
              } else if (text.includes('abort')) {
                this._runtimeNotes.push('model_fs_write_abort');
              } else if (text.includes('array buffer allocation failed')) {
                this._runtimeNotes.push('model_fs_write_arraybuffer_oom');
              } else if (isRetryableStreamNetworkError(error)) {
                this._runtimeNotes.push('model_fs_write_network_error');
                this._runtimeNotes.push('model_fs_write_failed');
              } else {
                this._runtimeNotes.push('model_fs_write_failed');
              }
              throw error;
            }
          }

          shardLoaded[shardIndex] = shardBytes;
          if (shardTotals[shardIndex] < shardBytes) {
            shardTotals[shardIndex] = shardBytes;
          }
          totalModelBytes += shardBytes;
          emitAggregateProgress();
        }

        this._modelPaths = modelPaths;
        this._modelPath = modelPaths[0] || null;
        this._modelBytes = totalModelBytes;

        let rc = 0;
        let nativeLoadStarted = false;
        let nativeAbortCleanupDone = false;
        try {
          throwIfAborted(operationSignal, abortMessage);
          nativeLoadStarted = true;
          rc = Number(
            await core.ccall(
              'llamadart_webgpu_load_model',
              'number',
              [
                'string',
                'number',
                'number',
                'number',
                'number',
                'number',
                'number',
                ...this._nativeLoadOptionTypes(),
              ],
              [
                this._modelPath,
                this._nCtx,
                this._threads,
                this._threadsBatch,
                this._nBatch,
                this._nUbatch,
                this._nGpuLayers,
                ...this._nativeLoadOptionValues(),
              ],
              { async: true },
            ),
          );
          if (operationSignal?.aborted) {
            try {
              core.ccall('llamadart_webgpu_shutdown', null, [], []);
              nativeAbortCleanupDone = true;
            } catch (_) {
              // best-effort cleanup after an unpreemptible native load
            }
            throw createAbortError(abortMessage);
          }
        } catch (error) {
          if (operationSignal?.aborted && nativeLoadStarted && !nativeAbortCleanupDone) {
            try {
              core.ccall('llamadart_webgpu_shutdown', null, [], []);
              nativeAbortCleanupDone = true;
            } catch (_) {
              // best-effort cleanup after an unpreemptible native load
            }
            error = createAbortError(abortMessage);
          }
          const text = String(error || '').toLowerCase();
          if (text.includes('bigint')) {
            this._runtimeNotes.push('model_load_ccall_bigint_error');
          } else if (text.includes('abort')) {
            this._runtimeNotes.push('model_load_ccall_abort');
          } else {
            this._runtimeNotes.push('model_load_ccall_failed');
          }
          throw error;
        }

        if (rc !== 0) {
          throw new Error(this._coreErrorMessage('Failed to load GGUF model', rc));
        }

        if (modelUrls.length > 1) {
          this._runtimeNotes.push(`model_split_loaded:${modelUrls.length}`);
        }
      } catch (error) {
        this._modelBytes = 0;
        this._modelPath = null;
        this._modelPaths = [];
        if (operationSignal?.aborted) {
          this._loadedModelUrl = null;
        }
        for (const modelPath of modelPaths) {
          this._deleteFsFile(modelPath);
        }
        throw error;
      }
    }

    const shouldProbeSafariGpu = this._isSafari
      && this._nGpuLayers > 0
      && options.safariGpuProbe !== false;

    if (shouldProbeSafariGpu) {
      if (loadedViaRemoteFetch) {
        this._runtimeNotes.push('safari_probe_on_fetch_backend');
      }

      const defaultProbePrompts = [
        'user: hi\nassistant:',
        'user: say hello in one short sentence\nassistant:',
      ];

      const probePrompts = Array.isArray(options.safariProbePrompts)
        ? options.safariProbePrompts
          .map((v) => String(v || '').trim())
          .filter((v) => v.length > 0)
        : (typeof options.safariProbePrompt === 'string' && options.safariProbePrompt.trim().length > 0
            ? [options.safariProbePrompt.trim()]
            : defaultProbePrompts);

      const probeTokensRaw = Number(options.safariProbeTokens);
      const probeTokens = Number.isFinite(probeTokensRaw) && probeTokensRaw > 0
        ? Math.min(Math.trunc(probeTokensRaw), 96)
        : 48;

      const runProbe = async (probePrompt: string, probeSeed: number) => {
        try {
          const probeOutput = await this.createCompletion(probePrompt, {
            nPredict: probeTokens,
            temp: 0,
            topK: 1,
            topP: 1,
            penalty: 1,
            seed: probeSeed,
          });
          return !looksLikeCorruptedGeneration(probeOutput);
        } catch (_) {
          return false;
        }
      };

      let initialProbePassed = true;
      for (let i = 0; i < probePrompts.length; i += 1) {
        const ok = await runProbe(probePrompts[i], i + 1);
        if (!ok) {
          initialProbePassed = false;
          break;
        }
      }

      if (!initialProbePassed) {
        this._runtimeNotes.push('safari_gpu_probe_failed');

        const retryCandidates = [];
        if (this._nGpuLayers > 1) {
          retryCandidates.push(1);
        }
        retryCandidates.push(0);

        let stabilized = false;
        for (const candidateLayers of retryCandidates) {
          try {
            core.ccall('llamadart_webgpu_shutdown', null, [], []);
          } catch (_) {
            // ignore shutdown retries
          }

          let retryRc = 0;
          if (loadedViaRemoteFetch) {
            const reloadUrl = remoteFetchReloadUrl || modelUrls[0] || null;
            if (!reloadUrl) {
              continue;
            }

            this._runtimeNotes.push(`safari_probe_remote_fetch_retry:${candidateLayers}`);
            retryRc = Number(
              await core.ccall(
                'llamadart_webgpu_load_model_from_url',
                'number',
                [
                  'string',
                  'number',
                  'number',
                  'number',
                  'number',
                  'number',
                  'number',
                  'number',
                  ...this._nativeLoadOptionTypes(),
                ],
                [
                  reloadUrl,
                  this._nCtx,
                  this._threads,
                  this._threadsBatch,
                  this._nBatch,
                  this._nUbatch,
                  candidateLayers,
                  remoteFetchReloadChunkBytes,
                  ...this._nativeLoadOptionValues(),
                ],
                { async: true },
              ),
            );
          } else {
            retryRc = Number(
              await core.ccall(
                'llamadart_webgpu_load_model',
                'number',
                [
                  'string',
                  'number',
                  'number',
                  'number',
                  'number',
                  'number',
                  'number',
                  ...this._nativeLoadOptionTypes(),
                ],
                [
                  this._modelPath,
                  this._nCtx,
                  this._threads,
                  this._threadsBatch,
                  this._nBatch,
                  this._nUbatch,
                  candidateLayers,
                  ...this._nativeLoadOptionValues(),
                ],
                { async: true },
              ),
            );
          }

          if (retryRc !== 0) {
            continue;
          }

          this._nGpuLayers = candidateLayers;
          if (candidateLayers === 0) {
            this._runtimeNotes.push('safari_fallback_cpu');
            stabilized = true;
            break;
          }

          let retryProbePassed = true;
          for (let i = 0; i < probePrompts.length; i += 1) {
            const ok = await runProbe(probePrompts[i], i + 11);
            if (!ok) {
              retryProbePassed = false;
              break;
            }
          }

          if (retryProbePassed) {
            this._runtimeNotes.push(`safari_gpu_layers_capped:${candidateLayers}`);
            stabilized = true;
            break;
          }
        }

        if (!stabilized) {
          throw new Error('Safari GPU probe failed and fallback attempts were unsuccessful.');
        }
      } else {
        this._runtimeNotes.push('safari_gpu_probe_passed');
      }
    }

    try {
      const effectiveNctx = Number(core.ccall('llamadart_webgpu_get_context_size', 'number', [], []));
      if (effectiveNctx > 0) {
        this._nCtx = effectiveNctx;
      }
    } catch (_) {
      // Keep requested nCtx if runtime query is unavailable.
    }

    this._mmProjPath = null;
    this._mmSupportsVision = false;
    this._mmSupportsAudio = false;
    this._mediaFileCounter = 0;
    this._stateFileCounter = 0;
    this._stagedMediaPaths = [];
    this._gpuActive = this._gpuActive && this._nGpuLayers > 0;

    if (this._releaseModelFiles() > 0) {
      this._runtimeNotes.push('model_file_released');
    }

    this._emitSuppressedWarmupWarningSummaryIfNeeded();

    return 1;
  }

  async loadMultimodalProjector(url: string) {
    if (!this._core || this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }

    if (typeof url !== 'string' || url.length === 0) {
      throw new Error('Multimodal projector URL/path is empty.');
    }

    const core = await this._ensureCore();

    ensureFsDirectory(core.FS, '/mmproj');

    const fileName = basenameFromUrl(url);
    const mmprojPath = `/mmproj/${fileName}`;
    const fetchTimeoutMs = this._resolveFetchTimeoutMs({}, 180000);
    const chunkTimeoutMs = this._resolveStreamChunkTimeoutMs({}, 90000);
    let lastError = null;

    for (let attempt = 0; attempt < 2; attempt += 1) {
      try {
        const response = await this._fetchWithTimeout(
          url,
          { cache: 'no-store' },
          fetchTimeoutMs,
        );
        if (!response.ok) {
          throw new Error(
            `Failed to fetch multimodal projector: ${response.status} ${response.statusText}`,
          );
        }

        this._mmProjPath = mmprojPath;
        await writeResponseToFsFileWithProgress(
          response,
          core.FS,
          this._mmProjPath,
          null,
          {
            useBigIntPosition: this._coreVariant === 'wasm64',
            chunkTimeoutMs,
          },
        );

        const rc = Number(
          await core.ccall(
            'llamadart_webgpu_mmproj_load',
            'number',
            ['string'],
            [this._mmProjPath],
            { async: true },
          ),
        );
        if (rc !== 0) {
          throw new Error(this._coreErrorMessage('Failed to load multimodal projector', rc));
        }

        this._mmSupportsVision = Number(
          core.ccall('llamadart_webgpu_mmproj_supports_vision', 'number', [], []),
        ) === 1;
        this._mmSupportsAudio = Number(
          core.ccall('llamadart_webgpu_mmproj_supports_audio', 'number', [], []),
        ) === 1;
        this._mmProjSourceUrl = url;
        return 1;
      } catch (error) {
        lastError = error;
        this._mmProjPath = null;
        this._mmSupportsVision = false;
        this._mmSupportsAudio = false;
        this._deleteFsFile(mmprojPath);

        const retryable = isRetryableStreamNetworkError(error);
        if (!retryable || attempt >= 1) {
          throw error;
        }

        this._runtimeNotes.push(`mmproj_load_retry:${attempt + 1}`);
      }
    }

    if (lastError) {
      throw lastError;
    }

    throw new Error('Failed to load multimodal projector');
  }

  async unloadMultimodalProjector() {
    if (!this._core) {
      this._mmProjPath = null;
      this._mmSupportsVision = false;
      this._mmSupportsAudio = false;
      this._mmProjSourceUrl = null;
      return;
    }

    const mmprojPath = this._mmProjPath;
    const rc = Number(
      this._core.ccall('llamadart_webgpu_mmproj_free', 'number', [], []),
    );
    if (rc !== 0) {
      throw new Error(this._coreErrorMessage('Failed to unload multimodal projector', rc));
    }
    this._deleteFsFile(mmprojPath);
    this._mmProjPath = null;
    this._mmSupportsVision = false;
    this._mmSupportsAudio = false;
    this._mmProjSourceUrl = null;
  }

  supportsVision() {
    return this._mmSupportsVision;
  }

  supportsAudio() {
    return this._mmSupportsAudio;
  }

  getTextToSpeechCapabilities() {
    if (!this._core) {
      return {
        apiVersion: 0,
        supported: false,
        modelType: 0,
        capabilities: 0,
        supportsLanguage: false,
        supportsSpeakerReference: false,
        sampleRate: 0,
        channels: 0,
        reason: 'WebGPU core is not initialized',
      };
    }
    const raw = this._core.ccall(
      'llamadart_webgpu_tts_info_json',
      'string',
      [],
      [],
    ) || '{}';
    try {
      return JSON.parse(raw) as TextToSpeechCapabilities;
    } catch (_) {
      return {
        apiVersion: 0,
        supported: false,
        modelType: 0,
        capabilities: 0,
        supportsLanguage: false,
        supportsSpeakerReference: false,
        sampleRate: 0,
        channels: 0,
        reason: 'WebGPU TTS capability response is invalid',
      };
    }
  }

  _ensureTextToSpeechDir() {
    try {
      this._core!.FS.mkdir('/tts');
    } catch (_) {
      // Shared directory. The following file operation reports real failures.
    }
  }

  async synthesizeSpeech(options: Partial<TextToSpeechOptions> = {}) {
    if (this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }
    if (options.signal?.aborted) {
      throw new DOMException('Text-to-speech synthesis was cancelled.', 'AbortError');
    }
    const capabilities = this.getTextToSpeechCapabilities();
    if (capabilities.supported !== true) {
      throw new Error(
        capabilities.reason || 'Loaded model/projector does not support text-to-speech.',
      );
    }

    const text = typeof options.text === 'string' ? options.text : '';
    if (text.length === 0) {
      throw new Error('Text-to-speech input text is empty.');
    }
    if (this._textToSpeechActive) {
      throw new Error('Text-to-speech synthesis is already active.');
    }
    this._ensureTextToSpeechDir();
    const taskId = `${Date.now()}_${++this._mediaFileCounter}`;
    const speakerPath = `/tts/speaker_${taskId}.bin`;
    const outputPath = `/tts/output_${taskId}.pcm`;
    const speakerBytes = toUint8Array(options.speakerAudio);
    if (speakerBytes && speakerBytes.length > 0) {
      this._core!.FS.writeFile(speakerPath, speakerBytes);
    }

    const promptBatchSize = Number(options.promptBatchSize) > 0
      ? Math.max(1, Math.trunc(Number(options.promptBatchSize)))
      : 512;
    const maxFrames = Number(options.maxFrames) > 0
      ? Math.max(1, Math.trunc(Number(options.maxFrames)))
      : 512;
    const topK = Number.isFinite(options.topK)
      ? Math.max(0, Math.trunc(Number(options.topK)))
      : 40;
    const topP = Number.isFinite(options.topP) ? Number(options.topP) : 0.95;
    const minP = Number.isFinite(options.minP) ? Number(options.minP) : 0.0;
    const temperature = Number.isFinite(options.temperature)
      ? Number(options.temperature)
      : 0.8;
    const seed = Number.isInteger(options.seed)
      ? Number(options.seed) >>> 0
      : Math.floor(Math.random() * 0xffffffff) >>> 0;
    const language = typeof options.language === 'string' ? options.language : '';
    let started = false;

    this._textToSpeechActive = true;
    this._textToSpeechDone = new Promise<void>((resolve) => {
      this._resolveTextToSpeechDone = resolve;
    });

    try {
      this._abortRequested = false;
      const startRc = Number(
        await this._core!.ccall(
          'llamadart_webgpu_tts_start',
          'number',
          ['string', 'string', 'string', 'number', 'number', 'number', 'number', 'number', 'number', 'number'],
          [
            text,
            language,
            speakerBytes && speakerBytes.length > 0 ? speakerPath : null,
            promptBatchSize,
            maxFrames,
            topK,
            topP,
            minP,
            temperature,
            seed,
          ],
          { async: true },
        ),
      );
      if (startRc !== 0) {
        throw new Error(this._coreErrorMessage('Failed to start text-to-speech synthesis', startRc));
      }
      started = true;

      // Each step replaces this with the core's progress before it is read.
      let progress = {} as TextToSpeechProgress;
      for (;;) {
        if (this._abortRequested || options.signal?.aborted) {
          this.cancel();
        }
        const stepRc = Number(
          await this._core!.ccall(
            'llamadart_webgpu_tts_step',
            'number',
            [],
            [],
            { async: true },
          ),
        );
        const rawProgress = this._core!.ccall(
          'llamadart_webgpu_tts_progress_json',
          'string',
          [],
          [],
        ) || '{}';
        progress = JSON.parse(rawProgress);
        options.onProgress?.(progress);
        if (stepRc === -6 || progress.state === 4) {
          throw new DOMException('Text-to-speech synthesis was cancelled.', 'AbortError');
        }
        if (stepRc !== 0) {
          throw new Error(this._coreErrorMessage('Text-to-speech synthesis failed', stepRc));
        }
        if (progress.state === 3) {
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 0));
      }

      const outputRc = Number(
        this._core!.ccall(
          'llamadart_webgpu_tts_write_pcm',
          'number',
          ['string'],
          [outputPath],
        ),
      );
      if (outputRc !== 0) {
        throw new Error(this._coreErrorMessage('Failed to read synthesized speech', outputRc));
      }
      const bytes = this._core!.FS.readFile(outputPath);
      if (!(bytes instanceof Uint8Array) || bytes.byteLength === 0 || (bytes.byteLength % 4) !== 0) {
        throw new Error('Synthesized PCM output is empty or malformed.');
      }
      const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      const pcm = new Float32Array(copy);
      return {
        pcm,
        sampleRate: Number(capabilities.sampleRate) || 0,
        channels: Number(capabilities.channels) || 1,
        sampleCount: pcm.length,
        framesGenerated: Number(progress.framesGenerated) || 0,
        truncated: progress.truncated === true,
      };
    } finally {
      if (started) {
        try {
          this._core!.ccall('llamadart_webgpu_tts_reset', 'number', [], []);
        } catch (_) {
          // best-effort task cleanup
        }
      }
      this._deleteFsFile(speakerPath);
      this._deleteFsFile(outputPath);
      this._textToSpeechActive = false;
      this._resolveTextToSpeechDone?.();
      this._resolveTextToSpeechDone = null;
      this._textToSpeechDone = null;
    }
  }

  getDecisionCapabilities() {
    const unsupported = (reason: string) => ({
      apiVersion: DECISION_API_VERSION,
      supported: false,
      reason,
    });
    const core = this._core;
    if (!core) {
      return unsupported('WebGPU core is not initialized');
    }
    if (typeof core._llamadart_webgpu_decision_capabilities_json !== 'function') {
      return unsupported('This WebGPU core build does not include decision heads.');
    }
    const coreVersion = Number(
      core.ccall('llamadart_webgpu_decision_api_version', 'number', [], []),
    );
    if (coreVersion !== DECISION_API_VERSION) {
      return unsupported(
        `The WebGPU core implements decision API version ${coreVersion}; `
        + `this bridge needs version ${DECISION_API_VERSION}.`,
      );
    }
    const raw = core.ccall(
      'llamadart_webgpu_decision_capabilities_json',
      'string',
      [],
      [],
    ) || '{}';
    try {
      return JSON.parse(raw) as DecisionCapabilities;
    } catch (_) {
      return unsupported('WebGPU decision capability response is invalid');
    }
  }

  _ensureDecisionDir() {
    try {
      this._core!.FS.mkdir('/decision');
    } catch (_) {
      // Shared directory. The following file operation reports real failures.
    }
  }

  async _stageDecisionHeadFromUrl(
    url: string,
    headPath: string,
    onProgress: DecisionHeadOptions['onProgress'],
  ) {
    const fetchTimeoutMs = this._resolveFetchTimeoutMs({}, 180000);
    const chunkTimeoutMs = this._resolveStreamChunkTimeoutMs({}, 90000);
    for (let attempt = 0; ; attempt += 1) {
      try {
        const response = await this._fetchWithTimeout(
          url,
          { cache: 'no-store' },
          fetchTimeoutMs,
        );
        if (!response.ok) {
          throw new Error(
            `Failed to fetch decision head: ${response.status} ${response.statusText}`,
          );
        }
        await writeResponseToFsFileWithProgress(
          response,
          this._core!.FS,
          headPath,
          typeof onProgress === 'function' ? onProgress as ProgressCallback : null,
          {
            useBigIntPosition: this._coreVariant === 'wasm64',
            chunkTimeoutMs,
          },
        );
        return;
      } catch (error) {
        this._unlinkDecisionFile(headPath);
        if (!isRetryableStreamNetworkError(error) || attempt >= 1) {
          throw error;
        }
        this._runtimeNotes.push(`decision_head_fetch_retry:${attempt + 1}`);
      }
    }
  }

  async loadDecisionHead(
    source: string | ArrayBuffer | ArrayBufferView,
    options: DecisionHeadOptions = {},
  ) {
    if (!this._core || this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }
    const configJson = options?.configJson ?? null;
    if (configJson !== null && typeof configJson !== 'string') {
      throw new TypeError('Decision head configJson must be a string.');
    }
    let bytes = null;
    if (typeof source === 'string') {
      if (source.length === 0) {
        throw new Error('Decision head URL is empty.');
      }
    } else {
      bytes = bufferSourceBytes(source);
      if (!bytes) {
        throw new TypeError('Decision head source must be a URL string, an ArrayBuffer or a typed array.');
      }
      if (bytes.byteLength === 0) {
        throw new Error('Decision head bytes are empty.');
      }
    }
    const capabilities = this.getDecisionCapabilities();
    if (capabilities.supported !== true) {
      throw new Error(
        capabilities.reason || 'The loaded model does not support decision heads.',
      );
    }

    const core = this._core;
    this._ensureDecisionDir();
    const taskId = `${Date.now()}_${++this._decisionFileCounter}`;
    const headPath = `/decision/head_${taskId}.safetensors`;
    // configJson travels as a file: ccall copies string arguments onto the
    // fixed-size wasm stack.
    const configPath = configJson === null ? null : `/decision/config_${taskId}.json`;
    try {
      let label = 'decision head bytes';
      if (bytes) {
        core.FS.writeFile(headPath, bytes);
      } else {
        label = basenameFromUrl(source).slice(0, 256);
        await this._stageDecisionHeadFromUrl(source as string, headPath, options?.onProgress);
      }
      if (configPath !== null) {
        core.FS.writeFile(configPath, textEncoder.encode(configJson!));
      }
      const handle = Number(
        await core.ccall(
          'llamadart_webgpu_decision_load',
          'number',
          ['string', 'string', 'string'],
          [headPath, label, configPath],
          { async: true },
        ),
      );
      if (handle <= 0) {
        throw new Error(this._coreErrorMessage('Failed to load decision head', handle));
      }
      const raw = core.ccall(
        'llamadart_webgpu_decision_head_info_json',
        'string',
        ['number'],
        [handle],
      ) || '{}';
      return JSON.parse(raw) as DecisionHeadInfo;
    } finally {
      this._unlinkDecisionFile(headPath);
      this._unlinkDecisionFile(configPath);
    }
  }

  async runDecision(handle: number, sequences: unknown) {
    const nativeHandle = decisionHandleFrom(handle);
    const input = encodeDecisionSequences(sequences);
    if (!this._core || this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }
    const core = this._core;
    this._ensureDecisionDir();
    const taskId = `${Date.now()}_${++this._decisionFileCounter}`;
    const inputPath = `/decision/input_${taskId}.bin`;
    const outputPath = `/decision/output_${taskId}.bin`;
    try {
      core.FS.writeFile(inputPath, input);
      const count = Number(
        await core.ccall(
          'llamadart_webgpu_decision_run',
          'number',
          ['number', 'string', 'string'],
          [nativeHandle, inputPath, outputPath],
          { async: true },
        ),
      );
      if (count < 0) {
        throw new Error(this._coreErrorMessage('Decision run failed', count));
      }
      return decodeDecisionOutputs(core.FS.readFile(outputPath), count);
    } finally {
      this._unlinkDecisionFile(inputPath);
      this._unlinkDecisionFile(outputPath);
    }
  }

  async freeDecisionHead(handle: number) {
    const nativeHandle = decisionHandleFrom(handle);
    if (!this._core) {
      return;
    }
    await this._core.ccall(
      'llamadart_webgpu_decision_free',
      'number',
      ['number'],
      [nativeHandle],
      { async: true },
    );
  }

  getLoraAdapterCapabilities(): LoraAdapterCapabilities {
    const unsupported = (reason: string) => ({
      apiVersion: LORA_API_VERSION,
      supported: false,
      reason,
    });
    const core = this._core;
    if (!core) {
      return unsupported('WebGPU core is not initialized');
    }
    if (typeof core._llamadart_webgpu_lora_api_version !== 'function') {
      return unsupported('This WebGPU core build does not include LoRA adapters.');
    }
    const coreVersion = Number(
      core.ccall('llamadart_webgpu_lora_api_version', 'number', [], []),
    );
    if (coreVersion !== LORA_API_VERSION) {
      return unsupported(
        `The WebGPU core implements LoRA API version ${coreVersion}; `
        + `this bridge needs version ${LORA_API_VERSION}.`,
      );
    }
    return { apiVersion: LORA_API_VERSION, supported: true };
  }

  _requireLoraCore() {
    if (!this._core || this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }
    const capabilities = this.getLoraAdapterCapabilities();
    if (capabilities.supported !== true) {
      throw new Error(capabilities.reason || 'This bridge cannot load LoRA adapters.');
    }
    return this._core;
  }

  /**
   * Downloads a URL adapter into `adapterPath`, through the Cache API unless
   * `options.useCache` is false. A cache miss stores the response while it is
   * written, so progress starts with the first chunk.
   */
  async _stageLoraAdapterFromUrl(
    url: string,
    adapterPath: string,
    options: RuntimeLoraAdapterLoadOptions,
  ) {
    const core = this._core!;
    const signal = options.signal || null;
    const progressCallback = typeof options.progressCallback === 'function'
      ? options.progressCallback
      : null;
    const fetchTimeoutMs = this._resolveFetchTimeoutMs(options, 180000);
    const chunkTimeoutMs = this._resolveStreamChunkTimeoutMs(options, 90000);
    const fetchInit: RequestInit = signal ? { signal } : {};
    const cacheKey = normalizeAbsoluteUrl(url);
    let cache: Cache | null = null;
    if (options.useCache !== false && typeof globalThis.caches?.open === 'function') {
      try {
        cache = await globalThis.caches.open(this._resolveCacheName(options));
      } catch (_) {
        this._runtimeNotes.push('lora_cache_error');
      }
    }
    for (let attempt = 0; ; attempt += 1) {
      throwIfAborted(signal, LORA_LOAD_ABORT_MESSAGE);
      try {
        let response: Response | null = null;
        if (cache) {
          try {
            response = (await cache.match(cacheKey)) || null;
          } catch (_) {
            this._runtimeNotes.push('lora_cache_error');
          }
        }
        let stored: Promise<void> | null = null;
        if (!response) {
          response = await this._fetchWithTimeout(
            url,
            cache ? fetchInit : { cache: 'no-store', ...fetchInit },
            fetchTimeoutMs,
          );
          if (!response.ok) {
            throw new Error(
              `Failed to fetch LoRA adapter: ${response.status} ${response.statusText}`,
            );
          }
          if (cache) {
            stored = cache.put(cacheKey, response.clone()).then(
              () => {
                this._runtimeNotes.push('lora_cache_stored');
              },
              () => {
                this._runtimeNotes.push('lora_cache_store_failed');
              },
            );
          }
        }
        await writeResponseToFsFileWithProgress(
          response,
          core.FS,
          adapterPath,
          progressCallback,
          {
            useBigIntPosition: this._coreVariant === 'wasm64',
            chunkTimeoutMs,
            signal,
            abortMessage: LORA_LOAD_ABORT_MESSAGE,
          },
        );
        await stored;
        return;
      } catch (error) {
        this._deleteFsFile(adapterPath);
        throwIfAborted(signal, LORA_LOAD_ABORT_MESSAGE);
        if (!isRetryableStreamNetworkError(error) || attempt >= 1) {
          throw error;
        }
        this._runtimeNotes.push(`lora_fetch_retry:${attempt + 1}`);
      }
    }
  }

  async loadLoraAdapter(
    source: string | ArrayBuffer | ArrayBufferView,
    options: RuntimeLoraAdapterLoadOptions = {},
  ): Promise<LoraAdapterInfo> {
    let bytes = null;
    if (typeof source === 'string') {
      if (source.length === 0) {
        throw new Error('LoRA adapter URL is empty.');
      }
    } else {
      bytes = bufferSourceBytes(source);
      if (!bytes) {
        throw new TypeError('LoRA adapter source must be a URL string, an ArrayBuffer or a typed array.');
      }
      if (bytes.byteLength === 0) {
        throw new Error('LoRA adapter bytes are empty.');
      }
    }
    const core = this._requireLoraCore();
    const signal = options.signal || null;
    throwIfAborted(signal, LORA_LOAD_ABORT_MESSAGE);
    ensureFsDirectory(core.FS, '/lora');
    const adapterPath = `/lora/adapter_${++this._loraFileCounter}.gguf`;
    try {
      if (bytes) {
        core.FS.writeFile(adapterPath, bytes);
      } else {
        await this._stageLoraAdapterFromUrl(source as string, adapterPath, options);
      }
      throwIfAborted(signal, LORA_LOAD_ABORT_MESSAGE);
      const handle = Number(
        await core.ccall(
          'llamadart_webgpu_lora_load',
          'number',
          ['string'],
          [adapterPath],
          { async: true },
        ),
      );
      if (handle <= 0) {
        throw new Error(this._coreErrorMessage('Failed to load LoRA adapter', handle));
      }
      this._loraAdaptersLoaded = true;
      return { handle };
    } finally {
      this._deleteFsFile(adapterPath);
    }
  }

  async _callLoraCore(name: string, argTypes: CcallArgType[], args: number[], failure: string) {
    const core = this._requireLoraCore();
    const rc = Number(await core.ccall(name, 'number', argTypes, args, { async: true }));
    if (rc !== 0) {
      throw new Error(this._coreErrorMessage(failure, rc));
    }
  }

  async setLoraAdapter(handle: number, scale = 1) {
    await this._callLoraCore(
      'llamadart_webgpu_lora_set',
      ['number', 'number'],
      [loraHandleFrom(handle), loraScaleFrom(scale)],
      'Failed to apply LoRA adapter',
    );
  }

  async removeLoraAdapter(handle: number) {
    await this._callLoraCore(
      'llamadart_webgpu_lora_remove',
      ['number'],
      [loraHandleFrom(handle)],
      'Failed to remove LoRA adapter',
    );
  }

  async clearLoraAdapters() {
    await this._callLoraCore(
      'llamadart_webgpu_lora_clear',
      [],
      [],
      'Failed to clear LoRA adapters',
    );
  }

  _clearStagedMediaFiles() {
    if (!this._core || this._stagedMediaPaths.length === 0) {
      this._stagedMediaPaths = [];
      return;
    }

    for (const mediaPath of this._stagedMediaPaths) {
      try {
        this._core.FS.unlink(mediaPath);
      } catch (_) {
        // ignore best-effort cleanup failures
      }
    }

    this._stagedMediaPaths = [];
  }

  _clearPendingMedia() {
    this._core?.ccall('llamadart_webgpu_media_clear_pending', null, [], []);
    this._clearStagedMediaFiles();
  }

  _persistMediaBytes(bytes: Uint8Array, extension = '.bin') {
    if (!this._core) {
      throw new Error('WebGPU core is not initialized.');
    }

    try {
      this._core.FS.mkdir('/media');
    } catch (_) {
      // The directory is shared across requests. A following write reports any
      // real filesystem failure without relying on wasm64 analyzePath support.
    }

    this._mediaFileCounter += 1;
    const suffix = typeof extension === 'string' && extension.startsWith('.')
      ? extension
      : '.bin';
    const mediaPath = `/media/input_${Date.now()}_${this._mediaFileCounter}${suffix}`;
    this._core.FS.writeFile(mediaPath, bytes);
    this._stagedMediaPaths.push(mediaPath);
    return mediaPath;
  }

  _addMediaFile(mediaPath: string) {
    const rc = Number(
      this._core!.ccall(
        'llamadart_webgpu_media_add_file',
        'number',
        ['string'],
        [mediaPath],
      ),
    );
    if (rc !== 0) {
      throw new Error(this._coreErrorMessage('Failed to add media file', rc));
    }
  }

  _addRawRgbMediaBytes(bytes: Uint8Array, width: number, height: number) {
    const useHeapBuffer =
      this._core
      && typeof this._core._malloc === 'function'
      && typeof this._core._free === 'function'
      && this._core.HEAPU8
      && typeof this._core.HEAPU8.set === 'function';

    let rc = 0;
    if (useHeapBuffer) {
      const ptr = this._core!._malloc!(bytes.length);
      if (!Number.isFinite(ptr) || ptr <= 0) {
        throw new Error('Failed to allocate core heap buffer for RGB media bytes');
      }

      try {
        this._core!.HEAPU8!.set(bytes, ptr);
        rc = Number(
          this._core!.ccall(
            'llamadart_webgpu_media_add_rgb',
            'number',
            ['number', 'number', 'number', 'number'],
            [width, height, ptr, bytes.length],
          ),
        );
      } finally {
        this._core!._free!(ptr);
      }
    } else {
      rc = Number(
        this._core!.ccall(
          'llamadart_webgpu_media_add_rgb',
          'number',
          ['number', 'number', 'array', 'number'],
          [width, height, bytes, bytes.length],
        ),
      );
    }

    if (rc !== 0) {
      throw new Error(this._coreErrorMessage('Failed to add raw RGB media bytes', rc));
    }
  }

  _addAudioSamples(samples: Float32Array) {
    const sampleBytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
    const useHeapBuffer =
      this._core
      && typeof this._core._malloc === 'function'
      && typeof this._core._free === 'function'
      && this._core.HEAPU8
      && typeof this._core.HEAPU8.set === 'function';

    let rc = 0;
    if (useHeapBuffer) {
      const ptr = this._core!._malloc!(sampleBytes.length);
      if (!Number.isFinite(ptr) || ptr <= 0) {
        throw new Error('Failed to allocate core heap buffer for audio samples');
      }

      try {
        this._core!.HEAPU8!.set(sampleBytes, ptr);
        rc = Number(
          this._core!.ccall(
            'llamadart_webgpu_media_add_audio_f32',
            'number',
            ['number', 'number'],
            [ptr, samples.length],
          ),
        );
      } finally {
        this._core!._free!(ptr);
      }
    } else {
      rc = Number(
        this._core!.ccall(
          'llamadart_webgpu_media_add_audio_f32',
          'number',
          ['array', 'number'],
          [sampleBytes, samples.length],
        ),
      );
    }

    if (rc !== 0) {
      throw new Error(this._coreErrorMessage('Failed to add audio samples', rc));
    }
  }

  async _fetchMediaBytes(url: string) {
    const response = await this._fetchWithTimeout(
      url,
      { cache: 'no-store' },
      this._resolveFetchTimeoutMs({}, 120000),
    );
    if (!response.ok) {
      throw new Error(`Failed to fetch media: ${response.status} ${response.statusText}`);
    }

    return new Uint8Array(await response.arrayBuffer());
  }

  async _prepareImageBytesForMultimodal(bytes: Uint8Array, options: MediaImageLimitOptions = {}) {
    const maxPixels = this._resolveMediaImageMaxPixels(options);
    const maxEdge = this._resolveMediaImageMaxEdge(options);
    if (maxPixels <= 0 && maxEdge <= 0) {
      return null;
    }

    return decodeImageBytesToRgb(bytes, {
      maxPixels,
      maxEdge,
    });
  }

  async _stageMultimodalParts(
    parts: CompletionOptions['parts'],
    options: MediaImageLimitOptions = {},
  ) {
    this._clearPendingMedia();

    const mediaParts = Array.isArray(parts) ? parts : [];
    if (mediaParts.length === 0) {
      return;
    }

    if (!this._mmProjPath) {
      throw new Error(
        'Multimodal input requires a loaded projector. Call loadMultimodalProjector first.',
      );
    }

    for (const rawPart of mediaParts) {
      const part = (rawPart && typeof rawPart === 'object' ? rawPart : {}) as MediaPartInput;
      const type = String(part.type || '').toLowerCase();

      if (type === 'image') {
        const bytes = toUint8Array(part.bytes);
        if (bytes && bytes.length > 0) {
          const width = Number(part.width);
          const height = Number(part.height);
          const isRawRgb = Number.isInteger(width)
            && Number.isInteger(height)
            && width > 0
            && height > 0
            && bytes.length === (width * height * 3);

          if (isRawRgb) {
            this._addRawRgbMediaBytes(bytes, width, height);
          } else {
            const prepared = await this._prepareImageBytesForMultimodal(bytes, options);
            if (prepared && prepared.bytes && prepared.bytes.length > 0) {
              const mediaPath = this._persistMediaBytes(prepared.bytes, '.img');
              this._addMediaFile(mediaPath);
              if (prepared.resized) {
                this._runtimeNotes.push(
                  `media_image_resized:${prepared.sourceWidth}x${prepared.sourceHeight}->${prepared.width}x${prepared.height}`,
                );
              }
            } else {
              const mediaPath = this._persistMediaBytes(bytes, '.img');
              this._addMediaFile(mediaPath);
            }
          }
          continue;
        }

        if (typeof part.url !== 'string' || part.url.length === 0) {
          throw new Error('Image part must provide bytes or url.');
        }

        const fetched = await this._fetchMediaBytes(part.url);
        const prepared = await this._prepareImageBytesForMultimodal(fetched, options);
        if (prepared && prepared.bytes && prepared.bytes.length > 0) {
          const mediaPath = this._persistMediaBytes(prepared.bytes, '.img');
          this._addMediaFile(mediaPath);
          if (prepared.resized) {
            this._runtimeNotes.push(
              `media_image_resized:${prepared.sourceWidth}x${prepared.sourceHeight}->${prepared.width}x${prepared.height}`,
            );
          }
        } else {
          const mediaPath = this._persistMediaBytes(fetched, '.img');
          this._addMediaFile(mediaPath);
        }
        continue;
      }

      if (type === 'audio') {
        const samples = toFloat32Array(part.samples);
        if (samples && samples.length > 0) {
          this._addAudioSamples(samples);
          continue;
        }

        const bytes = toUint8Array(part.bytes);
        if (bytes && bytes.length > 0) {
          const mediaPath = this._persistMediaBytes(bytes, '.aud');
          this._addMediaFile(mediaPath);
          continue;
        }

        if (typeof part.url !== 'string' || part.url.length === 0) {
          throw new Error('Audio part must provide samples, bytes, or url.');
        }

        const fetched = await this._fetchMediaBytes(part.url);
        const mediaPath = this._persistMediaBytes(fetched, '.aud');
        this._addMediaFile(mediaPath);
      }
    }
  }

  getCompletionCapabilities(): CompletionCapabilities {
    const core = this._core;
    if (!core || typeof core._llamadart_webgpu_completion_capabilities_json !== 'function') {
      return { ...NO_COMPLETION_CAPABILITIES };
    }
    return completionCapabilitiesFrom(
      core.ccall('llamadart_webgpu_completion_capabilities_json', 'string', [], []),
    );
  }

  async createCompletion(prompt: string, options: RuntimeCompletionOptions = {}): Promise<string> {
    if (this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }
    const sampling = resolveCompletionSamplingOptions(options);
    requireCompletionCapabilities(sampling, this.getCompletionCapabilities());

    this._abortRequested = false;
    const startedAt = performance.now();
    let firstTokenAt: number | null = null;

    let nPredict = Number(options.nPredict) > 0 ? Number(options.nPredict) : 256;
    const hasMediaParts = Array.isArray(options.parts) && options.parts.length > 0;
    if (hasMediaParts) {
      const requestedMediaCap = Number(options.mediaMaxPredict);
      const mediaCap = Number.isFinite(requestedMediaCap) && requestedMediaCap > 0
        ? Math.max(32, Math.trunc(requestedMediaCap))
        : 256;
      if (nPredict > mediaCap) {
        nPredict = mediaCap;
        this._runtimeNotes.push(`media_n_predict_capped:${mediaCap}`);
      }
    }
    const temp = Number.isFinite(options.temp) ? Number(options.temp) : 0.8;
    const topK = Number.isFinite(options.topK) ? Number(options.topK) : 40;
    const topP = Number.isFinite(options.topP) ? Number(options.topP) : 0.95;
    const penalty = Number.isFinite(options.penalty) ? Number(options.penalty) : 1.1;
    const grammar = typeof options.grammar === 'string' && options.grammar.length > 0
      ? options.grammar
      : null;
    const seed = Number.isInteger(options.seed)
      ? Number(options.seed)
      : Math.floor(Math.random() * 0xffffffff);

    await this._stageMultimodalParts(options.parts, options);

    let generationStarted = false;

    try {
      const beginRc = Number(
        await this._core!.ccall(
          'llamadart_webgpu_begin_generation',
          'number',
          [
            'string', 'number', 'number', 'number', 'number', 'string', 'number', 'number', 'number',
            'number', 'string', 'string', 'string',
          ],
          [
            String(prompt),
            temp,
            topK,
            topP,
            penalty,
            grammar,
            seed >>> 0,
            sampling.minP,
            sampling.presencePenalty,
            sampling.thinkingBudget?.maxTokens ?? 0,
            sampling.thinkingBudget?.startTag ?? null,
            sampling.thinkingBudget?.endTag ?? null,
            sampling.thinkingBudget?.forcedMessage ?? null,
          ],
          { async: true },
        ),
      );

      if (beginRc === GENERATION_ALREADY_ACTIVE_RC) {
        throw new Error(GENERATION_ALREADY_ACTIVE_MESSAGE);
      }

      if (beginRc !== 0) {
        throw new Error(this._coreErrorMessage('Failed to start generation', beginRc));
      }

      generationStarted = true;

      let generated = 0;
      const shouldEmitCurrentText = options.emitCurrentTextOnToken !== false;
      const tokenEventEncoding = typeof options.tokenEventEncoding === 'string'
        ? String(options.tokenEventEncoding || '').toLowerCase()
        : 'bytes';
      const emitTokenText = tokenEventEncoding === 'text';
      // Yield in workers as well as on the main thread. Without a macrotask
      // boundary, worker token events and cancellation messages remain queued
      // until generation has already completed.
      const yieldInterval = 4;
      let streamed = '';
      let emittedStableText = '';
      let finishReason: CompletionFinishReason = 'length';

      while (generated < nPredict) {
        if (this._abortRequested || options.signal?.aborted) {
          finishReason = 'cancelled';
          break;
        }

        const stepRc = Number(
          await this._core!.ccall(
            'llamadart_webgpu_next_token',
            'number',
            [],
            [],
            { async: true },
          ),
        );
        if (stepRc === 0) {
          finishReason = this._abortRequested || options.signal?.aborted
            ? 'cancelled'
            : 'stop';
          break;
        }

        if (stepRc < 0) {
          const stepErrorText = this._coreErrorMessage('Generation step failed', stepRc);

          if (generated > 0 && this._isContextLimitGenerationError(stepErrorText)) {
            this._runtimeNotes.push('generation_stopped_context_limit');
            this._emitLogger(
              'warn',
              'warning: generation reached context/memory limit; returning partial output.',
            );
            break;
          }

          if (this._shouldAttemptGenerationRecovery(stepErrorText, options, generated)) {
            const recovered = await this._recoverGenerationWithCpuFallback(options);
            if (recovered) {
              if (generationStarted) {
                try {
                  this._core!.ccall('llamadart_webgpu_end_generation', null, [], []);
                } catch (_) {
                  // best-effort cleanup before retry
                }
                generationStarted = false;
              }
              this._clearPendingMedia();

              const retryOptions = {
                ...options,
                _llamadartGenerationRecoveryAttempted: true,
              };
              return await this.createCompletion(prompt, retryOptions);
            }
          }

          throw new Error(stepErrorText);
        }

        generated += 1;
        const fullText = this._core!.ccall('llamadart_webgpu_last_output', 'string', [], []) || '';
        streamed = fullText;
        const stableText = trimUnstableUtf8Tail(fullText);

        if (!stableText.startsWith(emittedStableText)) {
          emittedStableText = '';
        }

        const deltaText = stableText.slice(emittedStableText.length);
        if (deltaText.length === 0) {
          continue;
        }
        emittedStableText = stableText;
        firstTokenAt ??= performance.now();

        if (typeof options.onToken === 'function') {
          const piecePayload = emitTokenText
            ? deltaText
            : textEncoder.encode(deltaText);
          options.onToken(piecePayload, shouldEmitCurrentText ? fullText : null);
        }

        if (yieldInterval > 0 && (generated % yieldInterval) === 0) {
          await new Promise((resolve) => setTimeout(resolve, 0));
        }
      }

      const text = this._core!.ccall('llamadart_webgpu_last_output', 'string', [], []) || streamed || '';
      const tailText = text.startsWith(emittedStableText)
        ? text.slice(emittedStableText.length)
        : '';
      if (tailText.length > 0) {
        firstTokenAt ??= performance.now();
        if (typeof options.onToken === 'function') {
          const piecePayload = emitTokenText
            ? tailText
            : textEncoder.encode(tailText);
          options.onToken(piecePayload, shouldEmitCurrentText ? text : null);
        }
      }
      if (typeof options.onUsage === 'function') {
        const counts = JSON.parse(
          this._core!.ccall('llamadart_webgpu_last_generation_usage_json', 'string', [], []) || '{}',
        );
        options.onUsage({
          promptTokens: Number(counts?.promptTokens) || 0,
          cachedPromptTokens: Number(counts?.cachedPromptTokens) || 0,
          completionTokens: Number(counts?.completionTokens) || 0,
          timeToFirstTokenMs: firstTokenAt == null ? null : firstTokenAt - startedAt,
          durationMs: performance.now() - startedAt,
          finishReason,
        });
      }
      return text;
    } finally {
      if (generationStarted) {
        this._core!.ccall('llamadart_webgpu_end_generation', null, [], []);
      }
      this._clearPendingMedia();
    }
  }

  async tokenize(text: string, _addSpecial = true) {
    if (this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }

    const rc = Number(
      await this._core!.ccall(
        'llamadart_webgpu_tokenize_to_json',
        'number',
        ['string', 'number'],
        [String(text), _addSpecial ? 1 : 0],
        { async: true },
      ),
    );

    if (rc < 0) {
      throw new Error(this._coreErrorMessage('Tokenization failed', rc));
    }

    const raw = this._core!.ccall('llamadart_webgpu_last_tokens_json', 'string', [], []) || '[]';
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.map((v) => Number(v) | 0)
      : [];
  }

  _ensureStateDir() {
    const core = this._core;
    if (!core?.FS) {
      throw new Error('Bridge filesystem is not initialized');
    }

    ensureFsDirectory(core.FS, '/states');
  }

  _normalizeStateTokens(tokens: number[] | ArrayLike<number> | null | undefined) {
    const normalized = Array.isArray(tokens)
      ? tokens
      : Array.from(tokens || []);
    return normalized.map((value) => Number(value) | 0);
  }

  _nextStateTempPath() {
    this._ensureStateDir();
    this._stateFileCounter += 1;
    return `/states/state_${Date.now()}_${this._stateFileCounter}.bin`;
  }

  async stateSaveFile(path: string, tokens: number[] | ArrayLike<number> = []): Promise<true> {
    if (this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }
    if (typeof path !== 'string' || path.length === 0) {
      throw new Error('State file path is empty.');
    }

    const normalized = this._normalizeStateTokens(tokens);
    const tokenText = JSON.stringify(normalized);
    const rc = Number(
      await this._core!.ccall(
        'llamadart_webgpu_state_save_file',
        'number',
        ['string', 'string'],
        [path, tokenText],
        { async: true },
      ),
    );

    if (rc < 0) {
      throw new Error(this._coreErrorMessage('State save failed', rc));
    }

    return true;
  }

  async stateLoadFile(path: string, tokenCapacity = this.getContextSize()) {
    if (this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }
    if (typeof path !== 'string' || path.length === 0) {
      throw new Error('State file path is empty.');
    }

    const capacity = Number(tokenCapacity) > 0
      ? Math.trunc(Number(tokenCapacity))
      : this.getContextSize();
    const rc = Number(
      await this._core!.ccall(
        'llamadart_webgpu_state_load_file',
        'number',
        ['string', 'number'],
        [path, capacity],
        { async: true },
      ),
    );

    if (rc < 0) {
      throw new Error(this._coreErrorMessage('State load failed', rc));
    }

    const raw = this._core!.ccall('llamadart_webgpu_last_tokens_json', 'string', [], []) || '[]';
    const parsed = JSON.parse(raw);
    const restoredTokens = Array.isArray(parsed)
      ? parsed.map((v) => Number(v) | 0)
      : [];

    return { tokens: restoredTokens };
  }

  async stateSaveBytes(tokens: number[] | ArrayLike<number> = []) {
    if (this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }

    const tempPath = this._nextStateTempPath();
    try {
      await this.stateSaveFile(tempPath, tokens);
      const bytes = this._core!.FS.readFile(tempPath);
      return bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes || []);
    } finally {
      this._deleteFsFile(tempPath);
    }
  }

  async stateLoadBytes(
    bytes: Uint8Array | ArrayBuffer | ArrayLike<number>,
    tokenCapacity = this.getContextSize(),
  ) {
    if (this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }

    const normalizedBytes = toUint8Array(bytes);
    if (!normalizedBytes || normalizedBytes.length === 0) {
      throw new Error('State bytes are empty.');
    }

    const tempPath = this._nextStateTempPath();
    try {
      this._core!.FS.writeFile(tempPath, normalizedBytes);
      return await this.stateLoadFile(tempPath, tokenCapacity);
    } finally {
      this._deleteFsFile(tempPath);
    }
  }

  async detokenize(tokens: number[] | ArrayLike<number>, _special = false) {
    if (this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }

    const normalized = Array.isArray(tokens)
      ? tokens
      : Array.from(tokens || []);
    const tokenText = JSON.stringify(normalized.map((v) => Number(v) | 0));

    const rc = Number(
      await this._core!.ccall(
        'llamadart_webgpu_detokenize_from_json',
        'number',
        ['string', 'number'],
        [tokenText, _special ? 1 : 0],
        { async: true },
      ),
    );

    if (rc < 0) {
      throw new Error(this._coreErrorMessage('Detokenization failed', rc));
    }

    return this._core!.ccall('llamadart_webgpu_last_detokenized', 'string', [], []) || '';
  }

  async embed(text: string, options: EmbedOptions = {}) {
    if (this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }

    const normalize = options?.normalize !== false;
    const rc = Number(
      await this._core!.ccall(
        'llamadart_webgpu_embed_to_json',
        'number',
        ['string', 'number'],
        [String(text), normalize ? 1 : 0],
        { async: true },
      ),
    );

    if (rc < 0) {
      throw new Error(this._coreErrorMessage('Embedding generation failed', rc));
    }

    const raw = this._core!.ccall('llamadart_webgpu_last_embedding_json', 'string', [], []) || '[]';
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.map((v) => {
        const numeric = Number(v);
        return Number.isFinite(numeric) ? numeric : 0;
      })
      : [];
  }

  async embedBatch(texts: string[], options: EmbedOptions = {}) {
    const normalized = Array.isArray(texts)
      ? texts
      : Array.from(texts || []);
    if (normalized.length === 0) {
      return [];
    }

    const normalize = options?.normalize !== false;
    const vectors = [];
    for (const text of normalized) {
      vectors.push(await this.embed(String(text), { normalize }));
    }
    return vectors;
  }

  async scoreNextToken(prompt: string, options: NextTokenScoreOptions = {}) {
    if (this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }
    if (typeof prompt !== 'string') {
      throw new TypeError('Next-token scoring prompt must be a string.');
    }

    const candidates = nextTokenCandidateIds(options?.candidates);
    const topK = options?.topK ?? 0;
    if (!isInt32(topK)) {
      throw new TypeError(`Next-token topK is ${String(topK)}; expected a 32-bit integer.`);
    }
    const reusePromptPrefix = options?.reusePromptPrefix !== false;

    const rc = Number(
      await this._core!.ccall(
        'llamadart_webgpu_score_next_token_to_json',
        'number',
        ['string', 'string', 'number', 'number'],
        [prompt, JSON.stringify(candidates), topK, reusePromptPrefix ? 1 : 0],
        { async: true },
      ),
    );

    if (rc < 0) {
      throw new Error(this._coreErrorMessage('Next-token scoring failed', rc));
    }

    const raw = this._core!.ccall('llamadart_webgpu_last_next_token_scores_json', 'string', [], []) || '{}';
    const parsed = JSON.parse(raw);
    return {
      candidates: scoredTokensFrom(parsed?.candidates),
      top: scoredTokensFrom(parsed?.top),
      promptTokens: Number(parsed?.promptTokens) || 0,
    };
  }

  getModelMetadata() {
    let modelMetadata = {};

    try {
      const raw = this._core?.ccall('llamadart_webgpu_model_meta_json', 'string', [], []);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object') {
          modelMetadata = parsed;
        }
      }
    } catch (_) {
      // Keep fallback metadata only.
    }

    return {
      ...modelMetadata,
      'llamadart.webgpu.prototype': '1',
      'llamadart.webgpu.backends': this._backendLabels.join(','),
      'llamadart.webgpu.model_bytes': String(this._modelBytes),
      'llamadart.webgpu.n_threads': String(this._threads),
      'llamadart.webgpu.n_threads_batch': String(this._threadsBatch),
      'llamadart.webgpu.n_batch': this._nBatch > 0 ? String(this._nBatch) : '',
      'llamadart.webgpu.n_ubatch': this._nUbatch > 0 ? String(this._nUbatch) : '',
      'llamadart.webgpu.n_seq_max': this._nSeqMax > 0 ? String(this._nSeqMax) : '',
      'llamadart.webgpu.flash_attention': String(this._flashAttention),
      'llamadart.webgpu.cache_type_k': String(this._cacheTypeK),
      'llamadart.webgpu.cache_type_v': String(this._cacheTypeV),
      'llamadart.webgpu.kv_unified':
        this._kvUnified >= 0 ? String(this._kvUnified) : '',
      'llamadart.webgpu.rope_freq_base':
        this._ropeFrequencyBase > 0 ? String(this._ropeFrequencyBase) : '',
      'llamadart.webgpu.rope_freq_scale':
        this._ropeFrequencyScale > 0 ? String(this._ropeFrequencyScale) : '',
      'llamadart.webgpu.split_mode':
        this._splitMode >= 0 ? String(this._splitMode) : '',
      'llamadart.webgpu.main_gpu':
        this._mainGpu >= 0 ? String(this._mainGpu) : '',
      'llamadart.webgpu.thread_pool_size':
        Number.isFinite(this._threadPoolSizeHint) && this._threadPoolSizeHint! > 0
          ? String(this._threadPoolSizeHint)
          : '',
      'llamadart.webgpu.n_gpu_layers': String(this._nGpuLayers),
      'llamadart.webgpu.core_variant': this._coreVariant,
      'llamadart.webgpu.model_source': this._modelSource,
      'llamadart.webgpu.model_cache_state': this._modelCacheState,
      'llamadart.webgpu.model_cache_name': this._modelCacheName,
      'llamadart.webgpu.runtime_notes': this._runtimeNotes.join(';'),
      'llamadart.webgpu.mmproj_loaded': this._mmProjPath ? '1' : '0',
      'llamadart.webgpu.supports_vision': this._mmSupportsVision ? '1' : '0',
      'llamadart.webgpu.supports_audio': this._mmSupportsAudio ? '1' : '0',
    };
  }

  getContextSize() {
    try {
      const nctx = Number(this._core?.ccall('llamadart_webgpu_get_context_size', 'number', [], []));
      if (nctx > 0) {
        return nctx;
      }
    } catch (_) {
      // fall through to cached value
    }

    return this._nCtx;
  }

  isGpuActive() {
    return this._gpuActive;
  }

  getBackendName() {
    if (this._nGpuLayers === 0) {
      return 'WASM (Prototype bridge)';
    }

    if (this._backendLabels.length > 0) {
      return this._backendLabels.join(', ');
    }
    return this._gpuActive
      ? 'WebGPU (Prototype bridge)'
      : 'WASM (Prototype bridge)';
  }

  setLogLevel(level: string | number) {
    if (Number.isFinite(level)) {
      this._logLevel = Math.max(0, Math.min(4, Math.trunc(level as number)));
    }
    this._applyCoreLogLevel();
  }

  cancel() {
    this._abortRequested = true;

    if (this._activeTransferAbortController) {
      try {
        this._activeTransferAbortController.abort();
      } catch (_) {
        // ignore best-effort transfer abort failures
      }
      this._activeTransferAbortController = null;
    }

    try {
      this._core?.ccall('llamadart_webgpu_request_cancel', null, [], []);
    } catch (_) {
      // ignore best-effort cancel failures
    }
  }

  async dispose() {
    if (this._textToSpeechActive && this._textToSpeechDone) {
      this.cancel();
      await this._textToSpeechDone;
    }

    if (this._activeTransferAbortController) {
      try {
        this._activeTransferAbortController.abort();
      } catch (_) {
        // ignore best-effort transfer abort failures
      }
      this._activeTransferAbortController = null;
    }

    if (this._core) {
      this._clearPendingMedia();
      const mmprojPath = this._mmProjPath;
      const mmprojFreeRc = Number(
        this._core.ccall('llamadart_webgpu_mmproj_free', 'number', [], []),
      );
      this._core.ccall('llamadart_webgpu_shutdown', null, [], []);
      this._deleteFsFile(mmprojPath);
      if (mmprojFreeRc !== 0) {
        this._emitLogger(
          'warn',
          this._coreErrorMessage('Failed to unload multimodal projector during dispose', mmprojFreeRc),
        );
      }
    }
    this._modelPath = null;
    this._modelPaths = [];
    this._modelBytes = 0;
    this._modelSource = 'network';
    this._modelCacheState = 'disabled';
    this._loadedModelUrl = null;
    this._mmProjPath = null;
    this._mmProjSourceUrl = null;
    this._mmSupportsVision = false;
    this._mmSupportsAudio = false;
    this._loraAdaptersLoaded = false;
    this._abortRequested = false;
    this._textToSpeechActive = false;
    this._textToSpeechDone = null;
    this._resolveTextToSpeechDone = null;
    this._activeTransferAbortController = null;
    this._suppressedWarmupWarningCount = 0;
    this._didReportWarmupWarningSuppression = false;
  }

  async applyChatTemplate(
    messages: Array<Record<string, unknown>>,
    addAssistant = true,
    _customTemplate: string | null = null,
  ) {
    return buildPromptFromMessages(messages, addAssistant);
  }
}
