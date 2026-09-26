export type LlamaCoreModuleFactory = (options?: Record<string, unknown>) => unknown | Promise<unknown>;

export interface LlamaWebGpuBridgeConfig {
  coreModuleUrl?: string;
  coreModuleUrlMem64?: string;
  wasmUrl?: string;
  wasmUrlMem64?: string;
  coreModuleFactory?: LlamaCoreModuleFactory | Promise<LlamaCoreModuleFactory>;
  workerUrl?: string;
  disableWorker?: boolean;
  preferMemory64?: boolean;
  workerRequestTimeoutMs?: number;
  workerInitTimeoutMs?: number;
  workerModelLoadTimeoutMs?: number;
  workerMmprojLoadTimeoutMs?: number;
  workerCompletionTimeoutMs?: number;
  workerTextToSpeechTimeoutMs?: number;
  workerGenerationStallTimeoutMs?: number;
  coreInitTimeoutMs?: number;
  cacheName?: string;
  threads?: number;
  threadsBatch?: number;
  threadPoolSize?: number;
  nBatch?: number;
  nUbatch?: number;
  nGpuLayers?: number;
  userAgent?: string;
  remoteFetchThresholdBytes?: number;
  remoteFetchChunkBytes?: number;
  mediaMaxImagePixels?: number;
  mediaMaxImageEdge?: number;
  disableImageDownscale?: boolean;
  allowAutoRemoteFetchBackend?: boolean;
  logLevel?: string | number;
  [key: string]: unknown;
}

export interface BridgeProgressEvent {
  loaded?: number;
  total?: number;
  shard?: number;
  shards?: number;
  url?: string;
  [key: string]: unknown;
}

export interface LoadModelOptions {
  progressCallback?: (progress: BridgeProgressEvent) => void;
  signal?: AbortSignal;
  remoteFetchThresholdBytes?: number;
  remoteFetchChunkBytes?: number;
  [key: string]: unknown;
}

export type TokenEventEncoding = 'bytes' | 'text' | (string & {});

export type CompletionMediaData = ArrayBuffer | ArrayBufferView | readonly number[];

export type CompletionImagePart = {
  type: 'image';
  width?: number;
  height?: number;
} & (
  | { bytes: CompletionMediaData; url?: string }
  | { bytes?: CompletionMediaData; url: string }
);

export type CompletionAudioPart = {
  type: 'audio';
} & (
  | { samples: CompletionMediaData; bytes?: CompletionMediaData; url?: string }
  | { samples?: CompletionMediaData; bytes: CompletionMediaData; url?: string }
  | { samples?: CompletionMediaData; bytes?: CompletionMediaData; url: string }
);

export type CompletionMediaPart = CompletionImagePart | CompletionAudioPart;

/** Why a completion stopped. */
export type CompletionFinishReason = 'stop' | 'length' | 'cancelled';

/** Token counts and timings of one completion. */
export interface CompletionUsage {
  /** Prompt tokens in the context when generation started, including `cachedPromptTokens`; context positions for a multimodal prompt. */
  promptTokens: number;
  /** Leading prompt tokens kept from the previous prompt or a loaded state instead of being evaluated again; 0 for a multimodal prompt. */
  cachedPromptTokens: number;
  /** Generated tokens in the returned text; the end-of-generation token is not counted. */
  completionTokens: number;
  /** Milliseconds from the runtime starting the completion to its first streamed text; null when it streamed none. */
  timeToFirstTokenMs: number | null;
  /** Milliseconds from the runtime starting the completion to its end. */
  durationMs: number;
  /** `stop` at an end-of-generation token, `length` at `nPredict` or the context limit, `cancelled` after `cancel()` or an abort. */
  finishReason: CompletionFinishReason;
}

export interface CompletionOptions {
  onToken?: (piece: string | Uint8Array, currentText: string | null) => void;
  /** Called at most once, with the usage of the last attempt whose generation returned, before `createCompletion` resolves or rejects with an `AbortError`. Not called when no generation returned: other rejections, a skipped multimodal warmup, or a worker failure after a cancel. */
  onUsage?: (usage: CompletionUsage) => void;
  signal?: AbortSignal;
  warmup?: boolean;
  emitCurrentTextOnToken?: boolean;
  tokenEventEncoding?: TokenEventEncoding;
  tokenEventFlushMs?: number;
  tokenEventFlushChars?: number;
  nPredict?: number;
  mediaMaxPredict?: number;
  parts?: readonly CompletionMediaPart[];
  temp?: number;
  topK?: number;
  topP?: number;
  penalty?: number;
  grammar?: string;
  seed?: number;
}

export interface EmbedOptions {
  normalize?: boolean;
}

export interface NextTokenScoreOptions {
  /** Token ids to score, in the order the result lists them. */
  candidates?: ArrayLike<number>;
  /** How many of the most probable tokens to return; 0 returns none. */
  topK?: number;
  /** Reuse the KV cache prefix shared with the previous prompt. Defaults to true. */
  reusePromptPrefix?: boolean;
}

export interface ScoredToken {
  token: number;
  /** The token's text as raw bytes; a piece may split a UTF-8 sequence. */
  bytes: Uint8Array;
  /** Natural-log probability; -Infinity when it is zero or a non-finite logit makes it undefined. */
  logprob: number;
}

export interface NextTokenScores {
  candidates: ScoredToken[];
  top: ScoredToken[];
  promptTokens: number;
}

export interface TextToSpeechCapabilities {
  apiVersion: number;
  supported: boolean;
  modelType: number;
  capabilities: number;
  supportsLanguage: boolean;
  supportsSpeakerReference: boolean;
  sampleRate: number;
  channels: number;
  reason: string;
}

export interface TextToSpeechProgress {
  state: number;
  promptTokensRemaining: number;
  framesGenerated: number;
  truncated: boolean;
}

export interface TextToSpeechOptions {
  text: string;
  language?: string;
  speakerAudio?: Uint8Array | ArrayBuffer | ArrayLike<number>;
  promptBatchSize?: number;
  maxFrames?: number;
  topK?: number;
  topP?: number;
  minP?: number;
  temperature?: number;
  seed?: number;
  signal?: AbortSignal;
  onProgress?: (progress: TextToSpeechProgress) => void;
}

export interface TextToSpeechResult {
  pcm: Float32Array;
  sampleRate: number;
  channels: number;
  sampleCount: number;
  framesGenerated: number;
  truncated: boolean;
}

export interface DecisionCapabilities {
  apiVersion: number;
  supported: boolean;
  /** Why the loaded model cannot run decision heads; absent when supported. */
  reason?: string;
}

export interface DecisionHeadOptions {
  /** Laya `rl_agent_config.json` text; omit to read the head's `laya.config` metadata. */
  configJson?: string | null;
  /** Download progress of a URL head, in bytes. */
  onProgress?: (progress: { loaded: number; total: number }) => void;
}

export interface DecisionHeadInfo {
  apiVersion: number;
  handle: number;
  hiddenSize: number;
  clsToken: number;
  sepToken: number;
  maskToken: number;
  maskText: string;
  configJson: string;
  deviceName: string;
}

/** 0 = choice, 1 = score, 2 = noul. */
export type DecisionQuestionType = 0 | 1 | 2;

export interface DecisionSequence {
  tokens: Int32Array | readonly number[];
  /** Token indexes of the option markers, in option order. */
  markers: Int32Array | readonly number[];
  questionType: DecisionQuestionType;
}

export interface DecisionOutput {
  /** One raw logit per marker. */
  logits: Float32Array;
  actLogits: Float32Array;
}

export type ModelMetadata = Record<string, unknown>;

export interface StateLoadResult {
  tokens: number[];
}

export function enableBridgeWorkerHost(): void;

/**
 * Runtime-backed async operations use a per-instance FIFO queue. Queued aborts
 * skip dispatch without cancelling the active owner; `cancel` and `setLogLevel`
 * remain out of band; disposal rejects queued/new work while active work settles
 * first. See docs/api.md "Operation serialization".
 */
export class LlamaWebGpuBridge {
  static supportsSafariAdaptiveGpu: boolean;
  /** `true` when `createCompletion` accepts `onUsage`. Bridges before v0.1.54 lack it; through v0.1.52 their worker mode rejects a function option with a `DataCloneError`. */
  static supportsCompletionUsage: boolean;

  constructor(config?: LlamaWebGpuBridgeConfig);

  loadModelFromUrl(url: string | string[], options?: LoadModelOptions): Promise<unknown>;
  prefetchModelToCache(url: string | string[], options?: LoadModelOptions): Promise<unknown>;
  evictModelFromCache(url: string | string[], options?: Record<string, unknown>): Promise<unknown>;

  createCompletion(prompt: string, options?: CompletionOptions): Promise<string>;
  tokenize(text: string, addSpecial?: boolean): Promise<number[]>;
  detokenize(tokens: number[] | ArrayLike<number>, special?: boolean): Promise<string>;
  applyChatTemplate(
    messages: Array<Record<string, unknown>>,
    addAssistant?: boolean,
    customTemplate?: string | null,
  ): Promise<string>;

  stateSaveFile(path: string, tokens?: number[] | ArrayLike<number>): Promise<true>;
  stateLoadFile(path: string, tokenCapacity?: number): Promise<StateLoadResult>;
  stateSaveBytes(tokens?: number[] | ArrayLike<number>): Promise<Uint8Array>;
  stateLoadBytes(bytes: Uint8Array | ArrayBuffer | ArrayLike<number>, tokenCapacity?: number): Promise<StateLoadResult>;

  embed(text: string, options?: EmbedOptions): Promise<number[]>;
  embedBatch(texts: string[], options?: EmbedOptions): Promise<number[][]>;
  scoreNextToken(prompt: string, options?: NextTokenScoreOptions): Promise<NextTokenScores>;

  loadMultimodalProjector(url: string): Promise<unknown>;
  unloadMultimodalProjector(): Promise<unknown>;
  supportsVision(): boolean;
  supportsAudio(): boolean;
  getTextToSpeechCapabilities(): Promise<TextToSpeechCapabilities>;
  synthesizeSpeech(options: TextToSpeechOptions): Promise<TextToSpeechResult>;
  getDecisionCapabilities(): Promise<DecisionCapabilities>;
  loadDecisionHead(
    source: string | ArrayBuffer | ArrayBufferView,
    options?: DecisionHeadOptions,
  ): Promise<DecisionHeadInfo>;
  runDecision(handle: number, sequences: readonly DecisionSequence[]): Promise<DecisionOutput[]>;
  freeDecisionHead(handle: number): Promise<void>;

  getModelMetadata(): ModelMetadata | null;
  getContextSize(): number;
  isGpuActive(): boolean;
  getBackendName(): string;
  setLogLevel(level: string | number): void;
  cancel(): void;
  dispose(): Promise<void>;
}

declare global {
  interface Window {
    LlamaWebGpuBridge?: typeof LlamaWebGpuBridge;
  }
}
