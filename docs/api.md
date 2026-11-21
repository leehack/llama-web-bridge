# LlamaWebGpuBridge public API

This document describes the browser-facing JavaScript API exported by
`llama_webgpu_bridge.js`. It is the reference for applications that use the
bridge assets directly instead of going through a higher-level SDK such as
`llamadart`.

The TypeScript declaration shipped with the browser asset is
`llama_webgpu_bridge.d.ts`; this page explains the runtime behavior behind those
signatures.

## Contents

- [Importing the bridge](#importing-the-bridge)
- [Minimal usage](#minimal-usage)
- [Runtime model](#runtime-model)
- [Operation serialization](#operation-serialization)
- [Constructor](#constructor)
- [Model loading and cache helpers](#model-loading-and-cache-helpers)
- [Text generation](#text-generation)
- [Draft models](#draft-models)
- [Tokenization and chat templates](#tokenization-and-chat-templates)
- [State persistence](#state-persistence)
- [Embeddings](#embeddings)
- [Next-token scoring](#next-token-scoring)
- [Multimodal projector APIs](#multimodal-projector-apis)
- [Text-to-speech](#text-to-speech)
- [Decision heads](#decision-heads)
- [LoRA adapters](#lora-adapters)
- [Runtime metadata and diagnostics](#runtime-metadata-and-diagnostics)
- [Worker host entrypoint](#worker-host-entrypoint)
- [Browser and asset requirements](#browser-and-asset-requirements)

## Importing the bridge

```js
import {
  LlamaWebGpuBridge,
  enableBridgeWorkerHost,
} from './webgpu_bridge/llama_webgpu_bridge.js';
```

A browser page that imports the bridge also receives `window.LlamaWebGpuBridge`
when the global is not already defined. Worker bootstrap code can either import
`llama_webgpu_bridge_worker.js` or import `llama_webgpu_bridge.js` and call
`enableBridgeWorkerHost()`.

The same module exports these error classes for callers to import:

| Export | Reference |
| --- | --- |
| `ModelUnloadedError` | [Native core aborts](#native-core-aborts) |
| `EmbeddingError` | [Embeddings](#embeddings) |
| `CrossOriginIsolationError` | [Browser and asset requirements](#browser-and-asset-requirements) |
| `SingleThreadCoreError` | [Model loading options](#loadmodelfromurlurl-options) |
| `ModelStagingError` | [Browser and asset requirements](#browser-and-asset-requirements) |
| `TextToSpeechStreamingUnsupportedError` | [Streamed audio](#streamed-audio) |
| `BlobWorkerBlockedError` | [Blocked blob workers](#blobworkerblockederror) |

## Minimal usage

```js
const bridge = new LlamaWebGpuBridge({
  coreModuleUrl: './webgpu_bridge/llama_webgpu_core.js',
  wasmUrl: './webgpu_bridge/llama_webgpu_core.wasm',
  workerUrl: './webgpu_bridge/llama_webgpu_bridge_worker.js',
  threads: 2,
});

try {
  await bridge.loadModelFromUrl('./models/model.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    progressCallback: ({ loaded, total }) => {
      if (total) {
        console.log(`loaded ${Math.round((loaded / total) * 100)}%`);
      }
    },
  });

  const answer = await bridge.createCompletion('2+2 =', {
    nPredict: 16,
    temp: 0,
    onToken: (piece, currentText) => {
      const text = typeof piece === 'string'
        ? piece
        : new TextDecoder().decode(piece);
      console.log('token', text, currentText);
    },
  });
  console.log(answer);
} finally {
  await bridge.dispose();
}
```

## Runtime model

`LlamaWebGpuBridge` is a facade over two execution modes:

- **Worker mode** is the default when a worker can be created. Model loading,
  generation, tokenization, embeddings, state APIs, and multimodal operations run
  in the worker runtime.
- **Direct runtime mode** is used when `disableWorker: true`, when a custom
  `coreModuleFactory` is supplied, or after recoverable worker setup/load errors.

Some APIs deliberately keep worker and direct runtime state separate. In
particular, worker WASMFS paths and direct-runtime WASMFS paths are not shared.
Use the byte state APIs when an application needs to persist data outside the
active bridge runtime.

Most methods require a model loaded by `loadModelFromUrl()` and reject with
`Error` when the model or runtime capability is unavailable.

### Native core aborts

The bridge retires a core after its native abort callback or a
`WebAssembly.RuntimeError`. It stops calling that core, ends its pthread
runtime when available, and releases its JS-backed staging data.

- In direct mode, the call fails with the original error and the model is
  unloaded. Calls that need a model reject with `ModelUnloadedError` (code
  `MODEL_UNLOADED_AFTER_CORE_ABORT`) until `loadModelFromUrl()` succeeds again.
  LoRA handles from the retired core are stale.
- In worker mode, the bridge replaces the worker and restores the remembered
  model on the next call that needs it. If a new load aborts, the previous
  model remains the one to restore. LoRA handles with a retained source remain
  valid. Media completion and speech synthesis on a GPU-loaded model retry
  once on the replacement worker with a CPU-only restore.

Decision heads stay tied to their original core and report `head-lost` after
retirement. State file paths also belong to that core. Applications can keep
serialized state bytes outside it.

Runtime notes include `core_retired`, `core_retired_threads:<n>` when a
pthread pool was stopped, and `core_retired_staged_bytes:<n>` for released
staging data. Older cores without a retirement hook report
`core_retired_threads_unavailable`. These notes appear only in direct mode.
Worker mode never shows them: the replacement worker starts with no notes.

## Operation serialization

Runtime-backed methods execute through a per-instance single-writer FIFO queue in
both worker and direct runtime modes:

`loadModelFromUrl`, `loadMultimodalProjector`, `unloadMultimodalProjector`,
`loadDraftModel`, `unloadDraftModel`, `getTextToSpeechCapabilities`,
`synthesizeSpeech`, `getDecisionCapabilities`, `loadDecisionHead`,
`runDecision`, `freeDecisionHead`, `getLoraAdapterCapabilities`,
`loadLoraAdapter`, `setLoraAdapter`, `removeLoraAdapter`, `clearLoraAdapters`,
`createCompletion`, `getCompletionCapabilities`, `tokenize`, `detokenize`,
`stateSaveFile`, `stateLoadFile`, `stateSaveBytes`,
`stateLoadBytes`, `getEmbeddingCapabilities`, `embed`, `embedBatch`,
`scoreNextToken`, `applyChatTemplate`.

Overlapping calls wait their turn and run in call order. A failing operation
releases the queue for the calls behind it, and separate bridge instances never
block each other.

These methods are deliberately **not** queued:

| Method | Observable guarantee |
| --- | --- |
| `cancel()` | Out-of-band control signal that reaches the running operation immediately, scoped to the worker generation and runtime that operation captured. |
| `setLogLevel(level)` | Synchronous, best-effort control signal. Finite numeric values are truncated and clamped to `0..4`. Delivery failures are absorbed without switching execution modes. After disposal, it throws `Bridge has been disposed.` |
| `prefetchModelToCache()`, `evictModelFromCache()`, `isModelCached()` | Cache and network operations that do not enter the model runtime. |
| `getModelMetadata()`, `getContextSize()`, `isGpuActive()`, `getBackendName()`, `supportsVision()`, `supportsAudio()` | Synchronous accessors served from the current bridge snapshot. |

### Cancellation while queued

`loadModelFromUrl()`, `loadDraftModel()`, `loadLoraAdapter()`,
`createCompletion()`, and `synthesizeSpeech()` accept a `signal`. An already
aborted signal or an abort while the call waits in the queue rejects the call
with an `AbortError` `DOMException`. The bridge skips the call's slot when its
turn arrives, leaving the active operation unaffected.

Once an operation has started, cancellation is scoped to that operation.
Aborting its `signal`, or calling `cancel()` while it owns the runtime, reaches
the worker generation and runtime the operation captured when it started, so a
cancel issued after an internal worker restart or main-thread fallback still
targets the owner of the request being cancelled rather than the replacement. A
queued successor is unaffected and dispatches normally once the cancelled
operation reaches a terminal state.

Cancellation remains cooperative, not preemptive. A direct model transfer is
checked before each stream read/write and immediately before native model-load
entry; an abort that arrives before that entry skips the native call and removes
partial files. Once an unpreemptible native call has started, cancellation waits
for it to return and then discards and cleans up its result.

### Disposal

`dispose()` marks the bridge disposed synchronously and then tears down on the
queue. Calls made after disposal, and calls still waiting in the queue, reject
with `Bridge has been disposed.` The operation that was already running is not
interrupted by disposal — it settles first, and teardown follows. `dispose()` is
idempotent and repeated calls return the identical promise object.

## Constructor

```ts
new LlamaWebGpuBridge(config?: LlamaWebGpuBridgeConfig)
```

Creates a bridge instance. The constructor may start a worker proxy immediately,
but the llama.cpp core is initialized lazily by the first operation that needs
it.

### `LlamaWebGpuBridgeConfig`

| Option | Type | Description |
| --- | --- | --- |
| `assetBaseUrl` | `string` | Directory for omitted bridge worker, core JS and Wasm URLs. Relative values resolve against the document base or the calling worker's URL. The trailing slash is optional. Explicit per-file URLs take precedence. |
| `coreModuleUrl` | `string` | URL for the wasm32 Emscripten JS loader, usually `llama_webgpu_core.js`. |
| `coreModuleUrlMem64` | `string` | URL for the optional wasm64 Emscripten JS loader. |
| `wasmUrl` | `string` | URL for the wasm32 `llama_webgpu_core.wasm` binary. |
| `wasmUrlMem64` | `string` | URL for the optional wasm64 wasm binary. |
| `coreModuleFactory` | `LlamaCoreModuleFactory \| Promise<LlamaCoreModuleFactory>` | Preloaded Emscripten factory. Supplying this disables worker mode because factories cannot be transferred to the worker. It also skips the default isolation check. A core that starts a pthread pool can hit `coreInitTimeoutMs` if the page cannot share memory with workers. See [the custom core contract](#custom-core-factory). A core instance holds one loaded model. Bridges that share the same core instance must not hold models at the same time. A load on either bridge releases the model in that core. Return a separate core instance for each bridge that needs an independent model. |
| `workerUrl` | `string` | URL for `llama_webgpu_bridge_worker.js`. Defaults to a sibling of the bridge module when possible. |
| `disableWorker` | `boolean` | Force direct runtime mode. |
| `allowSingleThreadCore` | `boolean` | Set to `true` to select the single-thread defaults on pages without cross-origin isolation. When supplying core URLs, point them at single-thread cores for those pages. Single-thread cores use one CPU thread and report `core_single_thread`. A core that starts a pthread pool can still hit `Bridge core init timeout` if the page cannot share memory with workers. See [single-thread setup and timeout behavior](single_thread.md). |
| `jsBackedModelStaging` | `boolean` | Unset enables JS-backed staging on WebKit when the core supports it. `false` opts out on any browser. `true` requires JS-backed staging support. JS-backed staging places models and projectors in a private per-operation `/jsmodels-<n>` mount. A supported core stages a single model file of known decoded size in one memory-mapped wasm heap block when the effective GPU layer count after the Safari cap is 0 or 1. Other JS-backed loads use 64 MiB JS `Uint8Array` chunks. |
| `preferMemory64` | `boolean` | Prefer the wasm64 core when available. Defaults to `true`; the runtime can still fall back to wasm32. |
| `workerRequestTimeoutMs` | `number` | Default timeout for worker RPC calls, 2 minutes unless a method has its own default. It also sets the `embed`/`embedBatch` budget, which defaults to 10 minutes when an input has image or audio parts. |
| `workerInitTimeoutMs` | `number` | Timeout for worker initialization. |
| `workerModelLoadTimeoutMs` | `number` | Worker `loadModelFromUrl()` and `loadDraftModel()` request timeout. Defaults to 3 minutes. |
| `workerMmprojLoadTimeoutMs` | `number` | Worker `loadMultimodalProjector()` request timeout. Defaults to 8 minutes. |
| `workerCompletionTimeoutMs` | `number` | Worker `createCompletion()` request timeout. Defaults to 6 minutes. |
| `workerTextToSpeechTimeoutMs` | `number` | Timeout for a worker text-to-speech request. Defaults to 20 minutes and is refreshed by progress events. |
| `workerGenerationStallTimeoutMs` | `number` | Worker generation timeout without token events. Explicit positive values are clamped to 5 seconds–5 minutes. Without an override, the timeout is 90 seconds for text, 180 seconds for GPU media, and disabled for CPU media. |
| `coreInitTimeoutMs` | `number` | Timeout while initializing the Emscripten core module. |
| `cacheName` | `string` | Cache Storage name used by model prefetch/load helpers. Defaults to `llamadart-webgpu-model-cache-v1`. |
| `threads` | `number` | Requested llama.cpp thread count. The bridge caps this to the compiled pthread pool, and to one thread for a single-thread core on a page that is not isolated. |
| `threadsBatch` | `number` | Requested batch thread count; defaults to `threads`. |
| `threadPoolSize` | `number` | Hint for the compiled pthread pool size, used before the core can report it. |
| `nBatch` | `number` | llama.cpp batch size override. |
| `nUbatch` | `number` | llama.cpp micro-batch size override. |
| `nGpuLayers` | `number` | Default GPU layer request. `0` forces CPU/WASM execution. Omitting the count or passing a negative value requests full offload, subject to the Safari cap. An effective count of `0` or `1` enables heap staging for eligible JS-backed loads, including the WebKit default. |
| `userAgent` | `string` | User-agent override for automatic WebKit staging and the Safari GPU layer cap. The Safari version check requires an Apple platform and Safari's product tokens. |
| `remoteFetchThresholdBytes` | `number` | Size threshold for trying the native remote-fetch backend for large single-file models. |
| `remoteFetchChunkBytes` | `number` | Chunk size used by remote-fetch streaming paths. |
| `mediaMaxImagePixels` | `number` | Maximum image pixel count before multimodal image downscaling. |
| `mediaMaxImageEdge` | `number` | Maximum image width/height before multimodal image downscaling. |
| `disableImageDownscale` | `boolean` | Disable bridge-side image downscaling for multimodal image parts. |
| `allowAutoRemoteFetchBackend` | `boolean` | Allow automatic selection of the native remote-fetch backend in a pthread worker when a single-file model qualifies. A single-thread worker uses streamed staging and reports `model_fetch_backend_requires_pthreads`. |
| `logLevel` | `string \| number` | Bridge/core logging level. Numeric values are clamped to the supported core range. |

Unknown config keys are accepted and may be consumed by current or future bridge
internals.

Worker timeout options are forwarded to the proxy, and `coreInitTimeoutMs` also
reaches the runtime inside the worker. Positive request budgets are truncated
and clamped to 5 seconds–1 hour; per-method settings override the global request
budget. Non-positive or non-finite settings retain the existing fallback
budgets. Worker initialization uses a 3-second–2-minute clamp (20-second
default), and core initialization uses 10 seconds–10 minutes (90-second default).
`embedBatch` restarts its budget after each finished input, so the budget bounds
one input rather than the batch.

When a worker request times out, crashes or aborts the core, most operations
move to the main-thread runtime and run the request again there; CPU-mode media
completions reject instead. A fallback logs a `falling back to main thread`
warning at the `warn` log level and sets `llamadart.webgpu.execution` to
`main-thread` and `llamadart.webgpu.worker_fallback_reason` in
`getModelMetadata()`. A result returned after a fallback did not run in the
worker.
Firefox falls back to the page thread when a CDN bridge worker fails to start.
With cross-origin isolation, a supported CDN pthread core can still run there.

### Custom core factory

Return an Emscripten module with writable or configurable `ccall`, `cwrap`,
`FS`, and filesystem method properties. The bridge wraps these properties to
prevent further native calls once the core retires.

A core instantiated in another JavaScript realm must expose that realm's
genuine `WebAssembly.RuntimeError` constructor as
`Module.llamadartWasmRuntimeError` for bare traps to retire the core. The
bridge trusts the supplied constructor. Built cores expose it. An older
cross-realm core without this export still retires through `onAbort` and
rethrows a bare trap with the core left in use.

Wrappers loaded through `coreModuleUrl` or `coreModuleUrlMem64` can use
cross-origin pthread startup by forwarding the factory options, including the
symbol-keyed hook, to a core built with support for it. Older cores, supplied
`coreModuleFactory` factories, and wrappers that drop the hook use the
core's original worker startup.

### Static properties

#### `LlamaWebGpuBridge.supportsSafariAdaptiveGpu`

`boolean` flag indicating that the bridge supports Safari's load-time GPU probe
and can retry with one GPU layer or on the CPU. Safari versions older than 27
and unidentified Safari versions default to a one-layer cap.

#### `LlamaWebGpuBridge.supportsCompletionUsage`

`true` when `createCompletion` accepts `onUsage`. Bridges before v0.1.54 lack
the flag; through v0.1.52 their worker mode rejects a function option with a
`DataCloneError`. Check it before passing `onUsage`.

#### `LlamaWebGpuBridge.supportsModelCacheQuery`

`true` when the bridge exposes `isModelCached()` and accepts `useCache` in
`loadMultimodalProjector()`. Earlier bridges lack the flag.

## Model loading and cache helpers

When text generation on wasm64/WebGPU qualifies for CPU recovery, the reload
preserves the model's resolved native load settings, including sequence count,
batch sizes, memory mapping/locking, KV cache types, flash attention, RoPE and
device split settings. It changes GPU placement to CPU and retains the existing
memory32 preference and transfer safeguards for recovery. Thread counts remain
subject to the replacement core's pool. Recovery
continues to refuse active LoRA adapters and speculative generation.

### `loadModelFromUrl(url, options?)`

```ts
loadModelFromUrl(url: string | string[], options?: LoadModelOptions): Promise<unknown>
```

Loads a GGUF model from one URL, an explicit shard URL array, or an auto-expanded
split GGUF URL such as `model-00001-of-00002.gguf`. The bridge streams model
bytes into the active WASM filesystem unless the native remote-fetch backend is
used for a qualifying large single-file model.
For full wasm32 responses with a valid `Content-Length` hint between 1 GiB
and 2 GiB minus one byte, the writer attempts to reserve filesystem capacity before streaming. It resets
the logical file length before writing and falls back to incremental writes if
the reservation fails. A CORS response can hide compression headers, so its
length is only an allocation hint when the encoding is unknown. Visibly
encoded, resumed and unknown-size responses keep incremental writes.

For an interrupted streamed download, the bridge validates an open-ended
range response using a matching strong ETag, the same final URL, and a
consistent identity-encoded `Content-Range`. When `Content-Range` is hidden,
a matching identity-encoded `Content-Length` for a known remainder also
verifies the range. A changed URL, differing comparable strong ETags, or
contradictory byte lengths or offsets restart the shard at byte zero.
The bridge remembers the first comparable strong ETag for each encoding
across interruptions. A tag first seen on a range can detect a later change
for that encoding while the original prefix stays unvalidated.
Missing or weak ETags and lengths measured with different or unknown
encodings can leave a resume unvalidated. A same-size revision change can
go undetected on this path.

The bridge checks the staged byte count against any known decoded size
before initializing the model. Earlier responses' declared ends still bound
the minimum size when a later range is shorter or omits size metadata. If an
interrupted range runs past its declared end, the bridge restarts the shard
from byte zero. An identity-encoded `206` response to a full GET that declares
a range starting at byte zero must match its declared span, including when
the total is `*`. If the response exposes numeric totals in both
`Content-Range` and `X-Linked-Size`, they must agree. For cross-origin
downloads, expose `ETag`, `Content-Range`, and any `Content-Encoding` with
`Access-Control-Expose-Headers`. When `Content-Encoding` is absent, the bridge
can establish identity encoding if it can read an exposure list that includes
`Content-Encoding` or `*`. If the encoding is unknown, HTTP length headers
are hints and the bridge resumes without range validation.

Relative URLs resolve against the page's base URL (`document.baseURI`, so a
`<base href>` applies) in both runtimes; worker mode resolves them before
posting them to the worker.

Common `options` keys:

| Option | Description |
| --- | --- |
| `progressCallback(progress)` | Receives aggregate `{ loaded, total }` events. Split models may also include shard progress metadata. `loaded` can decrease when a shard restarts from zero. |
| `signal` | `AbortSignal` used to cancel model transfer. |
| `nCtx` | Context size. Defaults to the bridge runtime default, initially 4096. |
| `nThreads`, `nThreadsBatch` | Per-load thread overrides, capped to the active runtime. |
| `nGpuLayers` | GPU layer request. Use `0` for CPU/WASM mode or a negative value to request full offload. With no configured or previously set count in the runtime, an omitted value requests full offload. Apple Safari applies the per-load cap to the request without changing the stored count. An effective count of `0` or `1` selects heap staging for eligible JS-backed model loads. Uncapped full offload uses JS chunks when JS-backed staging is enabled. |
| `useMmap` | Pass `false` to keep eligible JS-backed model loads on JS chunks. |
| `safariMaxGpuLayers` | Per-load Safari GPU layer cap. Apple Safari before 27 or with an unknown version defaults to one layer, including for full-offload requests. Safari 27 and newer default to no cap. On Apple Safari, finite values up to `2147483647` set a maximum (rounded down to an integer of at least `1`). Larger values, including `Infinity`, opt out. |
| `safariGpuProbe` | Set to `false` to skip the Safari load-time GPU probe. |
| `nBatch`, `nUbatch` | Per-load batch and micro-batch overrides. |
| `useCache`, `force` | Cache Storage controls for HTTP(S) model responses. A cache miss writes the model to Cache Storage during the download. If the cache write fails, the download can still finish and load the model. Cancellation during transfer discards the incomplete entry. WebKit stores 16 MiB parts and a manifest. A Web Lock allows one context to store a given model in a given cache at a time. When a load needs to download a model and the lock is busy, Web Locks are unavailable, or `crypto.randomUUID()` is unavailable, the load proceeds uncached. The cache layout is private on every engine. Use `isModelCached()` to check for a cached model. |
| `streamResumeRetries` | Maximum retry count per shard for resuming or restarting a streamed model download. |
| `remoteFetchThresholdBytes`, `remoteFetchChunkBytes` | Per-load remote-fetch tuning. |
| `forceRemoteFetchBackend` | Force the native fetch backend for a single-file model in a worker. On a single-thread worker, the load rejects with `SingleThreadCoreError` (`code: 'SINGLE_THREAD_CORE_UNSUPPORTED'`, `feature: 'remote_fetch_backend'`). Page-thread loads use streamed staging. |
| `loadMtp` | Loads the model's MTP layers, which `draft-mtp` speculative decoding needs. Defaults to `false`. |
| `speculativeRollbackTokenMax` | llama.cpp's `n_rs_seq`: rollback snapshots per sequence. A model with recurrent state, such as Qwen3.5, needs at least the draft length for `draft-*` speculative decoding. Defaults to `0`. |

The default Safari cap uses the major version from a standalone numeric
`Version/` product in the Apple Safari user agent. Safari-like user agents
that match the legacy detector continue to use its one-layer default cap and
completion probe. For a pooled-embedding model on recognized Apple Safari,
the probe runs two distinct prompts. If a prompt exceeds the effective
microbatch, the bridge tries a shorter pair. If that pair still exceeds the
microbatch, the bridge keeps the current GPU layers and records
`safari_gpu_probe_skipped_input_too_long`. Each embedding must match the native
output dimension, contain only finite values and have at least two components
with magnitude 0.001 or higher. Each L2 norm must differ from 1 by less than
0.01, and the cosine similarity must be below 0.999. Text models and older
cores without embedding capabilities use the completion probe. If the probe
fails, the bridge retries at one GPU layer when possible, then on the CPU.

Returns the underlying load result from the active runtime. After a successful
load, metadata and capability helpers reflect the loaded model.

Calling it again replaces the loaded model in the same runtime, in both direct
and worker modes. The bridge first frees the current model, its multimodal
projector and any remaining filesystem copies, then downloads the new model, so both
never occupy the WASM heap at once. If the download or native load then
fails, no model stays loaded; call `loadModelFromUrl()` again. The core refuses
the release while a generation or speech synthesis is active, and the current
model stays loaded. Load a projector, LoRA adapters and a draft model again
after switching models.

### `prefetchModelToCache(url, options?)`

```ts
prefetchModelToCache(url: string | string[], options?: LoadModelOptions): Promise<unknown>
```

Fetches model URL(s) into Cache Storage without loading them into llama.cpp.
This uses a direct runtime helper even when the main bridge instance is currently
worker-backed. Use it to warm the browser cache before calling
`loadModelFromUrl()`. After a successful cache-enabled HTTP(S) prefetch with
Cache Storage available, [`isModelCached()`](#ismodelcachedurl-options) reports
the committed record in the same cache. Other URL schemes load uncached.

If a WebKit prefetch needs to download a model and cannot acquire its cache
lock or generate a UUID, it downloads the body and then rejects with the
message `Failed to store prefetched model in browser cache.`

### `evictModelFromCache(url, options?)`

```ts
evictModelFromCache(url: string | string[], options?: Record<string, unknown>): Promise<unknown>
```

Removes one model URL, explicit shard array, or expanded split-model URL set from
Cache Storage, including both whole-body entries and WebKit part entries.
Eviction skips a model while an active store or cache cleanup holds its lock.
This includes cleanup in the same context immediately after cancellation.
Removing part entries requires Web Locks. Returns whether any requested model
or shard was removed. After successful eviction of an individual URL,
`isModelCached()` for that URL returns `false` unless another writer commits it.

### `isModelCached(url, options?)`

```ts
isModelCached(url: string | string[], options?: ModelCacheQueryOptions): Promise<boolean>
```

Resolves `true` when the bridge cache holds a committed record for every HTTP(S)
URL. A WebKit manifest or a whole-body entry counts on WebKit. On other engines,
only the whole-body entry counts. Pass `options.cacheName` to select a cache.
The default is the bridge's configured `cacheName`.

An explicit shard array or a split GGUF filename requires every shard. Relative
URLs resolve against the document base URL or the app-owned worker's script URL.
URL fragments are ignored. Empty input, missing entries, unavailable Cache
Storage, and lookup errors resolve `false`.

The query leaves Cache Storage and loaded-model metadata unchanged. It uses no
Web Lock and consumes no response bodies. It can run during another bridge
operation, including when the bridge is worker-backed. An in-progress part store
becomes visible after its manifest commits. An older committed manifest stays
visible during a refresh. The query does not verify individual parts. A later
cache-enabled model load that finds a missing or damaged part can recover from
the network.

## Text generation

### `createCompletion(prompt, options?)`

```ts
createCompletion(prompt: string, options?: CompletionOptions): Promise<string>
```

Runs llama.cpp generation for a loaded model and resolves to the final generated
text. The bridge cleans up active generation state before returning or throwing.

Common `options` keys:

| Option | Description |
| --- | --- |
| `nPredict` | Maximum number of generated tokens. Defaults to `256`. |
| `temp` | Sampling temperature. Defaults to `0.8`. |
| `topK` | Top-k sampling. Defaults to `40`. |
| `topP` | Top-p sampling. Defaults to `0.95`. |
| `minP` | Min-P threshold from `0` to `1`, applied after top-p. Defaults to `0`, which disables it. |
| `penalty` | Repetition penalty over the last 64 tokens of the prompt and this completion, as in native llamadart: the prompt counts only for a text prompt without `penaltyPromptTokens: false` and either no `grammar` string or an explicit `penaltyPromptTokensWithGrammar: true`. Defaults to `1.1`; `1` disables it. |
| `presencePenalty` | Subtracted once from the logit of each token in the same 64-token window as `penalty`. Defaults to `0`, which disables it. |
| `penaltyPromptTokens` | `false` keeps the prompt out of the `penalty` and `presencePenalty` window, as a `grammar` does. It exists for callers that drop a grammar native llamadart keeps: llamadart drops a lazy tool-call grammar on a core without `lazyGrammar` and sets this so the output matches native, which does not count the prompt when a lazy grammar is set. Omitted or `null` means `true`; any other non-boolean rejects with a `TypeError`. A core whose `getCompletionCapabilities()` lacks `penaltyPromptTokens` never counts the prompt. |
| `penaltyPromptTokensWithGrammar` | Explicitly counts a text prompt in the penalties window while a grammar string is attached. Only the penalties sampler receives prompt tokens; prompt trigger text cannot activate grammar or advance the reasoning sampler. Omitted, `null`, or `false` preserves grammar defaults. Requires the loaded core's same-name capability to be literal `true`; a missing/false probe rejects before generation. Missing grammar, media parts, and `penaltyPromptTokens: false` reject. |
| `grammar` | Optional llama.cpp GBNF grammar with a `root` rule, or the `grammarRoot` rule. An invalid grammar rejects before generation starts with an error containing `(invalid grammar)`; worker mode rethrows it without falling back to the main thread. Each rejection leaks the partly parsed grammar (a few KiB at most for typical grammars), so validate generated grammars before sending them in a loop. |
| `grammarRoot` | Rule the `grammar` starts from. Defaults to `root`; a grammar without that rule rejects with `(invalid grammar)`. |
| `grammarLazy` | `true` applies `grammar` only after a `grammarTriggers` entry fires, as llama.cpp's lazy grammar (`llama-server`'s `grammar_lazy`) does; see [Lazy grammars](#lazy-grammars). Needs a non-empty `grammar` and at least one trigger. Omitted or `null` means `false`. |
| `grammarTriggers` | Array of `{ type: 'word' \| 'pattern' \| 'pattern-full', value }` or `{ type: 'token', token }` that fire a lazy grammar. Always validated; applied only when `grammarLazy` is `true`. |
| `seed` | Integer seed; random when omitted. |
| `thinkingBudget` | `{ maxTokens, startTag, endTag, forcedMessage?, pauseGrammar? }`: llama.cpp's reasoning budget. After `maxTokens` tokens inside a block opened by `startTag`, plus any that finish a UTF-8 character, the sampler forces `forcedMessage` (default empty) and `endTag`. A block the prompt leaves open counts from the start of generation, so `maxTokens: 0` forces the end at once. Each new `startTag` gets a fresh budget. A grammar constrains `startTag` like any other text, then pauses until a natural end or the forced close completes, unless `pauseGrammar` is `false`: that keeps the grammar applied inside the block, for one that takes the block and `endTag` itself (such as one written for a prompt that ends inside the block; paused, it resumes at its start and takes the reply as reasoning) and so accepts `forcedMessage` and `endTag` wherever the budget can run out; a kept grammar that rejects the forced end tag fails the generation with `Grammar rejected every candidate token`. Omitted, `null` or `true` pauses it; `false` needs the `thinkingBudgetPauseGrammar` capability. `maxTokens` is an integer from `0` to `2147483647`; the tags must be non-empty. Each tag is tokenized on its own, as native llamadart does: with a vocabulary that adds a leading space to text, such as Llama 2's SentencePiece, a tag in the prompt matches only after a space, and the forced end tag starts with one. Text-only prompts: with `parts` it rejects. |
| `onUsage(usage)` | Called at most once, with the `CompletionUsage` of the last attempt whose generation returned, before the promise resolves or rejects with an `AbortError`. Not called when no generation returned: other rejections, a skipped multimodal warmup, or a worker failure after a cancel. |
| `onToken(piece, currentText)` | Token callback. By default `piece` is a `Uint8Array` containing complete UTF-8 characters; see [Streamed text](#streamed-text). `currentText` is the text streamed so far. Direct runtime mode provides it by default; worker mode provides `''` unless `emitCurrentTextOnToken: true` is set. |
| `signal` | `AbortSignal`; aborting cancels this operation and rejects with `AbortError`. |
| `warmup` | Marks a warmup generation. Some multimodal worker setup failures return an empty string instead of failing warmup. |
| `emitCurrentTextOnToken` | Direct runtime defaults to current text and uses `null` when set to `false`; worker mode sends current text only when this is `true` and otherwise sends `''`. |
| `tokenEventEncoding` | `'bytes'` (default) sends `Uint8Array` pieces; `'text'` sends string pieces. Worker events may already provide text pieces. |
| `tokenEventFlushMs` | Worker mode coalesces token events for up to this many milliseconds, for both byte and text encoding. `0` (default) disables coalescing. Values are clamped to `0..200`. |
| `tokenEventFlushChars` | When worker coalescing is enabled, flush once the buffered decoded text reaches this many JavaScript characters. `0` (default) disables the size threshold. Values are clamped to `1..1024` when positive. |
| `parts` | Optional multimodal parts after a projector is loaded. Image parts require `{ type: 'image', bytes }` or `{ type: 'image', url }`; audio parts require `{ type: 'audio', samples }`, `{ type: 'audio', bytes }`, or `{ type: 'audio', url }`. `bytes` and `samples` accept an `ArrayBuffer`, any `ArrayBuffer` view, or a number array. Image `width` and `height` identify raw RGB bytes when their dimensions match the byte length. |
| `mediaMaxPredict` | Cap for multimodal generation token count. |
| `speculativeDecoding` | `{ strategies, ... }`: llama.cpp speculative decoding; see [Speculative decoding](#speculative-decoding). |

`thinkingBudget.naturalEndTags` is the complete ordered list of natural ends,
including any normal close. Supply a non-empty array to use this option, which
requires `thinkingBudgetMultiEnd`. By default only `endTag` closes the block.
The forced close still uses `forcedMessage` followed by `endTag`.
`thinkingBudget.grammarReplayEndIndex` selects the natural end that a paused
tool grammar must accept when the reasoning block closes. Supply an index in
`naturalEndTags`. Omit the index or set it to `-1` to disable replay.

Call `cancel()` or abort the supplied signal to request a best-effort stop.

#### Streamed text

The bridge decodes the generated bytes itself, so the pieces always add up to
the returned text:

- A character split across tokens is held back until it completes, and
  `currentText` never includes that incomplete tail.
- Bytes that can no longer form a character stream as U+FFFD at once. An
  incomplete character left when generation stops, is cancelled or reaches
  `nPredict` is delivered as U+FFFD in the last piece.
- A leading U+FEFF is kept, and output after a NUL byte is delivered.
- After `cancel()`, pieces generated before the runtime reads it still stream
  until the call returns, in worker mode too. An aborted `signal` stops worker
  pieces at once and rejects with an `AbortError`.

Cores built before this behavior lack `llamadart_webgpu_last_output_bytes_json`.
With them the bridge keeps its previous behavior: it reads the output as a C
string, which stops at a NUL byte and can decode a split character in the first
16 bytes as a wrong one.

An invalid `minP`, a `presencePenalty` that is not finite as a 32-bit float,
an invalid `thinkingBudget`, or a malformed `grammarLazy`, `grammarTriggers` or
`grammarRoot` rejects with a `RangeError` or `TypeError` before generation
starts, and `grammarLazy: true` or another `grammarRoot` without a non-empty
`grammar`, or a lazy grammar without triggers, with an `Error`. A nonzero
`minP` or `presencePenalty`, any `thinkingBudget`,
`thinkingBudget.pauseGrammar: false`, `thinkingBudget.naturalEndTags`,
`grammarLazy: true`, and a `grammarRoot` other than `root` each reject when the
loaded core lacks the corresponding capability.

#### Lazy grammars

With `grammarLazy: true` the output is unconstrained until a trigger fires, then
`grammar` constrains it from where the trigger starts, so the grammar must
accept the trigger text. This is llama.cpp's
`llama_sampler_init_grammar_lazy_patterns`, with triggers converted as
`common_sampler_init` does:

| `type` | Fires when |
| --- | --- |
| `word` | `value` appears in the output. |
| `pattern` | The output contains a match of `value`, a C++ `std::regex` ECMAScript expression: no lookbehind, named groups or flags. The grammar starts at its first non-empty capture group, else at the match. |
| `pattern-full` | The whole output so far matches `value`, anchored with `^` and `$`; the grammar starts as for `pattern`. |
| `token` | The model samples token id `token`, an integer from `0` to `2147483647`. |

```js
await bridge.createCompletion(prompt, {
  grammar: 'root ::= "<tool_call>" call "</tool_call>"\ncall ::= ...',
  grammarLazy: true,
  grammarTriggers: [{ type: 'word', value: '<tool_call>' }],
});
```

A `value` must be a non-empty string without U+0000. A pattern that is not a
valid regular expression or a token outside the vocabulary rejects before
generation starts with an error containing `(invalid grammar)`, as an invalid
grammar does. When a trigger fires but the grammar rejects the trigger text, the
completion rejects with `The lazy grammar rejected the text that triggered it`.
The core stays usable after either, and worker mode rethrows both without
falling back to the main thread; like an invalid grammar, each rejection leaks
a few KiB at most. Triggers match only generated text, not the
prompt, and a `thinkingBudget` block pauses a lazy grammar (unless `pauseGrammar` is `false`), so text inside the
block fires no trigger. The bridge streams every special token's text, so
llama.cpp's `preserved_tokens` has no equivalent here.

`CompletionUsage` fields:

| Field | Description |
| --- | --- |
| `promptTokens` | Prompt tokens in the context when generation started, including `cachedPromptTokens`. For a multimodal prompt, the context positions it filled. |
| `cachedPromptTokens` | Leading prompt tokens kept from the previous prompt or a loaded state instead of being evaluated again; `0` for a multimodal prompt. |
| `completionTokens` | Generated tokens in the returned text. The end-of-generation token is not counted. |
| `timeToFirstTokenMs` | Milliseconds from the runtime starting the completion to its first streamed text, or `null` when it streamed none. |
| `durationMs` | Milliseconds from the runtime starting the completion to its end. |
| `finishReason` | `stop` at an end-of-generation token, `length` at `nPredict` or the context limit, `cancelled` after `cancel()` or an abort during generation. In worker mode an abort that reaches the worker only after generation ended still rejects with an `AbortError`, but the usage keeps that generation's `stop` or `length`. |
| `speculative` | Present only for a `speculativeDecoding` completion: `{ draftTokens, acceptedDraftTokens, draftAttempts, verifyTokens, replayTokens }`, the drafted and accepted tokens, draft steps (including those that drafted nothing), target tokens decoded to verify drafts, and target tokens decoded again after restoring a recurrent-state checkpoint. |

Both times are measured by the runtime that generates, inside the worker in
worker mode. They include staging media and evaluating the prompt, and exclude
time queued behind another operation. After an internal retry or main-thread
fallback, the usage is that of the last attempt whose generation returned. In
worker mode an abort during generation rejects the promise, while the direct
runtime resolves it with the partial text; in both, `onUsage` reports the
cancelled generation once it returns.

### `getCompletionCapabilities()`

```ts
getCompletionCapabilities(): Promise<CompletionCapabilities>
```

Returns `{ presencePenalty, minP, thinkingBudget, thinkingBudgetPauseGrammar,
thinkingBudgetMultiEnd, penaltyPromptTokens, penaltyPromptTokensWithGrammar,
lazyGrammar, speculativeDecoding }`. The flags indicate whether the loaded core
applies each `createCompletion` option. `thinkingBudgetMultiEnd` covers ordered
natural ends and the complete forced close. `lazyGrammar`
covers `grammarLazy`, `grammarTriggers` and `grammarRoot`.
`penaltyPromptTokensWithGrammar` separately permits the explicit grammar opt-in;
interface presence and an asset tag do not prove it.
`penaltyPromptTokens` also means the penalties count the prompt; without it they
never do. Every flag is
`false` until a model load initializes the core. `speculativeDecoding` maps each
strategy to whether the loaded models can run it now: the n-gram strategies need
only a speculative-capable core, `draft-mtp` needs a model with MTP layers
loaded with `loadMtp: true`, and the other `draft-*` strategies need a matching
draft from `loadDraftModel()`.

### Speculative decoding

`speculativeDecoding` drafts tokens and verifies them with the loaded model, as
llama.cpp's `llama-server` does. Every sampled token comes from the target
model, so greedy sampling (`temp: 0` or `topK: 1`) produces the same tokens as a
completion without it. Strategies use llama.cpp's `--spec-type` names:

| Strategy | Drafts from |
| --- | --- |
| `ngram-simple`, `ngram-map-k`, `ngram-map-k4v` | Matches of the last `ngramSizeN` tokens in the prompt and output. |
| `ngram-mod` | A hashed pool of earlier n-grams. |
| `ngram-cache` | Prompt and output n-grams, optionally with `ngramCacheStatic`/`ngramCacheDynamic` files from `llama-lookup-create`. |
| `draft-simple` | A standalone draft model from `loadDraftModel()` that shares the target's vocabulary. |
| `draft-eagle3` | An EAGLE3 draft from `loadDraftModel()`. |
| `draft-dflash`, `draft-dspark` | A DFlash-architecture block draft from `loadDraftModel()`; llama.cpp reads which of the two it is from its tensors. |
| `draft-mtp` | The target's own MTP layers; load the model with `loadMtp: true`. |

`strategies` combines any n-gram strategies with at most one `draft-*`
strategy. The other keys are optional; omitted ones keep llama.cpp's defaults:

| Key | Description |
| --- | --- |
| `draftTokenMax` | Maximum draft tokens per step, an integer from `0`. `draft-*` strategies use it or 3, `ngram-simple`/`ngram-map-k`/`ngram-map-k4v` use `ngramSizeM` or 48, `ngram-mod` uses `ngramTokenMax`, then it, then 64, and `ngram-cache` uses it or 8. The largest applies; 0 becomes 64. The resolved value must not exceed the context size. |
| `draftTokenMin`, `minProbability`, `draftSplitProbability` | Draft-model minimum length, minimum token probability and split probability. They need a `draft-*` strategy. |
| `ngramSizeN`, `ngramSizeM`, `ngramMinHits` | Lookup size, draft size and minimum hits of `ngram-simple`, `ngram-map-k` and `ngram-map-k4v`, from `1` to `65535`. |
| `ngramMatch`, `ngramTokenMin`, `ngramTokenMax` | Lookup length (`1` to `65535`) and draft length bounds of `ngram-mod`. `ngramTokenMax` defaults to `draftTokenMax` and must not exceed the context size. |
| `ngramCacheStatic`, `ngramCacheDynamic` | `ngram-cache` files as a URL string, `ArrayBuffer` or typed array. The dynamic cache is read, never written back. A malformed file rejects. |

The rules are those of native llamadart's `SpeculativeDecodingConfig`, and an
invalid combination rejects with a `TypeError`, `RangeError` or `Error` before
any work. Speculative decoding needs a text-only prompt and rejects with
`grammar`, `thinkingBudget` or `parts`. It evaluates the whole prompt instead of
reusing a cached prefix, and each step drafts at most `nPredict` minus the
tokens generated so far, minus one.

A model with recurrent state cannot drop rejected draft tokens from its memory.
With the n-gram strategies the bridge saves that state before each verification
and replays the accepted tokens after a rejection (`replayTokens`). The
`draft-*` strategies instead need the model loaded with
`speculativeRollbackTokenMax` of at least the draft length, and each draft is
capped at that value. A request the loaded models cannot run, such as
`draft-mtp` without MTP layers or `draft-dflash` with a DSpark draft, rejects
with a message naming what is missing.

The draft context inherits the target's `nBatch` and `nUbatch`. When its
buffers do not fit, generation rejects, the partial context is freed and the
target stays usable; reload the model with a smaller `nBatch` and `nUbatch`, or
use the memory64 core on wasm32.

## Draft models

### `loadDraftModel(url, options?)`

```ts
loadDraftModel(url: string, options?: DraftModelLoadOptions): Promise<DraftModelInfo>
```

Downloads a draft GGUF and loads it with the target model's load settings,
replacing any draft. `options` takes `progressCallback`, `signal`, `useCache`
and `force`, as `loadModelFromUrl()` does. Resolves to `{ architecture,
strategy? }`: the GGUF's `general.architecture`, and `draft-dflash` or
`draft-dspark` for a DFlash-architecture draft. An EAGLE3 or DFlash draft built
for a different hidden size than the loaded model's rejects, naming both sizes.
A target model load unloads the
draft; the bridge reloads it after it reloads the target on its own, such as
after a worker restart, and warns and forgets it when that reload fails.

The draft is staged in the WASM heap before it loads. When the server reports
the draft's size and it would not fit in the memory left, the call rejects
before reading the body; on the wasm32 core, whose heap is limited to 4 GiB, the
message says a memory64 core is required.

When a cached WebKit part is missing or damaged, the draft load rejects. In
direct mode the error's `name` is `ModelCacheEntryError`. In worker mode the
call rejects with an `Error` carrying the same message. The bridge removes the
damaged entry for the next call when it acquires the cache lock and the entry
is still current.

### `unloadDraftModel()`

```ts
unloadDraftModel(): Promise<void>
```

Frees the draft model. `draft-simple`, `draft-eagle3`, `draft-dflash` and
`draft-dspark` then reject until a draft is loaded again.

## Tokenization and chat templates

### `tokenize(text, addSpecial?)`

```ts
tokenize(text: string, addSpecial = true): Promise<number[]>
```

Tokenizes text with the loaded model. `addSpecial` controls whether llama.cpp
adds model-specific special tokens. Text after U+0000 is kept; see
[Text with U+0000](#text-with-u0000).

### Text with U+0000

`tokenize`, string inputs to `embed` and `embedBatch`, and the `scoreNextToken`
and `createCompletion` prompts keep the text after U+0000: the bridge passes the
core UTF-8 bytes and their length, not a C string. A lone surrogate becomes
U+FFFD.

- `grammar`, `grammarRoot`, string values in `grammarTriggers`, the tags in
  `thinkingBudget` (including `naturalEndTags`), `thinkingBudget.forcedMessage`,
  and `synthesizeSpeech` options `text` and `language` reject U+0000 with a
  `TypeError` before the core runs.
- Decision head load errors name a URL head whose file name holds U+0000 with
  `%00` in its place, as a parsed URL path does.
- Cores built before this behavior lack `llamadart_webgpu_tokenize_bytes_to_json`
  and the other `*_bytes` text exports. With them the bridge keeps the C-string
  call for text without U+0000 and rejects text that holds one.

### `detokenize(tokens, special?)`

```ts
detokenize(tokens: number[] | ArrayLike<number>, special = false): Promise<string>
```

Converts token IDs back to text. Non-array inputs are converted with
`Array.from()`. Bytes that do not form a complete UTF-8 character, such as a
byte-fallback token without the rest of its character, become U+FFFD; a
leading U+FEFF is kept. Cores without
`llamadart_webgpu_last_detokenized_bytes_json` keep the previous C-string read,
which stops at a NUL byte and can return a wrong character for a result of 16
bytes or fewer.

### `applyChatTemplate(messages, addAssistant?, customTemplate?)`

```ts
applyChatTemplate(
  messages: Array<Record<string, unknown>>,
  addAssistant = true,
  customTemplate: string | null = null,
): Promise<string>
```

Builds a prompt from chat messages. The current JavaScript bridge uses its
built-in fallback formatter and ignores `customTemplate`; higher-level SDKs may
supply their own template logic before calling `createCompletion()`.

## State persistence

The bridge exposes llama.cpp state/session save and load helpers after a model is
loaded. State snapshots are tied to the same model, llama.cpp build, and
compatible load parameters.

The `tokens` argument records the already-evaluated prompt/prefix token list in
the state file; it does not cause the bridge to evaluate those tokens. Save only
after the prompt or prefix you want to restore has already been evaluated.

### `stateSaveFile(path, tokens?)`

```ts
stateSaveFile(path: string, tokens?: number[] | ArrayLike<number>): Promise<true>
```

Saves the current llama.cpp session to a path inside the active runtime's WASMFS
and stores the supplied token list in the session metadata. The method returns
`true` on success. A path the runtime cannot create, such as one in a missing
directory, rejects with `Cannot open state file <path>: <reason>`.

### `stateLoadFile(path, tokenCapacity?)`

```ts
stateLoadFile(path: string, tokenCapacity = bridge.getContextSize()): Promise<{ tokens: number[] }>
```

Loads a state/session file from the active runtime's WASMFS. A `tokenCapacity`
whose numeric conversion is greater than zero is truncated and used; all other
values fall back to the active context size. The resolved capacity must be at
least large enough for the stored token list and no larger than the active
context size. The resolved `{ tokens }` value is the token list stored at save
time. A missing or unreadable path rejects with `Cannot open state file <path>:
<reason>` and keeps the current context state. A truncated snapshot, or one
llama.cpp rejects (another format version, model architecture or cache
layout), rejects with a message ending in `The current context state was
kept.`, and the context and its prompt cache are unchanged. Keeping them needs
memory for a copy of the current state; without it, a snapshot llama.cpp
rejects clears them instead, with `The current context state could not be
kept`. Snapshots carry no checksum, so corrupt bytes inside a well-formed
snapshot are not detected. WASMFS paths do not survive a page reload; use
`stateSaveBytes()` and `stateLoadBytes()` for durable state.

### `stateSaveBytes(tokens?)`

```ts
stateSaveBytes(tokens?: number[] | ArrayLike<number>): Promise<Uint8Array>
```

Saves the current state to a temporary runtime file and returns its bytes. This
is the preferred API for durable browser storage because the application can
then store the bytes in IndexedDB, OPFS, Cache API, or another app-managed store.

### `stateLoadBytes(bytes, tokenCapacity?)`

```ts
stateLoadBytes(
  bytes: Uint8Array | ArrayBuffer | ArrayLike<number>,
  tokenCapacity = bridge.getContextSize(),
): Promise<{ tokens: number[] }>
```

Loads state from bytes by staging them into a temporary runtime file. It uses the
same `tokenCapacity` resolution and snapshot rejection as `stateLoadFile()`. Empty input is rejected.
Worker mode transfers an internal copy to the worker, so caller-owned
`ArrayBuffer` or `Uint8Array` inputs are not detached.

## Embeddings

### `getEmbeddingCapabilities()`

```ts
getEmbeddingCapabilities(): Promise<{
  apiVersion: 1; text: boolean; image: boolean; audio: boolean;
}>
```

Probes the loaded model, allocated context, projector and mixed-batch API in
the active runtime. An unloaded model returns all false. Probe again after
projector load/unload; completion vision/audio support alone does not enable
media embeddings. Pooled embedding models support text; mixed media requires
a noncausal decoder with MEAN pooling and the extended batch API. Audio also
requires a 16 kHz projector. The v1.0.0 source pin exposes the mixed-batch API.
Use this probe to check the loaded assets and models. The local BF16 CPU
results use v0.7.0. [Qualification record](multimodal_embeddings.md)

An absent, malformed or unknown capability protocol throws a typed
`EmbeddingError` with `code: 'EMBEDDING_CAPABILITY_UNAVAILABLE'`; legacy string
embeddings remain available through their existing route.

After an unusable worker, embedding operations reload the model on the main
thread with the same options and GPU policy as existing string embeddings.
A remembered projector is restored eagerly for string, parts, batch and probe
fallbacks. Restoration is best effort: failure logs a payload-free warning and
keeps the model loaded and projector URL remembered. Capabilities report actual
loaded support, so image/audio remain false after failure. Each subsequent media
embedding or capability probe makes one further restoration attempt if the
projector is still unloaded. A failed media attempt rejects with
`EMBEDDING_PROJECTOR_MISSING`; strings and text-only parts remain usable without
retrying the projector. An already loaded or explicitly unloaded projector is
not retried. Cancellation during a restore rejects with `AbortError`; if the
bridge is disposed while a restore fails, a media call may reject with
`EMBEDDING_PROJECTOR_MISSING`.

### `embed(input, options?)`

```ts
type EmbedPart =
  | { type: 'text'; text: string }
  | { type: 'image'; bytes?: Uint8Array; url?: string; width?: number; height?: number }
  | { type: 'audio'; samples?: Float32Array; bytes?: Uint8Array; url?: string };
type EmbedInput = string | EmbedPart[];
embed(input: EmbedInput, options?: { normalize?: boolean }): Promise<number[]>
```

One string or one ordered parts array produces one full pooled vector. Strings
retain the existing execution route and defaults. A string must fit one physical
microbatch (`nUbatch`) when the model is encoder-only or memoryless, its
attention is noncausal, or its pooling is MEAN or CLS; a longer string rejects
before inference. Other strings may span several batches. Rank-pooled models
reject strings too.
Adjacent text is concatenated verbatim, including empty strings, whitespace and
Unicode. No task prefix, separator, chat template, truncation or per-part
averaging is added. Empty parts arrays and sparse arrays reject. The literal `<__media__>` marker is reserved
in parts text, because it would make media ordering ambiguous.

For each media part provide exactly one source. Encoded image/audio bytes and
URLs use the completion media decoder and configured image downscaling limits;
raw RGB bytes use `width * height * 3` as in completion input. PCM must be finite
mono 16 kHz `Float32Array` samples. Browser filesystem paths and video parts are
unsupported. URL inputs use the completion fetch behavior and require browser
fetch/CORS access; support for a modality does not guarantee every codec/source.
Caller buffers remain attached and unchanged; worker requests use structured
clone without transferring the caller's buffers. Lists and part descriptors are
snapshotted when queued. Keep buffers unchanged until the promise settles.

The entire expanded parts input must fit the actual context, per-sequence
context, logical batch and physical microbatch, with at least one sequence.
Overflow rejects before media encoding or model inference; the bridge never
chunks a parts input. Each mixed input uses one extended batch and one decode,
copying each media row into the batch before mtmd reuses its output buffer. Output width comes from
the pooled model output, which can differ from token embedding width.
Vectors are L2-normalized unless `normalize === false`; zero/nonfinite handling
matches the string path. Rank-pooled classifiers and encoder-decoder models
are unsupported by the parts API. Media capabilities are false for M-RoPE
projectors requiring multiple positions per embedding row.

```js
const capabilities = await bridge.getEmbeddingCapabilities();
if (capabilities.apiVersion === 1 && capabilities.image) {
  const vector = await bridge.embed([
    { type: 'text', text: 'A document: ' },
    { type: 'image', bytes: imageBytes },
  ]);
}
```

Parts failures are payload-free `EmbeddingError` instances with a stable code:

| Code | Meaning |
| --- | --- |
| `INVALID_EMBEDDING_INPUT` | Empty/malformed parts, ambiguous media sources or reserved marker |
| `EMBEDDING_CAPABILITY_UNAVAILABLE` | Missing/unknown bridge or mixed-batch API |
| `MODEL_NOT_EMBEDDING` | Unsupported model, pooling or attention configuration |
| `EMBEDDING_PROJECTOR_MISSING` | Media requires a loaded projector |
| `EMBEDDING_MODALITY_UNSUPPORTED` | Unsupported part, projector modality or PCM rate |
| `EMBEDDING_INPUT_TOO_LARGE` | Expanded input exceeds allocated capacity |
| `EMBEDDING_FAILED` | Media preparation, format, batch construction or inference failure |

Worker transport preserves these error types/codes. `cancel()` keeps the
existing best-effort semantics: observed cancellation rejects with `AbortError`,
and work that finishes before observing it can still resolve. Active work settles
before cleanup, disposal or the next operation. Interrupt latency is not guaranteed.
A subsequent text request can reuse the runtime.

### `embedBatch(inputs, options?)`

```ts
embedBatch(inputs: EmbedInput[], options?: { normalize?: boolean }): Promise<number[][]>
```

Returns one vector per input, in input order. Empty input resolves to `[]`
without inference. Part shapes and requested modalities across the batch are
checked before the first embedding; expanded capacity/format is checked for
each input before that input's inference. Execution is sequential; worker mode
forwards the batch as one request, whose timeout restarts after each finished
input. Each parts array is one vector.

## Next-token scoring

### `scoreNextToken(prompt, options?)`

```ts
scoreNextToken(
  prompt: string,
  options?: {
    candidates?: ArrayLike<number>;
    topK?: number;
    reusePromptPrefix?: boolean;
  },
): Promise<{
  candidates: { token: number; bytes: Uint8Array; logprob: number }[];
  top: { token: number; bytes: Uint8Array; logprob: number }[];
  promptTokens: number;
}>
```

Evaluates `prompt` (tokenized with special tokens, like `createCompletion`) and
returns natural-log probabilities for the next position without generating. The
values are a softmax over the raw logits, like llama-server `n_probs`; sampling
settings do not apply.

- `candidates` scores the given token ids, in request order.
- `topK` returns the most probable tokens, highest first; ties keep the lower id.
  Defaults to 0.
- `reusePromptPrefix` keeps the KV cache prefix shared with the previous prompt
  or completion. Defaults to `true`.
- `bytes` is the token's text as raw bytes; one piece can hold part of a UTF-8
  sequence. `logprob` is `-Infinity` for a zero probability, or when a
  non-finite logit leaves it undefined.

Each returned token costs a piece conversion, so keep `topK` small; values up to
the vocabulary size are accepted.

Pass `candidates`, a positive `topK`, or both. The call rejects when a token id
or `topK` is outside the vocabulary (the message contains
`is outside the vocabulary`), when media is pending, for encoder models, and
when the prompt exceeds the context. A `cancel()` issued while idle does not
affect later calls.

## Multimodal projector APIs

### `loadMultimodalProjector(url, options?)`

```ts
loadMultimodalProjector(url: string, options?: MultimodalProjectorLoadOptions): Promise<unknown>
```

Loads a multimodal projector (`mmproj`) file and updates the vision/audio
capability helpers. A relative URL resolves against the page's base URL, as
for `loadModelFromUrl()`. Once loaded, the downloaded file is freed from the
core filesystem, so it does not occupy the WASM heap. Multimodal generation uses `createCompletion(prompt,
{ parts })` after this succeeds. The bridge may restart or switch worker/direct
execution modes to satisfy projector constraints.

| Option | Behavior |
| --- | --- |
| `useCache` | Set to `true` to use the bridge cache for HTTP(S) projectors. The default is `false`, which preserves the `no-store` request for existing callers, including callers with credential-bearing URLs. |
| `cacheName` | Cache Storage cache name when `useCache` is enabled. Defaults to the bridge's configured `cacheName`. |

With caching enabled, this loads a prefetched projector from Cache Storage or
stores it while downloading. Use the same URL and cache name for prefetch and
load. If a WebKit part is missing or damaged, the bridge records
`mmproj_cache_entry_invalid` and retries from the network. Other URL schemes,
including `blob:` and `data:`, load uncached.

### `unloadMultimodalProjector()`

```ts
unloadMultimodalProjector(): Promise<unknown>
```

Unloads the active multimodal projector and clears cached multimodal capability
state.

### `supportsVision()` and `supportsAudio()`

```ts
supportsVision(): boolean
supportsAudio(): boolean
```

Return the multimodal capabilities reported by the loaded projector. They return
`false` before a projector is loaded or after it is unloaded.

## Text-to-speech

Text-to-speech is a versioned, capability-gated bridge feature. It requires a
compatible text model and generated-audio projector; ordinary audio-capable
multimodal models are not automatically TTS models.

The Qwen3-TTS 1.7B Q4 model plus Q8 projector used by the real-model smoke is
about 1.48 GB before runtime buffers. It is not practical in the wasm32 runtime.
Use the published memory64 core on a cross-origin-isolated Chromium page and
warn users that synthesis has a high browser-memory and latency cost. The
generated-audio path is experimental upstream. The checksum-pinned real-model
gate passes with WebGPU selected in both direct and worker runtimes. CPU/WASM is
a functional fallback, but it is considerably slower for this model pair.

### `getTextToSpeechCapabilities()`

```ts
getTextToSpeechCapabilities(): Promise<TextToSpeechCapabilities>
```

Returns the versioned capability record for the loaded model/projector. Check
`supported` before showing synthesis controls. `supportsLanguage` and
`supportsSpeakerReference` describe optional request inputs; `sampleRate` and
`channels` describe generated PCM. `reason` explains an unsupported state.
Check `supportsIncrementalAudio === true` before enabling streamed audio.
Older bridge assets may omit this capability.

### `synthesizeSpeech(options)`

```ts
synthesizeSpeech(options: TextToSpeechOptions): Promise<TextToSpeechResult>
```

A failed worker-mode synthesis request gets at most one main-thread retry, and
only when the model was not already loaded CPU-only and the error text matches
one of these existing recovery classifiers:

- WebGPU failure: `dispatch workgroup count`, `max compute workgroups per
  dimension`, `invalid commandbuffer`, or `ggml_webgpu: device error`.
- Worker timeout: `worker request timeout`, `worker init timeout`, or `worker
  timed out`.

The bridge disposes the worker, reloads the cached model and projector in the
main-thread runtime, and reruns the request there. WebGPU-classified failures
reload CPU-only and emit a CPU-fallback warning. The generic `worker timed out`
form also selects that CPU-only path. The exact `worker request timeout` and
`worker init timeout` forms preserve the original GPU offload settings and emit
no CPU-fallback warning. The retry itself is never recovered a second time.

Every other worker failure is rejected as-is. A timeout-classified failure
disposes the worker even when the model was already CPU-only and no retry runs.

If the core aborts during synthesis on a GPU-loaded model, the bridge
replaces the worker, restores the model CPU-only, and reruns synthesis once
on that worker. An abort on a CPU-only model surfaces to the caller.

Cancellation is classified before recovery. An aborted `signal`, `cancel()`, an
`AbortError`, or a worker error containing `cancel` rejects with an `AbortError`
`DOMException` and runs no main-thread retry.

Synthesizes speech and returns mono `Float32Array` PCM plus its sample rate,
sample count, generated-frame count, and truncation flag. The bridge does not
play or encode the audio. Applications can create a WAV or Web Audio buffer
from the returned PCM.

| Option | Description |
| --- | --- |
| `text` | Required non-empty text to speak. |
| `language` | Optional model language code, such as `en`; accepted values remain model-specific. |
| `speakerAudio` | Optional encoded reference-audio bytes when `supportsSpeakerReference` is true. |
| `promptBatchSize` | Prompt-processing batch size. Defaults to `512`. |
| `maxFrames` | Maximum generated audio frames. Defaults to `512`; a completed result reports `truncated` when this cap is reached. |
| `topK`, `topP`, `minP`, `temperature`, `seed` | Model sampling controls. Defaults are `40`, `0.95`, `0`, `0.8`, and a random seed. |
| `signal` | `AbortSignal` for cancellation. Cancellation rejects with `AbortError`. |
| `streamAudio` | Opt in to incremental PCM. Requires an `onProgress` callback. Defaults to `false`. |
| `onProgress` | Receives task state, remaining prompt tokens, generated frames, and truncation state. When streamed PCM is available, the event also carries `pcm`, `sampleOffset`, `sampleRate`, and `channels`. |

Direct and worker modes use the same API, and worker mode transfers the output
PCM buffer to the caller. Apart from the single eligible main-thread retry
described above, a failed request is surfaced to the caller, who may retry
explicitly. Streamed synthesis follows the rules below.

A second synthesis is not rejected. It — and any other shared-runtime operation
issued while synthesis is running — waits in the FIFO queue and runs in call
order; see [Operation serialization](#operation-serialization).

#### Streamed audio

With `streamAudio: true`, Qwen3-TTS delivers about 5.76 seconds of PCM per full
chunk. Any remaining tail arrives on the completing progress event. Concatenate
the chunks in `sampleOffset` order to obtain the same PCM as a non-streamed
request with the same sampling seed and settings. The result's `pcm` is empty
and its `sampleCount` reports the total number of samples.

Worker mode transfers each chunk to the caller. Cancellation stops chunk
delivery and rejects with `AbortError` if observed before the request settles,
including from the completing progress callback. After the first chunk reaches
the caller, a worker failure rejects the request without replaying synthesis.
A worker script without `supportsIncrementalAudio` rejects a streaming request
with `TextToSpeechStreamingUnsupportedError` (`TTS_STREAMING_UNSUPPORTED`).
Use bridge and worker scripts from the same release.

## Decision heads

Decision heads run Laya-style decision models: a ModernBERT encoder GGUF loaded
with `loadModelFromUrl()` plus a safetensors head that scores the options of a
question. The bridge returns raw scores; tokenizing prompts and turning scores
into answers (temperatures, probabilities, act thresholds) belong to the
caller, for example llamadart's `DecisionEngine`.

The loaded model must report `general.architecture` `modern-bert`, CLS (BOS),
SEP and MASK tokens, non-empty MASK text, and last-hidden-state output. The head
file holds F32, F16 or BF16 tensors under Laya's PyTorch names; the official
Laya `model.safetensors` checkpoint works too, and its `encoder.*` tensors are
ignored. The head's width must match the encoder's hidden size.

Each loaded head owns a private encoder context of the config's `max_len`
tokens (512 for Laya) on the loaded model, plus its weights (about 100 MB for
Laya). The head runs on WebGPU when the model was loaded with GPU layers and on
the CPU otherwise; `deviceName` reports which. Sequences run one at a time.

### `getDecisionCapabilities()`

```ts
getDecisionCapabilities(): Promise<DecisionCapabilities>
```

Returns `{ apiVersion, supported, reason? }` for the loaded model. `reason`
explains an unsupported state, such as no model, a model that is not a
ModernBERT encoder, a missing special token, or a core build without decision
heads.

### `loadDecisionHead(source, options?)`

```ts
loadDecisionHead(
  source: string | ArrayBuffer | ArrayBufferView,
  options?: {
    configJson?: string | null;
    onProgress?: (progress: { loaded: number; total: number }) => void;
  },
): Promise<DecisionHeadInfo>
```

Loads a head from a URL or from bytes and returns its handle, hidden size,
CLS/SEP/MASK token ids, MASK text, config text and head device. `configJson`
is the head's `rl_agent_config.json` text; without it the bridge reads the
head's `laya.config` safetensors metadata, and a head with neither is rejected.
The config sets `max_len` (default 512), which must not exceed the encoder's
trained context, and `head_layers` (default 2).

URL heads are fetched into the runtime's in-memory WASMFS and deleted after
loading, so peak memory includes the whole file: the official checkpoint is
about 800 MB. `onProgress` reports the download of a URL head. Worker mode
resolves a relative URL against the page and sends the worker a copy of byte
sources, so caller buffers are not detached. By default, a worker head load
times out after 10 minutes without download progress. Use
`workerRequestTimeoutMs` to override the budget.

### `runDecision(handle, sequences)`

```ts
runDecision(handle: number, sequences: readonly DecisionSequence[]): Promise<DecisionOutput[]>
```

Each sequence has `tokens` (the full token ids, including CLS and SEP),
`markers` (the token index of each option marker, in option order) and
`questionType` (`0` choice, `1` score, `2` noul). Each output has `logits`,
one raw value per marker, and the act head's `actLogits`, both `Float32Array`;
worker mode transfers them to the caller.

The bridge rejects non-integer values before calling the core. The core then
checks every sequence before its first encoder pass: 1 to `max_len` tokens, ids
inside the vocabulary, 1 to token-count markers, markers inside the sequence,
and a question type from 0 to 2. One invalid sequence rejects the whole call.

`cancel()` does not interrupt a native decision run. Later operations wait for
it to finish. In worker mode, if a decision RPC fails after a cancel request,
the bridge reports an `AbortError` with code `cancelled`. A worker run defaults
to a timeout of 10 minutes plus 1 minute per sequence. `workerRequestTimeoutMs`
overrides that budget. On timeout, the bridge falls back to the main-thread
runtime as described below.

### `freeDecisionHead(handle)`

```ts
freeDecisionHead(handle: number): Promise<void>
```

Frees a head. Freeing an unknown or already-freed handle does nothing.

Handles are not durable. `dispose()`, a worker restart, a fallback to the
main-thread runtime, and any model load that reaches the core, even one that
then fails, free every head, and a later
`runDecision()` with such a handle rejects with code `head-lost`. When the
worker fails during `runDecision()`, the bridge reloads the model on the main
thread and rejects the run with `head-lost`; heads must then be loaded again.
Handles are never reused within one bridge instance.

### Decision errors

Every rejection of the four decision methods has a string `code`
(`DecisionErrorCode`), in both direct and worker mode. The error keeps its
message and, as before, its type and name: cancellation is an `AbortError`,
whose numeric DOMException `code` this replaces, and an error from the worker is
a plain `Error`. Branch on `code`; messages may change.

| Code | Meaning |
| --- | --- |
| `head-lost` | The handle's head was freed, its model was unloaded, or the runtime that held it restarted or failed. Load the head again. |
| `no-model` | No model is loaded. |
| `disposed` | The bridge was disposed. |
| `cancelled` | `cancel()` stopped the operation. |
| `busy` | Generation or text-to-speech synthesis holds the runtime. |
| `context` | The head's private encoder context cannot be created, or its pooling or batch size cannot serve the head. |
| `invalid-argument` | A malformed handle, source, `configJson` or sequence, such as a non-integer token id. |
| `unsupported` | The core build or the loaded model cannot run decision heads. |
| `model` | The head or its config cannot be fetched or read, is malformed, or does not fit the encoder. |
| `inference` | The core rejected a sequence for the loaded encoder (length, token ids, markers or question type), or the encoder or head failed or returned malformed output. |
| `invalid-state` | Another runtime state rejected the call, such as exhausted head handles. |
| `runtime` | Any other failure, such as a worker or Wasm failure the bridge could not recover from. |

New codes may be added; treat an unknown code like `runtime`.

## LoRA adapters

LoRA adapters change the loaded model's output without reloading it, through
llama.cpp's `llama_adapter_lora_init_from_file_ptr` and
`llama_set_adapters_lora`, with native llamadart's `setLora`, `removeLora` and
`clearLoras` semantics. An
adapter is a GGUF LoRA file made for the loaded base model. Several adapters
apply at once, each at its own scale, to completions, embeddings and
next-token scores; decision heads use their own contexts and are unaffected.
Changing the applied set drops the cached prompt, so the next call evaluates
its whole prompt.

### `getLoraAdapterCapabilities()`

```ts
getLoraAdapterCapabilities(): Promise<LoraAdapterCapabilities>
```

Returns `{ apiVersion, supported, reason? }`. `supported` is true when the
core build implements the bridge's LoRA API version. Before the first model
load the core has not started, so it is false with `reason` `WebGPU core is not
initialized`. Gate LoRA support on this probe rather than on the asset tag.

### `loadLoraAdapter(source, options?)`

```ts
loadLoraAdapter(
  source: string | ArrayBuffer | ArrayBufferView,
  options?: LoraAdapterLoadOptions,
): Promise<LoraAdapterInfo>
```

Loads an adapter from a URL or from bytes and returns its `handle`. Loading
does not apply it; `setLoraAdapter()` does.

| Option | Description |
| --- | --- |
| `progressCallback(progress)` | Receives `{ loaded, total }` download events for a URL adapter. |
| `signal` | Cancels the download. A load whose adapter already reached the runtime still completes. |
| `useCache` | Reads and stores a URL adapter in Cache Storage, as models are. Defaults to true. |
| `cacheName` | Cache Storage cache for a URL adapter; defaults to the bridge's `cacheName`. |

The adapter is staged in the runtime's in-memory WASMFS and deleted after
loading. Worker mode resolves a relative URL against the page and sends the
worker a copy of byte sources, so caller buffers are not detached. A worker
adapter load times out after 10 minutes without download progress by default.
Use `workerRequestTimeoutMs` to override the budget.

The load rejects, and the runtime stays usable, for a file llama.cpp rejects,
including an adapter made for another base model (for example `tensor
'blk.0.attn_k.weight' has incorrect shape (hint: maybe wrong base model?)`),
and for an aLoRA adapter, which must activate only once its invocation tokens
appear in the prompt. Errors never include the adapter URL.

### `setLoraAdapter(handle, scale?)`

```ts
setLoraAdapter(handle: number, scale?: number): Promise<void>
```

Applies the adapter at `scale`, 1 by default. Setting an applied adapter again
changes its scale. A scale of 0 gives the same output as not applying the
adapter. `scale` must be a finite number and `handle` a positive integer, or
the call rejects with a `TypeError`. The core applies `scale` as a 32-bit
float, so a finite scale outside that range rejects with a `RangeError`.

### `removeLoraAdapter(handle)` and `clearLoraAdapters()`

```ts
removeLoraAdapter(handle: number): Promise<void>
clearLoraAdapters(): Promise<void>
```

`removeLoraAdapter()` stops applying one adapter and `clearLoraAdapters()`
stops applying all of them. Adapters stay loaded and can be set again.

Handles belong to the loaded model. `dispose()` and any model load that reaches
the core, even one that then fails, free every adapter, and a later call with
such a handle rejects and asks for the adapter to be loaded again. There is no
per-adapter free. Handles are never reused within one bridge instance.

When the bridge reloads the model itself, after a worker restart, a fallback
to the main-thread runtime or a switch to multimodal CPU mode, it then reloads
the applied adapters and applies them at their scales; the others reload when
next set. If this fails in the worker, the call rejects and the next call
tries again. With the worker enabled, the bridge keeps a copy of byte sources
for this. While an adapter is loaded, a generation that fails on
WebGPU is not retried by reloading the model on the CPU, since the reload would
free the adapters; the generation error is returned instead.

## Runtime metadata and diagnostics

### `getModelMetadata()`

```ts
getModelMetadata(): Record<string, unknown> | null
```

Returns model metadata from llama.cpp plus bridge diagnostic keys. Useful keys
include:

| Key | Meaning |
| --- | --- |
| `llamadart.webgpu.execution` | `worker` or `main-thread`. |
| `llamadart.webgpu.backends` | Comma-separated backend labels detected by the bridge. |
| `llamadart.webgpu.model_bytes` | Loaded model byte count. |
| `llamadart.webgpu.n_threads` | Active llama.cpp thread count. |
| `llamadart.webgpu.n_threads_batch` | Active batch thread count. |
| `llamadart.webgpu.thread_pool_size` | Detected or configured pthread pool size. |
| `llamadart.webgpu.n_gpu_layers` | Active GPU layer count. |
| `llamadart.webgpu.core_variant` | `wasm32`, `wasm64`, or initialization state. |
| `llamadart.webgpu.model_source` | Model load source, such as network/cache/remote fetch. |
| `llamadart.webgpu.model_cache_state` | Cache Storage state for the loaded model. |
| `llamadart.webgpu.runtime_notes` | Semicolon-separated bridge notes. When a memory64 core attempt fails, `core_mem64_unavailable:<reason>` accompanies `core_mem64_unavailable` and the bridge logs a warning with the same code before trying wasm32. The codes are `unsupported` (the engine rejects a memory64 feature probe after the core factory fails), `timeout` (the bridge's core initialization timer expires), and `init_error` (other failures, including module import). Single-thread cores report `core_single_thread` and `core_pthreads:0`. Automatic fetch-backed loading on a single-thread worker reports `model_fetch_backend_requires_pthreads` and streams into staging. `model_staging_js_backed` identifies JS-backed staging. `model_staging_js_backed_default_unavailable` identifies a WebKit load that uses standard WASMFS staging because the core is missing a staging export or lacks a supported JS model filesystem API version. `model_staging_heap_resident` identifies the final staged model file backed by a wasm heap block. When an eligible load uses JS chunks, `model_staging_heap_resident_unavailable:<reason>` reports `unknown_size` for an unverified decoded size, `unsupported` for an older or fixed-memory core, or the native allocation error code. Model downloads report `model_stream_resume_validated` for verified resumes, `model_stream_resume_unvalidated` for resumes without sufficient validation, and `model_stream_restart:<reason>` when a response requires restarting at zero. `model_cache_parts` identifies a WebKit part entry used or stored by the operation. `model_cache_entry_invalid` reports a missing or damaged part, and `model_stream_restart:cache` marks the shard retry. `mmproj_cache_entry_invalid` marks a projector retry after a missing or damaged part. `gpu_unavailable:<reason>` (`no_webgpu`, `no_adapter`, `adapter_missing_shader_f16`, `unknown`) says why a load with `nGpuLayers > 0` runs on the CPU. The bridge also logs one warning. Chrome hides `shader-f16` on NVIDIA drivers older than 615.71, and ggml-webgpu then registers no device. Safari reports `safari_gpu_layers_capped:<n>` when the cap lowers the effective offload, including when Apple Safari caps a full-offload request, `safari_gpu_probe_embedding` when probing with embeddings, `safari_gpu_probe_passed` or `safari_gpu_probe_failed` for the initial probe, `safari_gpu_probe_skipped_input_too_long` when both embedding prompt pairs exceed the effective microbatch, and `safari_fallback_cpu` when a retry loads on the CPU. |
| `llamadart.webgpu.mmproj_loaded` | `1` when a projector is loaded. |
| `llamadart.webgpu.supports_vision` | `1` when the active projector supports vision. |
| `llamadart.webgpu.supports_audio` | `1` when the active projector supports audio. |
| `llamadart.webgpu.worker_fallback_reason` | Present from a fallback from worker mode to direct runtime until `dispose()`, even when a later CPU-mode media request moves the model to a new worker. |

Worker mode reports `worker_start:direct` or `worker_start:blob`.
Threaded cores report `pthread_start:direct` when the bridge forwards the
core's worker URL and `pthread_start:blob` when the bridge supplies a module
blob. A blocked bridge worker blob start can leave a
`worker_start_blocked:<reason>` note on the page-thread runtime.

### `getContextSize()`

```ts
getContextSize(): number
```

Returns the active llama.cpp context size after model load. Before load, direct
runtime mode can return its cached/default context size, while worker-backed
instances return `0` until worker state has been populated.

### `isGpuActive()`

```ts
isGpuActive(): boolean
```

Returns whether the loaded model is actively using the WebGPU backend.

### `getBackendName()`

```ts
getBackendName(): string
```

Returns a user-facing backend label such as `WebGPU (Prototype bridge)`,
`WASM (Prototype bridge)`, or detected backend labels joined by comma.

### `setLogLevel(level)`

```ts
setLogLevel(level: string | number): void
```

Updates the bridge/core logging level. Finite numeric values are truncated and
clamped to `0..4`. String values are accepted for compatibility with callers
and may be interpreted by current or future bridge internals. After `dispose()`,
this method throws `Bridge has been disposed.`

### `cancel()`

```ts
cancel(): void
```

Requests best-effort cancellation for active model, draft, and LoRA transfers
(`loadModelFromUrl()`, `loadDraftModel()`, `loadLoraAdapter()`), token generation,
text-to-speech synthesis, and `embed()` and `embedBatch()` calls. It aborts the
active operation's transfer controller and asks that operation's captured worker
generation and runtime to stop. The signal bypasses the queue and stays scoped to
that operation through a worker restart or main-thread fallback. The worker
applies the signal only while that operation's call runs there. After disposal,
`cancel()` is a no-op.

Decision operations continue to run after `cancel()`. See
[runDecision](#rundecisionhandle-sequences) for the worker failure and queue
behavior.

### `dispose()`

```ts
dispose(): Promise<void>
```

Terminates the worker proxy, shuts down the direct runtime if it exists, unloads
model/projector state, and clears cached metadata. Call this when an application
is done with a bridge instance.

With updated bridge and core assets, direct-mode disposal also ends the pthread
runtime of a core loaded through `coreModuleUrl` or `coreModuleUrlMem64`, so
the browser can reclaim its memory. A core supplied through `coreModuleFactory`
belongs to the application and keeps running after disposal.

## Worker host entrypoint

### `enableBridgeWorkerHost()`

```ts
enableBridgeWorkerHost(): void
```

Installs the worker-side message handler used by `LlamaWebGpuBridge` worker
mode. Applications normally import `llama_webgpu_bridge_worker.js`, which calls
this automatically. Custom worker bundles can import the bridge module and call
this function themselves.

## Browser and asset requirements

- Use a secure context for WebGPU.
- The default pthread cores require
  cross-origin isolation (`Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`) so `SharedArrayBuffer` is
  available. `credentialless` does not isolate Safari. On a page that cannot
  share memory with workers, loading the default core configuration rejects
  with `CrossOriginIsolationError` (`code: 'CROSS_ORIGIN_ISOLATION_REQUIRED'`).
- The bridge can load the worker and cores from assets beside its module, or
  from an `assetBaseUrl` such as `'/webgpu_bridge/'`. Bundlers that process
  static `new URL(..., import.meta.url)` references can emit these assets with
  hashed names. On pages without isolation, set `allowSingleThreadCore: true`
  to select the `_st` variants. When passing explicit core URLs, use the
  single-thread loaders and supply matching Wasm URLs if served separately.
  Single-thread cores support vision, speech recognition and speech synthesis.
  See the [size measurements and limits](single_thread.md).
- On the Vite 7 dev server, use `optimizeDeps: { exclude: ['llama-web-bridge'] }`
  with the installed package name so its worker and core files stay beside the module.
- Serve both the core JS and Wasm from URLs reachable by the page and worker.
  For a threaded core on another origin, matching bridge and core assets start
  the core's pthread workers from a same-origin module `blob:` that imports
  the core. The page must be cross-origin isolated and the CDN must serve
  assets with CORS and a compatible `Cross-Origin-Resource-Policy`.
- With CSP, allow `worker-src blob: <asset-origin>`,
  `script-src <asset-origin> 'wasm-unsafe-eval'`, and `connect-src` for the
  asset and model hosts alongside the application's existing sources.
- Keep `llama_webgpu_bridge.js`, `llama_webgpu_bridge_worker.js`, the core JS,
  and wasm files from the same published bridge asset set.
- Use `dispose()` before dropping references when switching models or tearing
  down a page component.
- WebKit uses JS-backed staging by default when the core exposes the staging
  APIs and reports JS model filesystem API version 3 or later. With an
  unsupported core, the default uses `/models` (or `/mmproj` for projectors).
  On WebKit, staging failures can be `ModelStagingError` with the option unset.
  Set `jsBackedModelStaging: false` to opt out. On other browsers, use
  `jsBackedModelStaging: true` for models over about 1 GB on `wasm32`, or
  for any large model on `memory64` to cut the peak wasm heap (for example, when
  serving without cross-origin isolation using `WEBGPU_BRIDGE_PTHREADS=0` +
  `allowSingleThreadCore: true`). For single-file models with a known decoded
  size and an effective GPU layer count of 0 or 1 after the Safari cap, a
  supported growable core downloads into one memory-mapped wasm heap block. The block
  remains resident for the model's lifetime. Split GGUFs, multimodal projectors,
  `useMmap: false`, `useMlock: true`, and full or higher GPU offload use
  64 MiB JS `Uint8Array` chunks. Both paths use a private
  per-operation `/jsmodels-<n>` mount without copying through WASMFS. The path is
  unique per core instance, including across bridges that share one through
  `coreModuleFactory` from the same bridge bundle. Later loads use a fresh path
  even when a mount or its native registration remains after cleanup. The
  allocator checks individual paths and skips existing entries and native mount
  collisions. It reports `MODEL_STAGING_MOUNT_FAILED` if all 64 candidates are
  blocked. Each bridge cleans up its own mounts.
  Native memory grows by a few bytes per staging mount for the core's lifetime
  because Emscripten's WasmFS backend table retains created backends. Its
  cost is slightly slower native chunk reads on small loads (`Qwen3-0.6B-Q8_0`
  3-run median `nativeLoadMs` `480.9 ms` vs `444.6 ms` off; total load
  `1,317.7 ms` vs `1,660.5 ms` as direct staging skips wasm memory growth) and
  temporarily holding the full file in the JS heap until loaded and released. A
  JS allocation failure during staging throws `ModelStagingError` with code
  `'MODEL_STAGING_ALLOCATION_FAILED'`, explicitly enabling staging with an
  unsupported core throws `'MODEL_STAGING_UNSUPPORTED'`, an existing
  directory or mount at the staging path throws
  `'MODEL_STAGING_MOUNT_FAILED'` (without adopting it), and a single or split GGUF whose tensor
  payload exceeds the wasm32 `4 GiB` (`SIZE_MAX`) limit throws
  `ModelStagingError` with code `'MODEL_TENSOR_SIZE_LIMIT_EXCEEDED'`. Measured
  on Chrome WebGPU (Apple Metal-3, `nCtx: 2048`, `nGpuLayers: 999`,
  `crossOriginIsolated === false`, `typeof SharedArrayBuffer === 'undefined'`,
  `threads_capped_no_coi`) against `llama.cpp` v1.0.0: Qwen3-0.6B-Q8_0
  (767 MiB) cut `wasm32` peak wasm heap from `2.00 GiB` to `349 MiB` (`373.5` /
  `123.1` tok/s direct, `377.8` / `115.5` tok/s worker); Qwen3-4B-Q4_K_M
  (2.33 GiB; `wasm32` baseline failed at `1.37 GiB` during `FS.write`) loaded
  with `643 MiB` peak wasm heap in direct (`3,274 ms` load, `4.26 GB` peak JS
  heap before GC, `124.2` / `45.9` tok/s) and worker (`1,475 ms` load, `135.4` /
  `53.2` tok/s) modes; on `memory64` (`wasm64` no-pthreads), Qwen3-4B-Q4_K_M cut
  peak wasm heap from `8.01 GiB` (`8.60 GB`, `4,131 ms` load, `127.2` / `43.5`
  tok/s direct; `4,388 ms` load, `124.9` / `42.0` tok/s worker) to `656 MiB`
  (`0.69 GB`, `2,049 ms` load, `3.79 GB` peak JS heap dropping to `23.5 MB`,
  `124.8` / `50.7` tok/s direct; `1,585 ms` load, `128.6` / `50.0` tok/s worker);
  and `wasm32` Qwen3-8B-Q4_K_M (4.68 GiB) hit the `4 GiB`
  `gguf_init_from_reader` tensor size-overflow ceiling (`4,288,489,984 +
  9,437,184` > `4,294,967,295`).

### `BlobWorkerBlockedError`

When cross-origin pthread startup fails and the bridge can classify the
failure before initialization ends, it rejects with
`BlobWorkerBlockedError` (`code: 'BLOB_WORKER_BLOCKED'`, `target: 'pthread'`).
The error propagates to the caller from either execution mode. If the
core factory fails first or its deadline expires first, the caller receives
the original error.

The `reason` is best-effort. The bridge reports `csp` when it observes an
enforced blob-worker policy violation, `unavailable` when it cannot start or
confirm a module blob worker, and `script` when the module blob starts but
cannot import its core script. Firefox reports `unavailable` when a policy on
the bridge worker's response blocks its pthreads and Firefox does not deliver
the violation event to that worker. A blocked pthread start can report up to
one CSP violation per pool worker, four with the default pool size.
