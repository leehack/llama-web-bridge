# LlamaWebGpuBridge public API

This document describes the browser-facing JavaScript API exported by
`llama_webgpu_bridge.js`. It is the reference for applications that use the
bridge assets directly instead of going through a higher-level SDK such as
`llamadart`.

The TypeScript declaration shipped with the browser asset is
`llama_webgpu_bridge.d.ts`; this page explains the runtime behavior behind those
signatures.

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
| `coreModuleUrl` | `string` | URL for the wasm32 Emscripten JS loader, usually `llama_webgpu_core.js`. |
| `coreModuleUrlMem64` | `string` | URL for the optional wasm64 Emscripten JS loader. |
| `wasmUrl` | `string` | URL for the wasm32 `llama_webgpu_core.wasm` binary. |
| `wasmUrlMem64` | `string` | URL for the optional wasm64 wasm binary. |
| `coreModuleFactory` | `function \| Promise<function>` | Preloaded Emscripten factory. Supplying this disables worker mode because factories cannot be transferred to the worker. |
| `workerUrl` | `string` | URL for `llama_webgpu_bridge_worker.js`. Defaults to a sibling of the bridge module when possible. |
| `disableWorker` | `boolean` | Force direct runtime mode. |
| `preferMemory64` | `boolean` | Prefer the wasm64 core when available. Defaults to `true`; the runtime can still fall back to wasm32. |
| `workerRequestTimeoutMs` | `number` | Default timeout for worker RPC calls. |
| `workerInitTimeoutMs` | `number` | Timeout for worker initialization. |
| `workerModelLoadTimeoutMs` | `number` | Timeout for model load requests sent to a worker. |
| `workerMmprojLoadTimeoutMs` | `number` | Timeout for multimodal projector load requests sent to a worker. |
| `workerCompletionTimeoutMs` | `number` | Timeout for worker `createCompletion` RPC requests. |
| `workerGenerationStallTimeoutMs` | `number` | Stall timeout for worker generation after no token events arrive; clamped by the bridge. |
| `coreInitTimeoutMs` | `number` | Timeout while initializing the Emscripten core module. |
| `cacheName` | `string` | Cache Storage name used by model prefetch/load helpers. |
| `threads` | `number` | Requested llama.cpp thread count. The bridge caps this to the compiled pthread pool and runtime isolation support. |
| `threadsBatch` | `number` | Requested batch thread count; defaults to `threads`. |
| `threadPoolSize` | `number` | Hint for the compiled pthread pool size, used before the core can report it. |
| `nBatch` | `number` | llama.cpp batch size override. |
| `nUbatch` | `number` | llama.cpp micro-batch size override. |
| `nGpuLayers` | `number` | Default GPU layer count. `0` forces CPU/WASM execution; negative/omitted values let the bridge choose. |
| `userAgent` | `string` | User-agent override for Safari-specific GPU safeguards. |
| `remoteFetchThresholdBytes` | `number` | Size threshold for trying the native remote-fetch backend for large single-file models. |
| `remoteFetchChunkBytes` | `number` | Chunk size used by remote-fetch streaming paths. |
| `mediaMaxImagePixels` | `number` | Maximum image pixel count before multimodal image downscaling. |
| `mediaMaxImageEdge` | `number` | Maximum image width/height before multimodal image downscaling. |
| `disableImageDownscale` | `boolean` | Disable bridge-side image downscaling for multimodal image parts. |
| `allowAutoRemoteFetchBackend` | `boolean` | Allow automatic selection of the native remote-fetch backend when the model size qualifies. |
| `logLevel` | `string \| number` | Bridge/core logging level. Numeric values are clamped to the supported core range. |

Unknown config keys are accepted and may be consumed by current or future bridge
internals.

### Static properties

#### `LlamaWebGpuBridge.supportsSafariAdaptiveGpu`

`boolean` flag indicating that the bridge applies Safari-specific GPU layer
capping logic when Safari is detected.

## Model loading and cache helpers

### `loadModelFromUrl(url, options?)`

```ts
loadModelFromUrl(url: string | string[], options?: LoadModelOptions): Promise<unknown>
```

Loads a GGUF model from one URL, an explicit shard URL array, or an auto-expanded
split GGUF URL such as `model-00001-of-00002.gguf`. The bridge streams model
bytes into the active WASM filesystem unless the native remote-fetch backend is
used for a qualifying large single-file model.

Common `options` keys:

| Option | Description |
| --- | --- |
| `progressCallback(progress)` | Receives aggregate `{ loaded, total }` events. Split models may also include shard progress metadata. |
| `signal` | `AbortSignal` used to cancel model transfer. |
| `nCtx` | Context size. Defaults to the bridge runtime default, initially 4096. |
| `nThreads`, `nThreadsBatch` | Per-load thread overrides, capped to the active runtime. |
| `nGpuLayers` | Per-load GPU layer override. Use `0` for CPU/WASM mode. |
| `nBatch`, `nUbatch` | Per-load batch and micro-batch overrides. |
| `useCache`, `force` | Cache Storage controls for model responses. |
| `streamResumeRetries` | Retry count for resumable streamed model loads. |
| `remoteFetchThresholdBytes`, `remoteFetchChunkBytes` | Per-load remote-fetch tuning. |

Returns the underlying load result from the active runtime. After a successful
load, metadata and capability helpers reflect the loaded model.

### `prefetchModelToCache(url, options?)`

```ts
prefetchModelToCache(url: string | string[], options?: LoadModelOptions): Promise<unknown>
```

Fetches model URL(s) into Cache Storage without loading them into llama.cpp.
This uses a direct runtime helper even when the main bridge instance is currently
worker-backed. Use it to warm the browser cache before calling
`loadModelFromUrl()`.

### `evictModelFromCache(url, options?)`

```ts
evictModelFromCache(url: string | string[], options?: Record<string, unknown>): Promise<unknown>
```

Removes one model URL, explicit shard array, or expanded split-model URL set from
Cache Storage. Returns the runtime eviction result.

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
| `penalty` | Repetition penalty. Defaults to `1.1`. |
| `grammar` | Optional llama.cpp grammar string. |
| `seed` | Integer seed; random when omitted. |
| `onToken(piece, currentText)` | Token callback. By default `piece` is a `Uint8Array` containing stable UTF-8 bytes. Direct runtime mode provides the current full text by default; worker mode provides `''` unless `emitCurrentTextOnToken: true` is set. |
| `signal` | `AbortSignal`; aborting calls `cancel()`. |
| `warmup` | Marks a warmup generation. Some multimodal worker setup failures return an empty string instead of failing warmup. |
| `emitCurrentTextOnToken` | Direct runtime defaults to current text and uses `null` when set to `false`; worker mode sends current text only when this is `true` and otherwise sends `''`. |
| `tokenEventEncoding` | `'bytes'` (default) sends `Uint8Array` pieces; `'text'` sends string pieces. Worker events may already provide text pieces. |
| `parts` | Optional multimodal parts for image/audio prompts after a projector is loaded. |
| `mediaMaxPredict` | Cap for multimodal generation token count. |

Call `cancel()` or abort the supplied signal to request a best-effort stop.

## Tokenization and chat templates

### `tokenize(text, addSpecial?)`

```ts
tokenize(text: string, addSpecial = true): Promise<number[]>
```

Tokenizes text with the loaded model. `addSpecial` controls whether llama.cpp
adds model-specific special tokens.

### `detokenize(tokens, special?)`

```ts
detokenize(tokens: number[] | ArrayLike<number>, special = false): Promise<string>
```

Converts token IDs back to text. Non-array inputs are converted with
`Array.from()`.

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
`true` on success.

### `stateLoadFile(path, tokenCapacity?)`

```ts
stateLoadFile(path: string, tokenCapacity = bridge.getContextSize()): Promise<{ tokens: number[] }>
```

Loads a state/session file from the active runtime's WASMFS. `tokenCapacity` must
be positive, at least large enough for the stored token list, and no larger than
the active context size. The resolved `{ tokens }` value is the token list stored
at save time.

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

Loads state from bytes by staging them into a temporary runtime file. Empty input
is rejected. Worker mode transfers an internal copy to the worker, so
caller-owned `ArrayBuffer` or `Uint8Array` inputs are not detached.

## Embeddings

### `embed(text, options?)`

```ts
embed(text: string, options?: { normalize?: boolean }): Promise<number[]>
```

Generates an embedding vector for one string. Vectors are normalized unless
`options.normalize === false`.

### `embedBatch(texts, options?)`

```ts
embedBatch(texts: string[], options?: { normalize?: boolean }): Promise<number[][]>
```

Generates embeddings for multiple strings. Empty input resolves to `[]`. The
direct runtime currently processes the batch sequentially; worker mode forwards
the batch request to the worker runtime.

## Multimodal projector APIs

### `loadMultimodalProjector(url)`

```ts
loadMultimodalProjector(url: string): Promise<unknown>
```

Loads a multimodal projector (`mmproj`) file and updates the vision/audio
capability helpers. Multimodal generation uses `createCompletion(prompt,
{ parts })` after this succeeds. The bridge may restart or switch worker/direct
execution modes to satisfy projector constraints.

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
| `llamadart.webgpu.runtime_notes` | Semicolon-separated bridge notes such as thread caps or fallback reasons. |
| `llamadart.webgpu.mmproj_loaded` | `1` when a projector is loaded. |
| `llamadart.webgpu.supports_vision` | `1` when the active projector supports vision. |
| `llamadart.webgpu.supports_audio` | `1` when the active projector supports audio. |
| `llamadart.webgpu.worker_fallback_reason` | Present after falling back from worker mode to direct runtime. |

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

Updates the bridge/core logging level. Numeric values are clamped by the bridge;
string values are accepted for compatibility with callers but may be interpreted
by current or future bridge internals.

### `cancel()`

```ts
cancel(): void
```

Best-effort cancellation for active model transfer or generation. It aborts the
current transfer controller when present and asks the core/worker to stop active
generation.

### `dispose()`

```ts
dispose(): Promise<void>
```

Terminates the worker proxy, shuts down the direct runtime if it exists, unloads
model/projector state, and clears cached metadata. Call this when an application
is done with a bridge instance.

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
- Large single-file model loads and pthread-backed runtime paths require
  cross-origin isolation (`Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`) so `SharedArrayBuffer` is
  available.
- Serve `llama_webgpu_core.js` and `llama_webgpu_core.wasm` from URLs reachable
  by both the page and worker. If using wasm64, also serve the `_mem64` files.
- Keep `llama_webgpu_bridge.js`, `llama_webgpu_bridge_worker.js`, the core JS,
  and wasm files from the same published bridge asset set.
- Use `dispose()` before dropping references when switching models or tearing
  down a page component.
