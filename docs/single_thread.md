# GGUF on pages without cross-origin isolation

Set `allowSingleThreadCore: true` and use the explicit `_st` core URLs. These
cores run with one CPU thread. WebGPU can still offload model layers, and the
cores support vision, speech recognition and speech synthesis. Use the wasm32
core in Safari, where memory64 is unavailable. The bridge can load
single-thread cores from a CDN on non-isolated pages. With
`disableWorker: true`, the single-thread core starts no workers.

The default pthread cores require cross-origin isolation. Serve the page with
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`.

`allowSingleThreadCore` skips the bridge's initial isolation check. A core that
starts a pthread pool can stall if the page cannot share memory with workers.
When initialization stalls, the bridge retries once. Each runtime can wait for
`coreInitTimeoutMs` twice before reporting `Bridge core init timeout`, or four
times if both the memory64 and wasm32 cores stall. A shorter worker load
timeout can trigger a retry on the main thread.

A worker with automatic fetch-backed loading falls back to streamed staging
and reports `model_fetch_backend_requires_pthreads`. Forcing the fetch backend
on a single-thread worker raises `SingleThreadCoreError`. Every single-thread
load reports `core_single_thread`.

## wasm32 staging limits

- **WASMFS staging.** Model bytes occupy the same 4 GiB linear-memory address
  space as the runtime. Earlier measurements loaded files of 0.80 GB and
  1.04 GB. With Chromium 153 and Apple Metal, the 2.50 GB Qwen3-4B-Q4_K_M
  failed in WASMFS after writing 1 GiB in both direct and worker mode. A
  previous run reached 1.37 GiB. The available staging capacity varies with
  allocations and filesystem growth.
- **JS-backed staging.** With `jsBackedModelStaging: true`, the 2.50 GB
  Qwen3-4B-Q4_K_M generated on WebGPU in both modes and used 643 MiB peak wasm
  heap. This full-offload case uses JS chunks during staging. The browser's JS
  heap and device memory still limit model size. Qwen3-8B-Q4_K_M (4.68 GiB)
  returned `MODEL_TENSOR_SIZE_LIMIT_EXCEEDED` after staging because its
  tensor payload exceeds the wasm32 4 GiB bound.

Use JS-backed staging for models over about 1 GB. The
[API measurements](api.md#browser-and-asset-requirements) include staging and
heap costs on wasm32 and memory64.

## iOS Safari

WebKit defaults to JS-backed staging with a supported core. On iOS Safari
below 27, pass an explicit `nGpuLayers`. Set it to `0` for the highest
model-size ceiling. A positive request defaults to at most one GPU layer on
Safari below 27, which keeps GPU acceleration with a lower memory ceiling.
The default full-offload request uses JS chunks.

WebKit can hold several copies of a Cache Storage entry in WebContent.
On macOS Safari, a whole-model entry peaked at about three times the body
size during a write and twice the body size during a read. WebKit now caches
models in 16 MiB parts, commits a manifest after the last part, and reads one
part at a time. The manifest records the decoded size for heap staging on a
cache hit.

| Engine | Model cache behavior |
| --- | --- |
| WebKit with Web Locks and `crypto.randomUUID()` | 16 MiB parts and a manifest |
| WebKit missing one or both APIs | Existing cache entries are readable. |
| Chromium and Firefox | Whole model response under its URL |

When WebKit needs to download a model and the cache lock is busy, or either
Web Locks or `crypto.randomUUID()` is unavailable, a load proceeds uncached.
In those cases `prefetchModelToCache()` downloads the body and then rejects
with the message `Failed to store prefetched model in browser cache.`

On an iPhone 17 Pro running iOS 26.5.1, the 805 MB Q8 model loaded with
default caching at `nGpuLayers: 0`. Cold loads and cache-hit reloads worked in
direct and worker modes, and direct loads also worked with 448 MiB of extra JS
allocations. Prefetch wrote 48 parts and a manifest, and the next load hit the
cache. [Device Run results](https://console.cloud.google.com/storage/browser/llamadart-cuda-qa-20260916-devicerun/automation/sessions/session-8319c979-b975-4560-ba07-35ea7bdb6954/).

The memory estimates below come from Qwen3-0.6B-Q8_0 (767 MiB, `nCtx: 512`)
and earlier memory-headroom probes on the same device model. The model
architecture and context and batch settings affect the ceiling. Before the
WebKit default changed, iOS terminated WebContent in all six runs of this Q8
model with `jsBackedModelStaging` unset, across GPU-layer requests, cache
settings, and direct and worker modes.

| Effective GPU layers | Q8 result with WebKit heap staging | Estimated model-size ceiling |
| --- | --- | --- |
| 0 | Loads and generates with 448 MiB of additional JS allocations | About 1.2–1.4 GiB |
| 1 | Loads and generates; smaller batches leave more memory for weights | About 0.8–1.1 GiB at batch size 64 |

## Local WebGPU acceptance inputs

The [local smoke command](../CONTRIBUTING.md#validate-outputs) verifies the model
hash, absence of cross-origin isolation and `SharedArrayBuffer`, the requested
core and runtime, GPU activity, token generation, staged bytes and peak wasm heap.

| Memory mode | Model | SHA-256 |
| --- | --- | --- |
| wasm32 | Qwen3-0.6B-Q8_0.gguf | `361cc68159042c36ebff7715dc5a2e4612153e88f3e9c9c234820849d6dc9e1d` |
| wasm64 | Qwen3-4B-Q4_K_M.gguf | `ab27b9bfa375a178d6cba48f3ad892b94b7739659dcc7aae8058ce0ffed6b328` |
