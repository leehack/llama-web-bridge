# GGUF on pages without cross-origin isolation

Set `allowSingleThreadCore: true` to select the single-thread defaults. If you
supply core URLs, point them at the `_st` variants. These cores run with one
CPU thread and support WebGPU offload, vision, speech recognition, and speech
synthesis. The iOS Safari measurements below use wasm32. The bridge can load
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

With standard WASMFS staging, model bytes share the 4 GiB linear-memory
address space with the runtime. Earlier runs loaded files of 0.80 GB and
1.04 GB. The 2.50 GB Qwen3-4B-Q4_K_M failed in WASMFS after writing 1 GiB
in both direct and worker modes on Chromium 153 with Apple Metal. An earlier
run reached 1.37 GiB. Allocations and filesystem growth affect that limit.

JS-backed staging avoids storing a second full copy in the wasm heap. The
browser's JS heap, device memory, and wasm32 tensor address limit still
constrain the largest model. See the
[staging API](api.md#js-backed-model-staging) for browser defaults and errors.

### Chromium WebGPU measurements

The runs below used Chromium WebGPU on Apple Metal-3 with the llama.cpp
v1.0.0 pin, `nCtx: 2048`, and `nGpuLayers: 999`. The page was non-isolated
and `SharedArrayBuffer` was unavailable. The cores used one CPU thread.

| Model (size) | Staging / Mode | Load time | Peak wasm heap | Peak JS heap | Prefill / Decode | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B-Q8_0 (767 MiB) | `wasm32` Off (`/models`), direct | 1,660.5 ms (3-run median) | 2.00 GiB | 300 MB | 359.6 / 119.4 tok/s | Pass |
| Qwen3-0.6B-Q8_0 (767 MiB) | `wasm32` JS-backed (`/jsmodels`), direct | 1,317.7 ms (3-run median) | 349 MiB | 1.18 GB | 373.5 / 123.1 tok/s | Pass |
| Qwen3-0.6B-Q8_0 (767 MiB) | `wasm32` JS-backed (`/jsmodels`), worker | 731.8 ms | 349 MiB | — | 377.8 / 115.5 tok/s | Pass |
| Qwen3-4B-Q4_K_M (2.33 GiB) | `wasm32` Off (`/models`), direct | — | 2.75 GiB | 863 MB | — | Fail in `FS.write` at 1.37 GiB |
| Qwen3-4B-Q4_K_M (2.33 GiB) | `wasm32` JS-backed (`/jsmodels`), direct | 3,274 ms | 643 MiB | 4.26 GB (before GC) | 124.2 / 45.9 tok/s | Pass (`37/37` GPU layers) |
| Qwen3-4B-Q4_K_M (2.33 GiB) | `wasm32` JS-backed (`/jsmodels`), worker | 1,475 ms | 643 MiB | — | 135.4 / 53.2 tok/s | Pass (`37/37` GPU layers) |
| Qwen3-4B-Q4_K_M (2.33 GiB) | `memory64` Off (`/models`), direct | 4,131 ms | 8.01 GiB (8.60 GB) | 1.06 GB (drops to 8.1 MB) | 127.2 / 43.5 tok/s | Pass |
| Qwen3-4B-Q4_K_M (2.33 GiB) | `memory64` Off (`/models`), worker | 4,388 ms | 8.01 GiB (8.60 GB) | — | 124.9 / 42.0 tok/s | Pass |
| Qwen3-4B-Q4_K_M (2.33 GiB) | `memory64` JS-backed (`/jsmodels`), direct | 2,049 ms | 656 MiB (0.69 GB) | 3.79 GB (drops to 23.5 MB) | 124.8 / 50.7 tok/s | Pass |
| Qwen3-4B-Q4_K_M (2.33 GiB) | `memory64` JS-backed (`/jsmodels`), worker | 1,585 ms | 656 MiB (0.69 GB) | — | 128.6 / 50.0 tok/s | Pass |
| Gemma 4 E2B Q4_K_S (2.83 GiB) | `wasm32` JS-backed (`/jsmodels`), direct | 1,389 ms | 2.09 GiB | — | — / 64.0 tok/s | Pass (`36/36` GPU layers) |
| Qwen3-8B-Q4_K_M (4.68 GiB) | `wasm32` JS-backed (`/jsmodels`), direct | — | — | — | — | Fail: wasm32 4 GiB `gguf_init_from_reader` tensor overflow |

For the 767 MiB model, median native load time rose from 444.6 ms to 480.9 ms
with JS-backed staging. Total median load time fell from 1,660.5 ms to
1,317.7 ms as direct staging avoided wasm memory growth.

On the 4B run Chrome approached its roughly 4.4 GB V8 JS heap limit
before staging chunks were released. The 8B model exceeded the separate
wasm32 tensor payload limit of 4 GiB and reported
`MODEL_TENSOR_SIZE_LIMIT_EXCEEDED`.

The Gemma 4 run used Chrome 158 on an Apple M4 Max. The wasm32 core reads
the model's per-layer token embeddings eagerly so chunked staging can load
them without a file mapping. The embedding table uses about 1.5 GiB of CPU
memory with all GPU layers offloaded.

## iOS Safari

WebKit defaults to JS-backed staging with a supported core. On Apple Safari
before 27, the bridge caps full-offload requests, including the default and
negative `nGpuLayers`, to one GPU layer. This keeps GPU acceleration and
allows eligible models to use heap staging. Set `nGpuLayers` to `0` for the
highest model-size ceiling. Use `safariMaxGpuLayers: Infinity` to opt out for
a load. Uncapped full offload uses JS chunks.

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

Use [the bridge cache query](api.md#ismodelcachedurl-options) for a cached model
or projector. Pass `{ useCache: true }` to
[`loadMultimodalProjector`](api.md#loadmultimodalprojectorurl-options) to use the
same cache for the projector load.

On an iPhone 17 Pro running iOS 26.5.1, the 805 MB Q8 model loaded with
default caching at `nGpuLayers: 0`. Cold loads and cache-hit reloads worked in
direct and worker modes, and direct loads also worked with 448 MiB of extra JS
allocations. Prefetch wrote 48 parts and a manifest, and the next load hit the
cache. [Device Run results](https://console.cloud.google.com/storage/browser/llamadart-cuda-qa-20260916-devicerun/automation/sessions/session-8319c979-b975-4560-ba07-35ea7bdb6954/).

On the same device model and OS version, the wasm32 single-thread core
loaded Qwen3-0.6B-Q8_0 with default caching and heap-resident staging. The
omitted-count direct and worker cases and the direct `nGpuLayers: -1` case
loaded one GPU layer and passed the Safari GPU probe. The direct
`nGpuLayers: 0` control loaded zero GPU layers. All four cases generated
“Paris” with `nCtx: 512`. Device Run session
`session-e7d6dfa1-27eb-4cdd-90bd-e4bd37a5d736` recorded the results for
[#300](https://github.com/leehack/llama-web-bridge/pull/300). The temporary
public objects were deleted and returned HTTP 403 by HEAD and GET.

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
| wasm32 | [gemma-4-E2B-it-Q4_K_S.gguf](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/blob/0314792d7f1f7e229411f620751375812bb9faf2/gemma-4-E2B-it-Q4_K_S.gguf) | `96ee1ee1d562be52dbbc4c12694ed67de8c6ae09451d420fed7d067e552c25cb` |
| wasm64 | Qwen3-4B-Q4_K_M.gguf | `ab27b9bfa375a178d6cba48f3ad892b94b7739659dcc7aae8058ce0ffed6b328` |

For the Gemma 4 regression check, pass `--js-backed-model-staging --gpu-layers 999`.
The smoke verifies all 36 GPU layers offload and runs text generation.
