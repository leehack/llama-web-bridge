# GGUF on pages without cross-origin isolation

Set `allowSingleThreadCore: true` and use the explicit `_st` core URLs. These
cores run with one CPU thread. WebGPU can still offload model layers, and the
cores support vision, speech recognition and speech synthesis. Use the wasm32
core in Safari, where memory64 is unavailable.

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

- **Default staging.** Model bytes occupy the same 4 GiB linear-memory address
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

On iOS Safari below 27, pass an explicit `nGpuLayers`. With
`jsBackedModelStaging: true`, set `nGpuLayers: 0` for the highest model-size
ceiling. A positive request defaults to at most one GPU layer on Safari below
27, which keeps GPU acceleration with a lower memory ceiling. The default
full-offload request uses JS chunks.

The memory estimates below come from Qwen3-0.6B-Q8_0 (767 MiB, `nCtx: 512`)
and memory-headroom probes on an iPhone 17 Pro running iOS 26.5. The model
architecture and context and batch settings affect the ceiling.

| Effective GPU layers | Q8 result with heap staging | Estimated model-size ceiling |
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
