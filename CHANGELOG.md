# Changelog

## Unreleased

- Keep `setLogLevel()` updates in the bridge configuration when the worker core retires. Queued operations own worker replacement and apply the saved level to the new worker. (#261)

- Cache WebKit model downloads and prefetches in 16 MiB parts. Cache hits read one part at a time. If a cached part is missing or damaged, `loadModelFromUrl()` can retry from the network. `loadDraftModel()` rejects that call with an error named `ModelCacheEntryError` in direct mode. The bridge drops the damaged entry for the next call when it can acquire the cache lock. (#257)

- Start cross-origin core pthreads through module blob workers. Report blocked pthread starts as `BlobWorkerBlockedError` and record worker startup routes in the runtime notes. (#223)

- Clear the core initialization timer when initialization settles so it cannot keep a disposed or retired core in memory.

- Stream Qwen3-TTS PCM through `onProgress` with `streamAudio`. The bridge reports `supportsIncrementalAudio` and rejects worker failures after delivery begins so callers never receive replayed audio. (#220)

- Thinking budgets accept an ordered list of natural ends. A tool opener can close reasoning and activate the paused tool grammar, and an exhausted budget emits the full forced close. The loaded core reports support through `thinkingBudgetMultiEnd`.

- Escape control characters in tensor names in wasm32 GGUF size errors and report an overflowed declared size as at least 2^64 bytes.

- Keep both memory64 fallback notes in metadata for later loads that reuse the fallback wasm32 core.
- Document the single loaded model slot for bridges sharing a `coreModuleFactory` core instance.

- Allocate JS-backed staging paths per core and skip occupied filesystem paths. A bridge sharing a core through `coreModuleFactory` can load a model even when an earlier bridge left a busy staging mount or stale native registration.

- Retire the native core after its abort callback or a genuine WebAssembly runtime error. Direct callers receive `ModelUnloadedError` on later model operations until a load succeeds. Worker mode replaces the worker and restores the remembered model and its active LoRA adapters. Media completion and speech synthesis on a GPU-loaded model retry once on the replacement worker with a CPU-only restore. Abort-like error text alone no longer restarts a worker or selects CPU offload. All four cores expose their realm's `WebAssembly.RuntimeError` and a hook that releases pthreads and JS-backed staging data. Older cross-realm cores without the constructor export retire through `onAbort`. Use bridge JS and the worker script from the same release.

- Carry strong ETag and expected size evidence across repeated streamed GGUF interruptions. Restart overlong interrupted ranges, and enforce the declared span and consistent totals when a full GET returns an identity-encoded byte range starting at zero.

- Treat visible empty `Content-Encoding` lists as identity and compare coding lists independent of spacing and case. Worker model downloads preserve the `ModelTransferLengthError` name on rejected transfers.

- Report the tensor name, declared byte size, and accumulated bytes when wasm32 GGUF tensor preflight rejects a model.

- Recognized Safari 27 and newer lift the default one-layer GPU cap. In a macOS Safari 27 worker test, Qwen3.5-0.8B went from 17 to 31 tokens per second with 99 layers requested. Older and unidentified Safari versions keep the one-layer cap. `safariMaxGpuLayers` overrides either default.
- On Apple Safari, the GPU probe checks two distinct normalized embeddings for pooled-embedding models and reports `safari_gpu_probe_embedding`. This keeps EmbeddingGemma-2 on the GPU when its embeddings pass the probe.
- The Safari embedding probe tries shorter inputs for small microbatches and keeps the loaded GPU layers if the shorter inputs still exceed the limit. It also rejects single-axis embeddings and vectors with the wrong native dimension.

- Validate streamed GGUF resumes against the final URL, visible validators, and range metadata. Restart a shard when the response contradicts the original download and reject incorrect resumed byte counts before native load. Progress can decrease on a restart.

- When memory64 initialization fails, report `core_mem64_unavailable:<reason>` in model metadata and log the same reason at warn level before trying wasm32. The existing `core_mem64_unavailable` note remains available.

- JS-backed staging downloads a single model file into one memory-mapped wasm heap block when its decoded size is known and the effective GPU layer count is 0 or 1. Cache hits take the same path, and skipped cases report `model_staging_heap_resident_unavailable:<reason>`. This reduces peak memory on iOS Safari. Pass `useMmap: false` to keep JS-chunk staging.
- Add opt-in `jsBackedModelStaging: true` (`LlamaWebGpuBridgeConfig`) for >1 GB `wasm32` models or large `memory64` models to stream fetched GGUFs directly into 64 MiB JS `Uint8Array` chunks under a private per-operation `wasmfs::Backend` mount (`/jsmodels-<n>`) without copying through WASMFS, working with both threaded and no-pthreads `wasm32` and `memory64` cores across direct and worker runtimes and emitting the `model_staging_js_backed` runtime note. All staged chunks for a mount are unmounted and freed per mount on completion, failure, or `dispose()`, JS staging OOM, missing core exports, mount path collisions (never adopting existing mounts), and single or split >4 GiB wasm32 GGUF tensor payloads surface as typed `ModelStagingError` (`MODEL_STAGING_ALLOCATION_FAILED`, `MODEL_STAGING_UNSUPPORTED`, `MODEL_STAGING_MOUNT_FAILED`, `MODEL_TENSOR_SIZE_LIMIT_EXCEEDED`), and real non-isolated WebGPU runs (`crossOriginIsolated === false`, `typeof SharedArrayBuffer === 'undefined'`, `threads_capped_no_coi`) cut Qwen3-0.6B `wasm32` peak wasm heap from 2.00 GiB to 349 MiB, unblock Qwen3-4B-Q4_K_M on `wasm32` (2.33 GiB, direct and worker), and cut `memory64` Qwen3-4B-Q4_K_M peak wasm heap from 8.01 GiB (8.60 GB) to 656 MiB (0.69 GB) up to Chrome's ~4.4 GB JS heap / 4 GiB wasm32 tensor limits.

- Build and inventory single-thread (`WEBGPU_BRIDGE_PTHREADS=0`) cores (`llama_webgpu_core_st.{js,wasm}` and `llama_webgpu_core_mem64_st.{js,wasm}`) in `CMakeLists.txt`, `scripts/build/build_bridge.sh`, `scripts/release/manifest.mjs` (with per-core `{ addressType, shared }` memory validation), CI, candidate, and publish workflows; add non-isolated `--threading single` validation in `scripts/smoke/state_persistence.mjs` and bump `HARNESS_VERSION` to `5.5.0`.
- Report `core_single_thread` when a load uses a single-thread core. In worker mode, a forced native fetch-backed load rejects with `SingleThreadCoreError` (`SINGLE_THREAD_CORE_UNSUPPORTED`). With automatic fetch-backed loading enabled, the bridge streams into staging and records `model_fetch_backend_requires_pthreads`.

- Bump the default `llama.cpp` pin in `llama_cpp.version` from `v0.5.0` to `v1.0.0` (verified in CI at `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`), retire the separate `v1.0.0` CI matrix lane, remove the `v0.5.0`-only `mtmd-audio.cpp` mel-worker CMake workaround while retaining `mtmd-subcontext.cpp` per-graph worker handling, and make `publish_assets.yml` fail early if `upstream_tag` differs from `llama_cpp.version` unless `allow_upstream_tag_override=true` is set.

- A load on a page that is not cross-origin isolated rejects at once with `CrossOriginIsolationError`, which names `Cross-Origin-Embedder-Policy: require-corp` and that `credentialless` does not isolate Safari, instead of hanging until the core init timeout. The `threads_capped_no_coi` note now appears only for a core loaded with `allowSingleThreadCore`.
- A load with `nGpuLayers: 0` keeps WebGPU out of the model and context backends, so CPU decoding on Firefox no longer waits on its 100 ms WebGPU queue ticks at every sync (about 50x faster on a small model). `splitMode: 3` at 0 layers still loads.
- A load with `nGpuLayers > 0` that ends up on the CPU adds the runtime note `gpu_unavailable:<reason>` and one warning, such as `adapter_missing_shader_f16` for Chrome on NVIDIA drivers older than 615.71.
- `allowSingleThreadCore: true` lets a `coreModuleUrl` core built without pthreads load on a page that is not cross-origin isolated, capped to one thread.
- `CompletionOptions.thinkingBudget.pauseGrammar: false` keeps a grammar applied inside a reasoning block, so a grammar that takes the block and its end tag itself, such as one for a prompt that ends inside the block, sees the budget's forced `</think>`. It needs the new `thinkingBudgetPauseGrammar` capability; omitted, the grammar pauses as before.
- WebGPU text-to-speech applies the GGUF's suppress tokens in the talker sampler, as upstream does.
  Qwen3-TTS no longer fails with `code out of range` when a codec control token is sampled.
- `createCompletion` can explicitly count prompt tokens in penalties with a grammar using `penaltyPromptTokensWithGrammar`; a matching loaded-core capability is required, and existing grammar defaults stay unchanged.

- Large wasm32 streamed model files use bounded size hints to reserve filesystem capacity, avoiding geometric file-growth allocation failures at 1 GiB without treating hidden compression metadata as a decoded-size contract.

- In worker mode, `embed` and `embedBatch` with image or audio parts get a 10-minute request
  budget instead of 2 minutes, and `embedBatch`, text or media, restarts its budget after each
  finished input.
  A CPU/WASM media batch, such as EmbeddingGemma-2 BF16 at `nCtx`/`nBatch`/`nUbatch` 4096 on
  memory64, no longer times out and reruns on the main thread. `workerRequestTimeoutMs` still
  overrides the budget.
- `loadMultimodalProjector` frees its downloaded GGUF once the projector is loaded instead of
  keeping a second copy in the wasm heap. On the wasm32 core, repeated media `embedBatch` calls
  with EmbeddingGemma-2 BF16 at `nCtx`/`nBatch`/`nUbatch` 4096 no longer abort with
  `Aborted(undefined)` from a compute buffer allocation failure.
- A speculative draft context that cannot be allocated, such as an EAGLE3 draft at
  `nBatch`/`nUbatch` 2048 on the wasm32 core, is freed instead of leaking, so repeated
  attempts and later model loads no longer abort the runtime. Its error names smaller
  `nBatch` and `nUbatch` or the memory64 core. A model whose context does not fit now rejects
  its load instead of aborting.
- `createCompletion` takes `grammarLazy`, `grammarTriggers` (word, pattern, full-pattern and
  token triggers) and `grammarRoot`, which apply llama.cpp's lazy grammar and start symbol;
  `getCompletionCapabilities` reports `lazyGrammar`. An invalid trigger pattern or token rejects
  instead of aborting the core.
- Drop the unused reads of `__llamadartFetchBackendLastError` and `__llamadartFetchBackendStats`;
  the stock `-lwasmfs_fetch.js` never sets them, so the `model_fetch_js_error` and
  `model_fetch_stats` runtime notes were never emitted.
- On a model cache miss, `loadModelFromUrl` and `loadDraftModel` report progress as the model
  downloads instead of only after the whole file is in Cache Storage, and a cancelled or failed
  load no longer leaves the model cached.
- String `embed`/`embedBatch` reject rank-pooled models (rerankers) instead of returning
  a vector padded with out-of-bounds memory, and reject input longer than `nUbatch` when
  the model must embed it in one micro-batch instead of aborting the runtime.
- `getModelMetadata()` keeps `llamadart.webgpu.worker_fallback_reason` when a CPU-mode audio or
  image request later moves the model to a new worker, and qualification fails such a worker
  ASR or TTS mode.
- Give every decision API rejection a stable string `code`, such as `head-lost`, `busy` or
  `context`, in direct and worker mode, so callers need not match its unchanged message.
- Reject `synthesizeSpeech` `text` or `language` holding U+0000 with a `TypeError` instead
  of speaking only the text before it, and keep the whole file name of a decision head URL
  holding U+0000 in its load errors.
- Fail qualification when a requested worker ASR or TTS mode fell back to the main
  thread instead of recording it as a worker run.
- `penalty` and `presencePenalty` count the prompt's last tokens in their 64-token window, as
  native llamadart does, unless a `grammar` (even an empty one) or media `parts` are set.
  Output for the same seed changes whenever `penalty` is not 1, including the default 1.1,
  or `presencePenalty` is not 0. The new `penaltyPromptTokens: false` option keeps the
  prompt out, and `getCompletionCapabilities` reports `penaltyPromptTokens`.
- Fail the release manifest build when a core's memories contradict its `wasm32` or
  `memory64` capability, and fail qualification when a requested wasm64 ASR or TTS
  mode fell back to the wasm32 core instead of recording it as wasm64.
- Keep the text after U+0000 in `tokenize`, string `embed`/`embedBatch` inputs and
  `scoreNextToken`/`createCompletion` prompts instead of dropping it; a grammar or
  thinking-budget tag holding U+0000 now rejects.
- Fail the wasm64 build when `llama_webgpu_core_mem64.wasm` is not memory64, instead of
  shipping a wasm32 core whose patched `FS.read` and `FS.write` throw on BigInt arguments.
- In worker mode, reload the model before `getDecisionCapabilities` and `runDecision`
  when a cancelled multimodal setup or a failed worker-restart reload left the worker
  without it: the probe no longer reports decision heads unsupported, and a run asks
  for its head to be loaded again instead of failing with "No model loaded".
- In worker mode, scope each `cancel()` to the operation it targets: a cancel issued
  as a call starts now stops it, as in direct mode, and one that reaches the worker
  after its call returned is dropped instead of setting the runtime's cancel flag. A
  multimodal completion cancelled before its first token returns empty instead of
  restarting the worker and generating again.
- In worker mode, keep streaming the pieces generated before the worker reads a
  `cancel()`, so they add up to the returned text; an aborted `signal` still stops
  the stream and rejects.
- Stream a character that the model splits across tokens intact at the start of a
  response, instead of wrong characters and repeated text, and return U+FFFD from a
  short `detokenize` result that ends mid-character; invalid bytes become U+FFFD.
- Remove the unreferenced `js/wasmfs_fetch_patch.js`; the build already links the stock
  `-lwasmfs_fetch.js`, so shipped assets are unchanged.
- Add ordered text/image/audio parts to `embed` and `embedBatch`, with a versioned
  `getEmbeddingCapabilities` probe and typed, payload-free `EmbeddingError` codes
  in direct and worker runtimes. Existing string embedding behavior is unchanged.
- Add a guarded mixed mtmd/extended-batch path: validate expanded capacity, copy
  each media row into the batch, decode once and return the full pooled output. Unsupported
  upstream builds report media false without changing the source/toolchain pins.
- Add opt-in, local-weight Chromium qualification against an independent native
  reference, plus ordering, buffer ownership, lifecycle and capacity regressions.
  The BF16 CPU wasm memory64 path passes cosine >= 0.9998 and relative L2 < 2%
  against native CPU, exact direct/worker parity and pre-task string-path parity.
  WebGPU GPU-kernel qualification was not run; assets remain unpublished.
- Preserve worker recovery signals during embedding capability probing and core
  aborts, restore loaded projectors on fallback, marshal exact UTF-8 bytes, reject
  unsupported M-RoPE media, and retain legacy coercion/disposal behavior.
- Use the existing string-embedding model reload options and main-thread GPU
  policy for every embedding fallback. Eagerly restore a remembered projector
  best effort, including after string fallback; a failed restore warns and keeps
  the model usable. Later media calls and capability probes retry once per call;
  failed media calls return `EMBEDDING_PROJECTOR_MISSING` and probes report the
  actual unavailable modalities. Generation fallback retains its existing policy.
