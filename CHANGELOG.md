# Changelog

## Unreleased

- A speculative draft context that cannot be allocated, such as an EAGLE3 draft at
  `nBatch`/`nUbatch` 2048 on the wasm32 core, is freed instead of leaking, so repeated
  attempts and later model loads no longer abort the runtime. Its error names smaller
  `nBatch` and `nUbatch` or the memory64 core. A main or decision context that does not fit
  now rejects its load instead of aborting.
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
