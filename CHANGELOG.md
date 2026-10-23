# Changelog

## Unreleased

- `penalty` and `presencePenalty` count the prompt's last tokens in their 64-token window, as
  native llamadart does, unless a `grammar` or media `parts` are set. Output for the same
  seed changes whenever `penalty` is not 1, including the default 1.1, or `presencePenalty`
  is not 0.
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
