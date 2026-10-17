# Changelog

## Unreleased

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
- Preserve loaded embedding capacities during GPU-to-CPU worker recovery,
  including native batch defaults; restore projectors only for media inputs,
  lazily after string-only fallback, and identify embedding recovery in warnings.
