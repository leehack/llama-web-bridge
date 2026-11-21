# Multimodal embedding qualification

Tracking: [llamadart #1072](https://github.com/leehack/llamadart/issues/1072).

**Status: local BF16 CPU wasm memory64 qualification passed.** The run used
llama.cpp v0.7.0. The current bridge pin, v1.0.0, also exposes the mixed-batch
API. See [Embeddings](api.md#embeddings) for the runtime contract.

## Source and inputs

- llama.cpp v0.7.0: `620d2225cc436ab0b001e3200e590e3b286ef9ab` (clean external checkout).
- Official weights: `ggml-org/embeddinggemma-2-GGUF@bfcd2987`; existing local BF16 files, read-only. No weights downloaded or substituted.
- Model SHA-256: `68bae29d62fb8c7d23e98d21fd4662753ddd636e6b62b8a70dfe059a9844f216`.
- Projector SHA-256: `d2033b3cd0223cfb2e7a2b50dc80a664c78c816d375f5b017ec3ef80ee0bc766`.
- Native reference: `scripts/smoke/embedding_reference.cpp`, upstream C APIs only, macOS arm64 CPU, four threads, no Metal or BLAS, context/batch/microbatch 2048, MEAN noncausal pooling. Separate text llama_batch and mixed llama_batch_ext algorithm follows native `tools/embedding_smoke.cpp`; it does not include bridge code.
- Browser build: local Emscripten 6.0.11. The checked-in build uses the
  Emscripten version pinned in `emsdk.version`.

```bash
export EMBEDDING_MODEL_SHA256=68bae29d62fb8c7d23e98d21fd4662753ddd636e6b62b8a70dfe059a9844f216
export EMBEDDING_PROJECTOR_SHA256=d2033b3cd0223cfb2e7a2b50dc80a664c78c816d375f5b017ec3ef80ee0bc766
```

See [CONTRIBUTING.md](../CONTRIBUTING.md#multimodal-embedding-qualification)
for the build and smoke commands.

## BF16 qualification results

The full expanded input uses context/batch/microbatch 2048, four CPU threads and MEAN noncausal pooling. The browser selects memory64 explicitly, with the compiled 12 GiB linear-memory ceiling; no BF16 OOM or quantized substitution occurred. This is CPU/WASM qualification, not WebGPU GPU-kernel qualification.

Acceptance requires **cosine >= 0.9998 AND relative L2 < 2% per input/output against native CPU**, plus **exact direct/worker parity** and **exact legacy string-path parity against the pre-task bridge**. The same cosine/relative-L2 criteria apply to the measured raw vectors. Per-component max-absolute differences remain informational: BF16 wasm SIMD and native NEON reductions need a vector-level comparison.

The smoke also compares all 15 cross-input cosine similarities. The largest
difference from native CPU was `0.000975830487`.

| Input / output | Cosine vs native | Relative L2 vs native | Max abs vs native (info) | Direct vs worker max abs | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Text, normalized | 0.999993365 | 0.36429% | 0.00043192727 | 0 | PASS |
| Black image, normalized | 0.999993212 | 0.36845% | 0.000403673 | 0 | PASS |
| Checkerboard, normalized | 0.999892835 | 1.46400% | 0.0019313358 | 0 | PASS |
| Silence, normalized | 0.999962684 | 0.86390% | 0.00093686138 | 0 | PASS |
| 440 Hz, normalized | 0.999985174 | 0.54454% | 0.00072510657 | 0 | PASS |
| Interleaved, normalized | 0.999995083 | 0.31358% | 0.0004399643 | 0 | PASS |
| Text, raw | 0.999993365 | 0.36742% | 0.57669926 | 0 | PASS |
| Black image, raw | 0.999993212 | 0.37002% | 0.341034 | 0 | PASS |

Native comparison columns apply equally to both runtimes. All vectors have 768 components. Direct/worker parity passes for every measured normalized and raw vector. Image content changes the output by max component `0.0750488956`; silence versus tone changes it by `0.0706618503`. Independently normalizing the raw black/text vectors reproduces the bridge's normalized output within `1.49e-8`.

| Functional check | Direct | Worker |
| --- | --- | --- |
| BF16 load, memory64 identity, pooled output width | PASS | PASS |
| Capabilities: apiVersion 1, text/image/audio true | PASS | PASS |
| Ordered/interleaved input, batch order, adjacent text | PASS | PASS |
| Empty batch, microbatch overflow, caller buffers retained | PASS | PASS |
| Projector missing, unload/reload, subsequent reuse | PASS | PASS |
| Best-effort cancellation and subsequent text reuse | PASS (AbortError) | PASS (completed unchanged vector) |

Legacy string output matched the saved pre-task bridge vectors exactly in
both runtimes, including normalized and raw output. CPU precision controls
(without Accelerate or llamafile, portable ARMv8, and flash attention off)
did not eliminate the arithmetic gap. Its kernel-level cause remains unproven.

Bare `cancel()` is best effort. The real smoke accepts a completed, unchanged vector when work finishes without observing cancellation, or `AbortError` when it observes cancellation; either way it waits for settlement and verifies subsequent text reuse plus projector unload/reload. Unit fixtures additionally force cancellation during staging and native inference.

## Scope and remaining gates

These results cover the pinned BF16 fixtures and CPU/WASM inference.
WebGPU GPU-kernel qualification and retrieval quality are outside this run.
Hosted CI, immutable release qualification, and downstream adoption have
separate gates.
