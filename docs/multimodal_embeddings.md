# Multimodal embedding implementation and qualification

Tracking: https://github.com/leehack/llamadart/issues/1072. Base: `57d9971`.

**Status: implementation complete; strict native CPU numerical qualification is not passed.** The supplied BF16 pair fits memory64. Direct and worker outputs agree exactly, but native component differences exceed the strict comparison targets. No publication or downstream adoption is qualified.

## Design

Legacy strings keep their existing exports, defaults and output. Ordered parts are snapshotted and adjacent text is concatenated verbatim. The parts export receives an explicit UTF-8 length, preserving embedded NUL as well as Unicode. One input produces one full pooled vector; empty parts and unsupported sources reject before staging. The batch preflights every requested modality before its first inference.

The versioned probe uses loaded-model pooling/attention, actual projector modalities, audio sample rate and the compiled mixed-batch feature. Production upstream without extended batches supports text-only parts in one ordinary batch, with media flags false. The mixed path tokenizes all text/media, validates allocated context/sequence/batch/microbatch limits, copies each reused mtmd output and runs one extended batch with positions and output flags through one DECODE call. Output uses the pooled width, with the existing normalization/JSON rules.

Worker requests use structured clone with no transfer list, retaining caller buffer ownership and part order. Typed error codes survive the worker round trip. The existing operation queue drains cancellation before cleanup or the next request. Native cancellation state is cleared after parts inference and at the start of a fresh legacy string request, including cancellation during JS staging.

## Source and inputs

- llama.cpp v0.7.0: `620d2225cc436ab0b001e3200e590e3b286ef9ab` (clean external checkout).
- Official weights: `ggml-org/embeddinggemma-2-GGUF@bfcd2987`; existing local BF16 files, read-only. No weights downloaded or substituted.
- Model SHA-256: `68bae29d62fb8c7d23e98d21fd4662753ddd636e6b62b8a70dfe059a9844f216`.
- Projector SHA-256: `d2033b3cd0223cfb2e7a2b50dc80a664c78c816d375f5b017ec3ef80ee0bc766`.
- Native reference: `scripts/smoke/embedding_reference.cpp`, upstream C APIs only, macOS arm64 CPU, four threads, no Metal or BLAS, context/batch/microbatch 2048, MEAN noncausal pooling. Separate text llama_batch and mixed llama_batch_ext algorithm follows native `tools/embedding_smoke.cpp`; it does not include bridge code.
- Browser build: local Emscripten **6.0.11**. An external copy of the maintained build recipe bypassed only the version verification under the task authorization; the repository verifier and **6.0.8** pin are unchanged. CI with the pinned toolchain remains required.

```bash
export EMBEDDING_MODEL_SHA256=68bae29d62fb8c7d23e98d21fd4662753ddd636e6b62b8a70dfe059a9844f216
export EMBEDDING_PROJECTOR_SHA256=d2033b3cd0223cfb2e7a2b50dc80a664c78c816d375f5b017ec3ef80ee0bc766
```

Commands live in [CONTRIBUTING.md](../CONTRIBUTING.md#multimodal-embedding-qualification). Build, cache, reference vectors, Chromium artifacts and mutation workspaces are external under `/private/tmp/llamadart-015-d-bridge-tmp`. Servers bind an ephemeral loopback port; port 8080 is never used.

## Verification

- `npm run check:js`: 664 total, 661 pass, zero fail, 3 existing skips.
- 43 embedding Node contracts plus 2 smoke-harness contracts. Native fixtures exercise 55 mixed/export cases, 6 pre-extended-batch cases and a bounded-address-space multiplication overflow case.
- 84/84 isolated guard/lifecycle mutations cause a failing regression.
- CI reliability and whitespace: PASS. Generated JS/declarations match the tested dist.
- Local v0.7.0 builds: wasm32 and memory64, Emscripten 6.0.11; pins unchanged.
- Actual v0.5.0 C++ core syntax check: PASS with mixed API disabled; old-core executable contract proves text parts succeed and media remains false.
- Existing Chromium state/extraction/generation smoke: PASS, direct and worker on wasm32 and memory64 (four rows).
- Existing Qwen3.5 image-completion smoke: PASS, direct and worker on wasm32 (two rows); cached existing weights only.
- Independent review: all code blockers resolved; reviewer reran the embedding suites and whitespace check. The numerical qualification gap remains a separate release blocker.

### BF16 matrix

The full expanded input uses context/batch/microbatch 2048, four CPU threads and MEAN noncausal pooling. The browser selects memory64 explicitly, with the compiled 12 GiB linear-memory ceiling; no BF16 OOM or quantized substitution occurred. This is CPU/WASM qualification, not WebGPU GPU-kernel qualification.

Strict comparison targets are normalized text < `1e-5`, normalized media/interleaved < `2e-4`, raw text < `1e-4`, raw media < `2e-3`, and direct/worker < `1e-6`. These targets are retained; they are not relaxed to turn the observed differences into passes.

| Input / output | Direct vs native max abs | Worker vs native max abs | Direct vs worker | Strict native target | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Text, normalized | 4.3192727e-4 | 4.3192727e-4 | 0 | 1e-5 | FAIL |
| Black image, normalized | 4.03673e-4 | 4.03673e-4 | 0 | 2e-4 | FAIL |
| Checkerboard, normalized | 1.9313358e-3 | 1.9313358e-3 | 0 | 2e-4 | FAIL |
| Silence, normalized | 9.3686138e-4 | 9.3686138e-4 | 0 | 2e-4 | FAIL |
| 440 Hz, normalized | 7.2510657e-4 | 7.2510657e-4 | 0 | 2e-4 | FAIL |
| Interleaved, normalized | 4.399643e-4 | 4.399643e-4 | 0 | 2e-4 | FAIL |
| Text, raw | 0.57669926 | 0.57669926 | 0 | 1e-4 | FAIL |
| Black image, raw | 0.341034 | 0.341034 | 0 | 2e-3 | FAIL |

All vectors have 768 components. Direct/worker parity passes for every measured normalized and raw vector. Image content changes the output by max component `0.0750488956`; silence versus tone changes it by `0.0706618503`. Independently normalizing the raw black/text vectors reproduces the bridge's normalized output within `1.49e-8`.

| Functional check | Direct | Worker |
| --- | --- | --- |
| BF16 load, memory64 identity, pooled output width | PASS | PASS |
| Capabilities: apiVersion 1, text/image/audio true | PASS | PASS |
| Ordered/interleaved input, batch order, adjacent text | PASS | PASS |
| Empty batch, microbatch overflow, caller buffers retained | PASS | PASS |
| Projector missing, unload/reload, subsequent reuse | PASS | PASS |
| Best-effort cancellation and subsequent text reuse | PASS (AbortError) | PASS (completed unchanged vector) |

The final smoke exits `1` solely for the 16 strict native numerical failures (eight comparisons per runtime); all functional checks and direct/worker comparisons complete. Final artifact: `/private/tmp/llamadart-015-d-bridge-tmp/embedding-final/multimodal-embeddings-result.json`; native reference: `reference-native-default.json` in the same task directory. CPU precision-control vectors remain separate; the table uses the default native CPU profile.

Disabling native Accelerate/llamafile preserved the native text reference exactly and did not resolve the gap. A portable ARMv8 native CPU target and a flash-attention-off text control also failed to bring text within the strict target. The precise kernel-level cause remains unproven. A pre-task local v0.7.0 bridge build reproduces the task text vector exactly in both runtimes (max delta `0`) and reproduces the same `4.3192727e-4` native gap. Its generated wrapper is byte-identical to base `57d9971` (SHA-256 `994a81c48c0f4c43d39bed98a671e873115d8a8eb1be93c0ea62a09aff9834ad`); runtime source/CMake differences between its source `4553c48` and `57d9971` are empty. The text discrepancy therefore predates this task. This local baseline is not a published-asset attestation. Legacy string inference retains its route, defaults and exact baseline output.

Bare `cancel()` is best effort. The real smoke accepts a completed, unchanged vector when work finishes without observing cancellation, or `AbortError` when it observes cancellation; either way it waits for settlement and verifies subsequent text reuse plus projector unload/reload. Unit fixtures additionally force cancellation during staging and native inference.

## Mutation evidence

Each mutation bypasses a new guard or breaks an ownership/lifecycle invariant in an isolated source copy, then runs the JS or native embedding regression suite. All **84/84** cause a failing test; the control suites pass. Capacity limits are mutated individually. The native stub retains embedding pointers and deliberately reuses the mtmd buffer. A bounded-address-space fixture checks multiplication overflow independently of host pointer width.

| Mutation | Failing regression |
| --- | --- |
| `parts-empty` | empty batch does no work; empty parts reject without inference |
| `part-null` | empty batch does no work; empty parts reject without inference |
| `text-shape` | malformed or unsupported part is rejected without staging: text type,text |
| `unsupported-modality` | malformed or unsupported part is rejected without staging: video type,url |
| `filesystem-source` | malformed or unsupported part is rejected without staging: image type,path,bytes |
| `single-source` | malformed or unsupported part is rejected without staging: image type |
| `byte-type` | malformed or unsupported part is rejected without staging: image type,bytes |
| `byte-empty` | malformed or unsupported part is rejected without staging: image type,bytes |
| `url-type` | malformed or unsupported part is rejected without staging: image type,url |
| `url-empty` | malformed or unsupported part is rejected without staging: image type,url |
| `pcm-shape` | malformed or unsupported part is rejected without staging: image type,samples |
| `pcm-empty` | malformed or unsupported part is rejected without staging: audio type,samples |
| `pcm-finite` | malformed or unsupported part is rejected without staging: audio type,samples |
| `reserved-marker` | malformed or unsupported part is rejected without staging: text type,text |
| `model-capability` | non embedding model |
| `projector-capability` | projector is required before media staging |
| `modality-capability` | unsupported image |
| `probe-version` | unknown probe version |
| `probe-text-type` | malformed capability payloads reject every unknown protocol shape |
| `probe-image-type` | malformed capability payloads reject every unknown protocol shape |
| `probe-audio-type` | malformed capability payloads reject every unknown protocol shape |
| `probe-symbol` | absent probe and missing mixed export fail closed without affecting legacy strings |
| `mixed-symbol` | absent probe and missing mixed export fail closed without affecting legacy strings |
| `stage-cancel` | cancellation while core inference runs waits for it and discards the result |
| `parts-cleanup` | parts form one ordered vector with verbatim adjacent text and no prompt decoration |
| `sparse-parts` | empty batch does no work; empty parts reject without inference |
| `adjacent-concatenation` | parts form one ordered vector with verbatim adjacent text and no prompt decoration |
| `explicit-byte-length` | parts form one ordered vector with verbatim adjacent text and no prompt decoration |
| `model-null-context` | mixed native batch guards, retained media, output width, one process and cleanup |
| `model-null-model` | mixed native batch guards, retained media, output width, one process and cleanup |
| `pooled-output-width` | mixed native batch guards, retained media, output width, one process and cleanup |
| `encoder-decoder` | mixed native batch guards, retained media, output width, one process and cleanup |
| `pool-none` | mixed native batch guards, retained media, output width, one process and cleanup |
| `pool-rank` | mixed native batch guards, retained media, output width, one process and cleanup |
| `mixed-encoder` | mixed native batch guards, retained media, output width, one process and cleanup |
| `mixed-mean` | mixed native batch guards, retained media, output width, one process and cleanup |
| `mixed-causal` | mixed native batch guards, retained media, output width, one process and cleanup |
| `native-model-guard` | mixed native batch guards, retained media, output width, one process and cleanup |
| `native-projector` | mixed native batch guards, retained media, output width, one process and cleanup |
| `native-mixed-guard` | mixed native batch guards, retained media, output width, one process and cleanup |
| `native-modality` | mixed native batch guards, retained media, output width, one process and cleanup |
| `native-vision` | mixed native batch guards, retained media, output width, one process and cleanup |
| `chunks-allocation` | mixed native batch guards, retained media, output width, one process and cleanup |
| `tokenize-status` | mixed native batch guards, retained media, output width, one process and cleanup |
| `expanded-count-overflow` | mixed native batch guards, retained media, output width, one process and cleanup |
| `token-capacity-query` | mixed native batch guards, retained media, output width, one process and cleanup |
| `token-result` | mixed native batch guards, retained media, output width, one process and cleanup |
| `expanded-empty` | mixed native batch guards, retained media, output width, one process and cleanup |
| `allocated-context` | mixed native batch guards, retained media, output width, one process and cleanup |
| `allocated-sequence-context` | mixed native batch guards, retained media, output width, one process and cleanup |
| `allocated-batch` | mixed native batch guards, retained media, output width, one process and cleanup |
| `allocated-ubatch` | mixed native batch guards, retained media, output width, one process and cleanup |
| `allocated-sequence-count` | mixed native batch guards, retained media, output width, one process and cleanup |
| `input-width` | mixed native batch guards, retained media, output width, one process and cleanup |
| `batch-allocation` | mixed native batch guards, retained media, output width, one process and cleanup |
| `batch-append` | mixed native batch guards, retained media, output width, one process and cleanup |
| `batch-position` | mixed native batch guards, retained media, output width, one process and cleanup |
| `batch-output` | mixed native batch guards, retained media, output width, one process and cleanup |
| `text-chunk-pointer` | mixed native batch guards, retained media, output width, one process and cleanup |
| `encode-status` | mixed native batch guards, retained media, output width, one process and cleanup |
| `encode-output` | mixed native batch guards, retained media, output width, one process and cleanup |
| `process-status` | mixed native batch guards, retained media, output width, one process and cleanup |
| `pooled-pointer` | mixed native batch guards, retained media, output width, one process and cleanup |
| `media-buffer-lifetime` | mixed native batch guards, retained media, output width, one process and cleanup |
| `embedding-mode-cleanup` | mixed native batch guards, retained media, output width, one process and cleanup |
| `text-batch-allocation` | mixed native batch guards, retained media, output width, one process and cleanup |
| `text-process-status` | mixed native batch guards, retained media, output width, one process and cleanup |
| `text-capacity-guard` | mixed native batch guards, retained media, output width, one process and cleanup |
| `mixed-capacity-guard` | mixed native batch guards, retained media, output width, one process and cleanup |
| `text-batch-cleanup` | mixed native batch guards, retained media, output width, one process and cleanup |
| `native-active-preflight` | mixed native batch guards, retained media, output width, one process and cleanup |
| `native-media-cleanup` | mixed native batch guards, retained media, output width, one process and cleanup |
| `native-cancel-cleanup` | mixed native batch guards, retained media, output width, one process and cleanup |
| `embedding-return-code` | microbatch overflow is a typed inference error and cleans media |
| `parts-batch-preflight` | unknown probe version |
| `unloaded-capability` | unloaded model capabilities are all false without core IO |
| `normalization-option` | mixed native batch guards, retained media, output width, one process and cleanup |
| `worker-host-error-code` | worker host posts payload-free typed embedding error codes |
| `worker-error-class` | real proxy reconstructs typed embedding errors |
| `no-prompt-separator` | parts form one ordered vector with verbatim adjacent text and no prompt decoration |
| `media-element-overflow` | media element multiplication rejects overflow in a bounded address space |
| `token-query-int32-min` | mixed native batch guards, retained media, output width, one process and cleanup |
| `native-audio-support` | mixed native batch guards, retained media, output width, one process and cleanup |
| `native-audio-rate` | mixed native batch guards, retained media, output width, one process and cleanup |

Raw mutation records and failing outputs: `/private/tmp/llamadart-015-d-bridge-tmp/mutations/results.json` and neighboring `<guard>.log` files.

## Files

| Area | Files |
| --- | --- |
| Public API and direct path | `js/src/internal/embedding.ts`, `js/src/{bridge,runtime,internal/types}.ts`, `js/src/llama_webgpu_bridge.{js,d.ts}` |
| Worker transport | `js/src/worker_{host,proxy,protocol}.ts` |
| Generated assets | `js/llama_webgpu_bridge.{js,d.ts}` (generated, never hand-edited) |
| Native algorithm and ABI | `src/llama_webgpu_embedding.h`, `src/core/exports_embedding.inc`, `src/core/exports_context.inc`, `src/llama_webgpu_core.cpp`, `CMakeLists.txt` |
| Regressions | `tests/bridge/multimodal_embedding{,_native}_test.mjs`, operation queue lifecycle roster, `tests/smoke/multimodal_embeddings_test.mjs` |
| Reference and Chromium harness | `scripts/smoke/{embedding_reference.cpp,multimodal_embeddings.mjs}` |
| Integration and evidence | `README.md`, `CONTRIBUTING.md`, `docs/{api,multimodal_embeddings}.md`, `CHANGELOG.md` |

## Scope and remaining gates

No source/toolchain pin, workflow, release, asset publication or downstream Dart adoption changes. Hosted exact-head CI and immutable candidate qualification/publication remain separate. Synthetic fixtures verify wiring and order and measure numerical differences; they do not measure retrieval quality. Browser memory mode and any unqualified model/backend combinations must be reported explicitly.
