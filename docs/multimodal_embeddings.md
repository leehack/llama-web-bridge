# Multimodal embedding implementation and qualification

Tracking: https://github.com/leehack/llamadart/issues/1072. Base: `57d9971`.

**Status: local BF16 CPU wasm memory64 qualification passes.** Per input, cosine is at least 0.9998 and relative L2 is below 2% against the native CPU reference. Direct and worker outputs agree exactly, and legacy string output agrees exactly with the pre-task bridge. Max-absolute differences are informational. WebGPU GPU-kernel qualification was not run. Publication and downstream adoption remain separate.

## Design

Legacy strings keep their existing exports, defaults and output. Ordered parts are snapshotted and adjacent text is concatenated verbatim. The parts export receives an explicit UTF-8 length, preserving embedded NUL as well as Unicode. One input produces one full pooled vector; empty parts and unsupported sources reject before staging. The batch preflights every requested modality before its first inference.

The versioned probe uses loaded-model pooling/attention, actual projector modalities, audio sample rate and the compiled mixed-batch feature. Production upstream without extended batches supports text-only parts in one ordinary batch, with media flags false. The mixed path tokenizes all text/media, validates allocated context/sequence/batch/microbatch limits, copies each mtmd embedding row into the extended batch before reusing the output buffer and runs one extended batch with positions and output flags through one DECODE call. Output uses the pooled width, with the existing normalization/JSON rules. M-RoPE projectors are rejected and advertise no media support because this batch supplies one position per embedding row.

Worker requests use structured clone with no transfer list, retaining caller buffer ownership and part order. Typed error codes survive the worker round trip. The existing operation queue drains cancellation before cleanup or the next request. Native cancellation state is cleared after parts inference and at the start of a fresh legacy string request, including cancellation during JS staging.

Embedding worker fallback uses main's existing string-embedding model recovery unchanged: the same sanitized load options and main-thread GPU policy, with no embedding CPU caps, context reconstruction or private batch-default hint. The shared fallback method is byte-for-byte identical to main. Generation retains main's nonempty-parts detection (including uppercase image/audio); embedding media detection is separate, so text-only embedding parts do not request a projector.

Every embedding fallback (string, parts, batch and capability probe) eagerly attempts to restore a remembered projector after shared model recovery. This best-effort step is outside the shared dispose-on-failure catch. A failed projector load emits a payload-free warning, preserves the healthy model and remembered projector URL, and leaves actual media capabilities false. A subsequent media call or capability probe attempts one restore per bridge call if the projector remains unloaded; there is no bridge retry loop. Media preflight returns typed `EMBEDDING_PROJECTOR_MISSING` on failure. Strings and text-only parts remain usable and do not retry the projector. A successful restore is reused; explicitly unloading the projector clears its remembered source. Cancellation during a restore rejects with `AbortError`; a disposal racing a failed restore may surface `EMBEDDING_PROJECTOR_MISSING` to the media call.

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

- Round 4 `npm run check:js`: 734 total, 731 pass, zero fail, 3 existing skips. The full check requires unsandboxed process-control access to `ps` and `/usr/bin/time`; the sandboxed attempt failed those unrelated release tests.
- 115 targeted embedding/native/smoke tests. Native fixtures exercise 59 mixed/export/legacy cases, 6 pre-extended-batch cases and a bounded-address-space multiplication overflow case.
- Round-4 auditor mutations: 5/5 applicable `mut2` projector bypasses and 24/24 applicable `mutate` cases fail after adapting anchors; 29 cases target deleted machinery and are retired, not counted as kills. Original scripts were also run to inventory stale anchors.
- Round-4 own mutations: 21/21 fail, with an unmodified 115-test control passing. The three previous P3-3 CPU-policy guard/warning/note survivors are deleted; the surviving projector-loaded guard bypass now fails repeated-media regression tests.
- All five audit-probe3 behavioral probes pass. Only `lazy_fail_probe.mjs` changes setup: install the failing projector stub before string fallback because restoration is now eager; it also asserts the typed error, exactly two attempts across two calls, model identity and remembered sources. The other probes run unchanged. Main/head comparisons match model load options/warnings, all media-detection cases and lowercase/uppercase generation fallback. The new eager projector load is the intentional string-fallback difference.
- CI reliability and whitespace: PASS. Generated JS/declarations are regenerated by `check:js`.
- Round-4 Chromium state/extraction/generation: direct and worker on wasm32 and memory64, all four rows pass. Qwen3.5 image completion: direct and worker wasm32, both rows pass. These use the round-4 JS wrapper over existing round-2 WASM and existing pinned local weights; no downloads.
- Local v0.7.0 round-2 builds: wasm32 and memory64, Emscripten 6.0.11; pins unchanged. No round-4 WASM rebuild or BF16 rerun: runtime embedding staging/execution and C/C++ are unchanged. The runtime load edit only removes round 3's private default override and restores the ordinary main behavior. BF16 evidence below is explicitly from round 2.
- Round-1 actual v0.5.0 C++ core syntax check passed with mixed API disabled; current old-core executable contracts pass.

### BF16 matrix

The full expanded input uses context/batch/microbatch 2048, four CPU threads and MEAN noncausal pooling. The browser selects memory64 explicitly, with the compiled 12 GiB linear-memory ceiling; no BF16 OOM or quantized substitution occurred. This is CPU/WASM qualification, not WebGPU GPU-kernel qualification.

Acceptance requires **cosine >= 0.9998 AND relative L2 < 2% per input/output against native CPU**, plus **exact direct/worker parity** and **exact legacy string-path parity against the pre-task bridge**. The same cosine/relative-L2 criteria apply to the measured raw vectors. Per-component max-absolute differences remain informational: BF16 wasm SIMD and native NEON reductions need a vector-level comparison.

The independent auditor's injected structural defects (reversed or dropped media rows, swapped images and a dropped token) produced 1-cos gaps at least 15–20 times the observed bridge/native gap. This supports a bar sensitive to input-structure defects without classifying the existing cross-ISA variation as a regression. All 15 cross-input cosine comparisons are also recorded by the smoke; their maximum native difference is `0.000975830487` (< 1e-3).

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

The round-2 smoke exits `0`: all native vector criteria, exact direct/worker parity (normalized and raw), baseline string parity and functional checks pass. Artifact: `/private/tmp/llamadart-015-d-bridge-tmp/embedding-round2/multimodal-embeddings-result.json`; the existing default native CPU reference is `reference-native-default.json` in the task directory. Reference rebuilding was unnecessary because the inputs and upstream-only algorithm did not change.

CPU precision controls (without Accelerate/llamafile, portable ARMv8 and flash-attention off) did not eliminate the arithmetic gap; the exact kernel-level cause remains unproven. The pre-task v0.7.0 bridge reproduces the current text vector exactly in both runtimes, normalized and raw. Its generated wrapper is byte-identical to base `57d9971` (SHA-256 `994a81c48c0f4c43d39bed98a671e873115d8a8eb1be93c0ea62a09aff9834ad`); runtime source/CMake differences between its source `4553c48` and `57d9971` are empty. The smoke now requires these saved baseline vectors and checks exact string parity on every run. This local baseline is not a published-asset attestation.

Bare `cancel()` is best effort. The real smoke accepts a completed, unchanged vector when work finishes without observing cancellation, or `AbortError` when it observes cancellation; either way it waits for settlement and verifies subsequent text reuse plus projector unload/reload. Unit fixtures additionally force cancellation during staging and native inference.

## Mutation evidence

Each mutation bypasses a guard or breaks an ownership/lifecycle invariant in an isolated source copy, then runs the JS or native embedding regression suite. The earlier rounds recorded **116/116** failures at their historical heads; those totals include the now-deleted round-3 CPU/lazy design and are not claimed for round 4. The table retains the applicable earlier guard inventory and adds current recovery cases. Round-4 results are **29/29 applicable auditor mutations** and **21/21 own mutations** killed, with passing unmodified controls. Capacity limits are mutated individually. The native stub copies rows in add_embd, matching upstream set_token_embd, and deliberately reuses the mtmd output buffer. The obsolete pointer-retention mutation was removed. A bounded-address-space fixture checks multiplication overflow independently of host pointer width.

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
| `model-null-context` | mixed native batch guards, copied media, output width, one process and cleanup |
| `model-null-model` | mixed native batch guards, copied media, output width, one process and cleanup |
| `pooled-output-width` | mixed native batch guards, copied media, output width, one process and cleanup |
| `encoder-decoder` | mixed native batch guards, copied media, output width, one process and cleanup |
| `pool-none` | mixed native batch guards, copied media, output width, one process and cleanup |
| `pool-rank` | mixed native batch guards, copied media, output width, one process and cleanup |
| `mixed-encoder` | mixed native batch guards, copied media, output width, one process and cleanup |
| `mixed-mean` | mixed native batch guards, copied media, output width, one process and cleanup |
| `mixed-causal` | mixed native batch guards, copied media, output width, one process and cleanup |
| `native-model-guard` | mixed native batch guards, copied media, output width, one process and cleanup |
| `native-projector` | mixed native batch guards, copied media, output width, one process and cleanup |
| `native-mixed-guard` | mixed native batch guards, copied media, output width, one process and cleanup |
| `native-modality` | mixed native batch guards, copied media, output width, one process and cleanup |
| `native-vision` | mixed native batch guards, copied media, output width, one process and cleanup |
| `chunks-allocation` | mixed native batch guards, copied media, output width, one process and cleanup |
| `tokenize-status` | mixed native batch guards, copied media, output width, one process and cleanup |
| `expanded-count-overflow` | mixed native batch guards, copied media, output width, one process and cleanup |
| `token-capacity-query` | mixed native batch guards, copied media, output width, one process and cleanup |
| `token-result` | mixed native batch guards, copied media, output width, one process and cleanup |
| `expanded-empty` | mixed native batch guards, copied media, output width, one process and cleanup |
| `allocated-context` | mixed native batch guards, copied media, output width, one process and cleanup |
| `allocated-sequence-context` | mixed native batch guards, copied media, output width, one process and cleanup |
| `allocated-batch` | mixed native batch guards, copied media, output width, one process and cleanup |
| `allocated-ubatch` | mixed native batch guards, copied media, output width, one process and cleanup |
| `allocated-sequence-count` | mixed native batch guards, copied media, output width, one process and cleanup |
| `input-width` | mixed native batch guards, copied media, output width, one process and cleanup |
| `batch-allocation` | mixed native batch guards, copied media, output width, one process and cleanup |
| `batch-append` | mixed native batch guards, copied media, output width, one process and cleanup |
| `batch-position` | mixed native batch guards, copied media, output width, one process and cleanup |
| `batch-output` | mixed native batch guards, copied media, output width, one process and cleanup |
| `text-chunk-pointer` | mixed native batch guards, copied media, output width, one process and cleanup |
| `encode-status` | mixed native batch guards, copied media, output width, one process and cleanup |
| `encode-output` | mixed native batch guards, copied media, output width, one process and cleanup |
| `process-status` | mixed native batch guards, copied media, output width, one process and cleanup |
| `pooled-pointer` | mixed native batch guards, copied media, output width, one process and cleanup |
| `embedding-mode-cleanup` | mixed native batch guards, copied media, output width, one process and cleanup |
| `text-batch-allocation` | mixed native batch guards, copied media, output width, one process and cleanup |
| `text-process-status` | mixed native batch guards, copied media, output width, one process and cleanup |
| `text-capacity-guard` | mixed native batch guards, copied media, output width, one process and cleanup |
| `mixed-capacity-guard` | mixed native batch guards, copied media, output width, one process and cleanup |
| `text-batch-cleanup` | mixed native batch guards, copied media, output width, one process and cleanup |
| `native-active-preflight` | mixed native batch guards, copied media, output width, one process and cleanup |
| `native-media-cleanup` | mixed native batch guards, copied media, output width, one process and cleanup |
| `native-cancel-cleanup` | mixed native batch guards, copied media, output width, one process and cleanup |
| `embedding-return-code` | microbatch overflow is a typed inference error and cleans media |
| `parts-batch-preflight` | unknown probe version |
| `unloaded-capability` | unloaded model capabilities are all false without core IO |
| `normalization-option` | mixed native batch guards, copied media, output width, one process and cleanup |
| `worker-host-error-code` | worker host posts payload-free typed embedding error codes |
| `worker-error-class` | real proxy reconstructs typed embedding errors |
| `no-prompt-separator` | parts form one ordered vector with verbatim adjacent text and no prompt decoration |
| `media-element-overflow` | media element multiplication rejects overflow in a bounded address space |
| `token-query-int32-min` | mixed native batch guards, copied media, output width, one process and cleanup |
| `native-audio-support` | mixed native batch guards, copied media, output width, one process and cleanup |
| `native-audio-rate` | mixed native batch guards, copied media, output width, one process and cleanup |
| `legacy-cancel-reset` | fresh string inference after prior JS staging cancellation |
| `batch-item-abort` | batch cancellation between legacy string items prevents the next inference |
| `parts-abort-reset` | fresh parts embed clears a prior runtime abort |
| `mrope-projector` | native capabilities and inference reject multi-position projectors |
| `utf8-buffer` | exact UTF-8 bytes for lone surrogates, CJK and NUL |
| `core-abort-signal` | parts core abort retains the worker recovery signal |
| `legacy-string-coercion` | non-array embedding inputs retain legacy String coercion |
| `disposed-empty-batch` | empty batch still rejects on a disposed bridge |
| `asset-version` | capability error reports loaded version without URL secrets |
| `probe-crash-masking` | embed/batch/capabilities recover worker crash, timeout and core abort |
| `fallback-projector` | probe-time and inference-time crashes restore projector |
| `active-core-version` | capability error reports actual wasm32/wasm64 asset |
| `qualification-cosine` | independently observable cosine threshold |
| `qualification-relative-l2` | relative L2 threshold |
| `qualification-worker-exact` | exact normalized direct/worker harness guard |
| `qualification-worker-raw-exact` | exact raw direct/worker harness guard |
| `qualification-string-baseline` | exact pre-task normalized/raw string harness guard |
| `capability-direct-guard` | direct RuntimeError never enters worker recovery |
| `core-capability-code` | native -20 reports capability-unavailable and active asset version |
| `baseline-guard-wrapper` | rendered baseline guard throws on normalized/raw drift independently |
| `eager-projector-off` / `ensure-never` | every embedding fallback eagerly restores a remembered projector |
| `projector-loaded-guard-off` / `lazy-no-mmproj-check` | repeated media/probes reuse a loaded projector |
| `restore-error-fatal` | failed restore keeps model usable for string/text-only embedding |
| `failure-forgets-model` / `failure-disposes-runtime` | preserved model source and runtime identity after failure |
| `failure-forgets-projector` | remembered projector remains retryable |
| `failure-warning-off` / `failure-warning-leaks` | restore warns without exposing URL/error payload |
| `caps-no-retry` | capabilities retry and switch truthfully from false to true |
| `restore-twice` / `retry-after-failure-loop` | one projector restore per call |
| `text-retries-projector` / `audio-skipped` / `image-skipped` | only direct media embedding retries; both modalities covered |
| `generation-uppercase-rejected` | generation uppercase image/audio policy equals main |
| `embedding-cpu-policy-back` / `gpu-forced-zero` / `context-reduced` / `context-invented` | embedding load options match main, including omitted/zero defaults |
| `in-place-model-reload-forced` | main's healthy direct-model reuse behavior |
| `timeout-warning-off` | main's exact existing timeout warning |

Round-2 additions cover legacy native cancellation reset, the per-item batch abort check, fresh parts abort reset, M-RoPE rejection, exact UTF-8 buffers, core-abort propagation, legacy coercion, disposed empty batches, active asset-version diagnostics, probe crash recovery and projector reload. Each is independently bypassed and must fail its regression.

Raw mutation records and failing outputs: `/private/tmp/llamadart-015-d-bridge-tmp/mutations-round2/results.json` and neighboring `<guard>.log` files.

Round-4 auditor inventory: original `mut2.mjs` has 28 stale anchors: 5 surviving projector guards are adapted to their new locations, and 23 CPU-policy/flag/hint/note/warning/selective-generation cases are retired with the deleted design. Original `mutate.mjs` kills 23 cases and has 7 stale anchors: 1 capability eager-restore bypass is adapted and killed, while 6 cases depend on removed fallback options/parts/flags and are retired. Adaptations preserve the remaining bypasses; deleted code is not counted as tested. The own mutation suite additionally covers fatal restore errors, discarded sources/runtime, warning leakage, capability retries, duplicate attempts, both media modalities, unchanged model options and main's in-place/timeout behavior.

Evidence lives under `/private/tmp/llamadart-015-d-bridge-tmp`: `round4-check-unsandboxed.log`, `round4-targeted.log`, `round4-ci.log`, `round4-{mut2,mutate}-{original,adapted}.log`, `round4-own-mutations.log`, `round4/mutations/results.json`, `round4/*` adapted scripts, `round4-{lazy,caps}-probe.log`, main/head comparison logs and `round4-{state-wasm32,state-wasm64,multimodal}.log`. The round-2 native and BF16 records remain the unchanged numerical evidence.

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
