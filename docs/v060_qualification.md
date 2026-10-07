# llama.cpp v0.6.0 compatibility preparation

This change prepares the bridge to compile with the v0.6.0 candidate while
retaining the production v0.5.0 source pin. It does not publish assets or change
llamadart consumer pins. The exact candidate commit is asserted in
`.github/workflows/ci.yml`; Emscripten remains pinned to 6.0.8.

## Compatibility boundary

Upstream v0.6.0 removes `common_batch_clear`/`common_batch_add` and changes
`common_speculative_process` from `llama_batch` to `const common_batch &`.
The bridge owns explicit text batches for sequence zero. Its local helpers
preserve positions, token IDs and output flags, including upstream's null
sequence-pointer capacity sentinel assertion before any write or counter
increment. A C++17 declaration probe selects the process adapter; the probe
cache is reset on configuration so switching sources cannot retain a stale
result. This compatibility change adds no upstream source patches.

The new process adapter rejects negative counts, embeddings, missing explicit
metadata and nonzero or multiple sequences before forwarding. The existing
v0.5.0 declaration still receives the original batch. Both paths propagate
upstream failure. Prompt, speculative draft and replay generation call the
same bridge helpers.

The compiled regression test exercises both declarations, exact metadata,
empty and reused batches, upstream failures, malformed layouts and overflow.
The overflow test verifies both the unchanged count and untouched canaries.
Production-wiring checks cover the include, adapter calls and CMake probe.

## Local validation

Validated on macOS with the repository's locked Playwright Chromium and
Emscripten 6.0.8. Model checksums and runnable smoke recipes remain owned by
`CONTRIBUTING.md` and the existing smoke runners. Build/cache/model/diagnostic
files stayed outside the checkout.

| Check | Result |
| --- | --- |
| `npm run check:js` | 613 passed, 3 existing skips, zero failures |
| CI reliability contract, actionlint, diff whitespace | PASS |
| v0.5.0 and v0.6.0 production builds | PASS, wasm32 and memory64 for both |
| v0.6.0 full speculative CPU suite | PASS, six groups, 24 runtime rows, 76 completions and 24 rejection/recovery checks |
| Final guarded v0.6.0 tiny speculative CPU suite | PASS, four runtime rows, 28 completions and 8 rejection/recovery checks |
| Final guarded v0.5.0 tiny speculative CPU suite | PASS, four runtime rows, 28 completions and 8 rejection/recovery checks |
| Final guarded v0.6.0 tiny speculative WebGPU suite | PASS, four runtime rows, 28 completions and 8 rejection/recovery checks |
| Final guarded v0.6.0 SmolLM2 speculative WebGPU suite | PASS, four runtime rows, 28 completions and 4 rejection/recovery checks |
| Final guarded v0.6.0 tiny grammar, sampling and thinking controls | PASS, four runtime rows |
| Final guarded v0.6.0 Qwen3.5 grammar, sampling and thinking controls | PASS, four runtime rows |
| Final guarded v0.6.0 next-token scoring and invalid requests | PASS, four runtime rows |
| v0.6.0 state persistence and finite embeddings | PASS, direct and worker |
| v0.6.0 Qwen3.5 image inference | PASS, direct and worker, wasm32 |

Four runtime rows mean direct and worker execution on both wasm32 and memory64.
Speculative suites compare exact greedy text and token output with the baseline,
verify draft/accept/replay usage where required, and reject silent worker
fallback. The six CPU groups are tiny, SmolLM2, EAGLE3, MTP, DFlash and recurrent
DSpark. These are targeted bridge checks, not new public model-family claims.

SmolLM2 WebGPU logs confirmed actual transformer offloading: 31/31 target and
6/6 draft layers. Backend reports were `WebGPU, CPU`, with no worker fallback.
Tiny GPU evidence alone is not transformer qualification. The Qwen3.5 image
smoke produced the same 13-token caption through both runtimes and exercised
the resize path; this does not establish memory64 multimodal coverage.

The full CPU suite, image and state checks preceded the capacity assertion
repair. Final guarded builds were regenerated for both upstream versions and
memory modes, and the tiny CPU and both GPU suites ran against those builds.
The assertion changes only invalid over-capacity writes.

## Reproduction and remaining gates

Use the external-path build recipe in `CONTRIBUTING.md`, selecting a verified
local source directory with `LLAMA_CPP_DIR` and separate output/build paths for
each upstream. Run `scripts/smoke/speculative.mjs --group all` for the six CPU
groups, then the `tiny` and `smollm2` groups with `--gpu-layers 999` for GPU
checks. Use the existing state, multimodal, grammar and next-token scoring
commands with the pinned models; do not introduce one-off repro scripts.

CI builds and runs the normal browser smokes independently for the production
pin and exact candidate. The pinned check/artifact identities are preserved;
candidate identities carry the v0.6.0 suffix, and the aggregate gate requires
both selected lanes. Local passing results do not replace exact-head CI.

Hosted immutable-artifact release qualification, heavy ASR/TTS gates, asset
publication and downstream llamadart integration remain separate work. A
consumer pin update still requires published matching native/Web artifacts,
binding synchronization where needed and model-backed consumer checks.

Independent blocking-only review against base `0978c4b` found zero known
PR-caused P1 regressions. It independently reran the compiled adapter tests,
CI reliability contract and whitespace checks. No PR or review threads existed
at local preparation time; exact-head hosted CI and thread closure remain gates.
