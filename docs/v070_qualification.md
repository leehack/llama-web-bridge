# llama.cpp v0.7.0 compatibility preparation

The candidate lane builds upstream `620d2225cc436ab0b001e3200e590e3b286ef9ab`
(v0.7.0) alongside the unchanged v0.5.0 development pin. This prepares matching
Web qualification for published native v0.7.0. No Web assets or Dart packages
are published by this change.

## Extraction regression and repair

The unmodified v0.7.0 Web build compiled successfully, then aborted at
`ggml-backend.cpp:273` (`GGML_ASSERT(buffer)`) in the real tiny-model state
smoke. Enabling embeddings on an unpooled generation context changes graph
allocation after upstream PR #30445, but `llama_context::set_embeddings`
reserves only pooled contexts. The native release carries the same repair
under [llamadart-native#114](https://github.com/leehack/llamadart-native/issues/114).
Web text embeddings and text-to-speech both toggle extraction on generation
contexts, so a source pin alone is unsafe.

`cmake/embedding_mode_reserve.cmake` recognizes the affected guard and requires
SHA-256 `2a25e556cb0d16725e01ce853fb1b76da91dcb41b803ce1a6bacec89bfb94aa5`
for the exact upstream `src/llama-context.cpp`. It compiles a generated copy
that reserves whenever embedding extraction changes, including unpooled
contexts. The source checkout stays untouched. An affected file with another
fingerprint fails configuration pending review. The module remains inactive
for sources without the affected guard, including the production pin.

The existing real-model state smoke is the positive regression: it failed on
the unmodified build and passes with the repair. It extracts once and in a
batch, then generates, saves/restores state, reuses prompts and cancels
streaming. CI and the candidate workflow now require this smoke on both wasm32
and memory64, with direct/worker metadata assertions rejecting silent
fallback. The negative contract rejects unaudited affected source before any
replacement is emitted. Harness 5.1.0 binds these stronger gates.

## Validation

Local evidence uses Emscripten 6.0.8, Node 24 and the locked Playwright
Chromium on macOS arm64. Build, cache, model and diagnostic files remain outside
the checkout. Model/projector pins and maintained commands live in
[CONTRIBUTING.md](../CONTRIBUTING.md#validate-outputs).

| Check | Result |
| --- | --- |
| `npm run check:js` | 616 passed, 3 existing skips, zero failures |
| CI reliability contract, actionlint, whitespace | PASS |
| v0.7.0 production build | PASS, wasm32 and memory64 |
| v0.7.0 extraction/state/generation regression | PASS, direct and worker on both memory modes |
| v0.7.0 Qwen3.5 image inference | PASS, direct and worker on wasm32 |
| v0.7.0 tiny and Qwen3.5 grammar/sampling/thinking controls | PASS, four runtime rows per model |
| v0.7.0 next-token scoring and invalid requests | PASS, four runtime rows |
| v0.7.0 tiny, SmolLM2, MTP and recurrent DSpark speculative parity | PASS, four runtime rows per group |
| v0.7.0 SmolLM2 WebGPU speculative parity | PASS, four runtime rows; 31/31 target and 6/6 draft layers offloaded |
| EAGLE3 and DFlash converted draft fixtures | Unavailable locally; no real-model pass claimed |
| v0.5.0 unchanged-pin control | PASS, both builds, both state/extraction memory modes and wasm32 image inference through direct/worker |

Four runtime rows mean direct and worker execution on wasm32 and memory64.
Speculative checks require exact greedy text/token parity and rejection/recovery
behavior; they do not measure speedup. The image check is not memory64 image
qualification.

An independent reviewer found zero blocking findings against base
`c1f64536b9a4ea410066b711928c7cba03691fb0`. It independently ran 88 targeted
tests and the reliability/whitespace gates, verified the source fingerprint and
unchanged source checkout, and configured an isolated unity-build target to
prove that only the corrected copy compiles as its own translation unit.
The final old-pin control passed after that review. No PR or review threads
exist at local preparation time; exact-head hosted CI remains required.

Hosted exact-head CI, immutable candidate artifact qualification (including
heavy ASR/TTS), Web publication approval, published-byte verification and Dart
adoption remain separate gates. EAGLE3 and DFlash converted draft fixtures are
currently unavailable locally; no real-model qualification is claimed for
those families. This change makes no new public model-family or GPU-performance
claim, and does not change the Dart native/Web parity gate.
