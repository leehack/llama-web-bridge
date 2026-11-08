# llama.cpp v0.9.0 compatibility preparation

The candidate lane builds upstream `cd959bae99ff4bed22bef2604f45241c09bbb8c8`
(v0.9.0) alongside the unchanged v0.5.0 development pin, matching published
native `v0.9.0`. The bridge source needs no change for v0.9.0. No Web assets or
Dart packages are published by this change.

## Upstream and native patches

- The v0.7.0 embedding-mode reserve repair
  ([llamadart-native#114](https://github.com/leehack/llamadart-native/issues/114),
  [v0.7.0 notes](v070_qualification.md)) is upstream since v0.8.0:
  `set_embeddings` re-reserves on every toggle. `cmake/embedding_mode_reserve.cmake`
  finds no affected guard and stays inactive.
- The other native patches do not apply to the Web build: 0001/0002 are
  Vulkan, 0004 Metal, and 0005/0006 (F16 dot and GEMM F32 accumulation,
  [llamadart-native#174](https://github.com/leehack/llamadart-native/issues/174))
  change code only under `__ARM_NEON`/`__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`,
  which WASM never defines.
- The mtmd-audio single-thread rewrite still finds both `n_threads` markers.
- v0.9.0 drops `mtmd_helper_gen_audio_inp::voice` and adds
  `mtmd_gen_audio_info::n_voices` and Chatterbox voices. The bridge TTS path
  sets neither field, so Qwen3-TTS, with or without a speaker reference, is
  unaffected.

## Validation

Local evidence uses Emscripten 6.0.8, Node 24 and the locked Playwright
Chromium on macOS arm64, against a clean upstream checkout at the commit above.
Model/projector pins and commands live in
[CONTRIBUTING.md](../CONTRIBUTING.md#validate-outputs).

| Check | Result |
| --- | --- |
| `npm run check:js` | 602 passed, 4 existing skips, zero failures |
| CI reliability contract | PASS |
| v0.9.0 production build | PASS, wasm32 and memory64 |
| Extraction/state/generation smoke | PASS, direct and worker on both memory modes |
| Qwen3.5 image inference | PASS, direct and worker on wasm32 |
| Tiny and Qwen3.5 grammar, next-token scores, tiny speculative | PASS |
| Qwen3-ASR transcript, silence and cancellation | PASS, four runtime rows |
| Qwen3-TTS synthesis, cancellation and unload, WebGPU offload | PASS, four runtime rows with a speaker reference; direct and worker on memory64 without one |

Four runtime rows mean direct and worker execution on wasm32 and memory64.
Hosted exact-head CI, candidate qualification, publication approval and Dart
adoption remain separate gates.
