# llama.cpp v1.0.0 qualification

For this qualification, `llama_cpp.version` pinned `v1.0.0`. CI verified that the cloned tag resolved to
`b1b770d90d94cad9e9ea5b993bdbc383a522fb28` before building the threaded and
single-thread cores for both wasm32 and memory64.

Browser smokes include state persistence on all four cores, multimodal
completion, grammar, next-token scoring, and speculative decoding. Local
commands and fixture pins are in
[CONTRIBUTING.md](../CONTRIBUTING.md#validate-outputs).

## Multimodal compatibility

The bridge now links upstream's `mtmd` target so the source list follows the
selected llama.cpp release. On v1.0.0, the projector's persistent thread pool
holds workers needed by text decode within the same four-worker Emscripten
pool. The bridge builds a generated copy of `mtmd-subcontext.cpp` that uses
per-graph workers and preserves the configured thread count.

With the uncorrected v1.0.0 build and the pinned Qwen3.5 model and projector,
direct wasm32 image completion stalled at four threads and exceeded a
60-second host deadline. At one and three threads, the same build completed
in both direct and worker wasm32 runtimes. The corrected build completes at
four threads in direct and worker runtimes on wasm32 and memory64.

## Local verification

The v1.0.0 build and smokes ran on macOS arm64 with Emscripten 6.0.8, Node
24.14.1, and the locked Playwright Chromium 153.0.8010.12. The smoke bundle
was byte-identical to the output of the standard build script.

| Check | Result |
| --- | --- |
| JS and static contracts | 606 passed, 4 skipped |
| v1.0.0 production build | wasm32 and memory64 passed |
| Extraction, generation, and state restore | Tiny model passed in all four runtime rows |
| Qwen3.5 image completion | Pinned model and projector passed at four CPU threads in all four runtime rows |
| Grammar | Tiny model and Qwen3.5 each passed in all four runtime rows |
| Next-token scoring | Tiny model passed in all four runtime rows |
| Tiny speculative decoding | All four runtime rows passed |

The four runtime rows are direct and worker on each of wasm32 and memory64.

## Release gates

CI uploads diagnostic and build artifacts. A publishable candidate requires a
published stable `llamadart-native` manifest that names the same upstream tag
and commit. The owner then dispatches the exact Web candidate build, hosted
qualification, and publication in order. Hosted qualification checks ASR on
wasm32 and memory64, and TTS on memory64. Each supported combination runs
through direct and worker runtimes. Dart adoption uses the published bridge
asset tag. See the
[publish process](../CONTRIBUTING.md#publish-process) for the required inputs.
