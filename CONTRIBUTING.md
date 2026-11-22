# Contributing to llama-web-bridge

Thanks for contributing.

## Purpose

This repo builds the reusable JS/WASM bridge runtime for llama.cpp web usage.
Versioned browser assets are published to `llama-web-bridge-assets`.
`llamadart` owns the loading and fallback behavior for its integration.

## Prerequisites

- Emscripten SDK (`emcmake`, `emcc`) matching `emsdk.version`
- Node.js 22.18+ or 24.2+ (CI uses 24) and npm, for JS bridge bundling and
  type-checking; the tests run the `.ts` sources through Node's built-in type
  stripping
- CMake toolchain
- A host C++17 compiler (`c++`, or `$CXX`): `npm run check:js` compiles the
  media-helper compatibility contract with it
- Access to a llama.cpp checkout matching `llama_cpp.version`

## Setup

```bash
git clone https://github.com/leehack/llama-web-bridge.git
cd llama-web-bridge
npm ci
./scripts/build/build_bridge.sh --help
```

## Local Build

```bash
npm run check:js
./scripts/build/build_bridge.sh
# or
LLAMA_CPP_DIR=../llama.cpp OUT_DIR=dist ./scripts/build/build_bridge.sh
```

`./scripts/build/build_bridge.sh --help` is the complete list of environment
variables the build reads, with their defaults; the docs do not repeat the
list.

Bridge wrapper source lives under `js/src/`; `npm run build:js` regenerates the
checked-in browser ESM outputs and declarations under `js/`. `npm run check:js`
runs the same generator plus TypeScript and syntax checks, so commit any updated
`js/` outputs after source changes. It ends with `npm test`, which runs every
`tests/**/*_test.mjs` contract test in parallel through Node's test runner,
including the static state-persistence, text-to-speech, and decision API
contracts, the media-helper compatibility contract, the wasm64 runtime patch
and memory64 contract for `scripts/build/patch_wasm64_runtime.mjs`, and the CI
change selector.
Run one test file directly with `node tests/<area>/<name>_test.mjs`.

`js/src/llama_webgpu_bridge.js` is the public entry. It re-exports the API,
supports worker host auto-boot, and sets the `window.LlamaWebGpuBridge` global.
`js/src/llama_webgpu_bridge_worker.js` installs the worker host when loaded.
`bridge.ts` is the facade, `runtime.ts` the direct runtime, `worker_proxy.ts`,
`worker_host.ts`, and `worker_protocol.ts` the worker path, and `internal/`
holds shared helpers, with internal-only types in `internal/types.ts`.

Every module except the two entries (`llama_webgpu_bridge.js` and
`llama_webgpu_bridge_worker.js`, which are bundled separately) is TypeScript,
type-checked with `strict` (`tsconfig.strict.json`). The entries keep the
lenient `checkJs` pass (`tsconfig.bridge.json`), and `npm run typecheck:js` runs
both. `LlamaWebGpuBridge` implements the published `llama_webgpu_bridge.d.ts`
class, and `public_api_check.ts` (types only, never bundled) compares every
public method with strict parameter variance, so the compiler rejects an
implementation that accepts less or returns more than the declared API.
TypeScript here is limited to erasable syntax (`erasableSyntaxOnly`): types,
`import type`, `declare` fields, and casts only, no enums, namespaces, or
parameter properties. A class field without an initializer is `declare`d,
because a plain field declaration emits code;
`tests/bridge/declared_class_fields_test.mjs` enforces that. esbuild and Node both
strip the types without changing the code, so the tests run the `.ts` sources
directly. A type change must not change behaviour: with comments and whitespace
stripped, the bundle stays byte-identical. Import modules by their `.ts` path.

Keep `default_assets.ts` beside the entry modules so its static asset paths
resolve relative to each generated bundle.

The native core is one translation unit. `src/llama_webgpu_core.cpp` holds the
headers, the anonymous namespace, the `extern "C"` block with
`llamadart_webgpu_shutdown`, and `main`, and includes its parts from `src/core/`
inside the namespace and the block. The `exports_*.inc` parts hold the other
exported `llamadart_webgpu_*` functions grouped by feature; the remaining parts
hold the state and internal helpers they use. A part is not a standalone file:
it relies on everything included before it, so keep the include order. The
static contract checks read the core with its parts expanded
(`tests/bridge/native_core_source.mjs`), which is how the compiler sees it. It fails
if a part is not included exactly once as a plain `#include` line. The JS API
contract tests read the bridge the same way: `tests/bridge/bridge_js_source.mjs`
joins every `js/src` module in the order the former single-file source declared
them.

Growable cores link with `--wrap=mmap` and `--wrap=munmap` so llama.cpp can
map a heap-backed file directly. JS model FS ABI version 4 adds
`llamadart_webgpu_js_model_fs_supports_heap_resident` so fixed-memory builds
continue to use JS chunks. It also adds
`llamadart_webgpu_js_model_fs_open_direct_heap` and the
`llamadart_webgpu_js_model_fs_heap_resident_bytes` diagnostic. Production
cores also export `llamadart_webgpu_test_fail_next_model_load` for the
same-core load failure smoke.

For local agent/maintainer validation, prefer external build and cache paths so
generated files do not dirty the checkout:

```bash
export CCACHE_DIR=/private/tmp/llama_web_bridge_ccache
export EM_CACHE=/private/tmp/llama_web_bridge_emcache
BUILD_DIR=/private/tmp/llama_web_bridge_build \
MEM64_BUILD_DIR=/private/tmp/llama_web_bridge_build_mem64 \
SINGLE_THREAD_BUILD_DIR=/private/tmp/llama_web_bridge_build_st \
SINGLE_THREAD_MEM64_BUILD_DIR=/private/tmp/llama_web_bridge_build_mem64_st \
OUT_DIR=/private/tmp/llama_web_bridge_dist \
WEBGPU_BRIDGE_BUILD_MEM64=1 \
WEBGPU_BRIDGE_BUILD_SINGLE_THREAD=1 \
./scripts/build/build_bridge.sh
```

## Validate Outputs

### Expected build outputs

The build writes these files to `OUT_DIR`.

- `llama_webgpu_bridge.js`
- `llama_webgpu_bridge_worker.js`
- `llama_webgpu_bridge.d.ts`
- `llama_webgpu_core.js`
- `llama_webgpu_core.wasm`
- `llama_webgpu_core_mem64.js` (only with `WEBGPU_BRIDGE_BUILD_MEM64=1`, as
  above and in CI)
- `llama_webgpu_core_mem64.wasm` (same condition)
- `llama_webgpu_core_st.js` (only with `WEBGPU_BRIDGE_BUILD_SINGLE_THREAD=1`,
  as above and in CI)
- `llama_webgpu_core_st.wasm` (same condition)
- `llama_webgpu_core_mem64_st.js` (only with both
  `WEBGPU_BRIDGE_BUILD_SINGLE_THREAD=1` and `WEBGPU_BRIDGE_BUILD_MEM64=1`)
- `llama_webgpu_core_mem64_st.wasm` (same condition)

### Lightweight contracts

Before opening or updating a PR, run the lightweight contracts:

```bash
npm run check:js
node scripts/ci/verify_ci_reliability.mjs
```

### Core, state, and runtime smokes

Browser smokes use Node and the locked `playwright` dev dependency. After
`npm ci --ignore-scripts`, install Chromium once with
`npx --no-install playwright install --only-shell chromium`. Node's `fetch`
ignores `HTTP(S)_PROXY` unless `NODE_USE_ENV_PROXY=1` is set.

For state-persistence, worker, or workflow changes, also run the browser smoke
against a built dist directory. Point `OUT_DIR` at the build output for the
commands below. Use a checksum-pinned tiny model and keep caches and artifacts
outside the repository.

```bash
node scripts/smoke/state_persistence.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-url https://huggingface.co/aladar/llama-2-tiny-random-GGUF/resolve/main/llama-2-tiny-random.gguf \
  --model-sha256 81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb \
  --model-cache-dir ~/.cache/llama-web-bridge/state-smoke-models \
  --artifacts-dir /tmp/llama-web-bridge-state-smoke
```

Run the same state smoke with `--memory-mode wasm64` as well. Both runs
exercise extraction toggles, generation and state restore through direct and
worker paths; metadata assertions reject memory-mode or worker fallback. CI
and candidate qualification require both runs.

To validate the single-thread `_st` cores on a non-isolated page
(`crossOriginIsolated === false` and no `SharedArrayBuffer`):

```bash
node scripts/smoke/state_persistence.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --threading single \
  --model-url "$LLAMA_WEBGPU_SMOKE_MODEL_URL" \
  --model-sha256 "$LLAMA_WEBGPU_SMOKE_MODEL_SHA256" \
  --model-cache-dir ~/.cache/llama-web-bridge/state-smoke-models \
  --artifacts-dir /tmp/llama-web-bridge-state-smoke-st
```

Run the same single-thread smoke with `--memory-mode wasm64` as well. Both
non-isolated runs check that the worker rejects a forced fetch-backend load,
and that auto mode stages the model. CI and candidate qualification require both runs.

For worker startup or core URL changes, use the same checksum-pinned model:

```bash
node scripts/smoke/cross_origin_assets.mjs \
  --dist-dir "$OUT_DIR" \
  --model-url "$LLAMA_WEBGPU_SMOKE_MODEL_URL" \
  --model-sha256 "$LLAMA_WEBGPU_SMOKE_MODEL_SHA256"
```

Rerun with `--memory-mode wasm64` for memory64 and `--browser firefox`
with Playwright Firefox installed.

The resume integrity smoke uses the same checksum-pinned model as state
persistence and downloads it on a cache miss. Set the model URL, SHA-256 and
cache directory to the same values as the state smoke and run:

```bash
node scripts/smoke/resume_integrity.mjs --dist-dir "$OUT_DIR" --baseline-dist-dir "$OUT_DIR"
```

Repeat with `--threading single`, `--js-backed-model-staging`, and both flags
together. Repeat the JS-backed cases with `--no-mmap` to exercise JS chunks.
The baseline flag enables direct and worker comparisons for the
control cases. CI passes the built dist as its own baseline. To compare two
revisions locally, pass a separately built dist as the baseline.

For core disposal or retirement changes, use the same state-smoke model URL,
SHA-256, and cache directory to check that Chromium can collect the old core
after disposal or a native abort:

```bash
node scripts/smoke/core_release.mjs --dist-dir "$OUT_DIR"
```

Repeat with `--memory-mode wasm64`, `--threading single`, and both flags together.

For native abort recovery, use the same model environment variables. The
default cases include `staged-load-no-mmap` to cover retirement of JS-chunk
staging mounts.

```bash
node scripts/smoke/core_abort_recovery.mjs --dist-dir "$OUT_DIR"
node scripts/smoke/core_abort_recovery.mjs --dist-dir "$OUT_DIR" --threading single
```

To check collection while a stale decision handle is still live, pass a local
Laya encoder and a head with embedded `laya.config` metadata:

```bash
node scripts/smoke/core_release.mjs --dist-dir "$OUT_DIR" --case decision-abort \
  --decision-model-path /path/to/laya-Q8_0.gguf \
  --decision-head-path /path/to/laya-head.safetensors
```

The WasmFS remote-fetch smoke uses the same pinned model to check six consecutive
fetch-backed loads on one core, varying the URLs and chunk sizes with
`nThreads: 4`. It verifies inference and recovery after a native abort in both
application-owned and bridge-owned workers. With the state-smoke model environment
variables set, run both threaded memory modes.

```bash
node scripts/smoke/remote_fetch_backend.mjs --dist-dir "$OUT_DIR"
node scripts/smoke/remote_fetch_backend.mjs --dist-dir "$OUT_DIR" --memory-mode wasm64
```

The shared-core staging regression uses the same pinned model and cache. Run it
against a dist with all four cores to verify that a new bridge can load after
an earlier bridge leaves a busy staging mount, when the application occupies
the next path or leaves a native registration there, and when the root
directory allows mounting but denies listing.

```bash
BRIDGE_DIST_DIR="$OUT_DIR" \
LLAMA_WEBGPU_SMOKE_MODEL_URL="$LLAMA_WEBGPU_SMOKE_MODEL_URL" \
LLAMA_WEBGPU_SMOKE_MODEL_SHA256="$LLAMA_WEBGPU_SMOKE_MODEL_SHA256" \
LLAMA_WEBGPU_SMOKE_MODEL_CACHE=~/.cache/llama-web-bridge/state-smoke-models \
LLAMADART_BROWSER_STAGING_MOUNT_SMOKE=1 \
node --test tests/bridge/js_model_staging_shared_core_browser_test.mjs
```

For local WebGPU acceptance on a non-isolated page, use the model and checksum
for each memory mode in the [single-thread measurements](docs/single_thread.md):

```bash
node scripts/smoke/single_thread.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path "$MODEL_PATH" --model-sha256 "$MODEL_SHA256" \
  --memory-mode wasm32 --runtime-mode worker --gpu-layers 999 \
  --artifacts-dir /private/tmp/llama-web-bridge-single-thread-smoke
```

Run each memory mode in worker and direct mode, with and without
`--js-backed-model-staging`. Use separate artifact directories to retain each
run's staged byte count and peak wasm heap. The worker run also calls the
native URL loader directly and checks that it returns `-7`. This is a local
acceptance check.

For the heap-staging lifecycle check, run the same command with
`--js-backed-model-staging --gpu-layers 0 --heap-staging-lifecycle`. It checks
a same-core failed load, replacement, full-offload controls and output parity
with staging off. Use `--heap-staging-reuse` instead for a tightly capped core
to check an identical replacement without growing wasm memory or staging JS
chunks. To check fallback, pass
`--expect-heap-fallback unsupported` with a fixed-memory core or
`--expect-heap-fallback allocation` with a model larger than a growable
core's maximum heap size.

For the v0.7.0 embedding scheduler repair, the build compiles a
fingerprint-checked copy of `llama-context.cpp` under `BUILD_DIR/generated`.
The repair applies only to the affected guard and leaves the checkout untouched. See
[the CMake repair](cmake/embedding_mode_reserve.cmake) and
[llamadart-native #114](https://github.com/leehack/llamadart-native/issues/114).

### npm bundler packaging

Use a dist with all four cores and the state-smoke model environment variables.
The smoke stages a local npm tarball and exercises Vite, webpack, esbuild, and
cross-origin browser ESM on isolated and non-isolated pages.

```bash
npm ci --ignore-scripts --prefix scripts/smoke/bundler_deps
node scripts/smoke/bundler_packaging.mjs --dist-dir "$OUT_DIR" \
  --model-url "$LLAMA_WEBGPU_SMOKE_MODEL_URL" \
  --model-sha256 "$LLAMA_WEBGPU_SMOKE_MODEL_SHA256"
```

### Multimodal embedding qualification

The opt-in embedding smoke uses existing local model/projector files and an
independent upstream-only CPU reference. It never downloads weights. See
[the embedding qualification record](docs/multimodal_embeddings.md) for source
identity, model revision, checksums, tolerance and results. Export the checksum
variables from that record. Keep all outputs outside the repository.

Build the reference against that exact llama.cpp checkout (`LLAMA_CPP_DIR`):

```bash
cmake -S "$LLAMA_CPP_DIR" -B "$TMPDIR/embedding-native" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_OPENSSL=OFF
CCACHE_DIR="$TMPDIR/embedding-ccache" \
  cmake --build "$TMPDIR/embedding-native" --target mtmd -j 8
c++ -std=c++17 -O2 \
  -I"$LLAMA_CPP_DIR/include" -I"$LLAMA_CPP_DIR/ggml/include" \
  -I"$LLAMA_CPP_DIR/tools/mtmd" scripts/smoke/embedding_reference.cpp \
  "$TMPDIR/embedding-native/tools/mtmd/libmtmd.a" \
  "$TMPDIR/embedding-native/src/libllama.a" \
  "$TMPDIR/embedding-native/ggml/src/libggml.a" \
  "$TMPDIR/embedding-native/ggml/src/libggml-cpu.a" \
  "$TMPDIR/embedding-native/ggml/src/libggml-base.a" \
  "$TMPDIR/embedding-native/vendor/hash/libvendor-hash.a" \
  -framework Accelerate -o "$TMPDIR/embedding-reference"
"$TMPDIR/embedding-reference" \
  ~/llamadart-qa-models/embeddinggemma-2/gguf/embeddinggemma-2-BF16.gguf \
  ~/llamadart-qa-models/embeddinggemma-2/gguf/mmproj-embeddinggemma-2-BF16.gguf \
  "$TMPDIR/embedding-reference.json"
node scripts/smoke/multimodal_embeddings.mjs \
  --dist-dir "$OUT_DIR" \
  --model-path ~/llamadart-qa-models/embeddinggemma-2/gguf/embeddinggemma-2-BF16.gguf \
  --model-sha256 "$EMBEDDING_MODEL_SHA256" \
  --mmproj-path ~/llamadart-qa-models/embeddinggemma-2/gguf/mmproj-embeddinggemma-2-BF16.gguf \
  --mmproj-sha256 "$EMBEDDING_PROJECTOR_SHA256" \
  --reference-path "$TMPDIR/embedding-reference.json" \
  --baseline-path "$TMPDIR/baseline-text/baseline-text-result.json" \
  --memory-mode wasm64 --artifacts-dir "$TMPDIR/embedding-browser"
```

The native commands above target macOS CPU. The qualification record includes
CPU precision controls and numerical comparisons. Pass criteria are cosine >=
0.9998 and relative L2 < 2% per input against native CPU, exact direct/worker
parity, and exact legacy string parity with pre-task baseline vectors. Set
`--baseline-path` to the saved pre-task direct/worker JSON. Max-absolute
differences are informational. WebGPU GPU-kernel qualification was not run.
The browser smoke checks both
direct and worker paths, exact runtime selection, 768-wide raw/normalized
outputs, synthetic media sensitivity, interleaving, batch order, adjacent text,
overflow, projector unload/reload, cancellation and text reuse. It does not
qualify retrieval quality or publish assets. If the supplied BF16 pair exceeds
browser memory, record the runtime's exact limit and stop; do not substitute or
download other weights.

### Multimodal completion, samplers, and speculative decoding

For llama.cpp pin or multimodal changes, run checksum-pinned real image
inference through both direct and worker runtimes:

```bash
node scripts/smoke/multimodal.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/Qwen3.5-0.8B-Q4_K_M.gguf \
  --model-sha256 bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517 \
  --mmproj-path /path/to/mmproj-F16.gguf \
  --mmproj-sha256 56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453 \
  --artifacts-dir /tmp/llama-web-bridge-multimodal-smoke
```

For sampler or grammar changes, run grammar-constrained completion, the
seeded `minP`/`presencePenalty` checks, the prompt-penalty check and the greedy
`thinkingBudget` checks through direct and worker runtimes on both memory modes.
The same smoke checks the grammar prompt-penalty opt-in against the no-grammar
baseline, including a trigger present in the prompt that must remain inactive,
and confirms a generated trigger still constrains the suffix. Default grammar
and empty-grammar runs remain separate controls.
CI runs it twice: with the state-persistence model, then with the multimodal
model. Pass each model's `--model-url` (or `--model-path`) and `--model-sha256`
pin from the commands above; the smoke defaults to the
`LLAMA_WEBGPU_SMOKE_MODEL_URL` and `LLAMA_WEBGPU_SMOKE_MODEL_SHA256` environment
variables. Its end-tag-rejecting grammar check reproduces the original bug
only with the multimodal (Qwen3.5) model; the tiny model passes it either way:

```bash
node scripts/smoke/grammar.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-url "$LLAMA_WEBGPU_SMOKE_MODEL_URL" \
  --model-sha256 "$LLAMA_WEBGPU_SMOKE_MODEL_SHA256" \
  --artifacts-dir /tmp/llama-web-bridge-grammar-smoke
```

The smoke reports `GRAMMAR_STAGE` markers and uses available browser threads
up to its configured ceiling. Supported models reuse saved prompt prefixes in
repeated cases. Each mode still exercises prompt evaluation from scratch.

For next-token scoring changes, run `scoreNextToken` through direct and worker
runtimes on both memory modes with the state-persistence model:

```bash
node scripts/smoke/next_token_scores.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-url "$LLAMA_WEBGPU_SMOKE_MODEL_URL" \
  --model-sha256 "$LLAMA_WEBGPU_SMOKE_MODEL_SHA256" \
  --artifacts-dir /tmp/llama-web-bridge-next-token-scores-smoke
```

For speculative decoding changes, run greedy completions with and without each
strategy through direct and worker runtimes on both memory modes. The `tiny`
group runs the n-gram strategies and `draft-simple` on the state-persistence
model; `--group all` adds real-weight groups for every strategy, read from
`--models-dir` at the paths in the script's `FILES` table. The EAGLE3 and DFlash
drafts and the n-gram cache have no download URL, so place them there first:

```bash
node scripts/smoke/speculative.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --group all \
  --models-dir ~/.cache/llama-web-bridge/speculative-smoke-models \
  --artifacts-dir /tmp/llama-web-bridge-speculative-smoke
```

### Hosted speech qualification and opt-in smokes

Heavy Qwen3-ASR and Qwen3-TTS gates run in the hosted automated
qualification workflow, not in ordinary CI or the candidate build. Before
publishing, `.github/workflows/bridge_qualification.yml` runs them against the
exact candidate artifact built by `.github/workflows/bridge_candidate.yml`.
The workflow proves the candidate's workflow/run/source identity and unique
artifact ID, verifies every downloaded input checksum, installs Node.js 24 with
the candidate source's locked npm dependencies and Playwright Chromium, requires
GitHub Actions `github-hosted` runner identity, and emits one canonical
attestation. That attestation binds the candidate artifact ID/run/attempt/workflow/digest and the
producing qualification run ID/attempt/workflow/source SHA. The combined
`scripts/release/qualification.mjs qualify` command is therefore workflow-only.

For local reproduction, run the individual smokes directly:

```bash
node scripts/smoke/speech_to_text.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/Qwen3-ASR-0.6B-Q8_0.gguf \
  --model-sha256 bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971 \
  --mmproj-path /path/to/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --mmproj-sha256 41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d \
  --artifacts-dir /tmp/llama-web-bridge-speech-smoke
```

```bash
node scripts/smoke/text_to_speech.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf \
  --model-sha256 8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129 \
  --mmproj-path /path/to/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --mmproj-sha256 6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2 \
  --memory-mode wasm64 \
  --runtime-mode all \
  --gpu-layers 99 \
  --artifacts-dir /tmp/llama-web-bridge-text-to-speech-smoke
```

For decision-head changes, compare a Laya encoder and head with the Laya
reference fixture (llamadart's
`test/fixtures/decision/laya_0_3_5_reference.json`) through both runtimes and
memory modes. Pass `--config-path` with `rl_agent_config.json` for a head
without `laya.config` metadata, such as the official `model.safetensors`, and
`--gpu-layers 0` to check the CPU path:

```bash
node scripts/smoke/decision.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/laya-Q8_0.gguf \
  --head-path /path/to/laya-head.safetensors \
  --fixture-path /path/to/laya_0_3_5_reference.json \
  --gpu-layers 99 \
  --artifacts-dir /tmp/llama-web-bridge-decision-smoke
```

For LoRA adapter changes, run a base model and a LoRA adapter trained on it
through both runtimes and memory modes. The smoke defaults to a checksum-pinned
stories15M base and Shakespeare adapter and loads the adapter onto the
state-persistence model to check the mismatch error. The default
`--gpu-layers 0` runs on the CPU. A nonzero value enables WebGPU and skips the
model reload check, because reloading a model on one bridge aborts in headless
Chromium with WebGPU enabled, with or without adapters:

```bash
node scripts/smoke/lora_adapter.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --mismatch-model-url "$LLAMA_WEBGPU_SMOKE_MODEL_URL" \
  --mismatch-model-sha256 "$LLAMA_WEBGPU_SMOKE_MODEL_SHA256" \
  --artifacts-dir /tmp/llama-web-bridge-lora-smoke
```

If the smoke downloads from a URL, errors and diagnostics must redact userinfo,
query strings, and fragments before printing the location.

## Agent Workflow Guardrails

- Update `scripts/ci/verify_ci_reliability.mjs` when changing CI, candidate,
  qualification, asset publication, or npm publication workflows or their
  model pins. It checks permissions, environment gates, credential handling,
  pins, failure guards, contract execution, and the ban on Python in workflows.
  Name runnable tests under `tests/` with the `*_test.mjs` suffix. Other files
  there must be helpers imported by a test.
- Rotate the seven model/projector pins together in this guide, `ci.yml`, and
  `bridge_candidate.yml`. `scripts/release/qualification.mjs` carries those
  seven plus the ASR audio pin. Rotate the five speech, TTS, and ASR audio
  pins in `bridge_qualification.yml` too. Update the audio URL and SHA-256 in
  `scripts/smoke/speech_to_text_fixture.json`. The qualification pin test
  compares the audio digest, so check its URL by hand.
- The verifier matches each workflow URL and digest by role against
  `EXPECTED_MODEL_PINS` and matches each literal smoke digest here to its
  preceding model or projector filename. Workflow URLs for a role must match.
  It checks the inventory of mutable Hugging Face refs (the state and
  multimodal models) and the unversioned ASR audio URL. Keep model pins out of
  `README.md` and `AGENTS.md`.
- Run checksum-pinned multimodal inference through direct and worker runtimes
  in normal CI for every llama.cpp pin update.
- Hosted speech qualification rejects memory-mode and worker fallback. The
  candidate manifest must match the compiled core memories and mark speech
  and TTS `required-automated-qualification`. Keep real-device playback,
  intelligibility, and speaker-reference fidelity in `unproven_capabilities`.
- Keep the ordinary development and CI pin in `llama_cpp.version`. It holds
  one upstream tag, either stable `vMAJOR.MINOR.PATCH` or development `bNNNN`.
  Publication leaves that pin alone. CI and candidate builds verify the active
  `emcc` against `emsdk.version`. Publication verifies the manifest compiler
  against the exact bridge source's pin.
- Preserve `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` in CI and publication workflows.
  Upload state-persistence diagnostics only on failure.
- Pass workflow dispatch inputs through `env` and use quoted shell expansions.
  Keep `${{ inputs.* }}` out of workflow `run` scripts.
- Only push branches or tags or publish assets from local agent work when the
  maintainer explicitly requests it.

## Publish Process

Releases are manual: the repository owner dispatches four workflows from
`main`, in order, each after the previous run succeeds. Every stage is
first-attempt-only. After a failed build or qualification, diagnose it and
dispatch a new candidate with the same inputs.
Use one correlation ID for the first three stages, 1–128 characters of `A-Za-z0-9._:-`
starting alphanumeric, such as `manual-v0.1.83`. The input keeps its historical
name `orchestrator_correlation_id` because published manifests and release
notes record it.

The candidate workflow builds all four WebGPU cores in parallel from clean
source without CI artifacts or compiler caches. It runs the JS contract
checks once, then packages the verified wrappers and compiled cores after
running the hosted smoke gates. The qualification workflow runs the ASR and
TTS gates in parallel against that candidate, then assembles and re-verifies
the canonical attestation. Every job in both workflows checks that it is
running on the first attempt.

Before dispatching `bridge_candidate.yml` for `vX.Y.Z`, merge a release-prep
PR that moves the changelog entries under `## Unreleased` to
`## vX.Y.Z - YYYY-MM-DD` using the planned release date. Keep an empty
`## Unreleased` at the top. Set the candidate's `bridge_source_sha` to a
commit on `main` that includes the changelog update.

Gather the inputs. Use the newest stable `llamadart-native` release: the
candidate and publish workflows reject a development (`bNNNN`) native tag, but
nothing checks that native releases are published in order. Substitute the
actual tags and run IDs for the examples below.

```bash
NATIVE_TAG=v0.8.0-1
gh release download "$NATIVE_TAG" -R leehack/llamadart-native -p assets.json -O assets.json
jq -r '.llama_cpp_tag, .llama_cpp_commit' assets.json  # upstream_tag, upstream_commit
shasum -a 256 assets.json                              # native_manifest_sha256
git ls-remote https://github.com/leehack/llama-web-bridge refs/heads/main  # bridge_source_sha
gh api repos/leehack/llama-web-bridge-assets/releases --jq '.[].tag_name' \
  | sort -V | tail -1  # release_tag: the next patch above this
gh api repos/leehack/llama-web-bridge-assets/immutable-releases  # must be enabled
```

1. Build Exact Bridge Candidate:

   ```bash
   gh workflow run bridge_candidate.yml -R leehack/llama-web-bridge --ref main \
     -f orchestrator_correlation_id=manual-v0.1.83 \
     -f bridge_source_sha=<main SHA> \
     -f upstream_tag=<llama_cpp_tag> -f upstream_commit=<llama_cpp_commit> \
     -f native_release_tag="$NATIVE_TAG" -f native_manifest_sha256=<assets.json SHA-256> \
     -f release_tag=v0.1.83 -f release_rebuild=0 \
     -f assets_immutable_releases_enabled=true
   ```

2. Qualify Exact Bridge Candidate, with the candidate's run ID:

   ```bash
   gh workflow run bridge_qualification.yml -R leehack/llama-web-bridge --ref main \
     -f orchestrator_correlation_id=manual-v0.1.83 -f candidate_run_id=<candidate run ID>
   ```

3. Publish Exact Qualified Bridge Assets, with the step 1 inputs except
   `assets_immutable_releases_enabled`, plus both run IDs (add
   `-f allow_upstream_tag_override=true` only when `upstream_tag` intentionally
   differs from `bridge_source_sha`'s `llama_cpp.version`):

   ```bash
   gh workflow run publish_assets.yml -R leehack/llama-web-bridge --ref main \
     -f orchestrator_correlation_id=manual-v0.1.83 \
     -f bridge_source_sha=<main SHA> \
     -f upstream_tag=<llama_cpp_tag> -f upstream_commit=<llama_cpp_commit> \
     -f native_release_tag="$NATIVE_TAG" -f native_manifest_sha256=<assets.json SHA-256> \
     -f release_tag=v0.1.83 -f release_rebuild=0 \
     -f candidate_run_id=<candidate run ID> -f qualification_run_id=<qualification run ID> \
     -f publish_approved=true
   ```

4. Publish Exact Qualified Bridge to npm, using the immutable assets release
   from step 3. Complete the one-time setup below before the first dispatch.

   ```bash
   gh workflow run publish_npm.yml -R leehack/llama-web-bridge --ref main \
     -f release_tag=v0.1.83 -f publish_approved=true
   ```

`gh run list -R leehack/llama-web-bridge -w <workflow file> -L 1` shows a run's
ID. Each stage re-verifies everything it consumes: the candidate resolves the
upstream and native tags and checks the native `assets.json` digest and
inventory; qualification proves the candidate run and that its manifest carries
the same correlation ID; asset publication re-proves both runs and binds the
attestation to the release inputs. npm publication authenticates the immutable
release, verifies the candidate and publication runs, and downloads the
unexpired qualification attestation by artifact ID to confirm it covers the
exact release files. Dispatch step 4 while the qualification artifacts are
still available.

### Asset publication invariants

- **Immutable-release governance.** The candidate lacks permission to query
  assets repository governance. It requires and records the dispatch assertion
  `assets_immutable_releases_enabled=true`. Publication binds that assertion
  to the exact run-owned prequalification record and checks the
  live `immutable-releases` endpoint before ref or release mutation. The
  response must be HTTP 200 with the exact `enabled` and `enforced_by_owner`
  fields and boolean `enabled: true`.
- **Environment and credentials.** Set up `bridge-assets-publication` with
  administrator bypass disabled, custom deployment branches limited to
  `main`, and the environment-scoped `WEBGPU_BRIDGE_ASSETS_PAT`. The
  solo-maintainer contract requires no reviewers or self-review rule.
  `publish_approved=true` is the owner’s explicit approval. The default job
  token verifies the environment before the privileged job and again before
  its first PAT-bearing step, using the trusted workflow commit’s validator.
  Each PAT-bearing step
  requires a nonempty value without printing it. The PAT needs read access to
  provenance repositories plus write access and Administration read on
  `leehack/llama-web-bridge-assets`. It needs no permission to dispatch
  workflows in the bridge repository.
- **Exact inputs.** The candidate is the only build of publishable bytes.
  Its manifest embeds its run ID and URL, so a rebuild cannot reuse an
  attestation. Asset publication downloads candidate and qualification
  artifacts by immutable ID and verifies both runs and the attestation before
  and inside the privileged job. The trusted workflow commit supplies the
  publication policy. The selected bridge source supplies the exact
  harness, toolchain pin, and build identity.
- **Readback.** Every complete release, including a published noop, must
  resolve the exact tag commit. Independent reads by tag and release ID must
  agree on the published release identity and report boolean `immutable: true`.
  Verify the published digests with
  `gh release verify <tag> --repo leehack/llama-web-bridge-assets --format json`.
  The attestation must use `https://in-toto.io/attestation/release/v0.2`, the
  signer `https://dotcom.releases.github.com`, and the live release database
  ID. A mismatch is non-retryable `immutable-publication-unverified`.
- **Recovery.** Publication evaluates release histories by channel. It can
  recover exact ref-only partial states. Incomplete or mismatched published
  releases are never filled in, deleted, retagged, or overwritten. A failed
  post-mutation query records retryable `mutation-unknown` instead of guessing.
  Retry partial publication with a fresh first-attempt dispatch and the same
  candidate and qualification IDs. Each output tag belongs to one candidate.

New bridge asset tags are npm-shaped `vMAJOR.MINOR.PATCH` with rebuild `0`:
take the next free patch version above the newest published tag.
`scripts/release/contract.mjs validate-release` rejects
a new `-N` tag in the candidate and publish workflows, because npm orders
`vMAJOR.MINOR.PATCH-N` as a prerelease. Earlier `-N` bridge tags, the native
forms `vMAJOR.MINOR.PATCH-N`, `bNNNN` and `bNNNN-N`, historical
`bNNNN-llamadart.N` and prior wrapper forms are read-only compatibility inputs.

### One-time npm setup

1. In the bridge repository's GitHub settings, create the
   `bridge-npm-publication` environment. Disable administrator bypass, choose
   **Selected branches and tags**, and add only the branch `main`. Leave the
   environment without required reviewers or secrets.
2. Claim `llama-web-bridge` on npm from the maintainer's account. npm requires an
   existing package before it can bind a Trusted Publisher. For the initial
   claim, use a minimal `0.0.0-claim.0` package in a scratch directory under the
   `bootstrap` dist-tag, then deprecate that version. The first qualified
   release will set `latest`.
3. In the package's npm access settings, add a GitHub Actions Trusted Publisher
   with organization or user `leehack`, repository `llama-web-bridge`, workflow
   filename `publish_npm.yml`, and environment `bridge-npm-publication`. Allow
   publishing. After the first successful OIDC publish, set the package's
   publishing access to require two-factor authentication and disallow tokens.

The workflow accepts stable unsuffixed tags with rebuild `0` and rejects
`v0.0.0` and fixture manifests. It requires an HTTP 404 for the version before
running `npm publish` once with npm's fetch retries disabled. After publishing,
it polls version metadata, dist-tags, and attestations, then polls the tarball
under the same 30-minute deadline. It backs off from 15 to 60 seconds for 404s,
transient server failures, and transient network errors. Metadata or digest
mismatches fail immediately. The workflow verifies the tarball's digests, the
`latest` tag, and signed provenance before uploading a
`bridge-npm-publication-outcome` receipt. If publication succeeds and readback
fails, check the existing version and its provenance on npm.

## CI change selection and compiler cache

CI always runs the shared JS and workflow contracts. An explicit allowlist in
`scripts/ci/ci_scope.mjs` lets known documentation and tooling-only changes avoid the
WASM builds. Runtime JS, C++, browser harnesses, build inputs, workflows, pins,
and unknown paths retain the pinned build/smoke lane. Rename and
deletion comparisons include both paths. The `CI validation` result always
reports and rejects failed, cancelled, missing, or unexpectedly skipped work.
Only superseded PR runs are cancelled; main/manual runs remain independent.

CI builds the four core variants in parallel. Each variant can restore its
generated core JS and WASM using an exact key for the runner OS and architecture,
Emscripten version, verified llama.cpp commit, and a Git tree fingerprint. The
fingerprint covers tracked paths, modes, and blob IDs for the bridge native
sources, CMake inputs, the build scripts and their local imports, and the CI
workflow. Every selected run combines the cores with the checked-in JS wrappers
and runs the browser smokes. The separate JS checks verify the wrappers against
their sources.

On a core cache miss, CI restores a per-variant Emscripten sysroot cache
(`EM_CACHE`) keyed by runner OS, architecture, and exact SDK version. The
per-variant ccache key also includes the llama.cpp commit and native build
inputs. It can reuse objects from the same toolchain and upstream commit
because ccache verifies compiler contents, source inputs, and compile flags.
Only pushes to `main` save missing core, Emscripten, and ccache entries.
Candidate and publication workflows do not consume these CI caches or use this
change selector.
