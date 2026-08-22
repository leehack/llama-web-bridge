# AGENTS.md

Guidance for coding agents working in `llama-web-bridge`.

## Scope and Ownership

- This repository owns WebGPU bridge source/runtime build for llama.cpp web.
- It publishes versioned assets to `llama-web-bridge-assets` via workflow.
- `llamadart` consumes those published assets.

## Related Repositories

Common maintainer sibling layout:

```text
../llamadart
../llamadart-native
../llama-web-bridge
../llama-web-bridge-assets
```

## Build Commands

```bash
./scripts/build_bridge.sh
```

Useful environment overrides:

- `LLAMA_CPP_DIR` (defaults to `third_party/llama_cpp`; CI clones the tag from
  `llama_cpp.version`)
- The local Emscripten SDK must match the exact release in `emsdk.version`.
- `BUILD_DIR`
- `OUT_DIR`
- `CMAKE_BUILD_TYPE`

## Agent PR Workflow

For non-trivial runtime, workflow, or API changes, keep the PR path explicit:

1. Start from a clean topic branch and inspect `git status` before editing.
2. Add or update a regression/contract check before changing behavior when
   practical. Static contract scripts are acceptable for workflow invariants.
3. Keep Emscripten build directories, ccache, model caches, and Playwright
   artifacts outside the repository unless they are intentionally versioned.
4. Run the targeted checks in this file and the full browser smoke when the
   change touches `js/`, `src/`, `scripts/`, or GitHub workflows.
5. Use an independent review before committing PR-bound changes. Fix blocking
   findings, rerun the targeted checks, then commit locally; do not push or open
   a PR unless the maintainer asks.

### Local Verification Notes

When validating bridge runtime changes locally, keep build/cache output outside
the repo so generated wasm artifacts and toolchain caches do not dirty the
checkout or hit sandboxed Homebrew/cache paths:

```bash
export CCACHE_DIR=/private/tmp/llama_web_bridge_ccache
export EM_CACHE=/private/tmp/llama_web_bridge_emcache
BUILD_DIR=/private/tmp/llama_web_bridge_build MEM64_BUILD_DIR=/private/tmp/llama_web_bridge_build_mem64 OUT_DIR=/private/tmp/llama_web_bridge_dist WEBGPU_BRIDGE_BUILD_MEM64=1 ./scripts/build_bridge.sh
```

Minimum local checks before handing off a PR-ready branch:

```bash
npm run check:js
python3 -m py_compile scripts/verify_state_persistence_api.py scripts/verify_text_to_speech_api.py scripts/verify_ci_reliability.py scripts/state_persistence_browser_smoke.py scripts/multimodal_browser_smoke.py scripts/speech_to_text_browser_smoke.py scripts/text_to_speech_browser_smoke.py
python3 scripts/verify_state_persistence_api.py
python3 scripts/verify_text_to_speech_api.py
python3 scripts/verify_ci_reliability.py
```

For state-persistence or workflow changes, also run the browser smoke against a
built `OUT_DIR`. Keep the tiny model in a user cache or `/private/tmp`; do not
commit downloaded GGUFs or smoke artifacts:

```bash
python3 -m pip install --user playwright
python3 -m playwright install chromium
python3 scripts/state_persistence_browser_smoke.py \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-url https://huggingface.co/aladar/llama-2-tiny-random-GGUF/resolve/main/llama-2-tiny-random.gguf \
  --model-sha256 81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb \
  --model-cache-dir ~/.cache/llama-web-bridge/state-smoke-models \
  --artifacts-dir /private/tmp/llama_web_bridge_state_smoke_artifacts
```

For llama.cpp pin or multimodal changes, also run real image inference through
both direct and worker runtimes:

```bash
python3 scripts/multimodal_browser_smoke.py \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/Qwen3.5-0.8B-Q4_K_M.gguf \
  --model-sha256 bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517 \
  --mmproj-path /path/to/mmproj-F16.gguf \
  --mmproj-sha256 56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453 \
  --artifacts-dir /private/tmp/llama_web_bridge_multimodal_smoke_artifacts
```

Before publishing assets intended for typed Qwen3-ASR, also run the opt-in
speech smoke in wasm32 and memory64, through direct and worker runtimes:

```bash
python3 scripts/speech_to_text_browser_smoke.py \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/Qwen3-ASR-0.6B-Q8_0.gguf \
  --model-sha256 bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971 \
  --mmproj-path /path/to/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --mmproj-sha256 41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d
```

Before publishing assets intended for Qwen3-TTS, run the checksum-pinned
memory64 smoke through direct and worker runtimes. The validated 1.48 GB pair is
not a practical wasm32 product path:

```bash
python3 scripts/text_to_speech_browser_smoke.py \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf \
  --model-sha256 8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129 \
  --mmproj-path /path/to/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --mmproj-sha256 6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2 \
  --memory-mode wasm64 \
  --runtime-mode all \
  --gpu-layers 99
```

Keep this gate opt-in in ordinary CI because the pair is large and
memory64-only in practice. The asset release workflow must run it for every
publication because the bridge exposes Qwen3-TTS.

## CI / Release

- CI build gate: `.github/workflows/ci.yml`
  - Resolves the default llama.cpp checkout from `llama_cpp.version`.
  - Resolves `emsdk.version`, installs that exact compiler, verifies the active
    `emcc` identity, and contract-tests all five required wasm64 WASMFS patches.
  - Never dispatches asset publication. Bridge source changes, including changes
    to the default development pin, use ordinary PRs and ordinary CI.
- Native-aligned candidate scan: `.github/workflows/auto_llama_cpp_update.yml`
  - Scheduled/manual runs download the selected native `assets.json`, validate
    its exact upstream/native identities, and upload `release-candidate.json`.
  - The scan only prepares and reports candidate inputs. It never changes
    `llama_cpp.version`, opens a PR, dispatches publication, tags, or pushes.
- CI reliability contract: `scripts/verify_ci_reliability.py`
  - Keep this script updated when changing browser smoke behavior, action
    versions, JS build/type-checking, or workflow diagnostics.
  - CI and publish must run `npm run check:js`, which regenerates the checked-in
    generated bridge wrapper outputs and declarations, and then fail with
    `git diff --exit-code` if those generated outputs are stale.
  - The CI smoke must use a pinned tiny GGUF URL plus SHA-256, cache the model in
    the same expanded `~/.cache/llama-web-bridge/state-smoke-models` directory
    used by `actions/cache`, and upload `state-persistence-smoke-artifacts` on
    failure.
  - Both CI and publish workflows intentionally set
    `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` so action-runtime regressions are caught
    before Node 20 deprecation becomes a hard failure.
- Publish workflow: `.github/workflows/publish_assets.yml`
  - Is callable by the central orchestrator and manually dispatchable. It
    requires exact bridge source SHA, upstream tag/commit, native tag plus
    manifest SHA-256, output tag/rebuild, and assets repository inputs. It does
    not read `llama_cpp.version`.
  - Uses the same `emsdk.version` compiler identity as CI and records the
    runtime-verified version in `manifest.json`.
  - Requires `publish_approved=true`. Publication remains blocked until an
    administrator externally creates `bridge-assets-publication`, configures
    required reviewers, and stores `WEBGPU_BRIDGE_ASSETS_PAT` as an
    environment-scoped secret. Do not describe that environment as protected
    without current live evidence.
  - Emits only stable `vMAJOR.MINOR.PATCH[-N]` or development `bNNNN[-N]` tags.
    Historical `*-llamadart.N` forms are accepted only when reading old
    manifests and are never emitted.
  - Rejects identity/checksum mismatches, rollback, channel reversal, diverged
    bridge source, output collisions, and unmerged bridge source commits.
  - Builds exact wasm32/memory64 assets and always runs state, multimodal, ASR,
    and TTS durable release smokes; callers cannot downgrade exposed
    capabilities.
  - Publishes schema-v2 provenance with release tag, capabilities, bridge,
    upstream, and native commits, and per-artifact SHA-256 values.
  - Any future npm package version has an independent monotonic sequence and
    uses stable/nightly dist-tags; GitHub `vM.m.p-N` tags must not be reused as
    npm versions because npm orders them as prereleases.

## Change Boundaries

- Keep runtime bridge source code in `js/src/` and `src/`; generated browser
  wrapper/declaration outputs live in `js/`.
- Keep publishing logic in workflow only.
- Do not edit assets repository files from here outside publish flow.

## Cross-Repo Handoff to `llamadart`

After publishing assets tag:

1. Update/fetch pinned bridge assets in `llamadart`:
   `WEBGPU_BRIDGE_ASSETS_TAG=<tag> ./scripts/fetch_webgpu_bridge_assets.sh`
2. Update docs/changelog in `llamadart` if behavior changed.

## Regression Smoke Guidance

- For pthread/runtime changes, test a BERT-class embedding model in Chromium
  with cross-origin isolation enabled. The regression shape is:
  `loadModelFromUrl`, `tokenize`, `embed`, and `embedBatch` on a host where
  `navigator.hardwareConcurrency` is greater than the bridge pthread pool size.
- Run the smoke through both direct runtime (`disableWorker: true`) and the
  bridge worker path; both should report `n_threads` capped to the pool size.
- For state persistence, exercise both direct and worker runtimes with a real
  tiny model. The smoke should evaluate a prompt, save bytes, mutate state,
  reload bytes, and verify generation still works after restore.
- Worker and direct runtime filesystems are separate. Do not silently fall back
  from worker-owned state APIs to direct runtime state; byte APIs are the durable
  app-storage path for IndexedDB/OPFS/Cache API integrations.
- If the smoke downloads a model, never expose raw signed/authenticated locations in
  thrown errors or artifacts. Redact userinfo, query, and fragment values.
- Every llama.cpp pin update must pass checksum-pinned real multimodal inference
  in both direct and worker runtimes; a successful WASM build alone is not
  sufficient.
- Speech-capable asset releases must also pass the opt-in Qwen3-ASR smoke in
  wasm32 and memory64; keep the large model pair out of default CI.
