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
node --check js/llama_webgpu_bridge.js
node --check js/llama_webgpu_bridge_worker.js
python3 -m py_compile scripts/verify_state_persistence_api.py scripts/verify_ci_reliability.py scripts/state_persistence_browser_smoke.py
python3 scripts/verify_state_persistence_api.py
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

## CI / Release

- CI build gate: `.github/workflows/ci.yml`
  - Resolves the default llama.cpp checkout from `llama_cpp.version`.
- Automated llama.cpp bump PR: `.github/workflows/auto_llama_cpp_update.yml`
  - Runs on a schedule/manual dispatch, compares `llama_cpp.version` against the
    latest `ggml-org/llama.cpp` release, and manages one stable
    `automation/bump-llama-cpp` PR.
  - The PR body must include the upstream release notes, compare URL, commit
    range, and WebGPU/WASM review focus. If a newer upstream release appears
    while the PR is still open, update the same PR instead of opening a duplicate.
  - Skip instead of racing when a non-automation PR already changes
    `llama_cpp.version`.
- CI reliability contract: `scripts/verify_ci_reliability.py`
  - Keep this script updated when changing browser smoke behavior, action
    versions, or workflow diagnostics.
  - The CI smoke must use a pinned tiny GGUF URL plus SHA-256, cache the model in
    the same expanded `~/.cache/llama-web-bridge/state-smoke-models` directory
    used by `actions/cache`, and upload `state-persistence-smoke-artifacts` on
    failure.
  - Both CI and publish workflows intentionally set
    `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` so action-runtime regressions are caught
    before Node 20 deprecation becomes a hard failure.
- Publish workflow: `.github/workflows/publish_assets.yml`
  - Defaults to `llama_cpp.version`; workflow-dispatch `llama_cpp_tag` is only a
    temporary explicit override.
  - Requires `WEBGPU_BRIDGE_ASSETS_PAT`
  - Pushes assets + tag to `llama-web-bridge-assets`

## Change Boundaries

- Keep runtime bridge code in `js/` and `src/`.
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
