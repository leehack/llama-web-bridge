# Contributing to llama-web-bridge

Thanks for contributing.

## Purpose

This repo builds the reusable JS/WASM bridge runtime for llama.cpp web usage.
Published artifacts are consumed from `llama-web-bridge-assets`.

## Prerequisites

- Emscripten SDK (`emcmake`, `emcc`)
- CMake toolchain
- Access to a llama.cpp checkout matching `llama_cpp.version`

## Setup

```bash
git clone https://github.com/leehack/llama-web-bridge.git
cd llama-web-bridge
./scripts/build_bridge.sh --help
```

## Local Build

```bash
./scripts/build_bridge.sh
# or
LLAMA_CPP_DIR=../llama.cpp OUT_DIR=dist ./scripts/build_bridge.sh
```

For local agent/maintainer validation, prefer external build and cache paths so
generated files do not dirty the checkout:

```bash
export CCACHE_DIR=/private/tmp/llama_web_bridge_ccache
export EM_CACHE=/private/tmp/llama_web_bridge_emcache
BUILD_DIR=/private/tmp/llama_web_bridge_build \
MEM64_BUILD_DIR=/private/tmp/llama_web_bridge_build_mem64 \
OUT_DIR=/private/tmp/llama_web_bridge_dist \
WEBGPU_BRIDGE_BUILD_MEM64=1 \
./scripts/build_bridge.sh
```

## Validate Outputs

Expected files:

- `dist/llama_webgpu_bridge.js`
- `dist/llama_webgpu_bridge_worker.js`
- `dist/llama_webgpu_core.js`
- `dist/llama_webgpu_core.wasm`

Before opening or updating a PR, run the lightweight contracts:

```bash
node --check js/llama_webgpu_bridge.js
node --check js/llama_webgpu_bridge_worker.js
python3 -m py_compile scripts/verify_state_persistence_api.py scripts/verify_ci_reliability.py scripts/state_persistence_browser_smoke.py
python3 scripts/verify_state_persistence_api.py
python3 scripts/verify_ci_reliability.py
```

For state-persistence, worker, or workflow changes, also run the browser smoke
against a built dist directory. Use a checksum-pinned tiny model and keep caches
and artifacts outside the repository:

```bash
python3 scripts/state_persistence_browser_smoke.py \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-url https://huggingface.co/aladar/llama-2-tiny-random-GGUF/resolve/main/llama-2-tiny-random.gguf \
  --model-sha256 81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb \
  --model-cache-dir ~/.cache/llama-web-bridge/state-smoke-models \
  --artifacts-dir /tmp/llama-web-bridge-state-smoke
```

If the smoke downloads from a URL, errors and diagnostics must redact userinfo,
query strings, and fragments before printing the location.

## Agent Workflow Guardrails

- Keep workflow reliability rules in `scripts/verify_ci_reliability.py` when
  changing `.github/workflows/ci.yml`, `.github/workflows/publish_assets.yml`,
  `.github/workflows/auto_llama_cpp_update.yml`, or
  `scripts/state_persistence_browser_smoke.py`.
- Preserve `llama_cpp.version` as the single source of truth for default CI and
  publish builds. Manual publish overrides are allowed for temporary validation,
  but tag-triggered publishes should use the pinned file.
- The auto-update workflow manages the stable `automation/bump-llama-cpp` branch
  and updates an existing PR instead of opening duplicates. It should include the
  upstream release notes, compare link, and commit range in the PR body, then
  dispatch CI on the automation branch so bot-token updates are validated.
- Preserve `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` in CI and publish workflows so
  GitHub Action runtime changes are detected before they become mandatory.
- Upload state-persistence smoke diagnostics only on failure; successful CI runs
  should stay quiet beyond the normal build artifacts.
- Do not push branches, tags, or publish assets from local agent work unless the
  maintainer explicitly requests that side effect.

## Publish Process

Use workflow `.github/workflows/publish_assets.yml`:

1. Set input `assets_tag` (new tag).
2. Optionally set `assets_repo`; leave `llama_cpp_tag` empty to use
   `llama_cpp.version`, or set it only for an explicit temporary override.
3. Ensure `WEBGPU_BRIDGE_ASSETS_PAT` secret is configured.
4. Workflow builds, generates `manifest.json`/`sha256sums.txt`, pushes to
   assets repo, creates matching tag there, and uses the build job's resolved
   `llama.cpp` tag output in release notes.

## Repository Boundaries

- Bridge runtime source/build belongs here.
- Versioned static artifacts belong in `llama-web-bridge-assets`.
- Consumer integration (loading/fallback behavior) belongs in `llamadart`.
