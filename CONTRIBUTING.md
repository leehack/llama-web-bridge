# Contributing to llama-web-bridge

Thanks for contributing.

## Purpose

This repo builds the reusable JS/WASM bridge runtime for llama.cpp web usage.
Published artifacts are consumed from `llama-web-bridge-assets`.

## Prerequisites

- Emscripten SDK (`emcmake`, `emcc`)
- Node.js/npm for JS bridge bundling and TypeScript `checkJs`
- CMake toolchain
- Access to a llama.cpp checkout matching `llama_cpp.version`

## Setup

```bash
git clone https://github.com/leehack/llama-web-bridge.git
cd llama-web-bridge
npm ci
./scripts/build_bridge.sh --help
```

## Local Build

```bash
npm run check:js
./scripts/build_bridge.sh
# or
LLAMA_CPP_DIR=../llama.cpp OUT_DIR=dist ./scripts/build_bridge.sh
```

Bridge wrapper source lives under `js/src/`; `npm run build:js` regenerates the
checked-in browser ESM outputs and declarations under `js/`. `npm run check:js`
runs the same generator plus TypeScript and syntax checks, so commit any updated
`js/` outputs after source changes.

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
- `dist/llama_webgpu_bridge.d.ts`
- `dist/llama_webgpu_core.js`
- `dist/llama_webgpu_core.wasm`

Before opening or updating a PR, run the lightweight contracts:

```bash
npm run check:js
python3 -m py_compile scripts/verify_state_persistence_api.py scripts/verify_ci_reliability.py scripts/state_persistence_browser_smoke.py scripts/multimodal_browser_smoke.py
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

For llama.cpp pin or multimodal changes, run checksum-pinned real image
inference through both direct and worker runtimes:

```bash
python3 scripts/multimodal_browser_smoke.py \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/Qwen3.5-0.8B-Q4_K_M.gguf \
  --model-sha256 bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517 \
  --mmproj-path /path/to/mmproj-F16.gguf \
  --mmproj-sha256 56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453 \
  --artifacts-dir /tmp/llama-web-bridge-multimodal-smoke
```

If the smoke downloads from a URL, errors and diagnostics must redact userinfo,
query strings, and fragments before printing the location.

## Agent Workflow Guardrails

- Keep workflow reliability rules in `scripts/verify_ci_reliability.py` when
  changing `.github/workflows/ci.yml`, `.github/workflows/publish_assets.yml`,
  `.github/workflows/auto_llama_cpp_update.yml`, JS build pipeline files, or
  `scripts/state_persistence_browser_smoke.py`.
- Keep `scripts/multimodal_browser_smoke.py` in normal CI for every llama.cpp
  pin update; build-only validation does not cover mtmd prompt ingestion.
- Preserve `llama_cpp.version` as the single source of truth for default CI and
  publish builds. Manual publish overrides are allowed for temporary validation,
  but tag-triggered publishes should use the pinned file.
- Main-branch CI automatically dispatches `publish_assets.yml` after a successful
  build/smoke only when the pushed commit range changed `llama_cpp.version`. The
  dispatch passes the validated source SHA and lets the serialized publish
  workflow compute the next patch tag from `llama-web-bridge-assets`; PR CI must
  never publish assets.
- The auto-update workflow manages the stable `automation/bump-llama-cpp` branch
  and updates an existing PR instead of opening duplicates. It should include the
  upstream release notes, compare link, and commit range in the PR body, then
  create the PR with `WEBGPU_BRIDGE_ASSETS_PAT` so normal `pull_request` CI runs
  automatically, and wait for the exact automation head SHA.
- Preserve `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` in CI and publish workflows so
  GitHub Action runtime changes are detected before they become mandatory.
- Upload state-persistence smoke diagnostics only on failure; successful CI runs
  should stay quiet beyond the normal build artifacts.
- Do not push branches, tags, or publish assets from local agent work unless the
  maintainer explicitly requests that side effect.

## Publish Process

Use workflow `.github/workflows/publish_assets.yml`:

Automatic publish from CI:

1. Merge a PR that changes `llama_cpp.version`.
2. Let the `main` CI build and browser smoke pass.
3. The final CI job dispatches `publish_assets.yml` with `assets_tag=auto` and
   `source_ref` set to the validated main commit. The serialized publish workflow
   computes the next patch assets tag and leaves `llama_cpp_tag` empty so the
   publish reads the merged pin.

Manual publish:

1. Set input `assets_tag` (new tag).
2. Optionally set `assets_repo`; leave `llama_cpp_tag` empty to use
   `llama_cpp.version`, or set it only for an explicit temporary override.
3. Ensure `WEBGPU_BRIDGE_ASSETS_PAT` is configured with contents and
   pull-request write access to this repository plus write access to the assets
   repository.
4. Workflow builds, generates `manifest.json`/`sha256sums.txt`, pushes to
   assets repo, creates matching tag there, and uses the build job's resolved
   `llama.cpp` tag output in release notes.

## Repository Boundaries

- Bridge runtime source/build belongs here.
- Versioned static artifacts belong in `llama-web-bridge-assets`.
- Consumer integration (loading/fallback behavior) belongs in `llamadart`.
