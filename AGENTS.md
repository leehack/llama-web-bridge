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

- `LLAMA_CPP_DIR`
- `BUILD_DIR`
- `OUT_DIR`
- `CMAKE_BUILD_TYPE`

### Local Verification Notes

When validating bridge runtime changes locally, keep build/cache output outside
the repo so generated wasm artifacts and toolchain caches do not dirty the
checkout or hit sandboxed Homebrew/cache paths:

```bash
export CCACHE_DIR=/private/tmp/llama_web_bridge_ccache
export EM_CACHE=/private/tmp/llama_web_bridge_emcache
BUILD_DIR=/private/tmp/llama_web_bridge_build MEM64_BUILD_DIR=/private/tmp/llama_web_bridge_build_mem64 OUT_DIR=/private/tmp/llama_web_bridge_dist WEBGPU_BRIDGE_BUILD_MEM64=1 ./scripts/build_bridge.sh
```

## CI / Release

- CI build gate: `.github/workflows/ci.yml`
- Publish workflow: `.github/workflows/publish_assets.yml`
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
