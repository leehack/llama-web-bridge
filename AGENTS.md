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
