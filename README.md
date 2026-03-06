# llama-web-bridge

Reusable llama.cpp web bridge runtime (JS + WASM).

This repository provides:

- `src/llama_webgpu_core.cpp` (native bridge core)
- `js/llama_webgpu_bridge.js` (JS runtime wrapper)
- `CMakeLists.txt` for Emscripten builds

## Build

Requirements:

- Emscripten SDK (`emcmake`, `emcc`) in `PATH`
- llama.cpp source checkout

Build command:

```bash
./scripts/build_bridge.sh
```

Useful environment variables:

- `LLAMA_CPP_DIR` (path to llama.cpp source)
- `BUILD_DIR` (cmake build dir)
- `OUT_DIR` (output directory; defaults to `dist/`)
- `WEBGPU_BRIDGE_BUILD_MEM64` (`1` to also build optional wasm64 core assets)
- `WEBGPU_BRIDGE_MEM64_MAX_MEMORY` (optional wasm64 max linear memory bytes)
- `WEBGPU_BRIDGE_PTHREADS` (`1`/`0`, defaults to `1`)
- `WEBGPU_BRIDGE_PTHREAD_POOL_SIZE` (defaults to `4`)

Notes:

- wasm64 builds default to `WEBGPU_BRIDGE_MEM64_MAX_MEMORY=12884901888` (12 GiB).
- Large single-file remote model loading requires a cross-origin isolated page
  (`COOP`/`COEP`) so worker-thread runtime paths are available.
- pthread builds enable `-sPTHREAD_POOL_SIZE_STRICT=2` so pool exhaustion
  throws explicit errors instead of risking deadlock.

Build outputs:

- `dist/llama_webgpu_bridge.js`
- `dist/llama_webgpu_bridge_worker.js`
- `dist/llama_webgpu_core.js`
- `dist/llama_webgpu_core.wasm`

Optional outputs (when `WEBGPU_BRIDGE_BUILD_MEM64=1`):

- `dist/llama_webgpu_core_mem64.js`
- `dist/llama_webgpu_core_mem64.wasm`

## CI

This repo includes a wasm build gate in:

- `.github/workflows/ci.yml`

It builds against pinned `llama.cpp` tag `b8157` and uploads build artifacts.

## Publishing

Published, versioned artifacts are consumed from:

- `leehack/llama-web-bridge-assets`

Publish workflow:

- `.github/workflows/publish_assets.yml`

Trigger modes:

- Automatic: push a `v*` tag in this repo (for example `v0.1.5`)
- Manual: run workflow dispatch with explicit inputs

Required repository secret:

- `WEBGPU_BRIDGE_ASSETS_PAT` (token with write access to
  `leehack/llama-web-bridge-assets`)

Example publish:

1. Create/push a release tag in this repo (for example `v0.1.5`)
2. `Publish Bridge Assets` runs automatically and publishes the same tag to
   `leehack/llama-web-bridge-assets`
3. Workflow also creates/updates the matching GitHub Release in
   `leehack/llama-web-bridge-assets`

Manual override example:

1. Run `Publish Bridge Assets` workflow
2. Inputs:
   - `assets_tag`: `v0.1.5`
   - `assets_repo`: `leehack/llama-web-bridge-assets`
   - `llama_cpp_tag`: `b8157`

After publish, assets are CDN-available at:

- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@v0.1.1/llama_webgpu_bridge.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@v0.1.1/llama_webgpu_bridge_worker.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@v0.1.1/llama_webgpu_core.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@v0.1.1/llama_webgpu_core.wasm`

Note: CDN pinning fundamentally relies on git tags in the assets repo.

## Maintainer Docs

- `AGENTS.md`: agent workflow and cross-repo handoff
- `CONTRIBUTING.md`: contributor setup/build/publish steps
