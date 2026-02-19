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

Build outputs:

- `dist/llama_webgpu_bridge.js`
- `dist/llama_webgpu_core.js`
- `dist/llama_webgpu_core.wasm`

## CI

This repo includes a wasm build gate in:

- `.github/workflows/ci.yml`

It builds against pinned `llama.cpp` tag `b8011` and uploads build artifacts.

## Publishing

Published, versioned artifacts are consumed from:

- `leehack/llama-web-bridge-assets`

Publish workflow:

- `.github/workflows/publish_assets.yml`

Required repository secret:

- `WEBGPU_BRIDGE_ASSETS_PAT` (token with write access to
  `leehack/llama-web-bridge-assets`)

Example publish:

1. Run `Publish Bridge Assets` workflow
2. Inputs:
   - `assets_tag`: `v0.1.1`
   - `assets_repo`: `leehack/llama-web-bridge-assets`
   - `llama_cpp_tag`: `b8011`

After publish, assets are CDN-available at:

- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@v0.1.1/llama_webgpu_bridge.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@v0.1.1/llama_webgpu_core.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@v0.1.1/llama_webgpu_core.wasm`

## Maintainer Docs

- `AGENTS.md`: agent workflow and cross-repo handoff
- `CONTRIBUTING.md`: contributor setup/build/publish steps
