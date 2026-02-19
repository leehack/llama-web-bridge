# Contributing to llama-web-bridge

Thanks for contributing.

## Purpose

This repo builds the reusable JS/WASM bridge runtime for llama.cpp web usage.
Published artifacts are consumed from `llama-web-bridge-assets`.

## Prerequisites

- Emscripten SDK (`emcmake`, `emcc`)
- CMake toolchain
- Access to a llama.cpp checkout

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

## Validate Outputs

Expected files:

- `dist/llama_webgpu_bridge.js`
- `dist/llama_webgpu_core.js`
- `dist/llama_webgpu_core.wasm`

## Publish Process

Use workflow `.github/workflows/publish_assets.yml`:

1. Set input `assets_tag` (new tag).
2. Optionally set `assets_repo` and `llama_cpp_tag`.
3. Ensure `WEBGPU_BRIDGE_ASSETS_PAT` secret is configured.
4. Workflow builds, generates `manifest.json`/`sha256sums.txt`, pushes to
   assets repo, and creates matching tag there.

## Repository Boundaries

- Bridge runtime source/build belongs here.
- Versioned static artifacts belong in `llama-web-bridge-assets`.
- Consumer integration (loading/fallback behavior) belongs in `llamadart`.
