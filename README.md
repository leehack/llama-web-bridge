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

## Publishing

Published, versioned artifacts are consumed from:

- `leehack/llama-web-bridge-assets`
