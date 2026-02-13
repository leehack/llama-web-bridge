#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
BRIDGE_DIR="$ROOT_DIR"

DEFAULT_LLAMA_CPP_DIR="$ROOT_DIR/third_party/llama_cpp"
SIBLING_LLAMA_CPP_DIR="$ROOT_DIR/../llama.cpp"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$DEFAULT_LLAMA_CPP_DIR}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/.build/webgpu_bridge}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/dist}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Build llama-web-bridge wasm/js artifacts.

Environment variables:
  LLAMA_CPP_DIR      Path to llama.cpp source checkout
  BUILD_DIR          CMake build directory
  OUT_DIR            Output directory for built assets
  CMAKE_BUILD_TYPE   CMake build type (default: Release)

Example:
  LLAMA_CPP_DIR="$PWD/../llama.cpp" ./scripts/build_bridge.sh
USAGE
  exit 0
fi

if ! command -v emcmake >/dev/null 2>&1; then
  echo "error: emcmake not found in PATH"
  exit 1
fi

if ! command -v emcc >/dev/null 2>&1; then
  echo "error: emcc not found in PATH"
  exit 1
fi

if [[ ! -f "$LLAMA_CPP_DIR/CMakeLists.txt" && -f "$SIBLING_LLAMA_CPP_DIR/CMakeLists.txt" ]]; then
  LLAMA_CPP_DIR="$SIBLING_LLAMA_CPP_DIR"
fi

if [[ ! -f "$LLAMA_CPP_DIR/CMakeLists.txt" ]]; then
  echo "error: llama.cpp source not found at: $LLAMA_CPP_DIR"
  exit 1
fi

mkdir -p "$BUILD_DIR"
mkdir -p "$OUT_DIR"

echo "[bridge] configuring with emcmake"
emcmake cmake \
  -S "$BRIDGE_DIR" \
  -B "$BUILD_DIR" \
  -DLLAMA_CPP_DIR="$LLAMA_CPP_DIR" \
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"

echo "[bridge] building"
cmake --build "$BUILD_DIR" -j "$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"

CORE_JS="$BUILD_DIR/artifacts/llama_webgpu_core.js"
CORE_WASM="$BUILD_DIR/artifacts/llama_webgpu_core.wasm"
BRIDGE_JS="$BRIDGE_DIR/js/llama_webgpu_bridge.js"

if [[ ! -f "$CORE_JS" || ! -f "$CORE_WASM" || ! -f "$BRIDGE_JS" ]]; then
  echo "error: expected build outputs were not found"
  exit 1
fi

cp "$CORE_JS" "$OUT_DIR/llama_webgpu_core.js"
cp "$CORE_WASM" "$OUT_DIR/llama_webgpu_core.wasm"
cp "$BRIDGE_JS" "$OUT_DIR/llama_webgpu_bridge.js"

echo "[bridge] done"
echo "  - $OUT_DIR/llama_webgpu_bridge.js"
echo "  - $OUT_DIR/llama_webgpu_core.js"
echo "  - $OUT_DIR/llama_webgpu_core.wasm"
