#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
BRIDGE_DIR="$ROOT_DIR"

DEFAULT_LLAMA_CPP_DIR="$ROOT_DIR/third_party/llama_cpp"
SIBLING_LLAMA_CPP_DIR="$ROOT_DIR/../llama.cpp"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$DEFAULT_LLAMA_CPP_DIR}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/.build/webgpu_bridge}"
MEM64_BUILD_DIR="${MEM64_BUILD_DIR:-$ROOT_DIR/.build/webgpu_bridge_mem64}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/dist}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
BUILD_MEM64="${WEBGPU_BRIDGE_BUILD_MEM64:-0}"
MEM64_MAX_MEMORY="${WEBGPU_BRIDGE_MEM64_MAX_MEMORY:-12884901888}"
ENABLE_PTHREADS="${WEBGPU_BRIDGE_PTHREADS:-1}"
PTHREAD_POOL_SIZE="${WEBGPU_BRIDGE_PTHREAD_POOL_SIZE:-4}"
PTHREAD_POOL_SIZE_STRICT="${WEBGPU_BRIDGE_PTHREAD_POOL_SIZE_STRICT:-0}"
ALLOW_MEMORY_GROWTH="${WEBGPU_BRIDGE_ALLOW_MEMORY_GROWTH:-1}"
INITIAL_MEMORY="${WEBGPU_BRIDGE_INITIAL_MEMORY:-0}"

if [[ "$ENABLE_PTHREADS" == "0" ]]; then
  CMAKE_PTHREADS="OFF"
else
  CMAKE_PTHREADS="ON"
fi

if [[ "$ALLOW_MEMORY_GROWTH" == "0" ]]; then
  CMAKE_ALLOW_MEMORY_GROWTH="OFF"
else
  CMAKE_ALLOW_MEMORY_GROWTH="ON"
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Build llama-web-bridge wasm/js artifacts.

Requirements:
  npm               JS bridge bundling/type-checking
  emcmake, emcc     Emscripten SDK tools in PATH
  cmake             CMake configure/build driver
  llama.cpp         Source checkout via LLAMA_CPP_DIR or ../llama.cpp

Environment variables:
  LLAMA_CPP_DIR      Path to llama.cpp source checkout
  BUILD_DIR          CMake build directory
  OUT_DIR            Output directory for built assets
  CMAKE_BUILD_TYPE   CMake build type (default: Release)
  WEBGPU_BRIDGE_BUILD_MEM64  Build optional wasm64 artifacts (1/0)
  WEBGPU_BRIDGE_MEM64_MAX_MEMORY  wasm64 max linear memory bytes (default: 12884901888)
  WEBGPU_BRIDGE_PTHREADS  Enable pthread runtime support (default: 1)
  WEBGPU_BRIDGE_PTHREAD_POOL_SIZE  PThread pool size when enabled (default: 4)
  WEBGPU_BRIDGE_PTHREAD_POOL_SIZE_STRICT  PThread strictness when enabled (default: 0)
  WEBGPU_BRIDGE_ALLOW_MEMORY_GROWTH  Allow wasm memory growth (default: 1)
  WEBGPU_BRIDGE_INITIAL_MEMORY  Fixed wasm memory bytes when growth disabled

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

if ! command -v npm >/dev/null 2>&1; then
  echo "error: npm not found in PATH"
  exit 1
fi

if [[ ! -d "$BRIDGE_DIR/node_modules" ]]; then
  echo "[bridge] installing JS build dependencies"
  (cd "$BRIDGE_DIR" && npm ci --include=dev --ignore-scripts)
fi

echo "[bridge] building JS bridge wrapper"
(cd "$BRIDGE_DIR" && npm run check:js)

mkdir -p "$BUILD_DIR"
mkdir -p "$OUT_DIR"

echo "[bridge] configuring with emcmake"
emcmake cmake \
  -S "$BRIDGE_DIR" \
  -B "$BUILD_DIR" \
  -DLLAMA_CPP_DIR="$LLAMA_CPP_DIR" \
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -DLLAMADART_WEBGPU_PTHREADS="$CMAKE_PTHREADS" \
  -DLLAMADART_WEBGPU_PTHREAD_POOL_SIZE="$PTHREAD_POOL_SIZE" \
  -DLLAMADART_WEBGPU_PTHREAD_POOL_SIZE_STRICT="$PTHREAD_POOL_SIZE_STRICT" \
  -DLLAMADART_WEBGPU_ALLOW_MEMORY_GROWTH="$CMAKE_ALLOW_MEMORY_GROWTH" \
  -DLLAMADART_WEBGPU_INITIAL_MEMORY="$INITIAL_MEMORY"

echo "[bridge] building"
cmake --build "$BUILD_DIR" -j "$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"

CORE_JS="$BUILD_DIR/artifacts/llama_webgpu_core.js"
CORE_WASM="$BUILD_DIR/artifacts/llama_webgpu_core.wasm"
BRIDGE_JS="$BRIDGE_DIR/js/llama_webgpu_bridge.js"
BRIDGE_WORKER_JS="$BRIDGE_DIR/js/llama_webgpu_bridge_worker.js"
BRIDGE_DTS="$BRIDGE_DIR/js/llama_webgpu_bridge.d.ts"

if [[ ! -f "$CORE_JS" || ! -f "$CORE_WASM" || ! -f "$BRIDGE_JS" || ! -f "$BRIDGE_WORKER_JS" || ! -f "$BRIDGE_DTS" ]]; then
  echo "error: expected build outputs were not found"
  exit 1
fi

cp "$CORE_JS" "$OUT_DIR/llama_webgpu_core.js"
cp "$CORE_WASM" "$OUT_DIR/llama_webgpu_core.wasm"
cp "$BRIDGE_JS" "$OUT_DIR/llama_webgpu_bridge.js"
cp "$BRIDGE_WORKER_JS" "$OUT_DIR/llama_webgpu_bridge_worker.js"
cp "$BRIDGE_DTS" "$OUT_DIR/llama_webgpu_bridge.d.ts"

if [[ "$BUILD_MEM64" == "1" ]]; then
  mkdir -p "$MEM64_BUILD_DIR"

  echo "[bridge] configuring optional wasm64 build"
  emcmake cmake \
    -S "$BRIDGE_DIR" \
    -B "$MEM64_BUILD_DIR" \
    -DLLAMA_CPP_DIR="$LLAMA_CPP_DIR" \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -DLLAMADART_WEBGPU_MEM64=ON \
    -DLLAMADART_WEBGPU_MEM64_MAX_MEMORY="$MEM64_MAX_MEMORY" \
    -DLLAMADART_WEBGPU_PTHREADS="$CMAKE_PTHREADS" \
    -DLLAMADART_WEBGPU_PTHREAD_POOL_SIZE="$PTHREAD_POOL_SIZE" \
    -DLLAMADART_WEBGPU_PTHREAD_POOL_SIZE_STRICT="$PTHREAD_POOL_SIZE_STRICT" \
    -DLLAMADART_WEBGPU_ALLOW_MEMORY_GROWTH="$CMAKE_ALLOW_MEMORY_GROWTH" \
    -DLLAMADART_WEBGPU_INITIAL_MEMORY="$INITIAL_MEMORY"

  echo "[bridge] building optional wasm64 artifacts"
  cmake --build "$MEM64_BUILD_DIR" -j "$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"

  CORE_MEM64_JS="$MEM64_BUILD_DIR/artifacts/llama_webgpu_core_mem64.js"
  CORE_MEM64_WASM="$MEM64_BUILD_DIR/artifacts/llama_webgpu_core_mem64.wasm"

  if [[ ! -f "$CORE_MEM64_JS" || ! -f "$CORE_MEM64_WASM" ]]; then
    echo "error: expected wasm64 build outputs were not found"
    exit 1
  fi

  cp "$CORE_MEM64_JS" "$OUT_DIR/llama_webgpu_core_mem64.js"
  cp "$CORE_MEM64_WASM" "$OUT_DIR/llama_webgpu_core_mem64.wasm"

  echo "[bridge] applying wasm64 runtime bigint interop patch"
  python3 - <<'PY' "$OUT_DIR/llama_webgpu_core_mem64.js"
from pathlib import Path
import re
import sys

target = Path(sys.argv[1])
text = target.read_text(encoding='utf-8', errors='ignore')

def bigint_or_name(name):
    return rf"(?:BigInt\(\s*{name}\s*\)|{name})"

data_buffer = bigint_or_name("dataBuffer")
length = bigint_or_name("length")
position = bigint_or_name("position")
offset = bigint_or_name("offset")

replacements = [
    (
        rf"__wasmfs_read\(\s*stream\.fd\s*,\s*{data_buffer}\s*,\s*{length}\s*\)",
        "__wasmfs_read(stream.fd,BigInt(dataBuffer),BigInt(length))",
    ),
    (
        rf"__wasmfs_pread\(\s*stream\.fd\s*,\s*{data_buffer}\s*,\s*{length}\s*,\s*{position}\s*\)",
        "__wasmfs_pread(stream.fd,BigInt(dataBuffer),BigInt(length),BigInt(position))",
    ),
    (
        rf"__wasmfs_write\(\s*stream\.fd\s*,\s*{data_buffer}\s*,\s*{length}\s*\)",
        "__wasmfs_write(stream.fd,BigInt(dataBuffer),BigInt(length))",
    ),
    (
        rf"__wasmfs_pwrite\(\s*stream\.fd\s*,\s*{data_buffer}\s*,\s*{length}\s*,\s*{position}\s*\)",
        "__wasmfs_pwrite(stream.fd,BigInt(dataBuffer),BigInt(length),BigInt(position))",
    ),
    (
        rf"__wasmfs_mmap\(\s*{length}\s*,\s*prot\s*,\s*flags\s*,\s*stream\.fd\s*,\s*{offset}\s*\)",
        "__wasmfs_mmap(BigInt(length),prot,flags,stream.fd,BigInt(offset))",
    ),
]

changed = False
for pattern, replacement in replacements:
    text, count = re.subn(pattern, replacement, text)
    if count > 0:
        changed = True

if not changed:
    raise SystemExit('error: wasm64 runtime patch did not match expected symbols')

target.write_text(text, encoding='utf-8')
PY
fi

echo "[bridge] done"
echo "  - $OUT_DIR/llama_webgpu_bridge.js"
echo "  - $OUT_DIR/llama_webgpu_bridge_worker.js"
echo "  - $OUT_DIR/llama_webgpu_bridge.d.ts"
echo "  - $OUT_DIR/llama_webgpu_core.js"
echo "  - $OUT_DIR/llama_webgpu_core.wasm"
if [[ "$BUILD_MEM64" == "1" ]]; then
  echo "  - $OUT_DIR/llama_webgpu_core_mem64.js"
  echo "  - $OUT_DIR/llama_webgpu_core_mem64.wasm"
fi
