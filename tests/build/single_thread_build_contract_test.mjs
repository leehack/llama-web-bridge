// Static contracts for single-thread (_st) core builds and artifact wiring.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { ARTIFACTS } from '../../scripts/release/manifest.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const read = (relativePath) => fs.readFileSync(path.join(ROOT, relativePath), 'utf8');

test('CMakeLists.txt keeps pthreads ON by default and appends optional output suffix', () => {
  const cmake = read('CMakeLists.txt');
  assert.match(cmake, /option\(LLAMADART_WEBGPU_PTHREADS\s+"[^"]+"\s+ON\)/);
  assert.match(cmake, /set\(LLAMADART_WEBGPU_CORE_OUTPUT_SUFFIX\s+""\s+CACHE STRING\b/);
  assert.match(cmake, /set\(LLAMADART_WEBGPU_CORE_OUTPUT_NAME\s+"llama_webgpu_core"\)/);
  assert.match(cmake, /if\s*\(LLAMADART_WEBGPU_CORE_OUTPUT_SUFFIX\)\s*set\(LLAMADART_WEBGPU_CORE_OUTPUT_NAME\s+"\$\{LLAMADART_WEBGPU_CORE_OUTPUT_NAME\}\$\{LLAMADART_WEBGPU_CORE_OUTPUT_SUFFIX\}"\)/);
});

test('build_bridge.sh wires single-thread wasm32 and memory64 passes cleanly', () => {
  const script = read('scripts/build/build_bridge.sh');
  assert.match(script, /BUILD_SINGLE_THREAD="\$\{WEBGPU_BRIDGE_BUILD_SINGLE_THREAD:-0\}"/);
  assert.match(script, /SINGLE_THREAD_BUILD_DIR="\$\{SINGLE_THREAD_BUILD_DIR:-\$ROOT_DIR\/\.build\/webgpu_bridge_st\}"/);
  assert.match(script, /SINGLE_THREAD_MEM64_BUILD_DIR="\$\{SINGLE_THREAD_MEM64_BUILD_DIR:-\$ROOT_DIR\/\.build\/webgpu_bridge_mem64_st\}"/);
  assert.match(script, /WEBGPU_BRIDGE_BUILD_SINGLE_THREAD/);
  assert.match(script, /-B "\$SINGLE_THREAD_BUILD_DIR"[\s\S]*?-DLLAMADART_WEBGPU_CORE_OUTPUT_SUFFIX="_st"[\s\S]*?-DLLAMADART_WEBGPU_PTHREADS=OFF/);
  assert.match(script, /-B "\$SINGLE_THREAD_MEM64_BUILD_DIR"[\s\S]*?-DLLAMADART_WEBGPU_MEM64=ON[\s\S]*?-DLLAMADART_WEBGPU_CORE_OUTPUT_SUFFIX="_st"[\s\S]*?-DLLAMADART_WEBGPU_PTHREADS=OFF/);
  assert.match(script, /patch_wasm64_runtime\.mjs"\s*\\\s*"\$OUT_DIR\/llama_webgpu_core_mem64_st\.js"/);
  assert.match(script, /\$OUT_DIR\/llama_webgpu_core_st\.js/);
  assert.match(script, /\$OUT_DIR\/llama_webgpu_core_st\.wasm/);
  assert.match(script, /\$OUT_DIR\/llama_webgpu_core_mem64_st\.js/);
  assert.match(script, /\$OUT_DIR\/llama_webgpu_core_mem64_st\.wasm/);
});

test('manifest ARTIFACTS includes all 11 bridge and core files', () => {
  assert.deepEqual(ARTIFACTS, [
    'llama_webgpu_bridge.js',
    'llama_webgpu_bridge_worker.js',
    'llama_webgpu_bridge.d.ts',
    'llama_webgpu_core.js',
    'llama_webgpu_core.wasm',
    'llama_webgpu_core_mem64.js',
    'llama_webgpu_core_mem64.wasm',
    'llama_webgpu_core_st.js',
    'llama_webgpu_core_st.wasm',
    'llama_webgpu_core_mem64_st.js',
    'llama_webgpu_core_mem64_st.wasm',
  ]);
});
