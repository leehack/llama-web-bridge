import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const root = new URL('../../', import.meta.url);
const cmake = readFileSync(new URL('CMakeLists.txt', root), 'utf8');
const publicConstructors = [
  'BlobWorkerBlockedError', 'CoreAssetError', 'CrossOriginIsolationError', 'EmbeddingError',
  'LlamaWebGpuBridge', 'ModelStagingError', 'ModelUnloadedError', 'SingleThreadCoreError',
  'TextToSpeechStreamingUnsupportedError',
];

test('Release sizes model and bridge code while leaving CPU kernels at CMake defaults', () => {
  const targets = cmake.match(/foreach\(target IN ITEMS ([^)]+)\)\s+target_compile_options\(\$\{target\} PRIVATE "\$<\$<CONFIG:Release>:-Os>"\)/);
  assert.ok(targets, 'missing per-target Release -Os options');
  assert.deepEqual(targets[1].split(/\s+/).sort(), [
    'ggml-webgpu', 'llama', 'llama-common', 'llama-common-base', 'llama_webgpu_core', 'mtmd',
  ]);
  assert.match(cmake, /target_link_options\(llama_webgpu_core PRIVATE "\$<\$<CONFIG:Release>:-Os>"\s+\$\{LLAMADART_WEBGPU_LINK_OPTIONS\}\)/);
  assert.doesNotMatch(cmake, /CMAKE_(?:C|CXX)_FLAGS_RELEASE|add_compile_options\([^)]*-O[sz]|target_compile_options\(ggml-(?:cpu|base)[^)]*-O[sz]/);
});

test('the bridge removes upstream WebGPU exception flags and guards the source assumption', () => {
  const downstream = cmake.slice(cmake.indexOf('add_subdirectory('));
  assert.match(downstream, /file\(GLOB_RECURSE ggml_webgpu_sources CONFIGURE_DEPENDS/);
  assert.match(downstream, /foreach\(source IN LISTS ggml_webgpu_sources\)[\s\S]*?try[\s\S]*?catch[\s\S]*?message\(FATAL_ERROR/);
  assert.match(downstream, /get_target_property\(webgpu_compile_options ggml-webgpu COMPILE_OPTIONS\)\s+list\(REMOVE_ITEM webgpu_compile_options "-fexceptions"\)\s+set_property\(TARGET ggml-webgpu PROPERTY COMPILE_OPTIONS "\$\{webgpu_compile_options\}"\)/);
  assert.match(downstream, /get_target_property\(webgpu_link_options ggml-webgpu INTERFACE_LINK_OPTIONS\)\s+list\(REMOVE_ITEM webgpu_link_options "-exceptions"\)\s+set_property\(TARGET ggml-webgpu PROPERTY INTERFACE_LINK_OPTIONS "\$\{webgpu_link_options\}"\)/);
  assert.match(cmake, /"-sASYNCIFY=1"/);
  assert.match(cmake, /"-sFORCE_FILESYSTEM=1"/);
  assert.match(cmake, /\$\{LLAMADART_WEBGPU_EXCEPTION_CATCHING_ALLOWED\}/);
});

for (const file of ['llama_webgpu_bridge.js', 'llama_webgpu_bridge_worker.js']) {
  test(`${file} is minified and keeps the core import hints`, () => {
    const code = readFileSync(new URL(`js/${file}`, root), 'utf8');
    assert.ok(Buffer.byteLength(code) < 260 * 1024, 'expected a minified bundle below 260 KiB');
    assert.match(code, /\bimport\(\/\* @vite-ignore \*\/ \/\* webpackIgnore: true \*\/\s*[$\w]+\)/);
  });
}

test('the public bundle preserves constructor names', async () => {
  const bridge = await import(new URL('js/llama_webgpu_bridge.js', root));
  for (const name of publicConstructors) {
    assert.equal(bridge[name]?.name, name);
  }
});

test('the standalone worker retains the same constructor name metadata', () => {
  const code = readFileSync(new URL('js/llama_webgpu_bridge_worker.js', root), 'utf8');
  for (const name of publicConstructors) {
    // The standalone worker installs the host without exporting its constructors.
    assert.ok(new RegExp(`,\\s*"${name}"\\s*\\)`).test(code), `missing worker name for ${name}`);
  }
});
