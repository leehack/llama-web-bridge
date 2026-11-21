import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import test from 'node:test';

const root = new URL('../../', import.meta.url);

test('the bridge fetch backend replaces the stock library on pthread builds', () => {
  assert.equal(existsSync(new URL('js/wasmfs_fetch_patch.js', root)), false);
  const cmake = readFileSync(new URL('CMakeLists.txt', root), 'utf8');
  assert.doesNotMatch(cmake, /-lwasmfs_fetch\.js/);
  assert.doesNotMatch(cmake, /wasmfs_fetch_patch/);
  assert.match(cmake, /add_executable\(llama_webgpu_core[^)]*src\/llama_webgpu_fetch_fs\.cpp/);
  const threaded = [...cmake.matchAll(/^if \(LLAMADART_WEBGPU_PTHREADS\)\n([\s\S]*?)^endif\(\)/gm)];
  const link = threaded.find(([, block]) => block.includes('LINK_DEPENDS') && block.includes('--js-library='));
  assert.ok(link, 'missing pthread-only fetch library linkage and link dependency');
  for (const [, block] of threaded) {
    assert.equal(block.includes('llama_webgpu_fetch_fs.js'), block === link[1]);
  }
  assert.equal(cmake.match(/src\/llama_webgpu_fetch_fs\.js/g)?.length, 1);
  const [, variable] = link[1].match(/set\((\w+) "\$\{CMAKE_CURRENT_SOURCE_DIR\}\/src\/llama_webgpu_fetch_fs\.js"\)/) ?? [];
  assert.ok(variable, 'missing bridge fetch library path');
  const reference = '$' + '{' + variable + '}';
  assert.ok(link[1].includes('LINK_DEPENDS "' + reference + '"'));
  assert.ok(link[1].includes('"--js-library=' + reference + '"'));
});

test('bridge sources do not read fetch-backend globals nothing publishes', () => {
  for (const file of ['js/src/runtime.ts', 'js/llama_webgpu_bridge.js']) {
    assert.doesNotMatch(readFileSync(new URL(file, root), 'utf8'), /__llamadartFetchBackend/);
  }
});
