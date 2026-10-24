import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import test from 'node:test';

const root = new URL('../../', import.meta.url);

test('the wasmfs fetch patch stays deleted and the stock fetch backend stays linked', () => {
  assert.equal(existsSync(new URL('js/wasmfs_fetch_patch.js', root)), false);
  const cmake = readFileSync(new URL('CMakeLists.txt', root), 'utf8');
  assert.equal(cmake.match(/-lwasmfs_fetch\.js/g)?.length, 1);
  assert.doesNotMatch(cmake, /wasmfs_fetch_patch/);
});

test('bridge sources do not read fetch-backend globals nothing publishes', () => {
  for (const file of ['js/src/runtime.ts', 'js/llama_webgpu_bridge.js']) {
    assert.doesNotMatch(readFileSync(new URL(file, root), 'utf8'), /__llamadartFetchBackend/);
  }
});
