import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync, existsSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

const root = path.resolve(import.meta.dirname, '../..');
const modulePath = path.join(root, 'cmake/embedding_mode_reserve.cmake');

function configure(source) {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'embedding-reserve-'));
  try {
    mkdirSync(path.join(dir, 'upstream/src'), { recursive: true });
    const input = path.join(dir, 'upstream/src/llama-context.cpp');
    writeFileSync(input, source);
    const script = path.join(dir, 'check.cmake');
    writeFileSync(script, `set(LLAMA_CPP_DIR "${dir}/upstream")\n`
      + `set(CMAKE_BINARY_DIR "${dir}/build")\ninclude("${modulePath}")\n`);
    const result = spawnSync('cmake', ['-P', script], { encoding: 'utf8' });
    assert.equal(readFileSync(input, 'utf8'), source, 'upstream source must remain untouched');
    assert.equal(existsSync(path.join(dir, 'build/generated/llama-context-embedding-reserve.cpp')), false);
    return result;
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

test('unaffected upstream context is left unchanged', () => {
  const result = configure('void set_embeddings(bool value) { cparams.embeddings = value; }\n');
  assert.equal(result.status, 0, result.stderr);
});

test('an unaudited source with the affected guard fails before producing a replacement', () => {
  const result = configure('void set_embeddings(bool value) {\n'
    + 'if (cparams.embeddings != value && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE) {\n'
    + 'sched_need_reserve = true;\n}\ncparams.embeddings = value;\n}\n');
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Unaudited embedding-mode reserve guard/);
});

test('the production build compiles the corrected copy and keys its cache on the module', () => {
  const cmake = readFileSync(path.join(root, 'CMakeLists.txt'), 'utf8');
  assert.match(cmake, /include\(cmake\/embedding_mode_reserve\.cmake\)/);
  const module = readFileSync(modulePath, 'utf8');
  assert.match(module, /file\(SHA256/);
  assert.match(module, /HEADER_FILE_ONLY TRUE/);
  assert.match(module, /target_sources\(llama PRIVATE "\$\{LLAMADART_CONTEXT_PATCHED\}"\)/);
  assert.match(module, /SKIP_UNITY_BUILD_INCLUSION ON/);
  const ci = readFileSync(path.join(root, '.github/workflows/ci.yml'), 'utf8');
  assert.equal(ci.match(/hashFiles\('CMakeLists.txt', 'cmake\/\*\.cmake'/g)?.length, 2);
});
