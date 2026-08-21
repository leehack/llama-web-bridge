import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporaryDir = mkdtempSync(path.join(tmpdir(), 'llama-web-bridge-embedding-json-'));
const sourcePath = path.join(temporaryDir, 'embedding_json_contract_test.cpp');
const executablePath = path.join(temporaryDir, 'embedding_json_contract_test');

const source = String.raw`
#include <iostream>
#include <limits>
#include <vector>

#include "src/llama_webgpu_embedding_json.h"

int main() {
  const std::vector<float> values = {
      0.0123456789f,
      3.7e-8f,
      1.23456789f,
      std::numeric_limits<float>::denorm_min(),
      -0.0f,
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
  };
  std::cout << llamadart_webgpu_detail::serialize_embedding_json(values);
  return 0;
}
`;

function float32Bits(value) {
  const bytes = new ArrayBuffer(4);
  const view = new DataView(bytes);
  view.setFloat32(0, value, true);
  return view.getUint32(0, true);
}

try {
  writeFileSync(sourcePath, source);
  execFileSync(process.env.CXX || 'c++', [
    '-std=c++17',
    '-Wall',
    '-Wextra',
    '-Werror',
    '-I',
    rootDir,
    sourcePath,
    '-o',
    executablePath,
  ], { stdio: 'inherit' });

  const raw = execFileSync(executablePath, { encoding: 'utf8' });
  assert.doesNotMatch(raw, /nan|inf/i, 'serializer emitted a non-JSON number');
  assert.match(raw, /e[-+]\d+/i, 'small values did not use scientific notation');

  const parsed = JSON.parse(raw);
  assert.equal(parsed.length, 8);

  const finiteInputs = [
    Math.fround(0.0123456789),
    Math.fround(3.7e-8),
    Math.fround(1.23456789),
    Math.fround(2 ** -149),
    Math.fround(-0),
  ];
  for (let i = 0; i < finiteInputs.length; i += 1) {
    assert.equal(
      float32Bits(parsed[i]),
      float32Bits(finiteInputs[i]),
      `float32 component ${i} did not round-trip exactly`,
    );
  }

  assert.notEqual(parsed[1], 0, 'small embedding component was rounded to zero');
  assert.deepEqual(parsed.slice(5), [0, 0, 0]);
} finally {
  rmSync(temporaryDir, { recursive: true, force: true });
}

console.log('Embedding JSON C++/JS contract tests passed');
