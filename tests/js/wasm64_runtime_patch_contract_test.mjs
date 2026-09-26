// Regression contracts for the generated wasm64 JavaScript patch
// (scripts/patch_wasm64_runtime.mjs).
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { PATCHES, PatchError, patchWasm64Runtime } from '../../scripts/patch_wasm64_runtime.mjs';

// Representative fragments from the Emscripten 6.0.8 output shape validated on
// main. Both raw numeric values and already-wrapped values are included because
// the patch is intentionally idempotent.
const CURRENT_EMSCRIPTEN_OUTPUT = `
bytesRead=__wasmfs_pread(stream.fd,dataBuffer,length,position);
bytesRead=__wasmfs_read(stream.fd,BigInt(dataBuffer),BigInt(length));
bytesRead=__wasmfs_pwrite(stream.fd,dataBuffer,length,position);
bytesRead=__wasmfs_write(stream.fd,BigInt(dataBuffer),BigInt(length));
allocated=__wasmfs_mmap(length,prot,flags,stream.fd,offset);
`;

// The patch covers exactly the five WASMFS imports, and the current
// Emscripten output matches each of them exactly once.
{
  assert.deepEqual(PATCHES.map(([name]) => name), [
    '__wasmfs_read', '__wasmfs_pread', '__wasmfs_write', '__wasmfs_pwrite', '__wasmfs_mmap',
  ]);
  const { text, counts } = patchWasm64Runtime(CURRENT_EMSCRIPTEN_OUTPUT);
  assert.deepEqual(counts, Object.fromEntries(PATCHES.map(([name]) => [name, 1])));
  for (const [, , replacement] of PATCHES) assert.ok(text.includes(replacement), `patched output must contain ${replacement}`);
}

// Each single match still fails the five-symbol contract and names the others.
{
  const lines = CURRENT_EMSCRIPTEN_OUTPUT.split('\n').filter((line) => line);
  assert.equal(lines.length, 5);
  for (const [onlyName] of PATCHES) {
    const onlyLine = lines.find((line) => line.includes(onlyName));
    assert.throws(() => patchWasm64Runtime(onlyLine), (error) => {
      assert.ok(error instanceof PatchError, `only ${onlyName}: expected a PatchError`);
      for (const [requiredName] of PATCHES) {
        if (requiredName !== onlyName) {
          assert.ok(error.message.includes(requiredName), `only ${onlyName}: error must name ${requiredName}`);
        }
      }
      return true;
    });
  }
}

// Each missing symbol is named independently.
for (const [missingName] of PATCHES) {
  const partial = CURRENT_EMSCRIPTEN_OUTPUT.split('\n').filter((line) => !line.includes(missingName)).join('\n');
  assert.throws(() => patchWasm64Runtime(partial), (error) => {
    assert.ok(error instanceof PatchError, `missing ${missingName}: expected a PatchError`);
    assert.match(error.message, new RegExp(missingName));
    return true;
  });
}

// The command line patches the file in place, reports it, and is idempotent;
// a missing symbol exits 1 and leaves the file untouched.
{
  const script = fileURLToPath(new URL('../../scripts/patch_wasm64_runtime.mjs', import.meta.url));
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'wasm64-patch-'));
  try {
    const target = path.join(directory, 'core_mem64.js');
    const run = () => spawnSync(process.execPath, [script, target], { encoding: 'utf8' });
    fs.writeFileSync(target, CURRENT_EMSCRIPTEN_OUTPUT);
    const first = run();
    assert.equal(first.status, 0, first.stderr);
    assert.match(first.stdout, /^Patched wasm64 generated JavaScript: /);
    const patched = fs.readFileSync(target, 'utf8');
    assert.equal(patched, patchWasm64Runtime(CURRENT_EMSCRIPTEN_OUTPUT).text);
    assert.notEqual(patched, CURRENT_EMSCRIPTEN_OUTPUT);
    assert.equal(run().status, 0);
    assert.equal(fs.readFileSync(target, 'utf8'), patched);

    const partial = CURRENT_EMSCRIPTEN_OUTPUT.replace(/.*__wasmfs_mmap.*\n/, '');
    fs.writeFileSync(target, partial);
    const missing = run();
    assert.equal(missing.status, 1);
    assert.match(missing.stderr, /__wasmfs_mmap/);
    assert.equal(fs.readFileSync(target, 'utf8'), partial);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
}

console.log('wasm64 runtime patch contract passed');
