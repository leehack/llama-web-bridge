// Regression contracts for the generated wasm64 JavaScript patch
// (scripts/build/patch_wasm64_runtime.mjs).
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  assertWasmMemory64,
  Memory64Error,
  PATCHES,
  PatchError,
  patchWasm64Runtime,
  wasmMemoryAddressTypes,
} from '../../scripts/build/patch_wasm64_runtime.mjs';

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

// Tiny wasm modules shaped like the cores: a pthread link imports its shared
// memory (limits flags 0x03 wasm32, 0x07 memory64), a single-threaded link
// defines it (0x00, 0x04).
const u32 = (value) => {
  const out = [];
  do {
    out.push((value & 0x7f) | (value > 0x7f ? 0x80 : 0));
    value >>>= 7;
  } while (value > 0);
  return out;
};
const str = (text) => [...u32(text.length), ...Buffer.from(text)];
const section = (id, ...entries) => {
  const body = [...u32(entries.length), ...entries.flat()];
  return [id, ...u32(body.length), ...body];
};
const wasmModule = (...sections) => Uint8Array.from([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, ...sections.flat()]);
const importedMemory = (flags) => wasmModule(
  section(1, [0x60, 0x00, 0x00]),
  section(
    2,
    [...str('env'), ...str('f'), 0x00, 0x00],
    [...str('env'), ...str('g'), 0x03, 0x7f, 0x00],
    [...str('env'), ...str('memory'), 0x02, flags, 0x01, 0x80, 0x01],
  ),
);
const definedMemory = (flags) => wasmModule(section(5, [flags, 0x01]));
const WASM32_IMPORTED = importedMemory(0x03);
const WASM64_IMPORTED = importedMemory(0x07);

// The memory64 check reads imported and defined memories and accepts only
// memory64; a wasm32, memoryless, non-wasm or truncated module fails closed.
{
  assert.deepEqual(wasmMemoryAddressTypes(WASM32_IMPORTED), ['i32']);
  assert.deepEqual(wasmMemoryAddressTypes(WASM64_IMPORTED), ['i64']);
  assert.deepEqual(wasmMemoryAddressTypes(definedMemory(0x00)), ['i32']);
  assert.deepEqual(wasmMemoryAddressTypes(definedMemory(0x04)), ['i64']);
  const otherImportsAndPageSize = wasmModule(
    section(1, [0x60, 0x00, 0x00]),
    section(
      2,
      [...str('env'), ...str('t'), 0x01, 0x70, 0x00, 0x01],
      [...str('env'), ...str('e'), 0x04, 0x00, 0x00],
      [...str('env'), ...str('memory'), 0x02, 0x07, 0x01, 0x80, 0x01],
    ),
    section(5, [0x0c, 0x01, 0x10]),
  );
  assert.deepEqual(wasmMemoryAddressTypes(otherImportsAndPageSize), ['i64', 'i64']);
  assertWasmMemory64(WASM64_IMPORTED);
  assertWasmMemory64(definedMemory(0x04));
  for (const [label, bytes, message] of [
    ['wasm32 imported', WASM32_IMPORTED, /\[i32\], expected only i64/],
    ['wasm32 defined', definedMemory(0x00), /\[i32\], expected only i64/],
    ['no memory', wasmModule(), /\[\], expected only i64/],
    ['wasm64 and wasm32', wasmModule(...[WASM64_IMPORTED, definedMemory(0x00)].map((bytes) => [...bytes.subarray(8)])), /\[i64, i32\]/],
    ['generated JS', Buffer.from(CURRENT_EMSCRIPTEN_OUTPUT), /not a version 1 wasm binary module/],
    ['truncated', WASM64_IMPORTED.subarray(0, WASM64_IMPORTED.length - 1), /unexpected end of module/],
    ['unknown limits flags', definedMemory(0x14), /unknown limits flags 0x14/],
    ['section size mismatch', wasmModule([0x05, 0x05, 0x01, 0x04, 0x01, 0x00, 0x00]), /section 5 size/],
  ]) {
    assert.throws(() => assertWasmMemory64(bytes), (error) => {
      assert.ok(error instanceof Memory64Error, `${label}: expected a Memory64Error`);
      assert.match(error.message, message, label);
      return true;
    });
  }
}

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
// a sibling .wasm that is wasm32 or missing, or a missing symbol, exits 1 and
// leaves the file untouched.
{
  const script = fileURLToPath(new URL('../../scripts/build/patch_wasm64_runtime.mjs', import.meta.url));
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'wasm64-patch-'));
  try {
    const target = path.join(directory, 'core_mem64.js');
    const wasm = path.join(directory, 'core_mem64.wasm');
    const run = () => spawnSync(process.execPath, [script, target], { encoding: 'utf8' });
    fs.writeFileSync(target, CURRENT_EMSCRIPTEN_OUTPUT);

    const notJs = spawnSync(process.execPath, [script, path.join(directory, 'core_mem64.mjs')], { encoding: 'utf8' });
    assert.equal(notJs.status, 1);
    assert.match(notJs.stderr, /target must be a \.js file/);

    const noWasm = run();
    assert.equal(noWasm.status, 1);
    assert.match(noWasm.stderr, /core_mem64\.wasm/);
    assert.equal(fs.readFileSync(target, 'utf8'), CURRENT_EMSCRIPTEN_OUTPUT);

    fs.writeFileSync(wasm, WASM32_IMPORTED);
    const wasm32 = run();
    assert.equal(wasm32.status, 1);
    assert.match(wasm32.stderr, /core_mem64\.wasm: wasm memories are \[i32\], expected only i64/);
    assert.equal(wasm32.stdout, '');
    assert.equal(fs.readFileSync(target, 'utf8'), CURRENT_EMSCRIPTEN_OUTPUT);

    fs.writeFileSync(wasm, WASM64_IMPORTED);
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
