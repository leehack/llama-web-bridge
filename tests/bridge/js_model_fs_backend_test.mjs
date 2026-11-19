import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const jsFsSource = readFileSync(path.join(rootDir, 'src/llama_webgpu_js_model_fs.js'), 'utf8');
const cppFsSource = readFileSync(path.join(rootDir, 'src/llama_webgpu_js_model_fs.cpp'), 'utf8');

function instantiateJsModelFsLibrary({
  mutatedSource = jsFsSource,
  forceI32SizeAbi = false,
  failingChunkAlloc = null,
} = {}) {
  const HEAPU8 = new Uint8Array(4 * 1024 * 1024);
  const view = new DataView(HEAPU8.buffer);
  const cDefs = { EBADF: 8, ENOMEM: 12, EINVAL: 28 };
  const heapRef = HEAPU8;

  const transformed = mutatedSource
    .replace(/\{\{\{\s*cDefs\.([A-Z0-9_]+)\s*\}\}\}/g, (_, key) => String(cDefs[key]))
    .replace(
      /\{\{\{\s*makeSetValue\(\s*'([^']+)'\s*,\s*([^,]+)\s*,\s*'([^']+)'\s*,\s*'i64'\s*\)\s*\}\}\}/g,
      (_, ptrName, offsetExpr, valName) => (
        forceI32SizeAbi
          ? `view.setInt32((${ptrName}) + (${offsetExpr}), Number(${valName}) | 0, true)`
          : `view.setBigInt64((${ptrName}) + (${offsetExpr}), BigInt(Math.trunc(${valName})), true)`
      ),
    )
    .replace(
      /\{\{\{\s*makeSetValue\(\s*'([^']+)'\s*,\s*([^,]+)\s*,\s*'([^']+)'\s*,\s*'i32'\s*\)\s*\}\}\}/g,
      (_, ptrName, offsetExpr, valName) => `view.setInt32((${ptrName}) + (${offsetExpr}), Number(${valName}) | 0, true)`,
    );

  let registered = null;
  const Uint8ArrayProxy = failingChunkAlloc
    ? class extends Uint8Array {
        constructor(arg, ...rest) {
          if (typeof arg === 'number' && failingChunkAlloc(arg)) {
            throw new RangeError('Array buffer allocation failed');
          }
          super(arg, ...rest);
        }
      }
    : Uint8Array;

  const sandbox = {
    HEAPU8: heapRef,
    view,
    Uint8Array: Uint8ArrayProxy,
    Math,
    Map,
    Number,
    BigInt,
    addToLibrary(obj) {
      registered = obj;
      for (const [key, value] of Object.entries(obj)) {
        if (key.startsWith('$') && !key.includes('__')) {
          sandbox[key.slice(1)] = value;
        }
      }
    },
  };
  vm.runInNewContext(transformed, sandbox);
  assert.ok(registered, 'addToLibrary must register the backend symbols');

  const callNative = (symbol, args = []) => {
    const fn = registered[symbol];
    assert.equal(typeof fn, 'function', `missing library function ${symbol}`);
    const sig = registered[`${symbol}__sig`];
    const usesI53 = registered[`${symbol}__i53abi`] === true;
    const paramSig = typeof sig === 'string' ? sig.slice(1) : '';
    const marshalled = args.map((arg, index) => {
      const code = paramSig[index];
      if (code === 'j') {
        if (forceI32SizeAbi || !usesI53) {
          // Simulate C 32-bit signed truncation when __i53abi or int64 'j' is missing.
          return Number(arg) | 0;
        }
        return Number(arg);
      }
      if (code === 'i') {
        return Number(arg) | 0;
      }
      return arg;
    });
    return fn(...marshalled);
  };

  return { registered, callNative, HEAPU8, view, cDefs };
}

test('js model fs backend C++ and JS signatures enforce 64-bit i53 ABI', () => {
  const { registered } = instantiateJsModelFsLibrary();

  assert.equal(registered._llamadart_js_model_fs_set_next_prealloc_bytes__sig, 'vj');
  assert.equal(registered._llamadart_js_model_fs_set_next_prealloc_bytes__i53abi, true);
  assert.equal(registered._llamadart_js_model_fs_write__sig, 'ippij');
  assert.equal(registered._llamadart_js_model_fs_write__i53abi, true);
  assert.equal(registered._llamadart_js_model_fs_read__sig, 'ippij');
  assert.equal(registered._llamadart_js_model_fs_read__i53abi, true);
  assert.equal(registered._llamadart_js_model_fs_set_size__sig, 'ipj');
  assert.equal(registered._llamadart_js_model_fs_set_size__i53abi, true);
  assert.equal(registered._llamadart_js_model_fs_get_size__sig, 'vpp');

  assert.match(cppFsSource, /void _llamadart_js_model_fs_set_next_prealloc_bytes\(int64_t bytes\)/);
  assert.match(cppFsSource, /int32_t _llamadart_js_model_fs_write\([^)]*int64_t offset\)/);
  assert.match(cppFsSource, /int32_t _llamadart_js_model_fs_read\([^)]*int64_t offset\)/);
  assert.match(cppFsSource, /void _llamadart_js_model_fs_get_size\(uintptr_t file_id,\s*int64_t \* out_size\)/);
  assert.match(cppFsSource, /int32_t _llamadart_js_model_fs_set_size\(uintptr_t file_id,\s*int64_t size\)/);
  assert.doesNotMatch(jsFsSource, /buffer\s*>>>\s*0/);
});

function runOver2GiBChunkBoundaryScenario(simOpts = {}) {
  const { callNative, HEAPU8, view } = instantiateJsModelFsLibrary(simOpts);
  const fileId = 1024;
  const outSizePtr = 64;
  const gib2Offset = 2_214_592_500; // 12 bytes before 33 * 64 MiB boundary (2,214,592,512)
  const payload = new Uint8Array(64);
  for (let i = 0; i < payload.length; i += 1) {
    payload[i] = (i * 7 + 13) & 0xff;
  }

  // Test prealloc preservation on initial set_size(0) truncation while still small.
  callNative('_llamadart_js_model_fs_set_next_prealloc_bytes', [4096]);
  callNative('_llamadart_js_model_fs_alloc_file', [fileId]);
  assert.equal(callNative('_llamadart_js_model_fs_allocated_bytes'), 4096);
  assert.equal(callNative('_llamadart_js_model_fs_set_size', [fileId, 0]), 0);
  assert.equal(callNative('_llamadart_js_model_fs_allocated_bytes'), 4096);

  // Write 64 bytes straddling the 64 MiB chunk boundary above 2 GiB (32-bit int overflows).
  const srcPtr = 1024;
  HEAPU8.set(payload, srcPtr);
  const written = callNative('_llamadart_js_model_fs_write', [fileId, srcPtr, payload.length, gib2Offset]);
  assert.equal(written, payload.length);

  callNative('_llamadart_js_model_fs_get_size', [fileId, outSizePtr]);
  const reportedSize = Number(view.getBigInt64(outSizePtr, true));
  assert.equal(reportedSize, gib2Offset + payload.length);
  assert.ok(reportedSize > 0x7fffffff, 'file size must exceed 32-bit signed int limit');

  // Read across chunk boundary and past EOF.
  const dstPtr = 4096;
  HEAPU8.fill(0, dstPtr, dstPtr + 128);
  const overlapRead = callNative('_llamadart_js_model_fs_read', [fileId, dstPtr, 100, gib2Offset + 16]);
  assert.equal(overlapRead, 48, 'read crossing EOF must clamp to remaining bytes');
  assert.deepEqual(HEAPU8.subarray(dstPtr, dstPtr + 48), payload.subarray(16, 64));

  const pastEofRead = callNative('_llamadart_js_model_fs_read', [fileId, dstPtr, 32, reportedSize]);
  assert.equal(pastEofRead, 0, 'read starting at or past EOF must return 0');

  // Write past 4 GiB too to verify 64-bit math holds beyond unsigned 32-bit range.
  const gib4Offset = 4_362_076_144; // 16 bytes before 65 * 64 MiB boundary (4,362,076,160)
  assert.equal(callNative('_llamadart_js_model_fs_write', [fileId, srcPtr, 32, gib4Offset]), 32);
  callNative('_llamadart_js_model_fs_get_size', [fileId, outSizePtr]);
  assert.equal(Number(view.getBigInt64(outSizePtr, true)), gib4Offset + 32);

  HEAPU8.fill(0, dstPtr, dstPtr + 64);
  assert.equal(callNative('_llamadart_js_model_fs_read', [fileId, dstPtr, 32, gib4Offset]), 32);
  assert.deepEqual(HEAPU8.subarray(dstPtr, dstPtr + 32), payload.subarray(0, 32));

  callNative('_llamadart_js_model_fs_free_file', [fileId]);
  assert.equal(callNative('_llamadart_js_model_fs_allocated_bytes'), 0);
}

test('64-bit offsets/sizes > 2 GiB and > 4 GiB work across 64 MiB chunk boundaries and EOF', () => {
  runOver2GiBChunkBoundaryScenario();
});

test('mutation check: > 2 GiB test fails if size type reverts to 32-bit', () => {
  const mutatedSigSource = jsFsSource
    .replace("_llamadart_js_model_fs_write__sig: 'ippij'", "_llamadart_js_model_fs_write__sig: 'ippii'")
    .replace("_llamadart_js_model_fs_read__sig: 'ippij'", "_llamadart_js_model_fs_read__sig: 'ippii'")
    .replace("_llamadart_js_model_fs_set_size__sig: 'ipj'", "_llamadart_js_model_fs_set_size__sig: 'ipi'");

  assert.throws(
    () => runOver2GiBChunkBoundaryScenario({ mutatedSource: mutatedSigSource, forceI32SizeAbi: true }),
    /AssertionError/,
    '> 2 GiB size test must fail when signatures/sizes regress to 32-bit',
  );
  assert.throws(
    () => runOver2GiBChunkBoundaryScenario({ forceI32SizeAbi: true }),
    /AssertionError/,
    '> 2 GiB size test must fail when makeSetValue/offsets use 32-bit ints',
  );
});

test('js allocation failure records OOM bytes, leaks zero chunks, and releases cleanly', () => {
  let failAlloc = true;
  const { callNative, cDefs } = instantiateJsModelFsLibrary({
    failingChunkAlloc: (len) => failAlloc && len > 1024,
  });

  // Preallocation OOM returns -ENOMEM on write and leaks zero chunks.
  callNative('_llamadart_js_model_fs_set_next_prealloc_bytes', [8192]);
  callNative('_llamadart_js_model_fs_alloc_file', [1]);
  assert.equal(callNative('_llamadart_js_model_fs_allocated_bytes'), 0);
  assert.equal(callNative('_llamadart_js_model_fs_write', [1, 0, 16, 0]), -cDefs.ENOMEM);
  assert.equal(callNative('_llamadart_js_model_fs_consume_oom_bytes'), 8192);
  assert.equal(callNative('_llamadart_js_model_fs_consume_oom_bytes'), 0);
  callNative('_llamadart_js_model_fs_release_all');
  assert.equal(callNative('_llamadart_js_model_fs_allocated_bytes'), 0);

  // Mid-stream write OOM releases file chunks too and allows subsequent retry.
  callNative('_llamadart_js_model_fs_alloc_file', [2]);
  assert.equal(callNative('_llamadart_js_model_fs_write', [2, 0, 16, 0]), -cDefs.ENOMEM);
  assert.equal(callNative('_llamadart_js_model_fs_allocated_bytes'), 0);
  assert.ok(callNative('_llamadart_js_model_fs_consume_oom_bytes') > 0);
  callNative('_llamadart_js_model_fs_release_all');

  failAlloc = false;
  callNative('_llamadart_js_model_fs_set_next_prealloc_bytes', [4096]);
  callNative('_llamadart_js_model_fs_alloc_file', [3]);
  assert.equal(callNative('_llamadart_js_model_fs_allocated_bytes'), 4096);
  assert.equal(callNative('_llamadart_js_model_fs_write', [3, 0, 16, 0]), 16);
  callNative('_llamadart_js_model_fs_release_all');
  assert.equal(callNative('_llamadart_js_model_fs_allocated_bytes'), 0);
});
