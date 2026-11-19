import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
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
  wasm64 = false,
  heapWindowBase = 0,
  failingChunkAlloc = null,
} = {}) {
  const heapBacking = new Uint8Array(4 * 1024 * 1024);
  const view = new DataView(heapBacking.buffer);
  const toWindowIndex = (idx) => {
    const num = Number(idx);
    const local = num - heapWindowBase;
    assert.ok(
      Number.isSafeInteger(local) && local >= 0 && local <= heapBacking.length,
      `heap index ${num} outside test window [${heapWindowBase}, ${heapWindowBase + heapBacking.length}]`,
    );
    return local;
  };
  const HEAPU8 = heapWindowBase === 0
    ? heapBacking
    : {
        subarray(start, end) {
          return heapBacking.subarray(toWindowIndex(start), toWindowIndex(end));
        },
        set(src, offset) {
          heapBacking.set(src, toWindowIndex(offset));
        },
        fill(val, start, end) {
          heapBacking.fill(val, toWindowIndex(start), toWindowIndex(end));
        },
      };
  const cDefs = { EBADF: 8, ENOMEM: 12, EINVAL: 28 };
  const heapRef = HEAPU8;

  const transformed = mutatedSource
    .replace(/\{\{\{\s*cDefs\.([A-Z0-9_]+)\s*\}\}\}/g, (_, key) => String(cDefs[key]))
    .replace(
      /\{\{\{\s*makeSetValue\(\s*'([^']+)'\s*,\s*([^,]+)\s*,\s*'([^']+)'\s*,\s*'i64'\s*\)\s*\}\}\}/g,
      (_, ptrName, offsetExpr, valName) => (
        forceI32SizeAbi
          ? `view.setInt32((Number(${ptrName}) - ${heapWindowBase}) + (${offsetExpr}), Number(${valName}) | 0, true)`
          : `view.setBigInt64((Number(${ptrName}) - ${heapWindowBase}) + (${offsetExpr}), BigInt(Math.trunc(${valName})), true)`
      ),
    )
    .replace(
      /\{\{\{\s*makeSetValue\(\s*'([^']+)'\s*,\s*([^,]+)\s*,\s*'([^']+)'\s*,\s*'i32'\s*\)\s*\}\}\}/g,
      (_, ptrName, offsetExpr, valName) => `view.setInt32((Number(${ptrName}) - ${heapWindowBase}) + (${offsetExpr}), Number(${valName}) | 0, true)`,
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

  const bigintToI53Checked = (num) => {
    const n = typeof num === 'bigint' ? num : BigInt(num);
    if (n < -9007199254740992n || n > 9007199254740991n) {
      return Number.NaN;
    }
    return Number(n);
  };

  const callNative = (symbol, args = []) => {
    const fn = registered[symbol];
    assert.equal(typeof fn, 'function', `missing library function ${symbol}`);
    const sig = registered[`${symbol}__sig`];
    const usesI53 = registered[`${symbol}__i53abi`] === true;
    const paramSig = typeof sig === 'string' ? sig.slice(1) : '';
    const marshalled = args.map((arg, index) => {
      const code = paramSig[index];
      if (code === 'p') {
        // Under -m64 (wasm64), Emscripten passes pointers as BigInt and wraps 'p' with bigintToI53Checked.
        return wasm64 ? bigintToI53Checked(typeof arg === 'bigint' ? arg : BigInt(arg)) : Number(arg);
      }
      if (code === 'j') {
        if (forceI32SizeAbi || !usesI53) {
          // Simulate C 32-bit signed truncation when __i53abi or int64 'j' is missing.
          return Number(arg) | 0;
        }
        return wasm64 ? bigintToI53Checked(typeof arg === 'bigint' ? arg : BigInt(arg)) : Number(arg);
      }
      if (code === 'i') {
        return Number(arg) | 0;
      }
      return arg;
    });
    return fn(...marshalled);
  };

  return {
    registered,
    callNative,
    writeDirect: sandbox.llamadartJsModelFsWriteDirect,
    HEAPU8,
    heapBacking,
    view,
    cDefs,
  };
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

test('wasm64 BigInt pointers (> 4 GiB) and i53 offsets/sizes round-trip cleanly', () => {
  const heapWindowBase = 6 * 1024 * 1024 * 1024; // 6 GiB virtual base above 32-bit range
  const { callNative, heapBacking, view, registered } = instantiateJsModelFsLibrary({
    wasm64: true,
    heapWindowBase,
  });

  // Pointer parameters must not be truncated with 32-bit shifts/masks in the JS library.
  assert.doesNotMatch(jsFsSource, /\b(fileId|buffer|outSizePtr)\s*(\|\s*0|>>>\s*0|&\s*0xffffffff)/);

  const fileId = BigInt(heapWindowBase) + 0x1238n;
  const outSizePtr = BigInt(heapWindowBase) + 64n;
  const srcPtr = BigInt(heapWindowBase) + 1024n;
  const dstPtr = BigInt(heapWindowBase) + 4096n;
  const payload = new Uint8Array([11, 22, 33, 44, 55, 66, 77, 88]);
  heapBacking.set(payload, 1024);

  callNative('_llamadart_js_model_fs_set_next_prealloc_bytes', [4096n]);
  callNative('_llamadart_js_model_fs_alloc_file', [fileId]);
  assert.equal(callNative('_llamadart_js_model_fs_allocated_bytes'), 4096);

  const over4GiBOffset = 4_362_076_156n; // straddles 65 * 64 MiB boundary
  assert.equal(callNative('_llamadart_js_model_fs_write', [fileId, srcPtr, payload.length, over4GiBOffset]), payload.length);

  callNative('_llamadart_js_model_fs_get_size', [fileId, outSizePtr]);
  assert.equal(view.getBigInt64(64, true), over4GiBOffset + BigInt(payload.length));

  heapBacking.fill(0, 4096, 4096 + payload.length);
  assert.equal(callNative('_llamadart_js_model_fs_read', [fileId, dstPtr, payload.length, over4GiBOffset]), payload.length);
  assert.deepEqual(heapBacking.subarray(4096, 4096 + payload.length), payload);

  // Direct un-narrowed BigInt buffer pointer still converts safely via Number(buffer).
  assert.equal(
    registered._llamadart_js_model_fs_write(
      Number(fileId),
      srcPtr,
      4,
      0,
    ),
    4,
  );
  assert.equal(
    registered._llamadart_js_model_fs_read(
      Number(fileId),
      dstPtr + 32n,
      4,
      0,
    ),
    4,
  );
  assert.deepEqual(heapBacking.subarray(4128, 4132), payload.subarray(0, 4));

  callNative('_llamadart_js_model_fs_free_file', [fileId]);
  assert.equal(callNative('_llamadart_js_model_fs_allocated_bytes'), 0);
});

test('C++ mount rejects ordinary /jsmodels collision and unmount releases unlinked open files to 0 bytes', () => {
  const { callNative } = instantiateJsModelFsLibrary();
  const dir = mkdtempSync(path.join(tmpdir(), 'bridge-js-model-fs-cpp-'));
  try {
    // Minimal Emscripten/WASMFS headers so src/llama_webgpu_js_model_fs.cpp compiles directly with host c++.
    writeFileSync(path.join(dir, 'backend.h'), `
#pragma once
#include <memory>
#include <string>
#include <sys/stat.h>
namespace wasmfs {
class DataFile;
class Directory;
class Symlink;
class Backend {
 public:
  virtual ~Backend() = default;
  virtual std::shared_ptr<DataFile> createFile(mode_t) = 0;
  virtual std::shared_ptr<Directory> createDirectory(mode_t) = 0;
  virtual std::shared_ptr<Symlink> createSymlink(std::string) = 0;
};
using backend_t = Backend*;
}
`);
    writeFileSync(path.join(dir, 'file.h'), `
#pragma once
#include <cstdint>
#include <sys/stat.h>
namespace wasmfs {
using oflags_t = int;
class DataFile {
  backend_t backend_;
 public:
  DataFile(mode_t, backend_t b) : backend_(b) {}
  virtual ~DataFile() = default;
  virtual int open(oflags_t) = 0;
  virtual int close() = 0;
  virtual int flush() = 0;
  virtual ssize_t write(const uint8_t*, size_t, off_t) = 0;
  virtual ssize_t read(uint8_t*, size_t, off_t) = 0;
  virtual off_t getSize() = 0;
  virtual int setSize(off_t) = 0;
};
class Directory {};
class Symlink {};
}
`);
    writeFileSync(path.join(dir, 'memory_backend.h'), `
#pragma once
#include "file.h"
namespace wasmfs {
struct MemoryDirectory : Directory { MemoryDirectory(mode_t, backend_t) {} };
struct MemorySymlink : Symlink { MemorySymlink(std::string, backend_t) {} };
}
`);
    writeFileSync(path.join(dir, 'wasmfs.h'), `
#pragma once
#include <memory>
#include <vector>
#include "backend.h"
namespace wasmfs {
struct WasmFS {
  std::vector<std::unique_ptr<Backend>> backends;
  backend_t addBackend(std::unique_ptr<Backend> b) {
    backends.push_back(std::move(b));
    return backends.back().get();
  }
};
inline WasmFS wasmFS;
}
`);

    const cppHarness = `
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <sys/stat.h>

#include "backend.h"
#include "file.h"
#include "wasmfs.h"
#include "emscripten/wasmfs.h"

namespace {
struct StubEntry {
  ::backend_t backend = nullptr;
};
std::map<std::string, StubEntry> g_dirs;
struct DummyMemBackend {};
DummyMemBackend g_mem_backend_obj;
::backend_t g_mem_backend = reinterpret_cast<::backend_t>(&g_mem_backend_obj);
int64_t g_prealloc = 0;
std::map<uintptr_t, size_t> g_js_chunks;
size_t g_oom = 0;
}

extern "C" {
void _llamadart_js_model_fs_set_next_prealloc_bytes(int64_t b) { g_prealloc = b; }
void _llamadart_js_model_fs_alloc_file(uintptr_t id) {
  g_js_chunks[id] = g_prealloc > 0 ? static_cast<size_t>(g_prealloc) : 0;
  g_prealloc = 0;
}
void _llamadart_js_model_fs_free_file(uintptr_t id) { g_js_chunks.erase(id); }
int32_t _llamadart_js_model_fs_bind_direct(uintptr_t id) { return g_js_chunks.count(id) ? 1 : -EBADF; }
void _llamadart_js_model_fs_unbind_direct(int32_t) {}
int32_t _llamadart_js_model_fs_write(uintptr_t id, const uint8_t*, uint32_t len, int64_t) {
  if (g_js_chunks[id] < 64u * 1024u * 1024u) g_js_chunks[id] = 64u * 1024u * 1024u;
  return static_cast<int32_t>(len);
}
int32_t _llamadart_js_model_fs_read(uintptr_t, uint8_t*, uint32_t, int64_t) { return 0; }
void _llamadart_js_model_fs_get_size(uintptr_t, int64_t* out) { *out = 0; }
int32_t _llamadart_js_model_fs_set_size(uintptr_t, int64_t) { return 0; }
void _llamadart_js_model_fs_release_all() { g_js_chunks.clear(); }
double _llamadart_js_model_fs_consume_oom_bytes() { double r = g_oom; g_oom = 0; return r; }
double _llamadart_js_model_fs_allocated_bytes() {
  size_t total = 0;
  for (const auto& kv : g_js_chunks) total += kv.second;
  return static_cast<double>(total);
}

::backend_t wasmfs_get_backend_by_path(const char* path) {
  auto it = g_dirs.find(path ? path : "");
  return it == g_dirs.end() ? nullptr : it->second.backend;
}
int wasmfs_create_directory(const char* path, mode_t, ::backend_t backend) {
  std::string p = path ? path : "";
  if (g_dirs.count(p)) return -EEXIST;
  g_dirs[p] = {backend};
  return 0;
}
int wasmfs_unmount(const char* path) {
  std::string p = path ? path : "";
  auto it = g_dirs.find(p);
  if (it == g_dirs.end()) return -ENOENT;
  if (it->second.backend == g_mem_backend) return -EINVAL;
  g_dirs.erase(it);
  return 0;
}
int32_t llamadart_webgpu_mount_js_model_fs(const char* mount_path);
int32_t llamadart_webgpu_unmount_js_model_fs(const char* mount_path);
double llamadart_webgpu_js_model_fs_allocated_bytes();
bool llamadart_webgpu_is_js_model_path(const char* path);
}

int main() {
  // 1) An existing ordinary /jsmodels directory must fail mount with -EEXIST and not emit path match.
  g_dirs["/jsmodels"] = {g_mem_backend};
  assert(llamadart_webgpu_mount_js_model_fs("/jsmodels") == -EEXIST);
  assert(!llamadart_webgpu_is_js_model_path("/jsmodels/a.gguf"));
  g_dirs.erase("/jsmodels");

  // 2) Normal mount creates our backend directory; default /models/ paths do not match it.
  assert(llamadart_webgpu_mount_js_model_fs("/jsmodels") == 0);
  assert(llamadart_webgpu_is_js_model_path("/jsmodels/a.gguf"));
  assert(!llamadart_webgpu_is_js_model_path("/models/a.gguf"));

  // 3) An unlinked file still open (shared_ptr held across unmount) must release all chunks to 0 bytes.
  auto* backend = reinterpret_cast<wasmfs::Backend*>(g_dirs["/jsmodels"].backend);
  {
    std::shared_ptr<wasmfs::DataFile> open_fd = backend->createFile(0666);
    uint8_t byte = 42;
    assert(open_fd->write(&byte, 1, 0) == 1);
    assert(llamadart_webgpu_js_model_fs_allocated_bytes() == 64.0 * 1024.0 * 1024.0);
    // Unmount while open_fd still holds a strong reference (simulating open unlinked fd).
    assert(llamadart_webgpu_unmount_js_model_fs("/jsmodels") == 0);
    assert(llamadart_webgpu_js_model_fs_allocated_bytes() == 0.0);
  }
  return 0;
}
`;
    const emInc = path.join(dir, 'emscripten');
    mkdirSync(emInc, { recursive: true });
    writeFileSync(path.join(emInc, 'emscripten.h'), '#pragma once\n#define EMSCRIPTEN_KEEPALIVE\n');
    writeFileSync(path.join(emInc, 'wasmfs.h'), `
#pragma once
#include <sys/stat.h>
typedef struct Backend* backend_t;
extern "C" {
backend_t wasmfs_get_backend_by_path(const char* path);
int wasmfs_create_directory(const char* path, mode_t mode, backend_t backend);
int wasmfs_unmount(const char* path);
}
`);
    const harnessPath = path.join(dir, 'harness.cpp');
    const cppSourcePath = path.join(rootDir, 'src/llama_webgpu_js_model_fs.cpp');
    writeFileSync(harnessPath, cppHarness);
    const exe = path.join(dir, 'test_js_model_fs');
    const comp = spawnSync(
      process.env.CXX || 'c++',
      ['-std=c++17', '-I', dir, cppSourcePath, harnessPath, '-o', exe],
      { encoding: 'utf8' },
    );
    assert.equal(comp.status, 0, comp.stderr);
    const run = spawnSync(exe, [], { encoding: 'utf8' });
    assert.equal(run.status, 0, `${run.signal}: ${run.stderr}`);

    // Mutation check: removing _llamadart_js_model_fs_release_all() from unmount must fail.
    const mutatedCppPath = path.join(dir, 'mutated.cpp');
    const mutatedExe = path.join(dir, 'test_js_model_fs_mutated');
    writeFileSync(
      mutatedCppPath,
      cppFsSource.replace('  _llamadart_js_model_fs_release_all();\n', '  /* mutated_release_removed */\n'),
    );
    const mutComp = spawnSync(
      process.env.CXX || 'c++',
      ['-std=c++17', '-I', dir, mutatedCppPath, harnessPath, '-o', mutatedExe],
      { encoding: 'utf8' },
    );
    assert.equal(mutComp.status, 0, mutComp.stderr);
    const mutRun = spawnSync(mutatedExe, [], { encoding: 'utf8' });
    assert.notEqual(mutRun.status, 0, 'removing release_all from unmount must fail open-fd check');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
  assert.equal(callNative('_llamadart_js_model_fs_allocated_bytes'), 0);
});

test('direct JS chunk writer writes across chunk boundaries and >2 GiB without touching HEAPU8', () => {
  const { callNative, writeDirect, heapBacking, HEAPU8, view } = instantiateJsModelFsLibrary();
  const fileId = 4096;
  const outSizePtr = 64;
  callNative('_llamadart_js_model_fs_alloc_file', [fileId]);
  const handle = callNative('_llamadart_js_model_fs_bind_direct', [fileId]);
  assert.ok(handle > 0);

  // Poison HEAPU8 access during writeDirect to prove zero wasm linear-memory copy.
  const originalSubarray = heapBacking.subarray.bind(heapBacking);
  const originalSet = heapBacking.set.bind(heapBacking);
  let heapReadsDuringWrite = 0;
  let heapWritesDuringWrite = 0;
  heapBacking.subarray = (...args) => {
    heapReadsDuringWrite += 1;
    return originalSubarray(...args);
  };
  heapBacking.set = (...args) => {
    heapWritesDuringWrite += 1;
    return originalSet(...args);
  };

  const payload = new Uint8Array(96);
  for (let i = 0; i < payload.length; i += 1) {
    payload[i] = (i * 11 + 5) & 0xff;
  }
  const crossBoundaryOffset = 2_214_592_480; // 32 bytes before 33 * 64 MiB boundary
  const written = writeDirect(handle, payload, crossBoundaryOffset);
  assert.equal(written, payload.length);
  assert.equal(heapReadsDuringWrite, 0, 'direct chunk write must not read HEAPU8');
  assert.equal(heapWritesDuringWrite, 0, 'direct chunk write must not write HEAPU8');

  heapBacking.subarray = originalSubarray;
  heapBacking.set = originalSet;

  callNative('_llamadart_js_model_fs_get_size', [fileId, outSizePtr]);
  assert.equal(Number(view.getBigInt64(outSizePtr, true)), crossBoundaryOffset + payload.length);

  // llama.cpp seeks and rereads the same bytes multiple times from chunks.
  const dstPtr = 4096;
  for (let pass = 0; pass < 2; pass += 1) {
    HEAPU8.fill(0, dstPtr, dstPtr + 128);
    assert.equal(
      callNative('_llamadart_js_model_fs_read', [fileId, dstPtr, payload.length, crossBoundaryOffset]),
      payload.length,
    );
    assert.deepEqual(HEAPU8.subarray(dstPtr, dstPtr + payload.length), payload);
  }

  callNative('_llamadart_js_model_fs_unbind_direct', [handle]);
  assert.ok(writeDirect(handle, payload, 0) < 0, 'unbound direct handle must reject subsequent writes');

  callNative('_llamadart_js_model_fs_free_file', [fileId]);
  assert.equal(callNative('_llamadart_js_model_fs_allocated_bytes'), 0);
});
