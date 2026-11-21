import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const src = fileURLToPath(new URL('../../src/', import.meta.url));
const stub = `
#pragma once
#include <memory>
#include <string>
#include <sys/types.h>
#include <vector>

namespace wasmfs {
class Backend;
using backend_t = Backend *;
using oflags_t = int;
struct Directory { virtual ~Directory() = default; };
struct Symlink { virtual ~Symlink() = default; };
struct MemoryDirectory : Directory { MemoryDirectory(mode_t, backend_t) {} };
struct MemorySymlink : Symlink { MemorySymlink(std::string, backend_t) {} };
struct DataFile {
  DataFile(mode_t, backend_t) {}
  virtual ~DataFile() = default;
  virtual int open(oflags_t) = 0;
  virtual int close() = 0;
  virtual int flush() = 0;
  virtual ssize_t write(const uint8_t *, size_t, off_t) = 0;
  virtual ssize_t read(uint8_t *, size_t, off_t) = 0;
  virtual off_t getSize() = 0;
  virtual int setSize(off_t) = 0;
};
struct Backend {
  virtual ~Backend() = default;
  virtual std::shared_ptr<DataFile> createFile(mode_t) = 0;
  virtual std::shared_ptr<Directory> createDirectory(mode_t) = 0;
  virtual std::shared_ptr<Symlink> createSymlink(std::string) = 0;
};
struct WasmFS {
  std::vector<std::unique_ptr<Backend>> backends;
  backend_t addBackend(std::unique_ptr<Backend> backend) {
    backends.push_back(std::move(backend));
    return backends.back().get();
  }
};
inline WasmFS wasmFS;
}

using backend_t = struct Backend *;
int wasmfs_create_file(const char *, mode_t, backend_t);
inline bool emscripten_is_main_browser_thread() { return false; }

struct em_proxying_ctx {};
namespace emscripten {
struct ProxyWorker {
  inline static unsigned created = 0;
  ProxyWorker() { ++created; }
  template <typename Callback>
  void operator()(Callback callback) {
    em_proxying_ctx ctx;
    struct { em_proxying_ctx * ctx; } proxy_ctx{&ctx};
    callback(proxy_ctx);
  }
};
}
`;

const harness = `
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <sys/mman.h>
#include "llama_webgpu_fetch_fs.cpp"

struct FetchFile { std::string url; uint32_t chunk_size; };
std::map<uint32_t, FetchFile> files;
std::shared_ptr<wasmfs::DataFile> mounted;
std::vector<uint32_t> freed;
std::function<int32_t(uint8_t *, uint32_t, int64_t)> proxy_read;

int wasmfs_create_file(const char * path, mode_t mode, backend_t backend) {
  assert(mode == 0444);
  if (std::string(path) == "/occupied") return -EEXIST;
  mounted = reinterpret_cast<wasmfs::backend_t>(backend)->createFile(mode);
  return 3;
}

extern "C" void _llamadart_fetch_fs_alloc_file(
    em_proxying_ctx *, uint32_t file_id, const char * url, uint32_t chunk_size) {
  assert(files.emplace(file_id, FetchFile{url, chunk_size}).second);
}
extern "C" void _llamadart_fetch_fs_free_file(em_proxying_ctx *, uint32_t file_id) {
  assert(files.erase(file_id) == 1);
  freed.push_back(file_id);
}
extern "C" void _llamadart_fetch_fs_read(
    em_proxying_ctx *, uint32_t file_id, uint8_t * buffer,
    uint32_t length, int64_t offset, int32_t * result) {
  assert(files.count(file_id) == 1);
  *result = proxy_read(buffer, length, offset);
}
extern "C" void _llamadart_fetch_fs_get_size(
    em_proxying_ctx *, uint32_t file_id, int64_t * out_size) {
  assert(files.count(file_id) == 1);
  *out_size = 0x200000003;
}

void test_mounts() {
  using llamadart_webgpu::create_shared_fetch_file;
  const std::string base = "https://example.test/mm/tiny.gguf";
  const std::string signed_url = base + "?sig=a/b&token=" + std::string(280, 'x');
  assert(wasmfs::wasmFS.backends.empty());
  assert(emscripten::ProxyWorker::created == 0);
  for (uint32_t index = 0; index < 7; ++index) {
    if (index == 2) {
      assert(create_shared_fetch_file("/occupied", "unused", 77) == -EEXIST);
      assert(files.empty());
      assert(freed.size() == index);
    }
    const std::string url = index % 2 == 0 ? base : signed_url + std::to_string(index);
    const uint32_t chunk_size = index == 0 ? 0 : 65536 * index;
    assert(create_shared_fetch_file("/unused", url.c_str(), chunk_size) == 3);
    assert(wasmfs::wasmFS.backends.size() == 1);
    assert(emscripten::ProxyWorker::created == 1);
    assert(files.size() == 1);
    const uint32_t id = index + 1;
    assert(files.at(id).url == url);
    assert(files.at(id).chunk_size == (chunk_size == 0 ? 16 * 1024 * 1024 : chunk_size));
    assert(mounted->open(0) == 0 && mounted->close() == 0 && mounted->flush() == 0);
    uint8_t byte = 0;
    proxy_read = [&](uint8_t * buffer, uint32_t length, int64_t offset) {
      assert(buffer == &byte && length == 1 && offset == 0x100000001);
      *buffer = 42;
      return 1;
    };
    assert(mounted->read(&byte, 1, 0x100000001) == 1 && byte == 42);
    assert(mounted->getSize() == 0x200000003);
    assert(mounted->write(&byte, 1, 0) == -EROFS && mounted->setSize(0) == -EROFS);
    mounted.reset();
    assert(files.empty());
    assert(freed.size() == index + 1 && freed.back() == id);
  }
}

void test_large_reads() {
  using llamadart_webgpu::create_shared_fetch_file;
  assert(create_shared_fetch_file("/large", "https://example.test/large.gguf", 0) == 3);
  constexpr size_t signed_32_max = std::numeric_limits<int32_t>::max();
  constexpr auto base_offset = INT64_C(0x100000001);
  constexpr auto capacity = signed_32_max + 100;
  // Reserve address space for pointer arithmetic without committing model-sized RAM.
  void * mapping = mmap(nullptr, capacity, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
  assert(mapping != MAP_FAILED);
  auto * base = static_cast<uint8_t *>(mapping);

  for (const auto [requested, available] : {
      std::pair{signed_32_max + 17, signed_32_max + 17},
      std::pair{capacity, signed_32_max + 9},
  }) {
    size_t total = 0;
    std::vector<uint32_t> requested_steps;
    proxy_read = [&](uint8_t * buffer, uint32_t length, int64_t offset) {
      assert(buffer == base + total && offset == base_offset + static_cast<int64_t>(total));
      requested_steps.push_back(length);
      const auto got = std::min<size_t>(length, available - total);
      total += got;
      return static_cast<int32_t>(got);
    };
    assert(mounted->read(base, requested, base_offset) == static_cast<ssize_t>(available));
    assert(total == available);
    assert(requested_steps.size() == (requested == available ? 3 : 4));
    for (const uint32_t length : requested_steps) assert(length > 0 && length <= (1u << 30));
  }

  unsigned calls = 0;
  proxy_read = [&](uint8_t * buffer, uint32_t length, int64_t offset) {
    assert(buffer == base + calls * 4 && offset == base_offset + calls * 4);
    assert(length == 12 - calls * 4);
    ++calls;
    return 4;
  };
  assert(mounted->read(base, 12, base_offset) == 12 && calls == 3);
  calls = 0;
  proxy_read = [&](uint8_t *, uint32_t, int64_t) { return calls++ == 0 ? 4 : -EIO; };
  assert(mounted->read(base, 12, base_offset) == -EIO && calls == 2);

  calls = 0;
  assert(mounted->read(base, 0, base_offset) == 0);
  assert(mounted->read(base, 16, -1) == -EINVAL);
  assert(calls == 0);
  proxy_read = [&](uint8_t *, uint32_t, int64_t) { ++calls; return 2; };
  assert(mounted->read(base, 3, std::numeric_limits<int64_t>::max() - 1) == -EOVERFLOW);
  assert(calls == 1);
  assert(mounted->read(base, 1, base_offset) == -EIO);
  mounted.reset();
  assert(munmap(mapping, capacity) == 0);
}

#ifdef TEST_WASM32_WIDTH
void test_wasm32_width() {
  unsigned calls = 0;
  constexpr auto start = INT64_C(0xffffffff);
  constexpr uint32_t step = 1u << 30;
  const auto proxy_step = [&](uint32_t total, uint32_t length, int64_t offset) {
    assert(total == calls * step && offset == start + total);
    assert(length == (calls == 0 ? step : 17));
    ++calls;
    return static_cast<int32_t>(length);
  };
  assert(read_fetch_file<uint32_t>(0x80000000, start, proxy_step) == -EOVERFLOW);
  assert(calls == 0);
  assert(read_fetch_file<uint32_t>(step + 17, start, proxy_step) == step + 17);
  assert(calls == 2);
}
#endif

int main() {
#ifdef TEST_WASM32_WIDTH
  test_wasm32_width();
#else
  test_mounts();
  test_large_reads();
#endif
}
`;

function compileAndRun(t, source, threaded, { wasm32Width = false } = {}) {
  const dir = mkdtempSync(path.join(tmpdir(), 'lane301-fetch-fs-test-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  if (threaded) {
    mkdirSync(path.join(dir, 'emscripten'));
    writeFileSync(path.join(dir, 'stub.h'), stub);
    for (const name of [
      'backend.h', 'file.h', 'memory_backend.h', 'thread_utils.h', 'wasmfs.h',
      'emscripten/threading.h', 'emscripten/wasmfs.h',
    ]) {
      writeFileSync(path.join(dir, name), '#include "stub.h"\n');
    }
  }
  writeFileSync(path.join(dir, 'harness.cpp'), source);
  const build = spawnSync('c++', [
    '-std=c++17', '-pthread', threaded ? '-D__EMSCRIPTEN_PTHREADS__' : '-U__EMSCRIPTEN_PTHREADS__',
    '-I', dir, '-I', src,
    ...(wasm32Width ? ['-DTEST_WASM32_WIDTH'] : []),
    ...(threaded ? [] : [path.join(src, 'llama_webgpu_fetch_fs.cpp')]),
    'harness.cpp', '-o', 'harness',
  ], { cwd: dir, encoding: 'utf8' });
  assert.equal(build.status, 0, build.stderr || String(build.error));
  const run = spawnSync('./harness', { cwd: dir, encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || String(run.error));
}

test('consecutive mounts share one worker and preserve full reads past INT32_MAX', (t) => {
  compileAndRun(t, harness, true);
});

test('wasm32 rejects unrepresentable read counts and preserves 64-bit offsets', (t) => {
  compileAndRun(t, harness, true, { wasm32Width: true });
});

test('the single-thread backend stub needs no WasmFS or pthread headers', (t) => {
  compileAndRun(t, `
#include <cerrno>
#include "llama_webgpu_fetch_fs.h"
int main() {
  return llamadart_webgpu::create_shared_fetch_file(nullptr, nullptr, 0) == -ENOSYS ? 0 : 1;
}
`, false);
});

const errno = { EBADF: 8, EINVAL: 28, ENOENT: 44 };
const signedUrl = `https://example.test/mm/tiny.gguf?sig=a/b&token=${'x'.repeat(280)}`;
const payload = Uint8Array.from({ length: 21 }, (_, index) => index + 1);

function setup(fetch, heapOutput) {
  const requests = [];
  const finished = [];
  const heap = new Uint8Array(128);
  let files;
  let hooks;
  class TrackedMap extends Map {
    constructor() {
      super();
      files ??= this;
    }
  }
  const sandbox = {
    HEAPU8: heapOutput ?? heap,
    view: new DataView(heap.buffer),
    Map: TrackedMap,
    URL,
    self: { location: { origin: 'https://example.test' } },
    UTF8ToString: (url) => url,
    _emscripten_proxy_finish: (ctx) => finished.push(ctx),
    fetch(url, options = {}) {
      const { method = 'GET', headers } = options;
      const request = { url, method, range: headers?.Range };
      requests.push(request);
      return fetch(request);
    },
    addToLibrary(library) {
      hooks = library;
      for (const [key, fn] of Object.entries(library)) {
        if (key.startsWith('$') && !key.includes('__')) sandbox[key.slice(1)] = fn;
      }
    },
  };
  const source = readFileSync(path.join(src, 'llama_webgpu_fetch_fs.js'), 'utf8')
    .replace(/^#(?:if PTHREADS|endif).*$/gm, '')
    .replace(/\{\{\{\s*cDefs\.([A-Z]+)\s*\}\}\}/g, (_, key) => String(errno[key]))
    .replace(/\{\{\{\s*makeSetValue\('([\w]+)', 0, '([\w]+)', '(i32|i64)'\)\s*\}\}\}/g,
      (_, ptr, value, type) => type === 'i32'
        ? `view.setInt32(Number(${ptr}), ${value}, true)`
        : `view.setBigInt64(Number(${ptr}), BigInt(${value}), true)`);
  vm.runInNewContext(source, sandbox);
  async function call(name, ...args) {
    const ctx = Symbol(name);
    await hooks[`_llamadart_fetch_fs_${name}`](ctx, ...args);
    assert.deepEqual(finished.splice(0), [ctx], `${name} must finish exactly once`);
  }
  return {
    requests, heap, files: () => files, hooks,
    alloc: (url = signedUrl, chunkSize = 8) => call('alloc_file', 1, url, chunkSize),
    free: () => call('free_file', 1),
    async size() {
      await call('get_size', 1, 0);
      return Number(sandbox.view.getBigInt64(0, true));
    },
    async read(offset, length) {
      await call('read', 1, 32n, length, offset, 8);
      return sandbox.view.getInt32(8, true);
    },
  };
}

function rangedResponse({ method, range }) {
  if (method === 'HEAD') {
    return new Response(null, { headers: { 'Accept-Ranges': 'bytes', 'Content-Length': payload.length } });
  }
  const [, start, end] = /^bytes=(\d+)-(\d+)$/.exec(range);
  return new Response(payload.slice(Number(start), Number(end) + 1), {
    status: 206, headers: { 'Content-Range': `bytes ${start}-${end}/${payload.length}` },
  });
}

test('fetch proxy reads and caches ranges per file and deletes the entry before reusing its ID', async () => {
  const fs = setup(rangedResponse);
  assert.equal(fs.hooks._llamadart_fetch_fs_read__sig, 'vpipijp');
  assert.equal(fs.hooks._llamadart_fetch_fs_read__i53abi, true);
  await fs.alloc();
  assert.equal(await fs.size(), payload.length);
  assert.equal(await fs.read(0, 0), 0);
  assert.equal(await fs.read(30, 1), 0);
  assert.deepEqual(fs.requests, [{ method: 'HEAD', url: signedUrl, range: 'bytes=0-' }]);
  assert.equal(await fs.read(6, 6), 6);
  assert.deepEqual(fs.heap.slice(32, 38), payload.slice(6, 12));
  assert.equal(fs.requests.at(-1).range, 'bytes=0-15');
  assert.equal(await fs.read(6, 6), 6);
  assert.equal(fs.requests.length, 2);
  assert.equal(await fs.read(17, 8), 4);
  assert.deepEqual(fs.heap.slice(32, 36), payload.slice(17));
  assert.equal(fs.requests.at(-1).range, 'bytes=16-20');
  await fs.free();
  assert.equal(fs.files().has(1), false);
  await fs.alloc('/second.gguf', 5);
  assert.equal(await fs.read(0, 5), 5);
  assert.deepEqual(fs.requests.slice(-2), [
    { method: 'HEAD', url: 'https://example.test/second.gguf', range: 'bytes=0-' },
    { method: 'GET', url: 'https://example.test/second.gguf', range: 'bytes=0-4' },
  ]);
  await fs.free();
});

test('fetch proxy uses a whole-file response when ranges are unavailable or ignored', async () => {
  for (const headersAdvertiseRanges of [false, true]) {
    const fs = setup((request) => request.method === 'HEAD'
      ? headersAdvertiseRanges ? rangedResponse(request) : new Response()
      : new Response(payload));
    await fs.alloc();
    assert.equal(await fs.read(12, 8), 8);
    assert.deepEqual(fs.heap.slice(32, 40), payload.slice(12, 20));
    assert.equal(await fs.size(), payload.length);
    assert.equal(await fs.read(16, 2), 2);
    assert.equal(fs.requests.length, 2);
    assert.equal(fs.requests[1].range, headersAdvertiseRanges ? 'bytes=8-20' : undefined);
    await fs.free();
  }

  const empty = setup(() => new Response());
  await empty.alloc();
  assert.equal(await empty.size(), 0);
  assert.equal(await empty.read(0, 8), 0);
  await empty.free();
});

test('fetch proxy finishes on HTTP, network, URL, truncated-body, and missing-file errors', async () => {
  for (const status of [404, 500]) {
    const fs = setup(() => new Response(null, { status }));
    await fs.alloc();
    assert.equal(await fs.size(), 0);
    assert.equal(await fs.read(0, 1), status === 404 ? -errno.ENOENT : -errno.EBADF);
    await fs.free();
  }
  const rejected = setup(() => Promise.reject(new Error('network failure')));
  await rejected.alloc();
  assert.equal(await rejected.size(), 0);
  assert.equal(await rejected.read(0, 1), -errno.EBADF);
  await rejected.free();
  assert.equal(await rejected.read(0, 1), -errno.EBADF);
  assert.equal(await rejected.read(-1, 1), -errno.EINVAL);

  const invalid = setup(() => assert.fail('invalid URL must not be fetched'));
  await invalid.alloc('//[invalid');
  assert.equal(await invalid.size(), 0);
  assert.equal(await invalid.read(0, 1), -errno.ENOENT);
  assert.deepEqual(invalid.requests, []);
  await invalid.free();

  const truncated = setup((request) => request.method === 'HEAD' ? rangedResponse(request)
    : new Response(payload.slice(0, 7), { status: 206, headers: { 'Content-Range': 'bytes 0-7/21' } }));
  await truncated.alloc();
  assert.equal(await truncated.read(0, 8), -errno.EBADF);
  assert.deepEqual(truncated.heap.slice(32, 40), new Uint8Array(8));
  await truncated.free();
});

test('fetch proxy preserves offsets beyond 4 GiB', async () => {
  const size = 0x1_0000_0020;
  const fs = setup(({ method, range }) => {
    if (method === 'HEAD') {
      return new Response(null, { headers: { 'Accept-Ranges': 'bytes', 'Content-Length': size } });
    }
    assert.equal(range, 'bytes=4294967296-4294967327');
    return new Response(new Uint8Array(32).fill(42), { status: 206 });
  });
  await fs.alloc(signedUrl, 32);
  assert.equal(await fs.size(), size);
  assert.equal(await fs.read(0x1_0000_0005, 4), 4);
  assert.deepEqual(fs.heap.slice(32, 36), Uint8Array.of(42, 42, 42, 42));
  await fs.free();
});

test('fetch proxy skips cached interior chunks', async () => {
  const fs = setup(rangedResponse);
  await fs.alloc(signedUrl, 4);
  assert.equal(await fs.read(4, 8), 8);
  assert.equal(await fs.read(0, 16), 16);
  assert.deepEqual(fs.requests.map(({ range }) => range), ['bytes=0-', 'bytes=4-11', 'bytes=0-3', 'bytes=12-15']);
  assert.deepEqual(fs.heap.slice(32, 48), payload.slice(0, 16));
  await fs.free();
});

test('large reads request ranges in batches of at most 64 MiB', async () => {
  const chunkSize = 32 * 1024 * 1024;
  const size = 3 * chunkSize;
  let written = 0;
  const fs = setup(({ method, range }) => {
    if (method === 'HEAD') {
      return new Response(null, { headers: { 'Accept-Ranges': 'bytes', 'Content-Length': size } });
    }
    const [, start, end] = /^bytes=(\d+)-(\d+)$/.exec(range);
    const length = Number(end) - Number(start) + 1;
    assert.ok(length <= 2 * chunkSize, 'range spans more than 64 MiB');
    return { status: 206, ok: true, arrayBuffer: async () => new ArrayBuffer(length) };
  }, {
    set(bytes, offset) {
      assert.equal(offset, 32 + written);
      written += bytes.length;
    },
  });
  await fs.alloc(signedUrl, chunkSize);
  assert.equal(await fs.read(0, size), size);
  assert.equal(written, size);
  assert.deepEqual(fs.requests.map(({ range }) => range), ['bytes=0-', 'bytes=0-67108863', 'bytes=67108864-100663295']);
  await fs.free();
});
