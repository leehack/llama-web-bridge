import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

import { LlamaWebGpuBridge, ModelStagingError } from '../../js/src/llama_webgpu_bridge.js';
import { writeResponseToFsFileWithProgress } from '../../js/src/internal/download.ts';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';
import {
  createRealWorkerBridge,
  withStubWorkerEnvironment,
  workerDriver,
} from './bridge_operation_queue_fixtures.mjs';
import { readNativeCoreSource } from './native_core_source.mjs';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

function createStubFs({ failWriteWith = null } = {}) {
  const dirs = new Set(['/']);
  const files = new Map();
  let nextStreamId = 1;
  return {
    dirs,
    files,
    writes: [],
    unmountedPaths: [],
    readdir(dir) {
      if (!dirs.has(dir)) {
        throw new Error(`No such directory: ${dir}`);
      }
      return ['.', '..'];
    },
    mkdir(dir) {
      dirs.add(dir);
    },
    unlink(filePath) {
      if (!files.has(filePath)) {
        throw new Error(`ENOENT: ${filePath}`);
      }
      files.delete(filePath);
    },
    open(filePath, mode) {
      if (mode === 'w') {
        files.set(filePath, 0);
      } else if (!files.has(filePath)) {
        throw new Error(`ENOENT: ${filePath}`);
      }
      return { id: nextStreamId++, filePath };
    },
    write(stream, _bytes, _offset, length, position) {
      if (typeof failWriteWith === 'function') {
        const err = failWriteWith({ filePath: stream.filePath, length, position });
        if (err) {
          throw err;
        }
      }
      this.writes.push({
        filePath: stream.filePath,
        length,
        position: Number(position),
        positionType: typeof position,
      });
      files.set(stream.filePath, Math.max(files.get(stream.filePath) || 0, Number(position) + length));
      return length;
    },
    close() {},
    writeFile(filePath, bytes) {
      files.set(filePath, bytes.length);
    },
    readFile() {
      return new Uint8Array();
    },
  };
}

function createStagingHarness({
  pthreads = false,
  coreVariant = 'wasm32',
  jsBackedModelStaging = true,
  includeJsModelFsExports = true,
  mountResult = 0,
  failWriteWith = null,
  onLoadModel = null,
} = {}) {
  const fs = createStubFs({ failWriteWith });
  const ccalls = [];
  const directWrites = [];
  const openDirectHandles = new Set();
  const closedDirectHandles = [];
  let nextDirectHandle = 1;
  let preallocNextBytes = 0;
  let oomBytes = 0;
  let lastErrorText = '';
  let runtime = null;

  const core = {
    FS: fs,
    UTF8ToString: () => lastErrorText,
    ...(includeJsModelFsExports ? {
      _llamadart_webgpu_mount_js_model_fs: () => 0,
      _llamadart_webgpu_unmount_js_model_fs: () => 0,
      _llamadart_webgpu_js_model_fs_prealloc_next: () => 0,
      _llamadart_webgpu_js_model_fs_open_direct: () => 0,
      _llamadart_webgpu_js_model_fs_close_direct: () => 0,
      _llamadart_webgpu_js_model_fs_consume_oom_bytes: () => 0,
      _llamadart_webgpu_js_model_fs_allocated_bytes: () => 0,
      llamadartJsModelFsWriteDirect(handle, chunk, position) {
        if (!openDirectHandles.has(handle)) {
          return -8;
        }
        if (typeof failWriteWith === 'function') {
          const err = failWriteWith({ filePath: null, length: chunk.byteLength, position });
          if (err) {
            return -(Number(err.errno) || 12);
          }
        }
        directWrites.push({
          handle,
          chunkRef: chunk,
          length: chunk.byteLength,
          position,
        });
        return chunk.byteLength;
      },
    } : {}),
    ccall(name, _returnType, _argTypes, args = []) {
      ccalls.push([name, ...args]);
      if (name === 'llamadart_webgpu_mount_js_model_fs') {
        if (mountResult !== 0) {
          return mountResult;
        }
        fs.dirs.add(args[0]);
        return 0;
      }
      if (name === 'llamadart_webgpu_unmount_js_model_fs') {
        fs.unmountedPaths.push(args[0]);
        for (const p of [...fs.files.keys()]) {
          if (p.startsWith(`${args[0]}/`)) {
            fs.files.delete(p);
          }
        }
        fs.dirs.delete(args[0]);
        openDirectHandles.clear();
        preallocNextBytes = 0;
        return 0;
      }
      if (name === 'llamadart_webgpu_js_model_fs_prealloc_next') {
        preallocNextBytes = Number(args[0]) || 0;
        return 0;
      }
      if (name === 'llamadart_webgpu_js_model_fs_open_direct') {
        const [filePath, preallocBytes = 0, append = 0] = args;
        if (!append) {
          fs.files.set(filePath, 0);
          preallocNextBytes = Number(preallocBytes) || 0;
        } else if (!fs.files.has(filePath)) {
          return -2;
        }
        const handle = nextDirectHandle++;
        openDirectHandles.add(handle);
        return handle;
      }
      if (name === 'llamadart_webgpu_js_model_fs_close_direct') {
        const [handle] = args;
        openDirectHandles.delete(handle);
        closedDirectHandles.push(handle);
        return 0;
      }
      if (name === 'llamadart_webgpu_js_model_fs_consume_oom_bytes') {
        const value = oomBytes;
        oomBytes = 0;
        return value;
      }
      if (name === 'llamadart_webgpu_last_error') {
        return lastErrorText;
      }
      if (name === 'llamadart_webgpu_load_model' && typeof onLoadModel === 'function') {
        return onLoadModel({
          args,
          setLastError: (msg) => { lastErrorText = msg; },
          emitLog: (msg) => {
            runtime._lastCoreErrorText = msg;
            runtime._lastCoreErrorHint = msg;
          },
        });
      }
      if (name === 'llamadart_webgpu_get_context_size') {
        return 2048;
      }
      return 0;
    },
  };

  const bridge = new LlamaWebGpuBridge({
    disableWorker: true,
    allowSingleThreadCore: true,
    preferMemory64: coreVariant === 'wasm64',
    jsBackedModelStaging,
  });
  runtime = bridge._runtime;
  runtime._core = core;
  runtime._coreVariant = coreVariant;
  runtime._probeBackends = async () => false;
  runtime._ensureCore = async () => {
    runtime._core = core;
    runtime._coreVariant = coreVariant;
    return core;
  };
  runtime._coreSupportsPthreads = () => pthreads;
  runtime._syncThreadPoolSizeHintFromCore = () => {};
  runtime._resolveNativeLoadOptions = () => {};
  runtime._emitSuppressedWarmupWarningSummaryIfNeeded = () => {};
  runtime._getCachedModelResponse = async () => new Response(new Uint8Array([1, 2, 3, 4]), {
    status: 200,
    headers: { 'content-length': '4', 'content-encoding': 'identity' },
  });

  return {
    bridge,
    runtime,
    core,
    fs,
    ccalls,
    directWrites,
    openDirectHandles,
    closedDirectHandles,
    get preallocNextBytes() { return preallocNextBytes; },
    setOomBytes: (bytes) => { oomBytes = bytes; },
    setLastError: (msg) => { lastErrorText = msg; },
  };
}

test('jsBackedModelStaging defaults off and works opt-in for threaded and non-threaded cores', async () => {
  const defaultBridge = new LlamaWebGpuBridge({ disableWorker: true });
  assert.equal(defaultBridge._workerConfig().jsBackedModelStaging, undefined);

  const defaultHarness = createStagingHarness({ jsBackedModelStaging: false, pthreads: false });
  await defaultHarness.runtime.loadModelFromUrl('https://models.test/default.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(defaultHarness.fs.dirs.has('/jsmodels'), false);
  assert.equal(defaultHarness.fs.dirs.has('/models'), true);
  assert.ok(!defaultHarness.runtime._runtimeNotes.includes('model_staging_js_backed'));

  for (const pthreads of [false, true]) {
    const harness = createStagingHarness({ jsBackedModelStaging: true, pthreads });
    assert.equal(harness.bridge._workerConfig().jsBackedModelStaging, true);

    await harness.runtime.loadModelFromUrl('https://models.test/demo.gguf', {
      nCtx: 2048,
      nGpuLayers: 0,
      useCache: false,
      forceRemoteFetchBackend: false,
    });

    assert.deepEqual(harness.ccalls[0], ['llamadart_webgpu_mount_js_model_fs', '/jsmodels']);
    assert.deepEqual(harness.ccalls[1], ['llamadart_webgpu_js_model_fs_open_direct', '/jsmodels/demo.gguf', 4, 0]);
    assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_js_model_fs_close_direct'));
    assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_load_model' && c[1] === '/jsmodels/demo.gguf'));
    assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_unmount_js_model_fs' && c[1] === '/jsmodels'));
    assert.ok(harness.runtime._runtimeNotes.includes('model_staging_js_backed'));
    assert.equal(harness.fs.writes.length, 0);
    assert.equal(harness.directWrites.length, 1);
    assert.equal(harness.openDirectHandles.size, 0);
    assert.equal(harness.fs.files.size, 0);
    assert.deepEqual(harness.fs.unmountedPaths, ['/jsmodels']);
  }
});

test('skipMemoryReservation skips the 1-byte seek write while keeping exact transfer-length validation', async () => {
  const fs = createStubFs();
  const gib = 1073741824;
  const response = new Response(new Uint8Array([7, 8, 9]), {
    status: 200,
    headers: { 'content-length': String(gib) },
  });
  const written = await writeResponseToFsFileWithProgress(
    response,
    fs,
    '/jsmodels/big.gguf',
    null,
    { skipMemoryReservation: true },
  );
  assert.equal(written, 3);
  assert.deepEqual(fs.writes, [{
    filePath: '/jsmodels/big.gguf',
    length: 3,
    position: 0,
    positionType: 'number',
  }]);

  // With Content-Encoding: identity and a short body, skipMemoryReservation must still reject short transfers.
  for (const useBigIntPosition of [false, true]) {
    const shortFs = createStubFs();
    const shortBody = new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array(16));
        controller.close();
      },
    });
    const shortResp = new Response(shortBody, {
      status: 200,
      headers: { 'content-length': String(gib), 'content-encoding': 'identity' },
    });
    await assert.rejects(
      writeResponseToFsFileWithProgress(shortResp, shortFs, '/jsmodels/short.gguf', null, {
        skipMemoryReservation: true,
        useBigIntPosition,
      }),
      { name: 'ModelTransferLengthError' },
    );
    assert.equal(shortFs.files.has('/jsmodels/short.gguf'), false);
  }
});

test('jsBackedModelStaging on wasm64 writes directly via handle without FS.write round-trip', async () => {
  const harness = createStagingHarness({
    coreVariant: 'wasm64',
    jsBackedModelStaging: true,
    pthreads: false,
  });

  await harness.runtime.loadModelFromUrl('https://models.test/mem64.gguf', {
    nCtx: 2048,
    nGpuLayers: 999,
    useCache: false,
    forceRemoteFetchBackend: false,
  });

  assert.equal(harness.fs.writes.length, 0);
  assert.deepEqual(harness.directWrites.map(({ handle, length, position }) => ({
    handle,
    length,
    position,
  })), [{
    handle: 1,
    length: 4,
    position: 0,
  }]);
  assert.deepEqual(harness.ccalls[0], ['llamadart_webgpu_mount_js_model_fs', '/jsmodels']);
  assert.deepEqual(harness.ccalls[1], ['llamadart_webgpu_js_model_fs_open_direct', '/jsmodels/mem64.gguf', 4, 0]);
  assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_unmount_js_model_fs' && c[1] === '/jsmodels'));
  assert.ok(harness.runtime._runtimeNotes.includes('model_staging_js_backed'));
  assert.equal(harness.openDirectHandles.size, 0);
  assert.equal(harness.fs.files.size, 0);

  const workerBridge = new LlamaWebGpuBridge({
    disableWorker: false,
    allowSingleThreadCore: true,
    preferMemory64: true,
    jsBackedModelStaging: true,
  });
  assert.equal(workerBridge._workerConfig().preferMemory64, true);
  assert.equal(workerBridge._workerConfig().jsBackedModelStaging, true);
});

test('JS allocation failure during staging raises ModelStagingError, unmounts chunks, and retries cleanly', async () => {
  let failAlloc = true;
  const harness = createStagingHarness({
    jsBackedModelStaging: true,
    failWriteWith: () => {
      if (!failAlloc) return null;
      harness.setOomBytes(2_600_000_000);
      const err = new Error('FS error');
      err.name = 'ErrnoError';
      err.errno = 12;
      return err;
    },
  });

  const err = await harness.runtime.loadModelFromUrl('https://models.test/huge.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject'), (e) => e);

  assert.ok(err instanceof ModelStagingError, String(err));
  assert.equal(err.code, 'MODEL_STAGING_ALLOCATION_FAILED');
  assert.match(err.message, /JavaScript memory while staging/);
  assert.equal(harness.fs.files.size, 0, 'failed staging must unlink partial file');
  assert.deepEqual(harness.fs.unmountedPaths, ['/jsmodels'], 'failed staging must unmount/release chunks');

  // Retry succeeds on the same runtime with zero leaked chunks.
  failAlloc = false;
  await harness.runtime.loadModelFromUrl('https://models.test/small.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(harness.fs.files.size, 0);
  assert.deepEqual(harness.fs.unmountedPaths, ['/jsmodels', '/jsmodels']);
});

test('wasm32 4 GiB tensor-size limit raises ModelStagingError instead of raw abort and cleans up for retry', async () => {
  // Case 1: clean native preflight return code (-4).
  let blockOversized = true;
  const preflightHarness = createStagingHarness({
    jsBackedModelStaging: true,
    onLoadModel: ({ setLastError }) => {
      if (blockOversized) {
        setLastError('GGUF tensor payload exceeds the wasm32 4 GiB address space limit (4297927168 > 4294967295 bytes).');
        return -4;
      }
      return 0;
    },
  });

  const preflightErr = await preflightHarness.runtime.loadModelFromUrl('https://models.test/8b.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject'), (e) => e);

  assert.ok(preflightErr instanceof ModelStagingError, String(preflightErr));
  assert.equal(preflightErr.code, 'MODEL_TENSOR_SIZE_LIMIT_EXCEEDED');
  assert.match(preflightErr.message, /wasm32 4 GiB/);
  assert.equal(preflightHarness.fs.files.size, 0);
  assert.deepEqual(preflightHarness.fs.unmountedPaths, ['/jsmodels']);

  // Subsequent retry works cleanly on the still-live core.
  blockOversized = false;
  await preflightHarness.runtime.loadModelFromUrl('https://models.test/4b.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(preflightHarness.fs.files.size, 0);

  // Case 2: raw core abort with gguf_init_from_reader size overflow log is translated too.
  let shouldAbort = true;
  const abortHarness = createStagingHarness({
    jsBackedModelStaging: true,
    onLoadModel: ({ emitLog }) => {
      if (shouldAbort) {
        emitLog("gguf_init_from_reader: tensor 'blk.30.attn_output.weight' size overflow, cannot accumulate size 4288489984 + 9437184");
        throw new Error('RuntimeError: Aborted(undefined)');
      }
      return 0;
    },
  });

  const abortErr = await abortHarness.runtime.loadModelFromUrl('https://models.test/8b.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject'), (e) => e);

  assert.ok(abortErr instanceof ModelStagingError, String(abortErr));
  assert.equal(abortErr.code, 'MODEL_TENSOR_SIZE_LIMIT_EXCEEDED');
  assert.equal(abortHarness.fs.files.size, 0);
  assert.deepEqual(abortHarness.fs.unmountedPaths, ['/jsmodels']);

  // Core was reset cleanly so retry re-initializes instead of reusing a dead aborted core.
  shouldAbort = false;
  await abortHarness.runtime.loadModelFromUrl('https://models.test/0.6b.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(abortHarness.fs.files.size, 0);
});

test('ModelStagingError preserves class and code across worker host and proxy without main-thread fallback', async () => {
  await withStubWorkerEnvironment(async ({ workers }) => {
    const bridge = createRealWorkerBridge({ jsBackedModelStaging: true });
    assert.equal(bridge._shouldFallbackToMainThread(
      new ModelStagingError('MODEL_STAGING_ALLOCATION_FAILED', 'Array buffer allocation failed'),
    ), false);

    const proxy = bridge._workerProxy;
    const driver = workerDriver(proxy, workers[0]);
    driver.ready();

    for (const code of [
      'MODEL_STAGING_ALLOCATION_FAILED',
      'MODEL_TENSOR_SIZE_LIMIT_EXCEEDED',
      'MODEL_STAGING_MOUNT_FAILED',
      'MODEL_STAGING_UNSUPPORTED',
    ]) {
      const call = proxy.call('loadModelFromUrl', ['https://models.test/model.gguf', {}]);
      await driver.settle();
      const pending = driver.calls('loadModelFromUrl').at(-1);
      proxy._worker.onmessage({
        data: {
          type: 'error',
          id: pending.id,
          message: `staging error (${code})`,
          modelStagingErrorCode: code,
        },
      });
      const err = await call.then(() => assert.fail('must reject'), (e) => e);
      assert.ok(err instanceof ModelStagingError, String(err));
      assert.equal(err.code, code);
    }

    proxy._worker.onmessage({ data: { type: 'result', id: 0 } });
    await bridge.dispose().catch(() => {});
  });

  // Verify worker_host serializes ModelStagingError cleanly over postMessage too.
  const replies = [];
  const oldSelf = globalThis.self;
  globalThis.self = { postMessage: (msg) => replies.push(msg) };
  try {
    installBridgeWorkerHost();
    await self.onmessage({ data: { type: 'init', config: { jsBackedModelStaging: true } } });
    assert.ok(replies.some((m) => m.type === 'ready'));
  } finally {
    if (oldSelf === undefined) delete globalThis.self;
    else globalThis.self = oldSelf;
  }
});

test('native model_load.inc checks wasm32 GGUF tensor size limit across split shards only for JS-backed staging, using buffered I/O', () => {
  const source = readNativeCoreSource();
  const loader = source.slice(source.indexOf('int32_t load_model_internal('));
  const preLoad = loader.slice(0, loader.indexOf('llama_model_load_from_file('));
  assert.match(preLoad, /if \(llamadart_webgpu_is_js_model_path\(model_path\) &&\s*!check_wasm32_gguf_tensor_size_limit\(model_path\)\)\s*\{\s*return -4;\s*\}/);
  assert.match(source, /SIZE_MAX <= 0xffffffffULL/);

  // Compile and execute the real model_load.inc preflight against split GGUF fixtures
  // and default-off /models/ paths on the host C++ compiler.
  const modelLoadText = readFileSync(path.join(rootDir, 'src/core/model_load.inc'), 'utf8');
  const start = modelLoadText.indexOf('class BufferedFileReader {');
  const end = modelLoadText.indexOf('#endif', modelLoadText.indexOf('bool check_wasm32_gguf_tensor_size_limit('));
  assert.ok(start >= 0 && end > start, 'preflight block must exist in model_load.inc');
  const preflightImpl = modelLoadText
    .slice(start, end)
    .replaceAll('std::fread', 'counted_fread')
    .replaceAll('fseeko(', 'counted_fseeko(');

  const fixtureCpp = `#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

static unsigned long long g_freads = 0;
static unsigned long long g_fseeks = 0;
static size_t counted_fread(void * ptr, size_t sz, size_t n, std::FILE * fp) {
  ++g_freads;
  return std::fread(ptr, sz, n, fp);
}
static int counted_fseeko(std::FILE * fp, off_t off, int wh) {
  ++g_fseeks;
  return fseeko(fp, off, wh);
}

#undef SIZE_MAX
#define SIZE_MAX 0xffffffffULL

enum ggml_type { GGML_TYPE_F32 = 0, GGML_TYPE_COUNT = 2 };
enum { GGML_MAX_DIMS = 4, GGUF_DEFAULT_ALIGNMENT = 32 };
enum gguf_type {
  GGUF_TYPE_UINT8 = 0, GGUF_TYPE_INT8 = 1, GGUF_TYPE_UINT16 = 2, GGUF_TYPE_INT16 = 3,
  GGUF_TYPE_UINT32 = 4, GGUF_TYPE_INT32 = 5, GGUF_TYPE_FLOAT32 = 6, GGUF_TYPE_BOOL = 7,
  GGUF_TYPE_STRING = 8, GGUF_TYPE_ARRAY = 9, GGUF_TYPE_UINT64 = 10, GGUF_TYPE_INT64 = 11,
  GGUF_TYPE_FLOAT64 = 12,
};

size_t ggml_type_size(ggml_type) { return 4; }
int64_t ggml_blck_size(ggml_type) { return 1; }
int32_t llama_split_path(char * dst, size_t max, const char * prefix, int32_t no, int32_t count) {
  int w = std::snprintf(dst, max, "%s-%05d-of-%05d.gguf", prefix, no + 1, count);
  return (w > 0 && static_cast<size_t>(w) < max) ? w : 0;
}
int32_t llama_split_prefix(char * dst, size_t max, const char * path, int32_t no, int32_t count) {
  char suf[32];
  std::snprintf(suf, sizeof(suf), "-%05d-of-%05d.gguf", no + 1, count);
  std::string s(path), ss(suf);
  if (s.size() <= ss.size() || s.compare(s.size() - ss.size(), ss.size(), ss) != 0) return 0;
  size_t len = s.size() - ss.size();
  if (len + 1 > max) return 0;
  std::memcpy(dst, path, len);
  dst[len] = '\\0';
  return static_cast<int32_t>(len);
}
static std::string g_error;
void set_error(const std::string & s) { g_error = s; }
bool llamadart_webgpu_is_js_model_path(const char * p) {
  return p != nullptr && std::strncmp(p, "/jsmodels/", 10) == 0;
}

${preflightImpl}

static void write_u32(std::vector<uint8_t> & b, uint32_t v) {
  size_t o = b.size(); b.resize(o + 4); std::memcpy(b.data() + o, &v, 4);
}
static void write_i32(std::vector<uint8_t> & b, int32_t v) {
  size_t o = b.size(); b.resize(o + 4); std::memcpy(b.data() + o, &v, 4);
}
static void write_u16(std::vector<uint8_t> & b, uint16_t v) {
  size_t o = b.size(); b.resize(o + 2); std::memcpy(b.data() + o, &v, 2);
}
static void write_u64(std::vector<uint8_t> & b, uint64_t v) {
  size_t o = b.size(); b.resize(o + 8); std::memcpy(b.data() + o, &v, 8);
}
static void write_str(std::vector<uint8_t> & b, const std::string & s) {
  write_u64(b, s.size()); b.insert(b.end(), s.begin(), s.end());
}

static void write_gguf(
    const std::string & path,
    uint16_t split_count,
    uint16_t split_no,
    uint32_t extra_vocab_strings,
    const std::vector<std::pair<std::string, uint64_t>> & tensors_f32_elems) {
  std::vector<uint8_t> out;
  out.insert(out.end(), {'G', 'G', 'U', 'F'});
  write_u32(out, 3);
  write_u64(out, tensors_f32_elems.size());
  uint64_t n_kv = (split_count > 1 ? 2 : 0) + (extra_vocab_strings > 0 ? 1 : 0);
  write_u64(out, n_kv);
  if (split_count > 1) {
    write_str(out, "split.count"); write_u32(out, GGUF_TYPE_UINT16); write_u16(out, split_count);
    write_str(out, "split.no"); write_u32(out, GGUF_TYPE_UINT16); write_u16(out, split_no);
  }
  if (extra_vocab_strings > 0) {
    write_str(out, "tokenizer.ggml.tokens");
    write_u32(out, GGUF_TYPE_ARRAY);
    write_u32(out, GGUF_TYPE_STRING);
    write_u64(out, extra_vocab_strings);
    for (uint32_t i = 0; i < extra_vocab_strings; ++i) write_str(out, "tok_abcdef");
  }
  for (const auto & kv : tensors_f32_elems) {
    write_str(out, kv.first);
    write_u32(out, 1);
    write_u64(out, kv.second);
    write_i32(out, GGML_TYPE_F32);
    write_u64(out, 0);
  }
  FILE * fp = std::fopen(path.c_str(), "wb");
  assert(fp != nullptr);
  assert(std::fwrite(out.data(), 1, out.size(), fp) == out.size());
  std::fclose(fp);
}

int main(int argc, char ** argv) {
  std::string dir = argv[1];
  // 1) Split GGUF with metadata-only shard 1 + >4 GiB tensor shard 2 must fail preflight.
  std::string s1 = dir + "/m-00001-of-00002.gguf";
  std::string s2 = dir + "/m-00002-of-00002.gguf";
  write_gguf(s1, 2, 0, 10000, {});
  write_gguf(s2, 2, 1, 0, {{"t0", 600000000ULL}, {"t1", 600000000ULL}}); // 4.8 GB > 4 GiB
  g_error.clear(); g_freads = 0; g_fseeks = 0;
  assert(!check_wasm32_gguf_tensor_size_limit(s1.c_str()));
  assert(g_error.find("wasm32 4 GiB") != std::string::npos);
  // Buffered reader must scan 10,000 KV strings with <= 1 seek and minimal fread calls.
  assert(g_fseeks <= 1);
  assert(g_freads < 20);

  // 2) Combined >4 GiB across two individually <4 GiB shards must also fail preflight.
  std::string c1 = dir + "/c-00001-of-00002.gguf";
  std::string c2 = dir + "/c-00002-of-00002.gguf";
  write_gguf(c1, 2, 0, 0, {{"a", 650000000ULL}}); // 2.6 GB
  write_gguf(c2, 2, 1, 0, {{"b", 650000000ULL}}); // 2.6 GB => 5.2 GB combined
  g_error.clear();
  assert(!check_wasm32_gguf_tensor_size_limit(c1.c_str()));

  // 3) Default /models/ path does NOT invoke preflight (0 reads, 0 seeks).
  std::string def = "/models/default.gguf";
  g_freads = 0; g_fseeks = 0;
  bool would_run = llamadart_webgpu_is_js_model_path(def.c_str()) &&
      !check_wasm32_gguf_tensor_size_limit(def.c_str());
  assert(!would_run);
  assert(g_freads == 0 && g_fseeks == 0);
  return 0;
}
`;

  const dir = mkdtempSync(path.join(tmpdir(), 'bridge-preflight-native-'));
  try {
    const cpp = path.join(dir, 'preflight.cpp');
    const exe = path.join(dir, 'preflight');
    writeFileSync(cpp, fixtureCpp);
    const comp = spawnSync(process.env.CXX || 'c++', ['-std=c++17', cpp, '-o', exe], { encoding: 'utf8' });
    assert.equal(comp.status, 0, comp.stderr);
    const run = spawnSync(exe, [dir], { encoding: 'utf8' });
    assert.equal(run.status, 0, `${run.signal}: ${run.stderr}`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('generic Emscripten abort printErr does not overwrite wasm32 tensor-limit hint and resets dead core for retry', async () => {
  let stage = 0;
  const harness = createStagingHarness({
    jsBackedModelStaging: true,
    onLoadModel: ({ emitLog }) => {
      stage += 1;
      if (stage === 1) {
        emitLog("gguf_init_from_reader: tensor 'blk.30.attn_output.weight' size overflow, cannot accumulate size 4288489984 + 9437184");
        // Simulate Emscripten printErr/onAbort emitting generic abort messages afterward.
        harness.runtime._lastCoreErrorText = harness.runtime._lastCoreErrorText;
        harness.runtime._core._printErr?.('core abort');
        throw new WebAssembly.RuntimeError('Aborted(undefined). Build with -sASSERTIONS for more info.');
      }
      return 0;
    },
  });

  // Exercise the exact printErr handler logic on a mock core:
  harness.runtime._lastCoreErrorText = "gguf_init_from_reader: tensor 'blk.30.attn_output.weight' size overflow, cannot accumulate size 4288489984 + 9437184";
  harness.runtime._lastCoreErrorHint = harness.runtime._lastCoreErrorText;
  harness.runtime._lastWasm32TensorLimitError = harness.runtime._lastCoreErrorText;

  // Verify the generic abort guard refuses to clobber diagnostic with 'Aborted(undefined)'.
  const runtimeSrc = readFileSync(path.join(rootDir, 'js/src/runtime.ts'), 'utf8');
  assert.match(runtimeSrc, /!isGenericCoreAbortText\(trimmed\)/);

  const err = await harness.runtime.loadModelFromUrl('https://models.test/split-8b.gguf', {
    nCtx: 2048,
    nGpuLayers: 999,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject'), (e) => e);

  assert.ok(err instanceof ModelStagingError, String(err));
  assert.equal(err.code, 'MODEL_TENSOR_SIZE_LIMIT_EXCEEDED');
  // Core was reset so next retry re-initializes instead of hitting the aborted module.
  assert.equal(harness.runtime._core, null);

  await harness.runtime.loadModelFromUrl('https://models.test/0.6b.gguf', {
    nCtx: 2048,
    nGpuLayers: 999,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(harness.fs.files.size, 0);
});

test('jsBackedModelStaging fails cleanly with MODEL_STAGING_UNSUPPORTED on older core missing exports', async () => {
  const harness = createStagingHarness({
    jsBackedModelStaging: true,
    includeJsModelFsExports: false,
  });

  const err = await harness.runtime.loadModelFromUrl('https://models.test/small.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject'), (e) => e);

  assert.ok(err instanceof ModelStagingError, String(err));
  assert.equal(err.code, 'MODEL_STAGING_UNSUPPORTED');
  assert.match(err.message, /jsBackedModelStaging/);
  assert.match(err.message, /_llamadart_webgpu_mount_js_model_fs/);
  assert.equal(harness.ccalls.length, 0, 'must probe exports before calling ccall');
});

test('jsBackedModelStaging fails cleanly with MODEL_STAGING_MOUNT_FAILED on /jsmodels collision', async () => {
  const harness = createStagingHarness({
    jsBackedModelStaging: true,
    mountResult: -20,
  });

  const err = await harness.runtime.loadModelFromUrl('https://models.test/small.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject'), (e) => e);

  assert.ok(err instanceof ModelStagingError, String(err));
  assert.equal(err.code, 'MODEL_STAGING_MOUNT_FAILED');
  assert.match(err.message, /\/jsmodels/);
  assert.ok(!harness.runtime._runtimeNotes.includes('model_staging_js_backed'));
  assert.equal(harness.fs.unmountedPaths.length, 0, 'must not unmount pre-existing non-JS-backed directory');
});

test('direct JS-backed staging avoids slice clones and FS.write round-trips while keeping progress and length checks', async () => {
  const chunkA = new Uint8Array([10, 20, 30, 40]);
  const chunkB = new Uint8Array([50, 60, 70, 80, 90, 100]);
  let sliceCount = 0;
  for (const chunk of [chunkA, chunkB]) {
    const origSlice = chunk.slice.bind(chunk);
    chunk.slice = (...args) => {
      sliceCount += 1;
      return origSlice(...args);
    };
  }

  let openedWriteStreams = 0;
  let closedWriteStreams = 0;
  const harness = createStagingHarness({
    jsBackedModelStaging: true,
    pthreads: false,
  });
  const origOpen = harness.fs.open.bind(harness.fs);
  harness.fs.open = (...args) => {
    openedWriteStreams += 1;
    return origOpen(...args);
  };
  harness.fs.close = () => {
    closedWriteStreams += 1;
  };
  harness.runtime._getCachedModelResponse = async () => new Response(new ReadableStream({
    start(controller) {
      controller.enqueue(chunkA);
      controller.enqueue(chunkB);
      controller.close();
    },
  }), {
    status: 200,
    headers: { 'content-length': '10', 'content-encoding': 'identity' },
  });

  const progressEvents = [];
  await harness.runtime.loadModelFromUrl('https://models.test/direct.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
    progressCallback: (ev) => progressEvents.push(ev),
  });

  assert.equal(sliceCount, 0, 'direct path must not clone chunks with .slice()');
  assert.equal(openedWriteStreams, 0, 'direct path must not open FS write streams');
  assert.equal(closedWriteStreams, 0, 'direct path must not close FS write streams');
  assert.equal(harness.fs.writes.length, 0, 'direct path must not call FS.write');
  assert.equal(harness.directWrites.length, 2);
  assert.equal(harness.directWrites[0].chunkRef, chunkA);
  assert.equal(harness.directWrites[0].position, 0);
  assert.equal(harness.directWrites[1].chunkRef, chunkB);
  assert.equal(harness.directWrites[1].position, 4);
  assert.equal(progressEvents[progressEvents.length - 1].loaded, 10);
  assert.equal(progressEvents[progressEvents.length - 1].total, 10);

  // Transfer-length check still rejects short >= 1 GiB streams on the direct path.
  const gib = 1073741824;
  harness.runtime._getCachedModelResponse = async () => new Response(new ReadableStream({
    start(controller) {
      controller.enqueue(new Uint8Array(16));
      controller.close();
    },
  }), {
    status: 200,
    headers: { 'content-length': String(gib), 'content-encoding': 'identity' },
  });

  const shortErr = await harness.runtime.loadModelFromUrl('https://models.test/short-direct.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject short transfer'), (e) => e);
  assert.equal(shortErr.name, 'ModelTransferLengthError');
  assert.equal(harness.fs.writes.length, 0);
  assert.equal(harness.openDirectHandles.size, 0);
});

test('direct JS-backed writer completes positive short writes without gaps and counts only written bytes', async () => {
  const fullPayload = new Uint8Array(20);
  for (let i = 0; i < fullPayload.length; i += 1) {
    fullPayload[i] = i + 1;
  }
  const staged = new Uint8Array(20);
  const harness = createStagingHarness({ jsBackedModelStaging: true, pthreads: false });
  const baseWriteDirect = harness.core.llamadartJsModelFsWriteDirect;
  let injectedShort = false;
  harness.core.llamadartJsModelFsWriteDirect = (handle, chunk, position) => {
    if (!injectedShort && position === 4 && chunk.byteLength > 3) {
      injectedShort = true;
      const partial = chunk.subarray(0, chunk.byteLength - 3);
      staged.set(partial, position);
      return baseWriteDirect(handle, partial, position);
    }
    staged.set(chunk, position);
    return baseWriteDirect(handle, chunk, position);
  };
  harness.runtime._getCachedModelResponse = async () => new Response(new ReadableStream({
    start(controller) {
      controller.enqueue(fullPayload.subarray(0, 4));
      controller.enqueue(fullPayload.subarray(4));
      controller.close();
    },
  }), {
    status: 200,
    headers: { 'content-length': '20', 'content-encoding': 'identity' },
  });

  const progressEvents = [];
  await harness.runtime.loadModelFromUrl('https://models.test/short-write-ok.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
    progressCallback: (ev) => progressEvents.push(ev),
  });

  assert.equal(injectedShort, true);
  assert.deepEqual(staged, fullPayload);
  assert.deepEqual(
    harness.directWrites.map(({ position, length }) => ({ position, length })),
    [
      { position: 0, length: 4 },
      { position: 4, length: 13 },
      { position: 17, length: 3 },
    ],
  );
  assert.equal(progressEvents.at(-1).loaded, 20);
  assert.equal(progressEvents.at(-1).total, 20);

  // Zero forward progress on the remainder rejects without advancing loaded/progress past 4 bytes.
  const fs = createStubFs();
  const stalledProgress = [];
  let writeCall = 0;
  const stalledErr = await writeResponseToFsFileWithProgress(
    new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array(10).fill(7));
        controller.close();
      },
    }), { status: 200, headers: { 'content-length': '10' } }),
    fs,
    '/jsmodels/stalled.gguf',
    (ev) => stalledProgress.push(ev),
    {
      skipMemoryReservation: true,
      preservePartialOnError: true,
      directWriter: {
        writeChunk() {
          writeCall += 1;
          return writeCall === 1 ? 4 : 0;
        },
      },
    },
  ).then(() => assert.fail('must reject zero-progress short write'), (e) => e);

  assert.equal(stalledErr.llamadartLoadedBytes, 4);
  assert.equal(stalledProgress.at(-1).loaded, 4);
});

test('resumed 206 validates Content-Range start, end, and total before native load and preserves resume position', async () => {
  const fullPayload = Uint8Array.from({ length: 10 }, (_, i) => 10 + i);
  const makeCutResponse = () => {
    let chunkSent = false;
    return new Response(new ReadableStream({
      pull(controller) {
        if (!chunkSent) {
          chunkSent = true;
          controller.enqueue(fullPayload.subarray(0, 4));
          return;
        }
        controller.error(new Error('network error'));
      },
    }), {
      status: 200,
      headers: { 'content-length': '10', 'content-encoding': 'identity' },
    });
  };

  // 1) Invalid/short 206 responses reject before llamadart_webgpu_load_model and retry cleanly.
  const invalidRanges = [
    { contentRange: 'bytes 4-5/10', body: fullPayload.subarray(4, 6), cl: '2' },
    { contentRange: 'bytes 0-9/10', body: fullPayload, cl: '10' },
    { contentRange: 'bytes 4-9/12', body: fullPayload.subarray(4, 10), cl: '6' },
    { contentRange: 'bytes */10', body: new Uint8Array(0), cl: '0' },
  ];
  for (const invalid of invalidRanges) {
    let nativeLoads = 0;
    const harness = createStagingHarness({
      jsBackedModelStaging: true,
      onLoadModel: () => {
        nativeLoads += 1;
        return 0;
      },
    });
    const requests = [];
    let attempt = 0;
    harness.runtime._getCachedModelResponse = async (_url, opts) => {
      attempt += 1;
      requests.push(opts?.requestHeaders || null);
      if (attempt === 1) {
        return makeCutResponse();
      }
      if (attempt === 2) {
        return new Response(invalid.body, {
          status: 206,
          headers: { 'content-range': invalid.contentRange, 'content-length': invalid.cl },
        });
      }
      return new Response(fullPayload, {
        status: 200,
        headers: { 'content-length': '10', 'content-encoding': 'identity' },
      });
    };

    const err = await harness.runtime.loadModelFromUrl('https://models.test/bad-range.gguf', {
      nCtx: 2048,
      nGpuLayers: 0,
      useCache: false,
      forceRemoteFetchBackend: false,
    }).then(() => assert.fail(`must reject ${invalid.contentRange}`), (e) => e);

    assert.equal(err.name, 'ModelTransferLengthError');
    assert.equal(nativeLoads, 0, 'invalid 206 range must fail before native load');
    assert.deepEqual(requests[1], { Range: 'bytes=4-' });
    assert.equal(harness.fs.files.size, 0);
    assert.deepEqual(harness.fs.unmountedPaths, ['/jsmodels']);

    // Same bridge retries without needing core re-init.
    await harness.runtime.loadModelFromUrl('https://models.test/bad-range.gguf', {
      nCtx: 2048,
      nGpuLayers: 0,
      useCache: false,
      forceRemoteFetchBackend: false,
    });
    assert.equal(nativeLoads, 1);
  }

  // 2) Valid resume writes continuation at startOffset (4), not position 0 (catches resume-pos mutation).
  const staged = new Uint8Array(10);
  const resumeHarness = createStagingHarness({ jsBackedModelStaging: true, pthreads: false });
  const baseWriteDirect = resumeHarness.core.llamadartJsModelFsWriteDirect;
  resumeHarness.core.llamadartJsModelFsWriteDirect = (handle, chunk, position) => {
    staged.set(chunk, position);
    return baseWriteDirect(handle, chunk, position);
  };
  let callIndex = 0;
  resumeHarness.runtime._getCachedModelResponse = async (_url, opts) => {
    callIndex += 1;
    if (callIndex === 1) {
      return makeCutResponse();
    }
    assert.deepEqual(opts?.requestHeaders, { Range: 'bytes=4-' });
    return new Response(fullPayload.subarray(4), {
      status: 206,
      headers: {
        'content-range': 'bytes 4-9/10',
        'content-length': '6',
        'content-encoding': 'identity',
      },
    });
  };

  await resumeHarness.runtime.loadModelFromUrl('https://models.test/resumed.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.deepEqual(
    resumeHarness.directWrites.map(({ position, length }) => ({ position, length })),
    [
      { position: 0, length: 4 },
      { position: 4, length: 6 },
    ],
  );
  assert.deepEqual(staged, fullPayload);
});

test('non-tensor-limit native abort resets dead core so same-bridge retry succeeds', async () => {
  let loadAttempt = 0;
  let coreInits = 0;
  const harness = createStagingHarness({
    jsBackedModelStaging: true,
    onLoadModel: () => {
      loadAttempt += 1;
      if (loadAttempt === 1) {
        throw new WebAssembly.RuntimeError('Aborted(undefined). Build with -sASSERTIONS for more info.');
      }
      return 0;
    },
  });
  harness.runtime._ensureCore = async () => {
    coreInits += 1;
    harness.runtime._core = harness.core;
    harness.runtime._coreVariant = 'wasm32';
    return harness.core;
  };

  const err = await harness.runtime.loadModelFromUrl('https://models.test/truncated.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject native abort'), (e) => e);

  assert.equal(err.name, 'RuntimeError');
  assert.equal(harness.runtime._core, null, 'any native abort must reset dead core');
  assert.equal(harness.runtime._coreVariant, 'uninitialized');
  assert.ok(harness.runtime._runtimeNotes.includes('model_load_ccall_abort'));

  await harness.runtime.loadModelFromUrl('https://models.test/tiny.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(coreInits, 2, 'retry after native abort must initialize a fresh core');
  assert.equal(harness.fs.files.size, 0);
});

test('saved wasm32 tensor-limit hint survives later non-generic printErr lines before abort', async () => {
  let coreInits = 0;
  let printErrHandler = null;
  const harness = createStagingHarness({
    jsBackedModelStaging: true,
    onLoadModel: () => {
      if (coreInits === 1) {
        printErrHandler("gguf_init_from_reader: tensor 'blk.30.attn_output.weight' size overflow, cannot accumulate size 4288489984 + 9437184");
        // Emit a non-generic error line afterwards that overwrites _lastCoreErrorHint / _lastCoreErrorText.
        printErrHandler('llama_model_load: error loading model');
        printErrHandler('Aborted(undefined). Build with -sASSERTIONS for more info.');
        throw new WebAssembly.RuntimeError('Aborted(undefined)');
      }
      return 0;
    },
  });
  harness.core._llamadart_webgpu_device_backend_name = () => '';
  harness.core._llamadart_webgpu_get_thread_pool_size = () => 1;
  harness.core._llamadart_webgpu_supports_pthreads = () => 0;
  harness.runtime._probeBackends = async () => false;
  // Let real _ensureCore wire up printErr using a stub coreModuleFactory.
  delete harness.runtime._ensureCore;
  harness.runtime._core = null;
  harness.runtime._coreVariant = 'uninitialized';
  harness.runtime._config.coreInitTimeoutMs = 50;
  harness.runtime._config.coreModuleFactory = async (moduleArgs) => {
    coreInits += 1;
    printErrHandler = moduleArgs.printErr;
    return harness.core;
  };
  harness.runtime._syncThreadPoolSizeHintFromCore = () => {};
  harness.runtime._resolveNativeLoadOptions = () => {};

  const err = await harness.runtime.loadModelFromUrl('https://models.test/split-8b.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject with tensor limit'), (e) => e);

  assert.ok(err instanceof ModelStagingError, String(err));
  assert.equal(err.code, 'MODEL_TENSOR_SIZE_LIMIT_EXCEEDED');
  assert.match(err.message, /size overflow/);
  assert.equal(harness.runtime._core, null);

  await harness.runtime.loadModelFromUrl('https://models.test/tiny.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(coreInits, 2);
});

test('loadMultimodalProjector uses JS-backed direct staging when jsBackedModelStaging is enabled', async () => {
  for (const coreVariant of ['wasm32', 'wasm64']) {
    const harness = createStagingHarness({ jsBackedModelStaging: true, coreVariant });
    harness.runtime._modelBytes = 1024;
    harness.runtime._fetchWithTimeout = async () => new Response(new Uint8Array([5, 6, 7, 8, 9]), {
      status: 200,
      headers: { 'content-length': '5', 'content-encoding': 'identity' },
    });

    await harness.runtime.loadMultimodalProjector('https://models.test/mmproj-f16.gguf');

    assert.equal(harness.fs.writes.length, 0, 'projector staging must avoid FS.write when enabled');
    assert.deepEqual(
      harness.directWrites.map(({ length, position }) => ({ length, position })),
      [{ length: 5, position: 0 }],
    );
    assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_mmproj_load' && c[1] === '/jsmodels/mmproj-f16.gguf'));
    assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_unmount_js_model_fs' && c[1] === '/jsmodels'));
    assert.ok(harness.runtime._runtimeNotes.includes('mmproj_staging_js_backed'));
    assert.ok(harness.runtime._runtimeNotes.includes('mmproj_file_released'));
    assert.equal(harness.openDirectHandles.size, 0);
    assert.equal(harness.fs.files.size, 0);
  }

  // OOM during projector staging translates cleanly to ModelStagingError and unmounts /jsmodels.
  const oomHarness = createStagingHarness({
    jsBackedModelStaging: true,
    failWriteWith: () => {
      oomHarness.setOomBytes(128 * 1024 * 1024);
      const err = new Error('Array buffer allocation failed');
      err.name = 'RangeError';
      return err;
    },
  });
  oomHarness.runtime._modelBytes = 1024;
  oomHarness.runtime._fetchWithTimeout = async () => new Response(new Uint8Array([1, 2, 3]), {
    status: 200,
    headers: { 'content-length': '3', 'content-encoding': 'identity' },
  });

  const oomErr = await oomHarness.runtime
    .loadMultimodalProjector('https://models.test/huge-mmproj.gguf')
    .then(() => assert.fail('must reject OOM'), (e) => e);
  assert.ok(oomErr instanceof ModelStagingError, String(oomErr));
  assert.equal(oomErr.code, 'MODEL_STAGING_ALLOCATION_FAILED');
  assert.deepEqual(oomHarness.fs.unmountedPaths, ['/jsmodels']);

  // Native abort in llamadart_webgpu_mmproj_load resets dead core.
  const abortHarness = createStagingHarness({ jsBackedModelStaging: true });
  abortHarness.runtime._modelBytes = 1024;
  abortHarness.runtime._fetchWithTimeout = async () => new Response(new Uint8Array([1, 2, 3]), {
    status: 200,
    headers: { 'content-length': '3', 'content-encoding': 'identity' },
  });
  const origCcall = abortHarness.core.ccall.bind(abortHarness.core);
  abortHarness.core.ccall = (name, ...rest) => {
    if (name === 'llamadart_webgpu_mmproj_load') {
      throw new WebAssembly.RuntimeError('Aborted(undefined)');
    }
    return origCcall(name, ...rest);
  };
  await assert.rejects(
    abortHarness.runtime.loadMultimodalProjector('https://models.test/bad-mmproj.gguf'),
    { name: 'RuntimeError' },
  );
  assert.equal(abortHarness.runtime._core, null);
  assert.deepEqual(abortHarness.fs.unmountedPaths, ['/jsmodels']);
});

test('stale decision-head and LoRA handles reject after core abort and reload instead of targeting replacement handles', async () => {
  const decisionCalls = [];
  const loraCalls = [];
  let coreInits = 0;
  let abortNextModelLoad = false;

  const makeFactory = () => async () => {
    const coreId = ++coreInits;
    let nextHeadId = 1;
    let nextLoraId = 1;
    const files = new Map();
    return {
      FS: {
        mkdir() {},
        readdir() { return ['.', '..']; },
        unlink(p) { files.delete(p); },
        open() { return 1; },
        write() { return 0; },
        close() {},
        writeFile(p, bytes) { files.set(p, new Uint8Array(bytes)); },
        readFile(p) { return files.get(p) || new Uint8Array(0); },
      },
      ccall(name, _ret, _types, args = []) {
        if (name === 'llamadart_webgpu_probe') return 0;
        if (name === 'llamadart_webgpu_backends_json') return '["CPU"]';
        if (name === 'llamadart_webgpu_supports_pthreads') return 0;
        if (name === 'llamadart_webgpu_pthread_pool_size') return 1;
        if (name === 'llamadart_webgpu_device_backend_name') return 'WASM';
        if (name === 'llamadart_webgpu_get_context_size') return 2048;
        if (name === 'llamadart_webgpu_model_meta_json') return '{}';
        if (name === 'llamadart_webgpu_decision_api_version') return 1;
        if (name === 'llamadart_webgpu_decision_capabilities_json') {
          return JSON.stringify({ apiVersion: 1, supported: true, reason: null });
        }
        if (name === 'llamadart_webgpu_lora_api_version') return 1;
        if (name === 'llamadart_webgpu_load_model') {
          if (abortNextModelLoad) {
            abortNextModelLoad = false;
            throw new WebAssembly.RuntimeError('Aborted(load failure)');
          }
          return 0;
        }
        if (name === 'llamadart_webgpu_decision_load') {
          return nextHeadId++;
        }
        if (name === 'llamadart_webgpu_decision_head_info_json') {
          return JSON.stringify({
            handle: Number(args[0]),
            label: `head-${coreId}`,
            scoreNames: ['ok'],
            hiddenDim: 4,
            outputDim: 1,
          });
        }
        if (name === 'llamadart_webgpu_decision_run') {
          decisionCalls.push(['run', coreId, Number(args[0])]);
          const outPath = args[2];
          // count=0 empty output binary (0 bytes)
          files.set(outPath, new Uint8Array(0));
          return 0;
        }
        if (name === 'llamadart_webgpu_decision_free') {
          decisionCalls.push(['free', coreId, Number(args[0])]);
          return 0;
        }
        if (name === 'llamadart_webgpu_lora_load') {
          return nextLoraId++;
        }
        if (name === 'llamadart_webgpu_lora_set') {
          loraCalls.push(['set', coreId, Number(args[0])]);
          return 0;
        }
        if (name === 'llamadart_webgpu_lora_remove') {
          loraCalls.push(['remove', coreId, Number(args[0])]);
          return 0;
        }
        if (name === 'llamadart_webgpu_lora_clear') {
          return 0;
        }
        return 0;
      },
      _llamadart_webgpu_decision_capabilities_json() {},
      _llamadart_webgpu_lora_api_version() {},
    };
  };

  const bridge = new LlamaWebGpuBridge({
    disableWorker: true,
    allowSingleThreadCore: true,
    preferMemory64: false,
    coreInitTimeoutMs: 50,
    coreModuleFactory: makeFactory(),
  });
  const stubResponse = () => new Response(new Uint8Array([1, 2, 3, 4]), {
    status: 200,
    headers: { 'content-length': '4', 'content-encoding': 'identity' },
  });
  const origFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    if (String(url).startsWith('https://models.test/')) {
      return stubResponse();
    }
    return origFetch(url);
  };
  try {
    await bridge.loadModelFromUrl('https://models.test/m1.gguf', {
      nCtx: 2048,
      nGpuLayers: 0,
      useCache: false,
      forceRemoteFetchBackend: false,
    });
    const h1 = await bridge.loadDecisionHead(new Uint8Array([1, 2, 3, 4]));
    const l1 = await bridge.loadLoraAdapter(new Uint8Array([1, 2, 3, 4]));

    // Force abort during replacement model load, then reload on new core.
    abortNextModelLoad = true;
    await assert.rejects(
      bridge.loadModelFromUrl('https://models.test/fault.gguf', {
        nCtx: 2048,
        nGpuLayers: 0,
        useCache: false,
        forceRemoteFetchBackend: false,
      }),
      { name: 'RuntimeError' },
    );
    assert.equal(coreInits, 1);

    await bridge.loadModelFromUrl('https://models.test/m2.gguf', {
      nCtx: 2048,
      nGpuLayers: 0,
      useCache: false,
      forceRemoteFetchBackend: false,
    });
    assert.equal(coreInits, 2);

    const h2 = await bridge.loadDecisionHead(new Uint8Array([1, 2, 3, 4]));
    const l2 = await bridge.loadLoraAdapter(new Uint8Array([5, 6, 7, 8]));
    assert.equal(h1.handle, 1);
    assert.equal(h2.handle, 2);

    await assert.rejects(
      bridge.runDecision(h1.handle, []),
      (err) => err?.code === 'head-lost',
    );
    await bridge.freeDecisionHead(h1.handle);
    assert.equal(decisionCalls.length, 0, 'stale handle must never call native decision_run or decision_free on new head');

    await bridge.runDecision(h2.handle, []);
    assert.deepEqual(decisionCalls, [['run', 2, 1]]);

    await assert.rejects(
      bridge.setLoraAdapter(l1.handle, 1),
      /LoRA adapter \d+ is not loaded/,
    );
    await bridge.setLoraAdapter(l2.handle, 0.5);
    assert.deepEqual(loraCalls, [['set', 2, 1]]);
  } finally {
    globalThis.fetch = origFetch;
    await bridge.dispose().catch(() => {});
  }
});

test('core reset after abort terminates the aborted core PThread pool', async () => {
  const terminatedWorkers = [];
  let coreInits = 0;
  const harness = createStagingHarness({
    jsBackedModelStaging: true,
    pthreads: true,
    onLoadModel: () => {
      if (coreInits === 1) {
        throw new WebAssembly.RuntimeError('Aborted(pool leak check)');
      }
      return 0;
    },
  });

  const makeMockPool = (id) => {
    let terminateAllCount = 0;
    const workers = [0, 1, 2, 3].map((idx) => ({
      id: `${id}:${idx}`,
      terminate() {
        terminatedWorkers.push(`${id}:${idx}`);
      },
    }));
    return {
      terminateAllCount: () => terminateAllCount,
      PThread: {
        unusedWorkers: [...workers],
        runningWorkers: [],
        terminateAllThreads() {
          terminateAllCount += 1;
          for (const w of this.unusedWorkers) {
            w.terminate();
          }
          this.unusedWorkers = [];
        },
      },
    };
  };

  let activePool = null;
  let firstPool = null;
  delete harness.runtime._ensureCore;
  harness.runtime._core = null;
  harness.runtime._coreVariant = 'uninitialized';
  harness.runtime._coreGeneration = 0;
  harness.runtime._config.coreInitTimeoutMs = 50;
  harness.runtime._config.allowSingleThreadCore = true;
  harness.runtime._probeBackends = async () => false;
  harness.runtime._syncThreadPoolSizeHintFromCore = () => {};
  harness.runtime._resolveNativeLoadOptions = () => {};
  harness.runtime._config.coreModuleFactory = async () => {
    coreInits += 1;
    activePool = makeMockPool(coreInits);
    if (coreInits === 1) {
      firstPool = activePool;
    }
    harness.core.PThread = activePool.PThread;
    return {
      ...harness.core,
      PThread: activePool.PThread,
    };
  };

  await assert.rejects(
    harness.runtime.loadModelFromUrl('https://models.test/abort.gguf', {
      nCtx: 2048,
      nGpuLayers: 0,
      useCache: false,
      forceRemoteFetchBackend: false,
    }),
    { name: 'RuntimeError' },
  );

  assert.equal(firstPool.terminateAllCount(), 1);
  assert.deepEqual(terminatedWorkers, ['1:0', '1:1', '1:2', '1:3']);

  await harness.runtime.loadModelFromUrl('https://models.test/ok.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(coreInits, 2);
  assert.equal(activePool.terminateAllCount(), 0);
});

test('cross-realm WebAssembly.RuntimeError resets dead core on model and projector native loads', async () => {
  for (const kind of ['model', 'mmproj']) {
    let attempts = 0;
    const harness = createStagingHarness({
      jsBackedModelStaging: true,
      onLoadModel: () => {
        attempts += 1;
        if (kind === 'model' && attempts === 1) {
          throw vm.runInNewContext('new WebAssembly.RuntimeError("unreachable")', {});
        }
        return 0;
      },
    });
    const origCcall = harness.core.ccall.bind(harness.core);
    harness.core.ccall = (name, ...rest) => {
      if (kind === 'mmproj' && name === 'llamadart_webgpu_mmproj_load') {
        attempts += 1;
        throw vm.runInNewContext('new WebAssembly.RuntimeError("unreachable")', {});
      }
      return origCcall(name, ...rest);
    };

    let coreInits = 0;
    harness.runtime._ensureCore = async () => {
      coreInits += 1;
      harness.runtime._core = harness.core;
      harness.runtime._coreGeneration += 1;
      harness.runtime._coreVariant = 'wasm32';
      return harness.core;
    };

    if (kind === 'model') {
      await assert.rejects(
        harness.runtime.loadModelFromUrl('https://models.test/iframe-trap.gguf', {
          nCtx: 2048,
          nGpuLayers: 0,
          useCache: false,
          forceRemoteFetchBackend: false,
        }),
        /unreachable/,
      );
      assert.equal(harness.runtime._core, null);
      await harness.runtime.loadModelFromUrl('https://models.test/ok.gguf', {
        nCtx: 2048,
        nGpuLayers: 0,
        useCache: false,
        forceRemoteFetchBackend: false,
      });
      assert.equal(coreInits, 2);
    } else {
      harness.runtime._core = harness.core;
      harness.runtime._coreGeneration = 1;
      harness.runtime._modelBytes = 1024;
      harness.runtime._fetchWithTimeout = async () => new Response(new Uint8Array([1, 2, 3]), {
        status: 200,
        headers: { 'content-length': '3', 'content-encoding': 'identity' },
      });
      await assert.rejects(
        harness.runtime.loadMultimodalProjector('https://models.test/trap-mmproj.gguf'),
        /unreachable/,
      );
      assert.equal(harness.runtime._core, null);
    }
  }
});

test('worker-mode loadMultimodalProjector preserves ModelStagingError without restarting or retrying worker', async () => {
  await withStubWorkerEnvironment(async ({ workers }) => {
    const bridge = createRealWorkerBridge({ jsBackedModelStaging: true });
    const proxy = bridge._workerProxy;
    const driver = workerDriver(proxy, workers[0]);
    driver.ready();

    const loadPromise = bridge.loadModelFromUrl('https://models.test/cpu-model.gguf', {
      nCtx: 2048,
      nGpuLayers: 0,
      useCache: false,
    });
    await driver.settle();
    const loadReq = driver.calls('loadModelFromUrl').at(-1);
    proxy._worker.onmessage({
      data: {
        type: 'result',
        id: loadReq.id,
        value: 1,
        state: {
          metadata: { 'llamadart.webgpu.model_bytes': 1024 },
          contextSize: 2048,
          gpuActive: false,
          backendName: 'WASM (CPU)',
          supportsVision: false,
          supportsAudio: false,
          coreGeneration: 1,
        },
      },
    });
    await loadPromise;

    const projPromise = bridge.loadMultimodalProjector('https://models.test/oom-mmproj.gguf');
    await driver.settle();
    // _ensureWorkerMultimodalCpuMode only reloads when !_multimodalWorkerCpuMode; it already is false after loadModelFromUrl.
    const secondLoad = driver.calls('loadModelFromUrl');
    if (secondLoad.length > 1) {
      proxy._worker.onmessage({
        data: {
          type: 'result',
          id: secondLoad.at(-1).id,
          value: 1,
          state: {
            metadata: { 'llamadart.webgpu.model_bytes': 1024 },
            contextSize: 2048,
            gpuActive: false,
            backendName: 'WASM (CPU)',
            supportsVision: false,
            supportsAudio: false,
            coreGeneration: 1,
          },
        },
      });
      await driver.settle();
    }
    const projReq = driver.calls('loadMultimodalProjector').at(-1);
    proxy._worker.onmessage({
      data: {
        type: 'error',
        id: projReq.id,
        message: 'Failed to allocate JavaScript memory while staging the model.',
        modelStagingErrorCode: 'MODEL_STAGING_ALLOCATION_FAILED',
      },
    });

    const err = await projPromise.then(() => assert.fail('must reject with ModelStagingError'), (e) => e);
    assert.ok(err instanceof ModelStagingError, String(err));
    assert.equal(err.code, 'MODEL_STAGING_ALLOCATION_FAILED');
    assert.equal(workers.length, 1, 'must not spawn a replacement worker after ModelStagingError');
    assert.equal(driver.calls('loadMultimodalProjector').length, 1, 'must not retry projector load after ModelStagingError');
    assert.equal(bridge._workerProxy, proxy, 'must not fall back to main thread after ModelStagingError');

    proxy._worker.onmessage({ data: { type: 'result', id: 0 } });
    await bridge.dispose().catch(() => {});
  });
});

test('resume requests send If-Range when available and reject 200 or changed validators before native load', async () => {
  const fullPayload = Uint8Array.from({ length: 10 }, (_, i) => 20 + i);
  const makeCutResponse = (headers) => {
    let chunkSent = false;
    return new Response(new ReadableStream({
      pull(controller) {
        if (!chunkSent) {
          chunkSent = true;
          controller.enqueue(fullPayload.subarray(0, 4));
          return;
        }
        controller.error(new Error('network error'));
      },
    }), { status: 200, headers });
  };

  // 1) Strong ETag triggers If-Range and changed weak ETag / Last-Modified / 200 fail with ModelTransferLengthError.
  const scenarios = [
    {
      name: 'strong-etag-200',
      initialHeaders: { 'content-length': '10', etag: '"v1"', 'last-modified': 'Wed, 01 Jan 2025 00:00:00 GMT' },
      expectedIfRange: '"v1"',
      resumeResponse: () => new Response(fullPayload, { status: 200, headers: { 'content-length': '10', etag: '"v2"' } }),
      expectedPattern: /Range resume not honored for model shard: 200/,
    },
    {
      name: 'weak-etag-falls-back-to-last-modified-and-catches-weak-change',
      initialHeaders: { 'content-length': '10', etag: 'W/"weak-1"', 'last-modified': 'Wed, 01 Jan 2025 00:00:00 GMT' },
      expectedIfRange: 'Wed, 01 Jan 2025 00:00:00 GMT',
      resumeResponse: () => new Response(fullPayload.subarray(4), {
        status: 206,
        headers: {
          'content-range': 'bytes 4-9/10',
          'content-length': '6',
          etag: 'W/"weak-2"',
          'last-modified': 'Wed, 01 Jan 2025 00:00:00 GMT',
        },
      }),
      expectedPattern: /ETag validator changed/,
    },
    {
      name: 'weak-etag-only-catches-changed-weak-etag-without-if-range',
      initialHeaders: { 'content-length': '10', etag: 'W/"weak-1"' },
      expectedIfRange: undefined,
      resumeResponse: () => new Response(fullPayload.subarray(4), {
        status: 206,
        headers: {
          'content-range': 'bytes 4-9/10',
          'content-length': '6',
          etag: 'W/"weak-2"',
        },
      }),
      expectedPattern: /ETag validator changed/,
    },
  ];

  for (const s of scenarios) {
    let nativeLoads = 0;
    const seenHeaders = [];
    const harness = createStagingHarness({
      jsBackedModelStaging: true,
      onLoadModel: () => {
        nativeLoads += 1;
        return 0;
      },
    });
    let call = 0;
    harness.runtime._getCachedModelResponse = async (_url, opts) => {
      call += 1;
      seenHeaders.push(opts?.requestHeaders || null);
      return call === 1 ? makeCutResponse(s.initialHeaders) : s.resumeResponse();
    };
    const err = await harness.runtime.loadModelFromUrl(`https://models.test/${s.name}.gguf`, {
      nCtx: 2048,
      nGpuLayers: 0,
      useCache: false,
      forceRemoteFetchBackend: false,
    }).then(() => assert.fail(`must reject (${s.name})`), (e) => e);
    assert.equal(err.name, 'ModelTransferLengthError', s.name);
    assert.match(err.message, s.expectedPattern, s.name);
    assert.equal(nativeLoads, 0, `${s.name} must fail before native load`);
    assert.equal(seenHeaders[1]?.['If-Range'], s.expectedIfRange, s.name);
  }
});

test('cancellation during native abort still resets dead core before rejecting AbortError', async () => {
  const cancel = new AbortController();
  let coreInits = 0;
  let loadAttempts = 0;
  const harness = createStagingHarness({
    jsBackedModelStaging: true,
    onLoadModel: () => {
      loadAttempts += 1;
      if (loadAttempts === 1) {
        cancel.abort();
        throw new WebAssembly.RuntimeError('Aborted(coincident abort + cancel)');
      }
      return 0;
    },
  });
  harness.runtime._ensureCore = async () => {
    coreInits += 1;
    harness.runtime._core = harness.core;
    harness.runtime._coreGeneration += 1;
    harness.runtime._coreVariant = 'wasm32';
    return harness.core;
  };

  await assert.rejects(
    harness.runtime.loadModelFromUrl('https://models.test/cancel-abort.gguf', {
      nCtx: 2048,
      nGpuLayers: 0,
      useCache: false,
      forceRemoteFetchBackend: false,
      signal: cancel.signal,
    }),
    { name: 'AbortError' },
  );
  assert.equal(harness.runtime._core, null, 'coincident abort must reset _core despite cancellation');

  await harness.runtime.loadModelFromUrl('https://models.test/ok.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(coreInits, 2, 'subsequent load after coincident abort+cancel must initialize fresh core');
});
