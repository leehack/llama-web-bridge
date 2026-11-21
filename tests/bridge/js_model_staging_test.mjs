import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

import { LlamaWebGpuBridge, ModelStagingError } from '../../js/src/llama_webgpu_bridge.js';
import { guardCore, retireCore } from '../../js/src/internal/core_guard.ts';
import {
  isRetryableStreamNetworkError,
  writeResponseToFsFileWithProgress,
} from '../../js/src/internal/download.ts';
import { jsBackedStagingMode, supportsJsBackedStaging } from '../../js/src/internal/model_staging.ts';
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
    lstat(filePath) {
      if (!dirs.has(filePath) && !files.has(filePath)) {
        throw Object.assign(new Error(`ENOENT: ${filePath}`), { errno: 44 });
      }
      return {};
    },
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
  apiVersion = 3,
  heapOpenResult = null,
  supportsHeapResident = 1,
  userAgent,
  guard = false,
} = {}) {
  const fs = createStubFs({ failWriteWith });
  const ccalls = [];
  const directWrites = [];
  const openDirectHandles = new Set();
  const closedDirectHandles = [];
  let nextDirectHandle = 1;
  let nextMountId = 1;
  const mountIds = new Map();
  const releasedChunkMountIds = [];
  let preallocNextBytes = 0;
  let oomBytes = 0;
  let lastErrorText = '';
  let runtime = null;

  const core = {
    FS: fs,
    UTF8ToString: () => lastErrorText,
    ...(includeJsModelFsExports ? {
      _llamadart_webgpu_js_model_fs_api_version: () => apiVersion,
      _llamadart_webgpu_js_model_fs_mount_id: () => 0,
      _llamadart_webgpu_mount_js_model_fs: () => 0,
      _llamadart_webgpu_unmount_js_model_fs: () => 0,
      _llamadart_webgpu_js_model_fs_prealloc_next: () => 0,
      _llamadart_webgpu_js_model_fs_open_direct: () => 0,
      _llamadart_webgpu_js_model_fs_close_direct: () => 0,
      _llamadart_webgpu_js_model_fs_consume_oom_bytes: () => 0,
      _llamadart_webgpu_js_model_fs_allocated_bytes: () => 0,
      _llamadart_webgpu_js_model_fs_release_mount_chunks: () => 0,
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
    ...(apiVersion >= 4 ? {
      _llamadart_webgpu_js_model_fs_supports_heap_resident: () => supportsHeapResident,
      _llamadart_webgpu_js_model_fs_open_direct_heap: () => 0,
      _llamadart_webgpu_js_model_fs_heap_resident_bytes: () => 0,
    } : {}),
    ccall(name, _returnType, _argTypes, args = []) {
      ccalls.push([name, ...args]);
      if (name === 'llamadart_webgpu_js_model_fs_api_version') {
        return core._llamadart_webgpu_js_model_fs_api_version?.() ?? 0;
      }
      if (name === 'llamadart_webgpu_js_model_fs_mount_id') {
        return mountIds.get(args[0]) || 0;
      }
      if (name === 'llamadart_webgpu_js_model_fs_release_mount_chunks') {
        releasedChunkMountIds.push(Number(args[0]) || 0);
        return 0;
      }
      if (name === 'llamadart_webgpu_mount_js_model_fs') {
        if (mountResult !== 0) {
          return mountResult;
        }
        fs.dirs.add(args[0]);
        mountIds.set(args[0], nextMountId++);
        return 0;
      }
      if (name === 'llamadart_webgpu_unmount_js_model_fs') {
        fs.unmountedPaths.push(args[0]);
        const releasedId = mountIds.get(args[0]) || 0;
        if (releasedId > 0) {
          releasedChunkMountIds.push(releasedId);
        }
        mountIds.delete(args[0]);
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
      if (name === 'llamadart_webgpu_js_model_fs_supports_heap_resident') {
        return supportsHeapResident;
      }
      if (name === 'llamadart_webgpu_js_model_fs_open_direct_heap') {
        if (heapOpenResult !== null) return heapOpenResult;
        const [filePath] = args;
        fs.files.set(filePath, 0);
        const handle = nextDirectHandle++;
        openDirectHandles.add(handle);
        return handle;
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
    ...(jsBackedModelStaging === null ? {} : { jsBackedModelStaging }),
    userAgent,
  });
  runtime = bridge._runtime;
  const runtimeCore = guard
    ? guardCore(core, (cause, stats) => runtime._handleCoreRetired(cause, stats))
    : core;
  runtime._core = runtimeCore;
  runtime._coreVariant = coreVariant;
  runtime._probeBackends = async () => false;
  runtime._ensureCore = async () => {
    runtime._core = runtimeCore;
    runtime._coreVariant = coreVariant;
    return runtimeCore;
  };
  runtime._coreSupportsPthreads = () => pthreads;
  runtime._syncThreadPoolSizeHintFromCore = () => {};
  if (apiVersion < 4) runtime._resolveNativeLoadOptions = () => {};
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
    mountIds,
    releasedChunkMountIds,
    closedDirectHandles,
    get preallocNextBytes() { return preallocNextBytes; },
    setOomBytes: (bytes) => { oomBytes = bytes; },
    setLastError: (msg) => { lastErrorText = msg; },
  };
}

const SAFARI_26 = 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.5 Mobile/15E148 Safari/604.1';
const CHROMIUM = 'Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36';

test('closing a direct staging writer skips ownership checks on a retired core', () => {
  const { runtime } = createStagingHarness({ guard: true });
  const core = runtime._core;
  const mount = runtime._mountJsBackedStaging(core, 'model_staging_js_backed');
  const mountId = runtime._activeJsStagingMounts.get(mount);
  const writer = runtime._openDirectJsWriter(core, `${mount}/model.gguf`, 4, false,
    () => runtime._assertJsStagingMountOwned(core, mount, mountId));

  retireCore(core, 'writer abort');
  assert.doesNotThrow(() => writer.close());
});

test('releasing a retired staging mount drops its bookkeeping without checking ownership', (t) => {
  const { runtime } = createStagingHarness({ guard: true });
  const core = runtime._core;
  const mount = runtime._mountJsBackedStaging(core, 'model_staging_js_backed');
  const ownership = t.mock.method(runtime, '_isJsStagingMountOwned');

  retireCore(core, 'staging abort');
  runtime._releaseJsModelStaging(core, mount);

  assert.equal(runtime._activeJsStagingMounts.has(mount), false);
  assert.equal(runtime._jsBackedStagingMounted, false);
  assert.equal(ownership.mock.callCount(), 0);
});

test('a trap writing a staged model preserves the original error', async () => {
  const { runtime, core } = createStagingHarness({ guard: true });
  const trap = new WebAssembly.RuntimeError('unreachable');
  core.llamadartJsModelFsWriteDirect = () => { throw trap; };

  await assert.rejects(runtime.loadModelFromUrl('https://models.test/model.gguf'), (error) => error === trap);
  assert.ok(runtime._runtimeNotes.includes('model_fs_write_failed'));
});

test('heap model staging selects the writer using the effective layer count and the core ABI', async (t) => {
  const model = 'https://models.test/model.gguf';
  const cases = [
    { name: 'zero layers', options: { nGpuLayers: 0 }, layers: 0, heap: true },
    { name: 'one layer', options: { nGpuLayers: 1 }, layers: 1, heap: true },
    { name: 'omitted layers', options: {}, layers: -1 },
    ...[-1, -5, 2, 99].map((layers) => ({ name: `${layers} requested layers`, options: { nGpuLayers: layers }, layers })),
    { name: 'explicit mmap', options: { nGpuLayers: 0, useMmap: true }, layers: 0, heap: true },
    { name: 'mmap disabled', options: { nGpuLayers: 0, useMmap: false }, layers: 0 },
    { name: 'mlock', options: { nGpuLayers: 0, useMlock: true }, layers: 0 },
    { name: 'Safari capped', harness: { userAgent: SAFARI_26 }, options: { nGpuLayers: 99 }, layers: 1, heap: true },
    { name: 'Safari omitted', harness: { userAgent: SAFARI_26 }, options: {}, layers: 1, heap: true },
    { name: 'Safari negative', harness: { userAgent: SAFARI_26 }, options: { nGpuLayers: -5 }, layers: 1, heap: true },
    { name: 'Safari uncapped override', harness: { userAgent: SAFARI_26 }, options: { safariMaxGpuLayers: Infinity }, layers: -1 },
    { name: 'Safari 27 omitted', harness: { userAgent: MAC_SAFARI_27 }, options: {}, layers: -1 },
    { name: 'API v3', harness: { apiVersion: 3 }, options: { nGpuLayers: 0 }, layers: 0 },
    { name: 'missing heap export', missingExport: true, options: { nGpuLayers: 0 }, layers: 0 },
    { name: 'fixed heap', harness: { supportsHeapResident: 0 }, options: { nGpuLayers: 0 }, layers: 0 },
    { name: 'unknown decoded size', encoding: 'gzip', options: { nGpuLayers: 0 }, layers: 0 },
    { name: 'split model', split: true, options: { nGpuLayers: 0 }, layers: 0 },
    { name: 'staging off', harness: { jsBackedModelStaging: false }, options: { nGpuLayers: 0 }, layers: 0 },
    { name: 'heap allocation failure', harness: { heapOpenResult: -12 }, options: { nGpuLayers: 0 }, layers: 0 },
  ];

  for (const entry of cases) {
    await t.test(entry.name, async () => {
      const { runtime, core, ccalls } = createStagingHarness({ apiVersion: 4, userAgent: CHROMIUM, ...entry.harness });
      if (entry.harness?.apiVersion === 3) {
        core._llamadart_webgpu_js_model_fs_supports_heap_resident = () => 1;
        core._llamadart_webgpu_js_model_fs_open_direct_heap = () => 0;
        core._llamadart_webgpu_js_model_fs_heap_resident_bytes = () => 0;
      }
      if (entry.missingExport) delete core._llamadart_webgpu_js_model_fs_heap_resident_bytes;
      const chunk = Uint8Array.of(1, 2, 3, 4);
      runtime._getCachedModelResponse = async () => new Response(new ReadableStream({
        start(controller) {
          controller.enqueue(chunk.slice());
          controller.close();
        },
      }), { headers: { 'content-length': '4', 'content-encoding': entry.encoding ?? 'identity' } });
      runtime._gpuActive = true;
      const urls = entry.split ? [
        'https://models.test/model-00001-of-00002.gguf',
        'https://models.test/model-00002-of-00002.gguf',
      ] : model;
      await runtime.loadModelFromUrl(urls, {
        ...entry.options, nCtx: 128, useCache: false, forceRemoteFetchBackend: false, safariGpuProbe: false,
      });

      const heapCalls = ccalls.filter(([name]) => name === 'llamadart_webgpu_js_model_fs_open_direct_heap');
      const chunkCalls = ccalls.filter(([name]) => name === 'llamadart_webgpu_js_model_fs_open_direct');
      const failedAllocation = entry.harness?.heapOpenResult === -12;
      assert.deepEqual(heapCalls.map(([, , bytes]) => bytes), entry.heap || failedAllocation ? [4] : []);
      const expectedChunkWriters = entry.heap || entry.harness?.jsBackedModelStaging === false ? 0 : entry.split ? 2 : 1;
      assert.equal(chunkCalls.length, expectedChunkWriters);
      assert.equal(runtime._runtimeNotes.includes('model_staging_heap_resident'), entry.heap === true);
      assert.equal(runtime._runtimeNotes.includes('model_staging_heap_resident_unavailable:-12'), failedAllocation);
      const unsupported = entry.missingExport || entry.harness?.apiVersion === 3 || entry.harness?.supportsHeapResident === 0;
      assert.equal(runtime._runtimeNotes.includes('model_staging_heap_resident_unavailable:unsupported'), unsupported === true);
      const capabilityCalls = ccalls.filter(([name]) => name === 'llamadart_webgpu_js_model_fs_supports_heap_resident');
      const capabilityChecked = entry.heap || failedAllocation || entry.harness?.supportsHeapResident === 0;
      assert.equal(capabilityCalls.length, capabilityChecked ? 1 : 0);
      const nativeLoad = ccalls.find(([name]) => name === 'llamadart_webgpu_load_model');
      assert.equal(nativeLoad[7], entry.layers);
      assert.equal(nativeLoad[9], entry.options.useMmap ? 1 : 0);
    });
  }
});

test('model staging releases consumed stream chunks on WebKit only', async () => {
  for (const [userAgent, expectedLength] of [[SAFARI_26, 0], [CHROMIUM, 4]]) {
    const { runtime } = createStagingHarness({ apiVersion: 4, userAgent });
    const chunk = Uint8Array.of(1, 2, 3, 4);
    runtime._getCachedModelResponse = async () => new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(chunk);
        controller.close();
      },
    }), { headers: { 'content-length': '4' } });
    await runtime.loadModelFromUrl('https://models.test/model.gguf', {
      nCtx: 128, nGpuLayers: 0, useCache: false, forceRemoteFetchBackend: false,
    });
    assert.equal(chunk.buffer.byteLength, expectedLength);
  }
});

test('heap staging links mmap wrappers and releases mappings after their last native consumer', () => {
  const cmake = readFileSync(path.join(rootDir, 'CMakeLists.txt'), 'utf8');
  for (const name of ['mmap', 'munmap']) {
    assert.match(cmake, new RegExp(`--wrap=${name}\\b`));
  }
  for (const name of [
    'js_model_fs_supports_heap_resident',
    'js_model_fs_open_direct_heap',
    'js_model_fs_heap_resident_bytes',
    'test_fail_next_model_load',
  ]) {
    assert.ok(cmake.includes(`_llamadart_webgpu_${name}`), `missing core export ${name}`);
  }
  const support = readFileSync(path.join(rootDir, 'src/core/support.inc'), 'utf8');
  const freeRuntime = support.match(/void free_runtime\(\) \{([\s\S]*?)\n\}/)?.[1];
  assert.match(freeRuntime, /llama_model_free\(g_state\.model\);\s*g_state\.model = nullptr;\s*\}\s*llamadart_webgpu_release_heap_model_mappings\(\);/);
  const load = readFileSync(path.join(rootDir, 'src/core/model_load.inc'), 'utf8');
  assert.match(load, /g_state\.model = llama_model_load_from_file\([^;]+;\s*if \(g_state\.model == nullptr\) \{[^}]*llamadart_webgpu_release_heap_model_mappings\(\);[^}]*return -2;/);
  assert.match(load, /g_model_params = mparams;[^}]*g_model_params\.load_mode = requested_load_mode;/);
});

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
  assert.equal(defaultHarness.fs.dirs.has('/jsmodels-1'), false);
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

    const mountIdx = harness.ccalls.findIndex((c) => c[0] === 'llamadart_webgpu_mount_js_model_fs');
    assert.ok(mountIdx >= 0);
    assert.deepEqual(harness.ccalls[mountIdx], ['llamadart_webgpu_mount_js_model_fs', '/jsmodels-1']);
    assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_js_model_fs_open_direct' && c[1] === '/jsmodels-1/demo.gguf' && c[2] === 4 && c[3] === 0));
    assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_js_model_fs_close_direct'));
    assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_load_model' && c[1] === '/jsmodels-1/demo.gguf'));
    assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_unmount_js_model_fs' && c[1] === '/jsmodels-1'));
    assert.ok(harness.runtime._runtimeNotes.includes('model_staging_js_backed'));
    assert.equal(harness.fs.writes.length, 0);
    assert.equal(harness.directWrites.length, 1);
    assert.equal(harness.openDirectHandles.size, 0);
    assert.equal(harness.fs.files.size, 0);
    assert.deepEqual(harness.fs.unmountedPaths, ['/jsmodels-1']);
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
  assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_mount_js_model_fs' && c[1] === '/jsmodels-1'));
  assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_js_model_fs_open_direct' && c[1] === '/jsmodels-1/mem64.gguf' && c[2] === 4 && c[3] === 0));
  assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_unmount_js_model_fs' && c[1] === '/jsmodels-1'));
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
  assert.deepEqual(harness.fs.unmountedPaths, ['/jsmodels-1'], 'failed staging must unmount/release chunks');

  // Retry succeeds on the same runtime with zero leaked chunks.
  failAlloc = false;
  await harness.runtime.loadModelFromUrl('https://models.test/small.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(harness.fs.files.size, 0);
  assert.deepEqual(harness.fs.unmountedPaths, ['/jsmodels-1', '/jsmodels-2']);
});

test('wasm32 4 GiB tensor-size limit raises ModelStagingError instead of abort and cleans up for same-core retry', async () => {
  // Clean native preflight return code (-4) without aborting the core.
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
  const initialCore = preflightHarness.core;

  const preflightErr = await preflightHarness.runtime.loadModelFromUrl('https://models.test/8b.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject'), (e) => e);

  assert.ok(preflightErr instanceof ModelStagingError, String(preflightErr));
  assert.equal(preflightErr.code, 'MODEL_TENSOR_SIZE_LIMIT_EXCEEDED');
  assert.match(preflightErr.message, /wasm32 4 GiB/);
  assert.equal(preflightHarness.runtime._core, initialCore, 'preflight failure must leave the core alive');
  assert.equal(preflightHarness.fs.files.size, 0);
  assert.deepEqual(preflightHarness.fs.unmountedPaths, ['/jsmodels-1']);

  // Subsequent retry works cleanly on the still-live core.
  blockOversized = false;
  await preflightHarness.runtime.loadModelFromUrl('https://models.test/0.6b.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(preflightHarness.runtime._core, initialCore);
  assert.equal(preflightHarness.fs.files.size, 0);
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

  // Verify worker_host serializes ModelStagingError cleanly over postMessage and BridgeWorkerProxy restores it.
  const replies = [];
  const oldSelf = globalThis.self;
  globalThis.self = { postMessage: (msg) => replies.push(msg) };
  try {
    installBridgeWorkerHost();
    await self.onmessage({
      data: {
        type: 'init',
        config: {
          jsBackedModelStaging: true,
          coreInitTimeoutMs: 50,
          coreModuleFactory: async () => ({ FS: createStubFs(), ccall: () => 0 }),
        },
      },
    });
    assert.ok(replies.some((m) => m.type === 'ready'));

    await self.onmessage({
      data: {
        type: 'call',
        id: 42,
        method: 'loadModelFromUrl',
        args: ['https://models.test/unsupported.gguf', { nCtx: 512, nGpuLayers: 0, useCache: false, forceRemoteFetchBackend: false }],
      },
    });
    const hostReply = replies.find((m) => m.id === 42);
    assert.ok(hostReply, 'worker host must post error reply');
    assert.equal(hostReply.type, 'error');
    assert.equal(hostReply.modelStagingErrorCode, 'MODEL_STAGING_UNSUPPORTED');

    await withStubWorkerEnvironment(async ({ workers }) => {
      const roundTripBridge = createRealWorkerBridge({ jsBackedModelStaging: true });
      const proxy = roundTripBridge._workerProxy;
      const driver = workerDriver(proxy, workers[0]);
      driver.ready();
      const call = proxy.call('loadModelFromUrl', ['https://models.test/unsupported.gguf', {}]);
      await driver.settle();
      const pending = driver.calls('loadModelFromUrl').at(-1);
      proxy._worker.onmessage({ data: { ...hostReply, id: pending.id } });
      const err = await call.then(() => assert.fail('must reject'), (e) => e);
      assert.ok(err instanceof ModelStagingError, `Expected ModelStagingError, got ${err?.name}: ${err?.message}`);
      assert.equal(err.code, 'MODEL_STAGING_UNSUPPORTED');
      proxy._worker.onmessage({ data: { type: 'result', id: 0 } });
      await roundTripBridge.dispose().catch(() => {});
    });
  } finally {
    if (oldSelf === undefined) delete globalThis.self;
    else globalThis.self = oldSelf;
  }
});

test('native model_load.inc checks wasm32 GGUF tensor size limit across split shards only for JS-backed staging, using buffered I/O', async (t) => {
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
    const std::vector<std::pair<std::string, uint64_t>> & tensors_f32_elems,
    uint64_t second_dim = 1) {
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
    write_u32(out, second_dim == 1 ? 1 : 2);
    write_u64(out, kv.second);
    if (second_dim != 1) write_u64(out, second_dim);
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
  assert(g_error.find("tensor 'b' declares 2600000000 bytes, accumulated 2600000000 bytes") != std::string::npos);

  // 3) Default /models/ path does NOT invoke preflight (0 reads, 0 seeks).
  std::string def = "/models/default.gguf";
  g_freads = 0; g_fseeks = 0;
  bool would_run = llamadart_webgpu_is_js_model_path(def.c_str()) &&
      !check_wasm32_gguf_tensor_size_limit(def.c_str());
  assert(!would_run);
  assert(g_freads == 0 && g_fseeks == 0);

  const std::string oversized = dir + "/oversized.gguf";
  write_gguf(oversized, 1, 0, 0, {{"oversized.weight", 1200000000ULL}});
  g_error.clear();
  assert(!check_wasm32_gguf_tensor_size_limit(oversized.c_str()));
  std::puts(g_error.c_str());

  const std::string control_name = std::string("part\\n\\r\\t\\x1b\\x7f") + '\\0' +
      std::string(54, 'a') + "uncaptured";
  write_gguf(oversized, 1, 0, 0, {{control_name, 1200000000ULL}});
  assert(!check_wasm32_gguf_tensor_size_limit(oversized.c_str()));
  std::puts(g_error.c_str());

  write_gguf(oversized, 1, 0, 0, {{"caf\\xc3\\xa9\\xc2\\x85next", 1200000000ULL}});
  assert(!check_wasm32_gguf_tensor_size_limit(oversized.c_str()));
  std::puts(g_error.c_str());

  write_gguf(oversized, 1, 0, 0, {{"type-width-overflow", std::numeric_limits<int64_t>::max()}});
  assert(!check_wasm32_gguf_tensor_size_limit(oversized.c_str()));
  std::puts(g_error.c_str());

  write_gguf(oversized, 1, 0, 0, {{"dimension-overflow", 1ULL << 60}}, 4);
  assert(!check_wasm32_gguf_tensor_size_limit(oversized.c_str()));
  std::puts(g_error.c_str());

  write_gguf(oversized, 1, 0, 0, {{"max-aligned-f32", 1073741816ULL}});
  assert(check_wasm32_gguf_tensor_size_limit(oversized.c_str()));
  write_gguf(oversized, 1, 0, 0, {{"first-rejected-f32", 1073741817ULL}});
  assert(!check_wasm32_gguf_tensor_size_limit(oversized.c_str()));
  std::puts(g_error.c_str());
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
    assert.match(run.stdout, /tensor 'oversized\.weight' declares 4800000000 bytes, accumulated 0 bytes, limit 4294967295 bytes/);
    assert.match(run.stdout, /tensor 'first-rejected-f32' declares 4294967268 bytes/);
    const messages = run.stdout.split('\n');
    const hasDiagnostic = (detail) => messages.includes(
      `GGUF tensor payload exceeds the wasm32 4 GiB address space limit (${detail}, accumulated 0 bytes, `
        + 'limit 4294967295 bytes); use the memory64 core or a smaller quantization',
    );
    await t.test('escapes control bytes in the first 64 name bytes', () => {
      const escapedName = `part\\x0a\\x0d\\x09\\x1b\\x7f\\x00${'a'.repeat(54)}`;
      assert.ok(hasDiagnostic(`tensor '${escapedName}' declares 4800000000 bytes`), run.stdout);
    });
    await t.test('escapes UTF-8 control characters and preserves printable UTF-8', () => {
      assert.ok(hasDiagnostic("tensor 'café\\u0085next' declares 4800000000 bytes"), run.stdout);
    });
    await t.test('reports overflow when multiplying by the type width', () => {
      assert.ok(hasDiagnostic("tensor 'type-width-overflow' declares at least 2^64 bytes"), run.stdout);
    });
    await t.test('reports overflow when multiplying by another dimension', () => {
      assert.ok(hasDiagnostic("tensor 'dimension-overflow' declares at least 2^64 bytes"), run.stdout);
    });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
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

test('jsBackedModelStaging fails cleanly with MODEL_STAGING_MOUNT_FAILED on staging mount collision', async () => {
  const harness = createStagingHarness({
    jsBackedModelStaging: true,
    mountResult: -17,
  });

  const err = await harness.runtime.loadModelFromUrl('https://models.test/small.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject'), (e) => e);

  assert.ok(err instanceof ModelStagingError, String(err));
  assert.equal(err.code, 'MODEL_STAGING_MOUNT_FAILED');
  assert.match(err.message, /\/jsmodels-1/);
  assert.ok(!harness.runtime._runtimeNotes.includes('model_staging_js_backed'));
  assert.equal(harness.fs.unmountedPaths.length, 0, 'must not unmount pre-existing non-JS-backed directory');
});

test('jsBackedModelStaging bounds the search for a free staging path', async () => {
  const harness = createStagingHarness();
  for (let number = 1; number <= 64; number++) {
    harness.fs.mkdir(`/jsmodels-${number}`);
  }

  await assert.rejects(harness.runtime.loadModelFromUrl('https://models.test/small.gguf', {
    nCtx: 128,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }), {
    name: 'ModelStagingError',
    code: 'MODEL_STAGING_MOUNT_FAILED',
    message: /next 64 candidates/,
  });
  assert.ok(!harness.ccalls.some(([name]) => name === 'llamadart_webgpu_mount_js_model_fs'));
  assert.deepEqual(harness.fs.unmountedPaths, []);
});

test('jsBackedModelStaging bounds retries on native mount collisions', async () => {
  const harness = createStagingHarness({ mountResult: -20 });
  await assert.rejects(harness.runtime.loadModelFromUrl('https://models.test/small.gguf', {
    nCtx: 128,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }), {
    name: 'ModelStagingError',
    code: 'MODEL_STAGING_MOUNT_FAILED',
    message: /next 64 candidates/,
  });
  const attempts = harness.ccalls.filter(([name]) => name === 'llamadart_webgpu_mount_js_model_fs');
  assert.deepEqual(attempts.map(([, mountPath]) => mountPath),
    Array.from({ length: 64 }, (_, index) => `/jsmodels-${index + 1}`));
  assert.deepEqual(harness.fs.unmountedPaths, []);
  assert.deepEqual(harness.releasedChunkMountIds, []);
});

test('JS-backed staging does not mount when lstat fails for a reason other than ENOENT', async () => {
  const { runtime, fs, ccalls } = createStagingHarness();
  fs.lstat = () => {
    throw Object.assign(new Error('permission denied'), { errno: 2 });
  };

  await assert.rejects(runtime.loadModelFromUrl('https://models.test/one.gguf', {
    nGpuLayers: 0,
    forceRemoteFetchBackend: false,
  }), { code: 'MODEL_STAGING_MOUNT_FAILED', message: /inspect.*permission denied/ });
  assert.ok(!ccalls.some(([name]) => name === 'llamadart_webgpu_mount_js_model_fs'));
});

test('JS-backed staging rolls back an unverified mount before downloading', async () => {
  const { runtime, core, fs } = createStagingHarness();
  const ccall = core.ccall.bind(core);
  core.ccall = (name, ...args) =>
    name === 'llamadart_webgpu_js_model_fs_mount_id' ? 0 : ccall(name, ...args);
  let downloads = 0;
  const getResponse = runtime._getCachedModelResponse;
  runtime._getCachedModelResponse = (...args) => {
    downloads++;
    return getResponse(...args);
  };

  await assert.rejects(runtime.loadModelFromUrl('https://models.test/one.gguf', {
    nGpuLayers: 0,
    forceRemoteFetchBackend: false,
  }), { code: 'MODEL_STAGING_MOUNT_FAILED', message: /verify native mount identity/ });
  assert.equal(downloads, 0);
  assert.deepEqual(fs.unmountedPaths, ['/jsmodels-1']);
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

test('Safari Load failed resumes JS-backed staging with an HTTP 206 append', async () => {
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
        controller.error(new TypeError('Load failed'));
      },
    }), {
      status: 200,
      headers: { 'content-encoding': 'identity' },
    });
  };

  const staged = new Uint8Array(10);
  const resumeHarness = createStagingHarness({ apiVersion: 4 });
  assert.ok(resumeHarness.core._llamadart_webgpu_js_model_fs_open_direct_heap);
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
  assert.ok(!resumeHarness.ccalls.some(([name]) => name === 'llamadart_webgpu_js_model_fs_open_direct_heap'));
  const openCalls = resumeHarness.ccalls
    .filter((c) => c[0] === 'llamadart_webgpu_js_model_fs_open_direct')
    .map(([, filePath, preallocBytes, append]) => ({ filePath, preallocBytes, append }));
  assert.deepEqual(openCalls, [
    { filePath: '/jsmodels-1/resumed.gguf', preallocBytes: 0, append: 0 },
    { filePath: '/jsmodels-1/resumed.gguf', preallocBytes: 0, append: 1 },
  ]);
  assert.deepEqual(
    resumeHarness.directWrites.map(({ position, length }) => ({ position, length })),
    [
      { position: 0, length: 4 },
      { position: 4, length: 6 },
    ],
  );
  assert.deepEqual(staged, fullPayload);
});

test('the Safari load failure match requires the exact TypeError message', () => {
  assert.equal(isRetryableStreamNetworkError(new TypeError('Load failed')), true);
  assert.equal(isRetryableStreamNetworkError(new TypeError('Model load failed')), false);
  assert.equal(isRetryableStreamNetworkError(new Error('Load failed')), false);
  assert.equal(isRetryableStreamNetworkError(new DOMException('Load failed', 'AbortError')), false);
});

test('heap-file overrun preserves range restart and reports a false full length as a transfer error', async (t) => {
  await t.test('the stream rejects an oversized range that reaches EOF', async () => {
    const writes = [];
    const response = new Response(Uint8Array.of(3, 4, 5, 6, 7, 8), { status: 206 });
    await assert.rejects(writeResponseToFsFileWithProgress(response, {}, '/model.gguf', null, {
      directWriter: {
        writeChunk(chunk) { writes.push(chunk); return chunk.length; },
      },
      startOffset: 2,
      heapFileBytes: 6,
    }), { name: 'ModelTransferLengthError' });
    assert.deepEqual(writes, []);
  });
  for (const scenario of ['range overrun', 'closed overrun', 'valid append', 'ETag restart', 'false full length']) {
    await t.test(scenario, async () => {
      let request = 0;
      const { runtime, ccalls } = createStagingHarness({
        apiVersion: 4,
        failWriteWith: ({ position, length }) => (
          request <= 2 && position + length > 6 ? { errno: 22 } : null
        ),
      });
      const requests = [];
      const brokenResponse = (status, bytes, headers) => {
        let sent = false;
        return new Response(new ReadableStream({
          pull(controller) {
            if (sent) {
              controller.error(new TypeError('network error'));
            } else {
              sent = true;
              controller.enqueue(Uint8Array.from(bytes));
            }
          },
        }, { highWaterMark: 0 }), { status, headers });
      };
      runtime._getCachedModelResponse = async (_url, { requestHeaders }) => {
        requests.push(requestHeaders?.Range ?? null);
        request++;
        if (scenario === 'false full length') {
          return new Response(new Uint8Array(8), { headers: { 'x-linked-size': '6' } });
        }
        if (request === 1) {
          return brokenResponse(200, [1, 2], { 'content-length': '6', 'content-encoding': 'identity', etag: '"a"' });
        }
        if (request === 2) {
          const headers = {
            'content-range': 'bytes 2-5/6', 'content-encoding': 'identity',
            etag: scenario === 'ETag restart' ? '"b"' : '"a"',
          };
          if (scenario === 'closed overrun' || scenario === 'valid append') {
            const suffix = scenario === 'valid append' ? [3, 4, 5, 6] : [3, 4, 5, 6, 7, 8];
            return new Response(Uint8Array.from(suffix), { status: 206, headers });
          }
          return brokenResponse(206, [3, 4, 5, 6, 7, 8], headers);
        }
        return new Response(new Uint8Array(6));
      };
      const load = runtime.loadModelFromUrl('https://models.test/overrun.gguf', {
        nGpuLayers: 0, useCache: false, forceRemoteFetchBackend: false,
      });
      if (scenario === 'false full length' || scenario === 'closed overrun') {
        await assert.rejects(load, { name: 'ModelTransferLengthError' });
        assert.deepEqual(requests, scenario === 'closed overrun' ? [null, 'bytes=2-'] : [null]);
        assert.ok(!ccalls.some(([name]) => name === 'llamadart_webgpu_load_model'));
        return;
      }
      assert.equal(await load, 1);
      if (scenario === 'valid append') {
        assert.deepEqual(requests, [null, 'bytes=2-']);
        assert.ok(runtime._runtimeNotes.includes('model_staging_heap_resident'));
        return;
      }
      assert.deepEqual(requests, [null, 'bytes=2-', null]);
      const reason = scenario === 'ETag restart' ? 'etag' : 'range';
      assert.ok(runtime._runtimeNotes.includes(`model_stream_restart:${reason}`));
      assert.ok(!runtime._runtimeNotes.includes('model_staging_heap_resident'));
      assert.ok(runtime._runtimeNotes.includes('model_staging_heap_resident_unavailable:unknown_size'));
    });
  }
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
    assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_mmproj_load' && c[1] === '/jsmodels-1/mmproj-f16.gguf'));
    assert.ok(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_unmount_js_model_fs' && c[1] === '/jsmodels-1'));
    assert.ok(harness.runtime._runtimeNotes.includes('mmproj_staging_js_backed'));
    assert.ok(harness.runtime._runtimeNotes.includes('mmproj_file_released'));
    assert.equal(harness.openDirectHandles.size, 0);
    assert.equal(harness.fs.files.size, 0);
  }

  // OOM during projector staging translates cleanly to ModelStagingError and unmounts /jsmodels-1.
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
  assert.deepEqual(oomHarness.fs.unmountedPaths, ['/jsmodels-1']);
});

test('a trap loading a staged projector preserves the original error', async (t) => {
  const { runtime, core } = createStagingHarness({ guard: true });
  const trap = new WebAssembly.RuntimeError('unreachable');
  const ccall = core.ccall;
  t.mock.method(core, 'ccall', (name, ...args) => {
    if (name === 'llamadart_webgpu_mmproj_load') throw trap;
    return ccall(name, ...args);
  });
  runtime._modelBytes = 1024;
  runtime._fetchWithTimeout = async () => new Response(new Uint8Array([1, 2, 3]), {
    status: 200,
    headers: { 'content-length': '3', 'content-encoding': 'identity' },
  });

  await assert.rejects(runtime.loadMultimodalProjector('https://models.test/mmproj-f16.gguf'),
    (error) => error === trap);
  assert.equal(runtime._activeJsStagingMounts.size, 0);
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

test('projector ENOMEM after 503 worker-replacement retry preserves ModelStagingError code', async () => {
  let projectorCalls = 0;
  let proxyCreations = 0;
  const b = new LlamaWebGpuBridge({ disableWorker: true, jsBackedModelStaging: true });
  b._shouldUseWorker = () => true;
  b._createWorkerProxy = () => {
    proxyCreations += 1;
    const state = {
      modelBytes: 0,
      metadata: {},
      contextSize: 128,
      gpuActive: false,
      backendName: 'WASM',
      supportsVision: false,
      supportsAudio: false,
    };
    return {
      dispose() {},
      terminate() {},
      async call(method) {
        if (method === 'loadModelFromUrl') {
          state.modelBytes = 1024;
          state.metadata = { 'llamadart.webgpu.model_bytes': '1024' };
          return { result: 1, state: { ...state } };
        }
        if (method === 'loadMultimodalProjector') {
          projectorCalls += 1;
          if (projectorCalls <= 1) {
            throw new Error('Failed to fetch multimodal projector: 503 Service Unavailable');
          }
          throw new ModelStagingError(
            'MODEL_STAGING_ALLOCATION_FAILED',
            'Failed to allocate JavaScript memory while staging the model (48 bytes).',
          );
        }
        return { result: 1, state: { ...state } };
      },
    };
  };
  b._runtime = null;
  b._workerProxy = b._createWorkerProxy();
  b._workerGeneration = 1;

  await b.loadModelFromUrl('https://models.test/tiny.gguf', { nCtx: 128, nGpuLayers: 0 });
  const err = await b.loadMultimodalProjector('https://models.test/retry-oom.gguf')
    .then(() => assert.fail('must reject with ModelStagingError'), (e) => e);
  assert.equal(proxyCreations, 2, 'must replace worker proxy once before retry');
  assert.equal(err instanceof ModelStagingError, true, `Expected ModelStagingError, got ${err?.name}: ${err?.message}`);
  assert.equal(err.code, 'MODEL_STAGING_ALLOCATION_FAILED');
  await b.dispose();
});

test('projector mount collision preserves pre-existing /mmproj file', async () => {
  const harness = createStagingHarness({
    jsBackedModelStaging: true,
    mountResult: -20,
  });
  harness.runtime._modelBytes = 1024;
  harness.fs.dirs.add('/mmproj');
  harness.fs.files.set('/mmproj/existing.gguf', 4096);

  const err = await harness.runtime
    .loadMultimodalProjector('https://models.test/existing.gguf')
    .then(() => assert.fail('must reject mount collision'), (e) => e);
  assert.ok(err instanceof ModelStagingError, String(err));
  assert.equal(err.code, 'MODEL_STAGING_MOUNT_FAILED');
  assert.equal(
    harness.fs.files.get('/mmproj/existing.gguf'),
    4096,
    'failed /jsmodels mount must not delete pre-existing /mmproj file',
  );
});

test('failed Safari GPU probe fallback unmounts and releases JS-backed staged model', async () => {
  const harness = createStagingHarness({
    jsBackedModelStaging: true,
    pthreads: false,
    onLoadModel: ({ args }) => (args[6] === 999 ? 0 : -1),
  });
  harness.runtime._isSafari = true;
  harness.runtime._resolveNativeLoadOptions = (opts) => {
    harness.runtime._nCtx = opts.nCtx;
    harness.runtime._nGpuLayers = opts.nGpuLayers;
  };
  harness.runtime._fetchWithTimeout = async () => new Response(new Uint8Array([1, 2, 3, 4]));
  harness.runtime.createCompletion = async () => {
    throw new Error('corrupted safari gpu output');
  };

  await assert.rejects(
    harness.runtime.loadModelFromUrl('https://models.test/safari-fail.gguf', {
      nCtx: 2048,
      nGpuLayers: 999,
      safariMaxGpuLayers: 999,
      useCache: false,
      forceRemoteFetchBackend: false,
    }),
    /Safari GPU probe failed and fallback attempts were unsuccessful/,
  );
  assert.equal(harness.runtime._jsBackedStagingMounted, false);
  assert.deepEqual(harness.fs.unmountedPaths, ['/jsmodels-1']);
  assert.equal(harness.openDirectHandles.size, 0);
});

test('staging off preserves main cancellation error on replacement worker projector retry and parity behaviors', async () => {
  // 1) Cancellation during replacement-worker projector retry wraps as plain Error when staging is off.
  let projectorAttempt = 0;
  let b;
  b = new LlamaWebGpuBridge({ disableWorker: true });
  b._shouldUseWorker = () => true;
  b._createWorkerProxy = () => {
    const state = {
      modelBytes: 0,
      metadata: {},
      contextSize: 128,
      gpuActive: false,
      backendName: 'WASM (CPU)',
      supportsVision: false,
      supportsAudio: false,
    };
    return {
      dispose() {},
      terminate() {},
      async call(method) {
        if (method === 'loadModelFromUrl') {
          state.modelBytes = 1024;
          state.metadata = { 'llamadart.webgpu.model_bytes': '1024' };
          return { result: 1, state: { ...state } };
        }
        if (method === 'loadMultimodalProjector') {
          projectorAttempt += 1;
          if (projectorAttempt === 1) {
            throw new Error('Bridge worker crashed');
          }
          b.cancel();
          throw new Error('Projector operation was cancelled.');
        }
        return { result: 1, state: { ...state } };
      },
    };
  };
  b._runtime = null;
  b._workerProxy = b._createWorkerProxy();
  b._workerGeneration = 1;

  await b.loadModelFromUrl('https://models.test/cpu.gguf', { nCtx: 128, nGpuLayers: 0 });
  const cancelErr = await b.loadMultimodalProjector('https://models.test/mmproj.gguf')
    .then(() => assert.fail('must reject'), (e) => e);
  assert.equal(cancelErr.name, 'Error');
  assert.match(cancelErr.message, /^CPU multimodal projector setup failed \(/);
  await b.dispose();

  // 2) Custom core returning -4 with staging off throws plain Error, not ModelStagingError.
  const offHarness = createStagingHarness({
    jsBackedModelStaging: false,
    onLoadModel: () => -4,
  });
  const offErr = await offHarness.runtime.loadModelFromUrl('https://models.test/custom.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject -4'), (e) => e);
  assert.equal(offErr instanceof ModelStagingError, false);
  assert.equal(offErr.name, 'Error');

  // 3) Unknown-length projector with staging off performs the exact 6 header reads.
  const headerGets = [];
  const offMmprojHarness = createStagingHarness({ jsBackedModelStaging: false });
  offMmprojHarness.runtime._modelBytes = 1024;
  offMmprojHarness.runtime._fetchWithTimeout = async () => ({
    ok: true,
    status: 200,
    type: 'basic',
    headers: {
      get(name) {
        headerGets.push(String(name).toLowerCase());
        return null;
      },
      has() {
        return false;
      },
    },
    body: new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array([1, 2, 3, 4]));
        controller.close();
      },
    }),
  });
  await offMmprojHarness.runtime.loadMultimodalProjector('https://models.test/unknown.gguf');
  assert.equal(headerGets.length, 6, `expected 6 header reads matching main, got ${headerGets.length}: ${headerGets.join(', ')}`);
});

test('no-stream empty-body 1 GiB response rejects with ModelTransferLengthError before writing or loading projector', async () => {
  const fs = createStubFs();
  let writes = 0;
  let closed = false;
  await assert.rejects(
    writeResponseToFsFileWithProgress(
      new Response(null, {
        status: 200,
        headers: {
          'content-length': String(2 ** 30),
          'content-encoding': 'identity',
        },
      }),
      fs,
      '/jsmodels/empty.gguf',
      null,
      {
        skipMemoryReservation: true,
        directWriter: {
          writeChunk() {
            writes += 1;
            return 0;
          },
          close() {
            closed = true;
          },
        },
      },
    ),
    { name: 'ModelTransferLengthError' },
  );
  assert.equal(writes, 0);
  assert.equal(closed, true);

  // Full projector path: empty-body 200 claiming 1 GiB must never invoke llamadart_webgpu_mmproj_load.
  const harness = createStagingHarness({ jsBackedModelStaging: true });
  harness.runtime._modelBytes = 1024;
  harness.runtime._fetchWithTimeout = async () => new Response(null, {
    status: 200,
    headers: {
      'content-length': String(2 ** 30),
      'content-encoding': 'identity',
    },
  });
  await assert.rejects(
    harness.runtime.loadMultimodalProjector('https://models.test/empty-mmproj.gguf'),
    { name: 'ModelTransferLengthError' },
  );
  assert.equal(harness.ccalls.some((c) => c[0] === 'llamadart_webgpu_mmproj_load'), false);
  assert.deepEqual(harness.fs.unmountedPaths, ['/jsmodels-1']);
});

test('staging ownership preserves user files at /jsmodels, former staging paths, and other mounts', async () => {
  const harness = createStagingHarness({ jsBackedModelStaging: true });
  harness.runtime._resolveNativeLoadOptions = (opts) => {
    harness.runtime._nCtx = opts.nCtx;
    harness.runtime._nGpuLayers = opts.nGpuLayers;
  };
  harness.runtime._fetchWithTimeout = async () => new Response(new Uint8Array([1, 2, 3, 4]), {
    status: 200,
    headers: { 'content-length': '4', 'content-encoding': 'identity' },
  });
  await harness.runtime.loadModelFromUrl('https://models.test/demo.gguf', {
    nCtx: 128,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  await harness.runtime.loadMultimodalProjector('https://models.test/mmproj.gguf');

  // Create user files at /jsmodels/*, the former staging paths /jsmodels-1/* & /jsmodels-2/*,
  // and on an unrelated mount /snapshots/*.
  harness.fs.mkdir('/jsmodels');
  harness.fs.mkdir('/snapshots');
  harness.fs.writeFile('/jsmodels/demo.gguf', new Uint8Array([1, 2, 3]));
  harness.fs.writeFile('/jsmodels-1/demo.gguf', new Uint8Array([4, 5, 6]));
  harness.fs.writeFile('/jsmodels-2/mmproj.gguf', new Uint8Array([7, 8, 9]));
  harness.fs.writeFile('/snapshots/keep.bin', new Uint8Array([9, 4, 1]));

  await harness.runtime.unloadMultimodalProjector();
  assert.equal(harness.fs.files.get('/jsmodels-2/mmproj.gguf'), 3);
  assert.equal(harness.fs.files.get('/snapshots/keep.bin'), 3);

  // Replace model; user files in /jsmodels, /jsmodels-1, and /snapshots must all survive.
  await harness.runtime.loadModelFromUrl('https://models.test/demo.gguf', {
    nCtx: 128,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(harness.fs.files.get('/jsmodels/demo.gguf'), 3);
  assert.equal(harness.fs.files.get('/jsmodels-1/demo.gguf'), 3);
  assert.equal(harness.fs.files.get('/snapshots/keep.bin'), 3);
  assert.equal(harness.fs.unmountedPaths.includes('/snapshots'), false);
  assert.equal(harness.fs.unmountedPaths.includes('/jsmodels'), false);
});

test('staging off preserves existing projector capabilities on /mmproj mkdir failure, single root read on retry, and error semantics', async () => {
  // 1) /mmproj directory creation failure must not clear loaded projector state, and retry reads '/' once.
  const harness = createStagingHarness({ jsBackedModelStaging: false });
  harness.runtime._modelBytes = 1024;
  harness.runtime._fetchWithTimeout = async () => new Response(new Uint8Array([1, 2, 3, 4]), {
    status: 200,
    headers: { 'content-length': '4', 'content-encoding': 'identity' },
  });
  let supportsVisionNative = 1;
  const origCcall = harness.core.ccall.bind(harness.core);
  let mmprojAttempts = 0;
  harness.core.ccall = (name, returnType, argTypes, args = [], opts) => {
    if (name === 'llamadart_webgpu_mmproj_supports_vision') {
      return supportsVisionNative;
    }
    if (name === 'llamadart_webgpu_mmproj_load') {
      mmprojAttempts += 1;
      if (mmprojAttempts === 2) {
        throw new ModelStagingError('MODEL_STAGING_UNSUPPORTED', 'network error from custom core');
      }
    }
    return origCcall(name, returnType, argTypes, args, opts);
  };
  await harness.runtime.loadMultimodalProjector('https://models.test/first.gguf');
  assert.equal(harness.runtime._mmSupportsVision, true);

  let rootReads = 0;
  const origReaddir = harness.fs.readdir.bind(harness.fs);
  harness.fs.readdir = (dir) => {
    if (dir === '/') {
      rootReads += 1;
    }
    return origReaddir(dir);
  };
  // Custom core throwing ModelStagingError with retryable network message still retries when staging is off,
  // and ensureFsDirectory('/mmproj') runs once before the retry loop (1 root read total).
  await harness.runtime.loadMultimodalProjector('https://models.test/retry.gguf');
  assert.equal(mmprojAttempts, 3);
  assert.equal(rootReads, 1);

  const origMkdir = harness.fs.mkdir.bind(harness.fs);
  harness.fs.readdir = (dir) => (dir === '/' ? origReaddir(dir).filter((x) => x !== 'mmproj') : origReaddir(dir));
  harness.fs.mkdir = (dir) => {
    if (dir === '/mmproj') {
      throw new Error('directory unavailable');
    }
    return origMkdir(dir);
  };
  await assert.rejects(
    harness.runtime.loadMultimodalProjector('https://models.test/fail-dir.gguf'),
    /directory unavailable/,
  );
  assert.equal(harness.runtime._mmSupportsVision, true);
  assert.ok(harness.runtime._mmProjPath !== null);

  // 2) Worker proxy with jsBackedModelStaging off returns plain Error even if custom worker sends modelStagingErrorCode.
  await withStubWorkerEnvironment(async ({ workers }) => {
    const offBridge = createRealWorkerBridge({ jsBackedModelStaging: false });
    const proxy = offBridge._workerProxy;
    const driver = workerDriver(proxy, workers[0]);
    driver.ready();
    const pendingLoad = offBridge.loadModelFromUrl('https://models.test/x.gguf', { nCtx: 128, nGpuLayers: 0 });
    await driver.settle();
    const call = driver.calls('loadModelFromUrl')[0];
    proxy._worker.onmessage({
      data: {
        id: call.id,
        type: 'error',
        modelStagingErrorCode: 'MODEL_STAGING_UNSUPPORTED',
        message: 'custom worker refused load',
      },
    });
    const err = await pendingLoad.then(() => assert.fail('must reject'), (e) => e);
    assert.equal(err instanceof ModelStagingError, false);
    assert.equal(err.name, 'Error');
    assert.equal(err.message, 'custom worker refused load');
    await offBridge.dispose();
  });
});

test('jsBackedModelStaging rejects pre-v2 core missing _llamadart_webgpu_js_model_fs_api_version with MODEL_STAGING_UNSUPPORTED', async () => {
  const harness = createStagingHarness({ jsBackedModelStaging: true });
  delete harness.core._llamadart_webgpu_js_model_fs_api_version;

  const err = await harness.runtime.loadModelFromUrl('https://models.test/small.gguf', {
    nCtx: 2048,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  }).then(() => assert.fail('must reject'), (e) => e);

  assert.ok(err instanceof ModelStagingError, String(err));
  assert.equal(err.code, 'MODEL_STAGING_UNSUPPORTED');
  assert.match(err.message, /_llamadart_webgpu_js_model_fs_api_version/);
  assert.equal(harness.ccalls.length, 0, 'must reject v1 core before calling ccall');
});

test('failed unmount stays tracked and retries on next staging cleanup', async () => {
  const harness = createStagingHarness({ jsBackedModelStaging: true });
  harness.runtime._resolveNativeLoadOptions = (opts) => {
    harness.runtime._nCtx = opts.nCtx;
    harness.runtime._nGpuLayers = opts.nGpuLayers;
  };
  harness.runtime._fetchWithTimeout = async () => new Response(new Uint8Array([1, 2, 3, 4]));
  let busyMounts = new Set(['/jsmodels-1', '/jsmodels-2']);
  const origCcall = harness.core.ccall.bind(harness.core);
  const unmountAttempts = [];
  harness.core.ccall = (name, returnType, argTypes, args = [], opts) => {
    if (name === 'llamadart_webgpu_unmount_js_model_fs') {
      unmountAttempts.push(args[0]);
      if (busyMounts.has(args[0])) {
        return -16;
      }
    }
    return origCcall(name, returnType, argTypes, args, opts);
  };

  await harness.runtime.loadModelFromUrl('https://models.test/one.gguf', {
    nCtx: 128,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(harness.runtime._jsBackedStagingMounted, true);
  assert.equal(harness.runtime._activeJsStagingMounts.has('/jsmodels-1'), true);

  // Clear -EBUSY for /jsmodels-1 and load projector (/jsmodels-2, kept busy):
  // per-mount cleanup of /jsmodels-2 still retries /jsmodels-1 via _failedUnmountJsStagingMounts.
  busyMounts.delete('/jsmodels-1');
  await harness.runtime.loadMultimodalProjector('https://models.test/mmproj.gguf');
  assert.equal(harness.runtime._activeJsStagingMounts.has('/jsmodels-1'), false);
  assert.equal(harness.runtime._activeJsStagingMounts.has('/jsmodels-2'), true);
  assert.equal(harness.runtime._jsBackedStagingMounted, true);

  // The next staging cleanup retries /jsmodels-2 once unmount succeeds.
  busyMounts.clear();
  await harness.runtime.loadMultimodalProjector('https://models.test/mmproj.gguf');
  await harness.bridge.dispose();
  assert.equal(harness.runtime._activeJsStagingMounts.size, 0);
  assert.equal(harness.runtime._jsBackedStagingMounted, false);
  assert.ok(unmountAttempts.filter((p) => p === '/jsmodels-1').length >= 2);
  assert.ok(unmountAttempts.filter((p) => p === '/jsmodels-2').length >= 2);
});

test('failed-unmount retry checks native mount identity and never deletes a replacement mount', async () => {
  const harness = createStagingHarness({ jsBackedModelStaging: true });
  harness.runtime._resolveNativeLoadOptions = (opts) => {
    harness.runtime._nCtx = opts.nCtx;
    harness.runtime._nGpuLayers = opts.nGpuLayers;
  };
  const busyMounts = new Set(['/jsmodels-1']);
  const origCcall = harness.core.ccall.bind(harness.core);
  const unmountAttempts = [];
  harness.core.ccall = (name, returnType, argTypes, args = [], opts) => {
    if (name === 'llamadart_webgpu_unmount_js_model_fs') {
      unmountAttempts.push(args[0]);
      if (busyMounts.has(args[0])) {
        return -28;
      }
    }
    return origCcall(name, returnType, argTypes, args, opts);
  };

  await harness.runtime.loadModelFromUrl('https://models.test/one.gguf', {
    nCtx: 128,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(harness.runtime._failedUnmountJsStagingMounts.get('/jsmodels-1'), 1);

  // Another owner unmounts /jsmodels-1 and remounts the same path with a new mount id, storing a checkpoint.
  busyMounts.delete('/jsmodels-1');
  assert.equal(origCcall('llamadart_webgpu_unmount_js_model_fs', 'number', ['string'], ['/jsmodels-1']), 0);
  assert.equal(origCcall('llamadart_webgpu_mount_js_model_fs', 'number', ['string'], ['/jsmodels-1']), 0);
  const replacementMountId = origCcall('llamadart_webgpu_js_model_fs_mount_id', 'number', ['string'], ['/jsmodels-1']);
  assert.ok(replacementMountId > 1);
  harness.fs.writeFile('/jsmodels-1/checkpoint.bin', new Uint8Array([3, 8, 12]));

  // Loading the next model retries cleanup, sees mount_id changed, drops the stale retry, and preserves the checkpoint.
  const attemptsBeforeSecondLoad = unmountAttempts.length;
  await harness.runtime.loadModelFromUrl('https://models.test/two.gguf', {
    nCtx: 128,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(harness.fs.files.has('/jsmodels-1/checkpoint.bin'), true);
  assert.equal(harness.mountIds.get('/jsmodels-1'), replacementMountId);
  assert.equal(unmountAttempts.slice(attemptsBeforeSecondLoad).includes('/jsmodels-1'), false);
  assert.equal(harness.runtime._failedUnmountJsStagingMounts.has('/jsmodels-1'), false);
  await harness.bridge.dispose();
  assert.equal(harness.fs.files.has('/jsmodels-1/checkpoint.bin'), true);
});

for (const kind of ['model', 'projector']) {
  for (const replacement of ['another mount', 'an ordinary directory']) {
    test(`${kind} staging rejects when ${replacement} replaces the mount during fetch`, async () => {
      const { bridge, runtime, core, fs, mountIds, releasedChunkMountIds, ccalls } = createStagingHarness();
      const mountPath = '/jsmodels-1';
      const filePath = `${mountPath}/one.gguf`;
      const checkpoint = new Uint8Array([17, 23, 41, 97]);
      const requested = Promise.withResolvers();
      const response = Promise.withResolvers();
      const pendingFetch = () => {
        requested.resolve();
        return response.promise;
      };
      runtime._getCachedModelResponse = pendingFetch;
      runtime._fetchWithTimeout = pendingFetch;
      if (kind === 'projector') {
        runtime._modelBytes = 4;
      }
      const url = 'https://models.test/one.gguf';
      const pendingLoad = kind === 'model'
        ? bridge.loadModelFromUrl(url, { nGpuLayers: 0, useCache: false, forceRemoteFetchBackend: false })
        : bridge.loadMultimodalProjector(url);

      await requested.promise;
      const oldId = mountIds.get(mountPath);
      assert.ok(oldId > 0);
      if (replacement === 'another mount') {
        core.ccall('llamadart_webgpu_unmount_js_model_fs', 'number', ['string'], [mountPath]);
        core.ccall('llamadart_webgpu_mount_js_model_fs', 'number', ['string'], [mountPath]);
        assert.notEqual(mountIds.get(mountPath), oldId);
      } else {
        mountIds.delete(mountPath);
      }
      fs.files.set(filePath, checkpoint);
      const callStart = ccalls.length;
      releasedChunkMountIds.length = 0;
      response.resolve(new Response(new Uint8Array([1, 2, 3, 4, 5])));

      await assert.rejects(pendingLoad, { name: 'ModelStagingError', code: 'MODEL_STAGING_MOUNT_FAILED' });
      assert.deepEqual(fs.files.get(filePath), checkpoint);
      assert.equal(fs.dirs.has(mountPath), true);
      assert.equal(ccalls.slice(callStart).some(([name]) => name === 'llamadart_webgpu_js_model_fs_open_direct'), false);
      assert.deepEqual(releasedChunkMountIds, [oldId]);
      await bridge.dispose();
    });
  }

  for (const phase of ['write', 'close', 'reader cancellation', 'before native load', 'native load']) {
    test(`${kind} staging preserves a replacement file after the writer opens (${phase})`, async () => {
      const { bridge, runtime, core, fs, mountIds, releasedChunkMountIds, ccalls, directWrites } = createStagingHarness();
      const mountPath = '/jsmodels-1';
      const filePath = `${mountPath}/one.gguf`;
      const checkpoint = new Uint8Array([17, 23, 41, 97]);
      let replacedId = 0;
      const replaceMount = () => {
        if (replacedId === 0) {
          replacedId = mountIds.get(mountPath);
          mountIds.delete(mountPath);
          fs.files.set(filePath, checkpoint);
        }
      };
      const write = core.llamadartJsModelFsWriteDirect;
      core.llamadartJsModelFsWriteDirect = (...args) => {
        const written = write(...args);
        if (phase === 'write' || phase === 'close') {
          replaceMount();
        }
        return written;
      };
      const nativeLoadName = kind === 'model' ? 'llamadart_webgpu_load_model' : 'llamadart_webgpu_mmproj_load';
      const ccall = core.ccall.bind(core);
      core.ccall = (name, ...args) => {
        const result = ccall(name, ...args);
        if ((phase === 'native load' && name === nativeLoadName)
          || (phase === 'before native load' && name === 'llamadart_webgpu_js_model_fs_close_direct')) {
          replaceMount();
        }
        return result;
      };
      const transfer = async () => new Response(new ReadableStream({
        start(controller) {
          if (phase !== 'reader cancellation') {
            controller.enqueue(new Uint8Array([1, 2, 3]));
            if (phase === 'write') {
              controller.enqueue(new Uint8Array([4, 5]));
            }
            controller.close();
          }
        },
        cancel: replaceMount,
      }));
      runtime._getCachedModelResponse = transfer;
      runtime._fetchWithTimeout = transfer;
      if (phase === 'reader cancellation') {
        runtime._resolveStreamChunkTimeoutMs = () => 1;
      }
      if (kind === 'projector') {
        runtime._modelBytes = 4;
      }
      const url = 'https://models.test/one.gguf';
      const pendingLoad = kind === 'model'
        ? bridge.loadModelFromUrl(url, { nGpuLayers: 0, useCache: false, forceRemoteFetchBackend: false })
        : bridge.loadMultimodalProjector(url);

      await assert.rejects(pendingLoad, { name: 'ModelStagingError', code: 'MODEL_STAGING_MOUNT_FAILED' });
      assert.equal(replacedId, 1);
      assert.deepEqual(fs.files.get(filePath), checkpoint);
      assert.deepEqual(releasedChunkMountIds, [replacedId]);
      if (phase === 'native load') {
        const freeName = kind === 'model' ? 'llamadart_webgpu_free_model' : 'llamadart_webgpu_mmproj_free';
        const loadIndex = ccalls.findIndex(([name]) => name === nativeLoadName);
        const freeIndex = ccalls.findIndex(([name]) => name === freeName);
        assert.ok(loadIndex >= 0 && freeIndex > loadIndex);
        if (kind === 'model') {
          const beforeFree = ccalls.slice(loadIndex + 1, freeIndex);
          assert.ok(!beforeFree.some(([name]) => name === 'llamadart_webgpu_get_context_size'));
        }
      } else if (phase === 'before native load') {
        assert.equal(directWrites.length, 1);
        assert.ok(ccalls.some(([name]) => name === 'llamadart_webgpu_js_model_fs_close_direct'));
        assert.ok(!ccalls.some(([name]) => name === nativeLoadName));
      } else if (phase === 'write' || phase === 'close') {
        assert.equal(directWrites.length, 1);
        assert.equal(ccalls.some(([name]) => name === 'llamadart_webgpu_js_model_fs_close_direct'), false);
      }
      await bridge.dispose();
    });
  }
}

function replaceStagingMount({ fs, mountIds }, filePath) {
  const mountPath = path.posix.dirname(filePath);
  const originalId = mountIds.get(mountPath);
  assert.ok(originalId > 0);
  const replacementId = originalId + 100;
  const checkpoint = new Uint8Array([17, 23, 41, 97]);
  mountIds.set(mountPath, replacementId);
  fs.files.set(filePath, checkpoint);
  return { filePath, mountPath, originalId, replacementId, checkpoint };
}

function assertStagingReplacementPreserved({ fs, mountIds, releasedChunkMountIds }, replacement) {
  const { filePath, mountPath, originalId, replacementId, checkpoint } = replacement;
  assert.deepEqual(fs.files.get(filePath), checkpoint);
  assert.equal(mountIds.get(mountPath), replacementId);
  assert.ok(releasedChunkMountIds.includes(originalId));
  assert.ok(!releasedChunkMountIds.includes(replacementId));
}

for (const phase of ['fetch', 'cache settlement']) {
  test(`model staging reports ownership loss after a failed ${phase}`, async () => {
    const harness = createStagingHarness();
    const { bridge, runtime, ccalls } = harness;
    let replacement;
    const failAfterReplacement = async () => {
      replacement = replaceStagingMount(harness, '/jsmodels-1/one.gguf');
      throw new Error('injected transfer failure');
    };
    if (phase === 'fetch') {
      runtime._getCachedModelResponse = failAfterReplacement;
    } else {
      runtime._settleModelCacheStore = failAfterReplacement;
    }

    await assert.rejects(bridge.loadModelFromUrl('https://models.test/one.gguf', {
      nGpuLayers: 0,
      useCache: false,
      forceRemoteFetchBackend: false,
    }), { name: 'ModelStagingError', code: 'MODEL_STAGING_MOUNT_FAILED' });
    assertStagingReplacementPreserved(harness, replacement);
    assert.ok(!ccalls.some(([name]) => name === 'llamadart_webgpu_js_model_fs_consume_oom_bytes'));
    await bridge.dispose();
  });
}

for (const nativeLoadSucceeded of [true, false]) {
  const outcome = nativeLoadSucceeded ? 'successful' : 'failed';
  test(`split model rechecks ownership between shard unlinks after a ${outcome} native load`, async () => {
    const harness = createStagingHarness({ onLoadModel: () => nativeLoadSucceeded ? 0 : -1 });
    const { bridge, fs } = harness;
    const firstName = 'model-00001-of-00002.gguf';
    const secondName = 'model-00002-of-00002.gguf';
    const firstPath = `/jsmodels-1/${firstName}`;
    const secondPath = `/jsmodels-1/${secondName}`;
    const unlink = fs.unlink.bind(fs);
    let replacement;
    fs.unlink = (filePath) => {
      unlink(filePath);
      if (filePath === firstPath) {
        replacement = replaceStagingMount(harness, secondPath);
      }
    };

    await assert.rejects(bridge.loadModelFromUrl([
      `https://models.test/${firstName}`,
      `https://models.test/${secondName}`,
    ], { nGpuLayers: 0, useCache: false }), nativeLoadSucceeded
      ? { name: 'ModelStagingError', code: 'MODEL_STAGING_MOUNT_FAILED' }
      : /Failed to load GGUF model/);
    assert.ok(!fs.files.has(firstPath));
    assertStagingReplacementPreserved(harness, replacement);
    await bridge.dispose();
  });
}

test('Safari fallback does not reload a model from a replacement staging mount', async () => {
  const harness = createStagingHarness();
  const { bridge, runtime, ccalls } = harness;
  runtime._isSafari = true;
  runtime._resolveNativeLoadOptions = (options) => {
    runtime._nGpuLayers = options.nGpuLayers;
  };
  let replacement;
  runtime.createCompletion = async () => {
    replacement ??= replaceStagingMount(harness, '/jsmodels-1/one.gguf');
    throw new Error('injected Safari GPU probe failure');
  };

  await assert.rejects(bridge.loadModelFromUrl('https://models.test/one.gguf', {
    nGpuLayers: 2,
    safariMaxGpuLayers: 2,
    useCache: false,
    forceRemoteFetchBackend: false,
  }), { name: 'ModelStagingError', code: 'MODEL_STAGING_MOUNT_FAILED' });
  assert.equal(ccalls.filter(([name]) => name === 'llamadart_webgpu_load_model').length, 1);
  assertStagingReplacementPreserved(harness, replacement);
  await bridge.dispose();
});

for (const cleanup of ['unload', 'model replacement']) {
  test(`runtime ${cleanup} protects a replacement mount while projector load is in flight`, async () => {
    const harness = createStagingHarness();
    const { bridge, runtime, core } = harness;
    const modelOptions = { nGpuLayers: 0, useCache: false, forceRemoteFetchBackend: false };
    await runtime.loadModelFromUrl('https://models.test/one.gguf', modelOptions);
    runtime._fetchWithTimeout = runtime._getCachedModelResponse;

    const projectorStarted = Promise.withResolvers();
    const projectorResult = Promise.withResolvers();
    const ccall = core.ccall.bind(core);
    core.ccall = (name, ...args) => {
      if (name === 'llamadart_webgpu_mmproj_load') {
        projectorStarted.resolve(args[2][0]);
        return projectorResult.promise;
      }
      return ccall(name, ...args);
    };

    // Public bridge operations are serialized. Exercise the runtime's fallback
    // guards while the projector still owns a staged path.
    const pendingLoad = assert.rejects(
      runtime.loadMultimodalProjector('https://models.test/mmproj.gguf'),
      { name: 'ModelStagingError', code: 'MODEL_STAGING_MOUNT_FAILED' },
    );
    const projectorPath = await Promise.race([projectorStarted.promise, pendingLoad.then(() => {
      assert.fail('projector load rejected before entering native code');
    })]);
    const replacement = replaceStagingMount(harness, projectorPath);
    try {
      if (cleanup === 'unload') {
        await runtime.unloadMultimodalProjector();
      } else {
        await runtime.loadModelFromUrl('https://models.test/two.gguf', modelOptions);
      }
    } finally {
      projectorResult.resolve(-1);
      await pendingLoad;
    }
    assertStagingReplacementPreserved(harness, replacement);
    await bridge.dispose();
  });
}

test('dispose releases chunks by mount id before shutdown even when unmount fails', async () => {
  const harness = createStagingHarness({ jsBackedModelStaging: true });
  harness.runtime._resolveNativeLoadOptions = (opts) => {
    harness.runtime._nCtx = opts.nCtx;
    harness.runtime._nGpuLayers = opts.nGpuLayers;
  };
  harness.runtime._fetchWithTimeout = async () => new Response(new Uint8Array([1, 2, 3, 4]));
  const origCcall = harness.core.ccall.bind(harness.core);
  let shutdownCalls = 0;
  harness.core.ccall = (name, returnType, argTypes, args = [], opts) => {
    if (name === 'llamadart_webgpu_unmount_js_model_fs') {
      return -28;
    }
    if (name === 'llamadart_webgpu_shutdown') {
      shutdownCalls += 1;
      assert.deepEqual(harness.releasedChunkMountIds, [1, 2]);
    }
    return origCcall(name, returnType, argTypes, args, opts);
  };

  await harness.runtime.loadModelFromUrl('https://models.test/one.gguf', {
    nCtx: 128,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  await harness.runtime.loadMultimodalProjector('https://models.test/mmproj.gguf');

  const first = harness.bridge.dispose();
  await first;
  assert.equal(harness.bridge._runtime, null);
  assert.equal(harness.bridge.dispose(), first);
  assert.equal(shutdownCalls, 1);
});

test('staging off dispose keeps the same rejected promise after native shutdown fails', async () => {
  const { bridge, core } = createStagingHarness({ jsBackedModelStaging: false });
  const nativeError = new Error('injected shutdown failure');
  let shutdownCalls = 0;
  const ccall = core.ccall.bind(core);
  core.ccall = (name, ...args) => {
    if (name === 'llamadart_webgpu_shutdown' && ++shutdownCalls === 1) {
      throw nativeError;
    }
    return ccall(name, ...args);
  };

  const first = bridge.dispose();
  assert.equal(bridge.dispose(), first);
  await assert.rejects(first, (error) => error === nativeError);
  const second = bridge.dispose();
  assert.equal(second, first);
  await assert.rejects(second, (error) => error === nativeError);
  assert.equal(shutdownCalls, 1);
  assert.equal(bridge.getContextSize(), 0);
});

test('staging off parity trace avoids Response.type, core.ABORT, _ownedModelPaths, and undefined worker key', async () => {
  const offBridge = new LlamaWebGpuBridge({ disableWorker: true });
  assert.equal(Object.hasOwn(offBridge._workerConfig(), 'jsBackedModelStaging'), false);
  await offBridge.dispose();

  const harness = createStagingHarness({ jsBackedModelStaging: false });
  harness.runtime._resolveNativeLoadOptions = (opts) => {
    harness.runtime._nCtx = opts.nCtx;
    harness.runtime._nGpuLayers = opts.nGpuLayers;
  };
  let responseTypeReads = 0;
  let coreAbortReads = 0;
  Object.defineProperty(harness.core, 'ABORT', {
    configurable: true,
    get() {
      coreAbortReads += 1;
      return false;
    },
  });

  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new Uint8Array([1, 2, 3, 4]));
      controller.close();
    },
  });
  const largeHeaders = new Headers({ 'content-length': '2497280640' });
  const fakeResponse = {
    ok: true,
    status: 200,
    statusText: 'OK',
    headers: largeHeaders,
    body: stream,
    get type() {
      responseTypeReads += 1;
      return 'basic';
    },
  };
  const fs = harness.fs;
  await writeResponseToFsFileWithProgress(fakeResponse, fs, '/models/large.gguf', null, {
    skipMemoryReservation: false,
  });
  assert.equal(responseTypeReads, 0, 'staging-off download > 2 GiB must not read Response.type');

  await harness.runtime.loadModelFromUrl('https://models.test/one.gguf', {
    nCtx: 128,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.deepEqual(harness.runtime._ownedModelPaths, []);
  await harness.runtime.loadModelFromUrl('https://models.test/two.gguf', {
    nCtx: 128,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.deepEqual(harness.runtime._ownedModelPaths, []);
  await harness.bridge.dispose();
  assert.equal(coreAbortReads, 0, 'staging-off cleanup and dispose must not read core.ABORT');
});

test('staging off cleanup matches main unchanged (retry-unlink, 404/503 unlink, replace/unload)', async () => {
  const harness = createStagingHarness({ jsBackedModelStaging: false });
  harness.runtime._resolveNativeLoadOptions = (opts) => {
    harness.runtime._nCtx = opts.nCtx;
    harness.runtime._nGpuLayers = opts.nGpuLayers;
  };
  let responseStatus = 200;
  delete harness.runtime._getCachedModelResponse;
  harness.runtime._fetchWithTimeout = async () => new Response(
    responseStatus === 200 ? new Uint8Array([1, 2, 3, 4]) : null,
    { status: responseStatus, statusText: responseStatus === 200 ? 'OK' : 'Error' },
  );

  // One-shot unlink failure on post-load cleanup leaves file until unload/replace retries and removes it.
  let failOnce = '/models/one.gguf';
  const attempted = [];
  const origUnlink = harness.fs.unlink.bind(harness.fs);
  harness.fs.unlink = (p) => {
    const exists = harness.fs.files.has(p);
    if (exists) {
      attempted.push(p);
    }
    if (exists && failOnce === p) {
      failOnce = '';
      throw new Error('EBUSY');
    }
    return origUnlink(p);
  };
  await harness.runtime.loadModelFromUrl('https://models.test/one.gguf', {
    nCtx: 128,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(harness.fs.files.has('/models/one.gguf'), true);

  failOnce = '/mmproj/mmproj.gguf';
  await harness.runtime.loadMultimodalProjector('https://models.test/mmproj.gguf');
  assert.equal(harness.fs.files.has('/mmproj/mmproj.gguf'), true);

  await harness.runtime.unloadMultimodalProjector();
  assert.equal(harness.fs.files.has('/mmproj/mmproj.gguf'), false);

  await harness.runtime.loadModelFromUrl('https://models.test/two.gguf', {
    nCtx: 128,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });
  assert.equal(harness.fs.files.has('/models/one.gguf'), false);
  assert.equal(attempted.filter((p) => p === '/models/one.gguf').length, 2);
  assert.equal(attempted.filter((p) => p === '/mmproj/mmproj.gguf').length, 2);

  // Pre-existing /models/* on 404 and /mmproj/* on 404/503 are unlinked when staging is off, matching main.
  harness.fs.writeFile('/models/missing_model404.gguf', new Uint8Array([7, 3, 9]));
  responseStatus = 404;
  await assert.rejects(
    harness.runtime.loadModelFromUrl('https://models.test/missing_model404.gguf', {
      nCtx: 128,
      nGpuLayers: 0,
      useCache: false,
      forceRemoteFetchBackend: false,
    }),
    /404/,
  );
  assert.equal(harness.fs.files.has('/models/missing_model404.gguf'), false);

  responseStatus = 200;
  await harness.runtime.loadModelFromUrl('https://models.test/two.gguf', {
    nCtx: 128,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });

  harness.fs.writeFile('/mmproj/missing404.gguf', new Uint8Array([7, 3, 9]));
  responseStatus = 404;
  await assert.rejects(
    harness.runtime.loadMultimodalProjector('https://models.test/missing404.gguf'),
    /404/,
  );
  assert.equal(harness.fs.files.has('/mmproj/missing404.gguf'), false);

  harness.fs.writeFile('/mmproj/err503.gguf', new Uint8Array([7, 3, 9]));
  responseStatus = 503;
  await assert.rejects(
    harness.runtime.loadMultimodalProjector('https://models.test/err503.gguf'),
    /503/,
  );
  assert.equal(harness.fs.files.has('/mmproj/err503.gguf'), false);
});

const IPHONE_CHROME = 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/153.0.7000.10 Mobile/15E148 Safari/604.1';
const IPAD_DESKTOP = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.5 Safari/605.1.15';
const MAC_SAFARI_27 = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/27.0 Safari/605.1.15';
const FIREFOX = 'Mozilla/5.0 (X11; Linux x86_64; rv:150.0) Gecko/20100101 Firefox/150.0';

test('an unset jsBackedModelStaging stages on WebKit and stays off elsewhere', () => {
  for (const [config, expected] of [
    [{ userAgent: SAFARI_26 }, 'auto'],
    [{ userAgent: IPHONE_CHROME }, 'auto'],
    [{ userAgent: IPAD_DESKTOP }, 'auto'],
    [{ userAgent: MAC_SAFARI_27 }, 'auto'],
    [{ userAgent: SAFARI_26, jsBackedModelStaging: undefined }, 'auto'],
    [{ userAgent: SAFARI_26, jsBackedModelStaging: false }, 'off'],
    [{ userAgent: SAFARI_26, jsBackedModelStaging: true }, 'on'],
    [{ userAgent: CHROMIUM }, 'off'],
    [{ userAgent: FIREFOX }, 'off'],
    [{ userAgent: CHROMIUM, jsBackedModelStaging: true }, 'on'],
    [{ userAgent: CHROMIUM, jsBackedModelStaging: false }, 'off'],
    [{}, 'off'],
    [undefined, 'off'],
  ]) {
    assert.equal(jsBackedStagingMode(config), expected, JSON.stringify(config));
  }
});

test('the default staging mode reads the page user agent when the config has none', () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  try {
    Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { userAgent: SAFARI_26 } });
    assert.equal(jsBackedStagingMode({}), 'auto');
    assert.equal(jsBackedStagingMode({ userAgent: CHROMIUM }), 'off');
    assert.equal(jsBackedStagingMode({ jsBackedModelStaging: false }), 'off');
  } finally {
    Object.defineProperty(globalThis, 'navigator', descriptor);
  }
});

test('default WebKit staging requires every staging export and FS API version 3', () => {
  assert.equal(supportsJsBackedStaging(createStagingHarness({ apiVersion: 3 }).core), true);
  assert.equal(supportsJsBackedStaging(createStagingHarness({ apiVersion: 4 }).core), true);
  assert.equal(supportsJsBackedStaging(createStagingHarness({ apiVersion: 2 }).core), false);
  assert.equal(supportsJsBackedStaging(createStagingHarness({ apiVersion: 3.5 }).core), false);
  assert.equal(supportsJsBackedStaging(createStagingHarness({ apiVersion: Infinity }).core), false);
  assert.equal(supportsJsBackedStaging(createStagingHarness({ includeJsModelFsExports: false }).core), false);
  const partial = createStagingHarness({ apiVersion: 4 }).core;
  delete partial._llamadart_webgpu_js_model_fs_release_mount_chunks;
  assert.equal(supportsJsBackedStaging(partial), false);
  assert.equal(supportsJsBackedStaging(null), false);
});

test('model loads follow the default staging mode', async (t) => {
  const cases = [
    { name: 'iPhone Safari, zero layers', harness: { userAgent: SAFARI_26 }, options: { nGpuLayers: 0 }, staged: true, heap: true },
    { name: 'iPhone Safari, capped layers', harness: { userAgent: SAFARI_26 }, options: { nGpuLayers: 99 }, staged: true, heap: true },
    { name: 'iPhone Safari, omitted layers', harness: { userAgent: SAFARI_26 }, options: {}, staged: true, heap: true },
    { name: 'iPhone Safari, negative layers', harness: { userAgent: SAFARI_26 }, options: { nGpuLayers: -5 }, staged: true, heap: true },
    { name: 'iPhone Safari, uncapped override', harness: { userAgent: SAFARI_26 }, options: { safariMaxGpuLayers: Infinity }, staged: true },
    { name: 'macOS Safari 27, omitted layers', harness: { userAgent: MAC_SAFARI_27 }, options: {}, staged: true },
    { name: 'iPhone Chrome', harness: { userAgent: IPHONE_CHROME }, options: { nGpuLayers: 0 }, staged: true, heap: true },
    { name: 'iPad desktop user agent', harness: { userAgent: IPAD_DESKTOP }, options: { nGpuLayers: 0 }, staged: true, heap: true },
    { name: 'macOS Safari 27, full offload', harness: { userAgent: MAC_SAFARI_27 }, options: { nGpuLayers: 99 }, staged: true },
    { name: 'explicit false on iPhone Safari', harness: { userAgent: SAFARI_26, jsBackedModelStaging: false }, options: { nGpuLayers: 0 } },
    { name: 'Chromium', harness: { userAgent: CHROMIUM }, options: { nGpuLayers: 0 } },
    { name: 'Firefox', harness: { userAgent: FIREFOX }, options: { nGpuLayers: 0 } },
    { name: 'no user agent', harness: {}, options: { nGpuLayers: 0 } },
    { name: 'older core on iPhone Safari', harness: { userAgent: SAFARI_26, includeJsModelFsExports: false }, options: { nGpuLayers: 0 }, unavailable: true },
    { name: 'FS API version 2 on iPhone Safari', harness: { userAgent: SAFARI_26, apiVersion: 2 }, options: { nGpuLayers: 0 }, unavailable: true },
    { name: 'fractional FS API version on iPhone Safari', harness: { userAgent: SAFARI_26, apiVersion: 3.5 }, options: { nGpuLayers: 0 }, unavailable: true },
    { name: 'infinite FS API version on iPhone Safari', harness: { userAgent: SAFARI_26, apiVersion: Infinity }, options: { nGpuLayers: 0 }, unavailable: true },
  ];
  for (const entry of cases) {
    await t.test(entry.name, async () => {
      const { runtime, fs, ccalls } = createStagingHarness({ apiVersion: 4, jsBackedModelStaging: null, ...entry.harness });
      runtime._gpuActive = true;
      await runtime.loadModelFromUrl('https://models.test/model.gguf', {
        ...entry.options, nCtx: 128, useCache: false, forceRemoteFetchBackend: false, safariGpuProbe: false,
      });

      const notes = runtime._runtimeNotes;
      const mounts = ccalls.filter(([name]) => name === 'llamadart_webgpu_mount_js_model_fs');
      const nativeLoad = ccalls.find(([name]) => name === 'llamadart_webgpu_load_model');
      assert.equal(mounts.length, entry.staged ? 1 : 0);
      assert.equal(notes.includes('model_staging_js_backed'), entry.staged === true);
      assert.equal(notes.includes('model_staging_heap_resident'), entry.heap === true);
      assert.equal(notes.includes('model_staging_js_backed_default_unavailable'), entry.unavailable === true);
      assert.equal(notes.includes('model_staging_js_backed_unsupported'), false);
      assert.equal(fs.dirs.has('/models'), !entry.staged);
      assert.equal(nativeLoad[1].startsWith(entry.staged ? '/jsmodels-1/' : '/models/'), true, nativeLoad[1]);
    });
  }
});

test('an explicit jsBackedModelStaging still rejects an unsupported core on WebKit', async () => {
  for (const unsupported of [
    { includeJsModelFsExports: false },
    { apiVersion: 3.5 },
    { apiVersion: Infinity },
  ]) {
    const { runtime } = createStagingHarness({
      jsBackedModelStaging: true, userAgent: SAFARI_26, ...unsupported,
    });
    await assert.rejects(
      runtime.loadModelFromUrl('https://models.test/model.gguf', {
        nCtx: 128, nGpuLayers: 0, useCache: false, forceRemoteFetchBackend: false,
      }),
      (error) => error instanceof ModelStagingError && error.code === 'MODEL_STAGING_UNSUPPORTED',
    );
    assert.equal(runtime._runtimeNotes.includes('model_staging_js_backed_default_unavailable'), false);
  }
});

test('multimodal projector loads follow the default staging mode', async () => {
  for (const [harnessOptions, expectedPath] of [
    [{ userAgent: SAFARI_26 }, '/jsmodels-1/mmproj.gguf'],
    [{ userAgent: SAFARI_26, jsBackedModelStaging: false }, '/mmproj/mmproj.gguf'],
    [{ userAgent: SAFARI_26, includeJsModelFsExports: false }, '/mmproj/mmproj.gguf'],
    [{ userAgent: CHROMIUM }, '/mmproj/mmproj.gguf'],
  ]) {
    const harness = createStagingHarness({ jsBackedModelStaging: null, ...harnessOptions });
    harness.runtime._modelBytes = 1024;
    harness.runtime._fetchWithTimeout = async () => new Response(new Uint8Array([5, 6, 7]), {
      status: 200,
      headers: { 'content-length': '3', 'content-encoding': 'identity' },
    });
    await harness.runtime.loadMultimodalProjector('https://models.test/mmproj.gguf');
    const load = harness.ccalls.find(([name]) => name === 'llamadart_webgpu_mmproj_load');
    assert.equal(load[1], expectedPath, JSON.stringify(harnessOptions));
    assert.equal(
      harness.runtime._runtimeNotes.includes('mmproj_staging_js_backed'),
      expectedPath.startsWith('/jsmodels-'),
    );
  }
});

test('the worker receives an explicit jsBackedModelStaging and resolves an unset one itself', async () => {
  for (const [config, forwarded, workerMode] of [
    [{ userAgent: SAFARI_26 }, undefined, 'auto'],
    [{ userAgent: SAFARI_26, jsBackedModelStaging: false }, false, 'off'],
    [{ userAgent: SAFARI_26, jsBackedModelStaging: true }, true, 'on'],
    [{ userAgent: CHROMIUM }, undefined, 'off'],
    [{ userAgent: CHROMIUM, jsBackedModelStaging: false }, false, 'off'],
  ]) {
    const bridge = new LlamaWebGpuBridge({ disableWorker: true, ...config });
    const workerConfig = bridge._workerConfig();
    assert.equal(workerConfig.jsBackedModelStaging, forwarded, JSON.stringify(config));
    assert.equal(Object.hasOwn(workerConfig, 'jsBackedModelStaging'), forwarded !== undefined);
    assert.equal(jsBackedStagingMode(workerConfig), workerMode, JSON.stringify(config));
    await bridge.dispose();
  }
});

test('staging errors keep their type whenever staging can be active', async () => {
  const stagingError = new ModelStagingError('MODEL_STAGING_ALLOCATION_FAILED', 'out of memory');
  for (const [config, typed] of [
    [{ userAgent: SAFARI_26 }, true],
    [{ userAgent: SAFARI_26, jsBackedModelStaging: false }, false],
    [{ userAgent: CHROMIUM }, false],
    [{ userAgent: CHROMIUM, jsBackedModelStaging: true }, true],
  ]) {
    await withStubWorkerEnvironment(async ({ workers }) => {
      const bridge = createRealWorkerBridge(config);
      assert.equal(bridge._isNonRecoverableStagingError(stagingError), typed, JSON.stringify(config));
      const proxy = bridge._workerProxy;
      const driver = workerDriver(proxy, workers[0]);
      driver.ready();
      const pendingLoad = bridge.loadModelFromUrl('https://models.test/x.gguf', { nCtx: 128, nGpuLayers: 0 });
      await driver.settle();
      proxy._worker.onmessage({
        data: {
          id: driver.calls('loadModelFromUrl')[0].id,
          type: 'error',
          modelStagingErrorCode: 'MODEL_STAGING_ALLOCATION_FAILED',
          message: 'out of memory',
        },
      });
      const error = await pendingLoad.then(() => assert.fail('must reject'), (e) => e);
      assert.equal(error instanceof ModelStagingError, typed, JSON.stringify(config));
      await bridge.dispose();
    });
  }
});
