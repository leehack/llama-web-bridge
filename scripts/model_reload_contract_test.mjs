import assert from 'node:assert/strict';

import { LlamaWebGpuBridge } from '../js/src/llama_webgpu_bridge.js';

// Mirrors the generated WASMFS glue: every failure is an ErrnoError whose
// message is the bare "FS error", and analyzePath() reads the entry it finds.
class ErrnoError extends Error {
  constructor(errno) {
    super('FS error');
    this.name = 'ErrnoError';
    this.errno = errno;
  }
}

const EEXIST = 20;
const EISDIR = 31;
const ENOENT = 44;

function createWasmFs() {
  const dirs = new Set(['/']);
  const files = new Map();
  const analyzed = [];
  const parentOf = (path) => {
    const slash = path.lastIndexOf('/');
    return slash > 0 ? path.slice(0, slash) : '/';
  };
  const readFile = (path) => {
    if (dirs.has(path)) {
      throw new ErrnoError(EISDIR);
    }
    if (!files.has(path)) {
      throw new ErrnoError(ENOENT);
    }
    return files.get(path).slice();
  };
  const fs = {
    dirs,
    files,
    analyzed,
    mkdir(path) {
      if (dirs.has(path) || files.has(path)) {
        throw new ErrnoError(EEXIST);
      }
      if (!dirs.has(parentOf(path))) {
        throw new ErrnoError(ENOENT);
      }
      dirs.add(path);
    },
    readdir(path) {
      if (!dirs.has(path)) {
        throw new Error('No such directory');
      }
      const prefix = path === '/' ? '/' : `${path}/`;
      return ['.', '..', ...[...dirs, ...files.keys()]
        .filter((entry) => entry !== path && entry.startsWith(prefix))
        .map((entry) => entry.slice(prefix.length))
        .filter((entry) => !entry.includes('/'))];
    },
    analyzePath(path) {
      analyzed.push(path);
      const exists = dirs.has(path) || files.has(path);
      return { exists, object: { contents: exists ? readFile(path) : null } };
    },
    readFile,
    writeFile(path, bytes) {
      if (!dirs.has(parentOf(path))) {
        throw new ErrnoError(ENOENT);
      }
      files.set(path, new Uint8Array(bytes));
    },
    open(path, flags) {
      if (!dirs.has(parentOf(path))) {
        throw new ErrnoError(ENOENT);
      }
      if (flags === 'w' || !files.has(path)) {
        files.set(path, new Uint8Array());
      }
      return { path };
    },
    write(stream, bytes, offset = 0, length = bytes.length) {
      const previous = files.get(stream.path) || new Uint8Array();
      const next = new Uint8Array(previous.length + length);
      next.set(previous);
      next.set(bytes.subarray(offset, offset + length), previous.length);
      files.set(stream.path, next);
      return length;
    },
    close() {},
    unlink(path) {
      if (dirs.has(path)) {
        throw new ErrnoError(EISDIR);
      }
      if (!files.delete(path)) {
        throw new ErrnoError(ENOENT);
      }
    },
  };
  return fs;
}

function createCore(fs, { freeModelRc = 0 } = {}) {
  const core = {
    FS: fs,
    trace: [],
    model: null,
    projector: null,
    lastError: '',
    ccall(name, _returnType, _argTypes, args = []) {
      switch (name) {
        case 'llamadart_webgpu_load_model': {
          const [modelPath] = args;
          // Native load reads the file it was given, then the bridge releases it.
          fs.readFile(modelPath);
          core.trace.push(['load', modelPath]);
          core.model = modelPath;
          core.projector = null;
          return 0;
        }
        case 'llamadart_webgpu_free_model':
          core.trace.push(['free', core.model]);
          if (freeModelRc !== 0) {
            core.lastError = 'Model cannot be released during active generation or text-to-speech synthesis';
            return freeModelRc;
          }
          core.model = null;
          core.projector = null;
          return 0;
        case 'llamadart_webgpu_mmproj_load': {
          const [projectorPath] = args;
          fs.readFile(projectorPath);
          core.trace.push(['mmproj', projectorPath]);
          core.projector = projectorPath;
          return 0;
        }
        case 'llamadart_webgpu_mmproj_supports_vision':
          return core.projector ? 1 : 0;
        case 'llamadart_webgpu_mmproj_supports_audio':
          return 0;
        case 'llamadart_webgpu_get_context_size':
          return 128;
        case 'llamadart_webgpu_last_error':
          return core.lastError;
        default:
          throw new Error(`Unexpected ccall: ${name}`);
      }
    },
  };
  return core;
}

function streamResponse(url, bytes) {
  let sent = false;
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    url,
    headers: new Headers({ 'content-length': String(bytes.length) }),
    body: {
      getReader() {
        return {
          async read() {
            if (sent) {
              return { done: true, value: undefined };
            }
            sent = true;
            return { done: false, value: bytes };
          },
          async cancel() {},
        };
      },
    },
  };
}

function createRuntime(options = {}) {
  const fs = createWasmFs();
  const core = createCore(fs, options);
  const bridge = new LlamaWebGpuBridge({ disableWorker: true });
  const runtime = bridge._runtime;
  runtime._core = core;
  runtime._coreVariant = 'wasm32';
  runtime._probeBackends = async () => false;
  runtime._ensureCore = async () => core;
  runtime._coreSupportsPthreads = () => false;
  runtime._syncThreadPoolSizeHintFromCore = () => {};
  runtime._resolveNativeLoadOptions = () => {};
  runtime._emitSuppressedWarmupWarningSummaryIfNeeded = () => {};
  runtime._getCachedModelResponse = async (url) => {
    core.trace.push(['fetch', url]);
    if (url.includes('missing')) {
      return { ok: false, status: 404, statusText: 'Not Found', url, headers: new Headers() };
    }
    return streamResponse(url, new Uint8Array([1, 2, 3, 4]));
  };
  runtime._fetchWithTimeout = async (url) => {
    core.trace.push(['fetch', url]);
    return streamResponse(url, new Uint8Array([5, 6]));
  };
  return { bridge, runtime, core, fs };
}

const loadOptions = { nGpuLayers: 0, useCache: false, forceRemoteFetchBackend: false };

const cases = [
  ['a second load of the same URL replaces the model in place', async () => {
    const { runtime, core, fs } = createRuntime();

    assert.equal(await runtime.loadModelFromUrl('https://models.test/a.gguf', loadOptions), 1);
    assert.equal(await runtime.loadModelFromUrl('https://models.test/a.gguf', loadOptions), 1);

    assert.deepEqual(core.trace, [
      ['fetch', 'https://models.test/a.gguf'],
      ['load', '/models/a.gguf'],
      ['free', '/models/a.gguf'],
      ['fetch', 'https://models.test/a.gguf'],
      ['load', '/models/a.gguf'],
    ]);
    assert.ok(runtime._runtimeNotes.includes('previous_model_released'));
    assert.ok(runtime._modelBytes > 0, 'the replacement model must be loaded');
    assert.deepEqual(fs.readdir('/models'), ['.', '..'], 'loaded model files must be released');
    assert.deepEqual(fs.analyzed, [], 'WASMFS analyzePath reads whole entries and must not be used');
  }],

  ['switching models frees the previous model and projector before the download', async () => {
    const { runtime, core, fs } = createRuntime();

    await runtime.loadModelFromUrl('https://models.test/a.gguf', loadOptions);
    await runtime.loadMultimodalProjector('https://models.test/mmproj-a.gguf');
    assert.ok(fs.files.has('/mmproj/mmproj-a.gguf'));
    assert.equal(runtime.supportsVision(), true);
    fs.mkdir('/media');
    fs.writeFile('/media/staged.bin', new Uint8Array([9]));
    runtime._stagedMediaPaths = ['/media/staged.bin'];

    await runtime.loadModelFromUrl('https://models.test/b.gguf', loadOptions);

    assert.deepEqual(core.trace.slice(3), [
      ['mmproj', '/mmproj/mmproj-a.gguf'],
      ['free', '/models/a.gguf'],
      ['fetch', 'https://models.test/b.gguf'],
      ['load', '/models/b.gguf'],
    ]);
    assert.equal(fs.files.size, 0, 'model, projector and staged media files must be removed');
    assert.equal(runtime._mmProjPath, null);
    assert.equal(runtime.supportsVision(), false);

    await runtime.loadMultimodalProjector('https://models.test/mmproj-b.gguf');
    assert.deepEqual(fs.readdir('/mmproj'), ['.', '..', 'mmproj-b.gguf'], 'the projector directory must be reusable');
    assert.deepEqual(fs.analyzed, []);
  }],

  ['a failed replacement leaves no model and no files behind', async () => {
    const { runtime, core, fs } = createRuntime();

    await runtime.loadModelFromUrl('https://models.test/a.gguf', loadOptions);
    await assert.rejects(
      runtime.loadModelFromUrl('https://models.test/missing.gguf', loadOptions),
      /Failed to fetch model shard: 404/,
    );

    assert.deepEqual(core.trace.slice(2), [
      ['free', '/models/a.gguf'],
      ['fetch', 'https://models.test/missing.gguf'],
    ]);
    assert.equal(core.model, null);
    assert.equal(runtime._modelBytes, 0);
    assert.equal(fs.files.size, 0);

    assert.equal(await runtime.loadModelFromUrl('https://models.test/b.gguf', loadOptions), 1);
    assert.equal(core.model, '/models/b.gguf');
  }],

  ['a refused release keeps the current model and skips the download', async () => {
    const { runtime, core } = createRuntime({ freeModelRc: -1 });

    await runtime.loadModelFromUrl('https://models.test/a.gguf', { ...loadOptions, nCtx: 96 });
    await runtime.loadMultimodalProjector('https://models.test/mmproj-a.gguf');
    const before = {
      modelBytes: runtime._modelBytes,
      nCtx: runtime._nCtx,
      nGpuLayers: runtime._nGpuLayers,
      loadedModelUrl: runtime._loadedModelUrl,
      projectorSource: runtime._mmProjSourceUrl,
    };
    await assert.rejects(
      runtime.loadModelFromUrl('https://models.test/b.gguf', { ...loadOptions, nCtx: 32, nGpuLayers: 99 }),
      /Failed to release the loaded model.*active generation/,
    );

    assert.deepEqual(core.trace.slice(4), [['free', '/models/a.gguf']]);
    assert.equal(core.model, '/models/a.gguf');
    assert.deepEqual({
      modelBytes: runtime._modelBytes,
      nCtx: runtime._nCtx,
      nGpuLayers: runtime._nGpuLayers,
      loadedModelUrl: runtime._loadedModelUrl,
      projectorSource: runtime._mmProjSourceUrl,
    }, before, 'a refused release must not apply the rejected load options');
  }],

  ['a fetch-backed replacement also frees the previous model first', async () => {
    const { runtime, core, fs } = createRuntime();

    await runtime.loadModelFromUrl('https://models.test/a.gguf', loadOptions);
    runtime._tryLoadModelFromRemoteFetchBackend = async (_core, modelUrl) => {
      core.trace.push(['remote-load', modelUrl]);
      runtime._modelBytes = 4;
      return { loaded: true };
    };
    await runtime.loadModelFromUrl('https://models.test/large.gguf', loadOptions);

    assert.deepEqual(core.trace.slice(2), [
      ['free', '/models/a.gguf'],
      ['remote-load', 'https://models.test/large.gguf'],
    ]);
    assert.equal(fs.files.size, 0);
  }],

  ['the facade forgets a model that a failed replacement released', async () => {
    const { bridge, runtime } = createRuntime();

    await bridge.loadModelFromUrl('https://models.test/a.gguf', loadOptions);
    await bridge.loadMultimodalProjector('https://models.test/mmproj-a.gguf');
    await assert.rejects(
      bridge.loadModelFromUrl('https://models.test/missing.gguf', loadOptions),
      /Failed to fetch model shard: 404/,
    );

    assert.equal(runtime._modelBytes, 0);
    assert.equal(bridge._loadedModelUrl, null, 'recovery must not replay the released model');
    assert.equal(bridge._loadedMmProjUrl, null);
  }],

  ['the facade keeps a model whose release was refused', async () => {
    const { bridge } = createRuntime({ freeModelRc: -1 });

    await bridge.loadModelFromUrl('https://models.test/a.gguf', loadOptions);
    await bridge.loadMultimodalProjector('https://models.test/mmproj-a.gguf');
    await assert.rejects(
      bridge.loadModelFromUrl('https://models.test/b.gguf', loadOptions),
      /Failed to release the loaded model/,
    );

    assert.equal(bridge._loadedModelUrl, 'https://models.test/a.gguf');
    assert.equal(bridge._loadedMmProjUrl, 'https://models.test/mmproj-a.gguf');
  }],

  ['state snapshots reuse an existing /states directory', async () => {
    const { runtime, fs } = createRuntime();

    runtime._nextStateTempPath();
    runtime._nextStateTempPath();

    assert.ok(fs.dirs.has('/states'));
    assert.deepEqual(fs.analyzed, []);
  }],
];

for (const [name, run] of cases) {
  try {
    await run();
  } catch (error) {
    error.message = `${name}: ${error.message}`;
    throw error;
  }
}

console.log(`Model reload contract tests passed (${cases.length} cases)`);
