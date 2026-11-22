import assert from 'node:assert/strict';
import test from 'node:test';

import { createModelTransferLengthError } from '../../js/src/internal/download.ts';
import { REQUIRED_JS_MODEL_STAGING_EXPORTS } from '../../js/src/internal/model_staging.ts';
import { LlamaWebGpuBridge, ModelStagingError } from '../../js/src/llama_webgpu_bridge.js';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';
import { createRealWorkerBridge, withStubWorkerEnvironment } from './bridge_operation_queue_fixtures.mjs';

const MODEL = 'https://models.test/model.gguf';
const PROJECTOR = 'https://models.test/mmproj.gguf';
const parts = [{ type: 'image', url: 'https://models.test/image.png' }];
const transfer = Object.assign(
  createModelTransferLengthError('Model response ended before its declared byte length.'),
  { llamadartLoadedBytes: 3, llamadartFilePath: '/model/local.gguf' },
);
const stagingErrors = [
  'MODEL_STAGING_ALLOCATION_FAILED',
  'MODEL_TENSOR_SIZE_LIMIT_EXCEEDED',
  'MODEL_STAGING_MOUNT_FAILED',
  'MODEL_STAGING_UNSUPPORTED',
].map((code) => new ModelStagingError(code, 'Model staging failed.'));
const mountError = stagingErrors.find(({ code }) => code === 'MODEL_STAGING_MOUNT_FAILED');
const timeout = new Error('Worker request timed out');

async function withConnectedWorkers(t, staging, run, { coreModuleFactory, onReply } = {}) {
  return withStubWorkerEnvironment(async ({ workers }) => {
    const bridge = createRealWorkerBridge({ jsBackedModelStaging: staging, logLevel: 0 });
    const oldSelf = globalThis.self;
    const connectHost = async (worker) => {
      const host = {
        postMessage(reply) {
          onReply?.(reply);
          worker.onmessage({ data: structuredClone(reply) });
        },
      };
      globalThis.self = host;
      installBridgeWorkerHost();
      const receive = (request) => {
        globalThis.self = host;
        const data = structuredClone(request);
        if (data.type === 'init' && coreModuleFactory) {
          data.config.coreModuleFactory = coreModuleFactory;
        }
        return host.onmessage({ data });
      };
      worker.postMessage = (request) => {
        worker.posted.push(request);
        return receive(request);
      };
      await receive(worker.posted[0]);
    };
    const replace = bridge._replaceWorkerProxyForMultimodalCpuMode.bind(bridge);
    t.mock.method(bridge, '_replaceWorkerProxyForMultimodalCpuMode', async () => {
      await replace();
      await connectHost(workers.at(-1));
    });
    try {
      await connectHost(workers[0]);
      return await run(bridge, workers);
    } finally {
      try {
        await bridge.dispose();
      } finally {
        if (oldSelf === undefined) delete globalThis.self;
        else globalThis.self = oldSelf;
      }
    }
  });
}

async function runWorkerCase(t, nGpuLayers, {
  site = 'setup',
  restoreMethod = 'loadMultimodalProjector',
  restoreErrors = [],
  requestError,
  staging = true,
  warmup = false,
  directRestoreSucceeds = false,
} = {}) {
  const runtime = LlamaWebGpuBridgeRuntime.prototype;
  let active = false;
  let restoreAttempts = 0;
  let completionCalls = 0;
  let usageCalls = 0;
  let bridge;
  const load = async function () {
    if (active) {
      const attempt = restoreAttempts++;
      const failure = restoreErrors[Math.min(attempt, restoreErrors.length - 1)];
      if (failure && !(directRestoreSucceeds && this === bridge._runtime)) throw failure;
    }
    return 1;
  };
  t.mock.method(runtime, 'loadModelFromUrl', restoreMethod === 'loadModelFromUrl' ? load : async () => 1);
  t.mock.method(runtime, 'loadMultimodalProjector', restoreMethod === 'loadMultimodalProjector' ? load : async () => 1);
  t.mock.method(runtime, 'createCompletion', async function () {
    completionCalls += 1;
    if (requestError && this !== bridge._runtime) throw requestError;
    return site === 'empty response' && completionCalls === 1 ? '' : 'generated';
  });

  return withConnectedWorkers(t, staging, async (activeBridge, workers) => {
    bridge = activeBridge;
    const workerErrors = [];
    const callWorker = bridge._callWorker.bind(bridge);
    t.mock.method(bridge, '_callWorker', async (...args) => {
      try {
        return await callWorker(...args);
      } catch (error) {
        workerErrors.push(error);
        throw error;
      }
    });
    await bridge.loadModelFromUrl(MODEL, { nGpuLayers });
    await bridge.loadMultimodalProjector(PROJECTOR);
    assert.equal(bridge._isCpuModelMode(), nGpuLayers === 0);
    if (site === 'setup') await bridge._replaceWorkerProxyForMultimodalCpuMode();
    if (site === 'request restore') bridge._workerModelMissing = true;
    const workerCalls = (method) => workers.flatMap((worker) => worker.posted)
      .filter((request) => request.method === method).length;
    const workersBefore = workers.length;
    const workerRestoresBefore = workerCalls(restoreMethod);
    active = true;

    const outcome = await bridge.createCompletion('Describe the image.', {
      parts,
      warmup,
      onUsage: () => { usageCalls += 1; },
    }).then(
      (value) => ({ value }),
      (error) => ({ error }),
    );
    return {
      ...outcome,
      workerErrors,
      directRuntime: bridge._runtime !== null,
      restarts: workers.length - workersBefore,
      restoreAttempts,
      workerRestores: workerCalls(restoreMethod) - workerRestoresBefore,
      completionCalls,
      workerCompletions: workerCalls('createCompletion'),
      workerKept: bridge._workerProxy?._worker === workers.at(-1) && workers.at(-1).terminated === 0,
      usageCalls,
    };
  });
}

function assertWorkerAttempts(result, restoreAttempts, restarts, completionCalls, expectedError) {
  const { value, error, workerErrors, usageCalls, ...attempts } = result;
  const errorDetails = {
    constructor: error?.constructor,
    name: error?.name,
    code: error?.code,
    message: expectedError ? error?.message : undefined,
    llamadartLoadedBytes: error?.llamadartLoadedBytes,
    llamadartFilePath: error?.llamadartFilePath,
  };
  assert.deepEqual({ ...attempts, error: errorDetails }, {
    restoreAttempts,
    workerRestores: restoreAttempts,
    restarts,
    completionCalls,
    workerCompletions: completionCalls,
    workerKept: true,
    directRuntime: false,
    error: {
      constructor: expectedError?.constructor,
      name: expectedError?.name,
      code: expectedError?.code,
      message: expectedError?.message,
      llamadartLoadedBytes: undefined,
      llamadartFilePath: undefined,
    },
  });
}

function assertGenericFailure(result, nGpuLayers, cpuMessage, workerRestores, restarts, workerCompletions) {
  if (nGpuLayers === 0) {
    assert.equal(result.error?.constructor, Error);
    assert.match(result.error.message, cpuMessage);
    assertWorkerAttempts(result, workerRestores, restarts, workerCompletions, result.error);
  } else {
    assert.equal(result.value, 'generated');
    assert.equal(result.directRuntime, true);
    assert.equal(result.workerKept, false);
    assert.equal(result.restoreAttempts, workerRestores + 1);
    assert.equal(result.workerRestores, workerRestores);
    assert.equal(result.restarts, restarts);
    assert.equal(result.completionCalls, workerCompletions + 1);
    assert.equal(result.workerCompletions, workerCompletions);
  }
}

for (const [nGpuLayers, mode] of [[0, 'CPU'], [1, 'non-CPU']]) {
  for (const [row, restoreMethod] of [[1, 'loadMultimodalProjector'], [2, 'loadModelFromUrl']]) {
    test(`#${row} ${mode}: setup preserves transfer error from ${restoreMethod}`, async (t) => {
      const result = await runWorkerCase(t, nGpuLayers, { restoreMethod, restoreErrors: [transfer] });
      assertWorkerAttempts(result, 1, 0, 0, transfer);
    });
  }

  for (const error of stagingErrors) {
    test(`#3 ${mode}: setup preserves ${error.code}`, async (t) => {
      const result = await runWorkerCase(t, nGpuLayers, { restoreErrors: [error] });
      assertWorkerAttempts(result, 1, 0, 0, error);
    });
  }

  test(`#4 ${mode}: setup preserves transfer error after timeout retry`, async (t) => {
    const result = await runWorkerCase(t, nGpuLayers, { restoreErrors: [timeout, transfer] });
    assertWorkerAttempts(result, 2, 1, 0, transfer);
  });

  for (const [row, error] of [[5, transfer], ...stagingErrors.map((error) => [6, error])]) {
    test(`#${row} ${mode}: empty-response restore preserves ${error.code}`, async (t) => {
      const result = await runWorkerCase(t, nGpuLayers, { site: 'empty response', restoreErrors: [error] });
      assertWorkerAttempts(result, 1, 1, 1, error);
    });
  }

  for (const error of [transfer, mountError]) {
    test(`#7 ${mode}: request restore preserves ${error.code}`, async (t) => {
      const result = await runWorkerCase(t, nGpuLayers, { site: 'request restore', restoreErrors: [error] });
      assertWorkerAttempts(result, 1, 0, 0, error);
    });

    test(`#8 ${mode}: completion request preserves ${error.code}`, async (t) => {
      const result = await runWorkerCase(t, nGpuLayers, { site: 'request', requestError: error });
      assertWorkerAttempts(result, 0, 0, 1, error);
    });
  }

  for (const site of ['setup', 'empty response']) {
    test(`#10 ${mode}: ${site} recovers from a transient timeout`, async (t) => {
      const result = await runWorkerCase(t, nGpuLayers, { site, restoreErrors: [timeout, null] });
      const empty = site === 'empty response' ? 1 : 0;
      assert.equal(result.value, 'generated');
      assertWorkerAttempts(result, 2, 1 + empty, 1 + empty);
    });
  }

  test(`#10 ${mode}: setup handles a persistent timeout`, async (t) => {
    const result = await runWorkerCase(t, nGpuLayers, { restoreErrors: [timeout], directRestoreSucceeds: true });
    assertGenericFailure(result, nGpuLayers, /CPU multimodal worker setup failed/, 2, 1, 0);
  });

  test(`#10 ${mode}: request handles a timeout`, async (t) => {
    const result = await runWorkerCase(t, nGpuLayers, { site: 'request', requestError: timeout });
    assertGenericFailure(result, nGpuLayers, /CPU multimodal request failed/, 0, 0, 1);
  });

  for (const site of ['setup', 'empty response']) {
    const empty = site === 'empty response' ? 1 : 0;
    test(`#11 ${mode}: ${site} preserves transfer error with staging off`, async (t) => {
      const result = await runWorkerCase(t, nGpuLayers, { site, staging: false, restoreErrors: [transfer] });
      assertWorkerAttempts(result, 1, empty, empty, transfer);
    });

    for (const transient of [true, false]) {
      test(`#11 ${mode}: ${site} handles ${transient ? 'transient' : 'persistent'} staging error with staging off`, async (t) => {
        const result = await runWorkerCase(t, nGpuLayers, {
          site,
          staging: false,
          restoreErrors: transient ? [mountError, null] : [mountError],
          directRestoreSucceeds: true,
        });
        assert.ok(result.workerErrors.length > 0);
        for (const error of result.workerErrors) {
          assert.equal(error.constructor, Error);
          assert.equal(error.name, 'Error');
          assert.equal(error.code, undefined);
          assert.equal(error.message, mountError.message);
        }
        if (transient) {
          assert.equal(result.value, 'generated');
          assertWorkerAttempts(result, 2, 1 + empty, 1 + empty);
        } else {
          assertGenericFailure(result, nGpuLayers, /CPU multimodal worker setup failed/, 2 + empty * 2, 1 + empty * 2, empty);
        }
      });
    }
  }

  for (const error of [transfer, mountError]) {
    for (const site of ['setup', 'request restore', 'request']) {
      test(`#12 ${mode}: warmup skips ${error.code} from ${site}`, async (t) => {
        const result = await runWorkerCase(t, nGpuLayers, {
          site,
          warmup: true,
          restoreErrors: site === 'request' ? [] : [error],
          requestError: site === 'request' ? error : undefined,
        });
        assert.equal(result.value, '');
        assert.equal(result.usageCalls, 0);
        assertWorkerAttempts(result, site === 'request' ? 0 : 1, 0, site === 'request' ? 1 : 0);
      });
    }
  }

  for (const transient of [true, false]) {
    test(`#12 ${mode}: warmup handles a ${transient ? 'transient' : 'persistent'} setup timeout`, async (t) => {
      const result = await runWorkerCase(t, nGpuLayers, {
        warmup: true,
        restoreErrors: transient ? [timeout, null] : [timeout],
      });
      assert.equal(result.value, transient ? 'generated' : '');
      assert.equal(result.usageCalls, 0);
      assertWorkerAttempts(result, 2, 1, transient ? 1 : 0);
    });
  }

  test(`#12 ${mode}: warmup skips a request timeout`, async (t) => {
    const result = await runWorkerCase(t, nGpuLayers, { site: 'request', requestError: timeout, warmup: true });
    assert.equal(result.value, '');
    assert.equal(result.usageCalls, 0);
    assertWorkerAttempts(result, 0, 0, 1);
  });
}

for (const error of [transfer, ...stagingErrors, timeout]) {
  for (const warmup of [false, true]) {
    test(`#9 direct: ${warmup ? 'warmup' : 'completion'} preserves ${error.code ?? 'generic error'} identity`, async (t) => {
      const completion = t.mock.method(LlamaWebGpuBridgeRuntime.prototype, 'createCompletion', async () => {
        throw error;
      });
      const bridge = new LlamaWebGpuBridge({ disableWorker: true });
      try {
        await assert.rejects(bridge.createCompletion('Describe the image.', { parts, warmup }), (actual) => {
          assert.equal(actual, error);
          assert.equal(actual.name, error.name);
          assert.equal(actual.code, error.code);
          return true;
        });
        assert.equal(completion.mock.callCount(), 1);
      } finally {
        await bridge.dispose();
      }
    });
  }
}

async function runCoreRetirementCase(t, nGpuLayers, { truncateProjector, initialSetup = false, warmup = false }) {
  const runtime = LlamaWebGpuBridgeRuntime.prototype;
  const layerHistory = [];
  const wire = [];
  let projectorGets = 0;
  let completionCalls = 0;
  let nativeTraps = 0;

  t.mock.method(runtime, 'loadModelFromUrl', async function (_url, options) {
    await this._ensureCore();
    this._modelBytes = 1;
    layerHistory.push(options.nGpuLayers);
    return 1;
  });
  t.mock.method(runtime, 'createCompletion', async function () {
    if (++completionCalls === 1 && !initialSetup) {
      if (truncateProjector) return '';
      this._core.ccall('llamadart_webgpu_begin_generation', 'number', [], []);
    }
    return 'generated';
  });
  if (initialSetup) {
    t.mock.method(runtime, 'tokenize', async function () {
      this._core.ccall('llamadart_webgpu_tokenize_to_json', 'string', [], []);
    });
  }

  t.mock.method(globalThis, 'fetch', async (url) => {
    assert.equal(String(url), PROJECTOR);
    const truncated = ++projectorGets === 2 && truncateProjector;
    return new Response(new Uint8Array(truncated ? 3 : 4), {
      // The stream length check applies to large identity-encoded responses.
      headers: { 'content-length': truncated ? '1073741824' : '4', 'content-encoding': 'identity' },
    });
  });

  const coreModuleFactory = async () => ({
    ...Object.fromEntries(REQUIRED_JS_MODEL_STAGING_EXPORTS.map((name) => [name, () => 0])),
    FS: {
      lstat() { throw Object.assign(new Error('No such directory'), { errno: 44 }); },
      unlink() {},
    },
    llamadartJsModelFsWriteDirect: (_handle, chunk) => chunk.byteLength,
    ccall(name, returnType) {
      if (name === 'llamadart_webgpu_js_model_fs_api_version') return 3;
      if (name === 'llamadart_webgpu_js_model_fs_mount_id' || name === 'llamadart_webgpu_js_model_fs_open_direct') return 1;
      if (
        (name === 'llamadart_webgpu_unmount_js_model_fs' && truncateProjector && projectorGets === 2)
        || (name === 'llamadart_webgpu_mmproj_load' && initialSetup && !truncateProjector && projectorGets === 2)
        || name === 'llamadart_webgpu_begin_generation'
        || name === 'llamadart_webgpu_tokenize_to_json'
      ) {
        nativeTraps += 1;
        throw new WebAssembly.RuntimeError('memory access out of bounds');
      }
      return returnType === 'string' ? '{}' : 0;
    },
  });
  const onReply = ({ type, errorCode }) => {
    if (type === 'core-retired' || type === 'error') wire.push([type, errorCode]);
  };

  return withConnectedWorkers(t, true, async (bridge, workers) => {
    const chooseCpuRerun = t.mock.method(bridge, '_rerunOnCpuAfterCoreAbort');
    await bridge.loadModelFromUrl(MODEL, { nGpuLayers });
    await bridge.loadMultimodalProjector(PROJECTOR);
    if (initialSetup) {
      await assert.rejects(bridge.tokenize('test'), /memory access out of bounds/);
      assert.equal(bridge._coreRetirements, 1);
      assert.equal(workers.length, 2);
      assert.equal(workers[0].terminated, 1);
    }
    const outcome = await bridge.createCompletion('Describe the image.', { parts, warmup }).then(
      (value) => ({ value }),
      ({ name, code, message }) => ({ error: { name, code, message } }),
    );
    const retirements = initialSetup ? 2 : 1;
    assert.equal(nativeTraps, retirements);
    assert.equal(bridge._coreRetirements, retirements);
    assert.equal(bridge._runtime, null);
    assert.equal(bridge._workerProxy._worker, workers.at(-1));
    assert.deepEqual(workers.map((worker) => worker.terminated), initialSetup || truncateProjector ? [1, 1, 0] : [1, 0]);
    return {
      ...outcome,
      wire,
      projectorGets,
      completionCalls,
      layerHistory,
      rememberedGpuLayers: bridge._loadedModelOptions.nGpuLayers,
      cpuRerunDecisions: chooseCpuRerun.mock.callCount(),
    };
  }, { coreModuleFactory, onReply });
}

for (const [nGpuLayers, mode] of [[0, 'CPU'], [1, 'non-CPU']]) {
  for (const warmup of [false, true]) {
    test(`core retirement: ${mode} ${warmup ? 'warmup skips' : 'surfaces'} a typed projector transfer error during initial setup`, async (t) => {
      const outcome = warmup
        ? { value: '' }
        : { error: { name: transfer.name, code: transfer.code, message: transfer.message } };
      assert.deepEqual(await runCoreRetirementCase(t, nGpuLayers, {
        truncateProjector: true,
        initialSetup: true,
        warmup,
      }), {
        ...outcome,
        wire: [['core-retired', undefined], ['error', undefined], ['core-retired', undefined], ['error', transfer.code]],
        projectorGets: 2,
        completionCalls: 0,
        layerHistory: [nGpuLayers, nGpuLayers],
        rememberedGpuLayers: nGpuLayers,
        cpuRerunDecisions: 0,
      });
    });
  }

  test(`core retirement: ${mode} preserves generic initial setup trap recovery`, async (t) => {
    const cpu = nGpuLayers === 0;
    const outcome = cpu
      ? { error: { name: 'Error', code: undefined, message: 'memory access out of bounds' } }
      : { value: 'generated' };
    assert.deepEqual(await runCoreRetirementCase(t, nGpuLayers, {
      truncateProjector: false,
      initialSetup: true,
    }), {
      ...outcome,
      wire: [['core-retired', undefined], ['error', undefined], ['core-retired', undefined], ['error', undefined]],
      projectorGets: cpu ? 2 : 3,
      completionCalls: cpu ? 0 : 1,
      layerHistory: cpu ? [0, 0] : [nGpuLayers, nGpuLayers, 0],
      rememberedGpuLayers: 0,
      cpuRerunDecisions: 1,
    });
  });

  test(`core retirement: ${mode} surfaces a typed projector transfer error after an empty response`, async (t) => {
    assert.deepEqual(await runCoreRetirementCase(t, nGpuLayers, { truncateProjector: true }), {
      error: { name: transfer.name, code: transfer.code, message: transfer.message },
      wire: [['core-retired', undefined], ['error', transfer.code]],
      projectorGets: 2,
      completionCalls: 1,
      layerHistory: [nGpuLayers, nGpuLayers],
      rememberedGpuLayers: nGpuLayers,
      cpuRerunDecisions: 0,
    });
  });

  test(`core retirement: ${mode} preserves generic generation trap recovery`, async (t) => {
    const cpu = nGpuLayers === 0;
    const outcome = cpu
      ? { error: { name: 'Error', code: undefined, message: 'memory access out of bounds' } }
      : { value: 'generated' };
    assert.deepEqual(await runCoreRetirementCase(t, nGpuLayers, { truncateProjector: false }), {
      ...outcome,
      wire: [['core-retired', undefined], ['error', undefined]],
      projectorGets: cpu ? 1 : 2,
      completionCalls: cpu ? 1 : 2,
      layerHistory: cpu ? [0] : [nGpuLayers, 0],
      rememberedGpuLayers: 0,
      cpuRerunDecisions: 1,
    });
  });
}
