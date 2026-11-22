import assert from 'node:assert/strict';
import test from 'node:test';

import { createModelTransferLengthError } from '../../js/src/internal/download.ts';
import { LlamaWebGpuBridge, ModelStagingError } from '../../js/src/llama_webgpu_bridge.js';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';
import { createRealWorkerBridge, withStubWorkerEnvironment } from './bridge_operation_queue_fixtures.mjs';

const MODEL = 'https://models.test/model.gguf';
const PROJECTOR = 'https://models.test/mmproj.gguf';
const transferCode = 'MODEL_TRANSFER_LENGTH_ERROR';
const transferMessage = 'Model response ended before its declared byte length.';
const transferError = (message = transferMessage) => Object.assign(createModelTransferLengthError(message), {
  llamadartLoadedBytes: 3,
  llamadartFilePath: '/mmproj/local.gguf',
});
const stagingCodes = [
  'MODEL_STAGING_ALLOCATION_FAILED',
  'MODEL_TENSOR_SIZE_LIMIT_EXCEEDED',
  'MODEL_STAGING_MOUNT_FAILED',
  'MODEL_STAGING_UNSUPPORTED',
];

async function withWorkerProjector(t, nGpuLayers, loadProjector, run) {
  t.mock.method(LlamaWebGpuBridgeRuntime.prototype, 'loadModelFromUrl', async () => 1);
  const load = t.mock.method(LlamaWebGpuBridgeRuntime.prototype, 'loadMultimodalProjector', loadProjector);
  await withStubWorkerEnvironment(async ({ workers }) => {
    const bridge = createRealWorkerBridge({ jsBackedModelStaging: true, logLevel: 0 });
    const oldSelf = globalThis.self;
    const errorReplies = [];
    const connectHost = async (worker) => {
      const host = {
        postMessage: (reply) => {
          const data = structuredClone(reply);
          if (data.type === 'error') errorReplies.push(data);
          worker.onmessage({ data });
        },
      };
      globalThis.self = host;
      installBridgeWorkerHost();
      worker.postMessage = (request) => {
        worker.posted.push(request);
        return host.onmessage({ data: structuredClone(request) });
      };
      await host.onmessage({ data: structuredClone(worker.posted[0]) });
    };
    const replace = bridge._replaceWorkerProxyForMultimodalCpuMode.bind(bridge);
    t.mock.method(bridge, '_replaceWorkerProxyForMultimodalCpuMode', async () => {
      await replace();
      await connectHost(workers.at(-1));
    });
    try {
      await connectHost(workers[0]);
      await bridge.loadModelFromUrl(MODEL, { nGpuLayers });
      assert.equal(bridge._isCpuModelMode(), nGpuLayers === 0);
      const workerLoads = () => workers.flatMap((worker) => worker.posted)
        .filter((request) => request.method === 'loadMultimodalProjector').length;
      await run({ bridge, workers, workerLoads, totalLoads: () => load.mock.callCount(), errorReplies });
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

const failures = [
  [transferCode, transferError, Error, 'ModelTransferLengthError', transferMessage],
  ...stagingCodes.map((code) => [
    code, () => new ModelStagingError(code, 'Projector staging failed.'), ModelStagingError,
    'ModelStagingError', 'Projector staging failed.',
  ]),
];

for (const [nGpuLayers, mode] of [[0, 'CPU'], [1, 'non-CPU']]) {
  for (const [code, makeError, errorClass, name, message] of failures) {
    test(`worker projector rejects ${code} on the first ${mode} load`, async (t) => {
      await withWorkerProjector(t, nGpuLayers, async () => { throw makeError(); },
        async ({ bridge, workers, workerLoads, totalLoads }) => {
          const proxy = bridge._workerProxy;
          const err = await bridge.loadMultimodalProjector(PROJECTOR)
            .then(() => assert.fail('projector load should reject'), (error) => error);
          assert.equal(totalLoads(), 1);
          assert.equal(workerLoads(), 1);
          assert.equal(workers.length, 1);
          assert.equal(bridge._workerProxy, proxy);
          assert.equal(bridge._runtime, null);
          assert.equal(err.constructor, errorClass);
          assert.deepEqual({ name: err.name, message: err.message, code: err.code }, { name, message, code });
          assert.equal(err.llamadartLoadedBytes, undefined);
          assert.equal(err.llamadartFilePath, undefined);
        });
    });
  }

  test(`worker projector recovers from a generic timeout in ${mode} mode`, async (t) => {
    let calls = 0;
    await withWorkerProjector(t, nGpuLayers, async () => {
      if (++calls === 1) throw new Error('Worker request timed out');
      return 1;
    }, async ({ bridge, workers, workerLoads, totalLoads }) => {
      assert.equal(await bridge.loadMultimodalProjector(PROJECTOR), 1);
      assert.equal(totalLoads(), 2);
      const cpu = nGpuLayers === 0;
      assert.equal(workerLoads(), cpu ? 2 : 1);
      assert.equal(workers.length, cpu ? 2 : 1);
      assert.equal(bridge._workerProxy?._worker ?? null, cpu ? workers[1] : null);
    });
  });
}

test('worker projector preserves a transfer length error on the CPU retry', async (t) => {
  let calls = 0;
  await withWorkerProjector(t, 0, async () => {
    if (++calls === 1) throw new Error('Worker request timed out');
    throw transferError();
  }, async ({ bridge, workers, workerLoads, totalLoads }) => {
    await assert.rejects(bridge.loadMultimodalProjector(PROJECTOR), {
      name: 'ModelTransferLengthError', message: transferMessage, code: transferCode,
    });
    assert.equal(totalLoads(), 2);
    assert.equal(workerLoads(), 2);
    assert.equal(workers.length, 2);
  });
});

for (const [code, makeError] of [
  [transferCode, transferError],
  ['MODEL_STAGING_MOUNT_FAILED', () => new ModelStagingError('MODEL_STAGING_MOUNT_FAILED', 'Projector staging failed.')],
  [null, () => new Error('Worker request timed out')],
]) {
  test(`worker projector restore ${code ? `preserves ${code}` : 'recovers from a generic timeout'}`, async (t) => {
    const error = makeError();
    const nextProjector = 'https://models.test/next-mmproj.gguf';
    let rememberedLoads = 0;
    await withWorkerProjector(t, 0, async (url) => {
      if (url === PROJECTOR) {
        rememberedLoads += 1;
        if (rememberedLoads === 2 || (code && rememberedLoads > 2)) throw error;
      }
      return 1;
    }, async ({ bridge, workers, workerLoads }) => {
      await bridge.loadMultimodalProjector(PROJECTOR);
      await bridge._replaceWorkerProxyForMultimodalCpuMode();
      const restoreWorker = bridge._workerProxy;
      const workersBeforeRestore = workers.length;
      const load = bridge.loadMultimodalProjector(nextProjector);
      if (code) {
        await assert.rejects(load, { name: error.name, code });
        assert.equal(rememberedLoads - 1, 1, 'remembered projector restore attempts');
        assert.equal(workers.length - workersBeforeRestore, 0, 'extra worker restarts');
        assert.equal(workerLoads(), 2);
        assert.equal(bridge._workerProxy, restoreWorker);
      } else {
        assert.equal(await load, 1);
        assert.equal(rememberedLoads - 1, 2, 'remembered projector restore attempts');
        assert.equal(workers.length - workersBeforeRestore, 1, 'extra worker restarts');
        assert.equal(workerLoads(), 4);
      }
    });
  });
}

test('worker projector preserves a transfer length error during the CPU retry restore', async (t) => {
  const nextProjector = 'https://models.test/next-mmproj.gguf';
  let rememberedLoads = 0;
  await withWorkerProjector(t, 0, async (url) => {
    if (url === nextProjector) throw new Error('Worker request timed out');
    if (++rememberedLoads > 1) throw transferError();
    return 1;
  }, async ({ bridge, workers, workerLoads }) => {
    await bridge.loadMultimodalProjector(PROJECTOR);
    await assert.rejects(bridge.loadMultimodalProjector(nextProjector), {
      name: 'ModelTransferLengthError', code: transferCode,
    });
    assert.equal(rememberedLoads - 1, 1, 'remembered projector restore attempts');
    assert.equal(workers.length, 2, 'only the initial projector failure should restart the worker');
    assert.equal(workerLoads(), 3);
  });
});

test('an empty-message transfer error keeps download fields off the worker wire', async (t) => {
  await withWorkerProjector(t, 0, async () => { throw transferError(''); },
    async ({ bridge, errorReplies }) => {
      const error = await bridge.loadMultimodalProjector(PROJECTOR)
        .then(() => assert.fail('projector load should reject'), (err) => err);
      assert.equal(errorReplies.length, 1);
      const [reply] = errorReplies;
      assert.doesNotMatch(JSON.stringify(reply), /llamadartLoadedBytes|llamadartFilePath|local\.gguf/);
      assert.equal(reply.errorCode, transferCode);
      assert.equal(reply.message, 'ModelTransferLengthError');
      assert.deepEqual(
        { name: error.name, code: error.code, message: error.message },
        { name: 'ModelTransferLengthError', code: transferCode, message: 'ModelTransferLengthError' },
      );
    });
});

test('transfer length errors match by either name or code with staging disabled', () => {
  const bridge = Object.create(LlamaWebGpuBridge.prototype);
  bridge._config = { jsBackedModelStaging: false };
  assert.equal(bridge._isNonRetryableLoadError({ name: 'ModelTransferLengthError' }), true);
  assert.equal(bridge._isNonRetryableLoadError({ code: transferCode }), true);
  assert.equal(bridge._isNonRetryableLoadError(new Error('Worker request timed out')), false);
});

for (const [code, makeError, , name] of [
  ...failures,
  [undefined, () => new Error('Projector setup failed'), Error, 'Error'],
]) {
  test(`direct projector preserves ${code ?? 'a generic error'}`, async (t) => {
    const error = makeError();
    const load = t.mock.method(LlamaWebGpuBridgeRuntime.prototype, 'loadMultimodalProjector', async () => {
      throw error;
    });
    const bridge = new LlamaWebGpuBridge({ disableWorker: true, jsBackedModelStaging: true });
    try {
      await assert.rejects(bridge.loadMultimodalProjector(PROJECTOR), (err) => {
        assert.equal(err, error);
        assert.equal(err.name, name);
        assert.equal(err.code, code);
        return true;
      });
      assert.equal(load.mock.callCount(), 1);
    } finally {
      await bridge.dispose();
    }
  });
}
