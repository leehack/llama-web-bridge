import assert from 'node:assert/strict';

import {
  createRealWorkerBridge,
  createSnapshotRuntime,
  createWorkerBridge,
  settleQuietly,
  withStubWorkerEnvironment,
  workerDriver,
} from './bridge_operation_queue_fixtures.mjs';

export const WORKER_PROXY_CASES = [
  [5, 'worker dispatch FIFO ordering', async () => {
    const dispatched = [];
    const active = [];
    const releases = [];
    const bridge = createWorkerBridge({
      _callWorker: (method, args) => {
        dispatched.push(`${method}:${JSON.stringify(args[0])}`);
        active.push(method);
        assert.equal(active.length, 1, `worker dispatch overlapped on ${method}`);
        return new Promise((resolve) => {
          releases.push(() => {
            active.pop();
            resolve(method === 'createCompletion' ? 'WORKER-ALPHA' : [98]);
          });
        });
      },
    });

    const completion = settleQuietly(bridge.createCompletion('alpha'));
    const tokenize = settleQuietly(bridge.tokenize('b'));

    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.deepEqual(dispatched, ['createCompletion:"alpha"'], 'the queued tokenize must not dispatch yet');

    releases[0]();
    assert.equal(await completion, 'WORKER-ALPHA');
    await new Promise((resolve) => setTimeout(resolve, 0));
    releases[1]();
    assert.deepEqual(await tokenize, [98]);
    assert.deepEqual(dispatched, ['createCompletion:"alpha"', 'tokenize:"b"']);
  }],

  [6, 'worker cancellation stays out-of-band', async () => {
    const dispatched = [];
    let releaseCompletion = null;
    const bridge = createWorkerBridge({
      _callWorker: (method) => {
        dispatched.push(method);
        if (method === 'cancel') {
          return Promise.resolve(undefined);
        }
        return new Promise((resolve) => {
          releaseCompletion = () => resolve('WORKER-ALPHA');
        });
      },
    });

    const completion = settleQuietly(bridge.createCompletion('alpha'));
    settleQuietly(bridge.tokenize('b'));

    await new Promise((resolve) => setTimeout(resolve, 0));
    bridge.cancel();
    assert.deepEqual(
      dispatched,
      ['createCompletion', 'cancel'],
      'cancel must reach the worker before the in-flight completion settles',
    );

    releaseCompletion();
    assert.equal(await completion, 'WORKER-ALPHA');
  }],

  [7, 'recursive empty-multimodal retry does not deadlock', async () => {
    const dispatched = [];
    const bridge = createWorkerBridge({
      _ensureWorkerMultimodalCpuMode: async () => {},
      _replaceWorkerProxyForMultimodalCpuMode: async () => {},
      _callWorker: async (method, args) => {
        dispatched.push([method, args[0]]);
        return dispatched.length === 1 ? '' : 'RETRIED';
      },
    });

    const options = { parts: [{ type: 'image', bytes: new Uint8Array([1]) }] };
    const result = await bridge.createCompletion('alpha', options);

    assert.equal(result, 'RETRIED');
    assert.equal(dispatched.length, 2, 'the empty-response retry must reach the worker twice');
    assert.equal(await bridge.createCompletion('alpha', options), 'RETRIED', 'the queue must release');
  }],

  [11, 'abort while queued skips worker dispatch', async () => {
    const dispatched = [];
    let releaseFirst = null;
    const controller = new AbortController();
    const bridge = createWorkerBridge({
      _callWorker: (method, args) => {
        dispatched.push(`${method}:${JSON.stringify(args?.[0] ?? null)}`);
        if (method === 'cancel') {
          return Promise.resolve(undefined);
        }
        return new Promise((resolve) => {
          releaseFirst = () => resolve('FIRST');
        });
      },
    });

    const first = settleQuietly(bridge.createCompletion('first'));
    const second = settleQuietly(
      bridge.createCompletion('second', { signal: controller.signal }),
    );

    await new Promise((resolve) => setTimeout(resolve, 0));
    controller.abort();
    await assert.rejects(second, (error) => error?.name === 'AbortError');

    assert.deepEqual(
      dispatched,
      ['createCompletion:"first"'],
      'the aborted queued call must not dispatch, and must not cancel the predecessor',
    );

    releaseFirst();
    assert.equal(await first, 'FIRST');
  }],

  [12, 'queued synthesizeSpeech and loadModelFromUrl honour abort', async () => {
    const dispatched = [];
    let releaseFirst = null;
    const bridge = createWorkerBridge({
      _callWorker: (method) => {
        dispatched.push(method);
        if (method === 'cancel') {
          return Promise.resolve(undefined);
        }
        return new Promise((resolve) => {
          releaseFirst = () => resolve('FIRST');
        });
      },
    });

    const first = settleQuietly(bridge.createCompletion('first'));
    const ttsController = new AbortController();
    const loadController = new AbortController();
    const tts = settleQuietly(bridge.synthesizeSpeech({ text: 'hi', signal: ttsController.signal }));
    const load = settleQuietly(
      bridge.loadModelFromUrl('model.gguf', { signal: loadController.signal }),
    );

    await new Promise((resolve) => setTimeout(resolve, 0));
    ttsController.abort();
    loadController.abort();

    await assert.rejects(tts, (error) => error?.name === 'AbortError');
    await assert.rejects(load, (error) => error?.name === 'AbortError');
    assert.deepEqual(dispatched, ['createCompletion'], 'no skipped slot may dispatch');

    releaseFirst();
    assert.equal(await first, 'FIRST');
  }],

  [17, 'dispose rejects queued work and tears down after the active owner (worker)', async () => {
    const dispatched = [];
    let releaseActive = null;
    let disposeCalls = 0;
    const bridge = createWorkerBridge({
      _workerProxy: {
        async dispose() {
          disposeCalls += 1;
        },
      },
      _callWorker: (method) => {
        dispatched.push(method);
        return new Promise((resolve) => {
          releaseActive = () => resolve('ACTIVE');
        });
      },
    });

    const active = settleQuietly(bridge.createCompletion('alpha'));
    const queued = settleQuietly(bridge.tokenize('b'));

    await new Promise((resolve) => setTimeout(resolve, 0));
    const disposal = bridge.dispose();

    await assert.rejects(queued, /Bridge has been disposed/);
    assert.deepEqual(dispatched, ['createCompletion'], 'queued work must never dispatch');
    assert.equal(disposeCalls, 0, 'teardown must wait for the active owner');

    releaseActive();
    assert.equal(await active, 'ACTIVE', 'active work still settles normally');

    await disposal;
    await bridge.dispose();
    assert.equal(disposeCalls, 1, 'worker teardown must be idempotent');
    assert.equal(bridge._workerProxy, null);
    await assert.rejects(bridge.embed('x'), /Bridge has been disposed/);
  }],

  [24, 'unqueued helpers reject after dispose without dispatching (worker)', async () => {
    let dispatches = 0;
    let runtimesCreated = 0;
    const bridge = createWorkerBridge({
      _workerProxy: { async dispose() {} },
      _callWorker: async () => {
        dispatches += 1;
        return undefined;
      },
      _createRuntime: () => {
        runtimesCreated += 1;
        return {};
      },
    });

    const disposal = bridge.dispose();
    await disposal;

    await assert.rejects(bridge.prefetchModelToCache('model.gguf'), /Bridge has been disposed/);
    await assert.rejects(bridge.evictModelFromCache('model.gguf'), /Bridge has been disposed/);
    await assert.rejects(
      bridge.applyChatTemplate([{ role: 'user', content: 'hi' }], true),
      /Bridge has been disposed/,
    );

    assert.equal(dispatches, 0, 'a disposed bridge must never dispatch to a worker');
    assert.equal(runtimesCreated, 0, 'a disposed bridge must never rebuild a runtime');
    assert.equal(bridge._workerProxy, null);
    assert.equal(bridge._runtime, null);

    // The lifecycle gate must not disturb dispose() idempotence.
    assert.equal(bridge.dispose(), disposal, 'dispose must still return the same promise');
    assert.equal(bridge._runtime, null, 'repeated dispose must not resurrect a runtime');
  }],

  [25, 'real worker proxy dispatches queued operations in FIFO order', async () => {
    await withStubWorkerEnvironment(async ({ workers }) => {
      const bridge = createRealWorkerBridge();
      const driver = workerDriver(bridge._workerProxy, workers[0]);
      driver.ready();

      const first = settleQuietly(bridge.tokenize('a'));
      const second = settleQuietly(bridge.tokenize('b'));
      await driver.settle();

      assert.equal(driver.calls('tokenize').length, 1, 'the second call must wait its turn');

      driver.reply(driver.calls('tokenize')[0].id, [97]);
      assert.deepEqual(await first, [97]);
      await driver.settle();

      const tokenizeCalls = driver.calls('tokenize');
      assert.equal(tokenizeCalls.length, 2, 'the queued call dispatches once the slot frees');
      assert.deepEqual(tokenizeCalls.map((call) => call.args[0]), ['a', 'b'], 'FIFO order');

      driver.reply(tokenizeCalls[1].id, [98]);
      assert.deepEqual(await second, [98]);
      assert.equal(bridge._workerProxy._pending.size, 0, 'no pending requests may remain');
    });
  }],

  [26, 'queued abort neither dispatches nor cancels the real active owner', async () => {
    await withStubWorkerEnvironment(async ({ workers }) => {
      const bridge = createRealWorkerBridge();
      const driver = workerDriver(bridge._workerProxy, workers[0]);
      driver.ready();
      const controller = new AbortController();

      const active = settleQuietly(bridge.createCompletion('alpha'));
      const queued = settleQuietly(
        bridge.createCompletion('beta', { signal: controller.signal }),
      );
      await driver.settle();
      assert.equal(driver.calls('createCompletion').length, 1);

      controller.abort();
      await assert.rejects(queued, (error) => error?.name === 'AbortError');

      assert.equal(
        driver.calls('createCompletion').length,
        1,
        'the aborted queued call must never dispatch',
      );
      assert.equal(driver.calls('cancel').length, 0, 'the active owner must not be cancelled');

      driver.reply(driver.calls('createCompletion')[0].id, 'ALPHA');
      assert.equal(await active, 'ALPHA', 'the active owner completes normally');
    });
  }],

  [28, 'a superseded real proxy response cannot overwrite the facade snapshot', async () => {
    await withStubWorkerEnvironment(async ({ workers }) => {
      const bridge = createRealWorkerBridge();
      const staleProxy = bridge._workerProxy;
      const driver = workerDriver(staleProxy, workers[0]);
      driver.ready();

      const pending = settleQuietly(bridge._callWorker('tokenize', ['a']));
      await driver.settle();
      const request = driver.calls('tokenize')[0];

      // The proxy identity changes before its reply arrives.
      bridge._workerProxy = { id: 'fresh' };
      bridge._metadata = { fresh: '1' };
      bridge._contextSize = 2048;

      driver.reply(request.id, [97], {
        metadata: { resurrected: '1' },
        contextSize: 8192,
        supportsVision: true,
      });
      assert.deepEqual(await pending, [97], 'the caller still receives its own result');

      assert.deepEqual(
        bridge.getModelMetadata(),
        { fresh: '1', 'llamadart.webgpu.execution': 'worker' },
        'a superseded proxy must not write into the current snapshot',
      );
      assert.equal(bridge.getContextSize(), 2048);
      assert.equal(bridge.supportsVision(), false);

      bridge._workerProxy = staleProxy;
      staleProxy._worker.terminate();
    });
  }],

  [30, 'late worker rejection after dispose does not resurrect the runtime', async () => {
    let runtimesCreated = 0;
    let fallbacks = 0;
    let rejectLogLevel = null;
    const bridge = createWorkerBridge({
      _workerProxy: { async dispose() {} },
      _callWorker: () => new Promise((_, reject) => {
        rejectLogLevel = reject;
      }),
      _createRuntime: () => {
        runtimesCreated += 1;
        return createSnapshotRuntime();
      },
    });
    const realDisable = bridge._disableWorkerFallback.bind(bridge);
    bridge._disableWorkerFallback = (error) => {
      fallbacks += 1;
      return realDisable(error);
    };

    bridge.setLogLevel(1);
    await new Promise((resolve) => setTimeout(resolve, 0));

    const disposal = bridge.dispose();
    await disposal;
    assert.equal(bridge._runtime, null, 'teardown must leave no runtime');

    const unhandled = [];
    const onUnhandled = (reason) => unhandled.push(reason);
    process.on('unhandledRejection', onUnhandled);
    try {
      rejectLogLevel(new Error('worker died'));
      await new Promise((resolve) => setTimeout(resolve, 10));
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }

    assert.deepEqual(unhandled, [], 'setLogLevel must not raise an unhandled rejection');
    assert.equal(fallbacks, 0, 'no worker fallback may run after disposal');
    assert.equal(runtimesCreated, 0, 'no runtime may be rebuilt after disposal');
    assert.equal(bridge._runtime, null, 'no runtime may be resurrected');
    await assert.rejects(
      bridge.applyChatTemplate([{ role: 'user', content: 'hi' }], true),
      /Bridge has been disposed/,
    );
    assert.equal(bridge.dispose(), disposal, 'dispose stays identical and idempotent');
  }],

  [31, 'worker-mode cache helpers preserve the complete worker snapshot', async () => {
    let workerDispatches = 0;
    let coreCalls = 0;
    const cacheRuntime = {
      _modelSource: 'network',
      _modelCacheState: 'unavailable',
      _modelCacheName: 'default-cache',
      _runtimeNotes: [],
      _core: { ccall: () => { coreCalls += 1; return 0; } },
      async prefetchModelToCache() {
        this._modelCacheState = 'ready';
        this._modelCacheName = 'custom-cache-v9';
        this._modelSource = 'cache';
        this._runtimeNotes.push('model_cache_prefetched');
        return true;
      },
      async evictModelFromCache() {
        this._modelCacheState = 'evicted';
        this._runtimeNotes.push('model_cache_evicted');
        return true;
      },
    };
    const bridge = createWorkerBridge({
      _runtime: cacheRuntime,
      _metadata: {
        'llamadart.webgpu.n_gpu_layers': '99',
        'llamadart.webgpu.model_bytes': '4096',
        'llamadart.webgpu.model_source': 'worker-network',
        'llamadart.webgpu.model_cache_state': 'worker-old',
        'llamadart.webgpu.model_cache_name': 'worker-cache',
        'llamadart.webgpu.runtime_notes': 'worker_threads;worker_fallback',
      },
      _contextSize: 8192,
      _gpuActive: true,
      _backendName: 'WebGPU (Prototype bridge)',
      _supportsVision: true,
      _callWorker: async () => {
        workerDispatches += 1;
        return undefined;
      },
    });

    assert.equal(await bridge.prefetchModelToCache('model.gguf'), true);
    let metadata = bridge.getModelMetadata();
    assert.equal(metadata['llamadart.webgpu.model_cache_state'], 'worker-old');
    assert.equal(metadata['llamadart.webgpu.model_cache_name'], 'worker-cache');
    assert.equal(metadata['llamadart.webgpu.model_source'], 'worker-network');
    assert.equal(metadata['llamadart.webgpu.runtime_notes'], 'worker_threads;worker_fallback');

    // Worker-owned fields must survive untouched.
    assert.equal(metadata['llamadart.webgpu.execution'], 'worker');
    assert.equal(metadata['llamadart.webgpu.n_gpu_layers'], '99');
    assert.equal(metadata['llamadart.webgpu.model_bytes'], '4096');
    assert.equal(bridge.getContextSize(), 8192);
    assert.equal(bridge.isGpuActive(), true);
    assert.equal(bridge.getBackendName(), 'WebGPU (Prototype bridge)');
    assert.equal(bridge.supportsVision(), true);

    assert.equal(await bridge.evictModelFromCache('model.gguf'), true);
    metadata = bridge.getModelMetadata();
    assert.equal(metadata['llamadart.webgpu.model_cache_state'], 'worker-old');
    assert.equal(metadata['llamadart.webgpu.model_cache_name'], 'worker-cache');
    assert.equal(metadata['llamadart.webgpu.model_source'], 'worker-network');
    assert.equal(metadata['llamadart.webgpu.runtime_notes'], 'worker_threads;worker_fallback');
    assert.equal(metadata['llamadart.webgpu.execution'], 'worker');

    assert.equal(coreCalls, 0, 'cache helpers must never call into the core');
    assert.equal(workerDispatches, 0, 'cache helpers must never dispatch to the worker');
  }],

  [32, 'a state-bearing owner applies its state before applyChatTemplate can fall back', async () => {
    const events = [];
    const runtime = createSnapshotRuntime();
    let resolveLoad = null;
    const bridge = createWorkerBridge({
      _runtime: null,
      _workerProxy: { async dispose() { events.push('worker-disposed'); } },
      _createRuntime: () => runtime,
      _callWorker: (method) => {
        events.push(`dispatch:${method}`);
        if (method === 'applyChatTemplate') {
          return Promise.reject(new Error('worker died'));
        }
        return new Promise((resolve) => {
          resolveLoad = () => {
            events.push('owner-state-applied');
            bridge._contextSize = 8192;
            resolve('LOADED');
          };
        });
      },
    });

    const owner = settleQuietly(bridge.loadModelFromUrl('model.gguf'));
    const template = settleQuietly(
      bridge.applyChatTemplate([{ role: 'user', content: 'hi' }], true),
    );
    await new Promise((resolve) => setTimeout(resolve, 0));

    assert.deepEqual(
      events,
      ['dispatch:loadModelFromUrl'],
      'applyChatTemplate must not dispatch or replace topology while the owner runs',
    );
    assert.notEqual(bridge._workerProxy, null, 'the owner still has its worker');

    resolveLoad();
    assert.equal(await owner, 'LOADED');
    assert.equal(await template, 'user: hi\nassistant: ');

    assert.deepEqual(
      events,
      [
        'dispatch:loadModelFromUrl',
        'owner-state-applied',
        'dispatch:applyChatTemplate',
        'worker-disposed',
      ],
      'the owner applies its state before any fallback replaces the worker',
    );
  }],

  [33, 'applyChatTemplate serializes with other queued work and falls back under the queue', async () => {
    const dispatched = [];
    const runtime = createSnapshotRuntime();
    const bridge = createWorkerBridge({
      _runtime: null,
      _workerProxy: { async dispose() {} },
      _createRuntime: () => runtime,
      _callWorker: async (method) => {
        dispatched.push(method);
        if (method === 'applyChatTemplate') {
          throw new Error('worker died');
        }
        return [98];
      },
    });

    const results = await Promise.all([
      bridge.tokenize('b'),
      bridge.applyChatTemplate([{ role: 'user', content: 'hi' }], true),
    ]);

    assert.deepEqual(results[0], [98]);
    assert.equal(results[1], 'user: hi\nassistant: ', 'the direct fallback result is returned');
    assert.deepEqual(
      dispatched,
      ['tokenize', 'applyChatTemplate'],
      'applyChatTemplate waits its turn in FIFO order',
    );
    assert.equal(bridge._workerProxy, null, 'the fallback ran and replaced topology');

    assert.equal(
      await bridge.applyChatTemplate([{ role: 'user', content: 'again' }], false),
      'user: again',
    );
  }],

  [34, 'setLogLevel worker failure never replaces the worker or the direct runtime', async () => {
    let rejectLogLevel = null;
    let runtimesCreated = 0;
    const bridge = createWorkerBridge({
      _runtime: null,
      _createRuntime: () => {
        runtimesCreated += 1;
        return createSnapshotRuntime();
      },
      _callWorker: () => new Promise((_, reject) => {
        rejectLogLevel = reject;
      }),
    });
    const originalProxy = bridge._workerProxy;

    const unhandled = [];
    const onUnhandled = (reason) => unhandled.push(reason);
    process.on('unhandledRejection', onUnhandled);
    try {
      bridge.setLogLevel(1);
      await new Promise((resolve) => setTimeout(resolve, 0));
      rejectLogLevel(new Error('worker died'));
      await new Promise((resolve) => setTimeout(resolve, 20));
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }

    assert.deepEqual(unhandled, [], 'log-level delivery must not raise an unhandled rejection');
    assert.equal(bridge._workerProxy, originalProxy, 'the worker must not be replaced');
    assert.equal(runtimesCreated, 0, 'no direct runtime may be created');
    assert.equal(bridge._runtime, null);
    assert.equal(bridge._workerFallbackReason, null, 'no fallback may be recorded');
  }],

  [38, 'a rejected predecessor releases the queue (worker)', async () => {
    const dispatched = [];
    const bridge = createWorkerBridge({
      _callWorker: async (method, args) => {
        dispatched.push(`${method}:${JSON.stringify(args?.[0] ?? null)}`);
        if (method === 'tokenize') {
          throw new Error('worker tokenize exploded');
        }
        return 'OK';
      },
      _disableWorkerFallback: () => {
        throw new Error('worker tokenize exploded');
      },
    });

    const failing = settleQuietly(bridge.tokenize('a'));
    const following = settleQuietly(bridge.createCompletion('alpha'));

    await assert.rejects(failing, /worker tokenize exploded/);
    assert.equal(await following, 'OK', 'the queue must release after a rejection');
    assert.deepEqual(dispatched, ['tokenize:"a"', 'createCompletion:"alpha"']);
  }],
];
