import assert from 'node:assert/strict';

import {
  createDirectBridge,
  opTrace,
  settleQuietly,
} from './bridge_operation_queue_fixtures.mjs';

export const DIRECT_CASES = [
  [0, 'concurrent tokenization ownership (direct)', async () => {
    const { bridge, core } = createDirectBridge();

    const [alpha, beta] = await Promise.all([
      bridge.tokenize('a'),
      bridge.tokenize('b'),
    ]);

    assert.deepEqual(alpha, [97], 'the first tokenize must own its own result buffer');
    assert.deepEqual(beta, [98], 'the second tokenize must own its own result buffer');
    assert.deepEqual(
      opTrace(core).map(([kind, value]) => `${kind}:${value}`),
      ['tokenize:a', 'tokenize:b'],
      'tokenization must run in FIFO order',
    );
  }],

  [1, 'concurrent completions FIFO and intact outputs (direct)', async () => {
    const { bridge, core } = createDirectBridge({
      alpha: ['A', 'L', 'P', 'H', 'A'],
      beta: ['B', 'E', 'T', 'A'],
      gamma: ['G'],
    });

    const results = await Promise.all([
      bridge.createCompletion('alpha'),
      bridge.createCompletion('beta'),
      bridge.createCompletion('gamma'),
    ]);

    assert.deepEqual(results, ['ALPHA', 'BETA', 'G']);
    assert.equal(core.beginRejections, 0, 'the core guard must never be reached');
    assert.deepEqual(
      opTrace(core).map(([kind, value]) => `${kind}:${value}`),
      ['begin:alpha', 'end:alpha', 'begin:beta', 'end:beta', 'begin:gamma', 'end:gamma'],
      'completions must be strictly serialized in FIFO order',
    );
  }],

  [2, 'cross-operation ordering (direct)', async () => {
    const { bridge, core } = createDirectBridge({ alpha: ['A'] });

    const results = await Promise.all([
      bridge.createCompletion('alpha'),
      bridge.tokenize('b'),
      bridge.embed('c'),
      bridge.detokenize([100]),
      bridge.stateSaveBytes([1]),
    ]);

    assert.equal(results[0], 'A');
    assert.deepEqual(results[1], [98]);
    assert.deepEqual(results[2], [99]);
    assert.equal(results[3], 'd');
    assert.deepEqual([...results[4]], [1, 2, 3]);
    assert.deepEqual(
      opTrace(core).map(([kind]) => kind),
      ['begin', 'end', 'tokenize', 'embed', 'detokenize', 'stateSave'],
      'cross-operation ordering must follow call order without interleaving',
    );
  }],

  [3, 'error recovery releases the queue (direct)', async () => {
    const { bridge, core } = createDirectBridge({ beta: ['B'] });

    const failing = bridge.createCompletion('missing-script');
    const queued = bridge.createCompletion('beta');

    assert.equal(await failing, '', 'the stub yields no tokens for an unscripted prompt');
    assert.equal(await queued, 'B');

    bridge._runtime._modelBytes = 0;
    await assert.rejects(bridge.tokenize('a'), /No model loaded/);
    bridge._runtime._modelBytes = 1;
    assert.deepEqual(await bridge.tokenize('a'), [97], 'a rejected operation must not wedge the queue');
    assert.equal(core.beginRejections, 0);
  }],

  [4, 'cancellation stays out-of-band (direct)', async () => {
    // Keep the script under the 256-token nPredict default so an uncancelled run
    // would return every token and truncation is unambiguous.
    const scriptedTokens = 200;
    const { bridge, core } = createDirectBridge({
      alpha: Array.from({ length: scriptedTokens }, () => 'A'),
    });

    const completion = settleQuietly(bridge.createCompletion('alpha'));
    const queuedTokenize = settleQuietly(bridge.tokenize('b'));

    await new Promise((resolve) => setTimeout(resolve, 0));
    bridge.cancel();

    const text = await completion;
    assert.ok(
      text.length > 0 && text.length < scriptedTokens,
      `cancel must truncate generation, got ${text.length}/${scriptedTokens}`,
    );
    assert.deepEqual(await queuedTokenize, [98]);
    assert.deepEqual(
      opTrace(core).filter(([kind]) => kind !== 'begin' && kind !== 'end').map(([kind]) => kind),
      ['tokenize'],
      'the queued operation must run after the cancelled generation released the queue',
    );
  }],

  [8, 'recursive runtime CPU recovery does not deadlock', async () => {
    const scripts = { alpha: [new Error('failed to decode')] };
    const { bridge, core } = createDirectBridge(scripts);
    const runtime = bridge._runtime;
    runtime._shouldAttemptGenerationRecovery = () => true;
    runtime._recoverGenerationWithCpuFallback = async () => {
      scripts.alpha = ['O', 'K'];
      return true;
    };

    const result = await bridge.createCompletion('alpha');

    assert.equal(result, 'OK');
    assert.deepEqual(
      opTrace(core).filter(([kind]) => kind === 'begin').map(([, prompt]) => prompt),
      ['alpha', 'alpha'],
      'the runtime recovery retry must re-enter generation',
    );
    assert.equal(core.beginRejections, 0, 'recovery must end the failed generation before retrying');
  }],

  [9, 'pre-aborted calls never reach the core or the worker (direct)', async () => {
    const { bridge, core } = createDirectBridge({ alpha: ['A'] });
    const controller = new AbortController();
    controller.abort();

    await assert.rejects(
      bridge.createCompletion('alpha', { signal: controller.signal }),
      (error) => error?.name === 'AbortError',
    );
    assert.deepEqual(opTrace(core), [], 'a pre-aborted completion must not enter the core');

    assert.equal(await bridge.createCompletion('alpha'), 'A', 'the queue must stay usable');
  }],

  [10, 'abort while queued skips the slot and spares the predecessor (direct)', async () => {
    const scriptedTokens = 60;
    const { bridge, core } = createDirectBridge({
      first: Array.from({ length: scriptedTokens }, () => 'F'),
      second: ['S'],
    });
    const controller = new AbortController();

    const first = settleQuietly(bridge.createCompletion('first'));
    const second = settleQuietly(
      bridge.createCompletion('second', { signal: controller.signal }),
    );

    await new Promise((resolve) => setTimeout(resolve, 0));
    controller.abort();

    await assert.rejects(second, (error) => error?.name === 'AbortError');
    assert.equal(
      await first,
      'F'.repeat(scriptedTokens),
      'aborting a queued call must not cancel the unrelated active predecessor',
    );
    assert.deepEqual(
      opTrace(core).filter(([kind]) => kind === 'begin').map(([, prompt]) => prompt),
      ['first'],
      'the aborted queued completion must never call begin_generation',
    );
  }],

  [13, 'synchronous getters do not enter the core during active work (direct)', async () => {
    const scriptedTokens = 40;
    const { bridge, core } = createDirectBridge({
      alpha: Array.from({ length: scriptedTokens }, () => 'A'),
    });

    const completion = settleQuietly(bridge.createCompletion('alpha'));
    await new Promise((resolve) => setTimeout(resolve, 0));

    const traceLengthBeforeGetters = core.trace.length;
    const metadata = bridge.getModelMetadata();
    const contextSize = bridge.getContextSize();
    const backendName = bridge.getBackendName();
    const gpuActive = bridge.isGpuActive();
    const supportsVision = bridge.supportsVision();

    assert.equal(contextSize, 128, 'context size must be served from the facade cache');
    assert.equal(metadata['llamadart.webgpu.execution'], 'main-thread');
    assert.equal(metadata['llamadart.webgpu.stub'], '1', 'cached metadata must be real');
    assert.equal(typeof backendName, 'string');
    assert.equal(typeof gpuActive, 'boolean');
    assert.equal(typeof supportsVision, 'boolean');
    assert.deepEqual(
      core.trace.slice(traceLengthBeforeGetters),
      [],
      'no synchronous getter may ccall while an async operation owns the runtime',
    );

    await completion;
  }],

  [14, 'state load default capacity resolves inside the slot (direct)', async () => {
    const { bridge, core } = createDirectBridge({ alpha: ['A'] });

    // A queued load that grows the context; the state load is enqueued behind it
    // before the new size is observable.
    bridge._runtime.loadModelFromUrl = async () => {
      bridge._runtime._nCtx = 4096;
      core.ccall = ((inner) => function ccall(name, retType, argTypes, args = []) {
        if (name === 'llamadart_webgpu_get_context_size') {
          core.trace.push(['contextSize', null]);
          return 4096;
        }
        return inner.call(core, name, retType, argTypes, args);
      })(core.ccall);
      return undefined;
    };

    const completion = settleQuietly(bridge.createCompletion('alpha'));
    const load = settleQuietly(bridge.loadModelFromUrl('model.gguf'));
    const stateLoad = settleQuietly(bridge.stateLoadBytes(new Uint8Array([1, 2, 3])));

    await completion;
    await load;
    await stateLoad;

    const kinds = opTrace(core).map(([kind]) => kind);
    assert.deepEqual(
      kinds.filter((kind) => kind === 'begin' || kind === 'stateLoad'),
      ['begin', 'stateLoad'],
      'the state load must run after the earlier slots released',
    );

    const stateLoadCapacity = opTrace(core).find(([kind]) => kind === 'stateLoad')?.[1];
    assert.equal(
      stateLoadCapacity,
      4096,
      'the default capacity must be resolved inside the slot, after the queued load',
    );
    assert.equal(
      bridge.getContextSize(),
      4096,
      'the facade snapshot must be refreshed by the queued load',
    );
  }],

  [15, 'state load explicit capacities handle conversion exceptions and single coercion', async () => {
    const { bridge, core } = createDirectBridge();
    const resolved = [];
    let coercions = 0;
    const throwingPrimitive = {
      [Symbol.toPrimitive]() {
        throw new TypeError('cannot convert token capacity');
      },
    };
    const coercible = {
      valueOf() {
        coercions += 1;
        return 9.75;
      },
    };

    for (const tokenCapacity of [
      0,
      -1,
      Number.NaN,
      Number.POSITIVE_INFINITY,
      Symbol('invalid'),
      throwingPrimitive,
      coercible,
    ]) {
      await bridge.stateLoadBytes(new Uint8Array([1, 2, 3]), tokenCapacity);
      resolved.push(
        opTrace(core).filter(([kind]) => kind === 'stateLoad').at(-1)?.[1],
      );
    }

    assert.deepEqual(
      resolved,
      [128, 128, 128, Number.POSITIVE_INFINITY, 128, 128, 9],
      'invalid capacities fall back while valid numeric conversion is truncated',
    );
    assert.equal(coercions, 1, 'a capacity object must be coerced only once');
  }],

  [19, 'cache metadata reflects completed prefetch and eviction', async () => {
    const { bridge } = createDirectBridge();
    const runtime = bridge._runtime;

    const baseline = bridge.getModelMetadata();
    const baselineContext = bridge.getContextSize();

    runtime.prefetchModelToCache = async function prefetch() {
      this._modelCacheState = 'ready';
      this._modelCacheName = 'custom-cache-v9';
      this._modelSource = 'cache';
      this._runtimeNotes.push('model_cache_prefetched');
      return true;
    };
    assert.equal(await bridge.prefetchModelToCache('model.gguf'), true);

    let metadata = bridge.getModelMetadata();
    assert.equal(metadata['llamadart.webgpu.model_cache_state'], 'ready');
    assert.equal(metadata['llamadart.webgpu.model_cache_name'], 'custom-cache-v9');
    assert.equal(metadata['llamadart.webgpu.model_source'], 'cache');
    assert.match(metadata['llamadart.webgpu.runtime_notes'], /model_cache_prefetched/);

    // Non-cache facade state must be untouched by the cache-only refresh.
    assert.equal(bridge.getContextSize(), baselineContext);
    assert.equal(
      metadata['llamadart.webgpu.model_bytes'],
      baseline['llamadart.webgpu.model_bytes'],
    );
    assert.equal(metadata['llamadart.webgpu.execution'], 'main-thread');

    runtime.evictModelFromCache = async function evict() {
      this._modelCacheState = 'evicted';
      this._runtimeNotes.push('model_cache_evicted');
      return true;
    };
    assert.equal(await bridge.evictModelFromCache('model.gguf'), true);

    metadata = bridge.getModelMetadata();
    assert.equal(metadata['llamadart.webgpu.model_cache_state'], 'evicted');
    assert.match(metadata['llamadart.webgpu.runtime_notes'], /model_cache_evicted/);
  }],

  [22, 're-entrant begin maps to a stable error without reading last_error', async () => {
    const { bridge, core } = createDirectBridge({ alpha: ['A'] });
    core.lastError = 'stale unrelated error text';
    core.activePrompt = 'someone-else';

    await assert.rejects(
      bridge.createCompletion('alpha'),
      (error) => {
        assert.match(error.message, /Generation is already active on this bridge runtime/);
        assert.doesNotMatch(error.message, /stale unrelated error text/);
        return true;
      },
    );
    assert.equal(core.beginRejections, 1);
  }],

  [23, 'unqueued helpers reject after dispose without resurrecting the runtime (direct)', async () => {
    const { bridge } = createDirectBridge();
    let runtimesCreated = 0;
    bridge._createRuntime = () => {
      runtimesCreated += 1;
      return {};
    };

    await bridge.dispose();
    assert.equal(bridge._runtime, null, 'teardown must leave no runtime');

    await assert.rejects(bridge.prefetchModelToCache('model.gguf'), /Bridge has been disposed/);
    await assert.rejects(bridge.evictModelFromCache('model.gguf'), /Bridge has been disposed/);
    await assert.rejects(
      bridge.applyChatTemplate([{ role: 'user', content: 'hi' }], true),
      /Bridge has been disposed/,
    );

    assert.equal(runtimesCreated, 0, 'a disposed bridge must never rebuild a runtime');
    assert.equal(bridge._runtime, null, 'no runtime may be resurrected');
  }],

  [39, 'operation queues are per bridge instance', async () => {
    const slow = createDirectBridge({ alpha: Array.from({ length: 80 }, () => 'A') });
    const fast = createDirectBridge({ beta: ['B'] });

    const slowCompletion = settleQuietly(slow.bridge.createCompletion('alpha'));
    assert.equal(
      await fast.bridge.createCompletion('beta'),
      'B',
      'a second bridge instance must not wait on the first instance queue',
    );

    slow.bridge.cancel();
    await slowCompletion;
  }],
];
