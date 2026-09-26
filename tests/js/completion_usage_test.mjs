import assert from 'node:assert/strict';

import {
  createDirectBridge,
  createWorkerBridge,
  settleQuietly,
} from './bridge_operation_queue_fixtures.mjs';

function usageCalls(core) {
  return core.trace.filter(([kind]) => kind === 'usage').length;
}

const CANCELLED_USAGE = {
  promptTokens: 7,
  cachedPromptTokens: 0,
  completionTokens: 1,
  timeToFirstTokenMs: 4,
  durationMs: 9,
  finishReason: 'cancelled',
};

// A worker proxy whose completion streams one token, runs cancel(), then
// reports the cancelled generation the way the worker host does.
function cancellingProxy(cancel) {
  return {
    async call(method, _args, onEvent) {
      if (method !== 'createCompletion') {
        return { value: undefined };
      }
      onEvent?.({ type: 'event', event: 'token', payload: { pieceText: 'a' } });
      cancel();
      onEvent?.({ type: 'event', event: 'usage', payload: CANCELLED_USAGE });
      return { type: 'result', value: 'a' };
    },
    async dispose() {},
  };
}

function workerCrash() {
  return Object.assign(new Error('worker died'), { llamadartWorkerCrash: true });
}

const CASES = [
  ['a completion that stops reports its counts and timings once, before it resolves', async () => {
    const { bridge } = createDirectBridge({ hello: ['a', 'b', 'c'] });
    const events = [];
    const text = await bridge.createCompletion('hello', {
      onToken: () => events.push('token'),
      onUsage: (usage) => events.push(usage),
    }).then((value) => {
      events.push('resolved');
      return value;
    });

    assert.equal(text, 'abc');
    assert.deepEqual(events.slice(0, 3), ['token', 'token', 'token']);
    assert.equal(events[4], 'resolved');
    const usage = events[3];
    assert.equal(usage.promptTokens, 5);
    assert.equal(usage.cachedPromptTokens, 0);
    assert.equal(usage.completionTokens, 3);
    assert.equal(usage.finishReason, 'stop');
    assert.equal(typeof usage.timeToFirstTokenMs, 'number');
    assert.ok(usage.timeToFirstTokenMs >= 0);
    assert.ok(usage.durationMs >= usage.timeToFirstTokenMs);
  }],

  ['a completion stopped by nPredict ends with length', async () => {
    const { bridge } = createDirectBridge({ hello: ['a', 'b', 'c'] });
    let usage = null;
    assert.equal(
      await bridge.createCompletion('hello', { nPredict: 2, onUsage: (value) => { usage = value; } }),
      'ab',
    );
    assert.equal(usage.finishReason, 'length');
    assert.equal(usage.completionTokens, 2);
  }],

  ['a completion that streams no text has no time to first token', async () => {
    const { bridge } = createDirectBridge({ hello: [] });
    let usage = null;
    assert.equal(await bridge.createCompletion('hello', { onUsage: (value) => { usage = value; } }), '');
    assert.equal(usage.timeToFirstTokenMs, null);
    assert.equal(usage.finishReason, 'stop');
    assert.equal(usage.completionTokens, 0);
  }],

  ['cancel() during generation resolves with a cancelled usage', async () => {
    const { bridge } = createDirectBridge({ hello: ['a', 'b', 'c'] });
    let usage = null;
    const text = await bridge.createCompletion('hello', {
      onToken: () => bridge.cancel(),
      onUsage: (value) => { usage = value; },
    });
    assert.equal(text, 'a');
    assert.equal(usage.finishReason, 'cancelled');
    assert.equal(usage.completionTokens, 1);
  }],

  ['an abort during generation resolves with a cancelled usage', async () => {
    const { bridge } = createDirectBridge({ hello: ['a', 'b', 'c'] });
    const controller = new AbortController();
    let usage = null;
    const text = await bridge.createCompletion('hello', {
      signal: controller.signal,
      onToken: () => controller.abort(),
      onUsage: (value) => { usage = value; },
    });
    assert.equal(text, 'a');
    assert.equal(usage.finishReason, 'cancelled');
  }],

  ['a failed or pre-aborted completion reports no usage', async () => {
    const { bridge } = createDirectBridge({ broken: ['a', new Error('core failure')], hello: ['a'] });
    const reported = [];
    await assert.rejects(
      bridge.createCompletion('broken', { onUsage: (value) => reported.push(value) }),
    );

    const controller = new AbortController();
    controller.abort();
    await assert.rejects(
      bridge.createCompletion('hello', {
        signal: controller.signal,
        onUsage: (value) => reported.push(value),
      }),
      (error) => error?.name === 'AbortError',
    );
    assert.deepEqual(reported, []);
  }],

  ['without onUsage the direct runtime never reads usage from the core', async () => {
    const { bridge, core } = createDirectBridge({ hello: ['a'] });
    assert.equal(await bridge.createCompletion('hello'), 'a');
    assert.equal(usageCalls(core), 0);
    await bridge.createCompletion('hello', { onUsage: () => {} });
    assert.equal(usageCalls(core), 1);
  }],

  ['a worker completion forwards the worker usage and sends no function options', async () => {
    const workerUsage = {
      promptTokens: 7,
      cachedPromptTokens: 3,
      completionTokens: 2,
      timeToFirstTokenMs: 4,
      durationMs: 9,
      finishReason: 'stop',
    };
    let sentOptions = null;
    const bridge = createWorkerBridge({
      _callWorker: async (method, args, onEvent) => {
        assert.equal(method, 'createCompletion');
        sentOptions = args[1];
        onEvent({ type: 'event', event: 'token', payload: { pieceText: 'ab' } });
        onEvent({ type: 'event', event: 'usage', payload: workerUsage });
        return 'ab';
      },
    });

    const reported = [];
    assert.equal(
      await bridge.createCompletion('hello', {
        onToken: () => {},
        onUsage: (value) => reported.push(value),
      }),
      'ab',
    );
    assert.deepEqual(reported, [workerUsage]);
    assert.doesNotThrow(() => structuredClone(sentOptions));
    assert.equal('onUsage' in sentOptions, false);
  }],

  ['a worker generation cancelled by an abort reports its usage before rejecting', async () => {
    const controller = new AbortController();
    const bridge = createWorkerBridge({
      _workerProxy: cancellingProxy(() => controller.abort()),
    });

    const events = [];
    await assert.rejects(
      bridge.createCompletion('hello', {
        signal: controller.signal,
        onUsage: (value) => events.push(value),
      }).catch((error) => {
        events.push('rejected');
        throw error;
      }),
      (error) => error?.name === 'AbortError',
    );
    assert.deepEqual(events, [CANCELLED_USAGE, 'rejected']);
  }],

  ['a worker generation stopped by cancel() reports its usage before resolving', async () => {
    let bridge = null;
    bridge = createWorkerBridge({
      _workerProxy: cancellingProxy(() => bridge.cancel()),
    });

    const reported = [];
    assert.equal(
      await bridge.createCompletion('hello', { onUsage: (value) => reported.push(value) }),
      'a',
    );
    assert.deepEqual(reported, [CANCELLED_USAGE]);
  }],

  ['a retired worker cannot report usage for a cancelled operation', async () => {
    const replacement = { call: async () => ({ value: undefined }), async dispose() {} };
    let bridge = null;
    bridge = createWorkerBridge({
      _workerProxy: cancellingProxy(() => {
        bridge.cancel();
        bridge._workerProxy = replacement;
      }),
    });

    const reported = [];
    assert.equal(
      await bridge.createCompletion('hello', { onUsage: (value) => reported.push(value) }),
      'a',
    );
    assert.deepEqual(reported, []);
  }],

  ['a skipped multimodal warmup and a worker failure after a cancel report no usage', async () => {
    const reported = [];
    const warmupBridge = createWorkerBridge({
      _workerProxy: { async dispose() {} },
      _ensureWorkerMultimodalCpuMode: async () => {
        throw new Error('setup failed');
      },
    });
    assert.equal(
      await warmupBridge.createCompletion('hello', {
        warmup: true,
        parts: [{ type: 'image', bytes: new Uint8Array([1]) }],
        onUsage: (value) => reported.push(value),
      }),
      '',
    );

    const controller = new AbortController();
    const failingBridge = createWorkerBridge({
      _workerProxy: {
        async call(method, _args, onEvent) {
          if (method !== 'createCompletion') {
            return { value: undefined };
          }
          onEvent?.({ type: 'event', event: 'token', payload: { pieceText: 'a' } });
          controller.abort();
          throw new Error('core failure');
        },
        async dispose() {},
      },
    });
    await assert.rejects(
      failingBridge.createCompletion('hello', {
        signal: controller.signal,
        onUsage: (value) => reported.push(value),
      }),
      (error) => error?.name === 'AbortError',
    );
    assert.deepEqual(reported, []);
  }],

  ['a main-thread fallback reports only its own usage', async () => {
    const mainUsage = {
      promptTokens: 5,
      cachedPromptTokens: 0,
      completionTokens: 1,
      timeToFirstTokenMs: 1,
      durationMs: 2,
      finishReason: 'stop',
    };
    const bridge = createWorkerBridge({
      _workerProxy: { async dispose() {} },
      _ensureRuntimeReadyAfterWorkerFallback: async () => {},
      _runtime: {
        async createCompletion(_prompt, options) {
          options.onUsage(mainUsage);
          return 'MAIN';
        },
      },
      _callWorker: async (_method, _args, onEvent) => {
        onEvent({ type: 'event', event: 'usage', payload: { ...mainUsage, completionTokens: 99 } });
        throw workerCrash();
      },
    });

    const reported = [];
    assert.equal(
      await settleQuietly(bridge.createCompletion('hello', { onUsage: (value) => reported.push(value) })),
      'MAIN',
    );
    assert.deepEqual(reported, [mainUsage]);
  }],
];

for (const [name, run] of CASES) {
  try {
    await run();
  } catch (error) {
    console.error(`FAIL: ${name}`);
    throw error;
  }
  console.log(`ok - ${name}`);
}
