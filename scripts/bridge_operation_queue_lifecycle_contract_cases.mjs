import assert from 'node:assert/strict';

import {
  DISPOSE_METHOD,
  UNQUEUED_ASYNC_METHODS,
  UNQUEUED_SYNC_METHODS,
  createDirectBridge,
  createRealWorkerBridge,
  createStubCore,
  createWorkerBridge,
  opTrace,
  settleQuietly,
  withStubWorkerEnvironment,
  workerDriver,
} from './bridge_operation_queue_fixtures.mjs';

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { LlamaWebGpuBridge } from '../js/src/llama_webgpu_bridge.js';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export const LIFECYCLE_CONTRACT_CASES = [
  [16, 'dispose rejects queued work and tears down after the active owner (direct)', async () => {
    const scriptedTokens = 40;
    const { bridge, core } = createDirectBridge({
      alpha: Array.from({ length: scriptedTokens }, () => 'A'),
      beta: ['B'],
    });

    const active = settleQuietly(bridge.createCompletion('alpha'));
    const queued = settleQuietly(bridge.createCompletion('beta'));

    await new Promise((resolve) => setTimeout(resolve, 0));
    const disposal = bridge.dispose();

    await assert.rejects(queued, /Bridge has been disposed/);
    assert.equal(
      await active,
      'A'.repeat(scriptedTokens),
      'the active operation completes normally; disposal does not interrupt it',
    );
    assert.deepEqual(
      opTrace(core).filter(([kind]) => kind === 'begin').map(([, prompt]) => prompt),
      ['alpha'],
      'the queued completion must never reach the core',
    );

    await disposal;
    await bridge.dispose();
    await assert.rejects(bridge.tokenize('a'), /Bridge has been disposed/);
    assert.equal(bridge._runtime, null, 'teardown completes after the owner released');
  }],

  [18, 'settled operations do not accumulate lifecycle listeners', async () => {
    const { bridge } = createDirectBridge({ alpha: ['A'] });
    assert.equal(bridge._disposalWaiters.size, 0, 'a fresh bridge holds no waiters');

    for (let index = 0; index < 25; index += 1) {
      await bridge.tokenize('a');
    }
    assert.equal(
      bridge._disposalWaiters.size,
      0,
      'settled operations must detach their disposal listener',
    );

    const pending = settleQuietly(bridge.createCompletion('alpha'));
    assert.equal(
      bridge._disposalWaiters.size,
      0,
      'an active operation must detach its disposal listener once it owns the slot',
    );
    await pending;
    assert.equal(bridge._disposalWaiters.size, 0, 'settlement must leave no active listener');

    let releaseWorker = null;
    const workerBridge = createWorkerBridge({
      _callWorker: (method) => new Promise((resolve) => {
        if (method === 'createCompletion') {
          releaseWorker = resolve;
        } else {
          resolve([98]);
        }
      }),
    });
    const active = settleQuietly(workerBridge.createCompletion('alpha'));
    await new Promise((resolve) => setTimeout(resolve, 0));
    const queued = settleQuietly(workerBridge.tokenize('b'));
    assert.equal(
      workerBridge._disposalWaiters.size,
      1,
      'a queued operation registers exactly one disposal listener',
    );
    releaseWorker('WORKER');
    await active;
    await queued;
    assert.equal(workerBridge._disposalWaiters.size, 0, 'settlement must detach the listener');

    const rejected = settleQuietly(bridge.stateLoadBytes(new Uint8Array()));
    await assert.rejects(rejected, /State bytes are empty/);
    assert.equal(
      bridge._disposalWaiters.size,
      0,
      'a rejected operation must detach its listener too',
    );
  }],

  [20, 'queue covers every stateful public async method', async () => {
    const prototype = LlamaWebGpuBridge.prototype;
    const publicNames = Object.getOwnPropertyNames(prototype)
      .filter((name) => name !== 'constructor' && !name.startsWith('_'));

    const queued = [];
    for (const name of publicNames) {
      const descriptor = Object.getOwnPropertyDescriptor(prototype, name);
      assert.equal(typeof descriptor.value, 'function', `${name} must be a method`);

      const source = descriptor.value.toString();
      const isAsync = source.startsWith('async');
      const isQueued = source.includes('_runExclusive(');

      if (name === DISPOSE_METHOD) {
        assert.ok(!isAsync, 'dispose must not be async so repeated calls share one promise');
        assert.match(source, /this\._disposed = true/, 'dispose must mark disposal first');
        assert.match(
          source,
          /this\._notifyDisposalWaiters\(\)/,
          'dispose must reject queued work synchronously',
        );
        assert.match(
          source,
          /allowDisposed: true/,
          'dispose teardown must take the queue with allowDisposed',
        );
        continue;
      }

      if (!isAsync) {
        assert.ok(
          UNQUEUED_SYNC_METHODS.has(name),
          `synchronous method ${name} is not accounted for in UNQUEUED_SYNC_METHODS`,
        );
        assert.ok(!isQueued, `synchronous method ${name} must not take the queue`);
        continue;
      }

      if (UNQUEUED_ASYNC_METHODS.has(name)) {
        assert.ok(
          !isQueued,
          `${name} is documented as unqueued (${UNQUEUED_ASYNC_METHODS.get(name)}) but takes the queue`,
        );
        continue;
      }

      assert.ok(
        isQueued,
        `public async method ${name} touches the singleton runtime but is not queued; `
        + 'queue it or justify it in UNQUEUED_ASYNC_METHODS',
      );
      queued.push(name);
    }

    assert.deepEqual(
      queued,
      [
        'loadModelFromUrl',
        'createCompletion',
        'loadMultimodalProjector',
        'unloadMultimodalProjector',
        'getTextToSpeechCapabilities',
        'synthesizeSpeech',
        'tokenize',
        'stateSaveFile',
        'stateLoadFile',
        'stateSaveBytes',
        'stateLoadBytes',
        'detokenize',
        'embed',
        'embedBatch',
        'applyChatTemplate',
      ],
      'the set of queued public operations changed',
    );

    assert.equal(
      typeof prototype._createCompletionUnlocked,
      'function',
      '_createCompletionUnlocked must remain available for recursive retries',
    );
    assert.equal(
      typeof prototype._disposeUnlocked,
      'function',
      '_disposeUnlocked must remain the queued teardown boundary',
    );
    assert.match(
      prototype.createCompletion.toString(),
      /this\._createCompletionUnlocked\(/,
      'createCompletion must keep its re-entrant unlocked boundary',
    );
  }],

  [21, 'C++ begin_generation guard rejects before destroying state', async () => {
    const core = readFileSync(path.join(rootDir, 'src/llama_webgpu_core.cpp'), 'utf8');
    const start = core.indexOf('int32_t begin_generation_impl(');
    assert.ok(start > 0, 'begin_generation_impl must exist');
    const body = core.slice(start, core.indexOf('\n}\n', start));

    const guardIndex = body.indexOf('if (g_generation_active) {');
    assert.ok(guardIndex > 0, 'begin_generation_impl must reject a re-entrant begin');
    const guardBlock = body.slice(guardIndex, body.indexOf('}', guardIndex) + 1);
    assert.match(
      guardBlock,
      /if \(g_generation_active\) \{\s*\n\s*return -7;\s*\n\s*\}/,
      'the guard must return -7 and nothing else',
    );
    assert.doesNotMatch(
      guardBlock,
      /set_error|clear_error|g_last_/,
      'the guard must not touch g_last_error or any other shared buffer',
    );

    for (const destructive of [
      'clear_error();',
      'g_last_output.clear();',
      'g_last_piece.clear();',
      'end_generation_state();',
    ]) {
      const index = body.indexOf(destructive);
      assert.ok(index > 0, `begin_generation_impl must still contain ${destructive}`);
      assert.ok(
        guardIndex < index,
        `the re-entrancy guard must run before ${destructive}, `
        + `but the guard is at ${guardIndex} and ${destructive} at ${index}`,
      );
    }

    const stubCore = createStubCore({ alpha: ['A'] });
    assert.equal(stubCore.ccall('llamadart_webgpu_begin_generation', 'number', [], ['alpha']), 0);
    stubCore.lastOutput = 'PARTIAL';
    stubCore.lastError = 'earlier unrelated failure';
    assert.equal(stubCore.ccall('llamadart_webgpu_begin_generation', 'number', [], ['beta']), -7);
    assert.equal(stubCore.beginRejections, 1);
    assert.equal(
      stubCore.lastOutput,
      'PARTIAL',
      'a rejected begin must leave the active generation output untouched',
    );
    assert.equal(
      stubCore.activePrompt,
      'alpha',
      'a rejected begin must leave the active generation in place',
    );
    assert.equal(
      stubCore.ccall('llamadart_webgpu_last_error', 'string', [], []),
      'earlier unrelated failure',
      'a rejected begin must not overwrite g_last_error',
    );
  }],

  [27, 'dispose waits for the real active owner, then disposes and revokes once', async () => {
    await withStubWorkerEnvironment(async ({ workers, revoked }) => {
      const bridge = createRealWorkerBridge();
      const proxy = bridge._workerProxy;
      const driver = workerDriver(proxy, workers[0]);
      driver.ready();

      const active = settleQuietly(bridge.createCompletion('alpha'));
      const queued = settleQuietly(bridge.tokenize('b'));
      await driver.settle();

      const disposal = bridge.dispose();
      await assert.rejects(queued, /Bridge has been disposed/);
      await driver.settle();

      assert.equal(driver.calls('dispose').length, 0, 'teardown must wait for the owner');
      assert.equal(workers[0].terminated, 0);

      driver.reply(driver.calls('createCompletion')[0].id, 'ALPHA');
      assert.equal(await active, 'ALPHA');
      await driver.settle();

      const disposeCalls = driver.calls('dispose');
      assert.equal(disposeCalls.length, 1, 'teardown dispatches once the owner released');
      driver.reply(disposeCalls[0].id, null);
      await disposal;

      assert.equal(workers[0].terminated, 1, 'the worker must be terminated exactly once');
      assert.deepEqual(revoked, ['blob:stub-worker'], 'the blob URL must be revoked once');
      assert.equal(bridge._workerProxy, null);

      await bridge.dispose();
      assert.equal(workers[0].terminated, 1, 'repeated dispose must not terminate again');
      assert.deepEqual(revoked, ['blob:stub-worker'], 'repeated dispose must not revoke again');
    });
  }],

  [29, 'public dispose waits for superseded worker teardown', async () => {
    await withStubWorkerEnvironment(async ({ workers, revoked }) => {
      const bridge = createRealWorkerBridge();
      bridge._emitBridgeWarn = () => {};
      const staleProxy = bridge._workerProxy;
      const driver = workerDriver(staleProxy, workers[0]);
      driver.ready();

      const active = settleQuietly(
        bridge.applyChatTemplate([{ role: 'user', content: 'hi' }], true),
      );
      await driver.settle();
      const activeCall = driver.calls('applyChatTemplate')[0];
      assert.ok(activeCall, 'the worker-backed operation must own the queue');

      const disposal = bridge.dispose();
      let disposalResolved = false;
      disposal.then(() => {
        disposalResolved = true;
      });
      await driver.settle();
      assert.equal(driver.calls('dispose').length, 0, 'teardown waits for the active owner');
      assert.equal(disposalResolved, false, 'queued disposal must still be pending');

      driver.error(activeCall.id, 'worker died');
      assert.equal(await active, 'user: hi\nassistant: ');
      await driver.settle();

      const staleDisposeCall = driver.calls('dispose')[0];
      assert.ok(staleDisposeCall, 'fallback must start stale worker disposal');
      assert.equal(
        disposalResolved,
        false,
        'public disposal must wait while stale worker termination is held',
      );
      assert.equal(workers[0].terminated, 0, 'the stale worker must not terminate early');
      assert.deepEqual(revoked, [], 'the stale worker blob must remain live while pending');

      driver.reply(staleDisposeCall.id, null);
      await disposal;

      assert.equal(workers[0].terminated, 1, 'public disposal waits for worker termination');
      assert.deepEqual(revoked, ['blob:stub-worker'], 'public disposal waits for blob revocation');
      assert.equal(bridge._workerDisposePromise, null, 'the drained stale promise is cleared');
    });
  }],

  [35, 'setLogLevel after dispose throws the stable lifecycle error', async () => {
    const { bridge } = createDirectBridge();
    await bridge.dispose();
    assert.equal(bridge._runtime, null);
    assert.throws(() => bridge.setLogLevel(1), /Bridge has been disposed/);
  }],

  [36, 'dispose returns the identical promise for repeated calls', async () => {
    const { bridge } = createDirectBridge();
    const first = bridge.dispose();
    const second = bridge.dispose();
    assert.equal(first, second, 'dispose must return the same Promise object');
    await first;
    assert.equal(bridge.dispose(), first, 'dispose stays idempotent after teardown');
  }],

  [37, 're-entrant logger observes the reserved dispose promise', async () => {
    const { bridge, core } = createDirectBridge();
    bridge._runtime._mmProjPath = '/mmproj.gguf';
    const innerCcall = core.ccall.bind(core);
    core.ccall = (name, retType, argTypes, args) => {
      if (name === 'llamadart_webgpu_mmproj_free') {
        return 1;
      }
      return innerCcall(name, retType, argTypes, args);
    };

    let reentrantDispose = null;
    bridge._runtime._config.logger = {
      warn: () => {
        reentrantDispose = bridge.dispose();
      },
    };

    const first = bridge.dispose();
    assert.equal(
      reentrantDispose,
      first,
      'a synchronous logger callback must observe the stable public dispose promise',
    );
    await first;
  }],
];
