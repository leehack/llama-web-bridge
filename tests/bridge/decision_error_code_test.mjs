import assert from 'node:assert/strict';

import { decisionCoreStatusCode } from '../../js/src/internal/decision.ts';
import { LlamaWebGpuBridge, enableBridgeWorkerHost } from '../../js/src/llama_webgpu_bridge.js';
import { readRepoText } from './bridge_js_source.mjs';
import {
  createRealWorkerBridge,
  createWorkerBridge,
  withStubWorkerEnvironment,
  workerDriver,
} from './bridge_operation_queue_fixtures.mjs';

const sequence = { tokens: [1], markers: [0], questionType: 0 };

// Every core status has its own code, and every code is documented.
{
  const expected = {
    INVALID_ARGUMENT: 'invalid-argument',
    UNSUPPORTED: 'unsupported',
    INVALID_STATE: 'invalid-state',
    MODEL_ERROR: 'model',
    CONTEXT_ERROR: 'context',
    INFERENCE_ERROR: 'inference',
    BUSY: 'busy',
    HEAD_NOT_LOADED: 'head-lost',
    NO_MODEL: 'no-model',
  };
  const statuses = [...readRepoText('src/llama_webgpu_decision.h')
    .matchAll(/LLAMADART_WEBGPU_DECISION_STATUS_(\w+) = (-?\d+)/g)]
    .filter(([, name]) => name !== 'OK');
  assert.deepEqual(statuses.map(([, name]) => name).sort(), Object.keys(expected).sort());
  for (const [, name, value] of statuses) {
    assert.equal(decisionCoreStatusCode(Number(value)), expected[name], `core status ${name}`);
  }

  const declared = readRepoText('js/src/llama_webgpu_bridge.d.ts')
    .match(/export type DecisionErrorCode =([^;]+);/)[1]
    .match(/'[a-z-]+'/g)
    .map((code) => code.slice(1, -1));
  const documented = [...readRepoText('docs/api.md').matchAll(/^\| `([a-z-]+)` \|/gm)]
    .map(([, code]) => code);
  for (const code of declared) {
    assert.ok(documented.includes(code), `docs/api.md documents decision error code ${code}`);
  }
}

// Rejects with `code` and keeps the message llamadart's message fallback matches.
async function rejectsWith(promise, code, message) {
  await assert.rejects(promise, (error) => {
    assert.equal(error.code, code, `code for: ${error.message}`);
    assert.match(error.message, message);
    return true;
  });
}

function createDecisionCore({ loadStatus = 7, runStatus = 1, error = '' } = {}) {
  const files = new Map();
  return {
    FS: {
      mkdir: () => {},
      writeFile: (path, bytes) => files.set(path, new Uint8Array(bytes)),
      readFile: (path) => files.get(path) || new Uint8Array(),
      unlink: (path) => files.delete(path),
    },
    _llamadart_webgpu_decision_capabilities_json: () => 0,
    ccall(name, _returnType, _argTypes, args = []) {
      switch (name) {
        case 'llamadart_webgpu_decision_api_version':
          return 1;
        case 'llamadart_webgpu_decision_capabilities_json':
          return JSON.stringify({ apiVersion: 1, supported: true });
        case 'llamadart_webgpu_decision_load':
          return Promise.resolve(loadStatus);
        case 'llamadart_webgpu_decision_head_info_json':
          return JSON.stringify({ apiVersion: 1, handle: args[0] });
        case 'llamadart_webgpu_decision_run':
          files.set(args[2], new Uint8Array(4));
          return Promise.resolve(runStatus);
        case 'llamadart_webgpu_decision_free':
          return Promise.resolve(0);
        case 'llamadart_webgpu_last_error':
          return error;
        case 'llamadart_webgpu_media_clear_pending':
        case 'llamadart_webgpu_mmproj_free':
        case 'llamadart_webgpu_shutdown':
          return 0;
        default:
          throw new Error(`Unexpected ccall: ${name}`);
      }
    },
  };
}

function createDirectBridge(coreOptions) {
  const bridge = new LlamaWebGpuBridge({ disableWorker: true });
  bridge._runtime._core = createDecisionCore(coreOptions);
  bridge._runtime._modelBytes = 1;
  return bridge;
}

// Direct runtime: each core status maps to its code; the message is the core's.
for (const [status, code] of [
  [-1, 'invalid-argument'],
  [-2, 'unsupported'],
  [-3, 'invalid-state'],
  [-4, 'model'],
  [-5, 'context'],
  [-6, 'inference'],
  [-7, 'busy'],
  [-8, 'head-lost'],
  [-9, 'no-model'],
  [-99, 'runtime'],
]) {
  const error = `core status ${status}`;
  const loading = createDirectBridge({ loadStatus: status, error });
  await rejectsWith(
    loading.loadDecisionHead(new Uint8Array([1])),
    code,
    new RegExp(`^Failed to load decision head: ${error}$`),
  );

  const running = createDirectBridge({ runStatus: status, error });
  const { handle } = await running.loadDecisionHead(new Uint8Array([1]));
  await rejectsWith(
    running.runDecision(handle, [sequence]),
    code,
    new RegExp(`^Decision run failed: ${error}$`),
  );
}

// Direct runtime: rejections the bridge raises before or after the core.
{
  const bridge = createDirectBridge();
  for (const [call, message] of [
    [() => bridge.runDecision(0, []), /handle must be a positive integer/],
    [() => bridge.runDecision(1.5, []), /handle must be a positive integer/],
    [() => bridge.freeDecisionHead(-1), /handle must be a positive integer/],
    [() => bridge.loadDecisionHead([1]), /must be a URL string, an ArrayBuffer or a typed array/],
    [() => bridge.loadDecisionHead(new Uint8Array()), /^Decision head bytes are empty\.$/],
    [() => bridge.loadDecisionHead(''), /^Decision head URL is empty\.$/],
    [() => bridge.loadDecisionHead(new Uint8Array([1]), { configJson: 1 }), /configJson must be a string/],
  ]) {
    await rejectsWith(call(), 'invalid-argument', message);
  }

  const { handle } = await bridge.loadDecisionHead(new Uint8Array([1]));
  for (const sequences of [
    'tokens',
    [null],
    [{ tokens: [0.5], markers: [0], questionType: 0 }],
    [{ tokens: [1], markers: 0, questionType: 0 }],
    [{ tokens: [1], markers: [0], questionType: 'choice' }],
  ]) {
    await rejectsWith(bridge.runDecision(handle, sequences), 'invalid-argument', /^Decision sequence/);
  }
  const typeError = await bridge.runDecision(handle, 'tokens').catch((error) => error);
  assert.ok(typeError instanceof TypeError, 'argument errors stay TypeErrors');

  await rejectsWith(
    bridge.runDecision(handle, [sequence]),
    'inference',
    /^Decision output from the WebGPU core is malformed\.$/,
  );
  await rejectsWith(bridge.runDecision(99, []), 'head-lost', /Load the decision head again\.$/);
  await bridge.freeDecisionHead(handle);
  await rejectsWith(bridge.runDecision(handle, []), 'head-lost', /Load the decision head again\.$/);

  bridge._runtime._modelBytes = 0;
  await rejectsWith(bridge.loadDecisionHead(new Uint8Array([1])), 'no-model', /^No model loaded\./);
  await rejectsWith(bridge.runDecision(handle, []), 'head-lost', /Load the decision head again\.$/);
}

{
  const bridge = createDirectBridge();
  const { handle } = await bridge.loadDecisionHead(new Uint8Array([1]));
  bridge._runtime._modelBytes = 0;
  await rejectsWith(bridge.runDecision(handle, [sequence]), 'no-model', /^No model loaded\./);
}

{
  const bridge = createDirectBridge();
  delete bridge._runtime._core._llamadart_webgpu_decision_capabilities_json;
  await rejectsWith(
    bridge.loadDecisionHead(new Uint8Array([1])),
    'unsupported',
    /^This WebGPU core build does not include decision heads\.$/,
  );
}

{
  const bridge = createDirectBridge();
  bridge._runtime._fetchWithTimeout = async () => ({ ok: false, status: 404, statusText: 'Not Found' });
  await rejectsWith(
    bridge.loadDecisionHead('https://example.test/head.safetensors'),
    'model',
    /^Failed to fetch decision head: 404 Not Found$/,
  );
}

{
  const bridge = createDirectBridge();
  await bridge.dispose();
  for (const call of [
    () => bridge.getDecisionCapabilities(),
    () => bridge.loadDecisionHead(new Uint8Array([1])),
    () => bridge.runDecision(1, []),
    () => bridge.freeDecisionHead(1),
  ]) {
    await rejectsWith(call(), 'disposed', /^Bridge has been disposed\.$/);
  }
}

{
  const bridge = createDirectBridge();
  bridge._runtime = null;
  await rejectsWith(bridge.loadDecisionHead(new Uint8Array([1])), 'no-model', /^No model loaded\./);
}

// Worker path: cancellation, a lost worker, an unrecovered failure, and a
// coded worker error that is rethrown as is.
{
  let bridge;
  bridge = createWorkerBridge({
    _callWorker: async () => {
      bridge.cancel();
      throw new Error('Worker request failed');
    },
  });
  const error = await bridge.loadDecisionHead(new Uint8Array([1])).catch((caught) => caught);
  assert.equal(error.name, 'AbortError');
  assert.equal(error.code, 'cancelled');
  assert.equal(error.message, 'Decision head load was cancelled.');
}

{
  const runtime = {
    _modelBytes: 0,
    _runtimeNotes: [],
    async loadModelFromUrl() {
      this._modelBytes = 1;
    },
  };
  const bridge = createWorkerBridge({
    _workerProxy: { dispose: async () => {} },
    _createRuntime: () => runtime,
    _callWorker: async (method) => {
      if (method === 'loadDecisionHead') {
        return { apiVersion: 1, handle: 3 };
      }
      throw new Error('Bridge worker request failed: worker terminated');
    },
  });
  await bridge.loadDecisionHead(new Uint8Array([1]));
  await rejectsWith(
    bridge.runDecision(1, [sequence]),
    'head-lost',
    /^Decision head 1 was lost when the bridge worker failed .*Load the decision head again\.$/,
  );
}

{
  const bridge = createWorkerBridge({
    _callWorker: async () => {
      throw new Error('RuntimeError: Aborted(native code called abort())');
    },
  });
  await rejectsWith(bridge.getDecisionCapabilities(), 'runtime', /native code called abort/);
}

{
  const busy = Object.assign(new Error('Decision run failed: busy'), { code: 'busy' });
  const bridge = createWorkerBridge({
    _callWorker: async (method) => {
      if (method === 'loadDecisionHead') {
        return { apiVersion: 1, handle: 3 };
      }
      throw busy;
    },
  });
  await bridge.loadDecisionHead(new Uint8Array([1]));
  await rejectsWith(bridge.runDecision(1, [sequence]), 'busy', /^Decision run failed: busy$/);
}

// Worker host: the code crosses the boundary next to the message.
{
  const originals = {
    self: globalThis.self,
    loadUnlocked: LlamaWebGpuBridge.prototype._loadDecisionHeadUnlocked,
    capabilitiesUnlocked: LlamaWebGpuBridge.prototype._getDecisionCapabilitiesUnlocked,
  };
  const posted = [];
  globalThis.self = { postMessage: (message) => posted.push(message) };
  LlamaWebGpuBridge.prototype._loadDecisionHeadUnlocked = async () => {
    throw Object.assign(new Error('Failed to load decision head: bad shape'), { code: 'model' });
  };
  LlamaWebGpuBridge.prototype._getDecisionCapabilitiesUnlocked = async () => {
    throw new Error('uncoded failure');
  };
  try {
    enableBridgeWorkerHost();
    await globalThis.self.onmessage({ data: { type: 'init', config: {} } });
    await globalThis.self.onmessage({
      data: { type: 'call', id: 1, method: 'loadDecisionHead', args: [new Uint8Array([1]), {}] },
    });
    await globalThis.self.onmessage({
      data: { type: 'call', id: 2, method: 'getDecisionCapabilities', args: [] },
    });
    const errorFor = (id) => posted.find((message) => message.id === id && message.type === 'error');
    assert.equal(errorFor(1).decisionErrorCode, 'model');
    assert.equal(errorFor(1).message, 'Failed to load decision head: bad shape');
    assert.equal(errorFor(2).decisionErrorCode, 'runtime', 'the worker bridge codes uncoded failures');
  } finally {
    LlamaWebGpuBridge.prototype._loadDecisionHeadUnlocked = originals.loadUnlocked;
    LlamaWebGpuBridge.prototype._getDecisionCapabilitiesUnlocked = originals.capabilitiesUnlocked;
    globalThis.self = originals.self;
  }
}

// Worker proxy: the facade rejects with the worker's code and message.
await withStubWorkerEnvironment(async ({ workers }) => {
  const bridge = createRealWorkerBridge();
  const proxy = bridge._workerProxy;
  const driver = workerDriver(proxy, workers[0]);
  driver.ready();

  const loading = bridge.loadDecisionHead(new Uint8Array([1]));
  await driver.settle();
  const [load] = driver.calls('loadDecisionHead');
  driver.reply(load.id, { apiVersion: 1, handle: 5 });
  const { handle } = await loading;

  const running = bridge.runDecision(handle, [sequence]);
  await driver.settle();
  const [run] = driver.calls('runDecision');
  proxy._worker.onmessage({
    data: {
      type: 'error',
      id: run.id,
      message: 'Decision run failed: The decision encoder context accepts 512 tokens.',
      decisionErrorCode: 'context',
    },
  });
  await rejectsWith(running, 'context', /^Decision run failed: The decision encoder context/);

  const unknown = proxy.call('runDecision', [handle, []]);
  await driver.settle();
  const [, second] = driver.calls('runDecision');
  proxy._worker.onmessage({
    data: { type: 'error', id: second.id, message: 'x', decisionErrorCode: 'not-a-code' },
  });
  const error = await unknown.catch((caught) => caught);
  assert.equal(error.code, undefined, 'the proxy drops unknown codes');

  await bridge.dispose().catch(() => {});
});

console.log('Decision error code tests passed');
