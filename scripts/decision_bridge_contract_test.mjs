import assert from 'node:assert/strict';

import { LlamaWebGpuBridge, enableBridgeWorkerHost } from '../js/src/llama_webgpu_bridge.js';
import {
  createRealWorkerBridge,
  createWorkerBridge,
  withStubWorkerEnvironment,
  workerDriver,
} from './bridge_operation_queue_fixtures.mjs';

function encodeOutputs(outputs) {
  const words = outputs.reduce(
    (sum, output) => sum + 2 + output.logits.length + output.actLogits.length,
    0,
  );
  const bytes = new Uint8Array(words * 4);
  const view = new DataView(bytes.buffer);
  let offset = 0;
  for (const output of outputs) {
    view.setInt32(offset, output.logits.length, true);
    view.setInt32(offset + 4, output.actLogits.length, true);
    offset += 8;
    for (const value of [...output.logits, ...output.actLogits]) {
      view.setFloat32(offset, value, true);
      offset += 4;
    }
  }
  return bytes;
}

function readInt32Words(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from({ length: bytes.byteLength / 4 }, (_, index) => view.getInt32(index * 4, true));
}

function createDecisionCore({
  version = 1,
  exported = true,
  loadResult = 7,
  loadError = '',
  outputs = [{ logits: [0.5, -1.25], actLogits: [2, 3, 4] }],
  outputBytes = null,
} = {}) {
  const files = new Map();
  const core = {
    calls: [],
    files,
    lastError: '',
    FS: {
      mkdir: () => {},
      analyzePath: (path) => ({ exists: files.has(path) }),
      writeFile: (path, bytes) => files.set(path, new Uint8Array(bytes)),
      readFile: (path) => files.get(path) || new Uint8Array(),
      unlink: (path) => files.delete(path),
    },
    ccall(name, _returnType, _argTypes, args = []) {
      core.calls.push([name, ...args]);
      switch (name) {
        case 'llamadart_webgpu_decision_api_version':
          return version;
        case 'llamadart_webgpu_decision_capabilities_json':
          return JSON.stringify({ apiVersion: 1, supported: true });
        case 'llamadart_webgpu_decision_load':
          core.stagedHead = files.get(args[0]);
          core.stagedConfig = args[2] == null ? null : new TextDecoder().decode(files.get(args[2]));
          core.lastError = loadError;
          return Promise.resolve(loadResult);
        case 'llamadart_webgpu_decision_head_info_json':
          return JSON.stringify({ apiVersion: 1, handle: args[0], hiddenSize: 768 });
        case 'llamadart_webgpu_decision_run':
          core.input = files.get(args[1]);
          files.set(args[2], outputBytes ?? encodeOutputs(outputs));
          return Promise.resolve(outputs.length);
        case 'llamadart_webgpu_decision_free':
          return Promise.resolve(0);
        case 'llamadart_webgpu_last_error':
          return core.lastError;
        case 'llamadart_webgpu_media_clear_pending':
        case 'llamadart_webgpu_mmproj_free':
        case 'llamadart_webgpu_shutdown':
          return 0;
        default:
          throw new Error(`Unexpected ccall: ${name}`);
      }
    },
  };
  if (exported) {
    core._llamadart_webgpu_decision_capabilities_json = () => 0;
  }
  return core;
}

function createDirectDecisionBridge(coreOptions) {
  const bridge = new LlamaWebGpuBridge({ disableWorker: true });
  const core = createDecisionCore(coreOptions);
  bridge._runtime._core = core;
  bridge._runtime._modelBytes = 1;
  return { bridge, core };
}

const called = (core, name) => core.calls.some((call) => call[0] === name);

// Direct runtime: bytes staging, facade handles, input/output wire format.
{
  const { bridge, core } = createDirectDecisionBridge();
  const info = await bridge.loadDecisionHead(new Uint8Array([1, 2, 3]), {
    configJson: '{"max_len": 512}',
  });
  assert.equal(info.handle, 1, 'the facade returns its own handle');
  assert.equal(info.hiddenSize, 768);
  const load = core.calls.find((call) => call[0] === 'llamadart_webgpu_decision_load');
  assert.equal(load[2], 'decision head bytes');
  assert.match(load[3], /^\/decision\/config_.+\.json$/, 'configJson reaches the core as a file');
  assert.equal(core.stagedConfig, '{"max_len": 512}');
  assert.deepEqual(Array.from(core.stagedHead), [1, 2, 3]);
  assert.equal(core.files.size, 0, 'the staged head and config files are deleted after loading');

  const outputs = await bridge.runDecision(1, [
    { tokens: new Int32Array([50281, 5, 50282]), markers: [1], questionType: 2 },
  ]);
  const run = core.calls.find((call) => call[0] === 'llamadart_webgpu_decision_run');
  assert.equal(run[1], 7, 'the facade handle resolves to the core handle');
  assert.deepEqual(readInt32Words(core.input), [1, 2, 3, 1, 50281, 5, 50282, 1]);
  assert.equal(outputs.length, 1);
  assert.ok(outputs[0].logits instanceof Float32Array);
  assert.deepEqual(Array.from(outputs[0].logits), [0.5, -1.25]);
  assert.deepEqual(Array.from(outputs[0].actLogits), [2, 3, 4]);
  assert.equal(core.files.size, 0, 'decision input and output files are deleted');

  core.calls.length = 0;
  for (const [sequence, pattern] of [
    [{ tokens: [2 ** 31], markers: [0], questionType: 0 }, /tokens\[0\] is 2147483648; expected a 32-bit integer/],
    [{ tokens: [1.5], markers: [0], questionType: 0 }, /tokens\[0\] is 1\.5/],
    [{ tokens: ['1'], markers: [0], questionType: 0 }, /tokens\[0\] is 1;/],
    [{ tokens: [1], markers: 0, questionType: 0 }, /markers must be an array or typed array/],
    [{ tokens: [1], markers: [0], questionType: 'choice' }, /questionType is choice/],
  ]) {
    await assert.rejects(bridge.runDecision(1, [sequence]), pattern);
  }
  await assert.rejects(bridge.runDecision(1, 'tokens'), /Decision sequences must be an array/);
  assert.equal(called(core, 'llamadart_webgpu_decision_run'), false, 'malformed input never reaches the core');

  await assert.rejects(bridge.runDecision(0, []), /handle must be a positive integer/);
  await assert.rejects(bridge.runDecision(99, []), /Decision head 99 is not loaded/);

  await bridge.freeDecisionHead(1);
  assert.ok(called(core, 'llamadart_webgpu_decision_free'));
  core.calls.length = 0;
  await bridge.freeDecisionHead(1);
  assert.equal(called(core, 'llamadart_webgpu_decision_free'), false, 'free is idempotent');
  await assert.rejects(bridge.runDecision(1, []), /Decision head 1 is not loaded/);

  core.calls.length = 0;
  const second = await bridge.loadDecisionHead(new ArrayBuffer(4));
  assert.equal(second.handle, 2, 'facade handles are never reused');
  const secondLoad = core.calls.find((call) => call[0] === 'llamadart_webgpu_decision_load');
  assert.equal(secondLoad[3], null, 'no configJson means no config file');
  assert.equal(core.stagedConfig, null);
  const replacement = bridge._runtime;
  bridge._runtime = Object.create(Object.getPrototypeOf(replacement));
  Object.assign(bridge._runtime, replacement);
  await assert.rejects(
    bridge.runDecision(2, []),
    /Decision head 2 is not loaded; it was freed, its model was unloaded, or the bridge runtime restarted/,
  );
  await bridge.dispose();
  assert.equal(bridge._decisionHeads.size, 0);
}

// Direct runtime: load failures, bad arguments and capability probes.
{
  const { bridge, core } = createDirectDecisionBridge({
    loadResult: -4,
    loadError: 'Invalid safetensors file "decision head bytes": header is not a JSON object.',
  });
  await assert.rejects(
    bridge.loadDecisionHead(new Uint8Array([1])),
    /Failed to load decision head: Invalid safetensors file "decision head bytes"/,
  );
  assert.equal(core.files.size, 0, 'a failed load still deletes the staged file');
  await assert.rejects(bridge.loadDecisionHead([1, 2, 3]), /must be a URL string, an ArrayBuffer or a typed array/);
  await assert.rejects(bridge.loadDecisionHead(new Uint8Array()), /Decision head bytes are empty/);
  await assert.rejects(bridge.loadDecisionHead(''), /Decision head URL is empty/);
  await assert.rejects(
    bridge.loadDecisionHead(new Uint8Array([1]), { configJson: { max_len: 512 } }),
    /configJson must be a string/,
  );
}

{
  const { bridge } = createDirectDecisionBridge({ exported: false });
  const capabilities = await bridge.getDecisionCapabilities();
  assert.equal(capabilities.supported, false);
  assert.match(capabilities.reason, /does not include decision heads/);
  await assert.rejects(bridge.loadDecisionHead(new Uint8Array([1])), /does not include decision heads/);
}

{
  const { bridge } = createDirectDecisionBridge({ version: 2 });
  const capabilities = await bridge.getDecisionCapabilities();
  assert.equal(capabilities.supported, false);
  assert.match(capabilities.reason, /decision API version 2; this bridge needs version 1/);
}

{
  const { bridge } = createDirectDecisionBridge({ outputBytes: new Uint8Array(4) });
  await bridge.loadDecisionHead(new Uint8Array([1]));
  await assert.rejects(
    bridge.runDecision(1, [{ tokens: [1], markers: [0], questionType: 0 }]),
    /Decision output from the WebGPU core is malformed/,
  );
}

// Direct runtime: a URL head reports download progress and a short label.
{
  const { bridge, core } = createDirectDecisionBridge();
  const fetched = [];
  bridge._runtime._fetchWithTimeout = async (url) => {
    fetched.push(url);
    return {
      ok: true,
      status: 200,
      statusText: 'OK',
      body: null,
      headers: new Headers({ 'content-length': '3' }),
      arrayBuffer: async () => new Uint8Array([4, 5, 6]).buffer,
    };
  };
  const progress = [];
  await bridge.loadDecisionHead(`https://example.test/heads/${'h'.repeat(300)}.safetensors`, {
    onProgress: (event) => progress.push(event),
  });
  assert.equal(fetched.length, 1);
  assert.deepEqual(progress, [{ loaded: 3, total: 3 }]);
  assert.deepEqual(Array.from(core.stagedHead), [4, 5, 6]);
  const load = core.calls.find((call) => call[0] === 'llamadart_webgpu_decision_load');
  assert.equal(load[2], 'h'.repeat(256), 'the error label is capped');
  assert.equal(core.files.size, 0);
}

// Worker path: transfers a copy, rewrites handles, and keeps the worker on
// ordinary load errors.
{
  const workerCalls = [];
  const proxy = {};
  let failNext = null;
  const bridge = createWorkerBridge({
    _workerProxy: proxy,
    _callWorker: async (method, args, _onEvent, transferList = []) => {
      workerCalls.push({ method, args, transferList });
      if (failNext) {
        const error = failNext;
        failNext = null;
        throw error;
      }
      if (method === 'loadDecisionHead') {
        return { apiVersion: 1, handle: 41, hiddenSize: 768 };
      }
      if (method === 'runDecision') {
        return [{ logits: new Float32Array([1]), actLogits: new Float32Array([2]) }];
      }
      return undefined;
    },
  });

  const caller = new Uint8Array([9, 8, 7]);
  const info = await bridge.loadDecisionHead(caller, { configJson: '{}' });
  assert.equal(info.handle, 1);
  const load = workerCalls.at(-1);
  assert.notEqual(load.args[0].buffer, caller.buffer, 'the worker receives a copy');
  assert.deepEqual(load.transferList, [load.args[0].buffer]);
  assert.deepEqual(load.args[1], { configJson: '{}' });
  assert.equal(caller.byteLength, 3, "the caller's buffer is not detached");

  globalThis.window = { location: { href: 'https://example.test/app/index.html' } };
  try {
    await bridge.loadDecisionHead('heads/laya.safetensors');
  } finally {
    delete globalThis.window;
  }
  assert.equal(
    workerCalls.at(-1).args[0],
    'https://example.test/app/heads/laya.safetensors',
    'relative URLs resolve against the page, not the worker script',
  );

  await bridge.runDecision(1, [{ tokens: [1], markers: [0], questionType: 1 }]);
  assert.equal(workerCalls.at(-1).method, 'runDecision');
  assert.equal(workerCalls.at(-1).args[0], 41, 'worker calls use the worker handle');

  const callsBeforeInvalidRun = workerCalls.length;
  await assert.rejects(
    bridge.runDecision(1, [{ tokens: [0.5], markers: [0], questionType: 1 }]),
    /tokens\[0\] is 0\.5; expected a 32-bit integer/,
  );
  assert.equal(workerCalls.length, callsBeforeInvalidRun, 'malformed input never reaches the worker');

  failNext = new Error('Invalid safetensors file "x": header is not a JSON object.');
  await assert.rejects(bridge.loadDecisionHead('x.safetensors'), /Invalid safetensors file/);
  assert.equal(bridge._workerProxy, proxy, 'an ordinary load error keeps the worker');
}

// Worker path: head download progress events reach the caller.
{
  const bridge = createWorkerBridge({
    _callWorker: async (method, _args, onEvent) => {
      assert.equal(method, 'loadDecisionHead');
      onEvent({ event: 'progress', payload: { loaded: 1, total: 2 } });
      onEvent({ event: 'other', payload: {} });
      return { apiVersion: 1, handle: 5 };
    },
  });
  const progress = [];
  await bridge.loadDecisionHead('https://example.test/h.safetensors', {
    onProgress: (event) => progress.push(event),
  });
  assert.deepEqual(progress, [{ loaded: 1, total: 2 }]);
}

// Worker path: a worker failure during load, free or a capability probe moves
// the call to the main-thread runtime after reloading the model.
function createFallbackRuntime() {
  return {
    _modelBytes: 0,
    _runtimeNotes: [],
    loads: [],
    freed: [],
    async loadModelFromUrl() {
      this._modelBytes = 1;
    },
    getDecisionCapabilities() {
      return { apiVersion: 1, supported: true, from: 'runtime' };
    },
    async loadDecisionHead(source) {
      this.loads.push(source);
      return { apiVersion: 1, handle: 9, hiddenSize: 768 };
    },
    async freeDecisionHead(handle) {
      this.freed.push(handle);
    },
    async runDecision(handle) {
      return [{ logits: new Float32Array([handle]), actLogits: new Float32Array(0) }];
    },
  };
}

const workerFailure = () => new Error('Bridge worker request failed: worker terminated');

{
  const runtime = createFallbackRuntime();
  const bridge = createWorkerBridge({
    _workerProxy: { dispose: async () => {} },
    _createRuntime: () => runtime,
    _callWorker: async () => {
      throw workerFailure();
    },
  });
  const info = await bridge.loadDecisionHead('https://example.test/h.safetensors');
  assert.equal(bridge._workerProxy, null);
  assert.equal(runtime._modelBytes, 1, 'the model is reloaded on the main thread');
  assert.deepEqual(runtime.loads, ['https://example.test/h.safetensors'], 'the load is retried on the main thread');
  assert.equal(info.handle, 1);
  const outputs = await bridge.runDecision(info.handle, [{ tokens: [1], markers: [0], questionType: 0 }]);
  assert.deepEqual(Array.from(outputs[0].logits), [9], 'the facade handle resolves to the runtime head');
}

{
  const runtime = createFallbackRuntime();
  const bridge = createWorkerBridge({
    _workerProxy: { dispose: async () => {} },
    _createRuntime: () => runtime,
    _callWorker: async () => {
      throw workerFailure();
    },
  });
  const capabilities = await bridge.getDecisionCapabilities();
  assert.equal(capabilities.from, 'runtime');
  assert.equal(bridge._workerProxy, null);
  assert.equal(runtime._modelBytes, 1);
}

{
  const runtime = createFallbackRuntime();
  const proxy = { dispose: async () => {} };
  let failFree = false;
  const bridge = createWorkerBridge({
    _workerProxy: proxy,
    _createRuntime: () => runtime,
    _callWorker: async (method) => {
      if (method === 'loadDecisionHead') {
        return { apiVersion: 1, handle: 4 };
      }
      if (failFree) {
        throw workerFailure();
      }
      return undefined;
    },
  });
  const info = await bridge.loadDecisionHead(new Uint8Array([1]));
  failFree = true;
  await bridge.freeDecisionHead(info.handle);
  assert.equal(bridge._workerProxy, null, 'a failed worker free still falls back');
  assert.equal(runtime._modelBytes, 1);
  assert.deepEqual(runtime.freed, [], "the runtime never frees the worker's head");
  await assert.rejects(bridge.runDecision(info.handle, []), /Decision head 1 is not loaded/);
}

// Worker proxy: a head load is bounded by stalls between progress events, and
// a run's budget grows with its batch.
await withStubWorkerEnvironment(async ({ workers }) => {
  const bridge = createRealWorkerBridge();
  const proxy = bridge._workerProxy;
  assert.ok(proxy, 'the stub environment builds a real worker proxy');
  workerDriver(proxy, workers[0]).ready();
  const sequence = { tokens: [1], markers: [0], questionType: 0 };
  assert.equal(proxy._resolveRequestTimeoutMs('loadDecisionHead', ['h', {}]), 10 * 60 * 1000);
  assert.equal(proxy._resolveRequestTimeoutMs('runDecision', [1, []]), 10 * 60 * 1000);
  assert.equal(
    proxy._resolveRequestTimeoutMs('runDecision', [1, [sequence, sequence, sequence]]),
    10 * 60 * 1000 + 3 * 60 * 1000,
  );
  assert.equal(proxy._resolveRequestTimeoutMs('freeDecisionHead', [1]), 120000);
});

// Worker host: head loads post progress events, and run outputs are transferred.
{
  const originals = {
    self: globalThis.self,
    loadDecisionHead: LlamaWebGpuBridge.prototype.loadDecisionHead,
    runDecision: LlamaWebGpuBridge.prototype.runDecision,
  };
  const posted = [];
  globalThis.self = {
    postMessage(message, transfers = []) {
      posted.push({ message, transfers });
    },
  };
  let loadOptions = null;
  LlamaWebGpuBridge.prototype.loadDecisionHead = async function (_source, options) {
    loadOptions = options;
    options.onProgress({ loaded: 7, total: 9 });
    return { apiVersion: 1, handle: 2 };
  };
  LlamaWebGpuBridge.prototype.runDecision = async function () {
    return [
      { logits: new Float32Array([1, 2]), actLogits: new Float32Array([3]) },
      { logits: new Float32Array([4]), actLogits: new Float32Array([5, 6]) },
    ];
  };
  try {
    enableBridgeWorkerHost();
    await globalThis.self.onmessage({ data: { type: 'init', config: {} } });
    await globalThis.self.onmessage({
      data: { type: 'call', id: 1, method: 'loadDecisionHead', args: ['h.safetensors', { configJson: '{}' }] },
    });
    const loadMessages = posted.filter((entry) => entry.message.id === 1).map((entry) => entry.message);
    assert.deepEqual(loadMessages, [
      { type: 'event', id: 1, event: 'progress', payload: { loaded: 7, total: 9 } },
      { type: 'result', id: 1, value: { apiVersion: 1, handle: 2 } },
    ]);
    assert.equal(loadOptions.configJson, '{}');

    await globalThis.self.onmessage({
      data: { type: 'call', id: 2, method: 'runDecision', args: [2, []] },
    });
    const run = posted.find((entry) => entry.message.id === 2);
    assert.equal(run.message.type, 'result');
    const buffers = run.message.value.flatMap((output) => [output.logits.buffer, output.actLogits.buffer]);
    assert.equal(run.transfers.length, 4);
    assert.deepEqual(new Set(run.transfers), new Set(buffers), 'every output buffer is transferred');
  } finally {
    LlamaWebGpuBridge.prototype.loadDecisionHead = originals.loadDecisionHead;
    LlamaWebGpuBridge.prototype.runDecision = originals.runDecision;
    globalThis.self = originals.self;
  }
}

// Worker path: a failed worker loses its heads, and the model reloads on the
// main thread so later calls keep working.
{
  const runtime = {
    _modelBytes: 0,
    _runtimeNotes: [],
    async loadModelFromUrl() {
      this._modelBytes = 1;
    },
    async runDecision() {
      throw new Error('runtime must not run a head the worker owned');
    },
  };
  let bridge;
  bridge = createWorkerBridge({
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
  await assert.rejects(
    bridge.runDecision(1, [{ tokens: [1], markers: [0], questionType: 0 }]),
    /Decision head 1 was lost when the bridge worker failed \(.*worker terminated\)\. Load the decision head again\./,
  );
  assert.equal(bridge._workerProxy, null);
  assert.equal(bridge._runtime, runtime);
  assert.equal(runtime._modelBytes, 1, 'the model is reloaded on the main thread');
  await assert.rejects(bridge.runDecision(1, []), /Decision head 1 is not loaded/);
}

console.log('Decision bridge contract tests passed');
