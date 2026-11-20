import assert from 'node:assert/strict';
import test from 'node:test';
import { LlamaWebGpuBridge, ModelUnloadedError, enableBridgeWorkerHost } from '../../js/src/llama_webgpu_bridge.js';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';

function check(label, condition, detail = '') {
  assert.ok(condition, detail ? `${label}: ${detail}` : label);
}

test('core retirement and unloaded errors cross the worker transport', async () => {
  const MODEL_BYTES = 'llamadart.webgpu.model_bytes';
  LlamaWebGpuBridgeRuntime.prototype.loadModelFromUrl = async function loadModelFromUrl() {
    await this._ensureCore();
    this._modelBytes = 1;
    return 1;
  };
  LlamaWebGpuBridgeRuntime.prototype.getModelMetadata = function getModelMetadata() {
    return this._modelBytes > 0 ? { [MODEL_BYTES]: '1024' } : {};
  };

  const nativeCalls = [];
  let trapping = false;
  const trapError = new WebAssembly.RuntimeError('memory access out of bounds');
  let retireHookCalls = 0;
  const fakeCore = {
    HEAPU8: new Uint8Array(64),
    llamadartRetire() {
      retireHookCalls += 1;
    },
    ccall(name, returnType) {
      nativeCalls.push(name);
      if (trapping && name.startsWith('llamadart_webgpu_tokenize')) {
        throw trapError;
      }
      return returnType === 'string' ? '[1,2]' : 0;
    },
  };

  // The worker host posts through the global `self`; record the wire order.
  const wire = [];
  const workers = [];
  globalThis.self = {
    postMessage(message) {
      wire.push(message.type === 'error' ? `error:${message.errorCode ?? ''}` : message.type);
      const worker = workers[0];
      queueMicrotask(() => worker.onmessage?.({ data: message }));
    },
  };
  enableBridgeWorkerHost();

  // Phase A: the host alone. After the trap, the same inner facade is asked again.
  {
    const replies = [];
    workers.push({ onmessage: (event) => replies.push(event.data) });
    const send = async (message) => {
      await globalThis.self.onmessage({ data: message });
      await new Promise((resolve) => setTimeout(resolve, 0));
    };
    await send({ type: 'init', config: { coreInitTimeoutMs: 1000, coreModuleFactory: async () => fakeCore } });
    await send({ type: 'call', id: 1, method: 'loadModelFromUrl', args: ['https://example.invalid/model.gguf', {}] });
    trapping = true;
    wire.length = 0;
    await send({ type: 'call', id: 2, method: 'tokenize', args: ['hi'] });
    check('host posts core-retired before the error reply', wire.join(',') === 'core-retired,error:', wire.join(','));
    check('the error reply keeps the trap text', replies.at(-1)?.message === 'memory access out of bounds');
    const callsAtAbort = nativeCalls.length;
    await send({ type: 'call', id: 3, method: 'tokenize', args: ['hi'] });
    check('the next call on that worker carries the typed code',
      replies.at(-1)?.errorCode === 'MODEL_UNLOADED_AFTER_CORE_ABORT', JSON.stringify(replies.at(-1)));
    check('and makes no native call', nativeCalls.length === callsAtAbort);
    check('core-retired is posted once', wire.filter((entry) => entry === 'core-retired').length === 1, wire.join(','));
    trapping = false;
    retireHookCalls = 0;
    workers.length = 0;
  }

  globalThis.Worker = class StubWorker {
    constructor(url) {
      if (!String(url).startsWith('blob:')) {
        throw new Error('cross-origin worker blocked');
      }
      this.index = workers.length;
      this.terminated = 0;
      this.calls = [];
      workers.push(this);
    }

    postMessage(message) {
      if (this.terminated) {
        return;
      }
      if (this.index === 0) {
        const forwarded = message.type === 'init'
          ? { ...message, config: { ...message.config, coreModuleFactory: async () => fakeCore } }
          : message;
        queueMicrotask(() => globalThis.self.onmessage({ data: forwarded }));
        return;
      }
      // Replacement worker: scripted and healthy.
      queueMicrotask(() => {
        if (message.type === 'init') {
          this.onmessage({ data: { type: 'ready' } });
          return;
        }
        this.calls.push(message.method);
        const state = {
          metadata: { [MODEL_BYTES]: '1024' },
          contextSize: 128,
          gpuActive: false,
          backendName: 'WASM (Prototype bridge)',
          supportsVision: false,
          supportsAudio: false,
        };
        const value = message.method === 'tokenize' ? [5] : 1;
        this.onmessage({
          data: message.method === 'loadModelFromUrl'
            ? { type: 'result', id: message.id, value, state }
            : { type: 'result', id: message.id, value },
        });
      });
    }

    terminate() {
      this.terminated += 1;
    }
  };
  globalThis.URL.createObjectURL = () => 'blob:stub-worker';
  globalThis.URL.revokeObjectURL = () => {};

  const bridge = new LlamaWebGpuBridge({ coreInitTimeoutMs: 1000, workerUrl: 'https://example.invalid/llama_webgpu_bridge_worker.js' });
  const firstProxy = bridge._workerProxy;
  await bridge.loadModelFromUrl('https://example.invalid/model.gguf', {});
  check('model loaded through the real host', bridge.getModelMetadata()[MODEL_BYTES] === '1024');
  const healthy = await bridge.tokenize('hi');
  check('healthy call crosses the guard', Array.isArray(healthy) && healthy.length === 2, JSON.stringify(healthy));

  trapping = true;
  wire.length = 0;
  const callsBefore = nativeCalls.length;
  const failed = await bridge.tokenize('hi').then(() => null, (error) => error);
  const callsAtAbort = nativeCalls.length;
  check('integrated: core-retired precedes the error reply',
    wire.slice(0, 2).join(',') === 'core-retired,error:', wire.join(','));
  check('failing call rejects with the trap text as a plain Error',
    failed?.message === 'memory access out of bounds' && failed.constructor === Error, String(failed?.message));
  check('the trap reached the core exactly once', callsAtAbort - callsBefore === 1, String(callsAtAbort - callsBefore));
  check('retire hook ran once', retireHookCalls === 1, String(retireHookCalls));
  check('one worker replaced, old one terminated, no direct runtime',
    workers.length === 2 && workers[0].terminated === 1 && bridge._workerProxy !== firstProxy
    && bridge._runtime === null && bridge._workerFallbackReason === null,
    JSON.stringify({ workers: workers.length, terminated: workers[0].terminated }));
  check('facade still shows and remembers the model', bridge.getModelMetadata()[MODEL_BYTES] === '1024'
    && bridge._workerModelMissing === true);

  const next = await bridge.tokenize('hi');
  const third = await bridge.tokenize('hi');
  check('next call restores once on the replacement and succeeds',
    next[0] === 5 && third[0] === 5 && workers[1].calls.join(',') === 'loadModelFromUrl,tokenize,tokenize',
    workers[1].calls.join(','));
  check('no native call into the retired core after the abort', nativeCalls.length === callsAtAbort,
    `${nativeCalls.length - callsAtAbort} extra`);

  // The proxy rebuilds it.
  const typedBridge = new LlamaWebGpuBridge({ coreInitTimeoutMs: 1000, workerUrl: 'https://example.invalid/llama_webgpu_bridge_worker.js' });
  const typedWorker = workers.at(-1);
  typedWorker.postMessage = function record(message) {
    this.calls.push(message);
  };
  typedWorker.onmessage({ data: { type: 'ready' } });
  const pending = typedBridge._workerProxy.call('tokenize', ['hi']).then(() => null, (error) => error);
  await new Promise((resolve) => setTimeout(resolve, 0));
  const sent = typedWorker.calls.find((message) => message?.method === 'tokenize');
  typedWorker.onmessage({ data: { type: 'core-retired' } });
  typedWorker.onmessage({
    data: { type: 'error', id: sent.id, message: 'unloaded', errorCode: 'MODEL_UNLOADED_AFTER_CORE_ABORT' },
  });
  const rebuilt = await pending;
  check('the proxy rebuilds ModelUnloadedError and records core-retired',
    rebuilt instanceof ModelUnloadedError && rebuilt.code === 'MODEL_UNLOADED_AFTER_CORE_ABORT'
    && typedBridge._workerProxy.coreRetired === true, `${rebuilt?.name} ${rebuilt?.code}`);
});
