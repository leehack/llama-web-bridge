import { isMainThread, parentPort, Worker as NodeWorker, workerData } from 'node:worker_threads';

import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';
import { createStubCore } from './bridge_operation_queue_fixtures.mjs';

export const PROMPT = 'user: hi\nassistant:';
export const COMPLETION = 'Hello there.';
const PROBE_PROMPTS = [PROMPT, 'user: say hello in one short sentence\nassistant:'];

const fixtureUrl = new URL(import.meta.url);
const fetchModel = async () => new Response(Uint8Array.of(1, 2, 3));
// (module (func (export "n") (param i32) (result i32) local.get 0))
const { instance: layerCountAbi } = await WebAssembly.instantiate(Uint8Array.of(
  0, 97, 115, 109, 1, 0, 0, 0, 1, 6, 1, 96, 1, 127, 1, 127, 3, 2, 1, 0,
  7, 5, 1, 1, 110, 0, 0, 10, 6, 1, 4, 0, 32, 0, 11,
));
let coreState;

export default async function createCore() {
  const { reportLoad, reportCompletion, failGpuDecode, corruptProbe } = coreState;
  const core = createStubCore();
  const ccall = core.ccall.bind(core);
  let loadedLayers;
  core.ccall = (name, type, types, args = []) => {
    if (name === 'llamadart_webgpu_probe') return 1;
    if (name === 'llamadart_webgpu_backends_json') return '["WebGPU","CPU"]';
    if (name === 'llamadart_webgpu_load_model' || name === 'llamadart_webgpu_load_model_from_url') {
      loadedLayers = layerCountAbi.exports.n(args[6]);
      const reply = corruptProbe && loadedLayers > 0 ? '<pad>'.repeat(48) : COMPLETION;
      for (const prompt of PROBE_PROMPTS) core.scripts[prompt] = [reply];
      reportLoad(loadedLayers);
      return 0;
    }
    if (name === 'llamadart_webgpu_free_model') return 0;
    if (name.startsWith('llamadart_webgpu_begin_generation')) reportCompletion?.();
    if (failGpuDecode && loadedLayers > 0) {
      if (name === 'llamadart_webgpu_next_token') return -1;
      if (name === 'llamadart_webgpu_last_error') return 'failed to decode';
    }
    if (coreState.trapTokenize && name.startsWith('llamadart_webgpu_tokenize')) {
      throw new WebAssembly.RuntimeError('unreachable');
    }
    return ccall(name, type, types, args);
  };
  return core;
}

if (!isMainThread) {
  globalThis.fetch = fetchModel;
  globalThis.self = { postMessage: (...args) => parentPort.postMessage(...args) };
  coreState = { ...workerData, reportLoad: (layers) => parentPort.postMessage({ type: 'nativeLoad', layers }) };
  installBridgeWorkerHost();
  parentPort.on('message', (data) => {
    if (data.type === 'trapTokenize') coreState.trapTokenize = true;
    else self.onmessage({ data });
  });
}

export async function withSafariBridge(config, run, { failGpuDecode = false, corruptProbe = false } = {}) {
  const originals = { Worker: globalThis.Worker, fetch: globalThis.fetch };
  const trace = [];
  const workers = [];
  const calls = { completion: 0 };
  coreState = {
    failGpuDecode, corruptProbe, reportLoad: (layers) => trace.push(layers),
    reportCompletion: () => { calls.completion += 1; },
  };
  globalThis.fetch = fetchModel;
  globalThis.Worker = class {
    constructor() {
      this.thread = new NodeWorker(fixtureUrl, { workerData: { failGpuDecode, corruptProbe } });
      this.thread.on('message', (data) => {
        if (data.type === 'nativeLoad') {
          trace.push(data.layers);
        } else if (this.crashOnResult && data.type === 'result') {
          this.crashOnResult = false;
          this.onerror?.({ message: 'Bridge worker crashed' });
        } else {
          this.onmessage?.({ data });
        }
      });
      this.thread.on('error', (error) => this.onerror?.({ error, message: error.message }));
      workers.push(this);
    }

    postMessage(...args) { this.thread.postMessage(...args); }
    terminate() { return this.thread.terminate(); }
  };

  const bridge = new LlamaWebGpuBridge({
    coreModuleUrl: fixtureUrl.href,
    coreModuleUrlMem64: fixtureUrl.href,
    workerRequestTimeoutMs: 5000,
    allowSingleThreadCore: true,
    preferMemory64: false,
    logLevel: 0,
    ...config,
  });
  try {
    return await run({ bridge, trace, workers, calls });
  } finally {
    try {
      await bridge.dispose();
    } finally {
      globalThis.fetch = originals.fetch;
      if (originals.Worker === undefined) delete globalThis.Worker;
      else globalThis.Worker = originals.Worker;
      coreState = undefined;
    }
  }
}
