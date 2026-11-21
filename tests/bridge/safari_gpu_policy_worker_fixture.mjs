import { isMainThread, parentPort, Worker, workerData } from 'node:worker_threads';

import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';
import { createStubCore } from './bridge_operation_queue_fixtures.mjs';

export function safariPolicyWorker(loadedLayers, workers, failReplacementCore) {
  return class {
    constructor() {
      this.thread = new Worker(new URL(import.meta.url), {
        workerData: { failCore: failReplacementCore && workers.length === 1 },
      });
      this.thread.on('message', (data) => {
        if (data.type === 'native-load') loadedLayers.push(data.layers);
        else if (this.dropNextResult && data.type === 'result') {
          this.dropNextResult = false;
          this.onerror?.({ message: 'Bridge worker crashed' });
        }
        else this.onmessage?.({ data });
      });
      this.thread.on('error', (error) => this.onerror?.({ error, message: error.message }));
      workers.push(this);
    }

    postMessage(...args) { this.thread.postMessage(...args); }
    terminate() { return this.thread.terminate(); }
  };
}

export function safariPolicyCore(onLoad) {
  const core = createStubCore({
    'user: hi\nassistant:': ['Hello there.'],
    'user: say hello in one short sentence\nassistant:': ['Hello there.'],
  });
  const ccall = core.ccall.bind(core);
  core.ccall = (name, type, types, args = []) => {
    if (name === 'llamadart_webgpu_probe') return 1;
    if (name === 'llamadart_webgpu_load_model') {
      onLoad(args[6]);
      return 0;
    }
    if (name === 'llamadart_webgpu_free_model') return 0;
    return ccall(name, type, types, args);
  };
  return core;
}

if (!isMainThread) {
  let trapOnTokenize = false;
  let failNextModelLoad = false;
  const core = safariPolicyCore((layers) => {
    parentPort.postMessage({ type: 'native-load', layers });
    if (failNextModelLoad) {
      failNextModelLoad = false;
      throw new Error('FS error');
    }
  });
  const ccall = core.ccall.bind(core);
  core.ccall = (name, ...args) => {
    if (trapOnTokenize && name.startsWith('llamadart_webgpu_tokenize')) {
      throw new WebAssembly.RuntimeError('unreachable');
    }
    return ccall(name, ...args);
  };

  globalThis.fetch = async () => new Response(Uint8Array.of(1, 2, 3));
  globalThis.self = { postMessage: (...args) => parentPort.postMessage(...args) };
  installBridgeWorkerHost();
  parentPort.on('message', (message) => {
    if (message.type === 'trap-on-tokenize') trapOnTokenize = true;
    else if (message.type === 'fail-next-model-load') failNextModelLoad = true;
    else if (message.type === 'init') {
      const coreModuleFactory = async () => {
        if (workerData.failCore) throw new Error('Script error loading core');
        return core;
      };
      void self.onmessage({ data: { ...message, config: { ...message.config, coreModuleFactory } } });
    } else {
      void self.onmessage({ data: message });
    }
  });
}
