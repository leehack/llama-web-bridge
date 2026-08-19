import assert from 'node:assert/strict';

import { LlamaWebGpuBridge } from '../js/src/llama_webgpu_bridge.js';

function createBridge({ gpuLayers = 99, workerError = new Error('Aborted()') } = {}) {
  const calls = [];
  const warnings = [];
  const runtime = {
    _modelBytes: 0,
    _runtimeNotes: [],
    async loadModelFromUrl(url, options) {
      calls.push(['loadModelFromUrl', url, { ...options }]);
      this._modelBytes = 1;
      this._runtimeNotes = [];
    },
    async loadMultimodalProjector(url) {
      calls.push(['loadMultimodalProjector', url]);
    },
    async synthesizeSpeech(options) {
      calls.push(['synthesizeSpeech', options.text]);
      return { pcm: new Float32Array([0.25]), sampleRate: 24000, channels: 1 };
    },
  };
  const bridge = Object.create(LlamaWebGpuBridge.prototype);
  Object.assign(bridge, {
    _config: {},
    _runtime: runtime,
    _workerProxy: {
      async dispose() {
        calls.push(['disposeWorker']);
      },
    },
    _workerDisposePromise: null,
    _workerFallbackReason: null,
    _metadata: {},
    _loadedModelUrl: 'model.gguf',
    _loadedModelOptions: { nGpuLayers: gpuLayers, nCtx: 8192, nThreads: 8 },
    _loadedMmProjUrl: 'mmproj.gguf',
    _multimodalWorkerCpuMode: false,
    _bridgeWarnRecent: new Map(),
    _callWorker: async () => {
      throw workerError;
    },
    _emitBridgeWarn: (message) => warnings.push(message),
  });
  return { bridge, calls, runtime, warnings };
}

{
  const { bridge, calls, runtime, warnings } = createBridge();
  const result = await bridge.synthesizeSpeech({ text: 'Hello.' });

  assert.equal(result.sampleRate, 24000);
  assert.deepEqual(calls[0], ['disposeWorker']);
  assert.equal(calls[1][0], 'loadModelFromUrl');
  assert.equal(calls[1][2].nGpuLayers, 0);
  assert.equal(calls[1][2].nCtx, 4096);
  assert.equal(calls[1][2].nThreads, 4);
  assert.equal(calls[2][0], 'loadMultimodalProjector');
  assert.deepEqual(calls[3], ['synthesizeSpeech', 'Hello.']);
  assert.equal(bridge._workerProxy, null);
  assert.ok(
    warnings.some((message) => message.includes('CPU fallback after WebGPU failure')),
  );
  assert.ok(runtime._runtimeNotes.includes('worker_fallback_cpu_text_to_speech'));
}

{
  const { bridge, calls } = createBridge({ gpuLayers: 0 });
  await assert.rejects(bridge.synthesizeSpeech({ text: 'Hello.' }), /Aborted/);
  assert.equal(calls.length, 0);
  assert.notEqual(bridge._workerProxy, null);
}

{
  const cancelled = new DOMException('cancelled', 'AbortError');
  const { bridge, calls } = createBridge({ workerError: cancelled });
  await assert.rejects(
    bridge.synthesizeSpeech({ text: 'Hello.' }),
    (error) => error?.name === 'AbortError',
  );
  assert.equal(calls.length, 0);
  assert.notEqual(bridge._workerProxy, null);
}

console.log('Text-to-speech CPU recovery tests passed');
