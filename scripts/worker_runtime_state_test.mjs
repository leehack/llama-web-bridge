import assert from 'node:assert/strict';

import {
  LlamaWebGpuBridge,
  enableBridgeWorkerHost,
} from '../js/src/llama_webgpu_bridge.js';

const posted = [];
const originalSelf = globalThis.self;
const originalMethods = {
  createCompletion: LlamaWebGpuBridge.prototype.createCompletion,
  synthesizeSpeech: LlamaWebGpuBridge.prototype.synthesizeSpeech,
  getModelMetadata: LlamaWebGpuBridge.prototype.getModelMetadata,
  getContextSize: LlamaWebGpuBridge.prototype.getContextSize,
  isGpuActive: LlamaWebGpuBridge.prototype.isGpuActive,
  getBackendName: LlamaWebGpuBridge.prototype.getBackendName,
  supportsVision: LlamaWebGpuBridge.prototype.supportsVision,
  supportsAudio: LlamaWebGpuBridge.prototype.supportsAudio,
};

try {
  globalThis.self = {
    postMessage(message, transfers = []) {
      posted.push({ message, transfers });
    },
  };

  LlamaWebGpuBridge.prototype.getModelMetadata = function getModelMetadata() {
    return { diagnostic: this.__diagnostic || 'initial' };
  };
  LlamaWebGpuBridge.prototype.getContextSize = () => 4096;
  LlamaWebGpuBridge.prototype.isGpuActive = () => false;
  LlamaWebGpuBridge.prototype.getBackendName = () => 'WASM test runtime';
  LlamaWebGpuBridge.prototype.supportsVision = () => true;
  LlamaWebGpuBridge.prototype.supportsAudio = () => true;
  LlamaWebGpuBridge.prototype.createCompletion = async function createCompletion() {
    this.__diagnostic = 'generation_stopped_context_limit';
    return 'completion';
  };
  LlamaWebGpuBridge.prototype.synthesizeSpeech = async function synthesizeSpeech() {
    this.__diagnostic = 'tts_runtime_diagnostic';
    return {
      pcm: new Float32Array([0.25]),
      sampleRate: 24000,
      channels: 1,
    };
  };

  enableBridgeWorkerHost();
  await globalThis.self.onmessage({ data: { type: 'init', config: {} } });
  assert.equal(posted.at(-1)?.message?.type, 'ready');

  await globalThis.self.onmessage({
    data: { type: 'call', id: 1, method: 'createCompletion', args: ['prompt', {}] },
  });
  const completionResponse = posted.at(-1);
  assert.equal(completionResponse.message.value, 'completion');
  assert.equal(
    completionResponse.message.state.metadata.diagnostic,
    'generation_stopped_context_limit',
  );

  await globalThis.self.onmessage({
    data: { type: 'call', id: 2, method: 'synthesizeSpeech', args: [{ text: 'Hello.' }] },
  });
  const speechResponse = posted.at(-1);
  assert.equal(speechResponse.message.state.metadata.diagnostic, 'tts_runtime_diagnostic');
  assert.deepEqual(speechResponse.transfers, [speechResponse.message.value.pcm.buffer]);

  const publicBridge = Object.create(LlamaWebGpuBridge.prototype);
  Object.assign(publicBridge, {
    _workerProxy: {
      async call(method) {
        return method === 'createCompletion' ? completionResponse.message : speechResponse.message;
      },
    },
    _metadata: {},
    _contextSize: 0,
    _gpuActive: true,
    _backendName: '',
    _supportsVision: false,
    _supportsAudio: false,
  });

  await publicBridge._callWorker('createCompletion', []);
  assert.equal(publicBridge._metadata.diagnostic, 'generation_stopped_context_limit');
  assert.equal(publicBridge._contextSize, 4096);
  assert.equal(publicBridge._backendName, 'WASM test runtime');
  assert.equal(publicBridge._supportsVision, true);
  assert.equal(publicBridge._supportsAudio, true);

  await publicBridge._callWorker('synthesizeSpeech', []);
  assert.equal(publicBridge._metadata.diagnostic, 'tts_runtime_diagnostic');
} finally {
  globalThis.self = originalSelf;
  Object.assign(LlamaWebGpuBridge.prototype, originalMethods);
}

console.log('Worker runtime state propagation tests passed');
