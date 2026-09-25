// State and error shapes shared by the worker host and the bridge facade.

export function serializeWorkerError(error) {
  if (!error) {
    return 'Unknown worker error';
  }

  if (typeof error === 'string') {
    return error;
  }

  if (typeof error.message === 'string' && error.message.length > 0) {
    return error.message;
  }

  try {
    return JSON.stringify(error);
  } catch (_) {
    return String(error);
  }
}

export const bridgeWorkerModeParam = '__llamadartBridgeWorker';

// The snapshot of a bridge that holds no model.
export function emptyBridgeState() {
  return {
    metadata: {},
    contextSize: 0,
    gpuActive: false,
    backendName: 'WASM (Prototype bridge)',
    supportsVision: false,
    supportsAudio: false,
  };
}

export function snapshotBridgeState(target) {
  return {
    metadata: target.getModelMetadata(),
    contextSize: target.getContextSize(),
    gpuActive: target.isGpuActive(),
    backendName: target.getBackendName(),
    supportsVision: target.supportsVision(),
    supportsAudio: target.supportsAudio(),
  };
}
