// State and error shapes shared by the worker host and the bridge facade.

import type { LlamaWebGpuBridge, ModelMetadata } from './llama_webgpu_bridge.d.ts';

// Model state a worker reports so the facade can answer synchronous getters.
export interface BridgeStateSnapshot {
  metadata: ModelMetadata | null;
  contextSize: number;
  gpuActive: boolean;
  backendName: string;
  supportsVision: boolean;
  supportsAudio: boolean;
}

type BridgeStateSource = Pick<
  LlamaWebGpuBridge,
  | 'getModelMetadata'
  | 'getContextSize'
  | 'isGpuActive'
  | 'getBackendName'
  | 'supportsVision'
  | 'supportsAudio'
>;

export function serializeWorkerError(error: unknown): string {
  if (!error) {
    return 'Unknown worker error';
  }

  if (typeof error === 'string') {
    return error;
  }

  if (typeof (error as Partial<Error>).message === 'string' && (error as Error).message.length > 0) {
    return (error as Error).message;
  }

  try {
    return JSON.stringify(error);
  } catch (_) {
    return String(error);
  }
}

export const bridgeWorkerModeParam = '__llamadartBridgeWorker';

// The snapshot of a bridge that holds no model.
export function emptyBridgeState(): BridgeStateSnapshot {
  return {
    metadata: {},
    contextSize: 0,
    gpuActive: false,
    backendName: 'WASM (Prototype bridge)',
    supportsVision: false,
    supportsAudio: false,
  };
}

export function snapshotBridgeState(target: BridgeStateSource): BridgeStateSnapshot {
  return {
    metadata: target.getModelMetadata(),
    contextSize: target.getContextSize(),
    gpuActive: target.isGpuActive(),
    backendName: target.getBackendName(),
    supportsVision: target.supportsVision(),
    supportsAudio: target.supportsAudio(),
  };
}
