// State and error shapes shared by the worker host and the bridge facade.

import type { LlamaWebGpuBridge, LlamaWebGpuBridgeConfig, ModelMetadata } from './llama_webgpu_bridge.d.ts';

// Model state a worker reports so the facade can answer synchronous getters.
export interface BridgeStateSnapshot {
  metadata: ModelMetadata | null;
  contextSize: number;
  gpuActive: boolean;
  backendName: string;
  supportsVision: boolean;
  supportsAudio: boolean;
}

// A message from the worker host: a result, an error, or a progress/token event.
export interface WorkerResponse {
  type: string;
  id?: number;
  value?: unknown;
  state?: BridgeStateSnapshot;
  message?: string;
  event?: string;
  payload?: unknown;
}

export type WorkerEventHandler = (message: WorkerResponse) => void;

// A message to the worker host: `init` with the bridge config, or a `call`
// of a bridge method by name. Its fields come from another thread, so the
// host validates them rather than trusting these types.
export interface WorkerRequest {
  type?: string;
  id?: number;
  config?: LlamaWebGpuBridgeConfig;
  method?: unknown;
  args?: unknown;
}

// A failed worker request: the worker's state at the failure, or a flag that
// marks the worker itself as crashed or stalled.
export interface WorkerRequestError extends Error {
  state?: BridgeStateSnapshot;
  llamadartWorkerCrash?: boolean;
  llamadartWorkerTimeout?: boolean;
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
