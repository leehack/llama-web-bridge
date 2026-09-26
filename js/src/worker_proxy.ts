// Main-thread proxy that forwards bridge calls to a dedicated worker.

import { DECISION_WORKER_TIMEOUT_PER_SEQUENCE_MS } from './internal/decision.ts';
import type { LlamaWebGpuBridgeConfig } from './llama_webgpu_bridge.d.ts';
import { bridgeWorkerModeParam } from './worker_protocol.ts';
import type { WorkerEventHandler, WorkerResponse } from './worker_protocol.ts';

// A request waiting for the worker's result.
interface PendingRequest {
  resolve: (message: WorkerResponse) => void;
  reject: (error: unknown) => void;
  onEvent: WorkerEventHandler;
}

function createBridgeWorkerSource(moduleUrl: string): string {
  return `import * as workerModule from ${JSON.stringify(moduleUrl)};\nif (workerModule && typeof workerModule.enableBridgeWorkerHost === 'function') { workerModule.enableBridgeWorkerHost(); }\n`;
}

function resolveWorkerEntryUrl(moduleUrl: unknown): string | null {
  if (typeof moduleUrl !== 'string' || moduleUrl.length === 0) {
    return null;
  }

  try {
    const base = (typeof window !== 'undefined' && window.location?.href)
      ? window.location.href
      : undefined;
    const url = new URL(moduleUrl, base);
    const path = url.pathname || '';
    const usesDedicatedWorkerEntry = path.endsWith('_worker.js');
    if (!usesDedicatedWorkerEntry) {
      url.searchParams.set(bridgeWorkerModeParam, '1');
    }
    return url.toString();
  } catch (_) {
    return null;
  }
}

function deriveBridgeModuleUrlFromWorkerEntry(moduleUrl: unknown): string | null {
  if (typeof moduleUrl !== 'string' || moduleUrl.length === 0) {
    return null;
  }

  try {
    const base = (typeof window !== 'undefined' && window.location?.href)
      ? window.location.href
      : undefined;
    const url = new URL(moduleUrl, base);
    const path = url.pathname || '';
    if (!path.endsWith('_worker.js')) {
      return null;
    }

    url.pathname = path.replace(/_worker\.js$/, '.js');
    url.searchParams.delete(bridgeWorkerModeParam);
    return url.toString();
  } catch (_) {
    return null;
  }
}

export class BridgeWorkerProxy {
  declare _config: LlamaWebGpuBridgeConfig;
  declare _nextId: number;
  declare _pending: Map<number, PendingRequest>;
  declare _workerBlobUrl: string | null;
  declare _worker: Worker;
  declare _ready: Promise<void>;
  declare _readyResolve: () => void;
  declare _readyReject: (reason?: unknown) => void;
  declare _readyTimeoutHandle: ReturnType<typeof setTimeout> | null;

  constructor({ moduleUrl, config }: { moduleUrl: string; config: LlamaWebGpuBridgeConfig }) {
    this._config = config && typeof config === 'object' ? config : {};
    this._nextId = 1;
    this._pending = new Map();
    this._workerBlobUrl = null;

    let workerInitError: unknown = null;
    const moduleCandidates = [moduleUrl];
    const bridgeModuleFallback = deriveBridgeModuleUrlFromWorkerEntry(moduleUrl);
    if (bridgeModuleFallback && bridgeModuleFallback !== moduleUrl) {
      moduleCandidates.push(bridgeModuleFallback);
    }

    for (const candidate of moduleCandidates) {
      if (this._worker) {
        break;
      }

      const directWorkerUrl = resolveWorkerEntryUrl(candidate);
      if (!directWorkerUrl) {
        continue;
      }

      try {
        this._worker = new Worker(directWorkerUrl, { type: 'module' });
      } catch (error) {
        workerInitError = error;
      }
    }

    for (const candidate of moduleCandidates) {
      if (this._worker) {
        break;
      }

      const source = createBridgeWorkerSource(candidate);
      this._workerBlobUrl = URL.createObjectURL(
        new Blob([source], { type: 'text/javascript' }),
      );

      try {
        this._worker = new Worker(this._workerBlobUrl, { type: 'module' });
      } catch (error) {
        workerInitError = error;
        URL.revokeObjectURL(this._workerBlobUrl);
        this._workerBlobUrl = null;
      }
    }

    if (!this._worker) {
      throw workerInitError || new Error('Failed to initialize bridge worker');
    }

    this._ready = new Promise((resolve, reject) => {
      this._readyResolve = resolve;
      this._readyReject = reject;
    });
    this._readyTimeoutHandle = null;
    this._armReadyTimeout();

    this._worker.onmessage = (event: MessageEvent) => {
      const message: WorkerResponse = event.data || {};
      const type = message.type;
      if (type === 'ready') {
        this._clearReadyTimeout();
        this._readyResolve?.();
        return;
      }

      const id = Number(message.id || 0);
      const pending = this._pending.get(id);
      if (!pending) {
        return;
      }

      if (type === 'event') {
        pending.onEvent?.(message);
        return;
      }

      this._pending.delete(id);
      if (type === 'error') {
        const workerError: Error & { state?: unknown } = new Error(
          String(message.message || 'Worker request failed'),
        );
        if (message.state && typeof message.state === 'object') {
          workerError.state = message.state;
        }
        pending.reject(workerError);
        return;
      }

      pending.resolve(message);
    };

    this._worker.onerror = (event) => {
      const message = event?.message || 'Bridge worker crashed';
      // The uncaught error's text is arbitrary, so the flag is what marks the
      // worker itself as gone.
      const error: Error & { llamadartWorkerCrash?: boolean } = new Error(String(message));
      error.llamadartWorkerCrash = true;

      this._clearReadyTimeout();
      this._readyReject?.(error);

      for (const pending of this._pending.values()) {
        pending.reject(error);
      }
      this._pending.clear();
    };

    this._worker.postMessage({ type: 'init', config });
  }

  async call(
    method: string,
    args: unknown[],
    onEvent?: WorkerEventHandler | null,
    transferList: Transferable[] = [],
    operationMeta: Record<string, unknown> | null = null,
  ): Promise<WorkerResponse> {
    await this._ready;
    const id = this._nextId++;
    const timeoutMs = this._resolveRequestTimeoutMs(method, args);
    const transfers = Array.isArray(transferList)
      ? transferList.filter((item) => item != null)
      : [];

    return new Promise((resolve, reject) => {
      let timeoutHandle: ReturnType<typeof setTimeout> | null = null;
      const clearTimer = () => {
        if (timeoutHandle != null) {
          globalThis.clearTimeout(timeoutHandle);
          timeoutHandle = null;
        }
      };

      const armTimer = () => {
        clearTimer();
        if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
          return;
        }

        timeoutHandle = globalThis.setTimeout(() => {
          this._pending.delete(id);
          reject(new Error(`Worker request timeout (${method}, ${timeoutMs}ms)`));
        }, timeoutMs);
      };

      armTimer();
      this._pending.set(id, {
        resolve: (value: WorkerResponse) => {
          clearTimer();
          resolve(value);
        },
        reject: (error: unknown) => {
          clearTimer();
          reject(error);
        },
        onEvent: (event: WorkerResponse) => {
          armTimer();
          onEvent?.(event);
        },
      });
      const message = { type: 'call', id, method, args };
      if (operationMeta && typeof operationMeta === 'object') {
        Object.assign(message, operationMeta);
      }
      this._worker.postMessage(message, transfers);
    });
  }

  _resolveWorkerReadyTimeoutMs() {
    const configured = Number(this._config.workerInitTimeoutMs);
    if (!Number.isFinite(configured) || configured <= 0) {
      return 20000;
    }

    return Math.max(3000, Math.min(120000, Math.trunc(configured)));
  }

  _clearReadyTimeout() {
    if (this._readyTimeoutHandle != null) {
      globalThis.clearTimeout(this._readyTimeoutHandle);
      this._readyTimeoutHandle = null;
    }
  }

  _armReadyTimeout() {
    this._clearReadyTimeout();
    const timeoutMs = this._resolveWorkerReadyTimeoutMs();
    this._readyTimeoutHandle = globalThis.setTimeout(() => {
      this._readyTimeoutHandle = null;
      const timeoutError = new Error(`Bridge worker init timeout (${timeoutMs}ms)`);
      this._readyReject?.(timeoutError);
      try {
        this._worker?.terminate();
      } catch (_) {
        // best-effort termination only
      }
    }, timeoutMs);
  }

  _resolveRequestTimeoutMs(method: string, args: unknown[] = []): number {
    const explicitGlobal = Number(this._config.workerRequestTimeoutMs);
    const clamp = (value: number, fallback: number): number => {
      if (!Number.isFinite(value) || value <= 0) {
        return fallback;
      }
      return Math.max(5000, Math.min(3600000, Math.trunc(value)));
    };

    if (method === 'loadModelFromUrl' || method === 'loadDraftModel') {
      return clamp(Number(this._config.workerModelLoadTimeoutMs), clamp(explicitGlobal, 3 * 60 * 1000));
    }

    if (method === 'loadMultimodalProjector') {
      return clamp(Number(this._config.workerMmprojLoadTimeoutMs), clamp(explicitGlobal, 8 * 60 * 1000));
    }

    if (method === 'createCompletion') {
      return clamp(Number(this._config.workerCompletionTimeoutMs), clamp(explicitGlobal, 6 * 60 * 1000));
    }

    if (method === 'synthesizeSpeech') {
      return clamp(Number(this._config.workerTextToSpeechTimeoutMs), clamp(explicitGlobal, 20 * 60 * 1000));
    }

    if (method === 'loadDecisionHead' || method === 'loadLoraAdapter') {
      // Download progress events re-arm this timer, so it bounds a stall.
      return clamp(explicitGlobal, 10 * 60 * 1000);
    }

    if (method === 'runDecision') {
      // A run reports no progress, so its budget grows with the batch size.
      const sequences = Array.isArray(args?.[1]) ? args[1].length : 0;
      return clamp(explicitGlobal, 10 * 60 * 1000 + sequences * DECISION_WORKER_TIMEOUT_PER_SEQUENCE_MS);
    }

    return clamp(explicitGlobal, 120000);
  }

  async dispose() {
    this._clearReadyTimeout();
    let didTimeout = false;
    try {
      await Promise.race([
        this.call('dispose', []),
        new Promise((resolve) => {
          globalThis.setTimeout(() => {
            didTimeout = true;
            resolve(null);
          }, 800);
        }),
      ]);
    } catch (_) {
      // best-effort disposal
    }

    if (didTimeout) {
      // Worker became unresponsive; terminate below.
    }

    for (const pending of this._pending.values()) {
      pending.reject(new Error('Bridge worker disposed'));
    }
    this._pending.clear();

    this._worker.terminate();
    if (this._workerBlobUrl) {
      URL.revokeObjectURL(this._workerBlobUrl);
      this._workerBlobUrl = null;
    }
  }
}
