// Worker-side message host that runs a bridge inside a dedicated worker.

import { LlamaWebGpuBridge } from './bridge.ts';
import { toUint8Array } from './internal/typed_values.ts';
import {
  bridgeWorkerModeParam,
  emptyBridgeState,
  serializeWorkerError,
  snapshotBridgeState,
} from './worker_protocol.ts';
import type { WorkerRequest } from './worker_protocol.ts';
import type {
  CompletionOptions,
  CompletionUsage,
  DecisionHeadOptions,
  DecisionSequence,
  DraftModelLoadOptions,
  LoadModelOptions,
  LoraAdapterLoadOptions,
  TextToSpeechOptions,
} from './llama_webgpu_bridge.d.ts';

const textDecoder = new TextDecoder();

let bridgeWorkerHostInstalled = false;

export function installBridgeWorkerHost() {
  if (bridgeWorkerHostInstalled) {
    return;
  }

  if (typeof self === 'undefined' || typeof self.postMessage !== 'function') {
    throw new Error('Bridge worker host can only run inside a worker context');
  }

  bridgeWorkerHostInstalled = true;
  let bridge: LlamaWebGpuBridge | null = null;

  const postError = (id: unknown, error: unknown) => {
    let state;
    try {
      state = bridge ? snapshotBridgeState(bridge) : undefined;
    } catch (_) {
      state = undefined;
    }

    self.postMessage({
      type: 'error',
      id,
      message: serializeWorkerError(error),
      state,
    });
  };

  self.onmessage = async (event: MessageEvent) => {
    const message: WorkerRequest = event.data || {};
    const type = message.type;
    const id = message.id ?? 0;

    try {
      if (type === 'init') {
        bridge = new LlamaWebGpuBridge({
          ...(message.config || {}),
          disableWorker: true,
        });
        self.postMessage({ type: 'ready' });
        return;
      }

      if (type !== 'call') {
        return;
      }

      if (!bridge) {
        throw new Error('Bridge worker is not initialized');
      }

      const method = String(message.method || '');
      const args: unknown[] = Array.isArray(message.args) ? message.args : [];

      if (method === 'loadModelFromUrl') {
        const url = args[0] as string | string[];
        const options: LoadModelOptions = (args[1] && typeof args[1] === 'object') ? { ...args[1] } : {};
        options.progressCallback = (progress: unknown) => {
          self.postMessage({ type: 'event', id, event: 'progress', payload: progress || {} });
        };

        const value = await bridge.loadModelFromUrl(url, options);
        self.postMessage({ type: 'result', id, value, state: snapshotBridgeState(bridge) });
        return;
      }

      if (method === 'loadDraftModel') {
        const options: DraftModelLoadOptions = (args[1] && typeof args[1] === 'object') ? { ...args[1] } : {};
        delete options.signal;
        options.progressCallback = (progress: unknown) => {
          self.postMessage({ type: 'event', id, event: 'progress', payload: progress || {} });
        };
        const value = await bridge.loadDraftModel(args[0] as string, options);
        self.postMessage({ type: 'result', id, value });
        return;
      }

      if (method === 'createCompletion') {
        const prompt = args[0] as string;
        const options: CompletionOptions = (args[1] && typeof args[1] === 'object') ? { ...args[1] } : {};
        delete options.signal;
        let usage = null as CompletionUsage | null;
        options.onUsage = (value: CompletionUsage) => {
          usage = value;
        };
        const tokenEventEncoding = typeof options.tokenEventEncoding === 'string'
          ? String(options.tokenEventEncoding || '').toLowerCase()
          : 'bytes';
        const flushMsRaw = Number(options.tokenEventFlushMs);
        const tokenEventFlushMs = Number.isFinite(flushMsRaw) && flushMsRaw >= 0
          ? Math.max(0, Math.min(200, Math.trunc(flushMsRaw)))
          : 0;
        const flushCharsRaw = Number(options.tokenEventFlushChars);
        const tokenEventFlushChars = Number.isFinite(flushCharsRaw) && flushCharsRaw > 0
          ? Math.max(1, Math.min(1024, Math.trunc(flushCharsRaw)))
          : 0;
        const shouldEmitCurrentText = options.emitCurrentTextOnToken === true;

        let pendingPieceText = '';
        let pendingPieceBytes: Uint8Array[] = [];
        let pendingPieceByteLength = 0;
        let pendingPieceCharLength = 0;
        let pendingPieceCharDecoder = tokenEventFlushChars > 0
          ? new TextDecoder()
          : null;
        let pendingCurrentText = '';
        let flushTimer: ReturnType<typeof setTimeout> | null = null;

        const flushTokenTextPayload = () => {
          if (pendingPieceText.length === 0) {
            return;
          }

          self.postMessage({
            type: 'event',
            id,
            event: 'token',
            payload: {
              pieceText: pendingPieceText,
              currentText: shouldEmitCurrentText ? pendingCurrentText : '',
            },
          });
          pendingPieceText = '';
          pendingCurrentText = '';
        };

        const flushTokenBytePayload = () => {
          if (pendingPieceByteLength === 0) {
            return;
          }

          const piece = new Uint8Array(pendingPieceByteLength);
          let offset = 0;
          for (const chunk of pendingPieceBytes) {
            piece.set(chunk, offset);
            offset += chunk.byteLength;
          }

          self.postMessage({
            type: 'event',
            id,
            event: 'token',
            payload: {
              piece: Array.from(piece),
              currentText: shouldEmitCurrentText ? pendingCurrentText : '',
            },
          });
          pendingPieceBytes = [];
          pendingPieceByteLength = 0;
          pendingPieceCharLength = 0;
          pendingCurrentText = '';
        };

        const flushTokenPayload = () => {
          if (tokenEventEncoding === 'text') {
            flushTokenTextPayload();
            return;
          }
          flushTokenBytePayload();
        };

        const scheduleTokenFlush = () => {
          if (tokenEventFlushMs <= 0 || flushTimer != null) {
            return;
          }

          flushTimer = globalThis.setTimeout(() => {
            flushTimer = null;
            flushTokenPayload();
          }, tokenEventFlushMs);
        };

        options.onToken = (piece: unknown, currentText: unknown) => {
          if (tokenEventEncoding === 'text') {
            const pieceText = typeof piece === 'string'
              ? piece
              : textDecoder.decode(toUint8Array(piece) || new Uint8Array());
            if (pieceText.length === 0) {
              return;
            }

            if (tokenEventFlushMs > 0) {
              pendingPieceText += pieceText;
              if (shouldEmitCurrentText) {
                pendingCurrentText = String(currentText || '');
              }

              if (tokenEventFlushChars > 0 && pendingPieceText.length >= tokenEventFlushChars) {
                if (flushTimer != null) {
                  globalThis.clearTimeout(flushTimer);
                  flushTimer = null;
                }
                flushTokenTextPayload();
                return;
              }

              scheduleTokenFlush();
              return;
            }

            self.postMessage({
              type: 'event',
              id,
              event: 'token',
              payload: {
                pieceText,
                currentText: shouldEmitCurrentText ? String(currentText || '') : '',
              },
            });
            return;
          }

          const normalizedPieceBytes = toUint8Array(piece) || new Uint8Array();
          if (normalizedPieceBytes.byteLength === 0) {
            return;
          }

          if (tokenEventFlushMs > 0) {
            const pieceBytes = Array.isArray(piece)
              ? normalizedPieceBytes
              : Uint8Array.from(normalizedPieceBytes);
            pendingPieceBytes.push(pieceBytes);
            pendingPieceByteLength += pieceBytes.byteLength;
            if (pendingPieceCharDecoder != null) {
              pendingPieceCharLength += pendingPieceCharDecoder.decode(
                pieceBytes,
                { stream: true },
              ).length;
            }
            if (shouldEmitCurrentText) {
              pendingCurrentText = String(currentText || '');
            }

            if (
              tokenEventFlushChars > 0
              && pendingPieceCharLength >= tokenEventFlushChars
            ) {
              if (flushTimer != null) {
                globalThis.clearTimeout(flushTimer);
                flushTimer = null;
              }
              flushTokenBytePayload();
              return;
            }

            scheduleTokenFlush();
            return;
          }

          self.postMessage({
            type: 'event',
            id,
            event: 'token',
            payload: {
              piece: Array.from(normalizedPieceBytes),
              currentText: shouldEmitCurrentText ? String(currentText || '') : '',
            },
          });
        };

        let value;
        try {
          value = await bridge.createCompletion(prompt, options);
        } finally {
          if (flushTimer != null) {
            globalThis.clearTimeout(flushTimer);
            flushTimer = null;
          }
          flushTokenPayload();
        }
        flushTokenTextPayload();
        if (usage != null) {
          self.postMessage({ type: 'event', id, event: 'usage', payload: usage });
        }
        self.postMessage({ type: 'result', id, value, state: snapshotBridgeState(bridge) });
        return;
      }

      if (method === 'loadMultimodalProjector') {
        const value = await bridge.loadMultimodalProjector(args[0] as string);
        self.postMessage({ type: 'result', id, value, state: snapshotBridgeState(bridge) });
        return;
      }

      if (method === 'synthesizeSpeech') {
        const options: Partial<TextToSpeechOptions> = (args[0] && typeof args[0] === 'object') ? { ...args[0] } : {};
        delete options.signal;
        options.onProgress = (progress: unknown) => {
          self.postMessage({ type: 'event', id, event: 'progress', payload: progress || {} });
        };
        const value = await bridge.synthesizeSpeech(options);
        const transfers = value?.pcm?.buffer instanceof ArrayBuffer
          ? [value.pcm.buffer]
          : [];
        self.postMessage(
          { type: 'result', id, value, state: snapshotBridgeState(bridge) },
          transfers,
        );
        return;
      }

      if (method === 'loadDecisionHead') {
        const options: DecisionHeadOptions = (args[1] && typeof args[1] === 'object') ? { ...args[1] } : {};
        options.onProgress = (progress: unknown) => {
          self.postMessage({ type: 'event', id, event: 'progress', payload: progress || {} });
        };
        const value = await bridge.loadDecisionHead(args[0] as string | ArrayBuffer | ArrayBufferView, options);
        self.postMessage({ type: 'result', id, value });
        return;
      }

      if (method === 'loadLoraAdapter') {
        const options: LoraAdapterLoadOptions = (args[1] && typeof args[1] === 'object') ? { ...args[1] } : {};
        delete options.signal;
        options.progressCallback = (progress: unknown) => {
          self.postMessage({ type: 'event', id, event: 'progress', payload: progress || {} });
        };
        const value = await bridge.loadLoraAdapter(args[0] as string | ArrayBuffer | ArrayBufferView, options);
        self.postMessage({ type: 'result', id, value });
        return;
      }

      if (method === 'runDecision') {
        const value = await bridge.runDecision(args[0] as number, args[1] as DecisionSequence[]);
        const transfers: ArrayBuffer[] = [];
        for (const output of Array.isArray(value) ? value : []) {
          for (const values of [output?.logits, output?.actLogits]) {
            const buffer = values?.buffer;
            if (buffer instanceof ArrayBuffer && !transfers.includes(buffer)) {
              transfers.push(buffer);
            }
          }
        }
        self.postMessage({ type: 'result', id, value }, transfers);
        return;
      }

      if (method === 'unloadMultimodalProjector') {
        const value = await bridge.unloadMultimodalProjector();
        self.postMessage({ type: 'result', id, value, state: snapshotBridgeState(bridge) });
        return;
      }

      if (method === 'stateSaveBytes') {
        const value = await bridge.stateSaveBytes(args[0] as number[] | undefined);
        const transfers = value && value.buffer instanceof ArrayBuffer
          ? [value.buffer]
          : [];
        self.postMessage({ type: 'result', id, value }, transfers);
        return;
      }

      if (method === 'stateLoadBytes') {
        const value = await bridge.stateLoadBytes(args[0] as Uint8Array, args[1] as number | undefined);
        self.postMessage({ type: 'result', id, value });
        return;
      }

      if (method === 'dispose') {
        const value = await bridge.dispose();
        self.postMessage({
          type: 'result',
          id,
          value,
          state: emptyBridgeState(),
        });
        return;
      }

      // Any other bridge method, called by name with the caller's arguments.
      const value = await (bridge as unknown as Record<string, (...callArgs: unknown[]) => unknown>)[method](
        ...(args || []),
      );
      self.postMessage({ type: 'result', id, value });
    } catch (error) {
      postError(id, error);
    }
  };
}

export function shouldAutoBootBridgeWorkerHost() {
  if (typeof WorkerGlobalScope === 'undefined' || !(globalThis instanceof WorkerGlobalScope)) {
    return false;
  }

  try {
    const href = String(globalThis.location?.href || '');
    if (!href) {
      return false;
    }
    const url = new URL(href);
    return url.searchParams.get(bridgeWorkerModeParam) === '1';
  } catch (_) {
    return false;
  }
}

export function enableBridgeWorkerHost() {
  installBridgeWorkerHost();
}
