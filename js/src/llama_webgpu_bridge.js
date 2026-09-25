// Public entry point of the WebGPU bridge. Bundled by scripts/build_js_bridge.mjs.
// Modules under js/src/ are side-effect free; this file owns the load-time effects.

import { LlamaWebGpuBridge } from './bridge.ts';
import {
  enableBridgeWorkerHost,
  installBridgeWorkerHost,
  shouldAutoBootBridgeWorkerHost,
} from './worker_host.ts';

if (shouldAutoBootBridgeWorkerHost()) {
  installBridgeWorkerHost();
}

if (typeof window !== 'undefined') {
  const browserWindow = /** @type {Window & typeof globalThis & { LlamaWebGpuBridge?: typeof LlamaWebGpuBridge }} */ (window);
  if (!browserWindow.LlamaWebGpuBridge) {
    browserWindow.LlamaWebGpuBridge = LlamaWebGpuBridge;
  }
}

export { LlamaWebGpuBridge, enableBridgeWorkerHost };
