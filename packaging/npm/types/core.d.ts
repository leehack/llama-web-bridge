import type { LlamaCoreModuleFactory } from '../dist/llama_webgpu_bridge.d.ts';

declare const createLlamaWebGpuCoreModule: LlamaCoreModuleFactory;

export default createLlamaWebGpuCoreModule;
