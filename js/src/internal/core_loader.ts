// Loading the Emscripten core module factory.

import type { LlamaCoreModuleFactory } from '../llama_webgpu_bridge.d.ts';

// The shapes a dynamically imported core module may take.
type CoreModuleExport =
  | LlamaCoreModuleFactory
  | { default?: unknown; createLlamaWebGpuCoreModule?: unknown }
  | null
  | undefined;

function normalizeFactory(moduleExport: CoreModuleExport): LlamaCoreModuleFactory {
  if (typeof moduleExport === 'function') {
    return moduleExport;
  }

  if (moduleExport && typeof moduleExport.default === 'function') {
    return moduleExport.default as LlamaCoreModuleFactory;
  }

  if (moduleExport && typeof moduleExport.createLlamaWebGpuCoreModule === 'function') {
    return moduleExport.createLlamaWebGpuCoreModule as LlamaCoreModuleFactory;
  }

  throw new Error('Unable to resolve llama_webgpu_core factory function');
}

export async function importCoreFactory(moduleUrl: string): Promise<LlamaCoreModuleFactory> {
  const exportedModule = await import(moduleUrl);
  return normalizeFactory(exportedModule);
}
