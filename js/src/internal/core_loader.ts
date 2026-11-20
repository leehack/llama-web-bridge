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

// The URL import() loads for `moduleUrl`: resolved against this module and
// through the page's import map, like the import above.
export function resolveCoreModuleUrl(moduleUrl: string): string {
  try {
    const resolve = (import.meta as { resolve?: (specifier: string) => string }).resolve;
    return typeof resolve === 'function' ? resolve.call(import.meta, moduleUrl) : new URL(moduleUrl, import.meta.url).href;
  } catch (_) {
    return '';
  }
}
