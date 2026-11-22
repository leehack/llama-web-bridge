// Loading the Emscripten core module factory.

import type { LlamaCoreModuleFactory } from '../llama_webgpu_bridge.d.ts';

// The shapes a dynamically imported core module may take.
type CoreModuleExport =
  | LlamaCoreModuleFactory
  | { default?: unknown; createLlamaWebGpuCoreModule?: unknown }
  | null
  | undefined;

export class MissingCoreFactoryExportError extends Error {}

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

  throw new MissingCoreFactoryExportError('Unable to resolve llama_webgpu_core factory function');
}

export function resolveCoreLoaderUrl(moduleUrl: string): string {
  if (!/^(?:\.{0,2}\/|[a-z][a-z\d+.-]*:)/i.test(moduleUrl)) return moduleUrl;
  const loaderBaseUrl = import.meta.url;
  try {
    return new URL(moduleUrl, loaderBaseUrl).href;
  } catch (_) {
    return moduleUrl;
  }
}

export async function importCoreFactory(moduleUrl: string): Promise<LlamaCoreModuleFactory> {
  const exportedModule = await import(/* @vite-ignore */ /* webpackIgnore: true */ moduleUrl);
  return normalizeFactory(exportedModule);
}
