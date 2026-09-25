// Loading the Emscripten core module factory.

function normalizeFactory(moduleExport) {
  if (typeof moduleExport === 'function') {
    return moduleExport;
  }

  if (moduleExport && typeof moduleExport.default === 'function') {
    return moduleExport.default;
  }

  if (moduleExport && typeof moduleExport.createLlamaWebGpuCoreModule === 'function') {
    return moduleExport.createLlamaWebGpuCoreModule;
  }

  throw new Error('Unable to resolve llama_webgpu_core factory function');
}

export async function importCoreFactory(moduleUrl) {
  const exportedModule = await import(moduleUrl);
  return normalizeFactory(exportedModule);
}
