// Browser and runtime capability detection.

export function isSafariUserAgent(userAgent: unknown): boolean {
  if (typeof userAgent !== 'string' || userAgent.length === 0) {
    return false;
  }

  const hasSafariToken = /Safari\//.test(userAgent);
  const hasOtherBrowserToken = /(Chrome|Chromium|CriOS|Edg|OPR|Firefox|FxiOS)\//.test(userAgent);
  return hasSafariToken && !hasOtherBrowserToken;
}

// A pthread core preallocates workers and posts its shared memory to them.
// Browsers refuse that post (DataCloneError) unless the page is cross-origin
// isolated, and core init then never settles. Probing the post itself, rather
// than `crossOriginIsolated`, also accepts pages that enable shared memory
// another way, such as Chrome's SharedArrayBufferUnrestrictedAccessAllowed.
export function canPostSharedMemory(): boolean {
  if (typeof MessageChannel !== 'function' || typeof WebAssembly === 'undefined') {
    return true;
  }

  let channel: MessageChannel | null = null;
  try {
    channel = new MessageChannel();
    channel.port1.postMessage(new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true }));
    return true;
  } catch (_) {
    return false;
  } finally {
    channel?.port1.close();
    channel?.port2.close();
  }
}

export type WebGpuUnavailable = {
  reason: 'no_webgpu' | 'no_adapter' | 'adapter_missing_shader_f16' | 'unknown';
  adapter: string;
};

// ggml-webgpu registers its device only for an adapter with shader-f16 and
// logs nothing otherwise, so the page-side adapter is the only place that
// can say why a GPU load runs on the CPU.
export async function describeWebGpuUnavailable(): Promise<WebGpuUnavailable> {
  const gpu = (globalThis.navigator as { gpu?: { requestAdapter?: () => Promise<any> } } | undefined)?.gpu;
  if (!gpu || typeof gpu.requestAdapter !== 'function') {
    return { reason: 'no_webgpu', adapter: '' };
  }

  let adapter = null;
  try {
    adapter = await gpu.requestAdapter();
  } catch (_) {
    adapter = null;
  }
  if (!adapter) {
    return { reason: 'no_adapter', adapter: '' };
  }

  return {
    reason: adapter.features?.has?.('shader-f16') ? 'unknown' : 'adapter_missing_shader_f16',
    adapter: [adapter.info?.vendor, adapter.info?.architecture].filter(Boolean).join('/'),
  };
}
