// Browser and runtime capability detection.

export function isSafariUserAgent(userAgent: unknown): boolean {
  if (typeof userAgent !== 'string' || userAgent.length === 0) {
    return false;
  }

  const hasSafariToken = /Safari\//.test(userAgent);
  const hasOtherBrowserToken = /(Chrome|Chromium|CriOS|Edg|OPR|Firefox|FxiOS)\//.test(userAgent);
  return hasSafariToken && !hasOtherBrowserToken;
}

// Every iOS browser runs on WebKit, whatever its product token says.
export function isWebKitUserAgent(userAgent: unknown): boolean {
  return typeof userAgent === 'string'
    && /AppleWebKit\//.test(userAgent)
    && !/(Chrome|Chromium|Edg|OPR)\//.test(userAgent);
}

// Safari 27 is the first version validated for full GPU offload.
const SAFARI_UNCAPPED_GPU_MIN_MAJOR = 27;

// The legacy detector also matches other WebKit browsers. Require Apple's
// platform and Safari's product sequence before enabling the new policy.
const APPLE_SAFARI_PLATFORM = /^Mozilla\/5\.0 \((?:Macintosh; Intel Mac OS X \d+(?:_\d+)*|(?:iPhone|iPad|iPod touch); CPU (?:iPhone )?OS \d+(?:_\d+)* like Mac OS X)\) AppleWebKit\/\d+(?:\.\d+)* \(KHTML, like Gecko\) /;
const SAFARI_PRODUCTS = /^(?:Version\/(\d+(?:\.\d+)*) )?(?:Mobile\/[A-Za-z0-9]+ )?Safari\/\d+(?:\.\d+)*$/;

function appleSafariProducts(userAgent: unknown): RegExpMatchArray | null {
  if (typeof userAgent !== 'string') return null;
  const platform = APPLE_SAFARI_PLATFORM.exec(userAgent);
  if (!platform) return null;
  const products = SAFARI_PRODUCTS.exec(userAgent.slice(platform[0].length));
  // WebKitGTK also uses this fixed version with a Macintosh platform quirk.
  if (products?.[1] === '60.5' && userAgent.endsWith('Safari/605.1.15')) return null;
  return products;
}

export function isAppleSafariUserAgent(userAgent: unknown): boolean {
  return appleSafariProducts(userAgent) !== null;
}

export function safariMajorVersion(userAgent: unknown): number | null {
  const version = appleSafariProducts(userAgent)?.[1];
  const major = Number(version?.split('.')[0]);
  return Number.isSafeInteger(major) ? major : null;
}

// An unknown version keeps the cap of the Safari releases it was added for.
export function safariDefaultMaxGpuLayers(userAgent: unknown): number {
  const major = safariMajorVersion(userAgent);
  return major !== null && major >= SAFARI_UNCAPPED_GPU_MIN_MAJOR ? Infinity : 1;
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

export function isMemory64Unsupported(): boolean {
  try {
    // A module with one unshared memory, a 64-bit address type, and zero initial pages.
    return typeof WebAssembly !== 'undefined'
      && WebAssembly.validate(Uint8Array.of(0, 97, 115, 109, 1, 0, 0, 0, 5, 3, 1, 4, 0)) === false;
  } catch (_) {
    return false;
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
