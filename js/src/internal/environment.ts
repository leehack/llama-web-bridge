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
