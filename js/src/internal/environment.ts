// Browser and runtime capability detection.

export function isSafariUserAgent(userAgent: unknown): boolean {
  if (typeof userAgent !== 'string' || userAgent.length === 0) {
    return false;
  }

  const hasSafariToken = /Safari\//.test(userAgent);
  const hasOtherBrowserToken = /(Chrome|Chromium|CriOS|Edg|OPR|Firefox|FxiOS)\//.test(userAgent);
  return hasSafariToken && !hasOtherBrowserToken;
}

export function isCrossOriginIsolatedRuntime(): boolean {
  try {
    if (typeof globalThis.crossOriginIsolated === 'boolean') {
      return globalThis.crossOriginIsolated;
    }
  } catch (_) {
    // ignore environment probing failures
  }

  // Assume isolated in runtimes that do not expose the signal.
  return true;
}
