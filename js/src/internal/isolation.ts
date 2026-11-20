// Raised before core init when the page cannot share memory with workers.
export const CROSS_ORIGIN_ISOLATION_REQUIRED = 'CROSS_ORIGIN_ISOLATION_REQUIRED';

export class CrossOriginIsolationError extends Error {
  declare code: typeof CROSS_ORIGIN_ISOLATION_REQUIRED;
  constructor(message: string = CROSS_ORIGIN_ISOLATION_MESSAGE) {
    super(message);
    this.name = 'CrossOriginIsolationError';
    this.code = CROSS_ORIGIN_ISOLATION_REQUIRED;
  }
}

export const SINGLE_THREAD_CORE_UNSUPPORTED = 'SINGLE_THREAD_CORE_UNSUPPORTED';

export class SingleThreadCoreError extends Error {
  readonly code = SINGLE_THREAD_CORE_UNSUPPORTED;
  readonly feature = 'remote_fetch_backend';

  constructor(message = 'Fetch-backed model loading requires a core built with pthreads') {
    super(message);
    this.name = 'SingleThreadCoreError';
  }
}

export const CROSS_ORIGIN_ISOLATION_MESSAGE =
  'The llama.cpp web core needs shared memory, which this page cannot use because it is not '
  + 'cross-origin isolated. Serve the page with "Cross-Origin-Opener-Policy: same-origin" and '
  + '"Cross-Origin-Embedder-Policy: require-corp". "credentialless" does not isolate Safari.';

export function appendCoreInitTimeoutHint(error: Error, moduleUrl: string | undefined): void {
  try {
    if (!moduleUrl || globalThis.crossOriginIsolated !== false) return;
    const { protocol, pathname } = new URL(moduleUrl, import.meta.url);
    if (protocol !== 'http:' && protocol !== 'https:' && protocol !== 'file:') return;
    if (!/\/llama_webgpu_core(?:_mem64)?\.js$/.test(pathname)) return;
    error.message += '. If this URL serves a pthread core, its worker pool can stall on a page without '
      + 'cross-origin isolation. Use the single-thread "_st" cores with "allowSingleThreadCore", or serve '
      + 'the page with "Cross-Origin-Opener-Policy: same-origin" and "Cross-Origin-Embedder-Policy: require-corp".';
  } catch (_) {
    // The diagnostic must preserve the timeout and fallback even if the environment is unavailable.
  }
}
