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

export function appendCoreInitTimeoutHint(error: Error): void {
  try {
    if (globalThis.crossOriginIsolated !== false) return;
    error.message += '. If this core was built with pthreads, its worker pool may need cross-origin isolation. '
      + 'Serve the page with "Cross-Origin-Opener-Policy: same-origin" and "Cross-Origin-Embedder-Policy: require-corp", '
      + 'or use the single-thread "_st" cores with "allowSingleThreadCore".';
  } catch (_) {
    // The diagnostic must preserve the timeout and fallback even if the environment is unavailable.
  }
}
