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

// Appended to the core init timeout for a pthread core on a page that cannot
// post shared memory. It stays a hint on the timeout: a wrapper can still
// clear the pool's run dependency, so the refused pool alone does not mean
// init will fail.
export const PTHREAD_CORE_TIMEOUT_HINT =
  ' This core is built with pthreads, and the page cannot share memory with workers. Starting a pthread '
  + 'pool can stall initialization. Use the single-thread "_st" cores with "allowSingleThreadCore", '
  + 'or serve the page with "Cross-Origin-Opener-Policy: same-origin" and '
  + '"Cross-Origin-Embedder-Policy: require-corp".';

export function coreReportsPthreads(core: unknown): boolean {
  try {
    const supportsPthreads = (core as { _llamadart_webgpu_supports_pthreads?: () => unknown } | null)
      ?._llamadart_webgpu_supports_pthreads;
    return typeof supportsPthreads === 'function' && supportsPthreads.call(core) === 1;
  } catch (_) {
    return false;
  }
}
