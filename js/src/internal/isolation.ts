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

export const CROSS_ORIGIN_ISOLATION_MESSAGE =
  'The llama.cpp web core needs shared memory, which this page cannot use because it is not '
  + 'cross-origin isolated. Serve the page with "Cross-Origin-Opener-Policy: same-origin" and '
  + '"Cross-Origin-Embedder-Policy: require-corp". "credentialless" does not isolate Safari.';
