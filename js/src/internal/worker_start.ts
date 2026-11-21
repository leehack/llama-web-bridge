// Starting workers from a same-origin blob: wrapper where the page may not
// construct a Worker from the script's own URL.

export const BLOB_WORKER_BLOCKED = 'BLOB_WORKER_BLOCKED';
export type BlobWorkerBlockReason = 'csp' | 'unavailable' | 'script';
export type WorkerStartMode = 'direct' | 'blob';

export class BlobWorkerBlockedError extends Error {
  readonly code = BLOB_WORKER_BLOCKED;
  readonly target = 'pthread';
  declare readonly reason: BlobWorkerBlockReason;

  constructor(reason: BlobWorkerBlockReason, message?: string) {
    super(message ?? blobWorkerBlockedMessage(reason));
    this.name = 'BlobWorkerBlockedError';
    this.reason = reason;
  }
}

function blobWorkerBlockedMessage(reason: BlobWorkerBlockReason): string {
  const why = reason === 'csp'
    ? 'the Content Security Policy does not allow "worker-src blob:"'
    : reason === 'unavailable'
      ? 'this context cannot start module workers from blob: URLs'
      : 'the blob: worker could not load that script (check that "worker-src" allows its origin and that it is served with CORS)';
  return `The core pthread workers must start from a blob: URL for a cross-origin script, but ${why}.`;
}

// The browser's own rule for `new Worker(url)`: an http(s) script on another
// origin is refused (Chromium throws, Firefox fires an error event).
export function isCrossOriginScriptUrl(absoluteUrl: string): boolean {
  // location.origin is "null" even in a same-origin about:srcdoc frame.
  const origin = globalThis.origin;
  if (typeof origin !== 'string' || origin.length === 0) return false;
  try {
    const url = new URL(absoluteUrl);
    return (url.protocol === 'http:' || url.protocol === 'https:') && url.origin !== origin;
  } catch (_) {
    return false;
  }
}

export function watchBlobWorkerCsp(): () => boolean {
  let refused = false;
  const onViolation = (event: Event) => {
    const { blockedURI, disposition, effectiveDirective } = event as SecurityPolicyViolationEvent;
    if (blockedURI === 'blob' && disposition === 'enforce' && effectiveDirective === 'worker-src') {
      refused = true;
    }
  };
  globalThis.addEventListener?.('securitypolicyviolation', onViolation);
  return () => {
    globalThis.removeEventListener?.('securitypolicyviolation', onViolation);
    return refused;
  };
}

// Best effort: browsers do not say why a worker failed to start. Chromium
// cannot start a module blob worker from an opaque origin.
export function blobStartBlockReason(cspRefused: boolean, threwOnConstruct: boolean): BlobWorkerBlockReason {
  if (cspRefused) return 'csp';
  return threwOnConstruct || globalThis.origin === 'null' ? 'unavailable' : 'script';
}

// Firefox reports no violation inside a worker, so a refusal there looks like a
// script that failed to load. A blob worker with nothing to import tells them
// apart. It costs a worker, so only a start that has already failed asks.
const BLOB_START_CHECK_MS = 5000;

function blobWorkersStart(signal: AbortSignal): Promise<boolean> {
  return new Promise((resolve) => {
    let url: string | undefined;
    let worker: Worker | undefined;
    let finished = false;
    const finish = (started: boolean) => {
      if (finished) return;
      finished = true;
      globalThis.clearTimeout(timer);
      signal.removeEventListener('abort', abort);
      worker?.terminate();
      if (url !== undefined) URL.revokeObjectURL(url);
      resolve(started);
    };
    const abort = () => finish(false);
    const timer = globalThis.setTimeout(() => finish(false), BLOB_START_CHECK_MS);
    signal.addEventListener('abort', abort, { once: true });
    try {
      url = URL.createObjectURL(new Blob(['postMessage(0);\n'], { type: 'text/javascript' }));
      worker = new Worker(url, { type: 'module' });
      worker.addEventListener('message', () => finish(true), { once: true });
      worker.addEventListener('error', () => finish(false), { once: true });
    } catch (_) {
      finish(false);
    }
  });
}

type BlobWorkerBlockDiagnostic = BlobWorkerBlockReason | Promise<BlobWorkerBlockReason>;

const blockedStarts = new WeakMap<object, BlobWorkerBlockDiagnostic>();

export function markBlobStartBlocked(error: object, reason: BlobWorkerBlockDiagnostic): void {
  blockedStarts.set(error, reason);
}

export function blobStartBlockedReasonOf(error: unknown): BlobWorkerBlockDiagnostic | undefined {
  return error && typeof error === 'object' ? blockedStarts.get(error) : undefined;
}

const pthreadBlobUrls = new Map<string, string>();

// Never revoked: the core starts pthread workers lazily for as long as it lives.
// The static import keeps the worker's message queue closed until the core has
// installed its handler; a dynamic import() drops the first message.
function pthreadWorkerBlobUrl(absoluteScriptUrl: string): string {
  let url = pthreadBlobUrls.get(absoluteScriptUrl);
  if (!url) {
    url = URL.createObjectURL(new Blob([`import ${JSON.stringify(absoluteScriptUrl)};\n`], { type: 'text/javascript' }));
    pthreadBlobUrls.set(absoluteScriptUrl, url);
  }
  return url;
}

// The key under which a threaded core reads its pthread Worker constructor
// (src/llama_webgpu_core_pthread_pre.js).
export const PTHREAD_WORKER_HOOK = Symbol.for('llamadart.pthreadWorker');

export interface PthreadWorkerStart {
  Worker: new (url: string | URL, options?: WorkerOptions) => Worker;
  // Rejects with BlobWorkerBlockedError when a blob-started worker fails
  // before `settle()`.
  failed: Promise<never>;
  mode(): WorkerStartMode;
  // A recorded refusal can replace the original error without waiting.
  failure(error: unknown): unknown;
  checkNotAbandoned(): void;
  settle(adopted: boolean): void;
}

export function createPthreadWorkerStart(
  { pthreadScriptUrl }: { pthreadScriptUrl?: string } = {},
): PthreadWorkerStart {
  let mode: WorkerStartMode = 'direct';
  let settled = false;
  let abandoned = false;
  let refusing = false;
  let stopCspWatch: (() => boolean) | null = null;
  let refusalTimer: ReturnType<typeof globalThis.setTimeout> | undefined;
  let classifier: AbortController | undefined;
  let classifiedError: BlobWorkerBlockedError | undefined;
  const blobWorkers: Worker[] = [];
  let fail!: (error: BlobWorkerBlockedError) => void;
  const failed = new Promise<never>((_, reject) => { fail = reject; });
  failed.catch(() => {});

  const stopWatchingPool = (terminate: boolean) => {
    for (const worker of blobWorkers) {
      worker.removeEventListener('error', onWorkerError);
      if (terminate) worker.terminate();
    }
    blobWorkers.length = 0;
  };

  const recordFailure = (reason: BlobWorkerBlockReason) => {
    classifiedError = new BlobWorkerBlockedError(reason);
    fail(classifiedError);
  };

  const refuse = (threwOnConstruct: boolean) => {
    if (settled || refusing) return;
    refusing = true;
    const classify = () => {
      const cspRefused = stopCspWatch?.() ?? false;
      stopCspWatch = null;
      stopWatchingPool(true);
      return blobStartBlockReason(cspRefused, threwOnConstruct);
    };
    if (threwOnConstruct) {
      recordFailure(classify());
      return;
    }
    // The CSP violation event may follow the worker's error event.
    refusalTimer = globalThis.setTimeout(async () => {
      refusalTimer = undefined;
      const reason = classify();
      if (reason !== 'script') {
        recordFailure(reason);
        return;
      }
      classifier = new AbortController();
      const started = await blobWorkersStart(classifier.signal);
      classifier = undefined;
      if (!settled) recordFailure(started ? 'script' : 'unavailable');
    }, 0);
  };
  const onWorkerError = () => refuse(false);
  const checkNotAbandoned = () => {
    if (abandoned) throw new Error('Bridge core initialization was abandoned');
  };

  const PthreadWorker = function (url: string | URL, options?: WorkerOptions): Worker {
    let scriptUrl = String(url);
    if (pthreadScriptUrl && scriptUrl !== pthreadScriptUrl) {
      const stockName = /\/llama_webgpu_core(?:_mem64)?\.js$/.exec(scriptUrl)?.[0];
      if (stockName && scriptUrl === new URL(`.${stockName}`, pthreadScriptUrl).href) {
        // Emscripten asks for its original sibling filename even after bundling.
        url = scriptUrl = pthreadScriptUrl;
      }
    }
    if (!isCrossOriginScriptUrl(scriptUrl)) {
      return new Worker(url, options);
    }
    checkNotAbandoned();
    mode = 'blob';
    if (!settled) stopCspWatch ??= watchBlobWorkerCsp();
    let worker: Worker;
    try {
      worker = new Worker(pthreadWorkerBlobUrl(scriptUrl), options);
    } catch (error) {
      refuse(true);
      throw error;
    }
    if (!settled) {
      worker.addEventListener('error', onWorkerError, { once: true });
      blobWorkers.push(worker);
    }
    return worker;
  } as unknown as PthreadWorkerStart['Worker'];

  return {
    Worker: PthreadWorker,
    failed,
    mode: () => mode,
    failure: (error) => classifiedError ?? error,
    checkNotAbandoned,
    settle(adopted) {
      if (settled) return;
      settled = true;
      abandoned = !adopted;
      if (refusalTimer !== undefined) globalThis.clearTimeout(refusalTimer);
      classifier?.abort();
      stopCspWatch?.();
      stopCspWatch = null;
      stopWatchingPool(abandoned);
    },
  };
}
