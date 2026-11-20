// Starting workers from a same-origin blob: wrapper when their script is cross-origin.

export const BLOB_WORKER_BLOCKED = 'BLOB_WORKER_BLOCKED';
export type BlobWorkerBlockReason = 'csp' | 'unavailable' | 'script';
export type BlobWorkerTarget = 'bridge_worker' | 'pthread';
export type WorkerStartMode = 'direct' | 'blob';

export class BlobWorkerBlockedError extends Error {
  readonly code = BLOB_WORKER_BLOCKED;
  declare readonly reason: BlobWorkerBlockReason;
  declare readonly target: BlobWorkerTarget;

  constructor(reason: BlobWorkerBlockReason, target: BlobWorkerTarget, message?: string) {
    super(message ?? blobWorkerBlockedMessage(reason, target));
    this.name = 'BlobWorkerBlockedError';
    this.reason = reason;
    this.target = target;
  }
}

function blobWorkerBlockedMessage(reason: BlobWorkerBlockReason, target: BlobWorkerTarget): string {
  const what = target === 'pthread' ? 'The core pthread workers' : 'The bridge worker';
  const why = reason === 'csp'
    ? 'the Content Security Policy does not allow "worker-src blob:"'
    : reason === 'unavailable'
      ? 'this context cannot start module workers from blob: URLs'
      : 'a blob: worker could not import that script (check that "worker-src" allows its origin and that it is served with CORS)';
  return `${what} must start from a blob: URL for a cross-origin script, but ${why}.`;
}

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

function moduleBlobUrl(source: string): string {
  return URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
}

const pthreadBlobUrls = new Map<string, string>();

// Never revoked: the core starts pthread workers lazily for as long as it lives.
// The static import keeps the worker's message queue closed until the core has
// installed its handler; a dynamic import() drops the first message.
export function pthreadWorkerBlobUrl(absoluteCoreUrl: string): string {
  let url = pthreadBlobUrls.get(absoluteCoreUrl);
  if (!url) {
    url = moduleBlobUrl(`import ${JSON.stringify(absoluteCoreUrl)};\n`);
    pthreadBlobUrls.set(absoluteCoreUrl, url);
  }
  return url;
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

type ProbeResult = BlobWorkerBlockReason | null;

function runBlobWorkerProbe(importUrl: string, timeoutMs: number): Promise<ProbeResult> {
  return new Promise((resolve) => {
    const stopWatching = watchBlobWorkerCsp();
    let url: string | undefined;
    let worker: Worker | undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let settled = false;
    const finish = (failed: boolean) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      const cspBlocked = stopWatching();
      worker?.terminate();
      if (url) URL.revokeObjectURL(url);
      resolve(failed ? (cspBlocked ? 'csp' : importUrl ? 'script' : 'unavailable') : null);
    };

    try {
      const importSource = importUrl ? `import ${JSON.stringify(importUrl)};\n` : '';
      url = moduleBlobUrl(`${importSource}postMessage(0);`);
      worker = new Worker(url, { type: 'module' });
    } catch (_) {
      finish(true);
      return;
    }
    worker.onmessage = () => finish(false);
    // The CSP violation event may follow the worker's error event.
    worker.onerror = () => setTimeout(() => finish(true), 0);
    timer = setTimeout(() => finish(true), timeoutMs);
  });
}

const blobWorkerProbes = new Map<string, { ready: Promise<ProbeResult>; result?: ProbeResult }>();

export function probeBlobWorker(importUrl = '', timeoutMs = 5_000): Promise<ProbeResult> {
  const existing = blobWorkerProbes.get(importUrl);
  if (existing) return existing.ready;

  const deadline = performance.now() + timeoutMs;
  const probe = {
    result: undefined as ProbeResult | undefined,
    ready: (importUrl
      ? probeBlobWorker('', timeoutMs).then((blocked) => {
        if (blocked) return blocked;
        const remaining = deadline - performance.now();
        return remaining > 0 ? runBlobWorkerProbe(importUrl, remaining) : 'script';
      })
      : runBlobWorkerProbe('', timeoutMs)
    ).then((result) => {
      probe.result = result;
      return result;
    }),
  };
  blobWorkerProbes.set(importUrl, probe);
  return probe.ready;
}

export function recordBlobWorkerBlocked(reason: BlobWorkerBlockReason): void {
  blobWorkerProbes.set('', { result: reason, ready: Promise.resolve(reason) });
}

export function blobWorkerProbeResult(importUrl = ''): ProbeResult | undefined {
  return blobWorkerProbes.get('')?.result || blobWorkerProbes.get(importUrl)?.result;
}

function waitForPthreadProbe(scriptUrl: string, timeoutMs: number): Promise<ProbeResult | undefined> {
  // An in-flight probe shared with another bridge may have a longer budget.
  const probe = probeBlobWorker(scriptUrl, timeoutMs);
  let timer: ReturnType<typeof setTimeout> | undefined;
  return Promise.race([
    probe,
    new Promise<undefined>((resolve) => {
      timer = setTimeout(() => resolve(undefined), timeoutMs);
    }),
  ]).finally(() => clearTimeout(timer));
}

export class PthreadStartProbeRequired extends Error {
  declare readonly ready: Promise<ProbeResult | undefined>;

  constructor(ready: Promise<ProbeResult | undefined>) {
    super('The bridge is probing blob: worker support and will start the core again.');
    this.name = 'PthreadStartProbeRequired';
    this.ready = ready;
  }
}

export function pthreadScriptProperties(timeoutMs: number, onUse: () => void): PropertyDescriptorMap {
  let scriptUrl = '';
  return {
    // A redirect or re-export can load the factory from a different origin
    // than the caller's specifier. Only the core knows its worker script URL.
    [Symbol.for('llamadart.pthreadScriptUrl')]: {
      value(url: string) { scriptUrl = url; },
    },
    mainScriptUrlOrBlob: {
      configurable: true,
      enumerable: false,
      get() {
        if (!scriptUrl || !isCrossOriginScriptUrl(scriptUrl)) return undefined;
        const blocked = blobWorkerProbeResult(scriptUrl);
        if (blocked === undefined) {
          throw new PthreadStartProbeRequired(waitForPthreadProbe(scriptUrl, timeoutMs));
        }
        if (blocked) {
          throw new BlobWorkerBlockedError(blocked, 'pthread');
        }
        onUse();
        return pthreadWorkerBlobUrl(scriptUrl);
      },
    },
  };
}

export async function withPthreadStartProbe<T>(start: () => T | Promise<T>): Promise<T> {
  try {
    return await start();
  } catch (error) {
    if (!(error instanceof PthreadStartProbeRequired)) {
      throw error;
    }
    if (await error.ready === undefined) {
      const reason = blobWorkerProbeResult() === null ? 'script' : 'unavailable';
      throw new BlobWorkerBlockedError(reason, 'pthread');
    }
    return start();
  }
}
