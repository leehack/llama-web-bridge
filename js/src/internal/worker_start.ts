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

function runBlobWorkerProbe(importUrl: string): Promise<ProbeResult> {
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
    // Inconclusive: let the real start decide.
    timer = setTimeout(() => finish(false), 5000);
  });
}

const blobWorkerProbes = new Map<string, { ready: Promise<ProbeResult>; result?: ProbeResult }>();

export function probeBlobWorker(importUrl = ''): Promise<ProbeResult> {
  const existing = blobWorkerProbes.get(importUrl);
  if (existing) return existing.ready;

  const probe = {
    result: undefined as ProbeResult | undefined,
    ready: (importUrl
      ? probeBlobWorker().then((blocked) => blocked ?? runBlobWorkerProbe(importUrl))
      : runBlobWorkerProbe('')
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

export class PthreadStartProbeRequired extends Error {
  constructor() {
    super('The bridge is probing blob: worker support and will start the core again.');
    this.name = 'PthreadStartProbeRequired';
  }
}

// Emscripten reads the property synchronously inside allocateUnusedWorker(),
// so a core that starts no worker never reads it and never causes a probe.
export function pthreadScriptProperty(coreUrl: string, onUse: () => void): PropertyDescriptor {
  return {
    configurable: true,
    enumerable: false,
    get() {
      const blocked = blobWorkerProbeResult(coreUrl);
      if (blocked === undefined) {
        // Started here so that a later read finds a result even if nobody retries.
        void probeBlobWorker(coreUrl);
        throw new PthreadStartProbeRequired();
      }
      if (blocked) {
        throw new BlobWorkerBlockedError(blocked, 'pthread');
      }
      onUse();
      return pthreadWorkerBlobUrl(coreUrl);
    },
  };
}

export async function withPthreadStartProbe<T>(coreUrl: string, start: () => T | Promise<T>): Promise<T> {
  try {
    return await start();
  } catch (error) {
    if (!(error instanceof PthreadStartProbeRequired)) {
      throw error;
    }
    await probeBlobWorker(coreUrl);
    return start();
  }
}
