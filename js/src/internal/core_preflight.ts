import type {
  CoreAssetErrorCode,
  CoreAssetErrorDetails,
  CoreAssetKind,
  CoreMemoryVariant,
} from '../llama_webgpu_bridge.d.ts';

const CORE_ASSET_UNAVAILABLE = 'CORE_ASSET_UNAVAILABLE';
export const CORE_PREFLIGHT_UNSUPPORTED = 'CORE_PREFLIGHT_UNSUPPORTED';
export const PREFLIGHT_DIAGNOSTIC_TIMEOUT_MS = 10_000;

export interface CoreAssetErrorPayload {
  code: CoreAssetErrorCode;
  asset: CoreAssetKind;
  url: string | null;
  httpStatus: number | null;
  variant: CoreMemoryVariant | null;
}

export function isCoreAssetErrorPayload(value: unknown): value is CoreAssetErrorPayload {
  if (!value || typeof value !== 'object') return false;
  const { code, asset, url, httpStatus, variant } = value as Partial<CoreAssetErrorPayload>;
  return (code === CORE_ASSET_UNAVAILABLE || code === CORE_PREFLIGHT_UNSUPPORTED)
    && (asset === 'loader' || asset === 'wasm' || asset === 'worker')
    && (url === null || typeof url === 'string')
    && (httpStatus === null || (typeof httpStatus === 'number'
      && Number.isInteger(httpStatus) && httpStatus >= 0 && httpStatus <= 599))
    && (variant === null || variant === 'wasm32' || variant === 'memory64');
}

export function redactCoreAssetUrl(url: string): string {
  try {
    const { protocol, origin, pathname } = new URL(url);
    if (protocol === 'http:' || protocol === 'https:') return origin + pathname;
    return `${protocol}//[redacted]`;
  } catch (_) {
    const protocol = /^[a-z][a-z\d+.-]*:/i.exec(url)?.[0];
    return protocol ? `${protocol.toLowerCase()}//[redacted]` : '[invalid-url]';
  }
}

function hasCoreAssetUrlSecrets(url: string): boolean {
  return /[?#]/.test(url) || /^(?:[a-z][a-z\d+.-]*:)?\/\/[^/]*@/i.test(url);
}

export function createCoreAssetUrlRedactor() {
  const replacements = new Map<string, string>();

  function remember(url: string, absoluteUrl: string) {
    const redacted = /^[a-z][a-z\d+.-]*:/i.test(absoluteUrl)
      ? redactCoreAssetUrl(absoluteUrl)
      : url.split(/[?#]/, 1)[0];
    for (const form of [url, absoluteUrl]) {
      if (hasCoreAssetUrlSecrets(form) && form !== redacted) replacements.set(form, redacted);
    }
  }

  function text(message: string) {
    for (const [url, redacted] of replacements) message = message.replaceAll(url, () => redacted);
    // Fetch errors can name a redirect destination instead of the requested URL.
    return message.replace(/\bhttps?:\/\/[^\s"'<>`‘’“”]+/gi, (match) => {
      if (!hasCoreAssetUrlSecrets(match)) return match;
      let end = match.search(/[),.;\]}]+$/);
      if (end === -1) return redactCoreAssetUrl(match);
      const prefix = match.slice(0, end);
      let pending = (prefix.match(/\(/g)?.length ?? 0) - (prefix.match(/\)/g)?.length ?? 0);
      // A balanced pair belongs to the URL. Leave the enclosing prose punctuation alone.
      while (pending-- > 0 && match[end] === ')') end++;
      return redactCoreAssetUrl(match.slice(0, end)) + match.slice(end);
    });
  }

  function error(failure: unknown, depth = 0, seen = new WeakSet<object>()): unknown {
    if (typeof failure === 'string') return text(failure);
    if (!failure || typeof failure !== 'object') return failure;
    // Returning the original tail would expose unsanitized errors past the bound.
    if (depth > 5 || seen.has(failure)) return undefined;
    const { message: originalMessage, stack: originalStack, cause: originalCause, name } = failure as Partial<Error>;
    if (typeof originalMessage !== 'string') return failure;
    seen.add(failure);
    const message = text(originalMessage);
    const stack = typeof originalStack === 'string' ? text(originalStack) : originalStack;
    const cause = error(originalCause, depth + 1, seen);
    if (message === originalMessage && stack === originalStack && cause === originalCause) return failure;

    const sanitized = typeof DOMException === 'function' && Object.prototype.toString.call(failure) === '[object DOMException]'
      ? new DOMException(message, name)
      : new Error(message);
    Object.setPrototypeOf(sanitized, Object.getPrototypeOf(failure));
    const descriptors = Object.getOwnPropertyDescriptors(failure);
    if (cause !== originalCause) {
      descriptors.cause = {
        value: cause, writable: true, configurable: true, enumerable: descriptors.cause?.enumerable ?? false,
      };
    }
    return Object.defineProperties(sanitized, {
      ...descriptors,
      message: { value: message, writable: true, configurable: true },
      stack: { value: stack, writable: true, configurable: true },
    });
  }

  return { remember, text, error };
}

export class CoreAssetError extends Error implements CoreAssetErrorPayload {
  declare readonly code: CoreAssetErrorCode;
  declare readonly asset: CoreAssetKind;
  declare readonly url: string | null;
  declare readonly httpStatus: number | null;
  declare readonly variant: CoreMemoryVariant | null;

  constructor(code: CoreAssetErrorCode, message: string, details: CoreAssetErrorDetails) {
    const url = details.url ? redactCoreAssetUrl(details.url) : null;
    super(details.url && url ? message.replaceAll(details.url, () => url) : message);
    this.name = 'CoreAssetError';
    this.code = code;
    this.asset = details.asset;
    this.url = url;
    this.httpStatus = details.httpStatus ?? null;
    this.variant = details.variant ?? null;
  }
}

export function resolveCoreInitTimeoutMs(timeout: number | undefined): number {
  const configured = Number(timeout);
  if (!Number.isFinite(configured) || configured <= 0) return 90_000;
  return Math.max(10_000, Math.min(600_000, Math.trunc(configured)));
}

export function hasMemory64Candidate(preferMemory64: boolean, loaderUrl: unknown): loaderUrl is string {
  return preferMemory64 && typeof loaderUrl === 'string' && loaderUrl.length > 0;
}

const JS_MIME = /^(?:application\/(?:x-)?(?:ecma|java)script|text\/(?:(?:x-)?(?:ecma|java)script|javascript1\.[0-5]|jscript|livescript))$/;

export async function probeUnavailableCoreAsset(
  url: string,
  asset: 'loader' | 'wasm',
  variant: CoreMemoryVariant,
): Promise<CoreAssetError | null> {
  if (!/^https?:\/\//i.test(url)) return null;

  const abort = new AbortController();
  const timer = globalThis.setTimeout(() => abort.abort(), PREFLIGHT_DIAGNOSTIC_TIMEOUT_MS);
  let httpStatus: number | null = null;
  try {
    const response = await fetch(url, { method: 'GET', credentials: 'same-origin', signal: abort.signal });
    // The headers suffice. A streaming response must not hold up the error.
    void response.body?.cancel().catch(() => {});
    const mime = response.headers.get('Content-Type')?.split(';', 1)[0]?.trim().toLowerCase() ?? '';
    const plausible = asset === 'loader' ? JS_MIME.test(mime) : mime !== 'text/html';
    if (response.ok && plausible) return null;
    httpStatus = response.status;
  } catch (_) {
    // Network policy, connection failures, and the diagnostic timeout have no HTTP status.
  } finally {
    globalThis.clearTimeout(timer);
  }

  const status = httpStatus === null ? '' : ` (HTTP ${httpStatus})`;
  return new CoreAssetError(
    CORE_ASSET_UNAVAILABLE,
    `Bridge core ${asset} unavailable${status}: ${redactCoreAssetUrl(url)}`,
    { asset, url, httpStatus, variant },
  );
}
