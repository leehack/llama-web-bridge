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

export class CoreAssetError extends Error implements CoreAssetErrorPayload {
  declare readonly code: CoreAssetErrorCode;
  declare readonly asset: CoreAssetKind;
  declare readonly url: string | null;
  declare readonly httpStatus: number | null;
  declare readonly variant: CoreMemoryVariant | null;

  constructor(code: CoreAssetErrorCode, message: string, details: CoreAssetErrorDetails) {
    const url = details.url ? redactCoreAssetUrl(details.url) : null;
    super(details.url && url ? message.replaceAll(details.url, url) : message);
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
