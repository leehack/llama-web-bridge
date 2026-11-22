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

const HTTP_URL_START = /\bhttps?:\/\//gi;
const URL_TEXT_DELIMITER = /[\s"'<>`‘’“”]/;
const URL_PATH_CHARACTER = /[\p{L}\p{N}\p{M}._~%+\/-]/u;
const URL_QUERY_CONTINUATION = /[?&#=]/;
const URL_SENTENCE_BOUNDARY = /^[.,;:!?]+(?:$|[\s"'`‘’“”)\]}>])/;

function absoluteUrlEnd(message: string, start: number): number {
  const delimiter = message.slice(start).search(URL_TEXT_DELIMITER);
  return delimiter === -1 ? message.length : start + delimiter;
}

function trailingQueryPunctuation(token: string, queryStart: number, punctuation: string): string {
  const closingDelimiters: string[] = [];
  const suffixStart = token.length - punctuation.length;
  let trailing = '';
  for (let index = queryStart; index < token.length; index++) {
    const char = token[index];
    if (char === '[') closingDelimiters.push(']');
    else if (char === '(') closingDelimiters.push(')');
    else if (char === closingDelimiters.at(-1)) closingDelimiters.pop();
    else if (index >= suffixStart) trailing += char;
  }
  return trailing;
}

function queryFormatting(token: string, queryStart: number): string {
  const query = token.slice(queryStart);
  const punctuation = /[),.;!\]]+$/.exec(query)?.[0] ?? '';
  const position = /(?::\d+){1,2}$/.exec(query.slice(0, query.length - punctuation.length))?.[0] ?? '';
  const trailing = punctuation ? trailingQueryPunctuation(token, queryStart, punctuation) : '';
  return position + trailing;
}

function redactAbsoluteUrlToken(token: string): string {
  const queryStart = token.search(/[?#]/);
  const path = queryStart === -1 ? token : token.slice(0, queryStart);
  const url = path.replace(/(\bhttps?:\/\/)[A-Za-z0-9._~%!$&'*+=:-]+@(?![^/?#@,();|]*@)/gi, '$1');
  return queryStart === -1 ? url : url + queryFormatting(token, queryStart);
}

function atUrlStart(message: string, start: number): boolean {
  if (start === 0) return true;
  const previous = Array.from(message.slice(Math.max(0, start - 2), start)).at(-1)!;
  return !URL_PATH_CHARACTER.test(previous);
}

function atUrlEnd(message: string, end: number): boolean {
  const codePoint = message.codePointAt(end);
  if (codePoint === undefined) return true;
  const next = String.fromCodePoint(codePoint);
  return !(URL_PATH_CHARACTER.test(next) || URL_QUERY_CONTINUATION.test(next))
    || URL_SENTENCE_BOUNDARY.test(message.slice(end));
}

function replaceRegisteredUrl(message: string, url: string, redacted: string): string {
  return message.replaceAll(url, (match, index: number) => (
    atUrlStart(message, index) && atUrlEnd(message, index + match.length) ? redacted : match
  ));
}

export function createCoreAssetUrlRedactor() {
  const absoluteUrls = new Set<string>();
  const replacements = new Map<string, string>();

  function remember(url: string, absoluteUrl = url) {
    let serialized = absoluteUrl;
    try {
      serialized = new URL(absoluteUrl).href;
    } catch (_) {
      // Bare loader specifiers can still appear in diagnostics as written.
    }
    for (const form of [url, absoluteUrl, serialized]) {
      if (/^[a-z][a-z\d+.-]*:/i.test(form)) absoluteUrls.add(form);
    }
    if (/^[a-z][a-z\d+.-]*:/i.test(url)) return;
    const redacted = /^[a-z][a-z\d+.-]*:/i.test(absoluteUrl)
      ? redactCoreAssetUrl(absoluteUrl)
      : url.split(/[?#]/, 1)[0];
    if (hasCoreAssetUrlSecrets(url) && url !== redacted) replacements.set(url, redacted);
  }

  function knownHttpUrlAt(message: string, start: number) {
    if (!atUrlStart(message, start)) return null;
    for (const url of absoluteUrls) {
      if (!/^https?:\/\//i.test(url)) continue;
      const end = start + url.length;
      if (hasCoreAssetUrlSecrets(url) && message.startsWith(url, start)) {
        // Absolute queries can contain punctuation that delimits relative URLs in prose.
        const tail = message.slice(end, absoluteUrlEnd(message, end));
        if (queryFormatting(tail, 0) === tail) return { end, value: redactCoreAssetUrl(url) };
      }

      // Only the registered path can justify scanning past a literal quote to a new query or fragment.
      const path = url.split(/[?#]/, 1)[0];
      const queryStart = start + path.length;
      if (URL_TEXT_DELIMITER.test(path) && message.startsWith(path, start)
        && (message[queryStart] === '?' || message[queryStart] === '#')) {
        const end = absoluteUrlEnd(message, queryStart);
        const token = message.slice(start, end);
        return { end, value: redactCoreAssetUrl(url) + queryFormatting(token, path.length) };
      }
    }
    return null;
  }

  function redactRelativeText(message: string) {
    for (const [url, redacted] of replacements) message = replaceRegisteredUrl(message, url, redacted);
    return message;
  }

  function text(message: string) {
    for (const url of absoluteUrls) {
      if (!/^https?:\/\//i.test(url) && hasCoreAssetUrlSecrets(url)) {
        message = replaceRegisteredUrl(message, url, redactCoreAssetUrl(url));
      }
    }
    let offset = 0;
    let redacted = '';
    for (const match of message.matchAll(HTTP_URL_START)) {
      if (match.index < offset) continue;
      const known = knownHttpUrlAt(message, match.index);
      const end = known?.end ?? absoluteUrlEnd(message, match.index + match[0].length);
      redacted += redactRelativeText(message.slice(offset, match.index));
      redacted += known?.value ?? redactAbsoluteUrlToken(message.slice(match.index, end));
      offset = end;
    }
    return redacted + redactRelativeText(message.slice(offset));
  }

  function error(failure: unknown, depth = 0, seen = new WeakSet<object>()): unknown {
    if (typeof failure === 'string') return text(failure);
    if (!failure || typeof failure !== 'object') return failure;
    // Returning the original tail would expose unsanitized errors past the bound.
    if (depth > 5 || seen.has(failure)) return undefined;
    const {
      message: originalMessage, stack: originalStack, cause: originalCause, errors: originalErrors, name,
    } = failure as Partial<Error> & { errors?: unknown };
    if (typeof originalMessage !== 'string') return failure;
    seen.add(failure);
    const message = text(originalMessage);
    const stack = typeof originalStack === 'string' ? text(originalStack) : originalStack;
    const cause = error(originalCause, depth + 1, seen);
    const aggregate = failure instanceof AggregateError || name === 'AggregateError';
    const errors = aggregate && Array.isArray(originalErrors)
      ? originalErrors.map((child) => error(child, depth + 1, seen)) : null;
    seen.delete(failure);
    if (message === originalMessage && stack === originalStack && cause === originalCause
      && (!errors || errors.every((child, index) => child === (originalErrors as unknown[])[index]))) return failure;

    const sanitized = aggregate
      ? new AggregateError(errors ?? [], message)
      : typeof DOMException === 'function' && Object.prototype.toString.call(failure) === '[object DOMException]'
        ? new DOMException(message, name)
        : new Error(message);
    Object.setPrototypeOf(sanitized, Object.getPrototypeOf(failure));
    const descriptors = Object.getOwnPropertyDescriptors(failure);
    if (cause !== originalCause) {
      descriptors.cause = {
        value: cause, writable: true, configurable: true, enumerable: descriptors.cause?.enumerable ?? false,
      };
    }
    if (errors) {
      descriptors.errors = {
        value: errors, writable: true, configurable: true, enumerable: descriptors.errors?.enumerable ?? false,
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
  rememberRedirect?: (url: string) => void,
): Promise<CoreAssetError | null> {
  if (!/^https?:\/\//i.test(url)) return null;

  const abort = new AbortController();
  const timer = globalThis.setTimeout(() => abort.abort(), PREFLIGHT_DIAGNOSTIC_TIMEOUT_MS);
  let httpStatus: number | null = null;
  try {
    const response = await fetch(url, { method: 'GET', credentials: 'same-origin', signal: abort.signal });
    if (response.url) rememberRedirect?.(response.url);
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
