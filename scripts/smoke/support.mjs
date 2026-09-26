// Shared helpers for the Playwright browser smokes (scripts/smoke/*.mjs).
//
// Ported from the helpers in scripts/state_persistence_browser_smoke.py and
// kept behaviour-compatible with them: the same HTTP semantics and headers,
// model cache names, checksum and redaction rules, artifacts, and the same
// stdout bytes. The smokes print their result with pyJson, which writes what
// Python's json.dumps writes for the value Playwright's Python client returns,
// so NaN, undefined properties and float formatting do not change the output.

import { createHash } from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';

export const REQUIRED_ARTIFACTS = Object.freeze([
  'llama_webgpu_bridge.js',
  'llama_webgpu_bridge_worker.js',
  'llama_webgpu_core.js',
  'llama_webgpu_core.wasm',
]);
export const MEMORY64_ARTIFACTS = Object.freeze([
  'llama_webgpu_core_mem64.js',
  'llama_webgpu_core_mem64.wasm',
]);
export const DEFAULT_MODEL_CACHE = '~/.cache/llama-web-bridge/state-smoke-models';

// The download read timeout, per socket read like urllib's timeout=120.
const DOWNLOAD_IDLE_TIMEOUT_MS = 120_000;

export function ensure(condition, message) {
  if (!condition) throw new Error(message);
}

// --- Paths -----------------------------------------------------------------

// os.path.expanduser for the current user: `~` and `~/...` only.
export function expandHome(value) {
  if (value === '~' || value.startsWith('~/')) {
    const home = process.env.HOME || os.homedir();
    return value === '~' ? home : path.join(home, value.slice(2));
  }
  return value;
}

// pathlib.Path.resolve(strict=False), i.e. os.path.realpath: components are
// resolved left to right, so `link/..` is the parent of the link's target,
// and components that do not exist are kept as they are.
export function resolvePath(value) {
  const absolute = path.isAbsolute(value) ? value : `${process.cwd()}/${value}`;
  let resolved = '/';
  for (const component of absolute.split('/')) {
    if (component === '' || component === '.') continue;
    if (component === '..') {
      resolved = path.dirname(resolved);
      continue;
    }
    const candidate = path.join(resolved, component);
    try {
      resolved = fs.realpathSync(candidate);
    } catch {
      resolved = candidate;
    }
  }
  return resolved;
}

export function isFile(value) {
  try {
    return fs.statSync(value).isFile();
  } catch {
    return false;
  }
}

export function isDirectory(value) {
  try {
    return fs.statSync(value).isDirectory();
  } catch {
    return false;
  }
}

// --- Checksums, redaction, downloads ----------------------------------------

export async function sha256File(file) {
  const digest = createHash('sha256');
  for await (const chunk of fs.createReadStream(file, { highWaterMark: 1024 * 1024 })) {
    digest.update(chunk);
  }
  return digest.digest('hex');
}

// An empty expected value skips the check; the comparison ignores case.
export async function validateHash(file, expectedSha256) {
  if (!expectedSha256) return;
  const actual = await sha256File(file);
  ensure(
    actual.toLowerCase() === expectedSha256.toLowerCase(),
    `model checksum mismatch for ${file}: expected ${expectedSha256}, got ${actual}`,
  );
}

// urllib.parse.uses_params: the schemes whose last path segment loses `;...`.
const USES_PARAMS = new Set([
  '', 'ftp', 'hdl', 'prospero', 'http', 'imap', 'https', 'shttp', 'rtsp', 'rtsps', 'rtspu',
  'sip', 'sips', 'mms', 'sftp', 'tel',
]);

// urllib.parse.urlparse, reduced to the fields the smokes read.
export function pyUrlParse(url) {
  // urlsplit strips leading C0 controls and spaces and removes tab and newline.
  let rest = url.replace(/^[\x00-\x20]+/, '').replace(/[\t\r\n]/g, '');
  let scheme = '';
  const colon = rest.indexOf(':');
  if (colon > 0 && /^[A-Za-z][A-Za-z0-9+.-]*$/.test(rest.slice(0, colon))) {
    scheme = rest.slice(0, colon).toLowerCase();
    rest = rest.slice(colon + 1);
  }
  let netloc = '';
  if (rest.startsWith('//')) {
    const end = rest.slice(2).search(/[/?#]/);
    netloc = end < 0 ? rest.slice(2) : rest.slice(2, end + 2);
    rest = end < 0 ? '' : rest.slice(end + 2);
    if ((netloc.includes('[') && !netloc.includes(']')) || (netloc.includes(']') && !netloc.includes('['))) {
      throw new Error('Invalid IPv6 URL');
    }
  }
  const fragmentAt = rest.indexOf('#');
  if (fragmentAt >= 0) rest = rest.slice(0, fragmentAt);
  const queryAt = rest.indexOf('?');
  if (queryAt >= 0) rest = rest.slice(0, queryAt);
  let urlPath = rest;
  if (USES_PARAMS.has(scheme) && urlPath.includes(';')) {
    const at = urlPath.includes('/') ? urlPath.indexOf(';', urlPath.lastIndexOf('/')) : urlPath.indexOf(';');
    if (at >= 0) urlPath = urlPath.slice(0, at);
  }
  // ParseResult._hostinfo.
  const hostinfo = netloc.slice(netloc.lastIndexOf('@') + 1);
  let hostname;
  let port;
  const open = hostinfo.indexOf('[');
  if (open >= 0) {
    const bracketed = hostinfo.slice(open + 1);
    const close = bracketed.indexOf(']');
    hostname = close < 0 ? bracketed : bracketed.slice(0, close);
    const after = close < 0 ? '' : bracketed.slice(close + 1);
    port = after.includes(':') ? after.slice(after.indexOf(':') + 1) : '';
  } else {
    const colon = hostinfo.indexOf(':');
    hostname = colon < 0 ? hostinfo : hostinfo.slice(0, colon);
    port = colon < 0 ? '' : hostinfo.slice(colon + 1);
  }
  // ParseResult.hostname lowercases the part before a `%` zone ID.
  if (hostname) {
    const zone = hostname.indexOf('%');
    hostname = zone < 0 ? hostname.toLowerCase() : hostname.slice(0, zone).toLowerCase() + hostname.slice(zone);
  }
  return {
    scheme,
    netloc,
    path: urlPath,
    hostname: hostname || null,
    // ParseResult.port, which raises on a malformed port.
    get port() {
      if (!port) return null;
      if (!/^[0-9]+$/.test(port)) throw new Error(`Port could not be cast to integer value as '${port}'`);
      const value = Number(port);
      if (value > 65535) throw new Error('Port out of range 0-65535');
      return value;
    },
  };
}

// `scheme://host[:port]/path`: userinfo, params, query and fragment dropped.
export function redactLocation(location) {
  const parsed = pyUrlParse(location);
  if (!parsed.scheme || !parsed.netloc) return '[invalid-url]';
  let netloc = parsed.hostname || parsed.netloc.slice(parsed.netloc.lastIndexOf('@') + 1);
  if (parsed.port) netloc = `${netloc}:${parsed.port}`;
  return `${parsed.scheme}://${netloc}${parsed.path}`;
}

// Path(urlparse(url).path).name.
function urlPathName(url) {
  const parts = pyUrlParse(url).path.split('/').filter((part) => part !== '' && part !== '.');
  return parts.length ? parts[parts.length - 1] : '';
}

export function cachedModelName(url, defaultName = 'state-smoke-model.gguf') {
  const digest = createHash('sha256').update(url, 'utf8').digest('hex').slice(0, 12);
  return `${digest}-${urlPathName(url) || defaultName}`;
}

// The body of an HTTP download, failing on a non-2xx status like urlopen and
// on 120 s without data like its socket timeout.
async function fetchToFile(url, target) {
  const controller = new AbortController();
  let timer;
  const arm = () => {
    clearTimeout(timer);
    timer = setTimeout(() => controller.abort(new Error('timed out')), DOWNLOAD_IDLE_TIMEOUT_MS);
  };
  arm();
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      await response.body?.cancel().catch(() => {});
      throw new Error(`HTTP Error ${response.status}: ${response.statusText}`);
    }
    const output = await fsp.open(target, 'w');
    try {
      for await (const chunk of response.body) {
        arm();
        await output.write(chunk);
      }
    } finally {
      await output.close();
    }
  } finally {
    clearTimeout(timer);
  }
}

// A download error's text with every URL in it redacted: fetch names the URL
// when it is malformed or holds credentials.
function downloadErrorText(error) {
  const cause = error?.cause;
  const text = cause instanceof Error ? cause.message : error instanceof Error ? error.message : String(error);
  return text.replace(/[A-Za-z][A-Za-z0-9+.-]*:\/\/\S*/g, (url) => redactLocation(url));
}

// Download to `<cache>/<sha256(url)[:12]>-<name>`. A cached file is checked and
// never downloaded again. A download lands in `<target>.tmp` and is renamed
// after its checksum matches.
export async function downloadToCache(url, cacheDir, expectedSha256) {
  await fsp.mkdir(cacheDir, { recursive: true });
  const target = path.join(cacheDir, cachedModelName(url));
  if (isFile(target)) {
    await validateHash(target, expectedSha256);
    return target;
  }
  const temp = `${target}.tmp`;
  try {
    await fetchToFile(url, temp);
  } catch (error) {
    await fsp.unlink(temp).catch(() => {});
    throw new Error(`failed to download smoke model from ${redactLocation(url)}: ${downloadErrorText(error)}`);
  }
  // Like the Python helper, a mismatch leaves the .tmp file for the next run
  // to overwrite and reports the checksum error as is.
  await validateHash(temp, expectedSha256);
  await fsp.rename(temp, target);
  return target;
}

// The model of a smoke that requires a checksum: `--model-path` (with `~`
// expanded) wins over `--model-url`, which downloads through the cache.
export async function resolvePinnedModel({ modelPath, modelUrl, modelSha256, modelCacheDir }) {
  ensure(Boolean(modelSha256), 'model SHA-256 is required');
  if (modelPath !== null) {
    const resolved = resolvePath(expandHome(modelPath));
    ensure(isFile(resolved), `model path does not exist: ${resolved}`);
    await validateHash(resolved, modelSha256);
    return resolved;
  }
  ensure(Boolean(modelUrl), '--model-url or --model-path is required');
  return downloadToCache(modelUrl, resolvePath(expandHome(modelCacheDir)), modelSha256);
}

// resolve_file from the multimodal smoke: the same rules as resolvePinnedModel
// for any pinned input, with `label` naming it in the failures.
export async function resolvePinnedFile({ filePath, url, expectedSha256, cacheDir, label }) {
  ensure(Boolean(expectedSha256), `${label} SHA-256 is required`);
  if (filePath !== null) {
    const resolved = resolvePath(expandHome(filePath));
    ensure(isFile(resolved), `${label} path does not exist: ${resolved}`);
    await validateHash(resolved, expectedSha256);
    return resolved;
  }
  ensure(Boolean(url), `${label} URL or local path is required`);
  return downloadToCache(url, resolvePath(expandHome(cacheDir)), expectedSha256);
}

// --- Web root ----------------------------------------------------------------

export async function copyBridgeArtifacts(distDir, webRoot) {
  for (const artifact of REQUIRED_ARTIFACTS) {
    const source = path.join(distDir, artifact);
    ensure(isFile(source), `missing bridge artifact: ${source}`);
    await fsp.copyFile(source, path.join(webRoot, artifact));
  }
}

export async function copyMemory64Artifacts(distDir, webRoot) {
  for (const artifact of MEMORY64_ARTIFACTS) {
    const source = path.join(distDir, artifact);
    ensure(isFile(source), `missing wasm64 bridge artifact: ${source}`);
    await fsp.copyFile(source, path.join(webRoot, artifact));
  }
}

// Stage a large input without copying when the filesystem allows a hard link.
export async function stageFile(source, target) {
  try {
    await fsp.link(source, target);
  } catch {
    await fsp.copyFile(source, target);
  }
}

// tempfile.TemporaryDirectory: removed afterwards, even on failure.
export async function withTempDir(prefix, fn) {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), prefix));
  try {
    return await fn(dir);
  } finally {
    await fsp.rm(dir, { recursive: true, force: true });
  }
}

// --- Cross-origin-isolated static server --------------------------------------

// The headers every response carries, after the per-response ones.
export const ISOLATION_HEADERS = Object.freeze([
  ['Cross-Origin-Opener-Policy', 'same-origin'],
  ['Cross-Origin-Embedder-Policy', 'require-corp'],
  ['Cross-Origin-Resource-Policy', 'same-origin'],
  ['Access-Control-Allow-Origin', '*'],
  ['Cache-Control', 'no-store'],
]);

// Python's mimetypes for the files the smokes serve; anything else is
// application/octet-stream, as for .gguf.
const MIME_TYPES = new Map([
  ['.html', 'text/html'],
  ['.htm', 'text/html'],
  ['.js', 'text/javascript'],
  ['.mjs', 'text/javascript'],
  ['.json', 'application/json'],
  ['.wasm', 'application/wasm'],
  ['.png', 'image/png'],
  ['.txt', 'text/plain'],
  ['.wav', 'audio/x-wav'],
  ['.gz', 'application/gzip'],
]);

export function contentType(file) {
  const ext = path.extname(file);
  return MIME_TYPES.get(ext) ?? MIME_TYPES.get(ext.toLowerCase()) ?? 'application/octet-stream';
}

const ERROR_EXPLANATIONS = new Map([
  [404, 'Nothing matches the given URI'],
  [501, 'Server does not support this operation'],
]);

function escapeHtml(text) {
  return text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

// BaseHTTPRequestHandler.send_error.
function sendError(req, res, code, message) {
  const body = Buffer.from(
    '<!DOCTYPE HTML>\n<html lang="en">\n    <head>\n        <meta charset="utf-8">\n'
      + '        <style type="text/css">\n            :root {\n                color-scheme: light dark;\n'
      + '            }\n        </style>\n        <title>Error response</title>\n    </head>\n    <body>\n'
      + '        <h1>Error response</h1>\n'
      + `        <p>Error code: ${code}</p>\n`
      + `        <p>Message: ${escapeHtml(message)}.</p>\n`
      + `        <p>Error code explanation: ${code} - ${escapeHtml(ERROR_EXPLANATIONS.get(code))}.</p>\n`
      + '    </body>\n</html>\n',
    'utf8',
  );
  res.writeHead(code, message, [
    ['Connection', 'close'],
    ['Content-Type', 'text/html;charset=utf-8'],
    ['Content-Length', String(body.length)],
    ...ISOLATION_HEADERS,
  ]);
  res.end(req.method === 'HEAD' ? undefined : body);
}

// urllib.parse.unquote: invalid escapes stay, invalid UTF-8 becomes U+FFFD.
function unquote(text) {
  return text.replace(/(?:%[0-9A-Fa-f]{2})+/g, (run) => Buffer.from(run.replaceAll('%', ''), 'hex').toString('utf8'));
}

// SimpleHTTPRequestHandler.translate_path: no component can leave the root.
export function translatePath(root, requestPath) {
  const decoded = unquote(requestPath.split('#', 1)[0].split('?', 1)[0]);
  const trailingSlash = decoded.endsWith('/');
  const words = path.posix.normalize(decoded).split('/').filter((word) => word && word !== '.' && word !== '..');
  const translated = path.join(root, ...words);
  return trailingSlash ? `${translated}/` : translated;
}

// SimpleHTTPRequestHandler.send_head with the semantics of Python's HTTP/1.0
// server: GET and HEAD only, Range ignored (every file is a full 200, as the
// bridge's `bytes=0-0` probe expects from this server), and one response per
// connection (`Connection: close` on an HTTP/1.1 status line). A directory
// without index.html is a 404 rather than a listing; smoke web roots are flat.
async function handleRequest(root, req, res) {
  res.shouldKeepAlive = false;
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    sendError(req, res, 501, `Unsupported method ('${req.method}')`);
    return;
  }
  let file = translatePath(root, req.url);
  if (isDirectory(file)) {
    const cut = req.url.search(/[?#]/);
    const urlPath = cut < 0 ? req.url : req.url.slice(0, cut);
    if (!/(\/|%2f|%2F)$/.test(urlPath)) {
      const location = cut < 0 ? `${req.url}/` : `${urlPath}/${req.url.slice(cut)}`;
      res.writeHead(301, 'Moved Permanently', [['Location', location], ['Content-Length', '0'], ...ISOLATION_HEADERS]);
      res.end();
      return;
    }
    const index = ['index.html', 'index.htm'].map((name) => path.join(file, name)).find(isFile);
    if (!index) {
      sendError(req, res, 404, 'File not found');
      return;
    }
    file = index;
  }
  if (file.endsWith('/')) {
    sendError(req, res, 404, 'File not found');
    return;
  }
  let handle;
  try {
    handle = await fsp.open(file, 'r');
  } catch {
    sendError(req, res, 404, 'File not found');
    return;
  }
  try {
    const stat = await handle.stat();
    const modified = new Date(Math.floor(stat.mtimeMs / 1000) * 1000);
    const since = req.headers['if-modified-since'];
    if (since !== undefined && req.headers['if-none-match'] === undefined
      && /(?:GMT|UTC|[+-]0000)\s*$/.test(since) && modified.getTime() <= Date.parse(since)) {
      res.writeHead(304, 'Not Modified', [...ISOLATION_HEADERS]);
      res.end();
      await handle.close();
      return;
    }
    res.writeHead(200, 'OK', [
      ['Content-type', contentType(file)],
      ['Content-Length', String(stat.size)],
      ['Last-Modified', modified.toUTCString()],
      ...ISOLATION_HEADERS,
    ]);
  } catch (error) {
    await handle.close();
    throw error;
  }
  if (req.method === 'HEAD') {
    res.end();
    await handle.close();
    return;
  }
  // 256 KiB reads, like shutil.copyfileobj in Python 3.14. With Node's 64 KiB
  // default the page saw more body chunks than with the Python server, which
  // changed the state smoke's progressEvents count.
  const stream = handle.createReadStream({ autoClose: true, highWaterMark: 256 * 1024 });
  // A `Range: bytes=0-0` probe gets the whole file and aborts; stop reading.
  res.once('close', () => stream.destroy());
  stream.once('error', () => res.destroy());
  stream.pipe(res);
}

// Serve `webRoot` on 127.0.0.1 at a free port until `close()`.
export async function serveIsolated(webRoot) {
  const root = path.resolve(webRoot);
  const server = http.createServer((req, res) => {
    handleRequest(root, req, res).catch(() => res.destroy());
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const { port } = server.address();
  return {
    url: `http://127.0.0.1:${port}/index.html`,
    close: () => new Promise((resolve) => {
      server.close(() => resolve());
      server.closeAllConnections();
    }),
  };
}

export async function withServer(webRoot, fn) {
  const server = await serveIsolated(webRoot);
  try {
    return await fn(server.url);
  } finally {
    await server.close();
  }
}

// --- Python-compatible JSON and repr ------------------------------------------

// A float that json.loads read from a literal with a fraction or exponent, such
// as `1.0`, which a JS number cannot tell apart from the int 1. pyJsonLoads
// returns these so pyJson writes such a value back as Python did.
export class PyFloat {
  constructor(value) {
    this.value = value;
    Object.freeze(this);
  }
}

// Playwright's Python client turns a JS number into an int when its JSON text
// has no fraction or exponent, and into a float otherwise; -0 is a float. A
// bigint (an int json.loads read past 2**53) is an int.
function isPyInt(value) {
  if (typeof value === 'bigint') return true;
  return typeof value === 'number' && Number.isFinite(value) && !Object.is(value, -0) && !/[.e]/.test(String(value));
}

// float.__repr__: the shortest round-trip digits, with an exponent when the
// decimal point falls outside -4 < decpt <= 16.
export function pyFloatRepr(value) {
  if (Number.isNaN(value)) return 'nan';
  if (value === Infinity) return 'inf';
  if (value === -Infinity) return '-inf';
  if (value === 0) return Object.is(value, -0) ? '-0.0' : '0.0';
  const sign = value < 0 ? '-' : '';
  const [mantissa, exponentText] = Math.abs(value).toExponential().split('e');
  const digits = mantissa.replace('.', '');
  const exponent = Number(exponentText);
  const decpt = exponent + 1;
  if (decpt <= -4 || decpt > 16) {
    const lead = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    return `${sign}${lead}e${exponent < 0 ? '-' : '+'}${String(Math.abs(exponent)).padStart(2, '0')}`;
  }
  if (decpt <= 0) return `${sign}0.${'0'.repeat(-decpt)}${digits}`;
  if (decpt >= digits.length) return `${sign}${digits}${'0'.repeat(decpt - digits.length)}.0`;
  return `${sign}${digits.slice(0, decpt)}.${digits.slice(decpt)}`;
}

function pyNumberJson(value) {
  if (value instanceof PyFloat) value = value.value;
  else if (isPyInt(value)) return String(value);
  if (Number.isNaN(value)) return 'NaN';
  if (value === Infinity) return 'Infinity';
  if (value === -Infinity) return '-Infinity';
  return pyFloatRepr(value);
}

const JSON_ESCAPES = new Map([
  ['"', '\\"'], ['\\', '\\\\'], ['\n', '\\n'], ['\r', '\\r'], ['\t', '\\t'], ['\b', '\\b'], ['\f', '\\f'],
]);

// json.dumps(str) with ensure_ascii: UTF-16 code units outside ' '..'~' as \uXXXX.
function pyJsonString(text) {
  return `"${text.replace(/[\\"]|[^\x20-\x7e]/g, (char) => JSON_ESCAPES.get(char)
    ?? `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`)}"`;
}

// Python sorts str keys by code point, not by UTF-16 code unit.
function compareCodePoints(left, right) {
  const a = Array.from(left, (char) => char.codePointAt(0));
  const b = Array.from(right, (char) => char.codePointAt(0));
  for (let i = 0; i < Math.min(a.length, b.length); i += 1) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return a.length - b.length;
}

function isPlainObject(value) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function pyTypeName(value) {
  if (value === null || value === undefined) return 'NoneType';
  if (typeof value === 'boolean') return 'bool';
  if (typeof value === 'number' || typeof value === 'bigint') return isPyInt(value) ? 'int' : 'float';
  if (value instanceof PyFloat) return 'float';
  if (typeof value === 'string') return 'str';
  if (Array.isArray(value)) return 'list';
  if (value instanceof Date) return 'datetime';
  if (isPlainObject(value)) return 'dict';
  return value?.constructor?.name ?? typeof value;
}

// json.dumps(value, indent=indent, sort_keys=sortKeys) over the value Python
// would hold: undefined and null are None, NaN and the infinities are bare
// literals, and -0 is -0.0.
export function pyJson(value, { indent = null, sortKeys = false } = {}) {
  const itemSeparator = indent === null ? ', ' : ',';
  const encode = (item, level) => {
    if (item === null || item === undefined) return 'null';
    if (item === true) return 'true';
    if (item === false) return 'false';
    if (typeof item === 'number' || typeof item === 'bigint' || item instanceof PyFloat) return pyNumberJson(item);
    if (typeof item === 'string') return pyJsonString(item);
    let entries;
    let open;
    let close;
    if (Array.isArray(item)) {
      entries = Array.from(item, (element) => encode(element, level + 1));
      [open, close] = ['[', ']'];
    } else if (isPlainObject(item)) {
      const keys = Object.keys(item);
      if (sortKeys) keys.sort(compareCodePoints);
      entries = keys.map((key) => `${pyJsonString(key)}: ${encode(item[key], level + 1)}`);
      [open, close] = ['{', '}'];
    } else {
      throw new TypeError(`Object of type ${pyTypeName(item)} is not JSON serializable`);
    }
    if (entries.length === 0) return `${open}${close}`;
    if (indent === null) return `${open}${entries.join(itemSeparator)}${close}`;
    const inner = `\n${' '.repeat(indent * (level + 1))}`;
    return `${open}${inner}${entries.join(`${itemSeparator}${inner}`)}\n${' '.repeat(indent * level)}${close}`;
  };
  return encode(value, 0);
}

// str.isprintable is false for these categories, except for the space itself.
const NOT_PRINTABLE = /[\p{Cc}\p{Cf}\p{Cs}\p{Co}\p{Cn}\p{Zl}\p{Zp}\p{Zs}]/u;

function pyStrRepr(text) {
  const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
  let out = quote;
  for (const char of text) {
    const code = char.codePointAt(0);
    if (char === quote || char === '\\') out += `\\${char}`;
    else if (char === '\t') out += '\\t';
    else if (char === '\n') out += '\\n';
    else if (char === '\r') out += '\\r';
    else if (char !== ' ' && (code < 0x20 || code === 0x7f || (code > 0x7f && NOT_PRINTABLE.test(char)))) {
      if (code < 0x100) out += `\\x${code.toString(16).padStart(2, '0')}`;
      else if (code < 0x10000) out += `\\u${code.toString(16).padStart(4, '0')}`;
      else out += `\\U${code.toString(16).padStart(8, '0')}`;
    } else out += char;
  }
  return out + quote;
}

// repr() of the value Python would hold, for failure messages.
export function pyRepr(value) {
  if (value === null || value === undefined) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  if (typeof value === 'number' || typeof value === 'bigint') return isPyInt(value) ? String(value) : pyFloatRepr(value);
  if (value instanceof PyFloat) return pyFloatRepr(value.value);
  if (typeof value === 'string') return pyStrRepr(value);
  if (Array.isArray(value)) return `[${Array.from(value, pyRepr).join(', ')}]`;
  if (isPlainObject(value)) {
    return `{${Object.keys(value).map((key) => `${pyStrRepr(key)}: ${pyRepr(value[key])}`).join(', ')}}`;
  }
  return String(value);
}

export { isPlainObject as isDict, isPyInt, pyTypeName };

// str(): a str is itself, anything else its repr.
export function pyStr(value) {
  return typeof value === 'string' ? value : pyRepr(value);
}

// dict.get(key) on a value that must be a dict, as Python fails otherwise.
export function pyGet(value, key) {
  if (!isPlainObject(value)) throw new Error(`'${pyTypeName(value)}' object has no attribute 'get'`);
  return Object.hasOwn(value, key) && value[key] !== undefined ? value[key] : null;
}

// Python equality for JSON-shaped values (1 == 1.0, True == 1, lists by item).
export function pyEquals(left, right) {
  const scalar = (value) => {
    if (typeof value === 'boolean') return Number(value);
    if (value instanceof PyFloat) return value.value;
    if (typeof value === 'bigint' && Number.isSafeInteger(Number(value))) return Number(value);
    return value;
  };
  if (Array.isArray(left) || Array.isArray(right)) {
    return Array.isArray(left) && Array.isArray(right) && left.length === right.length
      && left.every((item, index) => pyEquals(item, right[index]));
  }
  if (isPlainObject(left) || isPlainObject(right)) {
    if (!isPlainObject(left) || !isPlainObject(right)) return false;
    const keys = Object.keys(left);
    return keys.length === Object.keys(right).length
      && keys.every((key) => Object.hasOwn(right, key) && pyEquals(left[key], right[key]));
  }
  const a = left === undefined ? null : scalar(left);
  const b = right === undefined ? null : scalar(right);
  return a === b;
}

// json.loads(text) with the int/float distinction kept: a number literal with
// a fraction or exponent is a PyFloat, an int past 2**53 is a bigint, and -0 is
// the int 0. Duplicate keys keep the last value, as in Python. Python's
// NaN/Infinity literals are not JSON and are rejected here, and a JS object
// lists integer-like keys ("1") first where a dict keeps insertion order.
export function pyJsonLoads(text) {
  return JSON.parse(text, (key, value, context) => {
    if (typeof value !== 'number') return value;
    const source = context?.source;
    if (typeof source !== 'string') throw new Error('JSON.parse source text access is required (Node.js 22.18 or newer)');
    if (/[.eE]/.test(source)) return new PyFloat(value);
    if (Number.isSafeInteger(value)) return value === 0 ? 0 : value;
    return BigInt(source);
  });
}

// Path.read_text(encoding="utf-8"): strict UTF-8 that keeps a BOM, with
// universal newlines (\r\n and \r read as \n).
export async function readPyText(file) {
  const bytes = await fsp.readFile(file);
  let text;
  try {
    text = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes);
  } catch {
    throw new Error(`'utf-8' codec can't decode the file ${file}`);
  }
  return text.replace(/\r\n?/g, '\n');
}

// The characters str.split() and str.strip() treat as whitespace.
const PY_WHITESPACE = '\t\n\v\f\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000';

// str.strip().
export function pyStrip(text) {
  return text.replace(new RegExp(`^[${PY_WHITESPACE}]+|[${PY_WHITESPACE}]+$`, 'g'), '');
}

// bool(value).
export function pyTruthy(value) {
  if (value === null || value === undefined || value === false) return false;
  if (typeof value === 'number') return value !== 0;
  if (typeof value === 'bigint') return value !== 0n;
  if (value instanceof PyFloat) return value.value !== 0;
  if (typeof value === 'string' || Array.isArray(value)) return value.length > 0;
  if (isPlainObject(value)) return Object.keys(value).length > 0;
  return true;
}

// dict.get(key, default): the default only when the key is absent. A key
// holding undefined holds None, as the Python client read it.
export function pyGetDefault(value, key, fallback) {
  if (!isPlainObject(value)) throw new Error(`'${pyTypeName(value)}' object has no attribute 'get'`);
  if (!Object.hasOwn(value, key)) return fallback;
  return value[key] === undefined ? null : value[key];
}

// dict[key], which raises KeyError (str() of it is the key's repr).
export function pyIndex(value, key) {
  if (!isPlainObject(value)) throw new Error(`'${pyTypeName(value)}' object is not subscriptable`);
  if (!Object.hasOwn(value, key)) throw new Error(pyRepr(key));
  return value[key] === undefined ? null : value[key];
}

// dict.pop(key, default).
export function pyPop(value, key, fallback) {
  const popped = pyGetDefault(value, key, fallback);
  delete value[key];
  return popped;
}

// iter(value) over a JSON-shaped value: a dict yields its keys and a str its
// characters; anything else that is not a list is not iterable.
export function pyIter(value) {
  if (Array.isArray(value)) return value;
  if (isPlainObject(value)) return Object.keys(value);
  if (typeof value === 'string') return Array.from(value);
  throw new Error(`'${pyTypeName(value)}' object is not iterable`);
}

// len(value) over a JSON-shaped value; a str counts code points.
export function pyLen(value) {
  if (Array.isArray(value)) return value.length;
  if (isPlainObject(value)) return Object.keys(value).length;
  if (typeof value === 'string') return Array.from(value).length;
  throw new Error(`object of type '${pyTypeName(value)}' has no len()`);
}

// `value <= limit` for a float limit, which raises for a value that is not a
// number (bool counts as an int).
export function pyLessEqual(value, limit) {
  if (typeof value === 'boolean') return Number(value) <= limit;
  if (typeof value === 'number') return value <= limit;
  if (typeof value === 'bigint') return Number(value) <= limit;
  if (value instanceof PyFloat) return value.value <= limit;
  throw new Error(`'<=' not supported between instances of '${pyTypeName(value)}' and 'float'`);
}

// base64.b64decode(str) for the canonical base64 a page's btoa() writes. A
// value that is not a str, or not canonical base64, fails; Python would
// decode some non-canonical forms, which no harness produces.
export function pyB64Decode(value) {
  if (typeof value !== 'string') {
    throw new Error(`argument should be a bytes-like object or ASCII string, not '${pyTypeName(value)}'`);
  }
  if (value.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(value)) {
    throw new Error('base64 data is not canonical');
  }
  return Buffer.from(value, 'base64');
}

// --- Mode results ----------------------------------------------------------------

// `<memory> <runtime>` for every memory mode, then every runtime mode.
export function expectedModes(memoryModes, runtimeModes) {
  return memoryModes.flatMap((memoryMode) => runtimeModes.map((runtimeMode) => `${memoryMode} ${runtimeMode}`));
}

// The failures of a `<memory> <runtime>` mode result that ran on another core
// variant or thread than its mode names.
export function modeRuntimeFailures(mode, entry) {
  const failures = [];
  const expectedVariant = mode.split(' ', 1)[0];
  if (!pyEquals(pyGet(entry, 'coreVariant'), expectedVariant)) {
    // The bridge falls back to wasm32 when the mem64 core fails to start,
    // which would otherwise pass as wasm64 coverage.
    failures.push(`${mode}: expected the ${expectedVariant} core, got ${pyRepr(pyGet(entry, 'coreVariant'))}`);
  }
  const expectedExecution = mode.endsWith(' worker') ? 'worker' : 'main-thread';
  if (!pyEquals(pyGet(entry, 'execution'), expectedExecution)) {
    failures.push(
      `${mode}: expected ${expectedExecution} execution, got ${pyRepr(pyGet(entry, 'execution'))} `
        + `(worker fallback reason: ${pyRepr(pyGet(entry, 'workerFallbackReason'))})`,
    );
  }
  return failures;
}

// entry["mode"] for a result whose mode list already matched.
export function modeOf(entry) {
  if (!isPlainObject(entry)) throw new Error(`'${pyTypeName(entry)}' object is not subscriptable`);
  return entry.mode;
}

// --- Artifacts -----------------------------------------------------------------

export async function writeTextArtifact(artifactsDir, name, content) {
  if (artifactsDir === null) return;
  await fsp.mkdir(artifactsDir, { recursive: true });
  await fsp.writeFile(path.join(artifactsDir, name), content, 'utf8');
}

// json.dumps(payload, indent=2, sort_keys=True), with no trailing newline.
export async function writeJsonArtifact(artifactsDir, name, payload) {
  await writeTextArtifact(artifactsDir, name, pyJson(payload, { indent: 2, sortKeys: true }));
}

// --- Playwright ----------------------------------------------------------------

async function loadChromium() {
  try {
    return (await import('playwright')).chromium;
  } catch (error) {
    if (error?.code !== 'ERR_MODULE_NOT_FOUND') throw error;
    throw new Error(
      'playwright is required; install with `npm ci --ignore-scripts` '
        + 'and `npx --no-install playwright install --only-shell chromium`',
    );
  }
}

// Load the harness in headless Chromium, wait for window.__smokeResult.ok, and
// return the result with the last 200 console lines. The console log and the
// result are written to `artifactsDir`, and a full-page screenshot when the
// page fails or reports ok !== true.
export async function runPlaywright(url, timeoutMs, artifactsDir, artifactPrefix = 'state-smoke') {
  const chromium = await loadChromium();
  const consoleLines = [];
  let payload = null;
  const browser = await chromium.launch({ args: ['--no-sandbox'] });
  try {
    const page = await browser.newPage();
    page.on('console', (message) => consoleLines.push(`${message.type()}: ${message.text()}`));
    page.on('pageerror', (error) => consoleLines.push(`pageerror: ${error.message}`));
    const screenshot = () => page.screenshot({
      path: path.join(artifactsDir, `${artifactPrefix}-page.png`),
      fullPage: true,
    });
    try {
      await page.goto(url, { waitUntil: 'load', timeout: timeoutMs });
      // The second argument is the page function's argument, not the options.
      await page.waitForFunction(
        () => window.__smokeResult && window.__smokeResult.ok !== undefined,
        undefined,
        { timeout: timeoutMs },
      );
      payload = await page.evaluate(() => window.__smokeResult);
      if (isPlainObject(payload) && payload.ok !== true && artifactsDir !== null) await screenshot();
    } catch (error) {
      if (artifactsDir !== null) {
        await fsp.mkdir(artifactsDir, { recursive: true });
        await screenshot();
      }
      throw error;
    }
  } finally {
    await browser.close();
  }
  if (!isPlainObject(payload)) throw new Error(`unexpected smoke result payload: ${pyRepr(payload)}`);
  payload.console = consoleLines.slice(-200);
  await writeTextArtifact(artifactsDir, `${artifactPrefix}-console.log`, `${consoleLines.join('\n')}\n`);
  await writeJsonArtifact(artifactsDir, `${artifactPrefix}-result.json`, payload);
  return payload;
}

// The Chromium flags of the text-to-speech and decision smokes, which enable
// WebGPU (Metal through ANGLE on macOS, Vulkan elsewhere).
export function webGpuLaunchArgs(platform = process.platform) {
  const args = ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-webgpu'];
  const features = ['SharedArrayBuffer'];
  if (platform === 'darwin') {
    args.push('--use-angle=metal');
  } else {
    args.push('--disable-vulkan-surface');
    features.push('Vulkan');
  }
  args.push(`--enable-features=${features.join(',')}`);
  return args;
}

// Load the harness in headless Chromium with WebGPU enabled and poll every 2 s
// until window.__smokeResult is an object with an `ok` key, for at most
// `timeoutMs` after the page loads. Console lines containing `stageMarker` are
// echoed to stderr, and with `stagePrefix` every change of window.__smokeStage
// is reported as `<stagePrefix>: <stage>`. A full-page screenshot is written
// only when this fails. `transform(payload)` runs before the console lines are
// added and the console log and result are written to `artifactsDir`.
export async function runPollingPlaywright(url, timeoutMs, artifactsDir, {
  artifactPrefix, stageMarker, stagePrefix = null, transform = async () => {},
}) {
  const chromium = await loadChromium();
  const consoleLines = [];
  let payload = null;
  const browser = await chromium.launch({ args: webGpuLaunchArgs() });
  try {
    const page = await browser.newPage();
    page.on('console', (message) => {
      const line = `${message.type()}: ${message.text()}`;
      consoleLines.push(line);
      if (line.includes(stageMarker)) process.stderr.write(`${line}\n`);
    });
    page.on('pageerror', (error) => consoleLines.push(`pageerror: ${error.message}`));
    try {
      await page.goto(url, { waitUntil: 'load', timeout: timeoutMs });
      const deadline = performance.now() + timeoutMs;
      let previousStage = null;
      let finished = false;
      while (performance.now() < deadline) {
        payload = await page.evaluate(() => window.__smokeResult || null);
        if (isPlainObject(payload) && Object.hasOwn(payload, 'ok')) {
          finished = true;
          break;
        }
        const stage = await page.evaluate(() => window.__smokeStage || 'starting');
        if (!pyEquals(stage, previousStage)) {
          if (stagePrefix !== null) process.stderr.write(`${stagePrefix}: ${pyStr(stage)}\n`);
          previousStage = stage;
        }
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
      if (!finished) throw new Error(`browser smoke timed out at stage: ${pyStr(previousStage)}`);
    } catch (error) {
      if (artifactsDir !== null) {
        await fsp.mkdir(artifactsDir, { recursive: true });
        await page.screenshot({ path: path.join(artifactsDir, `${artifactPrefix}-page.png`), fullPage: true });
      }
      throw error;
    }
  } finally {
    await browser.close();
  }
  if (!isPlainObject(payload)) throw new Error(`unexpected smoke result payload: ${pyRepr(payload)}`);
  await transform(payload);
  payload.console = consoleLines.slice(-200);
  await writeTextArtifact(artifactsDir, `${artifactPrefix}-console.log`, `${consoleLines.join('\n')}\n`);
  await writeJsonArtifact(artifactsDir, `${artifactPrefix}-result.json`, payload);
  return payload;
}

// --- Command line ----------------------------------------------------------------

export class UsageError extends Error {}
class HelpRequested extends Error {}

// int(text): surrounding whitespace, a sign and digit-group underscores.
export function pyInt(text) {
  const trimmed = String(text).trim();
  if (!/^[+-]?[0-9]+(?:_[0-9]+)*$/.test(trimmed)) {
    throw new Error(`invalid literal for int() with base 10: ${pyRepr(String(text))}`);
  }
  return Number(trimmed.replaceAll('_', ''));
}

// float(text): surrounding whitespace, a sign, digit-group underscores, and
// inf, infinity and nan in any case.
export function pyFloat(text) {
  const digits = '[0-9](?:_?[0-9])*';
  const trimmed = pyStrip(String(text));
  const number = new RegExp(`^[+-]?(?:${digits}(?:\\.(?:${digits})?)?|\\.${digits})(?:[eE][+-]?${digits})?$`);
  if (number.test(trimmed)) return Number(trimmed.replaceAll('_', ''));
  const special = /^([+-]?)(inf|infinity|nan)$/i.exec(trimmed);
  if (special) {
    if (special[2].toLowerCase() === 'nan') return NaN;
    return special[1] === '-' ? -Infinity : Infinity;
  }
  throw new Error(`could not convert string to float: ${pyRepr(String(text))}`);
}

// Path(text): an empty path is the current directory.
const pyPath = (text) => (text === '' ? '.' : text);

// Environment defaults, read when the arguments are parsed.
export const env = {
  string: (name, fallback = '') => process.env[name] ?? fallback,
  // Path(os.environ.get(name, fallback))
  path: (name, fallback) => pyPath(process.env[name] ?? fallback),
  // Path(os.environ[name]) if os.environ.get(name) else None
  optionalPath: (name) => (process.env[name] ? process.env[name] : null),
  // int(os.environ.get(name, fallback)); a bad value fails like Python's int().
  int: (name, fallback) => pyInt(process.env[name] ?? fallback),
};

// argparse-compatible parsing of `--flag value` and `--flag=value` options,
// with unique-prefix abbreviations, choices, and exit status 2 on misuse.
// `options` is a list of {flag, type: 'string' | 'path' | 'int' | 'float' |
// 'flag', default: () => value, choices?, required?, help?}; 'flag' is
// action="store_true", and defaults are evaluated in declaration order.
export function parseSmokeArgs(argv, { prog, description, options }) {
  const dest = (flag) => flag.slice(2).replace(/-([a-z0-9])/g, (_, char) => char.toUpperCase());
  const metavar = (option) => option.choices
    ? `{${option.choices.join(',')}}`
    : option.flag.slice(2).toUpperCase().replaceAll('-', '_');
  const invocation = (option) => (option.type === 'flag' ? option.flag : `${option.flag} ${metavar(option)}`);
  const usage = `usage: ${prog} [-h] ${options.map((option) => (option.required
    ? invocation(option)
    : `[${invocation(option)}]`)).join(' ')}`;
  const values = {};
  const seen = new Set();
  for (const option of options) values[dest(option.flag)] = option.default ? option.default() : null;
  const fail = (message) => {
    const error = new UsageError(message);
    error.usage = usage;
    error.prog = prog;
    throw error;
  };
  const isOptionLike = (arg) => arg.startsWith('-') && arg !== '-' && !/^-\d+$|^-\d*\.\d+$/.test(arg) && !arg.includes(' ');
  const unrecognized = [];
  const help = { flag: '--help' };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--') {
      unrecognized.push(...argv.slice(i + 1));
      break;
    }
    if (!isOptionLike(arg)) {
      unrecognized.push(arg);
      continue;
    }
    const equals = arg.indexOf('=');
    const name = equals >= 0 ? arg.slice(0, equals) : arg;
    let option = name === '-h' ? help : [help, ...options].find((candidate) => candidate.flag === name);
    if (!option && name.startsWith('--')) {
      const matches = [help, ...options].filter((candidate) => candidate.flag.startsWith(name));
      if (matches.length > 1) {
        fail(`ambiguous option: ${name} could match ${matches.map((match) => match.flag).join(', ')}`);
      }
      [option] = matches;
    }
    if (option === help) {
      if (equals >= 0) fail(`argument -h/--help: ignored explicit argument ${pyRepr(arg.slice(equals + 1))}`);
      const lines = options.map((option) => `  ${invocation(option)}${option.help ? `\n                        ${option.help}` : ''}`);
      const error = new HelpRequested();
      error.text = `${usage}\n\n${description}\n\noptions:\n  -h, --help            show this help message and exit\n${lines.join('\n')}\n`;
      throw error;
    }
    if (!option) {
      unrecognized.push(arg);
      continue;
    }
    seen.add(option);
    if (option.type === 'flag') {
      if (equals >= 0) fail(`argument ${option.flag}: ignored explicit argument ${pyRepr(arg.slice(equals + 1))}`);
      values[dest(option.flag)] = true;
      continue;
    }
    let raw;
    if (equals >= 0) {
      raw = arg.slice(equals + 1);
    } else if (i + 1 < argv.length && !isOptionLike(argv[i + 1])) {
      i += 1;
      raw = argv[i];
    } else {
      fail(`argument ${option.flag}: expected one argument`);
    }
    let value = raw;
    if (option.type === 'int') {
      try {
        value = pyInt(raw);
      } catch {
        fail(`argument ${option.flag}: invalid int value: ${pyRepr(raw)}`);
      }
    } else if (option.type === 'float') {
      try {
        value = pyFloat(raw);
      } catch {
        fail(`argument ${option.flag}: invalid float value: ${pyRepr(raw)}`);
      }
    } else if (option.type === 'path') {
      value = pyPath(raw);
    }
    if (option.choices && !option.choices.includes(value)) {
      fail(`argument ${option.flag}: invalid choice: ${pyRepr(value)} (choose from ${option.choices.map(pyRepr).join(', ')})`);
    }
    values[dest(option.flag)] = value;
  }
  const missing = options.filter((option) => option.required && !seen.has(option));
  if (missing.length) fail(`the following arguments are required: ${missing.map((option) => option.flag).join(', ')}`);
  if (unrecognized.length) fail(`unrecognized arguments: ${unrecognized.join(' ')}`);
  return values;
}

// Run a smoke's main(): its return value is the exit status. Usage errors exit
// 2; any other error prints `<label> browser smoke failed: <message>` and
// exits 1. process.exitCode, never process.exit(), so piped stdout is flushed.
export async function runMain(label, main) {
  try {
    process.exitCode = await main();
  } catch (error) {
    if (error instanceof HelpRequested) {
      process.stdout.write(error.text);
      process.exitCode = 0;
    } else if (error instanceof UsageError) {
      process.stderr.write(`${error.usage}\n${error.prog}: error: ${error.message}\n`);
      process.exitCode = 2;
    } else {
      process.stderr.write(`${label} browser smoke failed: ${error instanceof Error ? error.message : String(error)}\n`);
      process.exitCode = 1;
    }
  }
}

export function writeStdout(text) {
  process.stdout.write(text);
}
