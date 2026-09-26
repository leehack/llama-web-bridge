// The CPython 3.12 renderings the archive and WAV ports need that
// scripts/release/json.mjs does not provide: str(OSError) with the one- and
// two-filename forms (json.mjs has the strerror() table), repr() of bytes, and
// the cp437 codec zipfile decodes legacy member names with; plus
// Path.resolve()/expanduser() for the qualification CLI. Everything else
// (repr of str, PyException, UTF-8 decoding, paths) comes from json.mjs.

import fs from 'node:fs';
import os, { constants as osConstants } from 'node:os';
import process from 'node:process';

import {
  PY_WHITESPACE, PyException, isPyException, pyPath, pyRepr, pyStrerror, pyStrftimeUtc,
  pyStrptimeUtcOrRaise, pyTypeName,
} from './json.mjs';

// True for the errors Python raises as OSError: Node system errors and a
// PyException of the OSError family.
export function isOSError(error) {
  if (isPyException(error, 'OSError')) return true;
  return error instanceof Error && typeof error.code === 'string' && typeof error.errno === 'number';
}

// str(exc) of the OSError Python would raise for the same failure, with the
// filename arguments Python would have passed. A PyException already holds
// str(exc).
export function osErrorString(error, filename, filename2) {
  if (isPyException(error, 'OSError')) return error.message;
  const errno = osConstants.errno[error.code] ?? Math.abs(error.errno);
  const base = `[Errno ${errno}] ${pyStrerror(error.code, errno)}`;
  if (filename === undefined || filename === null) return base;
  if (filename2 === undefined || filename2 === null) return `${base}: ${pyRepr(filename)}`;
  return `${base}: ${pyRepr(filename)} -> ${pyRepr(filename2)}`;
}

// repr() of a Python bytes object.
export function pyReprBytes(bytes) {
  const data = Uint8Array.from(bytes);
  const quote = data.includes(0x27) && !data.includes(0x22) ? 0x22 : 0x27;
  let out = `b${String.fromCharCode(quote)}`;
  for (const byte of data) {
    if (byte === quote || byte === 0x5c) out += `\\${String.fromCharCode(byte)}`;
    else if (byte === 0x09) out += '\\t';
    else if (byte === 0x0a) out += '\\n';
    else if (byte === 0x0d) out += '\\r';
    else if (byte < 0x20 || byte >= 0x7f) out += `\\x${byte.toString(16).padStart(2, '0')}`;
    else out += String.fromCharCode(byte);
  }
  return out + String.fromCharCode(quote);
}

// Python's cp437 codec: ASCII below 0x80, this table above.
const CP437_HIGH = 'ÇüéâäàåçêëèïîìÄÅÉæÆôöòûùÿÖÜ¢£¥₧ƒáíóúñÑªº¿⌐¬½¼¡«»░▒▓│┤╡╢╖╕╣║╗╝╜╛┐└┴┬├─┼╞╟╚╔╩╦╠═╬╧╨╤╥╙╘╒╓╫╪┘┌█▄▌▐▀αßΓπΣσµτΦΘΩδ∞φε∩≡±≥≤⌠⌡÷≈°∙·√ⁿ²■\u{a0}';

// bytes.decode('cp437'), which never fails.
export function pyDecodeCp437(bytes) {
  let out = '';
  for (const byte of bytes) out += byte < 0x80 ? String.fromCharCode(byte) : CP437_HIGH[byte - 0x80];
  return out;
}

// --- pathlib.Path.resolve() and expanduser() --------------------------------

// posixpath.normpath(text): repeated slashes and '.' dropped, '..' applied
// lexically (dropped at the root, kept at the start of a relative path),
// exactly two leading slashes kept, '' as '.'.
export function pyNormpath(text) {
  if (text === '') return '.';
  const initial = text.startsWith('/') ? (text.startsWith('//') && !text.startsWith('///') ? '//' : '/') : '';
  const parts = [];
  for (const part of text.split('/')) {
    if (part === '' || part === '.') continue;
    if (part !== '..' || (!initial && parts.length === 0) || (parts.length > 0 && parts.at(-1) === '..')) parts.push(part);
    else if (parts.length > 0) parts.pop();
  }
  return `${initial}${parts.join('/')}` || '.';
}

function pyPosixJoin(left, right) {
  if (right.startsWith('/')) return right;
  if (left === '' || left.endsWith('/')) return `${left}${right}`;
  return `${left}/${right}`;
}

// posixpath.split(text).
function pyPosixSplit(text) {
  const index = text.lastIndexOf('/') + 1;
  let head = text.slice(0, index);
  const tail = text.slice(index);
  if (head && head !== '/'.repeat(head.length)) head = head.replace(/\/+$/u, '');
  return [head, tail];
}

// posixpath._joinrealpath(path, rest, strict=False, seen).
function joinRealpath(start, rest, seen) {
  let current = start;
  if (rest.startsWith('/')) {
    rest = rest.slice(1);
    current = '/';
  }
  while (rest) {
    const slash = rest.indexOf('/');
    const name = slash < 0 ? rest : rest.slice(0, slash);
    rest = slash < 0 ? '' : rest.slice(slash + 1);
    if (!name || name === '.') continue;
    if (name === '..') {
      if (current) {
        const [head, tail] = pyPosixSplit(current);
        current = tail === '..' ? pyPosixJoin(pyPosixJoin(head, '..'), '..') : head;
      } else {
        current = '..';
      }
      continue;
    }
    const newpath = pyPosixJoin(current, name);
    let isLink = false;
    try {
      isLink = fs.lstatSync(newpath).isSymbolicLink();
    } catch {
      isLink = false;
    }
    if (!isLink) {
      current = newpath;
      continue;
    }
    if (seen.has(newpath)) {
      const cached = seen.get(newpath);
      if (cached !== null) {
        current = cached;
        continue;
      }
      return [pyPosixJoin(newpath, rest), false];
    }
    seen.set(newpath, null);
    const [resolved, ok] = joinRealpath(current, fs.readlinkSync(newpath), seen);
    if (!ok) return [pyPosixJoin(resolved, rest), false];
    current = resolved;
    seen.set(newpath, resolved);
  }
  return [current, true];
}

// os.path.abspath(text).
export function pyAbspath(text) {
  return pyNormpath(text.startsWith('/') ? text : pyPosixJoin(process.cwd(), text));
}

// str(Path(text).resolve()) (strict=False): symlinks resolved component by
// component as os.path.realpath does, missing components kept, the result
// absolute and normalized. A symlink loop raises RuntimeError, as pathlib
// raises it.
export function pyResolve(text) {
  const [resolved] = joinRealpath('', String(text), new Map());
  const result = pyAbspath(resolved);
  try {
    fs.statSync(result);
  } catch (error) {
    if (error?.code === 'ELOOP') throw new PyException('RuntimeError', `Symlink loop from ${pyRepr(result)}`);
  }
  return result;
}

// str(Path(text).expanduser()) for a normalized path: a leading '~' becomes
// $HOME (else the account's home directory) and '~name' that account's home;
// an unknown home raises RuntimeError, as pathlib raises it. Node can only
// look up the current account, so '~name' for any other account raises.
export function pyExpanduser(text) {
  if (!text.startsWith('~')) return text;
  const slash = text.indexOf('/');
  const first = slash < 0 ? text : text.slice(0, slash);
  const tail = slash < 0 ? '' : text.slice(slash);
  let home = null;
  if (first === '~') {
    home = Object.hasOwn(process.env, 'HOME') ? process.env.HOME : os.userInfo().homedir;
  } else {
    const account = os.userInfo();
    if (first.slice(1) === account.username) home = account.homedir;
  }
  if (home === null) throw new PyException('RuntimeError', 'Could not determine home directory.');
  const expanded = home.replace(/\/+$/u, '') || '/';
  if (expanded.startsWith('~')) throw new PyException('RuntimeError', 'Could not determine home directory.');
  return pyPath(`${expanded}${tail}`);
}

// --- re, str.split, tuple ordering and datetime for the orchestrator port ----

// re.fullmatch(pattern, value) for an anchored /^...$/ pattern: the match or
// null, raising TypeError for a non-str as re does.
export function pyFullmatch(pattern, value) {
  if (typeof value !== 'string') {
    throw new PyException('TypeError', `expected string or bytes-like object, got '${pyTypeName(value)}'`);
  }
  return pattern.exec(value);
}

const WHITESPACE_RUN = new RegExp(`[${PY_WHITESPACE}]+`, 'u');
const WHITESPACE_ONLY = new RegExp(`^[${PY_WHITESPACE}]*$`, 'u');

// str.split() with no separator: runs of str.isspace() characters split,
// leading and trailing whitespace dropped, no empty fields.
export function pyStrSplit(text) {
  if (WHITESPACE_ONLY.test(text)) return [];
  const fields = text.split(WHITESPACE_RUN);
  if (fields[0] === '') fields.shift();
  if (fields.at(-1) === '') fields.pop();
  return fields;
}

// Python ordering of two tuples of ints (numbers or bigints): negative, zero
// or positive.
export function pyCompareIntTuples(left, right) {
  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    const a = BigInt(left[index]);
    const b = BigInt(right[index]);
    if (a !== b) return a < b ? -1 : 1;
  }
  return left.length - right.length;
}

// Days from 0001-01-01 of a proleptic Gregorian date (datetime.toordinal() - 1).
function daysFromCivil(year, month, day) {
  const y = month <= 2 ? year - 1 : year;
  const era = Math.floor(y / 400);
  const yoe = y - era * 400;
  const doy = Math.floor((153 * (month + (month > 2 ? -3 : 9)) + 2) / 5) + day - 1;
  const doe = yoe * 365 + Math.floor(yoe / 4) - Math.floor(yoe / 100) + doy;
  // 719162 days separate 0001-01-01 from 1970-01-01; 719468 separate 0000-03-01.
  return era * 146097 + doe - 719468 + 719162;
}

function civilFromDays(days) {
  const z = days - 719162 + 719468;
  const era = Math.floor(z / 146097);
  const doe = z - era * 146097;
  const yoe = Math.floor((doe - Math.floor(doe / 1460) + Math.floor(doe / 36524) - Math.floor(doe / 146096)) / 365);
  const doy = doe - (365 * yoe + Math.floor(yoe / 4) - Math.floor(yoe / 100));
  const mp = Math.floor((5 * doy + 2) / 153);
  const day = doy - Math.floor((153 * mp + 2) / 5) + 1;
  const month = mp + (mp < 10 ? 3 : -9);
  return { year: yoe + era * 400 + (month <= 2 ? 1 : 0), month, day };
}

const DAY_SECONDS = 86400;
// datetime.min and datetime.max (whole seconds), as seconds from datetime.min.
const MAX_SECONDS = (daysFromCivil(9999, 12, 31) + 1) * DAY_SECONDS - 1;

// An aware UTC datetime with whole seconds, the subset of datetime the
// orchestrator's run-history windows use: strptime of the canonical
// "%Y-%m-%dT%H:%M:%SZ" form, adding seconds (OverflowError past datetime.min
// or datetime.max, as timedelta arithmetic raises it), comparison and strftime.
export class PyUtcDatetime {
  constructor(seconds) {
    if (!Number.isSafeInteger(seconds) || seconds < 0 || seconds > MAX_SECONDS) {
      throw new PyException('OverflowError', 'date value out of range');
    }
    this.seconds = seconds;
    Object.freeze(this);
  }

  static fromFields({ year, month, day, hour, minute, second }) {
    return new PyUtcDatetime(daysFromCivil(year, month, day) * DAY_SECONDS + hour * 3600 + minute * 60 + second);
  }

  // datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).
  static strptime(value) {
    return PyUtcDatetime.fromFields(pyStrptimeUtcOrRaise(value));
  }

  // self + timedelta(seconds=delta).
  addSeconds(delta) {
    return new PyUtcDatetime(this.seconds + delta);
  }

  fields() {
    const days = Math.floor(this.seconds / DAY_SECONDS);
    const rest = this.seconds - days * DAY_SECONDS;
    return {
      ...civilFromDays(days),
      hour: Math.floor(rest / 3600),
      minute: Math.floor((rest % 3600) / 60),
      second: rest % 60,
    };
  }

  // .strftime("%Y-%m-%dT%H:%M:%SZ") with glibc's unpadded %Y.
  strftime() {
    return pyStrftimeUtc(this.fields());
  }
}
