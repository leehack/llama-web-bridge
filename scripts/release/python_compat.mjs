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
  PyException, isPyException, pyPath, pyRepr, pyStrerror,
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
