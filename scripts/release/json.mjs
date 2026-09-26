// Python-compatible primitives shared by the Node release tooling.
//
// The release modules are ports of Python scripts whose output bytes, accepted
// inputs, and error text are part of the release contract. This module holds
// the pieces of CPython 3.12 behaviour (the version on the ubuntu-24.04
// runners) those ports need to reproduce exactly:
//
// - the JSON value model: an int is a safe-integer number or a bigint, a float
//   is always a PyFloat (so `1` and `1.0` stay apart), a dict is a plain object;
// - json.loads with the C scanner's grammar, error messages and positions,
//   object_pairs_hook and parse_constant, and json.dumps (compact, indented,
//   sort_keys, ensure_ascii, allow_nan);
// - repr(), ==, dict.get and type names for error messages;
// - bytes.decode("utf-8"), Path.read_text/read_bytes (with OSError text),
//   str.splitlines/strip, base64.b64decode(validate=True), int(), strptime with
//   a strftime round-trip, and urllib.parse.quote/urlencode;
// - re's \w and \d, str.isprintable and int()'s digits as Unicode 15.0 (the
//   data of Python 3.12) defines them, whatever Unicode version Node carries
//   (./unicode15.mjs).
//
// ==, repr() and json.dumps walk containers iteratively, so they handle any
// nesting json.loads accepts without exhausting the JS stack.
//
// Zero dependencies: privileged workflow jobs never run `npm ci`.

import fs from 'node:fs';
import process from 'node:process';

import { ASSIGNED_AFTER_UNICODE_15_CLASS } from './unicode15.mjs';

// --- Python exceptions -------------------------------------------------------

const PY_BASES = Object.freeze({
  AttributeError: [],
  'binascii.Error': ['ValueError'],
  FileExistsError: ['OSError'],
  FileNotFoundError: ['OSError'],
  IsADirectoryError: ['OSError'],
  JSONDecodeError: ['ValueError'],
  KeyError: ['LookupError'],
  KeyboardInterrupt: ['BaseException'],
  NotADirectoryError: ['OSError'],
  OSError: [],
  PermissionError: ['OSError'],
  ProcessLookupError: ['OSError'],
  RecursionError: ['RuntimeError'],
  TypeError: [],
  UnicodeDecodeError: ['UnicodeError', 'ValueError'],
  UnicodeEncodeError: ['UnicodeError', 'ValueError'],
  ValueError: [],
});

const PY_QUALIFIED_NAMES = Object.freeze({ JSONDecodeError: 'json.decoder.JSONDecodeError' });

// A Python exception the ported code raises where the Python code would. Its
// message is str(exception); `pyType` names the Python class, and isInstance
// answers isinstance() against that class and its bases.
export class PyException extends Error {
  constructor(pyType, message) {
    super(message);
    this.name = pyType;
    this.pyType = pyType;
  }

  isInstance(...types) {
    return types.some((type) => type === this.pyType || (PY_BASES[this.pyType] ?? []).includes(type));
  }

  // The last line of the traceback Python prints when this escapes main().
  get tracebackLine() {
    const name = PY_QUALIFIED_NAMES[this.pyType] ?? this.pyType;
    return this.message === '' ? name : `${name}: ${this.message}`;
  }
}

export function isPyException(error, ...types) {
  return error instanceof PyException && error.isInstance(...types);
}

// json.JSONDecodeError(msg, doc, pos); pos counts code points, as a str index.
export class JSONDecodeError extends PyException {
  constructor(msg, codePoints, pos) {
    let lineno = 1;
    let lastNewline = -1;
    for (let index = 0; index < pos; index += 1) {
      if (codePoints[index] === 0x0a) {
        lineno += 1;
        lastNewline = index;
      }
    }
    const colno = pos - lastNewline;
    super('JSONDecodeError', `${msg}: line ${lineno} column ${colno} (char ${pos})`);
    this.msg = msg;
    this.pos = pos;
    this.lineno = lineno;
    this.colno = colno;
  }
}

// --- The JSON value model ----------------------------------------------------

// A Python float. json.loads returns one for every literal with a fraction or
// exponent and for NaN/Infinity, so `1.0` is never confused with the int 1.
export class PyFloat {
  constructor(value) {
    this.value = Number(value);
    Object.freeze(this);
  }
}

// json.loads returns the same float object for every NaN (and each infinity):
// its default parse_constant is a lookup in a module-level table. Container
// comparison checks identity first, so [NaN] == [NaN] holds for two parses
// while NaN == NaN does not.
export const PY_NAN = new PyFloat(Number.NaN);
export const PY_INFINITY = new PyFloat(Number.POSITIVE_INFINITY);
export const PY_NEGATIVE_INFINITY = new PyFloat(Number.NEGATIVE_INFINITY);
const PY_CONSTANTS = Object.freeze({ NaN: PY_NAN, Infinity: PY_INFINITY, '-Infinity': PY_NEGATIVE_INFINITY });

// An int: a safe-integer number (never -0) or a bigint. A non-integral or -0
// number is a float, as Playwright's Python client would have read it. A bool
// is never one: this is isinstance(value, int) and not isinstance(value, bool).
export function isPyInt(value) {
  if (typeof value === 'bigint') return true;
  return typeof value === 'number' && Number.isSafeInteger(value) && !Object.is(value, -0);
}

export function isPyFloat(value) {
  return value instanceof PyFloat || (typeof value === 'number' && !isPyInt(value));
}

// isinstance(value, int), which bool satisfies.
export function isPyIntLike(value) {
  return typeof value === 'boolean' || isPyInt(value);
}

export function isDict(value) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

export function isList(value) {
  return Array.isArray(value);
}

export function pyTypeName(value) {
  if (value === null || value === undefined) return 'NoneType';
  if (typeof value === 'boolean') return 'bool';
  if (typeof value === 'string') return 'str';
  if (value instanceof PyFloat) return 'float';
  if (typeof value === 'number' || typeof value === 'bigint') return isPyInt(value) ? 'int' : 'float';
  if (Array.isArray(value)) return 'list';
  if (value instanceof Uint8Array) return 'bytes';
  if (isDict(value) || value instanceof Map) return 'dict';
  return value?.constructor?.name ?? typeof value;
}

// A JS object lists array-index keys ("0", "1", ...) first, where a dict keeps
// insertion order. A dict built here that holds such a key records its order
// under this symbol; pyKeys honours it while the key set is unchanged.
const KEY_ORDER = Symbol('pyKeyOrder');
const ARRAY_INDEX_KEY = /^(?:0|[1-9][0-9]*)$/;

function isArrayIndexKey(key) {
  return ARRAY_INDEX_KEY.test(key) && Number(key) < 4294967295;
}

// obj[key] = value without the __proto__ setter.
export function pySetItem(object, key, value) {
  if (key === '__proto__') {
    Object.defineProperty(object, key, { value, writable: true, enumerable: true, configurable: true });
  } else {
    object[key] = value;
  }
}

// dict(pairs): a repeated key keeps its first position and its last value.
export function pyDict(pairs) {
  const object = {};
  const order = [];
  let reordered = false;
  for (const [key, value] of pairs) {
    if (!Object.hasOwn(object, key)) {
      order.push(key);
      if (isArrayIndexKey(key)) reordered = true;
    }
    pySetItem(object, key, value);
  }
  if (reordered) Object.defineProperty(object, KEY_ORDER, { value: order });
  return object;
}

// list(d): the dict's keys in insertion order.
export function pyKeys(object) {
  if (object instanceof Map) return [...object.keys()];
  const keys = Object.keys(object);
  const order = object[KEY_ORDER];
  if (order && order.length === keys.length && order.every((key) => Object.hasOwn(object, key))) return [...order];
  return keys;
}

// d.items() in insertion order.
export function pyItems(object) {
  if (object instanceof Map) return [...object.entries()];
  return pyKeys(object).map((key) => [key, object[key]]);
}

function pyValueOrNone(value) {
  return value === undefined ? null : value;
}

// d.get(key, default) on a value that must be a dict; Python raises
// AttributeError for anything else. A key that holds undefined holds None.
export function pyGet(object, key, fallback = null) {
  if (object instanceof Map) return object.has(key) ? pyValueOrNone(object.get(key)) : fallback;
  if (!isDict(object)) throw new PyException('AttributeError', `'${pyTypeName(object)}' object has no attribute 'get'`);
  return Object.hasOwn(object, key) ? pyValueOrNone(object[key]) : fallback;
}

// key in d.
export function pyHasKey(object, key) {
  if (object instanceof Map) return object.has(key);
  return Object.hasOwn(object, key);
}

// set(d) == set(keys).
export function pyKeySetEquals(object, keys) {
  const actual = pyKeys(object);
  const expected = new Set(keys);
  return actual.length === expected.size && actual.every((key) => expected.has(key));
}

// A number's Python view: an exact int (bigint) or a float (number), or null
// for a non-number. bool is the int 0 or 1.
function pyNumber(value) {
  if (typeof value === 'boolean') return { int: value ? 1n : 0n };
  if (typeof value === 'bigint') return { int: value };
  if (value instanceof PyFloat) return { float: value.value };
  if (typeof value === 'number') return isPyInt(value) ? { int: BigInt(value) } : { float: value };
  return null;
}

// `left == right` for JSON-shaped values: 1 == 1.0 == True, ints compare
// exactly with floats, NaN != NaN at the top level, lists by item and dicts by
// key set and value, both independent of dict order. undefined is None.
//
// Nested items compare as PyObject_RichCompareBool does, identity first,
// which is how two JSON NaNs (the same PY_NAN object) compare equal inside
// containers. The walk is iterative, so any nesting json.loads accepts
// compares without exhausting the JS stack; nothing in it can raise, so the
// result is the conjunction Python's recursive comparison computes.
export function pyEquals(left, right) {
  const pending = [[left, right, false]];
  while (pending.length > 0) {
    let [a, b, identityFirst] = pending.pop();
    if (identityFirst && a === b) continue;
    if (a === undefined) a = null;
    if (b === undefined) b = null;
    if (a === null || b === null) {
      if (a !== b) return false;
      continue;
    }
    const x = pyNumber(a);
    const y = pyNumber(b);
    if (x || y) {
      if (!x || !y) return false;
      if ('int' in x && 'int' in y) {
        if (x.int !== y.int) return false;
      } else if ('float' in x && 'float' in y) {
        if (x.float !== y.float) return false;
      } else {
        const [int, float] = 'int' in x ? [x.int, y.float] : [y.int, x.float];
        if (!Number.isInteger(float) || BigInt(float) !== int) return false;
      }
      continue;
    }
    if (typeof a === 'string' || typeof b === 'string') {
      if (a !== b) return false;
      continue;
    }
    if (Array.isArray(a) || Array.isArray(b)) {
      if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
      for (let index = a.length - 1; index >= 0; index -= 1) pending.push([a[index], b[index], true]);
      continue;
    }
    const aDict = isDict(a) || a instanceof Map;
    const bDict = isDict(b) || b instanceof Map;
    if (aDict || bDict) {
      if (!aDict || !bDict) return false;
      const items = pyItems(a);
      if (items.length !== pyKeys(b).length) return false;
      for (let index = items.length - 1; index >= 0; index -= 1) {
        const [key, value] = items[index];
        if (!pyHasKey(b, key)) return false;
        pending.push([pyValueOrNone(value), pyGet(b, key), true]);
      }
      continue;
    }
    if (a !== b) return false;
  }
  return true;
}

// --- repr() --------------------------------------------------------------------

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

// --- Unicode 15.0 character classes -------------------------------------------

// Python 3.12's unicodedata and re are Unicode 15.0.0; Node's \p{...} data is
// newer (16 or 17 on Node 24). A code point assigned since then is Cn to
// Python, so each class below leaves those out (./unicode15.mjs), and
// tests/release/unicode15_test.mjs checks every one against Python 3.12 over
// all code points.
const ASSIGNED_AFTER_UNICODE_15 = `[${ASSIGNED_AFTER_UNICODE_15_CLASS}]`;

// Regex source (u flag) for one character of Python's re \d in a str pattern,
// which is str.isdecimal(): the Unicode 15.0 Nd characters.
export const PY_DECIMAL = `(?:(?!${ASSIGNED_AFTER_UNICODE_15})\\p{Nd})`;

// Regex source (u flag) for one character of Python's re \w in a str pattern:
// str.isalnum() or '_'.
export const PY_WORD = `(?:(?!${ASSIGNED_AFTER_UNICODE_15})[\\p{L}\\p{N}_])`;

// str.isprintable is false for these categories, except for the space itself.
const NOT_PRINTABLE = new RegExp(`[\\p{Cc}\\p{Cf}\\p{Cs}\\p{Co}\\p{Cn}\\p{Zl}\\p{Zp}\\p{Zs}${ASSIGNED_AFTER_UNICODE_15_CLASS}]`, 'u');

export function pyStrRepr(text) {
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

function scalarRepr(value) {
  if (value === null || value === undefined) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  if (value instanceof PyFloat) return pyFloatRepr(value.value);
  if (typeof value === 'number' || typeof value === 'bigint') return isPyInt(value) ? String(value) : pyFloatRepr(value);
  if (typeof value === 'string') return pyStrRepr(value);
  return String(value);
}

// repr() of the value Python would hold. Iterative, so any nesting json.loads
// accepts renders without exhausting the JS stack.
export function pyRepr(value) {
  const out = [];
  // Work in reverse order: a string is emitted as is, { value } rendered.
  const work = [{ value }];
  while (work.length > 0) {
    const item = work.pop();
    if (typeof item === 'string') {
      out.push(item);
      continue;
    }
    const current = item.value;
    if (Array.isArray(current)) {
      work.push(']');
      for (let index = current.length - 1; index >= 0; index -= 1) {
        work.push({ value: current[index] });
        if (index > 0) work.push(', ');
      }
      out.push('[');
    } else if (isDict(current) || current instanceof Map) {
      const items = pyItems(current);
      work.push('}');
      for (let index = items.length - 1; index >= 0; index -= 1) {
        work.push({ value: items[index][1] }, ': ', { value: items[index][0] });
        if (index > 0) work.push(', ');
      }
      out.push('{');
    } else {
      out.push(scalarRepr(current));
    }
  }
  return out.join('');
}

// str(): a str is itself, anything else its repr (True, None, 1.0, ...).
export function pyStr(value) {
  return typeof value === 'string' ? value : pyRepr(value);
}

// --- json.loads ----------------------------------------------------------------

// CPython's C recursion budget runs out after this many nested containers when
// json.loads is called from a script's main() (measured on 3.12.11 and
// 3.12.14); the next level raises RecursionError. Near that limit, what the C
// scanner calls out to needs budget of its own, and a RecursionError replaces
// the call. With `remaining` levels left at the scanner's depth:
// - creating a C exception (StopIteration, ValueError) or calling the default
//   parse_constant needs 1: at 0, "... while calling a Python object";
// - raising a JSONDecodeError needs 4: at 3 and 0, "... while calling a Python
//   object", at 2 and 1, the bare message;
// - calling a Python hook needs 2 (the bare message), and a repr() inside the
//   hook 3 ("... while getting the repr of an object").
export const PY_JSON_MAX_DEPTH = 9997;

function recursionError(suffix = '') {
  return new PyException('RecursionError', `maximum recursion depth exceeded${suffix}`);
}

// repr(value) inside a json.loads hook, which raises RecursionError where
// CPython's budget has no room left for the repr call. `frame` is the second
// argument pyJsonLoads passes the hook.
export function pyHookRepr(frame, value) {
  if (frame && frame.depth > frame.maxDepth - 3) throw recursionError(' while getting the repr of an object');
  return pyRepr(value);
}

// sys.get_int_max_str_digits() default.
export const PY_INT_MAX_STR_DIGITS = 4300;

function intTooLong(digits) {
  return new PyException('ValueError', `Exceeds the limit (${PY_INT_MAX_STR_DIGITS} digits) for integer string conversion: value has ${digits} digits; use sys.set_int_max_str_digits() to increase the limit`);
}

// An int from its decimal digits (with an optional '-'): a number while it is a
// safe integer, else a bigint.
function intFromDigits(text) {
  const value = BigInt(text);
  return value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : value;
}

function isJsonWhitespace(code) {
  return code === 0x20 || code === 0x09 || code === 0x0a || code === 0x0d;
}

function isDigit(code) {
  return code >= 0x30 && code <= 0x39;
}

function hexValue(code) {
  if (code >= 0x30 && code <= 0x39) return code - 0x30;
  if (code >= 0x61 && code <= 0x66) return code - 0x61 + 10;
  if (code >= 0x41 && code <= 0x46) return code - 0x41 + 10;
  return -1;
}

class StopIteration {
  constructor(value) {
    this.value = value;
  }
}

// json.loads(text, object_pairs_hook=..., parse_constant=...) over a str.
//
// - objectPairsHook(pairs, frame) receives [key, value] pairs when an object
//   closes, exactly when Python calls it; without it an object is
//   pyDict(pairs).
// - parseConstant(name, frame) receives 'NaN', 'Infinity' or '-Infinity';
//   without it they are the PY_NAN/PY_INFINITY/PY_NEGATIVE_INFINITY
//   singletons, as plain json.loads accepts them.
// - `frame` is { depth, maxDepth }: the C recursion depth of the call, for
//   pyHookRepr.
//
// Errors are JSONDecodeError with CPython 3.12's message and position, and
// PyException ValueError (an int over 4300 digits) or RecursionError, which
// Python raises uncaught from the same places.
export function pyJsonLoads(text, { objectPairsHook = null, parseConstant = null, maxDepth = PY_JSON_MAX_DEPTH } = {}) {
  if (typeof text !== 'string') throw new PyException('TypeError', `the JSON object must be str, bytes or bytearray, not ${pyTypeName(text)}`);
  const cps = [];
  const offsets = [];
  for (let offset = 0; offset < text.length;) {
    const code = text.codePointAt(offset);
    cps.push(code);
    offsets.push(offset);
    offset += code > 0xffff ? 2 : 1;
  }
  offsets.push(text.length);
  const length = cps.length;
  const endIdx = length - 1;
  const fail = (msg, pos, depth = 0) => {
    const remaining = maxDepth - depth;
    if (remaining >= 4) return new JSONDecodeError(msg, cps, pos);
    return recursionError(remaining === 3 || remaining === 0 ? ' while calling a Python object' : '');
  };
  // A C-level exception or call at `depth`.
  const cCall = (depth) => {
    if (maxDepth - depth < 1) throw recursionError(' while calling a Python object');
  };
  const substring = (start, end) => text.slice(offsets[start], offsets[end]);

  if (cps[0] === 0xfeff) throw fail('Unexpected UTF-8 BOM (decode using utf-8-sig)', 0);

  // scanstring_unicode: `end` is the index after the opening quote.
  const scanString = (end, depth) => {
    const begin = end - 1;
    let chunks = '';
    for (;;) {
      let next = end;
      let c = 0;
      for (; next < length; next += 1) {
        c = cps[next];
        if (c === 0x22 || c === 0x5c) break;
        if (c <= 0x1f) throw fail('Invalid control character at', next, depth);
      }
      if (c !== 0x22 && c !== 0x5c) throw fail('Unterminated string starting at', begin, depth);
      chunks += substring(end, next);
      next += 1;
      if (c === 0x22) return [chunks, next];
      if (next === length) throw fail('Unterminated string starting at', begin, depth);
      c = cps[next];
      if (c !== 0x75) {
        end = next + 1;
        const simple = { 0x22: '"', 0x5c: '\\', 0x2f: '/', 0x62: '\b', 0x66: '\f', 0x6e: '\n', 0x72: '\r', 0x74: '\t' }[c];
        if (simple === undefined) throw fail('Invalid \\escape', end - 2, depth);
        chunks += simple;
        continue;
      }
      next += 1;
      end = next + 4;
      if (end >= length) throw fail('Invalid \\uXXXX escape', next - 1, depth);
      let unit = 0;
      for (; next < end; next += 1) {
        const digit = hexValue(cps[next]);
        if (digit < 0) throw fail('Invalid \\uXXXX escape', end - 5, depth);
        unit = (unit << 4) | digit;
      }
      let codePoint = unit;
      if (unit >= 0xd800 && unit <= 0xdbff && end + 6 < length && cps[next] === 0x5c && cps[next + 1] === 0x75) {
        next += 2;
        end += 6;
        let low = 0;
        for (; next < end; next += 1) {
          const digit = hexValue(cps[next]);
          if (digit < 0) throw fail('Invalid \\uXXXX escape', end - 5, depth);
          low = (low << 4) | digit;
        }
        if (low >= 0xdc00 && low <= 0xdfff) codePoint = 0x10000 + ((unit - 0xd800) << 10) + (low - 0xdc00);
        else end -= 6;
      }
      chunks += String.fromCodePoint(codePoint);
    }
  };

  const stopIteration = (value, depth) => {
    cCall(depth);
    return new StopIteration(value);
  };

  // _match_number_unicode.
  const matchNumber = (start, depth) => {
    let idx = start;
    let isFloat = false;
    if (cps[idx] === 0x2d) {
      idx += 1;
      if (idx > endIdx) throw stopIteration(start, depth);
    }
    if (cps[idx] >= 0x31 && cps[idx] <= 0x39) {
      idx += 1;
      while (idx <= endIdx && isDigit(cps[idx])) idx += 1;
    } else if (cps[idx] === 0x30) {
      idx += 1;
    } else {
      throw stopIteration(start, depth);
    }
    if (idx < endIdx && cps[idx] === 0x2e && isDigit(cps[idx + 1])) {
      isFloat = true;
      idx += 2;
      while (idx <= endIdx && isDigit(cps[idx])) idx += 1;
    }
    if (idx < endIdx && (cps[idx] === 0x65 || cps[idx] === 0x45)) {
      const eStart = idx;
      idx += 1;
      if (idx < endIdx && (cps[idx] === 0x2d || cps[idx] === 0x2b)) idx += 1;
      while (idx <= endIdx && isDigit(cps[idx])) idx += 1;
      if (isDigit(cps[idx - 1])) isFloat = true;
      else idx = eStart;
    }
    const numberText = substring(start, idx);
    if (isFloat) return [new PyFloat(Number(numberText)), idx];
    const digits = numberText.startsWith('-') ? numberText.length - 1 : numberText.length;
    if (digits > PY_INT_MAX_STR_DIGITS) {
      cCall(depth);
      throw intTooLong(digits);
    }
    return [intFromDigits(numberText), idx];
  };

  const callHook = (hook, argument, depth) => {
    if (depth > maxDepth - 2) throw recursionError();
    return hook(argument, { depth, maxDepth });
  };

  const constant = (name, idx, depth) => {
    let value;
    if (parseConstant) {
      value = callHook(parseConstant, name, depth);
    } else {
      cCall(depth);
      value = PY_CONSTANTS[name];
    }
    return [value, idx + name.length];
  };

  const matches = (idx, word) => {
    if (!(idx + word.length - 1 < length)) return false;
    for (let offset = 1; offset < word.length; offset += 1) {
      if (cps[idx + offset] !== word.charCodeAt(offset)) return false;
    }
    return true;
  };

  const skipWhitespace = (idx) => {
    while (idx <= endIdx && isJsonWhitespace(cps[idx])) idx += 1;
    return idx;
  };

  const finishObject = (pairs, depth) => (objectPairsHook ? callHook(objectPairsHook, pairs, depth) : pyDict(pairs));

  // scan_once_unicode, iterative so nesting is bounded by maxDepth (as the C
  // recursion budget bounds Python) rather than by the JS stack.
  const scanOnce = (startIdx) => {
    const stack = [];
    let idx = startIdx;
    // Parse the object key at idx (after '{' or ','), then the ':' delimiter.
    const readKey = (frame) => {
      const depth = stack.length;
      if (idx > endIdx || cps[idx] !== 0x22) throw fail('Expecting property name enclosed in double quotes', idx, depth);
      const [key, next] = scanString(idx + 1, depth);
      frame.key = key;
      idx = skipWhitespace(next);
      if (idx > endIdx || cps[idx] !== 0x3a) throw fail("Expecting ':' delimiter", idx, depth);
      idx = skipWhitespace(idx + 1);
    };
    for (;;) {
      let value;
      let haveValue = false;
      if (idx >= length) throw stopIteration(idx, stack.length);
      const c = cps[idx];
      if (c === 0x22) {
        [value, idx] = scanString(idx + 1, stack.length);
        haveValue = true;
      } else if (c === 0x7b || c === 0x5b) {
        if (stack.length >= maxDepth) {
          throw recursionError(` while decoding a JSON ${c === 0x7b ? 'object' : 'array'} from a unicode string`);
        }
        const frame = c === 0x7b ? { object: true, pairs: [], key: null } : { object: false, items: [] };
        idx = skipWhitespace(idx + 1);
        if (idx <= endIdx && cps[idx] === (frame.object ? 0x7d : 0x5d)) {
          idx += 1;
          value = frame.object ? finishObject(frame.pairs, stack.length + 1) : frame.items;
          haveValue = true;
        } else {
          stack.push(frame);
          if (frame.object) readKey(frame);
          continue;
        }
      } else if (c === 0x6e && matches(idx, 'null')) {
        [value, idx] = [null, idx + 4];
        haveValue = true;
      } else if (c === 0x74 && matches(idx, 'true')) {
        [value, idx] = [true, idx + 4];
        haveValue = true;
      } else if (c === 0x66 && matches(idx, 'false')) {
        [value, idx] = [false, idx + 5];
        haveValue = true;
      } else if (c === 0x4e && matches(idx, 'NaN')) {
        [value, idx] = constant('NaN', idx, stack.length);
        haveValue = true;
      } else if (c === 0x49 && matches(idx, 'Infinity')) {
        [value, idx] = constant('Infinity', idx, stack.length);
        haveValue = true;
      } else if (c === 0x2d && matches(idx, '-Infinity')) {
        [value, idx] = constant('-Infinity', idx, stack.length);
        haveValue = true;
      }
      if (!haveValue) [value, idx] = matchNumber(idx, stack.length);
      // Hand the value to the enclosing containers, closing any that end here.
      for (;;) {
        const frame = stack.at(-1);
        if (!frame) return [value, idx];
        if (frame.object) frame.pairs.push([frame.key, value]);
        else frame.items.push(value);
        idx = skipWhitespace(idx);
        if (idx <= endIdx && cps[idx] === (frame.object ? 0x7d : 0x5d)) {
          idx += 1;
          const depth = stack.length;
          stack.pop();
          value = frame.object ? finishObject(frame.pairs, depth) : frame.items;
          continue;
        }
        if (idx > endIdx || cps[idx] !== 0x2c) throw fail("Expecting ',' delimiter", idx, stack.length);
        idx = skipWhitespace(idx + 1);
        if (frame.object) readKey(frame);
        break;
      }
    }
  };

  const start = skipWhitespace(0);
  let value;
  let end;
  try {
    [value, end] = scanOnce(start);
  } catch (error) {
    if (error instanceof StopIteration) throw fail('Expecting value', error.value);
    throw error;
  }
  end = skipWhitespace(end);
  if (end !== length) throw fail('Extra data', end);
  return value;
}

// --- json.dumps ----------------------------------------------------------------

const JSON_ESCAPES = new Map([
  ['"', '\\"'], ['\\', '\\\\'], ['\n', '\\n'], ['\r', '\\r'], ['\t', '\\t'], ['\b', '\\b'], ['\f', '\\f'],
]);

// json.dumps(str) with ensure_ascii: UTF-16 code units outside ' '..'~' as \uXXXX.
export function pyJsonString(text) {
  return `"${text.replace(/[\\"]|[^\x20-\x7e]/g, (char) => JSON_ESCAPES.get(char)
    ?? `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`)}"`;
}

// Python sorts str keys by code point, not by UTF-16 code unit.
export function compareCodePoints(left, right) {
  const a = Array.from(left, (char) => char.codePointAt(0));
  const b = Array.from(right, (char) => char.codePointAt(0));
  for (let i = 0; i < Math.min(a.length, b.length); i += 1) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return a.length - b.length;
}

function pyNumberJson(value, allowNan) {
  let float;
  if (value instanceof PyFloat) float = value.value;
  else if (isPyInt(value)) return String(value);
  else float = value;
  if (Number.isFinite(float)) return pyFloatRepr(float);
  if (!allowNan) throw new PyException('ValueError', `Out of range float values are not JSON compliant: ${pyFloatRepr(float)}`);
  if (Number.isNaN(float)) return 'NaN';
  return float > 0 ? 'Infinity' : '-Infinity';
}

// json.dumps(value, indent=indent, sort_keys=sortKeys, allow_nan=allowNan)
// with ensure_ascii and the default separators (', ' and ': ' without an
// indent, ',' and ': ' with one). undefined is None; a dict's keys must be str.
//
// maxNesting models the recursion limit of the pure-Python encoder json.dumps
// uses with an indent: every container takes a Python frame, and so does a
// float (_floatstr), so either one inside `maxNesting` enclosing containers
// raises RecursionError, in the order Python meets it. The budget depends on
// the Python caller's own stack depth, so the caller supplies it.
export function pyJsonDumps(value, { indent = null, sortKeys = false, allowNan = true, maxNesting = Infinity } = {}) {
  const itemSeparator = indent === null ? ', ' : ',';
  const recursionError = () => new PyException('RecursionError', 'maximum recursion depth exceeded');
  // The text of a scalar at `level`, or a new frame for a container. Checks run
  // in the order the recursive encoder meets them.
  const open = (item, level, prefix) => {
    if (item === null || item === undefined) return 'null';
    if (item === true) return 'true';
    if (item === false) return 'false';
    if (typeof item === 'number' || typeof item === 'bigint' || item instanceof PyFloat) {
      if (level >= maxNesting && !isPyInt(item)) throw recursionError();
      return pyNumberJson(item, allowNan);
    }
    if (typeof item === 'string') return pyJsonString(item);
    if (level >= maxNesting && (Array.isArray(item) || isDict(item) || item instanceof Map)) throw recursionError();
    if (Array.isArray(item)) {
      const children = [];
      for (let index = 0; index < item.length; index += 1) children.push(['', item[index]]);
      return { level, prefix, children, next: 0, entries: [], brackets: ['[', ']'] };
    }
    if (isDict(item) || item instanceof Map) {
      let items = pyItems(item);
      if (items.some(([key]) => typeof key !== 'string')) {
        throw new PyException('TypeError', 'the Node port only serializes str dict keys');
      }
      if (sortKeys) items = items.sort(([left], [right]) => compareCodePoints(left, right));
      const children = items.map(([key, element]) => [`${pyJsonString(key)}: `, element]);
      return { level, prefix, children, next: 0, entries: [], brackets: ['{', '}'] };
    }
    throw new PyException('TypeError', `Object of type ${pyTypeName(item)} is not JSON serializable`);
  };
  const close = ({ level, entries, brackets: [first, last] }) => {
    if (entries.length === 0) return `${first}${last}`;
    if (indent === null) return `${first}${entries.join(itemSeparator)}${last}`;
    const inner = `\n${' '.repeat(indent * (level + 1))}`;
    return `${first}${inner}${entries.join(`${itemSeparator}${inner}`)}\n${' '.repeat(indent * level)}${last}`;
  };
  // Iterative, so any nesting json.loads accepts encodes without exhausting
  // the JS stack.
  const root = open(value, 0, '');
  if (typeof root === 'string') return root;
  const stack = [root];
  for (;;) {
    const frame = stack.at(-1);
    if (frame.next < frame.children.length) {
      const [prefix, child] = frame.children[frame.next];
      frame.next += 1;
      const encoded = open(child, frame.level + 1, prefix);
      if (typeof encoded === 'string') frame.entries.push(`${prefix}${encoded}`);
      else stack.push(encoded);
      continue;
    }
    stack.pop();
    const text = close(frame);
    if (stack.length === 0) return text;
    stack.at(-1).entries.push(`${frame.prefix}${text}`);
  }
}

// json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", the
// canonical form of every byte-compared release JSON file.
export function pyCanonicalJson(value) {
  return `${pyJsonDumps(value, { indent: 2, sortKeys: true, allowNan: false })}\n`;
}

// --- str and bytes -------------------------------------------------------------

// The characters str.isspace() accepts (and re's \s and str.split/strip use).
export const PY_WHITESPACE = '\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';
const STRIP = new RegExp(`^[${PY_WHITESPACE}]+|[${PY_WHITESPACE}]+$`, 'gu');

// str.strip().
export function pyStrip(text) {
  return text.replace(STRIP, '');
}

// Python re's \S: one character that str.isspace() rejects.
export const PY_NON_SPACE_CLASS = `[^${PY_WHITESPACE}]`;

const LINE_BREAK = /\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]/g;

// str.splitlines(): every Unicode line boundary, no trailing empty line.
export function pySplitlines(text) {
  const lines = text.split(LINE_BREAK);
  if (lines.length > 0 && lines.at(-1) === '') lines.pop();
  return lines;
}

function utf8Error(bytes, start, end, reason) {
  const where = end - start === 1
    ? `byte 0x${bytes[start].toString(16).padStart(2, '0')} in position ${start}`
    : `bytes in position ${start}-${end - 1}`;
  return new PyException('UnicodeDecodeError', `'utf-8' codec can't decode ${where}: ${reason}`);
}

// The byte range [start, end) and reason of the first invalid UTF-8 sequence
// at or after `from`, as CPython's decoder reports it, or null.
function utf8InvalidRange(bytes, from = 0) {
  const isCont = (value) => value >= 0x80 && value <= 0xbf;
  const end = bytes.length;
  for (let s = from; s < end;) {
    const ch = bytes[s];
    if (ch < 0x80) {
      s += 1;
      continue;
    }
    if (ch < 0xc2 || ch > 0xf4) return [s, s + 1, 'invalid start byte'];
    const need = ch < 0xe0 ? 2 : ch < 0xf0 ? 3 : 4;
    const ch2 = bytes[s + 1];
    const badSecond = (value) => !isCont(value)
      || (ch === 0xe0 && value < 0xa0) || (ch === 0xed && value >= 0xa0)
      || (ch === 0xf0 && value < 0x90) || (ch === 0xf4 && value >= 0x90);
    if (end - s < need) {
      if (end - s >= 2 && badSecond(ch2)) return [s, s + 1, 'invalid continuation byte'];
      if (need === 4 && end - s >= 3 && !isCont(bytes[s + 2])) return [s, s + 2, 'invalid continuation byte'];
      return [s, end, 'unexpected end of data'];
    }
    if (badSecond(ch2)) return [s, s + 1, 'invalid continuation byte'];
    if (need >= 3 && !isCont(bytes[s + 2])) return [s, s + 2, 'invalid continuation byte'];
    if (need === 4 && !isCont(bytes[s + 3])) return [s, s + 3, 'invalid continuation byte'];
    s += need;
  }
  return null;
}

// The first error CPython's UTF-8 decoder reports for bytes that are not
// valid UTF-8, with the same byte range and reason.
function firstUtf8Error(bytes) {
  const range = utf8InvalidRange(bytes);
  return range && utf8Error(bytes, ...range);
}

const UTF8 = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });

// bytes.decode("utf-8"): strict, keeping a BOM, raising UnicodeDecodeError
// with CPython's message.
export function pyDecodeUtf8(bytes) {
  const error = firstUtf8Error(bytes);
  if (error) throw error;
  return UTF8.decode(bytes);
}

// Universal newlines, as a text-mode read applies them.
export function pyUniversalNewlines(text) {
  return text.replace(/\r\n?/g, '\n');
}

// --- int() -------------------------------------------------------------------------

const DECIMAL_DIGIT = new RegExp(`^${PY_DECIMAL}$`, 'u');

// The value of a Unicode decimal digit: Nd characters come in runs of ten
// that start at zero.
function decimalValue(char) {
  let code = char.codePointAt(0);
  if (code >= 0x30 && code <= 0x39) return code - 0x30;
  let zero = code;
  while (DECIMAL_DIGIT.test(String.fromCodePoint(zero - 1))) zero -= 1;
  code -= zero;
  return code % 10;
}

// The non-ASCII characters str.isspace() accepts.
const NON_ASCII_SPACE = /^[\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]$/u;

// int() first maps the text to ASCII: an ASCII character stays, a non-ASCII
// space becomes ' ', a decimal digit its ASCII digit, and anything else ends
// the text as '?' (_PyUnicode_TransformDecimalAndSpaceToASCII).
function intAscii(text) {
  let out = '';
  for (const char of text) {
    const code = char.codePointAt(0);
    if (code < 0x7f) out += char;
    else if (NON_ASCII_SPACE.test(char)) out += ' ';
    else if (DECIMAL_DIGIT.test(char)) out += String(decimalValue(char));
    else return `${out}?`;
  }
  return out;
}

// Py_ISSPACE: the ASCII whitespace PyLong_FromString skips. \x1c-\x1f are not
// among them, so they make the text invalid.
const isAsciiSpace = (char) => char === ' ' || (char >= '\t' && char <= '\r');

// int(text) for a str, raising ValueError with CPython's message where Python
// raises it: surrounding whitespace, a sign, Unicode decimal digits and single
// underscores between digits, and at most 4300 digits (checked once the rest
// of the literal is valid, as CPython checks it).
export function pyIntFromString(text) {
  const invalid = () => new PyException('ValueError', `invalid literal for int() with base 10: ${Array.from(pyRepr(text)).slice(0, 200).join('')}`);
  const ascii = intAscii(text);
  let index = 0;
  while (index < ascii.length && isAsciiSpace(ascii[index])) index += 1;
  let sign = '';
  if (ascii[index] === '+' || ascii[index] === '-') {
    sign = ascii[index] === '-' ? '-' : '';
    index += 1;
  }
  if (ascii[index] === '_') throw invalid();
  let digits = '';
  let previous = '';
  for (; index < ascii.length && (isDigit(ascii.charCodeAt(index)) || ascii[index] === '_'); index += 1) {
    if (ascii[index] === '_' && previous === '_') throw invalid();
    if (ascii[index] !== '_') digits += ascii[index];
    previous = ascii[index];
  }
  if (previous === '_' || digits === '') throw invalid();
  while (index < ascii.length && isAsciiSpace(ascii[index])) index += 1;
  if (index !== ascii.length) throw invalid();
  if (digits.length > PY_INT_MAX_STR_DIGITS) throw intTooLong(digits.length);
  return intFromDigits(`${sign}${digits}`);
}

// --- base64 --------------------------------------------------------------------

const BASE64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const BASE64_VALUES = new Map(Array.from(BASE64_ALPHABET, (char, index) => [char.charCodeAt(0), index]));

// base64.b64decode(text, validate=True) for a str (binascii.a2b_base64 in
// strict mode, Python 3.11+). Raises ValueError for non-ASCII text and
// binascii.Error otherwise, with CPython's messages.
export function pyB64decodeValidate(text) {
  const codes = Array.from(text, (char) => char.codePointAt(0));
  if (codes.some((code) => code > 0x7f)) throw new PyException('ValueError', 'string argument should contain only ASCII characters');
  const error = (message) => new PyException('binascii.Error', message);
  if (codes.length > 0 && codes[0] === 0x3d) throw error('Leading padding not allowed');
  const out = [];
  let quadPos = 0;
  let leftChar = 0;
  let pads = 0;
  let paddingStarted = false;
  for (let i = 0; i < codes.length; i += 1) {
    const code = codes[i];
    if (code === 0x3d) {
      paddingStarted = true;
      if (quadPos === 0) throw error('Excess padding not allowed');
      if (quadPos >= 2) {
        pads += 1;
        if (quadPos + pads >= 4) {
          if (i + 1 < codes.length) throw error('Excess data after padding');
          return Buffer.from(out);
        }
      }
      continue;
    }
    const value = BASE64_VALUES.get(code);
    if (value === undefined) throw error('Only base64 data is allowed');
    if (paddingStarted) throw error('Discontinuous padding not allowed');
    pads = 0;
    if (quadPos === 0) {
      quadPos = 1;
      leftChar = value;
    } else if (quadPos === 1) {
      quadPos = 2;
      out.push(((leftChar << 2) | (value >> 4)) & 0xff);
      leftChar = value & 0x0f;
    } else if (quadPos === 2) {
      quadPos = 3;
      out.push(((leftChar << 4) | (value >> 2)) & 0xff);
      leftChar = value & 0x03;
    } else {
      quadPos = 0;
      out.push(((leftChar << 6) | value) & 0xff);
      leftChar = 0;
    }
  }
  if (quadPos === 1) {
    throw error(`Invalid base64-encoded string: number of data characters (${Math.floor(out.length / 3) * 4 + 1}) cannot be 1 more than a multiple of 4`);
  }
  if (quadPos !== 0) throw error('Incorrect padding');
  return Buffer.from(out);
}

// --- Files -----------------------------------------------------------------------

// str(pathlib.PurePosixPath(text)): repeated and trailing slashes and '.'
// components dropped, exactly two leading slashes kept, '' as '.'.
export function pyPath(text) {
  let root = '';
  let rest = text;
  if (text.startsWith('/')) {
    if (text.startsWith('//') && !text.startsWith('///')) {
      root = '//';
      rest = text.slice(2);
    } else {
      root = '/';
      rest = text.slice(1);
    }
  }
  const parts = rest.split('/').filter((part) => part !== '' && part !== '.');
  return `${root}${parts.join('/')}` || '.';
}

// strerror() text of the C library Python links against: glibc on the Linux
// runners, and macOS's libc where its text differs.
const STRERROR = Object.freeze({
  EPERM: 'Operation not permitted',
  ENOENT: 'No such file or directory',
  ESRCH: 'No such process',
  EINTR: 'Interrupted system call',
  EIO: 'Input/output error',
  ENXIO: { linux: 'No such device or address', darwin: 'Device not configured' },
  E2BIG: 'Argument list too long',
  ENOEXEC: 'Exec format error',
  EBADF: 'Bad file descriptor',
  EAGAIN: 'Resource temporarily unavailable',
  ENOMEM: 'Cannot allocate memory',
  EACCES: 'Permission denied',
  EFAULT: 'Bad address',
  EBUSY: { linux: 'Device or resource busy', darwin: 'Resource busy' },
  EEXIST: 'File exists',
  EXDEV: { linux: 'Invalid cross-device link', darwin: 'Cross-device link' },
  ENODEV: { linux: 'No such device', darwin: 'Operation not supported by device' },
  ENOTDIR: 'Not a directory',
  EISDIR: 'Is a directory',
  EINVAL: 'Invalid argument',
  ENFILE: 'Too many open files in system',
  EMFILE: 'Too many open files',
  ETXTBSY: 'Text file busy',
  EFBIG: 'File too large',
  ENOSPC: 'No space left on device',
  ESPIPE: 'Illegal seek',
  EROFS: 'Read-only file system',
  EMLINK: 'Too many links',
  EPIPE: 'Broken pipe',
  ENAMETOOLONG: 'File name too long',
  ENOTEMPTY: 'Directory not empty',
  ELOOP: 'Too many levels of symbolic links',
  EDQUOT: { linux: 'Disk quota exceeded', darwin: 'Disc quota exceeded' },
  EOPNOTSUPP: 'Operation not supported',
  ENOTSUP: 'Operation not supported',
});

// strerror(errno) for a Node system error `code` (ENOENT, ...).
export function pyStrerror(code, errno) {
  const entry = STRERROR[code];
  if (typeof entry === 'string') return entry;
  if (entry) return entry[process.platform] ?? entry.linux;
  return process.platform === 'darwin' ? `Unknown error: ${errno}` : `Unknown error ${errno}`;
}

const OS_ERROR_CLASSES = Object.freeze({
  EACCES: 'PermissionError',
  EISDIR: 'IsADirectoryError',
  ENOENT: 'FileNotFoundError',
  ENOTDIR: 'NotADirectoryError',
  EPERM: 'PermissionError',
});

// An OSError for a failed filesystem call on `path`, formatted as Python's
// "[Errno N] strerror: 'path'".
export function pyOSError(error, path) {
  if (!error || typeof error.code !== 'string' || typeof error.errno !== 'number') return error;
  const errno = Math.abs(error.errno);
  const exception = new PyException(OS_ERROR_CLASSES[error.code] ?? 'OSError', `[Errno ${errno}] ${pyStrerror(error.code, errno)}: ${pyRepr(path)}`);
  exception.errno = errno;
  exception.code = error.code;
  exception.filename = path;
  return exception;
}

// Path(text).read_bytes().
export function pyReadBytes(text) {
  const path = pyPath(text);
  try {
    return fs.readFileSync(path);
  } catch (error) {
    throw pyOSError(error, path);
  }
}

// Path(text).read_text(encoding="utf-8"): strict UTF-8 that keeps a BOM, with
// universal newlines.
export function pyReadText(text) {
  return pyUniversalNewlines(pyDecodeUtf8(pyReadBytes(text)));
}

// --- datetime ----------------------------------------------------------------------

// _strptime's regex for "%Y-%m-%dT%H:%M:%SZ" (re.IGNORECASE), \d as re has it.
const UTC_FORMAT = new RegExp(String.raw`^(\d{4})-(1[0-2]|0[1-9]|[1-9])-(3[0-1]|[1-2]\d|0[1-9]|[1-9]| [1-9])T(2[0-3]|[0-1]\d|\d):([0-5]\d|\d):(6[0-1]|[0-5]\d|\d)Z`
  .replaceAll(String.raw`\d`, PY_DECIMAL), 'iu');

function decimalText(text) {
  return Number(Array.from(text.trim(), (char) => decimalValue(char)).join(''));
}

function daysInMonth(year, month) {
  if (month === 2) return (year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0)) ? 29 : 28;
  return [4, 6, 9, 11].includes(month) ? 30 : 31;
}

// datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ"): the parsed fields, or null
// where Python raises ValueError (no match, unconverted data, or an invalid
// date or time such as second 60).
export function pyStrptimeUtc(value) {
  const match = UTC_FORMAT.exec(value);
  if (!match || match[0].length !== value.length) return null;
  const [year, month, day, hour, minute, second] = match.slice(1).map(decimalText);
  if (year < 1 || day > daysInMonth(year, month) || second > 59) return null;
  return { year, month, day, hour, minute, second };
}

// .strftime("%Y-%m-%dT%H:%M:%SZ") with glibc's unpadded %Y, as on the Linux
// runners (macOS pads a year below 1000 to four digits).
export function pyStrftimeUtc({ year, month, day, hour, minute, second }) {
  const two = (number) => String(number).padStart(2, '0');
  return `${year}-${two(month)}-${two(day)}T${two(hour)}:${two(minute)}:${two(second)}Z`;
}

// --- urllib.parse --------------------------------------------------------------

const ALWAYS_SAFE = new Set(Array.from('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-~'));

function isSurrogate(char) {
  const code = char.charCodeAt(0);
  return char.length === 1 && code >= 0xd800 && code <= 0xdfff;
}

// urllib.parse.quote(text, safe=safe): UTF-8 percent-encoding, upper-case hex.
export function pyQuote(text, safe = '/') {
  const safeSet = new Set([...ALWAYS_SAFE, ...Array.from(safe).filter((char) => char.codePointAt(0) < 0x80)]);
  const chars = Array.from(text);
  let out = '';
  for (let position = 0; position < chars.length; position += 1) {
    const char = chars[position];
    if (isSurrogate(char)) {
      let end = position + 1;
      while (end < chars.length && isSurrogate(chars[end])) end += 1;
      const where = end - position === 1
        ? `character '\\u${char.charCodeAt(0).toString(16)}' in position ${position}`
        : `characters in position ${position}-${end - 1}`;
      throw new PyException('UnicodeEncodeError', `'utf-8' codec can't encode ${where}: surrogates not allowed`);
    }
    if (safeSet.has(char)) out += char;
    else for (const byte of Buffer.from(char, 'utf8')) out += `%${byte.toString(16).toUpperCase().padStart(2, '0')}`;
  }
  return out;
}

// urllib.parse.quote_plus(text, safe=safe).
export function pyQuotePlus(text, safe = '') {
  if (!text.includes(' ')) return pyQuote(text, safe);
  return pyQuote(text, `${safe} `).replaceAll(' ', '+');
}

// urllib.parse.urlencode(query) over a dict or [key, value] pairs, with str()
// of each key and value.
export function pyUrlencode(query) {
  const pairs = Array.isArray(query) ? query : pyItems(query);
  return pairs.map(([key, value]) => `${pyQuotePlus(pyStr(key))}=${pyQuotePlus(pyStr(value))}`).join('&');
}

// --- json.loads over bytes -----------------------------------------------------

// json.detect_encoding(b): a BOM, else the NUL pattern of the first bytes.
export function pyJsonDetectEncoding(bytes) {
  const startsWith = (...prefix) => prefix.every((value, index) => bytes[index] === value);
  if (startsWith(0x00, 0x00, 0xfe, 0xff) || startsWith(0xff, 0xfe, 0x00, 0x00)) return 'utf-32';
  if (startsWith(0xfe, 0xff) || startsWith(0xff, 0xfe)) return 'utf-16';
  if (startsWith(0xef, 0xbb, 0xbf)) return 'utf-8-sig';
  if (bytes.length >= 4) {
    if (!bytes[0]) return bytes[1] ? 'utf-16-be' : 'utf-32-be';
    if (!bytes[1]) return bytes[2] || bytes[3] ? 'utf-16-le' : 'utf-32-le';
  } else if (bytes.length === 2) {
    if (!bytes[0]) return 'utf-16-be';
    if (!bytes[1]) return 'utf-16-le';
  }
  return 'utf-8';
}

function codecError(codec, bytes, start, end, reason) {
  const where = end - start === 1
    ? `byte 0x${bytes[start].toString(16).padStart(2, '0')} in position ${start}`
    : `bytes in position ${start}-${end - 1}`;
  return new PyException('UnicodeDecodeError', `'${codec}' codec can't decode ${where}: ${reason}`);
}

function fromCodeUnits(units) {
  let out = '';
  for (let index = 0; index < units.length; index += 8192) out += String.fromCharCode(...units.slice(index, index + 8192));
  return out;
}

// bytes.decode("utf-8", "surrogatepass"): a UTF-8 encoded surrogate
// (ED A0..BF 80..BF) decodes to that lone surrogate; anything else invalid
// raises the strict decoder's error.
function decodeUtf8Surrogatepass(bytes) {
  let out = '';
  let segment = 0;
  for (;;) {
    const range = utf8InvalidRange(bytes, segment);
    if (!range) break;
    const [start] = range;
    const encodedSurrogate = bytes[start] === 0xed && start + 2 < bytes.length
      && bytes[start + 1] >= 0xa0 && bytes[start + 1] <= 0xbf && bytes[start + 2] >= 0x80 && bytes[start + 2] <= 0xbf;
    if (!encodedSurrogate) throw utf8Error(bytes, ...range);
    out += UTF8.decode(bytes.subarray(segment, start));
    out += String.fromCharCode(((bytes[start] & 0x0f) << 12) | ((bytes[start + 1] & 0x3f) << 6) | (bytes[start + 2] & 0x3f));
    segment = start + 3;
  }
  return out + UTF8.decode(bytes.subarray(segment));
}

// bytes.decode("utf-16[-le|-be]", "surrogatepass") from `start` (after a
// BOM): lone surrogates pass, an odd trailing byte is truncated data.
function decodeUtf16Surrogatepass(bytes, start, littleEndian) {
  const codec = littleEndian ? 'utf-16-le' : 'utf-16-be';
  const units = [];
  let index = start;
  for (; index + 1 < bytes.length; index += 2) {
    units.push(littleEndian ? bytes[index] | (bytes[index + 1] << 8) : (bytes[index] << 8) | bytes[index + 1]);
  }
  if (index < bytes.length) throw codecError(codec, bytes, index, bytes.length, 'truncated data');
  return fromCodeUnits(units);
}

// bytes.decode("utf-32[-le|-be]", "surrogatepass") from `start`.
function decodeUtf32Surrogatepass(bytes, start, littleEndian) {
  const codec = littleEndian ? 'utf-32-le' : 'utf-32-be';
  const units = [];
  let index = start;
  for (; index + 3 < bytes.length; index += 4) {
    const [a, b, c, d] = littleEndian ? [bytes[index + 3], bytes[index + 2], bytes[index + 1], bytes[index]] : bytes.subarray(index, index + 4);
    const codePoint = ((a << 24) | (b << 16) | (c << 8) | d) >>> 0;
    if (codePoint > 0x10ffff) throw codecError(codec, bytes, index, index + 4, 'code point not in range(0x110000)');
    if (codePoint > 0xffff) units.push(0xd800 + ((codePoint - 0x10000) >> 10), 0xdc00 + ((codePoint - 0x10000) & 0x3ff));
    else units.push(codePoint);
  }
  if (index < bytes.length) throw codecError(codec, bytes, index, bytes.length, 'truncated data');
  return fromCodeUnits(units);
}

// s.decode(json.detect_encoding(s), "surrogatepass"), as json.loads decodes
// bytes. A BOM is consumed (a utf-8-sig error position counts from after it);
// UTF-16/32 error positions count from the start. Deviation: an encoded
// surrogate pair (\ud800 then \udc00 as two UTF-8 or UTF-32 sequences) is two
// code points in Python, but joins into one in the returned JS string.
export function pyDecodeJsonBytes(bytes) {
  const encoding = pyJsonDetectEncoding(bytes);
  if (encoding === 'utf-8') return decodeUtf8Surrogatepass(bytes);
  if (encoding === 'utf-8-sig') return decodeUtf8Surrogatepass(bytes.subarray(3));
  if (encoding === 'utf-16') return decodeUtf16Surrogatepass(bytes, 2, bytes[0] === 0xff);
  if (encoding === 'utf-32') return decodeUtf32Surrogatepass(bytes, 4, bytes[0] === 0xff);
  const littleEndian = encoding.endsWith('-le');
  return encoding.startsWith('utf-16') ? decodeUtf16Surrogatepass(bytes, 0, littleEndian) : decodeUtf32Surrogatepass(bytes, 0, littleEndian);
}

// json.loads(bytes, ...): pyJsonLoads over the decoded text. Unlike a str
// argument, decoded text that starts with U+FEFF is not reported as a BOM: the
// scanner finds no value there.
export function pyJsonLoadsBytes(bytes, options = {}) {
  const text = pyDecodeJsonBytes(bytes);
  if (text.charCodeAt(0) === 0xfeff) throw new JSONDecodeError('Expecting value', [], 0);
  return pyJsonLoads(text, options);
}

// bytes.strip(): ASCII whitespace (space, \t, \n, \v, \f, \r) at both ends.
export function pyBytesStrip(bytes) {
  const isSpace = (value) => value === 0x20 || (value >= 0x09 && value <= 0x0d);
  let start = 0;
  let end = bytes.length;
  while (start < end && isSpace(bytes[start])) start += 1;
  while (end > start && isSpace(bytes[end - 1])) end -= 1;
  return bytes.subarray(start, end);
}

// --- pathlib predicates and listings ---------------------------------------------

// os.fsdecode(bytes) on a UTF-8 system: undecodable bytes become the
// surrogate escapes U+DC80..U+DCFF.
export function pyFsDecode(bytes) {
  let out = '';
  let segment = 0;
  for (;;) {
    const range = utf8InvalidRange(bytes, segment);
    if (!range) break;
    const [start, end] = range;
    out += UTF8.decode(bytes.subarray(segment, start));
    for (let index = start; index < end; index += 1) out += String.fromCharCode(0xdc00 + bytes[index]);
    segment = end;
  }
  return out + UTF8.decode(bytes.subarray(segment));
}

// str(Path(directory) / name) for a normalized `directory` and a relative name.
export function pyJoinPath(directory, name) {
  return directory.endsWith('/') ? `${directory}${name}` : `${directory}/${name}`;
}

// pathlib (3.12) ignores these errors in is_dir/is_file/is_symlink.
const IGNORED_STAT_ERRORS = new Set(['ENOENT', 'ENOTDIR', 'EBADF', 'ELOOP']);

function statPredicate(text, stat, predicate) {
  let stats;
  try {
    stats = stat(text);
  } catch (error) {
    if (IGNORED_STAT_ERRORS.has(error?.code) || error?.code === 'ERR_INVALID_ARG_VALUE') return false;
    throw pyOSError(error, text);
  }
  return predicate(stats);
}

// Path(text).is_dir(), following symlinks.
export function pyIsDir(text) {
  return statPredicate(text, fs.statSync, (stats) => stats.isDirectory());
}

// Path(text).is_file(), following symlinks.
export function pyIsFile(text) {
  return statPredicate(text, fs.statSync, (stats) => stats.isFile());
}

// Path(text).is_symlink().
export function pyIsSymlink(text) {
  return statPredicate(text, fs.lstatSync, (stats) => stats.isSymbolicLink());
}

// os.listdir(text), the names Path(text).iterdir() yields: readdir order
// (unsorted, unlike fs.readdirSync) without '.' and '..', fsdecoded.
export function pyListdir(text) {
  let directory;
  try {
    directory = fs.opendirSync(text, { encoding: 'buffer' });
  } catch (error) {
    throw pyOSError(error, text);
  }
  try {
    const names = [];
    for (let entry = directory.readSync(); entry !== null; entry = directory.readSync()) names.push(pyFsDecode(entry.name));
    return names;
  } catch (error) {
    throw pyOSError(error, text);
  } finally {
    directory.closeSync();
  }
}

// Path(text).write_text(content, encoding="utf-8") for well-formed content.
export function pyWriteText(text, content) {
  try {
    fs.writeFileSync(text, content, 'utf8');
  } catch (error) {
    throw pyOSError(error, text);
  }
}
