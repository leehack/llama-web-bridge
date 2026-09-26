// Tests of scripts/release/json.mjs, the Python-compatible JSON and text
// primitives the release tooling ports share. Expected values are CPython
// 3.12's output for the same input.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

import {
  PY_JSON_MAX_DEPTH,
  PY_NAN,
  PyException,
  PyFloat,
  isDict,
  isPyException,
  isPyFloat,
  isPyInt,
  pyB64decodeValidate,
  pyCanonicalJson,
  pyDecodeUtf8,
  pyDict,
  pyEquals,
  pyGet,
  pyHookRepr,
  pyIntFromString,
  pyItems,
  pyJsonDumps,
  pyJsonLoads,
  pyKeys,
  pyPath,
  pyQuote,
  pyQuotePlus,
  pyReadText,
  pyRepr,
  pySplitlines,
  pyStrftimeUtc,
  pyStrip,
  pyStrptimeUtc,
  pyUrlencode,
} from '../../scripts/release/json.mjs';

function rejectDuplicates(pairs, frame) {
  const seen = new Set();
  for (const [key] of pairs) {
    if (seen.has(key)) throw new PyException('ValueError', `duplicate JSON key ${pyHookRepr(frame, key)}`);
    seen.add(key);
  }
  return pyDict(pairs);
}

function pyError(fn) {
  try {
    fn();
  } catch (error) {
    if (error instanceof PyException) return `${error.pyType}: ${error.message}`;
    throw error;
  }
  assert.fail('expected a Python exception');
}

test('loads reports JSONDecodeError positions like CPython', () => {
  const cases = [
    ['[1,]', 'Expecting value: line 1 column 4 (char 3)'],
    ['{"a":1,}', 'Expecting property name enclosed in double quotes: line 1 column 8 (char 7)'],
    ['\ufeff{}', 'Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)'],
    ['', 'Expecting value: line 1 column 1 (char 0)'],
    ['[[[]', "Expecting ',' delimiter: line 1 column 5 (char 4)"],
    ['nul', 'Expecting value: line 1 column 1 (char 0)'],
    ['{"a" 1}', "Expecting ':' delimiter: line 1 column 6 (char 5)"],
    ['"a\nb"', 'Invalid control character at: line 1 column 3 (char 2)'],
    ['"\u{1F600}" x', 'Extra data: line 1 column 5 (char 4)'],
  ];
  for (const [text, message] of cases) assert.equal(pyError(() => pyJsonLoads(text)), `JSONDecodeError: ${message}`, text);
  assert.ok(isPyException(new PyException('JSONDecodeError', 'x'), 'ValueError'));
});

test('loads keeps Python int, float and constant identities', () => {
  const value = pyJsonLoads('[1.0, 1, true, 1e400, NaN, 12345678901234567890, "\\ud800"]');
  assert.ok(isPyFloat(value[0]) && value[0].value === 1);
  assert.ok(isPyInt(value[1]));
  assert.equal(value[2], true);
  assert.ok(isPyFloat(value[3]) && value[3].value === Infinity);
  assert.equal(value[4], PY_NAN);
  assert.equal(value[5], 12345678901234567890n);
  assert.ok(isPyInt(value[5]));
  assert.equal(value[6], '\ud800');
  assert.equal(pyError(() => pyJsonLoads('1'.repeat(4301))),
    'ValueError: Exceeds the limit (4300 digits) for integer string conversion: value has 4301 digits; use sys.set_int_max_str_digits() to increase the limit');
});

test('loads rejects constants through parseConstant and duplicates through objectPairsHook', () => {
  const rejectConstant = (name) => { throw new PyException('ValueError', `invalid JSON numeric constant: ${name}`); };
  assert.equal(pyError(() => pyJsonLoads('{"a": NaN}', { parseConstant: rejectConstant })), 'ValueError: invalid JSON numeric constant: NaN');
  assert.equal(pyError(() => pyJsonLoads('{"a":1,"a":2}', { objectPairsHook: rejectDuplicates })), "ValueError: duplicate JSON key 'a'");
  const parsed = pyJsonLoads('{"b": 1, "a": {"x": []}}', { objectPairsHook: rejectDuplicates });
  assert.ok(isDict(parsed));
  assert.deepEqual(pyKeys(parsed), ['b', 'a']);
});

test('loads keeps dict insertion order for integer-like and prototype keys', () => {
  const parsed = pyJsonLoads('{"b": 1, "10": 2, "2": 3, "__proto__": 4}');
  assert.deepEqual(pyItems(parsed), [['b', 1], ['10', 2], ['2', 3], ['__proto__', 4]]);
  assert.equal(pyGet(parsed, '__proto__'), 4);
  assert.equal(pyGet(parsed, 'missing'), null);
  assert.equal(pyError(() => pyGet([], 'x')), "AttributeError: 'list' object has no attribute 'get'");
});

test('loads follows the CPython 3.12 nesting budget', () => {
  assert.equal(PY_JSON_MAX_DEPTH, 9997);
  const nested = (depth) => `${'['.repeat(depth)}${']'.repeat(depth)}`;
  assert.doesNotThrow(() => pyJsonLoads(nested(PY_JSON_MAX_DEPTH)));
  assert.equal(pyError(() => pyJsonLoads(nested(PY_JSON_MAX_DEPTH + 1))),
    'RecursionError: maximum recursion depth exceeded while decoding a JSON array from a unicode string');
});

test('dumps matches json.dumps with indent, sort_keys and ensure_ascii', () => {
  const value = pyDict([['b', [1, new PyFloat(1), true, null, '\u00e9\u2028']], ['a', pyDict([])], ['c', []]]);
  assert.equal(pyCanonicalJson(value),
    '{\n  "a": {},\n  "b": [\n    1,\n    1.0,\n    true,\n    null,\n    "\\u00e9\\u2028"\n  ],\n  "c": []\n}\n');
  assert.equal(pyJsonDumps(pyDict([['b', new PyFloat(1e16)], ['a', new PyFloat(-0)], ['c', new PyFloat(1.5e-7)]]), { sortKeys: true }),
    '{"a": -0.0, "b": 1e+16, "c": 1.5e-07}');
  assert.equal(pyJsonDumps([PY_NAN]), '[NaN]');
  assert.equal(pyError(() => pyJsonDumps([PY_NAN], { allowNan: false })), 'ValueError: Out of range float values are not JSON compliant: nan');
});

test('dumps round-trips loads output byte for byte', () => {
  const text = '{\n  "a": [\n    1,\n    2.5,\n    12345678901234567890\n  ],\n  "z": "\\u00e9"\n}\n';
  assert.equal(pyCanonicalJson(pyJsonLoads(text)), text);
});

test('repr matches Python for str, float, int and containers', () => {
  const value = pyDict([['a', [1, new PyFloat(1), null, true, "it's", 'q"\'', '\x00\u2028\u00e9']]]);
  assert.equal(pyRepr(value), `{'a': [1, 1.0, None, True, "it's", 'q"\\'', '\\x00\\u2028\u00e9']}`);
  assert.equal(pyRepr(new PyFloat(1e16)), '1e+16');
  assert.equal(pyRepr(new PyFloat(1e-5)), '1e-05');
  assert.equal(pyRepr(new PyFloat(123456789012345678)), '1.2345678901234568e+17');
  assert.equal(pyRepr(2n ** 70n), '1180591620717411303424');
});

test('equality follows Python numeric and NaN identity rules', () => {
  assert.ok(pyEquals(1, new PyFloat(1)));
  assert.ok(pyEquals(true, 1));
  assert.ok(pyEquals([PY_NAN], [PY_NAN]));
  assert.ok(!pyEquals(PY_NAN, PY_NAN));
  assert.ok(pyEquals(pyDict([['a', 1], ['b', 2]]), pyDict([['b', 2], ['a', new PyFloat(1)]])));
  assert.ok(!pyEquals([1], [1, 1]));
});

test('equality, repr and dumps handle any nesting json.loads accepts', () => {
  const depth = PY_JSON_MAX_DEPTH;
  const text = `${'['.repeat(depth)}1${']'.repeat(depth)}`;
  const value = pyJsonLoads(text);
  assert.equal(pyEquals(value, pyJsonLoads(text)), true);
  assert.equal(pyEquals(value, pyJsonLoads(text.replace('1', '1.0'))), true);
  assert.equal(pyEquals(value, pyJsonLoads(text.replace('1', '2'))), false);
  assert.equal(pyEquals(value, pyJsonLoads(text.replace('[1]', '[1, 1]'))), false);
  assert.equal(pyRepr(value), text);
  assert.equal(pyJsonDumps(value), text);
  const objects = pyJsonLoads(`${'{"a": ['.repeat(4000)}NaN${']}'.repeat(4000)}`);
  assert.equal(pyEquals(objects, objects), true);
  assert.equal(pyJsonDumps(objects), `${'{"a": ['.repeat(4000)}NaN${']}'.repeat(4000)}`);
  assert.equal(pyRepr(objects), `${"{'a': [".repeat(4000)}nan${']}'.repeat(4000)}`);
  assert.equal(pyJsonDumps(objects, { indent: 1 }).split('\n').length, 16001);
  // The indented encoder's recursion budget still applies, in encoding order.
  assert.throws(() => pyJsonDumps(value, { indent: 2, maxNesting: 100 }), (error) => error.tracebackLine === 'RecursionError: maximum recursion depth exceeded');
  assert.throws(() => pyJsonDumps([{ 1: 2 }, [[[]]]], { maxNesting: 3 }), (error) => error.tracebackLine === 'RecursionError: maximum recursion depth exceeded');
  assert.throws(() => pyJsonDumps([[[new Set()]], [[[]]]], { maxNesting: 3 }), (error) => error.tracebackLine === 'TypeError: Object of type Set is not JSON serializable');
});

test('splitlines and strip use Python line and space definitions', () => {
  assert.deepEqual(pySplitlines('x\ny\r\nz\x1c\u2028w'), ['x', 'y', 'z', '', 'w']);
  assert.deepEqual(pySplitlines('a\n'), ['a']);
  assert.deepEqual(pySplitlines(''), []);
  assert.equal(pyStrip('  \x1c a\u3000'), 'a');
});

test('utf-8 decoding raises UnicodeDecodeError with CPython messages', () => {
  assert.equal(pyDecodeUtf8(Buffer.from('\ufeffok', 'utf8')), '\ufeffok');
  assert.equal(pyError(() => pyDecodeUtf8(Buffer.from([0x61, 0x62, 0xff]))),
    "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 2: invalid start byte");
  assert.equal(pyError(() => pyDecodeUtf8(Buffer.from([0x61, 0x62, 0xe2, 0x82]))),
    "UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 2-3: unexpected end of data");
  assert.equal(pyError(() => pyDecodeUtf8(Buffer.from([0xe2, 0x28, 0xa1]))),
    "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe2 in position 0: invalid continuation byte");
});

test('int() parsing accepts Python literals only', () => {
  assert.equal(pyIntFromString(' \u2003 1_0\n'), 10);
  assert.equal(pyIntFromString('\u0665'), 5);
  assert.equal(pyIntFromString('-0'), 0);
  assert.equal(pyIntFromString('99999999999999999999'), 99999999999999999999n);
  assert.equal(pyIntFromString(' -\u0663 '), -3);
  assert.equal(pyIntFromString('1'.repeat(4300)), BigInt('1'.repeat(4300)));
  // ValueError text from CPython 3.12.
  for (const text of ['0\x1c', '1__0', '', '1.0', '+', '0x10', '_1', '1_', '\u{10d40}']) {
    assert.equal(pyError(() => pyIntFromString(text)), `ValueError: invalid literal for int() with base 10: ${pyRepr(text)}`, text);
  }
  const tooLong = 'ValueError: Exceeds the limit (4300 digits) for integer string conversion: value has 4301 digits; use sys.set_int_max_str_digits() to increase the limit';
  assert.equal(pyError(() => pyIntFromString('1'.repeat(4301))), tooLong);
  assert.equal(pyError(() => pyIntFromString(` +${'1_'.repeat(4300)}1\n`)), tooLong);
  // An otherwise invalid literal is invalid, whatever its length; the repr is
  // cut at 200 characters.
  for (const text of [`${'1'.repeat(4301)}x`, `${'1'.repeat(4301)}\xe9`, `${'1'.repeat(4301)}__`, 'x'.repeat(300)]) {
    assert.equal(pyError(() => pyIntFromString(text)), `ValueError: invalid literal for int() with base 10: ${pyRepr(text).slice(0, 200)}`);
  }
});

test('strict base64 decoding matches binascii strict mode', () => {
  assert.equal(Buffer.from(pyB64decodeValidate('Zm9v')).toString(), 'foo');
  const cases = [
    ['Zm9', 'binascii.Error: Incorrect padding'],
    ['Zm9v=', 'binascii.Error: Excess padding not allowed'],
    ['Zm 9v', 'binascii.Error: Only base64 data is allowed'],
    ['=Zm9v', 'binascii.Error: Leading padding not allowed'],
    ['Z', 'binascii.Error: Invalid base64-encoded string: number of data characters (1) cannot be 1 more than a multiple of 4'],
    ['Z\u00e9', 'ValueError: string argument should contain only ASCII characters'],
  ];
  for (const [text, message] of cases) assert.equal(pyError(() => pyB64decodeValidate(text)), message, text);
  assert.ok(isPyException(new PyException('binascii.Error', 'x'), 'ValueError'));
});

test('UTC timestamps round-trip through strptime and Linux strftime', () => {
  const roundTrip = (text) => {
    const fields = pyStrptimeUtc(text);
    return fields && pyStrftimeUtc(fields);
  };
  assert.equal(roundTrip('2026-08-20T22:15:59Z'), '2026-08-20T22:15:59Z');
  assert.equal(roundTrip('2026-8-20T22:15:59Z'), '2026-08-20T22:15:59Z');
  assert.equal(roundTrip('2024-02-29T00:00:00Z'), '2024-02-29T00:00:00Z');
  assert.equal(roundTrip('0999-01-01T00:00:00Z'), '999-01-01T00:00:00Z');
  for (const text of ['2026-08-20T22:15:60Z', '2026-02-29T00:00:00Z', '2026-08-20 22:15:59Z', '2026-08-20T22:15:59Z\n', '']) {
    assert.equal(pyStrptimeUtc(text), null, text);
  }
});

test('quote, quote_plus and urlencode match urllib.parse', () => {
  assert.equal(pyQuote('a b/\u00e9~\x00'), 'a%20b/%C3%A9~%00');
  assert.equal(pyQuotePlus('a b/c&'), 'a+b%2Fc%26');
  assert.equal(pyUrlencode(pyDict([['q', 'a b'], ['x', '\u00e9&']])), 'q=a+b&x=%C3%A9%26');
  assert.equal(pyError(() => pyQuote('a\ud800\ud801b')),
    "UnicodeEncodeError: 'utf-8' codec can't encode characters in position 1-2: surrogates not allowed");
});

test('paths and file reads report OSError like pathlib', () => {
  assert.equal(pyPath(''), '.');
  assert.equal(pyPath('a//b/./c/'), 'a/b/c');
  assert.equal(pyPath('//x/y'), '//x/y');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'release-json-'));
  try {
    const missing = path.join(dir, 'missing.json');
    assert.equal(pyError(() => pyReadText(missing)), `FileNotFoundError: [Errno 2] No such file or directory: '${missing}'`);
    assert.equal(pyError(() => pyReadText(dir)), `IsADirectoryError: [Errno 21] Is a directory: '${dir}'`);
    const file = path.join(dir, 'text.txt');
    fs.writeFileSync(file, 'a\r\nb\rc');
    assert.equal(pyReadText(file), 'a\nb\nc');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
