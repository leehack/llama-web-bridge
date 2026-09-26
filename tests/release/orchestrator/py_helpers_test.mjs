// Tests of the Python-compatible helpers the orchestrator port added to
// scripts/release/json.mjs (pyStrptimeUtcOrRaise, the pyJsonDumps separators
// option) and scripts/release/python_compat.mjs (pyFullmatch, pyStrSplit,
// pyCompareIntTuples, PyUtcDatetime). Every expected value was produced by
// CPython 3.12.11, the runners' Python.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { pyJsonDumps, pyStrptimeUtcOrRaise } from '../../../scripts/release/json.mjs';
import {
  PyUtcDatetime, pyCompareIntTuples, pyFullmatch, pyStrSplit,
} from '../../../scripts/release/python_compat.mjs';

function pyError(fn) {
  try {
    fn();
  } catch (error) {
    return `${error.pyType ?? error.name}: ${error.message}`;
  }
  return null;
}

test('strptime raises the ValueError CPython 3.12 raises', () => {
  const cases = [
    ['2026-08-19T12:34:56Z', 63922739696],
    ['2024-02-29T23:59:59Z', 63844847999],
    ['2026-13-01T00:00:00Z', "ValueError: time data '2026-13-01T00:00:00Z' does not match format '%Y-%m-%dT%H:%M:%SZ'"],
    ['2026-02-30T00:00:00Z', 'ValueError: day is out of range for month'],
    ['2026-01-01T00:00:60Z', 'ValueError: second must be in 0..59'],
    ['0000-01-01T00:00:00Z', 'ValueError: year 0 is out of range'],
    ['2026-08-19T12:34:56Zx', 'ValueError: unconverted data remains: x'],
    ['2026-8-9T1:2:3Z', 63921834123],
    ['0001-01-01T00:00:00Z', 0],
    ['9999-12-31T23:59:59Z', 315537897599],
    ['2026-08-19t12:34:56z', 63922739696],
  ];
  for (const [text, expected] of cases) {
    if (typeof expected === 'number') {
      assert.equal(PyUtcDatetime.strptime(text).seconds, expected, text);
      assert.ok(pyStrptimeUtcOrRaise(text), text);
    } else {
      assert.equal(pyError(() => pyStrptimeUtcOrRaise(text)), expected, text);
      assert.equal(pyError(() => PyUtcDatetime.strptime(text)), expected, text);
    }
  }
});

test('datetime arithmetic, strftime and overflow match CPython', () => {
  const start = PyUtcDatetime.strptime('2026-08-19T12:34:56Z');
  const expected = [
    [0, '2026-08-19T12:34:56Z'],
    [1, '2026-08-19T12:34:57Z'],
    [2591999, '2026-09-18T12:34:55Z'],
    [-86400 * 365, '2025-08-19T12:34:56Z'],
    [86400 * 10000, '2054-01-04T12:34:56Z'],
  ];
  for (const [delta, text] of expected) assert.equal(start.addSeconds(delta).strftime(), text, String(delta));
  assert.equal(PyUtcDatetime.strptime('2024-02-29T00:00:00Z').strftime(), '2024-02-29T00:00:00Z');
  // glibc's unpadded %Y, as on the Linux runners.
  assert.equal(PyUtcDatetime.strptime('0999-01-01T00:00:00Z').strftime(), '999-01-01T00:00:00Z');
  assert.equal(pyError(() => PyUtcDatetime.strptime('9999-12-31T23:59:59Z').addSeconds(1)), 'OverflowError: date value out of range');
  assert.equal(pyError(() => PyUtcDatetime.strptime('0001-01-01T00:00:00Z').addSeconds(-1)), 'OverflowError: date value out of range');
  // Every day of four centuries round-trips through the civil calendar.
  let day = PyUtcDatetime.strptime('1900-01-01T00:00:00Z');
  for (let index = 0; index < 146097; index += 1) {
    const text = day.strftime();
    assert.equal(PyUtcDatetime.strptime(text).seconds, day.seconds, text);
    day = day.addSeconds(86400);
  }
  assert.equal(day.strftime(), '2300-01-01T00:00:00Z');
});

test('json.dumps separators', () => {
  assert.equal(
    pyJsonDumps({ b: [1, { a: 2 }], a: 'x' }, { sortKeys: true, separators: [',', ':'] }),
    '{"a":"x","b":[1,{"a":2}]}',
  );
  // The defaults are unchanged.
  assert.equal(pyJsonDumps({ b: [1, 2], a: 'x' }, { sortKeys: true }), '{"a": "x", "b": [1, 2]}');
  assert.equal(pyJsonDumps({ a: [1] }, { indent: 2 }), '{\n  "a": [\n    1\n  ]\n}');
});

test('str.split() splits on str.isspace() runs', () => {
  const cases = [
    ['  a b　c\x1c ', ['a', 'b', 'c']],
    ['', []],
    [' \t\n', []],
    ['bridge-candidate x y', ['bridge-candidate', 'x', 'y']],
    ['a\x85b c', ['a', 'b', 'c']],
  ];
  for (const [text, fields] of cases) assert.deepEqual(pyStrSplit(text), fields, JSON.stringify(text));
});

test('re.fullmatch rejects a non-str and tuple ordering compares ints', () => {
  assert.equal(pyFullmatch(/^a+$/u, 'aaa')[0], 'aaa');
  assert.equal(pyFullmatch(/^a+$/u, 'aab'), null);
  assert.equal(pyError(() => pyFullmatch(/^a$/u, 1)), "TypeError: expected string or bytes-like object, got 'int'");
  assert.equal(pyError(() => pyFullmatch(/^a$/u, null)), "TypeError: expected string or bytes-like object, got 'NoneType'");
  assert.ok(pyCompareIntTuples([0, 2, 0, 1], [0, 2, 0, 0]) > 0);
  assert.ok(pyCompareIntTuples([0, 2], [0, 2, 0]) < 0);
  assert.equal(pyCompareIntTuples([1n, 2], [1, 2n]), 0);
  assert.ok(pyCompareIntTuples([2n ** 70n], [Number.MAX_SAFE_INTEGER]) > 0);
});
