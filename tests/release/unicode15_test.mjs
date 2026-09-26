// Python 3.12 runs on Unicode 15.0.0 data, Node 24's regex \p{...} classes on
// Unicode 16 or 17. These tests prove the Unicode 15.0 classes in
// scripts/release/json.mjs (re's \w and \d, str.isprintable, int()'s digits)
// against Python 3.12 over every code point (with \s and the [a-z0-9] part of
// str.lower(), which need no adjustment), and that
// scripts/release/unicode15.mjs is what tests/release/unicode15_generate.mjs
// derives. They need a python3.12 or python3 with unicodedata 15.0.0 (the
// ubuntu-24.04 runners' python3) and skip without one. The last tests pin
// the fail-open cases the skew caused, with Python 3.12's outputs.

import assert from 'node:assert/strict';
import process from 'node:process';
import { test } from 'node:test';

import { parseCommandLine } from '../../scripts/release/cli.mjs';
import {
  PY_DECIMAL, PY_WHITESPACE, PY_WORD, pyIntFromString, pyRepr, pyStrptimeUtc,
} from '../../scripts/release/json.mjs';
import { sanitizeDiagnosticText } from '../../scripts/release/qualification.mjs';
import { ASSIGNED_AFTER_UNICODE_15, GENERATED_UNICODE_VERSION } from '../../scripts/release/unicode15.mjs';
import { findPython312, pythonUnassigned, runPython, toRanges } from './unicode15_generate.mjs';

const PYTHON = findPython312();
const skip = PYTHON === null && 'no python3.12 or python3 with unicodedata 15.0.0';

const expand = (ranges) => ranges.flatMap(([start, end]) => Array.from({ length: end - start + 1 }, (_, index) => start + index));

test('unicode15.mjs lists exactly the code points Python 3.12 leaves unassigned and Node assigns', { skip }, () => {
  const unassigned = new Set(pythonUnassigned(PYTHON));
  const committed = expand(ASSIGNED_AFTER_UNICODE_15);
  assert.deepEqual(committed.filter((code) => !unassigned.has(code)), [], 'listed code points Python 3.12 assigns');
  const listed = new Set(committed);
  const cn = /\p{Cn}/u;
  const missing = [...unassigned].filter((code) => !cn.test(String.fromCodePoint(code)) && !listed.has(code));
  assert.deepEqual(missing, [], `Node ${process.version} (Unicode ${process.versions.unicode}) assigns code points unicode15.mjs lacks; regenerate it`);
  if (process.versions.unicode === GENERATED_UNICODE_VERSION) {
    assert.equal(committed.length, unassigned.size - [...unassigned].filter((code) => cn.test(String.fromCodePoint(code))).length);
  }
});

const PYTHON_CLASSES = `
import json, re, sys
word, digit, space, printable, values, lower = [], [], [], [], {}, {}
w, d, s = re.compile(r"\\w"), re.compile(r"\\d"), re.compile(r"\\s")
for code in range(0x110000):
    char = chr(code)
    if w.fullmatch(char): word.append(code)
    if d.fullmatch(char):
        digit.append(code)
        values[code] = int(char)
    if s.fullmatch(char): space.append(code)
    if char.isprintable(): printable.append(code)
    folded = re.sub(r"[^a-z0-9]+", " ", char.lower()) if 0xD800 > code or code > 0xDFFF else " "
    if folded != " ": lower[code] = folded
print(json.dumps({"word": word, "digit": digit, "space": space, "printable": printable, "values": values, "lower": lower}))
`;

// The code points (as ranges) whose one-character string `accept` accepts.
function nodeRanges(accept) {
  const codes = [];
  for (let code = 0; code < 0x110000; code += 1) if (accept(String.fromCodePoint(code))) codes.push(code);
  return toRanges(codes);
}

test('re\'s \\w and \\d, str.isprintable and int()\'s digits match Python 3.12 over every code point', { skip }, () => {
  const python = JSON.parse(runPython(PYTHON, PYTHON_CLASSES));
  const word = new RegExp(`^${PY_WORD}$`, 'u');
  const decimal = new RegExp(`^${PY_DECIMAL}$`, 'u');
  assert.deepEqual(nodeRanges((char) => word.test(char)), toRanges(python.word), '\\w');
  assert.deepEqual(nodeRanges((char) => decimal.test(char)), toRanges(python.digit), '\\d');
  // \s needs no Unicode 15.0 subtraction: no space was assigned since.
  const space = new RegExp(`^[${PY_WHITESPACE}]$`, 'u');
  assert.deepEqual(nodeRanges((char) => space.test(char)), toRanges(python.space), '\\s');
  // repr() leaves exactly the printable characters unescaped.
  const printable = (char) => char === "'" || char === '\\' || pyRepr(char) === `'${char}'`;
  assert.deepEqual(nodeRanges(printable), toRanges(python.printable), 'isprintable');
  for (const [code, value] of Object.entries(python.values)) {
    assert.equal(pyIntFromString(String.fromCodePoint(Number(code))), value, code);
  }
  // normalize_transcript keeps only [a-z0-9] of str.lower(), where Node's
  // toLowerCase() agrees with Python 3.12: the case pairs added since Unicode
  // 15.0 lower to letters outside ASCII either way.
  const lower = {};
  for (let code = 0; code < 0x110000; code += 1) {
    const folded = code >= 0xd800 && code <= 0xdfff ? ' ' : String.fromCodePoint(code).toLowerCase().replace(/[^a-z0-9]+/gu, ' ');
    if (folded !== ' ') lower[code] = folded;
  }
  assert.deepEqual(lower, python.lower);
});

test('credential names after a character assigned since Unicode 15.0 are redacted, as in Python 3.12', () => {
  // Python 3.12 sees each leading character as unassigned, so \b holds before
  // the credential name. Expected outputs are Python 3.12.3's.
  assert.equal(sanitizeDiagnosticText('\u{13556}password=hunter2'), '\u{13556}<redacted-credential>');
  assert.equal(sanitizeDiagnosticText('log: \u{1E5D0}Bearer ghp_abcdef'), 'log: \u{1E5D0}<redacted-credential>');
  assert.equal(sanitizeDiagnosticText('\u{2ECBE}token: s3cr3t'), '\u{2ECBE}<redacted-credential>');
  // A letter Unicode 15.0 already had still joins the word.
  assert.equal(sanitizeDiagnosticText('\u{e9}password=hunter2'), '\u{e9}password=hunter2');
});

test('a digit assigned since Unicode 15.0 is no digit to the CLI, int(), strptime or repr()', () => {
  const commands = { run: { options: [{ flag: '--release-rebuild', required: true, type: 'int' }] } };
  const parse = (argv) => {
    try {
      return parseCommandLine(argv, { prog: 'p', commands });
    } catch (error) {
      return { status: error.status, stderr: error.stderr };
    }
  };
  assert.deepEqual(parse(['run', '--release-rebuild', '\u{10d40}']), {
    status: 2,
    stderr: "usage: p run [-h] --release-rebuild RELEASE_REBUILD\np run: error: argument --release-rebuild: invalid int value: '\\U00010d40'\n",
  });
  assert.deepEqual(parse(['run', '--release-rebuild', '\u{1e950}']), { command: 'run', args: { releaseRebuild: 0 } });
  assert.equal(pyStrptimeUtc('\u{10d41}024-01-02T03:04:05Z'), null);
  assert.deepEqual(pyStrptimeUtc('\u{1e951}024-01-02T03:04:05Z'), { year: 1024, month: 1, day: 2, hour: 3, minute: 4, second: 5 });
  assert.equal(pyRepr('\u{10d40}\u{e9}'), "'\\U00010d40\u{e9}'");
});
