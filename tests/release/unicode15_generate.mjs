#!/usr/bin/env node
// Generates scripts/release/unicode15.mjs: the code points Python 3.12's
// unicodedata (Unicode 15.0.0) leaves unassigned (general category Cn) but the
// running Node's regex property data assigns. Run it on the newest Node the
// tooling supports, so the list covers every Unicode version an older Node
// could carry:
//
//   node tests/release/unicode15_generate.mjs [python3.12] > scripts/release/unicode15.mjs
//
// The Python argument defaults to python3.12, then python3; it must report
// unicodedata.unidata_version 15.0.0. tests/release/unicode15_test.mjs runs
// the same generation and compares it with the committed module.

import { spawnSync } from 'node:child_process';
import process from 'node:process';

export const PYTHON_UNIDATA_VERSION = '15.0.0';

const PYTHON_CN = `
import sys, unicodedata
print(unicodedata.unidata_version)
print(" ".join(str(c) for c in range(0x110000) if unicodedata.category(chr(c)) == "Cn"))
`;

// The first of `candidates` whose unicodedata is Unicode 15.0.0, or null.
export function findPython312(candidates = ['python3.12', 'python3']) {
  for (const python of candidates) {
    const result = spawnSync(python, ['-c', 'import unicodedata; print(unicodedata.unidata_version)'], { encoding: 'utf8' });
    if (!result.error && result.status === 0 && result.stdout.trim() === PYTHON_UNIDATA_VERSION) return python;
  }
  return null;
}

// Run `source` under `python` with `args`, returning its stdout.
export function runPython(python, source, args = []) {
  const result = spawnSync(python, ['-c', source, ...args], { encoding: 'utf8', maxBuffer: 1 << 28 });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${python} failed: ${result.stderr}`);
  return result.stdout;
}

// Sorted code points as inclusive [start, end] ranges.
export function toRanges(codePoints) {
  const ranges = [];
  for (const code of codePoints) {
    const last = ranges.at(-1);
    if (last && last[1] === code - 1) last[1] = code;
    else ranges.push([code, code]);
  }
  return ranges;
}

// The code points Python's unicodedata calls Cn.
export function pythonUnassigned(python) {
  const [version, list] = runPython(python, PYTHON_CN).split('\n');
  if (version !== PYTHON_UNIDATA_VERSION) throw new Error(`${python} has Unicode ${version}, not ${PYTHON_UNIDATA_VERSION}`);
  return list.split(' ').map(Number);
}

// The ranges Python leaves unassigned and this Node assigns.
export function assignedAfterUnicode15(python) {
  const unassigned = /\p{Cn}/u;
  return toRanges(pythonUnassigned(python).filter((code) => !unassigned.test(String.fromCodePoint(code))));
}

const hex = (code) => `0x${code.toString(16).toUpperCase().padStart(4, '0')}`;

export function moduleSource(ranges, unicodeVersion = process.versions.unicode) {
  const rows = [];
  for (let index = 0; index < ranges.length; index += 4) {
    rows.push(`  ${ranges.slice(index, index + 4).map(([start, end]) => `[${hex(start)}, ${hex(end)}]`).join(', ')},`);
  }
  return `// Code points assigned after Unicode 15.0.0, the version of Python 3.12's
// unicodedata and re (the runners' CPython 3.12.3), and assigned in Unicode
// ${unicodeVersion}, the version of the Node regex data this list was generated with.
// Node's \\p{...} classes follow the ICU of the running Node (Unicode 16 or
// 17 on Node 24), so each class that stands for a Python character predicate
// (str.isprintable, re's \\w and \\d, int()'s digits) must leave these out to
// accept exactly what Python 3.12 accepts. Python sees every one of them as
// an unassigned (Cn) character.
//
// Generated; do not edit. To regenerate on the newest supported Node:
//
//   node tests/release/unicode15_generate.mjs [python3.12] > scripts/release/unicode15.mjs
//
// tests/release/unicode15_test.mjs regenerates it whenever a Python with
// Unicode 15.0.0 is available and checks each derived class against Python.

export const GENERATED_UNICODE_VERSION = '${unicodeVersion}';

// Inclusive [first, last] code point ranges, ascending.
export const ASSIGNED_AFTER_UNICODE_15 = Object.freeze([
${rows.join('\n')}
].map((range) => Object.freeze(range)));

const escape = (code) => \`\\\\u{\${code.toString(16)}}\`;

// The ranges as the body of a character class for a regex with the u flag.
export const ASSIGNED_AFTER_UNICODE_15_CLASS = ASSIGNED_AFTER_UNICODE_15
  .map(([start, end]) => (start === end ? escape(start) : \`\${escape(start)}-\${escape(end)}\`))
  .join('');
`;
}

if (import.meta.main) {
  const python = process.argv[2] ?? findPython312();
  if (!python) {
    process.stderr.write(`no python3.12 or python3 with Unicode ${PYTHON_UNIDATA_VERSION} found\n`);
    process.exitCode = 1;
  } else {
    process.stdout.write(moduleSource(assignedAfterUnicode15(python)));
  }
}
