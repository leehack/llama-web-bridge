// Tests of scripts/build/verify_emscripten_version.mjs, the port of
// scripts/verify_emscripten_version.py. Each test runs the script as a
// process from a scratch checkout (the script, the scripts/release modules it
// imports, and an emsdk.version) with a fake `emcc` first on PATH, and checks
// stdout, stderr, the exit code and the env file byte for byte. The expected
// text is what CPython 3.12 prints for the same inputs; usage and help text are
// scripts/release/cli.mjs's layout (with the .mjs program name), and the
// parser is stricter than argparse where cli.mjs is (no abbreviations).

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, test } from 'node:test';

const REPO = path.resolve(import.meta.dirname, '..', '..');
const NORMAL = 'emcc (Emscripten gcc/clang-like replacement + linker emulating GNU ld) 6.0.8 (a1b2c3)\nCopyright (C) 2014 the Emscripten authors\n';
const USAGE = 'usage: verify_emscripten_version.mjs [-h] [--print-pin] [--emit-github-env PATH]\n';

let root;
let bin;
beforeEach(() => {
  root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'verify-emscripten-')));
  fs.mkdirSync(path.join(root, 'scripts', 'build'), { recursive: true });
  fs.copyFileSync(
    path.join(REPO, 'scripts', 'build', 'verify_emscripten_version.mjs'),
    path.join(root, 'scripts', 'build', 'verify_emscripten_version.mjs'),
  );
  fs.cpSync(path.join(REPO, 'scripts', 'release'), path.join(root, 'scripts', 'release'), { recursive: true });
  bin = path.join(root, 'bin');
  fs.mkdirSync(bin);
  writePin('6.0.8\n');
});
afterEach(() => {
  fs.rmSync(root, { recursive: true, force: true });
});

function writePin(bytes) {
  fs.writeFileSync(path.join(root, 'emsdk.version'), bytes);
}

// A fake emcc that prints `stdout` and `stderr` (bytes) and exits `status`.
function fakeEmcc(stdout, { stderr = '', status = 0 } = {}) {
  fs.writeFileSync(path.join(bin, 'out'), stdout);
  fs.writeFileSync(path.join(bin, 'err'), stderr);
  const script = path.join(bin, 'emcc');
  fs.writeFileSync(script, `#!/bin/sh\ncat '${bin}/out'\ncat '${bin}/err' >&2\nexit ${status}\n`);
  fs.chmodSync(script, 0o755);
}

function run(args, { withEmcc = true, env = {} } = {}) {
  const pathDirs = [...(withEmcc ? [bin] : []), path.dirname(process.execPath), '/usr/bin', '/bin'];
  const result = spawnSync(process.execPath, [path.join(root, 'scripts', 'build', 'verify_emscripten_version.mjs'), ...args], {
    cwd: root,
    env: { PATH: pathDirs.join(path.delimiter), ...env },
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  return { status: result.status, stdout: result.stdout, stderr: result.stderr };
}

test('a matching emcc is validated', () => {
  fakeEmcc(NORMAL);
  assert.deepEqual(run([]), { status: 0, stdout: 'Validated Emscripten 6.0.8 against emsdk.version\n', stderr: '' });
});

test('--print-pin prints the pin without invoking emcc', () => {
  writePin(' 6.0.8 \r\n');
  assert.deepEqual(run(['--print-pin'], { withEmcc: false }), { status: 0, stdout: '6.0.8\n', stderr: '' });
});

test('--emit-github-env appends the verified version', () => {
  fakeEmcc(NORMAL);
  const envFile = path.join(root, 'github-env');
  fs.writeFileSync(envFile, 'PREEXISTING=1\n');
  assert.deepEqual(run(['--emit-github-env', envFile]), {
    status: 0,
    stdout: 'Validated Emscripten 6.0.8 against emsdk.version\n',
    stderr: '',
  });
  assert.equal(fs.readFileSync(envFile, 'utf8'), 'PREEXISTING=1\nEMSCRIPTEN_VERSION=6.0.8\n');
});

test('a mismatched emcc fails without touching the env file', () => {
  fakeEmcc(NORMAL.replace('6.0.8', '6.0.9'));
  const envFile = path.join(root, 'github-env');
  fs.writeFileSync(envFile, 'PREEXISTING=1\n');
  assert.deepEqual(run(['--emit-github-env', envFile]), {
    status: 1,
    stdout: '',
    stderr: 'error: resolved Emscripten 6.0.9 does not match emsdk.version 6.0.8\n',
  });
  assert.equal(fs.readFileSync(envFile, 'utf8'), 'PREEXISTING=1\n');
});

test('only the first line is parsed, with Python\'s splitlines and \\b', () => {
  for (const [stdout, stderr] of [
    ['emcc (x) 6.0.8\r\nmore\r\n', null],
    // The greedy \(.*\) runs to the last parenthesis, as in Python.
    ['emcc (a) (b) 6.0.8 (c) 7.0.0\n', 'error: resolved Emscripten 7.0.0 does not match emsdk.version 6.0.8\n'],
    ['emcc (x) 6.0.8 extra', null],
    ['emcc (x) 6.0.8_\n', "error: could not parse Emscripten version from: 'emcc (x) 6.0.8_'\n"],
    ['emcc (x) 6.0.8\u{e9}\n', "error: could not parse Emscripten version from: 'emcc (x) 6.0.8\u{e9}'\n"],
    // Letters and digits assigned after Unicode 15.0 are unassigned to
    // Python 3.12, so \b holds before them; U+0D66 has been a digit since 1.1.
    ['emcc (x) 6.0.8\u{10d40}\n', null],
    ['emcc (x) 6.0.8\u{13556}\n', null],
    ['emcc (x) 6.0.8\u{d66}\n', "error: could not parse Emscripten version from: 'emcc (x) 6.0.8\u{d66}'\n"],
    // \b holds before '.', so a fourth component is ignored, as in Python.
    ['emcc (x) 6.0.8.1\n', null],
    ['emcc (x)\r6.0.8\n', "error: could not parse Emscripten version from: 'emcc (x)'\n"],
    ['emcc (x)\u{2028} 6.0.8\n', "error: could not parse Emscripten version from: 'emcc (x)'\n"],
    ['\nemcc (x) 6.0.8\n', "error: could not parse Emscripten version from: ''\n"],
    ['', "error: could not parse Emscripten version from: ''\n"],
    ['emcc \x07\u{200b}\u{a0} x\n', "error: could not parse Emscripten version from: 'emcc \\x07\\u200b\\xa0 x'\n"],
  ]) {
    fakeEmcc(stdout);
    const expected = stderr === null
      ? { status: 0, stdout: 'Validated Emscripten 6.0.8 against emsdk.version\n', stderr: '' }
      : { status: 1, stdout: '', stderr };
    assert.deepEqual(run([]), expected, JSON.stringify(stdout));
  }
});

test('a failing emcc reports stderr, then stdout, then "no output"', () => {
  fakeEmcc('out\n', { stderr: '  boom: \u{2717} bad\n\n', status: 1 });
  assert.deepEqual(run([]), { status: 1, stdout: '', stderr: 'error: emcc --version failed: boom: \u{2717} bad\n' });
  fakeEmcc('  only stdout\n', { status: 2 });
  assert.deepEqual(run([]), { status: 1, stdout: '', stderr: 'error: emcc --version failed: only stdout\n' });
  fakeEmcc(' \n', { stderr: '\t\n', status: 3 });
  assert.deepEqual(run([]), { status: 1, stdout: '', stderr: 'error: emcc --version failed: no output\n' });
});

test('emcc output that is not UTF-8 is a UnicodeDecodeError, reported as an error', () => {
  fakeEmcc(Buffer.from('emcc (x) 6.0.8\xff\n', 'latin1'));
  assert.deepEqual(run([]), {
    status: 1,
    stdout: '',
    stderr: "error: 'utf-8' codec can't decode byte 0xff in position 14: invalid start byte\n",
  });
});

test('a missing emcc is the OSError Python raises', () => {
  assert.deepEqual(run([], { withEmcc: false }), {
    status: 1,
    stdout: '',
    stderr: "error: [Errno 2] No such file or directory: 'emcc'\n",
  });
});

test('an invalid pin is refused in every mode', () => {
  fakeEmcc(NORMAL);
  for (const [pin, rendered] of [
    ['6.0\n', "'6.0'"],
    ['v6.0.8\n', "'v6.0.8'"],
    ['6.0.8\n6.0.9\n', "'6.0.8\\n6.0.9'"],
    ['\u{feff}6.0.8\n', "'\\ufeff6.0.8'"],
    ['6.0.8\r7\n', "'6.0.8\\n7'"],
    ['', "''"],
  ]) {
    writePin(pin);
    for (const args of [[], ['--print-pin']]) {
      assert.deepEqual(run(args), {
        status: 1,
        stdout: '',
        stderr: `error: emsdk.version must contain one semantic Emscripten version, got: ${rendered}\n`,
      }, JSON.stringify([pin, args]));
    }
  }
  writePin(Buffer.from([0xff, 0x36]));
  assert.deepEqual(run(['--print-pin']), {
    status: 1,
    stdout: '',
    stderr: "error: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte\n",
  });
  fs.rmSync(path.join(root, 'emsdk.version'));
  assert.deepEqual(run(['--print-pin']), {
    status: 1,
    stdout: '',
    stderr: `error: [Errno 2] No such file or directory: '${path.join(root, 'emsdk.version')}'\n`,
  });
});

test('an env file that cannot be opened is the OSError Python raises', () => {
  fakeEmcc(NORMAL);
  assert.deepEqual(run(['--emit-github-env', 'missing/dir//env']), {
    status: 1,
    stdout: '',
    stderr: "error: [Errno 2] No such file or directory: 'missing/dir/env'\n",
  });
});

test('--help and argparse errors match CPython 3.12, in cli.mjs\'s layout', () => {
  const help = `${USAGE}\noptions:\n`
    + '  -h, --help            show this help message and exit\n'
    + '  --print-pin\n'
    + '  --emit-github-env PATH\n';
  for (const args of [['-h'], ['--help'], ['--print-pin', '--help'], ['--bogus', '-h']]) {
    assert.deepEqual(run(args, { withEmcc: false }), { status: 0, stdout: help, stderr: '' }, JSON.stringify(args));
  }
  for (const [args, message] of [
    [['--bogus'], 'unrecognized arguments: --bogus'],
    [['x'], 'unrecognized arguments: x'],
    [['--emit-github-env'], 'argument --emit-github-env: expected one argument'],
    [['--emit-github-env', '--print-pin'], 'argument --emit-github-env: expected one argument'],
    [['--print-pin=1'], 'argument --print-pin: ignored explicit argument \'1\''],
    // Stricter than argparse, which takes these abbreviations.
    [['--h'], 'unrecognized arguments: --h'],
    [['--print'], 'unrecognized arguments: --print'],
  ]) {
    assert.deepEqual(run(args, { withEmcc: false }), {
      status: 2,
      stdout: '',
      stderr: `${USAGE}verify_emscripten_version.mjs: error: ${message}\n`,
    }, JSON.stringify(args));
  }
});

test('a value that looks like a negative number, in any Unicode digits, is an env file path', () => {
  fakeEmcc(NORMAL);
  // argparse's negative-number pattern uses re's Unicode \d.
  const name = '-\u{661}';
  assert.deepEqual(run(['--emit-github-env', name]), {
    status: 0,
    stdout: 'Validated Emscripten 6.0.8 against emsdk.version\n',
    stderr: '',
  });
  assert.equal(fs.readFileSync(path.join(root, name), 'utf8'), 'EMSCRIPTEN_VERSION=6.0.8\n');
  // U+10D40 is a digit only since Unicode 16.
  assert.equal(run(['--emit-github-env', '-\u{10d40}']).status, 2);
});
