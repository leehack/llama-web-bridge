#!/usr/bin/env node
// Resolve and verify the checked-in Emscripten SDK version, with the output
// and exit codes of the deleted scripts/verify_emscripten_version.py. It
// reads the emsdk.version of the checkout it lives in (two levels up), runs
// `emcc --version` from that checkout's root, and compares.
//
// The command line goes through scripts/release/cli.mjs, so it follows that
// parser's policy: argparse's accept/reject and error lines, except that it is
// stricter (no abbreviated options such as --print or --h, no `--`), and its
// usage and help text are cli.mjs's layout rather than argparse's wrapped one.

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { SystemExitError, parseArguments, progName, runCli } from '../release/cli.mjs';
import {
  PY_WORD, PyException, isPyException, pyDecodeUtf8, pyRepr, pySplitlines, pyStrip, pyUniversalNewlines,
} from '../release/json.mjs';
import { isOSError, osErrorString } from '../release/python_compat.mjs';

const ROOT = path.resolve(path.dirname(fs.realpathSync(fileURLToPath(import.meta.url))), '..', '..');
const PIN_NAME = 'emsdk.version';
const PIN_PATH = path.join(ROOT, PIN_NAME);
const VERSION_RE = /^[0-9]+\.[0-9]+\.[0-9]+$/;
// r"^emcc \(.*\) ([0-9]+\.[0-9]+\.[0-9]+)\b": after a digit, \b holds where
// the next character is not a word character (or there is none).
const EMCC_VERSION_RE = new RegExp(String.raw`^emcc \(.*\) ([0-9]+\.[0-9]+\.[0-9]+)(?!${PY_WORD})`, 'u');

const OPTIONS = Object.freeze([
  { flag: '--print-pin', type: 'flag' },
  { flag: '--emit-github-env', type: 'path', metavar: 'PATH' },
]);

// Text-mode decoding with universal newlines.
const decodeText = (data) => pyUniversalNewlines(pyDecodeUtf8(data));

// The OSError Python raises for a failed call on `filename`.
const osError = (error, filename) => (isOSError(error) ? new PyException('OSError', osErrorString(error, filename)) : error);

export function readPin() {
  let raw;
  try {
    raw = fs.readFileSync(PIN_PATH);
  } catch (error) {
    throw osError(error, PIN_PATH);
  }
  const version = pyStrip(decodeText(raw));
  if (!VERSION_RE.test(version)) {
    throw new PyException('ValueError', `${PIN_NAME} must contain one semantic Emscripten version, got: ${pyRepr(version)}`);
  }
  return version;
}

export function resolveEmccVersion() {
  const result = spawnSync('emcc', ['--version'], { cwd: ROOT, encoding: 'buffer', maxBuffer: Infinity });
  if (result.error) {
    // Python names the working directory when it is the one that is missing.
    const cwdMissing = result.error.code === 'ENOENT' && !fs.existsSync(ROOT);
    throw osError(result.error, cwdMissing ? ROOT : 'emcc');
  }
  const stdout = decodeText(result.stdout);
  const stderr = decodeText(result.stderr);
  if (result.status !== 0) {
    const detail = pyStrip(stderr) || pyStrip(stdout) || 'no output';
    throw new PyException('RuntimeError', `emcc --version failed: ${detail}`);
  }
  const firstLine = pySplitlines(stdout)[0] ?? '';
  const match = EMCC_VERSION_RE.exec(firstLine);
  if (match === null) throw new PyException('RuntimeError', `could not parse Emscripten version from: ${pyRepr(firstLine)}`);
  return match[1];
}

// main() of the Python script: `error: <exc>` and exit status 1 for an
// OSError, RuntimeError or ValueError; anything else escapes.
export function main(argv, write) {
  const args = parseArguments(argv, { prog: progName(import.meta.url), options: OPTIONS });
  try {
    const expected = readPin();
    if (args.printPin) {
      write(`${expected}\n`);
      return 0;
    }
    const resolved = resolveEmccVersion();
    if (resolved !== expected) {
      throw new PyException('RuntimeError', `resolved Emscripten ${resolved} does not match ${PIN_NAME} ${expected}`);
    }
    if (args.emitGithubEnv !== null) {
      try {
        fs.appendFileSync(args.emitGithubEnv, `EMSCRIPTEN_VERSION=${resolved}\n`, 'utf8');
      } catch (error) {
        throw osError(error, args.emitGithubEnv);
      }
    }
    write(`Validated Emscripten ${resolved} against ${PIN_NAME}\n`);
    return 0;
  } catch (error) {
    if (isPyException(error, 'OSError', 'RuntimeError', 'ValueError')) throw new SystemExitError(`error: ${error.message}`);
    throw error;
  }
}

if (import.meta.main) runCli(main);
