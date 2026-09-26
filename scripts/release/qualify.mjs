// The smoke-running half of release_qualification.py: the `qualify` command,
// the heavy-gate child processes it runs, and the parsing of their results
// into attestation phases. scripts/release/qualification.mjs holds the rest
// (constants, redaction, the attestation, the candidate, the CLI) and imports
// this module dynamically for `qualify`.
//
// Parity with Python 3.12 on the ubuntu-24.04 runners: the same accepted and
// rejected smoke results with the same error text, the same attestation phase
// contents, the same child environment, process group, signals and grace
// periods, and the same diagnostics files. The deviations, each for something
// Node cannot do the Python way:
//
// - Peak child RSS. Python reads getrusage(RUSAGE_CHILDREN).ru_maxrss, which
//   Node does not expose. Each smoke instead runs under /usr/bin/time, whose
//   wait4() reports the same ru_maxrss the kernel folds into the parent's
//   RUSAGE_CHILDREN when it reaps that smoke (the smoke's own peak or that of
//   any descendant it reaped, whichever is larger). maxRssBytes() is the
//   largest value over every smoke this process ran, which is cumulative
//   across phases like RUSAGE_CHILDREN. It leaves out the short-lived
//   non-smoke children Python also reaps (`node --version`, gh, git), which
//   stay far below a smoke's peak. See rssWrapper() for the Linux and macOS
//   forms, and for what happens without /usr/bin/time.
// - qualifyCmd() and runSmoke() are async: a child with its own session, a
//   timeout and a grace period cannot be run with spawnSync().
// - Output a smoke writes is kept in memory, as Python keeps it; text beyond
//   V8's maximum string length (about 512 MiB) raises a ContractError where
//   Python would carry on.

import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { ContractError } from './errors.mjs';
import {
  JSONDecodeError,
  PY_DECIMAL,
  PyException,
  PyFloat,
  compareCodePoints,
  isDict,
  isPyInt,
  pyDecodeUtf8,
  pyGet,
  pyIntFromString,
  pyIsFile,
  pyIsSymlink,
  pyJoinPath,
  pyJsonLoads,
  pyPath,
  pyRepr,
  pyStr,
  pyStrip,
  pyUniversalNewlines,
  pyWriteText,
} from './json.mjs';
import { rejectDuplicateKeys } from './contract.mjs';
import { osErrorString, pyExpanduser, pyResolve } from './python_compat.mjs';
import {
  EXPECTED_SPEECH_TRANSCRIPT,
  MAX_ATTESTATION_BYTES,
  SPEECH_AUDIO_SHA256,
  SPEECH_MMPROJ_SHA256,
  SPEECH_MODEL_SHA256,
  SPEECH_SMOKE,
  TTS_MMPROJ_SHA256,
  TTS_MODEL_SHA256,
  TTS_SMOKE,
  buildAttestation,
  canonicalJson,
  fetchCandidate,
  loadCandidate,
  normalizeTranscript,
  parseCancellationResult,
  qualificationEnvironment,
  qualificationRunIdentity,
  rejectNonstandardJsonConstant,
  requireHarnessMatchesBridgeSource,
  requirePositiveInt,
  requireStr,
  sanitizeDiagnosticStdout,
  sanitizeDiagnosticText,
  verifyAttestation,
} from './qualification.mjs';
import { readWavIdentity } from './wav.mjs';

const POSIX = process.platform !== 'win32';

// The directory the harness runs from: Python's Path(__file__).resolve().parent
// is scripts/, and this module lives in scripts/release/.
export const SCRIPTS_DIR = path.dirname(path.dirname(fs.realpathSync(fileURLToPath(import.meta.url))));

// KeyboardInterrupt, raised after a smoke interrupted by SIGINT is stopped and
// its diagnostics are written. Uncaught, it ends the process the way Python
// does: its traceback, then death by SIGINT (cli.mjs runCliAsync).
export class KeyboardInterrupt extends PyException {
  constructor() {
    super('KeyboardInterrupt', '');
  }
}

// subprocess.TimeoutExpired, whose str() is
// "Command '%s' timed out after %s seconds" % (cmd, timeout): the args list
// as str(list) inside quotes, the timeout as str(), so an int timeout has no
// fraction (a PyFloat keeps its '.0').
export class TimeoutExpired extends PyException {
  constructor(command, timeoutSeconds) {
    super('subprocess.TimeoutExpired', `Command '${pyStr(command)}' timed out after ${pyStr(timeoutSeconds)} seconds`);
    this.timeout = timeoutSeconds;
  }
}

// --- Python helpers ------------------------------------------------------------

// format(value, 'g'): six significant digits rounded half to even on the
// exact binary value, as Python rounds, trailing zeros dropped, and an
// exponent outside 1e-4 <= |value| < 1e6. toFixed(100) is the exact decimal
// expansion of every double this sees (any tie at the sixth digit terminates
// well before the 100th decimal); values outside its range fall back to
// toExponential, which breaks ties upward.
export function pyFormatG(value) {
  const number = Number(value);
  if (Number.isNaN(number)) return 'nan';
  if (number === Infinity) return 'inf';
  if (number === -Infinity) return '-inf';
  if (number === 0) return Object.is(number, -0) ? '-0' : '0';
  const sign = number < 0 ? '-' : '';
  const magnitude = Math.abs(number);
  let digits;
  let exponent;
  if (magnitude >= 1e-90 && magnitude < 1e21) {
    const [whole, fraction] = magnitude.toFixed(100).split('.');
    const all = `${whole}${fraction}`;
    const first = all.search(/[1-9]/);
    exponent = whole.length - 1 - first;
    const significant = all.slice(first);
    const head = BigInt(significant.slice(0, 6).padEnd(6, '0'));
    const rest = significant.slice(6).replace(/0+$/, '');
    let rounded = head;
    if (rest !== '' && (rest[0] > '5' || (rest[0] === '5' && (rest.length > 1 || head % 2n === 1n)))) rounded += 1n;
    digits = String(rounded);
    if (digits.length > 6) {
      digits = digits.slice(0, 6);
      exponent += 1;
    }
  } else {
    const [mantissa, exponentText] = magnitude.toExponential(5).split('e');
    digits = mantissa.replace('.', '');
    exponent = Number(exponentText);
  }
  if (exponent >= -4 && exponent < 6) {
    let text;
    if (exponent >= 0) text = `${digits.slice(0, exponent + 1).padEnd(exponent + 1, '0')}.${digits.slice(exponent + 1)}`;
    else text = `0.${'0'.repeat(-exponent - 1)}${digits}`;
    text = text.replace(/0+$/, '').replace(/\.$/, '');
    return sign + text;
  }
  const mantissa = `${digits[0]}.${digits.slice(1)}`.replace(/0+$/, '').replace(/\.$/, '');
  return `${sign}${mantissa}e${exponent < 0 ? '-' : '+'}${String(Math.abs(exponent)).padStart(2, '0')}`;
}

// sum() over Python ints held as safe-integer numbers or bigints.
function pySumInts(values) {
  let total = 0n;
  for (const value of values) total += BigInt(value);
  return total >= BigInt(Number.MIN_SAFE_INTEGER) && total <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(total) : total;
}

// int arithmetic on a parsed argument (a number or a bigint).
function pyIntAdd(value, addend) {
  return pySumInts([value, addend]);
}

function pyIntMultiply(value, factor) {
  const product = BigInt(value) * BigInt(factor);
  return product >= BigInt(Number.MIN_SAFE_INTEGER) && product <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(product) : product;
}

// PurePosixPath(text).name.
export function pyPathName(text) {
  const normalized = pyPath(text);
  if (normalized === '.') return '';
  return normalized.slice(normalized.lastIndexOf('/') + 1);
}

const OS_ERROR_TYPES = Object.freeze({
  EACCES: 'PermissionError',
  EEXIST: 'FileExistsError',
  EISDIR: 'IsADirectoryError',
  ENOENT: 'FileNotFoundError',
  ENOTDIR: 'NotADirectoryError',
  EPERM: 'PermissionError',
  ESRCH: 'ProcessLookupError',
});

// The OSError subclass Python raises for a failed call on `filename` (none
// when null), with its str().
function toPyOSError(error, filename = null) {
  if (!error || typeof error.code !== 'string' || typeof error.errno !== 'number') return error;
  const exception = new PyException(OS_ERROR_TYPES[error.code] ?? 'OSError', osErrorString(error, filename));
  exception.errno = Math.abs(error.errno);
  exception.code = error.code;
  exception.filename = filename;
  return exception;
}

function pyPathJoin(left, right) {
  if (right.startsWith('/')) return right;
  if (left === '' || left.endsWith('/')) return `${left}${right}`;
  return `${left}/${right}`;
}

// shutil.which(name) (3.12) for a bare name on POSIX.
export function pyWhich(name, env = process.env) {
  let searchPath = Object.hasOwn(env, 'PATH') ? env.PATH : (process.platform === 'darwin' ? '/usr/bin:/bin:/usr/sbin:/sbin' : '/bin:/usr/bin');
  if (!searchPath) return null;
  const seen = new Set();
  for (const directory of searchPath.split(':')) {
    if (seen.has(directory)) continue;
    seen.add(directory);
    const candidate = directory === '' ? name : pyPathJoin(directory, name);
    try {
      fs.accessSync(candidate, fs.constants.F_OK | fs.constants.X_OK);
      if (!fs.statSync(candidate).isDirectory()) return candidate;
    } catch {
      // not an executable file
    }
  }
  return null;
}

// --- Peak child RSS ---------------------------------------------------------------

// How a smoke is wrapped so its wait4() rusage is recorded:
// - Linux: GNU time (`time` is in the ubuntu-24.04 runner image's apt
//   toolset) with `-f %M -o <file>`, which writes ru_maxrss in KiB, the unit
//   Python multiplies by 1024; a signal death is recorded in the same file as
//   "Command terminated by signal N", so the exit status stays the smoke's.
// - macOS: the system BSD time with `-l -o <file>`, whose "maximum resident set
//   size" is in bytes, as Python's darwin branch reports. It reports a signal
//   death on stderr and exits by re-raising the signal (SIGKILL cannot be
//   re-raised: it prints "signal: Invalid argument" and exits 1); both lines
//   are removed from the smoke's stderr and the smoke's status recovered.
// - Anything else, or no usable /usr/bin/time: the smoke runs unwrapped,
//   exactly as Python runs it, and maxRssBytes() stays 0 as Python's would
//   with no child reaped; an attestation then fails its positive max_rss_bytes
//   check, so nothing unmeasured is ever attested.
const GNU_TIME = Object.freeze({
  kind: 'gnu',
  argv: (outputFile, command) => ['/usr/bin/time', '-f', '%M', '-o', outputFile, '--', ...command],
});
const BSD_TIME = Object.freeze({
  kind: 'bsd',
  argv: (outputFile, command) => ['/usr/bin/time', '-l', '-o', outputFile, '--', ...command],
});

let detectedWrapper;

export function rssWrapper() {
  if (detectedWrapper !== undefined) return detectedWrapper;
  detectedWrapper = null;
  if (process.platform === 'linux') {
    const probe = spawnSync('/usr/bin/time', ['--version'], { encoding: 'utf8', env: {}, timeout: 30000 });
    if (!probe.error && /GNU Time/.test(`${probe.stdout}${probe.stderr}`)) detectedWrapper = GNU_TIME;
  } else if (process.platform === 'darwin') {
    try {
      fs.accessSync('/usr/bin/time', fs.constants.X_OK);
      detectedWrapper = BSD_TIME;
    } catch {
      // unwrapped
    }
  }
  return detectedWrapper;
}

let peakChildRssBytes = 0;

// The peak RSS in bytes of every smoke this process has run so far: like
// RUSAGE_CHILDREN it is cumulative and cannot be reset, so a phase's value is
// the peak across every smoke up to the end of that phase, not that phase's
// contribution in isolation.
export function maxRssBytes() {
  return peakChildRssBytes;
}

function recordChildRss(bytes) {
  if (Number.isSafeInteger(bytes) && bytes > peakChildRssBytes) peakChildRssBytes = bytes;
}

const BSD_ABNORMAL = Buffer.from('time: command terminated abnormally\n');
const BSD_SIGNAL_EINVAL = Buffer.from('time: signal: Invalid argument\n');

function endsWith(buffer, suffix) {
  return buffer.length >= suffix.length && buffer.subarray(buffer.length - suffix.length).equals(suffix);
}

// --- The smoke child process -------------------------------------------------------

// _child_env: strip ambient smoke configuration so no gate can be silently
// redirected. Every model, projector, fixture, URL, and timeout is passed
// explicitly on the command line. An inherited LLAMA_WEBGPU_* variable could
// otherwise point a gate at a different file or at the network without
// appearing in the record.
const CHILD_ENV_ALLOWED = new Set([
  'DISPLAY',
  'HOME',
  'LANG',
  'LC_ALL',
  'LC_CTYPE',
  'NO_COLOR',
  'PATH',
  'PLAYWRIGHT_BROWSERS_PATH',
  'SSL_CERT_DIR',
  'SSL_CERT_FILE',
  'SYSTEMROOT',
  'TEMP',
  'TERM',
  'TMP',
  'TMPDIR',
  'WINDIR',
  'XDG_CACHE_HOME',
]);

export function childEnv(env = process.env) {
  const source = { ...env };
  const coerced = pythonCoercedLocale(env);
  if (coerced !== null) source.LC_CTYPE = coerced;
  const result = {};
  for (const [key, value] of Object.entries(source)) if (CHILD_ENV_ALLOWED.has(key)) result[key] = value;
  return result;
}

// PEP 538: a Python started in the legacy C locale sets LC_CTYPE=C.UTF-8 in
// its own environment, which _child_env then passes on. The C locale is what
// setlocale(LC_CTYPE, "") picks from LC_ALL, LC_CTYPE and LANG when they are
// unset, empty or "C" (and "POSIX" on glibc), and coercion is off when LC_ALL
// is set or PYTHONCOERCECLOCALE=0. A locale name that is not installed also
// leaves Python in the C locale; that case is not detected here. The
// runners set LANG=C.UTF-8, so nothing is coerced there.
export function pythonCoercedLocale(env = process.env, platform = process.platform) {
  if (env.PYTHONCOERCECLOCALE === '0' || env.LC_ALL) return null;
  const ctype = env.LC_CTYPE || env.LANG || '';
  const legacy = ctype === '' || ctype === 'C' || (platform !== 'darwin' && ctype === 'POSIX');
  return legacy ? 'C.UTF-8' : null;
}

// The OSError Popen raises when the program cannot be executed: checked
// before spawning, because a wrapped smoke would otherwise only report it as
// the wrapper's exit status 127 or 126.
function requireExecutable(program, env) {
  const fail = (code) => {
    const errno = os.constants.errno[code];
    const type = code === 'ENOENT' ? 'FileNotFoundError' : 'PermissionError';
    const exception = new PyException(type, `${osErrorString({ code, errno: -errno })}: ${pyRepr(program)}`);
    exception.errno = errno;
    exception.code = code;
    exception.filename = program;
    throw exception;
  };
  const executable = (candidate) => {
    let stats;
    try {
      stats = fs.statSync(candidate);
    } catch {
      return 'ENOENT';
    }
    // Root passes access(X_OK) on some filesystems for a file without any
    // execute bit, which execve() still refuses.
    if (!stats.isFile() || (stats.mode & 0o111) === 0) return 'EACCES';
    try {
      fs.accessSync(candidate, fs.constants.X_OK);
      return null;
    } catch {
      return 'EACCES';
    }
  };
  if (program.includes('/')) {
    const error = executable(program);
    if (error) fail(error);
    return;
  }
  const searchPath = Object.hasOwn(env, 'PATH') ? env.PATH : '/bin:/usr/bin';
  let saved = null;
  for (const directory of searchPath.split(':')) {
    const error = executable(pyPathJoin(directory, program));
    if (!error) return;
    if (error !== 'ENOENT' && saved === null) saved = error;
  }
  fail(saved ?? 'ENOENT');
}

function sleep(seconds, onTimer) {
  // setTimeout fires at once past 2**31-1 ms, so a long wait is chained.
  const MAX_DELAY = 2 ** 31 - 1;
  let remaining = Math.max(0, seconds * 1000);
  let handle;
  const promise = new Promise((resolve) => {
    const step = () => {
      if (remaining <= 0) {
        resolve();
        return;
      }
      const delay = Math.min(remaining, MAX_DELAY);
      remaining -= delay;
      handle = setTimeout(step, delay);
    };
    step();
  });
  onTimer?.(() => clearTimeout(handle));
  return promise;
}

// Popen(command, stdout=PIPE, stderr=PIPE, text=True, encoding="utf-8",
// errors="replace", env=_child_env(), start_new_session=True): a child in its
// own session (and so its own process group), stdin inherited, both output
// streams captured as bytes and decoded once they are complete.
export class SmokeProcess {
  constructor(command, { env = childEnv(), wrapper = rssWrapper() } = {}) {
    requireExecutable(command[0], env);
    this.command = command;
    this.wrapper = wrapper;
    let argv = command;
    if (wrapper) {
      this.rssDir = fs.mkdtempSync(path.join(os.tmpdir(), 'llama-web-bridge-qualification-rss-'));
      this.rssFile = path.join(this.rssDir, 'rusage.txt');
      argv = wrapper.argv(this.rssFile, command);
    }
    this.stdoutChunks = [];
    this.stderrChunks = [];
    try {
      this.child = spawn(argv[0], argv.slice(1), {
        detached: POSIX,
        env,
        stdio: ['inherit', 'pipe', 'pipe'],
      });
    } catch (error) {
      this.removeRssDir();
      throw error;
    }
    this.pid = this.child.pid;
    this.child.stdout.on('data', (chunk) => this.stdoutChunks.push(chunk));
    this.child.stderr.on('data', (chunk) => this.stderrChunks.push(chunk));
    this.returncode = null;
    this.closed = new Promise((resolve, reject) => {
      this.child.once('error', (error) => {
        // A child that never started may emit 'error' without 'close'.
        this.removeRssDir();
        reject(toPyOSError(error, command[0]));
      });
      this.child.once('close', (code, signal) => {
        try {
          this.finish(code, signal);
          resolve();
        } catch (error) {
          reject(error);
        }
      });
    });
    // Settled through communicate(); never an unhandled rejection meanwhile.
    this.closed.catch(() => {});
  }

  // Remove the wrapper's report directory; safe to call more than once.
  removeRssDir() {
    if (this.rssDir) fs.rmSync(this.rssDir, { recursive: true, force: true });
  }

  // Popen.poll(): null while the direct child has not exited.
  poll() {
    if (this.child.exitCode !== null) return this.child.exitCode;
    if (this.child.signalCode !== null) return -os.constants.signals[this.child.signalCode];
    return null;
  }

  finish(code, signal) {
    let returncode = signal ? -os.constants.signals[signal] : code;
    let stderr = Buffer.concat(this.stderrChunks);
    if (this.wrapper) {
      let report = '';
      try {
        report = fs.readFileSync(this.rssFile, 'utf8');
      } catch {
        // the wrapper was killed before it reported
      }
      this.removeRssDir();
      if (this.wrapper.kind === 'gnu') {
        const lines = report.split('\n').filter((line) => line !== '');
        const kib = /^[0-9]+$/.test(lines.at(-1) ?? '') ? Number(lines.at(-1)) : null;
        if (kib !== null) recordChildRss(kib * 1024);
        const died = lines.map((line) => /^Command terminated by signal ([0-9]+)$/.exec(line)).find(Boolean);
        if (!signal && died) returncode = -Number(died[1]);
      } else {
        const match = /^\s*([0-9]+)\s+maximum resident set size$/m.exec(report);
        if (match) recordChildRss(Number(match[1]));
        if (report && code === 1 && endsWith(stderr, Buffer.concat([BSD_ABNORMAL, BSD_SIGNAL_EINVAL]))) {
          stderr = stderr.subarray(0, stderr.length - BSD_ABNORMAL.length - BSD_SIGNAL_EINVAL.length);
          returncode = -os.constants.signals.SIGKILL;
        } else if (report && signal && endsWith(stderr, BSD_ABNORMAL)) {
          stderr = stderr.subarray(0, stderr.length - BSD_ABNORMAL.length);
        }
      }
    }
    this.returncode = returncode;
    this.stdoutBytes = Buffer.concat(this.stdoutChunks);
    this.stderrBytes = stderr;
  }

  // Popen.communicate(timeout=...): the decoded output once both streams are
  // closed and the child has exited, or TimeoutExpired, keeping what was read.
  async communicate(timeoutSeconds = null) {
    if (timeoutSeconds === null) {
      await this.closed;
    } else {
      let cancel;
      const timer = sleep(timeoutSeconds, (clear) => { cancel = clear; }).then(() => 'timeout');
      const outcome = await Promise.race([this.closed.then(() => 'closed'), timer]);
      cancel();
      if (outcome === 'timeout') throw new TimeoutExpired(this.command, timeoutSeconds);
    }
    return [decodeText(this.stdoutBytes), decodeText(this.stderrBytes)];
  }

  // os.killpg(proc.pid, sig), with ProcessLookupError ignored.
  killpg(signal) {
    try {
      if (POSIX) process.kill(-this.pid, signal);
      else this.child.kill(signal);
    } catch (error) {
      if (error?.code !== 'ESRCH') throw toPyOSError(error);
    }
  }
}

// bytes.decode("utf-8", "replace") (a BOM is kept) with text mode's universal
// newlines, as communicate() returns text.
export function decodeText(bytes) {
  let text;
  try {
    text = new TextDecoder('utf-8', { fatal: false, ignoreBOM: true }).decode(bytes);
  } catch (error) {
    if (error?.code === 'ERR_STRING_TOO_LONG' || error instanceof RangeError) {
      throw new ContractError('smoke output exceeds the maximum string length Node can hold');
    }
    throw error;
  }
  return text.replaceAll('\r\n', '\n').replaceAll('\r', '\n');
}

// _stop_smoke_process: terminate the entire smoke process group and drain its
// diagnostic pipes. SIGTERM unless the direct child already exited, then up to
// `graceSeconds` for the pipes to close and the child to exit, then SIGKILL
// and an unbounded wait.
export async function stopSmokeProcess(smoke, { graceSeconds = 5.0 } = {}) {
  if (smoke.poll() === null) smoke.killpg('SIGTERM');
  try {
    return await smoke.communicate(graceSeconds);
  } catch (error) {
    if (!(error instanceof TimeoutExpired)) throw error;
    smoke.killpg('SIGKILL');
    return smoke.communicate();
  }
}

export function writeSmokeDiagnostics(diagnosticsDir, label, stdout, stderr) {
  const directory = pyPath(String(diagnosticsDir));
  const sanitizedStderr = sanitizeDiagnosticText(stderr);
  pyWriteText(pyJoinPath(directory, `${label}-stderr.log`), sanitizedStderr);
  pyWriteText(pyJoinPath(directory, `${label}-stdout.json`), sanitizeDiagnosticStdout(stdout));
  return sanitizedStderr;
}

function defaultStderr(text) {
  process.stderr.write(text);
}

// _run_smoke: run one gate, write its sanitized diagnostics, and return its
// strict-JSON result, which must report ok=true.
export async function runSmoke(command, label, diagnosticsDir, {
  timeoutSeconds,
  stderr: writeStderr = defaultStderr,
  wrapper = rssWrapper(),
  graceSeconds = 5.0,
} = {}) {
  if (timeoutSeconds <= 0) throw new ContractError(`${label} timeout must be positive`);
  if (!wrapper) {
    writeStderr('warning: /usr/bin/time is unavailable, so the peak RSS of this smoke is not measured\n');
  }
  const smoke = new SmokeProcess(command, { env: childEnv(), wrapper });
  let removeInterrupt = () => {};
  const interrupted = new Promise((resolve) => {
    if (!POSIX) return;
    const onInterrupt = () => resolve();
    process.on('SIGINT', onInterrupt);
    removeInterrupt = () => process.removeListener('SIGINT', onInterrupt);
  });
  let timedOut = false;
  let stdout;
  let stderrText;
  // Python handles KeyboardInterrupt only around the first communicate(); a
  // later SIGINT gets Node's default handling, as it would escape Python.
  let outcome;
  try {
    outcome = await Promise.race([
      smoke.communicate(timeoutSeconds).then((output) => ({ output }), (error) => ({ error })),
      interrupted.then(() => ({ interrupted: true })),
    ]);
  } finally {
    removeInterrupt();
  }
  if (outcome.interrupted) {
    [stdout, stderrText] = await stopSmokeProcess(smoke, { graceSeconds });
    writeSmokeDiagnostics(diagnosticsDir, label, stdout, stderrText);
    throw new KeyboardInterrupt();
  }
  if (outcome.error) {
    if (!(outcome.error instanceof TimeoutExpired)) throw outcome.error;
    timedOut = true;
    [stdout, stderrText] = await stopSmokeProcess(smoke, { graceSeconds });
  } else {
    [stdout, stderrText] = outcome.output;
  }
  const sanitizedStderr = writeSmokeDiagnostics(diagnosticsDir, label, stdout, stderrText);
  if (timedOut) {
    writeStderr(sanitizedStderr);
    throw new ContractError(`${label} gate timed out after ${pyFormatG(timeoutSeconds)} seconds`);
  }
  if (smoke.returncode !== 0) {
    writeStderr(sanitizedStderr);
    throw new ContractError(`${label} gate failed with exit status ${smoke.returncode}`);
  }
  let payload;
  try {
    payload = pyJsonLoads(stdout, {
      objectPairsHook: rejectDuplicateKeys,
      parseConstant: rejectNonstandardJsonConstant,
    });
  } catch (error) {
    if (error instanceof ContractError || error instanceof JSONDecodeError) {
      const detail = sanitizeDiagnosticText(error.message);
      throw new ContractError(`${label} gate emitted invalid JSON: ${detail}`);
    }
    throw error;
  }
  if (!isDict(payload) || pyGet(payload, 'ok') !== true) throw new ContractError(`${label} gate did not report ok=true`);
  return payload;
}

// --- Gate results --------------------------------------------------------------------

function isMapping(value) {
  return isDict(value) || value instanceof Map;
}

// isinstance(value, int) and not isinstance(value, bool).
function isInt(value) {
  return isPyInt(value);
}

export function modeKey(entry, label) {
  const memoryMode = pyGet(entry, 'memoryMode');
  const runtimeMode = pyGet(entry, 'runtimeMode');
  if (typeof memoryMode !== 'string' || typeof runtimeMode !== 'string') {
    throw new ContractError(`${label} result is missing its mode identifiers`);
  }
  return [memoryMode, runtimeMode];
}

export function timing(entry, key, label) {
  const value = pyGet(entry, key);
  if (!isInt(value) || value < 0) throw new ContractError(`${label} is missing a non-negative ${key} timing`);
  return value;
}

function sortModes(modes) {
  return modes.sort((left, right) => compareCodePoints(left.memory_mode, right.memory_mode)
    || compareCodePoints(left.runtime_mode, right.runtime_mode));
}

export function speechPhase(payload, rss) {
  const results = pyGet(payload, 'modeResults');
  if (!Array.isArray(results)) throw new ContractError('speech gate did not report modeResults');
  const modes = [];
  for (const entry of results) {
    if (!isMapping(entry)) throw new ContractError('speech gate reported an invalid mode result');
    const [memoryMode, runtimeMode] = modeKey(entry, 'speech');
    const label = `speech ${memoryMode}/${runtimeMode}`;
    const timings = pyGet(entry, 'phaseTimingsMs');
    if (!isMapping(timings)) throw new ContractError(`${label} did not report per-phase timings`);
    const coldTranscript = pyGet(entry, 'coldTranscript');
    const warmTranscript = pyGet(entry, 'warmTranscript');
    const cancellationResult = pyGet(entry, 'cancellation');
    const silenceTranscript = pyGet(entry, 'silenceTranscript');
    for (const [field, value] of [
      ['coldTranscript', coldTranscript],
      ['warmTranscript', warmTranscript],
      ['cancellation', cancellationResult],
    ]) {
      if (typeof value !== 'string' || !value) throw new ContractError(`${label} did not report ${field} evidence`);
    }
    if (normalizeTranscript(coldTranscript) !== EXPECTED_SPEECH_TRANSCRIPT) {
      throw new ContractError(`${label} cold transcript does not match expected fixture transcript`);
    }
    if (normalizeTranscript(warmTranscript) !== EXPECTED_SPEECH_TRANSCRIPT) {
      throw new ContractError(`${label} warm transcript does not match expected fixture transcript`);
    }
    parseCancellationResult(cancellationResult, `${label} cancellation`);
    if (silenceTranscript !== '') throw new ContractError(`${label} silence transcript must be empty`);
    const totalMs = timing(entry, 'elapsedMs', label);
    const phaseTimingsMs = {
      cancellation: timing(timings, 'cancellationMs', label),
      cold_transcript: timing(timings, 'coldTranscriptMs', label),
      model_load: timing(timings, 'modelLoadMs', label),
      projector_load: timing(timings, 'projectorLoadMs', label),
      silence: timing(timings, 'silenceMs', label),
      warm_transcript: timing(timings, 'warmTranscriptMs', label),
    };
    modes.push({
      memory_mode: memoryMode,
      runtime_mode: runtimeMode,
      total_ms: totalMs,
      phase_timings_ms: phaseTimingsMs,
      cold_transcript: coldTranscript,
      warm_transcript: warmTranscript,
      cancellation_result: cancellationResult,
      silence_transcript: silenceTranscript,
    });
  }
  sortModes(modes);
  return {
    modes,
    total_ms: pySumInts(modes.map((mode) => mode.total_ms)),
    max_rss_bytes: rss,
  };
}

// isinstance(value, (int, float)) and not bool and math.isfinite(value) and
// value > 0; math.isfinite raises OverflowError for an int past float range.
function isPositiveFiniteNumber(value) {
  let number;
  if (value instanceof PyFloat) number = value.value;
  else if (typeof value === 'number') number = value;
  else if (typeof value === 'bigint') {
    number = Number(value);
    if (!Number.isFinite(number)) throw new PyException('OverflowError', 'int too large to convert to float');
  } else return false;
  return Number.isFinite(number) && number > 0;
}

function floatOf(value) {
  if (value instanceof PyFloat) return value.value;
  return Number(value);
}

function pyIsTrue(value) {
  return value === true;
}

export function ttsPhase(payload, rss, artifactsDir) {
  const results = pyGet(payload, 'modeResults');
  if (!Array.isArray(results)) throw new ContractError('text-to-speech gate did not report modeResults');
  const modes = [];
  for (const entry of results) {
    if (!isMapping(entry)) throw new ContractError('text-to-speech gate reported an invalid mode result');
    const [memoryMode, runtimeMode] = modeKey(entry, 'text-to-speech');
    const label = `text-to-speech ${memoryMode}/${runtimeMode}`;
    const audioName = pyGet(entry, 'audioArtifact');
    if (typeof audioName !== 'string' || !audioName) {
      throw new ContractError(`${label} did not persist a generated WAV artifact`);
    }
    if (pyPathName(audioName) !== audioName || !audioName.endsWith('.wav')) {
      throw new ContractError(`${label} WAV artifact name is unsafe: ${pyRepr(audioName)}`);
    }
    const artifactsRoot = pyResolve(pyPath(String(artifactsDir)));
    const wavPath = pyJoinPath(artifactsRoot, audioName);
    if (pyIsSymlink(wavPath) || !pyIsFile(wavPath)) {
      throw new ContractError(`${label} WAV artifact is missing: ${audioName}`);
    }
    const framesGenerated = pyGet(entry, 'framesGenerated');
    if (!isInt(framesGenerated) || framesGenerated <= 0) {
      throw new ContractError(`${label} framesGenerated must be a positive integer`);
    }
    const truncated = pyGet(entry, 'truncated');
    if (typeof truncated !== 'boolean' || truncated !== false) {
      throw new ContractError(`${label} truncated must be false, got ${pyRepr(truncated)}`);
    }
    if (!pyIsTrue(pyGet(entry, 'cancellationTested'))) throw new ContractError(`${label} cancellationTested must be true`);
    if (!pyIsTrue(pyGet(entry, 'preAbortedTested'))) throw new ContractError(`${label} preAbortedTested must be true`);
    const reuseSampleCount = pyGet(entry, 'reuseSampleCount');
    requirePositiveInt(reuseSampleCount, `${label}.reuseSampleCount`);
    if (!pyIsTrue(pyGet(entry, 'unloadTested'))) throw new ContractError(`${label} unloadTested must be true`);
    const peak = pyGet(entry, 'peak');
    const rms = pyGet(entry, 'rms');
    for (const [field, value] of [['peak', peak], ['rms', rms]]) {
      if (!isPositiveFiniteNumber(value)) throw new ContractError(`${label} ${field} must be a positive finite number`);
    }
    const { peak: measuredPeak, rms: measuredRms, ...wavIdentity } = readWavIdentity(wavPath);
    const quantizationTolerance = (1 / 32768.0) + 1e-9;
    if (Math.abs(measuredPeak - floatOf(peak)) > quantizationTolerance
      || Math.abs(measuredRms - floatOf(rms)) > quantizationTolerance) {
      throw new ContractError(`${label} reported waveform evidence does not match its WAV artifact`);
    }
    if (measuredPeak <= 0.001 || measuredRms <= 0.0001) throw new ContractError(`${label} generated WAV is effectively silent`);
    const phaseTimingsMs = {
      model_load: timing(entry, 'modelLoadMs', label),
      projector_load: timing(entry, 'projectorLoadMs', label),
      synthesis: timing(entry, 'synthesisMs', label),
    };
    const totalMs = timing(entry, 'totalElapsedMs', label);
    modes.push({
      cancellation_tested: true,
      frames_generated: framesGenerated,
      memory_mode: memoryMode,
      peak: new PyFloat(measuredPeak),
      phase_timings_ms: phaseTimingsMs,
      pre_aborted_tested: true,
      reuse_sample_count: reuseSampleCount,
      rms: new PyFloat(measuredRms),
      runtime_mode: runtimeMode,
      total_ms: totalMs,
      truncated: false,
      unload_tested: true,
      wav: wavIdentity,
    });
  }
  sortModes(modes);
  return {
    modes,
    total_ms: pySumInts(modes.map((mode) => mode.total_ms)),
    max_rss_bytes: rss,
  };
}

// --- The qualify command ---------------------------------------------------------------

// Resolve the Node.js that runs the browser smokes, failing closed. The
// qualification workflow installs Node.js 24 and the candidate's locked npm
// dependencies before qualify runs; a missing node never falls back to another
// interpreter.
// re.fullmatch(r"v(\d+)\.(\d+)\.\d+", version) with re's Unicode \d.
const NODE_VERSION_RE = new RegExp(`^v(${PY_DECIMAL}+)\\.(${PY_DECIMAL}+)\\.${PY_DECIMAL}+$`, 'u');

export function nodeExecutable({ which = pyWhich } = {}) {
  const found = which('node');
  if (found === null || found === undefined) {
    throw new ContractError(
      'node is required to run the qualification gates; install Node.js '
      + 'and run npm ci --ignore-scripts in the candidate source',
    );
  }
  const node = pyResolve(found);
  // The smokes start main() through import.meta.main (Node.js 22.18+); an
  // older node would exit 0 without running a gate.
  const result = spawnSync(node, ['--version'], { timeout: 30000, killSignal: 'SIGKILL', stdio: ['inherit', 'pipe', 'pipe'] });
  if (result.error) {
    if (result.error.code === 'ETIMEDOUT') throw new TimeoutExpired([node, '--version'], 30);
    throw toPyOSError(result.error, node);
  }
  // text=True decodes both streams strictly, stdout first.
  const version = pyStrip(pyUniversalNewlines(pyDecodeUtf8(result.stdout)));
  pyDecodeUtf8(result.stderr);
  const match = NODE_VERSION_RE.exec(version);
  const tooOld = match === null
    || (() => {
      const major = BigInt(pyIntFromString(match[1]));
      const minor = BigInt(pyIntFromString(match[2]));
      return major < 22n || (major === 22n && minor < 18n);
    })();
  if (tooOld) {
    throw new ContractError(`node ${version || '(unknown version)'} is too old; the qualification gates need Node.js 22.18 or newer`);
  }
  return node;
}

export function requireInputFile(inputPath, label) {
  const resolved = pyResolve(pyExpanduser(pyPath(String(inputPath))));
  if (!pyIsFile(resolved)) throw new ContractError(`${label} does not exist: ${resolved}`);
  return resolved;
}

// Path.mkdir(parents=parents, exist_ok=True).
function pyMkdirParents(text, parents = true) {
  try {
    fs.mkdirSync(text, { mode: 0o777 });
  } catch (error) {
    if (error?.code === 'ENOENT') {
      const parent = path.posix.dirname(text);
      if (!parents || parent === text) throw toPyOSError(error, text);
      pyMkdirParents(parent);
      pyMkdirParents(text, false);
      return;
    }
    let isDirectory = false;
    try {
      isDirectory = fs.statSync(text).isDirectory();
    } catch {
      // not a directory
    }
    if (!isDirectory) throw toPyOSError(error, text);
  }
}

// tempfile.mkdtemp(prefix=...) in the system temp root.
function makeTempDirectory(prefix) {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

function defaultStdout(text) {
  process.stdout.write(text);
}

// qualify_cmd, called by qualification.mjs's CLI as qualifyCmd(args, write):
// the canonical attestation goes to write() (sys.stdout), progress straight to
// stderr as Python prints it. `deps` replaces what the Python tests patch:
// every collaborator by its camelCase name, plus `scriptsDir` and a `stderr`
// writer.
export async function qualifyCmd(args, write = defaultStdout, deps = {}) {
  const use = {
    nodeExecutable,
    qualificationEnvironment,
    qualificationRunIdentity,
    fetchCandidate,
    loadCandidate,
    requireHarnessMatchesBridgeSource,
    runSmoke,
    speechPhase,
    ttsPhase,
    maxRssBytes,
    buildAttestation,
    verifyAttestation,
    canonicalJson,
    stderr: defaultStderr,
    ...deps,
  };
  const scriptsDir = deps.scriptsDir ?? SCRIPTS_DIR;
  const print = (text) => use.stderr(`${text}\n`);
  const node = await use.nodeExecutable();
  const environment = await use.qualificationEnvironment();
  const qualificationIdentity = await use.qualificationRunIdentity();
  if (args.ttsMaxFrames <= 0) throw new ContractError('tts_max_frames must be positive');
  if (args.speechTimeoutSeconds <= 0 || args.ttsTimeoutSeconds <= 0) {
    throw new ContractError('qualification gate timeouts must be positive');
  }
  const speechModel = requireInputFile(args.speechModelPath, 'Qwen3-ASR model');
  const speechMmproj = requireInputFile(args.speechMmprojPath, 'Qwen3-ASR projector');
  const speechAudio = requireInputFile(args.speechAudioPath, 'speech WAV fixture');
  const ttsModel = requireInputFile(args.ttsModelPath, 'Qwen3-TTS model');
  const ttsMmproj = requireInputFile(args.ttsMmprojPath, 'Qwen3-TTS projector');

  // Every scratch path lives in the system temp root so no download, artifact,
  // or diagnostic is ever written inside the repository working tree.
  const root = makeTempDirectory('llama-web-bridge-qualification-');
  let canonical;
  try {
    const candidateDir = pyJoinPath(root, 'candidate');
    // Diagnostics outlive the scratch root so a failed gate stays
    // inspectable, and they live outside the repository working tree.
    const diagnosticsDir = args.diagnosticsDir !== null && args.diagnosticsDir !== undefined
      ? pyResolve(pyExpanduser(pyPath(String(args.diagnosticsDir))))
      : makeTempDirectory('llama-web-bridge-qualification-diag-');
    pyMkdirParents(diagnosticsDir);
    print(`Sanitized diagnostics directory ${diagnosticsDir}`);
    const speechArtifacts = pyJoinPath(diagnosticsDir, 'speech-to-text');
    const ttsArtifacts = pyJoinPath(diagnosticsDir, 'text-to-speech');

    print(`Downloading candidate artifact from run ${args.candidateRunId}`);
    const [candidateArtifactId, candidateRunAttempt] = await use.fetchCandidate(args.candidateRunId, candidateDir);
    const [manifest, fingerprint] = await use.loadCandidate(candidateDir);
    const harnessDigest = await use.requireHarnessMatchesBridgeSource(
      scriptsDir,
      requireStr(manifest, 'bridge_commit', 'candidate manifest'),
    );
    print(`Candidate fingerprint ${fingerprint}`);

    print('Running Qwen3-ASR wasm32+wasm64 direct+worker gate');
    const speechPayload = await use.runSmoke(
      [
        node,
        pyJoinPath(scriptsDir, SPEECH_SMOKE),
        '--dist-dir', candidateDir,
        '--model-path', speechModel,
        '--model-sha256', SPEECH_MODEL_SHA256,
        '--mmproj-path', speechMmproj,
        '--mmproj-sha256', SPEECH_MMPROJ_SHA256,
        '--audio-path', speechAudio,
        '--audio-sha256', SPEECH_AUDIO_SHA256,
        '--memory-mode', 'all',
        '--timeout-ms', String(pyIntMultiply(args.speechTimeoutSeconds, 1000)),
        '--artifacts-dir', speechArtifacts,
      ],
      'speech-to-text',
      diagnosticsDir,
      { timeoutSeconds: Number(pyIntAdd(args.speechTimeoutSeconds, 60)), stderr: use.stderr },
    );
    const speech = await use.speechPhase(speechPayload, use.maxRssBytes());

    print('Running Qwen3-TTS wasm64 direct+worker gate');
    const ttsPayload = await use.runSmoke(
      [
        node,
        pyJoinPath(scriptsDir, TTS_SMOKE),
        '--dist-dir', candidateDir,
        '--model-path', ttsModel,
        '--model-sha256', TTS_MODEL_SHA256,
        '--mmproj-path', ttsMmproj,
        '--mmproj-sha256', TTS_MMPROJ_SHA256,
        '--memory-mode', 'wasm64',
        '--runtime-mode', 'all',
        '--max-frames', String(args.ttsMaxFrames),
        '--timeout-ms', String(pyIntMultiply(args.ttsTimeoutSeconds, 1000)),
        '--artifacts-dir', ttsArtifacts,
      ],
      'text-to-speech',
      diagnosticsDir,
      { timeoutSeconds: Number(pyIntAdd(args.ttsTimeoutSeconds, 60)), stderr: use.stderr },
    );
    const tts = await use.ttsPhase(ttsPayload, use.maxRssBytes(), ttsArtifacts);

    const attestation = await use.buildAttestation({
      manifest,
      candidateFingerprint: fingerprint,
      candidateRunId: args.candidateRunId,
      candidateArtifactId,
      candidateRunAttempt,
      qualificationRunId: qualificationIdentity.qualification_run_id,
      qualificationRunAttempt: qualificationIdentity.qualification_run_attempt,
      qualificationSourceSha: qualificationIdentity.qualification_source_sha,
      harnessDigest,
      environment,
      speechPhase: speech,
      ttsPhase: tts,
    });
    // Re-verify what was just built against the exact candidate so a harness
    // bug can never emit an attestation publication would later reject.
    await use.verifyAttestation({
      attestation,
      candidateDir,
      candidateArtifactId,
      candidateRunAttempt,
      qualificationRunId: qualificationIdentity.qualification_run_id,
      qualificationRunAttempt: qualificationIdentity.qualification_run_attempt,
      qualificationSourceSha: qualificationIdentity.qualification_source_sha,
    });
    canonical = await use.canonicalJson(attestation);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }

  if (Buffer.byteLength(canonical, 'utf8') > MAX_ATTESTATION_BYTES) {
    throw new ContractError(`attestation exceeds the ${MAX_ATTESTATION_BYTES}-byte artifact bound`);
  }
  const outputAttestation = pyPath(String(args.outputAttestation));
  pyWriteText(outputAttestation, canonical);
  write(canonical);
  print(`Canonical attestation written to ${outputAttestation}`);
  return 0;
}

// Exposed for the tests only.
export const internals = {
  resetPeakChildRss() {
    peakChildRssBytes = 0;
  },
  setRssWrapper(wrapper) {
    detectedWrapper = wrapper;
  },
  GNU_TIME,
  BSD_TIME,
  pySumInts,
  requireExecutable,
};
