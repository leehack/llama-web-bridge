// A subcommand command-line parser with argparse's accept/reject behaviour for
// the shapes the release CLIs use, plus Python's SystemExit reporting.
//
// Each subcommand declares options that take exactly one value and required
// positionals, like `add_argument("--flag", required=..., type=...)` under
// `add_subparsers(dest="command", required=True)`. Errors carry argparse's
// message and exit status 2 (Python 3.12.3, the ubuntu-24.04 runner). This
// parser is deliberately stricter than argparse, and never looser:
//
// - no abbreviated long options (argparse's allow_abbrev);
// - no `--` separator;
// - a value that starts with '-' must be '-' or a negative number (argparse
//   also takes one that contains a space);
// - usage and help text are this module's own, not argparse's wrapped layout.

import path from 'node:path';
import process from 'node:process';

import {
  PY_DECIMAL, PyException, isPyException, pyIntFromString, pyPath, pyRepr,
} from './json.mjs';
import { ContractError } from './errors.mjs';

// argparse ended the process: `status` 0 for --help, 2 for a usage error.
export class ArgparseExit extends Error {
  constructor(status, { stdout = '', stderr = '' } = {}) {
    super(stderr || stdout);
    this.name = 'ArgparseExit';
    this.status = status;
    this.stdout = stdout;
    this.stderr = stderr;
  }
}

// argparse's _negative_number_matcher, '^-\d+$|^-\d*\.\d+$', with re's
// Unicode \d.
const NEGATIVE_NUMBER = new RegExp(`^-${PY_DECIMAL}+$|^-${PY_DECIMAL}*\\.${PY_DECIMAL}+$`, 'u');

// argparse's classification of a token as a value ('A') rather than an option.
function isValueToken(token) {
  return !token.startsWith('-') || token === '-' || NEGATIVE_NUMBER.test(token);
}

function metavar(option) {
  return option.metavar ?? option.flag.replace(/^--/, '').replaceAll('-', '_').toUpperCase();
}

// An option's text in usage and help: `--flag` for a store_true flag, else
// `--flag METAVAR`.
function invocation(option) {
  return option.type === 'flag' ? option.flag : `${option.flag} ${metavar(option)}`;
}

function commandUsage(prog, name, spec) {
  const parts = ['[-h]'];
  for (const option of spec.options ?? []) {
    const text = `${option.flag} ${metavar(option)}`;
    parts.push(option.required ? text : `[${text}]`);
  }
  for (const positional of spec.positionals ?? []) parts.push(positional);
  return `usage: ${prog} ${name} ${parts.join(' ')}\n`;
}

function topUsage(prog, commands) {
  return `usage: ${prog} [-h] {${Object.keys(commands).join(',')}} ...\n`;
}

function commandHelp(prog, name, spec) {
  let text = `${commandUsage(prog, name, spec)}\n`;
  if (spec.positionals?.length) {
    text += `positional arguments:\n${spec.positionals.map((positional) => `  ${positional}\n`).join('')}\n`;
  }
  text += 'options:\n  -h, --help            show this help message and exit\n';
  for (const option of spec.options ?? []) text += `  ${option.flag} ${metavar(option)}\n`;
  return text;
}

function topHelp(prog, commands) {
  return `${topUsage(prog, commands)}\npositional arguments:\n  {${Object.keys(commands).join(',')}}\n\n`
    + 'options:\n  -h, --help            show this help message and exit\n';
}

// The dest argparse derives: `--release-tag` is release_tag, here releaseTag.
function destination(name) {
  return name.replace(/^--/, '').replace(/-([a-z0-9])/g, (_, char) => char.toUpperCase());
}

function convert(option, token, fail) {
  if (option.type === 'int') {
    try {
      return pyIntFromString(token);
    } catch (error) {
      if (!isPyException(error, 'ValueError')) throw error;
      return fail(`argument ${option.flag}: invalid int value: ${pyRepr(token)}`);
    }
  }
  if (option.type === 'path') return pyPath(token);
  return token;
}

// Parse argv (without the node and script paths) against `commands`, a map of
// subcommand name to { options: [{ flag, required, type: 'str'|'int'|'path',
// default }], positionals: [name] }. Returns { command, args } with camelCase
// keys, and an option that was not given holds its default (null if none,
// used as is, like argparse's non-string defaults).
export function parseCommandLine(argv, { prog, commands }) {
  const usageError = (usage, errorProg, message) => {
    throw new ArgparseExit(2, { stderr: `${usage}${errorProg}: error: ${message}\n` });
  };
  const topError = (message) => usageError(topUsage(prog, commands), prog, message);

  const extras = [];
  let index = 0;
  for (; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '-h' || token === '--help') throw new ArgparseExit(0, { stdout: topHelp(prog, commands) });
    if (token === '--') topError("argument '--' is not supported");
    if (isValueToken(token)) break;
    extras.push(token);
  }
  if (index >= argv.length) topError('the following arguments are required: command');
  const command = argv[index];
  const spec = commands[command];
  if (!Object.hasOwn(commands, command)) {
    topError(`argument command: invalid choice: ${pyRepr(command)} (choose from ${Object.keys(commands).map(pyRepr).join(', ')})`);
  }

  const subProg = `${prog} ${command}`;
  const fail = (message) => usageError(commandUsage(prog, command, spec), subProg, message);
  const options = new Map((spec.options ?? []).map((option) => [option.flag, option]));
  const args = {};
  for (const option of options.values()) args[destination(option.flag)] = option.default ?? null;
  const seen = new Set();
  const positionals = [];
  for (index += 1; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--') fail("argument '--' is not supported");
    if (token === '-h' || token === '--help') throw new ArgparseExit(0, { stdout: commandHelp(prog, command, spec) });
    if (isValueToken(token)) {
      if (positionals.length < (spec.positionals ?? []).length) positionals.push(token);
      else extras.push(token);
      continue;
    }
    const equals = token.indexOf('=');
    const flag = equals >= 0 ? token.slice(0, equals) : token;
    const option = options.get(flag);
    if (!option) {
      extras.push(token);
      continue;
    }
    let value;
    if (equals >= 0) {
      value = token.slice(equals + 1);
    } else {
      const next = argv[index + 1];
      if (next === undefined || next === '--' || !isValueToken(next)) fail(`argument ${option.flag}: expected one argument`);
      value = next;
      index += 1;
    }
    args[destination(option.flag)] = convert(option, value, fail);
    seen.add(option.flag);
  }

  const missing = [];
  for (const option of options.values()) if (option.required && !seen.has(option.flag)) missing.push(option.flag);
  (spec.positionals ?? []).forEach((name, position) => {
    if (position < positionals.length) args[destination(name)] = positionals[position];
    else missing.push(name);
  });
  if (missing.length > 0) fail(`the following arguments are required: ${missing.join(', ')}`);
  if (extras.length > 0) topError(`unrecognized arguments: ${extras.join(' ')}`);
  return { command, args };
}

// Run a CLI main(argv, write) the way `raise SystemExit(main())` runs in
// Python and return its stdout/stderr text and exit status; main() prints
// through write(text) and may return its integer exit status. A SystemExitError (main()'s
// `raise SystemExit(f"error: {error}")`) prints its message and exits 1; any
// other PyException escapes as Python's traceback would (its last line, exit
// 1); argparse exits print their own text. A KeyboardInterrupt prints its
// traceback and ends the process with SIGINT, as Python 3.8+ does
// (`signal: 'SIGINT'` and status 130, the status a shell reports for it).
export function runPythonStyleMain(main, argv = process.argv.slice(2)) {
  const out = { stdout: '', stderr: '', status: 0 };
  try {
    const status = main(argv, (text) => { out.stdout += text; });
    if (Number.isInteger(status)) out.status = status;
  } catch (error) {
    recordExit(out, error);
  }
  return out;
}

// How `error` ends the process: its output and exit status in `out`, or
// rethrown when Python would not have raised it.
function recordExit(out, error) {
  if (error instanceof ArgparseExit) {
    out.stdout += error.stdout;
    out.stderr += error.stderr;
    out.status = error.status;
  } else if (error instanceof SystemExitError) {
    out.stderr += `${error.message}\n`;
    out.status = 1;
  } else if (error instanceof PyException || error instanceof ContractError) {
    const line = error instanceof PyException ? error.tracebackLine : `__main__.ContractError: ${error.message}`;
    out.stderr += `Traceback (most recent call last):\n${line}\n`;
    out.status = 1;
    if (isPyException(error, 'KeyboardInterrupt')) {
      out.status = 130;
      out.signal = 'SIGINT';
    }
  } else {
    throw error;
  }
}

// runPythonStyleMain for a main() that may also return a Promise of its exit
// status (or reject), for a subcommand that awaits child processes.
export async function runPythonStyleMainAsync(main, argv = process.argv.slice(2)) {
  const out = { stdout: '', stderr: '', status: 0 };
  try {
    const status = await main(argv, (text) => { out.stdout += text; });
    if (Number.isInteger(status)) out.status = status;
  } catch (error) {
    recordExit(out, error);
  }
  return out;
}

// raise SystemExit("message"): the message on stderr and exit status 1.
export class SystemExitError extends Error {
  constructor(message) {
    super(message);
    this.name = 'SystemExitError';
  }
}

// The program name argparse shows: the script's base name.
export function progName(scriptUrl) {
  return path.basename(new URL(scriptUrl).pathname);
}

// Hand a run's output and exit status to the process. An uncaught
// KeyboardInterrupt ends it as CPython's exit_sigint() does: once the output
// is flushed, SIGINT with its default action, so the parent sees a death by
// SIGINT; status 130 stands if the signal does not end the process.
function exitProcess({ stdout, stderr, status, signal }) {
  if (stdout) process.stdout.write(stdout);
  if (stderr) process.stderr.write(stderr);
  process.exitCode = status;
  if (signal) {
    process.stdout.write('', () => process.stderr.write('', () => {
      process.removeAllListeners(signal);
      process.kill(process.pid, signal);
    }));
  }
}

// The CLI entry point: run main() and hand its output and status to the process.
export function runCli(main, argv = process.argv.slice(2)) {
  exitProcess(runPythonStyleMain(main, argv));
}

// runCli for a main() that may return a Promise (runPythonStyleMainAsync).
export async function runCliAsync(main, argv = process.argv.slice(2)) {
  exitProcess(await runPythonStyleMainAsync(main, argv));
}

function flatUsage(prog, options) {
  const parts = ['[-h]'];
  for (const option of options) {
    const text = invocation(option);
    parts.push(option.required ? text : `[${text}]`);
  }
  return `usage: ${prog} ${parts.join(' ')}\n`;
}

function flatHelp(prog, options) {
  let text = `${flatUsage(prog, options)}\noptions:\n  -h, --help            show this help message and exit\n`;
  for (const option of options) text += `  ${invocation(option)}\n`;
  return text;
}

// Parse argv (without the node and script paths) for a parser without
// subcommands, like `ArgumentParser()` with only `add_argument("--flag", ...)`
// options, under the same policy as parseCommandLine. `options` is
// [{ flag, required, type: 'str'|'int'|'path'|'flag', default, metavar }]; a
// 'flag' is `action="store_true"`: it takes no value and is true when given,
// false otherwise. Returns the args with camelCase keys; an option that was
// not given holds its default (null if none).
export function parseArguments(argv, { prog, options = [] }) {
  const usage = flatUsage(prog, options);
  const fail = (message) => {
    throw new ArgparseExit(2, { stderr: `${usage}${prog}: error: ${message}\n` });
  };
  const byFlag = new Map(options.map((option) => [option.flag, option]));
  const args = {};
  for (const option of options) args[destination(option.flag)] = option.type === 'flag' ? false : option.default ?? null;
  const seen = new Set();
  const extras = [];
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--') fail("argument '--' is not supported");
    if (token === '-h' || token === '--help') throw new ArgparseExit(0, { stdout: flatHelp(prog, options) });
    if (isValueToken(token)) {
      extras.push(token);
      continue;
    }
    const equals = token.indexOf('=');
    const flag = equals >= 0 ? token.slice(0, equals) : token;
    const option = byFlag.get(flag);
    if (!option) {
      extras.push(token);
      continue;
    }
    seen.add(option.flag);
    if (option.type === 'flag') {
      if (equals >= 0) fail(`argument ${option.flag}: ignored explicit argument ${pyRepr(token.slice(equals + 1))}`);
      args[destination(option.flag)] = true;
      continue;
    }
    let value;
    if (equals >= 0) {
      value = token.slice(equals + 1);
    } else {
      const next = argv[index + 1];
      if (next === undefined || next === '--' || !isValueToken(next)) fail(`argument ${option.flag}: expected one argument`);
      value = next;
      index += 1;
    }
    args[destination(option.flag)] = convert(option, value, fail);
  }
  const missing = options.filter((option) => option.required && !seen.has(option.flag)).map((option) => option.flag);
  if (missing.length > 0) fail(`the following arguments are required: ${missing.join(', ')}`);
  if (extras.length > 0) fail(`unrecognized arguments: ${extras.join(' ')}`);
  return args;
}
