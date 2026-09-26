// Tests of scripts/release/cli.mjs: the parser without subcommands, option
// defaults, and main()'s returned exit status. Expected texts are argparse's
// (Python 3.12) for the same parser, with this module's usage line.

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import process from 'node:process';
import { test } from 'node:test';

import {
  ArgparseExit, SystemExitError, parseArguments, parseCommandLine, runPythonStyleMain, runPythonStyleMainAsync,
} from '../../scripts/release/cli.mjs';
import { PyException } from '../../scripts/release/json.mjs';

const OPTIONS = [
  { flag: '--out-dir', required: true, type: 'path' },
  { flag: '--n', required: true, type: 'int' },
  { flag: '--repo', required: false, type: 'str', default: 'ggml-org/llama.cpp' },
];
const USAGE = 'usage: gen.py [-h] --out-dir OUT_DIR --n N [--repo REPO]\n';

function parse(argv) {
  try {
    return parseArguments(argv, { prog: 'gen.py', options: OPTIONS });
  } catch (error) {
    if (error instanceof ArgparseExit) return { status: error.status, stdout: error.stdout, stderr: error.stderr };
    throw error;
  }
}

test('parseArguments accepts, converts and defaults like argparse', () => {
  assert.deepEqual(parse(['--out-dir=a/', '--n=+1_0']), { outDir: 'a', n: 10, repo: 'ggml-org/llama.cpp' });
  assert.deepEqual(parse(['--n', '-3', '--out-dir', '', '--repo', 'x/y', '--n', '4']), { outDir: '.', n: 4, repo: 'x/y' });
});

test('parseArguments rejects with argparse error lines', () => {
  const cases = [
    [[], 'the following arguments are required: --out-dir, --n'],
    [['--out-dir', 'x'], 'the following arguments are required: --n'],
    [['--out-dir', 'x', '--n', '1', 'extra'], 'unrecognized arguments: extra'],
    [['--out-dir', 'x', '--n', 'z'], "argument --n: invalid int value: 'z'"],
    [['--out-dir'], 'argument --out-dir: expected one argument'],
    [['--bogus', '--out-dir', 'x', '--n', '1'], 'unrecognized arguments: --bogus'],
    [['--out-dir', 'x', '--', '--n', '1'], "argument '--' is not supported"],
  ];
  for (const [argv, message] of cases) {
    assert.deepEqual(parse(argv), { status: 2, stdout: '', stderr: `${USAGE}gen.py: error: ${message}\n` }, argv.join(' '));
  }
  const help = parse(['--out-dir', 'x', '-h']);
  assert.equal(help.status, 0);
  assert.ok(help.stdout.startsWith(`${USAGE}\noptions:\n`));
});

test('parseCommandLine gives an option that was not given its default', () => {
  const commands = { run: { options: [{ flag: '--mode', required: false, type: 'str', default: 'fast' }, { flag: '--other', required: false }] } };
  assert.deepEqual(parseCommandLine(['run'], { prog: 'p', commands }), { command: 'run', args: { mode: 'fast', other: null } });
});

test('runPythonStyleMain exits with the status main() returns', () => {
  assert.deepEqual(runPythonStyleMain((argv, write) => { write('false\n'); return 1; }, []), { stdout: 'false\n', stderr: '', status: 1 });
  assert.deepEqual(runPythonStyleMain(() => undefined, []), { stdout: '', stderr: '', status: 0 });
});

test('runPythonStyleMainAsync settles a main() that returns a Promise or a status', async () => {
  assert.deepEqual(await runPythonStyleMainAsync(async (argv, write) => { write('x\n'); return 3; }, []), { stdout: 'x\n', stderr: '', status: 3 });
  assert.deepEqual(await runPythonStyleMainAsync(() => 0, []), { stdout: '', stderr: '', status: 0 });
  assert.deepEqual(await runPythonStyleMainAsync(async () => { throw new SystemExitError('error: boom'); }, []), {
    stdout: '', stderr: 'error: boom\n', status: 1,
  });
  assert.deepEqual(await runPythonStyleMainAsync(() => Promise.reject(new PyException('TypeError', 'bad')), []), {
    stdout: '', stderr: 'Traceback (most recent call last):\nTypeError: bad\n', status: 1,
  });
  const usage = await runPythonStyleMainAsync((argv) => parseArguments(argv, { prog: 'gen.py', options: OPTIONS }), []);
  assert.equal(usage.status, 2);
  await assert.rejects(runPythonStyleMainAsync(async () => { throw new RangeError('not Python'); }, []), RangeError);
});

test('an uncaught KeyboardInterrupt prints its traceback and ends the process with SIGINT, as Python 3.8+ does', () => {
  const cli = new URL('../../scripts/release/cli.mjs', import.meta.url).href;
  const json = new URL('../../scripts/release/json.mjs', import.meta.url).href;
  for (const runner of ['runCli', 'runCliAsync']) {
    const source = `import { ${runner} } from '${cli}'; import { PyException } from '${json}';`
      + `process.on('SIGINT', () => {});`
      + `${runner}((argv, write) => { write('partial\\n'); throw new PyException('KeyboardInterrupt', ''); }, []);`;
    const result = spawnSync(process.execPath, ['--input-type=module', '-e', source], { encoding: 'utf8' });
    assert.equal(result.signal, 'SIGINT', runner);
    assert.equal(result.stdout, 'partial\n');
    assert.equal(result.stderr, 'Traceback (most recent call last):\nKeyboardInterrupt\n');
  }
  assert.deepEqual(runPythonStyleMain(() => { throw new PyException('KeyboardInterrupt', ''); }, []), {
    stdout: '', stderr: 'Traceback (most recent call last):\nKeyboardInterrupt\n', status: 130, signal: 'SIGINT',
  });
});

test('int options take int()\'s Unicode 15.0 digits, and a too-long literal is an invalid int value', () => {
  assert.deepEqual(parse(['--out-dir', 'x', '--n', '-\u{663}']), { outDir: 'x', n: -3, repo: 'ggml-org/llama.cpp' });
  assert.deepEqual(parse(['--out-dir', 'x', '--n', '\u{11066}']), { outDir: 'x', n: 0, repo: 'ggml-org/llama.cpp' });
  for (const [token, rendered] of [
    // U+10D40 (GARAY DIGIT ZERO) is a digit only since Unicode 16.
    ['\u{10d40}', "'\\U00010d40'"],
    ['1'.repeat(4301), `'${'1'.repeat(4301)}'`],
  ]) {
    assert.deepEqual(parse(['--out-dir', 'x', '--n', token]), {
      status: 2, stdout: '', stderr: `${USAGE}gen.py: error: argument --n: invalid int value: ${rendered}\n`,
    });
  }
  // A '-' token is a value only when it is a negative number in re's \d.
  assert.equal(parse(['--out-dir', 'x', '--n', '-\u{10d40}']).stderr, `${USAGE}gen.py: error: argument --n: expected one argument\n`);
});
