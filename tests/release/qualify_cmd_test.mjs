// Tests of qualifyCmd and nodeExecutable in scripts/release/qualify.mjs, the
// ports of release_qualification.py's qualify_cmd, node_executable and
// _require_input_file, plus the Python path and format helpers they use. The
// first block is one test per Python test of the same name; where Python
// patches module functions with mock, these pass qualifyCmd's `deps` and
// nodeExecutable's `which`.

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { afterEach, beforeEach, test } from 'node:test';

import { ContractError } from '../../scripts/release/errors.mjs';
import { isPyException } from '../../scripts/release/json.mjs';
import {
  QUALIFICATION_SMOKES,
  SPEECH_AUDIO_SHA256,
  SPEECH_MMPROJ_SHA256,
  SPEECH_MODEL_SHA256,
  TTS_MMPROJ_SHA256,
  TTS_MODEL_SHA256,
} from '../../scripts/release/qualification.mjs';
import {
  SCRIPTS_DIR,
  nodeExecutable,
  pyFormatG,
  pyPathName,
  pyWhich,
  qualifyCmd,
  requireInputFile,
} from '../../scripts/release/qualify.mjs';
import { REPO_SCRIPTS_DIR, makeTmp } from './qualify_fixtures.mjs';

const BRIDGE_SHA = '565c8396597ea7c0fb4e8d5d966da8d884b156d8';
const CANDIDATE_RUN_ID = '32919086955';

let tmp;
beforeEach(() => {
  tmp = makeTmp('qualify-cmd-test-');
});
afterEach(() => {
  fs.rmSync(tmp, { recursive: true, force: true });
});

async function assertRejects(promise, check) {
  let caught;
  try {
    await promise;
  } catch (error) {
    caught = error;
  }
  assert.ok(caught, 'expected a rejection');
  check(caught);
  return caught;
}

// The args qualification.mjs's CLI parses for `qualify`, camelCase, paths as
// str(Path(...)).
function qualifyArgs(inputs, overrides = {}) {
  return {
    candidateRunId: CANDIDATE_RUN_ID,
    speechModelPath: inputs.sm,
    speechMmprojPath: inputs.sp,
    speechAudioPath: inputs.sa,
    ttsModelPath: inputs.tm,
    ttsMmprojPath: inputs.tp,
    ttsMaxFrames: 24,
    speechTimeoutSeconds: 7,
    ttsTimeoutSeconds: 8,
    diagnosticsDir: path.join(tmp, 'diag'),
    outputAttestation: path.join(tmp, 'attestation.json'),
    ...overrides,
  };
}

function writeInputs() {
  const inputs = {};
  for (const name of ['sm', 'sp', 'sa', 'tm', 'tp']) {
    inputs[name] = path.join(tmp, `${name}.bin`);
    fs.writeFileSync(inputs[name], 'x');
  }
  return inputs;
}

// --- the Python tests ---------------------------------------------------------

test('test_node_executable_fails_closed_without_node', () => {
  assert.throws(() => nodeExecutable({ which: () => null }), (error) => error instanceof ContractError
    && error.message.includes('node is required'));
  const fake = path.join(tmp, 'bin', 'node');
  fs.mkdirSync(path.dirname(fake));
  const link = path.join(tmp, 'node-link');
  fs.symlinkSync(fake, link);
  for (const [version, accepted] of [['v24.2.0', true], ['v25.0.0', true], ['v22.18.0', true], ['v22.17.1', false], ['v23.11.0', false], ['v24.1.0', false], ['v20.19.0', false], ['', false]]) {
    fs.writeFileSync(fake, `#!/bin/sh\necho '${version}'\n`);
    fs.chmodSync(fake, 0o755);
    if (accepted) {
      assert.equal(nodeExecutable({ which: () => link }), fs.realpathSync(fake));
    } else {
      assert.throws(() => nodeExecutable({ which: () => link }), (error) => error instanceof ContractError
        && error.message.includes('too old'));
    }
  }
});

test('node --version digits follow re\'s Unicode 15.0 \\d and int()\'s 4300-digit limit', () => {
  const fake = path.join(tmp, 'bin', 'node');
  fs.mkdirSync(path.dirname(fake));
  const version = (text) => {
    fs.writeFileSync(path.join(tmp, 'version'), text);
    fs.writeFileSync(fake, `#!/bin/sh\ncat '${path.join(tmp, 'version')}'\n`);
    fs.chmodSync(fake, 0o755);
    return () => nodeExecutable({ which: () => fake });
  };
  // Python 3.12: \d takes U+0662 but not U+10D42 (a digit since Unicode 16).
  assert.equal(version('v\u{662}\u{664}.2.0\n')(), fs.realpathSync(fake));
  assert.throws(version('v\u{10d42}\u{664}.2.0\n'), (error) => error instanceof ContractError && error.message.includes('too old'));
  // int() of more than 4300 digits raises ValueError, uncaught as in Python.
  assert.throws(version(`v${'2'.repeat(4301)}.0.0\n`), (error) => isPyException(error, 'ValueError')
    && error.message.startsWith('Exceeds the limit (4300 digits) for integer string conversion: value has 4301 digits'));
});

test('test_qualify_runs_the_node_smokes_with_the_pinned_inputs', async () => {
  const inputs = writeInputs();
  const args = qualifyArgs(inputs);
  const calls = [];
  class Stop extends Error {}
  const harnessCalls = [];
  const deps = {
    nodeExecutable: () => '/opt/node/bin/node',
    qualificationEnvironment: () => ({}),
    qualificationRunIdentity: () => ({}),
    fetchCandidate: () => [7, 1],
    loadCandidate: () => [{ bridge_commit: BRIDGE_SHA }, 'f'],
    requireHarnessMatchesBridgeSource: (...callArgs) => {
      harnessCalls.push(callArgs);
      return 'd';
    },
    runSmoke: (command, label, diagnosticsDir, { timeoutSeconds }) => {
      calls.push([command, label, timeoutSeconds]);
      if (calls.length === 2) throw new Stop();
      return { ok: true };
    },
    speechPhase: () => ({}),
    stderr: () => {},
  };
  await assertRejects(qualifyCmd(args, () => {}, deps), (error) => assert.ok(error instanceof Stop));
  assert.deepEqual(harnessCalls, [[REPO_SCRIPTS_DIR, BRIDGE_SHA]]);
  const [[speech, speechLabel, speechTimeout], [tts, ttsLabel, ttsTimeout]] = calls;
  assert.deepEqual([speechLabel, speechTimeout, ttsLabel, ttsTimeout], ['speech-to-text', 67, 'text-to-speech', 68]);
  const diagnostics = fs.realpathSync(path.join(tmp, 'diag'));
  assert.deepEqual(speech, [
    '/opt/node/bin/node', path.join(REPO_SCRIPTS_DIR, 'smoke', 'speech_to_text.mjs'),
    '--dist-dir', speech[3],
    '--model-path', fs.realpathSync(inputs.sm),
    '--model-sha256', SPEECH_MODEL_SHA256,
    '--mmproj-path', fs.realpathSync(inputs.sp),
    '--mmproj-sha256', SPEECH_MMPROJ_SHA256,
    '--audio-path', fs.realpathSync(inputs.sa),
    '--audio-sha256', SPEECH_AUDIO_SHA256,
    '--memory-mode', 'all',
    '--timeout-ms', '7000',
    '--artifacts-dir', path.join(diagnostics, 'speech-to-text'),
  ]);
  assert.deepEqual(tts, [
    '/opt/node/bin/node', path.join(REPO_SCRIPTS_DIR, 'smoke', 'text_to_speech.mjs'),
    '--dist-dir', speech[3],
    '--model-path', fs.realpathSync(inputs.tm),
    '--model-sha256', TTS_MODEL_SHA256,
    '--mmproj-path', fs.realpathSync(inputs.tp),
    '--mmproj-sha256', TTS_MMPROJ_SHA256,
    '--memory-mode', 'wasm64',
    '--runtime-mode', 'all',
    '--max-frames', '24',
    '--timeout-ms', '8000',
    '--artifacts-dir', path.join(diagnostics, 'text-to-speech'),
  ]);
  assert.deepEqual([speech, tts].map((command) => path.relative(REPO_SCRIPTS_DIR, command[1])), [...QUALIFICATION_SMOKES]);
});

// --- qualifyCmd -----------------------------------------------------------------------

test('SCRIPTS_DIR is the scripts directory the smokes live in', () => {
  assert.equal(SCRIPTS_DIR, REPO_SCRIPTS_DIR);
  for (const smoke of QUALIFICATION_SMOKES) assert.ok(fs.existsSync(path.join(SCRIPTS_DIR, smoke)));
});

function happyDeps(record) {
  return {
    nodeExecutable: () => '/opt/node/bin/node',
    qualificationEnvironment: () => ({ execution: 'hosted-github-actions' }),
    qualificationRunIdentity: () => ({
      qualification_run_id: '32919086977',
      qualification_run_attempt: 1,
      qualification_source_sha: 'a'.repeat(40),
    }),
    fetchCandidate: async (runId, destination) => {
      record.fetch = [runId, destination];
      fs.mkdirSync(destination);
      return [7, 1];
    },
    loadCandidate: (directory) => [{ bridge_commit: BRIDGE_SHA, directory }, 'fingerprint'],
    requireHarnessMatchesBridgeSource: () => 'digest',
    runSmoke: async (command, label, diagnosticsDir, options) => {
      record.smokes.push({ label, diagnosticsDir, timeoutSeconds: options.timeoutSeconds });
      return { ok: true, label };
    },
    maxRssBytes: () => ++record.rss,
    speechPhase: (payload, rss) => ({ phase: payload.label, rss }),
    ttsPhase: (payload, rss, artifactsDir) => ({ phase: payload.label, rss, artifactsDir }),
    buildAttestation: (options) => {
      record.build = options;
      return { built: true };
    },
    verifyAttestation: (options) => {
      record.verify = options;
      return { verified: true };
    },
    canonicalJson: () => record.canonical,
    stderr: (text) => { record.stderr += text; },
  };
}

test('qualify builds, re-verifies and writes the canonical attestation', async () => {
  const inputs = writeInputs();
  const record = { smokes: [], rss: 0, stdout: '', stderr: '', canonical: '{\n  "x": 1\n}\n' };
  const args = qualifyArgs(inputs);
  assert.equal(await qualifyCmd(args, (text) => { record.stdout += text; }, happyDeps(record)), 0);
  const scratch = path.dirname(record.fetch[1]);
  assert.equal(fs.existsSync(scratch), false, 'the scratch root is removed');
  assert.equal(path.basename(record.fetch[1]), 'candidate');
  assert.ok(path.basename(scratch).startsWith('llama-web-bridge-qualification-'));
  assert.equal(record.fetch[0], CANDIDATE_RUN_ID);
  const diagnostics = fs.realpathSync(path.join(tmp, 'diag'));
  assert.deepEqual(record.smokes.map((smoke) => [smoke.label, smoke.diagnosticsDir, smoke.timeoutSeconds]), [
    ['speech-to-text', diagnostics, 67], ['text-to-speech', diagnostics, 68],
  ]);
  assert.deepEqual(record.build, {
    manifest: { bridge_commit: BRIDGE_SHA, directory: record.fetch[1] },
    candidateFingerprint: 'fingerprint',
    candidateRunId: CANDIDATE_RUN_ID,
    candidateArtifactId: 7,
    candidateRunAttempt: 1,
    qualificationRunId: '32919086977',
    qualificationRunAttempt: 1,
    qualificationSourceSha: 'a'.repeat(40),
    harnessDigest: 'digest',
    environment: { execution: 'hosted-github-actions' },
    speechPhase: { phase: 'speech-to-text', rss: 1 },
    ttsPhase: { phase: 'text-to-speech', rss: 2, artifactsDir: path.join(diagnostics, 'text-to-speech') },
  });
  assert.deepEqual(record.verify, {
    attestation: { built: true },
    candidateDir: record.fetch[1],
    candidateArtifactId: 7,
    candidateRunAttempt: 1,
    qualificationRunId: '32919086977',
    qualificationRunAttempt: 1,
    qualificationSourceSha: 'a'.repeat(40),
  });
  assert.equal(fs.readFileSync(args.outputAttestation, 'utf8'), record.canonical);
  assert.equal(record.stdout, record.canonical);
  assert.equal(record.stderr, [
    `Sanitized diagnostics directory ${diagnostics}`,
    `Downloading candidate artifact from run ${CANDIDATE_RUN_ID}`,
    'Candidate fingerprint fingerprint',
    'Running Qwen3-ASR wasm32+wasm64 direct+worker gate',
    'Running Qwen3-TTS wasm64 direct+worker gate',
    `Canonical attestation written to ${args.outputAttestation}`,
    '',
  ].join('\n'));
});

test('qualify refuses an attestation past the artifact bound, after removing the scratch root', async () => {
  const inputs = writeInputs();
  const record = { smokes: [], rss: 0, stdout: '', stderr: '', canonical: `"${'é'.repeat(16384)}"` };
  const args = qualifyArgs(inputs);
  await assertRejects(qualifyCmd(args, (text) => { record.stdout += text; }, happyDeps(record)), (error) => {
    assert.ok(error instanceof ContractError);
    assert.equal(error.message, 'attestation exceeds the 32768-byte artifact bound');
  });
  assert.equal(fs.existsSync(path.dirname(record.fetch[1])), false);
  assert.equal(fs.existsSync(args.outputAttestation), false);
  assert.equal(record.stdout, '');
});

test('qualify validates its arguments in Python\'s order, before any download', async () => {
  const inputs = writeInputs();
  const record = { smokes: [], rss: 0, stdout: '', stderr: '', canonical: '' };
  const cases = [
    [{ ttsMaxFrames: 0 }, 'tts_max_frames must be positive'],
    [{ ttsMaxFrames: 0, speechTimeoutSeconds: 0 }, 'tts_max_frames must be positive'],
    [{ speechTimeoutSeconds: 0 }, 'qualification gate timeouts must be positive'],
    [{ ttsTimeoutSeconds: -1n }, 'qualification gate timeouts must be positive'],
    [{ speechModelPath: path.join(tmp, 'absent') }, `Qwen3-ASR model does not exist: ${path.join(fs.realpathSync(tmp), 'absent')}`],
    [{ speechMmprojPath: tmp }, `Qwen3-ASR projector does not exist: ${fs.realpathSync(tmp)}`],
    [{ speechAudioPath: path.join(tmp, 'absent') }, `speech WAV fixture does not exist: ${path.join(fs.realpathSync(tmp), 'absent')}`],
    [{ ttsModelPath: path.join(tmp, 'absent') }, `Qwen3-TTS model does not exist: ${path.join(fs.realpathSync(tmp), 'absent')}`],
    [{ ttsMmprojPath: path.join(tmp, 'absent') }, `Qwen3-TTS projector does not exist: ${path.join(fs.realpathSync(tmp), 'absent')}`],
  ];
  for (const [overrides, message] of cases) {
    await assertRejects(qualifyCmd(qualifyArgs(inputs, overrides), (text) => { record.stdout += text; }, happyDeps(record)), (error) => {
      assert.ok(error instanceof ContractError);
      assert.equal(error.message, message);
    });
  }
  assert.equal(record.fetch, undefined);
});

test('qualify passes huge timeouts through as Python ints', async () => {
  const inputs = writeInputs();
  const commands = [];
  const deps = {
    ...happyDeps({ smokes: [], rss: 0, stdout: '', stderr: '', canonical: '' }),
    runSmoke: (command, label, diagnosticsDir, options) => {
      commands.push([command.at(-3), options.timeoutSeconds]);
      throw new ContractError('stop');
    },
  };
  await assertRejects(qualifyCmd(qualifyArgs(inputs, { speechTimeoutSeconds: 2n ** 60n }), () => {}, deps), () => {});
  assert.deepEqual(commands, [[String(2n ** 60n * 1000n), Number(2n ** 60n + 60n)]]);
});

test('a default diagnostics directory is made in the temp root', async () => {
  const inputs = writeInputs();
  const record = { smokes: [], rss: 0, stdout: '', stderr: '', canonical: '{}' };
  await qualifyCmd(qualifyArgs(inputs, { diagnosticsDir: null }), (text) => { record.stdout += text; }, happyDeps(record));
  const directory = record.smokes[0].diagnosticsDir;
  try {
    assert.ok(path.basename(directory).startsWith('llama-web-bridge-qualification-diag-'));
    assert.equal(path.dirname(directory), os.tmpdir());
    assert.ok(record.stderr.startsWith(`Sanitized diagnostics directory ${directory}\n`));
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test('a diagnostics path that is a file fails as Path.mkdir does', async () => {
  const inputs = writeInputs();
  const file = path.join(tmp, 'file');
  fs.writeFileSync(file, '');
  const record = { smokes: [], rss: 0, stdout: '', stderr: '', canonical: '{}' };
  const real = fs.realpathSync(file);
  await assertRejects(qualifyCmd(qualifyArgs(inputs, { diagnosticsDir: file }), (text) => { record.stdout += text; }, happyDeps(record)), (error) => {
    assert.ok(isPyException(error, 'FileExistsError') && isPyException(error, 'OSError'));
    assert.equal(error.message, `[Errno 17] File exists: '${real}'`);
  });
  await assertRejects(qualifyCmd(qualifyArgs(inputs, { diagnosticsDir: path.join(file, 'sub') }), (text) => { record.stdout += text; }, happyDeps(record)), (error) => {
    assert.ok(isPyException(error, 'NotADirectoryError'));
    assert.equal(error.message, `[Errno 20] Not a directory: '${real}/sub'`);
  });
});

test('the qualification CLI runs qualify and reports its failure as Python does', () => {
  const env = { ...process.env };
  for (const key of ['GITHUB_ACTIONS', 'RUNNER_ENVIRONMENT']) delete env[key];
  const result = spawnSync(process.execPath, [
    path.join(REPO_SCRIPTS_DIR, 'release', 'qualification.mjs'), 'qualify',
    '--candidate-run-id', CANDIDATE_RUN_ID,
    '--speech-model-path', 'x', '--speech-mmproj-path', 'x', '--speech-audio-path', 'x',
    '--tts-model-path', 'x', '--tts-mmproj-path', 'x',
    '--output-attestation', path.join(tmp, 'attestation.json'),
  ], { env, encoding: 'utf8', timeout: 60000, stdio: ['ignore', 'pipe', 'pipe'] });
  assert.equal(result.stderr, 'error: GITHUB_ACTIONS must identify a GitHub Actions run; qualification only runs on hosted GitHub Actions infrastructure\n');
  assert.equal(result.stdout, '');
  assert.equal(result.status, 1);
});

// --- nodeExecutable ----------------------------------------------------------------------

test('nodeExecutable accepts the running Node and reports what it cannot run', () => {
  assert.equal(nodeExecutable({ which: () => process.execPath }), fs.realpathSync(process.execPath));
  const plain = path.join(tmp, 'node');
  fs.writeFileSync(plain, '');
  assert.throws(() => nodeExecutable({ which: () => plain }), (error) => isPyException(error, 'PermissionError')
    && error.message === `[Errno 13] Permission denied: '${fs.realpathSync(plain)}'`);
  const noisy = path.join(tmp, 'noisy');
  fs.writeFileSync(noisy, "#!/bin/sh\necho ' v24.1.0-nightly '\n");
  fs.chmodSync(noisy, 0o755);
  assert.throws(() => nodeExecutable({ which: () => noisy }), (error) => error instanceof ContractError
    && error.message === 'node v24.1.0-nightly is too old; the qualification gates need Node.js 22.18+ or 24.2+');
  fs.writeFileSync(noisy, '#!/bin/sh\necho "v٢٢.١٨.0"\n');
  assert.equal(nodeExecutable({ which: () => noisy }), fs.realpathSync(noisy));
  fs.writeFileSync(noisy, '#!/bin/sh\nexit 3\n');
  assert.throws(() => nodeExecutable({ which: () => noisy }), (error) => error instanceof ContractError
    && error.message === 'node (unknown version) is too old; the qualification gates need Node.js 22.18+ or 24.2+');
});

test('pyWhich finds an executable file on PATH, as shutil.which does', () => {
  const bin = path.join(tmp, 'bin');
  fs.mkdirSync(bin);
  fs.mkdirSync(path.join(bin, 'node'));
  const other = path.join(tmp, 'other');
  fs.mkdirSync(other);
  fs.writeFileSync(path.join(other, 'node'), '');
  assert.equal(pyWhich('node', { PATH: `${bin}:${other}` }), null);
  fs.chmodSync(path.join(other, 'node'), 0o755);
  assert.equal(pyWhich('node', { PATH: `${bin}:${other}` }), path.join(other, 'node'));
  assert.equal(pyWhich('node', { PATH: '' }), null);
});

// --- path and format helpers -----------------------------------------------------------------

test('requireInputFile expands ~ and resolves symlinks', () => {
  const file = path.join(tmp, 'model.gguf');
  fs.writeFileSync(file, '');
  const link = path.join(tmp, 'link.gguf');
  fs.symlinkSync(file, link);
  assert.equal(requireInputFile(link, 'model'), fs.realpathSync(file));
  assert.equal(requireInputFile(`${tmp}//./link.gguf`, 'model'), fs.realpathSync(file));
  const home = process.env.HOME;
  process.env.HOME = `${tmp}/`;
  try {
    assert.equal(requireInputFile('~/model.gguf', 'model'), fs.realpathSync(file));
    assert.throws(() => requireInputFile('~/absent', 'model'), (error) => error instanceof ContractError
      && error.message === `model does not exist: ${path.join(fs.realpathSync(tmp), 'absent')}`);
  } finally {
    process.env.HOME = home;
  }
  assert.throws(() => requireInputFile('~no-such-user-here/x', 'model'), (error) => isPyException(error, 'RuntimeError')
    && error.message === 'Could not determine home directory.');
});

test('pyPathName is PurePosixPath.name', () => {
  for (const [text, name] of [['a.wav', 'a.wav'], ['a.wav/', 'a.wav'], ['./a.wav', 'a.wav'], ['/', ''], ['.', ''], ['..', '..'], ['x/..', '..'], ['//a', 'a']]) {
    assert.equal(pyPathName(text), name, text);
  }
});

test('pyFormatG matches Python format(value, "g")', () => {
  // Reference values from CPython: [repr(v), format(v, 'g')].
  const cases = [
    [67, '67'], [0.05, '0.05'], [5460, '5460'], [1e6, '1e+06'], [1234565, '1.23456e+06'], [1234575, '1.23458e+06'],
    [999999, '999999'], [9999995, '1e+07'], [0.0001, '0.0001'], [0.00001, '1e-05'], [1.5, '1.5'], [123456.5, '123456'],
    [123457.5, '123458'], [0.30000000000000004, '0.3'], [2.5e-5, '2.5e-05'], [1e21, '1e+21'], [1.5e300, '1.5e+300'],
    [5e-324, '4.94066e-324'], [100, '100'], [100000, '100000'], [999999.5, '1e+06'], [0.000123456789, '0.000123457'],
    [-3.25, '-3.25'], [1e16, '1e+16'], [7, '7'], [2n ** 60n, '1.15292e+18'],
  ];
  for (const [value, expected] of cases) assert.equal(pyFormatG(value), expected, String(value));
});
