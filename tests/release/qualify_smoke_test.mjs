// Tests of runSmoke and its process control in scripts/release/qualify.mjs,
// the ports of release_qualification.py's _run_smoke, _stop_smoke_process,
// _write_smoke_diagnostics, _child_env and max_rss_bytes. The first block is
// one test per Python test of the same name, with a Node child where Python
// ran sys.executable; the rest pin the process-group, signal, grace-period
// and decoding behaviour. Every child is awaited, and any grandchild a test
// starts is killed in teardown.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { afterEach, beforeEach, test } from 'node:test';

import { ContractError } from '../../scripts/release/errors.mjs';
import { isPyException } from '../../scripts/release/json.mjs';
import {
  KeyboardInterrupt,
  SmokeProcess,
  TimeoutExpired,
  childEnv,
  decodeText,
  internals,
  maxRssBytes,
  pythonCoercedLocale,
  rssWrapper,
  runSmoke,
  stopSmokeProcess,
  writeSmokeDiagnostics,
} from '../../scripts/release/qualify.mjs';
import { isAlive, killAll, makeTmp, waitUntilGone } from './qualify_fixtures.mjs';

const NODE = process.execPath;
const POSIX = process.platform !== 'win32';

let tmp;
const leftovers = [];
beforeEach(() => {
  tmp = makeTmp('qualify-smoke-test-');
});
afterEach(() => {
  killAll(leftovers);
  fs.rmSync(tmp, { recursive: true, force: true });
});

function diagnosticsDir(name) {
  const directory = path.join(tmp, name);
  fs.mkdirSync(directory);
  return directory;
}

function nodeCommand(source) {
  return [NODE, '-e', source];
}

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

function captureStderr() {
  const lines = [];
  const write = (text) => lines.push(text);
  write.text = () => lines.join('');
  return write;
}

// --- the Python tests ---------------------------------------------------------

test('test_successful_child_is_parsed_from_raw_stdout', async () => {
  const diagnostics = diagnosticsDir('raw-stdout-diagnostics');
  const payload = {
    ok: true,
    note: 'n_tokens = 65 n_tokens_batch = 65',
    modelUrl: 'https://example.com/m.gguf?token=secret123',
  };
  const parsed = await runSmoke(
    nodeCommand(`process.stdout.write(${JSON.stringify(JSON.stringify(payload))})`),
    'raw-probe',
    diagnostics,
    { timeoutSeconds: 30 },
  );
  assert.deepEqual(parsed, payload);
  const persisted = JSON.parse(fs.readFileSync(path.join(diagnostics, 'raw-probe-stdout.json'), 'utf8'));
  assert.equal(persisted.note, payload.note);
  assert.equal(persisted.modelUrl, 'https://example.com/m.gguf');
});

test('test_successful_child_rejects_non_strict_json', async () => {
  for (const [label, raw, expected] of [
    ['duplicate', '{"ok":true,"ok":true}', 'duplicate'],
    ['nonstandard', '{"ok":true,"metric":NaN}', 'non-standard'],
  ]) {
    const diagnostics = diagnosticsDir(`${label}-diagnostics`);
    await assertRejects(
      runSmoke(nodeCommand(`process.stdout.write(${JSON.stringify(raw)})`), label, diagnostics, { timeoutSeconds: 30 }),
      (error) => {
        assert.ok(error instanceof ContractError);
        assert.ok(error.message.toLowerCase().includes(expected), error.message);
      },
    );
  }
});

test('test_successful_child_duplicate_key_error_is_sanitized', async () => {
  const diagnostics = diagnosticsDir('duplicate-key-diagnostics');
  const duplicateKey = 'https://url-user:url-pass@example.com/m.gguf?token=child-secret#frag';
  const raw = `${JSON.stringify({ ok: true }).slice(0, -1)},${JSON.stringify(duplicateKey)}:1,${JSON.stringify(duplicateKey)}:2}`;
  const error = await assertRejects(
    runSmoke(nodeCommand(`process.stdout.write(${JSON.stringify(raw)})`), 'duplicate-key', diagnostics, { timeoutSeconds: 30 }),
    (caught) => assert.ok(caught instanceof ContractError),
  );
  const tracebackText = `${error.stack}\n${error.cause ?? ''}`;
  assert.ok(error.message.toLowerCase().includes('duplicate json key'));
  assert.ok(error.message.includes('https://example.com/m.gguf'), error.message);
  assert.equal(error.cause, undefined);
  for (const leaked of ['url-user', 'url-pass', 'child-secret', 'token=', '#frag']) {
    assert.ok(!error.message.includes(leaked), leaked);
    assert.ok(!tracebackText.includes(leaked), leaked);
  }
});

// The ubuntu-24.04 runners ship GNU /usr/bin/time, the only way the port
// measures child RSS; there its absence fails the RSS tests instead of
// skipping them.
const RSS_REQUIRED = process.platform === 'linux' && process.env.GITHUB_ACTIONS === 'true';

function requireRssWrapper() {
  if (RSS_REQUIRED) assert.ok(rssWrapper(), 'GNU /usr/bin/time is required to measure child RSS on the Linux runners');
  return rssWrapper() !== null;
}

test('test_max_rss_is_reported_in_bytes', { skip: rssWrapper() === null && !RSS_REQUIRED && 'no /usr/bin/time to measure child RSS' }, async () => {
  requireRssWrapper();
  // Python's version reaps any 64 MiB child; here the child must be a smoke,
  // the only children whose RSS the Node port measures.
  const diagnostics = diagnosticsDir('rss-diagnostics');
  await runSmoke(
    nodeCommand('const b = Buffer.alloc(64 * 1024 * 1024, 1); process.stdout.write(JSON.stringify({ ok: b[0] === 1 }))'),
    'rss-probe',
    diagnostics,
    { timeoutSeconds: 60 },
  );
  // ru_maxrss is kibibytes on Linux and bytes on macOS; the normalized value
  // must be plausible as bytes on either, never 1024x off.
  assert.ok(maxRssBytes() > 4_000_000, String(maxRssBytes()));
  assert.ok(maxRssBytes() > 64 * 1024 * 1024, String(maxRssBytes()));
  assert.ok(maxRssBytes() < 16 * 1024 ** 3, String(maxRssBytes()));
});

test('test_child_env_drops_ambient_smoke_configuration', () => {
  const env = childEnv({
    ...process.env,
    LLAMA_WEBGPU_SPEECH_MODEL_URL: 'https://evil.example/model.gguf',
    BRIDGE_DIST_DIR: '/tmp/elsewhere',
    GH_TOKEN: 'must-not-reach-browser',
    EXAMPLE_API_KEY: 'must-not-reach-browser',
  });
  assert.equal('LLAMA_WEBGPU_SPEECH_MODEL_URL' in env, false);
  assert.equal('BRIDGE_DIST_DIR' in env, false);
  assert.equal('GH_TOKEN' in env, false);
  assert.equal('EXAMPLE_API_KEY' in env, false);
});

test('test_smoke_timeout_is_bounded_and_writes_sanitized_diagnostics', async () => {
  const diagnostics = diagnosticsDir('timeout-diagnostics');
  const stderr = captureStderr();
  await assertRejects(
    runSmoke(
      nodeCommand("process.stderr.write('GH_TOKEN=secret\\n'); setTimeout(() => {}, 5000)"),
      'timeout-probe',
      diagnostics,
      { timeoutSeconds: 0.05, stderr },
    ),
    (error) => {
      assert.ok(error instanceof ContractError);
      assert.ok(error.message.includes('timed out'));
    },
  );
  const diagnostic = fs.readFileSync(path.join(diagnostics, 'timeout-probe-stderr.log'), 'utf8');
  assert.ok(!diagnostic.includes('secret'));
});

test('test_malformed_child_output_still_writes_sanitized_diagnostics', async () => {
  const diagnostics = diagnosticsDir('malformed-diagnostics');
  const program = "process.stdout.write(Buffer.from('GH_TOKEN=stdout-secret\\nbad\\xff', 'latin1'));"
    + "process.stderr.write(Buffer.from('api_key=stderr-secret\\nbad\\xff', 'latin1'));"
    + 'process.exitCode = 1;';
  const stderr = captureStderr();
  await assertRejects(
    runSmoke(nodeCommand(program), 'malformed-probe', diagnostics, { timeoutSeconds: 30, stderr }),
    (error) => assert.ok(error.message.includes('failed with exit status 1')),
  );
  const stdoutDiagnostic = fs.readFileSync(path.join(diagnostics, 'malformed-probe-stdout.json'), 'utf8');
  assert.equal(typeof JSON.parse(stdoutDiagnostic), 'string');
  const persisted = stdoutDiagnostic
    + fs.readFileSync(path.join(diagnostics, 'malformed-probe-stderr.log'), 'utf8')
    + stderr.text();
  assert.ok(!persisted.includes('stdout-secret'));
  assert.ok(!persisted.includes('stderr-secret'));
  assert.ok(persisted.includes('�'));
});

test('the child environment carries the LC_CTYPE a C-locale Python coerces itself to (PEP 538)', () => {
  const base = { PATH: '/usr/bin', HOME: '/home/x' };
  assert.deepEqual(childEnv({ ...base, LANG: 'C.UTF-8' }, 'linux'), { ...base, LANG: 'C.UTF-8' });
  assert.deepEqual(childEnv({ ...base }), { ...base, LC_CTYPE: 'C.UTF-8' });
  assert.deepEqual(childEnv({ ...base, LC_CTYPE: 'C', LANG: 'C.UTF-8' }), { ...base, LC_CTYPE: 'C.UTF-8', LANG: 'C.UTF-8' });
  assert.deepEqual(childEnv({ ...base, LC_ALL: 'C' }), { ...base, LC_ALL: 'C' });
  for (const [env, platform, expected] of [
    [{}, 'linux', 'C.UTF-8'],
    [{ LC_ALL: '', LANG: 'C' }, 'linux', 'C.UTF-8'],
    [{ LANG: 'POSIX' }, 'linux', 'C.UTF-8'],
    [{ LANG: 'POSIX' }, 'darwin', null],
    [{ LANG: 'en_US.UTF-8' }, 'linux', null],
    [{ LC_CTYPE: 'en_US.UTF-8', LANG: 'C' }, 'linux', null],
    [{ PYTHONCOERCECLOCALE: '0' }, 'linux', null],
    [{ LC_ALL: 'C' }, 'linux', null],
  ]) assert.equal(pythonCoercedLocale(env, platform), expected, JSON.stringify(env));
});

test('test_structured_stdout_diagnostic_stays_valid_json', () => {
  const diagnostics = diagnosticsDir('structured-diagnostics');
  const payload = {
    ok: true,
    note: 'n_tokens = 65\nsaid "hi"\tthen stopped',
    apiKey: 'plain-secret',
    modeResults: [{ log: 'Authorization: Bearer ghp_must_not_escape', n_tokens: 65 }],
    modelUrl: 'https://example.com/m.gguf?token=secret123#frag',
  };
  writeSmokeDiagnostics(diagnostics, 'structured', JSON.stringify(payload), '');
  const text = fs.readFileSync(path.join(diagnostics, 'structured-stdout.json'), 'utf8');
  const persisted = JSON.parse(text);
  assert.equal(persisted.note, payload.note);
  assert.equal(persisted.modeResults[0].n_tokens, 65);
  assert.equal(persisted.apiKey, '<redacted-credential>');
  assert.equal(persisted.modelUrl, 'https://example.com/m.gguf');
  assert.ok(!persisted.modeResults[0].log.includes('ghp_must_not_escape'));
  assert.ok(!text.includes('secret123'));
});

test('test_structured_stdout_preserves_escaped_unicode_safely', () => {
  const diagnostics = diagnosticsDir('unicode-diagnostics');
  writeSmokeDiagnostics(diagnostics, 'unicode', '{"ok":true,"note":"\\ud800"}', '');
  const serialized = fs.readFileSync(path.join(diagnostics, 'unicode-stdout.json'), 'utf8');
  assert.ok(serialized.includes('\\ud800'));
  assert.equal(JSON.parse(serialized).note, '\ud800');
});

// --- process control -----------------------------------------------------------------

test('a non-positive timeout is refused before anything runs', async () => {
  for (const timeoutSeconds of [0, -1]) {
    await assertRejects(runSmoke([path.join(tmp, 'absent')], 'probe', tmp, { timeoutSeconds }),
      (error) => assert.equal(error.message, 'probe timeout must be positive'));
  }
});

test('a missing or non-executable program raises the OSError Popen raises', async () => {
  const missing = path.join(tmp, 'absent');
  await assertRejects(runSmoke([missing], 'probe', tmp, { timeoutSeconds: 5 }), (error) => {
    assert.ok(isPyException(error, 'FileNotFoundError'));
    assert.equal(error.message, `[Errno 2] No such file or directory: '${missing}'`);
  });
  const plain = path.join(tmp, 'plain');
  fs.writeFileSync(plain, '');
  await assertRejects(runSmoke([plain], 'probe', tmp, { timeoutSeconds: 5 }), (error) => {
    assert.ok(isPyException(error, 'PermissionError'));
    assert.equal(error.message, `[Errno 13] Permission denied: '${plain}'`);
  });
  await assertRejects(runSmoke(['definitely-not-a-program-here'], 'probe', tmp, { timeoutSeconds: 5 }), (error) => {
    assert.equal(error.message, "[Errno 2] No such file or directory: 'definitely-not-a-program-here'");
  });
});

test('a gate that exits non-zero or reports ok other than true fails with Python\'s message', async () => {
  const stderr = captureStderr();
  await assertRejects(runSmoke(nodeCommand("process.stderr.write('why\\n'); process.exitCode = 3"), 'x', tmp, { timeoutSeconds: 30, stderr }),
    (error) => assert.equal(error.message, 'x gate failed with exit status 3'));
  assert.equal(stderr.text(), 'why\n');
  for (const stdout of ['[]', '{"ok":1}', '{"ok":"true"}', '{}']) {
    await assertRejects(runSmoke(nodeCommand(`process.stdout.write(${JSON.stringify(stdout)})`), 'x', tmp, { timeoutSeconds: 30 }),
      (error) => assert.equal(error.message, 'x gate did not report ok=true'));
  }
  await assertRejects(runSmoke(nodeCommand("process.stdout.write('')"), 'x', tmp, { timeoutSeconds: 30 }),
    (error) => assert.equal(error.message, 'x gate emitted invalid JSON: Expecting value: line 1 column 1 (char 0)'));
});

test('a gate killed by a signal reports the negative status Popen reports', { skip: !POSIX }, async () => {
  for (const signal of ['SIGKILL', 'SIGTERM', 'SIGSEGV']) {
    await assertRejects(
      runSmoke(nodeCommand(`process.kill(process.pid, '${signal}')`), 'sig', tmp, { timeoutSeconds: 30, stderr: () => {} }),
      (error) => assert.equal(error.message, `sig gate failed with exit status -${{ SIGKILL: 9, SIGTERM: 15, SIGSEGV: 11 }[signal]}`),
    );
  }
});

test('on timeout the whole process group gets SIGTERM, and SIGKILL after the grace period', { skip: !POSIX }, async () => {
  const pidFile = path.join(tmp, 'pids');
  // The child and a grandchild both ignore SIGTERM and keep the pipes open.
  const grandchild = "process.on('SIGTERM', () => process.stderr.write('grandchild ignored SIGTERM\\n')); setInterval(() => {}, 1000)";
  const program = [
    "const { spawn } = require('node:child_process');",
    `const g = spawn(process.execPath, ['-e', ${JSON.stringify(grandchild)}], { stdio: ['ignore', 'inherit', 'inherit'] });`,
    `require('node:fs').writeFileSync(${JSON.stringify(pidFile)}, process.pid + ' ' + g.pid);`,
    "process.on('SIGTERM', () => process.stderr.write('child ignored SIGTERM\\n'));",
    "process.stdout.write('partial');",
    'setInterval(() => {}, 1000);',
  ].join('\n');
  const stderr = captureStderr();
  const started = Date.now();
  await assertRejects(
    runSmoke(nodeCommand(program), 'stubborn', tmp, { timeoutSeconds: 1.5, graceSeconds: 0.5, stderr }),
    (error) => assert.equal(error.message, 'stubborn gate timed out after 1.5 seconds'),
  );
  const elapsed = Date.now() - started;
  assert.ok(elapsed >= 1900, `SIGKILL came before the grace period: ${elapsed} ms`);
  const [childPid, grandchildPid] = fs.readFileSync(pidFile, 'utf8').split(' ').map(Number);
  leftovers.push(childPid, grandchildPid);
  assert.ok(await waitUntilGone(childPid), 'child survived');
  assert.ok(await waitUntilGone(grandchildPid), 'grandchild survived');
  assert.match(stderr.text(), /child ignored SIGTERM/);
  assert.match(stderr.text(), /grandchild ignored SIGTERM/);
  assert.equal(fs.readFileSync(path.join(tmp, 'stubborn-stdout.json'), 'utf8'), '"partial"\n');
});

test('SIGTERM is skipped when the direct child already exited, and a lingering grandchild is SIGKILLed', { skip: !POSIX }, async () => {
  const pidFile = path.join(tmp, 'pid');
  const grandchild = "process.on('SIGTERM', () => process.stderr.write('grandchild got SIGTERM\\n')); setInterval(() => {}, 1000)";
  const program = [
    "const { spawn } = require('node:child_process');",
    `const g = spawn(process.execPath, ['-e', ${JSON.stringify(grandchild)}], { stdio: ['ignore', 'inherit', 'inherit'], detached: false });`,
    `require('node:fs').writeFileSync(${JSON.stringify(pidFile)}, String(g.pid));`,
    'g.unref(); process.exit(0);',
  ].join('\n');
  const stderr = captureStderr();
  await assertRejects(
    runSmoke(nodeCommand(program), 'linger', tmp, { timeoutSeconds: 1, graceSeconds: 0.3, stderr }),
    (error) => assert.equal(error.message, 'linger gate timed out after 1 seconds'),
  );
  const grandchildPid = Number(fs.readFileSync(pidFile, 'utf8'));
  leftovers.push(grandchildPid);
  assert.ok(await waitUntilGone(grandchildPid));
  assert.doesNotMatch(stderr.text(), /grandchild got SIGTERM/);
});

test('a child that exits on SIGTERM is not SIGKILLed and its last words are kept', { skip: !POSIX }, async () => {
  const program = "process.on('SIGTERM', () => { process.stderr.write('bye\\n'); process.exit(0); }); setInterval(() => {}, 1000)";
  const stderr = captureStderr();
  const started = Date.now();
  await assertRejects(runSmoke(nodeCommand(program), 'polite', tmp, { timeoutSeconds: 0.5, stderr }),
    (error) => assert.equal(error.message, 'polite gate timed out after 0.5 seconds'));
  assert.ok(Date.now() - started < 4000);
  assert.equal(stderr.text(), 'bye\n');
});

test('SIGINT stops the smoke, writes its diagnostics and raises KeyboardInterrupt', { skip: !POSIX }, async () => {
  const program = "process.stderr.write('api_key=hidden\\n'); process.stdout.write('{\"ok\": true'); setInterval(() => {}, 1000)";
  const pending = runSmoke(nodeCommand(program), 'interrupted', tmp, { timeoutSeconds: 30, stderr: () => {} });
  await new Promise((resolve) => setTimeout(resolve, 300));
  process.emit('SIGINT', 'SIGINT');
  await assertRejects(pending, (error) => assert.ok(error instanceof KeyboardInterrupt));
  assert.equal(process.listenerCount('SIGINT'), 0);
  assert.ok(fs.existsSync(path.join(tmp, 'interrupted-stdout.json')));
  assert.ok(!fs.readFileSync(path.join(tmp, 'interrupted-stderr.log'), 'utf8').includes('hidden'));
});

test('a flood of output on both streams is captured in full', async () => {
  const program = "const line = 'x'.repeat(1023) + '\\n'; for (let i = 0; i < 8192; i++) { process.stderr.write(line); }"
    + " process.stdout.write(JSON.stringify({ ok: true, pad: 'y'.repeat(8 * 1024 * 1024) }))";
  const payload = await runSmoke(nodeCommand(program), 'flood', tmp, { timeoutSeconds: 60 });
  assert.equal(payload.pad.length, 8 * 1024 * 1024);
  assert.equal(fs.statSync(path.join(tmp, 'flood-stderr.log')).size, 8 * 1024 * 1024);
});

test('output is decoded as text mode does: replacement characters, a kept BOM, universal newlines', () => {
  assert.equal(decodeText(Buffer.from([0xef, 0xbb, 0xbf, 0x61])), '﻿a');
  assert.equal(decodeText(Buffer.from('a\r\nb\rc\n', 'latin1')), 'a\nb\nc\n');
  assert.equal(decodeText(Buffer.from([0x61, 0xff, 0xe2, 0x82])), 'a��');
  assert.equal(decodeText(Buffer.from([0xed, 0xa0, 0x80])), '���');
});

test('a BOM before the result is invalid JSON, as json.loads(str) rejects it', async () => {
  await assertRejects(
    runSmoke(nodeCommand("process.stdout.write(Buffer.from([0xef, 0xbb, 0xbf])); process.stdout.write('{\"ok\":true}')"), 'bom', tmp, { timeoutSeconds: 30 }),
    (error) => assert.equal(error.message, 'bom gate emitted invalid JSON: Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)'),
  );
});

test('the smoke runs with the allowlisted environment only', async () => {
  process.env.LLAMA_WEBGPU_SPEECH_MODEL_URL = 'https://evil.example/model.gguf';
  try {
    const payload = await runSmoke(nodeCommand('process.stdout.write(JSON.stringify({ ok: true, keys: Object.keys(process.env).sort() }))'),
      'env', tmp, { timeoutSeconds: 30 });
    // macOS CoreFoundation adds __CF_USER_TEXT_ENCODING inside every process.
    const passed = payload.keys.filter((key) => key !== '__CF_USER_TEXT_ENCODING');
    assert.deepEqual(passed, Object.keys(childEnv()).sort());
    assert.ok(!passed.includes('GH_TOKEN'));
    assert.ok(!payload.keys.includes('LLAMA_WEBGPU_SPEECH_MODEL_URL'));
  } finally {
    delete process.env.LLAMA_WEBGPU_SPEECH_MODEL_URL;
  }
});

test('the smoke runs in a new process group, led by the direct child', { skip: !POSIX }, async () => {
  const program = "const { execFileSync } = require('node:child_process');"
    + "const pgid = Number(execFileSync('ps', ['-o', 'pgid=', '-p', String(process.pid)]).toString().trim());"
    + 'process.stdout.write(JSON.stringify({ ok: true, pgid, pid: process.pid, ppid: process.ppid }))';
  const payload = await runSmoke(nodeCommand(program), 'session', tmp, { timeoutSeconds: 30 });
  assert.notEqual(payload.pgid, process.pid);
  // With the RSS wrapper the smoke's parent is /usr/bin/time, which leads the
  // group; without it the smoke leads it.
  assert.equal(payload.pgid, requireRssWrapper() ? payload.ppid : payload.pid);
  const unwrapped = await runSmoke(nodeCommand(program), 'session', tmp, { timeoutSeconds: 30, wrapper: null, stderr: () => {} });
  assert.equal(unwrapped.pgid, unwrapped.pid);
  assert.equal(unwrapped.ppid, process.pid);
});

test('the peak RSS is the largest over every smoke, and stays 0 when nothing was measured', async () => {
  internals.resetPeakChildRss();
  assert.equal(maxRssBytes(), 0);
  await runSmoke(nodeCommand("process.stdout.write('{\"ok\":true}')"), 'unmeasured', tmp, { timeoutSeconds: 30, wrapper: null, stderr: () => {} });
  assert.equal(maxRssBytes(), 0);
  if (!requireRssWrapper()) return;
  await runSmoke(nodeCommand("const b = Buffer.alloc(96 * 1024 * 1024, 1); process.stdout.write('{\"ok\":true}')"), 'big', tmp, { timeoutSeconds: 60 });
  const peak = maxRssBytes();
  assert.ok(peak > 96 * 1024 * 1024);
  await runSmoke(nodeCommand("process.stdout.write('{\"ok\":true}')"), 'small', tmp, { timeoutSeconds: 30 });
  assert.equal(maxRssBytes(), peak);
});

test('writeSmokeDiagnostics writes the sanitized stderr log and stdout record', () => {
  const sanitized = writeSmokeDiagnostics(tmp, 'd', '{"ok": true}', 'GH_TOKEN=abc\n');
  assert.ok(!sanitized.includes('abc'));
  assert.equal(fs.readFileSync(path.join(tmp, 'd-stderr.log'), 'utf8'), sanitized);
  assert.ok(fs.existsSync(path.join(tmp, 'd-stdout.json')));
  assert.throws(() => writeSmokeDiagnostics(path.join(tmp, 'absent'), 'd', '', ''),
    (error) => isPyException(error, 'FileNotFoundError')
      && error.message === `[Errno 2] No such file or directory: '${path.join(tmp, 'absent', 'd-stderr.log')}'`);
});

test('stopSmokeProcess returns everything the smoke wrote', { skip: !POSIX }, async () => {
  const smoke = new SmokeProcess(nodeCommand("process.stdout.write('out'); process.stderr.write('err'); setInterval(() => {}, 1000)"));
  await new Promise((resolve) => setTimeout(resolve, 300));
  assert.equal(smoke.poll(), null);
  const [stdout, stderr] = await stopSmokeProcess(smoke, { graceSeconds: 2 });
  assert.equal(stdout, 'out');
  assert.equal(stderr, 'err');
  assert.equal(smoke.returncode, -15);
  assert.equal(isAlive(smoke.pid), false);
});

test('the RSS report directory is removed when the wrapped child cannot start', { skip: !POSIX }, async () => {
  const saved = process.env.TMPDIR;
  const scratch = path.join(tmp, 'tmpdir');
  fs.mkdirSync(scratch);
  process.env.TMPDIR = scratch;
  try {
    // spawn() emits 'error' for a wrapper that does not exist.
    const missing = { kind: 'gnu', argv: (file, command) => [path.join(tmp, 'no-such-time'), '-o', file, '--', ...command] };
    const smoke = new SmokeProcess(nodeCommand(''), { wrapper: missing });
    assert.equal(fs.readdirSync(scratch).length, 1);
    await assertRejects(smoke.communicate(), (error) => assert.ok(isPyException(error, 'OSError'), String(error)));
    assert.deepEqual(fs.readdirSync(scratch), []);
    // spawn() throws for an argument it refuses.
    const refused = { kind: 'gnu', argv: (file, command) => ['/usr/bin/time', 'a\0b', file, ...command] };
    assert.throws(() => new SmokeProcess(nodeCommand(''), { wrapper: refused }), (error) => error.code === 'ERR_INVALID_ARG_VALUE');
    assert.deepEqual(fs.readdirSync(scratch), []);
  } finally {
    if (saved === undefined) delete process.env.TMPDIR;
    else process.env.TMPDIR = saved;
  }
});

test('TimeoutExpired renders as subprocess.TimeoutExpired does', () => {
  assert.equal(new TimeoutExpired(['/usr/bin/node', '--version'], 30).tracebackLine,
    "subprocess.TimeoutExpired: Command '['/usr/bin/node', '--version']' timed out after 30 seconds");
  assert.equal(new TimeoutExpired(["it's"], 1.5).message, `Command '["it's"]' timed out after 1.5 seconds`);
});
