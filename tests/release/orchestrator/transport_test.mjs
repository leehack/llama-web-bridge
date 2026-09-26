// Tests of scripts/release/orchestrator/transport.mjs, one test per test method
// of scripts/release_orchestrator_transport_test.py, with the same names and
// assertions. The real argv/stdin boundary is exercised through a fake `gh`
// on a private PATH passed as the gateway's environment, so the tests never
// touch process.env and run in parallel.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { test } from 'node:test';

import { ASSETS_REPOSITORY, BRIDGE_REPOSITORY, ContractError } from '../../../scripts/release/contract.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as transport from '../../../scripts/release/orchestrator/transport.mjs';

// A fake gh that logs its argv, stdin and token environment, then prints
// GH_FAKE_STDOUT and GH_FAKE_STDERR and exits with GH_FAKE_EXIT.
function fakeGh(directory) {
  const executable = path.join(directory, 'gh');
  const log = path.join(directory, 'gh-log.json');
  fs.writeFileSync(executable, `#!/usr/bin/env node
const fs = require('node:fs');
// Only a dispatch writes stdin; every other call inherits the runner's stdin,
// which may never reach end of file.
const stdin = process.argv[2] === 'workflow' ? fs.readFileSync(0, 'utf8') : '';
fs.writeFileSync(process.env.GH_FAKE_LOG, JSON.stringify({
  argv: process.argv.slice(2),
  stdin,
  gh_token: process.env.GH_TOKEN ?? null,
  github_token_present: 'GITHUB_TOKEN' in process.env,
}));
process.stdout.write(process.env.GH_FAKE_STDOUT ?? '');
process.stderr.write(process.env.GH_FAKE_STDERR ?? '');
process.exitCode = Number(process.env.GH_FAKE_EXIT ?? '0');
`, { mode: 0o700 });
  fs.writeFileSync(path.join(directory, 'package.json'), '{"type": "commonjs"}');
  return [executable, log];
}

function withTemporaryDirectory(fn) {
  const directory = fs.mkdtempSync(path.join(tmpdir(), 'sro-transport-'));
  try {
    return fn(directory);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
}

function environment(directory, extra) {
  return { ...process.env, PATH: `${directory}${path.delimiter}${process.env.PATH ?? ''}`, ...extra };
}

test('test_workflow_dispatch_uses_json_stdin_and_accepts_url_stdout', () => {
  const observed = withTemporaryDirectory((directory) => {
    const [, log] = fakeGh(directory);
    const env = environment(directory, {
      GH_FAKE_LOG: log,
      GH_FAKE_STDOUT: `https://github.com/${BRIDGE_REPOSITORY}/actions/runs/501\n`,
      GITHUB_TOKEN: 'must-be-removed',
    });
    const gateway = new transport.GhGateway({ readToken: 'read-token', dispatchToken: 'dispatch-token', env });
    gateway.dispatchWorkflow({
      workflowFile: model.CANDIDATE_WORKFLOW_FILE,
      ref: 'main',
      inputs: { zeta: 'last', alpha: 'first' },
    });
    return JSON.parse(fs.readFileSync(log, 'utf8'));
  });
  assert.deepEqual(observed.argv, [
    'workflow',
    'run',
    model.CANDIDATE_WORKFLOW_FILE,
    '--repo',
    BRIDGE_REPOSITORY,
    '--ref',
    'main',
    '--json',
  ]);
  assert.deepEqual(JSON.parse(observed.stdin), { alpha: 'first', zeta: 'last' });
  // json.dumps(inputs, sort_keys=True), byte for byte.
  assert.equal(observed.stdin, '{"alpha": "first", "zeta": "last"}');
  assert.equal(observed.gh_token, 'dispatch-token');
  assert.equal(observed.github_token_present, false);
});

test('test_release_verify_uses_json_format_and_read_credential', () => {
  const observed = withTemporaryDirectory((directory) => {
    const [, log] = fakeGh(directory);
    const env = environment(directory, { GH_FAKE_LOG: log, GH_FAKE_STDOUT: '{}\n' });
    const gateway = new transport.GhGateway({ readToken: 'read-token', dispatchToken: 'dispatch-token', env });
    assert.deepEqual(gateway.releaseAttestation({ repository: ASSETS_REPOSITORY, releaseTag: 'v0.1.40' }), {});
    return JSON.parse(fs.readFileSync(log, 'utf8'));
  });
  assert.deepEqual(observed.argv, [
    'release',
    'verify',
    'v0.1.40',
    '--repo',
    ASSETS_REPOSITORY,
    '--format',
    'json',
  ]);
  assert.equal(observed.gh_token, 'read-token');
});

// Node-only checks of the rest of the GhGateway surface, beyond the Python
// suite: the api argv and token split, failure text, and the clock format.
test('node_api_json_paginates_and_splits_tokens', () => {
  withTemporaryDirectory((directory) => {
    const [, log] = fakeGh(directory);
    const env = environment(directory, { GH_FAKE_LOG: log, GH_FAKE_STDOUT: '[[1, 2.0]]' });
    const gateway = new transport.GhGateway({ readToken: 'read-token', dispatchToken: 'dispatch-token', env });
    const payload = gateway.apiJson('repos/x/y/releases?per_page=100', { paginate: true });
    assert.equal(payload.length, 1);
    let observed = JSON.parse(fs.readFileSync(log, 'utf8'));
    assert.deepEqual(observed.argv, ['api', '--paginate', '--slurp', 'repos/x/y/releases?per_page=100']);
    assert.equal(observed.gh_token, 'read-token');
    gateway.downloadBytes('repos/x/y/zip', { accept: 'application/zip', privileged: true });
    observed = JSON.parse(fs.readFileSync(log, 'utf8'));
    assert.deepEqual(observed.argv, ['api', '-H', 'Accept: application/zip', 'repos/x/y/zip']);
    assert.equal(observed.gh_token, 'dispatch-token');
  });
});

test('node_failures_and_missing_credentials_fail_closed', () => {
  withTemporaryDirectory((directory) => {
    const [, log] = fakeGh(directory);
    const env = environment(directory, { GH_FAKE_LOG: log, GH_FAKE_EXIT: '1', GH_FAKE_STDERR: '  HTTP 404: Not Found\n' });
    const gateway = new transport.GhGateway({ readToken: 'read-token', dispatchToken: '', env });
    assert.throws(() => gateway.apiJson('user'), { name: 'ContractError', message: 'gh api user failed: HTTP 404: Not Found' });
    assert.throws(
      () => gateway.apiJson('user', { privileged: true }),
      new ContractError('no credential is available for this GitHub request; refusing to continue'),
    );
    assert.equal(gateway.dispatchIdentity(), null);
    assert.throws(
      () => gateway.dispatchWorkflow({ workflowFile: 'x.yml', ref: 'main', inputs: {} }),
      new ContractError('workflow dispatch requires an orchestrator credential'),
    );
    const dispatcher = new transport.GhGateway({ readToken: 'read-token', dispatchToken: 'dispatch-token', env });
    assert.throws(
      () => dispatcher.dispatchWorkflow({ workflowFile: 'x.yml', ref: 'main', inputs: {} }),
      new ContractError('dispatching x.yml failed: HTTP 404: Not Found'),
    );
    assert.equal(dispatcher.dispatchIdentity(), null);
  });
});

test('node_utc_now_uses_the_canonical_format', () => {
  const gateway = new transport.GhGateway({ readToken: 'r', dispatchToken: null, clock: () => new Date(Date.UTC(2026, 7, 30, 1, 2, 3, 999)) });
  assert.equal(gateway.utcNow(), '2026-08-30T01:02:03Z');
  assert.match(new transport.GhGateway({ readToken: 'r', dispatchToken: null }).utcNow(), /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/u);
  const slept = [];
  new transport.GhGateway({ readToken: 'r', dispatchToken: null, sleeper: (seconds) => slept.push(seconds) }).sleep(5);
  assert.deepEqual(slept, [5]);
});
