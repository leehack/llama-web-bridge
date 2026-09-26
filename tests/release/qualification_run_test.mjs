// Tests of scripts/release/qualification.mjs: workflow-run and artifact
// provenance, and the command line, one test per test method of
// scripts/release_qualification_test.py with the same name and assertions
// (see qualification_fixtures.mjs for the mapping), plus the CLI end to end.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';

import { ArgparseExit, runPythonStyleMainAsync } from '../../scripts/release/cli.mjs';
import { ContractError } from '../../scripts/release/contract.mjs';
import { pyJsonDumps } from '../../scripts/release/json.mjs';
import {
  CANDIDATE_ARTIFACT_NAME,
  CANDIDATE_ALLOWED_MEMBERS,
  CANDIDATE_WORKFLOW_PATH,
  QUALIFICATION_WORKFLOW_PATH,
  canonicalJson,
  harnessSourceSha256,
  main,
  parseArgs,
  pyDeepCopy,
  validateArtifactInventory,
  validateWorkflowRun,
} from '../../scripts/release/qualification.mjs';
import {
  CANDIDATE_RUN_ID,
  DEFAULT_HEAD_BRANCH,
  DEFAULT_HEAD_SHA,
  SCRIPTS_DIR,
  artifactInventory,
  makeTempDir,
  setUp,
  workflowRun,
} from './qualification_fixtures.mjs';
import { ZIP_DEFLATED, writeZip } from './zip_fixture.mjs';

function raises(fn, message = undefined) {
  let caught;
  assert.throws(fn, (error) => {
    caught = error;
    return error instanceof ContractError;
  }, message);
  return caught.message;
}

function validateCandidateRun(run, options = {}) {
  return validateWorkflowRun(run, {
    expectedRunId: CANDIDATE_RUN_ID,
    expectedWorkflowPath: CANDIDATE_WORKFLOW_PATH,
    expectedHeadBranch: DEFAULT_HEAD_BRANCH,
    ...options,
  });
}

function validateCandidateInventory(inventory) {
  return validateArtifactInventory(inventory, { expectedRunId: CANDIDATE_RUN_ID, expectedName: CANDIDATE_ARTIFACT_NAME });
}

test('test_valid_candidate_run_accepted', () => {
  assert.equal(validateCandidateRun(workflowRun(), { expectedRunAttempt: 1 }), DEFAULT_HEAD_SHA);
  assert.equal(validateCandidateInventory(artifactInventory()), 7);
});

test('test_unrelated_or_unsuccessful_run_rejected', () => {
  for (const overrides of [
    { id: 12345 },
    { repository: { full_name: 'attacker/llama-web-bridge' } },
    { head_repository: { full_name: 'attacker/fork' } },
    { path: '.github/workflows/ci.yml' },
    { event: 'push' },
    { event: 'pull_request' },
    { status: 'in_progress' },
    { conclusion: 'failure' },
    { conclusion: null },
    { head_branch: 'feature/attack' },
    { head_branch: null },
    { head_sha: 'not-a-sha' },
    { head_sha: null },
  ]) {
    raises(() => validateCandidateRun(workflowRun(overrides)), JSON.stringify(overrides));
  }
});

test('test_candidate_run_must_be_the_first_build_attempt', () => {
  for (const runAttempt of [2, 0, true, '1', null]) {
    raises(() => validateCandidateRun(workflowRun({ run_attempt: runAttempt }), { expectedRunAttempt: 1 }), String(runAttempt));
  }
});

test('test_run_attempt_one_is_mandatory_even_without_a_caller_override', () => {
  for (const runAttempt of [2, 0, true, '1', null]) {
    raises(() => validateCandidateRun(workflowRun({ run_attempt: runAttempt })), String(runAttempt));
  }
});

test('test_qualification_run_must_be_the_qualification_workflow', () => {
  const message = raises(() => validateCandidateRun(workflowRun({ path: CANDIDATE_WORKFLOW_PATH }), {
    expectedWorkflowPath: QUALIFICATION_WORKFLOW_PATH,
  }));
  assert.ok(message.includes(QUALIFICATION_WORKFLOW_PATH));
});

test('test_run_id_and_head_sha_must_be_well_formed', () => {
  for (const runId of ['0', '-1', '01', 'abc', '']) {
    raises(() => validateCandidateRun(workflowRun(), { expectedRunId: runId }), runId);
  }
  raises(() => validateCandidateRun(workflowRun(), { expectedHeadBranch: '' }));
});

test('test_missing_duplicate_expired_or_foreign_artifact_rejected', () => {
  const [artifact] = artifactInventory().artifacts;
  for (const overrides of [
    { artifacts: [] },
    { artifacts: [artifact, pyDeepCopy(artifact)] },
    { artifacts: [{ ...artifact, expired: true }] },
    { artifacts: [{ ...artifact, workflow_run: { id: 1 } }] },
    { artifacts: [{ ...artifact, id: '7' }] },
    { artifacts: [{ ...artifact, id: true }] },
    { artifacts: [{ ...artifact, id: 0 }] },
    { artifacts: [{ name: 5 }] },
    { artifacts: 'none' },
  ]) {
    raises(() => validateCandidateInventory(artifactInventory(overrides)), JSON.stringify(overrides));
  }
});

test('test_truncated_artifact_inventory_rejected', () => {
  assert.ok(raises(() => validateCandidateInventory(artifactInventory({ total_count: 2 }))).includes('truncated'));
});

test('test_wrong_workflow_run_actor_rejected', () => {
  for (const actor of [{ login: 'attacker' }, { login: 'someone-else' }, null, {}]) {
    raises(() => validateCandidateRun(workflowRun({ actor }), { expectedRunAttempt: 1 }), JSON.stringify(actor));
  }
});

test('test_wrong_workflow_run_triggering_actor_rejected', () => {
  for (const actor of [{ login: 'attacker' }, { login: 'someone-else' }, null, {}]) {
    raises(() => validateCandidateRun(workflowRun({ triggering_actor: actor }), { expectedRunAttempt: 1 }), JSON.stringify(actor));
  }
});

test('test_artifact_name_alone_does_not_prove_provenance', () => {
  const inventory = artifactInventory({
    artifacts: [{ id: 7, name: CANDIDATE_ARTIFACT_NAME, expired: false, workflow_run: { id: 424242 } }],
  });
  assert.ok(raises(() => validateCandidateInventory(inventory)).includes('does not belong to run'));
});

test('test_qualify_rejects_an_unprovenanced_local_candidate', () => {
  const tmp = makeTempDir();
  try {
    assert.throws(
      () => parseArgs(['qualify', '--candidate-dist', path.join(tmp, 'candidate'), '--output-attestation', path.join(tmp, 'a.json')]),
      (error) => error instanceof ArgparseExit && error.status === 2,
    );
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('test_verify_run_cli_defaults_to_the_mandatory_first_attempt', () => {
  const { args } = parseArgs([
    'verify-run',
    '--run-json', 'run.json',
    '--artifacts-json', 'artifacts.json',
    '--run-id', CANDIDATE_RUN_ID,
    '--workflow-path', CANDIDATE_WORKFLOW_PATH,
    '--head-branch', DEFAULT_HEAD_BRANCH,
    '--artifact-name', CANDIDATE_ARTIFACT_NAME,
  ]);
  assert.equal(args.runAttempt, 1);
});

// --- Node-only: exact Python messages and the CLI end to end -------------------------

test('workflow run and inventory errors carry Python repr text', () => {
  assert.equal(
    raises(() => validateCandidateRun(workflowRun({ id: String(CANDIDATE_RUN_ID) }))),
    `workflow run id mismatch: expected ${CANDIDATE_RUN_ID}, got '${CANDIDATE_RUN_ID}'`,
  );
  // bool is an int to isinstance(); str(True) never matches a run id.
  assert.equal(raises(() => validateCandidateRun(workflowRun({ id: true }))), `workflow run id mismatch: expected ${CANDIDATE_RUN_ID}, got True`);
  assert.equal(raises(() => validateCandidateRun(workflowRun({ conclusion: null }))), 'workflow run conclusion must be success, got None');
  assert.equal(
    raises(() => validateCandidateRun(workflowRun({ run_attempt: 2 }))),
    'workflow run attempt must be 1, got 2',
  );
  assert.equal(raises(() => validateCandidateRun(workflowRun(), { expectedRunAttempt: true })), 'expected run attempt must be exactly 1');
  assert.throws(() => validateCandidateRun(workflowRun(), { expectedRunId: 1 }), (error) => error.pyType === 'TypeError');
  // str(run.get("id")) is compared, so a string id in the inventory passes.
  assert.equal(validateCandidateInventory(artifactInventory({
    artifacts: [{ id: 7, name: CANDIDATE_ARTIFACT_NAME, expired: false, workflow_run: { id: CANDIDATE_RUN_ID } }],
  })), 7);
  assert.equal(
    raises(() => validateCandidateInventory(artifactInventory({ total_count: 2 }))),
    'artifact inventory is truncated; all run artifacts must be inspected (1 records for total_count=2)',
  );
  assert.equal(
    raises(() => validateCandidateInventory(artifactInventory({ artifacts: [], total_count: 0 }))),
    `run ${CANDIDATE_RUN_ID} must expose exactly one 'exact-webgpu-bridge-dist' artifact, found 0`,
  );
});

async function cli(argv) {
  return runPythonStyleMainAsync(main, argv);
}

test('the CLI subcommands print Python bytes and errors', async () => {
  const context = setUp();
  try {
    const { tmp } = context;
    const runJson = path.join(tmp, 'run.json');
    const artifactsJson = path.join(tmp, 'artifacts.json');
    fs.writeFileSync(runJson, pyJsonDumps(workflowRun()));
    fs.writeFileSync(artifactsJson, pyJsonDumps(artifactInventory()));
    const verifyRun = [
      'verify-run', '--run-json', runJson, '--artifacts-json', artifactsJson, '--run-id', CANDIDATE_RUN_ID,
      '--workflow-path', CANDIDATE_WORKFLOW_PATH, '--head-branch', DEFAULT_HEAD_BRANCH, '--artifact-name', CANDIDATE_ARTIFACT_NAME,
    ];
    assert.deepEqual(await cli(verifyRun), {
      stdout: `{"artifact_id": 7, "head_sha": "${DEFAULT_HEAD_SHA}", "verified": true}\n`,
      stderr: '',
      status: 0,
    });
    assert.deepEqual(await cli([...verifyRun, '--run-attempt', '2']), {
      stdout: '', stderr: 'error: expected run attempt must be exactly 1\n', status: 1,
    });
    assert.deepEqual(await cli([...verifyRun.slice(0, 2), path.join(tmp, 'absent.json'), ...verifyRun.slice(3)]), {
      stdout: '', stderr: `error: [Errno 2] No such file or directory: '${path.join(tmp, 'absent.json')}'\n`, status: 1,
    });
    fs.writeFileSync(runJson, Buffer.from([0xff]));
    const undecodable = await cli(verifyRun);
    assert.equal(undecodable.status, 1);
    assert.match(undecodable.stderr, /^Traceback \(most recent call last\):\nUnicodeDecodeError: 'utf-8' codec can't decode byte 0xff/);

    const attestation = path.join(tmp, 'attestation.json');
    fs.writeFileSync(attestation, canonicalJson(context.attestation));
    const verified = await cli([
      'verify-attestation', '--attestation', attestation, '--candidate-dist', `${context.candidate}/`,
      '--candidate-artifact-id', '+7', '--harness-dir', SCRIPTS_DIR, '--release-rebuild', '1',
    ]);
    assert.equal(verified.status, 0, verified.stderr);
    assert.equal(verified.stdout, `${pyJsonDumps(context.verify(context.attestation), { indent: 2, sortKeys: true })}\n`);
    const mismatch = await cli(['verify-attestation', '--attestation', attestation, '--release-rebuild', '2']);
    assert.deepEqual(mismatch, { stdout: '', stderr: 'error: attestation release_rebuild mismatch: expected 2, got 1\n', status: 1 });
    const missingHarness = await cli(['verify-attestation', '--attestation', attestation, '--harness-dir', tmp]);
    assert.deepEqual(missingHarness, { stdout: '', stderr: 'error: harness source is missing: release/archive.mjs\n', status: 1 });

    assert.deepEqual(await cli(['candidate-fingerprint', '--candidate-dist', context.candidate]), {
      stdout: `${context.fingerprint}\n`, stderr: '', status: 0,
    });
    assert.deepEqual(await cli(['harness-digest', '--harness-dir', SCRIPTS_DIR]), {
      stdout: `${harnessSourceSha256(SCRIPTS_DIR)}\n`, stderr: '', status: 0,
    });
    const usage = await cli(['harness-digest']);
    assert.equal(usage.status, 2);
    assert.match(usage.stderr, /harness-digest: error: the following arguments are required: --harness-dir\n$/);
    // Error lines are sanitized like every other diagnostic.
    const leaked = await cli(['candidate-fingerprint', '--candidate-dist', path.join(tmp, 'api_key=hunter2')]);
    assert.deepEqual(leaked, { stdout: '', stderr: `error: candidate directory does not exist: ${tmp}/<redacted-credential>\n`, status: 1 });
  } finally {
    context.cleanup();
  }
});

// extract-artifact replaced the workflows' Python heredocs (the
// _extract_flat_artifact_archive calls and the inline zipfile extractor of
// the prequalification record): silent on success, `error: <message>` and
// exit status 1 on any rejection, with nothing left in the destination.
test('extract-artifact extracts each artifact type fail-closed', async () => {
  const context = setUp();
  try {
    const { tmp } = context;
    let counter = 0;
    const extract = async (type, entries, options) => {
      counter += 1;
      const archive = writeZip(path.join(tmp, `archive-${counter}.zip`), entries, options);
      const destination = path.join(tmp, `out-${counter}`);
      const result = await cli(['extract-artifact', '--type', type, '--archive', archive, '--destination', destination]);
      return { result, destination };
    };
    const ok = { stdout: '', stderr: '', status: 0 };

    const candidateFiles = [...CANDIDATE_ALLOWED_MEMBERS].map((name) => [name, fs.readFileSync(path.join(context.candidate, name))]);
    const candidate = await extract('candidate', candidateFiles, { compression: ZIP_DEFLATED });
    assert.deepEqual(candidate.result, ok);
    for (const [name, data] of candidateFiles) assert.deepEqual(fs.readFileSync(path.join(candidate.destination, name)), data, name);
    assert.deepEqual(fs.readdirSync(candidate.destination).sort(), [...CANDIDATE_ALLOWED_MEMBERS].sort());

    const attestationText = canonicalJson(context.attestation);
    const attestation = await extract('attestation', [['qualification-attestation.json', attestationText]]);
    assert.deepEqual(attestation.result, ok);
    assert.equal(fs.readFileSync(path.join(attestation.destination, 'qualification-attestation.json'), 'utf8'), attestationText);

    const record = `${pyJsonDumps({ schema_version: 1, candidate_fingerprint: 'f'.repeat(64) }, { indent: 2 })}\n`;
    const prequalification = await extract('prequalification', [['candidate-prequalification.json', record]], { compression: ZIP_DEFLATED });
    assert.deepEqual(prequalification.result, ok);
    assert.deepEqual(fs.readdirSync(prequalification.destination), ['candidate-prequalification.json']);
    assert.equal(fs.readFileSync(path.join(prequalification.destination, 'candidate-prequalification.json'), 'utf8'), record);
    // Exactly at the 128 KiB member bound is accepted (stored: a repeated
    // byte deflates past the compression-ratio bound).
    const atBound = await extract('prequalification', [['candidate-prequalification.json', 'x'.repeat(128 * 1024)]]);
    assert.deepEqual(atBound.result, ok);

    // Every refusal of the inline extractor it replaced, and the flat
    // extractor's stricter ones.
    const name = 'candidate-prequalification.json';
    for (const [label, type, entries, options, expected] of [
      ['a second member', 'prequalification', [[name, record], ['extra.json', '{}']], undefined,
        'artifact archive end-of-central-directory member count must be exactly 1, got 2'],
      ['another name', 'prequalification', [['prequalification.json', record]], undefined,
        "artifact archive contains unauthorized member: 'prequalification.json'"],
      ['a directory', 'prequalification', [[`${name}/`, '']], undefined, 'is not a flat regular file'],
      ['a symlink', 'prequalification', [{ name, data: 'target', externalAttr: 0o120777 << 16 }], undefined, 'is not a flat regular file'],
      ['a member over 128 KiB', 'prequalification', [[name, 'x'.repeat(128 * 1024 + 1)]], { compression: ZIP_DEFLATED },
        `artifact archive member '${name}' uncompressed size 131073 exceeds bound 131072`],
      ['an archive over 1 MiB', 'prequalification', [[name, 'x'.repeat(1024 * 1024)]], undefined,
        'artifact archive size 1048736 exceeds bound 1048576: '],
      ['hidden metadata', 'prequalification', [{ name, data: record, extra: [0xca, 0xfe, 0, 0] }], undefined, 'carries hidden metadata'],
      ['an attestation as a candidate', 'candidate', [['qualification-attestation.json', attestationText]], undefined,
        'member count must be exactly 9, got 1'],
      ['an unknown type', 'bogus', [[name, record]], undefined, "unknown artifact type: 'bogus'"],
    ]) {
      const { result, destination } = await extract(type, entries, options);
      assert.equal(result.status, 1, label);
      assert.equal(result.stdout, '', label);
      assert.ok(result.stderr.startsWith('error: ') && result.stderr.includes(expected), `${label}: ${result.stderr}`);
      assert.deepEqual(fs.existsSync(destination) ? fs.readdirSync(destination) : [], [], label);
    }
    // An encrypted member (flag bit 0) is refused.
    const encrypted = writeZip(path.join(tmp, 'encrypted.zip'), [[name, record]]);
    const bytes = fs.readFileSync(encrypted);
    bytes.writeUInt16LE(1, 6);
    bytes.writeUInt16LE(1, bytes.indexOf('PK\x01\x02', 0, 'latin1') + 8);
    fs.writeFileSync(encrypted, bytes);
    const refused = await cli(['extract-artifact', '--type', 'prequalification', '--archive', encrypted, '--destination', path.join(tmp, 'encrypted')]);
    assert.equal(refused.status, 1);
    assert.match(refused.stderr, /^error: artifact archive member is encrypted: 'candidate-prequalification.json'\n$/);
    // The destination must be new or empty, as the workflows create it.
    const occupied = path.join(tmp, 'occupied');
    fs.mkdirSync(occupied);
    fs.writeFileSync(path.join(occupied, 'stale'), '');
    const archive = writeZip(path.join(tmp, 'again.zip'), [[name, record]]);
    assert.deepEqual(await cli(['extract-artifact', '--type', 'prequalification', '--archive', archive, '--destination', occupied]), {
      stdout: '', stderr: `error: artifact destination is not empty: ${occupied}\n`, status: 1,
    });
    for (const argv of [
      ['extract-artifact', '--archive', archive, '--destination', occupied],
      ['extract-artifact', '--type', 'candidate', '--destination', occupied],
      ['extract-artifact', '--type', 'candidate', '--archive', archive],
    ]) {
      const usage = await cli(argv);
      assert.equal(usage.status, 2);
      assert.match(usage.stderr, /extract-artifact: error: the following arguments are required: /);
    }
  } finally {
    context.cleanup();
  }
});

test('qualify parses the Python options and defaults', () => {
  const { command, args } = parseArgs([
    'qualify', '--candidate-run-id', '1', '--speech-model-path', 'a//b/', '--speech-mmproj-path', 'b',
    '--speech-audio-path', 'c', '--tts-model-path', 'd', '--tts-mmproj-path', 'e', '--output-attestation', 'f',
  ]);
  assert.equal(command, 'qualify');
  assert.deepEqual(args, {
    candidateRunId: '1',
    speechModelPath: 'a/b',
    speechMmprojPath: 'b',
    speechAudioPath: 'c',
    ttsModelPath: 'd',
    ttsMmprojPath: 'e',
    ttsMaxFrames: 24,
    speechTimeoutSeconds: 1800,
    ttsTimeoutSeconds: 1800,
    diagnosticsDir: null,
    outputAttestation: 'f',
  });
});
