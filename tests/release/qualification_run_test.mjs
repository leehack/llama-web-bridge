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
    assert.deepEqual(missingHarness, { stdout: '', stderr: 'error: harness source is missing: generate_release_manifest.py\n', status: 1 });

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
