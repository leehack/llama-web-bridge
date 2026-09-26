// Tests of scripts/release/orchestrator/run_names.mjs, one test per test method
// of scripts/release_orchestrator_run_names_test.py, with the same names and
// assertions (a Python subTest is one loop iteration).

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { ContractError } from '../../../scripts/release/contract.mjs';
import { pySplitlines } from '../../../scripts/release/json.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as runNames from '../../../scripts/release/orchestrator/run_names.mjs';
import {
  ADVANCED_BRIDGE_SHA,
  BRIDGE_SHA,
  CANDIDATE_RUN_ID,
  NATIVE_MANIFEST_SHA,
  QUALIFICATION_RUN_ID,
  makeProvenance,
} from './fixtures.mjs';

const WORKFLOWS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../.github/workflows');

// --- CorrelationTest -------------------------------------------------------------

test('test_deterministic_correlation_id_ignores_orchestration_only_main', () => {
  const first = runNames.computeCorrelationId(makeProvenance());
  const second = runNames.computeCorrelationId(makeProvenance({ bridgeSourceSha: ADVANCED_BRIDGE_SHA }));
  assert.equal(first, second);
  assert.equal(first, `auto-stable-v0.2.0-${NATIVE_MANIFEST_SHA.slice(0, 16)}-build-${BRIDGE_SHA.slice(0, 16)}`);
});

test('test_correlation_id_changes_with_governed_build_source', () => {
  const other = runNames.computeCorrelationId(makeProvenance({
    bridgeSourceSha: ADVANCED_BRIDGE_SHA,
    bridgeBuildSha: ADVANCED_BRIDGE_SHA,
  }));
  assert.notEqual(runNames.computeCorrelationId(makeProvenance()), other);
});

test('test_correlation_id_changes_with_native_manifest', () => {
  const other = runNames.computeCorrelationId(makeProvenance({ nativeManifestSha256: 'b'.repeat(64) }));
  assert.notEqual(runNames.computeCorrelationId(makeProvenance()), other);
});

test('test_candidate_run_name_round_trips_persisted_binding', () => {
  const correlationId = runNames.computeCorrelationId(makeProvenance());
  const binding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.40', releaseRebuild: 0 });
  const name = runNames.candidateRunName(correlationId, binding);
  assert.equal(name, `bridge-candidate ${correlationId} source:${BRIDGE_SHA} tag:v0.1.40 rebuild:0`);
  assert.ok(runNames.parseCandidateRunName(name, correlationId).equals(binding));
});

test('test_foreign_candidate_run_name_is_not_adopted', () => {
  const correlationId = runNames.computeCorrelationId(makeProvenance());
  const foreignNames = [
    `bridge-candidate other-correlation source:${BRIDGE_SHA} tag:v0.1.40 rebuild:0`,
    'bridge-candidate other-correlation source:not-a-sha tag:v0.1.40 rebuild:0',
    'bridge-candidate other-correlation malformed',
    `bridge-candidate ${correlationId}-foreign source:not-a-sha tag:v0.1.40 rebuild:bad`,
    `bridge-candidate-other ${correlationId} malformed`,
    'some-unrelated-workflow-run',
  ];
  for (const foreignName of foreignNames) {
    assert.equal(runNames.parseCandidateRunName(foreignName, correlationId), null, foreignName);
  }
});

test('test_malformed_candidate_run_name_claiming_correlation_fails_closed', () => {
  const correlationId = runNames.computeCorrelationId(makeProvenance());
  const malformedNames = [
    `bridge-candidate ${correlationId} source:not-a-sha tag:v0.1.40 rebuild:0`,
    `bridge-candidate ${correlationId} source:${BRIDGE_SHA} tag:v0.1.40 rebuild:not-a-number`,
    `bridge-candidate ${correlationId} source:${BRIDGE_SHA} tag:v0.1.40 rebuild:-1`,
    `bridge-candidate ${correlationId} source:${BRIDGE_SHA} tag:v0.1.40 rebuild:01`,
    `bridge-candidate ${correlationId} source:${BRIDGE_SHA} tag:invalid_tag rebuild:0`,
    `bridge-candidate ${correlationId}`,
    `bridge-candidate ${correlationId} source:${BRIDGE_SHA}`,
    `bridge-candidate ${correlationId} source:${BRIDGE_SHA} tag:v0.1.40`,
    `bridge-candidate ${correlationId} malformed`,
    `bridge-candidate ${correlationId} source:${BRIDGE_SHA} tag:v0.1.40 rebuild:0\x00`,
    `bridge-candidate ${correlationId} ${'x'.repeat(200)}`,
  ];
  for (const malformedName of malformedNames) {
    assert.throws(() => runNames.parseCandidateRunName(malformedName, correlationId), ContractError, malformedName);
  }
});

test('test_generated_run_name_enforces_a_conservative_length_bound', () => {
  assert.throws(() => runNames.requireRunName('x'.repeat(runNames.MAX_RUN_NAME_CHARACTERS + 1)), ContractError);
});

// --- WorkflowRunNameContractTest: the rendered workflow run-name must be
// exactly what the parser expects. -----------------------------------------------

function render(workflow, inputs) {
  for (const line of pySplitlines(fs.readFileSync(path.join(WORKFLOWS, workflow), 'utf8'))) {
    if (!line.startsWith('run-name: ')) continue;
    let rendered = line.slice('run-name: '.length);
    for (const [name, value] of Object.entries(inputs)) {
      rendered = rendered.replaceAll(`\${{ inputs.${name} }}`, value);
    }
    assert.ok(!rendered.includes('${{'), rendered);
    return rendered;
  }
  throw new assert.AssertionError({ message: `${workflow} declares no run-name` });
}

test('test_candidate_workflow_renders_the_parsed_run_name', () => {
  const correlationId = runNames.computeCorrelationId(makeProvenance());
  const binding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.40', releaseRebuild: 0 });
  const rendered = render(model.CANDIDATE_WORKFLOW_FILE, {
    orchestrator_correlation_id: correlationId,
    bridge_source_sha: BRIDGE_SHA,
    release_tag: 'v0.1.40',
    release_rebuild: '0',
  });
  assert.equal(rendered, runNames.candidateRunName(correlationId, binding));
  assert.ok(runNames.parseCandidateRunName(rendered, correlationId).equals(binding));
});

test('test_qualification_workflow_renders_the_parsed_run_name', () => {
  const correlationId = runNames.computeCorrelationId(makeProvenance());
  const rendered = render(model.QUALIFICATION_WORKFLOW_FILE, {
    orchestrator_correlation_id: correlationId,
    candidate_run_id: CANDIDATE_RUN_ID,
  });
  assert.equal(rendered, runNames.qualificationRunName(correlationId, CANDIDATE_RUN_ID));
});

test('test_publish_workflow_renders_the_parsed_run_name', () => {
  const correlationId = runNames.computeCorrelationId(makeProvenance());
  const binding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.40', releaseRebuild: 0 });
  const rendered = render(model.PUBLISH_WORKFLOW_FILE, {
    orchestrator_correlation_id: correlationId,
    candidate_run_id: CANDIDATE_RUN_ID,
    qualification_run_id: QUALIFICATION_RUN_ID,
    bridge_source_sha: BRIDGE_SHA,
    release_tag: 'v0.1.40',
    release_rebuild: '0',
  });
  assert.equal(rendered, runNames.publishRunName(correlationId, CANDIDATE_RUN_ID, QUALIFICATION_RUN_ID, binding));
  const [candidateRunId, qualificationRunId, parsed] = runNames.parsePublishRunName(rendered, correlationId);
  assert.deepEqual([candidateRunId, qualificationRunId], [CANDIDATE_RUN_ID, QUALIFICATION_RUN_ID]);
  assert.ok(parsed.equals(binding));
});
