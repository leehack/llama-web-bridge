// Tests of scripts/release/orchestrator/planner.mjs, one test per test method
// of scripts/release_orchestrator_planner_test.py, with the same names and
// assertions.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { ASSETS_REPOSITORY, ContractError } from '../../../scripts/release/contract.mjs';
import { pySplitlines, pyStrip } from '../../../scripts/release/json.mjs';
import * as orchestrator from '../../../scripts/release/orchestrator/cli.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as planner from '../../../scripts/release/orchestrator/planner.mjs';
import * as runNames from '../../../scripts/release/orchestrator/run_names.mjs';
import {
  BRIDGE_SHA,
  CANDIDATE_RUN_ID,
  NATIVE_MANIFEST_SHA,
  QUALIFICATION_RUN_ID,
  UPSTREAM_COMMIT,
  makeProvenance,
} from './fixtures.mjs';

const WORKFLOWS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../.github/workflows');

// Read a workflow's exact workflow_dispatch input names, with the Python
// suite's dependency-free line scan.
function declaredWorkflowInputs(workflow) {
  const names = [];
  let inDispatch = false;
  let inInputs = false;
  for (const line of pySplitlines(fs.readFileSync(workflow, 'utf8'))) {
    const stripped = pyStrip(line);
    if (!stripped || stripped.startsWith('#')) continue;
    if (!inDispatch) {
      inDispatch = line === '  workflow_dispatch:';
      continue;
    }
    if (!inInputs) {
      if (line === '    inputs:') {
        inInputs = true;
        continue;
      }
      if (!line.startsWith('    ')) break;
      continue;
    }
    if (!line.startsWith('      ')) break;
    if (line.startsWith('       ')) continue;
    if (!stripped.endsWith(':')) throw new assert.AssertionError({ message: `unexpected input declaration: ${line}` });
    names.push(stripped.slice(0, -1));
  }
  if (names.length === 0) throw new assert.AssertionError({ message: `${path.basename(workflow)} declares no workflow_dispatch inputs` });
  return names;
}

const setOf = (values) => new Set(values);
const without = (values, ...removed) => new Set([...values].filter((value) => !removed.includes(value)));

function setUp() {
  const provenance = makeProvenance();
  return {
    provenance,
    correlationId: runNames.computeCorrelationId(provenance),
    binding: new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.40', releaseRebuild: 0 }),
  };
}

function plan(self, observation) {
  return planner.planPipeline({
    provenance: self.provenance,
    correlationId: self.correlationId,
    observation: new model.PipelineObservation(observation),
  });
}

// --- DispatchInputContractTest: a dispatch whose keys drift from the workflow
// is rejected at build time. ---------------------------------------------------------

test('test_candidate_dispatch_inputs_are_exactly_the_declared_inputs', () => {
  const self = setUp();
  const declared = declaredWorkflowInputs(path.join(WORKFLOWS, model.CANDIDATE_WORKFLOW_FILE));
  assert.deepEqual(setOf(declared), setOf(planner.CANDIDATE_DISPATCH_INPUTS));
  const inputs = planner.dispatchInputsForCandidate(self.provenance, self.correlationId, self.binding);
  // The live governance proof supplies the immutability assertion, so the
  // planner must not pre-declare it.
  assert.deepEqual(setOf(Object.keys(inputs)), without(declared, 'assets_immutable_releases_enabled'));
});

test('test_publish_dispatch_inputs_are_exactly_the_declared_inputs', () => {
  const self = setUp();
  const declared = declaredWorkflowInputs(path.join(WORKFLOWS, model.PUBLISH_WORKFLOW_FILE));
  assert.deepEqual(setOf(declared), setOf(planner.PUBLISH_DISPATCH_INPUTS));
  const inputs = planner.dispatchInputsForPublish(self.provenance, self.correlationId, self.binding, CANDIDATE_RUN_ID, QUALIFICATION_RUN_ID);
  // The live environment-policy proof supplies this approval assertion.
  assert.deepEqual(setOf(Object.keys(inputs)), without(declared, 'publish_approved'));
});

test('test_qualification_dispatch_inputs_are_exactly_the_declared_inputs', () => {
  const self = setUp();
  const declared = declaredWorkflowInputs(path.join(WORKFLOWS, model.QUALIFICATION_WORKFLOW_FILE));
  assert.deepEqual(setOf(declared), setOf(planner.QUALIFICATION_DISPATCH_INPUTS));
  const inputs = planner.dispatchInputsForQualification(self.correlationId, CANDIDATE_RUN_ID);
  assert.deepEqual(setOf(Object.keys(inputs)), setOf(declared));
});

test('test_missing_dispatch_input_fails_closed', () => {
  const self = setUp();
  const inputs = planner.dispatchInputsForPublish(self.provenance, self.correlationId, self.binding, CANDIDATE_RUN_ID, QUALIFICATION_RUN_ID);
  assert.throws(() => planner.requireExactDispatchInputs(model.PUBLISH_WORKFLOW_FILE, inputs), ContractError);
});

test('test_unknown_dispatch_input_fails_closed', () => {
  const self = setUp();
  const inputs = planner.dispatchInputsForCandidate(self.provenance, self.correlationId, self.binding);
  inputs.assets_immutable_releases_enabled = 'true';
  inputs.unexpected = 'value';
  assert.throws(() => planner.requireExactDispatchInputs(model.CANDIDATE_WORKFLOW_FILE, inputs), ContractError);
});

test('test_undispatchable_workflow_fails_closed', () => {
  assert.throws(() => planner.requireExactDispatchInputs('ci.yml', { candidate_run_id: CANDIDATE_RUN_ID }), ContractError);
});

test('test_non_string_dispatch_input_fails_closed', () => {
  const self = setUp();
  const inputs = planner.dispatchInputsForCandidate(self.provenance, self.correlationId, self.binding);
  inputs.assets_immutable_releases_enabled = 'true';
  inputs.release_rebuild = 0;
  assert.throws(() => planner.requireExactDispatchInputs(model.CANDIDATE_WORKFLOW_FILE, inputs), ContractError);
});

// --- PlanTest: exhaustive pure-state-machine coverage over already-proven
// evidence. -------------------------------------------------------------------------

test('test_new_provenance_plans_exactly_one_candidate_dispatch', () => {
  const self = setUp();
  const result = plan(self, { freshBinding: self.binding });
  assert.equal(result.action, model.OrchestrationAction.DISPATCH_CANDIDATE);
  assert.equal(result.dispatchWorkflow, model.CANDIDATE_WORKFLOW_FILE);
  const inputs = result.dispatchInputs ?? {};
  assert.equal(inputs.orchestrator_correlation_id, self.correlationId);
  assert.equal(inputs.bridge_source_sha, BRIDGE_SHA);
  assert.equal(inputs.upstream_tag, 'v0.2.0');
  assert.equal(inputs.upstream_commit, UPSTREAM_COMMIT);
  assert.equal(inputs.native_release_tag, 'v0.2.0');
  assert.equal(inputs.native_manifest_sha256, NATIVE_MANIFEST_SHA);
  assert.equal(inputs.release_tag, 'v0.1.40');
  assert.equal(inputs.release_rebuild, '0');
  // Everything the planner can know without a live read, and nothing else.
  assert.deepEqual(setOf(Object.keys(inputs)), without(planner.CANDIDATE_DISPATCH_INPUTS, 'assets_immutable_releases_enabled'));
});

test('test_published_provenance_is_an_exact_noop', () => {
  const self = setUp();
  const result = plan(self, {
    published: new model.PublishedRelease({
      releaseId: 4242,
      releaseTarget: new model.ReleaseTarget({ releaseTag: 'v0.1.40', releaseRebuild: 0 }),
      binding: self.binding,
      publishedAt: '2026-08-20T03:24:11Z',
    }),
  });
  assert.equal(result.action, model.OrchestrationAction.NOOP);
  assert.equal(result.dispatchWorkflow, null);
  assert.ok(result.reason.toLowerCase().includes('already published'));
});

test('test_in_flight_candidate_blocks_duplicate_dispatch', () => {
  const self = setUp();
  const result = plan(self, { freshBinding: self.binding, candidateInFlightRunId: '501', binding: self.binding });
  assert.equal(result.action, model.OrchestrationAction.IN_FLIGHT);
  assert.equal(result.inFlightWorkflow, model.CANDIDATE_WORKFLOW_FILE);
  assert.equal(result.inFlightRunId, '501');
  assert.equal(result.dispatchWorkflow, null);
});

test('test_candidate_ready_plans_exactly_one_hosted_qualification', () => {
  const self = setUp();
  const result = plan(self, { binding: self.binding, candidateRunId: CANDIDATE_RUN_ID });
  assert.equal(result.action, model.OrchestrationAction.DISPATCH_QUALIFICATION);
  assert.equal(result.candidateRunId, CANDIDATE_RUN_ID);
  assert.equal(result.dispatchWorkflow, model.QUALIFICATION_WORKFLOW_FILE);
  assert.equal(result.dispatchRunName, runNames.qualificationRunName(self.correlationId, CANDIDATE_RUN_ID));
  assert.deepEqual(result.dispatchInputs, {
    orchestrator_correlation_id: self.correlationId,
    candidate_run_id: CANDIDATE_RUN_ID,
  });
  assert.equal(result.qualificationRunId, null);
  // No routine state may require a maintainer-supplied payload or an owner
  // workflow_dispatch continuation to advance.
  assert.ok(!result.reason.toLowerCase().includes('maintainer'));
  assert.ok(!result.reason.toLowerCase().includes('attestation'));
});

test('test_in_flight_qualification_blocks_duplicate_publish', () => {
  const self = setUp();
  const result = plan(self, {
    binding: self.binding,
    candidateRunId: CANDIDATE_RUN_ID,
    qualificationInFlightRunId: QUALIFICATION_RUN_ID,
  });
  assert.equal(result.action, model.OrchestrationAction.IN_FLIGHT);
  assert.equal(result.inFlightWorkflow, model.QUALIFICATION_WORKFLOW_FILE);
});

test('test_candidate_and_qualification_ready_plan_publish', () => {
  const self = setUp();
  const result = plan(self, {
    binding: self.binding,
    candidateRunId: CANDIDATE_RUN_ID,
    qualificationRunId: QUALIFICATION_RUN_ID,
  });
  assert.equal(result.action, model.OrchestrationAction.DISPATCH_PUBLISH);
  assert.equal(result.dispatchWorkflow, model.PUBLISH_WORKFLOW_FILE);
  const inputs = result.dispatchInputs ?? {};
  assert.equal(inputs.candidate_run_id, CANDIDATE_RUN_ID);
  assert.equal(inputs.qualification_run_id, QUALIFICATION_RUN_ID);
  assert.equal(inputs.release_tag, 'v0.1.40');
  assert.equal(inputs.release_rebuild, '0');
  assert.ok(!Object.hasOwn(inputs, 'publish_approved'));
  assert.equal(inputs.assets_repo, ASSETS_REPOSITORY);
  assert.equal(inputs.bridge_source_sha, BRIDGE_SHA);
  assert.deepEqual(setOf(Object.keys(inputs)), without(planner.PUBLISH_DISPATCH_INPUTS, 'publish_approved'));
});

test('test_in_flight_publish_blocks_duplicate_publish', () => {
  const self = setUp();
  const result = plan(self, {
    binding: self.binding,
    candidateRunId: CANDIDATE_RUN_ID,
    qualificationRunId: QUALIFICATION_RUN_ID,
    publishInFlightRunId: '701',
  });
  assert.equal(result.action, model.OrchestrationAction.IN_FLIGHT);
  assert.equal(result.inFlightWorkflow, model.PUBLISH_WORKFLOW_FILE);
  assert.equal(result.inFlightRunId, '701');
});

test('test_successful_publish_without_immutable_release_fails_closed', () => {
  const self = setUp();
  assert.throws(() => plan(self, {
    binding: self.binding,
    candidateRunId: CANDIDATE_RUN_ID,
    qualificationRunId: QUALIFICATION_RUN_ID,
    publishSucceededRunId: '701',
  }), ContractError);
});

test('test_publish_retry_after_failed_publish_reuses_exact_state', () => {
  const self = setUp();
  const result = plan(self, {
    binding: self.binding,
    candidateRunId: CANDIDATE_RUN_ID,
    qualificationRunId: QUALIFICATION_RUN_ID,
    publishRetry: true,
  });
  assert.equal(result.action, model.OrchestrationAction.DISPATCH_PUBLISH);
  assert.equal(result.candidateRunId, CANDIDATE_RUN_ID);
  assert.equal(result.qualificationRunId, QUALIFICATION_RUN_ID);
  // A retry must be visible as a retry, not reported as a first attempt.
  assert.ok(result.reason.includes('retry'));
});

test('test_planner_never_pre_asserts_immutable_release_governance', () => {
  const self = setUp();
  const result = plan(self, { freshBinding: self.binding });
  assert.equal(result.action, model.OrchestrationAction.DISPATCH_CANDIDATE);
  assert.ok(!Object.hasOwn(result.dispatchInputs, 'assets_immutable_releases_enabled'));
});

test('test_planner_never_pre_asserts_publication_approval', () => {
  const self = setUp();
  const result = plan(self, {
    binding: self.binding,
    candidateRunId: CANDIDATE_RUN_ID,
    qualificationRunId: QUALIFICATION_RUN_ID,
  });
  assert.ok(!Object.hasOwn(result.dispatchInputs, 'publish_approved'));
});

test('test_missing_binding_for_ready_candidate_fails_closed', () => {
  const self = setUp();
  assert.throws(() => plan(self, { candidateRunId: CANDIDATE_RUN_ID }), ContractError);
});

test('test_summary_reports_the_exact_action', () => {
  const self = setUp();
  const summary = orchestrator.renderStepSummary(plan(self, { freshBinding: self.binding }));
  assert.ok(summary.includes('### Stable Web bridge release orchestration'));
  assert.ok(summary.includes('dispatch_candidate'));
  assert.ok(summary.includes(model.CANDIDATE_WORKFLOW_FILE));
  assert.ok(summary.includes(self.correlationId));
});
