// Tests of scripts/release/orchestrator/driver.mjs: governance and dispatch
// identity, and the candidate stage of advancePipeline. One test per test
// method of scripts/release_orchestrator_driver_test.py, with the same names
// and assertions; the other stages are in driver_<stage>_test.mjs.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

import { ASSETS_REPOSITORY, BRIDGE_REPOSITORY, ContractError } from '../../../scripts/release/contract.mjs';
import { pyIntFromString } from '../../../scripts/release/json.mjs';
import * as driver from '../../../scripts/release/orchestrator/driver.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as planner from '../../../scripts/release/orchestrator/planner.mjs';
import * as runNames from '../../../scripts/release/orchestrator/run_names.mjs';
import { CANDIDATE_ARTIFACT_NAME } from '../../../scripts/release/qualification.mjs';
import {
  ADVANCED_BRIDGE_SHA, ASSETS_TAG_COMMIT, BRIDGE_SHA, CANDIDATE_ARTIFACT_ID, CANDIDATE_RUN_ID, DEFAULT_BRANCH, FakeGateway,
  HEAD_SHA, QUALIFICATION_RUN_ID, artifactInventory, assetReleaseStub, directoryMembers, flatZip,
  makeProvenance, runPayload, runsResponse, withAdvancePipelineFixture, writeBridgeCandidate,
} from './fixtures.mjs';
import { onDispatch, runsKey } from './driver_fixtures.mjs';

// --- GovernanceAndDispatchIdentityTest ---------------------------------------------

function governanceSetUp() {
  const provenance = makeProvenance();
  return {
    provenance,
    correlationId: runNames.computeCorrelationId(provenance),
    gateway({ jsonRoutes = {}, ...options } = {}) {
      const routes = {
        [`repos/${ASSETS_REPOSITORY}/releases?per_page=100`]: [[]],
        [runsKey(model.CANDIDATE_WORKFLOW_FILE)]: runsResponse([]),
        ...jsonRoutes,
      };
      return new FakeGateway({ jsonRoutes: routes, ...options });
    },
  };
}

function withWorkspace(fn) {
  const workspace = fs.mkdtempSync(path.join(tmpdir(), 'sro-driver-'));
  try {
    return fn(workspace);
  } finally {
    fs.rmSync(workspace, { recursive: true, force: true });
  }
}

test('test_disabled_immutable_release_governance_fails_closed', () => {
  const gateway = governanceSetUp().gateway({ governance: { enabled: false, enforced_by_owner: false } });
  assert.throws(() => driver.requireImmutableReleaseGovernance(gateway), ContractError);
});

test('test_malformed_governance_response_fails_closed', () => {
  const gateway = governanceSetUp().gateway({ governance: { enabled: true } });
  assert.throws(() => driver.requireImmutableReleaseGovernance(gateway), ContractError);
});

test('test_live_governance_is_proven_not_asserted', () => {
  const proven = driver.requireImmutableReleaseGovernance(governanceSetUp().gateway());
  assert.equal(proven.repository, ASSETS_REPOSITORY);
  assert.equal(proven.enabled, true);
});

test('test_absent_dispatch_identity_blocks_without_dispatching', () => {
  const self = governanceSetUp();
  const gateway = self.gateway({ identity: null });
  const plan = withWorkspace((workspace) => driver.advancePipeline(gateway, { provenance: self.provenance, workspace }));
  assert.equal(plan.action, model.OrchestrationAction.BLOCKED);
  assert.deepEqual(gateway.dispatches, []);
  assert.ok(plan.reason.toLowerCase().includes('dispatch identity'));
});

test('test_non_owner_dispatch_identity_blocks_without_dispatching', () => {
  const self = governanceSetUp();
  const gateway = self.gateway({ identity: 'someone-else' });
  const plan = withWorkspace((workspace) => driver.advancePipeline(gateway, { provenance: self.provenance, workspace }));
  assert.equal(plan.action, model.OrchestrationAction.BLOCKED);
  assert.deepEqual(gateway.dispatches, []);
});

test('test_publish_approval_requires_live_environment_policy', () => {
  const self = governanceSetUp();
  const binding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.40', releaseRebuild: 0 });
  const plan = planner.planPipeline({
    provenance: self.provenance,
    correlationId: self.correlationId,
    observation: new model.PipelineObservation({
      binding,
      candidateRunId: CANDIDATE_RUN_ID,
      qualificationRunId: QUALIFICATION_RUN_ID,
    }),
  });
  const gateway = self.gateway();
  gateway.jsonRoutes[`repos/${BRIDGE_REPOSITORY}/environments/bridge-assets-publication`] = {
    name: 'bridge-assets-publication',
    can_admins_bypass: true,
    protection_rules: [{ type: 'branch_policy' }],
    deployment_branch_policy: { protected_branches: false, custom_branch_policies: true },
  };
  assert.throws(() => driver.executeDispatch(gateway, plan, {
    defaultBranch: DEFAULT_BRANCH,
    workflowPath: model.PUBLISH_WORKFLOW_PATH,
    dryRun: true,
  }), ContractError);
  assert.deepEqual(gateway.dispatches, []);
});

// --- AdvancePipelineTest: candidate stage ------------------------------------------

test('test_first_run_dispatches_exactly_one_candidate_with_structured_inputs', withAdvancePipelineFixture((self) => {
  const dispatched = runPayload({
    runId: '501', path: model.CANDIDATE_WORKFLOW_PATH, runName: self.candidateName, status: 'in_progress', conclusion: null,
  });
  const gateway = new FakeGateway({ jsonRoutes: self.routes({ releases: [assetReleaseStub()] }) });
  const expectedBinding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.40', releaseRebuild: 0 });
  const readbackKey = runsKey(model.CANDIDATE_WORKFLOW_FILE);
  onDispatch(gateway, () => {
    gateway.jsonRoutes[readbackKey] = runsResponse([dispatched]);
  });

  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.DISPATCH_CANDIDATE);
  assert.equal(gateway.dispatches.length, 1);
  const record = gateway.dispatches[0];
  assert.equal(record.workflowFile, model.CANDIDATE_WORKFLOW_FILE);
  assert.equal(record.ref, DEFAULT_BRANCH);
  assert.equal(record.inputs.release_tag, 'v0.1.40');
  assert.ok(Object.values(record.inputs).every((value) => typeof value === 'string'));
  assert.equal(plan.dispatchedRunId, '501');
  assert.equal(runNames.candidateRunName(self.correlationId, expectedBinding), self.candidateName);
}));

// _dispatch_with_other_build_claim.
function dispatchWithOtherBuildClaim(self, { qualificationStatus }) {
  const other = runNames.computeCorrelationId(makeProvenance({ bridgeBuildSha: 'b'.repeat(40) }));
  const orphan = runPayload({
    runId: '401', path: model.CANDIDATE_WORKFLOW_PATH, runName: runNames.candidateRunName(other, self.binding),
  });
  const qualification = runPayload({
    runId: '402',
    path: model.QUALIFICATION_WORKFLOW_PATH,
    runName: runNames.qualificationRunName(other, '401'),
    status: qualificationStatus,
    conclusion: qualificationStatus !== 'completed' ? null : 'failure',
  });
  const gateway = new FakeGateway({
    jsonRoutes: self.routes({ releases: [assetReleaseStub()], candidateRuns: [orphan], qualificationRuns: [qualification] }),
  });
  const readbackKey = runsKey(model.CANDIDATE_WORKFLOW_FILE);
  onDispatch(gateway, ({ inputs }) => {
    gateway.jsonRoutes[readbackKey] = runsResponse([
      orphan,
      runPayload({
        runId: '501',
        path: model.CANDIDATE_WORKFLOW_PATH,
        runName: (
          `bridge-candidate ${self.correlationId}`
          + ` source:${BRIDGE_SHA}`
          + ` tag:${inputs.release_tag}`
          + ` rebuild:${inputs.release_rebuild}`
        ),
        status: 'in_progress',
        conclusion: null,
      }),
    ]);
  });
  driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  return gateway;
}

test('test_finished_pipeline_of_another_build_does_not_force_a_rebuild_tag', withAdvancePipelineFixture((self) => {
  const { inputs } = dispatchWithOtherBuildClaim(self, { qualificationStatus: 'completed' }).dispatches[0];
  assert.equal(inputs.release_tag, 'v0.1.40');
  assert.equal(inputs.release_rebuild, '0');
}));

test('test_in_flight_pipeline_of_another_build_still_reserves_its_tag', withAdvancePipelineFixture((self) => {
  const { inputs } = dispatchWithOtherBuildClaim(self, { qualificationStatus: 'in_progress' }).dispatches[0];
  assert.equal(inputs.release_tag, 'v0.1.41');
  assert.equal(inputs.release_rebuild, '0');
}));

test('test_in_flight_candidate_produces_no_second_dispatch', withAdvancePipelineFixture((self) => {
  const inFlight = runPayload({
    runId: '501', path: model.CANDIDATE_WORKFLOW_PATH, runName: self.candidateName, status: 'in_progress', conclusion: null,
  });
  const gateway = new FakeGateway({ jsonRoutes: self.routes({ candidateRuns: [inFlight] }) });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.IN_FLIGHT);
  assert.deepEqual(gateway.dispatches, []);
}));

test('test_failed_candidate_is_terminal_without_daily_duplicate', withAdvancePipelineFixture((self) => {
  const failed = runPayload({
    runId: '501', path: model.CANDIDATE_WORKFLOW_PATH, runName: self.candidateName, status: 'completed', conclusion: 'failure',
  });
  // Main advancing cannot silently turn the same native provenance into a
  // fresh candidate attempt.
  const advanced = makeProvenance({ bridgeSourceSha: ADVANCED_BRIDGE_SHA });
  const gateway = new FakeGateway({ jsonRoutes: self.routes({ releases: [assetReleaseStub()], candidateRuns: [failed] }) });
  const plan = driver.advancePipeline(gateway, { provenance: advanced, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.BLOCKED);
  assert.deepEqual(gateway.dispatches, []);
  assert.ok(plan.reason.includes('automatic candidate retries are disabled'));
}));

// The routes a succeeded candidate run needs to be proven.
function succeededCandidateRoutes(self, routes, succeeded, extraArtifacts = null) {
  routes[`repos/${BRIDGE_REPOSITORY}/actions/runs/${CANDIDATE_RUN_ID}`] = succeeded;
  routes[`repos/${BRIDGE_REPOSITORY}/actions/runs/${CANDIDATE_RUN_ID}/artifacts?per_page=100`] = artifactInventory({
    runId: CANDIDATE_RUN_ID, name: CANDIDATE_ARTIFACT_NAME, artifactId: CANDIDATE_ARTIFACT_ID, extra: extraArtifacts,
  });
  routes[`repos/${BRIDGE_REPOSITORY}/compare/${HEAD_SHA}...${DEFAULT_BRANCH}`] = { status: 'identical' };
  return routes;
}

test('test_deliberate_success_after_failed_candidate_recovers_pipeline', withAdvancePipelineFixture((self) => {
  const candidateDir = path.join(self.tmp, 'candidate-after-failure');
  writeBridgeCandidate(candidateDir, {
    releaseTag: 'v0.1.40', releaseRebuild: 0, correlationId: self.correlationId, runId: CANDIDATE_RUN_ID,
  });
  const failed = runPayload({
    runId: '501', path: model.CANDIDATE_WORKFLOW_PATH, runName: self.candidateName, status: 'completed', conclusion: 'failure',
  });
  const succeeded = runPayload({ runId: CANDIDATE_RUN_ID, path: model.CANDIDATE_WORKFLOW_PATH, runName: self.candidateName });
  const routes = succeededCandidateRoutes(self, self.routes({ candidateRuns: [failed, succeeded] }), succeeded);
  const gateway = new FakeGateway({
    jsonRoutes: routes,
    blobRoutes: {
      [`repos/${BRIDGE_REPOSITORY}/actions/artifacts/${CANDIDATE_ARTIFACT_ID}/zip`]: flatZip(directoryMembers(candidateDir)),
    },
  });
  const dispatched = runPayload({
    runId: '4201',
    path: model.QUALIFICATION_WORKFLOW_PATH,
    runName: runNames.qualificationRunName(self.correlationId, CANDIDATE_RUN_ID),
    status: 'in_progress',
    conclusion: null,
  });
  const readbackKey = runsKey(model.QUALIFICATION_WORKFLOW_FILE);
  onDispatch(gateway, () => {
    gateway.jsonRoutes[readbackKey] = runsResponse([dispatched]);
  });

  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.DISPATCH_QUALIFICATION);
  assert.equal(plan.candidateRunId, CANDIDATE_RUN_ID);
  assert.equal(plan.dispatchedRunId, '4201');
  assert.deepEqual(gateway.dispatches.map((record) => record.workflowFile), [model.QUALIFICATION_WORKFLOW_FILE]);
}));

test('test_dry_run_plans_without_dispatching', withAdvancePipelineFixture((self) => {
  const gateway = new FakeGateway({ jsonRoutes: self.routes({ releases: [assetReleaseStub()] }) });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp, dryRun: true });
  assert.equal(plan.action, model.OrchestrationAction.DISPATCH_CANDIDATE);
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(plan.dispatchedRunId, null);
  // A dry run must still report the exact ref and inputs it would use.
  assert.equal(plan.dispatchRef, DEFAULT_BRANCH);
  assert.equal(plan.dispatchInputs.assets_immutable_releases_enabled, 'true');
}));

test('test_orphan_assets_tag_ref_is_reserved_before_candidate_dispatch', withAdvancePipelineFixture((self) => {
  const routes = self.routes({ releases: [assetReleaseStub()] });
  routes[`repos/${ASSETS_REPOSITORY}/git/matching-refs/tags?per_page=100`] = [[
    { ref: 'refs/tags/v0.1.40', object: { type: 'commit', sha: ASSETS_TAG_COMMIT } },
  ]];
  const gateway = new FakeGateway({ jsonRoutes: routes });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp, dryRun: true });
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.41');
  assert.equal(plan.releaseTarget.releaseRebuild, 0);
}));

test('test_candidate_dispatch_inputs_carry_the_live_governance_proof', withAdvancePipelineFixture((self) => {
  const dispatched = runPayload({
    runId: '501', path: model.CANDIDATE_WORKFLOW_PATH, runName: self.candidateName, status: 'in_progress', conclusion: null,
  });
  const gateway = new FakeGateway({ jsonRoutes: self.routes({ releases: [assetReleaseStub()] }) });
  const readbackKey = runsKey(model.CANDIDATE_WORKFLOW_FILE);
  onDispatch(gateway, () => {
    gateway.jsonRoutes[readbackKey] = runsResponse([dispatched]);
  });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  const { inputs } = gateway.dispatches[0];
  assert.deepEqual(new Set(Object.keys(inputs)), new Set(planner.CANDIDATE_DISPATCH_INPUTS));
  assert.equal(inputs.assets_immutable_releases_enabled, 'true');
  assert.equal(plan.dispatchRef, DEFAULT_BRANCH);
}));

test('test_duplicate_candidate_artifacts_fail_closed', withAdvancePipelineFixture((self) => {
  const candidateDir = path.join(self.tmp, 'candidate-src');
  writeBridgeCandidate(candidateDir, {
    releaseTag: 'v0.1.40', releaseRebuild: 0, correlationId: self.correlationId, runId: CANDIDATE_RUN_ID,
  });
  const members = directoryMembers(candidateDir);
  const succeeded = runPayload({ runId: CANDIDATE_RUN_ID, path: model.CANDIDATE_WORKFLOW_PATH, runName: self.candidateName });
  const routes = succeededCandidateRoutes(self, self.routes({ candidateRuns: [succeeded] }), succeeded, [{
    id: 8,
    name: CANDIDATE_ARTIFACT_NAME,
    expired: false,
    workflow_run: { id: pyIntFromString(CANDIDATE_RUN_ID) },
  }]);
  const gateway = new FakeGateway({
    jsonRoutes: routes,
    blobRoutes: { [`repos/${BRIDGE_REPOSITORY}/actions/artifacts/${CANDIDATE_ARTIFACT_ID}/zip`]: flatZip(members) },
  });
  assert.throws(() => driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp }), ContractError);
}));

// _assert_candidate_manifest_binding_mismatch.
function assertCandidateManifestBindingMismatch(self, candidateOverrides) {
  const candidateDir = path.join(self.tmp, 'candidate-src');
  writeBridgeCandidate(candidateDir, {
    releaseTag: 'v0.1.40',
    releaseRebuild: 0,
    correlationId: self.correlationId,
    runId: CANDIDATE_RUN_ID,
    ...candidateOverrides,
  });
  const members = directoryMembers(candidateDir);
  const succeeded = runPayload({ runId: CANDIDATE_RUN_ID, path: model.CANDIDATE_WORKFLOW_PATH, runName: self.candidateName });
  const routes = succeededCandidateRoutes(self, self.routes({ candidateRuns: [succeeded] }), succeeded);
  const gateway = new FakeGateway({
    jsonRoutes: routes,
    blobRoutes: { [`repos/${BRIDGE_REPOSITORY}/actions/artifacts/${CANDIDATE_ARTIFACT_ID}/zip`]: flatZip(members) },
  });
  assert.throws(() => driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp }), ContractError);
}

test('test_candidate_manifest_source_that_contradicts_run_name_fails_closed', withAdvancePipelineFixture((self) => {
  assertCandidateManifestBindingMismatch(self, { bridgeCommit: ADVANCED_BRIDGE_SHA });
}));

test('test_candidate_manifest_tag_that_contradicts_run_name_fails_closed', withAdvancePipelineFixture((self) => {
  assertCandidateManifestBindingMismatch(self, { releaseTag: 'v0.1.41' });
}));

test('test_post_dispatch_readback_absence_fails_closed', withAdvancePipelineFixture((self) => {
  const gateway = new FakeGateway({ jsonRoutes: self.routes({ releases: [assetReleaseStub()] }) });
  assert.throws(() => driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp }), ContractError);
  assert.equal(gateway.dispatches.length, 1);
}));

test('test_advance_pipeline_fails_closed_on_malformed_correlated_candidate_run', withAdvancePipelineFixture((self) => {
  const malformedRun = runPayload({
    runId: '501',
    path: model.CANDIDATE_WORKFLOW_PATH,
    runName: `bridge-candidate ${self.correlationId} source:not-a-sha tag:v0.1.40 rebuild:0`,
    status: 'in_progress',
    conclusion: null,
  });
  const gateway = new FakeGateway({ jsonRoutes: self.routes({ candidateRuns: [malformedRun] }) });
  assert.throws(() => driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp }), ContractError);
  assert.deepEqual(gateway.dispatches, []);
}));
