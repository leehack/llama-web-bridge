// Tests of the qualification and publication stages of advancePipeline, one
// test per test method of scripts/release_orchestrator_driver_publication_test.py,
// with the same names and assertions.

import assert from 'node:assert/strict';
import path from 'node:path';
import { test } from 'node:test';

import { BRIDGE_REPOSITORY, ContractError } from '../../../scripts/release/contract.mjs';
import * as driver from '../../../scripts/release/orchestrator/driver.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as runNames from '../../../scripts/release/orchestrator/run_names.mjs';
import { ATTESTATION_ARTIFACT_NAME, CANDIDATE_ARTIFACT_NAME } from '../../../scripts/release/qualification.mjs';
import {
  BRIDGE_SHA, CANDIDATE_ARTIFACT_ID, CANDIDATE_RUN_ID, DEFAULT_BRANCH, FakeGateway, HEAD_SHA, QUALIFICATION_ARTIFACT_ID,
  QUALIFICATION_RUN_ID, artifactInventory, directoryMembers, flatZip, runPayload, runsResponse, withAdvancePipelineFixture,
  writeBridgeCandidate,
} from './fixtures.mjs';
import { attestationBytes, onDispatch, runsKey } from './driver_fixtures.mjs';

const RUNS = `repos/${BRIDGE_REPOSITORY}/actions/runs`;
const ARTIFACTS = `repos/${BRIDGE_REPOSITORY}/actions/artifacts`;

test('test_candidate_success_dispatches_hosted_qualification', withAdvancePipelineFixture((self) => {
  const candidateDir = path.join(self.tmp, 'candidate-src');
  writeBridgeCandidate(candidateDir, {
    releaseTag: 'v0.1.40', releaseRebuild: 0, correlationId: self.correlationId, runId: CANDIDATE_RUN_ID,
  });
  const members = directoryMembers(candidateDir);
  const succeeded = runPayload({ runId: CANDIDATE_RUN_ID, path: model.CANDIDATE_WORKFLOW_PATH, runName: self.candidateName });
  const routes = self.routes({ candidateRuns: [succeeded] });
  routes[`${RUNS}/${CANDIDATE_RUN_ID}`] = succeeded;
  routes[`${RUNS}/${CANDIDATE_RUN_ID}/artifacts?per_page=100`] = artifactInventory({
    runId: CANDIDATE_RUN_ID, name: CANDIDATE_ARTIFACT_NAME, artifactId: CANDIDATE_ARTIFACT_ID,
  });
  routes[`repos/${BRIDGE_REPOSITORY}/compare/${HEAD_SHA}...${DEFAULT_BRANCH}`] = { status: 'identical' };
  routes[`repos/${BRIDGE_REPOSITORY}/compare/${BRIDGE_SHA}...${DEFAULT_BRANCH}`] = { status: 'ahead' };
  const gateway = new FakeGateway({
    jsonRoutes: routes,
    blobRoutes: { [`${ARTIFACTS}/${CANDIDATE_ARTIFACT_ID}/zip`]: flatZip(members) },
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
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.40');
  assert.equal(gateway.dispatches.length, 1);
  const record = gateway.dispatches[0];
  assert.equal(record.workflowFile, model.QUALIFICATION_WORKFLOW_FILE);
  assert.equal(record.ref, DEFAULT_BRANCH);
  assert.deepEqual(record.inputs, {
    orchestrator_correlation_id: self.correlationId,
    candidate_run_id: CANDIDATE_RUN_ID,
  });
}));

test('test_failed_qualification_blocks_instead_of_retrying_forever', withAdvancePipelineFixture((self) => {
  const candidateDir = path.join(self.tmp, 'candidate-failed-qualification');
  writeBridgeCandidate(candidateDir, {
    releaseTag: 'v0.1.40', releaseRebuild: 0, correlationId: self.correlationId, runId: CANDIDATE_RUN_ID,
  });
  const succeeded = runPayload({ runId: CANDIDATE_RUN_ID, path: model.CANDIDATE_WORKFLOW_PATH, runName: self.candidateName });
  const failedQualification = runPayload({
    runId: '4201',
    path: model.QUALIFICATION_WORKFLOW_PATH,
    runName: runNames.qualificationRunName(self.correlationId, CANDIDATE_RUN_ID),
    status: 'completed',
    conclusion: 'failure',
  });
  const routes = self.routes({ candidateRuns: [succeeded], qualificationRuns: [failedQualification] });
  routes[`${RUNS}/${CANDIDATE_RUN_ID}`] = succeeded;
  routes[`${RUNS}/${CANDIDATE_RUN_ID}/artifacts?per_page=100`] = artifactInventory({
    runId: CANDIDATE_RUN_ID, name: CANDIDATE_ARTIFACT_NAME, artifactId: CANDIDATE_ARTIFACT_ID,
  });
  routes[`repos/${BRIDGE_REPOSITORY}/compare/${HEAD_SHA}...${DEFAULT_BRANCH}`] = { status: 'identical' };
  const gateway = new FakeGateway({
    jsonRoutes: routes,
    blobRoutes: { [`${ARTIFACTS}/${CANDIDATE_ARTIFACT_ID}/zip`]: flatZip(directoryMembers(candidateDir)) },
  });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.BLOCKED);
  assert.deepEqual(gateway.dispatches, []);
  assert.ok(plan.reason.includes('automatic qualification retries are disabled'));
}));

// The routes and blobs of a proven candidate and its qualification, the
// attestation built from `attestationDir`'s candidate.
function qualifiedGateway(self, candidateDir, attestationDir) {
  const members = directoryMembers(candidateDir);
  const bytes = attestationBytes(attestationDir);
  const candidateRun = runPayload({ runId: CANDIDATE_RUN_ID, path: model.CANDIDATE_WORKFLOW_PATH, runName: self.candidateName });
  const qualificationRun = runPayload({
    runId: QUALIFICATION_RUN_ID,
    path: model.QUALIFICATION_WORKFLOW_PATH,
    runName: runNames.qualificationRunName(self.correlationId, CANDIDATE_RUN_ID),
  });
  const routes = self.routes({ candidateRuns: [candidateRun], qualificationRuns: [qualificationRun] });
  routes[`${RUNS}/${CANDIDATE_RUN_ID}`] = candidateRun;
  routes[`${RUNS}/${QUALIFICATION_RUN_ID}`] = qualificationRun;
  routes[`${RUNS}/${CANDIDATE_RUN_ID}/artifacts?per_page=100`] = artifactInventory({
    runId: CANDIDATE_RUN_ID, name: CANDIDATE_ARTIFACT_NAME, artifactId: CANDIDATE_ARTIFACT_ID,
  });
  routes[`${RUNS}/${QUALIFICATION_RUN_ID}/artifacts?per_page=100`] = artifactInventory({
    runId: QUALIFICATION_RUN_ID, name: ATTESTATION_ARTIFACT_NAME, artifactId: QUALIFICATION_ARTIFACT_ID,
  });
  routes[`repos/${BRIDGE_REPOSITORY}/compare/${HEAD_SHA}...${DEFAULT_BRANCH}`] = { status: 'identical' };
  return new FakeGateway({
    jsonRoutes: routes,
    blobRoutes: {
      [`${ARTIFACTS}/${CANDIDATE_ARTIFACT_ID}/zip`]: flatZip(members),
      [`${ARTIFACTS}/${QUALIFICATION_ARTIFACT_ID}/zip`]: flatZip({ 'qualification-attestation.json': bytes }),
    },
  });
}

test('test_full_pipeline_dispatches_publish_after_exact_attestation', withAdvancePipelineFixture((self) => {
  const candidateDir = path.join(self.tmp, 'candidate-src');
  writeBridgeCandidate(candidateDir, {
    releaseTag: 'v0.1.40', releaseRebuild: 0, correlationId: self.correlationId, runId: CANDIDATE_RUN_ID,
  });
  const gateway = qualifiedGateway(self, candidateDir, candidateDir);
  const publishName = runNames.publishRunName(self.correlationId, CANDIDATE_RUN_ID, QUALIFICATION_RUN_ID, self.binding);
  const publishRun = runPayload({
    runId: '701', path: model.PUBLISH_WORKFLOW_PATH, runName: publishName, status: 'in_progress', conclusion: null,
  });
  const publishKey = runsKey(model.PUBLISH_WORKFLOW_FILE);
  onDispatch(gateway, () => {
    gateway.jsonRoutes[publishKey] = runsResponse([publishRun]);
  });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.DISPATCH_PUBLISH);
  assert.equal(gateway.dispatches.length, 1);
  const { inputs } = gateway.dispatches[0];
  assert.equal(inputs.candidate_run_id, CANDIDATE_RUN_ID);
  assert.equal(inputs.qualification_run_id, QUALIFICATION_RUN_ID);
  assert.equal(inputs.release_tag, 'v0.1.40');
  assert.equal(inputs.bridge_source_sha, BRIDGE_SHA);
  assert.equal(inputs.publish_approved, 'true');
  assert.equal(gateway.dispatches[0].ref, DEFAULT_BRANCH);
  assert.equal(plan.dispatchedRunId, '701');
  const qualificationReachabilityPath = `repos/${BRIDGE_REPOSITORY}/compare/${HEAD_SHA}...${DEFAULT_BRANCH}`;
  assert.equal(gateway.apiPaths.filter((apiPath) => apiPath === qualificationReachabilityPath).length, 2);
}));

test('test_attestation_bound_to_another_candidate_fails_closed', withAdvancePipelineFixture((self) => {
  const candidateDir = path.join(self.tmp, 'candidate-src');
  writeBridgeCandidate(candidateDir, {
    releaseTag: 'v0.1.40', releaseRebuild: 0, correlationId: self.correlationId, runId: CANDIDATE_RUN_ID,
  });
  const otherDir = path.join(self.tmp, 'other-src');
  writeBridgeCandidate(otherDir, {
    releaseTag: 'v0.1.40',
    releaseRebuild: 0,
    correlationId: self.correlationId,
    runId: CANDIDATE_RUN_ID,
    marker: Buffer.from('tampered'),
  });
  const gateway = qualifiedGateway(self, candidateDir, otherDir);
  assert.throws(() => driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp }), ContractError);
  assert.deepEqual(gateway.dispatches, []);
}));
