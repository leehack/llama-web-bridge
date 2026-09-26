// Tests of candidates satisfied by an identical published release, one test
// per test method of scripts/release_orchestrator_driver_identical_release_test.py,
// with the same names and assertions. A Python subTest loop is one loop inside
// its test.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';

import { ASSETS_REPOSITORY, BRIDGE_REPOSITORY } from '../../../scripts/release/contract.mjs';
import * as orchestrator from '../../../scripts/release/orchestrator/cli.mjs';
import * as assetReleases from '../../../scripts/release/orchestrator/asset_releases.mjs';
import * as driver from '../../../scripts/release/orchestrator/driver.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as runNames from '../../../scripts/release/orchestrator/run_names.mjs';
import { ATTESTATION_ARTIFACT_NAME, CANDIDATE_ARTIFACT_NAME, loadCandidate } from '../../../scripts/release/qualification.mjs';
import { releaseAttestation } from '../contract_fixtures.mjs';
import {
  ADVANCED_BRIDGE_SHA, ASSETS_TAG_COMMIT, BRIDGE_SHA, CANDIDATE_ARTIFACT_ID, CANDIDATE_RUN_ID, FakeGateway,
  NATIVE_PUBLISHED_AT, QUALIFICATION_ARTIFACT_ID, QUALIFICATION_RUN_ID, artifactInventory, directoryMembers, flatZip,
  makeProvenance, releasePayload, runPayload, runsResponse, withAdvancePipelineFixture, writeBridgeCandidate,
} from './fixtures.mjs';
import {
  attestationBytes, onDispatch, replaceProvenance, runsKey,
} from './driver_fixtures.mjs';

const RUNS = `repos/${BRIDGE_REPOSITORY}/actions/runs`;

function withFixture(fn) {
  return withAdvancePipelineFixture(fn, { orchestrator });
}

// _aligned_release: [release, routes, blobs, attestations], a verifiable
// immutable release carrying the deterministic notes.
function alignedRelease(publishedDir, {
  tag, provenance, correlationId, releaseId = 4343,
}) {
  const members = directoryMembers(publishedDir);
  const fingerprint = loadCandidate(publishedDir)[1];
  const body = (
    `Candidate fingerprint: \`${fingerprint}\`\n`
    + `Native: \`${provenance.nativeRepo}@${provenance.nativeReleaseTag}\`\n`
    + `Native manifest SHA-256: \`${provenance.nativeManifestSha256}\`\n`
    + `Orchestrator correlation: \`${correlationId}\`\n`
  );
  const release = releasePayload({
    tag, body, members, releaseId,
  });
  const routes = {
    [`repos/${ASSETS_REPOSITORY}/git/ref/tags/${tag}`]: { ref: `refs/tags/${tag}`, object: { type: 'commit', sha: ASSETS_TAG_COMMIT } },
    [`repos/${ASSETS_REPOSITORY}/releases/tags/${tag}`]: release,
    [`repos/${ASSETS_REPOSITORY}/releases/${releaseId}`]: release,
  };
  const blobs = Object.fromEntries(release.assets.map((asset) => [
    `repos/${ASSETS_REPOSITORY}/releases/assets/${asset.id}`, members[asset.name],
  ]));
  const attestations = [[[ASSETS_REPOSITORY, tag], releaseAttestation({
    releaseTag: tag,
    assetsRepo: ASSETS_REPOSITORY,
    tagCommit: ASSETS_TAG_COMMIT,
    releaseId,
    assets: Object.fromEntries(Object.entries(members).map(([name, data]) => [name, createHash('sha256').update(data).digest('hex')])),
  })]];
  return [release, routes, blobs, attestations];
}

// _proven_candidate: [run, routes, blobs], a succeeded candidate run whose
// artifact is downloadable and proven.
function provenCandidate(candidateDir, { marker, releaseTag, provenance }) {
  const correlationId = runNames.computeCorrelationId(provenance);
  writeBridgeCandidate(candidateDir, {
    releaseTag,
    releaseRebuild: 0,
    correlationId,
    runId: CANDIDATE_RUN_ID,
    marker: Buffer.from(marker),
    upstreamTag: provenance.upstreamTag,
    upstreamCommit: provenance.upstreamCommit,
    nativeReleaseTag: provenance.nativeReleaseTag,
    nativeManifestSha256: provenance.nativeManifestSha256,
    nativeCommit: provenance.nativeCommit,
  });
  const run = runPayload({
    runId: CANDIDATE_RUN_ID,
    path: model.CANDIDATE_WORKFLOW_PATH,
    runName: runNames.candidateRunName(
      correlationId,
      new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag, releaseRebuild: 0 }),
    ),
  });
  const routes = {
    [`${RUNS}/${CANDIDATE_RUN_ID}`]: run,
    [`${RUNS}/${CANDIDATE_RUN_ID}/artifacts?per_page=100`]: artifactInventory({
      runId: CANDIDATE_RUN_ID, name: CANDIDATE_ARTIFACT_NAME, artifactId: CANDIDATE_ARTIFACT_ID,
    }),
  };
  const blobs = {
    [`repos/${BRIDGE_REPOSITORY}/actions/artifacts/${CANDIDATE_ARTIFACT_ID}/zip`]: flatZip(directoryMembers(candidateDir)),
  };
  return [run, routes, blobs];
}

// _aligned_candidate_gateway: v0.1.41 publishes "same" bytes for native v0.2.0
// under an earlier build identity; the scanned provenance has a proven
// candidate bound to v0.1.42. Returns [gateway, release, run].
function alignedCandidateGateway(self, { candidateMarker, provenance = null, siblingRuns = null }) {
  const scanned = provenance || self.provenance;
  const prior = replaceProvenance(self.provenance, { bridgeSourceSha: ADVANCED_BRIDGE_SHA, bridgeBuildSha: ADVANCED_BRIDGE_SHA });
  const priorCorrelation = runNames.computeCorrelationId(prior);
  const publishedDir = path.join(self.tmp, 'published-src');
  writeBridgeCandidate(publishedDir, {
    releaseTag: 'v0.1.41',
    releaseRebuild: 0,
    correlationId: priorCorrelation,
    bridgeCommit: ADVANCED_BRIDGE_SHA,
    runId: '777',
    marker: Buffer.from('same'),
  });
  const [release, releaseRoutes, releaseBlobs, attestations] = alignedRelease(publishedDir, {
    tag: 'v0.1.41', provenance: prior, correlationId: priorCorrelation,
  });
  const [run, runRoutes, runBlobs] = provenCandidate(path.join(self.tmp, 'candidate-src'), {
    marker: candidateMarker, releaseTag: 'v0.1.42', provenance: scanned,
  });
  const routes = { ...self.routes({ releases: [release], candidateRuns: [run, ...(siblingRuns || [])] }), ...releaseRoutes, ...runRoutes };
  const gateway = new FakeGateway({
    jsonRoutes: routes,
    blobRoutes: { ...releaseBlobs, ...runBlobs },
    releaseAttestations: attestations,
  });
  return [gateway, release, run];
}

// _expect_qualification_dispatch.
function expectQualificationDispatch(gateway, provenance) {
  const correlationId = runNames.computeCorrelationId(provenance);
  const readbackKey = runsKey(model.QUALIFICATION_WORKFLOW_FILE);
  const dispatched = runPayload({
    runId: '4201',
    path: model.QUALIFICATION_WORKFLOW_PATH,
    runName: runNames.qualificationRunName(correlationId, CANDIDATE_RUN_ID),
    status: 'in_progress',
    conclusion: null,
  });
  onDispatch(gateway, () => {
    gateway.jsonRoutes[readbackKey] = runsResponse([dispatched]);
  });
}

const touchedRelease = (gateway, release) => gateway.apiPaths.some((apiPath) => apiPath.includes(`/releases/${release.id}`));

test('test_identical_candidate_is_satisfied_by_the_aligned_release', withFixture((self) => {
  const [gateway, release] = alignedCandidateGateway(self, { candidateMarker: 'same' });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.41');
  assert.equal(plan.candidateRunId, CANDIDATE_RUN_ID);
  assert.equal(plan.dispatchWorkflow, null);
  assert.ok(plan.reason.includes('v0.1.42'));
  assert.deepEqual(gateway.dispatches, []);
  assert.ok(gateway.apiPaths.includes(`repos/${ASSETS_REPOSITORY}/releases/${release.id}`));
  assert.equal(plan.qualificationRunId, null);
  assert.ok(plan.reason.includes('other than manifest.json'));
  assert.equal(plan.toDict().action, 'satisfied_by_identical_release');
}));

// _qualify_candidate: record a succeeded, attested qualification of the
// aligned fixture's candidate and the given publication runs.
function qualifyCandidate(self, gateway, { publishRuns = [] } = {}) {
  const bytes = attestationBytes(path.join(self.tmp, 'candidate-src'));
  const qualificationRun = runPayload({
    runId: QUALIFICATION_RUN_ID,
    path: model.QUALIFICATION_WORKFLOW_PATH,
    runName: runNames.qualificationRunName(self.correlationId, CANDIDATE_RUN_ID),
  });
  gateway.jsonRoutes[`${RUNS}/${QUALIFICATION_RUN_ID}`] = qualificationRun;
  gateway.jsonRoutes[`${RUNS}/${QUALIFICATION_RUN_ID}/artifacts?per_page=100`] = artifactInventory({
    runId: QUALIFICATION_RUN_ID, name: ATTESTATION_ARTIFACT_NAME, artifactId: QUALIFICATION_ARTIFACT_ID,
  });
  for (const [workflowFile, listed] of [
    [model.QUALIFICATION_WORKFLOW_FILE, [qualificationRun]],
    [model.PUBLISH_WORKFLOW_FILE, [...publishRuns]],
  ]) {
    gateway.jsonRoutes[runsKey(workflowFile)] = runsResponse(listed);
  }
  gateway.blobRoutes[`repos/${BRIDGE_REPOSITORY}/actions/artifacts/${QUALIFICATION_ARTIFACT_ID}/zip`] = flatZip({
    'qualification-attestation.json': bytes,
  });
}

// _publish_run.
function publishRun(self, { status, conclusion }) {
  return runPayload({
    runId: '701',
    path: model.PUBLISH_WORKFLOW_PATH,
    runName: runNames.publishRunName(
      self.correlationId,
      CANDIDATE_RUN_ID,
      QUALIFICATION_RUN_ID,
      new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.42', releaseRebuild: 0 }),
    ),
    status,
    conclusion,
  });
}

test('test_qualified_identical_candidate_is_satisfied_before_publication', withFixture((self) => {
  const [gateway, release] = alignedCandidateGateway(self, { candidateMarker: 'same' });
  qualifyCandidate(self, gateway);
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.41');
  assert.equal(plan.qualificationRunId, QUALIFICATION_RUN_ID);
  assert.deepEqual(gateway.dispatches, []);
  assert.ok(gateway.apiPaths.includes(`repos/${ASSETS_REPOSITORY}/releases/${release.id}`));
}));

test('test_failed_publication_of_an_identical_candidate_is_satisfied_not_retried', withFixture((self) => {
  const [gateway] = alignedCandidateGateway(self, { candidateMarker: 'same' });
  qualifyCandidate(self, gateway, { publishRuns: [publishRun(self, { status: 'completed', conclusion: 'failure' })] });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE);
  assert.deepEqual(gateway.dispatches, []);
}));

test('test_identical_candidate_with_qualification_in_flight_is_not_compared', withFixture((self) => {
  const [gateway, release] = alignedCandidateGateway(self, { candidateMarker: 'same' });
  gateway.jsonRoutes[runsKey(model.QUALIFICATION_WORKFLOW_FILE)] = runsResponse([runPayload({
    runId: '4201',
    path: model.QUALIFICATION_WORKFLOW_PATH,
    runName: runNames.qualificationRunName(self.correlationId, CANDIDATE_RUN_ID),
    status: 'in_progress',
    conclusion: null,
  })]);
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.IN_FLIGHT);
  assert.equal(plan.inFlightWorkflow, model.QUALIFICATION_WORKFLOW_FILE);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.42');
  assert.equal(touchedRelease(gateway, release), false);
}));

// _route_newer_native: route a backlog scan of native v0.2.1 that sees `run`
// as the only candidate and lists the candidate it dispatches on readback.
function routeNewerNative(self, gateway, release, run) {
  const newer = self.newerNative();
  const newerSince = assetReleases.workflowHistorySince([release], newer);
  assert.notEqual(newerSince, NATIVE_PUBLISHED_AT);
  for (const workflowFile of [model.CANDIDATE_WORKFLOW_FILE, model.QUALIFICATION_WORKFLOW_FILE, model.PUBLISH_WORKFLOW_FILE]) {
    gateway.jsonRoutes[runsKey(workflowFile, newerSince)] = runsResponse(workflowFile === model.CANDIDATE_WORKFLOW_FILE ? [run] : []);
  }
  const readbackKey = runsKey(model.CANDIDATE_WORKFLOW_FILE, newer.nativeReleasePublishedAt);
  gateway.jsonRoutes[readbackKey] = runsResponse([run]);
  onDispatch(gateway, ({ inputs }) => {
    gateway.jsonRoutes[readbackKey] = runsResponse([
      run,
      runPayload({
        runId: '501',
        path: model.CANDIDATE_WORKFLOW_PATH,
        runName: (
          `bridge-candidate ${runNames.computeCorrelationId(newer)}`
          + ` source:${BRIDGE_SHA}`
          + ` tag:${inputs.release_tag}`
          + ` rebuild:${inputs.release_rebuild}`
        ),
        status: 'in_progress',
        conclusion: null,
      }),
    ]);
  });
  return newer;
}

const backlogRows = (backlog) => backlog.plans.map((item) => [item.provenance.native_release_tag, item.action, item.release_tag]);
const dispatchRows = (gateway) => gateway.dispatches.map((record) => [
  record.workflowFile, record.inputs.release_tag, record.inputs.release_rebuild,
]);

test('test_identical_candidate_with_publication_in_flight_keeps_its_tag', withFixture(async (self) => {
  const [gateway, release, run] = alignedCandidateGateway(self, { candidateMarker: 'same' });
  qualifyCandidate(self, gateway, { publishRuns: [publishRun(self, { status: 'in_progress', conclusion: null })] });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.IN_FLIGHT);
  assert.equal(plan.inFlightWorkflow, model.PUBLISH_WORKFLOW_FILE);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.42');
  assert.equal(touchedRelease(gateway, release), false);

  const newer = routeNewerNative(self, gateway, release, run);
  const [result, backlog] = await self.runBacklog(gateway, [self.provenance, newer]);
  assert.deepEqual(backlog.errors, []);
  assert.equal(result, 0);
  assert.deepEqual(backlogRows(backlog), [
    ['v0.2.0', 'in_flight', 'v0.1.42'],
    ['v0.2.1', 'dispatch_candidate', 'v0.1.43'],
  ]);
  assert.deepEqual(dispatchRows(gateway), [[model.CANDIDATE_WORKFLOW_FILE, 'v0.1.43', '0']]);
}));

test('test_differing_candidate_is_qualified_not_skipped', withFixture((self) => {
  const [gateway] = alignedCandidateGateway(self, { candidateMarker: 'changed' });
  expectQualificationDispatch(gateway, self.provenance);
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.DISPATCH_QUALIFICATION);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.42');
  assert.deepEqual(gateway.dispatches.map((record) => record.workflowFile), [model.QUALIFICATION_WORKFLOW_FILE]);
}));

test('test_identical_bytes_under_another_native_alignment_are_not_compared', withFixture((self) => {
  const rebuilt = makeProvenance({ nativeReleaseTag: 'v0.2.0-1' });
  const [gateway, release] = alignedCandidateGateway(self, { candidateMarker: 'same', provenance: rebuilt });
  expectQualificationDispatch(gateway, rebuilt);
  const plan = driver.advancePipeline(gateway, { provenance: rebuilt, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.DISPATCH_QUALIFICATION);
  assert.equal(touchedRelease(gateway, release), false);
  assert.deepEqual(gateway.dispatches.map((record) => record.workflowFile), [model.QUALIFICATION_WORKFLOW_FILE]);
}));

test('test_satisfied_correlation_stays_satisfied_and_frees_its_tag_on_later_scans', withFixture(async (self) => {
  const [gateway, release, run] = alignedCandidateGateway(self, { candidateMarker: 'same' });
  for (const scan of [1, 2]) {
    const workspace = path.join(self.tmp, `scan-${scan}`);
    fs.mkdirSync(workspace);
    const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace });
    assert.equal(plan.action, model.OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE, `scan ${scan}`);
  }
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(gateway.apiPaths.some((apiPath) => apiPath.includes(model.PUBLISH_WORKFLOW_FILE)), false);

  const newer = routeNewerNative(self, gateway, release, run);
  const [result, backlog] = await self.runBacklog(gateway, [self.provenance, newer]);
  assert.deepEqual(backlog.errors, []);
  assert.equal(result, 0);
  assert.deepEqual(backlogRows(backlog), [
    ['v0.2.0', 'satisfied_by_identical_release', 'v0.1.41'],
    ['v0.2.1', 'dispatch_candidate', 'v0.1.42'],
  ]);
  assert.deepEqual(dispatchRows(gateway), [[model.CANDIDATE_WORKFLOW_FILE, 'v0.1.42', '0']]);
}));

test('test_identical_candidate_still_publishes_when_a_sibling_claims_its_rebuild', withFixture((self) => {
  const sibling = runPayload({
    runId: '502',
    path: model.CANDIDATE_WORKFLOW_PATH,
    runName: runNames.candidateRunName(
      runNames.computeCorrelationId(self.newerNative()),
      new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.42-1', releaseRebuild: 1 }),
    ),
  });
  const [gateway, release] = alignedCandidateGateway(self, { candidateMarker: 'same', siblingRuns: [sibling] });
  expectQualificationDispatch(gateway, self.provenance);
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.DISPATCH_QUALIFICATION);
  assert.equal(touchedRelease(gateway, release), false);
}));
