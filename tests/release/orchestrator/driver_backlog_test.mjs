// Tests of advancePipeline across an ordered native backlog: publication
// barriers, superseded natives, and published noops. One test per test method
// of scripts/release_orchestrator_driver_backlog_test.py, with the same names
// and assertions. A Python subTest loop is one loop inside its test.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { test } from 'node:test';

import { ASSETS_REPOSITORY, BRIDGE_REPOSITORY, ContractError } from '../../../scripts/release/contract.mjs';
import { pyJsonDumps, pyJsonLoads } from '../../../scripts/release/json.mjs';
import * as orchestrator from '../../../scripts/release/orchestrator/cli.mjs';
import * as assetReleases from '../../../scripts/release/orchestrator/asset_releases.mjs';
import * as driver from '../../../scripts/release/orchestrator/driver.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as releaseTags from '../../../scripts/release/orchestrator/release_tags.mjs';
import * as runNames from '../../../scripts/release/orchestrator/run_names.mjs';
import {
  ATTESTATION_ARTIFACT_NAME, CANDIDATE_ARTIFACT_NAME, loadCandidate, loadPublishedCandidate,
} from '../../../scripts/release/qualification.mjs';
import { releaseAttestation } from '../contract_fixtures.mjs';
import {
  ADVANCED_BRIDGE_SHA, ASSETS_TAG_COMMIT, BRIDGE_SHA, CANDIDATE_ARTIFACT_ID, CANDIDATE_RUN_ID, DEFAULT_BRANCH, FakeGateway,
  HEAD_SHA, LEGACY_BRIDGE_SHA, LEGACY_CANDIDATE_RUN_ID, LEGACY_MANUAL_QUALIFICATION_GATES, LEGACY_MANUAL_UNPROVEN_CAPABILITIES,
  LEGACY_NATIVE_COMMIT, LEGACY_NATIVE_MANIFEST_SHA, LEGACY_UPSTREAM_COMMIT, QUALIFICATION_ARTIFACT_ID, QUALIFICATION_RUN_ID,
  alignedReleaseStub, artifactInventory, assetReleaseStub, directoryMembers, flatZip, makeLegacyV0140Provenance,
  makeProvenance, releasePayload, rewriteLegacyCandidateManifest, runPayload, runsResponse, withAdvancePipelineFixture,
  writeBridgeCandidate,
} from './fixtures.mjs';
import {
  attestationBytes, onDispatch, replaceProvenance, runsKey,
} from './driver_fixtures.mjs';

const RUNS = `repos/${BRIDGE_REPOSITORY}/actions/runs`;
const ARTIFACTS = `repos/${BRIDGE_REPOSITORY}/actions/artifacts`;

function withFixture(fn) {
  return withAdvancePipelineFixture(fn, { orchestrator });
}

// _published_candidate_gateway.
function publishedCandidateGateway(self, candidateDir, { provenance, correlationId, publishedManifestCompatibility = null }) {
  const members = directoryMembers(candidateDir);
  const manifest = pyJsonLoads(fs.readFileSync(path.join(candidateDir, 'manifest.json'), 'utf8'));
  const fingerprint = publishedManifestCompatibility === null
    ? loadCandidate(candidateDir)[1]
    : loadPublishedCandidate(candidateDir, {
      expectedQualificationGates: publishedManifestCompatibility[0],
      expectedUnprovenCapabilities: publishedManifestCompatibility[1],
    })[1];
  const body = `Candidate fingerprint: \`${fingerprint}\`\nOrchestrator correlation: \`${correlationId}\`\n`;
  const release = releasePayload({ tag: 'v0.1.40', body, members });
  const routes = self.routes({ releases: [release] });
  Object.assign(routes, {
    [`repos/${BRIDGE_REPOSITORY}/compare/${provenance.bridgeSourceSha}...${DEFAULT_BRANCH}`]: { status: 'ahead' },
    [`repos/${BRIDGE_REPOSITORY}/compare/${manifest.bridge_commit}...${DEFAULT_BRANCH}`]: { status: 'ahead' },
    [`repos/${ASSETS_REPOSITORY}/git/ref/tags/v0.1.40`]: { ref: 'refs/tags/v0.1.40', object: { type: 'commit', sha: ASSETS_TAG_COMMIT } },
    [`repos/${ASSETS_REPOSITORY}/releases/tags/v0.1.40`]: release,
    [`repos/${ASSETS_REPOSITORY}/releases/${release.id}`]: release,
  });
  const blobs = Object.fromEntries(release.assets.map((asset) => [
    `repos/${ASSETS_REPOSITORY}/releases/assets/${asset.id}`, members[asset.name],
  ]));
  const digests = Object.fromEntries(Object.entries(members).map(([name, data]) => [name, createHash('sha256').update(data).digest('hex')]));
  return new FakeGateway({
    jsonRoutes: routes,
    blobRoutes: blobs,
    releaseAttestations: [[[ASSETS_REPOSITORY, 'v0.1.40'], releaseAttestation({
      releaseTag: 'v0.1.40', assetsRepo: ASSETS_REPOSITORY, tagCommit: ASSETS_TAG_COMMIT, releaseId: release.id, assets: digests,
    })]],
  });
}

test('test_later_qualified_provenance_waits_for_earlier_immutable_publication', async () => {
  const root = fs.mkdtempSync(path.join(tmpdir(), 'sro-backlog-order-'));
  try {
    const first = makeProvenance();
    const second = makeProvenance({
      upstreamTag: 'v0.2.1',
      upstreamCommit: 'd'.repeat(40),
      nativeReleaseTag: 'v0.2.1',
      nativeCommit: 'e'.repeat(40),
      nativeManifestSha256: 'f'.repeat(64),
      nativeReleasePublishedAt: '2026-08-21T12:34:56Z',
    });
    const firstCorrelation = runNames.computeCorrelationId(first);
    const secondCorrelation = runNames.computeCorrelationId(second);
    const firstBinding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.40', releaseRebuild: 0 });
    const secondBinding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.40-1', releaseRebuild: 1 });
    const firstRunId = '4101';
    const secondRunId = '4201';
    const secondQualificationRunId = '4202';
    const firstArtifactId = 31;
    const secondArtifactId = 32;
    const secondQualificationArtifactId = 33;

    const firstCandidate = path.join(root, 'first-candidate');
    const secondCandidate = path.join(root, 'second-candidate');
    writeBridgeCandidate(firstCandidate, {
      releaseTag: firstBinding.releaseTag, releaseRebuild: firstBinding.releaseRebuild, correlationId: firstCorrelation, runId: firstRunId,
    });
    writeBridgeCandidate(secondCandidate, {
      releaseTag: secondBinding.releaseTag,
      releaseRebuild: secondBinding.releaseRebuild,
      correlationId: secondCorrelation,
      runId: secondRunId,
      upstreamTag: second.upstreamTag,
      upstreamCommit: second.upstreamCommit,
      nativeReleaseTag: second.nativeReleaseTag,
      nativeManifestSha256: second.nativeManifestSha256,
      nativeCommit: second.nativeCommit,
    });
    const secondQualification = attestationBytes(secondCandidate, {
      candidateRunId: secondRunId,
      candidateArtifactId: secondArtifactId,
      qualificationRunId: secondQualificationRunId,
    });

    const firstRun = runPayload({
      runId: firstRunId, path: model.CANDIDATE_WORKFLOW_PATH, runName: runNames.candidateRunName(firstCorrelation, firstBinding),
    });
    const secondRun = runPayload({
      runId: secondRunId, path: model.CANDIDATE_WORKFLOW_PATH, runName: runNames.candidateRunName(secondCorrelation, secondBinding),
    });
    const qualificationRun = runPayload({
      runId: secondQualificationRunId,
      path: model.QUALIFICATION_WORKFLOW_PATH,
      runName: runNames.qualificationRunName(secondCorrelation, secondRunId),
    });
    const releases = [assetReleaseStub()];
    const routes = {
      [`repos/${ASSETS_REPOSITORY}/releases?per_page=100`]: [releases],
      [`repos/${BRIDGE_REPOSITORY}/compare/${BRIDGE_SHA}...${DEFAULT_BRANCH}`]: { status: 'ahead' },
      [`repos/${BRIDGE_REPOSITORY}/compare/${HEAD_SHA}...${DEFAULT_BRANCH}`]: { status: 'identical' },
      [`${RUNS}/${firstRunId}`]: firstRun,
      [`${RUNS}/${secondRunId}`]: secondRun,
      [`${RUNS}/${secondQualificationRunId}`]: qualificationRun,
      [`${RUNS}/${firstRunId}/artifacts?per_page=100`]: artifactInventory({
        runId: firstRunId, name: CANDIDATE_ARTIFACT_NAME, artifactId: firstArtifactId,
      }),
      [`${RUNS}/${secondRunId}/artifacts?per_page=100`]: artifactInventory({
        runId: secondRunId, name: CANDIDATE_ARTIFACT_NAME, artifactId: secondArtifactId,
      }),
      [`${RUNS}/${secondQualificationRunId}/artifacts?per_page=100`]: artifactInventory({
        runId: secondQualificationRunId, name: ATTESTATION_ARTIFACT_NAME, artifactId: secondQualificationArtifactId,
      }),
    };
    const candidateRuns = runsResponse([firstRun, secondRun]);
    const qualificationRuns = runsResponse([qualificationRun]);
    const publishRuns = runsResponse([]);
    for (const provenance of [first, second]) {
      const since = assetReleases.workflowHistorySince(releases, provenance);
      routes[runsKey(model.CANDIDATE_WORKFLOW_FILE, since)] = candidateRuns;
      routes[runsKey(model.QUALIFICATION_WORKFLOW_FILE, since)] = qualificationRuns;
      routes[runsKey(model.PUBLISH_WORKFLOW_FILE, since)] = publishRuns;
    }
    const gateway = new FakeGateway({
      jsonRoutes: routes,
      blobRoutes: {
        [`${ARTIFACTS}/${firstArtifactId}/zip`]: flatZip(directoryMembers(firstCandidate)),
        [`${ARTIFACTS}/${secondArtifactId}/zip`]: flatZip(directoryMembers(secondCandidate)),
        [`${ARTIFACTS}/${secondQualificationArtifactId}/zip`]: flatZip({ 'qualification-attestation.json': secondQualification }),
      },
    });
    const firstQualificationRun = runPayload({
      runId: '4201',
      path: model.QUALIFICATION_WORKFLOW_PATH,
      runName: runNames.qualificationRunName(firstCorrelation, firstRunId),
      status: 'in_progress',
      conclusion: null,
    });
    const qualificationRouteKeys = [first, second].map((provenance) => runsKey(
      model.QUALIFICATION_WORKFLOW_FILE,
      assetReleases.workflowHistorySince(releases, provenance),
    ));
    onDispatch(gateway, () => {
      for (const key of qualificationRouteKeys) gateway.jsonRoutes[key] = runsResponse([qualificationRun, firstQualificationRun]);
    });

    const provenanceList = path.join(root, 'release-candidates.json');
    fs.writeFileSync(provenanceList, pyJsonDumps([orchestrator.provenanceToDict(first), orchestrator.provenanceToDict(second)]), 'utf8');
    const outputPlan = path.join(root, 'orchestration-plan.json');
    const result = orchestrator.main([
      'orchestrate-backlog',
      '--provenance-list-json',
      provenanceList,
      '--workspace',
      path.join(root, 'workspace'),
      '--output-plan-json',
      outputPlan,
    ], {
      env: {
        ...process.env,
        GITHUB_EVENT_NAME: 'schedule',
        GITHUB_ACTOR: 'github-actions',
        GITHUB_TRIGGERING_ACTOR: 'github-actions',
      },
      createGateway: () => gateway,
      stdout: () => {},
      stderr: () => {},
    });
    assert.equal(result, 0);
    const plan = pyJsonLoads(fs.readFileSync(outputPlan, 'utf8'));
    assert.deepEqual(plan.plans.map((item) => item.action), [
      model.OrchestrationAction.DISPATCH_QUALIFICATION,
      model.OrchestrationAction.WAITING_FOR_PRIOR_PUBLICATION,
    ]);
    // The later provenance is fully qualified but must not publish before the
    // earlier one, so exactly one qualification dispatch happens and no
    // publication is dispatched.
    assert.deepEqual(gateway.dispatches.map((record) => record.workflowFile), [model.QUALIFICATION_WORKFLOW_FILE]);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('test_native_behind_the_published_alignment_is_superseded', withFixture((self) => {
  const priorBuild = self.newerNative({ bridgeBuildSha: ADVANCED_BRIDGE_SHA });
  const gateway = new FakeGateway({
    jsonRoutes: self.routes({ releases: [assetReleaseStub(), alignedReleaseStub('v0.1.40', priorBuild)] }),
  });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp, newerNativeScanned: true });
  assert.equal(plan.action, model.OrchestrationAction.SUPERSEDED);
  assert.ok(plan.reason.includes('v0.2.1'));
  assert.ok(plan.reason.includes('v0.1.40'));
  assert.equal(plan.releaseTarget, null);
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(gateway.apiPaths.some((apiPath) => apiPath.includes('/actions/')), false);
}));

test('test_newest_scanned_native_is_never_skipped_as_superseded', withFixture(async (self) => {
  const priorBuild = self.newerNative({ bridgeBuildSha: ADVANCED_BRIDGE_SHA });
  const gateway = new FakeGateway({ jsonRoutes: self.routes({ releases: [alignedReleaseStub('v0.1.40', priorBuild)] }) });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.BLOCKED);
  assert.deepEqual(gateway.dispatches, []);
  const [result, backlog] = await self.runBacklog(gateway, [self.provenance]);
  assert.equal(result, 1);
  assert.deepEqual(backlog.plans.map((item) => item.action), [model.OrchestrationAction.BLOCKED]);
}));

test('test_newest_native_stays_eligible_after_a_governed_bridge_change', withFixture((self) => {
  const priorBuild = replaceProvenance(self.provenance, { bridgeBuildSha: ADVANCED_BRIDGE_SHA });
  const gateway = new FakeGateway({ jsonRoutes: self.routes({ releases: [alignedReleaseStub('v0.1.40', priorBuild)] }) });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp, dryRun: true });
  assert.equal(plan.action, model.OrchestrationAction.DISPATCH_CANDIDATE);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.41');
}));

test('test_native_rebuild_ahead_of_the_alignment_stays_eligible', withFixture((self) => {
  const rebuilt = makeProvenance({ nativeReleaseTag: 'v0.2.0-1' });
  const gateway = new FakeGateway({ jsonRoutes: self.routes({ releases: [alignedReleaseStub('v0.1.40', self.provenance)] }) });
  const plan = driver.advancePipeline(gateway, { provenance: rebuilt, workspace: self.tmp, dryRun: true });
  assert.equal(plan.action, model.OrchestrationAction.DISPATCH_CANDIDATE);
}));

test('test_native_rebuild_behind_the_alignment_is_not_republished', withFixture((self) => {
  const rebuilt = makeProvenance({ nativeReleaseTag: 'v0.2.0-1', bridgeBuildSha: ADVANCED_BRIDGE_SHA });
  for (const [newerNativeScanned, action] of [
    [true, model.OrchestrationAction.SUPERSEDED],
    [false, model.OrchestrationAction.BLOCKED],
  ]) {
    const gateway = new FakeGateway({ jsonRoutes: self.routes({ releases: [alignedReleaseStub('v0.1.40', rebuilt)] }) });
    const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp, newerNativeScanned });
    assert.equal(plan.action, action, `newerNativeScanned=${newerNativeScanned}`);
    assert.ok(plan.reason.includes('v0.2.0-1'));
    assert.deepEqual(gateway.dispatches, []);
  }
}));

test('test_first_publication_into_an_empty_assets_history', withFixture((self) => {
  const gateway = new FakeGateway({ jsonRoutes: self.routes({ releases: [] }) });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp, dryRun: true });
  assert.equal(plan.action, model.OrchestrationAction.DISPATCH_CANDIDATE);
  assert.equal(plan.releaseTarget.releaseTag, releaseTags.INITIAL_STABLE_RELEASE_TAG);
}));

test('test_published_older_native_stays_a_verified_noop', withFixture((self) => {
  const candidateDir = path.join(self.tmp, 'candidate');
  writeBridgeCandidate(candidateDir, { releaseTag: 'v0.1.40', releaseRebuild: 0, correlationId: self.correlationId });
  const gateway = publishedCandidateGateway(self, candidateDir, { provenance: self.provenance, correlationId: self.correlationId });
  gateway.jsonRoutes[`repos/${ASSETS_REPOSITORY}/releases?per_page=100`][0].push(alignedReleaseStub('v0.1.41', self.newerNative()));
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp, newerNativeScanned: true });
  assert.equal(plan.action, model.OrchestrationAction.NOOP);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.40');
}));

test('test_backlog_skips_superseded_native_and_publishes_the_newest', withFixture(async (self) => {
  const older = self.provenance;
  const newest = self.newerNative();
  const correlation = runNames.computeCorrelationId(newest);
  const binding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.41', releaseRebuild: 0 });
  const releases = [alignedReleaseStub('v0.1.40', replaceProvenance(newest, { bridgeBuildSha: ADVANCED_BRIDGE_SHA }))];
  const candidateDir = path.join(self.tmp, 'candidate-src');
  writeBridgeCandidate(candidateDir, {
    releaseTag: binding.releaseTag,
    releaseRebuild: binding.releaseRebuild,
    correlationId: correlation,
    upstreamTag: newest.upstreamTag,
    upstreamCommit: newest.upstreamCommit,
    nativeReleaseTag: newest.nativeReleaseTag,
    nativeManifestSha256: newest.nativeManifestSha256,
    nativeCommit: newest.nativeCommit,
  });
  const attestation = attestationBytes(candidateDir);
  const candidateRun = runPayload({
    runId: CANDIDATE_RUN_ID, path: model.CANDIDATE_WORKFLOW_PATH, runName: runNames.candidateRunName(correlation, binding),
  });
  const qualificationRun = runPayload({
    runId: QUALIFICATION_RUN_ID,
    path: model.QUALIFICATION_WORKFLOW_PATH,
    runName: runNames.qualificationRunName(correlation, CANDIDATE_RUN_ID),
  });
  const publishRun = runPayload({
    runId: '701',
    path: model.PUBLISH_WORKFLOW_PATH,
    runName: runNames.publishRunName(correlation, CANDIDATE_RUN_ID, QUALIFICATION_RUN_ID, binding),
    status: 'in_progress',
    conclusion: null,
  });
  const routes = self.routes({ releases });
  for (const [workflowFile, recorded] of [
    [model.CANDIDATE_WORKFLOW_FILE, [candidateRun]],
    [model.QUALIFICATION_WORKFLOW_FILE, [qualificationRun]],
    [model.PUBLISH_WORKFLOW_FILE, []],
  ]) {
    routes[runsKey(workflowFile, assetReleases.workflowHistorySince(releases, newest))] = runsResponse(recorded);
  }
  const publishReadbackKey = runsKey(model.PUBLISH_WORKFLOW_FILE, newest.nativeReleasePublishedAt);
  routes[publishReadbackKey] = runsResponse([]);
  Object.assign(routes, {
    [`${RUNS}/${CANDIDATE_RUN_ID}`]: candidateRun,
    [`${RUNS}/${QUALIFICATION_RUN_ID}`]: qualificationRun,
    [`${RUNS}/${CANDIDATE_RUN_ID}/artifacts?per_page=100`]: artifactInventory({
      runId: CANDIDATE_RUN_ID, name: CANDIDATE_ARTIFACT_NAME, artifactId: CANDIDATE_ARTIFACT_ID,
    }),
    [`${RUNS}/${QUALIFICATION_RUN_ID}/artifacts?per_page=100`]: artifactInventory({
      runId: QUALIFICATION_RUN_ID, name: ATTESTATION_ARTIFACT_NAME, artifactId: QUALIFICATION_ARTIFACT_ID,
    }),
  });
  const gateway = new FakeGateway({
    jsonRoutes: routes,
    blobRoutes: {
      [`${ARTIFACTS}/${CANDIDATE_ARTIFACT_ID}/zip`]: flatZip(directoryMembers(candidateDir)),
      [`${ARTIFACTS}/${QUALIFICATION_ARTIFACT_ID}/zip`]: flatZip({ 'qualification-attestation.json': attestation }),
    },
  });
  onDispatch(gateway, () => {
    gateway.jsonRoutes[publishReadbackKey] = runsResponse([publishRun]);
  });

  const [result, backlog] = await self.runBacklog(gateway, [newest, older]);
  assert.equal(result, 0);
  assert.deepEqual(backlog.errors, []);
  assert.deepEqual(backlog.plans.map((item) => [item.provenance.native_release_tag, item.action]), [
    ['v0.2.0', model.OrchestrationAction.SUPERSEDED],
    ['v0.2.1', model.OrchestrationAction.DISPATCH_PUBLISH],
  ]);
  assert.deepEqual(
    gateway.dispatches.map((record) => [record.workflowFile, record.inputs.release_tag]),
    [[model.PUBLISH_WORKFLOW_FILE, 'v0.1.41']],
  );
}));

test('test_backlog_dry_run_reserves_an_earlier_planned_output_tag', withFixture((self) => {
  const gateway = new FakeGateway({ jsonRoutes: self.routes({ releases: [assetReleaseStub()] }) });
  const plan = driver.advancePipeline(gateway, {
    provenance: self.provenance, workspace: self.tmp, dryRun: true, reservedReleaseTags: new Set(['v0.1.40']),
  });
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.41');
}));

test('test_published_release_is_a_noop_with_no_dispatch', withFixture((self) => {
  const candidateDir = path.join(self.tmp, 'candidate');
  writeBridgeCandidate(candidateDir, { releaseTag: 'v0.1.40', releaseRebuild: 0, correlationId: self.correlationId });
  const gateway = publishedCandidateGateway(self, candidateDir, { provenance: self.provenance, correlationId: self.correlationId });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.NOOP);
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.40');
}));

// A legacy v0.1.40 candidate directory with the historical manifest.
function legacyCandidate(self, name, correlationId) {
  const candidateDir = path.join(self.tmp, name);
  writeBridgeCandidate(candidateDir, {
    releaseTag: 'v0.1.40',
    releaseRebuild: 0,
    correlationId,
    bridgeCommit: LEGACY_BRIDGE_SHA,
    runId: LEGACY_CANDIDATE_RUN_ID,
    upstreamTag: 'v0.3.0',
    upstreamCommit: LEGACY_UPSTREAM_COMMIT,
    nativeReleaseTag: 'v0.3.0',
    nativeManifestSha256: LEGACY_NATIVE_MANIFEST_SHA,
    nativeCommit: LEGACY_NATIVE_COMMIT,
  });
  rewriteLegacyCandidateManifest(candidateDir);
  return candidateDir;
}

test('test_legacy_v0140_publication_is_a_terminal_noop', withFixture((self) => {
  const provenance = makeLegacyV0140Provenance({ bridgeSourceSha: ADVANCED_BRIDGE_SHA });
  const correlationId = runNames.computeCorrelationId(provenance);
  const candidateDir = legacyCandidate(self, 'legacy-v0140-candidate', correlationId);
  // The ordinary candidate path remains strict. Compatibility is selected
  // only after the immutable published-release identity is proven.
  assert.throws(() => loadCandidate(candidateDir), ContractError);
  const gateway = publishedCandidateGateway(self, candidateDir, {
    provenance,
    correlationId,
    publishedManifestCompatibility: [LEGACY_MANUAL_QUALIFICATION_GATES, LEGACY_MANUAL_UNPROVEN_CAPABILITIES],
  });
  const plan = driver.advancePipeline(gateway, { provenance, workspace: self.tmp });
  assert.equal(plan.action, model.OrchestrationAction.NOOP);
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.40');
}));

test('test_governed_source_change_does_not_reuse_legacy_v0140', withFixture((self) => {
  const legacy = makeLegacyV0140Provenance({ bridgeSourceSha: ADVANCED_BRIDGE_SHA });
  const legacyCorrelation = runNames.computeCorrelationId(legacy);
  const candidateDir = legacyCandidate(self, 'legacy-v0140-before-governed-change', legacyCorrelation);
  const gateway = publishedCandidateGateway(self, candidateDir, {
    provenance: legacy,
    correlationId: legacyCorrelation,
    publishedManifestCompatibility: [LEGACY_MANUAL_QUALIFICATION_GATES, LEGACY_MANUAL_UNPROVEN_CAPABILITIES],
  });
  const governed = makeLegacyV0140Provenance({ bridgeSourceSha: ADVANCED_BRIDGE_SHA, bridgeBuildSha: ADVANCED_BRIDGE_SHA });
  gateway.jsonRoutes[runsKey(model.CANDIDATE_WORKFLOW_FILE, '2026-08-20T03:24:11Z')] = runsResponse([]);
  const plan = driver.advancePipeline(gateway, {
    provenance: governed, workspace: path.join(self.tmp, 'governed-change'), dryRun: true,
  });
  assert.equal(plan.action, model.OrchestrationAction.DISPATCH_CANDIDATE);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.41');
  assert.equal(plan.dispatchInputs.bridge_source_sha, ADVANCED_BRIDGE_SHA);
  assert.deepEqual(gateway.dispatches, []);
}));

test('test_two_releases_claiming_one_correlation_fail_closed', withFixture((self) => {
  const body = `Orchestrator correlation: \`${self.correlationId}\``;
  const releases = [
    {
      tag_name: 'v0.1.40', draft: false, prerelease: false, body,
    },
    {
      tag_name: 'v0.1.41', draft: false, prerelease: true, body,
    },
  ];
  const gateway = new FakeGateway({ jsonRoutes: self.routes({ releases }) });
  assert.throws(() => driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp }), ContractError);
}));
