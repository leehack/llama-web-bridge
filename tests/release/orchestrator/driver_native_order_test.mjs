// Tests that output-tag order and publication order agree when several stable
// native releases share one scan: an older native never starts a pipeline
// behind a newer scanned native, an older pipeline holding a higher output tag
// than a newer native's is superseded, and a pipeline whose output tag a
// published release overtook is satisfied, superseded or replaced by a
// recovery pipeline instead of retried.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { test } from 'node:test';

import {
  ASSETS_REPOSITORY, BRIDGE_REPOSITORY, NATIVE_REPOSITORY, parseReleaseTag,
} from '../../../scripts/release/contract.mjs';
import { pyJsonDumps, pyJsonLoads } from '../../../scripts/release/json.mjs';
import * as orchestrator from '../../../scripts/release/orchestrator/cli.mjs';
import * as assetReleases from '../../../scripts/release/orchestrator/asset_releases.mjs';
import * as driver from '../../../scripts/release/orchestrator/driver.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as runNames from '../../../scripts/release/orchestrator/run_names.mjs';
import * as workflowRuns from '../../../scripts/release/orchestrator/workflow_runs.mjs';
import { ATTESTATION_ARTIFACT_NAME, CANDIDATE_ARTIFACT_NAME } from '../../../scripts/release/qualification.mjs';
import {
  ADVANCED_BRIDGE_SHA, BRIDGE_SHA, FakeGateway, alignedReleaseStub, artifactInventory, assetReleaseStub, directoryMembers,
  flatZip, provenanceToDict, runPayload, runsResponse, withAdvancePipelineFixture, writeBridgeCandidate,
} from './fixtures.mjs';
import {
  alignedRelease, attestationBytes, replaceProvenance, runsKey,
} from './driver_fixtures.mjs';

const RUNS = `repos/${BRIDGE_REPOSITORY}/actions/runs`;
const ARTIFACTS = `repos/${BRIDGE_REPOSITORY}/actions/artifacts`;
const { OrchestrationAction } = model;

// Run identities of the older (v0.2.0) and newer (v0.2.1) native's pipelines.
const OLDER_IDS = Object.freeze({
  candidate: '5101', candidateArtifact: 51, qualification: '5102', qualificationArtifact: 52, publish: '5103',
});
const NEWER_IDS = Object.freeze({
  candidate: '5201', candidateArtifact: 61, qualification: '5202', qualificationArtifact: 62, publish: '5203',
});

// A pipeline of `provenance` at the fixture's build identity and `recovery`
// generation bound to `releaseTag`: a succeeded candidate whose artifact
// carries `marker` bytes, a succeeded attested qualification when
// `qualified`, and a failed publication when `publishFailed`.
function pipeline(self, {
  provenance, releaseTag, ids, marker = 'same', qualified = true, publishFailed = false, recovery = 0,
}) {
  const correlationId = runNames.computeCorrelationId(provenance, { recovery });
  const binding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag, releaseRebuild: 0 });
  const directory = path.join(self.tmp, `candidate-${correlationId}`);
  writeBridgeCandidate(directory, {
    releaseTag,
    releaseRebuild: 0,
    correlationId,
    runId: ids.candidate,
    marker: Buffer.from(marker),
    upstreamTag: provenance.upstreamTag,
    upstreamCommit: provenance.upstreamCommit,
    nativeReleaseTag: provenance.nativeReleaseTag,
    nativeManifestSha256: provenance.nativeManifestSha256,
    nativeCommit: provenance.nativeCommit,
  });
  const candidateRun = runPayload({
    runId: ids.candidate, path: model.CANDIDATE_WORKFLOW_PATH, runName: runNames.candidateRunName(correlationId, binding),
  });
  const routes = {
    [`${RUNS}/${ids.candidate}`]: candidateRun,
    [`${RUNS}/${ids.candidate}/artifacts?per_page=100`]: artifactInventory({
      runId: ids.candidate, name: CANDIDATE_ARTIFACT_NAME, artifactId: ids.candidateArtifact,
    }),
  };
  const blobs = { [`${ARTIFACTS}/${ids.candidateArtifact}/zip`]: flatZip(directoryMembers(directory)) };
  const runs = { candidate: [candidateRun], qualification: [], publish: [] };
  if (qualified) {
    const qualificationRun = runPayload({
      runId: ids.qualification,
      path: model.QUALIFICATION_WORKFLOW_PATH,
      runName: runNames.qualificationRunName(correlationId, ids.candidate),
    });
    routes[`${RUNS}/${ids.qualification}`] = qualificationRun;
    routes[`${RUNS}/${ids.qualification}/artifacts?per_page=100`] = artifactInventory({
      runId: ids.qualification, name: ATTESTATION_ARTIFACT_NAME, artifactId: ids.qualificationArtifact,
    });
    blobs[`${ARTIFACTS}/${ids.qualificationArtifact}/zip`] = flatZip({
      'qualification-attestation.json': attestationBytes(directory, {
        candidateRunId: ids.candidate, candidateArtifactId: ids.candidateArtifact, qualificationRunId: ids.qualification,
      }),
    });
    runs.qualification.push(qualificationRun);
  }
  if (publishFailed) {
    runs.publish.push(runPayload({
      runId: ids.publish,
      path: model.PUBLISH_WORKFLOW_PATH,
      runName: runNames.publishRunName(correlationId, ids.candidate, ids.qualification, binding),
      conclusion: 'failure',
    }));
  }
  return { routes, blobs, runs };
}

// The native repository's release record of `provenance`, as its listing
// shows it: a draft when `draft`, its assets.json carrying `manifestSha256`.
function nativeRelease(provenance, { draft = false, manifestSha256 = provenance.nativeManifestSha256 } = {}) {
  return {
    tag_name: provenance.nativeReleaseTag,
    draft,
    prerelease: parseReleaseTag(provenance.nativeReleaseTag).githubPrerelease,
    published_at: provenance.nativeReleasePublishedAt,
    assets: [{ name: 'assets.json', digest: `sha256:${manifestSha256}` }],
  };
}

// A gateway listing `releases` and every pipeline's runs to each provenance's
// run-history window, and `nativeReleases` as the native repository's.
function scanGateway(self, {
  releases, provenances, pipelines = [], releaseRoutes = {}, releaseBlobs = {}, attestations = [], nativeReleases = [],
}) {
  const routes = {
    ...self.routes({ releases }),
    ...releaseRoutes,
    [`repos/${NATIVE_REPOSITORY}/releases?per_page=100`]: [nativeReleases],
  };
  const blobs = { ...releaseBlobs };
  const listed = { candidate: [], qualification: [], publish: [] };
  for (const item of pipelines) {
    Object.assign(routes, item.routes);
    Object.assign(blobs, item.blobs);
    for (const stage of Object.keys(listed)) listed[stage].push(...item.runs[stage]);
  }
  for (const provenance of provenances) {
    const since = assetReleases.workflowHistorySince(releases, provenance);
    routes[runsKey(model.CANDIDATE_WORKFLOW_FILE, since)] = runsResponse(listed.candidate);
    routes[runsKey(model.QUALIFICATION_WORKFLOW_FILE, since)] = runsResponse(listed.qualification);
    routes[runsKey(model.PUBLISH_WORKFLOW_FILE, since)] = runsResponse(listed.publish);
  }
  return new FakeGateway({ jsonRoutes: routes, blobRoutes: blobs, releaseAttestations: attestations });
}

// `orchestrate-backlog --dry-run` over `provenances`: [exit status, plan JSON].
async function scanBacklog(self, gateway, provenances) {
  const provenanceList = path.join(self.tmp, 'release-candidates.json');
  fs.writeFileSync(provenanceList, pyJsonDumps(provenances.map(provenanceToDict)), 'utf8');
  const outputPlan = path.join(self.tmp, 'orchestration-plan.json');
  const result = await orchestrator.main([
    'orchestrate-backlog',
    '--provenance-list-json', provenanceList,
    '--workspace', fs.mkdtempSync(path.join(self.tmp, 'workspace-')),
    '--output-plan-json', outputPlan,
    '--dry-run',
  ], {
    env: {
      ...process.env, GITHUB_EVENT_NAME: 'schedule', GITHUB_ACTOR: 'github-actions', GITHUB_TRIGGERING_ACTOR: 'github-actions',
    },
    createGateway: () => gateway,
    stdout: () => {},
    stderr: () => {},
  });
  return [result, pyJsonLoads(fs.readFileSync(outputPlan, 'utf8'))];
}

const rows = (backlog) => backlog.plans.map((plan) => [plan.provenance.native_release_tag, plan.action, plan.release_tag]);
const dispatched = (backlog) => backlog.plans
  .filter((plan) => plan.dispatch_workflow !== null)
  .map((plan) => [plan.dispatch_workflow, plan.dispatch_inputs.release_tag]);

// The older native's alignment, published under an earlier build identity.
function priorAlignment(self) {
  return alignedReleaseStub('v0.1.40', replaceProvenance(self.provenance, { bridgeBuildSha: ADVANCED_BRIDGE_SHA }));
}

// `provenance` (by default the older native) published at this build
// identity and `recovery` generation as `tag`, verifiable, with its
// publication files carrying `marker` bytes.
function olderPublication(self, {
  tag, marker = 'same', provenance = self.provenance, recovery = 0, releaseId = 4343,
}) {
  const correlationId = runNames.computeCorrelationId(provenance, { recovery });
  const directory = path.join(self.tmp, `published-${correlationId}`);
  writeBridgeCandidate(directory, {
    releaseTag: tag,
    releaseRebuild: 0,
    correlationId,
    runId: '777',
    marker: Buffer.from(marker),
    upstreamTag: provenance.upstreamTag,
    upstreamCommit: provenance.upstreamCommit,
    nativeReleaseTag: provenance.nativeReleaseTag,
    nativeManifestSha256: provenance.nativeManifestSha256,
    nativeCommit: provenance.nativeCommit,
  });
  const [release, releaseRoutes, assetBlobs, attestations] = alignedRelease(directory, {
    tag, provenance, correlationId, releaseId,
  });
  // alignedRelease numbers every release's assets alike; keep each one's own.
  const releaseBlobs = {};
  for (const asset of release.assets) {
    const blob = assetBlobs[`repos/${ASSETS_REPOSITORY}/releases/assets/${asset.id}`];
    asset.id += (releaseId - 4343) * 100;
    releaseBlobs[`repos/${ASSETS_REPOSITORY}/releases/assets/${asset.id}`] = blob;
  }
  return {
    release, releaseRoutes, releaseBlobs, attestations,
  };
}

test('older native at the published alignment is superseded by a newer scanned native', withAdvancePipelineFixture(async (self) => {
  const older = self.provenance;
  const newer = self.newerNative();
  const releases = [priorAlignment(self)];
  const gateway = scanGateway(self, { releases, provenances: [older, newer] });
  const [result, backlog] = await scanBacklog(self, gateway, [older, newer]);
  assert.deepEqual(backlog.errors, []);
  assert.equal(result, 0);
  assert.deepEqual(rows(backlog), [
    ['v0.2.0', OrchestrationAction.SUPERSEDED, null],
    ['v0.2.1', OrchestrationAction.DISPATCH_CANDIDATE, 'v0.1.41'],
  ]);
  assert.ok(backlog.plans[0].reason.includes('newer stable native release'));
  assert.deepEqual(dispatched(backlog), [[model.CANDIDATE_WORKFLOW_FILE, 'v0.1.41']]);
  assert.deepEqual(gateway.dispatches, []);
}));

test('older native with a failed or in-flight candidate is observed, not superseded', withAdvancePipelineFixture((self) => {
  const binding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.41', releaseRebuild: 0 });
  for (const [status, conclusion, action] of [
    ['completed', 'failure', OrchestrationAction.BLOCKED],
    ['in_progress', null, OrchestrationAction.IN_FLIGHT],
  ]) {
    const run = runPayload({
      runId: OLDER_IDS.candidate,
      path: model.CANDIDATE_WORKFLOW_PATH,
      runName: runNames.candidateRunName(self.correlationId, binding),
      status,
      conclusion,
    });
    const gateway = new FakeGateway({ jsonRoutes: self.routes({ releases: [priorAlignment(self)], candidateRuns: [run] }) });
    const plan = driver.advancePipeline(gateway, {
      provenance: self.provenance,
      workspace: self.tmp,
      dryRun: true,
      newerNativeScanned: true,
      scannedProvenances: [self.provenance, self.newerNative()],
    });
    assert.equal(plan.action, action, status);
    assert.equal(plan.releaseTarget.releaseTag, 'v0.1.41', status);
  }
}));

test('fresh older native does not hold back a newer native pipeline with a lower tag', withAdvancePipelineFixture(async (self) => {
  const older = self.provenance;
  const newer = self.newerNative();
  const releases = [priorAlignment(self)];
  const newerPipeline = pipeline(self, { provenance: newer, releaseTag: 'v0.1.41', ids: NEWER_IDS });
  const gateway = scanGateway(self, { releases, provenances: [older, newer], pipelines: [newerPipeline] });
  const [result, backlog] = await scanBacklog(self, gateway, [older, newer]);
  assert.deepEqual(backlog.errors, []);
  assert.equal(result, 0);
  assert.deepEqual(rows(backlog), [
    ['v0.2.0', OrchestrationAction.SUPERSEDED, null],
    ['v0.2.1', OrchestrationAction.DISPATCH_PUBLISH, 'v0.1.41'],
  ]);
  assert.deepEqual(dispatched(backlog), [[model.PUBLISH_WORKFLOW_FILE, 'v0.1.41']]);
}));

test('older pipeline above a newer native pipeline tag is superseded, not published first', withAdvancePipelineFixture(async (self) => {
  const older = self.provenance;
  const newer = self.newerNative();
  const releases = [priorAlignment(self)];
  for (const qualified of [true, false]) {
    const pipelines = [
      pipeline(self, {
        provenance: older, releaseTag: 'v0.1.42', ids: OLDER_IDS, qualified,
      }),
      pipeline(self, { provenance: newer, releaseTag: 'v0.1.41', ids: NEWER_IDS }),
    ];
    const gateway = scanGateway(self, { releases, provenances: [older, newer], pipelines });
    const [result, backlog] = await scanBacklog(self, gateway, [older, newer]);
    assert.deepEqual(backlog.errors, [], `qualified=${qualified}`);
    assert.equal(result, 0, `qualified=${qualified}`);
    assert.deepEqual(rows(backlog), [
      ['v0.2.0', OrchestrationAction.SUPERSEDED, 'v0.1.42'],
      ['v0.2.1', OrchestrationAction.DISPATCH_PUBLISH, 'v0.1.41'],
    ], `qualified=${qualified}`);
    assert.ok(backlog.plans[0].reason.includes('ordering rollback'));
    assert.deepEqual(dispatched(backlog), [[model.PUBLISH_WORKFLOW_FILE, 'v0.1.41']], `qualified=${qualified}`);
  }
}));

test('stranded pipeline overtaken by an identical publication is satisfied, not retried', withAdvancePipelineFixture(async (self) => {
  const older = self.provenance;
  const newer = self.newerNative();
  const published = olderPublication(self, { tag: 'v0.1.42' });
  const releases = [priorAlignment(self), published.release];
  const gateway = scanGateway(self, {
    releases,
    provenances: [older, newer],
    pipelines: [pipeline(self, {
      provenance: newer, releaseTag: 'v0.1.41', ids: NEWER_IDS, publishFailed: true,
    })],
    ...published,
  });
  const [result, backlog] = await scanBacklog(self, gateway, [older, newer]);
  assert.deepEqual(backlog.errors, []);
  assert.equal(result, 0);
  assert.deepEqual(rows(backlog), [
    ['v0.2.0', OrchestrationAction.NOOP, 'v0.1.42'],
    ['v0.2.1', OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE, 'v0.1.42'],
  ]);
  const [, satisfied] = backlog.plans;
  assert.equal(satisfied.candidate_run_id, NEWER_IDS.candidate);
  assert.equal(satisfied.qualification_run_id, NEWER_IDS.qualification);
  assert.ok(satisfied.reason.includes('ordering rollback'));
  assert.ok(satisfied.reason.includes('records native v0.2.0'));
  assert.deepEqual(dispatched(backlog), []);
}));

// Run identities of the newer native's recovery pipelines, by generation,
// production-sized so their run names meet the real length limit.
const recoveryIds = (generation) => Object.freeze({
  candidate: `4066806${generation}531`,
  candidateArtifact: 70 + generation * 2,
  qualification: `4067066${generation}532`,
  qualificationArtifact: 71 + generation * 2,
  publish: `4067425${generation}533`,
});

// The issue #158 end state: the newer native's qualified v0.1.41 failed to
// publish as an ordering rollback after a refresh of the older native took
// v0.1.42 and published first with other bytes.
function strandedScan(self, { pipelines = [], releases = [], qualified = true } = {}) {
  const older = self.provenance;
  const newer = self.newerNative();
  const published = olderPublication(self, { tag: 'v0.1.42', marker: 'older-bytes' });
  const extra = releases.map((release, index) => olderPublication(self, { ...release, releaseId: 4400 + index }));
  const gateway = scanGateway(self, {
    releases: [priorAlignment(self), published.release, ...extra.map((item) => item.release)],
    provenances: [older, newer],
    pipelines: [
      pipeline(self, {
        provenance: newer, releaseTag: 'v0.1.41', ids: NEWER_IDS, qualified, publishFailed: qualified,
      }),
      ...pipelines,
    ],
    releaseRoutes: Object.assign({}, published.releaseRoutes, ...extra.map((item) => item.releaseRoutes)),
    releaseBlobs: Object.assign({}, published.releaseBlobs, ...extra.map((item) => item.releaseBlobs)),
    attestations: [...published.attestations, ...extra.flatMap((item) => item.attestations)],
  });
  return scanBacklog(self, gateway, [older, newer]);
}

test('stranded pipeline overtaken by a differing publication is recovered under a fresh output tag', withAdvancePipelineFixture(async (self) => {
  const newer = self.newerNative();
  const stranded = runNames.computeCorrelationId(newer);
  const recovery = runNames.computeCorrelationId(newer, { recovery: 1 });
  const ids = recoveryIds(1);
  const newerPlan = (backlog) => backlog.plans[1];

  const [result, first] = await strandedScan(self);
  assert.deepEqual(first.errors, []);
  assert.equal(result, 0);
  assert.deepEqual(rows(first), [
    ['v0.2.0', OrchestrationAction.NOOP, 'v0.1.42'],
    ['v0.2.1', OrchestrationAction.DISPATCH_CANDIDATE, 'v0.1.43'],
  ]);
  assert.equal(newerPlan(first).correlation_id, recovery);
  assert.equal(newerPlan(first).dispatch_inputs.orchestrator_correlation_id, recovery);
  assert.equal(newerPlan(first).dispatch_inputs.native_release_tag, 'v0.2.1');
  assert.equal(newerPlan(first).candidate_run_id, null);
  assert.ok(newerPlan(first).reason.includes(`candidate ${NEWER_IDS.candidate}`));
  assert.ok(newerPlan(first).reason.includes('differ from v0.1.42'));
  assert.ok(newerPlan(first).reason.includes(`recovery pipeline ${recovery}`));
  assert.deepEqual(dispatched(first), [[model.CANDIDATE_WORKFLOW_FILE, 'v0.1.43']]);

  // Every later stage is a new first attempt of the recovery pipeline.
  const [, second] = await strandedScan(self, {
    pipelines: [pipeline(self, {
      provenance: newer, releaseTag: 'v0.1.43', ids, qualified: false, recovery: 1,
    })],
  });
  assert.deepEqual(rows(second)[1], ['v0.2.1', OrchestrationAction.DISPATCH_QUALIFICATION, 'v0.1.43']);
  assert.deepEqual(newerPlan(second).dispatch_inputs, {
    candidate_run_id: ids.candidate, orchestrator_correlation_id: recovery,
  });
  const [, third] = await strandedScan(self, {
    pipelines: [pipeline(self, {
      provenance: newer, releaseTag: 'v0.1.43', ids, recovery: 1,
    })],
  });
  assert.deepEqual(rows(third)[1], ['v0.2.1', OrchestrationAction.DISPATCH_PUBLISH, 'v0.1.43']);
  assert.equal(newerPlan(third).dispatch_inputs.candidate_run_id, ids.candidate);
  assert.equal(newerPlan(third).dispatch_inputs.qualification_run_id, ids.qualification);
  assert.equal(newerPlan(third).dispatch_inputs.orchestrator_correlation_id, recovery);

  // Once the recovery pipeline publishes, whatever its bytes, the stranded
  // one is settled by it.
  for (const marker of ['same', 'rebuilt-bytes']) {
    const [settledResult, settled] = await strandedScan(self, {
      pipelines: [pipeline(self, {
        provenance: newer, releaseTag: 'v0.1.43', ids, recovery: 1,
      })],
      releases: [{
        tag: 'v0.1.43', provenance: newer, recovery: 1, marker,
      }],
    });
    assert.deepEqual(settled.errors, [], marker);
    assert.equal(settledResult, 0, marker);
    assert.deepEqual(rows(settled)[1], ['v0.2.1', OrchestrationAction.NOOP, 'v0.1.43'], marker);
    assert.equal(newerPlan(settled).correlation_id, recovery, marker);
    assert.ok(newerPlan(settled).reason.includes(`candidate ${NEWER_IDS.candidate}`), marker);
    assert.deepEqual(dispatched(settled), [], marker);
  }
  assert.notEqual(stranded, recovery);
}));

test('a stale listing that omits a live recovery candidate re-plans instead of dispatching another', withAdvancePipelineFixture((self) => {
  const older = self.provenance;
  const newer = self.newerNative();
  const recovery = runNames.computeCorrelationId(newer, { recovery: 1 });
  const published = olderPublication(self, { tag: 'v0.1.42', marker: 'older-bytes' });
  const stranded = pipeline(self, {
    provenance: newer, releaseTag: 'v0.1.41', ids: NEWER_IDS, publishFailed: true,
  });
  const gateway = scanGateway(self, {
    releases: [priorAlignment(self), published.release],
    provenances: [newer],
    pipelines: [stranded],
    ...published,
  });
  // Only the guard's unfiltered newest page shows the recovery candidate.
  const live = runPayload({
    runId: recoveryIds(1).candidate,
    path: model.CANDIDATE_WORKFLOW_PATH,
    runName: runNames.candidateRunName(recovery, new model.PipelineBinding({
      bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.43', releaseRebuild: 0,
    })),
    status: 'in_progress',
  });
  gateway.jsonRoutes[workflowRuns.recentWorkflowRunsPath(model.CANDIDATE_WORKFLOW_FILE)] = runsResponse([
    live, ...stranded.runs.candidate,
  ]);
  const plan = driver.advancePipeline(gateway, {
    provenance: newer, workspace: self.tmp, dryRun: true, scannedProvenances: [older, newer],
  });
  assert.equal(plan.action, OrchestrationAction.IN_FLIGHT);
  assert.equal(plan.correlationId, recovery);
  assert.equal(plan.dispatchWorkflow, null);
  assert.ok(plan.reason.includes('re-planned once'));
}));

test('an unqualified overtaken candidate is recovered instead of qualified', withAdvancePipelineFixture(async (self) => {
  const [result, backlog] = await strandedScan(self, { qualified: false });
  assert.equal(result, 0);
  assert.deepEqual(rows(backlog)[1], ['v0.2.1', OrchestrationAction.DISPATCH_CANDIDATE, 'v0.1.43']);
  assert.equal(backlog.plans[1].correlation_id, runNames.computeCorrelationId(self.newerNative(), { recovery: 1 }));
}));

test('an overtaken recovery pipeline moves to the next generation until the last one blocks', withAdvancePipelineFixture(async (self) => {
  const newer = self.newerNative();
  // A recovery pipeline of the older native published later still.
  const overtaking = { tag: 'v0.1.50', marker: 'later-older-bytes', recovery: 1 };
  const generations = (count) => Array.from({ length: count }, (_, index) => pipeline(self, {
    provenance: newer, releaseTag: `v0.1.4${index + 3}`, ids: recoveryIds(index + 1), recovery: index + 1, publishFailed: true,
  }));
  for (let count = 1; count < driver.MAX_RECOVERY_GENERATIONS; count += 1) {
    const [result, backlog] = await strandedScan(self, { pipelines: generations(count), releases: [overtaking] });
    assert.equal(result, 0, `count=${count}`);
    assert.deepEqual(rows(backlog)[1], ['v0.2.1', OrchestrationAction.DISPATCH_CANDIDATE, 'v0.1.51'], `count=${count}`);
    assert.equal(backlog.plans[1].correlation_id, runNames.computeCorrelationId(newer, { recovery: count + 1 }));
  }
  const [result, backlog] = await strandedScan(self, {
    pipelines: generations(driver.MAX_RECOVERY_GENERATIONS), releases: [overtaking],
  });
  assert.equal(result, 1);
  const last = driver.MAX_RECOVERY_GENERATIONS;
  assert.deepEqual(rows(backlog)[1], ['v0.2.1', OrchestrationAction.BLOCKED, `v0.1.4${last + 2}`]);
  assert.equal(backlog.plans[1].correlation_id, runNames.computeCorrelationId(newer, { recovery: last }));
  assert.ok(backlog.plans[1].reason.includes(`recovery generation ${last} is the last automatic one`));
  assert.deepEqual(dispatched(backlog), []);
}));

test('a recovery publication of another scanned native is compared like its stranded pipeline', withAdvancePipelineFixture((self) => {
  const older = self.provenance;
  const newer = self.newerNative();
  const published = olderPublication(self, { tag: 'v0.1.42', recovery: 1 });
  const gateway = scanGateway(self, {
    releases: [published.release],
    provenances: [newer],
    pipelines: [pipeline(self, {
      provenance: newer, releaseTag: 'v0.1.41', ids: NEWER_IDS, publishFailed: true,
    })],
    ...published,
  });
  const plan = driver.advancePipeline(gateway, {
    provenance: newer, workspace: self.tmp, dryRun: true, scannedProvenances: [older, newer],
  });
  assert.equal(plan.action, OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.42');
}));

test('a recovery pipeline of a scanned newer native proves its own claim', withAdvancePipelineFixture((self) => {
  const older = self.provenance;
  const newer = self.newerNative();
  const gateway = scanGateway(self, {
    releases: [priorAlignment(self)],
    provenances: [older, newer],
    pipelines: [
      pipeline(self, { provenance: older, releaseTag: 'v0.1.44', ids: OLDER_IDS }),
      pipeline(self, {
        provenance: newer, releaseTag: 'v0.1.43', ids: recoveryIds(1), qualified: false, recovery: 1,
      }),
    ],
  });
  const plan = driver.advancePipeline(gateway, {
    provenance: older, workspace: self.tmp, dryRun: true, newerNativeScanned: true, scannedProvenances: [older, newer],
  });
  assert.equal(plan.action, OrchestrationAction.SUPERSEDED);
  assert.ok(plan.reason.includes('v0.1.43 claimed by the pipeline of newer native release v0.2.1'));
  assert.equal(gateway.apiPaths.includes(`repos/${NATIVE_REPOSITORY}/releases?per_page=100`), false);
}));

test('overtaken pipeline is blocked when the overtaking release is not of a scanned native', withAdvancePipelineFixture((self) => {
  const newer = self.newerNative();
  const published = olderPublication(self, { tag: 'v0.1.42' });
  const gateway = scanGateway(self, {
    releases: [published.release],
    provenances: [newer],
    pipelines: [pipeline(self, {
      provenance: newer, releaseTag: 'v0.1.41', ids: NEWER_IDS, publishFailed: true,
    })],
    ...published,
  });
  const plan = driver.advancePipeline(gateway, {
    provenance: newer, workspace: self.tmp, dryRun: true, scannedProvenances: [newer],
  });
  assert.equal(plan.action, OrchestrationAction.BLOCKED);
  assert.ok(plan.reason.includes('cannot be compared'));
  assert.equal(plan.dispatchWorkflow, null);
  assert.equal(gateway.apiPaths.includes(`repos/${ASSETS_REPOSITORY}/releases/${published.release.id}`), false);
}));

test('overtaken candidate is settled before a qualification is dispatched', withAdvancePipelineFixture((self) => {
  const older = self.provenance;
  const newer = self.newerNative();
  const published = olderPublication(self, { tag: 'v0.1.42' });
  const gateway = scanGateway(self, {
    releases: [published.release],
    provenances: [newer],
    pipelines: [pipeline(self, {
      provenance: newer, releaseTag: 'v0.1.41', ids: NEWER_IDS, qualified: false,
    })],
    ...published,
  });
  const plan = driver.advancePipeline(gateway, {
    provenance: newer, workspace: self.tmp, dryRun: true, scannedProvenances: [older, newer],
  });
  assert.equal(plan.action, OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.42');
  assert.equal(plan.qualificationRunId, null);
  assert.equal(plan.dispatchWorkflow, null);
}));

test('overtaken pipeline behind a publication barrier is settled, not left waiting', withAdvancePipelineFixture((self) => {
  const older = self.provenance;
  const newer = self.newerNative();
  const published = olderPublication(self, { tag: 'v0.1.42' });
  const gateway = scanGateway(self, {
    releases: [published.release],
    provenances: [newer],
    pipelines: [pipeline(self, { provenance: newer, releaseTag: 'v0.1.41', ids: NEWER_IDS })],
    ...published,
  });
  const plan = driver.advancePipeline(gateway, {
    provenance: newer,
    workspace: self.tmp,
    dryRun: true,
    publicationAllowed: false,
    publicationBarrierNativeTag: 'v0.2.0-1',
    scannedProvenances: [older, newer],
  });
  assert.equal(plan.action, OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE);
  assert.equal(plan.qualificationRunId, NEWER_IDS.qualification);
}));

test('older pipeline below a newer native pipeline tag still publishes first', withAdvancePipelineFixture(async (self) => {
  const older = self.provenance;
  const newer = self.newerNative();
  const pipelines = [
    pipeline(self, { provenance: older, releaseTag: 'v0.1.41', ids: OLDER_IDS }),
    pipeline(self, { provenance: newer, releaseTag: 'v0.1.42', ids: NEWER_IDS }),
  ];
  const draft = { ...assetReleaseStub('v0.1.50'), draft: true };
  const gateway = scanGateway(self, { releases: [priorAlignment(self), draft], provenances: [older, newer], pipelines });
  const [result, backlog] = await scanBacklog(self, gateway, [older, newer]);
  assert.deepEqual(backlog.errors, []);
  assert.equal(result, 0);
  assert.deepEqual(rows(backlog), [
    ['v0.2.0', OrchestrationAction.DISPATCH_PUBLISH, 'v0.1.41'],
    ['v0.2.1', OrchestrationAction.WAITING_FOR_PRIOR_PUBLICATION, 'v0.1.42'],
  ]);
  assert.deepEqual(dispatched(backlog), [[model.PUBLISH_WORKFLOW_FILE, 'v0.1.41']]);
}));

test('overtaken older pipeline with differing bytes does not hold back a newer native', withAdvancePipelineFixture(async (self) => {
  const older = self.provenance;
  const middle = self.newerNative();
  const newest = self.newerNative({ nativeReleaseTag: 'v0.2.1-1', nativeReleasePublishedAt: '2026-08-22T12:34:56Z' });
  const published = olderPublication(self, { tag: 'v0.1.42', marker: 'older-bytes' });
  const gateway = scanGateway(self, {
    releases: [priorAlignment(self), published.release],
    provenances: [older, middle, newest],
    pipelines: [
      pipeline(self, {
        provenance: middle, releaseTag: 'v0.1.41', ids: NEWER_IDS, publishFailed: true,
      }),
      pipeline(self, { provenance: newest, releaseTag: 'v0.1.43', ids: OLDER_IDS }),
    ],
    ...published,
  });
  const [result, backlog] = await scanBacklog(self, gateway, [older, middle, newest]);
  assert.deepEqual(backlog.errors, []);
  assert.equal(result, 0);
  assert.deepEqual(rows(backlog), [
    ['v0.2.0', OrchestrationAction.NOOP, 'v0.1.42'],
    ['v0.2.1', OrchestrationAction.SUPERSEDED, 'v0.1.41'],
    ['v0.2.1-1', OrchestrationAction.DISPATCH_PUBLISH, 'v0.1.43'],
  ]);
  assert.deepEqual(dispatched(backlog), [[model.PUBLISH_WORKFLOW_FILE, 'v0.1.43']]);
}));

test('overtaken candidate is satisfied by an identical release of its own native', withAdvancePipelineFixture((self) => {
  const newer = self.newerNative();
  const priorBuild = replaceProvenance(newer, { bridgeSourceSha: ADVANCED_BRIDGE_SHA, bridgeBuildSha: ADVANCED_BRIDGE_SHA });
  const correlationId = runNames.computeCorrelationId(priorBuild);
  const directory = path.join(self.tmp, 'published-prior-build');
  writeBridgeCandidate(directory, {
    releaseTag: 'v0.1.42',
    releaseRebuild: 0,
    correlationId,
    bridgeCommit: ADVANCED_BRIDGE_SHA,
    runId: '777',
    marker: Buffer.from('same'),
    upstreamTag: newer.upstreamTag,
    upstreamCommit: newer.upstreamCommit,
    nativeReleaseTag: newer.nativeReleaseTag,
    nativeManifestSha256: newer.nativeManifestSha256,
    nativeCommit: newer.nativeCommit,
  });
  const [release, releaseRoutes, releaseBlobs, attestations] = alignedRelease(directory, {
    tag: 'v0.1.42', provenance: priorBuild, correlationId,
  });
  const gateway = scanGateway(self, {
    releases: [release],
    provenances: [newer],
    pipelines: [pipeline(self, {
      provenance: newer, releaseTag: 'v0.1.41', ids: NEWER_IDS, publishFailed: true,
    })],
    releaseRoutes,
    releaseBlobs,
    attestations,
  });
  gateway.jsonRoutes[`repos/${BRIDGE_REPOSITORY}/compare/${ADVANCED_BRIDGE_SHA}...main`] = { status: 'ahead' };
  const plan = driver.advancePipeline(gateway, {
    provenance: newer, workspace: self.tmp, dryRun: true, scannedProvenances: [newer],
  });
  assert.equal(plan.action, OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.42');
}));

test('targeted newer-native scan then wider scans keep output tags in native order', withAdvancePipelineFixture(async (self) => {
  const older = self.provenance;
  const newer = self.newerNative();
  const releases = [priorAlignment(self)];
  const [, targeted] = await scanBacklog(self, scanGateway(self, { releases, provenances: [newer] }), [newer]);
  assert.deepEqual(rows(targeted), [['v0.2.1', OrchestrationAction.DISPATCH_CANDIDATE, 'v0.1.41']]);

  const candidate = pipeline(self, {
    provenance: newer, releaseTag: 'v0.1.41', ids: NEWER_IDS, qualified: false,
  });
  const [, woken] = await scanBacklog(self, scanGateway(self, {
    releases, provenances: [older, newer], pipelines: [candidate],
  }), [older, newer]);
  assert.deepEqual(rows(woken), [
    ['v0.2.0', OrchestrationAction.SUPERSEDED, null],
    ['v0.2.1', OrchestrationAction.DISPATCH_QUALIFICATION, 'v0.1.41'],
  ]);

  // A prior-native refresh that already holds a higher tag arrives later.
  const [result, later] = await scanBacklog(self, scanGateway(self, {
    releases,
    provenances: [older, newer],
    pipelines: [
      pipeline(self, { provenance: older, releaseTag: 'v0.1.42', ids: OLDER_IDS }),
      pipeline(self, { provenance: newer, releaseTag: 'v0.1.41', ids: NEWER_IDS }),
    ],
  }), [older, newer]);
  assert.equal(result, 0);
  assert.deepEqual(rows(later), [
    ['v0.2.0', OrchestrationAction.SUPERSEDED, 'v0.1.42'],
    ['v0.2.1', OrchestrationAction.DISPATCH_PUBLISH, 'v0.1.41'],
  ]);
}));

test('targeted older-native scan does not start a pipeline above a newer native pipeline', withAdvancePipelineFixture(async (self) => {
  const older = self.provenance;
  const newer = self.newerNative();
  for (const qualified of [true, false]) {
    const gateway = scanGateway(self, {
      releases: [priorAlignment(self)],
      provenances: [older],
      pipelines: [pipeline(self, {
        provenance: newer, releaseTag: 'v0.1.41', ids: NEWER_IDS, qualified,
      })],
      nativeReleases: [nativeRelease(newer)],
    });
    const [result, backlog] = await scanBacklog(self, gateway, [older]);
    assert.deepEqual(backlog.errors, [], `qualified=${qualified}`);
    assert.equal(result, 0, `qualified=${qualified}`);
    assert.deepEqual(rows(backlog), [['v0.2.0', OrchestrationAction.SUPERSEDED, null]], `qualified=${qualified}`);
    assert.ok(backlog.plans[0].reason.includes('v0.2.1'), `qualified=${qualified}`);
    assert.deepEqual(dispatched(backlog), [], `qualified=${qualified}`);
  }
}));

test('targeted older-native scan supersedes its pipeline above a newer native pipeline', withAdvancePipelineFixture(async (self) => {
  const older = self.provenance;
  const newer = self.newerNative();
  for (const qualified of [true, false]) {
    const gateway = scanGateway(self, {
      releases: [priorAlignment(self)],
      provenances: [older],
      pipelines: [
        pipeline(self, {
          provenance: older, releaseTag: 'v0.1.42', ids: OLDER_IDS, qualified,
        }),
        pipeline(self, {
          provenance: newer, releaseTag: 'v0.1.41', ids: NEWER_IDS, qualified: false,
        }),
      ],
      nativeReleases: [nativeRelease(newer)],
    });
    const [result, backlog] = await scanBacklog(self, gateway, [older]);
    assert.deepEqual(backlog.errors, [], `qualified=${qualified}`);
    assert.equal(result, 0, `qualified=${qualified}`);
    assert.deepEqual(rows(backlog), [['v0.2.0', OrchestrationAction.SUPERSEDED, 'v0.1.42']], `qualified=${qualified}`);
    assert.ok(backlog.plans[0].reason.includes('ordering rollback'), `qualified=${qualified}`);
    assert.deepEqual(dispatched(backlog), [], `qualified=${qualified}`);
  }
}));

test('targeted older-native scan still publishes a pipeline below a newer native pipeline', withAdvancePipelineFixture(async (self) => {
  const older = self.provenance;
  const newer = self.newerNative();
  const gateway = scanGateway(self, {
    releases: [priorAlignment(self)],
    provenances: [older],
    pipelines: [
      pipeline(self, { provenance: older, releaseTag: 'v0.1.41', ids: OLDER_IDS }),
      pipeline(self, {
        provenance: newer, releaseTag: 'v0.1.42', ids: NEWER_IDS, qualified: false,
      }),
    ],
  });
  const [result, backlog] = await scanBacklog(self, gateway, [older]);
  assert.equal(result, 0);
  assert.deepEqual(rows(backlog), [['v0.2.0', OrchestrationAction.DISPATCH_PUBLISH, 'v0.1.41']]);
}));

test('a released newer-native claim of another build does not supersede an older native', withAdvancePipelineFixture(async (self) => {
  const older = self.provenance;
  const newer = replaceProvenance(self.newerNative(), { bridgeBuildSha: ADVANCED_BRIDGE_SHA });
  const binding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.41', releaseRebuild: 0 });
  const failed = runPayload({
    runId: NEWER_IDS.candidate,
    path: model.CANDIDATE_WORKFLOW_PATH,
    runName: runNames.candidateRunName(runNames.computeCorrelationId(newer), binding),
    conclusion: 'failure',
  });
  const gateway = scanGateway(self, {
    releases: [priorAlignment(self)],
    provenances: [older],
    pipelines: [{ routes: {}, blobs: {}, runs: { candidate: [failed], qualification: [], publish: [] } }],
  });
  const [result, backlog] = await scanBacklog(self, gateway, [older]);
  assert.equal(result, 0);
  assert.deepEqual(rows(backlog), [['v0.2.0', OrchestrationAction.DISPATCH_CANDIDATE, 'v0.1.41']]);
}));

// A failed or successful candidate run of `provenance`'s pipeline bound to
// `releaseTag`, with no later stage.
function claimOnly(provenance, { releaseTag = 'v0.1.41', conclusion = 'failure', correlationId = null } = {}) {
  const binding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag, releaseRebuild: 0 });
  const run = runPayload({
    runId: NEWER_IDS.candidate,
    path: model.CANDIDATE_WORKFLOW_PATH,
    runName: runNames.candidateRunName(correlationId ?? runNames.computeCorrelationId(provenance), binding),
    conclusion,
  });
  return { routes: {}, blobs: {}, runs: { candidate: [run], qualification: [], publish: [] } };
}

// A targeted scan of the older native only must fail visibly, naming the claim.
async function assertUnprovenClaimBlocks(self, gateway, { releaseTag, claimed, native }) {
  const [result, backlog] = await scanBacklog(self, gateway, [self.provenance]);
  assert.deepEqual(backlog.errors, []);
  assert.equal(result, 1);
  assert.deepEqual(rows(backlog), [['v0.2.0', OrchestrationAction.BLOCKED, releaseTag]]);
  const { reason } = backlog.plans[0];
  assert.ok(reason.includes(native) && reason.includes(claimed), reason);
  assert.deepEqual(dispatched(backlog), []);
}

test('a claim of a newer native that is no longer published blocks a fresh older native', withAdvancePipelineFixture(async (self) => {
  const newer = self.newerNative();
  for (const [label, nativeReleases] of [
    ['deleted', []],
    ['draft', [nativeRelease(newer, { draft: true })]],
    ['replaced assets', [nativeRelease(newer, { manifestSha256: 'e'.repeat(64) })]],
  ]) {
    for (const conclusion of ['failure', 'success']) {
      const gateway = scanGateway(self, {
        releases: [priorAlignment(self)],
        provenances: [self.provenance],
        pipelines: [claimOnly(newer, { conclusion })],
        nativeReleases,
      });
      await assertUnprovenClaimBlocks(self, gateway, { releaseTag: null, claimed: 'v0.1.41', native: 'v0.2.1' });
      assert.ok(gateway.dispatches.length === 0, `${label} ${conclusion}`);
    }
  }
}));

test('a claim of a newer native that is no longer published blocks an older qualified pipeline', withAdvancePipelineFixture(async (self) => {
  const gateway = scanGateway(self, {
    releases: [priorAlignment(self)],
    provenances: [self.provenance],
    pipelines: [
      pipeline(self, { provenance: self.provenance, releaseTag: 'v0.1.42', ids: OLDER_IDS }),
      claimOnly(self.newerNative()),
    ],
  });
  await assertUnprovenClaimBlocks(self, gateway, { releaseTag: 'v0.1.42', claimed: 'v0.1.41', native: 'v0.2.1' });
}));

test('a crafted claim of a nonexistent newer native blocks instead of superseding', withAdvancePipelineFixture(async (self) => {
  const crafted = `auto-stable-v99.0.0-${'0'.repeat(16)}-build-${BRIDGE_SHA.slice(0, 16)}`;
  const gateway = scanGateway(self, {
    releases: [priorAlignment(self)],
    provenances: [self.provenance],
    pipelines: [claimOnly(self.provenance, { correlationId: crafted })],
    nativeReleases: [nativeRelease(self.newerNative())],
  });
  await assertUnprovenClaimBlocks(self, gateway, { releaseTag: null, claimed: 'v0.1.41', native: 'v99.0.0' });
}));
