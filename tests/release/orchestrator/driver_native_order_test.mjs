// Tests that output-tag order and publication order agree when several stable
// native releases share one scan: an older native never starts a pipeline
// behind a newer scanned native, an older pipeline holding a higher output tag
// than a newer native's is superseded, and a pipeline whose output tag a
// published release overtook is satisfied or blocked instead of retried.

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

// A pipeline of `provenance` at the fixture's build identity bound to
// `releaseTag`: a succeeded candidate whose artifact carries `marker` bytes,
// a succeeded attested qualification when `qualified`, and a failed
// publication when `publishFailed`.
function pipeline(self, {
  provenance, releaseTag, ids, marker = 'same', qualified = true, publishFailed = false,
}) {
  const correlationId = runNames.computeCorrelationId(provenance);
  const binding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag, releaseRebuild: 0 });
  const directory = path.join(self.tmp, `candidate-${provenance.nativeReleaseTag}`);
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

// The older native published at this build identity as `tag`, verifiable,
// with its publication files carrying `marker` bytes.
function olderPublication(self, { tag, marker = 'same' }) {
  const correlationId = runNames.computeCorrelationId(self.provenance);
  const directory = path.join(self.tmp, 'published-older');
  writeBridgeCandidate(directory, {
    releaseTag: tag, releaseRebuild: 0, correlationId, runId: '777', marker: Buffer.from(marker),
  });
  const [release, releaseRoutes, releaseBlobs, attestations] = alignedRelease(directory, {
    tag, provenance: self.provenance, correlationId,
  });
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

test('stranded pipeline overtaken by a differing publication is blocked, not retried', withAdvancePipelineFixture(async (self) => {
  const older = self.provenance;
  const newer = self.newerNative();
  const published = olderPublication(self, { tag: 'v0.1.42', marker: 'older-bytes' });
  const gateway = scanGateway(self, {
    releases: [priorAlignment(self), published.release],
    provenances: [older, newer],
    pipelines: [pipeline(self, {
      provenance: newer, releaseTag: 'v0.1.41', ids: NEWER_IDS, publishFailed: true,
    })],
    ...published,
  });
  const [result, backlog] = await scanBacklog(self, gateway, [older, newer]);
  assert.deepEqual(backlog.errors, []);
  assert.equal(result, 1);
  assert.deepEqual(rows(backlog), [
    ['v0.2.0', OrchestrationAction.NOOP, 'v0.1.42'],
    ['v0.2.1', OrchestrationAction.BLOCKED, 'v0.1.41'],
  ]);
  assert.ok(backlog.plans[1].reason.includes('ordering rollback'));
  assert.ok(backlog.plans[1].reason.includes('differ from v0.1.42'));
  assert.deepEqual(dispatched(backlog), []);
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
