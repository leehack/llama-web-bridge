// Node-only tests of the dispatch correlation guard in
// scripts/release/orchestrator/driver.mjs, which has no Python twin.
//
// Scan 36232509530 (2026-09-26T09:21Z) planned from a stale candidate listing
// that omitted candidate run 36230600813 (created 08:42Z, succeeded 08:55Z),
// chose DISPATCH_CANDIDATE again, and was stopped only because the named-run
// guard rendered the identical name. Every dispatch now first reads its stage
// again by correlation, through the filtered history and the newest
// unfiltered page; a contradicting run makes the pipeline re-plan once with
// that evidence, and a re-plan that still wants the same stage (or meets a
// second stale listing) ends BLOCKED without dispatching or raising.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';

import { BRIDGE_REPOSITORY, ContractError } from '../../../scripts/release/contract.mjs';
import * as orchestrator from '../../../scripts/release/orchestrator/cli.mjs';
import * as driver from '../../../scripts/release/orchestrator/driver.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as runNames from '../../../scripts/release/orchestrator/run_names.mjs';
import * as workflowRuns from '../../../scripts/release/orchestrator/workflow_runs.mjs';
import { ATTESTATION_ARTIFACT_NAME, CANDIDATE_ARTIFACT_NAME } from '../../../scripts/release/qualification.mjs';
import {
  ADVANCED_BRIDGE_SHA, BRIDGE_SHA, CANDIDATE_ARTIFACT_ID, CANDIDATE_RUN_ID, DEFAULT_BRANCH, FakeGateway, HEAD_SHA,
  NATIVE_PUBLISHED_AT, QUALIFICATION_ARTIFACT_ID, makeProvenance, QUALIFICATION_RUN_ID, artifactInventory, assetReleaseStub,
  directoryMembers, flatZip, runPayload, runsResponse, withAdvancePipelineFixture, writeBridgeCandidate,
} from './fixtures.mjs';
import { attestationBytes, onDispatch, runsKey } from './driver_fixtures.mjs';

const RUNS = `repos/${BRIDGE_REPOSITORY}/actions/runs`;
const ARTIFACTS = `repos/${BRIDGE_REPOSITORY}/actions/artifacts`;
// The run the 09:21 scan's listing omitted.
const MISSED_RUN_ID = '36230600813';
const { OrchestrationAction } = model;

// A FakeGateway whose jsonRoutes are the server's true state, with stale
// answers served first for chosen paths: staleFor(path, response, times)
// answers `response` for the next `times` reads of `path` (Infinity for
// every read), then the true route.
class StaleGateway extends FakeGateway {
  constructor(options) {
    super(options);
    this.staleAnswers = new Map();
    this.sequences = new Map();
  }

  staleFor(apiPath, response, times = 1) {
    this.staleAnswers.set(apiPath, { response, remaining: times });
    return this;
  }

  // Answer `responses` for the next reads of `apiPath`, one per read, then
  // the true route.
  sequence(apiPath, responses) {
    this.sequences.set(apiPath, [...responses]);
    return this;
  }

  apiJson(apiPath, options) {
    const queued = this.sequences.get(apiPath);
    if (queued !== undefined && queued.length > 0) {
      this.apiPaths.push(apiPath);
      return queued.shift();
    }
    const stale = this.staleAnswers.get(apiPath);
    if (stale !== undefined && stale.remaining > 0) {
      stale.remaining -= 1;
      this.apiPaths.push(apiPath);
      return stale.response;
    }
    return super.apiJson(apiPath, options);
  }

  reads(apiPath) {
    return this.apiPaths.filter((recorded) => recorded === apiPath).length;
  }
}

const CANDIDATE_LISTING = runsKey(model.CANDIDATE_WORKFLOW_FILE);
const QUALIFICATION_LISTING = runsKey(model.QUALIFICATION_WORKFLOW_FILE);
const PUBLISH_LISTING = runsKey(model.PUBLISH_WORKFLOW_FILE);
const RECENT_CANDIDATES = workflowRuns.recentWorkflowRunsPath(model.CANDIDATE_WORKFLOW_FILE);
const EMPTY = runsResponse([]);

function qualificationName(self, candidateRunId = CANDIDATE_RUN_ID) {
  return runNames.qualificationRunName(self.correlationId, candidateRunId);
}

// The routes and blob a successful candidate run needs to be proven; the
// candidate artifact carries `binding`.
function provenCandidate(self, routes, blobs, { runId = CANDIDATE_RUN_ID, binding = self.binding } = {}) {
  const candidateDir = path.join(self.tmp, `candidate-${runId}-${binding.releaseTag}-${binding.releaseRebuild}`);
  writeBridgeCandidate(candidateDir, {
    releaseTag: binding.releaseTag,
    releaseRebuild: binding.releaseRebuild,
    correlationId: self.correlationId,
    bridgeCommit: binding.bridgeSourceSha,
    runId,
  });
  const run = runPayload({
    runId, path: model.CANDIDATE_WORKFLOW_PATH, runName: runNames.candidateRunName(self.correlationId, binding),
  });
  routes[`${RUNS}/${runId}`] = run;
  routes[`${RUNS}/${runId}/artifacts?per_page=100`] = artifactInventory({
    runId, name: CANDIDATE_ARTIFACT_NAME, artifactId: CANDIDATE_ARTIFACT_ID,
  });
  routes[`repos/${BRIDGE_REPOSITORY}/compare/${HEAD_SHA}...${DEFAULT_BRANCH}`] = { status: 'identical' };
  blobs[`${ARTIFACTS}/${CANDIDATE_ARTIFACT_ID}/zip`] = flatZip(directoryMembers(candidateDir));
  return { run, candidateDir };
}

// The 09:21 state: candidate `runId` succeeded, qualification runs as given,
// and a candidate listing that omits the candidate on its first read.
function staleCandidateGateway(self, { qualificationRuns = [], runId = CANDIDATE_RUN_ID, binding = self.binding } = {}) {
  const routes = {};
  const blobs = {};
  const { run } = provenCandidate(self, routes, blobs, { runId, binding });
  Object.assign(routes, self.routes({ releases: [assetReleaseStub()], candidateRuns: [run], qualificationRuns }));
  return new StaleGateway({ jsonRoutes: routes, blobRoutes: blobs }).staleFor(CANDIDATE_LISTING, EMPTY);
}

// The routes and blobs of a proven candidate and its successful qualification
// (driver_publication_test's qualifiedGateway), with publish runs as given.
function qualifiedGateway(self, publishRuns) {
  const routes = {};
  const blobs = {};
  const { run: candidateRun, candidateDir } = provenCandidate(self, routes, blobs);
  const qualificationRun = runPayload({
    runId: QUALIFICATION_RUN_ID, path: model.QUALIFICATION_WORKFLOW_PATH, runName: qualificationName(self),
  });
  Object.assign(routes, self.routes({
    releases: [assetReleaseStub()], candidateRuns: [candidateRun], qualificationRuns: [qualificationRun], publishRuns,
  }));
  routes[`${RUNS}/${CANDIDATE_RUN_ID}`] = candidateRun;
  routes[`${RUNS}/${QUALIFICATION_RUN_ID}`] = qualificationRun;
  routes[`${RUNS}/${QUALIFICATION_RUN_ID}/artifacts?per_page=100`] = artifactInventory({
    runId: QUALIFICATION_RUN_ID, name: ATTESTATION_ARTIFACT_NAME, artifactId: QUALIFICATION_ARTIFACT_ID,
  });
  blobs[`${ARTIFACTS}/${QUALIFICATION_ARTIFACT_ID}/zip`] = flatZip({ 'qualification-attestation.json': attestationBytes(candidateDir) });
  return new StaleGateway({ jsonRoutes: routes, blobRoutes: blobs });
}

function publishName(self, qualificationRunId = QUALIFICATION_RUN_ID) {
  return runNames.publishRunName(self.correlationId, CANDIDATE_RUN_ID, qualificationRunId, self.binding);
}

function assertReplanned(plan, runDescription) {
  assert.ok(plan.reason.includes('(re-planned once: '), plan.reason);
  assert.ok(plan.reason.includes(runDescription), plan.reason);
}

// --- The 09:21 scan ---------------------------------------------------------------

test('stale listing that omits a successful candidate re-plans instead of dispatching a duplicate', withAdvancePipelineFixture((self) => {
  const qualification = runPayload({
    runId: '36231275207', path: model.QUALIFICATION_WORKFLOW_PATH, runName: qualificationName(self, MISSED_RUN_ID),
    status: 'in_progress', conclusion: null,
  });
  const gateway = staleCandidateGateway(self, { runId: MISSED_RUN_ID, qualificationRuns: [qualification] });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(plan.action, OrchestrationAction.IN_FLIGHT);
  assert.equal(plan.candidateRunId, MISSED_RUN_ID);
  assert.equal(plan.inFlightRunId, '36231275207');
  assertReplanned(plan, `bridge_candidate.yml run(s) ${MISSED_RUN_ID} (success)`);
  assert.ok(plan.reason.includes('planned dispatch_candidate from was stale'), plan.reason);
  // The planning read, the guard's fresh history read, and the re-plan's read.
  assert.equal(gateway.reads(CANDIDATE_LISTING), 3);
  assert.equal(gateway.reads(RECENT_CANDIDATES), 1);
}));

test('stale candidate listing re-plans to the qualification dispatch that is the next step', withAdvancePipelineFixture((self) => {
  const gateway = staleCandidateGateway(self);
  const dispatched = runPayload({
    runId: '4201', path: model.QUALIFICATION_WORKFLOW_PATH, runName: qualificationName(self), status: 'in_progress', conclusion: null,
  });
  onDispatch(gateway, () => {
    gateway.jsonRoutes[QUALIFICATION_LISTING] = runsResponse([dispatched]);
  });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.deepEqual(gateway.dispatches.map((record) => record.workflowFile), [model.QUALIFICATION_WORKFLOW_FILE]);
  assert.deepEqual(gateway.dispatches[0].inputs, {
    orchestrator_correlation_id: self.correlationId,
    candidate_run_id: CANDIDATE_RUN_ID,
  });
  assert.equal(plan.action, OrchestrationAction.DISPATCH_QUALIFICATION);
  assert.equal(plan.dispatchedRunId, '4201');
  assertReplanned(plan, `${CANDIDATE_RUN_ID} (success)`);
}));

test('dry run applies the correlation guard before reporting a dispatch', withAdvancePipelineFixture((self) => {
  const gateway = staleCandidateGateway(self);
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp, dryRun: true });
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(plan.action, OrchestrationAction.DISPATCH_QUALIFICATION);
  assert.equal(plan.dispatchedRunId, null);
  assert.equal(plan.candidateRunId, CANDIDATE_RUN_ID);
}));

test('the backlog scan records the re-planned outcome without an error', withAdvancePipelineFixture(async (self) => {
  const qualification = runPayload({
    runId: '4201', path: model.QUALIFICATION_WORKFLOW_PATH, runName: qualificationName(self), status: 'in_progress', conclusion: null,
  });
  const gateway = staleCandidateGateway(self, { qualificationRuns: [qualification] });
  const [result, backlog] = await self.runBacklog(gateway, [self.provenance]);
  assert.equal(result, 0);
  assert.deepEqual(backlog.errors, []);
  assert.deepEqual(backlog.plans.map((plan) => plan.action), [OrchestrationAction.IN_FLIGHT]);
  assert.deepEqual(gateway.dispatches, []);
}, { orchestrator }));

// --- A missed candidate under another run name ---------------------------------

const OTHER_BINDINGS = Object.freeze({
  'another tag': { releaseTag: 'v0.1.41', releaseRebuild: 0 },
  'another rebuild': { releaseTag: 'v0.1.40-1', releaseRebuild: 1 },
  'another source commit': { bridgeSourceSha: ADVANCED_BRIDGE_SHA },
});

for (const [label, overrides] of Object.entries(OTHER_BINDINGS)) {
  test(`missed in-flight candidate with ${label} blocks the fresh dispatch the name guard would allow`, withAdvancePipelineFixture((self) => {
    const binding = new model.PipelineBinding({
      bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.40', releaseRebuild: 0, ...overrides,
    });
    const missedName = runNames.candidateRunName(self.correlationId, binding);
    assert.notEqual(missedName, self.candidateName);
    const missed = runPayload({
      runId: MISSED_RUN_ID, path: model.CANDIDATE_WORKFLOW_PATH, runName: missedName, status: 'in_progress', conclusion: null,
    });
    const routes = self.routes({ releases: [assetReleaseStub()], candidateRuns: [missed] });
    // The named-run check alone finds nothing for the fresh v0.1.40 name.
    assert.equal(workflowRuns.findNamedRun(new FakeGateway({ jsonRoutes: routes }), {
      workflowFile: model.CANDIDATE_WORKFLOW_FILE,
      workflowPath: model.CANDIDATE_WORKFLOW_PATH,
      defaultBranch: DEFAULT_BRANCH,
      createdSince: NATIVE_PUBLISHED_AT,
      runName: self.candidateName,
    }), null);
    const gateway = new StaleGateway({ jsonRoutes: routes }).staleFor(CANDIDATE_LISTING, EMPTY);
    const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
    assert.deepEqual(gateway.dispatches, []);
    assert.equal(plan.action, OrchestrationAction.IN_FLIGHT);
    assert.equal(plan.inFlightRunId, MISSED_RUN_ID);
    assert.equal(plan.releaseTarget.releaseTag, binding.releaseTag);
    assert.equal(plan.releaseTarget.releaseRebuild, binding.releaseRebuild);
    assertReplanned(plan, `${MISSED_RUN_ID} (in_progress)`);
  }));
}

test('missed successful candidate with another tag is proven and advanced, not duplicated', withAdvancePipelineFixture((self) => {
  const binding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.41', releaseRebuild: 0 });
  const qualification = runPayload({
    runId: '4201', path: model.QUALIFICATION_WORKFLOW_PATH, runName: qualificationName(self), status: 'in_progress', conclusion: null,
  });
  const gateway = staleCandidateGateway(self, { binding, qualificationRuns: [qualification] });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(plan.action, OrchestrationAction.IN_FLIGHT);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.41');
  assert.equal(plan.inFlightRunId, '4201');
}));

test('missed in-flight candidate with the same name is waited for', withAdvancePipelineFixture((self) => {
  const missed = runPayload({
    runId: MISSED_RUN_ID, path: model.CANDIDATE_WORKFLOW_PATH, runName: self.candidateName, status: 'queued', conclusion: null,
  });
  const gateway = new StaleGateway({ jsonRoutes: self.routes({ releases: [assetReleaseStub()], candidateRuns: [missed] }) })
    .staleFor(CANDIDATE_LISTING, EMPTY);
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(plan.action, OrchestrationAction.IN_FLIGHT);
  assert.equal(plan.inFlightRunId, MISSED_RUN_ID);
}));

// --- Failed runs keep their existing rules ----------------------------------------

test('missed failed candidate keeps automatic candidate retries disabled', withAdvancePipelineFixture((self) => {
  const failed = runPayload({
    runId: MISSED_RUN_ID, path: model.CANDIDATE_WORKFLOW_PATH, runName: self.candidateName, conclusion: 'failure',
  });
  const gateway = new StaleGateway({ jsonRoutes: self.routes({ releases: [assetReleaseStub()], candidateRuns: [failed] }) })
    .staleFor(CANDIDATE_LISTING, EMPTY);
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(plan.action, OrchestrationAction.BLOCKED);
  assert.ok(plan.reason.includes('automatic candidate retries are disabled'), plan.reason);
  assertReplanned(plan, `${MISSED_RUN_ID} (failure)`);
}));

test('missed failed qualification keeps automatic qualification retries disabled', withAdvancePipelineFixture((self) => {
  const failed = runPayload({
    runId: '4201', path: model.QUALIFICATION_WORKFLOW_PATH, runName: qualificationName(self), conclusion: 'failure',
  });
  const gateway = staleCandidateGateway(self, { qualificationRuns: [failed] });
  gateway.staleAnswers.clear();
  gateway.staleFor(QUALIFICATION_LISTING, EMPTY);
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(plan.action, OrchestrationAction.BLOCKED);
  assert.ok(plan.reason.includes('automatic qualification retries are disabled'), plan.reason);
  assertReplanned(plan, 'bridge_qualification.yml run(s) 4201 (failure)');
}));

test('missed failed publication does not block the publication retry', withAdvancePipelineFixture((self) => {
  const failed = runPayload({ runId: '700', path: model.PUBLISH_WORKFLOW_PATH, runName: publishName(self), conclusion: 'failure' });
  const gateway = qualifiedGateway(self, [failed]).staleFor(PUBLISH_LISTING, EMPTY);
  const dispatched = runPayload({
    runId: '701', path: model.PUBLISH_WORKFLOW_PATH, runName: publishName(self), status: 'in_progress', conclusion: null,
  });
  onDispatch(gateway, () => {
    gateway.jsonRoutes[PUBLISH_LISTING] = runsResponse([failed, dispatched]);
  });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.deepEqual(gateway.dispatches.map((record) => record.workflowFile), [model.PUBLISH_WORKFLOW_FILE]);
  assert.equal(plan.action, OrchestrationAction.DISPATCH_PUBLISH);
  assert.equal(plan.dispatchedRunId, '701');
  assert.ok(!plan.reason.includes('re-planned'), plan.reason);
}));

// --- Qualification and publication stages -------------------------------------------

test('missed in-flight qualification is waited for', withAdvancePipelineFixture((self) => {
  const qualification = runPayload({
    runId: '4201', path: model.QUALIFICATION_WORKFLOW_PATH, runName: qualificationName(self), status: 'in_progress', conclusion: null,
  });
  const gateway = staleCandidateGateway(self, { qualificationRuns: [qualification] });
  gateway.staleAnswers.clear();
  gateway.staleFor(QUALIFICATION_LISTING, EMPTY);
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(plan.action, OrchestrationAction.IN_FLIGHT);
  assert.equal(plan.inFlightRunId, '4201');
  assertReplanned(plan, 'bridge_qualification.yml run(s) 4201 (in_progress)');
}));

test('missed in-flight publication is waited for', withAdvancePipelineFixture((self) => {
  const inFlight = runPayload({
    runId: '701', path: model.PUBLISH_WORKFLOW_PATH, runName: publishName(self), status: 'in_progress', conclusion: null,
  });
  const gateway = qualifiedGateway(self, [inFlight]).staleFor(PUBLISH_LISTING, EMPTY);
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(plan.action, OrchestrationAction.IN_FLIGHT);
  assert.equal(plan.inFlightRunId, '701');
  assertReplanned(plan, 'publish_assets.yml run(s) 701 (in_progress)');
}));

test('missed successful publication fails closed exactly as a fresh listing does', withAdvancePipelineFixture((self) => {
  const succeeded = runPayload({ runId: '701', path: model.PUBLISH_WORKFLOW_PATH, runName: publishName(self) });
  const fresh = qualifiedGateway(self, [succeeded]);
  const stale = qualifiedGateway(self, [succeeded]).staleFor(PUBLISH_LISTING, EMPTY);
  const expected = /publication run 701 succeeded but no immutable release/u;
  for (const [label, gateway] of [['fresh', fresh], ['stale', stale]]) {
    const workspace = path.join(self.tmp, label);
    fs.mkdirSync(workspace);
    assert.throws(() => driver.advancePipeline(gateway, { provenance: self.provenance, workspace }), expected);
  }
  assert.deepEqual(stale.dispatches, []);
}));

test('re-plan that still wants the same stage ends blocked without dispatching', withAdvancePipelineFixture((self) => {
  // A publication of this exact candidate under another qualification run
  // id: the correlation guard sees it, the exact-name observation does not.
  const other = runPayload({
    runId: '701', path: model.PUBLISH_WORKFLOW_PATH, runName: publishName(self, '4999'), status: 'in_progress', conclusion: null,
  });
  const gateway = qualifiedGateway(self, [other]);
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(plan.action, OrchestrationAction.BLOCKED);
  assert.equal(plan.candidateRunId, CANDIDATE_RUN_ID);
  assert.equal(plan.qualificationRunId, QUALIFICATION_RUN_ID);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.40');
  assert.ok(plan.reason.includes('publish_assets.yml run(s) 701 (in_progress)'), plan.reason);
  assert.ok(plan.reason.includes('the pipeline still planned dispatch_publish'), plan.reason);
  assert.ok(plan.reason.includes('nothing was dispatched'), plan.reason);
}));

test('a second stale listing on the next stage ends blocked after one re-plan', withAdvancePipelineFixture((self) => {
  const qualification = runPayload({
    runId: '4201', path: model.QUALIFICATION_WORKFLOW_PATH, runName: qualificationName(self), status: 'in_progress', conclusion: null,
  });
  const gateway = staleCandidateGateway(self, { qualificationRuns: [qualification] })
    .staleFor(QUALIFICATION_LISTING, EMPTY, Infinity);
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(plan.action, OrchestrationAction.BLOCKED);
  assert.ok(plan.reason.includes(`bridge_candidate.yml run(s) ${CANDIDATE_RUN_ID} (success)`), plan.reason);
  assert.ok(plan.reason.includes('the pipeline planned dispatch_qualification'), plan.reason);
  assert.ok(plan.reason.includes('bridge_qualification.yml run(s) 4201 (in_progress)'), plan.reason);
}));

// --- The two query shapes ----------------------------------------------------------

test('the unfiltered newest page catches a run every filtered read omits', withAdvancePipelineFixture((self) => {
  const qualification = runPayload({
    runId: '4201', path: model.QUALIFICATION_WORKFLOW_PATH, runName: qualificationName(self), status: 'in_progress', conclusion: null,
  });
  const gateway = staleCandidateGateway(self, { qualificationRuns: [qualification] })
    .staleFor(CANDIDATE_LISTING, EMPTY, Infinity);
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.deepEqual(gateway.dispatches, []);
  // The re-plan's own listing is stale too; the merged evidence carries it.
  assert.equal(plan.action, OrchestrationAction.IN_FLIGHT);
  assert.equal(plan.candidateRunId, CANDIDATE_RUN_ID);
}));

test('the named-run check still stops a duplicate both correlation reads miss', withAdvancePipelineFixture((self) => {
  const missed = runPayload({
    runId: MISSED_RUN_ID, path: model.CANDIDATE_WORKFLOW_PATH, runName: self.candidateName, status: 'in_progress', conclusion: null,
  });
  const gateway = new StaleGateway({ jsonRoutes: self.routes({ releases: [assetReleaseStub()], candidateRuns: [missed] }) })
    .staleFor(CANDIDATE_LISTING, EMPTY, 2)
    .staleFor(RECENT_CANDIDATES, EMPTY);
  assert.throws(
    () => driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp }),
    new ContractError(`a run named '${self.candidateName}' already exists; refusing to dispatch a duplicate`),
  );
  assert.deepEqual(gateway.dispatches, []);
}));

test('newest unfiltered page keeps pipeline runs and fails closed on a correlated off-filter run', () => {
  const correlationId = runNames.computeCorrelationId(makeProvenance());
  const other = runNames.computeCorrelationId(makeProvenance({ nativeReleaseTag: 'v0.2.1', upstreamTag: 'v0.2.1' }));
  const binding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.40', releaseRebuild: 0 });
  const name = runNames.candidateRunName(correlationId, binding);
  const otherName = runNames.candidateRunName(other, binding);
  const workflowPath = model.CANDIDATE_WORKFLOW_PATH;
  const read = (runs, { totalCount = runs.length, select = () => false } = {}) => workflowRuns.fetchRecentRuns(
    new FakeGateway({
      jsonRoutes: { [RECENT_CANDIDATES]: { total_count: totalCount, workflow_runs: runs } },
    }),
    {
      workflowFile: model.CANDIDATE_WORKFLOW_FILE,
      workflowPath,
      defaultBranch: DEFAULT_BRANCH,
      correlated: workflowRuns.candidateMatcher(correlationId),
      select,
    },
  );
  const kept = runPayload({ runId: '9', path: workflowPath, runName: name });
  const otherKept = runPayload({ runId: '4', path: workflowPath, runName: otherName });
  // Unrelated runs the server filter drops are skipped unparsed, as are
  // another pipeline's runs outside it even when selected.
  const skipped = [
    { ...runPayload({ runId: '8', path: workflowPath, runName: 'Merge pull request #7', event: 'push' }), actor: { login: 'someone' } },
    runPayload({ runId: '7', path: workflowPath, runName: 'Publish Exact Qualified Bridge Assets', actor: 'github-actions[bot]' }),
    runPayload({ runId: '6', path: workflowPath, runName: otherName, headBranch: 'topic' }),
    runPayload({ runId: '5', path: workflowPath, runName: otherName, actor: 'github-actions[bot]' }),
  ];
  assert.deepEqual(read([kept, otherKept, ...skipped]).map((record) => record.runId), ['9']);
  assert.deepEqual(
    read([kept, otherKept, ...skipped], { select: (title) => title === otherName }).map((record) => record.runId),
    ['9', '4'],
  );
  // Only the newest page is read; a longer history is not an error.
  const older = Array.from({ length: 99 }, (_, index) => runPayload({
    runId: String(1000 + index), path: workflowPath, runName: `v${index}`, event: 'push',
  }));
  assert.deepEqual(read([kept, ...older], { totalCount: 250 }).map((record) => record.runId), ['9']);
  assert.throws(() => read([kept], { totalCount: 2 }), /newest workflow run page .* has 1 records, expected 2/u);
  assert.throws(() => read([kept, kept]), /listed more than once/u);
  // A run named for the guarded pipeline must be exactly what the server
  // filter selects, and parse strictly.
  const offFilter = /workflow run 9 is named for this pipeline but is not a workflow_dispatch run by leehack on 'main'/u;
  assert.throws(() => read([runPayload({ runId: '9', path: workflowPath, runName: name, headBranch: 'topic' })]), offFilter);
  assert.throws(() => read([runPayload({ runId: '9', path: workflowPath, runName: name, event: 'push' })]), offFilter);
  assert.throws(() => read([runPayload({ runId: '9', path: workflowPath, runName: name, actor: 'github-actions[bot]' })]), offFilter);
  assert.throws(() => read([runPayload({ runId: '9', path: workflowPath, runName: name, triggeringActor: 'someone' })]), ContractError);
});

test('fresh reads of one run merge to its later state or fail closed', () => {
  const record = (overrides) => new workflowRuns.RunRecord({
    runId: '9',
    runName: 'n',
    status: 'completed',
    conclusion: 'success',
    headBranch: DEFAULT_BRANCH,
    headSha: HEAD_SHA,
    runAttempt: 1,
    ...overrides,
  });
  const live = record({ status: 'in_progress', conclusion: null });
  const done = record({});
  assert.deepEqual(workflowRuns.mergeRunRecords([live], [done]), [done]);
  assert.deepEqual(workflowRuns.mergeRunRecords([done], [live]), [done]);
  assert.deepEqual(workflowRuns.mergeRunRecords([record({ status: 'queued', conclusion: null })], [live]), [live]);
  const rerun = record({ status: 'in_progress', conclusion: null, runAttempt: 2 });
  assert.deepEqual(workflowRuns.mergeRunRecords([done], [rerun]), [rerun]);
  assert.deepEqual(workflowRuns.mergeRunRecords([rerun], [done]), [rerun]);
  assert.throws(() => workflowRuns.mergeRunRecords([done], [record({ conclusion: 'failure' })]), /contradictory outcomes/u);
  assert.throws(() => workflowRuns.mergeRunRecords([done], [record({ runName: 'other' })]), /contradictory identities/u);
});

// --- Output tags another pipeline still claims ------------------------------------

const TAG_V0140 = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.40', releaseRebuild: 0 });
const TAG_V0141 = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.41', releaseRebuild: 0 });
const OTHER_BUILD_SHA = 'b'.repeat(40);
const RECENT_QUALIFICATIONS = workflowRuns.recentWorkflowRunsPath(model.QUALIFICATION_WORKFLOW_FILE);

// Another pipeline's candidate run claiming `binding`'s tag: the next native
// release of this build (same build), or this native release of another build.
function claimRun(self, { runId = '401', binding = TAG_V0140, otherBuild = false, status = 'in_progress' } = {}) {
  const correlationId = runNames.computeCorrelationId(otherBuild
    ? makeProvenance({ bridgeBuildSha: OTHER_BUILD_SHA })
    : self.newerNative());
  const run = runPayload({
    runId,
    path: model.CANDIDATE_WORKFLOW_PATH,
    runName: runNames.candidateRunName(correlationId, binding),
    status,
    conclusion: status === 'completed' ? 'success' : null,
  });
  return { correlationId, run };
}

// A first-run candidate plan built from a listing that omits `claims`.
function staleClaimGateway(self, claims, { releases = [assetReleaseStub()], qualificationRuns = [] } = {}) {
  return new StaleGateway({
    jsonRoutes: self.routes({ releases, candidateRuns: claims, qualificationRuns }),
  }).staleFor(CANDIDATE_LISTING, EMPTY);
}

function dryRun(self, gateway) {
  return driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp, dryRun: true });
}

test('stale listing that omits an in-flight claim on the fresh tag re-plans to the next free tag', withAdvancePipelineFixture((self) => {
  const { correlationId, run } = claimRun(self);
  const gateway = staleClaimGateway(self, [run]);
  const plan = dryRun(self, gateway);
  assert.equal(plan.action, OrchestrationAction.DISPATCH_CANDIDATE);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.41');
  assert.equal(plan.dispatchInputs.release_tag, 'v0.1.41');
  assertReplanned(plan, `401 (in_progress, claiming v0.1.40 for ${correlationId})`);
  assert.deepEqual(gateway.dispatches, []);
}));

test('the re-planned candidate dispatches once with the next free tag', withAdvancePipelineFixture((self) => {
  const { run } = claimRun(self);
  const gateway = staleClaimGateway(self, [run]);
  const ownName = runNames.candidateRunName(self.correlationId, TAG_V0141);
  onDispatch(gateway, () => {
    gateway.staleAnswers.clear();
    gateway.jsonRoutes[CANDIDATE_LISTING] = runsResponse([run, runPayload({
      runId: '501', path: model.CANDIDATE_WORKFLOW_PATH, runName: ownName, status: 'in_progress', conclusion: null,
    })]);
  });
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.deepEqual(gateway.dispatches.map((record) => record.inputs.release_tag), ['v0.1.41']);
  assert.equal(plan.dispatchedRunId, '501');
}));

test('a claim of another build still counts while its qualification runs', withAdvancePipelineFixture((self) => {
  const { correlationId, run } = claimRun(self, { otherBuild: true, status: 'completed' });
  const qualification = runPayload({
    runId: '402',
    path: model.QUALIFICATION_WORKFLOW_PATH,
    runName: runNames.qualificationRunName(correlationId, '401'),
    status: 'in_progress',
    conclusion: null,
  });
  const gateway = staleClaimGateway(self, [run], { qualificationRuns: [qualification] });
  const plan = dryRun(self, gateway);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.41');
  assertReplanned(plan, `401 (success, claiming v0.1.40 for ${correlationId})`);
}));

test('a completed claim of this build holds until its tag is published', withAdvancePipelineFixture((self) => {
  const { correlationId, run } = claimRun(self, { status: 'completed' });
  const plan = dryRun(self, staleClaimGateway(self, [run]));
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.41');
  assertReplanned(plan, `401 (success, claiming v0.1.40 for ${correlationId})`);
}));

test('a stale-omitted completed claim whose tag is published does not re-plan', withAdvancePipelineFixture((self) => {
  const { run } = claimRun(self, { status: 'completed' });
  const gateway = staleClaimGateway(self, [run], { releases: [assetReleaseStub(), assetReleaseStub('v0.1.40')] });
  const plan = dryRun(self, gateway);
  // The published tag set the floor, so the fresh tag is already above it.
  assert.equal(plan.action, OrchestrationAction.DISPATCH_CANDIDATE);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.41');
  assert.ok(!plan.reason.includes('re-planned'), plan.reason);
  assert.equal(gateway.reads(RECENT_QUALIFICATIONS), 0);
}));

test('a released claim of another build does not block its tag', withAdvancePipelineFixture((self) => {
  const { run } = claimRun(self, { otherBuild: true, status: 'completed' });
  const gateway = staleClaimGateway(self, [run]);
  const plan = dryRun(self, gateway);
  assert.equal(plan.action, OrchestrationAction.DISPATCH_CANDIDATE);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.40');
  assert.ok(!plan.reason.includes('re-planned'), plan.reason);
  // The release was decided from fresh downstream reads, not assumed.
  assert.equal(gateway.reads(RECENT_QUALIFICATIONS), 1);
}));

test('a claim above the fresh tag re-plans too', withAdvancePipelineFixture((self) => {
  const { correlationId, run } = claimRun(self, { binding: TAG_V0141 });
  const plan = dryRun(self, staleClaimGateway(self, [run]));
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.42');
  assertReplanned(plan, `401 (in_progress, claiming v0.1.41 for ${correlationId})`);
}));

test('a tag still claimed after the re-plan ends blocked without dispatching', withAdvancePipelineFixture((self) => {
  const first = claimRun(self);
  const second = claimRun(self, { runId: '403', binding: TAG_V0141, otherBuild: true });
  // The filtered listing stays stale for the re-plan, and the second claim
  // appears only after the first guard's reads.
  const gateway = new StaleGateway({ jsonRoutes: self.routes({ releases: [assetReleaseStub()], candidateRuns: [first.run, second.run] }) })
    .sequence(CANDIDATE_LISTING, [EMPTY, runsResponse([first.run]), EMPTY])
    .sequence(RECENT_CANDIDATES, [runsResponse([first.run])]);
  const plan = driver.advancePipeline(gateway, { provenance: self.provenance, workspace: self.tmp });
  assert.deepEqual(gateway.dispatches, []);
  assert.equal(plan.action, OrchestrationAction.BLOCKED);
  assert.equal(plan.releaseTarget.releaseTag, 'v0.1.41');
  assert.ok(plan.reason.includes(`401 (in_progress, claiming v0.1.40 for ${first.correlationId})`), plan.reason);
  assert.ok(plan.reason.includes(`403 (in_progress, claiming v0.1.41 for ${second.correlationId})`), plan.reason);
}));

test('stale evidence refuses a re-plan whose tag its claims still reach', () => {
  const provenance = makeProvenance();
  const correlationId = runNames.computeCorrelationId(provenance);
  const record = new workflowRuns.RunRecord({
    runId: '401', runName: 'n', status: 'in_progress', conclusion: null, headBranch: DEFAULT_BRANCH, headSha: HEAD_SHA, runAttempt: 1,
  });
  const plan = (releaseTag) => new model.OrchestrationPlan({
    action: OrchestrationAction.DISPATCH_CANDIDATE,
    reason: 'r',
    provenance,
    correlationId,
    releaseTarget: new model.ReleaseTarget({ releaseTag, releaseRebuild: 0 }),
    dispatchWorkflow: model.CANDIDATE_WORKFLOW_FILE,
  });
  const stale = new driver.StaleObservation({
    workflowFile: model.CANDIDATE_WORKFLOW_FILE,
    plan: plan('v0.1.40'),
    evidence: new Map(),
    claims: [{ record, releaseTag: 'v0.1.41-1', correlationId: 'other' }],
  });
  assert.equal(stale.refuses(plan('v0.1.40')), true);
  assert.equal(stale.refuses(plan('v0.1.41')), true);
  assert.equal(stale.refuses(plan('v0.1.42')), false);
});
