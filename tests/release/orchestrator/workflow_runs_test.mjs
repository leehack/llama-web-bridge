// Tests of scripts/release/orchestrator/workflow_runs.mjs, one test per test
// method of scripts/release_orchestrator_workflow_runs_test.py, with the same
// names and assertions (a Python subTest is one loop iteration).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ContractError } from '../../../scripts/release/contract.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as runNames from '../../../scripts/release/orchestrator/run_names.mjs';
import * as workflowRuns from '../../../scripts/release/orchestrator/workflow_runs.mjs';
import {
  BRIDGE_SHA,
  CANDIDATE_RUN_ID,
  DEFAULT_BRANCH,
  FakeGateway,
  NATIVE_PUBLISHED_AT,
  OWNER,
  makeProvenance,
  runPayload,
  runsResponse,
} from './fixtures.mjs';

// --- WorkflowRunsResponseTest ----------------------------------------------------

// setUp: the candidate path, its run name and one successful run payload.
function responseSetUp() {
  const workflowPath = model.CANDIDATE_WORKFLOW_PATH;
  const correlationId = runNames.computeCorrelationId(makeProvenance());
  const runName = runNames.candidateRunName(
    correlationId,
    new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.40', releaseRebuild: 0 }),
  );
  const run = runPayload({ runId: CANDIDATE_RUN_ID, path: workflowPath, runName });
  return { path: workflowPath, runName, run };
}

function candidateRunsPath(options = {}) {
  return workflowRuns.workflowRunsPath({
    workflowFile: model.CANDIDATE_WORKFLOW_FILE,
    defaultBranch: DEFAULT_BRANCH,
    createdSince: NATIVE_PUBLISHED_AT,
    ...options,
  });
}

function fetchCandidateRuns(gateway, workflowPath) {
  return workflowRuns.fetchRuns(gateway, {
    workflowFile: model.CANDIDATE_WORKFLOW_FILE,
    workflowPath,
    defaultBranch: DEFAULT_BRANCH,
    createdSince: NATIVE_PUBLISHED_AT,
  });
}

test('test_dynamic_correlated_api_name_is_parsed', () => {
  const self = responseSetUp();
  assert.equal(self.run.name, self.runName);
  assert.notEqual(self.run.name, 'Build Exact Bridge Candidate');
  const runs = workflowRuns.parseWorkflowRuns(runsResponse([self.run]), { workflowPath: self.path, defaultBranch: DEFAULT_BRANCH });
  assert.deepEqual(runs.map((record) => record.runId), [CANDIDATE_RUN_ID]);
});

test('test_bare_list_response_is_rejected', () => {
  const self = responseSetUp();
  assert.throws(() => workflowRuns.parseWorkflowRuns([self.run], { workflowPath: self.path, defaultBranch: DEFAULT_BRANCH }), ContractError);
});

test('test_truncated_page_set_fails_closed', () => {
  const self = responseSetUp();
  assert.throws(() => workflowRuns.parseWorkflowRuns(
    { total_count: 5000, workflow_runs: [self.run] },
    { workflowPath: self.path, defaultBranch: DEFAULT_BRANCH },
  ), ContractError);
});

test('test_foreign_workflow_path_fails_closed', () => {
  const self = responseSetUp();
  const foreign = runPayload({
    runId: CANDIDATE_RUN_ID,
    path: '.github/workflows/ci.yml',
    runName: self.runName,
    apiName: self.runName,
  });
  assert.throws(() => workflowRuns.parseWorkflowRuns(runsResponse([foreign]), { workflowPath: self.path, defaultBranch: DEFAULT_BRANCH }), ContractError);
});

test('test_unsupported_workflow_path_fails_closed_even_when_record_matches', () => {
  const self = responseSetUp();
  const unsupportedPath = '.github/workflows/ci.yml';
  const unsupported = { ...self.run, path: unsupportedPath };
  assert.throws(() => workflowRuns.parseWorkflowRuns(
    runsResponse([unsupported]),
    { workflowPath: unsupportedPath, defaultBranch: DEFAULT_BRANCH },
  ), ContractError);
});

test('test_non_owner_actor_or_triggering_actor_fails_closed', () => {
  const self = responseSetUp();
  for (const field of ['actor', 'triggering_actor']) {
    const malformed = { ...self.run, [field]: { login: 'someone-else' } };
    assert.throws(() => workflowRuns.parseWorkflowRuns(
      runsResponse([malformed]),
      { workflowPath: self.path, defaultBranch: DEFAULT_BRANCH },
    ), ContractError, field);
  }
});

test('test_duplicate_run_ids_fail_closed', () => {
  const self = responseSetUp();
  assert.throws(() => workflowRuns.parseWorkflowRuns(
    { total_count: 2, workflow_runs: [self.run, { ...self.run }] },
    { workflowPath: self.path, defaultBranch: DEFAULT_BRANCH },
  ), ContractError);
});

test('test_server_query_bounds_only_relevant_owner_branch_history', () => {
  const runsPath = candidateRunsPath();
  assert.ok(runsPath.includes('per_page=100'));
  assert.ok(runsPath.includes('event=workflow_dispatch'));
  assert.ok(runsPath.includes('branch=main'));
  assert.ok(runsPath.includes(`actor=${OWNER}`));
  assert.ok(runsPath.includes('created=%3E%3D2026-08-19T12%3A34%3A56Z'));
});

test('test_filtered_history_paginates_beyond_one_hundred_without_global_count', () => {
  const self = responseSetUp();
  const payloads = Array.from({ length: 101 }, (_, index) => runPayload({
    runId: String(1000 + index), path: self.path, runName: `unrelated-${index}`,
  }));
  const gateway = new FakeGateway({
    jsonRoutes: {
      [candidateRunsPath()]: { total_count: 101, workflow_runs: payloads.slice(0, 100) },
      [candidateRunsPath({ page: 2 })]: { total_count: 101, workflow_runs: payloads.slice(100) },
    },
  });
  const records = fetchCandidateRuns(gateway, self.path);
  assert.equal(records.length, 101);
  assert.deepEqual(
    new Set(records.map((record) => record.runId)),
    new Set(Array.from({ length: 101 }, (_, index) => String(1000 + index))),
  );
});

test('test_page_total_change_fails_closed', () => {
  const self = responseSetUp();
  const payloads = Array.from({ length: 101 }, (_, index) => runPayload({
    runId: String(2000 + index), path: self.path, runName: `unrelated-${index}`,
  }));
  const gateway = new FakeGateway({
    jsonRoutes: {
      [candidateRunsPath()]: { total_count: 101, workflow_runs: payloads.slice(0, 100) },
      [candidateRunsPath({ page: 2 })]: { total_count: 102, workflow_runs: payloads.slice(100) },
    },
  });
  assert.throws(() => fetchCandidateRuns(gateway, self.path), ContractError);
});

test('test_saturated_search_is_split_into_complete_time_windows', () => {
  const self = responseSetUp();
  const end = '2026-08-19T12:34:57Z';
  const payloads = Array.from({ length: 1000 }, (_, index) => runPayload({
    runId: String(3000 + index), path: self.path, runName: `windowed-${index}`,
  }));
  const routes = {};
  routes[candidateRunsPath()] = { total_count: 1000, workflow_runs: payloads.slice(0, 100) };
  routes[candidateRunsPath({ createdUntil: end })] = { total_count: 1000, workflow_runs: payloads.slice(0, 100) };
  for (const [start, stop, windowStart, windowEnd] of [
    [0, 600, NATIVE_PUBLISHED_AT, NATIVE_PUBLISHED_AT],
    [600, 1000, end, end],
  ]) {
    const total = stop - start;
    let page = 1;
    for (let offset = start; offset < stop; offset += 100, page += 1) {
      const windowPath = workflowRuns.workflowRunsPath({
        workflowFile: model.CANDIDATE_WORKFLOW_FILE,
        defaultBranch: DEFAULT_BRANCH,
        createdSince: windowStart,
        createdUntil: windowEnd,
        page: page > 1 ? page : null,
      });
      routes[windowPath] = { total_count: total, workflow_runs: payloads.slice(offset, Math.min(offset + 100, stop)) };
    }
  }
  const gateway = new FakeGateway({ jsonRoutes: routes, now: end });
  const records = fetchCandidateRuns(gateway, self.path);
  assert.equal(records.length, 1000);
});

// --- RunSelectionTest ------------------------------------------------------------

function selectionRuns(...namesAndStates) {
  const payloads = namesAndStates.map(([runId, name, status, conclusion]) => runPayload({
    runId, path: model.CANDIDATE_WORKFLOW_PATH, runName: name, status, conclusion,
  }));
  return workflowRuns.parseWorkflowRuns(runsResponse(payloads), {
    workflowPath: model.CANDIDATE_WORKFLOW_PATH,
    defaultBranch: DEFAULT_BRANCH,
  });
}

const isTarget = (name) => name === 'target';

test('test_single_success_is_selected', () => {
  const runs = selectionRuns(['501', 'target', 'completed', 'success']);
  const selection = workflowRuns.selectPipelineRuns(runs, { label: 'candidate', matcher: isTarget });
  assert.equal(selection.succeededRunId, '501');
  assert.equal(selection.inFlightRunId, null);
});

test('test_duplicate_success_fails_closed', () => {
  const runs = selectionRuns(['501', 'target', 'completed', 'success'], ['502', 'target', 'completed', 'success']);
  assert.throws(() => workflowRuns.selectPipelineRuns(runs, { label: 'candidate', matcher: isTarget }), ContractError);
});

test('test_duplicate_in_flight_fails_closed', () => {
  const runs = selectionRuns(['501', 'target', 'in_progress', null], ['502', 'target', 'queued', null]);
  assert.throws(() => workflowRuns.selectPipelineRuns(runs, { label: 'candidate', matcher: isTarget }), ContractError);
});

test('test_success_racing_a_new_dispatch_fails_closed', () => {
  const runs = selectionRuns(['501', 'target', 'completed', 'success'], ['502', 'target', 'in_progress', null]);
  assert.throws(() => workflowRuns.selectPipelineRuns(runs, { label: 'candidate', matcher: isTarget }), ContractError);
});

test('test_failed_runs_are_retryable_not_selected', () => {
  const runs = selectionRuns(['501', 'target', 'completed', 'failure']);
  const selection = workflowRuns.selectPipelineRuns(runs, { label: 'candidate', matcher: isTarget });
  assert.equal(selection.succeededRunId, null);
  assert.equal(selection.inFlightRunId, null);
  assert.deepEqual(selection.unsuccessful.map((record) => record.runId), ['501']);
});

test('test_second_attempt_fails_closed', () => {
  const payload = runPayload({
    runId: '501', path: model.CANDIDATE_WORKFLOW_PATH, runName: 'target', runAttempt: 2,
  });
  const runs = workflowRuns.parseWorkflowRuns(runsResponse([payload]), {
    workflowPath: model.CANDIDATE_WORKFLOW_PATH,
    defaultBranch: DEFAULT_BRANCH,
  });
  assert.throws(() => workflowRuns.selectPipelineRuns(runs, { label: 'candidate', matcher: isTarget }), ContractError);
});

test('test_off_main_line_run_fails_closed', () => {
  const payload = runPayload({
    runId: '501', path: model.CANDIDATE_WORKFLOW_PATH, runName: 'target', headBranch: 'attacker',
  });
  const runs = workflowRuns.parseWorkflowRuns(runsResponse([payload]), {
    workflowPath: model.CANDIDATE_WORKFLOW_PATH,
    defaultBranch: DEFAULT_BRANCH,
  });
  assert.throws(() => workflowRuns.selectPipelineRuns(runs, { label: 'candidate', matcher: isTarget }), ContractError);
});

test('test_candidate_matcher_and_selection_fail_closed_on_correlated_malformed_run', () => {
  const correlationId = runNames.computeCorrelationId(makeProvenance());
  const matcher = workflowRuns.candidateMatcher(correlationId);
  const malformedNames = [
    `bridge-candidate ${correlationId} source:not-a-sha tag:v0.1.40 rebuild:0`,
    `bridge-candidate ${correlationId} source:${BRIDGE_SHA} tag:v0.1.40 rebuild:bad`,
    `bridge-candidate ${correlationId} source:${BRIDGE_SHA} tag:invalid_tag rebuild:0`,
    `bridge-candidate ${correlationId} source:${BRIDGE_SHA} tag:v0.1.40`,
    `bridge-candidate ${correlationId} malformed`,
    `bridge-candidate ${correlationId} source:${BRIDGE_SHA} tag:v0.1.40 rebuild:0\x00`,
  ];
  for (const malformedName of malformedNames) {
    assert.throws(() => matcher(malformedName), ContractError, malformedName);
    const runs = selectionRuns(['501', malformedName, 'in_progress', null]);
    assert.throws(() => workflowRuns.selectPipelineRuns(runs, { label: 'candidate', matcher }), ContractError, malformedName);
  }
});

test('test_candidate_matcher_and_selection_ignore_foreign_candidate_run', () => {
  const correlationId = runNames.computeCorrelationId(makeProvenance());
  const matcher = workflowRuns.candidateMatcher(correlationId);
  const foreignNames = [
    `bridge-candidate other-correlation source:${BRIDGE_SHA} tag:v0.1.40 rebuild:0`,
    'bridge-candidate other-correlation source:not-a-sha tag:v0.1.40 rebuild:0',
    'bridge-candidate other-correlation malformed',
    `bridge-candidate ${correlationId}-foreign source:not-a-sha tag:v0.1.40 rebuild:bad`,
    `bridge-candidate-other ${correlationId} malformed`,
    'completely-unrelated-workflow-run',
  ];
  for (const foreignName of foreignNames) {
    assert.equal(matcher(foreignName), false, foreignName);
    const runs = selectionRuns(['501', foreignName, 'in_progress', null]);
    const selection = workflowRuns.selectPipelineRuns(runs, { label: 'candidate', matcher });
    assert.equal(selection.matched.length, 0);
    assert.equal(selection.inFlightRunId, null);
    assert.equal(selection.succeededRunId, null);
  }
});
