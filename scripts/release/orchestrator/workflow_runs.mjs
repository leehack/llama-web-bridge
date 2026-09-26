// Workflow run inventory for the stable release orchestrator, the Node port of
// scripts/release_orchestrator_workflow_runs.py.
//
// Validates server-filtered actions/runs listings, recovers complete run
// history by paginating with stable counts and splitting searches at GitHub's
// 1,000-result cap into closed time windows, and selects at most one live and
// one successful first-attempt run per pipeline stage.

import { BRIDGE_REPOSITORY, ContractError } from '../contract.mjs';
import {
  isDict, isPyInt, pyEquals, pyGet, pyRepr, pyUrlencode,
} from '../json.mjs';
import { PyUtcDatetime, pyFullmatch } from '../python_compat.mjs';
import {
  COMMIT_RE, PyDataclass, PyValueSet, REPOSITORY_OWNER, REQUIRED, SUPPORTED_PIPELINE_WORKFLOW_PATHS, UTC_TIMESTAMP_RE,
  requirePositiveInt, requireStr,
} from './model.mjs';
import { parseCandidateRunName } from './run_names.mjs';

export const MAX_FILTERED_WORKFLOW_RUNS = 100;
export const MAX_GITHUB_FILTERED_SEARCH_RESULTS = 1000;
export const RUN_HISTORY_WINDOW_DAYS = 30;

const IN_FLIGHT_STATUSES = Object.freeze(new Set([
  'queued', 'in_progress', 'waiting', 'requested', 'pending', 'action_required',
]));

// Python int comparisons of a response count (a number or a bigint).
const big = (value) => BigInt(value);

export class RunRecord extends PyDataclass {
  static FIELDS = Object.freeze([
    ['runId', 'run_id', REQUIRED],
    ['runName', 'run_name', REQUIRED],
    ['status', 'status', REQUIRED],
    ['conclusion', 'conclusion', REQUIRED],
    ['headBranch', 'head_branch', REQUIRED],
    ['headSha', 'head_sha', REQUIRED],
    ['runAttempt', 'run_attempt', REQUIRED],
  ]);

  constructor(options) {
    super();
    this.initFields(options);
    Object.freeze(this);
  }

  get inFlight() {
    return this.status !== 'completed';
  }

  get succeeded() {
    return this.status === 'completed' && this.conclusion === 'success';
  }
}

// Validate one server-filtered response proven complete on one page.
//
// The live driver uses the lower-level page parser to paginate stable counts
// and split searches at GitHub's cap. This strict helper remains useful for
// callers and tests that expect exactly one complete response.
export function parseWorkflowRuns(payload, { workflowPath, defaultBranch }) {
  const [totalCount, records] = parseWorkflowRunsResponse(payload, { workflowPath });
  if (big(totalCount) > BigInt(MAX_FILTERED_WORKFLOW_RUNS) || BigInt(records.length) !== big(totalCount)) {
    throw new ContractError(
      `filtered workflow run listing for ${workflowPath} is truncated or `
      + `ambiguous: ${records.length} records for total_count=${totalCount}`,
    );
  }
  requireStr(defaultBranch, 'default branch');
  return records;
}

// _parse_workflow_runs_response: [totalCount, records].
export function parseWorkflowRunsResponse(payload, { workflowPath }) {
  if (!isDict(payload)) throw new ContractError('workflow runs response must be an object containing workflow_runs');
  const totalCount = pyGet(payload, 'total_count');
  if (!isPyInt(totalCount) || big(totalCount) < 0n) {
    throw new ContractError('workflow runs total_count must be a non-negative integer');
  }
  const runs = pyGet(payload, 'workflow_runs');
  if (!Array.isArray(runs)) throw new ContractError('workflow runs response is missing workflow_runs');
  if (runs.length > MAX_FILTERED_WORKFLOW_RUNS) throw new ContractError('workflow runs page exceeds the requested page size');
  const records = new Map();
  for (const run of runs) {
    const record = parseRunRecord(run, { workflowPath });
    if (records.has(record.runId)) throw new ContractError(`workflow run ${record.runId} is listed more than once`);
    records.set(record.runId, record);
  }
  return [totalCount, [...records.values()]];
}

// _parse_run_record.
export function parseRunRecord(run, { workflowPath }) {
  if (!isDict(run)) throw new ContractError('workflow run record must be a JSON object');
  if (!SUPPORTED_PIPELINE_WORKFLOW_PATHS.has(workflowPath)) throw new ContractError(`unsupported workflow path ${pyRepr(workflowPath)}`);
  const runId = String(requirePositiveInt(pyGet(run, 'id'), 'workflow run id'));
  if (!pyEquals(pyGet(run, 'path'), workflowPath)) {
    throw new ContractError(`workflow run ${runId} has path ${pyRepr(pyGet(run, 'path'))}, expected ${pyRepr(workflowPath)}`);
  }
  if (!pyEquals(pyGet(run, 'event'), 'workflow_dispatch')) throw new ContractError(`workflow run ${runId} was not a workflow_dispatch run`);
  // A workflow-level run-name replaces the static workflow label in the
  // Actions API's name field. Bind machine identity to the exact path returned
  // by the workflow-scoped endpoint above; the deterministic display_title
  // below carries the correlation and pipeline inputs.
  const runName = requireStr(pyGet(run, 'display_title'), `workflow run ${runId} display_title`);
  for (const field of ['repository', 'head_repository']) {
    const repository = pyGet(run, field);
    if (!isDict(repository) || !pyEquals(pyGet(repository, 'full_name'), BRIDGE_REPOSITORY)) {
      throw new ContractError(`workflow run ${runId} ${field} must be exactly ${BRIDGE_REPOSITORY}`);
    }
  }
  for (const field of ['actor', 'triggering_actor']) {
    const actor = pyGet(run, field);
    if (!isDict(actor) || !pyEquals(pyGet(actor, 'login'), REPOSITORY_OWNER)) {
      throw new ContractError(`workflow run ${runId} ${field} must be exactly ${REPOSITORY_OWNER}`);
    }
  }
  const status = requireStr(pyGet(run, 'status'), `workflow run ${runId} status`);
  const conclusion = pyGet(run, 'conclusion');
  if (conclusion !== null && typeof conclusion !== 'string') {
    throw new ContractError(`workflow run ${runId} conclusion must be a string or null`);
  }
  if (status === 'completed' && !conclusion) throw new ContractError(`completed workflow run ${runId} has no conclusion`);
  if (status !== 'completed' && !IN_FLIGHT_STATUSES.has(status)) {
    throw new ContractError(`workflow run ${runId} has unsupported status ${pyRepr(status)}`);
  }
  const headSha = requireStr(pyGet(run, 'head_sha'), `workflow run ${runId} head_sha`);
  if (COMMIT_RE.exec(headSha) === null) throw new ContractError(`workflow run ${runId} head_sha must be a 40-hex commit`);
  const headBranch = requireStr(pyGet(run, 'head_branch'), `workflow run ${runId} head_branch`);
  const runAttempt = requirePositiveInt(pyGet(run, 'run_attempt'), `workflow run ${runId} run_attempt`);
  return new RunRecord({ runId, runName, status, conclusion, headBranch, headSha, runAttempt });
}

// `matched` and `unsuccessful` are tuples of RunRecord (frozen arrays).
export class RunSelection extends PyDataclass {
  static FIELDS = Object.freeze([
    ['inFlightRunId', 'in_flight_run_id', REQUIRED],
    ['succeededRunId', 'succeeded_run_id', REQUIRED],
    ['matched', 'matched', Object.freeze([])],
    ['unsuccessful', 'unsuccessful', Object.freeze([])],
  ]);

  constructor(options) {
    super();
    this.initFields(options);
    this.matched = Object.freeze([...this.matched]);
    this.unsuccessful = Object.freeze([...this.unsuccessful]);
    Object.freeze(this);
  }
}

// Select at most one live and one successful run, or fail closed.
export function selectPipelineRuns(runs, { label, matcher, defaultBranch = 'main' }) {
  const matched = [];
  for (const record of runs) {
    if (!matcher(record.runName)) continue;
    if (!pyEquals(record.runAttempt, 1)) {
      throw new ContractError(`${label} run ${record.runId} is attempt ${record.runAttempt}; pipeline stages are first-attempt-only`);
    }
    if (record.headBranch !== defaultBranch) {
      throw new ContractError(`${label} run ${record.runId} ran from ${pyRepr(record.headBranch)}, not the ${pyRepr(defaultBranch)} main line`);
    }
    matched.push(record);
  }

  const inFlight = matched.filter((record) => record.inFlight);
  const succeeded = matched.filter((record) => record.succeeded);
  const unsuccessful = matched.filter((record) => !record.inFlight && !record.succeeded);
  if (inFlight.length > 1) {
    throw new ContractError(`${inFlight.length} duplicate in-flight ${label} runs claim one pipeline stage: ${inFlight.map((record) => record.runId).join(', ')}`);
  }
  if (succeeded.length > 1) {
    throw new ContractError(`${succeeded.length} duplicate successful ${label} runs claim one pipeline stage: ${succeeded.map((record) => record.runId).join(', ')}`);
  }
  if (inFlight.length > 0 && succeeded.length > 0) {
    throw new ContractError(
      `${label} stage has both a successful run (${succeeded[0].runId}) and an `
      + `in-flight run (${inFlight[0].runId}); resolve the duplicate dispatch`,
    );
  }
  return new RunSelection({
    inFlightRunId: inFlight.length > 0 ? inFlight[0].runId : null,
    succeededRunId: succeeded.length > 0 ? succeeded[0].runId : null,
    matched,
    unsuccessful,
  });
}

// _fetch_runs: the complete filtered history since `createdSince`.
export function fetchRuns(gateway, {
  workflowFile, workflowPath, defaultBranch, createdSince,
}) {
  const path = workflowRunsPath({ workflowFile, defaultBranch, createdSince });
  const [complete, records, initialTotal] = fetchCompleteRunQuery(gateway, {
    workflowFile,
    workflowPath,
    defaultBranch,
    createdSince,
    createdUntil: null,
    firstPayload: gateway.apiJson(path),
  });
  if (complete) return records;

  // GitHub caps filtered workflow-run searches at 1,000 results. Partition a
  // long-lived pipeline into closed 30-day windows and recursively split any
  // saturated window, rather than treating a repository-lifetime total as a
  // completeness proof or permanently failing after 1,000 later runs.
  const start = parseUtcTimestamp(createdSince, 'workflow run lower bound');
  const endText = gateway.utcNow();
  const end = parseUtcTimestamp(endText, 'workflow run upper bound');
  if (end.seconds < start.seconds) throw new ContractError('workflow run history upper bound precedes lower bound');
  const collected = new Map();
  let cursor = start;
  const windowSpan = RUN_HISTORY_WINDOW_DAYS * 86400 - 1;
  while (cursor.seconds <= end.seconds) {
    const candidateEnd = cursor.addSeconds(windowSpan);
    const windowEnd = end.seconds < candidateEnd.seconds ? end : candidateEnd;
    for (const record of fetchRunWindow(gateway, {
      workflowFile, workflowPath, defaultBranch, start: cursor, end: windowEnd,
    })) {
      if (collected.has(record.runId)) throw new ContractError(`workflow run ${record.runId} appeared in multiple history windows`);
      collected.set(record.runId, record);
    }
    cursor = windowEnd.addSeconds(1);
  }
  if (BigInt(collected.size) < big(initialTotal)) {
    throw new ContractError(
      `partitioned workflow run history for ${workflowPath} returned `
      + `${collected.size} records, below the initial count ${initialTotal}`,
    );
  }
  return [...collected.values()];
}

// _fetch_run_window: `start` and `end` are PyUtcDatetime values.
export function fetchRunWindow(gateway, {
  workflowFile, workflowPath, defaultBranch, start, end,
}) {
  const startText = formatUtcTimestamp(start);
  const endText = formatUtcTimestamp(end);
  const [complete, records] = fetchCompleteRunQuery(gateway, {
    workflowFile,
    workflowPath,
    defaultBranch,
    createdSince: startText,
    createdUntil: endText,
  });
  if (complete) return records;
  if (start.seconds >= end.seconds) {
    throw new ContractError(`workflow run history is saturated within second ${startText}; exact relevant history cannot be proven`);
  }
  const halfSeconds = Math.floor((end.seconds - start.seconds) / 2);
  const midpoint = start.addSeconds(halfSeconds);
  return [
    ...fetchRunWindow(gateway, {
      workflowFile, workflowPath, defaultBranch, start, end: midpoint,
    }),
    ...fetchRunWindow(gateway, {
      workflowFile, workflowPath, defaultBranch, start: midpoint.addSeconds(1), end,
    }),
  ];
}

// _fetch_complete_run_query: [complete, records, totalCount]. A null
// firstPayload (JSON null included, as Python's `is not None` test has it)
// fetches the first page.
export function fetchCompleteRunQuery(gateway, {
  workflowFile, workflowPath, defaultBranch, createdSince, createdUntil, firstPayload = null,
}) {
  const firstPath = workflowRunsPath({
    workflowFile, defaultBranch, createdSince, createdUntil,
  });
  const payload = firstPayload !== null && firstPayload !== undefined ? firstPayload : gateway.apiJson(firstPath);
  const [totalCount, firstRecords] = parseWorkflowRunsResponse(payload, { workflowPath });
  const expectedFirstPage = big(totalCount) < BigInt(MAX_FILTERED_WORKFLOW_RUNS) ? totalCount : MAX_FILTERED_WORKFLOW_RUNS;
  if (BigInt(firstRecords.length) !== big(expectedFirstPage)) {
    throw new ContractError(
      `filtered workflow run first page for ${workflowPath} has `
      + `${firstRecords.length} records, expected ${expectedFirstPage}`,
    );
  }
  if (big(totalCount) >= BigInt(MAX_GITHUB_FILTERED_SEARCH_RESULTS)) return [false, [], totalCount];

  const total = Number(totalCount);
  const records = new Map(firstRecords.map((record) => [record.runId, record]));
  const pageCount = Math.floor((total + MAX_FILTERED_WORKFLOW_RUNS - 1) / MAX_FILTERED_WORKFLOW_RUNS);
  for (let page = 2; page <= pageCount; page += 1) {
    const path = workflowRunsPath({
      workflowFile, defaultBranch, createdSince, createdUntil, page,
    });
    const [pageTotal, pageRecords] = parseWorkflowRunsResponse(gateway.apiJson(path), { workflowPath });
    if (big(pageTotal) !== big(totalCount)) {
      throw new ContractError(`filtered workflow run total changed during pagination for ${workflowPath}: ${totalCount} -> ${pageTotal}`);
    }
    const expectedPageSize = Math.min(MAX_FILTERED_WORKFLOW_RUNS, total - (page - 1) * MAX_FILTERED_WORKFLOW_RUNS);
    if (pageRecords.length !== expectedPageSize) {
      throw new ContractError(
        `filtered workflow run page ${page} for ${workflowPath} has `
        + `${pageRecords.length} records, expected ${expectedPageSize}`,
      );
    }
    for (const record of pageRecords) {
      if (records.has(record.runId)) throw new ContractError(`workflow run ${record.runId} is listed on multiple pages`);
      records.set(record.runId, record);
    }
  }
  if (records.size !== total) {
    throw new ContractError(
      `filtered workflow run listing for ${workflowPath} is incomplete: `
      + `${records.size} records for total_count=${totalCount}`,
    );
  }
  return [true, [...records.values()], totalCount];
}

// _parse_utc_timestamp: a PyUtcDatetime. A canonical-looking value that is
// not a real time raises strptime's ValueError, as in Python.
export function parseUtcTimestamp(value, label) {
  if (pyFullmatch(UTC_TIMESTAMP_RE, value) === null) throw new ContractError(`${label} must be a canonical UTC timestamp`);
  return PyUtcDatetime.strptime(value);
}

// _format_utc_timestamp.
export function formatUtcTimestamp(value) {
  if (!(value instanceof PyUtcDatetime)) throw new ContractError('workflow run timestamp has no timezone');
  return value.strftime();
}

// _workflow_runs_path.
export function workflowRunsPath({
  workflowFile, defaultBranch, createdSince, createdUntil = null, page = null,
}) {
  if (pyFullmatch(UTC_TIMESTAMP_RE, createdSince) === null) {
    throw new ContractError('workflow run lower bound must be a canonical UTC timestamp');
  }
  if (createdUntil !== null && pyFullmatch(UTC_TIMESTAMP_RE, createdUntil) === null) {
    throw new ContractError('workflow run upper bound must be a canonical UTC timestamp');
  }
  const parameters = [
    ['per_page', String(MAX_FILTERED_WORKFLOW_RUNS)],
    ['event', 'workflow_dispatch'],
    ['branch', defaultBranch],
    ['actor', REPOSITORY_OWNER],
    ['created', createdUntil !== null ? `${createdSince}..${createdUntil}` : `>=${createdSince}`],
  ];
  if (page !== null) {
    if (page < 2) throw new ContractError('workflow run page must be at least 2');
    parameters.push(['page', String(page)]);
  }
  return `repos/${BRIDGE_REPOSITORY}/actions/workflows/${workflowFile}/runs?${pyUrlencode(parameters)}`;
}

// _candidate_matcher.
export function candidateMatcher(correlationId) {
  return (name) => parseCandidateRunName(name, correlationId) !== null;
}

// _resolve_candidate_binding: recover the exact binding prior attempts
// persisted in their run names.
export function resolveCandidateBinding(selection, correlationId) {
  const bindings = new PyValueSet();
  for (const record of selection.matched) {
    const binding = parseCandidateRunName(record.runName, correlationId);
    if (binding !== null) bindings.add(binding);
  }
  if (bindings.size > 1) {
    throw new ContractError(`candidate runs for correlation ${pyRepr(correlationId)} advertise conflicting pipeline bindings`);
  }
  return bindings.size > 0 ? bindings[Symbol.iterator]().next().value : null;
}

// _find_named_run.
export function findNamedRun(gateway, {
  workflowFile, workflowPath, defaultBranch, createdSince, runName,
}) {
  const runs = fetchRuns(gateway, {
    workflowFile, workflowPath, defaultBranch, createdSince,
  });
  const selection = selectPipelineRuns(runs, {
    label: 'dispatched',
    matcher: (name) => name === runName,
    defaultBranch,
  });
  return selection.inFlightRunId || selection.succeededRunId;
}
