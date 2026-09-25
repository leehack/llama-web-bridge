"""Workflow run inventory for the stable release orchestrator.

Validates server-filtered ``actions/runs`` listings, recovers complete run
history by paginating with stable counts and splitting searches at GitHub's
1,000-result cap into closed time windows, and selects at most one live and one
successful first-attempt run per pipeline stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlencode

from release_contract import BRIDGE_REPOSITORY, ContractError
from release_orchestrator_model import (
    PipelineBinding,
    REPOSITORY_OWNER,
    SUPPORTED_PIPELINE_WORKFLOW_PATHS,
    _COMMIT_RE,
    _UTC_TIMESTAMP_RE,
    _require_positive_int,
    _require_str,
)
from release_orchestrator_run_names import parse_candidate_run_name
from release_orchestrator_transport import Gateway


MAX_FILTERED_WORKFLOW_RUNS = 100
MAX_GITHUB_FILTERED_SEARCH_RESULTS = 1000
RUN_HISTORY_WINDOW_DAYS = 30

_IN_FLIGHT_STATUSES = frozenset(
    {"queued", "in_progress", "waiting", "requested", "pending", "action_required"}
)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_name: str
    status: str
    conclusion: str | None
    head_branch: str
    head_sha: str
    run_attempt: int

    @property
    def in_flight(self) -> bool:
        return self.status != "completed"

    @property
    def succeeded(self) -> bool:
        return self.status == "completed" and self.conclusion == "success"


def parse_workflow_runs(
    payload: Any, *, workflow_path: str, default_branch: str
) -> list[RunRecord]:
    """Validate one server-filtered response proven complete on one page.

    The live driver uses the lower-level page parser to paginate stable counts
    and split searches at GitHub's cap. This strict helper remains useful for
    callers and tests that expect exactly one complete response.
    """
    total_count, records = _parse_workflow_runs_response(
        payload, workflow_path=workflow_path
    )
    if total_count > MAX_FILTERED_WORKFLOW_RUNS or len(records) != total_count:
        raise ContractError(
            f"filtered workflow run listing for {workflow_path} is truncated or "
            f"ambiguous: {len(records)} records for total_count={total_count}"
        )
    _require_str(default_branch, "default branch")
    return records


def _parse_workflow_runs_response(
    payload: Any, *, workflow_path: str
) -> tuple[int, list[RunRecord]]:
    if not isinstance(payload, Mapping):
        raise ContractError(
            "workflow runs response must be an object containing workflow_runs"
        )
    total_count = payload.get("total_count")
    if (
        not isinstance(total_count, int)
        or isinstance(total_count, bool)
        or total_count < 0
    ):
        raise ContractError("workflow runs total_count must be a non-negative integer")
    runs = payload.get("workflow_runs")
    if not isinstance(runs, list):
        raise ContractError("workflow runs response is missing workflow_runs")
    if len(runs) > MAX_FILTERED_WORKFLOW_RUNS:
        raise ContractError("workflow runs page exceeds the requested page size")
    records: dict[str, RunRecord] = {}
    for run in runs:
        record = _parse_run_record(run, workflow_path=workflow_path)
        if record.run_id in records:
            raise ContractError(f"workflow run {record.run_id} is listed more than once")
        records[record.run_id] = record
    return total_count, list(records.values())


def _parse_run_record(run: Any, *, workflow_path: str) -> RunRecord:
    if not isinstance(run, Mapping):
        raise ContractError("workflow run record must be a JSON object")
    if workflow_path not in SUPPORTED_PIPELINE_WORKFLOW_PATHS:
        raise ContractError(f"unsupported workflow path {workflow_path!r}")
    run_id = str(_require_positive_int(run.get("id"), "workflow run id"))
    if run.get("path") != workflow_path:
        raise ContractError(
            f"workflow run {run_id} has path {run.get('path')!r}, expected {workflow_path!r}"
        )
    if run.get("event") != "workflow_dispatch":
        raise ContractError(f"workflow run {run_id} was not a workflow_dispatch run")
    # A workflow-level ``run-name`` replaces the static workflow label in the
    # Actions API's ``name`` field.  Bind machine identity to the exact path
    # returned by the workflow-scoped endpoint above; the deterministic
    # ``display_title`` below carries the correlation and pipeline inputs.
    run_name = _require_str(
        run.get("display_title"), f"workflow run {run_id} display_title"
    )
    for field in ("repository", "head_repository"):
        repository = run.get(field)
        if (
            not isinstance(repository, Mapping)
            or repository.get("full_name") != BRIDGE_REPOSITORY
        ):
            raise ContractError(
                f"workflow run {run_id} {field} must be exactly {BRIDGE_REPOSITORY}"
            )
    for field in ("actor", "triggering_actor"):
        actor = run.get(field)
        if not isinstance(actor, Mapping) or actor.get("login") != REPOSITORY_OWNER:
            raise ContractError(
                f"workflow run {run_id} {field} must be exactly {REPOSITORY_OWNER}"
            )
    status = _require_str(run.get("status"), f"workflow run {run_id} status")
    conclusion = run.get("conclusion")
    if conclusion is not None and not isinstance(conclusion, str):
        raise ContractError(f"workflow run {run_id} conclusion must be a string or null")
    if status == "completed" and not conclusion:
        raise ContractError(f"completed workflow run {run_id} has no conclusion")
    if status != "completed" and status not in _IN_FLIGHT_STATUSES:
        raise ContractError(f"workflow run {run_id} has unsupported status {status!r}")
    head_sha = _require_str(run.get("head_sha"), f"workflow run {run_id} head_sha")
    if _COMMIT_RE.fullmatch(head_sha) is None:
        raise ContractError(f"workflow run {run_id} head_sha must be a 40-hex commit")
    return RunRecord(
        run_id=run_id,
        run_name=run_name,
        status=status,
        conclusion=conclusion,
        head_branch=_require_str(
            run.get("head_branch"), f"workflow run {run_id} head_branch"
        ),
        head_sha=head_sha,
        run_attempt=_require_positive_int(
            run.get("run_attempt"), f"workflow run {run_id} run_attempt"
        ),
    )


@dataclass(frozen=True)
class RunSelection:
    in_flight_run_id: str | None
    succeeded_run_id: str | None
    matched: tuple[RunRecord, ...] = ()
    unsuccessful: tuple[RunRecord, ...] = ()


def select_pipeline_runs(
    runs: Sequence[RunRecord],
    *,
    label: str,
    matcher: Callable[[str], bool],
    default_branch: str = "main",
) -> RunSelection:
    """Select at most one live and one successful run, or fail closed."""
    matched: list[RunRecord] = []
    for record in runs:
        if not matcher(record.run_name):
            continue
        if record.run_attempt != 1:
            raise ContractError(
                f"{label} run {record.run_id} is attempt {record.run_attempt}; "
                "pipeline stages are first-attempt-only"
            )
        if record.head_branch != default_branch:
            raise ContractError(
                f"{label} run {record.run_id} ran from {record.head_branch!r}, not the "
                f"{default_branch!r} main line"
            )
        matched.append(record)

    in_flight = [record for record in matched if record.in_flight]
    succeeded = [record for record in matched if record.succeeded]
    unsuccessful = [
        record for record in matched if not record.in_flight and not record.succeeded
    ]
    if len(in_flight) > 1:
        raise ContractError(
            f"{len(in_flight)} duplicate in-flight {label} runs claim one pipeline stage: "
            + ", ".join(record.run_id for record in in_flight)
        )
    if len(succeeded) > 1:
        raise ContractError(
            f"{len(succeeded)} duplicate successful {label} runs claim one pipeline stage: "
            + ", ".join(record.run_id for record in succeeded)
        )
    if in_flight and succeeded:
        raise ContractError(
            f"{label} stage has both a successful run ({succeeded[0].run_id}) and an "
            f"in-flight run ({in_flight[0].run_id}); resolve the duplicate dispatch"
        )
    return RunSelection(
        in_flight_run_id=in_flight[0].run_id if in_flight else None,
        succeeded_run_id=succeeded[0].run_id if succeeded else None,
        matched=tuple(matched),
        unsuccessful=tuple(unsuccessful),
    )


def _fetch_runs(
    gateway: Gateway,
    *,
    workflow_file: str,
    workflow_path: str,
    default_branch: str,
    created_since: str,
) -> list[RunRecord]:
    path = _workflow_runs_path(
        workflow_file=workflow_file,
        default_branch=default_branch,
        created_since=created_since,
    )
    complete, records, initial_total = _fetch_complete_run_query(
        gateway,
        workflow_file=workflow_file,
        workflow_path=workflow_path,
        default_branch=default_branch,
        created_since=created_since,
        created_until=None,
        first_payload=gateway.api_json(path),
    )
    if complete:
        return records

    # GitHub caps filtered workflow-run searches at 1,000 results. Partition a
    # long-lived pipeline into closed 30-day windows and recursively split any
    # saturated window, rather than treating a repository-lifetime total as a
    # completeness proof or permanently failing after 1,000 later runs.
    start = _parse_utc_timestamp(created_since, "workflow run lower bound")
    end_text = gateway.utc_now()
    end = _parse_utc_timestamp(end_text, "workflow run upper bound")
    if end < start:
        raise ContractError("workflow run history upper bound precedes lower bound")
    collected: dict[str, RunRecord] = {}
    cursor = start
    window_span = timedelta(days=RUN_HISTORY_WINDOW_DAYS) - timedelta(seconds=1)
    while cursor <= end:
        window_end = min(cursor + window_span, end)
        for record in _fetch_run_window(
            gateway,
            workflow_file=workflow_file,
            workflow_path=workflow_path,
            default_branch=default_branch,
            start=cursor,
            end=window_end,
        ):
            if record.run_id in collected:
                raise ContractError(
                    f"workflow run {record.run_id} appeared in multiple history windows"
                )
            collected[record.run_id] = record
        cursor = window_end + timedelta(seconds=1)
    if len(collected) < initial_total:
        raise ContractError(
            f"partitioned workflow run history for {workflow_path} returned "
            f"{len(collected)} records, below the initial count {initial_total}"
        )
    return list(collected.values())


def _fetch_run_window(
    gateway: Gateway,
    *,
    workflow_file: str,
    workflow_path: str,
    default_branch: str,
    start: datetime,
    end: datetime,
) -> list[RunRecord]:
    start_text = _format_utc_timestamp(start)
    end_text = _format_utc_timestamp(end)
    complete, records, _ = _fetch_complete_run_query(
        gateway,
        workflow_file=workflow_file,
        workflow_path=workflow_path,
        default_branch=default_branch,
        created_since=start_text,
        created_until=end_text,
    )
    if complete:
        return records
    if start >= end:
        raise ContractError(
            f"workflow run history is saturated within second {start_text}; "
            "exact relevant history cannot be proven"
        )
    half_seconds = int((end - start).total_seconds()) // 2
    midpoint = start + timedelta(seconds=half_seconds)
    return _fetch_run_window(
        gateway,
        workflow_file=workflow_file,
        workflow_path=workflow_path,
        default_branch=default_branch,
        start=start,
        end=midpoint,
    ) + _fetch_run_window(
        gateway,
        workflow_file=workflow_file,
        workflow_path=workflow_path,
        default_branch=default_branch,
        start=midpoint + timedelta(seconds=1),
        end=end,
    )


def _fetch_complete_run_query(
    gateway: Gateway,
    *,
    workflow_file: str,
    workflow_path: str,
    default_branch: str,
    created_since: str,
    created_until: str | None,
    first_payload: Any = None,
) -> tuple[bool, list[RunRecord], int]:
    first_path = _workflow_runs_path(
        workflow_file=workflow_file,
        default_branch=default_branch,
        created_since=created_since,
        created_until=created_until,
    )
    payload = first_payload if first_payload is not None else gateway.api_json(first_path)
    total_count, first_records = _parse_workflow_runs_response(
        payload, workflow_path=workflow_path
    )
    expected_first_page = min(total_count, MAX_FILTERED_WORKFLOW_RUNS)
    if len(first_records) != expected_first_page:
        raise ContractError(
            f"filtered workflow run first page for {workflow_path} has "
            f"{len(first_records)} records, expected {expected_first_page}"
        )
    if total_count >= MAX_GITHUB_FILTERED_SEARCH_RESULTS:
        return False, [], total_count

    records: dict[str, RunRecord] = {
        record.run_id: record for record in first_records
    }
    page_count = (
        total_count + MAX_FILTERED_WORKFLOW_RUNS - 1
    ) // MAX_FILTERED_WORKFLOW_RUNS
    for page in range(2, page_count + 1):
        path = _workflow_runs_path(
            workflow_file=workflow_file,
            default_branch=default_branch,
            created_since=created_since,
            created_until=created_until,
            page=page,
        )
        page_total, page_records = _parse_workflow_runs_response(
            gateway.api_json(path), workflow_path=workflow_path
        )
        if page_total != total_count:
            raise ContractError(
                f"filtered workflow run total changed during pagination for "
                f"{workflow_path}: {total_count} -> {page_total}"
            )
        expected_page_size = min(
            MAX_FILTERED_WORKFLOW_RUNS,
            total_count - (page - 1) * MAX_FILTERED_WORKFLOW_RUNS,
        )
        if len(page_records) != expected_page_size:
            raise ContractError(
                f"filtered workflow run page {page} for {workflow_path} has "
                f"{len(page_records)} records, expected {expected_page_size}"
            )
        for record in page_records:
            if record.run_id in records:
                raise ContractError(
                    f"workflow run {record.run_id} is listed on multiple pages"
                )
            records[record.run_id] = record
    if len(records) != total_count:
        raise ContractError(
            f"filtered workflow run listing for {workflow_path} is incomplete: "
            f"{len(records)} records for total_count={total_count}"
        )
    return True, list(records.values()), total_count


def _parse_utc_timestamp(value: str, label: str) -> datetime:
    if _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise ContractError(f"{label} must be a canonical UTC timestamp")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _format_utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ContractError("workflow run timestamp has no timezone")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _workflow_runs_path(
    *,
    workflow_file: str,
    default_branch: str,
    created_since: str,
    created_until: str | None = None,
    page: int | None = None,
) -> str:
    if _UTC_TIMESTAMP_RE.fullmatch(created_since) is None:
        raise ContractError("workflow run lower bound must be a canonical UTC timestamp")
    if created_until is not None and _UTC_TIMESTAMP_RE.fullmatch(created_until) is None:
        raise ContractError("workflow run upper bound must be a canonical UTC timestamp")
    parameters = {
        "per_page": str(MAX_FILTERED_WORKFLOW_RUNS),
        "event": "workflow_dispatch",
        "branch": default_branch,
        "actor": REPOSITORY_OWNER,
        "created": (
            f"{created_since}..{created_until}"
            if created_until is not None
            else f">={created_since}"
        ),
    }
    if page is not None:
        if page < 2:
            raise ContractError("workflow run page must be at least 2")
        parameters["page"] = str(page)
    query = urlencode(parameters)
    return (
        f"repos/{BRIDGE_REPOSITORY}/actions/workflows/{workflow_file}/runs?{query}"
    )


def _candidate_matcher(correlation_id: str) -> Callable[[str], bool]:
    def matcher(name: str) -> bool:
        return parse_candidate_run_name(name, correlation_id) is not None

    return matcher


def _resolve_candidate_binding(
    selection: RunSelection, correlation_id: str
) -> PipelineBinding | None:
    """Recover the exact binding prior attempts persisted in their run names."""
    bindings = set()
    for record in selection.matched:
        binding = parse_candidate_run_name(record.run_name, correlation_id)
        if binding is not None:
            bindings.add(binding)
    if len(bindings) > 1:
        raise ContractError(
            f"candidate runs for correlation {correlation_id!r} advertise conflicting "
            "pipeline bindings"
        )
    return next(iter(bindings)) if bindings else None


def _find_named_run(
    gateway: Gateway,
    *,
    workflow_file: str,
    workflow_path: str,
    default_branch: str,
    created_since: str,
    run_name: str,
) -> str | None:
    runs = _fetch_runs(
        gateway,
        workflow_file=workflow_file,
        workflow_path=workflow_path,
        default_branch=default_branch,
        created_since=created_since,
    )
    selection = select_pipeline_runs(
        runs,
        label="dispatched",
        matcher=lambda name: name == run_name,
        default_branch=default_branch,
    )
    return selection.in_flight_run_id or selection.succeeded_run_id
