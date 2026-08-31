#!/usr/bin/env python3
"""Artifact-driven automatic stable Web bridge release state machine.

Each event-driven scan resolves every stable native release after the immutable
automation baseline, then idempotently advances each three-stage pipeline. A
daily scheduled scan provides repair fallback:

1. Build Exact Bridge Candidate      (.github/workflows/bridge_candidate.yml)
2. Qualify Exact Bridge Candidate    (.github/workflows/bridge_qualification.yml)
3. Publish Exact Qualified Assets    (.github/workflows/publish_assets.yml)

Every transition is proven from downloaded artifact and release bytes. The live
``actions/runs`` API never echoes a run's dispatch inputs, so pipeline state is
carried by a deterministic ``run-name`` that each workflow renders from its own
exact inputs, and every named run is then re-proven against its run record, its
unique artifact, and that artifact's contents before it advances anything.

Every stage is dispatched by this orchestrator, so a discovered eligible native
release reaches a verified immutable Web asset release with no manual step.
Progression after discovery is event-driven: each stage's completion wakes the
next scan. The daily schedule also discovers releases and provides idempotent
repair fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import quote, urlencode

from release_contract import (
    ASSETS_REPOSITORY,
    BRIDGE_REPOSITORY,
    Channel,
    ContractError,
    NATIVE_REPOSITORY,
    _strict_json_loads,
    parse_release_tag,
    parse_upstream_tag,
    require_correlation_id,
    require_repository,
    require_sha256,
    resolve_native_manifest,
    validate_immutable_release_governance,
    validate_publication_environment,
    validate_release_attestation,
    validate_release_identity,
    validate_release_immutability,
)
from release_publication_state import PUBLICATION_FILES
from generate_release_manifest import ARTIFACTS
import release_qualification as rq


CANDIDATE_WORKFLOW_FILE = "bridge_candidate.yml"
QUALIFICATION_WORKFLOW_FILE = "bridge_qualification.yml"
PUBLISH_WORKFLOW_FILE = "publish_assets.yml"
CANDIDATE_WORKFLOW_PATH = rq.CANDIDATE_WORKFLOW_PATH
QUALIFICATION_WORKFLOW_PATH = rq.QUALIFICATION_WORKFLOW_PATH
PUBLISH_WORKFLOW_PATH = f".github/workflows/{PUBLISH_WORKFLOW_FILE}"
SUPPORTED_PIPELINE_WORKFLOW_PATHS = frozenset(
    {
        CANDIDATE_WORKFLOW_PATH,
        QUALIFICATION_WORKFLOW_PATH,
        PUBLISH_WORKFLOW_PATH,
    }
)

REPOSITORY_OWNER = BRIDGE_REPOSITORY.split("/", 1)[0]

# Bridge assets version independently of llama.cpp: an upstream v0.2.0 candidate
# publishes as the next bridge patch (v0.1.39 -> v0.1.40), never as v0.2.0.
INITIAL_STABLE_RELEASE_TAG = "v0.1.0"

CANDIDATE_DISPATCH_INPUTS = (
    "orchestrator_correlation_id",
    "bridge_source_sha",
    "upstream_tag",
    "upstream_commit",
    "native_release_tag",
    "native_manifest_sha256",
    "release_tag",
    "release_rebuild",
    "assets_immutable_releases_enabled",
)

QUALIFICATION_DISPATCH_INPUTS = (
    "orchestrator_correlation_id",
    "candidate_run_id",
)

PUBLISH_DISPATCH_INPUTS = (
    "orchestrator_correlation_id",
    "bridge_source_sha",
    "upstream_tag",
    "upstream_commit",
    "native_release_tag",
    "native_manifest_sha256",
    "release_tag",
    "release_rebuild",
    "assets_repo",
    "publish_approved",
    "candidate_run_id",
    "qualification_run_id",
)

# Every dispatch is checked against these before it leaves the process. GitHub
# rejects a dispatch whose inputs drift from the workflow's declared set, and it
# does so only at dispatch time, in production; the contract suite asserts these
# tuples are byte-for-byte the workflows' declared inputs.
WORKFLOW_DISPATCH_INPUTS: Mapping[str, tuple[str, ...]] = {
    CANDIDATE_WORKFLOW_FILE: CANDIDATE_DISPATCH_INPUTS,
    QUALIFICATION_WORKFLOW_FILE: QUALIFICATION_DISPATCH_INPUTS,
    PUBLISH_WORKFLOW_FILE: PUBLISH_DISPATCH_INPUTS,
}

# Run-name has no documented length guarantee. Keep this internal identity well
# below common database/display limits and accept only printable ASCII emitted
# by the exact input validators. A truncated title can therefore never be
# mistaken for a pipeline identity.
MAX_RUN_NAME_CHARACTERS = 200
MAX_FILTERED_WORKFLOW_RUNS = 100
MAX_GITHUB_FILTERED_SEARCH_RESULTS = 1000
RUN_HISTORY_WINDOW_DAYS = 30
DISPATCH_READBACK_ATTEMPTS = 12
DISPATCH_READBACK_DELAY_SECONDS = 5.0

# v0.2.0-1 was published as the first verified immutable automatic-publication
# baseline (Web assets v0.1.39). Older native releases belong to the historical
# pre-automation series and must not be silently rebuilt by backlog scans.
STABLE_AUTOMATION_BASELINE_NATIVE_TAG = "v0.2.0-1"
STABLE_AUTOMATION_BASELINE_PUBLISHED_AT = "2026-08-25T08:57:12Z"

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_RUN_ID_RE = re.compile(r"[1-9][0-9]*")
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9 ._:/-]+")
_UTC_TIMESTAMP_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z"
)
_IN_FLIGHT_STATUSES = frozenset(
    {"queued", "in_progress", "waiting", "requested", "pending", "action_required"}
)

_CANDIDATE_RUN_NAME_RE = re.compile(
    r"bridge-candidate (?P<correlation_id>\S+) source:(?P<bridge_source_sha>\S+)"
    r" tag:(?P<release_tag>\S+) rebuild:(?P<release_rebuild>\S+)"
)
_PUBLISH_RUN_NAME_RE = re.compile(
    r"publish-assets (?P<correlation_id>\S+) candidate:(?P<candidate_run_id>\S+)"
    r" qualification:(?P<qualification_run_id>\S+) source:(?P<bridge_source_sha>\S+)"
    r" tag:(?P<release_tag>\S+) rebuild:(?P<release_rebuild>\S+)"
)


@dataclass(frozen=True)
class NativeProvenance:
    """Exact identity of the native release this pipeline is aligned to."""

    bridge_source_sha: str
    upstream_tag: str
    upstream_commit: str
    native_repo: str
    native_release_tag: str
    native_commit: str
    native_manifest_sha256: str
    native_release_published_at: str

    def __post_init__(self) -> None:
        for field_name in ("bridge_source_sha", "upstream_commit", "native_commit"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
                raise ContractError(
                    f"{field_name} must be a lowercase full 40-character commit SHA"
                )
        require_sha256(self.native_manifest_sha256, "native_manifest_sha256")
        require_repository(self.native_repo, "native_repo")
        if self.native_repo != NATIVE_REPOSITORY:
            raise ContractError(f"native_repo must be exactly {NATIVE_REPOSITORY}")
        if (
            not isinstance(self.native_release_published_at, str)
            or _UTC_TIMESTAMP_RE.fullmatch(self.native_release_published_at) is None
        ):
            raise ContractError(
                "native_release_published_at must use YYYY-MM-DDTHH:MM:SSZ"
            )
        upstream = parse_upstream_tag(self.upstream_tag)
        native = parse_release_tag(self.native_release_tag, allow_legacy=True)
        if native.channel is not upstream.channel:
            raise ContractError(
                f"native release {self.native_release_tag!r} and upstream tag "
                f"{self.upstream_tag!r} are on different channels"
            )

    @property
    def channel(self) -> Channel:
        return parse_upstream_tag(self.upstream_tag).channel


def require_stable_provenance(provenance: NativeProvenance) -> NativeProvenance:
    """Only the stable channel is orchestrated; everything else is scan-only."""
    if provenance.channel is not Channel.STABLE:
        raise ContractError(
            "the release orchestrator only advances the stable channel, but this "
            f"provenance is {provenance.channel.value} "
            f"({provenance.native_release_tag}@{provenance.upstream_tag})"
        )
    return provenance


def require_orchestration_caller(
    event_name: str, actor: str, triggering_actor: str
) -> None:
    """Keep untrusted callers from turning the environment PAT into a deputy.

    Scheduled executions are authorized by the trusted default-branch workflow.
    A workflow_run continuation and a manual dispatch additionally require both
    GitHub actor identities to be the repository owner before any
    environment-scoped credential is used. A workflow_run event always executes
    the default-branch workflow definition, so the continuation cannot be
    redefined from a pull request or a fork.
    """
    if event_name == "schedule":
        return
    if (
        event_name in ("workflow_dispatch", "workflow_run")
        and actor == REPOSITORY_OWNER
        and triggering_actor == REPOSITORY_OWNER
    ):
        return
    raise ContractError(
        "stable orchestration requires a schedule event, or an owner-initiated "
        "workflow_dispatch or workflow_run continuation with owner actor and "
        "triggering_actor"
    )


@dataclass(frozen=True)
class ReleaseTarget:
    release_tag: str
    release_rebuild: int


@dataclass(frozen=True)
class PipelineBinding:
    """The exact source and output identity one pipeline attempt is pinned to."""

    bridge_source_sha: str
    release_tag: str
    release_rebuild: int

    def __post_init__(self) -> None:
        if _COMMIT_RE.fullmatch(self.bridge_source_sha) is None:
            raise ContractError(
                "bridge_source_sha must be a lowercase full 40-character commit SHA"
            )
        validate_release_identity(
            self.release_tag, self.release_rebuild, "v0.0.0"
        )

    @property
    def release_target(self) -> ReleaseTarget:
        return ReleaseTarget(self.release_tag, self.release_rebuild)


@dataclass(frozen=True)
class PublishedRelease:
    release_id: int
    release_target: ReleaseTarget
    binding: PipelineBinding
    published_at: str


class OrchestrationAction(str, Enum):
    NOOP = "noop"
    IN_FLIGHT = "in_flight"
    WAITING_FOR_PRIOR_PUBLICATION = "waiting_for_prior_publication"
    DISPATCH_CANDIDATE = "dispatch_candidate"
    DISPATCH_QUALIFICATION = "dispatch_qualification"
    DISPATCH_PUBLISH = "dispatch_publish"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class OrchestrationPlan:
    action: OrchestrationAction
    reason: str
    provenance: NativeProvenance
    correlation_id: str
    release_target: ReleaseTarget | None = None
    candidate_run_id: str | None = None
    qualification_run_id: str | None = None
    in_flight_workflow: str | None = None
    in_flight_run_id: str | None = None
    dispatch_workflow: str | None = None
    dispatch_ref: str | None = None
    dispatch_run_name: str | None = None
    dispatch_inputs: dict[str, str] | None = None
    dispatched_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        target = self.release_target
        return {
            "schema_version": 1,
            "action": self.action.value,
            "reason": self.reason,
            "correlation_id": self.correlation_id,
            "provenance": {
                "bridge_source_sha": self.provenance.bridge_source_sha,
                "upstream_tag": self.provenance.upstream_tag,
                "upstream_commit": self.provenance.upstream_commit,
                "native_repo": self.provenance.native_repo,
                "native_release_tag": self.provenance.native_release_tag,
                "native_commit": self.provenance.native_commit,
                "native_manifest_sha256": self.provenance.native_manifest_sha256,
                "native_release_published_at": (
                    self.provenance.native_release_published_at
                ),
            },
            "release_tag": target.release_tag if target else None,
            "release_rebuild": target.release_rebuild if target else None,
            "candidate_run_id": self.candidate_run_id,
            "qualification_run_id": self.qualification_run_id,
            "in_flight_workflow": self.in_flight_workflow,
            "in_flight_run_id": self.in_flight_run_id,
            "dispatch_workflow": self.dispatch_workflow,
            "dispatch_ref": self.dispatch_ref,
            "dispatch_run_name": self.dispatch_run_name,
            "dispatch_inputs": self.dispatch_inputs,
            "dispatched_run_id": self.dispatched_run_id,
        }


@dataclass(frozen=True)
class PipelineObservation:
    """Everything already proven about this correlation's pipeline."""

    published: PublishedRelease | None = None
    binding: PipelineBinding | None = None
    fresh_binding: PipelineBinding | None = None
    candidate_in_flight_run_id: str | None = None
    candidate_run_id: str | None = None
    qualification_in_flight_run_id: str | None = None
    qualification_run_id: str | None = None
    publish_in_flight_run_id: str | None = None
    publish_succeeded_run_id: str | None = None
    publish_retry: bool = False


# --------------------------------------------------------------------------
# Deterministic correlation and run names
# --------------------------------------------------------------------------


def compute_correlation_id(provenance: NativeProvenance) -> str:
    """Derive one stable pipeline identity from the native release alone.

    Deliberately independent of the bridge source commit: main advances while a
    pipeline is in flight, and a correlation that moved with it would orphan the
    candidate and let a later event or repair scan dispatch a duplicate.
    """
    raw = (
        f"auto-stable-{provenance.native_release_tag}"
        f"-{provenance.native_manifest_sha256[:16]}"
    )
    return require_correlation_id(raw)


def candidate_run_name(correlation_id: str, binding: PipelineBinding) -> str:
    require_correlation_id(correlation_id)
    return _require_run_name(
        f"bridge-candidate {correlation_id}"
        f" source:{binding.bridge_source_sha}"
        f" tag:{binding.release_tag}"
        f" rebuild:{binding.release_rebuild}"
    )


def qualification_run_name(correlation_id: str, candidate_run_id: str) -> str:
    require_correlation_id(correlation_id)
    if _RUN_ID_RE.fullmatch(candidate_run_id) is None:
        raise ContractError("candidate_run_id must be a positive integer")
    return _require_run_name(
        f"bridge-qualification {correlation_id} candidate:{candidate_run_id}"
    )


def publish_run_name(
    correlation_id: str,
    candidate_run_id: str,
    qualification_run_id: str,
    binding: PipelineBinding,
) -> str:
    require_correlation_id(correlation_id)
    for label, value in (
        ("candidate_run_id", candidate_run_id),
        ("qualification_run_id", qualification_run_id),
    ):
        if _RUN_ID_RE.fullmatch(value) is None:
            raise ContractError(f"{label} must be a positive integer")
    return _require_run_name(
        f"publish-assets {correlation_id}"
        f" candidate:{candidate_run_id}"
        f" qualification:{qualification_run_id}"
        f" source:{binding.bridge_source_sha}"
        f" tag:{binding.release_tag}"
        f" rebuild:{binding.release_rebuild}"
    )


def _require_run_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_RUN_NAME_CHARACTERS
        or _RUN_NAME_RE.fullmatch(value) is None
    ):
        raise ContractError(
            "workflow run name exceeds the conservative length/safe-character "
            "contract"
        )
    return value


def _parse_binding_fields(
    match: re.Match[str], label: str
) -> PipelineBinding:
    rebuild = match.group("release_rebuild")
    if re.fullmatch(r"0|[1-9][0-9]*", rebuild) is None:
        raise ContractError(f"{label} encodes a malformed rebuild counter")
    try:
        return PipelineBinding(
            bridge_source_sha=match.group("bridge_source_sha"),
            release_tag=match.group("release_tag"),
            release_rebuild=int(rebuild),
        )
    except ContractError as error:
        raise ContractError(f"{label} encodes an invalid pipeline binding: {error}") from error


def parse_candidate_run_name(
    run_name: str, correlation_id: str
) -> PipelineBinding | None:
    """Recover the binding a candidate run was dispatched with, or ``None``.

    ``None`` means the name belongs to a different correlation. A name that
    claims this correlation but cannot be parsed exactly fails closed.
    """
    require_correlation_id(correlation_id)
    if not isinstance(run_name, str):
        return None
    claims_correlation = run_name.split()[:2] == ["bridge-candidate", correlation_id]
    if (
        len(run_name) > MAX_RUN_NAME_CHARACTERS
        or _RUN_NAME_RE.fullmatch(run_name) is None
    ):
        if claims_correlation:
            raise ContractError(
                f"candidate run name claiming correlation {correlation_id!r} exceeds "
                "the length or character contract"
            )
        return None
    match = _CANDIDATE_RUN_NAME_RE.fullmatch(run_name)
    if match is not None:
        if match.group("correlation_id") != correlation_id:
            return None
        return _parse_binding_fields(match, "candidate run name")
    if claims_correlation:
        raise ContractError(
            f"candidate run name claiming correlation {correlation_id!r} is malformed"
        )
    return None


def parse_publish_run_name(
    run_name: str, correlation_id: str
) -> tuple[str, str, PipelineBinding] | None:
    require_correlation_id(correlation_id)
    if (
        not isinstance(run_name, str)
        or len(run_name) > MAX_RUN_NAME_CHARACTERS
        or _RUN_NAME_RE.fullmatch(run_name) is None
    ):
        return None
    match = _PUBLISH_RUN_NAME_RE.fullmatch(run_name)
    if match is None or match.group("correlation_id") != correlation_id:
        return None
    for label in ("candidate_run_id", "qualification_run_id"):
        if _RUN_ID_RE.fullmatch(match.group(label)) is None:
            raise ContractError(f"publish run name has a malformed {label}")
    return (
        match.group("candidate_run_id"),
        match.group("qualification_run_id"),
        _parse_binding_fields(match, "publish run name"),
    )


# --------------------------------------------------------------------------
# Native manifest scan
# --------------------------------------------------------------------------


_CHANNELS = {channel.value: channel for channel in Channel}


def require_channel(value: str) -> Channel:
    channel = _CHANNELS.get(value)
    if channel is None:
        raise ContractError(
            f"unsupported release channel {value!r}; expected one of "
            + ", ".join(sorted(_CHANNELS))
        )
    return channel


def scan_native_provenance(
    *,
    manifest_path: Path,
    native_release_tag: str,
    bridge_source_sha: str,
    channel: str,
    native_release_published_at: str,
) -> NativeProvenance:
    """Extract exact provenance from a native ``assets.json`` on either channel.

    Development scans are supported so a maintainer can inspect a ``bNNNN``
    native release, but only the stable channel is ever orchestrated; see
    :func:`require_stable_provenance`.
    """
    requested = require_channel(channel)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ContractError(f"native manifest is not a regular file: {manifest_path}")
    raw = manifest_path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(f"native manifest is not UTF-8: {error}") from error
    manifest = _strict_json_loads(text, "native manifest")
    if not isinstance(manifest, Mapping):
        raise ContractError("native manifest root must be a JSON object")
    identity = resolve_native_manifest(manifest, native_release_tag)
    provenance = NativeProvenance(
        bridge_source_sha=bridge_source_sha,
        upstream_tag=identity.upstream_tag,
        upstream_commit=identity.upstream_commit,
        native_repo=NATIVE_REPOSITORY,
        native_release_tag=identity.release_tag,
        native_commit=identity.native_commit,
        native_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        native_release_published_at=native_release_published_at,
    )
    if provenance.channel is not requested:
        raise ContractError(
            f"a {requested.value} scan resolved the {provenance.channel.value} native "
            f"release {identity.release_tag!r} ({identity.upstream_tag})"
        )
    return provenance


def select_stable_native_backlog(
    releases: Sequence[Any],
    *,
    baseline_tag: str = STABLE_AUTOMATION_BASELINE_NATIVE_TAG,
    baseline_published_at: str = STABLE_AUTOMATION_BASELINE_PUBLISHED_AT,
) -> list[str]:
    """Return every published stable native tag after the migration baseline.

    The publication timestamp defines which releases belong to automatic
    orchestration; the tag ordering independently rejects a post-baseline
    rollback. This lets a later release receive its candidate while an earlier
    release advances through qualification, without backfilling the mutable
    historical release series.
    """
    if _UTC_TIMESTAMP_RE.fullmatch(baseline_published_at) is None:
        raise ContractError("stable automation baseline timestamp is not canonical")
    baseline = parse_release_tag(baseline_tag)
    if baseline.channel is not Channel.STABLE:
        raise ContractError("stable automation baseline tag is not stable")
    baseline_order = (*baseline.version_parts, baseline.rebuild)

    selected: list[tuple[str, tuple[int, ...], str]] = []
    seen_tags: set[str] = set()
    for index, release in enumerate(releases):
        if not isinstance(release, Mapping):
            raise ContractError(f"native release listing entry {index} is not an object")
        draft = release.get("draft")
        if not isinstance(draft, bool):
            raise ContractError(f"native release listing entry {index} has no boolean draft")
        if draft:
            continue
        tag = release.get("tag_name")
        if not isinstance(tag, str) or not tag:
            raise ContractError(f"native release listing entry {index} has no tag_name")
        try:
            version = parse_release_tag(tag)
        except ContractError:
            # Development and historical/foreign tag forms are not stable
            # automatic-publication candidates.
            continue
        if version.channel is not Channel.STABLE:
            continue
        prerelease = release.get("prerelease")
        if not isinstance(prerelease, bool) or prerelease is not version.github_prerelease:
            raise ContractError(
                f"stable native release {tag!r} has inconsistent prerelease state"
            )
        published_at = release.get("published_at")
        if (
            not isinstance(published_at, str)
            or _UTC_TIMESTAMP_RE.fullmatch(published_at) is None
        ):
            raise ContractError(
                f"stable native release {tag!r} has no canonical published_at"
            )
        if published_at <= baseline_published_at:
            continue
        order = (*version.version_parts, version.rebuild)
        if order <= baseline_order:
            raise ContractError(
                f"post-baseline stable native release {tag!r} does not advance "
                f"{baseline_tag!r}"
            )
        if tag in seen_tags:
            raise ContractError(f"stable native release {tag!r} is duplicated")
        seen_tags.add(tag)
        selected.append((published_at, order, tag))

    selected.sort()
    return [tag for _, _, tag in selected]


# --------------------------------------------------------------------------
# Workflow run inventory
# --------------------------------------------------------------------------


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


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _pages(payload: Any, label: str) -> list[Any]:
    """Normalize ``gh api --paginate --slurp`` output to a page list."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        return [payload]
    raise ContractError(f"{label} response must be a JSON array or object")


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


# --------------------------------------------------------------------------
# Collision-free, independent bridge-asset release selection
# --------------------------------------------------------------------------


def select_next_release_target(
    release_tags: Sequence[str], *, upstream_tag: str, taken: Sequence[str] | set[str] = ()
) -> ReleaseTarget:
    """Pick the next free bridge-asset tag, independently of the upstream tag.

    ``release_tags`` are the assets repository's existing releases and set the
    version floor. ``taken`` are tags already claimed by an unfinished pipeline;
    they never advance the version but must not be collided with, so a claimed
    tag falls through to the next free rebuild of the same version.
    """
    parse_upstream_tag(upstream_tag)
    published = {tag for tag in release_tags if isinstance(tag, str)}
    claimed = published | {tag for tag in taken if isinstance(tag, str)}
    versions = []
    for tag in published:
        try:
            version = parse_release_tag(tag, allow_legacy=True)
        except ContractError:
            continue
        if version.channel is Channel.STABLE:
            versions.append(version)
    if versions:
        highest = max(versions, key=lambda value: (*value.version_parts, value.rebuild))
        major, minor, patch = highest.version_parts
        base_tag = f"v{major}.{minor}.{patch + 1}"
    else:
        base_tag = INITIAL_STABLE_RELEASE_TAG

    rebuild = 0
    tag = base_tag
    while tag in claimed:
        rebuild += 1
        tag = f"{base_tag}-{rebuild}"
    validate_release_identity(tag, rebuild, upstream_tag)
    return ReleaseTarget(tag, rebuild)


# --------------------------------------------------------------------------
# Live transport
# --------------------------------------------------------------------------


class Gateway(Protocol):
    def api_json(self, path: str, *, paginate: bool = ..., privileged: bool = ...) -> Any: ...

    def download_bytes(self, path: str, *, accept: str, privileged: bool = ...) -> bytes: ...

    def dispatch_identity(self) -> str | None: ...

    def release_attestation(self, *, repository: str, release_tag: str) -> Any: ...

    def dispatch_workflow(
        self, *, workflow_file: str, ref: str, inputs: Mapping[str, str]
    ) -> None: ...

    def sleep(self, seconds: float) -> None: ...

    def utc_now(self) -> str: ...


class GhGateway:
    """``gh``-backed transport. Read traffic and dispatch use distinct tokens."""

    def __init__(self, *, read_token: str, dispatch_token: str | None) -> None:
        self._read_token = read_token
        self._dispatch_token = dispatch_token or None

    def _run(self, args: Sequence[str], *, privileged: bool, binary: bool) -> bytes:
        token = self._dispatch_token if privileged else self._read_token
        if not token:
            raise ContractError(
                "no credential is available for this GitHub request; refusing to continue"
            )
        env = dict(os.environ)
        env["GH_TOKEN"] = token
        env.pop("GITHUB_TOKEN", None)
        completed = subprocess.run(
            list(args), env=env, capture_output=True, check=False
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", "replace").strip()
            raise ContractError(f"gh {' '.join(args[1:])} failed: {stderr}")
        return completed.stdout if binary else completed.stdout

    def api_json(self, path: str, *, paginate: bool = False, privileged: bool = False) -> Any:
        args = ["gh", "api"]
        if paginate:
            args += ["--paginate", "--slurp"]
        args.append(path)
        stdout = self._run(args, privileged=privileged, binary=False)
        return _strict_json_loads(stdout.decode("utf-8", "strict"), f"gh api {path}")

    def download_bytes(self, path: str, *, accept: str, privileged: bool = False) -> bytes:
        return self._run(
            ["gh", "api", "-H", f"Accept: {accept}", path],
            privileged=privileged,
            binary=True,
        )

    def dispatch_identity(self) -> str | None:
        if not self._dispatch_token:
            return None
        try:
            payload = self.api_json("user", privileged=True)
        except ContractError:
            return None
        login = payload.get("login") if isinstance(payload, Mapping) else None
        return login if isinstance(login, str) and login else None

    def release_attestation(self, *, repository: str, release_tag: str) -> Any:
        stdout = self._run(
            [
                "gh",
                "release",
                "verify",
                release_tag,
                "--repo",
                repository,
                "--format",
                "json",
            ],
            privileged=False,
            binary=False,
        )
        return _strict_json_loads(
            stdout.decode("utf-8", "strict"),
            f"gh release verify {repository}@{release_tag}",
        )

    def dispatch_workflow(
        self, *, workflow_file: str, ref: str, inputs: Mapping[str, str]
    ) -> None:
        if not self._dispatch_token:
            raise ContractError("workflow dispatch requires an orchestrator credential")
        env = dict(os.environ)
        env["GH_TOKEN"] = self._dispatch_token
        env.pop("GITHUB_TOKEN", None)
        completed = subprocess.run(
            [
                "gh",
                "workflow",
                "run",
                workflow_file,
                "--repo",
                BRIDGE_REPOSITORY,
                "--ref",
                ref,
                "--json",
            ],
            input=json.dumps(dict(inputs), sort_keys=True).encode("utf-8"),
            env=env,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", "replace").strip()
            raise ContractError(f"dispatching {workflow_file} failed: {stderr}")

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def utc_now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Live proofs
# --------------------------------------------------------------------------


def require_default_branch(gateway: Gateway) -> str:
    payload = gateway.api_json(f"repos/{BRIDGE_REPOSITORY}")
    if not isinstance(payload, Mapping):
        raise ContractError("repository response must be a JSON object")
    return _require_str(payload.get("default_branch"), "default_branch")


def require_immutable_release_governance(gateway: Gateway) -> dict[str, Any]:
    """Prove immutable-release governance live, never by constant assertion."""
    payload = gateway.api_json(
        f"repos/{ASSETS_REPOSITORY}/immutable-releases", privileged=True
    )
    return validate_immutable_release_governance(payload, ASSETS_REPOSITORY)


def require_publication_environment(gateway: Gateway) -> None:
    """Prove the existing environment policy before asserting publication approval."""
    environment = gateway.api_json(
        f"repos/{BRIDGE_REPOSITORY}/environments/bridge-assets-publication"
    )
    branch_policies = gateway.api_json(
        f"repos/{BRIDGE_REPOSITORY}/environments/bridge-assets-publication/"
        "deployment-branch-policies"
    )
    if not isinstance(environment, Mapping) or not isinstance(
        branch_policies, Mapping
    ):
        raise ContractError("publication environment responses must be JSON objects")
    validate_publication_environment(environment, branch_policies)


def resolve_repository_tag_commit(
    gateway: Gateway, *, repository: str, release_tag: str
) -> str:
    """Resolve a lightweight or annotated tag to its immutable commit."""
    require_repository(repository, "repository")
    parse_release_tag(release_tag)
    encoded_tag = quote(release_tag, safe="")
    payload = gateway.api_json(f"repos/{repository}/git/ref/tags/{encoded_tag}")
    if not isinstance(payload, Mapping) or payload.get("ref") != f"refs/tags/{release_tag}":
        raise ContractError("assets tag reference identity is missing or incorrect")
    object_payload = payload.get("object")
    seen: set[str] = set()
    for _ in range(8):
        if not isinstance(object_payload, Mapping):
            raise ContractError("assets tag reference object is malformed")
        object_type = object_payload.get("type")
        sha = object_payload.get("sha")
        if not isinstance(sha, str) or _COMMIT_RE.fullmatch(sha) is None:
            raise ContractError("assets tag reference object has no full commit SHA")
        if sha in seen:
            raise ContractError("assets annotated tag chain contains a cycle")
        seen.add(sha)
        if object_type == "commit":
            return sha
        if object_type != "tag":
            raise ContractError(
                f"assets tag reference points to unsupported object type {object_type!r}"
            )
        annotated = gateway.api_json(f"repos/{repository}/git/tags/{sha}")
        if not isinstance(annotated, Mapping) or annotated.get("sha") != sha:
            raise ContractError("assets annotated tag identity is malformed")
        object_payload = annotated.get("object")
    raise ContractError("assets annotated tag chain exceeds the validation bound")


def fetch_asset_releases(gateway: Gateway) -> list[Mapping[str, Any]]:
    payload = gateway.api_json(
        f"repos/{ASSETS_REPOSITORY}/releases?per_page=100", paginate=True
    )
    pages = _pages(payload, "asset releases")
    releases: list[Mapping[str, Any]] = []
    seen_tags: set[str] = set()
    for page in pages:
        page_releases = page if isinstance(page, list) else [page]
        for release in page_releases:
            if not isinstance(release, Mapping):
                raise ContractError("asset release record must be a JSON object")
            tag = _require_str(release.get("tag_name"), "asset release tag_name")
            if tag in seen_tags:
                raise ContractError(f"asset repository lists duplicate release {tag!r}")
            seen_tags.add(tag)
            releases.append(release)
    return releases


def fetch_asset_tag_names(gateway: Gateway) -> set[str]:
    """Return every existing assets tag ref so output selection cannot collide."""
    payload = gateway.api_json(
        f"repos/{ASSETS_REPOSITORY}/git/matching-refs/tags?per_page=100",
        paginate=True,
    )
    pages = _pages(payload, "asset tag refs")
    tags: set[str] = set()
    for page in pages:
        refs = page if isinstance(page, list) else [page]
        for record in refs:
            if not isinstance(record, Mapping):
                raise ContractError("asset tag ref record must be a JSON object")
            ref = record.get("ref")
            if not isinstance(ref, str) or not ref.startswith("refs/tags/"):
                raise ContractError("asset tag ref has an invalid ref name")
            tag = ref.removeprefix("refs/tags/")
            if not tag or tag in tags:
                raise ContractError(f"asset repository lists duplicate tag ref {tag!r}")
            object_payload = record.get("object")
            if not isinstance(object_payload, Mapping):
                raise ContractError(f"asset tag ref {tag!r} has no object")
            if object_payload.get("type") not in ("commit", "tag"):
                raise ContractError(f"asset tag ref {tag!r} has invalid object type")
            sha = object_payload.get("sha")
            if not isinstance(sha, str) or _COMMIT_RE.fullmatch(sha) is None:
                raise ContractError(f"asset tag ref {tag!r} has invalid object SHA")
            tags.add(tag)
    return tags


def _correlation_marker(correlation_id: str) -> str:
    return f"Orchestrator correlation: `{correlation_id}`"


def find_correlated_release(
    releases: Sequence[Mapping[str, Any]],
    correlation_id: str,
    provenance: NativeProvenance,
) -> Mapping[str, Any] | None:
    marker = _correlation_marker(correlation_id)
    # The manifest digest is independently selected from the native release and
    # is also written to deterministic release notes. If only the correlation
    # line is damaged, still classify the release as relevant and let the full
    # immutable readback reject it instead of dispatching duplicate provenance.
    native_manifest_marker = (
        "Native manifest SHA-256: "
        f"`{provenance.native_manifest_sha256}`"
    )
    native_release_marker = (
        f"Native: `{provenance.native_repo}@{provenance.native_release_tag}`"
    )
    matches = [
        release
        for release in releases
        if isinstance(release.get("body"), str)
        and (
            marker in release["body"]
            or (
                native_manifest_marker in release["body"]
                and native_release_marker in release["body"]
            )
        )
    ]
    if len(matches) > 1:
        raise ContractError(
            f"{len(matches)} asset releases claim correlation {correlation_id!r}: "
            + ", ".join(str(release.get("tag_name")) for release in matches)
        )
    return matches[0] if matches else None


def workflow_history_since(
    releases: Sequence[Mapping[str, Any]], provenance: NativeProvenance
) -> str:
    """Bound run recovery to state that can still claim the next output tag.

    Candidate claims made before the most recently published stable assets tag
    cannot collide with a later monotonic output version. Taking the earlier of
    that publication and the native release still includes a prior unfinished
    pipeline when a newer native release appears, without coupling recovery to
    the repository's unbounded lifetime run count.
    """
    stable_publications: list[str] = []
    for release in releases:
        tag = release.get("tag_name")
        if not isinstance(tag, str):
            continue
        try:
            parsed = parse_release_tag(tag, allow_legacy=True)
        except ContractError:
            continue
        if parsed.channel is not Channel.STABLE or release.get("draft") is True:
            continue
        if release.get("draft") is not False:
            raise ContractError(f"stable asset release {tag!r} has invalid draft state")
        if release.get("prerelease") is not parsed.github_prerelease:
            raise ContractError(
                f"stable asset release {tag!r} has invalid prerelease state"
            )
        published_at = release.get("published_at")
        if (
            not isinstance(published_at, str)
            or _UTC_TIMESTAMP_RE.fullmatch(published_at) is None
        ):
            raise ContractError(
                f"stable asset release {tag!r} has no canonical published_at"
            )
        stable_publications.append(published_at)
    if not stable_publications:
        return provenance.native_release_published_at
    return min(provenance.native_release_published_at, max(stable_publications))


def _release_assets(release: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = release.get("assets")
    if not isinstance(raw, list):
        raise ContractError("GitHub Release is missing its asset inventory")
    assets: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ContractError("GitHub Release contains an invalid asset record")
        name = _require_str(item.get("name"), "GitHub Release asset name")
        if name in assets:
            raise ContractError(f"GitHub Release has duplicate asset {name!r}")
        if item.get("state") != "uploaded":
            raise ContractError(f"GitHub Release asset {name!r} is not uploaded")
        digest = item.get("digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ContractError(f"GitHub Release asset {name!r} has no SHA-256 digest")
        require_sha256(digest.removeprefix("sha256:"), f"digest for {name}")
        size = item.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ContractError(f"GitHub Release asset {name!r} has an invalid size")
        _require_positive_int(item.get("id"), f"GitHub Release asset {name!r} id")
        assets[name] = item
    if set(assets) != set(PUBLICATION_FILES):
        unexpected = sorted(set(assets) - set(PUBLICATION_FILES))
        missing = sorted(set(PUBLICATION_FILES) - set(assets))
        raise ContractError(
            "GitHub Release asset inventory is not the exact publication set "
            f"(unexpected: {unexpected}, missing: {missing})"
        )
    return assets


def _candidate_fingerprint_marker(fingerprint: str) -> str:
    return f"Candidate fingerprint: `{fingerprint}`"


def verify_published_release(
    gateway: Gateway,
    *,
    release: Mapping[str, Any],
    provenance: NativeProvenance,
    correlation_id: str,
    workspace: Path,
) -> PublishedRelease:
    """Prove an existing release really is this provenance, published immutably.

    The release listing is discovery only. The exact tag commit, independent
    release reads by tag and ID, downloaded artifact bytes, and GitHub's signed
    release attestation are all re-proven with the same validators used after
    publication before this path may return a noop.
    """
    tag = _require_str(release.get("tag_name"), "release tag_name")
    parse_release_tag(tag)
    release_id = _require_positive_int(release.get("id"), "release id")
    tag_commit = resolve_repository_tag_commit(
        gateway, repository=ASSETS_REPOSITORY, release_tag=tag
    )
    encoded_tag = quote(tag, safe="")
    release_by_tag = gateway.api_json(
        f"repos/{ASSETS_REPOSITORY}/releases/tags/{encoded_tag}"
    )
    release_by_id = gateway.api_json(
        f"repos/{ASSETS_REPOSITORY}/releases/{release_id}"
    )
    if not isinstance(release_by_tag, Mapping) or not isinstance(
        release_by_id, Mapping
    ):
        raise ContractError("release readbacks must be JSON objects")
    resolved_id = validate_release_immutability(
        release_by_tag,
        release_tag=tag,
        tag_commit=tag_commit,
        release_id=release_id,
    )
    if (
        validate_release_immutability(
            release_by_id,
            release_tag=tag,
            tag_commit=tag_commit,
            release_id=resolved_id,
        )
        != resolved_id
    ):
        raise ContractError("release readbacks by tag and ID disagree on release id")
    for label, current in (("tag", release_by_tag), ("id", release_by_id)):
        if current.get("name") != tag:
            raise ContractError(f"release readback by {label} is not named {tag!r}")
        body = current.get("body")
        if not isinstance(body, str) or _correlation_marker(correlation_id) not in body:
            raise ContractError(
                f"release readback by {label} does not record correlation "
                f"{correlation_id!r}"
            )
    if release_by_tag.get("published_at") != release_by_id.get("published_at"):
        raise ContractError("release readbacks disagree on published_at")
    if release_by_tag.get("body") != release_by_id.get("body"):
        raise ContractError("release readbacks disagree on release body")
    published_at = _require_str(
        release_by_tag.get("published_at"), "release published_at"
    )
    body = _require_str(release_by_tag.get("body"), "release body")

    assets = _release_assets(release_by_tag)
    assets_by_id = _release_assets(release_by_id)
    for name in PUBLICATION_FILES:
        fields = ("id", "name", "state", "size", "digest")
        if any(assets[name].get(field) != assets_by_id[name].get(field) for field in fields):
            raise ContractError(
                f"release readbacks disagree on asset {name!r} identity"
            )
    directory = workspace / f"published-{release_id}"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    expected_asset_digests: dict[str, str] = {}
    for name, asset in assets.items():
        data = gateway.download_bytes(
            f"repos/{ASSETS_REPOSITORY}/releases/assets/{asset['id']}",
            accept="application/octet-stream",
        )
        actual = hashlib.sha256(data).hexdigest()
        if f"sha256:{actual}" != asset["digest"]:
            raise ContractError(f"release asset {name!r} does not match its GitHub digest")
        if len(data) != asset["size"]:
            raise ContractError(f"release asset {name!r} does not match its GitHub size")
        expected_asset_digests[name] = actual
        (directory / name).write_bytes(data)

    manifest, fingerprint = rq.load_candidate(directory)
    if _candidate_fingerprint_marker(fingerprint) not in body:
        raise ContractError(
            f"release {tag!r} does not record the fingerprint of the bytes it "
            "actually published"
        )
    binding = validate_candidate_manifest(
        manifest,
        provenance=provenance,
        correlation_id=correlation_id,
        expected_release_tag=tag,
    )
    for name in ARTIFACTS:
        recorded = manifest["artifacts"][name]
        if assets[name]["digest"] != f"sha256:{recorded['sha256']}":
            raise ContractError(f"release asset {name!r} digest is not the manifest digest")
        if assets[name]["size"] != recorded["size_bytes"]:
            raise ContractError(f"release asset {name!r} size is not the manifest size")
    validate_release_attestation(
        gateway.release_attestation(repository=ASSETS_REPOSITORY, release_tag=tag),
        assets_repo=ASSETS_REPOSITORY,
        release_tag=tag,
        tag_commit=tag_commit,
        release_id=resolved_id,
        expected_assets=expected_asset_digests,
    )
    return PublishedRelease(
        release_id=resolved_id,
        release_target=binding.release_target,
        binding=binding,
        published_at=published_at,
    )


def validate_candidate_manifest(
    manifest: Mapping[str, Any],
    *,
    provenance: NativeProvenance,
    correlation_id: str,
    expected_release_tag: str | None = None,
    expected_bridge_source_sha: str | None = None,
    expected_run_id: str | None = None,
) -> PipelineBinding:
    """Fail closed unless a manifest binds this exact provenance and correlation."""
    expected = {
        "assets_repository": ASSETS_REPOSITORY,
        "bridge_repository": BRIDGE_REPOSITORY,
        "bridge_commit": expected_bridge_source_sha or manifest.get("bridge_commit"),
        "upstream_repository": "ggml-org/llama.cpp",
        "upstream_tag": provenance.upstream_tag,
        "upstream_commit": provenance.upstream_commit,
        "native_repository": NATIVE_REPOSITORY,
        "native_release_tag": provenance.native_release_tag,
        "native_manifest_sha256": provenance.native_manifest_sha256,
        "native_commit": provenance.native_commit,
        "orchestrator_correlation_id": correlation_id,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ContractError(
                f"candidate manifest {key} is {manifest.get(key)!r}, expected {value!r}"
            )
    bridge_commit = _require_str(manifest.get("bridge_commit"), "manifest bridge_commit")
    if _COMMIT_RE.fullmatch(bridge_commit) is None:
        raise ContractError("manifest bridge_commit must be a 40-hex commit SHA")
    if expected_run_id is not None and manifest.get("github_run_id") != expected_run_id:
        raise ContractError(
            f"candidate manifest github_run_id is {manifest.get('github_run_id')!r}, "
            f"expected {expected_run_id!r}"
        )
    release_tag = _require_str(manifest.get("release_tag"), "manifest release_tag")
    rebuild = manifest.get("release_rebuild")
    if not isinstance(rebuild, int) or isinstance(rebuild, bool) or rebuild < 0:
        raise ContractError("manifest release_rebuild must be a non-negative integer")
    if expected_release_tag is not None and release_tag != expected_release_tag:
        raise ContractError(
            f"candidate manifest release_tag is {release_tag!r}, expected "
            f"{expected_release_tag!r}"
        )
    validate_release_identity(release_tag, rebuild, provenance.upstream_tag)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(ARTIFACTS):
        raise ContractError("candidate manifest does not record exactly the artifact set")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise ContractError(f"candidate manifest artifact {name!r} is malformed")
        require_sha256(_require_str(record.get("sha256"), f"{name} sha256"), f"{name} sha256")
        size = record.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ContractError(f"candidate manifest artifact {name!r} has an invalid size")
    return PipelineBinding(
        bridge_source_sha=bridge_commit,
        release_tag=release_tag,
        release_rebuild=rebuild,
    )


@dataclass(frozen=True)
class CandidateEvidence:
    run_id: str
    artifact_id: int
    fingerprint: str
    binding: PipelineBinding
    manifest: Mapping[str, Any]
    directory: Path


def _require_reachable_from_main(
    gateway: Gateway, *, commit: str, default_branch: str, label: str
) -> None:
    payload = gateway.api_json(
        f"repos/{BRIDGE_REPOSITORY}/compare/{commit}...{default_branch}"
    )
    if not isinstance(payload, Mapping):
        raise ContractError("compare response must be a JSON object")
    status = payload.get("status")
    if status not in ("ahead", "identical"):
        raise ContractError(
            f"{label} commit {commit} is not reachable from {default_branch} "
            f"(compare status {status!r})"
        )


def _download_run_artifact(
    gateway: Gateway,
    *,
    run_id: str,
    artifact_name: str,
    artifact_type: str,
    workspace: Path,
) -> tuple[int, Path]:
    inventory = gateway.api_json(
        f"repos/{BRIDGE_REPOSITORY}/actions/runs/{run_id}/artifacts?per_page=100"
    )
    artifact_id = rq.validate_artifact_inventory(
        inventory, expected_run_id=run_id, expected_name=artifact_name
    )
    archive = workspace / f"{artifact_type}-{run_id}.zip"
    archive.write_bytes(
        gateway.download_bytes(
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{artifact_id}/zip",
            accept="application/vnd.github+json",
        )
    )
    destination = workspace / f"{artifact_type}-{run_id}"
    destination.mkdir(parents=True, exist_ok=True)
    rq._extract_flat_artifact_archive(archive, destination, artifact_type=artifact_type)
    return artifact_id, destination


def verify_candidate_run(
    gateway: Gateway,
    *,
    run_id: str,
    provenance: NativeProvenance,
    correlation_id: str,
    binding: PipelineBinding,
    default_branch: str,
    workspace: Path,
) -> CandidateEvidence:
    run = gateway.api_json(f"repos/{BRIDGE_REPOSITORY}/actions/runs/{run_id}")
    head_sha = rq.validate_workflow_run(
        run,
        expected_run_id=run_id,
        expected_workflow_path=CANDIDATE_WORKFLOW_PATH,
        expected_head_branch=default_branch,
        expected_run_attempt=1,
    )
    _require_reachable_from_main(
        gateway, commit=head_sha, default_branch=default_branch, label="candidate run head"
    )
    artifact_id, directory = _download_run_artifact(
        gateway,
        run_id=run_id,
        artifact_name=rq.CANDIDATE_ARTIFACT_NAME,
        artifact_type="candidate",
        workspace=workspace,
    )
    manifest, fingerprint = rq.load_candidate(directory)
    manifest_binding = validate_candidate_manifest(
        manifest,
        provenance=provenance,
        correlation_id=correlation_id,
        expected_release_tag=binding.release_tag,
        expected_bridge_source_sha=binding.bridge_source_sha,
        expected_run_id=run_id,
    )
    if manifest_binding != binding:
        raise ContractError(
            "candidate manifest contradicts the binding its run name advertises"
        )
    _require_reachable_from_main(
        gateway,
        commit=binding.bridge_source_sha,
        default_branch=default_branch,
        label="candidate bridge source",
    )
    return CandidateEvidence(
        run_id=run_id,
        artifact_id=artifact_id,
        fingerprint=fingerprint,
        binding=binding,
        manifest=manifest,
        directory=directory,
    )


def verify_qualification_run(
    gateway: Gateway,
    *,
    run_id: str,
    candidate: CandidateEvidence,
    provenance: NativeProvenance,
    correlation_id: str,
    default_branch: str,
    workspace: Path,
) -> dict[str, Any]:
    run = gateway.api_json(f"repos/{BRIDGE_REPOSITORY}/actions/runs/{run_id}")
    qualification_source_sha = rq.validate_workflow_run(
        run,
        expected_run_id=run_id,
        expected_workflow_path=QUALIFICATION_WORKFLOW_PATH,
        expected_head_branch=default_branch,
        expected_run_attempt=1,
    )
    _, directory = _download_run_artifact(
        gateway,
        run_id=run_id,
        artifact_name=rq.ATTESTATION_ARTIFACT_NAME,
        artifact_type="attestation",
        workspace=workspace,
    )
    attestation_path = directory / "qualification-attestation.json"
    if not attestation_path.is_file():
        raise ContractError("attestation artifact does not contain the canonical payload")
    attestation = rq.load_attestation_file(attestation_path)
    return rq.verify_attestation(
        attestation=attestation,
        candidate_dir=candidate.directory,
        candidate_fingerprint=candidate.fingerprint,
        candidate_run_id=candidate.run_id,
        candidate_artifact_id=candidate.artifact_id,
        candidate_run_attempt=1,
        qualification_run_id=run_id,
        qualification_run_attempt=1,
        qualification_source_sha=qualification_source_sha,
        bridge_source_sha=candidate.binding.bridge_source_sha,
        upstream_tag=provenance.upstream_tag,
        upstream_commit=provenance.upstream_commit,
        native_release_tag=provenance.native_release_tag,
        native_manifest_sha256=provenance.native_manifest_sha256,
        native_commit=provenance.native_commit,
        emscripten_version=_require_str(
            candidate.manifest.get("emscripten_version"),
            "candidate manifest emscripten_version",
        ),
        release_tag=candidate.binding.release_tag,
        release_rebuild=candidate.binding.release_rebuild,
        orchestrator_correlation_id=correlation_id,
    )


# --------------------------------------------------------------------------
# Pure state machine
# --------------------------------------------------------------------------


def require_exact_dispatch_inputs(
    workflow_file: str, inputs: Mapping[str, Any]
) -> dict[str, str]:
    """Refuse to dispatch anything but the workflow's exact declared input set."""
    expected = WORKFLOW_DISPATCH_INPUTS.get(workflow_file)
    if expected is None:
        raise ContractError(
            f"{workflow_file} is not a workflow this orchestrator may dispatch"
        )
    if set(inputs) != set(expected):
        missing = sorted(set(expected) - set(inputs))
        unexpected = sorted(set(inputs) - set(expected))
        raise ContractError(
            f"dispatch inputs for {workflow_file} are not the exact declared set "
            f"(missing: {missing}, unexpected: {unexpected})"
        )
    ordered: dict[str, str] = {}
    for name in expected:
        value = inputs[name]
        if not isinstance(value, str) or not value:
            raise ContractError(
                f"dispatch input {name!r} for {workflow_file} must be a non-empty "
                f"string, got {value!r}"
            )
        ordered[name] = value
    return ordered


def _dispatch_inputs_for_candidate(
    provenance: NativeProvenance, correlation_id: str, binding: PipelineBinding
) -> dict[str, str]:
    """Build every candidate input the planner can know without live proof.

    ``assets_immutable_releases_enabled`` is deliberately absent: it is an
    assertion about the assets repository, so it is filled in from the live
    governance read taken immediately before dispatch, never from a constant.
    """
    return {
        "orchestrator_correlation_id": correlation_id,
        "bridge_source_sha": binding.bridge_source_sha,
        "upstream_tag": provenance.upstream_tag,
        "upstream_commit": provenance.upstream_commit,
        "native_release_tag": provenance.native_release_tag,
        "native_manifest_sha256": provenance.native_manifest_sha256,
        "release_tag": binding.release_tag,
        "release_rebuild": str(binding.release_rebuild),
    }


def _dispatch_inputs_for_qualification(
    correlation_id: str, candidate_run_id: str
) -> dict[str, str]:
    return {
        "orchestrator_correlation_id": correlation_id,
        "candidate_run_id": candidate_run_id,
    }


def _dispatch_inputs_for_publish(
    provenance: NativeProvenance,
    correlation_id: str,
    binding: PipelineBinding,
    candidate_run_id: str,
    qualification_run_id: str,
) -> dict[str, str]:
    return {
        "orchestrator_correlation_id": correlation_id,
        "bridge_source_sha": binding.bridge_source_sha,
        "upstream_tag": provenance.upstream_tag,
        "upstream_commit": provenance.upstream_commit,
        "native_release_tag": provenance.native_release_tag,
        "native_manifest_sha256": provenance.native_manifest_sha256,
        "release_tag": binding.release_tag,
        "release_rebuild": str(binding.release_rebuild),
        "assets_repo": ASSETS_REPOSITORY,
        "candidate_run_id": candidate_run_id,
        "qualification_run_id": qualification_run_id,
    }


def plan_pipeline(
    *,
    provenance: NativeProvenance,
    correlation_id: str,
    observation: PipelineObservation,
) -> OrchestrationPlan:
    """Decide the single next transition from already-proven evidence."""
    require_correlation_id(correlation_id)

    if observation.published is not None:
        published = observation.published
        return OrchestrationPlan(
            action=OrchestrationAction.NOOP,
            reason=(
                f"this provenance is already published as the immutable release "
                f"{published.release_target.release_tag} (id {published.release_id}, "
                f"published {published.published_at})"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=published.release_target,
        )

    if observation.candidate_in_flight_run_id is not None:
        return OrchestrationPlan(
            action=OrchestrationAction.IN_FLIGHT,
            reason=(
                f"candidate run {observation.candidate_in_flight_run_id} is still "
                "running; refusing to dispatch a duplicate"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=observation.binding.release_target
            if observation.binding
            else None,
            in_flight_workflow=CANDIDATE_WORKFLOW_FILE,
            in_flight_run_id=observation.candidate_in_flight_run_id,
        )

    if observation.candidate_run_id is None:
        binding = observation.fresh_binding
        if binding is None:
            raise ContractError("no pipeline binding is available for candidate dispatch")
        return OrchestrationPlan(
            action=OrchestrationAction.DISPATCH_CANDIDATE,
            reason=(
                "new stable native provenance requires exactly one candidate "
                f"build for {binding.release_tag}"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=binding.release_target,
            dispatch_workflow=CANDIDATE_WORKFLOW_FILE,
            dispatch_run_name=candidate_run_name(correlation_id, binding),
            dispatch_inputs=_dispatch_inputs_for_candidate(
                provenance, correlation_id, binding
            ),
        )

    binding = observation.binding
    if binding is None:
        raise ContractError(
            "a successful candidate run must carry its proven pipeline binding"
        )

    if observation.qualification_in_flight_run_id is not None:
        return OrchestrationPlan(
            action=OrchestrationAction.IN_FLIGHT,
            reason=(
                f"qualification run {observation.qualification_in_flight_run_id} "
                "is still running"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=binding.release_target,
            candidate_run_id=observation.candidate_run_id,
            in_flight_workflow=QUALIFICATION_WORKFLOW_FILE,
            in_flight_run_id=observation.qualification_in_flight_run_id,
        )

    if observation.qualification_run_id is None:
        return OrchestrationPlan(
            action=OrchestrationAction.DISPATCH_QUALIFICATION,
            reason=(
                f"candidate run {observation.candidate_run_id} is built and proven; "
                "dispatching exactly one hosted qualification run for the heavy "
                "real-model gates"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=binding.release_target,
            candidate_run_id=observation.candidate_run_id,
            dispatch_workflow=QUALIFICATION_WORKFLOW_FILE,
            dispatch_run_name=qualification_run_name(
                correlation_id, observation.candidate_run_id
            ),
            dispatch_inputs=_dispatch_inputs_for_qualification(
                correlation_id, observation.candidate_run_id
            ),
        )

    if observation.publish_in_flight_run_id is not None:
        return OrchestrationPlan(
            action=OrchestrationAction.IN_FLIGHT,
            reason=(
                f"publication run {observation.publish_in_flight_run_id} is still running"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=binding.release_target,
            candidate_run_id=observation.candidate_run_id,
            qualification_run_id=observation.qualification_run_id,
            in_flight_workflow=PUBLISH_WORKFLOW_FILE,
            in_flight_run_id=observation.publish_in_flight_run_id,
        )

    if observation.publish_succeeded_run_id is not None:
        raise ContractError(
            f"publication run {observation.publish_succeeded_run_id} succeeded but no "
            f"immutable release for correlation {correlation_id!r} could be verified"
        )

    return OrchestrationPlan(
        action=OrchestrationAction.DISPATCH_PUBLISH,
        reason=(
            f"{'retrying publication of' if observation.publish_retry else 'publishing'}"
            f" {binding.release_tag} from the proven candidate "
            f"{observation.candidate_run_id} and qualification "
            f"{observation.qualification_run_id}"
        ),
        provenance=provenance,
        correlation_id=correlation_id,
        release_target=binding.release_target,
        candidate_run_id=observation.candidate_run_id,
        qualification_run_id=observation.qualification_run_id,
        dispatch_workflow=PUBLISH_WORKFLOW_FILE,
        dispatch_run_name=publish_run_name(
            correlation_id,
            observation.candidate_run_id,
            observation.qualification_run_id,
            binding,
        ),
        dispatch_inputs=_dispatch_inputs_for_publish(
            provenance,
            correlation_id,
            binding,
            observation.candidate_run_id,
            observation.qualification_run_id,
        ),
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


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


def claimed_release_tags(runs: Sequence[RunRecord]) -> set[str]:
    """Output tags any candidate run has already claimed, across correlations."""
    claimed: set[str] = set()
    for record in runs:
        match = _CANDIDATE_RUN_NAME_RE.fullmatch(record.run_name)
        if match is not None:
            claimed.add(match.group("release_tag"))
    return claimed


def advance_pipeline(
    gateway: Gateway,
    *,
    provenance: NativeProvenance,
    workspace: Path,
    dry_run: bool = False,
    reserved_release_tags: Sequence[str] | set[str] = (),
    publication_allowed: bool = True,
    publication_barrier_native_tag: str | None = None,
) -> OrchestrationPlan:
    require_stable_provenance(provenance)
    if not publication_allowed:
        if not isinstance(publication_barrier_native_tag, str):
            raise ContractError(
                "a disabled publication transition requires the earlier native tag"
            )
        barrier = parse_release_tag(publication_barrier_native_tag)
        if barrier.channel is not Channel.STABLE:
            raise ContractError("publication barrier native tag must be stable")
    correlation_id = compute_correlation_id(provenance)
    default_branch = require_default_branch(gateway)
    # Governance is a prerequisite for every state classification, including a
    # noop: a disabled or unreadable policy is never reported as healthy stable
    # automation merely because an older release happens to exist.
    require_immutable_release_governance(gateway)

    releases = fetch_asset_releases(gateway)
    correlated = find_correlated_release(releases, correlation_id, provenance)
    published = None
    if correlated is not None:
        published = verify_published_release(
            gateway,
            release=correlated,
            provenance=provenance,
            correlation_id=correlation_id,
            workspace=workspace,
        )
        if not publication_allowed:
            return OrchestrationPlan(
                action=OrchestrationAction.BLOCKED,
                reason=(
                    f"{provenance.native_release_tag} is already published while "
                    f"earlier native release {publication_barrier_native_tag} is not; "
                    "the ordered publication history is inconsistent"
                ),
                provenance=provenance,
                correlation_id=correlation_id,
                release_target=published.release_target,
            )
        return plan_pipeline(
            provenance=provenance,
            correlation_id=correlation_id,
            observation=PipelineObservation(published=published),
        )

    run_history_since = workflow_history_since(releases, provenance)
    asset_tag_names = fetch_asset_tag_names(gateway)
    candidate_runs = _fetch_runs(
        gateway,
        workflow_file=CANDIDATE_WORKFLOW_FILE,
        workflow_path=CANDIDATE_WORKFLOW_PATH,
        default_branch=default_branch,
        created_since=run_history_since,
    )
    candidate_selection = select_pipeline_runs(
        candidate_runs,
        label="candidate",
        matcher=_candidate_matcher(correlation_id),
        default_branch=default_branch,
    )
    persisted = _resolve_candidate_binding(candidate_selection, correlation_id)
    fresh_binding = None
    if persisted is None:
        target = select_next_release_target(
            [str(release.get("tag_name")) for release in releases],
            upstream_tag=provenance.upstream_tag,
            taken=(
                asset_tag_names
                | claimed_release_tags(candidate_runs)
                | set(reserved_release_tags)
            ),
        )
        fresh_binding = PipelineBinding(
            bridge_source_sha=provenance.bridge_source_sha,
            release_tag=target.release_tag,
            release_rebuild=target.release_rebuild,
        )

    if candidate_selection.in_flight_run_id is not None:
        return plan_pipeline(
            provenance=provenance,
            correlation_id=correlation_id,
            observation=PipelineObservation(
                binding=persisted,
                fresh_binding=fresh_binding,
                candidate_in_flight_run_id=candidate_selection.in_flight_run_id,
            ),
        )

    if (
        candidate_selection.succeeded_run_id is None
        and candidate_selection.unsuccessful
    ):
        failed_ids = ", ".join(
            record.run_id for record in candidate_selection.unsuccessful
        )
        if persisted is None:
            raise ContractError("failed candidate runs have no exact persisted binding")
        return OrchestrationPlan(
            action=OrchestrationAction.BLOCKED,
            reason=(
                f"candidate run(s) {failed_ids} failed for this exact provenance; "
                "automatic candidate retries are disabled to prevent unbounded "
                "daily duplicates; a maintainer must diagnose the failure and "
                "explicitly dispatch one deliberate new first-attempt run with the "
                "same exact binding, or establish new provenance"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=persisted.release_target,
        )

    if candidate_selection.succeeded_run_id is None:
        observation = PipelineObservation(
            fresh_binding=fresh_binding,
        )
        plan = plan_pipeline(
            provenance=provenance,
            correlation_id=correlation_id,
            observation=observation,
        )
        return _execute_dispatch(
            gateway,
            plan,
            default_branch=default_branch,
            workflow_path=CANDIDATE_WORKFLOW_PATH,
            dry_run=dry_run,
        )

    if persisted is None:
        raise ContractError(
            "a successful candidate run advertises no parsable pipeline binding"
        )
    candidate = verify_candidate_run(
        gateway,
        run_id=candidate_selection.succeeded_run_id,
        provenance=provenance,
        correlation_id=correlation_id,
        binding=persisted,
        default_branch=default_branch,
        workspace=workspace,
    )

    qualification_runs = _fetch_runs(
        gateway,
        workflow_file=QUALIFICATION_WORKFLOW_FILE,
        workflow_path=QUALIFICATION_WORKFLOW_PATH,
        default_branch=default_branch,
        created_since=run_history_since,
    )
    expected_qualification_name = qualification_run_name(
        correlation_id, candidate.run_id
    )
    qualification_selection = select_pipeline_runs(
        qualification_runs,
        label="qualification",
        matcher=lambda name: name == expected_qualification_name,
        default_branch=default_branch,
    )
    if (
        qualification_selection.succeeded_run_id is None
        and qualification_selection.in_flight_run_id is None
        and qualification_selection.unsuccessful
    ):
        failed_ids = ", ".join(
            record.run_id for record in qualification_selection.unsuccessful
        )
        return OrchestrationPlan(
            action=OrchestrationAction.BLOCKED,
            reason=(
                f"qualification run(s) {failed_ids} failed for candidate "
                f"{candidate.run_id}; automatic qualification retries are disabled "
                "to prevent unbounded duplicates, so a maintainer must diagnose the "
                "heavy-gate failure before this candidate can advance"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=candidate.binding.release_target,
            candidate_run_id=candidate.run_id,
        )

    if qualification_selection.succeeded_run_id is None:
        plan = plan_pipeline(
            provenance=provenance,
            correlation_id=correlation_id,
            observation=PipelineObservation(
                binding=candidate.binding,
                candidate_run_id=candidate.run_id,
                qualification_in_flight_run_id=(
                    qualification_selection.in_flight_run_id
                ),
            ),
        )
        return _execute_dispatch(
            gateway,
            plan,
            default_branch=default_branch,
            workflow_path=QUALIFICATION_WORKFLOW_PATH,
            dry_run=dry_run,
        )

    verify_qualification_run(
        gateway,
        run_id=qualification_selection.succeeded_run_id,
        candidate=candidate,
        provenance=provenance,
        correlation_id=correlation_id,
        default_branch=default_branch,
        workspace=workspace,
    )

    publish_runs = _fetch_runs(
        gateway,
        workflow_file=PUBLISH_WORKFLOW_FILE,
        workflow_path=PUBLISH_WORKFLOW_PATH,
        default_branch=default_branch,
        created_since=run_history_since,
    )
    expected_publish_name = publish_run_name(
        correlation_id,
        candidate.run_id,
        qualification_selection.succeeded_run_id,
        candidate.binding,
    )
    publish_selection = select_pipeline_runs(
        publish_runs,
        label="publication",
        matcher=lambda name: name == expected_publish_name,
        default_branch=default_branch,
    )
    if not publication_allowed:
        if (
            publish_selection.in_flight_run_id is not None
            or publish_selection.succeeded_run_id is not None
        ):
            return OrchestrationPlan(
                action=OrchestrationAction.BLOCKED,
                reason=(
                    f"publication for {provenance.native_release_tag} was started "
                    f"before earlier native release {publication_barrier_native_tag} "
                    "was immutably published"
                ),
                provenance=provenance,
                correlation_id=correlation_id,
                release_target=candidate.binding.release_target,
                candidate_run_id=candidate.run_id,
                qualification_run_id=qualification_selection.succeeded_run_id,
                in_flight_workflow=(
                    PUBLISH_WORKFLOW_FILE
                    if publish_selection.in_flight_run_id is not None
                    else None
                ),
                in_flight_run_id=publish_selection.in_flight_run_id,
            )
        return OrchestrationPlan(
            action=OrchestrationAction.WAITING_FOR_PRIOR_PUBLICATION,
            reason=(
                f"candidate {candidate.run_id} and qualification "
                f"{qualification_selection.succeeded_run_id} are proven, but "
                f"earlier native release {publication_barrier_native_tag} must be "
                "immutably published first to preserve monotonic output ordering"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=candidate.binding.release_target,
            candidate_run_id=candidate.run_id,
            qualification_run_id=qualification_selection.succeeded_run_id,
        )
    plan = plan_pipeline(
        provenance=provenance,
        correlation_id=correlation_id,
        observation=PipelineObservation(
            binding=candidate.binding,
            candidate_run_id=candidate.run_id,
            qualification_run_id=qualification_selection.succeeded_run_id,
            publish_in_flight_run_id=publish_selection.in_flight_run_id,
            publish_succeeded_run_id=publish_selection.succeeded_run_id,
            publish_retry=bool(publish_selection.unsuccessful),
        ),
    )
    return _execute_dispatch(
        gateway,
        plan,
        default_branch=default_branch,
        workflow_path=PUBLISH_WORKFLOW_PATH,
        dry_run=dry_run,
    )


def _execute_dispatch(
    gateway: Gateway,
    plan: OrchestrationPlan,
    *,
    default_branch: str,
    workflow_path: str,
    dry_run: bool,
) -> OrchestrationPlan:
    if plan.dispatch_workflow is None or plan.dispatch_inputs is None:
        return plan
    run_name = plan.dispatch_run_name
    if run_name is None:
        raise ContractError("a dispatch plan must carry its deterministic run name")

    identity = gateway.dispatch_identity()
    if identity != REPOSITORY_OWNER:
        return OrchestrationPlan(
            action=OrchestrationAction.BLOCKED,
            reason=(
                "automatic dispatch identity could not be proven as "
                f"{REPOSITORY_OWNER!r} (resolved {identity!r}); the owner-only actor "
                "gates stay in force and nothing was dispatched"
            ),
            provenance=plan.provenance,
            correlation_id=plan.correlation_id,
            release_target=plan.release_target,
            candidate_run_id=plan.candidate_run_id,
            qualification_run_id=plan.qualification_run_id,
        )

    workflow_file = plan.dispatch_workflow
    governance = require_immutable_release_governance(gateway)
    inputs = dict(plan.dispatch_inputs)
    if "assets_immutable_releases_enabled" in WORKFLOW_DISPATCH_INPUTS.get(
        workflow_file, ()
    ):
        # Carried from the live read above, so a governance regression can never
        # be papered over by a literal the dispatcher writes about itself.
        inputs["assets_immutable_releases_enabled"] = (
            "true" if governance["enabled"] is True else "false"
        )
    if "publish_approved" in WORKFLOW_DISPATCH_INPUTS.get(workflow_file, ()):
        # In automatic mode the established solo-maintainer environment policy
        # is the approval boundary. Never let the pure planner assert it.
        require_publication_environment(gateway)
        inputs["publish_approved"] = "true"
    inputs = require_exact_dispatch_inputs(workflow_file, inputs)

    if dry_run:
        return _with_dispatch_record(plan, ref=default_branch, inputs=inputs)

    if _find_named_run(
        gateway,
        workflow_file=workflow_file,
        workflow_path=workflow_path,
        default_branch=default_branch,
        created_since=plan.provenance.native_release_published_at,
        run_name=run_name,
    ) is not None:
        raise ContractError(
            f"a run named {run_name!r} already exists; refusing to dispatch a duplicate"
        )

    gateway.dispatch_workflow(
        workflow_file=workflow_file, ref=default_branch, inputs=inputs
    )

    dispatched_run_id = None
    for attempt in range(DISPATCH_READBACK_ATTEMPTS):
        if attempt:
            gateway.sleep(DISPATCH_READBACK_DELAY_SECONDS)
        dispatched_run_id = _find_named_run(
            gateway,
            workflow_file=workflow_file,
            workflow_path=workflow_path,
            default_branch=default_branch,
            created_since=plan.provenance.native_release_published_at,
            run_name=run_name,
        )
        if dispatched_run_id is not None:
            break
    if dispatched_run_id is None:
        raise ContractError(
            f"dispatched {workflow_file} but no run named {run_name!r} appeared; "
            "the pipeline state is unknown"
        )
    return _with_dispatch_record(
        plan, ref=default_branch, inputs=inputs, dispatched_run_id=dispatched_run_id
    )


def _with_dispatch_record(
    plan: OrchestrationPlan,
    *,
    ref: str,
    inputs: Mapping[str, str],
    dispatched_run_id: str | None = None,
) -> OrchestrationPlan:
    """Record the exact ref and inputs a dispatch used, or would have used."""
    return OrchestrationPlan(
        action=plan.action,
        reason=plan.reason,
        provenance=plan.provenance,
        correlation_id=plan.correlation_id,
        release_target=plan.release_target,
        candidate_run_id=plan.candidate_run_id,
        qualification_run_id=plan.qualification_run_id,
        dispatch_workflow=plan.dispatch_workflow,
        dispatch_ref=ref,
        dispatch_run_name=plan.dispatch_run_name,
        dispatch_inputs=dict(inputs),
        dispatched_run_id=dispatched_run_id,
    )


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


# --------------------------------------------------------------------------
# Reporting and CLI
# --------------------------------------------------------------------------


def render_step_summary(plan: OrchestrationPlan) -> str:
    target = plan.release_target
    lines = [
        "### Stable Web bridge release orchestration",
        "",
        f"- Action: `{plan.action.value}`",
        f"- Reason: {plan.reason}",
        f"- Correlation: `{plan.correlation_id}`",
        f"- Bridge source: `{plan.provenance.bridge_source_sha}`",
        f"- llama.cpp: `{plan.provenance.upstream_tag}@{plan.provenance.upstream_commit}`",
        f"- Native release: `{plan.provenance.native_repo}@{plan.provenance.native_release_tag}`",
        f"- Native manifest SHA-256: `{plan.provenance.native_manifest_sha256}`",
    ]
    if target is not None:
        lines.append(
            f"- Output release: `{target.release_tag}` (rebuild `{target.release_rebuild}`)"
        )
    if plan.candidate_run_id:
        lines.append(f"- Candidate run: `{plan.candidate_run_id}`")
    if plan.qualification_run_id:
        lines.append(f"- Qualification run: `{plan.qualification_run_id}`")
    if plan.in_flight_workflow:
        lines.append(
            f"- In flight: `{plan.in_flight_workflow}` run `{plan.in_flight_run_id}`"
        )
    if plan.dispatch_workflow:
        lines.append(
            f"- Dispatched: `{plan.dispatch_workflow}` at `{plan.dispatch_ref}` "
            f"as `{plan.dispatch_run_name}` (run `{plan.dispatched_run_id}`)"
        )
    lines.append("")
    return "\n".join(lines)


_FAILING_ACTIONS = frozenset({OrchestrationAction.BLOCKED})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    scan = subparsers.add_parser(
        "scan-native", help="Extract exact provenance from a native assets.json"
    )
    scan.add_argument("--manifest", required=True, type=Path)
    scan.add_argument("--native-release-tag", required=True)
    scan.add_argument("--bridge-source-sha", required=True)
    scan.add_argument("--native-release-published-at", required=True)
    scan.add_argument(
        "--channel",
        required=True,
        choices=sorted(_CHANNELS),
        help="Channel the caller asked for; a mismatch fails closed",
    )
    scan.add_argument("--output-json", type=Path)

    select = subparsers.add_parser(
        "select-stable-native-backlog",
        help="Select every post-baseline stable native release",
    )
    select.add_argument("--releases-json", required=True, type=Path)
    select.add_argument("--output-json", type=Path)

    orchestrate = subparsers.add_parser(
        "orchestrate", help="Advance the stable release pipeline by one exact step"
    )
    orchestrate.add_argument("--provenance-json", required=True, type=Path)
    orchestrate.add_argument("--workspace", required=True, type=Path)
    orchestrate.add_argument("--output-plan-json", type=Path)
    orchestrate.add_argument("--step-summary-file", type=Path)
    orchestrate.add_argument("--dry-run", action="store_true")

    backlog = subparsers.add_parser(
        "orchestrate-backlog",
        help="Advance each exact stable provenance by at most one stage",
    )
    backlog.add_argument("--provenance-list-json", required=True, type=Path)
    backlog.add_argument("--workspace", required=True, type=Path)
    backlog.add_argument("--output-plan-json", type=Path)
    backlog.add_argument("--step-summary-file", type=Path)
    backlog.add_argument("--dry-run", action="store_true")
    return parser


def _provenance_to_dict(provenance: NativeProvenance) -> dict[str, str]:
    return {
        "bridge_source_sha": provenance.bridge_source_sha,
        "upstream_tag": provenance.upstream_tag,
        "upstream_commit": provenance.upstream_commit,
        "native_repo": provenance.native_repo,
        "native_release_tag": provenance.native_release_tag,
        "native_commit": provenance.native_commit,
        "native_manifest_sha256": provenance.native_manifest_sha256,
        "native_release_published_at": provenance.native_release_published_at,
    }


def _load_provenance_payload(payload: Any, label: str) -> NativeProvenance:
    if not isinstance(payload, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    expected = {
        "bridge_source_sha",
        "upstream_tag",
        "upstream_commit",
        "native_repo",
        "native_release_tag",
        "native_commit",
        "native_manifest_sha256",
        "native_release_published_at",
    }
    if set(payload) != expected:
        raise ContractError(f"{label} has missing or unexpected fields")
    return require_stable_provenance(NativeProvenance(**payload))


def _load_provenance(path: Path) -> NativeProvenance:
    payload = _strict_json_loads(path.read_text(encoding="utf-8"), "provenance")
    return _load_provenance_payload(payload, "provenance")


def _load_provenance_backlog(path: Path) -> list[NativeProvenance]:
    payload = _strict_json_loads(
        path.read_text(encoding="utf-8"), "provenance backlog"
    )
    if not isinstance(payload, list):
        raise ContractError("provenance backlog root must be a JSON array")
    provenances = [
        _load_provenance_payload(item, f"provenance backlog entry {index}")
        for index, item in enumerate(payload)
    ]
    correlations: set[str] = set()
    native_tags: set[str] = set()
    for provenance in provenances:
        correlation = compute_correlation_id(provenance)
        if correlation in correlations or provenance.native_release_tag in native_tags:
            raise ContractError("provenance backlog contains a duplicate pipeline")
        correlations.add(correlation)
        native_tags.add(provenance.native_release_tag)
    provenances.sort(
        key=lambda value: (
            value.native_release_published_at,
            (*parse_release_tag(value.native_release_tag).version_parts,
             parse_release_tag(value.native_release_tag).rebuild),
        )
    )
    return provenances


def _require_cli_caller() -> None:
    require_orchestration_caller(
        os.environ.get("GITHUB_EVENT_NAME", ""),
        os.environ.get("GITHUB_ACTOR", ""),
        os.environ.get("GITHUB_TRIGGERING_ACTOR", ""),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.subcommand == "scan-native":
        provenance = scan_native_provenance(
            manifest_path=args.manifest,
            native_release_tag=args.native_release_tag,
            bridge_source_sha=args.bridge_source_sha,
            channel=args.channel,
            native_release_published_at=args.native_release_published_at,
        )
        payload = json.dumps(_provenance_to_dict(provenance), indent=2, sort_keys=True)
        if args.output_json:
            args.output_json.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0

    if args.subcommand == "select-stable-native-backlog":
        releases = _strict_json_loads(
            args.releases_json.read_text(encoding="utf-8"),
            "native release listing",
        )
        if not isinstance(releases, list):
            raise ContractError("native release listing must be a JSON array")
        selected = select_stable_native_backlog(releases)
        payload = json.dumps(selected, indent=2)
        if args.output_json:
            args.output_json.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0

    _require_cli_caller()
    gateway = GhGateway(
        read_token=os.environ.get("GH_TOKEN", ""),
        dispatch_token=os.environ.get("WEBGPU_BRIDGE_ASSETS_PAT"),
    )
    args.workspace.mkdir(parents=True, exist_ok=True)

    if args.subcommand == "orchestrate-backlog":
        provenances = _load_provenance_backlog(args.provenance_list_json)
        plans: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        reserved_release_tags: set[str] = set()
        publication_barrier_native_tag: str | None = None
        for provenance in provenances:
            correlation_id = compute_correlation_id(provenance)
            pipeline_workspace = args.workspace / correlation_id
            pipeline_workspace.mkdir(parents=True, exist_ok=True)
            try:
                plan = advance_pipeline(
                    gateway,
                    provenance=provenance,
                    workspace=pipeline_workspace,
                    dry_run=args.dry_run,
                    reserved_release_tags=reserved_release_tags,
                    publication_allowed=publication_barrier_native_tag is None,
                    publication_barrier_native_tag=publication_barrier_native_tag,
                )
                plans.append(plan.to_dict())
                if plan.release_target is not None:
                    reserved_release_tags.add(plan.release_target.release_tag)
                if (
                    publication_barrier_native_tag is None
                    and plan.action is not OrchestrationAction.NOOP
                ):
                    publication_barrier_native_tag = provenance.native_release_tag
                if args.step_summary_file:
                    with open(args.step_summary_file, "a", encoding="utf-8") as handle:
                        handle.write(render_step_summary(plan) + "\n")
            except ContractError as error:
                errors.append(
                    {
                        "correlation_id": correlation_id,
                        "native_release_tag": provenance.native_release_tag,
                        "error": str(error),
                    }
                )
                if args.step_summary_file:
                    with open(args.step_summary_file, "a", encoding="utf-8") as handle:
                        handle.write(
                            "### Blocked stable Web bridge release orchestration\n\n"
                            f"- Correlation: `{correlation_id}`\n"
                            f"- Native release: `{provenance.native_release_tag}`\n"
                            f"- Error: {error}\n\n"
                        )
                # A transport/readback error can mean a dispatch occurred but
                # its state is not yet observable. Stop before another backlog
                # entry can claim a colliding output identity under uncertainty.
                break
        result = {
            "schema_version": 1,
            "plans": plans,
            "errors": errors,
        }
        payload = json.dumps(result, indent=2, sort_keys=True)
        if args.output_plan_json:
            args.output_plan_json.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        blocked = [plan for plan in plans if plan["action"] in {
            action.value for action in _FAILING_ACTIONS
        }]
        if errors or blocked:
            print("error: one or more stable pipelines are blocked", file=sys.stderr)
            return 1
        return 0

    provenance = _load_provenance(args.provenance_json)
    plan = advance_pipeline(
        gateway,
        provenance=provenance,
        workspace=args.workspace,
        dry_run=args.dry_run,
    )
    payload = json.dumps(plan.to_dict(), indent=2, sort_keys=True)
    if args.output_plan_json:
        args.output_plan_json.write_text(payload + "\n", encoding="utf-8")
    if args.step_summary_file:
        with open(args.step_summary_file, "a", encoding="utf-8") as handle:
            handle.write(render_step_summary(plan) + "\n")
    print(payload)
    if plan.action in _FAILING_ACTIONS:
        print(f"error: {plan.reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        raise SystemExit(f"error: {error}") from error
