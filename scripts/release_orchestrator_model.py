"""Shared identities and data types of the stable release orchestrator.

Workflow files and paths, the repository owner, the one pre-automation
publication (``v0.1.40``) that keeps its historical manifest contract, the
identity patterns and scalar validators every stage shares, and the provenance,
binding, plan, and observation types the stage modules pass to each other.
``stable_release_orchestrator.py`` is the CLI entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from release_contract import (
    BRIDGE_REPOSITORY,
    Channel,
    ContractError,
    NATIVE_REPOSITORY,
    parse_release_tag,
    parse_upstream_tag,
    require_repository,
    require_sha256,
    validate_release_identity,
)
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

# v0.1.40 was immutably published for native/upstream v0.3.0 immediately
# before hosted automatic qualification replaced maintainer-run attestation.
# Its exact published bytes retain the historical gate vocabulary. This
# compatibility identity is deliberately narrower than the general candidate
# validator: every new candidate must still require hosted qualification.
LEGACY_MANUAL_QUALIFICATION_RELEASE_TAG = "v0.1.40"
LEGACY_MANUAL_QUALIFICATION_BRIDGE_COMMIT = (
    "0bdc8286fd52b70da27f5b039e1b4278361da0be"
)
LEGACY_MANUAL_QUALIFICATION_NATIVE_TAG = "v0.3.0"
LEGACY_MANUAL_QUALIFICATION_NATIVE_COMMIT = (
    "28fca14873d4b4c531bef4425b261e2b911bdcce"
)
LEGACY_MANUAL_QUALIFICATION_UPSTREAM_TAG = "v0.3.0"
LEGACY_MANUAL_QUALIFICATION_UPSTREAM_COMMIT = (
    "c1d0e7a004015f23bc0233470b747b596f29b264"
)
LEGACY_MANUAL_QUALIFICATION_NATIVE_MANIFEST_SHA256 = (
    "811fda999e70c3ad2716d1c196688dd38db62cf11a78044855ca94f71fabed45"
)
LEGACY_MANUAL_QUALIFICATION_GATES = {
    "state_persistence": "passed",
    "multimodal": "passed",
    "speech_to_text": "required-local-attestation",
    "text_to_speech": "required-local-attestation",
}
LEGACY_MANUAL_UNPROVEN_CAPABILITIES = {
    "real_device_intelligibility": "unproven",
    "real_device_playback": "unproven",
    "speaker_reference_fidelity": "unproven",
}

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_UTC_TIMESTAMP_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z"
)


@dataclass(frozen=True)
class NativeProvenance:
    """Exact identity of the native release this pipeline is aligned to."""

    bridge_source_sha: str
    bridge_build_sha: str
    upstream_tag: str
    upstream_commit: str
    native_repo: str
    native_release_tag: str
    native_commit: str
    native_manifest_sha256: str
    native_release_published_at: str

    def __post_init__(self) -> None:
        for field_name in (
            "bridge_source_sha",
            "bridge_build_sha",
            "upstream_commit",
            "native_commit",
        ):
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


def _published_manifest_compatibility(
    *, tag: str, provenance: NativeProvenance
) -> tuple[Mapping[str, str], Mapping[str, str]] | None:
    """Return the one immutable pre-automation manifest contract, if applicable."""

    if (
        tag == LEGACY_MANUAL_QUALIFICATION_RELEASE_TAG
        and provenance.bridge_build_sha
        == LEGACY_MANUAL_QUALIFICATION_BRIDGE_COMMIT
        and provenance.native_release_tag == LEGACY_MANUAL_QUALIFICATION_NATIVE_TAG
        and provenance.native_commit == LEGACY_MANUAL_QUALIFICATION_NATIVE_COMMIT
        and provenance.upstream_tag == LEGACY_MANUAL_QUALIFICATION_UPSTREAM_TAG
        and provenance.upstream_commit
        == LEGACY_MANUAL_QUALIFICATION_UPSTREAM_COMMIT
        and provenance.native_manifest_sha256
        == LEGACY_MANUAL_QUALIFICATION_NATIVE_MANIFEST_SHA256
    ):
        return (
            LEGACY_MANUAL_QUALIFICATION_GATES,
            LEGACY_MANUAL_UNPROVEN_CAPABILITIES,
        )
    return None


def require_stable_provenance(provenance: NativeProvenance) -> NativeProvenance:
    """Only the stable channel is orchestrated; everything else is scan-only."""
    if provenance.channel is not Channel.STABLE:
        raise ContractError(
            "the release orchestrator only advances the stable channel, but this "
            f"provenance is {provenance.channel.value} "
            f"({provenance.native_release_tag}@{provenance.upstream_tag})"
        )
    return provenance


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
    directory: Path | None = None


class OrchestrationAction(str, Enum):
    NOOP = "noop"
    SUPERSEDED = "superseded"
    SATISFIED_BY_IDENTICAL_RELEASE = "satisfied_by_identical_release"
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
                "bridge_build_sha": self.provenance.bridge_build_sha,
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
    satisfied_by: PublishedRelease | None = None
    qualification_in_flight_run_id: str | None = None
    qualification_run_id: str | None = None
    publish_in_flight_run_id: str | None = None
    publish_succeeded_run_id: str | None = None
    publish_retry: bool = False


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value
