"""Deterministic correlation IDs and workflow run names.

The live ``actions/runs`` API never echoes a run's dispatch inputs, so each
stage workflow renders a run-name from its own exact inputs. These helpers
derive the correlation ID, build each stage's run name, and parse a run name
back into its pipeline binding, failing closed when a name claims this
correlation but does not parse exactly.
"""

from __future__ import annotations

import re

from release_contract import ContractError, require_correlation_id
from release_orchestrator_model import (
    LEGACY_MANUAL_QUALIFICATION_BRIDGE_COMMIT,
    LEGACY_MANUAL_QUALIFICATION_NATIVE_COMMIT,
    LEGACY_MANUAL_QUALIFICATION_NATIVE_MANIFEST_SHA256,
    LEGACY_MANUAL_QUALIFICATION_NATIVE_TAG,
    LEGACY_MANUAL_QUALIFICATION_UPSTREAM_COMMIT,
    LEGACY_MANUAL_QUALIFICATION_UPSTREAM_TAG,
    NativeProvenance,
    PipelineBinding,
)


# Run-name has no documented length guarantee. Keep this internal identity well
# below common database/display limits and accept only printable ASCII emitted
# by the exact input validators. A truncated title can therefore never be
# mistaken for a pipeline identity.
MAX_RUN_NAME_CHARACTERS = 200

_RUN_ID_RE = re.compile(r"[1-9][0-9]*")
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9 ._:/-]+")

_CANDIDATE_RUN_NAME_RE = re.compile(
    r"bridge-candidate (?P<correlation_id>\S+) source:(?P<bridge_source_sha>\S+)"
    r" tag:(?P<release_tag>\S+) rebuild:(?P<release_rebuild>\S+)"
)
_PUBLISH_RUN_NAME_RE = re.compile(
    r"publish-assets (?P<correlation_id>\S+) candidate:(?P<candidate_run_id>\S+)"
    r" qualification:(?P<qualification_run_id>\S+) source:(?P<bridge_source_sha>\S+)"
    r" tag:(?P<release_tag>\S+) rebuild:(?P<release_rebuild>\S+)"
)


def compute_correlation_id(provenance: NativeProvenance) -> str:
    """Derive one stable pipeline identity from native and governed build inputs.

    The checkout source may advance while a pipeline is in flight because
    workflows, tests, or docs changed. The governed build identity does not, so
    those changes cannot orphan a candidate. A runtime/build change deliberately
    creates a new correlation and therefore requires a new qualified candidate.

    The one immutable pre-automation publication keeps its historical
    correlation only while the governed build identity is still its exact
    bridge commit.
    """
    if (
        provenance.bridge_build_sha
        == LEGACY_MANUAL_QUALIFICATION_BRIDGE_COMMIT
        and provenance.native_release_tag == LEGACY_MANUAL_QUALIFICATION_NATIVE_TAG
        and provenance.native_commit == LEGACY_MANUAL_QUALIFICATION_NATIVE_COMMIT
        and provenance.upstream_tag == LEGACY_MANUAL_QUALIFICATION_UPSTREAM_TAG
        and provenance.upstream_commit
        == LEGACY_MANUAL_QUALIFICATION_UPSTREAM_COMMIT
        and provenance.native_manifest_sha256
        == LEGACY_MANUAL_QUALIFICATION_NATIVE_MANIFEST_SHA256
    ):
        return require_correlation_id(
            f"auto-stable-{provenance.native_release_tag}"
            f"-{provenance.native_manifest_sha256[:16]}"
        )
    raw = (
        f"auto-stable-{provenance.native_release_tag}"
        f"-{provenance.native_manifest_sha256[:16]}"
        f"-build-{provenance.bridge_build_sha[:16]}"
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
