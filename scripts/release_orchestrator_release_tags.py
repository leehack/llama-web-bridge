"""Bridge-asset output tag selection for the stable release orchestrator.

Picks the next free npm-shaped ``vMAJOR.MINOR.PATCH`` tag independently of the
upstream tag, and determines which output tags candidate runs still claim
across correlations.
"""

from __future__ import annotations

import re
from typing import Sequence

from release_contract import (
    Channel,
    ContractError,
    parse_release_tag,
    parse_upstream_tag,
    validate_new_release_identity,
)
from release_orchestrator_model import PipelineBinding, ReleaseTarget
from release_orchestrator_run_names import _CANDIDATE_RUN_NAME_RE
from release_orchestrator_workflow_runs import RunRecord


# Bridge assets version independently of llama.cpp: an upstream v0.2.0 candidate
# publishes as the next bridge patch (v0.1.39 -> v0.1.40), never as v0.2.0.
INITIAL_STABLE_RELEASE_TAG = "v0.1.0"


def select_next_release_target(
    release_tags: Sequence[str], *, upstream_tag: str, taken: Sequence[str] | set[str] = ()
) -> ReleaseTarget:
    """Pick the next free bridge-asset tag, independently of the upstream tag.

    ``release_tags`` are the assets repository's existing releases. ``taken``
    are tags still claimed by a pipeline. Both set the version floor, so a new
    tag is never lower than a published tag or a live claim. A claim that is
    released without publication stops counting, so its version may be taken
    again or stay unused. The result is always an unsuffixed
    ``vMAJOR.MINOR.PATCH`` with rebuild 0: npm orders ``-N`` as a prerelease of
    the same version, so a collision moves to the next free patch version
    instead of a rebuild suffix.
    """
    parse_upstream_tag(upstream_tag)
    published = {tag for tag in release_tags if isinstance(tag, str)}
    claimed = published | {tag for tag in taken if isinstance(tag, str)}
    versions = []
    for tag in claimed:
        try:
            version = parse_release_tag(tag, allow_legacy=True)
        except ContractError:
            continue
        if version.channel is Channel.STABLE:
            versions.append(version)
    if versions:
        highest = max(versions, key=lambda value: (*value.version_parts, value.rebuild))
        major, minor, patch = highest.version_parts
        patch += 1
    else:
        major, minor, patch = parse_release_tag(INITIAL_STABLE_RELEASE_TAG).version_parts

    tag = f"v{major}.{minor}.{patch}"
    while tag in claimed:
        patch += 1
        tag = f"v{major}.{minor}.{patch}"
    validate_new_release_identity(tag, 0, upstream_tag)
    return ReleaseTarget(tag, 0)


def _claimed_rebuilds_of(binding: PipelineBinding, claimed: set[str]) -> list[str]:
    """Claimed tags that can publish only after ``binding`` publishes rebuild 0."""
    if binding.release_rebuild != 0:
        return []
    version = parse_release_tag(binding.release_tag)
    dependents: list[str] = []
    for tag in claimed:
        try:
            claim = parse_release_tag(tag)
        except ContractError:
            continue
        if claim.version_parts == version.version_parts and claim.rebuild > 0:
            dependents.append(tag)
    return sorted(dependents)


_CORRELATION_BUILD_RE = re.compile(r"-build-(?P<build>[0-9a-f]{16})$")


def _run_correlation_id(record: RunRecord) -> str | None:
    fields = record.run_name.split()
    return fields[1] if len(fields) > 1 else None


def _names_other_build(correlation_id: str, bridge_build_sha: str) -> bool:
    build = _CORRELATION_BUILD_RE.search(correlation_id)
    return build is not None and build.group("build") != bridge_build_sha[:16]


def has_other_build_claim(
    candidate_runs: Sequence[RunRecord], *, bridge_build_sha: str
) -> bool:
    for record in candidate_runs:
        match = _CANDIDATE_RUN_NAME_RE.fullmatch(record.run_name)
        if match is not None and _names_other_build(
            match.group("correlation_id"), bridge_build_sha
        ):
            return True
    return False


def claimed_release_tags(
    candidate_runs: Sequence[RunRecord],
    *,
    bridge_build_sha: str,
    downstream_runs: Sequence[RunRecord] = (),
    satisfied_correlation_ids: Sequence[str] | set[str] = (),
) -> set[str]:
    """Output tags still claimed by candidate runs, across correlations.

    A claim is dropped when its correlation names a build identity other than
    ``bridge_build_sha`` and no candidate, qualification or publication run of
    that correlation is in flight, or when this scan already found that
    correlation satisfied by an identical release. Scans advance only the
    current build identity and a satisfied correlation publishes nothing, so
    nothing dispatches for such a correlation again.
    """
    in_flight = {
        _run_correlation_id(record)
        for record in (*candidate_runs, *downstream_runs)
        if record.in_flight
    }
    satisfied = set(satisfied_correlation_ids)
    claimed: set[str] = set()
    for record in candidate_runs:
        match = _CANDIDATE_RUN_NAME_RE.fullmatch(record.run_name)
        if match is None:
            continue
        correlation_id = match.group("correlation_id")
        if correlation_id in satisfied or (
            _names_other_build(correlation_id, bridge_build_sha)
            and correlation_id not in in_flight
        ):
            continue
        claimed.add(match.group("release_tag"))
    return claimed
