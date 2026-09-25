"""Native release scanning for the stable release orchestrator.

Extracts exact provenance from a native ``assets.json`` on either channel,
selects every published stable native release after the automation baseline,
and orders native release tags.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from release_contract import (
    Channel,
    ContractError,
    NATIVE_REPOSITORY,
    _strict_json_loads,
    parse_release_tag,
    resolve_native_manifest,
)
from release_orchestrator_model import NativeProvenance, _UTC_TIMESTAMP_RE


# v0.2.0-1 was published as the first verified immutable automatic-publication
# baseline (Web assets v0.1.39). Older native releases belong to the historical
# pre-automation series and must not be silently rebuilt by backlog scans.
STABLE_AUTOMATION_BASELINE_NATIVE_TAG = "v0.2.0-1"
STABLE_AUTOMATION_BASELINE_PUBLISHED_AT = "2026-08-25T08:57:12Z"

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
    bridge_build_sha: str,
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
        bridge_build_sha=bridge_build_sha,
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


def _native_release_order(native_release_tag: str) -> tuple[int, ...]:
    version = parse_release_tag(native_release_tag)
    return (*version.version_parts, version.rebuild)
