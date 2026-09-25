"""Assets-repository release discovery and proof for the orchestrator.

Resolves tag commits, lists releases and tag refs, finds the release a
correlation or native alignment points to, compares publication bytes, bounds
run recovery, and re-proves a published release through independent readbacks,
downloaded bytes, and its signed release attestation.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from release_contract import (
    ASSETS_REPOSITORY,
    Channel,
    ContractError,
    NATIVE_REPOSITORY,
    parse_release_tag,
    require_correlation_id,
    require_repository,
    require_sha256,
    validate_release_attestation,
    validate_release_immutability,
)
from release_publication_state import PUBLICATION_FILES
from generate_release_manifest import ARTIFACTS
import release_qualification as rq
from release_orchestrator_model import (
    LEGACY_MANUAL_QUALIFICATION_BRIDGE_COMMIT,
    LEGACY_MANUAL_QUALIFICATION_RELEASE_TAG,
    NativeProvenance,
    PublishedRelease,
    _COMMIT_RE,
    _UTC_TIMESTAMP_RE,
    _published_manifest_compatibility,
    _require_positive_int,
    _require_str,
)
from release_orchestrator_native import _native_release_order
from release_orchestrator_stage_proofs import validate_candidate_manifest
from release_orchestrator_transport import Gateway


def _pages(payload: Any, label: str) -> list[Any]:
    """Normalize ``gh api --paginate --slurp`` output to a page list."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        return [payload]
    raise ContractError(f"{label} response must be a JSON array or object")


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


_NATIVE_ALIGNMENT_RE = re.compile(
    r"(?m)^Native: `" + re.escape(NATIVE_REPOSITORY) + r"@([^`\r\n]+)`\r?$"
)


def latest_published_native_alignment(
    releases: Sequence[Mapping[str, Any]],
) -> tuple[str, str] | None:
    """Return ``(native_tag, asset_tag)`` for the newest stable native release
    named by a non-draft stable asset release's ``Native:`` marker, else None.

    Raises ContractError if one release names several native releases or a
    malformed tag.
    """
    latest: tuple[tuple[int, ...], str, str] | None = None
    for release in releases:
        asset_tag = release.get("tag_name")
        body = release.get("body")
        if not isinstance(asset_tag, str) or not isinstance(body, str):
            continue
        try:
            asset_version = parse_release_tag(asset_tag, allow_legacy=True)
        except ContractError:
            continue
        if (
            asset_version.channel is not Channel.STABLE
            or release.get("draft") is not False
        ):
            continue
        claims = set(_NATIVE_ALIGNMENT_RE.findall(body))
        if not claims:
            continue
        if len(claims) != 1:
            raise ContractError(
                f"asset release {asset_tag!r} records {len(claims)} native alignments"
            )
        native_tag = claims.pop()
        try:
            native_version = parse_release_tag(native_tag)
        except ContractError as error:
            raise ContractError(
                f"asset release {asset_tag!r} records a malformed native "
                f"alignment: {error}"
            ) from error
        if native_version.channel is not Channel.STABLE:
            continue
        order = _native_release_order(native_tag)
        if latest is None or order > latest[0]:
            latest = (order, native_tag, asset_tag)
    return None if latest is None else latest[1:]


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
    matches: list[Mapping[str, Any]] = []
    for release in releases:
        body = release.get("body")
        if not isinstance(body, str):
            continue
        if marker in body:
            matches.append(release)
            continue
        if native_manifest_marker not in body or native_release_marker not in body:
            continue
        claimed_correlations = re.findall(
            r"(?m)^Orchestrator correlation: `([^`\r\n]+)`$", body
        )
        valid_claims: list[str] = []
        for claim in claimed_correlations:
            try:
                valid_claims.append(require_correlation_id(claim))
            except ContractError:
                pass
        if valid_claims:
            # The same native release may intentionally have multiple bridge
            # publications after governed runtime/build changes. A different
            # well-formed correlation is another pipeline, not damaged state.
            continue
        # With no valid correlation marker, the native identity is still close
        # enough to block a duplicate until immutable readback diagnoses it.
        matches.append(release)
    if len(matches) > 1:
        raise ContractError(
            f"{len(matches)} asset releases claim correlation {correlation_id!r}: "
            + ", ".join(str(release.get("tag_name")) for release in matches)
        )
    return matches[0] if matches else None


def latest_aligned_release(
    releases: Sequence[Mapping[str, Any]], provenance: NativeProvenance
) -> tuple[Mapping[str, Any], str] | None:
    """Return the newest non-draft stable asset release whose notes record this
    provenance's exact native release and native manifest digest, with the one
    correlation it claims, else None.

    A release with the same native markers but not exactly one well-formed
    correlation marker is skipped. The pre-automation ``v0.1.40`` manifest
    follows a different contract and is only comparable by its own provenance.
    """
    native_release_marker = (
        f"Native: `{provenance.native_repo}@{provenance.native_release_tag}`"
    )
    native_manifest_marker = (
        f"Native manifest SHA-256: `{provenance.native_manifest_sha256}`"
    )
    latest: tuple[tuple[int, ...], Mapping[str, Any], str] | None = None
    for release in releases:
        tag = release.get("tag_name")
        body = release.get("body")
        if not isinstance(tag, str) or not isinstance(body, str):
            continue
        try:
            version = parse_release_tag(tag, allow_legacy=True)
        except ContractError:
            continue
        if version.channel is not Channel.STABLE or release.get("draft") is not False:
            continue
        if native_release_marker not in body or native_manifest_marker not in body:
            continue
        if (
            tag == LEGACY_MANUAL_QUALIFICATION_RELEASE_TAG
            and _published_manifest_compatibility(tag=tag, provenance=provenance)
            is None
        ):
            continue
        claims: list[str] = []
        for claim in re.findall(
            r"(?m)^Orchestrator correlation: `([^`\r\n]+)`$", body
        ):
            try:
                claims.append(require_correlation_id(claim))
            except ContractError:
                pass
        if len(claims) != 1:
            continue
        order = (*version.version_parts, version.rebuild)
        if latest is None or order > latest[0]:
            latest = (order, release, claims[0])
    return None if latest is None else latest[1:]


def publication_bytes_identical(
    candidate_digests: Mapping[str, str], published_digests: Mapping[str, str]
) -> bool:
    """True when every publication file except ``manifest.json`` has the same
    SHA-256 in both complete inventories.

    ``manifest.json`` embeds the candidate run ID/URL, output tag, correlation
    and bridge commit, so it differs between any two builds; ``sha256sums.txt``
    lists only the artifact digests and is compared.
    """
    expected = set(PUBLICATION_FILES)
    if set(candidate_digests) != expected or set(published_digests) != expected:
        raise ContractError("publication digest inventories must be complete")
    return all(
        candidate_digests[name] == published_digests[name]
        for name in PUBLICATION_FILES
        if name != "manifest.json"
    )


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

    compatibility = _published_manifest_compatibility(
        tag=tag,
        provenance=provenance,
    )
    if compatibility is None:
        manifest, fingerprint = rq.load_candidate(directory)
    else:
        manifest, fingerprint = rq.load_published_candidate(
            directory,
            expected_qualification_gates=compatibility[0],
            expected_unproven_capabilities=compatibility[1],
        )
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
        expected_bridge_source_sha=(
            LEGACY_MANUAL_QUALIFICATION_BRIDGE_COMMIT
            if compatibility is not None
            else None
        ),
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
        directory=directory,
    )
