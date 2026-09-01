#!/usr/bin/env python3
"""Classify exact bridge publication state for idempotent, fail-closed retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from generate_release_manifest import (
    ARTIFACTS,
    CAPABILITIES,
    QUALIFICATION_GATES,
    UNPROVEN_CAPABILITIES,
)
from release_contract import (
    ASSETS_REPOSITORY,
    BRIDGE_REPOSITORY,
    IMMUTABLE_RELEASE_ATTESTATION_PREDICATE_TYPE,
    IMMUTABLE_RELEASE_ATTESTATION_SIGNER,
    NATIVE_REPOSITORY,
    ContractError,
    Transition,
    compare_releases,
    compare_upstream,
    parse_release_tag,
    parse_upstream_tag,
    require_correlation_id,
    require_sha256,
    validate_native_identity,
    validate_release_attestation,
    validate_release_immutability,
)


APPROVED_ASSETS_REPOSITORY = ASSETS_REPOSITORY
PUBLICATION_FILES = (*ARTIFACTS, "manifest.json", "sha256sums.txt")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_RUN_ID_RE = re.compile(r"[1-9][0-9]*")
_SUM_RE = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)")


class RollbackError(ContractError):
    """Raised when a candidate violates release or upstream ordering."""


@dataclass(frozen=True)
class CandidateIdentity:
    release_tag: str
    release_rebuild: int
    assets_repo: str
    bridge_commit: str
    upstream_tag: str
    upstream_commit: str
    native_release_tag: str
    native_manifest_sha256: str
    native_commit: str
    emscripten_version: str
    orchestrator_correlation_id: str
    github_run_id: str
    github_run_url: str


def require_approved_assets_repo(value: str) -> str:
    if value != APPROVED_ASSETS_REPOSITORY:
        raise ContractError(
            f"assets repository {value!r} is not approved; expected "
            f"{APPROVED_ASSETS_REPOSITORY!r}"
        )
    return value


def _require_commit(value: str, field: str) -> str:
    if _COMMIT_RE.fullmatch(value) is None:
        raise ContractError(f"{field} must be a lowercase full commit SHA")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"could not read {label}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ContractError(f"{label} root must be a JSON object")
    return payload


def validate_candidate(
    directory: Path,
    identity: CandidateIdentity,
    *,
    expected_qualification_gates: Mapping[str, str] | None = None,
    expected_unproven_capabilities: Mapping[str, str] | None = None,
) -> str:
    """Validate candidate bytes against the current or an explicit historical contract.

    Historical expectations are only for immutable published-release readback.
    Candidate generation, qualification, and publication must omit them.
    """

    if not directory.is_dir() or directory.is_symlink():
        raise ContractError("candidate must be a real directory")
    entries = list(directory.iterdir())
    actual_names = {entry.name for entry in entries}
    if actual_names != set(PUBLICATION_FILES):
        unexpected = sorted(actual_names - set(PUBLICATION_FILES))
        missing = sorted(set(PUBLICATION_FILES) - actual_names)
        raise ContractError(
            "candidate directory must contain exactly the governed publication "
            f"files (unexpected: {unexpected}, missing: {missing})"
        )
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ContractError(
                f"candidate entry must be an immutable regular file: {entry.name}"
            )

    require_approved_assets_repo(identity.assets_repo)
    release = parse_release_tag(identity.release_tag)
    if release.rebuild != identity.release_rebuild:
        raise ContractError("release_rebuild does not match release_tag")
    # Bridge asset versions are independent of llama.cpp versions; both tags are
    # still syntactically exact and are recorded separately in the manifest.
    parse_upstream_tag(identity.upstream_tag)
    native_release = parse_release_tag(identity.native_release_tag)
    validate_native_identity(
        identity.native_release_tag,
        native_release.rebuild,
        identity.upstream_tag,
    )
    _require_commit(identity.bridge_commit, "bridge_commit")
    _require_commit(identity.upstream_commit, "upstream_commit")
    _require_commit(identity.native_commit, "native_commit")
    require_sha256(identity.native_manifest_sha256, "native_manifest_sha256")
    require_correlation_id(identity.orchestrator_correlation_id)
    if (
        not isinstance(identity.github_run_id, str)
        or _RUN_ID_RE.fullmatch(identity.github_run_id) is None
    ):
        raise ContractError("github_run_id must be a positive decimal string")
    expected_run_url = (
        f"https://github.com/{BRIDGE_REPOSITORY}/actions/runs/{identity.github_run_id}"
    )
    if identity.github_run_url != expected_run_url:
        raise ContractError(f"github_run_url must be exactly {expected_run_url}")

    manifest = _read_json(directory / "manifest.json", "candidate manifest")
    qualification_gates = (
        QUALIFICATION_GATES
        if expected_qualification_gates is None
        else dict(expected_qualification_gates)
    )
    unproven_capabilities = (
        UNPROVEN_CAPABILITIES
        if expected_unproven_capabilities is None
        else dict(expected_unproven_capabilities)
    )
    expected_fields: dict[str, object] = {
        "schema_version": 2,
        "release_tag": identity.release_tag,
        "release_channel": release.channel.value,
        "release_rebuild": identity.release_rebuild,
        "assets_repository": ASSETS_REPOSITORY,
        "bridge_repository": BRIDGE_REPOSITORY,
        "bridge_commit": identity.bridge_commit,
        "upstream_repository": "ggml-org/llama.cpp",
        "upstream_tag": identity.upstream_tag,
        "upstream_commit": identity.upstream_commit,
        "native_repository": NATIVE_REPOSITORY,
        "native_release_tag": identity.native_release_tag,
        "native_manifest_sha256": identity.native_manifest_sha256,
        "native_commit": identity.native_commit,
        "emscripten_version": identity.emscripten_version,
        "orchestrator_correlation_id": identity.orchestrator_correlation_id,
        "github_run_id": identity.github_run_id,
        "github_run_url": identity.github_run_url,
        "qualification_gates": qualification_gates,
        "unproven_capabilities": unproven_capabilities,
        "capabilities": CAPABILITIES,
        "bridge_assets_tag": identity.release_tag,
        "source_repository": BRIDGE_REPOSITORY,
        "source_commit": identity.bridge_commit,
        "llama_cpp_tag": identity.upstream_tag,
        "llama_cpp_commit": identity.upstream_commit,
    }
    if set(manifest) != {*expected_fields, "artifacts", "files"}:
        raise ContractError("candidate manifest schema has missing or unexpected fields")
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            raise ContractError(
                f"candidate manifest {field} mismatch: expected {expected!r}, "
                f"got {manifest.get(field)!r}"
            )

    try:
        sum_lines = (directory / "sha256sums.txt").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ContractError(f"could not read candidate checksums: {error}") from error
    sums: dict[str, str] = {}
    for line in sum_lines:
        match = _SUM_RE.fullmatch(line)
        if match is None or match.group(2) in sums:
            raise ContractError(f"invalid or duplicate checksum line: {line!r}")
        sums[match.group(2)] = match.group(1)
    if set(sums) != set(ARTIFACTS):
        raise ContractError("sha256sums.txt must contain exactly the release artifacts")

    artifact_manifest = manifest.get("artifacts")
    legacy_files = manifest.get("files")
    if not isinstance(artifact_manifest, Mapping) or artifact_manifest != legacy_files:
        raise ContractError("manifest artifacts/files maps must be identical objects")
    if set(artifact_manifest) != set(ARTIFACTS):
        raise ContractError("manifest artifact map must contain exactly release artifacts")
    for name in ARTIFACTS:
        path = directory / name
        if not path.is_file():
            raise ContractError(f"candidate artifact is missing: {name}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        metadata = artifact_manifest.get(name)
        if not isinstance(metadata, Mapping) or set(metadata) != {"sha256", "size_bytes"}:
            raise ContractError(f"manifest metadata schema mismatch for {name}")
        if sums[name] != digest or metadata.get("sha256") != digest:
            raise ContractError(f"artifact checksum mismatch for {name}")
        if metadata.get("size_bytes") != len(data):
            raise ContractError(f"artifact size mismatch for {name}")

    fingerprint = hashlib.sha256()
    for name in sorted(PUBLICATION_FILES):
        data = (directory / name).read_bytes()
        fingerprint.update(name.encode("utf-8") + b"\0")
        fingerprint.update(len(data).to_bytes(8, "big"))
        fingerprint.update(data)
    return fingerprint.hexdigest()


def _git(repository: Path, *args: str, text: bool = False) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )


def _git_file(repository: Path, commit: str, name: str) -> bytes | None:
    result = _git(repository, "show", f"{commit}:{name}")
    return result.stdout if result.returncode == 0 else None


def _snapshot_matches(repository: Path, commit: str, candidate: Path) -> bool:
    _require_commit(commit, "snapshot commit")
    return all(
        (existing := _git_file(repository, commit, name)) is not None
        and existing == (candidate / name).read_bytes()
        for name in PUBLICATION_FILES
    )


def _is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    return _git(repository, "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0


def _manifest_history_identity(
    repository: Path, commit: str
) -> tuple[str, str] | None:
    raw = _git_file(repository, commit, "manifest.json")
    if raw is None:
        return None
    try:
        manifest = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ContractError(f"assets branch manifest is invalid: {error}") from error
    if not isinstance(manifest, Mapping):
        raise ContractError("assets branch manifest root must be an object")
    release_tag = manifest.get("release_tag")
    legacy_release_tag = manifest.get("bridge_assets_tag")
    if release_tag is None:
        release_tag = legacy_release_tag
    elif legacy_release_tag is not None and legacy_release_tag != release_tag:
        raise ContractError("assets branch manifest release tag aliases conflict")
    upstream_tag = manifest.get("upstream_tag")
    legacy_upstream_tag = manifest.get("llama_cpp_tag")
    if upstream_tag is None:
        upstream_tag = legacy_upstream_tag
    elif legacy_upstream_tag is not None and legacy_upstream_tag != upstream_tag:
        raise ContractError("assets branch manifest upstream tag aliases conflict")
    if not isinstance(release_tag, str) or not isinstance(upstream_tag, str):
        raise ContractError("assets branch manifest is missing release/upstream tags")
    # Both identities are read exactly as recorded. Bridge asset versions have
    # always been independent of the llama.cpp line, in every schema, so neither
    # tag may be fabricated from the other.
    release = parse_release_tag(release_tag, allow_legacy=True)
    upstream = parse_upstream_tag(upstream_tag)
    return release.tag, upstream.tag


def _history_identities(repository: Path, commit: str) -> list[tuple[str, str]]:
    """Read every recorded (release_tag, upstream_tag) newest-first."""
    history = _git(repository, "rev-list", "--first-parent", commit, text=True)
    if history.returncode != 0:
        raise ContractError("could not inspect assets branch channel history")
    identities: list[tuple[str, str]] = []
    for index, history_commit in enumerate(history.stdout.splitlines()):
        previous = _manifest_history_identity(repository, history_commit)
        if previous is None:
            if index == 0:
                raise ContractError("assets branch is missing manifest.json")
            continue
        identities.append(previous)
    return identities


def _validate_transition(repository: Path, previous: str, identity: CandidateIdentity) -> None:
    """Order the asset release and upstream lines independently, both fail-closed.

    Asset release tags are ordered within their own channel, because stable and
    development asset histories advance independently. The upstream llama.cpp
    line is a single global line, so it is ordered against the newest recorded
    entry regardless of which asset channel published it.
    """
    candidate = parse_release_tag(identity.release_tag)
    history = _history_identities(repository, previous)
    previous_identity = next(
        (
            entry
            for entry in history
            if parse_release_tag(entry[0], allow_legacy=True).channel == candidate.channel
        ),
        None,
    )

    if previous_identity is None:
        if candidate.rebuild != 0:
            raise RollbackError(
                "the first artifact in a release channel must use rebuild 0"
            )
    else:
        previous_release, _ = previous_identity
        try:
            release_transition = compare_releases(previous_release, identity.release_tag)
        except ContractError as error:
            raise RollbackError(str(error)) from error
        if release_transition == Transition.EQUAL:
            raise ContractError(
                f"release tag {identity.release_tag!r} already exists in channel history"
            )
        if release_transition != Transition.FORWARD:
            raise RollbackError(f"release transition is {release_transition.value}")

    if history:
        _, latest_upstream = history[0]
        upstream_transition = compare_upstream(latest_upstream, identity.upstream_tag)
        # A development-to-stable upstream migration is a legal advance; only
        # backward and stable-to-development moves are rollbacks.
        if upstream_transition not in (
            Transition.EQUAL,
            Transition.FORWARD,
            Transition.STABLE_MIGRATION,
        ):
            raise RollbackError(f"upstream transition is {upstream_transition.value}")


def _validate_candidate_commit(
    repository: Path, commit: str, candidate: Path, identity: CandidateIdentity
) -> None:
    if not _snapshot_matches(repository, commit, candidate):
        raise ContractError("publication commit content differs from the candidate")
    parents = _git(repository, "rev-list", "--parents", "-n", "1", commit, text=True)
    if parents.returncode != 0:
        raise ContractError("could not inspect publication commit parent")
    fields = parents.stdout.strip().split()
    if len(fields) != 2:
        raise ContractError("publication commit must have exactly one parent")
    parent = fields[1]
    changed = _git(
        repository,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
        text=True,
    )
    changed_names = set(changed.stdout.splitlines())
    governed_names = set(PUBLICATION_FILES)
    if (
        changed.returncode != 0
        or "manifest.json" not in changed_names
        or not changed_names <= governed_names
    ):
        raise ContractError(
            "publication commit must change the manifest and only governed release files"
        )
    _validate_transition(repository, parent, identity)


def _validate_release(
    release: Mapping[str, Any],
    identity: CandidateIdentity,
    tag_commit: str,
    candidate: Path,
    fingerprint: str,
) -> list[str]:
    expected_prerelease = parse_release_tag(identity.release_tag).github_prerelease
    release_id = release.get("id")
    if (
        release.get("tag_name") != identity.release_tag
        or release.get("name") != identity.release_tag
        or release.get("draft") is not False
        or release.get("prerelease") is not expected_prerelease
        or release.get("target_commitish") != tag_commit
        or not isinstance(release_id, int)
        or isinstance(release_id, bool)
        or release_id <= 0
        or f"Candidate fingerprint: `{fingerprint}`" not in str(release.get("body", ""))
        or f"Orchestrator correlation: `{identity.orchestrator_correlation_id}`"
        not in str(release.get("body", ""))
        or identity.github_run_url not in str(release.get("body", ""))
    ):
        raise ContractError("GitHub Release metadata does not match the immutable candidate")
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise ContractError("GitHub Release asset inventory is missing")
    actual: dict[str, Mapping[str, Any]] = {}
    for asset in assets:
        if not isinstance(asset, Mapping) or not isinstance(asset.get("name"), str):
            raise ContractError("GitHub Release contains an invalid asset record")
        name = asset["name"]
        if name in actual:
            raise ContractError(f"GitHub Release contains duplicate asset {name!r}")
        actual[name] = asset
    unexpected = set(actual) - set(PUBLICATION_FILES)
    if unexpected:
        raise ContractError("GitHub Release asset inventory contains unexpected assets")
    for name in actual:
        data = (candidate / name).read_bytes()
        asset = actual[name]
        asset_size = asset.get("size")
        if (
            asset.get("state") != "uploaded"
            or not isinstance(asset_size, int)
            or isinstance(asset_size, bool)
            or asset_size != len(data)
            or asset.get("digest") != f"sha256:{hashlib.sha256(data).hexdigest()}"
        ):
            raise ContractError(f"GitHub Release asset digest/size mismatch for {name}")
    return sorted(set(PUBLICATION_FILES) - set(actual))


def _result(
    *,
    identity: CandidateIdentity,
    state: str,
    allowed: bool,
    action: str,
    outcome: str,
    reason_code: str,
    fingerprint: str,
    branch_commit: str,
    tag_commit: str | None,
    release: Mapping[str, Any] | None,
    reason: str,
    retryable: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "state": state,
        "allowed": allowed,
        "action": action,
        "outcome": outcome,
        "reason_code": reason_code,
        "reason": reason,
        "retryable": retryable,
        "mutated": False,
        "candidate_fingerprint": fingerprint,
        "assets_repository": identity.assets_repo,
        "release_tag": identity.release_tag,
        "branch_commit": branch_commit,
        "tag_commit": tag_commit,
        "release_id": release.get("id") if release else None,
        "orchestrator_correlation_id": identity.orchestrator_correlation_id,
        "github_run_id": identity.github_run_id,
        "github_run_url": identity.github_run_url,
        "qualification_gates": QUALIFICATION_GATES,
    }


def publication_state_changed(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> bool:
    """Report whether verified remote publication identity changed during this run."""
    def identity(state: Mapping[str, Any]) -> tuple[object, object, object, tuple[str, ...]]:
        missing = state.get("missing_release_assets", [])
        if not isinstance(missing, list) or not all(
            isinstance(name, str) for name in missing
        ):
            raise ContractError("publication state has invalid missing_release_assets")
        return (
            state.get("branch_commit"),
            state.get("tag_commit"),
            state.get("release_id"),
            tuple(sorted(missing)),
        )

    return identity(before) != identity(after)


def mutation_unknown_outcome(
    candidate: Path, identity: CandidateIdentity, reason_code: str
) -> dict[str, object]:
    """Emit a durable retry contract when a credentialed mutation cannot be re-read."""
    if reason_code not in {"ref-requery-failed", "release-requery-failed"}:
        raise ContractError("unsupported mutation-unknown reason_code")
    fingerprint = validate_candidate(candidate, identity)
    return {
        "schema_version": 1,
        "state": "mutation-unknown",
        "allowed": False,
        "action": "none",
        "outcome": "mutation-unknown",
        "reason_code": reason_code,
        "reason": "a credentialed mutation was attempted but exact remote state could not be re-read",
        "retryable": True,
        "mutated": None,
        "mutation_status": "unknown",
        "candidate_fingerprint": fingerprint,
        "assets_repository": identity.assets_repo,
        "release_tag": identity.release_tag,
        "branch_commit": None,
        "tag_commit": None,
        "release_id": None,
        "orchestrator_correlation_id": identity.orchestrator_correlation_id,
        "github_run_id": identity.github_run_id,
        "github_run_url": identity.github_run_url,
        "qualification_gates": QUALIFICATION_GATES,
    }


def mutation_unknown_from_requery(
    candidate: Path,
    identity: CandidateIdentity,
    reason_code: str,
    requery_path: Path,
) -> dict[str, object]:
    """Convert only an unavailable or semantically invalid re-query to unknown."""
    try:
        data = requery_path.read_bytes()
    except OSError:
        data = b""
    if data.strip():
        try:
            parsed = json.loads(data, object_pairs_hook=_reject_duplicate_keys)
        except (ContractError, UnicodeDecodeError, json.JSONDecodeError):
            parsed = None
        if (
            isinstance(parsed, Mapping)
            and parsed.get("reason_code") != "invalid-input-or-state"
        ):
            raise ContractError(
                "valid classifier JSON must be handled as exact state, not mutation-unknown"
            )
    return mutation_unknown_outcome(candidate, identity, reason_code)


def classify(
    *,
    repository: Path,
    candidate: Path,
    identity: CandidateIdentity,
    branch_commit: str,
    tag_commit: str | None,
    release: Mapping[str, Any] | None,
) -> dict[str, object]:
    fingerprint = validate_candidate(candidate, identity)
    _require_commit(branch_commit, "branch_commit")
    if tag_commit is not None:
        _require_commit(tag_commit, "tag_commit")

    def result(**kwargs: Any) -> dict[str, object]:
        return _result(
            identity=identity,
            fingerprint=fingerprint,
            branch_commit=branch_commit,
            tag_commit=tag_commit,
            release=release,
            **kwargs,
        )

    try:
        if release is not None and tag_commit is None:
            raise ContractError("GitHub Release exists without the immutable git tag")
        if tag_commit is not None:
            if not _is_ancestor(repository, tag_commit, branch_commit):
                raise ContractError("existing tag is not reachable from the assets branch")
            _validate_candidate_commit(repository, tag_commit, candidate, identity)
            if release is not None:
                missing_assets = _validate_release(
                    release, identity, tag_commit, candidate, fingerprint
                )
                if missing_assets:
                    partial = result(
                        state="release-assets-partial",
                        allowed=False,
                        action="none",
                        outcome="immutable-publication-unverified",
                        reason_code="immutable-release-assets-missing",
                        reason=(
                            "published release asset inventory is incomplete; immutable "
                            "releases must never be repaired or overwritten"
                        ),
                    )
                    partial["missing_release_assets"] = missing_assets
                    return partial
                return result(
                    state="complete",
                    allowed=True,
                    action="none",
                    outcome="already-complete",
                    reason_code="exact-release-complete",
                    reason="tag, release metadata, and every asset exactly match",
                )
            return result(
                state="tag-without-release",
                allowed=True,
                action="create-release",
                outcome="safely-resumed",
                reason_code="exact-tag-release-missing",
                reason="exact reachable tag exists and only GitHub Release is missing",
                retryable=True,
            )

        if _snapshot_matches(repository, branch_commit, candidate):
            _validate_candidate_commit(repository, branch_commit, candidate, identity)
            return result(
                state="branch-only",
                allowed=True,
                action="publish-tag-and-release",
                outcome="safely-resumed",
                reason_code="exact-branch-tag-release-missing",
                reason="exact governed publication commit is on the branch",
                retryable=True,
            )

        _validate_transition(repository, branch_commit, identity)
        return result(
            state="absent",
            allowed=True,
            action="publish-refs-and-release",
            outcome="newly-published",
            reason_code="new-publication",
            reason="candidate legally advances the current assets branch",
        )
    except ContractError as error:
        reason = str(error)
        rollback = isinstance(error, RollbackError)
        return result(
            state="rollback" if rollback else "collision",
            allowed=False,
            action="none",
            outcome="rollback" if rollback else "collision",
            reason_code="ordering-rollback" if rollback else "identity-collision",
            reason=reason,
        )


def candidate_publication_digests(candidate: Path) -> dict[str, str]:
    if not candidate.is_dir() or candidate.is_symlink():
        raise ContractError("candidate must be a real directory")
    entries = list(candidate.iterdir())
    if {entry.name for entry in entries} != set(PUBLICATION_FILES):
        raise ContractError(
            "candidate directory must contain exactly the governed publication files"
        )
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ContractError(
                f"candidate entry must be an immutable regular file: {entry.name}"
            )
    return {
        name: hashlib.sha256((candidate / name).read_bytes()).hexdigest()
        for name in PUBLICATION_FILES
    }


def verify_immutable_publication(
    *,
    candidate: Path,
    assets_repo: str,
    release_tag: str,
    tag_commit: str,
    release_id: int | None,
    release_by_tag: Mapping[str, Any],
    release_by_id: Mapping[str, Any],
    attestation: Mapping[str, Any],
) -> dict[str, object]:
    """Prove a just-published release is immutable and attested, or fail closed.

    Both readbacks are independent reads of the same release -- by tag and by
    ID -- so a tag that silently resolves elsewhere cannot satisfy the gate.
    Nothing here mutates, deletes, retags, or repairs remote state: a mismatch
    is reported and the release is left exactly as GitHub created it.
    """
    require_approved_assets_repo(assets_repo)
    _require_commit(tag_commit, "tag_commit")
    digests = candidate_publication_digests(candidate)
    resolved_id = validate_release_immutability(
        release_by_tag,
        release_tag=release_tag,
        tag_commit=tag_commit,
        release_id=release_id,
    )
    if validate_release_immutability(
        release_by_id,
        release_tag=release_tag,
        tag_commit=tag_commit,
        release_id=resolved_id,
    ) != resolved_id:
        raise ContractError("release readbacks by tag and by ID identify different releases")
    if release_by_tag.get("published_at") != release_by_id.get("published_at"):
        raise ContractError("release readbacks disagree on published_at")
    verified = validate_release_attestation(
        attestation,
        assets_repo=assets_repo,
        release_tag=release_tag,
        tag_commit=tag_commit,
        release_id=resolved_id,
        expected_assets=digests,
    )
    return {
        "schema_version": 1,
        "assets_repository": assets_repo,
        "release_tag": release_tag,
        "release_id": resolved_id,
        "tag_commit": tag_commit,
        "immutable": True,
        "published_at": release_by_tag["published_at"],
        "attestation_predicate_type": IMMUTABLE_RELEASE_ATTESTATION_PREDICATE_TYPE,
        "attestation_signer": IMMUTABLE_RELEASE_ATTESTATION_SIGNER,
        "attested_purl": verified["purl"],
        "verified_timestamps": verified["verified_timestamps"],
        "attested_assets": verified["assets"],
    }


def _identity(args: argparse.Namespace) -> CandidateIdentity:
    return CandidateIdentity(
        release_tag=args.release_tag,
        release_rebuild=args.release_rebuild,
        assets_repo=args.assets_repo,
        bridge_commit=args.bridge_commit,
        upstream_tag=args.upstream_tag,
        upstream_commit=args.upstream_commit,
        native_release_tag=args.native_release_tag,
        native_manifest_sha256=args.native_manifest_sha256,
        native_commit=args.native_commit,
        emscripten_version=args.emscripten_version,
        orchestrator_correlation_id=args.orchestrator_correlation_id,
        github_run_id=args.github_run_id,
        github_run_url=args.github_run_url,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    target = subparsers.add_parser("validate-target")
    target.add_argument("--assets-repo", required=True)

    immutable = subparsers.add_parser("verify-immutable-publication")
    immutable.add_argument("--candidate", required=True, type=Path)
    immutable.add_argument("--assets-repo", required=True)
    immutable.add_argument("--release-tag", required=True)
    immutable.add_argument("--tag-commit", required=True)
    immutable.add_argument("--release-id", type=int)
    immutable.add_argument("--release-json", required=True, type=Path)
    immutable.add_argument("--release-by-id-json", required=True, type=Path)
    immutable.add_argument("--attestation-json", required=True, type=Path)

    changed = subparsers.add_parser("state-changed")
    changed.add_argument("--before-json", required=True, type=Path)
    changed.add_argument("--after-json", required=True, type=Path)

    unknown = subparsers.add_parser("mutation-unknown")
    unknown.add_argument("--candidate", required=True, type=Path)
    unknown.add_argument("--reason-code", required=True)
    unknown.add_argument("--requery-json", required=True, type=Path)
    unknown.add_argument("--release-tag", required=True)
    unknown.add_argument("--release-rebuild", required=True, type=int)
    unknown.add_argument("--assets-repo", required=True)
    unknown.add_argument("--bridge-commit", required=True)
    unknown.add_argument("--upstream-tag", required=True)
    unknown.add_argument("--upstream-commit", required=True)
    unknown.add_argument("--native-release-tag", required=True)
    unknown.add_argument("--native-manifest-sha256", required=True)
    unknown.add_argument("--native-commit", required=True)
    unknown.add_argument("--emscripten-version", required=True)
    unknown.add_argument("--orchestrator-correlation-id", required=True)
    unknown.add_argument("--github-run-id", required=True)
    unknown.add_argument("--github-run-url", required=True)

    inspect = subparsers.add_parser("classify")
    inspect.add_argument("--repository", required=True, type=Path)
    inspect.add_argument("--candidate", required=True, type=Path)
    inspect.add_argument("--branch-commit", required=True)
    inspect.add_argument("--tag-commit")
    inspect.add_argument("--release-json", type=Path)
    inspect.add_argument("--release-tag", required=True)
    inspect.add_argument("--release-rebuild", required=True, type=int)
    inspect.add_argument("--assets-repo", required=True)
    inspect.add_argument("--bridge-commit", required=True)
    inspect.add_argument("--upstream-tag", required=True)
    inspect.add_argument("--upstream-commit", required=True)
    inspect.add_argument("--native-release-tag", required=True)
    inspect.add_argument("--native-manifest-sha256", required=True)
    inspect.add_argument("--native-commit", required=True)
    inspect.add_argument("--emscripten-version", required=True)
    inspect.add_argument("--orchestrator-correlation-id", required=True)
    inspect.add_argument("--github-run-id", required=True)
    inspect.add_argument("--github-run-url", required=True)
    return parser


def _fatal(
    reason: str, identity: CandidateIdentity | None = None
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": 1,
        "state": "collision",
        "allowed": False,
        "action": "none",
        "outcome": "collision",
        "reason_code": "invalid-input-or-state",
        "reason": reason,
        "retryable": False,
        "mutated": False,
    }
    if identity is not None:
        result.update(
            {
                "assets_repository": identity.assets_repo,
                "release_tag": identity.release_tag,
                "orchestrator_correlation_id": identity.orchestrator_correlation_id,
                "github_run_id": identity.github_run_id,
                "github_run_url": identity.github_run_url,
                "qualification_gates": QUALIFICATION_GATES,
            }
        )
    return result


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "validate-target":
            require_approved_assets_repo(args.assets_repo)
            print(json.dumps({"allowed": True, "assets_repository": args.assets_repo}))
            return 0
        if args.command == "verify-immutable-publication":
            # This gate reports immutability, never publication state, so it
            # exits with a plain error instead of a classifier outcome.
            try:
                verified = verify_immutable_publication(
                    candidate=args.candidate,
                    assets_repo=args.assets_repo,
                    release_tag=args.release_tag,
                    tag_commit=args.tag_commit,
                    release_id=args.release_id,
                    release_by_tag=_read_json(args.release_json, "published GitHub Release"),
                    release_by_id=_read_json(
                        args.release_by_id_json, "published GitHub Release by ID"
                    ),
                    attestation=_read_json(
                        args.attestation_json, "GitHub release attestation"
                    ),
                )
            except (ContractError, OSError) as error:
                raise SystemExit(f"error: {error}") from error
            print(json.dumps(verified, sort_keys=True))
            return 0
        if args.command == "state-changed":
            before = _read_json(args.before_json, "before publication state")
            after = _read_json(args.after_json, "after publication state")
            print(json.dumps(publication_state_changed(before, after)))
            return 0
        if args.command == "mutation-unknown":
            print(json.dumps(
                mutation_unknown_from_requery(
                    args.candidate,
                    _identity(args),
                    args.reason_code,
                    args.requery_json,
                ),
                sort_keys=True,
            ))
            return 0
        release = _read_json(args.release_json, "GitHub Release") if args.release_json else None
        result = classify(
            repository=args.repository,
            candidate=args.candidate,
            identity=_identity(args),
            branch_commit=args.branch_commit,
            tag_commit=args.tag_commit or None,
            release=release,
        )
    except (ContractError, OSError) as error:
        identity = (
            _identity(args)
            if args.command in {"classify", "mutation-unknown"}
            else None
        )
        result = _fatal(str(error), identity)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("allowed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
