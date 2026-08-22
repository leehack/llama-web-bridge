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

from generate_release_manifest import ARTIFACTS, CAPABILITIES
from release_contract import (
    ASSETS_REPOSITORY,
    BRIDGE_REPOSITORY,
    NATIVE_REPOSITORY,
    ContractError,
    Transition,
    compare_releases,
    compare_upstream,
    parse_release_tag,
    parse_upstream_tag,
    require_sha256,
)


APPROVED_ASSETS_REPOSITORY = ASSETS_REPOSITORY
PUBLICATION_FILES = (*ARTIFACTS, "manifest.json", "sha256sums.txt")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
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


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"could not read {label}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ContractError(f"{label} root must be a JSON object")
    return payload


def validate_candidate(directory: Path, identity: CandidateIdentity) -> str:
    require_approved_assets_repo(identity.assets_repo)
    release = parse_release_tag(identity.release_tag)
    if release.rebuild != identity.release_rebuild:
        raise ContractError("release_rebuild does not match release_tag")
    upstream = parse_upstream_tag(identity.upstream_tag)
    if release.channel != upstream.channel or release.upstream_parts != upstream.parts:
        raise ContractError("release_tag does not preserve upstream identity")
    _require_commit(identity.bridge_commit, "bridge_commit")
    _require_commit(identity.upstream_commit, "upstream_commit")
    _require_commit(identity.native_commit, "native_commit")
    require_sha256(identity.native_manifest_sha256, "native_manifest_sha256")

    manifest = _read_json(directory / "manifest.json", "candidate manifest")
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


def _previous_identity(repository: Path, commit: str) -> tuple[str, str]:
    raw = _git_file(repository, commit, "manifest.json")
    if raw is None:
        raise ContractError("assets branch is missing manifest.json")
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ContractError(f"assets branch manifest is invalid: {error}") from error
    if not isinstance(manifest, Mapping):
        raise ContractError("assets branch manifest root must be an object")
    release_tag = manifest.get("release_tag", manifest.get("bridge_assets_tag"))
    upstream_tag = manifest.get("upstream_tag", manifest.get("llama_cpp_tag"))
    if not isinstance(release_tag, str) or not isinstance(upstream_tag, str):
        raise ContractError("assets branch manifest is missing release/upstream tags")
    parse_release_tag(release_tag, allow_legacy=True)
    parse_upstream_tag(upstream_tag)
    return release_tag, upstream_tag


def _validate_transition(repository: Path, previous: str, identity: CandidateIdentity) -> None:
    previous_release, previous_upstream = _previous_identity(repository, previous)
    try:
        release_transition = compare_releases(previous_release, identity.release_tag)
    except ContractError as error:
        raise RollbackError(str(error)) from error
    upstream_transition = compare_upstream(previous_upstream, identity.upstream_tag)
    if release_transition not in (Transition.FORWARD, Transition.STABLE_MIGRATION):
        raise RollbackError(f"release transition is {release_transition.value}")
    if upstream_transition in (
        Transition.BACKWARD,
        Transition.FORBIDDEN_STABLE_TO_DEVELOPMENT,
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
    if (
        release.get("tag_name") != identity.release_tag
        or release.get("name") != identity.release_tag
        or release.get("draft") is not False
        or release.get("prerelease") is not expected_prerelease
        or release.get("target_commitish") != tag_commit
        or not isinstance(release.get("id"), int)
        or f"Candidate fingerprint: `{fingerprint}`" not in str(release.get("body", ""))
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
        if (
            asset.get("state") != "uploaded"
            or asset.get("size") != len(data)
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
                        allowed=True,
                        action="upload-release-assets",
                        outcome="safely-resumed",
                        reason_code="exact-release-assets-missing",
                        reason="release metadata and present assets match; assets are missing",
                        retryable=True,
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
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    target = subparsers.add_parser("validate-target")
    target.add_argument("--assets-repo", required=True)

    changed = subparsers.add_parser("state-changed")
    changed.add_argument("--before-json", required=True, type=Path)
    changed.add_argument("--after-json", required=True, type=Path)

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
    return parser


def _fatal(reason: str) -> dict[str, object]:
    return {
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


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "validate-target":
            require_approved_assets_repo(args.assets_repo)
            print(json.dumps({"allowed": True, "assets_repository": args.assets_repo}))
            return 0
        if args.command == "state-changed":
            before = _read_json(args.before_json, "before publication state")
            after = _read_json(args.after_json, "after publication state")
            print(json.dumps(publication_state_changed(before, after)))
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
        result = _fatal(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("allowed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
