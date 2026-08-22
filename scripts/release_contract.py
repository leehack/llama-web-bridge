#!/usr/bin/env python3
"""Release identity and ordering contract for Web bridge asset publication.

GitHub release tags are shared with the native release convention. Stable
releases use ``vMAJOR.MINOR.PATCH`` and rebuilds append ``-N``. Development
releases use ``bNNNN`` and rebuilds append ``-N``. Historical
``*-llamadart.N`` wrappers remain readable so an existing manifest can be used
as an ordering boundary, but this module never emits them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class ContractError(ValueError):
    """Raised when release provenance is ambiguous or unsafe."""


class Channel(str, Enum):
    DEVELOPMENT = "development"
    STABLE = "stable"


class Transition(str, Enum):
    EQUAL = "equal"
    FORWARD = "forward"
    BACKWARD = "backward"
    STABLE_MIGRATION = "stable-migration"
    FORBIDDEN_STABLE_TO_DEVELOPMENT = "forbidden-stable-to-development"


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_UPSTREAM_STABLE_RE = re.compile(
    r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
)
_UPSTREAM_DEVELOPMENT_RE = re.compile(r"b(0|[1-9][0-9]*)")
_STABLE_RELEASE_RE = re.compile(
    r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([1-9][0-9]*))?"
)
_DEVELOPMENT_RELEASE_RE = re.compile(
    r"b(0|[1-9][0-9]*)(?:-([1-9][0-9]*))?"
)
_LEGACY_STABLE_RE = re.compile(
    r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)-llamadart\.([1-9][0-9]*)"
)
_LEGACY_DEVELOPMENT_RE = re.compile(
    r"b(0|[1-9][0-9]*)-llamadart\.([1-9][0-9]*)"
)


@dataclass(frozen=True)
class UpstreamVersion:
    tag: str
    channel: Channel
    parts: tuple[int, ...]


@dataclass(frozen=True)
class ReleaseVersion:
    tag: str
    channel: Channel
    upstream_parts: tuple[int, ...]
    rebuild: int
    legacy: bool = False

    @property
    def upstream_tag(self) -> str:
        if self.channel == Channel.DEVELOPMENT:
            return f"b{self.upstream_parts[0]}"
        return "v" + ".".join(str(part) for part in self.upstream_parts)


@dataclass(frozen=True)
class NativeIdentity:
    release_tag: str
    upstream_tag: str
    upstream_commit: str
    native_commit: str


def _require_commit(value: Any, field: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ContractError(f"{field} must be a lowercase full 40-character commit SHA")
    return value


def require_sha256(value: str, field: str = "sha256") -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def require_repository(value: str, field: str) -> str:
    if _REPO_RE.fullmatch(value) is None:
        raise ContractError(f"{field} must use owner/repository syntax")
    return value


def parse_upstream_tag(tag: str) -> UpstreamVersion:
    development = _UPSTREAM_DEVELOPMENT_RE.fullmatch(tag)
    if development:
        return UpstreamVersion(tag, Channel.DEVELOPMENT, (int(development.group(1)),))

    stable = _UPSTREAM_STABLE_RE.fullmatch(tag)
    if stable:
        return UpstreamVersion(
            tag, Channel.STABLE, tuple(int(part) for part in stable.groups())
        )

    raise ContractError(
        f"unsupported upstream tag {tag!r}; expected exact vMAJOR.MINOR.PATCH or bNNNN"
    )


def parse_release_tag(tag: str, *, allow_legacy: bool = False) -> ReleaseVersion:
    stable = _STABLE_RELEASE_RE.fullmatch(tag)
    if stable:
        major, minor, patch = (int(part) for part in stable.groups()[:3])
        rebuild = int(stable.group(4) or 0)
        return ReleaseVersion(tag, Channel.STABLE, (major, minor, patch), rebuild)

    development = _DEVELOPMENT_RELEASE_RE.fullmatch(tag)
    if development:
        return ReleaseVersion(
            tag,
            Channel.DEVELOPMENT,
            (int(development.group(1)),),
            int(development.group(2) or 0),
        )

    if allow_legacy:
        legacy_stable = _LEGACY_STABLE_RE.fullmatch(tag)
        if legacy_stable:
            major, minor, patch, rebuild = map(int, legacy_stable.groups())
            return ReleaseVersion(
                tag, Channel.STABLE, (major, minor, patch), rebuild, legacy=True
            )
        legacy_development = _LEGACY_DEVELOPMENT_RE.fullmatch(tag)
        if legacy_development:
            build, rebuild = map(int, legacy_development.groups())
            return ReleaseVersion(
                tag, Channel.DEVELOPMENT, (build,), rebuild, legacy=True
            )

    suffix = " (legacy *-llamadart.N is read-only)" if not allow_legacy else ""
    raise ContractError(
        f"unsupported release tag {tag!r}; expected vMAJOR.MINOR.PATCH[-N] or bNNNN[-N]{suffix}"
    )


def compare_upstream(current_tag: str, target_tag: str) -> Transition:
    current = parse_upstream_tag(current_tag)
    target = parse_upstream_tag(target_tag)
    if current.tag == target.tag:
        return Transition.EQUAL
    if current.channel == target.channel:
        return Transition.FORWARD if target.parts > current.parts else Transition.BACKWARD
    if current.channel == Channel.DEVELOPMENT and target.channel == Channel.STABLE:
        return Transition.STABLE_MIGRATION
    return Transition.FORBIDDEN_STABLE_TO_DEVELOPMENT


def compare_releases(current_tag: str, target_tag: str) -> Transition:
    current = parse_release_tag(current_tag, allow_legacy=True)
    target = parse_release_tag(target_tag)

    if current.upstream_tag == target.upstream_tag:
        if current.rebuild == target.rebuild:
            return Transition.EQUAL
        return Transition.FORWARD if target.rebuild > current.rebuild else Transition.BACKWARD

    upstream_transition = compare_upstream(current.upstream_tag, target.upstream_tag)
    if upstream_transition in (Transition.FORWARD, Transition.STABLE_MIGRATION):
        if target.rebuild != 0:
            raise ContractError(
                "the first artifact for a new upstream version must use rebuild 0"
            )
        return upstream_transition
    return upstream_transition


def validate_release_identity(release_tag: str, rebuild: int, upstream_tag: str) -> ReleaseVersion:
    if rebuild < 0:
        raise ContractError("release_rebuild must be zero or greater")
    release = parse_release_tag(release_tag)
    upstream = parse_upstream_tag(upstream_tag)
    if release.rebuild != rebuild:
        raise ContractError(
            f"release tag {release_tag!r} encodes rebuild {release.rebuild}, not {rebuild}"
        )
    if release.channel != upstream.channel or release.upstream_parts != upstream.parts:
        raise ContractError(
            f"release tag {release_tag!r} does not preserve upstream identity {upstream_tag!r}"
        )
    return release


def resolve_native_manifest(
    manifest: Mapping[str, Any], release_tag: str
) -> NativeIdentity:
    legacy_tag = manifest.get("tag")
    native_tag = manifest.get("native_release_tag", legacy_tag)
    if not isinstance(native_tag, str) or not native_tag:
        raise ContractError("native manifest is missing native_release_tag/tag")
    if legacy_tag is not None and legacy_tag != native_tag:
        raise ContractError("native_release_tag does not match the legacy tag alias")
    if native_tag != release_tag:
        raise ContractError(
            f"native manifest tag {native_tag!r} does not match release {release_tag!r}"
        )

    native_version = parse_release_tag(native_tag, allow_legacy=True)
    upstream_tag = manifest.get("llama_cpp_tag")
    if upstream_tag is None and native_version.rebuild == 0:
        upstream_tag = native_version.upstream_tag
    if not isinstance(upstream_tag, str):
        raise ContractError("native manifest is missing an exact llama_cpp_tag")
    upstream = parse_upstream_tag(upstream_tag)
    if (
        native_version.channel != upstream.channel
        or native_version.upstream_parts != upstream.parts
    ):
        raise ContractError(
            f"native release {native_tag!r} does not preserve upstream identity {upstream_tag!r}"
        )

    return NativeIdentity(
        release_tag=native_tag,
        upstream_tag=upstream_tag,
        upstream_commit=_require_commit(manifest.get("llama_cpp_commit"), "llama_cpp_commit"),
        native_commit=_require_commit(manifest.get("native_commit"), "native_commit"),
    )


def validate_native_file(
    path: Path,
    expected_sha256: str,
    native_release_tag: str,
    upstream_tag: str,
    upstream_commit: str,
) -> NativeIdentity:
    expected_sha256 = require_sha256(expected_sha256, "native_manifest_sha256")
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ContractError(
            f"native manifest SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"could not read native manifest: {error}") from error
    if not isinstance(payload, Mapping):
        raise ContractError("native manifest root must be a JSON object")
    identity = resolve_native_manifest(payload, native_release_tag)
    if identity.upstream_tag != upstream_tag:
        raise ContractError(
            f"native manifest upstream tag {identity.upstream_tag!r} does not match {upstream_tag!r}"
        )
    if identity.upstream_commit != upstream_commit:
        raise ContractError(
            "native manifest llama_cpp_commit does not match the requested upstream commit"
        )
    return identity


def read_previous_manifest(path: Path) -> tuple[str, str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"could not read previous manifest: {error}") from error
    if not isinstance(payload, Mapping):
        raise ContractError("previous manifest root must be a JSON object")
    release_tag = payload.get("release_tag", payload.get("bridge_assets_tag"))
    upstream_tag = payload.get("upstream_tag", payload.get("llama_cpp_tag"))
    bridge_commit = payload.get("bridge_commit", payload.get("source_commit"))
    if not isinstance(release_tag, str) or not isinstance(upstream_tag, str):
        raise ContractError("previous manifest is missing release/upstream tag identity")
    parse_release_tag(release_tag, allow_legacy=True)
    parse_upstream_tag(upstream_tag)
    return release_tag, upstream_tag, _require_commit(bridge_commit, "bridge_commit")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-release")
    validate.add_argument("--release-tag", required=True)
    validate.add_argument("--release-rebuild", required=True, type=int)
    validate.add_argument("--upstream-tag", required=True)
    validate.add_argument("--previous-manifest", type=Path)

    native = subparsers.add_parser("validate-native")
    native.add_argument("--manifest", required=True, type=Path)
    native.add_argument("--manifest-sha256", required=True)
    native.add_argument("--native-release-tag", required=True)
    native.add_argument("--upstream-tag", required=True)
    native.add_argument("--upstream-commit", required=True)

    compare = subparsers.add_parser("compare-upstream")
    compare.add_argument("current")
    compare.add_argument("target")

    scan = subparsers.add_parser("scan-native")
    scan.add_argument("--manifest", required=True, type=Path)
    scan.add_argument("--native-release-tag", required=True)

    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "validate-release":
            release = validate_release_identity(
                args.release_tag, args.release_rebuild, args.upstream_tag
            )
            result: dict[str, Any] = {
                "release_tag": release.tag,
                "release_channel": release.channel.value,
                "release_rebuild": release.rebuild,
                "upstream_tag": release.upstream_tag,
            }
            if args.previous_manifest:
                previous_tag, previous_upstream, previous_bridge = read_previous_manifest(
                    args.previous_manifest
                )
                release_transition = compare_releases(previous_tag, release.tag)
                upstream_transition = compare_upstream(previous_upstream, args.upstream_tag)
                if release_transition not in (Transition.FORWARD, Transition.STABLE_MIGRATION):
                    raise ContractError(
                        f"release must advance from {previous_tag!r}: {release_transition.value}"
                    )
                if upstream_transition in (
                    Transition.BACKWARD,
                    Transition.FORBIDDEN_STABLE_TO_DEVELOPMENT,
                ):
                    raise ContractError(
                        f"upstream must not roll back from {previous_upstream!r}: {upstream_transition.value}"
                    )
                result.update(
                    previous_release_tag=previous_tag,
                    previous_upstream_tag=previous_upstream,
                    previous_bridge_commit=previous_bridge,
                    release_transition=release_transition.value,
                    upstream_transition=upstream_transition.value,
                )
            print(json.dumps(result, sort_keys=True))
        elif args.command == "validate-native":
            identity = validate_native_file(
                args.manifest,
                args.manifest_sha256,
                args.native_release_tag,
                args.upstream_tag,
                args.upstream_commit,
            )
            print(json.dumps(identity.__dict__, sort_keys=True))
        elif args.command == "compare-upstream":
            print(compare_upstream(args.current, args.target).value)
        else:
            payload = json.loads(args.manifest.read_text(encoding="utf-8"))
            identity = resolve_native_manifest(payload, args.native_release_tag)
            print(json.dumps(identity.__dict__, sort_keys=True))
    except (ContractError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
