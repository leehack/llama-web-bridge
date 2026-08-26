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


BRIDGE_REPOSITORY = "leehack/llama-web-bridge"
ASSETS_REPOSITORY = "leehack/llama-web-bridge-assets"
NATIVE_REPOSITORY = "leehack/llamadart-native"
NATIVE_HOOK_CONTRACT_VERSION = 1


class ContractError(ValueError):
    """Raised when release provenance is ambiguous or unsafe."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json_loads(text: str, label: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ContractError(f"could not parse {label}: {error}") from error


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
_HEX_COMMITISH_RE = re.compile(r"[0-9A-Fa-f]{4,40}")
_REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_CORRELATION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
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
_LS_REMOTE_LINE_RE = re.compile(r"([0-9a-f]{40})\t(\S+)")


@dataclass(frozen=True)
class UpstreamVersion:
    tag: str
    channel: Channel
    parts: tuple[int, ...]


@dataclass(frozen=True)
class ReleaseVersion:
    """A release tag's own identity, independent of any upstream llama.cpp tag.

    Bridge asset releases version independently of llama.cpp, so ``version_parts``
    is the tag's own version. Native releases separately promise that their base
    tag *is* their upstream tag; only the native validators enforce that.
    """

    tag: str
    channel: Channel
    version_parts: tuple[int, ...]
    rebuild: int
    legacy: bool = False

    @property
    def base_tag(self) -> str:
        """The tag with its rebuild suffix removed."""
        if self.channel == Channel.DEVELOPMENT:
            return f"b{self.version_parts[0]}"
        return "v" + ".".join(str(part) for part in self.version_parts)

    @property
    def github_prerelease(self) -> bool:
        """Match llamadart-native: nightlies and every rebuild are prereleases."""
        return self.channel == Channel.DEVELOPMENT or self.rebuild > 0


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


def require_correlation_id(value: str) -> str:
    if _CORRELATION_ID_RE.fullmatch(value) is None:
        raise ContractError(
            "orchestrator_correlation_id must be 1-128 safe identifier characters"
        )
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


def _compare_versions(
    current_channel: Channel,
    current_parts: tuple[int, ...],
    target_channel: Channel,
    target_parts: tuple[int, ...],
) -> Transition:
    """Order two same-kind identities: within a channel, then across channels."""
    if current_channel == target_channel:
        if current_parts == target_parts:
            return Transition.EQUAL
        return Transition.FORWARD if target_parts > current_parts else Transition.BACKWARD
    if current_channel == Channel.DEVELOPMENT and target_channel == Channel.STABLE:
        return Transition.STABLE_MIGRATION
    return Transition.FORBIDDEN_STABLE_TO_DEVELOPMENT


def compare_upstream(current_tag: str, target_tag: str) -> Transition:
    current = parse_upstream_tag(current_tag)
    target = parse_upstream_tag(target_tag)
    return _compare_versions(
        current.channel, current.parts, target.channel, target.parts
    )


def compare_releases(current_tag: str, target_tag: str) -> Transition:
    """Order two release tags by their own identity, never by an upstream tag."""
    current = parse_release_tag(current_tag, allow_legacy=True)
    target = parse_release_tag(target_tag)

    transition = _compare_versions(
        current.channel, current.version_parts, target.channel, target.version_parts
    )
    if transition == Transition.EQUAL:
        if current.rebuild == target.rebuild:
            return Transition.EQUAL
        return Transition.FORWARD if target.rebuild > current.rebuild else Transition.BACKWARD

    if transition in (Transition.FORWARD, Transition.STABLE_MIGRATION):
        if target.rebuild != 0:
            raise ContractError(
                "the first artifact for a new release version must use rebuild 0"
            )
    return transition


def validate_release_identity(release_tag: str, rebuild: int, upstream_tag: str) -> ReleaseVersion:
    """Validate a bridge asset release tag and its independent upstream tag.

    Bridge assets version independently of llama.cpp: ``v0.1.38`` may ship
    upstream ``v0.2.0``. Both identities are still syntactically exact, and the
    tag must encode the requested rebuild.
    """
    if rebuild < 0:
        raise ContractError("release_rebuild must be zero or greater")
    release = parse_release_tag(release_tag)
    parse_upstream_tag(upstream_tag)
    if release.rebuild != rebuild:
        raise ContractError(
            f"release tag {release_tag!r} encodes rebuild {release.rebuild}, not {rebuild}"
        )
    return release


def _require_native_upstream_identity(
    native_version: ReleaseVersion, upstream: UpstreamVersion
) -> None:
    """Native releases, unlike bridge assets, encode their upstream tag."""
    if (
        native_version.channel != upstream.channel
        or native_version.version_parts != upstream.parts
    ):
        raise ContractError(
            f"native release {native_version.tag!r} does not preserve upstream "
            f"identity {upstream.tag!r}"
        )


def validate_native_identity(
    native_release_tag: str, rebuild: int, upstream_tag: str
) -> ReleaseVersion:
    if rebuild < 0:
        raise ContractError("release_rebuild must be zero or greater")
    native_release = parse_release_tag(native_release_tag)
    upstream = parse_upstream_tag(upstream_tag)
    if native_release.rebuild != rebuild:
        raise ContractError(
            f"native release tag {native_release_tag!r} encodes rebuild "
            f"{native_release.rebuild}, not {rebuild}"
        )
    _require_native_upstream_identity(native_release, upstream)
    return native_release


def validate_native_request(
    native_release_tag: str,
    upstream_tag: str,
    upstream_commit: Any,
    manifest_sha256: str,
) -> ReleaseVersion:
    """Validate native release inputs before using them in network requests."""
    native_release = parse_release_tag(native_release_tag)
    validate_native_identity(
        native_release_tag,
        native_release.rebuild,
        upstream_tag,
    )
    _require_commit(upstream_commit, "upstream_commit")
    require_sha256(manifest_sha256, "native_manifest_sha256")
    return native_release


def validate_github_prerelease(tag: str, actual: Any, *, allow_legacy: bool = True) -> bool:
    version = parse_release_tag(tag, allow_legacy=allow_legacy)
    if actual is not version.github_prerelease:
        raise ContractError(
            f"GitHub prerelease state for {tag!r} must be {version.github_prerelease}"
        )
    return version.github_prerelease


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
        upstream_tag = native_version.base_tag
    if not isinstance(upstream_tag, str):
        raise ContractError("native manifest is missing an exact llama_cpp_tag")
    upstream = parse_upstream_tag(upstream_tag)
    _require_native_upstream_identity(native_version, upstream)

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
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ContractError(f"could not read native manifest: {error}") from error
    payload = _strict_json_loads(text, "native manifest")
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


_NATIVE_ARTIFACTS = {
    "android-arm64": ("android", "arm64", "core", "core"),
    "android-x64": ("android", "x64", "core", "core"),
    "ios-arm64": ("ios", "arm64", "core", "core"),
    "ios-arm64-sim": ("ios", "arm64-sim", "core", "core"),
    "ios-x86_64-sim": ("ios", "x86_64-sim", "core", "core"),
    "linux-arm64": ("linux", "arm64", "core", "core"),
    "linux-x64": ("linux", "x64", "core", "core"),
    "macos-arm64": ("macos", "arm64", "core", "core"),
    "macos-x86_64": ("macos", "x86_64", "core", "core"),
    "windows-arm64": ("windows", "arm64", "core", "core"),
    "windows-x64": ("windows", "x64", "core", "core"),
}


def expected_native_artifacts(tag: str) -> dict[str, tuple[str, str, str, str]]:
    result = {
        f"llamadart-native-{bundle}-{tag}.tar.gz": metadata
        for bundle, metadata in _NATIVE_ARTIFACTS.items()
    }
    result[f"llamadart-native-apple-xcframework-{tag}.zip"] = (
        "apple",
        "universal",
        "core",
        "spm-xcframework",
    )
    result[f"llamadart-native-headers-{tag}.tar.gz"] = (
        "all",
        "universal",
        "core",
        "headers",
    )
    return result


def _release_assets(release: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_assets = release.get("assets")
    if not isinstance(raw_assets, list):
        raise ContractError("native GitHub release is missing its asset inventory")
    assets: dict[str, Mapping[str, Any]] = {}
    for item in raw_assets:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise ContractError("native GitHub release contains an invalid asset record")
        name = item["name"]
        if name in assets:
            raise ContractError(f"native GitHub release has duplicate asset {name!r}")
        if item.get("state") != "uploaded":
            raise ContractError(f"native GitHub asset {name!r} is not uploaded")
        digest = item.get("digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ContractError(f"native GitHub asset {name!r} has no SHA-256 digest")
        require_sha256(digest.removeprefix("sha256:"), f"GitHub digest for {name}")
        if not isinstance(item.get("size"), int) or item["size"] < 0:
            raise ContractError(f"native GitHub asset {name!r} has an invalid size")
        assets[name] = item
    return assets


def resolve_tag_commit(ls_remote_output: str, tag: str) -> str:
    """Resolve the immutable commit a release tag names, peeling annotated tags.

    ``git ls-remote`` must be invoked with both ``refs/tags/<tag>`` and
    ``refs/tags/<tag>^{}`` patterns: the peeled entry is the only line that
    carries the commit for an annotated tag, and it is not tail-matched by the
    unpeeled pattern.
    """
    parse_release_tag(tag)
    plain_ref = f"refs/tags/{tag}"
    peeled_ref = f"{plain_ref}^{{}}"
    refs: dict[str, str] = {}
    for line in ls_remote_output.splitlines():
        if not line:
            raise ContractError("git ls-remote returned a blank line")
        match = _LS_REMOTE_LINE_RE.fullmatch(line)
        if match is None:
            raise ContractError(f"invalid git ls-remote line: {line!r}")
        object_id, ref = match.group(1), match.group(2)
        if ref not in (plain_ref, peeled_ref):
            raise ContractError(f"git ls-remote returned unrelated ref {ref!r}")
        if ref in refs:
            qualifier = "conflicting" if refs[ref] != object_id else "duplicate"
            raise ContractError(
                f"git ls-remote reported {qualifier} objects for {ref!r}"
            )
        refs[ref] = object_id

    plain_object = refs.get(plain_ref)
    if plain_object is None:
        raise ContractError(f"tag {tag!r} does not exist in the remote repository")
    peeled_object = refs.get(peeled_ref)
    if peeled_object == plain_object:
        raise ContractError("peeled tag commit must differ from its annotated tag object")
    return peeled_object or plain_object


def validate_native_release(
    manifest_path: Path,
    checksums_path: Path,
    release: Mapping[str, Any],
    expected_sha256: str,
    native_release_tag: str,
    upstream_tag: str,
    upstream_commit: str,
    native_tag_commit: Any,
) -> NativeIdentity:
    """Validate native provenance against both downloaded bytes and GitHub metadata."""
    identity = validate_native_file(
        manifest_path,
        expected_sha256,
        native_release_tag,
        upstream_tag,
        upstream_commit,
    )
    if release.get("tag_name") != native_release_tag or release.get("draft") is not False:
        raise ContractError("native GitHub release identity/draft state is not canonical")
    validate_github_prerelease(native_release_tag, release.get("prerelease"))
    # target_commitish is mutable: GitHub reports the branch a release was cut
    # from, not the tag's commit. Trust only the independently resolved
    # immutable tag commit, and still reject a target_commitish that pins a
    # different commit outright.
    if _require_commit(native_tag_commit, "native_tag_commit") != identity.native_commit:
        raise ContractError(
            "resolved native release tag commit does not match manifest native_commit"
        )
    target_commitish = release.get("target_commitish")
    if (
        not isinstance(target_commitish, str)
        or not target_commitish
        or target_commitish != target_commitish.strip()
    ):
        raise ContractError("native GitHub release has invalid target_commitish")
    if _COMMIT_RE.fullmatch(target_commitish) is not None:
        if target_commitish != identity.native_commit:
            raise ContractError("native GitHub release target does not match native_commit")
    elif _HEX_COMMITISH_RE.fullmatch(target_commitish) is not None:
        raise ContractError("native GitHub release target has an ambiguous commit form")
    elif target_commitish.startswith("refs/tags/"):
        raise ContractError("native GitHub release target must not name another tag ref")
    else:
        try:
            parse_release_tag(target_commitish)
        except ContractError:
            pass
        else:
            raise ContractError("native GitHub release target must not be tag-shaped")

    payload = _strict_json_loads(
        manifest_path.read_text(encoding="utf-8"), "native release manifest"
    )
    if payload.get("hook_contract_version") != NATIVE_HOOK_CONTRACT_VERSION:
        raise ContractError(
            f"unsupported native hook_contract_version: {payload.get('hook_contract_version')!r}"
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ContractError("native manifest artifacts must be a list")
    expected = expected_native_artifacts(native_release_tag)
    manifest_assets: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or not isinstance(artifact.get("file"), str):
            raise ContractError("native manifest contains an invalid artifact record")
        name = artifact["file"]
        if name in manifest_assets:
            raise ContractError(f"native manifest has duplicate artifact {name!r}")
        manifest_assets[name] = artifact
    if set(manifest_assets) != set(expected):
        raise ContractError("native manifest artifact inventory is incomplete or unexpected")

    github_assets = _release_assets(release)
    required_release_assets = {*expected, "assets.json", "SHA256SUMS"}
    if set(github_assets) != required_release_assets:
        raise ContractError("native GitHub release asset inventory is incomplete or unexpected")

    manifest_bytes = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_meta = github_assets["assets.json"]
    if (
        manifest_meta["digest"] != f"sha256:{manifest_digest}"
        or manifest_meta["size"] != len(manifest_bytes)
        or manifest_digest != expected_sha256
    ):
        raise ContractError("native assets.json bytes do not match GitHub digest/size")

    checksum_bytes = checksums_path.read_bytes()
    checksum_digest = hashlib.sha256(checksum_bytes).hexdigest()
    checksum_meta = github_assets["SHA256SUMS"]
    if (
        checksum_meta["digest"] != f"sha256:{checksum_digest}"
        or checksum_meta["size"] != len(checksum_bytes)
    ):
        raise ContractError("native SHA256SUMS bytes do not match GitHub digest/size")
    checksum_lines: dict[str, str] = {}
    for line in checksum_bytes.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if match is None or match.group(2) in checksum_lines:
            raise ContractError(f"invalid native SHA256SUMS line: {line!r}")
        checksum_lines[match.group(2)] = match.group(1)
    if set(checksum_lines) != set(expected):
        raise ContractError("native SHA256SUMS inventory does not match manifest artifacts")

    expected_keys = {"module", "platform", "arch", "backend", "file", "sha256", "size"}
    for name, (platform, arch, backend, module) in expected.items():
        artifact = manifest_assets[name]
        if set(artifact) != expected_keys:
            raise ContractError(f"native manifest schema mismatch for {name}")
        digest = artifact.get("sha256")
        size = artifact.get("size")
        github = github_assets[name]
        if (
            artifact.get("platform") != platform
            or artifact.get("arch") != arch
            or artifact.get("backend") != backend
            or artifact.get("module") != module
        ):
            raise ContractError(f"native artifact metadata mismatch for {name}")
        if (
            not isinstance(digest, str)
            or checksum_lines[name] != digest
            or github["digest"] != f"sha256:{digest}"
            or github["size"] != size
        ):
            raise ContractError(f"native artifact checksum/size mismatch for {name}")
    return identity


def select_stable_native_release(releases: list[Any]) -> str:
    """Select the highest supported stable-channel tag without GitHub latest semantics."""
    candidates: list[ReleaseVersion] = []
    for release in releases:
        if not isinstance(release, Mapping) or release.get("draft") is not False:
            continue
        tag = release.get("tag_name")
        if not isinstance(tag, str):
            continue
        try:
            version = parse_release_tag(tag, allow_legacy=True)
        except ContractError:
            continue
        if (
            version.channel == Channel.STABLE
            and release.get("prerelease") is version.github_prerelease
        ):
            candidates.append(version)
    if not candidates:
        raise ContractError("no supported non-draft stable native release exists")
    selected = max(candidates, key=lambda value: (*value.version_parts, value.rebuild))
    return selected.tag


def validate_publication_environment(
    environment: Mapping[str, Any],
    branch_policies: Mapping[str, Any],
) -> None:
    """Require the exact fail-closed solo-maintainer publication policy."""
    if environment.get("name") != "bridge-assets-publication":
        raise ContractError("publication environment identity is missing or incorrect")
    if environment.get("can_admins_bypass") is not False:
        raise ContractError("publication environment must disable administrator bypass")
    rules = environment.get("protection_rules")
    if not isinstance(rules, list):
        raise ContractError("publication environment has no protection rules")
    branch_rule_count = 0
    for rule in rules:
        if not isinstance(rule, Mapping):
            raise ContractError("publication environment protection rules are invalid")
        rule_type = rule.get("type")
        if rule_type == "required_reviewers":
            raise ContractError(
                "solo-maintainer publication environment must not require reviewers"
            )
        if rule_type == "branch_policy":
            branch_rule_count += 1
    if branch_rule_count != 1:
        raise ContractError("publication environment must have one branch policy rule")

    deployment_policy = environment.get("deployment_branch_policy")
    if (
        not isinstance(deployment_policy, Mapping)
        or set(deployment_policy) != {
            "protected_branches",
            "custom_branch_policies",
        }
        or deployment_policy.get("protected_branches") is not False
        or deployment_policy.get("custom_branch_policies") is not True
    ):
        raise ContractError(
            "publication environment must use only custom deployment branch policies"
        )

    policies = branch_policies.get("branch_policies")
    policy_count = branch_policies.get("total_count")
    if (
        not isinstance(policy_count, int)
        or isinstance(policy_count, bool)
        or policy_count != 1
        or not isinstance(policies, list)
        or len(policies) != 1
        or not isinstance(policies[0], Mapping)
        or policies[0].get("name") != "main"
        or policies[0].get("type") != "branch"
    ):
        raise ContractError(
            "publication environment must allow deployments only from the main branch"
        )


def read_previous_manifest(path: Path) -> tuple[str, str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"could not read previous manifest: {error}") from error
    if not isinstance(payload, Mapping):
        raise ContractError("previous manifest root must be a JSON object")
    release_tag = payload.get("release_tag")
    legacy_release_tag = payload.get("bridge_assets_tag")
    if release_tag is None:
        release_tag = legacy_release_tag
    elif legacy_release_tag is not None and legacy_release_tag != release_tag:
        raise ContractError("previous manifest release tag aliases conflict")
    upstream_tag = payload.get("upstream_tag")
    legacy_upstream_tag = payload.get("llama_cpp_tag")
    if upstream_tag is None:
        upstream_tag = legacy_upstream_tag
    elif legacy_upstream_tag is not None and legacy_upstream_tag != upstream_tag:
        raise ContractError("previous manifest upstream tag aliases conflict")
    bridge_commit = payload.get("bridge_commit")
    legacy_bridge_commit = payload.get("source_commit")
    if bridge_commit is None:
        bridge_commit = legacy_bridge_commit
    elif legacy_bridge_commit is not None and legacy_bridge_commit != bridge_commit:
        raise ContractError("previous manifest bridge commit aliases conflict")
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

    native_release = subparsers.add_parser("validate-native-release")
    native_release.add_argument("--manifest", required=True, type=Path)
    native_release.add_argument("--checksums", required=True, type=Path)
    native_release.add_argument("--release-json", required=True, type=Path)
    native_release.add_argument("--manifest-sha256", required=True)
    native_release.add_argument("--native-release-tag", required=True)
    native_release.add_argument("--upstream-tag", required=True)
    native_release.add_argument("--upstream-commit", required=True)
    native_release.add_argument("--native-tag-commit", required=True)

    native_request = subparsers.add_parser("validate-native-request")
    native_request.add_argument("--native-release-tag", required=True)
    native_request.add_argument("--upstream-tag", required=True)
    native_request.add_argument("--upstream-commit", required=True)
    native_request.add_argument("--manifest-sha256", required=True)

    resolve_tag = subparsers.add_parser("resolve-tag-commit")
    resolve_tag.add_argument("--ls-remote", required=True, type=Path)
    resolve_tag.add_argument("--tag", required=True)

    compare = subparsers.add_parser("compare-upstream")
    compare.add_argument("current")
    compare.add_argument("target")

    scan = subparsers.add_parser("scan-native")
    scan.add_argument("--manifest", required=True, type=Path)
    scan.add_argument("--native-release-tag", required=True)

    select = subparsers.add_parser("select-stable-native-release")
    select.add_argument("--releases-json", required=True, type=Path)

    environment = subparsers.add_parser("validate-environment")
    environment.add_argument("--environment-json", required=True, type=Path)
    environment.add_argument("--branch-policies-json", required=True, type=Path)

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
                "upstream_tag": args.upstream_tag,
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
        elif args.command == "validate-native-release":
            release_payload = json.loads(args.release_json.read_text(encoding="utf-8"))
            if not isinstance(release_payload, Mapping):
                raise ContractError("native GitHub release root must be an object")
            identity = validate_native_release(
                args.manifest,
                args.checksums,
                release_payload,
                args.manifest_sha256,
                args.native_release_tag,
                args.upstream_tag,
                args.upstream_commit,
                args.native_tag_commit,
            )
            print(json.dumps(identity.__dict__, sort_keys=True))
        elif args.command == "validate-native-request":
            native_release = validate_native_request(
                args.native_release_tag,
                args.upstream_tag,
                args.upstream_commit,
                args.manifest_sha256,
            )
            print(
                json.dumps(
                    {
                        "native_release_tag": native_release.tag,
                        "upstream_tag": args.upstream_tag,
                    },
                    sort_keys=True,
                )
            )
        elif args.command == "resolve-tag-commit":
            print(
                resolve_tag_commit(
                    args.ls_remote.read_text(encoding="utf-8"), args.tag
                )
            )
        elif args.command == "compare-upstream":
            print(compare_upstream(args.current, args.target).value)
        elif args.command == "scan-native":
            payload = json.loads(args.manifest.read_text(encoding="utf-8"))
            identity = resolve_native_manifest(payload, args.native_release_tag)
            print(json.dumps(identity.__dict__, sort_keys=True))
        elif args.command == "select-stable-native-release":
            releases = json.loads(args.releases_json.read_text(encoding="utf-8"))
            if not isinstance(releases, list):
                raise ContractError("native release listing must be a JSON array")
            print(select_stable_native_release(releases))
        else:
            environment_payload = json.loads(
                args.environment_json.read_text(encoding="utf-8")
            )
            branch_policies_payload = json.loads(
                args.branch_policies_json.read_text(encoding="utf-8")
            )
            if not isinstance(environment_payload, Mapping):
                raise ContractError("publication environment root must be an object")
            if not isinstance(branch_policies_payload, Mapping):
                raise ContractError("publication branch policies root must be an object")
            validate_publication_environment(
                environment_payload,
                branch_policies_payload,
            )
            print(json.dumps({
                "environment": "bridge-assets-publication",
            }, sort_keys=True))
    except (ContractError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
