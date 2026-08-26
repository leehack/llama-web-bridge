#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from release_contract import (
    Channel,
    ContractError,
    Transition,
    compare_releases,
    compare_upstream,
    expected_native_artifacts,
    parse_release_tag,
    parse_upstream_tag,
    read_previous_manifest,
    require_correlation_id,
    resolve_native_manifest,
    resolve_tag_commit,
    select_stable_native_release,
    validate_native_file,
    validate_native_release,
    validate_native_request,
    validate_github_prerelease,
    validate_publication_environment,
    validate_native_identity,
    validate_release_identity,
)


class ReleaseContractTest(unittest.TestCase):
    def test_accepts_emission_tags(self) -> None:
        cases = {
            "v0.2.0": (Channel.STABLE, (0, 2, 0), 0),
            "v0.2.0-3": (Channel.STABLE, (0, 2, 0), 3),
            "b10514": (Channel.DEVELOPMENT, (10514,), 0),
            "b10514-2": (Channel.DEVELOPMENT, (10514,), 2),
        }
        for tag, expected in cases.items():
            with self.subTest(tag=tag):
                parsed = parse_release_tag(tag)
                self.assertEqual(
                    (parsed.channel, parsed.version_parts, parsed.rebuild), expected
                )
                self.assertFalse(parsed.legacy)

    def test_github_prerelease_matches_native_policy(self) -> None:
        expected = {
            "v0.2.0": False,
            "v0.2.0-1": True,
            "b10514": True,
            "b10514-2": True,
        }
        for tag, prerelease in expected.items():
            with self.subTest(tag=tag):
                self.assertEqual(parse_release_tag(tag).github_prerelease, prerelease)
                self.assertEqual(validate_github_prerelease(tag, prerelease), prerelease)
                with self.assertRaises(ContractError):
                    validate_github_prerelease(tag, not prerelease)
        self.assertTrue(validate_github_prerelease("v0.2.0-llamadart.1", True))

    def test_stable_native_discovery_enumerates_wrappers(self) -> None:
        releases = [
            {"tag_name": "b10599", "draft": False, "prerelease": True},
            {"tag_name": "v0.2.0", "draft": False, "prerelease": False},
            {"tag_name": "v0.2.0-2", "draft": False, "prerelease": True},
            {"tag_name": "v0.3.0-1", "draft": False, "prerelease": False},
            {"tag_name": "v9.0.0", "draft": False, "prerelease": True},
            {"tag_name": "v9.0.0", "draft": True, "prerelease": False},
        ]
        self.assertEqual(select_stable_native_release(releases), "v0.2.0-2")

    def test_correlation_id_rejects_injection_and_ambiguity(self) -> None:
        self.assertEqual(require_correlation_id("llamadart-pin:run-123"), "llamadart-pin:run-123")
        for invalid in ("", " leading", "two words", "line\nbreak", "x" * 129, "../escape"):
            with self.subTest(invalid=invalid), self.assertRaises(ContractError):
                require_correlation_id(invalid)

    def test_publication_environment_requires_fail_closed_policy(self) -> None:
        configured = {
            "name": "bridge-assets-publication",
            "can_admins_bypass": False,
            "protection_rules": [{"type": "branch_policy"}],
            "deployment_branch_policy": {
                "protected_branches": False,
                "custom_branch_policies": True,
            },
        }
        branch_policies = {
            "total_count": 1,
            "branch_policies": [{"name": "main", "type": "branch"}],
        }
        self.assertIsNone(
            validate_publication_environment(configured, branch_policies)
        )

        invalid_cases = {
            "required-reviewers": {
                **configured,
                "protection_rules": [
                    {
                        "type": "required_reviewers",
                        "prevent_self_review": True,
                        "reviewers": [
                            {
                                "type": "User",
                                "reviewer": {"id": 1233094, "login": "leehack"},
                            }
                        ],
                    },
                    {"type": "branch_policy"},
                ],
            },
            "required-reviewers-with-self-review-allowed": {
                **configured,
                "protection_rules": [
                    {
                        "type": "required_reviewers",
                        "prevent_self_review": False,
                        "reviewers": [],
                    },
                    {"type": "branch_policy"},
                ],
            },
            "admin-bypass": {**configured, "can_admins_bypass": True},
            "all-protected-branches": {
                **configured,
                "deployment_branch_policy": {
                    "protected_branches": True,
                    "custom_branch_policies": False,
                },
            },
            "numeric-deployment-policy": {
                **configured,
                "deployment_branch_policy": {
                    "protected_branches": 0,
                    "custom_branch_policies": 1,
                },
            },
            "extra-deployment-policy-key": {
                **configured,
                "deployment_branch_policy": {
                    **configured["deployment_branch_policy"],
                    "unexpected": False,
                },
            },
            "wrong-name": {**configured, "name": "wrong"},
            "malformed-protection-rule": {
                **configured,
                "protection_rules": [*configured["protection_rules"], None],
            },
        }
        for label, invalid in invalid_cases.items():
            with self.subTest(label=label), self.assertRaises(ContractError):
                validate_publication_environment(invalid, branch_policies)

        for invalid_policies in (
            {"total_count": True, "branch_policies": [{"name": "main", "type": "branch"}]},
            {"total_count": 0, "branch_policies": []},
            {
                "total_count": 1,
                "branch_policies": [{"name": "release/*", "type": "branch"}],
            },
            {
                "total_count": 2,
                "branch_policies": [
                    {"name": "main", "type": "branch"},
                    {"name": "release/*", "type": "branch"},
                ],
            },
        ):
            with self.subTest(invalid_policies=invalid_policies), self.assertRaises(
                ContractError
            ):
                validate_publication_environment(configured, invalid_policies)

    def test_validate_environment_cli_reports_only_environment_identity(self) -> None:
        configured = {
            "name": "bridge-assets-publication",
            "can_admins_bypass": False,
            "protection_rules": [{"type": "branch_policy"}],
            "deployment_branch_policy": {
                "protected_branches": False,
                "custom_branch_policies": True,
            },
        }
        branch_policies = {
            "total_count": 1,
            "branch_policies": [{"name": "main", "type": "branch"}],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            environment_path = temporary_path / "environment.json"
            branch_policies_path = temporary_path / "branch-policies.json"
            environment_path.write_text(json.dumps(configured), encoding="utf-8")
            branch_policies_path.write_text(
                json.dumps(branch_policies), encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("release_contract.py")),
                    "validate-environment",
                    "--environment-json",
                    str(environment_path),
                    "--branch-policies-json",
                    str(branch_policies_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(
            json.loads(result.stdout),
            {"environment": "bridge-assets-publication"},
        )

    def test_legacy_wrappers_are_consumption_only(self) -> None:
        for tag in ("b10514-llamadart.2", "v0.2.0-llamadart.1"):
            with self.subTest(tag=tag):
                with self.assertRaises(ContractError):
                    parse_release_tag(tag)
                self.assertTrue(parse_release_tag(tag, allow_legacy=True).legacy)

    def test_rejects_invalid_and_prerelease_tags(self) -> None:
        for tag in (
            "v0.2",
            "v0.2.0-0",
            "v0.2.0-rc.1",
            "v00.2.0",
            "b010514",
            "b10514-0",
            "b10514-1-extra",
            " main ",
        ):
            with self.subTest(tag=tag), self.assertRaises(ContractError):
                parse_release_tag(tag)

    def test_upstream_channels_are_exact(self) -> None:
        self.assertEqual(parse_upstream_tag("v0.2.0").parts, (0, 2, 0))
        self.assertEqual(parse_upstream_tag("b10514").parts, (10514,))
        for tag in ("v0.2.0-1", "b10514-1", "v0.2.0-rc.1"):
            with self.subTest(tag=tag), self.assertRaises(ContractError):
                parse_upstream_tag(tag)

    def test_ordinary_pin_accepts_either_upstream_channel(self) -> None:
        """scripts/verify_ci_reliability.py gates llama_cpp.version with this parser."""
        pin_path = Path(__file__).resolve().parents[1] / "llama_cpp.version"
        pin_contents = pin_path.read_text(encoding="utf-8")
        self.assertEqual(pin_contents, "v0.2.0\n")
        pin = pin_contents.removesuffix("\n")
        self.assertIn(
            parse_upstream_tag(pin).channel, (Channel.STABLE, Channel.DEVELOPMENT)
        )
        for tag, channel in (("v0.2.0", Channel.STABLE), ("b10514", Channel.DEVELOPMENT)):
            with self.subTest(tag=tag):
                self.assertEqual(parse_upstream_tag(tag).channel, channel)
        for invalid in (
            "",
            "0.2.0",
            "v0.2",
            "v0.2.0.1",
            "v0.2.0-1",
            "V0.2.0",
            "b",
            "b10514-1",
            "b10514\n",
            "B10514",
            "main",
            "bb4caa7540188872173c44d161602d9271386413",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ContractError):
                parse_upstream_tag(invalid)

    def test_native_request_is_validated_before_network_use(self) -> None:
        accepted = validate_native_request(
            "v0.2.0-1",
            "v0.2.0",
            "bb4caa7540188872173c44d161602d9271386413",
            "2e5d29d7f98f0d71e75d3fa63b7c55f3b2a7933247cc34ea2b1c5e053d142452",
        )
        self.assertEqual(accepted.base_tag, "v0.2.0")
        for label, native_tag, upstream_tag, upstream_commit, manifest_sha256 in (
            ("malformed-tag", "v0.2.0-rc.1", "v0.2.0", "a" * 40, "b" * 64),
            ("wrong-upstream", "v0.2.0-1", "v0.2.1", "a" * 40, "b" * 64),
            ("malformed-upstream", "v0.2.0-1", "main", "a" * 40, "b" * 64),
            ("malformed-commit", "v0.2.0-1", "v0.2.0", "abc123", "b" * 64),
            ("malformed-sha256", "v0.2.0-1", "v0.2.0", "a" * 40, "B" * 64),
        ):
            with self.subTest(case=label), self.assertRaises(ContractError):
                validate_native_request(
                    native_tag,
                    upstream_tag,
                    upstream_commit,
                    manifest_sha256,
                )

    def test_channel_and_rebuild_ordering(self) -> None:
        cases = (
            ("b10514", "b10515", Transition.FORWARD),
            ("b10514", "v0.2.0", Transition.STABLE_MIGRATION),
            ("v0.2.0", "v0.2.1", Transition.FORWARD),
            ("v0.2.1", "v0.2.0", Transition.BACKWARD),
            ("v0.2.0", "b10515", Transition.FORBIDDEN_STABLE_TO_DEVELOPMENT),
            ("v0.2.0", "v0.2.0-1", Transition.FORWARD),
            ("b10514-llamadart.1", "b10514-2", Transition.FORWARD),
        )
        for current, target, expected in cases:
            with self.subTest(current=current, target=target):
                self.assertEqual(compare_releases(current, target), expected)

        self.assertEqual(
            compare_upstream("b10514", "v0.2.0"), Transition.STABLE_MIGRATION
        )
        with self.assertRaises(ContractError):
            compare_releases("v0.2.0", "v0.2.1-1")

    def test_release_and_upstream_orderings_are_independent(self) -> None:
        """The chief candidate: assets v0.1.37 -> v0.1.38 while upstream b10514 -> v0.2.0."""
        self.assertEqual(compare_releases("v0.1.37", "v0.1.38"), Transition.FORWARD)
        self.assertEqual(
            compare_upstream("b10514", "v0.2.0"), Transition.STABLE_MIGRATION
        )
        self.assertEqual(compare_releases("v0.1.38", "v0.1.37"), Transition.BACKWARD)
        self.assertEqual(compare_releases("v0.1.38", "v0.1.38"), Transition.EQUAL)
        self.assertEqual(compare_releases("v0.1.38", "v0.1.38-1"), Transition.FORWARD)
        self.assertEqual(compare_releases("v0.1.38-2", "v0.1.38-1"), Transition.BACKWARD)
        self.assertEqual(
            compare_upstream("v0.2.0", "b10600"),
            Transition.FORBIDDEN_STABLE_TO_DEVELOPMENT,
        )
        self.assertEqual(compare_upstream("v0.2.1", "v0.2.0"), Transition.BACKWARD)
        # A new asset version must restart at rebuild 0 regardless of upstream.
        with self.assertRaises(ContractError):
            compare_releases("v0.1.37", "v0.1.38-1")

    def test_release_identity_is_independent_of_upstream(self) -> None:
        """Bridge assets version independently: v0.1.38 may ship upstream v0.2.0."""
        candidate = validate_release_identity("v0.1.38", 0, "v0.2.0")
        self.assertEqual(
            (candidate.tag, candidate.channel, candidate.version_parts, candidate.rebuild),
            ("v0.1.38", Channel.STABLE, (0, 1, 38), 0),
        )
        for release, rebuild, upstream in (
            ("v0.2.0-2", 2, "v0.2.0"),
            ("v0.1.38", 0, "b10514"),
            ("b10600", 0, "v0.2.0"),
            ("v0.1.38-3", 3, "v0.2.0"),
        ):
            with self.subTest(release=release, upstream=upstream):
                self.assertEqual(
                    validate_release_identity(release, rebuild, upstream).tag, release
                )

    def test_release_identity_keeps_strict_syntax_and_rebuild(self) -> None:
        for label, release, rebuild, upstream in (
            ("rebuild-mismatch", "v0.2.0-2", 1, "v0.2.0"),
            ("rebuild-negative", "v0.1.38", -1, "v0.2.0"),
            ("rebuild-zero-mismatch", "v0.1.38-1", 0, "v0.2.0"),
            ("malformed-release", "v0.1", 0, "v0.2.0"),
            ("prerelease-release", "v0.1.38-rc.1", 0, "v0.2.0"),
            ("legacy-release", "v0.1.38-llamadart.1", 1, "v0.2.0"),
            ("malformed-upstream", "v0.1.38", 0, "main"),
            ("rebuild-bearing-upstream", "v0.1.38", 0, "v0.2.0-1"),
        ):
            with self.subTest(case=label), self.assertRaises(ContractError):
                validate_release_identity(release, rebuild, upstream)

    def test_native_identity_still_encodes_its_upstream(self) -> None:
        self.assertEqual(
            validate_native_identity("v0.2.0-1", 1, "v0.2.0").tag, "v0.2.0-1"
        )
        self.assertEqual(
            validate_native_identity("b10514", 0, "b10514").tag, "b10514"
        )
        for label, native_tag, rebuild, upstream in (
            ("stable-upstream-mismatch", "v0.2.0-1", 1, "v0.2.1"),
            ("independent-versioning-forbidden", "v0.1.38", 0, "v0.2.0"),
            ("channel-mismatch", "b10514", 0, "v0.2.0"),
            ("development-upstream-mismatch", "b10514", 0, "b10515"),
            ("rebuild-mismatch", "v0.2.0-1", 0, "v0.2.0"),
            ("negative-rebuild", "v0.2.0-1", -1, "v0.2.0"),
        ):
            with self.subTest(case=label), self.assertRaises(ContractError):
                validate_native_identity(native_tag, rebuild, upstream)

    def test_native_manifest_provenance_and_checksum(self) -> None:
        manifest = {
            "tag": "v0.2.0-1",
            "native_release_tag": "v0.2.0-1",
            "llama_cpp_tag": "v0.2.0",
            "llama_cpp_commit": "a" * 40,
            "native_commit": "b" * 40,
        }
        identity = resolve_native_manifest(manifest, "v0.2.0-1")
        self.assertEqual(identity.upstream_tag, "v0.2.0")
        self.assertEqual(identity.native_commit, "b" * 40)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "assets.json")
            path.write_text(json.dumps(manifest), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            validated = validate_native_file(
                path, digest, "v0.2.0-1", "v0.2.0", "a" * 40
            )
            self.assertEqual(validated, identity)
            with self.assertRaises(ContractError):
                validate_native_file(
                    path, "0" * 64, "v0.2.0-1", "v0.2.0", "a" * 40
                )

    @staticmethod
    def _native_release_fixture(
        directory: str,
        tag: str = "v0.2.0-1",
        upstream_tag: str = "v0.2.0",
    ) -> dict:
        """Build the real v0.2.0-1 release shape with a branch target."""
        upstream_commit = "bb4caa7540188872173c44d161602d9271386413"
        native_commit = "e5c240e34b525da953ed98dc743516eef78cb738"
        expected = expected_native_artifacts(tag)
        artifacts = []
        checksum_lines = []
        github_assets = []
        for index, (name, (platform, arch, backend, module)) in enumerate(expected.items()):
            digest = hashlib.sha256(f"artifact-{index}".encode()).hexdigest()
            size = index + 100
            artifacts.append({
                "module": module,
                "platform": platform,
                "arch": arch,
                "backend": backend,
                "file": name,
                "sha256": digest,
                "size": size,
            })
            checksum_lines.append(f"{digest}  {name}")
            github_assets.append({
                "name": name, "state": "uploaded", "size": size,
                "digest": f"sha256:{digest}",
            })
        manifest = {
            "tag": tag,
            "llama_cpp_tag": upstream_tag,
            "llama_cpp_commit": upstream_commit,
            "native_commit": native_commit,
            "generated_at": "2026-08-22T00:00:00Z",
            "hook_contract_version": 1,
            "artifacts": artifacts,
        }
        manifest_path = Path(directory, "assets.json")
        checksums_path = Path(directory, "SHA256SUMS")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        checksum_digest = hashlib.sha256(checksums_path.read_bytes()).hexdigest()
        github_assets.extend([
            {"name": "assets.json", "state": "uploaded", "size": manifest_path.stat().st_size, "digest": f"sha256:{manifest_digest}"},
            {"name": "SHA256SUMS", "state": "uploaded", "size": checksums_path.stat().st_size, "digest": f"sha256:{checksum_digest}"},
        ])
        return {
            "tag": tag,
            "upstream_tag": upstream_tag,
            "upstream_commit": upstream_commit,
            "native_commit": native_commit,
            "manifest": manifest,
            "manifest_path": manifest_path,
            "checksums_path": checksums_path,
            "manifest_digest": manifest_digest,
            "release": {
                "tag_name": tag,
                "draft": False,
                "prerelease": True,
                "target_commitish": "main",
                "assets": github_assets,
            },
        }

    def test_native_release_verifies_github_digest_hook_and_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._native_release_fixture(directory)
            tag = fixture["tag"]
            manifest = fixture["manifest"]
            manifest_path = fixture["manifest_path"]
            checksums_path = fixture["checksums_path"]
            manifest_digest = fixture["manifest_digest"]
            release = fixture["release"]
            identity = validate_native_release(
                manifest_path, checksums_path, release, manifest_digest,
                tag, fixture["upstream_tag"], fixture["upstream_commit"],
                fixture["native_commit"],
            )
            self.assertEqual(identity.native_commit, fixture["native_commit"])
            for mutation in ("hook", "digest", "inventory", "prerelease"):
                bad_manifest = json.loads(json.dumps(manifest))
                bad_release = json.loads(json.dumps(release))
                if mutation == "hook":
                    bad_manifest["hook_contract_version"] = 2
                    manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
                    bad_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                    bad_release["assets"][-2]["digest"] = f"sha256:{bad_digest}"
                    bad_release["assets"][-2]["size"] = manifest_path.stat().st_size
                elif mutation == "digest":
                    bad_release["assets"][0]["digest"] = "sha256:" + "0" * 64
                    bad_digest = manifest_digest
                else:
                    if mutation == "inventory":
                        bad_release["assets"].pop(0)
                    else:
                        bad_release["prerelease"] = False
                    bad_digest = manifest_digest
                with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                    validate_native_release(
                        manifest_path, checksums_path, bad_release, bad_digest,
                        tag, fixture["upstream_tag"], fixture["upstream_commit"],
                        fixture["native_commit"],
                    )
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_native_release_trusts_only_the_resolved_immutable_tag_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._native_release_fixture(directory)
            tag = fixture["tag"]

            def validate(release: dict, native_tag_commit) -> None:
                validate_native_release(
                    fixture["manifest_path"],
                    fixture["checksums_path"],
                    release,
                    fixture["manifest_digest"],
                    tag,
                    fixture["upstream_tag"],
                    fixture["upstream_commit"],
                    native_tag_commit,
                )

            branch_target = fixture["release"]
            self.assertEqual(branch_target["target_commitish"], "main")
            validate(branch_target, fixture["native_commit"])
            validate(
                {**branch_target, "target_commitish": fixture["native_commit"]},
                fixture["native_commit"],
            )

            for label, native_tag_commit in (
                ("mismatch", "c" * 40),
                ("missing", None),
                ("empty", ""),
                ("branch-name", "main"),
                ("short", "b" * 39),
                ("uppercase", "B" * 40),
                ("non-string", 0),
            ):
                with self.subTest(native_tag_commit=label), self.assertRaises(
                    ContractError
                ):
                    validate(branch_target, native_tag_commit)

            for label, target_commitish in (
                ("other-commit", "c" * 40),
                ("missing", None),
                ("empty", ""),
                ("abbreviated-commit", "c" * 12),
                ("uppercase-commit", "C" * 40),
                ("whitespace", " main"),
                ("tag-ref", "refs/tags/v0.1.0"),
                ("tag-shaped", "v0.1.0"),
            ):
                with self.subTest(target_commitish=label), self.assertRaises(
                    ContractError
                ):
                    validate(
                        {**branch_target, "target_commitish": target_commitish},
                        fixture["native_commit"],
                    )

    def test_resolve_tag_commit_peels_annotated_tags(self) -> None:
        upstream_annotated = (
            "8a35040e02747e136d901793604572c7ca6d0793\trefs/tags/v0.2.0\n"
            "bb4caa7540188872173c44d161602d9271386413\trefs/tags/v0.2.0^{}\n"
        )
        self.assertEqual(
            resolve_tag_commit(upstream_annotated, "v0.2.0"),
            "bb4caa7540188872173c44d161602d9271386413",
        )
        annotated = (
            "246e18e254d74452a32210992cefbcab8dc65010\trefs/tags/v0.2.0-1\n"
            "e5c240e34b525da953ed98dc743516eef78cb738\trefs/tags/v0.2.0-1^{}\n"
        )
        self.assertEqual(
            resolve_tag_commit(annotated, "v0.2.0-1"),
            "e5c240e34b525da953ed98dc743516eef78cb738",
        )
        lightweight = "246e18e254d74452a32210992cefbcab8dc65010\trefs/tags/v0.2.0-1\n"
        self.assertEqual(
            resolve_tag_commit(lightweight, "v0.2.0-1"),
            "246e18e254d74452a32210992cefbcab8dc65010",
        )

        for label, output in (
            ("missing", ""),
            ("blank-lines-only", "\n\n"),
            ("unrelated-ref", "b" * 40 + "\trefs/tags/v0.2.0-10\n"),
            ("branch-ref", "b" * 40 + "\trefs/heads/main\n"),
            ("malformed-line", "not-a-sha\trefs/tags/v0.2.0-1\n"),
            ("space-separated", "b" * 40 + " refs/tags/v0.2.0-1\n"),
            (
                "duplicate",
                ("b" * 40 + "\trefs/tags/v0.2.0-1\n") * 2,
            ),
            ("conflicting", "b" * 40 + "\trefs/tags/v0.2.0-1\n" + "c" * 40 + "\trefs/tags/v0.2.0-1\n"),
            ("peeled-only", "c" * 40 + "\trefs/tags/v0.2.0-1^{}\n"),
            (
                "same-object-and-commit",
                "b" * 40 + "\trefs/tags/v0.2.0-1\n"
                + "b" * 40 + "\trefs/tags/v0.2.0-1^{}\n",
            ),
            ("embedded-blank", "b" * 40 + "\trefs/tags/v0.2.0-1\n\n"),
        ):
            with self.subTest(output=label), self.assertRaises(ContractError):
                resolve_tag_commit(output, "v0.2.0-1")

        with self.assertRaises(ContractError):
            resolve_tag_commit(lightweight, "v0.2.0-1;touch-pwned")

    def test_resolve_tag_commit_cli_emits_only_the_peeled_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            refs = Path(directory, "tag-refs.txt")
            refs.write_text(
                "246e18e254d74452a32210992cefbcab8dc65010\trefs/tags/v0.2.0-1\n"
                "e5c240e34b525da953ed98dc743516eef78cb738\trefs/tags/v0.2.0-1^{}\n",
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(Path(__file__).with_name("release_contract.py")),
                "resolve-tag-commit",
                "--ls-remote",
                str(refs),
                "--tag",
                "v0.2.0-1",
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(
                result.stdout.strip(), "e5c240e34b525da953ed98dc743516eef78cb738"
            )

            refs.write_text("", encoding="utf-8")
            failure = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failure.returncode, 0)

    def test_rejects_mismatched_native_identity(self) -> None:
        base = {
            "tag": "v0.2.0-1",
            "native_release_tag": "v0.2.0-1",
            "llama_cpp_tag": "v0.2.0",
            "llama_cpp_commit": "a" * 40,
            "native_commit": "b" * 40,
        }
        invalid = (
            ({**base, "tag": "v0.2.0"}, "v0.2.0-1"),
            ({**base, "llama_cpp_tag": "v0.2.1"}, "v0.2.0-1"),
            ({**base, "llama_cpp_commit": "abc123"}, "v0.2.0-1"),
            ({**base, "native_commit": "B" * 40}, "v0.2.0-1"),
        )
        for manifest, tag in invalid:
            with self.subTest(manifest=manifest), self.assertRaises(ContractError):
                resolve_native_manifest(manifest, tag)

    def test_reads_current_and_legacy_previous_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory, "current.json")
            current.write_text(
                json.dumps(
                    {
                        "release_tag": "v0.2.0-1",
                        "upstream_tag": "v0.2.0",
                        "bridge_commit": "c" * 40,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                read_previous_manifest(current),
                ("v0.2.0-1", "v0.2.0", "c" * 40),
            )

            legacy = Path(directory, "legacy.json")
            legacy.write_text(
                json.dumps(
                    {
                        "bridge_assets_tag": "b10514-llamadart.1",
                        "llama_cpp_tag": "b10514",
                        "source_commit": "d" * 40,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(read_previous_manifest(legacy)[0], "b10514-llamadart.1")

            for field, value in (
                ("bridge_assets_tag", "v0.2.1"),
                ("llama_cpp_tag", "v0.2.1"),
                ("source_commit", "d" * 40),
            ):
                conflicting = Path(directory, f"conflicting-{field}.json")
                conflicting.write_text(
                    json.dumps(
                        {
                            "release_tag": "v0.2.0-1",
                            "upstream_tag": "v0.2.0",
                            "bridge_commit": "c" * 40,
                            field: value,
                        }
                    ),
                    encoding="utf-8",
                )
                with self.subTest(field=field), self.assertRaises(ContractError):
                    read_previous_manifest(conflicting)


if __name__ == "__main__":
    unittest.main()
