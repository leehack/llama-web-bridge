#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
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
    normalize_environment_secret_pages,
    read_previous_manifest,
    require_correlation_id,
    resolve_native_manifest,
    select_stable_native_release,
    validate_native_file,
    validate_native_release,
    validate_github_prerelease,
    validate_publication_environment,
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
                    (parsed.channel, parsed.upstream_parts, parsed.rebuild), expected
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
            "protection_rules": [
                {
                    "type": "required_reviewers",
                    "prevent_self_review": True,
                    "reviewers": [
                        {
                            "type": "User",
                            "reviewer": {"id": 1234, "login": "maintainer"},
                        }
                    ],
                },
                {"type": "branch_policy"},
            ],
            "deployment_branch_policy": {
                "protected_branches": False,
                "custom_branch_policies": True,
            },
        }
        branch_policies = {
            "total_count": 1,
            "branch_policies": [{"name": "main", "type": "branch"}],
        }
        environment_secrets = {
            "total_count": 1,
            "secrets": [{"name": "WEBGPU_BRIDGE_ASSETS_PAT"}],
        }
        self.assertEqual(
            validate_publication_environment(
                configured, branch_policies, environment_secrets
            ),
            1,
        )

        invalid_cases = {
            "missing-reviewers": {**configured, "protection_rules": [{"type": "branch_policy"}]},
            "admin-bypass": {**configured, "can_admins_bypass": True},
            "self-review": {
                **configured,
                "protection_rules": [
                    {
                        **configured["protection_rules"][0],
                        "prevent_self_review": False,
                    },
                    {"type": "branch_policy"},
                ],
            },
            "all-protected-branches": {
                **configured,
                "deployment_branch_policy": {
                    "protected_branches": True,
                    "custom_branch_policies": False,
                },
            },
            "wrong-name": {**configured, "name": "wrong"},
            "malformed-reviewer-entry": {
                **configured,
                "protection_rules": [
                    {
                        **configured["protection_rules"][0],
                        "reviewers": [{}],
                    },
                    {"type": "branch_policy"},
                ],
            },
            "null-reviewer-entry": {
                **configured,
                "protection_rules": [
                    {
                        **configured["protection_rules"][0],
                        "reviewers": [None],
                    },
                    {"type": "branch_policy"},
                ],
            },
            "missing-reviewer-id": {
                **configured,
                "protection_rules": [
                    {
                        **configured["protection_rules"][0],
                        "reviewers": [
                            {"type": "User", "reviewer": {"login": "maintainer"}}
                        ],
                    },
                    {"type": "branch_policy"},
                ],
            },
            "duplicate-reviewer-id": {
                **configured,
                "protection_rules": [
                    {
                        **configured["protection_rules"][0],
                        "reviewers": [
                            {"type": "User", "reviewer": {"id": 1234}},
                            {"type": "User", "reviewer": {"id": 1234}},
                        ],
                    },
                    {"type": "branch_policy"},
                ],
            },
        }
        for label, invalid in invalid_cases.items():
            with self.subTest(label=label), self.assertRaises(ContractError):
                validate_publication_environment(
                    invalid, branch_policies, environment_secrets
                )

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
                validate_publication_environment(
                    configured, invalid_policies, environment_secrets
                )

        for invalid_secrets in (
            {},
            {"total_count": True, "secrets": [{"name": "WEBGPU_BRIDGE_ASSETS_PAT"}]},
            {"total_count": 0, "secrets": []},
            {"total_count": 1, "secrets": [{"name": "WRONG_PAT"}]},
            {"total_count": 1, "secrets": "WEBGPU_BRIDGE_ASSETS_PAT"},
            {"total_count": 1, "secrets": [{"name": ""}]},
            {"total_count": 1, "secrets": [{"name": "1INVALID"}]},
            {"total_count": 1, "secrets": [{"name": "INVALID-NAME"}]},
            {"total_count": 1, "secrets": [{"name": "GITHUB_RESERVED"}]},
            {"total_count": 1, "secrets": [{"name": "webgpu_bridge_assets_pat"}]},
            {
                "total_count": 2,
                "secrets": [
                    {"name": "WEBGPU_BRIDGE_ASSETS_PAT"},
                    {"name": "BRIDGE_PUBLICATION_ENV_READ_TOKEN"},
                ],
            },
            {
                "total_count": 2,
                "secrets": [
                    {"name": "WEBGPU_BRIDGE_ASSETS_PAT"},
                    {"name": "WEBGPU_BRIDGE_ASSETS_PAT"},
                ],
            },
            {
                "total_count": 2,
                "secrets": [
                    {"name": "WEBGPU_BRIDGE_ASSETS_PAT"},
                    {"name": "webgpu_bridge_assets_pat"},
                ],
            },
            {"total_count": 1, "secrets": [{}]},
        ):
            with self.subTest(invalid_secrets=invalid_secrets), self.assertRaises(
                ContractError
            ):
                validate_publication_environment(
                    configured, branch_policies, invalid_secrets
                )

    def test_environment_secret_pages_fail_closed_before_normalization(self) -> None:
        self.assertEqual(
            normalize_environment_secret_pages(
                [
                    {"total_count": 2, "secrets": [{"name": "FIRST"}]},
                    {"total_count": 2, "secrets": [{"name": "SECOND"}]},
                ]
            ),
            {
                "total_count": 2,
                "secrets": [{"name": "FIRST"}, {"name": "SECOND"}],
            },
        )
        self.assertEqual(
            normalize_environment_secret_pages(
                [{"total_count": 0, "secrets": []}]
            ),
            {"total_count": 0, "secrets": []},
        )

        for invalid_pages in (
            [],
            {},
            [None],
            [{"total_count": True, "secrets": [{}]}],
            [{"total_count": -1, "secrets": []}],
            [{"total_count": 0, "secrets": "invalid"}],
            [
                {"total_count": 2, "secrets": [{"name": "FIRST"}]},
                {"total_count": 1, "secrets": [{"name": "SECOND"}]},
            ],
            [{"total_count": 2, "secrets": [{"name": "ONLY"}]}],
        ):
            with self.subTest(invalid_pages=invalid_pages), self.assertRaises(
                ContractError
            ):
                normalize_environment_secret_pages(invalid_pages)

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

    def test_release_must_preserve_upstream_and_rebuild(self) -> None:
        self.assertEqual(
            validate_release_identity("v0.2.0-2", 2, "v0.2.0").upstream_tag,
            "v0.2.0",
        )
        for release, rebuild, upstream in (
            ("v0.2.0-2", 1, "v0.2.0"),
            ("v0.2.1", 0, "v0.2.0"),
            ("b10514", 0, "v0.2.0"),
        ):
            with self.subTest(release=release), self.assertRaises(ContractError):
                validate_release_identity(release, rebuild, upstream)

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

    def test_native_release_verifies_github_digest_hook_and_inventory(self) -> None:
        tag = "v0.2.0"
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
            "llama_cpp_tag": tag,
            "llama_cpp_commit": "a" * 40,
            "native_commit": "b" * 40,
            "generated_at": "2026-08-22T00:00:00Z",
            "hook_contract_version": 1,
            "artifacts": artifacts,
        }
        with tempfile.TemporaryDirectory() as directory:
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
            release = {
                "tag_name": tag,
                "draft": False,
                "prerelease": False,
                "target_commitish": "b" * 40,
                "assets": github_assets,
            }
            identity = validate_native_release(
                manifest_path, checksums_path, release, manifest_digest,
                tag, tag, "a" * 40,
            )
            self.assertEqual(identity.native_commit, "b" * 40)
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
                        bad_release["prerelease"] = True
                    bad_digest = manifest_digest
                with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                    validate_native_release(
                        manifest_path, checksums_path, bad_release, bad_digest,
                        tag, tag, "a" * 40,
                    )
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

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


if __name__ == "__main__":
    unittest.main()
