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
    parse_release_tag,
    parse_upstream_tag,
    read_previous_manifest,
    resolve_native_manifest,
    validate_native_file,
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
