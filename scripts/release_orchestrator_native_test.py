#!/usr/bin/env python3
"""Contract tests for native provenance scans and the stable native backlog."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from release_contract import Channel, ContractError, NATIVE_REPOSITORY
from release_orchestrator_fixtures_test import (
    BRIDGE_SHA,
    NATIVE_COMMIT,
    NATIVE_MANIFEST_SHA,
    NATIVE_PUBLISHED_AT,
    UPSTREAM_COMMIT,
    make_provenance,
    native_manifest,
)
import release_orchestrator_model as model
import release_orchestrator_native as native
import stable_release_orchestrator as sro


class ProvenanceTest(unittest.TestCase):
    def test_scan_native_binds_exact_manifest_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "assets.json"
            payload = json.dumps(native_manifest()).encode("utf-8")
            path.write_bytes(payload)
            provenance = native.scan_native_provenance(
                manifest_path=path,
                native_release_tag="v0.2.0",
                bridge_source_sha=BRIDGE_SHA,
                bridge_build_sha=BRIDGE_SHA,
                channel="stable",
                native_release_published_at=NATIVE_PUBLISHED_AT,
            )
        self.assertEqual(provenance.upstream_tag, "v0.2.0")
        self.assertEqual(provenance.native_release_tag, "v0.2.0")
        self.assertEqual(
            provenance.native_manifest_sha256, hashlib.sha256(payload).hexdigest()
        )

    def test_malformed_native_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "assets.json"
            path.write_text('{"native_release_tag": "v0.2.0", "native_release_tag": "v0.2.0"}')
            with self.assertRaises(ContractError):
                native.scan_native_provenance(
                    manifest_path=path,
                    native_release_tag="v0.2.0",
                    bridge_source_sha=BRIDGE_SHA,
                    bridge_build_sha=BRIDGE_SHA,
                    channel="stable",
                    native_release_published_at=NATIVE_PUBLISHED_AT,
                )

    def test_channel_inconsistent_provenance_is_rejected(self) -> None:
        with self.assertRaises(ContractError):
            make_provenance(upstream_tag="b9165")
        with self.assertRaises(ContractError):
            make_provenance(native_release_tag="b9165")

    def test_native_release_timestamp_is_canonical_and_bounded(self) -> None:
        with self.assertRaises(ContractError):
            make_provenance(native_release_published_at="yesterday")


class DevelopmentScanTest(unittest.TestCase):
    """Manual development scans stay supported, and stay scan-only."""

    def _scan(self, channel: str) -> model.NativeProvenance:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "assets.json"
            path.write_bytes(
                json.dumps(
                    native_manifest(
                        native_release_tag="b9165",
                        llama_cpp_tag="b9165",
                    )
                ).encode("utf-8")
            )
            return native.scan_native_provenance(
                manifest_path=path,
                native_release_tag="b9165",
                bridge_source_sha=BRIDGE_SHA,
                bridge_build_sha=BRIDGE_SHA,
                channel=channel,
                native_release_published_at=NATIVE_PUBLISHED_AT,
            )

    def test_development_scan_still_prepares_exact_provenance(self) -> None:
        provenance = self._scan("development")
        self.assertEqual(provenance.upstream_tag, "b9165")
        self.assertEqual(provenance.native_release_tag, "b9165")
        self.assertIs(provenance.channel, Channel.DEVELOPMENT)

    def test_stable_scan_of_a_development_release_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            self._scan("stable")

    def test_development_scan_of_a_stable_release_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "assets.json"
            path.write_bytes(json.dumps(native_manifest()).encode("utf-8"))
            with self.assertRaises(ContractError):
                native.scan_native_provenance(
                    manifest_path=path,
                    native_release_tag="v0.2.0",
                    bridge_source_sha=BRIDGE_SHA,
                    bridge_build_sha=BRIDGE_SHA,
                    channel="development",
                    native_release_published_at=NATIVE_PUBLISHED_AT,
                )

    def test_orchestration_refuses_a_development_provenance(self) -> None:
        development = model.NativeProvenance(
            bridge_source_sha=BRIDGE_SHA,
            bridge_build_sha=BRIDGE_SHA,
            upstream_tag="b9165",
            upstream_commit=UPSTREAM_COMMIT,
            native_repo=NATIVE_REPOSITORY,
            native_release_tag="b9165",
            native_commit=NATIVE_COMMIT,
            native_manifest_sha256=NATIVE_MANIFEST_SHA,
            native_release_published_at=NATIVE_PUBLISHED_AT,
        )
        with self.assertRaises(ContractError):
            model.require_stable_provenance(development)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "provenance.json"
            path.write_text(
                json.dumps(
                    {
                        "bridge_source_sha": BRIDGE_SHA,
                        "bridge_build_sha": BRIDGE_SHA,
                        "upstream_tag": "b9165",
                        "upstream_commit": UPSTREAM_COMMIT,
                        "native_repo": NATIVE_REPOSITORY,
                        "native_release_tag": "b9165",
                        "native_commit": NATIVE_COMMIT,
                        "native_manifest_sha256": NATIVE_MANIFEST_SHA,
                        "native_release_published_at": NATIVE_PUBLISHED_AT,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ContractError):
                sro._load_provenance(path)


class StableNativeBacklogTest(unittest.TestCase):
    def release(
        self,
        tag: str,
        published_at: str,
        *,
        draft: bool = False,
        prerelease: bool | None = None,
    ) -> dict[str, Any]:
        if prerelease is None:
            prerelease = "-" in tag
        return {
            "tag_name": tag,
            "draft": draft,
            "prerelease": prerelease,
            "published_at": published_at,
        }

    def test_selects_every_post_baseline_stable_release_in_publication_order(self) -> None:
        releases = [
            self.release("v0.2.1", "2026-08-28T03:00:00Z"),
            self.release("b10599", "2026-08-28T02:00:00Z", prerelease=False),
            self.release("v0.2.0-1", native.STABLE_AUTOMATION_BASELINE_PUBLISHED_AT),
            self.release("v0.2.0-2", "2026-08-27T03:00:00Z"),
            self.release("v0.3.0", "2026-08-29T03:00:00Z", draft=True),
        ]
        self.assertEqual(
            native.select_stable_native_backlog(releases),
            ["v0.2.0-2", "v0.2.1"],
        )

    def test_post_baseline_stable_rollback_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            native.select_stable_native_backlog(
                [self.release("v0.1.99", "2026-08-28T03:00:00Z")]
            )

    def test_inconsistent_stable_prerelease_state_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            native.select_stable_native_backlog(
                [
                    self.release(
                        "v0.2.0-2",
                        "2026-08-28T03:00:00Z",
                        prerelease=False,
                    )
                ]
            )

    def test_duplicate_stable_tag_fails_closed(self) -> None:
        release = self.release("v0.2.1", "2026-08-28T03:00:00Z")
        with self.assertRaises(ContractError):
            native.select_stable_native_backlog([release, dict(release)])


if __name__ == "__main__":
    unittest.main()
