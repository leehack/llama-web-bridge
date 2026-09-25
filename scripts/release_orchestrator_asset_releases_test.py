#!/usr/bin/env python3
"""Contract tests for asset release discovery and published release proofs."""

from __future__ import annotations

import dataclasses
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from generate_release_manifest import ARTIFACTS
from release_contract import ASSETS_REPOSITORY, ContractError, NATIVE_REPOSITORY
from release_publication_state import PUBLICATION_FILES, candidate_publication_digests
from release_contract_test import release_attestation
import release_qualification as rq
import release_orchestrator_asset_releases as asset_releases
from release_orchestrator_fixtures_test import (
    ADVANCED_BRIDGE_SHA,
    ASSETS_TAG_COMMIT,
    BRIDGE_SHA,
    FakeGateway,
    LEGACY_MANUAL_QUALIFICATION_GATES,
    LEGACY_MANUAL_UNPROVEN_CAPABILITIES,
    NATIVE_MANIFEST_SHA,
    aligned_release_stub,
    asset_release_stub,
    directory_members,
    make_legacy_v0140_provenance,
    make_provenance,
    release_payload,
    write_bridge_candidate,
)
import release_orchestrator_model as model
import release_orchestrator_run_names as run_names


class PublishedNativeAlignmentTest(unittest.TestCase):
    def release(self, tag: str, *native_tags: str, **fields: Any) -> dict[str, Any]:
        release = asset_release_stub(tag)
        release["body"] = "".join(
            f"Native: `{NATIVE_REPOSITORY}@{native_tag}`\r\n"
            for native_tag in native_tags
        )
        release.update(fields)
        return release

    def test_no_recorded_alignment_keeps_every_provenance_eligible(self) -> None:
        self.assertIsNone(asset_releases.latest_published_native_alignment([]))
        self.assertIsNone(
            asset_releases.latest_published_native_alignment(
                [asset_release_stub("v0.1.37"), {"tag_name": "v0.1.36", "body": None}]
            )
        )

    def test_newest_native_order_wins_regardless_of_listing_order(self) -> None:
        releases = [
            self.release("v0.1.44", "v0.4.1"),
            self.release("v0.1.45", "v0.4.1-1"),
            self.release("v0.1.42", "v0.3.0"),
        ]
        self.assertEqual(
            asset_releases.latest_published_native_alignment(releases), ("v0.4.1-1", "v0.1.45")
        )

    def test_only_published_stable_native_markers_are_evidence(self) -> None:
        foreign = asset_release_stub("v0.1.47")
        foreign["body"] = "Native: `someone/else@v9.9.9`\nsee Native: `x@v9.9.9`\n"
        releases = [
            self.release("v0.1.44", "v0.4.1"),
            self.release("v0.1.45", "v0.5.0", draft=True),
            self.release("v0.1.46", "v0.5.0", draft=None),
            self.release("b9165", "b9165", prerelease=True),
            self.release("b9170", "v0.5.0", prerelease=True),
            self.release("v0.1.48", "b9165"),
            self.release("not-a-release", "v0.5.0"),
            foreign,
        ]
        self.assertEqual(
            asset_releases.latest_published_native_alignment(releases), ("v0.4.1", "v0.1.44")
        )

    def test_contradictory_or_malformed_alignment_fails_closed(self) -> None:
        for release in (
            self.release("v0.1.44", "v0.4.0", "v0.4.1"),
            self.release("v0.1.44", "v0.4"),
            self.release("v0.1.44", "v0.4.1-llamadart.1"),
        ):
            with self.subTest(body=release["body"]):
                with self.assertRaises(ContractError):
                    asset_releases.latest_published_native_alignment([release])
        self.assertEqual(
            asset_releases.latest_published_native_alignment(
                [self.release("v0.1.44", "v0.4.1", "v0.4.1")]
            ),
            ("v0.4.1", "v0.1.44"),
        )


class IdenticalPublicationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sro-identical-"))
        self.provenance = make_provenance()
        self.prior = dataclasses.replace(
            self.provenance,
            bridge_source_sha=ADVANCED_BRIDGE_SHA,
            bridge_build_sha=ADVANCED_BRIDGE_SHA,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _digests(self, name: str, *, marker: bytes, tag: str, provenance: Any) -> dict[str, str]:
        directory = self.tmp / name
        write_bridge_candidate(
            directory,
            release_tag=tag,
            release_rebuild=0,
            correlation_id=run_names.compute_correlation_id(provenance),
            bridge_commit=provenance.bridge_source_sha,
            run_id=str(700 + len(name)),
            marker=marker,
        )
        return candidate_publication_digests(directory)

    def test_manifest_is_the_only_file_that_differs_between_identical_builds(self) -> None:
        published = self._digests("published", marker=b"same", tag="v0.1.41", provenance=self.prior)
        candidate = self._digests("candidate", marker=b"same", tag="v0.1.42", provenance=self.provenance)
        self.assertEqual(
            {name for name in PUBLICATION_FILES if published[name] != candidate[name]},
            {"manifest.json"},
        )
        self.assertTrue(asset_releases.publication_bytes_identical(candidate, published))

    def test_one_differing_artifact_is_not_identical(self) -> None:
        published = self._digests("published", marker=b"same", tag="v0.1.41", provenance=self.prior)
        candidate = self._digests("candidate", marker=b"changed", tag="v0.1.42", provenance=self.provenance)
        self.assertFalse(asset_releases.publication_bytes_identical(candidate, published))

    def test_incomplete_inventory_fails_closed(self) -> None:
        published = self._digests("published", marker=b"same", tag="v0.1.41", provenance=self.prior)
        candidate = self._digests("candidate", marker=b"same", tag="v0.1.42", provenance=self.provenance)
        for side in ("candidate", "published"):
            with self.subTest(side=side):
                partial = dict(candidate if side == "candidate" else published)
                partial.pop(ARTIFACTS[0])
                with self.assertRaises(ContractError):
                    asset_releases.publication_bytes_identical(
                        partial if side == "candidate" else candidate,
                        published if side == "candidate" else partial,
                    )

    def test_latest_aligned_release_is_the_newest_tag_for_this_native(self) -> None:
        other_native = make_provenance(
            native_release_tag="v0.2.1", native_manifest_sha256="f" * 64
        )
        releases = [
            aligned_release_stub("v0.1.41", self.prior),
            aligned_release_stub("v0.1.41-2", self.prior),
            aligned_release_stub("v0.1.42", other_native),
            aligned_release_stub("v0.1.41-1", self.prior),
        ]
        selected = asset_releases.latest_aligned_release(releases, self.provenance)
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0]["tag_name"], "v0.1.41-2")
        self.assertEqual(selected[1], run_names.compute_correlation_id(self.prior))
        self.assertIsNone(asset_releases.latest_aligned_release(releases[2:3], self.provenance))

    def test_latest_aligned_release_skips_drafts_and_ambiguous_correlations(self) -> None:
        draft = aligned_release_stub("v0.1.41", self.prior)
        draft["draft"] = True
        self.assertIsNone(asset_releases.latest_aligned_release([draft], self.provenance))
        ambiguous = aligned_release_stub("v0.1.41", self.prior)
        ambiguous["body"] += (
            f"Orchestrator correlation: `{run_names.compute_correlation_id(self.provenance)}`\n"
        )
        malformed = aligned_release_stub("v0.1.41-1", self.prior)
        malformed["body"] = malformed["body"].replace(
            run_names.compute_correlation_id(self.prior), "not a correlation"
        )
        older = aligned_release_stub("v0.1.41", self.prior)
        for releases in ([ambiguous], [malformed]):
            self.assertIsNone(asset_releases.latest_aligned_release(releases, self.provenance))
        selected = asset_releases.latest_aligned_release(
            [older, ambiguous, malformed], self.provenance
        )
        self.assertIs(selected[0], older)

    def test_legacy_v0140_is_comparable_only_by_its_own_provenance(self) -> None:
        legacy = make_legacy_v0140_provenance()
        release = aligned_release_stub("v0.1.40", legacy)
        self.assertIsNotNone(asset_releases.latest_aligned_release([release], legacy))
        governed = make_legacy_v0140_provenance(bridge_build_sha=ADVANCED_BRIDGE_SHA)
        self.assertIsNone(asset_releases.latest_aligned_release([release], governed))


class PublishedReleaseVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sro-published-"))
        self.provenance = make_provenance()
        self.correlation_id = run_names.compute_correlation_id(self.provenance)
        self.candidate = self.tmp / "candidate"
        write_bridge_candidate(
            self.candidate,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
        )
        self.members = directory_members(self.candidate)
        self.fingerprint = rq.load_candidate(self.candidate)[1]
        self.body = (
            f"Candidate fingerprint: `{self.fingerprint}`\n"
            f"Orchestrator correlation: `{self.correlation_id}`\n"
        )
        self.release = release_payload(
            tag="v0.1.40", body=self.body, members=self.members
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _gateway_for_release(
        self,
        release: dict[str, Any],
        *,
        release_by_id: dict[str, Any] | None = None,
        tag_commit: str = ASSETS_TAG_COMMIT,
        attestation: Any = None,
    ) -> FakeGateway:
        blobs = {
            f"repos/{ASSETS_REPOSITORY}/releases/assets/{asset['id']}": self.members[
                asset["name"]
            ]
            for asset in release["assets"]
            if asset["name"] in self.members
        }
        tag = release["tag_name"]
        release_id = release["id"]
        digests = {
            name: hashlib.sha256(data).hexdigest()
            for name, data in self.members.items()
        }
        routes = {
            f"repos/{ASSETS_REPOSITORY}/git/ref/tags/{tag}": {
                "ref": f"refs/tags/{tag}",
                "object": {"type": "commit", "sha": tag_commit},
            },
            f"repos/{ASSETS_REPOSITORY}/releases/tags/{tag}": release,
            f"repos/{ASSETS_REPOSITORY}/releases/{release_id}": (
                release_by_id or release
            ),
        }
        attestations = {
            (ASSETS_REPOSITORY, tag): attestation
            or release_attestation(
                release_tag=tag,
                assets_repo=ASSETS_REPOSITORY,
                tag_commit=tag_commit,
                release_id=release_id,
                assets=digests,
            )
        }
        return FakeGateway(
            json_routes=routes,
            blob_routes=blobs,
            release_attestations=attestations,
        )

    def _verify(self, release: dict[str, Any], **kwargs: Any) -> Any:
        gateway = self._gateway_for_release(release, **kwargs)
        return asset_releases.verify_published_release(
            gateway,
            release=release,
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            workspace=self.tmp / "workspace",
        )

    def test_exact_immutable_publication_is_accepted(self) -> None:
        verified = self._verify(self.release)
        self.assertEqual(verified.release_target.release_tag, "v0.1.40")
        self.assertEqual(verified.release_target.release_rebuild, 0)
        self.assertEqual(verified.binding.bridge_source_sha, BRIDGE_SHA)

    def test_only_exact_v0140_identity_selects_legacy_manifest_contract(self) -> None:
        exact = make_legacy_v0140_provenance()
        self.assertEqual(
            model._published_manifest_compatibility(
                tag="v0.1.40",
                provenance=exact,
            ),
            (
                LEGACY_MANUAL_QUALIFICATION_GATES,
                LEGACY_MANUAL_UNPROVEN_CAPABILITIES,
            ),
        )
        near_misses = {
            "release tag": ("v0.1.40-1", exact),
            "native tag": (
                "v0.1.40",
                make_legacy_v0140_provenance(native_release_tag="v0.3.0-1"),
            ),
            "native commit": (
                "v0.1.40",
                make_legacy_v0140_provenance(native_commit="1" * 40),
            ),
            "upstream tag": (
                "v0.1.40",
                make_legacy_v0140_provenance(upstream_tag="v0.3.1"),
            ),
            "upstream commit": (
                "v0.1.40",
                make_legacy_v0140_provenance(upstream_commit="2" * 40),
            ),
            "governed bridge build": (
                "v0.1.40",
                make_legacy_v0140_provenance(bridge_build_sha="3" * 40),
            ),
            "native manifest": (
                "v0.1.40",
                make_legacy_v0140_provenance(native_manifest_sha256="0" * 64),
            ),
        }
        for label, (tag, provenance) in near_misses.items():
            with self.subTest(label=label):
                self.assertIsNone(
                    model._published_manifest_compatibility(
                        tag=tag,
                        provenance=provenance,
                    )
                )

    def test_native_manifest_marker_prevents_duplicate_when_correlation_is_damaged(
        self,
    ) -> None:
        malformed = dict(
            self.release,
            body=(
                f"Native: `{NATIVE_REPOSITORY}@v0.2.0`\n"
                f"Native manifest SHA-256: `{NATIVE_MANIFEST_SHA}`\n"
            ),
        )
        self.assertIs(
            asset_releases.find_correlated_release(
                [malformed], self.correlation_id, self.provenance
            ),
            malformed,
        )
        with self.assertRaises(ContractError):
            self._verify(malformed)

    def test_manifest_digest_without_exact_native_tag_cannot_hijack_state(self) -> None:
        foreign = dict(
            self.release,
            body=(
                f"Native: `{NATIVE_REPOSITORY}@v9.9.9`\n"
                f"Native manifest SHA-256: `{NATIVE_MANIFEST_SHA}`\n"
            ),
        )
        self.assertIsNone(
            asset_releases.find_correlated_release(
                [foreign], self.correlation_id, self.provenance
            )
        )

    def test_mutable_release_fails_closed(self) -> None:
        release = dict(self.release, immutable=False)
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_missing_immutability_field_fails_closed(self) -> None:
        release = dict(self.release)
        del release["immutable"]
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_missing_published_at_fails_closed(self) -> None:
        release = dict(self.release, published_at=None)
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_draft_release_fails_closed(self) -> None:
        release = dict(self.release, draft=True)
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_incomplete_asset_inventory_fails_closed(self) -> None:
        release = dict(self.release)
        release["assets"] = [
            asset for asset in release["assets"] if asset["name"] != "sha256sums.txt"
        ]
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_unexpected_extra_asset_fails_closed(self) -> None:
        release = dict(self.release)
        release["assets"] = release["assets"] + [
            {
                "id": 999,
                "name": "extra.bin",
                "state": "uploaded",
                "size": 1,
                "digest": f"sha256:{'0' * 64}",
            }
        ]
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_digest_mismatch_fails_closed(self) -> None:
        release = release_payload(
            tag="v0.1.40",
            body=self.body,
            members=self.members,
            asset_overrides={"manifest.json": {"digest": f"sha256:{'0' * 64}"}},
        )
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_manifest_bytes_that_do_not_bind_provenance_fail_closed(self) -> None:
        with self.assertRaises(ContractError):
            asset_releases.verify_published_release(
                self._gateway_for_release(self.release),
                release=self.release,
                provenance=make_provenance(native_commit="d" * 40),
                correlation_id=self.correlation_id,
                workspace=self.tmp / "workspace",
            )

    def test_release_tag_commit_is_resolved_independently(self) -> None:
        with self.assertRaises(ContractError):
            self._verify(self.release, tag_commit="d" * 40)

    def test_release_readback_by_id_must_match(self) -> None:
        by_id = dict(self.release, body="unrelated")
        with self.assertRaises(ContractError):
            self._verify(self.release, release_by_id=by_id)

    def test_signed_release_attestation_must_bind_every_asset(self) -> None:
        invalid = release_attestation(
            release_tag="v0.1.40",
            assets_repo=ASSETS_REPOSITORY,
            tag_commit=ASSETS_TAG_COMMIT,
            release_id=self.release["id"],
            assets={"manifest.json": "0" * 64},
        )
        with self.assertRaises(ContractError):
            self._verify(self.release, attestation=invalid)

    def test_release_body_without_the_candidate_fingerprint_fails_closed(self) -> None:
        # The publication contract binds the release body to the exact candidate
        # digest. A body that only names the correlation proves nothing about
        # which bytes were published.
        release = release_payload(
            tag="v0.1.40",
            body=f"Orchestrator correlation: `{self.correlation_id}`\n",
            members=self.members,
        )
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_release_body_with_a_foreign_candidate_fingerprint_fails_closed(self) -> None:
        release = release_payload(
            tag="v0.1.40",
            body=(
                f"Candidate fingerprint: `{'0' * 64}`\n"
                f"Orchestrator correlation: `{self.correlation_id}`\n"
            ),
            members=self.members,
        )
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_release_tag_that_contradicts_the_published_manifest_fails_closed(
        self,
    ) -> None:
        release = release_payload(
            tag="v0.1.41", body=self.body, members=self.members
        )
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_publication_from_an_earlier_bridge_source_stays_a_noop(self) -> None:
        # Main advances daily; the published release stays bound to the exact
        # candidate source it was built from, not to today's HEAD.
        verified = self._verify(self.release)
        self.assertEqual(verified.binding.bridge_source_sha, BRIDGE_SHA)
        advanced = asset_releases.verify_published_release(
            self._gateway_for_release(self.release),
            release=self.release,
            provenance=make_provenance(bridge_source_sha=ADVANCED_BRIDGE_SHA),
            correlation_id=self.correlation_id,
            workspace=self.tmp / "workspace",
        )
        self.assertEqual(advanced.binding.bridge_source_sha, BRIDGE_SHA)


if __name__ == "__main__":
    unittest.main()
