#!/usr/bin/env python3
"""Contract tests for output tag selection and still-claimed tags."""

from __future__ import annotations

import unittest
from typing import Any

import release_orchestrator_asset_releases as asset_releases
from release_orchestrator_fixtures_test import (
    BRIDGE_SHA,
    DEFAULT_BRANCH,
    asset_release_stub,
    make_legacy_v0140_provenance,
    make_provenance,
    run_payload,
    runs_response,
)
import release_orchestrator_model as model
import release_orchestrator_release_tags as release_tags
import release_orchestrator_run_names as run_names
import release_orchestrator_workflow_runs as workflow_runs


class ReleaseTargetTest(unittest.TestCase):
    def test_bridge_assets_version_independently_of_upstream(self) -> None:
        target = release_tags.select_next_release_target(
            ["v0.1.38", "v0.1.39"], upstream_tag="v0.2.0"
        )
        self.assertEqual(target.release_tag, "v0.1.40")
        self.assertEqual(target.release_rebuild, 0)

    def test_upstream_tag_is_never_used_as_the_output_tag(self) -> None:
        target = release_tags.select_next_release_target(
            ["v0.1.39"], upstream_tag="v0.2.0"
        )
        self.assertNotEqual(target.release_tag, "v0.2.0")

    def test_existing_tag_collision_selects_a_free_rebuild(self) -> None:
        target = release_tags.select_next_release_target(
            ["v0.1.39", "v0.1.40", "v0.1.40-1"], upstream_tag="v0.2.0"
        )
        self.assertEqual(target.release_tag, "v0.1.41")
        self.assertEqual(target.release_rebuild, 0)

    def test_draft_or_unparsable_tag_collision_is_avoided(self) -> None:
        target = release_tags.select_next_release_target(
            ["v0.1.39", "v0.1.40", "nightly-scratch"], upstream_tag="v0.2.0"
        )
        self.assertEqual(target.release_tag, "v0.1.41")

    def test_seed_release_when_no_stable_assets_exist(self) -> None:
        target = release_tags.select_next_release_target([], upstream_tag="v0.2.0")
        self.assertEqual(target.release_tag, release_tags.INITIAL_STABLE_RELEASE_TAG)
        self.assertEqual(target.release_rebuild, 0)

    def test_taken_next_version_moves_to_the_next_patch_version(self) -> None:
        target = release_tags.select_next_release_target(
            ["v0.1.39", "v0.1.40"], upstream_tag="v0.2.0", taken={"v0.1.41"}
        )
        self.assertEqual(target.release_tag, "v0.1.42")
        self.assertEqual(target.release_rebuild, 0)

    def test_claims_set_the_floor_so_tags_follow_dispatch_order(self) -> None:
        # v0.1.50 was claimed and released unpublished; v0.1.51 is still in
        # flight. A later pipeline must not take v0.1.50 below it.
        target = release_tags.select_next_release_target(
            ["v0.1.49"], upstream_tag="v0.5.0", taken={"v0.1.51"}
        )
        self.assertEqual(target.release_tag, "v0.1.52")
        self.assertEqual(target.release_rebuild, 0)

    def test_historical_rebuild_tag_sets_the_floor_but_is_never_emitted(self) -> None:
        target = release_tags.select_next_release_target(
            ["v0.1.47", "v0.1.47-1"], upstream_tag="v0.4.1", taken={"v0.1.48"}
        )
        self.assertEqual(target.release_tag, "v0.1.49")
        self.assertEqual(target.release_rebuild, 0)

    def test_seed_release_collision_moves_to_the_next_patch_version(self) -> None:
        target = release_tags.select_next_release_target(
            [], upstream_tag="v0.2.0", taken={release_tags.INITIAL_STABLE_RELEASE_TAG}
        )
        self.assertEqual(target.release_tag, "v0.1.1")
        self.assertEqual(target.release_rebuild, 0)

    def test_run_history_includes_claims_since_the_last_asset_publication(self) -> None:
        prior = asset_release_stub()
        prior["published_at"] = "2026-08-10T00:00:00Z"
        self.assertEqual(
            asset_releases.workflow_history_since([prior], make_provenance()),
            "2026-08-10T00:00:00Z",
        )


class ClaimedReleaseTagsTest(unittest.TestCase):
    OTHER_BUILD_SHA = "b" * 40

    def setUp(self) -> None:
        self.other = run_names.compute_correlation_id(
            make_provenance(bridge_build_sha=self.OTHER_BUILD_SHA)
        )
        self.current = run_names.compute_correlation_id(make_provenance())

    def _candidate_name(self, correlation_id: str, tag: str, rebuild: int) -> str:
        return run_names.candidate_run_name(
            correlation_id,
            model.PipelineBinding(
                bridge_source_sha=BRIDGE_SHA, release_tag=tag, release_rebuild=rebuild
            ),
        )

    def _records(
        self, path: str, *runs: tuple[str, str, str, str | None]
    ) -> list[Any]:
        return workflow_runs.parse_workflow_runs(
            runs_response(
                [
                    run_payload(
                        run_id=run_id, path=path, run_name=name,
                        status=status, conclusion=conclusion,
                    )
                    for run_id, name, status, conclusion in runs
                ]
            ),
            workflow_path=path,
            default_branch=DEFAULT_BRANCH,
        )

    def test_finished_claim_of_another_build_is_dropped(self) -> None:
        for conclusion in ("success", "failure"):
            with self.subTest(conclusion=conclusion):
                runs = self._records(
                    model.CANDIDATE_WORKFLOW_PATH,
                    ("501", self._candidate_name(self.other, "v0.1.40", 0),
                     "completed", conclusion),
                )
                self.assertTrue(
                    release_tags.has_other_build_claim(runs, bridge_build_sha=BRIDGE_SHA)
                )
                self.assertEqual(
                    release_tags.claimed_release_tags(runs, bridge_build_sha=BRIDGE_SHA),
                    set(),
                )

    def test_current_build_claim_is_kept(self) -> None:
        runs = self._records(
            model.CANDIDATE_WORKFLOW_PATH,
            ("501", self._candidate_name(self.current, "v0.1.40", 0),
             "completed", "failure"),
        )
        self.assertFalse(release_tags.has_other_build_claim(runs, bridge_build_sha=BRIDGE_SHA))
        self.assertEqual(
            release_tags.claimed_release_tags(runs, bridge_build_sha=BRIDGE_SHA),
            {"v0.1.40"},
        )

    def test_claim_without_a_build_identity_is_kept(self) -> None:
        legacy = run_names.compute_correlation_id(make_legacy_v0140_provenance())
        self.assertNotIn("-build-", legacy)
        runs = self._records(
            model.CANDIDATE_WORKFLOW_PATH,
            ("501", self._candidate_name(legacy, "v0.1.40", 0),
             "completed", "success"),
        )
        self.assertEqual(
            release_tags.claimed_release_tags(runs, bridge_build_sha=BRIDGE_SHA),
            {"v0.1.40"},
        )

    def test_in_flight_candidate_of_another_build_keeps_its_claim(self) -> None:
        runs = self._records(
            model.CANDIDATE_WORKFLOW_PATH,
            ("501", self._candidate_name(self.other, "v0.1.40", 0),
             "in_progress", None),
        )
        self.assertEqual(
            release_tags.claimed_release_tags(runs, bridge_build_sha=BRIDGE_SHA),
            {"v0.1.40"},
        )

    def test_in_flight_downstream_run_of_another_build_keeps_its_claim(self) -> None:
        candidates = self._records(
            model.CANDIDATE_WORKFLOW_PATH,
            ("501", self._candidate_name(self.other, "v0.1.40", 0),
             "completed", "success"),
        )
        binding = model.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )
        downstream = {
            model.QUALIFICATION_WORKFLOW_PATH: run_names.qualification_run_name(
                self.other, "501"
            ),
            model.PUBLISH_WORKFLOW_PATH: run_names.publish_run_name(
                self.other, "501", "601", binding
            ),
        }
        for path, name in downstream.items():
            with self.subTest(path=path):
                in_flight = self._records(path, ("701", name, "queued", None))
                self.assertEqual(
                    release_tags.claimed_release_tags(
                        candidates,
                        bridge_build_sha=BRIDGE_SHA,
                        downstream_runs=in_flight,
                    ),
                    {"v0.1.40"},
                )
                finished = self._records(path, ("701", name, "completed", "failure"))
                self.assertEqual(
                    release_tags.claimed_release_tags(
                        candidates,
                        bridge_build_sha=BRIDGE_SHA,
                        downstream_runs=finished,
                    ),
                    set(),
                )

    def test_satisfied_correlation_releases_its_claim(self) -> None:
        sibling = run_names.compute_correlation_id(
            make_provenance(native_release_tag="v0.2.1", native_manifest_sha256="f" * 64)
        )
        runs = self._records(
            model.CANDIDATE_WORKFLOW_PATH,
            ("501", self._candidate_name(self.current, "v0.1.40", 0),
             "completed", "success"),
            ("502", self._candidate_name(sibling, "v0.1.40-1", 1),
             "completed", "success"),
        )
        self.assertEqual(
            release_tags.claimed_release_tags(runs, bridge_build_sha=BRIDGE_SHA),
            {"v0.1.40", "v0.1.40-1"},
        )
        self.assertEqual(
            release_tags.claimed_release_tags(
                runs,
                bridge_build_sha=BRIDGE_SHA,
                satisfied_correlation_ids={self.current},
            ),
            {"v0.1.40-1"},
        )


if __name__ == "__main__":
    unittest.main()
