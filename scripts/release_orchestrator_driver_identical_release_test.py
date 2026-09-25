#!/usr/bin/env python3
"""Contract tests for candidates satisfied by an identical published release."""

from __future__ import annotations

import dataclasses
import hashlib
import unittest
from pathlib import Path
from typing import Any

from release_contract import ASSETS_REPOSITORY, BRIDGE_REPOSITORY
from release_contract_test import release_attestation

# The heavy attestation fixtures are pinned by the qualification suite that
# produces them. Restating them here would let this suite pass against an
# attestation the real gate would reject.
from release_qualification_test import (
    qualification_environment,
    qualification_identity,
    speech_phase,
    tts_phase,
)
import release_qualification as rq
import release_orchestrator_asset_releases as asset_releases
import release_orchestrator_driver as driver
from release_orchestrator_fixtures_test import (
    ADVANCED_BRIDGE_SHA,
    ASSETS_TAG_COMMIT,
    AdvancePipelineFixture,
    BRIDGE_SHA,
    CANDIDATE_ARTIFACT_ID,
    CANDIDATE_RUN_ID,
    DEFAULT_BRANCH,
    FakeGateway,
    HEAD_SHA,
    NATIVE_PUBLISHED_AT,
    QUALIFICATION_ARTIFACT_ID,
    QUALIFICATION_RUN_ID,
    artifact_inventory,
    directory_members,
    flat_zip,
    make_provenance,
    release_payload,
    run_payload,
    runs_response,
    write_bridge_candidate,
)
import release_orchestrator_model as model
import release_orchestrator_run_names as run_names
import release_orchestrator_workflow_runs as workflow_runs


class AdvancePipelineTest(AdvancePipelineFixture, unittest.TestCase):
    """Candidates whose publication bytes match an aligned immutable release."""

    def _aligned_release(
        self,
        published_dir: Path,
        *,
        tag: str,
        provenance: model.NativeProvenance,
        correlation_id: str,
        release_id: int = 4343,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes], dict[tuple[str, str], Any]]:
        """A verifiable immutable release carrying the deterministic notes."""
        members = directory_members(published_dir)
        fingerprint = rq.load_candidate(published_dir)[1]
        body = (
            f"Candidate fingerprint: `{fingerprint}`\n"
            f"Native: `{provenance.native_repo}@{provenance.native_release_tag}`\n"
            f"Native manifest SHA-256: `{provenance.native_manifest_sha256}`\n"
            f"Orchestrator correlation: `{correlation_id}`\n"
        )
        release = release_payload(
            tag=tag, body=body, members=members, release_id=release_id
        )
        routes = {
            f"repos/{ASSETS_REPOSITORY}/git/ref/tags/{tag}": {
                "ref": f"refs/tags/{tag}",
                "object": {"type": "commit", "sha": ASSETS_TAG_COMMIT},
            },
            f"repos/{ASSETS_REPOSITORY}/releases/tags/{tag}": release,
            f"repos/{ASSETS_REPOSITORY}/releases/{release_id}": release,
        }
        blobs = {
            f"repos/{ASSETS_REPOSITORY}/releases/assets/{asset['id']}": members[
                asset["name"]
            ]
            for asset in release["assets"]
        }
        attestations = {
            (ASSETS_REPOSITORY, tag): release_attestation(
                release_tag=tag,
                assets_repo=ASSETS_REPOSITORY,
                tag_commit=ASSETS_TAG_COMMIT,
                release_id=release_id,
                assets={
                    name: hashlib.sha256(data).hexdigest()
                    for name, data in members.items()
                },
            )
        }
        return release, routes, blobs, attestations

    def _proven_candidate(
        self,
        candidate_dir: Path,
        *,
        marker: bytes,
        release_tag: str,
        provenance: model.NativeProvenance,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
        """A succeeded candidate run whose artifact is downloadable and proven."""
        correlation_id = run_names.compute_correlation_id(provenance)
        write_bridge_candidate(
            candidate_dir,
            release_tag=release_tag,
            release_rebuild=0,
            correlation_id=correlation_id,
            run_id=CANDIDATE_RUN_ID,
            marker=marker,
            upstream_tag=provenance.upstream_tag,
            upstream_commit=provenance.upstream_commit,
            native_release_tag=provenance.native_release_tag,
            native_manifest_sha256=provenance.native_manifest_sha256,
            native_commit=provenance.native_commit,
        )
        run = run_payload(
            run_id=CANDIDATE_RUN_ID,
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name=run_names.candidate_run_name(
                correlation_id, model.PipelineBinding(BRIDGE_SHA, release_tag, 0)
            ),
        )
        runs = f"repos/{BRIDGE_REPOSITORY}/actions/runs"
        routes = {
            f"{runs}/{CANDIDATE_RUN_ID}": run,
            f"{runs}/{CANDIDATE_RUN_ID}/artifacts?per_page=100": artifact_inventory(
                run_id=CANDIDATE_RUN_ID,
                name=rq.CANDIDATE_ARTIFACT_NAME,
                artifact_id=CANDIDATE_ARTIFACT_ID,
            ),
        }
        blobs = {
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{CANDIDATE_ARTIFACT_ID}/zip": (
                flat_zip(directory_members(candidate_dir))
            )
        }
        return run, routes, blobs

    def _aligned_candidate_gateway(
        self,
        *,
        candidate_marker: bytes,
        provenance: model.NativeProvenance | None = None,
        sibling_runs: list[dict[str, Any]] | None = None,
    ) -> tuple[FakeGateway, dict[str, Any], dict[str, Any]]:
        """``v0.1.41`` publishes ``b"same"`` bytes for native ``v0.2.0`` under an
        earlier build identity; the scanned provenance has a proven candidate
        bound to ``v0.1.42``."""
        provenance = provenance or self.provenance
        prior = dataclasses.replace(
            self.provenance,
            bridge_source_sha=ADVANCED_BRIDGE_SHA,
            bridge_build_sha=ADVANCED_BRIDGE_SHA,
        )
        prior_correlation = run_names.compute_correlation_id(prior)
        published_dir = self.tmp / "published-src"
        write_bridge_candidate(
            published_dir,
            release_tag="v0.1.41",
            release_rebuild=0,
            correlation_id=prior_correlation,
            bridge_commit=ADVANCED_BRIDGE_SHA,
            run_id="777",
            marker=b"same",
        )
        release, release_routes, release_blobs, attestations = self._aligned_release(
            published_dir,
            tag="v0.1.41",
            provenance=prior,
            correlation_id=prior_correlation,
        )
        run, run_routes, run_blobs = self._proven_candidate(
            self.tmp / "candidate-src",
            marker=candidate_marker,
            release_tag="v0.1.42",
            provenance=provenance,
        )
        routes = self._routes(
            releases=[release], candidate_runs=[run, *(sibling_runs or [])]
        )
        routes.update(release_routes)
        routes.update(run_routes)
        gateway = FakeGateway(
            json_routes=routes,
            blob_routes={**release_blobs, **run_blobs},
            release_attestations=attestations,
        )
        return gateway, release, run

    def _expect_qualification_dispatch(
        self, gateway: FakeGateway, provenance: model.NativeProvenance
    ) -> None:
        correlation_id = run_names.compute_correlation_id(provenance)
        readback_key = workflow_runs._workflow_runs_path(
            workflow_file=model.QUALIFICATION_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        dispatched = run_payload(
            run_id="4201",
            path=model.QUALIFICATION_WORKFLOW_PATH,
            run_name=run_names.qualification_run_name(correlation_id, CANDIDATE_RUN_ID),
            status="in_progress",
            conclusion=None,
        )
        original_dispatch = gateway.dispatch_workflow

        def dispatch(**kwargs: Any) -> None:
            original_dispatch(**kwargs)
            gateway.json_routes[readback_key] = runs_response([dispatched])

        gateway.dispatch_workflow = dispatch  # type: ignore[assignment]

    def test_identical_candidate_is_satisfied_by_the_aligned_release(self) -> None:
        gateway, release, _ = self._aligned_candidate_gateway(candidate_marker=b"same")
        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(
            plan.action, model.OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE
        )
        self.assertEqual(plan.release_target.release_tag, "v0.1.41")
        self.assertEqual(plan.candidate_run_id, CANDIDATE_RUN_ID)
        self.assertIsNone(plan.dispatch_workflow)
        self.assertIn("v0.1.42", plan.reason)
        self.assertEqual(gateway.dispatches, [])
        self.assertIn(
            f"repos/{ASSETS_REPOSITORY}/releases/{release['id']}", gateway.api_paths
        )
        self.assertIsNone(plan.qualification_run_id)
        self.assertIn("other than manifest.json", plan.reason)
        self.assertEqual(plan.to_dict()["action"], "satisfied_by_identical_release")

    def _qualify_candidate(
        self, gateway: FakeGateway, *, publish_runs: tuple[dict[str, Any], ...] = ()
    ) -> None:
        """Record a succeeded, attested qualification of the aligned fixture's
        candidate and the given publication runs."""
        manifest, fingerprint = rq.load_candidate(self.tmp / "candidate-src")
        attestation = rq.build_attestation(
            manifest=manifest,
            candidate_fingerprint=fingerprint,
            candidate_run_id=CANDIDATE_RUN_ID,
            candidate_artifact_id=CANDIDATE_ARTIFACT_ID,
            candidate_run_attempt=1,
            **qualification_identity(
                qualification_run_id=QUALIFICATION_RUN_ID,
                qualification_source_sha=HEAD_SHA,
            ),
            harness_digest=rq.harness_source_sha256(Path(__file__).resolve().parent),
            environment=qualification_environment(),
            speech_phase=speech_phase(),
            tts_phase=tts_phase(),
        )
        qualification_run = run_payload(
            run_id=QUALIFICATION_RUN_ID,
            path=model.QUALIFICATION_WORKFLOW_PATH,
            run_name=run_names.qualification_run_name(self.correlation_id, CANDIDATE_RUN_ID),
        )
        runs = f"repos/{BRIDGE_REPOSITORY}/actions/runs"
        gateway.json_routes[f"{runs}/{QUALIFICATION_RUN_ID}"] = qualification_run
        gateway.json_routes[
            f"{runs}/{QUALIFICATION_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=QUALIFICATION_RUN_ID,
            name=rq.ATTESTATION_ARTIFACT_NAME,
            artifact_id=QUALIFICATION_ARTIFACT_ID,
        )
        for workflow_file, listed in (
            (model.QUALIFICATION_WORKFLOW_FILE, [qualification_run]),
            (model.PUBLISH_WORKFLOW_FILE, list(publish_runs)),
        ):
            gateway.json_routes[
                workflow_runs._workflow_runs_path(
                    workflow_file=workflow_file,
                    default_branch=DEFAULT_BRANCH,
                    created_since=NATIVE_PUBLISHED_AT,
                )
            ] = runs_response(listed)
        gateway.blob_routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{QUALIFICATION_ARTIFACT_ID}/zip"
        ] = flat_zip(
            {"qualification-attestation.json": rq.canonical_json(attestation).encode("utf-8")}
        )

    def _publish_run(self, *, status: str, conclusion: str | None) -> dict[str, Any]:
        return run_payload(
            run_id="701",
            path=model.PUBLISH_WORKFLOW_PATH,
            run_name=run_names.publish_run_name(
                self.correlation_id,
                CANDIDATE_RUN_ID,
                QUALIFICATION_RUN_ID,
                model.PipelineBinding(BRIDGE_SHA, "v0.1.42", 0),
            ),
            status=status,
            conclusion=conclusion,
        )

    def test_qualified_identical_candidate_is_satisfied_before_publication(self) -> None:
        gateway, release, _ = self._aligned_candidate_gateway(candidate_marker=b"same")
        self._qualify_candidate(gateway)
        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(
            plan.action, model.OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE
        )
        self.assertEqual(plan.release_target.release_tag, "v0.1.41")
        self.assertEqual(plan.qualification_run_id, QUALIFICATION_RUN_ID)
        self.assertEqual(gateway.dispatches, [])
        self.assertIn(
            f"repos/{ASSETS_REPOSITORY}/releases/{release['id']}", gateway.api_paths
        )

    def test_failed_publication_of_an_identical_candidate_is_satisfied_not_retried(
        self,
    ) -> None:
        gateway, _, _ = self._aligned_candidate_gateway(candidate_marker=b"same")
        self._qualify_candidate(
            gateway, publish_runs=(self._publish_run(status="completed", conclusion="failure"),)
        )
        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(
            plan.action, model.OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE
        )
        self.assertEqual(gateway.dispatches, [])

    def test_identical_candidate_with_qualification_in_flight_is_not_compared(
        self,
    ) -> None:
        gateway, release, _ = self._aligned_candidate_gateway(candidate_marker=b"same")
        gateway.json_routes[
            workflow_runs._workflow_runs_path(
                workflow_file=model.QUALIFICATION_WORKFLOW_FILE,
                default_branch=DEFAULT_BRANCH,
                created_since=NATIVE_PUBLISHED_AT,
            )
        ] = runs_response(
            [
                run_payload(
                    run_id="4201",
                    path=model.QUALIFICATION_WORKFLOW_PATH,
                    run_name=run_names.qualification_run_name(
                        self.correlation_id, CANDIDATE_RUN_ID
                    ),
                    status="in_progress",
                    conclusion=None,
                )
            ]
        )
        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(plan.action, model.OrchestrationAction.IN_FLIGHT)
        self.assertEqual(plan.in_flight_workflow, model.QUALIFICATION_WORKFLOW_FILE)
        self.assertEqual(plan.release_target.release_tag, "v0.1.42")
        self.assertFalse(
            any(f"/releases/{release['id']}" in path for path in gateway.api_paths)
        )

    def test_identical_candidate_with_publication_in_flight_keeps_its_tag(
        self,
    ) -> None:
        gateway, release, run = self._aligned_candidate_gateway(candidate_marker=b"same")
        self._qualify_candidate(
            gateway, publish_runs=(self._publish_run(status="in_progress", conclusion=None),)
        )
        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(plan.action, model.OrchestrationAction.IN_FLIGHT)
        self.assertEqual(plan.in_flight_workflow, model.PUBLISH_WORKFLOW_FILE)
        self.assertEqual(plan.release_target.release_tag, "v0.1.42")
        self.assertFalse(
            any(f"/releases/{release['id']}" in path for path in gateway.api_paths)
        )

        newer = self._route_newer_native(gateway, release, run)
        result, backlog = self._run_backlog(gateway, [self.provenance, newer])
        self.assertEqual(backlog["errors"], [])
        self.assertEqual(result, 0)
        self.assertEqual(
            [
                (item["provenance"]["native_release_tag"], item["action"], item["release_tag"])
                for item in backlog["plans"]
            ],
            [
                ("v0.2.0", "in_flight", "v0.1.42"),
                ("v0.2.1", "dispatch_candidate", "v0.1.43"),
            ],
        )
        self.assertEqual(
            [
                (record["workflow_file"], record["inputs"]["release_tag"], record["inputs"]["release_rebuild"])
                for record in gateway.dispatches
            ],
            [(model.CANDIDATE_WORKFLOW_FILE, "v0.1.43", "0")],
        )

    def test_differing_candidate_is_qualified_not_skipped(self) -> None:
        gateway, _, _ = self._aligned_candidate_gateway(candidate_marker=b"changed")
        self._expect_qualification_dispatch(gateway, self.provenance)
        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(plan.action, model.OrchestrationAction.DISPATCH_QUALIFICATION)
        self.assertEqual(plan.release_target.release_tag, "v0.1.42")
        self.assertEqual(
            [record["workflow_file"] for record in gateway.dispatches],
            [model.QUALIFICATION_WORKFLOW_FILE],
        )

    def test_identical_bytes_under_another_native_alignment_are_not_compared(self) -> None:
        rebuilt = make_provenance(native_release_tag="v0.2.0-1")
        gateway, release, _ = self._aligned_candidate_gateway(
            candidate_marker=b"same", provenance=rebuilt
        )
        self._expect_qualification_dispatch(gateway, rebuilt)
        plan = driver.advance_pipeline(gateway, provenance=rebuilt, workspace=self.tmp)
        self.assertEqual(plan.action, model.OrchestrationAction.DISPATCH_QUALIFICATION)
        self.assertFalse(
            any(f"/releases/{release['id']}" in path for path in gateway.api_paths)
        )
        self.assertEqual(
            [record["workflow_file"] for record in gateway.dispatches],
            [model.QUALIFICATION_WORKFLOW_FILE],
        )

    def _route_newer_native(
        self, gateway: FakeGateway, release: dict[str, Any], run: dict[str, Any]
    ) -> model.NativeProvenance:
        """Route a backlog scan of native ``v0.2.1`` that sees ``run`` as the only
        candidate and lists the candidate it dispatches on readback."""
        newer = self._newer_native()
        newer_since = asset_releases.workflow_history_since([release], newer)
        self.assertNotEqual(newer_since, NATIVE_PUBLISHED_AT)
        for workflow_file in (
            model.CANDIDATE_WORKFLOW_FILE,
            model.QUALIFICATION_WORKFLOW_FILE,
            model.PUBLISH_WORKFLOW_FILE,
        ):
            gateway.json_routes[
                workflow_runs._workflow_runs_path(
                    workflow_file=workflow_file,
                    default_branch=DEFAULT_BRANCH,
                    created_since=newer_since,
                )
            ] = runs_response([run] if workflow_file == model.CANDIDATE_WORKFLOW_FILE else [])
        readback_key = workflow_runs._workflow_runs_path(
            workflow_file=model.CANDIDATE_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=newer.native_release_published_at,
        )
        gateway.json_routes[readback_key] = runs_response([run])
        original_dispatch = gateway.dispatch_workflow

        def dispatch(**kwargs: Any) -> None:
            original_dispatch(**kwargs)
            gateway.json_routes[readback_key] = runs_response(
                [
                    run,
                    run_payload(
                        run_id="501",
                        path=model.CANDIDATE_WORKFLOW_PATH,
                        run_name=(
                            f"bridge-candidate {run_names.compute_correlation_id(newer)}"
                            f" source:{BRIDGE_SHA}"
                            f" tag:{kwargs['inputs']['release_tag']}"
                            f" rebuild:{kwargs['inputs']['release_rebuild']}"
                        ),
                        status="in_progress",
                        conclusion=None,
                    ),
                ]
            )

        gateway.dispatch_workflow = dispatch  # type: ignore[assignment]
        return newer

    def test_satisfied_correlation_stays_satisfied_and_frees_its_tag_on_later_scans(
        self,
    ) -> None:
        gateway, release, run = self._aligned_candidate_gateway(candidate_marker=b"same")
        for scan in (1, 2):
            with self.subTest(scan=scan):
                workspace = self.tmp / f"scan-{scan}"
                workspace.mkdir()
                plan = driver.advance_pipeline(
                    gateway, provenance=self.provenance, workspace=workspace
                )
                self.assertEqual(
                    plan.action,
                    model.OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE,
                )
        self.assertEqual(gateway.dispatches, [])
        self.assertFalse(
            any(model.PUBLISH_WORKFLOW_FILE in path for path in gateway.api_paths)
        )

        newer = self._route_newer_native(gateway, release, run)
        result, backlog = self._run_backlog(gateway, [self.provenance, newer])
        self.assertEqual(backlog["errors"], [])
        self.assertEqual(result, 0)
        self.assertEqual(
            [
                (item["provenance"]["native_release_tag"], item["action"], item["release_tag"])
                for item in backlog["plans"]
            ],
            [
                ("v0.2.0", "satisfied_by_identical_release", "v0.1.41"),
                ("v0.2.1", "dispatch_candidate", "v0.1.42"),
            ],
        )
        self.assertEqual(
            [
                (record["workflow_file"], record["inputs"]["release_tag"], record["inputs"]["release_rebuild"])
                for record in gateway.dispatches
            ],
            [(model.CANDIDATE_WORKFLOW_FILE, "v0.1.42", "0")],
        )

    def test_identical_candidate_still_publishes_when_a_sibling_claims_its_rebuild(
        self,
    ) -> None:
        sibling = run_payload(
            run_id="502",
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name=run_names.candidate_run_name(
                run_names.compute_correlation_id(self._newer_native()),
                model.PipelineBinding(BRIDGE_SHA, "v0.1.42-1", 1),
            ),
        )
        gateway, release, _ = self._aligned_candidate_gateway(
            candidate_marker=b"same", sibling_runs=[sibling]
        )
        self._expect_qualification_dispatch(gateway, self.provenance)
        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(plan.action, model.OrchestrationAction.DISPATCH_QUALIFICATION)
        self.assertFalse(
            any(f"/releases/{release['id']}" in path for path in gateway.api_paths)
        )


if __name__ == "__main__":
    unittest.main()
