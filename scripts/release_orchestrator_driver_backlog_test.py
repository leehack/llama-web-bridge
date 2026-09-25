#!/usr/bin/env python3
"""Contract tests for ``advance_pipeline`` across an ordered native backlog."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from release_contract import ASSETS_REPOSITORY, BRIDGE_REPOSITORY, ContractError
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
    LEGACY_BRIDGE_SHA,
    LEGACY_CANDIDATE_RUN_ID,
    LEGACY_MANUAL_QUALIFICATION_GATES,
    LEGACY_MANUAL_UNPROVEN_CAPABILITIES,
    LEGACY_NATIVE_COMMIT,
    LEGACY_NATIVE_MANIFEST_SHA,
    LEGACY_UPSTREAM_COMMIT,
    QUALIFICATION_ARTIFACT_ID,
    QUALIFICATION_RUN_ID,
    aligned_release_stub,
    artifact_inventory,
    asset_release_stub,
    directory_members,
    flat_zip,
    make_legacy_v0140_provenance,
    make_provenance,
    release_payload,
    rewrite_legacy_candidate_manifest,
    run_payload,
    runs_response,
    write_bridge_candidate,
)
import release_orchestrator_model as model
import release_orchestrator_release_tags as release_tags
import release_orchestrator_run_names as run_names
import release_orchestrator_workflow_runs as workflow_runs
import stable_release_orchestrator as sro


class AdvancePipelineTest(AdvancePipelineFixture, unittest.TestCase):
    """Backlog order: publication barriers, superseded natives, and published noops."""

    def _published_candidate_gateway(
        self,
        candidate_dir: Path,
        *,
        provenance: model.NativeProvenance,
        correlation_id: str,
        published_manifest_compatibility: tuple[
            dict[str, str], dict[str, str]
        ]
        | None = None,
    ) -> FakeGateway:
        members = directory_members(candidate_dir)
        manifest = json.loads(
            (candidate_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if published_manifest_compatibility is None:
            fingerprint = rq.load_candidate(candidate_dir)[1]
        else:
            fingerprint = rq.load_published_candidate(
                candidate_dir,
                expected_qualification_gates=published_manifest_compatibility[0],
                expected_unproven_capabilities=published_manifest_compatibility[1],
            )[1]
        body = (
            f"Candidate fingerprint: `{fingerprint}`\n"
            f"Orchestrator correlation: `{correlation_id}`\n"
        )
        release = release_payload(tag="v0.1.40", body=body, members=members)
        routes = self._routes(releases=[release])
        routes.update(
            {
                f"repos/{BRIDGE_REPOSITORY}/compare/"
                f"{provenance.bridge_source_sha}...{DEFAULT_BRANCH}": {
                    "status": "ahead"
                },
                f"repos/{BRIDGE_REPOSITORY}/compare/"
                f"{manifest['bridge_commit']}...{DEFAULT_BRANCH}": {
                    "status": "ahead"
                },
                f"repos/{ASSETS_REPOSITORY}/git/ref/tags/v0.1.40": {
                    "ref": "refs/tags/v0.1.40",
                    "object": {"type": "commit", "sha": ASSETS_TAG_COMMIT},
                },
                f"repos/{ASSETS_REPOSITORY}/releases/tags/v0.1.40": release,
                f"repos/{ASSETS_REPOSITORY}/releases/{release['id']}": release,
            }
        )
        blobs = {
            f"repos/{ASSETS_REPOSITORY}/releases/assets/{asset['id']}": members[
                asset["name"]
            ]
            for asset in release["assets"]
        }
        digests = {
            name: hashlib.sha256(data).hexdigest() for name, data in members.items()
        }
        return FakeGateway(
            json_routes=routes,
            blob_routes=blobs,
            release_attestations={
                (ASSETS_REPOSITORY, "v0.1.40"): release_attestation(
                    release_tag="v0.1.40",
                    assets_repo=ASSETS_REPOSITORY,
                    tag_commit=ASSETS_TAG_COMMIT,
                    release_id=release["id"],
                    assets=digests,
                )
            },
        )

    def test_later_qualified_provenance_waits_for_earlier_immutable_publication(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="sro-backlog-order-") as temp:
            root = Path(temp)
            first = make_provenance()
            second = make_provenance(
                upstream_tag="v0.2.1",
                upstream_commit="d" * 40,
                native_release_tag="v0.2.1",
                native_commit="e" * 40,
                native_manifest_sha256="f" * 64,
                native_release_published_at="2026-08-21T12:34:56Z",
            )
            first_correlation = run_names.compute_correlation_id(first)
            second_correlation = run_names.compute_correlation_id(second)
            first_binding = model.PipelineBinding(BRIDGE_SHA, "v0.1.40", 0)
            second_binding = model.PipelineBinding(BRIDGE_SHA, "v0.1.40-1", 1)
            first_run_id = "4101"
            second_run_id = "4201"
            second_qualification_run_id = "4202"
            first_artifact_id = 31
            second_artifact_id = 32
            second_qualification_artifact_id = 33

            first_candidate = root / "first-candidate"
            second_candidate = root / "second-candidate"
            write_bridge_candidate(
                first_candidate,
                release_tag=first_binding.release_tag,
                release_rebuild=first_binding.release_rebuild,
                correlation_id=first_correlation,
                run_id=first_run_id,
            )
            write_bridge_candidate(
                second_candidate,
                release_tag=second_binding.release_tag,
                release_rebuild=second_binding.release_rebuild,
                correlation_id=second_correlation,
                run_id=second_run_id,
                upstream_tag=second.upstream_tag,
                upstream_commit=second.upstream_commit,
                native_release_tag=second.native_release_tag,
                native_manifest_sha256=second.native_manifest_sha256,
                native_commit=second.native_commit,
            )
            second_manifest, second_fingerprint = rq.load_candidate(second_candidate)
            second_qualification = rq.build_attestation(
                manifest=second_manifest,
                candidate_fingerprint=second_fingerprint,
                candidate_run_id=second_run_id,
                candidate_artifact_id=second_artifact_id,
                candidate_run_attempt=1,
                **qualification_identity(
                    qualification_run_id=second_qualification_run_id,
                    qualification_source_sha=HEAD_SHA,
                ),
                harness_digest=rq.harness_source_sha256(Path(__file__).resolve().parent),
                environment=qualification_environment(),
                speech_phase=speech_phase(),
                tts_phase=tts_phase(),
            )

            first_run = run_payload(
                run_id=first_run_id,
                path=model.CANDIDATE_WORKFLOW_PATH,
                run_name=run_names.candidate_run_name(first_correlation, first_binding),
            )
            second_run = run_payload(
                run_id=second_run_id,
                path=model.CANDIDATE_WORKFLOW_PATH,
                run_name=run_names.candidate_run_name(second_correlation, second_binding),
            )
            qualification_run = run_payload(
                run_id=second_qualification_run_id,
                path=model.QUALIFICATION_WORKFLOW_PATH,
                run_name=run_names.qualification_run_name(
                    second_correlation, second_run_id
                ),
            )
            releases = [asset_release_stub()]
            routes: dict[str, Any] = {
                f"repos/{ASSETS_REPOSITORY}/releases?per_page=100": [releases],
                f"repos/{BRIDGE_REPOSITORY}/compare/{BRIDGE_SHA}...{DEFAULT_BRANCH}": {
                    "status": "ahead"
                },
                f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}": {
                    "status": "identical"
                },
                f"repos/{BRIDGE_REPOSITORY}/actions/runs/{first_run_id}": first_run,
                f"repos/{BRIDGE_REPOSITORY}/actions/runs/{second_run_id}": second_run,
                f"repos/{BRIDGE_REPOSITORY}/actions/runs/{second_qualification_run_id}": qualification_run,
                f"repos/{BRIDGE_REPOSITORY}/actions/runs/{first_run_id}/artifacts?per_page=100": artifact_inventory(
                    run_id=first_run_id,
                    name=rq.CANDIDATE_ARTIFACT_NAME,
                    artifact_id=first_artifact_id,
                ),
                f"repos/{BRIDGE_REPOSITORY}/actions/runs/{second_run_id}/artifacts?per_page=100": artifact_inventory(
                    run_id=second_run_id,
                    name=rq.CANDIDATE_ARTIFACT_NAME,
                    artifact_id=second_artifact_id,
                ),
                f"repos/{BRIDGE_REPOSITORY}/actions/runs/{second_qualification_run_id}/artifacts?per_page=100": artifact_inventory(
                    run_id=second_qualification_run_id,
                    name=rq.ATTESTATION_ARTIFACT_NAME,
                    artifact_id=second_qualification_artifact_id,
                ),
            }
            candidate_runs = runs_response([first_run, second_run])
            qualification_runs = runs_response([qualification_run])
            publish_runs = runs_response([])
            for provenance in (first, second):
                since = asset_releases.workflow_history_since(releases, provenance)
                routes[
                    workflow_runs._workflow_runs_path(
                        workflow_file=model.CANDIDATE_WORKFLOW_FILE,
                        default_branch=DEFAULT_BRANCH,
                        created_since=since,
                    )
                ] = candidate_runs
                routes[
                    workflow_runs._workflow_runs_path(
                        workflow_file=model.QUALIFICATION_WORKFLOW_FILE,
                        default_branch=DEFAULT_BRANCH,
                        created_since=since,
                    )
                ] = qualification_runs
                routes[
                    workflow_runs._workflow_runs_path(
                        workflow_file=model.PUBLISH_WORKFLOW_FILE,
                        default_branch=DEFAULT_BRANCH,
                        created_since=since,
                    )
                ] = publish_runs
            gateway = FakeGateway(
                json_routes=routes,
                blob_routes={
                    f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{first_artifact_id}/zip": flat_zip(
                        directory_members(first_candidate)
                    ),
                    f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{second_artifact_id}/zip": flat_zip(
                        directory_members(second_candidate)
                    ),
                    f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{second_qualification_artifact_id}/zip": flat_zip(
                        {
                            "qualification-attestation.json": rq.canonical_json(
                                second_qualification
                            ).encode("utf-8")
                        }
                    ),
                },
            )
            first_qualification_run = run_payload(
                run_id="4201",
                path=model.QUALIFICATION_WORKFLOW_PATH,
                run_name=run_names.qualification_run_name(first_correlation, first_run_id),
                status="in_progress",
                conclusion=None,
            )
            qualification_route_keys = [
                workflow_runs._workflow_runs_path(
                    workflow_file=model.QUALIFICATION_WORKFLOW_FILE,
                    default_branch=DEFAULT_BRANCH,
                    created_since=asset_releases.workflow_history_since(releases, provenance),
                )
                for provenance in (first, second)
            ]
            original_dispatch = gateway.dispatch_workflow

            def dispatch(**kwargs: Any) -> None:
                original_dispatch(**kwargs)
                for key in qualification_route_keys:
                    gateway.json_routes[key] = runs_response(
                        [qualification_run, first_qualification_run]
                    )

            gateway.dispatch_workflow = dispatch  # type: ignore[assignment]

            provenance_list = root / "release-candidates.json"
            provenance_list.write_text(
                json.dumps(
                    [
                        sro._provenance_to_dict(first),
                        sro._provenance_to_dict(second),
                    ]
                ),
                encoding="utf-8",
            )
            output_plan = root / "orchestration-plan.json"
            environment = {
                "GITHUB_EVENT_NAME": "schedule",
                "GITHUB_ACTOR": "github-actions",
                "GITHUB_TRIGGERING_ACTOR": "github-actions",
            }
            with (
                mock.patch.object(sro, "GhGateway", return_value=gateway),
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch("sys.stdout", io.StringIO()),
            ):
                result = sro.main(
                    [
                        "orchestrate-backlog",
                        "--provenance-list-json",
                        str(provenance_list),
                        "--workspace",
                        str(root / "workspace"),
                        "--output-plan-json",
                        str(output_plan),
                    ]
                )
            self.assertEqual(result, 0)
            plan = json.loads(output_plan.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["action"] for item in plan["plans"]],
                [
                    model.OrchestrationAction.DISPATCH_QUALIFICATION.value,
                    model.OrchestrationAction.WAITING_FOR_PRIOR_PUBLICATION.value,
                ],
            )
            # The later provenance is fully qualified but must not publish before
            # the earlier one, so exactly one qualification dispatch happens and
            # no publication is dispatched.
            self.assertEqual(
                [record["workflow_file"] for record in gateway.dispatches],
                [model.QUALIFICATION_WORKFLOW_FILE],
            )

    def test_native_behind_the_published_alignment_is_superseded(self) -> None:
        prior_build = self._newer_native(bridge_build_sha=ADVANCED_BRIDGE_SHA)
        gateway = FakeGateway(
            json_routes=self._routes(
                releases=[asset_release_stub(), aligned_release_stub("v0.1.40", prior_build)]
            )
        )
        plan = driver.advance_pipeline(
            gateway,
            provenance=self.provenance,
            workspace=self.tmp,
            newer_native_scanned=True,
        )
        self.assertEqual(plan.action, model.OrchestrationAction.SUPERSEDED)
        self.assertIn("v0.2.1", plan.reason)
        self.assertIn("v0.1.40", plan.reason)
        self.assertIsNone(plan.release_target)
        self.assertEqual(gateway.dispatches, [])
        self.assertFalse(any("/actions/" in path for path in gateway.api_paths))

    def test_newest_scanned_native_is_never_skipped_as_superseded(self) -> None:
        prior_build = self._newer_native(bridge_build_sha=ADVANCED_BRIDGE_SHA)
        gateway = FakeGateway(
            json_routes=self._routes(
                releases=[aligned_release_stub("v0.1.40", prior_build)]
            )
        )
        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(plan.action, model.OrchestrationAction.BLOCKED)
        self.assertEqual(gateway.dispatches, [])
        result, backlog = self._run_backlog(gateway, [self.provenance])
        self.assertEqual(result, 1)
        self.assertEqual(
            [item["action"] for item in backlog["plans"]],
            [model.OrchestrationAction.BLOCKED.value],
        )

    def test_newest_native_stays_eligible_after_a_governed_bridge_change(self) -> None:
        prior_build = dataclasses.replace(
            self.provenance, bridge_build_sha=ADVANCED_BRIDGE_SHA
        )
        gateway = FakeGateway(
            json_routes=self._routes(
                releases=[aligned_release_stub("v0.1.40", prior_build)]
            )
        )
        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp, dry_run=True
        )
        self.assertEqual(plan.action, model.OrchestrationAction.DISPATCH_CANDIDATE)
        self.assertEqual(plan.release_target.release_tag, "v0.1.41")

    def test_native_rebuild_ahead_of_the_alignment_stays_eligible(self) -> None:
        rebuilt = make_provenance(native_release_tag="v0.2.0-1")
        gateway = FakeGateway(
            json_routes=self._routes(
                releases=[aligned_release_stub("v0.1.40", self.provenance)]
            )
        )
        plan = driver.advance_pipeline(
            gateway, provenance=rebuilt, workspace=self.tmp, dry_run=True
        )
        self.assertEqual(plan.action, model.OrchestrationAction.DISPATCH_CANDIDATE)

    def test_native_rebuild_behind_the_alignment_is_not_republished(self) -> None:
        rebuilt = make_provenance(
            native_release_tag="v0.2.0-1", bridge_build_sha=ADVANCED_BRIDGE_SHA
        )
        for newer_native_scanned, action in (
            (True, model.OrchestrationAction.SUPERSEDED),
            (False, model.OrchestrationAction.BLOCKED),
        ):
            with self.subTest(newer_native_scanned=newer_native_scanned):
                gateway = FakeGateway(
                    json_routes=self._routes(
                        releases=[aligned_release_stub("v0.1.40", rebuilt)]
                    )
                )
                plan = driver.advance_pipeline(
                    gateway,
                    provenance=self.provenance,
                    workspace=self.tmp,
                    newer_native_scanned=newer_native_scanned,
                )
                self.assertEqual(plan.action, action)
                self.assertIn("v0.2.0-1", plan.reason)
                self.assertEqual(gateway.dispatches, [])

    def test_first_publication_into_an_empty_assets_history(self) -> None:
        gateway = FakeGateway(json_routes=self._routes(releases=[]))
        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp, dry_run=True
        )
        self.assertEqual(plan.action, model.OrchestrationAction.DISPATCH_CANDIDATE)
        self.assertEqual(
            plan.release_target.release_tag, release_tags.INITIAL_STABLE_RELEASE_TAG
        )

    def test_published_older_native_stays_a_verified_noop(self) -> None:
        candidate_dir = self.tmp / "candidate"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
        )
        gateway = self._published_candidate_gateway(
            candidate_dir,
            provenance=self.provenance,
            correlation_id=self.correlation_id,
        )
        gateway.json_routes[f"repos/{ASSETS_REPOSITORY}/releases?per_page=100"][
            0
        ].append(aligned_release_stub("v0.1.41", self._newer_native()))
        plan = driver.advance_pipeline(
            gateway,
            provenance=self.provenance,
            workspace=self.tmp,
            newer_native_scanned=True,
        )
        self.assertEqual(plan.action, model.OrchestrationAction.NOOP)
        self.assertEqual(plan.release_target.release_tag, "v0.1.40")

    def test_backlog_skips_superseded_native_and_publishes_the_newest(self) -> None:
        older = self.provenance
        newest = self._newer_native()
        correlation = run_names.compute_correlation_id(newest)
        binding = model.PipelineBinding(BRIDGE_SHA, "v0.1.41", 0)
        releases = [
            aligned_release_stub(
                "v0.1.40",
                dataclasses.replace(newest, bridge_build_sha=ADVANCED_BRIDGE_SHA),
            )
        ]
        candidate_dir = self.tmp / "candidate-src"
        write_bridge_candidate(
            candidate_dir,
            release_tag=binding.release_tag,
            release_rebuild=binding.release_rebuild,
            correlation_id=correlation,
            upstream_tag=newest.upstream_tag,
            upstream_commit=newest.upstream_commit,
            native_release_tag=newest.native_release_tag,
            native_manifest_sha256=newest.native_manifest_sha256,
            native_commit=newest.native_commit,
        )
        manifest, fingerprint = rq.load_candidate(candidate_dir)
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
        candidate_run = run_payload(
            run_id=CANDIDATE_RUN_ID,
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name=run_names.candidate_run_name(correlation, binding),
        )
        qualification_run = run_payload(
            run_id=QUALIFICATION_RUN_ID,
            path=model.QUALIFICATION_WORKFLOW_PATH,
            run_name=run_names.qualification_run_name(correlation, CANDIDATE_RUN_ID),
        )
        publish_run = run_payload(
            run_id="701",
            path=model.PUBLISH_WORKFLOW_PATH,
            run_name=run_names.publish_run_name(
                correlation, CANDIDATE_RUN_ID, QUALIFICATION_RUN_ID, binding
            ),
            status="in_progress",
            conclusion=None,
        )
        runs = f"repos/{BRIDGE_REPOSITORY}/actions/runs"
        routes = self._routes(releases=releases)
        for workflow_file, recorded in (
            (model.CANDIDATE_WORKFLOW_FILE, [candidate_run]),
            (model.QUALIFICATION_WORKFLOW_FILE, [qualification_run]),
            (model.PUBLISH_WORKFLOW_FILE, []),
        ):
            routes[
                workflow_runs._workflow_runs_path(
                    workflow_file=workflow_file,
                    default_branch=DEFAULT_BRANCH,
                    created_since=asset_releases.workflow_history_since(releases, newest),
                )
            ] = runs_response(recorded)
        publish_readback_key = workflow_runs._workflow_runs_path(
            workflow_file=model.PUBLISH_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=newest.native_release_published_at,
        )
        routes[publish_readback_key] = runs_response([])
        routes.update(
            {
                f"{runs}/{CANDIDATE_RUN_ID}": candidate_run,
                f"{runs}/{QUALIFICATION_RUN_ID}": qualification_run,
                f"{runs}/{CANDIDATE_RUN_ID}/artifacts?per_page=100": artifact_inventory(
                    run_id=CANDIDATE_RUN_ID,
                    name=rq.CANDIDATE_ARTIFACT_NAME,
                    artifact_id=CANDIDATE_ARTIFACT_ID,
                ),
                f"{runs}/{QUALIFICATION_RUN_ID}/artifacts?per_page=100": artifact_inventory(
                    run_id=QUALIFICATION_RUN_ID,
                    name=rq.ATTESTATION_ARTIFACT_NAME,
                    artifact_id=QUALIFICATION_ARTIFACT_ID,
                ),
            }
        )
        artifacts = f"repos/{BRIDGE_REPOSITORY}/actions/artifacts"
        gateway = FakeGateway(
            json_routes=routes,
            blob_routes={
                f"{artifacts}/{CANDIDATE_ARTIFACT_ID}/zip": flat_zip(
                    directory_members(candidate_dir)
                ),
                f"{artifacts}/{QUALIFICATION_ARTIFACT_ID}/zip": flat_zip(
                    {
                        "qualification-attestation.json": rq.canonical_json(
                            attestation
                        ).encode("utf-8")
                    }
                ),
            },
        )
        original_dispatch = gateway.dispatch_workflow

        def dispatch(**kwargs: Any) -> None:
            original_dispatch(**kwargs)
            gateway.json_routes[publish_readback_key] = runs_response([publish_run])

        gateway.dispatch_workflow = dispatch  # type: ignore[assignment]

        result, backlog = self._run_backlog(gateway, [newest, older])
        self.assertEqual(result, 0)
        self.assertEqual(backlog["errors"], [])
        self.assertEqual(
            [
                (item["provenance"]["native_release_tag"], item["action"])
                for item in backlog["plans"]
            ],
            [
                ("v0.2.0", model.OrchestrationAction.SUPERSEDED.value),
                ("v0.2.1", model.OrchestrationAction.DISPATCH_PUBLISH.value),
            ],
        )
        self.assertEqual(
            [
                (record["workflow_file"], record["inputs"]["release_tag"])
                for record in gateway.dispatches
            ],
            [(model.PUBLISH_WORKFLOW_FILE, "v0.1.41")],
        )

    def test_backlog_dry_run_reserves_an_earlier_planned_output_tag(self) -> None:
        gateway = FakeGateway(
            json_routes=self._routes(releases=[asset_release_stub()])
        )
        plan = driver.advance_pipeline(
            gateway,
            provenance=self.provenance,
            workspace=self.tmp,
            dry_run=True,
            reserved_release_tags={"v0.1.40"},
        )
        self.assertEqual(plan.release_target.release_tag, "v0.1.41")

    def test_published_release_is_a_noop_with_no_dispatch(self) -> None:
        candidate_dir = self.tmp / "candidate"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
        )
        gateway = self._published_candidate_gateway(
            candidate_dir,
            provenance=self.provenance,
            correlation_id=self.correlation_id,
        )
        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(plan.action, model.OrchestrationAction.NOOP)
        self.assertEqual(gateway.dispatches, [])
        self.assertEqual(plan.release_target.release_tag, "v0.1.40")

    def test_legacy_v0140_publication_is_a_terminal_noop(self) -> None:
        provenance = make_legacy_v0140_provenance(
            bridge_source_sha=ADVANCED_BRIDGE_SHA
        )
        correlation_id = run_names.compute_correlation_id(provenance)
        candidate_dir = self.tmp / "legacy-v0140-candidate"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=correlation_id,
            bridge_commit=LEGACY_BRIDGE_SHA,
            run_id=LEGACY_CANDIDATE_RUN_ID,
            upstream_tag="v0.3.0",
            upstream_commit=LEGACY_UPSTREAM_COMMIT,
            native_release_tag="v0.3.0",
            native_manifest_sha256=LEGACY_NATIVE_MANIFEST_SHA,
            native_commit=LEGACY_NATIVE_COMMIT,
        )
        rewrite_legacy_candidate_manifest(candidate_dir)

        # The ordinary candidate path remains strict. Compatibility is selected
        # only after the immutable published-release identity is proven.
        with self.assertRaises(ContractError):
            rq.load_candidate(candidate_dir)
        gateway = self._published_candidate_gateway(
            candidate_dir,
            provenance=provenance,
            correlation_id=correlation_id,
            published_manifest_compatibility=(
                LEGACY_MANUAL_QUALIFICATION_GATES,
                LEGACY_MANUAL_UNPROVEN_CAPABILITIES,
            ),
        )
        plan = driver.advance_pipeline(
            gateway,
            provenance=provenance,
            workspace=self.tmp,
        )
        self.assertEqual(plan.action, model.OrchestrationAction.NOOP)
        self.assertEqual(gateway.dispatches, [])
        self.assertEqual(plan.release_target.release_tag, "v0.1.40")

    def test_governed_source_change_does_not_reuse_legacy_v0140(self) -> None:
        legacy = make_legacy_v0140_provenance(
            bridge_source_sha=ADVANCED_BRIDGE_SHA
        )
        legacy_correlation = run_names.compute_correlation_id(legacy)
        candidate_dir = self.tmp / "legacy-v0140-before-governed-change"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=legacy_correlation,
            bridge_commit=LEGACY_BRIDGE_SHA,
            run_id=LEGACY_CANDIDATE_RUN_ID,
            upstream_tag="v0.3.0",
            upstream_commit=LEGACY_UPSTREAM_COMMIT,
            native_release_tag="v0.3.0",
            native_manifest_sha256=LEGACY_NATIVE_MANIFEST_SHA,
            native_commit=LEGACY_NATIVE_COMMIT,
        )
        rewrite_legacy_candidate_manifest(candidate_dir)
        gateway = self._published_candidate_gateway(
            candidate_dir,
            provenance=legacy,
            correlation_id=legacy_correlation,
            published_manifest_compatibility=(
                LEGACY_MANUAL_QUALIFICATION_GATES,
                LEGACY_MANUAL_UNPROVEN_CAPABILITIES,
            ),
        )
        governed = make_legacy_v0140_provenance(
            bridge_source_sha=ADVANCED_BRIDGE_SHA,
            bridge_build_sha=ADVANCED_BRIDGE_SHA,
        )
        gateway.json_routes[
            workflow_runs._workflow_runs_path(
                workflow_file=model.CANDIDATE_WORKFLOW_FILE,
                default_branch=DEFAULT_BRANCH,
                created_since="2026-08-20T03:24:11Z",
            )
        ] = runs_response([])
        plan = driver.advance_pipeline(
            gateway,
            provenance=governed,
            workspace=self.tmp / "governed-change",
            dry_run=True,
        )
        self.assertEqual(plan.action, model.OrchestrationAction.DISPATCH_CANDIDATE)
        self.assertEqual(plan.release_target.release_tag, "v0.1.41")
        self.assertEqual(plan.dispatch_inputs["bridge_source_sha"], ADVANCED_BRIDGE_SHA)
        self.assertEqual(gateway.dispatches, [])

    def test_two_releases_claiming_one_correlation_fail_closed(self) -> None:
        body = f"Orchestrator correlation: `{self.correlation_id}`"
        releases = [
            {"tag_name": "v0.1.40", "draft": False, "prerelease": False, "body": body},
            {"tag_name": "v0.1.41", "draft": False, "prerelease": True, "body": body},
        ]
        gateway = FakeGateway(json_routes=self._routes(releases=releases))
        with self.assertRaises(ContractError):
            driver.advance_pipeline(
                gateway, provenance=self.provenance, workspace=self.tmp
            )


if __name__ == "__main__":
    unittest.main()
