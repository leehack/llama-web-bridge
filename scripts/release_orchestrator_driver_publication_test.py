#!/usr/bin/env python3
"""Contract tests for the qualification and publication pipeline stages."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from release_contract import BRIDGE_REPOSITORY, ContractError

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
import release_orchestrator_driver as driver
from release_orchestrator_fixtures_test import (
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
    run_payload,
    runs_response,
    write_bridge_candidate,
)
import release_orchestrator_model as model
import release_orchestrator_run_names as run_names
import release_orchestrator_workflow_runs as workflow_runs


class AdvancePipelineTest(AdvancePipelineFixture, unittest.TestCase):
    """Qualification and publication stages after a proven candidate."""

    def test_candidate_success_dispatches_hosted_qualification(self) -> None:
        candidate_dir = self.tmp / "candidate-src"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
            run_id=CANDIDATE_RUN_ID,
        )
        members = directory_members(candidate_dir)
        succeeded = run_payload(
            run_id=CANDIDATE_RUN_ID,
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
        )
        routes = self._routes(candidate_runs=[succeeded])
        routes[f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}"] = succeeded
        routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=CANDIDATE_RUN_ID,
            name=rq.CANDIDATE_ARTIFACT_NAME,
            artifact_id=CANDIDATE_ARTIFACT_ID,
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}"] = {
            "status": "identical"
        }
        routes[f"repos/{BRIDGE_REPOSITORY}/compare/{BRIDGE_SHA}...{DEFAULT_BRANCH}"] = {
            "status": "ahead"
        }
        blobs = {
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{CANDIDATE_ARTIFACT_ID}/zip": flat_zip(
                members
            )
        }
        gateway = FakeGateway(json_routes=routes, blob_routes=blobs)
        dispatched = run_payload(
            run_id="4201",
            path=model.QUALIFICATION_WORKFLOW_PATH,
            run_name=run_names.qualification_run_name(self.correlation_id, CANDIDATE_RUN_ID),
            status="in_progress",
            conclusion=None,
        )
        readback_key = workflow_runs._workflow_runs_path(
            workflow_file=model.QUALIFICATION_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        original_dispatch = gateway.dispatch_workflow

        def dispatch(**kwargs: Any) -> None:
            original_dispatch(**kwargs)
            gateway.json_routes[readback_key] = runs_response([dispatched])

        gateway.dispatch_workflow = dispatch  # type: ignore[assignment]
        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(
            plan.action, model.OrchestrationAction.DISPATCH_QUALIFICATION
        )
        self.assertEqual(plan.candidate_run_id, CANDIDATE_RUN_ID)
        self.assertEqual(plan.release_target.release_tag, "v0.1.40")
        self.assertEqual(len(gateway.dispatches), 1)
        record = gateway.dispatches[0]
        self.assertEqual(record["workflow_file"], model.QUALIFICATION_WORKFLOW_FILE)
        self.assertEqual(record["ref"], DEFAULT_BRANCH)
        self.assertEqual(
            record["inputs"],
            {
                "orchestrator_correlation_id": self.correlation_id,
                "candidate_run_id": CANDIDATE_RUN_ID,
            },
        )

    def test_failed_qualification_blocks_instead_of_retrying_forever(self) -> None:
        candidate_dir = self.tmp / "candidate-failed-qualification"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
            run_id=CANDIDATE_RUN_ID,
        )
        succeeded = run_payload(
            run_id=CANDIDATE_RUN_ID,
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
        )
        failed_qualification = run_payload(
            run_id="4201",
            path=model.QUALIFICATION_WORKFLOW_PATH,
            run_name=run_names.qualification_run_name(self.correlation_id, CANDIDATE_RUN_ID),
            status="completed",
            conclusion="failure",
        )
        routes = self._routes(
            candidate_runs=[succeeded], qualification_runs=[failed_qualification]
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}"] = succeeded
        routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=CANDIDATE_RUN_ID,
            name=rq.CANDIDATE_ARTIFACT_NAME,
            artifact_id=CANDIDATE_ARTIFACT_ID,
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}"] = {
            "status": "identical"
        }
        gateway = FakeGateway(
            json_routes=routes,
            blob_routes={
                f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{CANDIDATE_ARTIFACT_ID}/zip": flat_zip(
                    directory_members(candidate_dir)
                )
            },
        )
        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(plan.action, model.OrchestrationAction.BLOCKED)
        self.assertEqual(gateway.dispatches, [])
        self.assertIn("automatic qualification retries are disabled", plan.reason)

    def test_full_pipeline_dispatches_publish_after_exact_attestation(self) -> None:
        candidate_dir = self.tmp / "candidate-src"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
            run_id=CANDIDATE_RUN_ID,
        )
        members = directory_members(candidate_dir)
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
        attestation_bytes = rq.canonical_json(attestation).encode("utf-8")

        candidate_run = run_payload(
            run_id=CANDIDATE_RUN_ID,
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
        )
        qualification_run = run_payload(
            run_id=QUALIFICATION_RUN_ID,
            path=model.QUALIFICATION_WORKFLOW_PATH,
            run_name=run_names.qualification_run_name(
                self.correlation_id, CANDIDATE_RUN_ID
            ),
        )
        routes = self._routes(
            candidate_runs=[candidate_run], qualification_runs=[qualification_run]
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}"] = candidate_run
        routes[f"repos/{BRIDGE_REPOSITORY}/actions/runs/{QUALIFICATION_RUN_ID}"] = qualification_run
        routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=CANDIDATE_RUN_ID,
            name=rq.CANDIDATE_ARTIFACT_NAME,
            artifact_id=CANDIDATE_ARTIFACT_ID,
        )
        routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{QUALIFICATION_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=QUALIFICATION_RUN_ID,
            name=rq.ATTESTATION_ARTIFACT_NAME,
            artifact_id=QUALIFICATION_ARTIFACT_ID,
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}"] = {
            "status": "identical"
        }
        blobs = {
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{CANDIDATE_ARTIFACT_ID}/zip": flat_zip(
                members
            ),
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{QUALIFICATION_ARTIFACT_ID}/zip": flat_zip(
                {"qualification-attestation.json": attestation_bytes}
            ),
        }
        gateway = FakeGateway(json_routes=routes, blob_routes=blobs)
        publish_name = run_names.publish_run_name(
            self.correlation_id,
            CANDIDATE_RUN_ID,
            QUALIFICATION_RUN_ID,
            self.binding,
        )
        publish_run = run_payload(
            run_id="701",
            path=model.PUBLISH_WORKFLOW_PATH,
            run_name=publish_name,
            status="in_progress",
            conclusion=None,
        )
        publish_key = workflow_runs._workflow_runs_path(
            workflow_file=model.PUBLISH_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        original_dispatch = gateway.dispatch_workflow

        def dispatch(**kwargs: Any) -> None:
            original_dispatch(**kwargs)
            gateway.json_routes[publish_key] = runs_response([publish_run])

        gateway.dispatch_workflow = dispatch  # type: ignore[assignment]

        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(plan.action, model.OrchestrationAction.DISPATCH_PUBLISH)
        self.assertEqual(len(gateway.dispatches), 1)
        inputs = gateway.dispatches[0]["inputs"]
        self.assertEqual(inputs["candidate_run_id"], CANDIDATE_RUN_ID)
        self.assertEqual(inputs["qualification_run_id"], QUALIFICATION_RUN_ID)
        self.assertEqual(inputs["release_tag"], "v0.1.40")
        self.assertEqual(inputs["bridge_source_sha"], BRIDGE_SHA)
        self.assertEqual(inputs["publish_approved"], "true")
        self.assertEqual(gateway.dispatches[0]["ref"], DEFAULT_BRANCH)
        self.assertEqual(plan.dispatched_run_id, "701")
        qualification_reachability_path = (
            f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}"
        )
        self.assertEqual(gateway.api_paths.count(qualification_reachability_path), 2)

    def test_attestation_bound_to_another_candidate_fails_closed(self) -> None:
        candidate_dir = self.tmp / "candidate-src"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
            run_id=CANDIDATE_RUN_ID,
        )
        members = directory_members(candidate_dir)
        other_dir = self.tmp / "other-src"
        write_bridge_candidate(
            other_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
            run_id=CANDIDATE_RUN_ID,
            marker=b"tampered",
        )
        other_manifest, other_fingerprint = rq.load_candidate(other_dir)
        attestation = rq.build_attestation(
            manifest=other_manifest,
            candidate_fingerprint=other_fingerprint,
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
        attestation_bytes = rq.canonical_json(attestation).encode("utf-8")
        candidate_run = run_payload(
            run_id=CANDIDATE_RUN_ID,
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
        )
        qualification_run = run_payload(
            run_id=QUALIFICATION_RUN_ID,
            path=model.QUALIFICATION_WORKFLOW_PATH,
            run_name=run_names.qualification_run_name(
                self.correlation_id, CANDIDATE_RUN_ID
            ),
        )
        routes = self._routes(
            candidate_runs=[candidate_run], qualification_runs=[qualification_run]
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}"] = candidate_run
        routes[f"repos/{BRIDGE_REPOSITORY}/actions/runs/{QUALIFICATION_RUN_ID}"] = qualification_run
        routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=CANDIDATE_RUN_ID,
            name=rq.CANDIDATE_ARTIFACT_NAME,
            artifact_id=CANDIDATE_ARTIFACT_ID,
        )
        routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{QUALIFICATION_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=QUALIFICATION_RUN_ID,
            name=rq.ATTESTATION_ARTIFACT_NAME,
            artifact_id=QUALIFICATION_ARTIFACT_ID,
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}"] = {
            "status": "identical"
        }
        blobs = {
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{CANDIDATE_ARTIFACT_ID}/zip": flat_zip(
                members
            ),
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{QUALIFICATION_ARTIFACT_ID}/zip": flat_zip(
                {"qualification-attestation.json": attestation_bytes}
            ),
        }
        gateway = FakeGateway(json_routes=routes, blob_routes=blobs)
        with self.assertRaises(ContractError):
            driver.advance_pipeline(
                gateway, provenance=self.provenance, workspace=self.tmp
            )
        self.assertEqual(gateway.dispatches, [])


if __name__ == "__main__":
    unittest.main()
