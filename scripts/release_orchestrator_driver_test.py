#!/usr/bin/env python3
"""Contract tests for the live driver and the candidate stage.

Covers governance and dispatch identity, and the candidate stage of
``advance_pipeline``. The other stages are in
``release_orchestrator_driver_<stage>_test.py``; every suite keeps the
original ``AdvancePipelineTest`` class name.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from release_contract import ASSETS_REPOSITORY, BRIDGE_REPOSITORY, ContractError
import release_qualification as rq
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
    QUALIFICATION_RUN_ID,
    artifact_inventory,
    asset_release_stub,
    directory_members,
    flat_zip,
    make_provenance,
    run_payload,
    runs_response,
    write_bridge_candidate,
)
import release_orchestrator_model as model
import release_orchestrator_planner as planner
import release_orchestrator_run_names as run_names
import release_orchestrator_workflow_runs as workflow_runs


class GovernanceAndDispatchIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.provenance = make_provenance()
        self.correlation_id = run_names.compute_correlation_id(self.provenance)

    def _gateway(self, **kwargs: Any) -> FakeGateway:
        routes = {
            f"repos/{ASSETS_REPOSITORY}/releases?per_page=100": [[]],
            workflow_runs._workflow_runs_path(
                workflow_file=model.CANDIDATE_WORKFLOW_FILE,
                default_branch=DEFAULT_BRANCH,
                created_since=NATIVE_PUBLISHED_AT,
            ): runs_response([]),
        }
        routes.update(kwargs.pop("json_routes", {}))
        return FakeGateway(json_routes=routes, **kwargs)

    def test_disabled_immutable_release_governance_fails_closed(self) -> None:
        gateway = self._gateway(governance={"enabled": False, "enforced_by_owner": False})
        with self.assertRaises(ContractError):
            driver.require_immutable_release_governance(gateway)

    def test_malformed_governance_response_fails_closed(self) -> None:
        gateway = self._gateway(governance={"enabled": True})
        with self.assertRaises(ContractError):
            driver.require_immutable_release_governance(gateway)

    def test_live_governance_is_proven_not_asserted(self) -> None:
        gateway = self._gateway()
        proven = driver.require_immutable_release_governance(gateway)
        self.assertEqual(proven["repository"], ASSETS_REPOSITORY)
        self.assertIs(proven["enabled"], True)

    def test_absent_dispatch_identity_blocks_without_dispatching(self) -> None:
        gateway = self._gateway(identity=None)
        with tempfile.TemporaryDirectory() as workspace:
            plan = driver.advance_pipeline(
                gateway,
                provenance=self.provenance,
                workspace=Path(workspace),
            )
        self.assertEqual(plan.action, model.OrchestrationAction.BLOCKED)
        self.assertEqual(gateway.dispatches, [])
        self.assertIn("dispatch identity", plan.reason.lower())

    def test_non_owner_dispatch_identity_blocks_without_dispatching(self) -> None:
        gateway = self._gateway(identity="someone-else")
        with tempfile.TemporaryDirectory() as workspace:
            plan = driver.advance_pipeline(
                gateway,
                provenance=self.provenance,
                workspace=Path(workspace),
            )
        self.assertEqual(plan.action, model.OrchestrationAction.BLOCKED)
        self.assertEqual(gateway.dispatches, [])

    def test_publish_approval_requires_live_environment_policy(self) -> None:
        binding = model.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )
        plan = planner.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=model.PipelineObservation(
                binding=binding,
                candidate_run_id=CANDIDATE_RUN_ID,
                qualification_run_id=QUALIFICATION_RUN_ID,
            ),
        )
        gateway = self._gateway()
        gateway.json_routes[
            f"repos/{BRIDGE_REPOSITORY}/environments/bridge-assets-publication"
        ] = {
            "name": "bridge-assets-publication",
            "can_admins_bypass": True,
            "protection_rules": [{"type": "branch_policy"}],
            "deployment_branch_policy": {
                "protected_branches": False,
                "custom_branch_policies": True,
            },
        }
        with self.assertRaises(ContractError):
            driver._execute_dispatch(
                gateway,
                plan,
                default_branch=DEFAULT_BRANCH,
                workflow_path=model.PUBLISH_WORKFLOW_PATH,
                dry_run=True,
            )
        self.assertEqual(gateway.dispatches, [])


class AdvancePipelineTest(AdvancePipelineFixture, unittest.TestCase):
    """Candidate stage: first dispatch, output tag claims, failures, and readback."""

    def test_first_run_dispatches_exactly_one_candidate_with_structured_inputs(self) -> None:
        dispatched = run_payload(
            run_id="501",
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
            status="in_progress",
            conclusion=None,
        )
        routes = self._routes(
            releases=[asset_release_stub()]
        )
        gateway = FakeGateway(json_routes=routes)
        expected_binding = model.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )
        readback_key = workflow_runs._workflow_runs_path(
            workflow_file=model.CANDIDATE_WORKFLOW_FILE,
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
        self.assertEqual(plan.action, model.OrchestrationAction.DISPATCH_CANDIDATE)
        self.assertEqual(len(gateway.dispatches), 1)
        dispatch_record = gateway.dispatches[0]
        self.assertEqual(dispatch_record["workflow_file"], model.CANDIDATE_WORKFLOW_FILE)
        self.assertEqual(dispatch_record["ref"], DEFAULT_BRANCH)
        self.assertEqual(dispatch_record["inputs"]["release_tag"], "v0.1.40")
        self.assertTrue(
            all(isinstance(value, str) for value in dispatch_record["inputs"].values())
        )
        self.assertEqual(plan.dispatched_run_id, "501")
        self.assertEqual(
            run_names.candidate_run_name(self.correlation_id, expected_binding),
            self.candidate_name,
        )

    def _dispatch_with_other_build_claim(
        self, *, qualification_status: str
    ) -> FakeGateway:
        other = run_names.compute_correlation_id(make_provenance(bridge_build_sha="b" * 40))
        orphan = run_payload(
            run_id="401",
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name=run_names.candidate_run_name(other, self.binding),
        )
        qualification = run_payload(
            run_id="402",
            path=model.QUALIFICATION_WORKFLOW_PATH,
            run_name=run_names.qualification_run_name(other, "401"),
            status=qualification_status,
            conclusion=None if qualification_status != "completed" else "failure",
        )
        gateway = FakeGateway(
            json_routes=self._routes(
                releases=[asset_release_stub()],
                candidate_runs=[orphan],
                qualification_runs=[qualification],
            )
        )
        readback_key = workflow_runs._workflow_runs_path(
            workflow_file=model.CANDIDATE_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        original_dispatch = gateway.dispatch_workflow

        def dispatch(**kwargs: Any) -> None:
            original_dispatch(**kwargs)
            gateway.json_routes[readback_key] = runs_response(
                [
                    orphan,
                    run_payload(
                        run_id="501",
                        path=model.CANDIDATE_WORKFLOW_PATH,
                        run_name=(
                            f"bridge-candidate {self.correlation_id}"
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
        driver.advance_pipeline(gateway, provenance=self.provenance, workspace=self.tmp)
        return gateway

    def test_finished_pipeline_of_another_build_does_not_force_a_rebuild_tag(
        self,
    ) -> None:
        gateway = self._dispatch_with_other_build_claim(
            qualification_status="completed"
        )
        inputs = gateway.dispatches[0]["inputs"]
        self.assertEqual(inputs["release_tag"], "v0.1.40")
        self.assertEqual(inputs["release_rebuild"], "0")

    def test_in_flight_pipeline_of_another_build_still_reserves_its_tag(self) -> None:
        gateway = self._dispatch_with_other_build_claim(
            qualification_status="in_progress"
        )
        inputs = gateway.dispatches[0]["inputs"]
        self.assertEqual(inputs["release_tag"], "v0.1.41")
        self.assertEqual(inputs["release_rebuild"], "0")

    def test_in_flight_candidate_produces_no_second_dispatch(self) -> None:
        in_flight = run_payload(
            run_id="501",
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
            status="in_progress",
            conclusion=None,
        )
        gateway = FakeGateway(json_routes=self._routes(candidate_runs=[in_flight]))
        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(plan.action, model.OrchestrationAction.IN_FLIGHT)
        self.assertEqual(gateway.dispatches, [])

    def test_failed_candidate_is_terminal_without_daily_duplicate(self) -> None:
        failed = run_payload(
            run_id="501",
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
            status="completed",
            conclusion="failure",
        )
        # Main advancing cannot silently turn the same native provenance into a
        # fresh candidate attempt.
        advanced = make_provenance(bridge_source_sha=ADVANCED_BRIDGE_SHA)
        routes = self._routes(
            releases=[asset_release_stub()],
            candidate_runs=[failed],
        )
        gateway = FakeGateway(json_routes=routes)
        plan = driver.advance_pipeline(
            gateway, provenance=advanced, workspace=self.tmp
        )
        self.assertEqual(plan.action, model.OrchestrationAction.BLOCKED)
        self.assertEqual(gateway.dispatches, [])
        self.assertIn("automatic candidate retries are disabled", plan.reason)

    def test_deliberate_success_after_failed_candidate_recovers_pipeline(self) -> None:
        candidate_dir = self.tmp / "candidate-after-failure"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
            run_id=CANDIDATE_RUN_ID,
        )
        failed = run_payload(
            run_id="501",
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
            status="completed",
            conclusion="failure",
        )
        succeeded = run_payload(
            run_id=CANDIDATE_RUN_ID,
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
        )
        routes = self._routes(candidate_runs=[failed, succeeded])
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
        self.assertEqual(plan.dispatched_run_id, "4201")
        self.assertEqual(
            [record["workflow_file"] for record in gateway.dispatches],
            [model.QUALIFICATION_WORKFLOW_FILE],
        )

    def test_dry_run_plans_without_dispatching(self) -> None:
        routes = self._routes(
            releases=[asset_release_stub()]
        )
        gateway = FakeGateway(json_routes=routes)
        plan = driver.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp, dry_run=True
        )
        self.assertEqual(plan.action, model.OrchestrationAction.DISPATCH_CANDIDATE)
        self.assertEqual(gateway.dispatches, [])
        self.assertIsNone(plan.dispatched_run_id)
        # A dry run must still report the exact ref and inputs it would use.
        self.assertEqual(plan.dispatch_ref, DEFAULT_BRANCH)
        self.assertEqual(
            plan.dispatch_inputs["assets_immutable_releases_enabled"], "true"
        )

    def test_orphan_assets_tag_ref_is_reserved_before_candidate_dispatch(self) -> None:
        routes = self._routes(releases=[asset_release_stub()])
        routes[
            f"repos/{ASSETS_REPOSITORY}/git/matching-refs/tags?per_page=100"
        ] = [
            [
                {
                    "ref": "refs/tags/v0.1.40",
                    "object": {"type": "commit", "sha": ASSETS_TAG_COMMIT},
                }
            ]
        ]
        gateway = FakeGateway(json_routes=routes)
        plan = driver.advance_pipeline(
            gateway,
            provenance=self.provenance,
            workspace=self.tmp,
            dry_run=True,
        )
        self.assertEqual(plan.release_target.release_tag, "v0.1.41")
        self.assertEqual(plan.release_target.release_rebuild, 0)

    def test_candidate_dispatch_inputs_carry_the_live_governance_proof(self) -> None:
        dispatched = run_payload(
            run_id="501",
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
            status="in_progress",
            conclusion=None,
        )
        routes = self._routes(
            releases=[asset_release_stub()]
        )
        gateway = FakeGateway(json_routes=routes)
        readback_key = workflow_runs._workflow_runs_path(
            workflow_file=model.CANDIDATE_WORKFLOW_FILE,
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
        inputs = gateway.dispatches[0]["inputs"]
        self.assertEqual(set(inputs), set(planner.CANDIDATE_DISPATCH_INPUTS))
        self.assertEqual(inputs["assets_immutable_releases_enabled"], "true")
        self.assertEqual(plan.dispatch_ref, DEFAULT_BRANCH)

    def test_duplicate_candidate_artifacts_fail_closed(self) -> None:
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
            extra=[
                {
                    "id": 8,
                    "name": rq.CANDIDATE_ARTIFACT_NAME,
                    "expired": False,
                    "workflow_run": {"id": int(CANDIDATE_RUN_ID)},
                }
            ],
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}"] = {
            "status": "identical"
        }
        blobs = {
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{CANDIDATE_ARTIFACT_ID}/zip": flat_zip(
                members
            )
        }
        gateway = FakeGateway(json_routes=routes, blob_routes=blobs)
        with self.assertRaises(ContractError):
            driver.advance_pipeline(
                gateway, provenance=self.provenance, workspace=self.tmp
            )

    def _assert_candidate_manifest_binding_mismatch(
        self, **candidate_overrides: Any
    ) -> None:
        candidate_dir = self.tmp / "candidate-src"
        candidate_fields: dict[str, Any] = {
            "release_tag": "v0.1.40",
            "release_rebuild": 0,
            "correlation_id": self.correlation_id,
            "run_id": CANDIDATE_RUN_ID,
        }
        candidate_fields.update(candidate_overrides)
        write_bridge_candidate(candidate_dir, **candidate_fields)
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
        blobs = {
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{CANDIDATE_ARTIFACT_ID}/zip": flat_zip(
                members
            )
        }
        gateway = FakeGateway(json_routes=routes, blob_routes=blobs)
        with self.assertRaises(ContractError):
            driver.advance_pipeline(
                gateway, provenance=self.provenance, workspace=self.tmp
            )

    def test_candidate_manifest_source_that_contradicts_run_name_fails_closed(
        self,
    ) -> None:
        self._assert_candidate_manifest_binding_mismatch(
            bridge_commit=ADVANCED_BRIDGE_SHA
        )

    def test_candidate_manifest_tag_that_contradicts_run_name_fails_closed(
        self,
    ) -> None:
        self._assert_candidate_manifest_binding_mismatch(release_tag="v0.1.41")

    def test_post_dispatch_readback_absence_fails_closed(self) -> None:
        routes = self._routes(
            releases=[asset_release_stub()]
        )
        gateway = FakeGateway(json_routes=routes)
        with self.assertRaises(ContractError):
            driver.advance_pipeline(
                gateway, provenance=self.provenance, workspace=self.tmp
            )
        self.assertEqual(len(gateway.dispatches), 1)

    def test_advance_pipeline_fails_closed_on_malformed_correlated_candidate_run(
        self,
    ) -> None:
        malformed_run = run_payload(
            run_id="501",
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name=(
                f"bridge-candidate {self.correlation_id} source:not-a-sha "
                "tag:v0.1.40 rebuild:0"
            ),
            status="in_progress",
            conclusion=None,
        )
        routes = self._routes(candidate_runs=[malformed_run])
        gateway = FakeGateway(json_routes=routes)
        with self.assertRaises(ContractError):
            driver.advance_pipeline(
                gateway, provenance=self.provenance, workspace=self.tmp
            )
        self.assertEqual(gateway.dispatches, [])


if __name__ == "__main__":
    unittest.main()
