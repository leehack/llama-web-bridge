#!/usr/bin/env python3
"""Contract tests for exact dispatch inputs and the pure pipeline planner."""

from __future__ import annotations

import unittest
from pathlib import Path

from release_contract import ASSETS_REPOSITORY, ContractError
from release_orchestrator_fixtures_test import (
    BRIDGE_SHA,
    CANDIDATE_RUN_ID,
    NATIVE_MANIFEST_SHA,
    QUALIFICATION_RUN_ID,
    UPSTREAM_COMMIT,
    make_provenance,
)
import release_orchestrator_model as model
import release_orchestrator_planner as planner
import release_orchestrator_run_names as run_names
import stable_release_orchestrator as sro


def declared_workflow_inputs(workflow: Path) -> tuple[str, ...]:
    """Read a workflow's exact ``workflow_dispatch`` input names.

    Deliberately dependency-free: the Python contract suites must stay runnable
    without PyYAML, and the input names are the whole point of the check.
    """
    names: list[str] = []
    in_dispatch = False
    in_inputs = False
    for line in workflow.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not in_dispatch:
            in_dispatch = line == "  workflow_dispatch:"
            continue
        if not in_inputs:
            if line == "    inputs:":
                in_inputs = True
                continue
            if not line.startswith("    "):
                break
            continue
        if not line.startswith("      "):
            break
        if line.startswith("       "):
            continue
        if not stripped.endswith(":"):
            raise AssertionError(f"unexpected input declaration: {line!r}")
        names.append(stripped[:-1])
    if not names:
        raise AssertionError(f"{workflow.name} declares no workflow_dispatch inputs")
    return tuple(names)


class DispatchInputContractTest(unittest.TestCase):
    """A dispatch whose keys drift from the workflow is rejected at build time."""

    WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"

    def setUp(self) -> None:
        self.provenance = make_provenance()
        self.correlation_id = run_names.compute_correlation_id(self.provenance)
        self.binding = model.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )

    def test_candidate_dispatch_inputs_are_exactly_the_declared_inputs(self) -> None:
        declared = declared_workflow_inputs(
            self.WORKFLOWS / model.CANDIDATE_WORKFLOW_FILE
        )
        self.assertEqual(set(declared), set(planner.CANDIDATE_DISPATCH_INPUTS))
        inputs = planner._dispatch_inputs_for_candidate(
            self.provenance, self.correlation_id, self.binding
        )
        # The live governance proof supplies the immutability assertion, so the
        # planner must not pre-declare it.
        self.assertEqual(
            set(inputs),
            set(declared) - {"assets_immutable_releases_enabled"},
        )

    def test_publish_dispatch_inputs_are_exactly_the_declared_inputs(self) -> None:
        declared = declared_workflow_inputs(self.WORKFLOWS / model.PUBLISH_WORKFLOW_FILE)
        self.assertEqual(set(declared), set(planner.PUBLISH_DISPATCH_INPUTS))
        inputs = planner._dispatch_inputs_for_publish(
            self.provenance,
            self.correlation_id,
            self.binding,
            CANDIDATE_RUN_ID,
            QUALIFICATION_RUN_ID,
        )
        # The live environment-policy proof supplies this approval assertion.
        self.assertEqual(set(inputs), set(declared) - {"publish_approved"})

    def test_qualification_dispatch_inputs_are_exactly_the_declared_inputs(self) -> None:
        declared = declared_workflow_inputs(
            self.WORKFLOWS / model.QUALIFICATION_WORKFLOW_FILE
        )
        self.assertEqual(set(declared), set(planner.QUALIFICATION_DISPATCH_INPUTS))
        inputs = planner._dispatch_inputs_for_qualification(
            self.correlation_id, CANDIDATE_RUN_ID
        )
        self.assertEqual(set(inputs), set(declared))

    def test_missing_dispatch_input_fails_closed(self) -> None:
        inputs = planner._dispatch_inputs_for_publish(
            self.provenance,
            self.correlation_id,
            self.binding,
            CANDIDATE_RUN_ID,
            QUALIFICATION_RUN_ID,
        )
        with self.assertRaises(ContractError):
            planner.require_exact_dispatch_inputs(model.PUBLISH_WORKFLOW_FILE, inputs)

    def test_unknown_dispatch_input_fails_closed(self) -> None:
        inputs = planner._dispatch_inputs_for_candidate(
            self.provenance, self.correlation_id, self.binding
        )
        inputs["assets_immutable_releases_enabled"] = "true"
        inputs["unexpected"] = "value"
        with self.assertRaises(ContractError):
            planner.require_exact_dispatch_inputs(model.CANDIDATE_WORKFLOW_FILE, inputs)

    def test_undispatchable_workflow_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            planner.require_exact_dispatch_inputs(
                "ci.yml", {"candidate_run_id": CANDIDATE_RUN_ID}
            )

    def test_non_string_dispatch_input_fails_closed(self) -> None:
        inputs = planner._dispatch_inputs_for_candidate(
            self.provenance, self.correlation_id, self.binding
        )
        inputs["assets_immutable_releases_enabled"] = "true"
        inputs["release_rebuild"] = 0  # type: ignore[assignment]
        with self.assertRaises(ContractError):
            planner.require_exact_dispatch_inputs(model.CANDIDATE_WORKFLOW_FILE, inputs)


class PlanTest(unittest.TestCase):
    """Exhaustive pure-state-machine coverage over already-proven evidence."""

    def setUp(self) -> None:
        self.provenance = make_provenance()
        self.correlation_id = run_names.compute_correlation_id(self.provenance)
        self.binding = model.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )

    def test_new_provenance_plans_exactly_one_candidate_dispatch(self) -> None:
        plan = planner.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=model.PipelineObservation(fresh_binding=self.binding),
        )
        self.assertEqual(plan.action, model.OrchestrationAction.DISPATCH_CANDIDATE)
        self.assertEqual(plan.dispatch_workflow, model.CANDIDATE_WORKFLOW_FILE)
        inputs = plan.dispatch_inputs or {}
        self.assertEqual(inputs["orchestrator_correlation_id"], self.correlation_id)
        self.assertEqual(inputs["bridge_source_sha"], BRIDGE_SHA)
        self.assertEqual(inputs["upstream_tag"], "v0.2.0")
        self.assertEqual(inputs["upstream_commit"], UPSTREAM_COMMIT)
        self.assertEqual(inputs["native_release_tag"], "v0.2.0")
        self.assertEqual(inputs["native_manifest_sha256"], NATIVE_MANIFEST_SHA)
        self.assertEqual(inputs["release_tag"], "v0.1.40")
        self.assertEqual(inputs["release_rebuild"], "0")
        # Everything the planner can know without a live read, and nothing else.
        self.assertEqual(
            set(inputs),
            set(planner.CANDIDATE_DISPATCH_INPUTS) - {"assets_immutable_releases_enabled"},
        )

    def test_published_provenance_is_an_exact_noop(self) -> None:
        plan = planner.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=model.PipelineObservation(
                published=model.PublishedRelease(
                    release_id=4242,
                    release_target=model.ReleaseTarget("v0.1.40", 0),
                    binding=self.binding,
                    published_at="2026-08-20T03:24:11Z",
                )
            ),
        )
        self.assertEqual(plan.action, model.OrchestrationAction.NOOP)
        self.assertIsNone(plan.dispatch_workflow)
        self.assertIn("already published", plan.reason.lower())

    def test_in_flight_candidate_blocks_duplicate_dispatch(self) -> None:
        plan = planner.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=model.PipelineObservation(
                fresh_binding=self.binding,
                candidate_in_flight_run_id="501",
                binding=self.binding,
            ),
        )
        self.assertEqual(plan.action, model.OrchestrationAction.IN_FLIGHT)
        self.assertEqual(plan.in_flight_workflow, model.CANDIDATE_WORKFLOW_FILE)
        self.assertEqual(plan.in_flight_run_id, "501")
        self.assertIsNone(plan.dispatch_workflow)

    def test_candidate_ready_plans_exactly_one_hosted_qualification(self) -> None:
        plan = planner.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=model.PipelineObservation(
                binding=self.binding, candidate_run_id=CANDIDATE_RUN_ID
            ),
        )
        self.assertEqual(
            plan.action, model.OrchestrationAction.DISPATCH_QUALIFICATION
        )
        self.assertEqual(plan.candidate_run_id, CANDIDATE_RUN_ID)
        self.assertEqual(plan.dispatch_workflow, model.QUALIFICATION_WORKFLOW_FILE)
        self.assertEqual(
            plan.dispatch_run_name,
            run_names.qualification_run_name(self.correlation_id, CANDIDATE_RUN_ID),
        )
        self.assertEqual(
            plan.dispatch_inputs,
            {
                "orchestrator_correlation_id": self.correlation_id,
                "candidate_run_id": CANDIDATE_RUN_ID,
            },
        )
        self.assertIsNone(plan.qualification_run_id)
        # No routine state may require a maintainer-supplied payload or an owner
        # workflow_dispatch continuation to advance.
        self.assertNotIn("maintainer", plan.reason.lower())
        self.assertNotIn("attestation", plan.reason.lower())

    def test_in_flight_qualification_blocks_duplicate_publish(self) -> None:
        plan = planner.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=model.PipelineObservation(
                binding=self.binding,
                candidate_run_id=CANDIDATE_RUN_ID,
                qualification_in_flight_run_id=QUALIFICATION_RUN_ID,
            ),
        )
        self.assertEqual(plan.action, model.OrchestrationAction.IN_FLIGHT)
        self.assertEqual(plan.in_flight_workflow, model.QUALIFICATION_WORKFLOW_FILE)

    def test_candidate_and_qualification_ready_plan_publish(self) -> None:
        plan = planner.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=model.PipelineObservation(
                binding=self.binding,
                candidate_run_id=CANDIDATE_RUN_ID,
                qualification_run_id=QUALIFICATION_RUN_ID,
            ),
        )
        self.assertEqual(plan.action, model.OrchestrationAction.DISPATCH_PUBLISH)
        self.assertEqual(plan.dispatch_workflow, model.PUBLISH_WORKFLOW_FILE)
        inputs = plan.dispatch_inputs or {}
        self.assertEqual(inputs["candidate_run_id"], CANDIDATE_RUN_ID)
        self.assertEqual(inputs["qualification_run_id"], QUALIFICATION_RUN_ID)
        self.assertEqual(inputs["release_tag"], "v0.1.40")
        self.assertEqual(inputs["release_rebuild"], "0")
        self.assertNotIn("publish_approved", inputs)
        self.assertEqual(inputs["assets_repo"], ASSETS_REPOSITORY)
        self.assertEqual(inputs["bridge_source_sha"], BRIDGE_SHA)
        self.assertEqual(
            set(inputs), set(planner.PUBLISH_DISPATCH_INPUTS) - {"publish_approved"}
        )

    def test_in_flight_publish_blocks_duplicate_publish(self) -> None:
        plan = planner.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=model.PipelineObservation(
                binding=self.binding,
                candidate_run_id=CANDIDATE_RUN_ID,
                qualification_run_id=QUALIFICATION_RUN_ID,
                publish_in_flight_run_id="701",
            ),
        )
        self.assertEqual(plan.action, model.OrchestrationAction.IN_FLIGHT)
        self.assertEqual(plan.in_flight_workflow, model.PUBLISH_WORKFLOW_FILE)
        self.assertEqual(plan.in_flight_run_id, "701")

    def test_successful_publish_without_immutable_release_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            planner.plan_pipeline(
                provenance=self.provenance,
                correlation_id=self.correlation_id,
                observation=model.PipelineObservation(
                    binding=self.binding,
                    candidate_run_id=CANDIDATE_RUN_ID,
                    qualification_run_id=QUALIFICATION_RUN_ID,
                    publish_succeeded_run_id="701",
                ),
            )

    def test_publish_retry_after_failed_publish_reuses_exact_state(self) -> None:
        plan = planner.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=model.PipelineObservation(
                binding=self.binding,
                candidate_run_id=CANDIDATE_RUN_ID,
                qualification_run_id=QUALIFICATION_RUN_ID,
                publish_retry=True,
            ),
        )
        self.assertEqual(plan.action, model.OrchestrationAction.DISPATCH_PUBLISH)
        self.assertEqual(plan.candidate_run_id, CANDIDATE_RUN_ID)
        self.assertEqual(plan.qualification_run_id, QUALIFICATION_RUN_ID)
        # A retry must be visible as a retry, not reported as a first attempt.
        self.assertIn("retry", plan.reason)

    def test_planner_never_pre_asserts_immutable_release_governance(self) -> None:
        plan = planner.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=model.PipelineObservation(fresh_binding=self.binding),
        )
        self.assertEqual(plan.action, model.OrchestrationAction.DISPATCH_CANDIDATE)
        self.assertNotIn("assets_immutable_releases_enabled", plan.dispatch_inputs)

    def test_planner_never_pre_asserts_publication_approval(self) -> None:
        plan = planner.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=model.PipelineObservation(
                binding=self.binding,
                candidate_run_id=CANDIDATE_RUN_ID,
                qualification_run_id=QUALIFICATION_RUN_ID,
            ),
        )
        self.assertNotIn("publish_approved", plan.dispatch_inputs)

    def test_missing_binding_for_ready_candidate_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            planner.plan_pipeline(
                provenance=self.provenance,
                correlation_id=self.correlation_id,
                observation=model.PipelineObservation(
                    candidate_run_id=CANDIDATE_RUN_ID
                ),
            )

    def test_summary_reports_the_exact_action(self) -> None:
        plan = planner.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=model.PipelineObservation(fresh_binding=self.binding),
        )
        summary = sro.render_step_summary(plan)
        self.assertIn("### Stable Web bridge release orchestration", summary)
        self.assertIn("dispatch_candidate", summary)
        self.assertIn(model.CANDIDATE_WORKFLOW_FILE, summary)
        self.assertIn(self.correlation_id, summary)


if __name__ == "__main__":
    unittest.main()
