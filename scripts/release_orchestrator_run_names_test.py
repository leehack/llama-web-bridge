#!/usr/bin/env python3
"""Contract tests for correlation IDs and deterministic workflow run names."""

from __future__ import annotations

import unittest
from pathlib import Path

from release_contract import ContractError
from release_orchestrator_fixtures_test import (
    ADVANCED_BRIDGE_SHA,
    BRIDGE_SHA,
    CANDIDATE_RUN_ID,
    NATIVE_MANIFEST_SHA,
    QUALIFICATION_RUN_ID,
    make_provenance,
)
import release_orchestrator_model as model
import release_orchestrator_run_names as run_names


class CorrelationTest(unittest.TestCase):
    def test_deterministic_correlation_id_ignores_orchestration_only_main(self) -> None:
        first = run_names.compute_correlation_id(make_provenance())
        second = run_names.compute_correlation_id(
            make_provenance(bridge_source_sha=ADVANCED_BRIDGE_SHA)
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            f"auto-stable-v0.2.0-{NATIVE_MANIFEST_SHA[:16]}-build-{BRIDGE_SHA[:16]}",
        )

    def test_correlation_id_changes_with_governed_build_source(self) -> None:
        other = run_names.compute_correlation_id(
            make_provenance(
                bridge_source_sha=ADVANCED_BRIDGE_SHA,
                bridge_build_sha=ADVANCED_BRIDGE_SHA,
            )
        )
        self.assertNotEqual(run_names.compute_correlation_id(make_provenance()), other)

    def test_correlation_id_changes_with_native_manifest(self) -> None:
        other = run_names.compute_correlation_id(
            make_provenance(native_manifest_sha256="b" * 64)
        )
        self.assertNotEqual(run_names.compute_correlation_id(make_provenance()), other)

    def test_candidate_run_name_round_trips_persisted_binding(self) -> None:
        correlation_id = run_names.compute_correlation_id(make_provenance())
        binding = model.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )
        name = run_names.candidate_run_name(correlation_id, binding)
        self.assertEqual(
            name,
            f"bridge-candidate {correlation_id} source:{BRIDGE_SHA} "
            "tag:v0.1.40 rebuild:0",
        )
        self.assertEqual(run_names.parse_candidate_run_name(name, correlation_id), binding)

    def test_foreign_candidate_run_name_is_not_adopted(self) -> None:
        correlation_id = run_names.compute_correlation_id(make_provenance())
        foreign_names = (
            f"bridge-candidate other-correlation source:{BRIDGE_SHA} "
            "tag:v0.1.40 rebuild:0",
            "bridge-candidate other-correlation source:not-a-sha "
            "tag:v0.1.40 rebuild:0",
            "bridge-candidate other-correlation malformed",
            f"bridge-candidate {correlation_id}-foreign source:not-a-sha "
            "tag:v0.1.40 rebuild:bad",
            f"bridge-candidate-other {correlation_id} malformed",
            "some-unrelated-workflow-run",
        )
        for foreign_name in foreign_names:
            with self.subTest(run_name=foreign_name):
                self.assertIsNone(
                    run_names.parse_candidate_run_name(foreign_name, correlation_id)
                )

    def test_malformed_candidate_run_name_claiming_correlation_fails_closed(
        self,
    ) -> None:
        correlation_id = run_names.compute_correlation_id(make_provenance())
        malformed_names = (
            f"bridge-candidate {correlation_id} source:not-a-sha "
            "tag:v0.1.40 rebuild:0",
            f"bridge-candidate {correlation_id} source:{BRIDGE_SHA} "
            "tag:v0.1.40 rebuild:not-a-number",
            f"bridge-candidate {correlation_id} source:{BRIDGE_SHA} "
            "tag:v0.1.40 rebuild:-1",
            f"bridge-candidate {correlation_id} source:{BRIDGE_SHA} "
            "tag:v0.1.40 rebuild:01",
            f"bridge-candidate {correlation_id} source:{BRIDGE_SHA} "
            "tag:invalid_tag rebuild:0",
            f"bridge-candidate {correlation_id}",
            f"bridge-candidate {correlation_id} source:{BRIDGE_SHA}",
            f"bridge-candidate {correlation_id} source:{BRIDGE_SHA} tag:v0.1.40",
            f"bridge-candidate {correlation_id} malformed",
            f"bridge-candidate {correlation_id} source:{BRIDGE_SHA} "
            "tag:v0.1.40 rebuild:0\x00",
            f"bridge-candidate {correlation_id} " + ("x" * 200),
        )
        for malformed_name in malformed_names:
            with self.subTest(run_name=malformed_name):
                with self.assertRaises(ContractError):
                    run_names.parse_candidate_run_name(malformed_name, correlation_id)

    def test_generated_run_name_enforces_a_conservative_length_bound(self) -> None:
        with self.assertRaises(ContractError):
            run_names._require_run_name("x" * (run_names.MAX_RUN_NAME_CHARACTERS + 1))


class WorkflowRunNameContractTest(unittest.TestCase):
    """The rendered workflow run-name must be exactly what the parser expects."""

    WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"

    def render(self, workflow: str, inputs: dict[str, str]) -> str:
        for line in (self.WORKFLOWS / workflow).read_text(encoding="utf-8").splitlines():
            if not line.startswith("run-name: "):
                continue
            rendered = line[len("run-name: ") :]
            for name, value in inputs.items():
                rendered = rendered.replace("${{ inputs." + name + " }}", value)
            self.assertNotIn("${{", rendered)
            return rendered
        raise AssertionError(f"{workflow} declares no run-name")

    def test_candidate_workflow_renders_the_parsed_run_name(self) -> None:
        correlation_id = run_names.compute_correlation_id(make_provenance())
        binding = model.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )
        rendered = self.render(
            model.CANDIDATE_WORKFLOW_FILE,
            {
                "orchestrator_correlation_id": correlation_id,
                "bridge_source_sha": BRIDGE_SHA,
                "release_tag": "v0.1.40",
                "release_rebuild": "0",
            },
        )
        self.assertEqual(rendered, run_names.candidate_run_name(correlation_id, binding))
        self.assertEqual(run_names.parse_candidate_run_name(rendered, correlation_id), binding)

    def test_qualification_workflow_renders_the_parsed_run_name(self) -> None:
        correlation_id = run_names.compute_correlation_id(make_provenance())
        rendered = self.render(
            model.QUALIFICATION_WORKFLOW_FILE,
            {
                "orchestrator_correlation_id": correlation_id,
                "candidate_run_id": CANDIDATE_RUN_ID,
            },
        )
        self.assertEqual(
            rendered, run_names.qualification_run_name(correlation_id, CANDIDATE_RUN_ID)
        )

    def test_publish_workflow_renders_the_parsed_run_name(self) -> None:
        correlation_id = run_names.compute_correlation_id(make_provenance())
        binding = model.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )
        rendered = self.render(
            model.PUBLISH_WORKFLOW_FILE,
            {
                "orchestrator_correlation_id": correlation_id,
                "candidate_run_id": CANDIDATE_RUN_ID,
                "qualification_run_id": QUALIFICATION_RUN_ID,
                "bridge_source_sha": BRIDGE_SHA,
                "release_tag": "v0.1.40",
                "release_rebuild": "0",
            },
        )
        self.assertEqual(
            rendered,
            run_names.publish_run_name(
                correlation_id, CANDIDATE_RUN_ID, QUALIFICATION_RUN_ID, binding
            ),
        )
        self.assertEqual(
            run_names.parse_publish_run_name(rendered, correlation_id),
            (CANDIDATE_RUN_ID, QUALIFICATION_RUN_ID, binding),
        )


if __name__ == "__main__":
    unittest.main()
