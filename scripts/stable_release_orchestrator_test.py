#!/usr/bin/env python3
"""Fail-closed contract tests for the daily stable release state machine.

This suite covers the CLI entry: governed-path classification, bridge
source identity, and caller authorization. The
``release_orchestrator_<concern>_test.py`` suites cover the modules the entry
imports and share ``release_orchestrator_fixtures_test.py``.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from release_contract import ContractError
from release_orchestrator_fixtures_test import OWNER
import stable_release_orchestrator as sro


class BridgeSourceIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repository = Path(self.temp.name) / "bridge"
        self.repository.mkdir()
        self._git("init", "--initial-branch=main")
        self._git("config", "user.name", "Bridge Test")
        self._git("config", "user.email", "bridge-test@example.com")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ("git", "-C", str(self.repository), *args),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    def _commit(self, message: str) -> str:
        self._git("add", ".")
        self._git("commit", "-m", message)
        return self._git("rev-parse", "HEAD")

    def test_actual_resolver_separates_orchestration_from_governed_changes(
        self,
    ) -> None:
        source = self.repository / "src" / "bridge.cpp"
        source.parent.mkdir()
        source.write_text("runtime-v1\n", encoding="utf-8")
        governed = self._commit("initial runtime")

        workflow = (
            self.repository
            / ".github"
            / "workflows"
            / "auto_llama_cpp_update.yml"
        )
        workflow.parent.mkdir(parents=True)
        workflow.write_text("name: scan\n", encoding="utf-8")
        docs = self.repository / "docs" / "release.md"
        docs.parent.mkdir()
        docs.write_text("release docs\n", encoding="utf-8")
        test = self.repository / "scripts" / "resolver_test.py"
        test.parent.mkdir()
        test.write_text("# regression\n", encoding="utf-8")
        (test.parent / "ci_scope.py").write_text("# selector\n", encoding="utf-8")
        orchestration_head = self._commit("workflow tests and docs")

        output = self.repository / "identity.json"
        self.assertEqual(
            sro.main(
                [
                    "resolve-bridge-source",
                    "--repository",
                    str(self.repository),
                    "--head",
                    "HEAD",
                    "--output-json",
                    str(output),
                ]
            ),
            0,
        )
        identity = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(identity["bridge_source_sha"], orchestration_head)
        self.assertEqual(identity["bridge_build_sha"], governed)

        # An unclassified new file is governed by default. This protects the
        # identity if a future build input is added without updating the list.
        (self.repository / "new-build-input.cfg").write_text(
            "runtime-v2\n", encoding="utf-8"
        )
        governed_head = self._commit("new governed build input")
        advanced = sro.resolve_bridge_source_identity(self.repository)
        self.assertEqual(advanced.bridge_source_sha, governed_head)
        self.assertEqual(advanced.bridge_build_sha, governed_head)

    def test_malformed_git_path_is_rejected(self) -> None:
        with self.assertRaises(ContractError):
            sro.is_governed_bridge_path("../outside")

    def test_path_contract_is_fail_closed_for_runtime_and_new_inputs(self) -> None:
        for path in (
            ".github/workflows/auto_llama_cpp_update.yml",
            "README.md",
            "docs/api.md",
            "scripts/bridge_js_source.py",
            "scripts/bridge_operation_queue_direct_cases.mjs",
            "scripts/bridge_operation_queue_fixtures.mjs",
            "scripts/bridge_operation_queue_lifecycle_contract_cases.mjs",
            "scripts/bridge_operation_queue_worker_proxy_cases.mjs",
            "scripts/browser_smoke_support.mjs",
            "scripts/ci_scope.mjs",
            "scripts/ci_scope.py",
            "scripts/ci_scope_test.py",
            "scripts/decision_browser_smoke.mjs",
            "scripts/decision_browser_smoke.py",
            "scripts/grammar_browser_smoke.mjs",
            "scripts/grammar_browser_smoke.py",
            "scripts/mtmd_compat_contract_test.py",
            "scripts/multimodal_browser_smoke.mjs",
            "scripts/multimodal_browser_smoke.py",
            "scripts/native_core_source.py",
            "scripts/next_token_scores_browser_smoke.mjs",
            "scripts/next_token_scores_browser_smoke.py",
            "scripts/orchestrator_source.py",
            "scripts/release_orchestrator_asset_releases.py",
            "scripts/release_orchestrator_asset_releases_test.py",
            "scripts/release_orchestrator_driver.py",
            "scripts/release_orchestrator_driver_backlog_test.py",
            "scripts/release_orchestrator_driver_identical_release_test.py",
            "scripts/release_orchestrator_driver_publication_test.py",
            "scripts/release_orchestrator_driver_test.py",
            "scripts/release_orchestrator_fixtures_test.py",
            "scripts/release_orchestrator_model.py",
            "scripts/release_orchestrator_native.py",
            "scripts/release_orchestrator_native_test.py",
            "scripts/release_orchestrator_planner.py",
            "scripts/release_orchestrator_planner_test.py",
            "scripts/release_orchestrator_release_tags.py",
            "scripts/release_orchestrator_release_tags_test.py",
            "scripts/release_orchestrator_run_names.py",
            "scripts/release_orchestrator_run_names_test.py",
            "scripts/release_orchestrator_stage_proofs.py",
            "scripts/release_orchestrator_transport.py",
            "scripts/release_orchestrator_transport_test.py",
            "scripts/release_orchestrator_workflow_runs.py",
            "scripts/release_orchestrator_workflow_runs_test.py",
            "scripts/release_qualification.py",
            "scripts/speech_to_text_browser_smoke.mjs",
            "scripts/speech_to_text_browser_smoke.py",
            "scripts/speech_to_text_fixture.json",
            "scripts/stable_release_orchestrator.py",
            "scripts/stable_release_orchestrator_test.py",
            "scripts/state_persistence_browser_smoke.mjs",
            "scripts/state_persistence_browser_smoke.py",
            "scripts/text_to_speech_browser_smoke.mjs",
            "scripts/text_to_speech_browser_smoke.py",
            "scripts/verify_ci_reliability.mjs",
            "scripts/verify_ci_reliability.py",
            "scripts/verify_ci_reliability_doc_facts_test.py",
            "scripts/verify_ci_reliability_pin_test.py",
            "scripts/verify_decision_api.py",
            "scripts/verify_state_persistence_api.py",
            "scripts/verify_text_to_speech_api.py",
            "scripts/wasm64_runtime_patch_contract_test.py",
            "scripts/worker_runtime_state_test.mjs",
            "tests/js/bridge_js_source.mjs",
            "tests/js/bridge_operation_queue_direct_cases.mjs",
            "tests/js/bridge_operation_queue_fixtures.mjs",
            "tests/js/bridge_operation_queue_lifecycle_contract_cases.mjs",
            "tests/js/bridge_operation_queue_worker_proxy_cases.mjs",
            "tests/js/browser_smoke_support_test.mjs",
            "tests/js/ci_scope_test.mjs",
            "tests/js/decision_api_contract_test.mjs",
            "tests/js/mtmd_compat_contract_test.mjs",
            "tests/js/multimodal_harness_parity_test.mjs",
            "tests/js/native_core_source.mjs",
            "tests/js/state_persistence_api_contract_test.mjs",
            "tests/js/state_persistence_harness_parity_test.mjs",
            "tests/js/text_to_speech_api_contract_test.mjs",
            "tests/js/verify_ci_reliability_test.mjs",
            "tests/js/wasm64_runtime_patch_contract_test.mjs",
            "tests/js/worker_runtime_state_test.mjs",
            "tests/js/nested/new_contract_test.mjs",
            "tests/new_suite/new_contract_test.mjs",
            "tests/release/contract_test.mjs",
            "scripts/release/archive.mjs",
            "scripts/release/publication_state.mjs",
            "scripts/release/qualification.mjs",
            "scripts/release/qualify.mjs",
            "scripts/release/wav.mjs",
            "scripts/release/orchestrator/asset_releases.mjs",
            "scripts/release/orchestrator/cli.mjs",
            "scripts/release/orchestrator/driver.mjs",
            "scripts/release/orchestrator/model.mjs",
            "scripts/release/orchestrator/native.mjs",
            "scripts/release/orchestrator/planner.mjs",
            "scripts/release/orchestrator/release_tags.mjs",
            "scripts/release/orchestrator/run_names.mjs",
            "scripts/release/orchestrator/stage_proofs.mjs",
            "scripts/release/orchestrator/transport.mjs",
            "scripts/release/orchestrator/workflow_runs.mjs",
            "tests/release/zip_fixture.mjs",
            "tests/release/orchestrator/fixtures.mjs",
            "tests/release/fixtures/attestation.json",
        ):
            with self.subTest(path=path):
                self.assertFalse(sro.is_governed_bridge_path(path))
        for path in (
            "CMakeLists.txt",
            ".github/workflows/bridge_candidate.yml",
            "package.json",
            "js/llama_webgpu_bridge.d.ts",
            "scripts/build_bridge.sh",
            "scripts/generate_release_manifest.py",
            # The wasm64 runtime patch rewrites a published asset: its Python
            # original and the Node port are both build inputs.
            "scripts/patch_wasm64_runtime.mjs",
            "scripts/patch_wasm64_runtime.py",
            "scripts/release_contract.py",
            # A new orchestrator module is listed explicitly, never by prefix.
            "scripts/release_orchestrator_unlisted.py",
            # So is a new smoke helper; only the smoke entry points match by suffix.
            "scripts/browser_smoke_unlisted.mjs",
            "scripts/browser_smoke.mjs",
            # Only the speech fixture is listed; another data file is governed.
            "scripts/text_to_speech_fixture.json",
            "scripts/verify_emscripten_version.py",
            "src/llama_webgpu_core.cpp",
            "src/core/exports_tts.inc",
            "src/core/new_part.inc",
            "tests/js/new_fixture.mjs",
            "tests/new_build_input.mjs",
            # Only tests/release/ is exempt as a whole.
            "tests/releases/new_fixture.mjs",
            "scripts/release/contract.mjs",
            "scripts/release/errors.mjs",
            "scripts/release/json.mjs",
            "scripts/release/cli.mjs",
            "scripts/release/manifest.mjs",
            "scripts/release/python_compat.mjs",
            "scripts/release/unlisted.mjs",
            "scripts/release/orchestrator/unlisted.mjs",
            "scripts/build/verify_emscripten_version.mjs",
            "unknown/new-build-input.cfg",
        ):
            with self.subTest(path=path):
                self.assertTrue(sro.is_governed_bridge_path(path))


class OrchestrationCallerAuthorizationTest(unittest.TestCase):
    WORKFLOW = (
        Path(__file__).resolve().parent.parent
        / ".github"
        / "workflows"
        / "auto_llama_cpp_update.yml"
    )

    def test_schedule_owner_manual_and_owner_continuation_are_authorized(self) -> None:
        sro.require_orchestration_caller("schedule", "github-actions", "github-actions")
        sro.require_orchestration_caller("workflow_dispatch", OWNER, OWNER)
        sro.require_orchestration_caller("workflow_run", OWNER, OWNER)

    def test_non_owner_manual_or_continuation_is_rejected(self) -> None:
        for event in ("workflow_dispatch", "workflow_run"):
            for actor, triggering_actor in (
                ("collaborator", OWNER),
                (OWNER, "collaborator"),
            ):
                with self.subTest(event=event, actor=actor), self.assertRaises(
                    ContractError
                ):
                    sro.require_orchestration_caller(event, actor, triggering_actor)

    def test_workflow_run_trigger_and_both_job_gates_are_exact(self) -> None:
        workflow = self.WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("fetch-depth: 0", workflow)
        self.assertIn(
            "scripts/stable_release_orchestrator.py resolve-bridge-source",
            workflow,
        )
        self.assertIn('--bridge-build-sha "${bridge_build_sha}"', workflow)
        manual_gate = (
            "github.event_name == 'workflow_dispatch' && "
            "github.actor == github.repository_owner && "
            "github.triggering_actor == github.repository_owner"
        )
        continuation_gate = (
            "github.event_name == 'workflow_run' && "
            "github.event.workflow_run.conclusion == 'success' && "
            "github.event.workflow_run.head_branch == "
            "github.event.repository.default_branch && "
            "github.event.workflow_run.actor.login == github.repository_owner && "
            "github.event.workflow_run.triggering_actor.login == "
            "github.repository_owner"
        )
        for trigger in (
            "- Build Exact Bridge Candidate",
            "- Qualify Exact Bridge Candidate",
            "- Publish Exact Qualified Bridge Assets",
            "types: [completed]",
        ):
            self.assertIn(trigger, workflow)
        self.assertEqual(workflow.count(manual_gate), 2)
        self.assertEqual(workflow.count(continuation_gate), 2)
        self.assertLess(workflow.rfind(manual_gate), workflow.find("environment:"))
        self.assertLess(workflow.rfind(continuation_gate), workflow.find("environment:"))
        proof = "Prove the exact workflow continuation before environment use"
        self.assertIn(proof, workflow)
        proof_block = workflow.split(f"      - name: {proof}\n", 1)[1].split(
            "\n      - name:", 1
        )[0]
        self.assertIn("scripts/release_qualification.py verify-run", proof_block)
        self.assertIn("--run-attempt 1", proof_block)
        for exact_mapping in (
            ".github/workflows/bridge_candidate.yml)\n"
            "              artifact_name=exact-webgpu-bridge-dist",
            ".github/workflows/bridge_qualification.yml)\n"
            "              artifact_name=qualification-attestation",
            ".github/workflows/publish_assets.yml)\n"
            "              artifact_name=bridge-qualification-outcome",
        ):
            self.assertIn(exact_mapping, proof_block)
        self.assertLess(workflow.find(proof), workflow.find("environment:"))


if __name__ == "__main__":
    unittest.main()
