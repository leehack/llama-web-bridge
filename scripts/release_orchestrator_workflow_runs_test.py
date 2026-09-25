#!/usr/bin/env python3
"""Contract tests for workflow run parsing, history recovery, and selection."""

from __future__ import annotations

import unittest
from typing import Any

from release_contract import ContractError
from release_orchestrator_fixtures_test import (
    BRIDGE_SHA,
    CANDIDATE_RUN_ID,
    DEFAULT_BRANCH,
    FakeGateway,
    NATIVE_PUBLISHED_AT,
    OWNER,
    make_provenance,
    run_payload,
    runs_response,
)
import release_orchestrator_model as model
import release_orchestrator_run_names as run_names
import release_orchestrator_workflow_runs as workflow_runs


class WorkflowRunsResponseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = model.CANDIDATE_WORKFLOW_PATH
        correlation_id = run_names.compute_correlation_id(make_provenance())
        self.run_name = run_names.candidate_run_name(
            correlation_id,
            model.PipelineBinding(
                bridge_source_sha=BRIDGE_SHA,
                release_tag="v0.1.40",
                release_rebuild=0,
            ),
        )
        self.run = run_payload(
            run_id=CANDIDATE_RUN_ID, path=self.path, run_name=self.run_name
        )

    def test_dynamic_correlated_api_name_is_parsed(self) -> None:
        self.assertEqual(self.run["name"], self.run_name)
        self.assertNotEqual(self.run["name"], "Build Exact Bridge Candidate")
        runs = workflow_runs.parse_workflow_runs(
            runs_response([self.run]),
            workflow_path=self.path,
            default_branch=DEFAULT_BRANCH,
        )
        self.assertEqual([record.run_id for record in runs], [CANDIDATE_RUN_ID])

    def test_bare_list_response_is_rejected(self) -> None:
        with self.assertRaises(ContractError):
            workflow_runs.parse_workflow_runs(
                [self.run], workflow_path=self.path, default_branch=DEFAULT_BRANCH
            )

    def test_truncated_page_set_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            workflow_runs.parse_workflow_runs(
                {"total_count": 5000, "workflow_runs": [self.run]},
                workflow_path=self.path,
                default_branch=DEFAULT_BRANCH,
            )

    def test_foreign_workflow_path_fails_closed(self) -> None:
        foreign = run_payload(
            run_id=CANDIDATE_RUN_ID,
            path=".github/workflows/ci.yml",
            run_name=self.run_name,
            api_name=self.run_name,
        )
        with self.assertRaises(ContractError):
            workflow_runs.parse_workflow_runs(
                runs_response([foreign]),
                workflow_path=self.path,
                default_branch=DEFAULT_BRANCH,
            )

    def test_unsupported_workflow_path_fails_closed_even_when_record_matches(self) -> None:
        unsupported_path = ".github/workflows/ci.yml"
        unsupported = dict(self.run, path=unsupported_path)
        with self.assertRaises(ContractError):
            workflow_runs.parse_workflow_runs(
                runs_response([unsupported]),
                workflow_path=unsupported_path,
                default_branch=DEFAULT_BRANCH,
            )

    def test_non_owner_actor_or_triggering_actor_fails_closed(self) -> None:
        for field in ("actor", "triggering_actor"):
            malformed = dict(self.run)
            malformed[field] = {"login": "someone-else"}
            with self.subTest(field=field), self.assertRaises(ContractError):
                workflow_runs.parse_workflow_runs(
                    runs_response([malformed]),
                    workflow_path=self.path,
                    default_branch=DEFAULT_BRANCH,
                )

    def test_duplicate_run_ids_fail_closed(self) -> None:
        with self.assertRaises(ContractError):
            workflow_runs.parse_workflow_runs(
                {"total_count": 2, "workflow_runs": [self.run, dict(self.run)]},
                workflow_path=self.path,
                default_branch=DEFAULT_BRANCH,
            )

    def test_server_query_bounds_only_relevant_owner_branch_history(self) -> None:
        path = workflow_runs._workflow_runs_path(
            workflow_file=model.CANDIDATE_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        self.assertIn("per_page=100", path)
        self.assertIn("event=workflow_dispatch", path)
        self.assertIn("branch=main", path)
        self.assertIn(f"actor={OWNER}", path)
        self.assertIn("created=%3E%3D2026-08-19T12%3A34%3A56Z", path)

    def test_filtered_history_paginates_beyond_one_hundred_without_global_count(self) -> None:
        payloads = [
            run_payload(
                run_id=str(1000 + index),
                path=self.path,
                run_name=f"unrelated-{index}",
            )
            for index in range(101)
        ]
        first_path = workflow_runs._workflow_runs_path(
            workflow_file=model.CANDIDATE_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        second_path = workflow_runs._workflow_runs_path(
            workflow_file=model.CANDIDATE_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
            page=2,
        )
        gateway = FakeGateway(
            json_routes={
                first_path: {"total_count": 101, "workflow_runs": payloads[:100]},
                second_path: {"total_count": 101, "workflow_runs": payloads[100:]},
            }
        )
        records = workflow_runs._fetch_runs(
            gateway,
            workflow_file=model.CANDIDATE_WORKFLOW_FILE,
            workflow_path=self.path,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        self.assertEqual(len(records), 101)
        self.assertEqual({record.run_id for record in records}, {
            str(1000 + index) for index in range(101)
        })

    def test_page_total_change_fails_closed(self) -> None:
        payloads = [
            run_payload(
                run_id=str(2000 + index),
                path=self.path,
                run_name=f"unrelated-{index}",
            )
            for index in range(101)
        ]
        first_path = workflow_runs._workflow_runs_path(
            workflow_file=model.CANDIDATE_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        second_path = workflow_runs._workflow_runs_path(
            workflow_file=model.CANDIDATE_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
            page=2,
        )
        gateway = FakeGateway(
            json_routes={
                first_path: {"total_count": 101, "workflow_runs": payloads[:100]},
                second_path: {"total_count": 102, "workflow_runs": payloads[100:]},
            }
        )
        with self.assertRaises(ContractError):
            workflow_runs._fetch_runs(
                gateway,
                workflow_file=model.CANDIDATE_WORKFLOW_FILE,
                workflow_path=self.path,
                default_branch=DEFAULT_BRANCH,
                created_since=NATIVE_PUBLISHED_AT,
            )

    def test_saturated_search_is_split_into_complete_time_windows(self) -> None:
        end = "2026-08-19T12:34:57Z"
        payloads = [
            run_payload(
                run_id=str(3000 + index),
                path=self.path,
                run_name=f"windowed-{index}",
            )
            for index in range(1000)
        ]
        routes: dict[str, Any] = {}
        initial_path = workflow_runs._workflow_runs_path(
            workflow_file=model.CANDIDATE_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        full_range_path = workflow_runs._workflow_runs_path(
            workflow_file=model.CANDIDATE_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
            created_until=end,
        )
        routes[initial_path] = {
            "total_count": 1000,
            "workflow_runs": payloads[:100],
        }
        routes[full_range_path] = {
            "total_count": 1000,
            "workflow_runs": payloads[:100],
        }
        for start, stop, window_start, window_end in (
            (0, 600, NATIVE_PUBLISHED_AT, NATIVE_PUBLISHED_AT),
            (600, 1000, end, end),
        ):
            total = stop - start
            for page, offset in enumerate(range(start, stop, 100), start=1):
                path = workflow_runs._workflow_runs_path(
                    workflow_file=model.CANDIDATE_WORKFLOW_FILE,
                    default_branch=DEFAULT_BRANCH,
                    created_since=window_start,
                    created_until=window_end,
                    page=page if page > 1 else None,
                )
                routes[path] = {
                    "total_count": total,
                    "workflow_runs": payloads[offset:min(offset + 100, stop)],
                }
        gateway = FakeGateway(json_routes=routes, now=end)
        records = workflow_runs._fetch_runs(
            gateway,
            workflow_file=model.CANDIDATE_WORKFLOW_FILE,
            workflow_path=self.path,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        self.assertEqual(len(records), 1000)


class RunSelectionTest(unittest.TestCase):
    def _runs(self, *names_and_states: tuple[str, str, str, str | None]) -> list[Any]:
        payloads = [
            run_payload(
                run_id=run_id, path=model.CANDIDATE_WORKFLOW_PATH, run_name=name,
                status=status, conclusion=conclusion,
            )
            for run_id, name, status, conclusion in names_and_states
        ]
        return workflow_runs.parse_workflow_runs(
            runs_response(payloads),
            workflow_path=model.CANDIDATE_WORKFLOW_PATH,
            default_branch=DEFAULT_BRANCH,
        )

    def test_single_success_is_selected(self) -> None:
        runs = self._runs(("501", "target", "completed", "success"))
        selection = workflow_runs.select_pipeline_runs(runs, label="candidate", matcher=lambda name: name == "target")
        self.assertEqual(selection.succeeded_run_id, "501")
        self.assertEqual(selection.in_flight_run_id, None)

    def test_duplicate_success_fails_closed(self) -> None:
        runs = self._runs(
            ("501", "target", "completed", "success"),
            ("502", "target", "completed", "success"),
        )
        with self.assertRaises(ContractError):
            workflow_runs.select_pipeline_runs(runs, label="candidate", matcher=lambda name: name == "target")

    def test_duplicate_in_flight_fails_closed(self) -> None:
        runs = self._runs(
            ("501", "target", "in_progress", None),
            ("502", "target", "queued", None),
        )
        with self.assertRaises(ContractError):
            workflow_runs.select_pipeline_runs(runs, label="candidate", matcher=lambda name: name == "target")

    def test_success_racing_a_new_dispatch_fails_closed(self) -> None:
        runs = self._runs(
            ("501", "target", "completed", "success"),
            ("502", "target", "in_progress", None),
        )
        with self.assertRaises(ContractError):
            workflow_runs.select_pipeline_runs(runs, label="candidate", matcher=lambda name: name == "target")

    def test_failed_runs_are_retryable_not_selected(self) -> None:
        runs = self._runs(("501", "target", "completed", "failure"))
        selection = workflow_runs.select_pipeline_runs(runs, label="candidate", matcher=lambda name: name == "target")
        self.assertIsNone(selection.succeeded_run_id)
        self.assertIsNone(selection.in_flight_run_id)
        self.assertEqual([record.run_id for record in selection.unsuccessful], ["501"])

    def test_second_attempt_fails_closed(self) -> None:
        payload = run_payload(
            run_id="501",
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name="target",
            run_attempt=2,
        )
        runs = workflow_runs.parse_workflow_runs(
            runs_response([payload]),
            workflow_path=model.CANDIDATE_WORKFLOW_PATH,
            default_branch=DEFAULT_BRANCH,
        )
        with self.assertRaises(ContractError):
            workflow_runs.select_pipeline_runs(runs, label="candidate", matcher=lambda name: name == "target")

    def test_off_main_line_run_fails_closed(self) -> None:
        payload = run_payload(
            run_id="501",
            path=model.CANDIDATE_WORKFLOW_PATH,
            run_name="target",
            head_branch="attacker",
        )
        runs = workflow_runs.parse_workflow_runs(
            runs_response([payload]),
            workflow_path=model.CANDIDATE_WORKFLOW_PATH,
            default_branch=DEFAULT_BRANCH,
        )
        with self.assertRaises(ContractError):
            workflow_runs.select_pipeline_runs(runs, label="candidate", matcher=lambda name: name == "target")

    def test_candidate_matcher_and_selection_fail_closed_on_correlated_malformed_run(
        self,
    ) -> None:
        correlation_id = run_names.compute_correlation_id(make_provenance())
        matcher = workflow_runs._candidate_matcher(correlation_id)
        malformed_names = (
            f"bridge-candidate {correlation_id} source:not-a-sha "
            "tag:v0.1.40 rebuild:0",
            f"bridge-candidate {correlation_id} source:{BRIDGE_SHA} "
            "tag:v0.1.40 rebuild:bad",
            f"bridge-candidate {correlation_id} source:{BRIDGE_SHA} "
            "tag:invalid_tag rebuild:0",
            f"bridge-candidate {correlation_id} source:{BRIDGE_SHA} "
            "tag:v0.1.40",
            f"bridge-candidate {correlation_id} malformed",
            f"bridge-candidate {correlation_id} source:{BRIDGE_SHA} "
            "tag:v0.1.40 rebuild:0\x00",
        )
        for malformed_name in malformed_names:
            with self.subTest(run_name=malformed_name):
                with self.assertRaises(ContractError):
                    matcher(malformed_name)
                runs = self._runs(("501", malformed_name, "in_progress", None))
                with self.assertRaises(ContractError):
                    workflow_runs.select_pipeline_runs(runs, label="candidate", matcher=matcher)

    def test_candidate_matcher_and_selection_ignore_foreign_candidate_run(
        self,
    ) -> None:
        correlation_id = run_names.compute_correlation_id(make_provenance())
        matcher = workflow_runs._candidate_matcher(correlation_id)
        foreign_names = (
            f"bridge-candidate other-correlation source:{BRIDGE_SHA} "
            "tag:v0.1.40 rebuild:0",
            "bridge-candidate other-correlation source:not-a-sha "
            "tag:v0.1.40 rebuild:0",
            "bridge-candidate other-correlation malformed",
            f"bridge-candidate {correlation_id}-foreign source:not-a-sha "
            "tag:v0.1.40 rebuild:bad",
            f"bridge-candidate-other {correlation_id} malformed",
            "completely-unrelated-workflow-run",
        )
        for foreign_name in foreign_names:
            with self.subTest(run_name=foreign_name):
                self.assertFalse(matcher(foreign_name))
                runs = self._runs(("501", foreign_name, "in_progress", None))
                selection = workflow_runs.select_pipeline_runs(
                    runs, label="candidate", matcher=matcher
                )
                self.assertEqual(len(selection.matched), 0)
                self.assertIsNone(selection.in_flight_run_id)
                self.assertIsNone(selection.succeeded_run_id)


if __name__ == "__main__":
    unittest.main()
