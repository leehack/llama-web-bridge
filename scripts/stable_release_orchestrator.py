#!/usr/bin/env python3
"""Artifact-driven automatic stable Web bridge release state machine.

Each event-driven scan resolves every stable native release after the immutable
automation baseline, then idempotently advances each three-stage pipeline. A
native release older than one a published asset release already records is
superseded, not rebuilt. A daily scheduled scan provides repair fallback:

1. Build Exact Bridge Candidate      (.github/workflows/bridge_candidate.yml)
2. Qualify Exact Bridge Candidate    (.github/workflows/bridge_qualification.yml)
3. Publish Exact Qualified Assets    (.github/workflows/publish_assets.yml)

Every transition is proven from downloaded artifact and release bytes. The live
``actions/runs`` API never echoes a run's dispatch inputs, so pipeline state is
carried by a deterministic ``run-name`` that each workflow renders from its own
exact inputs, and every named run is then re-proven against its run record, its
unique artifact, and that artifact's contents before it advances anything.

Every stage is dispatched by this orchestrator, so a discovered eligible native
release reaches a verified immutable Web asset release with no manual step.
Progression after discovery is event-driven: each stage's completion wakes the
next scan. The daily schedule also discovers releases and provides idempotent
repair fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from release_contract import ContractError, _strict_json_loads
from release_orchestrator_driver import advance_pipeline
from release_orchestrator_model import (
    NativeProvenance,
    OrchestrationAction,
    OrchestrationPlan,
    REPOSITORY_OWNER,
    _COMMIT_RE,
    require_stable_provenance,
)
from release_orchestrator_native import (
    _CHANNELS,
    _native_release_order,
    scan_native_provenance,
    select_stable_native_backlog,
)
from release_orchestrator_run_names import compute_correlation_id
from release_orchestrator_transport import GhGateway

# This file is the CLI entry. The state machine lives in the
# release_orchestrator_<concern> modules imported above; this file keeps the
# governed-path classifier that AGENTS.md and CONTRIBUTING.md name, the bridge
# source identity it resolves, and the command-line surface.


# These files can change how a release is discovered, qualified, or published,
# but they do not change the runtime/build inputs placed in the bridge artifact.
# Everything not explicitly classified here is governed by default so a newly
# added build input cannot silently inherit an older release identity.
_ORCHESTRATION_ONLY_PATHS = frozenset(
    {
        ".gitignore",
        ".github/workflows/auto_llama_cpp_update.yml",
        ".github/workflows/bridge_qualification.yml",
        ".github/workflows/ci.yml",
        ".github/workflows/publish_assets.yml",
        "AGENTS.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "README.md",
        "scripts/bridge_js_source.py",
        "scripts/bridge_operation_queue_direct_cases.mjs",
        "scripts/bridge_operation_queue_fixtures.mjs",
        "scripts/bridge_operation_queue_lifecycle_contract_cases.mjs",
        "scripts/bridge_operation_queue_worker_proxy_cases.mjs",
        "scripts/ci_scope.py",
        "scripts/native_core_source.py",
        "scripts/orchestrator_source.py",
        "scripts/release_orchestrator_asset_releases.py",
        "scripts/release_orchestrator_driver.py",
        "scripts/release_orchestrator_model.py",
        "scripts/release_orchestrator_native.py",
        "scripts/release_orchestrator_planner.py",
        "scripts/release_orchestrator_release_tags.py",
        "scripts/release_orchestrator_run_names.py",
        "scripts/release_orchestrator_stage_proofs.py",
        "scripts/release_orchestrator_transport.py",
        "scripts/release_orchestrator_workflow_runs.py",
        "scripts/release_publication_state.py",
        "scripts/release_qualification.py",
        "scripts/stable_release_orchestrator.py",
        "scripts/verify_ci_reliability.py",
        # Ported to tests/js/*_api_contract_test.mjs, which the _test.mjs
        # suffix covers. Keep these entries: the commit that ported them
        # deletes these paths, and history must not classify that commit as a
        # build input.
        "scripts/verify_decision_api.py",
        "scripts/verify_state_persistence_api.py",
        "scripts/verify_text_to_speech_api.py",
        # Current home of the scripts/bridge_operation_queue_* fixtures above.
        # Keep the scripts/ entries: the commit that moved them deletes those
        # paths, and history must not classify that commit as a build input.
        "tests/js/bridge_operation_queue_direct_cases.mjs",
        "tests/js/bridge_operation_queue_fixtures.mjs",
        "tests/js/bridge_operation_queue_lifecycle_contract_cases.mjs",
        "tests/js/bridge_operation_queue_worker_proxy_cases.mjs",
        # Source readers the JS contract tests import.
        "tests/js/bridge_js_source.mjs",
        "tests/js/native_core_source.mjs",
    }
)
_ORCHESTRATION_ONLY_PREFIXES = ("docs/",)
_ORCHESTRATION_ONLY_SCRIPT_SUFFIXES = (
    "_browser_smoke.py",
    "_test.mjs",
    "_test.py",
)
_ORCHESTRATION_ONLY_JS_TEST_SUFFIX = "_test.mjs"


@dataclass(frozen=True)
class BridgeSourceIdentity:
    """Current executable source plus its governed runtime/build identity."""

    bridge_source_sha: str
    bridge_build_sha: str


def is_governed_bridge_path(path: str) -> bool:
    """Return whether a repository path can change published bridge artifacts.

    The default is deliberately governed. Only known workflow, validation,
    test, and documentation surfaces are exempt, so deleting this classifier or
    adding a new build input makes the identity advance rather than silently
    reusing an older immutable release.
    """
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or "\\" in path
        or any(part in ("", ".", "..") for part in path.split("/"))
    ):
        raise ContractError(f"git reported an invalid repository path: {path!r}")
    if path in _ORCHESTRATION_ONLY_PATHS:
        return False
    if path.startswith(_ORCHESTRATION_ONLY_PREFIXES):
        return False
    if path.startswith("scripts/") and path.endswith(
        _ORCHESTRATION_ONLY_SCRIPT_SUFFIXES
    ):
        return False
    if path.startswith("tests/js/") and path.endswith(
        _ORCHESTRATION_ONLY_JS_TEST_SUFFIX
    ):
        return False
    return True


def _git_output(repository: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository), *args),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError):
            detail = error.stderr.strip()
        raise ContractError(
            f"could not resolve governed bridge source with git: {detail or error}"
        ) from error
    return completed.stdout


def resolve_bridge_source_identity(
    repository: Path, head: str = "HEAD"
) -> BridgeSourceIdentity:
    """Resolve HEAD and the newest first-parent governed build change.

    The scheduled checkout must have complete history. Walking first-parent
    makes the selected build identity a commit on the exact default-branch line,
    including merge commits that introduce governed files.
    """
    if not repository.is_dir() or repository.is_symlink():
        raise ContractError(f"bridge repository is not a directory: {repository}")
    source_sha = _git_output(repository, "rev-parse", "--verify", f"{head}^{{commit}}")
    source_sha = source_sha.strip()
    if _COMMIT_RE.fullmatch(source_sha) is None:
        raise ContractError("resolved bridge source is not a full lowercase commit SHA")
    history = _git_output(
        repository,
        "log",
        "--first-parent",
        "--format=%x00%H%x00",
        "--name-only",
        "--no-renames",
        source_sha,
    )
    fields = history.split("\0")
    if not fields or fields[0] != "" or len(fields) < 3 or len(fields) % 2 == 0:
        raise ContractError("git returned malformed bridge first-parent history")
    for index in range(1, len(fields), 2):
        commit = fields[index]
        if _COMMIT_RE.fullmatch(commit) is None:
            raise ContractError("git returned a malformed first-parent commit")
        changed_paths = [path for path in fields[index + 1].splitlines() if path]
        if any(is_governed_bridge_path(path) for path in changed_paths):
            return BridgeSourceIdentity(
                bridge_source_sha=source_sha,
                bridge_build_sha=commit,
            )
    raise ContractError("bridge history contains no governed runtime/build source")


def require_orchestration_caller(
    event_name: str, actor: str, triggering_actor: str
) -> None:
    """Keep untrusted callers from turning the environment PAT into a deputy.

    Scheduled executions are authorized by the trusted default-branch workflow.
    A workflow_run continuation and a manual dispatch additionally require both
    GitHub actor identities to be the repository owner before any
    environment-scoped credential is used. A workflow_run event always executes
    the default-branch workflow definition, so the continuation cannot be
    redefined from a pull request or a fork.
    """
    if event_name == "schedule":
        return
    if (
        event_name in ("workflow_dispatch", "workflow_run")
        and actor == REPOSITORY_OWNER
        and triggering_actor == REPOSITORY_OWNER
    ):
        return
    raise ContractError(
        "stable orchestration requires a schedule event, or an owner-initiated "
        "workflow_dispatch or workflow_run continuation with owner actor and "
        "triggering_actor"
    )


def render_step_summary(plan: OrchestrationPlan) -> str:
    target = plan.release_target
    lines = [
        "### Stable Web bridge release orchestration",
        "",
        f"- Action: `{plan.action.value}`",
        f"- Reason: {plan.reason}",
        f"- Correlation: `{plan.correlation_id}`",
        f"- Bridge source: `{plan.provenance.bridge_source_sha}`",
        f"- Governed bridge build: `{plan.provenance.bridge_build_sha}`",
        f"- llama.cpp: `{plan.provenance.upstream_tag}@{plan.provenance.upstream_commit}`",
        f"- Native release: `{plan.provenance.native_repo}@{plan.provenance.native_release_tag}`",
        f"- Native manifest SHA-256: `{plan.provenance.native_manifest_sha256}`",
    ]
    if target is not None:
        lines.append(
            f"- Output release: `{target.release_tag}` (rebuild `{target.release_rebuild}`)"
        )
    if plan.candidate_run_id:
        lines.append(f"- Candidate run: `{plan.candidate_run_id}`")
    if plan.qualification_run_id:
        lines.append(f"- Qualification run: `{plan.qualification_run_id}`")
    if plan.in_flight_workflow:
        lines.append(
            f"- In flight: `{plan.in_flight_workflow}` run `{plan.in_flight_run_id}`"
        )
    if plan.dispatch_workflow:
        lines.append(
            f"- Dispatched: `{plan.dispatch_workflow}` at `{plan.dispatch_ref}` "
            f"as `{plan.dispatch_run_name}` (run `{plan.dispatched_run_id}`)"
        )
    lines.append("")
    return "\n".join(lines)


_FAILING_ACTIONS = frozenset({OrchestrationAction.BLOCKED})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    scan = subparsers.add_parser(
        "scan-native", help="Extract exact provenance from a native assets.json"
    )
    scan.add_argument("--manifest", required=True, type=Path)
    scan.add_argument("--native-release-tag", required=True)
    scan.add_argument("--bridge-source-sha", required=True)
    scan.add_argument("--bridge-build-sha", required=True)
    scan.add_argument("--native-release-published-at", required=True)
    scan.add_argument(
        "--channel",
        required=True,
        choices=sorted(_CHANNELS),
        help="Channel the caller asked for; a mismatch fails closed",
    )
    scan.add_argument("--output-json", type=Path)

    source = subparsers.add_parser(
        "resolve-bridge-source",
        help="Resolve current source and governed runtime/build identity",
    )
    source.add_argument("--repository", required=True, type=Path)
    source.add_argument("--head", default="HEAD")
    source.add_argument("--output-json", type=Path)

    select = subparsers.add_parser(
        "select-stable-native-backlog",
        help="Select every post-baseline stable native release",
    )
    select.add_argument("--releases-json", required=True, type=Path)
    select.add_argument("--output-json", type=Path)

    orchestrate = subparsers.add_parser(
        "orchestrate", help="Advance the stable release pipeline by one exact step"
    )
    orchestrate.add_argument("--provenance-json", required=True, type=Path)
    orchestrate.add_argument("--workspace", required=True, type=Path)
    orchestrate.add_argument("--output-plan-json", type=Path)
    orchestrate.add_argument("--step-summary-file", type=Path)
    orchestrate.add_argument("--dry-run", action="store_true")

    backlog = subparsers.add_parser(
        "orchestrate-backlog",
        help="Advance each exact stable provenance by at most one stage",
    )
    backlog.add_argument("--provenance-list-json", required=True, type=Path)
    backlog.add_argument("--workspace", required=True, type=Path)
    backlog.add_argument("--output-plan-json", type=Path)
    backlog.add_argument("--step-summary-file", type=Path)
    backlog.add_argument("--dry-run", action="store_true")
    return parser


def _provenance_to_dict(provenance: NativeProvenance) -> dict[str, str]:
    return {
        "bridge_source_sha": provenance.bridge_source_sha,
        "bridge_build_sha": provenance.bridge_build_sha,
        "upstream_tag": provenance.upstream_tag,
        "upstream_commit": provenance.upstream_commit,
        "native_repo": provenance.native_repo,
        "native_release_tag": provenance.native_release_tag,
        "native_commit": provenance.native_commit,
        "native_manifest_sha256": provenance.native_manifest_sha256,
        "native_release_published_at": provenance.native_release_published_at,
    }


def _load_provenance_payload(payload: Any, label: str) -> NativeProvenance:
    if not isinstance(payload, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    expected = {
        "bridge_source_sha",
        "bridge_build_sha",
        "upstream_tag",
        "upstream_commit",
        "native_repo",
        "native_release_tag",
        "native_commit",
        "native_manifest_sha256",
        "native_release_published_at",
    }
    if set(payload) != expected:
        raise ContractError(f"{label} has missing or unexpected fields")
    return require_stable_provenance(NativeProvenance(**payload))


def _load_provenance(path: Path) -> NativeProvenance:
    payload = _strict_json_loads(path.read_text(encoding="utf-8"), "provenance")
    return _load_provenance_payload(payload, "provenance")


def _load_provenance_backlog(path: Path) -> list[NativeProvenance]:
    payload = _strict_json_loads(
        path.read_text(encoding="utf-8"), "provenance backlog"
    )
    if not isinstance(payload, list):
        raise ContractError("provenance backlog root must be a JSON array")
    provenances = [
        _load_provenance_payload(item, f"provenance backlog entry {index}")
        for index, item in enumerate(payload)
    ]
    correlations: set[str] = set()
    native_tags: set[str] = set()
    for provenance in provenances:
        correlation = compute_correlation_id(provenance)
        if correlation in correlations or provenance.native_release_tag in native_tags:
            raise ContractError("provenance backlog contains a duplicate pipeline")
        correlations.add(correlation)
        native_tags.add(provenance.native_release_tag)
    provenances.sort(
        key=lambda value: (
            value.native_release_published_at,
            _native_release_order(value.native_release_tag),
        )
    )
    return provenances


def _require_cli_caller() -> None:
    require_orchestration_caller(
        os.environ.get("GITHUB_EVENT_NAME", ""),
        os.environ.get("GITHUB_ACTOR", ""),
        os.environ.get("GITHUB_TRIGGERING_ACTOR", ""),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.subcommand == "scan-native":
        provenance = scan_native_provenance(
            manifest_path=args.manifest,
            native_release_tag=args.native_release_tag,
            bridge_source_sha=args.bridge_source_sha,
            bridge_build_sha=args.bridge_build_sha,
            channel=args.channel,
            native_release_published_at=args.native_release_published_at,
        )
        payload = json.dumps(_provenance_to_dict(provenance), indent=2, sort_keys=True)
        if args.output_json:
            args.output_json.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0

    if args.subcommand == "resolve-bridge-source":
        identity = resolve_bridge_source_identity(args.repository, args.head)
        payload = json.dumps(
            {
                "bridge_source_sha": identity.bridge_source_sha,
                "bridge_build_sha": identity.bridge_build_sha,
            },
            indent=2,
            sort_keys=True,
        )
        if args.output_json:
            args.output_json.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0

    if args.subcommand == "select-stable-native-backlog":
        releases = _strict_json_loads(
            args.releases_json.read_text(encoding="utf-8"),
            "native release listing",
        )
        if not isinstance(releases, list):
            raise ContractError("native release listing must be a JSON array")
        selected = select_stable_native_backlog(releases)
        payload = json.dumps(selected, indent=2)
        if args.output_json:
            args.output_json.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0

    _require_cli_caller()
    gateway = GhGateway(
        read_token=os.environ.get("GH_TOKEN", ""),
        dispatch_token=os.environ.get("WEBGPU_BRIDGE_ASSETS_PAT"),
    )
    args.workspace.mkdir(parents=True, exist_ok=True)

    if args.subcommand == "orchestrate-backlog":
        provenances = _load_provenance_backlog(args.provenance_list_json)
        plans: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        reserved_release_tags: set[str] = set()
        satisfied_correlation_ids: set[str] = set()
        publication_barrier_native_tag: str | None = None
        newest_native_order = max(
            (_native_release_order(value.native_release_tag) for value in provenances),
            default=(),
        )
        for provenance in provenances:
            correlation_id = compute_correlation_id(provenance)
            pipeline_workspace = args.workspace / correlation_id
            pipeline_workspace.mkdir(parents=True, exist_ok=True)
            try:
                plan = advance_pipeline(
                    gateway,
                    provenance=provenance,
                    workspace=pipeline_workspace,
                    dry_run=args.dry_run,
                    reserved_release_tags=reserved_release_tags,
                    satisfied_correlation_ids=satisfied_correlation_ids,
                    publication_allowed=publication_barrier_native_tag is None,
                    publication_barrier_native_tag=publication_barrier_native_tag,
                    newer_native_scanned=(
                        _native_release_order(provenance.native_release_tag)
                        < newest_native_order
                    ),
                )
                plans.append(plan.to_dict())
                if plan.action is OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE:
                    satisfied_correlation_ids.add(correlation_id)
                elif plan.release_target is not None:
                    reserved_release_tags.add(plan.release_target.release_tag)
                if publication_barrier_native_tag is None and plan.action not in (
                    OrchestrationAction.NOOP,
                    OrchestrationAction.SUPERSEDED,
                    OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE,
                ):
                    publication_barrier_native_tag = provenance.native_release_tag
                if args.step_summary_file:
                    with open(args.step_summary_file, "a", encoding="utf-8") as handle:
                        handle.write(render_step_summary(plan) + "\n")
            except ContractError as error:
                errors.append(
                    {
                        "correlation_id": correlation_id,
                        "native_release_tag": provenance.native_release_tag,
                        "error": str(error),
                    }
                )
                if args.step_summary_file:
                    with open(args.step_summary_file, "a", encoding="utf-8") as handle:
                        handle.write(
                            "### Blocked stable Web bridge release orchestration\n\n"
                            f"- Correlation: `{correlation_id}`\n"
                            f"- Native release: `{provenance.native_release_tag}`\n"
                            f"- Error: {error}\n\n"
                        )
                # A transport/readback error can mean a dispatch occurred but
                # its state is not yet observable. Stop before another backlog
                # entry can claim a colliding output identity under uncertainty.
                break
        result = {
            "schema_version": 1,
            "plans": plans,
            "errors": errors,
        }
        payload = json.dumps(result, indent=2, sort_keys=True)
        if args.output_plan_json:
            args.output_plan_json.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        blocked = [plan for plan in plans if plan["action"] in {
            action.value for action in _FAILING_ACTIONS
        }]
        if errors or blocked:
            print("error: one or more stable pipelines are blocked", file=sys.stderr)
            return 1
        return 0

    provenance = _load_provenance(args.provenance_json)
    plan = advance_pipeline(
        gateway,
        provenance=provenance,
        workspace=args.workspace,
        dry_run=args.dry_run,
    )
    payload = json.dumps(plan.to_dict(), indent=2, sort_keys=True)
    if args.output_plan_json:
        args.output_plan_json.write_text(payload + "\n", encoding="utf-8")
    if args.step_summary_file:
        with open(args.step_summary_file, "a", encoding="utf-8") as handle:
            handle.write(render_step_summary(plan) + "\n")
    print(payload)
    if plan.action in _FAILING_ACTIONS:
        print(f"error: {plan.reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        raise SystemExit(f"error: {error}") from error
