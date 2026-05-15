#!/usr/bin/env python3
"""Static checks for reliability-focused browser smoke and workflow coverage."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_required(relative_path: str, errors: list[str]) -> str:
    path = ROOT / relative_path
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"required file is not readable: {relative_path}: {exc}")
        return ""


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    errors: list[str] = []
    smoke = read_required("scripts/state_persistence_browser_smoke.py", errors)
    ci = read_required(".github/workflows/ci.yml", errors)
    publish = read_required(".github/workflows/publish_assets.yml", errors)
    auto_update = read_required(".github/workflows/auto_llama_cpp_update.yml", errors)
    version = read_required("llama_cpp.version", errors).strip()
    agents = read_required("AGENTS.md", errors)
    readme = read_required("README.md", errors)
    contributing = read_required("CONTRIBUTING.md", errors)

    for name, workflow in (
        ("ci.yml", ci),
        ("publish_assets.yml", publish),
        ("auto_llama_cpp_update.yml", auto_update),
    ):
        require(
            "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24" in workflow,
            f"{name} must opt into Node 24 action runtime to catch Node 20 deprecation breakage early",
            errors,
        )
        require(
            "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true" not in workflow,
            f"{name} must quote FORCE_JAVASCRIPT_ACTIONS_TO_NODE24 so the workflow env value is a string",
            errors,
        )

    require(
        version.startswith("b") and version[1:].isdigit(),
        "llama_cpp.version must contain a llama.cpp release tag like b9165",
        errors,
    )
    require(
        "tr -d '[:space:]' < llama_cpp.version" in ci
        and "Resolve llama.cpp pin" in ci
        and "workflow_dispatch:" in ci
        and "LLAMA_CPP_TAG: b9116" not in ci,
        "ci.yml must resolve the llama.cpp tag from llama_cpp.version, support explicit dispatch, and avoid hard-coded stale defaults",
        errors,
    )
    require(
        "REQUESTED_LLAMA_CPP_TAG" in publish
        and "tr -d '[:space:]' < llama_cpp.version" in publish
        and "outputs:" in publish
        and "llama_cpp_tag: ${{ steps.resolve-publish-parameters.outputs.llama_cpp_tag }}" in publish
        and "LLAMA_CPP_TAG: ${{ needs.build-bridge-assets.outputs.llama_cpp_tag }}" in publish
        and "default: b9116" not in publish
        and "|| 'b9116'" not in publish,
        "publish_assets.yml must default to llama_cpp.version, pass the resolved tag across jobs, and still allow a manual override",
        errors,
    )
    require(
        "concurrency:" in auto_update
        and "group: llama-cpp-auto-update" in auto_update
        and "UPDATE_BRANCH: automation/bump-llama-cpp" in auto_update
        and "ggml-org/llama.cpp" in auto_update
        and "create-pull-request" in auto_update
        and "actions: write" in auto_update
        and "id: create-pr" in auto_update
        and "gh workflow run ci.yml --repo \"$GITHUB_REPOSITORY\" --ref \"$UPDATE_BRANCH\"" in auto_update
        and "body-path: /tmp/llama_cpp_update_pr.md" in auto_update
        and "Upstream changelog" in auto_update
        and "not racing a non-automation PR" in auto_update,
        "auto_llama_cpp_update.yml must update one stable PR branch with upstream changelog context, dispatch CI for bot updates, and avoid racing human PRs",
        errors,
    )

    require(
        "--model-url" in smoke and "--model-sha256" in smoke,
        "browser smoke must support an integrity-checked model-backed state round-trip",
        errors,
    )
    require(
        "model_cache_dir.expanduser().resolve()" in smoke,
        "browser smoke must expand '~' in the model cache path so actions/cache uses the same directory",
        errors,
    )
    require(
        "stateSaveBytes" in smoke
        and "stateLoadBytes" in smoke
        and "createCompletion" in smoke
        and "tokenize" in smoke,
        "browser smoke must exercise actual state save/load after prompt evaluation",
        errors,
    )
    require(
        "detachedAfterLoadTransfer" in smoke and "workerSaveSnapshotReturned" in smoke,
        "browser smoke must assert worker stateLoadBytes transfer detaches ArrayBuffers and stateSaveBytes returns bytes",
        errors,
    )
    require(
        "--artifacts-dir" in smoke and "screenshot" in smoke and "state-smoke-result.json" in smoke,
        "browser smoke must write debuggable failure artifacts",
        errors,
    )
    require(
        "LLAMA_WEBGPU_SMOKE_MODEL_URL" in ci
        and "LLAMA_WEBGPU_SMOKE_MODEL_SHA256" in ci
        and "Run state persistence browser smoke" in ci,
        "CI must run model-backed state persistence smoke with a pinned model URL and checksum",
        errors,
    )
    require(
        "state-persistence-smoke-artifacts" in ci and "if: failure()" in ci,
        "CI must upload browser smoke diagnostics on failure",
        errors,
    )
    require(
        "LLAMA_WEBGPU_SMOKE_ARTIFACTS_DIR: ${{ runner.temp }}" not in ci,
        "CI must not use runner context in job-level env for smoke artifacts",
        errors,
    )
    require(
        "Verify CI reliability contract" in ci and "scripts/verify_ci_reliability.py" in ci,
        "CI must run this reliability contract check",
        errors,
    )
    require(
        "Agent PR Workflow" in agents
        and "independent review" in agents
        and "state_persistence_browser_smoke.py" in agents
        and "llama_cpp.version" in agents
        and "auto_llama_cpp_update.yml" in agents,
        "AGENTS.md must document the agent PR workflow, browser smoke expectations, and llama.cpp auto-update policy",
        errors,
    )
    require(
        "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24" in readme
        and "state-persistence-smoke-artifacts" in readme
        and "scripts/verify_ci_reliability.py" in readme
        and "llama_cpp.version" in readme
        and "auto_llama_cpp_update.yml" in readme,
        "README.md must document CI reliability, diagnostics, Node 24 action-runtime coverage, and llama.cpp pin automation",
        errors,
    )
    require(
        "Agent Workflow Guardrails" in contributing
        and "scripts/verify_ci_reliability.py" in contributing
        and "--model-sha256" in contributing
        and "llama_cpp.version" in contributing,
        "CONTRIBUTING.md must document maintainer/agent workflow guardrails, checksum-pinned smoke usage, and llama.cpp pin handling",
        errors,
    )

    if errors:
        print("CI reliability contract failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("CI reliability contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
