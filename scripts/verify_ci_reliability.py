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
    agents = read_required("AGENTS.md", errors)
    readme = read_required("README.md", errors)
    contributing = read_required("CONTRIBUTING.md", errors)

    for name, workflow in (("ci.yml", ci), ("publish_assets.yml", publish)):
        require(
            "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24" in workflow,
            f"{name} must opt into Node 24 action runtime to catch Node 20 deprecation breakage early",
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
        "Verify CI reliability contract" in ci and "scripts/verify_ci_reliability.py" in ci,
        "CI must run this reliability contract check",
        errors,
    )
    require(
        "Agent PR Workflow" in agents
        and "independent review" in agents
        and "state_persistence_browser_smoke.py" in agents,
        "AGENTS.md must document the agent PR workflow and browser smoke expectations",
        errors,
    )
    require(
        "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24" in readme
        and "state-persistence-smoke-artifacts" in readme
        and "scripts/verify_ci_reliability.py" in readme,
        "README.md must document CI reliability, diagnostics, and Node 24 action-runtime coverage",
        errors,
    )
    require(
        "Agent Workflow Guardrails" in contributing
        and "scripts/verify_ci_reliability.py" in contributing
        and "--model-sha256" in contributing,
        "CONTRIBUTING.md must document maintainer/agent workflow guardrails and checksum-pinned smoke usage",
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
