#!/usr/bin/env python3
"""Static checks for reliability-focused browser smoke and workflow coverage."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = (ROOT / "scripts" / "state_persistence_browser_smoke.py").read_text(encoding="utf-8")
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
PUBLISH = (ROOT / ".github" / "workflows" / "publish_assets.yml").read_text(encoding="utf-8")


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    errors: list[str] = []

    for name, workflow in (("ci.yml", CI), ("publish_assets.yml", PUBLISH)):
        require(
            "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24" in workflow,
            f"{name} must opt into Node 24 action runtime to catch Node 20 deprecation breakage early",
            errors,
        )

    require(
        "--model-url" in SMOKE and "--model-sha256" in SMOKE,
        "browser smoke must support an integrity-checked model-backed state round-trip",
        errors,
    )
    require(
        "model_cache_dir.expanduser().resolve()" in SMOKE,
        "browser smoke must expand '~' in the model cache path so actions/cache uses the same directory",
        errors,
    )
    require(
        "stateSaveBytes" in SMOKE
        and "stateLoadBytes" in SMOKE
        and "createCompletion" in SMOKE
        and "tokenize" in SMOKE,
        "browser smoke must exercise actual state save/load after prompt evaluation",
        errors,
    )
    require(
        "detachedAfterLoadTransfer" in SMOKE and "workerSaveSnapshotReturned" in SMOKE,
        "browser smoke must assert worker stateLoadBytes transfer detaches ArrayBuffers and stateSaveBytes returns bytes",
        errors,
    )
    require(
        "--artifacts-dir" in SMOKE and "screenshot" in SMOKE and "state-smoke-result.json" in SMOKE,
        "browser smoke must write debuggable failure artifacts",
        errors,
    )
    require(
        "LLAMA_WEBGPU_SMOKE_MODEL_URL" in CI
        and "LLAMA_WEBGPU_SMOKE_MODEL_SHA256" in CI
        and "Run state persistence browser smoke" in CI,
        "CI must run model-backed state persistence smoke with a pinned model URL and checksum",
        errors,
    )
    require(
        "state-persistence-smoke-artifacts" in CI and "if: failure()" in CI,
        "CI must upload browser smoke diagnostics on failure",
        errors,
    )
    require(
        "Verify CI reliability contract" in CI and "scripts/verify_ci_reliability.py" in CI,
        "CI must run this reliability contract check",
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
