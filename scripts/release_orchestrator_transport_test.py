#!/usr/bin/env python3
"""Contract tests for the ``gh``-backed orchestrator transport."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from release_contract import ASSETS_REPOSITORY, BRIDGE_REPOSITORY
import release_orchestrator_model as model
import release_orchestrator_transport as transport


class GhGatewayProcessBoundaryTest(unittest.TestCase):
    """Exercise the real argv/stdin boundary, not only the fake gateway API."""

    def _fake_gh(self, directory: Path) -> tuple[Path, Path]:
        executable = directory / "gh"
        log = directory / "gh-log.json"
        executable.write_text(
            """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

Path(os.environ["GH_FAKE_LOG"]).write_text(json.dumps({
    "argv": sys.argv[1:],
    "stdin": sys.stdin.read(),
    "gh_token": os.environ.get("GH_TOKEN"),
    "github_token_present": "GITHUB_TOKEN" in os.environ,
}), encoding="utf-8")
sys.stdout.write(os.environ.get("GH_FAKE_STDOUT", ""))
raise SystemExit(int(os.environ.get("GH_FAKE_EXIT", "0")))
""",
            encoding="utf-8",
        )
        executable.chmod(0o700)
        return executable, log

    def test_workflow_dispatch_uses_json_stdin_and_accepts_url_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _, log = self._fake_gh(directory)
            env = {
                "PATH": f"{directory}{os.pathsep}{os.environ.get('PATH', '')}",
                "GH_FAKE_LOG": str(log),
                "GH_FAKE_STDOUT": (
                    f"https://github.com/{BRIDGE_REPOSITORY}/actions/runs/501\n"
                ),
                "GITHUB_TOKEN": "must-be-removed",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                gateway = transport.GhGateway(
                    read_token="read-token", dispatch_token="dispatch-token"
                )
                gateway.dispatch_workflow(
                    workflow_file=model.CANDIDATE_WORKFLOW_FILE,
                    ref="main",
                    inputs={"zeta": "last", "alpha": "first"},
                )
            observed = json.loads(log.read_text(encoding="utf-8"))
        self.assertEqual(
            observed["argv"],
            [
                "workflow",
                "run",
                model.CANDIDATE_WORKFLOW_FILE,
                "--repo",
                BRIDGE_REPOSITORY,
                "--ref",
                "main",
                "--json",
            ],
        )
        self.assertEqual(
            json.loads(observed["stdin"]), {"alpha": "first", "zeta": "last"}
        )
        self.assertEqual(observed["gh_token"], "dispatch-token")
        self.assertIs(observed["github_token_present"], False)

    def test_release_verify_uses_json_format_and_read_credential(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _, log = self._fake_gh(directory)
            env = {
                "PATH": f"{directory}{os.pathsep}{os.environ.get('PATH', '')}",
                "GH_FAKE_LOG": str(log),
                "GH_FAKE_STDOUT": "{}\n",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                gateway = transport.GhGateway(
                    read_token="read-token", dispatch_token="dispatch-token"
                )
                self.assertEqual(
                    gateway.release_attestation(
                        repository=ASSETS_REPOSITORY, release_tag="v0.1.40"
                    ),
                    {},
                )
            observed = json.loads(log.read_text(encoding="utf-8"))
        self.assertEqual(
            observed["argv"],
            [
                "release",
                "verify",
                "v0.1.40",
                "--repo",
                ASSETS_REPOSITORY,
                "--format",
                "json",
            ],
        )
        self.assertEqual(observed["gh_token"], "read-token")


if __name__ == "__main__":
    unittest.main()
