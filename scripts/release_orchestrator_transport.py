"""Live GitHub transport for the stable release orchestrator.

``Gateway`` is the protocol every live proof and dispatch is written against;
``GhGateway`` implements it with the ``gh`` CLI, keeping read traffic and
dispatch on distinct tokens.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, Sequence

from release_contract import BRIDGE_REPOSITORY, ContractError, _strict_json_loads


class Gateway(Protocol):
    def api_json(self, path: str, *, paginate: bool = ..., privileged: bool = ...) -> Any: ...

    def download_bytes(self, path: str, *, accept: str, privileged: bool = ...) -> bytes: ...

    def dispatch_identity(self) -> str | None: ...

    def release_attestation(self, *, repository: str, release_tag: str) -> Any: ...

    def dispatch_workflow(
        self, *, workflow_file: str, ref: str, inputs: Mapping[str, str]
    ) -> None: ...

    def sleep(self, seconds: float) -> None: ...

    def utc_now(self) -> str: ...


class GhGateway:
    """``gh``-backed transport. Read traffic and dispatch use distinct tokens."""

    def __init__(self, *, read_token: str, dispatch_token: str | None) -> None:
        self._read_token = read_token
        self._dispatch_token = dispatch_token or None

    def _run(self, args: Sequence[str], *, privileged: bool, binary: bool) -> bytes:
        token = self._dispatch_token if privileged else self._read_token
        if not token:
            raise ContractError(
                "no credential is available for this GitHub request; refusing to continue"
            )
        env = dict(os.environ)
        env["GH_TOKEN"] = token
        env.pop("GITHUB_TOKEN", None)
        completed = subprocess.run(
            list(args), env=env, capture_output=True, check=False
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", "replace").strip()
            raise ContractError(f"gh {' '.join(args[1:])} failed: {stderr}")
        return completed.stdout if binary else completed.stdout

    def api_json(self, path: str, *, paginate: bool = False, privileged: bool = False) -> Any:
        args = ["gh", "api"]
        if paginate:
            args += ["--paginate", "--slurp"]
        args.append(path)
        stdout = self._run(args, privileged=privileged, binary=False)
        return _strict_json_loads(stdout.decode("utf-8", "strict"), f"gh api {path}")

    def download_bytes(self, path: str, *, accept: str, privileged: bool = False) -> bytes:
        return self._run(
            ["gh", "api", "-H", f"Accept: {accept}", path],
            privileged=privileged,
            binary=True,
        )

    def dispatch_identity(self) -> str | None:
        if not self._dispatch_token:
            return None
        try:
            payload = self.api_json("user", privileged=True)
        except ContractError:
            return None
        login = payload.get("login") if isinstance(payload, Mapping) else None
        return login if isinstance(login, str) and login else None

    def release_attestation(self, *, repository: str, release_tag: str) -> Any:
        stdout = self._run(
            [
                "gh",
                "release",
                "verify",
                release_tag,
                "--repo",
                repository,
                "--format",
                "json",
            ],
            privileged=False,
            binary=False,
        )
        return _strict_json_loads(
            stdout.decode("utf-8", "strict"),
            f"gh release verify {repository}@{release_tag}",
        )

    def dispatch_workflow(
        self, *, workflow_file: str, ref: str, inputs: Mapping[str, str]
    ) -> None:
        if not self._dispatch_token:
            raise ContractError("workflow dispatch requires an orchestrator credential")
        env = dict(os.environ)
        env["GH_TOKEN"] = self._dispatch_token
        env.pop("GITHUB_TOKEN", None)
        completed = subprocess.run(
            [
                "gh",
                "workflow",
                "run",
                workflow_file,
                "--repo",
                BRIDGE_REPOSITORY,
                "--ref",
                ref,
                "--json",
            ],
            input=json.dumps(dict(inputs), sort_keys=True).encode("utf-8"),
            env=env,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", "replace").strip()
            raise ContractError(f"dispatching {workflow_file} failed: {stderr}")

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def utc_now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
