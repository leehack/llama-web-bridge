#!/usr/bin/env python3
"""Fail-closed contract tests for the daily stable release state machine."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest import mock

import stable_release_orchestrator as sro
from generate_release_manifest import ARTIFACTS, generate
from release_contract import (
    ASSETS_REPOSITORY,
    BRIDGE_REPOSITORY,
    ContractError,
    NATIVE_REPOSITORY,
)
from release_publication_state import PUBLICATION_FILES
from release_contract_test import release_attestation

# The heavy attestation fixtures are pinned by the qualification suite that
# produces them. Restating them here would let this suite pass against an
# attestation the real gate would reject.
from release_qualification_test import (
    qualification_environment,
    qualification_identity,
    speech_phase,
    tts_phase,
)
import release_qualification as rq


BRIDGE_SHA = "565c8396597ea7c0fb4e8d5d966da8d884b156d8"
ADVANCED_BRIDGE_SHA = "9" * 40
UPSTREAM_COMMIT = "bb4caa7540188872173c44d161602d9271386413"
NATIVE_COMMIT = "1" * 40
NATIVE_MANIFEST_SHA = (
    "2e5d29d7f98f0d71e75d3fa63b7c55f3b2a7933247cc34ea2b1c5e053d142452"
)
CANDIDATE_RUN_ID = "32919086955"
QUALIFICATION_RUN_ID = "32919086977"
CANDIDATE_ARTIFACT_ID = 7
QUALIFICATION_ARTIFACT_ID = 9
DEFAULT_BRANCH = "main"
HEAD_SHA = "a" * 40
ASSETS_TAG_COMMIT = "c" * 40
OWNER = BRIDGE_REPOSITORY.split("/")[0]
EMSCRIPTEN_VERSION = "6.0.8"
NATIVE_PUBLISHED_AT = "2026-08-19T12:34:56Z"
LEGACY_BRIDGE_SHA = "0bdc8286fd52b70da27f5b039e1b4278361da0be"
LEGACY_UPSTREAM_COMMIT = "c1d0e7a004015f23bc0233470b747b596f29b264"
LEGACY_NATIVE_COMMIT = "28fca14873d4b4c531bef4425b261e2b911bdcce"
LEGACY_NATIVE_MANIFEST_SHA = (
    "811fda999e70c3ad2716d1c196688dd38db62cf11a78044855ca94f71fabed45"
)
LEGACY_CANDIDATE_RUN_ID = "33225744070"
LEGACY_MANUAL_QUALIFICATION_GATES = {
    "state_persistence": "passed",
    "multimodal": "passed",
    "speech_to_text": "required-local-attestation",
    "text_to_speech": "required-local-attestation",
}
LEGACY_MANUAL_UNPROVEN_CAPABILITIES = {
    "real_device_intelligibility": "unproven",
    "real_device_playback": "unproven",
    "speaker_reference_fidelity": "unproven",
}


def make_provenance(**overrides: Any) -> sro.NativeProvenance:
    fields: dict[str, Any] = {
        "bridge_source_sha": BRIDGE_SHA,
        "upstream_tag": "v0.2.0",
        "upstream_commit": UPSTREAM_COMMIT,
        "native_repo": NATIVE_REPOSITORY,
        "native_release_tag": "v0.2.0",
        "native_commit": NATIVE_COMMIT,
        "native_manifest_sha256": NATIVE_MANIFEST_SHA,
        "native_release_published_at": NATIVE_PUBLISHED_AT,
    }
    fields.update(overrides)
    return sro.NativeProvenance(**fields)


def make_legacy_v0140_provenance(**overrides: Any) -> sro.NativeProvenance:
    fields: dict[str, Any] = {
        "bridge_source_sha": LEGACY_BRIDGE_SHA,
        "upstream_tag": "v0.3.0",
        "upstream_commit": LEGACY_UPSTREAM_COMMIT,
        "native_repo": NATIVE_REPOSITORY,
        "native_release_tag": "v0.3.0",
        "native_commit": LEGACY_NATIVE_COMMIT,
        "native_manifest_sha256": LEGACY_NATIVE_MANIFEST_SHA,
        "native_release_published_at": "2026-08-28T12:34:56Z",
    }
    fields.update(overrides)
    return sro.NativeProvenance(**fields)


def native_manifest(**overrides: Any) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "native_release_tag": "v0.2.0",
        "llama_cpp_tag": "v0.2.0",
        "llama_cpp_commit": UPSTREAM_COMMIT,
        "native_commit": NATIVE_COMMIT,
    }
    manifest.update(overrides)
    return manifest


def write_bridge_candidate(
    directory: Path,
    *,
    release_tag: str,
    release_rebuild: int,
    correlation_id: str,
    bridge_commit: str = BRIDGE_SHA,
    run_id: str = CANDIDATE_RUN_ID,
    marker: bytes = b"candidate",
    upstream_tag: str = "v0.2.0",
    upstream_commit: str = UPSTREAM_COMMIT,
    native_release_tag: str = "v0.2.0",
    native_manifest_sha256: str = NATIVE_MANIFEST_SHA,
    native_commit: str = NATIVE_COMMIT,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(ARTIFACTS):
        (directory / name).write_bytes(marker + f"-{index}-{name}".encode("utf-8"))
    generate(
        argparse.Namespace(
            out_dir=directory,
            release_tag=release_tag,
            release_rebuild=release_rebuild,
            assets_repo=ASSETS_REPOSITORY,
            bridge_repo=BRIDGE_REPOSITORY,
            bridge_commit=bridge_commit,
            upstream_repo="ggml-org/llama.cpp",
            upstream_tag=upstream_tag,
            upstream_commit=upstream_commit,
            native_repo=NATIVE_REPOSITORY,
            native_release_tag=native_release_tag,
            native_manifest_sha256=native_manifest_sha256,
            native_commit=native_commit,
            emscripten_version=EMSCRIPTEN_VERSION,
            orchestrator_correlation_id=correlation_id,
            github_run_id=run_id,
            github_run_url=(
                f"https://github.com/{BRIDGE_REPOSITORY}/actions/runs/{run_id}"
            ),
        )
    )


def rewrite_legacy_candidate_manifest(directory: Path) -> None:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["qualification_gates"] = LEGACY_MANUAL_QUALIFICATION_GATES
    manifest["unproven_capabilities"] = LEGACY_MANUAL_UNPROVEN_CAPABILITIES
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def flat_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(name, data)
    return buffer.getvalue()


def directory_members(directory: Path) -> dict[str, bytes]:
    return {name: (directory / name).read_bytes() for name in PUBLICATION_FILES}


def release_payload(
    *,
    tag: str,
    body: str,
    members: dict[str, bytes],
    release_id: int = 4242,
    immutable: bool = True,
    published_at: str = "2026-08-20T03:24:11Z",
    draft: bool = False,
    prerelease: bool | None = None,
    target_commitish: str = ASSETS_TAG_COMMIT,
    asset_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    assets = []
    for index, (name, data) in enumerate(sorted(members.items())):
        asset = {
            "id": 900 + index,
            "name": name,
            "state": "uploaded",
            "size": len(data),
            "digest": f"sha256:{hashlib.sha256(data).hexdigest()}",
        }
        if asset_overrides and name in asset_overrides:
            asset.update(asset_overrides[name])
        assets.append(asset)
    if prerelease is None:
        prerelease = "-" in tag
    return {
        "id": release_id,
        "tag_name": tag,
        "name": tag,
        "draft": draft,
        "prerelease": prerelease,
        "immutable": immutable,
        "target_commitish": target_commitish,
        "published_at": published_at,
        "body": body,
        "assets": assets,
    }


def asset_release_stub(tag: str = "v0.1.39") -> dict[str, Any]:
    return {
        "tag_name": tag,
        "draft": False,
        "prerelease": "-" in tag,
        "published_at": "2026-08-20T03:24:11Z",
        "body": "",
    }


def run_payload(
    *,
    run_id: str,
    path: str,
    run_name: str,
    status: str = "completed",
    conclusion: str | None = "success",
    head_branch: str = DEFAULT_BRANCH,
    head_sha: str = HEAD_SHA,
    run_attempt: int = 1,
    event: str = "workflow_dispatch",
    api_name: str | None = None,
    actor: str = OWNER,
    triggering_actor: str = OWNER,
) -> dict[str, Any]:
    if api_name is None:
        api_name = run_name
    return {
        "id": int(run_id),
        # Workflows with ``run-name`` expose the rendered correlation string in
        # both fields.  Keep the fixture aligned with the live Actions API.
        "name": api_name,
        "display_title": run_name,
        "path": path,
        "event": event,
        "status": status,
        "conclusion": conclusion if status == "completed" else None,
        "head_branch": head_branch,
        "head_sha": head_sha,
        "run_attempt": run_attempt,
        "repository": {"full_name": BRIDGE_REPOSITORY},
        "head_repository": {"full_name": BRIDGE_REPOSITORY},
        "actor": {"login": actor},
        "triggering_actor": {"login": triggering_actor},
    }


def runs_response(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """The live API answers an object holding workflow_runs, never a bare list."""
    return {"total_count": len(runs), "workflow_runs": runs}


def artifact_inventory(
    *, run_id: str, name: str, artifact_id: int, extra: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = [
        {
            "id": artifact_id,
            "name": name,
            "expired": False,
            "workflow_run": {"id": int(run_id)},
        }
    ]
    artifacts.extend(extra or [])
    return {"total_count": len(artifacts), "artifacts": artifacts}


class FakeGateway:
    """Deterministic stand-in for the live gh transport."""

    def __init__(
        self,
        *,
        json_routes: dict[str, Any] | None = None,
        blob_routes: dict[str, bytes] | None = None,
        identity: str | None = OWNER,
        governance: Any = None,
        release_attestations: dict[tuple[str, str], Any] | None = None,
        now: str = "2026-08-30T00:00:00Z",
    ) -> None:
        self.json_routes = dict(json_routes or {})
        self.blob_routes = dict(blob_routes or {})
        self.identity = identity
        self.release_attestations = dict(release_attestations or {})
        self.now = now
        self.dispatches: list[dict[str, Any]] = []
        self.slept: list[float] = []
        self.api_paths: list[str] = []
        self.json_routes.setdefault(
            f"repos/{BRIDGE_REPOSITORY}", {"default_branch": DEFAULT_BRANCH}
        )
        self.json_routes.setdefault(
            f"repos/{BRIDGE_REPOSITORY}/environments/bridge-assets-publication",
            {
                "name": "bridge-assets-publication",
                "can_admins_bypass": False,
                "protection_rules": [{"type": "branch_policy"}],
                "deployment_branch_policy": {
                    "protected_branches": False,
                    "custom_branch_policies": True,
                },
            },
        )
        self.json_routes.setdefault(
            f"repos/{BRIDGE_REPOSITORY}/environments/bridge-assets-publication/"
            "deployment-branch-policies",
            {
                "total_count": 1,
                "branch_policies": [{"name": "main", "type": "branch"}],
            },
        )
        self.json_routes.setdefault(
            f"repos/{ASSETS_REPOSITORY}/immutable-releases",
            governance
            if governance is not None
            else {"enabled": True, "enforced_by_owner": True},
        )
        self.json_routes.setdefault(
            f"repos/{ASSETS_REPOSITORY}/git/matching-refs/tags?per_page=100",
            [[]],
        )

    def api_json(self, path: str, *, paginate: bool = False, privileged: bool = False) -> Any:
        self.api_paths.append(path)
        if path not in self.json_routes:
            raise ContractError(f"unmapped API path in test gateway: {path}")
        return self.json_routes[path]

    def download_bytes(self, path: str, *, accept: str, privileged: bool = False) -> bytes:
        if path not in self.blob_routes:
            raise ContractError(f"unmapped blob path in test gateway: {path}")
        return self.blob_routes[path]

    def dispatch_identity(self) -> str | None:
        return self.identity

    def release_attestation(self, *, repository: str, release_tag: str) -> Any:
        key = (repository, release_tag)
        if key not in self.release_attestations:
            raise ContractError(f"unmapped release attestation in test gateway: {key}")
        return self.release_attestations[key]

    def dispatch_workflow(self, *, workflow_file: str, ref: str, inputs: Any) -> None:
        self.dispatches.append(
            {"workflow_file": workflow_file, "ref": ref, "inputs": dict(inputs)}
        )

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)

    def utc_now(self) -> str:
        return self.now


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
                gateway = sro.GhGateway(
                    read_token="read-token", dispatch_token="dispatch-token"
                )
                gateway.dispatch_workflow(
                    workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
                    ref="main",
                    inputs={"zeta": "last", "alpha": "first"},
                )
            observed = json.loads(log.read_text(encoding="utf-8"))
        self.assertEqual(
            observed["argv"],
            [
                "workflow",
                "run",
                sro.CANDIDATE_WORKFLOW_FILE,
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
                gateway = sro.GhGateway(
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


class ProvenanceTest(unittest.TestCase):
    def test_scan_native_binds_exact_manifest_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "assets.json"
            payload = json.dumps(native_manifest()).encode("utf-8")
            path.write_bytes(payload)
            provenance = sro.scan_native_provenance(
                manifest_path=path,
                native_release_tag="v0.2.0",
                bridge_source_sha=BRIDGE_SHA,
                channel="stable",
                native_release_published_at=NATIVE_PUBLISHED_AT,
            )
        self.assertEqual(provenance.upstream_tag, "v0.2.0")
        self.assertEqual(provenance.native_release_tag, "v0.2.0")
        self.assertEqual(
            provenance.native_manifest_sha256, hashlib.sha256(payload).hexdigest()
        )

    def test_malformed_native_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "assets.json"
            path.write_text('{"native_release_tag": "v0.2.0", "native_release_tag": "v0.2.0"}')
            with self.assertRaises(ContractError):
                sro.scan_native_provenance(
                    manifest_path=path,
                    native_release_tag="v0.2.0",
                    bridge_source_sha=BRIDGE_SHA,
                    channel="stable",
                    native_release_published_at=NATIVE_PUBLISHED_AT,
                )

    def test_channel_inconsistent_provenance_is_rejected(self) -> None:
        with self.assertRaises(ContractError):
            make_provenance(upstream_tag="b9165")
        with self.assertRaises(ContractError):
            make_provenance(native_release_tag="b9165")

    def test_native_release_timestamp_is_canonical_and_bounded(self) -> None:
        with self.assertRaises(ContractError):
            make_provenance(native_release_published_at="yesterday")


class DevelopmentScanTest(unittest.TestCase):
    """Manual development scans stay supported, and stay scan-only."""

    def _scan(self, channel: str) -> sro.NativeProvenance:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "assets.json"
            path.write_bytes(
                json.dumps(
                    native_manifest(
                        native_release_tag="b9165",
                        llama_cpp_tag="b9165",
                    )
                ).encode("utf-8")
            )
            return sro.scan_native_provenance(
                manifest_path=path,
                native_release_tag="b9165",
                bridge_source_sha=BRIDGE_SHA,
                channel=channel,
                native_release_published_at=NATIVE_PUBLISHED_AT,
            )

    def test_development_scan_still_prepares_exact_provenance(self) -> None:
        provenance = self._scan("development")
        self.assertEqual(provenance.upstream_tag, "b9165")
        self.assertEqual(provenance.native_release_tag, "b9165")
        self.assertIs(provenance.channel, sro.Channel.DEVELOPMENT)

    def test_stable_scan_of_a_development_release_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            self._scan("stable")

    def test_development_scan_of_a_stable_release_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "assets.json"
            path.write_bytes(json.dumps(native_manifest()).encode("utf-8"))
            with self.assertRaises(ContractError):
                sro.scan_native_provenance(
                    manifest_path=path,
                    native_release_tag="v0.2.0",
                    bridge_source_sha=BRIDGE_SHA,
                    channel="development",
                    native_release_published_at=NATIVE_PUBLISHED_AT,
                )

    def test_orchestration_refuses_a_development_provenance(self) -> None:
        development = sro.NativeProvenance(
            bridge_source_sha=BRIDGE_SHA,
            upstream_tag="b9165",
            upstream_commit=UPSTREAM_COMMIT,
            native_repo=NATIVE_REPOSITORY,
            native_release_tag="b9165",
            native_commit=NATIVE_COMMIT,
            native_manifest_sha256=NATIVE_MANIFEST_SHA,
            native_release_published_at=NATIVE_PUBLISHED_AT,
        )
        with self.assertRaises(ContractError):
            sro.require_stable_provenance(development)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "provenance.json"
            path.write_text(
                json.dumps(
                    {
                        "bridge_source_sha": BRIDGE_SHA,
                        "upstream_tag": "b9165",
                        "upstream_commit": UPSTREAM_COMMIT,
                        "native_repo": NATIVE_REPOSITORY,
                        "native_release_tag": "b9165",
                        "native_commit": NATIVE_COMMIT,
                        "native_manifest_sha256": NATIVE_MANIFEST_SHA,
                        "native_release_published_at": NATIVE_PUBLISHED_AT,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ContractError):
                sro._load_provenance(path)


class StableNativeBacklogTest(unittest.TestCase):
    def release(
        self,
        tag: str,
        published_at: str,
        *,
        draft: bool = False,
        prerelease: bool | None = None,
    ) -> dict[str, Any]:
        if prerelease is None:
            prerelease = "-" in tag
        return {
            "tag_name": tag,
            "draft": draft,
            "prerelease": prerelease,
            "published_at": published_at,
        }

    def test_selects_every_post_baseline_stable_release_in_publication_order(self) -> None:
        releases = [
            self.release("v0.2.1", "2026-08-28T03:00:00Z"),
            self.release("b10599", "2026-08-28T02:00:00Z", prerelease=False),
            self.release("v0.2.0-1", sro.STABLE_AUTOMATION_BASELINE_PUBLISHED_AT),
            self.release("v0.2.0-2", "2026-08-27T03:00:00Z"),
            self.release("v0.3.0", "2026-08-29T03:00:00Z", draft=True),
        ]
        self.assertEqual(
            sro.select_stable_native_backlog(releases),
            ["v0.2.0-2", "v0.2.1"],
        )

    def test_post_baseline_stable_rollback_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            sro.select_stable_native_backlog(
                [self.release("v0.1.99", "2026-08-28T03:00:00Z")]
            )

    def test_inconsistent_stable_prerelease_state_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            sro.select_stable_native_backlog(
                [
                    self.release(
                        "v0.2.0-2",
                        "2026-08-28T03:00:00Z",
                        prerelease=False,
                    )
                ]
            )

    def test_duplicate_stable_tag_fails_closed(self) -> None:
        release = self.release("v0.2.1", "2026-08-28T03:00:00Z")
        with self.assertRaises(ContractError):
            sro.select_stable_native_backlog([release, dict(release)])


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


class CorrelationTest(unittest.TestCase):
    def test_deterministic_correlation_id_ignores_advancing_main(self) -> None:
        first = sro.compute_correlation_id(make_provenance())
        second = sro.compute_correlation_id(
            make_provenance(bridge_source_sha=ADVANCED_BRIDGE_SHA)
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first, f"auto-stable-v0.2.0-{NATIVE_MANIFEST_SHA[:16]}"
        )

    def test_correlation_id_changes_with_native_manifest(self) -> None:
        other = sro.compute_correlation_id(
            make_provenance(native_manifest_sha256="b" * 64)
        )
        self.assertNotEqual(sro.compute_correlation_id(make_provenance()), other)

    def test_candidate_run_name_round_trips_persisted_binding(self) -> None:
        correlation_id = sro.compute_correlation_id(make_provenance())
        binding = sro.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )
        name = sro.candidate_run_name(correlation_id, binding)
        self.assertEqual(
            name,
            f"bridge-candidate {correlation_id} source:{BRIDGE_SHA} "
            "tag:v0.1.40 rebuild:0",
        )
        self.assertEqual(sro.parse_candidate_run_name(name, correlation_id), binding)

    def test_foreign_candidate_run_name_is_not_adopted(self) -> None:
        correlation_id = sro.compute_correlation_id(make_provenance())
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
                    sro.parse_candidate_run_name(foreign_name, correlation_id)
                )

    def test_malformed_candidate_run_name_claiming_correlation_fails_closed(
        self,
    ) -> None:
        correlation_id = sro.compute_correlation_id(make_provenance())
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
                    sro.parse_candidate_run_name(malformed_name, correlation_id)

    def test_generated_run_name_enforces_a_conservative_length_bound(self) -> None:
        with self.assertRaises(ContractError):
            sro._require_run_name("x" * (sro.MAX_RUN_NAME_CHARACTERS + 1))


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
        correlation_id = sro.compute_correlation_id(make_provenance())
        binding = sro.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )
        rendered = self.render(
            sro.CANDIDATE_WORKFLOW_FILE,
            {
                "orchestrator_correlation_id": correlation_id,
                "bridge_source_sha": BRIDGE_SHA,
                "release_tag": "v0.1.40",
                "release_rebuild": "0",
            },
        )
        self.assertEqual(rendered, sro.candidate_run_name(correlation_id, binding))
        self.assertEqual(sro.parse_candidate_run_name(rendered, correlation_id), binding)

    def test_qualification_workflow_renders_the_parsed_run_name(self) -> None:
        correlation_id = sro.compute_correlation_id(make_provenance())
        rendered = self.render(
            sro.QUALIFICATION_WORKFLOW_FILE,
            {
                "orchestrator_correlation_id": correlation_id,
                "candidate_run_id": CANDIDATE_RUN_ID,
            },
        )
        self.assertEqual(
            rendered, sro.qualification_run_name(correlation_id, CANDIDATE_RUN_ID)
        )

    def test_publish_workflow_renders_the_parsed_run_name(self) -> None:
        correlation_id = sro.compute_correlation_id(make_provenance())
        binding = sro.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )
        rendered = self.render(
            sro.PUBLISH_WORKFLOW_FILE,
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
            sro.publish_run_name(
                correlation_id, CANDIDATE_RUN_ID, QUALIFICATION_RUN_ID, binding
            ),
        )
        self.assertEqual(
            sro.parse_publish_run_name(rendered, correlation_id),
            (CANDIDATE_RUN_ID, QUALIFICATION_RUN_ID, binding),
        )


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
        self.correlation_id = sro.compute_correlation_id(self.provenance)
        self.binding = sro.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )

    def test_candidate_dispatch_inputs_are_exactly_the_declared_inputs(self) -> None:
        declared = declared_workflow_inputs(
            self.WORKFLOWS / sro.CANDIDATE_WORKFLOW_FILE
        )
        self.assertEqual(set(declared), set(sro.CANDIDATE_DISPATCH_INPUTS))
        inputs = sro._dispatch_inputs_for_candidate(
            self.provenance, self.correlation_id, self.binding
        )
        # The live governance proof supplies the immutability assertion, so the
        # planner must not pre-declare it.
        self.assertEqual(
            set(inputs),
            set(declared) - {"assets_immutable_releases_enabled"},
        )

    def test_publish_dispatch_inputs_are_exactly_the_declared_inputs(self) -> None:
        declared = declared_workflow_inputs(self.WORKFLOWS / sro.PUBLISH_WORKFLOW_FILE)
        self.assertEqual(set(declared), set(sro.PUBLISH_DISPATCH_INPUTS))
        inputs = sro._dispatch_inputs_for_publish(
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
            self.WORKFLOWS / sro.QUALIFICATION_WORKFLOW_FILE
        )
        self.assertEqual(set(declared), set(sro.QUALIFICATION_DISPATCH_INPUTS))
        inputs = sro._dispatch_inputs_for_qualification(
            self.correlation_id, CANDIDATE_RUN_ID
        )
        self.assertEqual(set(inputs), set(declared))

    def test_missing_dispatch_input_fails_closed(self) -> None:
        inputs = sro._dispatch_inputs_for_publish(
            self.provenance,
            self.correlation_id,
            self.binding,
            CANDIDATE_RUN_ID,
            QUALIFICATION_RUN_ID,
        )
        with self.assertRaises(ContractError):
            sro.require_exact_dispatch_inputs(sro.PUBLISH_WORKFLOW_FILE, inputs)

    def test_unknown_dispatch_input_fails_closed(self) -> None:
        inputs = sro._dispatch_inputs_for_candidate(
            self.provenance, self.correlation_id, self.binding
        )
        inputs["assets_immutable_releases_enabled"] = "true"
        inputs["unexpected"] = "value"
        with self.assertRaises(ContractError):
            sro.require_exact_dispatch_inputs(sro.CANDIDATE_WORKFLOW_FILE, inputs)

    def test_undispatchable_workflow_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            sro.require_exact_dispatch_inputs(
                "ci.yml", {"candidate_run_id": CANDIDATE_RUN_ID}
            )

    def test_non_string_dispatch_input_fails_closed(self) -> None:
        inputs = sro._dispatch_inputs_for_candidate(
            self.provenance, self.correlation_id, self.binding
        )
        inputs["assets_immutable_releases_enabled"] = "true"
        inputs["release_rebuild"] = 0  # type: ignore[assignment]
        with self.assertRaises(ContractError):
            sro.require_exact_dispatch_inputs(sro.CANDIDATE_WORKFLOW_FILE, inputs)


class WorkflowRunsResponseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = sro.CANDIDATE_WORKFLOW_PATH
        correlation_id = sro.compute_correlation_id(make_provenance())
        self.run_name = sro.candidate_run_name(
            correlation_id,
            sro.PipelineBinding(
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
        runs = sro.parse_workflow_runs(
            runs_response([self.run]),
            workflow_path=self.path,
            default_branch=DEFAULT_BRANCH,
        )
        self.assertEqual([record.run_id for record in runs], [CANDIDATE_RUN_ID])

    def test_bare_list_response_is_rejected(self) -> None:
        with self.assertRaises(ContractError):
            sro.parse_workflow_runs(
                [self.run], workflow_path=self.path, default_branch=DEFAULT_BRANCH
            )

    def test_truncated_page_set_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            sro.parse_workflow_runs(
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
            sro.parse_workflow_runs(
                runs_response([foreign]),
                workflow_path=self.path,
                default_branch=DEFAULT_BRANCH,
            )

    def test_unsupported_workflow_path_fails_closed_even_when_record_matches(self) -> None:
        unsupported_path = ".github/workflows/ci.yml"
        unsupported = dict(self.run, path=unsupported_path)
        with self.assertRaises(ContractError):
            sro.parse_workflow_runs(
                runs_response([unsupported]),
                workflow_path=unsupported_path,
                default_branch=DEFAULT_BRANCH,
            )

    def test_non_owner_actor_or_triggering_actor_fails_closed(self) -> None:
        for field in ("actor", "triggering_actor"):
            malformed = dict(self.run)
            malformed[field] = {"login": "someone-else"}
            with self.subTest(field=field), self.assertRaises(ContractError):
                sro.parse_workflow_runs(
                    runs_response([malformed]),
                    workflow_path=self.path,
                    default_branch=DEFAULT_BRANCH,
                )

    def test_duplicate_run_ids_fail_closed(self) -> None:
        with self.assertRaises(ContractError):
            sro.parse_workflow_runs(
                {"total_count": 2, "workflow_runs": [self.run, dict(self.run)]},
                workflow_path=self.path,
                default_branch=DEFAULT_BRANCH,
            )

    def test_server_query_bounds_only_relevant_owner_branch_history(self) -> None:
        path = sro._workflow_runs_path(
            workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
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
        first_path = sro._workflow_runs_path(
            workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        second_path = sro._workflow_runs_path(
            workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
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
        records = sro._fetch_runs(
            gateway,
            workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
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
        first_path = sro._workflow_runs_path(
            workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        second_path = sro._workflow_runs_path(
            workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
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
            sro._fetch_runs(
                gateway,
                workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
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
        initial_path = sro._workflow_runs_path(
            workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        full_range_path = sro._workflow_runs_path(
            workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
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
                path = sro._workflow_runs_path(
                    workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
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
        records = sro._fetch_runs(
            gateway,
            workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
            workflow_path=self.path,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        self.assertEqual(len(records), 1000)


class RunSelectionTest(unittest.TestCase):
    def _runs(self, *names_and_states: tuple[str, str, str, str | None]) -> list[Any]:
        payloads = [
            run_payload(
                run_id=run_id, path=sro.CANDIDATE_WORKFLOW_PATH, run_name=name,
                status=status, conclusion=conclusion,
            )
            for run_id, name, status, conclusion in names_and_states
        ]
        return sro.parse_workflow_runs(
            runs_response(payloads),
            workflow_path=sro.CANDIDATE_WORKFLOW_PATH,
            default_branch=DEFAULT_BRANCH,
        )

    def test_single_success_is_selected(self) -> None:
        runs = self._runs(("501", "target", "completed", "success"))
        selection = sro.select_pipeline_runs(runs, label="candidate", matcher=lambda name: name == "target")
        self.assertEqual(selection.succeeded_run_id, "501")
        self.assertEqual(selection.in_flight_run_id, None)

    def test_duplicate_success_fails_closed(self) -> None:
        runs = self._runs(
            ("501", "target", "completed", "success"),
            ("502", "target", "completed", "success"),
        )
        with self.assertRaises(ContractError):
            sro.select_pipeline_runs(runs, label="candidate", matcher=lambda name: name == "target")

    def test_duplicate_in_flight_fails_closed(self) -> None:
        runs = self._runs(
            ("501", "target", "in_progress", None),
            ("502", "target", "queued", None),
        )
        with self.assertRaises(ContractError):
            sro.select_pipeline_runs(runs, label="candidate", matcher=lambda name: name == "target")

    def test_success_racing_a_new_dispatch_fails_closed(self) -> None:
        runs = self._runs(
            ("501", "target", "completed", "success"),
            ("502", "target", "in_progress", None),
        )
        with self.assertRaises(ContractError):
            sro.select_pipeline_runs(runs, label="candidate", matcher=lambda name: name == "target")

    def test_failed_runs_are_retryable_not_selected(self) -> None:
        runs = self._runs(("501", "target", "completed", "failure"))
        selection = sro.select_pipeline_runs(runs, label="candidate", matcher=lambda name: name == "target")
        self.assertIsNone(selection.succeeded_run_id)
        self.assertIsNone(selection.in_flight_run_id)
        self.assertEqual([record.run_id for record in selection.unsuccessful], ["501"])

    def test_second_attempt_fails_closed(self) -> None:
        payload = run_payload(
            run_id="501",
            path=sro.CANDIDATE_WORKFLOW_PATH,
            run_name="target",
            run_attempt=2,
        )
        runs = sro.parse_workflow_runs(
            runs_response([payload]),
            workflow_path=sro.CANDIDATE_WORKFLOW_PATH,
            default_branch=DEFAULT_BRANCH,
        )
        with self.assertRaises(ContractError):
            sro.select_pipeline_runs(runs, label="candidate", matcher=lambda name: name == "target")

    def test_off_main_line_run_fails_closed(self) -> None:
        payload = run_payload(
            run_id="501",
            path=sro.CANDIDATE_WORKFLOW_PATH,
            run_name="target",
            head_branch="attacker",
        )
        runs = sro.parse_workflow_runs(
            runs_response([payload]),
            workflow_path=sro.CANDIDATE_WORKFLOW_PATH,
            default_branch=DEFAULT_BRANCH,
        )
        with self.assertRaises(ContractError):
            sro.select_pipeline_runs(runs, label="candidate", matcher=lambda name: name == "target")

    def test_candidate_matcher_and_selection_fail_closed_on_correlated_malformed_run(
        self,
    ) -> None:
        correlation_id = sro.compute_correlation_id(make_provenance())
        matcher = sro._candidate_matcher(correlation_id)
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
                    sro.select_pipeline_runs(runs, label="candidate", matcher=matcher)

    def test_candidate_matcher_and_selection_ignore_foreign_candidate_run(
        self,
    ) -> None:
        correlation_id = sro.compute_correlation_id(make_provenance())
        matcher = sro._candidate_matcher(correlation_id)
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
                selection = sro.select_pipeline_runs(
                    runs, label="candidate", matcher=matcher
                )
                self.assertEqual(len(selection.matched), 0)
                self.assertIsNone(selection.in_flight_run_id)
                self.assertIsNone(selection.succeeded_run_id)


class ReleaseTargetTest(unittest.TestCase):
    def test_bridge_assets_version_independently_of_upstream(self) -> None:
        target = sro.select_next_release_target(
            ["v0.1.38", "v0.1.39"], upstream_tag="v0.2.0"
        )
        self.assertEqual(target.release_tag, "v0.1.40")
        self.assertEqual(target.release_rebuild, 0)

    def test_upstream_tag_is_never_used_as_the_output_tag(self) -> None:
        target = sro.select_next_release_target(
            ["v0.1.39"], upstream_tag="v0.2.0"
        )
        self.assertNotEqual(target.release_tag, "v0.2.0")

    def test_existing_tag_collision_selects_a_free_rebuild(self) -> None:
        target = sro.select_next_release_target(
            ["v0.1.39", "v0.1.40", "v0.1.40-1"], upstream_tag="v0.2.0"
        )
        self.assertEqual(target.release_tag, "v0.1.41")
        self.assertEqual(target.release_rebuild, 0)

    def test_draft_or_unparsable_tag_collision_is_avoided(self) -> None:
        target = sro.select_next_release_target(
            ["v0.1.39", "v0.1.40", "nightly-scratch"], upstream_tag="v0.2.0"
        )
        self.assertEqual(target.release_tag, "v0.1.41")

    def test_seed_release_when_no_stable_assets_exist(self) -> None:
        target = sro.select_next_release_target([], upstream_tag="v0.2.0")
        self.assertEqual(target.release_tag, sro.INITIAL_STABLE_RELEASE_TAG)
        self.assertEqual(target.release_rebuild, 0)

    def test_rebuild_suffix_when_next_version_tag_is_taken(self) -> None:
        target = sro.select_next_release_target(
            ["v0.1.39", "v0.1.40"], upstream_tag="v0.2.0", taken={"v0.1.41"}
        )
        self.assertEqual(target.release_tag, "v0.1.41-1")
        self.assertEqual(target.release_rebuild, 1)

    def test_run_history_includes_claims_since_the_last_asset_publication(self) -> None:
        prior = asset_release_stub()
        prior["published_at"] = "2026-08-10T00:00:00Z"
        self.assertEqual(
            sro.workflow_history_since([prior], make_provenance()),
            "2026-08-10T00:00:00Z",
        )


class PublishedReleaseVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sro-published-"))
        self.provenance = make_provenance()
        self.correlation_id = sro.compute_correlation_id(self.provenance)
        self.candidate = self.tmp / "candidate"
        write_bridge_candidate(
            self.candidate,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
        )
        self.members = directory_members(self.candidate)
        self.fingerprint = rq.load_candidate(self.candidate)[1]
        self.body = (
            f"Candidate fingerprint: `{self.fingerprint}`\n"
            f"Orchestrator correlation: `{self.correlation_id}`\n"
        )
        self.release = release_payload(
            tag="v0.1.40", body=self.body, members=self.members
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _gateway_for_release(
        self,
        release: dict[str, Any],
        *,
        release_by_id: dict[str, Any] | None = None,
        tag_commit: str = ASSETS_TAG_COMMIT,
        attestation: Any = None,
    ) -> FakeGateway:
        blobs = {
            f"repos/{ASSETS_REPOSITORY}/releases/assets/{asset['id']}": self.members[
                asset["name"]
            ]
            for asset in release["assets"]
            if asset["name"] in self.members
        }
        tag = release["tag_name"]
        release_id = release["id"]
        digests = {
            name: hashlib.sha256(data).hexdigest()
            for name, data in self.members.items()
        }
        routes = {
            f"repos/{ASSETS_REPOSITORY}/git/ref/tags/{tag}": {
                "ref": f"refs/tags/{tag}",
                "object": {"type": "commit", "sha": tag_commit},
            },
            f"repos/{ASSETS_REPOSITORY}/releases/tags/{tag}": release,
            f"repos/{ASSETS_REPOSITORY}/releases/{release_id}": (
                release_by_id or release
            ),
        }
        attestations = {
            (ASSETS_REPOSITORY, tag): attestation
            or release_attestation(
                release_tag=tag,
                assets_repo=ASSETS_REPOSITORY,
                tag_commit=tag_commit,
                release_id=release_id,
                assets=digests,
            )
        }
        return FakeGateway(
            json_routes=routes,
            blob_routes=blobs,
            release_attestations=attestations,
        )

    def _verify(self, release: dict[str, Any], **kwargs: Any) -> Any:
        gateway = self._gateway_for_release(release, **kwargs)
        return sro.verify_published_release(
            gateway,
            release=release,
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            workspace=self.tmp / "workspace",
        )

    def test_exact_immutable_publication_is_accepted(self) -> None:
        verified = self._verify(self.release)
        self.assertEqual(verified.release_target.release_tag, "v0.1.40")
        self.assertEqual(verified.release_target.release_rebuild, 0)
        self.assertEqual(verified.binding.bridge_source_sha, BRIDGE_SHA)

    def test_only_exact_v0140_identity_selects_legacy_manifest_contract(self) -> None:
        exact = make_legacy_v0140_provenance()
        self.assertEqual(
            sro._published_manifest_compatibility(
                tag="v0.1.40",
                provenance=exact,
            ),
            (
                LEGACY_MANUAL_QUALIFICATION_GATES,
                LEGACY_MANUAL_UNPROVEN_CAPABILITIES,
            ),
        )
        near_misses = {
            "release tag": ("v0.1.40-1", exact),
            "native tag": (
                "v0.1.40",
                make_legacy_v0140_provenance(native_release_tag="v0.3.0-1"),
            ),
            "native commit": (
                "v0.1.40",
                make_legacy_v0140_provenance(native_commit="1" * 40),
            ),
            "upstream tag": (
                "v0.1.40",
                make_legacy_v0140_provenance(upstream_tag="v0.3.1"),
            ),
            "upstream commit": (
                "v0.1.40",
                make_legacy_v0140_provenance(upstream_commit="2" * 40),
            ),
            "bridge commit": (
                "v0.1.40",
                make_legacy_v0140_provenance(bridge_source_sha="3" * 40),
            ),
            "native manifest": (
                "v0.1.40",
                make_legacy_v0140_provenance(native_manifest_sha256="0" * 64),
            ),
        }
        for label, (tag, provenance) in near_misses.items():
            with self.subTest(label=label):
                self.assertIsNone(
                    sro._published_manifest_compatibility(
                        tag=tag,
                        provenance=provenance,
                    )
                )

    def test_native_manifest_marker_prevents_duplicate_when_correlation_is_damaged(
        self,
    ) -> None:
        malformed = dict(
            self.release,
            body=(
                f"Native: `{NATIVE_REPOSITORY}@v0.2.0`\n"
                f"Native manifest SHA-256: `{NATIVE_MANIFEST_SHA}`\n"
            ),
        )
        self.assertIs(
            sro.find_correlated_release(
                [malformed], self.correlation_id, self.provenance
            ),
            malformed,
        )
        with self.assertRaises(ContractError):
            self._verify(malformed)

    def test_manifest_digest_without_exact_native_tag_cannot_hijack_state(self) -> None:
        foreign = dict(
            self.release,
            body=(
                f"Native: `{NATIVE_REPOSITORY}@v9.9.9`\n"
                f"Native manifest SHA-256: `{NATIVE_MANIFEST_SHA}`\n"
            ),
        )
        self.assertIsNone(
            sro.find_correlated_release(
                [foreign], self.correlation_id, self.provenance
            )
        )

    def test_mutable_release_fails_closed(self) -> None:
        release = dict(self.release, immutable=False)
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_missing_immutability_field_fails_closed(self) -> None:
        release = dict(self.release)
        del release["immutable"]
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_missing_published_at_fails_closed(self) -> None:
        release = dict(self.release, published_at=None)
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_draft_release_fails_closed(self) -> None:
        release = dict(self.release, draft=True)
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_incomplete_asset_inventory_fails_closed(self) -> None:
        release = dict(self.release)
        release["assets"] = [
            asset for asset in release["assets"] if asset["name"] != "sha256sums.txt"
        ]
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_unexpected_extra_asset_fails_closed(self) -> None:
        release = dict(self.release)
        release["assets"] = release["assets"] + [
            {
                "id": 999,
                "name": "extra.bin",
                "state": "uploaded",
                "size": 1,
                "digest": f"sha256:{'0' * 64}",
            }
        ]
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_digest_mismatch_fails_closed(self) -> None:
        release = release_payload(
            tag="v0.1.40",
            body=self.body,
            members=self.members,
            asset_overrides={"manifest.json": {"digest": f"sha256:{'0' * 64}"}},
        )
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_manifest_bytes_that_do_not_bind_provenance_fail_closed(self) -> None:
        with self.assertRaises(ContractError):
            sro.verify_published_release(
                self._gateway_for_release(self.release),
                release=self.release,
                provenance=make_provenance(native_commit="d" * 40),
                correlation_id=self.correlation_id,
                workspace=self.tmp / "workspace",
            )

    def test_release_tag_commit_is_resolved_independently(self) -> None:
        with self.assertRaises(ContractError):
            self._verify(self.release, tag_commit="d" * 40)

    def test_release_readback_by_id_must_match(self) -> None:
        by_id = dict(self.release, body="unrelated")
        with self.assertRaises(ContractError):
            self._verify(self.release, release_by_id=by_id)

    def test_signed_release_attestation_must_bind_every_asset(self) -> None:
        invalid = release_attestation(
            release_tag="v0.1.40",
            assets_repo=ASSETS_REPOSITORY,
            tag_commit=ASSETS_TAG_COMMIT,
            release_id=self.release["id"],
            assets={"manifest.json": "0" * 64},
        )
        with self.assertRaises(ContractError):
            self._verify(self.release, attestation=invalid)

    def test_release_body_without_the_candidate_fingerprint_fails_closed(self) -> None:
        # The publication contract binds the release body to the exact candidate
        # digest. A body that only names the correlation proves nothing about
        # which bytes were published.
        release = release_payload(
            tag="v0.1.40",
            body=f"Orchestrator correlation: `{self.correlation_id}`\n",
            members=self.members,
        )
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_release_body_with_a_foreign_candidate_fingerprint_fails_closed(self) -> None:
        release = release_payload(
            tag="v0.1.40",
            body=(
                f"Candidate fingerprint: `{'0' * 64}`\n"
                f"Orchestrator correlation: `{self.correlation_id}`\n"
            ),
            members=self.members,
        )
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_release_tag_that_contradicts_the_published_manifest_fails_closed(
        self,
    ) -> None:
        release = release_payload(
            tag="v0.1.41", body=self.body, members=self.members
        )
        with self.assertRaises(ContractError):
            self._verify(release)

    def test_publication_from_an_earlier_bridge_source_stays_a_noop(self) -> None:
        # Main advances daily; the published release stays bound to the exact
        # candidate source it was built from, not to today's HEAD.
        verified = self._verify(self.release)
        self.assertEqual(verified.binding.bridge_source_sha, BRIDGE_SHA)
        advanced = sro.verify_published_release(
            self._gateway_for_release(self.release),
            release=self.release,
            provenance=make_provenance(bridge_source_sha=ADVANCED_BRIDGE_SHA),
            correlation_id=self.correlation_id,
            workspace=self.tmp / "workspace",
        )
        self.assertEqual(advanced.binding.bridge_source_sha, BRIDGE_SHA)


class PlanTest(unittest.TestCase):
    """Exhaustive pure-state-machine coverage over already-proven evidence."""

    def setUp(self) -> None:
        self.provenance = make_provenance()
        self.correlation_id = sro.compute_correlation_id(self.provenance)
        self.binding = sro.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )

    def test_new_provenance_plans_exactly_one_candidate_dispatch(self) -> None:
        plan = sro.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=sro.PipelineObservation(fresh_binding=self.binding),
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.DISPATCH_CANDIDATE)
        self.assertEqual(plan.dispatch_workflow, sro.CANDIDATE_WORKFLOW_FILE)
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
            set(sro.CANDIDATE_DISPATCH_INPUTS) - {"assets_immutable_releases_enabled"},
        )

    def test_published_provenance_is_an_exact_noop(self) -> None:
        plan = sro.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=sro.PipelineObservation(
                published=sro.PublishedRelease(
                    release_id=4242,
                    release_target=sro.ReleaseTarget("v0.1.40", 0),
                    binding=self.binding,
                    published_at="2026-08-20T03:24:11Z",
                )
            ),
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.NOOP)
        self.assertIsNone(plan.dispatch_workflow)
        self.assertIn("already published", plan.reason.lower())

    def test_in_flight_candidate_blocks_duplicate_dispatch(self) -> None:
        plan = sro.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=sro.PipelineObservation(
                fresh_binding=self.binding,
                candidate_in_flight_run_id="501",
                binding=self.binding,
            ),
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.IN_FLIGHT)
        self.assertEqual(plan.in_flight_workflow, sro.CANDIDATE_WORKFLOW_FILE)
        self.assertEqual(plan.in_flight_run_id, "501")
        self.assertIsNone(plan.dispatch_workflow)

    def test_candidate_ready_plans_exactly_one_hosted_qualification(self) -> None:
        plan = sro.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=sro.PipelineObservation(
                binding=self.binding, candidate_run_id=CANDIDATE_RUN_ID
            ),
        )
        self.assertEqual(
            plan.action, sro.OrchestrationAction.DISPATCH_QUALIFICATION
        )
        self.assertEqual(plan.candidate_run_id, CANDIDATE_RUN_ID)
        self.assertEqual(plan.dispatch_workflow, sro.QUALIFICATION_WORKFLOW_FILE)
        self.assertEqual(
            plan.dispatch_run_name,
            sro.qualification_run_name(self.correlation_id, CANDIDATE_RUN_ID),
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
        plan = sro.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=sro.PipelineObservation(
                binding=self.binding,
                candidate_run_id=CANDIDATE_RUN_ID,
                qualification_in_flight_run_id=QUALIFICATION_RUN_ID,
            ),
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.IN_FLIGHT)
        self.assertEqual(plan.in_flight_workflow, sro.QUALIFICATION_WORKFLOW_FILE)

    def test_candidate_and_qualification_ready_plan_publish(self) -> None:
        plan = sro.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=sro.PipelineObservation(
                binding=self.binding,
                candidate_run_id=CANDIDATE_RUN_ID,
                qualification_run_id=QUALIFICATION_RUN_ID,
            ),
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.DISPATCH_PUBLISH)
        self.assertEqual(plan.dispatch_workflow, sro.PUBLISH_WORKFLOW_FILE)
        inputs = plan.dispatch_inputs or {}
        self.assertEqual(inputs["candidate_run_id"], CANDIDATE_RUN_ID)
        self.assertEqual(inputs["qualification_run_id"], QUALIFICATION_RUN_ID)
        self.assertEqual(inputs["release_tag"], "v0.1.40")
        self.assertEqual(inputs["release_rebuild"], "0")
        self.assertNotIn("publish_approved", inputs)
        self.assertEqual(inputs["assets_repo"], ASSETS_REPOSITORY)
        self.assertEqual(inputs["bridge_source_sha"], BRIDGE_SHA)
        self.assertEqual(
            set(inputs), set(sro.PUBLISH_DISPATCH_INPUTS) - {"publish_approved"}
        )

    def test_in_flight_publish_blocks_duplicate_publish(self) -> None:
        plan = sro.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=sro.PipelineObservation(
                binding=self.binding,
                candidate_run_id=CANDIDATE_RUN_ID,
                qualification_run_id=QUALIFICATION_RUN_ID,
                publish_in_flight_run_id="701",
            ),
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.IN_FLIGHT)
        self.assertEqual(plan.in_flight_workflow, sro.PUBLISH_WORKFLOW_FILE)
        self.assertEqual(plan.in_flight_run_id, "701")

    def test_successful_publish_without_immutable_release_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            sro.plan_pipeline(
                provenance=self.provenance,
                correlation_id=self.correlation_id,
                observation=sro.PipelineObservation(
                    binding=self.binding,
                    candidate_run_id=CANDIDATE_RUN_ID,
                    qualification_run_id=QUALIFICATION_RUN_ID,
                    publish_succeeded_run_id="701",
                ),
            )

    def test_publish_retry_after_failed_publish_reuses_exact_state(self) -> None:
        plan = sro.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=sro.PipelineObservation(
                binding=self.binding,
                candidate_run_id=CANDIDATE_RUN_ID,
                qualification_run_id=QUALIFICATION_RUN_ID,
                publish_retry=True,
            ),
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.DISPATCH_PUBLISH)
        self.assertEqual(plan.candidate_run_id, CANDIDATE_RUN_ID)
        self.assertEqual(plan.qualification_run_id, QUALIFICATION_RUN_ID)
        # A retry must be visible as a retry, not reported as a first attempt.
        self.assertIn("retry", plan.reason)

    def test_planner_never_pre_asserts_immutable_release_governance(self) -> None:
        plan = sro.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=sro.PipelineObservation(fresh_binding=self.binding),
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.DISPATCH_CANDIDATE)
        self.assertNotIn("assets_immutable_releases_enabled", plan.dispatch_inputs)

    def test_planner_never_pre_asserts_publication_approval(self) -> None:
        plan = sro.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=sro.PipelineObservation(
                binding=self.binding,
                candidate_run_id=CANDIDATE_RUN_ID,
                qualification_run_id=QUALIFICATION_RUN_ID,
            ),
        )
        self.assertNotIn("publish_approved", plan.dispatch_inputs)

    def test_missing_binding_for_ready_candidate_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            sro.plan_pipeline(
                provenance=self.provenance,
                correlation_id=self.correlation_id,
                observation=sro.PipelineObservation(
                    candidate_run_id=CANDIDATE_RUN_ID
                ),
            )

    def test_summary_reports_the_exact_action(self) -> None:
        plan = sro.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=sro.PipelineObservation(fresh_binding=self.binding),
        )
        summary = sro.render_step_summary(plan)
        self.assertIn("### Stable Web bridge release orchestration", summary)
        self.assertIn("dispatch_candidate", summary)
        self.assertIn(sro.CANDIDATE_WORKFLOW_FILE, summary)
        self.assertIn(self.correlation_id, summary)


class GovernanceAndDispatchIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.provenance = make_provenance()
        self.correlation_id = sro.compute_correlation_id(self.provenance)

    def _gateway(self, **kwargs: Any) -> FakeGateway:
        routes = {
            f"repos/{ASSETS_REPOSITORY}/releases?per_page=100": [[]],
            sro._workflow_runs_path(
                workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
                default_branch=DEFAULT_BRANCH,
                created_since=NATIVE_PUBLISHED_AT,
            ): runs_response([]),
        }
        routes.update(kwargs.pop("json_routes", {}))
        return FakeGateway(json_routes=routes, **kwargs)

    def test_disabled_immutable_release_governance_fails_closed(self) -> None:
        gateway = self._gateway(governance={"enabled": False, "enforced_by_owner": False})
        with self.assertRaises(ContractError):
            sro.require_immutable_release_governance(gateway)

    def test_malformed_governance_response_fails_closed(self) -> None:
        gateway = self._gateway(governance={"enabled": True})
        with self.assertRaises(ContractError):
            sro.require_immutable_release_governance(gateway)

    def test_live_governance_is_proven_not_asserted(self) -> None:
        gateway = self._gateway()
        proven = sro.require_immutable_release_governance(gateway)
        self.assertEqual(proven["repository"], ASSETS_REPOSITORY)
        self.assertIs(proven["enabled"], True)

    def test_absent_dispatch_identity_blocks_without_dispatching(self) -> None:
        gateway = self._gateway(identity=None)
        with tempfile.TemporaryDirectory() as workspace:
            plan = sro.advance_pipeline(
                gateway,
                provenance=self.provenance,
                workspace=Path(workspace),
            )
        self.assertEqual(plan.action, sro.OrchestrationAction.BLOCKED)
        self.assertEqual(gateway.dispatches, [])
        self.assertIn("dispatch identity", plan.reason.lower())

    def test_non_owner_dispatch_identity_blocks_without_dispatching(self) -> None:
        gateway = self._gateway(identity="someone-else")
        with tempfile.TemporaryDirectory() as workspace:
            plan = sro.advance_pipeline(
                gateway,
                provenance=self.provenance,
                workspace=Path(workspace),
            )
        self.assertEqual(plan.action, sro.OrchestrationAction.BLOCKED)
        self.assertEqual(gateway.dispatches, [])

    def test_publish_approval_requires_live_environment_policy(self) -> None:
        binding = sro.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )
        plan = sro.plan_pipeline(
            provenance=self.provenance,
            correlation_id=self.correlation_id,
            observation=sro.PipelineObservation(
                binding=binding,
                candidate_run_id=CANDIDATE_RUN_ID,
                qualification_run_id=QUALIFICATION_RUN_ID,
            ),
        )
        gateway = self._gateway()
        gateway.json_routes[
            f"repos/{BRIDGE_REPOSITORY}/environments/bridge-assets-publication"
        ] = {
            "name": "bridge-assets-publication",
            "can_admins_bypass": True,
            "protection_rules": [{"type": "branch_policy"}],
            "deployment_branch_policy": {
                "protected_branches": False,
                "custom_branch_policies": True,
            },
        }
        with self.assertRaises(ContractError):
            sro._execute_dispatch(
                gateway,
                plan,
                default_branch=DEFAULT_BRANCH,
                workflow_path=sro.PUBLISH_WORKFLOW_PATH,
                dry_run=True,
            )
        self.assertEqual(gateway.dispatches, [])


class AdvancePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sro-advance-"))
        self.provenance = make_provenance()
        self.correlation_id = sro.compute_correlation_id(self.provenance)
        self.binding = sro.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )
        self.candidate_name = sro.candidate_run_name(self.correlation_id, self.binding)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _routes(
        self,
        *,
        releases: list[dict[str, Any]] | None = None,
        candidate_runs: list[dict[str, Any]] | None = None,
        qualification_runs: list[dict[str, Any]] | None = None,
        publish_runs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            f"repos/{ASSETS_REPOSITORY}/releases?per_page=100": [releases or []],
            f"repos/{BRIDGE_REPOSITORY}/compare/{BRIDGE_SHA}...{DEFAULT_BRANCH}": {
                "status": "ahead"
            },
            f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}": {
                "status": "identical"
            },
            sro._workflow_runs_path(
                workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
                default_branch=DEFAULT_BRANCH,
                created_since=NATIVE_PUBLISHED_AT,
            ): runs_response(
                candidate_runs or []
            ),
            sro._workflow_runs_path(
                workflow_file=sro.QUALIFICATION_WORKFLOW_FILE,
                default_branch=DEFAULT_BRANCH,
                created_since=NATIVE_PUBLISHED_AT,
            ): runs_response(
                qualification_runs or []
            ),
            sro._workflow_runs_path(
                workflow_file=sro.PUBLISH_WORKFLOW_FILE,
                default_branch=DEFAULT_BRANCH,
                created_since=NATIVE_PUBLISHED_AT,
            ): runs_response(
                publish_runs or []
            ),
        }

    def _published_candidate_gateway(
        self,
        candidate_dir: Path,
        *,
        provenance: sro.NativeProvenance,
        correlation_id: str,
        expected_qualification_gates: dict[str, str] | None = None,
        expected_unproven_capabilities: dict[str, str] | None = None,
    ) -> FakeGateway:
        members = directory_members(candidate_dir)
        fingerprint = rq.load_candidate(
            candidate_dir,
            expected_qualification_gates=expected_qualification_gates,
            expected_unproven_capabilities=expected_unproven_capabilities,
        )[1]
        body = (
            f"Candidate fingerprint: `{fingerprint}`\n"
            f"Orchestrator correlation: `{correlation_id}`\n"
        )
        release = release_payload(tag="v0.1.40", body=body, members=members)
        routes = self._routes(releases=[release])
        routes.update(
            {
                f"repos/{BRIDGE_REPOSITORY}/compare/"
                f"{provenance.bridge_source_sha}...{DEFAULT_BRANCH}": {
                    "status": "ahead"
                },
                f"repos/{ASSETS_REPOSITORY}/git/ref/tags/v0.1.40": {
                    "ref": "refs/tags/v0.1.40",
                    "object": {"type": "commit", "sha": ASSETS_TAG_COMMIT},
                },
                f"repos/{ASSETS_REPOSITORY}/releases/tags/v0.1.40": release,
                f"repos/{ASSETS_REPOSITORY}/releases/{release['id']}": release,
            }
        )
        blobs = {
            f"repos/{ASSETS_REPOSITORY}/releases/assets/{asset['id']}": members[
                asset["name"]
            ]
            for asset in release["assets"]
        }
        digests = {
            name: hashlib.sha256(data).hexdigest() for name, data in members.items()
        }
        return FakeGateway(
            json_routes=routes,
            blob_routes=blobs,
            release_attestations={
                (ASSETS_REPOSITORY, "v0.1.40"): release_attestation(
                    release_tag="v0.1.40",
                    assets_repo=ASSETS_REPOSITORY,
                    tag_commit=ASSETS_TAG_COMMIT,
                    release_id=release["id"],
                    assets=digests,
                )
            },
        )

    def test_first_run_dispatches_exactly_one_candidate_with_structured_inputs(self) -> None:
        dispatched = run_payload(
            run_id="501",
            path=sro.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
            status="in_progress",
            conclusion=None,
        )
        routes = self._routes(
            releases=[asset_release_stub()]
        )
        gateway = FakeGateway(json_routes=routes)
        expected_binding = sro.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )
        readback_key = sro._workflow_runs_path(
            workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )

        original_dispatch = gateway.dispatch_workflow

        def dispatch(**kwargs: Any) -> None:
            original_dispatch(**kwargs)
            gateway.json_routes[readback_key] = runs_response([dispatched])

        gateway.dispatch_workflow = dispatch  # type: ignore[assignment]

        plan = sro.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.DISPATCH_CANDIDATE)
        self.assertEqual(len(gateway.dispatches), 1)
        dispatch_record = gateway.dispatches[0]
        self.assertEqual(dispatch_record["workflow_file"], sro.CANDIDATE_WORKFLOW_FILE)
        self.assertEqual(dispatch_record["ref"], DEFAULT_BRANCH)
        self.assertEqual(dispatch_record["inputs"]["release_tag"], "v0.1.40")
        self.assertTrue(
            all(isinstance(value, str) for value in dispatch_record["inputs"].values())
        )
        self.assertEqual(plan.dispatched_run_id, "501")
        self.assertEqual(
            sro.candidate_run_name(self.correlation_id, expected_binding),
            self.candidate_name,
        )


    def test_later_qualified_provenance_waits_for_earlier_immutable_publication(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="sro-backlog-order-") as temp:
            root = Path(temp)
            first = make_provenance()
            second = make_provenance(
                upstream_tag="v0.2.1",
                upstream_commit="d" * 40,
                native_release_tag="v0.2.1",
                native_commit="e" * 40,
                native_manifest_sha256="f" * 64,
                native_release_published_at="2026-08-21T12:34:56Z",
            )
            first_correlation = sro.compute_correlation_id(first)
            second_correlation = sro.compute_correlation_id(second)
            first_binding = sro.PipelineBinding(BRIDGE_SHA, "v0.1.40", 0)
            second_binding = sro.PipelineBinding(BRIDGE_SHA, "v0.1.40-1", 1)
            first_run_id = "4101"
            second_run_id = "4201"
            second_qualification_run_id = "4202"
            first_artifact_id = 31
            second_artifact_id = 32
            second_qualification_artifact_id = 33

            first_candidate = root / "first-candidate"
            second_candidate = root / "second-candidate"
            write_bridge_candidate(
                first_candidate,
                release_tag=first_binding.release_tag,
                release_rebuild=first_binding.release_rebuild,
                correlation_id=first_correlation,
                run_id=first_run_id,
            )
            write_bridge_candidate(
                second_candidate,
                release_tag=second_binding.release_tag,
                release_rebuild=second_binding.release_rebuild,
                correlation_id=second_correlation,
                run_id=second_run_id,
                upstream_tag=second.upstream_tag,
                upstream_commit=second.upstream_commit,
                native_release_tag=second.native_release_tag,
                native_manifest_sha256=second.native_manifest_sha256,
                native_commit=second.native_commit,
            )
            second_manifest, second_fingerprint = rq.load_candidate(second_candidate)
            second_qualification = rq.build_attestation(
                manifest=second_manifest,
                candidate_fingerprint=second_fingerprint,
                candidate_run_id=second_run_id,
                candidate_artifact_id=second_artifact_id,
                candidate_run_attempt=1,
                **qualification_identity(
                    qualification_run_id=second_qualification_run_id,
                    qualification_source_sha=HEAD_SHA,
                ),
                harness_digest=rq.harness_source_sha256(Path(__file__).resolve().parent),
                environment=qualification_environment(),
                speech_phase=speech_phase(),
                tts_phase=tts_phase(),
            )

            first_run = run_payload(
                run_id=first_run_id,
                path=sro.CANDIDATE_WORKFLOW_PATH,
                run_name=sro.candidate_run_name(first_correlation, first_binding),
            )
            second_run = run_payload(
                run_id=second_run_id,
                path=sro.CANDIDATE_WORKFLOW_PATH,
                run_name=sro.candidate_run_name(second_correlation, second_binding),
            )
            qualification_run = run_payload(
                run_id=second_qualification_run_id,
                path=sro.QUALIFICATION_WORKFLOW_PATH,
                run_name=sro.qualification_run_name(
                    second_correlation, second_run_id
                ),
            )
            releases = [asset_release_stub()]
            routes: dict[str, Any] = {
                f"repos/{ASSETS_REPOSITORY}/releases?per_page=100": [releases],
                f"repos/{BRIDGE_REPOSITORY}/compare/{BRIDGE_SHA}...{DEFAULT_BRANCH}": {
                    "status": "ahead"
                },
                f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}": {
                    "status": "identical"
                },
                f"repos/{BRIDGE_REPOSITORY}/actions/runs/{first_run_id}": first_run,
                f"repos/{BRIDGE_REPOSITORY}/actions/runs/{second_run_id}": second_run,
                f"repos/{BRIDGE_REPOSITORY}/actions/runs/{second_qualification_run_id}": qualification_run,
                f"repos/{BRIDGE_REPOSITORY}/actions/runs/{first_run_id}/artifacts?per_page=100": artifact_inventory(
                    run_id=first_run_id,
                    name=rq.CANDIDATE_ARTIFACT_NAME,
                    artifact_id=first_artifact_id,
                ),
                f"repos/{BRIDGE_REPOSITORY}/actions/runs/{second_run_id}/artifacts?per_page=100": artifact_inventory(
                    run_id=second_run_id,
                    name=rq.CANDIDATE_ARTIFACT_NAME,
                    artifact_id=second_artifact_id,
                ),
                f"repos/{BRIDGE_REPOSITORY}/actions/runs/{second_qualification_run_id}/artifacts?per_page=100": artifact_inventory(
                    run_id=second_qualification_run_id,
                    name=rq.ATTESTATION_ARTIFACT_NAME,
                    artifact_id=second_qualification_artifact_id,
                ),
            }
            candidate_runs = runs_response([first_run, second_run])
            qualification_runs = runs_response([qualification_run])
            publish_runs = runs_response([])
            for provenance in (first, second):
                since = sro.workflow_history_since(releases, provenance)
                routes[
                    sro._workflow_runs_path(
                        workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
                        default_branch=DEFAULT_BRANCH,
                        created_since=since,
                    )
                ] = candidate_runs
                routes[
                    sro._workflow_runs_path(
                        workflow_file=sro.QUALIFICATION_WORKFLOW_FILE,
                        default_branch=DEFAULT_BRANCH,
                        created_since=since,
                    )
                ] = qualification_runs
                routes[
                    sro._workflow_runs_path(
                        workflow_file=sro.PUBLISH_WORKFLOW_FILE,
                        default_branch=DEFAULT_BRANCH,
                        created_since=since,
                    )
                ] = publish_runs
            gateway = FakeGateway(
                json_routes=routes,
                blob_routes={
                    f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{first_artifact_id}/zip": flat_zip(
                        directory_members(first_candidate)
                    ),
                    f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{second_artifact_id}/zip": flat_zip(
                        directory_members(second_candidate)
                    ),
                    f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{second_qualification_artifact_id}/zip": flat_zip(
                        {
                            "qualification-attestation.json": rq.canonical_json(
                                second_qualification
                            ).encode("utf-8")
                        }
                    ),
                },
            )
            first_qualification_run = run_payload(
                run_id="4201",
                path=sro.QUALIFICATION_WORKFLOW_PATH,
                run_name=sro.qualification_run_name(first_correlation, first_run_id),
                status="in_progress",
                conclusion=None,
            )
            qualification_route_keys = [
                sro._workflow_runs_path(
                    workflow_file=sro.QUALIFICATION_WORKFLOW_FILE,
                    default_branch=DEFAULT_BRANCH,
                    created_since=sro.workflow_history_since(releases, provenance),
                )
                for provenance in (first, second)
            ]
            original_dispatch = gateway.dispatch_workflow

            def dispatch(**kwargs: Any) -> None:
                original_dispatch(**kwargs)
                for key in qualification_route_keys:
                    gateway.json_routes[key] = runs_response(
                        [qualification_run, first_qualification_run]
                    )

            gateway.dispatch_workflow = dispatch  # type: ignore[assignment]

            provenance_list = root / "release-candidates.json"
            provenance_list.write_text(
                json.dumps(
                    [
                        sro._provenance_to_dict(first),
                        sro._provenance_to_dict(second),
                    ]
                ),
                encoding="utf-8",
            )
            output_plan = root / "orchestration-plan.json"
            environment = {
                "GITHUB_EVENT_NAME": "schedule",
                "GITHUB_ACTOR": "github-actions",
                "GITHUB_TRIGGERING_ACTOR": "github-actions",
            }
            with (
                mock.patch.object(sro, "GhGateway", return_value=gateway),
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch("sys.stdout", io.StringIO()),
            ):
                result = sro.main(
                    [
                        "orchestrate-backlog",
                        "--provenance-list-json",
                        str(provenance_list),
                        "--workspace",
                        str(root / "workspace"),
                        "--output-plan-json",
                        str(output_plan),
                    ]
                )
            self.assertEqual(result, 0)
            plan = json.loads(output_plan.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["action"] for item in plan["plans"]],
                [
                    sro.OrchestrationAction.DISPATCH_QUALIFICATION.value,
                    sro.OrchestrationAction.WAITING_FOR_PRIOR_PUBLICATION.value,
                ],
            )
            # The later provenance is fully qualified but must not publish before
            # the earlier one, so exactly one qualification dispatch happens and
            # no publication is dispatched.
            self.assertEqual(
                [record["workflow_file"] for record in gateway.dispatches],
                [sro.QUALIFICATION_WORKFLOW_FILE],
            )

    def test_in_flight_candidate_produces_no_second_dispatch(self) -> None:
        in_flight = run_payload(
            run_id="501",
            path=sro.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
            status="in_progress",
            conclusion=None,
        )
        gateway = FakeGateway(json_routes=self._routes(candidate_runs=[in_flight]))
        plan = sro.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.IN_FLIGHT)
        self.assertEqual(gateway.dispatches, [])

    def test_failed_candidate_is_terminal_without_daily_duplicate(self) -> None:
        failed = run_payload(
            run_id="501",
            path=sro.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
            status="completed",
            conclusion="failure",
        )
        # Main advancing cannot silently turn the same native provenance into a
        # fresh candidate attempt.
        advanced = make_provenance(bridge_source_sha=ADVANCED_BRIDGE_SHA)
        routes = self._routes(
            releases=[asset_release_stub()],
            candidate_runs=[failed],
        )
        gateway = FakeGateway(json_routes=routes)
        plan = sro.advance_pipeline(
            gateway, provenance=advanced, workspace=self.tmp
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.BLOCKED)
        self.assertEqual(gateway.dispatches, [])
        self.assertIn("automatic candidate retries are disabled", plan.reason)

    def test_deliberate_success_after_failed_candidate_recovers_pipeline(self) -> None:
        candidate_dir = self.tmp / "candidate-after-failure"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
            run_id=CANDIDATE_RUN_ID,
        )
        failed = run_payload(
            run_id="501",
            path=sro.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
            status="completed",
            conclusion="failure",
        )
        succeeded = run_payload(
            run_id=CANDIDATE_RUN_ID,
            path=sro.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
        )
        routes = self._routes(candidate_runs=[failed, succeeded])
        routes[f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}"] = succeeded
        routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=CANDIDATE_RUN_ID,
            name=rq.CANDIDATE_ARTIFACT_NAME,
            artifact_id=CANDIDATE_ARTIFACT_ID,
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}"] = {
            "status": "identical"
        }
        gateway = FakeGateway(
            json_routes=routes,
            blob_routes={
                f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{CANDIDATE_ARTIFACT_ID}/zip": flat_zip(
                    directory_members(candidate_dir)
                )
            },
        )
        dispatched = run_payload(
            run_id="4201",
            path=sro.QUALIFICATION_WORKFLOW_PATH,
            run_name=sro.qualification_run_name(self.correlation_id, CANDIDATE_RUN_ID),
            status="in_progress",
            conclusion=None,
        )
        readback_key = sro._workflow_runs_path(
            workflow_file=sro.QUALIFICATION_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        original_dispatch = gateway.dispatch_workflow

        def dispatch(**kwargs: Any) -> None:
            original_dispatch(**kwargs)
            gateway.json_routes[readback_key] = runs_response([dispatched])

        gateway.dispatch_workflow = dispatch  # type: ignore[assignment]

        plan = sro.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(
            plan.action, sro.OrchestrationAction.DISPATCH_QUALIFICATION
        )
        self.assertEqual(plan.candidate_run_id, CANDIDATE_RUN_ID)
        self.assertEqual(plan.dispatched_run_id, "4201")
        self.assertEqual(
            [record["workflow_file"] for record in gateway.dispatches],
            [sro.QUALIFICATION_WORKFLOW_FILE],
        )

    def test_dry_run_plans_without_dispatching(self) -> None:
        routes = self._routes(
            releases=[asset_release_stub()]
        )
        gateway = FakeGateway(json_routes=routes)
        plan = sro.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp, dry_run=True
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.DISPATCH_CANDIDATE)
        self.assertEqual(gateway.dispatches, [])
        self.assertIsNone(plan.dispatched_run_id)
        # A dry run must still report the exact ref and inputs it would use.
        self.assertEqual(plan.dispatch_ref, DEFAULT_BRANCH)
        self.assertEqual(
            plan.dispatch_inputs["assets_immutable_releases_enabled"], "true"
        )

    def test_backlog_dry_run_reserves_an_earlier_planned_output_tag(self) -> None:
        gateway = FakeGateway(
            json_routes=self._routes(releases=[asset_release_stub()])
        )
        plan = sro.advance_pipeline(
            gateway,
            provenance=self.provenance,
            workspace=self.tmp,
            dry_run=True,
            reserved_release_tags={"v0.1.40"},
        )
        self.assertEqual(plan.release_target.release_tag, "v0.1.40-1")

    def test_orphan_assets_tag_ref_is_reserved_before_candidate_dispatch(self) -> None:
        routes = self._routes(releases=[asset_release_stub()])
        routes[
            f"repos/{ASSETS_REPOSITORY}/git/matching-refs/tags?per_page=100"
        ] = [
            [
                {
                    "ref": "refs/tags/v0.1.40",
                    "object": {"type": "commit", "sha": ASSETS_TAG_COMMIT},
                }
            ]
        ]
        gateway = FakeGateway(json_routes=routes)
        plan = sro.advance_pipeline(
            gateway,
            provenance=self.provenance,
            workspace=self.tmp,
            dry_run=True,
        )
        self.assertEqual(plan.release_target.release_tag, "v0.1.40-1")
        self.assertEqual(plan.release_target.release_rebuild, 1)

    def test_candidate_dispatch_inputs_carry_the_live_governance_proof(self) -> None:
        dispatched = run_payload(
            run_id="501",
            path=sro.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
            status="in_progress",
            conclusion=None,
        )
        routes = self._routes(
            releases=[asset_release_stub()]
        )
        gateway = FakeGateway(json_routes=routes)
        readback_key = sro._workflow_runs_path(
            workflow_file=sro.CANDIDATE_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        original_dispatch = gateway.dispatch_workflow

        def dispatch(**kwargs: Any) -> None:
            original_dispatch(**kwargs)
            gateway.json_routes[readback_key] = runs_response([dispatched])

        gateway.dispatch_workflow = dispatch  # type: ignore[assignment]

        plan = sro.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        inputs = gateway.dispatches[0]["inputs"]
        self.assertEqual(set(inputs), set(sro.CANDIDATE_DISPATCH_INPUTS))
        self.assertEqual(inputs["assets_immutable_releases_enabled"], "true")
        self.assertEqual(plan.dispatch_ref, DEFAULT_BRANCH)

    def test_published_release_is_a_noop_with_no_dispatch(self) -> None:
        candidate_dir = self.tmp / "candidate"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
        )
        gateway = self._published_candidate_gateway(
            candidate_dir,
            provenance=self.provenance,
            correlation_id=self.correlation_id,
        )
        plan = sro.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.NOOP)
        self.assertEqual(gateway.dispatches, [])
        self.assertEqual(plan.release_target.release_tag, "v0.1.40")

    def test_legacy_v0140_publication_is_a_terminal_noop(self) -> None:
        provenance = make_legacy_v0140_provenance()
        correlation_id = sro.compute_correlation_id(provenance)
        candidate_dir = self.tmp / "legacy-v0140-candidate"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=correlation_id,
            bridge_commit=LEGACY_BRIDGE_SHA,
            run_id=LEGACY_CANDIDATE_RUN_ID,
            upstream_tag="v0.3.0",
            upstream_commit=LEGACY_UPSTREAM_COMMIT,
            native_release_tag="v0.3.0",
            native_manifest_sha256=LEGACY_NATIVE_MANIFEST_SHA,
            native_commit=LEGACY_NATIVE_COMMIT,
        )
        rewrite_legacy_candidate_manifest(candidate_dir)

        # The ordinary candidate path remains strict. Compatibility is selected
        # only after the immutable published-release identity is proven.
        with self.assertRaises(ContractError):
            rq.load_candidate(candidate_dir)
        gateway = self._published_candidate_gateway(
            candidate_dir,
            provenance=provenance,
            correlation_id=correlation_id,
            expected_qualification_gates=LEGACY_MANUAL_QUALIFICATION_GATES,
            expected_unproven_capabilities=LEGACY_MANUAL_UNPROVEN_CAPABILITIES,
        )
        plan = sro.advance_pipeline(
            gateway,
            provenance=provenance,
            workspace=self.tmp,
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.NOOP)
        self.assertEqual(gateway.dispatches, [])
        self.assertEqual(plan.release_target.release_tag, "v0.1.40")

    def test_two_releases_claiming_one_correlation_fail_closed(self) -> None:
        body = f"Orchestrator correlation: `{self.correlation_id}`"
        releases = [
            {"tag_name": "v0.1.40", "draft": False, "prerelease": False, "body": body},
            {"tag_name": "v0.1.41", "draft": False, "prerelease": True, "body": body},
        ]
        gateway = FakeGateway(json_routes=self._routes(releases=releases))
        with self.assertRaises(ContractError):
            sro.advance_pipeline(
                gateway, provenance=self.provenance, workspace=self.tmp
            )

    def test_candidate_success_dispatches_hosted_qualification(self) -> None:
        candidate_dir = self.tmp / "candidate-src"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
            run_id=CANDIDATE_RUN_ID,
        )
        members = directory_members(candidate_dir)
        succeeded = run_payload(
            run_id=CANDIDATE_RUN_ID,
            path=sro.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
        )
        routes = self._routes(candidate_runs=[succeeded])
        routes[f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}"] = succeeded
        routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=CANDIDATE_RUN_ID,
            name=rq.CANDIDATE_ARTIFACT_NAME,
            artifact_id=CANDIDATE_ARTIFACT_ID,
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}"] = {
            "status": "identical"
        }
        routes[f"repos/{BRIDGE_REPOSITORY}/compare/{BRIDGE_SHA}...{DEFAULT_BRANCH}"] = {
            "status": "ahead"
        }
        blobs = {
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{CANDIDATE_ARTIFACT_ID}/zip": flat_zip(
                members
            )
        }
        gateway = FakeGateway(json_routes=routes, blob_routes=blobs)
        dispatched = run_payload(
            run_id="4201",
            path=sro.QUALIFICATION_WORKFLOW_PATH,
            run_name=sro.qualification_run_name(self.correlation_id, CANDIDATE_RUN_ID),
            status="in_progress",
            conclusion=None,
        )
        readback_key = sro._workflow_runs_path(
            workflow_file=sro.QUALIFICATION_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        original_dispatch = gateway.dispatch_workflow

        def dispatch(**kwargs: Any) -> None:
            original_dispatch(**kwargs)
            gateway.json_routes[readback_key] = runs_response([dispatched])

        gateway.dispatch_workflow = dispatch  # type: ignore[assignment]
        plan = sro.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(
            plan.action, sro.OrchestrationAction.DISPATCH_QUALIFICATION
        )
        self.assertEqual(plan.candidate_run_id, CANDIDATE_RUN_ID)
        self.assertEqual(plan.release_target.release_tag, "v0.1.40")
        self.assertEqual(len(gateway.dispatches), 1)
        record = gateway.dispatches[0]
        self.assertEqual(record["workflow_file"], sro.QUALIFICATION_WORKFLOW_FILE)
        self.assertEqual(record["ref"], DEFAULT_BRANCH)
        self.assertEqual(
            record["inputs"],
            {
                "orchestrator_correlation_id": self.correlation_id,
                "candidate_run_id": CANDIDATE_RUN_ID,
            },
        )

    def test_failed_qualification_blocks_instead_of_retrying_forever(self) -> None:
        candidate_dir = self.tmp / "candidate-failed-qualification"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
            run_id=CANDIDATE_RUN_ID,
        )
        succeeded = run_payload(
            run_id=CANDIDATE_RUN_ID,
            path=sro.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
        )
        failed_qualification = run_payload(
            run_id="4201",
            path=sro.QUALIFICATION_WORKFLOW_PATH,
            run_name=sro.qualification_run_name(self.correlation_id, CANDIDATE_RUN_ID),
            status="completed",
            conclusion="failure",
        )
        routes = self._routes(
            candidate_runs=[succeeded], qualification_runs=[failed_qualification]
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}"] = succeeded
        routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=CANDIDATE_RUN_ID,
            name=rq.CANDIDATE_ARTIFACT_NAME,
            artifact_id=CANDIDATE_ARTIFACT_ID,
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}"] = {
            "status": "identical"
        }
        gateway = FakeGateway(
            json_routes=routes,
            blob_routes={
                f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{CANDIDATE_ARTIFACT_ID}/zip": flat_zip(
                    directory_members(candidate_dir)
                )
            },
        )
        plan = sro.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.BLOCKED)
        self.assertEqual(gateway.dispatches, [])
        self.assertIn("automatic qualification retries are disabled", plan.reason)

    def test_duplicate_candidate_artifacts_fail_closed(self) -> None:
        candidate_dir = self.tmp / "candidate-src"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
            run_id=CANDIDATE_RUN_ID,
        )
        members = directory_members(candidate_dir)
        succeeded = run_payload(
            run_id=CANDIDATE_RUN_ID,
            path=sro.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
        )
        routes = self._routes(candidate_runs=[succeeded])
        routes[f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}"] = succeeded
        routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=CANDIDATE_RUN_ID,
            name=rq.CANDIDATE_ARTIFACT_NAME,
            artifact_id=CANDIDATE_ARTIFACT_ID,
            extra=[
                {
                    "id": 8,
                    "name": rq.CANDIDATE_ARTIFACT_NAME,
                    "expired": False,
                    "workflow_run": {"id": int(CANDIDATE_RUN_ID)},
                }
            ],
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}"] = {
            "status": "identical"
        }
        blobs = {
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{CANDIDATE_ARTIFACT_ID}/zip": flat_zip(
                members
            )
        }
        gateway = FakeGateway(json_routes=routes, blob_routes=blobs)
        with self.assertRaises(ContractError):
            sro.advance_pipeline(
                gateway, provenance=self.provenance, workspace=self.tmp
            )

    def _assert_candidate_manifest_binding_mismatch(
        self, **candidate_overrides: Any
    ) -> None:
        candidate_dir = self.tmp / "candidate-src"
        candidate_fields: dict[str, Any] = {
            "release_tag": "v0.1.40",
            "release_rebuild": 0,
            "correlation_id": self.correlation_id,
            "run_id": CANDIDATE_RUN_ID,
        }
        candidate_fields.update(candidate_overrides)
        write_bridge_candidate(candidate_dir, **candidate_fields)
        members = directory_members(candidate_dir)
        succeeded = run_payload(
            run_id=CANDIDATE_RUN_ID,
            path=sro.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
        )
        routes = self._routes(candidate_runs=[succeeded])
        routes[f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}"] = succeeded
        routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=CANDIDATE_RUN_ID,
            name=rq.CANDIDATE_ARTIFACT_NAME,
            artifact_id=CANDIDATE_ARTIFACT_ID,
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}"] = {
            "status": "identical"
        }
        blobs = {
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{CANDIDATE_ARTIFACT_ID}/zip": flat_zip(
                members
            )
        }
        gateway = FakeGateway(json_routes=routes, blob_routes=blobs)
        with self.assertRaises(ContractError):
            sro.advance_pipeline(
                gateway, provenance=self.provenance, workspace=self.tmp
            )

    def test_candidate_manifest_source_that_contradicts_run_name_fails_closed(
        self,
    ) -> None:
        self._assert_candidate_manifest_binding_mismatch(
            bridge_commit=ADVANCED_BRIDGE_SHA
        )

    def test_candidate_manifest_tag_that_contradicts_run_name_fails_closed(
        self,
    ) -> None:
        self._assert_candidate_manifest_binding_mismatch(release_tag="v0.1.41")

    def test_full_pipeline_dispatches_publish_after_exact_attestation(self) -> None:
        candidate_dir = self.tmp / "candidate-src"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
            run_id=CANDIDATE_RUN_ID,
        )
        members = directory_members(candidate_dir)
        manifest, fingerprint = rq.load_candidate(candidate_dir)
        attestation = rq.build_attestation(
            manifest=manifest,
            candidate_fingerprint=fingerprint,
            candidate_run_id=CANDIDATE_RUN_ID,
            candidate_artifact_id=CANDIDATE_ARTIFACT_ID,
            candidate_run_attempt=1,
            **qualification_identity(
                qualification_run_id=QUALIFICATION_RUN_ID,
                qualification_source_sha=HEAD_SHA,
            ),
            harness_digest=rq.harness_source_sha256(Path(__file__).resolve().parent),
            environment=qualification_environment(),
            speech_phase=speech_phase(),
            tts_phase=tts_phase(),
        )
        attestation_bytes = rq.canonical_json(attestation).encode("utf-8")

        candidate_run = run_payload(
            run_id=CANDIDATE_RUN_ID,
            path=sro.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
        )
        qualification_run = run_payload(
            run_id=QUALIFICATION_RUN_ID,
            path=sro.QUALIFICATION_WORKFLOW_PATH,
            run_name=sro.qualification_run_name(
                self.correlation_id, CANDIDATE_RUN_ID
            ),
        )
        routes = self._routes(
            candidate_runs=[candidate_run], qualification_runs=[qualification_run]
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}"] = candidate_run
        routes[f"repos/{BRIDGE_REPOSITORY}/actions/runs/{QUALIFICATION_RUN_ID}"] = qualification_run
        routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=CANDIDATE_RUN_ID,
            name=rq.CANDIDATE_ARTIFACT_NAME,
            artifact_id=CANDIDATE_ARTIFACT_ID,
        )
        routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{QUALIFICATION_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=QUALIFICATION_RUN_ID,
            name=rq.ATTESTATION_ARTIFACT_NAME,
            artifact_id=QUALIFICATION_ARTIFACT_ID,
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}"] = {
            "status": "identical"
        }
        blobs = {
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{CANDIDATE_ARTIFACT_ID}/zip": flat_zip(
                members
            ),
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{QUALIFICATION_ARTIFACT_ID}/zip": flat_zip(
                {"qualification-attestation.json": attestation_bytes}
            ),
        }
        gateway = FakeGateway(json_routes=routes, blob_routes=blobs)
        publish_name = sro.publish_run_name(
            self.correlation_id,
            CANDIDATE_RUN_ID,
            QUALIFICATION_RUN_ID,
            self.binding,
        )
        publish_run = run_payload(
            run_id="701",
            path=sro.PUBLISH_WORKFLOW_PATH,
            run_name=publish_name,
            status="in_progress",
            conclusion=None,
        )
        publish_key = sro._workflow_runs_path(
            workflow_file=sro.PUBLISH_WORKFLOW_FILE,
            default_branch=DEFAULT_BRANCH,
            created_since=NATIVE_PUBLISHED_AT,
        )
        original_dispatch = gateway.dispatch_workflow

        def dispatch(**kwargs: Any) -> None:
            original_dispatch(**kwargs)
            gateway.json_routes[publish_key] = runs_response([publish_run])

        gateway.dispatch_workflow = dispatch  # type: ignore[assignment]

        plan = sro.advance_pipeline(
            gateway, provenance=self.provenance, workspace=self.tmp
        )
        self.assertEqual(plan.action, sro.OrchestrationAction.DISPATCH_PUBLISH)
        self.assertEqual(len(gateway.dispatches), 1)
        inputs = gateway.dispatches[0]["inputs"]
        self.assertEqual(inputs["candidate_run_id"], CANDIDATE_RUN_ID)
        self.assertEqual(inputs["qualification_run_id"], QUALIFICATION_RUN_ID)
        self.assertEqual(inputs["release_tag"], "v0.1.40")
        self.assertEqual(inputs["bridge_source_sha"], BRIDGE_SHA)
        self.assertEqual(inputs["publish_approved"], "true")
        self.assertEqual(gateway.dispatches[0]["ref"], DEFAULT_BRANCH)
        self.assertEqual(plan.dispatched_run_id, "701")
        qualification_reachability_path = (
            f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}"
        )
        self.assertEqual(gateway.api_paths.count(qualification_reachability_path), 2)

    def test_attestation_bound_to_another_candidate_fails_closed(self) -> None:
        candidate_dir = self.tmp / "candidate-src"
        write_bridge_candidate(
            candidate_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
            run_id=CANDIDATE_RUN_ID,
        )
        members = directory_members(candidate_dir)
        other_dir = self.tmp / "other-src"
        write_bridge_candidate(
            other_dir,
            release_tag="v0.1.40",
            release_rebuild=0,
            correlation_id=self.correlation_id,
            run_id=CANDIDATE_RUN_ID,
            marker=b"tampered",
        )
        other_manifest, other_fingerprint = rq.load_candidate(other_dir)
        attestation = rq.build_attestation(
            manifest=other_manifest,
            candidate_fingerprint=other_fingerprint,
            candidate_run_id=CANDIDATE_RUN_ID,
            candidate_artifact_id=CANDIDATE_ARTIFACT_ID,
            candidate_run_attempt=1,
            **qualification_identity(
                qualification_run_id=QUALIFICATION_RUN_ID,
                qualification_source_sha=HEAD_SHA,
            ),
            harness_digest=rq.harness_source_sha256(Path(__file__).resolve().parent),
            environment=qualification_environment(),
            speech_phase=speech_phase(),
            tts_phase=tts_phase(),
        )
        attestation_bytes = rq.canonical_json(attestation).encode("utf-8")
        candidate_run = run_payload(
            run_id=CANDIDATE_RUN_ID,
            path=sro.CANDIDATE_WORKFLOW_PATH,
            run_name=self.candidate_name,
        )
        qualification_run = run_payload(
            run_id=QUALIFICATION_RUN_ID,
            path=sro.QUALIFICATION_WORKFLOW_PATH,
            run_name=sro.qualification_run_name(
                self.correlation_id, CANDIDATE_RUN_ID
            ),
        )
        routes = self._routes(
            candidate_runs=[candidate_run], qualification_runs=[qualification_run]
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}"] = candidate_run
        routes[f"repos/{BRIDGE_REPOSITORY}/actions/runs/{QUALIFICATION_RUN_ID}"] = qualification_run
        routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=CANDIDATE_RUN_ID,
            name=rq.CANDIDATE_ARTIFACT_NAME,
            artifact_id=CANDIDATE_ARTIFACT_ID,
        )
        routes[
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{QUALIFICATION_RUN_ID}/artifacts?per_page=100"
        ] = artifact_inventory(
            run_id=QUALIFICATION_RUN_ID,
            name=rq.ATTESTATION_ARTIFACT_NAME,
            artifact_id=QUALIFICATION_ARTIFACT_ID,
        )
        routes[f"repos/{BRIDGE_REPOSITORY}/compare/{HEAD_SHA}...{DEFAULT_BRANCH}"] = {
            "status": "identical"
        }
        blobs = {
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{CANDIDATE_ARTIFACT_ID}/zip": flat_zip(
                members
            ),
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{QUALIFICATION_ARTIFACT_ID}/zip": flat_zip(
                {"qualification-attestation.json": attestation_bytes}
            ),
        }
        gateway = FakeGateway(json_routes=routes, blob_routes=blobs)
        with self.assertRaises(ContractError):
            sro.advance_pipeline(
                gateway, provenance=self.provenance, workspace=self.tmp
            )
        self.assertEqual(gateway.dispatches, [])

    def test_post_dispatch_readback_absence_fails_closed(self) -> None:
        routes = self._routes(
            releases=[asset_release_stub()]
        )
        gateway = FakeGateway(json_routes=routes)
        with self.assertRaises(ContractError):
            sro.advance_pipeline(
                gateway, provenance=self.provenance, workspace=self.tmp
            )
        self.assertEqual(len(gateway.dispatches), 1)

    def test_advance_pipeline_fails_closed_on_malformed_correlated_candidate_run(
        self,
    ) -> None:
        malformed_run = run_payload(
            run_id="501",
            path=sro.CANDIDATE_WORKFLOW_PATH,
            run_name=(
                f"bridge-candidate {self.correlation_id} source:not-a-sha "
                "tag:v0.1.40 rebuild:0"
            ),
            status="in_progress",
            conclusion=None,
        )
        routes = self._routes(candidate_runs=[malformed_run])
        gateway = FakeGateway(json_routes=routes)
        with self.assertRaises(ContractError):
            sro.advance_pipeline(
                gateway, provenance=self.provenance, workspace=self.tmp
            )
        self.assertEqual(gateway.dispatches, [])


if __name__ == "__main__":
    unittest.main()
