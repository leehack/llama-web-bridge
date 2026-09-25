"""Shared fixtures for the stable release orchestrator contract suites.

Pinned identities, provenance and payload builders, the fake ``Gateway``,
and the ``AdvancePipelineTest`` setup that the driver suites share. It
defines no tests of its own.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from unittest import mock

from generate_release_manifest import ARTIFACTS, generate
from release_contract import (
    ASSETS_REPOSITORY,
    BRIDGE_REPOSITORY,
    ContractError,
    NATIVE_REPOSITORY,
)
from release_publication_state import PUBLICATION_FILES
import release_orchestrator_model as model
import release_orchestrator_run_names as run_names
import release_orchestrator_workflow_runs as workflow_runs
import stable_release_orchestrator as sro


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


def make_provenance(**overrides: Any) -> model.NativeProvenance:
    fields: dict[str, Any] = {
        "bridge_source_sha": BRIDGE_SHA,
        "bridge_build_sha": BRIDGE_SHA,
        "upstream_tag": "v0.2.0",
        "upstream_commit": UPSTREAM_COMMIT,
        "native_repo": NATIVE_REPOSITORY,
        "native_release_tag": "v0.2.0",
        "native_commit": NATIVE_COMMIT,
        "native_manifest_sha256": NATIVE_MANIFEST_SHA,
        "native_release_published_at": NATIVE_PUBLISHED_AT,
    }
    fields.update(overrides)
    return model.NativeProvenance(**fields)


def make_legacy_v0140_provenance(**overrides: Any) -> model.NativeProvenance:
    fields: dict[str, Any] = {
        "bridge_source_sha": LEGACY_BRIDGE_SHA,
        "bridge_build_sha": LEGACY_BRIDGE_SHA,
        "upstream_tag": "v0.3.0",
        "upstream_commit": LEGACY_UPSTREAM_COMMIT,
        "native_repo": NATIVE_REPOSITORY,
        "native_release_tag": "v0.3.0",
        "native_commit": LEGACY_NATIVE_COMMIT,
        "native_manifest_sha256": LEGACY_NATIVE_MANIFEST_SHA,
        "native_release_published_at": "2026-08-28T12:34:56Z",
    }
    fields.update(overrides)
    return model.NativeProvenance(**fields)


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


def aligned_release_stub(
    tag: str, provenance: model.NativeProvenance
) -> dict[str, Any]:
    """A listed publication carrying the deterministic release-note markers."""
    release = asset_release_stub(tag)
    release["body"] = (
        f"Native: `{provenance.native_repo}@{provenance.native_release_tag}`\n"
        f"Native manifest SHA-256: `{provenance.native_manifest_sha256}`\n"
        f"Orchestrator correlation: `{run_names.compute_correlation_id(provenance)}`\n"
    )
    return release


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


class AdvancePipelineFixture:
    """Setup and routes shared by every ``AdvancePipelineTest`` suite.

    Each ``release_orchestrator_driver*_test.py`` suite declares
    ``AdvancePipelineTest(AdvancePipelineFixture, unittest.TestCase)``, so
    its test IDs keep the original class name.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sro-advance-"))
        self.provenance = make_provenance()
        self.correlation_id = run_names.compute_correlation_id(self.provenance)
        self.binding = model.PipelineBinding(
            bridge_source_sha=BRIDGE_SHA, release_tag="v0.1.40", release_rebuild=0
        )
        self.candidate_name = run_names.candidate_run_name(self.correlation_id, self.binding)

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
            workflow_runs._workflow_runs_path(
                workflow_file=model.CANDIDATE_WORKFLOW_FILE,
                default_branch=DEFAULT_BRANCH,
                created_since=NATIVE_PUBLISHED_AT,
            ): runs_response(
                candidate_runs or []
            ),
            workflow_runs._workflow_runs_path(
                workflow_file=model.QUALIFICATION_WORKFLOW_FILE,
                default_branch=DEFAULT_BRANCH,
                created_since=NATIVE_PUBLISHED_AT,
            ): runs_response(
                qualification_runs or []
            ),
            workflow_runs._workflow_runs_path(
                workflow_file=model.PUBLISH_WORKFLOW_FILE,
                default_branch=DEFAULT_BRANCH,
                created_since=NATIVE_PUBLISHED_AT,
            ): runs_response(
                publish_runs or []
            ),
        }

    def _newer_native(self, **overrides: Any) -> model.NativeProvenance:
        fields: dict[str, Any] = {
            "upstream_tag": "v0.2.1",
            "upstream_commit": "d" * 40,
            "native_release_tag": "v0.2.1",
            "native_commit": "e" * 40,
            "native_manifest_sha256": "f" * 64,
            "native_release_published_at": "2026-08-21T12:34:56Z",
        }
        fields.update(overrides)
        return make_provenance(**fields)

    def _run_backlog(
        self, gateway: FakeGateway, provenances: list[model.NativeProvenance]
    ) -> tuple[int, dict[str, Any]]:
        provenance_list = self.tmp / "release-candidates.json"
        provenance_list.write_text(
            json.dumps([sro._provenance_to_dict(value) for value in provenances]),
            encoding="utf-8",
        )
        output_plan = self.tmp / "orchestration-plan.json"
        environment = {
            "GITHUB_EVENT_NAME": "schedule",
            "GITHUB_ACTOR": "github-actions",
            "GITHUB_TRIGGERING_ACTOR": "github-actions",
        }
        with (
            mock.patch.object(sro, "GhGateway", return_value=gateway),
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            result = sro.main(
                [
                    "orchestrate-backlog",
                    "--provenance-list-json",
                    str(provenance_list),
                    "--workspace",
                    str(self.tmp / "workspace"),
                    "--output-plan-json",
                    str(output_plan),
                ]
            )
        return result, json.loads(output_plan.read_text(encoding="utf-8"))
