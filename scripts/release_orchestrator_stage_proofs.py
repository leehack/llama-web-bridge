"""Candidate and qualification run proofs for the stable release orchestrator.

Binds a candidate manifest to its exact provenance and correlation, and proves
a candidate or qualification run from its run record, reachability from the
default branch, its unique artifact, and the hosted qualification attestation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from release_contract import (
    ASSETS_REPOSITORY,
    BRIDGE_REPOSITORY,
    ContractError,
    NATIVE_REPOSITORY,
    require_sha256,
    validate_release_identity,
)
from generate_release_manifest import ARTIFACTS
import release_qualification as rq
from release_orchestrator_model import (
    CANDIDATE_WORKFLOW_PATH,
    NativeProvenance,
    PipelineBinding,
    QUALIFICATION_WORKFLOW_PATH,
    _COMMIT_RE,
    _require_str,
)
from release_orchestrator_transport import Gateway


def validate_candidate_manifest(
    manifest: Mapping[str, Any],
    *,
    provenance: NativeProvenance,
    correlation_id: str,
    expected_release_tag: str | None = None,
    expected_bridge_source_sha: str | None = None,
    expected_run_id: str | None = None,
) -> PipelineBinding:
    """Fail closed unless a manifest binds this exact provenance and correlation."""
    expected = {
        "assets_repository": ASSETS_REPOSITORY,
        "bridge_repository": BRIDGE_REPOSITORY,
        "bridge_commit": expected_bridge_source_sha or manifest.get("bridge_commit"),
        "upstream_repository": "ggml-org/llama.cpp",
        "upstream_tag": provenance.upstream_tag,
        "upstream_commit": provenance.upstream_commit,
        "native_repository": NATIVE_REPOSITORY,
        "native_release_tag": provenance.native_release_tag,
        "native_manifest_sha256": provenance.native_manifest_sha256,
        "native_commit": provenance.native_commit,
        "orchestrator_correlation_id": correlation_id,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ContractError(
                f"candidate manifest {key} is {manifest.get(key)!r}, expected {value!r}"
            )
    bridge_commit = _require_str(manifest.get("bridge_commit"), "manifest bridge_commit")
    if _COMMIT_RE.fullmatch(bridge_commit) is None:
        raise ContractError("manifest bridge_commit must be a 40-hex commit SHA")
    if expected_run_id is not None and manifest.get("github_run_id") != expected_run_id:
        raise ContractError(
            f"candidate manifest github_run_id is {manifest.get('github_run_id')!r}, "
            f"expected {expected_run_id!r}"
        )
    release_tag = _require_str(manifest.get("release_tag"), "manifest release_tag")
    rebuild = manifest.get("release_rebuild")
    if not isinstance(rebuild, int) or isinstance(rebuild, bool) or rebuild < 0:
        raise ContractError("manifest release_rebuild must be a non-negative integer")
    if expected_release_tag is not None and release_tag != expected_release_tag:
        raise ContractError(
            f"candidate manifest release_tag is {release_tag!r}, expected "
            f"{expected_release_tag!r}"
        )
    validate_release_identity(release_tag, rebuild, provenance.upstream_tag)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(ARTIFACTS):
        raise ContractError("candidate manifest does not record exactly the artifact set")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise ContractError(f"candidate manifest artifact {name!r} is malformed")
        require_sha256(_require_str(record.get("sha256"), f"{name} sha256"), f"{name} sha256")
        size = record.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ContractError(f"candidate manifest artifact {name!r} has an invalid size")
    return PipelineBinding(
        bridge_source_sha=bridge_commit,
        release_tag=release_tag,
        release_rebuild=rebuild,
    )


@dataclass(frozen=True)
class CandidateEvidence:
    run_id: str
    artifact_id: int
    fingerprint: str
    binding: PipelineBinding
    manifest: Mapping[str, Any]
    directory: Path


def _require_reachable_from_main(
    gateway: Gateway, *, commit: str, default_branch: str, label: str
) -> None:
    payload = gateway.api_json(
        f"repos/{BRIDGE_REPOSITORY}/compare/{commit}...{default_branch}"
    )
    if not isinstance(payload, Mapping):
        raise ContractError("compare response must be a JSON object")
    status = payload.get("status")
    if status not in ("ahead", "identical"):
        raise ContractError(
            f"{label} commit {commit} is not reachable from {default_branch} "
            f"(compare status {status!r})"
        )


def _download_run_artifact(
    gateway: Gateway,
    *,
    run_id: str,
    artifact_name: str,
    artifact_type: str,
    workspace: Path,
) -> tuple[int, Path]:
    inventory = gateway.api_json(
        f"repos/{BRIDGE_REPOSITORY}/actions/runs/{run_id}/artifacts?per_page=100"
    )
    artifact_id = rq.validate_artifact_inventory(
        inventory, expected_run_id=run_id, expected_name=artifact_name
    )
    archive = workspace / f"{artifact_type}-{run_id}.zip"
    archive.write_bytes(
        gateway.download_bytes(
            f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{artifact_id}/zip",
            accept="application/vnd.github+json",
        )
    )
    destination = workspace / f"{artifact_type}-{run_id}"
    destination.mkdir(parents=True, exist_ok=True)
    rq._extract_flat_artifact_archive(archive, destination, artifact_type=artifact_type)
    return artifact_id, destination


def verify_candidate_run(
    gateway: Gateway,
    *,
    run_id: str,
    provenance: NativeProvenance,
    correlation_id: str,
    binding: PipelineBinding,
    default_branch: str,
    workspace: Path,
) -> CandidateEvidence:
    run = gateway.api_json(f"repos/{BRIDGE_REPOSITORY}/actions/runs/{run_id}")
    head_sha = rq.validate_workflow_run(
        run,
        expected_run_id=run_id,
        expected_workflow_path=CANDIDATE_WORKFLOW_PATH,
        expected_head_branch=default_branch,
        expected_run_attempt=1,
    )
    _require_reachable_from_main(
        gateway, commit=head_sha, default_branch=default_branch, label="candidate run head"
    )
    artifact_id, directory = _download_run_artifact(
        gateway,
        run_id=run_id,
        artifact_name=rq.CANDIDATE_ARTIFACT_NAME,
        artifact_type="candidate",
        workspace=workspace,
    )
    manifest, fingerprint = rq.load_candidate(directory)
    manifest_binding = validate_candidate_manifest(
        manifest,
        provenance=provenance,
        correlation_id=correlation_id,
        expected_release_tag=binding.release_tag,
        expected_bridge_source_sha=binding.bridge_source_sha,
        expected_run_id=run_id,
    )
    if manifest_binding != binding:
        raise ContractError(
            "candidate manifest contradicts the binding its run name advertises"
        )
    _require_reachable_from_main(
        gateway,
        commit=binding.bridge_source_sha,
        default_branch=default_branch,
        label="candidate bridge source",
    )
    return CandidateEvidence(
        run_id=run_id,
        artifact_id=artifact_id,
        fingerprint=fingerprint,
        binding=binding,
        manifest=manifest,
        directory=directory,
    )


def verify_qualification_run(
    gateway: Gateway,
    *,
    run_id: str,
    candidate: CandidateEvidence,
    provenance: NativeProvenance,
    correlation_id: str,
    default_branch: str,
    workspace: Path,
) -> dict[str, Any]:
    run = gateway.api_json(f"repos/{BRIDGE_REPOSITORY}/actions/runs/{run_id}")
    qualification_source_sha = rq.validate_workflow_run(
        run,
        expected_run_id=run_id,
        expected_workflow_path=QUALIFICATION_WORKFLOW_PATH,
        expected_head_branch=default_branch,
        expected_run_attempt=1,
    )
    _require_reachable_from_main(
        gateway,
        commit=qualification_source_sha,
        default_branch=default_branch,
        label="qualification workflow source",
    )
    _, directory = _download_run_artifact(
        gateway,
        run_id=run_id,
        artifact_name=rq.ATTESTATION_ARTIFACT_NAME,
        artifact_type="attestation",
        workspace=workspace,
    )
    attestation_path = directory / "qualification-attestation.json"
    if not attestation_path.is_file():
        raise ContractError("attestation artifact does not contain the canonical payload")
    attestation = rq.load_attestation_file(attestation_path)
    return rq.verify_attestation(
        attestation=attestation,
        candidate_dir=candidate.directory,
        candidate_fingerprint=candidate.fingerprint,
        candidate_run_id=candidate.run_id,
        candidate_artifact_id=candidate.artifact_id,
        candidate_run_attempt=1,
        qualification_run_id=run_id,
        qualification_run_attempt=1,
        qualification_source_sha=qualification_source_sha,
        bridge_source_sha=candidate.binding.bridge_source_sha,
        upstream_tag=provenance.upstream_tag,
        upstream_commit=provenance.upstream_commit,
        native_release_tag=provenance.native_release_tag,
        native_manifest_sha256=provenance.native_manifest_sha256,
        native_commit=provenance.native_commit,
        emscripten_version=_require_str(
            candidate.manifest.get("emscripten_version"),
            "candidate manifest emscripten_version",
        ),
        release_tag=candidate.binding.release_tag,
        release_rebuild=candidate.binding.release_rebuild,
        orchestrator_correlation_id=correlation_id,
    )
