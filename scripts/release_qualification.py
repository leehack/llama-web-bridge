#!/usr/bin/env python3
"""Digest-bound local release qualification and attestation validation.

Heavy real-model Qwen3-ASR and Qwen3-TTS gates cannot run on hosted CI runners,
so they are split into one required local command. That command consumes the
exact hosted candidate artifact -- it never rebuilds it -- runs the heavy gates,
and emits a canonical attestation bound to the candidate digest and to every
provenance identity recorded in the candidate manifest. A maintainer-only
ingestion workflow validates the attestation against the same candidate, and
publication refuses to publish unless an exact successful ingestion run attested
the exact candidate it is about to publish.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import resource
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import wave
import zipfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from release_contract import (
    BRIDGE_REPOSITORY,
    NATIVE_REPOSITORY,
    ContractError,
    require_correlation_id,
    require_sha256,
)
from generate_release_manifest import LOCAL_ATTESTATION_REQUIRED
from release_publication_state import (
    PUBLICATION_FILES,
    CandidateIdentity,
    validate_candidate,
)


QUALIFICATION_SCHEMA_VERSION = 1
ATTESTATION_TYPE = "llama-web-bridge-local-qualification"
HARNESS_VERSION = "2.0.0"

# The attestation is transported into the maintainer ingestion workflow as a
# single-line base64 blob so multiline dispatch inputs cannot be truncated,
# re-quoted, or line-folded in transit. A 32 KiB decoded ceiling leaves ample
# room for base64 expansion and the rest of the workflow-dispatch payload while
# remaining far above the canonical attestation's expected size.
MAX_ATTESTATION_BYTES = 32768
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")

CANDIDATE_WORKFLOW_PATH = ".github/workflows/bridge_candidate.yml"
CANDIDATE_ARTIFACT_NAME = "exact-webgpu-bridge-dist"
ATTESTATION_WORKFLOW_PATH = ".github/workflows/qualification_attestation.yml"
ATTESTATION_ARTIFACT_NAME = "qualification-attestation"

# Gates the hosted candidate run proves. They are copied from the candidate
# manifest and re-checked, never asserted by the local harness itself.
HOSTED_GATES = ("state_persistence", "multimodal")
# Gates this local harness proves against the exact candidate artifact.
LOCAL_GATES = ("speech_to_text", "text_to_speech")

# Local qualification proves programmatic transcript, lifecycle, and WAV
# container correctness only. Nothing here listens to the generated audio, so
# these three capabilities stay explicitly unproven in every attestation.
REQUIRED_UNPROVEN_CAPABILITIES = {
    "real_device_intelligibility": "unproven",
    "real_device_playback": "unproven",
    "speaker_reference_fidelity": "unproven",
}

REQUIRED_SPEECH_MODES = (
    ("wasm32", "direct"),
    ("wasm32", "worker"),
    ("wasm64", "direct"),
    ("wasm64", "worker"),
)
# The validated Qwen3-TTS pair is memory64-only in practice; wasm32 is not a
# product path for it, so it is deliberately absent rather than silently skipped.
REQUIRED_TTS_MODES = (
    ("wasm64", "direct"),
    ("wasm64", "worker"),
)

SPEECH_PHASE_KEYS = (
    "cancellation",
    "cold_transcript",
    "model_load",
    "projector_load",
    "silence",
    "warm_transcript",
)
TTS_PHASE_KEYS = ("model_load", "projector_load", "synthesis")

TTS_WAV_SAMPLE_RATE = 24000
TTS_WAV_CHANNELS = 1
TTS_WAV_BITS_PER_SAMPLE = 16

STATE_SMOKE_MODEL_SHA256 = (
    "81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb"
)
MULTIMODAL_MODEL_SHA256 = (
    "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517"
)
MULTIMODAL_MMPROJ_SHA256 = (
    "56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453"
)
SPEECH_MODEL_SHA256 = (
    "bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971"
)
SPEECH_MMPROJ_SHA256 = (
    "41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d"
)
SPEECH_AUDIO_SHA256 = (
    "f9b4440ac8393e47c14a6240e9739dea09b645bb1592b8f2dd48feb9666cea7f"
)
TTS_MODEL_SHA256 = (
    "8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129"
)
TTS_MMPROJ_SHA256 = (
    "6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2"
)

EXPECTED_MODEL_PINS = {
    "multimodal_mmproj_sha256": MULTIMODAL_MMPROJ_SHA256,
    "multimodal_model_sha256": MULTIMODAL_MODEL_SHA256,
    "speech_audio_sha256": SPEECH_AUDIO_SHA256,
    "speech_mmproj_sha256": SPEECH_MMPROJ_SHA256,
    "speech_model_sha256": SPEECH_MODEL_SHA256,
    "state_smoke_model_sha256": STATE_SMOKE_MODEL_SHA256,
    "tts_mmproj_sha256": TTS_MMPROJ_SHA256,
    "tts_model_sha256": TTS_MODEL_SHA256,
}

# Every source file whose behaviour the heavy gates depend on. The digest binds
# an attestation to the exact harness that produced it, so publication can prove
# the maintainer ran the harness from the exact bridge source being published.
HARNESS_SOURCES = (
    "multimodal_browser_smoke.py",
    "release_qualification.py",
    "speech_to_text_browser_smoke.py",
    "state_persistence_browser_smoke.py",
    "text_to_speech_browser_smoke.py",
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_RE = re.compile(r"^[1-9][0-9]*$")

ATTESTATION_KEYS = (
    "attestation_type",
    "bridge_repository",
    "bridge_source_sha",
    "candidate_fingerprint",
    "candidate_run_id",
    "candidate_run_url",
    "emscripten_version",
    "harness_source_sha256",
    "harness_version",
    "hosted_gates",
    "local_gates",
    "model_pins",
    "native_commit",
    "native_manifest_sha256",
    "native_release_tag",
    "native_repository",
    "orchestrator_correlation_id",
    "phases",
    "release_rebuild",
    "release_tag",
    "schema_version",
    "unproven_capabilities",
    "upstream_commit",
    "upstream_repository",
    "upstream_tag",
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def parse_attestation_json(raw_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_json, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractError(f"malformed attestation JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("attestation root must be a JSON object")
    return payload


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def encode_attestation(canonical_text: str) -> str:
    """Encode canonical attestation text as one bounded single-line base64 blob."""
    encoded = canonical_text.encode("utf-8")
    if len(encoded) > MAX_ATTESTATION_BYTES:
        raise ContractError(
            f"attestation is {len(encoded)} bytes; the transport bound is "
            f"{MAX_ATTESTATION_BYTES}"
        )
    return base64.b64encode(encoded).decode("ascii")


def decode_attestation(blob: str) -> tuple[dict[str, Any], str]:
    """Decode a transported attestation, rejecting oversized or noncanonical input.

    Returns the parsed payload and its canonical text. The decoded bytes must be
    byte-for-byte the canonical serialization of the payload, so a re-ordered,
    re-indented, or padded variant of an otherwise valid attestation is refused
    instead of being silently normalized.
    """
    # The local output file ends with one conventional newline. The dispatch
    # value itself must otherwise be one uninterrupted base64 line; silently
    # folding arbitrary whitespace would contradict the transport boundary.
    if blob.endswith("\r\n"):
        encoded = blob[:-2]
    elif blob.endswith("\n"):
        encoded = blob[:-1]
    else:
        encoded = blob
    if not encoded:
        raise ContractError("attestation payload is empty")
    if any(character.isspace() for character in encoded):
        raise ContractError("attestation payload must be exactly one base64 line")
    maximum_encoded_length = ((MAX_ATTESTATION_BYTES + 2) // 3) * 4
    if len(encoded) > maximum_encoded_length:
        raise ContractError("encoded attestation exceeds the transport bound")
    if _BASE64_RE.fullmatch(encoded) is None:
        raise ContractError("attestation payload is not single-line base64")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ContractError(f"attestation payload is not valid base64: {exc}") from exc
    if base64.b64encode(raw).decode("ascii") != encoded:
        raise ContractError("attestation payload is not canonical base64")
    if len(raw) > MAX_ATTESTATION_BYTES:
        raise ContractError(
            f"decoded attestation is {len(raw)} bytes; the bound is "
            f"{MAX_ATTESTATION_BYTES}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError(f"attestation payload is not UTF-8: {exc}") from exc
    payload = parse_attestation_json(text)
    canonical = canonical_json(payload)
    if text != canonical:
        raise ContractError(
            "attestation payload is not the canonical serialization of its own content"
        )
    return payload, canonical


def sanitize_diagnostic_text(text: str) -> str:
    """Drop credentials, query strings, and fragments from any URL in diagnostics."""
    def replace(match: re.Match[str]) -> str:
        parts = urlsplit(match.group(0))
        netloc = parts.netloc.rsplit("@", 1)[-1]
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))

    sanitized = re.sub(r"https?://[^\s\"'<>]+", replace, text)
    sanitized = re.sub(
        r"(?im)\bauthorization\s*:\s*(?:bearer|basic)\s+\S+",
        "Authorization: <redacted>",
        sanitized,
    )
    sanitized = re.sub(
        r"(?im)\b(?:[A-Za-z0-9_-]*(?:token|secret|password|credential)"
        r"[A-Za-z0-9_-]*|[A-Za-z0-9_-]*api[_-]?key[A-Za-z0-9_-]*)"
        r"\s*[:=]\s*\S+",
        "<redacted-credential>",
        sanitized,
    )
    return sanitized


def harness_source_sha256(scripts_dir: Path) -> str:
    """Digest every harness source so an attestation names the code that ran."""
    digest = hashlib.sha256()
    for name in sorted(HARNESS_SOURCES):
        path = scripts_dir / name
        if not path.is_file():
            raise ContractError(f"harness source is missing: {name}")
        data = path.read_bytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _harness_source_sha256_at_commit(repository: Path, bridge_sha: str) -> str:
    if _COMMIT_RE.fullmatch(bridge_sha) is None:
        raise ContractError("candidate bridge source must be a lowercase 40-hex SHA")
    digest = hashlib.sha256()
    for name in sorted(HARNESS_SOURCES):
        result = subprocess.run(
            ["git", "-C", str(repository), "show", f"{bridge_sha}:scripts/{name}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            diagnostic = result.stderr.decode("utf-8", errors="replace")
            raise ContractError(
                f"could not read harness source {name!r} at {bridge_sha}: "
                f"{sanitize_diagnostic_text(diagnostic.strip())}"
            )
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(len(result.stdout).to_bytes(8, "big"))
        digest.update(result.stdout)
    return digest.hexdigest()


def require_harness_matches_bridge_source(
    scripts_dir: Path, bridge_sha: str
) -> str:
    """Require the local harness bytes to equal the exact candidate source."""
    local_digest = harness_source_sha256(scripts_dir)
    source_digest = _harness_source_sha256_at_commit(scripts_dir.parent, bridge_sha)
    if local_digest != source_digest:
        raise ContractError(
            "local qualification harness does not match the exact candidate "
            f"bridge source {bridge_sha}: local={local_digest}, source={source_digest}"
        )
    return local_digest


def _require_str(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} {key} must be a non-empty string")
    return value


def _require_int(payload: Mapping[str, Any], key: str, label: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError(f"{label} {key} must be an integer")
    return value


def load_candidate(directory: Path) -> tuple[dict[str, Any], str]:
    """Validate an exact candidate directory and return its manifest and digest.

    The candidate is validated with the same routine publication uses, so the
    fingerprint an attestation binds is the identical fingerprint publication
    recomputes; a divergent second definition could never be kept in step.
    """
    if not directory.is_dir() or directory.is_symlink():
        raise ContractError(f"candidate directory does not exist: {directory}")
    entries = list(directory.iterdir())
    actual = {entry.name for entry in entries}
    expected = set(PUBLICATION_FILES)
    if actual != expected:
        unexpected = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise ContractError(
            "candidate directory must contain exactly the publication files "
            f"(unexpected: {unexpected}, missing: {missing})"
        )
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ContractError(
                f"candidate entry must be an immutable regular file: {entry.name}"
            )

    manifest_path = directory / "manifest.json"
    try:
        manifest = parse_attestation_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractError(f"could not read candidate manifest: {exc}") from exc

    identity = CandidateIdentity(
        release_tag=_require_str(manifest, "release_tag", "candidate manifest"),
        release_rebuild=_require_int(manifest, "release_rebuild", "candidate manifest"),
        assets_repo=_require_str(manifest, "assets_repository", "candidate manifest"),
        bridge_commit=_require_str(manifest, "bridge_commit", "candidate manifest"),
        upstream_tag=_require_str(manifest, "upstream_tag", "candidate manifest"),
        upstream_commit=_require_str(manifest, "upstream_commit", "candidate manifest"),
        native_release_tag=_require_str(
            manifest, "native_release_tag", "candidate manifest"
        ),
        native_manifest_sha256=_require_str(
            manifest, "native_manifest_sha256", "candidate manifest"
        ),
        native_commit=_require_str(manifest, "native_commit", "candidate manifest"),
        emscripten_version=_require_str(
            manifest, "emscripten_version", "candidate manifest"
        ),
        orchestrator_correlation_id=_require_str(
            manifest, "orchestrator_correlation_id", "candidate manifest"
        ),
        github_run_id=_require_str(manifest, "github_run_id", "candidate manifest"),
        github_run_url=_require_str(manifest, "github_run_url", "candidate manifest"),
    )
    fingerprint = validate_candidate(directory, identity)
    return manifest, fingerprint


def _manifest_hosted_gates(manifest: Mapping[str, Any]) -> dict[str, str]:
    gates = manifest.get("qualification_gates")
    if not isinstance(gates, Mapping):
        raise ContractError("candidate manifest qualification_gates must be an object")
    hosted: dict[str, str] = {}
    for name in HOSTED_GATES:
        value = gates.get(name)
        if value != "passed":
            raise ContractError(
                f"candidate manifest hosted gate {name!r} is {value!r}; must be 'passed'"
            )
        hosted[name] = value
    for name in LOCAL_GATES:
        value = gates.get(name)
        if value != LOCAL_ATTESTATION_REQUIRED:
            raise ContractError(
                f"candidate manifest gate {name!r} is {value!r}; a candidate must "
                f"declare {LOCAL_ATTESTATION_REQUIRED!r} rather than claim a "
                "hosted pass it never ran"
            )
    return hosted


def validate_workflow_run(
    run: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_workflow_path: str,
    expected_head_branch: str,
    expected_run_attempt: int | None = None,
) -> str:
    """Fail closed unless a run is the exact successful dispatch we require.

    Artifact names are attacker-chosen strings, so a run ID alone proves nothing.
    Repository, workflow file, dispatch event, default branch, and success are
    all checked before any artifact from the run is trusted.

    Returns the run's head commit. The caller proves that commit is reachable
    from the default branch rather than equal to its current head, so an
    unrelated push to the default branch cannot invalidate an in-flight
    candidate while a fork or feature branch is still refused.
    """
    if not isinstance(run, Mapping):
        raise ContractError("workflow run payload must be a JSON object")
    if _RUN_ID_RE.fullmatch(expected_run_id) is None:
        raise ContractError("run id must be a positive integer")
    actual_id = run.get("id")
    if not isinstance(actual_id, int) or str(actual_id) != expected_run_id:
        raise ContractError(
            f"workflow run id mismatch: expected {expected_run_id}, got {actual_id!r}"
        )
    for field in ("repository", "head_repository"):
        repository = run.get(field)
        if not isinstance(repository, Mapping) or repository.get(
            "full_name"
        ) != BRIDGE_REPOSITORY:
            raise ContractError(
                f"workflow run {field} must be exactly {BRIDGE_REPOSITORY}"
            )
    if run.get("path") != expected_workflow_path:
        raise ContractError(
            f"workflow run path mismatch: expected {expected_workflow_path}, "
            f"got {run.get('path')!r}"
        )
    if run.get("event") != "workflow_dispatch":
        raise ContractError(
            f"workflow run event must be workflow_dispatch, got {run.get('event')!r}"
        )
    if run.get("status") != "completed":
        raise ContractError(
            f"workflow run status must be completed, got {run.get('status')!r}"
        )
    if run.get("conclusion") != "success":
        raise ContractError(
            f"workflow run conclusion must be success, got {run.get('conclusion')!r}"
        )
    if expected_run_attempt is not None:
        if (
            not isinstance(expected_run_attempt, int)
            or isinstance(expected_run_attempt, bool)
            or expected_run_attempt <= 0
        ):
            raise ContractError("expected run attempt must be a positive integer")
        actual_run_attempt = run.get("run_attempt")
        if (
            not isinstance(actual_run_attempt, int)
            or isinstance(actual_run_attempt, bool)
            or actual_run_attempt != expected_run_attempt
        ):
            raise ContractError(
                f"workflow run attempt must be {expected_run_attempt}, got "
                f"{actual_run_attempt!r}"
            )
    if not expected_head_branch:
        raise ContractError("expected head branch is required")
    if run.get("head_branch") != expected_head_branch:
        raise ContractError(
            f"workflow run head_branch must be {expected_head_branch!r}, got "
            f"{run.get('head_branch')!r}"
        )
    head_sha = run.get("head_sha")
    if not isinstance(head_sha, str) or _COMMIT_RE.fullmatch(head_sha) is None:
        raise ContractError("workflow run head_sha must be a 40-hex commit SHA")
    return head_sha


def validate_artifact_inventory(
    inventory: Mapping[str, Any], *, expected_run_id: str, expected_name: str
) -> int:
    """Require exactly one live artifact of the expected name in the exact run."""
    if not isinstance(inventory, Mapping):
        raise ContractError("artifact inventory must be a JSON object")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list):
        raise ContractError("artifact inventory is missing an artifacts array")
    total_count = inventory.get("total_count")
    if (
        not isinstance(total_count, int)
        or isinstance(total_count, bool)
        or total_count < 0
    ):
        raise ContractError("artifact inventory total_count must be a non-negative integer")
    if total_count != len(artifacts):
        raise ContractError(
            "artifact inventory is truncated; all run artifacts must be inspected "
            f"({len(artifacts)} records for total_count={total_count})"
        )
    matches: list[Mapping[str, Any]] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or not isinstance(
            artifact.get("name"), str
        ):
            raise ContractError("artifact inventory contains an invalid record")
        if artifact["name"] != expected_name:
            continue
        run = artifact.get("workflow_run")
        if not isinstance(run, Mapping) or str(run.get("id")) != expected_run_id:
            raise ContractError(
                f"artifact {expected_name!r} does not belong to run {expected_run_id}"
            )
        if artifact.get("expired") is not False:
            raise ContractError(f"artifact {expected_name!r} has expired")
        matches.append(artifact)
    if len(matches) != 1:
        raise ContractError(
            f"run {expected_run_id} must expose exactly one {expected_name!r} "
            f"artifact, found {len(matches)}"
        )
    artifact_id = matches[0].get("id")
    if (
        not isinstance(artifact_id, int)
        or isinstance(artifact_id, bool)
        or artifact_id <= 0
    ):
        raise ContractError(f"artifact {expected_name!r} has no positive integer id")
    return artifact_id


def build_attestation(
    *,
    manifest: Mapping[str, Any],
    candidate_fingerprint: str,
    candidate_run_id: str,
    harness_digest: str,
    speech_phase: Mapping[str, Any],
    tts_phase: Mapping[str, Any],
) -> dict[str, Any]:
    require_sha256(candidate_fingerprint, "candidate_fingerprint")
    require_sha256(harness_digest, "harness_source_sha256")
    if _RUN_ID_RE.fullmatch(candidate_run_id) is None:
        raise ContractError("candidate_run_id must be a positive integer")
    manifest_run_id = _require_str(manifest, "github_run_id", "candidate manifest")
    if manifest_run_id != candidate_run_id:
        raise ContractError(
            "candidate manifest github_run_id does not match the candidate run: "
            f"{manifest_run_id} != {candidate_run_id}"
        )
    require_correlation_id(
        _require_str(manifest, "orchestrator_correlation_id", "candidate manifest")
    )

    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "attestation_type": ATTESTATION_TYPE,
        "candidate_fingerprint": candidate_fingerprint,
        "candidate_run_id": candidate_run_id,
        "candidate_run_url": _require_str(
            manifest, "github_run_url", "candidate manifest"
        ),
        "bridge_repository": BRIDGE_REPOSITORY,
        "bridge_source_sha": manifest["bridge_commit"],
        "upstream_repository": "ggml-org/llama.cpp",
        "upstream_tag": manifest["upstream_tag"],
        "upstream_commit": manifest["upstream_commit"],
        "native_repository": NATIVE_REPOSITORY,
        "native_release_tag": manifest["native_release_tag"],
        "native_manifest_sha256": manifest["native_manifest_sha256"],
        "native_commit": manifest["native_commit"],
        "emscripten_version": manifest["emscripten_version"],
        "release_tag": manifest["release_tag"],
        "release_rebuild": manifest["release_rebuild"],
        "orchestrator_correlation_id": manifest["orchestrator_correlation_id"],
        "harness_version": HARNESS_VERSION,
        "harness_source_sha256": harness_digest,
        "model_pins": copy.deepcopy(EXPECTED_MODEL_PINS),
        "hosted_gates": _manifest_hosted_gates(manifest),
        "local_gates": {gate: "passed" for gate in LOCAL_GATES},
        "unproven_capabilities": copy.deepcopy(REQUIRED_UNPROVEN_CAPABILITIES),
        "phases": {
            "speech_to_text": copy.deepcopy(dict(speech_phase)),
            "text_to_speech": copy.deepcopy(dict(tts_phase)),
        },
    }


def _require_exact_mapping(
    value: Any, keys: Sequence[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    actual = set(value.keys())
    expected = set(keys)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise ContractError(f"{label} is missing required keys: {missing}")
    if unexpected:
        raise ContractError(f"{label} contains unexpected keys: {unexpected}")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _require_non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{label} must be a non-negative integer")
    return value


def _validate_phase(
    phase: Any,
    *,
    label: str,
    required_modes: Sequence[tuple[str, str]],
    phase_keys: Sequence[str],
    require_wav: bool,
) -> None:
    _require_exact_mapping(phase, ("max_rss_bytes", "modes", "total_ms"), label)
    _require_positive_int(phase["max_rss_bytes"], f"{label}.max_rss_bytes")
    phase_total = _require_non_negative_int(phase["total_ms"], f"{label}.total_ms")
    modes = phase["modes"]
    if not isinstance(modes, list):
        raise ContractError(f"{label}.modes must be an array")
    seen: list[tuple[str, str]] = []
    for index, mode in enumerate(modes):
        mode_label = f"{label}.modes[{index}]"
        keys = ["memory_mode", "phase_timings_ms", "runtime_mode", "total_ms"]
        if require_wav:
            keys.extend(("frames_generated", "peak", "rms", "truncated", "wav"))
        else:
            keys.extend(
                (
                    "cancellation_result",
                    "cold_transcript",
                    "silence_transcript",
                    "warm_transcript",
                )
            )
        _require_exact_mapping(mode, keys, mode_label)
        memory_mode = mode["memory_mode"]
        runtime_mode = mode["runtime_mode"]
        if not isinstance(memory_mode, str) or not isinstance(runtime_mode, str):
            raise ContractError(f"{mode_label} mode identifiers must be strings")
        mode_total = _require_non_negative_int(
            mode["total_ms"], f"{mode_label}.total_ms"
        )
        timings = _require_exact_mapping(
            mode["phase_timings_ms"], phase_keys, f"{mode_label}.phase_timings_ms"
        )
        for key in phase_keys:
            _require_non_negative_int(
                timings[key], f"{mode_label}.phase_timings_ms.{key}"
            )
        timing_total = sum(timings[key] for key in phase_keys)
        # Each browser phase is measured sequentially with Math.round(). Allow
        # one millisecond of aggregate rounding per phase, but reject a mode
        # total that cannot contain the timings it claims.
        if mode_total + len(phase_keys) < timing_total:
            raise ContractError(
                f"{mode_label}.total_ms is shorter than its recorded phase timings"
            )
        if require_wav:
            wav = _require_exact_mapping(
                mode["wav"],
                (
                    "bits_per_sample",
                    "byte_length",
                    "channels",
                    "frame_count",
                    "sample_rate",
                    "sha256",
                ),
                f"{mode_label}.wav",
            )
            wav_sha256 = wav["sha256"]
            if not isinstance(wav_sha256, str):
                raise ContractError(f"{mode_label}.wav.sha256 must be a string")
            require_sha256(wav_sha256, f"{mode_label}.wav.sha256")
            byte_length = _require_positive_int(
                wav["byte_length"], f"{mode_label}.wav.byte_length"
            )
            frame_count = _require_positive_int(
                wav["frame_count"], f"{mode_label}.wav.frame_count"
            )
            channels = _require_positive_int(
                wav["channels"], f"{mode_label}.wav.channels"
            )
            bits_per_sample = _require_positive_int(
                wav["bits_per_sample"], f"{mode_label}.wav.bits_per_sample"
            )
            _require_positive_int(
                wav["sample_rate"], f"{mode_label}.wav.sample_rate"
            )
            # The codec-frame cap is a CLI flag, so the generated count is
            # recorded as auditable evidence of how much synthesis actually ran
            # rather than being compared against a guessed floor.
            _require_positive_int(
                mode["frames_generated"], f"{mode_label}.frames_generated"
            )
            if not isinstance(mode["truncated"], bool):
                raise ContractError(f"{mode_label}.truncated must be a boolean")
            for measurement in ("peak", "rms"):
                value = mode[measurement]
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ContractError(
                        f"{mode_label}.{measurement} must be a positive finite number"
                    )
            if mode["peak"] <= 0.001 or mode["rms"] <= 0.0001:
                raise ContractError(
                    f"{mode_label} waveform evidence is below the required peak/RMS floor"
                )
            if (
                wav["sample_rate"] != TTS_WAV_SAMPLE_RATE
                or channels != TTS_WAV_CHANNELS
                or bits_per_sample != TTS_WAV_BITS_PER_SAMPLE
            ):
                raise ContractError(
                    f"{mode_label}.wav must be PCM16 mono {TTS_WAV_SAMPLE_RATE} Hz"
                )
            minimum_pcm_bytes = frame_count * channels * (bits_per_sample // 8)
            if byte_length < minimum_pcm_bytes:
                raise ContractError(
                    f"{mode_label}.wav.byte_length is smaller than its PCM frame data"
                )
        else:
            for transcript_field in (
                "cold_transcript",
                "warm_transcript",
                "cancellation_result",
            ):
                transcript = mode[transcript_field]
                if not isinstance(transcript, str) or not transcript:
                    raise ContractError(
                        f"{mode_label}.{transcript_field} must be a non-empty string"
                    )
            if mode["silence_transcript"] != "":
                raise ContractError(
                    f"{mode_label}.silence_transcript must stay empty"
                )
        pair = (memory_mode, runtime_mode)
        if pair in seen:
            raise ContractError(f"{label} repeats mode {memory_mode}/{runtime_mode}")
        seen.append(pair)
    if sorted(seen) != sorted(required_modes):
        raise ContractError(
            f"{label} must cover exactly {sorted(required_modes)}, covered {sorted(seen)}"
        )
    computed_total = sum(mode["total_ms"] for mode in modes)
    if phase_total != computed_total:
        raise ContractError(
            f"{label}.total_ms must equal the sum of mode totals: "
            f"{phase_total} != {computed_total}"
        )


def verify_attestation(
    *,
    attestation: Mapping[str, Any],
    candidate_dir: Path | None = None,
    candidate_fingerprint: str | None = None,
    candidate_run_id: str | None = None,
    bridge_source_sha: str | None = None,
    upstream_tag: str | None = None,
    upstream_commit: str | None = None,
    native_release_tag: str | None = None,
    native_manifest_sha256: str | None = None,
    native_commit: str | None = None,
    emscripten_version: str | None = None,
    release_tag: str | None = None,
    release_rebuild: int | None = None,
    orchestrator_correlation_id: str | None = None,
    harness_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless the attestation exactly binds the candidate and identities."""
    _require_exact_mapping(attestation, ATTESTATION_KEYS, "attestation")

    if attestation["schema_version"] != QUALIFICATION_SCHEMA_VERSION:
        raise ContractError(
            f"unsupported attestation schema_version: {attestation['schema_version']!r}"
        )
    if attestation["attestation_type"] != ATTESTATION_TYPE:
        raise ContractError(
            f"attestation_type must be {ATTESTATION_TYPE!r}, got "
            f"{attestation['attestation_type']!r}"
        )
    if attestation["harness_version"] != HARNESS_VERSION:
        raise ContractError(
            f"harness_version must be {HARNESS_VERSION!r}, got "
            f"{attestation['harness_version']!r}"
        )
    if attestation["bridge_repository"] != BRIDGE_REPOSITORY:
        raise ContractError(f"bridge_repository must be exactly {BRIDGE_REPOSITORY!r}")
    if attestation["native_repository"] != NATIVE_REPOSITORY:
        raise ContractError(f"native_repository must be exactly {NATIVE_REPOSITORY!r}")
    if attestation["upstream_repository"] != "ggml-org/llama.cpp":
        raise ContractError("upstream_repository must be exactly 'ggml-org/llama.cpp'")

    fingerprint = _require_str(attestation, "candidate_fingerprint", "attestation")
    require_sha256(fingerprint, "candidate_fingerprint")
    require_sha256(
        _require_str(attestation, "harness_source_sha256", "attestation"),
        "harness_source_sha256",
    )
    require_sha256(
        _require_str(attestation, "native_manifest_sha256", "attestation"),
        "native_manifest_sha256",
    )
    for field in ("bridge_source_sha", "upstream_commit", "native_commit"):
        value = _require_str(attestation, field, "attestation")
        if _COMMIT_RE.fullmatch(value) is None:
            raise ContractError(f"{field} must be a lowercase 40-hex commit SHA")
    run_id = _require_str(attestation, "candidate_run_id", "attestation")
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise ContractError("candidate_run_id must be a positive integer")
    expected_run_url = (
        f"https://github.com/{BRIDGE_REPOSITORY}/actions/runs/{run_id}"
    )
    if attestation["candidate_run_url"] != expected_run_url:
        raise ContractError(f"candidate_run_url must be exactly {expected_run_url}")
    _require_str(attestation, "emscripten_version", "attestation")
    _require_str(attestation, "release_tag", "attestation")
    _require_int(attestation, "release_rebuild", "attestation")
    require_correlation_id(
        _require_str(attestation, "orchestrator_correlation_id", "attestation")
    )

    hosted = _require_exact_mapping(
        attestation["hosted_gates"], HOSTED_GATES, "hosted_gates"
    )
    local = _require_exact_mapping(
        attestation["local_gates"], LOCAL_GATES, "local_gates"
    )
    for name, conclusion in (*hosted.items(), *local.items()):
        if conclusion != "passed":
            raise ContractError(
                f"qualification gate {name!r} is {conclusion!r}; must be 'passed'"
            )

    unproven = _require_exact_mapping(
        attestation["unproven_capabilities"],
        REQUIRED_UNPROVEN_CAPABILITIES,
        "unproven_capabilities",
    )
    for key, expected in REQUIRED_UNPROVEN_CAPABILITIES.items():
        if unproven[key] != expected:
            raise ContractError(
                f"unproven capability {key!r} must stay {expected!r}, got "
                f"{unproven[key]!r}"
            )

    pins = _require_exact_mapping(
        attestation["model_pins"], EXPECTED_MODEL_PINS, "model_pins"
    )
    for name, expected in EXPECTED_MODEL_PINS.items():
        if pins[name] != expected:
            raise ContractError(
                f"model pin {name!r} mismatch: expected {expected}, got {pins[name]!r}"
            )

    phases = _require_exact_mapping(attestation["phases"], LOCAL_GATES, "phases")
    _validate_phase(
        phases["speech_to_text"],
        label="phases.speech_to_text",
        required_modes=REQUIRED_SPEECH_MODES,
        phase_keys=SPEECH_PHASE_KEYS,
        require_wav=False,
    )
    _validate_phase(
        phases["text_to_speech"],
        label="phases.text_to_speech",
        required_modes=REQUIRED_TTS_MODES,
        phase_keys=TTS_PHASE_KEYS,
        require_wav=True,
    )

    if candidate_dir is not None:
        manifest, computed = load_candidate(candidate_dir)
        if computed != fingerprint:
            raise ContractError(
                f"candidate fingerprint mismatch: attestation={fingerprint}, "
                f"candidate={computed}"
            )
        manifest_bindings = {
            "bridge_source_sha": manifest["bridge_commit"],
            "candidate_run_id": manifest["github_run_id"],
            "candidate_run_url": manifest["github_run_url"],
            "emscripten_version": manifest["emscripten_version"],
            "native_commit": manifest["native_commit"],
            "native_manifest_sha256": manifest["native_manifest_sha256"],
            "native_release_tag": manifest["native_release_tag"],
            "orchestrator_correlation_id": manifest["orchestrator_correlation_id"],
            "release_rebuild": manifest["release_rebuild"],
            "release_tag": manifest["release_tag"],
            "upstream_commit": manifest["upstream_commit"],
            "upstream_tag": manifest["upstream_tag"],
        }
        for field, expected in manifest_bindings.items():
            if attestation[field] != expected:
                raise ContractError(
                    f"attestation {field} does not match the candidate manifest: "
                    f"{attestation[field]!r} != {expected!r}"
                )
        if hosted != _manifest_hosted_gates(manifest):
            raise ContractError(
                "attestation hosted_gates do not match the candidate manifest"
            )

    expectations: dict[str, Any] = {
        "bridge_source_sha": bridge_source_sha,
        "candidate_fingerprint": candidate_fingerprint,
        "candidate_run_id": candidate_run_id,
        "emscripten_version": emscripten_version,
        "harness_source_sha256": harness_sha256,
        "native_commit": native_commit,
        "native_manifest_sha256": native_manifest_sha256,
        "native_release_tag": native_release_tag,
        "orchestrator_correlation_id": orchestrator_correlation_id,
        "release_rebuild": release_rebuild,
        "release_tag": release_tag,
        "upstream_commit": upstream_commit,
        "upstream_tag": upstream_tag,
    }
    for field, expected in expectations.items():
        if expected is None:
            continue
        if attestation[field] != expected:
            raise ContractError(
                f"attestation {field} mismatch: expected {expected!r}, got "
                f"{attestation[field]!r}"
            )

    return {
        "verified": True,
        "candidate_fingerprint": fingerprint,
        "candidate_run_id": run_id,
        "release_tag": attestation["release_tag"],
        "bridge_source_sha": attestation["bridge_source_sha"],
        "harness_source_sha256": attestation["harness_source_sha256"],
    }


def max_rss_bytes() -> int:
    """Report peak child RSS in bytes on both reporting conventions.

    Linux reports ru_maxrss in kibibytes while macOS and the BSDs report bytes,
    so a raw value would be off by 1024x depending on where qualification ran.

    RUSAGE_CHILDREN is cumulative and cannot be reset, so a phase's recorded
    value is the peak across every harness child up to the end of that phase,
    not that phase's contribution in isolation.
    """
    raw = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform == "darwin":
        return int(raw)
    return int(raw) * 1024


def _child_env() -> dict[str, str]:
    """Strip ambient smoke configuration so no gate can be silently redirected.

    Every model, projector, fixture, URL, and timeout is passed explicitly on the
    command line. An inherited LLAMA_WEBGPU_* variable could otherwise point a
    gate at a different file or at the network without appearing in the record.
    """
    allowed = {
        "DISPLAY",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NO_COLOR",
        "PATH",
        "PLAYWRIGHT_BROWSERS_PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "WINDIR",
        "XDG_CACHE_HOME",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}


def _stop_smoke_process(
    proc: subprocess.Popen[str], *, grace_seconds: float = 5.0
) -> tuple[str, str]:
    """Terminate an entire smoke process group and drain its diagnostic pipes."""
    if proc.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM)
            else:  # pragma: no cover - local qualification targets POSIX today
                proc.terminate()
        except ProcessLookupError:
            pass
    try:
        return proc.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:  # pragma: no cover - local qualification targets POSIX today
                proc.kill()
        except ProcessLookupError:
            pass
        return proc.communicate()


def _write_smoke_diagnostics(
    diagnostics_dir: Path, label: str, stdout: str, stderr: str
) -> tuple[str, str]:
    sanitized_stdout = sanitize_diagnostic_text(stdout)
    sanitized_stderr = sanitize_diagnostic_text(stderr)
    (diagnostics_dir / f"{label}-stderr.log").write_text(
        sanitized_stderr, encoding="utf-8"
    )
    (diagnostics_dir / f"{label}-stdout.json").write_text(
        sanitized_stdout, encoding="utf-8"
    )
    return sanitized_stdout, sanitized_stderr


def _run_smoke(
    command: list[str],
    label: str,
    diagnostics_dir: Path,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ContractError(f"{label} timeout must be positive")
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(),
        start_new_session=os.name == "posix",
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        stdout, stderr = _stop_smoke_process(proc)
    except KeyboardInterrupt:
        stdout, stderr = _stop_smoke_process(proc)
        _write_smoke_diagnostics(diagnostics_dir, label, stdout, stderr)
        raise
    stdout, stderr = _write_smoke_diagnostics(
        diagnostics_dir, label, stdout, stderr
    )
    if timed_out:
        sys.stderr.write(stderr)
        raise ContractError(f"{label} gate timed out after {timeout_seconds:g} seconds")
    if proc.returncode != 0:
        sys.stderr.write(stderr)
        raise ContractError(f"{label} gate failed with exit status {proc.returncode}")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} gate emitted unparsable JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise ContractError(f"{label} gate did not report ok=true")
    return payload


def _mode_key(entry: Mapping[str, Any], label: str) -> tuple[str, str]:
    memory_mode = entry.get("memoryMode")
    runtime_mode = entry.get("runtimeMode")
    if not isinstance(memory_mode, str) or not isinstance(runtime_mode, str):
        raise ContractError(f"{label} result is missing its mode identifiers")
    return memory_mode, runtime_mode


def _timing(entry: Mapping[str, Any], key: str, label: str) -> int:
    value = entry.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{label} is missing a non-negative {key} timing")
    return value


def _speech_phase(payload: Mapping[str, Any], rss: int) -> dict[str, Any]:
    results = payload.get("modeResults")
    if not isinstance(results, list):
        raise ContractError("speech gate did not report modeResults")
    modes: list[dict[str, Any]] = []
    for entry in results:
        if not isinstance(entry, Mapping):
            raise ContractError("speech gate reported an invalid mode result")
        memory_mode, runtime_mode = _mode_key(entry, "speech")
        label = f"speech {memory_mode}/{runtime_mode}"
        timings = entry.get("phaseTimingsMs")
        if not isinstance(timings, Mapping):
            raise ContractError(f"{label} did not report per-phase timings")
        cold_transcript = entry.get("coldTranscript")
        warm_transcript = entry.get("warmTranscript")
        cancellation_result = entry.get("cancellation")
        silence_transcript = entry.get("silenceTranscript")
        for field, value in (
            ("coldTranscript", cold_transcript),
            ("warmTranscript", warm_transcript),
            ("cancellation", cancellation_result),
        ):
            if not isinstance(value, str) or not value:
                raise ContractError(f"{label} did not report {field} evidence")
        if silence_transcript != "":
            raise ContractError(f"{label} silence transcript must be empty")
        modes.append(
            {
                "memory_mode": memory_mode,
                "runtime_mode": runtime_mode,
                "total_ms": _timing(entry, "elapsedMs", label),
                "phase_timings_ms": {
                    "cancellation": _timing(timings, "cancellationMs", label),
                    "cold_transcript": _timing(timings, "coldTranscriptMs", label),
                    "model_load": _timing(timings, "modelLoadMs", label),
                    "projector_load": _timing(timings, "projectorLoadMs", label),
                    "silence": _timing(timings, "silenceMs", label),
                    "warm_transcript": _timing(timings, "warmTranscriptMs", label),
                },
                "cold_transcript": cold_transcript,
                "warm_transcript": warm_transcript,
                "cancellation_result": cancellation_result,
                "silence_transcript": silence_transcript,
            }
        )
    modes.sort(key=lambda mode: (mode["memory_mode"], mode["runtime_mode"]))
    return {
        "modes": modes,
        "total_ms": sum(mode["total_ms"] for mode in modes),
        "max_rss_bytes": rss,
    }


def read_wav_identity(path: Path) -> dict[str, Any]:
    """Verify a generated WAV container and return its measured identity."""
    data = path.read_bytes()
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            compression = handle.getcomptype()
            pcm_bytes = handle.readframes(frame_count)
    except (wave.Error, EOFError) as exc:
        raise ContractError(f"generated audio is not a readable WAV: {exc}") from exc
    if channels != TTS_WAV_CHANNELS or sample_width * 8 != TTS_WAV_BITS_PER_SAMPLE:
        raise ContractError("generated audio must be mono PCM16")
    if compression != "NONE":
        raise ContractError("generated audio must contain uncompressed PCM")
    if sample_rate != TTS_WAV_SAMPLE_RATE:
        raise ContractError(
            f"generated audio sample rate must be {TTS_WAV_SAMPLE_RATE}, got {sample_rate}"
        )
    if frame_count <= 0:
        raise ContractError("generated audio contains no frames")
    expected_pcm_bytes = frame_count * channels * sample_width
    if len(pcm_bytes) != expected_pcm_bytes:
        raise ContractError("generated audio PCM payload is truncated")
    samples = struct.unpack(f"<{frame_count * channels}h", pcm_bytes)
    peak = max(abs(sample) / 32768.0 for sample in samples)
    rms = math.sqrt(
        sum((sample / 32768.0) ** 2 for sample in samples) / len(samples)
    )
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_length": len(data),
        "channels": channels,
        "bits_per_sample": sample_width * 8,
        "sample_rate": sample_rate,
        "frame_count": frame_count,
        "peak": peak,
        "rms": rms,
    }


def _tts_phase(
    payload: Mapping[str, Any], rss: int, artifacts_dir: Path
) -> dict[str, Any]:
    results = payload.get("modeResults")
    if not isinstance(results, list):
        raise ContractError("text-to-speech gate did not report modeResults")
    modes: list[dict[str, Any]] = []
    for entry in results:
        if not isinstance(entry, Mapping):
            raise ContractError("text-to-speech gate reported an invalid mode result")
        memory_mode, runtime_mode = _mode_key(entry, "text-to-speech")
        label = f"text-to-speech {memory_mode}/{runtime_mode}"
        audio_name = entry.get("audioArtifact")
        if not isinstance(audio_name, str) or not audio_name:
            raise ContractError(f"{label} did not persist a generated WAV artifact")
        if Path(audio_name).name != audio_name or not audio_name.endswith(".wav"):
            raise ContractError(f"{label} WAV artifact name is unsafe: {audio_name!r}")
        artifacts_root = artifacts_dir.resolve()
        wav_path = artifacts_root / audio_name
        if wav_path.is_symlink() or not wav_path.is_file():
            raise ContractError(f"{label} WAV artifact is missing: {audio_name}")
        frames_generated = entry.get("framesGenerated")
        if (
            not isinstance(frames_generated, int)
            or isinstance(frames_generated, bool)
            or frames_generated <= 0
        ):
            raise ContractError(f"{label} framesGenerated must be a positive integer")
        truncated = entry.get("truncated")
        if not isinstance(truncated, bool):
            raise ContractError(f"{label} truncated must be a boolean")
        peak = entry.get("peak")
        rms = entry.get("rms")
        for field, value in (("peak", peak), ("rms", rms)):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ContractError(f"{label} {field} must be a positive finite number")
        wav_identity = read_wav_identity(wav_path)
        measured_peak = wav_identity.pop("peak")
        measured_rms = wav_identity.pop("rms")
        quantization_tolerance = (1 / 32768.0) + 1e-9
        if (
            abs(measured_peak - peak) > quantization_tolerance
            or abs(measured_rms - rms) > quantization_tolerance
        ):
            raise ContractError(
                f"{label} reported waveform evidence does not match its WAV artifact"
            )
        if measured_peak <= 0.001 or measured_rms <= 0.0001:
            raise ContractError(f"{label} generated WAV is effectively silent")
        modes.append(
            {
                "memory_mode": memory_mode,
                "runtime_mode": runtime_mode,
                "total_ms": _timing(entry, "totalElapsedMs", label),
                "phase_timings_ms": {
                    "model_load": _timing(entry, "modelLoadMs", label),
                    "projector_load": _timing(entry, "projectorLoadMs", label),
                    "synthesis": _timing(entry, "synthesisMs", label),
                },
                "frames_generated": frames_generated,
                "peak": measured_peak,
                "rms": measured_rms,
                "truncated": truncated,
                "wav": wav_identity,
            }
        )
    modes.sort(key=lambda mode: (mode["memory_mode"], mode["runtime_mode"]))
    return {
        "modes": modes,
        "total_ms": sum(mode["total_ms"] for mode in modes),
        "max_rss_bytes": rss,
    }


def _gh_json(args: list[str]) -> Any:
    try:
        proc = subprocess.run(
            ["gh", *args], check=False, capture_output=True, text=True
        )
    except OSError as exc:
        raise ContractError(f"could not run the gh CLI: {exc}") from exc
    if proc.returncode != 0:
        raise ContractError(
            f"gh {' '.join(args)} failed: {sanitize_diagnostic_text(proc.stderr.strip())}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError(f"gh {' '.join(args)} emitted unparsable JSON: {exc}") from exc


def _extract_flat_artifact_archive(archive_path: Path, destination: Path) -> None:
    """Extract a GitHub artifact only when every member is a unique flat file."""
    if destination.is_symlink():
        raise ContractError(f"artifact destination must not be a symlink: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        raise ContractError(f"artifact destination is not a directory: {destination}")
    if any(destination.iterdir()):
        raise ContractError(f"artifact destination is not empty: {destination}")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            seen: set[str] = set()
            members = archive.infolist()
            if not members:
                raise ContractError("artifact archive is empty")
            for member in members:
                relative = PurePosixPath(member.filename)
                mode_type = stat.S_IFMT(member.external_attr >> 16)
                if (
                    member.is_dir()
                    or len(relative.parts) != 1
                    or relative.name in ("", ".", "..")
                    or "\\" in relative.name
                    or mode_type not in (0, stat.S_IFREG)
                ):
                    raise ContractError(
                        f"artifact archive member is not a flat regular file: "
                        f"{member.filename!r}"
                    )
                if relative.name in seen:
                    raise ContractError(
                        f"artifact archive repeats member: {relative.name!r}"
                    )
                seen.add(relative.name)
                (destination / relative.name).write_bytes(archive.read(member))
    except (OSError, zipfile.BadZipFile) as exc:
        raise ContractError(f"could not read artifact archive: {exc}") from exc


def _download_artifact(artifact_id: int, destination: Path) -> None:
    if not isinstance(artifact_id, int) or isinstance(artifact_id, bool):
        raise ContractError("artifact id must be an integer")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, archive_name = tempfile.mkstemp(
        prefix=f"github-artifact-{artifact_id}-",
        suffix=".zip",
        dir=destination.parent,
    )
    os.close(descriptor)
    archive_path = Path(archive_name)
    try:
        with archive_path.open("wb") as output:
            proc = subprocess.run(
                [
                    "gh",
                    "api",
                    "-H",
                    "Accept: application/vnd.github+json",
                    f"repos/{BRIDGE_REPOSITORY}/actions/artifacts/{artifact_id}/zip",
                ],
                check=False,
                stdout=output,
                stderr=subprocess.PIPE,
                text=False,
            )
        if proc.returncode != 0:
            diagnostic = proc.stderr.decode("utf-8", errors="replace")
            raise ContractError(
                "could not download exact candidate artifact: "
                f"{sanitize_diagnostic_text(diagnostic.strip())}"
            )
        _extract_flat_artifact_archive(archive_path, destination)
    finally:
        archive_path.unlink(missing_ok=True)


def fetch_candidate(run_id: str, destination: Path) -> None:
    """Prove the candidate run's identity, then download its unique artifact."""
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise ContractError("candidate run id must be a positive integer")
    repository = _gh_json(["api", f"repos/{BRIDGE_REPOSITORY}"])
    branch = repository.get("default_branch") if isinstance(repository, Mapping) else None
    if not isinstance(branch, str) or not branch:
        raise ContractError("could not resolve the bridge default branch")
    run = _gh_json(["api", f"repos/{BRIDGE_REPOSITORY}/actions/runs/{run_id}"])
    head_sha = validate_workflow_run(
        run,
        expected_run_id=run_id,
        expected_workflow_path=CANDIDATE_WORKFLOW_PATH,
        expected_head_branch=branch,
        expected_run_attempt=1,
    )
    comparison = _gh_json(
        ["api", f"repos/{BRIDGE_REPOSITORY}/compare/{head_sha}...{branch}"]
    )
    status = comparison.get("status") if isinstance(comparison, Mapping) else None
    if status not in ("ahead", "identical"):
        raise ContractError(
            f"candidate run head {head_sha} is not reachable from {branch}"
        )
    inventory = _gh_json(
        [
            "api",
            f"repos/{BRIDGE_REPOSITORY}/actions/runs/{run_id}/artifacts?per_page=100",
        ]
    )
    artifact_id = validate_artifact_inventory(
        inventory, expected_run_id=run_id, expected_name=CANDIDATE_ARTIFACT_NAME
    )
    _download_artifact(artifact_id, destination)


def _require_input_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ContractError(f"{label} does not exist: {resolved}")
    return resolved


def qualify_cmd(args: argparse.Namespace) -> int:
    scripts_dir = Path(__file__).resolve().parent
    if args.tts_max_frames <= 0:
        raise ContractError("tts_max_frames must be positive")
    if args.speech_timeout_seconds <= 0 or args.tts_timeout_seconds <= 0:
        raise ContractError("qualification gate timeouts must be positive")
    speech_model = _require_input_file(args.speech_model_path, "Qwen3-ASR model")
    speech_mmproj = _require_input_file(args.speech_mmproj_path, "Qwen3-ASR projector")
    speech_audio = _require_input_file(args.speech_audio_path, "speech WAV fixture")
    tts_model = _require_input_file(args.tts_model_path, "Qwen3-TTS model")
    tts_mmproj = _require_input_file(args.tts_mmproj_path, "Qwen3-TTS projector")

    # Every scratch path lives in the system temp root so no download, artifact,
    # or diagnostic is ever written inside the repository working tree.
    with tempfile.TemporaryDirectory(prefix="llama-web-bridge-qualification-") as scratch:
        root = Path(scratch)
        candidate_dir = root / "candidate"
        # Diagnostics outlive the scratch root so a failed gate stays
        # inspectable, and they live outside the repository working tree.
        diagnostics_dir = (
            args.diagnostics_dir.expanduser().resolve()
            if args.diagnostics_dir
            else Path(tempfile.mkdtemp(prefix="llama-web-bridge-qualification-diag-"))
        )
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        print(f"Sanitized diagnostics directory {diagnostics_dir}", file=sys.stderr)
        speech_artifacts = diagnostics_dir / "speech-to-text"
        tts_artifacts = diagnostics_dir / "text-to-speech"

        print(
            f"Downloading candidate artifact from run {args.candidate_run_id}",
            file=sys.stderr,
        )
        fetch_candidate(args.candidate_run_id, candidate_dir)
        manifest, fingerprint = load_candidate(candidate_dir)
        harness_digest = require_harness_matches_bridge_source(
            scripts_dir,
            _require_str(manifest, "bridge_commit", "candidate manifest"),
        )
        print(f"Candidate fingerprint {fingerprint}", file=sys.stderr)

        print("Running Qwen3-ASR wasm32+wasm64 direct+worker gate", file=sys.stderr)
        speech_payload = _run_smoke(
            [
                sys.executable,
                str(scripts_dir / "speech_to_text_browser_smoke.py"),
                "--dist-dir", str(candidate_dir),
                "--model-path", str(speech_model),
                "--model-sha256", SPEECH_MODEL_SHA256,
                "--mmproj-path", str(speech_mmproj),
                "--mmproj-sha256", SPEECH_MMPROJ_SHA256,
                "--audio-path", str(speech_audio),
                "--audio-sha256", SPEECH_AUDIO_SHA256,
                "--memory-mode", "all",
                "--timeout-ms", str(args.speech_timeout_seconds * 1000),
                "--artifacts-dir", str(speech_artifacts),
            ],
            "speech-to-text",
            diagnostics_dir,
            timeout_seconds=args.speech_timeout_seconds + 60,
        )
        speech_phase = _speech_phase(speech_payload, max_rss_bytes())

        print("Running Qwen3-TTS wasm64 direct+worker gate", file=sys.stderr)
        tts_payload = _run_smoke(
            [
                sys.executable,
                str(scripts_dir / "text_to_speech_browser_smoke.py"),
                "--dist-dir", str(candidate_dir),
                "--model-path", str(tts_model),
                "--model-sha256", TTS_MODEL_SHA256,
                "--mmproj-path", str(tts_mmproj),
                "--mmproj-sha256", TTS_MMPROJ_SHA256,
                "--memory-mode", "wasm64",
                "--runtime-mode", "all",
                "--max-frames", str(args.tts_max_frames),
                "--timeout-ms", str(args.tts_timeout_seconds * 1000),
                "--artifacts-dir", str(tts_artifacts),
            ],
            "text-to-speech",
            diagnostics_dir,
            timeout_seconds=args.tts_timeout_seconds + 60,
        )
        tts_phase = _tts_phase(tts_payload, max_rss_bytes(), tts_artifacts)

        attestation = build_attestation(
            manifest=manifest,
            candidate_fingerprint=fingerprint,
            candidate_run_id=args.candidate_run_id,
            harness_digest=harness_digest,
            speech_phase=speech_phase,
            tts_phase=tts_phase,
        )
        # Re-verify what was just built against the exact candidate so a harness
        # bug can never emit an attestation publication would later reject.
        verify_attestation(attestation=attestation, candidate_dir=candidate_dir)
        canonical = canonical_json(attestation)

    args.output_attestation.write_text(canonical, encoding="utf-8")
    args.output_base64.write_text(encode_attestation(canonical) + "\n", encoding="utf-8")
    sys.stdout.write(canonical)
    print(
        f"Canonical attestation written to {args.output_attestation}\n"
        f"Dispatch payload written to {args.output_base64}",
        file=sys.stderr,
    )
    return 0


def decode_attestation_cmd(args: argparse.Namespace) -> int:
    _, canonical = decode_attestation(args.input.read_text(encoding="utf-8"))
    args.output.write_text(canonical, encoding="utf-8")
    return 0


def verify_run_cmd(args: argparse.Namespace) -> int:
    run = parse_attestation_json(args.run_json.read_text(encoding="utf-8"))
    head_sha = validate_workflow_run(
        run,
        expected_run_id=args.run_id,
        expected_workflow_path=args.workflow_path,
        expected_head_branch=args.head_branch,
        expected_run_attempt=args.run_attempt,
    )
    inventory = parse_attestation_json(args.artifacts_json.read_text(encoding="utf-8"))
    artifact_id = validate_artifact_inventory(
        inventory, expected_run_id=args.run_id, expected_name=args.artifact_name
    )
    print(
        json.dumps(
            {"verified": True, "artifact_id": artifact_id, "head_sha": head_sha},
            sort_keys=True,
        )
    )
    return 0


def verify_attestation_cmd(args: argparse.Namespace) -> int:
    attestation = parse_attestation_json(args.attestation.read_text(encoding="utf-8"))
    result = verify_attestation(
        attestation=attestation,
        candidate_dir=args.candidate_dist.resolve() if args.candidate_dist else None,
        candidate_fingerprint=args.candidate_fingerprint,
        candidate_run_id=args.candidate_run_id,
        bridge_source_sha=args.bridge_commit,
        upstream_tag=args.upstream_tag,
        upstream_commit=args.upstream_commit,
        native_release_tag=args.native_release_tag,
        native_manifest_sha256=args.native_manifest_sha256,
        native_commit=args.native_commit,
        emscripten_version=args.emscripten_version,
        release_tag=args.release_tag,
        release_rebuild=args.release_rebuild,
        orchestrator_correlation_id=args.orchestrator_correlation_id,
        harness_sha256=(
            harness_source_sha256(args.harness_dir.resolve())
            if args.harness_dir
            else None
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def candidate_fingerprint_cmd(args: argparse.Namespace) -> int:
    _, fingerprint = load_candidate(args.candidate_dist.resolve())
    print(fingerprint)
    return 0


def harness_digest_cmd(args: argparse.Namespace) -> int:
    print(harness_source_sha256(args.harness_dir.resolve()))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    qualify = subparsers.add_parser(
        "qualify",
        help="Run the heavy local gates against an exact hosted candidate run.",
    )
    qualify.add_argument("--candidate-run-id", required=True)
    qualify.add_argument("--speech-model-path", required=True, type=Path)
    qualify.add_argument("--speech-mmproj-path", required=True, type=Path)
    qualify.add_argument("--speech-audio-path", required=True, type=Path)
    qualify.add_argument("--tts-model-path", required=True, type=Path)
    qualify.add_argument("--tts-mmproj-path", required=True, type=Path)
    qualify.add_argument("--tts-max-frames", type=int, default=24)
    qualify.add_argument("--speech-timeout-seconds", type=int, default=1800)
    qualify.add_argument("--tts-timeout-seconds", type=int, default=1800)
    qualify.add_argument("--diagnostics-dir", type=Path)
    qualify.add_argument("--output-attestation", required=True, type=Path)
    qualify.add_argument("--output-base64", required=True, type=Path)

    decode = subparsers.add_parser(
        "decode-attestation",
        help="Decode and canonicality-check a transported base64 attestation.",
    )
    decode.add_argument("--input", required=True, type=Path)
    decode.add_argument("--output", required=True, type=Path)

    verify_run = subparsers.add_parser(
        "verify-run",
        help="Validate a workflow run and its unique artifact from fetched JSON.",
    )
    verify_run.add_argument("--run-json", required=True, type=Path)
    verify_run.add_argument("--artifacts-json", required=True, type=Path)
    verify_run.add_argument("--run-id", required=True)
    verify_run.add_argument("--workflow-path", required=True)
    verify_run.add_argument("--head-branch", required=True)
    verify_run.add_argument("--artifact-name", required=True)
    verify_run.add_argument("--run-attempt", type=int)

    verify = subparsers.add_parser(
        "verify-attestation", help="Verify a canonical qualification attestation."
    )
    verify.add_argument("--attestation", required=True, type=Path)
    verify.add_argument("--candidate-dist", type=Path)
    verify.add_argument("--candidate-fingerprint")
    verify.add_argument("--candidate-run-id")
    verify.add_argument("--bridge-commit")
    verify.add_argument("--upstream-tag")
    verify.add_argument("--upstream-commit")
    verify.add_argument("--native-release-tag")
    verify.add_argument("--native-manifest-sha256")
    verify.add_argument("--native-commit")
    verify.add_argument("--emscripten-version")
    verify.add_argument("--release-tag")
    verify.add_argument("--release-rebuild", type=int)
    verify.add_argument("--orchestrator-correlation-id")
    verify.add_argument("--harness-dir", type=Path)

    fingerprint = subparsers.add_parser(
        "candidate-fingerprint", help="Validate a candidate and print its fingerprint."
    )
    fingerprint.add_argument("--candidate-dist", required=True, type=Path)

    harness = subparsers.add_parser(
        "harness-digest", help="Print the digest of the heavy-gate harness sources."
    )
    harness.add_argument("--harness-dir", required=True, type=Path)

    return parser


COMMANDS = {
    "qualify": qualify_cmd,
    "decode-attestation": decode_attestation_cmd,
    "verify-run": verify_run_cmd,
    "verify-attestation": verify_attestation_cmd,
    "candidate-fingerprint": candidate_fingerprint_cmd,
    "harness-digest": harness_digest_cmd,
}


def main() -> int:
    args = _build_parser().parse_args()
    try:
        return COMMANDS[args.command](args)
    except (ContractError, OSError) as exc:
        print(f"error: {sanitize_diagnostic_text(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
