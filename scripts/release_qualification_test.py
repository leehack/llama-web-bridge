#!/usr/bin/env python3
"""Fail-closed tests for digest-bound local qualification and attestation checks."""

from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import hashlib
import io
import json
import shutil
import struct
import tempfile
import unittest
from unittest import mock
import wave
import warnings
import zipfile
from pathlib import Path

import release_qualification as rq
from generate_release_manifest import ARTIFACTS, generate
from release_contract import BRIDGE_REPOSITORY, ContractError
from speech_to_text_browser_smoke import DEFAULT_EXPECTED_TEXT


BRIDGE_SHA = "565c8396597ea7c0fb4e8d5d966da8d884b156d8"
UPSTREAM_COMMIT = "bb4caa7540188872173c44d161602d9271386413"
NATIVE_MANIFEST_SHA = (
    "2e5d29d7f98f0d71e75d3fa63b7c55f3b2a7933247cc34ea2b1c5e053d142452"
)
NATIVE_COMMIT = "1" * 40
CANDIDATE_RUN_ID = "32919086955"
CANDIDATE_ARTIFACT_ID = 7
CANDIDATE_RUN_URL = (
    f"https://github.com/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}"
)
CORRELATION_ID = "llamadart-pin:run-123"
DEFAULT_HEAD_BRANCH = "main"
DEFAULT_HEAD_SHA = "a" * 40


# Pinned by the speech gate that produces it, not restated here, so this suite
# cannot pass against a transcript the real gate would reject.
EXPECTED_SPEECH_TRANSCRIPT_RAW = DEFAULT_EXPECTED_TEXT


def candidate_archive_members() -> dict[str, bytes]:
    """Every member a legitimate candidate archive must carry, and nothing else."""
    members = {name: b"data\n" for name in ARTIFACTS}
    members["manifest.json"] = b"{}\n"
    members["sha256sums.txt"] = b"sums\n"
    return members


def write_candidate(directory: Path, *, marker: bytes = b"candidate") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(ARTIFACTS):
        (directory / name).write_bytes(marker + f"-{index}-{name}".encode("utf-8"))
    generate(
        argparse.Namespace(
            out_dir=directory,
            release_tag="v0.2.0-1",
            release_rebuild=1,
            assets_repo="leehack/llama-web-bridge-assets",
            bridge_repo=BRIDGE_REPOSITORY,
            bridge_commit=BRIDGE_SHA,
            upstream_repo="ggml-org/llama.cpp",
            upstream_tag="v0.2.0",
            upstream_commit=UPSTREAM_COMMIT,
            native_repo="leehack/llamadart-native",
            native_release_tag="v0.2.0-1",
            native_manifest_sha256=NATIVE_MANIFEST_SHA,
            native_commit=NATIVE_COMMIT,
            emscripten_version="6.0.8",
            orchestrator_correlation_id=CORRELATION_ID,
            github_run_id=CANDIDATE_RUN_ID,
            github_run_url=CANDIDATE_RUN_URL,
        )
    )


def write_wav(path: Path, *, sample_rate: int = 24000, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack("<8h", *range(8)) * channels)


def speech_phase() -> dict[str, object]:
    return {
        "modes": [
            {
                "memory_mode": memory_mode,
                "runtime_mode": runtime_mode,
                "total_ms": 93000,
                "phase_timings_ms": {
                    "cancellation": 4000,
                    "cold_transcript": 30000,
                    "model_load": 20000,
                    "projector_load": 5000,
                    "silence": 4000,
                    "warm_transcript": 30000,
                },
                "cold_transcript": EXPECTED_SPEECH_TRANSCRIPT_RAW,
                "warm_transcript": EXPECTED_SPEECH_TRANSCRIPT_RAW,
                "cancellation_result": "cancel:resolved:12",
                "silence_transcript": "",
            }
            for memory_mode, runtime_mode in rq.REQUIRED_SPEECH_MODES
        ],
        "total_ms": 372000,
        "max_rss_bytes": 8737062912,
    }


def tts_phase() -> dict[str, object]:
    return {
        "modes": [
            {
                "memory_mode": memory_mode,
                "runtime_mode": runtime_mode,
                "total_ms": 99980,
                "phase_timings_ms": {
                    "model_load": 60000,
                    "projector_load": 10000,
                    "synthesis": 29980,
                },
                "frames_generated": 24,
                "peak": 0.5,
                "rms": 0.1,
                "truncated": False,
                "cancellation_tested": True,
                "pre_aborted_tested": True,
                "reuse_sample_count": 2400,
                "unload_tested": True,
                "wav": {
                    "sha256": hashlib.sha256(runtime_mode.encode()).hexdigest(),
                    "byte_length": 96044,
                    "channels": 1,
                    "bits_per_sample": 16,
                    "sample_rate": 24000,
                    "frame_count": 48000,
                },
            }
            for memory_mode, runtime_mode in rq.REQUIRED_TTS_MODES
        ],
        "total_ms": 199960,
        "max_rss_bytes": 10338385920,
    }


def workflow_run(**overrides: object) -> dict[str, object]:
    run = {
        "id": int(CANDIDATE_RUN_ID),
        "path": rq.CANDIDATE_WORKFLOW_PATH,
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
        "head_branch": DEFAULT_HEAD_BRANCH,
        "head_sha": DEFAULT_HEAD_SHA,
        "run_attempt": 1,
        "repository": {"full_name": BRIDGE_REPOSITORY},
        "head_repository": {"full_name": BRIDGE_REPOSITORY},
        "actor": {"login": "leehack"},
        "triggering_actor": {"login": "leehack"},
    }
    run.update(overrides)
    return run


def artifact_inventory(**overrides: object) -> dict[str, object]:
    inventory = {
        "total_count": 1,
        "artifacts": [
            {
                "id": 7,
                "name": rq.CANDIDATE_ARTIFACT_NAME,
                "expired": False,
                "workflow_run": {"id": int(CANDIDATE_RUN_ID)},
            }
        ],
    }
    inventory.update(overrides)
    return inventory


class QualificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="llama-web-bridge-qual-test-"))
        self.candidate = self.tmp / "candidate"
        write_candidate(self.candidate)
        self.manifest, self.fingerprint = rq.load_candidate(self.candidate)
        self.harness_digest = rq.harness_source_sha256(
            Path(__file__).resolve().parent
        )
        self.attestation = rq.build_attestation(
            manifest=self.manifest,
            candidate_fingerprint=self.fingerprint,
            candidate_run_id=CANDIDATE_RUN_ID,
            candidate_artifact_id=7,
            candidate_run_attempt=1,
            harness_digest=self.harness_digest,
            speech_phase=speech_phase(),
            tts_phase=tts_phase(),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def verify(self, attestation: object, **kwargs: object) -> object:
        return rq.verify_attestation(
            attestation=attestation,
            candidate_dir=kwargs.pop("candidate_dir", self.candidate),
            **kwargs,
        )

    # --- happy path -----------------------------------------------------

    def test_valid_attestation_binds_candidate_and_every_identity(self) -> None:
        result = self.verify(
            self.attestation,
            candidate_fingerprint=self.fingerprint,
            candidate_run_id=CANDIDATE_RUN_ID,
            bridge_source_sha=BRIDGE_SHA,
            upstream_tag="v0.2.0",
            upstream_commit=UPSTREAM_COMMIT,
            native_release_tag="v0.2.0-1",
            native_manifest_sha256=NATIVE_MANIFEST_SHA,
            native_commit=NATIVE_COMMIT,
            emscripten_version="6.0.8",
            release_tag="v0.2.0-1",
            release_rebuild=1,
            orchestrator_correlation_id=CORRELATION_ID,
            harness_sha256=self.harness_digest,
        )
        self.assertTrue(result["verified"])
        self.assertEqual(result["candidate_fingerprint"], self.fingerprint)
        self.assertEqual(result["candidate_run_id"], CANDIDATE_RUN_ID)

    def test_attestation_is_deterministic_and_canonical(self) -> None:
        again = rq.build_attestation(
            manifest=self.manifest,
            candidate_fingerprint=self.fingerprint,
            candidate_run_id=CANDIDATE_RUN_ID,
            candidate_artifact_id=CANDIDATE_ARTIFACT_ID,
            harness_digest=self.harness_digest,
            speech_phase=speech_phase(),
            tts_phase=tts_phase(),
        )
        text = rq.canonical_json(self.attestation)
        self.assertEqual(text, rq.canonical_json(again))
        self.assertEqual(text, rq.canonical_json(json.loads(text)))

    def test_attestation_leaves_playback_capabilities_unproven(self) -> None:
        self.assertEqual(
            self.attestation["unproven_capabilities"],
            {
                "real_device_intelligibility": "unproven",
                "real_device_playback": "unproven",
                "speaker_reference_fidelity": "unproven",
            },
        )

    def test_attestation_records_hosted_and_local_gates_separately(self) -> None:
        self.assertEqual(
            self.attestation["hosted_gates"],
            {"state_persistence": "passed", "multimodal": "passed"},
        )
        self.assertEqual(
            self.attestation["local_gates"],
            {"speech_to_text": "passed", "text_to_speech": "passed"},
        )

    # --- candidate identity ---------------------------------------------

    def test_candidate_manifest_must_not_claim_a_hosted_heavy_gate_pass(self) -> None:
        manifest_path = self.candidate / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["qualification_gates"]["speech_to_text"] = "passed"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with self.assertRaises(ContractError) as ctx:
            rq.build_attestation(
                manifest=manifest,
                candidate_fingerprint=self.fingerprint,
                candidate_run_id=CANDIDATE_RUN_ID,
                candidate_artifact_id=CANDIDATE_ARTIFACT_ID,
                harness_digest=self.harness_digest,
                speech_phase=speech_phase(),
                tts_phase=tts_phase(),
            )
        self.assertIn("hosted pass it never ran", str(ctx.exception))

    def test_candidate_directory_with_unexpected_file_rejected(self) -> None:
        (self.candidate / "extra.txt").write_text("stow", encoding="utf-8")
        with self.assertRaises(ContractError) as ctx:
            rq.load_candidate(self.candidate)
        self.assertIn("unexpected", str(ctx.exception))

    def test_candidate_directory_missing_file_rejected(self) -> None:
        (self.candidate / "sha256sums.txt").unlink()
        with self.assertRaises(ContractError) as ctx:
            rq.load_candidate(self.candidate)
        self.assertIn("missing", str(ctx.exception))

    def test_tampered_candidate_artifact_rejected(self) -> None:
        (self.candidate / "llama_webgpu_core.wasm").write_bytes(b"tampered")
        with self.assertRaises(ContractError):
            rq.load_candidate(self.candidate)

    def test_candidate_manifest_with_duplicate_keys_rejected(self) -> None:
        (self.candidate / "manifest.json").write_text(
            '{"schema_version": 2, "schema_version": 2}', encoding="utf-8"
        )
        with self.assertRaises(ContractError) as ctx:
            rq.load_candidate(self.candidate)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_candidate_manifest_with_wrong_field_type_rejected(self) -> None:
        manifest_path = self.candidate / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["release_rebuild"] = "1"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ContractError) as ctx:
            rq.load_candidate(self.candidate)
        self.assertIn("release_rebuild", str(ctx.exception))

    def test_attestation_for_a_different_candidate_rejected(self) -> None:
        other = self.tmp / "other-candidate"
        write_candidate(other, marker=b"rebuilt")
        with self.assertRaises(ContractError) as ctx:
            self.verify(self.attestation, candidate_dir=other)
        self.assertIn("fingerprint mismatch", str(ctx.exception))

    def test_publication_refuses_a_rebuilt_candidate_from_another_run(self) -> None:
        rebuilt = self.tmp / "rebuilt"
        rebuilt.mkdir()
        for name in ARTIFACTS:
            shutil.copyfile(self.candidate / name, rebuilt / name)
        generate(
            argparse.Namespace(
                out_dir=rebuilt,
                release_tag="v0.2.0-1",
                release_rebuild=1,
                assets_repo="leehack/llama-web-bridge-assets",
                bridge_repo=BRIDGE_REPOSITORY,
                bridge_commit=BRIDGE_SHA,
                upstream_repo="ggml-org/llama.cpp",
                upstream_tag="v0.2.0",
                upstream_commit=UPSTREAM_COMMIT,
                native_repo="leehack/llamadart-native",
                native_release_tag="v0.2.0-1",
                native_manifest_sha256=NATIVE_MANIFEST_SHA,
                native_commit=NATIVE_COMMIT,
                emscripten_version="6.0.8",
                orchestrator_correlation_id=CORRELATION_ID,
                github_run_id="99999999999",
                github_run_url=(
                    f"https://github.com/{BRIDGE_REPOSITORY}/actions/runs/99999999999"
                ),
            )
        )
        # Byte-identical binaries, different run provenance: the manifest differs,
        # so the fingerprint differs and the attestation cannot be reused.
        with self.assertRaises(ContractError) as ctx:
            self.verify(self.attestation, candidate_dir=rebuilt)
        self.assertIn("fingerprint mismatch", str(ctx.exception))

    def test_stale_attestation_replayed_against_new_release_rejected(self) -> None:
        newer = self.tmp / "newer"
        newer.mkdir()
        for name in ARTIFACTS:
            shutil.copyfile(self.candidate / name, newer / name)
        generate(
            argparse.Namespace(
                out_dir=newer,
                release_tag="v0.2.0-2",
                release_rebuild=2,
                assets_repo="leehack/llama-web-bridge-assets",
                bridge_repo=BRIDGE_REPOSITORY,
                bridge_commit=BRIDGE_SHA,
                upstream_repo="ggml-org/llama.cpp",
                upstream_tag="v0.2.0",
                upstream_commit=UPSTREAM_COMMIT,
                native_repo="leehack/llamadart-native",
                native_release_tag="v0.2.0-1",
                native_manifest_sha256=NATIVE_MANIFEST_SHA,
                native_commit=NATIVE_COMMIT,
                emscripten_version="6.0.8",
                orchestrator_correlation_id=CORRELATION_ID,
                github_run_id=CANDIDATE_RUN_ID,
                github_run_url=CANDIDATE_RUN_URL,
            )
        )
        with self.assertRaises(ContractError):
            self.verify(self.attestation, candidate_dir=newer)

    # --- attestation shape ----------------------------------------------

    def test_missing_required_field_rejected(self) -> None:
        for key in rq.ATTESTATION_KEYS:
            bad = copy.deepcopy(self.attestation)
            del bad[key]
            with self.assertRaises(ContractError) as ctx:
                self.verify(bad)
            self.assertIn(key, str(ctx.exception))

    def test_unexpected_field_rejected(self) -> None:
        bad = copy.deepcopy(self.attestation)
        bad["injected_token"] = "sneaky"
        with self.assertRaises(ContractError) as ctx:
            self.verify(bad)
        self.assertIn("injected_token", str(ctx.exception))

    def test_unexpected_nested_field_rejected(self) -> None:
        for path in (
            ("hosted_gates", "extra_gate"),
            ("local_gates", "extra_gate"),
            ("model_pins", "extra_pin"),
            ("unproven_capabilities", "extra_claim"),
            ("phases", "extra_phase"),
        ):
            bad = copy.deepcopy(self.attestation)
            bad[path[0]][path[1]] = "passed"
            with self.assertRaises(ContractError) as ctx:
                self.verify(bad)
            self.assertIn(path[1], str(ctx.exception))

    def test_unexpected_phase_mode_field_rejected(self) -> None:
        bad = copy.deepcopy(self.attestation)
        bad["phases"]["speech_to_text"]["modes"][0]["extra"] = 1
        with self.assertRaises(ContractError) as ctx:
            self.verify(bad)
        self.assertIn("extra", str(ctx.exception))

    def test_wrong_attestation_type_or_schema_rejected(self) -> None:
        for key, value in (
            ("schema_version", 2),
            ("schema_version", True),
            ("schema_version", 1.0),
            ("attestation_type", "something-else"),
            ("harness_version", "0.0.1"),
            ("bridge_repository", "attacker/bridge"),
            ("native_repository", "attacker/native"),
            ("upstream_repository", "attacker/llama.cpp"),
        ):
            bad = copy.deepcopy(self.attestation)
            bad[key] = value
            with self.assertRaises(ContractError):
                self.verify(bad)

    def test_malformed_types_rejected(self) -> None:
        for key, value in (
            ("candidate_fingerprint", 123),
            ("candidate_run_id", 32919086955),
            ("release_rebuild", "1"),
            ("release_rebuild", -1),
            ("release_rebuild", True),
            ("hosted_gates", ["state_persistence"]),
            ("model_pins", "none"),
            ("phases", []),
            ("unproven_capabilities", None),
        ):
            bad = copy.deepcopy(self.attestation)
            bad[key] = value
            with self.assertRaises(ContractError):
                self.verify(bad)

    def test_non_passed_gate_rejected(self) -> None:
        for group, gate in (
            ("hosted_gates", "state_persistence"),
            ("hosted_gates", "multimodal"),
            ("local_gates", "speech_to_text"),
            ("local_gates", "text_to_speech"),
        ):
            for status in ("failed", "skipped", "pending", "success"):
                bad = copy.deepcopy(self.attestation)
                bad[group][gate] = status
                with self.assertRaises(ContractError) as ctx:
                    self.verify(bad)
                self.assertIn(gate, str(ctx.exception))

    def test_missing_gate_rejected(self) -> None:
        for group, gate in (
            ("hosted_gates", "multimodal"),
            ("local_gates", "text_to_speech"),
        ):
            bad = copy.deepcopy(self.attestation)
            del bad[group][gate]
            with self.assertRaises(ContractError) as ctx:
                self.verify(bad)
            self.assertIn(gate, str(ctx.exception))

    def test_unproven_capability_claimed_as_proven_rejected(self) -> None:
        for key in rq.REQUIRED_UNPROVEN_CAPABILITIES:
            bad = copy.deepcopy(self.attestation)
            bad["unproven_capabilities"][key] = "proven"
            with self.assertRaises(ContractError) as ctx:
                self.verify(bad)
            self.assertIn(key, str(ctx.exception))

    def test_model_pin_mismatch_rejected(self) -> None:
        for pin in rq.EXPECTED_MODEL_PINS:
            bad = copy.deepcopy(self.attestation)
            bad["model_pins"][pin] = "0" * 64
            with self.assertRaises(ContractError) as ctx:
                self.verify(bad)
            self.assertIn(pin, str(ctx.exception))

    def test_harness_source_digest_mismatch_rejected(self) -> None:
        with self.assertRaises(ContractError) as ctx:
            self.verify(self.attestation, harness_sha256="0" * 64)
        self.assertIn("harness_source_sha256", str(ctx.exception))

    def test_candidate_run_url_must_match_run_id(self) -> None:
        bad = copy.deepcopy(self.attestation)
        bad["candidate_run_url"] = (
            f"https://github.com/{BRIDGE_REPOSITORY}/actions/runs/1"
        )
        with self.assertRaises(ContractError) as ctx:
            self.verify(bad)
        self.assertIn("candidate_run_url", str(ctx.exception))

    def test_identity_expectation_mismatch_rejected(self) -> None:
        for field, value in (
            ("candidate_fingerprint", "0" * 64),
            ("candidate_run_id", "1"),
            ("bridge_source_sha", "f" * 40),
            ("upstream_tag", "v0.2.1"),
            ("upstream_commit", "2" * 40),
            ("native_release_tag", "v0.2.0-2"),
            ("native_manifest_sha256", "f" * 64),
            ("native_commit", "3" * 40),
            ("emscripten_version", "6.0.7"),
            ("release_tag", "v0.2.0-2"),
            ("release_rebuild", 2),
            ("orchestrator_correlation_id", "other-correlation"),
        ):
            with self.assertRaises(ContractError) as ctx:
                self.verify(self.attestation, **{field: value})
            self.assertIn(field, str(ctx.exception))

    # --- phase coverage ---------------------------------------------------

    def test_missing_speech_memory_or_runtime_mode_rejected(self) -> None:
        for dropped in range(len(rq.REQUIRED_SPEECH_MODES)):
            bad = copy.deepcopy(self.attestation)
            del bad["phases"]["speech_to_text"]["modes"][dropped]
            with self.assertRaises(ContractError) as ctx:
                self.verify(bad)
            self.assertIn("must cover exactly", str(ctx.exception))

    def test_missing_tts_runtime_mode_rejected(self) -> None:
        for dropped in range(len(rq.REQUIRED_TTS_MODES)):
            bad = copy.deepcopy(self.attestation)
            del bad["phases"]["text_to_speech"]["modes"][dropped]
            with self.assertRaises(ContractError):
                self.verify(bad)

    def test_duplicate_mode_instead_of_coverage_rejected(self) -> None:
        bad = copy.deepcopy(self.attestation)
        bad["phases"]["speech_to_text"]["modes"][1] = copy.deepcopy(
            bad["phases"]["speech_to_text"]["modes"][0]
        )
        with self.assertRaises(ContractError) as ctx:
            self.verify(bad)
        self.assertIn("repeats mode", str(ctx.exception))

    def test_missing_speech_phase_timing_rejected(self) -> None:
        for key in rq.SPEECH_PHASE_KEYS:
            bad = copy.deepcopy(self.attestation)
            del bad["phases"]["speech_to_text"]["modes"][0]["phase_timings_ms"][key]
            with self.assertRaises(ContractError) as ctx:
                self.verify(bad)
            self.assertIn(key, str(ctx.exception))

    def test_missing_tts_phase_timing_rejected(self) -> None:
        for key in rq.TTS_PHASE_KEYS:
            bad = copy.deepcopy(self.attestation)
            del bad["phases"]["text_to_speech"]["modes"][0]["phase_timings_ms"][key]
            with self.assertRaises(ContractError) as ctx:
                self.verify(bad)
            self.assertIn(key, str(ctx.exception))

    def test_missing_or_invalid_resource_measurement_rejected(self) -> None:
        for value in (0, -1, "8737062912", None):
            bad = copy.deepcopy(self.attestation)
            bad["phases"]["speech_to_text"]["max_rss_bytes"] = value
            with self.assertRaises(ContractError) as ctx:
                self.verify(bad)
            self.assertIn("max_rss_bytes", str(ctx.exception))

    def test_phase_total_must_equal_the_sum_of_mode_totals(self) -> None:
        for gate in rq.LOCAL_GATES:
            bad = copy.deepcopy(self.attestation)
            bad["phases"][gate]["total_ms"] += 1
            with self.subTest(gate=gate):
                with self.assertRaises(ContractError) as ctx:
                    self.verify(bad)
                self.assertIn("total_ms", str(ctx.exception))

    def test_mode_total_cannot_be_shorter_than_its_recorded_phases(self) -> None:
        bad = copy.deepcopy(self.attestation)
        bad["phases"]["speech_to_text"]["modes"][0]["total_ms"] = 1
        with self.assertRaises(ContractError) as ctx:
            self.verify(bad)
        self.assertIn("phase timings", str(ctx.exception))

    def test_tts_wav_must_be_pcm16_mono_24khz(self) -> None:
        for field, value in (
            ("sample_rate", 16000),
            ("channels", 2),
            ("bits_per_sample", 8),
        ):
            bad = copy.deepcopy(self.attestation)
            bad["phases"]["text_to_speech"]["modes"][0]["wav"][field] = value
            with self.assertRaises(ContractError) as ctx:
                self.verify(bad)
            self.assertIn("PCM16 mono", str(ctx.exception))

    def test_tts_wav_must_contain_frames(self) -> None:
        for frames in (0, -1, "48000", None):
            bad = copy.deepcopy(self.attestation)
            bad["phases"]["text_to_speech"]["modes"][0]["wav"]["frame_count"] = frames
            with self.assertRaises(ContractError) as ctx:
                self.verify(bad)
            self.assertIn("frame_count", str(ctx.exception))

    def test_tts_waveform_evidence_must_be_finite_and_non_silent(self) -> None:
        for field, value in (
            ("peak", 0.001),
            ("rms", 0.0001),
            ("peak", float("inf")),
            ("rms", "0.1"),
        ):
            bad = copy.deepcopy(self.attestation)
            bad["phases"]["text_to_speech"]["modes"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(
                ContractError
            ):
                self.verify(bad)

    def test_speech_result_evidence_is_required(self) -> None:
        for field, value in (
            ("cold_transcript", ""),
            ("warm_transcript", None),
            ("cancellation_result", ""),
            ("silence_transcript", "hallucination"),
        ):
            bad = copy.deepcopy(self.attestation)
            bad["phases"]["speech_to_text"]["modes"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                self.verify(bad)

    def test_tts_synthesis_evidence_required(self) -> None:
        for field, value in (
            ("frames_generated", 0),
            ("frames_generated", "24"),
            ("truncated", "true"),
        ):
            bad = copy.deepcopy(self.attestation)
            bad["phases"]["text_to_speech"]["modes"][0][field] = value
            with self.assertRaises(ContractError) as ctx:
                self.verify(bad)
            self.assertIn(field, str(ctx.exception))
        missing = copy.deepcopy(self.attestation)
        del missing["phases"]["text_to_speech"]["modes"][0]["frames_generated"]
        with self.assertRaises(ContractError) as ctx:
            self.verify(missing)
        self.assertIn("frames_generated", str(ctx.exception))

    def test_speech_phase_must_not_carry_a_wav_claim(self) -> None:
        bad = copy.deepcopy(self.attestation)
        bad["phases"]["speech_to_text"]["modes"][0]["wav"] = copy.deepcopy(
            bad["phases"]["text_to_speech"]["modes"][0]["wav"]
        )
        with self.assertRaises(ContractError) as ctx:
            self.verify(bad)
        self.assertIn("wav", str(ctx.exception))

    # --- generated WAV inspection ----------------------------------------

    def test_generated_wav_identity_measured_from_the_file(self) -> None:
        wav_path = self.tmp / "audio" / "tts.wav"
        write_wav(wav_path)
        identity = rq.read_wav_identity(wav_path)
        self.assertEqual(identity["sample_rate"], 24000)
        self.assertEqual(identity["channels"], 1)
        self.assertEqual(identity["bits_per_sample"], 16)
        self.assertEqual(identity["frame_count"], 8)
        self.assertGreater(identity["peak"], 0)
        self.assertGreater(identity["rms"], 0)
        self.assertEqual(
            identity["sha256"], hashlib.sha256(wav_path.read_bytes()).hexdigest()
        )

    def test_invalid_generated_wav_rejected(self) -> None:
        broken = self.tmp / "broken.wav"
        broken.write_bytes(b"not a wav at all")
        with self.assertRaises(ContractError):
            rq.read_wav_identity(broken)
        stereo = self.tmp / "stereo.wav"
        write_wav(stereo, channels=2)
        with self.assertRaises(ContractError):
            rq.read_wav_identity(stereo)
        resampled = self.tmp / "resampled.wav"
        write_wav(resampled, sample_rate=16000)
        with self.assertRaises(ContractError):
            rq.read_wav_identity(resampled)

    def test_tts_phase_rejects_wav_path_escape_and_symlink(self) -> None:
        artifacts = self.tmp / "tts-artifacts"
        artifacts.mkdir()
        outside = self.tmp / "outside.wav"
        write_wav(outside)
        result = {
            "memoryMode": "wasm64",
            "runtimeMode": "direct",
            "totalElapsedMs": 10,
            "modelLoadMs": 3,
            "projectorLoadMs": 2,
            "synthesisMs": 5,
            "framesGenerated": 1,
            "peak": 0.5,
            "rms": 0.1,
            "truncated": False,
            "audioArtifact": "../outside.wav",
        }
        with self.assertRaises(ContractError):
            rq._tts_phase({"modeResults": [result]}, 1, artifacts)

        link = artifacts / "linked.wav"
        link.symlink_to(outside)
        result["audioArtifact"] = link.name
        with self.assertRaises(ContractError):
            rq._tts_phase({"modeResults": [result]}, 1, artifacts)

    def test_tts_phase_rejects_coerced_result_types(self) -> None:
        artifacts = self.tmp / "typed-tts-artifacts"
        wav_path = artifacts / "generated.wav"
        write_wav(wav_path)
        baseline = {
            "memoryMode": "wasm64",
            "runtimeMode": "direct",
            "totalElapsedMs": 10,
            "modelLoadMs": 3,
            "projectorLoadMs": 2,
            "synthesisMs": 5,
            "framesGenerated": 1,
            "peak": 0.5,
            "rms": 0.1,
            "truncated": False,
            "audioArtifact": wav_path.name,
        }
        for field, value in (("framesGenerated", 1.5), ("truncated", "false")):
            result = {**baseline, field: value}
            with self.subTest(field=field), self.assertRaises(ContractError):
                rq._tts_phase({"modeResults": [result]}, 1, artifacts)

    def test_tts_phase_measures_waveform_evidence_from_the_wav(self) -> None:
        artifacts = self.tmp / "waveform-tts-artifacts"
        wav_path = artifacts / "generated.wav"
        artifacts.mkdir()
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(struct.pack("<4h", 0, 8192, -8192, 4096))
        measured = rq.read_wav_identity(wav_path)
        result = {
            "memoryMode": "wasm64",
            "runtimeMode": "direct",
            "totalElapsedMs": 10,
            "modelLoadMs": 3,
            "projectorLoadMs": 2,
            "synthesisMs": 5,
            "framesGenerated": 1,
            "peak": measured["peak"],
            "rms": measured["rms"],
            "truncated": False,
            "cancellationTested": True,
            "preAbortedTested": True,
            "reuseSampleCount": 1,
            "unloadTested": True,
            "audioArtifact": wav_path.name,
        }
        phase = rq._tts_phase({"modeResults": [result]}, 1, artifacts)
        self.assertEqual(phase["modes"][0]["peak"], measured["peak"])
        result["peak"] = 0.9
        with self.assertRaises(ContractError):
            rq._tts_phase({"modeResults": [result]}, 1, artifacts)

    # --- transport --------------------------------------------------------

    def test_base64_transport_round_trip(self) -> None:
        canonical = rq.canonical_json(self.attestation)
        payload, decoded = rq.decode_attestation(rq.encode_attestation(canonical))
        self.assertEqual(decoded, canonical)
        self.assertEqual(payload, self.attestation)

    def test_transport_allows_one_terminal_newline_but_rejects_line_folding(self) -> None:
        blob = rq.encode_attestation(rq.canonical_json(self.attestation))
        folded = "\n".join(blob[index : index + 76] for index in range(0, len(blob), 76))
        self.assertEqual(
            rq.decode_attestation(blob + "\n")[1], rq.canonical_json(self.attestation)
        )
        with self.assertRaises(ContractError):
            rq.decode_attestation(folded)
        with self.assertRaises(ContractError):
            rq.decode_attestation(blob[:-8] + "!!!!!!!!")

    def test_noncanonical_transport_payload_rejected(self) -> None:
        compact = json.dumps(self.attestation, sort_keys=True)
        blob = base64.b64encode(compact.encode("utf-8")).decode("ascii")
        with self.assertRaises(ContractError) as ctx:
            rq.decode_attestation(blob)
        self.assertIn("canonical", str(ctx.exception))

    def test_noncanonical_base64_padding_bits_rejected(self) -> None:
        canonical = rq.canonical_json({"a": 1})
        blob = rq.encode_attestation(canonical)
        self.assertTrue(blob.endswith("g=="))
        with self.assertRaises(ContractError) as ctx:
            rq.decode_attestation(blob[:-3] + "h==")
        self.assertIn("base64", str(ctx.exception))

    def test_reordered_transport_payload_rejected(self) -> None:
        reordered = json.dumps(self.attestation, indent=2, sort_keys=False) + "\n"
        blob = base64.b64encode(reordered.encode("utf-8")).decode("ascii")
        with self.assertRaises(ContractError):
            rq.decode_attestation(blob)

    def test_duplicate_keys_in_transport_payload_rejected(self) -> None:
        blob = base64.b64encode(
            b'{\n  "schema_version": 1,\n  "schema_version": 1\n}\n'
        ).decode("ascii")
        with self.assertRaises(ContractError) as ctx:
            rq.decode_attestation(blob)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_nonstandard_json_constants_rejected_before_schema_validation(self) -> None:
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                with self.assertRaises(ContractError) as ctx:
                    rq.parse_attestation_json(f'{{"value": {constant}}}')
                self.assertIn("non-standard", str(ctx.exception))
        with self.assertRaises(ContractError):
            rq.canonical_json({"value": float("nan")})

    def test_non_object_transport_payload_rejected(self) -> None:
        for raw in (b"[1, 2, 3]\n", b'"text"\n', b"123\n", b"null\n", b"{ broken"):
            blob = base64.b64encode(raw).decode("ascii")
            with self.assertRaises(ContractError):
                rq.decode_attestation(blob)

    def test_empty_transport_payload_rejected(self) -> None:
        for blob in ("", "   \n  "):
            with self.assertRaises(ContractError):
                rq.decode_attestation(blob)

    def test_oversized_transport_payload_rejected(self) -> None:
        oversized = base64.b64encode(b"x" * (rq.MAX_ATTESTATION_BYTES + 1)).decode(
            "ascii"
        )
        with self.assertRaises(ContractError) as ctx:
            rq.decode_attestation(oversized)
        self.assertIn("bound", str(ctx.exception))
        with self.assertRaises(ContractError):
            rq.encode_attestation("y" * (rq.MAX_ATTESTATION_BYTES + 1))

    def test_shell_metacharacter_payload_is_never_valid(self) -> None:
        blob = base64.b64encode(b'"; rm -rf / #').decode("ascii")
        with self.assertRaises(ContractError):
            rq.decode_attestation(blob)
        with self.assertRaises(ContractError):
            rq.decode_attestation('$(whoami)')

    # --- workflow run provenance ------------------------------------------

    def test_valid_candidate_run_accepted(self) -> None:
        self.assertEqual(
            rq.validate_workflow_run(
                workflow_run(),
                expected_run_id=CANDIDATE_RUN_ID,
                expected_workflow_path=rq.CANDIDATE_WORKFLOW_PATH,
                expected_head_branch=DEFAULT_HEAD_BRANCH,
                expected_run_attempt=1,
            ),
            DEFAULT_HEAD_SHA,
        )
        self.assertEqual(
            rq.validate_artifact_inventory(
                artifact_inventory(),
                expected_run_id=CANDIDATE_RUN_ID,
                expected_name=rq.CANDIDATE_ARTIFACT_NAME,
            ),
            7,
        )

    def test_unrelated_or_unsuccessful_run_rejected(self) -> None:
        cases = (
            {"id": 12345},
            {"repository": {"full_name": "attacker/llama-web-bridge"}},
            {"head_repository": {"full_name": "attacker/fork"}},
            {"path": ".github/workflows/ci.yml"},
            {"event": "push"},
            {"event": "pull_request"},
            {"status": "in_progress"},
            {"conclusion": "failure"},
            {"conclusion": None},
            {"head_branch": "feature/attack"},
            {"head_branch": None},
            {"head_sha": "not-a-sha"},
            {"head_sha": None},
        )
        for overrides in cases:
            with self.assertRaises(ContractError):
                rq.validate_workflow_run(
                    workflow_run(**overrides),
                    expected_run_id=CANDIDATE_RUN_ID,
                    expected_workflow_path=rq.CANDIDATE_WORKFLOW_PATH,
                    expected_head_branch=DEFAULT_HEAD_BRANCH,
                )

    def test_candidate_run_must_be_the_first_build_attempt(self) -> None:
        for run_attempt in (2, 0, True, "1", None):
            with self.subTest(run_attempt=run_attempt), self.assertRaises(
                ContractError
            ):
                rq.validate_workflow_run(
                    workflow_run(run_attempt=run_attempt),
                    expected_run_id=CANDIDATE_RUN_ID,
                    expected_workflow_path=rq.CANDIDATE_WORKFLOW_PATH,
                    expected_head_branch=DEFAULT_HEAD_BRANCH,
                    expected_run_attempt=1,
                )

    def test_run_attempt_one_is_mandatory_even_without_a_caller_override(self) -> None:
        for run_attempt in (2, 0, True, "1", None):
            with self.subTest(run_attempt=run_attempt), self.assertRaises(
                ContractError
            ):
                rq.validate_workflow_run(
                    workflow_run(run_attempt=run_attempt),
                    expected_run_id=CANDIDATE_RUN_ID,
                    expected_workflow_path=rq.CANDIDATE_WORKFLOW_PATH,
                    expected_head_branch=DEFAULT_HEAD_BRANCH,
                )

    def test_attestation_run_must_be_the_ingestion_workflow(self) -> None:
        with self.assertRaises(ContractError) as ctx:
            rq.validate_workflow_run(
                workflow_run(path=rq.CANDIDATE_WORKFLOW_PATH),
                expected_run_id=CANDIDATE_RUN_ID,
                expected_workflow_path=rq.ATTESTATION_WORKFLOW_PATH,
                expected_head_branch=DEFAULT_HEAD_BRANCH,
            )
        self.assertIn(rq.ATTESTATION_WORKFLOW_PATH, str(ctx.exception))

    def test_run_id_and_head_sha_must_be_well_formed(self) -> None:
        for run_id in ("0", "-1", "01", "abc", ""):
            with self.assertRaises(ContractError):
                rq.validate_workflow_run(
                    workflow_run(),
                    expected_run_id=run_id,
                    expected_workflow_path=rq.CANDIDATE_WORKFLOW_PATH,
                    expected_head_branch=DEFAULT_HEAD_BRANCH,
                )
        with self.assertRaises(ContractError):
            rq.validate_workflow_run(
                workflow_run(),
                expected_run_id=CANDIDATE_RUN_ID,
                expected_workflow_path=rq.CANDIDATE_WORKFLOW_PATH,
                expected_head_branch="",
            )

    def test_missing_duplicate_expired_or_foreign_artifact_rejected(self) -> None:
        artifact = artifact_inventory()["artifacts"][0]
        cases = (
            {"artifacts": []},
            {"artifacts": [artifact, copy.deepcopy(artifact)]},
            {"artifacts": [{**artifact, "expired": True}]},
            {"artifacts": [{**artifact, "workflow_run": {"id": 1}}]},
            {"artifacts": [{**artifact, "id": "7"}]},
            {"artifacts": [{**artifact, "id": True}]},
            {"artifacts": [{**artifact, "id": 0}]},
            {"artifacts": [{"name": 5}]},
            {"artifacts": "none"},
        )
        for overrides in cases:
            with self.assertRaises(ContractError):
                rq.validate_artifact_inventory(
                    artifact_inventory(**overrides),
                    expected_run_id=CANDIDATE_RUN_ID,
                    expected_name=rq.CANDIDATE_ARTIFACT_NAME,
                )

    def test_truncated_artifact_inventory_rejected(self) -> None:
        with self.assertRaises(ContractError) as ctx:
            rq.validate_artifact_inventory(
                artifact_inventory(total_count=2),
                expected_run_id=CANDIDATE_RUN_ID,
                expected_name=rq.CANDIDATE_ARTIFACT_NAME,
            )
        self.assertIn("truncated", str(ctx.exception))

    def write_candidate_archive(
        self, name: str, *, replace: dict[str, str] | None = None
    ) -> Path:
        """Write a member-complete candidate archive so rejections are on merit.

        A short archive would be refused on member count alone, which would let
        a traversal or duplicate check regress without any test noticing.
        """
        members = candidate_archive_members()
        if replace:
            for original, _ in replace.items():
                members.pop(original, None)
            members.update({new: b"bad" for new in replace.values()})
        archive = self.tmp / name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(archive, "w") as handle:
                for member, data in members.items():
                    handle.writestr(member, data)
        return archive

    def test_artifact_archive_rejects_path_escape_and_duplicate_members(self) -> None:
        for name, swapped, expected in (
            ("escape.zip", "../escape", "not a flat regular file"),
            ("nested.zip", "nested/file", "not a flat regular file"),
            ("unauthorized.zip", "evil.sh", "unauthorized member"),
        ):
            archive = self.write_candidate_archive(
                name, replace={"sha256sums.txt": swapped}
            )
            destination = self.tmp / f"extract-{name}"
            with self.subTest(name=name), self.assertRaises(ContractError) as ctx:
                rq._extract_flat_artifact_archive(
                    archive, destination, artifact_type="candidate"
                )
            self.assertIn(expected, str(ctx.exception))

        duplicate = self.tmp / "duplicate.zip"
        members = candidate_archive_members()
        members.pop("sha256sums.txt")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(duplicate, "w") as handle:
                for member, data in members.items():
                    handle.writestr(member, data)
                handle.writestr("manifest.json", b"second\n")
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                duplicate, self.tmp / "extract-duplicate", artifact_type="candidate"
            )
        self.assertIn("repeats member", str(ctx.exception))

    def test_artifact_archive_extracts_only_flat_regular_files(self) -> None:
        archive = self.write_candidate_archive("valid-candidate.zip")
        destination = self.tmp / "valid-candidate-extract"
        rq._extract_flat_artifact_archive(archive, destination, artifact_type="candidate")
        self.assertEqual((destination / "manifest.json").read_bytes(), b"{}\n")
        self.assertEqual(
            {entry.name for entry in destination.iterdir()},
            set(candidate_archive_members()),
        )

        real_destination = self.tmp / "real-artifact-destination"
        real_destination.mkdir()
        linked_destination = self.tmp / "linked-artifact-destination"
        linked_destination.symlink_to(real_destination, target_is_directory=True)
        with self.assertRaises(ContractError):
            rq._extract_flat_artifact_archive(archive, linked_destination, artifact_type="candidate")

    def test_attestation_zip_bomb_rejected_on_compression_ratio(self) -> None:
        # Small enough to clear the per-member byte cap, so the ratio bound is
        # the check that has to refuse it.
        archive = self.tmp / "attestation-bomb.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            handle.writestr("qualification-attestation.json", b"\0" * 32000)
        self.assertLess(
            archive.stat().st_size, rq.MAX_ATTESTATION_MEMBER_BYTES
        )
        destination = self.tmp / "attestation-bomb-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="attestation"
            )
        self.assertIn("compression ratio", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_candidate_zip_bomb_rejected_on_compression_ratio(self) -> None:
        archive = self.tmp / "candidate-bomb.zip"
        members = candidate_archive_members()
        members["sha256sums.txt"] = b"\0" * (1024 * 1024)
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            for member, data in members.items():
                handle.writestr(member, data)
        destination = self.tmp / "candidate-bomb-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="candidate"
            )
        self.assertIn("compression ratio", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_oversized_candidate_member_rejected_before_extraction(self) -> None:
        archive = self.write_candidate_archive("oversize-candidate.zip")
        data = bytearray(archive.read_bytes())
        oversize = rq.MAX_CANDIDATE_MEMBER_BYTES + 1
        cd_offset = data.find(b"PK\x01\x02")
        while cd_offset != -1:
            name_length = struct.unpack_from("<H", data, cd_offset + 28)[0]
            if data[cd_offset + 46 : cd_offset + 46 + name_length] == b"sha256sums.txt":
                local_offset = struct.unpack_from("<I", data, cd_offset + 42)[0]
                struct.pack_into("<I", data, cd_offset + 20, oversize)
                struct.pack_into("<I", data, cd_offset + 24, oversize)
                struct.pack_into("<I", data, local_offset + 18, oversize)
                struct.pack_into("<I", data, local_offset + 22, oversize)
                break
            cd_offset = data.find(b"PK\x01\x02", cd_offset + 4)
        self.assertNotEqual(cd_offset, -1)
        archive.write_bytes(data)
        destination = self.tmp / "oversize-candidate-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="candidate"
            )
        self.assertIn("exceeds bound", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_oversized_attestation_member_rejected_before_extraction(self) -> None:
        archive = self.tmp / "oversize-attestation.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(
                "qualification-attestation.json",
                b"x" * (rq.MAX_ATTESTATION_MEMBER_BYTES + 1),
            )
        destination = self.tmp / "oversize-attestation-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="attestation"
            )
        self.assertIn("exceeds bound", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_total_uncompressed_size_is_bounded_before_extraction(self) -> None:
        archive = self.write_candidate_archive("oversize-total-candidate.zip")
        data = bytearray(archive.read_bytes())
        claimed_size = 32 * 1024 * 1024
        cd_offset = data.find(b"PK\x01\x02")
        rewritten = 0
        while cd_offset != -1:
            local_offset = struct.unpack_from("<I", data, cd_offset + 42)[0]
            struct.pack_into("<I", data, cd_offset + 20, claimed_size)
            struct.pack_into("<I", data, cd_offset + 24, claimed_size)
            struct.pack_into("<I", data, local_offset + 18, claimed_size)
            struct.pack_into("<I", data, local_offset + 22, claimed_size)
            rewritten += 1
            cd_offset = data.find(b"PK\x01\x02", cd_offset + 4)
        self.assertEqual(rewritten, len(candidate_archive_members()))
        archive.write_bytes(data)
        destination = self.tmp / "oversize-total-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="candidate"
            )
        self.assertIn("total uncompressed size", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_symlink_archive_member_rejected(self) -> None:
        archive = self.tmp / "symlink-attestation.zip"
        member = zipfile.ZipInfo("qualification-attestation.json")
        member.create_system = 3
        member.external_attr = (rq.stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(member, b"target.json")
        destination = self.tmp / "symlink-attestation-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="attestation"
            )
        self.assertIn("not a flat regular file", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_unauthorized_artifact_member_rejected(self) -> None:
        archive = self.tmp / "unauthorized-attestation.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("evil.sh", b"echo evil\n")
        destination = self.tmp / "unauthorized-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="attestation"
            )
        self.assertIn("unauthorized member", str(ctx.exception))

    def test_short_candidate_member_inventory_rejected(self) -> None:
        archive = self.tmp / "incomplete-candidate.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            for name in ARTIFACTS:
                handle.writestr(name, b"data\n")
        destination = self.tmp / "incomplete-candidate-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="candidate"
            )
        self.assertIn("member count must be exactly", str(ctx.exception))
        self.assertIn(str(len(candidate_archive_members())), str(ctx.exception))

    def test_eocd_count_is_bounded_before_zipfile_parses_members(self) -> None:
        archive = self.tmp / "too-many-attestation-members.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("qualification-attestation.json", b"evidence\n")
            handle.writestr("evil.txt", b"extra\n")
        destination = self.tmp / "too-many-attestation-members-extract"
        with mock.patch.object(
            rq.zipfile,
            "ZipFile",
            side_effect=AssertionError("ZipFile must not parse an over-count archive"),
        ):
            with self.assertRaises(ContractError) as ctx:
                rq._extract_flat_artifact_archive(
                    archive, destination, artifact_type="attestation"
                )
        self.assertIn("end-of-central-directory member count", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_archive_preamble_cannot_hide_outside_the_member_inventory(self) -> None:
        archive = self.tmp / "prefixed-attestation.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("qualification-attestation.json", b"evidence\n")
        original = bytearray(archive.read_bytes())
        old_cd_offset = original.find(b"PK\x01\x02")
        old_eocd_offset = original.find(b"PK\x05\x06")
        self.assertNotEqual(old_cd_offset, -1)
        self.assertNotEqual(old_eocd_offset, -1)
        prefix = b"JUNK"
        data = bytearray(prefix) + original
        cd_offset = old_cd_offset + len(prefix)
        eocd_offset = old_eocd_offset + len(prefix)
        local_offset = struct.unpack_from("<I", data, cd_offset + 42)[0]
        struct.pack_into("<I", data, cd_offset + 42, local_offset + len(prefix))
        struct.pack_into("<I", data, eocd_offset + 16, cd_offset)
        archive.write_bytes(data)
        destination = self.tmp / "prefixed-attestation-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="attestation"
            )
        self.assertIn("start at byte zero", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_encrypted_zip_member_rejected(self) -> None:
        for flag in (0x1, 0x40, 0x2000):
            archive = self.tmp / f"encrypted-{flag:x}.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("qualification-attestation.json", b"secret")
            data = bytearray(archive.read_bytes())
            local_flags = struct.unpack_from("<H", data, 6)[0]
            struct.pack_into("<H", data, 6, local_flags | flag)
            cd_offset = data.find(b"PK\x01\x02")
            self.assertNotEqual(cd_offset, -1)
            central_flags = struct.unpack_from("<H", data, cd_offset + 8)[0]
            struct.pack_into("<H", data, cd_offset + 8, central_flags | flag)
            archive.write_bytes(data)
            destination = self.tmp / f"encrypted-{flag:x}-extract"
            with self.subTest(flag=flag):
                with self.assertRaises(ContractError) as ctx:
                    rq._extract_flat_artifact_archive(
                        archive, destination, artifact_type="attestation"
                    )
                self.assertIn("encrypted", str(ctx.exception))

    def test_data_descriptor_zip_member_rejected(self) -> None:
        archive = self.tmp / "data-descriptor.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("qualification-attestation.json", b"evidence\n")
        data = bytearray(archive.read_bytes())
        local_flags = struct.unpack_from("<H", data, 6)[0]
        struct.pack_into("<H", data, 6, local_flags | 0x8)
        struct.pack_into("<III", data, 14, 0, 0, 0)
        cd_offset = data.find(b"PK\x01\x02")
        self.assertNotEqual(cd_offset, -1)
        central_flags = struct.unpack_from("<H", data, cd_offset + 8)[0]
        struct.pack_into("<H", data, cd_offset + 8, central_flags | 0x8)
        archive.write_bytes(data)
        destination = self.tmp / "data-descriptor-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="attestation"
            )
        self.assertIn("data descriptor", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_signed_github_style_data_descriptor_is_exactly_validated(self) -> None:
        archive = self.tmp / "signed-data-descriptor.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("qualification-attestation.json", b"evidence\n")
        data = bytearray(archive.read_bytes())
        cd_offset = data.find(b"PK\x01\x02")
        self.assertNotEqual(cd_offset, -1)
        crc = struct.unpack_from("<I", data, cd_offset + 16)[0]
        compressed_size = struct.unpack_from("<I", data, cd_offset + 20)[0]
        file_size = struct.unpack_from("<I", data, cd_offset + 24)[0]
        local_flags = struct.unpack_from("<H", data, 6)[0]
        struct.pack_into("<H", data, 6, local_flags | 0x8)
        struct.pack_into("<III", data, 14, 0, 0, 0)
        central_flags = struct.unpack_from("<H", data, cd_offset + 8)[0]
        struct.pack_into("<H", data, cd_offset + 8, central_flags | 0x8)
        descriptor = struct.pack(
            "<4sIII", b"PK\x07\x08", crc, compressed_size, file_size
        )
        data[cd_offset:cd_offset] = descriptor
        eocd_offset = data.find(b"PK\x05\x06", cd_offset + len(descriptor))
        self.assertNotEqual(eocd_offset, -1)
        struct.pack_into("<I", data, eocd_offset + 16, cd_offset + len(descriptor))
        archive.write_bytes(data)

        destination = self.tmp / "signed-data-descriptor-extract"
        rq._extract_flat_artifact_archive(
            archive, destination, artifact_type="attestation"
        )
        self.assertEqual(
            (destination / "qualification-attestation.json").read_bytes(),
            b"evidence\n",
        )

        mismatched = bytearray(archive.read_bytes())
        descriptor_offset = mismatched.find(b"PK\x07\x08")
        self.assertNotEqual(descriptor_offset, -1)
        struct.pack_into("<I", mismatched, descriptor_offset + 4, crc ^ 0xFFFFFFFF)
        archive.write_bytes(mismatched)
        destination = self.tmp / "mismatched-data-descriptor-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="attestation"
            )
        self.assertIn("data descriptor", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_local_and_central_flag_mismatch_rejected(self) -> None:
        archive = self.tmp / "flag-mismatch.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("qualification-attestation.json", b"evidence\n")
        data = bytearray(archive.read_bytes())
        local_flags = struct.unpack_from("<H", data, 6)[0]
        struct.pack_into("<H", data, 6, local_flags ^ 0x800)
        archive.write_bytes(data)
        destination = self.tmp / "flag-mismatch-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="attestation"
            )
        self.assertIn("flags", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_nul_truncated_member_name_cannot_masquerade_as_allowlisted(self) -> None:
        archive = self.write_candidate_archive(
            "nul-name-candidate.zip",
            replace={"manifest.json": "manifest.jsonX"},
        )
        data = bytearray(archive.read_bytes())
        cd_offset = data.find(b"PK\x01\x02")
        while cd_offset != -1:
            name_length = struct.unpack_from("<H", data, cd_offset + 28)[0]
            name_start = cd_offset + 46
            if data[name_start : name_start + name_length] == b"manifest.jsonX":
                local_offset = struct.unpack_from("<I", data, cd_offset + 42)[0]
                local_name_start = local_offset + 30
                data[local_name_start + name_length - 1] = 0
                data[name_start + name_length - 1] = 0
                break
            cd_offset = data.find(b"PK\x01\x02", cd_offset + 4)
        self.assertNotEqual(cd_offset, -1)
        archive.write_bytes(data)
        destination = self.tmp / "nul-name-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="candidate"
            )
        self.assertIn("ambiguous", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_member_extra_fields_and_comments_cannot_hide_metadata(self) -> None:
        for kind in ("extra", "comment"):
            archive = self.tmp / f"hidden-{kind}-attestation.zip"
            member = zipfile.ZipInfo("qualification-attestation.json")
            if kind == "extra":
                member.extra = struct.pack("<HH", 0xCAFE, 0)
            else:
                member.comment = b"hidden"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(member, b"evidence\n")
            destination = self.tmp / f"hidden-{kind}-extract"
            with self.subTest(kind=kind):
                with self.assertRaises(ContractError) as ctx:
                    rq._extract_flat_artifact_archive(
                        archive, destination, artifact_type="attestation"
                    )
                self.assertIn("metadata", str(ctx.exception))
                self.assertEqual(list(destination.iterdir()), [])

    def test_local_only_extra_field_cannot_hide_metadata(self) -> None:
        archive = self.tmp / "local-extra-attestation.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("qualification-attestation.json", b"evidence\n")
        data = bytearray(archive.read_bytes())
        name_length = struct.unpack_from("<H", data, 26)[0]
        payload_offset = 30 + name_length
        local_extra = struct.pack("<HH", 0xCAFE, 0)
        data[payload_offset:payload_offset] = local_extra
        struct.pack_into("<H", data, 28, len(local_extra))
        cd_offset = data.find(b"PK\x01\x02")
        eocd_offset = data.find(b"PK\x05\x06")
        self.assertNotEqual(cd_offset, -1)
        self.assertNotEqual(eocd_offset, -1)
        struct.pack_into("<I", data, eocd_offset + 16, cd_offset)
        archive.write_bytes(data)
        destination = self.tmp / "local-extra-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="attestation"
            )
        self.assertIn("metadata", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_unsupported_compression_method_rejected(self) -> None:
        archive = self.write_candidate_archive("bzip2-candidate.zip")
        data = bytearray(archive.read_bytes())
        cd_offset = data.find(b"PK\x01\x02")
        while cd_offset != -1:
            name_length = struct.unpack_from("<H", data, cd_offset + 28)[0]
            if data[cd_offset + 46 : cd_offset + 46 + name_length] == b"manifest.json":
                local_offset = struct.unpack_from("<I", data, cd_offset + 42)[0]
                struct.pack_into("<H", data, cd_offset + 10, zipfile.ZIP_BZIP2)
                struct.pack_into("<H", data, local_offset + 8, zipfile.ZIP_BZIP2)
                break
            cd_offset = data.find(b"PK\x01\x02", cd_offset + 4)
        self.assertNotEqual(cd_offset, -1)
        archive.write_bytes(data)
        destination = self.tmp / "bzip2-candidate-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="candidate"
            )
        self.assertIn("unsupported compression method", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_local_header_disagreeing_with_central_directory_rejected(self) -> None:
        # Only the local header is rewritten, so the central directory still
        # looks benign. The archive is ambiguous about what it would actually
        # decompress and must be refused rather than reconciled.
        archive = self.write_candidate_archive("local-mismatch-candidate.zip")
        data = bytearray(archive.read_bytes())
        cd_offset = data.find(b"PK\x01\x02")
        local_offset = struct.unpack_from("<I", data, cd_offset + 42)[0]
        struct.pack_into("<I", data, local_offset + 22, 4096)
        archive.write_bytes(data)
        destination = self.tmp / "local-mismatch-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="candidate"
            )
        self.assertIn("disagrees with the central directory", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_local_header_name_disagreeing_with_central_directory_rejected(self) -> None:
        archive = self.write_candidate_archive("local-name-candidate.zip")
        data = bytearray(archive.read_bytes())
        cd_offset = data.find(b"PK\x01\x02")
        name_length = struct.unpack_from("<H", data, cd_offset + 28)[0]
        name = bytes(data[cd_offset + 46 : cd_offset + 46 + name_length])
        local_offset = struct.unpack_from("<I", data, cd_offset + 42)[0]
        start = local_offset + 30
        self.assertEqual(bytes(data[start : start + name_length]), name)
        data[start] = ord("Z")
        archive.write_bytes(data)
        destination = self.tmp / "local-name-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="candidate"
            )
        self.assertIn("disagrees with the central directory", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_overlapping_members_rejected(self) -> None:
        archive = self.write_candidate_archive("overlapping-candidate.zip")
        data = bytearray(archive.read_bytes())
        offsets = []
        cd_offset = data.find(b"PK\x01\x02")
        while cd_offset != -1:
            offsets.append(cd_offset + 42)
            cd_offset = data.find(b"PK\x01\x02", cd_offset + 4)
        self.assertGreater(len(offsets), 1)
        first = struct.unpack_from("<I", data, offsets[0])[0]
        # Point a second member's local header back inside the first member's
        # payload so the two entries claim overlapping bytes.
        struct.pack_into("<I", data, offsets[1], first)
        archive.write_bytes(data)
        destination = self.tmp / "overlapping-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="candidate"
            )
        self.assertIn("overlapping members", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_member_payload_cannot_overlap_the_central_directory(self) -> None:
        archive = self.write_candidate_archive("central-overlap-candidate.zip")
        data = bytearray(archive.read_bytes())
        cd_offset = data.find(b"PK\x01\x02")
        while cd_offset != -1:
            name_length = struct.unpack_from("<H", data, cd_offset + 28)[0]
            name = data[cd_offset + 46 : cd_offset + 46 + name_length]
            if name == b"sha256sums.txt":
                local_offset = struct.unpack_from("<I", data, cd_offset + 42)[0]
                compressed_size = struct.unpack_from("<I", data, cd_offset + 20)[0]
                file_size = struct.unpack_from("<I", data, cd_offset + 24)[0]
                struct.pack_into("<I", data, cd_offset + 20, compressed_size + 1)
                struct.pack_into("<I", data, cd_offset + 24, file_size + 1)
                struct.pack_into("<I", data, local_offset + 18, compressed_size + 1)
                struct.pack_into("<I", data, local_offset + 22, file_size + 1)
                break
            cd_offset = data.find(b"PK\x01\x02", cd_offset + 4)
        self.assertNotEqual(cd_offset, -1)
        archive.write_bytes(data)
        destination = self.tmp / "central-overlap-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="candidate"
            )
        self.assertIn("central directory", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_unclaimed_gap_before_central_directory_rejected(self) -> None:
        archive = self.tmp / "central-gap-attestation.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("qualification-attestation.json", b"evidence\n")
        data = bytearray(archive.read_bytes())
        cd_offset = data.find(b"PK\x01\x02")
        eocd_offset = data.find(b"PK\x05\x06")
        self.assertNotEqual(cd_offset, -1)
        self.assertNotEqual(eocd_offset, -1)
        gap = b"JUNK"
        data[cd_offset:cd_offset] = gap
        eocd_offset += len(gap)
        struct.pack_into("<I", data, eocd_offset + 16, cd_offset + len(gap))
        archive.write_bytes(data)
        destination = self.tmp / "central-gap-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="attestation"
            )
        self.assertIn("unclaimed gap", str(ctx.exception))
        self.assertEqual(list(destination.iterdir()), [])

    def test_failed_extraction_leaves_no_partial_trusted_output(self) -> None:
        # The corrupted member sorts last, so earlier members extract cleanly
        # before the failure and would survive without staged extraction.
        archive = self.write_candidate_archive("corrupt-candidate.zip")
        data = bytearray(archive.read_bytes())
        cd_offset = data.find(b"PK\x01\x02")
        while cd_offset != -1:
            name_length = struct.unpack_from("<H", data, cd_offset + 28)[0]
            if data[cd_offset + 46 : cd_offset + 46 + name_length] == b"sha256sums.txt":
                local_offset = struct.unpack_from("<I", data, cd_offset + 42)[0]
                payload = local_offset + 30 + name_length
                data[payload] ^= 0xFF
                break
            cd_offset = data.find(b"PK\x01\x02", cd_offset + 4)
        self.assertNotEqual(cd_offset, -1)
        archive.write_bytes(data)
        destination = self.tmp / "corrupt-candidate-extract"
        with self.assertRaises(ContractError):
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="candidate"
            )
        self.assertTrue(destination.is_dir())
        self.assertEqual(list(destination.iterdir()), [])
        self.assertEqual(
            [entry for entry in destination.parent.iterdir() if entry.name.startswith(".artifact-extract-")],
            [],
        )

    def test_interrupted_placement_leaves_no_partial_trusted_output(self) -> None:
        # Every member has already passed, so the failure lands while verified
        # files are being moved in. The destination must still come back empty
        # rather than half populated.
        archive = self.write_candidate_archive("interrupted-candidate.zip")
        destination = self.tmp / "interrupted-extract"
        real_replace = rq.os.replace
        calls = {"count": 0}

        def failing_replace(
            source: object, target: object, *args: object, **kwargs: object
        ) -> None:
            calls["count"] += 1
            if calls["count"] > 3:
                raise OSError("simulated placement failure")
            real_replace(source, target, *args, **kwargs)

        with mock.patch.object(rq.os, "replace", failing_replace):
            with self.assertRaises(ContractError) as ctx:
                rq._extract_flat_artifact_archive(
                    archive, destination, artifact_type="candidate"
                )
        self.assertIn("could not place verified artifact members", str(ctx.exception))
        self.assertGreater(calls["count"], 3)
        self.assertTrue(destination.is_dir())
        self.assertEqual(list(destination.iterdir()), [])

    def test_destination_swap_cannot_redirect_or_preserve_verified_output(self) -> None:
        archive = self.write_candidate_archive("destination-swap-candidate.zip")
        destination = self.tmp / "destination-swap-extract"
        displaced = self.tmp / "displaced-destination"
        attacker_target = self.tmp / "attacker-target"
        attacker_target.mkdir()
        real_replace = rq.os.replace
        swapped = False

        def swapping_replace(
            source: object, target: object, *args: object, **kwargs: object
        ) -> None:
            nonlocal swapped
            if not swapped:
                destination.rename(displaced)
                destination.symlink_to(attacker_target, target_is_directory=True)
                swapped = True
            real_replace(source, target, *args, **kwargs)

        with mock.patch.object(rq.os, "replace", swapping_replace):
            with self.assertRaises(ContractError) as ctx:
                rq._extract_flat_artifact_archive(
                    archive, destination, artifact_type="candidate"
                )
        self.assertIn("changed during extraction", str(ctx.exception))
        self.assertEqual(list(attacker_target.iterdir()), [])
        self.assertTrue(displaced.is_dir())
        self.assertEqual(list(displaced.iterdir()), [])

    def test_unknown_artifact_type_rejected(self) -> None:
        archive = self.write_candidate_archive("unknown-type-candidate.zip")
        destination = self.tmp / "unknown-extract"
        with self.assertRaises(ContractError) as ctx:
            rq._extract_flat_artifact_archive(
                archive, destination, artifact_type="invalid-type"
            )
        self.assertIn("unknown artifact type", str(ctx.exception))

    def test_mutated_asr_cold_transcript_rejected(self) -> None:
        bad = copy.deepcopy(self.attestation)
        bad["phases"]["speech_to_text"]["modes"][0]["cold_transcript"] = "completely wrong transcript"
        with self.assertRaises(ContractError) as ctx:
            self.verify(bad)
        self.assertIn("transcript", str(ctx.exception).lower())

    def test_mutated_asr_warm_transcript_rejected(self) -> None:
        bad = copy.deepcopy(self.attestation)
        bad["phases"]["speech_to_text"]["modes"][0]["warm_transcript"] = "completely wrong transcript"
        with self.assertRaises(ContractError) as ctx:
            self.verify(bad)
        self.assertIn("transcript", str(ctx.exception).lower())

    def test_malformed_asr_cancellation_schema_rejected(self) -> None:
        for malformed in (
            "cancelled",
            "cancel:invalid:0",
            "cancel:resolved:-1",
            "cancel:resolved:abc",
            "cancel:resolved:01",
            "cancel:rejected:00",
            "cancel:resolved:" + "9" * 10000,
            "",
            None,
        ):
            bad = copy.deepcopy(self.attestation)
            bad["phases"]["speech_to_text"]["modes"][0]["cancellation_result"] = malformed
            with self.subTest(malformed=malformed), self.assertRaises(ContractError):
                self.verify(bad)

    def test_rejected_cancellation_must_report_no_output(self) -> None:
        # The gate only reaches the rejected state by throwing before it ever
        # assigns output, so a rejected result carrying characters means the
        # recorded state and the recorded output contradict each other.
        bad = copy.deepcopy(self.attestation)
        bad["phases"]["speech_to_text"]["modes"][0][
            "cancellation_result"
        ] = "cancel:rejected:37"
        with self.assertRaises(ContractError) as ctx:
            self.verify(bad)
        self.assertIn("rejected but reported", str(ctx.exception))

        good = copy.deepcopy(self.attestation)
        good["phases"]["speech_to_text"]["modes"][0][
            "cancellation_result"
        ] = "cancel:rejected:0"
        self.assertTrue(self.verify(good)["verified"])

    def test_speech_phase_rejects_a_rejected_cancellation_with_output(self) -> None:
        payload = {
            "modeResults": [
                {
                    "memoryMode": "wasm32",
                    "runtimeMode": "direct",
                    "elapsedMs": 10,
                    "phaseTimingsMs": {
                        "cancellationMs": 1,
                        "coldTranscriptMs": 1,
                        "modelLoadMs": 1,
                        "projectorLoadMs": 1,
                        "silenceMs": 1,
                        "warmTranscriptMs": 1,
                    },
                    "coldTranscript": DEFAULT_EXPECTED_TEXT,
                    "warmTranscript": DEFAULT_EXPECTED_TEXT,
                    "cancellation": "cancel:rejected:12",
                    "silenceTranscript": "",
                }
            ]
        }
        with self.assertRaises(ContractError) as ctx:
            rq._speech_phase(payload, 1)
        self.assertIn("rejected but reported", str(ctx.exception))

    def test_asr_silence_transcript_must_stay_empty(self) -> None:
        bad = copy.deepcopy(self.attestation)
        bad["phases"]["speech_to_text"]["modes"][0]["silence_transcript"] = "hallucinated speech"
        with self.assertRaises(ContractError):
            self.verify(bad)

    def test_tts_truncated_must_be_false(self) -> None:
        for val in (True, "false", None, 1):
            bad = copy.deepcopy(self.attestation)
            bad["phases"]["text_to_speech"]["modes"][0]["truncated"] = val
            with self.subTest(val=val), self.assertRaises(ContractError):
                self.verify(bad)

    def test_tts_lifecycle_evidence_cancellation_tested_must_be_true(self) -> None:
        for val in (False, "true", None, 0):
            bad = copy.deepcopy(self.attestation)
            bad["phases"]["text_to_speech"]["modes"][0]["cancellation_tested"] = val
            with self.subTest(val=val), self.assertRaises(ContractError):
                self.verify(bad)

    def test_tts_lifecycle_evidence_pre_aborted_tested_must_be_true(self) -> None:
        for val in (False, "true", None, 0):
            bad = copy.deepcopy(self.attestation)
            bad["phases"]["text_to_speech"]["modes"][0]["pre_aborted_tested"] = val
            with self.subTest(val=val), self.assertRaises(ContractError):
                self.verify(bad)

    def test_tts_lifecycle_evidence_reuse_sample_count_must_be_positive(self) -> None:
        for val in (0, -1, "2400", None, True):
            bad = copy.deepcopy(self.attestation)
            bad["phases"]["text_to_speech"]["modes"][0]["reuse_sample_count"] = val
            with self.subTest(val=val), self.assertRaises(ContractError):
                self.verify(bad)

    def test_tts_lifecycle_evidence_unload_tested_must_be_true(self) -> None:
        for val in (False, "true", None, 0):
            bad = copy.deepcopy(self.attestation)
            bad["phases"]["text_to_speech"]["modes"][0]["unload_tested"] = val
            with self.subTest(val=val), self.assertRaises(ContractError):
                self.verify(bad)

    def test_wrong_workflow_run_actor_rejected(self) -> None:
        for actor in ({"login": "attacker"}, {"login": "someone-else"}, None, {}):
            with self.subTest(actor=actor), self.assertRaises(ContractError):
                rq.validate_workflow_run(
                    workflow_run(actor=actor),
                    expected_run_id=CANDIDATE_RUN_ID,
                    expected_workflow_path=rq.CANDIDATE_WORKFLOW_PATH,
                    expected_head_branch=DEFAULT_HEAD_BRANCH,
                    expected_run_attempt=1,
                )

    def test_wrong_workflow_run_triggering_actor_rejected(self) -> None:
        for actor in ({"login": "attacker"}, {"login": "someone-else"}, None, {}):
            with self.subTest(actor=actor), self.assertRaises(ContractError):
                rq.validate_workflow_run(
                    workflow_run(triggering_actor=actor),
                    expected_run_id=CANDIDATE_RUN_ID,
                    expected_workflow_path=rq.CANDIDATE_WORKFLOW_PATH,
                    expected_head_branch=DEFAULT_HEAD_BRANCH,
                    expected_run_attempt=1,
                )

    def test_candidate_artifact_id_and_run_attempt_bound_into_attestation(self) -> None:
        self.assertEqual(self.attestation["candidate_artifact_id"], 7)
        self.assertEqual(self.attestation["candidate_run_attempt"], 1)

    def test_candidate_artifact_id_mismatch_rejected(self) -> None:
        with self.assertRaises(ContractError) as ctx:
            self.verify(self.attestation, candidate_artifact_id=999)
        self.assertIn("candidate_artifact_id", str(ctx.exception))

    def test_candidate_run_attempt_mismatch_rejected(self) -> None:
        with self.assertRaises(ContractError) as ctx:
            self.verify(self.attestation, candidate_run_attempt=2)
        self.assertIn("candidate_run_attempt", str(ctx.exception))

    def test_attestation_receipt_contains_artifact_id_and_run_attempt(self) -> None:
        receipt = self.verify(
            self.attestation,
            candidate_artifact_id=7,
            candidate_run_attempt=1,
        )
        self.assertEqual(receipt["candidate_artifact_id"], 7)
        self.assertEqual(receipt["candidate_run_attempt"], 1)

    def test_artifact_name_alone_does_not_prove_provenance(self) -> None:
        inventory = artifact_inventory(
            artifacts=[
                {
                    "id": 7,
                    "name": rq.CANDIDATE_ARTIFACT_NAME,
                    "expired": False,
                    "workflow_run": {"id": 424242},
                }
            ]
        )
        with self.assertRaises(ContractError) as ctx:
            rq.validate_artifact_inventory(
                inventory,
                expected_run_id=CANDIDATE_RUN_ID,
                expected_name=rq.CANDIDATE_ARTIFACT_NAME,
            )
        self.assertIn("does not belong to run", str(ctx.exception))

    # --- diagnostics and resources ----------------------------------------

    def test_diagnostics_are_sanitized(self) -> None:
        raw = (
            "failed https://huggingface.co/model.gguf?token=secret123#frag and "
            "https://user:pass@example.com/a/b?x=1\n"
            "Authorization: Bearer ghp_this_must_not_escape\n"
            "GH_TOKEN=another-secret\n"
            "api_key=plain-secret"
        )
        sanitized = rq.sanitize_diagnostic_text(raw)
        for leaked in (
            "secret123",
            "token=",
            "pass@",
            "#frag",
            "?x=1",
            "ghp_this_must_not_escape",
            "another-secret",
            "plain-secret",
        ):
            self.assertNotIn(leaked, sanitized)
        self.assertIn("https://huggingface.co/model.gguf", sanitized)
        self.assertIn("https://example.com/a/b", sanitized)

    def test_real_credentials_are_redacted_no_matter_where_they_appear(self) -> None:
        raw = (
            "Authorization: Basic dXNlcjpwYXNz\n"
            "apiKey: plain-secret\n"
            "MY_PASSWORD=hunter2\n"
            "service.credential = cred-value\n"
            "client_secret=shhh\n"
            "fetch https://example.com/m.gguf?sig=abc#frag"
        )
        sanitized = rq.sanitize_diagnostic_text(raw)
        for leaked in (
            "dXNlcjpwYXNz",
            "plain-secret",
            "hunter2",
            "cred-value",
            "shhh",
            "sig=abc",
            "#frag",
        ):
            self.assertNotIn(leaked, sanitized)
        self.assertIn("https://example.com/m.gguf", sanitized)

    def test_malformed_url_diagnostic_is_redacted_without_raising(self) -> None:
        raw = "failed https://user:pass@[broken?token=super-secret#frag"
        sanitized = rq.sanitize_diagnostic_text(raw)
        for leaked in ("user", "pass", "super-secret", "token=", "#frag"):
            self.assertNotIn(leaked, sanitized)
        self.assertIn("https://<redacted-url>", sanitized)

    def test_malformed_url_authorities_are_redacted_without_leaking(self) -> None:
        for raw in (
            "https:///user:pass@example.com/model.gguf?token=secret#frag",
            "https://user:password/model.gguf?token=secret#frag",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(
                    rq.sanitize_diagnostic_text(raw),
                    "https://<redacted-url>",
                )

    def test_url_sanitization_is_case_insensitive(self) -> None:
        raw = "HTTPS://user:pass@example.com/m.gguf?sig=secret#frag"
        sanitized = rq.sanitize_diagnostic_text(raw)
        for leaked in ("user", "pass", "sig=secret", "#frag"):
            self.assertNotIn(leaked, sanitized)
        self.assertEqual(sanitized, "https://example.com/m.gguf")

    def test_llama_token_counters_are_not_treated_as_credentials(self) -> None:
        raw = (
            "llama_context: n_tokens = 65\n"
            "llama_context: n_tokens_batch = 65\n"
            "decoded n_tokens=65 n_tokens_batch=65\n"
        )
        self.assertEqual(rq.sanitize_diagnostic_text(raw), raw)

    def test_plural_token_credentials_do_not_use_the_counter_carveout(self) -> None:
        raw = (
            "access_tokens=access-secret\n"
            'refresh_tokens_json="refresh secret"\n'
            "n_tokens=counter-shaped-secret\n"
            "n_tokens_secret=65\n"
        )
        sanitized = rq.sanitize_diagnostic_text(raw)
        for leaked in ("access-secret", "refresh secret", "counter-shaped-secret"):
            self.assertNotIn(leaked, sanitized)
        self.assertNotIn("n_tokens_secret", sanitized)

        structured = json.loads(
            rq.sanitize_diagnostic_stdout(
                json.dumps(
                    {
                        "access_tokens": "access-secret",
                        "n_tokens": "counter-shaped-secret",
                        "n_tokens_batch": 65,
                        "n_tokens_api_key": 65,
                    }
                )
            )
        )
        self.assertEqual(structured["access_tokens"], rq.REDACTED_CREDENTIAL)
        self.assertEqual(structured["n_tokens"], rq.REDACTED_CREDENTIAL)
        self.assertEqual(structured["n_tokens_batch"], 65)
        self.assertEqual(structured["n_tokens_api_key"], rq.REDACTED_CREDENTIAL)

    def test_credential_names_with_alphanumeric_suffixes_are_redacted(self) -> None:
        raw = (
            "tokenValue=token-secret\n"
            "secretValue=secret-secret\n"
            "passwordValue=password-secret\n"
            "credentialValue=credential-secret\n"
            "apiKeyV2=api-key-secret\n"
        )
        sanitized = rq.sanitize_diagnostic_text(raw)
        for leaked in (
            "token-secret",
            "secret-secret",
            "password-secret",
            "credential-secret",
            "api-key-secret",
        ):
            self.assertNotIn(leaked, sanitized)

    def test_quoted_credential_assignment_values_are_fully_redacted(self) -> None:
        raw = (
            'PASSWORD="correct horse battery staple"\n'
            "api_key='quoted api key'\n"
        )
        sanitized = rq.sanitize_diagnostic_text(raw)
        for leaked in ("correct", "horse", "battery", "staple", "quoted", "api key"):
            self.assertNotIn(leaked, sanitized)
        self.assertEqual(
            sanitized,
            f"{rq.REDACTED_CREDENTIAL}\n{rq.REDACTED_CREDENTIAL}\n",
        )

    def test_quoted_credential_keys_are_redacted_in_text_fallback(self) -> None:
        malformed_json = (
            '{"apiKey": "plain secret", '
            '"Authorization": "Bearer bearer-secret", '
            '"password": "hunter2"'
        )
        sanitized = rq.sanitize_diagnostic_stdout(malformed_json)
        self.assertNotIn("plain secret", sanitized)
        self.assertNotIn("bearer-secret", sanitized)
        self.assertNotIn("hunter2", sanitized)

        counters = '"n_tokens": 65, "n_tokens_batch": 65'
        self.assertEqual(rq.sanitize_diagnostic_text(counters), counters)

    def test_structured_stdout_diagnostic_stays_valid_json(self) -> None:
        diagnostics = self.tmp / "structured-diagnostics"
        diagnostics.mkdir()
        payload = {
            "ok": True,
            "note": 'n_tokens = 65\nsaid "hi"\tthen stopped',
            "apiKey": "plain-secret",
            "modeResults": [
                {"log": "Authorization: Bearer ghp_must_not_escape", "n_tokens": 65}
            ],
            "modelUrl": "https://example.com/m.gguf?token=secret123#frag",
        }
        rq._write_smoke_diagnostics(
            diagnostics, "structured", json.dumps(payload), ""
        )
        persisted = json.loads(
            (diagnostics / "structured-stdout.json").read_text(encoding="utf-8")
        )
        self.assertEqual(persisted["note"], payload["note"])
        self.assertEqual(persisted["modeResults"][0]["n_tokens"], 65)
        self.assertEqual(persisted["apiKey"], "<redacted-credential>")
        self.assertEqual(persisted["modelUrl"], "https://example.com/m.gguf")
        self.assertNotIn("ghp_must_not_escape", persisted["modeResults"][0]["log"])
        self.assertNotIn(
            "secret123",
            (diagnostics / "structured-stdout.json").read_text(encoding="utf-8"),
        )

    def test_structured_stdout_preserves_escaped_unicode_safely(self) -> None:
        diagnostics = self.tmp / "unicode-diagnostics"
        diagnostics.mkdir()
        rq._write_smoke_diagnostics(
            diagnostics, "unicode", r'{"ok":true,"note":"\ud800"}', ""
        )
        serialized = (diagnostics / "unicode-stdout.json").read_text(encoding="utf-8")
        self.assertIn(r"\ud800", serialized)
        self.assertEqual(json.loads(serialized)["note"], "\ud800")

    def test_structured_authorization_values_are_redacted(self) -> None:
        payload = {
            "headers": {
                "Authorization": "Bearer bearer-secret",
                "authorization": "Basic basic-secret",
            }
        }
        sanitized = json.loads(rq.sanitize_diagnostic_stdout(json.dumps(payload)))
        self.assertEqual(
            sanitized,
            {
                "headers": {
                    "Authorization": "<redacted-credential>",
                    "authorization": "<redacted-credential>",
                }
            },
        )

    def test_structured_keys_cannot_leak_urls_or_assignments(self) -> None:
        payload = {
            "https://user:pass@example.com/m.gguf?sig=secret#frag": "url-key",
            "api_key=key-secret": "assignment-key",
            "ordinary": {"n_tokens": 65},
        }
        serialized = rq.sanitize_diagnostic_stdout(json.dumps(payload))
        sanitized = json.loads(serialized)
        for leaked in ("user", "pass", "sig=secret", "#frag", "key-secret"):
            self.assertNotIn(leaked, serialized)
        self.assertEqual(sanitized["https://example.com/m.gguf"], "url-key")
        self.assertEqual(sanitized[rq.REDACTED_CREDENTIAL], "assignment-key")
        self.assertEqual(sanitized["ordinary"]["n_tokens"], 65)

    def test_successful_child_is_parsed_from_raw_stdout(self) -> None:
        import sys as _sys

        diagnostics = self.tmp / "raw-stdout-diagnostics"
        diagnostics.mkdir()
        payload = {
            "ok": True,
            "note": "n_tokens = 65 n_tokens_batch = 65",
            "modelUrl": "https://example.com/m.gguf?token=secret123",
        }
        program = f"import json,sys; sys.stdout.write({json.dumps(json.dumps(payload))})"
        parsed = rq._run_smoke(
            [_sys.executable, "-c", program],
            "raw-probe",
            diagnostics,
            timeout_seconds=30,
        )
        self.assertEqual(parsed, payload)
        persisted = json.loads(
            (diagnostics / "raw-probe-stdout.json").read_text(encoding="utf-8")
        )
        self.assertEqual(persisted["note"], payload["note"])
        self.assertEqual(persisted["modelUrl"], "https://example.com/m.gguf")

    def test_max_rss_is_reported_in_bytes(self) -> None:
        import subprocess
        import sys as _sys

        subprocess.run([_sys.executable, "-c", "bytearray(64 * 1024 * 1024)"], check=True)
        # ru_maxrss is kibibytes on Linux and bytes on macOS; the normalized value
        # must be plausible as bytes on either, never 1024x off.
        self.assertGreater(rq.max_rss_bytes(), 4_000_000)

    def test_child_env_drops_ambient_smoke_configuration(self) -> None:
        import os

        os.environ["LLAMA_WEBGPU_SPEECH_MODEL_URL"] = "https://evil.example/model.gguf"
        os.environ["BRIDGE_DIST_DIR"] = "/tmp/elsewhere"
        os.environ["GH_TOKEN"] = "must-not-reach-browser"
        os.environ["EXAMPLE_API_KEY"] = "must-not-reach-browser"
        try:
            env = rq._child_env()
        finally:
            del os.environ["LLAMA_WEBGPU_SPEECH_MODEL_URL"]
            del os.environ["BRIDGE_DIST_DIR"]
            del os.environ["GH_TOKEN"]
            del os.environ["EXAMPLE_API_KEY"]
        self.assertFalse("LLAMA_WEBGPU_SPEECH_MODEL_URL" in env)
        self.assertFalse("BRIDGE_DIST_DIR" in env)
        self.assertFalse("GH_TOKEN" in env)
        self.assertFalse("EXAMPLE_API_KEY" in env)

    def test_smoke_timeout_is_bounded_and_writes_sanitized_diagnostics(self) -> None:
        import sys as _sys

        diagnostics = self.tmp / "timeout-diagnostics"
        diagnostics.mkdir()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(ContractError) as ctx:
                rq._run_smoke(
                    [
                        _sys.executable,
                        "-c",
                        "import sys,time; print('GH_TOKEN=secret', file=sys.stderr); time.sleep(5)",
                    ],
                    "timeout-probe",
                    diagnostics,
                    timeout_seconds=0.05,
                )
        self.assertIn("timed out", str(ctx.exception))
        diagnostic = (diagnostics / "timeout-probe-stderr.log").read_text()
        self.assertNotIn("secret", diagnostic)

    def test_malformed_child_output_still_writes_sanitized_diagnostics(self) -> None:
        import sys as _sys

        diagnostics = self.tmp / "malformed-diagnostics"
        diagnostics.mkdir()
        program = (
            "import sys; "
            "sys.stdout.buffer.write(b'GH_TOKEN=stdout-secret\\nbad\\xff'); "
            "sys.stderr.buffer.write(b'api_key=stderr-secret\\nbad\\xff'); "
            "raise SystemExit(1)"
        )
        captured_stderr = io.StringIO()
        with contextlib.redirect_stderr(captured_stderr):
            with self.assertRaises(ContractError) as ctx:
                rq._run_smoke(
                    [_sys.executable, "-c", program],
                    "malformed-probe",
                    diagnostics,
                    timeout_seconds=30,
                )
        self.assertIn("failed with exit status 1", str(ctx.exception))
        persisted = (
            (diagnostics / "malformed-probe-stdout.json").read_text(encoding="utf-8")
            + (diagnostics / "malformed-probe-stderr.log").read_text(encoding="utf-8")
            + captured_stderr.getvalue()
        )
        self.assertNotIn("stdout-secret", persisted)
        self.assertNotIn("stderr-secret", persisted)
        self.assertIn("\ufffd", persisted)

    def test_harness_digest_covers_every_heavy_gate_source(self) -> None:
        scripts_dir = Path(__file__).resolve().parent
        baseline = rq.harness_source_sha256(scripts_dir)
        for name in rq.HARNESS_SOURCES:
            mirror = self.tmp / f"mirror-{name}"
            mirror.mkdir()
            for other in rq.HARNESS_SOURCES:
                shutil.copyfile(scripts_dir / other, mirror / other)
            (mirror / name).write_bytes(
                (scripts_dir / name).read_bytes() + b"\n# drift\n"
            )
            self.assertNotEqual(baseline, rq.harness_source_sha256(mirror))

    def test_local_harness_must_match_the_exact_bridge_source(self) -> None:
        import subprocess

        repository = self.tmp / "harness-repository"
        scripts_dir = repository / "scripts"
        scripts_dir.mkdir(parents=True)
        source_scripts = Path(__file__).resolve().parent
        for name in rq.HARNESS_SOURCES:
            shutil.copyfile(source_scripts / name, scripts_dir / name)
        subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(repository), "add", "scripts"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-q",
                "-m",
                "harness",
            ],
            check=True,
        )
        bridge_sha = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        rq.require_harness_matches_bridge_source(scripts_dir, bridge_sha)
        (scripts_dir / rq.HARNESS_SOURCES[0]).write_text("drift\n", encoding="utf-8")
        with self.assertRaises(ContractError) as ctx:
            rq.require_harness_matches_bridge_source(scripts_dir, bridge_sha)
        self.assertIn("does not match", str(ctx.exception))

    def test_qualify_rejects_an_unprovenanced_local_candidate(self) -> None:
        parser = rq._build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "qualify",
                        "--candidate-dist",
                        str(self.candidate),
                        "--output-attestation",
                        str(self.tmp / "a.json"),
                    ]
                )

    def test_verify_run_cli_defaults_to_the_mandatory_first_attempt(self) -> None:
        args = rq._build_parser().parse_args(
            [
                "verify-run",
                "--run-json",
                str(self.tmp / "run.json"),
                "--artifacts-json",
                str(self.tmp / "artifacts.json"),
                "--run-id",
                CANDIDATE_RUN_ID,
                "--workflow-path",
                rq.CANDIDATE_WORKFLOW_PATH,
                "--head-branch",
                DEFAULT_HEAD_BRANCH,
                "--artifact-name",
                rq.CANDIDATE_ARTIFACT_NAME,
            ]
        )
        self.assertEqual(args.run_attempt, 1)


if __name__ == "__main__":
    unittest.main()
