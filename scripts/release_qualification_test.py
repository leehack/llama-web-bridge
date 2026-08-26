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
import wave
import warnings
import zipfile
from pathlib import Path

import release_qualification as rq
from generate_release_manifest import ARTIFACTS, generate
from release_contract import BRIDGE_REPOSITORY, ContractError


BRIDGE_SHA = "565c8396597ea7c0fb4e8d5d966da8d884b156d8"
UPSTREAM_COMMIT = "bb4caa7540188872173c44d161602d9271386413"
NATIVE_MANIFEST_SHA = (
    "2e5d29d7f98f0d71e75d3fa63b7c55f3b2a7933247cc34ea2b1c5e053d142452"
)
NATIVE_COMMIT = "1" * 40
CANDIDATE_RUN_ID = "32919086955"
CANDIDATE_RUN_URL = (
    f"https://github.com/{BRIDGE_REPOSITORY}/actions/runs/{CANDIDATE_RUN_ID}"
)
CORRELATION_ID = "llamadart-pin:run-123"
DEFAULT_HEAD_BRANCH = "main"
DEFAULT_HEAD_SHA = "a" * 40


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
                "cold_transcript": "fixture transcript",
                "warm_transcript": "fixture transcript",
                "cancellation_result": "cancel:AbortError:0",
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
                "truncated": True,
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

    def test_artifact_archive_rejects_path_escape_and_duplicate_members(self) -> None:
        for name, members in (
            ("escape.zip", (("../escape", b"bad"),)),
            ("nested.zip", (("nested/file", b"bad"),)),
            ("duplicate.zip", (("file", b"one"), ("file", b"two"))),
        ):
            archive = self.tmp / name
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(archive, "w") as handle:
                    for member, data in members:
                        handle.writestr(member, data)
            destination = self.tmp / f"extract-{name}"
            with self.subTest(name=name), self.assertRaises(ContractError):
                rq._extract_flat_artifact_archive(archive, destination)

    def test_artifact_archive_extracts_only_flat_regular_files(self) -> None:
        archive = self.tmp / "valid.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("manifest.json", b"{}\n")
        destination = self.tmp / "valid-extract"
        rq._extract_flat_artifact_archive(archive, destination)
        self.assertEqual((destination / "manifest.json").read_bytes(), b"{}\n")

        real_destination = self.tmp / "real-artifact-destination"
        real_destination.mkdir()
        linked_destination = self.tmp / "linked-artifact-destination"
        linked_destination.symlink_to(real_destination, target_is_directory=True)
        with self.assertRaises(ContractError):
            rq._extract_flat_artifact_archive(archive, linked_destination)

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


if __name__ == "__main__":
    unittest.main()
