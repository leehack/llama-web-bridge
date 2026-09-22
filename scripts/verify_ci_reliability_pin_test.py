#!/usr/bin/env python3
"""Contract tests for role-aware model pin, URL, and revision parity checks."""

from __future__ import annotations

import unittest
from unittest import mock

from release_qualification import EXPECTED_MODEL_PINS
import verify_ci_reliability as verifier

SPEECH_REVISION = "928ab958557df9aa2ef1c93e0e83c7ad0933fae2"
TTS_REVISION = "ca27d74bc954b73dadab5b71ca265d87fc861a7c"
ASR_REPO = f"https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF/resolve/{SPEECH_REVISION}"
TTS_REPO = (
    "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/"
    f"{TTS_REVISION}"
)
MODEL_URLS = {
    "LLAMA_WEBGPU_SMOKE_MODEL": (
        "https://huggingface.co/aladar/llama-2-tiny-random-GGUF/resolve/main/"
        "llama-2-tiny-random.gguf"
    ),
    "LLAMA_WEBGPU_MULTIMODAL_MODEL": (
        "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/"
        "Qwen3.5-0.8B-Q4_K_M.gguf"
    ),
    "LLAMA_WEBGPU_MULTIMODAL_MMPROJ": (
        "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-F16.gguf"
    ),
    "LLAMA_WEBGPU_SPEECH_MODEL": f"{ASR_REPO}/Qwen3-ASR-0.6B-Q8_0.gguf?download=true",
    "LLAMA_WEBGPU_SPEECH_MMPROJ": (
        f"{ASR_REPO}/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf?download=true"
    ),
    "LLAMA_WEBGPU_SPEECH_AUDIO": (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
    ),
    "LLAMA_WEBGPU_TTS_MODEL": (
        f"{TTS_REPO}/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf?download=true"
    ),
    "LLAMA_WEBGPU_TTS_MMPROJ": (
        f"{TTS_REPO}/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf?download=true"
    ),
}
BUILD_ROLES = (
    "LLAMA_WEBGPU_SMOKE_MODEL",
    "LLAMA_WEBGPU_MULTIMODAL_MODEL",
    "LLAMA_WEBGPU_MULTIMODAL_MMPROJ",
    "LLAMA_WEBGPU_SPEECH_MODEL",
    "LLAMA_WEBGPU_SPEECH_MMPROJ",
    "LLAMA_WEBGPU_TTS_MODEL",
    "LLAMA_WEBGPU_TTS_MMPROJ",
)
QUALIFICATION_ROLES = (
    "LLAMA_WEBGPU_SPEECH_MODEL",
    "LLAMA_WEBGPU_SPEECH_MMPROJ",
    "LLAMA_WEBGPU_SPEECH_AUDIO",
    "LLAMA_WEBGPU_TTS_MODEL",
    "LLAMA_WEBGPU_TTS_MMPROJ",
)
DOCUMENTED_ROLES = (
    ("--model-url", MODEL_URLS["LLAMA_WEBGPU_SMOKE_MODEL"], "--model-sha256",
     "state_smoke_model_sha256"),
    ("--model-path", "/path/to/Qwen3.5-0.8B-Q4_K_M.gguf", "--model-sha256",
     "multimodal_model_sha256"),
    ("--mmproj-path", "/path/to/mmproj-F16.gguf", "--mmproj-sha256",
     "multimodal_mmproj_sha256"),
    ("--model-path", "/path/to/Qwen3-ASR-0.6B-Q8_0.gguf", "--model-sha256",
     "speech_model_sha256"),
    ("--mmproj-path", "/path/to/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf", "--mmproj-sha256",
     "speech_mmproj_sha256"),
    ("--model-path", "/path/to/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf", "--model-sha256",
     "tts_model_sha256"),
    ("--mmproj-path", "/path/to/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
     "--mmproj-sha256", "tts_mmproj_sha256"),
)
CI_PATH = ".github/workflows/ci.yml"
CANDIDATE_PATH = ".github/workflows/bridge_candidate.yml"
QUALIFICATION_PATH = ".github/workflows/bridge_qualification.yml"


def workflow(roles, urls=None, pins=None) -> str:
    resolved_urls = {**MODEL_URLS, **(urls or {})}
    resolved_pins = {**EXPECTED_MODEL_PINS, **(pins or {})}
    lines = ["name: fixture", "on: push", "env:"]
    for role in roles:
        name = verifier.CANONICAL_MODEL_PIN_NAMES[f"{role}_SHA256"]
        lines.append(f"      {role}_URL: {resolved_urls[role]}")
        lines.append(f"      {role}_SHA256: {resolved_pins[name]}")
    return "\n".join(lines) + "\n"


def markdown(pins=None, roles=DOCUMENTED_ROLES) -> str:
    resolved = {**EXPECTED_MODEL_PINS, **(pins or {})}
    lines = ["```bash", "python3 scripts/example_browser_smoke.py \\"]
    for role_flag, value, pin_flag, name in roles:
        lines.append(f"  {role_flag} {value} \\")
        lines.append(f"  {pin_flag} {resolved[name]} \\")
    lines.append("  --artifacts-dir /tmp/example")
    lines.append("```")
    return "\n".join(lines) + "\n"


def collect_errors(ci=None, candidate=None, qualification=None, contributing=None):
    contents = {
        CI_PATH: workflow(BUILD_ROLES) if ci is None else ci,
        CANDIDATE_PATH: workflow(BUILD_ROLES) if candidate is None else candidate,
        QUALIFICATION_PATH: (
            workflow(QUALIFICATION_ROLES) if qualification is None else qualification
        ),
    }
    errors: list[str] = []
    roles = {
        path: verifier.extract_model_sha_pin_roles(path, content, errors)
        for path, content in contents.items()
    }
    urls = {
        path: verifier.extract_model_urls(path, content, errors)
        for path, content in contents.items()
    }
    verifier.require_qualification_model_sha_pin_roles(
        roles[QUALIFICATION_PATH], errors
    )
    verifier.require_canonical_model_sha_pins(roles, errors)
    verifier.require_paired_model_urls_and_pins(urls, roles, errors)
    verifier.require_identical_model_urls(urls, errors)
    verifier.require_pinned_model_url_revisions(urls, errors)
    verifier.require_markdown_model_pin_roles(
        "CONTRIBUTING.md",
        markdown() if contributing is None else contributing,
        verifier.model_file_name_roles(urls, errors),
        errors,
    )
    return errors


class ModelPinParityTest(unittest.TestCase):
    def assertRejected(self, errors, fragment):
        self.assertTrue(errors, "expected at least one error")
        self.assertTrue(
            any(fragment in error for error in errors),
            f"no error contained {fragment!r}: {errors}",
        )

    def test_fixture_baseline_accepted(self):
        self.assertEqual(collect_errors(), [])

    def test_repository_tree_accepted(self):
        errors = collect_errors(
            ci=(verifier.ROOT / CI_PATH).read_text(encoding="utf-8"),
            candidate=(verifier.ROOT / CANDIDATE_PATH).read_text(encoding="utf-8"),
            qualification=(verifier.ROOT / QUALIFICATION_PATH).read_text(
                encoding="utf-8"
            ),
            contributing=(verifier.ROOT / "CONTRIBUTING.md").read_text(
                encoding="utf-8"
            ),
        )
        self.assertEqual(errors, [])

    def test_markdown_only_role_swap_rejected(self):
        swapped = {
            "speech_model_sha256": EXPECTED_MODEL_PINS["speech_mmproj_sha256"],
            "speech_mmproj_sha256": EXPECTED_MODEL_PINS["speech_model_sha256"],
        }
        errors = collect_errors(contributing=markdown(pins=swapped))
        self.assertRejected(errors, "canonical speech_model_sha256 is")
        self.assertRejected(errors, "canonical speech_mmproj_sha256 is")

    def test_identical_cross_workflow_swap_rejected(self):
        swapped = {
            "speech_model_sha256": EXPECTED_MODEL_PINS["tts_model_sha256"],
            "tts_model_sha256": EXPECTED_MODEL_PINS["speech_model_sha256"],
        }
        errors = collect_errors(
            ci=workflow(BUILD_ROLES, pins=swapped),
            candidate=workflow(BUILD_ROLES, pins=swapped),
        )
        self.assertRejected(errors, "binds LLAMA_WEBGPU_SPEECH_MODEL_SHA256")
        self.assertRejected(errors, "binds LLAMA_WEBGPU_TTS_MODEL_SHA256")

    def test_bridge_qualification_pin_drift_rejected(self):
        drifted = {"speech_audio_sha256": "0" * 64}
        errors = collect_errors(
            qualification=workflow(QUALIFICATION_ROLES, pins=drifted)
        )
        self.assertRejected(errors, "canonical speech_audio_sha256 is")

    def test_url_drift_rejected(self):
        drifted = {
            "LLAMA_WEBGPU_MULTIMODAL_MMPROJ": (
                "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/"
                "mmproj-F32.gguf"
            )
        }
        errors = collect_errors(candidate=workflow(BUILD_ROLES, urls=drifted))
        self.assertRejected(
            errors, "LLAMA_WEBGPU_MULTIMODAL_MMPROJ_URL requests different bytes"
        )

    def test_revision_drift_rejected(self):
        drifted = {
            "LLAMA_WEBGPU_TTS_MODEL": MODEL_URLS["LLAMA_WEBGPU_TTS_MODEL"].replace(
                TTS_REVISION, "f" * 40
            )
        }
        errors = collect_errors(
            qualification=workflow(QUALIFICATION_ROLES, urls=drifted)
        )
        self.assertRejected(
            errors, "LLAMA_WEBGPU_TTS_MODEL_URL requests different bytes"
        )

    def test_dropped_qualification_role_rejected(self):
        errors = collect_errors(
            qualification=workflow(
                tuple(
                    role
                    for role in QUALIFICATION_ROLES
                    if role != "LLAMA_WEBGPU_TTS_MMPROJ"
                )
            )
        )
        self.assertRejected(errors, "declares model SHA-256 env keys")

    def test_extra_qualification_role_rejected(self):
        errors = collect_errors(
            qualification=workflow(
                QUALIFICATION_ROLES + ("LLAMA_WEBGPU_SMOKE_MODEL",)
            )
        )
        self.assertRejected(errors, "declares model SHA-256 env keys")

    def test_newly_unversioned_url_rejected(self):
        unversioned = {
            "LLAMA_WEBGPU_TTS_MODEL": (
                "https://mirror.example/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf"
            )
        }
        errors = collect_errors(
            ci=workflow(BUILD_ROLES, urls=unversioned),
            candidate=workflow(BUILD_ROLES, urls=unversioned),
            qualification=workflow(QUALIFICATION_ROLES, urls=unversioned),
        )
        self.assertRejected(errors, "carrying no revision segment")

    def test_duplicate_model_url_rejected(self):
        content = workflow(QUALIFICATION_ROLES) + (
            "      LLAMA_WEBGPU_SPEECH_AUDIO_URL: "
            "https://mirror.example/asr_en.wav\n"
        )
        errors = collect_errors(qualification=content)
        self.assertRejected(errors, "redefines model URL env key")

    def test_shared_model_file_name_rejected(self):
        collided = {
            "LLAMA_WEBGPU_MULTIMODAL_MMPROJ": (
                f"{ASR_REPO}/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf?download=true"
            )
        }
        errors = collect_errors(
            ci=workflow(BUILD_ROLES, urls=collided),
            candidate=workflow(BUILD_ROLES, urls=collided),
        )
        self.assertRejected(errors, "also downloads")

    def test_markdown_missing_documented_pin_rejected(self):
        errors = collect_errors(contributing=markdown(roles=DOCUMENTED_ROLES[:-1]))
        self.assertRejected(errors, "documented model SHA-256 pins with a role")

    def test_markdown_duplicate_role_rejected(self):
        errors = collect_errors(
            contributing=markdown(roles=DOCUMENTED_ROLES + DOCUMENTED_ROLES[:1])
        )
        self.assertRejected(errors, "pins canonical state_smoke_model_sha256 again")

    def test_uncanonical_pin_name_rejected(self):
        with mock.patch.dict(
            verifier.CANONICAL_MODEL_PIN_NAMES,
            {"LLAMA_WEBGPU_ABSENT_MODEL_SHA256": "absent_model_sha256"},
        ):
            errors = collect_errors()
        self.assertRejected(errors, "EXPECTED_MODEL_PINS does not declare")

    def test_url_trailing_comment_accepted(self):
        content = workflow(QUALIFICATION_ROLES).replace(
            f"      LLAMA_WEBGPU_SPEECH_AUDIO_URL: "
            f"{MODEL_URLS['LLAMA_WEBGPU_SPEECH_AUDIO']}\n",
            f"      LLAMA_WEBGPU_SPEECH_AUDIO_URL: "
            f"{MODEL_URLS['LLAMA_WEBGPU_SPEECH_AUDIO']}  # not a Hugging Face object\n",
        )
        self.assertEqual(collect_errors(qualification=content), [])

    def test_newly_mutable_revision_rejected(self):
        mutable = {
            role: MODEL_URLS[role].replace(SPEECH_REVISION, "main")
            for role in ("LLAMA_WEBGPU_SPEECH_MODEL", "LLAMA_WEBGPU_SPEECH_MMPROJ")
        }
        errors = collect_errors(
            ci=workflow(BUILD_ROLES, urls=mutable),
            candidate=workflow(BUILD_ROLES, urls=mutable),
            qualification=workflow(QUALIFICATION_ROLES, urls=mutable),
        )
        self.assertRejected(errors, "resolving through a mutable revision")

    def test_missing_canonical_role_rejected(self):
        errors = collect_errors(
            qualification=workflow(
                tuple(
                    role
                    for role in QUALIFICATION_ROLES
                    if role != "LLAMA_WEBGPU_SPEECH_AUDIO"
                )
            )
        )
        self.assertRejected(errors, "speech_audio_sha256")

    def test_unnamed_role_key_rejected(self):
        content = workflow(QUALIFICATION_ROLES).replace(
            "LLAMA_WEBGPU_SPEECH_AUDIO_SHA256", "LLAMA_WEBGPU_SPEECH_SAMPLE_SHA256"
        )
        errors = collect_errors(qualification=content)
        self.assertRejected(
            errors, "CANONICAL_MODEL_PIN_NAMES does not name"
        )

    def test_url_without_pin_rejected(self):
        content = workflow(QUALIFICATION_ROLES).replace(
            f"      LLAMA_WEBGPU_SPEECH_AUDIO_SHA256: "
            f"{EXPECTED_MODEL_PINS['speech_audio_sha256']}\n",
            "",
        )
        errors = collect_errors(qualification=content)
        self.assertRejected(errors, "with no matching <ROLE>_SHA256 pin")

    def test_pin_without_url_rejected(self):
        content = workflow(QUALIFICATION_ROLES).replace(
            f"      LLAMA_WEBGPU_SPEECH_AUDIO_URL: "
            f"{MODEL_URLS['LLAMA_WEBGPU_SPEECH_AUDIO']}\n",
            "",
        )
        errors = collect_errors(qualification=content)
        self.assertRejected(errors, "with no matching <ROLE>_URL")

    def test_markdown_pin_without_named_file_rejected(self):
        content = markdown().replace(
            "  --model-path /path/to/Qwen3-ASR-0.6B-Q8_0.gguf \\\n", ""
        )
        errors = collect_errors(contributing=content)
        self.assertRejected(errors, "no preceding --model-url")

    def test_markdown_unknown_file_rejected(self):
        content = markdown().replace(
            "/path/to/Qwen3-ASR-0.6B-Q8_0.gguf", "/path/to/Qwen3-ASR-0.6B-Q4_K_M.gguf"
        )
        errors = collect_errors(contributing=content)
        self.assertRejected(errors, "Qwen3-ASR-0.6B-Q4_K_M.gguf, which")


if __name__ == "__main__":
    unittest.main()
