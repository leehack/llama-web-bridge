#!/usr/bin/env python3
"""Static API contract checks for versioned Web text-to-speech support."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = (ROOT / "src" / "llama_webgpu_core.cpp").read_text(encoding="utf-8")
TTS = (ROOT / "src" / "llama_webgpu_tts.cpp").read_text(encoding="utf-8")
HEADER = (ROOT / "src" / "llama_webgpu_tts.h").read_text(encoding="utf-8")
JS = (ROOT / "js" / "src" / "llama_webgpu_bridge.js").read_text(encoding="utf-8")
DTS = (ROOT / "js" / "src" / "llama_webgpu_bridge.d.ts").read_text(encoding="utf-8")
CMAKE = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
API_DOCS = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
SMOKE = (ROOT / "scripts" / "text_to_speech_browser_smoke.py").read_text(
    encoding="utf-8"
)

NATIVE_EXPORTS = (
    "llamadart_webgpu_tts_api_version",
    "llamadart_webgpu_tts_info_json",
    "llamadart_webgpu_tts_start",
    "llamadart_webgpu_tts_step",
    "llamadart_webgpu_tts_progress_json",
    "llamadart_webgpu_tts_write_pcm",
    "llamadart_webgpu_tts_reset",
)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    errors: list[str] = []

    require(
        "LLAMADART_WEBGPU_TTS_API_VERSION = 1" in HEADER,
        "TTS wrapper API must have an explicit version",
        errors,
    )
    for symbol in NATIVE_EXPORTS:
        require(symbol in CORE, f"missing core wrapper {symbol}", errors)
        require(f"_{symbol}" in CMAKE, f"missing exported symbol _{symbol}", errors)

    require(
        "mtmd_helper_gen_audio_init" in TTS
        and "mtmd_helper_gen_audio_step_prompt" in TTS
        and "mtmd_helper_gen_audio_step_gen" in TTS
        and "mtmd_helper_gen_audio_get_output" in TTS,
        "TTS wrapper must use the upstream stepwise generated-audio API",
        errors,
    )
    require(
        "llama_set_embeddings(tts->llama, true)" in TTS
        and "llama_set_embeddings(tts->llama, false)" in TTS,
        "TTS must scope embedding-output mode to the active task",
        errors,
    )
    require(
        "g_generation_active || g_tts_active" in CORE,
        "shared llama.cpp context operations must reject TTS overlap",
        errors,
    )
    require(
        re.search(
            r"llamadart_webgpu_tts_start\(.*?g_cancel_requested = false;.*?"
            r"llama_webgpu_tts_start\(g_tts, request\)",
            CORE,
            re.DOTALL,
        )
        is not None
        and re.search(
            r"llamadart_webgpu_tts_reset\(\) \{.*?g_cancel_requested = false;",
            CORE,
            re.DOTALL,
        )
        is not None,
        "TTS reset/start must clear the shared llama.cpp cancellation latch",
        errors,
    )

    require(
        len(re.findall(r"\basync\s+synthesizeSpeech\s*\(", JS)) >= 2,
        "expected direct-runtime and public synthesizeSpeech methods",
        errors,
    )
    require(
        "method === 'synthesizeSpeech'" in JS
        and "event: 'progress'" in JS
        and "[value.pcm.buffer]" in JS,
        "worker TTS must forward progress and transfer PCM without copying it back",
        errors,
    )
    require(
        "Do not implicitly repeat a potentially long synthesis" in JS,
        "worker synthesis failures must not silently repeat on the main thread",
        errors,
    )
    require(
        "interface TextToSpeechCapabilities" in DTS
        and "interface TextToSpeechOptions" in DTS
        and "interface TextToSpeechResult" in DTS,
        "TypeScript declarations must expose TTS capabilities, options, and result",
        errors,
    )
    require(
        "getTextToSpeechCapabilities" in API_DOCS
        and "synthesizeSpeech" in API_DOCS
        and "wasm64" in API_DOCS,
        "public API docs must document TTS and its memory64 requirement",
        errors,
    )
    require(
        "Text-to-speech" in README and "text_to_speech_browser_smoke.py" in README,
        "README must document TTS and its real-model smoke",
        errors,
    )
    require(
        'RUNTIME_MODES = ("direct", "worker")' in SMOKE
        and '"wasm64" in memory_modes' in SMOKE
        and "model_sha256" in SMOKE
        and "mmproj_sha256" in SMOKE
        and 'parser.add_argument("--gpu-layers"' in SMOKE
        and 'parser.add_argument("--speaker-audio-path"' in SMOKE
        and "speakerReferenceTested" in SMOKE
        and "output.pcm instanceof Float32Array" in SMOKE,
        "real-model smoke must cover direct/worker memory64 PCM with pinned assets, optional speaker reference, and selectable GPU offload",
        errors,
    )
    require(
        "AbortController" in SMOKE
        and "preAbortedTested" in SMOKE
        and "cancellationTested" in SMOKE
        and "unloadTested" in SMOKE
        and "runtime reuse after cancellation" in SMOKE,
        "real-model smoke must validate pre-abort, active cancellation, warm reuse, and projector unload",
        errors,
    )
    require(
        "const mmprojPath = this._mmProjPath;" in JS
        and "this._deleteFsFile(mmprojPath);" in JS
        and re.search(
            r"async unloadMultimodalProjector\(\).*?if \(rc !== 0\).*?"
            r"this\._deleteFsFile\(mmprojPath\);",
            JS,
            re.DOTALL,
        )
        is not None,
        "projector unload must preserve state on native rejection and release the WasmFS file after success",
        errors,
    )
    require(
        re.search(
            r"async synthesizeSpeech\(options = \{\}\).*?"
            r"if \(this\._textToSpeechActive\).*?"
            r"this\._ensureTextToSpeechDir\(\)",
            JS,
            re.DOTALL,
        )
        is not None,
        "direct TTS must reserve the operation before writing temporary speaker input",
        errors,
    )

    if errors:
        print("Text-to-speech API contract failed:", file=sys.stderr)
        for item in errors:
            print(f"- {item}", file=sys.stderr)
        return 1

    print("Text-to-speech API contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
