#!/usr/bin/env python3
"""Real-model browser smoke for Qwen3-ASR audio ingestion."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from multimodal_browser_smoke import resolve_file
from state_persistence_browser_smoke import (
    copy_artifacts,
    require,
    run_playwright,
    serve,
    sha256_file,
)

DEFAULT_MODEL_CACHE = (
    Path.home() / ".cache" / "llama-web-bridge" / "speech-smoke-models"
)
DEFAULT_AUDIO_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/"
    "Qwen3-ASR-Repo/asr_en.wav"
)
DEFAULT_AUDIO_SHA256 = (
    "f9b4440ac8393e47c14a6240e9739dea09b645bb1592b8f2dd48feb9666cea7f"
)
DEFAULT_EXPECTED_TEXT = (
    "Hmm. Oh, yeah, yeah. He wasn't even that big when I started listening "
    "to him, but and his solo music didn't do overly well, but he did very "
    "well when he started writing for other people."
)
MEMORY_MODES = ("wasm32", "wasm64")


def stage_file(source: Path, target: Path) -> None:
    """Stage a large fixture without copying when the filesystem permits it."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def copy_memory64_artifacts(dist_dir: Path, web_root: Path) -> None:
    for name in ("llama_webgpu_core_mem64.js", "llama_webgpu_core_mem64.wasm"):
        source = dist_dir / name
        require(source.is_file(), f"missing wasm64 bridge artifact: {source}")
        shutil.copyfile(source, web_root / name)


def write_harness(
    web_root: Path,
    *,
    expected_text: str,
    audio_sha256: str,
    memory_modes: tuple[str, ...],
) -> None:
    expected_json = json.dumps(expected_text)
    audio_sha_json = json.dumps(audio_sha256)
    memory_modes_json = json.dumps(memory_modes)
    script = f"""
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge Qwen3-ASR smoke</title>
<pre id="result">pending</pre>
<script type="module">
(async () => {{
  const result = document.getElementById('result');
  const finish = (payload) => {{
    result.textContent = JSON.stringify(payload);
    window.__smokeResult = payload;
  }};
  const assert = (condition, message) => {{
    if (!condition) throw new Error(message);
  }};
  const normalizeTranscript = (value) => String(value || '')
    .replace(/^\\s*language\\s+[^<\\r\\n]+?\\s*<asr_text>\\s*/i, '')
    .replace(/^\\s*<asr_text>\\s*/i, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();

  try {{
    assert(window.crossOriginIsolated, 'test page is not cross-origin isolated');
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    assert(
      typeof LlamaWebGpuBridge === 'function',
      'LlamaWebGpuBridge export was not registered',
    );

    const audioResponse = await fetch('/speech.wav', {{ cache: 'no-store' }});
    assert(audioResponse.ok, `audio fetch failed: ${{audioResponse.status}}`);
    const audioBytes = new Uint8Array(await audioResponse.arrayBuffer());
    assert(audioBytes.byteLength > 44, 'audio fixture is empty');

    const expected = normalizeTranscript({expected_json});
    assert(expected.length > 0, 'expected transcript is empty');
    const memoryModes = {memory_modes_json};
    const modeResults = [];

    const verifyMode = async (memoryMode, disableWorker, runtimeMode) => {{
      const useMemory64 = memoryMode === 'wasm64';
      const bridge = new LlamaWebGpuBridge({{
        disableWorker,
        logLevel: 1,
        preferMemory64: useMemory64,
        coreModuleUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.js' : undefined,
        wasmUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.wasm' : undefined,
      }});
      const startedAt = performance.now();
      try {{
        await bridge.loadModelFromUrl('/qwen3-asr-model.gguf', {{
          nCtx: 4096,
          nGpuLayers: 0,
          nThreads: 4,
          nBatch: 512,
          nUbatch: 256,
          useCache: false,
          forceRemoteFetchBackend: false,
        }});
        await bridge.loadMultimodalProjector('/qwen3-asr-mmproj.gguf');
        assert(bridge.supportsAudio(), `${{memoryMode}} ${{runtimeMode}} did not report audio support`);

        const transcribe = (signal = undefined, onToken = undefined) => bridge.createCompletion(
            'Transcribe this audio accurately.',
            {{
              nPredict: 512,
              temp: 0,
              topK: 1,
              topP: 1,
              penalty: 1,
              seed: 1,
              tokenEventEncoding: 'text',
              signal,
              onToken,
              parts: [{{ type: 'audio', bytes: audioBytes }}],
            }},
          );
        const transcripts = [];
        for (let attempt = 0; attempt < 2; attempt += 1) {{
          const output = await transcribe();
          const outputText = String(output || '').trim();
          const normalized = normalizeTranscript(outputText);
          assert(normalized.length > 0, `${{memoryMode}} ${{runtimeMode}} returned an empty transcript`);
          assert(
            normalized === expected,
            `${{memoryMode}} ${{runtimeMode}} transcript mismatch: ${{outputText}}`,
          );
          transcripts.push(outputText.slice(0, 512));

          if (attempt === 0) {{
            const controller = new AbortController();
            let cancelledOutput = '';
            let cancellationState = 'resolved';
            let cancellationRequested = false;
            let cancellationWatchdog;
            let tokenWatchdog;
            let rejectWatchdog;
            const watchdogPromise = new Promise((_, reject) => {{
              rejectWatchdog = reject;
              tokenWatchdog = setTimeout(
                () => reject(new Error('speech did not emit a cancellable token within 180 seconds')),
                180000,
              );
            }});
            try {{
              cancelledOutput = String(await Promise.race([
                transcribe(controller.signal, () => {{
                  if (cancellationRequested) return;
                  cancellationRequested = true;
                  clearTimeout(tokenWatchdog);
                  cancellationWatchdog = setTimeout(
                    () => rejectWatchdog(new Error('speech cancellation did not settle within 30 seconds')),
                    30000,
                  );
                  controller.abort();
                  bridge.cancel();
                }}),
                watchdogPromise,
              ]) || '');
            }} catch (error) {{
              const message = String(error?.message || error || '');
              assert(
                /abort|cancel|interrupt/i.test(message),
                `${{memoryMode}} ${{runtimeMode}} cancellation failed: ${{message}}`,
              );
              cancellationState = 'rejected';
            }} finally {{
              clearTimeout(tokenWatchdog);
              clearTimeout(cancellationWatchdog);
            }}
            assert(cancellationRequested, `${{memoryMode}} ${{runtimeMode}} did not emit a token to cancel`);
            assert(controller.signal.aborted, `${{memoryMode}} ${{runtimeMode}} cancellation was not requested`);
            assert(
              normalizeTranscript(cancelledOutput) !== expected,
              `${{memoryMode}} ${{runtimeMode}} ignored cancellation and returned a full transcript`,
            );
            transcripts.push(`cancel:${{cancellationState}}:${{cancelledOutput.length}}`);
          }}
        }}

        modeResults.push({{
          memoryMode,
          runtimeMode,
          elapsedMs: Math.round(performance.now() - startedAt),
          coldTranscript: transcripts[0],
          cancellation: transcripts[1],
          warmTranscript: transcripts[2],
        }});
      }} finally {{
        await bridge.dispose();
      }}
    }};

    for (const memoryMode of memoryModes) {{
      await verifyMode(memoryMode, true, 'direct');
      await verifyMode(memoryMode, false, 'worker');
    }}

    finish({{
      ok: true,
      fixture: {{
        sha256: {audio_sha_json},
        encodedByteLength: audioBytes.byteLength,
      }},
      modeResults,
    }});
  }} catch (error) {{
    finish({{
      ok: false,
      error: String(error && error.stack ? error.stack : error),
    }});
  }}
}})();
</script>
"""
    (web_root / "index.html").write_text(script, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path(os.environ.get("BRIDGE_DIST_DIR", "dist")),
        help="Directory containing built bridge artifacts.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=int(os.environ.get("LLAMA_WEBGPU_SPEECH_TIMEOUT_MS", "900000")),
        help="Browser operation timeout in milliseconds.",
    )
    parser.add_argument(
        "--model-url",
        default=os.environ.get("LLAMA_WEBGPU_SPEECH_MODEL_URL", ""),
        help="Qwen3-ASR GGUF URL.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=(
            Path(os.environ["LLAMA_WEBGPU_SPEECH_MODEL_PATH"])
            if os.environ.get("LLAMA_WEBGPU_SPEECH_MODEL_PATH")
            else None
        ),
        help="Local Qwen3-ASR GGUF path.",
    )
    parser.add_argument(
        "--model-sha256",
        default=os.environ.get("LLAMA_WEBGPU_SPEECH_MODEL_SHA256", ""),
        help="Expected model SHA-256.",
    )
    parser.add_argument(
        "--mmproj-url",
        default=os.environ.get("LLAMA_WEBGPU_SPEECH_MMPROJ_URL", ""),
        help="Qwen3-ASR projector GGUF URL.",
    )
    parser.add_argument(
        "--mmproj-path",
        type=Path,
        default=(
            Path(os.environ["LLAMA_WEBGPU_SPEECH_MMPROJ_PATH"])
            if os.environ.get("LLAMA_WEBGPU_SPEECH_MMPROJ_PATH")
            else None
        ),
        help="Local Qwen3-ASR projector GGUF path.",
    )
    parser.add_argument(
        "--mmproj-sha256",
        default=os.environ.get("LLAMA_WEBGPU_SPEECH_MMPROJ_SHA256", ""),
        help="Expected projector SHA-256.",
    )
    parser.add_argument(
        "--audio-url",
        default=os.environ.get("LLAMA_WEBGPU_SPEECH_AUDIO_URL", DEFAULT_AUDIO_URL),
        help="WAV fixture URL used when --audio-path is omitted.",
    )
    parser.add_argument(
        "--audio-path",
        type=Path,
        default=(
            Path(os.environ["LLAMA_WEBGPU_SPEECH_AUDIO_PATH"])
            if os.environ.get("LLAMA_WEBGPU_SPEECH_AUDIO_PATH")
            else None
        ),
        help="Local WAV fixture path.",
    )
    parser.add_argument(
        "--audio-sha256",
        default=os.environ.get(
            "LLAMA_WEBGPU_SPEECH_AUDIO_SHA256", DEFAULT_AUDIO_SHA256
        ),
        help="Expected WAV SHA-256.",
    )
    parser.add_argument(
        "--expect",
        default=os.environ.get(
            "LLAMA_WEBGPU_SPEECH_EXPECTED_TEXT", DEFAULT_EXPECTED_TEXT
        ),
        help="Expected normalized transcript.",
    )
    parser.add_argument(
        "--memory-mode",
        choices=("all", *MEMORY_MODES),
        default=os.environ.get("LLAMA_WEBGPU_SPEECH_MEMORY_MODE", "all"),
        help="WASM memory variant to validate (default: both).",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path(
            os.environ.get("LLAMA_WEBGPU_SPEECH_MODEL_CACHE", DEFAULT_MODEL_CACHE)
        ),
        help="Cache directory used for downloaded model files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=(
            Path(os.environ["LLAMA_WEBGPU_SPEECH_ARTIFACTS_DIR"])
            if os.environ.get("LLAMA_WEBGPU_SPEECH_ARTIFACTS_DIR")
            else None
        ),
        help="Directory for JSON, console, and screenshot diagnostics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dist_dir = args.dist_dir.resolve()
    require(dist_dir.is_dir(), f"dist directory does not exist: {dist_dir}")
    model_path = resolve_file(
        path=args.model_path,
        url=args.model_url,
        expected_sha256=args.model_sha256,
        cache_dir=args.model_cache_dir.expanduser().resolve(),
        label="Qwen3-ASR model",
    )
    mmproj_path = resolve_file(
        path=args.mmproj_path,
        url=args.mmproj_url,
        expected_sha256=args.mmproj_sha256,
        cache_dir=args.model_cache_dir.expanduser().resolve(),
        label="Qwen3-ASR projector",
    )
    audio_path = resolve_file(
        path=args.audio_path,
        url=args.audio_url,
        expected_sha256=args.audio_sha256,
        cache_dir=args.model_cache_dir.expanduser().resolve(),
        label="speech fixture",
    )
    actual_audio_sha256 = sha256_file(audio_path)
    require(bool(args.expect.strip()), "expected transcript is required")
    memory_modes = MEMORY_MODES if args.memory_mode == "all" else (args.memory_mode,)
    artifacts_dir = args.artifacts_dir.resolve() if args.artifacts_dir else None

    with tempfile.TemporaryDirectory(prefix="llama-web-bridge-speech-") as tmp:
        web_root = Path(tmp)
        copy_artifacts(dist_dir, web_root)
        if "wasm64" in memory_modes:
            copy_memory64_artifacts(dist_dir, web_root)
        stage_file(model_path, web_root / "qwen3-asr-model.gguf")
        stage_file(mmproj_path, web_root / "qwen3-asr-mmproj.gguf")
        stage_file(audio_path, web_root / "speech.wav")
        write_harness(
            web_root,
            expected_text=args.expect,
            audio_sha256=actual_audio_sha256,
            memory_modes=memory_modes,
        )
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(web_root / "index.html", artifacts_dir / "index.html")
        with serve(web_root) as url:
            payload = asyncio.run(
                run_playwright(
                    url,
                    args.timeout_ms,
                    artifacts_dir,
                    artifact_prefix="speech-to-text-smoke",
                )
            )

    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload.get("ok") is not True:
        return 1
    mode_results = payload.get("modeResults")
    require(
        isinstance(mode_results, list) and len(mode_results) == len(memory_modes) * 2,
        "speech-to-text mode results are incomplete",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"speech-to-text browser smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
