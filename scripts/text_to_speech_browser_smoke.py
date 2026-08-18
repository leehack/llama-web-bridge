#!/usr/bin/env python3
"""Real-model browser smoke for Qwen3-TTS audio generation."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile

from speech_to_text_browser_smoke import (
    MEMORY_MODES,
    copy_memory64_artifacts,
    stage_file,
)
from state_persistence_browser_smoke import (
    copy_artifacts,
    require,
    serve,
    sha256_file,
    write_json_artifact,
    write_text_artifact,
)


RUNTIME_MODES = ("direct", "worker")


def write_harness(
    web_root: Path,
    *,
    prompt: str,
    model_sha256: str,
    mmproj_sha256: str,
    speaker_audio_sha256: str | None,
    memory_modes: tuple[str, ...],
    runtime_modes: tuple[str, ...],
    max_frames: int,
    gpu_layers: int,
    test_cancellation: bool,
) -> None:
    script = f"""
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge Qwen3-TTS smoke</title>
<pre id="result">pending</pre>
<script type="module">
(async () => {{
  const resultNode = document.getElementById('result');
  const finish = (payload) => {{
    resultNode.textContent = JSON.stringify(payload);
    window.__smokeResult = payload;
  }};
  const setStage = (stage) => {{
    window.__smokeStage = stage;
    console.log(`tts-smoke-stage:${{stage}}`);
  }};
    const assert = (condition, message) => {{
      if (!condition) throw new Error(message);
    }};
    const pcmToWavBase64 = (pcm, sampleRate) => {{
      const bytes = new Uint8Array(44 + pcm.length * 2);
      const view = new DataView(bytes.buffer);
      const writeText = (offset, value) => {{
        for (let index = 0; index < value.length; index += 1) {{
          view.setUint8(offset + index, value.charCodeAt(index));
        }}
      }};
      writeText(0, 'RIFF');
      view.setUint32(4, bytes.length - 8, true);
      writeText(8, 'WAVE');
      writeText(12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeText(36, 'data');
      view.setUint32(40, pcm.length * 2, true);
      for (let index = 0; index < pcm.length; index += 1) {{
        const sample = Math.max(-1, Math.min(1, pcm[index]));
        view.setInt16(
          44 + index * 2,
          sample < 0 ? Math.round(sample * 32768) : Math.round(sample * 32767),
          true,
        );
      }}
      let binary = '';
      for (let offset = 0; offset < bytes.length; offset += 0x8000) {{
        binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
      }}
      return btoa(binary);
    }};
  try {{
    assert(window.crossOriginIsolated, 'test page is not cross-origin isolated');
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    assert(typeof LlamaWebGpuBridge === 'function', 'bridge export was not registered');
    const speakerAudio = {json.dumps(speaker_audio_sha256 is not None)}
      ? new Uint8Array(await (await fetch('/speaker-reference.wav')).arrayBuffer())
      : null;
    if (speakerAudio) {{
      assert(speakerAudio.byteLength > 44, 'speaker-reference WAV is empty');
    }}

    const modeResults = [];
    const memoryModes = {json.dumps(memory_modes)};
    const runtimeModes = {json.dumps(runtime_modes)};
    for (const memoryMode of memoryModes) {{
      for (const runtimeMode of runtimeModes) {{
        const useMemory64 = memoryMode === 'wasm64';
        const bridge = new LlamaWebGpuBridge({{
          disableWorker: runtimeMode === 'direct',
          logLevel: 2,
          preferMemory64: useMemory64,
          coreModuleUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.js' : undefined,
          wasmUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.wasm' : undefined,
          workerTextToSpeechTimeoutMs: 1200000,
        }});
        const startedAt = performance.now();
        try {{
          setStage(`${{memoryMode}}:${{runtimeMode}}:load-model`);
          const modelStartedAt = performance.now();
          await bridge.loadModelFromUrl('/qwen3-tts-model.gguf', {{
            nCtx: 4096,
            nGpuLayers: {gpu_layers},
            nThreads: 4,
            nBatch: 512,
            nUbatch: 256,
            useCache: false,
            forceRemoteFetchBackend: false,
            progressCallback: (event) => {{
              const loaded = Number(event.loaded) || 0;
              const total = Number(event.total) || 0;
              window.__smokeStage = `${{memoryMode}}:${{runtimeMode}}:load-model:${{loaded}}/${{total}}`;
            }},
          }});
          const modelLoadMs = Math.round(performance.now() - modelStartedAt);
          setStage(`${{memoryMode}}:${{runtimeMode}}:load-projector`);
          const projectorStartedAt = performance.now();
          await bridge.loadMultimodalProjector('/qwen3-tts-mmproj.gguf');
          const projectorLoadMs = Math.round(performance.now() - projectorStartedAt);
          setStage(`${{memoryMode}}:${{runtimeMode}}:capabilities`);
          const capabilities = await bridge.getTextToSpeechCapabilities();
          assert(capabilities.apiVersion === 1, 'unexpected TTS API version');
          assert(capabilities.supported === true, `TTS unsupported: ${{capabilities.reason}}`);
          assert(capabilities.sampleRate === 24000, 'unexpected sample rate');
          assert(capabilities.channels === 1, 'unexpected channel count');
          if (speakerAudio) {{
            assert(
              capabilities.supportsSpeakerReference === true,
              'loaded projector does not report speaker-reference support',
            );
          }}

          const preAbortedController = new AbortController();
          preAbortedController.abort();
          try {{
            await bridge.synthesizeSpeech({{
              text: 'This pre-aborted task must not start.',
              signal: preAbortedController.signal,
            }});
            throw new Error('pre-aborted synthesis unexpectedly completed');
          }} catch (error) {{
            assert(error?.name === 'AbortError', `unexpected pre-abort error: ${{error}}`);
          }}

          const progress = [];
          setStage(`${{memoryMode}}:${{runtimeMode}}:synthesize`);
          const synthesisStartedAt = performance.now();
          const output = await bridge.synthesizeSpeech({{
            text: {json.dumps(prompt)},
            language: 'en',
            speakerAudio: speakerAudio || undefined,
            promptBatchSize: 512,
            maxFrames: {max_frames},
            topK: 40,
            topP: 0.95,
            minP: 0,
            temperature: 0.8,
            seed: 1,
            onProgress: (event) => progress.push({{
              state: event.state,
              promptTokensRemaining: event.promptTokensRemaining,
              framesGenerated: event.framesGenerated,
            }}),
          }});
          const synthesisMs = Math.round(performance.now() - synthesisStartedAt);
          setStage(`${{memoryMode}}:${{runtimeMode}}:validate-output`);
          assert(output.pcm instanceof Float32Array, 'PCM output is not Float32Array');
          assert(output.sampleRate === 24000, 'result sample rate mismatch');
          assert(output.channels === 1, 'result channel mismatch');
          assert(output.sampleCount === output.pcm.length, 'sample count mismatch');
          assert(output.sampleCount >= 2400, 'synthesized audio is too short');
          let peak = 0;
          let energy = 0;
          for (const sample of output.pcm) {{
            assert(Number.isFinite(sample), 'PCM contains a non-finite sample');
            peak = Math.max(peak, Math.abs(sample));
            energy += sample * sample;
          }}
          const rms = Math.sqrt(energy / output.pcm.length);
          assert(peak > 0.001, `synthesized audio peak is too low: ${{peak}}`);
          assert(rms > 0.0001, `synthesized audio RMS is too low: ${{rms}}`);
          assert(progress.some((event) => event.state === 2), 'generation progress was not observed');
          assert(progress.some((event) => event.state === 3), 'completion progress was not observed');

          let cancellationTested = false;
          let reuseSampleCount = 0;
          if ({json.dumps(test_cancellation)}) {{
            setStage(`${{memoryMode}}:${{runtimeMode}}:cancel`);
            const controller = new AbortController();
            let abortRequested = false;
            try {{
              await bridge.synthesizeSpeech({{
                text: 'Cancel this browser speech task.',
                language: 'en',
                maxFrames: 24,
                seed: 2,
                signal: controller.signal,
                onProgress: (event) => {{
                  if (!abortRequested && event.state === 2 && event.framesGenerated >= 1) {{
                    abortRequested = true;
                    controller.abort();
                  }}
                }},
              }});
              throw new Error('cancelled synthesis unexpectedly completed');
            }} catch (error) {{
              assert(error?.name === 'AbortError', `unexpected cancellation error: ${{error}}`);
            }}
            assert(abortRequested, 'cancellation was not requested during audio generation');

            setStage(`${{memoryMode}}:${{runtimeMode}}:reuse`);
            const reuse = await bridge.synthesizeSpeech({{
              text: 'Ready.',
              language: 'en',
              maxFrames: 1,
              seed: 3,
            }});
            assert(reuse.pcm instanceof Float32Array, 'reuse PCM output is invalid');
            assert(reuse.sampleCount > 0, 'runtime reuse after cancellation returned no audio');
            cancellationTested = true;
            reuseSampleCount = reuse.sampleCount;
          }}
          setStage(`${{memoryMode}}:${{runtimeMode}}:unload-projector`);
          await bridge.unloadMultimodalProjector();
          const unloadedCapabilities = await bridge.getTextToSpeechCapabilities();
          assert(
            unloadedCapabilities.supported === false,
            'TTS capability remained enabled after projector unload',
          );
          modeResults.push({{
            memoryMode,
            runtimeMode,
            requestedGpuLayers: {gpu_layers},
            gpuActive: bridge.isGpuActive(),
            backendName: bridge.getBackendName(),
            totalElapsedMs: Math.round(performance.now() - startedAt),
            modelLoadMs,
            projectorLoadMs,
            synthesisMs,
            sampleRate: output.sampleRate,
            sampleCount: output.sampleCount,
            durationSeconds: output.sampleCount / output.sampleRate,
            framesGenerated: output.framesGenerated,
            truncated: output.truncated,
            peak,
            rms,
            cancellationTested,
            preAbortedTested: true,
            reuseSampleCount,
            speakerReferenceTested: speakerAudio !== null,
            unloadTested: true,
            _wavBase64: pcmToWavBase64(output.pcm, output.sampleRate),
          }});
        }} finally {{
          await bridge.dispose();
        }}
      }}
    }}

    finish({{
      ok: true,
      modelSha256: {json.dumps(model_sha256)},
      mmprojSha256: {json.dumps(mmproj_sha256)},
      speakerAudioSha256: {json.dumps(speaker_audio_sha256)},
      modeResults,
    }});
  }} catch (error) {{
    finish({{ ok: false, error: String(error?.stack || error) }});
  }}
}})();
</script>
"""
    (web_root / "index.html").write_text(script, encoding="utf-8")


async def run_tts_playwright(
    url: str,
    timeout_ms: int,
    artifacts_dir: Path | None,
) -> dict[str, object]:
    try:
        from playwright.async_api import async_playwright  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - setup failure
        raise RuntimeError("playwright is required for the browser smoke") from exc

    console_lines: list[str] = []
    payload: object = None
    async with async_playwright() as playwright:
        browser_args = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--enable-unsafe-webgpu",
        ]
        features = ["SharedArrayBuffer"]
        if platform.system() == "Darwin":
            browser_args.append("--use-angle=metal")
        else:
            browser_args.append("--disable-vulkan-surface")
            features.append("Vulkan")
        browser_args.append(f"--enable-features={','.join(features)}")
        browser = await playwright.chromium.launch(args=browser_args)
        page = await browser.new_page()

        def record_console(message: object) -> None:
            line = f"{getattr(message, 'type', 'log')}: {getattr(message, 'text', message)}"
            console_lines.append(line)
            if "tts-smoke-stage:" in line:
                print(line, file=sys.stderr, flush=True)

        page.on("console", record_console)
        page.on("pageerror", lambda error: console_lines.append(f"pageerror: {error}"))
        try:
            await page.goto(url, wait_until="load", timeout=timeout_ms)
            deadline = asyncio.get_running_loop().time() + timeout_ms / 1000
            previous_stage: object = None
            while asyncio.get_running_loop().time() < deadline:
                payload = await page.evaluate("() => window.__smokeResult || null")
                if isinstance(payload, dict) and "ok" in payload:
                    break
                stage = await page.evaluate("() => window.__smokeStage || 'starting'")
                if stage != previous_stage:
                    print(f"tts smoke: {stage}", file=sys.stderr, flush=True)
                    previous_stage = stage
                await asyncio.sleep(2)
            else:
                raise TimeoutError(f"browser smoke timed out at stage: {previous_stage}")
        except Exception:
            if artifacts_dir is not None:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                await page.screenshot(
                    path=str(artifacts_dir / "text-to-speech-smoke-page.png"),
                    full_page=True,
                )
            raise
        finally:
            await browser.close()

    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected smoke result payload: {payload!r}")
    for mode_result in payload.get("modeResults", []):
        if not isinstance(mode_result, dict):
            continue
        encoded_wav = mode_result.pop("_wavBase64", "")
        if not encoded_wav or artifacts_dir is None:
            continue
        memory_mode = str(mode_result.get("memoryMode", "unknown"))
        runtime_mode = str(mode_result.get("runtimeMode", "unknown"))
        audio_name = f"text-to-speech-{memory_mode}-{runtime_mode}.wav"
        (artifacts_dir / audio_name).write_bytes(base64.b64decode(encoded_wav))
        mode_result["audioArtifact"] = audio_name
    payload["console"] = console_lines[-200:]
    write_text_artifact(
        artifacts_dir,
        "text-to-speech-smoke-console.log",
        "\n".join(console_lines) + "\n",
    )
    write_json_artifact(
        artifacts_dir,
        "text-to-speech-smoke-result.json",
        payload,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--mmproj-path", type=Path, required=True)
    parser.add_argument("--model-sha256", default="")
    parser.add_argument("--mmproj-sha256", default="")
    parser.add_argument("--speaker-audio-path", type=Path)
    parser.add_argument("--speaker-audio-sha256", default="")
    parser.add_argument("--prompt", default="Hello from llamadart.")
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument(
        "--memory-mode", choices=("all", *MEMORY_MODES), default="all"
    )
    parser.add_argument(
        "--runtime-mode", choices=("all", *RUNTIME_MODES), default="all"
    )
    parser.add_argument("--timeout-ms", type=int, default=1_200_000)
    parser.add_argument("--skip-cancellation", action="store_true")
    parser.add_argument("--artifacts-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dist_dir = args.dist_dir.resolve()
    model_path = args.model_path.resolve()
    mmproj_path = args.mmproj_path.resolve()
    speaker_audio_path = (
        args.speaker_audio_path.resolve() if args.speaker_audio_path else None
    )
    require(dist_dir.is_dir(), f"dist directory does not exist: {dist_dir}")
    require(model_path.is_file(), f"model does not exist: {model_path}")
    require(mmproj_path.is_file(), f"projector does not exist: {mmproj_path}")
    if speaker_audio_path is not None:
        require(
            speaker_audio_path.is_file(),
            f"speaker-reference audio does not exist: {speaker_audio_path}",
        )
    require(
        not args.speaker_audio_sha256 or speaker_audio_path is not None,
        "speaker audio checksum requires --speaker-audio-path",
    )
    require(args.max_frames > 0, "max frames must be positive")
    model_sha256 = sha256_file(model_path)
    mmproj_sha256 = sha256_file(mmproj_path)
    speaker_audio_sha256 = (
        sha256_file(speaker_audio_path) if speaker_audio_path is not None else None
    )
    if args.model_sha256:
        require(model_sha256 == args.model_sha256.lower(), "model checksum mismatch")
    if args.mmproj_sha256:
        require(mmproj_sha256 == args.mmproj_sha256.lower(), "projector checksum mismatch")
    if args.speaker_audio_sha256:
        require(
            speaker_audio_sha256 == args.speaker_audio_sha256.lower(),
            "speaker-reference audio checksum mismatch",
        )
    memory_modes = MEMORY_MODES if args.memory_mode == "all" else (args.memory_mode,)
    runtime_modes = (
        RUNTIME_MODES if args.runtime_mode == "all" else (args.runtime_mode,)
    )

    with tempfile.TemporaryDirectory(prefix="llama-web-bridge-tts-") as tmp:
        web_root = Path(tmp)
        copy_artifacts(dist_dir, web_root)
        if "wasm64" in memory_modes:
            copy_memory64_artifacts(dist_dir, web_root)
        stage_file(model_path, web_root / "qwen3-tts-model.gguf")
        stage_file(mmproj_path, web_root / "qwen3-tts-mmproj.gguf")
        if speaker_audio_path is not None:
            stage_file(speaker_audio_path, web_root / "speaker-reference.wav")
        write_harness(
            web_root,
            prompt=args.prompt,
            model_sha256=model_sha256,
            mmproj_sha256=mmproj_sha256,
            speaker_audio_sha256=speaker_audio_sha256,
            memory_modes=memory_modes,
            runtime_modes=runtime_modes,
            max_frames=args.max_frames,
            gpu_layers=args.gpu_layers,
            test_cancellation=not args.skip_cancellation,
        )
        artifacts_dir = args.artifacts_dir.resolve() if args.artifacts_dir else None
        if artifacts_dir:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(web_root / "index.html", artifacts_dir / "index.html")
        with serve(web_root) as url:
            payload = asyncio.run(
                run_tts_playwright(
                    url,
                    args.timeout_ms,
                    artifacts_dir,
                )
            )

    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload.get("ok") is not True:
        return 1
    expected_modes = len(memory_modes) * len(runtime_modes)
    require(len(payload.get("modeResults", [])) == expected_modes, "mode results are incomplete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"text-to-speech browser smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
