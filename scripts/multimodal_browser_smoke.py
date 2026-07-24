#!/usr/bin/env python3
"""Real-model browser smoke for multimodal bridge prompt ingestion."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from state_persistence_browser_smoke import (
    copy_artifacts,
    download_model,
    require,
    run_playwright,
    serve,
    validate_model_hash,
)

DEFAULT_MODEL_CACHE = (
    Path.home() / ".cache" / "llama-web-bridge" / "multimodal-smoke-models"
)


def resolve_file(
    *,
    path: Path | None,
    url: str,
    expected_sha256: str,
    cache_dir: Path,
    label: str,
) -> Path:
    require(bool(expected_sha256), f"{label} SHA-256 is required")
    if path is not None:
        resolved = path.expanduser().resolve()
        require(resolved.is_file(), f"{label} path does not exist: {resolved}")
        validate_model_hash(resolved, expected_sha256)
        return resolved
    require(bool(url), f"{label} URL or local path is required")
    return download_model(
        url,
        cache_dir.expanduser().resolve(),
        expected_sha256,
    )


def write_harness(web_root: Path) -> None:
    script = """
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge multimodal smoke</title>
<pre id="result">pending</pre>
<script type="module">
(async () => {
  const result = document.getElementById('result');
  const finish = (payload) => {
    result.textContent = JSON.stringify(payload);
    window.__smokeResult = payload;
  };
  const assert = (condition, message) => {
    if (!condition) {
      throw new Error(message);
    }
  };
  const createImageBytes = async () => {
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 180;
    const context = canvas.getContext('2d');
    context.fillStyle = '#0f172a';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#22d3ee';
    context.fillRect(16, 16, 288, 148);
    context.fillStyle = '#111827';
    context.font = 'bold 42px sans-serif';
    context.fillText('HELLO', 80, 108);
    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob((value) => {
        if (value) {
          resolve(value);
        } else {
          reject(new Error('synthetic image encoding failed'));
        }
      }, 'image/png');
    });
    return new Uint8Array(await blob.arrayBuffer());
  };

  try {
    assert(window.crossOriginIsolated, 'test page is not cross-origin isolated');
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge =
      module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    assert(
      typeof LlamaWebGpuBridge === 'function',
      'LlamaWebGpuBridge export was not registered',
    );

    const imageBytes = await createImageBytes();
    const modeResults = [];
    const verifyMode = async (disableWorker, mode) => {
      const bridge = new LlamaWebGpuBridge({ disableWorker, logLevel: 1 });
      const startedAt = performance.now();
      try {
        await bridge.loadModelFromUrl('/multimodal-model.gguf', {
          nCtx: 4096,
          nGpuLayers: 0,
          nThreads: 4,
          nBatch: 512,
          nUbatch: 256,
          useCache: false,
          forceRemoteFetchBackend: false,
        });
        await bridge.loadMultimodalProjector('/multimodal-mmproj.gguf');
        assert(bridge.supportsVision(), `${mode} did not report vision support`);

        const output = await bridge.createCompletion('what do you see?', {
          nPredict: 64,
          temp: 0,
          topK: 1,
          topP: 1,
          seed: 42,
          tokenEventEncoding: 'text',
          parts: [{ type: 'image', bytes: imageBytes }],
          mediaMaxImagePixels: 307200,
          mediaMaxImageEdge: 768,
        });
        const outputText = String(output || '').trim();
        assert(outputText.length > 0, `${mode} returned empty multimodal output`);
        assert(
          outputText.toLowerCase().includes('hello'),
          `${mode} did not recognize the synthetic HELLO image: ${outputText}`,
        );
        modeResults.push({
          mode,
          elapsedMs: Math.round(performance.now() - startedAt),
          output: outputText.slice(0, 160),
        });
      } finally {
        await bridge.dispose();
      }
    };

    await verifyMode(true, 'direct runtime');
    await verifyMode(false, 'worker runtime');
    finish({
      ok: true,
      modes: ['direct runtime', 'worker runtime'],
      modeResults,
    });
  } catch (error) {
    finish({
      ok: false,
      error: String(error && error.stack ? error.stack : error),
    });
  }
})();
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
        default=int(os.environ.get("LLAMA_WEBGPU_MULTIMODAL_TIMEOUT_MS", "420000")),
        help="Browser operation timeout in milliseconds.",
    )
    parser.add_argument(
        "--model-url",
        default=os.environ.get("LLAMA_WEBGPU_MULTIMODAL_MODEL_URL", ""),
        help="Qwen multimodal GGUF URL.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=(
            Path(os.environ["LLAMA_WEBGPU_MULTIMODAL_MODEL_PATH"])
            if os.environ.get("LLAMA_WEBGPU_MULTIMODAL_MODEL_PATH")
            else None
        ),
        help="Local Qwen multimodal GGUF path.",
    )
    parser.add_argument(
        "--model-sha256",
        default=os.environ.get("LLAMA_WEBGPU_MULTIMODAL_MODEL_SHA256", ""),
        help="Expected model SHA-256.",
    )
    parser.add_argument(
        "--mmproj-url",
        default=os.environ.get("LLAMA_WEBGPU_MULTIMODAL_MMPROJ_URL", ""),
        help="Qwen multimodal projector GGUF URL.",
    )
    parser.add_argument(
        "--mmproj-path",
        type=Path,
        default=(
            Path(os.environ["LLAMA_WEBGPU_MULTIMODAL_MMPROJ_PATH"])
            if os.environ.get("LLAMA_WEBGPU_MULTIMODAL_MMPROJ_PATH")
            else None
        ),
        help="Local Qwen multimodal projector GGUF path.",
    )
    parser.add_argument(
        "--mmproj-sha256",
        default=os.environ.get("LLAMA_WEBGPU_MULTIMODAL_MMPROJ_SHA256", ""),
        help="Expected multimodal projector SHA-256.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "LLAMA_WEBGPU_MULTIMODAL_MODEL_CACHE",
                DEFAULT_MODEL_CACHE,
            )
        ),
        help="Cache directory used for downloaded model files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=(
            Path(os.environ["LLAMA_WEBGPU_MULTIMODAL_ARTIFACTS_DIR"])
            if os.environ.get("LLAMA_WEBGPU_MULTIMODAL_ARTIFACTS_DIR")
            else None
        ),
        help="Directory for JSON, console, and screenshot diagnostics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dist_dir = args.dist_dir.resolve()
    require(dist_dir.is_dir(), f"dist directory does not exist: {dist_dir}")
    cache_dir = args.model_cache_dir.expanduser().resolve()
    model_path = resolve_file(
        path=args.model_path,
        url=args.model_url,
        expected_sha256=args.model_sha256,
        cache_dir=cache_dir,
        label="multimodal model",
    )
    mmproj_path = resolve_file(
        path=args.mmproj_path,
        url=args.mmproj_url,
        expected_sha256=args.mmproj_sha256,
        cache_dir=cache_dir,
        label="multimodal projector",
    )
    artifacts_dir = args.artifacts_dir.resolve() if args.artifacts_dir else None

    with tempfile.TemporaryDirectory(prefix="llama-web-bridge-multimodal-") as tmp:
        web_root = Path(tmp)
        copy_artifacts(dist_dir, web_root)
        shutil.copyfile(model_path, web_root / "multimodal-model.gguf")
        shutil.copyfile(mmproj_path, web_root / "multimodal-mmproj.gguf")
        write_harness(web_root)
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(web_root / "index.html", artifacts_dir / "index.html")
        with serve(web_root) as url:
            payload = asyncio.run(
                run_playwright(
                    url,
                    args.timeout_ms,
                    artifacts_dir,
                    artifact_prefix="multimodal-smoke",
                )
            )

    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload.get("ok") is not True:
        return 1
    require(
        payload.get("modes") == ["direct runtime", "worker runtime"],
        "multimodal smoke modes payload mismatch",
    )
    mode_results = payload.get("modeResults")
    require(
        isinstance(mode_results, list) and len(mode_results) == 2,
        "multimodal mode results missing",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"multimodal browser smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
