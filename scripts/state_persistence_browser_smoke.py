#!/usr/bin/env python3
"""Browser smoke test for bridge state-persistence API wiring.

The default smoke stays lightweight by verifying API shape and clean pre-load
failures in Chromium. When a tiny GGUF model is supplied, the same harness also
loads the model in both direct and worker runtimes, evaluates a prompt, saves a
state snapshot as bytes, mutates the context, reloads the snapshot, and verifies
restored token metadata. CI supplies an integrity-pinned tiny model so state
persistence regressions are caught without large downloads.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import socket
import socketserver
import sys
import tempfile
import threading
import urllib.request

REQUIRED_ARTIFACTS = (
    "llama_webgpu_bridge.js",
    "llama_webgpu_bridge_worker.js",
    "llama_webgpu_core.js",
    "llama_webgpu_core.wasm",
)
REQUIRED_METHODS = (
    "stateSaveFile",
    "stateLoadFile",
    "stateSaveBytes",
    "stateLoadBytes",
)
DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "llama-web-bridge" / "state-smoke-models"


class QuietIsolatedHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:  # noqa: D401 - inherited hook
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_hash(path: Path, expected_sha256: str | None) -> None:
    if not expected_sha256:
        return
    actual = sha256_file(path)
    require(
        actual.lower() == expected_sha256.lower(),
        f"model checksum mismatch for {path}: expected {expected_sha256}, got {actual}",
    )


def redacted_model_location(location: str) -> str:
    parsed = urllib.request.urlparse(location)
    if not parsed.scheme or not parsed.netloc:
        return "[invalid-url]"
    netloc = parsed.hostname or parsed.netloc.rsplit("@", 1)[-1]
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urllib.request.urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))


def download_model(model_url: str, cache_dir: Path, expected_sha256: str | None) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = Path(urllib.request.urlparse(model_url).path).name or "state-smoke-model.gguf"
    digest_prefix = hashlib.sha256(model_url.encode("utf-8")).hexdigest()[:12]
    target = cache_dir / f"{digest_prefix}-{name}"

    if target.is_file():
        validate_model_hash(target, expected_sha256)
        return target

    temp = target.with_suffix(target.suffix + ".tmp")
    try:
        with urllib.request.urlopen(model_url, timeout=120) as response, temp.open("wb") as output:
            shutil.copyfileobj(response, output)
        validate_model_hash(temp, expected_sha256)
        temp.replace(target)
        return target
    except OSError as exc:
        try:
            temp.unlink()
        except OSError:
            pass
        safe_location = redacted_model_location(model_url)
        raise RuntimeError(f"failed to download smoke model from {safe_location}: {exc}") from exc


def resolve_model(args: argparse.Namespace) -> Path | None:
    if args.model_path:
        model_path = args.model_path.resolve()
        require(model_path.is_file(), f"model path does not exist: {model_path}")
        validate_model_hash(model_path, args.model_sha256)
        return model_path

    if args.model_url:
        return download_model(args.model_url, args.model_cache_dir.expanduser().resolve(), args.model_sha256)

    return None


def copy_artifacts(dist_dir: Path, web_root: Path) -> None:
    for artifact in REQUIRED_ARTIFACTS:
        source = dist_dir / artifact
        require(source.is_file(), f"missing bridge artifact: {source}")
        (web_root / artifact).write_bytes(source.read_bytes())


def copy_model(model_path: Path | None, web_root: Path) -> str | None:
    if model_path is None:
        return None
    target_name = "state-smoke-model.gguf"
    shutil.copyfile(model_path, web_root / target_name)
    return target_name


def write_harness(web_root: Path, model_filename: str | None) -> None:
    methods_json = json.dumps(REQUIRED_METHODS)
    model_url_json = json.dumps(f"/{model_filename}" if model_filename else None)
    script = f"""
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge state smoke</title>
<pre id="result">pending</pre>
<script type="module">
(async () => {{
  const result = document.getElementById('result');
  const finish = (payload) => {{
    result.textContent = JSON.stringify(payload);
    window.__smokeResult = payload;
  }};
  const assert = (condition, message) => {{
    if (!condition) {{
      throw new Error(message);
    }}
  }};
  const arraysEqual = (left, right) => (
    Array.isArray(left) && Array.isArray(right)
      && left.length === right.length
      && left.every((value, index) => value === right[index])
  );
  try {{
    if (!window.crossOriginIsolated) {{
      throw new Error('test page is not cross-origin isolated');
    }}
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    if (typeof LlamaWebGpuBridge !== 'function') {{
      throw new Error('LlamaWebGpuBridge export was not registered');
    }}
    const methods = {methods_json};
    const modelUrl = {model_url_json};
    const modeResults = [];

    const verifyPreloadStateApi = async (bridge, mode) => {{
      const missing = methods.filter((name) => typeof bridge[name] !== 'function');
      assert(missing.length === 0, `${{mode}} missing public state methods: ${{missing.join(', ')}}`);

      let saveRejected = false;
      try {{
        await bridge.stateSaveBytes([]);
      }} catch (error) {{
        saveRejected = String(error && error.message ? error.message : error).includes('No model loaded');
      }}
      assert(saveRejected, `${{mode}} stateSaveBytes did not reject cleanly before model load`);

      let loadRejected = false;
      try {{
        await bridge.stateLoadBytes(new Uint8Array([0]), 1);
      }} catch (error) {{
        loadRejected = String(error && error.message ? error.message : error).includes('No model loaded');
      }}
      assert(loadRejected, `${{mode}} stateLoadBytes did not reject cleanly before model load`);
    }};

    const verifyModelRoundTrip = async (bridge, mode) => {{
      const progress = [];
      await bridge.loadModelFromUrl(modelUrl, {{
        nCtx: 64,
        nThreads: 1,
        nGpuLayers: 0,
        nBatch: 32,
        nUbatch: 32,
        useCache: false,
        forceRemoteFetchBackend: false,
        progressCallback: (event) => progress.push(event || {{}}),
      }});

      const prompt = 'Hello';
      const mutationPrompt = 'Completely different prompt';
      const tokens = await bridge.tokenize(prompt, true);
      assert(Array.isArray(tokens) && tokens.length > 0, `${{mode}} tokenization returned no tokens`);

      const firstText = await bridge.createCompletion(prompt, {{
        nPredict: 1,
        temp: 0,
        topK: 1,
        topP: 1,
        seed: 1,
        tokenEventEncoding: 'text',
      }});
      assert(typeof firstText === 'string', `${{mode}} initial completion did not return text`);

      const snapshot = await bridge.stateSaveBytes(tokens);
      assert(snapshot instanceof Uint8Array, `${{mode}} stateSaveBytes did not return Uint8Array`);
      assert(snapshot.byteLength > 0, `${{mode}} stateSaveBytes returned empty snapshot`);

      await bridge.createCompletion(mutationPrompt, {{
        nPredict: 1,
        temp: 0,
        topK: 1,
        topP: 1,
        seed: 2,
        tokenEventEncoding: 'text',
      }});

      let restored;
      let detachedAfterLoadTransfer = null;
      if (mode === 'worker runtime' && bridge._workerProxy && typeof bridge._callWorker === 'function') {{
        const transferableBytes = snapshot.slice();
        restored = await bridge._callWorker(
          'stateLoadBytes',
          [transferableBytes, bridge.getContextSize()],
          null,
          [transferableBytes.buffer],
        );
        detachedAfterLoadTransfer = transferableBytes.buffer.byteLength === 0;
      }} else {{
        restored = await bridge.stateLoadBytes(snapshot.slice(), bridge.getContextSize());
      }}

      assert(restored && arraysEqual(restored.tokens, tokens), `${{mode}} restored tokens did not match saved prompt tokens`);

      const afterRestoreText = await bridge.createCompletion(prompt, {{
        nPredict: 1,
        temp: 0,
        topK: 1,
        topP: 1,
        seed: 3,
        tokenEventEncoding: 'text',
      }});
      assert(typeof afterRestoreText === 'string', `${{mode}} completion after state restore did not return text`);

      const workerSaveSnapshotReturned = mode === 'worker runtime'
        ? snapshot instanceof Uint8Array && snapshot.byteLength > 0
        : null;
      return {{
        progressEvents: progress.length,
        savedBytes: snapshot.byteLength,
        restoredTokens: restored.tokens.length,
        detachedAfterLoadTransfer,
        workerSaveSnapshotReturned,
      }};
    }};

    const verifyBridgeStateApi = async (bridge, mode) => {{
      try {{
        await verifyPreloadStateApi(bridge, mode);
        const modeResult = {{ mode, preload: true }};
        if (modelUrl) {{
          Object.assign(modeResult, await verifyModelRoundTrip(bridge, mode));
        }}
        modeResults.push(modeResult);
      }} finally {{
        await bridge.dispose();
      }}
    }};

    await verifyBridgeStateApi(new LlamaWebGpuBridge({{ disableWorker: true }}), 'direct runtime');
    await verifyBridgeStateApi(new LlamaWebGpuBridge({{ disableWorker: false }}), 'worker runtime');

    if (modelUrl) {{
      const worker = modeResults.find((entry) => entry.mode === 'worker runtime');
      assert(worker && worker.detachedAfterLoadTransfer === true, 'worker stateLoadBytes transfer did not detach transferred buffer');
      assert(worker && worker.workerSaveSnapshotReturned === true, 'worker stateSaveBytes did not return a byte snapshot');
    }}

    finish({{ ok: true, methods, modes: ['direct runtime', 'worker runtime'], modelBacked: Boolean(modelUrl), modeResults }});
  }} catch (error) {{
    finish({{ ok: false, error: String(error && error.stack ? error.stack : error) }});
  }}
}})();
</script>
"""
    (web_root / "index.html").write_text(script, encoding="utf-8")


def free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def serve(web_root: Path):
    port = free_port()
    handler = functools.partial(QuietIsolatedHandler, directory=str(web_root))
    httpd = ReusableThreadingTCPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/index.html"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def write_text_artifact(artifacts_dir: Path | None, name: str, content: str) -> None:
    if artifacts_dir is None:
        return
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / name).write_text(content, encoding="utf-8")


def write_json_artifact(artifacts_dir: Path | None, name: str, payload: object) -> None:
    write_text_artifact(artifacts_dir, name, json.dumps(payload, indent=2, sort_keys=True))


async def run_playwright(
    url: str,
    timeout_ms: int,
    artifacts_dir: Path | None,
    artifact_prefix: str = "state-smoke",
) -> dict[str, object]:
    try:
        from playwright.async_api import async_playwright  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - CI setup failure path
        raise RuntimeError(
            "playwright is required; install with `python3 -m pip install playwright` "
            "and `python3 -m playwright install chromium`"
        ) from exc

    console_lines: list[str] = []
    payload: object = None
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(args=["--no-sandbox"])
        page = await browser.new_page()
        page.on("console", lambda message: console_lines.append(f"{message.type}: {message.text}"))
        page.on("pageerror", lambda error: console_lines.append(f"pageerror: {error}"))
        try:
            await page.goto(url, wait_until="load", timeout=timeout_ms)
            await page.wait_for_function(
                "() => window.__smokeResult && window.__smokeResult.ok !== undefined",
                timeout=timeout_ms,
            )
            payload = await page.evaluate("() => window.__smokeResult")
            if isinstance(payload, dict) and payload.get("ok") is not True and artifacts_dir is not None:
                await page.screenshot(
                    path=str(artifacts_dir / f"{artifact_prefix}-page.png"),
                    full_page=True,
                )
        except Exception:
            if artifacts_dir is not None:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                await page.screenshot(
                    path=str(artifacts_dir / f"{artifact_prefix}-page.png"),
                    full_page=True,
                )
            raise
        finally:
            await browser.close()

    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected smoke result payload: {payload!r}")
    payload["console"] = console_lines[-200:]
    write_text_artifact(
        artifacts_dir,
        f"{artifact_prefix}-console.log",
        "\n".join(console_lines) + "\n",
    )
    write_json_artifact(
        artifacts_dir,
        f"{artifact_prefix}-result.json",
        payload,
    )
    return payload


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
        default=int(os.environ.get("LLAMA_WEBGPU_SMOKE_TIMEOUT_MS", "120000")),
        help="Browser operation timeout in milliseconds.",
    )
    parser.add_argument(
        "--model-url",
        default=os.environ.get("LLAMA_WEBGPU_SMOKE_MODEL_URL", ""),
        help="Optional tiny GGUF URL for model-backed state round-trip smoke.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.environ["LLAMA_WEBGPU_SMOKE_MODEL_PATH"]) if os.environ.get("LLAMA_WEBGPU_SMOKE_MODEL_PATH") else None,
        help="Optional local tiny GGUF path for model-backed state round-trip smoke.",
    )
    parser.add_argument(
        "--model-sha256",
        default=os.environ.get("LLAMA_WEBGPU_SMOKE_MODEL_SHA256", ""),
        help="Expected SHA-256 for the optional smoke model.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path(os.environ.get("LLAMA_WEBGPU_SMOKE_MODEL_CACHE", DEFAULT_MODEL_CACHE)),
        help="Cache directory used when --model-url is supplied.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(os.environ["LLAMA_WEBGPU_SMOKE_ARTIFACTS_DIR"]) if os.environ.get("LLAMA_WEBGPU_SMOKE_ARTIFACTS_DIR") else None,
        help="Directory for JSON/console/screenshot diagnostics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dist_dir = args.dist_dir.resolve()
    require(dist_dir.is_dir(), f"dist directory does not exist: {dist_dir}")
    artifacts_dir = args.artifacts_dir.resolve() if args.artifacts_dir else None

    import asyncio

    model_path = resolve_model(args)
    with tempfile.TemporaryDirectory(prefix="llama-web-bridge-smoke-") as tmp:
        web_root = Path(tmp)
        copy_artifacts(dist_dir, web_root)
        model_filename = copy_model(model_path, web_root)
        write_harness(web_root, model_filename)
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(web_root / "index.html", artifacts_dir / "index.html")
        with serve(web_root) as url:
            payload = asyncio.run(run_playwright(url, args.timeout_ms, artifacts_dir))

    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload.get("ok") is not True:
        return 1
    require(payload.get("methods") == list(REQUIRED_METHODS), "smoke methods payload mismatch")
    require(
        payload.get("modes") == ["direct runtime", "worker runtime"],
        "smoke modes payload mismatch",
    )
    if model_path is not None:
        require(payload.get("modelBacked") is True, "model-backed smoke did not run")
        mode_results = payload.get("modeResults")
        require(isinstance(mode_results, list) and len(mode_results) == 2, "model-backed mode results missing")
        for entry in mode_results:
            require(isinstance(entry.get("savedBytes"), int) and entry["savedBytes"] > 0, "state snapshot was empty")
            require(isinstance(entry.get("restoredTokens"), int) and entry["restoredTokens"] > 0, "restored token metadata missing")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"state persistence browser smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
