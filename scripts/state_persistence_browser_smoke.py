#!/usr/bin/env python3
"""Browser smoke test for bridge state-persistence API wiring.

The full native behavior is covered by the source contract check and the WASM
build. This browser smoke keeps CI lightweight and deterministic by importing
the built bridge module in Chromium, instantiating the public bridge, and
verifying the state-persistence methods are present and reject cleanly before a
model is loaded.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import json
import os
from pathlib import Path
import socket
import socketserver
import sys
import tempfile
import threading

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


def copy_artifacts(dist_dir: Path, web_root: Path) -> None:
    for artifact in REQUIRED_ARTIFACTS:
        source = dist_dir / artifact
        require(source.is_file(), f"missing bridge artifact: {source}")
        (web_root / artifact).write_bytes(source.read_bytes())


def write_harness(web_root: Path) -> None:
    methods_json = json.dumps(REQUIRED_METHODS)
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
  try {{
    if (!window.crossOriginIsolated) {{
      throw new Error('test page is not cross-origin isolated');
    }}
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    if (typeof LlamaWebGpuBridge !== 'function') {{
      throw new Error('LlamaWebGpuBridge export was not registered');
    }}
    const bridge = new LlamaWebGpuBridge({{ disableWorker: true }});
    const methods = {methods_json};
    const missing = methods.filter((name) => typeof bridge[name] !== 'function');
    if (missing.length > 0) {{
      throw new Error(`missing public state methods: ${{missing.join(', ')}}`);
    }}

    let saveRejected = false;
    try {{
      await bridge.stateSaveBytes([]);
    }} catch (error) {{
      saveRejected = String(error && error.message ? error.message : error).includes('No model loaded');
    }}
    if (!saveRejected) {{
      throw new Error('stateSaveBytes did not reject cleanly before model load');
    }}

    let loadRejected = false;
    try {{
      await bridge.stateLoadBytes(new Uint8Array([0]), 1);
    }} catch (error) {{
      loadRejected = String(error && error.message ? error.message : error).includes('No model loaded');
    }}
    if (!loadRejected) {{
      throw new Error('stateLoadBytes did not reject cleanly before model load');
    }}

    await bridge.dispose();
    finish({{ ok: true, methods }});
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


async def run_playwright(url: str, timeout_ms: int) -> dict[str, object]:
    try:
        from playwright.async_api import async_playwright  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - CI setup failure path
        raise RuntimeError(
            "playwright is required; install with `python3 -m pip install playwright` "
            "and `python3 -m playwright install chromium`"
        ) from exc

    console_lines: list[str] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(args=["--no-sandbox"])
        try:
            page = await browser.new_page()
            page.on("console", lambda message: console_lines.append(f"{message.type}: {message.text}"))
            page.on("pageerror", lambda error: console_lines.append(f"pageerror: {error}"))
            await page.goto(url, wait_until="load", timeout=timeout_ms)
            await page.wait_for_function(
                "() => window.__smokeResult && window.__smokeResult.ok !== undefined",
                timeout=timeout_ms,
            )
            payload = await page.evaluate("() => window.__smokeResult")
        finally:
            await browser.close()

    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected smoke result payload: {payload!r}")
    payload["console"] = console_lines[-20:]
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
        default=int(os.environ.get("LLAMA_WEBGPU_SMOKE_TIMEOUT_MS", "30000")),
        help="Browser operation timeout in milliseconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dist_dir = args.dist_dir.resolve()
    require(dist_dir.is_dir(), f"dist directory does not exist: {dist_dir}")

    import asyncio

    with tempfile.TemporaryDirectory(prefix="llama-web-bridge-smoke-") as tmp:
        web_root = Path(tmp)
        copy_artifacts(dist_dir, web_root)
        write_harness(web_root)
        with serve(web_root) as url:
            payload = asyncio.run(run_playwright(url, args.timeout_ms))

    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload.get("ok") is not True:
        return 1
    require(payload.get("methods") == list(REQUIRED_METHODS), "smoke methods payload mismatch")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"state persistence browser smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
