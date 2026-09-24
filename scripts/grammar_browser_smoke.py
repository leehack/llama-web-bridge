#!/usr/bin/env python3
"""Browser smoke for grammar-constrained completion.

Runs grammar-constrained ``createCompletion`` calls through the direct and
worker runtimes of the wasm32 and wasm64 cores with a checksum-pinned GGUF and checks that every result is
text the grammar accepts. It covers greedy ``topK: 1`` decoding, which used to
truncate the candidates to a token the grammar rejects and abort the Wasm core
(issue #115), a sampled ``topK: 40`` run, top-p-only truncation, and a small
JSON grammar. The worker
runtime must still own the model afterwards: an abort there used to move the
bridge to the main thread for the rest of the session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from speech_to_text_browser_smoke import copy_memory64_artifacts
from state_persistence_browser_smoke import (
    DEFAULT_MODEL_CACHE,
    copy_artifacts,
    download_model,
    require,
    run_playwright,
    serve,
    validate_model_hash,
)

MODEL_FILENAME = "grammar-smoke-model.gguf"
# The issue #115 repro prompt. Its ChatML markers are plain text to a model
# without that template, which is fine: the grammar decides the output.
PROMPT = (
    "<|im_start|>user\nIs the sky blue? Answer yes or no.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
YES_NO_GRAMMAR = 'root ::= "yes" | "no"'
JSON_GRAMMAR = "\n".join(
    (
        'root ::= "{" ws "\\"answer\\"" ws ":" ws answer ws "," ws '
        '"\\"confidence\\"" ws ":" ws digit ws "}"',
        'answer ::= "\\"yes\\"" | "\\"no\\""',
        "digit ::= [0-9]",
        'ws ::= " "?',
    )
)
CASES = (
    {
        "name": "yes-no-greedy",
        "grammar": YES_NO_GRAMMAR,
        "kind": "yes-no",
        "options": {"nPredict": 8, "temp": 0, "topK": 1, "seed": 1},
    },
    {
        "name": "yes-no-sampled",
        "grammar": YES_NO_GRAMMAR,
        "kind": "yes-no",
        "options": {"nPredict": 8, "temp": 0.8, "topK": 40, "seed": 1},
    },
    {
        "name": "json-greedy",
        "grammar": JSON_GRAMMAR,
        "kind": "json",
        "options": {"nPredict": 64, "temp": 0, "topK": 1, "seed": 1},
    },
    {
        "name": "json-top-p",
        "grammar": JSON_GRAMMAR,
        "kind": "json",
        "options": {"nPredict": 64, "temp": 0.7, "topK": 0, "topP": 0.1, "seed": 7},
    },
)
MEMORY_MODES = ("wasm32", "wasm64")
RUNTIME_MODES = ("direct", "worker")


def expected_modes(memory_modes: tuple[str, ...]) -> list[str]:
    return [
        f"{memory_mode} {runtime_mode}"
        for memory_mode in memory_modes
        for runtime_mode in RUNTIME_MODES
    ]


def write_harness(web_root: Path, n_ctx: int, memory_modes: tuple[str, ...]) -> None:
    config = json.dumps(
        {
            "modelUrl": f"/{MODEL_FILENAME}",
            "prompt": PROMPT,
            "cases": CASES,
            "nCtx": n_ctx,
            "memoryModes": memory_modes,
            "runtimeModes": RUNTIME_MODES,
        }
    )
    script = f"""
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge grammar smoke</title>
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
  const errorText = (error) => String(error && error.message ? error.message : error);
  const checkText = (kind, text) => {{
    if (kind === 'yes-no') {{
      return text === 'yes' || text === 'no';
    }}
    let parsed;
    try {{
      parsed = JSON.parse(text);
    }} catch (_) {{
      return false;
    }}
    return parsed !== null
      && typeof parsed === 'object'
      && Object.keys(parsed).join(',') === 'answer,confidence'
      && (parsed.answer === 'yes' || parsed.answer === 'no')
      && Number.isInteger(parsed.confidence)
      && parsed.confidence >= 0
      && parsed.confidence <= 9;
  }};
  try {{
    if (!window.crossOriginIsolated) {{
      throw new Error('test page is not cross-origin isolated');
    }}
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    assert(typeof LlamaWebGpuBridge === 'function', 'LlamaWebGpuBridge export was not registered');
    const config = {config};
    const modeResults = [];

    const runMode = async (memoryMode, runtimeMode) => {{
      const mode = `${{memoryMode}} ${{runtimeMode}}`;
      const useMemory64 = memoryMode === 'wasm64';
      const bridge = new LlamaWebGpuBridge({{
        disableWorker: runtimeMode === 'direct',
        preferMemory64: useMemory64,
        coreModuleUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.js' : undefined,
        wasmUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.wasm' : undefined,
      }});
      const cases = [];
      try {{
        await bridge.loadModelFromUrl(config.modelUrl, {{
          nCtx: config.nCtx,
          nThreads: 2,
          nGpuLayers: 0,
          useCache: false,
          forceRemoteFetchBackend: false,
        }});
        for (const testCase of config.cases) {{
          let text = null;
          let error = null;
          try {{
            text = await bridge.createCompletion(config.prompt, {{
              ...testCase.options,
              grammar: testCase.grammar,
              tokenEventEncoding: 'text',
            }});
          }} catch (caught) {{
            error = errorText(caught);
          }}
          cases.push({{
            name: testCase.name,
            text,
            error,
            valid: error === null && typeof text === 'string' && checkText(testCase.kind, text),
          }});
        }}

        // Generation state must be clean after the grammar runs.
        let plainError = null;
        try {{
          await bridge.createCompletion(config.prompt, {{
            nPredict: 4,
            temp: 0,
            topK: 1,
            seed: 1,
            tokenEventEncoding: 'text',
          }});
        }} catch (caught) {{
          plainError = errorText(caught);
        }}

        const metadata = bridge.getModelMetadata();
        return {{
          mode,
          cases,
          plainCompletionError: plainError,
          execution: metadata['llamadart.webgpu.execution'] || null,
          workerFallbackReason: metadata['llamadart.webgpu.worker_fallback_reason'] || null,
        }};
      }} finally {{
        await bridge.dispose();
      }}
    }};

    for (const memoryMode of config.memoryModes) {{
      for (const runtimeMode of config.runtimeModes) {{
        modeResults.push(await runMode(memoryMode, runtimeMode));
      }}
    }}
    finish({{
      ok: true,
      modeResults,
      globalWorkerFallbackReason: globalThis.__llamadartBridgeWorkerFallbackReason || null,
    }});
  }} catch (error) {{
    finish({{ ok: false, error: String(error && error.stack ? error.stack : error) }});
  }}
}})();
</script>
"""
    (web_root / "index.html").write_text(script, encoding="utf-8")


def validate_payload(payload: dict[str, object], memory_modes: tuple[str, ...]) -> list[str]:
    failures: list[str] = []
    if payload.get("ok") is not True:
        return [f"harness failed: {payload.get('error')}"]
    mode_results = payload.get("modeResults")
    if not isinstance(mode_results, list) or [
        entry.get("mode") for entry in mode_results if isinstance(entry, dict)
    ] != expected_modes(memory_modes):
        return ["mode results missing"]
    expected_cases = [case["name"] for case in CASES]
    for entry in mode_results:
        mode = entry["mode"]
        cases = entry.get("cases")
        if not isinstance(cases, list) or [case.get("name") for case in cases] != expected_cases:
            failures.append(f"{mode}: case results missing")
            continue
        for case in cases:
            if case.get("valid") is not True:
                failures.append(
                    f"{mode} {case.get('name')}: expected grammar-valid text, "
                    f"got text={case.get('text')!r} error={case.get('error')!r}"
                )
        if entry.get("plainCompletionError") is not None:
            failures.append(
                f"{mode}: completion after the grammar runs failed: {entry.get('plainCompletionError')!r}"
            )
        expected_execution = "worker" if mode.endswith(" worker") else "main-thread"
        if entry.get("execution") != expected_execution:
            failures.append(
                f"{mode}: expected {expected_execution} execution, got {entry.get('execution')!r} "
                f"(worker fallback reason: {entry.get('workerFallbackReason')!r})"
            )
    if payload.get("globalWorkerFallbackReason") is not None:
        failures.append(
            f"worker fell back to the main thread: {payload.get('globalWorkerFallbackReason')!r}"
        )
    return failures


def resolve_model(args: argparse.Namespace) -> Path:
    require(bool(args.model_sha256), "model SHA-256 is required")
    if args.model_path is not None:
        model_path = args.model_path.expanduser().resolve()
        require(model_path.is_file(), f"model path does not exist: {model_path}")
        validate_model_hash(model_path, args.model_sha256)
        return model_path
    require(bool(args.model_url), "--model-url or --model-path is required")
    return download_model(
        args.model_url,
        args.model_cache_dir.expanduser().resolve(),
        args.model_sha256,
    )


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
        default=int(os.environ.get("LLAMA_WEBGPU_GRAMMAR_TIMEOUT_MS", "300000")),
        help="Browser operation timeout in milliseconds.",
    )
    parser.add_argument(
        "--model-url",
        default=os.environ.get("LLAMA_WEBGPU_SMOKE_MODEL_URL", ""),
        help="GGUF URL; defaults to the state-persistence smoke model.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=(
            Path(os.environ["LLAMA_WEBGPU_SMOKE_MODEL_PATH"])
            if os.environ.get("LLAMA_WEBGPU_SMOKE_MODEL_PATH")
            else None
        ),
        help="Local GGUF path.",
    )
    parser.add_argument(
        "--model-sha256",
        default=os.environ.get("LLAMA_WEBGPU_SMOKE_MODEL_SHA256", ""),
        help="Expected model SHA-256.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path(os.environ.get("LLAMA_WEBGPU_SMOKE_MODEL_CACHE", DEFAULT_MODEL_CACHE)),
        help="Cache directory used with --model-url.",
    )
    parser.add_argument(
        "--memory-mode",
        choices=("all", *MEMORY_MODES),
        default="all",
        help="Core memory mode to run; wasm64 needs the mem64 artifacts.",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=1024,
        help="Context size passed to loadModelFromUrl.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=(
            Path(os.environ["LLAMA_WEBGPU_GRAMMAR_ARTIFACTS_DIR"])
            if os.environ.get("LLAMA_WEBGPU_GRAMMAR_ARTIFACTS_DIR")
            else None
        ),
        help="Directory for JSON/console/screenshot diagnostics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dist_dir = args.dist_dir.resolve()
    require(dist_dir.is_dir(), f"dist directory does not exist: {dist_dir}")
    artifacts_dir = args.artifacts_dir.resolve() if args.artifacts_dir else None
    model_path = resolve_model(args)
    memory_modes = MEMORY_MODES if args.memory_mode == "all" else (args.memory_mode,)

    with tempfile.TemporaryDirectory(prefix="llama-web-bridge-grammar-smoke-") as tmp:
        web_root = Path(tmp)
        copy_artifacts(dist_dir, web_root)
        if "wasm64" in memory_modes:
            copy_memory64_artifacts(dist_dir, web_root)
        shutil.copyfile(model_path, web_root / MODEL_FILENAME)
        write_harness(web_root, args.n_ctx, memory_modes)
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(web_root / "index.html", artifacts_dir / "index.html")
        with serve(web_root) as url:
            payload = asyncio.run(
                run_playwright(url, args.timeout_ms, artifacts_dir, "grammar-smoke")
            )

    print(json.dumps(payload, indent=2, sort_keys=True))
    failures = validate_payload(payload, memory_modes)
    for failure in failures:
        print(f"grammar browser smoke: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"grammar browser smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
