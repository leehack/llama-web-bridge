#!/usr/bin/env python3
"""Browser smoke for next-token scoring.

Runs ``scoreNextToken`` through the direct and worker runtimes of the wasm32
and wasm64 cores with a checksum-pinned GGUF. Scores must be log-probabilities
in descending order, candidate scores must match the top-k scores of the same
tokens, and prompt-prefix reuse must match a fresh evaluation, for a repeated
prompt and for an extended one. Token bytes must decode like ``detokenize``,
and a greedy one-token completion must pick a top-scoring token. A cancel
issued while idle must not abort the next score. Out-of-range
and empty requests must reject with stable messages and leave the runtime
usable, and the worker runtime must still own the model afterwards.
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

MODEL_FILENAME = "next-token-scores-smoke-model.gguf"
PROMPT = "The quick brown fox jumps over the lazy"
PROMPT_SUFFIX = " dog, and then the fox"
TOP_K = 8
# CPU logits of one position differ slightly between a whole-prompt batch and
# a single re-decoded token.
REUSE_TOLERANCE = 1e-3
OUT_OF_VOCABULARY_ERROR = "is outside the vocabulary"
EMPTY_REQUEST_ERROR = "Pass candidates, a positive topK, or both"
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
            "suffix": PROMPT_SUFFIX,
            "topK": TOP_K,
            "tolerance": REUSE_TOLERANCE,
            "outOfVocabularyError": OUT_OF_VOCABULARY_ERROR,
            "emptyRequestError": EMPTY_REQUEST_ERROR,
            "nCtx": n_ctx,
            "memoryModes": memory_modes,
            "runtimeModes": RUNTIME_MODES,
        }
    )
    script = f"""
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge next-token scores smoke</title>
<pre id="result">pending</pre>
<script type="module">
(async () => {{
  const result = document.getElementById('result');
  const finish = (payload) => {{
    result.textContent = JSON.stringify(payload);
    window.__smokeResult = payload;
  }};
  const errorText = (error) => String(error && error.message ? error.message : error);
  const decode = (bytes) => new TextDecoder().decode(bytes);
  try {{
    if (!window.crossOriginIsolated) {{
      throw new Error('test page is not cross-origin isolated');
    }}
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    if (typeof LlamaWebGpuBridge !== 'function') {{
      throw new Error('LlamaWebGpuBridge export was not registered');
    }}
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
      const failures = [];
      const check = (condition, message) => {{
        if (!condition) {{
          failures.push(message);
        }}
      }};
      const scoresById = (entries) => new Map(entries.map((entry) => [entry.token, entry.logprob]));
      const compareScores = (label, expected, actual) => {{
        const actualById = scoresById(actual);
        for (const entry of expected) {{
          const other = actualById.get(entry.token);
          check(
            other !== undefined && Math.abs(other - entry.logprob) <= config.tolerance,
            `${{label}}: token ${{entry.token}} scored ${{other}}, expected ${{entry.logprob}}`,
          );
        }}
      }};
      const checkTop = (label, scores) => {{
        check(scores.top.length === config.topK, `${{label}}: top has ${{scores.top.length}} entries`);
        check(scores.promptTokens > 0, `${{label}}: promptTokens is ${{scores.promptTokens}}`);
        check(new Set(scores.top.map((entry) => entry.token)).size === scores.top.length, `${{label}}: top ids repeat`);
        let previous = 0;
        let mass = 0;
        for (const entry of scores.top) {{
          check(Number.isFinite(entry.logprob) && entry.logprob <= previous, `${{label}}: logprob ${{entry.logprob}} out of order`);
          check(entry.bytes instanceof Uint8Array, `${{label}}: token ${{entry.token}} bytes are not a Uint8Array`);
          previous = entry.logprob;
          mass += Math.exp(entry.logprob);
        }}
        check(mass <= 1 + 1e-6, `${{label}}: top probabilities sum to ${{mass}}`);
      }};
      try {{
        await bridge.loadModelFromUrl(config.modelUrl, {{
          nCtx: config.nCtx,
          nThreads: 2,
          nGpuLayers: 0,
          useCache: false,
          forceRemoteFetchBackend: false,
        }});

        // A cancel with nothing running must not abort the next scoring decode.
        bridge.cancel();
        const first = await bridge.scoreNextToken(config.prompt, {{ topK: config.topK }});
        checkTop('first', first);
        const ids = first.top.map((entry) => entry.token).reverse();

        const candidates = await bridge.scoreNextToken(config.prompt, {{ candidates: ids }});
        check(
          JSON.stringify(candidates.candidates.map((entry) => entry.token)) === JSON.stringify(ids),
          'candidates must keep request order',
        );
        check(candidates.top.length === 0, 'topK 0 must return no top tokens');
        compareScores('candidates vs top', first.top, candidates.candidates);

        const fresh = await bridge.scoreNextToken(config.prompt, {{ candidates: ids, reusePromptPrefix: false }});
        compareScores('repeated prompt reuse vs fresh', first.top, fresh.candidates);

        const extendedPrompt = config.prompt + config.suffix;
        const extended = await bridge.scoreNextToken(extendedPrompt, {{ topK: config.topK }});
        checkTop('extended', extended);
        check(extended.promptTokens > first.promptTokens, 'the extended prompt must have more tokens');
        const extendedIds = extended.top.map((entry) => entry.token);
        const extendedFresh = await bridge.scoreNextToken(extendedPrompt, {{
          candidates: extendedIds,
          reusePromptPrefix: false,
        }});
        compareScores('extended prompt reuse vs fresh', extended.top, extendedFresh.candidates);

        for (const entry of first.top.slice(0, 4)) {{
          const text = await bridge.detokenize([entry.token], true);
          check(decode(entry.bytes) === text, `token ${{entry.token}} bytes decode differently from detokenize`);
        }}

        const completion = await bridge.createCompletion(config.prompt, {{
          nPredict: 1,
          temp: 0,
          topK: 1,
          seed: 1,
          tokenEventEncoding: 'text',
        }});
        const best = first.top[0].logprob;
        check(
          first.top.some((entry) => entry.logprob >= best - config.tolerance && decode(entry.bytes) === completion),
          `greedy completion ${{JSON.stringify(completion)}} is not a top-scoring token`,
        );

        const errors = {{}};
        for (const [name, options, expected] of [
          ['out-of-vocabulary candidate', {{ candidates: [0x7fffffff] }}, config.outOfVocabularyError],
          ['negative candidate', {{ candidates: [-1] }}, config.outOfVocabularyError],
          ['out-of-vocabulary topK', {{ topK: 0x7fffffff }}, config.outOfVocabularyError],
          ['empty request', {{}}, config.emptyRequestError],
        ]) {{
          let error = null;
          try {{
            await bridge.scoreNextToken(config.prompt, options);
          }} catch (caught) {{
            error = errorText(caught);
          }}
          errors[name] = error;
          check(error !== null && error.includes(expected), `${{name}}: expected ${{expected}}, got ${{error}}`);
        }}

        const after = await bridge.scoreNextToken(config.prompt, {{ topK: config.topK }});
        compareScores('after rejected requests', first.top, after.top);

        const metadata = bridge.getModelMetadata();
        return {{
          mode,
          failures,
          errors,
          top: first.top.map((entry) => [entry.token, entry.logprob]),
          promptTokens: [first.promptTokens, extended.promptTokens],
          completion,
          execution: metadata['llamadart.webgpu.execution'] || null,
          coreVariant: metadata['llamadart.webgpu.core_variant'] || null,
          workerFallbackReason: metadata['llamadart.webgpu.worker_fallback_reason'] || null,
        }};
      }} catch (error) {{
        return {{ mode, failures: [...failures, `threw: ${{errorText(error)}}`] }};
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
    for entry in mode_results:
        mode = entry["mode"]
        mode_failures = entry.get("failures")
        if not isinstance(mode_failures, list):
            failures.append(f"{mode}: failures missing")
            continue
        failures.extend(f"{mode}: {failure}" for failure in mode_failures)
        if mode_failures:
            continue
        expected_variant = mode.split(" ", 1)[0]
        if entry.get("coreVariant") != expected_variant:
            # The bridge falls back to wasm32 when the mem64 core fails to
            # start, which would otherwise pass as wasm64 coverage.
            failures.append(
                f"{mode}: expected the {expected_variant} core, got {entry.get('coreVariant')!r}"
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
        default=int(os.environ.get("LLAMA_WEBGPU_NEXT_TOKEN_SCORES_TIMEOUT_MS", "300000")),
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
            Path(os.environ["LLAMA_WEBGPU_NEXT_TOKEN_SCORES_ARTIFACTS_DIR"])
            if os.environ.get("LLAMA_WEBGPU_NEXT_TOKEN_SCORES_ARTIFACTS_DIR")
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

    with tempfile.TemporaryDirectory(prefix="llama-web-bridge-next-token-scores-smoke-") as tmp:
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
                run_playwright(url, args.timeout_ms, artifacts_dir, "next-token-scores-smoke")
            )

    print(json.dumps(payload, indent=2, sort_keys=True))
    failures = validate_payload(payload, memory_modes)
    for failure in failures:
        print(f"next-token scores browser smoke: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"next-token scores browser smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
