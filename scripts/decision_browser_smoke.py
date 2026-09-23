#!/usr/bin/env python3
"""Real-model browser smoke for Laya decision heads on a ModernBERT encoder."""

from __future__ import annotations

import argparse
import asyncio
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
HEAD_SOURCES = ("url", "bytes")
QUESTION_TYPES = {"choice": 0, "score": 1, "noul": 2}


def load_fixture(path: Path) -> dict[str, object]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(fixture, dict), "fixture is not a JSON object")
    rows = fixture.get("rows")
    require(isinstance(rows, list) and rows, "fixture has no rows")
    special = fixture.get("specialTokens")
    require(
        isinstance(special, dict)
        and all(isinstance(special.get(name), int) for name in ("cls", "sep", "mask")),
        "fixture specialTokens must name cls, sep and mask ids",
    )
    sequences = []
    for index, row in enumerate(rows):
        question_type = row.get("question", {}).get("type")
        require(
            question_type in QUESTION_TYPES,
            f"fixture row {index} has question type {question_type!r}",
        )
        require(
            len(row.get("markers", [])) == len(row.get("rawLogits", [])),
            f"fixture row {index} has mismatched markers and logits",
        )
        sequences.append(
            {
                "id": row.get("id", str(index)),
                "tokens": row["ids"],
                "markers": row["markers"],
                "questionType": QUESTION_TYPES[question_type],
                "rawLogits": row["rawLogits"],
                "rawActLogits": row["rawActLogits"],
            }
        )
    return {"specialTokens": special, "sequences": sequences}


def write_harness(
    web_root: Path,
    *,
    fixture: dict[str, object],
    config_json: str | None,
    memory_modes: tuple[str, ...],
    runtime_modes: tuple[str, ...],
    head_source: str,
    gpu_layers: int,
    context_size: int,
) -> None:
    script = f"""
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge decision smoke</title>
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
    console.log(`decision-smoke-stage:${{stage}}`);
  }};
  const assert = (condition, message) => {{
    if (!condition) throw new Error(message);
  }};
  const rejects = async (promise, pattern, label) => {{
    try {{
      await promise;
    }} catch (error) {{
      const text = String(error?.message || error);
      assert(pattern.test(text), `${{label}} rejected with an unexpected error: ${{text}}`);
      return text;
    }}
    throw new Error(`${{label}} unexpectedly succeeded`);
  }};
  const softmax = (values) => {{
    const top = Math.max(...values);
    const exps = values.map((value) => Math.exp(value - top));
    const sum = exps.reduce((total, value) => total + value, 0);
    return exps.map((value) => value / sum);
  }};
  const argmax = (values) => values.reduce(
    (best, value, index) => (value > values[best] ? index : best),
    0,
  );
  // A safetensors file with the given header text and no tensor data.
  const safetensorsBytes = (headerText) => {{
    const header = new TextEncoder().encode(headerText);
    const bytes = new Uint8Array(8 + header.length);
    new DataView(bytes.buffer).setBigUint64(0, BigInt(header.length), true);
    bytes.set(header, 8);
    return bytes;
  }};
  // Deep enough to overflow a recursive JSON copy or dump.
  const deepJson = '['.repeat(200000) + ']'.repeat(200000);
  try {{
    assert(window.crossOriginIsolated, 'test page is not cross-origin isolated');
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    assert(typeof LlamaWebGpuBridge === 'function', 'bridge export was not registered');
    const fixture = {json.dumps(fixture)};
    const configJson = {json.dumps(config_json)};
    const headSource = {json.dumps(head_source)};
    const sequences = fixture.sequences.map((row) => ({{
      tokens: Int32Array.from(row.tokens),
      markers: row.markers,
      questionType: row.questionType,
    }}));

    const modeResults = [];
    for (const memoryMode of {json.dumps(memory_modes)}) {{
      for (const runtimeMode of {json.dumps(runtime_modes)}) {{
        const useMemory64 = memoryMode === 'wasm64';
        const bridge = new LlamaWebGpuBridge({{
          disableWorker: runtimeMode === 'direct',
          logLevel: 2,
          preferMemory64: useMemory64,
          coreModuleUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.js' : undefined,
          wasmUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.wasm' : undefined,
        }});
        const startedAt = performance.now();
        try {{
          setStage(`${{memoryMode}}:${{runtimeMode}}:unloaded-capabilities`);
          const unloaded = await bridge.getDecisionCapabilities();
          assert(unloaded.apiVersion === 1, 'unexpected decision API version');
          assert(unloaded.supported === false, 'decision heads reported support before a model loaded');

          setStage(`${{memoryMode}}:${{runtimeMode}}:load-model`);
          const modelStartedAt = performance.now();
          await bridge.loadModelFromUrl('/decision-model.gguf', {{
            nCtx: {context_size},
            nGpuLayers: {gpu_layers},
            nThreads: 4,
            nBatch: {context_size},
            nUbatch: {context_size},
            useCache: false,
            forceRemoteFetchBackend: false,
          }});
          const modelLoadMs = Math.round(performance.now() - modelStartedAt);

          setStage(`${{memoryMode}}:${{runtimeMode}}:capabilities`);
          const capabilities = await bridge.getDecisionCapabilities();
          assert(capabilities.apiVersion === 1, 'unexpected decision API version');
          assert(capabilities.supported === true, `decision heads unsupported: ${{capabilities.reason}}`);

          setStage(`${{memoryMode}}:${{runtimeMode}}:reject-invalid-head`);
          await rejects(
            bridge.loadDecisionHead(new Uint8Array([1, 2, 3])),
            /Invalid safetensors file "decision head bytes"/,
            'truncated head bytes',
          );

          setStage(`${{memoryMode}}:${{runtimeMode}}:reject-deep-header`);
          await rejects(
            bridge.loadDecisionHead(safetensorsBytes(
              `{{"x":{{"dtype":"F32","shape":${{deepJson}},"data_offsets":[0,0]}}}}`,
            )),
            /Invalid safetensors file "decision head bytes": header nests JSON deeper than 64 levels/,
            'deeply nested head header',
          );

          setStage(`${{memoryMode}}:${{runtimeMode}}:reject-deep-config`);
          await rejects(
            bridge.loadDecisionHead(safetensorsBytes('{{}}'), {{
              configJson: `{{"temperature":${{deepJson}}}}`,
            }}),
            /Decision head config nests JSON deeper than 64 levels/,
            'deeply nested configJson',
          );

          setStage(`${{memoryMode}}:${{runtimeMode}}:large-config`);
          // Past the 1 MiB wasm stack, so configJson must not travel as a
          // stack-copied string. The empty head then fails its layout check.
          await rejects(
            bridge.loadDecisionHead(safetensorsBytes('{{}}'), {{
              configJson: JSON.stringify({{ max_len: 512, pad: 'x'.repeat(3 * 1024 * 1024) }}),
            }}),
            /has no tensor "head[.]layers[.]0[.]linear1[.]weight"/,
            'large configJson',
          );

          setStage(`${{memoryMode}}:${{runtimeMode}}:load-head`);
          const headOptions = configJson === null ? {{}} : {{ configJson }};
          const headStartedAt = performance.now();
          const source = headSource === 'bytes'
            ? new Uint8Array(await (await fetch('/decision-head.safetensors')).arrayBuffer())
            : '/decision-head.safetensors';
          const info = await bridge.loadDecisionHead(source, headOptions);
          const headLoadMs = Math.round(performance.now() - headStartedAt);
          assert(info.apiVersion === 1, 'unexpected head info API version');
          assert(Number.isInteger(info.handle) && info.handle > 0, 'head handle is not positive');
          assert(info.clsToken === fixture.specialTokens.cls, `CLS token ${{info.clsToken}} does not match the fixture`);
          assert(info.sepToken === fixture.specialTokens.sep, `SEP token ${{info.sepToken}} does not match the fixture`);
          assert(info.maskToken === fixture.specialTokens.mask, `MASK token ${{info.maskToken}} does not match the fixture`);
          assert(info.maskText.length > 0, 'MASK token text is empty');
          assert(info.configJson.length > 0, 'head info has no config');
          assert(info.deviceName.length > 0, 'head info has no device name');

          setStage(`${{memoryMode}}:${{runtimeMode}}:reject-oversized-sequence`);
          // One token past the longest fixture row, which the encoder must accept.
          const oversizedLength = Math.max(...sequences.map((row) => row.tokens.length)) + 1;
          const oversized = new Int32Array(oversizedLength).fill(sequences[0].tokens[0]);
          const tokenLimitError = await rejects(
            bridge.runDecision(info.handle, [
              sequences[0],
              {{ tokens: oversized, markers: [1], questionType: 0 }},
            ]),
            new RegExp(`Decision sequence 1 has ${{oversizedLength}} tokens; the decision encoder accepts 1 to [0-9]+[.]`),
            'oversized sequence',
          );

          setStage(`${{memoryMode}}:${{runtimeMode}}:reject-excess-markers`);
          const firstTokens = sequences[0].tokens;
          await rejects(
            bridge.runDecision(info.handle, [{{
              tokens: firstTokens,
              markers: new Array(firstTokens.length + 1).fill(1),
              questionType: sequences[0].questionType,
            }}]),
            new RegExp(`Decision sequence 0 has ${{firstTokens.length + 1}} markers for its ${{firstTokens.length}} tokens`),
            'more markers than tokens',
          );

          setStage(`${{memoryMode}}:${{runtimeMode}}:run`);
          const runStartedAt = performance.now();
          const outputs = await bridge.runDecision(info.handle, sequences);
          const runMs = Math.round(performance.now() - runStartedAt);
          assert(outputs.length === sequences.length, 'decision output count mismatch');

          let worstLogitDiff = 0;
          let worstProbabilityDiff = 0;
          let worstActProbabilityDiff = 0;
          let worstActRelativeDiff = 0;
          const argmaxChanges = [];
          const actDecisionChanges = [];
          outputs.forEach((output, index) => {{
            const row = fixture.sequences[index];
            assert(output.logits instanceof Float32Array, `row ${{row.id}} logits are not Float32Array`);
            assert(output.actLogits instanceof Float32Array, `row ${{row.id}} act logits are not Float32Array`);
            assert(output.logits.length === row.rawLogits.length, `row ${{row.id}} logit count mismatch`);
            assert(output.actLogits.length === row.rawActLogits.length, `row ${{row.id}} act logit count mismatch`);
            const logits = Array.from(output.logits);
            const actLogits = Array.from(output.actLogits);
            for (const value of [...logits, ...actLogits]) {{
              assert(Number.isFinite(value), `row ${{row.id}} has a non-finite output`);
            }}
            logits.forEach((value, option) => {{
              worstLogitDiff = Math.max(worstLogitDiff, Math.abs(value - row.rawLogits[option]));
            }});
            const probabilities = softmax(logits);
            const reference = softmax(row.rawLogits);
            probabilities.forEach((value, option) => {{
              worstProbabilityDiff = Math.max(worstProbabilityDiff, Math.abs(value - reference[option]));
            }});
            if (argmax(logits) !== argmax(row.rawLogits)) {{
              argmaxChanges.push(row.id);
            }}
            actLogits.forEach((value, action) => {{
              const expected = row.rawActLogits[action];
              worstActRelativeDiff = Math.max(
                worstActRelativeDiff,
                Math.abs(value - expected) / Math.max(1, Math.abs(expected)),
              );
            }});
            const actProbability = softmax(actLogits)[0];
            const referenceActProbability = softmax(row.rawActLogits)[0];
            worstActProbabilityDiff = Math.max(
              worstActProbabilityDiff,
              Math.abs(actProbability - referenceActProbability),
            );
            if ((actProbability >= 0.5) !== (referenceActProbability >= 0.5)) {{
              actDecisionChanges.push(row.id);
            }}
          }});

          setStage(`${{memoryMode}}:${{runtimeMode}}:free`);
          await bridge.freeDecisionHead(info.handle);
          await bridge.freeDecisionHead(info.handle);
          await rejects(
            bridge.runDecision(info.handle, [sequences[0]]),
            /Decision head \\d+ is not loaded/,
            'freed head',
          );

          modeResults.push({{
            memoryMode,
            runtimeMode,
            headSource,
            requestedGpuLayers: {gpu_layers},
            gpuActive: bridge.isGpuActive(),
            backendName: bridge.getBackendName(),
            headDevice: info.deviceName,
            hiddenSize: info.hiddenSize,
            rows: outputs.length,
            totalElapsedMs: Math.round(performance.now() - startedAt),
            modelLoadMs,
            headLoadMs,
            runMs,
            msPerQuestion: runMs / outputs.length,
            worstLogitDiff,
            worstProbabilityDiff,
            worstActProbabilityDiff,
            worstActRelativeDiff,
            argmaxChanges,
            actDecisionChanges,
            tokenLimitError,
            invalidHeadRejected: true,
            hostileInputsRejected: true,
            freeTested: true,
          }});
        }} finally {{
          await bridge.dispose();
        }}
      }}
    }}

    finish({{ ok: true, modeResults }});
  }} catch (error) {{
    finish({{ ok: false, error: String(error?.stack || error) }});
  }}
}})();
</script>
"""
    (web_root / "index.html").write_text(script, encoding="utf-8")


async def run_decision_playwright(
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
            if "decision-smoke-stage:" in line:
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
                    previous_stage = stage
                await asyncio.sleep(2)
            else:
                raise TimeoutError(f"browser smoke timed out at stage: {previous_stage}")
        except Exception:
            if artifacts_dir is not None:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                await page.screenshot(
                    path=str(artifacts_dir / "decision-smoke-page.png"),
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
        "decision-smoke-console.log",
        "\n".join(console_lines) + "\n",
    )
    write_json_artifact(artifacts_dir, "decision-smoke-result.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--head-path", type=Path, required=True)
    parser.add_argument(
        "--config-path",
        type=Path,
        help="rl_agent_config.json for a head without laya.config metadata",
    )
    parser.add_argument(
        "--fixture-path",
        type=Path,
        required=True,
        help="Laya reference fixture with ids, markers, rawLogits and rawActLogits",
    )
    parser.add_argument("--model-sha256", default="")
    parser.add_argument("--head-sha256", default="")
    parser.add_argument("--config-sha256", default="")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--head-source", choices=HEAD_SOURCES, default="url")
    parser.add_argument("--max-logit-diff", type=float, default=0.25)
    parser.add_argument("--max-probability-diff", type=float, default=0.06)
    parser.add_argument("--max-act-probability-diff", type=float, default=0.05)
    # Laya act logits are thousands apart, so softmax saturates and only the
    # raw logits can show a broken act head.
    parser.add_argument("--max-act-relative-diff", type=float, default=0.05)
    parser.add_argument(
        "--memory-mode", choices=("all", *MEMORY_MODES), default="all"
    )
    parser.add_argument(
        "--runtime-mode", choices=("all", *RUNTIME_MODES), default="all"
    )
    parser.add_argument("--timeout-ms", type=int, default=1_800_000)
    parser.add_argument("--artifacts-dir", type=Path)
    return parser.parse_args()


def check_parity(payload: dict[str, object], args: argparse.Namespace) -> None:
    for result in payload.get("modeResults", []):
        mode = f"{result.get('memoryMode')}/{result.get('runtimeMode')}"
        require(
            result.get("worstLogitDiff", float("inf")) <= args.max_logit_diff,
            f"{mode}: worst logit difference {result.get('worstLogitDiff')} exceeds {args.max_logit_diff}",
        )
        require(
            result.get("worstProbabilityDiff", float("inf")) <= args.max_probability_diff,
            f"{mode}: worst probability difference {result.get('worstProbabilityDiff')} "
            f"exceeds {args.max_probability_diff}",
        )
        require(
            result.get("worstActProbabilityDiff", float("inf"))
            <= args.max_act_probability_diff,
            f"{mode}: worst act probability difference {result.get('worstActProbabilityDiff')} "
            f"exceeds {args.max_act_probability_diff}",
        )
        require(
            result.get("worstActRelativeDiff", float("inf")) <= args.max_act_relative_diff,
            f"{mode}: worst act logit relative difference {result.get('worstActRelativeDiff')} "
            f"exceeds {args.max_act_relative_diff}",
        )
        require(
            not result.get("argmaxChanges") and not result.get("actDecisionChanges"),
            f"{mode}: decisions changed against the reference: "
            f"options {result.get('argmaxChanges')}, act {result.get('actDecisionChanges')}",
        )


def main() -> int:
    args = parse_args()
    dist_dir = args.dist_dir.resolve()
    model_path = args.model_path.resolve()
    head_path = args.head_path.resolve()
    config_path = args.config_path.resolve() if args.config_path else None
    fixture_path = args.fixture_path.resolve()
    require(dist_dir.is_dir(), f"dist directory does not exist: {dist_dir}")
    require(model_path.is_file(), f"model does not exist: {model_path}")
    require(head_path.is_file(), f"decision head does not exist: {head_path}")
    require(fixture_path.is_file(), f"fixture does not exist: {fixture_path}")
    if config_path is not None:
        require(config_path.is_file(), f"head config does not exist: {config_path}")
    require(
        not args.config_sha256 or config_path is not None,
        "config checksum requires --config-path",
    )
    require(args.context_size > 0, "context size must be positive")
    checksums = {
        "model": (model_path, args.model_sha256),
        "head": (head_path, args.head_sha256),
        "config": (config_path, args.config_sha256),
    }
    digests: dict[str, str | None] = {}
    for name, (path, expected) in checksums.items():
        digests[name] = sha256_file(path) if path is not None else None
        if expected:
            require(digests[name] == expected.lower(), f"{name} checksum mismatch")
    fixture = load_fixture(fixture_path)
    config_json = (
        config_path.read_text(encoding="utf-8") if config_path is not None else None
    )
    memory_modes = MEMORY_MODES if args.memory_mode == "all" else (args.memory_mode,)
    runtime_modes = (
        RUNTIME_MODES if args.runtime_mode == "all" else (args.runtime_mode,)
    )

    with tempfile.TemporaryDirectory(prefix="llama-web-bridge-decision-") as tmp:
        web_root = Path(tmp)
        copy_artifacts(dist_dir, web_root)
        if "wasm64" in memory_modes:
            copy_memory64_artifacts(dist_dir, web_root)
        stage_file(model_path, web_root / "decision-model.gguf")
        stage_file(head_path, web_root / "decision-head.safetensors")
        write_harness(
            web_root,
            fixture=fixture,
            config_json=config_json,
            memory_modes=memory_modes,
            runtime_modes=runtime_modes,
            head_source=args.head_source,
            gpu_layers=args.gpu_layers,
            context_size=args.context_size,
        )
        artifacts_dir = args.artifacts_dir.resolve() if args.artifacts_dir else None
        if artifacts_dir:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(web_root / "index.html", artifacts_dir / "index.html")
        with serve(web_root) as url:
            payload = asyncio.run(
                run_decision_playwright(url, args.timeout_ms, artifacts_dir)
            )

    payload["modelSha256"] = digests["model"]
    payload["headSha256"] = digests["head"]
    payload["configSha256"] = digests["config"]
    print(json.dumps({k: v for k, v in payload.items() if k != "console"}, indent=2, sort_keys=True))
    if payload.get("ok") is not True:
        return 1
    expected_modes = len(memory_modes) * len(runtime_modes)
    require(len(payload.get("modeResults", [])) == expected_modes, "mode results are incomplete")
    check_parity(payload, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"decision browser smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
