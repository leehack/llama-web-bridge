#!/usr/bin/env python3
"""Static API contract checks for versioned Web decision-head support."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from bridge_js_source import bridge_js_source, method_body
from native_core_source import native_core_source

ROOT = Path(__file__).resolve().parents[1]
CORE = native_core_source(ROOT)
DECISION = (ROOT / "src" / "llama_webgpu_decision.cpp").read_text(encoding="utf-8")
HEADER = (ROOT / "src" / "llama_webgpu_decision.h").read_text(encoding="utf-8")
JS = bridge_js_source(ROOT)
BRIDGE_JS = (ROOT / "js" / "src" / "bridge.js").read_text(encoding="utf-8")
DTS = (ROOT / "js" / "src" / "llama_webgpu_bridge.d.ts").read_text(encoding="utf-8")
CMAKE = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
API_DOCS = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
API_DOCS_FLAT = " ".join(API_DOCS.split())
PACKAGE = (ROOT / "package.json").read_text(encoding="utf-8")
SMOKE = (ROOT / "scripts" / "decision_browser_smoke.py").read_text(encoding="utf-8")
CONTRACT_TEST = (ROOT / "tests" / "js" / "decision_bridge_contract_test.mjs").read_text(
    encoding="utf-8"
)

NATIVE_EXPORTS = (
    "llamadart_webgpu_decision_api_version",
    "llamadart_webgpu_decision_capabilities_json",
    "llamadart_webgpu_decision_load",
    "llamadart_webgpu_decision_head_info_json",
    "llamadart_webgpu_decision_run",
    "llamadart_webgpu_decision_free",
)
PUBLIC_METHODS = (
    "getDecisionCapabilities",
    "loadDecisionHead",
    "runDecision",
    "freeDecisionHead",
)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def function_body(source: str, signature: str) -> str:
    start = source.find(signature)
    if start < 0:
        return ""
    end = source.find("\n}\n", start)
    return source[start : end if end > 0 else len(source)]


def worker_host_branch(method: str) -> str:
    """Returns the worker host's dedicated handler for `method`, up to its return."""
    host = function_body(JS, "function installBridgeWorkerHost() {")
    start = host.find(f"if (method === '{method}') {{")
    if start < 0:
        return ""
    end = host.find("return;", start)
    return host[start : end if end > 0 else len(host)]


def main() -> int:
    errors: list[str] = []

    require(
        "LLAMADART_WEBGPU_DECISION_API_VERSION = 1" in HEADER
        and "const DECISION_API_VERSION = 1;" in JS,
        "decision API must have one explicit version shared by the core and the bridge",
        errors,
    )
    for symbol in NATIVE_EXPORTS:
        require(
            re.search(rf"EMSCRIPTEN_KEEPALIVE [^\n]*\b{symbol}\(", CORE) is not None,
            f"missing core wrapper {symbol}",
            errors,
        )
        require(f"'_{symbol}'" in CMAKE, f"missing exported symbol _{symbol}", errors)
    require(
        "src/llama_webgpu_decision.cpp" in CMAKE and '"${LLAMA_CPP_DIR}/vendor"' in CMAKE,
        "CMake must build the decision module with the vendored nlohmann/json include path",
        errors,
    )

    free_runtime = function_body(CORE, "void free_runtime() {")
    require(
        "free_decision_heads();" in free_runtime
        and free_runtime.find("free_decision_heads();")
        < free_runtime.find("llama_free(g_state.ctx)")
        < free_runtime.find("llama_model_free(g_state.model)"),
        "decision heads must be freed before the shared context and model in free_runtime",
        errors,
    )
    for signature in (
        "int32_t llamadart_webgpu_decision_load(",
        "int32_t llamadart_webgpu_decision_run(",
    ):
        require(
            "g_generation_active || g_tts_active" in function_body(CORE, signature),
            f"{signature.split('(')[0].split()[-1]} must refuse to run during generation or text-to-speech",
            errors,
        )
    require(
        "g_next_decision_handle++" in CORE and "g_next_decision_handle = 1" in CORE
        and "g_next_decision_handle = 0" not in CORE,
        "core decision handles must be monotonic within a runtime",
        errors,
    )

    run = function_body(DECISION, "llama_webgpu_decision_status llama_webgpu_decision_run(")
    require(
        0 <= run.find("llama_webgpu_decision_validate_sequences(") < run.find("llama_encode("),
        "every decision sequence must be validated before the first llama_encode",
        errors,
    )
    require(
        "token_count > static_cast<size_t>(std::max(token_limit, 0))" in DECISION
        and "llama_n_ubatch(head->context)" in DECISION,
        "the sequence token limit must come from the head context's n_ubatch",
        errors,
    )
    require(
        "context_params.pooling_type = LLAMA_POOLING_TYPE_NONE;" in DECISION
        and "context_params.embeddings = true;" in DECISION
        and "context_params.n_seq_max = 1;" in DECISION,
        "each head needs a private per-token embedding context",
        errors,
    )
    require(
        re.search(
            r"backends\[backend_count\+\+\] = head->device_backend;.*?"
            r"backends\[backend_count\+\+\] = head->cpu_backend;.*?ggml_backend_sched_new\(",
            DECISION,
            re.DOTALL,
        )
        is not None,
        "the head scheduler must list the CPU backend last",
        errors,
    )
    require(
        "std::erf(" in DECISION and "ggml_gelu_erf(" in DECISION,
        "the head must use erf GELU on both the graph and the host act MLP",
        errors,
    )
    require(
        "kMaxHeaderBytes" in DECISION
        and "static_assert(sizeof(off_t) == 8" in DECISION
        and "fall outside the" in DECISION,
        "safetensors parsing must bound the header and tensor offsets with 64-bit file offsets",
        errors,
    )
    require(
        "ordered_json::parse(text, nullptr, false)" in DECISION
        and "ordered_json::parse(header_text, nullptr, false)" in DECISION
        and "error_handler_t::replace" in DECISION
        and "throw " not in DECISION,
        "decision JSON handling must not depend on C++ exceptions",
        errors,
    )
    for text in ("header_text", "text"):
        parse_at = DECISION.find(f"ordered_json::parse({text}, nullptr, false)")
        check_at = DECISION.find(f"json_depth_within_limit({text})")
        require(
            0 <= check_at < parse_at,
            f"untrusted JSON ({text}) must pass the nesting limit before it is parsed",
            errors,
        )
    validate = function_body(
        DECISION, "llama_webgpu_decision_status llama_webgpu_decision_validate_sequences("
    )
    require(
        "sequence.markers.size() > token_count" in validate,
        "the marker count must be bounded by the sequence's token count",
        errors,
    )
    require(
        "const char * config_path;" in HEADER
        and "[headPath, label, configPath]" in JS
        and "core.FS.writeFile(configPath, textEncoder.encode(configJson))" in JS,
        "configJson must reach the core as a WASMFS file, not a stack-copied ccall string",
        errors,
    )

    for method in PUBLIC_METHODS:
        require(
            len(re.findall(rf"\basync\s+{method}\s*\(", JS)) >= 2
            or (method == "getDecisionCapabilities" and f"  {method}() {{" in JS),
            f"expected direct-runtime and public {method} methods",
            errors,
        )
        require(method in DTS, f"TypeScript declarations must expose {method}", errors)
        require(method in API_DOCS, f"public API docs must document {method}", errors)
    for kind in (
        "decision-capabilities",
        "decision-head-load",
        "decision-run",
        "decision-head-free",
    ):
        require(f"kind: '{kind}'" in JS, f"decision operation {kind} must take the queue", errors)
    run_branch = worker_host_branch("runDecision")
    require(
        "transfers.push(buffer)" in run_branch
        and "self.postMessage({ type: 'result', id, value }, transfers)" in run_branch,
        "worker decision runs must transfer output buffers instead of copying them",
        errors,
    )
    require(
        "event: 'progress'" in worker_host_branch("loadDecisionHead"),
        "worker head loads must post download progress so the request timer re-arms",
        errors,
    )
    require(
        "sequences * DECISION_WORKER_TIMEOUT_PER_SEQUENCE_MS" in JS,
        "the worker runDecision timeout must grow with the number of sequences",
        errors,
    )
    require(
        "_resolveDecisionHead(handle)" in JS
        and "entry.owner !== this._decisionOwner()" in JS
        and "or the bridge runtime restarted" in JS,
        "facade decision handles must be bound to the worker or runtime that loaded them",
        errors,
    )
    require(
        re.search(
            r"if \(!this\._shouldFallbackToMainThread\(error\)\) \{\s*throw error;",
            method_body(BRIDGE_JS, "async _loadDecisionHeadUnlocked("),
            re.DOTALL,
        )
        is not None,
        "worker head-load errors must only fall back to the main thread for worker failures",
        errors,
    )
    require(
        "_llamadart_webgpu_decision_capabilities_json !== 'function'" in JS
        and "this bridge needs version ${DECISION_API_VERSION}" in JS,
        "decision capabilities must probe the core export and API version",
        errors,
    )
    require(
        "interface DecisionCapabilities" in DTS
        and "interface DecisionHeadInfo" in DTS
        and "interface DecisionSequence" in DTS
        and "interface DecisionOutput" in DTS,
        "TypeScript declarations must expose decision capabilities, head info, sequences and outputs",
        errors,
    )
    require(
        '"test:decision"' in PACKAGE and "npm run test:decision" in PACKAGE,
        "check:js must define and run the decision bridge contract test",
        errors,
    )
    require(
        "the bridge runtime restarted" in CONTRACT_TEST
        and "malformed input never reaches the core" in CONTRACT_TEST
        and "an ordinary load error keeps the worker" in CONTRACT_TEST
        and "the model is reloaded on the main thread" in CONTRACT_TEST
        and "the load is retried on the main thread" in CONTRACT_TEST
        and "a failed worker free still falls back" in CONTRACT_TEST
        and "every output buffer is transferred" in CONTRACT_TEST
        and "_resolveRequestTimeoutMs('runDecision'" in CONTRACT_TEST
        and "decision API version 2" in CONTRACT_TEST,
        "decision contract test must cover stale handles, input validation, worker "
        "fallback scoping, worker loss, worker-host transfers, timeouts, and API version skew",
        errors,
    )
    require(
        "Decision heads" in API_DOCS
        and "ModernBERT" in API_DOCS_FLAT
        and "not durable" in API_DOCS_FLAT,
        "public API docs must document decision heads, their encoder requirement, and handle lifetime",
        errors,
    )
    require(
        "Decision heads" in README and "decision_browser_smoke.py" in README,
        "README must document decision heads and their real-model smoke",
        errors,
    )
    require(
        'RUNTIME_MODES = ("direct", "worker")' in SMOKE
        and '"--fixture-path",' in SMOKE
        and '"--config-path",' in SMOKE
        and 'parser.add_argument("--gpu-layers"' in SMOKE
        and "check_parity(payload, args)" in SMOKE
        and "worstLogitDiff" in SMOKE
        and "argmaxChanges" in SMOKE
        and "oversizedLength" in SMOKE
        and "Invalid safetensors file" in SMOKE
        and "freeDecisionHead" in SMOKE,
        "real-model smoke must compare direct/worker outputs with the reference fixture, "
        "reject oversized sequences and invalid heads, and free heads",
        errors,
    )
    parity_start = SMOKE.find("def check_parity(")
    parity = SMOKE[parity_start : SMOKE.find("\ndef ", parity_start + 1)] if parity_start >= 0 else ""
    require(
        "args.max_act_relative_diff" in parity and "worstActRelativeDiff" in parity,
        "real-model smoke must gate raw act logits, which saturated act probabilities cannot",
        errors,
    )
    for stage in (
        "reject-deep-header",
        "reject-deep-config",
        "large-config",
        "reject-excess-markers",
    ):
        require(
            f":{stage}`" in SMOKE,
            f"real-model smoke must run the {stage} stage",
            errors,
        )

    if errors:
        print("Decision API contract failed:", file=sys.stderr)
        for item in errors:
            print(f"- {item}", file=sys.stderr)
        return 1

    print("Decision API contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
