#!/usr/bin/env python3
"""Static API contract checks for bridge state-persistence support."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "src" / "llama_webgpu_core.cpp").read_text(encoding="utf-8")
JS = (ROOT / "js" / "src" / "llama_webgpu_bridge.js").read_text(encoding="utf-8")
CMAKE = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")

REQUIRED_NATIVE = [
    "llamadart_webgpu_state_save_file",
    "llamadart_webgpu_state_load_file",
]

REQUIRED_JS_METHODS = [
    "stateSaveFile",
    "stateLoadFile",
    "stateSaveBytes",
    "stateLoadBytes",
]


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def method_count(name: str) -> int:
    return len(re.findall(rf"\basync\s+{re.escape(name)}\s*\(", JS))


def extract_native_function(name: str) -> str:
    match = re.search(
        rf"EMSCRIPTEN_KEEPALIVE\s+int32_t\s+{re.escape(name)}\s*\(",
        SRC,
    )
    if match is None:
        return ""

    brace = SRC.find("{", match.end())
    if brace < 0:
        return ""

    depth = 0
    for index in range(brace, len(SRC)):
        char = SRC[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return SRC[brace : index + 1]
    return ""


def main() -> int:
    errors: list[str] = []

    for symbol in REQUIRED_NATIVE:
        require(symbol in SRC, f"missing native wrapper {symbol} in src/llama_webgpu_core.cpp", errors)
        require(
            f"_{symbol}" in CMAKE,
            f"missing exported function _{symbol} in CMakeLists.txt",
            errors,
        )

    require(
        "llama_state_save_file" in SRC,
        "native save wrapper must call llama_state_save_file",
        errors,
    )
    require(
        "llama_state_load_file" in SRC,
        "native load wrapper must call llama_state_load_file",
        errors,
    )
    require(
        re.search(r"g_cached_prompt_tokens\s*=\s*restored_tokens", SRC) is not None,
        "state load must restore g_cached_prompt_tokens for prompt-prefix reuse",
        errors,
    )
    load_wrapper = extract_native_function("llamadart_webgpu_state_load_file")
    require(
        "if (!loaded)" in load_wrapper
        and "llama_memory_clear(llama_get_memory(g_state.ctx), false);" in load_wrapper
        and "g_cached_prompt_tokens.clear();" in load_wrapper
        and "g_last_output.clear();" in load_wrapper
        and "g_last_piece.clear();" in load_wrapper
        and "g_last_detokenized.clear();" in load_wrapper,
        "state load failure must clear potentially stale KV/prompt-cache/output state",
        errors,
    )
    require(
        "g_generation_active" in SRC and "State cannot be saved or loaded during active generation" in SRC,
        "native wrappers must reject save/load while generation is active",
        errors,
    )

    for method in REQUIRED_JS_METHODS:
        count = method_count(method)
        require(
            count >= 2,
            f"expected runtime and public bridge async methods for {method}, found {count}",
            errors,
        )

    require(
        "llamadart_webgpu_state_save_file" in JS,
        "JS runtime must ccall llamadart_webgpu_state_save_file",
        errors,
    )
    require(
        "llamadart_webgpu_state_load_file" in JS,
        "JS runtime must ccall llamadart_webgpu_state_load_file",
        errors,
    )
    require(
        "FS.readFile" in JS and "FS.writeFile" in JS,
        "bytes helpers must use WASMFS readFile/writeFile",
        errors,
    )
    require(
        "_worker.postMessage({ type: 'call', id, method, args }, transfers)" in JS,
        "worker proxy calls must support transfer lists",
        errors,
    )
    require(
        "if (method === 'stateSaveBytes')" in JS
        and "self.postMessage({ type: 'result', id, value }, transfers)" in JS,
        "worker stateSaveBytes must transfer the state Uint8Array buffer back to the main thread",
        errors,
    )
    require(
        "[transferableBytes.buffer]" in JS,
        "worker stateLoadBytes must transfer a copied state buffer into the worker",
        errors,
    )
    require(
        "State persistence" in README and "stateSaveBytes" in README and "stateLoadBytes" in README,
        "README must document state persistence semantics and bytes APIs",
        errors,
    )
    require(
        "stateSave*` snapshots the current llama.cpp context" in README
        and "The `tokens` argument is stored" in README
        and "`stateLoad*` requires `tokenCapacity`" in README,
        "README must document snapshot timing, token metadata semantics, and tokenCapacity requirements",
        errors,
    )

    if errors:
        print("State persistence API contract failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("State persistence API contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
