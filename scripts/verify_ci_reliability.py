#!/usr/bin/env python3
"""Static checks for reliability-focused browser smoke and workflow coverage."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# These files hand-copy the model/projector SHA-256 pins. A rotation that misses
# one leaves CI green while a contributor hits an opaque checksum failure -- or,
# for publish_assets.yml, while the release job that consumes the pins fails.
MODEL_SHA_PIN_FILES = (
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    ".github/workflows/ci.yml",
    ".github/workflows/publish_assets.yml",
)
EXPECTED_MODEL_SHA_PIN_COUNT = 7
SHA256_HEX_PATTERN = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
# A pin is a 64-hex literal introduced by a `--<name>-sha256` smoke CLI flag or a
# `*_SHA256:` workflow env key, joined to it by spacing, `=`, quotes, or a shell
# line continuation. A bare newline never joins the two, so an empty env value
# cannot adopt the next 64-hex line. Any other 64-hex literal is reported and
# skipped, never compared as a model pin.
MODEL_SHA_PIN_MARKER = re.compile(
    r"""(--[A-Za-z0-9][A-Za-z0-9-]*-sha256|_SHA256:)(?:[ \t="'`]|\\\r?\n)+\Z"""
)
# The set check above is role-blind: swapping two pins between roles inside one
# file keeps every value present and passes, yet the workflows pair each SHA with
# a role-specific URL and fail checksum verification at release time. Both
# workflows name the role in the env key, so comparing key-to-hex between them
# catches a swap made in one of them. Not caught: a swap applied identically to
# both workflows, or one confined to README.md, AGENTS.md, and CONTRIBUTING.md,
# whose bare `--model-sha256` / `--mmproj-sha256` flags carry no role and whose
# pin order differs between README.md and the other two -- there the role lives
# only in the `--model-url` / `--model-path` / `--mmproj-path` value beside each
# flag: a URL for the state-persistence pin, a `/path/to/<file>` placeholder for
# the other six, each naming a distinct model or projector file.
WORKFLOW_MODEL_SHA_PIN_FILES = (
    ".github/workflows/ci.yml",
    ".github/workflows/publish_assets.yml",
)
WORKFLOW_MODEL_SHA_PIN_ASSIGNMENT = re.compile(
    r"""^[ \t]*([A-Z][A-Z0-9_]*_SHA256):[ \t]*["']?([0-9a-fA-F]{64})["']?[ \t]*$""",
    re.MULTILINE,
)


def read_required(relative_path: str, errors: list[str]) -> str:
    path = ROOT / relative_path
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"required file is not readable: {relative_path}: {exc}")
        return ""


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def list_typescript_files(errors: list[str]) -> set[Path]:
    tsc = ROOT / "node_modules" / "typescript" / "bin" / "tsc"
    try:
        result = subprocess.run(
            ["node", str(tsc), "-p", "tsconfig.bridge.json", "--noEmit", "--listFiles"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        errors.append(f"failed to run tsc --listFiles: {exc}")
        return set()

    if result.returncode != 0:
        errors.append(f"tsc --listFiles failed with exit code {result.returncode}")
        return set()

    return {Path(line).resolve() for line in result.stdout.splitlines() if line.strip()}


def count_unescaped_pipes(line: str) -> int:
    count = 0
    escaped = False
    for char in line:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "|":
            count += 1
    return count


def require_well_formed_markdown_tables(relative_path: str, content: str, errors: list[str]) -> None:
    in_fenced_code = False
    expected_pipe_count: int | None = None
    expected_line_number = 0

    for line_number, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fenced_code = not in_fenced_code
            expected_pipe_count = None
            expected_line_number = 0
            continue

        if in_fenced_code:
            continue

        if not stripped.startswith("|"):
            expected_pipe_count = None
            expected_line_number = 0
            continue

        pipe_count = count_unescaped_pipes(stripped)
        require(
            pipe_count >= 2,
            f"{relative_path}:{line_number} markdown table row has {pipe_count} unescaped pipes; escape literal pipes as \\|",
            errors,
        )
        if pipe_count < 2:
            continue

        if expected_pipe_count is None:
            expected_pipe_count = pipe_count
            expected_line_number = line_number
            continue

        require(
            pipe_count == expected_pipe_count,
            f"{relative_path}:{line_number} markdown table row has {pipe_count} unescaped pipes, expected {expected_pipe_count} from table starting at line {expected_line_number}; escape literal pipes as \\|",
            errors,
        )


def extract_model_sha_pins(relative_path: str, content: str, errors: list[str]) -> set[str]:
    pins: list[str] = []
    for match in SHA256_HEX_PATTERN.finditer(content):
        pin = match.group(0)
        line_number = content.count("\n", 0, match.start()) + 1
        is_pin = MODEL_SHA_PIN_MARKER.search(content[: match.start()]) is not None
        require(
            is_pin,
            f"{relative_path}:{line_number} has 64-hex literal {pin} that is not a "
            "--<name>-sha256 argument or a *_SHA256 workflow env value; the model "
            "pin consistency check only understands model pins",
            errors,
        )
        if not is_pin:
            continue
        pins.append(pin)

    duplicates = sorted({pin for pin in pins if pins.count(pin) > 1})
    require(
        not duplicates,
        f"{relative_path} repeats model SHA-256 pin(s) {', '.join(duplicates)}; "
        "each pinned model/projector must appear exactly once per file",
        errors,
    )
    require(
        len(pins) == EXPECTED_MODEL_SHA_PIN_COUNT,
        f"{relative_path} declares {len(pins)} model SHA-256 pins, expected "
        f"{EXPECTED_MODEL_SHA_PIN_COUNT}; update EXPECTED_MODEL_SHA_PIN_COUNT in "
        "scripts/verify_ci_reliability.py when the pinned model set changes",
        errors,
    )
    return set(pins)


def require_identical_model_sha_pins(
    pins_by_file: dict[str, set[str]], errors: list[str]
) -> None:
    if len({frozenset(pins) for pins in pins_by_file.values()}) <= 1:
        return

    reported = len(errors)
    for relative_path, pins in pins_by_file.items():
        others = [other for path, other in pins_by_file.items() if path != relative_path]
        if not others:
            continue
        unique = sorted(pins.difference(*others))
        missing = sorted(set.intersection(*others) - pins)
        if not unique and not missing:
            continue
        details = []
        if unique:
            details.append("only in this file: " + ", ".join(unique))
        if missing:
            details.append("present everywhere else but missing here: " + ", ".join(missing))
        errors.append(
            f"{relative_path} model SHA-256 pins diverge ({'; '.join(details)}); "
            "every pin must be byte-identical across " + ", ".join(MODEL_SHA_PIN_FILES)
        )

    if len(errors) == reported:
        errors.append(
            "model SHA-256 pins differ across "
            + ", ".join(MODEL_SHA_PIN_FILES)
            + " with partially overlapping sets: "
            + "; ".join(
                f"{path}=[{', '.join(sorted(pins))}]"
                for path, pins in pins_by_file.items()
            )
        )


def extract_workflow_model_sha_pin_roles(
    relative_path: str, content: str, errors: list[str]
) -> dict[str, str]:
    roles: dict[str, str] = {}
    for match in WORKFLOW_MODEL_SHA_PIN_ASSIGNMENT.finditer(content):
        key, pin = match.group(1), match.group(2)
        line_number = content.count("\n", 0, match.start()) + 1
        require(
            key not in roles,
            f"{relative_path}:{line_number} redefines model SHA-256 env key {key} "
            f"(was {roles.get(key)}, now {pin}); each role must be declared once",
            errors,
        )
        roles[key] = pin
    require(
        len(roles) == EXPECTED_MODEL_SHA_PIN_COUNT,
        f"{relative_path} declares {len(roles)} role-bearing model SHA-256 env "
        f"keys, expected {EXPECTED_MODEL_SHA_PIN_COUNT}; every pin must be a "
        "literal `<ROLE>_SHA256: <64-hex>` assignment so the role-to-hash check "
        "can compare it across workflows",
        errors,
    )
    return roles


def require_identical_workflow_model_sha_pin_roles(
    roles_by_file: dict[str, dict[str, str]], errors: list[str]
) -> None:
    paths = list(roles_by_file)
    base_path = paths[0]
    base = roles_by_file[base_path]
    for other_path in paths[1:]:
        other = roles_by_file[other_path]
        for key in sorted(set(base) | set(other)):
            base_pin = base.get(key, "absent")
            other_pin = other.get(key, "absent")
            require(
                base_pin == other_pin,
                f"model SHA-256 env key {key} is bound to {base_pin} in "
                f"{base_path} but to {other_pin} in {other_path}; each role must "
                "carry the same pin in both workflows because every SHA is paired "
                "with a role-specific model URL",
                errors,
            )


def main() -> int:
    errors: list[str] = []
    smoke = read_required("scripts/state_persistence_browser_smoke.py", errors)
    multimodal_smoke = read_required("scripts/multimodal_browser_smoke.py", errors)
    speech_smoke = read_required("scripts/speech_to_text_browser_smoke.py", errors)
    tts_smoke = read_required("scripts/text_to_speech_browser_smoke.py", errors)
    tts_contract = read_required("scripts/verify_text_to_speech_api.py", errors)
    embedding_contract = read_required("scripts/embedding_json_contract_test.mjs", errors)
    worker_token_contract = read_required(
        "scripts/worker_token_coalescing_test.mjs", errors
    )
    worker_state_contract = read_required(
        "scripts/worker_runtime_state_test.mjs", errors
    )
    ci = read_required(".github/workflows/ci.yml", errors)
    publish = read_required(".github/workflows/publish_assets.yml", errors)
    auto_update = read_required(".github/workflows/auto_llama_cpp_update.yml", errors)
    build_bridge = read_required("scripts/build_bridge.sh", errors)
    emscripten_version_check = read_required(
        "scripts/verify_emscripten_version.py", errors
    )
    wasm64_patch = read_required("scripts/patch_wasm64_runtime.py", errors)
    wasm64_patch_contract = read_required(
        "scripts/wasm64_runtime_patch_contract_test.py", errors
    )
    js_build = read_required("scripts/build_js_bridge.mjs", errors)
    package_json = read_required("package.json", errors)
    tsconfig = read_required("tsconfig.bridge.json", errors)
    js_source = read_required("js/src/llama_webgpu_bridge.js", errors)
    js_output = read_required("js/llama_webgpu_bridge.js", errors)
    js_dts = read_required("js/llama_webgpu_bridge.d.ts", errors)
    cmake = read_required("CMakeLists.txt", errors)
    core = read_required("src/llama_webgpu_core.cpp", errors)
    version = read_required("llama_cpp.version", errors).strip()
    emscripten_version = read_required("emsdk.version", errors).strip()
    agents = read_required("AGENTS.md", errors)
    readme = read_required("README.md", errors)
    api_docs = read_required("docs/api.md", errors)
    contributing = read_required("CONTRIBUTING.md", errors)
    workflow_input_transport = read_required(
        "scripts/workflow_input_transport_test.mjs", errors
    )
    typechecked_files = list_typescript_files(errors)
    publish_job = publish.split("\n  publish-assets:\n", 1)[-1]

    require_well_formed_markdown_tables("docs/api.md", api_docs, errors)

    for name, workflow in (
        ("ci.yml", ci),
        ("publish_assets.yml", publish),
        ("auto_llama_cpp_update.yml", auto_update),
    ):
        require(
            "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24" in workflow,
            f"{name} must opt into Node 24 action runtime to catch Node 20 deprecation breakage early",
            errors,
        )
        require(
            "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true" not in workflow,
            f"{name} must quote FORCE_JAVASCRIPT_ACTIONS_TO_NODE24 so the workflow env value is a string",
            errors,
        )
    transport_result = subprocess.run(
        ["node", "scripts/workflow_input_transport_test.mjs"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    require(
        bool(workflow_input_transport) and transport_result.returncode == 0,
        "resolved workflow input transport contract failed: "
        + (transport_result.stderr.strip() or transport_result.stdout.strip()),
        errors,
    )
    require(
        '"check:js"' in package_json
        and '"typecheck:js"' in package_json
        and '"build:js"' in package_json
        and '"syntax:js"' in package_json
        and '"test:embedding-json"' in package_json
        and '"test:worker-token-coalescing"' in package_json
        and '"test:worker-state"' in package_json
        and '"yaml": "2.8.1"' in package_json
        and '"esbuild"' in package_json
        and '"typescript"' in package_json,
        "package.json must define JS build/typecheck/syntax/runtime-state scripts and pin esbuild + TypeScript dev dependencies",
        errors,
    )
    require(
        "generation_stopped_context_limit" in worker_state_contract
        and "tts_runtime_diagnostic" in worker_state_contract
        and "_callWorker" in worker_state_contract
        and "speechResponse.transfers" in worker_state_contract,
        "worker state contract must exercise completion/TTS diagnostics, public shadow state, and PCM transfer",
        errors,
    )
    require(
        "media_image_resized:320x180->" in multimodal_smoke
        and "bridge.getModelMetadata()" in multimodal_smoke
        and "imageResizeDiagnostic" in multimodal_smoke,
        "multimodal real-model smoke must verify image-resize diagnostics in direct and worker metadata",
        errors,
    )
    require(
        "tokenEventEncoding: 'bytes'" in worker_token_contract
        and "tokenEventFlushMs: 100" in worker_token_contract
        and "tokenEventFlushChars: 3" in worker_token_contract
        and "split-unicode-threshold" in worker_token_contract
        and "split-unicode-timer-boundary" in worker_token_contract
        and "waitForMessageCount" in worker_token_contract
        and "byte-timer-flush" in worker_token_contract
        and "text-timer-flush" in worker_token_contract
        and "cleared timers must not emit after an error" in worker_token_contract
        and "Uint8Array.from" in worker_token_contract,
        "worker token coalescing contract must cover byte batching, decoded-character thresholds, and error cleanup",
        errors,
    )
    require(
        "llama_webgpu_embedding_json.h" in embedding_contract
        and "JSON.parse(raw)" in embedding_contract
        and "float32Bits" in embedding_contract
        and "quiet_NaN" in embedding_contract
        and "infinity" in embedding_contract,
        "embedding JSON contract must compile the C++ serializer and verify JS float32 parsing plus non-finite sanitization",
        errors,
    )
    require(
        '"checkJs": true' in tsconfig
        and '"allowJs": true' in tsconfig,
        "tsconfig.bridge.json must enable JavaScript type-checking",
        errors,
    )
    require(
        (ROOT / "js/src/llama_webgpu_bridge.js").resolve() in typechecked_files,
        "tsc --listFiles must include js/src/llama_webgpu_bridge.js",
        errors,
    )
    require(
        "js/src/llama_webgpu_bridge.js" in js_build
        and "llama_webgpu_bridge.d.ts" in js_build
        and "bundle: true" in js_build
        and "format: 'esm'" in js_build
        and "minify: false" in js_build,
        "scripts/build_js_bridge.mjs must bundle a readable browser ESM bridge from js/src and copy declarations",
        errors,
    )
    require(
        "Generated by scripts/build_js_bridge.mjs" in js_output
        and "export {" in js_output
        and "LlamaWebGpuBridge" in js_output
        and "enableBridgeWorkerHost" in js_output,
        "generated js/llama_webgpu_bridge.js must keep the build banner and public bridge exports",
        errors,
    )
    require(
        "export class LlamaWebGpuBridge" in js_dts
        and "enableBridgeWorkerHost" in js_dts
        and "stateSaveBytes" in js_dts
        and "workerGenerationStallTimeoutMs" in js_dts
        and "loadMultimodalProjector" in js_dts,
        "js/llama_webgpu_bridge.d.ts must expose the public bridge class, worker host entrypoint, and key APIs",
        errors,
    )
    require(
        "getTextToSpeechCapabilities" in js_dts
        and "synthesizeSpeech" in js_dts
        and "workerTextToSpeechTimeoutMs" in js_dts,
        "js/llama_webgpu_bridge.d.ts must expose the versioned TTS surface and worker timeout",
        errors,
    )
    require(
        "export class LlamaWebGpuBridge" in js_source
        and "export function enableBridgeWorkerHost" in js_source,
        "js/src/llama_webgpu_bridge.js must remain the source of the public bridge exports",
        errors,
    )
    require(
        "npm run check:js" in build_bridge
        and "llama_webgpu_bridge.d.ts" in build_bridge,
        "scripts/build_bridge.sh must run the JS bridge check and copy the declaration asset",
        errors,
    )
    require(
        "verify_text_to_speech_api.py" in ci
        and "text_to_speech_browser_smoke.py" in ci
        and "run_text_to_speech_smoke" in ci
        and "Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/ca27d74bc954b73dadab5b71ca265d87fc861a7c" in ci
        and "LLAMA_WEBGPU_TTS_MODEL_SHA256" in ci
        and "LLAMA_WEBGPU_TTS_MMPROJ_SHA256" in ci
        and "--memory-mode wasm64" in ci
        and "--runtime-mode all" in ci
        and "text-to-speech-smoke-artifacts" in ci
        and "LLAMADART_WEBGPU_TTS_API_VERSION" in tts_contract
        and 'RUNTIME_MODES = ("direct", "worker")' in tts_smoke,
        "CI must syntax-check the TTS smoke, run its static API contract, and preserve the opt-in immutable real-model gate",
        errors,
    )
    for name, workflow in (("ci.yml", ci), ("publish_assets.yml", publish)):
        require(
            "actions/setup-node@v4" in workflow
            and "node-version: 24" in workflow
            and "npm ci --ignore-scripts" in workflow
            and "npm run check:js" in workflow
            and "git ls-files --error-unmatch js/llama_webgpu_bridge.js js/llama_webgpu_bridge_worker.js js/llama_webgpu_bridge.d.ts" in workflow
            and "git diff --exit-code -- js/llama_webgpu_bridge.js js/llama_webgpu_bridge_worker.js js/llama_webgpu_bridge.d.ts" in workflow
            and "llama_webgpu_bridge.d.ts" in workflow,
            f"{name} must validate the JS bridge build/type-check gate and publish/check the tracked declaration output",
            errors,
        )

    require(
        version.startswith("b") and version[1:].isdigit(),
        "llama_cpp.version must contain a llama.cpp release tag like b9165",
        errors,
    )
    require(
        re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", emscripten_version) is not None,
        "emsdk.version must contain one exact Emscripten semantic version",
        errors,
    )
    for name, workflow in (("ci.yml", ci), ("publish_assets.yml", publish)):
        require(
            "Resolve Emscripten SDK pin" in workflow
            and "scripts/verify_emscripten_version.py --print-pin" in workflow
            and "version: ${{ env.EMSCRIPTEN_VERSION }}" in workflow
            and (
                "Verify resolved Emscripten compiler" in workflow
                or "Verify compiler and build wasm32 plus memory64" in workflow
            )
            and 'scripts/verify_emscripten_version.py --emit-github-env "$GITHUB_ENV"'
            in workflow
            and "version: latest" not in workflow,
            f"{name} must install emsdk.version and verify the resolved emcc version before building",
            errors,
        )
    require(
        "emsdk.version" in emscripten_version_check
        and '["emcc", "--version"]' in emscripten_version_check
        and "resolved != expected" in emscripten_version_check
        and "EMSCRIPTEN_VERSION={resolved}" in emscripten_version_check
        and "scripts/verify_emscripten_version.py" in build_bridge
        and build_bridge.find("scripts/verify_emscripten_version.py")
        < build_bridge.find('echo "[bridge] configuring with emcmake"'),
        "the Emscripten verifier must compare emcc against emsdk.version, export the resolved compiler identity, and gate direct builds",
        errors,
    )
    require(
        "--emscripten-version \"${EMSCRIPTEN_VERSION}\"" in publish
        and '"emscripten_version": args.emscripten_version' in read_required(
            "scripts/generate_release_manifest.py", errors
        ),
        "the asset manifest must record the runtime-verified Emscripten compiler version",
        errors,
    )
    required_wasmfs_symbols = (
        "__wasmfs_read",
        "__wasmfs_pread",
        "__wasmfs_write",
        "__wasmfs_pwrite",
        "__wasmfs_mmap",
    )
    require(
        all(symbol in wasm64_patch for symbol in required_wasmfs_symbols)
        and "if count == 0" in wasm64_patch
        and "inspect the pinned Emscripten output" in wasm64_patch
        and "scripts/patch_wasm64_runtime.py" in build_bridge
        and "llama_webgpu_core_mem64.js" in build_bridge,
        "the wasm64 generated-JS transform must require every named WASMFS symbol and report actionable failures",
        errors,
    )
    require(
        "test_each_single_match_still_fails_the_five_symbol_contract"
        in wasm64_patch_contract
        and "test_each_missing_symbol_is_named_independently"
        in wasm64_patch_contract
        and "CURRENT_EMSCRIPTEN_OUTPUT" in wasm64_patch_contract
        and "Validate wasm64 runtime patch contract" in ci,
        "CI must prove one-of-five matches cannot pass and retain the current Emscripten output contract",
        errors,
    )
    require(
        "NATIVE_REPO: leehack/llamadart-native" in auto_update
        and "release-candidate.json" in auto_update
        and "select-stable-native-release" in auto_update
        and "gh release view" not in auto_update
        and "native_manifest_sha256" in auto_update
        and "scripts/release_contract.py scan-native" in auto_update
        and "gh workflow run" not in auto_update
        and "create-pull-request" not in auto_update
        and "git push" not in auto_update
        and "did not change" in auto_update,
        "the scheduled native scan must only prepare exact candidate inputs and must not change pins, open PRs, dispatch publication, or push",
        errors,
    )
    require(
        "may dispatch the bridge-owned" in auto_update
        and "through GitHub's API" in auto_update
        and "required correlation ID" in auto_update
        and "may call `publish_assets.yml`" not in auto_update,
        "the scheduled scan summary must describe the bridge-owned API dispatch contract",
        errors,
    )
    require(
        re.search(
            r"target_link_libraries\s*\(\s*llamadart_mtmd\b[^)]*\bvendor::hash\b[^)]*\)",
            cmake,
            re.DOTALL,
        )
        is not None,
        "llamadart_mtmd must retain llama.cpp's vendor::hash dependency",
        errors,
    )
    require(
        'set(LLAMADART_WEBGPU_STACK_SIZE "1048576"' in cmake
        and '"-sSTACK_SIZE=${LLAMADART_WEBGPU_STACK_SIZE}"' in cmake
        and 'WEBGPU_BRIDGE_STACK_SIZE:-1048576' in build_bridge
        and build_bridge.count('-DLLAMADART_WEBGPU_STACK_SIZE="$STACK_SIZE"') == 2,
        "wasm32 and wasm64 builds must each provision the explicit 1 MiB stack required by current llama.cpp graph parameters",
        errors,
    )
    require(
        "const int32_t vocab_size = llama_vocab_n_tokens(g_state.vocab);" in core
        and "vocab_size, 64, repeat_penalty, 0.0f, 0.0f" in core,
        "the repetition-penalty sampler must receive the active vocabulary size",
        errors,
    )
    require(
        "tr -d '[:space:]' < llama_cpp.version" in ci
        and "Resolve llama.cpp pin" in ci
        and "workflow_dispatch:" in ci
        and "LLAMA_CPP_TAG: b9116" not in ci,
        "ci.yml must resolve the llama.cpp tag from llama_cpp.version, support explicit dispatch, and avoid hard-coded stale defaults",
        errors,
    )
    require(
        "python3 -m playwright install chromium" in ci
        and "playwright install --with-deps chromium" not in ci,
        "ci.yml must use the pre-provisioned runner dependencies instead of rerunning the OS package installer for Playwright",
        errors,
    )
    require(
        "dispatch-publish-assets:" not in ci
        and "gh workflow run publish_assets.yml" not in ci
        and "actions: write" not in ci,
        "ordinary CI must never dispatch or authorize cross-repository asset publication",
        errors,
    )
    require(
        "workflow_call:" not in publish
        and "workflow_dispatch:" in publish
        and "orchestrator_correlation_id:" in publish
        and "bridge_source_sha:" in publish
        and "upstream_tag:" in publish
        and "upstream_commit:" in publish
        and "native_release_tag:" in publish
        and "native_manifest_sha256:" in publish
        and "release_tag:" in publish
        and "release_rebuild:" in publish
        and "assets_repo:" in publish
        and "REQUESTED_ASSETS_REPO: ${{ inputs.assets_repo }}" in publish
        and 'if [ "${REQUESTED_ASSETS_REPO}" != "${APPROVED_ASSETS_REPO}" ]'
        in publish
        and "release_capabilities:" not in publish
        and "publish_approved:" in publish
        and "llama_cpp.version" not in publish,
        "bridge-owned publish_assets.yml must expose exact orchestrator/bridge/upstream/native/output inputs without a bridge pin",
        errors,
    )
    require(
        "BRIDGE_REPO: leehack/llama-web-bridge" in publish
        and len(re.findall(r"repository: leehack/llama-web-bridge\s*$", publish, re.MULTILINE)) == 4
        and publish.count("ref: ${{ github.sha }}") == 2
        and '--bridge-repo "${BRIDGE_REPO}"' in publish
        and 'repos/${GITHUB_REPOSITORY}' not in publish,
        "dispatch-only publication must always checkout and verify the owning bridge repository",
        errors,
    )
    require(
        "publish_approved=true requires explicit maintainer approval" in publish
        and "Verify externally configured approval environment" in publish
        and "release_contract.py validate-environment" in publish
        and publish.count("deployment-branch-policies") == 2
        and publish.count("--branch-policies-json") == 2
        and publish.count("bridge-assets-publication/secrets") == 2
        and publish.count("--environment-secrets-json") == 2
        and publish.count("secrets.BRIDGE_PUBLICATION_ENV_READ_TOKEN") == 2
        and publish.count("ENVIRONMENT_READ_TOKEN: ${{ secrets.BRIDGE_PUBLICATION_ENV_READ_TOKEN }}")
        == 2
        and publish.count('GH_TOKEN="${ENVIRONMENT_READ_TOKEN}" gh api') == 2
        and publish.count("gh api --paginate --slurp") == 2
        and publish.count("secrets?per_page=100") == 2
        and publish.count(
            "{total_count: ([.[].total_count] | max // 0), secrets: [.[].secrets[]]}"
        )
        == 2
        and publish.count(
            "BRIDGE_PUBLICATION_ENV_READ_TOKEN is required for read-only environment metadata verification"
        )
        == 2
        and re.search(
            r"verify-publication-environment:\s+.*?permissions:\s+actions: read\s+contents: read",
            publish,
            re.DOTALL,
        )
        is not None
        and 'if [ "${GITHUB_REPOSITORY}" != "${BRIDGE_REPO}" ]' in publish
        and 'if [ "${GITHUB_REF}" != "refs/heads/${bridge_default}" ]' in publish
        and 'echo "environment_name=bridge-assets-publication" >> "${GITHUB_OUTPUT}"'
        in publish
        and "name: ${{ needs.verify-publication-environment.outputs.environment_name }}"
        in publish
        and "environment: bridge-assets-publication" not in publish
        and "push:" not in publish.split("permissions:", 1)[0]
        and "schedule:" not in publish,
        "cross-repository publication must require explicit input confirmation plus a preverified fail-closed environment policy and have no automatic trigger",
        errors,
    )
    post_approval_environment_check = publish_job.find(
        "Revalidate approved environment protection after approval"
    )
    first_pat_reference = publish_job.find("secrets.WEBGPU_BRIDGE_ASSETS_PAT")
    post_approval_read_token_reference = publish_job.find(
        "secrets.BRIDGE_PUBLICATION_ENV_READ_TOKEN",
        post_approval_environment_check,
    )
    require(
        publish.count("release_contract.py validate-environment") == 2
        and re.search(
            r"publish-assets:\s+.*?permissions:\s+actions: read\s+contents: read",
            publish,
            re.DOTALL,
        )
        is not None
        and post_approval_environment_check >= 0
        and post_approval_read_token_reference > post_approval_environment_check
        and first_pat_reference > post_approval_environment_check
        and first_pat_reference > post_approval_read_token_reference
        and 'GH_TOKEN="${ENVIRONMENT_READ_TOKEN}" gh api' in publish_job[
            post_approval_environment_check:first_pat_reference
        ]
        and "python3 publication-policy/scripts/release_contract.py validate-environment"
        in publish_job[post_approval_environment_check:first_pat_reference]
        and publish_job[
            post_approval_environment_check:first_pat_reference
        ].count("\n      - name:")
        == 1,
        "the privileged job must revalidate bypass, reviewer, main-branch, and environment-secret protections after approval and immediately before its first PAT-bearing step",
        errors,
    )
    require(
        "scripts/release_contract.py validate-native-release" in publish
        and '"repos/${NATIVE_REPO}/releases/assets/${asset_id}"' in publish
        and "--checksums \"${RUNNER_TEMP}/native-release/SHA256SUMS\"" in publish
        and "--manifest-sha256 \"${NATIVE_MANIFEST_SHA256}\"" in publish
        and "git clone --depth 1 --branch \"${UPSTREAM_TAG}\"" in publish
        and "git clone --depth 1 --branch \"${NATIVE_RELEASE_TAG}\"" in publish
        and "checked-out bridge source does not match the requested full SHA" in publish,
        "publication must verify exact bridge, upstream tag/commit, native tag/commit, and native manifest checksum identities",
        errors,
    )
    require(
        "concurrency:" in publish
        and "group: publish-bridge-assets-leehack-llama-web-bridge-assets" in publish
        and "cancel-in-progress: false" in publish,
        "publish_assets.yml must serialize asset publishes so automatic and manual releases cannot race",
        errors,
    )
    require(
        "scripts/release_publication_state.py classify" in publish
        and "recoverable-partial" in publish
        and "upload-release-assets" in publish
        and "gh release upload" in publish
        and "publication_outcome" in publish
        and 'cp "${RUNNER_TEMP}/preflight.json" "${RUNNER_TEMP}/publication-outcome.json"' in publish
        and "state_changed()" in publish
        and "mutation-unknown" in publish
        and "ref-requery-failed" in publish
        and "release-requery-failed" in publish
        and "classify_remote() (" in publish
        and '.reason_code != "invalid-input-or-state"' in publish
        and ".orchestrator_correlation_id == env.ORCHESTRATOR_CORRELATION_ID"
        in publish
        and publish.count(
            '.qualification_gates == {state_persistence:"passed",multimodal:"passed",'
        )
        == 2
        and publish.count('speech_to_text:"passed",text_to_speech:"passed"}') >= 2
        and "persist-credentials: false" in publish
        and "sha256sum --check sha256sums.txt" in publish
        and "git -C assets-repo push --atomic" in publish,
        "publication must classify exact retry states, reject failed classifier re-queries, emit outcomes, lock credentials, and recheck after mutation",
        errors,
    )
    require(
        "Run durable state persistence gate" in publish
        and "Run durable multimodal gate" in publish
        and "Run durable Qwen3-ASR gate" in publish
        and "Run durable Qwen3-TTS memory64 gate" in publish
        and "bridge-qualification-outcome" in publish
        and "STATE_CONCLUSION:" in publish
        and "TTS_CONCLUSION:" in publish
        and "contains(inputs.release_capabilities" not in publish
        and "WEBGPU_BRIDGE_BUILD_MEM64: 1" in publish,
        "exact publication must always build wasm32/memory64 and run state, multimodal, ASR, and TTS gates",
        errors,
    )
    require(
        "scripts/generate_release_manifest.py" in publish
        and '"schema_version": 2' in read_required(
            "scripts/generate_release_manifest.py", errors
        )
        and '"release_tag": release.tag' in read_required(
            "scripts/generate_release_manifest.py", errors
        )
        and '"capabilities": CAPABILITIES' in read_required(
            "scripts/generate_release_manifest.py", errors
        )
        and '"bridge_commit": bridge_commit' in read_required(
            "scripts/generate_release_manifest.py", errors
        )
        and '"upstream_commit": upstream_commit' in read_required(
            "scripts/generate_release_manifest.py", errors
        )
        and '"native_commit": native_commit' in read_required(
            "scripts/generate_release_manifest.py", errors
        )
        and '"orchestrator_correlation_id": correlation_id' in read_required(
            "scripts/generate_release_manifest.py", errors
        )
        and '"github_run_id": args.github_run_id' in read_required(
            "scripts/generate_release_manifest.py", errors
        )
        and '"qualification_gates": QUALIFICATION_GATES' in read_required(
            "scripts/generate_release_manifest.py", errors
        )
        and '"sha256": digest' in read_required(
            "scripts/generate_release_manifest.py", errors
        ),
        "published manifests must carry schema, capability gates, run/correlation identity, three-source provenance, and artifact SHA-256 fields",
        errors,
    )

    require(
        "--model-url" in smoke and "--model-sha256" in smoke,
        "browser smoke must support an integrity-checked model-backed state round-trip",
        errors,
    )
    require(
        "model_cache_dir.expanduser().resolve()" in smoke,
        "browser smoke must expand '~' in the model cache path so actions/cache uses the same directory",
        errors,
    )
    require(
        "stateSaveBytes" in smoke
        and "stateLoadBytes" in smoke
        and "createCompletion" in smoke
        and "tokenize" in smoke,
        "browser smoke must exercise actual state save/load after prompt evaluation",
        errors,
    )
    require(
        "detachedAfterLoadTransfer" in smoke and "workerSaveSnapshotReturned" in smoke,
        "browser smoke must assert worker stateLoadBytes transfer detaches ArrayBuffers and stateSaveBytes returns bytes",
        errors,
    )
    require(
        "--artifacts-dir" in smoke
        and "screenshot" in smoke
        and 'artifact_prefix: str = "state-smoke"' in smoke
        and 'f"{artifact_prefix}-result.json"' in smoke,
        "browser smoke must write debuggable failure artifacts",
        errors,
    )
    require(
        "input_text.text_len = normalized_prompt.size();" in core,
        "multimodal prompt ingestion must populate mtmd_input_text.text_len",
        errors,
    )
    require(
        "constexpr llama_load_mode resolve_load_mode" in core
        and "LLAMA_LOAD_MODE_NONE" in core
        and "LLAMA_LOAD_MODE_MMAP" in core
        and "LLAMA_LOAD_MODE_MLOCK" in core
        and "mparams.load_mode = resolve_load_mode(use_mmap, use_mlock);" in core
        and "mparams.use_mmap" not in core
        and "mparams.use_mlock" not in core,
        "native bridge model loading must map legacy mmap/mlock options through llama_load_mode",
        errors,
    )
    require(
        "--model-sha256" in multimodal_smoke
        and "--mmproj-sha256" in multimodal_smoke
        and "--artifacts-dir" in multimodal_smoke
        and "loadMultimodalProjector" in multimodal_smoke
        and "createCompletion" in multimodal_smoke
        and "direct runtime" in multimodal_smoke
        and "worker runtime" in multimodal_smoke
        and 'f"{artifact_prefix}-result.json"' in smoke
        and 'artifact_prefix="multimodal-smoke"' in multimodal_smoke,
        "multimodal browser smoke must integrity-check model inputs and run real direct/worker image inference",
        errors,
    )
    require(
        "LLAMA_WEBGPU_SMOKE_MODEL_URL" in ci
        and "LLAMA_WEBGPU_SMOKE_MODEL_SHA256" in ci
        and "Run state persistence browser smoke" in ci,
        "CI must run model-backed state persistence smoke with a pinned model URL and checksum",
        errors,
    )
    require(
        "state-persistence-smoke-artifacts" in ci and "if: failure()" in ci,
        "CI must upload browser smoke diagnostics on failure",
        errors,
    )
    require(
        "LLAMA_WEBGPU_MULTIMODAL_MODEL_URL" in ci
        and "LLAMA_WEBGPU_MULTIMODAL_MODEL_SHA256" in ci
        and "LLAMA_WEBGPU_MULTIMODAL_MMPROJ_URL" in ci
        and "LLAMA_WEBGPU_MULTIMODAL_MMPROJ_SHA256" in ci
        and "Cache multimodal smoke models" in ci
        and "Run multimodal browser smoke" in ci
        and "multimodal-smoke-artifacts" in ci,
        "CI must run checksum-pinned real multimodal inference and upload diagnostics on failure",
        errors,
    )
    require(
        "LLAMA_WEBGPU_SMOKE_ARTIFACTS_DIR: ${{ runner.temp }}" not in ci,
        "CI must not use runner context in job-level env for smoke artifacts",
        errors,
    )
    require(
        "Verify CI reliability contract" in ci and "scripts/verify_ci_reliability.py" in ci,
        "CI must run this reliability contract check",
        errors,
    )
    require(
        "run_speech_to_text_smoke" in ci
        and "scripts/speech_to_text_browser_smoke.py" in ci
        and "LLAMA_WEBGPU_SPEECH_MODEL_SHA256" in ci
        and "LLAMA_WEBGPU_SPEECH_MMPROJ_SHA256" in ci,
        "CI must expose the checksum-pinned Qwen3-ASR smoke as an opt-in manual gate",
        errors,
    )
    require(
        "wasm32" in speech_smoke
        and "wasm64" in speech_smoke
        and "direct" in speech_smoke
        and "worker" in speech_smoke
        and "AbortController" in speech_smoke
        and "DEFAULT_AUDIO_SHA256" in speech_smoke
        and "DEFAULT_EXPECTED_TEXT" in speech_smoke,
        "speech-to-text smoke must validate both memory/runtime modes, cancellation, and a pinned transcript",
        errors,
    )
    require(
        "npm run check:js" in agents
        and "js/src/" in agents
        and "generated bridge wrapper outputs" in agents
        and "independent review" in agents
        and "state_persistence_browser_smoke.py" in agents
        and "multimodal_browser_smoke.py" in agents
        and "speech_to_text_browser_smoke.py" in agents
        and "text_to_speech_browser_smoke.py" in agents
        and "llama_cpp.version" in agents
        and "emsdk.version" in agents
        and "auto_llama_cpp_update.yml" in agents,
        "AGENTS.md must document the JS build gate, agent PR workflow, browser smoke expectations, and pinned toolchain policies",
        errors,
    )
    require(
        "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24" in readme
        and "npm run check:js" in readme
        and "js/src/llama_webgpu_bridge.js" in readme
        and "llama_webgpu_bridge.d.ts" in readme
        and "docs/api.md" in readme
        and "state-persistence-smoke-artifacts" in readme
        and "multimodal-smoke-artifacts" in readme
        and "scripts/multimodal_browser_smoke.py" in readme
        and "scripts/speech_to_text_browser_smoke.py" in readme
        and "scripts/text_to_speech_browser_smoke.py" in readme
        and "scripts/verify_ci_reliability.py" in readme
        and "llama_cpp.version" in readme
        and "emsdk.version" in readme
        and "auto_llama_cpp_update.yml" in readme,
        "README.md must document the public API reference, CI reliability, JS build/type-checking, diagnostics, and pinned toolchain automation",
        errors,
    )
    for api_name in (
        "LlamaWebGpuBridge",
        "LlamaWebGpuBridgeConfig",
        "enableBridgeWorkerHost",
        "workerGenerationStallTimeoutMs",
        "loadModelFromUrl",
        "prefetchModelToCache",
        "evictModelFromCache",
        "createCompletion",
        "tokenize",
        "detokenize",
        "applyChatTemplate",
        "stateSaveFile",
        "stateLoadFile",
        "stateSaveBytes",
        "stateLoadBytes",
        "embed",
        "embedBatch",
        "loadMultimodalProjector",
        "unloadMultimodalProjector",
        "supportsVision",
        "supportsAudio",
        "getModelMetadata",
        "getContextSize",
        "isGpuActive",
        "getBackendName",
        "setLogLevel",
        "cancel",
        "dispose",
    ):
        require(
            api_name in api_docs,
            f"docs/api.md must document the public JavaScript API member: {api_name}",
            errors,
        )
    require(
        "worker" in api_docs.lower()
        and "direct runtime" in api_docs.lower()
        and "llama_webgpu_bridge.d.ts" in api_docs
        and "state persistence" in api_docs.lower(),
        "docs/api.md must explain declarations, worker/direct runtime behavior, and state persistence semantics",
        errors,
    )
    require(
        "Agent Workflow Guardrails" in contributing
        and "npm run check:js" in contributing
        and "js/src/" in contributing
        and "scripts/verify_ci_reliability.py" in contributing
        and "scripts/multimodal_browser_smoke.py" in contributing
        and "scripts/speech_to_text_browser_smoke.py" in contributing
        and "scripts/text_to_speech_browser_smoke.py" in contributing
        and "--model-sha256" in contributing
        and "--mmproj-sha256" in contributing
        and "llama_cpp.version" in contributing
        and "emsdk.version" in contributing,
        "CONTRIBUTING.md must document JS build/type-checking, maintainer/agent workflow guardrails, checksum-pinned smoke usage, and toolchain pin handling",
        errors,
    )
    pin_file_contents = {
        "README.md": readme,
        "AGENTS.md": agents,
        "CONTRIBUTING.md": contributing,
        ".github/workflows/ci.yml": ci,
        ".github/workflows/publish_assets.yml": publish,
    }
    require_identical_model_sha_pins(
        {
            relative_path: extract_model_sha_pins(
                relative_path, pin_file_contents[relative_path], errors
            )
            for relative_path in MODEL_SHA_PIN_FILES
        },
        errors,
    )
    require_identical_workflow_model_sha_pin_roles(
        {
            relative_path: extract_workflow_model_sha_pin_roles(
                relative_path, pin_file_contents[relative_path], errors
            )
            for relative_path in WORKFLOW_MODEL_SHA_PIN_FILES
        },
        errors,
    )

    if errors:
        print("CI reliability contract failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("CI reliability contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
