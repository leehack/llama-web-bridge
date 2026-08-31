#!/usr/bin/env python3
"""Static checks for reliability-focused browser smoke and workflow coverage."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from release_contract import ContractError, parse_upstream_tag

ROOT = Path(__file__).resolve().parents[1]

# These files hand-copy the model/projector SHA-256 pins. A rotation that misses
# one leaves CI green while a contributor hits an opaque checksum failure -- or,
# for bridge_candidate.yml, while the candidate job that consumes the pins fails.
# publish_assets.yml no longer builds or smokes anything, so it holds no pins.
MODEL_SHA_PIN_FILES = (
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    ".github/workflows/ci.yml",
    ".github/workflows/bridge_candidate.yml",
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
    ".github/workflows/bridge_candidate.yml",
)
WORKFLOW_MODEL_SHA_PIN_ASSIGNMENT = re.compile(
    r"""^[ \t]*([A-Z][A-Z0-9_]*_SHA256):[ \t]*["']?([0-9a-fA-F]{64})["']?[ \t]*$""",
    re.MULTILINE,
)
PUBLICATION_PAT_NAME = "WEBGPU_BRIDGE_ASSETS_PAT"
PUBLICATION_PAT_REFERENCE = re.compile(
    r"secrets\s*(?:\.\s*WEBGPU_BRIDGE_ASSETS_PAT|\[\s*['\"]WEBGPU_BRIDGE_ASSETS_PAT['\"]\s*\])"
)
PUBLICATION_PAT_GUARD_ERROR = (
    "error: WEBGPU_BRIDGE_ASSETS_PAT is required for asset publication"
)
EXPECTED_PUBLICATION_PAT_STEPS = (
    ("publish-assets", "Verify source ancestry and classify exact remote state"),
    ("publish-assets", "Apply only the classified ref mutation"),
    ("publish-assets", "Re-fetch and finish or safely classify partial publication"),
)
RESOLVE_WORKFLOW_STEPS_JS = r"""
const fs = require('fs');
const YAML = require('yaml');

const workflow = YAML.parse(fs.readFileSync(0, 'utf8'), { merge: true });
const mapping = (value) =>
  value && typeof value === 'object' && !Array.isArray(value) ? value : {};
const patReference = /secrets\s*(?:\.\s*WEBGPU_BRIDGE_ASSETS_PAT|\[\s*['"]WEBGPU_BRIDGE_ASSETS_PAT['"]\s*\])/;
const containsPat = (value) => {
  if (typeof value === 'string') return patReference.test(value);
  if (Array.isArray(value)) return value.some(containsPat);
  if (value && typeof value === 'object') return Object.values(value).some(containsPat);
  return false;
};

const rootEnv = mapping(mapping(workflow).env);
const resolved = [];
const patOutsideEnv = [];
const rootOutsideEnv = { ...mapping(workflow) };
delete rootOutsideEnv.env;
delete rootOutsideEnv.jobs;
if (containsPat(rootOutsideEnv)) patOutsideEnv.push('workflow root properties');
for (const [jobName, rawJob] of Object.entries(mapping(mapping(workflow).jobs))) {
  const job = mapping(rawJob);
  const jobEnv = { ...rootEnv, ...mapping(job.env) };
  const jobOutsideEnv = { ...job };
  delete jobOutsideEnv.env;
  delete jobOutsideEnv.steps;
  if (containsPat(jobOutsideEnv)) patOutsideEnv.push(`job ${jobName} properties`);
  const steps = Array.isArray(job.steps) ? job.steps : [];
  steps.forEach((rawStep, index) => {
    const step = mapping(rawStep);
    const stepOutsideEnv = { ...step };
    delete stepOutsideEnv.env;
    resolved.push({
      job: jobName,
      index,
      name: typeof step.name === 'string' ? step.name : '',
      env: { ...jobEnv, ...mapping(step.env) },
      run: typeof step.run === 'string' ? step.run : '',
      patOutsideEnv: containsPat(stepOutsideEnv)
        ? `step ${jobName}/${typeof step.name === 'string' ? step.name : index} properties`
        : '',
    });
  });
}
process.stdout.write(JSON.stringify({ steps: resolved, patOutsideEnv }));
"""


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


def run_required_python_contract(relative_path: str, errors: list[str]) -> None:
    """Execute a fast contract suite so CI verifies behavior, not test-name text."""
    try:
        result = subprocess.run(
            [sys.executable, relative_path],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        errors.append(f"failed to execute {relative_path}: {exc}")
        return
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        errors.append(f"{relative_path} failed: {detail}")


def resolve_workflow_steps(
    workflow: str,
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    try:
        result = subprocess.run(
            ["node", "-e", RESOLVE_WORKFLOW_STEPS_JS],
            cwd=ROOT,
            input=workflow,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return [], [], [f"failed to run Node yaml workflow resolver: {exc}"]

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return [], [], [f"Node yaml workflow resolver failed: {detail}"]
    try:
        resolved = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return [], [], [f"Node yaml workflow resolver returned invalid JSON: {exc}"]
    if not isinstance(resolved, dict):
        return [], [], ["Node yaml workflow resolver did not return an object"]
    steps = resolved.get("steps")
    pat_outside_env = resolved.get("patOutsideEnv")
    if not isinstance(steps, list) or not isinstance(pat_outside_env, list):
        return [], [], ["Node yaml workflow resolver returned invalid contract data"]
    return steps, [str(location) for location in pat_outside_env], []


def shell_variable_reference(variable: str) -> re.Pattern[str]:
    escaped = re.escape(variable)
    return re.compile(rf"\$(?:\{{{escaped}(?:[^A-Za-z0-9_][^}}]*)?\}}|{escaped}\b)")


def fail_closed_guard_end(script: str, variable: str) -> int:
    escaped = re.escape(variable)
    blank_or_comment = r"[ \t]*(?:\#[^\r\n]*)?\r?\n"
    guard = re.compile(
        rf"\A(?:{blank_or_comment})*"
        rf"[ \t]*set[ \t]+-euo[ \t]+pipefail[ \t]*\r?\n"
        rf"(?:{blank_or_comment})*"
        rf"[ \t]*if[ \t]+\[[ \t]+-z[ \t]+\"\$\{{{escaped}\}}\"[ \t]+\];"
        rf"[ \t]+then[ \t]*\r?\n"
        rf"[ \t]*echo[ \t]+\"{re.escape(PUBLICATION_PAT_GUARD_ERROR)}\"[ \t]*\r?\n"
        rf"[ \t]*exit[ \t]+[1-9][0-9]*[ \t]*\r?\n"
        rf"[ \t]*fi(?:[ \t]*\r?\n|[ \t]*\Z)"
    )
    match = guard.match(script)
    return match.end() if match else -1


def first_sensitive_command(script: str) -> tuple[int, str] | None:
    offset = 0
    for line in script.splitlines(keepends=True):
        code = line.lstrip()
        if not code or code.startswith("#"):
            offset += len(line)
            continue
        command = re.search(r"\b(?:gh|curl|wget)\b", code)
        git_operation = re.search(
            r"\bgit\b[^\n]*?\b(?:clone|fetch|pull|push|ls-remote|remote|add|commit|tag)\b",
            code,
        )
        matches = [match for match in (command, git_operation) if match is not None]
        if matches:
            match = min(matches, key=lambda item: item.start())
            return offset + len(line) - len(code) + match.start(), match.group(0)
        offset += len(line)
    return None


def credential_logging_errors(script: str, variables: list[str]) -> list[str]:
    collapsed = re.sub(r"\\\r?\n", " ", script)
    found: list[str] = []
    if re.search(r"(?m)^\s*set\s+(?:-[A-Za-z]*x[A-Za-z]*\b|-o\s+xtrace\b)", collapsed):
        found.append("set -x/xtrace")
    if re.search(r"\bprintenv\b", collapsed):
        found.append("printenv")
    if re.search(
        r"(?m)(?:^|[;&|])\s*(?:command\s+)?"
        r"[\"']?(?:env|/(?:[^/\s;&|]+/)*env)[\"']?\s*(?:$|[;&|>])",
        collapsed,
    ):
        found.append("bare env")
    for variable in variables:
        reference = shell_variable_reference(variable)
        for line in collapsed.splitlines():
            if re.search(r"\b(?:echo|printf)\b", line) and reference.search(line):
                found.append(f"echo/printf of {variable}")
                break
    return found


def validate_publication_pat_contract(
    workflow: str, expected_steps: tuple[tuple[str, str], ...]
) -> list[str]:
    steps, pat_outside_env, resolution_errors = resolve_workflow_steps(workflow)
    if resolution_errors:
        return resolution_errors

    errors = [
        f"{location} references {PUBLICATION_PAT_NAME} outside resolved env"
        for location in pat_outside_env
    ]
    pat_steps: list[tuple[dict[str, object], list[str]]] = []
    for step in steps:
        job = str(step.get("job", ""))
        name = str(step.get("name", ""))
        if step.get("patOutsideEnv"):
            errors.append(
                f"{step['patOutsideEnv']} references {PUBLICATION_PAT_NAME} outside resolved env"
            )
        env = step.get("env", {})
        if not isinstance(env, dict):
            continue
        variables = [
            str(key)
            for key, value in env.items()
            if PUBLICATION_PAT_REFERENCE.search(str(value))
        ]
        if variables:
            pat_steps.append((step, variables))

    actual_steps = [(str(step.get("job", "")), str(step.get("name", ""))) for step, _ in pat_steps]
    if sorted(actual_steps) != sorted(expected_steps):
        errors.append(
            f"PAT-bearing steps are {actual_steps!r}, expected exactly {list(expected_steps)!r}"
        )

    for step, variables in pat_steps:
        job = str(step.get("job", ""))
        name = str(step.get("name", ""))
        script = str(step.get("run", ""))
        for variable in variables:
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable) is None:
                errors.append(f"{job}/{name} binds the PAT to invalid shell variable {variable!r}")
                continue
            guard_end = fail_closed_guard_end(script, variable)
            if guard_end < 0:
                errors.append(
                    f"{job}/{name} must start with set -euo pipefail followed immediately "
                    f"by the canonical executable empty-token guard for {variable}"
                )
                continue
            sensitive = first_sensitive_command(script)
            if sensitive is not None and sensitive[0] < guard_end:
                errors.append(
                    f"{job}/{name} runs sensitive command {sensitive[1]!r} before its {variable} credential guard"
                )
        for logging_path in credential_logging_errors(script, variables):
            errors.append(f"{job}/{name} exposes the PAT through {logging_path}")
    return errors


def require_publication_pat_contract_self_tests(errors: list[str]) -> None:
    expected = (("publish-assets", "Safe publication"),)
    safe_fixture = """
env: &publication-env
  RELEASE_CREDENTIAL: '${{ secrets.WEBGPU_BRIDGE_ASSETS_PAT }}'
jobs:
  publish-assets:
    steps:
      - name: Safe publication
        env:
          <<: *publication-env
        run: |
          set -euo pipefail
          # The credential guard must be the first executable block after setup.
          if [ -z "${RELEASE_CREDENTIAL}" ]; then
            echo "error: WEBGPU_BRIDGE_ASSETS_PAT is required for asset publication"
            exit 1
          fi
          git -C assets push origin main
"""
    require(
        not validate_publication_pat_contract(safe_fixture, expected),
        "publication PAT contract rejected its safe YAML merge/alternate-variable fixture",
        errors,
    )

    canonical_empty_guard = (
        'if [ -z "${TOKEN_ALIAS}" ]; then\n'
        f'  echo "{PUBLICATION_PAT_GUARD_ERROR}"\n'
        "  exit 1\n"
        "fi"
    )
    canonical_script = "set -euo pipefail\n" + canonical_empty_guard
    canonical_guard_error = "canonical executable empty-token guard for TOKEN_ALIAS"
    unsafe_scripts = {
        "missing guard": ("set -euo pipefail\ntrue", canonical_guard_error),
        "non-executing false-and-exit empty-token block": (
            "set -euo pipefail\n"
            'false && if [ -z "${TOKEN_ALIAS}" ]; then\n'
            f'  echo "{PUBLICATION_PAT_GUARD_ERROR}"\n'
            "  exit 1\n"
            "fi",
            canonical_guard_error,
        ),
        "single-quoted non-expanding parameter guard": (
            "set -euo pipefail\n"
            ": '${TOKEN_ALIAS:?WEBGPU_BRIDGE_ASSETS_PAT is required}'\n"
            "git -C assets push origin main",
            canonical_guard_error,
        ),
        "indirect gh invocation before guard": (
            "set -euo pipefail\n"
            "client=gh\n"
            '"${client}" api repos/example/assets\n'
            + canonical_empty_guard,
            canonical_guard_error,
        ),
        "set -x": (
            canonical_script + "\nset -x",
            "set -x/xtrace",
        ),
        "printenv": (
            canonical_script + "\nprintenv",
            "through printenv",
        ),
        "bare env": (
            canonical_script + "\nenv",
            "through bare env",
        ),
        "absolute-path env": (canonical_script + "\n/usr/bin/env", "through bare env"),
        "echo leakage": (
            canonical_script + '\necho "${TOKEN_ALIAS}"',
            "echo/printf of TOKEN_ALIAS",
        ),
        "printf leakage": (
            canonical_script + "\nprintf '%s' \"$TOKEN_ALIAS\"",
            "echo/printf of TOKEN_ALIAS",
        ),
        "secret expansion in guard error": (
            "set -euo pipefail\n"
            'if [ -z "${TOKEN_ALIAS}" ]; then\n'
            '  echo "error: ${TOKEN_ALIAS} is required"\n'
            "  exit 1\n"
            "fi",
            canonical_guard_error,
        ),
    }
    for fixture_name, (script, expected_error) in unsafe_scripts.items():
        indented_script = "".join(
            f"          {line}\n" for line in script.splitlines()
        )
        fixture = f"""
jobs:
  publish-assets:
    steps:
      - name: Safe publication
        env:
          TOKEN_ALIAS: "${{{{secrets.WEBGPU_BRIDGE_ASSETS_PAT}}}}"
        run: |-
{indented_script}"""
        fixture_errors = validate_publication_pat_contract(fixture, expected)
        require(
            any(expected_error in error for error in fixture_errors),
            f"publication PAT contract did not reject adversarial fixture "
            f"{fixture_name} for the expected reason ({expected_error})",
            errors,
        )

    job_container_credential_fixture = """
jobs:
  publish-assets:
    container:
      image: example.invalid/publisher:latest
      credentials:
        username: publisher
        password: ${{ secrets.WEBGPU_BRIDGE_ASSETS_PAT }}
    steps:
      - name: Safe publication
        run: echo safe
"""
    fixture_errors = validate_publication_pat_contract(
        job_container_credential_fixture, expected
    )
    require(
        any(
            "job publish-assets properties references "
            f"{PUBLICATION_PAT_NAME} outside resolved env" in error
            for error in fixture_errors
        ),
        "publication PAT contract did not reject a job-level container "
        "credentials password PAT reference",
        errors,
    )


def extract_section(
    relative_path: str,
    content: str,
    start_marker: str,
    end_marker: str,
    errors: list[str],
) -> str:
    start = content.find(start_marker)
    if start < 0:
        errors.append(f"{relative_path} is missing the expected section marker: {start_marker}")
        return ""
    end = content.find(end_marker, start + len(start_marker))
    if end < 0:
        errors.append(f"{relative_path} is missing the expected section terminator: {end_marker}")
        return content[start:]
    return content[start:end]


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
    release_qualification = read_required("scripts/release_qualification.py", errors)
    release_qualification_test = read_required(
        "scripts/release_qualification_test.py", errors
    )
    bridge_qualification = read_required(
        ".github/workflows/bridge_qualification.yml", errors
    )
    candidate = read_required(".github/workflows/bridge_candidate.yml", errors)
    embedding_contract = read_required("scripts/embedding_json_contract_test.mjs", errors)
    worker_token_contract = read_required(
        "scripts/worker_token_coalescing_test.mjs", errors
    )
    worker_state_contract = read_required(
        "scripts/worker_runtime_state_test.mjs", errors
    )
    operation_queue_contract = read_required(
        "scripts/bridge_operation_queue_test.mjs", errors
    )
    operation_lifecycle_contract = read_required(
        "scripts/bridge_operation_lifecycle_test.mjs", errors
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
    version_contents = read_required("llama_cpp.version", errors)
    version = (
        version_contents[:-1]
        if version_contents.endswith("\n")
        else version_contents
    )
    emscripten_version = read_required("emsdk.version", errors).strip()
    agents = read_required("AGENTS.md", errors)
    readme = read_required("README.md", errors)
    api_docs = read_required("docs/api.md", errors)
    contributing = read_required("CONTRIBUTING.md", errors)
    release_contract = read_required("scripts/release_contract.py", errors)
    release_contract_test = read_required("scripts/release_contract_test.py", errors)
    publication_state = read_required("scripts/release_publication_state.py", errors)
    publication_state_test = read_required(
        "scripts/release_publication_state_test.py", errors
    )
    orchestrator = read_required("scripts/stable_release_orchestrator.py", errors)
    orchestrator_test = read_required(
        "scripts/stable_release_orchestrator_test.py", errors
    )
    if orchestrator and orchestrator_test:
        run_required_python_contract(
            "scripts/stable_release_orchestrator_test.py", errors
        )
    workflow_input_transport = read_required(
        "scripts/workflow_input_transport_test.mjs", errors
    )
    agents_publication = extract_section(
        "AGENTS.md",
        agents,
        "- Publish workflow: `.github/workflows/publish_assets.yml`",
        "\n## Change Boundaries",
        errors,
    )
    readme_publication = extract_section(
        "README.md",
        readme,
        "## Publishing",
        "\n## Maintainer Docs",
        errors,
    )
    readme_publication_text = " ".join(readme_publication.split())
    readme_publication_credentials = extract_section(
        "README.md",
        readme_publication,
        "Required externally configured credentials:",
        "\nEvery request supplies",
        errors,
    )
    readme_publication_credentials_text = " ".join(
        readme_publication_credentials.split()
    )
    typechecked_files = list_typescript_files(errors)
    verify_environment_job = publish.split(
        "\n  verify-publication-environment:\n", 1
    )[-1].split("\n  publish-assets:\n", 1)[0]
    publish_job = publish.split("\n  publish-assets:\n", 1)[-1]

    require_well_formed_markdown_tables("docs/api.md", api_docs, errors)
    require_publication_pat_contract_self_tests(errors)

    for name, workflow in (
        ("ci.yml", ci),
        ("publish_assets.yml", publish),
        ("auto_llama_cpp_update.yml", auto_update),
        ("bridge_qualification.yml", bridge_qualification),
        ("bridge_candidate.yml", candidate),
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
        '"test:operation-queue"' in package_json
        and "npm run test:operation-queue" in package_json,
        "check:js must define and run the bridge operation queue contract",
        errors,
    )
    require(
        "Bridge operation queue tests passed" in operation_queue_contract,
        "the operation queue contract must retain its aggregate success output; the Node suite owns its case matrix",
        errors,
    )
    require(
        '"test:operation-lifecycle"' in package_json
        and "npm run test:operation-lifecycle" in package_json,
        "check:js must define and run the bridge operation lifecycle contract",
        errors,
    )
    require(
        "Bridge operation lifecycle tests passed" in operation_lifecycle_contract,
        "the operation lifecycle contract must retain its aggregate success output; the Node suite owns its case matrix",
        errors,
    )
    require(
        "_runExclusive(" in js_source and "_runExclusive(" in js_output,
        "the single-writer operation queue must be present in bridge source and in the "
        "generated bridge output",
        errors,
    )
    require(
        "_captureDirectRuntimeState()" in js_source
        and "_captureDirectRuntimeState()" in js_output,
        "the facade must snapshot direct runtime state so synchronous getters never ccall "
        "outside the operation queue",
        errors,
    )
    require(
        "single-writer" in api_docs and "FIFO" in api_docs,
        "docs/api.md must document the single-writer FIFO operation queue",
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
        and "Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/ca27d74bc954b73dadab5b71ca265d87fc861a7c" in ci
        and "LLAMA_WEBGPU_TTS_MODEL_SHA256" in ci
        and "LLAMA_WEBGPU_TTS_MMPROJ_SHA256" in ci
        and "run_text_to_speech_smoke" not in ci
        and "Run Qwen3-TTS browser smoke" not in ci
        and "Download verified text-to-speech smoke models" not in ci
        and "LLAMADART_WEBGPU_TTS_API_VERSION" in tts_contract
        and 'RUNTIME_MODES = ("direct", "worker")' in tts_smoke,
        "CI must syntax-check the TTS smoke and run its static API contract without executing the heavy real-model gate on hosted runners",
        errors,
    )
    for name, workflow in (
        ("ci.yml", ci),
        ("bridge_candidate.yml", candidate),
        ("publish_assets.yml", publish),
    ):
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

    # The ordinary pin follows the same upstream channels the release contract
    # emits, so a stable llama.cpp release is a valid development pin too.
    try:
        parse_upstream_tag(version)
        version_is_exact = True
    except ContractError:
        version_is_exact = False
    require(
        version_is_exact and "\n" not in version and "\r" not in version,
        "llama_cpp.version must contain one exact llama.cpp release tag: stable "
        "vMAJOR.MINOR.PATCH like v0.2.0 or development bNNNN like b9165",
        errors,
    )
    require(
        re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", emscripten_version) is not None,
        "emsdk.version must contain one exact Emscripten semantic version",
        errors,
    )
    for name, workflow in (("ci.yml", ci), ("bridge_candidate.yml", candidate)):
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
        "--emscripten-version \"${EMSCRIPTEN_VERSION}\"" in candidate
        and "--emscripten-version \"${EMSCRIPTEN_VERSION}\"" in publish
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
        and "BRIDGE_REPO: leehack/llama-web-bridge" in auto_update
        and "ASSETS_REPO: leehack/llama-web-bridge-assets" in auto_update
        and "release-candidates.json" in auto_update
        and "orchestration-plan.json" in auto_update
        and "select-stable-native-backlog" in auto_update
        and "gh release view" not in auto_update
        and "gh release download" not in auto_update
        and "releases/assets/${asset_id}" in auto_update
        and "validate-native-release" in auto_update
        and "resolve-tag-commit" in auto_update
        and "scripts/stable_release_orchestrator.py scan-native" in auto_update
        and "scripts/stable_release_orchestrator.py orchestrate-backlog" in auto_update
        and '--channel "${REQUESTED_CHANNEL}"' in auto_update
        and "ORCHESTRATOR_DISPATCH_TOKEN" not in auto_update
        and "WEBGPU_BRIDGE_ASSETS_PAT: ${{ secrets.WEBGPU_BRIDGE_ASSETS_PAT }}"
        in auto_update
        and "name: bridge-assets-publication" in auto_update
        and "validate-environment" in auto_update
        and "gh workflow run" not in auto_update
        and "create-pull-request" not in auto_update
        and "git push" not in auto_update
        and "Manual development scans only prepare exact candidate inputs" in auto_update
        and "env.REQUESTED_CHANNEL != 'stable'" in auto_update,
        "automatic scans must download every post-baseline stable provenance by unique asset id, reuse only the existing environment-scoped publication credential, advance stable publication only through stable_release_orchestrator.py, and keep development scans scan-only",
        errors,
    )
    owner_manual_gate = (
        "github.event_name == 'workflow_dispatch' && "
        "github.actor == github.repository_owner && "
        "github.triggering_actor == github.repository_owner"
    )
    require(
        auto_update.count(owner_manual_gate) == 2
        and auto_update.rfind(owner_manual_gate) < auto_update.find("environment:"),
        "both automatic-orchestration jobs must reject non-owner manual callers before the publication environment can expose its credential",
        errors,
    )
    owner_workflow_run_gate = (
        "github.event_name == 'workflow_run' && "
        "github.event.workflow_run.conclusion == 'success' && "
        "github.event.workflow_run.head_branch == github.event.repository.default_branch && "
        "github.event.workflow_run.actor.login == github.repository_owner && "
        "github.event.workflow_run.triggering_actor.login == github.repository_owner"
    )
    continuation_proof_name = (
        "Prove the exact workflow continuation before environment use"
    )
    continuation_proof = ""
    continuation_marker = f"      - name: {continuation_proof_name}\n"
    if continuation_marker in auto_update:
        continuation_proof = auto_update.split(continuation_marker, 1)[1].split(
            "\n      - name:", 1
        )[0]
    require(
        "workflow_run:" in auto_update
        and "- Build Exact Bridge Candidate" in auto_update
        and "- Qualify Exact Bridge Candidate" in auto_update
        and "- Publish Exact Qualified Bridge Assets" in auto_update
        and "types: [completed]" in auto_update
        and auto_update.count(owner_workflow_run_gate) == 2
        and auto_update.rfind(owner_workflow_run_gate)
        < auto_update.find("environment:")
        and continuation_proof
        and "scripts/release_qualification.py verify-run" in continuation_proof
        and "--run-attempt 1" in continuation_proof
        and (
            ".github/workflows/bridge_candidate.yml)\n"
            "              artifact_name=exact-webgpu-bridge-dist"
        )
        in continuation_proof
        and (
            ".github/workflows/bridge_qualification.yml)\n"
            "              artifact_name=qualification-attestation"
        )
        in continuation_proof
        and (
            ".github/workflows/publish_assets.yml)\n"
            "              artifact_name=bridge-qualification-outcome"
        )
        in continuation_proof
        and auto_update.find(continuation_proof_name)
        < auto_update.find("environment:"),
        "event-driven continuation must listen only to the three exact stage workflows and reject unsuccessful, non-default-branch, or non-owner runs before environment use",
        errors,
    )
    require(
        "workflow_runs" in orchestrator
        and "def parse_workflow_runs" in orchestrator
        and "def plan_pipeline" in orchestrator
        and "def compute_correlation_id" in orchestrator
        and "def select_next_release_target" in orchestrator
        and "def verify_published_release" in orchestrator
        and "def verify_candidate_run" in orchestrator
        and "def verify_qualification_run" in orchestrator
        and "def require_immutable_release_governance" in orchestrator
        and "def require_publication_environment" in orchestrator
        and "def require_orchestration_caller" in orchestrator
        and "def select_stable_native_backlog" in orchestrator
        and "orchestrate-backlog" in orchestrator
        and "validate_immutable_release_governance" in orchestrator
        and "validate_release_immutability" in orchestrator
        and "validate_release_attestation" in orchestrator
        and "validate_publication_environment" in orchestrator
        and "release_attestation" in orchestrator
        and "rq.validate_workflow_run" in orchestrator
        and "rq.validate_artifact_inventory" in orchestrator
        and "rq.load_candidate" in orchestrator
        and "rq.verify_attestation" in orchestrator
        and "OrchestrationAction.BLOCKED" in orchestrator
        # The routine path must dispatch qualification itself; no state may wait
        # for a maintainer-supplied attestation.
        and "OrchestrationAction.DISPATCH_QUALIFICATION" in orchestrator
        and "WAITING_FOR_ATTESTATION" not in orchestrator
        and "def qualification_run_name" in orchestrator
        and "QUALIFICATION_DISPATCH_INPUTS" in orchestrator
        and '"--ref",' in orchestrator
        and '"--json",' in orchestrator,
        "stable_release_orchestrator must reuse the release, qualification, environment, and immutable-attestation validators for every live transition and published noop",
        errors,
    )
    require(
        "def require_exact_dispatch_inputs" in orchestrator
        and "WORKFLOW_DISPATCH_INPUTS" in orchestrator
        and "def require_stable_provenance" in orchestrator
        and "def require_channel" in orchestrator
        and '"assets_immutable_releases_enabled": "true"' not in orchestrator
        and '"publish_approved": "true"' not in orchestrator
        and "_candidate_fingerprint_marker" in orchestrator,
        "the orchestrator must dispatch exactly each workflow's declared inputs, derive governance and approval booleans only from live proofs, orchestrate the stable channel only, and bind published bytes to their exact candidate fingerprint",
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
        and "qualification_run_id:" in publish
        and "REQUESTED_ASSETS_REPO: ${{ inputs.assets_repo }}" in publish
        and 'if [ "${REQUESTED_ASSETS_REPO}" != "${APPROVED_ASSETS_REPO}" ]'
        in publish
        and "release_capabilities:" not in publish
        and "publish_approved:" in publish
        and "llama_cpp.version" not in publish,
        "bridge-owned publish_assets.yml must expose exact orchestrator/bridge/upstream/native/qualification/output inputs without a bridge pin",
        errors,
    )
    require(
        "BRIDGE_REPO: leehack/llama-web-bridge" in publish
        and len(re.findall(r"repository: leehack/llama-web-bridge\s*$", publish, re.MULTILINE)) == 5
        and publish.count("ref: ${{ github.sha }}") == 3
        and '--bridge-repo "${BRIDGE_REPO}"' in candidate
        and 'repos/${GITHUB_REPOSITORY}' not in publish
        and 'repos/${GITHUB_REPOSITORY}' not in candidate,
        "dispatch-only publication must always checkout and verify the owning bridge repository",
        errors,
    )
    require(
        "publish_approved=true is required before publication" in publish
        and "Verify externally configured publication environment" in publish
        and "release_contract.py validate-environment" in publish
        and publish.count("deployment-branch-policies") == 2
        and publish.count("--branch-policies-json") == 2
        and verify_environment_job.count("GH_TOKEN: ${{ github.token }}") == 1
        and "secrets." not in verify_environment_job
        and verify_environment_job.count(
            'gh api "repos/${BRIDGE_REPO}/environments/bridge-assets-publication"'
        )
        == 1
        and verify_environment_job.count("deployment-branch-policies") == 1
        and "scripts/release_contract.py validate-environment"
        in verify_environment_job
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
        and "required reviewers" not in publish
        and "prevent_self_review" not in publish
        and "push:" not in publish.split("permissions:", 1)[0]
        and "schedule:" not in publish,
        "cross-repository publication must require explicit input confirmation plus a preverified fail-closed environment policy and have no automatic trigger",
        errors,
    )
    obsolete_environment_secret_contract = (
        "BRIDGE_PUBLICATION_ENV_READ_TOKEN",
        "bridge-assets-publication/secrets",
        "--environment-secrets-json",
        "normalize-environment-secrets",
        "publication-environment-secret-pages",
    )
    require(
        all(
            obsolete not in content
            for obsolete in obsolete_environment_secret_contract
            for content in (
                publish,
                agents,
                readme_publication,
                contributing,
                release_contract,
                release_contract_test,
            )
        ),
        "the single-credential publication contract must not reintroduce a second token or environment-secret inventory API path",
        errors,
    )
    require(
        "solo-maintainer publication contract does not require a reviewer rule"
        in readme_publication_text
        and "administrator bypass disabled" in readme_publication_text
        and "exact `main` branch policy" in readme_publication_text
        and "environment-scoped secret" in readme_publication_text
        and "`WEBGPU_BRIDGE_ASSETS_PAT`" in readme_publication_credentials
        and "only externally configured publication credential"
        in readme_publication_credentials
        and readme_publication_credentials.count("\n- ") == 1
        and readme_publication_credentials_text
        == (
            "Required externally configured credentials: - "
            "`WEBGPU_BRIDGE_ASSETS_PAT` (read access to provenance repositories "
            "and write access to `leehack/llama-web-bridge-assets`, stored only "
            "in the publication environment). This is the only externally "
            "configured publication credential."
        )
        and "`github.token`" in readme_publication_text
        and "before entering the privileged job and again immediately before"
        in readme_publication_text
        and "immediately before the first publication-PAT-bearing step"
        in readme_publication_text
        and "trusted workflow commit on `main`" in readme_publication_text
        and "historical bridge source supplies only exact candidate harness "
        "bytes, toolchain pin, and build identity"
        in readme_publication_text
        and "credential is non-empty" in readme_publication_text
        and "does not print its value" in readme_publication_text
        and readme_publication_text.lower().count("reviewer") == 1
        and "self-review" not in readme_publication_text.lower()
        and "self review" not in readme_publication_text.lower()
        and re.search(
            r"\b(?:second|additional|separate|another|distinct)\b.{0,64}"
            r"\b(?:credential|token|secret)\b",
            readme_publication_text,
            re.IGNORECASE,
        )
        is None
        and all(
            obsolete not in readme_publication_text
            for obsolete in (
                "self-review prevented",
                "pinned repository-owner maintainer reviewer",
                "required-reviewer inventory",
                "after reviewer approval",
            )
        ),
        "README Publishing guidance must match the solo-maintainer, single-credential, main-only publication contract",
        errors,
    )
    post_approval_environment_check = publish_job.find(
        "Revalidate publication environment policy before using PAT"
    )
    first_pat_reference = publish_job.find("secrets.WEBGPU_BRIDGE_ASSETS_PAT")
    require(
        publish.count("release_contract.py validate-environment") == 2
        and re.search(
            r"publish-assets:\s+.*?permissions:\s+actions: read\s+contents: read",
            publish,
            re.DOTALL,
        )
        is not None
        and post_approval_environment_check >= 0
        and first_pat_reference > post_approval_environment_check
        and "GH_TOKEN: ${{ github.token }}" in publish_job[
            post_approval_environment_check:first_pat_reference
        ]
        and "secrets." not in publish_job[
            post_approval_environment_check:first_pat_reference
        ]
        and publish_job[post_approval_environment_check:first_pat_reference].count(
            "gh api"
        )
        == 2
        and "python3 publication-policy/scripts/release_contract.py validate-environment"
        in publish_job[post_approval_environment_check:first_pat_reference]
        and publish_job[
            post_approval_environment_check:first_pat_reference
        ].count("\n      - name:")
        == 1,
        "the privileged job must revalidate environment identity, bypass, and main-branch policy with github.token after the approval assertion and immediately before its first PAT-bearing step",
        errors,
    )
    errors.extend(
        validate_publication_pat_contract(publish, EXPECTED_PUBLICATION_PAT_STEPS)
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
    native_request_validation = publish.find(
        "scripts/release_contract.py validate-native-request"
    )
    upstream_tag_lookup = publish.find(
        "git ls-remote https://github.com/ggml-org/llama.cpp.git"
    )
    native_tag_lookup = publish.find(
        'git ls-remote "https://github.com/${NATIVE_REPO}.git"'
    )
    require(
        "git ls-remote https://github.com/ggml-org/llama.cpp.git" in publish
        and '"refs/tags/${UPSTREAM_TAG}" "refs/tags/${UPSTREAM_TAG}^{}"' in publish
        and "resolved upstream tag commit does not match upstream_commit" in publish
        and "git ls-remote \"https://github.com/${NATIVE_REPO}.git\"" in publish
        and "\"refs/tags/${NATIVE_RELEASE_TAG}\" \"refs/tags/${NATIVE_RELEASE_TAG}^{}\""
        in publish
        and "scripts/release_contract.py resolve-tag-commit" in publish
        and "NATIVE_TAG_COMMIT: ${{ steps.native_tag.outputs.native_tag_commit }}"
        in publish
        and "--native-tag-commit \"${NATIVE_TAG_COMMIT}\"" in publish
        and "resolved native release tag commit does not match manifest native_commit"
        in release_contract,
        "publication must resolve the immutable native release tag commit itself and reject any release whose manifest native_commit differs, because target_commitish reports a mutable branch",
        errors,
    )
    require(
        0 <= native_request_validation < upstream_tag_lookup < native_tag_lookup
        and "--native-release-tag \"${NATIVE_RELEASE_TAG}\"" in publish[
            native_request_validation:native_tag_lookup
        ]
        and "--upstream-commit \"${UPSTREAM_COMMIT}\"" in publish[
            native_request_validation:native_tag_lookup
        ]
        and "--manifest-sha256 \"${NATIVE_MANIFEST_SHA256}\"" in publish[
            native_request_validation:native_tag_lookup
        ],
        "publication must validate native tag, upstream commit, and manifest SHA-256 inputs before the first native network request",
        errors,
    )
    require(
        publish.count(
            "publication-policy/scripts/release_contract.py resolve-tag-commit"
        )
        == 2
        and publish.count("fetched asset release tag changed after immutable resolution")
        == 2
        and "awk '$2 ~ /\\^\\{\\}$/ {print $1}'" not in publish,
        "publication preflight and recovery must resolve existing asset tags with the strict annotated/lightweight tag parser and verify the fetched ref did not change",
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
        and "upload-release-assets" not in publish
        and "gh release upload" not in publish
        and "immutable-release-assets-missing" in publication_state
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
        >= 3
        # Publication compares against the honest candidate manifest gates: the
        # heavy gates are proven by the separate qualification run, never by the
        # candidate build, so a re-query claiming a candidate-run ASR/TTS pass
        # fails closed.
        and publish.count(
            'speech_to_text:"required-automated-qualification",'
        )
        >= 3
        and 'speech_to_text:"passed"' not in publish
        and 'text_to_speech:"passed"' not in publish
        and ".candidate_fingerprint == env.CANDIDATE_FINGERPRINT" in publish
        and "persist-credentials: false" in publish
        and "sha256sum --check sha256sums.txt" in publish
        and "git -C assets-repo push --atomic" in publish,
        "publication must classify exact retry states, reject failed classifier re-queries, emit outcomes, lock credentials, and recheck after mutation",
        errors,
    )
    governance_check = publish_job.find(
        "release_contract.py \\\n            validate-immutable-release-governance"
    )
    first_ref_mutation = publish_job.find("Apply only the classified ref mutation")
    require(
        # Governance is proven through the supported repository API, whose 404
        # covers both "disabled" and "unreadable", so only an exact 200 body
        # reaching the validator can pass -- and it must pass before publication
        # pushes anything.
        publish_job.count('"repos/${ASSETS_REPO}/immutable-releases"') >= 3
        and "validate-immutable-release-governance" in publish_job
        and "--repository \"${ASSETS_REPO}\"" in publish_job
        and 0 <= governance_check < first_ref_mutation
        and "validate_immutable_release_governance" in release_contract
        and "immutable releases are not enabled for" in release_contract
        and "must be an explicit JSON boolean" in release_contract
        and "test_immutable_release_governance_must_be_enabled" in release_contract_test
        and "ASSETS_IMMUTABLE_RELEASES_ENABLED: ${{ inputs.assets_immutable_releases_enabled }}"
        in candidate
        and '"${ASSETS_IMMUTABLE_RELEASES_ENABLED}" != "true"' in candidate
        and "--argjson immutable \"${ASSETS_IMMUTABLE_RELEASES_ENABLED}\"" in candidate
        and "assets_immutable_releases_enabled:$immutable" in candidate
        and "CANDIDATE_PREQUALIFICATION_ARTIFACT_NAME: bridge-candidate-prequalification"
        in publish
        and "CANDIDATE_PREQUALIFICATION_ARTIFACT_ID" in publish
        and "candidate-prequalification.zip" in publish
        and "validate-candidate-prequalification" in publish
        and "validate_candidate_prequalification" in release_contract
        and "test_candidate_prequalification_binds_true_governance_assertion"
        in release_contract_test
        and "immutable-governance-unverified" in publish_job,
        "candidate dispatch and publication must fail closed unless the assets repository already enables immutable releases, proven at publication through the supported repository API",
        errors,
    )
    require(
        # A created release is only trustworthy once GitHub itself reports it as
        # immutable and has signed a release attestation over the exact bytes.
        publish_job.count('"repos/${ASSETS_REPO}/releases/tags/${RELEASE_TAG}"') >= 1
        and '"repos/${ASSETS_REPO}/releases/${published_release_id}"' in publish_job
        and 'gh release verify "${RELEASE_TAG}" --repo "${ASSETS_REPO}"' in publish_job
        and "--format json" in publish_job
        # An unsupported runner CLI must fail before publication pushes, not
        # after it has created a release nothing can verify.
        and 0 <= publish_job.find("gh release verify --help | grep -F -- '--format'")
        < first_ref_mutation
        and "verify-immutable-publication" in publish_job
        and '--release-by-id-json "${RUNNER_TEMP}/published-release-by-id.json"'
        in publish_job
        and "immutable-publication-unverified" in publish_job
        and "verify_immutable_publication" in publication_state
        and "candidate_publication_digests" in publication_state
        and "validate_release_immutability" in release_contract
        and "validate_release_attestation" in release_contract
        and "predicate does not bind the exact release id" in release_contract
        and "release_id=resolved_id" in publication_state
        and "https://in-toto.io/attestation/release/v0.2" in release_contract
        and "https://dotcom.releases.github.com" in release_contract
        and "does not report immutability" in release_contract
        and "target_commitish does not bind the exact tag commit" in release_contract
        and '_require_utc_timestamp(release.get("published_at"), "published_at")'
        in release_contract
        and "DSSE envelope has no signature" in release_contract
        and "verified_timestamps" in release_contract
        and "test_created_release_must_read_back_immutable" in release_contract_test
        and "test_release_attestation_binds_the_exact_published_release"
        in release_contract_test
        and "test_published_release_must_be_immutable_and_attested"
        in publication_state_test
        and "test_non_immutable_or_malformed_readback_fails_closed"
        in publication_state_test
        and "test_release_attestation_must_cover_the_published_candidate"
        in publication_state_test
        and "test_immutable_publication_cli_reports_exact_failures"
        in publication_state_test
        and "Every complete state is verified" in publish_job
        and 'touch "${RUNNER_TEMP}/ref-push-attempted"' in publish_job
        and 'touch "${RUNNER_TEMP}/release-mutation-attempted"' in publish_job
        and "steps.mutate.outcome == 'failure'" in publish_job
        and 'fallback_state="mutation-unknown"' in publish_job
        and 'fallback_mutated=null' in publish_job
        and publish_job.find('elif [ "${post_action}" = "none" ]')
        < publish_job.find("Every complete state is verified"),
        "publication must read the new release back by tag and by ID, require an explicit immutable:true, and verify GitHub's signed release attestation over the exact published assets",
        errors,
    )
    require(
        # An immutable release cannot be repaired, so a failed immutability or
        # attestation check must never provoke a cleanup attempt.
        all(
            destructive not in publish
            for destructive in (
                "gh release delete",
                "gh release edit",
                "gh release upload",
                "git push --delete",
                "--force-with-lease",
                "push --force",
                "-X DELETE",
            )
        )
        and "never deletes, retags, overwrites, or repairs" in publish,
        "publication must never attempt cleanup, deletion, retagging, or mutation when an immutability or attestation check fails",
        errors,
    )
    require(
        all(
            "immutable-releases" in document
            and "https://in-toto.io/attestation/release/v0.2" in document
            and "gh release verify" in document
            and "immutable-publication-unverified" in document
            for document in (agents_publication, readme_publication, contributing)
        )
        and "assets_immutable_releases_enabled" in agents
        and "assets_immutable_releases_enabled" in readme_publication
        and "assets_immutable_releases_enabled=true" in contributing
        # The immutable automation baseline is a fresh identity; v0.1.38 is
        # historical and never a repair/rebuild path.
        and "### Immutable Automation Baseline" in agents
        and "kanban:t_7f112b91:web-v0.1.39" in agents
        and "`release_tag`: `v0.1.39`" in agents
        and "`release_rebuild`: `0`" in agents,
        "AGENTS.md, README.md, and CONTRIBUTING.md must document the immutable-release governance gate, the post-publication readback and attestation verification, and the verified immutable automation baseline",
        errors,
    )
    require(
        "Run durable state persistence gate" in candidate
        and "Run durable multimodal gate" in candidate
        and "WEBGPU_BRIDGE_BUILD_MEM64: 1" in candidate
        and candidate.count("./scripts/build_bridge.sh") == 1
        and "GITHUB_RUN_ATTEMPT_STRING: ${{ github.run_attempt }}" in candidate
        and '"${GITHUB_RUN_ATTEMPT_STRING}" != "1"' in candidate
        and "STATE_CONCLUSION:" in candidate
        and "MULTIMODAL_CONCLUSION:" in candidate
        and "exact-webgpu-bridge-dist" in candidate
        and "bridge-candidate-prequalification" in candidate
        and "pending-automated-qualification" in candidate
        and "contains(inputs.release_capabilities" not in candidate
        # The candidate workflow is the only place assets are built, and it is
        # deliberately unprivileged: no publication environment, no PAT.
        and "WEBGPU_BRIDGE_ASSETS_PAT" not in candidate
        and "environment:" not in candidate
        and "Run Qwen3-ASR browser smoke" not in candidate
        and "Run Qwen3-TTS browser smoke" not in candidate,
        "the candidate workflow must build wasm32/memory64 once, run the hosted state and multimodal gates, and upload an honest prequalification record without publication credentials",
        errors,
    )
    require(
        all(
            "DISPATCHING_ACTOR: ${{ github.triggering_actor }}" in workflow
            and "REPOSITORY_OWNER: ${{ github.repository_owner }}" in workflow
            and '"${DISPATCHING_ACTOR}" != "${REPOSITORY_OWNER}"' in workflow
            for workflow in (candidate, bridge_qualification, publish)
        ),
        "candidate, qualification, and publication dispatches must be restricted to the repository owner identity",
        errors,
    )
    require(
        # A dispatching actor alone leaves the run's own actor and a re-run
        # unproven, so both privileged workflows pin them too.
        all(
            '"${GITHUB_ACTOR}" != "${REPOSITORY_OWNER}"' in workflow
            and '"${GITHUB_RUN_ATTEMPT}" != "1"' in workflow
            for workflow in (bridge_qualification, publish)
        )
        and "workflow run actor must be" in release_qualification
        and "workflow run triggering_actor must be" in release_qualification
        and "test_wrong_workflow_run_actor_rejected" in release_qualification_test
        and "test_wrong_workflow_run_triggering_actor_rejected"
        in release_qualification_test,
        "qualification and publication must pin the run actor and refuse replayed run attempts",
        errors,
    )
    require(
        # The attestation must bind the exact artifact it was produced from, and
        # every verification site must re-assert that binding.
        "candidate_artifact_id" in release_qualification
        and "candidate_run_attempt" in release_qualification
        and "attestation candidate_run_attempt must be 1" in release_qualification
        and "candidate_workflow_path" in release_qualification
        and "qualification_workflow_path" in release_qualification
        and "qualification_run_id" in release_qualification
        and "qualification_run_attempt" in release_qualification
        and "qualification_source_sha" in release_qualification
        and publish.count('--candidate-artifact-id "${CANDIDATE_ARTIFACT_ID}"') == 2
        and publish.count("--candidate-run-attempt 1") == 2
        and publish.count('--qualification-run-id "${QUALIFICATION_RUN_ID}"') == 2
        and publish.count("--qualification-run-attempt 1") == 2
        and publish.count(
            '--qualification-source-sha "${QUALIFICATION_SOURCE_SHA}"'
        )
        == 2
        and "candidate_artifact_id: ${{ steps.runs.outputs.candidate_artifact_id }}"
        in publish
        and "CANDIDATE_ARTIFACT_ID: ${{ needs.verify-candidate-and-qualification.outputs.candidate_artifact_id }}"
        in publish
        # Nothing exports a qualification artifact id across jobs; publication
        # verifies and consumes it within the same job.
        and "qualification_artifact_id:" not in publish
        and '--candidate-artifact-id "${CANDIDATE_ARTIFACT_ID}"'
        in bridge_qualification
        and "--candidate-run-attempt 1" in bridge_qualification
        and '--qualification-run-id "${GITHUB_RUN_ID}"' in bridge_qualification
        and "--qualification-run-attempt 1" in bridge_qualification
        and '--qualification-source-sha "${GITHUB_SHA}"' in bridge_qualification
        and "test_candidate_artifact_id_mismatch_rejected" in release_qualification_test
        and "test_candidate_run_attempt_mismatch_rejected"
        in release_qualification_test
        and "test_qualification_workflow_and_run_identity_are_bound"
        in release_qualification_test
        and "test_qualification_run_identity_comes_from_actions_environment"
        in release_qualification_test
        and "test_attestation_receipt_contains_artifact_id_and_run_attempt"
        in release_qualification_test,
        "the attestation must bind the exact candidate artifact and qualification producer identities at every verification site",
        errors,
    )
    require(
        # Archive parsing must refuse a hostile archive from its metadata, before
        # a single byte is decompressed into a trusted destination.
        'artifact_type="candidate"' in publish
        and 'artifact_type="attestation"' in publish
        and 'artifact_type="candidate"' in bridge_qualification
        and "ALLOWED_COMPRESS_TYPES" in release_qualification
        and "MAX_COMPRESSION_RATIO" in release_qualification
        and "MAX_CANDIDATE_MEMBER_BYTES" in release_qualification
        and "MAX_CANDIDATE_TOTAL_BYTES" in release_qualification
        and "CANDIDATE_ALLOWED_MEMBERS" in release_qualification
        and "ATTESTATION_ALLOWED_MEMBERS" in release_qualification
        and "_validate_local_header" in release_qualification
        and "_stage_artifact_archive" in release_qualification
        and "_preflight_zip_end_record" in release_qualification
        and "_ZIP_ENCRYPTION_FLAGS" in release_qualification
        and "_ZIP_DATA_DESCRIPTOR_FLAG" in release_qualification
        and "dst_dir_fd=destination_fd" in release_qualification
        and "_unlink_placed_members" in release_qualification
        and "test_attestation_zip_bomb_rejected_on_compression_ratio"
        in release_qualification_test
        and "test_candidate_zip_bomb_rejected_on_compression_ratio"
        in release_qualification_test
        and "test_oversized_candidate_member_rejected_before_extraction"
        in release_qualification_test
        and "test_oversized_attestation_member_rejected_before_extraction"
        in release_qualification_test
        and "test_total_uncompressed_size_is_bounded_before_extraction"
        in release_qualification_test
        and "test_unsupported_compression_method_rejected"
        in release_qualification_test
        and "test_local_header_disagreeing_with_central_directory_rejected"
        in release_qualification_test
        and "test_local_header_name_disagreeing_with_central_directory_rejected"
        in release_qualification_test
        and "test_overlapping_members_rejected" in release_qualification_test
        and "test_member_payload_cannot_overlap_the_central_directory"
        in release_qualification_test
        and "test_unclaimed_gap_before_central_directory_rejected"
        in release_qualification_test
        and "test_encrypted_zip_member_rejected" in release_qualification_test
        and "test_data_descriptor_zip_member_rejected" in release_qualification_test
        and "test_signed_github_style_data_descriptor_is_exactly_validated"
        in release_qualification_test
        and "test_local_and_central_flag_mismatch_rejected"
        in release_qualification_test
        and "test_nul_truncated_member_name_cannot_masquerade_as_allowlisted"
        in release_qualification_test
        and "test_member_extra_fields_and_comments_cannot_hide_metadata"
        in release_qualification_test
        and "test_local_only_extra_field_cannot_hide_metadata"
        in release_qualification_test
        and "test_unauthorized_artifact_member_rejected" in release_qualification_test
        and "test_symlink_archive_member_rejected" in release_qualification_test
        and "test_short_candidate_member_inventory_rejected"
        in release_qualification_test
        and "test_eocd_count_is_bounded_before_zipfile_parses_members"
        in release_qualification_test
        and "test_archive_preamble_cannot_hide_outside_the_member_inventory"
        in release_qualification_test
        and "test_failed_extraction_leaves_no_partial_trusted_output"
        in release_qualification_test
        and "test_interrupted_placement_leaves_no_partial_trusted_output"
        in release_qualification_test
        and "test_destination_swap_cannot_redirect_or_preserve_verified_output"
        in release_qualification_test,
        "artifact archives must be count- and byte-bounded, allow-listed, and local/central/descriptor consistent before extraction, and a rejected archive must leave no partial trusted output",
        errors,
    )
    require(
        # The transcript the heavy gate proves and the transcript qualification
        # accepts must be one pinned value, not two copies that can drift.
        "from speech_to_text_browser_smoke import DEFAULT_EXPECTED_TEXT"
        in release_qualification
        and "normalize_transcript(DEFAULT_EXPECTED_TEXT)" in release_qualification
        and "Hmm. Oh, yeah, yeah." not in release_qualification
        and "does not match expected transcript" in release_qualification
        and "rejected but reported" in release_qualification
        and "test_mutated_asr_cold_transcript_rejected" in release_qualification_test
        and "test_mutated_asr_warm_transcript_rejected" in release_qualification_test
        and "test_malformed_asr_cancellation_schema_rejected"
        in release_qualification_test
        and "test_rejected_cancellation_must_report_no_output"
        in release_qualification_test
        and "test_asr_silence_transcript_must_stay_empty" in release_qualification_test
        and "test_tts_truncated_must_be_false" in release_qualification_test
        and "test_tts_lifecycle_evidence_cancellation_tested_must_be_true"
        in release_qualification_test
        and "test_tts_lifecycle_evidence_pre_aborted_tested_must_be_true"
        in release_qualification_test
        and "test_tts_lifecycle_evidence_reuse_sample_count_must_be_positive"
        in release_qualification_test
        and "test_tts_lifecycle_evidence_unload_tested_must_be_true"
        in release_qualification_test,
        "attestation evidence must match the single pinned transcript and prove exact cancellation, silence, and TTS lifecycle semantics",
        errors,
    )
    require(
        # Publication consumes the candidate; rebuilding it would change the
        # manifest and therefore the digest no attestation could then match.
        "scripts/build_bridge.sh" not in publish
        and "WEBGPU_BRIDGE_BUILD_MEM64" not in publish
        and "setup-emsdk" not in publish
        and "Run durable state persistence gate" not in publish
        and "Run durable multimodal gate" not in publish
        and "scripts/generate_release_manifest.py" not in publish
        and "Prove the candidate and qualification run identities" in publish
        and "scripts/release_qualification.py verify-run" in publish
        and publish.count("scripts/release_qualification.py verify-attestation") == 2
        and "--workflow-path \"${workflow_path}\"" in publish
        and "--head-branch \"${bridge_default}\"" in publish
        # A run head on the default-branch line, not merely on some branch, and
        # not pinned to a moving head that an unrelated push would invalidate.
        and 'repos/${BRIDGE_REPO}/compare/${head_sha}...${bridge_default}' in publish
        and 'repos/${BRIDGE_REPO}/compare/${head_sha}...${bridge_default}'
        in bridge_qualification
        and "CANDIDATE_WORKFLOW_PATH: .github/workflows/bridge_candidate.yml" in publish
        and "QUALIFICATION_WORKFLOW_PATH: .github/workflows/bridge_qualification.yml"
        in publish
        and 'actions/artifacts/${CANDIDATE_ARTIFACT_ID}/zip' in publish
        and 'actions/artifacts/${QUALIFICATION_ARTIFACT_ID}/zip' in publish
        and "?per_page=100" in publish
        and "run_attempt_args=(--run-attempt 1)" in publish
        and "_extract_flat_artifact_archive" in publish
        and "Checkout trusted candidate verification policy" in publish
        and "publication-policy/scripts/release_qualification.py verify-attestation"
        in publish
        and "publication-policy/scripts/release_publication_state.py classify"
        in publish
        and "bridge-source/scripts/release_qualification.py" not in publish
        and "bridge-source/scripts/release_publication_state.py" not in publish
        and "--harness-dir" in publish
        and "exact-webgpu-bridge-dist" in publish
        and "verified-qualification-attestation" in publish
        and "bridge-qualification-outcome" in publish,
        "publication must consume the exact candidate artifact, prove both run identities, and verify the digest-bound attestation instead of rebuilding",
        errors,
    )
    require(
        "Validate release qualification contract" in ci
        and "scripts/release_qualification_test.py" in ci
        and "scripts/release_qualification.py" in ci
        and "scripts/release_qualification_test.py" in publish
        and "scripts/release_qualification_test.py" in candidate
        and "build_attestation" in release_qualification
        and "validate_workflow_run" in release_qualification
        and "validate_artifact_inventory" in release_qualification
        and "verify_attestation" in release_qualification
        and "harness_source_sha256" in release_qualification
        and "require_harness_matches_bridge_source" in release_qualification
        and "REQUIRED_UNPROVEN_CAPABILITIES" in release_qualification
        and "qualification_environment" in release_qualification
        and "hosted-github-actions" in release_qualification
        and 'os.environ.get("GITHUB_ACTIONS") != "true"'
        in release_qualification
        and 'os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"'
        in release_qualification
        and '"release_contract.py"' in release_qualification
        and '"release_publication_state.py"' in release_qualification
        and all(
            lane in release_contract
            for lane in (
                "hardware_gpu_acceleration",
                "real_device_intelligibility",
                "real_device_playback",
                "speaker_reference_fidelity",
                "wasm32_text_to_speech",
            )
        )
        # The heavy gates must prove every required memory/runtime mode and
        # must never silently reach the network for a model or fixture.
        and '("wasm32", "direct")' in release_qualification
        and '("wasm32", "worker")' in release_qualification
        and '("wasm64", "direct")' in release_qualification
        and '("wasm64", "worker")' in release_qualification
        and "LLAMA_WEBGPU_" in release_qualification
        and "read_wav_identity" in release_qualification
        and "phaseTimingsMs" in speech_smoke
        and "coldTranscriptMs" in speech_smoke
        and "warmTranscriptMs" in speech_smoke
        and "cancellationMs" in speech_smoke
        and "silenceMs" in speech_smoke
        # Replay, unrelated-run, malformed, and duplicate payload cases are
        # regression-tested rather than merely asserted in prose.
        and "test_stale_attestation_replayed_against_new_release_rejected"
        in release_qualification_test
        and "test_publication_refuses_a_rebuilt_candidate_from_another_run"
        in release_qualification_test
        and "test_unrelated_or_unsuccessful_run_rejected" in release_qualification_test
        and "test_candidate_run_must_be_the_first_build_attempt"
        in release_qualification_test
        and "test_run_attempt_one_is_mandatory_even_without_a_caller_override"
        in release_qualification_test
        and "test_missing_duplicate_expired_or_foreign_artifact_rejected"
        in release_qualification_test
        and "test_truncated_artifact_inventory_rejected"
        in release_qualification_test
        and "test_artifact_archive_rejects_path_escape_and_duplicate_members"
        in release_qualification_test
        and "test_tts_phase_rejects_wav_path_escape_and_symlink"
        in release_qualification_test
        and "test_tts_waveform_evidence_must_be_finite_and_non_silent"
        in release_qualification_test
        and "test_speech_result_evidence_is_required"
        in release_qualification_test
        and "test_smoke_timeout_is_bounded_and_writes_sanitized_diagnostics"
        in release_qualification_test
        and "test_local_harness_must_match_the_exact_bridge_source"
        in release_qualification_test
        and "test_noncanonical_attestation_artifact_rejected"
        in release_qualification_test
        and "test_duplicate_keys_in_attestation_artifact_rejected"
        in release_qualification_test
        # Automatic publication must be unable to accept an attestation that was
        # not produced on hosted infrastructure.
        and "test_non_hosted_execution_claim_rejected" in release_qualification_test
        and "test_environment_probe_requires_github_hosted_runner_markers"
        in release_qualification_test
        and "test_harness_digest_covers_every_heavy_gate_source"
        in release_qualification_test
        and "test_nonstandard_json_constants_rejected_before_schema_validation"
        in release_qualification_test
        and "test_verify_run_cli_defaults_to_the_mandatory_first_attempt"
        in release_qualification_test
        and "test_candidate_manifest_must_not_claim_a_hosted_heavy_gate_pass"
        in release_qualification_test
        # The routine path takes no hand-produced attestation: the workflow runs
        # the gates itself and never accepts a transported payload.
        and "attestation_base64:" not in bridge_qualification
        and "attestation_json:" not in bridge_qualification
        and "decode-attestation" not in bridge_qualification
        and "Validate dispatcher and repository boundary" in bridge_qualification
        and '"${DISPATCHING_ACTOR}" != "${REPOSITORY_OWNER}"'
        in bridge_qualification
        and "Prove the candidate run identity and artifact uniqueness"
        in bridge_qualification
        and "scripts/release_qualification.py verify-run" in bridge_qualification
        and "release_qualification.py qualify" in bridge_qualification
        and "scripts/release_qualification.py verify-attestation"
        in bridge_qualification
        and "Checkout exact candidate harness source" in bridge_qualification
        and "--harness-dir candidate-source/scripts" in bridge_qualification
        and 'candidate_correlation_id="$(jq -er' in bridge_qualification
        and '"${candidate_correlation_id}" != "${ORCHESTRATOR_CORRELATION_ID}"'
        in bridge_qualification
        and "?per_page=100" in bridge_qualification
        and "--run-attempt 1" in bridge_qualification
        and 'actions/artifacts/${CANDIDATE_ARTIFACT_ID}/zip'
        in bridge_qualification
        and "Upload verified qualification attestation" in bridge_qualification
        and "WEBGPU_BRIDGE_ASSETS_PAT" not in bridge_qualification,
        "CI, candidate, qualification, and publication must enforce the digest-bound qualification contract with regression coverage for replay and malformed payloads",
        errors,
    )
    require(
        "scripts/generate_release_manifest.py" in candidate
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
        # The manifest must state the automated-qualification requirement rather
        # than claim a heavy-gate pass the candidate build never executed.
        and '"speech_to_text": AUTOMATED_QUALIFICATION_REQUIRED' in read_required(
            "scripts/generate_release_manifest.py", errors
        )
        and '"text_to_speech": AUTOMATED_QUALIFICATION_REQUIRED' in read_required(
            "scripts/generate_release_manifest.py", errors
        )
        and '"unproven_capabilities": UNPROVEN_CAPABILITIES' in read_required(
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
        "scripts/speech_to_text_browser_smoke.py" in ci
        and "LLAMA_WEBGPU_SPEECH_MODEL_SHA256" in ci
        and "LLAMA_WEBGPU_SPEECH_MMPROJ_SHA256" in ci
        and "run_speech_to_text_smoke" not in ci
        and "Run Qwen3-ASR browser smoke" not in ci,
        "CI must syntax-check the checksum-pinned Qwen3-ASR harness without executing the heavy real-model gate on hosted runners",
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
        "solo-maintainer publication contract does not require a reviewer rule" in agents_publication
        and "custom deployment branches to `main`" in agents_publication
        and "`WEBGPU_BRIDGE_ASSETS_PAT` as an environment-scoped secret" in agents_publication
        and "Use the default job token to validate the environment identity" in agents_publication
        and "fail closed unless" in agents_publication
        and "without printing its value" in agents_publication
        and "pinned repository-owner maintainer reviewer" not in agents_publication
        and "required_reviewers" not in agents_publication
        and "prevent_self_review" not in agents_publication
        and "bypass and self-review" not in agents_publication
        and "Require the reviewer inventory" not in agents_publication
        and "Revalidate the bypass, reviewer" not in agents_publication,
        "AGENTS.md publish guidance must preserve the solo-maintainer PAT/main-only contract without pinned-reviewer, self-review, quorum, or reviewer-revalidation requirements",
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
        ".github/workflows/bridge_candidate.yml": candidate,
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
