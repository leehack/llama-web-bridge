"""Release orchestrator source text for the static contract checks.

scripts/stable_release_orchestrator.py is only the CLI entry; the state machine
lives in the scripts/release_orchestrator_<concern>.py modules it imports. Its
contract suites are scripts/stable_release_orchestrator_test.py plus every
scripts/release_orchestrator_<concern>_test.py, which share
scripts/release_orchestrator_fixtures_test.py. The checks read each set joined,
entry first and then by file name, so a text check keeps passing when code
moves between modules and fails when it leaves the set.

A release_orchestrator_* module that the entry does not reach through its
imports, an import of a missing one, an import of any other scripts/ module
outside SHARED_RELEASE_MODULES, or an orchestrator suite named outside the
suite pattern raises instead of silently widening or narrowing a check.
"""
from __future__ import annotations

import ast
from pathlib import Path

ENTRY = "scripts/stable_release_orchestrator.py"
ENTRY_TEST = "scripts/stable_release_orchestrator_test.py"
TEST_FIXTURES = "scripts/release_orchestrator_fixtures_test.py"
_ENTRY_MODULE = "stable_release_orchestrator"
_MODULE_PREFIX = "release_orchestrator_"
# The shared release modules the orchestrator imported before it was split.
# They have their own contracts, so their text is not part of the orchestrator
# source; any other scripts/ import would hide code from the checks and raises.
SHARED_RELEASE_MODULES = frozenset(
    {
        "generate_release_manifest",
        "release_contract",
        "release_publication_state",
        "release_qualification",
    }
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                raise ValueError(f"{path.name} uses a relative import")
            names.add(node.module.split(".")[0])
    return names


def _is_scripts_module(scripts: Path, name: str) -> bool:
    return (scripts / f"{name}.py").exists() or (scripts / name).is_dir()


def orchestrator_modules(root: Path) -> list[Path]:
    """Returns the entry, then every module it reaches through imports, by name."""
    scripts = root / "scripts"
    present = {
        path.stem
        for path in scripts.glob(f"{_MODULE_PREFIX}*.py")
        if not path.stem.endswith("_test")
    }
    reached: set[str] = set()
    pending = [root / ENTRY]
    while pending:
        path = pending.pop()
        imported = _imported_modules(path)
        unexpected = sorted(
            name
            for name in imported
            if not name.startswith(_MODULE_PREFIX)
            and name not in SHARED_RELEASE_MODULES
            and _is_scripts_module(scripts, name)
        )
        if unexpected:
            raise ValueError(
                f"{path.name} imports scripts/ modules that are neither "
                f"{_MODULE_PREFIX}* nor shared release modules: {unexpected}"
            )
        for name in sorted(
            name for name in imported if name.startswith(_MODULE_PREFIX)
        ):
            if name not in reached:
                reached.add(name)
                pending.append(scripts / f"{name}.py")
    if reached != present:
        raise ValueError(
            "scripts/stable_release_orchestrator.py must reach exactly the "
            f"scripts/{_MODULE_PREFIX}*.py modules through its imports "
            f"(unreached: {sorted(present - reached)}, "
            f"missing: {sorted(reached - present)})"
        )
    return [root / ENTRY, *(scripts / f"{name}.py" for name in sorted(reached))]


def orchestrator_test_suites(root: Path) -> list[Path]:
    """Returns every runnable orchestrator suite; the shared fixtures run no tests.

    A scripts/*_test.py that imports an orchestrator module but falls outside
    the suite names raises, so a rename cannot drop a suite from the checks.
    """
    scripts = root / "scripts"
    suites = sorted(
        path
        for path in scripts.glob(f"{_MODULE_PREFIX}*_test.py")
        if path != root / TEST_FIXTURES
    )
    covered = {root / ENTRY_TEST, root / TEST_FIXTURES, *suites}
    stray = sorted(
        path.name
        for path in scripts.glob("*_test.py")
        if path not in covered
        and any(
            name == _ENTRY_MODULE or name.startswith(_MODULE_PREFIX)
            for name in _imported_modules(path)
        )
    )
    if stray:
        raise ValueError(
            "suites that import the release orchestrator must be named "
            f"stable_release_orchestrator_test.py or {_MODULE_PREFIX}*_test.py: "
            f"{stray}"
        )
    return [root / ENTRY_TEST, *suites]


def orchestrator_source(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in orchestrator_modules(root)
    )


def orchestrator_test_source(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in (*orchestrator_test_suites(root), root / TEST_FIXTURES)
    )
