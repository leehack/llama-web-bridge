"""C++ core source text for the static API contract checks.

src/llama_webgpu_core.cpp includes its parts from src/core/ inside its
anonymous namespace and extern "C" block. The checks read it with each
`#include "core/<part>.inc"` line replaced by that part, which is the order
the compiler sees. tests/js/native_core_source.mjs is the JS twin.

A part that is missing, included in another form, or not included exactly once
raises instead of silently narrowing a check.
"""
from __future__ import annotations

import re
from pathlib import Path

_PART_INCLUDE = re.compile(r'^#include "(core/[A-Za-z0-9_]+\.inc)"$', re.MULTILINE)
_ANY_PART_INCLUDE = re.compile(r'#\s*include\s*["<]core/')


def native_core_source(root: Path) -> str:
    source_dir = root / "src"
    core = (source_dir / "llama_webgpu_core.cpp").read_text(encoding="utf-8")
    expanded: list[str] = []

    def expand(match: re.Match[str]) -> str:
        expanded.append(match.group(1))
        return (source_dir / match.group(1)).read_text(encoding="utf-8").rstrip("\n")

    source = _PART_INCLUDE.sub(expand, core)
    parts = sorted(f"core/{path.name}" for path in (source_dir / "core").glob("*.inc"))
    if _ANY_PART_INCLUDE.search(source) or sorted(expanded) != parts:
        raise ValueError(
            "src/llama_webgpu_core.cpp must include every src/core/*.inc part "
            'exactly once, each on its own `#include "core/<part>.inc"` line'
        )
    return source
