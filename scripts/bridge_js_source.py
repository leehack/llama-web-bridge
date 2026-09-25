"""Bridge JS source text for the static API contract checks.

js/src/llama_webgpu_bridge.js is only the public entry; the implementation
lives in the .js and .ts modules it imports (the public .d.ts is excluded). The
checks read every module, joined in the order the former single-file source
declared them (helpers, worker host and proxy, direct runtime, facade), so
multi-token patterns keep their scope. A missing listed module raises instead
of silently narrowing a check.
"""
from __future__ import annotations

from pathlib import Path

_ORDERED_MODULES = (
    "worker_protocol.ts",
    "worker_host.js",
    "worker_proxy.js",
    "runtime.js",
    "bridge.js",
    "llama_webgpu_bridge.js",
)


def _modules(directory: Path, pattern: str) -> list[Path]:
    return [
        path
        for suffix in (".js", ".ts")
        for path in directory.glob(pattern + suffix)
        if not path.name.endswith(".d.ts")
    ]


def bridge_js_source(root: Path) -> str:
    source_dir = root / "js" / "src"
    internal = sorted(_modules(source_dir / "internal", "*"))
    ordered = [source_dir / name for name in _ORDERED_MODULES]
    listed = {*internal, *ordered}
    remaining = sorted(
        path for path in _modules(source_dir, "**/*") if path not in listed
    )
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in (*internal, *ordered, *remaining)
    )


def method_body(source: str, signature: str) -> str:
    """Returns a class method from its signature through its closing brace.

    Scoping an ordered pattern to one method keeps it from matching text in a
    later method. A missing signature returns "", so the check fails closed.
    """
    start = source.find(f"\n  {signature}")
    if start < 0:
        return ""
    end = source.find("\n  }\n", start)
    return source[start : end + len("\n  }") if end > 0 else len(source)]
