#!/usr/bin/env python3
"""Patch the Emscripten wasm64 WASMFS JavaScript boundary."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def bigint_or_name(name: str) -> str:
    return rf"(?:BigInt\(\s*{name}\s*\)|{name})"


DATA_BUFFER = bigint_or_name("dataBuffer")
LENGTH = bigint_or_name("length")
POSITION = bigint_or_name("position")
OFFSET = bigint_or_name("offset")
PATCHES = (
    (
        "__wasmfs_read",
        rf"__wasmfs_read\(\s*stream\.fd\s*,\s*{DATA_BUFFER}\s*,\s*{LENGTH}\s*\)",
        "__wasmfs_read(stream.fd,BigInt(dataBuffer),BigInt(length))",
    ),
    (
        "__wasmfs_pread",
        rf"__wasmfs_pread\(\s*stream\.fd\s*,\s*{DATA_BUFFER}\s*,\s*{LENGTH}\s*,\s*{POSITION}\s*\)",
        "__wasmfs_pread(stream.fd,BigInt(dataBuffer),BigInt(length),BigInt(position))",
    ),
    (
        "__wasmfs_write",
        rf"__wasmfs_write\(\s*stream\.fd\s*,\s*{DATA_BUFFER}\s*,\s*{LENGTH}\s*\)",
        "__wasmfs_write(stream.fd,BigInt(dataBuffer),BigInt(length))",
    ),
    (
        "__wasmfs_pwrite",
        rf"__wasmfs_pwrite\(\s*stream\.fd\s*,\s*{DATA_BUFFER}\s*,\s*{LENGTH}\s*,\s*{POSITION}\s*\)",
        "__wasmfs_pwrite(stream.fd,BigInt(dataBuffer),BigInt(length),BigInt(position))",
    ),
    (
        "__wasmfs_mmap",
        rf"__wasmfs_mmap\(\s*{LENGTH}\s*,\s*prot\s*,\s*flags\s*,\s*stream\.fd\s*,\s*{OFFSET}\s*\)",
        "__wasmfs_mmap(BigInt(length),prot,flags,stream.fd,BigInt(offset))",
    ),
)


class PatchError(RuntimeError):
    """Raised when required generated-JS symbols cannot be patched."""


def patch_wasm64_runtime(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for name, pattern, replacement in PATCHES:
        text, count = re.subn(pattern, replacement, text)
        counts[name] = count

    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise PatchError(
            "wasm64 runtime patch did not match required generated-JS symbols: "
            f"{', '.join(missing)}; inspect the pinned Emscripten output before "
            "changing the expected symbol set"
        )
    return text, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path, help="generated wasm64 JavaScript file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        text = args.target.read_text(encoding="utf-8", errors="ignore")
        patched, counts = patch_wasm64_runtime(text)
        args.target.write_text(patched, encoding="utf-8")
    except (OSError, PatchError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = ", ".join(f"{name}={count}" for name, count in counts.items())
    print(f"Patched wasm64 generated JavaScript: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
