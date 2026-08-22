#!/usr/bin/env python3
"""Resolve and verify the checked-in Emscripten SDK version."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = ROOT / "emsdk.version"
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
EMCC_VERSION_RE = re.compile(r"^emcc \(.*\) ([0-9]+\.[0-9]+\.[0-9]+)\b")


def read_pin() -> str:
    version = PIN_PATH.read_text(encoding="utf-8").strip()
    if not VERSION_RE.fullmatch(version):
        raise ValueError(
            f"{PIN_PATH.name} must contain one semantic Emscripten version, got: {version!r}"
        )
    return version


def resolve_emcc_version() -> str:
    result = subprocess.run(
        ["emcc", "--version"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(f"emcc --version failed: {detail}")

    lines = result.stdout.splitlines()
    first_line = lines[0] if lines else ""
    match = EMCC_VERSION_RE.match(first_line)
    if match is None:
        raise RuntimeError(f"could not parse Emscripten version from: {first_line!r}")
    return match.group(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-pin",
        action="store_true",
        help="print the checked-in version without invoking emcc",
    )
    parser.add_argument(
        "--emit-github-env",
        type=Path,
        metavar="PATH",
        help="append the verified resolved version to a GitHub Actions env file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        expected = read_pin()
        if args.print_pin:
            print(expected)
            return 0

        resolved = resolve_emcc_version()
        if resolved != expected:
            raise RuntimeError(
                f"resolved Emscripten {resolved} does not match {PIN_PATH.name} {expected}"
            )

        if args.emit_github_env is not None:
            with args.emit_github_env.open("a", encoding="utf-8") as env_file:
                env_file.write(f"EMSCRIPTEN_VERSION={resolved}\n")

        print(f"Validated Emscripten {resolved} against {PIN_PATH.name}")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
