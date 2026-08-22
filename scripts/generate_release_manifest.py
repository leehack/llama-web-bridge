#!/usr/bin/env python3
"""Generate the checksummed bridge asset manifest from validated inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from release_contract import (
    ContractError,
    require_repository,
    require_sha256,
    validate_release_identity,
)


ARTIFACTS = (
    "llama_webgpu_bridge.js",
    "llama_webgpu_bridge_worker.js",
    "llama_webgpu_bridge.d.ts",
    "llama_webgpu_core.js",
    "llama_webgpu_core.wasm",
    "llama_webgpu_core_mem64.js",
    "llama_webgpu_core_mem64.wasm",
)


def _commit(value: str, field: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"{field} must be a lowercase full 40-character commit SHA")
    return value


def generate(args: argparse.Namespace) -> dict[str, object]:
    release = validate_release_identity(
        args.release_tag, args.release_rebuild, args.upstream_tag
    )
    require_repository(args.assets_repo, "assets_repo")
    require_repository(args.bridge_repo, "bridge_repo")
    require_repository(args.upstream_repo, "upstream_repo")
    require_repository(args.native_repo, "native_repo")
    require_sha256(args.native_manifest_sha256, "native_manifest_sha256")
    bridge_commit = _commit(args.bridge_commit, "bridge_commit")
    upstream_commit = _commit(args.upstream_commit, "upstream_commit")
    native_commit = _commit(args.native_commit, "native_commit")

    files: dict[str, dict[str, int | str]] = {}
    checksums: list[str] = []
    for name in ARTIFACTS:
        path = args.out_dir / name
        if not path.is_file():
            raise ContractError(f"required artifact is missing: {name}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        files[name] = {"size_bytes": len(data), "sha256": digest}
        checksums.append(f"{digest}  {name}")

    capabilities: dict[str, object] = {
        "wasm32": True,
        "memory64": True,
        "state_persistence": {"direct": True, "worker": True},
        "multimodal": {"direct": True, "worker": True},
        "speech_to_text": {
            "advertised": args.enable_asr,
            "direct": args.enable_asr,
            "worker": args.enable_asr,
            "wasm32": args.enable_asr,
            "memory64": args.enable_asr,
        },
        "text_to_speech": {
            "advertised": args.enable_tts,
            "direct": args.enable_tts,
            "worker": args.enable_tts,
            "wasm32": False,
            "memory64": args.enable_tts,
        },
    }

    manifest: dict[str, object] = {
        "schema_version": 2,
        "release_tag": release.tag,
        "release_channel": release.channel.value,
        "release_rebuild": release.rebuild,
        "assets_repository": args.assets_repo,
        "bridge_repository": args.bridge_repo,
        "bridge_commit": bridge_commit,
        "upstream_repository": args.upstream_repo,
        "upstream_tag": args.upstream_tag,
        "upstream_commit": upstream_commit,
        "native_repository": args.native_repo,
        "native_release_tag": args.native_release_tag,
        "native_manifest_sha256": args.native_manifest_sha256,
        "native_commit": native_commit,
        "emscripten_version": args.emscripten_version,
        "capabilities": capabilities,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts": files,
        # Compatibility aliases are read by existing consumers. New tooling must
        # use the explicit schema-v2 names above.
        "bridge_assets_tag": release.tag,
        "source_repository": args.bridge_repo,
        "source_commit": bridge_commit,
        "llama_cpp_tag": args.upstream_tag,
        "llama_cpp_commit": upstream_commit,
        "files": files,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "sha256sums.txt").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--release-rebuild", required=True, type=int)
    parser.add_argument("--assets-repo", required=True)
    parser.add_argument("--bridge-repo", required=True)
    parser.add_argument("--bridge-commit", required=True)
    parser.add_argument("--upstream-repo", default="ggml-org/llama.cpp")
    parser.add_argument("--upstream-tag", required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--native-repo", required=True)
    parser.add_argument("--native-release-tag", required=True)
    parser.add_argument("--native-manifest-sha256", required=True)
    parser.add_argument("--native-commit", required=True)
    parser.add_argument("--emscripten-version", required=True)
    parser.add_argument("--enable-asr", action="store_true")
    parser.add_argument("--enable-tts", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        generate(args)
    except (ContractError, OSError) as error:
        raise SystemExit(f"error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
