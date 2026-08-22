#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from generate_release_manifest import ARTIFACTS, generate


class GenerateReleaseManifestTest(unittest.TestCase):
    def test_generates_schema_v2_with_legacy_aliases_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            for index, name in enumerate(ARTIFACTS):
                (out_dir / name).write_bytes(f"artifact-{index}".encode())
            args = argparse.Namespace(
                out_dir=out_dir,
                release_tag="v0.2.0-1",
                release_rebuild=1,
                assets_repo="leehack/llama-web-bridge-assets",
                bridge_repo="leehack/llama-web-bridge",
                bridge_commit="a" * 40,
                upstream_repo="ggml-org/llama.cpp",
                upstream_tag="v0.2.0",
                upstream_commit="b" * 40,
                native_repo="leehack/llamadart-native",
                native_release_tag="v0.2.0-1",
                native_manifest_sha256="c" * 64,
                native_commit="d" * 40,
                emscripten_version="6.0.8",
            )
            manifest = generate(args)
            first_bytes = (out_dir / "manifest.json").read_bytes()
            generate(args)
            self.assertEqual((out_dir / "manifest.json").read_bytes(), first_bytes)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["release_tag"], "v0.2.0-1")
            self.assertEqual(manifest["bridge_commit"], "a" * 40)
            self.assertEqual(manifest["upstream_commit"], "b" * 40)
            self.assertEqual(manifest["native_commit"], "d" * 40)
            self.assertEqual(manifest["bridge_assets_tag"], "v0.2.0-1")
            self.assertTrue(manifest["capabilities"]["speech_to_text"]["advertised"])
            self.assertTrue(manifest["capabilities"]["text_to_speech"]["advertised"])
            self.assertNotIn("generated_at_utc", manifest)

            payload = json.loads((out_dir / "manifest.json").read_text())
            artifact = ARTIFACTS[0]
            expected = hashlib.sha256((out_dir / artifact).read_bytes()).hexdigest()
            self.assertEqual(payload["artifacts"][artifact]["sha256"], expected)
            sums = (out_dir / "sha256sums.txt").read_text()
            self.assertIn(f"{expected}  {artifact}", sums)
            self.assertNotIn("manifest.json", sums)


if __name__ == "__main__":
    unittest.main()
