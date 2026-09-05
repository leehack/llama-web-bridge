#!/usr/bin/env python3
"""Compile both media-helper API shapes and guard all production call sites."""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MediaCompatibilityTest(unittest.TestCase):
    def test_both_upstream_api_shapes(self):
        for options in (False, True):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                extra = ", mtmd_helper_init_opt options" if options else ""
                check = "assert(options.sentinel == 73);" if options else ""
                declarations = """
struct mtmd_helper_init_opt { int sentinel; };
inline mtmd_helper_init_opt mtmd_helper_init_opt_default() { return {73}; }
""" if options else ""
                (directory / "mtmd-helper.h").write_text(
                    """#pragma once
#include <cassert>
#include <cstddef>
struct mtmd_context {};
struct mtmd_helper_bitmap_wrapper { int result; };
inline mtmd_context expected_context;
inline const unsigned char expected_bytes[] = {1, 2, 3};
inline const char expected_path[] = "reference.wav";
""" + declarations + f"""
inline mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(
    mtmd_context *ctx, const unsigned char *bytes, size_t size, bool placeholder{extra}) {{
  assert(ctx == &expected_context && bytes == expected_bytes && size == 3);
  assert(!placeholder); {check}
  return {{17}};
}}
inline mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_file(
    mtmd_context *ctx, const char *path, bool placeholder{extra}) {{
  assert(ctx == &expected_context && path == expected_path);
  assert(!placeholder); {check}
  return {{29}};
}}
""", encoding="utf-8")
                source = directory / "test.cpp"
                source.write_text("""
#include "llama_webgpu_mtmd_compat.h"
int main() {
  assert(llama_webgpu_bitmap_from_buffer(&expected_context, expected_bytes, 3).result == 17);
  assert(llama_webgpu_bitmap_from_file(&expected_context, expected_path).result == 29);
}
""", encoding="utf-8")
                executable = directory / "test"
                subprocess.run([
                    *shlex.split(os.environ.get("CXX", "c++")), "-std=c++17",
                    "-Wall", "-Wextra", "-Werror",
                    f"-DLLAMADART_MTMD_HELPER_HAS_OPTIONS={int(options)}",
                    f"-I{directory}", f"-I{ROOT / 'src'}", str(source),
                    "-o", str(executable),
                ], check=True, capture_output=True, text=True)
                subprocess.run([str(executable)], check=True)

    def test_production_routes_through_compatibility_helpers(self):
        core = (ROOT / "src/llama_webgpu_core.cpp").read_text()
        tts = (ROOT / "src/llama_webgpu_tts.cpp").read_text()
        self.assertEqual(core.count("llama_webgpu_bitmap_from_file("), 1)
        self.assertEqual(core.count("llama_webgpu_bitmap_from_buffer("), 1)
        self.assertEqual(tts.count("llama_webgpu_bitmap_from_buffer("), 1)
        for source in (core, tts):
            self.assertNotIn("mtmd_helper_bitmap_init_from_", source)


if __name__ == "__main__":
    unittest.main()
