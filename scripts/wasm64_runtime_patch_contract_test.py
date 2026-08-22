#!/usr/bin/env python3
"""Regression contracts for the generated wasm64 JavaScript patch."""

from __future__ import annotations

import unittest

from patch_wasm64_runtime import PATCHES, PatchError, patch_wasm64_runtime

# Representative fragments from the Emscripten 6.0.8 output shape validated on
# main. Both raw numeric values and already-wrapped values are included because
# the patch is intentionally idempotent.
CURRENT_EMSCRIPTEN_OUTPUT = """
bytesRead=__wasmfs_pread(stream.fd,dataBuffer,length,position);
bytesRead=__wasmfs_read(stream.fd,BigInt(dataBuffer),BigInt(length));
bytesRead=__wasmfs_pwrite(stream.fd,dataBuffer,length,position);
bytesRead=__wasmfs_write(stream.fd,BigInt(dataBuffer),BigInt(length));
allocated=__wasmfs_mmap(length,prot,flags,stream.fd,offset);
"""


class Wasm64RuntimePatchContractTest(unittest.TestCase):
    def test_current_emscripten_output_matches_all_required_symbols(self) -> None:
        patched, counts = patch_wasm64_runtime(CURRENT_EMSCRIPTEN_OUTPUT)

        self.assertEqual({name: 1 for name, _, _ in PATCHES}, counts)
        for _, _, replacement in PATCHES:
            self.assertIn(replacement, patched)

    def test_each_single_match_still_fails_the_five_symbol_contract(self) -> None:
        lines = [line for line in CURRENT_EMSCRIPTEN_OUTPUT.splitlines() if line]
        self.assertEqual(5, len(lines))

        for only_name, _, _ in PATCHES:
            with self.subTest(only_match=only_name):
                only_line = next(line for line in lines if only_name in line)
                with self.assertRaises(PatchError) as caught:
                    patch_wasm64_runtime(only_line)

                message = str(caught.exception)
                for required_name, _, _ in PATCHES:
                    if required_name != only_name:
                        self.assertIn(required_name, message)

    def test_each_missing_symbol_is_named_independently(self) -> None:
        for missing_name, _, _ in PATCHES:
            with self.subTest(missing=missing_name):
                partial = "\n".join(
                    line
                    for line in CURRENT_EMSCRIPTEN_OUTPUT.splitlines()
                    if missing_name not in line
                )
                with self.assertRaisesRegex(PatchError, missing_name):
                    patch_wasm64_runtime(partial)


if __name__ == "__main__":
    unittest.main()
