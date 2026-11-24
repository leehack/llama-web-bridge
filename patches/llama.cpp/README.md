# Carried llama.cpp patches

Prefer a llama.cpp release that contains the fix. Carry a small patch when a
bridge build has a reproduced runtime problem that cannot be repaired in the
bridge's own source or build settings.

Each `series.json` entry names a patch, the CMake `target` that compiles its
source, the bridge `tracking` issue, the `upstream` issue or PR, and a
`wasm_marker` compiled by the patch. Replace the clearly marked pending issue
URLs after their owners file them. The release candidate workflow runs
`verify-release` before compiling any core to reject either placeholder.
A patch is a unified diff of one existing
`.c` or `.cpp` translation unit. Edit a scratch copy to produce the diff and
leave the upstream checkout intact.

CMake checks the series in filename order. Every hunk needs one exact,
whole-line match, and each patch applies as a unit. The target builds the
resulting source from `<build>/llama.cpp-patched/`. A default build fails on
a mismatch. When explicitly qualifying another upstream in a separate build
directory, pass `-DLLAMADART_LLAMA_CPP_PATCH_QUALIFICATION=ON` to
`emcmake cmake` to warn and skip a non-applicable patch. Keep strict mode for
CI and release candidate builds.

A build checks the Wasm binary for every applied patch's marker. CI checks
each core again before uploading or saving it, including on a core cache
hit. To check all local outputs against the pinned source:

```sh
node scripts/build/llama_cpp_patches.mjs verify "$LLAMA_CPP_DIR" "$OUT_DIR"/llama_webgpu_core*.wasm
```

When a tested upstream repair lands, optionally set `upstream_fix` to a
distinctive source fragment from that repair. Builds skip the carry and its
marker only if the untouched source for that entry contains the fragment.
Use the original affected model and backend to verify that the new upstream
works without the carry before removing it. The release candidate already
records the bridge source commit containing the series and the exact
upstream commit it built.

## WebGPU weight uploads

`0001-webgpu-weight-upload-batching.patch` splits browser WebGPU weight writes
into at most 32 MiB each and waits for queue completion after every 64 MiB of
aligned weight writes on that queue. It uses the backend's existing timed
`WaitAny` path, which Emdawnwebgpu fulfills asynchronously through
`GPUQueue.onSubmittedWorkDone()`. Bridge model loads enter Wasm through
async `ccall`, and the bridge serializes operations on the core while the
load yields.
