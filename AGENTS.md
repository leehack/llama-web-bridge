# AGENTS.md

Guidance for coding agents working in `llama-web-bridge`.

- [README.md](README.md) covers the bridge, installation, and CDN assets.
- [API reference](docs/api.md) defines the public JavaScript API.
- [Single-thread guide](docs/single_thread.md) records browser staging limits and measurements.
- [CONTRIBUTING.md](CONTRIBUTING.md) owns build recipes, smoke commands and pins,
  workflow guardrails, and publication.

## Scope and Ownership

- This repository owns the WebGPU bridge source and runtime build for llama.cpp.
- `llama-web-bridge-assets` holds versioned artifacts published by workflow.
- `llamadart` consumes those artifacts.

### Related repositories

```text
../llamadart
../llamadart-native
../llama-web-bridge
../llama-web-bridge-assets
```

## Build Commands

```bash
./scripts/build/build_bridge.sh
```

Match the llama.cpp checkout to `llama_cpp.version` and Emscripten to
`emsdk.version`. See [Local Build](CONTRIBUTING.md#local-build).

## Agent PR Workflow

For non-trivial runtime, workflow, or API changes:

1. Start from a clean topic branch and inspect `git status` before editing.
2. Add or update a regression or contract check before changing behavior when
   practical. Static checks can cover workflow invariants.
3. Keep Emscripten build directories, ccache, model caches, and Playwright
   artifacts outside the checkout unless intentionally versioned.
4. Run the targeted checks below and the full browser smoke when changes touch
   `js/`, `src/`, `scripts/`, or GitHub workflows.
5. Get an independent review before committing PR-bound changes. Fix blockers,
   rerun the targeted checks, then commit locally. Push or open a PR only when
   the maintainer asks.

## Verification and Regression Guidance

The [external-path build recipe](CONTRIBUTING.md#local-build) sets `CCACHE_DIR`,
`EM_CACHE`, `BUILD_DIR`, `MEM64_BUILD_DIR`, `SINGLE_THREAD_BUILD_DIR`,
`SINGLE_THREAD_MEM64_BUILD_DIR`, and `OUT_DIR`, with
`WEBGPU_BRIDGE_BUILD_MEM64` and `WEBGPU_BRIDGE_BUILD_SINGLE_THREAD` enabled.
Run the lightweight contracts, starting with `npm run check:js`, and use the
[smoke commands](CONTRIBUTING.md#validate-outputs) with their pinned inputs.
The scripts below live in `scripts/smoke/` unless a different path is shown.

| Change | Smoke |
| --- | --- |
| Core disposal, retirement, or core pre/post JS | `core_release.mjs` on both memory modes and threading modes |
| Native abort or trap recovery | `core_abort_recovery.mjs` with default and single threading |
| State persistence or workflows | `state_persistence.mjs` on wasm32 and wasm64, threaded and single-threaded |
| Worker startup or core URLs | `cross_origin_assets.mjs` |
| Streaming downloads and retry integrity | `resume_integrity.mjs` |
| WasmFS fetch backend | `remote_fetch_backend.mjs` on both threaded memory modes |
| JS-backed staging mounts and shared cores | `tests/bridge/js_model_staging_shared_core_browser_test.mjs` |
| Single-thread behavior | Non-isolated `state_persistence.mjs` and local `single_thread.mjs` WebGPU acceptance |
| Asset resolution or npm packaging | `bundler_packaging.mjs` |
| llama.cpp pin or multimodal completion | `multimodal.mjs` through direct and worker runtimes |
| Multimodal embeddings | `multimodal_embeddings.mjs` with the pinned BF16 fixtures |
| Samplers or grammars | `grammar.mjs` with both the state-persistence and multimodal models |
| Next-token scoring or prompt evaluation | `next_token_scores.mjs` with the state-persistence model |
| Speculative decoding | `speculative.mjs` with the affected strategies |
| Speech recognition or synthesis | `speech_to_text.mjs` and `text_to_speech.mjs` individually |
| Decision heads | `decision.mjs` with a Laya encoder, head, and reference fixture |
| LoRA adapters | `lora_adapter.mjs` with its default base model and adapter |

- For pthread changes, test a BERT-class encoder in isolated Chromium on a host
  whose `navigator.hardwareConcurrency` exceeds the compiled pool. Run
  `loadModelFromUrl`, `tokenize`, `embed`, and `embedBatch` through both
  direct (`disableWorker: true`) and worker runtimes. Both must cap
  `n_threads` to the pool size.
- Worker and direct WASMFS are separate. Keep worker-owned state operations in
  the worker. Use the byte APIs for durable application storage.
- Keep downloaded GGUFs in a user cache or temporary directory and keep smoke
  artifacts outside the checkout. Redact userinfo, query values, and fragments
  from model URLs in errors and artifacts.
- Every llama.cpp pin change needs checksum-pinned real multimodal inference
  in both runtimes.
- Speech-capable releases require the hosted memory/runtime qualification
  matrix. Keep the heavy gates outside ordinary CI. The combined
  `qualification.mjs qualify` command requires a GitHub-hosted Actions runner.

## CI and Release

- `AGENTS.md` is outside the docs allowlist in `scripts/ci/ci_scope.mjs`.
  Editing it runs the full WASM build and browser smoke lane.
- See the [current pin qualification](docs/v100_qualification.md).
  Keep `src/llama_webgpu_mtmd_compat.h` and its static contract. CI no longer
  builds llama.cpp v0.4.0.
- Follow the [workflow guardrails](CONTRIBUTING.md#agent-workflow-guardrails)
  and the [four-stage publish process](CONTRIBUTING.md#publish-process).
  Keep SHA-256 pins in the files named by the verifier.
- Verify live settings before describing `bridge-assets-publication` or
  `bridge-npm-publication` as protected.

## Change Boundaries

- Keep runtime bridge source code in `js/src/` and `src/`. The
  generated bridge wrapper outputs and declaration (`js/llama_webgpu_bridge.js`,
  `js/llama_webgpu_bridge_worker.js`, `js/llama_webgpu_bridge.d.ts`) are
  regenerated by `npm run check:js`; CI, candidate, and publish fail on a stale
  copy with `git diff --exit-code`, so never hand-edit them.
- Bridge runtime and API contract tests, including the static API contracts,
  and their fixtures live in `tests/bridge/`; tooling tests live in
  `tests/build/`, `tests/ci/`, `tests/smoke/`, and `tests/release/`. Build, CI,
  smoke, and release tooling lives in `scripts/build/`, `scripts/ci/`,
  `scripts/smoke/`, and `scripts/release/`. The qualification harness digest
  covers `HARNESS_SOURCES` in `scripts/release/qualification.mjs` (harness
  `5.5.0`, the Node harness), and the candidate/publication checkouts address
  harness files by their `scripts/` path at the candidate's commit, so moving
  or adding a harness module changes the digest and needs a new candidate.
  `tests/release/qualification_harness_test.mjs` requires that list to equal
  the harness's import closure. No workflow runs Python;
  `scripts/ci/verify_ci_reliability.mjs` fails on a Python interpreter or
  package tool, a `*.py` script, a python shell, a Python image, or a
  `setup-python`/`setup-uv` step.
- Keep publishing logic in workflow only.
- Do not edit assets repository files from here outside publish flow.
- C++ exception catching is enabled only for the functions listed in
  `EXCEPTION_CATCHING_ALLOWED` (`CMakeLists.txt`):
  `llamadart_webgpu_grammar_sampler_init` and
  `llamadart_webgpu_lora_adapter_init`, which turn llama.cpp's grammar parser,
  trigger pattern and LoRA loader throws into `(invalid grammar)` and `Failed to
  load LoRA adapter` errors, `llamadart_webgpu_grammar_sampler_accept`, which
  turns a lazy grammar's throw on rejected trigger text into a generation error,
  and the speculative wrappers in
  `src/llama_webgpu_speculative.cpp`, which turn draft model, draft context and
  speculative setup throws into load or generation errors, and
  `llamadart_webgpu_state_set_data` and `llamadart_webgpu_state_read_file`,
  which turn state restore throws on a truncated or mismatched snapshot into
  a rejected state load. It also names `llama_init_from_model`, the
  `llama_context` constructor and `sched_reserve` (mangled) and is applied to
  `llama-context.cpp`, so a context that cannot be allocated is freed by
  llama.cpp's own catch instead of leaking; configuration fails when those
  upstream signatures change. Do not widen it with `-fexceptions` or
  `-fwasm-exceptions`: whole-TU catching puts `invoke_*` trampolines and their
  ASYNCIFY instrumentation on hot paths, and Wasm EH conflicts with ASYNCIFY.
  The link setting makes every other uncaught throw escape `ccall` as a
  `CppException`; `src/llama_webgpu_core_post.js` turns it back into
  `abort()` for `ccall`, the bridge's only entry into the core, so keep the two
  together.

## Cross-Repo Handoff to `llamadart`

After publishing assets tag:

1. Update/fetch pinned bridge assets in `llamadart`:
   `WEBGPU_BRIDGE_ASSETS_TAG=<tag> ./scripts/fetch_webgpu_bridge_assets.sh`
2. Update docs/changelog in `llamadart` if behavior changed.
