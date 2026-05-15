# llama-web-bridge

Reusable llama.cpp web bridge runtime (JS + WASM).

This repository provides:

- `src/llama_webgpu_core.cpp` (native bridge core)
- `js/llama_webgpu_bridge.js` (JS runtime wrapper)
- `CMakeLists.txt` for Emscripten builds

## Build

Requirements:

- Emscripten SDK (`emcmake`, `emcc`) in `PATH`
- llama.cpp source checkout matching `llama_cpp.version` or a compatible checkout exposing
  `llama_state_save_file` / `llama_state_load_file` with the signatures used by
  `src/llama_webgpu_core.cpp`

Build command:

```bash
./scripts/build_bridge.sh
```

Useful environment variables:

- `LLAMA_CPP_DIR` (path to llama.cpp source)
- `BUILD_DIR` (cmake build dir)
- `OUT_DIR` (output directory; defaults to `dist/`)
- `WEBGPU_BRIDGE_BUILD_MEM64` (`1` to also build optional wasm64 core assets)
- `WEBGPU_BRIDGE_MEM64_MAX_MEMORY` (optional wasm64 max linear memory bytes)
- `WEBGPU_BRIDGE_PTHREADS` (`1`/`0`, defaults to `1`)
- `WEBGPU_BRIDGE_PTHREAD_POOL_SIZE` (defaults to `4`)
- `WEBGPU_BRIDGE_PTHREAD_POOL_SIZE_STRICT` (defaults to `0`)

Notes:

- wasm64 builds default to `WEBGPU_BRIDGE_MEM64_MAX_MEMORY=12884901888` (12 GiB).
- Large single-file remote model loading requires a cross-origin isolated page
  (`COOP`/`COEP`) so worker-thread runtime paths are available.
- pthread builds preallocate `WEBGPU_BRIDGE_PTHREAD_POOL_SIZE` workers and cap
  bridge-selected thread counts to that compiled pool size.
  `WEBGPU_BRIDGE_PTHREAD_POOL_SIZE_STRICT` defaults to `0` so an unexpected
  over-pool request does not hard-abort the wasm runtime, but it can be
  overridden for stricter local diagnostics.

Build outputs:

- `dist/llama_webgpu_bridge.js`
- `dist/llama_webgpu_bridge_worker.js`
- `dist/llama_webgpu_core.js`
- `dist/llama_webgpu_core.wasm`

Optional outputs (when `WEBGPU_BRIDGE_BUILD_MEM64=1`):

- `dist/llama_webgpu_core_mem64.js`
- `dist/llama_webgpu_core_mem64.wasm`

## State persistence

The bridge exposes llama.cpp session/state persistence through both direct runtime
and worker-backed `LlamaWebGpuBridge` instances.

API:

- `await bridge.stateSaveFile(path, tokens = []) -> true`
- `await bridge.stateLoadFile(path, tokenCapacity = bridge.getContextSize()) -> { tokens }`
- `await bridge.stateSaveBytes(tokens = []) -> Uint8Array`
- `await bridge.stateLoadBytes(bytes, tokenCapacity = bridge.getContextSize()) -> { tokens }`

`stateSave*` snapshots the current llama.cpp context; it does not tokenize or
evaluate the supplied `tokens`. Save only after the prompt/prefix you want to
restore has already been evaluated by the bridge, then pass the exact token
sequence for that evaluated prompt/prefix:

```js
// After loadModelFromUrl(...) and after prompt/prefix evaluation:
const prefixTokens = await bridge.tokenize(prefixText, true);
await bridge.stateSaveFile('/prompt-state.bin', prefixTokens);

const restored = await bridge.stateLoadFile(
  '/prompt-state.bin',
  bridge.getContextSize(),
);
console.log(restored.tokens);

const bytes = await bridge.stateSaveBytes(prefixTokens);
await bridge.stateLoadBytes(bytes, bridge.getContextSize());
```

State files are opaque llama.cpp state/session files. They are tied to the same
model, llama.cpp build, and compatible runtime/model-load parameters. Loading a
state file from a different model/build can fail.

The `tokens` argument is stored in the llama.cpp state/session file and is
returned by `stateLoad*`; it is not evaluated by `stateSave*` and is not validated
against the KV cache. Passing the wrong token list can make later prompt-prefix
reuse incorrect. Passing `[]` is allowed, but gives the bridge no restored
prefix-token metadata to reuse.

`stateLoad*` requires `tokenCapacity` to be positive, large enough for the stored
token list, and no larger than the active context size. If omitted, the JS API
uses `bridge.getContextSize()`. Empty `stateLoadBytes` input is rejected. All
four state methods require a loaded model.

`stateSaveFile` and `stateLoadFile` operate on the active WASMFS instance. In a
browser this filesystem is virtual and not durable by default, and worker-mode
paths live inside the worker runtime. Use `stateSaveBytes` and `stateLoadBytes`
when the application needs to persist snapshots in IndexedDB, OPFS, Cache API, or
another app-managed durable store.

State save/load is rejected while generation is active. On successful load the
bridge restores the prompt token list returned as `{ tokens }`, so reissuing the
same prompt can reuse the loaded KV state via the existing prompt-prefix reuse
path.

## CI

This repo includes a wasm build gate in:

- `.github/workflows/ci.yml`

It builds against the pinned `llama.cpp` tag in `llama_cpp.version`, uploads build artifacts, and
runs the static CI reliability contract:

```bash
python3 scripts/verify_ci_reliability.py
```

The reliability contract protects the browser smoke and workflow invariants that
are easy to regress during agent-driven maintenance:

- both CI and publish workflows resolve the default llama.cpp tag from
  `llama_cpp.version`, so tag-triggered asset publishes cannot silently rebuild
  against a stale workflow default;
- `.github/workflows/auto_llama_cpp_update.yml` opens or updates one stable
  `automation/bump-llama-cpp` PR when a newer upstream release exists, with the
  upstream release notes, compare link, and commit range in the PR body, then
  dispatches the CI workflow on the automation branch so `GITHUB_TOKEN` branch
  updates still get a head-SHA validation run;
- both CI and publish workflows opt into `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24`
  to catch action-runtime deprecation issues early;
- the state-persistence browser smoke supports an integrity-checked tiny GGUF
  model round trip;
- the CI model cache path expands `~` before resolving so it matches the
  `actions/cache` directory;
- browser smoke failures upload `state-persistence-smoke-artifacts` with console
  logs, result JSON, and screenshots when available.

Run the model-backed smoke locally after building the bridge if a change touches
state persistence, workers, browser smoke, or workflow diagnostics:

```bash
python3 scripts/state_persistence_browser_smoke.py \
  --dist-dir /path/to/webgpu_bridge_dist \
  --model-url https://huggingface.co/aladar/llama-2-tiny-random-GGUF/resolve/main/llama-2-tiny-random.gguf \
  --model-sha256 81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb \
  --model-cache-dir ~/.cache/llama-web-bridge/state-smoke-models \
  --artifacts-dir /tmp/llama-web-bridge-state-smoke
```

Do not commit downloaded GGUFs, Playwright screenshots, console logs, generated
`dist/` assets, or Emscripten build/cache directories.

## Publishing

Published, versioned artifacts are consumed from:

- `leehack/llama-web-bridge-assets`

Publish workflow:

- `.github/workflows/publish_assets.yml`

Trigger modes:

- Automatic: push a `v*` tag in this repo (for example `v0.1.5`)
- Manual: run workflow dispatch with explicit inputs

Required repository secret:

- `WEBGPU_BRIDGE_ASSETS_PAT` (token with write access to
  `leehack/llama-web-bridge-assets`)

The publish workflow carries the resolved `llama.cpp` tag from the build job to
the release job as an explicit job output, so the asset release notes match the
`manifest.json` `llama_cpp_tag` value.

Example publish:

1. Create/push a release tag in this repo (for example `v0.1.5`)
2. `Publish Bridge Assets` runs automatically and publishes the same tag to
   `leehack/llama-web-bridge-assets`
3. Workflow also creates/updates the matching GitHub Release in
   `leehack/llama-web-bridge-assets`

Manual override example:

1. Run `Publish Bridge Assets` workflow
2. Inputs:
   - `assets_tag`: `v0.1.5`
   - `assets_repo`: `leehack/llama-web-bridge-assets`
   - `llama_cpp_tag`: leave empty to use `llama_cpp.version`, or set an explicit
     temporary override such as `b9165`

After publish, assets are CDN-available at:

- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@v0.1.1/llama_webgpu_bridge.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@v0.1.1/llama_webgpu_bridge_worker.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@v0.1.1/llama_webgpu_core.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@v0.1.1/llama_webgpu_core.wasm`

Note: CDN pinning fundamentally relies on git tags in the assets repo.

## Maintainer Docs

- `AGENTS.md`: agent workflow and cross-repo handoff
- `CONTRIBUTING.md`: contributor setup/build/publish steps
