# llama-web-bridge

Reusable llama.cpp web bridge runtime (JS + WASM).

This repository provides:

- `src/llama_webgpu_core.cpp` (native bridge core; it includes its parts from
  `src/core/`)
- `js/src/llama_webgpu_bridge.js` (JS runtime wrapper entry; its modules live
  under `js/src/`)
- `js/llama_webgpu_bridge.js` (generated bundled browser ESM wrapper)
- `js/llama_webgpu_bridge.d.ts` (public TypeScript declaration asset)
- `docs/api.md` (public JavaScript API reference)
- `CMakeLists.txt` for Emscripten builds

## Install via npm

```bash
npm install llama-web-bridge
```

```js
import { LlamaWebGpuBridge } from 'llama-web-bridge';

const bridge = new LlamaWebGpuBridge();
```

The package entry point includes TypeScript declarations and exports
`enableBridgeWorkerHost` for custom worker bootstraps. Vite production builds
and webpack 5 can emit the default worker and core assets alongside your app.

Threaded cores need cross-origin isolation. Serve your app with the following
response headers.

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

If the page cannot be isolated, construct the bridge with
`{ allowSingleThreadCore: true }` to select the single-thread variants. They
use one CPU thread and can still offload layers to WebGPU. If you use CSP,
allow `'self'`, `blob:`, and the asset origin in `worker-src`. See
[deployment requirements](docs/api.md#browser-and-asset-requirements) for
cross-origin assets and the other CSP directives.

For instruction-tuned models, format the prompt with the loaded model's chat
template before calling `createCompletion()`. The JS/TS API accepts formatted
prompt text and leaves chat templating to the caller.

### Bundler configuration

In Vite 7 dev, add this setting to `vite.config.ts` so the bridge module stays
beside its worker and core files.

```ts
export default {
  optimizeDeps: { exclude: ['llama-web-bridge'] },
};
```

If your bundler does not emit `new URL(..., import.meta.url)` assets, as with
standalone esbuild, serve the package's `dist/` runtime files together at a
static URL. Pass `{ assetBaseUrl: '/webgpu-bridge/' }` if that is where they are
served. The base can also be an absolute CDN directory URL. Explicit per-file
URLs take precedence.

### Custom core factory

The package exports the worker at `llama-web-bridge/worker` and core factories at
`llama-web-bridge/core`, `llama-web-bridge/core-mem64`,
`llama-web-bridge/core-st`, and `llama-web-bridge/core-mem64-st`. Each core
subpath has a matching `.wasm` subpath. Providing a factory selects the direct
runtime. Pass the matching Wasm URL explicitly so the factory can find the
binary even when the bundler renames it.

```js
import { LlamaWebGpuBridge } from 'llama-web-bridge';
import createCore from 'llama-web-bridge/core-st';

const bridge = new LlamaWebGpuBridge({
  allowSingleThreadCore: true,
  coreModuleFactory: createCore,
  wasmUrl: new URL('llama-web-bridge/core-st.wasm', import.meta.url).href,
});
```

### CDN modules

You can import the same package from jsDelivr or unpkg in a browser module.
Replace `<version>` with a published npm version. The same isolation and CSP
rules apply.

```html
<script type="module">
  import { LlamaWebGpuBridge } from 'https://cdn.jsdelivr.net/npm/llama-web-bridge@<version>/dist/llama_webgpu_bridge.js';

  const bridge = new LlamaWebGpuBridge();
</script>
```

### 0.x compatibility

npm versions mirror unsuffixed stable bridge asset tags (`v0.1.X` becomes
`0.1.X`). During `0.x`, any release may change the JavaScript API. See
[CHANGELOG.md](CHANGELOG.md) for changes. Strict SemVer starts at `1.0.0`.

## Public API

See the [LlamaWebGpuBridge reference](docs/api.md) for model loading, generation,
embeddings, state persistence, diagnostics, and browser deployment. The
published browser assets include `llama_webgpu_bridge.d.ts`.

## Build

Use Node.js and the Emscripten SDK version listed in the
[local build guide](CONTRIBUTING.md#local-build). Check out llama.cpp at the tag
in `llama_cpp.version`.

```bash
npm ci
npm run check:js
./scripts/build/build_bridge.sh
```

The default build produces the bridge JS, worker and declarations alongside
the threaded wasm32 core. Enable memory64 with `WEBGPU_BRIDGE_BUILD_MEM64=1`
and single-thread cores with `WEBGPU_BRIDGE_BUILD_SINGLE_THREAD=1`. Keep the
1 MiB stack default unless real-model smokes qualify a smaller stack. Threaded
cores cap requested thread counts to their compiled pthread pool. See the
[build recipe and outputs](CONTRIBUTING.md#validate-outputs) for details.

## Capabilities

- **Single-thread cores and JS-backed staging.** Use single-thread cores on
  non-isolated pages. WebKit defaults to JS-backed staging on a supported
  core. On Apple Safari before version 27, the default cap is one GPU layer,
  including on full-offload requests.
  [Staging API](docs/api.md#js-backed-model-staging) ·
  [Browser measurements](docs/single_thread.md)
- **Multimodal embeddings.** Pool ordered text, image, and audio parts with
  compatible models and projectors. Probe the active runtime with
  `getEmbeddingCapabilities()`.
  [API](docs/api.md#embeddings) · [BF16 CPU results](docs/multimodal_embeddings.md) ·
  [Qualification command](CONTRIBUTING.md#multimodal-embedding-qualification)
- **State persistence.** Save and restore a session in either runtime. Use
  `stateSaveBytes` and `stateLoadBytes` for durable browser storage.
  [State API](docs/api.md#state-persistence)
- **Text-to-speech.** Probe Qwen3-TTS support before synthesizing PCM or
  streaming audio chunks. The validated model pair uses memory64.
  [TTS API](docs/api.md#text-to-speech) ·
  [Real-model smoke](scripts/smoke/text_to_speech.mjs)
- **Decision heads.** Load a compatible head for an encoder and run decision
  sequences. Head handles belong to the active core.
  [Decision API](docs/api.md#decision-heads) · [Smoke](scripts/smoke/decision.mjs)
- **LoRA adapters.** Load and apply compatible adapters to a loaded model.
  [LoRA API](docs/api.md#lora-adapters) · [Smoke](scripts/smoke/lora_adapter.mjs)

## CI and releases

[CI](.github/workflows/ci.yml) checks the pinned llama.cpp commit and builds
threaded and single-thread cores for wasm32 and memory64. Browser smokes cover
npm packaging, state persistence, cross-origin assets, shared-core staging,
resume integrity, core abort recovery and release, remote fetch, multimodal
completion, grammar, next-token scoring, and tiny speculative decoding.
[Current pin qualification](docs/v100_qualification.md)

```bash
npm run check:js
node scripts/ci/verify_ci_reliability.mjs
```

The maintainer dispatches the candidate build, hosted qualification, and
immutable asset publication in order. Asset publication uses the
`WEBGPU_BRIDGE_ASSETS_PAT` environment secret. The separate
[`publish_npm.yml`](.github/workflows/publish_npm.yml) workflow stages the exact
qualified release bytes and publishes using npm Trusted Publishing.
[CI guardrails](CONTRIBUTING.md#agent-workflow-guardrails) ·
[Asset publish process](CONTRIBUTING.md#publish-process)

## Published CDN assets

Use one pinned [asset release](https://github.com/leehack/llama-web-bridge-assets/releases)
tag for the bridge, worker, and cores. For CORS, CORP, and CSP on cross-origin
deployments, see [browser requirements](docs/api.md#browser-and-asset-requirements).

- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_bridge.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_bridge_worker.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_bridge.d.ts`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_core.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_core.wasm`

When `WEBGPU_BRIDGE_BUILD_MEM64=1`, the assets repo also publishes optional
memory64 core artifacts:

- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_core_mem64.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_core_mem64.wasm`

When `WEBGPU_BRIDGE_BUILD_SINGLE_THREAD=1` (and `WEBGPU_BRIDGE_BUILD_MEM64=1`), the assets repo also publishes single-thread (`_st`) core artifacts:

- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_core_st.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_core_st.wasm`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_core_mem64_st.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_core_mem64_st.wasm`

## Maintainer docs

- [CONTRIBUTING.md](CONTRIBUTING.md) covers build recipes, runnable smokes, CI
  guardrails, and asset publishing.
- [AGENTS.md](AGENTS.md) covers agent workflow and change boundaries.
- [API reference](docs/api.md) and [browser measurements](docs/single_thread.md)
  cover integration and runtime behavior.
