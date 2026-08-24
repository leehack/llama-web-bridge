# Contributing to llama-web-bridge

Thanks for contributing.

## Purpose

This repo builds the reusable JS/WASM bridge runtime for llama.cpp web usage.
Published artifacts are consumed from `llama-web-bridge-assets`.

## Prerequisites

- Emscripten SDK (`emcmake`, `emcc`) matching `emsdk.version`
- Node.js/npm for JS bridge bundling and TypeScript `checkJs`
- CMake toolchain
- Access to a llama.cpp checkout matching `llama_cpp.version`

## Setup

```bash
git clone https://github.com/leehack/llama-web-bridge.git
cd llama-web-bridge
npm ci
./scripts/build_bridge.sh --help
```

## Local Build

```bash
npm run check:js
./scripts/build_bridge.sh
# or
LLAMA_CPP_DIR=../llama.cpp OUT_DIR=dist ./scripts/build_bridge.sh
```

Bridge wrapper source lives under `js/src/`; `npm run build:js` regenerates the
checked-in browser ESM outputs and declarations under `js/`. `npm run check:js`
runs the same generator plus TypeScript and syntax checks, so commit any updated
`js/` outputs after source changes.

For local agent/maintainer validation, prefer external build and cache paths so
generated files do not dirty the checkout:

```bash
export CCACHE_DIR=/private/tmp/llama_web_bridge_ccache
export EM_CACHE=/private/tmp/llama_web_bridge_emcache
BUILD_DIR=/private/tmp/llama_web_bridge_build \
MEM64_BUILD_DIR=/private/tmp/llama_web_bridge_build_mem64 \
OUT_DIR=/private/tmp/llama_web_bridge_dist \
WEBGPU_BRIDGE_BUILD_MEM64=1 \
./scripts/build_bridge.sh
```

## Validate Outputs

Expected files:

- `dist/llama_webgpu_bridge.js`
- `dist/llama_webgpu_bridge_worker.js`
- `dist/llama_webgpu_bridge.d.ts`
- `dist/llama_webgpu_core.js`
- `dist/llama_webgpu_core.wasm`

Before opening or updating a PR, run the lightweight contracts:

```bash
npm run check:js
python3 -m py_compile scripts/verify_state_persistence_api.py scripts/verify_text_to_speech_api.py scripts/verify_ci_reliability.py scripts/state_persistence_browser_smoke.py scripts/multimodal_browser_smoke.py scripts/speech_to_text_browser_smoke.py scripts/text_to_speech_browser_smoke.py
python3 scripts/verify_state_persistence_api.py
python3 scripts/verify_text_to_speech_api.py
python3 scripts/verify_ci_reliability.py
```

For state-persistence, worker, or workflow changes, also run the browser smoke
against a built dist directory. Use a checksum-pinned tiny model and keep caches
and artifacts outside the repository:

```bash
python3 scripts/state_persistence_browser_smoke.py \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-url https://huggingface.co/aladar/llama-2-tiny-random-GGUF/resolve/main/llama-2-tiny-random.gguf \
  --model-sha256 81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb \
  --model-cache-dir ~/.cache/llama-web-bridge/state-smoke-models \
  --artifacts-dir /tmp/llama-web-bridge-state-smoke
```

For llama.cpp pin or multimodal changes, run checksum-pinned real image
inference through both direct and worker runtimes:

```bash
python3 scripts/multimodal_browser_smoke.py \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/Qwen3.5-0.8B-Q4_K_M.gguf \
  --model-sha256 bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517 \
  --mmproj-path /path/to/mmproj-F16.gguf \
  --mmproj-sha256 56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453 \
  --artifacts-dir /tmp/llama-web-bridge-multimodal-smoke
```

Before publishing assets intended to support typed Qwen3-ASR, run the opt-in
checksum-pinned speech gate. It validates cold, cancellation, and warm-reuse
results in wasm32 and memory64, through both direct and worker runtimes. Qwen's
official English WAV fixture, its SHA-256, and the expected Web transcript are
pinned by the script:

```bash
python3 scripts/speech_to_text_browser_smoke.py \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/Qwen3-ASR-0.6B-Q8_0.gguf \
  --model-sha256 bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971 \
  --mmproj-path /path/to/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --mmproj-sha256 41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d \
  --artifacts-dir /tmp/llama-web-bridge-speech-smoke
```

Before publishing assets intended to support Qwen3-TTS, run the opt-in
checksum-pinned memory64 gate through both direct and worker runtimes. The
validated model/projector pair is too large for a practical wasm32 product
path:

```bash
python3 scripts/text_to_speech_browser_smoke.py \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf \
  --model-sha256 8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129 \
  --mmproj-path /path/to/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --mmproj-sha256 6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2 \
  --memory-mode wasm64 \
  --runtime-mode all \
  --gpu-layers 99 \
  --artifacts-dir /tmp/llama-web-bridge-text-to-speech-smoke
```

If the smoke downloads from a URL, errors and diagnostics must redact userinfo,
query strings, and fragments before printing the location.

## Agent Workflow Guardrails

- Keep workflow reliability rules in `scripts/verify_ci_reliability.py` when
  changing `.github/workflows/ci.yml`, `.github/workflows/publish_assets.yml`,
  `.github/workflows/auto_llama_cpp_update.yml`, JS build pipeline files, or
  `scripts/state_persistence_browser_smoke.py`.
- Rotate all 7 model/projector SHA-256 pins in the five files that hard-code
  them together: `README.md`, `AGENTS.md`, `CONTRIBUTING.md`,
  `.github/workflows/ci.yml`, `.github/workflows/publish_assets.yml`.
  `scripts/verify_ci_reliability.py` requires the five sets to be identical with
  exactly 7 pins each; a stale `publish_assets.yml` breaks the release job, not
  just CI.
- The script also compares pins role by role across the two workflows, whose env
  keys name the role, so a swap between roles in one workflow fails the gate. A
  swap applied identically to both passes, as does one confined to the three
  markdown files, whose bare `--model-sha256` / `--mmproj-sha256` flags carry no
  role -- check each markdown pin by hand against the `--model-url` /
  `--model-path` / `--mmproj-path` value directly above it, whose filename names
  the model or projector the pin belongs to.
- Keep `scripts/multimodal_browser_smoke.py` in normal CI for every llama.cpp
  pin update; build-only validation does not cover mtmd prompt ingestion.
- Keep `scripts/speech_to_text_browser_smoke.py` opt-in because its model pair
  is large, but require it before publishing assets advertised for typed ASR.
- Keep `scripts/text_to_speech_browser_smoke.py` opt-in because its model pair
  is large and requires memory64, but require it before publishing assets
  advertised for Qwen3-TTS.
- Preserve `llama_cpp.version` as the default ordinary CI/development build pin.
  Exact release publication receives upstream identity from the orchestrator and
  must not require a bridge pin PR.
- Preserve `emsdk.version` as the single compiler source for CI and publish.
  Both workflows must verify the active `emcc` version, and published manifests
  must record that verified identity.
- Main-branch and PR CI never dispatch publication. Scheduled discovery only
  prepares `release-candidate.json` and a job summary.
- Bridge source changes still require ordinary PRs. A dependency-version release
  may reuse an already-merged exact bridge source SHA without changing
  `llama_cpp.version`.
- Preserve `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` in CI and publish workflows so
  GitHub Action runtime changes are detected before they become mandatory.
- Upload state-persistence smoke diagnostics only on failure; successful CI runs
  should stay quiet beyond the normal build artifacts.
- Do not push branches, tags, or publish assets from local agent work unless the
  maintainer explicitly requests that side effect.

## Publish Process

Use `.github/workflows/publish_assets.yml` only after explicit publication
approval. The central orchestrator or a maintainer dispatches this bridge-owned,
non-reusable workflow with a required correlation ID, exact bridge SHA,
upstream tag/commit, native release tag plus
`assets.json` SHA-256, output release tag/rebuild, and assets repository.

The request must set `publish_approved=true`. Publication remains blocked until
repository administrators separately create `bridge-assets-publication`, add
the pinned repository-owner maintainer reviewer, disable administrator bypass
and self-review, restrict the
custom deployment branch policy to `main`, and store the assets PAT as an
environment-scoped secret. Administrators must also configure the separate
repository or organization Actions secret `BRIDGE_PUBLICATION_ENV_READ_TOKEN`
as a fine-grained credential scoped to this repository with Environments read
permission. The workflow fails closed if either external credential or the
environment protection is absent. It verifies all identities plus native
GitHub asset digests/inventory, uses the default job token for environment and
main-branch policy metadata, and uses the separate read credential only for the
complete paginated environment-secret name inventory. Do not duplicate the read
credential in the environment, where environment-secret precedence would
shadow the intended repository/organization credential after approval. The
workflow repeats those checks after approval immediately before the first
publication-PAT-bearing step, using the trusted workflow commit's validator so
an older requested build-source SHA remains compatible. The API
check proves the expected secret name is environment-scoped without reading its
value. It
builds wasm32 and memory64, runs mandatory state/multimodal/ASR/TTS gates,
orders stable and development histories independently, generates a
deterministic schema-v2 manifest, and recovers exact partial states while
rejecting any mismatch. Manifest and outcome records include the exact bridge
workflow run ID/URL and explicit conclusions for every mandatory capability
gate; unavailable post-mutation re-queries produce retryable
`mutation-unknown` records rather than guessed state.
Retry partial publication by rerunning the same GitHub Actions run so its
fingerprinted run ID/URL remains stable. Do not redispatch a new run against an
existing output tag.

GitHub artifact tags are `vMAJOR.MINOR.PATCH`, `vMAJOR.MINOR.PATCH-N`, `bNNNN`,
or `bNNNN-N`. Historical `bNNNN-llamadart.N` and prior wrapper forms are
read-only compatibility inputs. Any future npm package must use an independently
monotonic version with stable/nightly dist-tags because npm treats
`vMAJOR.MINOR.PATCH-N` as a prerelease.

Never interpolate `${{ inputs.* }}` directly inside a workflow `run` script.
Transport dispatch inputs through `env` and use quoted shell expansions.

## Repository Boundaries

- Bridge runtime source/build belongs here.
- Versioned static artifacts belong in `llama-web-bridge-assets`.
- Consumer integration (loading/fallback behavior) belongs in `llamadart`.
