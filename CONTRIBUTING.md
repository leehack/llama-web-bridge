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

Heavy Qwen3-ASR and Qwen3-TTS gates never run on hosted runners. Before
publishing, run the one required local qualification command against the exact
candidate artifact built by `.github/workflows/bridge_candidate.yml`:

```bash
python3 scripts/release_qualification.py qualify \
  --candidate-run-id <CANDIDATE_RUN_ID> \
  --speech-model-path /path/to/Qwen3-ASR-0.6B-Q8_0.gguf \
  --speech-mmproj-path /path/to/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --speech-audio-path /path/to/asr_en.wav \
  --tts-model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf \
  --tts-mmproj-path /path/to/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --output-attestation /tmp/qualification-attestation.json \
  --output-base64 /tmp/qualification-attestation.b64
```

The command downloads the candidate by immutable artifact ID only after proving
the named run is a successful `bridge_candidate.yml` dispatch on the
default-branch line whose complete inventory contains exactly one live
`exact-webgpu-bridge-dist` artifact. It refuses an unprovenanced local dist,
requires every model/projector/fixture path, gives smoke children only a narrow
non-secret environment allowlist, bounds both gate timeouts, and terminates the
full smoke process group on timeout or cancellation. It also compares the local
harness bytes with the exact candidate bridge commit before running a model
gate, so run it from a checkout that contains that commit. Dispatch the `.b64`
payload and the same candidate run ID to
`.github/workflows/qualification_attestation.yml`, then pass that ingestion run
ID to publication as `attestation_run_id`.

Direct individual smokes can also be run locally:

```bash
python3 scripts/speech_to_text_browser_smoke.py \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/Qwen3-ASR-0.6B-Q8_0.gguf \
  --model-sha256 bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971 \
  --mmproj-path /path/to/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --mmproj-sha256 41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d \
  --artifacts-dir /tmp/llama-web-bridge-speech-smoke
```

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
  changing `.github/workflows/ci.yml`, `.github/workflows/bridge_candidate.yml`,
  `.github/workflows/publish_assets.yml`,
  `.github/workflows/auto_llama_cpp_update.yml`,
  `.github/workflows/qualification_attestation.yml`, JS build pipeline files,
  `scripts/release_qualification.py`, or
  `scripts/state_persistence_browser_smoke.py`.
- Rotate all 7 model/projector SHA-256 pins in the five files that hard-code
  them together: `README.md`, `AGENTS.md`, `CONTRIBUTING.md`,
  `.github/workflows/ci.yml`, `.github/workflows/bridge_candidate.yml`.
  `scripts/verify_ci_reliability.py` requires the five sets to be identical with
  exactly 7 pins each; a stale `bridge_candidate.yml` breaks the candidate job,
  not just CI. `publish_assets.yml` holds no pins because it neither builds nor
  smokes. `scripts/release_qualification.py` carries the same 7 plus the pinned
  ASR audio fixture, and every attestation must match them exactly.
- The script also compares pins role by role across the two workflows, whose env
  keys name the role, so a swap between roles in one workflow fails the gate. A
  swap applied identically to both passes, as does one confined to the three
  markdown files, whose bare `--model-sha256` / `--mmproj-sha256` flags carry no
  role -- check each markdown pin by hand against the `--model-url` /
  `--model-path` / `--mmproj-path` value directly above it, whose filename names
  the model or projector the pin belongs to.
- Keep `scripts/multimodal_browser_smoke.py` in normal CI for every llama.cpp
  pin update; build-only validation does not cover mtmd prompt ingestion.
- Heavy real-model ASR and TTS gates run only through
  `scripts/release_qualification.py` during local qualification. Keep the
  candidate manifest honest: `generate_release_manifest.py` must record those
  two gates as `required-local-attestation`, never as a hosted pass, and must
  keep real-device playback, intelligibility, and speaker-reference fidelity in
  `unproven_capabilities`.
- Nothing may rebuild the candidate after it is built. The manifest embeds the
  candidate run ID/URL, so a rebuild changes the digest and no attestation could
  match it. Candidate workflow attempts are first-attempt-only; after a failed
  build or hosted gate, dispatch a new candidate run instead of rerunning one.
- Preserve `llama_cpp.version` as the default ordinary CI/development build pin.
  It holds exactly one upstream tag in either channel, stable
  `vMAJOR.MINOR.PATCH` or development `bNNNN`; `scripts/verify_ci_reliability.py`
  rejects every other form. Exact release publication receives upstream identity
  from the orchestrator and must not require a bridge pin PR.
- Preserve `emsdk.version` as the single compiler source for CI and publish.
  Both workflows must verify the active `emcc` version, and published manifests
  must record that verified identity.
- Main-branch and PR CI never dispatch publication. Scheduled stable release
  discovery prepares an ordered `release-candidates.json` backlog for every
  stable native release after the immutable native `v0.2.0-1` / Web-assets
  `v0.1.39` baseline and invokes `scripts/stable_release_orchestrator.py` to
  idempotently advance candidate, qualification-attestation, and publication
  stages. The scan downloads every selected `assets.json` and `SHA256SUMS` by
  unique release-asset ID and runs the complete native release/tag/inventory
  contract first. Each exact pipeline advances by at most one stage per scan,
  so an older local-qualification wait does not starve a newer stable release.
  Manual `development` scans stay scan-only: they resolve exact `bNNNN`
  provenance and report it, and the orchestrator refuses non-stable provenance.
  A failed candidate is never retried automatically each day; after diagnosis,
  a maintainer may deliberately dispatch one new first-attempt run with the same
  exact binding, and a later unique success supersedes the recorded failed run.
  A successful candidate waits for maintainer-run local ASR/TTS qualification.
  After its owner-authorized attestation is ingested, a default-branch
  `workflow_run` continuation re-proves that exact successful ingestion and
  advances only its exact candidate through the same orchestrator. It classifies
  earlier stable pipelines without mutation so the existing publication-order
  barrier remains binding; the next scheduled or manual stable scan is the
  fallback if the continuation is delayed or missed. Run recovery paginates
  each exact workflow's filtered history with stable-count checks and splits
  searches at GitHub's 1,000-result cap into closed time windows, so an older
  waiting pipeline is not lost as later workflow history grows.
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

Use `.github/workflows/publish_assets.yml` only after the publication approval
assertion is backed by the live `bridge-assets-publication` policy. The
repository owner, either directly or through the scheduled orchestrator using
that existing authorized identity, dispatches this
bridge-owned, non-reusable workflow with a required correlation ID, exact bridge SHA,
upstream tag/commit, native release tag plus `assets.json` SHA-256,
output release tag/rebuild, distinct required `candidate_run_id` and
`attestation_run_id`, and assets repository.

Immutable releases must already be enabled on the assets repository before the
candidate is dispatched. Confirm it with
`gh api repos/leehack/llama-web-bridge-assets/immutable-releases`, then dispatch
`bridge_candidate.yml` with `assets_immutable_releases_enabled=true`; the
candidate cannot read that setting itself and fails closed on the assertion.
Publication downloads the exact run-owned prequalification record and binds its
explicit boolean assertion to the candidate and run identities before it
re-proves the real state through the same endpoint and before it pushes,
requiring the exact `{enabled, enforced_by_owner}` shape with `enabled`
explicitly boolean `true`, and treats every non-200, missing, false, or
non-boolean answer as a failure. Every complete release state, including one
found by a retry, is read back by tag and by release ID. Both reads must bind the
exact tag, release ID, tag commit, and published state and report `immutable` as
explicit boolean `true`. Publication verifies GitHub's signed release attestation with
`gh release verify <tag> --repo leehack/llama-web-bridge-assets --format json`,
requiring predicate type `https://in-toto.io/attestation/release/v0.2` signed by
`https://dotcom.releases.github.com`, predicate database ID equal to the live
readback release ID, and the exact published artifact digests. A failure is reported as
`immutable-publication-unverified`. An incomplete published release is never
filled in; the release is never deleted, retagged, overwritten, or repaired.

The request must set `publish_approved=true`. Publication remains blocked until
repository administrators separately create `bridge-assets-publication`, disable
administrator bypass, restrict the custom deployment branch policy to `main`,
and store the assets PAT as an environment-scoped secret. The solo-maintainer
publication contract does not require a reviewer rule, `prevent_self_review`, or
a two-person approval quorum. The workflow uses the default job token to verify
the environment identity, disabled administrator bypass, and exact `main`
deployment branch policy before approval. It repeats those checks after approval
immediately before the first publication-PAT-bearing step, using the trusted
workflow commit's validator so an older requested build-source SHA remains
compatible. The trusted commit also supplies attestation, provenance, and
publication-state policy; the historical source supplies only the exact harness
bytes, toolchain pin, and build identity. The environment-scoped
`WEBGPU_BRIDGE_ASSETS_PAT` is the only
external credential; each step that can use it fails closed unless the injected
value is non-empty and never prints the value. The workflow also verifies all
identities plus native GitHub asset digests/inventory. It never builds: it
proves the repository, workflow file, dispatch event, default-branch-line head,
success, complete artifact inventory, and uniqueness of both the candidate run
and the attestation ingestion run, downloads both by immutable artifact ID, and
verifies the attestation against the candidate twice -- once before approval and
again inside the privileged job. It then orders stable and development histories
independently and recovers exact ref-only partial states while rejecting any
published-release mismatch rather than trying to repair it.
Manifest and outcome records carry the candidate run ID/URL, because that is the
identity the candidate manifest embeds; unavailable post-mutation re-queries
produce retryable `mutation-unknown` records rather than guessed state.
Publication run attempts are first-attempt-only: retry partial publication by
redispatching a new publication run against the same `candidate_run_id` and
`attestation_run_id`, never rerunning the old run. The fingerprinted identity
comes from the candidate run, so it stays stable. A different candidate must
never target an existing output tag.

The scheduled orchestrator reuses that same existing secret only from a
`bridge-assets-publication` environment job; it does not require an
`ORCHESTRATOR_DISPATCH_TOKEN`. For automatic progression, the owner-bound token
must retain Actions write permission on `leehack/llama-web-bridge` in addition
to the assets-repository permissions publication already requires. The job token
validates the environment policy first; the orchestrator then proves the PAT's
owner identity and live immutable-release governance before dispatch. It adds
`assets_immutable_releases_enabled=true` only from that governance response and
adds `publish_approved=true` only after the live publication-environment policy
passes. An already-published noop still resolves the exact assets tag commit,
validates independent release reads by tag and ID, and validates the signed
`gh release verify --format json` attestation against every downloaded asset
digest. Manual runs require both the initiating actor and triggering actor to be
the repository owner before either workflow job starts, preventing a
collaborator from using the environment PAT as a confused deputy. Trusted
default-branch schedule events remain automatic.

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
