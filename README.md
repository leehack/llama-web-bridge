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

## Public API

Applications that consume the browser assets directly should start with the
[`LlamaWebGpuBridge` public API reference](docs/api.md). The generated
`llama_webgpu_bridge.d.ts` file is published with the JS/WASM assets; the API
reference explains the runtime behavior for model loading, generation,
tokenization, embeddings, next-token scoring, multimodal projector support, LoRA adapters,
state persistence, metadata, cancellation, disposal, and worker-host bootstrap.

## Build

Requirements:

- Emscripten SDK (`emcmake`, `emcc`) matching `emsdk.version` in `PATH`
- Node.js 22.18+ or 24.2+ and npm for the JS bridge build pipeline (`npm ci`,
  `npm run check:js`)
- llama.cpp source checkout matching `llama_cpp.version` or a compatible checkout exposing
  `llama_state_save_file` / `llama_state_load_file` with the signatures used by
  the native core (`src/llama_webgpu_core.cpp` and `src/core/`)

Build command:

```bash
npm ci
npm run check:js
./scripts/build/build_bridge.sh
```

`./scripts/build/build_bridge.sh` also runs the JS bridge build before copying wrapper
assets, but running `npm run check:js` explicitly is useful before PRs because it
performs TypeScript `checkJs`, regenerates the checked-in browser ESM wrapper and
declaration files with esbuild, and syntax-checks the generated bridge files.

`./scripts/build/build_bridge.sh --help` lists every environment variable the build
reads (source, build, and output directories, wasm64, stack, memory, and
pthread settings) with its default.

Notes:

- Keep `WEBGPU_BRIDGE_STACK_SIZE` at or above its 1 MiB default unless both
  wasm32 and memory64 real-model smokes prove a lower value safe. Current
  llama.cpp graph parameters can overflow Emscripten's 64 KiB default stack.
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
- `dist/llama_webgpu_bridge.d.ts`
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

For `stateLoad*`, a `tokenCapacity` whose numeric conversion is greater than zero
is truncated and used; all other values fall back to
`bridge.getContextSize()`. The resolved capacity must be large enough for the
stored token list and no larger than the active context size. Empty
`stateLoadBytes` input is rejected. All four state methods require a loaded
model.

`stateSaveFile` and `stateLoadFile` operate on the active WASMFS instance. In a
browser this filesystem is virtual and not durable by default, and worker-mode
paths live inside the worker runtime. Use `stateSaveBytes` and `stateLoadBytes`
when the application needs to persist snapshots in IndexedDB, OPFS, Cache API, or
another app-managed durable store.

State save/load requests issued while generation is active wait in the bridge's
FIFO operation queue and run after generation settles. On successful load the
bridge restores the prompt token list returned as `{ tokens }`, so reissuing the
same prompt can reuse the loaded KV state via the existing prompt-prefix reuse
path.

## Text-to-speech

The bridge exposes versioned, capability-gated Qwen3-TTS synthesis through
`getTextToSpeechCapabilities()` and `synthesizeSpeech()`. The result is 24 kHz
mono `Float32Array` PCM for the currently validated Qwen3-TTS projector; audio
playback and WAV encoding belong to the consuming application. Both direct and
worker runtimes support progress, cancellation, language selection, and
optional encoded speaker-reference audio when the loaded projector reports the
capability.

The validated Qwen3-TTS 1.7B model/projector pair totals about 1.48 GB before
runtime buffers. It requires the memory64 browser runtime in practice and has a
high browser-memory and latency cost. wasm32 is not a supported product path for
this pair. Both direct and worker runtimes generate valid audio with a
WebGPU-selected model/projector path, while the CPU/WASM fallback remains
functional but substantially slower. This generated-audio support is
experimental upstream and stays out of default CI.

Run the checksum-pinned real-model gate,
`scripts/smoke/text_to_speech.mjs`, before publishing TTS-capable
assets; the invocation and pins are in [CONTRIBUTING.md](CONTRIBUTING.md#validate-outputs).

## Decision heads

The bridge runs Laya-style decision models through `getDecisionCapabilities()`,
`loadDecisionHead()`, `runDecision()` and `freeDecisionHead()`: a ModernBERT
encoder GGUF plus a safetensors head that returns raw option logits and act
logits for pre-tokenized sequences. Each head gets a private encoder context and
runs on WebGPU when the model was loaded with GPU layers, in both direct and
worker runtimes and in wasm32 and memory64. See [docs/api.md](docs/api.md#decision-heads).

The real-model smoke, `scripts/smoke/decision.mjs`, compares every
fixture row with a Laya reference and stays out of default CI; the invocation is
in [CONTRIBUTING.md](CONTRIBUTING.md#validate-outputs). No qualification gate
runs it, so release manifests list no decision capability; probe support with
`getDecisionCapabilities()`.

## LoRA adapters

`loadLoraAdapter()` loads a GGUF LoRA adapter from a URL or bytes, and
`setLoraAdapter()`, `removeLoraAdapter()` and `clearLoraAdapters()` change the
applied set and scales at runtime, in direct and worker runtimes and in wasm32
and memory64. aLoRA adapters are rejected. Probe support with
`getLoraAdapterCapabilities()`. See [docs/api.md](docs/api.md#lora-adapters).

The real-model smoke, `scripts/smoke/lora_adapter.mjs`, stays out of
default CI; the invocation is in
[CONTRIBUTING.md](CONTRIBUTING.md#validate-outputs).

## CI

This repo includes a wasm build gate in:

- `.github/workflows/ci.yml`

After the JS/compatibility contracts pass, it builds wasm32 and memory64 against
the pinned `llama.cpp` tag in `llama_cpp.version` and runs real
state-persistence, multimodal, grammar and next-token-score browser smokes. A
successful run uploads its seven built files as `webgpu-bridge-dist`; a failed
run uploads no dist and instead the smoke diagnostics that exist
(`state-persistence-smoke-artifacts`, `multimodal-smoke-artifacts`,
`grammar-smoke-artifacts`, `next-token-scores-smoke-artifacts`). CI never
changes a pin or publishes assets. Builds against llama.cpp v0.4.0 are no longer
tested in CI.
To run the media-helper and static CI contracts locally:

```bash
node tests/bridge/mtmd_compat_contract_test.mjs
node scripts/ci/verify_ci_reliability.mjs
```

The reliability contract checks the publication-safety invariants of the CI,
candidate, qualification, publication, and orchestration workflows. It parses
the workflows rather than matching their wording, so rewording a step or its
comments never fails it:

- CI, candidate, and publish workflows install dependencies with
  `npm ci --ignore-scripts`, run `npm run check:js`, and fail on stale checked-in
  generated output via `git diff --exit-code`; `check:js` ends with `npm test`,
  which runs every `tests/**/*_test.mjs` contract test, including the release
  contract suites; the candidate and publish workflows run `check:js` and this
  contract themselves; and no workflow runs a Python interpreter or package
  tool, a `*.py` script, a python shell or a Python image;
- `llama_cpp.version` holds one exact upstream tag in either channel (stable
  `vMAJOR.MINOR.PATCH` or development `bNNNN`) and CI builds it;
  `emsdk.version` holds one exact Emscripten version, which CI and the candidate
  install and verify against the resolved `emcc`, and the candidate records as
  `emscripten_version` in `manifest.json`;
- all 7 model/projector SHA-256 pins are identical across `CONTRIBUTING.md`,
  `ci.yml`, and `bridge_candidate.yml`, every role's pin and URL match
  `scripts/release/qualification.mjs` and each other across workflows, and `README.md`
  and `AGENTS.md` hold no pins;
- every workflow and job keeps the job token read-only, and no guard step or
  job may `continue-on-error` (only the publication ref mutation, whose outcome
  a live re-query classifies);
- `WEBGPU_BRIDGE_ASSETS_PAT` is the only secret in any workflow, referenced only
  by the publication and orchestration workflows and only through a step's
  `env`; in the publication workflow every step that binds it starts with
  `set -euo pipefail` and a fail-closed empty-token guard, and never prints it;
- publication is dispatch-only, requires `publish_approved=true`, serializes
  runs, validates the `bridge-assets-publication` environment policy with the
  job token before the privileged job and again immediately before the first
  PAT-bearing step, and keeps both jobs read-only;
- candidate, qualification, and publication refuse a non-owner dispatcher and
  any run attempt other than 1; the candidate holds no PAT or environment and
  builds once, memory64 included, recording its state-persistence and
  multimodal gate outcomes; qualification accepts no transported attestation; publication
  never rebuilds, downloads the candidate and attestation by immutable artifact
  ID, verifies the attestation in both jobs with the trusted validators, proves
  immutable-release governance before any ref mutation, reads the release back
  as immutable and attested, and never deletes, retags, or overwrites a release;
- `.github/workflows/auto_llama_cpp_update.yml` fetches provenance by release
  asset ID, gates every job on the repository owner, uses the PAT only inside the
  validated publication environment, dispatches only through the orchestrator,
  and never opens a PR or pushes; ordinary CI never dispatches publication or
  holds `actions: write`;
- CI runs the state-persistence, multimodal, grammar, and next-token-score
  browser smokes.

Run `scripts/smoke/state_persistence.mjs` locally after building the
bridge if a change touches state persistence, workers, browser smoke, or
workflow diagnostics, and `scripts/smoke/multimodal.mjs` for llama.cpp
pin or multimodal changes. Both invocations, with their pinned models, are in
[CONTRIBUTING.md](CONTRIBUTING.md#validate-outputs).

## Automated Qualification and Attestation

`.github/workflows/bridge_candidate.yml` builds the exact candidate once, runs
the compiler/build/contract/state-persistence/multimodal gates, and uploads the
candidate bundle. Nothing rebuilds that bundle afterwards: automated
qualification and publication consume the same immutable artifact. Its
`manifest.json` records `speech_to_text` and `text_to_speech` as
`required-automated-qualification`, which is a requirement rather than a pass
claim. Candidate runs are first-attempt-only; after a failed candidate run,
dispatch a deliberate new first-attempt candidate instead of rerunning it.

`.github/workflows/bridge_qualification.yml` runs the real-model Qwen3-ASR and
Qwen3-TTS gates on `ubuntu-latest`. It downloads the candidate by immutable
artifact ID only after proving the exact successful first-attempt candidate
workflow/run/source identity and unique complete artifact inventory. It then
checks out the candidate source, downloads every model/projector/fixture with a
pinned SHA-256, installs Node.js 24 with that source's locked npm dependencies
and Playwright Chromium, verifies that the harness bytes match that source, and
runs the source's Node smokes with GitHub's hosted-runner markers present. Each
gate has a bounded timeout and terminates its full process group on timeout or
cancellation. Smoke children
receive only a narrow, non-secret environment allowlist; ambient
`LLAMA_WEBGPU_*`, token, secret, and credential variables cannot redirect a
gate or reach the browser process.

The harness:

- runs Qwen3-ASR across wasm32 and wasm64 in direct and worker modes, covering
  cold transcript, cancellation, warm reuse, and silence-hallucination
  rejection, and records those per-mode results in the attestation;
- runs Qwen3-TTS across wasm64 direct and worker modes, covering the full
  lifecycle and verifying each generated file is a readable PCM16 mono 24 kHz
  WAV with a recorded SHA-256, byte length, frame count, and non-silent
  peak/RMS waveform evidence;
- records per-phase and per-mode timing plus peak RSS, measured for each smoke
  through `/usr/bin/time` and normalized to bytes on both the Linux (kibibyte)
  and macOS (byte) `ru_maxrss` conventions; the value is the largest peak of
  any smoke run up to the end of each phase;
- pins every model, projector, and audio fixture SHA-256, and digests its own
  harness sources so the attestation names the code that produced it;
- leaves real-device playback, intelligibility, and speaker-reference fidelity
  explicitly `unproven`, because no gate listens to the generated audio;
- emits one canonical attestation bound to the candidate artifact ID, run ID,
  first attempt, workflow path, digest, and release provenance, and to the
  producing qualification run ID, first attempt, workflow path, and source SHA.

The automated qualification workflow performs the same provenance checks,
downloads and digest-verifies every pinned input, runs the harness against the
exact candidate artifact, and uploads one canonical `qualification-attestation`
artifact. The orchestrator advances publication only after that exact
first-attempt qualification run succeeds. No maintainer creates, transports, or
submits an attestation, and the qualification workflow holds no publication PAT.

`scripts/smoke/speech_to_text.mjs` and
`scripts/smoke/text_to_speech.mjs` remain runnable on their own while
iterating locally. The combined `scripts/release/qualification.mjs qualify` command is
reserved for the hosted workflow because it requires GitHub Actions and
`github-hosted` runner identity. Both invocations are in
[CONTRIBUTING.md](CONTRIBUTING.md#validate-outputs).

The default audio is Qwen's official English example fixture.
`scripts/smoke/speech_to_text_fixture.json` pins its URL, its SHA-256 and the exact
normalized transcript observed through the Web runtime, for both the speech
smoke and `scripts/release/qualification.mjs`, so an upstream fixture replacement fails
loudly. Use `--memory-mode wasm32` or `--memory-mode wasm64` to classify one
variant; the default validates
both. Diagnostics land in `speech-to-text-smoke-artifacts` and
`text-to-speech-smoke-artifacts`.

Do not commit downloaded GGUFs, Playwright screenshots, console logs, generated
`dist/` assets, or Emscripten build/cache directories.

## Publishing

Published, versioned artifacts are consumed from:

- `leehack/llama-web-bridge-assets`

Publish workflow:

- `.github/workflows/publish_assets.yml`

Candidate build workflow (the only workflow that builds publishable assets):

- `.github/workflows/bridge_candidate.yml`

Automated qualification workflow:

- `.github/workflows/bridge_qualification.yml`

Trigger modes:

- The bridge-owned orchestrator dispatches candidate build, automated
  qualification, and publication through GitHub's API using the repository
  owner's existing authorized identity; owner checks reject other dispatch
  actors. A successful candidate, qualification, or publication run triggers an
  immediate idempotent continuation scan. The daily schedule performs initial
  native-release discovery and repairs missed or delayed event delivery. The
  stage workflows are deliberately not reusable
  because reusable job environments execute in the caller repository's context.

The scheduled resolver keeps the current default-branch source SHA separate
from the newest main-line commit that changed governed runtime/build inputs.
Non-build orchestration/qualification/publication/CI workflow, verification,
test, and documentation-only commits therefore preserve verified pipeline
state. The candidate workflow and any runtime/build input change create a new
correlation and must produce a new automatically qualified candidate.

There is no push, tag, or ordinary-CI publication trigger. The schedule and
`workflow_run` events wake the orchestrator; only its exact `workflow_dispatch`
calls start a build, qualification, or publication stage.

Immutable releases must already be enabled on `leehack/llama-web-bridge-assets`
before a candidate is dispatched. `bridge_candidate.yml` requires
`assets_immutable_releases_enabled=true` and records that assertion in its
prequalification artifact. Publication downloads that exact run-owned record,
binds its explicit boolean assertion to the candidate fingerprint, run, source,
release, compiler, and native identities, and proves the real state itself through
`GET /repos/{owner}/{repo}/immutable-releases`, requiring the exact
`{enabled, enforced_by_owner}` shape with `enabled` explicitly boolean `true`,
before it pushes anything. That endpoint returns 404 both when immutable
releases are off and when it cannot be read, so every non-200, missing, false,
or non-boolean answer fails closed.

Every complete release state, including one found by a retry, is read back by
tag and by release ID. Both reads must bind the exact tag, release ID, tag
commit, and published state and must report `immutable` as explicit boolean
`true`. Publication then requires
`gh release verify <tag> --repo leehack/llama-web-bridge-assets --format json`
to prove GitHub's signed release attestation: predicate type
`https://in-toto.io/attestation/release/v0.2`, signer
`https://dotcom.releases.github.com`, predicate database ID equal to the live
readback release ID, the `pkg:github/<repo>@<tag>` subject bound to the resolved
tag commit, and a SHA-256 subject for every published artifact matching the
exact candidate bytes. A mismatch is reported as a
non-retryable `immutable-publication-unverified` outcome and the release is left
untouched. An incomplete published release is not repaired by uploading missing
assets; publication never deletes, retags, overwrites, or repairs it.

Required externally configured credentials:

- `WEBGPU_BRIDGE_ASSETS_PAT` (read access to provenance repositories and write
  access to `leehack/llama-web-bridge-assets`, stored only in the publication
  environment). This is the only externally configured publication credential.

Every request supplies a required orchestrator correlation ID, the exact bridge
source SHA, upstream llama.cpp tag/commit, native release tag plus `assets.json`
SHA-256, output release tag/rebuild, a distinct required `candidate_run_id` and
`qualification_run_id`, and an exact assertion of the fixed assets repository.
Dispatch values enter shell scripts only through environment variables and
quoted expansions. The orchestrator sets `publish_approved=true` only after its
live publication-environment policy check succeeds; no human continuation is
part of the release pipeline.

Publication is deliberately blocked until repository administrators externally
configure `bridge-assets-publication` with administrator bypass disabled,
restrict exactly one custom deployment branch policy to `main`, and store
`WEBGPU_BRIDGE_ASSETS_PAT` as an environment-scoped secret. The solo-maintainer
publication contract does not require a reviewer rule or two-person quorum.
Merging this code does not establish the external settings.
The workflow uses `github.token` to validate the environment identity,
administrator-bypass setting, and exact `main` branch policy before entering
the privileged job and again immediately before the first publication-PAT-bearing
step. Attestation, provenance, and publication-state
policy validators come from the trusted workflow commit on `main`; the requested
historical bridge source supplies only exact candidate harness bytes, toolchain
pin, and build identity. Before any network use of the injected publication PAT,
each PAT-bearing step fails closed unless the credential is non-empty and does
not print its value. Missing or weakened bypass, branch, credential, or PAT
guard protection therefore fails closed.

The workflow verifies that bridge source is already merged, all upstream/native
tag and commit identities match, the native manifest checksum matches, and the
new release advances its own channel history without collision or rollback.
Stable releases compare only with stable history; development releases and
rebuilds compare only with the development history and upstream line.

The publication workflow never builds. `bridge_candidate.yml` built the exact
wasm32 and memory64 artifacts once with `emsdk.version` and ran the hosted state
and multimodal gates; publication downloads that exact artifact from
`candidate_run_id` and the attestation from `qualification_run_id`, after proving
for both runs that they are successful `workflow_dispatch` runs of the expected
workflow file in this repository, on the default-branch line, exposing exactly
one live artifact of the expected name in each complete inventory. Both are
downloaded by the validated immutable artifact IDs. It then verifies the canonical
attestation against the candidate fingerprint, bridge/upstream/native identity,
compiler, release tag/rebuild, correlation ID, candidate run ID, harness source
digest, every required gate, every required ASR/TTS memory and runtime mode, and
the producing qualification workflow path, run ID, first attempt, and source
SHA -- once before entering the privileged job and again inside it.

The published schema-v2 manifest carries the release tag, capabilities,
bridge/upstream/native commits, orchestrator correlation ID, the candidate run
ID/URL, per-artifact SHA-256 values, `qualification_gates` recording
`state_persistence` and `multimodal` as `passed` with `speech_to_text` and
`text_to_speech` as `required-automated-qualification`, and `unproven_capabilities`
recording real-device playback, intelligibility, and speaker-reference fidelity
as `unproven`. Those two gate values state the requirement rather than a hosted
pass that never happened; publication is what enforces it, refusing to publish
the artifact without a verified attestation bound to its exact digest. The
manifest is byte-deterministic for the same inputs and artifacts. Exact ref-only
partial states are recovered only after branch, tag, release metadata,
provenance, and asset-digest checks. An incomplete published release is an
immutable mismatch, not a repair path; mismatches fail closed with a
machine-readable JSON outcome.
If a credentialed ref or release mutation is followed by an unavailable or
empty re-query, the workflow emits a durable, retryable `mutation-unknown`
outcome with `mutated: null`; it never guesses that publication did or did not
occur.
Every successful candidate run uploads `bridge-candidate-prequalification`, recording its
hosted gate conclusions alongside `pending-automated-qualification` for the two
heavy gates. Every publication run uploads `bridge-qualification-outcome` with
the correlation ID, candidate and qualification run IDs, and candidate
fingerprint. The candidate run ID is part of the immutable candidate
fingerprint, so a partial publication is recovered by redispatching a new
publication workflow run using the same `candidate_run_id` and
`qualification_run_id`; never rerun the old run. A different candidate must never
overwrite an earlier one under the same output tag.

New bridge asset release tags are npm-shaped `vMAJOR.MINOR.PATCH` with
`release_rebuild` `0`. When the next version is already published or claimed by
an unfinished pipeline, the orchestrator takes the next free patch version; it
never appends `-N`, which npm orders as a prerelease below the plain version.
A new tag is never lower than a published tag or a live claim; a claim released
without publication stops counting, so its version may be taken again or stay
unused. `scripts/release/contract.mjs validate-release`, which the candidate and publish
workflows run, rejects a new tag with a rebuild suffix.

Tags published before this rule, such as `v0.1.47-1`, and the shared native
forms (`vMAJOR.MINOR.PATCH-N`, `bNNNN`, `bNNNN-N`) stay readable. Historical
`bNNNN-llamadart.N` and earlier wrapper forms are accepted only when reading
existing manifests and are never emitted.

After publish, assets are CDN-available at:

- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_bridge.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_bridge_worker.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_bridge.d.ts`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_core.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_core.wasm`

When `WEBGPU_BRIDGE_BUILD_MEM64=1`, the assets repo also publishes optional
memory64 core artifacts:

- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_core_mem64.js`
- `https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_core_mem64.wasm`

Note: CDN pinning fundamentally relies on git tags in the assets repo.

## Maintainer Docs

- `CONTRIBUTING.md`: setup, build, the runnable smoke invocations and their
  pins, workflow guardrails, and the publish process
- `AGENTS.md`: agent PR workflow, orchestrator invariants stated nowhere else,
  the `js/src/`-source-vs-`js/`-generated rule, cross-repo handoff, and
  regression smoke guidance
