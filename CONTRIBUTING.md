# Contributing to llama-web-bridge

Thanks for contributing.

## Purpose

This repo builds the reusable JS/WASM bridge runtime for llama.cpp web usage.
Published artifacts are consumed from `llama-web-bridge-assets`.

## Prerequisites

- Emscripten SDK (`emcmake`, `emcc`) matching `emsdk.version`
- Node.js 22.18 or newer (CI uses 24) and npm, for JS bridge bundling and
  type-checking; the tests run the `.ts` sources through Node's built-in type
  stripping
- CMake toolchain
- A host C++17 compiler (`c++`, or `$CXX`): `npm run check:js` compiles the
  media-helper compatibility contract with it
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

`./scripts/build_bridge.sh --help` is the complete list of environment
variables the build reads, with their defaults; the docs do not repeat the
list.

Bridge wrapper source lives under `js/src/`; `npm run build:js` regenerates the
checked-in browser ESM outputs and declarations under `js/`. `npm run check:js`
runs the same generator plus TypeScript and syntax checks, so commit any updated
`js/` outputs after source changes. It ends with `npm test`, which runs every
`tests/**/*_test.mjs` contract test in parallel through Node's test runner,
including the static state-persistence, text-to-speech, and decision API
contracts, the media-helper compatibility contract, the wasm64 runtime patch
contract for `scripts/patch_wasm64_runtime.mjs`, and the CI change selector.
Run one test file directly with `node tests/js/<name>_test.mjs`.

`js/src/llama_webgpu_bridge.js` is the public entry. It re-exports the API and
owns the only load-time side effects (worker host auto-boot and the
`window.LlamaWebGpuBridge` global); every other module is side-effect free.
`bridge.ts` is the facade, `runtime.ts` the direct runtime, `worker_proxy.ts`,
`worker_host.ts`, and `worker_protocol.ts` the worker path, and `internal/`
holds shared helpers, with internal-only types in `internal/types.ts`.

Every module except the two entries (`llama_webgpu_bridge.js` and
`llama_webgpu_bridge_worker.js`, which is copied unbundled) is TypeScript,
type-checked with `strict` (`tsconfig.strict.json`). The entries keep the
lenient `checkJs` pass (`tsconfig.bridge.json`), and `npm run typecheck:js` runs
both. `LlamaWebGpuBridge` implements the published `llama_webgpu_bridge.d.ts`
class, and `public_api_check.ts` (types only, never bundled) compares every
public method with strict parameter variance, so the compiler rejects an
implementation that accepts less or returns more than the declared API.
TypeScript here is limited to erasable syntax (`erasableSyntaxOnly`): types,
`import type`, `declare` fields, and casts only, no enums, namespaces, or
parameter properties. A class field without an initializer is `declare`d,
because a plain field declaration emits code;
`tests/js/declared_class_fields_test.mjs` enforces that. esbuild and Node both
strip the types without changing the code, so the tests run the `.ts` sources
directly. A type change must not change behaviour: with comments and whitespace
stripped, the bundle stays byte-identical. Import modules by their `.ts` path.

Keep `bridge.ts` beside `llama_webgpu_bridge_worker.js`: it resolves the worker
entry relative to `import.meta.url`.

The native core is one translation unit. `src/llama_webgpu_core.cpp` holds the
headers, the anonymous namespace, the `extern "C"` block with
`llamadart_webgpu_shutdown`, and `main`, and includes its parts from `src/core/`
inside the namespace and the block. The `exports_*.inc` parts hold the other
exported `llamadart_webgpu_*` functions grouped by feature; the remaining parts
hold the state and internal helpers they use. A part is not a standalone file:
it relies on everything included before it, so keep the include order. The
static contract checks read the core with its parts expanded
(`tests/js/native_core_source.mjs`), which is how the compiler sees it. It fails
if a part is not included exactly once as a plain `#include` line. The JS API
contract tests read the bridge the same way: `tests/js/bridge_js_source.mjs`
joins every `js/src` module in the order the former single-file source declared
them.

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

Expected files in `OUT_DIR`:

- `llama_webgpu_bridge.js`
- `llama_webgpu_bridge_worker.js`
- `llama_webgpu_bridge.d.ts`
- `llama_webgpu_core.js`
- `llama_webgpu_core.wasm`
- `llama_webgpu_core_mem64.js` (only with `WEBGPU_BRIDGE_BUILD_MEM64=1`, as
  above and in CI)
- `llama_webgpu_core_mem64.wasm` (same condition)

This file owns the runnable smoke invocations and the model/projector pins
they carry; `README.md` and `AGENTS.md` link here instead of repeating them.

Before opening or updating a PR, run the lightweight contracts:

```bash
npm run check:js
python3 -m unittest discover -s scripts -p '*_test.py'
node scripts/verify_ci_reliability.mjs
```

Every browser smoke runs on Node with the locked `playwright` dev dependency:
in CI, in the candidate's state and multimodal gates, and in the automated
qualification's speech and text-to-speech gates. After
`npm ci --ignore-scripts`, install its browser once with
`npx --no-install playwright install --only-shell chromium`. Node's `fetch`
ignores `HTTP(S)_PROXY` unless `NODE_USE_ENV_PROXY=1` is set.

For state-persistence, worker, or workflow changes, also run the browser smoke
against a built dist directory. Use a checksum-pinned tiny model and keep caches
and artifacts outside the repository:

```bash
node scripts/state_persistence_browser_smoke.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-url https://huggingface.co/aladar/llama-2-tiny-random-GGUF/resolve/main/llama-2-tiny-random.gguf \
  --model-sha256 81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb \
  --model-cache-dir ~/.cache/llama-web-bridge/state-smoke-models \
  --artifacts-dir /tmp/llama-web-bridge-state-smoke
```

For llama.cpp pin or multimodal changes, run checksum-pinned real image
inference through both direct and worker runtimes:

```bash
node scripts/multimodal_browser_smoke.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/Qwen3.5-0.8B-Q4_K_M.gguf \
  --model-sha256 bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517 \
  --mmproj-path /path/to/mmproj-F16.gguf \
  --mmproj-sha256 56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453 \
  --artifacts-dir /tmp/llama-web-bridge-multimodal-smoke
```

For sampler or grammar changes, run grammar-constrained completion, the
seeded `minP`/`presencePenalty` checks and the greedy `thinkingBudget` checks
through direct and worker runtimes on both memory modes. CI runs it twice: with
the state-persistence model, then with the multimodal model. Pass each model's
`--model-url` (or `--model-path`) and `--model-sha256` pin from the commands
above; the smoke defaults to the `LLAMA_WEBGPU_SMOKE_MODEL_URL` and
`LLAMA_WEBGPU_SMOKE_MODEL_SHA256` environment variables:

```bash
node scripts/grammar_browser_smoke.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-url "$LLAMA_WEBGPU_SMOKE_MODEL_URL" \
  --model-sha256 "$LLAMA_WEBGPU_SMOKE_MODEL_SHA256" \
  --artifacts-dir /tmp/llama-web-bridge-grammar-smoke
```

For next-token scoring changes, run `scoreNextToken` through direct and worker
runtimes on both memory modes with the state-persistence model:

```bash
node scripts/next_token_scores_browser_smoke.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-url "$LLAMA_WEBGPU_SMOKE_MODEL_URL" \
  --model-sha256 "$LLAMA_WEBGPU_SMOKE_MODEL_SHA256" \
  --artifacts-dir /tmp/llama-web-bridge-next-token-scores-smoke
```

For speculative decoding changes, run greedy completions with and without each
strategy through direct and worker runtimes on both memory modes. The `tiny`
group runs the n-gram strategies and `draft-simple` on the state-persistence
model; `--group all` adds real-weight groups for every strategy, read from
`--models-dir` at the paths in the script's `FILES` table. The EAGLE3 and DFlash
drafts and the n-gram cache have no download URL, so place them there first:

```bash
node scripts/speculative_browser_smoke.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --group all \
  --models-dir ~/.cache/llama-web-bridge/speculative-smoke-models \
  --artifacts-dir /tmp/llama-web-bridge-speculative-smoke
```

Heavy Qwen3-ASR and Qwen3-TTS gates run in the hosted automated
qualification workflow, not in ordinary CI or the candidate build. Before
publishing, `.github/workflows/bridge_qualification.yml` runs them against the
exact candidate artifact built by `.github/workflows/bridge_candidate.yml`.
The workflow proves the candidate's workflow/run/source identity and unique
artifact ID, verifies every downloaded input checksum, installs Node.js 24 with
the candidate source's locked npm dependencies and Playwright Chromium, requires
GitHub Actions `github-hosted` runner identity, and emits one canonical
attestation. That attestation binds the candidate artifact ID/run/attempt/workflow/digest and the
producing qualification run ID/attempt/workflow/source SHA. The combined
`release_qualification.py qualify` command is therefore workflow-only.

For local reproduction, run the individual smokes directly:

```bash
node scripts/speech_to_text_browser_smoke.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/Qwen3-ASR-0.6B-Q8_0.gguf \
  --model-sha256 bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971 \
  --mmproj-path /path/to/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --mmproj-sha256 41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d \
  --artifacts-dir /tmp/llama-web-bridge-speech-smoke
```

```bash
node scripts/text_to_speech_browser_smoke.mjs \
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

For decision-head changes, compare a Laya encoder and head with the Laya
reference fixture (llamadart's
`test/fixtures/decision/laya_0_3_5_reference.json`) through both runtimes and
memory modes. Pass `--config-path` with `rl_agent_config.json` for a head
without `laya.config` metadata, such as the official `model.safetensors`, and
`--gpu-layers 0` to check the CPU path:

```bash
node scripts/decision_browser_smoke.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/laya-Q8_0.gguf \
  --head-path /path/to/laya-head.safetensors \
  --fixture-path /path/to/laya_0_3_5_reference.json \
  --gpu-layers 99 \
  --artifacts-dir /tmp/llama-web-bridge-decision-smoke
```

For LoRA adapter changes, run a base model and a LoRA adapter trained on it
through both runtimes and memory modes. The smoke defaults to a checksum-pinned
stories15M base and Shakespeare adapter and loads the adapter onto the
state-persistence model to check the mismatch error. The default
`--gpu-layers 0` runs on the CPU. A nonzero value enables WebGPU and skips the
model reload check, because reloading a model on one bridge aborts in headless
Chromium with WebGPU enabled, with or without adapters:

```bash
node scripts/lora_adapter_browser_smoke.mjs \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --mismatch-model-url "$LLAMA_WEBGPU_SMOKE_MODEL_URL" \
  --mismatch-model-sha256 "$LLAMA_WEBGPU_SMOKE_MODEL_SHA256" \
  --artifacts-dir /tmp/llama-web-bridge-lora-smoke
```

If the smoke downloads from a URL, errors and diagnostics must redact userinfo,
query strings, and fragments before printing the location.

## Agent Workflow Guardrails

- Keep the publication-safety rules in `scripts/verify_ci_reliability.mjs`
  current when changing `.github/workflows/ci.yml`,
  `.github/workflows/bridge_candidate.yml`,
  `.github/workflows/publish_assets.yml`,
  `.github/workflows/auto_llama_cpp_update.yml`,
  `.github/workflows/bridge_qualification.yml`, or the model pins in
  `scripts/release_qualification.py`. It checks permissions, environment gates,
  PAT handling, pins, fail-closed guards, and that CI runs the contract tests;
  it does not check wording, step names it does not anchor on, or docs prose.
  A new test in `tests/js/` runs once it is named `*_test.mjs`; any other file
  under `tests/` must be a helper that a test imports.
- Rotate all 7 model/projector SHA-256 pins in the three files that hard-code
  them together: `CONTRIBUTING.md`, `.github/workflows/ci.yml`,
  `.github/workflows/bridge_candidate.yml`.
  `scripts/verify_ci_reliability.mjs` requires the three sets to be identical with
  exactly 7 pins each; a stale `bridge_candidate.yml` breaks the candidate job,
  not just CI. `README.md` and `AGENTS.md` hold no pins and link here.
  `publish_assets.yml` holds no pins because it neither builds nor
  smokes. `scripts/release_qualification.py` carries the same 7 plus the pinned
  ASR audio fixture, and every attestation must match them exactly.
  `.github/workflows/bridge_qualification.yml` hand-copies a 5-pin speech, TTS,
  and ASR audio subset, so rotate it with the rest.
  `scripts/speech_to_text_fixture.json` holds the same ASR audio URL and
  SHA-256, which the speech smoke uses as defaults, with the expected
  transcript that `scripts/release_qualification.py` also reads.
  `scripts/release_qualification_test.py` requires its SHA-256 to equal the
  pin; nothing compares its URL, so rotate that by hand.
- The script maps every workflow `<ROLE>_SHA256` env key to its canonical name in
  `EXPECTED_MODEL_PINS` and requires equality in all three workflows, so a role
  swap fails even when applied identically to every one of them. It requires
  every name in `EXPECTED_MODEL_PINS` to be bound by some env key across those
  three files, requires `bridge_qualification.yml` to declare exactly its five
  roles, and requires every role to declare both `<ROLE>_URL` and
  `<ROLE>_SHA256` in each file that mentions it -- every `LLAMA_WEBGPU_*_URL` key
  in those files counts as a pinned model download. It also requires each role's
  `<ROLE>_URL` to be byte-identical across the workflows that declare it, and it
  pairs each `--model-sha256` / `--mmproj-sha256` flag here with the
  `--model-url` / `--model-path` / `--mmproj-path` value above it, resolving that
  filename to the role whose workflow URL downloads the same name. The
  state-persistence and multimodal URLs resolve through the mutable
  `resolve/main` ref rather than an immutable 40-hex revision, and the ASR audio
  fixture is not a Hugging Face object and carries no revision segment at all;
  the script lists both sets and fails when a role joins or leaves them.
- Keep `scripts/multimodal_browser_smoke.mjs` in normal CI for every llama.cpp
  pin update; build-only validation does not cover mtmd prompt ingestion.
- Heavy real-model ASR and TTS gates run through
  `scripts/release_qualification.py` in automated qualification. Keep the
  candidate manifest honest: `generate_release_manifest.py` must record those
  two gates as `required-automated-qualification`, never as a candidate-build
  pass, and must
  keep real-device playback, intelligibility, and speaker-reference fidelity in
  `unproven_capabilities`.
- Nothing may rebuild the candidate after it is built. The manifest embeds the
  candidate run ID/URL, so a rebuild changes the digest and no attestation could
  match it. Candidate workflow attempts are first-attempt-only; after a failed
  build or hosted gate, dispatch a new candidate run instead of rerunning one.
- Preserve `llama_cpp.version` as the default ordinary CI/development build pin.
  It holds exactly one upstream tag in either channel, stable
  `vMAJOR.MINOR.PATCH` or development `bNNNN`; `scripts/verify_ci_reliability.mjs`
  rejects every other form. Exact release publication receives upstream identity
  from the orchestrator and must not require a bridge pin PR.
- Preserve `emsdk.version` as the single compiler source for CI and publish.
  Both workflows must verify the active `emcc` version, and published manifests
  must record that verified identity.
- Main-branch and PR CI never dispatch publication. Stable release discovery
  prepares an ordered `release-candidates.json` backlog for every
  stable native release after the immutable native `v0.2.0-1` / Web-assets
  `v0.1.39` baseline and invokes `scripts/release/orchestrator/cli.mjs` to
  idempotently advance candidate, automated qualification, and publication
  stages. Each successful stage wakes an immediate `workflow_run` continuation;
  the daily schedule discovers new native releases and is the idempotent repair
  fallback. The scan downloads
  every selected `assets.json` and `SHA256SUMS` by
  unique release-asset ID and runs the complete native release/tag/inventory
  contract first. Each exact pipeline advances by at most one stage per scan,
  and the complete backlog is visited so an older in-flight gate does not starve
  a newer candidate. Publication remains ordered by native release time.
  An entry without its own exact publication is `superseded` when a published
  stable asset release's `Native:` marker names a newer native release (version,
  then rebuild); the newest native release in a scan is `blocked` instead.
  A proven candidate whose publication files other than `manifest.json` are
  byte-identical to the newest published release for the same native release
  and native manifest digest is `satisfied_by_identical_release`: nothing
  further is dispatched and its output tag is released to later pipelines in
  the same scan. The comparison runs only where a qualification or publication
  would otherwise be dispatched, never while one of the correlation's runs is
  in flight, and never across native alignments.
  Manual `development` scans stay scan-only: they resolve exact `bNNNN`
  provenance and report it, and the orchestrator refuses non-stable provenance.
  A failed candidate or qualification is never retried automatically; after diagnosis,
  a maintainer may deliberately dispatch one new first-attempt run with the same
  exact binding, and a later unique success supersedes the recorded failed run.
  On the normal path, a successful candidate is immediately followed by hosted
  ASR/TTS qualification, then by publication after the exact attestation is
  verified. Run recovery paginates each exact workflow's filtered
  history with stable-count checks and splits searches at GitHub's 1,000-result
  cap into closed time windows, so an older waiting pipeline is not lost as
  later workflow history grows.
- The full-history scheduled resolver records both the exact default-branch
  source SHA that will execute a new candidate and the newest first-parent
  commit that changed governed runtime/build inputs. Correlation uses the latter:
  non-build orchestration/qualification/publication/CI workflow, verification,
  test, and documentation-only commits preserve a verified terminal release.
  The candidate workflow, manifest/release-contract/compiler inputs, and every
  other new path are governed by default and require a new candidate plus
  automated qualification. Bridge source changes still
  require ordinary PRs; release orchestration never changes `llama_cpp.version`.
- Preserve `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` in CI and publish workflows so
  GitHub Action runtime changes are detected before they become mandatory.
- Upload state-persistence smoke diagnostics only on failure; successful CI runs
  should stay quiet beyond the normal build artifacts.
- Do not push branches, tags, or publish assets from local agent work unless the
  maintainer explicitly requests that side effect.

## Publish Process

Use `.github/workflows/publish_assets.yml` only after the publication approval
assertion is backed by the live `bridge-assets-publication` policy. The
bridge-owned orchestrator dispatches this
bridge-owned, non-reusable workflow with a required correlation ID, exact bridge SHA,
upstream tag/commit, native release tag plus `assets.json` SHA-256,
output release tag/rebuild, distinct required `candidate_run_id` and
`qualification_run_id`, and assets repository.

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

The orchestrator must set `publish_approved=true` only after live policy
verification. Publication remains blocked until
repository administrators separately create `bridge-assets-publication`, disable
administrator bypass, restrict the custom deployment branch policy to `main`,
and store the assets PAT as an environment-scoped secret. The solo-maintainer
publication contract does not require a reviewer rule, `prevent_self_review`, or
a two-person approval quorum. The workflow uses the default job token to verify
the environment identity, disabled administrator bypass, and exact `main`
deployment branch policy before entering the privileged job. It repeats those
checks inside that job immediately before the first publication-PAT-bearing
step, using the trusted
workflow commit's validator so an older requested build-source SHA remains
compatible. The trusted commit also supplies attestation, provenance, and
publication-state policy; the historical source supplies only the exact harness
bytes, toolchain pin, and build identity. The environment-scoped
`WEBGPU_BRIDGE_ASSETS_PAT` is the only
external credential; each step that can use it fails closed unless the injected
value is non-empty and never prints the value. The workflow also verifies all
identities plus native GitHub asset digests/inventory. It never builds: it
proves the repository, workflow file, dispatch event, default-branch-line head,
success, complete artifact inventory, and uniqueness of both the candidate and
qualification runs, downloads both by immutable artifact ID, and verifies the
attestation against the candidate and its producing qualification run twice --
once before entering the privileged job and again inside it. It then orders stable and development histories
independently and recovers exact ref-only partial states while rejecting any
published-release mismatch rather than trying to repair it.
Manifest and outcome records carry the candidate run ID/URL, because that is the
identity the candidate manifest embeds; unavailable post-mutation re-queries
produce retryable `mutation-unknown` records rather than guessed state.
Publication run attempts are first-attempt-only: retry partial publication by
redispatching a new publication run against the same `candidate_run_id` and
`qualification_run_id`, never rerunning the old run. The fingerprinted identity
comes from the candidate run, so it stays stable. A different candidate must
never target an existing output tag.

The event-driven orchestrator and its scheduled repair fallback reuse that same existing secret only from a
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
default-branch schedule events remain automatic; `workflow_run` continuations
also require a successful stage run on the default branch with owner actor and
triggering actor before either job starts.

New bridge asset tags are npm-shaped `vMAJOR.MINOR.PATCH` with rebuild `0`:
`select_next_release_target` skips a published or claimed version by taking
the next free patch version, and `release_contract.py validate-release` rejects
a new `-N` tag in the candidate and publish workflows, because npm orders
`vMAJOR.MINOR.PATCH-N` as a prerelease. Earlier `-N` bridge tags, the native
forms `vMAJOR.MINOR.PATCH-N`, `bNNNN` and `bNNNN-N`, historical
`bNNNN-llamadart.N` and prior wrapper forms are read-only compatibility inputs.

Never interpolate `${{ inputs.* }}` directly inside a workflow `run` script.
Transport dispatch inputs through `env` and use quoted shell expansions.

## Repository Boundaries

- Bridge runtime source/build belongs here.
- Versioned static artifacts belong in `llama-web-bridge-assets`.
- Consumer integration (loading/fallback behavior) belongs in `llamadart`.

## CI change selection and compiler cache

CI always runs the shared JS and workflow contracts. An explicit allowlist in
`scripts/ci_scope.mjs` lets known documentation and tooling-only changes avoid the
WASM builds. Runtime JS, C++, browser harnesses, build inputs, workflows, pins,
and unknown paths retain the pinned build/smoke lane. Rename and
deletion comparisons include both paths. The `CI validation` result always
reports and rejects failed, cancelled, missing, or unexpectedly skipped work.
Only superseded PR runs are cancelled; main/manual runs remain independent.

The CI compiler cache stores objects outside the checkout, separated by runner
OS/architecture, exact Emscripten version, resolved llama.cpp commit, and build
script/CMake/patch inputs. ccache also checks compiler contents, source inputs,
and compile flags. Every selected build still links fresh artifacts and runs
every CI browser smoke. Candidate and publication workflows do not consume
this cache or this change selector, so `scripts/ci_scope.mjs` is listed in
`ORCHESTRATION_ONLY_PATHS` in `scripts/release/orchestrator/cli.mjs`. List any
new CI-only script there too: an unclassified path is governed by default and
advances the release build identity. Track follow-up work in
[llamadart issue #532](https://github.com/leehack/llamadart/issues/532).
