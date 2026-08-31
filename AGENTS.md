# AGENTS.md

Guidance for coding agents working in `llama-web-bridge`.

## Scope and Ownership

- This repository owns WebGPU bridge source/runtime build for llama.cpp web.
- It publishes versioned assets to `llama-web-bridge-assets` via workflow.
- `llamadart` consumes those published assets.

## Related Repositories

Common maintainer sibling layout:

```text
../llamadart
../llamadart-native
../llama-web-bridge
../llama-web-bridge-assets
```

## Build Commands

```bash
./scripts/build_bridge.sh
```

Useful environment overrides:

- `LLAMA_CPP_DIR` (defaults to `third_party/llama_cpp`; CI clones the tag from
  `llama_cpp.version`)
- The local Emscripten SDK must match the exact release in `emsdk.version`.
- `BUILD_DIR`
- `OUT_DIR`
- `CMAKE_BUILD_TYPE`

## Agent PR Workflow

For non-trivial runtime, workflow, or API changes, keep the PR path explicit:

1. Start from a clean topic branch and inspect `git status` before editing.
2. Add or update a regression/contract check before changing behavior when
   practical. Static contract scripts are acceptable for workflow invariants.
3. Keep Emscripten build directories, ccache, model caches, and Playwright
   artifacts outside the repository unless they are intentionally versioned.
4. Run the targeted checks in this file and the full browser smoke when the
   change touches `js/`, `src/`, `scripts/`, or GitHub workflows.
5. Use an independent review before committing PR-bound changes. Fix blocking
   findings, rerun the targeted checks, then commit locally; do not push or open
   a PR unless the maintainer asks.

### Local Verification Notes

When validating bridge runtime changes locally, keep build/cache output outside
the repo so generated wasm artifacts and toolchain caches do not dirty the
checkout or hit sandboxed Homebrew/cache paths:

```bash
export CCACHE_DIR=/private/tmp/llama_web_bridge_ccache
export EM_CACHE=/private/tmp/llama_web_bridge_emcache
BUILD_DIR=/private/tmp/llama_web_bridge_build MEM64_BUILD_DIR=/private/tmp/llama_web_bridge_build_mem64 OUT_DIR=/private/tmp/llama_web_bridge_dist WEBGPU_BRIDGE_BUILD_MEM64=1 ./scripts/build_bridge.sh
```

Minimum local checks before handing off a PR-ready branch:

```bash
npm run check:js
python3 -m py_compile scripts/verify_state_persistence_api.py scripts/verify_text_to_speech_api.py scripts/verify_ci_reliability.py scripts/state_persistence_browser_smoke.py scripts/multimodal_browser_smoke.py scripts/speech_to_text_browser_smoke.py scripts/text_to_speech_browser_smoke.py
python3 scripts/verify_state_persistence_api.py
python3 scripts/verify_text_to_speech_api.py
python3 scripts/verify_ci_reliability.py
```

For state-persistence or workflow changes, also run the browser smoke against a
built `OUT_DIR`. Keep the tiny model in a user cache or `/private/tmp`; do not
commit downloaded GGUFs or smoke artifacts:

```bash
python3 -m pip install --user playwright
python3 -m playwright install chromium
python3 scripts/state_persistence_browser_smoke.py \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-url https://huggingface.co/aladar/llama-2-tiny-random-GGUF/resolve/main/llama-2-tiny-random.gguf \
  --model-sha256 81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb \
  --model-cache-dir ~/.cache/llama-web-bridge/state-smoke-models \
  --artifacts-dir /private/tmp/llama_web_bridge_state_smoke_artifacts
```

For llama.cpp pin or multimodal changes, also run real image inference through
both direct and worker runtimes:

```bash
python3 scripts/multimodal_browser_smoke.py \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/Qwen3.5-0.8B-Q4_K_M.gguf \
  --model-sha256 bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517 \
  --mmproj-path /path/to/mmproj-F16.gguf \
  --mmproj-sha256 56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453 \
  --artifacts-dir /private/tmp/llama_web_bridge_multimodal_smoke_artifacts
```

Heavy Qwen3-ASR and Qwen3-TTS gates run in the hosted automated qualification
workflow, not in ordinary CI or the candidate build. For local reproduction
against the exact artifact produced by `.github/workflows/bridge_candidate.yml`:

```bash
python3 scripts/release_qualification.py qualify \
  --candidate-run-id <CANDIDATE_RUN_ID> \
  --speech-model-path /path/to/Qwen3-ASR-0.6B-Q8_0.gguf \
  --speech-mmproj-path /path/to/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --speech-audio-path /path/to/asr_en.wav \
  --tts-model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf \
  --tts-mmproj-path /path/to/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --output-attestation /private/tmp/qualification-attestation.json \
  --output-base64 /private/tmp/qualification-attestation.b64
```

Every model, projector, and fixture path is required. Smoke children receive a
narrow non-secret environment allowlist, each gate has a bounded timeout, and
the full process group is terminated on timeout or cancellation. The command
never accepts an unprovenanced local dist and rejects local harness bytes that
do not match the exact candidate bridge commit. The release path does
not accept this local payload: the orchestrator dispatches
`.github/workflows/bridge_qualification.yml`, and publication accepts only its
exact successful first-attempt run and artifact.

Run individual smokes when testing isolated changes:

```bash
python3 scripts/speech_to_text_browser_smoke.py \
  --dist-dir /private/tmp/llama_web_bridge_dist \
  --model-path /path/to/Qwen3-ASR-0.6B-Q8_0.gguf \
  --model-sha256 bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971 \
  --mmproj-path /path/to/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --mmproj-sha256 41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d
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
  --gpu-layers 99
```

The candidate manifest records `speech_to_text` and `text_to_speech` as
`required-automated-qualification`, never as a candidate-build pass.
Publication fails closed unless a verified attestation from an exact successful
automated qualification run binds the candidate digest it is about to publish.

## CI / Release

- CI build gate: `.github/workflows/ci.yml`
  - Resolves the default llama.cpp checkout from `llama_cpp.version`.
  - Resolves `emsdk.version`, installs that exact compiler, verifies the active
    `emcc` identity, and contract-tests all five required wasm64 WASMFS patches.
  - Never dispatches asset publication. Bridge source changes, including changes
    to the default development pin, use ordinary PRs and ordinary CI.
- Candidate build: `.github/workflows/bridge_candidate.yml`
  - The only workflow that builds publishable assets. It builds wasm32 and
    memory64 once, runs the compiler, contract, state-persistence, and
    multimodal gates it can afford, generates the schema-v2 manifest against its
    own run ID/URL, and uploads `exact-webgpu-bridge-dist` plus an honest
    `bridge-candidate-prequalification` record.
  - Holds no publication environment and no PAT. Nothing rebuilds its artifact:
    a rebuild would change the manifest, and therefore the digest, so no
    attestation could ever match what is published. The workflow refuses
    `github.run_attempt != 1`; dispatch a new candidate after failure rather than
    rerunning it.
  - Requires `assets_immutable_releases_enabled=true`. The candidate holds no
    credential that can read another repository's administration settings, so it
    cannot call the governance API itself: it fails closed on the dispatcher's
    assertion and records it in `bridge-candidate-prequalification`, while
    `publish_assets.yml` downloads that exact run-owned record, binds its boolean
    assertion to the candidate identity, and independently proves the real state
    before publishing.
    Confirm the assertion first with
    `gh api repos/leehack/llama-web-bridge-assets/immutable-releases`.
- Automated qualification: `.github/workflows/bridge_qualification.yml`
  - Owner-dispatched by the orchestrator with the exact candidate run ID and
    correlation ID; no maintainer-supplied attestation input exists.
  - Proves the candidate run is a successful `bridge_candidate.yml`
    `workflow_dispatch` in this repository whose head is on the default-branch
    line whose complete inventory has exactly one live
    `exact-webgpu-bridge-dist` artifact, downloads it by immutable artifact ID,
    checks out the exact candidate source, verifies pinned model inputs, runs
    the hosted real-model gates, canonicality-checks the result, and uploads one
    `qualification-attestation` artifact using repository access and no PAT.
- Native-aligned candidate scan and publication orchestrator: `.github/workflows/auto_llama_cpp_update.yml`
  - Scheduled runs select every stable native release published after the
    immutable `v0.2.0-1` / Web-assets `v0.1.39` automation baseline. They
    download each release's `assets.json` and `SHA256SUMS` by unique GitHub
    Release asset ID, validate the exact release/tag/asset inventory, collect
    exact provenance in `release-candidates.json`, and idempotently advance each
    three-stage pipeline (Build Exact Bridge Candidate -> Qualify Exact Bridge
    Candidate -> Publish Exact Qualified Bridge Assets). Processing the whole
    ordered backlog prevents an older qualification wait from hiding or
    starving a newer stable native release; each exact pipeline advances by at
    most one stage per scheduled/manual scan.
  - Only the stable channel is orchestrated. Manual `development` scans still
    resolve and report exact `bNNNN` provenance, but `require_stable_provenance`
    refuses to advance them, so they never dispatch anything.
  - Every dispatch sends exactly the target workflow's declared `workflow_dispatch`
    inputs (`require_exact_dispatch_inputs`), at the exact default-branch `--ref`,
    after a live `immutable-releases` governance read; a duplicate-run check runs
    before dispatch and a run-name readback runs after it.
  - Run recovery queries the exact workflow with server-side owner, event,
    default-branch, and relevant-publication-time filters, then accepts only an
    exact supported workflow path plus its deterministic `display_title`, owner
    actor/triggering actor, first attempt, repository, and branch. GitHub exposes
    a workflow's rendered `run-name` in the run record's `name` field, so that
    field is not treated as the static workflow identity. A
    multi-page query must retain a stable filtered count. A search at GitHub's
    1,000-result cap is split into closed time windows until every relevant page
    is complete; a saturated one-second window or ambiguous result fails closed
    without depending on a repository-wide history count.
  - The dispatch job reuses the existing `bridge-assets-publication` environment
    and its environment-scoped `WEBGPU_BRIDGE_ASSETS_PAT`; there is no separate
    orchestrator secret. Automatic dispatch requires that owner credential to
    retain assets-governance read access and Actions write access on
    `leehack/llama-web-bridge`. Missing capability is reported as blocked. The
    planner supplies neither `publish_approved=true` nor
    `assets_immutable_releases_enabled=true`; those values are added only after
    the corresponding live environment/governance proofs.
  - A manual orchestration run is admitted only when both `github.actor` and
    `github.triggering_actor` are the repository owner. The gate applies to both
    jobs before the publication environment can expose its PAT; trusted
    default-branch schedule events remain automatic.
  - A failed candidate run is terminal for that exact native provenance, so the
    daily schedule cannot create unbounded duplicate candidates. After diagnosis,
    a maintainer may explicitly dispatch one deliberate new first-attempt run
    with the same exact binding. Publication retries may reuse only the exact
    successful candidate and attestation runs.
  - A successful candidate waits for maintainer-run local ASR/TTS qualification
    and ingestion without blocking candidate creation for later backlog entries.
    After that external ingestion succeeds, the next daily or manual stable scan
    dispatches publication; no hosted run fabricates or weakens the attestation.
  - An already-published noop independently resolves the assets tag commit,
    validates release reads by tag and ID, downloads and hashes the exact asset
    inventory, and validates `gh release verify --format json` with the same
    immutable-release and release-attestation contracts as publication.
  - It never changes `llama_cpp.version`, opens a PR, tags, or pushes directly.
- CI reliability contract: `scripts/verify_ci_reliability.py`
  - Keep this script updated when changing browser smoke behavior, action
    versions, JS build/type-checking, or workflow diagnostics.
  - Requires the 7 model/projector SHA-256 pins to be one identical set across
    the five files that hard-code them -- `README.md`, `AGENTS.md`,
    `CONTRIBUTING.md`, `ci.yml`, and `bridge_candidate.yml` -- and
    role-consistent between the two workflows. CONTRIBUTING.md owns the rotation
    rules and the residual gaps.
  - CI, candidate, and publish must run `npm run check:js`, which regenerates the checked-in
    generated bridge wrapper outputs and declarations, and then fail with
    `git diff --exit-code` if those generated outputs are stale.
  - The CI smoke must use a pinned tiny GGUF URL plus SHA-256, cache the model in
    the same expanded `~/.cache/llama-web-bridge/state-smoke-models` directory
    used by `actions/cache`, and upload `state-persistence-smoke-artifacts` on
    failure.
  - CI, candidate, ingestion, and publish workflows intentionally set
    `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` so action-runtime regressions are caught
    before Node 20 deprecation becomes a hard failure.
- Publish workflow: `.github/workflows/publish_assets.yml`
  - Is a bridge-owned, non-reusable manual workflow that the central orchestrator
    may dispatch through GitHub's API using the repository owner's existing
    authorized identity; other dispatch actors are rejected. This keeps the
    environment gate in the bridge repository rather than the caller context. It requires an orchestrator
    correlation ID, exact bridge source SHA, upstream tag/commit, native tag plus
    manifest SHA-256, output tag/rebuild, required distinct `candidate_run_id`
    and `attestation_run_id`, and assets repository inputs. It does not read
    `llama_cpp.version`.
  - Never builds. It downloads the exact candidate artifact from
    `candidate_run_id` and the attestation from `attestation_run_id` by immutable
    artifact ID after proving both runs' repository, workflow file, dispatch
    event, default-branch-line head, success, complete inventory, and artifact
    uniqueness.
  - Compares the candidate manifest's `emscripten_version` against the
    `emsdk.version` pin at the exact bridge source SHA.
  - Requires `publish_approved=true`. Publication remains blocked until an
    administrator externally creates `bridge-assets-publication`, disables
    administrator bypass, restricts custom deployment branches to `main`, and
    stores `WEBGPU_BRIDGE_ASSETS_PAT` as an environment-scoped secret. The
    solo-maintainer publication contract does not require a reviewer rule. Do
    not describe that environment as protected without current live evidence.
    Use the default job token to validate the environment identity,
    administrator-bypass setting, and exact `main` branch policy before approval
    and again after approval. Use the trusted workflow commit's validator rather
    than the requested historical bridge build source. Immediately before any
    network use of the environment-scoped publication PAT, fail closed unless
    the injected credential is non-empty, without printing its value.
  - Emits only stable `vMAJOR.MINOR.PATCH[-N]` or development `bNNNN[-N]` tags.
    Historical `*-llamadart.N` forms are accepted only when reading old
    manifests and are never emitted.
  - Orders stable and development histories independently, while rejecting
    rollback or collisions within each channel, diverged bridge source, output
    identity/checksum mismatches, and unmerged bridge source commits.
  - Transports dispatch inputs through workflow environment variables; never
    embed `${{ inputs.* }}` directly in a shell `run` block.
  - Verifies the digest-bound attestation twice: once before approval and again
    inside the privileged job against the artifact it is about to publish. The
    attestation must match the candidate fingerprint, bridge/upstream/native
    identities, compiler, release tag/rebuild, correlation ID, candidate run ID,
    harness source digest, every required gate, and every required ASR/TTS
    memory and runtime mode.
  - Records the candidate run ID/URL as the publication identity, because that
    is what the candidate manifest embeds. Retry by redispatching against the
    same candidate and attestation runs.
  - Proves immutable-release governance on the assets repository through
    `GET /repos/{owner}/{repo}/immutable-releases` before any ref or release
    mutation, and requires the exact `{enabled, enforced_by_owner}` shape with
    `enabled` exactly boolean `true`. That endpoint answers 404 both when
    governance is disabled and when the credential cannot read it, so every
    non-200, missing, false, or non-boolean response fails closed.
  - Reads every complete release state, including a retry that found an existing
    exact release, back by tag and by release ID. Both reads must bind the exact
    tag, release ID, tag commit, published state, and explicit boolean
    `immutable: true`; publication then requires
    `gh release verify <tag> --repo <assets repo> --format json` to prove
    GitHub's signed release attestation: predicate type
    `https://in-toto.io/attestation/release/v0.2`, signer
    `https://dotcom.releases.github.com`, predicate database ID equal to the
    live readback release ID, the exact `pkg:github/<repo>@<tag>` subject bound
    to the resolved tag commit, and a SHA-256 subject for every published
    artifact matching the candidate bytes. A mismatch is reported as
    a non-retryable `immutable-publication-unverified` outcome. Publication
    never uploads into an incomplete published release and never deletes,
    retags, overwrites, or otherwise repairs the release.
  - Publishes schema-v2 provenance with release tag, capabilities, bridge,
    upstream, and native commits, exact run ID/URL, mandatory gate conclusions,
    correlation ID, and per-artifact SHA-256 values. An unreadable post-mutation
    remote state must emit retryable `mutation-unknown`, never guessed mutation
    state.
  - Any future npm package version has an independent monotonic sequence and
    uses stable/nightly dist-tags; GitHub `vM.m.p-N` tags must not be reused as
    npm versions because npm orders them as prereleases.

### Immutable Automation Baseline

The historical `v0.1.38` release is not repaired or reused. The verified
immutable baseline is:

- `release_tag`: `v0.1.39`
- `release_rebuild`: `0`
- `orchestrator_correlation_id`: `kanban:t_7f112b91:web-v0.1.39`
- `assets_immutable_releases_enabled`: `true`

Daily backlog selection starts after native `v0.2.0-1`, published at
`2026-08-25T08:57:12Z`. Every later stable native release gets its own new
candidate and exact `candidate_run_id`/`attestation_run_id` pair.

## Change Boundaries

- Keep runtime bridge source code in `js/src/` and `src/`; generated browser
  wrapper/declaration outputs live in `js/`.
- Keep publishing logic in workflow only.
- Do not edit assets repository files from here outside publish flow.

## Cross-Repo Handoff to `llamadart`

After publishing assets tag:

1. Update/fetch pinned bridge assets in `llamadart`:
   `WEBGPU_BRIDGE_ASSETS_TAG=<tag> ./scripts/fetch_webgpu_bridge_assets.sh`
2. Update docs/changelog in `llamadart` if behavior changed.

## Regression Smoke Guidance

- For pthread/runtime changes, test a BERT-class embedding model in Chromium
  with cross-origin isolation enabled. The regression shape is:
  `loadModelFromUrl`, `tokenize`, `embed`, and `embedBatch` on a host where
  `navigator.hardwareConcurrency` is greater than the bridge pthread pool size.
- Run the smoke through both direct runtime (`disableWorker: true`) and the
  bridge worker path; both should report `n_threads` capped to the pool size.
- For state persistence, exercise both direct and worker runtimes with a real
  tiny model. The smoke should evaluate a prompt, save bytes, mutate state,
  reload bytes, and verify generation still works after restore.
- Worker and direct runtime filesystems are separate. Do not silently fall back
  from worker-owned state APIs to direct runtime state; byte APIs are the durable
  app-storage path for IndexedDB/OPFS/Cache API integrations.
- If the smoke downloads a model, never expose raw signed/authenticated locations in
  thrown errors or artifacts. Redact userinfo, query, and fragment values.
- Every llama.cpp pin update must pass checksum-pinned real multimodal inference
  in both direct and worker runtimes; a successful WASM build alone is not
  sufficient.
- Speech-capable asset releases must pass the required local qualification in
  wasm32 and memory64; never run the large Qwen3-ASR/Qwen3-TTS pairs on hosted
  runners.
