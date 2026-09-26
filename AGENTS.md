# AGENTS.md

Guidance for coding agents working in `llama-web-bridge`. `README.md` owns the
integration surface, CI/qualification/publication narrative, and CDN list;
`CONTRIBUTING.md` owns the runnable build recipes, smoke invocations and their
pins, workflow guardrails, and publish process. This file keeps only what
neither states.

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

`./scripts/build_bridge.sh --help` lists every environment variable the build
reads, with defaults. CI clones the `llama_cpp.version` tag into the script's
default `third_party/llama_cpp` and installs the exact `emsdk.version`
compiler; a local Emscripten SDK must match that release.

## Agent PR Workflow

For non-trivial runtime, workflow, or API changes, keep the PR path explicit:

1. Start from a clean topic branch and inspect `git status` before editing.
2. Add or update a regression/contract check before changing behavior when
   practical. Static contract scripts are acceptable for workflow invariants.
3. Keep Emscripten build directories, ccache, model caches, and Playwright
   artifacts outside the repository unless they are intentionally versioned.
4. Run the targeted checks below and the full browser smoke when the change
   touches `js/`, `src/`, `scripts/`, or GitHub workflows.
5. Use an independent review before committing PR-bound changes. Fix blocking
   findings, rerun the targeted checks, then commit locally; do not push or open
   a PR unless the maintainer asks.

### Local Verification Notes

The external-path build recipe (`CCACHE_DIR`, `EM_CACHE`, `BUILD_DIR`,
`MEM64_BUILD_DIR`, `OUT_DIR`), the lightweight contract list starting with
`npm run check:js`, and every browser smoke invocation live in
[CONTRIBUTING.md](CONTRIBUTING.md#validate-outputs) under Local Build and
Validate Outputs. Run them from there; do not copy them here. Which smoke
applies:

- state-persistence or workflow changes:
  `scripts/state_persistence_browser_smoke.mjs` against a built `OUT_DIR`;
- llama.cpp pin or multimodal changes: `scripts/multimodal_browser_smoke.mjs`
  through both direct and worker runtimes;
- sampler or grammar changes: `scripts/grammar_browser_smoke.mjs` with the
  state-persistence model and the multimodal model;
- next-token scoring or prompt-evaluation changes:
  `scripts/next_token_scores_browser_smoke.mjs` with the state-persistence model;
- speech changes: `scripts/speech_to_text_browser_smoke.py` and
  `scripts/text_to_speech_browser_smoke.py` individually. The combined
  `release_qualification.py qualify` command is workflow-only because it
  requires GitHub Actions and `github-hosted` runner identity;
- decision-head changes: `scripts/decision_browser_smoke.py` with a Laya
  encoder GGUF, head and reference fixture.

Keep the tiny model in a user cache or `/private/tmp`; do not commit downloaded
GGUFs or smoke artifacts.

## CI / Release

- CI build gate: `.github/workflows/ci.yml`
  - Builds wasm32/memory64 against the `llama_cpp.version` pin in one lane.
    Preserve its check/artifact identities (`Build WebGPU Bridge (WASM)`,
    `webgpu-bridge-dist`). It neither publishes nor changes the source pin.
    Builds against llama.cpp v0.4.0 are no longer tested in CI; the
    `src/llama_webgpu_mtmd_compat.h` shim and its static contract remain.
  - Verifies the active `emcc` identity against `emsdk.version` and
    contract-tests all five required wasm64 WASMFS patches.
  - `scripts/ci_scope.mjs` lets `README.md`, `CONTRIBUTING.md`, `LICENSE`, and
    `docs/*.md` changes skip the build lane; `AGENTS.md` is not in that
    allowlist, so editing it still runs it.
- Candidate build: `.github/workflows/bridge_candidate.yml`
  - The only workflow that builds publishable assets. It uploads the bundle as
    `exact-webgpu-bridge-dist`, which qualification and publication download by
    immutable artifact ID. It refuses `github.run_attempt != 1`; dispatch a new
    candidate after failure rather than rerunning it.
  - Holds no credential that can read another repository's administration
    settings, so it fails closed on the dispatcher's
    `assets_immutable_releases_enabled` assertion and records it in
    `bridge-candidate-prequalification`. Confirm the assertion first with
    `gh api repos/leehack/llama-web-bridge-assets/immutable-releases`.
- Automated qualification: `.github/workflows/bridge_qualification.yml`
  - Owner-dispatched by the orchestrator with the exact candidate run ID and
    correlation ID; no maintainer-supplied attestation input exists. It uploads
    one `qualification-attestation` artifact using repository access and no
    PAT.
- Orchestrator: `.github/workflows/auto_llama_cpp_update.yml`
  - A candidate run reserves its output tag for other pipelines unless its
    correlation names another governed build identity and none of that
    correlation's candidate, qualification or publication runs is in flight.
  - The resolver uses full default-branch history to keep the exact source SHA
    that executes a new candidate separate from the newest first-parent commit
    that changed governed runtime/build inputs. `AGENTS.md`, `README.md`, and
    `CONTRIBUTING.md` are in `_ORCHESTRATION_ONLY_PATHS` and `docs/` in
    `_ORCHESTRATION_ONLY_PREFIXES` in `scripts/stable_release_orchestrator.py`,
    so docs-only commits never advance the build identity; every unclassified
    new path is governed by default.
  - Every dispatch sends exactly the target workflow's declared
    `workflow_dispatch` inputs (`require_exact_dispatch_inputs`) at the exact
    default-branch `--ref` after a live `immutable-releases` governance read;
    duplicate in-flight or successful runs for one stage fail closed, and a
    run-name readback follows each dispatch.
  - Run recovery accepts only an exact supported workflow path plus its
    deterministic `display_title`, owner actor/triggering actor, first attempt,
    repository, and branch. GitHub exposes a workflow's rendered `run-name` in
    the run record's `name` field, so that field is not treated as the static
    workflow identity. A search at GitHub's 1,000-result cap is split into
    closed time windows; a saturated one-second window or ambiguous result
    fails closed.
  - The dispatch job reuses the `bridge-assets-publication` environment and its
    `WEBGPU_BRIDGE_ASSETS_PAT`, which must identify the repository owner, read
    assets immutable-release governance, and hold Actions write permission on
    `leehack/llama-web-bridge`. The orchestrator proves the owner identity and
    the governance read live and fails closed if either is absent.
- CI reliability contract: `scripts/verify_ci_reliability.mjs`
  - Checks publication-safety invariants, not wording: it parses the workflows
    with `yaml` and reads commands from resolved `run` scripts. It covers
    read-only workflow and job permissions, no `continue-on-error` outside the
    publication ref mutation, the environment gates, `WEBGPU_BRIDGE_ASSETS_PAT` as the only
    secret with its fail-closed, never-printed guard, immutable-release
    governance and readback, owner-only first-attempt runs, artifact download
    by immutable ID, the toolchain pins, and that CI, candidate, and publish run
    the contract tests (`check:js` must run every `tests/js/*_test.mjs`). It
    asserts the 7-pin set in `CONTRIBUTING.md` and both build workflows, and
    that `README.md` and `AGENTS.md` hold no pins. It checks no documentation
    prose. `tests/js/verify_ci_reliability_test.mjs` tests its pin and PAT
    checks.
- Publish workflow: `.github/workflows/publish_assets.yml`
  - Never builds. It downloads the exact candidate artifact and attestation by
    immutable artifact ID and verifies the candidate manifest's
    `emscripten_version` against the `emsdk.version` pin at the exact bridge
    source SHA.
  - Requires `publish_approved=true`. Publication remains blocked until an
    administrator externally creates `bridge-assets-publication`, disables
    administrator bypass, restricts custom deployment branches to `main`, and
    stores `WEBGPU_BRIDGE_ASSETS_PAT` as an environment-scoped secret. The
    solo-maintainer publication contract does not require a reviewer rule. Do
    not describe that environment as protected without current live evidence.
    Use the default job token to validate the environment identity,
    administrator-bypass setting, and exact `main` branch policy before approval
    and again after approval, with the trusted workflow commit's validator.
    Immediately before any network use of the environment-scoped publication
    PAT, fail closed unless the injected credential is non-empty,
    without printing its value.
  - Proves immutable-release governance on the assets repository through
    `GET /repos/{owner}/{repo}/immutable-releases` before any ref or release
    mutation, reads every complete release back by tag and by release ID with
    explicit boolean `immutable: true`, and requires
    `gh release verify <tag> --repo <assets repo> --format json` to prove the
    `https://in-toto.io/attestation/release/v0.2` attestation over the exact
    published bytes. A mismatch is a non-retryable
    `immutable-publication-unverified` outcome; publication never deletes,
    retags, overwrites, or otherwise repairs the release.

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

- Keep runtime bridge source code in `js/src/` and `src/`. The
  generated bridge wrapper outputs and declaration (`js/llama_webgpu_bridge.js`,
  `js/llama_webgpu_bridge_worker.js`, `js/llama_webgpu_bridge.d.ts`) are
  regenerated by `npm run check:js`; CI, candidate, and publish fail on a stale
  copy with `git diff --exit-code`, so never hand-edit them.
- JS contract tests, including the static API contracts, and their fixtures
  live in `tests/js/`. Build, release, smoke, and verify tooling and its Python
  tests stay in `scripts/`: the qualification harness digest and the
  candidate/publication checkouts address those files by their `scripts/` path
  at older commits, so moving them breaks in-flight candidates.
- `scripts/stable_release_orchestrator.py` is the orchestrator's CLI entry; the
  state machine lives in the `scripts/release_orchestrator_<concern>.py` modules
  it imports, tested by `scripts/release_orchestrator_<concern>_test.py`
  suites that share `release_orchestrator_fixtures_test.py`. List a new module in
  `_ORCHESTRATION_ONLY_PATHS` and in `TOOLING` in `scripts/ci_scope.mjs`.
- Keep publishing logic in workflow only.
- Do not edit assets repository files from here outside publish flow.
- C++ exception catching is enabled only for
  `llamadart_webgpu_grammar_sampler_init` (`EXCEPTION_CATCHING_ALLOWED` in
  `CMakeLists.txt`), which turns llama.cpp's grammar parser throws into an
  `(invalid grammar)` error. Do not widen it with `-fexceptions` or
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
- Speech-capable asset releases must pass the hosted automated qualification
  workflow's exact required memory/runtime matrix before publication. Keep
  these heavy gates outside ordinary CI; reproduce locally with the individual
  smoke scripts, not the workflow-only combined qualifier.
