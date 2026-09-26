// Shared fixtures for the stable release orchestrator suites, the port of
// scripts/release_orchestrator_fixtures_test.py.
//
// Pinned identities, provenance and payload builders, the fake Gateway, and the
// AdvancePipelineFixture setup the driver suites share. It defines no tests of
// its own. Every name keeps its Python meaning in camelCase: a keyword
// argument is an options key, a builder that returns API or manifest JSON keeps
// the JSON's snake_case keys, and a dataclass builder takes the model's
// camelCase options.

import { createHash } from 'node:crypto';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';

import { ASSETS_REPOSITORY, BRIDGE_REPOSITORY, ContractError, NATIVE_REPOSITORY } from '../../../scripts/release/contract.mjs';
import {
  compareCodePoints, pyIntFromString, pyJsonDumps, pyJsonLoads, pyRepr,
} from '../../../scripts/release/json.mjs';
import { ARTIFACTS, generate } from '../../../scripts/release/manifest.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as runNames from '../../../scripts/release/orchestrator/run_names.mjs';
import * as workflowRuns from '../../../scripts/release/orchestrator/workflow_runs.mjs';
import { PUBLICATION_FILES } from '../../../scripts/release/publication_state.mjs';
import { ZIP_DEFLATED, buildZip } from '../zip_fixture.mjs';

export const BRIDGE_SHA = '565c8396597ea7c0fb4e8d5d966da8d884b156d8';
export const ADVANCED_BRIDGE_SHA = '9'.repeat(40);
export const UPSTREAM_COMMIT = 'bb4caa7540188872173c44d161602d9271386413';
export const NATIVE_COMMIT = '1'.repeat(40);
export const NATIVE_MANIFEST_SHA = '2e5d29d7f98f0d71e75d3fa63b7c55f3b2a7933247cc34ea2b1c5e053d142452';
export const CANDIDATE_RUN_ID = '32919086955';
export const QUALIFICATION_RUN_ID = '32919086977';
export const CANDIDATE_ARTIFACT_ID = 7;
export const QUALIFICATION_ARTIFACT_ID = 9;
export const DEFAULT_BRANCH = 'main';
export const HEAD_SHA = 'a'.repeat(40);
export const ASSETS_TAG_COMMIT = 'c'.repeat(40);
export const OWNER = BRIDGE_REPOSITORY.split('/')[0];
export const EMSCRIPTEN_VERSION = '6.0.8';
export const NATIVE_PUBLISHED_AT = '2026-08-19T12:34:56Z';
export const LEGACY_BRIDGE_SHA = '0bdc8286fd52b70da27f5b039e1b4278361da0be';
export const LEGACY_UPSTREAM_COMMIT = 'c1d0e7a004015f23bc0233470b747b596f29b264';
export const LEGACY_NATIVE_COMMIT = '28fca14873d4b4c531bef4425b261e2b911bdcce';
export const LEGACY_NATIVE_MANIFEST_SHA = '811fda999e70c3ad2716d1c196688dd38db62cf11a78044855ca94f71fabed45';
export const LEGACY_CANDIDATE_RUN_ID = '33225744070';
export const LEGACY_MANUAL_QUALIFICATION_GATES = Object.freeze({
  state_persistence: 'passed',
  multimodal: 'passed',
  speech_to_text: 'required-local-attestation',
  text_to_speech: 'required-local-attestation',
});
export const LEGACY_MANUAL_UNPROVEN_CAPABILITIES = Object.freeze({
  real_device_intelligibility: 'unproven',
  real_device_playback: 'unproven',
  speaker_reference_fidelity: 'unproven',
});

// make_provenance(**overrides), with camelCase override keys.
export function makeProvenance(overrides = {}) {
  return new model.NativeProvenance({
    bridgeSourceSha: BRIDGE_SHA,
    bridgeBuildSha: BRIDGE_SHA,
    upstreamTag: 'v0.2.0',
    upstreamCommit: UPSTREAM_COMMIT,
    nativeRepo: NATIVE_REPOSITORY,
    nativeReleaseTag: 'v0.2.0',
    nativeCommit: NATIVE_COMMIT,
    nativeManifestSha256: NATIVE_MANIFEST_SHA,
    nativeReleasePublishedAt: NATIVE_PUBLISHED_AT,
    ...overrides,
  });
}

export function makeLegacyV0140Provenance(overrides = {}) {
  return new model.NativeProvenance({
    bridgeSourceSha: LEGACY_BRIDGE_SHA,
    bridgeBuildSha: LEGACY_BRIDGE_SHA,
    upstreamTag: 'v0.3.0',
    upstreamCommit: LEGACY_UPSTREAM_COMMIT,
    nativeRepo: NATIVE_REPOSITORY,
    nativeReleaseTag: 'v0.3.0',
    nativeCommit: LEGACY_NATIVE_COMMIT,
    nativeManifestSha256: LEGACY_NATIVE_MANIFEST_SHA,
    nativeReleasePublishedAt: '2026-08-28T12:34:56Z',
    ...overrides,
  });
}

// native_manifest(**overrides): the assets.json dict (snake_case JSON keys).
export function nativeManifest(overrides = {}) {
  return {
    schema_version: 1,
    native_release_tag: 'v0.2.0',
    llama_cpp_tag: 'v0.2.0',
    llama_cpp_commit: UPSTREAM_COMMIT,
    native_commit: NATIVE_COMMIT,
    ...overrides,
  };
}

// write_bridge_candidate(directory, ...): the artifacts (marker bytes plus
// "-<index>-<name>") and generate()'s manifest.json and sha256sums.txt.
export function writeBridgeCandidate(directory, {
  releaseTag,
  releaseRebuild,
  correlationId,
  bridgeCommit = BRIDGE_SHA,
  runId = CANDIDATE_RUN_ID,
  marker = Buffer.from('candidate'),
  upstreamTag = 'v0.2.0',
  upstreamCommit = UPSTREAM_COMMIT,
  nativeReleaseTag = 'v0.2.0',
  nativeManifestSha256 = NATIVE_MANIFEST_SHA,
  nativeCommit = NATIVE_COMMIT,
}) {
  fs.mkdirSync(directory, { recursive: true });
  ARTIFACTS.forEach((name, index) => {
    fs.writeFileSync(path.join(directory, name), Buffer.concat([Buffer.from(marker), Buffer.from(`-${index}-${name}`, 'utf8')]));
  });
  generate({
    outDir: directory,
    releaseTag,
    releaseRebuild,
    assetsRepo: ASSETS_REPOSITORY,
    bridgeRepo: BRIDGE_REPOSITORY,
    bridgeCommit,
    upstreamRepo: 'ggml-org/llama.cpp',
    upstreamTag,
    upstreamCommit,
    nativeRepo: NATIVE_REPOSITORY,
    nativeReleaseTag,
    nativeManifestSha256,
    nativeCommit,
    emscriptenVersion: EMSCRIPTEN_VERSION,
    orchestratorCorrelationId: correlationId,
    githubRunId: runId,
    githubRunUrl: `https://github.com/${BRIDGE_REPOSITORY}/actions/runs/${runId}`,
  });
}

// Rewrite manifest.json with the legacy v0.1.40 gate vocabulary, as
// json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n".
export function rewriteLegacyCandidateManifest(directory) {
  const manifestPath = path.join(directory, 'manifest.json');
  const manifest = pyJsonLoads(fs.readFileSync(manifestPath, 'utf8'));
  manifest.qualification_gates = { ...LEGACY_MANUAL_QUALIFICATION_GATES };
  manifest.unproven_capabilities = { ...LEGACY_MANUAL_UNPROVEN_CAPABILITIES };
  fs.writeFileSync(manifestPath, `${pyJsonDumps(manifest, { sortKeys: true, separators: [',', ':'] })}\n`, 'utf8');
}

// flat_zip(members): a deflated archive of { name: bytes }, members in sorted
// name order.
export function flatZip(members) {
  const entries = Object.entries(members).sort(([left], [right]) => compareCodePoints(left, right));
  return buildZip(entries.map(([name, data]) => [name, Buffer.from(data)]), { compression: ZIP_DEFLATED });
}

// directory_members(directory): { name: bytes } for every PUBLICATION_FILES
// member, in that order.
export function directoryMembers(directory) {
  return Object.fromEntries(PUBLICATION_FILES.map((name) => [name, fs.readFileSync(path.join(directory, name))]));
}

function sha256Hex(data) {
  return createHash('sha256').update(data).digest('hex');
}

// release_payload(...): an assets-repository release object with one uploaded
// asset per member (sorted by name, ids from 900), each digest the member's
// SHA-256. assetOverrides maps a member name to fields merged into its asset.
export function releasePayload({
  tag,
  body,
  members,
  releaseId = 4242,
  immutable = true,
  publishedAt = '2026-08-20T03:24:11Z',
  draft = false,
  prerelease = null,
  targetCommitish = ASSETS_TAG_COMMIT,
  assetOverrides = null,
}) {
  const assets = Object.entries(members)
    .sort(([left], [right]) => compareCodePoints(left, right))
    .map(([name, data], index) => {
      const asset = {
        id: 900 + index,
        name,
        state: 'uploaded',
        size: data.length,
        digest: `sha256:${sha256Hex(data)}`,
      };
      if (assetOverrides && Object.hasOwn(assetOverrides, name)) Object.assign(asset, assetOverrides[name]);
      return asset;
    });
  return {
    id: releaseId,
    tag_name: tag,
    name: tag,
    draft,
    prerelease: prerelease === null ? tag.includes('-') : prerelease,
    immutable,
    target_commitish: targetCommitish,
    published_at: publishedAt,
    body,
    assets,
  };
}

export function assetReleaseStub(tag = 'v0.1.39') {
  return {
    tag_name: tag,
    draft: false,
    prerelease: tag.includes('-'),
    published_at: '2026-08-20T03:24:11Z',
    body: '',
  };
}

// A listed publication carrying the deterministic release-note markers.
export function alignedReleaseStub(tag, provenance) {
  const release = assetReleaseStub(tag);
  release.body = (
    `Native: \`${provenance.nativeRepo}@${provenance.nativeReleaseTag}\`\n`
    + `Native manifest SHA-256: \`${provenance.nativeManifestSha256}\`\n`
    + `Orchestrator correlation: \`${runNames.computeCorrelationId(provenance)}\`\n`
  );
  return release;
}

// run_payload(...): one actions/runs record. `conclusion` defaults to
// "success"; pass null for None. It is always null for a run not completed.
export function runPayload({
  runId,
  path: workflowPath,
  runName,
  status = 'completed',
  conclusion = 'success',
  headBranch = DEFAULT_BRANCH,
  headSha = HEAD_SHA,
  runAttempt = 1,
  event = 'workflow_dispatch',
  apiName = null,
  actor = OWNER,
  triggeringActor = OWNER,
}) {
  return {
    id: pyIntFromString(runId),
    // Workflows with run-name expose the rendered correlation string in both
    // fields. Keep the fixture aligned with the live Actions API.
    name: apiName === null ? runName : apiName,
    display_title: runName,
    path: workflowPath,
    event,
    status,
    conclusion: status === 'completed' ? conclusion : null,
    head_branch: headBranch,
    head_sha: headSha,
    run_attempt: runAttempt,
    repository: { full_name: BRIDGE_REPOSITORY },
    head_repository: { full_name: BRIDGE_REPOSITORY },
    actor: { login: actor },
    triggering_actor: { login: triggeringActor },
  };
}

// The live API answers an object holding workflow_runs, never a bare list.
export function runsResponse(runs) {
  return { total_count: runs.length, workflow_runs: runs };
}

export function artifactInventory({
  runId, name, artifactId, extra = null,
}) {
  const artifacts = [{
    id: artifactId,
    name,
    expired: false,
    workflow_run: { id: pyIntFromString(runId) },
  }];
  artifacts.push(...(extra ?? []));
  return { total_count: artifacts.length, artifacts };
}

// The key of FakeGateway's releaseAttestations: the (repository, release_tag)
// tuple.
export function attestationKey(repository, releaseTag) {
  return JSON.stringify([repository, releaseTag]);
}

// Deterministic stand-in for the live gh transport (GhGateway's interface).
//
// jsonRoutes and blobRoutes map an exact API path to its answer; they are
// plain objects the tests may extend after construction. releaseAttestations
// is an iterable of [[repository, releaseTag], payload] entries. The gateway
// records every apiJson path in apiPaths, every dispatch as
// { workflowFile, ref, inputs } in dispatches, and every sleep in slept;
// utcNow() is the fixed `now`. `identity` and `governance` take null for
// Python's None; an omitted identity is the owner.
export class FakeGateway {
  constructor({
    jsonRoutes = null,
    blobRoutes = null,
    identity = OWNER,
    governance = null,
    releaseAttestations = null,
    now = '2026-08-30T00:00:00Z',
  } = {}) {
    this.jsonRoutes = { ...(jsonRoutes ?? {}) };
    this.blobRoutes = { ...(blobRoutes ?? {}) };
    this.identity = identity;
    this.releaseAttestations = new Map();
    for (const [[repository, releaseTag], payload] of releaseAttestations ?? []) {
      this.releaseAttestations.set(attestationKey(repository, releaseTag), payload);
    }
    this.now = now;
    this.dispatches = [];
    this.slept = [];
    this.apiPaths = [];
    const setdefault = (key, value) => {
      if (!Object.hasOwn(this.jsonRoutes, key)) this.jsonRoutes[key] = value;
    };
    setdefault(`repos/${BRIDGE_REPOSITORY}`, { default_branch: DEFAULT_BRANCH });
    setdefault(`repos/${BRIDGE_REPOSITORY}/environments/bridge-assets-publication`, {
      name: 'bridge-assets-publication',
      can_admins_bypass: false,
      protection_rules: [{ type: 'branch_policy' }],
      deployment_branch_policy: {
        protected_branches: false,
        custom_branch_policies: true,
      },
    });
    setdefault(`repos/${BRIDGE_REPOSITORY}/environments/bridge-assets-publication/deployment-branch-policies`, {
      total_count: 1,
      branch_policies: [{ name: 'main', type: 'branch' }],
    });
    setdefault(
      `repos/${ASSETS_REPOSITORY}/immutable-releases`,
      governance !== null ? governance : { enabled: true, enforced_by_owner: true },
    );
    setdefault(`repos/${ASSETS_REPOSITORY}/git/matching-refs/tags?per_page=100`, [[]]);
  }

  apiJson(apiPath, { paginate: _paginate = false, privileged: _privileged = false } = {}) {
    this.apiPaths.push(apiPath);
    if (!Object.hasOwn(this.jsonRoutes, apiPath)) throw new ContractError(`unmapped API path in test gateway: ${apiPath}`);
    return this.jsonRoutes[apiPath];
  }

  downloadBytes(apiPath, { accept: _accept, privileged: _privileged = false } = {}) {
    if (!Object.hasOwn(this.blobRoutes, apiPath)) throw new ContractError(`unmapped blob path in test gateway: ${apiPath}`);
    return this.blobRoutes[apiPath];
  }

  dispatchIdentity() {
    return this.identity;
  }

  releaseAttestation({ repository, releaseTag }) {
    const key = attestationKey(repository, releaseTag);
    if (!this.releaseAttestations.has(key)) {
      throw new ContractError(`unmapped release attestation in test gateway: (${pyRepr(repository)}, ${pyRepr(releaseTag)})`);
    }
    return this.releaseAttestations.get(key);
  }

  dispatchWorkflow({ workflowFile, ref, inputs }) {
    this.dispatches.push({ workflowFile, ref, inputs: { ...inputs } });
  }

  sleep(seconds) {
    this.slept.push(seconds);
  }

  utcNow() {
    return this.now;
  }
}

// _provenance_to_dict of stable_release_orchestrator.py.
export function provenanceToDict(provenance) {
  return {
    bridge_source_sha: provenance.bridgeSourceSha,
    bridge_build_sha: provenance.bridgeBuildSha,
    upstream_tag: provenance.upstreamTag,
    upstream_commit: provenance.upstreamCommit,
    native_repo: provenance.nativeRepo,
    native_release_tag: provenance.nativeReleaseTag,
    native_commit: provenance.nativeCommit,
    native_manifest_sha256: provenance.nativeManifestSha256,
    native_release_published_at: provenance.nativeReleasePublishedAt,
  };
}

// Setup and routes shared by every AdvancePipelineTest suite: the Python
// mixin's setUp state (tmp, provenance, correlationId, binding, candidateName)
// and helpers (routes, newerNative, runBacklog). withAdvancePipelineFixture(fn)
// wraps one test on a fresh fixture and always removes its directory.
//
// runBacklog drives the CLI entry's `orchestrate-backlog` with the fake
// gateway, which Python does by patching stable_release_orchestrator.GhGateway.
// The entry module is injected as `orchestrator` (constructor option or
// runBacklog option) and must export
//   main(argv, { env, createGateway, stdout, stderr }) -> exit status (or a
//   Promise of it)
// where createGateway({ readToken, dispatchToken }) replaces the GhGateway
// constructor, env replaces process.env, and stdout/stderr receive the
// output text. If it exports provenanceToDict, runBacklog uses it.
export class AdvancePipelineFixture {
  constructor({ orchestrator = null } = {}) {
    this.orchestrator = orchestrator;
    this.tmp = fs.mkdtempSync(path.join(tmpdir(), 'sro-advance-'));
    this.provenance = makeProvenance();
    this.correlationId = runNames.computeCorrelationId(this.provenance);
    this.binding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.40', releaseRebuild: 0 });
    this.candidateName = runNames.candidateRunName(this.correlationId, this.binding);
  }

  tearDown() {
    fs.rmSync(this.tmp, { recursive: true, force: true });
  }

  // _routes(...): the paginated releases listing, both compare answers, and the
  // three filtered run listings since NATIVE_PUBLISHED_AT.
  routes({
    releases = null, candidateRuns = null, qualificationRuns = null, publishRuns = null,
  } = {}) {
    const runsPath = (workflowFile) => workflowRuns.workflowRunsPath({
      workflowFile,
      defaultBranch: DEFAULT_BRANCH,
      createdSince: NATIVE_PUBLISHED_AT,
    });
    return {
      [`repos/${ASSETS_REPOSITORY}/releases?per_page=100`]: [releases || []],
      [`repos/${BRIDGE_REPOSITORY}/compare/${BRIDGE_SHA}...${DEFAULT_BRANCH}`]: { status: 'ahead' },
      [`repos/${BRIDGE_REPOSITORY}/compare/${HEAD_SHA}...${DEFAULT_BRANCH}`]: { status: 'identical' },
      [runsPath(model.CANDIDATE_WORKFLOW_FILE)]: runsResponse(candidateRuns || []),
      [runsPath(model.QUALIFICATION_WORKFLOW_FILE)]: runsResponse(qualificationRuns || []),
      [runsPath(model.PUBLISH_WORKFLOW_FILE)]: runsResponse(publishRuns || []),
    };
  }

  // _newer_native(**overrides): a v0.2.1 provenance.
  newerNative(overrides = {}) {
    return makeProvenance({
      upstreamTag: 'v0.2.1',
      upstreamCommit: 'd'.repeat(40),
      nativeReleaseTag: 'v0.2.1',
      nativeCommit: 'e'.repeat(40),
      nativeManifestSha256: 'f'.repeat(64),
      nativeReleasePublishedAt: '2026-08-21T12:34:56Z',
      ...overrides,
    });
  }

  // _run_backlog(gateway, provenances): [exit status, parsed plan JSON].
  async runBacklog(gateway, provenances, { orchestrator = this.orchestrator } = {}) {
    if (!orchestrator) {
      throw new Error('AdvancePipelineFixture.runBacklog needs the orchestrator CLI entry module (stage 2)');
    }
    const toDict = orchestrator.provenanceToDict ?? provenanceToDict;
    const provenanceList = path.join(this.tmp, 'release-candidates.json');
    fs.writeFileSync(provenanceList, pyJsonDumps(provenances.map((value) => toDict(value))), 'utf8');
    const outputPlan = path.join(this.tmp, 'orchestration-plan.json');
    const env = {
      ...process.env,
      GITHUB_EVENT_NAME: 'schedule',
      GITHUB_ACTOR: 'github-actions',
      GITHUB_TRIGGERING_ACTOR: 'github-actions',
    };
    const result = await orchestrator.main([
      'orchestrate-backlog',
      '--provenance-list-json',
      provenanceList,
      '--workspace',
      path.join(this.tmp, 'workspace'),
      '--output-plan-json',
      outputPlan,
    ], {
      env,
      createGateway: () => gateway,
      stdout: () => {},
      stderr: () => {},
    });
    return [result, pyJsonLoads(fs.readFileSync(outputPlan, 'utf8'))];
  }
}

// One AdvancePipelineTest test: fn(fixture) on a fresh fixture (options go to
// the constructor), which is always torn down.
export function withAdvancePipelineFixture(fn, options = {}) {
  return async () => {
    const fixture = new AdvancePipelineFixture(options);
    try {
      await fn(fixture);
    } finally {
      fixture.tearDown();
    }
  };
}
