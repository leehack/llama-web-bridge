// Tests of scripts/release/publication_state.mjs, one test per test method of
// scripts/release_publication_state_test.py, with the same names and
// assertions. Each test builds its own assets git repository and candidate in
// a fresh temporary directory, so the tests are independent and parallel-safe.

import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  copyFileSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, symlinkSync, unlinkSync, writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { ContractError } from '../../scripts/release/contract.mjs';
import { ARTIFACTS, AUTOMATED_QUALIFICATION_REQUIRED, generate } from '../../scripts/release/manifest.mjs';
import { pyJsonDumps, pyJsonLoads } from '../../scripts/release/json.mjs';
import {
  APPROVED_ASSETS_REPOSITORY,
  CandidateIdentity,
  PUBLICATION_FILES,
  candidatePublicationDigests,
  classify,
  fatal,
  mutationUnknownFromRequery,
  mutationUnknownOutcome,
  publicationStateChanged,
  requireApprovedAssetsRepo,
  validateCandidate,
  verifyImmutablePublication,
} from '../../scripts/release/publication_state.mjs';
import { releaseAttestation } from './contract_fixtures.mjs';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const CLI = path.join(rootDir, 'scripts/release/publication_state.mjs');

function git(repository, ...args) {
  return execFileSync('git', ['-C', repository, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] }).trim();
}

// json.dumps(value) with Python's default separators.
function dumps(value) {
  return pyJsonDumps(value);
}

function sha256(data) {
  return createHash('sha256').update(data).digest('hex');
}

function generateArgs(outDir, identity) {
  return {
    outDir,
    releaseTag: identity.releaseTag,
    releaseRebuild: identity.releaseRebuild,
    assetsRepo: APPROVED_ASSETS_REPOSITORY,
    bridgeRepo: 'leehack/llama-web-bridge',
    bridgeCommit: identity.bridgeCommit,
    upstreamRepo: 'ggml-org/llama.cpp',
    upstreamTag: identity.upstreamTag,
    upstreamCommit: identity.upstreamCommit,
    nativeRepo: 'leehack/llamadart-native',
    nativeReleaseTag: identity.nativeReleaseTag,
    nativeManifestSha256: identity.nativeManifestSha256,
    nativeCommit: identity.nativeCommit,
    emscriptenVersion: identity.emscriptenVersion,
    orchestratorCorrelationId: identity.orchestratorCorrelationId,
    githubRunId: identity.githubRunId,
    githubRunUrl: identity.githubRunUrl,
  };
}

// The Python test case's setUp state and helper methods.
class Fixture {
  constructor() {
    this.temporary = mkdtempSync(path.join(tmpdir(), 'release-publication-state-'));
    this.repository = path.join(this.temporary, 'assets');
    this.candidate = path.join(this.temporary, 'candidate');
    mkdirSync(this.repository);
    mkdirSync(this.candidate);
    git(this.repository, 'init', '-q');
    git(this.repository, 'config', 'user.name', 'test');
    git(this.repository, 'config', 'user.email', 'test@example.com');

    ARTIFACTS.forEach((name, index) => writeFileSync(path.join(this.candidate, name), `candidate-${index}`));
    this.identity = new CandidateIdentity({
      releaseTag: 'v0.2.0',
      releaseRebuild: 0,
      assetsRepo: APPROVED_ASSETS_REPOSITORY,
      bridgeCommit: 'a'.repeat(40),
      upstreamTag: 'v0.2.0',
      upstreamCommit: 'bb4caa7540188872173c44d161602d9271386413',
      nativeReleaseTag: 'v0.2.0',
      nativeManifestSha256: '2e5d29d7f98f0d71e75d3fa63b7c55f3b2a7933247cc34ea2b1c5e053d142452',
      nativeCommit: 'e5c240e34b525da953ed98dc743516eef78cb738',
      emscriptenVersion: '6.0.8',
      orchestratorCorrelationId: 'kanban:t_fcc0b814:web-v0.1.38',
      githubRunId: '123456789',
      githubRunUrl: 'https://github.com/leehack/llama-web-bridge/actions/runs/123456789',
    });
    generate(generateArgs(this.candidate, this.identity));
    const previous = { bridge_assets_tag: 'v0.1.23', llama_cpp_tag: 'b10514', source_commit: 'e'.repeat(40) };
    writeFileSync(path.join(this.repository, 'manifest.json'), `${dumps(previous)}\n`);
    writeFileSync(path.join(this.repository, 'README.md'), 'assets\n');
    git(this.repository, 'add', '.');
    git(this.repository, 'commit', '-q', '-m', 'previous');
  }

  cleanup() {
    rmSync(this.temporary, { recursive: true, force: true });
  }

  get head() {
    return git(this.repository, 'rev-parse', 'HEAD');
  }

  publishedRelease(overrides = {}) {
    return { ...this.releaseData(this.head), immutable: true, ...overrides };
  }

  publishedAttestation(overrides = {}) {
    return releaseAttestation({
      releaseTag: 'v0.2.0',
      assetsRepo: APPROVED_ASSETS_REPOSITORY,
      tagCommit: this.head,
      releaseId: 42,
      assets: candidatePublicationDigests(this.candidate),
      ...overrides,
    });
  }

  verifyPublished(overrides = {}) {
    return verifyImmutablePublication({
      candidate: this.candidate,
      assetsRepo: APPROVED_ASSETS_REPOSITORY,
      releaseTag: 'v0.2.0',
      tagCommit: this.head,
      releaseId: 42,
      releaseByTag: this.publishedRelease(),
      releaseById: this.publishedRelease(),
      attestation: this.publishedAttestation(),
      ...overrides,
    });
  }

  releaseData(commit) {
    const fingerprint = validateCandidate(this.candidate, this.identity);
    const assets = PUBLICATION_FILES.map((name) => {
      const data = readFileSync(path.join(this.candidate, name));
      return { name, state: 'uploaded', size: data.length, digest: `sha256:${sha256(data)}` };
    });
    return {
      id: 42,
      tag_name: 'v0.2.0',
      name: 'v0.2.0',
      draft: false,
      prerelease: false,
      target_commitish: commit,
      published_at: '2026-08-20T22:15:59Z',
      body: `Candidate fingerprint: \`${fingerprint}\`\n`
        + `Orchestrator correlation: \`${this.identity.orchestratorCorrelationId}\`\n`
        + `Bridge workflow run: ${this.identity.githubRunUrl}`,
      assets,
    };
  }

  inspect({ tag = null, release = false } = {}) {
    const releaseData = release === true ? this.releaseData(tag ?? this.head) : release || null;
    return classify({
      repository: this.repository,
      candidate: this.candidate,
      identity: this.identity,
      branchCommit: this.head,
      tagCommit: tag,
      release: releaseData,
    });
  }

  copyCandidateIntoRepository() {
    for (const name of readdirSync(this.candidate)) copyFileSync(path.join(this.candidate, name), path.join(this.repository, name));
  }

  publishBranchCandidate() {
    this.copyCandidateIntoRepository();
    git(this.repository, 'add', '.');
    git(this.repository, 'commit', '-q', '-m', 'candidate');
    return this.head;
  }

  commitHistoryManifest(releaseTag, upstreamTag) {
    const manifest = { bridge_assets_tag: releaseTag, llama_cpp_tag: upstreamTag, source_commit: 'e'.repeat(40) };
    writeFileSync(path.join(this.repository, 'manifest.json'), `${dumps(manifest)}\n`);
    git(this.repository, 'add', 'manifest.json');
    git(this.repository, 'commit', '-q', '-m', `history ${releaseTag}`);
  }

  regenerateCandidate(releaseTag, releaseRebuild, upstreamTag, nativeReleaseTag = null) {
    const identity = new CandidateIdentity({
      ...this.identity,
      releaseTag,
      releaseRebuild,
      upstreamTag,
      nativeReleaseTag: nativeReleaseTag ?? releaseTag,
    });
    generate(generateArgs(this.candidate, identity));
    return identity;
  }

  classifyIdentity(identity) {
    return classify({
      repository: this.repository,
      candidate: this.candidate,
      identity,
      branchCommit: this.head,
      tagCommit: null,
      release: null,
    });
  }
}

// test(name, fn) with a fresh fixture that is always cleaned up.
function fixtureTest(name, fn) {
  test(name, () => {
    const fixture = new Fixture();
    try {
      fn(fixture);
    } finally {
      fixture.cleanup();
    }
  });
}

fixtureTest('test_target_is_locked_before_credentials', () => {
  assert.equal(requireApprovedAssetsRepo(APPROVED_ASSETS_REPOSITORY), APPROVED_ASSETS_REPOSITORY);
  assert.throws(() => requireApprovedAssetsRepo('attacker/unrelated-assets'), ContractError);
});

fixtureTest('test_no_refs_is_new_publication', (self) => {
  const state = self.inspect();
  assert.equal(state.state, 'absent');
  assert.equal(state.action, 'publish-refs-and-release');
  assert.equal(state.outcome, 'newly-published');
});

fixtureTest('test_development_advances_from_current_legacy_stable_tag', (self) => {
  self.commitHistoryManifest('v0.1.37', 'b10514');
  const identity = self.regenerateCandidate('b10515', 0, 'b10515');
  const state = self.classifyIdentity(identity);
  assert.equal(state.allowed, true);
  assert.equal(state.action, 'publish-refs-and-release');
});

fixtureTest('test_development_history_rejects_rollback_and_collision', (self) => {
  self.commitHistoryManifest('b10514', 'b10514');
  const rollback = self.regenerateCandidate('b10513', 0, 'b10513');
  assert.equal(self.classifyIdentity(rollback).outcome, 'rollback');
  const collision = self.regenerateCandidate('b10514', 0, 'b10514');
  assert.equal(self.classifyIdentity(collision).outcome, 'collision');
});

fixtureTest('test_development_rebuild_advances_on_its_own_release_line', (self) => {
  self.commitHistoryManifest('b10514', 'b10514');
  const identity = self.regenerateCandidate('b10514-1', 1, 'b10514');
  assert.equal(self.classifyIdentity(identity).allowed, true);
});

// A rebuild of an asset tag never published in this channel is fail-closed.
fixtureTest('test_first_release_in_a_channel_must_use_rebuild_zero', (self) => {
  self.commitHistoryManifest('v0.1.37', 'b10514');
  const identity = self.regenerateCandidate('b10514-1', 1, 'b10514');
  const state = self.classifyIdentity(identity);
  assert.equal(state.allowed, false);
  assert.equal(state.outcome, 'rollback');
});

fixtureTest('test_development_rebuild_history_rejects_rollback_and_collision', (self) => {
  self.commitHistoryManifest('b10514-2', 'b10514');
  const rollback = self.regenerateCandidate('b10514-1', 1, 'b10514');
  assert.equal(self.classifyIdentity(rollback).outcome, 'rollback');
  const collision = self.regenerateCandidate('b10514-2', 2, 'b10514');
  assert.equal(self.classifyIdentity(collision).outcome, 'collision');
});

// v0.1.38/v0.2.0 after v0.1.37/b10514: assets advance, upstream migrates.
fixtureTest('test_chief_candidate_advances_independent_release_and_upstream_lines', (self) => {
  self.commitHistoryManifest('v0.1.37', 'b10514');
  const identity = self.regenerateCandidate('v0.1.38', 0, 'v0.2.0', 'v0.2.0-1');
  const state = self.classifyIdentity(identity);
  assert.equal(state.allowed, true, state.reason);
  assert.equal(state.state, 'absent');
  assert.equal(state.action, 'publish-refs-and-release');
  assert.equal(state.outcome, 'newly-published');
  assert.equal(state.release_tag, 'v0.1.38');
});

fixtureTest('test_schema_v2_stores_independent_release_and_upstream_identities', (self) => {
  const identity = self.regenerateCandidate('v0.1.38', 0, 'v0.2.0', 'v0.2.0-1');
  assert.equal(typeof validateCandidate(self.candidate, identity), 'string');
  const manifest = pyJsonLoads(readFileSync(path.join(self.candidate, 'manifest.json'), 'utf8'));
  assert.equal(manifest.schema_version, 2);
  assert.equal(manifest.release_tag, 'v0.1.38');
  assert.equal(manifest.upstream_tag, 'v0.2.0');
  assert.equal(manifest.release_channel, 'stable');
  assert.equal(manifest.release_rebuild, 0);
  assert.equal(manifest.native_release_tag, 'v0.2.0-1');
  // Legacy aliases must mirror their own field, never the other identity.
  assert.equal(manifest.bridge_assets_tag, 'v0.1.38');
  assert.equal(manifest.llama_cpp_tag, 'v0.2.0');
});

fixtureTest('test_schema_v2_history_is_read_without_fabricating_identities', (self) => {
  self.regenerateCandidate('v0.1.38', 0, 'v0.2.0', 'v0.2.0-1');
  copyFileSync(path.join(self.candidate, 'manifest.json'), path.join(self.repository, 'manifest.json'));
  git(self.repository, 'add', 'manifest.json');
  git(self.repository, 'commit', '-q', '-m', 'schema-v2 history');

  const advance = self.regenerateCandidate('v0.1.39', 0, 'v0.2.1', 'v0.2.1');
  assert.equal(self.classifyIdentity(advance).allowed, true);
  // Ordering uses the recorded v0.1.38, not the recorded upstream v0.2.0.
  const releaseRollback = self.regenerateCandidate('v0.1.37', 0, 'v0.2.0', 'v0.2.0-1');
  assert.equal(self.classifyIdentity(releaseRollback).outcome, 'rollback');
  // ...and the upstream dimension uses the recorded v0.2.0, not v0.1.38.
  const upstreamRollback = self.regenerateCandidate('v0.1.39', 0, 'v0.1.9', 'v0.1.9');
  const upstreamState = self.classifyIdentity(upstreamRollback);
  assert.equal(upstreamState.outcome, 'rollback');
  assert.ok(upstreamState.reason.includes('upstream transition is backward'), upstreamState.reason);
});

fixtureTest('test_conflicting_history_aliases_fail_closed', (self) => {
  for (const [field, value] of [['bridge_assets_tag', 'v0.1.37'], ['llama_cpp_tag', 'b10514']]) {
    const manifest = {
      release_tag: 'v0.1.38',
      bridge_assets_tag: 'v0.1.38',
      upstream_tag: 'v0.2.0',
      llama_cpp_tag: 'v0.2.0',
    };
    manifest[field] = value;
    writeFileSync(path.join(self.repository, 'manifest.json'), `${dumps(manifest)}\n`);
    git(self.repository, 'add', 'manifest.json');
    git(self.repository, 'commit', '-q', '-m', `conflicting ${field}`);
    const identity = self.regenerateCandidate('v0.1.39', 0, 'v0.2.1', 'v0.2.1');
    const state = self.classifyIdentity(identity);
    assert.equal(state.allowed, false, field);
    assert.equal(state.outcome, 'collision', field);
    assert.ok(state.reason.includes('aliases conflict'), field);
  }
});

fixtureTest('test_independent_release_rollback_and_collision_are_rejected', (self) => {
  self.commitHistoryManifest('v0.1.38', 'v0.2.0');
  const rollback = self.regenerateCandidate('v0.1.37', 0, 'v0.2.0', 'v0.2.0-1');
  const rollbackState = self.classifyIdentity(rollback);
  assert.equal(rollbackState.outcome, 'rollback');
  assert.equal(rollbackState.allowed, false);
  const collision = self.regenerateCandidate('v0.1.38', 0, 'v0.2.0', 'v0.2.0-1');
  const collisionState = self.classifyIdentity(collision);
  assert.equal(collisionState.outcome, 'collision');
  assert.equal(collisionState.allowed, false);
});

// The asset tag advances legally, but the upstream line must not regress.
fixtureTest('test_independent_upstream_rollback_is_rejected', (self) => {
  self.commitHistoryManifest('v0.1.38', 'v0.2.1');
  const identity = self.regenerateCandidate('v0.1.39', 0, 'v0.2.0', 'v0.2.0-1');
  const state = self.classifyIdentity(identity);
  assert.equal(state.allowed, false);
  assert.equal(state.outcome, 'rollback');
  assert.ok(state.reason.includes('upstream transition is backward'), state.reason);
});

fixtureTest('test_stable_to_development_upstream_is_forbidden', (self) => {
  self.commitHistoryManifest('v0.1.38', 'v0.2.0');
  const identity = self.regenerateCandidate('v0.1.39', 0, 'b10600', 'b10600');
  const state = self.classifyIdentity(identity);
  assert.equal(state.allowed, false);
  assert.equal(state.outcome, 'rollback');
  assert.ok(state.reason.includes('upstream transition is forbidden-stable-to-development'), state.reason);
});

fixtureTest('test_new_release_version_must_restart_rebuild_numbering', (self) => {
  self.commitHistoryManifest('v0.1.37', 'b10514');
  const identity = self.regenerateCandidate('v0.1.38-1', 1, 'v0.2.0', 'v0.2.0-1');
  const state = self.classifyIdentity(identity);
  assert.equal(state.allowed, false);
  assert.equal(state.outcome, 'rollback');
});

fixtureTest('test_release_rebuild_must_match_the_release_tag', (self) => {
  const identity = self.regenerateCandidate('v0.1.38', 0, 'v0.2.0', 'v0.2.0-1');
  const mismatched = new CandidateIdentity({ ...identity, releaseRebuild: 1 });
  assert.throws(() => validateCandidate(self.candidate, mismatched), ContractError);
});

fixtureTest('test_candidate_upstream_tag_syntax_stays_exact', (self) => {
  const identity = self.regenerateCandidate('v0.1.38', 0, 'v0.2.0', 'v0.2.0-1');
  for (const invalid of ['main', 'v0.2', 'v0.2.0-1', 'b10514-1', '']) {
    const broken = new CandidateIdentity({ ...identity, upstreamTag: invalid });
    assert.throws(() => validateCandidate(self.candidate, broken), ContractError, invalid);
  }
});

// Bridge assets decouple from upstream; native releases never do.
fixtureTest('test_native_release_tag_must_still_encode_its_upstream', (self) => {
  const accepted = self.regenerateCandidate('v0.1.38', 0, 'v0.2.0', 'v0.2.0-1');
  assert.equal(typeof validateCandidate(self.candidate, accepted), 'string');
  for (const [nativeTag, upstreamTag] of [
    ['v0.1.38', 'v0.2.0'],
    ['v0.2.1-1', 'v0.2.0'],
    ['b10514', 'v0.2.0'],
    ['b10515', 'b10514'],
  ]) {
    const identity = self.regenerateCandidate('v0.1.38', 0, upstreamTag, nativeTag);
    assert.throws(() => validateCandidate(self.candidate, identity), ContractError, `${nativeTag} ${upstreamTag}`);
  }
});

fixtureTest('test_stable_history_ignores_later_development_publication', (self) => {
  self.commitHistoryManifest('v0.2.0', 'v0.2.0');
  self.commitHistoryManifest('b10515', 'b10515');
  const identity = self.regenerateCandidate('v0.2.1', 0, 'v0.2.1');
  assert.equal(self.classifyIdentity(identity).allowed, true);
});

fixtureTest('test_branch_only_is_safely_resumable', (self) => {
  self.publishBranchCandidate();
  const state = self.inspect();
  assert.equal(state.state, 'branch-only');
  assert.equal(state.action, 'publish-tag-and-release');
  assert.equal(state.outcome, 'safely-resumed');
});

fixtureTest('test_same_source_rebuild_may_change_only_manifest', (self) => {
  self.publishBranchCandidate();
  const rebuild = new CandidateIdentity({ ...self.identity, releaseTag: 'v0.2.0-1', releaseRebuild: 1 });
  generate(generateArgs(self.candidate, rebuild));
  copyFileSync(path.join(self.candidate, 'manifest.json'), path.join(self.repository, 'manifest.json'));
  git(self.repository, 'add', 'manifest.json');
  git(self.repository, 'commit', '-q', '-m', 'rebuild');
  const state = classify({
    repository: self.repository,
    candidate: self.candidate,
    identity: rebuild,
    branchCommit: self.head,
    tagCommit: null,
    release: null,
  });
  assert.equal(state.allowed, true);
  assert.equal(state.state, 'branch-only');
});

fixtureTest('test_post_push_release_failure_retries_then_completes', (self) => {
  const candidateCommit = self.publishBranchCandidate();
  git(self.repository, 'tag', 'v0.2.0', candidateCommit);

  const retry = self.inspect({ tag: candidateCommit, release: false });
  assert.equal(retry.state, 'tag-without-release');
  assert.equal(retry.action, 'create-release');
  assert.equal(retry.outcome, 'safely-resumed');

  const complete = self.inspect({ tag: candidateCommit, release: true });
  assert.equal(complete.state, 'complete');
  assert.equal(complete.action, 'none');
  assert.equal(complete.outcome, 'already-complete');
  assert.equal(complete.release_id, 42);
});

fixtureTest('test_rejected_and_ambiguous_push_mutation_outcomes_are_exact', (self) => {
  const before = self.inspect();
  const unchanged = self.inspect();
  assert.equal(publicationStateChanged(before, unchanged), false);

  const candidateCommit = self.publishBranchCandidate();
  git(self.repository, 'tag', 'v0.2.0', candidateCommit);
  const acceptedButClientFailed = self.inspect({ tag: candidateCommit, release: false });
  assert.equal(publicationStateChanged(before, acceptedButClientFailed), true);
});

fixtureTest('test_empty_requery_after_ambiguous_push_is_retryable_mutation_unknown', (self) => {
  const emptyRequery = path.join(path.dirname(self.candidate), 'empty-requery.json');
  writeFileSync(emptyRequery, '');
  const outcome = mutationUnknownFromRequery(self.candidate, self.identity, 'ref-requery-failed', emptyRequery);
  assert.equal(outcome.state, 'mutation-unknown');
  assert.equal(outcome.outcome, 'mutation-unknown');
  assert.equal(outcome.reason_code, 'ref-requery-failed');
  assert.equal(outcome.retryable, true);
  assert.equal(outcome.mutated, null);
  assert.equal(outcome.mutation_status, 'unknown');
  assert.equal(outcome.orchestrator_correlation_id, 'kanban:t_fcc0b814:web-v0.1.38');
  assert.equal(outcome.github_run_id, '123456789');
  assert.equal(outcome.qualification_gates.text_to_speech, AUTOMATED_QUALIFICATION_REQUIRED);
  assert.throws(() => mutationUnknownOutcome(self.candidate, self.identity, 'attacker-value'), ContractError);
  writeFileSync(emptyRequery, dumps({ schema_version: 1, state: 'exact' }));
  assert.throws(
    () => mutationUnknownFromRequery(self.candidate, self.identity, 'ref-requery-failed', emptyRequery),
    ContractError,
  );
});

fixtureTest('test_empty_release_requery_is_retryable_mutation_unknown', (self) => {
  const emptyRequery = path.join(path.dirname(self.candidate), 'empty-release-requery.json');
  writeFileSync(emptyRequery, '');
  const outcome = mutationUnknownFromRequery(self.candidate, self.identity, 'release-requery-failed', emptyRequery);
  assert.equal(outcome.state, 'mutation-unknown');
  assert.equal(outcome.reason_code, 'release-requery-failed');
  assert.equal(outcome.retryable, true);
  assert.equal(outcome.mutated, null);
});

fixtureTest('test_fatal_classifier_requery_is_not_treated_as_exact_remote_state', (self) => {
  const fatalRequery = path.join(path.dirname(self.candidate), 'fatal-requery.json');
  writeFileSync(fatalRequery, dumps(fatal('empty branch identity', self.identity)));
  const outcome = mutationUnknownFromRequery(self.candidate, self.identity, 'ref-requery-failed', fatalRequery);
  assert.equal(outcome.state, 'mutation-unknown');
  assert.equal(outcome.retryable, true);
  assert.equal(outcome.mutated, null);
});

fixtureTest('test_fatal_outcome_preserves_run_and_qualification_provenance', (self) => {
  const outcome = fatal('invalid candidate', self.identity);
  assert.equal(outcome.orchestrator_correlation_id, self.identity.orchestratorCorrelationId);
  assert.equal(outcome.github_run_id, self.identity.githubRunId);
  assert.equal(outcome.github_run_url, self.identity.githubRunUrl);
  assert.equal(outcome.qualification_gates.state_persistence, 'passed');
  assert.equal(outcome.qualification_gates.multimodal, 'passed');
  assert.equal(outcome.qualification_gates.speech_to_text, AUTOMATED_QUALIFICATION_REQUIRED);
  assert.equal(outcome.qualification_gates.text_to_speech, AUTOMATED_QUALIFICATION_REQUIRED);
});

fixtureTest('test_ambiguous_release_create_failure_is_recovered_after_exact_requery', (self) => {
  const candidateCommit = self.publishBranchCandidate();
  git(self.repository, 'tag', 'v0.2.0', candidateCommit);
  let remoteRelease = null;
  let mutationCalls = 0;

  const before = self.inspect({ tag: candidateCommit, release: remoteRelease || false });
  assert.equal(before.action, 'create-release');
  mutationCalls += 1;
  remoteRelease = self.releaseData(candidateCommit);
  // Simulate the server committing the release and the client then failing.
  const after = self.inspect({ tag: candidateCommit, release: remoteRelease });
  assert.equal(after.outcome, 'already-complete');
  assert.equal(mutationCalls, 1);
});

fixtureTest('test_release_create_failure_without_server_commit_remains_retryable', (self) => {
  const candidateCommit = self.publishBranchCandidate();
  git(self.repository, 'tag', 'v0.2.0', candidateCommit);
  const after = self.inspect({ tag: candidateCommit, release: false });
  assert.equal(after.reason_code, 'exact-tag-release-missing');
  assert.equal(after.retryable, true);
});

fixtureTest('test_partial_published_release_is_not_repaired', (self) => {
  const candidateCommit = self.publishBranchCandidate();
  git(self.repository, 'tag', 'v0.2.0', candidateCommit);
  const partialRelease = self.releaseData(candidateCommit);
  const missing = partialRelease.assets.pop();
  const partial = self.inspect({ tag: candidateCommit, release: partialRelease });
  assert.equal(partial.allowed, false);
  assert.equal(partial.state, 'release-assets-partial');
  assert.equal(partial.action, 'none');
  assert.equal(partial.outcome, 'immutable-publication-unverified');
  assert.equal(partial.retryable, false);
  assert.deepEqual(partial.missing_release_assets, [missing.name]);
});

fixtureTest('test_exact_complete_release_remains_idempotent_after_branch_advances', (self) => {
  const candidateCommit = self.publishBranchCandidate();
  git(self.repository, 'tag', 'v0.2.0', candidateCommit);
  writeFileSync(path.join(self.repository, 'README.md'), 'later release\n');
  git(self.repository, 'add', 'README.md');
  git(self.repository, 'commit', '-q', '-m', 'later');
  const state = self.inspect({ tag: candidateCommit, release: true });
  assert.equal(state.outcome, 'already-complete');
});

fixtureTest('test_existing_tag_mismatch_fails_closed_as_collision', (self) => {
  const wrongCommit = self.head;
  const state = self.inspect({ tag: wrongCommit });
  assert.equal(state.allowed, false);
  assert.equal(state.outcome, 'collision');
});

fixtureTest('test_unreachable_exact_tag_fails_closed', (self) => {
  const candidateCommit = self.publishBranchCandidate();
  const previousBranch = git(self.repository, 'rev-parse', `${candidateCommit}^`);
  git(self.repository, 'checkout', '-q', '--detach', previousBranch);
  self.copyCandidateIntoRepository();
  git(self.repository, 'add', '.');
  git(self.repository, 'commit', '-q', '-m', 'orphan-candidate');
  const orphan = self.head;
  git(self.repository, 'tag', 'v0.2.0', orphan);
  const state = classify({
    repository: self.repository,
    candidate: self.candidate,
    identity: self.identity,
    branchCommit: candidateCommit,
    tagCommit: orphan,
    release: null,
  });
  assert.equal(state.allowed, false);
  assert.ok(state.reason.includes('not reachable'), state.reason);
});

fixtureTest('test_branch_only_with_unrelated_diff_fails_closed', (self) => {
  self.copyCandidateIntoRepository();
  writeFileSync(path.join(self.repository, 'UNEXPECTED'), 'injected\n');
  git(self.repository, 'add', '.');
  git(self.repository, 'commit', '-q', '-m', 'candidate-plus-extra');
  const state = self.inspect();
  assert.equal(state.allowed, false);
  assert.ok(state.reason.includes('governed release files'), state.reason);
});

fixtureTest('test_release_without_tag_fails_closed', (self) => {
  const state = self.inspect({ release: true });
  assert.equal(state.allowed, false);
  assert.equal(state.outcome, 'collision');
});

fixtureTest('test_release_metadata_and_asset_digest_mismatch_fail_closed', (self) => {
  const candidateCommit = self.publishBranchCandidate();
  git(self.repository, 'tag', 'v0.2.0', candidateCommit);
  for (const [field, value] of [
    ['prerelease', true],
    ['target_commitish', 'f'.repeat(40)],
    ['id', true],
    ['id', 0],
  ]) {
    const release = self.releaseData(candidateCommit);
    release[field] = value;
    assert.equal(self.inspect({ tag: candidateCommit, release }).allowed, false, field);
  }
  let release = self.releaseData(candidateCommit);
  release.assets[0].digest = `sha256:${'0'.repeat(64)}`;
  assert.equal(self.inspect({ tag: candidateCommit, release }).allowed, false);
  release = self.releaseData(candidateCommit);
  release.body = release.body.replace(self.identity.orchestratorCorrelationId, 'different-run');
  assert.equal(self.inspect({ tag: candidateCommit, release }).allowed, false);
  release = self.releaseData(candidateCommit);
  release.assets.push({ name: 'unexpected.bin', state: 'uploaded', size: 0, digest: `sha256:${sha256(Buffer.alloc(0))}` });
  assert.equal(self.inspect({ tag: candidateCommit, release }).allowed, false);
});

fixtureTest('test_candidate_checksum_mismatch_fails_closed', (self) => {
  writeFileSync(path.join(self.candidate, ARTIFACTS[0]), 'tampered');
  assert.throws(() => classify({
    repository: self.repository,
    candidate: self.candidate,
    identity: self.identity,
    branchCommit: self.head,
    tagCommit: null,
    release: null,
  }), ContractError);
});

fixtureTest('test_candidate_inventory_must_be_exact', (self) => {
  writeFileSync(path.join(self.candidate, 'unexpected.bin'), 'not governed');
  assert.throws(() => validateCandidate(self.candidate, self.identity), ContractError);
});

fixtureTest('test_candidate_manifest_duplicate_keys_fail_closed', (self) => {
  const manifestPath = path.join(self.candidate, 'manifest.json');
  const original = readFileSync(manifestPath, 'utf8');
  const duplicate = original.replace('{\n', `{\n  "release_tag": "${self.identity.releaseTag}",\n`);
  writeFileSync(manifestPath, duplicate, 'utf8');
  assert.throws(() => validateCandidate(self.candidate, self.identity), ContractError);
});

fixtureTest('test_candidate_symlink_is_not_an_immutable_regular_file', (self) => {
  const artifact = path.join(self.candidate, ARTIFACTS[0]);
  const target = path.join(self.temporary, 'outside-artifact');
  writeFileSync(target, readFileSync(artifact));
  unlinkSync(artifact);
  symlinkSync(target, artifact);
  assert.throws(() => validateCandidate(self.candidate, self.identity), ContractError);
  assert.throws(() => candidatePublicationDigests(self.candidate), ContractError);
});

fixtureTest('test_candidate_run_and_gate_provenance_tampering_fails_closed', (self) => {
  for (const [field, value] of [
    ['orchestrator_correlation_id', 'different-run'],
    ['github_run_url', 'https://github.com/attacker/repo/actions/runs/123'],
    ['qualification_gates', { state_persistence: 'passed' }],
    [
      'qualification_gates',
      {
        state_persistence: 'passed',
        multimodal: 'passed',
        speech_to_text: 'passed',
        text_to_speech: 'passed',
      },
    ],
    ['unproven_capabilities', { real_device_playback: 'proven' }],
  ]) {
    const manifestPath = path.join(self.candidate, 'manifest.json');
    const original = readFileSync(manifestPath, 'utf8');
    const manifest = pyJsonLoads(original);
    manifest[field] = value;
    writeFileSync(manifestPath, dumps(manifest), 'utf8');
    assert.throws(() => validateCandidate(self.candidate, self.identity), ContractError, field);
    writeFileSync(manifestPath, original, 'utf8');
  }
});

fixtureTest('test_rollback_has_machine_readable_outcome', (self) => {
  const newer = pyJsonLoads(readFileSync(path.join(self.repository, 'manifest.json'), 'utf8'));
  newer.bridge_assets_tag = 'v0.3.0';
  newer.llama_cpp_tag = 'v0.3.0';
  writeFileSync(path.join(self.repository, 'manifest.json'), `${dumps(newer)}\n`);
  git(self.repository, 'add', 'manifest.json');
  git(self.repository, 'commit', '-q', '-m', 'newer');
  const state = self.inspect();
  assert.equal(state.allowed, false);
  assert.equal(state.state, 'rollback');
  assert.equal(state.outcome, 'rollback');
});

fixtureTest('test_new_upstream_rebuild_is_machine_readable_rollback', (self) => {
  const identity = new CandidateIdentity({
    ...self.identity,
    releaseTag: 'v0.2.1-1',
    releaseRebuild: 1,
    upstreamTag: 'v0.2.1',
    nativeReleaseTag: 'v0.2.1-1',
  });
  generate(generateArgs(self.candidate, identity));
  const state = classify({
    repository: self.repository,
    candidate: self.candidate,
    identity,
    branchCommit: self.head,
    tagCommit: null,
    release: null,
  });
  assert.equal(state.outcome, 'rollback');
  assert.equal(state.reason_code, 'ordering-rollback');
});

fixtureTest('test_published_release_must_be_immutable_and_attested', (self) => {
  const verified = self.verifyPublished();
  assert.equal(verified.immutable, true);
  assert.equal(verified.release_id, 42);
  assert.equal(verified.tag_commit, self.head);
  assert.equal(verified.published_at, '2026-08-20T22:15:59Z');
  assert.equal(verified.attested_purl, `pkg:github/${APPROVED_ASSETS_REPOSITORY}@v0.2.0`);
  assert.deepEqual(verified.attested_assets, candidatePublicationDigests(self.candidate));
});

fixtureTest('test_non_immutable_or_malformed_readback_fails_closed', (self) => {
  const release = self.publishedRelease();
  const { immutable: _immutable, ...withoutImmutable } = release;
  const cases = {
    'tag-readback-immutable-false': { releaseByTag: { ...release, immutable: false } },
    'tag-readback-immutable-missing': { releaseByTag: withoutImmutable },
    'tag-readback-immutable-string': { releaseByTag: { ...release, immutable: 'true' } },
    'id-readback-immutable-false': { releaseById: { ...release, immutable: false } },
    'id-readback-immutable-missing': { releaseById: withoutImmutable },
    'id-readback-is-a-different-release': { releaseById: { ...release, id: 43 } },
    'readbacks-disagree-on-target': { releaseById: { ...release, target_commitish: 'f'.repeat(40) } },
    'readbacks-disagree-on-published-at': { releaseById: { ...release, published_at: '2026-08-20T22:16:00Z' } },
    'classified-release-id-mismatch': { releaseId: 4242 },
    'readback-is-a-draft': { releaseByTag: { ...release, draft: true } },
    'readback-is-unpublished': { releaseByTag: { ...release, published_at: null } },
    'readback-is-another-tag': { releaseByTag: { ...release, tag_name: 'v0.1.38' } },
    'readback-not-an-object': { releaseByTag: [release] },
    'unapproved-assets-repository': { assetsRepo: 'leehack/other-assets' },
    'malformed-tag-commit': { tagCommit: 'not-a-commit' },
  };
  for (const [label, overrides] of Object.entries(cases)) {
    assert.throws(() => self.verifyPublished(overrides), ContractError, label);
  }
});

fixtureTest('test_release_attestation_must_cover_the_published_candidate', (self) => {
  const digests = candidatePublicationDigests(self.candidate);
  const { 'manifest.json': _manifest, ...withoutManifest } = digests;
  const attestation = (overrides) => releaseAttestation({
    releaseTag: 'v0.2.0',
    assetsRepo: APPROVED_ASSETS_REPOSITORY,
    tagCommit: self.head,
    releaseId: 42,
    assets: digests,
    ...overrides,
  });
  const cases = {
    'attestation-missing': { attestation: {} },
    'attestation-null': { attestation: null },
    'attestation-for-another-tag': { attestation: attestation({ releaseTag: 'v0.1.38' }) },
    'attestation-for-another-release-id': { attestation: attestation({ releaseId: 43 }) },
    'attestation-for-another-commit': { attestation: attestation({ tagCommit: 'd'.repeat(40) }) },
    'attestation-omits-an-asset': { attestation: attestation({ assets: withoutManifest }) },
    'attestation-digest-mismatch': { attestation: attestation({ assets: { ...digests, 'manifest.json': 'b'.repeat(64) } }) },
    'attestation-signed-by-an-impostor': {
      attestation: self.publishedAttestation({
        resultOverrides: { signature: { certificate: { subjectAlternativeName: 'https://evil.example' } } },
      }),
    },
  };
  for (const [label, overrides] of Object.entries(cases)) {
    assert.throws(() => self.verifyPublished(overrides), ContractError, label);
  }
});

fixtureTest('test_immutable_publication_cli_reports_exact_failures', (self) => {
  const root = self.temporary;
  const digests = candidatePublicationDigests(self.candidate);

  const run = (payloads) => {
    const paths = {};
    for (const [key, payload] of Object.entries(payloads)) {
      paths[key] = path.join(root, `${key}.json`);
      writeFileSync(paths[key], dumps(payload), 'utf8');
    }
    const result = spawnSync(process.execPath, [
      CLI,
      'verify-immutable-publication',
      '--candidate', self.candidate,
      '--assets-repo', APPROVED_ASSETS_REPOSITORY,
      '--release-tag', 'v0.2.0',
      '--tag-commit', self.head,
      '--release-id', '42',
      '--release-json', paths.release,
      '--release-by-id-json', paths.release_by_id,
      '--attestation-json', paths.attestation,
    ], { encoding: 'utf8' });
    return { returncode: result.status, stdout: result.stdout, stderr: result.stderr };
  };

  const accepted = run({
    release: self.publishedRelease(),
    release_by_id: self.publishedRelease(),
    attestation: self.publishedAttestation(),
  });
  assert.equal(accepted.returncode, 0, accepted.stderr);
  assert.deepEqual(pyJsonLoads(accepted.stdout).attested_assets, digests);

  const rejected = run({
    release: self.publishedRelease({ immutable: false }),
    release_by_id: self.publishedRelease(),
    attestation: self.publishedAttestation(),
  });
  assert.equal(rejected.returncode, 1);
  assert.ok(rejected.stderr.includes('is not immutable'), rejected.stderr);

  const unattested = run({
    release: self.publishedRelease(),
    release_by_id: self.publishedRelease(),
    attestation: {},
  });
  assert.equal(unattested.returncode, 1);
  assert.ok(unattested.stderr.includes('error:'), unattested.stderr);
});
