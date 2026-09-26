// Contract tests for scripts/release/contract.mjs, one test per test method of
// scripts/release_contract_test.py, with the same names and assertions.

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  Channel,
  ContractError,
  IMMUTABLE_RELEASE_ATTESTATION_PREDICATE_TYPE,
  Transition,
  compareReleases,
  compareUpstream,
  expectedNativeArtifacts,
  parseReleaseTag,
  parseUpstreamTag,
  readPreviousManifest,
  requireCorrelationId,
  resolveNativeManifest,
  resolveTagCommit,
  selectStableNativeRelease,
  validateCandidatePrequalification,
  validateGithubPrerelease,
  validateImmutableReleaseGovernance,
  validateNativeFile,
  validateNativeIdentity,
  validateNativeRelease,
  validateNativeRequest,
  validateNewReleaseIdentity,
  validatePublicationEnvironment,
  validateReleaseAttestation,
  validateReleaseIdentity,
  validateReleaseImmutability,
} from '../../scripts/release/contract.mjs';
import { PyException, pyItems, pyJsonDumps, pyJsonLoads } from '../../scripts/release/json.mjs';
import { ASSETS_REPO, RUN_ID, RUN_URL, TAG_COMMIT, releaseAttestation } from './contract_fixtures.mjs';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const CLI = path.join(rootDir, 'scripts/release/contract.mjs');

function runCli(args) {
  const result = spawnSync(process.execPath, [CLI, ...args], { encoding: 'utf8' });
  return { returncode: result.status, stdout: result.stdout, stderr: result.stderr };
}

function withTemporaryDirectory(fn) {
  const directory = mkdtempSync(path.join(tmpdir(), 'release-contract-'));
  try {
    return fn(directory);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

function sha256(data) {
  return createHash('sha256').update(data).digest('hex');
}

// json.dumps(value): Python's default separators, which the fixture files use.
function dumps(value) {
  return pyJsonDumps(value);
}

function clone(value) {
  return structuredClone(value);
}

function without(object, key) {
  const { [key]: _removed, ...rest } = object;
  return rest;
}

test('test_accepts_emission_tags', () => {
  const cases = {
    'v0.2.0': [Channel.STABLE, [0, 2, 0], 0],
    'v0.2.0-3': [Channel.STABLE, [0, 2, 0], 3],
    b10514: [Channel.DEVELOPMENT, [10514], 0],
    'b10514-2': [Channel.DEVELOPMENT, [10514], 2],
  };
  for (const [tag, expected] of Object.entries(cases)) {
    const parsed = parseReleaseTag(tag);
    assert.deepEqual([parsed.channel, [...parsed.versionParts], parsed.rebuild], expected, tag);
    assert.equal(parsed.legacy, false, tag);
  }
});

test('test_github_prerelease_matches_native_policy', () => {
  const expected = { 'v0.2.0': false, 'v0.2.0-1': true, b10514: true, 'b10514-2': true };
  for (const [tag, prerelease] of Object.entries(expected)) {
    assert.equal(parseReleaseTag(tag).githubPrerelease, prerelease, tag);
    assert.equal(validateGithubPrerelease(tag, prerelease), prerelease, tag);
    assert.throws(() => validateGithubPrerelease(tag, !prerelease), ContractError, tag);
  }
  assert.equal(validateGithubPrerelease('v0.2.0-llamadart.1', true), true);
});

test('test_stable_native_discovery_enumerates_wrappers', () => {
  const releases = [
    { tag_name: 'b10599', draft: false, prerelease: true },
    { tag_name: 'v0.2.0', draft: false, prerelease: false },
    { tag_name: 'v0.2.0-2', draft: false, prerelease: true },
    { tag_name: 'v0.3.0-1', draft: false, prerelease: false },
    { tag_name: 'v9.0.0', draft: false, prerelease: true },
    { tag_name: 'v9.0.0', draft: true, prerelease: false },
  ];
  assert.equal(selectStableNativeRelease(releases), 'v0.2.0-2');
});

// int() refuses more than 4300 digits with ValueError, which escapes the
// contract uncaught, as it does in Python.
const TOO_LONG = 'ValueError: Exceeds the limit (4300 digits) for integer string conversion: value has 4301 digits; use sys.set_int_max_str_digits() to increase the limit';

test('a tag component over 4300 digits raises int()\'s ValueError, never passes', () => {
  const ones = '1'.repeat(4301);
  const tooLong = (error) => error instanceof PyException && error.tracebackLine === TOO_LONG;
  for (const tag of [`v${ones}.0.0`, `v1.${ones}.0`, `v1.0.0-${ones}`, `b${ones}`, `b1-${ones}`]) {
    assert.throws(() => parseReleaseTag(tag), tooLong, tag);
  }
  assert.throws(() => parseReleaseTag(`v1.0.0-llamadart.${ones}`, { allowLegacy: true }), tooLong);
  assert.throws(() => parseUpstreamTag(`v${ones}.0.0`), tooLong);
  assert.throws(() => parseUpstreamTag(`b${ones}`), tooLong);
  assert.throws(() => selectStableNativeRelease([
    { tag_name: 'v0.2.0', draft: false, prerelease: false },
    { tag_name: `v${ones}.0.0`, draft: false, prerelease: false },
  ]), tooLong);
  const result = runCli(['validate-release', '--release-tag', `v${ones}.0.0`, '--release-rebuild', '0', '--upstream-tag', 'v1.0.0']);
  assert.deepEqual(result, { returncode: 1, stdout: '', stderr: `Traceback (most recent call last):\n${TOO_LONG}\n` });
});

test('test_correlation_id_rejects_injection_and_ambiguity', () => {
  assert.equal(requireCorrelationId('llamadart-pin:run-123'), 'llamadart-pin:run-123');
  for (const invalid of ['', ' leading', 'two words', 'line\nbreak', 'x'.repeat(129), '../escape']) {
    assert.throws(() => requireCorrelationId(invalid), ContractError, JSON.stringify(invalid));
  }
});

test('test_publication_environment_requires_fail_closed_policy', () => {
  const configured = {
    name: 'bridge-assets-publication',
    can_admins_bypass: false,
    protection_rules: [{ type: 'branch_policy' }],
    deployment_branch_policy: { protected_branches: false, custom_branch_policies: true },
  };
  const branchPolicies = { total_count: 1, branch_policies: [{ name: 'main', type: 'branch' }] };
  assert.equal(validatePublicationEnvironment(configured, branchPolicies), undefined);

  const invalidCases = {
    'required-reviewers': {
      ...configured,
      protection_rules: [
        {
          type: 'required_reviewers',
          prevent_self_review: true,
          reviewers: [{ type: 'User', reviewer: { id: 1233094, login: 'leehack' } }],
        },
        { type: 'branch_policy' },
      ],
    },
    'required-reviewers-with-self-review-allowed': {
      ...configured,
      protection_rules: [
        { type: 'required_reviewers', prevent_self_review: false, reviewers: [] },
        { type: 'branch_policy' },
      ],
    },
    'admin-bypass': { ...configured, can_admins_bypass: true },
    'all-protected-branches': {
      ...configured,
      deployment_branch_policy: { protected_branches: true, custom_branch_policies: false },
    },
    'numeric-deployment-policy': {
      ...configured,
      deployment_branch_policy: { protected_branches: 0, custom_branch_policies: 1 },
    },
    'extra-deployment-policy-key': {
      ...configured,
      deployment_branch_policy: { ...configured.deployment_branch_policy, unexpected: false },
    },
    'wrong-name': { ...configured, name: 'wrong' },
    'malformed-protection-rule': { ...configured, protection_rules: [...configured.protection_rules, null] },
  };
  for (const [label, invalid] of Object.entries(invalidCases)) {
    assert.throws(() => validatePublicationEnvironment(invalid, branchPolicies), ContractError, label);
  }

  for (const invalidPolicies of [
    { total_count: true, branch_policies: [{ name: 'main', type: 'branch' }] },
    { total_count: 0, branch_policies: [] },
    { total_count: 1, branch_policies: [{ name: 'release/*', type: 'branch' }] },
    {
      total_count: 2,
      branch_policies: [{ name: 'main', type: 'branch' }, { name: 'release/*', type: 'branch' }],
    },
  ]) {
    assert.throws(() => validatePublicationEnvironment(configured, invalidPolicies), ContractError, dumps(invalidPolicies));
  }
});

test('test_validate_environment_cli_reports_only_environment_identity', () => {
  const configured = {
    name: 'bridge-assets-publication',
    can_admins_bypass: false,
    protection_rules: [{ type: 'branch_policy' }],
    deployment_branch_policy: { protected_branches: false, custom_branch_policies: true },
  };
  const branchPolicies = { total_count: 1, branch_policies: [{ name: 'main', type: 'branch' }] };
  const result = withTemporaryDirectory((directory) => {
    const environmentPath = path.join(directory, 'environment.json');
    const branchPoliciesPath = path.join(directory, 'branch-policies.json');
    writeFileSync(environmentPath, dumps(configured), 'utf8');
    writeFileSync(branchPoliciesPath, dumps(branchPolicies), 'utf8');
    return runCli([
      'validate-environment',
      '--environment-json', environmentPath,
      '--branch-policies-json', branchPoliciesPath,
    ]);
  });
  assert.equal(result.returncode, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), { environment: 'bridge-assets-publication' });
});

test('test_legacy_wrappers_are_consumption_only', () => {
  for (const tag of ['b10514-llamadart.2', 'v0.2.0-llamadart.1']) {
    assert.throws(() => parseReleaseTag(tag), ContractError, tag);
    assert.equal(parseReleaseTag(tag, { allowLegacy: true }).legacy, true, tag);
  }
});

test('test_rejects_invalid_and_prerelease_tags', () => {
  for (const tag of ['v0.2', 'v0.2.0-0', 'v0.2.0-rc.1', 'v00.2.0', 'b010514', 'b10514-0', 'b10514-1-extra', ' main ']) {
    assert.throws(() => parseReleaseTag(tag), ContractError, tag);
  }
});

test('test_upstream_channels_are_exact', () => {
  assert.deepEqual([...parseUpstreamTag('v0.2.0').parts], [0, 2, 0]);
  assert.deepEqual([...parseUpstreamTag('b10514').parts], [10514]);
  for (const tag of ['v0.2.0-1', 'b10514-1', 'v0.2.0-rc.1']) {
    assert.throws(() => parseUpstreamTag(tag), ContractError, tag);
  }
});

// scripts/ci/verify_ci_reliability.mjs gates llama_cpp.version with the same form.
test('test_ordinary_pin_accepts_either_upstream_channel', () => {
  const pinContents = readFileSync(path.join(rootDir, 'llama_cpp.version'), 'utf8');
  assert.equal(pinContents, 'v0.5.0\n');
  const pin = pinContents.endsWith('\n') ? pinContents.slice(0, -1) : pinContents;
  assert.ok([Channel.STABLE, Channel.DEVELOPMENT].includes(parseUpstreamTag(pin).channel));
  for (const [tag, channel] of [['v0.2.0', Channel.STABLE], ['b10514', Channel.DEVELOPMENT]]) {
    assert.equal(parseUpstreamTag(tag).channel, channel, tag);
  }
  for (const invalid of [
    '', '0.2.0', 'v0.2', 'v0.2.0.1', 'v0.2.0-1', 'V0.2.0', 'b', 'b10514-1', 'b10514\n', 'B10514', 'main',
    'bb4caa7540188872173c44d161602d9271386413',
  ]) {
    assert.throws(() => parseUpstreamTag(invalid), ContractError, JSON.stringify(invalid));
  }
});

test('test_native_request_is_validated_before_network_use', () => {
  const accepted = validateNativeRequest(
    'v0.2.0-1',
    'v0.2.0',
    'bb4caa7540188872173c44d161602d9271386413',
    '2e5d29d7f98f0d71e75d3fa63b7c55f3b2a7933247cc34ea2b1c5e053d142452',
  );
  assert.equal(accepted.baseTag, 'v0.2.0');
  for (const [label, nativeTag, upstreamTag, upstreamCommit, manifestSha256] of [
    ['malformed-tag', 'v0.2.0-rc.1', 'v0.2.0', 'a'.repeat(40), 'b'.repeat(64)],
    ['wrong-upstream', 'v0.2.0-1', 'v0.2.1', 'a'.repeat(40), 'b'.repeat(64)],
    ['malformed-upstream', 'v0.2.0-1', 'main', 'a'.repeat(40), 'b'.repeat(64)],
    ['malformed-commit', 'v0.2.0-1', 'v0.2.0', 'abc123', 'b'.repeat(64)],
    ['malformed-sha256', 'v0.2.0-1', 'v0.2.0', 'a'.repeat(40), 'B'.repeat(64)],
  ]) {
    assert.throws(() => validateNativeRequest(nativeTag, upstreamTag, upstreamCommit, manifestSha256), ContractError, label);
  }
});

test('test_channel_and_rebuild_ordering', () => {
  const cases = [
    ['b10514', 'b10515', Transition.FORWARD],
    ['b10514', 'v0.2.0', Transition.STABLE_MIGRATION],
    ['v0.2.0', 'v0.2.1', Transition.FORWARD],
    ['v0.2.1', 'v0.2.0', Transition.BACKWARD],
    ['v0.2.0', 'b10515', Transition.FORBIDDEN_STABLE_TO_DEVELOPMENT],
    ['v0.2.0', 'v0.2.0-1', Transition.FORWARD],
    ['b10514-llamadart.1', 'b10514-2', Transition.FORWARD],
  ];
  for (const [current, target, expected] of cases) {
    assert.equal(compareReleases(current, target), expected, `${current} -> ${target}`);
  }

  assert.equal(compareUpstream('b10514', 'v0.2.0'), Transition.STABLE_MIGRATION);
  assert.throws(() => compareReleases('v0.2.0', 'v0.2.1-1'), ContractError);
});

// The chief candidate: assets v0.1.37 -> v0.1.38 while upstream b10514 -> v0.2.0.
test('test_release_and_upstream_orderings_are_independent', () => {
  assert.equal(compareReleases('v0.1.37', 'v0.1.38'), Transition.FORWARD);
  assert.equal(compareUpstream('b10514', 'v0.2.0'), Transition.STABLE_MIGRATION);
  assert.equal(compareReleases('v0.1.38', 'v0.1.37'), Transition.BACKWARD);
  assert.equal(compareReleases('v0.1.38', 'v0.1.38'), Transition.EQUAL);
  assert.equal(compareReleases('v0.1.38', 'v0.1.38-1'), Transition.FORWARD);
  assert.equal(compareReleases('v0.1.38-2', 'v0.1.38-1'), Transition.BACKWARD);
  assert.equal(compareUpstream('v0.2.0', 'b10600'), Transition.FORBIDDEN_STABLE_TO_DEVELOPMENT);
  assert.equal(compareUpstream('v0.2.1', 'v0.2.0'), Transition.BACKWARD);
  // A new asset version must restart at rebuild 0 regardless of upstream.
  assert.throws(() => compareReleases('v0.1.37', 'v0.1.38-1'), ContractError);
});

// Bridge assets version independently: v0.1.38 may ship upstream v0.2.0.
test('test_release_identity_is_independent_of_upstream', () => {
  const candidate = validateReleaseIdentity('v0.1.38', 0, 'v0.2.0');
  assert.deepEqual(
    [candidate.tag, candidate.channel, [...candidate.versionParts], candidate.rebuild],
    ['v0.1.38', Channel.STABLE, [0, 1, 38], 0],
  );
  for (const [release, rebuild, upstream] of [
    ['v0.2.0-2', 2, 'v0.2.0'],
    ['v0.1.38', 0, 'b10514'],
    ['b10600', 0, 'v0.2.0'],
    ['v0.1.38-3', 3, 'v0.2.0'],
  ]) {
    assert.equal(validateReleaseIdentity(release, rebuild, upstream).tag, release, `${release} ${upstream}`);
  }
});

test('test_release_identity_keeps_strict_syntax_and_rebuild', () => {
  for (const [label, release, rebuild, upstream] of [
    ['rebuild-mismatch', 'v0.2.0-2', 1, 'v0.2.0'],
    ['rebuild-negative', 'v0.1.38', -1, 'v0.2.0'],
    ['rebuild-zero-mismatch', 'v0.1.38-1', 0, 'v0.2.0'],
    ['malformed-release', 'v0.1', 0, 'v0.2.0'],
    ['prerelease-release', 'v0.1.38-rc.1', 0, 'v0.2.0'],
    ['legacy-release', 'v0.1.38-llamadart.1', 1, 'v0.2.0'],
    ['malformed-upstream', 'v0.1.38', 0, 'main'],
    ['rebuild-bearing-upstream', 'v0.1.38', 0, 'v0.2.0-1'],
  ]) {
    assert.throws(() => validateReleaseIdentity(release, rebuild, upstream), ContractError, label);
  }
});

// npm orders v0.1.50-1 before v0.1.50, so new assets stay unsuffixed.
test('test_new_bridge_release_never_carries_a_rebuild_suffix', () => {
  assert.equal(validateNewReleaseIdentity('v0.1.50', 0, 'v0.5.0').tag, 'v0.1.50');
  for (const [release, rebuild, upstream] of [
    ['v0.1.50-1', 1, 'v0.5.0'],
    ['v0.1.47-1', 1, 'v0.4.1'],
    ['b10600-1', 1, 'b10600'],
  ]) {
    assert.throws(() => validateNewReleaseIdentity(release, rebuild, upstream), ContractError, release);
  }
  // Suffixed tags published before the rule stay readable.
  assert.equal(validateReleaseIdentity('v0.1.47-1', 1, 'v0.4.1').rebuild, 1);
});

// bridge_candidate.yml and publish_assets.yml gate on this command.
test('test_validate_release_cli_rejects_a_new_rebuild_suffix', () => {
  const command = ['validate-release', '--upstream-tag', 'v0.5.0'];
  const accepted = runCli([...command, '--release-tag', 'v0.1.50', '--release-rebuild', '0']);
  assert.equal(accepted.returncode, 0, accepted.stderr);
  const rejected = runCli([...command, '--release-tag', 'v0.1.50-1', '--release-rebuild', '1']);
  assert.notEqual(rejected.returncode, 0);
  assert.ok((rejected.stderr + rejected.stdout).includes('must not carry a rebuild suffix'));
});

test('test_native_identity_still_encodes_its_upstream', () => {
  assert.equal(validateNativeIdentity('v0.2.0-1', 1, 'v0.2.0').tag, 'v0.2.0-1');
  assert.equal(validateNativeIdentity('b10514', 0, 'b10514').tag, 'b10514');
  for (const [label, nativeTag, rebuild, upstream] of [
    ['stable-upstream-mismatch', 'v0.2.0-1', 1, 'v0.2.1'],
    ['independent-versioning-forbidden', 'v0.1.38', 0, 'v0.2.0'],
    ['channel-mismatch', 'b10514', 0, 'v0.2.0'],
    ['development-upstream-mismatch', 'b10514', 0, 'b10515'],
    ['rebuild-mismatch', 'v0.2.0-1', 0, 'v0.2.0'],
    ['negative-rebuild', 'v0.2.0-1', -1, 'v0.2.0'],
  ]) {
    assert.throws(() => validateNativeIdentity(nativeTag, rebuild, upstream), ContractError, label);
  }
});

test('test_native_manifest_provenance_and_checksum', () => {
  const manifest = {
    tag: 'v0.2.0-1',
    native_release_tag: 'v0.2.0-1',
    llama_cpp_tag: 'v0.2.0',
    llama_cpp_commit: 'a'.repeat(40),
    native_commit: 'b'.repeat(40),
  };
  const identity = resolveNativeManifest(manifest, 'v0.2.0-1');
  assert.equal(identity.upstreamTag, 'v0.2.0');
  assert.equal(identity.nativeCommit, 'b'.repeat(40));

  withTemporaryDirectory((directory) => {
    const file = path.join(directory, 'assets.json');
    writeFileSync(file, dumps(manifest), 'utf8');
    const digest = sha256(readFileSync(file));
    const validated = validateNativeFile(file, digest, 'v0.2.0-1', 'v0.2.0', 'a'.repeat(40));
    assert.deepEqual(validated, identity);
    assert.throws(() => validateNativeFile(file, '0'.repeat(64), 'v0.2.0-1', 'v0.2.0', 'a'.repeat(40)), ContractError);
    const duplicate = readFileSync(file, 'utf8').replace('{', `{"native_commit":"${'b'.repeat(40)}",`);
    writeFileSync(file, duplicate, 'utf8');
    const duplicateDigest = sha256(readFileSync(file));
    assert.throws(() => validateNativeFile(file, duplicateDigest, 'v0.2.0-1', 'v0.2.0', 'a'.repeat(40)), ContractError);
  });
});

// Build the real v0.2.0-1 release shape with a branch target.
function nativeReleaseFixture(directory, tag = 'v0.2.0-1', upstreamTag = 'v0.2.0') {
  const upstreamCommit = 'bb4caa7540188872173c44d161602d9271386413';
  const nativeCommit = 'e5c240e34b525da953ed98dc743516eef78cb738';
  const expected = expectedNativeArtifacts(tag);
  const artifacts = [];
  const checksumLines = [];
  const githubAssets = [];
  pyItems(expected).forEach(([name, [platform, arch, backend, module]], index) => {
    const digest = sha256(`artifact-${index}`);
    const size = index + 100;
    artifacts.push({ module, platform, arch, backend, file: name, sha256: digest, size });
    checksumLines.push(`${digest}  ${name}`);
    githubAssets.push({ name, state: 'uploaded', size, digest: `sha256:${digest}` });
  });
  const manifest = {
    tag,
    llama_cpp_tag: upstreamTag,
    llama_cpp_commit: upstreamCommit,
    native_commit: nativeCommit,
    generated_at: '2026-08-22T00:00:00Z',
    hook_contract_version: 1,
    artifacts,
  };
  const manifestPath = path.join(directory, 'assets.json');
  const checksumsPath = path.join(directory, 'SHA256SUMS');
  writeFileSync(manifestPath, dumps(manifest), 'utf8');
  writeFileSync(checksumsPath, `${checksumLines.join('\n')}\n`, 'utf8');
  const manifestDigest = sha256(readFileSync(manifestPath));
  const checksumDigest = sha256(readFileSync(checksumsPath));
  githubAssets.push(
    { name: 'assets.json', state: 'uploaded', size: statSync(manifestPath).size, digest: `sha256:${manifestDigest}` },
    { name: 'SHA256SUMS', state: 'uploaded', size: statSync(checksumsPath).size, digest: `sha256:${checksumDigest}` },
  );
  return {
    tag,
    upstreamTag,
    upstreamCommit,
    nativeCommit,
    manifest,
    manifestPath,
    checksumsPath,
    manifestDigest,
    release: { tag_name: tag, draft: false, prerelease: true, target_commitish: 'main', assets: githubAssets },
  };
}

test('test_native_release_verifies_github_digest_hook_and_inventory', () => {
  withTemporaryDirectory((directory) => {
    const fixture = nativeReleaseFixture(directory);
    const { tag, manifest, manifestPath, checksumsPath, manifestDigest, release } = fixture;
    const identity = validateNativeRelease(
      manifestPath, checksumsPath, release, manifestDigest,
      tag, fixture.upstreamTag, fixture.upstreamCommit, fixture.nativeCommit,
    );
    assert.equal(identity.nativeCommit, fixture.nativeCommit);
    for (const mutation of ['hook', 'digest', 'inventory', 'prerelease']) {
      const badManifest = clone(manifest);
      const badRelease = clone(release);
      let badDigest;
      if (mutation === 'hook') {
        badManifest.hook_contract_version = 2;
        writeFileSync(manifestPath, dumps(badManifest), 'utf8');
        badDigest = sha256(readFileSync(manifestPath));
        badRelease.assets.at(-2).digest = `sha256:${badDigest}`;
        badRelease.assets.at(-2).size = statSync(manifestPath).size;
      } else if (mutation === 'digest') {
        badRelease.assets[0].digest = `sha256:${'0'.repeat(64)}`;
        badDigest = manifestDigest;
      } else {
        if (mutation === 'inventory') badRelease.assets.shift();
        else badRelease.prerelease = false;
        badDigest = manifestDigest;
      }
      assert.throws(() => validateNativeRelease(
        manifestPath, checksumsPath, badRelease, badDigest,
        tag, fixture.upstreamTag, fixture.upstreamCommit, fixture.nativeCommit,
      ), ContractError, mutation);
      writeFileSync(manifestPath, dumps(manifest), 'utf8');
    }
  });
});

test('test_native_release_trusts_only_the_resolved_immutable_tag_commit', () => {
  withTemporaryDirectory((directory) => {
    const fixture = nativeReleaseFixture(directory);
    const validate = (release, nativeTagCommit) => validateNativeRelease(
      fixture.manifestPath,
      fixture.checksumsPath,
      release,
      fixture.manifestDigest,
      fixture.tag,
      fixture.upstreamTag,
      fixture.upstreamCommit,
      nativeTagCommit,
    );

    const branchTarget = fixture.release;
    assert.equal(branchTarget.target_commitish, 'main');
    validate(branchTarget, fixture.nativeCommit);
    validate({ ...branchTarget, target_commitish: fixture.nativeCommit }, fixture.nativeCommit);

    for (const [label, nativeTagCommit] of [
      ['mismatch', 'c'.repeat(40)],
      ['missing', null],
      ['empty', ''],
      ['branch-name', 'main'],
      ['short', 'b'.repeat(39)],
      ['uppercase', 'B'.repeat(40)],
      ['non-string', 0],
    ]) {
      assert.throws(() => validate(branchTarget, nativeTagCommit), ContractError, label);
    }

    for (const [label, targetCommitish] of [
      ['other-commit', 'c'.repeat(40)],
      ['missing', null],
      ['empty', ''],
      ['abbreviated-commit', 'c'.repeat(12)],
      ['uppercase-commit', 'C'.repeat(40)],
      ['whitespace', ' main'],
      ['tag-ref', 'refs/tags/v0.1.0'],
      ['tag-shaped', 'v0.1.0'],
    ]) {
      assert.throws(
        () => validate({ ...branchTarget, target_commitish: targetCommitish }, fixture.nativeCommit),
        ContractError,
        label,
      );
    }
  });
});

test('test_resolve_tag_commit_peels_annotated_tags', () => {
  const upstreamAnnotated = '8a35040e02747e136d901793604572c7ca6d0793\trefs/tags/v0.2.0\n'
    + 'bb4caa7540188872173c44d161602d9271386413\trefs/tags/v0.2.0^{}\n';
  assert.equal(resolveTagCommit(upstreamAnnotated, 'v0.2.0'), 'bb4caa7540188872173c44d161602d9271386413');
  const annotated = '246e18e254d74452a32210992cefbcab8dc65010\trefs/tags/v0.2.0-1\n'
    + 'e5c240e34b525da953ed98dc743516eef78cb738\trefs/tags/v0.2.0-1^{}\n';
  assert.equal(resolveTagCommit(annotated, 'v0.2.0-1'), 'e5c240e34b525da953ed98dc743516eef78cb738');
  const lightweight = '246e18e254d74452a32210992cefbcab8dc65010\trefs/tags/v0.2.0-1\n';
  assert.equal(resolveTagCommit(lightweight, 'v0.2.0-1'), '246e18e254d74452a32210992cefbcab8dc65010');

  const b = 'b'.repeat(40);
  const c = 'c'.repeat(40);
  for (const [label, output] of [
    ['missing', ''],
    ['blank-lines-only', '\n\n'],
    ['unrelated-ref', `${b}\trefs/tags/v0.2.0-10\n`],
    ['branch-ref', `${b}\trefs/heads/main\n`],
    ['malformed-line', 'not-a-sha\trefs/tags/v0.2.0-1\n'],
    ['space-separated', `${b} refs/tags/v0.2.0-1\n`],
    ['duplicate', `${b}\trefs/tags/v0.2.0-1\n`.repeat(2)],
    ['conflicting', `${b}\trefs/tags/v0.2.0-1\n${c}\trefs/tags/v0.2.0-1\n`],
    ['peeled-only', `${c}\trefs/tags/v0.2.0-1^{}\n`],
    ['same-object-and-commit', `${b}\trefs/tags/v0.2.0-1\n${b}\trefs/tags/v0.2.0-1^{}\n`],
    ['embedded-blank', `${b}\trefs/tags/v0.2.0-1\n\n`],
  ]) {
    assert.throws(() => resolveTagCommit(output, 'v0.2.0-1'), ContractError, label);
  }

  assert.throws(() => resolveTagCommit(lightweight, 'v0.2.0-1;touch-pwned'), ContractError);
});

test('test_resolve_tag_commit_cli_emits_only_the_peeled_commit', () => {
  withTemporaryDirectory((directory) => {
    const refs = path.join(directory, 'tag-refs.txt');
    writeFileSync(
      refs,
      '246e18e254d74452a32210992cefbcab8dc65010\trefs/tags/v0.2.0-1\n'
        + 'e5c240e34b525da953ed98dc743516eef78cb738\trefs/tags/v0.2.0-1^{}\n',
      'utf8',
    );
    const command = ['resolve-tag-commit', '--ls-remote', refs, '--tag', 'v0.2.0-1'];
    const result = runCli(command);
    assert.equal(result.returncode, 0, result.stderr);
    assert.equal(result.stdout.trim(), 'e5c240e34b525da953ed98dc743516eef78cb738');

    writeFileSync(refs, '', 'utf8');
    const failure = runCli(command);
    assert.notEqual(failure.returncode, 0);
  });
});

test('test_rejects_mismatched_native_identity', () => {
  const base = {
    tag: 'v0.2.0-1',
    native_release_tag: 'v0.2.0-1',
    llama_cpp_tag: 'v0.2.0',
    llama_cpp_commit: 'a'.repeat(40),
    native_commit: 'b'.repeat(40),
  };
  const invalid = [
    [{ ...base, tag: 'v0.2.0' }, 'v0.2.0-1'],
    [{ ...base, llama_cpp_tag: 'v0.2.1' }, 'v0.2.0-1'],
    [{ ...base, llama_cpp_commit: 'abc123' }, 'v0.2.0-1'],
    [{ ...base, native_commit: 'B'.repeat(40) }, 'v0.2.0-1'],
  ];
  for (const [manifest, tag] of invalid) {
    assert.throws(() => resolveNativeManifest(manifest, tag), ContractError, dumps(manifest));
  }
});

test('test_immutable_release_governance_must_be_enabled', () => {
  assert.deepEqual(
    validateImmutableReleaseGovernance({ enabled: true, enforced_by_owner: false }, ASSETS_REPO),
    { repository: ASSETS_REPO, enabled: true, enforced_by_owner: false },
  );
  assert.equal(
    validateImmutableReleaseGovernance({ enabled: true, enforced_by_owner: true }, ASSETS_REPO).enforced_by_owner,
    true,
  );
  const invalid = {
    disabled: { enabled: false, enforced_by_owner: false },
    'missing-enabled': { enforced_by_owner: false },
    'missing-enforced': { enabled: true },
    'extra-field': { enabled: true, enforced_by_owner: false, enforced_by_enterprise: false },
    'string-true': { enabled: 'true', enforced_by_owner: false },
    'numeric-true': { enabled: 1, enforced_by_owner: false },
    'null-enabled': { enabled: null, enforced_by_owner: false },
    'string-enforced': { enabled: true, enforced_by_owner: 'false' },
    empty: {},
    'not-an-object': [{ enabled: true, enforced_by_owner: false }],
    'null-body': null,
  };
  for (const [label, payload] of Object.entries(invalid)) {
    assert.throws(() => validateImmutableReleaseGovernance(payload, ASSETS_REPO), ContractError, label);
  }
  assert.throws(
    () => validateImmutableReleaseGovernance({ enabled: true, enforced_by_owner: false }, 'not-a-repository'),
    ContractError,
  );
});

test('test_immutable_release_governance_cli_rejects_malformed_json', () => {
  withTemporaryDirectory((directory) => {
    const governance = path.join(directory, 'governance.json');
    const command = [
      'validate-immutable-release-governance',
      '--governance-json', governance,
      '--repository', ASSETS_REPO,
    ];
    writeFileSync(governance, dumps({ enabled: true, enforced_by_owner: false }), 'utf8');
    const accepted = runCli(command);
    assert.equal(accepted.returncode, 0, accepted.stderr);
    assert.equal(pyJsonLoads(accepted.stdout).enabled, true);

    for (const [label, contents] of Object.entries({
      'invalid-json': '{',
      'duplicate-enabled': '{"enabled":true,"enabled":false,"enforced_by_owner":false}',
      'wrong-root': '[]',
    })) {
      writeFileSync(governance, contents, 'utf8');
      const rejected = runCli(command);
      assert.notEqual(rejected.returncode, 0, label);
      assert.ok(rejected.stderr.includes('error:'), label);
    }
  });
});

test('test_candidate_prequalification_binds_true_governance_assertion', () => {
  const fingerprint = 'b'.repeat(64);
  const harness = 'c'.repeat(64);
  const correlation = 'kanban:t_7f112b91:web-v0.1.39';
  const bridge = 'd'.repeat(40);
  const native = 'e'.repeat(40);
  const record = {
    schema_version: 1,
    candidate_fingerprint: fingerprint,
    harness_source_sha256: harness,
    assets_immutable_releases_enabled: true,
    orchestrator_correlation_id: correlation,
    github_run_id: RUN_ID,
    github_run_url: RUN_URL,
    bridge_source_sha: bridge,
    release_tag: 'v0.1.39',
    emscripten_version: '6.0.8',
    native_commit: native,
    hosted_gates: { state_persistence: 'success', multimodal: 'success' },
    heavy_gates: {
      speech_to_text: 'pending-automated-qualification',
      text_to_speech: 'pending-automated-qualification',
    },
    unproven_capabilities: {
      hardware_gpu_acceleration: 'unavailable-on-hosted-runners',
      real_device_intelligibility: 'unproven',
      real_device_playback: 'unproven',
      speaker_reference_fidelity: 'unproven',
      wasm32_text_to_speech: 'unsupported',
    },
  };
  const argumentsForRecord = {
    candidateFingerprint: fingerprint,
    harnessSourceSha256: harness,
    orchestratorCorrelationId: correlation,
    githubRunId: RUN_ID,
    githubRunUrl: RUN_URL,
    bridgeSourceSha: bridge,
    releaseTag: 'v0.1.39',
    emscriptenVersion: '6.0.8',
    nativeCommit: native,
  };
  assert.equal(
    validateCandidatePrequalification(record, argumentsForRecord).assets_immutable_releases_enabled,
    true,
  );
  for (const [label, broken] of Object.entries({
    'governance-false': { ...record, assets_immutable_releases_enabled: false },
    'governance-missing': without(record, 'assets_immutable_releases_enabled'),
    'governance-string': { ...record, assets_immutable_releases_enabled: 'true' },
    'wrong-run': { ...record, github_run_id: '987654321' },
    'wrong-fingerprint': { ...record, candidate_fingerprint: 'f'.repeat(64) },
    'boolean-schema-version': { ...record, schema_version: true },
    'failed-hosted-gate': { ...record, hosted_gates: { state_persistence: 'success', multimodal: 'failure' } },
  })) {
    assert.throws(() => validateCandidatePrequalification(broken, argumentsForRecord), ContractError, label);
  }
});

test('test_created_release_must_read_back_immutable', () => {
  const release = {
    tag_name: 'v0.1.39',
    id: 4242,
    draft: false,
    prerelease: false,
    target_commitish: TAG_COMMIT,
    published_at: '2026-08-20T22:15:59Z',
    immutable: true,
  };
  assert.equal(validateReleaseImmutability(release, { releaseTag: 'v0.1.39', tagCommit: TAG_COMMIT }), 4242);
  assert.equal(
    validateReleaseImmutability(release, { releaseTag: 'v0.1.39', tagCommit: TAG_COMMIT, releaseId: 4242 }),
    4242,
  );

  const invalid = {
    'immutable-false': { ...release, immutable: false },
    'immutable-missing': without(release, 'immutable'),
    'immutable-null': { ...release, immutable: null },
    'immutable-string': { ...release, immutable: 'true' },
    'immutable-numeric': { ...release, immutable: 1 },
    draft: { ...release, draft: true },
    'draft-missing': without(release, 'draft'),
    'wrong-tag': { ...release, tag_name: 'v0.1.38' },
    'wrong-prerelease': { ...release, prerelease: true },
    'wrong-target': { ...release, target_commitish: 'f'.repeat(40) },
    'missing-target': without(release, 'target_commitish'),
    'missing-published-at': without(release, 'published_at'),
    'malformed-published-at': { ...release, published_at: 'yesterday' },
    'missing-id': without(release, 'id'),
    'boolean-id': { ...release, id: true },
    'zero-id': { ...release, id: 0 },
    'not-an-object': [release],
  };
  for (const [label, payload] of Object.entries(invalid)) {
    assert.throws(
      () => validateReleaseImmutability(payload, { releaseTag: 'v0.1.39', tagCommit: TAG_COMMIT }),
      ContractError,
      label,
    );
  }
  assert.throws(
    () => validateReleaseImmutability(release, { releaseTag: 'v0.1.39', tagCommit: TAG_COMMIT, releaseId: 9999 }),
    ContractError,
  );
});

test('a deeply nested verified statement is compared with its signed payload, as Python compares it', () => {
  const assets = { 'manifest.json': 'b'.repeat(64), 'sha256sums.txt': 'c'.repeat(64) };
  const expectation = { assetsRepo: ASSETS_REPO, releaseTag: 'v0.1.39', tagCommit: TAG_COMMIT, releaseId: 1, expectedAssets: assets };
  for (const depth of [3000, 9000]) {
    // Python 3.12 compares both depths; recursive JS overflowed near 3000.
    const nested = (leaf) => {
      let value = [leaf];
      for (let level = 1; level < depth; level += 1) value = [value];
      return value;
    };
    const payload = releaseAttestation({ assets, statementOverrides: { extension: nested(1) } });
    assert.deepEqual(validateReleaseAttestation(payload, expectation).assets, assets, String(depth));
    payload.verificationResult.statement.extension = nested(2);
    assert.throws(() => validateReleaseAttestation(payload, expectation),
      (error) => error instanceof ContractError && error.message === 'verified statement does not match the signed attestation payload');
  }
});

test('test_release_attestation_binds_the_exact_published_release', () => {
  const assets = { 'manifest.json': 'b'.repeat(64), 'sha256sums.txt': 'c'.repeat(64) };
  const payload = releaseAttestation({ assets });
  const expectation = {
    assetsRepo: ASSETS_REPO,
    releaseTag: 'v0.1.39',
    tagCommit: TAG_COMMIT,
    releaseId: 1,
    expectedAssets: assets,
  };
  const verified = validateReleaseAttestation(payload, expectation);
  assert.equal(verified.purl, `pkg:github/${ASSETS_REPO}@v0.1.39`);
  assert.deepEqual(verified.assets, assets);
  assert.equal(verified.predicate_type, IMMUTABLE_RELEASE_ATTESTATION_PREDICATE_TYPE);
  assert.deepEqual(verified.verified_timestamps, [
    { type: 'TimestampAuthority', uri: 'timestamp.githubapp.com', timestamp: '2026-08-20T22:15:59Z' },
  ]);

  const statementCase = (overrides) => releaseAttestation({ assets, statementOverrides: overrides });

  const { statement } = payload.verificationResult;
  const invalid = {
    'wrong-predicate-type': statementCase({ predicateType: 'https://slsa.dev/provenance/v1' }),
    'missing-predicate-type': statementCase({ predicateType: null }),
    'wrong-statement-type': statementCase({ _type: 'https://in-toto.io/Statement/v0.1' }),
    'predicate-names-other-repository': statementCase({
      predicate: { ...statement.predicate, repository: 'leehack/other' },
    }),
    'predicate-names-other-tag': statementCase({ predicate: { ...statement.predicate, tag: 'v0.1.38' } }),
    'predicate-missing-release-id': statementCase({ predicate: without(statement.predicate, 'databaseId') }),
    'predicate-malformed-release-id': statementCase({ predicate: { ...statement.predicate, databaseId: '01' } }),
    'predicate-not-an-object': statementCase({ predicate: 'release' }),
    'no-release-subject': statementCase({
      subject: Object.entries(assets).map(([name, digest]) => ({ name, digest: { sha256: digest } })),
    }),
    'two-release-subjects': statementCase({ subject: [...statement.subject, statement.subject[0]] }),
    'release-subject-wrong-commit': statementCase({
      subject: [
        { uri: `pkg:github/${ASSETS_REPO}@v0.1.39`, digest: { sha1: 'd'.repeat(40) } },
        ...statement.subject.slice(1),
      ],
    }),
    'duplicate-asset-subject': statementCase({ subject: [...statement.subject, statement.subject[1]] }),
    'asset-digest-mismatch': statementCase({
      subject: [
        statement.subject[0],
        { name: 'manifest.json', digest: { sha256: 'e'.repeat(64) } },
        statement.subject[2],
      ],
    }),
    'asset-missing-from-attestation': statementCase({ subject: statement.subject.slice(0, 2) }),
    'unexpected-asset-in-attestation': statementCase({
      subject: [...statement.subject, { name: 'unexpected.bin', digest: { sha256: 'f'.repeat(64) } }],
    }),
    'malformed-asset-digest': statementCase({
      subject: [
        statement.subject[0],
        { name: 'manifest.json', digest: { sha256: 'not-a-digest' } },
        statement.subject[2],
      ],
    }),
    'subject-list-empty': statementCase({ subject: [] }),
    'subject-not-a-list': statementCase({ subject: { name: 'manifest.json' } }),
    'asset-subject-with-null-uri': statementCase({
      subject: [statement.subject[0], { ...statement.subject[1], uri: null }, statement.subject[2]],
    }),
  };
  invalid['untrusted-signer'] = releaseAttestation({
    assets,
    resultOverrides: { signature: { certificate: { subjectAlternativeName: 'https://evil.example/attester' } } },
  });
  invalid['no-verified-timestamp'] = releaseAttestation({ assets, resultOverrides: { verifiedTimestamps: [] } });
  invalid['incomplete-verified-timestamp'] = releaseAttestation({
    assets,
    resultOverrides: { verifiedTimestamps: [{ uri: '', timestamp: '' }] },
  });
  invalid['malformed-verified-timestamp'] = releaseAttestation({
    assets,
    resultOverrides: {
      verifiedTimestamps: [{ type: 'TimestampAuthority', uri: 'timestamp.githubapp.com', timestamp: 'not-a-timestamp' }],
    },
  });
  invalid['wrong-verified-signer-policy'] = releaseAttestation({
    assets,
    resultOverrides: { verifiedIdentity: { subjectAlternativeName: { subjectAlternativeName: '', regexp: '.*' } } },
  });
  invalid['missing-verification-result'] = { attestation: payload.attestation };
  invalid['missing-attestation'] = { verificationResult: payload.verificationResult };

  const tampered = clone(payload);
  tampered.verificationResult.statement.predicate.tag = 'v0.1.39';
  tampered.attestation.bundle.dsseEnvelope.payload = Buffer.from(
    dumps({ _type: 'https://in-toto.io/Statement/v1' }),
    'utf8',
  ).toString('base64');
  invalid['signed-payload-disagrees-with-verified-statement'] = tampered;

  const unsigned = clone(payload);
  unsigned.attestation.bundle.dsseEnvelope.payloadType = 'text/plain';
  invalid['wrong-dsse-payload-type'] = unsigned;

  const noSignature = clone(payload);
  noSignature.attestation.bundle.dsseEnvelope.signatures = [];
  invalid['missing-dsse-signature'] = noSignature;

  const malformedSignature = clone(payload);
  malformedSignature.attestation.bundle.dsseEnvelope.signatures = [{ sig: '!!!' }];
  invalid['malformed-dsse-signature'] = malformedSignature;

  const undecodable = clone(payload);
  undecodable.attestation.bundle.dsseEnvelope.payload = '!!!';
  invalid['undecodable-dsse-payload'] = undecodable;

  const notABundle = clone(payload);
  notABundle.attestation.bundle.mediaType = 'application/json';
  invalid['not-a-sigstore-bundle'] = notABundle;

  const unsupportedBundle = clone(payload);
  unsupportedBundle.attestation.bundle.mediaType = 'application/vnd.dev.sigstore.bundle.attacker+json';
  invalid['unsupported-sigstore-bundle-version'] = unsupportedBundle;

  const missingTimestampMaterial = clone(payload);
  delete missingTimestampMaterial.attestation.bundle.verificationMaterial.timestampVerificationData;
  invalid['missing-signed-timestamp-material'] = missingTimestampMaterial;

  for (const [label, candidate] of Object.entries(invalid)) {
    assert.throws(() => validateReleaseAttestation(candidate, expectation), ContractError, label);
  }

  for (const [label, overrides] of Object.entries({
    'attestation-for-another-repository': { assetsRepo: 'leehack/other-repo' },
    'attestation-for-another-tag': { releaseTag: 'v0.1.38' },
    'attestation-for-another-commit': { tagCommit: 'f'.repeat(40) },
    'attestation-for-another-release-id': { releaseId: 9999 },
  })) {
    assert.throws(() => validateReleaseAttestation(payload, { ...expectation, ...overrides }), ContractError, label);
  }
});

test('test_reads_current_and_legacy_previous_manifest', () => {
  withTemporaryDirectory((directory) => {
    const current = path.join(directory, 'current.json');
    writeFileSync(
      current,
      dumps({ release_tag: 'v0.2.0-1', upstream_tag: 'v0.2.0', bridge_commit: 'c'.repeat(40) }),
      'utf8',
    );
    assert.deepEqual(readPreviousManifest(current), ['v0.2.0-1', 'v0.2.0', 'c'.repeat(40)]);

    const legacy = path.join(directory, 'legacy.json');
    writeFileSync(
      legacy,
      dumps({ bridge_assets_tag: 'b10514-llamadart.1', llama_cpp_tag: 'b10514', source_commit: 'd'.repeat(40) }),
      'utf8',
    );
    assert.equal(readPreviousManifest(legacy)[0], 'b10514-llamadart.1');

    for (const [field, value] of [
      ['bridge_assets_tag', 'v0.2.1'],
      ['llama_cpp_tag', 'v0.2.1'],
      ['source_commit', 'd'.repeat(40)],
    ]) {
      const conflicting = path.join(directory, `conflicting-${field}.json`);
      writeFileSync(
        conflicting,
        dumps({ release_tag: 'v0.2.0-1', upstream_tag: 'v0.2.0', bridge_commit: 'c'.repeat(40), [field]: value }),
        'utf8',
      );
      assert.throws(() => readPreviousManifest(conflicting), ContractError, field);
    }
  });
});
