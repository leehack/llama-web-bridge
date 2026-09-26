// Tests of scripts/release/orchestrator/native.mjs, one test per test method of
// scripts/release_orchestrator_native_test.py, with the same names and
// assertions. Each scan writes into its own temporary directory.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

import { Channel, ContractError, NATIVE_REPOSITORY } from '../../../scripts/release/contract.mjs';
import { pyJsonDumps } from '../../../scripts/release/json.mjs';
import * as orchestrator from '../../../scripts/release/orchestrator/cli.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as native from '../../../scripts/release/orchestrator/native.mjs';
import {
  BRIDGE_SHA,
  NATIVE_COMMIT,
  NATIVE_MANIFEST_SHA,
  NATIVE_PUBLISHED_AT,
  UPSTREAM_COMMIT,
  makeProvenance,
  nativeManifest,
} from './fixtures.mjs';

function withTemporaryDirectory(fn) {
  const directory = fs.mkdtempSync(path.join(tmpdir(), 'sro-native-'));
  try {
    return fn(directory);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
}

function scan(manifestPath, { nativeReleaseTag = 'v0.2.0', channel = 'stable' } = {}) {
  return native.scanNativeProvenance({
    manifestPath,
    nativeReleaseTag,
    bridgeSourceSha: BRIDGE_SHA,
    bridgeBuildSha: BRIDGE_SHA,
    channel,
    nativeReleasePublishedAt: NATIVE_PUBLISHED_AT,
  });
}

// --- ProvenanceTest --------------------------------------------------------------

test('test_scan_native_binds_exact_manifest_bytes', () => {
  const payload = Buffer.from(pyJsonDumps(nativeManifest()), 'utf8');
  const provenance = withTemporaryDirectory((directory) => {
    const manifestPath = path.join(directory, 'assets.json');
    fs.writeFileSync(manifestPath, payload);
    return scan(manifestPath);
  });
  assert.equal(provenance.upstreamTag, 'v0.2.0');
  assert.equal(provenance.nativeReleaseTag, 'v0.2.0');
  assert.equal(provenance.nativeManifestSha256, createHash('sha256').update(payload).digest('hex'));
});

test('test_malformed_native_manifest_fails_closed', () => {
  withTemporaryDirectory((directory) => {
    const manifestPath = path.join(directory, 'assets.json');
    fs.writeFileSync(manifestPath, '{"native_release_tag": "v0.2.0", "native_release_tag": "v0.2.0"}');
    assert.throws(() => scan(manifestPath), ContractError);
  });
});

test('test_channel_inconsistent_provenance_is_rejected', () => {
  assert.throws(() => makeProvenance({ upstreamTag: 'b9165' }), ContractError);
  assert.throws(() => makeProvenance({ nativeReleaseTag: 'b9165' }), ContractError);
});

test('test_native_release_timestamp_is_canonical_and_bounded', () => {
  assert.throws(() => makeProvenance({ nativeReleasePublishedAt: 'yesterday' }), ContractError);
});

// --- DevelopmentScanTest: manual development scans stay supported, and stay
// scan-only. ------------------------------------------------------------------------

function scanDevelopment(channel) {
  return withTemporaryDirectory((directory) => {
    const manifestPath = path.join(directory, 'assets.json');
    fs.writeFileSync(manifestPath, pyJsonDumps(nativeManifest({ native_release_tag: 'b9165', llama_cpp_tag: 'b9165' })));
    return scan(manifestPath, { nativeReleaseTag: 'b9165', channel });
  });
}

test('test_development_scan_still_prepares_exact_provenance', () => {
  const provenance = scanDevelopment('development');
  assert.equal(provenance.upstreamTag, 'b9165');
  assert.equal(provenance.nativeReleaseTag, 'b9165');
  assert.equal(provenance.channel, Channel.DEVELOPMENT);
});

test('test_stable_scan_of_a_development_release_fails_closed', () => {
  assert.throws(() => scanDevelopment('stable'), ContractError);
});

test('test_development_scan_of_a_stable_release_fails_closed', () => {
  withTemporaryDirectory((directory) => {
    const manifestPath = path.join(directory, 'assets.json');
    fs.writeFileSync(manifestPath, pyJsonDumps(nativeManifest()));
    assert.throws(() => scan(manifestPath, { channel: 'development' }), ContractError);
  });
});

test('test_orchestration_refuses_a_development_provenance', () => {
  const development = new model.NativeProvenance({
    bridgeSourceSha: BRIDGE_SHA,
    bridgeBuildSha: BRIDGE_SHA,
    upstreamTag: 'b9165',
    upstreamCommit: UPSTREAM_COMMIT,
    nativeRepo: NATIVE_REPOSITORY,
    nativeReleaseTag: 'b9165',
    nativeCommit: NATIVE_COMMIT,
    nativeManifestSha256: NATIVE_MANIFEST_SHA,
    nativeReleasePublishedAt: NATIVE_PUBLISHED_AT,
  });
  assert.throws(() => model.requireStableProvenance(development), ContractError);
  withTemporaryDirectory((directory) => {
    const provenancePath = path.join(directory, 'provenance.json');
    fs.writeFileSync(provenancePath, pyJsonDumps({
      bridge_source_sha: BRIDGE_SHA,
      bridge_build_sha: BRIDGE_SHA,
      upstream_tag: 'b9165',
      upstream_commit: UPSTREAM_COMMIT,
      native_repo: NATIVE_REPOSITORY,
      native_release_tag: 'b9165',
      native_commit: NATIVE_COMMIT,
      native_manifest_sha256: NATIVE_MANIFEST_SHA,
      native_release_published_at: NATIVE_PUBLISHED_AT,
    }), 'utf8');
    assert.throws(() => orchestrator.loadProvenance(provenancePath), ContractError);
  });
});

// --- StableNativeBacklogTest -------------------------------------------------------

function release(tag, publishedAt, { draft = false, prerelease = null } = {}) {
  return {
    tag_name: tag,
    draft,
    prerelease: prerelease === null ? tag.includes('-') : prerelease,
    published_at: publishedAt,
  };
}

test('test_selects_every_post_baseline_stable_release_in_publication_order', () => {
  const releases = [
    release('v0.2.1', '2026-08-28T03:00:00Z'),
    release('b10599', '2026-08-28T02:00:00Z', { prerelease: false }),
    release('v0.2.0-1', native.STABLE_AUTOMATION_BASELINE_PUBLISHED_AT),
    release('v0.2.0-2', '2026-08-27T03:00:00Z'),
    release('v0.3.0', '2026-08-29T03:00:00Z', { draft: true }),
  ];
  assert.deepEqual(native.selectStableNativeBacklog(releases), ['v0.2.0-2', 'v0.2.1']);
});

test('test_post_baseline_stable_rollback_fails_closed', () => {
  assert.throws(() => native.selectStableNativeBacklog([release('v0.1.99', '2026-08-28T03:00:00Z')]), ContractError);
});

test('test_inconsistent_stable_prerelease_state_fails_closed', () => {
  assert.throws(
    () => native.selectStableNativeBacklog([release('v0.2.0-2', '2026-08-28T03:00:00Z', { prerelease: false })]),
    ContractError,
  );
});

test('test_duplicate_stable_tag_fails_closed', () => {
  const entry = release('v0.2.1', '2026-08-28T03:00:00Z');
  assert.throws(() => native.selectStableNativeBacklog([entry, { ...entry }]), ContractError);
});
