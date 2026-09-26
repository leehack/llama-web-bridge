// Tests of scripts/release/orchestrator/asset_releases.mjs, one test per test
// method of scripts/release_orchestrator_asset_releases_test.py, with the same
// names and assertions. A Python subTest loop is one loop inside its test.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

import { ASSETS_REPOSITORY, ContractError, NATIVE_REPOSITORY } from '../../../scripts/release/contract.mjs';
import { pyGet, pyItems } from '../../../scripts/release/json.mjs';
import { ARTIFACTS } from '../../../scripts/release/manifest.mjs';
import * as assetReleases from '../../../scripts/release/orchestrator/asset_releases.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as runNames from '../../../scripts/release/orchestrator/run_names.mjs';
import { PUBLICATION_FILES, candidatePublicationDigests } from '../../../scripts/release/publication_state.mjs';
import { loadCandidate } from '../../../scripts/release/qualification.mjs';
import { releaseAttestation } from '../contract_fixtures.mjs';
import {
  ADVANCED_BRIDGE_SHA, ASSETS_TAG_COMMIT, BRIDGE_SHA, FakeGateway, LEGACY_MANUAL_QUALIFICATION_GATES,
  LEGACY_MANUAL_UNPROVEN_CAPABILITIES, NATIVE_MANIFEST_SHA, alignedReleaseStub, assetReleaseStub, directoryMembers,
  makeLegacyV0140Provenance, makeProvenance, releasePayload, writeBridgeCandidate,
} from './fixtures.mjs';
import { replaceProvenance } from './driver_fixtures.mjs';

function sha256Hex(data) {
  return createHash('sha256').update(data).digest('hex');
}

// --- PublishedNativeAlignmentTest ------------------------------------------------

function alignmentRelease(tag, nativeTags, fields = {}) {
  const release = assetReleaseStub(tag);
  release.body = nativeTags.map((nativeTag) => `Native: \`${NATIVE_REPOSITORY}@${nativeTag}\`\r\n`).join('');
  Object.assign(release, fields);
  return release;
}

test('test_no_recorded_alignment_keeps_every_provenance_eligible', () => {
  assert.equal(assetReleases.latestPublishedNativeAlignment([]), null);
  assert.equal(
    assetReleases.latestPublishedNativeAlignment([assetReleaseStub('v0.1.37'), { tag_name: 'v0.1.36', body: null }]),
    null,
  );
});

test('test_newest_native_order_wins_regardless_of_listing_order', () => {
  const releases = [
    alignmentRelease('v0.1.44', ['v0.4.1']),
    alignmentRelease('v0.1.45', ['v0.4.1-1']),
    alignmentRelease('v0.1.42', ['v0.3.0']),
  ];
  assert.deepEqual(assetReleases.latestPublishedNativeAlignment(releases), ['v0.4.1-1', 'v0.1.45']);
});

test('test_only_published_stable_native_markers_are_evidence', () => {
  const foreign = assetReleaseStub('v0.1.47');
  foreign.body = 'Native: `someone/else@v9.9.9`\nsee Native: `x@v9.9.9`\n';
  const releases = [
    alignmentRelease('v0.1.44', ['v0.4.1']),
    alignmentRelease('v0.1.45', ['v0.5.0'], { draft: true }),
    alignmentRelease('v0.1.46', ['v0.5.0'], { draft: null }),
    alignmentRelease('b9165', ['b9165'], { prerelease: true }),
    alignmentRelease('b9170', ['v0.5.0'], { prerelease: true }),
    alignmentRelease('v0.1.48', ['b9165']),
    alignmentRelease('not-a-release', ['v0.5.0']),
    foreign,
  ];
  assert.deepEqual(assetReleases.latestPublishedNativeAlignment(releases), ['v0.4.1', 'v0.1.44']);
});

test('test_contradictory_or_malformed_alignment_fails_closed', () => {
  for (const release of [
    alignmentRelease('v0.1.44', ['v0.4.0', 'v0.4.1']),
    alignmentRelease('v0.1.44', ['v0.4']),
    alignmentRelease('v0.1.44', ['v0.4.1-llamadart.1']),
  ]) {
    assert.throws(() => assetReleases.latestPublishedNativeAlignment([release]), ContractError, release.body);
  }
  assert.deepEqual(
    assetReleases.latestPublishedNativeAlignment([alignmentRelease('v0.1.44', ['v0.4.1', 'v0.4.1'])]),
    ['v0.4.1', 'v0.1.44'],
  );
});

// --- IdenticalPublicationTest ------------------------------------------------------

function withIdentical(fn) {
  return () => {
    const tmp = fs.mkdtempSync(path.join(tmpdir(), 'sro-identical-'));
    const provenance = makeProvenance();
    const self = {
      tmp,
      provenance,
      prior: replaceProvenance(provenance, { bridgeSourceSha: ADVANCED_BRIDGE_SHA, bridgeBuildSha: ADVANCED_BRIDGE_SHA }),
      digests(name, { marker, tag, provenance: bound }) {
        const directory = path.join(tmp, name);
        writeBridgeCandidate(directory, {
          releaseTag: tag,
          releaseRebuild: 0,
          correlationId: runNames.computeCorrelationId(bound),
          bridgeCommit: bound.bridgeSourceSha,
          runId: String(700 + name.length),
          marker: Buffer.from(marker),
        });
        return candidatePublicationDigests(directory);
      },
    };
    try {
      fn(self);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  };
}

test('test_manifest_is_the_only_file_that_differs_between_identical_builds', withIdentical((self) => {
  const published = self.digests('published', { marker: 'same', tag: 'v0.1.41', provenance: self.prior });
  const candidate = self.digests('candidate', { marker: 'same', tag: 'v0.1.42', provenance: self.provenance });
  assert.deepEqual(
    PUBLICATION_FILES.filter((name) => pyGet(published, name) !== pyGet(candidate, name)),
    ['manifest.json'],
  );
  assert.equal(assetReleases.publicationBytesIdentical(candidate, published), true);
}));

test('test_one_differing_artifact_is_not_identical', withIdentical((self) => {
  const published = self.digests('published', { marker: 'same', tag: 'v0.1.41', provenance: self.prior });
  const candidate = self.digests('candidate', { marker: 'changed', tag: 'v0.1.42', provenance: self.provenance });
  assert.equal(assetReleases.publicationBytesIdentical(candidate, published), false);
}));

test('test_incomplete_inventory_fails_closed', withIdentical((self) => {
  const published = self.digests('published', { marker: 'same', tag: 'v0.1.41', provenance: self.prior });
  const candidate = self.digests('candidate', { marker: 'same', tag: 'v0.1.42', provenance: self.provenance });
  for (const side of ['candidate', 'published']) {
    const partial = Object.fromEntries(pyItems(side === 'candidate' ? candidate : published).filter(([name]) => name !== ARTIFACTS[0]));
    assert.throws(() => assetReleases.publicationBytesIdentical(
      side === 'candidate' ? partial : candidate,
      side === 'candidate' ? published : partial,
    ), ContractError, side);
  }
}));

test('test_latest_aligned_release_is_the_newest_tag_for_this_native', withIdentical((self) => {
  const otherNative = makeProvenance({ nativeReleaseTag: 'v0.2.1', nativeManifestSha256: 'f'.repeat(64) });
  const releases = [
    alignedReleaseStub('v0.1.41', self.prior),
    alignedReleaseStub('v0.1.41-2', self.prior),
    alignedReleaseStub('v0.1.42', otherNative),
    alignedReleaseStub('v0.1.41-1', self.prior),
  ];
  const selected = assetReleases.latestAlignedRelease(releases, self.provenance);
  assert.notEqual(selected, null);
  assert.equal(selected[0].tag_name, 'v0.1.41-2');
  assert.equal(selected[1], runNames.computeCorrelationId(self.prior));
  assert.equal(assetReleases.latestAlignedRelease(releases.slice(2, 3), self.provenance), null);
}));

test('test_latest_aligned_release_skips_drafts_and_ambiguous_correlations', withIdentical((self) => {
  const draft = alignedReleaseStub('v0.1.41', self.prior);
  draft.draft = true;
  assert.equal(assetReleases.latestAlignedRelease([draft], self.provenance), null);
  const ambiguous = alignedReleaseStub('v0.1.41', self.prior);
  ambiguous.body += `Orchestrator correlation: \`${runNames.computeCorrelationId(self.provenance)}\`\n`;
  const malformed = alignedReleaseStub('v0.1.41-1', self.prior);
  malformed.body = malformed.body.replace(runNames.computeCorrelationId(self.prior), 'not a correlation');
  const older = alignedReleaseStub('v0.1.41', self.prior);
  for (const releases of [[ambiguous], [malformed]]) {
    assert.equal(assetReleases.latestAlignedRelease(releases, self.provenance), null);
  }
  const selected = assetReleases.latestAlignedRelease([older, ambiguous, malformed], self.provenance);
  assert.equal(selected[0], older);
}));

test('test_legacy_v0140_is_comparable_only_by_its_own_provenance', () => {
  const legacy = makeLegacyV0140Provenance();
  const release = alignedReleaseStub('v0.1.40', legacy);
  assert.notEqual(assetReleases.latestAlignedRelease([release], legacy), null);
  const governed = makeLegacyV0140Provenance({ bridgeBuildSha: ADVANCED_BRIDGE_SHA });
  assert.equal(assetReleases.latestAlignedRelease([release], governed), null);
});

// --- PublishedReleaseVerificationTest ---------------------------------------------

function withPublished(fn) {
  return () => {
    const tmp = fs.mkdtempSync(path.join(tmpdir(), 'sro-published-'));
    try {
      const provenance = makeProvenance();
      const correlationId = runNames.computeCorrelationId(provenance);
      const candidate = path.join(tmp, 'candidate');
      writeBridgeCandidate(candidate, { releaseTag: 'v0.1.40', releaseRebuild: 0, correlationId });
      const members = directoryMembers(candidate);
      const fingerprint = loadCandidate(candidate)[1];
      const body = `Candidate fingerprint: \`${fingerprint}\`\nOrchestrator correlation: \`${correlationId}\`\n`;
      const self = {
        tmp,
        provenance,
        correlationId,
        candidate,
        members,
        fingerprint,
        body,
        release: releasePayload({ tag: 'v0.1.40', body, members }),
        gatewayForRelease(release, { releaseById = null, tagCommit = ASSETS_TAG_COMMIT, attestation = null } = {}) {
          const blobs = {};
          for (const asset of release.assets) {
            if (Object.hasOwn(members, asset.name)) blobs[`repos/${ASSETS_REPOSITORY}/releases/assets/${asset.id}`] = members[asset.name];
          }
          const tag = release.tag_name;
          const releaseId = release.id;
          const digests = Object.fromEntries(Object.entries(members).map(([name, data]) => [name, sha256Hex(data)]));
          const routes = {
            [`repos/${ASSETS_REPOSITORY}/git/ref/tags/${tag}`]: { ref: `refs/tags/${tag}`, object: { type: 'commit', sha: tagCommit } },
            [`repos/${ASSETS_REPOSITORY}/releases/tags/${tag}`]: release,
            [`repos/${ASSETS_REPOSITORY}/releases/${releaseId}`]: releaseById || release,
          };
          const attestations = [[[ASSETS_REPOSITORY, tag], attestation || releaseAttestation({
            releaseTag: tag, assetsRepo: ASSETS_REPOSITORY, tagCommit, releaseId, assets: digests,
          })]];
          return new FakeGateway({ jsonRoutes: routes, blobRoutes: blobs, releaseAttestations: attestations });
        },
        verify(release, options = {}) {
          return assetReleases.verifyPublishedRelease(self.gatewayForRelease(release, options), {
            release, provenance, correlationId, workspace: path.join(tmp, 'workspace'),
          });
        },
      };
      fn(self);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  };
}

test('test_exact_immutable_publication_is_accepted', withPublished((self) => {
  const verified = self.verify(self.release);
  assert.equal(verified.releaseTarget.releaseTag, 'v0.1.40');
  assert.equal(verified.releaseTarget.releaseRebuild, 0);
  assert.equal(verified.binding.bridgeSourceSha, BRIDGE_SHA);
}));

test('test_only_exact_v0140_identity_selects_legacy_manifest_contract', () => {
  const exact = makeLegacyV0140Provenance();
  assert.deepEqual(
    model.publishedManifestCompatibility({ tag: 'v0.1.40', provenance: exact }),
    [LEGACY_MANUAL_QUALIFICATION_GATES, LEGACY_MANUAL_UNPROVEN_CAPABILITIES],
  );
  const nearMisses = {
    'release tag': ['v0.1.40-1', exact],
    'native tag': ['v0.1.40', makeLegacyV0140Provenance({ nativeReleaseTag: 'v0.3.0-1' })],
    'native commit': ['v0.1.40', makeLegacyV0140Provenance({ nativeCommit: '1'.repeat(40) })],
    'upstream tag': ['v0.1.40', makeLegacyV0140Provenance({ upstreamTag: 'v0.3.1' })],
    'upstream commit': ['v0.1.40', makeLegacyV0140Provenance({ upstreamCommit: '2'.repeat(40) })],
    'governed bridge build': ['v0.1.40', makeLegacyV0140Provenance({ bridgeBuildSha: '3'.repeat(40) })],
    'native manifest': ['v0.1.40', makeLegacyV0140Provenance({ nativeManifestSha256: '0'.repeat(64) })],
  };
  for (const [label, [tag, provenance]] of Object.entries(nearMisses)) {
    assert.equal(model.publishedManifestCompatibility({ tag, provenance }), null, label);
  }
});

test('test_native_manifest_marker_prevents_duplicate_when_correlation_is_damaged', withPublished((self) => {
  const malformed = {
    ...self.release,
    body: `Native: \`${NATIVE_REPOSITORY}@v0.2.0\`\nNative manifest SHA-256: \`${NATIVE_MANIFEST_SHA}\`\n`,
  };
  assert.equal(assetReleases.findCorrelatedRelease([malformed], self.correlationId, self.provenance), malformed);
  assert.throws(() => self.verify(malformed), ContractError);
}));

test('test_manifest_digest_without_exact_native_tag_cannot_hijack_state', withPublished((self) => {
  const foreign = {
    ...self.release,
    body: `Native: \`${NATIVE_REPOSITORY}@v9.9.9\`\nNative manifest SHA-256: \`${NATIVE_MANIFEST_SHA}\`\n`,
  };
  assert.equal(assetReleases.findCorrelatedRelease([foreign], self.correlationId, self.provenance), null);
}));

test('test_mutable_release_fails_closed', withPublished((self) => {
  assert.throws(() => self.verify({ ...self.release, immutable: false }), ContractError);
}));

test('test_missing_immutability_field_fails_closed', withPublished((self) => {
  const release = { ...self.release };
  delete release.immutable;
  assert.throws(() => self.verify(release), ContractError);
}));

test('test_missing_published_at_fails_closed', withPublished((self) => {
  assert.throws(() => self.verify({ ...self.release, published_at: null }), ContractError);
}));

test('test_draft_release_fails_closed', withPublished((self) => {
  assert.throws(() => self.verify({ ...self.release, draft: true }), ContractError);
}));

test('test_incomplete_asset_inventory_fails_closed', withPublished((self) => {
  const release = { ...self.release, assets: self.release.assets.filter((asset) => asset.name !== 'sha256sums.txt') };
  assert.throws(() => self.verify(release), ContractError);
}));

test('test_unexpected_extra_asset_fails_closed', withPublished((self) => {
  const release = {
    ...self.release,
    assets: [...self.release.assets, {
      id: 999, name: 'extra.bin', state: 'uploaded', size: 1, digest: `sha256:${'0'.repeat(64)}`,
    }],
  };
  assert.throws(() => self.verify(release), ContractError);
}));

test('test_digest_mismatch_fails_closed', withPublished((self) => {
  const release = releasePayload({
    tag: 'v0.1.40',
    body: self.body,
    members: self.members,
    assetOverrides: { 'manifest.json': { digest: `sha256:${'0'.repeat(64)}` } },
  });
  assert.throws(() => self.verify(release), ContractError);
}));

test('test_manifest_bytes_that_do_not_bind_provenance_fail_closed', withPublished((self) => {
  assert.throws(() => assetReleases.verifyPublishedRelease(self.gatewayForRelease(self.release), {
    release: self.release,
    provenance: makeProvenance({ nativeCommit: 'd'.repeat(40) }),
    correlationId: self.correlationId,
    workspace: path.join(self.tmp, 'workspace'),
  }), ContractError);
}));

test('test_release_tag_commit_is_resolved_independently', withPublished((self) => {
  assert.throws(() => self.verify(self.release, { tagCommit: 'd'.repeat(40) }), ContractError);
}));

test('test_release_readback_by_id_must_match', withPublished((self) => {
  assert.throws(() => self.verify(self.release, { releaseById: { ...self.release, body: 'unrelated' } }), ContractError);
}));

test('test_signed_release_attestation_must_bind_every_asset', withPublished((self) => {
  const invalid = releaseAttestation({
    releaseTag: 'v0.1.40',
    assetsRepo: ASSETS_REPOSITORY,
    tagCommit: ASSETS_TAG_COMMIT,
    releaseId: self.release.id,
    assets: { 'manifest.json': '0'.repeat(64) },
  });
  assert.throws(() => self.verify(self.release, { attestation: invalid }), ContractError);
}));

test('test_release_body_without_the_candidate_fingerprint_fails_closed', withPublished((self) => {
  // The publication contract binds the release body to the exact candidate
  // digest. A body that only names the correlation proves nothing about
  // which bytes were published.
  const release = releasePayload({
    tag: 'v0.1.40', body: `Orchestrator correlation: \`${self.correlationId}\`\n`, members: self.members,
  });
  assert.throws(() => self.verify(release), ContractError);
}));

test('test_release_body_with_a_foreign_candidate_fingerprint_fails_closed', withPublished((self) => {
  const release = releasePayload({
    tag: 'v0.1.40',
    body: `Candidate fingerprint: \`${'0'.repeat(64)}\`\nOrchestrator correlation: \`${self.correlationId}\`\n`,
    members: self.members,
  });
  assert.throws(() => self.verify(release), ContractError);
}));

test('test_release_tag_that_contradicts_the_published_manifest_fails_closed', withPublished((self) => {
  assert.throws(() => self.verify(releasePayload({ tag: 'v0.1.41', body: self.body, members: self.members })), ContractError);
}));

test('test_publication_from_an_earlier_bridge_source_stays_a_noop', withPublished((self) => {
  // Main advances daily; the published release stays bound to the exact
  // candidate source it was built from, not to today's HEAD.
  const verified = self.verify(self.release);
  assert.equal(verified.binding.bridgeSourceSha, BRIDGE_SHA);
  const advanced = assetReleases.verifyPublishedRelease(self.gatewayForRelease(self.release), {
    release: self.release,
    provenance: makeProvenance({ bridgeSourceSha: ADVANCED_BRIDGE_SHA }),
    correlationId: self.correlationId,
    workspace: path.join(self.tmp, 'workspace'),
  });
  assert.equal(advanced.binding.bridgeSourceSha, BRIDGE_SHA);
}));
