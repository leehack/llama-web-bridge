import assert from 'node:assert/strict';
import path from 'node:path';
import { test } from 'node:test';

import { verifyReleaseRuns } from '../../scripts/release/npm_release.mjs';
import { ATTESTATION_ARTIFACT_NAME, QUALIFICATION_WORKFLOW_PATH } from '../../scripts/release/qualification.mjs';
import {
  BRIDGE_SHA, CANDIDATE_RUN_ID, CORRELATION_ID, QUALIFICATION_RUN_ID,
  setUp, workflowRun, writeCandidate,
} from './qualification_fixtures.mjs';

const repo = 'leehack/llama-web-bridge';
const publicationRunId = '32919087000';
const qualificationArtifactId = 867;
const publicationArtifactId = 868;
const publishedAttestationId = 869;
const makeArtifact = (id, name, runId) => ({
  id,
  name,
  digest: `sha256:${'a'.repeat(64)}`,
  expired: false,
  workflow_run: { id: Number(runId) },
});

function fixture(t) {
  const { attestation, candidate, manifest, fingerprint, tmp, cleanup } = setUp();
  t.after(cleanup);
  const { release_tag: tag, release_rebuild: rebuild } = manifest;
  const releaseNotes = [
    `Qualification run: \`${QUALIFICATION_RUN_ID}\``,
    `Publication run: \`${publicationRunId}\` (https://github.com/${repo}/actions/runs/${publicationRunId})`,
  ].join('\n');
  const candidateRun = workflowRun({
    display_title: `bridge-candidate ${CORRELATION_ID} source:${BRIDGE_SHA} tag:${tag} rebuild:${rebuild}`,
  });
  const qualification = workflowRun({
    id: Number(QUALIFICATION_RUN_ID),
    path: QUALIFICATION_WORKFLOW_PATH,
    display_title: `bridge-qualification ${CORRELATION_ID} candidate:${CANDIDATE_RUN_ID}`,
  });
  const publication = workflowRun({
    id: Number(publicationRunId),
    path: '.github/workflows/publish_assets.yml',
    display_title: `publish-assets ${CORRELATION_ID} candidate:${CANDIDATE_RUN_ID} qualification:${QUALIFICATION_RUN_ID} source:${BRIDGE_SHA} tag:${tag} rebuild:${rebuild}`,
  });
  const qualificationArtifact = makeArtifact(qualificationArtifactId, ATTESTATION_ARTIFACT_NAME, QUALIFICATION_RUN_ID);
  const qualificationInventory = { total_count: 1, artifacts: [qualificationArtifact] };
  const publicationInventory = {
    total_count: 2,
    artifacts: [
      makeArtifact(publicationArtifactId, 'bridge-qualification-outcome', publicationRunId),
      makeArtifact(publishedAttestationId, 'verified-qualification-attestation', publicationRunId),
    ],
  };
  const outcome = {
    candidate_run_id: CANDIDATE_RUN_ID,
    qualification_run_id: QUALIFICATION_RUN_ID,
    github_run_id: publicationRunId,
    candidate_fingerprint: fingerprint,
  };
  const runs = `repos/${repo}/actions/runs`;
  const endpoints = new Map([
    [`${runs}/${CANDIDATE_RUN_ID}`, candidateRun],
    [`${runs}/${QUALIFICATION_RUN_ID}`, qualification],
    [`${runs}/${publicationRunId}`, publication],
    [`${runs}/${QUALIFICATION_RUN_ID}/artifacts?per_page=100`, qualificationInventory],
    [`${runs}/${publicationRunId}/artifacts?per_page=100`, publicationInventory],
  ]);
  const publishedAttestation = { ...attestation };
  const artifacts = new Map([
    [qualificationArtifactId, attestation],
    [publicationArtifactId, outcome],
    [publishedAttestationId, publishedAttestation],
  ]);
  const options = {
    manifest,
    releaseNotes,
    dist: candidate,
    api(endpoint) {
      assert.ok(endpoints.has(endpoint), `unexpected GitHub endpoint ${endpoint}`);
      return endpoints.get(endpoint);
    },
    readArtifact({ id }) {
      assert.ok(artifacts.has(id), `unexpected artifact ID ${id}`);
      return artifacts.get(id);
    },
  };
  return {
    options, candidateRun, qualification, publication, qualificationArtifact, qualificationInventory,
    publicationInventory, attestation, publishedAttestation, outcome, fingerprint, tmp,
  };
}

test('release run discovery binds exact qualification and publication evidence', (t) => {
  const { options, fingerprint, publicationInventory } = fixture(t);
  const expected = {
    candidate_run_id: CANDIDATE_RUN_ID,
    qualification_run_id: QUALIFICATION_RUN_ID,
    qualification_artifact_id: qualificationArtifactId,
    publication_run_id: publicationRunId,
    candidate_fingerprint: fingerprint,
  };
  assert.deepEqual(verifyReleaseRuns(options), expected);
  Object.assign(publicationInventory, { total_count: 0, artifacts: [] });
  assert.deepEqual(verifyReleaseRuns(options), expected);
});

for (const [label, modify, failure] of [
  ['wrong workflow path', ({ qualification }) => { qualification.path = '.github/workflows/ci.yml'; }, /workflow run path mismatch/],
  ['non-main branch', ({ qualification }) => { qualification.head_branch = 'feature'; }, /workflow run head_branch must be 'main'/],
  ['failed conclusion', ({ qualification }) => { qualification.conclusion = 'failure'; }, /workflow run conclusion must be success/],
  ['unrelated candidate title', ({ candidateRun }) => { candidateRun.display_title += ' tag:v1.0.0'; }, /candidate run title/],
  ['another candidate in the attestation', ({ attestation }) => {
    attestation.candidate_run_id = '123456';
    attestation.candidate_run_url = `https://github.com/${repo}/actions/runs/123456`;
  }, /attestation candidate_run_id does not match/],
  ['different qualified bytes with valid checksums', ({ options, tmp }) => {
    options.dist = path.join(tmp, 'different-candidate');
    writeCandidate(options.dist, { marker: 'different candidate' });
  }, /candidate fingerprint mismatch/],
  ['expired qualification artifact', ({ qualificationArtifact }) => { qualificationArtifact.expired = true; }, /has expired/],
  ['missing qualification artifact', ({ qualificationInventory }) => {
    Object.assign(qualificationInventory, { total_count: 0, artifacts: [] });
  }, /must expose exactly one 'qualification-attestation' artifact/],
  ['qualification artifact owned by another run', ({ qualificationArtifact }) => {
    qualificationArtifact.workflow_run.id = 123456;
  }, /does not belong to run/],
  ['notes pointing to another repo', ({ options }) => {
    options.releaseNotes = options.releaseNotes.replace(repo, 'another/repository');
  }, /release notes publication run URL does not match/],
  ['qualification run from another repo', ({ qualification }) => {
    qualification.repository.full_name = 'another/repository';
  }, /workflow run repository must be exactly/],
  ['publication run with another qualification ID', ({ publication }) => {
    publication.display_title = publication.display_title.replace(QUALIFICATION_RUN_ID, '123456');
  }, /publication run title does not match/],
  ['publication outcome with another qualification ID', ({ outcome }) => {
    outcome.qualification_run_id = '123456';
  }, /publication outcome does not bind/],
  ['publication attestation copy with another qualification ID', ({ publicationInventory, publishedAttestation }) => {
    publicationInventory.artifacts.shift();
    publicationInventory.total_count = 1;
    publishedAttestation.qualification_run_id = '123456';
  }, /publication attestation does not match/],
]) {
  test(`release run verification rejects ${label}`, (t) => {
    const state = fixture(t);
    modify(state);
    assert.throws(() => verifyReleaseRuns(state.options), failure);
  });
}
