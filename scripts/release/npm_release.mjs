#!/usr/bin/env node

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { isDeepStrictEqual } from 'node:util';

import { extractFlatArtifactArchive } from './archive.mjs';
import { parseArguments, runCli, SystemExitError } from './cli.mjs';
import {
  ASSETS_REPOSITORY, BRIDGE_REPOSITORY, ContractError, strictJsonLoads, validateReleaseImmutability,
} from './contract.mjs';
import { FIXTURE_RELEASE_TAG, resolveVersion } from './npm_package.mjs';
import { PUBLICATION_FILES, verifyImmutablePublication } from './publication_state.mjs';
import {
  ATTESTATION_ARTIFACT_NAME, CANDIDATE_WORKFLOW_PATH, QUALIFICATION_WORKFLOW_PATH,
  loadAttestationFile, loadCandidate, validateArtifactInventory, validateWorkflowRun, verifyAttestation,
} from './qualification.mjs';

const PUBLICATION_WORKFLOW_PATH = '.github/workflows/publish_assets.yml';
const PUBLICATION_OUTCOME = 'bridge-qualification-outcome';
const PUBLICATION_ATTESTATION = 'verified-qualification-attestation';
const BRIDGE_RUNS = `repos/${BRIDGE_REPOSITORY}/actions/runs`;
const ASSETS_RELEASES = `repos/${ASSETS_REPOSITORY}/releases`;
const SHA256_DIGEST = /^sha256:[0-9a-f]{64}$/;
const MAX_RELEASE_ASSET_BYTES = 64 * 1024 * 1024;
const MAX_RUN_ARTIFACT_BYTES = 1024 * 1024;

const sha256 = (bytes) => `sha256:${createHash('sha256').update(bytes).digest('hex')}`;

function requireThat(condition, message) {
  if (!condition) throw new ContractError(message);
}

function releaseNoteRunIds(notes) {
  requireThat(typeof notes === 'string', 'release notes must identify the qualification and publication runs');
  function findRun(label, includesUrl) {
    const lines = notes.split(/\r?\n/).filter((line) => line.startsWith(`${label} run:`));
    const match = /^`([1-9][0-9]*)`(?: \(([^)]+)\))?$/.exec(lines[0]?.slice(`${label} run: `.length) ?? '');
    requireThat(lines.length === 1 && match, `release notes must identify exactly one ${label.toLowerCase()} run ID`);
    const [, id, url] = match;
    requireThat(includesUrl ? url === `https://github.com/${BRIDGE_REPOSITORY}/actions/runs/${id}` : url === undefined,
      `release notes ${label.toLowerCase()} run URL does not match ${BRIDGE_REPOSITORY} and run ${id}`);
    return id;
  }
  return { qualificationRunId: findRun('Qualification', false), publicationRunId: findRun('Publication', true) };
}

function getRun(api, id, expectedWorkflowPath) {
  const run = api(`${BRIDGE_RUNS}/${id}`);
  const headSha = validateWorkflowRun(run, {
    expectedRunId: id,
    expectedWorkflowPath,
    expectedHeadBranch: 'main',
  });
  return { headSha, title: run.display_title };
}

function getArtifact(inventory, runId, name, { optional = false } = {}) {
  if (optional) {
    requireThat(Array.isArray(inventory?.artifacts) && inventory.total_count === inventory.artifacts.length,
      `incomplete artifact inventory for publication run ${runId}`);
    if (!inventory.artifacts.some((artifact) => artifact?.name === name)) return null;
  }
  const id = validateArtifactInventory(inventory, { expectedRunId: runId, expectedName: name });
  const artifact = inventory.artifacts.find((entry) => entry.id === id);
  requireThat(Number.isSafeInteger(id) && SHA256_DIGEST.test(artifact?.digest),
    `run artifact ${name} has an invalid ID or SHA-256 digest`);
  return artifact;
}

export function verifyReleaseRuns({ manifest, releaseNotes, dist, api, readArtifact }) {
  const { qualificationRunId, publicationRunId } = releaseNoteRunIds(releaseNotes);
  const { github_run_id: candidateRunId, source_commit: source, release_tag: tag,
    release_rebuild: rebuild, orchestrator_correlation_id: correlation } = manifest;
  requireThat(new Set([candidateRunId, qualificationRunId, publicationRunId]).size === 3,
    'candidate, qualification and publication run IDs must be distinct');

  const candidate = getRun(api, candidateRunId, CANDIDATE_WORKFLOW_PATH);
  requireThat(candidate.title === `bridge-candidate ${correlation} source:${source} tag:${tag} rebuild:${rebuild}`,
    'candidate run title does not match the signed manifest');

  const qualification = getRun(api, qualificationRunId, QUALIFICATION_WORKFLOW_PATH);
  requireThat(qualification.title === `bridge-qualification ${correlation} candidate:${candidateRunId}`,
    'qualification run title does not match the signed candidate identity');
  const qualificationArtifact = getArtifact(api(`${BRIDGE_RUNS}/${qualificationRunId}/artifacts?per_page=100`),
    qualificationRunId, ATTESTATION_ARTIFACT_NAME);
  const attestation = readArtifact(qualificationArtifact);
  // The attested fingerprint hashes all 13 files, including the exact checksum
  // list and manifest. Passing candidateDir also checks each listed digest.
  const verified = verifyAttestation({
    attestation,
    candidateDir: dist,
    candidateRunId,
    qualificationRunId,
    qualificationSourceSha: qualification.headSha,
  });

  const publication = getRun(api, publicationRunId, PUBLICATION_WORKFLOW_PATH);
  requireThat(publication.title === `publish-assets ${correlation} candidate:${candidateRunId} qualification:${qualificationRunId} source:${source} tag:${tag} rebuild:${rebuild}`,
    'publication run title does not match the verified tag, candidate and qualification runs');
  const publicationInventory = api(`${BRIDGE_RUNS}/${publicationRunId}/artifacts?per_page=100`);
  const publicationArtifact = getArtifact(publicationInventory, publicationRunId, PUBLICATION_OUTCOME, { optional: true });
  if (publicationArtifact) {
    const outcome = readArtifact(publicationArtifact);
    requireThat(outcome?.candidate_run_id === candidateRunId && outcome.qualification_run_id === qualificationRunId
      && outcome.github_run_id === publicationRunId && outcome.candidate_fingerprint === verified.candidate_fingerprint,
    'publication outcome does not bind the verified candidate and qualification runs');
  }
  const publishedAttestation = getArtifact(publicationInventory, publicationRunId, PUBLICATION_ATTESTATION, { optional: true });
  if (publishedAttestation) {
    requireThat(isDeepStrictEqual(readArtifact(publishedAttestation), attestation),
      'publication attestation does not match the verified qualification attestation');
  }

  return {
    candidate_run_id: candidateRunId,
    qualification_run_id: qualificationRunId,
    qualification_artifact_id: qualificationArtifact.id,
    publication_run_id: publicationRunId,
    candidate_fingerprint: verified.candidate_fingerprint,
  };
}

function ghApi(endpoint, {
  binary = false, accept = 'application/vnd.github+json', maxBuffer = MAX_RUN_ARTIFACT_BYTES,
} = {}) {
  try {
    const output = execFileSync('gh', ['api', '--hostname', 'github.com', endpoint, '-H', `Accept: ${accept}`],
      { maxBuffer, stdio: ['ignore', 'pipe', 'inherit'] });
    return binary ? output : strictJsonLoads(output.toString('utf8'), endpoint);
  } catch (error) {
    throw new ContractError(`GitHub API evidence unavailable for ${endpoint}: ${error.message}`, { cause: error });
  }
}

function releaseAssets(release) {
  requireThat(Array.isArray(release.assets) && release.assets.length === PUBLICATION_FILES.length,
    'GitHub Release asset inventory must contain exactly the publication files');
  const inventory = new Map();
  const ids = new Set();
  for (const { id, name, size, digest, state } of release.assets) {
    requireThat(PUBLICATION_FILES.includes(name) && !inventory.has(name) && Number.isSafeInteger(id) && id > 0
      && !ids.has(id) && Number.isSafeInteger(size) && size > 0 && size <= MAX_RELEASE_ASSET_BYTES
      && SHA256_DIGEST.test(digest) && state === 'uploaded', `invalid GitHub Release asset record for ${name}`);
    inventory.set(name, { id, size, digest });
    ids.add(id);
  }
  return inventory;
}

function fetchRelease({ releaseTag, releaseAttestation, outDir }) {
  const { version, distTag } = resolveVersion(releaseTag, 0);
  requireThat(releaseTag !== FIXTURE_RELEASE_TAG, 'npm publication refuses fixture version 0.0.0');

  const releaseByTag = ghApi(`${ASSETS_RELEASES}/tags/${releaseTag}`);
  const ref = ghApi(`repos/${ASSETS_REPOSITORY}/git/ref/tags/${releaseTag}`);
  requireThat(ref.ref === `refs/tags/${releaseTag}` && ref.object?.type === 'commit',
    'assets tag must point directly to a commit');
  const tagCommit = ref.object.sha;
  const releaseId = validateReleaseImmutability(releaseByTag, { releaseTag, tagCommit });
  const releaseById = ghApi(`${ASSETS_RELEASES}/${releaseId}`);
  validateReleaseImmutability(releaseById, { releaseTag, tagCommit, releaseId });

  const assets = releaseAssets(releaseByTag);
  const readbackAssets = releaseAssets(releaseById);
  const dist = path.join(outDir, 'dist');
  mkdirSync(dist, { mode: 0o700 });
  for (const [name, { id, size, digest }] of assets) {
    const readback = readbackAssets.get(name);
    requireThat(readback.id === id && readback.size === size && readback.digest === digest,
      `GitHub Release asset inventory changed between the tag and ID readbacks: ${name}`);
    const bytes = ghApi(`${ASSETS_RELEASES}/assets/${id}`, {
      binary: true, accept: 'application/octet-stream', maxBuffer: MAX_RELEASE_ASSET_BYTES,
    });
    requireThat(bytes.length === size && sha256(bytes) === digest, `GitHub Release asset digest mismatch: ${name}`);
    writeFileSync(path.join(dist, name), bytes, { flag: 'wx' });
  }

  verifyImmutablePublication({
    candidate: dist,
    assetsRepo: ASSETS_REPOSITORY,
    releaseTag,
    tagCommit,
    releaseId,
    releaseByTag,
    releaseById,
    attestation: strictJsonLoads(readFileSync(releaseAttestation, 'utf8'), 'verified release attestation'),
  });

  const [manifest] = loadCandidate(dist);
  requireThat(manifest.release_tag === releaseTag && manifest.release_channel === 'stable' && manifest.release_rebuild === 0,
    'signed manifest must identify the requested stable release with rebuild 0');

  const runEvidence = verifyReleaseRuns({
    manifest,
    releaseNotes: releaseById.body,
    dist,
    api: ghApi,
    readArtifact({ id, name, digest }) {
      const bytes = ghApi(`repos/${BRIDGE_REPOSITORY}/actions/artifacts/${id}/zip`, { binary: true });
      requireThat(sha256(bytes) === digest, `run artifact ZIP digest mismatch: ${name} (${id})`);
      const archive = path.join(outDir, `${id}.zip`);
      writeFileSync(archive, bytes, { flag: 'wx' });
      if (name === ATTESTATION_ARTIFACT_NAME || name === PUBLICATION_ATTESTATION) {
        const destination = path.join(outDir, `attestation-${id}`);
        extractFlatArtifactArchive(archive, destination, { artifactType: 'attestation' });
        return loadAttestationFile(path.join(destination, 'qualification-attestation.json'));
      }
      const output = execFileSync('unzip', ['-p', archive, 'qualification-outcome.json'],
        { encoding: 'utf8', maxBuffer: 128 * 1024 });
      return strictJsonLoads(output, 'publication qualification outcome');
    },
  });

  return {
    release_id: releaseId,
    release_tag: releaseTag,
    source_commit: manifest.source_commit,
    npm_version: version,
    npm_dist_tag: distTag,
    ...runEvidence,
  };
}

function main(argv, write) {
  const args = parseArguments(argv, {
    prog: 'npm_release.mjs',
    options: ['--release-tag', '--release-attestation', '--out-dir']
      .map((flag) => ({ flag, required: true, type: 'str' })),
  });
  try {
    write(`${JSON.stringify(fetchRelease(args))}\n`);
  } catch (error) {
    throw new SystemExitError(`error: ${error.message}`);
  }
  return 0;
}

if (import.meta.main) runCli(main);
