// Candidate and qualification run proofs for the stable release orchestrator,
// the Node port of scripts/release_orchestrator_stage_proofs.py.
//
// Binds a candidate manifest to its exact provenance and correlation, and
// proves a candidate or qualification run from its run record, reachability
// from the default branch, its unique artifact, and the hosted qualification
// attestation.
//
// Paths are strings with pathlib's str() form: the workspace is normalized
// with pyPath and every child is joined with pyJoinPath, so error text that
// names a path matches Python's.

import fs from 'node:fs';

import { extractFlatArtifactArchive } from '../archive.mjs';
import {
  ASSETS_REPOSITORY, BRIDGE_REPOSITORY, ContractError, NATIVE_REPOSITORY, requireSha256, validateReleaseIdentity,
} from '../contract.mjs';
import {
  PyException, isDict, isPyInt, pyEquals, pyGet, pyIsFile, pyItems, pyJoinPath, pyKeys, pyOSError, pyPath, pyRepr,
} from '../json.mjs';
import { ARTIFACTS } from '../manifest.mjs';
import { pyFullmatch } from '../python_compat.mjs';
import {
  ATTESTATION_ARTIFACT_NAME, CANDIDATE_ARTIFACT_NAME, loadAttestationFile, loadCandidate, validateArtifactInventory,
  validateWorkflowRun, verifyAttestation,
} from '../qualification.mjs';
import {
  CANDIDATE_WORKFLOW_PATH, COMMIT_RE, PipelineBinding, PyDataclass, QUALIFICATION_WORKFLOW_PATH, REQUIRED, requireStr,
} from './model.mjs';

// isinstance(value, int) and not isinstance(value, bool) and value >= 0.
function isNonNegativeInt(value) {
  return isPyInt(value) && BigInt(value) >= 0n;
}

// Fail closed unless a manifest binds this exact provenance and correlation.
export function validateCandidateManifest(manifest, {
  provenance,
  correlationId,
  expectedReleaseTag = null,
  expectedBridgeSourceSha = null,
  expectedRunId = null,
}) {
  const expected = [
    ['assets_repository', ASSETS_REPOSITORY],
    ['bridge_repository', BRIDGE_REPOSITORY],
    // `expected_bridge_source_sha or manifest.get("bridge_commit")`.
    ['bridge_commit', expectedBridgeSourceSha || pyGet(manifest, 'bridge_commit')],
    ['upstream_repository', 'ggml-org/llama.cpp'],
    ['upstream_tag', provenance.upstreamTag],
    ['upstream_commit', provenance.upstreamCommit],
    ['native_repository', NATIVE_REPOSITORY],
    ['native_release_tag', provenance.nativeReleaseTag],
    ['native_manifest_sha256', provenance.nativeManifestSha256],
    ['native_commit', provenance.nativeCommit],
    ['orchestrator_correlation_id', correlationId],
  ];
  for (const [key, value] of expected) {
    if (!pyEquals(pyGet(manifest, key), value)) {
      throw new ContractError(`candidate manifest ${key} is ${pyRepr(pyGet(manifest, key))}, expected ${pyRepr(value)}`);
    }
  }
  const bridgeCommit = requireStr(pyGet(manifest, 'bridge_commit'), 'manifest bridge_commit');
  if (pyFullmatch(COMMIT_RE, bridgeCommit) === null) throw new ContractError('manifest bridge_commit must be a 40-hex commit SHA');
  if (expectedRunId !== null && !pyEquals(pyGet(manifest, 'github_run_id'), expectedRunId)) {
    throw new ContractError(
      `candidate manifest github_run_id is ${pyRepr(pyGet(manifest, 'github_run_id'))}, `
      + `expected ${pyRepr(expectedRunId)}`,
    );
  }
  const releaseTag = requireStr(pyGet(manifest, 'release_tag'), 'manifest release_tag');
  const rebuild = pyGet(manifest, 'release_rebuild');
  if (!isNonNegativeInt(rebuild)) throw new ContractError('manifest release_rebuild must be a non-negative integer');
  if (expectedReleaseTag !== null && releaseTag !== expectedReleaseTag) {
    throw new ContractError(`candidate manifest release_tag is ${pyRepr(releaseTag)}, expected ${pyRepr(expectedReleaseTag)}`);
  }
  validateReleaseIdentity(releaseTag, rebuild, provenance.upstreamTag);
  const artifacts = pyGet(manifest, 'artifacts');
  const artifactNames = isDict(artifacts) || artifacts instanceof Map ? pyKeys(artifacts) : null;
  if (
    artifactNames === null
    || new Set(artifactNames).size !== ARTIFACTS.length
    || !ARTIFACTS.every((name) => artifactNames.includes(name))
  ) {
    throw new ContractError('candidate manifest does not record exactly the artifact set');
  }
  for (const [name, record] of pyItems(artifacts)) {
    if (!isDict(record) && !(record instanceof Map)) throw new ContractError(`candidate manifest artifact ${pyRepr(name)} is malformed`);
    requireSha256(requireStr(pyGet(record, 'sha256'), `${name} sha256`), `${name} sha256`);
    if (!isNonNegativeInt(pyGet(record, 'size_bytes'))) {
      throw new ContractError(`candidate manifest artifact ${pyRepr(name)} has an invalid size`);
    }
  }
  return new PipelineBinding({ bridgeSourceSha: bridgeCommit, releaseTag, releaseRebuild: rebuild });
}

// `directory` is the path string of the extracted candidate artifact.
export class CandidateEvidence extends PyDataclass {
  static FIELDS = Object.freeze([
    ['runId', 'run_id', REQUIRED],
    ['artifactId', 'artifact_id', REQUIRED],
    ['fingerprint', 'fingerprint', REQUIRED],
    ['binding', 'binding', REQUIRED],
    ['manifest', 'manifest', REQUIRED],
    ['directory', 'directory', REQUIRED],
  ]);

  constructor(options) {
    super();
    this.initFields(options);
    Object.freeze(this);
  }
}

// _require_reachable_from_main.
export function requireReachableFromMain(gateway, {
  commit, defaultBranch, label,
}) {
  const payload = gateway.apiJson(`repos/${BRIDGE_REPOSITORY}/compare/${commit}...${defaultBranch}`);
  if (!isDict(payload)) throw new ContractError('compare response must be a JSON object');
  const status = pyGet(payload, 'status');
  if (!pyEquals(status, 'ahead') && !pyEquals(status, 'identical')) {
    throw new ContractError(`${label} commit ${commit} is not reachable from ${defaultBranch} (compare status ${pyRepr(status)})`);
  }
}

// Path.write_bytes(data).
function writeBytes(target, data) {
  try {
    fs.writeFileSync(target, data);
  } catch (error) {
    throw pyOSError(error, target);
  }
}

// Path.mkdir(parents=True, exist_ok=True): an existing non-directory raises
// FileExistsError.
export function makeDirectories(target) {
  try {
    fs.mkdirSync(target, { recursive: true });
  } catch (error) {
    const exception = pyOSError(error, target);
    if (error?.code === 'EEXIST' && exception instanceof PyException) {
      const replaced = new PyException('FileExistsError', exception.message);
      Object.assign(replaced, { errno: exception.errno, code: exception.code, filename: exception.filename });
      throw replaced;
    }
    throw exception;
  }
}

// _download_run_artifact: [artifactId, destination].
export function downloadRunArtifact(gateway, {
  runId, artifactName, artifactType, workspace,
}) {
  const inventory = gateway.apiJson(`repos/${BRIDGE_REPOSITORY}/actions/runs/${runId}/artifacts?per_page=100`);
  const artifactId = validateArtifactInventory(inventory, { expectedRunId: runId, expectedName: artifactName });
  const root = pyPath(String(workspace));
  const archive = pyJoinPath(root, `${artifactType}-${runId}.zip`);
  writeBytes(archive, gateway.downloadBytes(
    `repos/${BRIDGE_REPOSITORY}/actions/artifacts/${artifactId}/zip`,
    { accept: 'application/vnd.github+json' },
  ));
  const destination = pyJoinPath(root, `${artifactType}-${runId}`);
  makeDirectories(destination);
  extractFlatArtifactArchive(archive, destination, { artifactType });
  return [artifactId, destination];
}

export function verifyCandidateRun(gateway, {
  runId, provenance, correlationId, binding, defaultBranch, workspace,
}) {
  const run = gateway.apiJson(`repos/${BRIDGE_REPOSITORY}/actions/runs/${runId}`);
  const headSha = validateWorkflowRun(run, {
    expectedRunId: runId,
    expectedWorkflowPath: CANDIDATE_WORKFLOW_PATH,
    expectedHeadBranch: defaultBranch,
    expectedRunAttempt: 1,
  });
  requireReachableFromMain(gateway, { commit: headSha, defaultBranch, label: 'candidate run head' });
  const [artifactId, directory] = downloadRunArtifact(gateway, {
    runId,
    artifactName: CANDIDATE_ARTIFACT_NAME,
    artifactType: 'candidate',
    workspace,
  });
  const [manifest, fingerprint] = loadCandidate(directory);
  const manifestBinding = validateCandidateManifest(manifest, {
    provenance,
    correlationId,
    expectedReleaseTag: binding.releaseTag,
    expectedBridgeSourceSha: binding.bridgeSourceSha,
    expectedRunId: runId,
  });
  if (!manifestBinding.equals(binding)) {
    throw new ContractError('candidate manifest contradicts the binding its run name advertises');
  }
  requireReachableFromMain(gateway, { commit: binding.bridgeSourceSha, defaultBranch, label: 'candidate bridge source' });
  return new CandidateEvidence({
    runId, artifactId, fingerprint, binding, manifest, directory,
  });
}

export function verifyQualificationRun(gateway, {
  runId, candidate, provenance, correlationId, defaultBranch, workspace,
}) {
  const run = gateway.apiJson(`repos/${BRIDGE_REPOSITORY}/actions/runs/${runId}`);
  const qualificationSourceSha = validateWorkflowRun(run, {
    expectedRunId: runId,
    expectedWorkflowPath: QUALIFICATION_WORKFLOW_PATH,
    expectedHeadBranch: defaultBranch,
    expectedRunAttempt: 1,
  });
  requireReachableFromMain(gateway, { commit: qualificationSourceSha, defaultBranch, label: 'qualification workflow source' });
  const [, directory] = downloadRunArtifact(gateway, {
    runId,
    artifactName: ATTESTATION_ARTIFACT_NAME,
    artifactType: 'attestation',
    workspace,
  });
  const attestationPath = pyJoinPath(directory, 'qualification-attestation.json');
  if (!pyIsFile(attestationPath)) throw new ContractError('attestation artifact does not contain the canonical payload');
  const attestation = loadAttestationFile(attestationPath);
  const emscriptenVersion = requireStr(pyGet(candidate.manifest, 'emscripten_version'), 'candidate manifest emscripten_version');
  return verifyAttestation({
    attestation,
    candidateDir: candidate.directory,
    candidateFingerprint: candidate.fingerprint,
    candidateRunId: candidate.runId,
    candidateArtifactId: candidate.artifactId,
    candidateRunAttempt: 1,
    qualificationRunId: runId,
    qualificationRunAttempt: 1,
    qualificationSourceSha,
    bridgeSourceSha: candidate.binding.bridgeSourceSha,
    upstreamTag: provenance.upstreamTag,
    upstreamCommit: provenance.upstreamCommit,
    nativeReleaseTag: provenance.nativeReleaseTag,
    nativeManifestSha256: provenance.nativeManifestSha256,
    nativeCommit: provenance.nativeCommit,
    emscriptenVersion,
    releaseTag: candidate.binding.releaseTag,
    releaseRebuild: candidate.binding.releaseRebuild,
    orchestratorCorrelationId: correlationId,
  });
}
