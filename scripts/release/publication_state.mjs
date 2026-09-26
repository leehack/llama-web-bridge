#!/usr/bin/env node
// Classify exact bridge publication state for idempotent, fail-closed retries,
// ported from scripts/release_publication_state.py (deleted at the harness
// 5.0.0 cutover).
//
// Every validator accepts and rejects what the Python module did, with the
// same ContractError text; the CLI prints the same outcome JSON
// (json.dumps(..., sort_keys=True)) and exits 0 exactly when the outcome is
// allowed. The assets repository is read through the same `git -C <repo>`
// subprocess calls.

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';

import {
  ArgparseExit, SystemExitError, parseCommandLine, progName, runCli,
} from './cli.mjs';
import {
  ASSETS_REPOSITORY,
  BRIDGE_REPOSITORY,
  IMMUTABLE_RELEASE_ATTESTATION_PREDICATE_TYPE,
  IMMUTABLE_RELEASE_ATTESTATION_SIGNER,
  NATIVE_REPOSITORY,
  ContractError,
  Transition,
  compareReleases,
  compareUpstream,
  parseReleaseTag,
  parseUpstreamTag,
  rejectDuplicateKeys,
  requireCorrelationId,
  requireSha256,
  validateNativeIdentity,
  validateReleaseAttestation,
  validateReleaseImmutability,
} from './contract.mjs';
import {
  ARTIFACTS,
  CAPABILITIES,
  QUALIFICATION_GATES,
  UNPROVEN_CAPABILITIES,
} from './manifest.mjs';
import {
  JSONDecodeError, PY_WHITESPACE, PyException, compareCodePoints, isDict, isPyException, isPyInt,
  pyBytesStrip, pyDecodeUtf8, pyDict, pyEquals, pyGet, pyIsDir, pyIsFile, pyIsSymlink, pyItems,
  pyJoinPath, pyJsonDumps, pyJsonLoads, pyJsonLoadsBytes, pyKeySetEquals, pyKeys, pyListdir, pyOSError,
  pyPath, pyReadBytes, pyReadText, pyRepr, pySplitlines, pyStr, pyStrip, pyTypeName,
  pyUniversalNewlines,
} from './json.mjs';

export { ContractError };

export const APPROVED_ASSETS_REPOSITORY = ASSETS_REPOSITORY;
export const PUBLICATION_FILES = Object.freeze([...ARTIFACTS, 'manifest.json', 'sha256sums.txt']);
const COMMIT_RE = /^[0-9a-f]{40}$/u;
const RUN_ID_RE = /^[1-9][0-9]*$/u;
const SUM_RE = /^([0-9a-f]{64}) {2}([A-Za-z0-9_.-]+)$/u;
const SPLIT_WHITESPACE = new RegExp(`[${PY_WHITESPACE}]+`, 'u');

// Raised when a candidate violates release or upstream ordering.
export class RollbackError extends ContractError {
  constructor(message, options) {
    super(message, options);
    this.name = 'RollbackError';
  }
}

const IDENTITY_FIELDS = Object.freeze([
  ['releaseTag', 'release_tag'],
  ['releaseRebuild', 'release_rebuild'],
  ['assetsRepo', 'assets_repo'],
  ['bridgeCommit', 'bridge_commit'],
  ['upstreamTag', 'upstream_tag'],
  ['upstreamCommit', 'upstream_commit'],
  ['nativeReleaseTag', 'native_release_tag'],
  ['nativeManifestSha256', 'native_manifest_sha256'],
  ['nativeCommit', 'native_commit'],
  ['emscriptenVersion', 'emscripten_version'],
  ['orchestratorCorrelationId', 'orchestrator_correlation_id'],
  ['githubRunId', 'github_run_id'],
  ['githubRunUrl', 'github_run_url'],
]);

// The frozen CandidateIdentity dataclass, with camelCase fields; every field
// is required, as the dataclass requires it.
export class CandidateIdentity {
  constructor(fields = {}) {
    const missing = IDENTITY_FIELDS.filter(([key]) => !Object.hasOwn(fields, key)).map(([, name]) => `'${name}'`);
    if (missing.length > 0) {
      const names = missing.length === 1 ? missing[0] : `${missing.slice(0, -1).join(', ')} and ${missing.at(-1)}`;
      throw new PyException(
        'TypeError',
        `CandidateIdentity.__init__() missing ${missing.length} required positional argument${missing.length === 1 ? '' : 's'}: ${names}`,
      );
    }
    for (const [key] of IDENTITY_FIELDS) this[key] = fields[key];
    Object.freeze(this);
  }
}

// re.fullmatch(pattern, value), which raises TypeError for a non-str.
function fullmatch(pattern, value) {
  if (typeof value !== 'string') {
    throw new PyException('TypeError', `expected string or bytes-like object, got '${pyTypeName(value)}'`);
  }
  return pattern.exec(value);
}

export function requireApprovedAssetsRepo(value) {
  if (!pyEquals(value, APPROVED_ASSETS_REPOSITORY)) {
    throw new ContractError(`assets repository ${pyRepr(value)} is not approved; expected ${pyRepr(APPROVED_ASSETS_REPOSITORY)}`);
  }
  return value;
}

function requireCommit(value, field) {
  if (fullmatch(COMMIT_RE, value) === null) throw new ContractError(`${field} must be a lowercase full commit SHA`);
  return value;
}

// json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=...) that
// must yield an object; a read or decode error is `could not read <label>`.
function readJson(path, label) {
  let payload;
  try {
    payload = pyJsonLoads(pyReadText(path), { objectPairsHook: rejectDuplicateKeys });
  } catch (error) {
    if (isPyException(error, 'OSError', 'UnicodeDecodeError', 'JSONDecodeError')) {
      throw new ContractError(`could not read ${label}: ${error.message}`, { cause: error });
    }
    throw error;
  }
  if (!isDict(payload)) throw new ContractError(`${label} root must be a JSON object`);
  return payload;
}

function sha256Hex(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function sortedStrings(values) {
  return [...values].sort(compareCodePoints);
}

// Directory paths arrive as Path objects in Python; str() of one is normalized.
function directoryPath(directory) {
  return pyPath(String(directory));
}

// Validate candidate bytes against the current manifest contract.
export function validateCandidate(directory, identity) {
  return validateCandidateContract(directory, identity, {
    expectedQualificationGates: QUALIFICATION_GATES,
    expectedUnprovenCapabilities: UNPROVEN_CAPABILITIES,
  });
}

// Validate immutable published bytes against an explicit historical contract.
export function validatePublishedCandidate(directory, identity, {
  expectedQualificationGates,
  expectedUnprovenCapabilities,
} = {}) {
  return validateCandidateContract(directory, identity, { expectedQualificationGates, expectedUnprovenCapabilities });
}

// Validate candidate bytes against an already selected manifest contract.
//
// The public strict and published-readback entry points select the contract;
// ordinary candidate callers cannot supply historical expectations.
function validateCandidateContract(directory, identity, { expectedQualificationGates, expectedUnprovenCapabilities }) {
  const root = directoryPath(directory);
  if (!pyIsDir(root) || pyIsSymlink(root)) throw new ContractError('candidate must be a real directory');
  const names = pyListdir(root);
  const actualNames = new Set(names);
  const governed = new Set(PUBLICATION_FILES);
  if (actualNames.size !== governed.size || ![...actualNames].every((name) => governed.has(name))) {
    const unexpected = sortedStrings([...actualNames].filter((name) => !governed.has(name)));
    const missing = sortedStrings(PUBLICATION_FILES.filter((name) => !actualNames.has(name)));
    throw new ContractError(
      'candidate directory must contain exactly the governed publication '
      + `files (unexpected: ${pyRepr(unexpected)}, missing: ${pyRepr(missing)})`,
    );
  }
  for (const name of names) {
    const entry = pyJoinPath(root, name);
    if (pyIsSymlink(entry) || !pyIsFile(entry)) throw new ContractError(`candidate entry must be an immutable regular file: ${name}`);
  }

  requireApprovedAssetsRepo(identity.assetsRepo);
  const release = parseReleaseTag(identity.releaseTag);
  if (!pyEquals(release.rebuild, identity.releaseRebuild)) throw new ContractError('release_rebuild does not match release_tag');
  // Bridge asset versions are independent of llama.cpp versions; both tags are
  // still syntactically exact and are recorded separately in the manifest.
  parseUpstreamTag(identity.upstreamTag);
  const nativeRelease = parseReleaseTag(identity.nativeReleaseTag);
  validateNativeIdentity(identity.nativeReleaseTag, nativeRelease.rebuild, identity.upstreamTag);
  requireCommit(identity.bridgeCommit, 'bridge_commit');
  requireCommit(identity.upstreamCommit, 'upstream_commit');
  requireCommit(identity.nativeCommit, 'native_commit');
  requireSha256(identity.nativeManifestSha256, 'native_manifest_sha256');
  requireCorrelationId(identity.orchestratorCorrelationId);
  if (typeof identity.githubRunId !== 'string' || RUN_ID_RE.exec(identity.githubRunId) === null) {
    throw new ContractError('github_run_id must be a positive decimal string');
  }
  const expectedRunUrl = `https://github.com/${BRIDGE_REPOSITORY}/actions/runs/${identity.githubRunId}`;
  if (!pyEquals(identity.githubRunUrl, expectedRunUrl)) throw new ContractError(`github_run_url must be exactly ${expectedRunUrl}`);

  const manifest = readJson(pyJoinPath(root, 'manifest.json'), 'candidate manifest');
  const qualificationGates = pyDict(pyItems(expectedQualificationGates));
  const unprovenCapabilities = pyDict(pyItems(expectedUnprovenCapabilities));
  const expectedFields = [
    ['schema_version', 2],
    ['release_tag', identity.releaseTag],
    ['release_channel', release.channel],
    ['release_rebuild', identity.releaseRebuild],
    ['assets_repository', ASSETS_REPOSITORY],
    ['bridge_repository', BRIDGE_REPOSITORY],
    ['bridge_commit', identity.bridgeCommit],
    ['upstream_repository', 'ggml-org/llama.cpp'],
    ['upstream_tag', identity.upstreamTag],
    ['upstream_commit', identity.upstreamCommit],
    ['native_repository', NATIVE_REPOSITORY],
    ['native_release_tag', identity.nativeReleaseTag],
    ['native_manifest_sha256', identity.nativeManifestSha256],
    ['native_commit', identity.nativeCommit],
    ['emscripten_version', identity.emscriptenVersion],
    ['orchestrator_correlation_id', identity.orchestratorCorrelationId],
    ['github_run_id', identity.githubRunId],
    ['github_run_url', identity.githubRunUrl],
    ['qualification_gates', qualificationGates],
    ['unproven_capabilities', unprovenCapabilities],
    ['capabilities', CAPABILITIES],
    ['bridge_assets_tag', identity.releaseTag],
    ['source_repository', BRIDGE_REPOSITORY],
    ['source_commit', identity.bridgeCommit],
    ['llama_cpp_tag', identity.upstreamTag],
    ['llama_cpp_commit', identity.upstreamCommit],
  ];
  if (!pyKeySetEquals(manifest, [...expectedFields.map(([field]) => field), 'artifacts', 'files'])) {
    throw new ContractError('candidate manifest schema has missing or unexpected fields');
  }
  for (const [field, expected] of expectedFields) {
    if (!pyEquals(pyGet(manifest, field), expected)) {
      throw new ContractError(`candidate manifest ${field} mismatch: expected ${pyRepr(expected)}, got ${pyRepr(pyGet(manifest, field))}`);
    }
  }

  let sumLines;
  try {
    sumLines = pySplitlines(pyReadText(pyJoinPath(root, 'sha256sums.txt')));
  } catch (error) {
    if (isPyException(error, 'OSError', 'UnicodeDecodeError')) {
      throw new ContractError(`could not read candidate checksums: ${error.message}`, { cause: error });
    }
    throw error;
  }
  const sums = new Map();
  for (const line of sumLines) {
    const match = SUM_RE.exec(line);
    if (match === null || sums.has(match[2])) throw new ContractError(`invalid or duplicate checksum line: ${pyRepr(line)}`);
    sums.set(match[2], match[1]);
  }
  if (sums.size !== ARTIFACTS.length || !ARTIFACTS.every((name) => sums.has(name))) {
    throw new ContractError('sha256sums.txt must contain exactly the release artifacts');
  }

  const artifactManifest = pyGet(manifest, 'artifacts');
  const legacyFiles = pyGet(manifest, 'files');
  if (!isDict(artifactManifest) || !pyEquals(artifactManifest, legacyFiles)) {
    throw new ContractError('manifest artifacts/files maps must be identical objects');
  }
  if (!pyKeySetEquals(artifactManifest, ARTIFACTS)) {
    throw new ContractError('manifest artifact map must contain exactly release artifacts');
  }
  for (const name of ARTIFACTS) {
    const path = pyJoinPath(root, name);
    if (!pyIsFile(path)) throw new ContractError(`candidate artifact is missing: ${name}`);
    const data = pyReadBytes(path);
    const digest = sha256Hex(data);
    const metadata = pyGet(artifactManifest, name);
    if (!isDict(metadata) || !pyKeySetEquals(metadata, ['sha256', 'size_bytes'])) {
      throw new ContractError(`manifest metadata schema mismatch for ${name}`);
    }
    if (sums.get(name) !== digest || !pyEquals(pyGet(metadata, 'sha256'), digest)) {
      throw new ContractError(`artifact checksum mismatch for ${name}`);
    }
    if (!pyEquals(pyGet(metadata, 'size_bytes'), data.length)) throw new ContractError(`artifact size mismatch for ${name}`);
  }

  return candidateFingerprint(root);
}

// sha256 over the sorted publication files: name, NUL, the byte length as an
// unsigned 64-bit big-endian integer, then the bytes.
function candidateFingerprint(root) {
  const fingerprint = createHash('sha256');
  for (const name of sortedStrings(PUBLICATION_FILES)) {
    const data = pyReadBytes(pyJoinPath(root, name));
    const length = Buffer.alloc(8);
    length.writeBigUInt64BE(BigInt(data.length));
    fingerprint.update(Buffer.from(`${name}\0`, 'utf8'));
    fingerprint.update(length);
    fingerprint.update(data);
  }
  return fingerprint.digest('hex');
}

// subprocess.run(["git", "-C", repository, *args], stdout=PIPE, stderr=PIPE):
// { returncode, stdout } with stdout as bytes, or as text (strict UTF-8 with
// universal newlines) when `text` is set.
function git(repository, args, { text = false } = {}) {
  const result = spawnSync('git', ['-C', String(repository), ...args], {
    stdio: ['inherit', 'pipe', 'pipe'],
    maxBuffer: Infinity,
  });
  if (result.error) throw pyOSError(result.error, 'git');
  const stdout = text ? pyUniversalNewlines(pyDecodeUtf8(result.stdout)) : result.stdout;
  return { returncode: result.status ?? 1, stdout };
}

function gitFile(repository, commit, name) {
  const result = git(repository, ['show', `${commit}:${name}`]);
  return result.returncode === 0 ? result.stdout : null;
}

function snapshotMatches(repository, commit, candidate) {
  requireCommit(commit, 'snapshot commit');
  return PUBLICATION_FILES.every((name) => {
    const existing = gitFile(repository, commit, name);
    return existing !== null && existing.equals(pyReadBytes(pyJoinPath(candidate, name)));
  });
}

function isAncestor(repository, ancestor, descendant) {
  return git(repository, ['merge-base', '--is-ancestor', ancestor, descendant]).returncode === 0;
}

function manifestHistoryIdentity(repository, commit) {
  const raw = gitFile(repository, commit, 'manifest.json');
  if (raw === null) return null;
  let manifest;
  try {
    manifest = pyJsonLoadsBytes(raw, { objectPairsHook: rejectDuplicateKeys });
  } catch (error) {
    if (error instanceof JSONDecodeError) throw new ContractError(`assets branch manifest is invalid: ${error.message}`, { cause: error });
    throw error;
  }
  if (!isDict(manifest)) throw new ContractError('assets branch manifest root must be an object');
  let releaseTag = pyGet(manifest, 'release_tag');
  const legacyReleaseTag = pyGet(manifest, 'bridge_assets_tag');
  if (releaseTag === null) releaseTag = legacyReleaseTag;
  else if (legacyReleaseTag !== null && !pyEquals(legacyReleaseTag, releaseTag)) {
    throw new ContractError('assets branch manifest release tag aliases conflict');
  }
  let upstreamTag = pyGet(manifest, 'upstream_tag');
  const legacyUpstreamTag = pyGet(manifest, 'llama_cpp_tag');
  if (upstreamTag === null) upstreamTag = legacyUpstreamTag;
  else if (legacyUpstreamTag !== null && !pyEquals(legacyUpstreamTag, upstreamTag)) {
    throw new ContractError('assets branch manifest upstream tag aliases conflict');
  }
  if (typeof releaseTag !== 'string' || typeof upstreamTag !== 'string') {
    throw new ContractError('assets branch manifest is missing release/upstream tags');
  }
  // Both identities are read exactly as recorded. Bridge asset versions have
  // always been independent of the llama.cpp line, in every schema, so neither
  // tag may be fabricated from the other.
  const release = parseReleaseTag(releaseTag, { allowLegacy: true });
  const upstream = parseUpstreamTag(upstreamTag);
  return [release.tag, upstream.tag];
}

// Read every recorded [release_tag, upstream_tag] newest-first.
function historyIdentities(repository, commit) {
  const history = git(repository, ['rev-list', '--first-parent', commit], { text: true });
  if (history.returncode !== 0) throw new ContractError('could not inspect assets branch channel history');
  const identities = [];
  pySplitlines(history.stdout).forEach((historyCommit, index) => {
    const previous = manifestHistoryIdentity(repository, historyCommit);
    if (previous === null) {
      if (index === 0) throw new ContractError('assets branch is missing manifest.json');
      return;
    }
    identities.push(previous);
  });
  return identities;
}

// Order the asset release and upstream lines independently, both fail-closed.
//
// Asset release tags are ordered within their own channel, because stable and
// development asset histories advance independently. The upstream llama.cpp
// line is a single global line, so it is ordered against the newest recorded
// entry regardless of which asset channel published it.
function validateTransition(repository, previous, identity) {
  const candidate = parseReleaseTag(identity.releaseTag);
  const history = historyIdentities(repository, previous);
  const previousIdentity = history.find(([releaseTag]) => parseReleaseTag(releaseTag, { allowLegacy: true }).channel === candidate.channel) ?? null;

  if (previousIdentity === null) {
    if (!pyEquals(candidate.rebuild, 0)) throw new RollbackError('the first artifact in a release channel must use rebuild 0');
  } else {
    const [previousRelease] = previousIdentity;
    let releaseTransition;
    try {
      releaseTransition = compareReleases(previousRelease, identity.releaseTag);
    } catch (error) {
      if (error instanceof ContractError) throw new RollbackError(error.message, { cause: error });
      throw error;
    }
    if (releaseTransition === Transition.EQUAL) {
      throw new ContractError(`release tag ${pyRepr(identity.releaseTag)} already exists in channel history`);
    }
    if (releaseTransition !== Transition.FORWARD) throw new RollbackError(`release transition is ${releaseTransition}`);
  }

  if (history.length > 0) {
    const [, latestUpstream] = history[0];
    const upstreamTransition = compareUpstream(latestUpstream, identity.upstreamTag);
    // A development-to-stable upstream migration is a legal advance; only
    // backward and stable-to-development moves are rollbacks.
    if (![Transition.EQUAL, Transition.FORWARD, Transition.STABLE_MIGRATION].includes(upstreamTransition)) {
      throw new RollbackError(`upstream transition is ${upstreamTransition}`);
    }
  }
}

// str.split(): runs of whitespace, no empty fields.
function splitWhitespace(text) {
  const stripped = pyStrip(text);
  return stripped === '' ? [] : stripped.split(SPLIT_WHITESPACE);
}

function validateCandidateCommit(repository, commit, candidate, identity) {
  if (!snapshotMatches(repository, commit, candidate)) throw new ContractError('publication commit content differs from the candidate');
  const parents = git(repository, ['rev-list', '--parents', '-n', '1', commit], { text: true });
  if (parents.returncode !== 0) throw new ContractError('could not inspect publication commit parent');
  const fields = splitWhitespace(parents.stdout);
  if (fields.length !== 2) throw new ContractError('publication commit must have exactly one parent');
  const parent = fields[1];
  const changed = git(repository, ['diff-tree', '--no-commit-id', '--name-only', '-r', commit], { text: true });
  const changedNames = new Set(pySplitlines(changed.stdout));
  const governedNames = new Set(PUBLICATION_FILES);
  if (changed.returncode !== 0 || !changedNames.has('manifest.json') || ![...changedNames].every((name) => governedNames.has(name))) {
    throw new ContractError('publication commit must change the manifest and only governed release files');
  }
  validateTransition(repository, parent, identity);
}

function validateRelease(release, identity, tagCommit, candidate, fingerprint) {
  const expectedPrerelease = parseReleaseTag(identity.releaseTag).githubPrerelease;
  const releaseId = pyGet(release, 'id');
  const body = () => pyStr(pyGet(release, 'body', ''));
  if (
    !pyEquals(pyGet(release, 'tag_name'), identity.releaseTag)
    || !pyEquals(pyGet(release, 'name'), identity.releaseTag)
    || pyGet(release, 'draft') !== false
    || pyGet(release, 'prerelease') !== expectedPrerelease
    || !pyEquals(pyGet(release, 'target_commitish'), tagCommit)
    || !isPyInt(releaseId)
    || releaseId <= 0
    || !body().includes(`Candidate fingerprint: \`${fingerprint}\``)
    || !body().includes(`Orchestrator correlation: \`${pyStr(identity.orchestratorCorrelationId)}\``)
    || !body().includes(identity.githubRunUrl)
  ) {
    throw new ContractError('GitHub Release metadata does not match the immutable candidate');
  }
  const assets = pyGet(release, 'assets');
  if (!Array.isArray(assets)) throw new ContractError('GitHub Release asset inventory is missing');
  const actual = new Map();
  for (const asset of assets) {
    if (!isDict(asset) || typeof pyGet(asset, 'name') !== 'string') {
      throw new ContractError('GitHub Release contains an invalid asset record');
    }
    const name = pyGet(asset, 'name');
    if (actual.has(name)) throw new ContractError(`GitHub Release contains duplicate asset ${pyRepr(name)}`);
    actual.set(name, asset);
  }
  const governed = new Set(PUBLICATION_FILES);
  if ([...actual.keys()].some((name) => !governed.has(name))) {
    throw new ContractError('GitHub Release asset inventory contains unexpected assets');
  }
  for (const [name, asset] of actual) {
    const data = pyReadBytes(pyJoinPath(candidate, name));
    const assetSize = pyGet(asset, 'size');
    if (
      !pyEquals(pyGet(asset, 'state'), 'uploaded')
      || !isPyInt(assetSize)
      || !pyEquals(assetSize, data.length)
      || !pyEquals(pyGet(asset, 'digest'), `sha256:${sha256Hex(data)}`)
    ) {
      throw new ContractError(`GitHub Release asset digest/size mismatch for ${name}`);
    }
  }
  return sortedStrings(PUBLICATION_FILES.filter((name) => !actual.has(name)));
}

// The truthiness of a release mapping: a non-empty dict.
function hasRelease(release) {
  return release !== null && release !== undefined && pyKeys(release).length > 0;
}

function result({
  identity, state, allowed, action, outcome, reasonCode, fingerprint, branchCommit, tagCommit, release, reason,
  retryable = false,
}) {
  return {
    schema_version: 1,
    state,
    allowed,
    action,
    outcome,
    reason_code: reasonCode,
    reason,
    retryable,
    mutated: false,
    candidate_fingerprint: fingerprint,
    assets_repository: identity.assetsRepo,
    release_tag: identity.releaseTag,
    branch_commit: branchCommit,
    tag_commit: tagCommit ?? null,
    release_id: hasRelease(release) ? pyGet(release, 'id') : null,
    orchestrator_correlation_id: identity.orchestratorCorrelationId,
    github_run_id: identity.githubRunId,
    github_run_url: identity.githubRunUrl,
    qualification_gates: QUALIFICATION_GATES,
  };
}

// Report whether verified remote publication identity changed during this run.
export function publicationStateChanged(before, after) {
  const identity = (state) => {
    const missing = pyGet(state, 'missing_release_assets', []);
    if (!Array.isArray(missing) || !missing.every((name) => typeof name === 'string')) {
      throw new ContractError('publication state has invalid missing_release_assets');
    }
    return [pyGet(state, 'branch_commit'), pyGet(state, 'tag_commit'), pyGet(state, 'release_id'), sortedStrings(missing)];
  };
  return !pyEquals(identity(before), identity(after));
}

const MUTATION_UNKNOWN_REASON_CODES = Object.freeze(['ref-requery-failed', 'release-requery-failed']);

// Emit a durable retry contract when a credentialed mutation cannot be re-read.
export function mutationUnknownOutcome(candidate, identity, reasonCode) {
  if (!MUTATION_UNKNOWN_REASON_CODES.some((code) => pyEquals(code, reasonCode))) {
    throw new ContractError('unsupported mutation-unknown reason_code');
  }
  const fingerprint = validateCandidate(candidate, identity);
  return {
    schema_version: 1,
    state: 'mutation-unknown',
    allowed: false,
    action: 'none',
    outcome: 'mutation-unknown',
    reason_code: reasonCode,
    reason: 'a credentialed mutation was attempted but exact remote state could not be re-read',
    retryable: true,
    mutated: null,
    mutation_status: 'unknown',
    candidate_fingerprint: fingerprint,
    assets_repository: identity.assetsRepo,
    release_tag: identity.releaseTag,
    branch_commit: null,
    tag_commit: null,
    release_id: null,
    orchestrator_correlation_id: identity.orchestratorCorrelationId,
    github_run_id: identity.githubRunId,
    github_run_url: identity.githubRunUrl,
    qualification_gates: QUALIFICATION_GATES,
  };
}

// Convert only an unavailable or semantically invalid re-query to unknown.
export function mutationUnknownFromRequery(candidate, identity, reasonCode, requeryPath) {
  let data;
  try {
    data = pyReadBytes(requeryPath);
  } catch (error) {
    if (!isPyException(error, 'OSError')) throw error;
    data = Buffer.alloc(0);
  }
  if (pyBytesStrip(data).length > 0) {
    let parsed;
    try {
      parsed = pyJsonLoadsBytes(data, { objectPairsHook: rejectDuplicateKeys });
    } catch (error) {
      if (!(error instanceof ContractError) && !isPyException(error, 'UnicodeDecodeError', 'JSONDecodeError')) throw error;
      parsed = null;
    }
    if (isDict(parsed) && !pyEquals(pyGet(parsed, 'reason_code'), 'invalid-input-or-state')) {
      throw new ContractError('valid classifier JSON must be handled as exact state, not mutation-unknown');
    }
  }
  return mutationUnknownOutcome(candidate, identity, reasonCode);
}

export function classify({
  repository, candidate, identity, branchCommit, tagCommit = null, release = null,
}) {
  const fingerprint = validateCandidate(candidate, identity);
  requireCommit(branchCommit, 'branch_commit');
  const tag = tagCommit ?? null;
  const releaseData = release ?? null;
  if (tag !== null) requireCommit(tag, 'tag_commit');
  const repositoryPath = directoryPath(repository);
  const candidatePath = directoryPath(candidate);

  const outcome = (fields) => result({
    identity, fingerprint, branchCommit, tagCommit: tag, release: releaseData, ...fields,
  });

  try {
    if (releaseData !== null && tag === null) throw new ContractError('GitHub Release exists without the immutable git tag');
    if (tag !== null) {
      if (!isAncestor(repositoryPath, tag, branchCommit)) throw new ContractError('existing tag is not reachable from the assets branch');
      validateCandidateCommit(repositoryPath, tag, candidatePath, identity);
      if (releaseData !== null) {
        const missingAssets = validateRelease(releaseData, identity, tag, candidatePath, fingerprint);
        if (missingAssets.length > 0) {
          const partial = outcome({
            state: 'release-assets-partial',
            allowed: false,
            action: 'none',
            outcome: 'immutable-publication-unverified',
            reasonCode: 'immutable-release-assets-missing',
            reason: 'published release asset inventory is incomplete; immutable releases must never be repaired or overwritten',
          });
          partial.missing_release_assets = missingAssets;
          return partial;
        }
        return outcome({
          state: 'complete',
          allowed: true,
          action: 'none',
          outcome: 'already-complete',
          reasonCode: 'exact-release-complete',
          reason: 'tag, release metadata, and every asset exactly match',
        });
      }
      return outcome({
        state: 'tag-without-release',
        allowed: true,
        action: 'create-release',
        outcome: 'safely-resumed',
        reasonCode: 'exact-tag-release-missing',
        reason: 'exact reachable tag exists and only GitHub Release is missing',
        retryable: true,
      });
    }

    if (snapshotMatches(repositoryPath, branchCommit, candidatePath)) {
      validateCandidateCommit(repositoryPath, branchCommit, candidatePath, identity);
      return outcome({
        state: 'branch-only',
        allowed: true,
        action: 'publish-tag-and-release',
        outcome: 'safely-resumed',
        reasonCode: 'exact-branch-tag-release-missing',
        reason: 'exact governed publication commit is on the branch',
        retryable: true,
      });
    }

    validateTransition(repositoryPath, branchCommit, identity);
    return outcome({
      state: 'absent',
      allowed: true,
      action: 'publish-refs-and-release',
      outcome: 'newly-published',
      reasonCode: 'new-publication',
      reason: 'candidate legally advances the current assets branch',
    });
  } catch (error) {
    if (!(error instanceof ContractError)) throw error;
    const rollback = error instanceof RollbackError;
    return outcome({
      state: rollback ? 'rollback' : 'collision',
      allowed: false,
      action: 'none',
      outcome: rollback ? 'rollback' : 'collision',
      reasonCode: rollback ? 'ordering-rollback' : 'identity-collision',
      reason: error.message,
    });
  }
}

export function candidatePublicationDigests(candidate) {
  const root = directoryPath(candidate);
  if (!pyIsDir(root) || pyIsSymlink(root)) throw new ContractError('candidate must be a real directory');
  const names = pyListdir(root);
  const actualNames = new Set(names);
  if (actualNames.size !== PUBLICATION_FILES.length || !PUBLICATION_FILES.every((name) => actualNames.has(name))) {
    throw new ContractError('candidate directory must contain exactly the governed publication files');
  }
  for (const name of names) {
    const entry = pyJoinPath(root, name);
    if (pyIsSymlink(entry) || !pyIsFile(entry)) throw new ContractError(`candidate entry must be an immutable regular file: ${name}`);
  }
  return pyDict(PUBLICATION_FILES.map((name) => [name, sha256Hex(pyReadBytes(pyJoinPath(root, name)))]));
}

// Prove a just-published release is immutable and attested, or fail closed.
//
// Both readbacks are independent reads of the same release -- by tag and by
// ID -- so a tag that silently resolves elsewhere cannot satisfy the gate.
// Nothing here mutates, deletes, retags, or repairs remote state: a mismatch
// is reported and the release is left exactly as GitHub created it.
export function verifyImmutablePublication({
  candidate, assetsRepo, releaseTag, tagCommit, releaseId = null, releaseByTag, releaseById, attestation,
}) {
  requireApprovedAssetsRepo(assetsRepo);
  requireCommit(tagCommit, 'tag_commit');
  const digests = candidatePublicationDigests(candidate);
  const resolvedId = validateReleaseImmutability(releaseByTag, { releaseTag, tagCommit, releaseId: releaseId ?? null });
  const byIdResolved = validateReleaseImmutability(releaseById, { releaseTag, tagCommit, releaseId: resolvedId });
  if (!pyEquals(byIdResolved, resolvedId)) throw new ContractError('release readbacks by tag and by ID identify different releases');
  if (!pyEquals(pyGet(releaseByTag, 'published_at'), pyGet(releaseById, 'published_at'))) {
    throw new ContractError('release readbacks disagree on published_at');
  }
  const verified = validateReleaseAttestation(attestation, {
    assetsRepo,
    releaseTag,
    tagCommit,
    releaseId: resolvedId,
    expectedAssets: digests,
  });
  return {
    schema_version: 1,
    assets_repository: assetsRepo,
    release_tag: releaseTag,
    release_id: resolvedId,
    tag_commit: tagCommit,
    immutable: true,
    published_at: pyGet(releaseByTag, 'published_at'),
    attestation_predicate_type: IMMUTABLE_RELEASE_ATTESTATION_PREDICATE_TYPE,
    attestation_signer: IMMUTABLE_RELEASE_ATTESTATION_SIGNER,
    attested_purl: verified.purl,
    verified_timestamps: verified.verified_timestamps,
    attested_assets: verified.assets,
  };
}

function identityFromArgs(args) {
  return new CandidateIdentity(Object.fromEntries(IDENTITY_FIELDS.map(([key]) => [key, args[key]])));
}

const required = (flag, type = 'str') => ({ flag, required: true, type });
const optional = (flag, type = 'str') => ({ flag, required: false, type });

const IDENTITY_OPTIONS = Object.freeze([
  required('--release-tag'),
  required('--release-rebuild', 'int'),
  required('--assets-repo'),
  required('--bridge-commit'),
  required('--upstream-tag'),
  required('--upstream-commit'),
  required('--native-release-tag'),
  required('--native-manifest-sha256'),
  required('--native-commit'),
  required('--emscripten-version'),
  required('--orchestrator-correlation-id'),
  required('--github-run-id'),
  required('--github-run-url'),
]);

export const COMMANDS = Object.freeze({
  'validate-target': { options: [required('--assets-repo')] },
  'verify-immutable-publication': {
    options: [
      required('--candidate', 'path'),
      required('--assets-repo'),
      required('--release-tag'),
      required('--tag-commit'),
      optional('--release-id', 'int'),
      required('--release-json', 'path'),
      required('--release-by-id-json', 'path'),
      required('--attestation-json', 'path'),
    ],
  },
  'state-changed': { options: [required('--before-json', 'path'), required('--after-json', 'path')] },
  'mutation-unknown': {
    options: [
      required('--candidate', 'path'),
      required('--reason-code'),
      required('--requery-json', 'path'),
      ...IDENTITY_OPTIONS,
    ],
  },
  classify: {
    options: [
      required('--repository', 'path'),
      required('--candidate', 'path'),
      required('--branch-commit'),
      optional('--tag-commit'),
      optional('--release-json', 'path'),
      ...IDENTITY_OPTIONS,
    ],
  },
});

export function fatal(reason, identity = null) {
  const outcome = {
    schema_version: 1,
    state: 'collision',
    allowed: false,
    action: 'none',
    outcome: 'collision',
    reason_code: 'invalid-input-or-state',
    reason,
    retryable: false,
    mutated: false,
  };
  if (identity !== null) {
    Object.assign(outcome, {
      assets_repository: identity.assetsRepo,
      release_tag: identity.releaseTag,
      orchestrator_correlation_id: identity.orchestratorCorrelationId,
      github_run_id: identity.githubRunId,
      github_run_url: identity.githubRunUrl,
      qualification_gates: QUALIFICATION_GATES,
    });
  }
  return outcome;
}

function dumps(value) {
  return `${pyJsonDumps(value, { sortKeys: true })}\n`;
}

// Run one subcommand; returns the exit status, or the outcome to print.
function runCommand(command, args, write) {
  if (command === 'validate-target') {
    requireApprovedAssetsRepo(args.assetsRepo);
    write(`${pyJsonDumps({ allowed: true, assets_repository: args.assetsRepo })}\n`);
    return 0;
  }
  if (command === 'verify-immutable-publication') {
    // This gate reports immutability, never publication state, so it exits
    // with a plain error instead of a classifier outcome.
    let verified;
    try {
      verified = verifyImmutablePublication({
        candidate: args.candidate,
        assetsRepo: args.assetsRepo,
        releaseTag: args.releaseTag,
        tagCommit: args.tagCommit,
        releaseId: args.releaseId,
        releaseByTag: readJson(args.releaseJson, 'published GitHub Release'),
        releaseById: readJson(args.releaseByIdJson, 'published GitHub Release by ID'),
        attestation: readJson(args.attestationJson, 'GitHub release attestation'),
      });
    } catch (error) {
      if (error instanceof ContractError || isPyException(error, 'OSError')) throw new SystemExitError(`error: ${error.message}`);
      throw error;
    }
    write(dumps(verified));
    return 0;
  }
  if (command === 'state-changed') {
    const before = readJson(args.beforeJson, 'before publication state');
    const after = readJson(args.afterJson, 'after publication state');
    write(`${pyJsonDumps(publicationStateChanged(before, after))}\n`);
    return 0;
  }
  if (command === 'mutation-unknown') {
    write(dumps(mutationUnknownFromRequery(args.candidate, identityFromArgs(args), args.reasonCode, args.requeryJson)));
    return 0;
  }
  const release = args.releaseJson !== null ? readJson(args.releaseJson, 'GitHub Release') : null;
  return classify({
    repository: args.repository,
    candidate: args.candidate,
    identity: identityFromArgs(args),
    branchCommit: args.branchCommit,
    tagCommit: args.tagCommit || null,
    release,
  });
}

// The CLI: `main(argv, write)` parses argv like release_publication_state.py's
// argparse parser, prints through write() and returns the exit status: an
// outcome exits 0 exactly when it is allowed, and a contract or file error is
// printed as the invalid-input-or-state outcome.
export function main(argv, write) {
  const { command, args } = parseCommandLine(argv, { prog: progName(import.meta.url), commands: COMMANDS });
  let outcome;
  try {
    outcome = runCommand(command, args, write);
    if (typeof outcome === 'number') return outcome;
  } catch (error) {
    if (!(error instanceof ContractError) && !isPyException(error, 'OSError')) throw error;
    const identity = command === 'classify' || command === 'mutation-unknown' ? identityFromArgs(args) : null;
    outcome = fatal(error.message, identity);
  }
  write(dumps(outcome));
  return pyGet(outcome, 'allowed') === true ? 0 : 1;
}

export { ArgparseExit };

if (import.meta.main) runCli(main);
