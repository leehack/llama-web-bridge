#!/usr/bin/env node
// Release identity and ordering contract for Web bridge asset publication,
// ported from scripts/release_contract.py (deleted at the harness 5.0.0
// cutover).
//
// GitHub release tags are shared with the native release convention. Stable
// releases use `vMAJOR.MINOR.PATCH` and rebuilds append `-N`. Development
// releases use `bNNNN` and rebuilds append `-N`. Bridge asset releases keep
// npm-compatible ordering, so a new bridge asset release never carries `-N`
// (validateNewReleaseIdentity); suffixed bridge tags published before that
// rule stay readable. Historical `*-llamadart.N` wrappers remain readable so an
// existing manifest can be used as an ordering boundary, but this module never
// emits them.
//
// Every validator accepts and rejects exactly what the Python module did,
// with the same ContractError text; values follow the JSON value model of
// ./json.mjs (int vs PyFloat, Python equality and repr).

import { createHash } from 'node:crypto';

import {
  ArgparseExit, SystemExitError, parseCommandLine, progName, runCli,
} from './cli.mjs';
import { ContractError } from './errors.mjs';
import {
  JSONDecodeError, PY_NON_SPACE_CLASS, PyException, isDict, isPyException, isPyInt,
  isPyIntLike, pyB64decodeValidate, pyDecodeUtf8, pyDict, pyEquals, pyGet,
  pyHasKey, pyHookRepr, pyIntFromString, pyItems, pyJsonDumps, pyJsonLoads, pyKeySetEquals, pyKeys,
  pyReadBytes, pyReadText, pyRepr, pySplitlines, pyStr, pyStrftimeUtc, pyStrip,
  pyStrptimeUtc, pyTypeName, PyFloat,
} from './json.mjs';

export { ContractError };

export const BRIDGE_REPOSITORY = 'leehack/llama-web-bridge';
export const ASSETS_REPOSITORY = 'leehack/llama-web-bridge-assets';
export const NATIVE_REPOSITORY = 'leehack/llamadart-native';
export const NATIVE_HOOK_CONTRACT_VERSION = 1;

// The heavy real-model ASR/TTS gates run in their own hosted qualification run
// rather than inside the candidate build, so a candidate declares the pending
// requirement and a published manifest declares the satisfied requirement.
export const AUTOMATED_QUALIFICATION_REQUIRED = 'required-automated-qualification';
export const AUTOMATED_QUALIFICATION_PENDING = 'pending-automated-qualification';

// Qualification proves transcript, lifecycle, and WAV container correctness on
// a hosted runner. Nothing listens to generated audio on a real device, hosted
// runners expose no real GPU, and the pinned Qwen3-TTS pair is memory64-only.
export const UNPROVEN_CAPABILITIES = Object.freeze({
  hardware_gpu_acceleration: 'unavailable-on-hosted-runners',
  real_device_intelligibility: 'unproven',
  real_device_playback: 'unproven',
  speaker_reference_fidelity: 'unproven',
  wasm32_text_to_speech: 'unsupported',
});

// GitHub exposes exactly three supported signals that a published release can
// never be moved: repository immutable-release governance
// (`GET /repos/{owner}/{repo}/immutable-releases`), the release object's own
// `immutable` boolean, and the release attestation GitHub signs for immutable
// releases. Nothing else is invented here.
export const IMMUTABLE_RELEASE_ATTESTATION_PREDICATE_TYPE = 'https://in-toto.io/attestation/release/v0.2';
export const IMMUTABLE_RELEASE_ATTESTATION_SIGNER = 'https://dotcom.releases.github.com';
const IN_TOTO_STATEMENT_TYPE = 'https://in-toto.io/Statement/v1';
const DSSE_IN_TOTO_PAYLOAD_TYPE = 'application/vnd.in-toto+json';
const SIGSTORE_BUNDLE_MEDIA_TYPE = 'application/vnd.dev.sigstore.bundle.v0.3+json';
const SIGSTORE_VERIFICATION_RESULT_MEDIA_TYPE = 'application/vnd.dev.sigstore.verificationresult+json;version=0.1';
const RELEASE_SIGNER_IDENTITY_REGEXP = '^https://dotcom\\.releases\\.github\\.com$';

// Each pattern is anchored: re.fullmatch.
const ATTESTATION_ASSET_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/u;
const SHA256_RE = /^[0-9a-f]{64}$/u;
const COMMIT_RE = /^[0-9a-f]{40}$/u;
const HEX_COMMITISH_RE = /^[0-9A-Fa-f]{4,40}$/u;
const REPO_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/u;
const CORRELATION_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/u;
const UPSTREAM_STABLE_RE = /^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/u;
const UPSTREAM_DEVELOPMENT_RE = /^b(0|[1-9][0-9]*)$/u;
const STABLE_RELEASE_RE = /^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([1-9][0-9]*))?$/u;
const DEVELOPMENT_RELEASE_RE = /^b(0|[1-9][0-9]*)(?:-([1-9][0-9]*))?$/u;
const LEGACY_STABLE_RE = /^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)-llamadart\.([1-9][0-9]*)$/u;
const LEGACY_DEVELOPMENT_RE = /^b(0|[1-9][0-9]*)-llamadart\.([1-9][0-9]*)$/u;
const LS_REMOTE_LINE_RE = new RegExp(`^([0-9a-f]{40})\\t(${PY_NON_SPACE_CLASS}+)$`, 'u');
const SHA256SUMS_LINE_RE = /^([0-9a-f]{64}) {2}([A-Za-z0-9_.-]+)$/u;
const POSITIVE_DECIMAL_RE = /^[1-9][0-9]*$/u;

// re.fullmatch(pattern, value), which raises TypeError for a non-str.
function fullmatch(pattern, value) {
  if (typeof value !== 'string') {
    throw new PyException('TypeError', `expected string or bytes-like object, got '${pyTypeName(value)}'`);
  }
  return pattern.exec(value);
}

// The sign of `value - 0` for an int (bool included) or float, raising
// TypeError for anything else as Python's `value <op> 0` does.
function compareToZero(value, operator) {
  if (typeof value === 'boolean' || isPyInt(value)) {
    const number = BigInt(value);
    return number > 0n ? 1 : number < 0n ? -1 : 0;
  }
  if (value instanceof PyFloat || typeof value === 'number') return Math.sign(Number(value instanceof PyFloat ? value.value : value));
  throw new PyException('TypeError', `'${operator}' not supported between instances of '${pyTypeName(value)}' and 'int'`);
}

function compareInts(left, right) {
  const a = BigInt(left);
  const b = BigInt(right);
  if (a < b) return -1;
  return a > b ? 1 : 0;
}

// Tuple ordering of int tuples.
function compareIntTuples(left, right) {
  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    const order = compareInts(left[index], right[index]);
    if (order !== 0) return order;
  }
  return left.length - right.length;
}

function intsEqual(left, right) {
  return left.length === right.length && compareIntTuples(left, right) === 0;
}

// dict(pairs) that rejects a repeated key: release_contract's object_pairs_hook.
export function rejectDuplicateKeys(pairs, frame = null) {
  const seen = new Set();
  for (const [key] of pairs) {
    if (seen.has(key)) throw new ContractError(`duplicate JSON key: ${pyHookRepr(frame, key)}`);
    seen.add(key);
  }
  return pyDict(pairs);
}

// json.loads(text, object_pairs_hook=_reject_duplicate_keys), with a decode
// error reported as `could not parse <label>: <error>`. NaN and Infinity are
// accepted, as the Python loader accepts them.
export function strictJsonLoads(text, label) {
  try {
    return pyJsonLoads(text, { objectPairsHook: rejectDuplicateKeys });
  } catch (error) {
    if (error instanceof JSONDecodeError) throw new ContractError(`could not parse ${label}: ${error.message}`, { cause: error });
    throw error;
  }
}

export const Channel = Object.freeze({
  DEVELOPMENT: 'development',
  STABLE: 'stable',
});

export const Transition = Object.freeze({
  EQUAL: 'equal',
  FORWARD: 'forward',
  BACKWARD: 'backward',
  STABLE_MIGRATION: 'stable-migration',
  FORBIDDEN_STABLE_TO_DEVELOPMENT: 'forbidden-stable-to-development',
});

export class UpstreamVersion {
  constructor(tag, channel, parts) {
    this.tag = tag;
    this.channel = channel;
    this.parts = Object.freeze([...parts]);
    Object.freeze(this);
  }
}

// A release tag's own identity, independent of any upstream llama.cpp tag.
//
// Bridge asset releases version independently of llama.cpp, so versionParts is
// the tag's own version. Native releases separately promise that their base
// tag *is* their upstream tag; only the native validators enforce that.
export class ReleaseVersion {
  constructor(tag, channel, versionParts, rebuild, legacy = false) {
    this.tag = tag;
    this.channel = channel;
    this.versionParts = Object.freeze([...versionParts]);
    this.rebuild = rebuild;
    this.legacy = legacy;
    Object.freeze(this);
  }

  // The tag with its rebuild suffix removed.
  get baseTag() {
    if (this.channel === Channel.DEVELOPMENT) return `b${this.versionParts[0]}`;
    return `v${this.versionParts.join('.')}`;
  }

  // Match llamadart-native: nightlies and every rebuild are prereleases.
  get githubPrerelease() {
    return this.channel === Channel.DEVELOPMENT || compareInts(this.rebuild, 0) > 0;
  }
}

export class NativeIdentity {
  constructor({ releaseTag, upstreamTag, upstreamCommit, nativeCommit }) {
    this.releaseTag = releaseTag;
    this.upstreamTag = upstreamTag;
    this.upstreamCommit = upstreamCommit;
    this.nativeCommit = nativeCommit;
    Object.freeze(this);
  }

  // The dataclass's __dict__, as the CLI prints it.
  toDict() {
    return {
      release_tag: this.releaseTag,
      upstream_tag: this.upstreamTag,
      upstream_commit: this.upstreamCommit,
      native_commit: this.nativeCommit,
    };
  }
}

function requireCommit(value, field) {
  if (typeof value !== 'string' || COMMIT_RE.exec(value) === null) {
    throw new ContractError(`${field} must be a lowercase full 40-character commit SHA`);
  }
  return value;
}

export function requireSha256(value, field = 'sha256') {
  if (fullmatch(SHA256_RE, value) === null) throw new ContractError(`${field} must be a lowercase 64-character SHA-256`);
  return value;
}

export function requireRepository(value, field) {
  if (fullmatch(REPO_RE, value) === null) throw new ContractError(`${field} must use owner/repository syntax`);
  return value;
}

export function requireCorrelationId(value) {
  if (fullmatch(CORRELATION_ID_RE, value) === null) {
    throw new ContractError('orchestrator_correlation_id must be 1-128 safe identifier characters');
  }
  return value;
}

export function parseUpstreamTag(tag) {
  const development = fullmatch(UPSTREAM_DEVELOPMENT_RE, tag);
  if (development) return new UpstreamVersion(tag, Channel.DEVELOPMENT, [pyIntFromString(development[1])]);

  const stable = fullmatch(UPSTREAM_STABLE_RE, tag);
  if (stable) return new UpstreamVersion(tag, Channel.STABLE, stable.slice(1, 4).map((text) => pyIntFromString(text)));

  throw new ContractError(`unsupported upstream tag ${pyRepr(tag)}; expected exact vMAJOR.MINOR.PATCH or bNNNN`);
}

export function parseReleaseTag(tag, { allowLegacy = false } = {}) {
  const stable = fullmatch(STABLE_RELEASE_RE, tag);
  if (stable) return new ReleaseVersion(tag, Channel.STABLE, stable.slice(1, 4).map((text) => pyIntFromString(text)), pyIntFromString(stable[4] ?? '0'));

  const development = fullmatch(DEVELOPMENT_RELEASE_RE, tag);
  if (development) {
    return new ReleaseVersion(tag, Channel.DEVELOPMENT, [pyIntFromString(development[1])], pyIntFromString(development[2] ?? '0'));
  }

  if (allowLegacy) {
    const legacyStable = fullmatch(LEGACY_STABLE_RE, tag);
    if (legacyStable) {
      const [major, minor, patch, rebuild] = legacyStable.slice(1, 5).map((text) => pyIntFromString(text));
      return new ReleaseVersion(tag, Channel.STABLE, [major, minor, patch], rebuild, true);
    }
    const legacyDevelopment = fullmatch(LEGACY_DEVELOPMENT_RE, tag);
    if (legacyDevelopment) {
      const [build, rebuild] = legacyDevelopment.slice(1, 3).map((text) => pyIntFromString(text));
      return new ReleaseVersion(tag, Channel.DEVELOPMENT, [build], rebuild, true);
    }
  }

  const suffix = allowLegacy ? '' : ' (legacy *-llamadart.N is read-only)';
  throw new ContractError(`unsupported release tag ${pyRepr(tag)}; expected vMAJOR.MINOR.PATCH[-N] or bNNNN[-N]${suffix}`);
}

// Order two same-kind identities: within a channel, then across channels.
function compareVersions(currentChannel, currentParts, targetChannel, targetParts) {
  if (currentChannel === targetChannel) {
    if (intsEqual(currentParts, targetParts)) return Transition.EQUAL;
    return compareIntTuples(targetParts, currentParts) > 0 ? Transition.FORWARD : Transition.BACKWARD;
  }
  if (currentChannel === Channel.DEVELOPMENT && targetChannel === Channel.STABLE) return Transition.STABLE_MIGRATION;
  return Transition.FORBIDDEN_STABLE_TO_DEVELOPMENT;
}

export function compareUpstream(currentTag, targetTag) {
  const current = parseUpstreamTag(currentTag);
  const target = parseUpstreamTag(targetTag);
  return compareVersions(current.channel, current.parts, target.channel, target.parts);
}

// Order two release tags by their own identity, never by an upstream tag.
export function compareReleases(currentTag, targetTag) {
  const current = parseReleaseTag(currentTag, { allowLegacy: true });
  const target = parseReleaseTag(targetTag);

  const transition = compareVersions(current.channel, current.versionParts, target.channel, target.versionParts);
  if (transition === Transition.EQUAL) {
    const order = compareInts(target.rebuild, current.rebuild);
    if (order === 0) return Transition.EQUAL;
    return order > 0 ? Transition.FORWARD : Transition.BACKWARD;
  }

  if ((transition === Transition.FORWARD || transition === Transition.STABLE_MIGRATION) && compareInts(target.rebuild, 0) !== 0) {
    throw new ContractError('the first artifact for a new release version must use rebuild 0');
  }
  return transition;
}

// Validate a bridge asset release tag and its independent upstream tag.
//
// Bridge assets version independently of llama.cpp: v0.1.38 may ship upstream
// v0.2.0. Both identities are still syntactically exact, and the tag must
// encode the requested rebuild.
export function validateReleaseIdentity(releaseTag, rebuild, upstreamTag) {
  if (compareToZero(rebuild, '<') < 0) throw new ContractError('release_rebuild must be zero or greater');
  const release = parseReleaseTag(releaseTag);
  parseUpstreamTag(upstreamTag);
  if (!pyEquals(release.rebuild, rebuild)) {
    throw new ContractError(`release tag ${pyRepr(releaseTag)} encodes rebuild ${release.rebuild}, not ${pyStr(rebuild)}`);
  }
  return release;
}

// Validate a bridge asset release that is about to be built or published.
//
// Bridge asset tags keep npm-compatible ordering, so a new release is always an
// unsuffixed tag with rebuild 0. npm orders vMAJOR.MINOR.PATCH-N as a
// prerelease before vMAJOR.MINOR.PATCH, which would rank newer assets below
// older ones. Suffixed tags published before this rule, such as v0.1.47-1,
// stay readable through validateReleaseIdentity.
export function validateNewReleaseIdentity(releaseTag, rebuild, upstreamTag) {
  const release = validateReleaseIdentity(releaseTag, rebuild, upstreamTag);
  if (compareInts(release.rebuild, 0) !== 0) {
    throw new ContractError(`new bridge asset release tag ${pyRepr(releaseTag)} must not carry a rebuild suffix; take the next free unsuffixed version instead`);
  }
  return release;
}

// Native releases, unlike bridge assets, encode their upstream tag.
function requireNativeUpstreamIdentity(nativeVersion, upstream) {
  if (nativeVersion.channel !== upstream.channel || !intsEqual(nativeVersion.versionParts, upstream.parts)) {
    throw new ContractError(`native release ${pyRepr(nativeVersion.tag)} does not preserve upstream identity ${pyRepr(upstream.tag)}`);
  }
}

export function validateNativeIdentity(nativeReleaseTag, rebuild, upstreamTag) {
  if (compareToZero(rebuild, '<') < 0) throw new ContractError('release_rebuild must be zero or greater');
  const nativeRelease = parseReleaseTag(nativeReleaseTag);
  const upstream = parseUpstreamTag(upstreamTag);
  if (!pyEquals(nativeRelease.rebuild, rebuild)) {
    throw new ContractError(`native release tag ${pyRepr(nativeReleaseTag)} encodes rebuild ${nativeRelease.rebuild}, not ${pyStr(rebuild)}`);
  }
  requireNativeUpstreamIdentity(nativeRelease, upstream);
  return nativeRelease;
}

// Validate native release inputs before using them in network requests.
export function validateNativeRequest(nativeReleaseTag, upstreamTag, upstreamCommit, manifestSha256) {
  const nativeRelease = parseReleaseTag(nativeReleaseTag);
  validateNativeIdentity(nativeReleaseTag, nativeRelease.rebuild, upstreamTag);
  requireCommit(upstreamCommit, 'upstream_commit');
  requireSha256(manifestSha256, 'native_manifest_sha256');
  return nativeRelease;
}

export function validateGithubPrerelease(tag, actual, { allowLegacy = true } = {}) {
  const version = parseReleaseTag(tag, { allowLegacy });
  if (actual !== version.githubPrerelease) {
    throw new ContractError(`GitHub prerelease state for ${pyRepr(tag)} must be ${pyRepr(version.githubPrerelease)}`);
  }
  return version.githubPrerelease;
}

export function resolveNativeManifest(manifest, releaseTag) {
  const legacyTag = pyGet(manifest, 'tag');
  const nativeTag = pyGet(manifest, 'native_release_tag', legacyTag);
  if (typeof nativeTag !== 'string' || !nativeTag) throw new ContractError('native manifest is missing native_release_tag/tag');
  if (legacyTag !== null && !pyEquals(legacyTag, nativeTag)) {
    throw new ContractError('native_release_tag does not match the legacy tag alias');
  }
  if (!pyEquals(nativeTag, releaseTag)) {
    throw new ContractError(`native manifest tag ${pyRepr(nativeTag)} does not match release ${pyRepr(releaseTag)}`);
  }

  const nativeVersion = parseReleaseTag(nativeTag, { allowLegacy: true });
  let upstreamTag = pyGet(manifest, 'llama_cpp_tag');
  if (upstreamTag === null && compareInts(nativeVersion.rebuild, 0) === 0) upstreamTag = nativeVersion.baseTag;
  if (typeof upstreamTag !== 'string') throw new ContractError('native manifest is missing an exact llama_cpp_tag');
  const upstream = parseUpstreamTag(upstreamTag);
  requireNativeUpstreamIdentity(nativeVersion, upstream);

  const upstreamCommit = requireCommit(pyGet(manifest, 'llama_cpp_commit'), 'llama_cpp_commit');
  const nativeCommit = requireCommit(pyGet(manifest, 'native_commit'), 'native_commit');
  return new NativeIdentity({ releaseTag: nativeTag, upstreamTag, upstreamCommit, nativeCommit });
}

function sha256Hex(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

export function validateNativeFile(path, expectedSha256, nativeReleaseTag, upstreamTag, upstreamCommit) {
  const expected = requireSha256(expectedSha256, 'native_manifest_sha256');
  const actualSha256 = sha256Hex(pyReadBytes(path));
  if (actualSha256 !== expected) {
    throw new ContractError(`native manifest SHA-256 mismatch: expected ${expected}, got ${actualSha256}`);
  }
  let text;
  try {
    text = pyReadText(path);
  } catch (error) {
    if (isPyException(error, 'OSError', 'UnicodeDecodeError')) {
      throw new ContractError(`could not read native manifest: ${error.message}`, { cause: error });
    }
    throw error;
  }
  const payload = strictJsonLoads(text, 'native manifest');
  if (!isDict(payload)) throw new ContractError('native manifest root must be a JSON object');
  const identity = resolveNativeManifest(payload, nativeReleaseTag);
  if (!pyEquals(identity.upstreamTag, upstreamTag)) {
    throw new ContractError(`native manifest upstream tag ${pyRepr(identity.upstreamTag)} does not match ${pyRepr(upstreamTag)}`);
  }
  if (!pyEquals(identity.upstreamCommit, upstreamCommit)) {
    throw new ContractError('native manifest llama_cpp_commit does not match the requested upstream commit');
  }
  return identity;
}

const NATIVE_ARTIFACTS = Object.freeze({
  'android-arm64': ['android', 'arm64', 'core', 'core'],
  'android-x64': ['android', 'x64', 'core', 'core'],
  'ios-arm64': ['ios', 'arm64', 'core', 'core'],
  'ios-arm64-sim': ['ios', 'arm64-sim', 'core', 'core'],
  'ios-x86_64-sim': ['ios', 'x86_64-sim', 'core', 'core'],
  'linux-arm64': ['linux', 'arm64', 'core', 'core'],
  'linux-x64': ['linux', 'x64', 'core', 'core'],
  'macos-arm64': ['macos', 'arm64', 'core', 'core'],
  'macos-x86_64': ['macos', 'x86_64', 'core', 'core'],
  'windows-arm64': ['windows', 'arm64', 'core', 'core'],
  'windows-x64': ['windows', 'x64', 'core', 'core'],
});

// File name -> [platform, arch, backend, module] for every artifact a native
// release must carry.
export function expectedNativeArtifacts(tag) {
  const pairs = Object.entries(NATIVE_ARTIFACTS)
    .map(([bundle, metadata]) => [`llamadart-native-${bundle}-${pyStr(tag)}.tar.gz`, Object.freeze([...metadata])]);
  pairs.push([`llamadart-native-apple-xcframework-${pyStr(tag)}.zip`, Object.freeze(['apple', 'universal', 'core', 'spm-xcframework'])]);
  pairs.push([`llamadart-native-headers-${pyStr(tag)}.tar.gz`, Object.freeze(['all', 'universal', 'core', 'headers'])]);
  return pyDict(pairs);
}

function releaseAssets(release) {
  const rawAssets = pyGet(release, 'assets');
  if (!Array.isArray(rawAssets)) throw new ContractError('native GitHub release is missing its asset inventory');
  const assets = new Map();
  for (const item of rawAssets) {
    if (!isDict(item) || typeof pyGet(item, 'name') !== 'string') {
      throw new ContractError('native GitHub release contains an invalid asset record');
    }
    const name = item.name;
    if (assets.has(name)) throw new ContractError(`native GitHub release has duplicate asset ${pyRepr(name)}`);
    if (pyGet(item, 'state') !== 'uploaded') throw new ContractError(`native GitHub asset ${pyRepr(name)} is not uploaded`);
    const digest = pyGet(item, 'digest');
    if (typeof digest !== 'string' || !digest.startsWith('sha256:')) {
      throw new ContractError(`native GitHub asset ${pyRepr(name)} has no SHA-256 digest`);
    }
    requireSha256(digest.slice('sha256:'.length), `GitHub digest for ${name}`);
    const size = pyGet(item, 'size');
    if (!isPyIntLike(size) || compareToZero(size, '<') < 0) {
      throw new ContractError(`native GitHub asset ${pyRepr(name)} has an invalid size`);
    }
    assets.set(name, item);
  }
  return assets;
}

// Resolve the immutable commit a release tag names, peeling annotated tags.
//
// `git ls-remote` must be invoked with both `refs/tags/<tag>` and
// `refs/tags/<tag>^{}` patterns: the peeled entry is the only line that
// carries the commit for an annotated tag, and it is not tail-matched by the
// unpeeled pattern.
export function resolveTagCommit(lsRemoteOutput, tag) {
  parseReleaseTag(tag);
  const plainRef = `refs/tags/${tag}`;
  const peeledRef = `${plainRef}^{}`;
  const refs = new Map();
  for (const line of pySplitlines(lsRemoteOutput)) {
    if (!line) throw new ContractError('git ls-remote returned a blank line');
    const match = LS_REMOTE_LINE_RE.exec(line);
    if (match === null) throw new ContractError(`invalid git ls-remote line: ${pyRepr(line)}`);
    const [, objectId, ref] = match;
    if (ref !== plainRef && ref !== peeledRef) throw new ContractError(`git ls-remote returned unrelated ref ${pyRepr(ref)}`);
    if (refs.has(ref)) {
      const qualifier = refs.get(ref) !== objectId ? 'conflicting' : 'duplicate';
      throw new ContractError(`git ls-remote reported ${qualifier} objects for ${pyRepr(ref)}`);
    }
    refs.set(ref, objectId);
  }

  const plainObject = refs.get(plainRef);
  if (plainObject === undefined) throw new ContractError(`tag ${pyRepr(tag)} does not exist in the remote repository`);
  const peeledObject = refs.get(peeledRef);
  if (peeledObject === plainObject) throw new ContractError('peeled tag commit must differ from its annotated tag object');
  return peeledObject || plainObject;
}

// Validate native provenance against both downloaded bytes and GitHub metadata.
export function validateNativeRelease(
  manifestPath,
  checksumsPath,
  release,
  expectedSha256,
  nativeReleaseTag,
  upstreamTag,
  upstreamCommit,
  nativeTagCommit,
) {
  const identity = validateNativeFile(manifestPath, expectedSha256, nativeReleaseTag, upstreamTag, upstreamCommit);
  if (!pyEquals(pyGet(release, 'tag_name'), nativeReleaseTag) || pyGet(release, 'draft') !== false) {
    throw new ContractError('native GitHub release identity/draft state is not canonical');
  }
  validateGithubPrerelease(nativeReleaseTag, pyGet(release, 'prerelease'));
  // target_commitish is mutable: GitHub reports the branch a release was cut
  // from, not the tag's commit. Trust only the independently resolved
  // immutable tag commit, and still reject a target_commitish that pins a
  // different commit outright.
  if (requireCommit(nativeTagCommit, 'native_tag_commit') !== identity.nativeCommit) {
    throw new ContractError('resolved native release tag commit does not match manifest native_commit');
  }
  const targetCommitish = pyGet(release, 'target_commitish');
  if (typeof targetCommitish !== 'string' || !targetCommitish || targetCommitish !== pyStrip(targetCommitish)) {
    throw new ContractError('native GitHub release has invalid target_commitish');
  }
  if (COMMIT_RE.exec(targetCommitish) !== null) {
    if (targetCommitish !== identity.nativeCommit) throw new ContractError('native GitHub release target does not match native_commit');
  } else if (HEX_COMMITISH_RE.exec(targetCommitish) !== null) {
    throw new ContractError('native GitHub release target has an ambiguous commit form');
  } else if (targetCommitish.startsWith('refs/tags/')) {
    throw new ContractError('native GitHub release target must not name another tag ref');
  } else {
    let tagShaped = true;
    try {
      parseReleaseTag(targetCommitish);
    } catch (error) {
      if (!(error instanceof ContractError)) throw error;
      tagShaped = false;
    }
    if (tagShaped) throw new ContractError('native GitHub release target must not be tag-shaped');
  }

  const payload = strictJsonLoads(pyReadText(manifestPath), 'native release manifest');
  const hookContractVersion = pyGet(payload, 'hook_contract_version');
  if (!pyEquals(hookContractVersion, NATIVE_HOOK_CONTRACT_VERSION)) {
    throw new ContractError(`unsupported native hook_contract_version: ${pyRepr(hookContractVersion)}`);
  }
  const artifacts = pyGet(payload, 'artifacts');
  if (!Array.isArray(artifacts)) throw new ContractError('native manifest artifacts must be a list');
  const expected = expectedNativeArtifacts(nativeReleaseTag);
  const expectedNames = pyKeys(expected);
  const manifestAssets = new Map();
  for (const artifact of artifacts) {
    if (!isDict(artifact) || typeof pyGet(artifact, 'file') !== 'string') {
      throw new ContractError('native manifest contains an invalid artifact record');
    }
    const name = artifact.file;
    if (manifestAssets.has(name)) throw new ContractError(`native manifest has duplicate artifact ${pyRepr(name)}`);
    manifestAssets.set(name, artifact);
  }
  if (!pyKeySetEquals(manifestAssets, expectedNames)) {
    throw new ContractError('native manifest artifact inventory is incomplete or unexpected');
  }

  const githubAssets = releaseAssets(release);
  if (!pyKeySetEquals(githubAssets, [...expectedNames, 'assets.json', 'SHA256SUMS'])) {
    throw new ContractError('native GitHub release asset inventory is incomplete or unexpected');
  }

  const manifestBytes = pyReadBytes(manifestPath);
  const manifestDigest = sha256Hex(manifestBytes);
  const manifestMeta = githubAssets.get('assets.json');
  if (
    manifestMeta.digest !== `sha256:${manifestDigest}`
    || !pyEquals(manifestMeta.size, manifestBytes.length)
    || manifestDigest !== expectedSha256
  ) {
    throw new ContractError('native assets.json bytes do not match GitHub digest/size');
  }

  const checksumBytes = pyReadBytes(checksumsPath);
  const checksumDigest = sha256Hex(checksumBytes);
  const checksumMeta = githubAssets.get('SHA256SUMS');
  if (checksumMeta.digest !== `sha256:${checksumDigest}` || !pyEquals(checksumMeta.size, checksumBytes.length)) {
    throw new ContractError('native SHA256SUMS bytes do not match GitHub digest/size');
  }
  const checksumLines = new Map();
  for (const line of pySplitlines(pyDecodeUtf8(checksumBytes))) {
    const match = SHA256SUMS_LINE_RE.exec(line);
    if (match === null || checksumLines.has(match[2])) throw new ContractError(`invalid native SHA256SUMS line: ${pyRepr(line)}`);
    checksumLines.set(match[2], match[1]);
  }
  if (!pyKeySetEquals(checksumLines, expectedNames)) {
    throw new ContractError('native SHA256SUMS inventory does not match manifest artifacts');
  }

  const expectedKeys = ['module', 'platform', 'arch', 'backend', 'file', 'sha256', 'size'];
  for (const [name, [platform, arch, backend, module]] of pyItems(expected)) {
    const artifact = manifestAssets.get(name);
    if (!pyKeySetEquals(artifact, expectedKeys)) throw new ContractError(`native manifest schema mismatch for ${name}`);
    const digest = pyGet(artifact, 'sha256');
    const size = pyGet(artifact, 'size');
    const github = githubAssets.get(name);
    if (
      pyGet(artifact, 'platform') !== platform
      || pyGet(artifact, 'arch') !== arch
      || pyGet(artifact, 'backend') !== backend
      || pyGet(artifact, 'module') !== module
    ) {
      throw new ContractError(`native artifact metadata mismatch for ${name}`);
    }
    if (
      typeof digest !== 'string'
      || checksumLines.get(name) !== digest
      || github.digest !== `sha256:${digest}`
      || !pyEquals(github.size, size)
    ) {
      throw new ContractError(`native artifact checksum/size mismatch for ${name}`);
    }
  }
  return identity;
}

// Select the highest supported stable-channel tag without GitHub latest
// semantics. On a tie the first listed release wins, as max() keeps it.
export function selectStableNativeRelease(releases) {
  let selected = null;
  let selectedKey = null;
  for (const release of releases) {
    if (!isDict(release) || pyGet(release, 'draft') !== false) continue;
    const tag = pyGet(release, 'tag_name');
    if (typeof tag !== 'string') continue;
    let version;
    try {
      version = parseReleaseTag(tag, { allowLegacy: true });
    } catch (error) {
      if (error instanceof ContractError) continue;
      throw error;
    }
    if (version.channel === Channel.STABLE && pyGet(release, 'prerelease') === version.githubPrerelease) {
      const key = [...version.versionParts, version.rebuild];
      if (selected === null || compareIntTuples(key, selectedKey) > 0) {
        selected = version;
        selectedKey = key;
      }
    }
  }
  if (selected === null) throw new ContractError('no supported non-draft stable native release exists');
  return selected.tag;
}

// Require the exact fail-closed solo-maintainer publication policy.
export function validatePublicationEnvironment(environment, branchPolicies) {
  if (pyGet(environment, 'name') !== 'bridge-assets-publication') {
    throw new ContractError('publication environment identity is missing or incorrect');
  }
  if (pyGet(environment, 'can_admins_bypass') !== false) {
    throw new ContractError('publication environment must disable administrator bypass');
  }
  const rules = pyGet(environment, 'protection_rules');
  if (!Array.isArray(rules)) throw new ContractError('publication environment has no protection rules');
  let branchRuleCount = 0;
  for (const rule of rules) {
    if (!isDict(rule)) throw new ContractError('publication environment protection rules are invalid');
    const ruleType = pyGet(rule, 'type');
    if (ruleType === 'required_reviewers') {
      throw new ContractError('solo-maintainer publication environment must not require reviewers');
    }
    if (ruleType === 'branch_policy') branchRuleCount += 1;
  }
  if (branchRuleCount !== 1) throw new ContractError('publication environment must have one branch policy rule');

  const deploymentPolicy = pyGet(environment, 'deployment_branch_policy');
  if (
    !isDict(deploymentPolicy)
    || !pyKeySetEquals(deploymentPolicy, ['protected_branches', 'custom_branch_policies'])
    || pyGet(deploymentPolicy, 'protected_branches') !== false
    || pyGet(deploymentPolicy, 'custom_branch_policies') !== true
  ) {
    throw new ContractError('publication environment must use only custom deployment branch policies');
  }

  const policies = pyGet(branchPolicies, 'branch_policies');
  const policyCount = pyGet(branchPolicies, 'total_count');
  if (
    !isPyInt(policyCount)
    || !pyEquals(policyCount, 1)
    || !Array.isArray(policies)
    || policies.length !== 1
    || !isDict(policies[0])
    || pyGet(policies[0], 'name') !== 'main'
    || pyGet(policies[0], 'type') !== 'branch'
  ) {
    throw new ContractError('publication environment must allow deployments only from the main branch');
  }
}

function requireJsonBoolean(value, field) {
  if (typeof value !== 'boolean') throw new ContractError(`${field} must be an explicit JSON boolean`);
  return value;
}

function requireMapping(value, field) {
  if (!isDict(value)) throw new ContractError(`${field} must be a JSON object`);
  return value;
}

function requireUtcTimestamp(value, field) {
  if (typeof value !== 'string' || !value) throw new ContractError(`${field} must be a non-empty UTC timestamp`);
  const parsed = pyStrptimeUtc(value);
  if (parsed === null) throw new ContractError(`${field} must use the exact YYYY-MM-DDTHH:MM:SSZ format`);
  if (pyStrftimeUtc(parsed) !== value) throw new ContractError(`${field} is not a canonical UTC timestamp`);
  return value;
}

// Require enabled immutable-release governance on the assets repository.
//
// `GET /repos/{owner}/{repo}/immutable-releases` answers 404 both when
// governance is off and when the credential lacks administration read, so the
// caller must fail on any non-200; only an exact 200 body reaches this
// validator, and only `enabled: true` passes it.
export function validateImmutableReleaseGovernance(payload, repository) {
  requireRepository(repository, 'repository');
  const governance = requireMapping(payload, 'immutable-release governance response');
  if (!pyKeySetEquals(governance, ['enabled', 'enforced_by_owner'])) {
    throw new ContractError('immutable-release governance response has missing or unexpected fields');
  }
  const enforcedByOwner = requireJsonBoolean(pyGet(governance, 'enforced_by_owner'), 'enforced_by_owner');
  if (requireJsonBoolean(pyGet(governance, 'enabled'), 'enabled') !== true) {
    throw new ContractError(`immutable releases are not enabled for ${repository}`);
  }
  return { repository, enabled: true, enforced_by_owner: enforcedByOwner };
}

// Bind the candidate's pre-dispatch governance assertion to its identity.
export function validateCandidatePrequalification(payload, {
  candidateFingerprint,
  harnessSourceSha256,
  orchestratorCorrelationId,
  githubRunId,
  githubRunUrl,
  bridgeSourceSha,
  releaseTag,
  emscriptenVersion,
  nativeCommit,
} = {}) {
  const record = requireMapping(payload, 'candidate prequalification record');
  const expected = pyDict([
    ['schema_version', 1],
    ['candidate_fingerprint', requireSha256(candidateFingerprint, 'candidate_fingerprint')],
    ['harness_source_sha256', requireSha256(harnessSourceSha256, 'harness_source_sha256')],
    ['assets_immutable_releases_enabled', true],
    ['orchestrator_correlation_id', requireCorrelationId(orchestratorCorrelationId)],
    ['github_run_id', githubRunId ?? null],
    ['github_run_url', githubRunUrl ?? null],
    ['bridge_source_sha', requireCommit(bridgeSourceSha, 'bridge_source_sha')],
    ['release_tag', parseReleaseTag(releaseTag).tag],
    ['emscripten_version', emscriptenVersion ?? null],
    ['native_commit', requireCommit(nativeCommit, 'native_commit')],
    ['hosted_gates', { state_persistence: 'success', multimodal: 'success' }],
    ['heavy_gates', { speech_to_text: AUTOMATED_QUALIFICATION_PENDING, text_to_speech: AUTOMATED_QUALIFICATION_PENDING }],
    ['unproven_capabilities', { ...UNPROVEN_CAPABILITIES }],
  ]);
  if (!pyKeySetEquals(record, pyKeys(expected))) {
    throw new ContractError('candidate prequalification record has missing or unexpected fields');
  }
  const schemaVersion = pyGet(record, 'schema_version');
  if (!isPyInt(schemaVersion) || !pyEquals(schemaVersion, 1)) {
    throw new ContractError('candidate prequalification schema_version must be integer 1');
  }
  if (typeof githubRunId !== 'string' || POSITIVE_DECIMAL_RE.exec(githubRunId) === null) {
    throw new ContractError('github_run_id must be a positive decimal string');
  }
  const expectedRunUrl = `https://github.com/${BRIDGE_REPOSITORY}/actions/runs/${githubRunId}`;
  if (!pyEquals(githubRunUrl, expectedRunUrl)) throw new ContractError(`github_run_url must be exactly ${expectedRunUrl}`);
  if (typeof emscriptenVersion !== 'string' || !emscriptenVersion) {
    throw new ContractError('emscripten_version must be a non-empty string');
  }
  for (const [field, value] of pyItems(expected)) {
    if (field === 'assets_immutable_releases_enabled') {
      if (requireJsonBoolean(pyGet(record, field), 'assets_immutable_releases_enabled') !== true) {
        throw new ContractError('candidate did not assert immutable releases before dispatch');
      }
    } else if (!pyEquals(pyGet(record, field), value)) {
      throw new ContractError(`candidate prequalification ${field} mismatch`);
    }
  }
  return {
    candidate_fingerprint: candidateFingerprint,
    github_run_id: githubRunId,
    assets_immutable_releases_enabled: true,
  };
}

// Require an exact, published, immutable GitHub Release readback.
export function validateReleaseImmutability(release, { releaseTag, tagCommit, releaseId = null } = {}) {
  const parsedRelease = parseReleaseTag(releaseTag);
  requireCommit(tagCommit, 'tag_commit');
  const readback = requireMapping(release, 'GitHub Release response');
  if (!pyEquals(pyGet(readback, 'tag_name'), releaseTag)) {
    throw new ContractError(`GitHub Release tag ${pyRepr(pyGet(readback, 'tag_name'))} does not match ${pyRepr(releaseTag)}`);
  }
  const actualId = pyGet(readback, 'id');
  if (!isPyInt(actualId) || compareToZero(actualId, '<=') <= 0) {
    throw new ContractError('GitHub Release id must be a positive integer');
  }
  if (releaseId !== null && releaseId !== undefined && !pyEquals(actualId, releaseId)) {
    throw new ContractError('GitHub Release id does not match the classified release id');
  }
  if (pyGet(readback, 'draft') !== false) throw new ContractError('GitHub Release must be published, not a draft');
  if (pyGet(readback, 'prerelease') !== parsedRelease.githubPrerelease) {
    throw new ContractError('GitHub Release prerelease state does not match its tag');
  }
  if (!pyEquals(pyGet(readback, 'target_commitish'), tagCommit)) {
    throw new ContractError('GitHub Release target_commitish does not bind the exact tag commit');
  }
  requireUtcTimestamp(pyGet(readback, 'published_at'), 'published_at');
  if (!pyHasKey(readback, 'immutable')) throw new ContractError('GitHub Release response does not report immutability');
  if (requireJsonBoolean(pyGet(readback, 'immutable'), 'immutable') !== true) {
    throw new ContractError(`GitHub Release ${pyRepr(releaseTag)} is not immutable`);
  }
  return actualId;
}

// base64.b64decode(value, validate=True), reporting a decode failure as
// `could not decode <what>: <error>`.
function decodeBase64(value, what) {
  try {
    return pyB64decodeValidate(value);
  } catch (error) {
    if (isPyException(error, 'ValueError')) throw new ContractError(`could not decode ${what}: ${error.message}`, { cause: error });
    throw error;
  }
}

// Validate the release attestation `gh release verify --format json` proved.
//
// The CLI performs the cryptographic verification; this validator refuses to
// accept its report unless the verified signer, statement, predicate, tag
// commit, and every per-asset digest bind the exact release being published,
// and unless the signed DSSE payload still says the same thing.
export function validateReleaseAttestation(payload, {
  assetsRepo,
  releaseTag,
  tagCommit,
  releaseId,
  expectedAssets,
} = {}) {
  requireRepository(assetsRepo, 'assets_repo');
  parseReleaseTag(releaseTag);
  requireCommit(tagCommit, 'tag_commit');
  if (!isPyInt(releaseId) || compareToZero(releaseId, '<=') <= 0) {
    throw new ContractError('release_id must be a positive integer');
  }
  if (!(isDict(expectedAssets) || expectedAssets instanceof Map) || pyKeys(expectedAssets).length === 0) {
    throw new ContractError('expected release asset inventory must be a non-empty object');
  }
  const canonicalExpectedAssets = new Map();
  for (const [name, digest] of pyItems(expectedAssets)) {
    if (typeof name !== 'string' || ATTESTATION_ASSET_NAME_RE.exec(name) === null) {
      throw new ContractError(`invalid expected release asset name: ${pyRepr(name)}`);
    }
    canonicalExpectedAssets.set(name, requireSha256(pyValue(digest), `expected release asset digest for ${name}`));
  }
  const report = requireMapping(payload, 'release attestation response');
  const attestation = requireMapping(pyGet(report, 'attestation'), 'release attestation');
  const result = requireMapping(pyGet(report, 'verificationResult'), 'release attestation verification result');

  const bundle = requireMapping(pyGet(attestation, 'bundle'), 'release attestation bundle');
  if (pyGet(bundle, 'mediaType') !== SIGSTORE_BUNDLE_MEDIA_TYPE) throw new ContractError('release attestation is not a Sigstore bundle');
  if (pyGet(result, 'mediaType') !== SIGSTORE_VERIFICATION_RESULT_MEDIA_TYPE) {
    throw new ContractError('release attestation verification result is unsupported');
  }
  const verificationMaterial = requireMapping(pyGet(bundle, 'verificationMaterial'), 'release attestation verification material');
  const bundleCertificate = requireMapping(pyGet(verificationMaterial, 'certificate'), 'release attestation bundle certificate');
  const encodedCertificate = pyGet(bundleCertificate, 'rawBytes');
  if (typeof encodedCertificate !== 'string' || !encodedCertificate) {
    throw new ContractError('release attestation bundle certificate is missing');
  }
  if (decodeBase64(encodedCertificate, 'release attestation bundle certificate').length === 0) {
    throw new ContractError('release attestation bundle certificate is empty');
  }
  const timestampMaterial = requireMapping(
    pyGet(verificationMaterial, 'timestampVerificationData'),
    'release attestation timestamp verification material',
  );
  const signedTimestamps = pyGet(timestampMaterial, 'rfc3161Timestamps');
  if (!Array.isArray(signedTimestamps) || signedTimestamps.length === 0) {
    throw new ContractError('release attestation has no signed timestamp material');
  }
  for (const entry of signedTimestamps) {
    const signedTimestamp = pyGet(requireMapping(entry, 'release attestation signed timestamp'), 'signedTimestamp');
    if (typeof signedTimestamp !== 'string' || !signedTimestamp) {
      throw new ContractError('release attestation signed timestamp is missing');
    }
    if (decodeBase64(signedTimestamp, 'release attestation signed timestamp').length === 0) {
      throw new ContractError('release attestation signed timestamp is empty');
    }
  }

  const signature = requireMapping(pyGet(result, 'signature'), 'release attestation signature');
  const certificate = requireMapping(pyGet(signature, 'certificate'), 'release attestation certificate');
  if (pyGet(certificate, 'subjectAlternativeName') !== IMMUTABLE_RELEASE_ATTESTATION_SIGNER) {
    throw new ContractError("release attestation was not signed by GitHub's release attester");
  }
  const verifiedIdentity = requireMapping(pyGet(result, 'verifiedIdentity'), 'release attestation verified identity');
  const signerIdentity = requireMapping(
    pyGet(verifiedIdentity, 'subjectAlternativeName'),
    'release attestation verified signer identity',
  );
  if (pyGet(signerIdentity, 'regexp') !== RELEASE_SIGNER_IDENTITY_REGEXP) {
    throw new ContractError("release attestation verification policy does not bind GitHub's release signer");
  }
  const timestamps = pyGet(result, 'verifiedTimestamps');
  if (!Array.isArray(timestamps) || timestamps.length === 0) {
    throw new ContractError('release attestation has no verified timestamp');
  }
  const verifiedTimestamps = [];
  for (const entry of timestamps) {
    const verifiedAt = requireMapping(entry, 'release attestation timestamp');
    const uri = pyGet(verifiedAt, 'uri');
    if (pyGet(verifiedAt, 'type') !== 'TimestampAuthority' || typeof uri !== 'string' || !uri) {
      throw new ContractError('release attestation timestamp is incomplete');
    }
    const timestamp = requireUtcTimestamp(pyGet(verifiedAt, 'timestamp'), 'release attestation verified timestamp');
    verifiedTimestamps.push({ type: 'TimestampAuthority', uri, timestamp });
  }

  const statement = requireMapping(pyGet(result, 'statement'), 'release attestation statement');
  const envelope = requireMapping(pyGet(bundle, 'dsseEnvelope'), 'release attestation DSSE envelope');
  if (pyGet(envelope, 'payloadType') !== DSSE_IN_TOTO_PAYLOAD_TYPE) {
    throw new ContractError('release attestation DSSE payload is not in-toto JSON');
  }
  const signatures = pyGet(envelope, 'signatures');
  if (!Array.isArray(signatures) || signatures.length === 0) {
    throw new ContractError('release attestation DSSE envelope has no signature');
  }
  for (const entry of signatures) {
    const encodedSignature = pyGet(requireMapping(entry, 'release attestation DSSE signature'), 'sig');
    if (typeof encodedSignature !== 'string' || !encodedSignature) {
      throw new ContractError('release attestation DSSE signature is missing');
    }
    if (decodeBase64(encodedSignature, 'release attestation DSSE signature').length === 0) {
      throw new ContractError('release attestation DSSE signature is empty');
    }
  }
  const encoded = pyGet(envelope, 'payload');
  if (typeof encoded !== 'string' || !encoded) throw new ContractError('release attestation DSSE payload is missing');
  let decoded;
  try {
    decoded = pyDecodeUtf8(pyB64decodeValidate(encoded));
  } catch (error) {
    if (isPyException(error, 'ValueError')) {
      throw new ContractError(`could not decode release attestation DSSE payload: ${error.message}`, { cause: error });
    }
    throw error;
  }
  if (!pyEquals(strictJsonLoads(decoded, 'release attestation DSSE payload'), statement)) {
    throw new ContractError('verified statement does not match the signed attestation payload');
  }

  if (pyGet(statement, '_type') !== IN_TOTO_STATEMENT_TYPE) throw new ContractError('release attestation is not an in-toto statement');
  if (pyGet(statement, 'predicateType') !== IMMUTABLE_RELEASE_ATTESTATION_PREDICATE_TYPE) {
    throw new ContractError("release attestation predicate type is not GitHub's release predicate");
  }
  const purl = `pkg:github/${assetsRepo}@${releaseTag}`;
  const predicate = requireMapping(pyGet(statement, 'predicate'), 'release attestation predicate');
  for (const field of ['databaseId', 'ownerId', 'packageId', 'repositoryId']) {
    const value = pyGet(predicate, field);
    if (typeof value !== 'string' || POSITIVE_DECIMAL_RE.exec(value) === null) {
      throw new ContractError(`release attestation predicate ${field} must be a positive decimal string`);
    }
  }
  if (predicate.databaseId !== String(releaseId)) {
    throw new ContractError('release attestation predicate does not bind the exact release id');
  }
  if (pyGet(predicate, 'repository') !== assetsRepo || pyGet(predicate, 'tag') !== releaseTag || pyGet(predicate, 'purl') !== purl) {
    throw new ContractError('release attestation predicate does not name this exact release');
  }

  const subjects = pyGet(statement, 'subject');
  if (!Array.isArray(subjects) || subjects.length === 0) throw new ContractError('release attestation has no subject inventory');
  let releaseSubjects = 0;
  const attestedAssets = new Map();
  for (const entry of subjects) {
    const subject = requireMapping(entry, 'release attestation subject');
    const digest = requireMapping(pyGet(subject, 'digest'), 'release attestation subject digest');
    if (pyHasKey(subject, 'uri')) {
      if (!pyKeySetEquals(subject, ['uri', 'digest'])) throw new ContractError('release attestation release subject is malformed');
      const uri = pyGet(subject, 'uri');
      releaseSubjects += 1;
      if (uri !== purl) throw new ContractError('release attestation release subject names a different release');
      if (!pyKeySetEquals(digest, ['sha1']) || pyGet(digest, 'sha1') !== tagCommit) {
        throw new ContractError('release attestation release subject does not bind the exact tag commit');
      }
      continue;
    }
    if (!pyKeySetEquals(subject, ['name', 'digest'])) throw new ContractError('release attestation asset subject is malformed');
    const name = pyGet(subject, 'name');
    if (typeof name !== 'string' || ATTESTATION_ASSET_NAME_RE.exec(name) === null) {
      throw new ContractError('release attestation asset subject has no name');
    }
    if (attestedAssets.has(name)) throw new ContractError(`release attestation has duplicate subject ${pyRepr(name)}`);
    const assetDigest = pyGet(digest, 'sha256');
    if (!pyKeySetEquals(digest, ['sha256']) || typeof assetDigest !== 'string') {
      throw new ContractError(`release attestation subject ${pyRepr(name)} has no SHA-256`);
    }
    attestedAssets.set(name, requireSha256(assetDigest, `release attestation digest for ${name}`));
  }
  if (releaseSubjects !== 1) throw new ContractError('release attestation must carry exactly one release subject');
  if (!pyEquals(attestedAssets, canonicalExpectedAssets)) {
    throw new ContractError('release attestation asset digests do not match the published candidate');
  }
  return {
    purl,
    release_id: releaseId,
    tag_commit: tagCommit,
    predicate_type: IMMUTABLE_RELEASE_ATTESTATION_PREDICATE_TYPE,
    signer: IMMUTABLE_RELEASE_ATTESTATION_SIGNER,
    verified_timestamps: verifiedTimestamps,
    assets: pyDict(attestedAssets),
  };
}

function pyValue(value) {
  return value === undefined ? null : value;
}

// [release_tag, upstream_tag, bridge_commit] of a previous release manifest,
// accepting the legacy bridge_assets_tag/llama_cpp_tag/source_commit aliases.
export function readPreviousManifest(path) {
  let payload;
  try {
    payload = pyJsonLoads(pyReadText(path));
  } catch (error) {
    if (isPyException(error, 'OSError', 'UnicodeDecodeError', 'JSONDecodeError')) {
      throw new ContractError(`could not read previous manifest: ${error.message}`, { cause: error });
    }
    throw error;
  }
  if (!isDict(payload)) throw new ContractError('previous manifest root must be a JSON object');
  let releaseTag = pyGet(payload, 'release_tag');
  const legacyReleaseTag = pyGet(payload, 'bridge_assets_tag');
  if (releaseTag === null) releaseTag = legacyReleaseTag;
  else if (legacyReleaseTag !== null && !pyEquals(legacyReleaseTag, releaseTag)) {
    throw new ContractError('previous manifest release tag aliases conflict');
  }
  let upstreamTag = pyGet(payload, 'upstream_tag');
  const legacyUpstreamTag = pyGet(payload, 'llama_cpp_tag');
  if (upstreamTag === null) upstreamTag = legacyUpstreamTag;
  else if (legacyUpstreamTag !== null && !pyEquals(legacyUpstreamTag, upstreamTag)) {
    throw new ContractError('previous manifest upstream tag aliases conflict');
  }
  let bridgeCommit = pyGet(payload, 'bridge_commit');
  const legacyBridgeCommit = pyGet(payload, 'source_commit');
  if (bridgeCommit === null) bridgeCommit = legacyBridgeCommit;
  else if (legacyBridgeCommit !== null && !pyEquals(legacyBridgeCommit, bridgeCommit)) {
    throw new ContractError('previous manifest bridge commit aliases conflict');
  }
  if (typeof releaseTag !== 'string' || typeof upstreamTag !== 'string') {
    throw new ContractError('previous manifest is missing release/upstream tag identity');
  }
  parseReleaseTag(releaseTag, { allowLegacy: true });
  parseUpstreamTag(upstreamTag);
  return [releaseTag, upstreamTag, requireCommit(bridgeCommit, 'bridge_commit')];
}

const required = (flag, type = 'str') => ({ flag, required: true, type });

export const COMMANDS = Object.freeze({
  'validate-release': {
    options: [
      required('--release-tag'),
      required('--release-rebuild', 'int'),
      required('--upstream-tag'),
      { flag: '--previous-manifest', required: false, type: 'path' },
    ],
  },
  'validate-native': {
    options: [
      required('--manifest', 'path'),
      required('--manifest-sha256'),
      required('--native-release-tag'),
      required('--upstream-tag'),
      required('--upstream-commit'),
    ],
  },
  'validate-native-release': {
    options: [
      required('--manifest', 'path'),
      required('--checksums', 'path'),
      required('--release-json', 'path'),
      required('--manifest-sha256'),
      required('--native-release-tag'),
      required('--upstream-tag'),
      required('--upstream-commit'),
      required('--native-tag-commit'),
    ],
  },
  'validate-native-request': {
    options: [
      required('--native-release-tag'),
      required('--upstream-tag'),
      required('--upstream-commit'),
      required('--manifest-sha256'),
    ],
  },
  'resolve-tag-commit': { options: [required('--ls-remote', 'path'), required('--tag')] },
  'compare-upstream': { positionals: ['current', 'target'] },
  'scan-native': { options: [required('--manifest', 'path'), required('--native-release-tag')] },
  'select-stable-native-release': { options: [required('--releases-json', 'path')] },
  'validate-immutable-release-governance': {
    options: [required('--governance-json', 'path'), required('--repository')],
  },
  'validate-candidate-prequalification': {
    options: [
      required('--prequalification-json', 'path'),
      required('--candidate-fingerprint'),
      required('--harness-source-sha256'),
      required('--orchestrator-correlation-id'),
      required('--github-run-id'),
      required('--github-run-url'),
      required('--bridge-source-sha'),
      required('--release-tag'),
      required('--emscripten-version'),
      required('--native-commit'),
    ],
  },
  'validate-environment': {
    options: [required('--environment-json', 'path'), required('--branch-policies-json', 'path')],
  },
  // The workflows' input checks: silent on success, `error: <message>` and
  // exit status 1 otherwise.
  'require-correlation-id': { options: [required('--orchestrator-correlation-id')] },
  'require-repository': { options: [required('--repository'), required('--field')] },
});

function dumps(value) {
  return `${pyJsonDumps(value, { sortKeys: true })}\n`;
}

function runCommand(command, args, write) {
  if (command === 'validate-release') {
    const release = validateNewReleaseIdentity(args.releaseTag, args.releaseRebuild, args.upstreamTag);
    const result = {
      release_tag: release.tag,
      release_channel: release.channel,
      release_rebuild: release.rebuild,
      upstream_tag: args.upstreamTag,
    };
    if (args.previousManifest !== null) {
      const [previousTag, previousUpstream, previousBridge] = readPreviousManifest(args.previousManifest);
      const releaseTransition = compareReleases(previousTag, release.tag);
      const upstreamTransition = compareUpstream(previousUpstream, args.upstreamTag);
      if (releaseTransition !== Transition.FORWARD && releaseTransition !== Transition.STABLE_MIGRATION) {
        throw new ContractError(`release must advance from ${pyRepr(previousTag)}: ${releaseTransition}`);
      }
      if (upstreamTransition === Transition.BACKWARD || upstreamTransition === Transition.FORBIDDEN_STABLE_TO_DEVELOPMENT) {
        throw new ContractError(`upstream must not roll back from ${pyRepr(previousUpstream)}: ${upstreamTransition}`);
      }
      Object.assign(result, {
        previous_release_tag: previousTag,
        previous_upstream_tag: previousUpstream,
        previous_bridge_commit: previousBridge,
        release_transition: releaseTransition,
        upstream_transition: upstreamTransition,
      });
    }
    write(dumps(result));
  } else if (command === 'validate-native') {
    const identity = validateNativeFile(args.manifest, args.manifestSha256, args.nativeReleaseTag, args.upstreamTag, args.upstreamCommit);
    write(dumps(identity.toDict()));
  } else if (command === 'validate-native-release') {
    const releasePayload = pyJsonLoads(pyReadText(args.releaseJson));
    if (!isDict(releasePayload)) throw new ContractError('native GitHub release root must be an object');
    const identity = validateNativeRelease(
      args.manifest,
      args.checksums,
      releasePayload,
      args.manifestSha256,
      args.nativeReleaseTag,
      args.upstreamTag,
      args.upstreamCommit,
      args.nativeTagCommit,
    );
    write(dumps(identity.toDict()));
  } else if (command === 'validate-native-request') {
    const nativeRelease = validateNativeRequest(args.nativeReleaseTag, args.upstreamTag, args.upstreamCommit, args.manifestSha256);
    write(dumps({ native_release_tag: nativeRelease.tag, upstream_tag: args.upstreamTag }));
  } else if (command === 'resolve-tag-commit') {
    write(`${resolveTagCommit(pyReadText(args.lsRemote), args.tag)}\n`);
  } else if (command === 'compare-upstream') {
    write(`${compareUpstream(args.current, args.target)}\n`);
  } else if (command === 'scan-native') {
    const payload = pyJsonLoads(pyReadText(args.manifest));
    write(dumps(resolveNativeManifest(payload, args.nativeReleaseTag).toDict()));
  } else if (command === 'select-stable-native-release') {
    const releases = pyJsonLoads(pyReadText(args.releasesJson));
    if (!Array.isArray(releases)) throw new ContractError('native release listing must be a JSON array');
    write(`${selectStableNativeRelease(releases)}\n`);
  } else if (command === 'validate-immutable-release-governance') {
    const governancePayload = strictJsonLoads(pyReadText(args.governanceJson), 'immutable-release governance response');
    write(dumps(validateImmutableReleaseGovernance(governancePayload, args.repository)));
  } else if (command === 'validate-candidate-prequalification') {
    const prequalificationPayload = strictJsonLoads(pyReadText(args.prequalificationJson), 'candidate prequalification record');
    write(dumps(validateCandidatePrequalification(prequalificationPayload, {
      candidateFingerprint: args.candidateFingerprint,
      harnessSourceSha256: args.harnessSourceSha256,
      orchestratorCorrelationId: args.orchestratorCorrelationId,
      githubRunId: args.githubRunId,
      githubRunUrl: args.githubRunUrl,
      bridgeSourceSha: args.bridgeSourceSha,
      releaseTag: args.releaseTag,
      emscriptenVersion: args.emscriptenVersion,
      nativeCommit: args.nativeCommit,
    })));
  } else if (command === 'require-correlation-id') {
    requireCorrelationId(args.orchestratorCorrelationId);
  } else if (command === 'require-repository') {
    requireRepository(args.repository, args.field);
  } else {
    const environmentPayload = pyJsonLoads(pyReadText(args.environmentJson));
    const branchPoliciesPayload = pyJsonLoads(pyReadText(args.branchPoliciesJson));
    if (!isDict(environmentPayload)) throw new ContractError('publication environment root must be an object');
    if (!isDict(branchPoliciesPayload)) throw new ContractError('publication branch policies root must be an object');
    validatePublicationEnvironment(environmentPayload, branchPoliciesPayload);
    write(dumps({ environment: 'bridge-assets-publication' }));
  }
}

// The CLI: `main(argv, write)` parses argv like release_contract.py's argparse
// parser and prints through write(); a contract, file, decode or JSON error
// becomes `error: <message>` with exit status 1.
export function main(argv, write) {
  const { command, args } = parseCommandLine(argv, { prog: progName(import.meta.url), commands: COMMANDS });
  try {
    runCommand(command, args, write);
  } catch (error) {
    if (error instanceof ContractError || isPyException(error, 'OSError', 'UnicodeDecodeError', 'JSONDecodeError')) {
      throw new SystemExitError(`error: ${error.message}`);
    }
    throw error;
  }
  return 0;
}

export { ArgparseExit };

if (import.meta.main) runCli(main);
