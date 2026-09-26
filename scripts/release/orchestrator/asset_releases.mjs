// Assets-repository release discovery and proof for the orchestrator, the Node
// port of scripts/release_orchestrator_asset_releases.py.
//
// Resolves tag commits, lists releases and tag refs, finds the release a
// correlation or native alignment points to, compares publication bytes,
// bounds run recovery, and re-proves a published release through independent
// readbacks, downloaded bytes, and its signed release attestation.
//
// A release listing is JSON: plain objects (or Maps) read with pyGet. A tuple
// result is an array ([nativeTag, assetTag], [release, correlationId]) and
// None is null.

import { createHash } from 'node:crypto';
import fs from 'node:fs';

import {
  ASSETS_REPOSITORY, Channel, ContractError, NATIVE_REPOSITORY, parseReleaseTag, requireCorrelationId, requireRepository,
  requireSha256, validateReleaseAttestation, validateReleaseImmutability,
} from '../contract.mjs';
import {
  compareCodePoints, isDict, isPyInt, pyEquals, pyGet, pyJoinPath, pyKeys, pyOSError, pyPath, pyQuote, pyRepr, pyStr,
} from '../json.mjs';
import { ARTIFACTS } from '../manifest.mjs';
import { PUBLICATION_FILES } from '../publication_state.mjs';
import { pyCompareIntTuples, pyFullmatch } from '../python_compat.mjs';
import { loadCandidate, loadPublishedCandidate } from '../qualification.mjs';
import {
  COMMIT_RE, LEGACY_MANUAL_QUALIFICATION_BRIDGE_COMMIT, LEGACY_MANUAL_QUALIFICATION_RELEASE_TAG, PublishedRelease,
  UTC_TIMESTAMP_RE, publishedManifestCompatibility, requirePositiveInt, requireStr,
} from './model.mjs';
import { nativeReleaseOrder } from './native.mjs';
import { validateCandidateManifest } from './stage_proofs.mjs';

// isinstance(value, Mapping) for a JSON value.
function isMapping(value) {
  return isDict(value) || value instanceof Map;
}

// Parse a release tag, or null where Python catches the ContractError.
function tryParseReleaseTag(tag, options) {
  try {
    return parseReleaseTag(tag, options);
  } catch (error) {
    if (error instanceof ContractError) return null;
    throw error;
  }
}

// re.findall(r"(?m)^<prefix>([^`\r\n]+)`<suffix>$", text) for a pattern that
// cannot cross a line: Python's multiline ^ and $ match only at "\n" (JS's
// also match at "\r", U+2028 and U+2029), so each "\n"-separated line is
// matched on its own.
function findLineMatches(lineRe, text) {
  const found = [];
  for (const line of text.split('\n')) {
    const match = lineRe.exec(line);
    if (match !== null) found.push(match[1]);
  }
  return found;
}

// Anchored per line; see findLineMatches.
const NATIVE_ALIGNMENT_LINE_RE = new RegExp(`^Native: \`${NATIVE_REPOSITORY.replace(/[\\^$.*+?()[\]{}|/]/gu, '\\$&')}@([^\`\\r\\n]+)\`\\r?$`, 'u');
const CORRELATION_LINE_RE = /^Orchestrator correlation: `([^`\r\n]+)`$/u;

// _pages: normalize `gh api --paginate --slurp` output to a page list.
export function pages(payload, label) {
  if (Array.isArray(payload)) return payload;
  if (isMapping(payload)) return [payload];
  throw new ContractError(`${label} response must be a JSON array or object`);
}

// Resolve a lightweight or annotated tag to its immutable commit.
export function resolveRepositoryTagCommit(gateway, { repository, releaseTag }) {
  requireRepository(repository, 'repository');
  parseReleaseTag(releaseTag);
  const encodedTag = pyQuote(releaseTag, '');
  const payload = gateway.apiJson(`repos/${repository}/git/ref/tags/${encodedTag}`);
  if (!isMapping(payload) || !pyEquals(pyGet(payload, 'ref'), `refs/tags/${releaseTag}`)) {
    throw new ContractError('assets tag reference identity is missing or incorrect');
  }
  let objectPayload = pyGet(payload, 'object');
  const seen = new Set();
  for (let step = 0; step < 8; step += 1) {
    if (!isMapping(objectPayload)) throw new ContractError('assets tag reference object is malformed');
    const objectType = pyGet(objectPayload, 'type');
    const sha = pyGet(objectPayload, 'sha');
    if (typeof sha !== 'string' || COMMIT_RE.exec(sha) === null) {
      throw new ContractError('assets tag reference object has no full commit SHA');
    }
    if (seen.has(sha)) throw new ContractError('assets annotated tag chain contains a cycle');
    seen.add(sha);
    if (pyEquals(objectType, 'commit')) return sha;
    if (!pyEquals(objectType, 'tag')) {
      throw new ContractError(`assets tag reference points to unsupported object type ${pyRepr(objectType)}`);
    }
    const annotated = gateway.apiJson(`repos/${repository}/git/tags/${sha}`);
    if (!isMapping(annotated) || !pyEquals(pyGet(annotated, 'sha'), sha)) {
      throw new ContractError('assets annotated tag identity is malformed');
    }
    objectPayload = pyGet(annotated, 'object');
  }
  throw new ContractError('assets annotated tag chain exceeds the validation bound');
}

export function fetchAssetReleases(gateway) {
  const payload = gateway.apiJson(`repos/${ASSETS_REPOSITORY}/releases?per_page=100`, { paginate: true });
  const releases = [];
  const seenTags = new Set();
  for (const page of pages(payload, 'asset releases')) {
    for (const release of Array.isArray(page) ? page : [page]) {
      if (!isMapping(release)) throw new ContractError('asset release record must be a JSON object');
      const tag = requireStr(pyGet(release, 'tag_name'), 'asset release tag_name');
      if (seenTags.has(tag)) throw new ContractError(`asset repository lists duplicate release ${pyRepr(tag)}`);
      seenTags.add(tag);
      releases.push(release);
    }
  }
  return releases;
}

// Every existing assets tag ref, as a Set, so output selection cannot collide.
export function fetchAssetTagNames(gateway) {
  const payload = gateway.apiJson(`repos/${ASSETS_REPOSITORY}/git/matching-refs/tags?per_page=100`, { paginate: true });
  const tags = new Set();
  for (const page of pages(payload, 'asset tag refs')) {
    for (const record of Array.isArray(page) ? page : [page]) {
      if (!isMapping(record)) throw new ContractError('asset tag ref record must be a JSON object');
      const ref = pyGet(record, 'ref');
      if (typeof ref !== 'string' || !ref.startsWith('refs/tags/')) throw new ContractError('asset tag ref has an invalid ref name');
      const tag = ref.slice('refs/tags/'.length);
      if (!tag || tags.has(tag)) throw new ContractError(`asset repository lists duplicate tag ref ${pyRepr(tag)}`);
      const objectPayload = pyGet(record, 'object');
      if (!isMapping(objectPayload)) throw new ContractError(`asset tag ref ${pyRepr(tag)} has no object`);
      const objectType = pyGet(objectPayload, 'type');
      if (!pyEquals(objectType, 'commit') && !pyEquals(objectType, 'tag')) {
        throw new ContractError(`asset tag ref ${pyRepr(tag)} has invalid object type`);
      }
      const sha = pyGet(objectPayload, 'sha');
      if (typeof sha !== 'string' || COMMIT_RE.exec(sha) === null) {
        throw new ContractError(`asset tag ref ${pyRepr(tag)} has invalid object SHA`);
      }
      tags.add(tag);
    }
  }
  return tags;
}

// _correlation_marker.
export function correlationMarker(correlationId) {
  return `Orchestrator correlation: \`${correlationId}\``;
}

// (*version.version_parts, version.rebuild).
function releaseOrder(version) {
  return [...version.versionParts, version.rebuild];
}

// [nativeTag, assetTag] for the newest stable native release named by a
// non-draft stable asset release's `Native:` marker, else null.
//
// Raises ContractError if one release names several native releases or a
// malformed tag.
export function latestPublishedNativeAlignment(releases) {
  let latest = null;
  for (const release of releases) {
    const assetTag = pyGet(release, 'tag_name');
    const body = pyGet(release, 'body');
    if (typeof assetTag !== 'string' || typeof body !== 'string') continue;
    const assetVersion = tryParseReleaseTag(assetTag, { allowLegacy: true });
    if (assetVersion === null) continue;
    if (assetVersion.channel !== Channel.STABLE || pyGet(release, 'draft') !== false) continue;
    const claims = new Set(findLineMatches(NATIVE_ALIGNMENT_LINE_RE, body));
    if (claims.size === 0) continue;
    if (claims.size !== 1) throw new ContractError(`asset release ${pyRepr(assetTag)} records ${claims.size} native alignments`);
    const [nativeTag] = claims;
    let nativeVersion;
    try {
      nativeVersion = parseReleaseTag(nativeTag);
    } catch (error) {
      if (error instanceof ContractError) {
        throw new ContractError(`asset release ${pyRepr(assetTag)} records a malformed native alignment: ${error.message}`, { cause: error });
      }
      throw error;
    }
    if (nativeVersion.channel !== Channel.STABLE) continue;
    const order = nativeReleaseOrder(nativeTag);
    if (latest === null || pyCompareIntTuples(order, latest[0]) > 0) latest = [order, nativeTag, assetTag];
  }
  return latest === null ? null : latest.slice(1);
}

// The one valid correlation claims of a release body, as require_correlation_id
// accepts them.
function validCorrelationClaims(body) {
  const claims = [];
  for (const claim of findLineMatches(CORRELATION_LINE_RE, body)) {
    try {
      claims.push(requireCorrelationId(claim));
    } catch (error) {
      if (!(error instanceof ContractError)) throw error;
    }
  }
  return claims;
}

export function findCorrelatedRelease(releases, correlationId, provenance) {
  const marker = correlationMarker(correlationId);
  // The manifest digest is independently selected from the native release and
  // is also written to deterministic release notes. If only the correlation
  // line is damaged, still classify the release as relevant and let the full
  // immutable readback reject it instead of dispatching duplicate provenance.
  const nativeManifestMarker = `Native manifest SHA-256: \`${provenance.nativeManifestSha256}\``;
  const nativeReleaseMarker = `Native: \`${provenance.nativeRepo}@${provenance.nativeReleaseTag}\``;
  const matches = [];
  for (const release of releases) {
    const body = pyGet(release, 'body');
    if (typeof body !== 'string') continue;
    if (body.includes(marker)) {
      matches.push(release);
      continue;
    }
    if (!body.includes(nativeManifestMarker) || !body.includes(nativeReleaseMarker)) continue;
    // The same native release may intentionally have multiple bridge
    // publications after governed runtime/build changes. A different
    // well-formed correlation is another pipeline, not damaged state.
    if (validCorrelationClaims(body).length > 0) continue;
    // With no valid correlation marker, the native identity is still close
    // enough to block a duplicate until immutable readback diagnoses it.
    matches.push(release);
  }
  if (matches.length > 1) {
    throw new ContractError(
      `${matches.length} asset releases claim correlation ${pyRepr(correlationId)}: `
      + matches.map((release) => pyStr(pyGet(release, 'tag_name'))).join(', '),
    );
  }
  return matches.length > 0 ? matches[0] : null;
}

// [release, correlationId] for the newest non-draft stable asset release whose
// notes record this provenance's exact native release and native manifest
// digest, with the one correlation it claims, else null.
//
// A release with the same native markers but not exactly one well-formed
// correlation marker is skipped. The pre-automation v0.1.40 manifest follows a
// different contract and is only comparable by its own provenance.
export function latestAlignedRelease(releases, provenance) {
  const nativeReleaseMarker = `Native: \`${provenance.nativeRepo}@${provenance.nativeReleaseTag}\``;
  const nativeManifestMarker = `Native manifest SHA-256: \`${provenance.nativeManifestSha256}\``;
  let latest = null;
  for (const release of releases) {
    const tag = pyGet(release, 'tag_name');
    const body = pyGet(release, 'body');
    if (typeof tag !== 'string' || typeof body !== 'string') continue;
    const version = tryParseReleaseTag(tag, { allowLegacy: true });
    if (version === null) continue;
    if (version.channel !== Channel.STABLE || pyGet(release, 'draft') !== false) continue;
    if (!body.includes(nativeReleaseMarker) || !body.includes(nativeManifestMarker)) continue;
    if (tag === LEGACY_MANUAL_QUALIFICATION_RELEASE_TAG && publishedManifestCompatibility({ tag, provenance }) === null) continue;
    const claims = validCorrelationClaims(body);
    if (claims.length !== 1) continue;
    const order = releaseOrder(version);
    if (latest === null || pyCompareIntTuples(order, latest[0]) > 0) latest = [order, release, claims[0]];
  }
  return latest === null ? null : latest.slice(1);
}

// True when every publication file except manifest.json has the same SHA-256
// in both complete inventories (dicts or Maps of name to digest).
//
// manifest.json embeds the candidate run ID/URL, output tag, correlation and
// bridge commit, so it differs between any two builds; sha256sums.txt lists
// only the artifact digests and is compared.
export function publicationBytesIdentical(candidateDigests, publishedDigests) {
  const complete = (digests) => {
    const names = new Set(pyKeys(digests));
    return names.size === PUBLICATION_FILES.length && PUBLICATION_FILES.every((name) => names.has(name));
  };
  if (!complete(candidateDigests) || !complete(publishedDigests)) {
    throw new ContractError('publication digest inventories must be complete');
  }
  return PUBLICATION_FILES
    .filter((name) => name !== 'manifest.json')
    .every((name) => pyEquals(pyGet(candidateDigests, name), pyGet(publishedDigests, name)));
}

// Bound run recovery to state that can still claim the next output tag.
//
// Candidate claims made before the most recently published stable assets tag
// cannot collide with a later monotonic output version. Taking the earlier of
// that publication and the native release still includes a prior unfinished
// pipeline when a newer native release appears, without coupling recovery to
// the repository's unbounded lifetime run count.
export function workflowHistorySince(releases, provenance) {
  const stablePublications = [];
  for (const release of releases) {
    const tag = pyGet(release, 'tag_name');
    if (typeof tag !== 'string') continue;
    const parsed = tryParseReleaseTag(tag, { allowLegacy: true });
    if (parsed === null) continue;
    if (parsed.channel !== Channel.STABLE || pyGet(release, 'draft') === true) continue;
    if (pyGet(release, 'draft') !== false) throw new ContractError(`stable asset release ${pyRepr(tag)} has invalid draft state`);
    if (pyGet(release, 'prerelease') !== parsed.githubPrerelease) {
      throw new ContractError(`stable asset release ${pyRepr(tag)} has invalid prerelease state`);
    }
    const publishedAt = pyGet(release, 'published_at');
    if (typeof publishedAt !== 'string' || pyFullmatch(UTC_TIMESTAMP_RE, publishedAt) === null) {
      throw new ContractError(`stable asset release ${pyRepr(tag)} has no canonical published_at`);
    }
    stablePublications.push(publishedAt);
  }
  if (stablePublications.length === 0) return provenance.nativeReleasePublishedAt;
  const newest = stablePublications.reduce((left, right) => (compareCodePoints(right, left) > 0 ? right : left));
  return compareCodePoints(newest, provenance.nativeReleasePublishedAt) < 0 ? newest : provenance.nativeReleasePublishedAt;
}

// _release_assets: a Map of name to asset record, in listing order.
export function releaseAssets(release) {
  const raw = pyGet(release, 'assets');
  if (!Array.isArray(raw)) throw new ContractError('GitHub Release is missing its asset inventory');
  const assets = new Map();
  for (const item of raw) {
    if (!isMapping(item)) throw new ContractError('GitHub Release contains an invalid asset record');
    const name = requireStr(pyGet(item, 'name'), 'GitHub Release asset name');
    if (assets.has(name)) throw new ContractError(`GitHub Release has duplicate asset ${pyRepr(name)}`);
    if (!pyEquals(pyGet(item, 'state'), 'uploaded')) throw new ContractError(`GitHub Release asset ${pyRepr(name)} is not uploaded`);
    const digest = pyGet(item, 'digest');
    if (typeof digest !== 'string' || !digest.startsWith('sha256:')) {
      throw new ContractError(`GitHub Release asset ${pyRepr(name)} has no SHA-256 digest`);
    }
    requireSha256(digest.slice('sha256:'.length), `digest for ${name}`);
    const size = pyGet(item, 'size');
    if (!isPyInt(size) || BigInt(size) < 0n) throw new ContractError(`GitHub Release asset ${pyRepr(name)} has an invalid size`);
    requirePositiveInt(pyGet(item, 'id'), `GitHub Release asset ${pyRepr(name)} id`);
    assets.set(name, item);
  }
  const expected = new Set(PUBLICATION_FILES);
  const unexpected = [...assets.keys()].filter((name) => !expected.has(name)).sort(compareCodePoints);
  const missing = PUBLICATION_FILES.filter((name) => !assets.has(name)).sort(compareCodePoints);
  if (unexpected.length > 0 || missing.length > 0) {
    throw new ContractError(
      'GitHub Release asset inventory is not the exact publication set '
      + `(unexpected: ${pyRepr(unexpected)}, missing: ${pyRepr(missing)})`,
    );
  }
  return assets;
}

// _candidate_fingerprint_marker.
export function candidateFingerprintMarker(fingerprint) {
  return `Candidate fingerprint: \`${fingerprint}\``;
}

// shutil.rmtree(directory) if it exists, then Path.mkdir(parents=True).
function freshDirectory(directory) {
  try {
    if (fs.existsSync(directory)) fs.rmSync(directory, { recursive: true });
    fs.mkdirSync(directory, { recursive: true });
  } catch (error) {
    throw pyOSError(error, directory);
  }
}

// Prove an existing release really is this provenance, published immutably.
//
// The release listing is discovery only. The exact tag commit, independent
// release reads by tag and ID, downloaded artifact bytes, and GitHub's signed
// release attestation are all re-proven with the same validators used after
// publication before this path may return a noop. `workspace` is a path string.
export function verifyPublishedRelease(gateway, {
  release, provenance, correlationId, workspace,
}) {
  const tag = requireStr(pyGet(release, 'tag_name'), 'release tag_name');
  parseReleaseTag(tag);
  const releaseId = requirePositiveInt(pyGet(release, 'id'), 'release id');
  const tagCommit = resolveRepositoryTagCommit(gateway, { repository: ASSETS_REPOSITORY, releaseTag: tag });
  const encodedTag = pyQuote(tag, '');
  const releaseByTag = gateway.apiJson(`repos/${ASSETS_REPOSITORY}/releases/tags/${encodedTag}`);
  const releaseById = gateway.apiJson(`repos/${ASSETS_REPOSITORY}/releases/${releaseId}`);
  if (!isMapping(releaseByTag) || !isMapping(releaseById)) throw new ContractError('release readbacks must be JSON objects');
  const resolvedId = validateReleaseImmutability(releaseByTag, { releaseTag: tag, tagCommit, releaseId });
  if (!pyEquals(validateReleaseImmutability(releaseById, { releaseTag: tag, tagCommit, releaseId: resolvedId }), resolvedId)) {
    throw new ContractError('release readbacks by tag and ID disagree on release id');
  }
  for (const [label, current] of [['tag', releaseByTag], ['id', releaseById]]) {
    if (!pyEquals(pyGet(current, 'name'), tag)) throw new ContractError(`release readback by ${label} is not named ${pyRepr(tag)}`);
    const body = pyGet(current, 'body');
    if (typeof body !== 'string' || !body.includes(correlationMarker(correlationId))) {
      throw new ContractError(`release readback by ${label} does not record correlation ${pyRepr(correlationId)}`);
    }
  }
  if (!pyEquals(pyGet(releaseByTag, 'published_at'), pyGet(releaseById, 'published_at'))) {
    throw new ContractError('release readbacks disagree on published_at');
  }
  if (!pyEquals(pyGet(releaseByTag, 'body'), pyGet(releaseById, 'body'))) {
    throw new ContractError('release readbacks disagree on release body');
  }
  const publishedAt = requireStr(pyGet(releaseByTag, 'published_at'), 'release published_at');
  const body = requireStr(pyGet(releaseByTag, 'body'), 'release body');

  const assets = releaseAssets(releaseByTag);
  const assetsById = releaseAssets(releaseById);
  for (const name of PUBLICATION_FILES) {
    const fields = ['id', 'name', 'state', 'size', 'digest'];
    if (fields.some((field) => !pyEquals(pyGet(assets.get(name), field), pyGet(assetsById.get(name), field)))) {
      throw new ContractError(`release readbacks disagree on asset ${pyRepr(name)} identity`);
    }
  }
  const directory = pyJoinPath(pyPath(String(workspace)), `published-${releaseId}`);
  freshDirectory(directory);
  const expectedAssetDigests = new Map();
  for (const [name, asset] of assets) {
    const data = gateway.downloadBytes(`repos/${ASSETS_REPOSITORY}/releases/assets/${pyGet(asset, 'id')}`, {
      accept: 'application/octet-stream',
    });
    const actual = createHash('sha256').update(data).digest('hex');
    if (!pyEquals(`sha256:${actual}`, pyGet(asset, 'digest'))) {
      throw new ContractError(`release asset ${pyRepr(name)} does not match its GitHub digest`);
    }
    if (!pyEquals(data.length, pyGet(asset, 'size'))) throw new ContractError(`release asset ${pyRepr(name)} does not match its GitHub size`);
    expectedAssetDigests.set(name, actual);
    const target = pyJoinPath(directory, name);
    try {
      fs.writeFileSync(target, data);
    } catch (error) {
      throw pyOSError(error, target);
    }
  }

  const compatibility = publishedManifestCompatibility({ tag, provenance });
  const [manifest, fingerprint] = compatibility === null
    ? loadCandidate(directory)
    : loadPublishedCandidate(directory, {
      expectedQualificationGates: compatibility[0],
      expectedUnprovenCapabilities: compatibility[1],
    });
  if (!body.includes(candidateFingerprintMarker(fingerprint))) {
    throw new ContractError(`release ${pyRepr(tag)} does not record the fingerprint of the bytes it actually published`);
  }
  const binding = validateCandidateManifest(manifest, {
    provenance,
    correlationId,
    expectedReleaseTag: tag,
    expectedBridgeSourceSha: compatibility !== null ? LEGACY_MANUAL_QUALIFICATION_BRIDGE_COMMIT : null,
  });
  const recordedArtifacts = pyGet(manifest, 'artifacts');
  for (const name of ARTIFACTS) {
    const recorded = pyGet(recordedArtifacts, name);
    if (!pyEquals(pyGet(assets.get(name), 'digest'), `sha256:${pyStr(pyGet(recorded, 'sha256'))}`)) {
      throw new ContractError(`release asset ${pyRepr(name)} digest is not the manifest digest`);
    }
    if (!pyEquals(pyGet(assets.get(name), 'size'), pyGet(recorded, 'size_bytes'))) {
      throw new ContractError(`release asset ${pyRepr(name)} size is not the manifest size`);
    }
  }
  validateReleaseAttestation(gateway.releaseAttestation({ repository: ASSETS_REPOSITORY, releaseTag: tag }), {
    assetsRepo: ASSETS_REPOSITORY,
    releaseTag: tag,
    tagCommit,
    releaseId: resolvedId,
    expectedAssets: expectedAssetDigests,
  });
  return new PublishedRelease({
    releaseId: resolvedId,
    releaseTarget: binding.releaseTarget,
    binding,
    publishedAt,
    directory,
  });
}
