// Native release scanning for the stable release orchestrator, the Node port
// of scripts/release_orchestrator_native.py.
//
// Extracts exact provenance from a native assets.json on either channel,
// selects every published stable native release after the automation baseline,
// and orders native release tags.

import { createHash } from 'node:crypto';

import {
  Channel, ContractError, NATIVE_REPOSITORY, parseReleaseTag, resolveNativeManifest, strictJsonLoads,
} from '../contract.mjs';
import {
  compareCodePoints, isDict, isPyException, pyDecodeUtf8, pyGet, pyIsFile, pyIsSymlink, pyPath,
  pyReadBytes, pyRepr,
} from '../json.mjs';
import { pyCompareIntTuples, pyFullmatch } from '../python_compat.mjs';
import { NativeProvenance, UTC_TIMESTAMP_RE } from './model.mjs';

// v0.2.0-1 was published as the first verified immutable automatic-publication
// baseline (Web assets v0.1.39). Older native releases belong to the historical
// pre-automation series and must not be silently rebuilt by backlog scans.
export const STABLE_AUTOMATION_BASELINE_NATIVE_TAG = 'v0.2.0-1';
export const STABLE_AUTOMATION_BASELINE_PUBLISHED_AT = '2026-08-25T08:57:12Z';

// _CHANNELS: each channel's value, mapped to itself.
export const CHANNELS = Object.freeze(new Map(Object.values(Channel).map((channel) => [channel, channel])));

export function requireChannel(value) {
  const channel = typeof value === 'string' ? CHANNELS.get(value) : undefined;
  if (channel === undefined) {
    throw new ContractError(`unsupported release channel ${pyRepr(value)}; expected one of ${[...CHANNELS.keys()].sort(compareCodePoints).join(', ')}`);
  }
  return channel;
}

// Extract exact provenance from a native assets.json on either channel.
//
// Development scans are supported so a maintainer can inspect a bNNNN native
// release, but only the stable channel is ever orchestrated; see
// requireStableProvenance. `manifestPath` is a path string.
export function scanNativeProvenance({
  manifestPath,
  nativeReleaseTag,
  bridgeSourceSha,
  bridgeBuildSha,
  channel,
  nativeReleasePublishedAt,
}) {
  const requested = requireChannel(channel);
  const path = pyPath(String(manifestPath));
  if (!pyIsFile(path) || pyIsSymlink(path)) throw new ContractError(`native manifest is not a regular file: ${path}`);
  const raw = pyReadBytes(path);
  let text;
  try {
    text = pyDecodeUtf8(raw);
  } catch (error) {
    if (isPyException(error, 'UnicodeDecodeError')) {
      throw new ContractError(`native manifest is not UTF-8: ${error.message}`, { cause: error });
    }
    throw error;
  }
  const manifest = strictJsonLoads(text, 'native manifest');
  if (!isDict(manifest)) throw new ContractError('native manifest root must be a JSON object');
  const identity = resolveNativeManifest(manifest, nativeReleaseTag);
  const provenance = new NativeProvenance({
    bridgeSourceSha,
    bridgeBuildSha,
    upstreamTag: identity.upstreamTag,
    upstreamCommit: identity.upstreamCommit,
    nativeRepo: NATIVE_REPOSITORY,
    nativeReleaseTag: identity.releaseTag,
    nativeCommit: identity.nativeCommit,
    nativeManifestSha256: createHash('sha256').update(raw).digest('hex'),
    nativeReleasePublishedAt,
  });
  if (provenance.channel !== requested) {
    throw new ContractError(
      `a ${requested} scan resolved the ${provenance.channel} native `
      + `release ${pyRepr(identity.releaseTag)} (${identity.upstreamTag})`,
    );
  }
  return provenance;
}

// (*version.version_parts, version.rebuild).
function releaseOrder(version) {
  return [...version.versionParts, version.rebuild];
}

// Return every published stable native tag after the migration baseline.
//
// The publication timestamp defines which releases belong to automatic
// orchestration; the tag ordering independently rejects a post-baseline
// rollback. This lets a later release receive its candidate while an earlier
// release advances through qualification, without backfilling the mutable
// historical release series.
export function selectStableNativeBacklog(releases, {
  baselineTag = STABLE_AUTOMATION_BASELINE_NATIVE_TAG,
  baselinePublishedAt = STABLE_AUTOMATION_BASELINE_PUBLISHED_AT,
} = {}) {
  if (pyFullmatch(UTC_TIMESTAMP_RE, baselinePublishedAt) === null) {
    throw new ContractError('stable automation baseline timestamp is not canonical');
  }
  const baseline = parseReleaseTag(baselineTag);
  if (baseline.channel !== Channel.STABLE) throw new ContractError('stable automation baseline tag is not stable');
  const baselineOrder = releaseOrder(baseline);

  const selected = [];
  const seenTags = new Set();
  releases.forEach((release, index) => {
    if (!isDict(release)) throw new ContractError(`native release listing entry ${index} is not an object`);
    const draft = pyGet(release, 'draft');
    if (typeof draft !== 'boolean') throw new ContractError(`native release listing entry ${index} has no boolean draft`);
    if (draft) return;
    const tag = pyGet(release, 'tag_name');
    if (typeof tag !== 'string' || !tag) throw new ContractError(`native release listing entry ${index} has no tag_name`);
    let version;
    try {
      version = parseReleaseTag(tag);
    } catch (error) {
      // Development and historical/foreign tag forms are not stable automatic-
      // publication candidates.
      if (error instanceof ContractError) return;
      throw error;
    }
    if (version.channel !== Channel.STABLE) return;
    const prerelease = pyGet(release, 'prerelease');
    if (typeof prerelease !== 'boolean' || prerelease !== version.githubPrerelease) {
      throw new ContractError(`stable native release ${pyRepr(tag)} has inconsistent prerelease state`);
    }
    const publishedAt = pyGet(release, 'published_at');
    if (typeof publishedAt !== 'string' || UTC_TIMESTAMP_RE.exec(publishedAt) === null) {
      throw new ContractError(`stable native release ${pyRepr(tag)} has no canonical published_at`);
    }
    if (compareCodePoints(publishedAt, baselinePublishedAt) <= 0) return;
    const order = releaseOrder(version);
    if (pyCompareIntTuples(order, baselineOrder) <= 0) {
      throw new ContractError(`post-baseline stable native release ${pyRepr(tag)} does not advance ${pyRepr(baselineTag)}`);
    }
    if (seenTags.has(tag)) throw new ContractError(`stable native release ${pyRepr(tag)} is duplicated`);
    seenTags.add(tag);
    selected.push([publishedAt, order, tag]);
  });

  // sorted() over (published_at, order, tag) tuples.
  selected.sort((left, right) => compareCodePoints(left[0], right[0])
    || pyCompareIntTuples(left[1], right[1])
    || compareCodePoints(left[2], right[2]));
  return selected.map(([, , tag]) => tag);
}

// _native_release_order.
export function nativeReleaseOrder(nativeReleaseTag) {
  return releaseOrder(parseReleaseTag(nativeReleaseTag));
}
