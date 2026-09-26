// Bridge-asset output tag selection for the stable release orchestrator, the
// Node port of scripts/release_orchestrator_release_tags.py.
//
// Picks the next free npm-shaped vMAJOR.MINOR.PATCH tag independently of the
// upstream tag, and determines which output tags candidate runs still claim
// across correlations.

import {
  Channel, ContractError, parseReleaseTag, parseUpstreamTag, validateNewReleaseIdentity,
} from '../contract.mjs';
import { compareCodePoints } from '../json.mjs';
import { pyCompareIntTuples, pyStrSplit } from '../python_compat.mjs';
import { ReleaseTarget } from './model.mjs';
import { CANDIDATE_RUN_NAME_RE } from './run_names.mjs';

// Bridge assets version independently of llama.cpp: an upstream v0.2.0
// candidate publishes as the next bridge patch (v0.1.39 -> v0.1.40), never as
// v0.2.0.
export const INITIAL_STABLE_RELEASE_TAG = 'v0.1.0';

// The version parts of a patch bump, as Python ints: bigint when the number
// leaves the safe-integer range.
function increment(value) {
  if (typeof value === 'bigint') return value + 1n;
  return Number.isSafeInteger(value + 1) ? value + 1 : BigInt(value) + 1n;
}

// Pick the next free bridge-asset tag, independently of the upstream tag.
//
// `releaseTags` are the assets repository's existing releases. `taken` are
// tags still claimed by a pipeline (any iterable). Both set the version floor,
// so a new tag is never lower than a published tag or a live claim. A claim
// that is released without publication stops counting, so its version may be
// taken again or stay unused. The result is always an unsuffixed
// vMAJOR.MINOR.PATCH with rebuild 0: npm orders -N as a prerelease of the same
// version, so a collision moves to the next free patch version instead of a
// rebuild suffix.
export function selectNextReleaseTarget(releaseTags, { upstreamTag, taken = [] }) {
  parseUpstreamTag(upstreamTag);
  const published = new Set([...releaseTags].filter((tag) => typeof tag === 'string'));
  const claimed = new Set([...published, ...[...taken].filter((tag) => typeof tag === 'string')]);
  const versions = [];
  for (const tag of claimed) {
    let version;
    try {
      version = parseReleaseTag(tag, { allowLegacy: true });
    } catch (error) {
      if (error instanceof ContractError) continue;
      throw error;
    }
    if (version.channel === Channel.STABLE) versions.push(version);
  }
  let major;
  let minor;
  let patch;
  if (versions.length > 0) {
    // max() with a tuple key keeps the first of equal keys; equal keys share
    // version parts, so the result does not depend on set order.
    let highest = versions[0];
    for (const version of versions.slice(1)) {
      if (pyCompareIntTuples([...version.versionParts, version.rebuild], [...highest.versionParts, highest.rebuild]) > 0) {
        highest = version;
      }
    }
    [major, minor, patch] = highest.versionParts;
    patch = increment(patch);
  } else {
    [major, minor, patch] = parseReleaseTag(INITIAL_STABLE_RELEASE_TAG).versionParts;
  }

  let tag = `v${major}.${minor}.${patch}`;
  while (claimed.has(tag)) {
    patch = increment(patch);
    tag = `v${major}.${minor}.${patch}`;
  }
  validateNewReleaseIdentity(tag, 0, upstreamTag);
  return new ReleaseTarget({ releaseTag: tag, releaseRebuild: 0 });
}

// _claimed_rebuilds_of: claimed tags that can publish only after `binding`
// publishes rebuild 0, sorted.
export function claimedRebuildsOf(binding, claimed) {
  if (BigInt(binding.releaseRebuild) !== 0n) return [];
  const version = parseReleaseTag(binding.releaseTag);
  const dependents = [];
  for (const tag of claimed) {
    let claim;
    try {
      claim = parseReleaseTag(tag);
    } catch (error) {
      if (error instanceof ContractError) continue;
      throw error;
    }
    if (pyCompareIntTuples(claim.versionParts, version.versionParts) === 0 && BigInt(claim.rebuild) > 0n) {
      dependents.push(tag);
    }
  }
  return dependents.sort(compareCodePoints);
}

const CORRELATION_BUILD_RE = /-build-(?<build>[0-9a-f]{16})$/u;

// _run_correlation_id.
function runCorrelationId(record) {
  const fields = pyStrSplit(record.runName);
  return fields.length > 1 ? fields[1] : null;
}

// _names_other_build.
function namesOtherBuild(correlationId, bridgeBuildSha) {
  const build = CORRELATION_BUILD_RE.exec(correlationId);
  return build !== null && build.groups.build !== bridgeBuildSha.slice(0, 16);
}

export function hasOtherBuildClaim(candidateRuns, { bridgeBuildSha }) {
  for (const record of candidateRuns) {
    const match = CANDIDATE_RUN_NAME_RE.exec(record.runName);
    if (match !== null && namesOtherBuild(match.groups.correlation_id, bridgeBuildSha)) return true;
  }
  return false;
}

// Output tags still claimed by candidate runs, across correlations, as a Set.
//
// A claim is dropped when its correlation names a build identity other than
// `bridgeBuildSha` and no candidate, qualification or publication run of that
// correlation is in flight, or when this scan already found that correlation
// satisfied by an identical release. Scans advance only the current build
// identity and a satisfied correlation publishes nothing, so nothing
// dispatches for such a correlation again.
export function claimedReleaseTags(candidateRuns, {
  bridgeBuildSha, downstreamRuns = [], satisfiedCorrelationIds = [],
}) {
  const inFlight = new Set(
    [...candidateRuns, ...downstreamRuns].filter((record) => record.inFlight).map(runCorrelationId),
  );
  const satisfied = new Set(satisfiedCorrelationIds);
  const claimed = new Set();
  for (const record of candidateRuns) {
    const match = CANDIDATE_RUN_NAME_RE.exec(record.runName);
    if (match === null) continue;
    const correlationId = match.groups.correlation_id;
    if (satisfied.has(correlationId) || (namesOtherBuild(correlationId, bridgeBuildSha) && !inFlight.has(correlationId))) {
      continue;
    }
    claimed.add(match.groups.release_tag);
  }
  return claimed;
}
