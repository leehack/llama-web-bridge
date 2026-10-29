// Live driver of the stable release orchestrator, the Node port of
// scripts/release_orchestrator_driver.py.
//
// Proves repository governance, gathers the evidence one pipeline already
// has, plans its single next transition, and dispatches it with a
// deterministic run-name readback.

import {
  ASSETS_REPOSITORY, BRIDGE_REPOSITORY, Channel, ContractError, NATIVE_REPOSITORY, parseReleaseTag,
  validateImmutableReleaseGovernance, validatePublicationEnvironment,
} from '../contract.mjs';
import {
  isDict, pyGet, pyJoinPath, pyPath, pyRepr, pyStr,
} from '../json.mjs';
import { candidatePublicationDigests } from '../publication_state.mjs';
import { pyCompareIntTuples } from '../python_compat.mjs';
import {
  correlationMarker, fetchAssetReleases, fetchAssetTagNames, findCorrelatedRelease, latestAlignedRelease,
  latestPublishedNativeAlignment, newestPublishedStableRelease, pages, publicationBytesIdentical, verifyPublishedRelease,
  workflowHistorySince,
} from './asset_releases.mjs';
import {
  CANDIDATE_WORKFLOW_FILE, CANDIDATE_WORKFLOW_PATH, OrchestrationAction, OrchestrationPlan, PUBLISH_WORKFLOW_FILE,
  PUBLISH_WORKFLOW_PATH, PipelineBinding, PipelineObservation, QUALIFICATION_WORKFLOW_FILE, QUALIFICATION_WORKFLOW_PATH,
  REPOSITORY_OWNER, requireStableProvenance, requireStr,
} from './model.mjs';
import { nativeReleaseOrder } from './native.mjs';
import { WORKFLOW_DISPATCH_INPUTS, planPipeline, requireExactDispatchInputs } from './planner.mjs';
import {
  candidateTagClaim, claimNamesOtherBuild, claimedReleaseTags, claimedRebuildsOf, claimsNotBelow, hasOtherBuildClaim,
  releaseTagClaims, runNameCorrelationId, selectNextReleaseTarget,
} from './release_tags.mjs';
import {
  computeCorrelationId, correlationNativeIdentity, correlationRecoveryBase, parsePublishRunName, publishRunName,
  qualificationRunName,
} from './run_names.mjs';
import { makeDirectories, verifyCandidateRun, verifyQualificationRun } from './stage_proofs.mjs';
import {
  candidateMatcher, fetchRecentRuns, fetchRuns, findNamedRun, mergeRunRecords, resolveCandidateBinding,
  selectPipelineRuns,
} from './workflow_runs.mjs';

// isinstance(value, Mapping) for a JSON value.
function isMapping(value) {
  return isDict(value) || value instanceof Map;
}

export const DISPATCH_READBACK_ATTEMPTS = 12;
export const DISPATCH_READBACK_DELAY_SECONDS = 5.0;
// Each recovery generation follows a further publication that overtook the
// previous one; more than this means output-tag order keeps breaking, which a
// maintainer must diagnose.
export const MAX_RECOVERY_GENERATIONS = 3;

export function requireDefaultBranch(gateway) {
  const payload = gateway.apiJson(`repos/${BRIDGE_REPOSITORY}`);
  if (!isMapping(payload)) throw new ContractError('repository response must be a JSON object');
  return requireStr(pyGet(payload, 'default_branch'), 'default_branch');
}

// Prove immutable-release governance live, never by constant assertion.
export function requireImmutableReleaseGovernance(gateway) {
  const payload = gateway.apiJson(`repos/${ASSETS_REPOSITORY}/immutable-releases`, { privileged: true });
  return validateImmutableReleaseGovernance(payload, ASSETS_REPOSITORY);
}

// Prove the existing environment policy before asserting publication approval.
export function requirePublicationEnvironment(gateway) {
  const environment = gateway.apiJson(`repos/${BRIDGE_REPOSITORY}/environments/bridge-assets-publication`);
  const branchPolicies = gateway.apiJson(
    `repos/${BRIDGE_REPOSITORY}/environments/bridge-assets-publication/deployment-branch-policies`,
  );
  if (!isMapping(environment) || !isMapping(branchPolicies)) {
    throw new ContractError('publication environment responses must be JSON objects');
  }
  validatePublicationEnvironment(environment, branchPolicies);
}

// Evidence that the run listing a dispatch plan was built from was stale: a
// fresh read of the target stage found runs that contradict the plan.
// `conflicting` are the pipeline's own runs that forbid the dispatch, and
// `claims` are other pipelines' candidate runs, as { record, releaseTag,
// correlationId }, still claiming an output tag at or above a fresh candidate
// binding's. `evidence` maps each workflow file read to the fresh records a
// re-plan merges into its listing. `repeated` marks a re-plan that wanted the
// same stage again despite them.
export class StaleObservation {
  constructor({
    workflowFile, plan, evidence, conflicting = [], claims = [], repeated = false,
  }) {
    this.workflowFile = workflowFile;
    this.plan = plan;
    this.evidence = new Map([...evidence].map(([file, runs]) => [file, Object.freeze([...runs])]));
    this.conflicting = Object.freeze([...conflicting]);
    this.claims = Object.freeze([...claims]);
    this.repeated = repeated;
    Object.freeze(this);
  }

  // Whether a re-plan's `plan` is still contradicted by this evidence: it
  // dispatches the same stage while the pipeline's own runs forbid it, or
  // with an output tag a claim here is not below.
  refuses(plan) {
    if (plan.dispatchWorkflow !== this.workflowFile) return false;
    if (this.conflicting.length > 0) return true;
    const releaseTag = plan.releaseTarget?.releaseTag;
    return typeof releaseTag === 'string' && claimsNotBelow(releaseTag, this.claims.map((claim) => claim.releaseTag)).length > 0;
  }

  // The same evidence, refusing `plan`, a re-plan that still wants this stage.
  repeatedBy(plan) {
    return new StaleObservation({
      workflowFile: this.workflowFile,
      plan,
      evidence: this.evidence,
      conflicting: this.conflicting,
      claims: this.claims,
      repeated: true,
    });
  }

  describe() {
    const state = (record) => (record.inFlight ? record.status : record.conclusion);
    const runs = [
      ...this.conflicting.map((record) => `${record.runId} (${state(record)})`),
      ...this.claims.map(({ record, releaseTag, correlationId }) => (
        `${record.runId} (${state(record)}, claiming ${releaseTag} for ${correlationId})`
      )),
    ];
    return `${this.workflowFile} run(s) ${runs.join(', ')}`;
  }
}

// Stages whose failed runs the observation already turns into an automatic
// retry. A failed candidate or qualification run blocks its stage instead
// (see the BLOCKED plans in observeAndPlan), so for those stages a fresh
// failed run contradicts a dispatch plan as much as a live or successful one.
const RETRIED_AFTER_FAILURE_WORKFLOWS = Object.freeze(new Set([PUBLISH_WORKFLOW_FILE]));

const STAGE_LABELS = Object.freeze(new Map([
  [CANDIDATE_WORKFLOW_FILE, 'candidate'],
  [QUALIFICATION_WORKFLOW_FILE, 'qualification'],
  [PUBLISH_WORKFLOW_FILE, 'publication'],
]));

const DOWNSTREAM_WORKFLOWS = Object.freeze([
  [QUALIFICATION_WORKFLOW_FILE, QUALIFICATION_WORKFLOW_PATH],
  [PUBLISH_WORKFLOW_FILE, PUBLISH_WORKFLOW_PATH],
]);

// Which runs of a dispatch plan's stage belong to its pipeline, whatever
// binding they were dispatched with: any candidate of the correlation, and a
// qualification or publication of the correlation's exact candidate run.
export function stageCorrelationMatcher(plan) {
  const { correlationId, candidateRunId } = plan;
  if (plan.dispatchWorkflow === CANDIDATE_WORKFLOW_FILE) return candidateMatcher(correlationId);
  if (plan.dispatchWorkflow === QUALIFICATION_WORKFLOW_FILE) {
    const expected = qualificationRunName(correlationId, requireStr(candidateRunId, 'candidate_run_id'));
    return (name) => name === expected;
  }
  if (plan.dispatchWorkflow === PUBLISH_WORKFLOW_FILE) {
    requireStr(candidateRunId, 'candidate_run_id');
    return (name) => {
      const parsed = parsePublishRunName(name, correlationId);
      return parsed !== null && parsed[0] === candidateRunId;
    };
  }
  throw new ContractError(`${pyRepr(plan.dispatchWorkflow)} is not a workflow this orchestrator may dispatch`);
}

// Whether `correlationId` is a pipeline of a native release newer than
// `nativeReleaseTag`.
function namesNewerNative(correlationId, nativeReleaseTag) {
  const other = correlationNativeIdentity(correlationId);
  return other !== null
    && pyCompareIntTuples(nativeReleaseOrder(other.nativeReleaseTag), nativeReleaseOrder(nativeReleaseTag)) > 0;
}

export function fetchNativeReleases(gateway) {
  const payload = gateway.apiJson(`repos/${NATIVE_REPOSITORY}/releases?per_page=100`, { paginate: true });
  const releases = [];
  for (const page of pages(payload, 'native releases')) {
    for (const release of Array.isArray(page) ? page : [page]) {
      if (!isMapping(release)) throw new ContractError('native release record must be a JSON object');
      releases.push(release);
    }
  }
  return releases;
}

// Why a newer native's tag `claim` proves nothing, or null when it is the
// pipeline of a published stable native release whose assets.json is the
// manifest its correlation names. Native releases are mutable, so a deleted,
// drafted or re-uploaded release leaves a claim that no scan advances.
export function unprovenNativeClaim(nativeReleases, claim) {
  const { nativeReleaseTag, nativeManifestPrefix } = correlationNativeIdentity(claim.correlationId);
  const subject = (
    `the pipeline of newer native release ${nativeReleaseTag} claims ${claim.releaseTag} `
    + `(correlation ${claim.correlationId})`
  );
  const release = nativeReleases.find((record) => pyGet(record, 'tag_name') === nativeReleaseTag);
  if (
    release === undefined
    || pyGet(release, 'draft') !== false
    || pyGet(release, 'prerelease') !== parseReleaseTag(nativeReleaseTag).githubPrerelease
  ) {
    return `${subject}, but ${nativeReleaseTag} is not a published stable native release`;
  }
  const assets = pyGet(release, 'assets');
  const manifests = Array.isArray(assets)
    ? assets.filter((asset) => isMapping(asset) && pyGet(asset, 'name') === 'assets.json')
    : [];
  const digest = manifests.length === 1 ? pyGet(manifests[0], 'digest') : null;
  if (typeof digest !== 'string' || !digest.startsWith(`sha256:${nativeManifestPrefix}`)) {
    return `${subject}, but the published assets.json of ${nativeReleaseTag} is not the manifest that correlation names`;
  }
  return null;
}

// Both fresh reads of one workflow, merged: the complete filtered history and
// the newest unfiltered page. `correlated` names the guarded pipeline's own
// runs; `select` other pipelines' runs worth keeping.
function freshStageRuns(gateway, {
  workflowFile, workflowPath, defaultBranch, createdSince, correlated, select = () => false,
}) {
  const searched = fetchRuns(gateway, {
    workflowFile, workflowPath, defaultBranch, createdSince,
  }).filter((record) => correlated(record.runName) || select(record.runName));
  const recent = fetchRecentRuns(gateway, {
    workflowFile, workflowPath, defaultBranch, correlated, select,
  });
  return mergeRunRecords(searched, recent);
}

// Other pipelines' candidate runs in `candidateRuns` that still claim an
// output tag at or above `releaseTag`, or any output tag for a newer native
// release, as claimedReleaseTags decides for each claiming correlation. A
// claim of another build identity holds only while a run of its correlation
// is in flight, so before such a claim is released the downstream stages are
// read fresh too, and recorded in `evidence`.
function freshTagClaims(gateway, plan, candidateRuns, {
  defaultBranch, createdSince, satisfiedCorrelationIds, evidence,
}) {
  const { bridgeBuildSha } = plan.provenance;
  const releaseTag = plan.releaseTarget.releaseTag;
  const byCorrelation = new Map();
  const candidates = [];
  for (const record of candidateRuns) {
    const claim = candidateTagClaim(record.runName);
    if (claim === null || claim.correlationId === plan.correlationId) continue;
    if (!byCorrelation.has(claim.correlationId)) byCorrelation.set(claim.correlationId, []);
    byCorrelation.get(claim.correlationId).push(record);
    if (
      claimsNotBelow(releaseTag, [claim.releaseTag]).length > 0
      || namesNewerNative(claim.correlationId, plan.provenance.nativeReleaseTag)
    ) {
      candidates.push({ record, ...claim });
    }
  }
  if (candidates.length === 0) return { claims: [], records: [] };
  const releasable = new Set(candidates
    .map(({ correlationId }) => correlationId)
    .filter((correlationId) => (
      claimNamesOtherBuild(correlationId, bridgeBuildSha)
      && !byCorrelation.get(correlationId).some((record) => record.inFlight)
    )));
  const downstreamRuns = [];
  if (releasable.size > 0) {
    for (const [workflowFile, workflowPath] of DOWNSTREAM_WORKFLOWS) {
      const runs = freshStageRuns(gateway, {
        workflowFile,
        workflowPath,
        defaultBranch,
        createdSince,
        correlated: () => false,
        select: (name) => releasable.has(runNameCorrelationId(name)),
      });
      evidence.set(workflowFile, runs);
      downstreamRuns.push(...runs);
    }
  }
  const claims = candidates.filter(({ correlationId, releaseTag: claimed }) => claimedReleaseTags(
    byCorrelation.get(correlationId),
    { bridgeBuildSha, downstreamRuns, satisfiedCorrelationIds },
  ).has(claimed));
  const claimants = new Set(claims.map(({ correlationId }) => correlationId));
  return { claims, records: [...claimants].flatMap((correlationId) => byCorrelation.get(correlationId)) };
}

// The correlation guard every dispatch passes: read the target stage again,
// through two differently shaped queries (the complete filtered history and
// the newest unfiltered page), and return a StaleObservation when a run of
// this pipeline is live or successful, or has failed at a stage that is never
// retried automatically, or, for a candidate, when another pipeline still
// claims its output tag or a higher one. Returns null when the dispatch is
// still the next step.
export function guardStageCorrelation(gateway, plan, {
  defaultBranch, workflowPath, createdSince, satisfiedCorrelationIds = [],
}) {
  const workflowFile = plan.dispatchWorkflow;
  const matcher = stageCorrelationMatcher(plan);
  const isCandidate = workflowFile === CANDIDATE_WORKFLOW_FILE;
  const runs = freshStageRuns(gateway, {
    workflowFile,
    workflowPath,
    defaultBranch,
    createdSince,
    correlated: matcher,
    select: isCandidate ? (name) => candidateTagClaim(name) !== null : () => false,
  });
  const selection = selectPipelineRuns(runs, { label: STAGE_LABELS.get(workflowFile), matcher, defaultBranch });
  const retried = RETRIED_AFTER_FAILURE_WORKFLOWS.has(workflowFile);
  const conflicting = selection.matched.filter((record) => record.inFlight || record.succeeded || !retried);
  const evidence = new Map();
  const { claims, records } = isCandidate && conflicting.length === 0
    ? freshTagClaims(gateway, plan, runs, {
      defaultBranch, createdSince, satisfiedCorrelationIds, evidence,
    })
    : { claims: [], records: [] };
  if (conflicting.length === 0 && claims.length === 0) return null;
  evidence.set(workflowFile, [...selection.matched, ...records]);
  return new StaleObservation({
    workflowFile, plan, evidence, conflicting, claims,
  });
}

// A copy of `plan` with `reason`.
function withReason(plan, reason) {
  return new OrchestrationPlan(Object.fromEntries(OrchestrationPlan.FIELDS.map(([camel]) => [
    camel, camel === 'reason' ? reason : plan[camel],
  ])));
}

// Advance one pipeline by at most one stage. `workspace` is a path string;
// reservedReleaseTags and satisfiedCorrelationIds are any iterables.
//
// A dispatch plan is built from run listings that can be stale. When the
// correlation guard finds this pipeline's runs that the listing omitted, the
// pipeline is observed and planned once more with that evidence merged in. A
// re-plan that wants the same stage again, or meets a second stale listing,
// ends as BLOCKED without dispatching; the next scan observes it afresh.
export function advancePipeline(gateway, options) {
  const first = observeAndPlan(gateway, options, null);
  if (!(first instanceof StaleObservation)) return first;
  // The re-plan proves its stages again from fresh downloads, which refuse a
  // non-empty destination, so it works in its own directory.
  const workspace = pyJoinPath(pyPath(String(options.workspace)), 'replan');
  makeDirectories(workspace);
  const second = observeAndPlan(gateway, { ...options, workspace }, first);
  const stale = (
    `the ${first.workflowFile} run listing this scan planned ${first.plan.action} from was stale: `
    + `a fresh correlation read found ${first.describe()} that it omitted`
  );
  if (!(second instanceof StaleObservation)) return withReason(second, `${second.reason} (re-planned once: ${stale})`);
  const target = second.plan.releaseTarget === null ? '' : ` for ${second.plan.releaseTarget.releaseTag}`;
  const outcome = second.repeated
    ? `re-planned once with that evidence, the pipeline still planned ${second.plan.action}${target}`
    : (
      `re-planned once with that evidence, the pipeline planned ${second.plan.action}, `
      + `but a fresh correlation read found ${second.describe()} that its listing omitted too`
    );
  return new OrchestrationPlan({
    action: OrchestrationAction.BLOCKED,
    reason: `${stale}; ${outcome}; nothing was dispatched, and the next scan observes the pipeline again`,
    provenance: second.plan.provenance,
    correlationId: second.plan.correlationId,
    releaseTarget: second.plan.releaseTarget,
    candidateRunId: second.plan.candidateRunId,
    qualificationRunId: second.plan.qualificationRunId,
  });
}

// One observation and plan of advancePipeline. `evidence` is null, or the
// StaleObservation of an earlier pass, whose runs are merged into the listing
// of its stage; a plan that wants that stage again returns it as repeated.
function observeAndPlan(gateway, options, evidence) {
  const {
    provenance,
    workspace,
    dryRun = false,
    reservedReleaseTags = [],
    satisfiedCorrelationIds = [],
    publicationAllowed = true,
    publicationBarrierNativeTag = null,
    newerNativeScanned = false,
    scannedProvenances = [],
    recovery = 0,
  } = options;
  requireStableProvenance(provenance);
  if (!publicationAllowed) {
    if (typeof publicationBarrierNativeTag !== 'string') {
      throw new ContractError('a disabled publication transition requires the earlier native tag');
    }
    const barrier = parseReleaseTag(publicationBarrierNativeTag);
    if (barrier.channel !== Channel.STABLE) throw new ContractError('publication barrier native tag must be stable');
  }
  const correlationId = computeCorrelationId(provenance, { recovery });
  const defaultBranch = requireDefaultBranch(gateway);
  // Governance is a prerequisite for every state classification, including a
  // noop: a disabled or unreadable policy is never reported as healthy stable
  // automation merely because an older release happens to exist.
  requireImmutableReleaseGovernance(gateway);

  const releases = fetchAssetReleases(gateway);
  const correlated = findCorrelatedRelease(releases, correlationId, provenance);
  if (correlated !== null) {
    const published = verifyPublishedRelease(gateway, {
      release: correlated,
      provenance,
      correlationId,
      workspace,
    });
    if (!publicationAllowed) {
      return new OrchestrationPlan({
        action: OrchestrationAction.BLOCKED,
        reason: (
          `${provenance.nativeReleaseTag} is already published while `
          + `earlier native release ${publicationBarrierNativeTag} is not; `
          + 'the ordered publication history is inconsistent'
        ),
        provenance,
        correlationId,
        releaseTarget: published.releaseTarget,
      });
    }
    return planPipeline({ provenance, correlationId, observation: new PipelineObservation({ published }) });
  }

  const alignment = latestPublishedNativeAlignment(releases);
  if (
    alignment !== null
    && pyCompareIntTuples(nativeReleaseOrder(provenance.nativeReleaseTag), nativeReleaseOrder(alignment[0])) < 0
  ) {
    const [alignedNativeTag, alignedAssetTag] = alignment;
    if (!newerNativeScanned) {
      return new OrchestrationPlan({
        action: OrchestrationAction.BLOCKED,
        reason: (
          `asset release ${alignedAssetTag} records native release `
          + `${alignedNativeTag}, newer than ${provenance.nativeReleaseTag}, `
          + 'but no newer stable native release is part of this scan; '
          + 'refusing to skip the newest scanned native release'
        ),
        provenance,
        correlationId,
      });
    }
    return new OrchestrationPlan({
      action: OrchestrationAction.SUPERSEDED,
      reason: (
        `${provenance.nativeReleaseTag} is behind native release `
        + `${alignedNativeTag}, already published as ${alignedAssetTag}; `
        + 'only native releases at or ahead of the published alignment '
        + 'are advanced for a new build identity'
      ),
      provenance,
      correlationId,
    });
  }

  const runHistorySince = workflowHistorySince(releases, provenance);
  const observedRuns = (workflowFile, workflowPath) => {
    const runs = fetchRuns(gateway, {
      workflowFile,
      workflowPath,
      defaultBranch,
      createdSince: runHistorySince,
    });
    const fresh = evidence?.evidence.get(workflowFile);
    return fresh === undefined ? runs : mergeRunRecords(runs, fresh);
  };
  const dispatch = (plan, workflowPath) => {
    if (evidence !== null && evidence.refuses(plan)) return evidence.repeatedBy(plan);
    return executeDispatch(gateway, plan, {
      defaultBranch, workflowPath, dryRun, createdSince: runHistorySince, satisfiedCorrelationIds,
    });
  };
  const assetTagNames = fetchAssetTagNames(gateway);
  const candidateRuns = observedRuns(CANDIDATE_WORKFLOW_FILE, CANDIDATE_WORKFLOW_PATH);
  const candidateSelection = selectPipelineRuns(candidateRuns, {
    label: 'candidate',
    matcher: candidateMatcher(correlationId),
    defaultBranch,
  });
  const persisted = resolveCandidateBinding(candidateSelection, correlationId);

  const stillClaimed = () => {
    const downstreamWorkflows = hasOtherBuildClaim(candidateRuns, { bridgeBuildSha: provenance.bridgeBuildSha })
      ? [[QUALIFICATION_WORKFLOW_FILE, QUALIFICATION_WORKFLOW_PATH], [PUBLISH_WORKFLOW_FILE, PUBLISH_WORKFLOW_PATH]]
      : [];
    const downstreamRuns = downstreamWorkflows.flatMap(([workflowFile, workflowPath]) => observedRuns(workflowFile, workflowPath));
    return releaseTagClaims(candidateRuns, {
      bridgeBuildSha: provenance.bridgeBuildSha,
      downstreamRuns,
      satisfiedCorrelationIds,
    });
  };
  const stillClaimedTags = () => new Set(stillClaimed().map((claim) => claim.releaseTag));
  // Newer native releases' pipelines, whether or not this scan includes them:
  // a targeted scan of one older native sees no newer provenance.
  const newerNativeClaims = () => stillClaimed()
    .filter((claim) => namesNewerNative(claim.correlationId, provenance.nativeReleaseTag));
  // A scanned provenance proves its own claims. Any other claim is held to the
  // native repository, read at most once per observation.
  const scannedCorrelationIds = new Set([...scannedProvenances].map(computeCorrelationId));
  let nativeReleases = null;
  const unprovenClaim = (claims) => {
    for (const claim of claims) {
      if (scannedCorrelationIds.has(correlationRecoveryBase(claim.correlationId))) continue;
      nativeReleases ??= fetchNativeReleases(gateway);
      const why = unprovenNativeClaim(nativeReleases, claim);
      if (why !== null) return why;
    }
    return null;
  };
  const unprovenReason = (why) => (
    `${why}; output-tag order against that claim cannot be proven, so nothing was dispatched; `
    + 'a maintainer must resolve the claim or establish a new governed bridge build identity'
  );

  // A fresh pipeline would take the next free output tag, above any newer
  // native's pipeline, and the publication barrier would then publish this
  // older native first under the higher tag.
  if (candidateSelection.matched.length === 0) {
    if (newerNativeScanned) {
      return new OrchestrationPlan({
        action: OrchestrationAction.SUPERSEDED,
        reason: (
          `${provenance.nativeReleaseTag} has no pipeline for this build identity `
          + 'and a newer stable native release is part of this scan; only the '
          + 'newest scanned native release starts a new build identity'
        ),
        provenance,
        correlationId,
      });
    }
    const newerClaims = newerNativeClaims();
    const unproven = unprovenClaim(newerClaims);
    if (unproven !== null) {
      return new OrchestrationPlan({
        action: OrchestrationAction.BLOCKED, reason: unprovenReason(unproven), provenance, correlationId,
      });
    }
    const [newerClaim] = newerClaims;
    if (newerClaim !== undefined) {
      const newer = correlationNativeIdentity(newerClaim.correlationId).nativeReleaseTag;
      return new OrchestrationPlan({
        action: OrchestrationAction.SUPERSEDED,
        reason: (
          `${provenance.nativeReleaseTag} has no pipeline for this build identity `
          + `and the pipeline of newer native release ${newer} claims ${newerClaim.releaseTag}; `
          + 'a fresh pipeline would take a higher output tag and publish first, '
          + `making ${newerClaim.releaseTag} an ordering rollback, so ${newer} supersedes it`
        ),
        provenance,
        correlationId,
      });
    }
  }

  let freshBinding = null;
  if (persisted === null) {
    const target = selectNextReleaseTarget(releases.map((release) => pyStr(pyGet(release, 'tag_name'))), {
      upstreamTag: provenance.upstreamTag,
      taken: new Set([...assetTagNames, ...stillClaimedTags(), ...reservedReleaseTags]),
    });
    freshBinding = new PipelineBinding({
      bridgeSourceSha: provenance.bridgeSourceSha,
      releaseTag: target.releaseTag,
      releaseRebuild: target.releaseRebuild,
    });
  }

  if (candidateSelection.inFlightRunId !== null) {
    return planPipeline({
      provenance,
      correlationId,
      observation: new PipelineObservation({
        binding: persisted,
        freshBinding,
        candidateInFlightRunId: candidateSelection.inFlightRunId,
      }),
    });
  }

  if (candidateSelection.succeededRunId === null && candidateSelection.unsuccessful.length > 0) {
    const failedIds = candidateSelection.unsuccessful.map((record) => record.runId).join(', ');
    if (persisted === null) throw new ContractError('failed candidate runs have no exact persisted binding');
    return new OrchestrationPlan({
      action: OrchestrationAction.BLOCKED,
      reason: (
        `candidate run(s) ${failedIds} failed for this exact provenance; `
        + 'automatic candidate retries are disabled to prevent unbounded '
        + 'daily duplicates; a maintainer must diagnose the failure and '
        + 'explicitly dispatch one deliberate new first-attempt run with the '
        + 'same exact binding, or establish new provenance'
      ),
      provenance,
      correlationId,
      releaseTarget: persisted.releaseTarget,
    });
  }

  if (candidateSelection.succeededRunId === null) {
    const plan = planPipeline({ provenance, correlationId, observation: new PipelineObservation({ freshBinding }) });
    return dispatch(plan, CANDIDATE_WORKFLOW_PATH);
  }

  if (persisted === null) throw new ContractError('a successful candidate run advertises no parsable pipeline binding');
  const candidate = verifyCandidateRun(gateway, {
    runId: candidateSelection.succeededRunId,
    provenance,
    correlationId,
    binding: persisted,
    defaultBranch,
    workspace,
  });

  const satisfiedPlan = (qualificationRunId) => {
    const aligned = latestAlignedRelease(releases, provenance);
    if (aligned === null || claimedRebuildsOf(candidate.binding, stillClaimedTags()).length > 0) return null;
    const [alignedRelease, alignedCorrelationId] = aligned;
    const published = verifyPublishedRelease(gateway, {
      release: alignedRelease,
      provenance,
      correlationId: alignedCorrelationId,
      workspace,
    });
    if (published.directory === null) throw new ContractError('verified release did not retain its downloaded bytes');
    if (!publicationBytesIdentical(
      candidatePublicationDigests(candidate.directory),
      candidatePublicationDigests(published.directory),
    )) {
      return null;
    }
    return planPipeline({
      provenance,
      correlationId,
      observation: new PipelineObservation({
        binding: candidate.binding,
        candidateRunId: candidate.runId,
        qualificationRunId,
        satisfiedBy: published,
      }),
    });
  };

  const releaseTag = candidate.binding.releaseTag;
  const retired = (action, reason, { qualificationRunId, releaseTarget = candidate.binding.releaseTarget }) => (
    new OrchestrationPlan({
      action, reason, provenance, correlationId, releaseTarget, candidateRunId: candidate.runId, qualificationRunId,
    })
  );

  // Publication follows native order, so an older native whose output tag is
  // above a newer native's would publish first and turn the newer
  // native's lower tag into an ordering rollback.
  const outOfOrderPlan = (qualificationRunId) => {
    const below = newerNativeClaims().filter((claim) => claimsNotBelow(claim.releaseTag, [releaseTag]).length > 0);
    if (below.length === 0) return null;
    const unproven = unprovenClaim(below);
    if (unproven !== null) return retired(OrchestrationAction.BLOCKED, unprovenReason(unproven), { qualificationRunId });
    const [claim] = below;
    const newer = correlationNativeIdentity(claim.correlationId).nativeReleaseTag;
    return retired(OrchestrationAction.SUPERSEDED, (
      `candidate ${candidate.runId} is bound to ${releaseTag}, above ${claim.releaseTag} `
      + `claimed by the pipeline of newer native release ${newer}; publishing `
      + `${provenance.nativeReleaseTag} first would make ${claim.releaseTag} an ordering `
      + `rollback, so ${newer} supersedes it and nothing further is dispatched`
    ), { qualificationRunId });
  };

  // Once a published release reaches this candidate's output tag, publishing it
  // is an ordering rollback, so it is never retried. It is satisfied only by a
  // proven byte-identical publication, and replaced by a recovery pipeline when
  // its bytes are proven to differ.
  const overtakenPlan = (qualificationRunId) => {
    const head = newestPublishedStableRelease(releases);
    if (head === null) return null;
    const headTag = pyStr(pyGet(head, 'tag_name'));
    if (claimsNotBelow(releaseTag, [headTag]).length === 0) return null;
    const overtaken = (
      `published release ${headTag} is at or above output tag ${releaseTag} of `
      + `candidate ${candidate.runId}, so publishing it would be an ordering rollback`
    );
    // The next recovery generation of this provenance: a new correlation, so a
    // fresh output tag and new first-attempt candidate, qualification and
    // publication.
    const recover = (why) => {
      const recovered = observeAndPlan(gateway, { ...options, recovery: recovery + 1 }, evidence);
      if (recovered instanceof StaleObservation) return recovered;
      return withReason(recovered, (
        `${overtaken}; ${why}; recovery pipeline ${recovered.correlationId}: ${recovered.reason}`
      ));
    };
    // A published later generation settles this one, whatever its bytes.
    const laterGenerations = Array.from(
      { length: Math.max(0, MAX_RECOVERY_GENERATIONS - recovery) },
      (_, index) => correlationMarker(computeCorrelationId(provenance, { recovery: recovery + 1 + index })),
    );
    if (releases.some((release) => {
      const body = pyGet(release, 'body');
      return typeof body === 'string' && laterGenerations.some((marker) => body.includes(marker));
    })) {
      return recover('a later recovery generation of this provenance is published');
    }
    const satisfied = satisfiedPlan(qualificationRunId);
    if (satisfied !== null) return satisfied;
    // Blocked, it would hold the publication barrier and starve the newer
    // native release that supersedes it anyway.
    if (newerNativeScanned) {
      return retired(OrchestrationAction.SUPERSEDED, (
        `${overtaken}; a newer stable native release is part of this scan and `
        + 'supersedes it, so nothing further is dispatched'
      ), { qualificationRunId });
    }
    const blocked = (why) => retired(OrchestrationAction.BLOCKED, (
      `${overtaken}; ${why}; nothing was dispatched`
    ), { qualificationRunId });
    const body = pyGet(head, 'body');
    const [publisher, publisherCorrelationId] = (typeof body !== 'string' ? undefined : [...scannedProvenances]
      .filter((other) => other.nativeReleaseTag !== provenance.nativeReleaseTag)
      .flatMap((other) => Array.from(
        { length: MAX_RECOVERY_GENERATIONS + 1 },
        (_, generation) => [other, computeCorrelationId(other, { recovery: generation })],
      ))
      .find(([, otherCorrelationId]) => body.includes(correlationMarker(otherCorrelationId)))) ?? [];
    if (publisher === undefined) {
      return blocked(
        `${headTag} is not the publication of another native release in this scan, so its bytes cannot be compared; `
        + 'a scan that includes the native release it records compares them',
      );
    }
    const published = verifyPublishedRelease(gateway, {
      release: head,
      provenance: publisher,
      correlationId: publisherCorrelationId,
      workspace,
    });
    if (published.directory === null) throw new ContractError('verified release did not retain its downloaded bytes');
    if (!publicationBytesIdentical(
      candidatePublicationDigests(candidate.directory),
      candidatePublicationDigests(published.directory),
    )) {
      if (recovery >= MAX_RECOVERY_GENERATIONS) {
        return blocked(
          `its publication files differ from ${headTag}, and recovery generation ${recovery} is the last automatic one; `
          + 'a maintainer must diagnose why output tags keep being overtaken',
        );
      }
      return recover(`its publication files differ from ${headTag}`);
    }
    return retired(OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE, (
      `${overtaken}; its publication files other than manifest.json are byte-identical `
      + `to the immutable release ${headTag} (id ${pyStr(published.releaseId)}, published `
      + `${published.publishedAt}), which records native ${publisher.nativeReleaseTag}; `
      + 'nothing further is dispatched'
    ), { qualificationRunId, releaseTarget: published.releaseTarget });
  };

  const qualificationRuns = observedRuns(QUALIFICATION_WORKFLOW_FILE, QUALIFICATION_WORKFLOW_PATH);
  const expectedQualificationName = qualificationRunName(correlationId, candidate.runId);
  const qualificationSelection = selectPipelineRuns(qualificationRuns, {
    label: 'qualification',
    matcher: (name) => name === expectedQualificationName,
    defaultBranch,
  });
  if (
    qualificationSelection.succeededRunId === null
    && qualificationSelection.inFlightRunId === null
    && qualificationSelection.unsuccessful.length > 0
  ) {
    const failedIds = qualificationSelection.unsuccessful.map((record) => record.runId).join(', ');
    return new OrchestrationPlan({
      action: OrchestrationAction.BLOCKED,
      reason: (
        `qualification run(s) ${failedIds} failed for candidate `
        + `${candidate.runId}; automatic qualification retries are disabled `
        + 'to prevent unbounded duplicates, so a maintainer must diagnose the '
        + 'heavy-gate failure before this candidate can advance'
      ),
      provenance,
      correlationId,
      releaseTarget: candidate.binding.releaseTarget,
      candidateRunId: candidate.runId,
    });
  }

  if (qualificationSelection.succeededRunId === null) {
    if (qualificationSelection.inFlightRunId === null) {
      const settled = outOfOrderPlan(null) ?? overtakenPlan(null) ?? satisfiedPlan(null);
      if (settled !== null) return settled;
    }
    const plan = planPipeline({
      provenance,
      correlationId,
      observation: new PipelineObservation({
        binding: candidate.binding,
        candidateRunId: candidate.runId,
        qualificationInFlightRunId: qualificationSelection.inFlightRunId,
      }),
    });
    return dispatch(plan, QUALIFICATION_WORKFLOW_PATH);
  }

  verifyQualificationRun(gateway, {
    runId: qualificationSelection.succeededRunId,
    candidate,
    provenance,
    correlationId,
    defaultBranch,
    workspace,
  });

  const publishRuns = observedRuns(PUBLISH_WORKFLOW_FILE, PUBLISH_WORKFLOW_PATH);
  const expectedPublishName = publishRunName(
    correlationId,
    candidate.runId,
    qualificationSelection.succeededRunId,
    candidate.binding,
  );
  const publishSelection = selectPipelineRuns(publishRuns, {
    label: 'publication',
    matcher: (name) => name === expectedPublishName,
    defaultBranch,
  });
  const publishIdle = publishSelection.inFlightRunId === null && publishSelection.succeededRunId === null;
  if (publishIdle) {
    const settled = outOfOrderPlan(qualificationSelection.succeededRunId)
      ?? overtakenPlan(qualificationSelection.succeededRunId);
    if (settled !== null) return settled;
  }
  if (!publicationAllowed) {
    if (publishSelection.inFlightRunId !== null || publishSelection.succeededRunId !== null) {
      return new OrchestrationPlan({
        action: OrchestrationAction.BLOCKED,
        reason: (
          `publication for ${provenance.nativeReleaseTag} was started `
          + `before earlier native release ${publicationBarrierNativeTag} `
          + 'was immutably published'
        ),
        provenance,
        correlationId,
        releaseTarget: candidate.binding.releaseTarget,
        candidateRunId: candidate.runId,
        qualificationRunId: qualificationSelection.succeededRunId,
        inFlightWorkflow: publishSelection.inFlightRunId !== null ? PUBLISH_WORKFLOW_FILE : null,
        inFlightRunId: publishSelection.inFlightRunId,
      });
    }
    return new OrchestrationPlan({
      action: OrchestrationAction.WAITING_FOR_PRIOR_PUBLICATION,
      reason: (
        `candidate ${candidate.runId} and qualification `
        + `${qualificationSelection.succeededRunId} are proven, but `
        + `earlier native release ${publicationBarrierNativeTag} must be `
        + 'immutably published first to preserve monotonic output ordering'
      ),
      provenance,
      correlationId,
      releaseTarget: candidate.binding.releaseTarget,
      candidateRunId: candidate.runId,
      qualificationRunId: qualificationSelection.succeededRunId,
    });
  }
  if (publishIdle) {
    const satisfied = satisfiedPlan(qualificationSelection.succeededRunId);
    if (satisfied !== null) return satisfied;
  }
  const plan = planPipeline({
    provenance,
    correlationId,
    observation: new PipelineObservation({
      binding: candidate.binding,
      candidateRunId: candidate.runId,
      qualificationRunId: qualificationSelection.succeededRunId,
      publishInFlightRunId: publishSelection.inFlightRunId,
      publishSucceededRunId: publishSelection.succeededRunId,
      publishRetry: publishSelection.unsuccessful.length > 0,
    }),
  });
  return dispatch(plan, PUBLISH_WORKFLOW_PATH);
}

// _execute_dispatch. Returns the plan with its dispatch record, a BLOCKED
// plan, or the guard's StaleObservation when a fresh correlation read
// contradicts the plan (advancePipeline re-plans on it; nothing is
// dispatched). `createdSince` bounds the guard's history read; it defaults to
// the native release's publication, before which no run of the correlation
// can exist.
export function executeDispatch(gateway, plan, {
  defaultBranch, workflowPath, dryRun, createdSince = null, satisfiedCorrelationIds = [],
}) {
  if (plan.dispatchWorkflow === null || plan.dispatchInputs === null) return plan;
  const runName = plan.dispatchRunName;
  if (runName === null) throw new ContractError('a dispatch plan must carry its deterministic run name');

  const identity = gateway.dispatchIdentity();
  if (identity !== REPOSITORY_OWNER) {
    return new OrchestrationPlan({
      action: OrchestrationAction.BLOCKED,
      reason: (
        'automatic dispatch identity could not be proven as '
        + `${pyRepr(REPOSITORY_OWNER)} (resolved ${pyRepr(identity ?? null)}); the owner-only actor `
        + 'gates stay in force and nothing was dispatched'
      ),
      provenance: plan.provenance,
      correlationId: plan.correlationId,
      releaseTarget: plan.releaseTarget,
      candidateRunId: plan.candidateRunId,
      qualificationRunId: plan.qualificationRunId,
    });
  }

  const workflowFile = plan.dispatchWorkflow;
  const governance = requireImmutableReleaseGovernance(gateway);
  let inputs = { ...plan.dispatchInputs };
  const declared = WORKFLOW_DISPATCH_INPUTS.get(workflowFile) ?? [];
  if (declared.includes('assets_immutable_releases_enabled')) {
    // Carried from the live read above, so a governance regression can never
    // be papered over by a literal the dispatcher writes about itself.
    inputs.assets_immutable_releases_enabled = governance.enabled === true ? 'true' : 'false';
  }
  if (declared.includes('publish_approved')) {
    // In automatic mode the established solo-maintainer environment policy
    // is the approval boundary. Never let the pure planner assert it.
    requirePublicationEnvironment(gateway);
    inputs.publish_approved = 'true';
  }
  inputs = requireExactDispatchInputs(workflowFile, inputs);

  // Guard on the pipeline, not only on this plan's run name: a stale listing
  // plus a fresh binding (another tag, rebuild or source commit) renders a
  // different name, which the named-run check below would not catch.
  const stale = guardStageCorrelation(gateway, plan, {
    defaultBranch,
    workflowPath,
    createdSince: createdSince ?? plan.provenance.nativeReleasePublishedAt,
    satisfiedCorrelationIds,
  });
  if (stale !== null) return stale;

  if (dryRun) return withDispatchRecord(plan, { ref: defaultBranch, inputs });

  const namedRun = () => findNamedRun(gateway, {
    workflowFile,
    workflowPath,
    defaultBranch,
    createdSince: plan.provenance.nativeReleasePublishedAt,
    runName,
  });
  if (namedRun() !== null) {
    throw new ContractError(`a run named ${pyRepr(runName)} already exists; refusing to dispatch a duplicate`);
  }

  gateway.dispatchWorkflow({ workflowFile, ref: defaultBranch, inputs });

  let dispatchedRunId = null;
  for (let attempt = 0; attempt < DISPATCH_READBACK_ATTEMPTS; attempt += 1) {
    if (attempt) gateway.sleep(DISPATCH_READBACK_DELAY_SECONDS);
    dispatchedRunId = namedRun();
    if (dispatchedRunId !== null) break;
  }
  if (dispatchedRunId === null) {
    throw new ContractError(
      `dispatched ${workflowFile} but no run named ${pyRepr(runName)} appeared; `
      + 'the pipeline state is unknown',
    );
  }
  return withDispatchRecord(plan, { ref: defaultBranch, inputs, dispatchedRunId });
}

// _with_dispatch_record: record the exact ref and inputs a dispatch used, or
// would have used.
export function withDispatchRecord(plan, { ref, inputs, dispatchedRunId = null }) {
  return new OrchestrationPlan({
    action: plan.action,
    reason: plan.reason,
    provenance: plan.provenance,
    correlationId: plan.correlationId,
    releaseTarget: plan.releaseTarget,
    candidateRunId: plan.candidateRunId,
    qualificationRunId: plan.qualificationRunId,
    dispatchWorkflow: plan.dispatchWorkflow,
    dispatchRef: ref,
    dispatchRunName: plan.dispatchRunName,
    dispatchInputs: { ...inputs },
    dispatchedRunId,
  });
}
