// Live driver of the stable release orchestrator, the Node port of
// scripts/release_orchestrator_driver.py.
//
// Proves repository governance, gathers the evidence one pipeline already
// has, plans its single next transition, and dispatches it with a
// deterministic run-name readback.

import {
  ASSETS_REPOSITORY, BRIDGE_REPOSITORY, Channel, ContractError, parseReleaseTag, validateImmutableReleaseGovernance,
  validatePublicationEnvironment,
} from '../contract.mjs';
import {
  isDict, pyGet, pyRepr, pyStr,
} from '../json.mjs';
import { candidatePublicationDigests } from '../publication_state.mjs';
import { pyCompareIntTuples } from '../python_compat.mjs';
import {
  fetchAssetReleases, fetchAssetTagNames, findCorrelatedRelease, latestAlignedRelease, latestPublishedNativeAlignment,
  publicationBytesIdentical, verifyPublishedRelease, workflowHistorySince,
} from './asset_releases.mjs';
import {
  CANDIDATE_WORKFLOW_FILE, CANDIDATE_WORKFLOW_PATH, OrchestrationAction, OrchestrationPlan, PUBLISH_WORKFLOW_FILE,
  PUBLISH_WORKFLOW_PATH, PipelineBinding, PipelineObservation, QUALIFICATION_WORKFLOW_FILE, QUALIFICATION_WORKFLOW_PATH,
  REPOSITORY_OWNER, requireStableProvenance, requireStr,
} from './model.mjs';
import { nativeReleaseOrder } from './native.mjs';
import { WORKFLOW_DISPATCH_INPUTS, planPipeline, requireExactDispatchInputs } from './planner.mjs';
import {
  claimedReleaseTags, claimedRebuildsOf, hasOtherBuildClaim, selectNextReleaseTarget,
} from './release_tags.mjs';
import { computeCorrelationId, publishRunName, qualificationRunName } from './run_names.mjs';
import { verifyCandidateRun, verifyQualificationRun } from './stage_proofs.mjs';
import {
  candidateMatcher, fetchRuns, findNamedRun, resolveCandidateBinding, selectPipelineRuns,
} from './workflow_runs.mjs';

// isinstance(value, Mapping) for a JSON value.
function isMapping(value) {
  return isDict(value) || value instanceof Map;
}

export const DISPATCH_READBACK_ATTEMPTS = 12;
export const DISPATCH_READBACK_DELAY_SECONDS = 5.0;

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

// Advance one pipeline by at most one stage. `workspace` is a path string;
// reservedReleaseTags and satisfiedCorrelationIds are any iterables.
export function advancePipeline(gateway, {
  provenance,
  workspace,
  dryRun = false,
  reservedReleaseTags = [],
  satisfiedCorrelationIds = [],
  publicationAllowed = true,
  publicationBarrierNativeTag = null,
  newerNativeScanned = false,
}) {
  requireStableProvenance(provenance);
  if (!publicationAllowed) {
    if (typeof publicationBarrierNativeTag !== 'string') {
      throw new ContractError('a disabled publication transition requires the earlier native tag');
    }
    const barrier = parseReleaseTag(publicationBarrierNativeTag);
    if (barrier.channel !== Channel.STABLE) throw new ContractError('publication barrier native tag must be stable');
  }
  const correlationId = computeCorrelationId(provenance);
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
  const assetTagNames = fetchAssetTagNames(gateway);
  const candidateRuns = fetchRuns(gateway, {
    workflowFile: CANDIDATE_WORKFLOW_FILE,
    workflowPath: CANDIDATE_WORKFLOW_PATH,
    defaultBranch,
    createdSince: runHistorySince,
  });
  const candidateSelection = selectPipelineRuns(candidateRuns, {
    label: 'candidate',
    matcher: candidateMatcher(correlationId),
    defaultBranch,
  });
  const persisted = resolveCandidateBinding(candidateSelection, correlationId);

  const stillClaimedTags = () => {
    const downstreamWorkflows = hasOtherBuildClaim(candidateRuns, { bridgeBuildSha: provenance.bridgeBuildSha })
      ? [[QUALIFICATION_WORKFLOW_FILE, QUALIFICATION_WORKFLOW_PATH], [PUBLISH_WORKFLOW_FILE, PUBLISH_WORKFLOW_PATH]]
      : [];
    const downstreamRuns = downstreamWorkflows.flatMap(([workflowFile, workflowPath]) => fetchRuns(gateway, {
      workflowFile,
      workflowPath,
      defaultBranch,
      createdSince: runHistorySince,
    }));
    return claimedReleaseTags(candidateRuns, {
      bridgeBuildSha: provenance.bridgeBuildSha,
      downstreamRuns,
      satisfiedCorrelationIds,
    });
  };

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
    return executeDispatch(gateway, plan, { defaultBranch, workflowPath: CANDIDATE_WORKFLOW_PATH, dryRun });
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

  const qualificationRuns = fetchRuns(gateway, {
    workflowFile: QUALIFICATION_WORKFLOW_FILE,
    workflowPath: QUALIFICATION_WORKFLOW_PATH,
    defaultBranch,
    createdSince: runHistorySince,
  });
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
      const satisfied = satisfiedPlan(null);
      if (satisfied !== null) return satisfied;
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
    return executeDispatch(gateway, plan, { defaultBranch, workflowPath: QUALIFICATION_WORKFLOW_PATH, dryRun });
  }

  verifyQualificationRun(gateway, {
    runId: qualificationSelection.succeededRunId,
    candidate,
    provenance,
    correlationId,
    defaultBranch,
    workspace,
  });

  const publishRuns = fetchRuns(gateway, {
    workflowFile: PUBLISH_WORKFLOW_FILE,
    workflowPath: PUBLISH_WORKFLOW_PATH,
    defaultBranch,
    createdSince: runHistorySince,
  });
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
  if (publishSelection.inFlightRunId === null && publishSelection.succeededRunId === null) {
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
  return executeDispatch(gateway, plan, { defaultBranch, workflowPath: PUBLISH_WORKFLOW_PATH, dryRun });
}

// _execute_dispatch.
export function executeDispatch(gateway, plan, { defaultBranch, workflowPath, dryRun }) {
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
