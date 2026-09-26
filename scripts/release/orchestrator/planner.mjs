// Pure state machine of the stable release orchestrator, the Node port of
// scripts/release_orchestrator_planner.py.
//
// Holds each stage workflow's exact declared dispatch inputs and decides the
// single next transition from already-proven evidence, without live access.

import { ASSETS_REPOSITORY, ContractError, requireCorrelationId } from '../contract.mjs';
import {
  compareCodePoints, isDict, pyKeys, pyRepr, pyStr,
} from '../json.mjs';
import {
  CANDIDATE_WORKFLOW_FILE,
  OrchestrationAction,
  OrchestrationPlan,
  PUBLISH_WORKFLOW_FILE,
  QUALIFICATION_WORKFLOW_FILE,
} from './model.mjs';
import { candidateRunName, publishRunName, qualificationRunName } from './run_names.mjs';

export const CANDIDATE_DISPATCH_INPUTS = Object.freeze([
  'orchestrator_correlation_id',
  'bridge_source_sha',
  'upstream_tag',
  'upstream_commit',
  'native_release_tag',
  'native_manifest_sha256',
  'release_tag',
  'release_rebuild',
  'assets_immutable_releases_enabled',
]);

export const QUALIFICATION_DISPATCH_INPUTS = Object.freeze([
  'orchestrator_correlation_id',
  'candidate_run_id',
]);

export const PUBLISH_DISPATCH_INPUTS = Object.freeze([
  'orchestrator_correlation_id',
  'bridge_source_sha',
  'upstream_tag',
  'upstream_commit',
  'native_release_tag',
  'native_manifest_sha256',
  'release_tag',
  'release_rebuild',
  'assets_repo',
  'publish_approved',
  'candidate_run_id',
  'qualification_run_id',
]);

// Every dispatch is checked against these before it leaves the process. GitHub
// rejects a dispatch whose inputs drift from the workflow's declared set, and it
// does so only at dispatch time, in production; the contract suite asserts these
// lists are byte-for-byte the workflows' declared inputs.
export const WORKFLOW_DISPATCH_INPUTS = Object.freeze(new Map([
  [CANDIDATE_WORKFLOW_FILE, CANDIDATE_DISPATCH_INPUTS],
  [QUALIFICATION_WORKFLOW_FILE, QUALIFICATION_DISPATCH_INPUTS],
  [PUBLISH_WORKFLOW_FILE, PUBLISH_DISPATCH_INPUTS],
]));

// Refuse to dispatch anything but the workflow's exact declared input set.
// `inputs` is a dict (plain object or Map); the result is a new object in the
// declared order.
export function requireExactDispatchInputs(workflowFile, inputs) {
  const expected = WORKFLOW_DISPATCH_INPUTS.get(workflowFile);
  if (expected === undefined) throw new ContractError(`${workflowFile} is not a workflow this orchestrator may dispatch`);
  const actual = new Set(inputs instanceof Map || isDict(inputs) ? pyKeys(inputs) : inputs);
  const wanted = new Set(expected);
  const sameSet = actual.size === wanted.size && [...actual].every((name) => wanted.has(name));
  if (!sameSet) {
    const missing = [...wanted].filter((name) => !actual.has(name)).sort(compareCodePoints);
    const unexpected = [...actual].filter((name) => !wanted.has(name)).sort(compareCodePoints);
    throw new ContractError(
      `dispatch inputs for ${workflowFile} are not the exact declared set `
      + `(missing: ${pyRepr(missing)}, unexpected: ${pyRepr(unexpected)})`,
    );
  }
  const ordered = {};
  for (const name of expected) {
    const value = inputs instanceof Map ? inputs.get(name) : inputs[name];
    if (typeof value !== 'string' || !value) {
      throw new ContractError(`dispatch input ${pyRepr(name)} for ${workflowFile} must be a non-empty string, got ${pyRepr(value)}`);
    }
    ordered[name] = value;
  }
  return ordered;
}

// _dispatch_inputs_for_candidate: every candidate input the planner can know
// without live proof.
//
// assets_immutable_releases_enabled is deliberately absent: it is an assertion
// about the assets repository, so it is filled in from the live governance
// read taken immediately before dispatch, never from a constant.
export function dispatchInputsForCandidate(provenance, correlationId, binding) {
  return {
    orchestrator_correlation_id: correlationId,
    bridge_source_sha: binding.bridgeSourceSha,
    upstream_tag: provenance.upstreamTag,
    upstream_commit: provenance.upstreamCommit,
    native_release_tag: provenance.nativeReleaseTag,
    native_manifest_sha256: provenance.nativeManifestSha256,
    release_tag: binding.releaseTag,
    release_rebuild: pyStr(binding.releaseRebuild),
  };
}

// _dispatch_inputs_for_qualification.
export function dispatchInputsForQualification(correlationId, candidateRunId) {
  return {
    orchestrator_correlation_id: correlationId,
    candidate_run_id: candidateRunId,
  };
}

// _dispatch_inputs_for_publish.
export function dispatchInputsForPublish(provenance, correlationId, binding, candidateRunId, qualificationRunId) {
  return {
    orchestrator_correlation_id: correlationId,
    bridge_source_sha: binding.bridgeSourceSha,
    upstream_tag: provenance.upstreamTag,
    upstream_commit: provenance.upstreamCommit,
    native_release_tag: provenance.nativeReleaseTag,
    native_manifest_sha256: provenance.nativeManifestSha256,
    release_tag: binding.releaseTag,
    release_rebuild: pyStr(binding.releaseRebuild),
    assets_repo: ASSETS_REPOSITORY,
    candidate_run_id: candidateRunId,
    qualification_run_id: qualificationRunId,
  };
}

// Decide the single next transition from already-proven evidence.
export function planPipeline({ provenance, correlationId, observation }) {
  requireCorrelationId(correlationId);

  if (observation.published !== null) {
    const published = observation.published;
    return new OrchestrationPlan({
      action: OrchestrationAction.NOOP,
      reason: (
        'this provenance is already published as the immutable release '
        + `${published.releaseTarget.releaseTag} (id ${pyStr(published.releaseId)}, `
        + `published ${published.publishedAt})`
      ),
      provenance,
      correlationId,
      releaseTarget: published.releaseTarget,
    });
  }

  if (observation.candidateInFlightRunId !== null) {
    return new OrchestrationPlan({
      action: OrchestrationAction.IN_FLIGHT,
      reason: `candidate run ${observation.candidateInFlightRunId} is still running; refusing to dispatch a duplicate`,
      provenance,
      correlationId,
      releaseTarget: observation.binding ? observation.binding.releaseTarget : null,
      inFlightWorkflow: CANDIDATE_WORKFLOW_FILE,
      inFlightRunId: observation.candidateInFlightRunId,
    });
  }

  if (observation.candidateRunId === null) {
    const binding = observation.freshBinding;
    if (binding === null) throw new ContractError('no pipeline binding is available for candidate dispatch');
    return new OrchestrationPlan({
      action: OrchestrationAction.DISPATCH_CANDIDATE,
      reason: `new stable native provenance requires exactly one candidate build for ${binding.releaseTag}`,
      provenance,
      correlationId,
      releaseTarget: binding.releaseTarget,
      dispatchWorkflow: CANDIDATE_WORKFLOW_FILE,
      dispatchRunName: candidateRunName(correlationId, binding),
      dispatchInputs: dispatchInputsForCandidate(provenance, correlationId, binding),
    });
  }

  const binding = observation.binding;
  if (binding === null) throw new ContractError('a successful candidate run must carry its proven pipeline binding');

  if (observation.satisfiedBy !== null) {
    const satisfiedBy = observation.satisfiedBy;
    return new OrchestrationPlan({
      action: OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE,
      reason: (
        `candidate run ${observation.candidateRunId} bound to `
        + `${binding.releaseTag} (rebuild ${pyStr(binding.releaseRebuild)}) has `
        + 'publication files other than manifest.json byte-identical to the '
        + `immutable release ${satisfiedBy.releaseTarget.releaseTag} (id `
        + `${pyStr(satisfiedBy.releaseId)}, published ${satisfiedBy.publishedAt}) `
        + `for native ${provenance.nativeReleaseTag}; nothing further is `
        + 'dispatched'
      ),
      provenance,
      correlationId,
      releaseTarget: satisfiedBy.releaseTarget,
      candidateRunId: observation.candidateRunId,
      qualificationRunId: observation.qualificationRunId,
    });
  }

  if (observation.qualificationInFlightRunId !== null) {
    return new OrchestrationPlan({
      action: OrchestrationAction.IN_FLIGHT,
      reason: `qualification run ${observation.qualificationInFlightRunId} is still running`,
      provenance,
      correlationId,
      releaseTarget: binding.releaseTarget,
      candidateRunId: observation.candidateRunId,
      inFlightWorkflow: QUALIFICATION_WORKFLOW_FILE,
      inFlightRunId: observation.qualificationInFlightRunId,
    });
  }

  if (observation.qualificationRunId === null) {
    return new OrchestrationPlan({
      action: OrchestrationAction.DISPATCH_QUALIFICATION,
      reason: (
        `candidate run ${observation.candidateRunId} is built and proven; `
        + 'dispatching exactly one hosted qualification run for the heavy '
        + 'real-model gates'
      ),
      provenance,
      correlationId,
      releaseTarget: binding.releaseTarget,
      candidateRunId: observation.candidateRunId,
      dispatchWorkflow: QUALIFICATION_WORKFLOW_FILE,
      dispatchRunName: qualificationRunName(correlationId, observation.candidateRunId),
      dispatchInputs: dispatchInputsForQualification(correlationId, observation.candidateRunId),
    });
  }

  if (observation.publishInFlightRunId !== null) {
    return new OrchestrationPlan({
      action: OrchestrationAction.IN_FLIGHT,
      reason: `publication run ${observation.publishInFlightRunId} is still running`,
      provenance,
      correlationId,
      releaseTarget: binding.releaseTarget,
      candidateRunId: observation.candidateRunId,
      qualificationRunId: observation.qualificationRunId,
      inFlightWorkflow: PUBLISH_WORKFLOW_FILE,
      inFlightRunId: observation.publishInFlightRunId,
    });
  }

  if (observation.publishSucceededRunId !== null) {
    throw new ContractError(
      `publication run ${observation.publishSucceededRunId} succeeded but no `
      + `immutable release for correlation ${pyRepr(correlationId)} could be verified`,
    );
  }

  return new OrchestrationPlan({
    action: OrchestrationAction.DISPATCH_PUBLISH,
    reason: (
      `${observation.publishRetry ? 'retrying publication of' : 'publishing'}`
      + ` ${binding.releaseTag} from the proven candidate `
      + `${observation.candidateRunId} and qualification `
      + `${observation.qualificationRunId}`
    ),
    provenance,
    correlationId,
    releaseTarget: binding.releaseTarget,
    candidateRunId: observation.candidateRunId,
    qualificationRunId: observation.qualificationRunId,
    dispatchWorkflow: PUBLISH_WORKFLOW_FILE,
    dispatchRunName: publishRunName(correlationId, observation.candidateRunId, observation.qualificationRunId, binding),
    dispatchInputs: dispatchInputsForPublish(
      provenance,
      correlationId,
      binding,
      observation.candidateRunId,
      observation.qualificationRunId,
    ),
  });
}
