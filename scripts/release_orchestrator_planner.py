"""Pure state machine of the stable release orchestrator.

Holds each stage workflow's exact declared dispatch inputs and decides the
single next transition from already-proven evidence, without live access.
"""

from __future__ import annotations

from typing import Any, Mapping

from release_contract import ASSETS_REPOSITORY, ContractError, require_correlation_id
from release_orchestrator_model import (
    CANDIDATE_WORKFLOW_FILE,
    NativeProvenance,
    OrchestrationAction,
    OrchestrationPlan,
    PUBLISH_WORKFLOW_FILE,
    PipelineBinding,
    PipelineObservation,
    QUALIFICATION_WORKFLOW_FILE,
)
from release_orchestrator_run_names import (
    candidate_run_name,
    publish_run_name,
    qualification_run_name,
)


CANDIDATE_DISPATCH_INPUTS = (
    "orchestrator_correlation_id",
    "bridge_source_sha",
    "upstream_tag",
    "upstream_commit",
    "native_release_tag",
    "native_manifest_sha256",
    "release_tag",
    "release_rebuild",
    "assets_immutable_releases_enabled",
)

QUALIFICATION_DISPATCH_INPUTS = (
    "orchestrator_correlation_id",
    "candidate_run_id",
)

PUBLISH_DISPATCH_INPUTS = (
    "orchestrator_correlation_id",
    "bridge_source_sha",
    "upstream_tag",
    "upstream_commit",
    "native_release_tag",
    "native_manifest_sha256",
    "release_tag",
    "release_rebuild",
    "assets_repo",
    "publish_approved",
    "candidate_run_id",
    "qualification_run_id",
)

# Every dispatch is checked against these before it leaves the process. GitHub
# rejects a dispatch whose inputs drift from the workflow's declared set, and it
# does so only at dispatch time, in production; the contract suite asserts these
# tuples are byte-for-byte the workflows' declared inputs.
WORKFLOW_DISPATCH_INPUTS: Mapping[str, tuple[str, ...]] = {
    CANDIDATE_WORKFLOW_FILE: CANDIDATE_DISPATCH_INPUTS,
    QUALIFICATION_WORKFLOW_FILE: QUALIFICATION_DISPATCH_INPUTS,
    PUBLISH_WORKFLOW_FILE: PUBLISH_DISPATCH_INPUTS,
}


def require_exact_dispatch_inputs(
    workflow_file: str, inputs: Mapping[str, Any]
) -> dict[str, str]:
    """Refuse to dispatch anything but the workflow's exact declared input set."""
    expected = WORKFLOW_DISPATCH_INPUTS.get(workflow_file)
    if expected is None:
        raise ContractError(
            f"{workflow_file} is not a workflow this orchestrator may dispatch"
        )
    if set(inputs) != set(expected):
        missing = sorted(set(expected) - set(inputs))
        unexpected = sorted(set(inputs) - set(expected))
        raise ContractError(
            f"dispatch inputs for {workflow_file} are not the exact declared set "
            f"(missing: {missing}, unexpected: {unexpected})"
        )
    ordered: dict[str, str] = {}
    for name in expected:
        value = inputs[name]
        if not isinstance(value, str) or not value:
            raise ContractError(
                f"dispatch input {name!r} for {workflow_file} must be a non-empty "
                f"string, got {value!r}"
            )
        ordered[name] = value
    return ordered


def _dispatch_inputs_for_candidate(
    provenance: NativeProvenance, correlation_id: str, binding: PipelineBinding
) -> dict[str, str]:
    """Build every candidate input the planner can know without live proof.

    ``assets_immutable_releases_enabled`` is deliberately absent: it is an
    assertion about the assets repository, so it is filled in from the live
    governance read taken immediately before dispatch, never from a constant.
    """
    return {
        "orchestrator_correlation_id": correlation_id,
        "bridge_source_sha": binding.bridge_source_sha,
        "upstream_tag": provenance.upstream_tag,
        "upstream_commit": provenance.upstream_commit,
        "native_release_tag": provenance.native_release_tag,
        "native_manifest_sha256": provenance.native_manifest_sha256,
        "release_tag": binding.release_tag,
        "release_rebuild": str(binding.release_rebuild),
    }


def _dispatch_inputs_for_qualification(
    correlation_id: str, candidate_run_id: str
) -> dict[str, str]:
    return {
        "orchestrator_correlation_id": correlation_id,
        "candidate_run_id": candidate_run_id,
    }


def _dispatch_inputs_for_publish(
    provenance: NativeProvenance,
    correlation_id: str,
    binding: PipelineBinding,
    candidate_run_id: str,
    qualification_run_id: str,
) -> dict[str, str]:
    return {
        "orchestrator_correlation_id": correlation_id,
        "bridge_source_sha": binding.bridge_source_sha,
        "upstream_tag": provenance.upstream_tag,
        "upstream_commit": provenance.upstream_commit,
        "native_release_tag": provenance.native_release_tag,
        "native_manifest_sha256": provenance.native_manifest_sha256,
        "release_tag": binding.release_tag,
        "release_rebuild": str(binding.release_rebuild),
        "assets_repo": ASSETS_REPOSITORY,
        "candidate_run_id": candidate_run_id,
        "qualification_run_id": qualification_run_id,
    }


def plan_pipeline(
    *,
    provenance: NativeProvenance,
    correlation_id: str,
    observation: PipelineObservation,
) -> OrchestrationPlan:
    """Decide the single next transition from already-proven evidence."""
    require_correlation_id(correlation_id)

    if observation.published is not None:
        published = observation.published
        return OrchestrationPlan(
            action=OrchestrationAction.NOOP,
            reason=(
                f"this provenance is already published as the immutable release "
                f"{published.release_target.release_tag} (id {published.release_id}, "
                f"published {published.published_at})"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=published.release_target,
        )

    if observation.candidate_in_flight_run_id is not None:
        return OrchestrationPlan(
            action=OrchestrationAction.IN_FLIGHT,
            reason=(
                f"candidate run {observation.candidate_in_flight_run_id} is still "
                "running; refusing to dispatch a duplicate"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=observation.binding.release_target
            if observation.binding
            else None,
            in_flight_workflow=CANDIDATE_WORKFLOW_FILE,
            in_flight_run_id=observation.candidate_in_flight_run_id,
        )

    if observation.candidate_run_id is None:
        binding = observation.fresh_binding
        if binding is None:
            raise ContractError("no pipeline binding is available for candidate dispatch")
        return OrchestrationPlan(
            action=OrchestrationAction.DISPATCH_CANDIDATE,
            reason=(
                "new stable native provenance requires exactly one candidate "
                f"build for {binding.release_tag}"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=binding.release_target,
            dispatch_workflow=CANDIDATE_WORKFLOW_FILE,
            dispatch_run_name=candidate_run_name(correlation_id, binding),
            dispatch_inputs=_dispatch_inputs_for_candidate(
                provenance, correlation_id, binding
            ),
        )

    binding = observation.binding
    if binding is None:
        raise ContractError(
            "a successful candidate run must carry its proven pipeline binding"
        )

    if observation.satisfied_by is not None:
        satisfied_by = observation.satisfied_by
        return OrchestrationPlan(
            action=OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE,
            reason=(
                f"candidate run {observation.candidate_run_id} bound to "
                f"{binding.release_tag} (rebuild {binding.release_rebuild}) has "
                "publication files other than manifest.json byte-identical to the "
                f"immutable release {satisfied_by.release_target.release_tag} (id "
                f"{satisfied_by.release_id}, published {satisfied_by.published_at}) "
                f"for native {provenance.native_release_tag}; nothing further is "
                "dispatched"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=satisfied_by.release_target,
            candidate_run_id=observation.candidate_run_id,
            qualification_run_id=observation.qualification_run_id,
        )

    if observation.qualification_in_flight_run_id is not None:
        return OrchestrationPlan(
            action=OrchestrationAction.IN_FLIGHT,
            reason=(
                f"qualification run {observation.qualification_in_flight_run_id} "
                "is still running"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=binding.release_target,
            candidate_run_id=observation.candidate_run_id,
            in_flight_workflow=QUALIFICATION_WORKFLOW_FILE,
            in_flight_run_id=observation.qualification_in_flight_run_id,
        )

    if observation.qualification_run_id is None:
        return OrchestrationPlan(
            action=OrchestrationAction.DISPATCH_QUALIFICATION,
            reason=(
                f"candidate run {observation.candidate_run_id} is built and proven; "
                "dispatching exactly one hosted qualification run for the heavy "
                "real-model gates"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=binding.release_target,
            candidate_run_id=observation.candidate_run_id,
            dispatch_workflow=QUALIFICATION_WORKFLOW_FILE,
            dispatch_run_name=qualification_run_name(
                correlation_id, observation.candidate_run_id
            ),
            dispatch_inputs=_dispatch_inputs_for_qualification(
                correlation_id, observation.candidate_run_id
            ),
        )

    if observation.publish_in_flight_run_id is not None:
        return OrchestrationPlan(
            action=OrchestrationAction.IN_FLIGHT,
            reason=(
                f"publication run {observation.publish_in_flight_run_id} is still running"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=binding.release_target,
            candidate_run_id=observation.candidate_run_id,
            qualification_run_id=observation.qualification_run_id,
            in_flight_workflow=PUBLISH_WORKFLOW_FILE,
            in_flight_run_id=observation.publish_in_flight_run_id,
        )

    if observation.publish_succeeded_run_id is not None:
        raise ContractError(
            f"publication run {observation.publish_succeeded_run_id} succeeded but no "
            f"immutable release for correlation {correlation_id!r} could be verified"
        )

    return OrchestrationPlan(
        action=OrchestrationAction.DISPATCH_PUBLISH,
        reason=(
            f"{'retrying publication of' if observation.publish_retry else 'publishing'}"
            f" {binding.release_tag} from the proven candidate "
            f"{observation.candidate_run_id} and qualification "
            f"{observation.qualification_run_id}"
        ),
        provenance=provenance,
        correlation_id=correlation_id,
        release_target=binding.release_target,
        candidate_run_id=observation.candidate_run_id,
        qualification_run_id=observation.qualification_run_id,
        dispatch_workflow=PUBLISH_WORKFLOW_FILE,
        dispatch_run_name=publish_run_name(
            correlation_id,
            observation.candidate_run_id,
            observation.qualification_run_id,
            binding,
        ),
        dispatch_inputs=_dispatch_inputs_for_publish(
            provenance,
            correlation_id,
            binding,
            observation.candidate_run_id,
            observation.qualification_run_id,
        ),
    )
