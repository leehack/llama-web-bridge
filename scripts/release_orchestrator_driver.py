"""Live driver of the stable release orchestrator.

Proves repository governance, gathers the evidence one pipeline already has,
plans its single next transition, and dispatches it with a deterministic
run-name readback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from release_contract import (
    ASSETS_REPOSITORY,
    BRIDGE_REPOSITORY,
    Channel,
    ContractError,
    parse_release_tag,
    validate_immutable_release_governance,
    validate_publication_environment,
)
from release_publication_state import candidate_publication_digests
from release_orchestrator_asset_releases import (
    fetch_asset_releases,
    fetch_asset_tag_names,
    find_correlated_release,
    latest_aligned_release,
    latest_published_native_alignment,
    publication_bytes_identical,
    verify_published_release,
    workflow_history_since,
)
from release_orchestrator_model import (
    CANDIDATE_WORKFLOW_FILE,
    CANDIDATE_WORKFLOW_PATH,
    NativeProvenance,
    OrchestrationAction,
    OrchestrationPlan,
    PUBLISH_WORKFLOW_FILE,
    PUBLISH_WORKFLOW_PATH,
    PipelineBinding,
    PipelineObservation,
    QUALIFICATION_WORKFLOW_FILE,
    QUALIFICATION_WORKFLOW_PATH,
    REPOSITORY_OWNER,
    _require_str,
    require_stable_provenance,
)
from release_orchestrator_native import _native_release_order
from release_orchestrator_planner import (
    WORKFLOW_DISPATCH_INPUTS,
    plan_pipeline,
    require_exact_dispatch_inputs,
)
from release_orchestrator_release_tags import (
    _claimed_rebuilds_of,
    claimed_release_tags,
    has_other_build_claim,
    select_next_release_target,
)
from release_orchestrator_run_names import (
    compute_correlation_id,
    publish_run_name,
    qualification_run_name,
)
from release_orchestrator_stage_proofs import (
    verify_candidate_run,
    verify_qualification_run,
)
from release_orchestrator_transport import Gateway
from release_orchestrator_workflow_runs import (
    _candidate_matcher,
    _fetch_runs,
    _find_named_run,
    _resolve_candidate_binding,
    select_pipeline_runs,
)


DISPATCH_READBACK_ATTEMPTS = 12
DISPATCH_READBACK_DELAY_SECONDS = 5.0


def require_default_branch(gateway: Gateway) -> str:
    payload = gateway.api_json(f"repos/{BRIDGE_REPOSITORY}")
    if not isinstance(payload, Mapping):
        raise ContractError("repository response must be a JSON object")
    return _require_str(payload.get("default_branch"), "default_branch")


def require_immutable_release_governance(gateway: Gateway) -> dict[str, Any]:
    """Prove immutable-release governance live, never by constant assertion."""
    payload = gateway.api_json(
        f"repos/{ASSETS_REPOSITORY}/immutable-releases", privileged=True
    )
    return validate_immutable_release_governance(payload, ASSETS_REPOSITORY)


def require_publication_environment(gateway: Gateway) -> None:
    """Prove the existing environment policy before asserting publication approval."""
    environment = gateway.api_json(
        f"repos/{BRIDGE_REPOSITORY}/environments/bridge-assets-publication"
    )
    branch_policies = gateway.api_json(
        f"repos/{BRIDGE_REPOSITORY}/environments/bridge-assets-publication/"
        "deployment-branch-policies"
    )
    if not isinstance(environment, Mapping) or not isinstance(
        branch_policies, Mapping
    ):
        raise ContractError("publication environment responses must be JSON objects")
    validate_publication_environment(environment, branch_policies)


def advance_pipeline(
    gateway: Gateway,
    *,
    provenance: NativeProvenance,
    workspace: Path,
    dry_run: bool = False,
    reserved_release_tags: Sequence[str] | set[str] = (),
    satisfied_correlation_ids: Sequence[str] | set[str] = (),
    publication_allowed: bool = True,
    publication_barrier_native_tag: str | None = None,
    newer_native_scanned: bool = False,
) -> OrchestrationPlan:
    require_stable_provenance(provenance)
    if not publication_allowed:
        if not isinstance(publication_barrier_native_tag, str):
            raise ContractError(
                "a disabled publication transition requires the earlier native tag"
            )
        barrier = parse_release_tag(publication_barrier_native_tag)
        if barrier.channel is not Channel.STABLE:
            raise ContractError("publication barrier native tag must be stable")
    correlation_id = compute_correlation_id(provenance)
    default_branch = require_default_branch(gateway)
    # Governance is a prerequisite for every state classification, including a
    # noop: a disabled or unreadable policy is never reported as healthy stable
    # automation merely because an older release happens to exist.
    require_immutable_release_governance(gateway)

    releases = fetch_asset_releases(gateway)
    correlated = find_correlated_release(releases, correlation_id, provenance)
    published = None
    if correlated is not None:
        published = verify_published_release(
            gateway,
            release=correlated,
            provenance=provenance,
            correlation_id=correlation_id,
            workspace=workspace,
        )
        if not publication_allowed:
            return OrchestrationPlan(
                action=OrchestrationAction.BLOCKED,
                reason=(
                    f"{provenance.native_release_tag} is already published while "
                    f"earlier native release {publication_barrier_native_tag} is not; "
                    "the ordered publication history is inconsistent"
                ),
                provenance=provenance,
                correlation_id=correlation_id,
                release_target=published.release_target,
            )
        return plan_pipeline(
            provenance=provenance,
            correlation_id=correlation_id,
            observation=PipelineObservation(published=published),
        )

    alignment = latest_published_native_alignment(releases)
    if alignment is not None and _native_release_order(
        provenance.native_release_tag
    ) < _native_release_order(alignment[0]):
        aligned_native_tag, aligned_asset_tag = alignment
        if not newer_native_scanned:
            return OrchestrationPlan(
                action=OrchestrationAction.BLOCKED,
                reason=(
                    f"asset release {aligned_asset_tag} records native release "
                    f"{aligned_native_tag}, newer than {provenance.native_release_tag}, "
                    "but no newer stable native release is part of this scan; "
                    "refusing to skip the newest scanned native release"
                ),
                provenance=provenance,
                correlation_id=correlation_id,
            )
        return OrchestrationPlan(
            action=OrchestrationAction.SUPERSEDED,
            reason=(
                f"{provenance.native_release_tag} is behind native release "
                f"{aligned_native_tag}, already published as {aligned_asset_tag}; "
                "only native releases at or ahead of the published alignment "
                "are advanced for a new build identity"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
        )

    run_history_since = workflow_history_since(releases, provenance)
    asset_tag_names = fetch_asset_tag_names(gateway)
    candidate_runs = _fetch_runs(
        gateway,
        workflow_file=CANDIDATE_WORKFLOW_FILE,
        workflow_path=CANDIDATE_WORKFLOW_PATH,
        default_branch=default_branch,
        created_since=run_history_since,
    )
    candidate_selection = select_pipeline_runs(
        candidate_runs,
        label="candidate",
        matcher=_candidate_matcher(correlation_id),
        default_branch=default_branch,
    )
    persisted = _resolve_candidate_binding(candidate_selection, correlation_id)

    def still_claimed_tags() -> set[str]:
        downstream_workflows = (
            (
                (QUALIFICATION_WORKFLOW_FILE, QUALIFICATION_WORKFLOW_PATH),
                (PUBLISH_WORKFLOW_FILE, PUBLISH_WORKFLOW_PATH),
            )
            if has_other_build_claim(
                candidate_runs, bridge_build_sha=provenance.bridge_build_sha
            )
            else ()
        )
        downstream_runs = [
            record
            for workflow_file, workflow_path in downstream_workflows
            for record in _fetch_runs(
                gateway,
                workflow_file=workflow_file,
                workflow_path=workflow_path,
                default_branch=default_branch,
                created_since=run_history_since,
            )
        ]
        return claimed_release_tags(
            candidate_runs,
            bridge_build_sha=provenance.bridge_build_sha,
            downstream_runs=downstream_runs,
            satisfied_correlation_ids=satisfied_correlation_ids,
        )

    fresh_binding = None
    if persisted is None:
        target = select_next_release_target(
            [str(release.get("tag_name")) for release in releases],
            upstream_tag=provenance.upstream_tag,
            taken=(
                asset_tag_names | still_claimed_tags() | set(reserved_release_tags)
            ),
        )
        fresh_binding = PipelineBinding(
            bridge_source_sha=provenance.bridge_source_sha,
            release_tag=target.release_tag,
            release_rebuild=target.release_rebuild,
        )

    if candidate_selection.in_flight_run_id is not None:
        return plan_pipeline(
            provenance=provenance,
            correlation_id=correlation_id,
            observation=PipelineObservation(
                binding=persisted,
                fresh_binding=fresh_binding,
                candidate_in_flight_run_id=candidate_selection.in_flight_run_id,
            ),
        )

    if (
        candidate_selection.succeeded_run_id is None
        and candidate_selection.unsuccessful
    ):
        failed_ids = ", ".join(
            record.run_id for record in candidate_selection.unsuccessful
        )
        if persisted is None:
            raise ContractError("failed candidate runs have no exact persisted binding")
        return OrchestrationPlan(
            action=OrchestrationAction.BLOCKED,
            reason=(
                f"candidate run(s) {failed_ids} failed for this exact provenance; "
                "automatic candidate retries are disabled to prevent unbounded "
                "daily duplicates; a maintainer must diagnose the failure and "
                "explicitly dispatch one deliberate new first-attempt run with the "
                "same exact binding, or establish new provenance"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=persisted.release_target,
        )

    if candidate_selection.succeeded_run_id is None:
        observation = PipelineObservation(
            fresh_binding=fresh_binding,
        )
        plan = plan_pipeline(
            provenance=provenance,
            correlation_id=correlation_id,
            observation=observation,
        )
        return _execute_dispatch(
            gateway,
            plan,
            default_branch=default_branch,
            workflow_path=CANDIDATE_WORKFLOW_PATH,
            dry_run=dry_run,
        )

    if persisted is None:
        raise ContractError(
            "a successful candidate run advertises no parsable pipeline binding"
        )
    candidate = verify_candidate_run(
        gateway,
        run_id=candidate_selection.succeeded_run_id,
        provenance=provenance,
        correlation_id=correlation_id,
        binding=persisted,
        default_branch=default_branch,
        workspace=workspace,
    )

    def satisfied_plan(qualification_run_id: str | None) -> OrchestrationPlan | None:
        aligned = latest_aligned_release(releases, provenance)
        if aligned is None or _claimed_rebuilds_of(
            candidate.binding, still_claimed_tags()
        ):
            return None
        aligned_release, aligned_correlation_id = aligned
        published = verify_published_release(
            gateway,
            release=aligned_release,
            provenance=provenance,
            correlation_id=aligned_correlation_id,
            workspace=workspace,
        )
        if published.directory is None:
            raise ContractError("verified release did not retain its downloaded bytes")
        if not publication_bytes_identical(
            candidate_publication_digests(candidate.directory),
            candidate_publication_digests(published.directory),
        ):
            return None
        return plan_pipeline(
            provenance=provenance,
            correlation_id=correlation_id,
            observation=PipelineObservation(
                binding=candidate.binding,
                candidate_run_id=candidate.run_id,
                qualification_run_id=qualification_run_id,
                satisfied_by=published,
            ),
        )

    qualification_runs = _fetch_runs(
        gateway,
        workflow_file=QUALIFICATION_WORKFLOW_FILE,
        workflow_path=QUALIFICATION_WORKFLOW_PATH,
        default_branch=default_branch,
        created_since=run_history_since,
    )
    expected_qualification_name = qualification_run_name(
        correlation_id, candidate.run_id
    )
    qualification_selection = select_pipeline_runs(
        qualification_runs,
        label="qualification",
        matcher=lambda name: name == expected_qualification_name,
        default_branch=default_branch,
    )
    if (
        qualification_selection.succeeded_run_id is None
        and qualification_selection.in_flight_run_id is None
        and qualification_selection.unsuccessful
    ):
        failed_ids = ", ".join(
            record.run_id for record in qualification_selection.unsuccessful
        )
        return OrchestrationPlan(
            action=OrchestrationAction.BLOCKED,
            reason=(
                f"qualification run(s) {failed_ids} failed for candidate "
                f"{candidate.run_id}; automatic qualification retries are disabled "
                "to prevent unbounded duplicates, so a maintainer must diagnose the "
                "heavy-gate failure before this candidate can advance"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=candidate.binding.release_target,
            candidate_run_id=candidate.run_id,
        )

    if qualification_selection.succeeded_run_id is None:
        if qualification_selection.in_flight_run_id is None:
            satisfied = satisfied_plan(None)
            if satisfied is not None:
                return satisfied
        plan = plan_pipeline(
            provenance=provenance,
            correlation_id=correlation_id,
            observation=PipelineObservation(
                binding=candidate.binding,
                candidate_run_id=candidate.run_id,
                qualification_in_flight_run_id=(
                    qualification_selection.in_flight_run_id
                ),
            ),
        )
        return _execute_dispatch(
            gateway,
            plan,
            default_branch=default_branch,
            workflow_path=QUALIFICATION_WORKFLOW_PATH,
            dry_run=dry_run,
        )

    verify_qualification_run(
        gateway,
        run_id=qualification_selection.succeeded_run_id,
        candidate=candidate,
        provenance=provenance,
        correlation_id=correlation_id,
        default_branch=default_branch,
        workspace=workspace,
    )

    publish_runs = _fetch_runs(
        gateway,
        workflow_file=PUBLISH_WORKFLOW_FILE,
        workflow_path=PUBLISH_WORKFLOW_PATH,
        default_branch=default_branch,
        created_since=run_history_since,
    )
    expected_publish_name = publish_run_name(
        correlation_id,
        candidate.run_id,
        qualification_selection.succeeded_run_id,
        candidate.binding,
    )
    publish_selection = select_pipeline_runs(
        publish_runs,
        label="publication",
        matcher=lambda name: name == expected_publish_name,
        default_branch=default_branch,
    )
    if not publication_allowed:
        if (
            publish_selection.in_flight_run_id is not None
            or publish_selection.succeeded_run_id is not None
        ):
            return OrchestrationPlan(
                action=OrchestrationAction.BLOCKED,
                reason=(
                    f"publication for {provenance.native_release_tag} was started "
                    f"before earlier native release {publication_barrier_native_tag} "
                    "was immutably published"
                ),
                provenance=provenance,
                correlation_id=correlation_id,
                release_target=candidate.binding.release_target,
                candidate_run_id=candidate.run_id,
                qualification_run_id=qualification_selection.succeeded_run_id,
                in_flight_workflow=(
                    PUBLISH_WORKFLOW_FILE
                    if publish_selection.in_flight_run_id is not None
                    else None
                ),
                in_flight_run_id=publish_selection.in_flight_run_id,
            )
        return OrchestrationPlan(
            action=OrchestrationAction.WAITING_FOR_PRIOR_PUBLICATION,
            reason=(
                f"candidate {candidate.run_id} and qualification "
                f"{qualification_selection.succeeded_run_id} are proven, but "
                f"earlier native release {publication_barrier_native_tag} must be "
                "immutably published first to preserve monotonic output ordering"
            ),
            provenance=provenance,
            correlation_id=correlation_id,
            release_target=candidate.binding.release_target,
            candidate_run_id=candidate.run_id,
            qualification_run_id=qualification_selection.succeeded_run_id,
        )
    if (
        publish_selection.in_flight_run_id is None
        and publish_selection.succeeded_run_id is None
    ):
        satisfied = satisfied_plan(qualification_selection.succeeded_run_id)
        if satisfied is not None:
            return satisfied
    plan = plan_pipeline(
        provenance=provenance,
        correlation_id=correlation_id,
        observation=PipelineObservation(
            binding=candidate.binding,
            candidate_run_id=candidate.run_id,
            qualification_run_id=qualification_selection.succeeded_run_id,
            publish_in_flight_run_id=publish_selection.in_flight_run_id,
            publish_succeeded_run_id=publish_selection.succeeded_run_id,
            publish_retry=bool(publish_selection.unsuccessful),
        ),
    )
    return _execute_dispatch(
        gateway,
        plan,
        default_branch=default_branch,
        workflow_path=PUBLISH_WORKFLOW_PATH,
        dry_run=dry_run,
    )


def _execute_dispatch(
    gateway: Gateway,
    plan: OrchestrationPlan,
    *,
    default_branch: str,
    workflow_path: str,
    dry_run: bool,
) -> OrchestrationPlan:
    if plan.dispatch_workflow is None or plan.dispatch_inputs is None:
        return plan
    run_name = plan.dispatch_run_name
    if run_name is None:
        raise ContractError("a dispatch plan must carry its deterministic run name")

    identity = gateway.dispatch_identity()
    if identity != REPOSITORY_OWNER:
        return OrchestrationPlan(
            action=OrchestrationAction.BLOCKED,
            reason=(
                "automatic dispatch identity could not be proven as "
                f"{REPOSITORY_OWNER!r} (resolved {identity!r}); the owner-only actor "
                "gates stay in force and nothing was dispatched"
            ),
            provenance=plan.provenance,
            correlation_id=plan.correlation_id,
            release_target=plan.release_target,
            candidate_run_id=plan.candidate_run_id,
            qualification_run_id=plan.qualification_run_id,
        )

    workflow_file = plan.dispatch_workflow
    governance = require_immutable_release_governance(gateway)
    inputs = dict(plan.dispatch_inputs)
    if "assets_immutable_releases_enabled" in WORKFLOW_DISPATCH_INPUTS.get(
        workflow_file, ()
    ):
        # Carried from the live read above, so a governance regression can never
        # be papered over by a literal the dispatcher writes about itself.
        inputs["assets_immutable_releases_enabled"] = (
            "true" if governance["enabled"] is True else "false"
        )
    if "publish_approved" in WORKFLOW_DISPATCH_INPUTS.get(workflow_file, ()):
        # In automatic mode the established solo-maintainer environment policy
        # is the approval boundary. Never let the pure planner assert it.
        require_publication_environment(gateway)
        inputs["publish_approved"] = "true"
    inputs = require_exact_dispatch_inputs(workflow_file, inputs)

    if dry_run:
        return _with_dispatch_record(plan, ref=default_branch, inputs=inputs)

    if _find_named_run(
        gateway,
        workflow_file=workflow_file,
        workflow_path=workflow_path,
        default_branch=default_branch,
        created_since=plan.provenance.native_release_published_at,
        run_name=run_name,
    ) is not None:
        raise ContractError(
            f"a run named {run_name!r} already exists; refusing to dispatch a duplicate"
        )

    gateway.dispatch_workflow(
        workflow_file=workflow_file, ref=default_branch, inputs=inputs
    )

    dispatched_run_id = None
    for attempt in range(DISPATCH_READBACK_ATTEMPTS):
        if attempt:
            gateway.sleep(DISPATCH_READBACK_DELAY_SECONDS)
        dispatched_run_id = _find_named_run(
            gateway,
            workflow_file=workflow_file,
            workflow_path=workflow_path,
            default_branch=default_branch,
            created_since=plan.provenance.native_release_published_at,
            run_name=run_name,
        )
        if dispatched_run_id is not None:
            break
    if dispatched_run_id is None:
        raise ContractError(
            f"dispatched {workflow_file} but no run named {run_name!r} appeared; "
            "the pipeline state is unknown"
        )
    return _with_dispatch_record(
        plan, ref=default_branch, inputs=inputs, dispatched_run_id=dispatched_run_id
    )


def _with_dispatch_record(
    plan: OrchestrationPlan,
    *,
    ref: str,
    inputs: Mapping[str, str],
    dispatched_run_id: str | None = None,
) -> OrchestrationPlan:
    """Record the exact ref and inputs a dispatch used, or would have used."""
    return OrchestrationPlan(
        action=plan.action,
        reason=plan.reason,
        provenance=plan.provenance,
        correlation_id=plan.correlation_id,
        release_target=plan.release_target,
        candidate_run_id=plan.candidate_run_id,
        qualification_run_id=plan.qualification_run_id,
        dispatch_workflow=plan.dispatch_workflow,
        dispatch_ref=ref,
        dispatch_run_name=plan.dispatch_run_name,
        dispatch_inputs=dict(inputs),
        dispatched_run_id=dispatched_run_id,
    )
