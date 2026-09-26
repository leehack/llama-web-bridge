// Deterministic correlation IDs and workflow run names, the Node port of
// scripts/release_orchestrator_run_names.py.
//
// The live actions/runs API never echoes a run's dispatch inputs, so each
// stage workflow renders a run-name from its own exact inputs. These helpers
// derive the correlation ID, build each stage's run name, and parse a run name
// back into its pipeline binding, failing closed when a name claims this
// correlation but does not parse exactly.

import { ContractError, requireCorrelationId } from '../contract.mjs';
import { PY_NON_SPACE_CLASS, pyIntFromString, pyRepr, pyStr } from '../json.mjs';
import { pyFullmatch, pyStrSplit } from '../python_compat.mjs';
import {
  LEGACY_MANUAL_QUALIFICATION_BRIDGE_COMMIT,
  LEGACY_MANUAL_QUALIFICATION_NATIVE_COMMIT,
  LEGACY_MANUAL_QUALIFICATION_NATIVE_MANIFEST_SHA256,
  LEGACY_MANUAL_QUALIFICATION_NATIVE_TAG,
  LEGACY_MANUAL_QUALIFICATION_UPSTREAM_COMMIT,
  LEGACY_MANUAL_QUALIFICATION_UPSTREAM_TAG,
  PipelineBinding,
} from './model.mjs';

// Run-name has no documented length guarantee. Keep this internal identity well
// below common database/display limits and accept only printable ASCII emitted
// by the exact input validators. A truncated title can therefore never be
// mistaken for a pipeline identity.
export const MAX_RUN_NAME_CHARACTERS = 200;

const RUN_ID_RE = /^[1-9][0-9]*$/u;
const RUN_NAME_RE = /^[A-Za-z0-9 ._:/-]+$/u;
const REBUILD_RE = /^(?:0|[1-9][0-9]*)$/u;

// Python's \S+ (a run of characters str.isspace() rejects).
const S = `${PY_NON_SPACE_CLASS}+`;

// _CANDIDATE_RUN_NAME_RE and _PUBLISH_RUN_NAME_RE, anchored for fullmatch.
export const CANDIDATE_RUN_NAME_RE = new RegExp(
  `^bridge-candidate (?<correlation_id>${S}) source:(?<bridge_source_sha>${S})`
  + ` tag:(?<release_tag>${S}) rebuild:(?<release_rebuild>${S})$`,
  'u',
);
export const PUBLISH_RUN_NAME_RE = new RegExp(
  `^publish-assets (?<correlation_id>${S}) candidate:(?<candidate_run_id>${S})`
  + ` qualification:(?<qualification_run_id>${S}) source:(?<bridge_source_sha>${S})`
  + ` tag:(?<release_tag>${S}) rebuild:(?<release_rebuild>${S})$`,
  'u',
);

// len(str): code points.
function pyLen(text) {
  let length = 0;
  for (const _char of text) length += 1;
  return length;
}

// Derive one stable pipeline identity from native and governed build inputs.
//
// The checkout source may advance while a pipeline is in flight because
// workflows, tests, or docs changed. The governed build identity does not, so
// those changes cannot orphan a candidate. A runtime/build change deliberately
// creates a new correlation and therefore requires a new qualified candidate.
//
// The one immutable pre-automation publication keeps its historical
// correlation only while the governed build identity is still its exact
// bridge commit.
export function computeCorrelationId(provenance) {
  if (
    provenance.bridgeBuildSha === LEGACY_MANUAL_QUALIFICATION_BRIDGE_COMMIT
    && provenance.nativeReleaseTag === LEGACY_MANUAL_QUALIFICATION_NATIVE_TAG
    && provenance.nativeCommit === LEGACY_MANUAL_QUALIFICATION_NATIVE_COMMIT
    && provenance.upstreamTag === LEGACY_MANUAL_QUALIFICATION_UPSTREAM_TAG
    && provenance.upstreamCommit === LEGACY_MANUAL_QUALIFICATION_UPSTREAM_COMMIT
    && provenance.nativeManifestSha256 === LEGACY_MANUAL_QUALIFICATION_NATIVE_MANIFEST_SHA256
  ) {
    return requireCorrelationId(`auto-stable-${provenance.nativeReleaseTag}-${provenance.nativeManifestSha256.slice(0, 16)}`);
  }
  const raw = `auto-stable-${provenance.nativeReleaseTag}`
    + `-${provenance.nativeManifestSha256.slice(0, 16)}`
    + `-build-${provenance.bridgeBuildSha.slice(0, 16)}`;
  return requireCorrelationId(raw);
}

export function candidateRunName(correlationId, binding) {
  requireCorrelationId(correlationId);
  return requireRunName(
    `bridge-candidate ${correlationId}`
    + ` source:${binding.bridgeSourceSha}`
    + ` tag:${binding.releaseTag}`
    + ` rebuild:${pyStr(binding.releaseRebuild)}`,
  );
}

export function qualificationRunName(correlationId, candidateRunId) {
  requireCorrelationId(correlationId);
  if (pyFullmatch(RUN_ID_RE, candidateRunId) === null) throw new ContractError('candidate_run_id must be a positive integer');
  return requireRunName(`bridge-qualification ${correlationId} candidate:${candidateRunId}`);
}

export function publishRunName(correlationId, candidateRunId, qualificationRunId, binding) {
  requireCorrelationId(correlationId);
  for (const [label, value] of [['candidate_run_id', candidateRunId], ['qualification_run_id', qualificationRunId]]) {
    if (pyFullmatch(RUN_ID_RE, value) === null) throw new ContractError(`${label} must be a positive integer`);
  }
  return requireRunName(
    `publish-assets ${correlationId}`
    + ` candidate:${candidateRunId}`
    + ` qualification:${qualificationRunId}`
    + ` source:${binding.bridgeSourceSha}`
    + ` tag:${binding.releaseTag}`
    + ` rebuild:${pyStr(binding.releaseRebuild)}`,
  );
}

// _require_run_name.
export function requireRunName(value) {
  if (
    typeof value !== 'string'
    || !value
    || pyLen(value) > MAX_RUN_NAME_CHARACTERS
    || RUN_NAME_RE.exec(value) === null
  ) {
    throw new ContractError('workflow run name exceeds the conservative length/safe-character contract');
  }
  return value;
}

// _parse_binding_fields.
function parseBindingFields(groups, label) {
  const rebuild = groups.release_rebuild;
  if (REBUILD_RE.exec(rebuild) === null) throw new ContractError(`${label} encodes a malformed rebuild counter`);
  try {
    return new PipelineBinding({
      bridgeSourceSha: groups.bridge_source_sha,
      releaseTag: groups.release_tag,
      releaseRebuild: pyIntFromString(rebuild),
    });
  } catch (error) {
    if (error instanceof ContractError) {
      throw new ContractError(`${label} encodes an invalid pipeline binding: ${error.message}`, { cause: error });
    }
    throw error;
  }
}

// Recover the binding a candidate run was dispatched with, or null.
//
// null means the name belongs to a different correlation. A name that claims
// this correlation but cannot be parsed exactly fails closed.
export function parseCandidateRunName(runName, correlationId) {
  requireCorrelationId(correlationId);
  if (typeof runName !== 'string') return null;
  const fields = pyStrSplit(runName);
  const claimsCorrelation = fields.length >= 2 && fields[0] === 'bridge-candidate' && fields[1] === correlationId;
  if (pyLen(runName) > MAX_RUN_NAME_CHARACTERS || RUN_NAME_RE.exec(runName) === null) {
    if (claimsCorrelation) {
      throw new ContractError(`candidate run name claiming correlation ${pyRepr(correlationId)} exceeds the length or character contract`);
    }
    return null;
  }
  const match = CANDIDATE_RUN_NAME_RE.exec(runName);
  if (match !== null) {
    if (match.groups.correlation_id !== correlationId) return null;
    return parseBindingFields(match.groups, 'candidate run name');
  }
  if (claimsCorrelation) {
    throw new ContractError(`candidate run name claiming correlation ${pyRepr(correlationId)} is malformed`);
  }
  return null;
}

// [candidateRunId, qualificationRunId, binding], or null.
export function parsePublishRunName(runName, correlationId) {
  requireCorrelationId(correlationId);
  if (typeof runName !== 'string' || pyLen(runName) > MAX_RUN_NAME_CHARACTERS || RUN_NAME_RE.exec(runName) === null) {
    return null;
  }
  const match = PUBLISH_RUN_NAME_RE.exec(runName);
  if (match === null || match.groups.correlation_id !== correlationId) return null;
  for (const label of ['candidate_run_id', 'qualification_run_id']) {
    if (RUN_ID_RE.exec(match.groups[label]) === null) throw new ContractError(`publish run name has a malformed ${label}`);
  }
  return [
    match.groups.candidate_run_id,
    match.groups.qualification_run_id,
    parseBindingFields(match.groups, 'publish run name'),
  ];
}
