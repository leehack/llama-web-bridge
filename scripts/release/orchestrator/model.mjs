// Shared identities and data types of the stable release orchestrator, the
// Node port of scripts/release_orchestrator_model.py.
//
// Workflow files and paths, the repository owner, the one pre-automation
// publication (v0.1.40) that keeps its historical manifest contract, the
// identity patterns and scalar validators every stage shares, and the
// provenance, binding, plan, and observation types the stage modules pass to
// each other.
//
// The Python types are frozen dataclasses. Their Node counterparts are frozen
// PyDataclass instances built from one options object with camelCase keys;
// `equals` is dataclass ==, and `pyHashKey` gives the value identity a Python
// set or dict key uses, which PyValueSet builds on.

import {
  BRIDGE_REPOSITORY, Channel, ContractError, NATIVE_REPOSITORY, parseReleaseTag, parseUpstreamTag,
  requireRepository, requireSha256, validateReleaseIdentity,
} from '../contract.mjs';
import {
  PyException, PyFloat, isPyInt, pyEquals, pyRepr, pyTypeName,
} from '../json.mjs';
import { pyFullmatch } from '../python_compat.mjs';
import { CANDIDATE_WORKFLOW_PATH as RQ_CANDIDATE_WORKFLOW_PATH, QUALIFICATION_WORKFLOW_PATH as RQ_QUALIFICATION_WORKFLOW_PATH } from '../qualification.mjs';

export { ContractError };

export const CANDIDATE_WORKFLOW_FILE = 'bridge_candidate.yml';
export const QUALIFICATION_WORKFLOW_FILE = 'bridge_qualification.yml';
export const PUBLISH_WORKFLOW_FILE = 'publish_assets.yml';
export const CANDIDATE_WORKFLOW_PATH = RQ_CANDIDATE_WORKFLOW_PATH;
export const QUALIFICATION_WORKFLOW_PATH = RQ_QUALIFICATION_WORKFLOW_PATH;
export const PUBLISH_WORKFLOW_PATH = `.github/workflows/${PUBLISH_WORKFLOW_FILE}`;
export const SUPPORTED_PIPELINE_WORKFLOW_PATHS = Object.freeze(new Set([
  CANDIDATE_WORKFLOW_PATH,
  QUALIFICATION_WORKFLOW_PATH,
  PUBLISH_WORKFLOW_PATH,
]));

export const REPOSITORY_OWNER = BRIDGE_REPOSITORY.split('/')[0];

// v0.1.40 was immutably published for native/upstream v0.3.0 immediately
// before hosted automatic qualification replaced maintainer-run attestation.
// Its exact published bytes retain the historical gate vocabulary. This
// compatibility identity is deliberately narrower than the general candidate
// validator: every new candidate must still require hosted qualification.
export const LEGACY_MANUAL_QUALIFICATION_RELEASE_TAG = 'v0.1.40';
export const LEGACY_MANUAL_QUALIFICATION_BRIDGE_COMMIT = '0bdc8286fd52b70da27f5b039e1b4278361da0be';
export const LEGACY_MANUAL_QUALIFICATION_NATIVE_TAG = 'v0.3.0';
export const LEGACY_MANUAL_QUALIFICATION_NATIVE_COMMIT = '28fca14873d4b4c531bef4425b261e2b911bdcce';
export const LEGACY_MANUAL_QUALIFICATION_UPSTREAM_TAG = 'v0.3.0';
export const LEGACY_MANUAL_QUALIFICATION_UPSTREAM_COMMIT = 'c1d0e7a004015f23bc0233470b747b596f29b264';
export const LEGACY_MANUAL_QUALIFICATION_NATIVE_MANIFEST_SHA256 = '811fda999e70c3ad2716d1c196688dd38db62cf11a78044855ca94f71fabed45';
export const LEGACY_MANUAL_QUALIFICATION_GATES = Object.freeze({
  state_persistence: 'passed',
  multimodal: 'passed',
  speech_to_text: 'required-local-attestation',
  text_to_speech: 'required-local-attestation',
});
export const LEGACY_MANUAL_UNPROVEN_CAPABILITIES = Object.freeze({
  real_device_intelligibility: 'unproven',
  real_device_playback: 'unproven',
  speaker_reference_fidelity: 'unproven',
});

// _COMMIT_RE and _UTC_TIMESTAMP_RE, anchored for pyFullmatch.
export const COMMIT_RE = /^[0-9a-f]{40}$/u;
export const UTC_TIMESTAMP_RE = /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$/u;

// --- frozen dataclasses ----------------------------------------------------------

// The FIELDS default of a field without one.
export const REQUIRED = Symbol('required');

// "'a'", "'a' and 'b'", "'a', 'b', and 'c'", as CPython lists missing arguments.
function listArguments(names) {
  const quoted = names.map((name) => `'${name}'`);
  if (quoted.length === 1) return quoted[0];
  if (quoted.length === 2) return `${quoted[0]} and ${quoted[1]}`;
  return `${quoted.slice(0, -1).join(', ')}, and ${quoted.at(-1)}`;
}

// A frozen dataclass. A subclass declares static FIELDS as
// [camelName, snake_name, default] triples (default REQUIRED for none) and
// calls initFields(options) from its constructor, which assigns every field,
// raising the TypeError the dataclass __init__ raises for an unknown or a
// missing argument, before its own __post_init__ checks and freeze().
export class PyDataclass {
  initFields(options = {}) {
    const fields = this.constructor.FIELDS;
    const known = new Set(fields.map(([camel]) => camel));
    const init = `${this.constructor.name}.__init__()`;
    for (const key of Object.keys(options)) {
      if (!known.has(key)) throw new PyException('TypeError', `${init} got an unexpected keyword argument '${key}'`);
    }
    const missing = fields.filter(([camel, , fallback]) => fallback === REQUIRED && options[camel] === undefined);
    if (missing.length > 0) {
      const count = missing.length;
      throw new PyException('TypeError', `${init} missing ${count} required positional argument${count === 1 ? '' : 's'}: ${listArguments(missing.map(([, snake]) => snake))}`);
    }
    for (const [camel, , fallback] of fields) {
      const value = options[camel];
      this[camel] = value === undefined ? (fallback === REQUIRED ? null : fallback) : value;
    }
  }

  // dataclass __eq__: the same class and equal fields, in order.
  equals(other) {
    if (!(other instanceof PyDataclass) || other.constructor !== this.constructor) return false;
    return this.constructor.FIELDS.every(([camel]) => pyValueEquals(this[camel], other[camel]));
  }

  // The identity hash() and == give this value in a set or dict key.
  get pyHashKey() {
    return pyHashKey(this);
  }
}

// == of two field values: dataclasses and tuples by item, else Python ==.
export function pyValueEquals(left, right) {
  if (left instanceof PyDataclass || right instanceof PyDataclass) {
    return left instanceof PyDataclass && left.equals(right);
  }
  if (Array.isArray(left) && Array.isArray(right)) {
    return left.length === right.length && left.every((item, index) => pyValueEquals(item, right[index]));
  }
  return pyEquals(left ?? null, right ?? null);
}

// A string that is equal for two values exactly when Python's hash() and ==
// agree they are the same set member: 1, 1.0 and True share one key. An array
// is a dataclass's tuple field and hashes by item; a dict is unhashable, as in
// Python.
export function pyHashKey(value) {
  if (value === null || value === undefined) return 'N';
  if (value instanceof PyDataclass) {
    return `D${value.constructor.name}(${value.constructor.FIELDS.map(([camel]) => pyHashKey(value[camel])).join(',')})`;
  }
  if (typeof value === 'boolean') return `i${value ? 1 : 0}`;
  if (isPyInt(value)) return `i${BigInt(value)}`;
  if (value instanceof PyFloat || typeof value === 'number') {
    const number = value instanceof PyFloat ? value.value : value;
    return Number.isInteger(number) ? `i${BigInt(number)}` : `f${number}`;
  }
  if (typeof value === 'string') return `s${JSON.stringify(value)}`;
  if (Array.isArray(value)) return `t(${value.map(pyHashKey).join(',')})`;
  throw new PyException('TypeError', `unhashable type: '${pyTypeName(value)}'`);
}

// A Python set of dataclasses or scalars: membership by pyHashKey, iteration
// in insertion order (the first equal member added is the one kept).
export class PyValueSet {
  constructor(values = []) {
    this.members = new Map();
    for (const value of values) this.add(value);
  }

  add(value) {
    const key = pyHashKey(value);
    if (!this.members.has(key)) this.members.set(key, value);
    return this;
  }

  has(value) {
    return this.members.has(pyHashKey(value));
  }

  get size() {
    return this.members.size;
  }

  [Symbol.iterator]() {
    return this.members.values();
  }
}

// --- provenance ------------------------------------------------------------------

const PROVENANCE_COMMIT_FIELDS = [
  ['bridgeSourceSha', 'bridge_source_sha'],
  ['bridgeBuildSha', 'bridge_build_sha'],
  ['upstreamCommit', 'upstream_commit'],
  ['nativeCommit', 'native_commit'],
];

// Exact identity of the native release this pipeline is aligned to.
export class NativeProvenance extends PyDataclass {
  static FIELDS = Object.freeze([
    ['bridgeSourceSha', 'bridge_source_sha', REQUIRED],
    ['bridgeBuildSha', 'bridge_build_sha', REQUIRED],
    ['upstreamTag', 'upstream_tag', REQUIRED],
    ['upstreamCommit', 'upstream_commit', REQUIRED],
    ['nativeRepo', 'native_repo', REQUIRED],
    ['nativeReleaseTag', 'native_release_tag', REQUIRED],
    ['nativeCommit', 'native_commit', REQUIRED],
    ['nativeManifestSha256', 'native_manifest_sha256', REQUIRED],
    ['nativeReleasePublishedAt', 'native_release_published_at', REQUIRED],
  ]);

  constructor(options) {
    super();
    this.initFields(options);
    for (const [camel, snake] of PROVENANCE_COMMIT_FIELDS) {
      const value = this[camel];
      if (typeof value !== 'string' || COMMIT_RE.exec(value) === null) {
        throw new ContractError(`${snake} must be a lowercase full 40-character commit SHA`);
      }
    }
    requireSha256(this.nativeManifestSha256, 'native_manifest_sha256');
    requireRepository(this.nativeRepo, 'native_repo');
    if (this.nativeRepo !== NATIVE_REPOSITORY) throw new ContractError(`native_repo must be exactly ${NATIVE_REPOSITORY}`);
    if (typeof this.nativeReleasePublishedAt !== 'string' || UTC_TIMESTAMP_RE.exec(this.nativeReleasePublishedAt) === null) {
      throw new ContractError('native_release_published_at must use YYYY-MM-DDTHH:MM:SSZ');
    }
    const upstream = parseUpstreamTag(this.upstreamTag);
    const native = parseReleaseTag(this.nativeReleaseTag, { allowLegacy: true });
    if (native.channel !== upstream.channel) {
      throw new ContractError(`native release ${pyRepr(this.nativeReleaseTag)} and upstream tag ${pyRepr(this.upstreamTag)} are on different channels`);
    }
    Object.freeze(this);
  }

  get channel() {
    return parseUpstreamTag(this.upstreamTag).channel;
  }
}

// _published_manifest_compatibility: the one immutable pre-automation
// manifest contract as [gates, capabilities], if applicable, else null.
export function publishedManifestCompatibility({ tag, provenance }) {
  if (
    tag === LEGACY_MANUAL_QUALIFICATION_RELEASE_TAG
    && provenance.bridgeBuildSha === LEGACY_MANUAL_QUALIFICATION_BRIDGE_COMMIT
    && provenance.nativeReleaseTag === LEGACY_MANUAL_QUALIFICATION_NATIVE_TAG
    && provenance.nativeCommit === LEGACY_MANUAL_QUALIFICATION_NATIVE_COMMIT
    && provenance.upstreamTag === LEGACY_MANUAL_QUALIFICATION_UPSTREAM_TAG
    && provenance.upstreamCommit === LEGACY_MANUAL_QUALIFICATION_UPSTREAM_COMMIT
    && provenance.nativeManifestSha256 === LEGACY_MANUAL_QUALIFICATION_NATIVE_MANIFEST_SHA256
  ) {
    return [LEGACY_MANUAL_QUALIFICATION_GATES, LEGACY_MANUAL_UNPROVEN_CAPABILITIES];
  }
  return null;
}

// Only the stable channel is orchestrated; everything else is scan-only.
export function requireStableProvenance(provenance) {
  if (provenance.channel !== Channel.STABLE) {
    throw new ContractError(
      'the release orchestrator only advances the stable channel, but this '
      + `provenance is ${provenance.channel} `
      + `(${provenance.nativeReleaseTag}@${provenance.upstreamTag})`,
    );
  }
  return provenance;
}

export class ReleaseTarget extends PyDataclass {
  static FIELDS = Object.freeze([
    ['releaseTag', 'release_tag', REQUIRED],
    ['releaseRebuild', 'release_rebuild', REQUIRED],
  ]);

  constructor(options) {
    super();
    this.initFields(options);
    Object.freeze(this);
  }
}

// The exact source and output identity one pipeline attempt is pinned to.
export class PipelineBinding extends PyDataclass {
  static FIELDS = Object.freeze([
    ['bridgeSourceSha', 'bridge_source_sha', REQUIRED],
    ['releaseTag', 'release_tag', REQUIRED],
    ['releaseRebuild', 'release_rebuild', REQUIRED],
  ]);

  constructor(options) {
    super();
    this.initFields(options);
    if (pyFullmatch(COMMIT_RE, this.bridgeSourceSha) === null) {
      throw new ContractError('bridge_source_sha must be a lowercase full 40-character commit SHA');
    }
    validateReleaseIdentity(this.releaseTag, this.releaseRebuild, 'v0.0.0');
    Object.freeze(this);
  }

  get releaseTarget() {
    return new ReleaseTarget({ releaseTag: this.releaseTag, releaseRebuild: this.releaseRebuild });
  }
}

// `directory` is the path string of the verified release download, or null.
export class PublishedRelease extends PyDataclass {
  static FIELDS = Object.freeze([
    ['releaseId', 'release_id', REQUIRED],
    ['releaseTarget', 'release_target', REQUIRED],
    ['binding', 'binding', REQUIRED],
    ['publishedAt', 'published_at', REQUIRED],
    ['directory', 'directory', null],
  ]);

  constructor(options) {
    super();
    this.initFields(options);
    Object.freeze(this);
  }
}

// OrchestrationAction(str, Enum): each member is its value.
export const OrchestrationAction = Object.freeze({
  NOOP: 'noop',
  SUPERSEDED: 'superseded',
  SATISFIED_BY_IDENTICAL_RELEASE: 'satisfied_by_identical_release',
  IN_FLIGHT: 'in_flight',
  WAITING_FOR_PRIOR_PUBLICATION: 'waiting_for_prior_publication',
  DISPATCH_CANDIDATE: 'dispatch_candidate',
  DISPATCH_QUALIFICATION: 'dispatch_qualification',
  DISPATCH_PUBLISH: 'dispatch_publish',
  BLOCKED: 'blocked',
});

export class OrchestrationPlan extends PyDataclass {
  static FIELDS = Object.freeze([
    ['action', 'action', REQUIRED],
    ['reason', 'reason', REQUIRED],
    ['provenance', 'provenance', REQUIRED],
    ['correlationId', 'correlation_id', REQUIRED],
    ['releaseTarget', 'release_target', null],
    ['candidateRunId', 'candidate_run_id', null],
    ['qualificationRunId', 'qualification_run_id', null],
    ['inFlightWorkflow', 'in_flight_workflow', null],
    ['inFlightRunId', 'in_flight_run_id', null],
    ['dispatchWorkflow', 'dispatch_workflow', null],
    ['dispatchRef', 'dispatch_ref', null],
    ['dispatchRunName', 'dispatch_run_name', null],
    ['dispatchInputs', 'dispatch_inputs', null],
    ['dispatchedRunId', 'dispatched_run_id', null],
  ]);

  constructor(options) {
    super();
    this.initFields(options);
    Object.freeze(this);
  }

  // to_dict(): the plan record, keys in Python's insertion order.
  toDict() {
    const target = this.releaseTarget;
    const provenance = this.provenance;
    return {
      schema_version: 1,
      action: this.action,
      reason: this.reason,
      correlation_id: this.correlationId,
      provenance: {
        bridge_source_sha: provenance.bridgeSourceSha,
        bridge_build_sha: provenance.bridgeBuildSha,
        upstream_tag: provenance.upstreamTag,
        upstream_commit: provenance.upstreamCommit,
        native_repo: provenance.nativeRepo,
        native_release_tag: provenance.nativeReleaseTag,
        native_commit: provenance.nativeCommit,
        native_manifest_sha256: provenance.nativeManifestSha256,
        native_release_published_at: provenance.nativeReleasePublishedAt,
      },
      release_tag: target ? target.releaseTag : null,
      release_rebuild: target ? target.releaseRebuild : null,
      candidate_run_id: this.candidateRunId,
      qualification_run_id: this.qualificationRunId,
      in_flight_workflow: this.inFlightWorkflow,
      in_flight_run_id: this.inFlightRunId,
      dispatch_workflow: this.dispatchWorkflow,
      dispatch_ref: this.dispatchRef,
      dispatch_run_name: this.dispatchRunName,
      dispatch_inputs: this.dispatchInputs,
      dispatched_run_id: this.dispatchedRunId,
    };
  }
}

// Everything already proven about this correlation's pipeline.
export class PipelineObservation extends PyDataclass {
  static FIELDS = Object.freeze([
    ['published', 'published', null],
    ['binding', 'binding', null],
    ['freshBinding', 'fresh_binding', null],
    ['candidateInFlightRunId', 'candidate_in_flight_run_id', null],
    ['candidateRunId', 'candidate_run_id', null],
    ['satisfiedBy', 'satisfied_by', null],
    ['qualificationInFlightRunId', 'qualification_in_flight_run_id', null],
    ['qualificationRunId', 'qualification_run_id', null],
    ['publishInFlightRunId', 'publish_in_flight_run_id', null],
    ['publishSucceededRunId', 'publish_succeeded_run_id', null],
    ['publishRetry', 'publish_retry', false],
  ]);

  constructor(options = {}) {
    super();
    this.initFields(options);
    Object.freeze(this);
  }
}

// _require_str.
export function requireStr(value, label) {
  if (typeof value !== 'string' || !value) throw new ContractError(`${label} must be a non-empty string`);
  return value;
}

// _require_positive_int.
export function requirePositiveInt(value, label) {
  if (!isPyInt(value) || BigInt(value) <= 0n) throw new ContractError(`${label} must be a positive integer`);
  return value;
}
