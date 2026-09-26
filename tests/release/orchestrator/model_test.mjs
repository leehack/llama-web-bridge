// Tests of scripts/release/orchestrator/model.mjs. The Python module has no
// suite of its own (its types are exercised through the other suites); these
// Node-only tests pin what the port adds: dataclass equality and set identity,
// the __init__ TypeError text, the validation messages, and to_dict().

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ContractError } from '../../../scripts/release/contract.mjs';
import { pyJsonDumps } from '../../../scripts/release/json.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as runNames from '../../../scripts/release/orchestrator/run_names.mjs';
import {
  BRIDGE_SHA, NATIVE_MANIFEST_SHA, makeLegacyV0140Provenance, makeProvenance,
} from './fixtures.mjs';

function message(fn) {
  try {
    fn();
  } catch (error) {
    return `${error.pyType ?? error.name}: ${error.message}`;
  }
  return null;
}

const binding = (overrides = {}) => new model.PipelineBinding({
  bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.40', releaseRebuild: 0, ...overrides,
});

test('dataclass equality and set identity follow Python', () => {
  assert.ok(binding().equals(binding()));
  assert.ok(!binding().equals(binding({ releaseTag: 'v0.1.41' })));
  assert.ok(!binding().equals(new model.ReleaseTarget({ releaseTag: 'v0.1.40', releaseRebuild: 0 })));
  assert.ok(binding().releaseTarget.equals(new model.ReleaseTarget({ releaseTag: 'v0.1.40', releaseRebuild: 0 })));
  const bindings = new model.PyValueSet([binding(), binding(), binding({ releaseTag: 'v0.1.41' })]);
  assert.equal(bindings.size, 2);
  assert.ok(bindings.has(binding()));
  // 1 == 1.0 == True share one member, as they share one hash.
  assert.equal(new model.PyValueSet([1, true]).size, 1);
  assert.equal(message(() => model.pyHashKey({ a: 1 })), "TypeError: unhashable type: 'dict'");
  assert.ok(makeProvenance().equals(makeProvenance()));
  assert.ok(Object.isFrozen(makeProvenance()));
});

test('dataclass __init__ raises Python TypeError text', () => {
  assert.equal(
    message(() => new model.ReleaseTarget({ releaseTag: 'v0.1.40' })),
    "TypeError: ReleaseTarget.__init__() missing 1 required positional argument: 'release_rebuild'",
  );
  assert.equal(
    message(() => new model.PipelineBinding({})),
    "TypeError: PipelineBinding.__init__() missing 3 required positional arguments: 'bridge_source_sha', 'release_tag', and 'release_rebuild'",
  );
  assert.equal(
    message(() => new model.PipelineObservation({ unknown: 1 })),
    "TypeError: PipelineObservation.__init__() got an unexpected keyword argument 'unknown'",
  );
  const observation = new model.PipelineObservation();
  assert.equal(observation.publishRetry, false);
  assert.equal(observation.binding, null);
});

test('provenance and binding validation messages', () => {
  assert.equal(message(() => makeProvenance({ bridgeBuildSha: 'A'.repeat(40) })), 'ContractError: bridge_build_sha must be a lowercase full 40-character commit SHA');
  assert.equal(message(() => makeProvenance({ nativeManifestSha256: 'x' })), 'ContractError: native_manifest_sha256 must be a lowercase 64-character SHA-256');
  assert.equal(message(() => makeProvenance({ nativeRepo: 'someone/llamadart-native' })), 'ContractError: native_repo must be exactly leehack/llamadart-native');
  assert.equal(message(() => makeProvenance({ nativeReleasePublishedAt: 'yesterday' })), 'ContractError: native_release_published_at must use YYYY-MM-DDTHH:MM:SSZ');
  assert.equal(
    message(() => makeProvenance({ nativeReleaseTag: 'b9165' })),
    "ContractError: native release 'b9165' and upstream tag 'v0.2.0' are on different channels",
  );
  assert.equal(message(() => binding({ bridgeSourceSha: 'abc' })), 'ContractError: bridge_source_sha must be a lowercase full 40-character commit SHA');
  assert.equal(message(() => binding({ bridgeSourceSha: 1 })), "TypeError: expected string or bytes-like object, got 'int'");
  assert.equal(message(() => binding({ releaseRebuild: 1 })), "ContractError: release tag 'v0.1.40' encodes rebuild 0, not 1");
  assert.throws(() => model.requireStableProvenance(new model.NativeProvenance({
    ...makeProvenance(), upstreamTag: 'b1', nativeReleaseTag: 'b1',
  })), new ContractError('the release orchestrator only advances the stable channel, but this provenance is development (b1@b1)'));
  assert.equal(model.requireStr('x', 'label'), 'x');
  assert.throws(() => model.requireStr('', 'label'), new ContractError('label must be a non-empty string'));
  assert.throws(() => model.requirePositiveInt(true, 'label'), new ContractError('label must be a positive integer'));
  assert.equal(model.requirePositiveInt(2n ** 70n, 'label'), 2n ** 70n);
});

test('published manifest compatibility is the one legacy v0.1.40 identity', () => {
  const legacy = makeLegacyV0140Provenance();
  assert.deepEqual(model.publishedManifestCompatibility({ tag: 'v0.1.40', provenance: legacy }), [
    model.LEGACY_MANUAL_QUALIFICATION_GATES,
    model.LEGACY_MANUAL_UNPROVEN_CAPABILITIES,
  ]);
  assert.equal(model.publishedManifestCompatibility({ tag: 'v0.1.41', provenance: legacy }), null);
  assert.equal(model.publishedManifestCompatibility({ tag: 'v0.1.40', provenance: makeProvenance() }), null);
  assert.equal(runNames.computeCorrelationId(legacy), `auto-stable-v0.3.0-${legacy.nativeManifestSha256.slice(0, 16)}`);
});

test('plan to_dict keeps Python key order and values', () => {
  const provenance = makeProvenance();
  const plan = new model.OrchestrationPlan({
    action: model.OrchestrationAction.DISPATCH_CANDIDATE,
    reason: 'r',
    provenance,
    correlationId: 'c',
    releaseTarget: new model.ReleaseTarget({ releaseTag: 'v0.1.40', releaseRebuild: 0 }),
    dispatchInputs: { b: '1', a: '2' },
  });
  assert.equal(
    pyJsonDumps(plan.toDict()),
    '{"schema_version": 1, "action": "dispatch_candidate", "reason": "r", "correlation_id": "c", "provenance": '
    + `{"bridge_source_sha": "${BRIDGE_SHA}", "bridge_build_sha": "${BRIDGE_SHA}", "upstream_tag": "v0.2.0", `
    + '"upstream_commit": "bb4caa7540188872173c44d161602d9271386413", "native_repo": "leehack/llamadart-native", '
    + '"native_release_tag": "v0.2.0", "native_commit": "1111111111111111111111111111111111111111", '
    + `"native_manifest_sha256": "${NATIVE_MANIFEST_SHA}", "native_release_published_at": "2026-08-19T12:34:56Z"}, `
    + '"release_tag": "v0.1.40", "release_rebuild": 0, "candidate_run_id": null, "qualification_run_id": null, '
    + '"in_flight_workflow": null, "in_flight_run_id": null, "dispatch_workflow": null, "dispatch_ref": null, '
    + '"dispatch_run_name": null, "dispatch_inputs": {"b": "1", "a": "2"}, "dispatched_run_id": null}',
  );
});
