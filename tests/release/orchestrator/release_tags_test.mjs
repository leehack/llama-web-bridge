// Tests of scripts/release/orchestrator/release_tags.mjs, one test per test
// method of scripts/release_orchestrator_release_tags_test.py, with the same
// names and assertions (a Python subTest is one loop iteration).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import * as assetReleases from '../../../scripts/release/orchestrator/asset_releases.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as releaseTags from '../../../scripts/release/orchestrator/release_tags.mjs';
import * as runNames from '../../../scripts/release/orchestrator/run_names.mjs';
import * as workflowRuns from '../../../scripts/release/orchestrator/workflow_runs.mjs';
import {
  BRIDGE_SHA,
  DEFAULT_BRANCH,
  assetReleaseStub,
  makeLegacyV0140Provenance,
  makeProvenance,
  runPayload,
  runsResponse,
} from './fixtures.mjs';

// --- ReleaseTargetTest -----------------------------------------------------------

test('test_bridge_assets_version_independently_of_upstream', () => {
  const target = releaseTags.selectNextReleaseTarget(['v0.1.38', 'v0.1.39'], { upstreamTag: 'v0.2.0' });
  assert.equal(target.releaseTag, 'v0.1.40');
  assert.equal(target.releaseRebuild, 0);
});

test('test_upstream_tag_is_never_used_as_the_output_tag', () => {
  const target = releaseTags.selectNextReleaseTarget(['v0.1.39'], { upstreamTag: 'v0.2.0' });
  assert.notEqual(target.releaseTag, 'v0.2.0');
});

test('test_existing_tag_collision_selects_a_free_rebuild', () => {
  const target = releaseTags.selectNextReleaseTarget(['v0.1.39', 'v0.1.40', 'v0.1.40-1'], { upstreamTag: 'v0.2.0' });
  assert.equal(target.releaseTag, 'v0.1.41');
  assert.equal(target.releaseRebuild, 0);
});

test('test_draft_or_unparsable_tag_collision_is_avoided', () => {
  const target = releaseTags.selectNextReleaseTarget(['v0.1.39', 'v0.1.40', 'nightly-scratch'], { upstreamTag: 'v0.2.0' });
  assert.equal(target.releaseTag, 'v0.1.41');
});

test('test_seed_release_when_no_stable_assets_exist', () => {
  const target = releaseTags.selectNextReleaseTarget([], { upstreamTag: 'v0.2.0' });
  assert.equal(target.releaseTag, releaseTags.INITIAL_STABLE_RELEASE_TAG);
  assert.equal(target.releaseRebuild, 0);
});

test('test_taken_next_version_moves_to_the_next_patch_version', () => {
  const target = releaseTags.selectNextReleaseTarget(['v0.1.39', 'v0.1.40'], { upstreamTag: 'v0.2.0', taken: new Set(['v0.1.41']) });
  assert.equal(target.releaseTag, 'v0.1.42');
  assert.equal(target.releaseRebuild, 0);
});

test('test_claims_set_the_floor_so_tags_follow_dispatch_order', () => {
  // v0.1.50 was claimed and released unpublished; v0.1.51 is still in
  // flight. A later pipeline must not take v0.1.50 below it.
  const target = releaseTags.selectNextReleaseTarget(['v0.1.49'], { upstreamTag: 'v0.5.0', taken: new Set(['v0.1.51']) });
  assert.equal(target.releaseTag, 'v0.1.52');
  assert.equal(target.releaseRebuild, 0);
});

test('test_historical_rebuild_tag_sets_the_floor_but_is_never_emitted', () => {
  const target = releaseTags.selectNextReleaseTarget(['v0.1.47', 'v0.1.47-1'], { upstreamTag: 'v0.4.1', taken: new Set(['v0.1.48']) });
  assert.equal(target.releaseTag, 'v0.1.49');
  assert.equal(target.releaseRebuild, 0);
});

test('test_seed_release_collision_moves_to_the_next_patch_version', () => {
  const target = releaseTags.selectNextReleaseTarget([], {
    upstreamTag: 'v0.2.0', taken: new Set([releaseTags.INITIAL_STABLE_RELEASE_TAG]),
  });
  assert.equal(target.releaseTag, 'v0.1.1');
  assert.equal(target.releaseRebuild, 0);
});

test('test_run_history_includes_claims_since_the_last_asset_publication', () => {
  const prior = assetReleaseStub();
  prior.published_at = '2026-08-10T00:00:00Z';
  assert.equal(assetReleases.workflowHistorySince([prior], makeProvenance()), '2026-08-10T00:00:00Z');
});

// --- ClaimedReleaseTagsTest --------------------------------------------------------

const OTHER_BUILD_SHA = 'b'.repeat(40);

function claimedSetUp() {
  return {
    other: runNames.computeCorrelationId(makeProvenance({ bridgeBuildSha: OTHER_BUILD_SHA })),
    current: runNames.computeCorrelationId(makeProvenance()),
  };
}

function candidateName(correlationId, tag, rebuild) {
  return runNames.candidateRunName(
    correlationId,
    new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: tag, releaseRebuild: rebuild }),
  );
}

function records(workflowPath, ...runs) {
  return workflowRuns.parseWorkflowRuns(
    runsResponse(runs.map(([runId, name, status, conclusion]) => runPayload({
      runId, path: workflowPath, runName: name, status, conclusion,
    }))),
    { workflowPath, defaultBranch: DEFAULT_BRANCH },
  );
}

test('test_finished_claim_of_another_build_is_dropped', () => {
  const self = claimedSetUp();
  for (const conclusion of ['success', 'failure']) {
    const runs = records(model.CANDIDATE_WORKFLOW_PATH, ['501', candidateName(self.other, 'v0.1.40', 0), 'completed', conclusion]);
    assert.equal(releaseTags.hasOtherBuildClaim(runs, { bridgeBuildSha: BRIDGE_SHA }), true, conclusion);
    assert.deepEqual(releaseTags.claimedReleaseTags(runs, { bridgeBuildSha: BRIDGE_SHA }), new Set(), conclusion);
  }
});

test('test_current_build_claim_is_kept', () => {
  const self = claimedSetUp();
  const runs = records(model.CANDIDATE_WORKFLOW_PATH, ['501', candidateName(self.current, 'v0.1.40', 0), 'completed', 'failure']);
  assert.equal(releaseTags.hasOtherBuildClaim(runs, { bridgeBuildSha: BRIDGE_SHA }), false);
  assert.deepEqual(releaseTags.claimedReleaseTags(runs, { bridgeBuildSha: BRIDGE_SHA }), new Set(['v0.1.40']));
});

test('test_claim_without_a_build_identity_is_kept', () => {
  const legacy = runNames.computeCorrelationId(makeLegacyV0140Provenance());
  assert.ok(!legacy.includes('-build-'));
  const runs = records(model.CANDIDATE_WORKFLOW_PATH, ['501', candidateName(legacy, 'v0.1.40', 0), 'completed', 'success']);
  assert.deepEqual(releaseTags.claimedReleaseTags(runs, { bridgeBuildSha: BRIDGE_SHA }), new Set(['v0.1.40']));
});

test('test_in_flight_candidate_of_another_build_keeps_its_claim', () => {
  const self = claimedSetUp();
  const runs = records(model.CANDIDATE_WORKFLOW_PATH, ['501', candidateName(self.other, 'v0.1.40', 0), 'in_progress', null]);
  assert.deepEqual(releaseTags.claimedReleaseTags(runs, { bridgeBuildSha: BRIDGE_SHA }), new Set(['v0.1.40']));
});

test('test_in_flight_downstream_run_of_another_build_keeps_its_claim', () => {
  const self = claimedSetUp();
  const candidates = records(model.CANDIDATE_WORKFLOW_PATH, ['501', candidateName(self.other, 'v0.1.40', 0), 'completed', 'success']);
  const binding = new model.PipelineBinding({ bridgeSourceSha: BRIDGE_SHA, releaseTag: 'v0.1.40', releaseRebuild: 0 });
  const downstream = [
    [model.QUALIFICATION_WORKFLOW_PATH, runNames.qualificationRunName(self.other, '501')],
    [model.PUBLISH_WORKFLOW_PATH, runNames.publishRunName(self.other, '501', '601', binding)],
  ];
  for (const [workflowPath, name] of downstream) {
    const inFlight = records(workflowPath, ['701', name, 'queued', null]);
    assert.deepEqual(
      releaseTags.claimedReleaseTags(candidates, { bridgeBuildSha: BRIDGE_SHA, downstreamRuns: inFlight }),
      new Set(['v0.1.40']),
      workflowPath,
    );
    const finished = records(workflowPath, ['701', name, 'completed', 'failure']);
    assert.deepEqual(
      releaseTags.claimedReleaseTags(candidates, { bridgeBuildSha: BRIDGE_SHA, downstreamRuns: finished }),
      new Set(),
      workflowPath,
    );
  }
});

test('test_satisfied_correlation_releases_its_claim', () => {
  const self = claimedSetUp();
  const sibling = runNames.computeCorrelationId(makeProvenance({ nativeReleaseTag: 'v0.2.1', nativeManifestSha256: 'f'.repeat(64) }));
  const runs = records(
    model.CANDIDATE_WORKFLOW_PATH,
    ['501', candidateName(self.current, 'v0.1.40', 0), 'completed', 'success'],
    ['502', candidateName(sibling, 'v0.1.40-1', 1), 'completed', 'success'],
  );
  assert.deepEqual(releaseTags.claimedReleaseTags(runs, { bridgeBuildSha: BRIDGE_SHA }), new Set(['v0.1.40', 'v0.1.40-1']));
  assert.deepEqual(
    releaseTags.claimedReleaseTags(runs, { bridgeBuildSha: BRIDGE_SHA, satisfiedCorrelationIds: new Set([self.current]) }),
    new Set(['v0.1.40-1']),
  );
});
