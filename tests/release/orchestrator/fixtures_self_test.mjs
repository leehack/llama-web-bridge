// Node-only checks that tests/release/orchestrator/fixtures.mjs builds what
// scripts/release_orchestrator_fixtures_test.py builds, so the stage-2 driver
// suites can rely on it: the fake gateway's routes, recording and error text,
// candidate directories the release validators accept, flat zips the artifact
// extractor accepts, and the release and run payload shapes.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

import { extractFlatArtifactArchive } from '../../../scripts/release/archive.mjs';
import { ASSETS_REPOSITORY, BRIDGE_REPOSITORY, ContractError } from '../../../scripts/release/contract.mjs';
import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as runNames from '../../../scripts/release/orchestrator/run_names.mjs';
import { loadCandidate, loadPublishedCandidate } from '../../../scripts/release/qualification.mjs';
import {
  ASSETS_TAG_COMMIT, CANDIDATE_RUN_ID, DEFAULT_BRANCH, FakeGateway, HEAD_SHA,
  LEGACY_MANUAL_QUALIFICATION_GATES, LEGACY_MANUAL_UNPROVEN_CAPABILITIES, NATIVE_PUBLISHED_AT, OWNER,
  alignedReleaseStub, artifactInventory, directoryMembers, flatZip, makeProvenance, releasePayload,
  rewriteLegacyCandidateManifest, runPayload, runsResponse, withAdvancePipelineFixture, writeBridgeCandidate,
} from './fixtures.mjs';

test('fake gateway defaults, recording and error text', () => {
  const gateway = new FakeGateway({ jsonRoutes: { 'repos/x': { a: 1 } }, releaseAttestations: [[[ASSETS_REPOSITORY, 'v0.1.40'], { ok: true }]] });
  assert.deepEqual(gateway.apiJson(`repos/${BRIDGE_REPOSITORY}`), { default_branch: DEFAULT_BRANCH });
  assert.deepEqual(gateway.apiJson(`repos/${ASSETS_REPOSITORY}/immutable-releases`), { enabled: true, enforced_by_owner: true });
  assert.deepEqual(gateway.apiJson(`repos/${ASSETS_REPOSITORY}/git/matching-refs/tags?per_page=100`, { paginate: true }), [[]]);
  assert.deepEqual(gateway.apiJson('repos/x'), { a: 1 });
  assert.throws(() => gateway.apiJson('repos/y'), new ContractError('unmapped API path in test gateway: repos/y'));
  assert.deepEqual(gateway.apiPaths, [
    `repos/${BRIDGE_REPOSITORY}`,
    `repos/${ASSETS_REPOSITORY}/immutable-releases`,
    `repos/${ASSETS_REPOSITORY}/git/matching-refs/tags?per_page=100`,
    'repos/x',
    'repos/y',
  ]);
  assert.throws(() => gateway.downloadBytes('blob', { accept: 'application/zip' }), new ContractError('unmapped blob path in test gateway: blob'));
  assert.deepEqual(gateway.releaseAttestation({ repository: ASSETS_REPOSITORY, releaseTag: 'v0.1.40' }), { ok: true });
  assert.throws(
    () => gateway.releaseAttestation({ repository: ASSETS_REPOSITORY, releaseTag: 'v0.1.41' }),
    new ContractError(`unmapped release attestation in test gateway: ('${ASSETS_REPOSITORY}', 'v0.1.41')`),
  );
  gateway.dispatchWorkflow({ workflowFile: 'a.yml', ref: 'main', inputs: { x: '1' } });
  gateway.sleep(5);
  assert.deepEqual(gateway.dispatches, [{ workflowFile: 'a.yml', ref: 'main', inputs: { x: '1' } }]);
  assert.deepEqual(gateway.slept, [5]);
  assert.equal(gateway.utcNow(), '2026-08-30T00:00:00Z');
  assert.equal(gateway.dispatchIdentity(), OWNER);
  assert.equal(new FakeGateway({ identity: null }).dispatchIdentity(), null);
  assert.deepEqual(new FakeGateway({ governance: { enabled: false } }).apiJson(`repos/${ASSETS_REPOSITORY}/immutable-releases`), { enabled: false });
});

test('candidate, zip and release builders produce valid release inputs', () => {
  const root = fs.mkdtempSync(path.join(tmpdir(), 'sro-fixtures-'));
  try {
    const provenance = makeProvenance();
    const correlationId = runNames.computeCorrelationId(provenance);
    const candidate = path.join(root, 'candidate');
    writeBridgeCandidate(candidate, { releaseTag: 'v0.1.40', releaseRebuild: 0, correlationId });
    const [manifest] = loadCandidate(candidate);
    assert.equal(manifest.orchestrator_correlation_id, correlationId);
    assert.equal(manifest.github_run_id, CANDIDATE_RUN_ID);

    const members = directoryMembers(candidate);
    const archive = path.join(root, 'candidate.zip');
    fs.writeFileSync(archive, flatZip(members));
    const extracted = path.join(root, 'extracted');
    extractFlatArtifactArchive(archive, extracted, { artifactType: 'candidate' });
    for (const [name, data] of Object.entries(members)) assert.deepEqual(fs.readFileSync(path.join(extracted, name)), data, name);

    const release = releasePayload({ tag: 'v0.1.40', body: 'b', members });
    assert.equal(release.prerelease, false);
    assert.equal(release.target_commitish, ASSETS_TAG_COMMIT);
    assert.deepEqual(release.assets.map((asset) => asset.name), Object.keys(members).sort());
    assert.equal(release.assets[0].id, 900);
    assert.match(release.assets[0].digest, /^sha256:[0-9a-f]{64}$/u);
    assert.equal(releasePayload({ tag: 'v0.1.40-1', body: '', members: {} }).prerelease, true);
    assert.ok(alignedReleaseStub('v0.1.40', provenance).body.includes(`Orchestrator correlation: \`${correlationId}\``));

    rewriteLegacyCandidateManifest(candidate);
    const text = fs.readFileSync(path.join(candidate, 'manifest.json'), 'utf8');
    assert.ok(text.startsWith('{"artifacts":') && text.endsWith('}\n') && !text.includes(': '));
    assert.deepEqual(JSON.parse(text).qualification_gates, LEGACY_MANUAL_QUALIFICATION_GATES);
    assert.deepEqual(JSON.parse(text).unproven_capabilities, LEGACY_MANUAL_UNPROVEN_CAPABILITIES);
    // The rewritten manifest satisfies the historical published contract.
    const [published] = loadPublishedCandidate(candidate, {
      expectedQualificationGates: LEGACY_MANUAL_QUALIFICATION_GATES,
      expectedUnprovenCapabilities: LEGACY_MANUAL_UNPROVEN_CAPABILITIES,
    });
    assert.deepEqual(published.qualification_gates, LEGACY_MANUAL_QUALIFICATION_GATES);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('run payloads and the advance fixture routes', withAdvancePipelineFixture((self) => {
  const run = runPayload({ runId: '501', path: model.CANDIDATE_WORKFLOW_PATH, runName: self.candidateName, status: 'in_progress' });
  assert.equal(run.conclusion, null);
  assert.equal(run.id, 501);
  assert.equal(run.head_sha, HEAD_SHA);
  assert.equal(runPayload({ runId: '502', path: 'p', runName: 'n', conclusion: null }).conclusion, null);
  assert.equal(runPayload({ runId: '503', path: 'p', runName: 'n', apiName: 'Static' }).name, 'Static');
  assert.deepEqual(artifactInventory({ runId: '501', name: 'x', artifactId: 7 }), {
    total_count: 1,
    artifacts: [{ id: 7, name: 'x', expired: false, workflow_run: { id: 501 } }],
  });
  const routes = self.routes({ candidateRuns: [run] });
  assert.deepEqual(routes[`repos/${ASSETS_REPOSITORY}/releases?per_page=100`], [[]]);
  const candidateKey = Object.keys(routes).find((key) => key.includes('/bridge_candidate.yml/runs?'));
  assert.equal(
    candidateKey,
    `repos/${BRIDGE_REPOSITORY}/actions/workflows/bridge_candidate.yml/runs?per_page=100&event=workflow_dispatch&branch=main&actor=${OWNER}&created=%3E%3D${NATIVE_PUBLISHED_AT.replaceAll(':', '%3A')}`,
  );
  assert.deepEqual(routes[candidateKey], runsResponse([run]));
  assert.equal(self.newerNative().nativeReleaseTag, 'v0.2.1');
  assert.ok(fs.statSync(self.tmp).isDirectory());
  assert.ok(self.binding.equals(new model.PipelineBinding({ bridgeSourceSha: self.provenance.bridgeSourceSha, releaseTag: 'v0.1.40', releaseRebuild: 0 })));
}));

test('runBacklog drives an injected CLI entry', async () => {
  const calls = [];
  const orchestrator = {
    main(argv, { env, createGateway }) {
      calls.push([argv[0], env.GITHUB_EVENT_NAME, createGateway({ readToken: 'r', dispatchToken: null })]);
      const output = argv[argv.indexOf('--output-plan-json') + 1];
      const provenances = JSON.parse(fs.readFileSync(argv[argv.indexOf('--provenance-list-json') + 1], 'utf8'));
      fs.writeFileSync(output, JSON.stringify({ plans: provenances.map((item) => item.native_release_tag) }));
      return 0;
    },
  };
  const gateway = new FakeGateway();
  await withAdvancePipelineFixture(async (self) => {
    const [result, plan] = await self.runBacklog(gateway, [self.provenance, self.newerNative()]);
    assert.equal(result, 0);
    assert.deepEqual(plan, { plans: ['v0.2.0', 'v0.2.1'] });
  }, { orchestrator })();
  assert.deepEqual(calls, [['orchestrate-backlog', 'schedule', gateway]]);
  await assert.rejects(withAdvancePipelineFixture((self) => self.runBacklog(gateway, [])));
});
