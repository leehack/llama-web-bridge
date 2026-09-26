// Tests of scripts/release/manifest.mjs, one test per test method of
// scripts/generate_release_manifest_test.py, with the same names and
// assertions.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

import { ContractError } from '../../scripts/release/contract.mjs';
import { parseArguments } from '../../scripts/release/cli.mjs';
import { ARTIFACTS, OPTIONS, generate, renderManifestFiles } from '../../scripts/release/manifest.mjs';
import { pyJsonDumps, pyJsonLoads } from '../../scripts/release/json.mjs';
import * as V0_1_53 from './manifest_v0_1_53_fixture.mjs';
import { loadCandidate } from './publication_fixtures.mjs';

// The capabilities of every published schema-v2 manifest (v0.1.44 to v0.1.46).
// The orchestrator re-verifies published releases with
// release_qualification.load_candidate, which requires the generator's exact
// capabilities, so a new capability needs a historical readback contract first.
const PUBLISHED_CAPABILITIES = {
  wasm32: true,
  memory64: true,
  state_persistence: { direct: true, worker: true },
  multimodal: { direct: true, worker: true },
  speech_to_text: {
    advertised: true,
    direct: true,
    worker: true,
    wasm32: true,
    memory64: true,
  },
  text_to_speech: {
    advertised: true,
    direct: true,
    worker: true,
    wasm32: false,
    memory64: true,
  },
};

function withTemporaryDirectory(fn) {
  const directory = mkdtempSync(path.join(tmpdir(), 'release-manifest-'));
  try {
    return fn(directory);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

function manifestArgs(outDir) {
  ARTIFACTS.forEach((name, index) => writeFileSync(path.join(outDir, name), `artifact-${index}`));
  return {
    outDir,
    releaseTag: 'v0.2.0-1',
    releaseRebuild: 1,
    assetsRepo: 'leehack/llama-web-bridge-assets',
    bridgeRepo: 'leehack/llama-web-bridge',
    bridgeCommit: 'a'.repeat(40),
    upstreamRepo: 'ggml-org/llama.cpp',
    upstreamTag: 'v0.2.0',
    upstreamCommit: 'b'.repeat(40),
    nativeRepo: 'leehack/llamadart-native',
    nativeReleaseTag: 'v0.2.0-1',
    nativeManifestSha256: 'c'.repeat(64),
    nativeCommit: 'd'.repeat(40),
    emscriptenVersion: '6.0.8',
    orchestratorCorrelationId: 'llamadart-pin:run-123',
    githubRunId: '123456789',
    githubRunUrl: 'https://github.com/leehack/llama-web-bridge/actions/runs/123456789',
  };
}

test('test_published_manifests_still_pass_readback_validation', () => {
  withTemporaryDirectory((outDir) => {
    generate(manifestArgs(outDir));
    const manifestPath = path.join(outDir, 'manifest.json');
    const manifest = pyJsonLoads(readFileSync(manifestPath, 'utf8'));
    manifest.capabilities = PUBLISHED_CAPABILITIES;
    writeFileSync(manifestPath, pyJsonDumps(manifest));
    const [loaded] = loadCandidate(outDir);
    assert.deepEqual(loaded.capabilities, PUBLISHED_CAPABILITIES);
  });
});

test('test_generates_schema_v2_with_legacy_aliases_and_checksums', () => {
  withTemporaryDirectory((outDir) => {
    const args = manifestArgs(outDir);
    const manifest = generate(args);
    const firstBytes = readFileSync(path.join(outDir, 'manifest.json'));
    generate(args);
    assert.deepEqual(readFileSync(path.join(outDir, 'manifest.json')), firstBytes);
    assert.equal(manifest.schema_version, 2);
    assert.equal(manifest.release_tag, 'v0.2.0-1');
    assert.equal(manifest.bridge_commit, 'a'.repeat(40));
    assert.equal(manifest.upstream_commit, 'b'.repeat(40));
    assert.equal(manifest.native_commit, 'd'.repeat(40));
    assert.equal(manifest.bridge_assets_tag, 'v0.2.0-1');
    assert.equal(manifest.orchestrator_correlation_id, 'llamadart-pin:run-123');
    assert.equal(manifest.github_run_id, '123456789');
    assert.equal(manifest.github_run_url, 'https://github.com/leehack/llama-web-bridge/actions/runs/123456789');
    // Heavy real-model gates never run in the candidate workflow, so the
    // manifest states the automated-qualification requirement instead of
    // claiming a pass this candidate run never produced.
    assert.deepEqual(manifest.qualification_gates, {
      state_persistence: 'passed',
      multimodal: 'passed',
      speech_to_text: 'required-automated-qualification',
      text_to_speech: 'required-automated-qualification',
    });
    assert.deepEqual(manifest.unproven_capabilities, {
      hardware_gpu_acceleration: 'unavailable-on-hosted-runners',
      real_device_intelligibility: 'unproven',
      real_device_playback: 'unproven',
      speaker_reference_fidelity: 'unproven',
      wasm32_text_to_speech: 'unsupported',
    });
    assert.equal(manifest.capabilities.speech_to_text.advertised, true);
    assert.equal(manifest.capabilities.text_to_speech.advertised, true);
    assert.equal(Object.hasOwn(manifest, 'generated_at_utc'), false);

    const payload = pyJsonLoads(readFileSync(path.join(outDir, 'manifest.json'), 'utf8'));
    const artifact = ARTIFACTS[0];
    const expected = createHash('sha256').update(readFileSync(path.join(outDir, artifact))).digest('hex');
    assert.equal(payload.artifacts[artifact].sha256, expected);
    const sums = readFileSync(path.join(outDir, 'sha256sums.txt'), 'utf8');
    assert.ok(sums.includes(`${expected}  ${artifact}`));
    assert.ok(!sums.includes('manifest.json'));

    for (const invalidRunId of ['0', '01', '١٢٣', true, null]) {
      const runId = invalidRunId === true ? 'True' : invalidRunId === null ? 'None' : invalidRunId;
      assert.throws(
        () => generate({
          ...args,
          githubRunId: invalidRunId,
          githubRunUrl: `https://github.com/leehack/llama-web-bridge/actions/runs/${runId}`,
        }),
        ContractError,
        String(invalidRunId),
      );
    }
  });
});

// The real v0.1.53 candidate (see manifest_v0_1_53_fixture.mjs): Python's
// manifest.json and sha256sums.txt bytes, the arguments that produced them,
// and the digest and size of each built file.
test('the Node generator reproduces the published v0.1.53 manifest.json and sha256sums.txt byte for byte', () => {
  const sha256 = (text) => createHash('sha256').update(text, 'utf8').digest('hex');
  assert.equal(sha256(V0_1_53.MANIFEST_JSON), V0_1_53.MANIFEST_JSON_SHA256);
  assert.equal(sha256(V0_1_53.SHA256SUMS), V0_1_53.SHA256SUMS_SHA256);
  const args = parseArguments(['--out-dir', 'unused', ...V0_1_53.ARGV], { prog: 'generate_release_manifest.py', options: OPTIONS });
  assert.deepEqual(V0_1_53.ARTIFACTS.map(({ name }) => name), ARTIFACTS);
  const rendered = renderManifestFiles(args, V0_1_53.ARTIFACTS);
  assert.equal(rendered['manifest.json'], V0_1_53.MANIFEST_JSON);
  assert.equal(rendered['sha256sums.txt'], V0_1_53.SHA256SUMS);
  // The measured digests are the ones the published manifest records.
  const published = pyJsonLoads(V0_1_53.MANIFEST_JSON);
  for (const { name, sha256: digest, sizeBytes } of V0_1_53.ARTIFACTS) {
    assert.deepEqual({ ...published.files[name] }, { sha256: digest, size_bytes: sizeBytes }, name);
  }
});
