// Tests of scripts/release/qualification.mjs: the attestation, candidate,
// phase, environment and artifact-payload tests of
// scripts/release_qualification_test.py, one test per Python test method with
// the same name and assertions (see qualification_fixtures.mjs for the full
// mapping).

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';

import { ContractError } from '../../scripts/release/contract.mjs';
import {
  PyFloat, pyDict, pyEquals, pyItems, pyJsonDumps, pyJsonLoads,
} from '../../scripts/release/json.mjs';
import { ARTIFACTS, generate } from '../../scripts/release/manifest.mjs';
import {
  ATTESTATION_CANONICAL_NESTING,
  ATTESTATION_KEYS,
  EXPECTED_MODEL_PINS,
  HEAVY_GATES,
  MAX_ATTESTATION_BYTES,
  MAX_QUALIFICATION_CPU_COUNT,
  MAX_QUALIFICATION_MEMORY_BYTES,
  REQUIRED_SPEECH_MODES,
  REQUIRED_TTS_MODES,
  REQUIRED_UNPROVEN_CAPABILITIES,
  SPEECH_PHASE_KEYS,
  TTS_PHASE_KEYS,
  buildAttestation,
  canonicalJson,
  loadAttestationFile,
  loadCandidate,
  parseAttestationJson,
  pyDeepCopy,
  qualificationEnvironment as probeEnvironment,
  qualificationRunIdentity,
} from '../../scripts/release/qualification.mjs';
import {
  BRIDGE_SHA,
  CANDIDATE_ARTIFACT_ID,
  CANDIDATE_RUN_ID,
  CORRELATION_ID,
  DEFAULT_HEAD_SHA,
  NATIVE_COMMIT,
  NATIVE_MANIFEST_SHA,
  QUALIFICATION_RUN_ID,
  UPSTREAM_COMMIT,
  camel,
  manifestArgs,
  qualificationEnvironment,
  qualificationIdentity,
  setUp,
  speechPhase,
  ttsPhase,
  writeCandidate,
} from './qualification_fixtures.mjs';

const BRIDGE_REPOSITORY = 'leehack/llama-web-bridge';

// A test with QualificationTest.setUp()/tearDown().
function qualificationTest(name, fn) {
  test(name, () => {
    const context = setUp();
    try {
      fn(context);
    } finally {
      context.cleanup();
    }
  });
}

// assertRaises(ContractError); returns str(exception).
function raises(fn, message = undefined) {
  let caught;
  assert.throws(fn, (error) => {
    caught = error;
    return error instanceof ContractError;
  }, message);
  return caught.message;
}

const copy = pyDeepCopy;

// --- happy path -----------------------------------------------------------

qualificationTest('test_valid_attestation_binds_candidate_and_every_identity', (t) => {
  const result = t.verify(t.attestation, {
    candidateFingerprint: t.fingerprint,
    candidateRunId: CANDIDATE_RUN_ID,
    bridgeSourceSha: BRIDGE_SHA,
    upstreamTag: 'v0.2.0',
    upstreamCommit: UPSTREAM_COMMIT,
    nativeReleaseTag: 'v0.2.0-1',
    nativeManifestSha256: NATIVE_MANIFEST_SHA,
    nativeCommit: NATIVE_COMMIT,
    emscriptenVersion: '6.0.8',
    releaseTag: 'v0.2.0-1',
    releaseRebuild: 1,
    orchestratorCorrelationId: CORRELATION_ID,
    harnessSha256: t.harnessDigest,
    ...qualificationIdentity(),
  });
  assert.equal(result.verified, true);
  assert.equal(result.candidate_fingerprint, t.fingerprint);
  assert.equal(result.candidate_run_id, CANDIDATE_RUN_ID);
  assert.equal(result.qualification_run_id, QUALIFICATION_RUN_ID);
  assert.equal(result.qualification_source_sha, DEFAULT_HEAD_SHA);
});

qualificationTest('test_attestation_is_deterministic_and_canonical', (t) => {
  const again = buildAttestation({
    manifest: t.manifest,
    candidateFingerprint: t.fingerprint,
    candidateRunId: CANDIDATE_RUN_ID,
    candidateArtifactId: CANDIDATE_ARTIFACT_ID,
    ...qualificationIdentity(),
    harnessDigest: t.harnessDigest,
    environment: qualificationEnvironment(),
    speechPhase: speechPhase(),
    ttsPhase: ttsPhase(),
  });
  const text = canonicalJson(t.attestation);
  assert.equal(text, canonicalJson(again));
  assert.equal(text, canonicalJson(pyJsonLoads(text)));
});

qualificationTest('test_attestation_states_every_uncovered_lane_explicitly', (t) => {
  assert.deepEqual(t.attestation.unproven_capabilities, {
    hardware_gpu_acceleration: 'unavailable-on-hosted-runners',
    real_device_intelligibility: 'unproven',
    real_device_playback: 'unproven',
    speaker_reference_fidelity: 'unproven',
    wasm32_text_to_speech: 'unsupported',
  });
});

qualificationTest('test_attestation_records_the_hosted_environment_it_ran_in', (t) => {
  assert.ok(pyEquals(t.attestation.qualification_environment, qualificationEnvironment()));
  assert.equal(t.attestation.qualification_environment.execution, 'hosted-github-actions');
});

qualificationTest('test_attestation_records_candidate_and_heavy_gates_separately', (t) => {
  assert.deepEqual(t.attestation.candidate_gates, { state_persistence: 'passed', multimodal: 'passed' });
  assert.deepEqual(t.attestation.heavy_gates, { speech_to_text: 'passed', text_to_speech: 'passed' });
});

// --- candidate identity -----------------------------------------------------

qualificationTest('test_candidate_manifest_must_not_claim_a_hosted_heavy_gate_pass', (t) => {
  const manifestPath = path.join(t.candidate, 'manifest.json');
  const manifest = pyJsonLoads(fs.readFileSync(manifestPath, 'utf8'));
  manifest.qualification_gates.speech_to_text = 'passed';
  fs.writeFileSync(manifestPath, `${pyJsonDumps(manifest, { indent: 2, sortKeys: true })}\n`);
  const message = raises(() => buildAttestation({
    manifest,
    candidateFingerprint: t.fingerprint,
    candidateRunId: CANDIDATE_RUN_ID,
    candidateArtifactId: CANDIDATE_ARTIFACT_ID,
    ...qualificationIdentity(),
    harnessDigest: t.harnessDigest,
    environment: qualificationEnvironment(),
    speechPhase: speechPhase(),
    ttsPhase: ttsPhase(),
  }));
  assert.ok(message.includes('a pass its own run never executed'), message);
});

qualificationTest('test_candidate_directory_with_unexpected_file_rejected', (t) => {
  fs.writeFileSync(path.join(t.candidate, 'extra.txt'), 'stow');
  assert.ok(raises(() => loadCandidate(t.candidate)).includes('unexpected'));
});

qualificationTest('test_candidate_directory_missing_file_rejected', (t) => {
  fs.unlinkSync(path.join(t.candidate, 'sha256sums.txt'));
  assert.ok(raises(() => loadCandidate(t.candidate)).includes('missing'));
});

qualificationTest('test_tampered_candidate_artifact_rejected', (t) => {
  fs.writeFileSync(path.join(t.candidate, 'llama_webgpu_core.wasm'), 'tampered');
  raises(() => loadCandidate(t.candidate));
});

qualificationTest('test_candidate_manifest_with_duplicate_keys_rejected', (t) => {
  fs.writeFileSync(path.join(t.candidate, 'manifest.json'), '{"schema_version": 2, "schema_version": 2}');
  assert.ok(raises(() => loadCandidate(t.candidate)).toLowerCase().includes('duplicate'));
});

qualificationTest('test_candidate_manifest_with_wrong_field_type_rejected', (t) => {
  const manifestPath = path.join(t.candidate, 'manifest.json');
  const manifest = pyJsonLoads(fs.readFileSync(manifestPath, 'utf8'));
  manifest.release_rebuild = '1';
  fs.writeFileSync(manifestPath, pyJsonDumps(manifest));
  assert.ok(raises(() => loadCandidate(t.candidate)).includes('release_rebuild'));
});

qualificationTest('test_attestation_for_a_different_candidate_rejected', (t) => {
  const other = path.join(t.tmp, 'other-candidate');
  writeCandidate(other, { marker: 'rebuilt' });
  assert.ok(raises(() => t.verify(t.attestation, { candidateDir: other })).includes('fingerprint mismatch'));
});

qualificationTest('test_publication_refuses_a_rebuilt_candidate_from_another_run', (t) => {
  const rebuilt = path.join(t.tmp, 'rebuilt');
  fs.mkdirSync(rebuilt);
  for (const name of ARTIFACTS) fs.copyFileSync(path.join(t.candidate, name), path.join(rebuilt, name));
  generate(manifestArgs(rebuilt, {
    githubRunId: '99999999999',
    githubRunUrl: `https://github.com/${BRIDGE_REPOSITORY}/actions/runs/99999999999`,
  }));
  // Byte-identical binaries, different run provenance: the manifest differs,
  // so the fingerprint differs and the attestation cannot be reused.
  assert.ok(raises(() => t.verify(t.attestation, { candidateDir: rebuilt })).includes('fingerprint mismatch'));
});

qualificationTest('test_stale_attestation_replayed_against_new_release_rejected', (t) => {
  const newer = path.join(t.tmp, 'newer');
  fs.mkdirSync(newer);
  for (const name of ARTIFACTS) fs.copyFileSync(path.join(t.candidate, name), path.join(newer, name));
  generate(manifestArgs(newer, { releaseTag: 'v0.2.0-2', releaseRebuild: 2 }));
  raises(() => t.verify(t.attestation, { candidateDir: newer }));
});

// --- attestation shape --------------------------------------------------------

qualificationTest('test_missing_required_field_rejected', (t) => {
  for (const key of ATTESTATION_KEYS) {
    const bad = copy(t.attestation);
    delete bad[key];
    assert.ok(raises(() => t.verify(bad)).includes(key), key);
  }
});

qualificationTest('test_unexpected_field_rejected', (t) => {
  const bad = copy(t.attestation);
  bad.injected_token = 'sneaky';
  assert.ok(raises(() => t.verify(bad)).includes('injected_token'));
});

qualificationTest('test_unexpected_nested_field_rejected', (t) => {
  for (const [group, key] of [
    ['candidate_gates', 'extra_gate'],
    ['heavy_gates', 'extra_gate'],
    ['model_pins', 'extra_pin'],
    ['unproven_capabilities', 'extra_claim'],
    ['phases', 'extra_phase'],
  ]) {
    const bad = copy(t.attestation);
    bad[group][key] = 'passed';
    assert.ok(raises(() => t.verify(bad)).includes(key), group);
  }
});

qualificationTest('test_unexpected_phase_mode_field_rejected', (t) => {
  const bad = copy(t.attestation);
  bad.phases.speech_to_text.modes[0].extra = 1;
  assert.ok(raises(() => t.verify(bad)).includes('extra'));
});

qualificationTest('test_wrong_attestation_type_or_schema_rejected', (t) => {
  for (const [key, value] of [
    // Schema 1 is the retired manual-attestation shape and must not be
    // replayable against the automated pipeline.
    ['schema_version', 1],
    ['schema_version', 3],
    ['schema_version', true],
    ['schema_version', new PyFloat(2.0)],
    ['attestation_type', 'something-else'],
    ['harness_version', '0.0.1'],
    ['bridge_repository', 'attacker/bridge'],
    ['native_repository', 'attacker/native'],
    ['upstream_repository', 'attacker/llama.cpp'],
  ]) {
    const bad = copy(t.attestation);
    bad[key] = value;
    raises(() => t.verify(bad), `${key}=${String(value)}`);
  }
});

qualificationTest('test_malformed_types_rejected', (t) => {
  for (const [key, value] of [
    ['candidate_fingerprint', 123],
    ['candidate_run_id', 32919086955],
    ['release_rebuild', '1'],
    ['release_rebuild', -1],
    ['release_rebuild', true],
    ['candidate_gates', ['state_persistence']],
    ['model_pins', 'none'],
    ['phases', []],
    ['unproven_capabilities', null],
  ]) {
    const bad = copy(t.attestation);
    bad[key] = value;
    raises(() => t.verify(bad), key);
  }
});

qualificationTest('test_non_passed_gate_rejected', (t) => {
  for (const [group, gate] of [
    ['candidate_gates', 'state_persistence'],
    ['candidate_gates', 'multimodal'],
    ['heavy_gates', 'speech_to_text'],
    ['heavy_gates', 'text_to_speech'],
  ]) {
    for (const status of ['failed', 'skipped', 'pending', 'success']) {
      const bad = copy(t.attestation);
      bad[group][gate] = status;
      assert.ok(raises(() => t.verify(bad)).includes(gate));
    }
  }
});

qualificationTest('test_missing_gate_rejected', (t) => {
  for (const [group, gate] of [['candidate_gates', 'multimodal'], ['heavy_gates', 'text_to_speech']]) {
    const bad = copy(t.attestation);
    delete bad[group][gate];
    assert.ok(raises(() => t.verify(bad)).includes(gate));
  }
});

qualificationTest('test_unproven_capability_claimed_as_proven_rejected', (t) => {
  for (const key of Object.keys(REQUIRED_UNPROVEN_CAPABILITIES)) {
    const bad = copy(t.attestation);
    bad.unproven_capabilities[key] = 'proven';
    assert.ok(raises(() => t.verify(bad)).includes(key));
  }
});

qualificationTest('test_model_pin_mismatch_rejected', (t) => {
  for (const pin of Object.keys(EXPECTED_MODEL_PINS)) {
    const bad = copy(t.attestation);
    bad.model_pins[pin] = '0'.repeat(64);
    assert.ok(raises(() => t.verify(bad)).includes(pin));
  }
});

qualificationTest('test_harness_source_digest_mismatch_rejected', (t) => {
  assert.ok(raises(() => t.verify(t.attestation, { harnessSha256: '0'.repeat(64) })).includes('harness_source_sha256'));
});

qualificationTest('test_candidate_run_url_must_match_run_id', (t) => {
  const bad = copy(t.attestation);
  bad.candidate_run_url = `https://github.com/${BRIDGE_REPOSITORY}/actions/runs/1`;
  assert.ok(raises(() => t.verify(bad)).includes('candidate_run_url'));
});

qualificationTest('test_qualification_workflow_and_run_identity_are_bound', (t) => {
  for (const [key, value] of [
    ['candidate_workflow_path', '.github/workflows/ci.yml'],
    ['qualification_workflow_path', '.github/workflows/ci.yml'],
    ['qualification_run_id', '1'],
    ['qualification_run_attempt', 2],
    ['qualification_run_url', `https://github.com/${BRIDGE_REPOSITORY}/actions/runs/1`],
    ['qualification_source_sha', 'b'.repeat(40)],
  ]) {
    const bad = copy(t.attestation);
    bad[key] = value;
    raises(() => t.verify(bad, qualificationIdentity()), key);
  }
});

qualificationTest('test_identity_expectation_mismatch_rejected', (t) => {
  for (const [key, value] of [
    ['candidate_fingerprint', '0'.repeat(64)],
    ['candidate_run_id', '1'],
    ['bridge_source_sha', 'f'.repeat(40)],
    ['upstream_tag', 'v0.2.1'],
    ['upstream_commit', '2'.repeat(40)],
    ['native_release_tag', 'v0.2.0-2'],
    ['native_manifest_sha256', 'f'.repeat(64)],
    ['native_commit', '3'.repeat(40)],
    ['emscripten_version', '6.0.7'],
    ['release_tag', 'v0.2.0-2'],
    ['release_rebuild', 2],
    ['orchestrator_correlation_id', 'other-correlation'],
  ]) {
    assert.ok(raises(() => t.verify(t.attestation, { [camel(key)]: value })).includes(key), key);
  }
});

// --- phase coverage -------------------------------------------------------------

qualificationTest('test_missing_speech_memory_or_runtime_mode_rejected', (t) => {
  for (let dropped = 0; dropped < REQUIRED_SPEECH_MODES.length; dropped += 1) {
    const bad = copy(t.attestation);
    bad.phases.speech_to_text.modes.splice(dropped, 1);
    assert.ok(raises(() => t.verify(bad)).includes('must cover exactly'));
  }
});

qualificationTest('test_missing_tts_runtime_mode_rejected', (t) => {
  for (let dropped = 0; dropped < REQUIRED_TTS_MODES.length; dropped += 1) {
    const bad = copy(t.attestation);
    bad.phases.text_to_speech.modes.splice(dropped, 1);
    raises(() => t.verify(bad));
  }
});

qualificationTest('test_duplicate_mode_instead_of_coverage_rejected', (t) => {
  const bad = copy(t.attestation);
  bad.phases.speech_to_text.modes[1] = copy(bad.phases.speech_to_text.modes[0]);
  assert.ok(raises(() => t.verify(bad)).includes('repeats mode'));
});

qualificationTest('test_missing_speech_phase_timing_rejected', (t) => {
  for (const key of SPEECH_PHASE_KEYS) {
    const bad = copy(t.attestation);
    delete bad.phases.speech_to_text.modes[0].phase_timings_ms[key];
    assert.ok(raises(() => t.verify(bad)).includes(key));
  }
});

qualificationTest('test_missing_tts_phase_timing_rejected', (t) => {
  for (const key of TTS_PHASE_KEYS) {
    const bad = copy(t.attestation);
    delete bad.phases.text_to_speech.modes[0].phase_timings_ms[key];
    assert.ok(raises(() => t.verify(bad)).includes(key));
  }
});

qualificationTest('test_missing_or_invalid_resource_measurement_rejected', (t) => {
  for (const value of [0, -1, '8737062912', null]) {
    const bad = copy(t.attestation);
    bad.phases.speech_to_text.max_rss_bytes = value;
    assert.ok(raises(() => t.verify(bad)).includes('max_rss_bytes'));
  }
});

qualificationTest('test_phase_total_must_equal_the_sum_of_mode_totals', (t) => {
  for (const gate of HEAVY_GATES) {
    const bad = copy(t.attestation);
    bad.phases[gate].total_ms += 1;
    assert.ok(raises(() => t.verify(bad)).includes('total_ms'), gate);
  }
});

qualificationTest('test_mode_total_cannot_be_shorter_than_its_recorded_phases', (t) => {
  const bad = copy(t.attestation);
  bad.phases.speech_to_text.modes[0].total_ms = 1;
  assert.ok(raises(() => t.verify(bad)).includes('phase timings'));
});

qualificationTest('test_tts_wav_must_be_pcm16_mono_24khz', (t) => {
  for (const [key, value] of [['sample_rate', 16000], ['channels', 2], ['bits_per_sample', 8]]) {
    const bad = copy(t.attestation);
    bad.phases.text_to_speech.modes[0].wav[key] = value;
    assert.ok(raises(() => t.verify(bad)).includes('PCM16 mono'));
  }
});

qualificationTest('test_tts_wav_must_contain_frames', (t) => {
  for (const frames of [0, -1, '48000', null]) {
    const bad = copy(t.attestation);
    bad.phases.text_to_speech.modes[0].wav.frame_count = frames;
    assert.ok(raises(() => t.verify(bad)).includes('frame_count'));
  }
});

qualificationTest('test_tts_waveform_evidence_must_be_finite_and_non_silent', (t) => {
  for (const [key, value] of [
    ['peak', new PyFloat(0.001)],
    ['rms', new PyFloat(0.0001)],
    ['peak', new PyFloat(Infinity)],
    ['rms', '0.1'],
  ]) {
    const bad = copy(t.attestation);
    bad.phases.text_to_speech.modes[0][key] = value;
    raises(() => t.verify(bad), key);
  }
});

qualificationTest('test_speech_result_evidence_is_required', (t) => {
  for (const [key, value] of [
    ['cold_transcript', ''],
    ['warm_transcript', null],
    ['cancellation_result', ''],
    ['silence_transcript', 'hallucination'],
  ]) {
    const bad = copy(t.attestation);
    bad.phases.speech_to_text.modes[0][key] = value;
    raises(() => t.verify(bad), key);
  }
});

qualificationTest('test_tts_synthesis_evidence_required', (t) => {
  for (const [key, value] of [['frames_generated', 0], ['frames_generated', '24'], ['truncated', 'true']]) {
    const bad = copy(t.attestation);
    bad.phases.text_to_speech.modes[0][key] = value;
    assert.ok(raises(() => t.verify(bad)).includes(key));
  }
  const missing = copy(t.attestation);
  delete missing.phases.text_to_speech.modes[0].frames_generated;
  assert.ok(raises(() => t.verify(missing)).includes('frames_generated'));
});

qualificationTest('test_speech_phase_must_not_carry_a_wav_claim', (t) => {
  const bad = copy(t.attestation);
  bad.phases.speech_to_text.modes[0].wav = copy(bad.phases.text_to_speech.modes[0].wav);
  assert.ok(raises(() => t.verify(bad)).includes('wav'));
});

// --- hosted qualification environment -------------------------------------------

qualificationTest('test_non_hosted_execution_claim_rejected', (t) => {
  for (const execution of ['local', 'self-hosted', '', null]) {
    const bad = copy(t.attestation);
    bad.qualification_environment.execution = execution;
    assert.ok(raises(() => t.verify(bad)).includes('execution'), String(execution));
  }
});

qualificationTest('test_qualification_environment_must_be_exact_and_plausible', (t) => {
  for (const override of [
    { cpu_count: 0 },
    { cpu_count: MAX_QUALIFICATION_CPU_COUNT + 1 },
    { cpu_count: '4' },
    { cpu_count: true },
    { total_memory_bytes: 0 },
    { total_memory_bytes: MAX_QUALIFICATION_MEMORY_BYTES + 1 },
    { runner_os: '' },
    { runner_os: 'Linux runner' },
    { runner_arch: 64 },
  ]) {
    const bad = copy(t.attestation);
    Object.assign(bad.qualification_environment, override);
    raises(() => t.verify(bad), JSON.stringify(override));
  }
  for (const mutate of [
    (env) => { delete env.runner_os; },
    (env) => { env.extra = 'field'; },
  ]) {
    const bad = copy(t.attestation);
    mutate(bad.qualification_environment);
    raises(() => t.verify(bad));
  }
  const bad = copy(t.attestation);
  bad.qualification_environment = 'hosted';
  raises(() => t.verify(bad));
});

test('test_environment_probe_requires_github_hosted_runner_markers', () => {
  const hosted = {
    GITHUB_ACTIONS: 'true',
    RUNNER_ENVIRONMENT: 'github-hosted',
    RUNNER_OS: 'Linux',
    RUNNER_ARCH: 'X64',
  };
  const probed = probeEnvironment({ env: hosted });
  assert.equal(probed.execution, 'hosted-github-actions');
  assert.equal(probed.runner_os, 'Linux');
  assert.equal(probed.runner_arch, 'X64');
  for (const missing of Object.keys(hosted)) {
    const env = Object.fromEntries(Object.entries(hosted).filter(([key]) => key !== missing));
    assert.ok(raises(() => probeEnvironment({ env })).includes('hosted GitHub Actions'), missing);
  }
});

test('test_qualification_run_identity_comes_from_actions_environment', () => {
  const environment = {
    GITHUB_RUN_ID: QUALIFICATION_RUN_ID,
    GITHUB_RUN_ATTEMPT: '1',
    GITHUB_SHA: DEFAULT_HEAD_SHA,
    GITHUB_REPOSITORY: BRIDGE_REPOSITORY,
  };
  const identity = qualificationRunIdentity({ env: environment });
  assert.equal(identity.qualification_run_id, QUALIFICATION_RUN_ID);
  assert.equal(identity.qualification_run_attempt, 1);
  assert.equal(identity.qualification_source_sha, DEFAULT_HEAD_SHA);
  for (const missing of Object.keys(environment)) {
    const incomplete = Object.fromEntries(Object.entries(environment).filter(([key]) => key !== missing));
    raises(() => qualificationRunIdentity({ env: incomplete }), missing);
  }
});

// --- attestation artifact payload -------------------------------------------------

qualificationTest('test_canonical_attestation_artifact_round_trip', (t) => {
  const file = path.join(t.tmp, 'qualification-attestation.json');
  fs.writeFileSync(file, canonicalJson(t.attestation));
  assert.ok(pyEquals(loadAttestationFile(file), t.attestation));
});

qualificationTest('test_noncanonical_attestation_artifact_rejected', (t) => {
  const file = path.join(t.tmp, 'qualification-attestation.json');
  fs.writeFileSync(file, pyJsonDumps(t.attestation, { sortKeys: true }));
  assert.ok(raises(() => loadAttestationFile(file)).includes('canonical'));
});

qualificationTest('test_reordered_attestation_artifact_rejected', (t) => {
  const file = path.join(t.tmp, 'qualification-attestation.json');
  fs.writeFileSync(file, `${pyJsonDumps(t.attestation, { indent: 2, sortKeys: false })}\n`);
  raises(() => loadAttestationFile(file));
});

qualificationTest('test_duplicate_keys_in_attestation_artifact_rejected', (t) => {
  const file = path.join(t.tmp, 'qualification-attestation.json');
  fs.writeFileSync(file, '{\n  "schema_version": 2,\n  "schema_version": 2\n}\n');
  assert.ok(raises(() => loadAttestationFile(file)).toLowerCase().includes('duplicate'));
});

test('test_nonstandard_json_constants_rejected_before_schema_validation', () => {
  for (const constant of ['NaN', 'Infinity', '-Infinity']) {
    assert.ok(raises(() => parseAttestationJson(`{"value": ${constant}}`)).includes('non-standard'), constant);
  }
  raises(() => canonicalJson({ value: new PyFloat(Number.NaN) }));
});

qualificationTest('test_non_object_attestation_artifact_rejected', (t) => {
  const file = path.join(t.tmp, 'qualification-attestation.json');
  for (const raw of ['[1, 2, 3]\n', '"text"\n', '123\n', 'null\n', '{ broken']) {
    fs.writeFileSync(file, raw);
    raises(() => loadAttestationFile(file), raw);
  }
});

qualificationTest('test_oversized_attestation_artifact_rejected', (t) => {
  const file = path.join(t.tmp, 'qualification-attestation.json');
  fs.writeFileSync(file, 'x'.repeat(MAX_ATTESTATION_BYTES + 1));
  assert.ok(raises(() => loadAttestationFile(file)).includes('bound'));
});

qualificationTest('test_non_utf8_attestation_artifact_rejected', (t) => {
  const file = path.join(t.tmp, 'qualification-attestation.json');
  fs.writeFileSync(file, Buffer.from([0xff, 0xfe, 0x7b, 0x7d]));
  assert.ok(raises(() => loadAttestationFile(file)).includes('UTF-8'));
});

// --- speech and text-to-speech evidence ---------------------------------------------

qualificationTest('test_mutated_asr_cold_transcript_rejected', (t) => {
  const bad = copy(t.attestation);
  bad.phases.speech_to_text.modes[0].cold_transcript = 'completely wrong transcript';
  assert.ok(raises(() => t.verify(bad)).toLowerCase().includes('transcript'));
});

qualificationTest('test_mutated_asr_warm_transcript_rejected', (t) => {
  const bad = copy(t.attestation);
  bad.phases.speech_to_text.modes[0].warm_transcript = 'completely wrong transcript';
  assert.ok(raises(() => t.verify(bad)).toLowerCase().includes('transcript'));
});

qualificationTest('test_malformed_asr_cancellation_schema_rejected', (t) => {
  for (const malformed of [
    'cancelled',
    'cancel:invalid:0',
    'cancel:resolved:-1',
    'cancel:resolved:abc',
    'cancel:resolved:01',
    'cancel:rejected:00',
    `cancel:resolved:${'9'.repeat(10000)}`,
    '',
    null,
  ]) {
    const bad = copy(t.attestation);
    bad.phases.speech_to_text.modes[0].cancellation_result = malformed;
    raises(() => t.verify(bad), String(malformed).slice(0, 40));
  }
});

qualificationTest('test_rejected_cancellation_must_report_no_output', (t) => {
  // The gate only reaches the rejected state by throwing before it ever
  // assigns output, so a rejected result carrying characters means the
  // recorded state and the recorded output contradict each other.
  const bad = copy(t.attestation);
  bad.phases.speech_to_text.modes[0].cancellation_result = 'cancel:rejected:37';
  assert.ok(raises(() => t.verify(bad)).includes('rejected but reported'));

  const good = copy(t.attestation);
  good.phases.speech_to_text.modes[0].cancellation_result = 'cancel:rejected:0';
  assert.equal(t.verify(good).verified, true);
});

qualificationTest('test_asr_silence_transcript_must_stay_empty', (t) => {
  const bad = copy(t.attestation);
  bad.phases.speech_to_text.modes[0].silence_transcript = 'hallucinated speech';
  raises(() => t.verify(bad));
});

qualificationTest('test_tts_truncated_must_be_false', (t) => {
  for (const value of [true, 'false', null, 1]) {
    const bad = copy(t.attestation);
    bad.phases.text_to_speech.modes[0].truncated = value;
    raises(() => t.verify(bad), String(value));
  }
});

for (const key of ['cancellation_tested', 'pre_aborted_tested', 'unload_tested']) {
  qualificationTest(`test_tts_lifecycle_evidence_${key}_must_be_true`, (t) => {
    for (const value of [false, 'true', null, 0]) {
      const bad = copy(t.attestation);
      bad.phases.text_to_speech.modes[0][key] = value;
      raises(() => t.verify(bad), String(value));
    }
  });
}

qualificationTest('test_tts_lifecycle_evidence_reuse_sample_count_must_be_positive', (t) => {
  for (const value of [0, -1, '2400', null, true]) {
    const bad = copy(t.attestation);
    bad.phases.text_to_speech.modes[0].reuse_sample_count = value;
    raises(() => t.verify(bad), String(value));
  }
});

qualificationTest('test_candidate_artifact_id_and_run_attempt_bound_into_attestation', (t) => {
  assert.equal(t.attestation.candidate_artifact_id, 7);
  assert.equal(t.attestation.candidate_run_attempt, 1);
});

qualificationTest('test_qualification_run_identity_bound_into_attestation', (t) => {
  assert.equal(t.attestation.qualification_run_id, QUALIFICATION_RUN_ID);
  assert.equal(t.attestation.qualification_run_attempt, 1);
  assert.equal(t.attestation.qualification_source_sha, DEFAULT_HEAD_SHA);
});

qualificationTest('test_candidate_artifact_id_mismatch_rejected', (t) => {
  assert.ok(raises(() => t.verify(t.attestation, { candidateArtifactId: 999 })).includes('candidate_artifact_id'));
});

qualificationTest('test_candidate_run_attempt_mismatch_rejected', (t) => {
  assert.ok(raises(() => t.verify(t.attestation, { candidateRunAttempt: 2 })).includes('candidate_run_attempt'));
});

qualificationTest('test_attestation_receipt_contains_artifact_id_and_run_attempt', (t) => {
  const receipt = t.verify(t.attestation, { candidateArtifactId: 7, candidateRunAttempt: 1 });
  assert.equal(receipt.candidate_artifact_id, 7);
  assert.equal(receipt.candidate_run_attempt, 1);
});

// --- Node-only: the value model and error text the port must keep ---------------------

qualificationTest('attestation floats, ints and error text follow Python', (t) => {
  // A float that happens to be integral stays a float in the canonical bytes.
  const attestation = copy(t.attestation);
  attestation.phases.text_to_speech.modes[0].peak = new PyFloat(1.0);
  assert.match(canonicalJson(attestation), /"peak": 1\.0,/);
  assert.equal(t.verify(attestation).verified, true);
  // Python repr in messages.
  const bad = copy(t.attestation);
  bad.harness_version = new PyFloat(4.0);
  assert.equal(raises(() => t.verify(bad)), "harness_version must be '4.0.0', got 4.0");
  const phase = copy(t.attestation);
  phase.phases.speech_to_text.modes.splice(0, 1);
  assert.equal(
    raises(() => t.verify(phase)),
    "phases.speech_to_text must cover exactly [('wasm32', 'direct'), ('wasm32', 'worker'), ('wasm64', 'direct'), "
    + "('wasm64', 'worker')], covered [('wasm32', 'worker'), ('wasm64', 'direct'), ('wasm64', 'worker')]",
  );
  // A huge int is still an int; converting it to a float overflows as in Python.
  const huge = copy(t.attestation);
  huge.phases.text_to_speech.modes[0].rms = 10n ** 400n;
  assert.throws(() => t.verify(huge), (error) => error.pyType === 'OverflowError');
  // Ints of any size keep exact comparisons.
  const big = copy(t.attestation);
  big.phases.speech_to_text.total_ms = 2n ** 70n;
  assert.equal(
    raises(() => t.verify(big)),
    `phases.speech_to_text.total_ms must equal the sum of mode totals: ${2n ** 70n} != 372000`,
  );
  // pyDeepCopy keeps dict order and floats.
  const copied = copy(pyDict([['b', new PyFloat(1)], ['a', [1]]]));
  assert.deepEqual(pyItems(copied).map(([key]) => key), ['b', 'a']);
  assert.ok(copied.b instanceof PyFloat);
});

qualificationTest('load_candidate rejects symlinked and irregular entries with Python text', (t) => {
  const link = path.join(t.tmp, 'linked');
  fs.symlinkSync(t.candidate, link);
  assert.equal(raises(() => loadCandidate(link)), `candidate directory does not exist: ${link}`);
  assert.equal(raises(() => loadCandidate(path.join(t.tmp, 'absent/'))), `candidate directory does not exist: ${path.join(t.tmp, 'absent')}`);
  fs.rmSync(path.join(t.candidate, 'sha256sums.txt'));
  fs.mkdirSync(path.join(t.candidate, 'sha256sums.txt'));
  assert.equal(raises(() => loadCandidate(t.candidate)), 'candidate entry must be an immutable regular file: sha256sums.txt');
  fs.rmdirSync(path.join(t.candidate, 'sha256sums.txt'));
  fs.writeFileSync(path.join(t.candidate, 'zz'), '');
  assert.equal(
    raises(() => loadCandidate(t.candidate)),
    "candidate directory must contain exactly the publication files (unexpected: ['zz'], missing: ['sha256sums.txt'])",
  );
});

qualificationTest('canonical_json raises RecursionError where Python runs out of frames', (t) => {
  // Measured through `verify-attestation` on Python 3.12: 992 nested
  // containers encode, 993 do not, and a float costs one more; the error
  // surfaces in traversal order, after a ValueError in an earlier sorted key.
  const write = (text) => {
    const file = path.join(t.tmp, 'deep.json');
    fs.writeFileSync(file, text);
    return file;
  };
  const nest = (depth, leaf) => `${'{"a": '.repeat(depth)}${leaf}${'}'.repeat(depth)}`;
  const recursion = (error) => error.pyType === 'RecursionError' && error.message === 'maximum recursion depth exceeded';
  const notCanonical = 'attestation artifact is not the canonical serialization of its own content';
  assert.equal(raises(() => loadAttestationFile(write(nest(ATTESTATION_CANONICAL_NESTING, '1')))), notCanonical);
  assert.throws(() => loadAttestationFile(write(nest(ATTESTATION_CANONICAL_NESTING, '1.5'))), recursion);
  assert.throws(() => loadAttestationFile(write(nest(ATTESTATION_CANONICAL_NESTING, '[]'))), recursion);
  assert.throws(() => loadAttestationFile(write(nest(ATTESTATION_CANONICAL_NESTING + 1, '1'))), recursion);
  assert.throws(() => loadAttestationFile(write(nest(4000, '1'))), recursion);
  const inner = nest(ATTESTATION_CANONICAL_NESTING - 1, '1');
  assert.equal(
    raises(() => loadAttestationFile(write(`{"b": 1e400, "a": ${inner}}`))),
    'attestation is not canonical JSON data: Out of range float values are not JSON compliant: inf',
  );
  assert.throws(() => loadAttestationFile(write(`{"b": 1e400, "a": ${nest(ATTESTATION_CANONICAL_NESTING, '1')}}`)), recursion);
  // Without a budget the encoder has no Python stack to run out of.
  assert.match(canonicalJson(pyJsonLoads(nest(ATTESTATION_CANONICAL_NESTING + 1, '1'))), /^\{/);
});
