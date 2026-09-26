// Fixtures shared by the tests of scripts/release/qualification.mjs, the
// port of release_qualification_test.py's module-level helpers and setUp().
//
// Mapping of scripts/release_qualification_test.py (154 tests) to Node:
//
//   qualification_attestation_test.mjs (69): every test from
//     test_valid_attestation_binds_candidate_and_every_identity through
//     test_speech_phase_must_not_carry_a_wav_claim (42), the hosted
//     environment tests test_non_hosted_execution_claim_rejected,
//     test_qualification_environment_must_be_exact_and_plausible,
//     test_environment_probe_requires_github_hosted_runner_markers,
//     test_qualification_run_identity_comes_from_actions_environment (4), the
//     artifact payload tests test_canonical_attestation_artifact_round_trip
//     through test_non_utf8_attestation_artifact_rejected (8), and
//     test_mutated_asr_cold_transcript_rejected,
//     test_mutated_asr_warm_transcript_rejected,
//     test_malformed_asr_cancellation_schema_rejected,
//     test_rejected_cancellation_must_report_no_output,
//     test_asr_silence_transcript_must_stay_empty,
//     test_tts_truncated_must_be_false, the four
//     test_tts_lifecycle_evidence_* tests,
//     test_candidate_artifact_id_and_run_attempt_bound_into_attestation,
//     test_qualification_run_identity_bound_into_attestation,
//     test_candidate_artifact_id_mismatch_rejected,
//     test_candidate_run_attempt_mismatch_rejected,
//     test_attestation_receipt_contains_artifact_id_and_run_attempt (15).
//   qualification_run_test.mjs (13): test_valid_candidate_run_accepted,
//     test_unrelated_or_unsuccessful_run_rejected,
//     test_candidate_run_must_be_the_first_build_attempt,
//     test_run_attempt_one_is_mandatory_even_without_a_caller_override,
//     test_qualification_run_must_be_the_qualification_workflow,
//     test_run_id_and_head_sha_must_be_well_formed,
//     test_missing_duplicate_expired_or_foreign_artifact_rejected,
//     test_truncated_artifact_inventory_rejected,
//     test_wrong_workflow_run_actor_rejected,
//     test_wrong_workflow_run_triggering_actor_rejected,
//     test_artifact_name_alone_does_not_prove_provenance,
//     test_qualify_rejects_an_unprovenanced_local_candidate,
//     test_verify_run_cli_defaults_to_the_mandatory_first_attempt.
//   qualification_diagnostics_test.mjs (20): test_diagnostics_are_sanitized
//     through test_structured_keys_cannot_leak_urls_or_assignments, except
//     the two _write_smoke_diagnostics tests below.
//   qualification_harness_test.mjs (6):
//     test_harness_digest_covers_every_heavy_gate_source,
//     test_speech_fixture_holds_the_pinned_audio_and_transcript,
//     test_speech_fixture_fails_closed,
//     test_harness_sources_are_exactly_what_the_gates_execute_or_read,
//     test_harness_version_moves_with_the_harness_sources,
//     test_local_harness_must_match_the_exact_bridge_source.
//   archive_test.mjs (29) and wav_test.mjs (2), already ported: the archive
//     extraction tests test_artifact_archive_rejects_path_escape_and_duplicate_members
//     through test_unknown_artifact_type_rejected, and
//     test_generated_wav_identity_measured_from_the_file,
//     test_invalid_generated_wav_rejected.
//   tests/release/qualify*_test.mjs (15), the smoke-running half
//     (scripts/release/qualify.mjs): test_tts_phase_rejects_wav_path_escape_and_symlink,
//     test_tts_phase_rejects_coerced_result_types,
//     test_tts_phase_measures_waveform_evidence_from_the_wav,
//     test_speech_phase_rejects_a_rejected_cancellation_with_output,
//     test_structured_stdout_diagnostic_stays_valid_json,
//     test_structured_stdout_preserves_escaped_unicode_safely,
//     test_successful_child_is_parsed_from_raw_stdout,
//     test_successful_child_rejects_non_strict_json,
//     test_successful_child_duplicate_key_error_is_sanitized,
//     test_max_rss_is_reported_in_bytes,
//     test_child_env_drops_ambient_smoke_configuration,
//     test_smoke_timeout_is_bounded_and_writes_sanitized_diagnostics,
//     test_malformed_child_output_still_writes_sanitized_diagnostics,
//     test_node_executable_fails_closed_without_node,
//     test_qualify_runs_the_node_smokes_with_the_pinned_inputs.
//
// 69 + 13 + 20 + 6 + 31 + 15 = 154.

import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { BRIDGE_REPOSITORY } from '../../scripts/release/contract.mjs';
import { PyFloat } from '../../scripts/release/json.mjs';
import { ARTIFACTS, generate } from '../../scripts/release/manifest.mjs';
import {
  CANDIDATE_ARTIFACT_NAME,
  CANDIDATE_WORKFLOW_PATH,
  REQUIRED_SPEECH_MODES,
  REQUIRED_TTS_MODES,
  buildAttestation,
  harnessSourceSha256,
  loadCandidate,
  verifyAttestation,
} from '../../scripts/release/qualification.mjs';

export const BRIDGE_SHA = '565c8396597ea7c0fb4e8d5d966da8d884b156d8';
export const UPSTREAM_COMMIT = 'bb4caa7540188872173c44d161602d9271386413';
export const NATIVE_MANIFEST_SHA = '2e5d29d7f98f0d71e75d3fa63b7c55f3b2a7933247cc34ea2b1c5e053d142452';
export const NATIVE_COMMIT = '1'.repeat(40);
export const CANDIDATE_RUN_ID = '32919086955';
export const CANDIDATE_ARTIFACT_ID = 7;
export const QUALIFICATION_RUN_ID = '32919086977';
export const CANDIDATE_RUN_URL = `https://github.com/${BRIDGE_REPOSITORY}/actions/runs/${CANDIDATE_RUN_ID}`;
export const CORRELATION_ID = 'llamadart-pin:run-123';
export const DEFAULT_HEAD_BRANCH = 'main';
export const DEFAULT_HEAD_SHA = 'a'.repeat(40);

export const REPO = path.resolve(import.meta.dirname, '..', '..');
export const SCRIPTS_DIR = path.join(REPO, 'scripts');

// Pinned in the fixture the speech gate reads, not restated here, so this
// suite cannot pass against a transcript the real gate would reject.
export const DEFAULT_EXPECTED_TEXT = JSON.parse(
  fs.readFileSync(path.join(SCRIPTS_DIR, 'speech_to_text_fixture.json'), 'utf8'),
).expected_text;

export function makeTempDir(prefix = 'llama-web-bridge-qual-test-') {
  return fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), prefix)));
}

export function manifestArgs(directory, overrides = {}) {
  return {
    outDir: directory,
    releaseTag: 'v0.2.0-1',
    releaseRebuild: 1,
    assetsRepo: 'leehack/llama-web-bridge-assets',
    bridgeRepo: BRIDGE_REPOSITORY,
    bridgeCommit: BRIDGE_SHA,
    upstreamRepo: 'ggml-org/llama.cpp',
    upstreamTag: 'v0.2.0',
    upstreamCommit: UPSTREAM_COMMIT,
    nativeRepo: 'leehack/llamadart-native',
    nativeReleaseTag: 'v0.2.0-1',
    nativeManifestSha256: NATIVE_MANIFEST_SHA,
    nativeCommit: NATIVE_COMMIT,
    emscriptenVersion: '6.0.8',
    orchestratorCorrelationId: CORRELATION_ID,
    githubRunId: CANDIDATE_RUN_ID,
    githubRunUrl: CANDIDATE_RUN_URL,
    ...overrides,
  };
}

export function writeCandidate(directory, { marker = 'candidate' } = {}) {
  fs.mkdirSync(directory, { recursive: true });
  ARTIFACTS.forEach((name, index) => fs.writeFileSync(path.join(directory, name), `${marker}-${index}-${name}`));
  generate(manifestArgs(directory));
}

export function speechPhase() {
  return {
    modes: REQUIRED_SPEECH_MODES.map(([memoryMode, runtimeMode]) => ({
      memory_mode: memoryMode,
      runtime_mode: runtimeMode,
      total_ms: 93000,
      phase_timings_ms: {
        cancellation: 4000,
        cold_transcript: 30000,
        model_load: 20000,
        projector_load: 5000,
        silence: 4000,
        warm_transcript: 30000,
      },
      cold_transcript: DEFAULT_EXPECTED_TEXT,
      warm_transcript: DEFAULT_EXPECTED_TEXT,
      cancellation_result: 'cancel:resolved:12',
      silence_transcript: '',
    })),
    total_ms: 372000,
    max_rss_bytes: 8737062912,
  };
}

export function ttsPhase() {
  return {
    modes: REQUIRED_TTS_MODES.map(([memoryMode, runtimeMode]) => ({
      memory_mode: memoryMode,
      runtime_mode: runtimeMode,
      total_ms: 99980,
      phase_timings_ms: {
        model_load: 60000,
        projector_load: 10000,
        synthesis: 29980,
      },
      frames_generated: 24,
      peak: new PyFloat(0.5),
      rms: new PyFloat(0.1),
      truncated: false,
      cancellation_tested: true,
      pre_aborted_tested: true,
      reuse_sample_count: 2400,
      unload_tested: true,
      wav: {
        sha256: createHash('sha256').update(runtimeMode).digest('hex'),
        byte_length: 96044,
        channels: 1,
        bits_per_sample: 16,
        sample_rate: 24000,
        frame_count: 48000,
      },
    })),
    total_ms: 199960,
    max_rss_bytes: 10338385920,
  };
}

export function qualificationEnvironment(overrides = {}) {
  return {
    execution: 'hosted-github-actions',
    runner_os: 'Linux',
    runner_arch: 'X64',
    cpu_count: 4,
    total_memory_bytes: 16766181376,
    ...overrides,
  };
}

// qualification_identity() as buildAttestation/verifyAttestation options.
export function qualificationIdentity({ qualificationRunId = QUALIFICATION_RUN_ID, qualificationSourceSha = DEFAULT_HEAD_SHA } = {}) {
  return { qualificationRunId, qualificationRunAttempt: 1, qualificationSourceSha };
}

export function workflowRun(overrides = {}) {
  return {
    id: Number(CANDIDATE_RUN_ID),
    path: CANDIDATE_WORKFLOW_PATH,
    event: 'workflow_dispatch',
    status: 'completed',
    conclusion: 'success',
    head_branch: DEFAULT_HEAD_BRANCH,
    head_sha: DEFAULT_HEAD_SHA,
    run_attempt: 1,
    repository: { full_name: BRIDGE_REPOSITORY },
    head_repository: { full_name: BRIDGE_REPOSITORY },
    actor: { login: 'leehack' },
    triggering_actor: { login: 'leehack' },
    ...overrides,
  };
}

export function artifactInventory(overrides = {}) {
  return {
    total_count: 1,
    artifacts: [
      {
        id: 7,
        name: CANDIDATE_ARTIFACT_NAME,
        expired: false,
        workflow_run: { id: Number(CANDIDATE_RUN_ID) },
      },
    ],
    ...overrides,
  };
}

// snake_case keyword argument name -> camelCase option name.
export function camel(name) {
  return name.replace(/_([a-z0-9])/g, (_, char) => char.toUpperCase());
}

// setUp(): a written candidate, its manifest and fingerprint, the harness
// digest of scripts/, and the attestation that binds them. verify() is
// QualificationTest.verify. Call cleanup() when done.
export function setUp() {
  const tmp = makeTempDir();
  const candidate = path.join(tmp, 'candidate');
  writeCandidate(candidate);
  const [manifest, fingerprint] = loadCandidate(candidate);
  const harnessDigest = harnessSourceSha256(SCRIPTS_DIR);
  const attestation = buildAttestation({
    manifest,
    candidateFingerprint: fingerprint,
    candidateRunId: CANDIDATE_RUN_ID,
    candidateArtifactId: 7,
    candidateRunAttempt: 1,
    ...qualificationIdentity(),
    harnessDigest,
    environment: qualificationEnvironment(),
    speechPhase: speechPhase(),
    ttsPhase: ttsPhase(),
  });
  return {
    tmp,
    candidate,
    manifest,
    fingerprint,
    harnessDigest,
    attestation,
    verify(value, { candidateDir = candidate, ...kwargs } = {}) {
      return verifyAttestation({ attestation: value, candidateDir, ...kwargs });
    },
    cleanup() {
      fs.rmSync(tmp, { recursive: true, force: true });
    },
  };
}
