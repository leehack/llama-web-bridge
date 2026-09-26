// Shared fixtures of the qualify_*_test.mjs suites, which test
// scripts/release/qualify.mjs (the smoke-running half of
// release_qualification.py).
//
// Where each of the 15 scripts/release_qualification_test.py tests that
// exercise this half lives (same test name in the Node suite):
//
//   Python test                                                    Node suite
//   test_tts_phase_rejects_wav_path_escape_and_symlink             qualify_phase_test.mjs
//   test_tts_phase_rejects_coerced_result_types                    qualify_phase_test.mjs
//   test_tts_phase_measures_waveform_evidence_from_the_wav         qualify_phase_test.mjs
//   test_speech_phase_rejects_a_rejected_cancellation_with_output  qualify_phase_test.mjs
//   test_successful_child_is_parsed_from_raw_stdout                qualify_smoke_test.mjs
//   test_successful_child_rejects_non_strict_json                  qualify_smoke_test.mjs
//   test_successful_child_duplicate_key_error_is_sanitized         qualify_smoke_test.mjs
//   test_max_rss_is_reported_in_bytes                              qualify_smoke_test.mjs
//   test_child_env_drops_ambient_smoke_configuration               qualify_smoke_test.mjs
//   test_smoke_timeout_is_bounded_and_writes_sanitized_diagnostics qualify_smoke_test.mjs
//   test_malformed_child_output_still_writes_sanitized_diagnostics qualify_smoke_test.mjs
//   test_structured_stdout_diagnostic_stays_valid_json             qualify_smoke_test.mjs
//   test_structured_stdout_preserves_escaped_unicode_safely        qualify_smoke_test.mjs
//   test_node_executable_fails_closed_without_node                 qualify_cmd_test.mjs
//   test_qualify_runs_the_node_smokes_with_the_pinned_inputs       qualify_cmd_test.mjs
//
// Tests that span both halves went to the half they mostly exercise:
// test_local_harness_must_match_the_exact_bridge_source (harness) and
// test_qualify_rejects_an_unprovenanced_local_candidate (argument parser) are
// in the qualification_*_test.mjs suites, as is every test that drives
// verify_attestation, including the speech/TTS evidence ones
// (test_mutated_asr_*, test_tts_lifecycle_evidence_*, ...).
// test_generated_wav_identity_measured_from_the_file and
// test_invalid_generated_wav_rejected are in wav_test.mjs.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const REPO_SCRIPTS_DIR = path.resolve(path.dirname(fs.realpathSync(fileURLToPath(import.meta.url))), '../../scripts');

// The pinned transcript the speech gate reads, as release_qualification_test.py
// reads it, so a transcript the real gate would reject cannot pass here.
export const DEFAULT_EXPECTED_TEXT = JSON.parse(
  fs.readFileSync(path.join(REPO_SCRIPTS_DIR, 'smoke', 'speech_to_text_fixture.json'), 'utf8'),
).expected_text;

// A fresh directory in the system temp root; parallel test files never share one.
export function makeTmp(prefix) {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

function chunk(id, body) {
  const header = Buffer.alloc(8);
  header.write(id, 0, 'latin1');
  header.writeUInt32LE(body.length, 4);
  return Buffer.concat([header, body, Buffer.alloc(body.length & 1)]);
}

// The bytes wave.open(path, "wb") writes for mono or multi-channel 16-bit PCM.
export function wavBytes(samples, { sampleRate = 24000, channels = 1 } = {}) {
  const fmt = Buffer.alloc(16);
  fmt.writeUInt16LE(1, 0);
  fmt.writeUInt16LE(channels, 2);
  fmt.writeUInt32LE(sampleRate, 4);
  fmt.writeUInt32LE(sampleRate * channels * 2, 8);
  fmt.writeUInt16LE(channels * 2, 12);
  fmt.writeUInt16LE(16, 14);
  const data = Buffer.alloc(samples.length * 2);
  samples.forEach((sample, index) => data.writeInt16LE(sample, index * 2));
  const body = Buffer.concat([Buffer.from('WAVE', 'latin1'), chunk('fmt ', fmt), chunk('data', data)]);
  return Buffer.concat([chunk('RIFF', body).subarray(0, 8), body]);
}

// release_qualification_test.write_wav: frames 0..7 repeated per channel.
export function writeWav(filePath, { sampleRate = 24000, channels = 1 } = {}) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const samples = Array.from({ length: 8 }, (_, frame) => Array(channels).fill(frame)).flat();
  fs.writeFileSync(filePath, wavBytes(samples, { sampleRate, channels }));
  return filePath;
}

// One speech smoke mode result, as smoke/speech_to_text.mjs reports it.
export function speechModeResult(memoryMode, runtimeMode, overrides = {}) {
  return {
    memoryMode,
    runtimeMode,
    elapsedMs: 10,
    phaseTimingsMs: {
      cancellationMs: 1,
      coldTranscriptMs: 1,
      modelLoadMs: 1,
      projectorLoadMs: 1,
      silenceMs: 1,
      warmTranscriptMs: 1,
    },
    coldTranscript: DEFAULT_EXPECTED_TEXT,
    warmTranscript: DEFAULT_EXPECTED_TEXT,
    cancellation: 'cancel:resolved:12',
    silenceTranscript: '',
    ...overrides,
  };
}

// One text-to-speech smoke mode result naming `audioArtifact`.
export function ttsModeResult(audioArtifact, overrides = {}) {
  return {
    memoryMode: 'wasm64',
    runtimeMode: 'direct',
    totalElapsedMs: 10,
    modelLoadMs: 3,
    projectorLoadMs: 2,
    synthesisMs: 5,
    framesGenerated: 1,
    peak: 0.5,
    rms: 0.1,
    truncated: false,
    cancellationTested: true,
    preAbortedTested: true,
    reuseSampleCount: 1,
    unloadTested: true,
    audioArtifact,
    ...overrides,
  };
}

// True while `pid` names a live process (a zombie counts as gone).
export function isAlive(pid) {
  try {
    process.kill(pid, 0);
  } catch {
    return false;
  }
  try {
    const status = fs.readFileSync(`/proc/${pid}/status`, 'utf8');
    return !/^State:\s+Z/m.test(status);
  } catch {
    return true;
  }
}

export async function waitUntilGone(pid, seconds = 5) {
  const deadline = Date.now() + seconds * 1000;
  while (Date.now() < deadline) {
    if (!isAlive(pid)) return true;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  return !isAlive(pid);
}

// Kill whatever a test left behind.
export function killAll(pids) {
  for (const pid of pids) {
    try {
      process.kill(pid, 'SIGKILL');
    } catch {
      // already gone
    }
  }
  pids.length = 0;
}
