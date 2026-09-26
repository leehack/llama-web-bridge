// Tests of speechPhase/ttsPhase in scripts/release/qualify.mjs, the ports of
// release_qualification.py's _speech_phase and _tts_phase. The first block is
// one test per Python test of the same name; the rest pin the error text and
// phase contents the attestation carries.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { afterEach, beforeEach, test } from 'node:test';

import { ContractError } from '../../scripts/release/errors.mjs';
import { PyFloat, isPyException, pyCanonicalJson } from '../../scripts/release/json.mjs';
import { modeKey, speechPhase, timing, ttsPhase } from '../../scripts/release/qualify.mjs';
import { readWavIdentity } from '../../scripts/release/wav.mjs';
import { DEFAULT_EXPECTED_TEXT, makeTmp, speechModeResult, ttsModeResult, wavBytes, writeWav } from './qualify_fixtures.mjs';

let tmp;
beforeEach(() => {
  tmp = makeTmp('qualify-phase-test-');
});
afterEach(() => {
  fs.rmSync(tmp, { recursive: true, force: true });
});

function rejects(run, message) {
  assert.throws(run, (error) => {
    assert.ok(error instanceof ContractError, `expected ContractError, got ${error}`);
    if (message !== undefined) assert.equal(error.message, message);
    return true;
  });
}

// --- the Python tests ---------------------------------------------------------

test('test_tts_phase_rejects_wav_path_escape_and_symlink', () => {
  const artifacts = path.join(tmp, 'tts-artifacts');
  fs.mkdirSync(artifacts);
  const outside = writeWav(path.join(tmp, 'outside.wav'));
  const result = ttsModeResult('../outside.wav');
  delete result.cancellationTested;
  delete result.preAbortedTested;
  delete result.reuseSampleCount;
  delete result.unloadTested;
  rejects(() => ttsPhase({ modeResults: [result] }, 1, artifacts));

  const link = path.join(artifacts, 'linked.wav');
  fs.symlinkSync(outside, link);
  result.audioArtifact = path.basename(link);
  rejects(() => ttsPhase({ modeResults: [result] }, 1, artifacts));
});

test('test_tts_phase_rejects_coerced_result_types', () => {
  const artifacts = path.join(tmp, 'typed-tts-artifacts');
  const wavPath = writeWav(path.join(artifacts, 'generated.wav'));
  const baseline = ttsModeResult(path.basename(wavPath));
  for (const key of ['cancellationTested', 'preAbortedTested', 'reuseSampleCount', 'unloadTested']) delete baseline[key];
  for (const [field, value] of [['framesGenerated', 1.5], ['truncated', 'false']]) {
    rejects(() => ttsPhase({ modeResults: [{ ...baseline, [field]: value }] }, 1, artifacts));
  }
});

test('test_tts_phase_measures_waveform_evidence_from_the_wav', () => {
  const artifacts = path.join(tmp, 'waveform-tts-artifacts');
  fs.mkdirSync(artifacts);
  const wavPath = path.join(artifacts, 'generated.wav');
  fs.writeFileSync(wavPath, wavBytes([0, 8192, -8192, 4096]));
  const measured = readWavIdentity(wavPath);
  const result = ttsModeResult('generated.wav', { peak: measured.peak, rms: measured.rms });
  const phase = ttsPhase({ modeResults: [result] }, 1, artifacts);
  assert.equal(phase.modes[0].peak.value, measured.peak);
  result.peak = 0.9;
  rejects(() => ttsPhase({ modeResults: [result] }, 1, artifacts));
});

test('test_speech_phase_rejects_a_rejected_cancellation_with_output', () => {
  const payload = { modeResults: [speechModeResult('wasm32', 'direct', { cancellation: 'cancel:rejected:12' })] };
  assert.throws(() => speechPhase(payload, 1), (error) => error instanceof ContractError
    && error.message.includes('rejected but reported'));
});

// --- speechPhase ------------------------------------------------------------------

test('a speech phase records every mode, sorted, with the summed total and the given RSS', () => {
  const payload = {
    ok: true,
    modeResults: [
      speechModeResult('wasm64', 'worker', { elapsedMs: 4 }),
      speechModeResult('wasm32', 'worker', { elapsedMs: 2, cancellation: 'cancel:rejected:0' }),
      speechModeResult('wasm64', 'direct', { elapsedMs: 3 }),
      speechModeResult('wasm32', 'direct', { elapsedMs: 1 }),
    ],
  };
  const phase = speechPhase(payload, 231329792);
  assert.deepEqual(phase.modes.map((mode) => [mode.memory_mode, mode.runtime_mode, mode.total_ms]), [
    ['wasm32', 'direct', 1], ['wasm32', 'worker', 2], ['wasm64', 'direct', 3], ['wasm64', 'worker', 4],
  ]);
  assert.equal(phase.total_ms, 10);
  assert.equal(phase.max_rss_bytes, 231329792);
  assert.deepEqual(Object.keys(phase), ['modes', 'total_ms', 'max_rss_bytes']);
  assert.deepEqual(phase.modes[1], {
    memory_mode: 'wasm32',
    runtime_mode: 'worker',
    total_ms: 2,
    phase_timings_ms: {
      cancellation: 1, cold_transcript: 1, model_load: 1, projector_load: 1, silence: 1, warm_transcript: 1,
    },
    cold_transcript: DEFAULT_EXPECTED_TEXT,
    warm_transcript: DEFAULT_EXPECTED_TEXT,
    cancellation_result: 'cancel:rejected:0',
    silence_transcript: '',
  });
});

test('speech results are rejected with release_qualification.py\'s messages', () => {
  const label = 'speech wasm32/direct';
  const cases = [
    [{}, 'speech gate did not report modeResults'],
    [{ modeResults: {} }, 'speech gate did not report modeResults'],
    [{ modeResults: [[]] }, 'speech gate reported an invalid mode result'],
    [{ modeResults: [speechModeResult(1, 'direct')] }, 'speech result is missing its mode identifiers'],
    [{ modeResults: [speechModeResult('wasm32', 'direct', { phaseTimingsMs: [] })] }, `${label} did not report per-phase timings`],
    [{ modeResults: [speechModeResult('wasm32', 'direct', { coldTranscript: '' })] }, `${label} did not report coldTranscript evidence`],
    [{ modeResults: [speechModeResult('wasm32', 'direct', { warmTranscript: null })] }, `${label} did not report warmTranscript evidence`],
    [{ modeResults: [speechModeResult('wasm32', 'direct', { cancellation: 7 })] }, `${label} did not report cancellation evidence`],
    [{ modeResults: [speechModeResult('wasm32', 'direct', { coldTranscript: 'wrong' })] }, `${label} cold transcript does not match expected fixture transcript`],
    [{ modeResults: [speechModeResult('wasm32', 'direct', { warmTranscript: 'wrong' })] }, `${label} warm transcript does not match expected fixture transcript`],
    [{ modeResults: [speechModeResult('wasm32', 'direct', { cancellation: 'cancelled' })] }, `${label} cancellation must match 'cancel:<resolved|rejected>:<canonical-count>', got 'cancelled'`],
    [{ modeResults: [speechModeResult('wasm32', 'direct', { silenceTranscript: 'hallucination' })] }, `${label} silence transcript must be empty`],
    [{ modeResults: [speechModeResult('wasm32', 'direct', { silenceTranscript: undefined })] }, `${label} silence transcript must be empty`],
    [{ modeResults: [speechModeResult('wasm32', 'direct', { elapsedMs: -1 })] }, `${label} is missing a non-negative elapsedMs timing`],
    [{ modeResults: [speechModeResult('wasm32', 'direct', { elapsedMs: 1.5 })] }, `${label} is missing a non-negative elapsedMs timing`],
    [{ modeResults: [speechModeResult('wasm32', 'direct', { elapsedMs: true })] }, `${label} is missing a non-negative elapsedMs timing`],
    [{ modeResults: [speechModeResult('wasm32', 'direct', { elapsedMs: new PyFloat(1) })] }, `${label} is missing a non-negative elapsedMs timing`],
    [{ modeResults: [speechModeResult('wasm32', 'direct', { phaseTimingsMs: { cancellationMs: 1 } })] }, `${label} is missing a non-negative coldTranscriptMs timing`],
  ];
  for (const [payload, message] of cases) rejects(() => speechPhase(payload, 1), message);
});

test('a speech payload that is not a dict raises AttributeError, as payload.get does', () => {
  assert.throws(() => speechPhase([], 1), (error) => isPyException(error, 'AttributeError')
    && error.message === "'list' object has no attribute 'get'");
});

test('speech timings keep Python ints exactly, past the safe-integer range', () => {
  const big = 2n ** 60n;
  const payload = { modeResults: [speechModeResult('wasm32', 'direct', { elapsedMs: big }), speechModeResult('wasm64', 'direct', { elapsedMs: big })] };
  const phase = speechPhase(payload, 1);
  assert.equal(phase.total_ms, 2n ** 61n);
  assert.equal(phase.modes[0].total_ms, big);
});

test('modeKey and timing report the missing field', () => {
  rejects(() => modeKey({ memoryMode: 'wasm32' }, 'x'), 'x result is missing its mode identifiers');
  assert.equal(timing({ a: 0 }, 'a', 'x'), 0);
  rejects(() => timing({}, 'a', 'x'), 'x is missing a non-negative a timing');
});

// --- ttsPhase -----------------------------------------------------------------------

function ttsArtifacts(samples = [0, 8192, -8192, 4096]) {
  const artifacts = path.join(tmp, 'tts');
  fs.mkdirSync(artifacts, { recursive: true });
  fs.writeFileSync(path.join(artifacts, 'generated.wav'), wavBytes(samples));
  return artifacts;
}

test('a text-to-speech phase records the measured waveform and WAV identity', () => {
  const artifacts = ttsArtifacts([32767, -32768, 0, 1]);
  const measured = readWavIdentity(path.join(artifacts, 'generated.wav'));
  assert.equal(measured.peak, 1);
  const phase = ttsPhase({
    modeResults: [
      ttsModeResult('generated.wav', { runtimeMode: 'worker', peak: 1, rms: measured.rms, totalElapsedMs: 7, reuseSampleCount: 1920 }),
      ttsModeResult('generated.wav', { peak: new PyFloat(1), rms: measured.rms, totalElapsedMs: 5, framesGenerated: 17 }),
    ],
  }, 42, artifacts);
  assert.deepEqual(phase.modes.map((mode) => mode.runtime_mode), ['direct', 'worker']);
  assert.equal(phase.total_ms, 12);
  assert.equal(phase.max_rss_bytes, 42);
  const [direct] = phase.modes;
  assert.deepEqual(Object.keys(direct), [
    'cancellation_tested', 'frames_generated', 'memory_mode', 'peak', 'phase_timings_ms', 'pre_aborted_tested',
    'reuse_sample_count', 'rms', 'runtime_mode', 'total_ms', 'truncated', 'unload_tested', 'wav',
  ]);
  assert.deepEqual(direct.phase_timings_ms, { model_load: 3, projector_load: 2, synthesis: 5 });
  assert.equal(direct.frames_generated, 17);
  assert.deepEqual(direct.wav, {
    sha256: measured.sha256,
    byte_length: measured.byte_length,
    channels: 1,
    bits_per_sample: 16,
    sample_rate: 24000,
    frame_count: 4,
  });
  // A peak of exactly 1.0 stays a float in the attestation.
  assert.ok(direct.peak instanceof PyFloat && direct.rms instanceof PyFloat);
  assert.match(pyCanonicalJson(direct), /"peak": 1\.0,/);
});

test('text-to-speech results are rejected with release_qualification.py\'s messages', () => {
  const artifacts = ttsArtifacts();
  const measured = readWavIdentity(path.join(artifacts, 'generated.wav'));
  const good = { peak: measured.peak, rms: measured.rms };
  fs.writeFileSync(path.join(artifacts, 'silent.wav'), wavBytes([0, 1, 0, -1]));
  const silent = readWavIdentity(path.join(artifacts, 'silent.wav'));
  const label = 'text-to-speech wasm64/direct';
  const cases = [
    [{}, 'text-to-speech gate did not report modeResults'],
    [{ modeResults: [1] }, 'text-to-speech gate reported an invalid mode result'],
    [{ modeResults: [ttsModeResult('generated.wav', { runtimeMode: null })] }, 'text-to-speech result is missing its mode identifiers'],
    [{ modeResults: [ttsModeResult('')] }, `${label} did not persist a generated WAV artifact`],
    [{ modeResults: [ttsModeResult(3)] }, `${label} did not persist a generated WAV artifact`],
    [{ modeResults: [ttsModeResult('sub/x.wav')] }, `${label} WAV artifact name is unsafe: 'sub/x.wav'`],
    [{ modeResults: [ttsModeResult('x.wav/')] }, `${label} WAV artifact name is unsafe: 'x.wav/'`],
    [{ modeResults: [ttsModeResult('./x.wav')] }, `${label} WAV artifact name is unsafe: './x.wav'`],
    [{ modeResults: [ttsModeResult('x.WAV')] }, `${label} WAV artifact name is unsafe: 'x.WAV'`],
    [{ modeResults: [ttsModeResult("it's.wav\n")] }, `${label} WAV artifact name is unsafe: "it's.wav\\n"`],
    [{ modeResults: [ttsModeResult('absent.wav')] }, `${label} WAV artifact is missing: absent.wav`],
    [{ modeResults: [ttsModeResult('nul\u0000.wav')] }, `${label} WAV artifact is missing: nul\u0000.wav`],
    [{ modeResults: [ttsModeResult('generated.wav', { framesGenerated: 0 })] }, `${label} framesGenerated must be a positive integer`],
    [{ modeResults: [ttsModeResult('generated.wav', { framesGenerated: true })] }, `${label} framesGenerated must be a positive integer`],
    [{ modeResults: [ttsModeResult('generated.wav', { truncated: true })] }, `${label} truncated must be false, got True`],
    [{ modeResults: [ttsModeResult('generated.wav', { truncated: undefined })] }, `${label} truncated must be false, got None`],
    [{ modeResults: [ttsModeResult('generated.wav', { truncated: 0 })] }, `${label} truncated must be false, got 0`],
    [{ modeResults: [ttsModeResult('generated.wav', { cancellationTested: 1 })] }, `${label} cancellationTested must be true`],
    [{ modeResults: [ttsModeResult('generated.wav', { preAbortedTested: 'true' })] }, `${label} preAbortedTested must be true`],
    [{ modeResults: [ttsModeResult('generated.wav', { reuseSampleCount: 0 })] }, `${label}.reuseSampleCount must be a positive integer`],
    [{ modeResults: [ttsModeResult('generated.wav', { unloadTested: false })] }, `${label} unloadTested must be true`],
    [{ modeResults: [ttsModeResult('generated.wav', { peak: 0 })] }, `${label} peak must be a positive finite number`],
    [{ modeResults: [ttsModeResult('generated.wav', { peak: true })] }, `${label} peak must be a positive finite number`],
    [{ modeResults: [ttsModeResult('generated.wav', { rms: new PyFloat(Infinity) })] }, `${label} rms must be a positive finite number`],
    [{ modeResults: [ttsModeResult('generated.wav', { rms: '0.1' })] }, `${label} rms must be a positive finite number`],
    [{ modeResults: [ttsModeResult('generated.wav', { ...good, peak: 0.9 })] }, `${label} reported waveform evidence does not match its WAV artifact`],
    [{ modeResults: [ttsModeResult('silent.wav', { peak: silent.peak, rms: silent.rms })] }, `${label} generated WAV is effectively silent`],
    [{ modeResults: [ttsModeResult('generated.wav', { ...good, synthesisMs: -1 })] }, `${label} is missing a non-negative synthesisMs timing`],
    [{ modeResults: [ttsModeResult('generated.wav', { ...good, totalElapsedMs: null })] }, `${label} is missing a non-negative totalElapsedMs timing`],
  ];
  for (const [payload, message] of cases) rejects(() => ttsPhase(payload, 1, artifacts), message);
});

test('a WAV that is not PCM16 mono 24 kHz is rejected by the WAV reader', () => {
  const artifacts = path.join(tmp, 'tts');
  writeWav(path.join(artifacts, 'stereo.wav'), { channels: 2 });
  rejects(() => ttsPhase({ modeResults: [ttsModeResult('stereo.wav')] }, 1, artifacts), 'generated audio must be mono PCM16');
});

test('an int peak past float range raises OverflowError, as math.isfinite does', () => {
  const artifacts = ttsArtifacts();
  assert.throws(() => ttsPhase({ modeResults: [ttsModeResult('generated.wav', { peak: 10n ** 400n })] }, 1, artifacts),
    (error) => isPyException(error, 'OverflowError') && error.message === 'int too large to convert to float');
});

test('the artifacts directory is resolved through symlinks, and a symlinked WAV is refused', () => {
  const artifacts = ttsArtifacts();
  const measured = readWavIdentity(path.join(artifacts, 'generated.wav'));
  const alias = path.join(tmp, 'alias');
  fs.symlinkSync(artifacts, alias);
  const phase = ttsPhase({ modeResults: [ttsModeResult('generated.wav', { peak: measured.peak, rms: measured.rms })] }, 1, alias);
  assert.equal(phase.modes[0].wav.sha256, measured.sha256);
  fs.symlinkSync(path.join(artifacts, 'generated.wav'), path.join(artifacts, 'link.wav'));
  rejects(() => ttsPhase({ modeResults: [ttsModeResult('link.wav', { peak: measured.peak, rms: measured.rms })] }, 1, artifacts),
    'text-to-speech wasm64/direct WAV artifact is missing: link.wav');
});
