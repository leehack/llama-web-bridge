// Every qualification constant, pinned to a literal. They were checked
// against scripts/release_qualification.py until the Node harness replaced
// it; these are the values that parity held, so a change to any of them is
// deliberate and shows up here. HARNESS_VERSION and HARNESS_SOURCES are
// pinned in qualification_harness_test.mjs, with the import closure they
// must equal.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import * as archive from '../../scripts/release/archive.mjs';
import { pyEquals, pyRepr } from '../../scripts/release/json.mjs';
import * as qualification from '../../scripts/release/qualification.mjs';

// Sets are listed sorted, tuples as arrays.
const PINNED = {
  QUALIFICATION_SCHEMA_VERSION: 2,
  ATTESTATION_TYPE: 'llama-web-bridge-automated-qualification',
  MAX_ATTESTATION_BYTES: 32768,
  MAX_COMPRESSION_RATIO: 100,
  MAX_CANDIDATE_MEMBER_BYTES: 67108864,
  MAX_CANDIDATE_TOTAL_BYTES: 268435456,
  CANDIDATE_ALLOWED_MEMBERS: [
    'llama_webgpu_bridge.d.ts',
    'llama_webgpu_bridge.js',
    'llama_webgpu_bridge_worker.js',
    'llama_webgpu_core.js',
    'llama_webgpu_core.wasm',
    'llama_webgpu_core_mem64.js',
    'llama_webgpu_core_mem64.wasm',
    'manifest.json',
    'sha256sums.txt',
  ],
  MAX_ATTESTATION_MEMBER_BYTES: 32768,
  MAX_ATTESTATION_TOTAL_BYTES: 32768,
  ATTESTATION_ALLOWED_MEMBERS: ['qualification-attestation.json'],
  ALLOWED_COMPRESS_TYPES: [0, 8],
  MAX_CANCELLATION_OUTPUT_CHARACTERS: 1000000,
  CANDIDATE_WORKFLOW_PATH: '.github/workflows/bridge_candidate.yml',
  CANDIDATE_ARTIFACT_NAME: 'exact-webgpu-bridge-dist',
  QUALIFICATION_WORKFLOW_PATH: '.github/workflows/bridge_qualification.yml',
  ATTESTATION_ARTIFACT_NAME: 'qualification-attestation',
  CANDIDATE_GATES: ['state_persistence', 'multimodal'],
  HEAVY_GATES: ['speech_to_text', 'text_to_speech'],
  REQUIRED_UNPROVEN_CAPABILITIES: {
    hardware_gpu_acceleration: 'unavailable-on-hosted-runners',
    real_device_intelligibility: 'unproven',
    real_device_playback: 'unproven',
    speaker_reference_fidelity: 'unproven',
    wasm32_text_to_speech: 'unsupported',
  },
  QUALIFICATION_EXECUTION: 'hosted-github-actions',
  QUALIFICATION_ENVIRONMENT_KEYS: ['cpu_count', 'execution', 'runner_arch', 'runner_os', 'total_memory_bytes'],
  MAX_QUALIFICATION_CPU_COUNT: 1024,
  MAX_QUALIFICATION_MEMORY_BYTES: 17592186044416,
  REQUIRED_SPEECH_MODES: [['wasm32', 'direct'], ['wasm32', 'worker'], ['wasm64', 'direct'], ['wasm64', 'worker']],
  REQUIRED_TTS_MODES: [['wasm64', 'direct'], ['wasm64', 'worker']],
  SPEECH_PHASE_KEYS: ['cancellation', 'cold_transcript', 'model_load', 'projector_load', 'silence', 'warm_transcript'],
  TTS_PHASE_KEYS: ['model_load', 'projector_load', 'synthesis'],
  TTS_WAV_SAMPLE_RATE: 24000,
  TTS_WAV_CHANNELS: 1,
  TTS_WAV_BITS_PER_SAMPLE: 16,
  STATE_SMOKE_MODEL_SHA256: '81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb',
  MULTIMODAL_MODEL_SHA256: 'bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517',
  MULTIMODAL_MMPROJ_SHA256: '56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453',
  SPEECH_MODEL_SHA256: 'bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971',
  SPEECH_MMPROJ_SHA256: '41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d',
  SPEECH_AUDIO_SHA256: 'f9b4440ac8393e47c14a6240e9739dea09b645bb1592b8f2dd48feb9666cea7f',
  TTS_MODEL_SHA256: '8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129',
  TTS_MMPROJ_SHA256: '6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2',
  EXPECTED_MODEL_PINS: {
    multimodal_mmproj_sha256: '56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453',
    multimodal_model_sha256: 'bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517',
    speech_audio_sha256: 'f9b4440ac8393e47c14a6240e9739dea09b645bb1592b8f2dd48feb9666cea7f',
    speech_mmproj_sha256: '41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d',
    speech_model_sha256: 'bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971',
    state_smoke_model_sha256: '81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb',
    tts_mmproj_sha256: '6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2',
    tts_model_sha256: '8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129',
  },
  SPEECH_SMOKE: 'smoke/speech_to_text.mjs',
  TTS_SMOKE: 'smoke/text_to_speech.mjs',
  QUALIFICATION_SMOKES: ['smoke/speech_to_text.mjs', 'smoke/text_to_speech.mjs'],
  ATTESTATION_KEYS: [
    'attestation_type',
    'bridge_repository',
    'bridge_source_sha',
    'candidate_artifact_id',
    'candidate_fingerprint',
    'candidate_gates',
    'candidate_run_attempt',
    'candidate_run_id',
    'candidate_run_url',
    'candidate_workflow_path',
    'emscripten_version',
    'harness_source_sha256',
    'harness_version',
    'heavy_gates',
    'model_pins',
    'native_commit',
    'native_manifest_sha256',
    'native_release_tag',
    'native_repository',
    'orchestrator_correlation_id',
    'phases',
    'qualification_environment',
    'qualification_run_attempt',
    'qualification_run_id',
    'qualification_run_url',
    'qualification_source_sha',
    'qualification_workflow_path',
    'release_rebuild',
    'release_tag',
    'schema_version',
    'unproven_capabilities',
    'upstream_commit',
    'upstream_repository',
    'upstream_tag',
  ],
  SPEECH_FIXTURE_FILE: 'smoke/speech_to_text_fixture.json',
  SPEECH_FIXTURE_KEYS: ['audio_sha256', 'audio_url', 'expected_text'],
  SPEECH_FIXTURE: {
    audio_sha256: 'f9b4440ac8393e47c14a6240e9739dea09b645bb1592b8f2dd48feb9666cea7f',
    audio_url: 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav',
    expected_text: 'Hmm. Oh, yeah, yeah. He wasn\'t even that big when I started listening to him, but and his solo music didn\'t do overly well, but he did very well when he started writing for other people.',
  },
  EXPECTED_SPEECH_TRANSCRIPT: 'hmm oh yeah yeah he wasn t even that big when i started listening to him but and his solo music didn t do overly well but he did very well when he started writing for other people',
  REDACTED_CREDENTIAL: '<redacted-credential>',
};

test('every qualification constant equals its pinned literal', () => {
  for (const [name, expected] of Object.entries(PINNED)) {
    let actual = Object.hasOwn(qualification, name) ? qualification[name] : undefined;
    assert.notEqual(actual, undefined, `${name} is not exported by qualification.mjs`);
    if (actual instanceof Set) actual = [...actual].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    if (Array.isArray(actual)) actual = actual.map((item) => (Array.isArray(item) ? [...item] : item));
    if (actual !== null && typeof actual === 'object' && !Array.isArray(actual)) actual = { ...actual };
    assert.ok(pyEquals(actual, expected), `${name}: ${pyRepr(actual)} != pinned ${pyRepr(expected)}`);
  }
  // The archive bounds qualification.mjs re-exports are archive.mjs's own.
  assert.equal(qualification.MAX_ATTESTATION_BYTES, archive.MAX_ATTESTATION_BYTES);
  assert.equal(qualification.CANDIDATE_ALLOWED_MEMBERS, archive.CANDIDATE_ALLOWED_MEMBERS);
});

test('the model pins are the eight pinned SHA-256 constants, by role', () => {
  assert.deepEqual({ ...qualification.EXPECTED_MODEL_PINS }, {
    multimodal_mmproj_sha256: qualification.MULTIMODAL_MMPROJ_SHA256,
    multimodal_model_sha256: qualification.MULTIMODAL_MODEL_SHA256,
    speech_audio_sha256: qualification.SPEECH_AUDIO_SHA256,
    speech_mmproj_sha256: qualification.SPEECH_MMPROJ_SHA256,
    speech_model_sha256: qualification.SPEECH_MODEL_SHA256,
    state_smoke_model_sha256: qualification.STATE_SMOKE_MODEL_SHA256,
    tts_mmproj_sha256: qualification.TTS_MMPROJ_SHA256,
    tts_model_sha256: qualification.TTS_MODEL_SHA256,
  });
  const pins = Object.values(qualification.EXPECTED_MODEL_PINS);
  assert.equal(new Set(pins).size, 8);
  for (const pin of pins) assert.match(pin, /^[0-9a-f]{64}$/);
  assert.equal(qualification.SPEECH_FIXTURE.audio_sha256, qualification.SPEECH_AUDIO_SHA256);
});
