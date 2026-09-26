// Parity of scripts/release/qualification.mjs's constants with
// scripts/release_qualification.py, which stays authoritative until the
// harness cutover. This imports the Python module and compares every constant
// the Node port carries; it goes away with the .py at the cutover.
//
// The redaction and validation regexes cannot be compared as values, so the
// Python pattern sources are pinned to the ones the Node patterns were
// translated from: a change on the Python side fails here until the port
// follows it.

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { test } from 'node:test';

import * as archive from '../../scripts/release/archive.mjs';
import { pyEquals, pyJsonLoads, pyRepr } from '../../scripts/release/json.mjs';
import * as qualification from '../../scripts/release/qualification.mjs';

const SCRIPTS_DIR = path.resolve(import.meta.dirname, '..', '..', 'scripts');

const PYTHON = String.raw`
import json, re, sys
sys.path.insert(0, sys.argv[1])
import release_qualification as rq

def plain(value):
    if isinstance(value, re.Pattern):
        return {"pattern": value.pattern, "flags": value.flags}
    if isinstance(value, (frozenset, set)):
        return sorted(plain(item) for item in value)
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"repr": repr(value)}

names = sorted(name for name in vars(rq) if re.fullmatch(r"_?[A-Z][A-Z0-9_]*", name))
print(json.dumps({name: plain(getattr(rq, name)) for name in names if not callable(getattr(rq, name))}))
`;

function pythonConstants() {
  const result = spawnSync('python3', ['-B', '-c', PYTHON, SCRIPTS_DIR], {
    encoding: 'utf8',
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' },
  });
  assert.equal(result.status, 0, result.stderr);
  return pyJsonLoads(result.stdout);
}

// Python constants the Node module exports under the same name.
const SHARED = [
  'QUALIFICATION_SCHEMA_VERSION', 'ATTESTATION_TYPE', 'HARNESS_VERSION',
  'MAX_ATTESTATION_BYTES', 'MAX_COMPRESSION_RATIO', 'MAX_CANDIDATE_MEMBER_BYTES', 'MAX_CANDIDATE_TOTAL_BYTES',
  'CANDIDATE_ALLOWED_MEMBERS', 'MAX_ATTESTATION_MEMBER_BYTES', 'MAX_ATTESTATION_TOTAL_BYTES',
  'ATTESTATION_ALLOWED_MEMBERS', 'ALLOWED_COMPRESS_TYPES', 'MAX_CANCELLATION_OUTPUT_CHARACTERS',
  'CANDIDATE_WORKFLOW_PATH', 'CANDIDATE_ARTIFACT_NAME', 'QUALIFICATION_WORKFLOW_PATH', 'ATTESTATION_ARTIFACT_NAME',
  'CANDIDATE_GATES', 'HEAVY_GATES', 'REQUIRED_UNPROVEN_CAPABILITIES', 'QUALIFICATION_EXECUTION',
  'QUALIFICATION_ENVIRONMENT_KEYS', 'MAX_QUALIFICATION_CPU_COUNT', 'MAX_QUALIFICATION_MEMORY_BYTES',
  'REQUIRED_SPEECH_MODES', 'REQUIRED_TTS_MODES', 'SPEECH_PHASE_KEYS', 'TTS_PHASE_KEYS',
  'TTS_WAV_SAMPLE_RATE', 'TTS_WAV_CHANNELS', 'TTS_WAV_BITS_PER_SAMPLE',
  'STATE_SMOKE_MODEL_SHA256', 'MULTIMODAL_MODEL_SHA256', 'MULTIMODAL_MMPROJ_SHA256', 'SPEECH_MODEL_SHA256',
  'SPEECH_MMPROJ_SHA256', 'SPEECH_AUDIO_SHA256', 'TTS_MODEL_SHA256', 'TTS_MMPROJ_SHA256', 'EXPECTED_MODEL_PINS',
  'SPEECH_SMOKE', 'TTS_SMOKE', 'QUALIFICATION_SMOKES', 'HARNESS_SOURCES', 'ATTESTATION_KEYS',
  'SPEECH_FIXTURE_FILE', 'SPEECH_FIXTURE_KEYS', 'SPEECH_FIXTURE', 'EXPECTED_SPEECH_TRANSCRIPT', 'REDACTED_CREDENTIAL',
];

// The Python patterns the Node port reproduces, and the private constants
// that are ported by value under another shape.
const PINNED = {
  _COMMIT_RE: { pattern: '^[0-9a-f]{40}$', flags: 32 },
  _RUN_ID_RE: { pattern: '^[1-9][0-9]*$', flags: 32 },
  _RUNNER_LABEL_RE: { pattern: '[A-Za-z0-9][A-Za-z0-9_.-]{0,31}', flags: 32 },
  _CANCELLATION_RESULT_RE: { pattern: '^cancel:(resolved|rejected):(0|[1-9][0-9]*)$', flags: 32 },
  _CREDENTIAL_NAME_PATTERN: '[A-Za-z0-9_.-]*(?:token|secret|password|credential|api[_-]?key)[A-Za-z0-9_.-]*',
  _CREDENTIAL_NAME_RE: {
    pattern: '(?i)\\A[A-Za-z0-9_.-]*(?:token|secret|password|credential|api[_-]?key)[A-Za-z0-9_.-]*\\Z',
    flags: 34,
  },
  _AUTHORIZATION_NAME_PATTERN: '(?:[A-Za-z0-9]+[._-])*authorization(?:[._-][A-Za-z0-9]+)*',
  _AUTHORIZATION_NAME_RE: { pattern: '(?i)\\A(?:[A-Za-z0-9]+[._-])*authorization(?:[._-][A-Za-z0-9]+)*\\Z', flags: 34 },
  _LLAMA_TOKEN_COUNTER_NAME_RE: { pattern: '(?i)\\A(?:[A-Za-z0-9_-]+\\.)*n_tokens(?:[_-][A-Za-z0-9_-]+)?\\Z', flags: 34 },
  _CREDENTIAL_NAME_EXCEPT_PLURAL_TOKENS_RE: {
    pattern: '(?i)\\A[A-Za-z0-9_.-]*(?:token(?!s(?:\\b|[_-]))|secret|password|credential|api[_-]?key)[A-Za-z0-9_.-]*\\Z',
    flags: 34,
  },
  _COUNTER_TEXT_VALUE_RE: { pattern: '\\A[0-9]+[.,;)}\\]]?\\Z', flags: 32 },
  _CREDENTIAL_ASSIGNMENT_RE: {
    pattern: '(?im)(?:(?P<key_quote>["\'])(?P<quoted_name>[A-Za-z0-9_.-]*(?:token|secret|password|credential|api[_-]?key)'
      + '[A-Za-z0-9_.-]*)(?P=key_quote)|\\b(?P<name>[A-Za-z0-9_.-]*(?:token|secret|password|credential|api[_-]?key)'
      + '[A-Za-z0-9_.-]*))\\s*[:=]\\s*(?P<value>"(?:\\\\.|[^"\\\\\\r\\n])*"|\'(?:\\\\.|[^\'\\\\\\r\\n])*\'|\\S+)',
    flags: 42,
  },
  _AUTHORIZATION_RE: {
    pattern: '(?im)(?:"(?:[A-Za-z0-9]+[._-])*authorization(?:[._-][A-Za-z0-9]+)*"|\'(?:[A-Za-z0-9]+[._-])*authorization'
      + '(?:[._-][A-Za-z0-9]+)*\'|\\b(?:[A-Za-z0-9]+[._-])*authorization(?:[._-][A-Za-z0-9]+)*)\\s*[:=]\\s*'
      + '(?:"(?:\\\\.|[^"\\\\\\r\\n])*"|\'(?:\\\\.|[^\'\\\\\\r\\n])*\'|[^\\r\\n]+)',
    flags: 42,
  },
  _AUTH_SCHEME_RE: { pattern: '(?im)\\b(?:bearer|basic)[ \\t]+\\S+[ \\t]*(?=\\r?$)', flags: 42 },
};

test('every Node constant equals its Python value', () => {
  const python = pythonConstants();
  for (const name of SHARED) {
    assert.ok(Object.hasOwn(python, name), `${name} is not a Python constant`);
    const expected = python[name];
    let actual = Object.hasOwn(qualification, name) ? qualification[name] : undefined;
    assert.notEqual(actual, undefined, `${name} is not exported by qualification.mjs`);
    if (actual instanceof Set) actual = [...actual].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    if (Array.isArray(actual)) actual = actual.map((item) => (Array.isArray(item) ? [...item] : item));
    if (actual !== null && typeof actual === 'object' && !Array.isArray(actual)) actual = { ...actual };
    assert.ok(pyEquals(actual, expected), `${name}: node ${pyRepr(actual)} != python ${pyRepr(expected)}`);
  }
  // The archive bounds qualification.mjs re-exports are archive.mjs's own.
  assert.equal(qualification.MAX_ATTESTATION_BYTES, archive.MAX_ATTESTATION_BYTES);
  assert.equal(qualification.CANDIDATE_ALLOWED_MEMBERS, archive.CANDIDATE_ALLOWED_MEMBERS);
});

test('the Python patterns are the ones the Node port was translated from', () => {
  const python = pythonConstants();
  for (const [name, expected] of Object.entries(PINNED)) {
    assert.ok(pyEquals(python[name], expected), `${name}: python ${pyRepr(python[name])} changed; port it and update the pin`);
  }
});

test('no Python constant is left unaccounted for', () => {
  const python = pythonConstants();
  // Private constants of the archive and WAV code, ported in archive.mjs and
  // wav.mjs, and the typing alias.
  const portedElsewhere = new Set([
    '_DATA_DESCRIPTOR_SIGNATURE', '_DATA_DESCRIPTOR_STRUCT', '_END_OF_CENTRAL_DIRECTORY_SIGNATURE',
    '_END_OF_CENTRAL_DIRECTORY_STRUCT', '_EXTRACT_CHUNK_BYTES', '_LOCAL_HEADER_SIGNATURE', '_LOCAL_HEADER_STRUCT',
    '_MAX_CENTRAL_DIRECTORY_BYTES', '_ZIP_DATA_DESCRIPTOR_FLAG', '_ZIP_ENCRYPTED_FLAG', '_ZIP_ENCRYPTION_FLAGS',
    '_ZIP_MASKED_HEADER_FLAG', '_ZIP_STRONG_ENCRYPTION_FLAG', '_ZIP_UTF8_NAME_FLAG', '_PublishedManifestContract',
    // Imported from release_contract / release_publication_state.
    'AUTOMATED_QUALIFICATION_REQUIRED', 'BRIDGE_REPOSITORY', 'NATIVE_REPOSITORY', 'UNPROVEN_CAPABILITIES',
    'PUBLICATION_FILES', 'COMMANDS',
  ]);
  const accounted = new Set([...SHARED, ...Object.keys(PINNED), ...portedElsewhere]);
  const missing = Object.keys(python).filter((name) => !accounted.has(name));
  assert.deepEqual(missing, [], 'a new Python constant needs a Node port and a parity check');
});
