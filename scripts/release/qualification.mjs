#!/usr/bin/env node
// Digest-bound automated release qualification and attestation validation,
// the Node port of scripts/release_qualification.py (everything but running
// the heavy gates, which lives in ./qualify.mjs).
//
// The heavy real-model Qwen3-ASR and Qwen3-TTS gates run in their own hosted
// qualification workflow rather than inside the candidate build. That run
// consumes the exact hosted candidate artifact -- it never rebuilds it -- runs
// the heavy gates, and emits a canonical attestation bound to the candidate
// digest and to every provenance identity recorded in the candidate manifest.
// Publication refuses to publish unless an exact successful qualification run
// attested the exact candidate it is about to publish.
//
// The attestation records the environment the gates actually ran in and the
// lanes that stay unproven or unavailable there, so an automatic publication
// never claims coverage nothing executed.
//
// Every validator accepts and rejects exactly what the Python module does,
// with the same ContractError text; values follow the JSON value model of
// ./json.mjs (int vs PyFloat, Python equality and repr). The attestation bytes
// are Python's json.dumps(indent=2, sort_keys=True) + "\n". Keyword arguments
// of the Python functions are camelCase option objects here.

import { spawnSync } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';

import {
  ALLOWED_COMPRESS_TYPES,
  ATTESTATION_ALLOWED_MEMBERS,
  CANDIDATE_ALLOWED_MEMBERS,
  MAX_ATTESTATION_BYTES,
  MAX_ATTESTATION_MEMBER_BYTES,
  MAX_ATTESTATION_TOTAL_BYTES,
  MAX_CANDIDATE_MEMBER_BYTES,
  MAX_CANDIDATE_TOTAL_BYTES,
  MAX_COMPRESSION_RATIO,
  artifactArchiveBounds,
  extractFlatArtifactArchive,
} from './archive.mjs';
import {
  ArgparseExit, SystemExitError, parseCommandLine, progName, runCliAsync,
} from './cli.mjs';
import {
  AUTOMATED_QUALIFICATION_REQUIRED,
  BRIDGE_REPOSITORY,
  NATIVE_REPOSITORY,
  UNPROVEN_CAPABILITIES,
  ContractError,
  rejectDuplicateKeys,
  requireCorrelationId,
  requireSha256,
} from './contract.mjs';
import {
  JSONDecodeError, PY_WHITESPACE, PY_NON_SPACE_CLASS, PY_WORD, PyException, PyFloat, compareCodePoints,
  isDict, isPyException, isPyFloat, isPyInt, pyDecodeUtf8, pyDict, pyEquals, pyGet, pyHasKey, pyHookRepr,
  pyIsDir, pyIsFile, pyIsSymlink, pyItems, pyJoinPath, pyJsonDumps, pyJsonLoads, pyKeys,
  pyListdir, pyOSError, pyPath, pyReadBytes, pyReadText, pyRepr, pyStr, pyStrip, pyTypeName,
  pyUniversalNewlines,
} from './json.mjs';
import {
  CandidateIdentity,
  PUBLICATION_FILES,
  validateCandidate,
  validatePublishedCandidate,
} from './publication_state.mjs';
import { isOSError, osErrorString, pyResolve } from './python_compat.mjs';
import {
  TTS_WAV_BITS_PER_SAMPLE,
  TTS_WAV_CHANNELS,
  TTS_WAV_SAMPLE_RATE,
  readWavIdentity,
} from './wav.mjs';

export { ContractError };
export {
  ALLOWED_COMPRESS_TYPES,
  ATTESTATION_ALLOWED_MEMBERS,
  CANDIDATE_ALLOWED_MEMBERS,
  MAX_ATTESTATION_BYTES,
  MAX_ATTESTATION_MEMBER_BYTES,
  MAX_ATTESTATION_TOTAL_BYTES,
  MAX_CANDIDATE_MEMBER_BYTES,
  MAX_CANDIDATE_TOTAL_BYTES,
  MAX_COMPRESSION_RATIO,
  TTS_WAV_BITS_PER_SAMPLE,
  TTS_WAV_CHANNELS,
  TTS_WAV_SAMPLE_RATE,
  artifactArchiveBounds,
  extractFlatArtifactArchive,
  readWavIdentity,
};

export const QUALIFICATION_SCHEMA_VERSION = 2;
export const ATTESTATION_TYPE = 'llama-web-bridge-automated-qualification';
// 4.0.0: the heavy gates run the Node smokes (scripts/*_browser_smoke.mjs).
export const HARNESS_VERSION = '4.0.0';

// --- small Python helpers ---------------------------------------------------

// re.fullmatch(pattern, value) for an anchored pattern, which raises
// TypeError for a non-str.
function fullmatch(pattern, value) {
  if (typeof value !== 'string') {
    throw new PyException('TypeError', `expected string or bytes-like object, got '${pyTypeName(value)}'`);
  }
  return pattern.exec(value);
}

// isinstance(value, Mapping) for JSON-shaped values.
function isMapping(value) {
  return isDict(value) || value instanceof Map;
}

// mapping[key], raising KeyError like Python.
function getItem(mapping, key) {
  if (!pyHasKey(mapping, key)) throw new PyException('KeyError', pyRepr(key));
  return pyGet(mapping, key);
}

// An int's exact value (bool is 0 or 1).
function intValue(value) {
  return BigInt(value);
}

// A number's value for an ordered comparison with a small float bound.
function numericValue(value) {
  return value instanceof PyFloat ? value.value : Number(value);
}

// copy.deepcopy of a JSON-shaped value: dicts keep their key order, floats
// stay PyFloat.
export function pyDeepCopy(value) {
  if (Array.isArray(value)) return value.map(pyDeepCopy);
  if (value instanceof Map) return new Map([...value].map(([key, item]) => [key, pyDeepCopy(item)]));
  if (isDict(value)) return pyDict(pyItems(value).map(([key, item]) => [key, pyDeepCopy(item)]));
  return value;
}

// Python truthiness of a JSON-shaped value.
function pyTruthy(value) {
  if (value === null || value === undefined || value === false) return false;
  if (typeof value === 'string') return value.length > 0;
  if (typeof value === 'number') return value !== 0;
  if (typeof value === 'bigint') return value !== 0n;
  if (value instanceof PyFloat) return value.value !== 0;
  if (Array.isArray(value)) return value.length > 0;
  if (isMapping(value)) return pyKeys(value).length > 0;
  return true;
}

// sorted() of str values: code point order.
function sortedStrings(values) {
  return [...values].sort(compareCodePoints);
}

// repr() of a list of (str, str) tuples.
function reprPairs(pairs) {
  return `[${pairs.map(([left, right]) => `(${pyRepr(left)}, ${pyRepr(right)})`).join(', ')}]`;
}

function comparePairs(left, right) {
  return compareCodePoints(left[0], right[0]) || compareCodePoints(left[1], right[1]);
}

// --- transcripts and cancellation ---------------------------------------------

// Python's re.IGNORECASE matches these beyond the ASCII case pair of a
// letter (its simple case mapping plus sre's equivalence fixes); JavaScript's
// `i` flag uses Unicode case folding and misses U+0130 and U+0131. Patterns
// here spell the case variants out instead of using the `i` flag.
// U+0130/U+0131 dotted/dotless i, U+017F long s, U+212A Kelvin sign.
const IGNORECASE_EXTRA = Object.freeze({ i: '\u0130\u0131', k: '\u212A', s: '\u017F' });
const IGNORECASE_CLASS_EXTRA = '\u0130\u0131\u017F\u212A';

// A case-insensitive literal (letters, '_', '<', '>').
function ci(word) {
  return Array.from(word, (char) => (/[a-z]/u.test(char)
    ? `[${char}${char.toUpperCase()}${IGNORECASE_EXTRA[char] ?? ''}]`
    : char)).join('');
}

// Python's \s, \S, \w (PY_WORD) and \b for str patterns, and `.` without
// DOTALL.
const PY_SPACE = `[${PY_WHITESPACE}]`;
const PY_WORD_BOUNDARY = `(?:(?<=${PY_WORD})(?!${PY_WORD})|(?<!${PY_WORD})(?=${PY_WORD}))`;
const PY_ANY = String.raw`[^\n]`;

const LANGUAGE_PREFIX_RE = new RegExp(
  `^${PY_SPACE}*${ci('language')}${PY_SPACE}+[^<\\r\\n]+?${PY_SPACE}*${ci('<asr_text>')}${PY_SPACE}*`,
  'u',
);
const ASR_TEXT_PREFIX_RE = new RegExp(`^${PY_SPACE}*${ci('<asr_text>')}${PY_SPACE}*`, 'u');

export function normalizeTranscript(value) {
  let text = pyTruthy(value) ? pyStr(value) : '';
  text = text.replace(LANGUAGE_PREFIX_RE, '');
  text = text.replace(ASR_TEXT_PREFIX_RE, '');
  text = text.toLowerCase();
  return pyStrip(text.replace(/[^a-z0-9]+/gu, ' '));
}

export const MAX_CANCELLATION_OUTPUT_CHARACTERS = 1_000_000;
const CANCELLATION_RESULT_RE = /^cancel:(resolved|rejected):(0|[1-9][0-9]*)$/u;

// [state, count] of a cancellation result 'cancel:<resolved|rejected>:<n>'.
export function parseCancellationResult(value, label) {
  if (typeof value !== 'string' || !value) throw new ContractError(`${label} must be a non-empty string`);
  const match = CANCELLATION_RESULT_RE.exec(value);
  if (match === null) {
    throw new ContractError(`${label} must match 'cancel:<resolved|rejected>:<canonical-count>', got ${pyRepr(value)}`);
  }
  const countText = match[2];
  if (countText.length > String(MAX_CANCELLATION_OUTPUT_CHARACTERS).length) {
    throw new ContractError(`${label} output count exceeds the qualification bound`);
  }
  const count = Number(countText);
  if (count > MAX_CANCELLATION_OUTPUT_CHARACTERS) throw new ContractError(`${label} output count exceeds the qualification bound`);
  const state = match[1];
  if (state === 'rejected' && count !== 0) throw new ContractError(`${label} rejected but reported ${count} characters of output`);
  return [state, count];
}

// --- workflow, artifact, gate and phase constants ---------------------------------

export const CANDIDATE_WORKFLOW_PATH = '.github/workflows/bridge_candidate.yml';
export const CANDIDATE_ARTIFACT_NAME = 'exact-webgpu-bridge-dist';
export const QUALIFICATION_WORKFLOW_PATH = '.github/workflows/bridge_qualification.yml';
export const ATTESTATION_ARTIFACT_NAME = 'qualification-attestation';

// Gates the candidate build run proves. They are copied from the candidate
// manifest and re-checked, never asserted by the qualification harness itself.
export const CANDIDATE_GATES = Object.freeze(['state_persistence', 'multimodal']);
// Heavy real-model gates this harness proves against the exact candidate
// artifact in its own hosted qualification run.
export const HEAVY_GATES = Object.freeze(['speech_to_text', 'text_to_speech']);

// Qualification proves programmatic transcript, lifecycle, and WAV container
// correctness on a hosted runner. Nothing listens to generated audio, and no
// hosted runner exposes a real GPU, so those lanes stay explicitly uncovered.
export const REQUIRED_UNPROVEN_CAPABILITIES = Object.freeze({ ...UNPROVEN_CAPABILITIES });

// The gates must execute on hosted GitHub Actions infrastructure. Pinning the
// recorded execution mode is what stops a machine-produced attestation from
// claiming coverage the automatic pipeline never observed.
export const QUALIFICATION_EXECUTION = 'hosted-github-actions';
export const QUALIFICATION_ENVIRONMENT_KEYS = Object.freeze([
  'cpu_count',
  'execution',
  'runner_arch',
  'runner_os',
  'total_memory_bytes',
]);
export const MAX_QUALIFICATION_CPU_COUNT = 1024;
export const MAX_QUALIFICATION_MEMORY_BYTES = 2 ** 44;
const RUNNER_LABEL_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$/u;

export const REQUIRED_SPEECH_MODES = Object.freeze([
  Object.freeze(['wasm32', 'direct']),
  Object.freeze(['wasm32', 'worker']),
  Object.freeze(['wasm64', 'direct']),
  Object.freeze(['wasm64', 'worker']),
]);
// The validated Qwen3-TTS pair is memory64-only in practice; wasm32 is not a
// product path for it, so it is deliberately absent rather than silently skipped.
export const REQUIRED_TTS_MODES = Object.freeze([
  Object.freeze(['wasm64', 'direct']),
  Object.freeze(['wasm64', 'worker']),
]);

export const SPEECH_PHASE_KEYS = Object.freeze([
  'cancellation',
  'cold_transcript',
  'model_load',
  'projector_load',
  'silence',
  'warm_transcript',
]);
export const TTS_PHASE_KEYS = Object.freeze(['model_load', 'projector_load', 'synthesis']);

export const STATE_SMOKE_MODEL_SHA256 = '81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb';
export const MULTIMODAL_MODEL_SHA256 = 'bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517';
export const MULTIMODAL_MMPROJ_SHA256 = '56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453';
export const SPEECH_MODEL_SHA256 = 'bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971';
export const SPEECH_MMPROJ_SHA256 = '41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d';
export const SPEECH_AUDIO_SHA256 = 'f9b4440ac8393e47c14a6240e9739dea09b645bb1592b8f2dd48feb9666cea7f';
export const TTS_MODEL_SHA256 = '8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129';
export const TTS_MMPROJ_SHA256 = '6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2';

export const EXPECTED_MODEL_PINS = Object.freeze({
  multimodal_mmproj_sha256: MULTIMODAL_MMPROJ_SHA256,
  multimodal_model_sha256: MULTIMODAL_MODEL_SHA256,
  speech_audio_sha256: SPEECH_AUDIO_SHA256,
  speech_mmproj_sha256: SPEECH_MMPROJ_SHA256,
  speech_model_sha256: SPEECH_MODEL_SHA256,
  state_smoke_model_sha256: STATE_SMOKE_MODEL_SHA256,
  tts_mmproj_sha256: TTS_MMPROJ_SHA256,
  tts_model_sha256: TTS_MODEL_SHA256,
});

// The Node smokes qualify runs with the candidate's locked Playwright.
export const SPEECH_SMOKE = 'speech_to_text_browser_smoke.mjs';
export const TTS_SMOKE = 'text_to_speech_browser_smoke.mjs';
export const QUALIFICATION_SMOKES = Object.freeze([SPEECH_SMOKE, TTS_SMOKE]);

// Every scripts/ file the gates execute or read at the candidate source, the
// same list as release_qualification.py's HARNESS_SOURCES while the Python
// harness is authoritative (so this verifier accepts the attestations it
// produces): the qualification command with its Python import closure, the
// speech and text-to-speech smokes it runs, the state-persistence and
// multimodal smokes the candidate build runs as its hosted gates, the module
// all four import, and the speech fixture. A name is a path relative to the
// scripts/ directory and may contain '/'. The digest binds an attestation to
// the exact harness that produced it, so publication can prove the harness
// that ran is the exact bridge source being published.
export const HARNESS_SOURCES = Object.freeze([
  'browser_smoke_support.mjs',
  'generate_release_manifest.py',
  'multimodal_browser_smoke.mjs',
  'release_contract.py',
  'release_publication_state.py',
  'release_qualification.py',
  'speech_to_text_browser_smoke.mjs',
  'speech_to_text_fixture.json',
  'state_persistence_browser_smoke.mjs',
  'text_to_speech_browser_smoke.mjs',
]);

const COMMIT_RE = /^[0-9a-f]{40}$/u;
const RUN_ID_RE = /^[1-9][0-9]*$/u;

export const ATTESTATION_KEYS = Object.freeze([
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
]);

// --- strict JSON ----------------------------------------------------------------

// _reject_nonstandard_json_constant: the parse_constant hook.
export function rejectNonstandardJsonConstant(value) {
  throw new ContractError(`non-standard JSON numeric constant: ${value}`);
}

// json.loads(text, object_pairs_hook=_reject_duplicate_keys,
// parse_constant=_reject_nonstandard_json_constant).
function strictLoads(text) {
  return pyJsonLoads(text, { objectPairsHook: rejectDuplicateKeys, parseConstant: rejectNonstandardJsonConstant });
}

// Path(text).name.
function pyPathName(text) {
  const normalized = pyPath(text);
  if (normalized === '.' || /^\/+$/u.test(normalized)) return '';
  return normalized.slice(normalized.lastIndexOf('/') + 1);
}

// The speech gate's fixture pin and transcript, which the speech smoke reads
// as its defaults. One file for both, because a second pinned copy here could
// drift into silently disagreeing with the gate about what a passing
// transcript is.
export const SPEECH_FIXTURE_FILE = 'speech_to_text_fixture.json';
export const SPEECH_FIXTURE_KEYS = Object.freeze(['audio_sha256', 'audio_url', 'expected_text']);

export function loadSpeechFixture(fixturePath) {
  const name = pyPathName(String(fixturePath));
  let fixture;
  try {
    fixture = strictLoads(pyReadText(String(fixturePath)));
  } catch (error) {
    if (error instanceof ContractError || isPyException(error, 'OSError', 'ValueError')) {
      throw new ContractError(`could not read ${name}: ${error.message}`);
    }
    throw error;
  }
  const keys = isDict(fixture) ? sortedStrings(pyKeys(fixture)) : null;
  if (keys === null || keys.length !== SPEECH_FIXTURE_KEYS.length || keys.some((key, index) => key !== SPEECH_FIXTURE_KEYS[index])) {
    throw new ContractError(`${name} must hold exactly ${SPEECH_FIXTURE_KEYS.join(', ')}`);
  }
  for (const key of SPEECH_FIXTURE_KEYS) {
    const value = fixture[key];
    if (typeof value !== 'string' || !pyStrip(value)) throw new ContractError(`${name} ${key} must be a non-empty string`);
  }
  return fixture;
}

// The fixture sits in scripts/, beside the Python harness that reads it.
export const SPEECH_FIXTURE = Object.freeze(loadSpeechFixture(path.join(import.meta.dirname, '..', SPEECH_FIXTURE_FILE)));
export const EXPECTED_SPEECH_TRANSCRIPT = normalizeTranscript(SPEECH_FIXTURE.expected_text);

// --- the attestation artifact ---------------------------------------------------

export function parseAttestationJson(rawJson) {
  let payload;
  try {
    payload = strictLoads(rawJson);
  } catch (error) {
    if (error instanceof JSONDecodeError || isPyException(error, 'UnicodeDecodeError')) {
      throw new ContractError(`malformed attestation JSON: ${error.message}`, { cause: error });
    }
    throw error;
  }
  if (!isDict(payload)) throw new ContractError('attestation root must be a JSON object');
  return payload;
}

// json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n".
export function canonicalJson(payload, { maxNesting = Infinity } = {}) {
  try {
    return `${pyJsonDumps(payload, { indent: 2, sortKeys: true, allowNan: false, maxNesting })}\n`;
  } catch (error) {
    if (isPyException(error, 'TypeError', 'ValueError')) {
      throw new ContractError(`attestation is not canonical JSON data: ${error.message}`, { cause: error });
    }
    throw error;
  }
}

// Containers canonical_json encodes below `verify-attestation`'s call chain
// before Python's recursion limit (1000 frames) raises RecursionError, which
// the CLI does not catch (measured on 3.12.3 and 3.12.11: 992 pass, 993 fail;
// a float costs one more).
export const ATTESTATION_CANONICAL_NESTING = 992;

// Read an attestation artifact, refusing anything but its canonical form.
//
// A re-ordered, re-indented, or padded variant of an otherwise valid
// attestation is refused rather than silently normalized, so the bytes a
// qualification run uploaded are the exact bytes publication verifies.
export function loadAttestationFile(attestationPath) {
  let raw;
  try {
    raw = pyReadBytes(String(attestationPath));
  } catch (error) {
    if (isPyException(error, 'OSError')) throw new ContractError(`could not read attestation artifact: ${error.message}`, { cause: error });
    throw error;
  }
  if (raw.length > MAX_ATTESTATION_BYTES) {
    throw new ContractError(`attestation artifact is ${raw.length} bytes; the bound is ${MAX_ATTESTATION_BYTES}`);
  }
  let text;
  try {
    text = pyDecodeUtf8(raw);
  } catch (error) {
    if (isPyException(error, 'UnicodeDecodeError')) throw new ContractError(`attestation artifact is not UTF-8: ${error.message}`, { cause: error });
    throw error;
  }
  const payload = parseAttestationJson(text);
  if (text !== canonicalJson(payload, { maxNesting: ATTESTATION_CANONICAL_NESTING })) {
    throw new ContractError('attestation artifact is not the canonical serialization of its own content');
  }
  return payload;
}

// --- diagnostic redaction ------------------------------------------------------------

export const REDACTED_CREDENTIAL = '<redacted-credential>';

// [A-Za-z0-9_.-] under re.IGNORECASE, and the other name classes.
const NAME_CHAR = `[A-Za-z0-9_.\\-${IGNORECASE_CLASS_EXTRA}]`;
const ALNUM_CHAR = `[A-Za-z0-9${IGNORECASE_CLASS_EXTRA}]`;
const COUNTER_NAME_CHAR = `[A-Za-z0-9_\\-${IGNORECASE_CLASS_EXTRA}]`;
const CREDENTIAL_WORDS = `${ci('token')}|${ci('secret')}|${ci('password')}|${ci('credential')}|${ci('api')}[_\\-]?${ci('key')}`;

const CREDENTIAL_NAME_PATTERN = `${NAME_CHAR}*(?:${CREDENTIAL_WORDS})${NAME_CHAR}*`;
const CREDENTIAL_NAME_RE = new RegExp(`^${CREDENTIAL_NAME_PATTERN}$`, 'u');
const AUTHORIZATION_NAME_PATTERN = `(?:${ALNUM_CHAR}+[._\\-])*${ci('authorization')}(?:[._\\-]${ALNUM_CHAR}+)*`;
const AUTHORIZATION_NAME_RE = new RegExp(`^${AUTHORIZATION_NAME_PATTERN}$`, 'u');
const LLAMA_TOKEN_COUNTER_NAME_RE = new RegExp(
  `^(?:${COUNTER_NAME_CHAR}+\\.)*${ci('n_tokens')}(?:[_\\-]${COUNTER_NAME_CHAR}+)?$`,
  'u',
);
const CREDENTIAL_NAME_EXCEPT_PLURAL_TOKENS_RE = new RegExp(
  `^${NAME_CHAR}*(?:${ci('token')}(?!${ci('s')}(?:${PY_WORD_BOUNDARY}|[_\\-]))|${ci('secret')}|${ci('password')}`
  + `|${ci('credential')}|${ci('api')}[_\\-]?${ci('key')})${NAME_CHAR}*$`,
  'u',
);
const COUNTER_TEXT_VALUE_RE = /^[0-9]+[.,;)}\]]?$/u;
const QUOTED_VALUE = String.raw`"(?:\\${PY_ANY}|[^"\\\r\n])*"|'(?:\\${PY_ANY}|[^'\\\r\n])*'`;
const CREDENTIAL_ASSIGNMENT_RE = new RegExp(
  `(?:(?<key_quote>["'])(?<quoted_name>${CREDENTIAL_NAME_PATTERN})\\k<key_quote>`
  + `|${PY_WORD_BOUNDARY}(?<name>${CREDENTIAL_NAME_PATTERN}))${PY_SPACE}*[:=]${PY_SPACE}*`
  + `(?<value>${QUOTED_VALUE}|${PY_NON_SPACE_CLASS}+)`,
  'gu',
);
const AUTHORIZATION_RE = new RegExp(
  `(?:"${AUTHORIZATION_NAME_PATTERN}"|'${AUTHORIZATION_NAME_PATTERN}'|${PY_WORD_BOUNDARY}${AUTHORIZATION_NAME_PATTERN})`
  + `${PY_SPACE}*[:=]${PY_SPACE}*(?:${QUOTED_VALUE}|[^\\r\\n]+)`,
  'gu',
);
// (?m) `$`: the end of the text or just before a '\n'.
const AUTH_SCHEME_RE = new RegExp(
  `${PY_WORD_BOUNDARY}(?:${ci('bearer')}|${ci('basic')})[ \\t]+${PY_NON_SPACE_CLASS}+[ \\t]*(?=\\r?(?:\\n|$))`,
  'gu',
);
const URL_RE = new RegExp(`${ci('https')}?:(?:\\\\/|/){2}[^${PY_WHITESPACE}"'<>]+`, 'gu');

function isLlamaTokenCounterName(name) {
  return LLAMA_TOKEN_COUNTER_NAME_RE.test(name) && !CREDENTIAL_NAME_EXCEPT_PLURAL_TOKENS_RE.test(name);
}

function redactCredentialAssignment(match, groups) {
  const name = groups.name || groups.quoted_name;
  if (isLlamaTokenCounterName(name) && COUNTER_TEXT_VALUE_RE.test(groups.value)) return match;
  return REDACTED_CREDENTIAL;
}

// --- urllib.parse.urlsplit, as the URL redaction relies on it ----------------------

const SCHEME_CHARS = /^[A-Za-z0-9+\-.]+$/u;
const HEX_DIGITS = /^[0-9a-fA-F]*$/u;

// ipaddress.IPv4Address(text) accepts it.
function isIPv4(text) {
  if (!text || text.includes('/')) return false;
  const octets = text.split('.');
  if (octets.length !== 4) return false;
  return octets.every((octet) => /^[0-9]{1,3}$/u.test(octet) && (octet === '0' || octet[0] !== '0') && Number(octet) <= 255);
}

// ipaddress.IPv6Address(text) accepts it.
function isIPv6(text) {
  if (text.includes('/')) return false;
  const percent = text.indexOf('%');
  let address = text;
  if (percent >= 0) {
    const scope = text.slice(percent + 1);
    if (!scope || scope.includes('%')) return false;
    address = text.slice(0, percent);
  }
  if (!address) return false;
  const parts = address.split(':');
  if (parts.length < 3) return false;
  if (parts.at(-1).includes('.')) {
    if (!isIPv4(parts.pop())) return false;
    parts.push('0', '0');
  }
  if (parts.length > 9) return false;
  let skipIndex = null;
  for (let index = 1; index < parts.length - 1; index += 1) {
    if (!parts[index]) {
      if (skipIndex !== null) return false;
      skipIndex = index;
    }
  }
  let partsHi;
  let partsLo;
  if (skipIndex !== null) {
    partsHi = skipIndex;
    partsLo = parts.length - skipIndex - 1;
    if (!parts[0]) {
      partsHi -= 1;
      if (partsHi) return false;
    }
    if (!parts.at(-1)) {
      partsLo -= 1;
      if (partsLo) return false;
    }
    if (8 - (partsHi + partsLo) < 1) return false;
  } else {
    if (parts.length !== 8 || !parts[0] || !parts.at(-1)) return false;
    partsHi = parts.length;
    partsLo = 0;
  }
  const parsed = [...parts.slice(0, partsHi), ...(partsLo ? parts.slice(-partsLo) : [])];
  return parsed.every((hextet) => HEX_DIGITS.test(hextet) && hextet.length <= 4 && hextet.length > 0);
}

function partition(text, separator) {
  const index = text.indexOf(separator);
  return index < 0 ? [text, '', ''] : [text.slice(0, index), separator, text.slice(index + separator.length)];
}

function rpartition(text, separator) {
  const index = text.lastIndexOf(separator);
  return index < 0 ? ['', '', text] : [text.slice(0, index), separator, text.slice(index + separator.length)];
}

// urllib.parse._check_bracketed_netloc: raises ValueError.
function checkBracketedNetloc(netloc) {
  const hostnameAndPort = rpartition(netloc, '@')[2];
  const [beforeBracket, haveOpenBracket, bracketed] = partition(hostnameAndPort, '[');
  let hostname;
  if (haveOpenBracket) {
    if (beforeBracket) throw new PyException('ValueError', 'Invalid IPv6 URL');
    let port;
    [hostname, , port] = partition(bracketed, ']');
    if (port && !port.startsWith(':')) throw new PyException('ValueError', 'Invalid IPv6 URL');
  } else {
    [hostname] = partition(hostnameAndPort, ':');
  }
  if (hostname.startsWith('v')) {
    if (!/^v[a-fA-F0-9]+\.[^\n]+$/u.test(hostname)) throw new PyException('ValueError', 'IPvFuture address is invalid');
  } else if (!isIPv6(hostname)) {
    throw new PyException('ValueError', isIPv4(hostname)
      ? 'An IPv4 address cannot be in brackets'
      : `${pyRepr(hostname)} does not appear to be an IPv4 or IPv6 address`);
  }
}

// urllib.parse._checknetloc: raises ValueError.
function checkNetloc(netloc) {
  if (!netloc || /^[\x00-\x7f]*$/u.test(netloc)) return;
  const n = netloc.replaceAll('@', '').replaceAll(':', '').replaceAll('#', '').replaceAll('?', '');
  const normalized = n.normalize('NFKC');
  if (n === normalized) return;
  for (const char of '/?#@:') {
    if (normalized.includes(char)) {
      throw new PyException('ValueError', `netloc '${netloc}' contains invalid characters under NFKC normalization`);
    }
  }
}

// urllib.parse.urlsplit(url) for a str without C0 controls or spaces (the
// URL redaction never hands it any): { scheme, netloc, path, query, fragment }.
function urlsplit(url) {
  let scheme = '';
  let netloc = '';
  let query = '';
  let fragment = '';
  let rest = url;
  const colon = rest.indexOf(':');
  if (colon > 0 && /^[A-Za-z]/u.test(rest) && SCHEME_CHARS.test(rest.slice(0, colon))) {
    scheme = rest.slice(0, colon).toLowerCase();
    rest = rest.slice(colon + 1);
  }
  if (rest.startsWith('//')) {
    let delim = rest.length;
    for (const char of '/?#') {
      const index = rest.indexOf(char, 2);
      if (index >= 0) delim = Math.min(delim, index);
    }
    netloc = rest.slice(2, delim);
    rest = rest.slice(delim);
    if ((netloc.includes('[') && !netloc.includes(']')) || (netloc.includes(']') && !netloc.includes('['))) {
      throw new PyException('ValueError', 'Invalid IPv6 URL');
    }
    if (netloc.includes('[') && netloc.includes(']')) checkBracketedNetloc(netloc);
  }
  if (rest.includes('#')) [rest, , fragment] = partition(rest, '#');
  if (rest.includes('?')) [rest, , query] = partition(rest, '?');
  checkNetloc(netloc);
  return { scheme, netloc, path: rest, query, fragment };
}

// SplitResult._hostinfo: [hostname, port or null].
function hostinfo(netloc) {
  const host = rpartition(netloc, '@')[2];
  const [, haveOpenBracket, bracketed] = partition(host, '[');
  let hostname;
  let port;
  if (haveOpenBracket) {
    let afterBracket;
    [hostname, , afterBracket] = partition(bracketed, ']');
    [, , port] = partition(afterBracket, ':');
  } else {
    [hostname, , port] = partition(host, ':');
  }
  return [hostname, port || null];
}

// The sanitized form of one matched http(s) URL.
function sanitizeUrl(matched) {
  const candidate = matched.replaceAll('\\/', '/');
  let parts;
  try {
    parts = urlsplit(candidate);
    const [hostname, port] = hostinfo(parts.netloc);
    if (!parts.netloc || !hostname) throw new PyException('ValueError', 'HTTP(S) URL is missing its authority');
    if (port !== null && (!/^[0-9]+$/u.test(port) || BigInt(port) > 65535n)) {
      throw new PyException('ValueError', 'invalid port');
    }
  } catch (error) {
    if (!isPyException(error, 'ValueError')) throw error;
    return `${partition(candidate, ':')[0].toLowerCase()}://<redacted-url>`;
  }
  const netloc = rpartition(parts.netloc, '@')[2];
  // urlunsplit((scheme, netloc, path, "", "")) with a non-empty netloc.
  let url = parts.path && !parts.path.startsWith('/') ? `/${parts.path}` : parts.path;
  url = `//${netloc}${url}`;
  return parts.scheme ? `${parts.scheme}:${url}` : url;
}

// Drop credentials, query strings, and fragments from any URL in diagnostics.
export function sanitizeDiagnosticText(text) {
  let sanitized = text.replace(URL_RE, sanitizeUrl);
  sanitized = sanitized.replace(AUTHORIZATION_RE, () => 'Authorization: <redacted>');
  sanitized = sanitized.replace(AUTH_SCHEME_RE, () => REDACTED_CREDENTIAL);
  return sanitized.replace(CREDENTIAL_ASSIGNMENT_RE, (...args) => redactCredentialAssignment(args[0], args.at(-1)));
}

// Python's recursion limit (1000 frames) under the qualify command's call
// chain (<module>, main, qualify_cmd, _run_smoke, _write_smoke_diagnostics,
// sanitize_diagnostic_stdout) leaves this many frames for
// sanitize_diagnostic_value: one per nesting level (the top-level value is
// level 1), plus what the value's own work calls below it. A non-str value
// takes 3 more (isinstance(value, Mapping) runs ABCMeta's __instancecheck__,
// __subclasscheck__ and __subclasshook__), and so does a str or a dict key
// (sanitize_diagnostic_text, re.sub, re._compile), or 5 when it holds a URL
// (urlsplit and its helpers). Deeper, Python raises RecursionError; measured
// on 3.12.3 and 3.12.11. A bracketed URL host goes further (ipaddress) by an
// amount that depends on urlsplit's lru_cache, which is not modelled.
export const SANITIZE_FRAME_BUDGET = 994;
// How many containers around a scalar (itself one level deeper) sanitize.
export const MAX_SANITIZE_NESTING = SANITIZE_FRAME_BUDGET - 4;

function textFrames(text) {
  URL_RE.lastIndex = 0;
  return URL_RE.test(text) ? 5 : 3;
}

function requireSanitizeFrames(level, frames) {
  if (level + frames > SANITIZE_FRAME_BUDGET) throw new PyException('RecursionError', 'maximum recursion depth exceeded');
}

// Sanitize decoded JSON in place of its serialization, so it stays valid JSON.
// `level` is the Python nesting of the call (1 for the top-level value).
export function sanitizeDiagnosticValue(value, level = 1) {
  if (typeof value === 'string') {
    requireSanitizeFrames(level, textFrames(value));
    return sanitizeDiagnosticText(value);
  }
  requireSanitizeFrames(level, 3);
  if (isMapping(value)) {
    return pyDict(pyItems(value).map(([key, item]) => {
      const isKey = typeof key === 'string';
      const redact = isKey && (
        AUTHORIZATION_NAME_RE.test(key)
        || (CREDENTIAL_NAME_RE.test(key)
          && !(isLlamaTokenCounterName(key) && isPyInt(item) && intValue(item) >= 0n))
      );
      if (isKey) requireSanitizeFrames(level, textFrames(key));
      return [isKey ? sanitizeDiagnosticText(key) : key, redact ? REDACTED_CREDENTIAL : sanitizeDiagnosticValue(item, level + 1)];
    }));
  }
  if (Array.isArray(value)) return value.map((item) => sanitizeDiagnosticValue(item, level + 1));
  return value;
}

// Sanitize child stdout without destroying the structure a reader needs.
export function sanitizeDiagnosticStdout(stdout) {
  let serialized;
  try {
    let payload = pyJsonLoads(stdout, { parseConstant: rejectNonstandardJsonConstant });
    payload = sanitizeDiagnosticValue(payload);
    serialized = pyJsonDumps(payload, { allowNan: false });
  } catch (error) {
    if (!(error instanceof ContractError) && !isPyException(error, 'ValueError')) throw error;
    serialized = pyJsonDumps(sanitizeDiagnosticText(stdout), { allowNan: false });
  }
  return `${serialized}\n`;
}

// --- harness digests ------------------------------------------------------------------

// bytes.decode("utf-8", errors="replace"), keeping a BOM.
function decodeReplace(bytes) {
  return new TextDecoder('utf-8', { ignoreBOM: true }).decode(bytes);
}

// The OSError Python raises for a failed call on `filename`.
function toOSError(error, filename) {
  if (!isOSError(error) || error instanceof PyException) return error;
  const classes = { ENOENT: 'FileNotFoundError', EISDIR: 'IsADirectoryError', ENOTDIR: 'NotADirectoryError', EACCES: 'PermissionError', EPERM: 'PermissionError' };
  const exception = new PyException(classes[error.code] ?? 'OSError', osErrorString(error, filename));
  exception.code = error.code;
  return exception;
}

function harnessDigest(sources, readSource) {
  const digest = createHash('sha256');
  for (const name of sortedStrings(sources)) {
    const data = readSource(name);
    const length = Buffer.alloc(8);
    length.writeBigUInt64BE(BigInt(data.length));
    digest.update(Buffer.concat([Buffer.from(name, 'utf8'), Buffer.from([0])]));
    digest.update(length);
    digest.update(data);
  }
  return digest.digest('hex');
}

// Digest every harness source so an attestation names the code that ran. A
// source name is a path relative to `scriptsDir` ('/'-separated).
export function harnessSourceSha256(scriptsDir, { sources = HARNESS_SOURCES } = {}) {
  const directory = pyPath(String(scriptsDir));
  return harnessDigest(sources, (name) => {
    const sourcePath = pyJoinPath(directory, name);
    if (!pyIsFile(sourcePath)) throw new ContractError(`harness source is missing: ${name}`);
    return pyReadBytes(sourcePath);
  });
}

// The same digest over `git show <bridgeSha>:scripts/<name>` in `repository`.
export function harnessSourceSha256AtCommit(repository, bridgeSha, { sources = HARNESS_SOURCES } = {}) {
  if (fullmatch(COMMIT_RE, bridgeSha) === null) throw new ContractError('candidate bridge source must be a lowercase 40-hex SHA');
  const repositoryPath = pyPath(String(repository));
  return harnessDigest(sources, (name) => {
    const result = spawnSync('git', ['-C', repositoryPath, 'show', `${bridgeSha}:scripts/${name}`], {
      stdio: ['inherit', 'pipe', 'pipe'],
      maxBuffer: 1024 * 1024 * 1024,
    });
    if (result.error) throw toOSError(result.error, 'git');
    if (result.status !== 0) {
      const diagnostic = decodeReplace(result.stderr);
      throw new ContractError(`could not read harness source ${pyRepr(name)} at ${bridgeSha}: ${sanitizeDiagnosticText(pyStrip(diagnostic))}`);
    }
    return result.stdout;
  });
}

// Path(text).parent for a normalized path.
function pyPathParent(text) {
  if (text === '.' || /^\/+$/u.test(text)) return text;
  const index = text.lastIndexOf('/');
  if (index < 0) return '.';
  const head = text.slice(0, index);
  return head === '' || /^\/+$/u.test(head) ? text.slice(0, index + 1) : head;
}

// Require the local harness bytes to equal the exact candidate source.
export function requireHarnessMatchesBridgeSource(scriptsDir, bridgeSha, { sources = HARNESS_SOURCES } = {}) {
  const directory = pyPath(String(scriptsDir));
  const localDigest = harnessSourceSha256(directory, { sources });
  const sourceDigest = harnessSourceSha256AtCommit(pyPathParent(directory), bridgeSha, { sources });
  if (localDigest !== sourceDigest) {
    throw new ContractError(
      `qualification harness does not match the exact candidate bridge source ${bridgeSha}: local=${localDigest}, source=${sourceDigest}`,
    );
  }
  return localDigest;
}

// --- candidates -------------------------------------------------------------------------

export function requireStr(payload, key, label) {
  const value = pyGet(payload, key);
  if (typeof value !== 'string' || !value) throw new ContractError(`${label} ${key} must be a non-empty string`);
  return value;
}

export function requireInt(payload, key, label) {
  const value = pyGet(payload, key);
  if (!isPyInt(value)) throw new ContractError(`${label} ${key} must be an integer`);
  return value;
}

// Validate a candidate directory against the current strict contract:
// [manifest, fingerprint].
export function loadCandidate(directory) {
  return loadCandidateContract(directory);
}

// Validate immutable published bytes against an explicit historical contract.
export function loadPublishedCandidate(directory, { expectedQualificationGates, expectedUnprovenCapabilities } = {}) {
  return loadCandidateContract(directory, {
    publishedManifestContract: [expectedQualificationGates, expectedUnprovenCapabilities],
  });
}

// Validate an exact candidate directory and return its manifest and digest.
//
// The candidate is validated with the same routine publication uses, so the
// fingerprint an attestation binds is the identical fingerprint publication
// recomputes; a divergent second definition could never be kept in step.
export function loadCandidateContract(directory, { publishedManifestContract = null } = {}) {
  const root = pyPath(String(directory));
  if (!pyIsDir(root) || pyIsSymlink(root)) throw new ContractError(`candidate directory does not exist: ${root}`);
  const names = pyListdir(root);
  const actual = new Set(names);
  const expected = new Set(PUBLICATION_FILES);
  if (actual.size !== expected.size || [...actual].some((name) => !expected.has(name))) {
    const unexpected = sortedStrings([...actual].filter((name) => !expected.has(name)));
    const missing = sortedStrings([...expected].filter((name) => !actual.has(name)));
    throw new ContractError(
      `candidate directory must contain exactly the publication files (unexpected: ${pyRepr(unexpected)}, missing: ${pyRepr(missing)})`,
    );
  }
  for (const name of names) {
    const entry = pyJoinPath(root, name);
    if (pyIsSymlink(entry) || !pyIsFile(entry)) throw new ContractError(`candidate entry must be an immutable regular file: ${name}`);
  }

  let manifestText;
  try {
    manifestText = pyReadText(pyJoinPath(root, 'manifest.json'));
  } catch (error) {
    if (isPyException(error, 'OSError', 'UnicodeDecodeError')) {
      throw new ContractError(`could not read candidate manifest: ${error.message}`, { cause: error });
    }
    throw error;
  }
  const manifest = parseAttestationJson(manifestText);

  const label = 'candidate manifest';
  const identity = new CandidateIdentity({
    releaseTag: requireStr(manifest, 'release_tag', label),
    releaseRebuild: requireInt(manifest, 'release_rebuild', label),
    assetsRepo: requireStr(manifest, 'assets_repository', label),
    bridgeCommit: requireStr(manifest, 'bridge_commit', label),
    upstreamTag: requireStr(manifest, 'upstream_tag', label),
    upstreamCommit: requireStr(manifest, 'upstream_commit', label),
    nativeReleaseTag: requireStr(manifest, 'native_release_tag', label),
    nativeManifestSha256: requireStr(manifest, 'native_manifest_sha256', label),
    nativeCommit: requireStr(manifest, 'native_commit', label),
    emscriptenVersion: requireStr(manifest, 'emscripten_version', label),
    orchestratorCorrelationId: requireStr(manifest, 'orchestrator_correlation_id', label),
    githubRunId: requireStr(manifest, 'github_run_id', label),
    githubRunUrl: requireStr(manifest, 'github_run_url', label),
  });
  const fingerprint = publishedManifestContract === null
    ? validateCandidate(root, identity)
    : validatePublishedCandidate(root, identity, {
      expectedQualificationGates: publishedManifestContract[0],
      expectedUnprovenCapabilities: publishedManifestContract[1],
    });
  return [manifest, fingerprint];
}

export function manifestCandidateGates(manifest) {
  const gates = pyGet(manifest, 'qualification_gates');
  if (!isMapping(gates)) throw new ContractError('candidate manifest qualification_gates must be an object');
  const proven = {};
  for (const name of CANDIDATE_GATES) {
    const value = pyGet(gates, name);
    if (!pyEquals(value, 'passed')) throw new ContractError(`candidate manifest gate ${pyRepr(name)} is ${pyRepr(value)}; must be 'passed'`);
    proven[name] = value;
  }
  for (const name of HEAVY_GATES) {
    const value = pyGet(gates, name);
    if (!pyEquals(value, AUTOMATED_QUALIFICATION_REQUIRED)) {
      throw new ContractError(
        `candidate manifest gate ${pyRepr(name)} is ${pyRepr(value)}; a candidate must declare `
        + `${pyRepr(AUTOMATED_QUALIFICATION_REQUIRED)} rather than claim a pass its own run never executed`,
      );
    }
  }
  return proven;
}

// --- the qualification run and environment ----------------------------------------------

function envGet(env, key, fallback = null) {
  return Object.hasOwn(env, key) ? env[key] : fallback;
}

// Record the hosted runner the heavy gates actually executed on.
//
// The values come from the runner itself rather than from a caller-supplied
// claim. GitHub's hosted-runner marker and Actions marker are both required;
// OS/architecture labels alone are locally forgeable and are not a hosted
// execution proof. `env` is os.environ.
export function qualificationEnvironment({ env = process.env } = {}) {
  if (envGet(env, 'GITHUB_ACTIONS') !== 'true') {
    throw new ContractError(
      'GITHUB_ACTIONS must identify a GitHub Actions run; qualification only runs on hosted GitHub Actions infrastructure',
    );
  }
  if (envGet(env, 'RUNNER_ENVIRONMENT') !== 'github-hosted') {
    throw new ContractError(
      "RUNNER_ENVIRONMENT must be 'github-hosted'; qualification only runs on hosted GitHub Actions infrastructure",
    );
  }
  const labels = {};
  for (const [key, field] of [['RUNNER_OS', 'runner_os'], ['RUNNER_ARCH', 'runner_arch']]) {
    const value = envGet(env, key, '');
    if (RUNNER_LABEL_RE.exec(value) === null) {
      throw new ContractError(`${key} must identify the hosted runner; qualification only runs on hosted GitHub Actions infrastructure`);
    }
    labels[field] = value;
  }
  // os.cpu_count() and SC_PAGE_SIZE * SC_PHYS_PAGES.
  const cpuCount = os.cpus().length || 0;
  const totalMemoryBytes = os.totalmem();
  return {
    execution: QUALIFICATION_EXECUTION,
    cpu_count: requirePositiveInt(cpuCount, 'runner cpu_count'),
    total_memory_bytes: requirePositiveInt(totalMemoryBytes, 'runner total_memory_bytes'),
    ...labels,
  };
}

// Bind the attestation to the exact trusted workflow run that produced it.
export function qualificationRunIdentity({ env = process.env } = {}) {
  const runId = envGet(env, 'GITHUB_RUN_ID', '');
  if (RUN_ID_RE.exec(runId) === null) throw new ContractError('GITHUB_RUN_ID must identify the qualification run');
  if (envGet(env, 'GITHUB_RUN_ATTEMPT', '') !== '1') throw new ContractError('qualification GITHUB_RUN_ATTEMPT must be 1');
  const sourceSha = envGet(env, 'GITHUB_SHA', '');
  if (COMMIT_RE.exec(sourceSha) === null) throw new ContractError('GITHUB_SHA must identify the qualification workflow source');
  if (envGet(env, 'GITHUB_REPOSITORY') !== BRIDGE_REPOSITORY) throw new ContractError(`GITHUB_REPOSITORY must be exactly ${BRIDGE_REPOSITORY}`);
  return {
    qualification_run_id: runId,
    qualification_run_attempt: 1,
    qualification_run_url: `https://github.com/${BRIDGE_REPOSITORY}/actions/runs/${runId}`,
    qualification_source_sha: sourceSha,
    qualification_workflow_path: QUALIFICATION_WORKFLOW_PATH,
  };
}

export function validateQualificationEnvironment(value) {
  const environment = requireExactMapping(value, QUALIFICATION_ENVIRONMENT_KEYS, 'qualification_environment');
  const execution = pyGet(environment, 'execution');
  if (!pyEquals(execution, QUALIFICATION_EXECUTION)) {
    throw new ContractError(`qualification_environment execution must be ${pyRepr(QUALIFICATION_EXECUTION)}, got ${pyRepr(execution)}`);
  }
  for (const field of ['runner_os', 'runner_arch']) {
    const label = pyGet(environment, field);
    if (typeof label !== 'string' || RUNNER_LABEL_RE.exec(label) === null) {
      throw new ContractError(`qualification_environment ${field} must be a short runner label`);
    }
  }
  const cpuCount = requirePositiveInt(pyGet(environment, 'cpu_count'), 'qualification_environment cpu_count');
  if (intValue(cpuCount) > BigInt(MAX_QUALIFICATION_CPU_COUNT)) throw new ContractError('qualification_environment cpu_count is implausible');
  const memory = requirePositiveInt(pyGet(environment, 'total_memory_bytes'), 'qualification_environment total_memory_bytes');
  if (intValue(memory) > BigInt(MAX_QUALIFICATION_MEMORY_BYTES)) {
    throw new ContractError('qualification_environment total_memory_bytes is implausible');
  }
}

// --- workflow runs and artifacts ------------------------------------------------------------

// Fail closed unless a run is the exact successful dispatch we require.
//
// Artifact names are attacker-chosen strings, so a run ID alone proves
// nothing. Repository, workflow file, dispatch event, default branch, and
// success are all checked before any artifact from the run is trusted.
//
// Returns the run's head commit. The caller proves that commit is reachable
// from the default branch rather than equal to its current head, so an
// unrelated push to the default branch cannot invalidate an in-flight
// candidate while a fork or feature branch is still refused.
export function validateWorkflowRun(run, {
  expectedRunId, expectedWorkflowPath, expectedHeadBranch, expectedRunAttempt = 1,
} = {}) {
  if (!isMapping(run)) throw new ContractError('workflow run payload must be a JSON object');
  if (fullmatch(RUN_ID_RE, expectedRunId) === null) throw new ContractError('run id must be a positive integer');
  const actualId = pyGet(run, 'id');
  if (!(typeof actualId === 'boolean' || isPyInt(actualId)) || pyStr(actualId) !== expectedRunId) {
    throw new ContractError(`workflow run id mismatch: expected ${expectedRunId}, got ${pyRepr(actualId)}`);
  }
  for (const field of ['repository', 'head_repository']) {
    const repository = pyGet(run, field);
    if (!isMapping(repository) || !pyEquals(pyGet(repository, 'full_name'), BRIDGE_REPOSITORY)) {
      throw new ContractError(`workflow run ${field} must be exactly ${BRIDGE_REPOSITORY}`);
    }
  }
  const expectedOwner = BRIDGE_REPOSITORY.split('/')[0];
  const actor = pyGet(run, 'actor');
  if (!isMapping(actor) || !pyEquals(pyGet(actor, 'login'), expectedOwner)) throw new ContractError(`workflow run actor must be ${expectedOwner}`);
  const triggeringActor = pyGet(run, 'triggering_actor');
  if (!isMapping(triggeringActor) || !pyEquals(pyGet(triggeringActor, 'login'), expectedOwner)) {
    throw new ContractError(`workflow run triggering_actor must be ${expectedOwner}`);
  }
  if (!pyEquals(pyGet(run, 'path'), expectedWorkflowPath)) {
    throw new ContractError(`workflow run path mismatch: expected ${pyStr(expectedWorkflowPath)}, got ${pyRepr(pyGet(run, 'path'))}`);
  }
  if (!pyEquals(pyGet(run, 'event'), 'workflow_dispatch')) {
    throw new ContractError(`workflow run event must be workflow_dispatch, got ${pyRepr(pyGet(run, 'event'))}`);
  }
  if (!pyEquals(pyGet(run, 'status'), 'completed')) {
    throw new ContractError(`workflow run status must be completed, got ${pyRepr(pyGet(run, 'status'))}`);
  }
  if (!pyEquals(pyGet(run, 'conclusion'), 'success')) {
    throw new ContractError(`workflow run conclusion must be success, got ${pyRepr(pyGet(run, 'conclusion'))}`);
  }
  if (!isPyInt(expectedRunAttempt) || !pyEquals(expectedRunAttempt, 1)) throw new ContractError('expected run attempt must be exactly 1');
  const actualRunAttempt = pyGet(run, 'run_attempt');
  if (!isPyInt(actualRunAttempt) || !pyEquals(actualRunAttempt, expectedRunAttempt)) {
    throw new ContractError(`workflow run attempt must be ${pyStr(expectedRunAttempt)}, got ${pyRepr(actualRunAttempt)}`);
  }
  if (!pyTruthy(expectedHeadBranch)) throw new ContractError('expected head branch is required');
  if (!pyEquals(pyGet(run, 'head_branch'), expectedHeadBranch)) {
    throw new ContractError(`workflow run head_branch must be ${pyRepr(expectedHeadBranch)}, got ${pyRepr(pyGet(run, 'head_branch'))}`);
  }
  const headSha = pyGet(run, 'head_sha');
  if (typeof headSha !== 'string' || COMMIT_RE.exec(headSha) === null) throw new ContractError('workflow run head_sha must be a 40-hex commit SHA');
  return headSha;
}

// Require exactly one live artifact of the expected name in the exact run.
export function validateArtifactInventory(inventory, { expectedRunId, expectedName } = {}) {
  if (!isMapping(inventory)) throw new ContractError('artifact inventory must be a JSON object');
  const artifacts = pyGet(inventory, 'artifacts');
  if (!Array.isArray(artifacts)) throw new ContractError('artifact inventory is missing an artifacts array');
  const totalCount = pyGet(inventory, 'total_count');
  if (!isPyInt(totalCount) || intValue(totalCount) < 0n) throw new ContractError('artifact inventory total_count must be a non-negative integer');
  if (intValue(totalCount) !== BigInt(artifacts.length)) {
    throw new ContractError(
      `artifact inventory is truncated; all run artifacts must be inspected (${artifacts.length} records for total_count=${totalCount})`,
    );
  }
  const matches = [];
  for (const artifact of artifacts) {
    if (!isMapping(artifact) || typeof pyGet(artifact, 'name') !== 'string') throw new ContractError('artifact inventory contains an invalid record');
    if (!pyEquals(getItem(artifact, 'name'), expectedName)) continue;
    const run = pyGet(artifact, 'workflow_run');
    if (!isMapping(run) || !pyEquals(pyStr(pyGet(run, 'id')), expectedRunId)) {
      throw new ContractError(`artifact ${pyRepr(expectedName)} does not belong to run ${pyStr(expectedRunId)}`);
    }
    if (pyGet(artifact, 'expired') !== false) throw new ContractError(`artifact ${pyRepr(expectedName)} has expired`);
    matches.push(artifact);
  }
  if (matches.length !== 1) {
    throw new ContractError(`run ${pyStr(expectedRunId)} must expose exactly one ${pyRepr(expectedName)} artifact, found ${matches.length}`);
  }
  const artifactId = pyGet(matches[0], 'id');
  if (!isPyInt(artifactId) || intValue(artifactId) <= 0n) throw new ContractError(`artifact ${pyRepr(expectedName)} has no positive integer id`);
  return artifactId;
}

// --- attestations -------------------------------------------------------------------------------

export function buildAttestation({
  manifest,
  candidateFingerprint,
  candidateRunId,
  candidateArtifactId,
  candidateRunAttempt = 1,
  qualificationRunId,
  qualificationRunAttempt,
  qualificationSourceSha,
  harnessDigest: digest,
  environment,
  speechPhase,
  ttsPhase,
} = {}) {
  requireSha256(candidateFingerprint, 'candidate_fingerprint');
  requireSha256(digest, 'harness_source_sha256');
  if (fullmatch(RUN_ID_RE, candidateRunId) === null) throw new ContractError('candidate_run_id must be a positive integer');
  requirePositiveInt(candidateArtifactId, 'candidate_artifact_id');
  if (!isPyInt(candidateRunAttempt) || !pyEquals(candidateRunAttempt, 1)) throw new ContractError('candidate_run_attempt must be 1');
  if (fullmatch(RUN_ID_RE, qualificationRunId) === null) throw new ContractError('qualification_run_id must be a positive integer');
  if (qualificationRunId === candidateRunId) throw new ContractError('candidate and qualification runs must be distinct');
  if (!isPyInt(qualificationRunAttempt) || !pyEquals(qualificationRunAttempt, 1)) throw new ContractError('qualification_run_attempt must be 1');
  if (fullmatch(COMMIT_RE, qualificationSourceSha) === null) {
    throw new ContractError('qualification_source_sha must be a lowercase 40-hex commit SHA');
  }
  const manifestRunId = requireStr(manifest, 'github_run_id', 'candidate manifest');
  if (manifestRunId !== candidateRunId) {
    throw new ContractError(`candidate manifest github_run_id does not match the candidate run: ${manifestRunId} != ${candidateRunId}`);
  }
  requireCorrelationId(requireStr(manifest, 'orchestrator_correlation_id', 'candidate manifest'));
  validateQualificationEnvironment(environment);

  const attestation = {};
  attestation.schema_version = QUALIFICATION_SCHEMA_VERSION;
  attestation.attestation_type = ATTESTATION_TYPE;
  attestation.candidate_fingerprint = candidateFingerprint;
  attestation.candidate_run_id = candidateRunId;
  attestation.candidate_artifact_id = candidateArtifactId;
  attestation.candidate_run_attempt = candidateRunAttempt;
  attestation.candidate_run_url = requireStr(manifest, 'github_run_url', 'candidate manifest');
  attestation.candidate_workflow_path = CANDIDATE_WORKFLOW_PATH;
  attestation.qualification_run_id = qualificationRunId;
  attestation.qualification_run_attempt = qualificationRunAttempt;
  attestation.qualification_run_url = `https://github.com/${BRIDGE_REPOSITORY}/actions/runs/${qualificationRunId}`;
  attestation.qualification_source_sha = qualificationSourceSha;
  attestation.qualification_workflow_path = QUALIFICATION_WORKFLOW_PATH;
  attestation.bridge_repository = BRIDGE_REPOSITORY;
  attestation.bridge_source_sha = getItem(manifest, 'bridge_commit');
  attestation.upstream_repository = 'ggml-org/llama.cpp';
  attestation.upstream_tag = getItem(manifest, 'upstream_tag');
  attestation.upstream_commit = getItem(manifest, 'upstream_commit');
  attestation.native_repository = NATIVE_REPOSITORY;
  attestation.native_release_tag = getItem(manifest, 'native_release_tag');
  attestation.native_manifest_sha256 = getItem(manifest, 'native_manifest_sha256');
  attestation.native_commit = getItem(manifest, 'native_commit');
  attestation.emscripten_version = getItem(manifest, 'emscripten_version');
  attestation.release_tag = getItem(manifest, 'release_tag');
  attestation.release_rebuild = getItem(manifest, 'release_rebuild');
  attestation.orchestrator_correlation_id = getItem(manifest, 'orchestrator_correlation_id');
  attestation.harness_version = HARNESS_VERSION;
  attestation.harness_source_sha256 = digest;
  attestation.model_pins = { ...EXPECTED_MODEL_PINS };
  attestation.candidate_gates = manifestCandidateGates(manifest);
  attestation.heavy_gates = Object.fromEntries(HEAVY_GATES.map((gate) => [gate, 'passed']));
  attestation.unproven_capabilities = { ...REQUIRED_UNPROVEN_CAPABILITIES };
  attestation.qualification_environment = pyDeepCopy(environment);
  attestation.phases = {
    speech_to_text: pyDeepCopy(speechPhase),
    text_to_speech: pyDeepCopy(ttsPhase),
  };
  return attestation;
}

export function requireExactMapping(value, keys, label) {
  if (!isMapping(value)) throw new ContractError(`${label} must be a JSON object`);
  const actual = new Set(pyKeys(value));
  const expected = new Set(Array.isArray(keys) ? keys : pyKeys(keys));
  const missing = sortedStrings([...expected].filter((key) => !actual.has(key)));
  const unexpected = sortedStrings([...actual].filter((key) => !expected.has(key)));
  if (missing.length) throw new ContractError(`${label} is missing required keys: ${pyRepr(missing)}`);
  if (unexpected.length) throw new ContractError(`${label} contains unexpected keys: ${pyRepr(unexpected)}`);
  return value;
}

export function requirePositiveInt(value, label) {
  if (!isPyInt(value) || intValue(value) <= 0n) throw new ContractError(`${label} must be a positive integer`);
  return value;
}

export function requireNonNegativeInt(value, label) {
  if (!isPyInt(value) || intValue(value) < 0n) throw new ContractError(`${label} must be a non-negative integer`);
  return value;
}

// math.isfinite(value) for an int or float; an int too large for a float
// raises OverflowError, as Python raises it.
function isFiniteNumber(value) {
  if (typeof value === 'bigint') {
    if (!Number.isFinite(Number(value))) throw new PyException('OverflowError', 'int too large to convert to float');
    return true;
  }
  return Number.isFinite(numericValue(value));
}

const TTS_MODE_KEYS = Object.freeze([
  'cancellation_tested',
  'frames_generated',
  'peak',
  'pre_aborted_tested',
  'reuse_sample_count',
  'rms',
  'truncated',
  'unload_tested',
  'wav',
]);
const SPEECH_MODE_KEYS = Object.freeze(['cancellation_result', 'cold_transcript', 'silence_transcript', 'warm_transcript']);
const WAV_KEYS = Object.freeze(['bits_per_sample', 'byte_length', 'channels', 'frame_count', 'sample_rate', 'sha256']);

export function validatePhase(phase, { label, requiredModes, phaseKeys, requireWav }) {
  requireExactMapping(phase, ['max_rss_bytes', 'modes', 'total_ms'], label);
  requirePositiveInt(getItem(phase, 'max_rss_bytes'), `${label}.max_rss_bytes`);
  const phaseTotal = requireNonNegativeInt(getItem(phase, 'total_ms'), `${label}.total_ms`);
  const modes = getItem(phase, 'modes');
  if (!Array.isArray(modes)) throw new ContractError(`${label}.modes must be an array`);
  const seen = [];
  modes.forEach((mode, index) => {
    const modeLabel = `${label}.modes[${index}]`;
    const keys = ['memory_mode', 'phase_timings_ms', 'runtime_mode', 'total_ms', ...(requireWav ? TTS_MODE_KEYS : SPEECH_MODE_KEYS)];
    requireExactMapping(mode, keys, modeLabel);
    const memoryMode = getItem(mode, 'memory_mode');
    const runtimeMode = getItem(mode, 'runtime_mode');
    if (typeof memoryMode !== 'string' || typeof runtimeMode !== 'string') throw new ContractError(`${modeLabel} mode identifiers must be strings`);
    const modeTotal = requireNonNegativeInt(getItem(mode, 'total_ms'), `${modeLabel}.total_ms`);
    const timings = requireExactMapping(getItem(mode, 'phase_timings_ms'), phaseKeys, `${modeLabel}.phase_timings_ms`);
    for (const key of phaseKeys) requireNonNegativeInt(getItem(timings, key), `${modeLabel}.phase_timings_ms.${key}`);
    const timingTotal = phaseKeys.reduce((total, key) => total + intValue(getItem(timings, key)), 0n);
    // Each browser phase is measured sequentially with Math.round(). Allow
    // one millisecond of aggregate rounding per phase, but reject a mode
    // total that cannot contain the timings it claims.
    if (intValue(modeTotal) + BigInt(phaseKeys.length) < timingTotal) {
      throw new ContractError(`${modeLabel}.total_ms is shorter than its recorded phase timings`);
    }
    if (requireWav) {
      if (getItem(mode, 'truncated') !== false) throw new ContractError(`${modeLabel}.truncated must be false`);
      if (getItem(mode, 'cancellation_tested') !== true) throw new ContractError(`${modeLabel}.cancellation_tested must be true`);
      if (getItem(mode, 'pre_aborted_tested') !== true) throw new ContractError(`${modeLabel}.pre_aborted_tested must be true`);
      requirePositiveInt(getItem(mode, 'reuse_sample_count'), `${modeLabel}.reuse_sample_count`);
      if (getItem(mode, 'unload_tested') !== true) throw new ContractError(`${modeLabel}.unload_tested must be true`);
      const wav = requireExactMapping(getItem(mode, 'wav'), WAV_KEYS, `${modeLabel}.wav`);
      const wavSha256 = getItem(wav, 'sha256');
      if (typeof wavSha256 !== 'string') throw new ContractError(`${modeLabel}.wav.sha256 must be a string`);
      requireSha256(wavSha256, `${modeLabel}.wav.sha256`);
      const byteLength = requirePositiveInt(getItem(wav, 'byte_length'), `${modeLabel}.wav.byte_length`);
      const frameCount = requirePositiveInt(getItem(wav, 'frame_count'), `${modeLabel}.wav.frame_count`);
      const channels = requirePositiveInt(getItem(wav, 'channels'), `${modeLabel}.wav.channels`);
      const bitsPerSample = requirePositiveInt(getItem(wav, 'bits_per_sample'), `${modeLabel}.wav.bits_per_sample`);
      requirePositiveInt(getItem(wav, 'sample_rate'), `${modeLabel}.wav.sample_rate`);
      // The codec-frame cap is a CLI flag, so the generated count is
      // recorded as auditable evidence of how much synthesis actually ran
      // rather than being compared against a guessed floor.
      requirePositiveInt(getItem(mode, 'frames_generated'), `${modeLabel}.frames_generated`);
      for (const measurement of ['peak', 'rms']) {
        const value = getItem(mode, measurement);
        const numeric = isPyFloat(value) || isPyInt(value);
        if (!numeric || !isFiniteNumber(value) || numericValue(value) <= 0) {
          throw new ContractError(`${modeLabel}.${measurement} must be a positive finite number`);
        }
      }
      if (numericValue(getItem(mode, 'peak')) <= 0.001 || numericValue(getItem(mode, 'rms')) <= 0.0001) {
        throw new ContractError(`${modeLabel} waveform evidence is below the required peak/RMS floor`);
      }
      if (!pyEquals(getItem(wav, 'sample_rate'), TTS_WAV_SAMPLE_RATE) || !pyEquals(channels, TTS_WAV_CHANNELS)
        || !pyEquals(bitsPerSample, TTS_WAV_BITS_PER_SAMPLE)) {
        throw new ContractError(`${modeLabel}.wav must be PCM16 mono ${TTS_WAV_SAMPLE_RATE} Hz`);
      }
      const minimumPcmBytes = intValue(frameCount) * intValue(channels) * (intValue(bitsPerSample) / 8n);
      if (intValue(byteLength) < minimumPcmBytes) throw new ContractError(`${modeLabel}.wav.byte_length is smaller than its PCM frame data`);
    } else {
      for (const transcriptField of ['cold_transcript', 'warm_transcript']) {
        const transcript = getItem(mode, transcriptField);
        if (typeof transcript !== 'string' || !transcript) throw new ContractError(`${modeLabel}.${transcriptField} must be a non-empty string`);
        if (normalizeTranscript(transcript) !== EXPECTED_SPEECH_TRANSCRIPT) {
          throw new ContractError(`${modeLabel}.${transcriptField} does not match expected transcript`);
        }
      }
      parseCancellationResult(getItem(mode, 'cancellation_result'), `${modeLabel}.cancellation_result`);
      if (!pyEquals(getItem(mode, 'silence_transcript'), '')) throw new ContractError(`${modeLabel}.silence_transcript must stay empty`);
    }
    if (seen.some(([memory, runtime]) => memory === memoryMode && runtime === runtimeMode)) {
      throw new ContractError(`${label} repeats mode ${memoryMode}/${runtimeMode}`);
    }
    seen.push([memoryMode, runtimeMode]);
  });
  const sortedSeen = [...seen].sort(comparePairs);
  const sortedRequired = [...requiredModes].sort(comparePairs);
  if (sortedSeen.length !== sortedRequired.length || sortedSeen.some((pair, index) => comparePairs(pair, sortedRequired[index]) !== 0)) {
    throw new ContractError(`${label} must cover exactly ${reprPairs(sortedRequired)}, covered ${reprPairs(sortedSeen)}`);
  }
  const computedTotal = modes.reduce((total, mode) => total + intValue(getItem(mode, 'total_ms')), 0n);
  if (intValue(phaseTotal) !== computedTotal) {
    throw new ContractError(`${label}.total_ms must equal the sum of mode totals: ${phaseTotal} != ${computedTotal}`);
  }
}

// Fail closed unless the attestation exactly binds the candidate and
// identities. Every expectation left null (Python's None) is not checked.
export function verifyAttestation({
  attestation,
  candidateDir = null,
  candidateFingerprint = null,
  candidateRunId = null,
  candidateArtifactId = null,
  candidateRunAttempt = null,
  qualificationRunId = null,
  qualificationRunAttempt = null,
  qualificationSourceSha = null,
  bridgeSourceSha = null,
  upstreamTag = null,
  upstreamCommit = null,
  nativeReleaseTag = null,
  nativeManifestSha256 = null,
  nativeCommit = null,
  emscriptenVersion = null,
  releaseTag = null,
  releaseRebuild = null,
  orchestratorCorrelationId = null,
  harnessSha256 = null,
} = {}) {
  requireExactMapping(attestation, ATTESTATION_KEYS, 'attestation');
  const field = (key) => getItem(attestation, key);

  const schemaVersion = field('schema_version');
  if (!isPyInt(schemaVersion) || !pyEquals(schemaVersion, QUALIFICATION_SCHEMA_VERSION)) {
    throw new ContractError(`unsupported attestation schema_version: ${pyRepr(schemaVersion)}`);
  }
  if (!pyEquals(field('attestation_type'), ATTESTATION_TYPE)) {
    throw new ContractError(`attestation_type must be ${pyRepr(ATTESTATION_TYPE)}, got ${pyRepr(field('attestation_type'))}`);
  }
  if (!pyEquals(field('harness_version'), HARNESS_VERSION)) {
    throw new ContractError(`harness_version must be ${pyRepr(HARNESS_VERSION)}, got ${pyRepr(field('harness_version'))}`);
  }
  if (!pyEquals(field('bridge_repository'), BRIDGE_REPOSITORY)) throw new ContractError(`bridge_repository must be exactly ${pyRepr(BRIDGE_REPOSITORY)}`);
  if (!pyEquals(field('native_repository'), NATIVE_REPOSITORY)) throw new ContractError(`native_repository must be exactly ${pyRepr(NATIVE_REPOSITORY)}`);
  if (!pyEquals(field('upstream_repository'), 'ggml-org/llama.cpp')) throw new ContractError("upstream_repository must be exactly 'ggml-org/llama.cpp'");
  if (!pyEquals(field('candidate_workflow_path'), CANDIDATE_WORKFLOW_PATH)) {
    throw new ContractError(`candidate_workflow_path must be exactly ${pyRepr(CANDIDATE_WORKFLOW_PATH)}`);
  }
  if (!pyEquals(field('qualification_workflow_path'), QUALIFICATION_WORKFLOW_PATH)) {
    throw new ContractError(`qualification_workflow_path must be exactly ${pyRepr(QUALIFICATION_WORKFLOW_PATH)}`);
  }

  const fingerprint = requireStr(attestation, 'candidate_fingerprint', 'attestation');
  requireSha256(fingerprint, 'candidate_fingerprint');
  requireSha256(requireStr(attestation, 'harness_source_sha256', 'attestation'), 'harness_source_sha256');
  requireSha256(requireStr(attestation, 'native_manifest_sha256', 'attestation'), 'native_manifest_sha256');
  for (const key of ['bridge_source_sha', 'upstream_commit', 'native_commit']) {
    const value = requireStr(attestation, key, 'attestation');
    if (COMMIT_RE.exec(value) === null) throw new ContractError(`${key} must be a lowercase 40-hex commit SHA`);
  }
  const runId = requireStr(attestation, 'candidate_run_id', 'attestation');
  if (RUN_ID_RE.exec(runId) === null) throw new ContractError('candidate_run_id must be a positive integer');
  const expectedRunUrl = `https://github.com/${BRIDGE_REPOSITORY}/actions/runs/${runId}`;
  if (!pyEquals(field('candidate_run_url'), expectedRunUrl)) throw new ContractError(`candidate_run_url must be exactly ${expectedRunUrl}`);
  const artifactId = requirePositiveInt(pyGet(attestation, 'candidate_artifact_id'), 'attestation candidate_artifact_id');
  const attempt = pyGet(attestation, 'candidate_run_attempt');
  if (!isPyInt(attempt) || !pyEquals(attempt, 1)) throw new ContractError('attestation candidate_run_attempt must be 1');
  const qualificationId = requireStr(attestation, 'qualification_run_id', 'attestation');
  if (RUN_ID_RE.exec(qualificationId) === null) throw new ContractError('qualification_run_id must be a positive integer');
  if (qualificationId === runId) throw new ContractError('candidate and qualification runs must be distinct');
  const expectedQualificationUrl = `https://github.com/${BRIDGE_REPOSITORY}/actions/runs/${qualificationId}`;
  if (!pyEquals(field('qualification_run_url'), expectedQualificationUrl)) {
    throw new ContractError(`qualification_run_url must be exactly ${expectedQualificationUrl}`);
  }
  const qualificationAttempt = pyGet(attestation, 'qualification_run_attempt');
  if (!isPyInt(qualificationAttempt) || !pyEquals(qualificationAttempt, 1)) {
    throw new ContractError('attestation qualification_run_attempt must be 1');
  }
  const qualificationSha = requireStr(attestation, 'qualification_source_sha', 'attestation');
  if (COMMIT_RE.exec(qualificationSha) === null) throw new ContractError('qualification_source_sha must be a lowercase 40-hex commit SHA');
  requireStr(attestation, 'emscripten_version', 'attestation');
  requireStr(attestation, 'release_tag', 'attestation');
  requireNonNegativeInt(pyGet(attestation, 'release_rebuild'), 'attestation release_rebuild');
  requireCorrelationId(requireStr(attestation, 'orchestrator_correlation_id', 'attestation'));

  const candidateGates = requireExactMapping(field('candidate_gates'), CANDIDATE_GATES, 'candidate_gates');
  const heavy = requireExactMapping(field('heavy_gates'), HEAVY_GATES, 'heavy_gates');
  for (const [name, conclusion] of [...pyItems(candidateGates), ...pyItems(heavy)]) {
    if (!pyEquals(conclusion, 'passed')) throw new ContractError(`qualification gate ${pyRepr(name)} is ${pyRepr(conclusion)}; must be 'passed'`);
  }

  validateQualificationEnvironment(field('qualification_environment'));

  const unproven = requireExactMapping(field('unproven_capabilities'), REQUIRED_UNPROVEN_CAPABILITIES, 'unproven_capabilities');
  for (const [key, expected] of Object.entries(REQUIRED_UNPROVEN_CAPABILITIES)) {
    if (!pyEquals(getItem(unproven, key), expected)) {
      throw new ContractError(`unproven capability ${pyRepr(key)} must stay ${pyRepr(expected)}, got ${pyRepr(getItem(unproven, key))}`);
    }
  }

  const pins = requireExactMapping(field('model_pins'), EXPECTED_MODEL_PINS, 'model_pins');
  for (const [name, expected] of Object.entries(EXPECTED_MODEL_PINS)) {
    if (!pyEquals(getItem(pins, name), expected)) {
      throw new ContractError(`model pin ${pyRepr(name)} mismatch: expected ${expected}, got ${pyRepr(getItem(pins, name))}`);
    }
  }

  const phases = requireExactMapping(field('phases'), HEAVY_GATES, 'phases');
  validatePhase(getItem(phases, 'speech_to_text'), {
    label: 'phases.speech_to_text',
    requiredModes: REQUIRED_SPEECH_MODES,
    phaseKeys: SPEECH_PHASE_KEYS,
    requireWav: false,
  });
  validatePhase(getItem(phases, 'text_to_speech'), {
    label: 'phases.text_to_speech',
    requiredModes: REQUIRED_TTS_MODES,
    phaseKeys: TTS_PHASE_KEYS,
    requireWav: true,
  });

  if (candidateDir !== null && candidateDir !== undefined) {
    const [manifest, computed] = loadCandidate(candidateDir);
    if (computed !== fingerprint) throw new ContractError(`candidate fingerprint mismatch: attestation=${fingerprint}, candidate=${computed}`);
    const manifestBindings = [
      ['bridge_source_sha', getItem(manifest, 'bridge_commit')],
      ['candidate_run_id', getItem(manifest, 'github_run_id')],
      ['candidate_run_url', getItem(manifest, 'github_run_url')],
      ['emscripten_version', getItem(manifest, 'emscripten_version')],
      ['native_commit', getItem(manifest, 'native_commit')],
      ['native_manifest_sha256', getItem(manifest, 'native_manifest_sha256')],
      ['native_release_tag', getItem(manifest, 'native_release_tag')],
      ['orchestrator_correlation_id', getItem(manifest, 'orchestrator_correlation_id')],
      ['release_rebuild', getItem(manifest, 'release_rebuild')],
      ['release_tag', getItem(manifest, 'release_tag')],
      ['upstream_commit', getItem(manifest, 'upstream_commit')],
      ['upstream_tag', getItem(manifest, 'upstream_tag')],
    ];
    for (const [key, expected] of manifestBindings) {
      if (!pyEquals(field(key), expected)) {
        throw new ContractError(`attestation ${key} does not match the candidate manifest: ${pyRepr(field(key))} != ${pyRepr(expected)}`);
      }
    }
    if (!pyEquals(candidateGates, manifestCandidateGates(manifest))) {
      throw new ContractError('attestation candidate_gates do not match the candidate manifest');
    }
  }

  const expectations = [
    ['bridge_source_sha', bridgeSourceSha],
    ['candidate_artifact_id', candidateArtifactId],
    ['candidate_fingerprint', candidateFingerprint],
    ['candidate_run_attempt', candidateRunAttempt],
    ['candidate_run_id', candidateRunId],
    ['qualification_run_attempt', qualificationRunAttempt],
    ['qualification_run_id', qualificationRunId],
    ['qualification_source_sha', qualificationSourceSha],
    ['emscripten_version', emscriptenVersion],
    ['harness_source_sha256', harnessSha256],
    ['native_commit', nativeCommit],
    ['native_manifest_sha256', nativeManifestSha256],
    ['native_release_tag', nativeReleaseTag],
    ['orchestrator_correlation_id', orchestratorCorrelationId],
    ['release_rebuild', releaseRebuild],
    ['release_tag', releaseTag],
    ['upstream_commit', upstreamCommit],
    ['upstream_tag', upstreamTag],
  ];
  for (const [key, expected] of expectations) {
    if (expected === null || expected === undefined) continue;
    if (!pyEquals(field(key), expected)) {
      throw new ContractError(`attestation ${key} mismatch: expected ${pyRepr(expected)}, got ${pyRepr(field(key))}`);
    }
  }

  return {
    verified: true,
    candidate_fingerprint: fingerprint,
    candidate_run_id: runId,
    candidate_artifact_id: artifactId,
    candidate_run_attempt: attempt,
    qualification_run_id: qualificationId,
    qualification_run_attempt: qualificationAttempt,
    qualification_source_sha: qualificationSha,
    release_tag: field('release_tag'),
    bridge_source_sha: field('bridge_source_sha'),
    harness_source_sha256: field('harness_source_sha256'),
  };
}

// --- fetching the candidate through the gh CLI -------------------------------------------

const GH_MAX_BUFFER = 1024 * 1024 * 1024;

// subprocess.run(["gh", *args], capture_output=True, text=True) output as
// Python decodes it: strict UTF-8 with universal newlines.
function decodeText(bytes) {
  return pyUniversalNewlines(pyDecodeUtf8(bytes));
}

export function ghJson(args, { gh = 'gh' } = {}) {
  const proc = spawnSync(gh, args, { stdio: ['inherit', 'pipe', 'pipe'], maxBuffer: GH_MAX_BUFFER });
  if (proc.error) {
    if (typeof proc.error.code === 'string' && typeof proc.error.errno === 'number') {
      throw new ContractError(`could not run the gh CLI: ${osErrorString(proc.error, gh)}`, { cause: proc.error });
    }
    throw proc.error;
  }
  const stdout = decodeText(proc.stdout);
  const stderr = decodeText(proc.stderr);
  if (proc.status !== 0) throw new ContractError(`gh ${args.join(' ')} failed: ${sanitizeDiagnosticText(pyStrip(stderr))}`);
  try {
    return pyJsonLoads(stdout);
  } catch (error) {
    if (error instanceof JSONDecodeError) throw new ContractError(`gh ${args.join(' ')} emitted unparsable JSON: ${error.message}`, { cause: error });
    throw error;
  }
}

// tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory): a new
// private file, returned closed.
function makeTempFile(directory, prefix, suffix) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const name = `${prefix}${randomBytes(6).toString('base64url').toLowerCase().replaceAll('-', '_')}${suffix}`;
    const candidate = pyJoinPath(directory, name);
    try {
      fs.closeSync(fs.openSync(candidate, fs.constants.O_RDWR | fs.constants.O_CREAT | fs.constants.O_EXCL, 0o600));
      return candidate;
    } catch (error) {
      if (error?.code !== 'EEXIST') throw toOSError(error, candidate);
    }
  }
  throw new PyException('OSError', '[Errno 17] No usable temporary file name found (search exhausted)');
}

export function downloadArtifact(artifactId, destination, { artifactType = 'candidate', gh = 'gh' } = {}) {
  if (!isPyInt(artifactId)) throw new ContractError('artifact id must be an integer');
  const parent = pyPathParent(pyPath(String(destination)));
  try {
    fs.mkdirSync(parent, { recursive: true });
  } catch (error) {
    throw toOSError(error, parent);
  }
  const archivePath = makeTempFile(parent, `github-artifact-${artifactId}-`, '.zip');
  try {
    let output;
    try {
      output = fs.openSync(archivePath, 'w');
    } catch (error) {
      throw toOSError(error, archivePath);
    }
    let proc;
    try {
      proc = spawnSync(gh, [
        'api',
        '-H',
        'Accept: application/vnd.github+json',
        `repos/${BRIDGE_REPOSITORY}/actions/artifacts/${artifactId}/zip`,
      ], { stdio: ['inherit', output, 'pipe'], maxBuffer: GH_MAX_BUFFER });
    } finally {
      fs.closeSync(output);
    }
    if (proc.error) throw toOSError(proc.error, gh);
    if (proc.status !== 0) {
      const diagnostic = decodeReplace(proc.stderr);
      throw new ContractError(`could not download exact ${artifactType} artifact: ${sanitizeDiagnosticText(pyStrip(diagnostic))}`);
    }
    extractFlatArtifactArchive(archivePath, destination, { artifactType });
  } finally {
    fs.rmSync(archivePath, { force: true });
  }
}

// Prove the candidate run's identity, then download its unique artifact.
//
// Returns [artifactId, runAttempt]: the attempt the run was just proven to
// report, so the attestation binds the attempt the run actually reported
// rather than one the caller assumed.
export function fetchCandidate(runId, destination, { gh = 'gh' } = {}) {
  if (fullmatch(RUN_ID_RE, runId) === null) throw new ContractError('candidate run id must be a positive integer');
  const repository = ghJson(['api', `repos/${BRIDGE_REPOSITORY}`], { gh });
  const branch = isMapping(repository) ? pyGet(repository, 'default_branch') : null;
  if (typeof branch !== 'string' || !branch) throw new ContractError('could not resolve the bridge default branch');
  const run = ghJson(['api', `repos/${BRIDGE_REPOSITORY}/actions/runs/${runId}`], { gh });
  const headSha = validateWorkflowRun(run, {
    expectedRunId: runId,
    expectedWorkflowPath: CANDIDATE_WORKFLOW_PATH,
    expectedHeadBranch: branch,
    expectedRunAttempt: 1,
  });
  const runAttempt = requirePositiveInt(pyGet(run, 'run_attempt'), 'candidate run run_attempt');
  const comparison = ghJson(['api', `repos/${BRIDGE_REPOSITORY}/compare/${headSha}...${branch}`], { gh });
  const status = isMapping(comparison) ? pyGet(comparison, 'status') : null;
  if (!pyEquals(status, 'ahead') && !pyEquals(status, 'identical')) {
    throw new ContractError(`candidate run head ${headSha} is not reachable from ${branch}`);
  }
  const inventory = ghJson(['api', `repos/${BRIDGE_REPOSITORY}/actions/runs/${runId}/artifacts?per_page=100`], { gh });
  const artifactId = validateArtifactInventory(inventory, { expectedRunId: runId, expectedName: CANDIDATE_ARTIFACT_NAME });
  downloadArtifact(artifactId, destination, { artifactType: 'candidate', gh });
  return [artifactId, runAttempt];
}

// --- the command line --------------------------------------------------------------------------

const required = (flag, type = 'str') => ({ flag, required: true, type });
const optional = (flag, type = 'str', fallback = null) => ({ flag, required: false, type, default: fallback });

// The subcommands and options of release_qualification.py's argparse parser.
// A parsed `qualify` hands qualifyCmd camelCase args: candidateRunId (str),
// speechModelPath, speechMmprojPath, speechAudioPath, ttsModelPath,
// ttsMmprojPath, outputAttestation (normalized path strings), diagnosticsDir
// (path string or null), ttsMaxFrames, speechTimeoutSeconds and
// ttsTimeoutSeconds (ints).
export const COMMANDS = Object.freeze({
  qualify: {
    options: [
      required('--candidate-run-id'),
      required('--speech-model-path', 'path'),
      required('--speech-mmproj-path', 'path'),
      required('--speech-audio-path', 'path'),
      required('--tts-model-path', 'path'),
      required('--tts-mmproj-path', 'path'),
      optional('--tts-max-frames', 'int', 24),
      optional('--speech-timeout-seconds', 'int', 1800),
      optional('--tts-timeout-seconds', 'int', 1800),
      optional('--diagnostics-dir', 'path'),
      required('--output-attestation', 'path'),
    ],
  },
  'verify-run': {
    options: [
      required('--run-json', 'path'),
      required('--artifacts-json', 'path'),
      required('--run-id'),
      required('--workflow-path'),
      required('--head-branch'),
      required('--artifact-name'),
      optional('--run-attempt', 'int', 1),
    ],
  },
  'verify-attestation': {
    options: [
      required('--attestation', 'path'),
      optional('--candidate-dist', 'path'),
      optional('--candidate-fingerprint'),
      optional('--candidate-run-id'),
      optional('--candidate-artifact-id', 'int'),
      optional('--candidate-run-attempt', 'int'),
      optional('--qualification-run-id'),
      optional('--qualification-run-attempt', 'int'),
      optional('--qualification-source-sha'),
      optional('--bridge-commit'),
      optional('--upstream-tag'),
      optional('--upstream-commit'),
      optional('--native-release-tag'),
      optional('--native-manifest-sha256'),
      optional('--native-commit'),
      optional('--emscripten-version'),
      optional('--release-tag'),
      optional('--release-rebuild', 'int'),
      optional('--orchestrator-correlation-id'),
      optional('--harness-dir', 'path'),
    ],
  },
  'candidate-fingerprint': { options: [required('--candidate-dist', 'path')] },
  'harness-digest': { options: [required('--harness-dir', 'path')] },
});

export const PROG = progName(import.meta.url);

// parse_args(argv) of the parser: { command, args }, or an ArgparseExit.
export function parseArgs(argv) {
  return parseCommandLine(argv, { prog: PROG, commands: COMMANDS });
}

function verifyRunCmd(args, write) {
  const run = parseAttestationJson(pyReadText(args.runJson));
  const headSha = validateWorkflowRun(run, {
    expectedRunId: args.runId,
    expectedWorkflowPath: args.workflowPath,
    expectedHeadBranch: args.headBranch,
    expectedRunAttempt: args.runAttempt,
  });
  const inventory = parseAttestationJson(pyReadText(args.artifactsJson));
  const artifactId = validateArtifactInventory(inventory, { expectedRunId: args.runId, expectedName: args.artifactName });
  write(`${pyJsonDumps({ verified: true, artifact_id: artifactId, head_sha: headSha }, { sortKeys: true })}\n`);
  return 0;
}

function verifyAttestationCmd(args, write) {
  const attestation = loadAttestationFile(args.attestation);
  const result = verifyAttestation({
    attestation,
    candidateDir: args.candidateDist !== null ? pyResolve(args.candidateDist) : null,
    candidateFingerprint: args.candidateFingerprint,
    candidateRunId: args.candidateRunId,
    candidateArtifactId: args.candidateArtifactId,
    candidateRunAttempt: args.candidateRunAttempt,
    qualificationRunId: args.qualificationRunId,
    qualificationRunAttempt: args.qualificationRunAttempt,
    qualificationSourceSha: args.qualificationSourceSha,
    bridgeSourceSha: args.bridgeCommit,
    upstreamTag: args.upstreamTag,
    upstreamCommit: args.upstreamCommit,
    nativeReleaseTag: args.nativeReleaseTag,
    nativeManifestSha256: args.nativeManifestSha256,
    nativeCommit: args.nativeCommit,
    emscriptenVersion: args.emscriptenVersion,
    releaseTag: args.releaseTag,
    releaseRebuild: args.releaseRebuild,
    orchestratorCorrelationId: args.orchestratorCorrelationId,
    harnessSha256: args.harnessDir !== null ? harnessSourceSha256(pyResolve(args.harnessDir)) : null,
  });
  write(`${pyJsonDumps(result, { indent: 2, sortKeys: true })}\n`);
  return 0;
}

function candidateFingerprintCmd(args, write) {
  const [, fingerprint] = loadCandidate(pyResolve(args.candidateDist));
  write(`${fingerprint}\n`);
  return 0;
}

function harnessDigestCmd(args, write) {
  write(`${harnessSourceSha256(pyResolve(args.harnessDir))}\n`);
  return 0;
}

// `qualify` runs the heavy gates, which live in ./qualify.mjs.
async function qualifyCmd(args, write) {
  const module = await import('./qualify.mjs');
  return module.qualifyCmd(args, write);
}

const RUNNERS = Object.freeze({
  qualify: qualifyCmd,
  'verify-run': verifyRunCmd,
  'verify-attestation': verifyAttestationCmd,
  'candidate-fingerprint': candidateFingerprintCmd,
  'harness-digest': harnessDigestCmd,
});

// str(exc) of a ContractError or OSError.
function errorText(error) {
  if (error instanceof ContractError || error instanceof PyException) return error.message;
  return osErrorString(error, error.path ?? null, error.dest ?? null);
}

// main()'s `except (ContractError, OSError)`: `error: <sanitized message>`
// on stderr and exit status 1; anything else escapes as a traceback.
function reportFailure(error) {
  if (error instanceof ContractError || isOSError(error)) {
    throw new SystemExitError(`error: ${sanitizeDiagnosticText(errorText(error))}`);
  }
  throw error;
}

// The CLI: `main(argv, write)` parses argv like release_qualification.py's
// argparse parser, prints through write() and returns the exit status (a
// Promise of it for `qualify`).
export function main(argv, write) {
  const { command, args } = parseArgs(argv);
  let result;
  try {
    result = RUNNERS[command](args, write);
  } catch (error) {
    reportFailure(error);
  }
  if (result && typeof result.then === 'function') return result.catch(reportFailure);
  return result;
}

export { ArgparseExit };

// Not awaited: `qualify` imports qualify.mjs, which imports this module, and
// that import cannot settle while this module's evaluation awaits it.
if (import.meta.main) runCliAsync(main);
