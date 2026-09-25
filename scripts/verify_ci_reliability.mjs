#!/usr/bin/env node
// Static checks for the publication-safety invariants of the CI, candidate,
// qualification, publication and orchestration workflows: permissions,
// immutability, environment gates, publication PAT handling, model and
// toolchain pins, fail-closed guards, first-attempt-only runs, artifact
// download by immutable ID, and that CI runs the contract tests.
//
// It checks facts, not wording: workflows are parsed with `yaml`, and commands
// are looked for in resolved `run` scripts, so indentation, comments and step
// names may change freely unless a check names a step as an anchor.

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { isMap, isScalar, parse as parseYaml, parseDocument } from 'yaml';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const WORKFLOWS = {
  ci: '.github/workflows/ci.yml',
  candidate: '.github/workflows/bridge_candidate.yml',
  qualification: '.github/workflows/bridge_qualification.yml',
  publish: '.github/workflows/publish_assets.yml',
  autoUpdate: '.github/workflows/auto_llama_cpp_update.yml',
};

// ---------------------------------------------------------------------------
// Model pins
// ---------------------------------------------------------------------------

// These files hand-copy the model/projector SHA-256 pins. A rotation that misses
// one leaves CI green while a contributor hits an opaque checksum failure -- or,
// for bridge_candidate.yml, while the candidate job that consumes the pins fails.
// publish_assets.yml neither builds nor smokes anything, so it holds no pins.
export const MODEL_SHA_PIN_FILES = ['CONTRIBUTING.md', WORKFLOWS.ci, WORKFLOWS.candidate];
export const EXPECTED_MODEL_SHA_PIN_COUNT = 7;
const SHA256_HEX = /(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])/g;
// A pin is a 64-hex literal introduced by a `--<name>-sha256` smoke CLI flag or a
// `*_SHA256:` workflow env key, joined to it by spacing, `=`, quotes, or a shell
// line continuation. A bare newline never joins the two, so an empty env value
// cannot adopt the next 64-hex line. Any other 64-hex literal is reported and
// skipped, never compared as a model pin. Without the m flag `$` is the end of
// the input, like Python's `\Z`.
const MODEL_SHA_PIN_MARKER = /(--[A-Za-z0-9][A-Za-z0-9-]*-sha256|_SHA256:)(?:[ \t="'`]|\\\r?\n)+$/;
// The set check above is role-blind: swapping two pins between roles inside one
// file keeps every value present and passes. Both workflows name the role in
// the env key, and every workflow pin is compared against
// release_qualification.EXPECTED_MODEL_PINS, which every attestation's
// model_pins must equal, so a swap applied identically everywhere still fails.
// CONTRIBUTING.md's bare `--model-sha256` / `--mmproj-sha256` flags carry no
// role; it lives in the `--model-url` / `--model-path` / `--mmproj-path` value
// beside each flag, which names a distinct model or projector file.
export const WORKFLOW_MODEL_SHA_PIN_FILES = [WORKFLOWS.ci, WORKFLOWS.candidate];
const WORKFLOW_MODEL_SHA_PIN_ASSIGNMENT = /^[ \t]*([A-Z][A-Z0-9_]*_SHA256):[ \t]*["']?([0-9a-fA-F]{64})["']?[ \t]*$/gm;
// bridge_qualification.yml hand-copies its own five-role subset: the speech,
// TTS and audio inputs the heavy gates download. Without the roster a dropped
// `<ROLE>_URL` + `<ROLE>_SHA256` pair would pass, because ci.yml still binds the
// canonical role. Its speech audio pin is the eighth role, documented in no
// command block, which is why the canonical map holds eight names against
// EXPECTED_MODEL_SHA_PIN_COUNT.
export const MODEL_PIN_ROLE_FILES = [WORKFLOWS.ci, WORKFLOWS.candidate, WORKFLOWS.qualification];
export const QUALIFICATION_MODEL_PIN_FILE = WORKFLOWS.qualification;
export const QUALIFICATION_MODEL_PIN_ROLES = [
  'LLAMA_WEBGPU_SPEECH_AUDIO_SHA256',
  'LLAMA_WEBGPU_SPEECH_MMPROJ_SHA256',
  'LLAMA_WEBGPU_SPEECH_MODEL_SHA256',
  'LLAMA_WEBGPU_TTS_MMPROJ_SHA256',
  'LLAMA_WEBGPU_TTS_MODEL_SHA256',
];
// The canonical name of a role does not follow mechanically from its env key:
// LLAMA_WEBGPU_SMOKE_MODEL_SHA256 is state_smoke_model_sha256. Mutable, so a
// test can add a name and remove it again.
export const CANONICAL_MODEL_PIN_NAMES = {
  LLAMA_WEBGPU_MULTIMODAL_MMPROJ_SHA256: 'multimodal_mmproj_sha256',
  LLAMA_WEBGPU_MULTIMODAL_MODEL_SHA256: 'multimodal_model_sha256',
  LLAMA_WEBGPU_SMOKE_MODEL_SHA256: 'state_smoke_model_sha256',
  LLAMA_WEBGPU_SPEECH_AUDIO_SHA256: 'speech_audio_sha256',
  LLAMA_WEBGPU_SPEECH_MMPROJ_SHA256: 'speech_mmproj_sha256',
  LLAMA_WEBGPU_SPEECH_MODEL_SHA256: 'speech_model_sha256',
  LLAMA_WEBGPU_TTS_MMPROJ_SHA256: 'tts_mmproj_sha256',
  LLAMA_WEBGPU_TTS_MODEL_SHA256: 'tts_model_sha256',
};
const MODEL_URL_ASSIGNMENT = /^[ \t]*(LLAMA_WEBGPU_[A-Z0-9_]*)_URL:[ \t]*["']?(\S+?)["']?(?:[ \t]+#[^\n]*)?[ \t]*$/gm;
const HUGGING_FACE_REVISION_SEGMENT = /\/resolve\/([^/]+)\//;
const IMMUTABLE_REVISION = /^[0-9a-f]{40}$/;
// These three roles resolve through the mutable `main` branch of a third-party
// repository, so the pinned SHA gates the bytes but names no retrievable
// revision once upstream moves. Listing them keeps a newly introduced mutable
// URL from joining them unnoticed.
export const MUTABLE_REVISION_MODEL_URL_ROLES = [
  'LLAMA_WEBGPU_MULTIMODAL_MMPROJ',
  'LLAMA_WEBGPU_MULTIMODAL_MODEL',
  'LLAMA_WEBGPU_SMOKE_MODEL',
];
// The speech audio sample is not a Hugging Face object and carries no revision
// segment at all.
export const UNVERSIONED_MODEL_URL_ROLES = ['LLAMA_WEBGPU_SPEECH_AUDIO'];
const MARKDOWN_MODEL_ROLE_FLAG = /--(?:model-url|model-path|mmproj-path)[ \t]+(\S+)/;
const MARKDOWN_MODEL_PIN_FLAG = /--[A-Za-z0-9][A-Za-z0-9-]*-sha256[ \t]+([0-9a-fA-F]{64})/;
export const EXPECTED_CANONICAL_MODEL_PIN_COUNT = 8;

// Python's str.splitlines() boundaries, so line numbers match the Python tools.
const LINE_BREAKS = '\\n\\r\\v\\f\\x1c\\x1d\\x1e\\x85\\u2028\\u2029';
const LINE_BREAK = new RegExp(`\\r\\n|[${LINE_BREAKS}]`);
const LINE_WITH_END = new RegExp(`[^${LINE_BREAKS}]*(?:\\r\\n|[${LINE_BREAKS}])|[^${LINE_BREAKS}]+$`, 'g');

function splitLines(text) {
  const lines = text.split(LINE_BREAK);
  if (lines.length > 0 && lines[lines.length - 1] === '') lines.pop();
  return lines;
}

function lineNumberAt(content, index) {
  let line = 1;
  for (let i = content.indexOf('\n'); i !== -1 && i < index; i = content.indexOf('\n', i + 1)) line += 1;
  return line;
}

const sorted = (values) => [...values].sort();
const sameList = (left, right) => left.length === right.length && left.every((value, i) => value === right[i]);
const has = (object, key) => Object.prototype.hasOwnProperty.call(object, key);

/**
 * Reads EXPECTED_MODEL_PINS from scripts/release_qualification.py: its
 * `NAME_SHA256 = ("<hex>")` constants and the dict naming them. Throws unless it
 * resolves exactly eight distinct names, each to a 64-hex constant.
 */
export function parseExpectedModelPins(source) {
  const constants = new Map();
  const constant = /^([A-Z][A-Z0-9_]*_SHA256)[ \t]*=[ \t]*\(?\s*["']([0-9a-fA-F]{64})["']\s*\)?[ \t]*$/gm;
  for (const match of source.matchAll(constant)) {
    if (constants.has(match[1])) throw new Error(`release_qualification.py defines ${match[1]} twice`);
    constants.set(match[1], match[2]);
  }
  const block = /^EXPECTED_MODEL_PINS[ \t]*=[ \t]*\{([^}]*)\}/m.exec(source);
  if (!block) throw new Error('release_qualification.py has no EXPECTED_MODEL_PINS = {...} literal');
  const body = block[1].replace(/#[^\n]*/g, '');
  const entry = /["']([a-z0-9_]+)["']\s*:\s*([A-Z][A-Z0-9_]*)\s*(?:,|$)/g;
  const leftover = body.replace(entry, '').trim();
  if (leftover) throw new Error(`EXPECTED_MODEL_PINS has an entry this reader cannot resolve: ${leftover.split('\n')[0]}`);
  const pins = {};
  for (const [, name, reference] of body.matchAll(entry)) {
    if (has(pins, name)) throw new Error(`EXPECTED_MODEL_PINS names ${name} twice`);
    if (!constants.has(reference)) {
      throw new Error(`EXPECTED_MODEL_PINS maps ${name} to ${reference}, which is not a 64-hex constant`);
    }
    pins[name] = constants.get(reference);
  }
  const found = Object.keys(pins).length;
  if (found !== EXPECTED_CANONICAL_MODEL_PIN_COUNT) {
    throw new Error(`EXPECTED_MODEL_PINS declares ${found} pins, expected ${EXPECTED_CANONICAL_MODEL_PIN_COUNT}`);
  }
  return pins;
}

let expectedModelPinsError = '';
function loadExpectedModelPins() {
  try {
    return parseExpectedModelPins(fs.readFileSync(path.join(ROOT, 'scripts/release_qualification.py'), 'utf8'));
  } catch (error) {
    expectedModelPinsError = `cannot read release_qualification.EXPECTED_MODEL_PINS: ${error.message}`;
    return {};
  }
}
export const EXPECTED_MODEL_PINS = Object.freeze(loadExpectedModelPins());

export function extractModelShaPins(relativePath, content, errors) {
  const pins = [];
  for (const match of content.matchAll(SHA256_HEX)) {
    if (!MODEL_SHA_PIN_MARKER.test(content.slice(0, match.index))) {
      errors.push(
        `${relativePath}:${lineNumberAt(content, match.index)} has 64-hex literal ${match[0]} that is not a `
        + '--<name>-sha256 argument or a *_SHA256 workflow env value; the model '
        + 'pin consistency check only understands model pins',
      );
      continue;
    }
    pins.push(match[0]);
  }
  const duplicates = sorted(new Set(pins.filter((pin, i) => pins.indexOf(pin) !== i)));
  if (duplicates.length > 0) {
    errors.push(
      `${relativePath} repeats model SHA-256 pin(s) ${duplicates.join(', ')}; `
      + 'each pinned model/projector must appear exactly once per file',
    );
  }
  if (pins.length !== EXPECTED_MODEL_SHA_PIN_COUNT) {
    errors.push(
      `${relativePath} declares ${pins.length} model SHA-256 pins, expected `
      + `${EXPECTED_MODEL_SHA_PIN_COUNT}; update EXPECTED_MODEL_SHA_PIN_COUNT in `
      + 'scripts/verify_ci_reliability.mjs when the pinned model set changes',
    );
  }
  return new Set(pins);
}

export function requireIdenticalModelShaPins(pinsByFile, errors) {
  const entries = Object.entries(pinsByFile);
  if (new Set(entries.map(([, pins]) => sorted(pins).join(','))).size <= 1) return;
  const reported = errors.length;
  for (const [relativePath, pins] of entries) {
    const others = entries.filter(([other]) => other !== relativePath).map(([, other]) => other);
    if (others.length === 0) continue;
    const unique = sorted([...pins].filter((pin) => others.every((other) => !other.has(pin))));
    const missing = sorted([...others[0]].filter((pin) => others.every((other) => other.has(pin)) && !pins.has(pin)));
    if (unique.length === 0 && missing.length === 0) continue;
    const details = [];
    if (unique.length > 0) details.push(`only in this file: ${unique.join(', ')}`);
    if (missing.length > 0) details.push(`present everywhere else but missing here: ${missing.join(', ')}`);
    errors.push(
      `${relativePath} model SHA-256 pins diverge (${details.join('; ')}); `
      + `every pin must be byte-identical across ${MODEL_SHA_PIN_FILES.join(', ')}`,
    );
  }
  if (errors.length === reported) {
    errors.push(
      `model SHA-256 pins differ across ${MODEL_SHA_PIN_FILES.join(', ')} with partially overlapping sets: `
      + entries.map(([file, pins]) => `${file}=[${sorted(pins).join(', ')}]`).join('; '),
    );
  }
}

export function extractModelShaPinRoles(relativePath, content, errors) {
  const roles = {};
  for (const match of content.matchAll(WORKFLOW_MODEL_SHA_PIN_ASSIGNMENT)) {
    const [, key, pin] = match;
    if (has(roles, key)) {
      errors.push(
        `${relativePath}:${lineNumberAt(content, match.index)} redefines model SHA-256 env key ${key} `
        + `(was ${roles[key]}, now ${pin}); each role must be declared once`,
      );
    }
    roles[key] = pin;
  }
  return roles;
}

export function requireCompleteWorkflowModelShaPinRoles(relativePath, roles, errors) {
  const declared = Object.keys(roles).length;
  if (declared !== EXPECTED_MODEL_SHA_PIN_COUNT) {
    errors.push(
      `${relativePath} declares ${declared} role-bearing model SHA-256 env keys, expected `
      + `${EXPECTED_MODEL_SHA_PIN_COUNT}; every pin must be a literal \`<ROLE>_SHA256: <64-hex>\` `
      + 'assignment so the role-to-hash check can compare it across workflows',
    );
  }
}

export function requireQualificationModelShaPinRoles(roles, errors) {
  const declared = sorted(Object.keys(roles));
  const expected = sorted(QUALIFICATION_MODEL_PIN_ROLES);
  if (!sameList(declared, expected)) {
    errors.push(
      `${QUALIFICATION_MODEL_PIN_FILE} declares model SHA-256 env keys ${declared.join(', ') || 'none'}, `
      + `expected ${expected.join(', ')}; the heavy gates download every one of them, so a dropped `
      + 'role would run them against an unpinned file',
    );
  }
}

export function requireIdenticalWorkflowModelShaPinRoles(rolesByFile, errors) {
  const [basePath, ...otherPaths] = Object.keys(rolesByFile);
  const base = rolesByFile[basePath];
  for (const otherPath of otherPaths) {
    const other = rolesByFile[otherPath];
    for (const key of sorted(new Set([...Object.keys(base), ...Object.keys(other)]))) {
      const basePin = has(base, key) ? base[key] : 'absent';
      const otherPin = has(other, key) ? other[key] : 'absent';
      if (basePin !== otherPin) {
        errors.push(
          `model SHA-256 env key ${key} is bound to ${basePin} in ${basePath} but to ${otherPin} `
          + `in ${otherPath}; each role must carry the same pin in both workflows because every `
          + 'SHA is paired with a role-specific model URL',
        );
      }
    }
  }
}

export function modelFileName(urlOrPath) {
  const trimmed = urlOrPath.split('?', 1)[0].split('#', 1)[0].replace(/\/+$/, '');
  return trimmed.slice(trimmed.lastIndexOf('/') + 1);
}

export function extractModelUrls(relativePath, content, errors) {
  const urls = {};
  for (const match of content.matchAll(MODEL_URL_ASSIGNMENT)) {
    const [, role, url] = match;
    if (has(urls, role)) {
      errors.push(
        `${relativePath}:${lineNumberAt(content, match.index)} redefines model URL env key `
        + `${role}_URL (was ${urls[role]}, now ${url}); each role must be declared once`,
      );
    }
    urls[role] = url;
  }
  return urls;
}

export function requireCanonicalModelShaPins(rolesByFile, errors) {
  const unmapped = sorted(new Set(Object.values(CANONICAL_MODEL_PIN_NAMES)))
    .filter((name) => !has(EXPECTED_MODEL_PINS, name));
  if (unmapped.length > 0) {
    errors.push(
      `CANONICAL_MODEL_PIN_NAMES maps to canonical pin name(s) ${unmapped.join(', ')} `
      + 'that release_qualification.EXPECTED_MODEL_PINS does not declare',
    );
  }
  const covered = new Set();
  for (const relativePath of sorted(Object.keys(rolesByFile))) {
    const roles = rolesByFile[relativePath];
    for (const key of sorted(Object.keys(roles))) {
      if (!has(CANONICAL_MODEL_PIN_NAMES, key)) {
        errors.push(
          `${relativePath} binds model SHA-256 env key ${key} that CANONICAL_MODEL_PIN_NAMES does `
          + 'not name; add the role there and to release_qualification.EXPECTED_MODEL_PINS so the '
          + 'pin is compared against a canonical value',
        );
        continue;
      }
      const name = CANONICAL_MODEL_PIN_NAMES[key];
      covered.add(name);
      const expected = has(EXPECTED_MODEL_PINS, name) ? EXPECTED_MODEL_PINS[name] : 'absent';
      if (roles[key] !== expected) {
        errors.push(
          `${relativePath} binds ${key} to ${roles[key]} but canonical ${name} is ${expected}; `
          + 'every workflow copy must equal the pin release_qualification.py checks the '
          + 'attestation against, so a role swap repeated across workflows still fails here',
        );
      }
    }
  }
  const missing = sorted(Object.keys(EXPECTED_MODEL_PINS)).filter((name) => !covered.has(name));
  if (missing.length > 0) {
    errors.push(
      `canonical model pin(s) ${missing.join(', ')} are declared in `
      + 'release_qualification.EXPECTED_MODEL_PINS but bound by no <ROLE>_SHA256 env key in '
      + MODEL_PIN_ROLE_FILES.join(', '),
    );
  }
}

export function requirePairedModelUrlsAndPins(urlsByFile, rolesByFile, errors) {
  for (const relativePath of sorted(Object.keys(urlsByFile))) {
    const urls = urlsByFile[relativePath];
    const roles = rolesByFile[relativePath];
    const unpinned = sorted(Object.keys(urls)).filter((role) => !has(roles, `${role}_SHA256`));
    if (unpinned.length > 0) {
      errors.push(
        `${relativePath} declares model URL env key(s) ${unpinned.map((role) => `${role}_URL`).join(', ')} `
        + 'with no matching <ROLE>_SHA256 pin; an unpinned download is never checksum-verified',
      );
    }
    const undownloaded = sorted(Object.keys(roles)).filter((key) => !has(urls, key.replace(/_SHA256$/, '')));
    if (undownloaded.length > 0) {
      errors.push(
        `${relativePath} declares model SHA-256 env key(s) ${undownloaded.join(', ')} with no `
        + 'matching <ROLE>_URL; a pin with no URL beside it cannot be checked for role parity',
      );
    }
  }
}

export function requireIdenticalModelUrls(urlsByFile, errors) {
  const roles = sorted(new Set(Object.values(urlsByFile).flatMap((urls) => Object.keys(urls))));
  for (const role of roles) {
    const bound = sorted(Object.keys(urlsByFile))
      .filter((file) => has(urlsByFile[file], role))
      .map((file) => [file, urlsByFile[file][role]]);
    if (new Set(bound.map(([, value]) => value)).size > 1) {
      errors.push(
        `model URL env key ${role}_URL requests different bytes per workflow (`
        + bound.map(([file, value]) => `${value} in ${file}`).join('; ')
        + '); every workflow that downloads a role must request the same object',
      );
    }
  }
}

export function requirePinnedModelUrlRevisions(urlsByFile, errors) {
  const mutable = new Set();
  const unversioned = new Set();
  for (const urls of Object.values(urlsByFile)) {
    for (const [role, url] of Object.entries(urls)) {
      const segment = HUGGING_FACE_REVISION_SEGMENT.exec(url);
      if (segment === null) unversioned.add(role);
      else if (!IMMUTABLE_REVISION.test(segment[1])) mutable.add(role);
    }
  }
  if (!sameList(sorted(mutable), sorted(MUTABLE_REVISION_MODEL_URL_ROLES))) {
    errors.push(
      `model URL env keys resolving through a mutable revision are ${sorted(mutable).join(', ') || 'none'}, `
      + `expected ${sorted(MUTABLE_REVISION_MODEL_URL_ROLES).join(', ')}; update `
      + 'MUTABLE_REVISION_MODEL_URL_ROLES when a role gains or loses an immutable 40-hex revision',
    );
  }
  if (!sameList(sorted(unversioned), sorted(UNVERSIONED_MODEL_URL_ROLES))) {
    errors.push(
      `model URL env keys carrying no revision segment are ${sorted(unversioned).join(', ') || 'none'}, `
      + `expected ${sorted(UNVERSIONED_MODEL_URL_ROLES).join(', ')}; update `
      + 'UNVERSIONED_MODEL_URL_ROLES when a role moves to or from a revisioned host',
    );
  }
}

export function modelFileNameRoles(urlsByFile, errors) {
  const names = {};
  for (const relativePath of sorted(Object.keys(urlsByFile))) {
    const urls = urlsByFile[relativePath];
    for (const role of sorted(Object.keys(urls))) {
      if (!has(CANONICAL_MODEL_PIN_NAMES, `${role}_SHA256`)) continue;
      const name = CANONICAL_MODEL_PIN_NAMES[`${role}_SHA256`];
      const fileName = modelFileName(urls[role]);
      if (!has(names, fileName)) names[fileName] = name;
      if (names[fileName] !== name) {
        errors.push(
          `${relativePath} binds ${role}_URL to a file named ${fileName} that canonical ${names[fileName]} `
          + 'also downloads; every role must download a distinctly named file so a documented '
          + 'command that names the file names its role',
        );
      }
    }
  }
  return names;
}

export function requireMarkdownModelPinRoles(relativePath, content, fileNameRoles, errors) {
  let pending = null;
  const paired = new Map();
  splitLines(content).forEach((line, index) => {
    const lineNumber = index + 1;
    const roleFlag = MARKDOWN_MODEL_ROLE_FLAG.exec(line);
    if (roleFlag !== null) pending = modelFileName(roleFlag[1]);
    const pinFlag = MARKDOWN_MODEL_PIN_FLAG.exec(line);
    if (pinFlag === null) return;
    const pin = pinFlag[1];
    const fileName = pending;
    pending = null;
    if (fileName === null) {
      errors.push(
        `${relativePath}:${lineNumber} pins ${pin} with no preceding `
        + '--model-url/--model-path/--mmproj-path naming the file it pins; '
        + 'the bare --*-sha256 flag carries no role',
      );
      return;
    }
    if (!has(fileNameRoles, fileName)) {
      errors.push(
        `${relativePath}:${lineNumber} pins ${pin} for ${fileName}, which no <ROLE>_URL in `
        + `${MODEL_PIN_ROLE_FILES.join(', ')} downloads; documented commands must name a file `
        + 'some workflow pins',
      );
      return;
    }
    const name = fileNameRoles[fileName];
    if (paired.has(name)) {
      errors.push(
        `${relativePath}:${lineNumber} pins canonical ${name} again (first at line `
        + `${paired.get(name)}); each role must be documented once`,
      );
    } else {
      paired.set(name, lineNumber);
    }
    const expected = has(EXPECTED_MODEL_PINS, name) ? EXPECTED_MODEL_PINS[name] : 'absent';
    if (pin !== expected) {
      errors.push(
        `${relativePath}:${lineNumber} pins ${fileName} at ${pin} but canonical ${name} is `
        + `${expected}; a documented pin must match the role named by the `
        + '--model-url/--model-path/--mmproj-path value beside it',
      );
    }
  });
  if (paired.size !== EXPECTED_MODEL_SHA_PIN_COUNT) {
    errors.push(
      `${relativePath} pairs ${paired.size} documented model SHA-256 pins with a role, expected `
      + `${EXPECTED_MODEL_SHA_PIN_COUNT}; every pin must sit on or after the line naming the file it pins`,
    );
  }
}

// ---------------------------------------------------------------------------
// Publication PAT contract
// ---------------------------------------------------------------------------

export const PUBLICATION_PAT_NAME = 'WEBGPU_BRIDGE_ASSETS_PAT';
const PUBLICATION_PAT_REFERENCE = /secrets\s*(?:\.\s*WEBGPU_BRIDGE_ASSETS_PAT|\[\s*['"]WEBGPU_BRIDGE_ASSETS_PAT['"]\s*\])/;
export const PUBLICATION_PAT_GUARD_ERROR = 'error: WEBGPU_BRIDGE_ASSETS_PAT is required for asset publication';
export const EXPECTED_PUBLICATION_PAT_STEPS = [
  ['publish-assets', 'Verify source ancestry and classify exact remote state'],
  ['publish-assets', 'Apply only the classified ref mutation'],
  ['publish-assets', 'Re-fetch and finish or safely classify partial publication'],
];
const READ_ONLY_ACTIONS_PERMISSIONS = { actions: 'read', contents: 'read' };
const SECRETS_CONTEXT = /\bsecrets\b/i;

const isMapping = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const mapping = (value) => (isMapping(value) ? value : {});

function containsMatch(value, pattern) {
  if (typeof value === 'string') return pattern.test(value);
  if (Array.isArray(value)) return value.some((item) => containsMatch(item, pattern));
  if (isMapping(value)) return Object.values(value).some((item) => containsMatch(item, pattern));
  return value !== null && value !== undefined && pattern.test(String(value));
}
const containsPat = (value) => containsMatch(value, PUBLICATION_PAT_REFERENCE);

/**
 * Resolves every step with its effective env (workflow, then job, then step,
 * YAML merge keys applied), and reports each PAT reference outside an env
 * mapping value, since only env values can be checked for the fail-closed guard.
 */
export function resolveWorkflowSteps(workflow) {
  let root;
  try {
    root = mapping(parseYaml(workflow, { merge: true }));
  } catch (error) {
    return { steps: [], patOutsideEnv: [], errors: [`workflow YAML cannot be resolved: ${error.message}`] };
  }
  const patOutsideEnv = [];
  const { env: rootEnv, jobs, ...rootOutsideEnv } = root;
  if (containsPat(rootOutsideEnv) || (!isMapping(rootEnv) && containsPat(rootEnv))) {
    patOutsideEnv.push('workflow root properties');
  }
  const steps = [];
  for (const [jobName, rawJob] of Object.entries(mapping(jobs))) {
    const { env: jobEnv, steps: rawSteps, ...jobOutsideEnv } = mapping(rawJob);
    if (containsPat(jobOutsideEnv) || (!isMapping(jobEnv) && containsPat(jobEnv))) {
      patOutsideEnv.push(`job ${jobName} properties`);
    }
    const inheritedEnv = { ...mapping(rootEnv), ...mapping(jobEnv) };
    (Array.isArray(rawSteps) ? rawSteps : []).forEach((rawStep, index) => {
      const step = mapping(rawStep);
      const { env: stepEnv, ...stepOutsideEnv } = step;
      const name = typeof step.name === 'string' ? step.name : '';
      steps.push({
        job: jobName,
        index,
        name,
        id: typeof step.id === 'string' ? step.id : '',
        uses: typeof step.uses === 'string' ? step.uses : '',
        with: mapping(step.with),
        env: { ...inheritedEnv, ...mapping(stepEnv) },
        run: typeof step.run === 'string' ? step.run : '',
        raw: step,
        patOutsideEnv: containsPat(stepOutsideEnv) || (!isMapping(stepEnv) && containsPat(stepEnv))
          ? `step ${jobName}/${name || index} properties`
          : '',
      });
    });
  }
  return { steps, patOutsideEnv, errors: [] };
}

const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function shellVariableReference(variable) {
  const escaped = escapeRegExp(variable);
  return new RegExp(`\\$(?:\\{${escaped}(?:[^A-Za-z0-9_][^}]*)?\\}|${escaped}\\b)`);
}

function failClosedGuardEnd(script, variable) {
  const blankOrComment = '[ \\t]*(?:#[^\\r\\n]*)?\\r?\\n';
  // No m flag: ^ is the start of the script and $ its end.
  const guard = new RegExp(
    `^(?:${blankOrComment})*`
    + '[ \\t]*set[ \\t]+-euo[ \\t]+pipefail[ \\t]*\\r?\\n'
    + `(?:${blankOrComment})*`
    + `[ \\t]*if[ \\t]+\\[[ \\t]+-z[ \\t]+"\\$\\{${escapeRegExp(variable)}\\}"[ \\t]+\\];`
    + '[ \\t]+then[ \\t]*\\r?\\n'
    + `[ \\t]*echo[ \\t]+"${escapeRegExp(PUBLICATION_PAT_GUARD_ERROR)}"[ \\t]*\\r?\\n`
    + '[ \\t]*exit[ \\t]+[1-9][0-9]*[ \\t]*\\r?\\n'
    + '[ \\t]*fi(?:[ \\t]*\\r?\\n|[ \\t]*$)',
  );
  const match = guard.exec(script);
  return match ? match[0].length : -1;
}

function firstSensitiveCommand(script) {
  let offset = 0;
  for (const line of script.match(LINE_WITH_END) ?? []) {
    const code = line.trimStart();
    if (!code || code.startsWith('#')) {
      offset += line.length;
      continue;
    }
    const matches = [
      /\b(?:gh|curl|wget)\b/.exec(code),
      /\bgit\b[^\n]*?\b(?:clone|fetch|pull|push|ls-remote|remote|add|commit|tag)\b/.exec(code),
    ].filter((match) => match !== null);
    if (matches.length > 0) {
      const match = matches.reduce((first, item) => (item.index < first.index ? item : first));
      return [offset + line.length - code.length + match.index, match[0]];
    }
    offset += line.length;
  }
  return null;
}

function credentialLoggingErrors(script, variables) {
  const collapsed = script.replace(/\\\r?\n/g, ' ');
  const found = [];
  if (/^\s*set\s+(?:-[A-Za-z]*x[A-Za-z]*\b|-o\s+xtrace\b)/m.test(collapsed)) found.push('set -x/xtrace');
  if (/\bprintenv\b/.test(collapsed)) found.push('printenv');
  if (/(?:^|[;&|])\s*(?:command\s+)?["']?(?:env|\/(?:[^/\s;&|]+\/)*env)["']?\s*(?:$|[;&|>])/m.test(collapsed)) {
    found.push('bare env');
  }
  const lines = splitLines(collapsed);
  for (const variable of variables) {
    const reference = shellVariableReference(variable);
    if (lines.some((line) => /\b(?:echo|printf)\b/.test(line) && reference.test(line))) {
      found.push(`echo/printf of ${variable}`);
    }
  }
  return found;
}

/**
 * The publication PAT may be bound only through a step's resolved env, and only
 * in the expected steps. Each such step must open with `set -euo pipefail` and
 * the canonical empty-token guard, run no network command before it, and never
 * print the credential.
 */
export function validatePublicationPatContract(workflow, expectedSteps) {
  const resolved = resolveWorkflowSteps(workflow);
  if (resolved.errors.length > 0) return resolved.errors;
  const errors = resolved.patOutsideEnv.map(
    (location) => `${location} references ${PUBLICATION_PAT_NAME} outside resolved env`,
  );
  const patSteps = [];
  for (const step of resolved.steps) {
    if (step.patOutsideEnv) {
      errors.push(`${step.patOutsideEnv} references ${PUBLICATION_PAT_NAME} outside resolved env`);
    }
    const variables = Object.entries(step.env).filter(([, value]) => containsPat(value)).map(([key]) => key);
    if (variables.length > 0) patSteps.push([step, variables]);
  }
  const actual = patSteps.map(([step]) => [step.job, step.name]);
  const keys = (pairs) => pairs.map((pair) => JSON.stringify(pair)).sort();
  if (!sameList(keys(actual), keys(expectedSteps))) {
    errors.push(`PAT-bearing steps are ${JSON.stringify(actual)}, expected exactly ${JSON.stringify(expectedSteps)}`);
  }
  for (const [step, variables] of patSteps) {
    const where = `${step.job}/${step.name}`;
    for (const variable of variables) {
      if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(variable)) {
        errors.push(`${where} binds the PAT to invalid shell variable ${JSON.stringify(variable)}`);
        continue;
      }
      const guardEnd = failClosedGuardEnd(step.run, variable);
      if (guardEnd < 0) {
        errors.push(
          `${where} must start with set -euo pipefail followed immediately by the canonical `
          + `executable empty-token guard for ${variable}`,
        );
        continue;
      }
      const sensitive = firstSensitiveCommand(step.run);
      if (sensitive !== null && sensitive[0] < guardEnd) {
        errors.push(`${where} runs sensitive command '${sensitive[1]}' before its ${variable} credential guard`);
      }
    }
    for (const leak of credentialLoggingErrors(step.run, variables)) {
      errors.push(`${where} exposes the PAT through ${leak}`);
    }
  }
  return errors;
}

// ---------------------------------------------------------------------------
// Workflow model
// ---------------------------------------------------------------------------

class Workflow {
  constructor(relativePath, text, errors) {
    this.path = relativePath;
    this.text = text;
    this.data = {};
    this.document = null;
    try {
      this.document = parseDocument(text, { merge: true });
      if (this.document.errors.length > 0) throw this.document.errors[0];
      this.data = mapping(this.document.toJS({ maxAliasCount: 100 }));
    } catch (error) {
      errors.push(`${relativePath}: invalid YAML: ${error.message}`);
    }
    const resolved = resolveWorkflowSteps(text);
    this.steps = resolved.steps;
    this.patOutsideEnv = [...resolved.patOutsideEnv, ...resolved.steps.map((step) => step.patOutsideEnv).filter(Boolean)];
    // Every run script, so a command in a YAML comment or a description never counts.
    this.runText = this.steps.map((step) => step.run).join('\n');
  }

  jobRunText(name) {
    return this.jobSteps(name).map((step) => step.run).join('\n');
  }

  stepIndex(job, predicate) {
    const step = this.jobSteps(job).find(predicate);
    return step === undefined ? -1 : step.index;
  }

  get jobs() {
    return mapping(this.data.jobs);
  }

  job(name) {
    return mapping(this.jobs[name]);
  }

  jobSteps(name) {
    return this.steps.filter((step) => step.job === name);
  }

  step(job, name) {
    return this.steps.find((step) => step.job === job && step.name === name);
  }

  // The job's source text, located through the YAML tree rather than indentation.
  jobSource(name) {
    const jobs = this.document?.get('jobs', true);
    if (!isMap(jobs)) return '';
    const pair = jobs.items.find((item) => isScalar(item.key) && item.key.value === name);
    if (!pair?.value?.range) return '';
    return this.text.slice(pair.key.range[0], pair.value.range[2]);
  }

  needs(name) {
    const needs = this.job(name).needs;
    if (typeof needs === 'string') return [needs];
    return Array.isArray(needs) ? needs.filter((need) => typeof need === 'string') : [];
  }

  transitiveNeeds(name, seen = new Set()) {
    for (const need of this.needs(name)) {
      if (!seen.has(need)) {
        seen.add(need);
        this.transitiveNeeds(need, seen);
      }
    }
    return seen;
  }
}

const deepEqual = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const occurrences = (text, needle) => text.split(needle).length - 1;
const normalizeSpace = (text) => (typeof text === 'string' ? text.replace(/\s+/g, ' ').trim() : '');
const quoteAll = (values) => values.map((value) => JSON.stringify(value)).join(', ');

function checker(errors) {
  return {
    require(condition, message) {
      if (!condition) errors.push(message);
    },
    includes(label, text, needles, purpose) {
      const missing = needles.filter((needle) => !text.includes(needle));
      if (missing.length > 0) errors.push(`${label} must ${purpose}; missing: ${quoteAll(missing)}`);
    },
    excludes(label, text, needles, purpose) {
      const found = needles.filter((needle) => text.includes(needle));
      if (found.length > 0) errors.push(`${label} must ${purpose}; found: ${quoteAll(found)}`);
    },
  };
}

function environmentValidationMissing(run, validator) {
  return [
    'gh api "repos/${BRIDGE_REPO}/environments/bridge-assets-publication"',
    'gh api "repos/${BRIDGE_REPO}/environments/bridge-assets-publication/deployment-branch-policies"',
    `python3 ${validator} validate-environment`,
    '--branch-policies-json',
  ].filter((needle) => !run.includes(needle));
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

// R1: every JS contract test in tests/js is run by `npm run check:js`.
export function checkJsContractTestsRegistered(packageJson, testFiles, errors) {
  let scripts = {};
  try {
    scripts = mapping(mapping(JSON.parse(packageJson)).scripts);
  } catch (error) {
    errors.push(`package.json is not valid JSON: ${error.message}`);
  }
  const commands = new Set();
  for (const step of String(scripts['check:js'] ?? '').split(' && ')) {
    const match = /^npm run (\S+)$/.exec(step);
    if (match && typeof scripts[match[1]] === 'string') commands.add(scripts[match[1]]);
  }
  if (testFiles.length === 0) errors.push('tests/js has no *_test.mjs contract tests');
  for (const file of testFiles) {
    if (!commands.has(`node tests/js/${file}`)) {
      errors.push(`npm run check:js must run tests/js/${file} through an npm script that is exactly "node tests/js/${file}"`);
    }
  }
}

// R4: the publication PAT is the only secret, referenced only by the
// publication and orchestration workflows. Every workflow file is scanned, so a
// new one cannot escape.
function checkSecretReferences(workflows, errors) {
  const reference = new RegExp(`^${PUBLICATION_PAT_REFERENCE.source}`);
  for (const workflow of workflows) {
    const allowed = workflow.path === WORKFLOWS.publish || workflow.path === WORKFLOWS.autoUpdate;
    for (const match of workflow.text.matchAll(new RegExp(SECRETS_CONTEXT.source, 'gi'))) {
      if (!allowed || !reference.test(workflow.text.slice(match.index))) {
        errors.push(
          `${workflow.path}:${lineNumberAt(workflow.text, match.index)} references the secrets context; only `
          + `${WORKFLOWS.publish} and ${WORKFLOWS.autoUpdate} may, and only as secrets.${PUBLICATION_PAT_NAME}`,
        );
      }
    }
  }
}

// Every workflow and job grants the token read access at most; write access
// would let a compromised step mutate the repository without the PAT.
const readOnlyPermissions = (value) => value === undefined || value === 'read-all' || (isMapping(value)
  && Object.values(value).every((level) => level === 'read' || level === 'none'));

function checkPermissions(workflows, errors) {
  for (const workflow of workflows) {
    if (!isMapping(workflow.data.permissions) || !readOnlyPermissions(workflow.data.permissions)) {
      errors.push(`${workflow.path} must declare read-only top-level permissions`);
    }
    for (const name of Object.keys(workflow.jobs)) {
      if (!readOnlyPermissions(workflow.job(name).permissions)) {
        errors.push(`${workflow.path} job ${name} must not grant write permissions`);
      }
    }
  }
}

// A step or job that may fail without failing the run would turn any guard
// into a warning. The one exception lets a failed ref mutation reach the live
// re-query that classifies it.
const ALLOWED_CONTINUE_ON_ERROR = [[WORKFLOWS.publish, 'publish-assets', 'mutate']];

function checkNoContinueOnError(workflows, errors) {
  for (const workflow of workflows) {
    for (const name of Object.keys(workflow.jobs)) {
      if (workflow.job(name)['continue-on-error'] !== undefined) {
        errors.push(`${workflow.path} job ${name} must not set continue-on-error`);
      }
    }
    for (const step of workflow.steps) {
      const allowed = ALLOWED_CONTINUE_ON_ERROR.some(([file, job, id]) => file === workflow.path && job === step.job && id === step.id);
      if (step.raw['continue-on-error'] !== undefined && !allowed) {
        errors.push(`${workflow.path} step ${step.job}/${step.name || step.index} must not set continue-on-error`);
      }
    }
  }
}

function checkCiRunsContracts({ ci, candidate, publish }, errors) {
  const check = checker(errors);
  // Row 41: CI, candidate and publish run the JS contract tests without npm
  // lifecycle scripts and prove the tracked generated bridge matches its source.
  for (const workflow of [ci, candidate, publish]) {
    check.includes(workflow.path, workflow.runText, [
      'npm ci --ignore-scripts',
      'npm run check:js',
      'git ls-files --error-unmatch js/llama_webgpu_bridge.js js/llama_webgpu_bridge_worker.js js/llama_webgpu_bridge.d.ts',
      'git diff --exit-code -- js/llama_webgpu_bridge.js js/llama_webgpu_bridge_worker.js js/llama_webgpu_bridge.d.ts',
    ], 'install dependencies without lifecycle scripts, run npm run check:js, and prove the tracked generated bridge outputs are current');
  }
  // R2 and row 105: CI runs every Python contract suite and this contract.
  check.includes(ci.path, ci.runText, [
    "python3 -m unittest discover -s scripts -p '*_test.py'",
    'node scripts/verify_ci_reliability.mjs',
  ], 'run every Python contract suite and the CI reliability contract');
  // R3: the privileged workflows run the release contract suites from their
  // own checkout.
  for (const workflow of [candidate, publish]) {
    check.includes(workflow.path, workflow.runText, [
      'python3 scripts/release_contract_test.py',
      'python3 scripts/generate_release_manifest_test.py',
      'python3 scripts/release_publication_state_test.py',
      'python3 scripts/release_qualification_test.py',
      'node scripts/verify_ci_reliability.mjs',
    ], 'run the release contract suites and the CI reliability contract');
  }
  // Rows 21, 22, 101, 103: CI runs the checksum-pinned browser smokes.
  check.require(
    (ci.runText.match(/python3 scripts\/grammar_browser_smoke\.py\b/g) ?? []).length === 2
      && ci.runText.includes('--model-sha256 "$LLAMA_WEBGPU_MULTIMODAL_MODEL_SHA256"'),
    `${ci.path} must run the grammar smoke twice, once with the checksum-pinned multimodal model`,
  );
  check.includes(ci.path, ci.runText, [
    'python3 scripts/next_token_scores_browser_smoke.py',
    'python3 scripts/state_persistence_browser_smoke.py',
    'python3 scripts/multimodal_browser_smoke.py',
  ], 'run the next-token scores, state persistence and multimodal browser smokes');
  // Row 57: CI builds the llama.cpp pin.
  check.includes(ci.path, ci.runText, ["tr -d '[:space:]' < llama_cpp.version"], 'resolve the llama.cpp tag from llama_cpp.version');
  // Row 61: ordinary CI never dispatches publication or holds actions: write.
  check.excludes(ci.path, ci.text, ['dispatch-publish-assets:', 'gh workflow run'], 'never dispatch publication');
  const permissions = [ci.data.permissions, ...Object.keys(ci.jobs).map((name) => ci.job(name).permissions)];
  check.require(
    permissions.every((value) => value !== 'write-all' && mapping(value).actions !== 'write'),
    `${ci.path} must never grant actions: write`,
  );
}

function checkToolchainPins({ ci, candidate, publish }, files, errors) {
  const check = checker(errors);
  // Row 42: the ordinary pin follows release_contract.parse_upstream_tag. One
  // trailing newline is stripped; no other CR or LF is allowed.
  const contents = files['llama_cpp.version'];
  const version = contents.endsWith('\n') ? contents.slice(0, -1) : contents;
  check.require(
    /^(?:v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)|b(?:0|[1-9][0-9]*))$/.test(version),
    'llama_cpp.version must contain one exact llama.cpp release tag: stable vMAJOR.MINOR.PATCH like v0.2.0 or development bNNNN like b9165',
  );
  // Row 43.
  check.require(
    /^[0-9]+\.[0-9]+\.[0-9]+$/.test(files['emsdk.version'].trim()),
    'emsdk.version must contain one exact Emscripten semantic version',
  );
  // Row 44: install exactly the pinned emsdk and verify the resolved emcc.
  for (const workflow of [ci, candidate]) {
    const emsdk = workflow.steps.filter((step) => /(?:^|\/)setup-emsdk@/.test(step.uses));
    check.require(
      emsdk.length > 0 && emsdk.every((step) => step.with.version === '${{ env.EMSCRIPTEN_VERSION }}'),
      `${workflow.path} must install the resolved emsdk.version (version: \${{ env.EMSCRIPTEN_VERSION }}) in every setup-emsdk step`,
    );
    check.includes(workflow.path, workflow.runText, [
      'scripts/verify_emscripten_version.py --print-pin',
      'scripts/verify_emscripten_version.py --emit-github-env "$GITHUB_ENV"',
    ], 'resolve emsdk.version and verify the resolved emcc version before building');
  }
  // Row 45: the verifier compares emcc with emsdk.version and gates direct builds.
  const verifier = files['scripts/verify_emscripten_version.py'];
  const build = files['scripts/build_bridge.sh'];
  check.require(
    verifier.includes('emsdk.version')
      && verifier.includes('["emcc", "--version"]')
      && verifier.includes('resolved != expected')
      && verifier.includes('EMSCRIPTEN_VERSION={resolved}')
      && build.includes('scripts/verify_emscripten_version.py')
      && build.indexOf('scripts/verify_emscripten_version.py') < build.indexOf('echo "[bridge] configuring with emcmake"'),
    'the Emscripten verifier must compare emcc against emsdk.version, export the resolved compiler identity, and gate direct builds',
  );
  // Row 46: the manifest records the verified compiler.
  check.require(
    candidate.runText.includes('--emscripten-version "${EMSCRIPTEN_VERSION}"')
      && publish.runText.includes('--emscripten-version "${EMSCRIPTEN_VERSION}"')
      && files['scripts/generate_release_manifest.py'].includes('"emscripten_version": args.emscripten_version'),
    'the asset manifest must record the runtime-verified Emscripten compiler version',
  );
}

function checkOrchestration(autoUpdate, orchestratorSources, errors) {
  const check = checker(errors);
  // Row 49: provenance by immutable asset id, one credential used only inside
  // the validated publication environment, dispatch only through the
  // orchestrator, and development scans never publish.
  check.includes(autoUpdate.path, autoUpdate.text, ['releases/assets/${asset_id}', "env.REQUESTED_CHANNEL != 'stable'"],
    'download provenance by unique release-asset id and keep development scans scan-only');
  check.excludes(autoUpdate.path, autoUpdate.text, [
    'gh release view', 'gh release download', 'ORCHESTRATOR_DISPATCH_TOKEN',
    'gh workflow run', 'create-pull-request', 'git push',
  ], 'fetch provenance only by asset id and dispatch only through stable_release_orchestrator.py');
  check.require(
    autoUpdate.patOutsideEnv.length === 0,
    `${autoUpdate.path} must reference the publication PAT only in a step env value; found in ${autoUpdate.patOutsideEnv.join(', ')}`,
  );
  const patSteps = autoUpdate.steps.filter((step) => Object.values(step.env).some(containsPat));
  check.require(
    patSteps.length > 0
      && patSteps.every((step) => step.env[PUBLICATION_PAT_NAME] === '${{ secrets.WEBGPU_BRIDGE_ASSETS_PAT }}'),
    `${autoUpdate.path} must bind the publication PAT only as ${PUBLICATION_PAT_NAME}: \${{ secrets.WEBGPU_BRIDGE_ASSETS_PAT }}`,
  );
  for (const step of patSteps) {
    const environment = autoUpdate.job(step.job).environment;
    const environmentName = typeof environment === 'string' ? environment : mapping(environment).name;
    const validated = autoUpdate.jobSteps(step.job).some((other) => other.index < step.index
      && other.env.GH_TOKEN === '${{ github.token }}'
      && environmentValidationMissing(other.run, 'scripts/release_contract.py').length === 0);
    check.require(
      environmentName === 'bridge-assets-publication' && validated,
      `${autoUpdate.path} job ${step.job} may use the PAT only inside the bridge-assets-publication environment, after a github.token step validates that environment policy with scripts/release_contract.py validate-environment`,
    );
  }
  // Rows 50-51: every job, and so every environment job, rejects a non-owner
  // manual caller and an unsuccessful, non-default-branch or non-owner
  // workflow_run before the environment exposes the credential.
  const ownerManualGate = "github.event_name == 'workflow_dispatch' && github.actor == github.repository_owner && github.triggering_actor == github.repository_owner";
  const ownerWorkflowRunGate = "github.event_name == 'workflow_run' && github.event.workflow_run.conclusion == 'success' && "
    + 'github.event.workflow_run.head_branch == github.event.repository.default_branch && '
    + 'github.event.workflow_run.actor.login == github.repository_owner && '
    + 'github.event.workflow_run.triggering_actor.login == github.repository_owner';
  const jobNames = Object.keys(autoUpdate.jobs);
  check.require(
    jobNames.length > 0 && jobNames.every((name) => {
      const condition = normalizeSpace(autoUpdate.job(name).if);
      return condition.includes(ownerManualGate) && condition.includes(ownerWorkflowRunGate);
    }),
    `${autoUpdate.path}: every job must gate manual runs and workflow_run continuations on the repository owner before the publication environment can expose its credential`,
  );
  const workflowRun = mapping(mapping(autoUpdate.data.on).workflow_run);
  check.require(
    sameList(sorted(Array.isArray(workflowRun.workflows) ? workflowRun.workflows : []),
      ['Build Exact Bridge Candidate', 'Publish Exact Qualified Bridge Assets', 'Qualify Exact Bridge Candidate'])
      && deepEqual(workflowRun.types, ['completed']),
    `${autoUpdate.path} must listen only to completed runs of the three exact stage workflows`,
  );
  const proof = autoUpdate.steps.find((step) => step.name === 'Prove the exact workflow continuation before environment use');
  check.require(
    proof !== undefined
      && autoUpdate.job(proof.job).environment === undefined
      && jobNames.every((name) => autoUpdate.job(name).environment === undefined || autoUpdate.transitiveNeeds(name).has(proof.job))
      && proof.run.includes('scripts/release_qualification.py verify-run')
      && proof.run.includes('--run-attempt 1')
      && /\.github\/workflows\/bridge_candidate\.yml\)\s+artifact_name=exact-webgpu-bridge-dist\b/.test(proof.run)
      && /\.github\/workflows\/bridge_qualification\.yml\)\s+artifact_name=qualification-attestation\b/.test(proof.run)
      && /\.github\/workflows\/publish_assets\.yml\)\s+artifact_name=bridge-qualification-outcome\b/.test(proof.run),
    `${autoUpdate.path} must prove a continuation's first attempt and exact stage artifact with release_qualification.py verify-run, in a job every environment job needs`,
  );
  // Row 53: the governance and approval booleans come only from live proofs.
  for (const [relativePath, source] of Object.entries(orchestratorSources)) {
    check.excludes(relativePath, source, ['"assets_immutable_releases_enabled": "true"', '"publish_approved": "true"'],
      'derive the governance and approval booleans only from live proofs, never a literal');
  }
}

function checkPublication({ publish, candidate }, errors) {
  const check = checker(errors);
  const { text, runText } = publish;
  const publishJob = publish.jobRunText('publish-assets');

  // Rows 62 and 64: dispatch-only, with an explicit approval input, and no
  // automatic trigger or reusable entry point.
  const on = mapping(publish.data.on);
  check.require(
    sameList(Object.keys(on), ['workflow_dispatch'])
      && has(mapping(mapping(on.workflow_dispatch).inputs), 'publish_approved'),
    `${publish.path} must be triggered only by workflow_dispatch (no push, schedule or workflow_call) and take a publish_approved input`,
  );
  const approval = publish.steps.find((step) => step.run.includes('if [ "${PUBLISH_APPROVED}" != "true" ]'));
  check.require(
    approval !== undefined
      && publish.transitiveNeeds('publish-assets').has(approval.job)
      && publish.transitiveNeeds('verify-publication-environment').has(approval.job)
      && approval.env.PUBLISH_APPROVED === '${{ inputs.publish_approved }}'
      && approval.run.includes('publish_approved=true is required before publication'),
    `${publish.path} must refuse to publish unless the publish_approved input is true`,
  );
  const assetsRepo = publish.steps.find((step) => step.run.includes('if [ "${REQUESTED_ASSETS_REPO}" != "${APPROVED_ASSETS_REPO}" ]'));
  check.require(
    assetsRepo !== undefined
      && assetsRepo.env.REQUESTED_ASSETS_REPO === '${{ inputs.assets_repo }}'
      && assetsRepo.env.APPROVED_ASSETS_REPO === 'leehack/llama-web-bridge-assets',
    `${publish.path} must refuse any assets_repo other than leehack/llama-web-bridge-assets`,
  );
  check.excludes(publish.path, text, ['llama_cpp.version'], 'never read the bridge llama.cpp pin');
  check.includes(publish.path, runText, [
    'if [ "${GITHUB_REPOSITORY}" != "${BRIDGE_REPO}" ]',
    'if [ "${GITHUB_REF}" != "refs/heads/${bridge_default}" ]',
  ], 'run only in the bridge repository, dispatched from its default branch');

  // Row 63: publication always checks out and verifies the owning repository.
  check.require(
    mapping(publish.data.env).BRIDGE_REPO === 'leehack/llama-web-bridge',
    `${publish.path} must pin BRIDGE_REPO to leehack/llama-web-bridge`,
  );
  const checkouts = publish.steps.filter((step) => /^actions\/checkout@/.test(step.uses));
  check.require(
    checkouts.length > 0 && checkouts.every((step) => step.with['persist-credentials'] === false && (
      step.with.repository === 'leehack/llama-web-bridge-assets'
      || (step.with.repository === 'leehack/llama-web-bridge'
        && ['${{ github.sha }}', '${{ env.BRIDGE_SOURCE_SHA }}'].includes(step.with.ref)))),
    `${publish.path}: every checkout must name leehack/llama-web-bridge at github.sha or BRIDGE_SOURCE_SHA (or the fixed assets repository) and not persist credentials`,
  );
  check.require(
    candidate.runText.includes('--bridge-repo "${BRIDGE_REPO}"')
      && !text.includes('repos/${GITHUB_REPOSITORY}')
      && !candidate.text.includes('repos/${GITHUB_REPOSITORY}'),
    'candidate and publication must verify the owning bridge repository, never the executing one',
  );

  // Row 64: the job token proves the environment policy after the approval
  // check; the privileged job reaches the environment only through that proof.
  const verifyJob = publish.job('verify-publication-environment');
  const verifyStep = publish.jobSteps('verify-publication-environment')
    .find((step) => step.run.includes('echo "environment_name=bridge-assets-publication" >> "${GITHUB_OUTPUT}"'));
  check.require(
    deepEqual(verifyJob.permissions, READ_ONLY_ACTIONS_PERMISSIONS)
      && !containsMatch(verifyJob, SECRETS_CONTEXT)
      && verifyStep !== undefined
      && verifyStep.env.GH_TOKEN === '${{ github.token }}'
      && environmentValidationMissing(verifyStep.run, 'scripts/release_contract.py').length === 0
      && verifyStep.id !== ''
      && mapping(verifyJob.outputs).environment_name === `\${{ steps.${verifyStep.id}.outputs.environment_name }}`
      && publish.transitiveNeeds('verify-publication-environment').has('validate-request'),
    `${publish.path} job verify-publication-environment must run after the approval check with read-only permissions and no secret, validate the environment and its deployment branch policies with github.token and scripts/release_contract.py validate-environment, and only then output the environment name`,
  );
  const environmentJobs = Object.keys(publish.jobs).filter((name) => publish.job(name).environment !== undefined);
  check.require(
    sameList(environmentJobs, ['publish-assets'])
      && deepEqual(publish.job('publish-assets').environment, { name: '${{ needs.verify-publication-environment.outputs.environment_name }}' })
      && publish.needs('publish-assets').includes('verify-publication-environment'),
    `${publish.path}: only publish-assets may enter an environment, and only by the name verify-publication-environment outputs`,
  );

  // Row 71: the privileged job revalidates the policy with the job token in the
  // step immediately before its first PAT-bearing step.
  const revalidate = publish.step('publish-assets', 'Revalidate publication environment policy before using PAT');
  const firstPat = publish.jobSteps('publish-assets').find((step) => containsPat(step.raw) || Object.values(step.env).some(containsPat));
  check.require(
    deepEqual(publish.job('publish-assets').permissions, READ_ONLY_ACTIONS_PERMISSIONS)
      && revalidate !== undefined
      && firstPat !== undefined
      && firstPat.index === revalidate.index + 1
      && revalidate.env.GH_TOKEN === '${{ github.token }}'
      && !containsMatch(revalidate.raw, SECRETS_CONTEXT)
      && environmentValidationMissing(revalidate.run, 'publication-policy/scripts/release_contract.py').length === 0,
    `${publish.path}: publish-assets must have read-only permissions and revalidate the environment policy with github.token and the trusted publication-policy validator in the step immediately before its first PAT-bearing step`,
  );

  // Row 72: the PAT is fail-closed, never printed, and used only by the
  // expected steps.
  errors.push(...validatePublicationPatContract(text, EXPECTED_PUBLICATION_PAT_STEPS));

  // Rows 73-75: exact provenance identities, fetched by immutable asset id,
  // with inputs validated before the first native network request.
  check.includes(publish.path, runText, [
    'scripts/release_contract.py validate-native-release',
    '"repos/${NATIVE_REPO}/releases/assets/${asset_id}"',
    '--checksums "${RUNNER_TEMP}/native-release/SHA256SUMS"',
    '--manifest-sha256 "${NATIVE_MANIFEST_SHA256}"',
    'git clone --depth 1 --branch "${UPSTREAM_TAG}"',
    'git clone --depth 1 --branch "${NATIVE_RELEASE_TAG}"',
    'checked-out bridge source does not match the requested full SHA',
    'git ls-remote https://github.com/ggml-org/llama.cpp.git',
    '"refs/tags/${UPSTREAM_TAG}" "refs/tags/${UPSTREAM_TAG}^{}"',
    'resolved upstream tag commit does not match upstream_commit',
    'git ls-remote "https://github.com/${NATIVE_REPO}.git"',
    '"refs/tags/${NATIVE_RELEASE_TAG}" "refs/tags/${NATIVE_RELEASE_TAG}^{}"',
    'scripts/release_contract.py resolve-tag-commit',
    '--native-tag-commit "${NATIVE_TAG_COMMIT}"',
  ], 'verify the exact bridge, upstream tag/commit, native tag/commit and native manifest checksum identities');
  check.require(
    occurrences(runText, 'python3 scripts/release_contract.py resolve-tag-commit') >= 2
      && publish.steps.some((step) => step.run.includes('--native-tag-commit "${NATIVE_TAG_COMMIT}"')
        && step.env.NATIVE_TAG_COMMIT === '${{ steps.native_tag.outputs.native_tag_commit }}'),
    `${publish.path} must resolve the upstream and native tag commits itself and check the manifest against the resolved native tag commit (steps.native_tag)`,
  );
  const nativeRequest = runText.indexOf('scripts/release_contract.py validate-native-request');
  const upstreamLookup = runText.indexOf('git ls-remote https://github.com/ggml-org/llama.cpp.git');
  const nativeLookup = runText.indexOf('git ls-remote "https://github.com/${NATIVE_REPO}.git"');
  const requestSpan = runText.slice(nativeRequest, nativeLookup);
  check.require(
    nativeRequest >= 0 && nativeRequest < upstreamLookup && upstreamLookup < nativeLookup
      && requestSpan.includes('--native-release-tag "${NATIVE_RELEASE_TAG}"')
      && requestSpan.includes('--upstream-commit "${UPSTREAM_COMMIT}"')
      && requestSpan.includes('--manifest-sha256 "${NATIVE_MANIFEST_SHA256}"'),
    `${publish.path} must validate native tag, upstream commit, and manifest SHA-256 inputs before the first native network request`,
  );
  // Row 76: preflight and recovery resolve existing asset tags strictly.
  for (const name of ['Verify source ancestry and classify exact remote state', 'Re-fetch and finish or safely classify partial publication']) {
    const step = publish.step('publish-assets', name);
    check.require(
      step !== undefined
        && step.run.includes('publication-policy/scripts/release_contract.py resolve-tag-commit')
        && step.run.includes('fetched asset release tag changed after immutable resolution'),
      `${publish.path} step ${name} must resolve the existing asset tag with the strict trusted parser and verify the fetched ref did not change`,
    );
  }
  check.excludes(publish.path, text, ["awk '$2 ~ /\\^\\{\\}$/ {print $1}'"], 'never fall back to an awk tag parser');
  // Row 77: publishes are serialized.
  const concurrency = mapping(publish.data.concurrency);
  check.require(
    concurrency.group === 'publish-bridge-assets-leehack-llama-web-bridge-assets' && concurrency['cancel-in-progress'] === false,
    `${publish.path} must serialize asset publishes (group publish-bridge-assets-leehack-llama-web-bridge-assets, cancel-in-progress: false)`,
  );
  // Row 78: fail-closed classification, honest gates, checksum verification
  // and an atomic push.
  check.includes(publish.path, runText, [
    'scripts/release_publication_state.py classify',
    'recoverable-partial',
    'publication_outcome',
    'cp "${RUNNER_TEMP}/preflight.json" "${RUNNER_TEMP}/publication-outcome.json"',
    'state_changed()',
    'mutation-unknown',
    'ref-requery-failed',
    'release-requery-failed',
    'classify_remote() (',
    '.reason_code != "invalid-input-or-state"',
    '.orchestrator_correlation_id == env.ORCHESTRATOR_CORRELATION_ID',
    '.candidate_fingerprint == env.CANDIDATE_FINGERPRINT',
    'sha256sum --check sha256sums.txt',
    'git -C assets-repo push --atomic',
  ], 'classify exact retry states, reject failed classifier re-queries, emit outcomes, and recheck after mutation');
  check.require(
    occurrences(runText, '.qualification_gates == {state_persistence:"passed",multimodal:"passed",') >= 3
      && occurrences(runText, 'speech_to_text:"required-automated-qualification",') >= 3
      && !text.includes('speech_to_text:"passed"')
      && !text.includes('text_to_speech:"passed"'),
    `${publish.path} must compare every re-queried manifest with the honest candidate gates, never a candidate-run speech or TTS pass`,
  );
  // Row 79: immutable-release governance is proven before any ref mutation.
  const firstRefMutation = publish.stepIndex('publish-assets', (step) => step.name === 'Apply only the classified ref mutation');
  const governanceCheck = publish.stepIndex('publish-assets',
    (step) => /release_contract\.py\s+(?:\\\s+)?validate-immutable-release-governance/.test(step.run));
  check.require(
    occurrences(publishJob, '"repos/${ASSETS_REPO}/immutable-releases"') >= 3
      && publishJob.includes('--repository "${ASSETS_REPO}"')
      && governanceCheck >= 0 && firstRefMutation >= 0 && governanceCheck < firstRefMutation
      && publishJob.includes('immutable-governance-unverified')
      && mapping(publish.data.env).CANDIDATE_PREQUALIFICATION_ARTIFACT_NAME === 'bridge-candidate-prequalification'
      && runText.includes('CANDIDATE_PREQUALIFICATION_ARTIFACT_ID')
      && runText.includes('candidate-prequalification.zip')
      && runText.includes('validate-candidate-prequalification'),
    `${publish.path} must prove immutable-release governance through the repository API before the first ref mutation and bind the candidate's prequalification record`,
  );
  const governanceAssertion = candidate.steps.find((step) => step.run.includes('"${ASSETS_IMMUTABLE_RELEASES_ENABLED}" != "true"'));
  check.require(
    governanceAssertion !== undefined
      && governanceAssertion.env.ASSETS_IMMUTABLE_RELEASES_ENABLED === '${{ inputs.assets_immutable_releases_enabled }}'
      && candidate.runText.includes('--argjson immutable "${ASSETS_IMMUTABLE_RELEASES_ENABLED}"')
      && candidate.runText.includes('assets_immutable_releases_enabled:$immutable'),
    `${candidate.path} must refuse to build unless the dispatcher asserts immutable releases are enabled, and record that assertion`,
  );
  // Row 80: the published release is read back immutable and attested.
  check.includes(`${publish.path} (publish-assets)`, publishJob, [
    '"repos/${ASSETS_REPO}/releases/tags/${RELEASE_TAG}"',
    '"repos/${ASSETS_REPO}/releases/${published_release_id}"',
    'gh release verify "${RELEASE_TAG}" --repo "${ASSETS_REPO}"',
    '--format json',
    'verify-immutable-publication',
    '--release-by-id-json "${RUNNER_TEMP}/published-release-by-id.json"',
    'immutable-publication-unverified',
    'touch "${RUNNER_TEMP}/ref-push-attempted"',
    'touch "${RUNNER_TEMP}/release-mutation-attempted"',
    'fallback_state="mutation-unknown"',
    'fallback_mutated=null',
  ], "read the new release back by tag and by id, require immutable: true, and verify GitHub's signed release attestation over the exact assets");
  // A failed mutation step (it may continue on error) must still reach the
  // live re-query that decides the outcome.
  const requery = publish.jobSteps('publish-assets').find((step) => step.index > firstRefMutation
    && normalizeSpace(String(step.raw.if ?? '')).includes("steps.mutate.outcome == 'failure'"));
  check.require(
    firstRefMutation >= 0 && requery !== undefined && requery.run.includes('scripts/release_publication_state.py classify'),
    `${publish.path} must re-query the remote state after a failed ref mutation`,
  );
  const formatProbe = publish.stepIndex('publish-assets', (step) => step.run.includes("gh release verify --help | grep -F -- '--format'"));
  const noopBranch = publishJob.indexOf('elif [ "${post_action}" = "none" ]');
  const finalCheck = publishJob.indexOf('Every complete state is verified');
  check.require(
    formatProbe >= 0 && firstRefMutation >= 0 && formatProbe < firstRefMutation
      && noopBranch >= 0 && finalCheck >= 0 && noopBranch < finalCheck,
    `${publish.path} must probe gh release verify --format support before the first ref mutation and verify every complete state, including the no-op path`,
  );
  // Row 81: an immutable release is never repaired.
  check.excludes(publish.path, text, [
    'gh release delete', 'gh release edit', 'gh release upload', 'upload-release-assets',
    'git push --delete', '--force-with-lease', 'push --force', '-X DELETE',
  ], 'never delete, retag, overwrite or repair a release');
}

function checkCandidateAndQualification({ candidate, qualification, publish }, errors) {
  const check = checker(errors);
  // Row 84: the candidate is built once, on the first attempt, unprivileged.
  const attempt = candidate.steps.find((step) => step.run.includes('"${GITHUB_RUN_ATTEMPT_STRING}" != "1"'));
  check.require(
    attempt !== undefined && attempt.env.GITHUB_RUN_ATTEMPT_STRING === '${{ github.run_attempt }}',
    `${candidate.path} must refuse any run attempt other than 1`,
  );
  check.require(
    (candidate.runText.match(/\.\/scripts\/build_bridge\.sh\b/g) ?? []).length === 1,
    `${candidate.path} must build the candidate exactly once`,
  );
  check.require(
    !candidate.text.includes(PUBLICATION_PAT_NAME)
      && Object.keys(candidate.jobs).every((name) => candidate.job(name).environment === undefined),
    `${candidate.path} must stay unprivileged: no publication PAT and no environment`,
  );
  const build = candidate.steps.find((step) => /\.\/scripts\/build_bridge\.sh\b/.test(step.run));
  const stateGate = candidate.steps.find((step) => step.id === 'state_gate');
  const multimodalGate = candidate.steps.find((step) => step.id === 'multimodal_gate');
  check.require(
    build !== undefined && String(build.env.WEBGPU_BRIDGE_BUILD_MEM64) === '1'
      && stateGate?.run.includes('python3 scripts/state_persistence_browser_smoke.py')
      && multimodalGate?.run.includes('python3 scripts/multimodal_browser_smoke.py')
      && candidate.steps.some((step) => step.env.STATE_CONCLUSION === '${{ steps.state_gate.outcome }}'
        && step.env.MULTIMODAL_CONCLUSION === '${{ steps.multimodal_gate.outcome }}'),
    `${candidate.path} must build the memory64 core and record the outcomes of its state persistence and multimodal gates`,
  );
  const uploads = candidate.steps.filter((step) => /^actions\/upload-artifact@/.test(step.uses)).map((step) => step.with.name);
  check.require(
    uploads.includes('exact-webgpu-bridge-dist') && uploads.includes('bridge-candidate-prequalification')
      && candidate.text.includes('pending-automated-qualification'),
    `${candidate.path} must upload exact-webgpu-bridge-dist and the bridge-candidate-prequalification record, with the heavy gates pending automated qualification`,
  );
  // Row 85: every stage is dispatched by the repository owner.
  for (const workflow of [candidate, qualification, publish]) {
    const actor = workflow.steps.find((step) => step.run.includes('"${DISPATCHING_ACTOR}" != "${REPOSITORY_OWNER}"'));
    check.require(
      actor !== undefined
        && actor.env.DISPATCHING_ACTOR === '${{ github.triggering_actor }}'
        && actor.env.REPOSITORY_OWNER === '${{ github.repository_owner }}',
      `${workflow.path} must refuse any dispatching actor other than the repository owner`,
    );
  }
  // Row 86: qualification and publication pin the run actor and first attempt.
  for (const workflow of [qualification, publish]) {
    check.includes(workflow.path, workflow.runText, [
      '"${GITHUB_ACTOR}" != "${REPOSITORY_OWNER}"',
      '"${GITHUB_RUN_ATTEMPT}" != "1"',
    ], 'refuse a non-owner run actor and any run attempt other than 1');
  }
  // Rows 87, 88, 90: publication verifies the attestation against the exact
  // candidate artifact id and first attempts in both of its jobs, with the
  // trusted validators, and never rebuilds.
  const bindingArgs = [
    '--candidate-artifact-id "${CANDIDATE_ARTIFACT_ID}"',
    '--candidate-run-attempt 1',
    '--qualification-run-id "${QUALIFICATION_RUN_ID}"',
    '--qualification-run-attempt 1',
    '--qualification-source-sha "${QUALIFICATION_SOURCE_SHA}"',
  ];
  for (const job of ['verify-candidate-and-qualification', 'publish-assets']) {
    const verification = publish.jobSteps(job).find((step) => step.run.includes('scripts/release_qualification.py verify-attestation'));
    check.require(
      verification !== undefined && bindingArgs.every((arg) => verification.run.includes(arg)),
      `${publish.path} job ${job} must verify the attestation against the exact candidate artifact id, both first attempts, and the qualification run and source`,
    );
  }
  check.require(
    mapping(publish.job('verify-candidate-and-qualification').outputs).candidate_artifact_id === '${{ steps.runs.outputs.candidate_artifact_id }}'
      && publish.jobSteps('publish-assets').some((step) => step.run.includes('verify-attestation')
        && step.env.CANDIDATE_ARTIFACT_ID === '${{ needs.verify-candidate-and-qualification.outputs.candidate_artifact_id }}')
      && Object.keys(publish.jobs).every((name) => !has(mapping(publish.job(name).outputs), 'qualification_artifact_id')),
    `${publish.path} must carry the proven candidate artifact id into the privileged job and never export a qualification artifact id`,
  );
  check.includes(qualification.path, qualification.runText, [
    '--candidate-artifact-id "${CANDIDATE_ARTIFACT_ID}"',
    '--candidate-run-attempt 1',
    '--qualification-run-id "${GITHUB_RUN_ID}"',
    '--qualification-run-attempt 1',
    '--qualification-source-sha "${GITHUB_SHA}"',
    'artifact_type="candidate"',
  ], 'bind the attestation to the exact candidate artifact id, both first attempts, and its own run and source');
  check.includes(publish.path, publish.runText, [
    'artifact_type="candidate"',
    'artifact_type="attestation"',
    'scripts/release_qualification.py verify-run',
    '--workflow-path "${workflow_path}"',
    '--head-branch "${bridge_default}"',
    'repos/${BRIDGE_REPO}/compare/${head_sha}...${bridge_default}',
    'actions/artifacts/${CANDIDATE_ARTIFACT_ID}/zip',
    'actions/artifacts/${QUALIFICATION_ARTIFACT_ID}/zip',
    '?per_page=100',
    'run_attempt_args=(--run-attempt 1)',
    '_extract_flat_artifact_archive',
    'publication-policy/scripts/release_qualification.py verify-attestation',
    'publication-policy/scripts/release_publication_state.py classify',
    '--harness-dir',
  ], 'prove both run identities on the default-branch line and download both artifacts by immutable id with the trusted validators');
  const publishEnv = mapping(publish.data.env);
  check.require(
    publishEnv.CANDIDATE_WORKFLOW_PATH === '.github/workflows/bridge_candidate.yml'
      && publishEnv.QUALIFICATION_WORKFLOW_PATH === '.github/workflows/bridge_qualification.yml'
      && ['exact-webgpu-bridge-dist', 'verified-qualification-attestation', 'bridge-qualification-outcome']
        .every((name) => publish.text.includes(name)),
    `${publish.path} must pin the candidate and qualification workflow paths and the exact artifact names`,
  );
  check.excludes(publish.path, publish.text, [
    'scripts/build_bridge.sh', 'WEBGPU_BRIDGE_BUILD_MEM64', 'setup-emsdk', 'scripts/generate_release_manifest.py',
    'bridge-source/scripts/release_qualification.py', 'bridge-source/scripts/release_publication_state.py',
  ], 'never rebuild the candidate or run a validator from the historical build source');
  // Row 91: qualification accepts no hand-produced attestation, proves the
  // exact candidate before running the gates on its harness, and holds no PAT.
  check.excludes(qualification.path, qualification.text, [
    'attestation_base64:', 'attestation_json:', 'decode-attestation', PUBLICATION_PAT_NAME,
  ], 'never accept a transported attestation or hold the publication PAT');
  check.includes(qualification.path, qualification.runText, [
    'scripts/release_qualification.py verify-run',
    'release_qualification.py qualify',
    'scripts/release_qualification.py verify-attestation',
    '--harness-dir candidate-source/scripts',
    'candidate_correlation_id="$(jq -er',
    '"${candidate_correlation_id}" != "${ORCHESTRATOR_CORRELATION_ID}"',
    '?per_page=100',
    '--run-attempt 1',
    'actions/artifacts/${CANDIDATE_ARTIFACT_ID}/zip',
    'repos/${BRIDGE_REPO}/compare/${head_sha}...${bridge_default}',
  ], 'prove the candidate run, first attempt, artifact id and correlation before running the gates on its exact harness');
}

function checkModelPins(files, workflows, errors) {
  // Row 119: pins live only where the pin checks compare them.
  for (const relativePath of ['README.md', 'AGENTS.md']) {
    if (files[relativePath].search(SHA256_HEX) >= 0) {
      errors.push(`${relativePath} must hold no SHA-256 pins; CONTRIBUTING.md owns the smoke invocations and their pins`);
    }
  }
  // Rows 120-129.
  const contents = {
    'CONTRIBUTING.md': files['CONTRIBUTING.md'],
    [WORKFLOWS.ci]: workflows.ci.text,
    [WORKFLOWS.candidate]: workflows.candidate.text,
    [WORKFLOWS.qualification]: workflows.qualification.text,
  };
  const pinsByFile = Object.fromEntries(
    MODEL_SHA_PIN_FILES.map((relativePath) => [relativePath, extractModelShaPins(relativePath, contents[relativePath], errors)]),
  );
  requireIdenticalModelShaPins(pinsByFile, errors);
  const roles = Object.fromEntries(
    MODEL_PIN_ROLE_FILES.map((relativePath) => [relativePath, extractModelShaPinRoles(relativePath, contents[relativePath], errors)]),
  );
  const urls = Object.fromEntries(
    MODEL_PIN_ROLE_FILES.map((relativePath) => [relativePath, extractModelUrls(relativePath, contents[relativePath], errors)]),
  );
  for (const relativePath of WORKFLOW_MODEL_SHA_PIN_FILES) {
    requireCompleteWorkflowModelShaPinRoles(relativePath, roles[relativePath], errors);
  }
  requireQualificationModelShaPinRoles(roles[QUALIFICATION_MODEL_PIN_FILE], errors);
  requireIdenticalWorkflowModelShaPinRoles(
    Object.fromEntries(WORKFLOW_MODEL_SHA_PIN_FILES.map((relativePath) => [relativePath, roles[relativePath]])),
    errors,
  );
  requireCanonicalModelShaPins(roles, errors);
  requirePairedModelUrlsAndPins(urls, roles, errors);
  requireIdenticalModelUrls(urls, errors);
  requirePinnedModelUrlRevisions(urls, errors);
  requireMarkdownModelPinRoles('CONTRIBUTING.md', contents['CONTRIBUTING.md'], modelFileNameRoles(urls, errors), errors);
}

function readRequired(relativePath, errors) {
  try {
    return fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
  } catch (error) {
    errors.push(`required file is not readable: ${relativePath}: ${error.message}`);
    return '';
  }
}

function listRequired(relativeDirectory, pattern, errors) {
  try {
    return fs.readdirSync(path.join(ROOT, relativeDirectory)).filter((name) => pattern.test(name)).sort();
  } catch (error) {
    errors.push(`required directory is not readable: ${relativeDirectory}: ${error.message}`);
    return [];
  }
}

export function collectErrors() {
  const errors = [];
  if (expectedModelPinsError) errors.push(expectedModelPinsError);
  const workflows = Object.fromEntries(Object.entries(WORKFLOWS).map(
    ([key, relativePath]) => [key, new Workflow(relativePath, readRequired(relativePath, errors), errors)],
  ));
  const files = Object.fromEntries([
    'package.json', 'llama_cpp.version', 'emsdk.version', 'README.md', 'AGENTS.md', 'CONTRIBUTING.md',
    'scripts/verify_emscripten_version.py', 'scripts/build_bridge.sh', 'scripts/generate_release_manifest.py',
  ].map((relativePath) => [relativePath, readRequired(relativePath, errors)]));
  const orchestratorSources = Object.fromEntries([
    'stable_release_orchestrator.py',
    ...listRequired('scripts', /^release_orchestrator_.*\.py$/, errors).filter((name) => !name.endsWith('_test.py')),
  ].map((name) => [`scripts/${name}`, readRequired(`scripts/${name}`, errors)]));

  const known = new Set(Object.values(WORKFLOWS));
  const allWorkflows = [
    ...Object.values(workflows),
    ...listRequired('.github/workflows', /\.ya?ml$/, errors)
      .map((name) => `.github/workflows/${name}`)
      .filter((relativePath) => !known.has(relativePath))
      .map((relativePath) => new Workflow(relativePath, readRequired(relativePath, errors), errors)),
  ];

  checkJsContractTestsRegistered(files['package.json'], listRequired('tests/js', /_test\.mjs$/, errors), errors);
  checkSecretReferences(allWorkflows, errors);
  checkPermissions(allWorkflows, errors);
  checkNoContinueOnError(allWorkflows, errors);
  checkCiRunsContracts(workflows, errors);
  checkToolchainPins(workflows, files, errors);
  checkOrchestration(workflows.autoUpdate, orchestratorSources, errors);
  checkPublication(workflows, errors);
  checkCandidateAndQualification(workflows, errors);
  checkModelPins(files, workflows, errors);
  return errors;
}

function main() {
  const errors = collectErrors();
  if (errors.length > 0) {
    console.error('CI reliability contract failed:');
    for (const error of errors) console.error(`- ${error}`);
    return 1;
  }
  console.log('CI reliability contract passed');
  return 0;
}

function invokedAsEntry() {
  try {
    return Boolean(process.argv[1])
      && fs.realpathSync(process.argv[1]) === fs.realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (invokedAsEntry()) {
  process.exitCode = main();
}
