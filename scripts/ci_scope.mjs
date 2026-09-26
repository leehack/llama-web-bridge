#!/usr/bin/env node
// Conservative CI build selection; release and candidate workflows are
// unchanged. Ported from scripts/ci_scope.py with identical classification,
// stdout and GITHUB_OUTPUT.
//
//   node scripts/ci_scope.mjs                   BASE_SHA -> native=true|false
//   node scripts/ci_scope.mjs --check-results   NEEDS_JSON -> exit 0|1

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

// Only paths with no compiled-runtime dependency may avoid native jobs.
export const TOOLING = new Set([
  'scripts/release_publication_state.py', 'scripts/release_publication_state_test.py',
  'scripts/stable_release_orchestrator.py', 'scripts/stable_release_orchestrator_test.py',
  'scripts/release_orchestrator_asset_releases.py', 'scripts/release_orchestrator_asset_releases_test.py',
  'scripts/release_orchestrator_driver.py', 'scripts/release_orchestrator_driver_test.py',
  'scripts/release_orchestrator_driver_backlog_test.py',
  'scripts/release_orchestrator_driver_identical_release_test.py',
  'scripts/release_orchestrator_driver_publication_test.py',
  'scripts/release_orchestrator_fixtures_test.py',
  'scripts/release_orchestrator_model.py',
  'scripts/release_orchestrator_native.py', 'scripts/release_orchestrator_native_test.py',
  'scripts/release_orchestrator_planner.py', 'scripts/release_orchestrator_planner_test.py',
  'scripts/release_orchestrator_release_tags.py', 'scripts/release_orchestrator_release_tags_test.py',
  'scripts/release_orchestrator_run_names.py', 'scripts/release_orchestrator_run_names_test.py',
  'scripts/release_orchestrator_stage_proofs.py',
  'scripts/release_orchestrator_transport.py', 'scripts/release_orchestrator_transport_test.py',
  'scripts/release_orchestrator_workflow_runs.py', 'scripts/release_orchestrator_workflow_runs_test.py',
  'scripts/release_qualification_test.py', 'scripts/release_contract_test.py',
  'scripts/generate_release_manifest_test.py',
  'tests/js/native_core_source.mjs',
  'tests/js/bridge_js_source.mjs', 'tests/js/state_persistence_api_contract_test.mjs',
  'tests/js/text_to_speech_api_contract_test.mjs', 'tests/js/decision_api_contract_test.mjs',
  'tests/js/decision_bridge_contract_test.mjs',
  'scripts/verify_ci_reliability.mjs', 'tests/js/verify_ci_reliability_test.mjs',
  'tests/js/mtmd_compat_contract_test.mjs',
  'tests/js/wasm64_runtime_patch_contract_test.mjs',
  'tests/js/embedding_json_contract_test.mjs', 'tests/js/declared_class_fields_test.mjs', 'tests/js/native_load_option_arity_test.mjs',
  'tests/js/model_reload_contract_test.mjs',
  'tests/js/bridge_operation_queue_test.mjs',
  'tests/js/bridge_operation_lifecycle_test.mjs', 'tests/js/text_to_speech_recovery_test.mjs',
  'tests/js/bridge_type_declaration_contract_test.mjs', 'tests/js/worker_runtime_state_test.mjs',
  'tests/js/worker_token_coalescing_test.mjs', 'tests/js/workflow_input_transport_test.mjs',
]);

const DOCUMENTATION = new Set(['README.md', 'CONTRIBUTING.md', 'LICENSE']);

export function nativeRequired(paths) {
  return paths.length === 0 || paths.some((path) => !TOOLING.has(path) && !(
    DOCUMENTATION.has(path) || (path.startsWith('docs/') && path.endsWith('.md'))
  ));
}

// UTF-8 decoding with Python's errors="surrogateescape": each byte of a
// maximal ill-formed subsequence becomes U+DC80-U+DCFF, so a non-UTF-8 name
// is reported exactly as ci_scope.py reported it.
function decodeUtf8SurrogateEscape(bytes) {
  try {
    return new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes);
  } catch {
    // Fall through to the escaping decoder below.
  }
  let text = '';
  let start = 0;
  let needed = 0;
  let codePoint = 0;
  let lower = 0x80;
  let upper = 0xbf;
  const escape = (end) => {
    for (let i = start; i < end; i += 1) text += String.fromCharCode(0xdc00 + bytes[i]);
  };
  for (let i = 0; i < bytes.length; i += 1) {
    const byte = bytes[i];
    if (needed === 0) {
      start = i;
      if (byte <= 0x7f) {
        text += String.fromCharCode(byte);
      } else if (byte >= 0xc2 && byte <= 0xdf) {
        needed = 1;
        codePoint = byte & 0x1f;
      } else if (byte >= 0xe0 && byte <= 0xef) {
        if (byte === 0xe0) lower = 0xa0;
        if (byte === 0xed) upper = 0x9f;
        needed = 2;
        codePoint = byte & 0x0f;
      } else if (byte >= 0xf0 && byte <= 0xf4) {
        if (byte === 0xf0) lower = 0x90;
        if (byte === 0xf4) upper = 0x8f;
        needed = 3;
        codePoint = byte & 0x07;
      } else {
        escape(i + 1);
      }
      continue;
    }
    if (byte < lower || byte > upper) {
      // Escape the incomplete sequence and decode this byte afresh.
      escape(i);
      needed = 0;
      lower = 0x80;
      upper = 0xbf;
      i -= 1;
      continue;
    }
    lower = 0x80;
    upper = 0xbf;
    codePoint = (codePoint << 6) | (byte & 0x3f);
    needed -= 1;
    if (needed === 0) text += String.fromCodePoint(codePoint);
  }
  if (needed !== 0) escape(bytes.length);
  return text;
}

export function changedPaths(base, head = 'HEAD') {
  // --no-renames reports both old deletion and new addition, including moves
  // from native inputs into docs/tooling. NUL records preserve arbitrary names.
  if (!base || /^0+$/.test(base)) return [];
  const output = execFileSync('git', ['diff', '--no-renames', '--name-only', '-z', base, head, '--'], {
    stdio: ['inherit', 'pipe', 'inherit'],
    maxBuffer: Infinity,
  });
  return output.length > 0 ? decodeUtf8SurrogateEscape(output).replace(/\0+$/, '').split('\0') : [];
}

const isObject = (value) => typeof value === 'object' && value !== null && !Array.isArray(value);
const get = (value, key) => (isObject(value) && Object.hasOwn(value, key) ? value[key] : undefined);

export function validateResults(needs) {
  if (!isObject(needs)) return false;
  const scope = get(needs, 'changes') ?? {};
  if (get(scope, 'result') !== 'success') return false;
  const selected = get(get(scope, 'outputs') ?? {}, 'native');
  if (selected !== 'true' && selected !== 'false') return false;
  const expected = selected === 'true' ? 'success' : 'skipped';
  const jobs = ['build-webgpu-bridge'];
  if (get(get(needs, 'checks') ?? {}, 'result') !== 'success') return false;
  const names = new Set([...jobs, 'changes', 'checks']);
  const keys = Object.keys(needs);
  return keys.length === names.size && keys.every((key) => names.has(key))
    && jobs.every((job) => get(needs[job], 'result') === expected);
}

// json.dumps(value) with Python's default separators and ensure_ascii.
function pythonJsonDumps(value) {
  const string = (text) => JSON.stringify(text)
    .replace(/[\u007f-\uffff]/g, (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`);
  if (typeof value === 'string') return string(value);
  if (Array.isArray(value)) return `[${value.map(pythonJsonDumps).join(', ')}]`;
  return `{${Object.entries(value).map(([key, item]) => `${string(key)}: ${pythonJsonDumps(item)}`).join(', ')}}`;
}

const USAGE = 'usage: ci_scope.mjs [-h] [--check-results]';

export function main(argv = process.argv.slice(2)) {
  let checkResults = false;
  for (const arg of argv) {
    if (arg === '-h' || arg === '--help') {
      console.log(`${USAGE}\n\noptions:\n  -h, --help       show this help message and exit\n  --check-results`);
      return 0;
    }
    if (arg === '--check-results') {
      checkResults = true;
    } else {
      console.error(`${USAGE}\nci_scope.mjs: error: unrecognized arguments: ${arg}`);
      return 2;
    }
  }
  try {
    if (checkResults) {
      const needs = process.env.NEEDS_JSON;
      if (needs === undefined) throw new Error('NEEDS_JSON is not set');
      return validateResults(JSON.parse(needs)) ? 0 : 1;
    }
    // An unavailable diff fails the classifier, never silently skips work.
    const paths = changedPaths(process.env.BASE_SHA ?? '');
    const value = String(nativeRequired(paths));
    console.log(pythonJsonDumps({ native: value, paths }));
    const output = process.env.GITHUB_OUTPUT;
    if (output === undefined) throw new Error('GITHUB_OUTPUT is not set');
    fs.appendFileSync(output, `native=${value}\n`);
    return 0;
  } catch (error) {
    console.error(`error: ${error.message}`);
    return 1;
  }
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
