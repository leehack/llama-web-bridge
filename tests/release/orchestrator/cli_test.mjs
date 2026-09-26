// Tests of scripts/release/orchestrator/cli.mjs, the CLI entry: governed-path
// classification, bridge source identity, and caller authorization. One test
// per test method of scripts/stable_release_orchestrator_test.py, with the
// same names and assertions, plus Node-only checks: the classifier sets equal
// the Python sets they copy (the source of truth until the workflow switches
// to this entry), and the command-line surface.

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

import { ArgparseExit } from '../../../scripts/release/cli.mjs';
import { ContractError } from '../../../scripts/release/contract.mjs';
import { pyJsonLoads } from '../../../scripts/release/json.mjs';
import * as sro from '../../../scripts/release/orchestrator/cli.mjs';
import { OWNER } from './fixtures.mjs';

const ROOT = path.resolve(import.meta.dirname, '..', '..', '..');

// --- BridgeSourceIdentityTest ------------------------------------------------------

function withRepository(fn) {
  return () => {
    const temp = fs.mkdtempSync(path.join(tmpdir(), 'sro-cli-'));
    const repository = path.join(temp, 'bridge');
    fs.mkdirSync(repository);
    const git = (...args) => execFileSync('git', ['-C', repository, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
    git('init', '--initial-branch=main');
    git('config', 'user.name', 'Bridge Test');
    git('config', 'user.email', 'bridge-test@example.com');
    const commit = (message) => {
      git('add', '.');
      git('commit', '-m', message);
      return git('rev-parse', 'HEAD');
    };
    try {
      fn({ repository, git, commit });
    } finally {
      fs.rmSync(temp, { recursive: true, force: true });
    }
  };
}

function write(file, text) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, text, 'utf8');
}

test('test_actual_resolver_separates_orchestration_from_governed_changes', withRepository(({ repository, commit }) => {
  write(path.join(repository, 'src', 'bridge.cpp'), 'runtime-v1\n');
  const governed = commit('initial runtime');

  write(path.join(repository, '.github', 'workflows', 'auto_llama_cpp_update.yml'), 'name: scan\n');
  write(path.join(repository, 'docs', 'release.md'), 'release docs\n');
  write(path.join(repository, 'scripts', 'resolver_test.py'), '# regression\n');
  write(path.join(repository, 'scripts', 'ci_scope.py'), '# selector\n');
  const orchestrationHead = commit('workflow tests and docs');

  const output = path.join(repository, 'identity.json');
  assert.equal(sro.main(
    ['resolve-bridge-source', '--repository', repository, '--head', 'HEAD', '--output-json', output],
    { stdout: () => {}, stderr: () => {} },
  ), 0);
  const identity = pyJsonLoads(fs.readFileSync(output, 'utf8'));
  assert.equal(identity.bridge_source_sha, orchestrationHead);
  assert.equal(identity.bridge_build_sha, governed);

  // An unclassified new file is governed by default. This protects the
  // identity if a future build input is added without updating the list.
  write(path.join(repository, 'new-build-input.cfg'), 'runtime-v2\n');
  const governedHead = commit('new governed build input');
  const advanced = sro.resolveBridgeSourceIdentity(repository);
  assert.equal(advanced.bridgeSourceSha, governedHead);
  assert.equal(advanced.bridgeBuildSha, governedHead);
}));

test('test_malformed_git_path_is_rejected', () => {
  assert.throws(() => sro.isGovernedBridgePath('../outside'), ContractError);
});

test('test_path_contract_is_fail_closed_for_runtime_and_new_inputs', () => {
  for (const file of [
    '.github/workflows/auto_llama_cpp_update.yml',
    'README.md',
    'docs/api.md',
    'scripts/bridge_js_source.py',
    'scripts/bridge_operation_queue_direct_cases.mjs',
    'scripts/bridge_operation_queue_fixtures.mjs',
    'scripts/bridge_operation_queue_lifecycle_contract_cases.mjs',
    'scripts/bridge_operation_queue_worker_proxy_cases.mjs',
    'scripts/browser_smoke_support.mjs',
    'scripts/ci_scope.mjs',
    'scripts/ci_scope.py',
    'scripts/ci_scope_test.py',
    'scripts/decision_browser_smoke.mjs',
    'scripts/decision_browser_smoke.py',
    'scripts/grammar_browser_smoke.mjs',
    'scripts/grammar_browser_smoke.py',
    'scripts/mtmd_compat_contract_test.py',
    'scripts/multimodal_browser_smoke.mjs',
    'scripts/multimodal_browser_smoke.py',
    'scripts/native_core_source.py',
    'scripts/next_token_scores_browser_smoke.mjs',
    'scripts/next_token_scores_browser_smoke.py',
    'scripts/orchestrator_source.py',
    'scripts/release_orchestrator_asset_releases.py',
    'scripts/release_orchestrator_asset_releases_test.py',
    'scripts/release_orchestrator_driver.py',
    'scripts/release_orchestrator_driver_backlog_test.py',
    'scripts/release_orchestrator_driver_identical_release_test.py',
    'scripts/release_orchestrator_driver_publication_test.py',
    'scripts/release_orchestrator_driver_test.py',
    'scripts/release_orchestrator_fixtures_test.py',
    'scripts/release_orchestrator_model.py',
    'scripts/release_orchestrator_native.py',
    'scripts/release_orchestrator_native_test.py',
    'scripts/release_orchestrator_planner.py',
    'scripts/release_orchestrator_planner_test.py',
    'scripts/release_orchestrator_release_tags.py',
    'scripts/release_orchestrator_release_tags_test.py',
    'scripts/release_orchestrator_run_names.py',
    'scripts/release_orchestrator_run_names_test.py',
    'scripts/release_orchestrator_stage_proofs.py',
    'scripts/release_orchestrator_transport.py',
    'scripts/release_orchestrator_transport_test.py',
    'scripts/release_orchestrator_workflow_runs.py',
    'scripts/release_orchestrator_workflow_runs_test.py',
    'scripts/release_qualification.py',
    'scripts/speech_to_text_browser_smoke.mjs',
    'scripts/speech_to_text_browser_smoke.py',
    'scripts/speech_to_text_fixture.json',
    'scripts/stable_release_orchestrator.py',
    'scripts/stable_release_orchestrator_test.py',
    'scripts/state_persistence_browser_smoke.mjs',
    'scripts/state_persistence_browser_smoke.py',
    'scripts/text_to_speech_browser_smoke.mjs',
    'scripts/text_to_speech_browser_smoke.py',
    'scripts/verify_ci_reliability.mjs',
    'scripts/verify_ci_reliability.py',
    'scripts/verify_ci_reliability_doc_facts_test.py',
    'scripts/verify_ci_reliability_pin_test.py',
    'scripts/verify_decision_api.py',
    'scripts/verify_state_persistence_api.py',
    'scripts/verify_text_to_speech_api.py',
    'scripts/wasm64_runtime_patch_contract_test.py',
    'scripts/worker_runtime_state_test.mjs',
    'tests/js/bridge_js_source.mjs',
    'tests/js/bridge_operation_queue_direct_cases.mjs',
    'tests/js/bridge_operation_queue_fixtures.mjs',
    'tests/js/bridge_operation_queue_lifecycle_contract_cases.mjs',
    'tests/js/bridge_operation_queue_worker_proxy_cases.mjs',
    'tests/js/browser_smoke_support_test.mjs',
    'tests/js/ci_scope_test.mjs',
    'tests/js/decision_api_contract_test.mjs',
    'tests/js/mtmd_compat_contract_test.mjs',
    'tests/js/multimodal_harness_parity_test.mjs',
    'tests/js/native_core_source.mjs',
    'tests/js/state_persistence_api_contract_test.mjs',
    'tests/js/state_persistence_harness_parity_test.mjs',
    'tests/js/text_to_speech_api_contract_test.mjs',
    'tests/js/verify_ci_reliability_test.mjs',
    'tests/js/wasm64_runtime_patch_contract_test.mjs',
    'tests/js/worker_runtime_state_test.mjs',
    'tests/js/nested/new_contract_test.mjs',
    'tests/new_suite/new_contract_test.mjs',
    'tests/release/contract_test.mjs',
    'scripts/release/archive.mjs',
    'scripts/release/publication_state.mjs',
    'scripts/release/qualification.mjs',
    'scripts/release/qualify.mjs',
    'scripts/release/wav.mjs',
    'scripts/release/orchestrator/asset_releases.mjs',
    'scripts/release/orchestrator/cli.mjs',
    'scripts/release/orchestrator/driver.mjs',
    'scripts/release/orchestrator/model.mjs',
    'scripts/release/orchestrator/native.mjs',
    'scripts/release/orchestrator/planner.mjs',
    'scripts/release/orchestrator/release_tags.mjs',
    'scripts/release/orchestrator/run_names.mjs',
    'scripts/release/orchestrator/stage_proofs.mjs',
    'scripts/release/orchestrator/transport.mjs',
    'scripts/release/orchestrator/workflow_runs.mjs',
    'tests/release/zip_fixture.mjs',
    'tests/release/orchestrator/fixtures.mjs',
    'tests/release/fixtures/attestation.json',
  ]) {
    assert.equal(sro.isGovernedBridgePath(file), false, file);
  }
  for (const file of [
    'CMakeLists.txt',
    '.github/workflows/bridge_candidate.yml',
    'package.json',
    'js/llama_webgpu_bridge.d.ts',
    'scripts/build_bridge.sh',
    'scripts/generate_release_manifest.py',
    // The wasm64 runtime patch rewrites a published asset: its Python
    // original and the Node port are both build inputs.
    'scripts/patch_wasm64_runtime.mjs',
    'scripts/patch_wasm64_runtime.py',
    'scripts/release_contract.py',
    // A new orchestrator module is listed explicitly, never by prefix.
    'scripts/release_orchestrator_unlisted.py',
    // So is a new smoke helper; only the smoke entry points match by suffix.
    'scripts/browser_smoke_unlisted.mjs',
    'scripts/browser_smoke.mjs',
    // Only the speech fixture is listed; another data file is governed.
    'scripts/text_to_speech_fixture.json',
    'scripts/verify_emscripten_version.py',
    'src/llama_webgpu_core.cpp',
    'src/core/exports_tts.inc',
    'src/core/new_part.inc',
    'tests/js/new_fixture.mjs',
    'tests/new_build_input.mjs',
    // Only tests/release/ is exempt as a whole.
    'tests/releases/new_fixture.mjs',
    'scripts/release/contract.mjs',
    'scripts/release/errors.mjs',
    'scripts/release/json.mjs',
    'scripts/release/cli.mjs',
    'scripts/release/manifest.mjs',
    'scripts/release/python_compat.mjs',
    'scripts/release/unlisted.mjs',
    'scripts/release/orchestrator/unlisted.mjs',
    'scripts/build/verify_emscripten_version.mjs',
    'unknown/new-build-input.cfg',
  ]) {
    assert.equal(sro.isGovernedBridgePath(file), true, file);
  }
});

// --- OrchestrationCallerAuthorizationTest ----------------------------------------

const WORKFLOW = path.join(ROOT, '.github', 'workflows', 'auto_llama_cpp_update.yml');

test('test_schedule_owner_manual_and_owner_continuation_are_authorized', () => {
  sro.requireOrchestrationCaller('schedule', 'github-actions', 'github-actions');
  sro.requireOrchestrationCaller('workflow_dispatch', OWNER, OWNER);
  sro.requireOrchestrationCaller('workflow_run', OWNER, OWNER);
});

test('test_non_owner_manual_or_continuation_is_rejected', () => {
  for (const event of ['workflow_dispatch', 'workflow_run']) {
    for (const [actor, triggeringActor] of [['collaborator', OWNER], [OWNER, 'collaborator']]) {
      assert.throws(() => sro.requireOrchestrationCaller(event, actor, triggeringActor), ContractError, `${event} ${actor}`);
    }
  }
});

test('test_workflow_run_trigger_and_both_job_gates_are_exact', () => {
  const workflow = fs.readFileSync(WORKFLOW, 'utf8');
  const count = (needle) => workflow.split(needle).length - 1;
  assert.ok(workflow.includes('fetch-depth: 0'));
  assert.ok(workflow.includes('scripts/stable_release_orchestrator.py resolve-bridge-source'));
  assert.ok(workflow.includes('--bridge-build-sha "${bridge_build_sha}"'));
  const manualGate = (
    "github.event_name == 'workflow_dispatch' && "
    + 'github.actor == github.repository_owner && '
    + 'github.triggering_actor == github.repository_owner'
  );
  const continuationGate = (
    "github.event_name == 'workflow_run' && "
    + "github.event.workflow_run.conclusion == 'success' && "
    + 'github.event.workflow_run.head_branch == '
    + 'github.event.repository.default_branch && '
    + 'github.event.workflow_run.actor.login == github.repository_owner && '
    + 'github.event.workflow_run.triggering_actor.login == '
    + 'github.repository_owner'
  );
  for (const trigger of [
    '- Build Exact Bridge Candidate',
    '- Qualify Exact Bridge Candidate',
    '- Publish Exact Qualified Bridge Assets',
    'types: [completed]',
  ]) {
    assert.ok(workflow.includes(trigger), trigger);
  }
  assert.equal(count(manualGate), 2);
  assert.equal(count(continuationGate), 2);
  assert.ok(workflow.lastIndexOf(manualGate) < workflow.indexOf('environment:'));
  assert.ok(workflow.lastIndexOf(continuationGate) < workflow.indexOf('environment:'));
  const proof = 'Prove the exact workflow continuation before environment use';
  assert.ok(workflow.includes(proof));
  const marker = `      - name: ${proof}\n`;
  const proofBlock = workflow.slice(workflow.indexOf(marker) + marker.length).split('\n      - name:')[0];
  assert.ok(proofBlock.includes('scripts/release_qualification.py verify-run'));
  assert.ok(proofBlock.includes('--run-attempt 1'));
  for (const exactMapping of [
    '.github/workflows/bridge_candidate.yml)\n              artifact_name=exact-webgpu-bridge-dist',
    '.github/workflows/bridge_qualification.yml)\n              artifact_name=qualification-attestation',
    '.github/workflows/publish_assets.yml)\n              artifact_name=bridge-qualification-outcome',
  ]) {
    assert.ok(proofBlock.includes(exactMapping), exactMapping);
  }
  assert.ok(workflow.indexOf(proof) < workflow.indexOf('environment:'));
});

// --- Node only ------------------------------------------------------------------

// The double-quoted string literals of a Python expression, comments
// skipped; the classifier's literals hold no quote, backslash or '#'.
function pythonStrings(source) {
  const strings = [];
  for (const line of source.split('\n')) {
    const code = line.replace(/#.*$/u, '');
    for (const match of code.matchAll(/"([^"\\]*)"/gu)) strings.push(match[1]);
  }
  return strings;
}

// The source text of `name = <opening>...<closing>` in the Python entry.
function pythonAssignment(source, name, opening, closing) {
  const start = source.indexOf(`\n${name} = ${opening}`);
  assert.notEqual(start, -1, name);
  const end = source.indexOf(closing, start);
  assert.notEqual(end, -1, name);
  return source.slice(start, end + closing.length);
}

test('classifier sets equal the Python sets they copy', () => {
  const source = fs.readFileSync(path.join(ROOT, 'scripts', 'stable_release_orchestrator.py'), 'utf8');
  const paths = pythonStrings(pythonAssignment(source, '_ORCHESTRATION_ONLY_PATHS', 'frozenset(\n    {', '\n    }\n)'));
  assert.equal(new Set(paths).size, paths.length);
  assert.deepEqual([...sro.ORCHESTRATION_ONLY_PATHS].sort(), [...paths].sort());
  assert.deepEqual(sro.ORCHESTRATION_ONLY_PREFIXES, pythonStrings(pythonAssignment(source, '_ORCHESTRATION_ONLY_PREFIXES', '(', ')\n')));
  assert.deepEqual(
    sro.ORCHESTRATION_ONLY_SCRIPT_SUFFIXES,
    pythonStrings(pythonAssignment(source, '_ORCHESTRATION_ONLY_SCRIPT_SUFFIXES', '(', '\n)\n')),
  );
  assert.deepEqual(
    [sro.ORCHESTRATION_ONLY_JS_TEST_SUFFIX],
    pythonStrings(pythonAssignment(source, '_ORCHESTRATION_ONLY_JS_TEST_SUFFIX', '"', '"\n')),
  );
  // Every orchestrator module is listed explicitly, never by prefix.
  const modules = fs.readdirSync(path.join(ROOT, 'scripts', 'release', 'orchestrator'))
    .filter((name) => name.endsWith('.mjs'))
    .map((name) => `scripts/release/orchestrator/${name}`);
  for (const module of modules) assert.ok(sro.ORCHESTRATION_ONLY_PATHS.has(module), module);
});

test('invalid paths are rejected with the Python repr', () => {
  for (const [value, repr] of [
    ['', "''"], ['/abs', "'/abs'"], ['a\\b', "'a\\\\b'"], ['a//b', "'a//b'"], ['a/./b', "'a/./b'"], ['a/', "'a/'"],
    [null, 'None'], [7, '7'],
  ]) {
    assert.throws(() => sro.isGovernedBridgePath(value), new ContractError(`git reported an invalid repository path: ${repr}`));
  }
});

test('the caller gate names the accepted callers', () => {
  assert.throws(
    () => sro.requireOrchestrationCaller('push', OWNER, OWNER),
    new ContractError(
      'stable orchestration requires a schedule event, or an owner-initiated '
      + 'workflow_dispatch or workflow_run continuation with owner actor and triggering_actor',
    ),
  );
});

test('orchestrating subcommands check the caller before any credential use', () => {
  for (const subcommand of ['orchestrate', 'orchestrate-backlog']) {
    const argv = [subcommand, subcommand === 'orchestrate' ? '--provenance-json' : '--provenance-list-json', 'missing.json', '--workspace', 'unused'];
    assert.throws(() => sro.main(argv, {
      env: { GITHUB_EVENT_NAME: 'pull_request' },
      createGateway: () => assert.fail('no gateway before the caller gate'),
      stdout: () => {},
      stderr: () => {},
    }), ContractError);
  }
});

test('the command line is argparse-compatible and never looser', () => {
  const parse = (argv) => sro.parseCommand(argv);
  assert.deepEqual(parse(['orchestrate', '--provenance-json', 'p.json', '--workspace', 'w/', '--dry-run']), {
    subcommand: 'orchestrate',
    args: {
      provenanceJson: 'p.json', workspace: 'w', outputPlanJson: null, stepSummaryFile: null, dryRun: true,
    },
  });
  assert.equal(parse(['orchestrate-backlog', '--provenance-list-json', 'p', '--workspace', 'w']).args.dryRun, false);
  assert.equal(parse(['resolve-bridge-source', '--repository', '.']).args.head, 'HEAD');
  const usage = (argv, status = 2) => assert.throws(() => parse(argv), (error) => error instanceof ArgparseExit && error.status === status);
  usage([]);
  usage(['bogus']);
  usage(['orchestrate', '--workspace', 'w']);
  usage(['orchestrate', '--provenance-json', 'p', '--workspace', 'w', '--dry-run=yes']);
  usage(['orchestrate', '--provenance-json', 'p', '--workspace', '--dry-run']);
  usage(['orchestrate', '--provenance-json', 'p', '--workspace', 'w', '--dry']);
  usage(['orchestrate', '--provenance-json', 'p', '--workspace', 'w', 'extra']);
  usage([
    'scan-native', '--manifest', 'm', '--native-release-tag', 't', '--bridge-source-sha', 's', '--bridge-build-sha', 'b',
    '--native-release-published-at', 'p', '--channel', 'nightly',
  ]);
  usage(['-h'], 0);
  usage(['orchestrate', '--help'], 0);
  assert.equal(parse([
    'scan-native', '--manifest', 'm', '--native-release-tag', 't', '--bridge-source-sha', 's', '--bridge-build-sha', 'b',
    '--native-release-published-at', 'p', '--channel', 'development',
  ]).args.channel, 'development');
});

test('runMain reports errors as the Python entry does', () => {
  const run = (argv, env = {}) => {
    let stdout = '';
    let stderr = '';
    const status = sro.runMain(argv, { env, stdout: (text) => { stdout += text; }, stderr: (text) => { stderr += text; } });
    return { status, stdout, stderr };
  };
  const denied = run(['orchestrate', '--provenance-json', 'p', '--workspace', 'w'], { GITHUB_EVENT_NAME: 'push' });
  assert.equal(denied.status, 1);
  assert.match(denied.stderr, /^error: stable orchestration requires a schedule event/u);
  const bogus = run(['bogus']);
  assert.equal(bogus.status, 2);
  assert.match(bogus.stderr, /error: argument subcommand: invalid choice: 'bogus'/u);
  const missing = run(['select-stable-native-backlog', '--releases-json', path.join(tmpdir(), 'sro-cli-missing', 'releases.json')]);
  assert.equal(missing.status, 1);
  assert.match(missing.stderr, /^Traceback \(most recent call last\):\nFileNotFoundError: \[Errno 2\] No such file or directory: /u);
});
