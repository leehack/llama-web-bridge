// Contract tests for the CI change selector (scripts/ci_scope.mjs) and the
// CI workflow shape it relies on.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { build as traceBuildModules } from 'esbuild';
import { parse as parseYaml } from 'yaml';

import { changedPaths, nativeRequired, validateResults } from '../../scripts/ci/ci_scope.mjs';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const workflow = parseYaml(readFileSync(path.join(rootDir, '.github/workflows/ci.yml'), 'utf8'));
const smokeJobs = ['smoke-core', 'smoke-resume', 'smoke-multimodal', 'smoke-grammar-qwen'];
const nativeJobs = ['build-core', ...smokeJobs];

// Documentation and tooling-only paths skip the native lane; everything else,
// including unknown paths, keeps it.
{
  for (const file of [
    'README.md', 'docs/guide.md', 'scripts/release_publication_state.py',
    'tests/js/native_core_source.mjs', 'tests/js/bridge_js_source.mjs',
    'tests/js/state_persistence_api_contract_test.mjs',
    'tests/js/text_to_speech_api_contract_test.mjs',
    'tests/js/decision_api_contract_test.mjs',
    'tests/js/mtmd_compat_contract_test.mjs',
    'tests/js/wasm64_runtime_patch_contract_test.mjs',
    'tests/js/worker_runtime_state_test.mjs',
    'tests/js/declared_class_fields_test.mjs',
    'scripts/verify_ci_reliability.mjs', 'tests/js/verify_ci_reliability_test.mjs',
    'tests/release/contract_test.mjs', 'tests/release/fixtures/attestation.json',
    'scripts/release/publication_state.mjs',
    // The tooling-only files moved from tests/js/ and scripts/ keep skipping the lane.
    'tests/bridge/native_core_source.mjs', 'tests/bridge/bridge_js_source.mjs',
    'tests/bridge/state_persistence_api_contract_test.mjs',
    'tests/bridge/text_to_speech_api_contract_test.mjs',
    'tests/bridge/decision_api_contract_test.mjs', 'tests/bridge/decision_bridge_contract_test.mjs',
    'tests/bridge/mtmd_compat_contract_test.mjs',
    'tests/build/wasm64_runtime_patch_contract_test.mjs',
    'tests/bridge/embedding_json_contract_test.mjs', 'tests/bridge/declared_class_fields_test.mjs',
    'tests/bridge/native_load_option_arity_test.mjs', 'tests/bridge/model_reload_contract_test.mjs',
    'tests/bridge/bridge_operation_queue_test.mjs', 'tests/bridge/bridge_operation_lifecycle_test.mjs',
    'tests/bridge/text_to_speech_recovery_test.mjs', 'tests/bridge/bridge_type_declaration_contract_test.mjs',
    'tests/bridge/worker_runtime_state_test.mjs', 'tests/bridge/worker_token_coalescing_test.mjs',
    'tests/bridge/workflow_input_transport_test.mjs',
    'scripts/ci/verify_ci_reliability.mjs', 'tests/ci/verify_ci_reliability_test.mjs',
  ]) {
    assert.equal(nativeRequired([file]), false, file);
  }
  for (const file of [
    'src/core.cpp', 'src/core/exports_tts.inc', 'src/core/new_part.inc', 'js/llama_webgpu_bridge.ts', 'package-lock.json',
    'scripts/build_bridge.sh', 'scripts/patch_wasm64_runtime.mjs', 'scripts/state_persistence_browser_smoke.py',
    'scripts/multimodal_browser_smoke.py', 'scripts/ci_scope.py', 'scripts/ci_scope.mjs',
    // Browser harnesses keep the build and smoke lanes, as their .py originals did.
    'scripts/state_persistence_browser_smoke.mjs', 'scripts/grammar_browser_smoke.mjs',
    'scripts/next_token_scores_browser_smoke.mjs', 'scripts/multimodal_browser_smoke.mjs',
    'scripts/browser_smoke_support.mjs', 'tests/js/browser_smoke_support_test.mjs',
    'tests/js/multimodal_harness_parity_test.mjs',
    'scripts/speech_to_text_browser_smoke.mjs', 'scripts/text_to_speech_browser_smoke.mjs',
    'scripts/decision_browser_smoke.mjs', 'scripts/speech_to_text_browser_smoke.py',
    'scripts/text_to_speech_browser_smoke.py', 'scripts/decision_browser_smoke.py',
    'tests/js/heavy_browser_smokes_test.mjs',
    // Read by the speech smoke, like release_qualification.py, which also keeps the lanes.
    'scripts/speech_to_text_fixture.json', 'scripts/release_qualification.py',
    'scripts/new_test.py', 'llama_cpp.version', 'emsdk.version',
    '.github/workflows/ci.yml', '.github/workflows/bridge_candidate.yml',
    'CMakeLists.txt', 'docs/native.cpp', 'unknown.lock',
    // Only the release tests are exempt; the release modules keep the lanes.
    'scripts/release/contract.mjs', 'scripts/release/qualification.mjs', 'tests/releases/x_test.mjs',
    // The files moved into scripts/build/, scripts/ci/, scripts/smoke/ and
    // tests/ keep the lanes their old paths kept.
    'scripts/build/build_bridge.sh', 'scripts/build/build_js_bridge.mjs', 'scripts/build/patch_wasm64_runtime.mjs',
    'scripts/ci/ci_scope.mjs', 'tests/ci/ci_scope_test.mjs',
    'scripts/smoke/state_persistence.mjs', 'scripts/smoke/grammar.mjs', 'scripts/smoke/next_token_scores.mjs',
    'scripts/smoke/multimodal.mjs', 'scripts/smoke/speech_to_text.mjs', 'scripts/smoke/text_to_speech.mjs',
    'scripts/smoke/decision.mjs', 'scripts/smoke/lora_adapter.mjs', 'scripts/smoke/speculative.mjs',
    'scripts/smoke/support.mjs', 'scripts/smoke/speech_to_text_fixture.json',
    'tests/smoke/browser_smoke_support_test.mjs', 'tests/smoke/heavy_browser_smokes_test.mjs',
    'tests/bridge/completion_usage_test.mjs', 'tests/bridge/lora_adapter_contract_test.mjs',
    'tests/bridge/speculative_decoding_test.mjs', 'tests/bridge/bridge_operation_queue_fixtures.mjs',
    // An unlisted file in a new directory is not tooling by its directory.
    'scripts/smoke/unlisted.mjs', 'tests/bridge/new_contract_test.mjs',
  ]) {
    assert.equal(nativeRequired(['README.md', file]), true, file);
  }
  assert.equal(nativeRequired([]), true);
}

// The aggregate passes only when every selected job succeeded and the skipped
// native lane was deselected.
{
  const states = ['success', 'failure', 'cancelled', 'skipped'];
  for (const selected of ['true', 'false']) {
    for (const changes of states) {
      for (const checks of states) {
        for (const job of nativeJobs) {
          for (const result of states) {
            const wanted = selected === 'true' ? 'success' : 'skipped';
            const needs = {
              changes: { result: changes, outputs: { native: selected } },
              checks: { result: checks },
              ...Object.fromEntries(nativeJobs.map((name) => [name, { result: wanted }])),
              [job]: { result },
            };
            const expected = changes === 'success' && checks === 'success' && result === wanted;
            assert.equal(validateResults(needs), expected, JSON.stringify(needs));
          }
        }
      }
    }
  }
  assert.equal(validateResults({}), false);
}

// Renames and deletions report both paths, an all-zero base selects the native
// lane, and a missing ref fails instead of skipping work.
{
  const temp = mkdtempSync(path.join(tmpdir(), 'ci-scope-'));
  const git = (...args) => execFileSync('git', ['-C', temp, '-c', 'commit.gpgsign=false', ...args], { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
  const previous = process.cwd();
  try {
    git('init');
    git('config', 'user.name', 'Fixture');
    git('config', 'user.email', 'fixture@example.test');
    mkdirSync(path.join(temp, 'src'));
    mkdirSync(path.join(temp, 'docs'));
    writeFileSync(path.join(temp, 'src/core.cpp'), 'native');
    git('add', '.');
    git('commit', '-m', 'base');
    let base = git('rev-parse', 'HEAD');
    process.chdir(temp);
    git('mv', 'src/core.cpp', 'docs/core.md');
    git('commit', '-m', 'rename');
    assert.deepEqual(new Set(changedPaths(base)), new Set(['src/core.cpp', 'docs/core.md']));
    assert.equal(nativeRequired(changedPaths(base)), true);
    writeFileSync(path.join(temp, 'unknown.lock'), 'dependency');
    git('add', '.');
    git('commit', '-m', 'add');
    base = git('rev-parse', 'HEAD');
    git('rm', 'unknown.lock');
    git('commit', '-m', 'delete');
    assert.equal(nativeRequired(changedPaths(base)), true);
    assert.equal(nativeRequired(changedPaths('0'.repeat(40))), true);
    assert.throws(() => execFileSync(process.execPath, [path.join(rootDir, 'scripts/ci/ci_scope.mjs')], {
      env: { ...process.env, BASE_SHA: 'missing-ref', GITHUB_OUTPUT: path.join(temp, 'output') },
      stdio: 'ignore',
    }));
    assert.throws(() => changedPaths('missing-ref'));
  } finally {
    process.chdir(previous);
    rmSync(temp, { recursive: true, force: true });
  }
}

{
  const { steps } = workflow.jobs['build-core'];
  const step = steps.find((step) => step.id === 'native-inputs');
  assert.ok(step, 'CI must fingerprint its checked-out native inputs');
  const scope = step.run.match(/git ls-tree -r -z HEAD -- ([^|]+)\| sha256sum/)?.[1];
  assert.ok(scope, 'CI must hash Git tree records');
  const inputs = scope.replaceAll('\\\n', ' ').trim().split(/\s+/);
  const entryPoints = readdirSync(path.join(rootDir, 'scripts/build'), { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile() && /\.[cm]?[jt]s$/.test(entry.name))
    .map((entry) => path.join(entry.parentPath, entry.name));
  const { metafile } = await traceBuildModules({
    absWorkingDir: rootDir, entryPoints, bundle: true, packages: 'external', platform: 'node',
    format: 'esm', outdir: 'out', write: false, metafile: true,
  });
  for (const module of Object.keys(metafile.inputs)) {
    assert.ok(inputs.some((input) => input === module || module.startsWith(`${input}/`)),
      `build module must be in the native input tree scope: ${module}`);
  }
  const temp = mkdtempSync(path.join(tmpdir(), 'ci-native-inputs-'));
  const output = path.join(temp, 'output');
  const git = (...args) => execFileSync('git', ['-C', temp, '-c', 'commit.gpgsign=false', ...args], { stdio: ['ignore', 'pipe', 'ignore'] });
  const run = () => {
    writeFileSync(output, '');
    execFileSync('bash', ['-c', `command -v sha256sum >/dev/null || sha256sum() { shasum -a 256; }\n${step.run}`], {
      cwd: temp, env: { ...process.env, GITHUB_OUTPUT: output }, stdio: 'pipe',
    });
    const result = readFileSync(output, 'utf8');
    assert.match(result, /^sha256=[0-9a-f]{64}\n$/);
    return result;
  };
  try {
    git('init');
    git('config', 'user.name', 'Fixture');
    git('config', 'user.email', 'fixture@example.test');
    assert.throws(run);
    assert.equal(readFileSync(output, 'utf8'), '');
    git('commit', '--allow-empty', '-m', 'empty');
    assert.throws(run, (error) => /native input tree fingerprint is missing or invalid/.test(error.stderr?.toString()));
    assert.equal(readFileSync(output, 'utf8'), '');
    mkdirSync(path.join(temp, 'src'));
    mkdirSync(path.join(temp, 'scripts/build'), { recursive: true });
    const header = 'src/llama_webgpu_grammar.h';
    const buildScript = 'scripts/build/build_bridge.sh';
    const compilerVerifier = 'scripts/build/verify_emscripten_version.mjs';
    writeFileSync(path.join(temp, header), '#pragma once\n');
    writeFileSync(path.join(temp, buildScript), '#!/usr/bin/env bash\n');
    writeFileSync(path.join(temp, compilerVerifier), 'export const maxBuffer = Infinity;\n');
    git('add', header, buildScript, compilerVerifier);
    git('update-index', '--chmod=+x', buildScript);
    git('commit', '-m', 'native inputs');
    const base = git('rev-parse', 'HEAD').toString().trim();
    const fingerprint = run();
    git('mv', header, 'src/llama_webgpu_grammar.hpp');
    git('commit', '-m', 'rename header');
    assert.notEqual(run(), fingerprint, 'a same-byte source rename must change the core fingerprint');
    git('checkout', '--detach', base);
    writeFileSync(path.join(temp, compilerVerifier), 'export const maxBuffer = 256;\n');
    git('add', compilerVerifier);
    git('commit', '-m', 'change compiler verifier');
    assert.notEqual(run(), fingerprint, 'changing the Emscripten verifier must change the core fingerprint');
    git('checkout', '--detach', base);
    git('update-index', '--chmod=-x', buildScript);
    git('commit', '-m', 'remove executable bit');
    assert.notEqual(run(), fingerprint, 'a mode-only change must change the core fingerprint');
  } finally {
    rmSync(temp, { recursive: true, force: true });
  }
}

// CI builds the pinned native lane behind the selector and reports a truthful
// aggregate.
{
  const { jobs } = workflow;
  const build = jobs['build-core'];
  const { checks } = jobs;
  assert.ok(build, 'CI builds the cores in a matrix');
  assert.deepEqual(build.strategy.matrix, { variant: ['wasm32', 'wasm64', 'wasm32_st', 'wasm64_st'] });
  assert.equal(build.strategy['fail-fast'], true);
  assert.deepEqual(build.needs, ['changes']);
  for (const name of nativeJobs) {
    assert.equal(jobs[name].if, "needs.changes.outputs.native == 'true'", name);
  }
  for (const name of smokeJobs) {
    assert.deepEqual(jobs[name].needs, ['changes', 'build-core'], name);
  }
  assert.equal(jobs['ci-result'].name, 'CI validation');
  assert.equal(jobs['ci-result'].if, 'always()');
  assert.deepEqual(new Set(jobs['ci-result'].needs), new Set(['changes', 'checks', ...nativeJobs]));
  assert.ok(jobs['ci-result'].steps.some((step) => step.run === 'node scripts/ci/ci_scope.mjs --check-results'));
  assert.ok(jobs.changes.steps.some((step) => step.run === 'node scripts/ci/ci_scope.mjs'));
  assert.equal(
    Object.values(jobs).flatMap((job) => job.steps).filter((step) => (step.run ?? '').includes('npm run check:js')).length,
    1,
  );
  const clone = build.steps.find((step) => step.name === 'Clone llama.cpp source').run;
  assert.doesNotMatch(clone, /matrix\.upstream/);
  assert.match(
    clone,
    /test "\$\(git -C third_party\/llama_cpp rev-parse HEAD\)" = 171527522610996f5f0acf12334e185156ca8f44\b/,
  );
  const resolver = build.steps.find((step) => step.name === 'Resolve llama.cpp pin').run;
  assert.match(resolver, /llama_cpp.version/);
  assert.doesNotMatch(resolver, /matrix\.upstream/);
  const pinnedArtifacts = [
    'state-persistence-smoke-artifacts',
    'multimodal-smoke-artifacts',
    'grammar-smoke-artifacts',
    'grammar-smoke-artifacts-qwen-${{ matrix.memory }}-${{ matrix.runtime }}',
    'next-token-scores-smoke-artifacts',
    'speculative-smoke-artifacts',
    'webgpu-bridge-dist',
  ];
  const uploadNames = smokeJobs.flatMap((name) => jobs[name].steps)
    .filter((step) => step.uses?.startsWith('actions/upload-artifact@'))
    .map((step) => step.with.name);
  assert.deepEqual(uploadNames.sort(), [...pinnedArtifacts].sort());
  const buildSteps = Object.fromEntries(build.steps.map((step) => [step.name, step]));
  assert.ok(buildSteps['Upload core artifacts'].with['retention-days'] >= 30, 'core artifacts must outlive the 30-day rerun window');
  const coreCache = buildSteps['Restore built core'];
  assert.match(coreCache.with.key, /^webgpu-core-v1-/);
  assert.equal('restore-keys' in coreCache.with, false);
  assert.equal(coreCache.with.path, '${{ runner.temp }}/webgpu_bridge_dist');
  const cache = buildSteps['Restore compiler objects'].with;
  assert.match(cache.key, /^webgpu-ccache-v3-/);
  assert.doesNotMatch(cache.key, /github\.sha/);
  assert.match(buildSteps['Restore compiler objects'].uses, /^actions\/cache\/restore@/);
  const save = buildSteps['Save compiler objects'];
  assert.match(save.if, /github\.event_name == 'push' && github\.ref == 'refs\/heads\/main'/);
  assert.match(save.if, /steps\.ccache\.outputs\.cache-hit != 'true'/);
  assert.equal(save.with.path, cache.path);
  assert.equal(save.with.key, '${{ steps.ccache.outputs.cache-primary-key }}');
  const order = build.steps.map((step) => step.name);
  assert.ok(order.indexOf('Build bridge artifacts') < order.indexOf('Save compiler objects'));
  assert.equal(cache.path, '${{ runner.temp }}/webgpu-ccache');
  for (const token of ['runner.arch', 'matrix.variant', 'env.EMSCRIPTEN_VERSION', 'steps.compiler-cache.outputs.upstream', 'hashFiles']) {
    assert.ok(cache.key.includes(token), token);
  }
  assert.equal(buildSteps['Build bridge artifacts'].env.EM_COMPILER_WRAPPER, 'ccache');
  assert.ok(workflow.concurrency['cancel-in-progress'].includes("github.event_name == 'pull_request'"));
  for (const trigger of ['pull_request', 'push']) assert.equal('paths' in workflow.on[trigger], false, trigger);
  assert.ok(checks.steps.some((step) => (step.run ?? '').includes('node scripts/ci/verify_ci_reliability.mjs')));
  for (const job of Object.values(workflow.jobs)) {
    for (const step of job.steps ?? []) assert.doesNotMatch(step.run ?? '', /\bpython[0-9.]*\b|unittest|py_compile/, step.name);
  }
}

console.log('CI scope contract passed');
