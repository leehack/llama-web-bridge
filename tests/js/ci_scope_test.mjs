// Contract tests for the CI change selector (scripts/ci_scope.mjs) and the
// CI workflow shape it relies on.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { parse as parseYaml } from 'yaml';

import { changedPaths, nativeRequired, validateResults } from '../../scripts/ci_scope.mjs';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

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
    'scripts/stable_release_orchestrator.py', 'scripts/stable_release_orchestrator_test.py',
    'scripts/release_orchestrator_asset_releases.py',
    'scripts/release_orchestrator_asset_releases_test.py', 'scripts/release_orchestrator_driver.py',
    'scripts/release_orchestrator_driver_test.py', 'scripts/release_orchestrator_driver_backlog_test.py',
    'scripts/release_orchestrator_driver_identical_release_test.py',
    'scripts/release_orchestrator_driver_publication_test.py', 'scripts/release_orchestrator_fixtures_test.py',
    'scripts/release_orchestrator_model.py', 'scripts/release_orchestrator_native.py',
    'scripts/release_orchestrator_native_test.py', 'scripts/release_orchestrator_planner.py',
    'scripts/release_orchestrator_planner_test.py', 'scripts/release_orchestrator_release_tags.py',
    'scripts/release_orchestrator_release_tags_test.py', 'scripts/release_orchestrator_run_names.py',
    'scripts/release_orchestrator_run_names_test.py', 'scripts/release_orchestrator_stage_proofs.py',
    'scripts/release_orchestrator_transport.py', 'scripts/release_orchestrator_transport_test.py',
    'scripts/release_orchestrator_workflow_runs.py', 'scripts/release_orchestrator_workflow_runs_test.py',
    'tests/js/worker_runtime_state_test.mjs',
    'tests/js/declared_class_fields_test.mjs',
    'scripts/verify_ci_reliability.mjs', 'tests/js/verify_ci_reliability_test.mjs',
  ]) {
    assert.equal(nativeRequired([file]), false, file);
  }
  for (const file of [
    'src/core.cpp', 'src/core/exports_tts.inc', 'src/core/new_part.inc', 'js/llama_webgpu_bridge.ts', 'package-lock.json',
    'scripts/build_bridge.sh', 'scripts/patch_wasm64_runtime.mjs', 'scripts/state_persistence_browser_smoke.py',
    'scripts/multimodal_browser_smoke.py', 'scripts/ci_scope.py', 'scripts/ci_scope.mjs',
    'scripts/new_test.py', 'llama_cpp.version', 'emsdk.version',
    '.github/workflows/ci.yml', '.github/workflows/bridge_candidate.yml',
    'CMakeLists.txt', 'docs/native.cpp', 'unknown.lock',
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
        for (const build of states) {
          const needs = {
            changes: { result: changes, outputs: { native: selected } },
            checks: { result: checks },
            'build-webgpu-bridge': { result: build },
          };
          const expected = changes === 'success' && checks === 'success'
            && build === (selected === 'true' ? 'success' : 'skipped');
          assert.equal(validateResults(needs), expected, JSON.stringify(needs));
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
    assert.throws(() => execFileSync(process.execPath, [path.join(rootDir, 'scripts/ci_scope.mjs')], {
      env: { ...process.env, BASE_SHA: 'missing-ref', GITHUB_OUTPUT: path.join(temp, 'output') },
      stdio: 'ignore',
    }));
    assert.throws(() => changedPaths('missing-ref'));
  } finally {
    process.chdir(previous);
    rmSync(temp, { recursive: true, force: true });
  }
}

// CI builds one native lane behind the selector and reports a truthful
// aggregate.
{
  const workflow = parseYaml(readFileSync(path.join(rootDir, '.github/workflows/ci.yml'), 'utf8'));
  const { jobs } = workflow;
  const build = jobs['build-webgpu-bridge'];
  const { checks } = jobs;
  assert.equal(build.name, 'Build WebGPU Bridge (WASM)');
  assert.equal('strategy' in build, false);
  assert.deepEqual(build.needs, ['changes', 'checks']);
  assert.equal(build.if, "needs.changes.outputs.native == 'true'");
  assert.equal(jobs['ci-result'].if, 'always()');
  assert.deepEqual(new Set(jobs['ci-result'].needs), new Set(['changes', 'checks', 'build-webgpu-bridge']));
  assert.ok(jobs['ci-result'].steps.some((step) => step.run === 'node scripts/ci_scope.mjs --check-results'));
  assert.ok(jobs.changes.steps.some((step) => step.run === 'node scripts/ci_scope.mjs'));
  assert.equal(
    Object.values(jobs).flatMap((job) => job.steps).filter((step) => (step.run ?? '').includes('npm run check:js')).length,
    1,
  );
  const buildSteps = Object.fromEntries(build.steps.map((step) => [step.name, step]));
  for (const name of ['Run state persistence browser smoke', 'Run multimodal browser smoke', 'Build bridge artifacts', 'Verify outputs']) {
    assert.equal('if' in buildSteps[name], false, name);
  }
  const cache = buildSteps['Restore compiler objects'].with;
  assert.match(buildSteps['Restore compiler objects'].uses, /^actions\/cache\/restore@/);
  const save = buildSteps['Save compiler objects'];
  assert.equal(save.if, "github.event_name == 'push' && github.ref == 'refs/heads/main'");
  assert.equal(save.with.path, cache.path);
  assert.equal(save.with.key, '${{ steps.ccache.outputs.cache-primary-key }}');
  const order = build.steps.map((step) => step.name);
  assert.ok(order.indexOf('Build bridge artifacts') < order.indexOf('Save compiler objects'));
  assert.ok(order.indexOf('Save compiler objects') < order.indexOf('Verify outputs'));
  assert.equal(cache.path, '${{ runner.temp }}/webgpu-ccache');
  for (const token of ['runner.arch', 'env.EMSCRIPTEN_VERSION', 'steps.compiler-cache.outputs.upstream', 'hashFiles']) {
    assert.ok(cache.key.includes(token), token);
    assert.ok(cache['restore-keys'].includes(token), token);
  }
  assert.equal(buildSteps['Build bridge artifacts'].env.EM_COMPILER_WRAPPER, 'ccache');
  assert.ok(workflow.concurrency['cancel-in-progress'].includes("github.event_name == 'pull_request'"));
  for (const trigger of ['pull_request', 'push']) assert.equal('paths' in workflow.on[trigger], false, trigger);
  assert.ok(checks.steps.some((step) => (step.run ?? '').includes("unittest discover -s scripts -p '*_test.py'")));
}

console.log('CI scope contract passed');
