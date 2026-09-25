import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from ci_scope import changed_paths, native_required, validate_results

ROOT = Path(__file__).resolve().parents[1]


class ScopeTests(unittest.TestCase):
    def test_allowlist_and_unknown_inputs(self):
        for path in ('README.md', 'docs/guide.md', 'scripts/release_publication_state.py', 'scripts/bridge_js_source.py', 'scripts/native_core_source.py',
                     'tests/js/native_core_source.mjs',
                     'scripts/stable_release_orchestrator.py', 'scripts/stable_release_orchestrator_test.py',
                     'scripts/orchestrator_source.py', 'scripts/release_orchestrator_asset_releases.py',
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
                     'scripts/verify_ci_reliability_pin_test.py', 'scripts/verify_ci_reliability_doc_facts_test.py'):
            self.assertFalse(native_required([path]), path)
        for path in ('src/core.cpp', 'src/core/exports_tts.inc', 'src/core/new_part.inc', 'js/llama_webgpu_bridge.ts', 'package-lock.json',
                     'scripts/build_bridge.sh', 'scripts/state_persistence_browser_smoke.py',
                     'scripts/multimodal_browser_smoke.py', 'scripts/ci_scope.py',
                     'scripts/new_test.py', 'llama_cpp.version', 'emsdk.version',
                     '.github/workflows/ci.yml', '.github/workflows/bridge_candidate.yml',
                     'CMakeLists.txt', 'docs/native.cpp', 'unknown.lock'):
            self.assertTrue(native_required(['README.md', path]), path)
        self.assertTrue(native_required([]))

    def test_all_aggregate_results(self):
        for selected in ('true', 'false'):
            for states in itertools.product(('success', 'failure', 'cancelled', 'skipped'), repeat=3):
                needs = {name: {'result': state} for name, state in zip(('changes', 'checks', 'build-webgpu-bridge'), states)}
                needs['changes']['outputs'] = {'native': selected}
                self.assertEqual(validate_results(needs), states == ('success', 'success', 'success' if selected == 'true' else 'skipped'))
        self.assertFalse(validate_results({}))

    def test_git_rename_delete_and_missing_ref(self):
        with tempfile.TemporaryDirectory() as temp:
            def git(*args):
                return subprocess.check_output(['git', '-C', temp, *args], stderr=subprocess.DEVNULL).decode().strip()
            git('init'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.test')
            root = Path(temp); (root/'src').mkdir(); (root/'docs').mkdir()
            (root/'src/core.cpp').write_text('native'); git('add', '.'); git('commit', '-m', 'base')
            base = git('rev-parse', 'HEAD'); previous = Path.cwd()
            try:
                os.chdir(root)
                git('mv', 'src/core.cpp', 'docs/core.md'); git('commit', '-m', 'rename')
                self.assertEqual(set(changed_paths(base)), {'src/core.cpp', 'docs/core.md'})
                self.assertTrue(native_required(changed_paths(base)))
                (root/'unknown.lock').write_text('dependency'); git('add', '.'); git('commit', '-m', 'add')
                base = git('rev-parse', 'HEAD'); git('rm', 'unknown.lock'); git('commit', '-m', 'delete')
                self.assertTrue(native_required(changed_paths(base)))
                self.assertTrue(native_required(changed_paths('0'*40)))
                with self.assertRaises(subprocess.CalledProcessError): changed_paths('missing-ref')
            finally:
                os.chdir(previous)

    def test_workflow_preserves_both_lanes_and_truthful_aggregate(self):
        workflow = json.loads(subprocess.check_output(['node', '-e', "const fs=require('fs'),yaml=require('yaml');process.stdout.write(JSON.stringify(yaml.parse(fs.readFileSync('.github/workflows/ci.yml','utf8'))))"], cwd=ROOT))
        jobs = workflow['jobs']; build = jobs['build-webgpu-bridge']; checks = jobs['checks']
        self.assertEqual(build['strategy']['matrix']['upstream'], ['pinned', 'v0.4.0'])
        self.assertEqual(build['needs'], ['changes', 'checks'])
        self.assertEqual(build['if'], "needs.changes.outputs.native == 'true'")
        self.assertEqual(jobs['ci-result']['if'], 'always()')
        self.assertEqual(set(jobs['ci-result']['needs']), {'changes', 'checks', 'build-webgpu-bridge'})
        self.assertTrue(any(step.get('run') == 'python3 scripts/ci_scope.py --check-results' for step in jobs['ci-result']['steps']))
        self.assertEqual(sum('npm run check:js' in step.get('run', '') for job in jobs.values() for step in job['steps']), 1)
        build_steps = {step.get('name'): step for step in build['steps']}
        for name in ('Run state persistence browser smoke', 'Run multimodal browser smoke', 'Build bridge artifacts', 'Verify outputs'):
            self.assertNotIn('if', build_steps[name])
        cache = build_steps['Cache compiler objects']['with']
        self.assertEqual(cache['path'], '${{ runner.temp }}/webgpu-ccache')
        for token in ('runner.arch', 'env.EMSCRIPTEN_VERSION', 'steps.compiler-cache.outputs.upstream', 'hashFiles'):
            self.assertIn(token, cache['key']); self.assertIn(token, cache['restore-keys'])
        self.assertEqual(build_steps['Build bridge artifacts']['env']['EM_COMPILER_WRAPPER'], 'ccache')
        self.assertIn("github.event_name == 'pull_request'", workflow['concurrency']['cancel-in-progress'])
        for trigger in ('pull_request', 'push'):
            self.assertNotIn('paths', workflow['on'][trigger])
        self.assertTrue(any("unittest discover -s scripts -p '*_test.py'" in step.get('run', '') for step in checks['steps']))


if __name__ == '__main__': unittest.main()
