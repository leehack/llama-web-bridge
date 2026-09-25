#!/usr/bin/env python3
"""Conservative CI build selection; release and candidate workflows are unchanged."""
import argparse
import json
import os
from pathlib import Path
import subprocess

# Only paths with no compiled-runtime dependency may avoid native jobs.
TOOLING = frozenset({
    'scripts/release_publication_state.py', 'scripts/release_publication_state_test.py',
    'scripts/stable_release_orchestrator.py', 'scripts/stable_release_orchestrator_test.py',
    'scripts/orchestrator_source.py',
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
    'scripts/bridge_js_source.py', 'scripts/native_core_source.py', 'tests/js/native_core_source.mjs',
    'tests/js/bridge_js_source.mjs', 'tests/js/state_persistence_api_contract_test.mjs',
    'tests/js/text_to_speech_api_contract_test.mjs', 'tests/js/decision_api_contract_test.mjs',
    'tests/js/decision_bridge_contract_test.mjs',
    'scripts/verify_ci_reliability.py', 'scripts/verify_ci_reliability_pin_test.py',
    'scripts/verify_ci_reliability_doc_facts_test.py',
    'scripts/mtmd_compat_contract_test.py',
    'scripts/wasm64_runtime_patch_contract_test.py',
    'tests/js/embedding_json_contract_test.mjs', 'tests/js/declared_class_fields_test.mjs', 'tests/js/native_load_option_arity_test.mjs',
    'tests/js/model_reload_contract_test.mjs',
    'tests/js/bridge_operation_queue_test.mjs',
    'tests/js/bridge_operation_lifecycle_test.mjs', 'tests/js/text_to_speech_recovery_test.mjs',
    'tests/js/bridge_type_declaration_contract_test.mjs', 'tests/js/worker_runtime_state_test.mjs',
    'tests/js/worker_token_coalescing_test.mjs', 'tests/js/workflow_input_transport_test.mjs',
})


def native_required(paths):
    return not paths or any(
        path not in TOOLING and not (
            path in {'README.md', 'CONTRIBUTING.md', 'LICENSE'}
            or (path.startswith('docs/') and path.endswith('.md'))
        ) for path in paths
    )


def changed_paths(base, head='HEAD'):
    # --no-renames reports both old deletion and new addition, including moves
    # from native inputs into docs/tooling. NUL records preserve arbitrary names.
    if not base or set(base) == {'0'}:
        return []
    output = subprocess.check_output([
        'git', 'diff', '--no-renames', '--name-only', '-z', base, head, '--'
    ])
    return output.decode('utf-8', errors='surrogateescape').rstrip('\0').split('\0') if output else []


def validate_results(needs):
    scope = needs.get('changes', {})
    if scope.get('result') != 'success':
        return False
    selected = scope.get('outputs', {}).get('native')
    if selected not in ('true', 'false'):
        return False
    expected = 'success' if selected == 'true' else 'skipped'
    jobs = {'build-webgpu-bridge'}
    if needs.get('checks', {}).get('result') != 'success':
        return False
    return set(needs) == jobs | {'changes', 'checks'} and all(
        needs[job].get('result') == expected for job in jobs
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-results', action='store_true')
    args = parser.parse_args()
    if args.check_results:
        return 0 if validate_results(json.loads(os.environ['NEEDS_JSON'])) else 1
    # An unavailable diff fails the classifier, never silently skips work.
    paths = changed_paths(os.environ.get('BASE_SHA', ''))
    value = str(native_required(paths)).lower()
    print(json.dumps({'native': value, 'paths': paths}))
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as out:
        out.write(f'native={value}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
