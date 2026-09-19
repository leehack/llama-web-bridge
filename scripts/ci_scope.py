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
    'scripts/release_qualification_test.py', 'scripts/release_contract_test.py',
    'scripts/generate_release_manifest_test.py',
    'scripts/verify_state_persistence_api.py', 'scripts/verify_text_to_speech_api.py',
    'scripts/verify_ci_reliability.py', 'scripts/mtmd_compat_contract_test.py',
    'scripts/wasm64_runtime_patch_contract_test.py',
    'scripts/embedding_json_contract_test.mjs', 'scripts/bridge_operation_queue_test.mjs',
    'scripts/bridge_operation_lifecycle_test.mjs', 'scripts/text_to_speech_recovery_test.mjs',
    'scripts/bridge_type_declaration_contract_test.mjs', 'scripts/worker_runtime_state_test.mjs',
    'scripts/worker_token_coalescing_test.mjs', 'scripts/workflow_input_transport_test.mjs',
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
