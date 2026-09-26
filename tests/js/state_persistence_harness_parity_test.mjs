// While the candidate's state gate and the qualification harness still run
// scripts/state_persistence_browser_smoke.py, CI runs the Node port. Both must
// serve the same harness page, so a change to one without the other fails
// here instead of in a first-attempt-only candidate. Delete this test with the
// Python smoke.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { renderHarness } from '../../scripts/state_persistence_browser_smoke.mjs';

const scripts = fileURLToPath(new URL('../../scripts/', import.meta.url));
const renderPython = `
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import state_persistence_browser_smoke as smoke
pages = {}
for name in (None, "state-smoke-model.gguf"):
    with tempfile.TemporaryDirectory() as tmp:
        smoke.write_harness(Path(tmp), name)
        pages[str(name)] = (Path(tmp) / "index.html").read_text(encoding="utf-8")
print(json.dumps(pages))
`;
const result = spawnSync('python3', ['-c', renderPython, scripts], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
assert.equal(result.status, 0, result.stderr);
const pages = JSON.parse(result.stdout);
assert.equal(renderHarness(null), pages.None, 'the no-model harness must match the Python smoke');
assert.equal(renderHarness('state-smoke-model.gguf'), pages['state-smoke-model.gguf'], 'the model-backed harness must match the Python smoke');

console.log('State persistence harness parity passed');
