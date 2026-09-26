// scripts/multimodal_browser_smoke.py stays only as a qualification harness
// source (the speech smoke imports it) while CI and the candidate's multimodal
// gate run the Node port. Both must serve the same harness page, so a change
// to one without the other fails here instead of after the cutover. Delete
// this test with the Python smoke.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { renderHarness } from '../../scripts/multimodal_browser_smoke.mjs';

const scripts = fileURLToPath(new URL('../../scripts/', import.meta.url));
const renderPython = `
import sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import multimodal_browser_smoke as smoke
with tempfile.TemporaryDirectory() as tmp:
    smoke.write_harness(Path(tmp))
    sys.stdout.write((Path(tmp) / "index.html").read_text(encoding="utf-8"))
`;
const result = spawnSync('python3', ['-c', renderPython, scripts], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
assert.equal(result.status, 0, result.stderr);
assert.equal(renderHarness(), result.stdout, 'the multimodal harness must match the Python smoke');

console.log('Multimodal harness parity passed');
