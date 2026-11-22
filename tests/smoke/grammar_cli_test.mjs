import assert from 'node:assert/strict';
import { test } from 'node:test';

import { parseArgs, renderHarness, validatePayload } from '../../scripts/smoke/grammar.mjs';

test('grammar CLI selects one runtime or defaults to both', () => {
  assert.equal(parseArgs([]).runtimeMode, 'all');
  assert.equal(parseArgs(['--runtime-mode', 'direct']).runtimeMode, 'direct');
  assert.equal(parseArgs(['--runtime-mode', 'worker']).runtimeMode, 'worker');
  assert.throws(() => parseArgs(['--runtime-mode', 'other']), /invalid choice/);
});

test('grammar renders and validates only the selected mode', () => {
  assert.match(renderHarness(1024, ['wasm32'], ['direct']), /"runtimeModes": \["direct"\]/);
  const direct = { mode: 'wasm32 direct' };
  const worker = { mode: 'wasm32 worker' };
  assert.deepEqual(validatePayload({ ok: true, modeResults: [direct] }, ['wasm32'], ['direct']), [
    'wasm32 direct: case results missing',
  ]);
  for (const modeResults of [[worker], [direct, worker]]) {
    assert.deepEqual(validatePayload({ ok: true, modeResults }, ['wasm32'], ['direct']), ['mode results missing']);
  }
});
