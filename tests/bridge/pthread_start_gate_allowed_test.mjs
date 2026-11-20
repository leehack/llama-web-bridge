import assert from 'node:assert/strict';
import { test } from 'node:test';

import { withCrossOriginCore } from './pthread_start_gate_fixture.mjs';

test('a cross-origin core is retried once after requesting a pthread worker', async () => {
  await withCrossOriginCore({ readsWorkerScript: true }, async (runtime, state) => {
    const core = await runtime._ensureCore();
    assert.equal(state.coreCalls.length, 2);
    assert.notEqual(state.coreCalls[0], state.coreCalls[1]);
    assert.equal(state.workers.length, 2);
    assert.match(core.script, /^blob:/);
    assert.equal(runtime._pthreadStart, 'blob');
  });
});
