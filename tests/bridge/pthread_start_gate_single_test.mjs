import assert from 'node:assert/strict';
import { test } from 'node:test';

import { withCrossOriginCore } from './pthread_start_gate_fixture.mjs';

test('a cross-origin core that starts no worker never probes', async () => {
  await withCrossOriginCore({ readsWorkerScript: false }, async (runtime, state) => {
    await runtime._ensureCore();
    assert.equal(state.coreCalls.length, 1);
    assert.equal(state.workers.length, 0);
    assert.equal(state.blobs.length, 0);
    assert.equal(runtime._pthreadStart, 'direct');
  });
});
