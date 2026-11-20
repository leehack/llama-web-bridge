import assert from 'node:assert/strict';
import { test } from 'node:test';

import { withCrossOriginCore } from './pthread_start_gate_fixture.mjs';

test('a blocked cross-origin pthread worker rejects with the public error', async () => {
  await withCrossOriginCore({ readsWorkerScript: true, blocked: true }, async (runtime, state) => {
    await assert.rejects(runtime._ensureCore(), {
      name: 'BlobWorkerBlockedError',
      code: 'BLOB_WORKER_BLOCKED',
      target: 'pthread',
      reason: 'unavailable',
    });
    assert.equal(state.coreCalls.length, 2);
    assert.equal(state.workers.length, 1);
  });
});
