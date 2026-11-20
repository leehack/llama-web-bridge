import assert from 'node:assert/strict';
import { test } from 'node:test';

import { withCrossOriginCore } from './pthread_start_gate_fixture.mjs';

test('a CDN specifier that resolves to a same-origin core bypasses the blob probe', async () => {
  await withCrossOriginCore({
    readsWorkerScript: true, blocked: true, actualUrl: 'https://app.test/real.js',
  }, async (runtime, state) => {
    const core = await runtime._ensureCore();
    assert.equal(core.script, undefined);
    assert.equal(runtime._pthreadStart, 'direct');
    assert.equal(state.coreCalls.length, 1);
    assert.equal(state.workers.length, 0);
  });
});

test('a same-origin alias for a CDN core probes and wraps the actual script URL', async () => {
  await withCrossOriginCore({
    readsWorkerScript: true, coreUrl: 'https://app.test/alias.js', actualUrl: 'https://cdn.test/real.js',
  }, async (runtime, state) => {
    const core = await runtime._ensureCore();
    assert.match(core.script, /^blob:/);
    assert.equal(runtime._pthreadStart, 'blob');
    assert.equal(state.coreCalls.length, 2);
    assert.equal(state.workers.length, 2);
    assert.equal(await state.blobs.at(-1).text(), 'import "https://cdn.test/real.js";\n');
  });
});
