import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { test } from 'node:test';

import { probeBlobWorker } from '../../js/src/internal/worker_start.ts';
import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';
import { withCrossOriginCore } from './pthread_start_gate_fixture.mjs';

test('the public model load stops at its own shared probe deadline', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const coreModuleUrl = 'https://cdn.test/core.js';
  await withCrossOriginCore({ readsWorkerScript: true, suspendImportProbe: true }, async (_runtime, state) => {
    const longProbe = probeBlobWorker(coreModuleUrl, 20_000);
    const shortBridge = new LlamaWebGpuBridge({
      disableWorker: true, allowSingleThreadCore: true, coreModuleUrl, coreInitTimeoutMs: 10_000,
    });
    let outcome;
    const loading = shortBridge.loadModelFromUrl('https://models.test/tiny.gguf').then(
      () => { outcome = 'success'; },
      (error) => { outcome = error; },
    );
    try {
      for (let attempt = 0; state.coreCalls.length === 0 && attempt < 40; attempt++) {
        await setImmediate();
      }
      assert.equal(state.coreCalls.length, 1);
      assert.equal(state.workers.length, 2);

      t.mock.timers.tick(10_000);
      await setImmediate();
      assert.equal(outcome?.code, 'BLOB_WORKER_BLOCKED', 'the public load outlived its probe budget');
      assert.equal(outcome.reason, 'script');
      assert.equal(outcome.target, 'pthread');
      assert.equal(state.coreCalls.length, 1);
    } finally {
      state.workers[1].onerror({});
      t.mock.timers.tick(0);
      assert.equal(await longProbe, 'script');
      await loading;
      await shortBridge.dispose();
    }
  });
});
