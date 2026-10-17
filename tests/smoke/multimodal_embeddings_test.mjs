import assert from 'node:assert/strict';
import { test } from 'node:test';
import { main, renderHarness } from '../../scripts/smoke/multimodal_embeddings.mjs';

test('embedding qualification requires local inputs before any download or browser launch', async () => {
  await assert.rejects(main([]), /Local model, projector, reference and artifacts paths are required/);
});

test('embedding harness covers independent native parity, runtime identity and media ownership', () => {
  for (const mode of ['wasm32', 'wasm64']) {
    const page = renderHarness(mode);
    assert.ok(page.includes('for(const disableWorker of [true,false])'));
    for (const contract of ['reference[name].normalized','reference.black.raw','reference.text.raw',
      "'interleaved'",'directWorker','EMBEDDING_INPUT_TOO_LARGE','unloadMultimodalProjector',
      'caller buffers detached','text recovery after cancellation']) {
      // The interleaved fixture is a named object property, not a quoted key.
      assert.ok(page.includes(contract === "'interleaved'" ? 'interleaved:' : contract), contract);
    }
    assert.equal(page.includes("coreModuleUrlMem64:'/llama_webgpu_core_mem64.js'"), mode === 'wasm64');
  }
});
