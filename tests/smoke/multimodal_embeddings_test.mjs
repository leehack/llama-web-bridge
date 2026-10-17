import assert from 'node:assert/strict';
import { test } from 'node:test';
import vm from 'node:vm';
import { main, renderHarness, embeddingComparison } from '../../scripts/smoke/multimodal_embeddings.mjs';

test('embedding qualification requires local inputs before any download or browser launch', async () => {
  await assert.rejects(main([]), /Local model, projector, reference, baseline and artifacts paths are required/);
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


test('BF16 comparison gates cosine and relative L2 while max abs is informational',()=>{
  assert.equal(embeddingComparison([1000,0],[1001,0]).pass,true);
  assert.equal(embeddingComparison([1.02,0],[1,0]).pass,false);
  const angle=Math.acos(0.9998)-1e-8;
  assert.equal(embeddingComparison([Math.cos(angle),Math.sin(angle)],[1,0]).pass,true);
  assert.equal(embeddingComparison([Math.cos(angle+2e-8),Math.sin(angle+2e-8)],[1,0]).pass,false);
  // At the closest point on a ray, relative L2 can pass while cosine just
  // misses its own bound. This makes both criteria independently observable.
  const cosine=0.99979999;
  const independent=embeddingComparison([cosine*cosine,cosine*Math.sqrt(1-cosine*cosine)],[1,0]);
  assert.ok(independent.relativeL2<0.02);
  assert.equal(independent.pass,false);
  assert.equal(embeddingComparison([1,0],[1,0]).relativeL2,0);
  assert.throws(()=>embeddingComparison([NaN],[1]),/Invalid/);
});

test('qualification requires exact worker and pre-task string parity',()=>{
  const page=renderHarness('wasm64');
  for(const guard of ["directWorker[name]===0", "directWorkerRaw[name]===0", "row.stringBaseline.normalized===0 && row.stringBaseline.raw===0"])
    assert.ok(page.includes(guard));
});


// Execute the actual rendered guard rather than checking for its expression:
// an expression without check() must fail this regression too.
test('rendered string-baseline guard rejects normalized and raw drift independently', () => {
  for (const mode of ['wasm32', 'wasm64']) {
    const page = renderHarness(mode);
    const guard = page.split('\n').find((line) => line.includes("'pre-task string-path parity'"));
    // Match the expression line as well when a mutation drops the check wrapper.
    const expressionLine = guard || page.split('\n').find((line) => line.includes('row.stringBaseline.normalized===0'));
    assert.ok(expressionLine);
    for (const [normalized, raw, rejects] of [[0, 0, false], [1e-12, 0, true], [0, 1e-12, true]]) {
      const execute = () => vm.runInNewContext(expressionLine, {
        row: { stringBaseline: { normalized, raw } },
        check(condition, message) { if (!condition) throw new Error(message); },
      });
      if (rejects) assert.throws(execute, /pre-task string-path parity/);
      else assert.doesNotThrow(execute);
    }
  }
});
