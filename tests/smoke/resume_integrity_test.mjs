import assert from 'node:assert/strict';
import test from 'node:test';

import { parseArgs, renderHarness } from '../../scripts/smoke/resume_integrity.mjs';

test('resume smoke defaults to threaded and accepts a pinned single-thread staging run', () => {
  assert.equal(parseArgs([]).threading, 'default');
  const args = parseArgs([
    '--dist-dir', 'dist', '--baseline-dist-dir', 'main-dist', '--threading', 'single',
    '--model-path', 'model.gguf', '--model-sha256', 'sha', '--js-backed-model-staging',
  ]);
  assert.equal(args.distDir, 'dist');
  assert.equal(args.baselineDistDir, 'main-dist');
  assert.equal(args.threading, 'single');
  assert.equal(args.modelPath, 'model.gguf');
  assert.equal(args.modelSha256, 'sha');
  assert.equal(args.jsBackedModelStaging, true);
  assert.throws(() => parseArgs(['--threading', 'invalid']), /invalid choice/);
});

test('resume page renders the requested core, execution mode, staging and model origin', () => {
  const modelUrl = 'http://127.0.0.1:41000/model.gguf';
  const direct = renderHarness(
    { threading: 'default', jsBackedModelStaging: false },
    { modelUrl, worker: false, recover: true },
  );
  assert.match(direct, /"disableWorker":true,"allowSingleThreadCore":false,"jsBackedModelStaging":false/);
  assert.match(direct, /"wasmUrl":"\/llama_webgpu_core\.wasm"/);
  assert.ok(direct.includes(`loadModelFromUrl(${JSON.stringify(modelUrl)}`));
  assert.match(direct, /if \(true\) originalCore = await bridge\._runtime\._ensureCore\(\)/);
  const worker = renderHarness(
    { threading: 'single', jsBackedModelStaging: true },
    { modelUrl, worker: true, recover: false },
  );
  assert.match(worker, /"disableWorker":false,"allowSingleThreadCore":true,"jsBackedModelStaging":true/);
  assert.match(worker, /"wasmUrl":"\/llama_webgpu_core_st\.wasm"/);
  assert.match(worker, /if \(false\) originalCore = await bridge\._runtime\._ensureCore\(\)/);
});
