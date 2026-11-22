import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';

import { main, parseArgs, renderHarness } from '../../scripts/smoke/resume_integrity.mjs';
import { cachedModelName, isFile, sha256File, withServer, withTempDir } from '../../scripts/smoke/support.mjs';

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

test('resume smoke downloads and verifies the state fixture with an empty cache', async () => {
  await withTempDir('resume-integrity-cache-test-', async (root) => {
    const fixture = path.join(root, 'model.gguf');
    const cache = path.join(root, 'cache');
    const bytes = 'bad fixture';
    await writeFile(fixture, bytes);
    await withServer(root, async (base) => {
      const modelUrl = new URL('/model.gguf', base).href;
      const cached = path.join(cache, cachedModelName(modelUrl));
      const previous = process.env;
      process.env = { ...previous, LLAMA_WEBGPU_SMOKE_MODEL_URL: modelUrl };
      delete process.env.LLAMA_WEBGPU_SMOKE_MODEL_PATH;
      const args = ['--dist-dir', root, '--baseline-dist-dir', root, '--model-cache-dir', cache];
      try {
        await assert.rejects(main([...args, '--model-sha256', '0'.repeat(64)]), /model checksum mismatch/);
        assert.equal(isFile(cached), false);
        await assert.rejects(main([...args, '--model-sha256', await sha256File(fixture)]), {
          code: 'ERR_ASSERTION', expected: 'GGUF',
        });
        assert.equal(await readFile(cached, 'utf8'), bytes);
      } finally {
        process.env = previous;
      }
    });
  });
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
