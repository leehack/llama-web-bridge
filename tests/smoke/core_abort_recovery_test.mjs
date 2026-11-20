import assert from 'node:assert/strict';
import test from 'node:test';

import { parseArgs, renderHarness, renderInstrumentedCore } from '../../scripts/smoke/core_abort_recovery.mjs';

test('abort recovery smoke defaults to every runtime mode on the threaded wasm32 core', () => {
  const args = parseArgs([]);
  assert.equal(args.threading, 'default');
  assert.equal(args.memoryMode, 'wasm32');
  assert.equal(args.runtimeMode, 'all');
  assert.equal(args.case, null);
});

test('abort recovery smoke accepts pinned single-thread, baseline and case selections', () => {
  const args = parseArgs([
    '--dist-dir', 'dist', '--baseline-dist-dir', 'main-dist', '--threading', 'single',
    '--memory-mode', 'wasm64', '--runtime-mode', 'direct', '--case', 'older-iframe',
  ]);
  assert.equal(args.distDir, 'dist');
  assert.equal(args.baselineDistDir, 'main-dist');
  assert.equal(args.threading, 'single');
  assert.equal(args.memoryMode, 'wasm64');
  assert.equal(args.runtimeMode, 'direct');
  assert.equal(args.case, 'older-iframe');
  assert.throws(() => parseArgs(['--case', 'invalid']), /invalid choice/);
  assert.throws(() => parseArgs(['--runtime-mode', 'invalid']), /invalid choice/);
});

test('the instrumented core hides the exported RuntimeError only for the older-core case', () => {
  const current = renderInstrumentedCore('llama_webgpu_core_st');
  assert.match(current, /import createCore from '\/llama_webgpu_core_st\.js'/);
  assert.match(current, /\{"hideRuntimeError":false\}/);
  assert.match(renderInstrumentedCore('llama_webgpu_core', { hideRuntimeError: true }), /\{"hideRuntimeError":true\}/);
});

test('the harness page embeds its scenario settings', () => {
  const page = renderHarness({ scenario: 'next-token', worker: true, single: false });
  assert.match(page, /await runBrowserCase\(\{"scenario":"next-token","worker":true,"single":false\}\)/);
});
