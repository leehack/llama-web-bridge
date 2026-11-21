import assert from 'node:assert/strict';
import test from 'node:test';

import { launchArgs, parseArgs, renderHarness, renderTrackedCore } from '../../scripts/smoke/core_release.mjs';

test('core release smoke defaults to both cases on the threaded wasm32 core', () => {
  const args = parseArgs([]);
  assert.equal(args.threading, 'default');
  assert.equal(args.memoryMode, 'wasm32');
  assert.equal(args.case, null);
});

test('core release smoke accepts single-thread, wasm64 and case selections', () => {
  const args = parseArgs(['--dist-dir', 'dist', '--threading', 'single', '--memory-mode', 'wasm64', '--case', 'abort']);
  assert.equal(args.distDir, 'dist');
  assert.equal(args.threading, 'single');
  assert.equal(args.memoryMode, 'wasm64');
  assert.equal(args.case, 'abort');
  assert.throws(() => parseArgs(['--case', 'invalid']), /invalid choice/);
});

test('core release smoke accepts a local decision encoder and head', () => {
  const args = parseArgs(['--case', 'decision-abort', '--decision-model-path', 'encoder.gguf',
    '--decision-head-path', 'head.safetensors']);
  assert.equal(args.case, 'decision-abort');
  assert.equal(args.decisionModelPath, 'encoder.gguf');
  assert.equal(args.decisionHeadPath, 'head.safetensors');
});

test('the tracked core holds each core only through a WeakRef', () => {
  const source = renderTrackedCore('llama_webgpu_core_mem64');
  assert.match(source, /import createCore from '\/llama_webgpu_core_mem64\.js'/);
  assert.match(source, /__trackedCores\.push\(new WeakRef\(core\)\)/);
});

test('Chromium exposes gc() and shared memory only to the threaded core', () => {
  assert.ok(launchArgs('linux').includes('--js-flags=--expose-gc'));
  assert.ok(launchArgs('linux').some((arg) => arg.includes('SharedArrayBuffer')));
  assert.ok(!launchArgs('linux', { single: true }).some((arg) => arg.includes('SharedArrayBuffer')));
});

test('the harness page embeds its scenario settings', () => {
  const page = renderHarness({ scenario: 'dispose', single: false });
  assert.match(page, /await runBrowserCase\(\{"scenario":"dispose","single":false\}\)/);
});
