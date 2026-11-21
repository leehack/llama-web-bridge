import assert from 'node:assert/strict';
import { test } from 'node:test';

import { parseArgs, renderHarness, verifyScenario } from '../../scripts/smoke/bundler_packaging.mjs';

test('parses the selected bundler suite, browser and local dist', () => {
  const args = parseArgs(['--suite', 'webpack', '--browser', 'firefox', '--dist-dir', '/test/dist']);
  assert.equal(args.suite, 'webpack');
  assert.equal(args.browser, 'firefox');
  assert.equal(args.distDir, '/test/dist');
  assert.throws(() => parseArgs(['--suite', 'unsupported']), /invalid choice/);
});

test('bundlers use a bare import and the CDN harness imports the hosted distribution', () => {
  assert.match(renderHarness('llama-web-bridge'), /from "llama-web-bridge"/);
  assert.match(renderHarness('http://127.0.0.1:8123/npm/dist/llama_webgpu_bridge.js'),
    /from "http:\/\/127\.0\.0\.1:8123\/npm\/dist\/llama_webgpu_bridge\.js"/);
});

test('hashed threaded cases require a hashed Wasm and the same hashed core for pthreads', () => {
  const origin = 'http://127.0.0.1:8123';
  const scenario = {
    isolated: true,
    execution: 'worker',
    memory: 'wasm64',
    expectedBase: origin,
    hashed: true,
    pthreadStart: 'direct',
    workerStart: 'direct',
    workerUrl: `${origin}/assets/llama_webgpu_bridge_worker-PS8a32.js`,
  };
  const result = {
    isolated: true, tokens: 4, execution: 'worker', memory: 'wasm64', threads: '4',
    notes: 'worker_start:direct;pthread_start:direct',
  };
  const requests = [
    { url: `${scenario.workerUrl}?__llamadartBridgeWorker=1`, dest: 'worker' },
    { url: `${origin}/assets/llama_webgpu_core_mem64-BHD753.js`, dest: 'script' },
    { url: `${origin}/assets/llama_webgpu_core_mem64-BHD753.js`, dest: 'worker' },
    { url: `${origin}/assets/llama_webgpu_core_mem64-2BD37Z.wasm`, dest: '' },
  ];
  verifyScenario(scenario, result, requests, []);
  assert.throws(() => verifyScenario(scenario, result, requests.filter((entry) => entry !== requests[2]), []),
    /hashed core URL for pthread/);
  assert.throws(() => verifyScenario(scenario, result, requests.slice(1), []), /worker entry/);
  assert.throws(() => verifyScenario({ ...scenario, workerUrl: `${scenario.workerUrl}?explicit=1` }, result, requests, []),
    /worker entry/);
  assert.throws(() => verifyScenario({ ...scenario, wasmUrl: `${requests[3].url}?explicit=1` }, result, requests, []),
    /explicit Wasm URL/);
  assert.throws(() => verifyScenario(scenario, result, requests, ['404 core']), /404 core/);
  assert.throws(() => verifyScenario(scenario, { ...result, tokens: 0 }, requests, []), /completion tokens/);
});
