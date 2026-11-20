import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  coreFileStem, coreWithoutWorkerHook, parseArgs, renderHarness, verifyScenario,
} from '../../scripts/smoke/cross_origin_assets.mjs';

function result({ single = false, direct = false, notes = [] } = {}) {
  return {
    isolated: !single,
    sharedMemory: single ? 'undefined' : 'function',
    violations: [],
    completionTokens: 6,
    metadata: {
      'llamadart.webgpu.execution': direct ? 'main-thread' : 'worker',
      'llamadart.webgpu.n_threads': single ? '1' : '4',
      'llamadart.webgpu.core_variant': 'wasm32',
      'llamadart.webgpu.runtime_notes': [...(single ? ['core_single_thread'] : []), ...notes].join(';'),
    },
  };
}

test('the smoke selects a memory mode and accepts one case or the full matrix', () => {
  const options = parseArgs(['--dist-dir', '/tmp/build', '--case', 'old-core', '--memory-mode', 'wasm64']);
  assert.equal(options.distDir, '/tmp/build');
  assert.equal(options.case, 'old-core');
  assert.equal(options.memoryMode, 'wasm64');
  assert.equal(parseArgs([]).case, null);
  assert.throws(() => parseArgs(['--case', 'unknown']));
  assert.equal(coreFileStem('wasm64', true), 'llama_webgpu_core_mem64_st');
});

test('the harness serves bridge and core from the CDN and monitors security violations', () => {
  const cdn = 'http://127.0.0.1:3001';
  const single = renderHarness('plain-st-direct', cdn, 'wasm32');
  assert.match(single, /"disableWorker":true/);
  assert.match(single, /"allowSingleThreadCore":true/);
  assert.ok(single.includes(cdn + '/llama_webgpu_core_st.js'));
  assert.match(single, /securitypolicyviolation/);
  assert.match(single, /nGpuLayers: 0/);
  assert.match(renderHarness('plain-st', cdn, 'wasm32'), /"workerUrl":"http:[^"]+\/llama_webgpu_bridge\.js"/);
  assert.match(renderHarness('old-core', cdn, 'wasm64'), /\/old\/llama_webgpu_core_mem64\.js/);
});

test('the legacy control removes the generated hook and keeps the direct worker branch', () => {
  const direct = 'worker=new Worker(new URL("llama_webgpu_core.js",import.meta.url),{type:"module"});';
  const compiled = 'var worker;if(Module["mainScriptUrlOrBlob"]){worker=new Worker(blob)}else ' + direct;
  assert.equal(coreWithoutWorkerHook(compiled), 'var worker;' + direct);
  assert.throws(() => coreWithoutWorkerHook(direct), /could not find the generated pthread hook/);
});

test('the direct single-thread acceptance rejects an extra worker or CSP violation', () => {
  const direct = result({ single: true, direct: true });
  verifyScenario('plain-st-direct', 'wasm32', direct, [], 1);
  verifyScenario('csp-st-direct', 'wasm32', direct, [], 1);
  assert.throws(() => verifyScenario('plain-st-direct', 'wasm32', direct, ['blob:x'], 1), /worker count/);
  const blocked = { ...direct, violations: [{ directive: 'worker-src', blocked: 'blob' }] };
  assert.throws(() => verifyScenario('csp-st-direct', 'wasm32', blocked, [], 1));
});

test('the threaded acceptance checks the requested topology and one Wasm load', () => {
  const worker = result({ notes: ['worker_start:blob', 'pthread_start:blob'] });
  const workers = Array(7).fill('blob:http://example.test/worker');
  verifyScenario('isolated-threaded', 'wasm32', worker, workers, 1);
  assert.throws(() => verifyScenario('isolated-threaded', 'wasm32', worker, workers, 2), /Wasm requests/);
  assert.throws(() => verifyScenario('isolated-threaded', 'wasm32', worker, ['http://app.test/worker', ...workers], 1));
});

test('CSP acceptance requires a prompt typed pthread rejection', () => {
  const blocked = {
    isolated: true,
    sharedMemory: 'function',
    violations: [{ directive: 'worker-src', blocked: 'blob' }],
    error: { name: 'BlobWorkerBlockedError', code: 'BLOB_WORKER_BLOCKED', reason: 'csp', target: 'pthread', typed: true, elapsedMs: 50 },
  };
  verifyScenario('csp-threaded', 'wasm32', blocked, [], 0);
  const slow = { ...blocked, error: { ...blocked.error, elapsedMs: 2000 } };
  assert.throws(() => verifyScenario('csp-threaded', 'wasm32', slow, [], 0), /did not reject promptly/);
  const untyped = { ...blocked, error: { ...blocked.error, typed: false } };
  assert.throws(() => verifyScenario('csp-threaded', 'wasm32', untyped, [], 0));
});
