import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { test } from 'node:test';

import { stubWorkerStart, workerStartModules } from './worker_blob_start_fixture.mjs';

const coreUrl = 'https://cdn.test/core.js';
const proxyOptions = (moduleUrl) => ({ moduleUrl, config: {} });
const emitsMessage = (worker) => queueMicrotask(() => worker.onmessage({ data: 0 }));
const emitsError = (worker) => queueMicrotask(() => worker.onerror({}));

test('cross-origin screening uses the browsing origin and only HTTP(S) scripts', async (t) => {
  stubWorkerStart(t);
  const { isCrossOriginScriptUrl } = await workerStartModules('origin');
  for (const [url, expected] of [
    ['https://app.test/script.js', false],
    [coreUrl, true],
    ['data:text/javascript,', false],
    ['blob:https://cdn.test/id', false],
  ]) {
    assert.equal(isCrossOriginScriptUrl(url), expected, url);
  }
  globalThis.origin = 'null';
  assert.equal(isCrossOriginScriptUrl('https://app.test/script.js'), true);
  delete globalThis.origin;
  assert.equal(isCrossOriginScriptUrl(coreUrl), false);
});

test('pthread workers reuse a static module blob for the lifetime of the realm', async (t) => {
  const state = stubWorkerStart(t);
  const { pthreadWorkerBlobUrl } = await workerStartModules('pthread-blob');
  const url = pthreadWorkerBlobUrl(coreUrl);
  assert.equal(pthreadWorkerBlobUrl(coreUrl), url);
  assert.equal(state.blobs.length, 1);
  assert.equal(await state.blobs[0].text(), `import "${coreUrl}";\n`);
  assert.deepEqual(state.revoked, []);
});

for (const [outcome, count, signal] of [
  [null, 2, emitsMessage],
  ['unavailable', 1, emitsError],
  ['script', 2, (worker, count) => (count === 1 ? emitsMessage : emitsError)(worker)],
  ['csp', 1, (worker) => queueMicrotask(() => { worker.onerror({}); })],
  ['constructor', 1, () => { throw new DOMException('Worker blocked', 'SecurityError'); }],
]) {
  test(`probe classifies ${outcome ?? 'allowed'} and cleans up`, async (t) => {
    const state = stubWorkerStart(t, {
      construct(worker, count) {
        signal(worker, count);
        if (outcome === 'csp') queueMicrotask(() => state.violation());
      },
    });
    const { probeBlobWorker } = await workerStartModules(`probe-${outcome}`);
    const expected = outcome === 'constructor' ? 'unavailable' : outcome;
    assert.equal(await probeBlobWorker(coreUrl), expected);
    assert.equal(await probeBlobWorker(coreUrl), expected);
    assert.equal(state.workers.length, count);
    assert.equal(state.revoked.length, count);
    assert.equal(state.listeners.size, 0);
    if (outcome !== 'constructor') assert.ok(state.workers.every((worker) => worker.terminated));
  });
}

for (const blocked of [false, true]) {
  test(`descriptor gates startup and remains non-enumerable (${blocked ? 'blocked' : 'allowed'})`, async (t) => {
    const state = stubWorkerStart(t, { construct: blocked ? emitsError : emitsMessage });
    const start = await workerStartModules(`descriptor-${blocked}`);
    let used = 0;
    const options = Object.defineProperties({}, start.pthreadScriptProperties(10_000, () => { used++; }));
    options[Symbol.for('llamadart.pthreadScriptUrl')](coreUrl);
    assert.deepEqual(Object.assign({}, options), {});
    assert.deepEqual({ ...options }, {});
    assert.equal(state.workers.length, 0);
    assert.throws(() => options.mainScriptUrlOrBlob, start.PthreadStartProbeRequired);
    assert.equal(state.workers.length, 1);
    assert.equal(used, 0);
    await start.probeBlobWorker(coreUrl);
    if (blocked) {
      assert.throws(() => options.mainScriptUrlOrBlob, {
        name: 'BlobWorkerBlockedError', code: 'BLOB_WORKER_BLOCKED', target: 'pthread', reason: 'unavailable',
      });
      assert.equal(used, 0);
    } else {
      const script = options.mainScriptUrlOrBlob;
      assert.match(script, /^blob:/);
      assert.equal(options.mainScriptUrlOrBlob, script);
      assert.equal(used, 2);
    }
  });
}

test('threaded factory retry is specific to the internal signal and happens once', async (t) => {
  const state = stubWorkerStart(t, { construct: emitsMessage });
  const start = await workerStartModules('retry');
  const { withPthreadStartProbe, PthreadStartProbeRequired } = start;
  assert.equal(await withPthreadStartProbe(() => 42), 42);
  const other = new Error('factory failure');
  await assert.rejects(withPthreadStartProbe(() => { throw other; }), (error) => error === other);
  assert.equal(state.workers.length, 0);
  let calls = 0;
  assert.equal(await withPthreadStartProbe(() => {
    if (++calls === 1) throw new PthreadStartProbeRequired(start.probeBlobWorker(coreUrl));
    assert.equal(start.blobWorkerProbeResult(coreUrl), null);
    return 42;
  }), 42);
  assert.equal(calls, 2);
  calls = 0;
  await assert.rejects(withPthreadStartProbe(() => {
    calls++;
    throw new PthreadStartProbeRequired(start.probeBlobWorker(coreUrl));
  }), PthreadStartProbeRequired);
  assert.equal(calls, 2);
});

test('the core worker URL uses the same resolution as the module import', async (t) => {
  const loaderUrl = new URL('../../js/src/internal/core_loader.ts', import.meta.url).href;
  const mappedUrl = 'https://cdn.test/mapped-core.js';
  const hooks = registerHooks({
    resolve(specifier, context, nextResolve) {
      if (context.parentURL === loaderUrl && specifier === 'mapped-core') {
        return { url: mappedUrl, shortCircuit: true };
      }
      return nextResolve(specifier, context);
    },
  });
  t.after(() => hooks.deregister());
  const { resolveCoreModuleUrl } = await import(loaderUrl);
  assert.equal(resolveCoreModuleUrl('data:text/javascript,'), 'data:text/javascript,');
  assert.equal(resolveCoreModuleUrl('./core.js'), new URL('./core.js', loaderUrl).href);
  assert.equal(resolveCoreModuleUrl('unresolvable-core'), '');
  assert.equal(resolveCoreModuleUrl('mapped-core'), mappedUrl);
});

test('cross-origin worker entry bypasses direct construction and resolves the static import', async (t) => {
  const state = stubWorkerStart(t);
  const { BridgeWorkerProxy } = await workerStartModules('proxy-cross-origin');
  const proxy = new BridgeWorkerProxy(proxyOptions('//cdn.test/llama_webgpu_bridge_worker.js'));
  assert.equal(state.workers.length, 1);
  const [worker] = state.workers;
  assert.match(worker.url, /^blob:/);
  assert.deepEqual(worker.options, { type: 'module' });
  assert.match(await state.blobs[0].text(), /^import \* as workerModule from "https:\/\/cdn\.test\/llama_webgpu_bridge_worker\.js";/);
  assert.equal(worker.posted[0].config.workerStart, 'blob');
  worker.onmessage({ data: { type: 'ready' } });
  assert.equal(state.listeners.size, 0);
  assert.deepEqual(state.revoked, []);
  await proxy.dispose();
  assert.deepEqual(state.revoked, [worker.url]);
});

test('same-origin worker starts directly and keeps the existing startup error', async (t) => {
  const state = stubWorkerStart(t);
  const { BridgeWorkerProxy } = await workerStartModules('proxy-direct');
  const proxy = new BridgeWorkerProxy(proxyOptions('llama_webgpu_bridge_worker.js'));
  const [worker] = state.workers;
  assert.equal(worker.url, 'https://app.test/page/llama_webgpu_bridge_worker.js');
  assert.equal(state.blobs.length, 0);
  assert.equal(worker.posted[0].config.workerStart, 'direct');
  worker.onerror({});
  await assert.rejects(proxy._ready, { name: 'Error', message: 'Bridge worker crashed', llamadartWorkerCrash: true });
  await proxy.dispose();
});

for (const order of ['violation-first', 'error-first']) {
  test(`proxy recognizes the real CSP violation without probing (${order})`, { timeout: 1500 }, async (t) => {
    const state = stubWorkerStart(t, {
      construct(worker, count) {
        if (count === 2) queueMicrotask(() => { state.violation(); worker.onerror({}); });
      },
    });
    const { BridgeWorkerProxy, blobWorkerProbeResult } = await workerStartModules(order);
    const proxy = new BridgeWorkerProxy(proxyOptions('https://cdn.test/llama_webgpu_bridge_worker.js'));
    if (order === 'violation-first') state.violation();
    state.workers[0].onerror({});
    if (order === 'error-first') queueMicrotask(() => state.violation());
    await assert.rejects(proxy._ready, {
      name: 'BlobWorkerBlockedError', code: 'BLOB_WORKER_BLOCKED', target: 'bridge_worker', reason: 'csp',
      llamadartWorkerCrash: true,
    });
    assert.equal(state.workers.length, 1);
    assert.equal(blobWorkerProbeResult(), 'csp');
    await proxy.dispose();
  });
}

for (const reason of ['unavailable', 'script']) {
  test(`proxy uses one diagnostic probe for ${reason}`, { timeout: 1500 }, async (t) => {
    const state = stubWorkerStart(t, {
      construct(worker, count) {
        if (count === 2) (reason === 'script' ? emitsMessage : emitsError)(worker);
      },
    });
    const { BridgeWorkerProxy } = await workerStartModules(reason);
    const proxy = new BridgeWorkerProxy(proxyOptions('https://cdn.test/llama_webgpu_bridge_worker.js'));
    state.workers[0].onerror({});
    await assert.rejects(proxy._ready, { name: 'BlobWorkerBlockedError', target: 'bridge_worker', reason });
    assert.equal(state.workers.length, 2);
    await proxy.dispose();
  });
}

test('disposing a worker that never became ready releases the CSP listener', async (t) => {
  const state = stubWorkerStart(t);
  const { BridgeWorkerProxy } = await workerStartModules('dispose-before-ready');
  const proxy = new BridgeWorkerProxy(proxyOptions('https://cdn.test/llama_webgpu_bridge_worker.js'));
  assert.equal(state.listeners.size, 1);
  await proxy.dispose();
  assert.equal(state.listeners.size, 0);
  assert.equal(state.workers[0].terminated, true);
});

test('a slow failed script import never grants permission to start pthreads', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const state = stubWorkerStart(t, {
    construct(worker, count) {
      if (count === 1) emitsMessage(worker);
    },
  });
  const { probeBlobWorker, blobWorkerProbeResult } = await workerStartModules('slow-import');
  assert.equal(await probeBlobWorker(), null);
  const probe = probeBlobWorker(coreUrl, 10_000);
  await Promise.resolve();
  t.mock.timers.tick(5500);
  await Promise.resolve();
  assert.equal(blobWorkerProbeResult(coreUrl), undefined);
  assert.equal(state.workers[1].terminated, false);
  state.workers[1].onerror({});
  t.mock.timers.tick(0);
  assert.equal(await probe, 'script');
  assert.equal(state.workers[1].terminated, true);
});

test('a shared pending probe respects this pthread initialization budget', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const state = stubWorkerStart(t, {
    construct(worker, count) {
      if (count === 1) emitsMessage(worker);
    },
  });
  const start = await workerStartModules('shared-probe');
  assert.equal(await start.probeBlobWorker(), null);
  const longerProbe = start.probeBlobWorker(coreUrl, 20_000);
  const options = Object.defineProperties({}, start.pthreadScriptProperties(10_000, assert.fail));
  options[Symbol.for('llamadart.pthreadScriptUrl')](coreUrl);
  let factoryCalls = 0;
  const failure = assert.rejects(start.withPthreadStartProbe(() => {
    factoryCalls++;
    return options.mainScriptUrlOrBlob;
  }), { name: 'BlobWorkerBlockedError', target: 'pthread', reason: 'script' });
  await Promise.resolve();
  t.mock.timers.tick(10_000);
  await failure;
  assert.equal(factoryCalls, 1);
  assert.equal(state.workers.length, 2);
  state.workers[1].onerror({});
  t.mock.timers.tick(0);
  assert.equal(await longerProbe, 'script');
});
