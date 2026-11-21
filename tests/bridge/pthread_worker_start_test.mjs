// New behaviour: the pthread Worker constructor the runtime hands to a threaded core.
import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';
import { test } from 'node:test';
import { BlobWorkerBlockedError, PTHREAD_WORKER_HOOK, createPthreadWorkerStart } from '../../js/src/internal/worker_start.ts';

function stub(t, { construct } = {}) {
  // `checks` are the import-free blob workers that tell "script" from "unavailable".
  const state = { workers: [], checks: [], check: 'message', blobs: [], revoked: [], events: new EventTarget(), listeners: 0 };
  const globals = {
    origin: 'https://app.test',
    addEventListener(type, listener) { state.listeners += 1; state.events.addEventListener(type, listener); },
    removeEventListener(type, listener) { state.listeners -= 1; state.events.removeEventListener(type, listener); },
    Worker: class extends EventTarget {
      constructor(url, options) {
        super();
        Object.assign(this, { url, options, terminated: false });
        if (options?.name !== 'em-pthread') {
          state.checks.push(this);
          if (state.check === 'throw') throw new Error('refused');
          if (state.check !== 'silent') queueMicrotask(() => this.dispatchEvent(new Event(state.check)));
          return;
        }
        state.workers.push(this);
        construct?.(this, state.workers.length);
      }
      terminate() { this.terminated = true; }
    },
  };
  const originals = Object.fromEntries(Object.keys(globals).map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
  t.after(() => {
    for (const [key, descriptor] of Object.entries(originals)) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
    }
  });
  t.mock.method(URL, 'createObjectURL', (blob) => { state.blobs.push(blob); return `blob:https://app.test/${state.blobs.length}`; });
  t.mock.method(URL, 'revokeObjectURL', (url) => state.revoked.push(url));
  state.violation = (overrides = {}) => state.events.dispatchEvent(Object.assign(new Event('securitypolicyviolation'), {
    blockedURI: 'blob', effectiveDirective: 'worker-src', disposition: 'enforce', ...overrides,
  }));
  return state;
}
const OPTIONS = { type: 'module', name: 'em-pthread' };

test('the hook key is the registered symbol the core pre-js reads', () => {
  assert.equal(PTHREAD_WORKER_HOOK, Symbol.for('llamadart.pthreadWorker'));
});

for (const url of [
  new URL('https://app.test/assets/llama_webgpu_core.js'),
  'https://app.test/assets/llama_webgpu_core.js',
  new URL('blob:https://app.test/1234'),
  new URL('data:text/javascript,0'),
  './relative.js',
]) {
  test(`a script the page may start itself is constructed exactly as the core asked: ${url}`, (t) => {
    const state = stub(t);
    const start = createPthreadWorkerStart();
    const worker = new start.Worker(url, OPTIONS);
    assert.equal(state.workers.length, 1);
    assert.equal(worker, state.workers[0]);
    assert.equal(worker.url, url);            // same value, same identity
    assert.equal(worker.options, OPTIONS);
    assert.equal(state.blobs.length, 0);
    assert.equal(state.listeners, 0);
    assert.equal(start.mode(), 'direct');
    start.settle();
  });
}

test('a cross-origin script starts from one cached module blob that statically imports it', async (t) => {
  const state = stub(t);
  const start = createPthreadWorkerStart();
  const url = new URL('https://cdn.test/v/llama_webgpu_core.js?x=1');
  const first = new start.Worker(url, OPTIONS);
  const second = new start.Worker(url, OPTIONS);
  assert.equal(state.blobs.length, 1);
  assert.equal(await state.blobs[0].text(), 'import "https://cdn.test/v/llama_webgpu_core.js?x=1";\n');
  assert.equal(state.blobs[0].type, 'text/javascript');
  assert.equal(first.url, 'blob:https://app.test/1');
  assert.equal(second.url, first.url);
  assert.equal(first.options, OPTIONS);
  assert.equal(start.mode(), 'blob');
  start.settle();
  assert.equal(state.listeners, 0);
  assert.deepEqual(state.revoked, []);
  // A later core instance reuses the same blob URL.
  const again = createPthreadWorkerStart();
  assert.equal(new again.Worker(url, OPTIONS).url, first.url);
  again.settle();
});

test('a blob worker that errors before the core is ready rejects with the typed error and stops the pool', async (t) => {
  const state = stub(t);
  const start = createPthreadWorkerStart();
  const pool = [1, 2, 3, 4].map(() => new start.Worker(new URL('https://cdn.test/a/core.js'), OPTIONS));
  pool[2].dispatchEvent(new Event('error'));
  pool[3].dispatchEvent(new Event('error'));
  await assert.rejects(start.failed, (error) => error instanceof BlobWorkerBlockedError
    && error.code === 'BLOB_WORKER_BLOCKED' && error.reason === 'script' && error.target === 'pthread' && error.name === 'BlobWorkerBlockedError');
  assert.ok(pool.every((worker) => worker.terminated));
  assert.equal(state.workers.length, 4);
  // One import-free blob worker started, so the script is what failed.
  assert.equal(state.checks.length, 1);
  assert.equal(await state.blobs[1].text(), 'postMessage(0);\n');
  assert.deepEqual(state.checks[0].options, { type: 'module' });
  assert.ok(state.checks[0].terminated);
  assert.deepEqual(state.revoked, [state.checks[0].url]);
  const original = new Error('core init failed');
  assert.equal(await start.failure(original), await start.failed.catch((error) => error));
  start.settle();
  assert.equal(state.listeners, 0);
});

test('a refused start with a worker-src blob violation is "csp" and starts no further worker', async (t) => {
  for (const violationFirst of [true, false]) {
    const state = stub(t);
    const start = createPthreadWorkerStart();
    const worker = new start.Worker(new URL('https://cdn.test/b/core.js'), OPTIONS);
    if (violationFirst) state.violation();
    worker.dispatchEvent(new Event('error'));
    if (!violationFirst) state.violation();
    await assert.rejects(start.failed, (error) => error.reason === 'csp');
    assert.equal(state.workers.length, 1);
    assert.equal(state.checks.length, 0);
    start.settle();
    assert.equal(state.listeners, 0);
  }
});

test('an unrelated violation does not make the reason "csp"', async (t) => {
  const state = stub(t);
  const start = createPthreadWorkerStart();
  const worker = new start.Worker(new URL('https://cdn.test/c/core.js'), OPTIONS);
  state.violation({ blockedURI: 'https://cdn.test/c/core.js' });
  state.violation({ effectiveDirective: 'script-src-elem' });
  state.violation({ disposition: 'report' });
  worker.dispatchEvent(new Event('error'));
  await assert.rejects(start.failed, (error) => error.reason === 'script');
  assert.equal(state.workers.length, 1);
  start.settle();
});

test('in an opaque origin a refused start without a violation is "unavailable"', async (t) => {
  const state = stub(t);
  globalThis.origin = 'null';
  const start = createPthreadWorkerStart();
  const worker = new start.Worker(new URL('https://cdn.test/o/core.js'), OPTIONS);
  assert.equal(worker.url, 'blob:https://app.test/1');
  worker.dispatchEvent(new Event('error'));
  await assert.rejects(start.failed, (error) => error.reason === 'unavailable');
  assert.equal(state.workers.length, 1);
  assert.equal(state.checks.length, 0);
  start.settle();
});

// Firefox dispatches no violation inside a worker whose response carries the policy.
for (const check of ['error', 'throw', 'silent']) {
  test(`a refused start where no blob worker starts at all is "unavailable": check ${check}`, async (t) => {
    const state = stub(t);
    state.check = check;
    if (check === 'silent') t.mock.timers.enable({ apis: ['setTimeout'] });
    const start = createPthreadWorkerStart();
    const worker = new start.Worker(new URL('https://cdn.test/u/core.js'), OPTIONS);
    worker.dispatchEvent(new Event('error'));
    if (check === 'silent') {
      t.mock.timers.tick(0);
      await Promise.resolve();
      t.mock.timers.tick(4999);
      let settled = false;
      start.failed.catch(() => { settled = true; });
      await new Promise((resolve) => setImmediate(resolve));
      assert.equal(settled, false);
      t.mock.timers.tick(1);
    }
    await assert.rejects(start.failed, (error) => error.reason === 'unavailable');
    assert.equal(state.checks.length, 1);
    assert.deepEqual(state.revoked, [state.checks[0].url]);
    if (check !== 'throw') assert.ok(state.checks[0].terminated);
    start.settle();
  });
}

test('a blob constructor that throws is "unavailable", and the core still sees the browser error first', async (t) => {
  const refusal = new TypeError('This document requires TrustedScriptURL assignment');
  const state = stub(t, { construct() { throw refusal; } });
  const start = createPthreadWorkerStart();
  assert.throws(() => new start.Worker(new URL('https://cdn.test/d/core.js'), OPTIONS), (error) => error === refusal);
  const reported = await start.failure(refusal);
  assert.ok(reported instanceof BlobWorkerBlockedError);
  assert.equal(reported.reason, 'unavailable');
  assert.equal(state.workers.length, 1);
  assert.equal(state.checks.length, 0);
  start.settle();
});

test('with no refusal, failure() returns the caller error itself', async (t) => {
  stub(t);
  const start = createPthreadWorkerStart();
  new start.Worker(new URL('https://cdn.test/e/core.js'), OPTIONS);
  const timeout = new Error('Bridge core init timeout (10000ms)');
  assert.equal(await start.failure(timeout), timeout);
  start.settle();
});

test('abandoning a classifier discards both its pending and late results', async (t) => {
  const state = stub(t);
  state.check = 'silent';
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const start = createPthreadWorkerStart();
  const worker = new start.Worker(new URL('https://cdn.test/abandoned/core.js'), OPTIONS);
  worker.dispatchEvent(new Event('error'));
  t.mock.timers.tick(1);
  const [classifier] = state.checks;
  assert.ok(classifier);
  start.settle();
  assert.ok(classifier.terminated);
  assert.deepEqual(state.revoked, [classifier.url]);

  classifier.dispatchEvent(new Event('message'));
  await Promise.resolve();
  const original = new Error('factory failed');
  assert.equal(start.failure(original), original);
});

test('after the core is ready, worker errors are the core\'s business and late workers still start from the blob', async (t) => {
  const state = stub(t);
  const start = createPthreadWorkerStart();
  const worker = new start.Worker(new URL('https://cdn.test/f/core.js'), OPTIONS);
  start.settle(true);
  worker.dispatchEvent(new Event('error'));
  const late = new start.Worker(new URL('https://cdn.test/f/core.js'), OPTIONS);
  late.dispatchEvent(new Event('error'));
  assert.equal(late.url, worker.url);
  assert.equal(start.mode(), 'blob');
  await delay(20);
  assert.equal(state.workers.length, 2);
  assert.equal(state.checks.length, 0);
  assert.ok(!worker.terminated && !late.terminated);
  assert.equal(state.listeners, 0);
  let settled = false;
  start.failed.catch(() => { settled = true; });
  await delay(5);
  assert.equal(settled, false);
});

test('an abandoned start releases its pool and refuses late cross-origin workers', (t) => {
  const state = stub(t);
  const start = createPthreadWorkerStart();
  const url = new URL('https://cdn.test/abandoned-pool/core.js');
  const worker = new start.Worker(url, OPTIONS);
  start.settle(false);

  assert.ok(worker.terminated);
  assert.throws(() => new start.Worker(url, OPTIONS), /core initialization was abandoned/);
  assert.equal(state.workers.length, 1);
  assert.equal(state.blobs.length, 1);

  const sameOrigin = new URL('https://app.test/untouched.js');
  assert.equal(new start.Worker(sameOrigin, OPTIONS).url, sameOrigin);
});

test('two starts share nothing but the blob URL: one refusal does not touch the other', async (t) => {
  const state = stub(t);
  const a = createPthreadWorkerStart();
  const b = createPthreadWorkerStart();
  const wa = new a.Worker(new URL('https://cdn.test/g/core.js'), OPTIONS);
  const wb = new b.Worker(new URL('https://cdn.test/g/core.js'), OPTIONS);
  wa.dispatchEvent(new Event('error'));
  await assert.rejects(a.failed, BlobWorkerBlockedError);
  assert.ok(wa.terminated);
  assert.ok(!wb.terminated);
  const mine = new Error('mine');
  assert.equal(await b.failure(mine), mine);
  a.settle(); b.settle();
  // Nothing is remembered: a new start after a refusal tries again.
  const c = createPthreadWorkerStart();
  assert.equal(new c.Worker(new URL('https://cdn.test/g/core.js'), OPTIONS).url, wa.url);
  assert.equal(c.mode(), 'blob');
  c.settle();
  assert.equal(state.listeners, 0);
});
