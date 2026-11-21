import assert from 'node:assert/strict';
import { test } from 'node:test';
import { setImmediate as nextTurn } from 'node:timers/promises';

import * as download from '../../js/src/internal/download.ts';
import { createDirectBridge } from './bridge_operation_queue_fixtures.mjs';

const { openModelCacheParts, startModelCachePartStore } = download;

const MODEL_URL = 'https://models.test/model.gguf';
const CACHE_NAME = 'model-cache-parts-test';
const MANIFEST = 'https://llamadart-cache.invalid/model-parts/';
const PART = 'https://llamadart-cache.invalid/model-part/';
const SAFARI = 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.5 Mobile/15E148 Safari/604.1';

// Like the Cache API: put() reads the whole body before it commits an entry.
function installCaches() {
  const entries = new Map();
  const headers = new Map();
  const calls = { put: [], match: [], keys: 0, delete: [] };
  const hooks = { afterMatch: null, beforePut: null };
  const urlOf = (key) => (typeof key === 'string' ? key : key.url).split('#')[0];
  const cache = {
    async match(key) {
      const url = urlOf(key);
      calls.match.push(url);
      const stored = entries.get(url);
      const snapshot = stored ? new Response(stored.slice(), { headers: headers.get(url) }) : undefined;
      await hooks.afterMatch?.(url);
      return snapshot;
    },
    async put(key, response) {
      const url = urlOf(key);
      calls.put.push(url);
      await hooks.beforePut?.(url);
      const bytes = new Uint8Array(await response.arrayBuffer());
      entries.set(url, bytes);
      headers.set(url, new Headers(response.headers));
    },
    async keys(key, { ignoreSearch } = {}) {
      calls.keys += 1;
      const path = (url) => (ignoreSearch ? url.split('?')[0] : url);
      return [...entries.keys()]
        .filter((url) => !key || path(url) === path(urlOf(key)))
        .map((url) => new Request(url));
    },
    async delete(key) {
      const url = urlOf(key);
      calls.delete.push(url);
      headers.delete(url);
      return entries.delete(url);
    },
  };
  const original = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  Object.defineProperty(globalThis, 'caches', {
    configurable: true,
    value: { open: async (name) => (name === CACHE_NAME ? cache : assert.fail('unexpected cache ' + name)) },
  });
  return {
    cache,
    entries,
    headers,
    calls,
    hooks,
    urls: (prefix) => [...entries.keys()].filter((url) => url.startsWith(prefix)).sort(),
    manifest() {
      const url = [...entries.keys()].find((candidate) => candidate.startsWith(MANIFEST));
      return url ? headers.get(url) : null;
    },
    partBytes(record) {
      return [...entries.keys()]
        .filter((url) => url.startsWith(PART) && url.includes('record=' + record + '&'))
        .sort((a, b) => Number(new URL(a).searchParams.get('part')) - Number(new URL(b).searchParams.get('part')))
        .map((url) => [...entries.get(url)]);
    },
    restore() {
      if (original) Object.defineProperty(globalThis, 'caches', original);
      else delete globalThis.caches;
    },
  };
}

// navigator.locks with the one mode the store uses: an exclusive ifAvailable request.
function installLocks({ available = true, present = true } = {}) {
  const held = new Set();
  const pendingReleases = new Set();
  const requests = [];
  const locks = {
    async request(name, options, callback) {
      requests.push({ name, options });
      if (!available || held.has(name)) return callback(null);
      const released = Promise.withResolvers();
      held.add(name);
      pendingReleases.add(released.promise);
      try {
        return await callback({ name });
      } finally {
        held.delete(name);
        pendingReleases.delete(released.promise);
        released.resolve();
      }
    },
  };
  const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: present ? { locks } : {},
  });
  return {
    held,
    requests,
    whenIdle: () => Promise.all(pendingReleases),
    restore() {
      if (original) Object.defineProperty(globalThis, 'navigator', original);
      else delete globalThis.navigator;
    },
  };
}

// A response whose body the test feeds; `reads` counts the consumer's reads.
function controlledResponse(headers = {}) {
  const queue = [];
  const state = { cancelled: false, closed: false, reads: 0, error: null };
  let wake = null;
  const notify = () => {
    const resolve = wake;
    wake = null;
    resolve?.();
  };
  const stream = new ReadableStream({
    async pull(controller) {
      state.reads += 1;
      while (queue.length === 0 && !state.closed) {
        await new Promise((resolve) => {
          wake = resolve;
        });
      }
      if (state.cancelled) return;
      if (queue.length > 0) controller.enqueue(queue.shift());
      else if (state.error) throw state.error;
      else controller.close();
    },
    cancel() {
      state.cancelled = true;
      state.closed = true;
      notify();
    },
  }, { highWaterMark: 0 });
  const response = new Response(stream, { status: 200, statusText: 'OK', headers });
  Object.defineProperty(response, 'url', { value: MODEL_URL });
  return {
    response,
    state,
    push(bytes) {
      queue.push(Uint8Array.from(bytes));
      notify();
    },
    close() {
      state.closed = true;
      notify();
    },
    fail(error) {
      state.error = error;
      state.closed = true;
      notify();
    },
  };
}

const bodyOf = (bytes, chunk = 3, headers = {}) => new Response(new ReadableStream({
  start(controller) {
    for (let offset = 0; offset < bytes.length; offset += chunk) {
      controller.enqueue(Uint8Array.from(bytes.slice(offset, offset + chunk)));
    }
    controller.close();
  },
}), { headers });

const readAll = async (response) => [...new Uint8Array(await response.arrayBuffer())];
const TEN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

async function withFakes(options, run) {
  const caches = installCaches();
  const locks = installLocks(options);
  const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  if (options.uuid === false) {
    Object.defineProperty(globalThis, 'crypto', {
      configurable: true,
      value: { subtle: globalThis.crypto.subtle },
    });
  }
  try {
    await run(caches, locks);
  } finally {
    await locks.whenIdle();
    if (options.uuid === false) Object.defineProperty(globalThis, 'crypto', originalCrypto);
    locks.restore();
    caches.restore();
  }
}

async function withStoreReader(store, run) {
  const reader = store.response.body.getReader();
  const settled = store.stored.catch(() => {});
  try {
    await run(reader);
  } finally {
    await reader.cancel().catch(() => {});
    await settled;
  }
}

async function storeParts(caches, bytes = TEN, partBytes = 4, headers = { etag: '"v1"' }) {
  const store = startModelCachePartStore(caches.cache, MODEL_URL, bodyOf(bytes, 3, headers), CACHE_NAME, partBytes);
  const body = await readAll(store.response);
  await store.stored;
  return body;
}

function createRuntime(fetchModel, userAgent) {
  const { bridge, core } = createDirectBridge();
  const runtime = bridge._runtime;
  const written = new Map();
  core.FS.open = (path) => {
    written.set(path, []);
    return path;
  };
  core.FS.write = (path, bytes, offset, length) => {
    written.get(path).push(bytes.slice(offset, offset + length));
    return length;
  };
  const ccall = core.ccall.bind(core);
  const loads = [];
  core.ccall = (name, type, types, args = []) => {
    if (name === 'llamadart_webgpu_load_model') {
      loads.push(written.get(args[0]).flatMap((chunk) => [...chunk]));
      return 0;
    }
    if (name === 'llamadart_webgpu_free_model') return 0;
    if (name === 'llamadart_webgpu_get_context_size') return 128;
    return ccall(name, type, types, args);
  };
  runtime._modelBytes = 0;
  runtime._coreVariant = 'wasm32';
  runtime._probeBackends = async () => {};
  runtime._ensureCore = async () => core;
  runtime._coreSupportsPthreads = () => false;
  runtime._syncThreadPoolSizeHintFromCore = () => {};
  runtime._emitLogger = () => {};
  runtime._isWebKit = userAgent === SAFARI;
  const fetches = [];
  runtime._fetchWithTimeout = async (url, init) => {
    fetches.push(url);
    return fetchModel(url, init);
  };
  return { runtime, core, loads, fetches };
}

const loadOptions = (extra = {}) => ({
  nCtx: 128,
  nThreads: 1,
  nGpuLayers: 0,
  forceRemoteFetchBackend: false,
  cacheName: CACHE_NAME,
  ...extra,
});

// 40 bytes in 16-byte parts would need a 16 MiB buffer; the runtime cases use
// the production part size and a body smaller than one part.
const CASES = [
  ['a stored body becomes fixed-size parts under a manifest written last', () => withFakes({}, async (caches, locks) => {
    const source = controlledResponse({ etag: '"v1"', 'content-length': '10' });
    const store = startModelCachePartStore(caches.cache, MODEL_URL, source.response, CACHE_NAME, 4);
    assert.equal(store.response.url, MODEL_URL);
    assert.equal(store.response.status, 200);
    assert.equal(store.response.headers.get('etag'), '"v1"');
    await withStoreReader(store, async (reader) => {
      source.push([1, 2, 3]);
      source.push([4, 5, 6]);
      const received = [...(await reader.read()).value, ...(await reader.read()).value];
      assert.equal(caches.urls(PART).length, 1, 'the first full part is committed while the body is still open');
      assert.equal(caches.manifest(), null, 'no manifest before the body completes');
      source.push([7, 8, 9]);
      source.push([10]);
      source.close();
      received.push(...(await reader.read()).value, ...(await reader.read()).value);
      assert.equal((await reader.read()).done, true);
      assert.deepEqual(received, TEN);
      await store.stored;
    });

    const manifest = caches.manifest();
    const record = manifest.get('x-llamadart-cache-record');
    assert.match(record, /^[0-9a-f-]{36}$/);
    assert.equal(manifest.get('x-llamadart-decoded-size'), '10');
    assert.equal(manifest.get('x-llamadart-part-count'), '3');
    assert.equal(manifest.get('x-llamadart-part-size'), '4');
    assert.equal(manifest.get('etag'), '"v1"');
    assert.deepEqual(caches.partBytes(record), [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]]);
    assert.equal(caches.entries.has(MODEL_URL), false, 'no whole-body entry');
    assert.equal(caches.calls.put.at(-1).startsWith(MANIFEST), true, 'the manifest is the last write');
    await locks.whenIdle();
    assert.equal(locks.held.size, 0, 'the lock is released after the commit');
    assert.deepEqual(locks.requests[0].options, { ifAvailable: true });
    assert.ok(locks.requests[0].name.includes(CACHE_NAME));
  })],

  ['a part still being written holds the transfer back', () => withFakes({}, async (caches) => {
    const { promise: hold, resolve: allow } = Promise.withResolvers();
    const putStarted = Promise.withResolvers();
    caches.hooks.beforePut = (url) => {
      if (!url.startsWith(PART)) return;
      putStarted.resolve();
      return hold;
    };
    const source = controlledResponse();
    const store = startModelCachePartStore(caches.cache, MODEL_URL, source.response, CACHE_NAME, 4);
    await withStoreReader(store, async (reader) => {
      let delivered = false;
      const first = reader.read().then((result) => {
        delivered = true;
        return result;
      });
      const second = reader.read();
      source.push([1, 2, 3, 4]);
      source.push([5]);
      try {
        await Promise.race([putStarted.promise, first]);
        await nextTurn();
        assert.equal(delivered, false, 'the first chunk is held until the put commits');
        assert.equal(source.state.reads, 1, 'the source is not read past a part that has not committed');
      } finally {
        allow();
      }
      assert.deepEqual([...(await first).value], [1, 2, 3, 4]);
      assert.deepEqual([...(await second).value], [5]);
      source.close();
      assert.equal((await reader.read()).done, true);
      await store.stored;
    });
  })],

  ['the caller may release its chunks without changing the cached parts', () => withFakes({}, async (caches) => {
    const chunk = Uint8Array.of(1, 2, 3, 4, 5, 6);
    const source = new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(chunk);
        controller.close();
      },
    }));
    const store = startModelCachePartStore(caches.cache, MODEL_URL, source, CACHE_NAME, 4);
    const reader = store.response.body.getReader();
    const { value } = await reader.read();
    assert.equal(value, chunk, 'the chunk is passed through, not copied');
    value.buffer.transfer(0);
    assert.equal((await reader.read()).done, true);
    await store.stored;
    assert.deepEqual(caches.partBytes(caches.manifest().get('x-llamadart-cache-record')), [[1, 2, 3, 4], [5, 6]]);
  })],

  ['a cancelled, failed or empty transfer commits nothing and removes its parts', () => withFakes({}, async (caches, locks) => {
    const cancelled = controlledResponse();
    let store = startModelCachePartStore(caches.cache, MODEL_URL, cancelled.response, CACHE_NAME, 4);
    let reader = store.response.body.getReader();
    let read = reader.read();
    cancelled.push([1, 2, 3, 4, 5]);
    await read;
    await reader.cancel(new Error('load cancelled'));
    await assert.rejects(store.stored, /load cancelled/);
    assert.equal(store.cancelled, true);
    assert.equal(cancelled.state.cancelled, true);
    await locks.whenIdle();
    assert.equal(caches.entries.size, 0);
    assert.equal(locks.held.size, 0);

    const failed = controlledResponse();
    store = startModelCachePartStore(caches.cache, MODEL_URL, failed.response, CACHE_NAME, 4);
    reader = store.response.body.getReader();
    read = reader.read();
    failed.push([1, 2, 3, 4, 5]);
    await read;
    failed.fail(new Error('network error'));
    await assert.rejects(reader.read(), /network error/);
    await assert.rejects(store.stored, /network error/);
    assert.equal(store.cancelled, false);
    await locks.whenIdle();
    assert.equal(caches.entries.size, 0);
    assert.equal(locks.held.size, 0);

    store = startModelCachePartStore(caches.cache, MODEL_URL, bodyOf([]), CACHE_NAME, 4);
    assert.deepEqual(await readAll(store.response), []);
    await assert.rejects(store.stored, /empty/);
    await locks.whenIdle();
    assert.equal(caches.entries.size, 0);
  })],

  ['cancelling during a pending put prunes the part before releasing the lock', () => withFakes({}, async (caches, locks) => {
    const putStarted = Promise.withResolvers();
    const finishPut = Promise.withResolvers();
    caches.hooks.beforePut = async () => {
      putStarted.resolve();
      await finishPut.promise;
    };
    const store = startModelCachePartStore(caches.cache, MODEL_URL, bodyOf(TEN, TEN.length), CACHE_NAME, 4);
    const reader = store.response.body.getReader();
    const read = reader.read();
    await putStarted.promise;
    const rejection = assert.rejects(store.stored, /load cancelled/);
    const cancel = reader.cancel(new Error('load cancelled'));
    await nextTurn();
    const lockedDuringPut = locks.held.size === 1;
    finishPut.resolve();
    await Promise.all([cancel, read, rejection]);
    await locks.whenIdle();
    assert.equal(lockedDuringPut, true);
    assert.equal(locks.held.size, 0);
    assert.equal(caches.calls.put.length, 1);
    assert.equal(caches.entries.size, 0);
  })],

  ['a part that cannot be written stops the store and leaves the transfer intact', () => withFakes({}, async (caches, locks) => {
    let puts = 0;
    caches.hooks.beforePut = (url) => {
      if (!url.startsWith(PART)) return;
      puts += 1;
      if (puts === 2) throw new DOMException('quota', 'QuotaExceededError');
    };
    const store = startModelCachePartStore(caches.cache, MODEL_URL, bodyOf(TEN), CACHE_NAME, 4);
    assert.deepEqual(await readAll(store.response), TEN);
    await assert.rejects(store.stored, { name: 'QuotaExceededError' });
    await locks.whenIdle();
    assert.equal(puts, 2, 'no part is written after the failure');
    assert.equal(caches.entries.size, 0);
    assert.equal(locks.held.size, 0);
  })],

  ['a store never runs beside another store of the same model', async () => {
    await withFakes({ available: false }, async (caches) => {
      const store = startModelCachePartStore(caches.cache, MODEL_URL, bodyOf(TEN), CACHE_NAME, 4);
      assert.deepEqual(await readAll(store.response), TEN);
      await assert.rejects(store.stored, /Another context is storing this model/);
      assert.deepEqual(caches.calls.put, []);
      assert.deepEqual(caches.calls.delete, []);
    });
    await withFakes({ present: false }, async (caches) => {
      const store = startModelCachePartStore(caches.cache, MODEL_URL, bodyOf(TEN), CACHE_NAME, 4);
      assert.deepEqual(await readAll(store.response), TEN);
      await assert.rejects(store.stored, /Web Locks are unavailable/);
      assert.deepEqual(caches.calls.put, []);
    });
  }],

  ['missing UUID support and Web Locks leave loads uncached without retrying the download', async () => {
    for (const present of [true, false]) {
      await withFakes({ present, uuid: false }, async (caches, locks) => {
        const { runtime, loads, fetches } = createRuntime(() => bodyOf(TEN), SAFARI);
        await assert.rejects(
          runtime.prefetchModelToCache(MODEL_URL, { cacheName: CACHE_NAME }),
          /Failed to store prefetched model/,
        );
        assert.equal(await runtime.loadModelFromUrl(MODEL_URL, loadOptions()), 1);
        assert.deepEqual(loads, [TEN]);
        assert.equal(runtime._modelCacheState, 'store_failed');
        assert.equal(fetches.length, 2);
        assert.equal(caches.entries.size, 0);
        assert.equal(locks.held.size, 0);
      });
    }
  }],

  ['a new store keeps the committed model readable until it commits, then replaces it', () => withFakes({}, async (caches) => {
    await storeParts(caches);
    const first = caches.manifest().get('x-llamadart-cache-record');
    // A killed tab's parts, and a whole-body entry from an earlier version.
    caches.entries.set(caches.urls(PART)[0].replace(first, '11111111-1111-4111-8111-111111111111'), Uint8Array.of(9));
    caches.entries.set(MODEL_URL, Uint8Array.of(7));

    const source = controlledResponse();
    const store = startModelCachePartStore(caches.cache, MODEL_URL, source.response, CACHE_NAME, 4);
    await withStoreReader(store, async (reader) => {
      source.push([21, 22, 23, 24, 25]);
      assert.deepEqual([...(await reader.read()).value], [21, 22, 23, 24, 25]);
      assert.equal(caches.partBytes('11111111-1111-4111-8111-111111111111').length, 0, 'uncommitted parts are pruned first');
      assert.equal(caches.manifest().get('x-llamadart-cache-record'), first);
      assert.deepEqual(await readAll((await openModelCacheParts(caches.cache, MODEL_URL, CACHE_NAME)).response), TEN);
      source.close();
      assert.equal((await reader.read()).done, true);
      await store.stored;
    });

    const second = caches.manifest().get('x-llamadart-cache-record');
    assert.notEqual(second, first);
    assert.deepEqual(caches.partBytes(first), []);
    assert.deepEqual(caches.partBytes(second), [[21, 22, 23, 24], [25]]);
    assert.equal(caches.entries.has(MODEL_URL), false, 'the whole-body entry is a stale duplicate');
    assert.equal(caches.entries.size, 3);
  })],

  ['a part-stored model reads back as one response with its decoded size', () => withFakes({}, async (caches) => {
    assert.equal(await openModelCacheParts(caches.cache, MODEL_URL, CACHE_NAME), null);
    await storeParts(caches, TEN, 4, { etag: '"v1"', 'content-encoding': 'gzip', 'content-length': '3' });
    const before = caches.calls.match.length;
    const parts = await openModelCacheParts(caches.cache, MODEL_URL + '#fragment', CACHE_NAME);
    assert.equal(parts.decodedBytes, 10);
    assert.equal(parts.response.headers.get('content-length'), '10');
    assert.equal(parts.response.headers.get('content-encoding'), null);
    assert.equal(parts.response.headers.get('etag'), '"v1"');
    for (const name of ['x-llamadart-cache-record', 'x-llamadart-decoded-size', 'x-llamadart-part-count', 'x-llamadart-part-size']) {
      assert.equal(parts.response.headers.get(name), null, name);
    }
    assert.equal(caches.calls.match.length, before + 1, 'opening reads the manifest only');
    const reader = parts.response.body.getReader();
    assert.deepEqual([...(await reader.read()).value], [1, 2, 3, 4]);
    assert.equal(caches.calls.match.length, before + 2, 'one part is read per pull');
    assert.deepEqual([...(await reader.read()).value], [5, 6, 7, 8]);
    assert.deepEqual([...(await reader.read()).value], [9, 10]);
    assert.equal((await reader.read()).done, true);
  })],

  ['a missing or short part fails the read and drops the entry, and a damaged manifest is a miss', () => withFakes({}, async (caches) => {
    await storeParts(caches);
    const partUrl = (index) => caches.urls(PART).find((url) => url.endsWith('&part=' + index));
    caches.entries.delete(partUrl(1));
    await assert.rejects(readAll((await openModelCacheParts(caches.cache, MODEL_URL, CACHE_NAME)).response), { name: 'ModelCacheEntryError' });
    assert.equal(caches.manifest(), null, 'the damaged entry is no longer a hit');
    assert.equal(await openModelCacheParts(caches.cache, MODEL_URL, CACHE_NAME), null);

    await storeParts(caches);
    caches.entries.set(partUrl(2), Uint8Array.of(9));
    await assert.rejects(readAll((await openModelCacheParts(caches.cache, MODEL_URL, CACHE_NAME)).response), { name: 'ModelCacheEntryError' });
    assert.equal(caches.manifest(), null);

    // A reader of a record that a newer store replaced must not drop the new one.
    await storeParts(caches);
    const stale = await openModelCacheParts(caches.cache, MODEL_URL, CACHE_NAME);
    await storeParts(caches, [21, 22, 23, 24, 25]);
    await assert.rejects(readAll(stale.response), { name: 'ModelCacheEntryError' });
    assert.deepEqual(await readAll((await openModelCacheParts(caches.cache, MODEL_URL, CACHE_NAME)).response), [21, 22, 23, 24, 25]);

    caches.manifest().set('x-llamadart-part-count', '3');
    assert.equal(await openModelCacheParts(caches.cache, MODEL_URL, CACHE_NAME), null);
  })],

  ['invalidating a stale snapshot under the store lock preserves a newer record', () => withFakes({}, async (caches, locks) => {
    await storeParts(caches);
    const oldRecord = caches.manifest().get('x-llamadart-cache-record');
    const snapshot = Promise.withResolvers();
    const returnSnapshot = Promise.withResolvers();
    caches.hooks.afterMatch = async (url) => {
      if (!url.startsWith(MANIFEST)) return;
      caches.hooks.afterMatch = null;
      snapshot.resolve();
      await returnSnapshot.promise;
    };
    const opening = openModelCacheParts(caches.cache, MODEL_URL, CACHE_NAME);
    await snapshot.promise;
    const fresh = [21, 22, 23, 24, 25];
    await storeParts(caches, fresh);
    const newRecord = caches.manifest().get('x-llamadart-cache-record');
    returnSnapshot.resolve();
    const stale = await opening;
    const checkedWhileLocked = [];
    caches.hooks.afterMatch = (url) => {
      if (url.startsWith(MANIFEST)) checkedWhileLocked.push([...locks.held]);
    };
    await assert.rejects(readAll(stale.response), { name: 'ModelCacheEntryError' });
    assert.deepEqual(checkedWhileLocked, [[locks.requests[0].name]]);
    assert.notEqual(newRecord, oldRecord);
    assert.equal(caches.manifest().get('x-llamadart-cache-record'), newRecord);
    assert.deepEqual(caches.partBytes(newRecord), [[21, 22, 23, 24], [25]]);
    assert.deepEqual(await readAll((await openModelCacheParts(caches.cache, MODEL_URL, CACHE_NAME)).response), fresh);
  })],

  ['WebKit stores a loaded model in parts, releases its chunks and reloads it from the parts', () => withFakes({}, async (caches) => {
    const chunk = Uint8Array.of(1, 2, 3, 4, 5, 6);
    const { runtime, core, loads, fetches } = createRuntime(() => new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(chunk);
        controller.close();
      },
    }), { headers: { 'content-length': '2', 'content-encoding': 'gzip' } }), SAFARI);
    runtime._config.jsBackedModelStaging = true;
    runtime._mountJsBackedStaging = () => '/jsmodels';
    runtime._assertJsStagingMountOwned = () => {};
    const heapSizes = [];
    runtime._openDirectJsWriter = (_core, filePath, _total, _append, _assertOwned, heapBytes) => {
      const handle = core.FS.open(filePath, 'w');
      heapSizes.push(heapBytes ?? 0);
      return {
        heapResident: heapBytes > 0,
        writeChunk(bytes, position) {
          return core.FS.write(handle, bytes, 0, bytes.length, position);
        },
      };
    };

    assert.equal(await runtime.loadModelFromUrl(MODEL_URL, loadOptions()), 1);
    assert.equal(runtime._modelCacheState, 'stored');
    assert.ok(runtime._runtimeNotes.includes('model_cache_stored'));
    assert.ok(runtime._runtimeNotes.includes('model_cache_parts'));
    assert.equal(chunk.byteLength, 0, 'a chunk the cache does not share is released');
    assert.equal(caches.entries.has(MODEL_URL), false);
    assert.equal(caches.urls(PART).length, 1);
    assert.equal(caches.manifest().get('x-llamadart-decoded-size'), '6');

    assert.equal(await runtime.loadModelFromUrl(MODEL_URL, loadOptions()), 1);
    assert.equal(fetches.length, 1, 'the reload reads the cache');
    assert.equal(runtime._modelCacheState, 'hit');
    assert.equal(runtime._modelSource, 'cache');
    assert.ok(runtime._runtimeNotes.includes('model_cache_parts'));
    assert.deepEqual(heapSizes, [0, 6], 'the reload stages into a heap block of the recorded decoded size');
    assert.ok(runtime._runtimeNotes.includes('model_staging_heap_resident'));
    assert.deepEqual(loads, [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]);
  })],

  ['a damaged part entry is dropped and the model is downloaded again', () => withFakes({}, async (caches) => {
    // Small parts, so the reload stages the first part before it meets the missing one.
    await storeParts(caches, TEN, 4);
    const second = caches.urls(PART).find((url) => url.endsWith('&part=1'));
    caches.entries.delete(second);
    const fresh = [21, 22, 23, 24, 25, 26];
    const { runtime, loads, fetches } = createRuntime(() => bodyOf(fresh), SAFARI);

    assert.equal(await runtime.loadModelFromUrl(MODEL_URL, loadOptions()), 1);
    assert.equal(fetches.length, 1);
    assert.ok(runtime._runtimeNotes.includes('model_fs_write_loaded:4'), 'the first part was staged');
    assert.ok(runtime._runtimeNotes.includes('model_cache_entry_invalid'));
    assert.ok(runtime._runtimeNotes.includes('model_stream_restart:cache'));
    assert.equal(runtime._modelCacheState, 'stored');
    assert.equal(runtime._modelSource, 'network');
    assert.deepEqual(loads[0], fresh, 'the staged cache prefix is replaced, not kept');
    assert.equal(caches.manifest().get('x-llamadart-decoded-size'), '6');
    assert.equal(caches.urls(PART).length, 1, 'the damaged entry is gone');

    assert.equal(await runtime.loadModelFromUrl(MODEL_URL, loadOptions()), 1);
    assert.equal(fetches.length, 1, 'the re-stored model is a hit');
    assert.deepEqual(loads[1], fresh);
  })],

  ['a damaged cache hit retries uncached if a writer owns the lock', () => withFakes({}, async (caches, locks) => {
    await storeParts(caches);
    const record = caches.manifest().get('x-llamadart-cache-record');
    caches.entries.delete(caches.urls(PART)[0]);
    const writes = caches.calls.put.length;
    const newBytes = [21, 22, 23];
    const { runtime, loads, fetches } = createRuntime(() => bodyOf(newBytes), SAFARI);
    const lock = locks.requests[0].name;
    locks.held.add(lock);
    try {
      assert.equal(await runtime.loadModelFromUrl(MODEL_URL, loadOptions()), 1);
      assert.ok(runtime._runtimeNotes.includes('model_stream_restart:cache'));
      assert.equal(runtime._modelCacheState, 'disabled');
      assert.deepEqual(loads[0], newBytes);
      assert.equal(fetches.length, 1);
      assert.equal(caches.calls.put.length, writes);
      assert.equal(caches.manifest().get('x-llamadart-cache-record'), record);
    } finally {
      locks.held.delete(lock);
    }
  })],

  ['failed staging setup leaves no cache lock behind for a healthy retry', () => withFakes({}, async (caches, locks) => {
    const { runtime, core, fetches } = createRuntime(() => bodyOf(TEN), SAFARI);
    const open = core.FS.open;
    core.FS.open = () => { throw new Error('FS open failed'); };
    await assert.rejects(runtime.loadModelFromUrl(MODEL_URL, loadOptions()), /FS open failed/);
    await nextTurn();
    assert.equal(locks.requests.length, 0);
    core.FS.open = open;
    assert.equal(await runtime.loadModelFromUrl(MODEL_URL, loadOptions()), 1);
    assert.equal(fetches.length, 2);
    assert.equal(runtime._modelCacheState, 'stored');
    assert.equal(locks.requests.length, 1, 'only the healthy retry takes a lock');
    assert.equal(caches.entries.size, 2);
  })],

  ['partial HTTP responses never become full model cache hits', () => withFakes({}, async (caches, locks) => {
    const { runtime, loads, fetches } = createRuntime(() => new Response(Uint8Array.of(1, 2, 3, 4), {
      status: 206,
      headers: { 'content-range': 'bytes 0-3/10', 'content-length': '4', 'content-encoding': 'identity' },
    }), SAFARI);
    await assert.rejects(
      runtime.prefetchModelToCache(MODEL_URL, { cacheName: CACHE_NAME }),
      /Failed to store prefetched model/,
    );
    for (let attempt = 0; attempt < 2; attempt++) {
      await assert.rejects(runtime.loadModelFromUrl(MODEL_URL, loadOptions()), { name: 'ModelTransferLengthError' });
    }
    assert.equal(locks.requests.length, 0);
    assert.deepEqual(caches.calls.put, []);
    assert.deepEqual(loads, []);
    assert.equal(fetches.length, 3);
  })],

  ['a model with a trustworthy length mismatch never commits a manifest', async () => {
    for (const length of ['2', '10']) {
      await withFakes({}, async (caches, locks) => {
        const { runtime, loads, fetches } = createRuntime(() => bodyOf([1, 2, 3, 4], 3, {
          'content-length': length, 'content-encoding': 'identity',
        }), SAFARI);
        await assert.rejects(
          runtime.prefetchModelToCache(MODEL_URL, { cacheName: CACHE_NAME }),
          /Failed to store prefetched model/,
        );
        for (let attempt = 0; attempt < 2; attempt++) {
          await assert.rejects(runtime.loadModelFromUrl(MODEL_URL, loadOptions()), { name: 'ModelTransferLengthError' });
        }
        await locks.whenIdle();
        assert.equal(caches.entries.size, 0);
        assert.deepEqual(loads, []);
        assert.equal(fetches.length, 3);
      });
    }
  }],

  ['a compressed or unverified wire length does not prevent a decoded model from being cached', async () => {
    for (const headers of [
      { 'content-length': '2', 'content-encoding': 'gzip', 'x-linked-size': '4' },
      { 'content-length': '2' },
    ]) {
      await withFakes({}, async (caches) => {
        const bytes = [1, 2, 3, 4];
        const { runtime, loads, fetches } = createRuntime(() => {
          const response = bodyOf(bytes, 3, headers);
          Object.defineProperty(response, 'type', { value: 'cors' });
          return response;
        }, SAFARI);
        assert.equal(await runtime.prefetchModelToCache(MODEL_URL, { cacheName: CACHE_NAME }), 1);
        assert.equal(caches.manifest().get('x-llamadart-decoded-size'), '4');
        assert.equal(await runtime.loadModelFromUrl(MODEL_URL, loadOptions()), 1);
        assert.equal(runtime._modelCacheState, 'hit');
        assert.deepEqual(loads, [bytes]);
        assert.equal(fetches.length, 1);
      });
    }
  }],

  ['eviction skips a locked part store and can evict it after commit', () => withFakes({}, async (caches) => {
    const { runtime } = createRuntime(() => assert.fail('unexpected fetch'), SAFARI);
    const source = controlledResponse();
    const store = startModelCachePartStore(caches.cache, MODEL_URL, source.response, CACHE_NAME, 4);
    const reader = store.response.body.getReader();
    source.push([1, 2, 3, 4, 5]);
    assert.deepEqual([...(await reader.read()).value], [1, 2, 3, 4, 5]);
    assert.equal(caches.urls(PART).length, 1);
    const options = { cacheName: CACHE_NAME };
    assert.equal(await runtime.evictModelFromCache(MODEL_URL, options), false);
    assert.equal(caches.urls(PART).length, 1, 'eviction preserved the uncommitted part');
    source.push([6, 7, 8, 9, 10]);
    source.close();
    assert.deepEqual([...(await reader.read()).value], [6, 7, 8, 9, 10]);
    assert.equal((await reader.read()).done, true);
    await store.stored;
    assert.deepEqual(await readAll((await openModelCacheParts(caches.cache, MODEL_URL, CACHE_NAME)).response), TEN);
    assert.equal(await runtime.evictModelFromCache(MODEL_URL, options), true);
    assert.equal(caches.entries.size, 0);
  })],

  ['WebKit prefetch, cache hit and eviction use the part entries', () => withFakes({}, async (caches) => {
    const { runtime, fetches } = createRuntime(() => bodyOf([1, 2, 3, 4, 5, 6], 3, {
      'content-length': '2', 'content-encoding': 'gzip',
    }), SAFARI);
    const options = { cacheName: CACHE_NAME };
    assert.equal(await runtime.prefetchModelToCache(MODEL_URL + '#one', options), 1);
    assert.equal(runtime._modelCacheState, 'stored');
    assert.equal(runtime._runtimeNotes.filter((note) => note === 'model_cache_parts').length, 1);
    assert.equal(caches.entries.has(MODEL_URL), false);
    assert.equal(caches.entries.size, 2);

    const progress = [];
    assert.equal(await runtime.prefetchModelToCache(MODEL_URL + '#two', {
      ...options, progressCallback: (event) => progress.push(event),
    }), 1);
    assert.equal(runtime._modelCacheState, 'hit');
    assert.equal(runtime._runtimeNotes.filter((note) => note === 'model_cache_parts').length, 2);
    assert.deepEqual(progress.at(-1), { loaded: 6, total: 6 });
    assert.equal(fetches.length, 1);

    assert.equal(await runtime.prefetchModelToCache(MODEL_URL, { ...options, force: true }), 1);
    assert.equal(fetches.length, 2);
    assert.equal(caches.entries.size, 2, 'only the latest parts and their manifest remain');

    assert.equal(await runtime.evictModelFromCache(MODEL_URL + '#three', options), true);
    assert.equal(caches.entries.size, 0);
    assert.equal(runtime._modelCacheState, 'evicted');
    assert.equal(await runtime.evictModelFromCache(MODEL_URL, options), false);
  })],

  ['WebKit still reads a whole-body entry stored by an earlier version', () => withFakes({}, async (caches) => {
    caches.entries.set(MODEL_URL, Uint8Array.of(1, 2, 3));
    caches.headers.set(MODEL_URL, new Headers({ 'content-length': '3' }));
    const { runtime, loads, fetches } = createRuntime(() => assert.fail('unexpected fetch'), SAFARI);
    assert.equal(await runtime.loadModelFromUrl(MODEL_URL, loadOptions()), 1);
    assert.equal(fetches.length, 0);
    assert.equal(runtime._modelCacheState, 'hit');
    assert.ok(!runtime._runtimeNotes.includes('model_cache_parts'));
    assert.deepEqual(loads[0], [1, 2, 3]);
    assert.equal(await runtime.evictModelFromCache(MODEL_URL, { cacheName: CACHE_NAME }), true);
    assert.equal(caches.entries.size, 0);
  })],

  ['other engines keep the whole-body entry and never touch part entries', () => withFakes({}, async (caches, locks) => {
    const { runtime, loads, fetches } = createRuntime(() => bodyOf([1, 2, 3, 4, 5, 6]), 'Chrome');
    assert.equal(await runtime.loadModelFromUrl(MODEL_URL, loadOptions()), 1);
    assert.deepEqual([...caches.entries.get(MODEL_URL)], [1, 2, 3, 4, 5, 6]);
    assert.equal(caches.urls(MANIFEST).length + caches.urls(PART).length, 0);
    assert.equal(locks.requests.length, 0);
    assert.ok(!runtime._runtimeNotes.includes('model_cache_parts'));
    const matches = caches.calls.match.length;
    assert.equal(await runtime.loadModelFromUrl(MODEL_URL, loadOptions()), 1);
    assert.equal(fetches.length, 1);
    assert.deepEqual(
      caches.calls.match.slice(matches).filter((url) => url.startsWith(MANIFEST) || url.startsWith(PART)),
      [],
      'a hit on another engine looks up no part entry',
    );
    assert.deepEqual(loads[1], [1, 2, 3, 4, 5, 6]);
  })],
];

for (const [name, run] of CASES) {
  test(name, run);
}
