import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';

import { teeResponseForCache } from '../../js/src/internal/download.ts';
import { createDirectBridge } from './bridge_operation_queue_fixtures.mjs';

const MODEL_URL = 'https://models.test/model.gguf';
const CACHE_NAME = 'model-cache-stream-test';

// A network response whose body the test feeds chunk by chunk. `reads` counts
// the reads the consumer made of it.
function controlledResponse(total, url = MODEL_URL) {
  const queue = [];
  const state = { cancelled: false, closed: false, reads: 0 };
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
      if (state.cancelled) {
        return;
      }
      if (queue.length === 0 && state.error) {
        throw state.error;
      }
      if (queue.length > 0) {
        controller.enqueue(queue.shift());
      } else {
        controller.close();
      }
    },
    cancel() {
      state.cancelled = true;
      state.closed = true;
      notify();
    },
  }, { highWaterMark: 0 });
  const response = new Response(stream, {
    status: 200,
    statusText: 'OK',
    headers: { 'content-length': String(total) },
  });
  Object.defineProperty(response, 'url', { value: url });
  return {
    response,
    state,
    push(bytes) {
      queue.push(new Uint8Array(bytes));
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

const concat = (chunks) => new Uint8Array(chunks.flatMap((chunk) => [...chunk]));

async function readAll(response) {
  return new Uint8Array(await response.arrayBuffer());
}

// Like the Cache API, put() reads the whole body and commits an entry only when
// the body completes; a body that errors rejects put() and leaves no entry.
function installCaches({ putError = null, holdPut = null, commitDelayMs = 0 } = {}) {
  const entries = new Map();
  const original = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  const cache = {
    async match(key) {
      const stored = entries.get(key);
      return stored
        ? new Response(stored, { headers: { 'content-length': String(stored.length) } })
        : undefined;
    },
    async put(key, response) {
      if (putError) {
        throw putError;
      }
      await holdPut;
      const bytes = await readAll(response);
      await delay(commitDelayMs);
      entries.set(key, bytes);
    },
  };
  Object.defineProperty(globalThis, 'caches', {
    configurable: true,
    value: { open: async (name) => (name === CACHE_NAME ? cache : assert.fail(`unexpected cache ${name}`)) },
  });
  return {
    entries,
    restore() {
      if (original) {
        Object.defineProperty(globalThis, 'caches', original);
      } else {
        delete globalThis.caches;
      }
    },
  };
}

function createRuntime(fetchModel) {
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
      loads.push(concat(written.get(args[0])));
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
  const logs = [];
  runtime._emitLogger = (level, message) => logs.push([level, message]);
  const fetches = [];
  runtime._fetchWithTimeout = async (url, init) => {
    fetches.push(url);
    return fetchModel(url, init);
  };
  return { runtime, core, loads, fetches, logs };
}

const DRAFT_URL = 'https://models.test/draft.gguf';

// A runtime with a target model loaded from the cache, ready to load a draft.
function draftRuntime(source, headroomBytes) {
  const { runtime, core } = createRuntime(() => source.response);
  runtime._modelBytes = 1;
  runtime._modelSource = 'cache';
  runtime._modelCacheState = 'hit';
  runtime._config.cacheName = CACHE_NAME;
  const ccall = core.ccall;
  core.ccall = (name, type, types, args = []) => {
    if (name === 'llamadart_webgpu_heap_headroom_bytes') return headroomBytes;
    if (name === 'llamadart_webgpu_draft_model_free') return 0;
    if (name === 'llamadart_webgpu_draft_model_load') return Promise.resolve(0);
    if (name === 'llamadart_webgpu_draft_model_info_json') return '{}';
    return ccall(name, type, types, args);
  };
  return { runtime, core };
}

// Feeds `chunk` each time the consumer reads, up to `limit` chunks in all,
// until it stops reading. Returns how many chunks were fed.
async function feedChunks(source, chunk, limit, fed = 0) {
  for (let idle = 0; idle < 20 && fed < limit;) {
    if (source.state.reads > fed) {
      source.push(chunk);
      fed += 1;
      idle = 0;
    } else {
      idle += 1;
      await delay(1);
    }
  }
  return fed;
}

const loadOptions = (extra = {}) => ({
  nCtx: 128,
  nThreads: 1,
  nGpuLayers: 0,
  forceRemoteFetchBackend: false,
  cacheName: CACHE_NAME,
  ...extra,
});

function nextProgress() {
  const events = [];
  let waiter = null;
  return {
    events,
    callback(event) {
      events.push(event);
      waiter?.();
    },
    async next(timeoutMs = 2000) {
      const count = events.length;
      while (events.length === count) {
        const arrived = new Promise((resolve) => {
          waiter = resolve;
        });
        const result = await Promise.race([arrived.then(() => true), delay(timeoutMs).then(() => false)]);
        assert.ok(result, 'expected a progress event before the body completed');
      }
      return events.at(-1);
    },
  };
}

const CASES = [
  ['the cache copy and the caller copy carry the whole body, status, headers and url', async () => {
    const source = controlledResponse(6);
    const tee = teeResponseForCache(source.response);
    assert.equal(tee.response.url, MODEL_URL);
    assert.equal(tee.response.status, 200);
    assert.equal(tee.response.headers.get('content-length'), '6');
    assert.equal(tee.cacheResponse.headers.get('content-length'), '6');
    const cached = readAll(tee.cacheResponse);
    const body = readAll(tee.response);
    source.push([1, 2, 3]);
    source.push([4, 5, 6]);
    source.close();
    assert.deepEqual([...await body], [1, 2, 3, 4, 5, 6]);
    assert.deepEqual([...await cached], [1, 2, 3, 4, 5, 6]);
  }],

  ['the source is read no further ahead of an unread cache copy than the lag bound', async () => {
    const source = controlledResponse(64);
    const tee = teeResponseForCache(source.response, 8);
    const reader = tee.response.body.getReader();
    for (let i = 0; i < 2; i += 1) {
      const read = reader.read();
      source.push([1, 2, 3, 4]);
      assert.equal((await read).value.length, 4);
    }
    const blocked = reader.read();
    await delay(20);
    assert.equal(source.state.reads, 2, 'with 8 bytes queued for the cache, the source is not read again');

    const cacheReader = tee.cacheResponse.body.getReader();
    assert.equal((await cacheReader.read()).value.length, 4);
    await delay(0);
    assert.equal(source.state.reads, 3, 'draining the cache copy resumes the transfer');
    source.push([5]);
    assert.deepEqual([...(await blocked).value], [5]);
    await reader.cancel();
    await assert.rejects(cacheReader.read(), /cancelled before the cache copy completed/);
  }],

  ['cancelling the caller copy errors the cache copy and cancels the source', async () => {
    const source = controlledResponse(8);
    const tee = teeResponseForCache(source.response);
    const cached = readAll(tee.cacheResponse);
    const reader = tee.response.body.getReader();
    const read = reader.read();
    source.push([1, 2]);
    await read;
    await reader.cancel(new Error('load cancelled'));
    await assert.rejects(cached, /load cancelled/);
    assert.equal(source.state.cancelled, true);
  }],

  ['a network error ends both copies, so the cache store settles', async () => {
    const source = controlledResponse(8);
    const tee = teeResponseForCache(source.response);
    const cached = readAll(tee.cacheResponse);
    const body = readAll(tee.response);
    source.push([1, 2]);
    source.fail(new TypeError('network error'));
    await assert.rejects(body, /network error/);
    await assert.rejects(cached, /network error/);
  }],

  ['a cache copy that stops reading no longer holds the caller back', async () => {
    const source = controlledResponse(4);
    const tee = teeResponseForCache(source.response, 1);
    tee.detachCache(new Error('quota'));
    await assert.rejects(readAll(tee.cacheResponse), /quota/);
    const body = readAll(tee.response);
    source.push([1, 2]);
    source.push([3, 4]);
    source.close();
    assert.deepEqual([...await body], [1, 2, 3, 4]);
  }],

  ['a cache miss reports progress while downloading and caches the model after the load', async () => {
    const caches = installCaches({ commitDelayMs: 10 });
    try {
      const source = controlledResponse(6);
      const { runtime, loads, fetches } = createRuntime(() => source.response);
      const progress = nextProgress();
      const loading = runtime.loadModelFromUrl(MODEL_URL, loadOptions({ progressCallback: progress.callback }));

      source.push([1, 2, 3]);
      assert.deepEqual(await progress.next(), { loaded: 3, total: 6 }, 'progress follows the first chunk');
      assert.equal(caches.entries.size, 0, 'the entry is committed only once the body completes');
      source.push([4, 5, 6]);
      source.close();
      assert.equal(await loading, 1);

      assert.deepEqual([...loads[0]], [1, 2, 3, 4, 5, 6]);
      assert.deepEqual([...caches.entries.get(MODEL_URL)], [1, 2, 3, 4, 5, 6]);
      assert.equal(runtime._modelCacheState, 'stored');
      assert.ok(runtime._runtimeNotes.includes('model_cache_stored'));

      assert.equal(await runtime.loadModelFromUrl(MODEL_URL, loadOptions()), 1);
      assert.equal(fetches.length, 1, 'the reload reads the cache');
      assert.equal(runtime._modelCacheState, 'hit');
      assert.ok(runtime._runtimeNotes.includes('model_cache_hit'));
      assert.deepEqual([...loads[1]], [1, 2, 3, 4, 5, 6]);
    } finally {
      caches.restore();
    }
  }],

  ['a failed cache store is logged and the model still loads', async () => {
    const caches = installCaches({ putError: new DOMException('Quota exceeded', 'QuotaExceededError') });
    try {
      const source = controlledResponse(4);
      const { runtime, loads, logs } = createRuntime(() => source.response);
      const loading = runtime.loadModelFromUrl(MODEL_URL, loadOptions());
      source.push([1, 2]);
      source.push([3, 4]);
      source.close();
      assert.equal(await loading, 1);
      assert.deepEqual([...loads[0]], [1, 2, 3, 4]);
      assert.equal(caches.entries.size, 0);
      assert.equal(runtime._modelCacheState, 'store_failed');
      assert.ok(runtime._runtimeNotes.includes('model_cache_store_failed'));
      assert.deepEqual(logs, [['warn', 'Model was not cached: Quota exceeded']]);
    } finally {
      caches.restore();
    }
  }],

  ['a cache store that fails without reading does not hold back a large download', async () => {
    const caches = installCaches({ putError: new DOMException('Quota exceeded', 'QuotaExceededError') });
    try {
      const chunk = new Uint8Array(1024 * 1024);
      const source = controlledResponse(12 * chunk.length);
      const { runtime } = createRuntime(() => source.response);
      const loading = runtime.loadModelFromUrl(MODEL_URL, loadOptions());
      assert.equal(await feedChunks(source, chunk, 12), 12);
      source.close();
      assert.equal(await loading, 1);
      assert.equal(runtime._modelCacheState, 'store_failed');
      assert.equal(caches.entries.size, 0);
    } finally {
      caches.restore();
    }
  }],

  ['a load cancelled mid-download leaves no cache entry', async () => {
    const caches = installCaches();
    try {
      const source = controlledResponse(6);
      const { runtime, loads, logs } = createRuntime(() => source.response);
      const controller = new AbortController();
      const progress = nextProgress();
      const loading = runtime.loadModelFromUrl(
        MODEL_URL,
        loadOptions({ signal: controller.signal, progressCallback: progress.callback }),
      );
      source.push([1, 2, 3]);
      await progress.next();
      controller.abort();
      source.push([4]);
      await assert.rejects(loading, (error) => error.name === 'AbortError');
      await delay(0);
      assert.equal(source.state.cancelled, true, 'the download stops');
      assert.equal(caches.entries.size, 0);
      assert.equal(loads.length, 0);
      await delay(10);
      assert.equal(runtime._runtimeNotes.includes('model_cache_store_failed'), false, 'a cancelled load records no cache failure');
      assert.deepEqual(logs, []);
    } finally {
      caches.restore();
    }
  }],

  ['a cache that writes slowly holds the download back instead of buffering it', async () => {
    let release;
    const caches = installCaches({ holdPut: new Promise((resolve) => { release = resolve; }) });
    try {
      const chunk = new Uint8Array(1024 * 1024);
      const source = controlledResponse(32 * chunk.length);
      const { runtime } = createRuntime(() => source.response);
      const loading = runtime.loadModelFromUrl(MODEL_URL, loadOptions());
      let pushed = await feedChunks(source, chunk, 32);
      assert.equal(pushed, 8, 'reading stops once 8 MiB wait for the cache');
      release();
      pushed = await feedChunks(source, chunk, 32, pushed);
      assert.equal(pushed, 32);
      source.close();
      assert.equal(await loading, 1);
      assert.equal(caches.entries.get(MODEL_URL).length, 32 * chunk.length);
    } finally {
      caches.restore();
    }
  }],

  ['a draft model streams into the cache and keeps the target cache fields', async () => {
    const caches = installCaches({ commitDelayMs: 10 });
    try {
      const source = controlledResponse(4, DRAFT_URL);
      const { runtime } = draftRuntime(source, 1 << 30);
      const progress = nextProgress();
      const loading = runtime.loadDraftModel(DRAFT_URL, { progressCallback: progress.callback });
      source.push([1, 2]);
      assert.deepEqual(await progress.next(), { loaded: 2, total: 4 });
      source.push([3, 4]);
      source.close();
      await loading;
      assert.deepEqual([...caches.entries.get(DRAFT_URL)], [1, 2, 3, 4]);
      assert.deepEqual([runtime._modelSource, runtime._modelCacheState], ['cache', 'hit']);
    } finally {
      caches.restore();
    }
  }],

  ['a draft rejected before its download is read cancels the download and caches nothing', async () => {
    const caches = installCaches();
    try {
      const source = controlledResponse(4096, DRAFT_URL);
      const { runtime } = draftRuntime(source, 1024);
      await assert.rejects(runtime.loadDraftModel(DRAFT_URL), /The draft model needs about/);
      await delay(0);
      assert.equal(source.state.cancelled, true);
      assert.equal(caches.entries.size, 0);
    } finally {
      caches.restore();
    }
  }],
];

for (const [name, run] of CASES) {
  try {
    await run();
  } catch (error) {
    error.message = `${name}: ${error.message}`;
    throw error;
  }
}
