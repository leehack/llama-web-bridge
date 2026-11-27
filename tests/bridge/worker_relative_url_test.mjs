import assert from 'node:assert/strict';
import test from 'node:test';

import { CALLER_MODEL_SOURCE_URL } from '../../js/src/internal/caller_model_source.ts';
import { normalizeAbsoluteUrl, resolveModelSource } from '../../js/src/internal/model_source.ts';
import { createWorkerBridge } from './bridge_operation_queue_fixtures.mjs';

// A page at a nested route whose <base href> points elsewhere, as a Flutter
// app with path URL strategy has.
async function withPage(run, { baseURI = 'https://example.test/app/' } = {}) {
  globalThis.window = { location: { href: 'https://example.test/app/chat/42' } };
  if (baseURI !== null) {
    globalThis.document = { baseURI };
  }
  try {
    return await run();
  } finally {
    delete globalThis.window;
    delete globalThis.document;
  }
}

function recordingWorkerBridge() {
  const calls = [];
  const bridge = createWorkerBridge({
    _callWorker: async (method, args) => {
      calls.push({ method, args: structuredClone(args) });
      return method === 'createCompletion' ? 'x' : 1;
    },
  });
  return { bridge, calls };
}

const argsOf = (calls, method) => calls.filter((call) => call.method === method).map((call) => call.args);

test('worker model, shard and projector URLs resolve against the page base URL', async () => {
  const { bridge, calls } = recordingWorkerBridge();
  await withPage(async () => {
    await bridge.loadModelFromUrl('models/a.gguf');
    await bridge.loadModelFromUrl(['shards/m-00001-of-00002.gguf', 'shards/m-00002-of-00002.gguf']);
    await bridge.loadModelFromUrl('models/m-00001-of-00002.gguf');
    await bridge.loadMultimodalProjector('models/mmproj.gguf');
  });

  assert.deepEqual(argsOf(calls, 'loadModelFromUrl').map(([source]) => source), [
    'https://example.test/app/models/a.gguf',
    ['https://example.test/app/shards/m-00001-of-00002.gguf', 'https://example.test/app/shards/m-00002-of-00002.gguf'],
    // One URL stays a string, so the worker still expands the split name.
    'https://example.test/app/models/m-00001-of-00002.gguf',
  ]);
  assert.deepEqual(argsOf(calls, 'loadMultimodalProjector').at(-1), ['https://example.test/app/models/mmproj.gguf']);
  assert.equal(bridge._loadedMmProjUrl, 'https://example.test/app/models/mmproj.gguf');

  // A worker restart reloads the remembered sources outside the page context.
  calls.length = 0;
  await bridge._loadRememberedModelIntoWorker({});
  assert.deepEqual(calls.map((call) => call.args[0]), [
    'https://example.test/app/models/m-00001-of-00002.gguf',
    'https://example.test/app/models/mmproj.gguf',
  ]);
});

test('worker completion media URLs resolve against the page base URL', async () => {
  const { bridge, calls } = recordingWorkerBridge();
  const bytes = new Uint8Array([1, 2, 3]);
  const parts = [
    { type: 'image', url: 'images/cat.png' },
    { type: 'audio', url: 'https://cdn.example/a.wav' },
    { type: 'image', bytes },
  ];
  await withPage(() => bridge.createCompletion('hello', { parts }));

  const [, options] = argsOf(calls, 'createCompletion').at(-1);
  assert.deepEqual(options.parts.map((part) => part.url), [
    'https://example.test/app/images/cat.png',
    'https://cdn.example/a.wav',
    undefined,
  ]);
  assert.deepEqual([...options.parts[2].bytes], [1, 2, 3]);
  assert.equal(parts[0].url, 'images/cat.png', "the caller's parts are not modified");
});

test('worker embedding media URLs resolve against the page base URL', async () => {
  const { bridge, calls } = recordingWorkerBridge();
  bridge._callWorker = async (method, args) => {
    calls.push({ method, args: structuredClone(args) });
    if (method === 'getEmbeddingCapabilities') return { apiVersion: 1, text: true, image: true, audio: true };
    return method === 'embedBatch' ? [[0]] : [0];
  };
  const parts = [{ type: 'text', text: 'a' }, { type: 'image', url: 'images/cat.png' }];
  await withPage(async () => {
    await bridge.embed(parts);
    await bridge.embedBatch(['plain', parts]);
  });

  assert.deepEqual(argsOf(calls, 'embed').at(-1)[0], [
    { type: 'text', text: 'a' },
    { type: 'image', url: 'https://example.test/app/images/cat.png' },
  ]);
  const [batch] = argsOf(calls, 'embedBatch').at(-1);
  assert.equal(batch[0], 'plain');
  assert.equal(batch[1][1].url, 'https://example.test/app/images/cat.png');
  assert.equal(parts[1].url, 'images/cat.png', "the caller's parts are not modified");
});

test('without a base URI the page location applies, and outside a page URLs stay as given', async () => {
  await withPage(() => {
    assert.equal(normalizeAbsoluteUrl('models/a.gguf'), 'https://example.test/app/chat/models/a.gguf');
  }, { baseURI: null });

  // A worker has neither window nor document, so it must not invent a base.
  assert.equal(normalizeAbsoluteUrl('models/a.gguf'), 'models/a.gguf');
  assert.equal(resolveModelSource('  models/a.gguf '), 'models/a.gguf');
});

test('resolving a model source never drops, joins or rewrites non-URL shard entries', async () => {
  await withPage(() => {
    assert.equal(resolveModelSource(''), '');
    assert.equal(resolveModelSource('blob:https://example.test/1'), 'blob:https://example.test/1');
    assert.deepEqual(
      resolveModelSource(['a.gguf', '', null, 'https://cdn.example/b.gguf']),
      ['https://example.test/app/a.gguf', '', null, 'https://cdn.example/b.gguf'],
    );
  });
});

test('a caller source reaches the worker by reference and a worker restart reloads the same one', async () => {
  const { bridge, calls } = recordingWorkerBridge();
  const blob = new Blob([new Uint8Array(16)]);
  await bridge.loadModelFromSource(blob, { nCtx: 256 });

  const [url, options] = argsOf(calls, 'loadModelFromUrl').at(-1);
  assert.equal(url, CALLER_MODEL_SOURCE_URL);
  assert.equal(options.callerModelSource.size, 16);
  assert.equal(bridge._loadedModelOptions.callerModelSource.source, blob, 'the facade keeps the caller reference');

  calls.length = 0;
  await bridge._loadRememberedModelIntoWorker(bridge._sanitizeModelLoadOptions(bridge._loadedModelOptions));
  const [replayUrl, replayOptions] = argsOf(calls, 'loadModelFromUrl').at(-1);
  assert.equal(replayUrl, CALLER_MODEL_SOURCE_URL);
  assert.equal(replayOptions.callerModelSource.size, 16);
  assert.equal(replayOptions.nCtx, 256);
});

test('the facade rejects an invalid source before it reaches a worker and forgets the source with the model', async () => {
  const { bridge, calls } = recordingWorkerBridge();
  await assert.rejects(bridge.loadModelFromSource('https://x/model.gguf'), (error) => error.code === 'MODEL_SOURCE_INVALID');
  await assert.rejects(bridge.loadModelFromSource(new Blob([])), (error) => error.code === 'MODEL_SOURCE_INVALID');
  assert.equal(argsOf(calls, 'loadModelFromUrl').length, 0);

  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    bridge.loadModelFromSource(new Blob([new Uint8Array(4)]), { signal: controller.signal }),
    (error) => error.name === 'AbortError',
  );

  await bridge.loadModelFromSource(new Blob([new Uint8Array(4)]));
  bridge._forgetLoadedModel();
  assert.equal(bridge._loadedModelOptions, null);
});

test('a worker script without caller-source support is refused before dispatch', async () => {
  const { BridgeWorkerProxy } = await import('../../js/src/worker_proxy.ts');
  const proxy = Object.create(BridgeWorkerProxy.prototype);
  proxy._supportsCallerModelSource = false;
  proxy._config = {};
  proxy._worker = { postMessage() { assert.fail('the request must not be posted'); } };
  proxy._supportsCorePreflight = true;
  await assert.rejects(
    proxy.call('loadModelFromUrl', [CALLER_MODEL_SOURCE_URL, { callerModelSource: { size: 1 } }]),
    (error) => error.name === 'ModelStagingError' && error.code === 'MODEL_SOURCE_UNSUPPORTED',
  );
});

test('a source the browser cannot clone to the worker is a typed error and leaves no pending request', async () => {
  const { BridgeWorkerProxy } = await import('../../js/src/worker_proxy.ts');
  const proxy = Object.create(BridgeWorkerProxy.prototype);
  proxy._supportsCallerModelSource = true;
  proxy._supportsCorePreflight = true;
  proxy._config = { workerRequestTimeoutMs: 60000 };
  proxy._pending = new Map();
  proxy._nextId = 1;
  proxy._worker = {
    postMessage() { throw new DOMException('not cloneable', 'DataCloneError'); },
  };
  await assert.rejects(
    proxy.call('loadModelFromUrl', [CALLER_MODEL_SOURCE_URL, { callerModelSource: { size: 1 } }]),
    (error) => error.name === 'ModelStagingError' && error.code === 'MODEL_SOURCE_INVALID',
  );
  assert.equal(proxy._pending.size, 0);

  proxy._worker.postMessage = () => { throw new DOMException('other', 'DataCloneError'); };
  await assert.rejects(
    proxy.call('loadModelFromUrl', ['https://x/model.gguf', {}]),
    (error) => error.name === 'DataCloneError',
  );
});
