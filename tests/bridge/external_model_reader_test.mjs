import assert from 'node:assert/strict';
import test from 'node:test';

import { blobModelReader, cachePartsModelReader, opfsModelReader } from '../../js/src/internal/external_model_reader.ts';
import { externalReadFailure, openCachedExternalModel } from '../../js/src/internal/external_model_source.ts';

test('Blob reader reuses a bounded header window and detaches slices on consumption', async () => {
  const windows = [];
  const buffers = [];
  const blob = {
    size: 48 * 1024 * 1024,
    slice(start, end) {
      windows.push([start, end]);
      return { arrayBuffer: async () => {
        const bytes = new Uint8Array(end - start);
        bytes.fill(7);
        buffers.push(bytes.buffer);
        return bytes.buffer;
      } };
    },
  };
  const reader = blobModelReader(blob, null);
  const check = (bytes) => assert.equal(bytes[0], 7);
  await reader.read(10, 4, check);
  await reader.read(20, 4, check);
  assert.deepEqual(windows, [[10, 10 + 4 * 1024 * 1024]]);
  await reader.read(5 * 1024 * 1024, 16 * 1024 * 1024, check);
  assert.equal(buffers[1].byteLength, 0);
  assert.deepEqual(windows[1], [5 * 1024 * 1024, 21 * 1024 * 1024]);
  reader.close();
  assert.equal(buffers[0].byteLength, 0);
});

test('OPFS reader preserves >4 GiB offsets, reuses a private bounce buffer, and detects a short read', () => {
  const offsets = [];
  const buffers = [];
  let closeCount = 0;
  const size = 2 ** 32 + 64;
  const handle = {
    read(buffer, { at }) {
      offsets.push(at);
      buffers.push(buffer.buffer);
      assert.ok(buffer.buffer instanceof ArrayBuffer);
      buffer.fill(7);
      return at === size - 6 ? buffer.byteLength - 1 : buffer.byteLength;
    },
    close() { ++closeCount; },
  };
  const reader = opfsModelReader(handle, size, null);
  const check = (bytes) => assert.equal(bytes[0], 7);
  reader.read(size - 4, 4, check);
  reader.read(size - 5, 5, check);
  assert.throws(() => reader.read(size - 6, 6, check), /short read/);
  assert.deepEqual(offsets, [size - 4, size - 5, size - 6]);
  assert.equal(buffers[0], buffers[1]);
  assert.equal(buffers[1], buffers[2]);
  reader.close();
  reader.close();
  assert.equal(closeCount, 1);
  assert.equal(buffers[0].byteLength, 0);
});

test('cache part reader serves reads across part boundaries from one resident part and detaches it', async () => {
  const partBytes = 16;
  const decodedBytes = 40;
  const loaded = [];
  const buffers = [];
  const parts = {
    decodedBytes,
    partBytes,
    async readPart(index) {
      loaded.push(index);
      const length = index === 2 ? decodedBytes - 2 * partBytes : partBytes;
      const bytes = Uint8Array.from({ length }, (_, at) => index * partBytes + at);
      buffers.push(bytes.buffer);
      return bytes.buffer;
    },
  };
  const reader = cachePartsModelReader(parts, null);
  assert.equal(reader.size, decodedBytes);
  const seen = [];
  const take = (bytes) => seen.push(Array.from(bytes));
  await reader.read(2, 4, take);
  await reader.read(6, 4, take);
  assert.deepEqual(loaded, [0]);
  await reader.read(14, 6, take);
  assert.deepEqual(seen.at(-1), [14, 15, 16, 17, 18, 19]);
  assert.equal(buffers[0].byteLength, 0, 'a part is detached once the next one replaces it');
  await reader.read(32, 8, take);
  assert.deepEqual(seen.at(-1), [32, 33, 34, 35, 36, 37, 38, 39]);
  reader.close();
  assert.ok(buffers.every((buffer) => buffer.byteLength === 0));
  await assert.rejects(async () => reader.read(0, 1, take), /closed/);
});

test('cache part reader reports a damaged part to the native read', async () => {
  const reader = cachePartsModelReader({
    decodedBytes: 16, partBytes: 16,
    readPart: async () => { throw new Error('part is damaged'); },
  }, null);
  await assert.rejects(reader.read(0, 4, () => assert.fail('no bytes')), /part is damaged/);
});

test('a cached model is read from its Cache Storage entry only where a whole body is safe', async () => {
  const model = new Blob([new Uint8Array(1024).fill(9)]);
  const requests = [];
  const cache = {
    async match(request) {
      requests.push(String(request));
      return String(request) === 'https://models.test/model.gguf' ? new Response(model) : undefined;
    },
  };
  const original = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  Object.defineProperty(globalThis, 'caches', { value: { open: async () => cache }, configurable: true });
  try {
    const whole = await openCachedExternalModel('models', 'https://models.test/model.gguf',
      { worker: false, wholeBody: true, signal: null });
    assert.equal(whole.kind, 'blob_cache');
    assert.equal(whole.reader.size, 1024);
    whole.reader.close();
    assert.equal(await openCachedExternalModel('models', 'https://models.test/model.gguf',
      { worker: false, wholeBody: false, signal: null }), null);
    assert.equal(await openCachedExternalModel('models', 'https://models.test/other.gguf',
      { worker: false, wholeBody: true, signal: null }), null);
  } finally {
    if (original) Object.defineProperty(globalThis, 'caches', original);
    else delete globalThis.caches;
  }
});

test('the first recorded read failure of any reader is the one reported', () => {
  const failure = { code: 5, error: new Error('first') };
  assert.equal(externalReadFailure([{}, { failure }, { failure: { code: 6, error: 'later' } }]), failure);
  assert.equal(externalReadFailure([{}, {}]), null);
});
