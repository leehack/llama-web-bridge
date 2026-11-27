import assert from 'node:assert/strict';
import test from 'node:test';

import {
  describeCallerModelSource,
  fingerprintRanges,
  fingerprintSource,
  inspectCallerModel,
  isCallerModelSource,
  openCallerModelSource,
} from '../../js/src/internal/caller_model_source.ts';
import { blobModelReader } from '../../js/src/internal/external_model_reader.ts';

const u32 = (value) => { const b = Buffer.alloc(4); b.writeUInt32LE(value); return b; };
const u64 = (value) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(value)); return b; };
const string = (value) => Buffer.concat([u64(Buffer.byteLength(value)), Buffer.from(value)]);

function gguf(entries) {
  return Buffer.concat([
    Buffer.from('GGUF'), u32(3), u64(0), u64(entries.length),
    ...entries.flatMap(([key, type, value]) => [string(key), u32(type), value]),
  ]);
}

const stringArray = (count) => Buffer.concat([
  u32(8), u64(count), ...Array.from({ length: count }, (_, i) => string(`token-${i}`)),
]);
const floatArray = (count) => Buffer.concat([u32(6), u64(count), Buffer.alloc(4 * count)]);

const reader = (bytes) => blobModelReader(new Blob([bytes]), null);

async function code(promise) {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  assert.fail('expected a rejection');
}

const opfsHandles = new WeakSet();
Object.defineProperty(globalThis.navigator, 'storage', {
  configurable: true,
  value: { getDirectory: async () => ({ resolve: async (handle) => (opfsHandles.has(handle) ? ['model.gguf'] : null) }) },
});

function fileHandle({
  opfs = true,
  size = 64,
  lastModified = 1000,
  syncSize = size,
  syncError = null,
  getFileError = null,
} = {}) {
  const calls = { sync: 0, syncClosed: 0, getFile: 0 };
  let bytes = Uint8Array.from({ length: size }, (_, index) => (index * 31 + 7) % 251);
  let currentModified = lastModified;
  const handle = {
    kind: 'file',
    name: 'model.gguf',
    async getFile() {
      calls.getFile += 1;
      if (getFileError) throw getFileError;
      return new File([bytes.slice()], 'model.gguf', { lastModified: currentModified });
    },
    async createSyncAccessHandle() {
      calls.sync += 1;
      if (syncError) throw syncError;
      return {
        getSize: () => syncSize,
        read: (buffer, { at }) => {
          const chunk = bytes.subarray(at, at + buffer.length);
          buffer.set(chunk);
          return chunk.length;
        },
        close: () => { calls.syncClosed += 1; },
      };
    },
  };
  if (opfs) opfsHandles.add(handle);
  return {
    handle,
    calls,
    set size(value) { bytes = Uint8Array.from({ length: value }, (_, index) => bytes[index] ?? 0); },
    set modified(value) { currentModified = value; },
    // A write in place that keeps the size and the modification time, as
    // Chromium's in-memory OPFS does for a sync access handle or keepExistingData.
    editInPlace(position) { bytes[position] ^= 0xff; },
  };
}

const named = (name) => new DOMException(name, name);

test('only a Blob, a File or a file handle describes a source', async () => {
  assert.equal((await describeCallerModelSource(new Blob(['abcd']))).size, 4);
  const file = new File(['abcd'], 'a.gguf', { lastModified: 77 });
  assert.deepEqual(
    { size: (await describeCallerModelSource(file)).size, modified: (await describeCallerModelSource(file)).lastModified },
    { size: 4, modified: 77 },
  );
  const { handle } = fileHandle({ size: 9, lastModified: 5 });
  const described = await describeCallerModelSource(handle);
  assert.deepEqual([described.size, described.lastModified, described.source === handle], [9, 5, true]);
  assert.match(described.fingerprint, /^[0-9a-f]{16}$/);
  assert.equal((await describeCallerModelSource(file)).fingerprint, null);
  assert.ok(isCallerModelSource(described));
  assert.equal(isCallerModelSource({ ...described, fingerprint: undefined }), false);

  for (const bad of [null, 'https://x/model.gguf', new Uint8Array(4), { kind: 'directory', getFile() {} }, {}]) {
    assert.equal((await code(describeCallerModelSource(bad))).code, 'MODEL_SOURCE_INVALID');
  }
  assert.equal((await code(describeCallerModelSource(new Blob([])))).code, 'MODEL_SOURCE_INVALID');
  assert.equal((await code(describeCallerModelSource(fileHandle({ size: 0 }).handle))).code, 'MODEL_SOURCE_INVALID');
});

test('a handle that cannot be read is invalid, and one that is gone has changed', async () => {
  const denied = fileHandle({ getFileError: named('NotAllowedError') }).handle;
  assert.equal((await code(describeCallerModelSource(denied))).code, 'MODEL_SOURCE_INVALID');
  const gone = fileHandle({ getFileError: named('NotFoundError') }).handle;
  assert.equal((await code(describeCallerModelSource(gone))).code, 'MODEL_SOURCE_CHANGED');
});

test('a Blob source reads on demand and a changed File is refused', async () => {
  const file = new File([new Uint8Array(32)], 'm.gguf', { lastModified: 10 });
  const description = await describeCallerModelSource(file);
  const opened = await openCallerModelSource(description, { worker: true, signal: null });
  assert.deepEqual([opened.kind, opened.reader.mode, opened.reader.size], ['caller_file', 1, 32]);
  opened.reader.close();

  const blobOpened = await openCallerModelSource(await describeCallerModelSource(new Blob([new Uint8Array(8)])), {
    worker: false, signal: null,
  });
  assert.equal(blobOpened.kind, 'caller_blob');

  const sizeChanged = await code(openCallerModelSource({ ...description, size: 33 }, { worker: false, signal: null }));
  assert.equal(sizeChanged.code, 'MODEL_SOURCE_CHANGED');
  const timeChanged = await code(openCallerModelSource({ ...description, lastModified: 11 }, { worker: false, signal: null }));
  assert.equal(timeChanged.code, 'MODEL_SOURCE_CHANGED');
});

test('a worker reads an OPFS handle through its own sync access handle and closes only that', async () => {
  const fixture = fileHandle();
  const description = await describeCallerModelSource(fixture.handle);
  const opened = await openCallerModelSource(description, { worker: true, signal: null });
  assert.deepEqual([opened.kind, opened.reader.mode, opened.reader.size], ['caller_opfs_sync', 0, 64]);
  assert.equal(fixture.calls.sync, 1);
  await opened.verify();
  opened.reader.close();
  assert.equal(fixture.calls.syncClosed, 1);
  assert.deepEqual(Object.keys(fixture.handle).sort(), ['createSyncAccessHandle', 'getFile', 'kind', 'name']);
});

test('the main thread and a handle without a sync access handle read the live file', async () => {
  const main = fileHandle();
  const description = await describeCallerModelSource(main.handle);
  const direct = await openCallerModelSource(description, { worker: false, signal: null });
  assert.deepEqual([direct.kind, direct.reader.mode, main.calls.sync], ['caller_opfs_blob', 1, 0]);
  await direct.verify();

  const picker = fileHandle({ opfs: false, syncError: named('NotAllowedError') });
  const fallback = await openCallerModelSource(await describeCallerModelSource(picker.handle), {
    worker: true, signal: null,
  });
  assert.deepEqual([fallback.kind, picker.calls.sync], ['caller_opfs_blob', 1]);
});

test('a worker refuses a file another sync access handle holds instead of reading it live', async () => {
  // Chromium and Firefox name the refusal NoModificationAllowedError, WebKit InvalidStateError.
  for (const name of ['NoModificationAllowedError', 'InvalidStateError', 'NotAllowedError', 'TypeError']) {
    const busy = fileHandle({ syncError: named(name) });
    const description = await describeCallerModelSource(busy.handle);
    const getFiles = busy.calls.getFile;
    const error = await code(openCallerModelSource(description, { worker: true, signal: null }));
    assert.equal(error.code, 'MODEL_SOURCE_BUSY', name);
    assert.ok(error.message.includes(name), error.message);
    assert.equal(busy.calls.getFile, getFiles, `${name} read the live file`);
  }
});

test('a handle outside OPFS that refuses a sync access handle is read live, and a name alone is not enough for BUSY', async () => {
  const picker = fileHandle({ opfs: false, syncError: named('InvalidStateError') });
  const opened = await openCallerModelSource(await describeCallerModelSource(picker.handle), {
    worker: true, signal: null,
  });
  assert.equal(opened.kind, 'caller_opfs_blob');
  const outsideOnlyByName = fileHandle({ opfs: false, syncError: named('NoModificationAllowedError') });
  const error = await code(openCallerModelSource(await describeCallerModelSource(outsideOnlyByName.handle), {
    worker: true, signal: null,
  }));
  assert.equal(error.code, 'MODEL_SOURCE_BUSY');
});

test('a handle whose file changed is refused and the sync access handle is released', async () => {
  const fixture = fileHandle();
  const description = await describeCallerModelSource(fixture.handle);

  fixture.size = 65;
  const resized = await code(openCallerModelSource(description, { worker: true, signal: null }));
  assert.equal(resized.code, 'MODEL_SOURCE_CHANGED');
  assert.equal(fixture.calls.syncClosed, 1);

  fixture.size = 64;
  fixture.modified = 2000;
  const touched = await code(openCallerModelSource(description, { worker: true, signal: null }));
  assert.equal(touched.code, 'MODEL_SOURCE_CHANGED');
  assert.equal(fixture.calls.syncClosed, 2);

  const racing = fileHandle({ syncSize: 63 });
  const raced = await code(openCallerModelSource(await describeCallerModelSource(racing.handle), { worker: true, signal: null }));
  assert.equal(raced.code, 'MODEL_SOURCE_CHANGED');
  assert.equal(racing.calls.syncClosed, 1);

  for (const worker of [true, false]) {
    const inPlace = fileHandle();
    const inPlaceDescription = await describeCallerModelSource(inPlace.handle);
    inPlace.editInPlace(40);
    const edited = await code(openCallerModelSource(inPlaceDescription, { worker, signal: null }));
    assert.equal(edited.code, 'MODEL_SOURCE_CHANGED');
    assert.equal(inPlace.calls.syncClosed, worker ? 1 : 0);
  }

  const removed = fileHandle();
  const removedDescription = await describeCallerModelSource(removed.handle);
  removed.handle.getFile = async () => { throw named('NotFoundError'); };
  const missing = await code(openCallerModelSource(removedDescription, { worker: true, signal: null }));
  assert.equal(missing.code, 'MODEL_SOURCE_CHANGED');
  assert.equal(removed.calls.syncClosed, 1);
});

test('a file that is not a GGUF is invalid', async () => {
  const error = await code(inspectCallerModel(reader(Buffer.from('not a gguf file at all'))));
  assert.equal(error.code, 'MODEL_SOURCE_INVALID');
});

test('a split GGUF is a typed capability error wherever its split metadata sits', async () => {
  const split = (count) => ['split.count', 2, Buffer.from([count, 0])];
  const early = gguf([split(3), ['general.architecture', 8, string('llama')]]);
  assert.equal((await code(inspectCallerModel(reader(early)))).code, 'MODEL_SOURCE_SPLIT_UNSUPPORTED');

  const afterTokens = gguf([
    ['tokenizer.ggml.tokens', 9, stringArray(40_000)],
    ['tokenizer.ggml.scores', 9, floatArray(40_000)],
    ['general.name', 8, string('x')],
    split(5),
  ]);
  assert.ok(afterTokens.length > 4 * 1024 * 1024 / 8);
  assert.equal((await code(inspectCallerModel(reader(afterTokens)))).code, 'MODEL_SOURCE_SPLIT_UNSUPPORTED');

  const uint32 = gguf([['split.count', 4, u32(2)]]);
  assert.equal((await code(inspectCallerModel(reader(uint32)))).code, 'MODEL_SOURCE_SPLIT_UNSUPPORTED');
});

test('a single-file GGUF passes, with or without split metadata', async () => {
  await inspectCallerModel(reader(gguf([['general.architecture', 8, string('llama')]])));
  await inspectCallerModel(reader(gguf([['split.count', 2, Buffer.from([1, 0])], ['split.no', 2, Buffer.from([0, 0])]])));
  await inspectCallerModel(reader(gguf([
    ['a', 7, Buffer.from([1])], ['b', 10, u64(5)], ['c', 12, Buffer.alloc(8)], ['d', 9, Buffer.concat([u32(9), u64(0)])],
  ])));
});

test('a truncated GGUF is invalid and a read that throws while scanning is a read failure', async () => {
  const bytes = gguf([['tokenizer.ggml.tokens', 9, stringArray(2000)]]);
  const truncated = blobModelReader(new Blob([bytes.subarray(0, bytes.length - 100)]), null);
  assert.equal((await code(inspectCallerModel(truncated))).code, 'MODEL_SOURCE_INVALID');

  const source = blobModelReader(new Blob([bytes]), null);
  const failing = { ...source, read: () => { throw new Error('injected'); } };
  assert.equal((await code(inspectCallerModel(failing))).code, 'MODEL_READ_FAILED');
  const recorded = {
    ...source,
    failure: { code: 5, error: new Error('recorded') },
    read: (offset, length, copy) => source.read(offset, length, copy),
  };
  assert.equal((await code(inspectCallerModel(recorded))).code, 'MODEL_READ_FAILED');
});

test('a cancelled read is not reported as a failed model', async () => {
  const controller = new AbortController();
  const bytes = gguf([['general.architecture', 8, string('llama')]]);
  controller.abort();
  const error = await code(inspectCallerModel(blobModelReader(new Blob([bytes]), controller.signal)));
  assert.equal(error.name, 'AbortError');
});

test('an edit after the open is caught when the loader rechecks the source', async () => {
  for (const worker of [true, false]) {
    const fixture = fileHandle();
    const opened = await openCallerModelSource(await describeCallerModelSource(fixture.handle), {
      worker, signal: null,
    });
    await opened.verify();
    fixture.editInPlace(10);
    assert.equal((await code(opened.verify())).code, 'MODEL_SOURCE_CHANGED');
    opened.reader.close();
  }
  const resized = fileHandle();
  const opened = await openCallerModelSource(await describeCallerModelSource(resized.handle), { worker: false, signal: null });
  resized.size = 65;
  assert.equal((await code(opened.verify())).code, 'MODEL_SOURCE_CHANGED');
  const file = new File([new Uint8Array(8)], 'm.gguf', { lastModified: 3 });
  await (await openCallerModelSource(await describeCallerModelSource(file), { worker: false, signal: null })).verify();
});

const MIB = 1024 * 1024;
const patterned = (offset) => (offset * 131 + 17) % 251;
const patternReader = (edits = new Set()) => async (offset, length) => {
  const bytes = new Uint8Array(length);
  for (let index = 0; index < length; index += 1) bytes[index] = edits.has(offset + index) ? 0xff : patterned(offset + index);
  return bytes;
};

test('a fingerprint samples a bounded number of bytes whatever the file size', async () => {
  for (const size of [1, 1000, 64 * 1024, 1.25 * MIB, 40 * MIB, 3 * 1024 * MIB]) {
    const ranges = fingerprintRanges(size);
    const sampled = ranges.reduce((sum, [start, end]) => sum + end - start, 0);
    assert.ok(sampled <= 18 * 64 * 1024, `${size}: ${sampled}`);
    assert.equal(ranges[0][0], 0);
    assert.equal(ranges.at(-1)[1], size);
  }
  assert.deepEqual(fingerprintRanges(1000), [[0, 1000]]);
  assert.deepEqual(fingerprintRanges(MIB), [[0, MIB]]);
  assert.ok(fingerprintRanges(2 * 1024 * MIB).length > 10);

  let bytesRead = 0;
  await fingerprintSource(2 * 1024 * MIB, async (offset, length) => {
    bytesRead += length;
    return patternReader()(offset, length);
  });
  assert.ok(bytesRead <= 18 * 64 * 1024);
});

test('a change inside a sampled stripe changes the fingerprint and one in a gap does not', async () => {
  const size = 200 * MIB;
  const ranges = fingerprintRanges(size);
  const baseline = await fingerprintSource(size, patternReader());
  const positions = [
    ranges[0][0],
    ...ranges.map(([start, end]) => Math.floor((start + end) / 2)),
    ranges.at(-1)[1] - 1,
  ];
  for (const position of positions) {
    const edited = await fingerprintSource(size, patternReader(new Set([position])));
    assert.notEqual(edited, baseline, `stripe byte ${position}`);
  }
  // Documented limit: bytes between the stripes are not sampled, so an in-place
  // edit there that keeps the size and the modification time is undetectable.
  const gap = Math.floor((ranges[0][1] + ranges[1][0]) / 2);
  assert.ok(gap > ranges[0][1] && gap < ranges[1][0]);
  assert.equal(await fingerprintSource(size, patternReader(new Set([gap]))), baseline);
});
