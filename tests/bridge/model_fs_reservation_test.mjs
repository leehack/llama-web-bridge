import assert from 'node:assert/strict';
import test from 'node:test';
import { writeResponseToFsFileWithProgress } from '../../js/src/internal/download.ts';

const gib = 1073741824;
function filesystem({ failReservation = false, shortReservation = false } = {}) {
  return {
    length: 0, capacity: 0, exists: false, opens: [], writes: [], closes: 0,
    readdir() { return []; }, mkdir() {}, unlink() { this.exists = false; this.length = 0; },
    open(path, mode) { this.opens.push(mode); this.exists = true; if (mode === 'w') this.length = 0; return {}; },
    close() { this.closes++; }, writeFile() { throw new Error('Unexpected non-stream write'); },
    write(_stream, _bytes, _offset, length, position) {
      const positionType = typeof position;
      position = Number(position);
      this.writes.push({ length, position, positionType });
      if (length === 1 && position >= gib - 1) {
        if (failReservation) throw { excPtr: 123 };
        if (shortReservation) return 0;
        this.capacity = position + 1;
      }
      if (position + length > gib && this.capacity < position + length) throw new Error('Geometric growth');
      this.length = Math.max(this.length, position + length);
      return length;
    },
  };
}
function response({ bytes = 3, headers = { 'content-length': String(gib), 'content-encoding': 'identity' }, status = 200, fail = false, abort } = {}) {
  const state = { cancelled: false };
  const chunk = new Uint8Array(Math.min(bytes, 1048576));
  let remaining = bytes;
  const body = new ReadableStream({
    pull(controller) {
      if (abort) { abort.abort(); return; }
      if (fail) { controller.error(new Error('network error')); return; }
      if (!remaining) { controller.close(); return; }
      const size = Math.min(remaining, chunk.length);
      controller.enqueue(chunk.subarray(0, size)); remaining -= size;
    },
    cancel() { state.cancelled = true; },
  }, { highWaterMark: 0 });
  return { value: new Response(body, { headers, status }), state };
}

test('known large decoded length reserves capacity and restores logical length before streaming', async () => {
  const fs = filesystem();
  const { value } = response({ bytes: gib + 65536, headers: { 'content-length': String(gib + 65536) } });
  assert.equal(await writeResponseToFsFileWithProgress(value, fs, '/model'), gib + 65536);
  assert.deepEqual(fs.opens, ['w', 'w']);
  assert.deepEqual(fs.writes[0], { length: 1, position: gib + 65535, positionType: 'number' });
  assert.equal(fs.length, gib + 65536);
  assert.equal(fs.closes, 2);
});

test('memory64 reserves capacity above 2 GiB with a BigInt file position', async () => {
  const fs = filesystem();
  const length = 3 * gib + 65536;
  const { value } = response({
    bytes: length, headers: { 'content-length': String(length), 'content-encoding': 'identity' },
  });
  const options = { useBigIntPosition: true, memory64ReservationLimitBytes: 4 * gib };
  assert.equal(await writeResponseToFsFileWithProgress(value, fs, '/model', null, options), length);
  assert.deepEqual(fs.writes[0], { length: 1, position: length - 1, positionType: 'bigint' });
  assert.deepEqual(fs.opens, ['w', 'w']);
  assert.equal(fs.length, length);
});

for (const [name, headers, options, status] of [
  ['encoded', { 'content-length': String(gib), 'content-encoding': 'gzip' }, {}, 200],
  ['unknown', {}, {}, 200],
  ['small', { 'content-length': '3' }, {}, 200],
  ['oversized', { 'content-length': '2147483648' }, {}, 200],
  ['unsafe', { 'content-length': '9007199254740992' }, {}, 200],
  ['fractional', { 'content-length': '1073741824.5' }, {}, 200],
  ['resumed', { 'content-length': String(gib) }, { startOffset: 4 }, 200],
  ['range', { 'content-length': String(gib), 'content-range': 'bytes 0-2/1073741824' }, {}, 206],
  ['memory64 without a heap limit', { 'content-length': String(gib) }, { useBigIntPosition: true }, 200],
  ['memory64 beyond heap headroom', { 'content-length': String(3 * gib) },
    { useBigIntPosition: true, memory64ReservationLimitBytes: 2 * gib }, 200],
  ['JS-backed memory64 fallback', { 'content-length': String(3 * gib) },
    { useBigIntPosition: true, memory64ReservationLimitBytes: 4 * gib, skipMemoryReservation: true }, 200],
]) test(`${name} responses preserve incremental streaming`, async () => {
  const fs = filesystem();
  const { value } = response({ headers, status });
  assert.equal(await writeResponseToFsFileWithProgress(value, fs, '/model', null, options), (options.startOffset || 0) + 3);
  assert.equal(fs.writes.length, 1);
  assert.equal(fs.writes[0].length, 3);
});

for (const bytes of [3, gib + 1]) test(`declared length mismatch (${bytes}) is rejected and removed`, async () => {
  const fs = filesystem(); const { value, state } = response({ bytes });
  await assert.rejects(writeResponseToFsFileWithProgress(value, fs, '/model', null, { preservePartialOnError: true }), { name: 'ModelTransferLengthError' });
  assert.equal(fs.exists, false); if (bytes > gib) assert.equal(state.cancelled, true); assert.equal(fs.closes, 2);
});

for (const option of ['failReservation', 'shortReservation']) test(`${option} falls back without padding or rejecting an uncertain length`, async () => {
  const fs = filesystem({ [option]: true });
  const { value } = response({ headers: { 'content-length': String(gib) } });
  assert.equal(await writeResponseToFsFileWithProgress(value, fs, '/model', null, { preservePartialOnError: true }), 3);
  assert.equal(fs.length, 3); assert.equal(fs.exists, true); assert.equal(fs.closes, 2);
});

test('hidden encoding size hints do not enforce wire length on a decoded body', async () => {
  const fs = filesystem();
  const { value } = response({ headers: { 'content-length': String(gib) } });
  assert.equal(await writeResponseToFsFileWithProgress(value, fs, '/model'), 3);
  assert.equal(fs.length, 3);
});

test('cancellation during the first read cleans reserved storage and later loads work', async () => {
  const fs = filesystem(); const abort = new AbortController(); const { value, state } = response({ abort });
  await assert.rejects(writeResponseToFsFileWithProgress(value, fs, '/model', null, { signal: abort.signal, chunkTimeoutMs: 5 }), { name: 'AbortError' });
  assert.equal(fs.exists, false); assert.equal(state.cancelled, true);
  const next = response({ headers: { 'content-length': '3' } });
  assert.equal(await writeResponseToFsFileWithProgress(next.value, fs, '/model'), 3);
  assert.equal(fs.length, 3);
});

test('a failed first resumed read preserves the already downloaded prefix', async () => {
  const fs = filesystem(); fs.length = 4; fs.exists = true;
  const { value } = response({ fail: true });
  await assert.rejects(writeResponseToFsFileWithProgress(value, fs, '/model', null, { startOffset: 4, preservePartialOnError: true }), /network error/);
  assert.equal(fs.exists, true); assert.equal(fs.length, 4); assert.equal(fs.writes.length, 0);
});

test('progress size estimates never authorize a large allocation', async () => {
  const fs = filesystem(); const { value } = response({ headers: {} });
  assert.equal(await writeResponseToFsFileWithProgress(value, fs, '/model', null, { totalBytes: gib }), 3);
  assert.equal(fs.writes.length, 1); assert.equal(fs.writes[0].length, 3);
});
