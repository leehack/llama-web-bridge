import assert from 'node:assert/strict';
import test from 'node:test';

import * as download from '../../js/src/internal/download.ts';
import { createDirectBridge } from './bridge_operation_queue_fixtures.mjs';

const MODEL = 'https://models.test/model.gguf';
const OTHER = 'https://models.test/other.gguf';
const LAST_MODIFIED = 'Wed, 21 Oct 2015 07:28:00 GMT';
const A = [1, 2, 3, 4];
const B = [5, 6, 7, 8];
const HYBRID = [1, 2, 7, 8];
const RANGE = 'bytes 2-3/4';

test('strong ETags use quoted opaque-tag syntax', () => {
  for (const value of ['"abc"', '""']) {
    assert.equal(download.strongEtagOf(value), value);
  }
  for (const value of ['W/"a"', 'w/"a"', 'abc', '"a", "b"', '\u00a0"a"', '"a', '', null]) {
    assert.equal(download.strongEtagOf(value), null, String(value));
  }
});

for (const [name, verdict, initial, resumed, finalUrl = MODEL, total = 4] of [
  ['different URL', 'changed', { ETag: '"a"' }, { ETag: '"a"' }, OTHER],
  ['changed strong ETag', 'changed', { ETag: '"a"' }, { ETag: '"b"' }],
  ['missing strong ETag', 'unvalidated', { ETag: '"a"' }, {}],
  ['non-HTTP whitespace makes resumed ETag malformed', 'unvalidated', { ETag: '"a"' },
    { ETag: '\u00a0"a"', 'Content-Range': RANGE }],
  ['equal malformed ETags are unvalidated', 'unvalidated', { ETag: '\u00a0"a"' },
    { ETag: '\u00a0"a"', 'Content-Range': RANGE }],
  ['changed weak ETag', 'unvalidated', { ETag: 'W/"a"' }, { ETag: 'W/"b"' }],
  ['changed Last-Modified', 'unvalidated', { 'Last-Modified': LAST_MODIFIED },
    { 'Last-Modified': 'Thu, 22 Oct 2015 07:28:00 GMT' }],
  ['new ETag', 'unvalidated', {}, { ETag: '"a"', 'Content-Range': RANGE }],
  ['wrong start', 'changed', {}, { 'Content-Range': 'bytes 0-3/4' }],
  ['wrong end', 'changed', {}, { 'Content-Range': 'bytes 2-2/4' }],
  ['wrong total', 'changed', {}, { 'Content-Range': 'bytes 2-4/5' }],
  ['wildcard total with known full size', 'unvalidated', {}, { 'Content-Range': 'bytes 2-3/*' }],
  ['wildcard total with a wrong end', 'changed', {}, { 'Content-Range': 'bytes 2-2/*' }],
  ['malformed range', 'unvalidated', {}, { 'Content-Range': 'bytes 2/4' }],
  ['empty range', 'unvalidated', {}, { 'Content-Range': '' }],
  ['hidden range and wrong length', 'changed', {}, { 'Content-Length': '4' }],
  ['hidden range and conflicting linked size', 'changed', {}, { 'Content-Length': '2', 'x-linked-size': '5' }],
  ['visible range and conflicting linked size', 'changed', {}, { 'Content-Range': RANGE, 'x-linked-size': '5' }],
  ['matched strong ETag and range', 'validated', { ETag: '"a"' },
    { ETag: '"a"', 'Content-Range': RANGE }],
  ['matched strong ETag and capitalized range unit', 'validated', { ETag: '"a"' },
    { ETag: '"a"', 'Content-Range': 'Bytes 2-3/4' }],
  ['matched strong ETag and wildcard range with known full size', 'validated', { ETag: '"a"' },
    { ETag: '"a"', 'Content-Range': 'bytes 2-3/*' }],
  ['matched strong ETag and wildcard range without a known full size', 'unvalidated', { ETag: '"a"' },
    { ETag: '"a"', 'Content-Range': 'bytes 2-3/*' }, MODEL, 0],
  ['matched strong ETag and tail length', 'validated', { ETag: '"a"' },
    { ETag: '"a"', 'Content-Length': '2' }],
  ['equal weak ETag', 'unvalidated', { ETag: 'W/"same"' },
    { ETag: 'W/"same"', 'Content-Range': RANGE }],
  ['equal Last-Modified', 'unvalidated', { 'Last-Modified': LAST_MODIFIED },
    { 'Last-Modified': LAST_MODIFIED, 'Content-Range': RANGE }],
  ['no validators', 'unvalidated', {}, { 'Content-Range': RANGE }],
  ['matched strong ETag with hidden range and length', 'unvalidated', { ETag: '"a"' },
    { ETag: '"a"' }],
  ['matched strong ETag with hidden range and invalid length', 'unvalidated', { ETag: '"a"' },
    { ETag: '"a"', 'Content-Length': '2.0' }],
  ['matched strong ETag with hidden range and unknown total', 'unvalidated', { ETag: '"a"' },
    { ETag: '"a"', 'Content-Length': '2' }, MODEL, 0],
]) {
  test(`resume classification: ${name}`, () => {
    const initialHeaders = total > 0 ? { 'Content-Length': String(total), ...initial } : initial;
    const identity = download.captureResumeIdentity(transfer([], { headers: initialHeaders }).response, MODEL);
    const { response } = transfer([], { status: 206, headers: resumed });
    assert.equal(download.classifyResumeResponse(identity, response, finalUrl, 2).verdict, verdict);
  });
}

test('model download: reject a short tail when the resume discloses the full size', async (t) => {
  const initial = transfer(A.slice(0, 2), { cut: true });
  const resumed = transfer([3], { status: 206, headers: { 'x-linked-size': '4' } });
  const { load, loads, requests } = runTransfer(t, [initial, resumed]);
  await assert.rejects(load, { name: 'ModelTransferLengthError' });
  assert.deepEqual(loads, []);
  assert.deepEqual(requests, [fullGet, rangeGet]);
});

function transfer(bytes, { status = 200, headers = {}, url = MODEL, cut = false, type = 'basic' } = {}) {
  let started = false;
  let cancelled = false;
  const body = new ReadableStream({
    pull(controller) {
      if (!started) {
        started = true;
        controller.enqueue(Uint8Array.from(bytes));
        return;
      }
      if (cut) {
        controller.error(new TypeError('network error'));
      } else {
        controller.close();
      }
    },
    cancel() {
      cancelled = true;
    },
  }, { highWaterMark: 0 });
  const response = new Response(body, { status, headers });
  Object.defineProperties(response, { url: { value: url }, type: { value: type } });
  return { response, wasCancelled: () => cancelled };
}

function runTransfer(t, replies, options = {}) {
  const { bridge, core } = createDirectBridge();
  const { _runtime: runtime } = bridge;
  const files = new Map();
  const loads = [];
  core.FS.open = (path, mode) => {
    if (mode === 'w') files.set(path, []);
    assert.ok(files.has(path), `missing file for ${mode}`);
    return path;
  };
  core.FS.write = (path, bytes, offset, length, position) => {
    const file = files.get(path);
    for (let i = 0; i < length; i += 1) file[Number(position) + i] = bytes[offset + i];
    return length;
  };
  core.FS.unlink = (path) => files.delete(path);
  const originalCcall = core.ccall;
  core.ccall = (name, type, types, args) => {
    if (name === 'llamadart_webgpu_load_model') {
      loads.push(files.get(args[0]).slice());
      return 0;
    }
    if (name === 'llamadart_webgpu_get_context_size') return 128;
    if (name === 'llamadart_webgpu_free_model') return 0;
    return originalCcall(name, type, types, args);
  };
  runtime._modelBytes = 0;
  runtime._coreVariant = 'wasm32';
  runtime._probeBackends = async () => {};
  runtime._ensureCore = async () => core;
  runtime._coreSupportsPthreads = () => false;
  runtime._syncThreadPoolSizeHintFromCore = () => {};

  const requests = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    const { response } = replies[requests.length] ?? {};
    requests.push([init.method ?? 'GET', url, [...new Headers(init.headers)]]);
    assert.ok(response, 'unexpected request');
    return response;
  });
  const progress = [];
  const load = runtime.loadModelFromUrl(MODEL, {
    nCtx: 128,
    nThreads: 1,
    nGpuLayers: 0,
    useCache: false,
    allowRemoteFetchBackend: false,
    progressCallback: ({ loaded, total }) => progress.push([loaded, total]),
    ...options,
  });
  return { load, loads, requests, progress, notes: () => runtime._runtimeNotes };
}

const fullGet = ['GET', MODEL, []];
const rangeGet = ['GET', MODEL, [['range', 'bytes=2-']]];
const secondRangeGet = ['GET', MODEL, [['range', 'bytes=3-']]];
const first = (headers) => transfer(A.slice(0, 2), { headers: { 'Content-Length': '4', ...headers }, cut: true });
const tail = (bytes, headers, extra = {}) => transfer(bytes, {
  status: 206, headers: { 'Content-Range': RANGE, ...headers }, ...extra,
});
const fullB = () => transfer(B, { headers: { 'Content-Length': '4' } });
const validated = 'model_stream_resume_validated';
const unvalidated = 'model_stream_resume_unvalidated';

test('model download: rejects a truncated-size claim on the replacement response before native load', async (t) => {
  const replacement = transfer(B, { headers: { 'x-linked-size': '3', 'Content-Length': '4' } });
  const { load, loads, requests } = runTransfer(t, [
    first({ ETag: '"a"' }),
    tail(B.slice(2), { ETag: '"b"' }),
    replacement,
  ]);
  await assert.rejects(load, { name: 'ModelTransferLengthError' });
  assert.deepEqual(loads, []);
  assert.deepEqual(requests, [fullGet, rangeGet, fullGet]);
});

for (const type of ['basic', 'cors']) {
  test(`model download: resumes a decoded ${type} full GET with an identity tail`, async (t) => {
    const headers = { 'Content-Length': '3', ...(type === 'basic' ? { 'Content-Encoding': 'gzip' } : {}) };
    const { load, loads, requests, notes } = runTransfer(t, [
      transfer(A.slice(0, 2), { headers, type, cut: true }),
      tail(A.slice(2), {}, { type }),
    ]);
    await load;
    assert.deepEqual(loads, [A]);
    assert.deepEqual(requests, [fullGet, rangeGet]);
    assert.ok(notes().includes(unvalidated));
  });
}

test('model download: restarts when comparable strong ETags first seen on ranges disagree', async (t) => {
  const firstRange = transfer([3], {
    status: 206, headers: { ETag: '"a"', 'Content-Range': 'bytes 2-3/4' }, cut: true,
  });
  const finalRange = tail(B.slice(3), { ETag: '"b"', 'Content-Range': 'bytes 3-3/4' });
  const { load, loads, requests, notes } = runTransfer(t, [first(), firstRange, finalRange, fullB()]);
  await load;
  assert.deepEqual(loads, [B]);
  assert.deepEqual(requests, [fullGet, rangeGet, secondRangeGet, fullGet]);
  assert.deepEqual(notes().filter((note) => note === validated || note === unvalidated
    || note.startsWith('model_stream_restart:')), [unvalidated, 'model_stream_restart:etag']);
  assert.ok(finalRange.wasCancelled());
});

test('model download: matching range ETags leave the original prefix unvalidated', async (t) => {
  const firstRange = tail([3], { ETag: '"a"' }, { cut: true });
  const finalRange = tail([4], { ETag: '"a"', 'Content-Range': 'bytes 3-3/4' });
  const { load, loads, requests, notes } = runTransfer(t, [first(), firstRange, finalRange]);
  await load;
  assert.deepEqual(loads, [A]);
  assert.deepEqual(requests, [fullGet, rangeGet, secondRangeGet]);
  assert.deepEqual(notes().filter((note) => note === validated || note === unvalidated),
    [unvalidated, unvalidated]);
});

test('model download: differing range ETags remain incomparable when encoding changes', async (t) => {
  const firstRange = tail([3], { ETag: '"a"' }, { cut: true });
  const finalRange = tail([4], { ETag: '"b"', 'Content-Encoding': 'gzip' });
  const { load, loads, requests } = runTransfer(t, [first(), firstRange, finalRange]);
  await load;
  assert.deepEqual(loads, [A]);
  assert.deepEqual(requests, [fullGet, rangeGet, secondRangeGet]);
});

for (const headers of [{ 'Content-Length': '2' }, { 'Content-Range': 'bytes 2-3/*' }]) {
  test(`model download: retains the expected end from ${Object.keys(headers)[0]} across interruptions`, async (t) => {
    const initial = transfer(A.slice(0, 2), { cut: true });
    const firstRange = transfer([3], { status: 206, headers, cut: true });
    const finalRange = transfer([], { status: 206 });
    const { load, loads, requests } = runTransfer(t, [initial, firstRange, finalRange]);
    await assert.rejects(load, { name: 'ModelTransferLengthError', message: /staged 3 bytes, expected (at least )?4/ });
    assert.deepEqual(loads, []);
    assert.deepEqual(requests, [fullGet, rangeGet, secondRangeGet]);
  });
}

test('model download: a later explicit total replaces an earlier inferred end', async (t) => {
  const initial = transfer(A.slice(0, 2), { cut: true });
  const firstRange = tail([3], { 'Content-Range': 'bytes 2-3/*' }, { cut: true });
  const finalRange = tail([4, 5], { 'Content-Range': 'bytes 3-4/5' });
  const { load, loads, requests } = runTransfer(t, [initial, firstRange, finalRange]);
  await load;
  assert.deepEqual(loads, [[...A, 5]]);
  assert.deepEqual(requests, [fullGet, rangeGet, secondRangeGet]);
});

test('model download: a shorter wildcard range cannot erase an earlier declared end', async (t) => {
  const initial = transfer([1], { cut: true });
  const firstRange = tail([2], { 'Content-Range': 'bytes 1-3/*' }, { cut: true });
  const finalRange = tail([3], { 'Content-Range': 'bytes 2-2/*' });
  const { load, loads, requests } = runTransfer(t, [initial, firstRange, finalRange]);
  await assert.rejects(load, { name: 'ModelTransferLengthError', message: /staged 3 bytes, expected at least 4/ });
  assert.deepEqual(loads, []);
  assert.deepEqual(requests, [fullGet, ['GET', MODEL, [['range', 'bytes=1-']]], rangeGet]);
});

test('model download: restarts when a later explicit total contradicts an earlier declared end', async (t) => {
  const initial = transfer(A.slice(0, 2), { cut: true });
  const firstRange = tail([3], { 'Content-Range': 'bytes 2-4/*' }, { cut: true });
  const finalRange = tail([4], { 'Content-Range': 'bytes 3-3/4' });
  const { load, loads, requests, notes } = runTransfer(t, [initial, firstRange, finalRange, fullB()]);
  await load;
  assert.deepEqual(loads, [B]);
  assert.deepEqual(requests, [fullGet, rangeGet, secondRangeGet, fullGet]);
  assert.ok(notes().includes('model_stream_restart:range'));
});

test('model download: an interrupted full GET 206 retains its declared end', async (t) => {
  const initial = tail(A.slice(0, 2), { 'Content-Range': 'bytes 0-3/*' }, { cut: true });
  const finalRange = transfer([3], { status: 206 });
  const { load, loads, requests } = runTransfer(t, [initial, finalRange]);
  await assert.rejects(load, { name: 'ModelTransferLengthError', message: /staged 3 bytes, expected (at least )?4/ });
  assert.deepEqual(loads, []);
  assert.deepEqual(requests, [fullGet, rangeGet]);
});

test('model download: a later range can extend beyond the span of an interrupted full GET 206', async (t) => {
  const initial = tail(A.slice(0, 2), { 'Content-Range': 'bytes 0-2/*' }, { cut: true });
  const finalRange = transfer(A.slice(2), { status: 206 });
  const { load, loads, requests } = runTransfer(t, [initial, finalRange]);
  await load;
  assert.deepEqual(loads, [A]);
  assert.deepEqual(requests, [fullGet, rangeGet]);
});

test('model download: rejects a short wildcard 206 answering a full GET', async (t) => {
  const initial = transfer(A.slice(0, 3), { status: 206, headers: { 'Content-Range': 'bytes 0-3/*' } });
  const { load, loads, requests } = runTransfer(t, [initial]);
  await assert.rejects(load, { name: 'ModelTransferLengthError', message: /staged 3 bytes, expected 4/ });
  assert.deepEqual(loads, []);
  assert.deepEqual(requests, [fullGet]);
});

test('model download: enforces the span of a full GET 206 even if the body matches the numeric total', async (t) => {
  const initial = tail(A, { 'Content-Range': 'bytes 0-2/4' });
  const { load, loads, requests } = runTransfer(t, [initial]);
  await assert.rejects(load, { name: 'ModelTransferLengthError', message: /staged 4 bytes, expected 3/ });
  assert.deepEqual(loads, []);
  assert.deepEqual(requests, [fullGet]);
});

test('model download: accepts a complete wildcard 206 answering a full GET', async (t) => {
  const initial = tail(A, { 'Content-Range': 'bytes 0-3/*' });
  const { load, loads, requests } = runTransfer(t, [initial]);
  await load;
  assert.deepEqual(loads, [A]);
  assert.deepEqual(requests, [fullGet]);
});

for (const exposure of ['*', 'ETag, Content-Range, Content-Encoding, Access-Control-Expose-Headers']) {
  test(`model download: restarts on different strong ETags with visible CORS encoding metadata (${exposure})`, async (t) => {
    const exposed = { 'Access-Control-Expose-Headers': exposure };
    const initial = transfer(A.slice(0, 2), {
      headers: { 'Content-Length': '4', ETag: '"a"', ...exposed }, type: 'cors', cut: true,
    });
    const resumed = tail(B.slice(2), { ETag: '"b"', ...exposed }, { type: 'cors' });
    const { load, loads, requests, notes } = runTransfer(t, [initial, resumed, fullB()]);
    await load;
    assert.deepEqual(loads, [B]);
    assert.deepEqual(requests, [fullGet, rangeGet, fullGet]);
    assert.ok(notes().includes('model_stream_restart:etag'));
  });
}

test('model download: accepts different strong ETags when CORS hides the range encoding', async (t) => {
  const initial = transfer(A.slice(0, 2), {
    headers: { 'Content-Length': '3', 'Content-Encoding': 'gzip', ETag: '"gzip"' },
    type: 'cors',
    cut: true,
  });
  const resumed = tail(A.slice(2), { ETag: '"identity"' }, { type: 'cors' });
  const { load, loads, requests, notes } = runTransfer(t, [initial, resumed]);
  await load;
  assert.deepEqual(loads, [A]);
  assert.deepEqual(requests, [fullGet, rangeGet]);
  assert.ok(notes().includes(unvalidated));
});

for (const {
  name, initial, resumed, restart, expected, note, error, retryLimit, restartReason,
} of [
  { name: 'strong ETag changes', initial: { ETag: '"a"' },
    resumed: () => tail(B.slice(2), { ETag: '"b"' }), restart: true, expected: B, restartReason: 'etag' },
  { name: 'weak ETag changes', initial: { ETag: 'W/"a"' },
    resumed: () => tail(B.slice(2), { ETag: 'W/"b"' }), expected: HYBRID, note: unvalidated },
  { name: 'weak ETag unchanged', initial: { ETag: 'W/"same"' },
    resumed: () => tail(B.slice(2), { ETag: 'W/"same"' }), expected: HYBRID, note: unvalidated },
  { name: 'Last-Modified unchanged', initial: { 'Last-Modified': LAST_MODIFIED },
    resumed: () => tail(B.slice(2), { 'Last-Modified': LAST_MODIFIED }), expected: HYBRID, note: unvalidated },
  { name: 'unquoted ETag unchanged', initial: { ETag: 'same' },
    resumed: () => tail(B.slice(2), { ETag: 'same' }), expected: HYBRID, note: unvalidated },
  { name: 'resume redirects with the same ETag', initial: { ETag: '"same"' },
    resumed: () => tail(B.slice(2), { ETag: '"same"' }, { url: OTHER }),
    restart: true, expected: B, restartReason: 'url' },
  { name: 'Content-Range wrong start', initial: { ETag: '"a"' },
    resumed: () => tail(A, { ETag: '"a"', 'Content-Range': 'bytes 0-3/4' }),
    restart: true, expected: B, restartReason: 'range' },
  { name: 'Content-Range wrong total', initial: { ETag: '"a"' },
    resumed: () => tail([3, 4, 5], { ETag: '"a"', 'Content-Range': 'bytes 2-4/5' }),
    restart: true, expected: B, restartReason: 'range' },
  { name: 'Content-Range and Content-Length disagree', initial: { ETag: '"a"' },
    resumed: () => tail([3], { ETag: '"a"', 'Content-Length': '1' }),
    restart: true, expected: B, restartReason: 'range' },
  { name: 'first ETag becomes visible on the resume',
    resumed: () => tail(A.slice(2), { ETag: '"a"' }), expected: A, note: unvalidated },
  { name: 'clean short tail', initial: { ETag: '"a"' },
    resumed: () => tail([3], { ETag: '"a"' }), note: validated,
    error: { name: 'ModelTransferLengthError' } },
  { name: 'clean oversize tail', initial: { ETag: '"a"' },
    resumed: () => tail([3, 4, 5], { ETag: '"a"' }), note: validated,
    error: { name: 'ModelTransferLengthError' } },
  { name: 'no validator and same file',
    resumed: () => tail(A.slice(2)), expected: A, note: unvalidated },
  { name: 'no validator and every full GET cut',
    resumed: () => tail(A.slice(2)), expected: A, note: unvalidated },
  { name: 'strong ETag unchanged', initial: { ETag: '"a"' },
    resumed: () => tail(A.slice(2), { ETag: '"a"' }), expected: A, note: validated },
  { name: 'cross-origin hidden range with a known full size',
    resumed: () => transfer(A.slice(2), { status: 206, headers: { 'Content-Length': '2' } }),
    expected: A, note: unvalidated },
  { name: '200 response to a range request',
    resumed: fullB, error: /Range resume not honored/, note: 'model_stream_resume_status:200' },
  { name: 'restart budget exhausted', retryLimit: 1, initial: { ETag: '"a"' },
    resumed: () => tail(B.slice(2), { ETag: '"b"' }),
    error: /changed during download/i, restartReason: 'etag' },
]) {
  test(`model download: ${name}`, async (t) => {
    const resumedTransfer = resumed();
    const replies = [first(initial), resumedTransfer, restart ? fullB() : first(initial)];
    const { load, loads, requests, progress, notes } = runTransfer(t, replies, {
      streamResumeRetries: retryLimit ?? 2,
    });
    if (error) {
      await assert.rejects(load, error);
      assert.deepEqual(loads, []);
    } else {
      await load;
      assert.deepEqual(loads, [expected]);
    }
    assert.deepEqual(requests, restart ? [fullGet, rangeGet, fullGet] : [fullGet, rangeGet]);
    assert.ok(notes().includes('model_stream_resume_retry:1'));
    const verdictNotes = notes().filter((entry) => entry === validated || entry === unvalidated
      || entry.startsWith('model_stream_restart:') || entry.startsWith('model_stream_resume_status:'));
    const verdictNote = restartReason ? `model_stream_restart:${restartReason}` : note;
    assert.deepEqual(verdictNotes, [verdictNote]);
    assert.equal(resumedTransfer.wasCancelled(), Boolean(restartReason));
    const rewinds = progress.filter(([loaded], i) => i > 0 && loaded < progress[i - 1][0]);
    assert.deepEqual(rewinds.map(([loaded]) => loaded), restart ? [0] : []);
  });
}
