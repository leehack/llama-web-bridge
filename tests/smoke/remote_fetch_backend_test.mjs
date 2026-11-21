import assert from 'node:assert/strict';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { test } from 'node:test';

import {
  assertModelRequests, parseArgs, renderAppWorker, renderHarness, withModelRangeServer,
} from '../../scripts/smoke/remote_fetch_backend.mjs';
import { withTempDir } from '../../scripts/smoke/support.mjs';

test('remote fetch smoke defaults to wasm32 and accepts only supported memory modes', () => {
  assert.equal(parseArgs([]).memoryMode, 'wasm32');
  assert.equal(parseArgs([]).timeoutMs, 90_000);
  assert.equal(parseArgs(['--dist-dir', '/tmp/dist', '--memory-mode', 'wasm64']).distDir, '/tmp/dist');
  assert.equal(parseArgs(['--memory-mode', 'wasm64']).memoryMode, 'wasm64');
  assert.throws(() => parseArgs(['--memory-mode', 'wasm128']), /invalid choice/);
});

test('remote fetch smoke runs distinct app-worker and bridge-worker entry points', () => {
  assert.match(renderAppWorker(), /runBrowserCase\(settings, true\)/);
  assert.match(renderHarness({ scenario: 'app-worker' }), /new Worker\('\/app_worker\.mjs', \{ type: 'module' \}\)/);
  const bridgeHarness = renderHarness({ scenario: 'bridge-worker', memoryMode: 'wasm64' });
  assert.match(bridgeHarness, /"memoryMode":"wasm64"/);
  assert.match(bridgeHarness, /runBrowserCase\(settings, false\)/);
});

test('isolated model server supports exact-path HEAD and byte ranges without a directory fallback', async () => {
  const model = Buffer.from(Array.from({ length: 65541 }, (_, i) => i % 251));
  await withTempDir('remote-fetch-test-', async (webRoot) => {
    await fsp.writeFile(path.join(webRoot, 'index.html'), '<title>fixture</title>');
    await withModelRangeServer(webRoot, model, async (url, requests) => {
      assert.equal(await (await fetch(url)).text(), '<title>fixture</title>');
      const request = (pathname, options) => fetch(new URL(pathname, url), { redirect: 'manual', ...options });
      for (const [method, pathname] of [['HEAD', '/mm'], ['GET', '/mm'], ['GET', '/mm/'], ['GET', '/mm/other.gguf']]) {
        const response = await request(pathname, { method });
        assert.equal(response.status, 404);
        assert.equal(response.headers.get('location'), null);
        assert.equal(response.headers.get('cross-origin-embedder-policy'), 'require-corp');
      }

      const head = await request('/mm/tiny.gguf', { method: 'HEAD', headers: { Range: 'bytes=0-' } });
      assert.equal(head.status, 200);
      assert.equal(head.headers.get('accept-ranges'), 'bytes');
      assert.equal(head.headers.get('content-length'), String(model.length));
      assert.equal(head.headers.get('content-type'), 'application/octet-stream');
      assert.equal((await head.arrayBuffer()).byteLength, 0);

      const signedPath = `/mm/tiny.gguf?sig=a/b&token=${'x'.repeat(280)}`;
      for (const [range, start, end] of [
        ['bytes=0-0', 0, 0],
        ['bytes=0-65535', 0, 65535],
        ['bytes=65536-69999', 65536, model.length - 1],
        ['bytes=65536-', 65536, model.length - 1],
      ]) {
        const reply = await request(signedPath, { headers: { Range: range } });
        assert.equal(reply.status, 206);
        assert.equal(reply.headers.get('content-range'), `bytes ${start}-${end}/${model.length}`);
        assert.equal(reply.headers.get('content-length'), String(end - start + 1));
        assert.deepEqual(Buffer.from(await reply.arrayBuffer()), model.subarray(start, end + 1));
        assert.deepEqual(requests.at(-1), { method: 'GET', url: signedPath, pathname: '/mm/tiny.gguf', range });
      }

      const full = await request('/mm/tiny.gguf');
      assert.equal(full.status, 200);
      assert.deepEqual(Buffer.from(await full.arrayBuffer()), model);
      for (const range of ['bytes=65541-', 'bytes=2-1', 'bytes=bad']) {
        const invalid = await request('/mm/tiny.gguf', { headers: { Range: range } });
        assert.equal(invalid.status, 416);
        assert.equal(invalid.headers.get('content-range'), `bytes */${model.length}`);
      }
    });
  });
});

test('the smoke requires the exact model URL and both native range requests', () => {
  const url = '/mm/tiny.gguf?sig=a/b';
  const expected = [
    { url, method: 'HEAD', range: 'bytes=0-' },
    { url, method: 'GET', range: 'bytes=0-65535' },
  ];
  assert.doesNotThrow(() => assertModelRequests(expected, url));
  for (const unexpected of ['/mm', '/mm/', '/mm/tiny.gguf']) {
    assert.throws(() => assertModelRequests([...expected, { url: unexpected }], url), /unexpected model URL/);
  }
  assert.throws(() => assertModelRequests(expected.slice(1), url), /missing WasmFS HEAD/);
  assert.throws(() => assertModelRequests(expected.slice(0, 1), url), /missing WasmFS chunk GET/);
});
