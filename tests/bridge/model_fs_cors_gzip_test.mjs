import assert from 'node:assert/strict';
import test from 'node:test';
import http from 'node:http';
import { Readable } from 'node:stream';
import { createGzip } from 'node:zlib';
import { build } from 'esbuild';

// This large-body browser regression is opt-in; ordinary CI stays model-free
// and does not transfer a GiB. No model or GPU is needed.
test('CORS-hidden gzip encoding never makes wire length a decoded-body contract', {
  skip: process.env.LLAMADART_BROWSER_FS_CORS_SMOKE !== '1',
  timeout: 180000,
}, async () => {
  const { chromium } = await import('playwright');
  const gib = 1073741824;
  function payload() {
    const chunk = Buffer.alloc(65536);
    return Readable.from((function* () {
      for (let offset = 0; offset < gib; offset += chunk.length) yield chunk;
    })());
  }
  let wireLength = 0;
  for await (const chunk of payload().pipe(createGzip({ level: 0 }))) wireLength += chunk.length;
  assert.ok(wireLength > gib && wireLength < 2147483647);
  const bundle = await build({ entryPoints: ['js/src/internal/download.ts'], bundle: true, format: 'esm', write: false });
  const servers = [];
  let browser;
  try {
    const start = async (handler) => {
      const server = http.createServer(handler); servers.push(server);
      await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
      return `http://127.0.0.1:${server.address().port}`;
    };
    const modelOrigin = await start((request, response) => {
      response.writeHead(200, { 'Content-Type': 'application/octet-stream',
        'Content-Length': wireLength, 'Content-Encoding': 'gzip',
        'Access-Control-Allow-Origin': '*', 'Cross-Origin-Resource-Policy': 'cross-origin' });
      const source = payload(); const gzip = createGzip({ level: 0 });
      response.on('close', () => { source.destroy(); gzip.destroy(); });
      source.pipe(gzip).pipe(response);
    });
    const pageOrigin = await start((request, response) => {
      response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      if (request.url === '/writer.js') {
        response.setHeader('Content-Type', 'text/javascript'); response.end(bundle.outputFiles[0].contents);
      } else response.end('<!doctype html><title>Owned CORS regression</title>');
    });
    browser = await chromium.launch({ headless: true,
      ...(process.env.LLAMADART_BROWSER_EXECUTABLE ? { executablePath: process.env.LLAMADART_BROWSER_EXECUTABLE } : {}) });
    const page = await browser.newPage(); await page.goto(pageOrigin);
    const result = await page.evaluate(async ({ modelOrigin }) => {
      const { writeResponseToFsFileWithProgress: write } = await import('/writer.js');
      const response = await fetch(modelOrigin + '/gzip');
      const result = { type: response.type, encoding: response.headers.get('content-encoding'),
        wireLength: Number(response.headers.get('content-length')) };
      const fs = { length: 0, exists: false, reservations: 0,
        unlink() { this.exists = false; this.length = 0; },
        open(_path, mode) { this.exists = true; if (mode === 'w') this.length = 0; return {}; },
        close() {},
        write(_stream, _data, _offset, length, position) {
          if (length === 1 && position > 1e9) this.reservations++;
          this.length = Math.max(this.length, Number(position) + length); return length;
        } };
      result.bytes = await write(response, fs, '/model');
      result.logicalLength = fs.length; result.reservations = fs.reservations;
      return result;
    }, { modelOrigin });
    assert.equal(result.type, 'cors'); assert.equal(result.encoding, null);
    assert.equal(result.wireLength, wireLength);
    assert.equal(result.reservations, 1);
    assert.equal(result.bytes, gib); assert.equal(result.logicalLength, gib);
    console.log(JSON.stringify(result));
  } finally {
    await browser?.close();
    await Promise.all(servers.map(server => new Promise(resolve => {
      server.closeAllConnections(); server.close(resolve);
    })));
  }
});
