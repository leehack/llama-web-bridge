import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';

import { readRegistry } from '../../scripts/release/npm_registry_readback.mjs';

const version = '0.1.91';
const registry = 'https://registry.npmjs.org';
const versionUrl = `${registry}/llama-web-bridge/${version}`;
const tagsUrl = `${registry}/-/package/llama-web-bridge/dist-tags`;
const attestationsUrl = `${registry}/-/npm/v1/attestations/llama-web-bridge@${version}`;
const tarballUrl = `${registry}/llama-web-bridge/-/llama-web-bridge-${version}.tgz`;
const predicateType = 'https://slsa.dev/provenance/v1';
const tarball = Buffer.from('the verified npm tarball');
const sha512 = createHash('sha512').update(tarball).digest();
const expectedTarballSha256 = createHash('sha256').update(tarball).digest('hex');
const expectedIntegrity = `sha512-${sha512.toString('base64')}`;
const bundle = { mediaType: 'application/vnd.dev.sigstore.bundle+json;version=0.3' };

const metadata = (integrity = expectedIntegrity, advertisedPredicate = predicateType) => Response.json({
  name: 'llama-web-bridge',
  version,
  dist: {
    integrity,
    tarball: tarballUrl,
    attestations: { url: attestationsUrl, provenance: { predicateType: advertisedPredicate } },
  },
});

function fixture(overrides = {}) {
  const calls = [];
  const waits = [];
  let elapsed = 0;
  const endpoints = {
    [versionUrl]: () => metadata(),
    [tagsUrl]: () => Response.json({ latest: version }),
    [attestationsUrl]: () => Response.json({ attestations: [{ predicateType, bundle }] }),
    [tarballUrl]: () => new Response(tarball),
    ...overrides,
  };
  return {
    calls, waits, elapsed: () => elapsed,
    run: (integrity = expectedIntegrity) => readRegistry({ version, expectedTarballSha256, expectedIntegrity: integrity }, {
      fetchImpl: async (url) => {
        calls.push(url);
        assert.ok(Object.hasOwn(endpoints, url), `unexpected registry request ${url}`);
        return endpoints[url](elapsed);
      },
      now: () => elapsed,
      sleep: async (ms) => {
        waits.push(ms);
        elapsed += ms;
      },
    }),
  };
}

test('registry readback retries delayed metadata, attestations and tarball separately', async () => {
  let tarballAttempts = 0;
  const { run, calls, waits } = fixture({
    [versionUrl]: (elapsed) => elapsed === 0 ? new Response(null, { status: 404 }) : metadata(),
    [attestationsUrl]: (elapsed) => elapsed === 15_000
      ? new Response(null, { status: 404 })
      : Response.json({ attestations: [{ predicateType, bundle }] }),
    [tarballUrl]: () => {
      tarballAttempts++;
      if (tarballAttempts === 1) return new Response(null, { status: 404 });
      if (tarballAttempts === 2) return new Response(null, { status: 503 });
      if (tarballAttempts === 3) throw new TypeError('fetch failed', { cause: { code: 'ECONNRESET' } });
      return new Response(tarball);
    },
  });

  assert.deepEqual(await run(), { tarball, sha512: sha512.toString('hex'), bundle });
  assert.deepEqual(calls, [
    versionUrl, versionUrl, tagsUrl, attestationsUrl, attestationsUrl,
    tarballUrl, tarballUrl, tarballUrl, tarballUrl,
  ]);
  assert.deepEqual(waits, [15_000, 15_000, 15_000, 30_000, 60_000]);
});

test('registry readback fails on the first tarball digest mismatch', async () => {
  let attempts = 0;
  const { run, calls, waits } = fixture({
    [tarballUrl]: () => new Response(attempts++ === 0 ? 'different bytes' : tarball),
  });

  await assert.rejects(run(), /tarball SHA-256 mismatch/);
  assert.deepEqual(calls, [versionUrl, tagsUrl, attestationsUrl, tarballUrl]);
  assert.deepEqual(waits, []);
});

test('registry readback retries an interrupted tarball response body', async () => {
  const interrupted = new Response(new ReadableStream({
    start(controller) {
      controller.error(new TypeError('terminated', { cause: { code: 'UND_ERR_RES_CONTENT_LENGTH_MISMATCH' } }));
    },
  }));
  const { run, calls, waits } = fixture({
    [tarballUrl]: (elapsed) => elapsed === 0 ? interrupted : new Response(tarball),
  });

  assert.deepEqual((await run()).tarball, tarball);
  assert.deepEqual(calls, [versionUrl, tagsUrl, attestationsUrl, tarballUrl, tarballUrl]);
  assert.deepEqual(waits, [15_000]);
});

test('registry readback checks the tarball integrity against its bytes', async () => {
  const integrity = `sha512-${Buffer.alloc(64).toString('base64')}`;
  const { run, calls, waits } = fixture({ [versionUrl]: () => metadata(integrity) });

  await assert.rejects(run(integrity), /tarball integrity mismatch/);
  assert.deepEqual(calls, [versionUrl, tagsUrl, attestationsUrl, tarballUrl]);
  assert.deepEqual(waits, []);
});

test('registry readback fails on a metadata integrity mismatch or permanent HTTP status', async () => {
  for (const [response, error] of [
    [() => metadata('sha512-wrong'), /registry metadata integrity mismatch/],
    [() => new Response(null, { status: 403 }), /version metadata.*HTTP 403/],
  ]) {
    const { run, calls, waits } = fixture({ [versionUrl]: response });
    await assert.rejects(run(), error);
    assert.deepEqual(calls, [versionUrl]);
    assert.deepEqual(waits, []);
  }
});

test('registry readback rejects an unexpected advertised provenance predicate immediately', async () => {
  const { run, calls, waits } = fixture({
    [versionUrl]: () => metadata(expectedIntegrity, 'https://slsa.dev/provenance/v0.2'),
  });

  await assert.rejects(run(), /version metadata does not match the expected package or provenance/);
  assert.deepEqual(calls, [versionUrl]);
  assert.deepEqual(waits, []);
});

test('registry readback rejects a different latest dist-tag immediately', async () => {
  const { run, calls, waits } = fixture({
    [tagsUrl]: () => Response.json({ latest: '0.1.90' }),
  });

  await assert.rejects(run(), /latest dist-tag does not match/);
  assert.deepEqual(calls, [versionUrl, tagsUrl]);
  assert.deepEqual(waits, []);
});

test('registry readback rejects a missing provenance bundle immediately', async () => {
  const { run, calls, waits } = fixture({
    [attestationsUrl]: () => Response.json({ attestations: [] }),
  });

  await assert.rejects(run(), /expected exactly one SLSA provenance bundle/);
  assert.deepEqual(calls, [versionUrl, tagsUrl, attestationsUrl]);
  assert.deepEqual(waits, []);
});

test('registry readback shares one deadline across metadata and tarball', async () => {
  const notFound = () => new Response(null, { status: 404 });
  const { run, calls, waits, elapsed } = fixture({
    [versionUrl]: (ms) => ms < 1_700_000 ? notFound() : metadata(),
    [tarballUrl]: notFound,
  });

  await assert.rejects(run(), /deadline exceeded waiting for npm registry tarball.*HTTP 404/);
  assert.deepEqual(calls.slice(-5), [tagsUrl, attestationsUrl, tarballUrl, tarballUrl, tarballUrl]);
  assert.deepEqual(waits.slice(-3), [15_000, 30_000, 30_000]);
  assert.equal(elapsed(), 30 * 60 * 1000);
});
