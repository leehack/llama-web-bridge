import { createHash } from 'node:crypto';
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { setTimeout as sleepTimer } from 'node:timers/promises';

const REGISTRY = 'https://registry.npmjs.org';
const PACKAGE = 'llama-web-bridge';
const PREDICATE = 'https://slsa.dev/provenance/v1';
const WAIT_MS = 30 * 60_000;
const RETRYABLE_STATUSES = new Set([404, 500, 502, 503, 504]);
const RETRYABLE_NETWORK_CODES = new Set([
  'EAI_AGAIN', 'ENOTFOUND', 'ECONNRESET', 'ECONNREFUSED', 'EHOSTUNREACH', 'ENETUNREACH', 'EPIPE', 'ETIMEDOUT',
  'UND_ERR_CONNECT_TIMEOUT', 'UND_ERR_HEADERS_TIMEOUT', 'UND_ERR_BODY_TIMEOUT', 'UND_ERR_SOCKET',
  'UND_ERR_RES_CONTENT_LENGTH_MISMATCH',
]);

export async function readRegistry({ version, expectedIntegrity, expectedTarballSha256 }, {
  fetchImpl = fetch,
  now = () => performance.now(),
  sleep = sleepTimer,
  log = () => {},
} = {}) {
  if (!/^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/.test(version)
      || version === '0.0.0' || !/^[a-f0-9]{64}$/.test(expectedTarballSha256)
      || !/^sha512-[A-Za-z0-9+/]{86}==$/.test(expectedIntegrity)) {
    throw new Error('invalid expected npm version or tarball digests');
  }
  const deadline = now() + WAIT_MS;

  async function read(url, label, requestTimeout = 30_000) {
    let delay = 15_000;
    let failure = 'no time remaining';
    while (true) {
      const remaining = Math.ceil(deadline - now());
      if (remaining <= 0) break;
      try {
        const response = await fetchImpl(url, {
          cache: 'no-store',
          redirect: 'manual',
          signal: AbortSignal.timeout(Math.min(requestTimeout, remaining)),
        });
        if (response.status === 200) return Buffer.from(await response.arrayBuffer());
        void response.body?.cancel().catch(() => {});
        failure = `HTTP ${response.status}`;
        if (!RETRYABLE_STATUSES.has(response.status)) throw new Error(`npm registry ${label}: ${failure}`);
      } catch (error) {
        const code = error.cause?.code;
        if (error.name !== 'TimeoutError' && !(error instanceof TypeError && RETRYABLE_NETWORK_CODES.has(code))) {
          throw error;
        }
        failure = code ?? error.name;
      }
      const wait = Math.min(delay, deadline - now());
      if (wait <= 0) break;
      log(`npm registry ${label}: ${failure}; waiting ${Math.ceil(wait / 1000)}s`);
      await sleep(Math.ceil(wait));
      delay = Math.min(2 * delay, 60_000);
    }
    throw new Error(`deadline exceeded waiting for npm registry ${label} (${failure})`);
  }

  const versionUrl = `${REGISTRY}/${PACKAGE}/${version}`;
  const tagsUrl = `${REGISTRY}/-/package/${PACKAGE}/dist-tags`;
  const attestationsUrl = `${REGISTRY}/-/npm/v1/attestations/${PACKAGE}@${version}`;
  const tarballUrl = `${REGISTRY}/${PACKAGE}/-/${PACKAGE}-${version}.tgz`;
  const metadata = JSON.parse(await read(versionUrl, 'version metadata'));
  if (metadata?.dist?.integrity !== expectedIntegrity) {
    throw new Error('npm registry metadata integrity mismatch');
  }
  if (metadata.name !== PACKAGE || metadata.version !== version || metadata.dist.tarball !== tarballUrl
      || metadata.dist.attestations?.url !== attestationsUrl
      || metadata.dist.attestations?.provenance?.predicateType !== PREDICATE) {
    throw new Error('npm registry version metadata does not match the expected package or provenance');
  }
  const tags = JSON.parse(await read(tagsUrl, 'dist-tags'));
  if (tags?.latest !== version) throw new Error(`npm registry latest dist-tag does not match ${version}`);
  const document = JSON.parse(await read(attestationsUrl, 'attestations'));
  const provenance = Array.isArray(document?.attestations)
    ? document.attestations.filter((entry) => entry?.predicateType === PREDICATE)
    : [];
  const bundle = provenance[0]?.bundle;
  if (provenance.length !== 1 || !bundle || typeof bundle !== 'object' || Array.isArray(bundle)) {
    throw new Error('npm registry attestations: expected exactly one SLSA provenance bundle');
  }

  const tarball = await read(tarballUrl, 'tarball', 120_000);
  if (createHash('sha256').update(tarball).digest('hex') !== expectedTarballSha256) {
    throw new Error('npm registry tarball SHA-256 mismatch');
  }
  const sha512 = createHash('sha512').update(tarball).digest();
  if (`sha512-${sha512.toString('base64')}` !== expectedIntegrity) {
    throw new Error('npm registry tarball integrity mismatch');
  }
  return { tarball, sha512: sha512.toString('hex'), bundle };
}

if (import.meta.main) {
  const { VERSION, EXPECTED_INTEGRITY, EXPECTED_TARBALL_SHA256, RUNNER_TEMP } = process.env;
  try {
    const { tarball, sha512, bundle } = await readRegistry({
      version: VERSION, expectedIntegrity: EXPECTED_INTEGRITY, expectedTarballSha256: EXPECTED_TARBALL_SHA256,
    }, { log: console.error });
    writeFileSync(join(RUNNER_TEMP, 'registry.tgz'), tarball);
    writeFileSync(join(RUNNER_TEMP, 'npm-provenance.json'), JSON.stringify(bundle));
    console.log(sha512);
  } catch (error) {
    console.error(`error: ${error.message}`);
    process.exitCode = 1;
  }
}
