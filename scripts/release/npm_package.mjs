#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { lstatSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { gunzipSync } from 'node:zlib';

import { ArgparseExit, parseArguments, progName, runCli, SystemExitError } from './cli.mjs';
import { Channel, ContractError, parseReleaseTag, strictJsonLoads } from './contract.mjs';
import { ARTIFACTS } from './manifest.mjs';
import { PUBLICATION_FILES } from './publication_state.mjs';
import { loadCandidate } from './qualification.mjs';

const packaging = new URL('../../packaging/npm/', import.meta.url);
const CHECKSUM_LINE = /^([0-9a-f]{64}) {2}([A-Za-z0-9_.-]+)$/;
export const FIXTURE_RELEASE_TAG = 'v0.0.0';

function requireThat(condition, message) {
  if (!condition) throw new ContractError(message);
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function sameFiles(names, expected) {
  return names.length === expected.length && expected.every((name) => names.includes(name));
}

export function resolveVersion(releaseTag, releaseRebuild) {
  const release = parseReleaseTag(releaseTag);
  requireThat(release.channel === Channel.STABLE && release.tag === release.baseTag
    && release.rebuild === 0 && releaseRebuild === 0
    && release.versionParts.every(Number.isSafeInteger),
  'npm packaging requires an unsuffixed stable release tag and rebuild 0');
  return { version: release.tag.slice(1), distTag: 'latest' };
}

function readDist(distDir, releaseTag, { allowTestFixture = false } = {}) {
  requireThat(lstatSync(distDir).isDirectory(), 'dist must be a real directory');
  requireThat(sameFiles(readdirSync(distDir), PUBLICATION_FILES),
    'dist must contain exactly the release files, including manifest.json and sha256sums.txt');
  const files = new Map(PUBLICATION_FILES.map((name) => {
    const file = path.join(distDir, name);
    requireThat(lstatSync(file).isFile(), `dist entry must be a regular file: ${name}`);
    return [name, readFileSync(file)];
  }));
  const manifest = strictJsonLoads(files.get('manifest.json').toString('utf8'), 'npm package manifest');
  requireThat(manifest && typeof manifest === 'object' && !Array.isArray(manifest),
    'manifest root must be an object');

  const fixture = Object.hasOwn(manifest, 'test_fixture');
  if (fixture) {
    requireThat(allowTestFixture, 'test fixture requires --allow-test-fixture for staging');
    requireThat(manifest.test_fixture === true && releaseTag === FIXTURE_RELEASE_TAG
      && sameFiles(Object.keys(manifest), ['test_fixture', 'release_tag', 'release_rebuild', 'release_channel', 'artifacts']),
    'test fixture must use v0.0.0 and contain only fixture metadata');
  }
  requireThat(manifest.release_tag === releaseTag && manifest.release_rebuild === 0
    && manifest.release_channel === Channel.STABLE, 'manifest release_tag, release_rebuild or release_channel mismatch');

  const sums = new Map();
  for (const line of files.get('sha256sums.txt').toString('utf8').replace(/\n$/, '').split('\n')) {
    const match = CHECKSUM_LINE.exec(line);
    requireThat(match && !sums.has(match[2]), `invalid or duplicate checksum line: ${JSON.stringify(line)}`);
    sums.set(match[2], match[1]);
  }
  requireThat(sameFiles([...sums.keys()], ARTIFACTS), 'sha256sums.txt must contain exactly the release artifacts');
  requireThat(manifest.artifacts && typeof manifest.artifacts === 'object'
    && sameFiles(Object.keys(manifest.artifacts), ARTIFACTS), 'manifest must describe exactly the release artifacts');
  for (const name of ARTIFACTS) {
    const bytes = files.get(name);
    const digest = sha256(bytes);
    const artifact = manifest.artifacts[name];
    requireThat(artifact && typeof artifact === 'object' && sameFiles(Object.keys(artifact), ['sha256', 'size_bytes']),
      `invalid artifact metadata for ${name}`);
    requireThat(sums.get(name) === digest && artifact.sha256 === digest, `artifact checksum mismatch for ${name}`);
    requireThat(artifact.size_bytes === bytes.length, `artifact size mismatch for ${name}`);
  }
  if (!fixture) {
    try {
      const [validatedManifest] = loadCandidate(distDir);
      requireThat(isDeepStrictEqual(manifest, validatedManifest), 'manifest changed during validation');
      requireThat(/^[0-9]+\.[0-9]+\.[0-9]+$/.test(manifest.emscripten_version), 'invalid emscripten_version');
    } catch (error) {
      throw new ContractError(`release manifest provenance validation failed: ${error.message}`, { cause: error });
    }
  }
  return files;
}

function packageJson(version) {
  const template = JSON.parse(readFileSync(new URL('package.template.json', packaging), 'utf8'));
  return { ...template, version };
}

export function stagePackage({ distDir, releaseTag, releaseRebuild, license, readme, outDir, allowTestFixture }) {
  const identity = resolveVersion(releaseTag, releaseRebuild);
  const files = readDist(distDir, releaseTag, { allowTestFixture });
  const pkg = packageJson(identity.version);
  const contents = new Map([...files].map(([name, bytes]) => [`dist/${name}`, bytes]));
  contents.set('types/core.d.ts', readFileSync(new URL('types/core.d.ts', packaging)));
  contents.set('types/worker.d.ts', readFileSync(new URL('types/worker.d.ts', packaging)));
  contents.set('LICENSE', readFileSync(license));
  contents.set('README.md', readFileSync(readme));
  contents.set('package.json', Buffer.from(`${JSON.stringify(pkg, null, 2)}\n`));

  mkdirSync(outDir, { mode: 0o700 });
  try {
    for (const [name, bytes] of contents) {
      const file = path.join(outDir, name);
      mkdirSync(path.dirname(file), { recursive: true });
      writeFileSync(file, bytes, { flag: 'wx' });
    }
  } catch (error) {
    rmSync(outDir, { recursive: true, force: true });
    throw error;
  }
  return identity;
}

const TAR_BLOCK = 512;
const MAX_TARBALL_BYTES = 256 * 1024 * 1024;

function tarNumber(field) {
  const value = field.toString('latin1').replace(/[\0 ]+$/, '');
  requireThat(/^[0-7]+$/.test(value), 'invalid tar header number');
  return Number.parseInt(value, 8);
}

// npm emits short ustar file names. Reading only regular entries avoids
// writing anything from the tarball onto the verifier's filesystem.
function readTar(tar, expected) {
  let offset = 0;
  const entries = new Map();
  while (offset + TAR_BLOCK <= tar.length) {
    const header = tar.subarray(offset, offset + TAR_BLOCK);
    if (header.every((byte) => byte === 0)) break;
    const checksum = header.reduce((sum, byte, index) => sum + (index >= 148 && index < 156 ? 32 : byte), 0);
    requireThat(checksum === tarNumber(header.subarray(148, 156)), 'invalid tar header checksum');
    requireThat(header[156] === 48 && header.subarray(257, 263).equals(Buffer.from('ustar\0'))
      && header.subarray(157, 257).every((byte) => byte === 0)
      && header.subarray(345, 500).every((byte) => byte === 0), 'tarball must contain only regular ustar files');
    const fullName = header.subarray(0, 100).toString('latin1').replace(/\0+$/, '');
    const name = fullName.slice('package/'.length);
    requireThat(fullName.startsWith('package/') && expected.includes(name) && !entries.has(name),
      `unexpected or duplicate tarball member: ${JSON.stringify(fullName)}`);
    const size = tarNumber(header.subarray(124, 136));
    const start = offset + TAR_BLOCK;
    requireThat(start + size <= tar.length, `truncated tarball member: ${name}`);
    entries.set(name, tar.subarray(start, start + size));
    offset = start + Math.ceil(size / TAR_BLOCK) * TAR_BLOCK;
  }
  requireThat(tar.length - offset >= TAR_BLOCK * 2 && tar.length % TAR_BLOCK === 0
    && tar.subarray(offset).every((byte) => byte === 0), 'invalid tarball terminator');
  requireThat(sameFiles([...entries.keys()], expected), 'tarball is missing required package files');
  return entries;
}

export function verifyTarball({ tarball, distDir, expectedVersion }) {
  const releaseTag = `v${expectedVersion}`;
  const { version } = resolveVersion(releaseTag, 0);
  const dist = readDist(distDir, releaseTag, { allowTestFixture: releaseTag === FIXTURE_RELEASE_TAG });
  const expected = packageJson(version);
  const stat = lstatSync(tarball);
  requireThat(stat.isFile() && stat.size <= MAX_TARBALL_BYTES, 'tarball must be a regular file within the verification size limit');
  const archive = readFileSync(tarball);
  const files = readTar(gunzipSync(archive, { maxOutputLength: MAX_TARBALL_BYTES }), [...expected.files, 'package.json']);
  const pkg = strictJsonLoads(files.get('package.json').toString('utf8'), 'tarball package.json');
  requireThat(isDeepStrictEqual(pkg, expected), 'tarball package.json differs from the release template');
  const exported = Object.values(pkg.exports).flatMap((entry) => typeof entry === 'string' ? [entry] : Object.values(entry));
  for (const target of [pkg.main, pkg.types, ...exported]) {
    requireThat(target.startsWith('./') && files.has(target.slice(2)), `export missing from tarball: ${target}`);
  }
  for (const [name, bytes] of dist) {
    requireThat(files.get(`dist/${name}`)?.equals(bytes), `tarball checksum mismatch for ${name}`);
  }
  for (const name of ['types/core.d.ts', 'types/worker.d.ts']) {
    requireThat(files.get(name)?.equals(readFileSync(new URL(name, packaging))), `tarball declaration differs: ${name}`);
  }
  return {
    tarball_sha256: sha256(archive),
    tarball_integrity: `sha512-${createHash('sha512').update(archive).digest('base64')}`,
    unpacked_bytes: [...files.values()].reduce((total, bytes) => total + bytes.length, 0),
    packed_bytes: archive.length,
    file_count: files.size,
  };
}

const required = (flag) => ({ flag, required: true, type: 'str' });
const releaseOptions = [required('--release-tag'), required('--release-rebuild')];
const commands = new Map([
  ['resolve-version', releaseOptions],
  ['stage', [...releaseOptions, required('--dist-dir'), required('--out-dir'),
    required('--license'), required('--readme'), { flag: '--allow-test-fixture', type: 'flag' }]],
  ['verify-tarball', [required('--tarball'), required('--dist-dir'), required('--expected-version')]],
]);

function main(argv, write) {
  const [command, ...flags] = argv;
  const prog = progName(import.meta.url);
  if (!commands.has(command)) {
    const usage = `usage: ${prog} {${[...commands.keys()].join(',')}} [options]\n`;
    if (command === '--help' || command === '-h') throw new ArgparseExit(0, { stdout: usage });
    throw new ArgparseExit(2, { stderr: usage });
  }
  const args = parseArguments(flags, { prog: `${prog} ${command}`, options: commands.get(command) });
  try {
    if (command !== 'verify-tarball') {
      requireThat(args.releaseRebuild === '0', 'npm packaging requires rebuild 0');
      args.releaseRebuild = 0;
    }
    const result = command === 'stage' ? stagePackage(args)
      : command === 'verify-tarball' ? verifyTarball(args) : resolveVersion(args.releaseTag, args.releaseRebuild);
    write(`${JSON.stringify(result)}\n`);
  } catch (error) {
    throw new SystemExitError(`error: ${error.message}`);
  }
  return 0;
}

if (import.meta.main) runCli(main);
