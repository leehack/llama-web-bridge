import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { ARTIFACTS, renderManifestFiles } from '../../scripts/release/manifest.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const templatePath = path.join(root, 'packaging/npm/package.template.json');
const script = path.join(root, 'scripts/release/npm_package.mjs');
const releaseArgs = {
  releaseTag: 'v0.1.90',
  releaseRebuild: 0,
  assetsRepo: 'leehack/llama-web-bridge-assets',
  bridgeRepo: 'leehack/llama-web-bridge',
  bridgeCommit: 'a'.repeat(40),
  upstreamRepo: 'ggml-org/llama.cpp',
  upstreamTag: 'v1.0.0',
  upstreamCommit: 'b'.repeat(40),
  nativeRepo: 'leehack/llamadart-native',
  nativeReleaseTag: 'v1.0.0',
  nativeManifestSha256: 'c'.repeat(64),
  nativeCommit: 'd'.repeat(40),
  emscriptenVersion: '6.0.8',
  orchestratorCorrelationId: 'npm-package-test',
  githubRunId: '123',
  githubRunUrl: 'https://github.com/leehack/llama-web-bridge/actions/runs/123',
};

function run(command, args, options = {}) {
  return spawnSync(command, args, { encoding: 'utf8', ...options });
}

function succeeded(result) {
  assert.equal(result.status, 0, result.error?.message ?? result.stderr);
  return result.stdout;
}

const cli = (...args) => run(process.execPath, [script, ...args]);

function fixture(t, { marked = false } = {}) {
  const temp = mkdtempSync(path.join(tmpdir(), 'lane-npm2-contract-'));
  t.after(() => rmSync(temp, { recursive: true, force: true }));
  const dist = path.join(temp, 'dist');
  const staged = path.join(temp, 'staged');
  mkdirSync(dist);
  const records = ARTIFACTS.map((name) => {
    const bytes = name.endsWith('.d.ts')
      ? readFileSync(path.join(root, 'js', name))
      : Buffer.from(`fixture ${name}\n`);
    writeFileSync(path.join(dist, name), bytes);
    return { name, sizeBytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
  });
  const rendered = renderManifestFiles(releaseArgs, records);
  const manifest = marked ? {
    test_fixture: true,
    release_tag: 'v0.0.0',
    release_channel: 'stable',
    release_rebuild: 0,
    artifacts: rendered.manifest.artifacts,
  } : rendered.manifest;
  const manifestPath = path.join(dist, 'manifest.json');
  const saveManifest = () => writeFileSync(manifestPath, JSON.stringify(manifest));
  saveManifest();
  writeFileSync(path.join(dist, 'sha256sums.txt'), rendered['sha256sums.txt']);
  const stage = (...flags) => cli('stage', '--dist-dir', dist, '--out-dir', staged,
    '--release-tag', manifest.release_tag, '--release-rebuild', '0',
    '--license', path.join(root, 'LICENSE'), '--readme', path.join(root, 'README.md'), ...flags);
  const pack = () => {
    const [packed] = JSON.parse(succeeded(run('npm', ['pack', '--ignore-scripts', '--json', staged], { cwd: temp })));
    return { ...packed, tarball: path.join(temp, packed.filename) };
  };
  const verify = (tarball) => cli('verify-tarball', '--tarball', tarball, '--dist-dir', dist,
    '--expected-version', manifest.release_tag.slice(1));
  return { temp, dist, staged, manifest, stage, pack, verify, saveManifest };
}

test('files and exports are explicit and stable', () => {
  const template = JSON.parse(readFileSync(templatePath, 'utf8'));
  assert.deepEqual(template.files, [
    'dist/llama_webgpu_bridge.js',
    'dist/llama_webgpu_bridge_worker.js',
    'dist/llama_webgpu_bridge.d.ts',
    'dist/llama_webgpu_core.js',
    'dist/llama_webgpu_core.wasm',
    'dist/llama_webgpu_core_mem64.js',
    'dist/llama_webgpu_core_mem64.wasm',
    'dist/llama_webgpu_core_st.js',
    'dist/llama_webgpu_core_st.wasm',
    'dist/llama_webgpu_core_mem64_st.js',
    'dist/llama_webgpu_core_mem64_st.wasm',
    'dist/manifest.json',
    'dist/sha256sums.txt',
    'types/core.d.ts',
    'types/worker.d.ts',
    'LICENSE',
    'README.md',
  ]);
  const js = (types, runtime) => ({ types, import: runtime, default: runtime });
  assert.deepEqual(template.exports, {
    '.': js('./dist/llama_webgpu_bridge.d.ts', './dist/llama_webgpu_bridge.js'),
    './worker': js('./types/worker.d.ts', './dist/llama_webgpu_bridge_worker.js'),
    './core': js('./types/core.d.ts', './dist/llama_webgpu_core.js'),
    './core.wasm': './dist/llama_webgpu_core.wasm',
    './core-mem64': js('./types/core.d.ts', './dist/llama_webgpu_core_mem64.js'),
    './core-mem64.wasm': './dist/llama_webgpu_core_mem64.wasm',
    './core-st': js('./types/core.d.ts', './dist/llama_webgpu_core_st.js'),
    './core-st.wasm': './dist/llama_webgpu_core_st.wasm',
    './core-mem64-st': js('./types/core.d.ts', './dist/llama_webgpu_core_mem64_st.js'),
    './core-mem64-st.wasm': './dist/llama_webgpu_core_mem64_st.wasm',
    './manifest.json': './dist/manifest.json',
    './sha256sums.txt': './dist/sha256sums.txt',
    './package.json': './package.json',
  });
  for (const field of ['private', 'scripts', 'engines', 'dependencies', 'devDependencies']) {
    assert.ok(!Object.hasOwn(template, field), `published package contains ${field}`);
  }
  assert.equal(JSON.parse(readFileSync(path.join(root, 'package.json'), 'utf8')).private, true);
});

test('npm version mapping allows only unsuffixed stable tags with rebuild zero', () => {
  const resolve = (tag, rebuild) => cli('resolve-version', '--release-tag', tag, '--release-rebuild', rebuild);
  assert.deepEqual(JSON.parse(succeeded(resolve('v0.1.90', '0'))), { version: '0.1.90', distTag: 'latest' });
  for (const [tag, rebuild] of [['v0.1.90-1', '1'], ['v0.1.90', '1'], ['b9165', '0'], ['v0.1.90-rc.1', '0'],
    ['v00.1.90', '0'], ['v0.1.90-0', '0'], ['v0.1.90', '00']]) {
    assert.equal(resolve(tag, rebuild).status, 1, `${tag}, rebuild ${rebuild}`);
  }
});

test('staging requires explicit opt-in for marked smoke metadata', (t) => {
  const f = fixture(t, { marked: true });
  const rejected = f.stage();
  assert.notEqual(rejected.status, 0);
  assert.match(rejected.stderr, /test fixture.*--allow-test-fixture/);
  assert.ok(!existsSync(f.staged));

  const record = f.manifest.artifacts['llama_webgpu_core.wasm'];
  record.github_run_id = '123';
  f.saveManifest();
  assert.match(f.stage('--allow-test-fixture').stderr, /artifact metadata.*llama_webgpu_core\.wasm/);
  delete record.github_run_id;
  f.saveManifest();
  succeeded(f.stage('--allow-test-fixture'));
  const packed = f.pack();
  const proof = JSON.parse(succeeded(f.verify(packed.tarball)));
  assert.equal(proof.file_count, 18);
  assert.equal(proof.packed_bytes, packed.size);
  assert.equal(proof.unpacked_bytes, packed.unpackedSize);
  assert.equal(proof.tarball_integrity, packed.integrity);
});

test('release-shaped manifests need provenance and a stable channel', (t) => {
  const f = fixture(t);
  const { github_run_id: runId } = f.manifest;
  delete f.manifest.github_run_id;
  f.saveManifest();
  assert.match(f.stage('--allow-test-fixture').stderr, /provenance|github_run_id/);
  f.manifest.github_run_id = runId;
  const { emscripten_version: version } = f.manifest;
  f.manifest.emscripten_version = null;
  f.saveManifest();
  assert.match(f.stage().stderr, /emscripten_version/);
  f.manifest.emscripten_version = version;
  f.manifest.release_channel = 'development';
  f.saveManifest();
  assert.match(f.stage().stderr, /release_channel|stable/);
});

test('npm pack includes every exported path and declarations resolve for TypeScript consumers', (t) => {
  const f = fixture(t);
  succeeded(f.stage());
  const packed = f.pack();
  const pkg = JSON.parse(readFileSync(path.join(f.staged, 'package.json'), 'utf8'));
  const actual = packed.files.map(({ path: name }) => name).sort();
  for (const target of Object.values(pkg.exports).flatMap((entry) =>
    typeof entry === 'string' ? [entry] : Object.values(entry))) {
    assert.ok(actual.includes(target.slice(2)), `export missing from tarball: ${target}`);
  }
  assert.deepEqual(actual, [...pkg.files, 'package.json'].sort());
  succeeded(f.verify(packed.tarball));

  const installed = path.join(f.temp, 'node_modules', pkg.name);
  mkdirSync(installed, { recursive: true });
  succeeded(run('tar', ['-xzf', packed.tarball, '--strip-components', '1', '-C', installed]));
  const consumer = path.join(f.temp, 'consumer.mts');
  writeFileSync(consumer, `
import { LlamaWebGpuBridge } from 'llama-web-bridge';
import type { LlamaCoreModuleFactory } from 'llama-web-bridge';
import core from 'llama-web-bridge/core';
import coreMem64 from 'llama-web-bridge/core-mem64';
import coreSt from 'llama-web-bridge/core-st';
import coreMem64St from 'llama-web-bridge/core-mem64-st';
import * as worker from 'llama-web-bridge/worker';
const factories: LlamaCoreModuleFactory[] = [core, coreMem64, coreSt, coreMem64St];
new LlamaWebGpuBridge({ coreModuleFactory: factories[0] });
void worker;
`);
  for (const [resolution, module] of [['bundler', 'esnext'], ['node16', 'node16']]) {
    const result = run(process.execPath, [path.join(root, 'node_modules/typescript/bin/tsc'),
      '--strict', '--noEmit', '--target', 'es2022', '--module', module, '--moduleResolution', resolution, consumer]);
    assert.equal(result.status, 0, result.stdout + result.stderr);
  }
});

test('staging and tarball verification reject changed core bytes', (t) => {
  const f = fixture(t, { marked: true });
  succeeded(f.stage('--allow-test-fixture'));
  const core = 'llama_webgpu_core.wasm';
  writeFileSync(path.join(f.staged, 'dist', core), 'changed');
  assert.match(f.verify(f.pack().tarball).stderr, /checksum mismatch.*llama_webgpu_core\.wasm/);
  rmSync(f.staged, { recursive: true });
  writeFileSync(path.join(f.dist, core), 'changed');
  assert.match(f.stage('--allow-test-fixture').stderr, /checksum mismatch.*llama_webgpu_core\.wasm/);
  assert.ok(!existsSync(f.staged));
});
