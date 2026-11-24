import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { applySeries, loadSeries, verifyMarkers, verifyReleaseLinks } from '../../scripts/build/llama_cpp_patches.mjs';

const root = fileURLToPath(new URL('../..', import.meta.url));
const patchFile = '0001-edit-upload.patch';
const sourceFile = 'src/upload.cpp';
const marker = 'flushing upload batch';
const original = 'int head;\nint old_a;\nint separator;\nint old_b;\nint tail;\n';
const patch = `--- a/${sourceFile}
+++ b/${sourceFile}
@@ -1,2 +1,2 @@
 int head;
-int old_a;
+int new_a;
@@ -4,2 +4,2 @@
-int old_b;
+const char * name = "${marker}";
 int tail;
`;

function fixture(t, { source = original, diff = patch, entry = {} } = {}) {
  const dir = mkdtempSync(path.join(tmpdir(), 'llama-cpp-patches-test-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const upstream = path.join(dir, 'upstream');
  const patches = path.join(dir, 'patches');
  const target = path.join(upstream, sourceFile);
  mkdirSync(path.dirname(target), { recursive: true });
  mkdirSync(patches);
  writeFileSync(target, source);
  writeFileSync(path.join(patches, patchFile), diff);
  const series = [{
    file: patchFile,
    target: 'ggml-webgpu',
    tracking: 'https://github.com/leehack/llama-web-bridge/issues/TRACKING-ISSUE-PENDING',
    upstream: 'https://github.com/ggml-org/llama.cpp/issues/UPSTREAM-ISSUE-PENDING',
    wasm_marker: marker,
    ...entry,
  }];
  writeFileSync(path.join(patches, 'series.json'), JSON.stringify({ patches: series }));
  return { dir, upstream, target, patches };
}

test('unique whole-line hunks apply at shifted offsets without editing upstream', (t) => {
  const source = 'int extra;\n' + original;
  const f = fixture(t, { source });
  const result = applySeries(f.upstream, f.patches);
  assert.equal(result.applied[0].text, source.replace('old_a', 'new_a')
    .replace('int old_b;', `const char * name = "${marker}";`));
  assert.equal(readFileSync(f.target, 'utf8'), source);
});

for (const [name, source, count] of [
  ['missing context', original.replace('int old_b;', 'int foreign;'), 0],
  ['duplicate context', original + 'int old_b;\nint tail;\n', 2],
  ['context inside a line', original.replace('int old_b;', '// int old_b;'), 0],
]) {
  test(`${name} fails strict and skips the whole patch for qualification`, (t) => {
    const f = fixture(t, { source });
    assert.throws(() => applySeries(f.upstream, f.patches), new RegExp(`matches ${count} times`));
    const result = applySeries(f.upstream, f.patches, { qualification: true });
    assert.deepEqual(result.applied, []);
    assert.match(result.skipped[0], new RegExp(`matches ${count} times`));
    assert.equal(readFileSync(f.target, 'utf8'), source);
  });
}

test('CMake uses the patched copy and warns only during qualification on a source mismatch', (t) => {
  const f = fixture(t);
  const build = path.join(f.dir, 'build');
  const cmakeSource = path.join(f.dir, 'project');
  mkdirSync(cmakeSource);
  writeFileSync(path.join(f.upstream, 'CMakeLists.txt'), 'add_library(ggml-webgpu STATIC src/upload.cpp)\n');
  writeFileSync(path.join(cmakeSource, 'CMakeLists.txt'), `cmake_minimum_required(VERSION 3.20)
project(patch_fixture CXX)
set(LLAMA_CPP_DIR "${f.upstream}")
add_subdirectory("${f.upstream}" upstream)
include("${root}/cmake/llama_cpp_patches.cmake")
llamadart_apply_llama_cpp_patches("${root}" "${f.patches}")
`);
  const config = (qualification) => spawnSync('cmake', ['-S', cmakeSource, '-B', build,
    `-DLLAMADART_LLAMA_CPP_PATCH_QUALIFICATION=${qualification}`, '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'], { encoding: 'utf8' });
  const applied = config('OFF');
  if (applied.error?.code === 'ENOENT') {
    t.skip('cmake is not available on PATH');
    return;
  }
  assert.ifError(applied.error);
  assert.equal(applied.status, 0, applied.stderr);
  assert.match(applied.stdout, /ggml-webgpu compiles patched src\/upload\.cpp/);
  assert.equal(readFileSync(f.target, 'utf8'), original);
  assert.match(readFileSync(path.join(build, 'llama.cpp-patched', sourceFile), 'utf8'), /int new_a;/);
  const sources = JSON.parse(readFileSync(path.join(build, 'compile_commands.json'), 'utf8'));
  assert.deepEqual(sources.map(({ file }) => file), [path.join(build, 'llama.cpp-patched', sourceFile)]);
  writeFileSync(f.target, original.replace('int old_b;', 'int foreign;'));
  const strict = config('OFF');
  assert.equal(strict.status, 1, strict.stderr);
  assert.match(strict.stderr, /CMake Error[\s\S]*matches 0\s+times/);
  const permissive = config('ON');
  assert.equal(permissive.status, 0, permissive.stderr);
  assert.match(permissive.stderr, /CMake Warning[\s\S]*matches 0\s+times/);
  assert.doesNotMatch(permissive.stdout, /compiles patched/);
});

test('series schema requires owner/upstream links and a marker', (t) => {
  for (const entry of [
    { tracking: 'https://github.com/another/repo/issues/1' },
    { tracking: undefined }, { upstream: undefined },
    { wasm_marker: '' }, { target: 'one;other' }, { extra: 'typo' },
  ]) {
    const f = fixture(t, { entry });
    assert.throws(() => loadSeries(f.patches), /series\.json/);
  }
  assert.ok(loadSeries(path.join(root, 'patches/llama.cpp')).length > 0);
});

test('refuses malformed hunks and edits outside existing C/C++ sources even for qualification', (t) => {
  for (const diff of [
    patch.replace('@@ -1,2 +1,2 @@', '@@ -1,3 +1,2 @@'),
    patch.replaceAll(sourceFile, 'src/upload.h'),
    patch.replaceAll(sourceFile, '../upload.cpp'),
    patch + patch,
  ]) {
    const f = fixture(t, { diff });
    assert.throws(() => applySeries(f.upstream, f.patches, { qualification: true }), /0001-edit-upload/);
  }
});

test('every requested Wasm must carry each active patch marker', (t) => {
  const f = fixture(t);
  const { applied } = applySeries(f.upstream, f.patches);
  const withMarker = path.join(f.dir, 'with.wasm');
  const withoutMarker = path.join(f.dir, 'without.wasm');
  writeFileSync(withMarker, '\0asm\x01\0\0\0' + marker);
  writeFileSync(withoutMarker, '\0asm\x01\0\0\0');
  verifyMarkers(applied, [withMarker]);
  assert.throws(() => verifyMarkers(applied, [withMarker, withoutMarker]), /without\.wasm.*0001-edit-upload/);
});

test('a marker already present upstream cannot certify an applied patch', (t) => {
  const f = fixture(t, { source: original + `// ${marker}\n` });
  assert.throws(() => applySeries(f.upstream, f.patches, { qualification: true }), /marker already present/);
});

test('retirement applies only when the untouched source contains the upstream fix', (t) => {
  const upstreamFix = 'int upstream_batches_uploads;';
  const f = fixture(t, { entry: { upstream_fix: upstreamFix } });
  assert.equal(applySeries(f.upstream, f.patches).applied.length, 1);
  writeFileSync(f.target, upstreamFix + '\n');
  const { applied, skipped } = applySeries(f.upstream, f.patches);
  assert.deepEqual(applied, []);
  assert.deepEqual(skipped, []);
  const wasm = path.join(f.dir, 'unpatched.wasm');
  writeFileSync(wasm, '\0asm\x01\0\0\0');
  verifyMarkers(applied, [wasm]);
  rmSync(f.target);
  assert.throws(() => applySeries(f.upstream, f.patches), /does not exist/);
});

test('release candidates require real owner and upstream issue links', (t) => {
  const links = {
    tracking: 'https://github.com/leehack/llama-web-bridge/issues/1',
    upstream: 'https://github.com/ggml-org/llama.cpp/pull/2',
  };
  assert.throws(() => verifyReleaseLinks(fixture(t).patches), /pending tracking issue URL/);
  assert.throws(() => verifyReleaseLinks(fixture(t, { entry: { tracking: links.tracking } }).patches),
    /pending upstream issue URL/);
  verifyReleaseLinks(fixture(t, { entry: links }).patches);
});

test('series lists all patch files exactly once in order', (t) => {
  const { patches } = fixture(t);
  const second = '0002-edit-upload.patch';
  writeFileSync(path.join(patches, second), patch);
  const seriesPath = path.join(patches, 'series.json');
  const list = JSON.parse(readFileSync(seriesPath, 'utf8'));
  assert.throws(() => loadSeries(patches), /exactly once in filename order/);
  const [first] = list.patches;
  const next = { ...first, file: second, wasm_marker: 'second marker' };
  writeFileSync(seriesPath, JSON.stringify({ patches: [next, first] }));
  assert.throws(() => loadSeries(patches), /exactly once in filename order/);
  writeFileSync(seriesPath, JSON.stringify({ patches: [first, next] }));
  assert.equal(loadSeries(patches).length, 2);
});

test('overlapping hunks cannot apply even when both contexts occur once', (t) => {
  const diff = patch.replace('@@ -4,2 +4,2 @@\n-int old_b;\n', '@@ -2,2 +2,2 @@\n-int old_a;\n')
    .replace(' int tail;\n', ' int separator;\n');
  const { upstream, patches } = fixture(t, { diff });
  assert.throws(() => applySeries(upstream, patches), /overlaps or precedes another hunk/);
});

test('patch must introduce its Wasm marker into the source', (t) => {
  const { upstream, patches } = fixture(t, { diff: patch.replace(marker, 'different text') });
  assert.throws(() => applySeries(upstream, patches), /patched source lacks its Wasm marker/);
});

test('marker certificate needs a nonempty list of Wasm binaries', (t) => {
  const { dir, upstream, patches } = fixture(t);
  const { applied } = applySeries(upstream, patches);
  const artifact = path.join(dir, 'not-wasm.wasm');
  writeFileSync(artifact, marker);
  assert.throws(() => verifyMarkers(applied, [artifact]), /expected a Wasm binary/);
  assert.throws(() => verifyMarkers(applied, []), /expected at least one Wasm file/);
});

test('build input cannot escape the upstream checkout through a symlink', (t) => {
  const { dir, upstream, target, patches } = fixture(t);
  const outside = path.join(dir, 'outside.cpp');
  renameSync(target, outside);
  try {
    symlinkSync(outside, target, 'file');
  } catch (error) {
    if (process.platform !== 'win32' || error.code !== 'EPERM') throw error;
    t.skip('file symlinks require permission on this Windows host');
    return;
  }
  assert.throws(() => applySeries(upstream, patches), /must not resolve through a symlink/);
});

test('the carried WebGPU patch drains the queue before resetting its weight byte budget', () => {
  const file = path.join(root, 'patches/llama.cpp/0001-webgpu-weight-upload-batching.patch');
  const added = readFileSync(file, 'utf8').split('\n').filter((line) => line.startsWith('+'))
    .map((line) => line.slice(1)).join('\n');
  assert.match(added, /ctx->weight_upload_bytes == batch_bytes\) \{[^}]*ggml_backend_webgpu_wait_queue\(ctx\);\s*ctx->weight_upload_bytes = 0;/);
});
