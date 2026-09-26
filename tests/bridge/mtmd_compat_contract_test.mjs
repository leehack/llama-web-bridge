// Compile src/llama_webgpu_mtmd_compat.h against both upstream media-helper
// API shapes (with and without mtmd_helper_init_opt) and guard that every
// production call site routes through its helpers. Uses $CXX (default c++).
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { readNativeCoreSource } from './native_core_source.mjs';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

// POSIX shell word splitting of $CXX without expansion, as Python's
// shlex.split does, so `CXX="ccache c++"` or a quoted path works.
function splitCommand(command) {
  const words = [];
  let word = null;
  for (let i = 0; i < command.length; i += 1) {
    const char = command[i];
    if (char === "'") {
      const end = command.indexOf("'", i + 1);
      if (end < 0) throw new Error(`CXX has an unterminated quote: ${command}`);
      word = (word ?? '') + command.slice(i + 1, end);
      i = end;
    } else if (char === '"') {
      word ??= '';
      for (i += 1; command[i] !== '"'; i += 1) {
        if (i >= command.length) throw new Error(`CXX has an unterminated quote: ${command}`);
        if (command[i] === '\\' && '\\"'.includes(command[i + 1] ?? '')) i += 1;
        word += command[i];
      }
    } else if (char === '\\') {
      if (i + 1 >= command.length) throw new Error(`CXX ends with an escape: ${command}`);
      i += 1;
      word = (word ?? '') + command[i];
    } else if (' \t\r\n'.includes(char)) {
      if (word !== null) words.push(word);
      word = null;
    } else {
      word = (word ?? '') + char;
    }
  }
  if (word !== null) words.push(word);
  return words;
}

function run(command, args) {
  const result = spawnSync(command, args, { encoding: 'utf8' });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${[command, ...args].join(' ')} exited with ${result.status ?? result.signal}\n${result.stdout}${result.stderr}`);
  }
}

const mediaHelperHeader = (options) => {
  const extra = options ? ', mtmd_helper_init_opt options' : '';
  const check = options ? 'assert(options.sentinel == 73);' : '';
  const declarations = options ? `
struct mtmd_helper_init_opt { int sentinel; };
inline mtmd_helper_init_opt mtmd_helper_init_opt_default() { return {73}; }
` : '';
  return `#pragma once
#include <cassert>
#include <cstddef>
struct mtmd_context {};
struct mtmd_helper_bitmap_wrapper { int result; };
inline mtmd_context expected_context;
inline const unsigned char expected_bytes[] = {1, 2, 3};
inline const char expected_path[] = "reference.wav";
${declarations}
inline mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(
    mtmd_context *ctx, const unsigned char *bytes, size_t size, bool placeholder${extra}) {
  assert(ctx == &expected_context && bytes == expected_bytes && size == 3);
  assert(!placeholder); ${check}
  return {17};
}
inline mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_file(
    mtmd_context *ctx, const char *path, bool placeholder${extra}) {
  assert(ctx == &expected_context && path == expected_path);
  assert(!placeholder); ${check}
  return {29};
}
`;
};

const compatTest = `
#include "llama_webgpu_mtmd_compat.h"
int main() {
  assert(llama_webgpu_bitmap_from_buffer(&expected_context, expected_bytes, 3).result == 17);
  assert(llama_webgpu_bitmap_from_file(&expected_context, expected_path).result == 29);
}
`;

// Both upstream API shapes compile warning-free and pass the expected
// arguments (and the upstream default options) through.
for (const options of [false, true]) {
  const directory = mkdtempSync(path.join(tmpdir(), 'mtmd-compat-'));
  try {
    writeFileSync(path.join(directory, 'mtmd-helper.h'), mediaHelperHeader(options), 'utf8');
    const source = path.join(directory, 'test.cpp');
    writeFileSync(source, compatTest, 'utf8');
    const executable = path.join(directory, 'test');
    const [compiler, ...compilerArgs] = splitCommand(process.env.CXX ?? 'c++');
    run(compiler, [
      ...compilerArgs, '-std=c++17', '-Wall', '-Wextra', '-Werror',
      `-DLLAMADART_MTMD_HELPER_HAS_OPTIONS=${options ? 1 : 0}`,
      `-I${directory}`, `-I${path.join(rootDir, 'src')}`, source,
      '-o', executable,
    ]);
    run(executable, []);
  } catch (error) {
    error.message = `media helper API shape with options=${options}: ${error.message}`;
    throw error;
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

// Production routes every media-helper call through the compatibility helpers.
{
  const count = (text, needle) => text.split(needle).length - 1;
  const core = readNativeCoreSource();
  const tts = readFileSync(path.join(rootDir, 'src/llama_webgpu_tts.cpp'), 'utf8');
  assert.equal(count(core, 'llama_webgpu_bitmap_from_file('), 1, 'core must call llama_webgpu_bitmap_from_file exactly once');
  assert.equal(count(core, 'llama_webgpu_bitmap_from_buffer('), 1, 'core must call llama_webgpu_bitmap_from_buffer exactly once');
  assert.equal(count(tts, 'llama_webgpu_bitmap_from_buffer('), 1, 'TTS must call llama_webgpu_bitmap_from_buffer exactly once');
  for (const [name, text] of [['core', core], ['TTS', tts]]) {
    assert.ok(!text.includes('mtmd_helper_bitmap_init_from_'), `${name} must not call mtmd_helper_bitmap_init_from_* directly`);
  }
}

console.log('mtmd compatibility contract passed');
