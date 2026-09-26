// C++ core source text for the JS contract tests. src/llama_webgpu_core.cpp
// includes its parts from src/core/, so read it with each
// `#include "core/<part>.inc"` line replaced by that part, as the compiler
// sees it. A part that is missing, included in another form, or not included
// exactly once throws instead of silently narrowing a check.
import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const sourceDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../src');

export function readNativeCoreSource() {
  const expanded = [];
  const source = readFileSync(path.join(sourceDir, 'llama_webgpu_core.cpp'), 'utf8')
    .replace(/^#include "(core\/[A-Za-z0-9_]+\.inc)"$/gm, (_, part) => {
      expanded.push(part);
      return readFileSync(path.join(sourceDir, part), 'utf8').replace(/\n+$/, '');
    });
  const parts = readdirSync(path.join(sourceDir, 'core'))
    .filter((name) => name.endsWith('.inc'))
    .map((name) => `core/${name}`)
    .sort();
  if (/#\s*include\s*["<]core\//.test(source) || expanded.sort().join('\n') !== parts.join('\n')) {
    throw new Error(
      'src/llama_webgpu_core.cpp must include every src/core/*.inc part exactly once, '
      + 'each on its own `#include "core/<part>.inc"` line',
    );
  }
  return source;
}
