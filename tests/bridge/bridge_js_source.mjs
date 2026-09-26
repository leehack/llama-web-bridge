// Bridge JS source text for the static API contract tests. js/src/llama_webgpu_bridge.js is only the public
// entry; the implementation lives in the .js and .ts modules it imports (the
// public .d.ts is excluded). The tests read every module, joined in the order
// the former single-file source declared them (helpers, worker host and proxy,
// direct runtime, facade), so multi-token patterns keep their scope. A missing
// listed module throws instead of silently narrowing a check.
import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

const ORDERED_MODULES = [
  'worker_protocol.ts',
  'worker_host.ts',
  'worker_proxy.ts',
  'runtime.ts',
  'bridge.ts',
  'llama_webgpu_bridge.js',
];

// Reads a repository file as Python's read_text() does: universal newlines.
export function readRepoText(relativePath) {
  return readFileSync(path.join(repoRoot, relativePath), 'utf8').replace(/\r\n?/g, '\n');
}

function isModule(name) {
  return (name.endsWith('.js') || name.endsWith('.ts')) && !name.endsWith('.d.ts');
}

// Python sorts paths part by part, not as flat strings.
function comparePaths(left, right) {
  const a = left.split('/');
  const b = right.split('/');
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    if (a[index] !== b[index]) {
      return a[index] < b[index] ? -1 : 1;
    }
  }
  return a.length - b.length;
}

export function bridgeJsSource() {
  const sourceDir = 'js/src';
  const internal = readdirSync(path.join(repoRoot, sourceDir, 'internal'), { withFileTypes: true })
    .filter((entry) => entry.isFile() && isModule(entry.name))
    .map((entry) => `${sourceDir}/internal/${entry.name}`)
    .sort(comparePaths);
  const ordered = ORDERED_MODULES.map((name) => `${sourceDir}/${name}`);
  const listed = new Set([...internal, ...ordered]);
  const remaining = readdirSync(path.join(repoRoot, sourceDir), { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile() && isModule(entry.name))
    .map((entry) => path.relative(repoRoot, path.join(entry.parentPath, entry.name)).split(path.sep).join('/'))
    .filter((relativePath) => !listed.has(relativePath))
    .sort(comparePaths);
  return [...internal, ...ordered, ...remaining].map(readRepoText).join('\n');
}

// Returns a class method from its signature through its closing brace.
// Scoping an ordered pattern to one method keeps it from matching text in a
// later method. A missing signature returns "", so the check fails closed.
export function methodBody(source, signature) {
  const start = source.indexOf(`\n  ${signature}`);
  if (start < 0) {
    return '';
  }
  const end = source.indexOf('\n  }\n', start);
  return source.slice(start, end > 0 ? end + '\n  }'.length : source.length);
}
