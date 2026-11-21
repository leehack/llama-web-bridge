// Static contract: threaded cores read their pthread Worker constructor from the hook; nothing else about the link changes.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../..', import.meta.url));
const cmake = await readFile(`${root}/CMakeLists.txt`, 'utf8');

test('the pre-js shadows Worker with the hook, falling back to the global constructor', async () => {
  const source = await readFile(`${root}/src/llama_webgpu_core_pthread_pre.js`, 'utf8');
  const code = source.split('\n').filter((line) => line.trim() && !line.trim().startsWith('//')).join('\n');
  assert.equal(code, "var Worker = Module[Symbol.for('llamadart.pthreadWorker')] || globalThis.Worker;");
});

test('only threaded builds link the pre-js, and a change to it relinks the core', () => {
  assert.equal(cmake.match(/--pre-js=/g)?.length, 1);
  const block = cmake.match(/if \(LLAMADART_WEBGPU_PTHREADS\)\n((?:(?!endif\(\)).*\n)*?)endif\(\)/g).find((b) => b.includes('--pre-js='));
  assert.ok(block, 'the pre-js is not inside if (LLAMADART_WEBGPU_PTHREADS)');
  assert.match(block, /"--pre-js=\$\{LLAMADART_WEBGPU_CORE_PTHREAD_PRE_JS\}"/);
  assert.match(block, /set\(LLAMADART_WEBGPU_CORE_PTHREAD_PRE_JS "\$\{CMAKE_CURRENT_SOURCE_DIR\}\/src\/llama_webgpu_core_pthread_pre\.js"\)/);
  assert.match(block, /APPEND PROPERTY LINK_DEPENDS "\$\{LLAMADART_WEBGPU_CORE_PTHREAD_PRE_JS\}"/);
});

test('the link keeps Emscripten\'s default incoming module API', () => {
  assert.doesNotMatch(cmake, /INCOMING_MODULE_JS_API/);
  assert.doesNotMatch(cmake, /mainScriptUrlOrBlob/);
});
