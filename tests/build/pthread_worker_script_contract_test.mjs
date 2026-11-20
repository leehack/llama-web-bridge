import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

test('the core link options extend the Emscripten 6.0.8 incoming module API', () => {
  const cmake = readFileSync(new URL('../../CMakeLists.txt', import.meta.url), 'utf8');
  const flags = [...cmake.matchAll(/"-sINCOMING_MODULE_JS_API=\[([^\]]*)\]"/g)];
  assert.equal(flags.length, 1);
  const defaults = [
    'ENVIRONMENT', 'arguments', 'canvas', 'dynamicLibraries', 'elementPointerLock',
    'instantiateWasm', 'locateFile', 'monitorRunDependencies', 'noExitRuntime', 'noInitialRun',
    'onAbort', 'onExit', 'onRuntimeInitialized', 'postRun', 'preInit', 'preRun',
    'print', 'printErr', 'setStatus', 'statusMessage', 'stderr', 'stdin', 'stdout',
    'thisProgram', 'wasm', 'websocket',
  ];
  const incoming = flags[0][1].split(',').map((entry) => entry.trim().replace(/^'(.*)'$/, '$1'));
  assert.deepEqual(incoming.sort(), [...defaults, 'mainScriptUrlOrBlob'].sort());
  const shared = cmake.match(/^set\(LLAMADART_WEBGPU_LINK_OPTIONS\n([\s\S]*?)^\)/m)?.[1];
  assert.ok(shared?.includes(flags[0][0]));
});

test('the compiled pthread core reports the actual module URL before starting the pool', () => {
  const cmake = readFileSync(new URL('../../CMakeLists.txt', import.meta.url), 'utf8');
  const threaded = cmake.match(/^if \(LLAMADART_WEBGPU_PTHREADS\)\n([^]*?"--pre-js=[^"]+"[^]*?)^endif\(\)/m)?.[1];
  assert.match(threaded ?? '', /llama_webgpu_core_pthread_pre\.js/);
  assert.match(threaded ?? '', /APPEND PROPERTY LINK_DEPENDS/);
  const preJs = readFileSync(new URL('../../src/llama_webgpu_core_pthread_pre.js', import.meta.url), 'utf8');
  assert.match(preJs, /Module\[Symbol\.for\('llamadart\.pthreadScriptUrl'\)\]\?\.\(import\.meta\.url\)/);
});
