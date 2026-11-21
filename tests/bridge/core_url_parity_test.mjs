// Characterization of how main's runtime hands URL options to import() and to the core. Passes on main.
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';
const LOADER = new URL('../../js/src/internal/core_loader.ts', import.meta.url).href;

// Serves a fake core for any specifier in `specifiers` and records what import() was asked for.
function fakeCores(t, specifiers) {
  const seen = [];
  const serial = (fakeCores.serial = (fakeCores.serial ?? 0) + 1);
  globalThis.__parityCoreArgs = [];
  const hook = registerHooks({
    resolve(specifier, context, nextResolve) {
      if (!specifiers.includes(specifier)) return nextResolve(specifier, context);
      seen.push({ specifier, parentURL: context.parentURL });
      return { url: `fake-core:${serial}-${specifiers.indexOf(specifier)}`, shortCircuit: true };
    },
    load(url, context, nextLoad) {
      if (!url.startsWith('fake-core:')) return nextLoad(url, context);
      return { format: 'module', shortCircuit: true, source: 'export default async (arg) => { globalThis.__parityCoreArgs.push(arg); return { ccall: () => 0 }; };' };
    },
  });
  t.after(() => { hook.deregister(); delete globalThis.__parityCoreArgs; });
  return seen;
}

const forms = { relative: './dir/core', 'root-relative': '/dir/core', 'absolute same-origin': 'https://app.test/dir/core', 'absolute CDN': 'https://cdn.test/dir/core', bare: 'llama-core' };

for (const [form, base] of Object.entries(forms)) {
  test(`coreModuleUrl ${form}: import() gets the caller's specifier from the core loader module`, async (t) => {
    const specifier = `${base}.js`;
    const seen = fakeCores(t, [specifier]);
    const runtime = new LlamaWebGpuBridgeRuntime({ allowSingleThreadCore: true, coreModuleUrl: specifier });
    await runtime._ensureCore();
    assert.deepEqual(seen, [{ specifier, parentURL: LOADER }]);
    assert.equal(globalThis.__parityCoreArgs.length, 1);
    assert.equal(runtime._coreVariant, 'wasm32');
  });

  test(`coreModuleUrlMem64 ${form}: import() gets the caller's specifier, before the wasm32 one`, async (t) => {
    const mem64 = `${base}_mem64.js`;
    const seen = fakeCores(t, [mem64, 'https://app.test/core32.js']);
    const runtime = new LlamaWebGpuBridgeRuntime({ allowSingleThreadCore: true, preferMemory64: true, coreModuleUrlMem64: mem64, coreModuleUrl: 'https://app.test/core32.js' });
    runtime._preferMemory64 = true;
    await runtime._ensureCore();
    assert.deepEqual(seen.map((s) => s.specifier), [mem64, 'https://app.test/core32.js']);
    assert.ok(seen.every((s) => s.parentURL === LOADER));
    assert.equal(globalThis.__parityCoreArgs.length, 1);
    assert.equal(runtime._coreVariant, 'wasm64');
  });

  test(`wasmUrl ${form}: locateFile returns the caller's value untouched`, async (t) => {
    const wasmUrl = `${base}.wasm`;
    fakeCores(t, ['https://app.test/core32.js']);
    const runtime = new LlamaWebGpuBridgeRuntime({ allowSingleThreadCore: true, coreModuleUrl: 'https://app.test/core32.js', wasmUrl });
    await runtime._ensureCore();
    const [arg] = globalThis.__parityCoreArgs;
    assert.equal(arg.locateFile('llama_webgpu_core.wasm', 'https://cdn.test/prefix/'), wasmUrl);
    assert.equal(arg.locateFile('other.data', 'https://cdn.test/prefix/'), 'https://cdn.test/prefix/other.data');
  });

  test(`wasmUrlMem64 ${form}: the memory64 core gets it, and falls back to wasmUrl`, async (t) => {
    const wasmUrlMem64 = `${base}_mem64.wasm`;
    for (const config of [{ wasmUrlMem64, wasmUrl: 'https://app.test/32.wasm' }, { wasmUrl: wasmUrlMem64 }]) {
      fakeCores(t, ['https://app.test/core64.js', 'https://app.test/core32.js']);
      const runtime = new LlamaWebGpuBridgeRuntime({ allowSingleThreadCore: true, preferMemory64: true, coreModuleUrlMem64: 'https://app.test/core64.js', coreModuleUrl: 'https://app.test/core32.js', ...config });
      runtime._preferMemory64 = true;
      await runtime._ensureCore();
      assert.equal(globalThis.__parityCoreArgs.at(-1).locateFile('x.wasm', 'p/'), wasmUrlMem64);
    }
  });
}

test('no wasmUrl: locateFile keeps the core prefix', async (t) => {
  fakeCores(t, ['https://app.test/core32.js']);
  const runtime = new LlamaWebGpuBridgeRuntime({ allowSingleThreadCore: true, coreModuleUrl: 'https://app.test/core32.js' });
  await runtime._ensureCore();
  assert.equal(globalThis.__parityCoreArgs[0].locateFile('llama_webgpu_core.wasm', 'https://cdn.test/prefix/'), 'https://cdn.test/prefix/llama_webgpu_core.wasm');
});

test('the core factory is called once with exactly the Emscripten options main passes', async (t) => {
  fakeCores(t, ['https://cdn.test/core.js']);
  const runtime = new LlamaWebGpuBridgeRuntime({ allowSingleThreadCore: true, coreModuleUrl: 'https://cdn.test/core.js' });
  await runtime._ensureCore();
  await runtime._ensureCore();
  assert.equal(globalThis.__parityCoreArgs.length, 1);
  const [arg] = globalThis.__parityCoreArgs;
  assert.deepEqual(Object.keys(arg), ['locateFile', 'print', 'printErr', 'onAbort']);
  assert.deepEqual(Object.getOwnPropertyNames(arg), ['locateFile', 'print', 'printErr', 'onAbort']);
  assert.equal(Object.getOwnPropertyDescriptor(arg, 'mainScriptUrlOrBlob'), undefined);
});

test('a custom coreModuleFactory gets exactly the options main passes and no pthread hook', async () => {
  const args = [];
  const runtime = new LlamaWebGpuBridgeRuntime({ coreModuleFactory: async (arg) => { args.push(arg); return { ccall: () => 0 }; } });
  await runtime._ensureCore();
  assert.deepEqual(Reflect.ownKeys(args[0]), ['locateFile', 'print', 'printErr', 'onAbort']);
});
