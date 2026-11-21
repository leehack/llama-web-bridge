import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { LlamaWebGpuBridge } from '../../js/src/bridge.ts';
import { guardCore, isCoreRetired, releaseCore } from '../../js/src/internal/core_guard.ts';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';

function fakeCore(events) {
  return {
    ccall: (name) => {
      events.push(name);
      return 0;
    },
    llamadartRetire: () => {
      events.push('retire-hook');
      return { threads: 4, stagedBytes: 0 };
    },
  };
}

test('releasing a guarded core runs its retirement hook once without reporting a retirement', () => {
  const events = [];
  let reported = 0;
  const core = guardCore(fakeCore(events), () => { reported += 1; });

  releaseCore(core);
  releaseCore(core);

  assert.deepEqual(events, ['retire-hook']);
  assert.equal(reported, 0);
  assert.equal(isCoreRetired(core), true);
  assert.throws(() => core.ccall('llamadart_webgpu_probe'), /The bridge core was disposed/);
  assert.deepEqual(events, ['retire-hook']);
});

test('a core that already retired is not released again', () => {
  const events = [];
  const raw = fakeCore(events);
  raw.ccall = () => { throw new WebAssembly.RuntimeError('unreachable'); };
  const core = guardCore(raw, () => {});
  assert.throws(() => core.ccall('llamadart_webgpu_probe'), WebAssembly.RuntimeError);

  releaseCore(core);

  assert.deepEqual(events, ['retire-hook']);
});

test('dispose releases a core the runtime imported itself', async (t) => {
  const events = [];
  globalThis.__coreReleaseTestCore = fakeCore(events);
  t.after(() => { delete globalThis.__coreReleaseTestCore; });
  let retirements = 0;
  const runtime = new LlamaWebGpuBridgeRuntime({
    allowSingleThreadCore: true,
    coreModuleUrl: 'data:text/javascript,export default async () => globalThis.__coreReleaseTestCore;',
  });
  runtime._onCoreRetired = () => { retirements += 1; };
  await runtime._ensureCore();
  events.length = 0;

  await runtime.dispose();

  assert.deepEqual(events.slice(-2), ['llamadart_webgpu_shutdown', 'retire-hook']);
  assert.equal(runtime._core, null);
  assert.equal(retirements, 0);
});

test('dispose leaves an application coreModuleFactory core running', async () => {
  const events = [];
  const raw = fakeCore(events);
  const runtime = new LlamaWebGpuBridgeRuntime({ coreModuleFactory: async () => raw });
  const core = await runtime._ensureCore();

  await runtime.dispose();

  assert.ok(events.includes('llamadart_webgpu_shutdown'));
  assert.ok(!events.includes('retire-hook'));
  assert.equal(isCoreRetired(core), false);
});

for (const { name, owned, failShutdown, failRelease } of [
  { name: 'releases a URL-loaded core after rejected shutdown', owned: true, failShutdown: true },
  { name: 'preserves the shutdown error if release also fails', owned: true, failShutdown: true, failRelease: true },
  { name: 'reports release failure after successful shutdown', owned: true, failRelease: true },
  { name: 'leaves an application core running after rejected shutdown', owned: false, failShutdown: true },
]) {
  test(`dispose ${name}`, async (t) => {
    const events = [];
    const shutdownError = new Error('shutdown rejected');
    const releaseError = new Error('release failed');
    const raw = fakeCore(events);
    const ccall = raw.ccall;
    const retire = raw.llamadartRetire;
    raw.ccall = (name) => {
      const result = ccall(name);
      if (name === 'llamadart_webgpu_shutdown' && failShutdown) {
        return Promise.reject(shutdownError);
      }
      return result;
    };
    raw.llamadartRetire = () => {
      const result = retire();
      if (failRelease) throw releaseError;
      return result;
    };
    globalThis.__coreReleaseTestCore = raw;
    t.after(() => { delete globalThis.__coreReleaseTestCore; });
    const source = owned
      ? { coreModuleUrl: 'data:text/javascript,export default async () => globalThis.__coreReleaseTestCore;' }
      : { coreModuleFactory: async () => raw };
    const bridge = new LlamaWebGpuBridge({ disableWorker: true, allowSingleThreadCore: true, ...source });
    await assert.rejects(bridge.loadModelFromUrl(''));
    const runtime = bridge._runtime;
    const core = runtime._core;
    assert.ok(core, 'The bridge must adopt the core before disposal');
    events.length = 0;

    const disposal = bridge.dispose();
    await assert.rejects(disposal, (error) => error === (failShutdown ? shutdownError : releaseError));
    assert.equal(bridge.dispose(), disposal);
    assert.deepEqual(events.filter((event) => event === 'llamadart_webgpu_shutdown' || event === 'retire-hook'),
      owned ? ['llamadart_webgpu_shutdown', 'retire-hook'] : ['llamadart_webgpu_shutdown']);
    assert.equal(isCoreRetired(core), owned);
    assert.equal(runtime._core, owned ? null : core);
  });
}

test('the retirement hook ends the pthread runtime, not only its workers', () => {
  const source = readFileSync(new URL('../../src/llama_webgpu_core_post.js', import.meta.url), 'utf8');
  const hook = source.slice(source.indexOf("Module['llamadartRetire']"));
  assert.match(hook, /try \{\s*PThread\.terminateRuntime\(\);\s*\} catch \(_\) \{\s*PThread\.terminateAllThreads\(\);\s*\}/);
});
