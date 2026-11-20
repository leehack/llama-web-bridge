import assert from 'node:assert/strict';

import { CrossOriginIsolationError, LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';

// Emscripten calls locateFile as a Module method, then runs preRun hooks before waiting for dependencies.
function fakeCoreUrl({ kind, pthreads, hangs = true, detachedLocator = false, wrapperPreRun = false }) {
  return `data:text/javascript,${encodeURIComponent(`export default async (options) => {
  let repaired = false;
  const module = ${wrapperPreRun ? '{ preRun: [() => { repaired = true; }], ...options }' : 'options'};
  const record = ((globalThis.__fakeCores ??= {})[${JSON.stringify(kind)}] ??= { exportCalls: 0 });
  ${pthreads === undefined ? '' : `module._llamadart_webgpu_supports_pthreads = () => { record.exportCalls += 1; ${pthreads} };`}
  ${detachedLocator ? 'const { locateFile } = module; locateFile("core.wasm", "/");' : 'module.locateFile("core.wasm", "/");'}
  for (const hook of [].concat(module.preRun ?? [])) hook(module);
  await new Promise((resolve) => { if (!${hangs} || repaired) setTimeout(resolve, 20); });
  module.kind = ${JSON.stringify(kind)};
  return module;
};`)}`;
}

async function onPage({ sharedMemory }, run) {
  const original = { MessageChannel: globalThis.MessageChannel, setTimeout: globalThis.setTimeout };
  if (!sharedMemory) {
    globalThis.MessageChannel = class {
      constructor() {
        this.port1 = {
          postMessage() {
            throw new DOMException('The object can not be cloned.', 'DataCloneError');
          },
          close() {},
        };
        this.port2 = { close() {} };
      }
    };
  }
  // Fire the 10 s core init timeout after 40 ms.
  globalThis.setTimeout = (fn, ms, ...args) => original.setTimeout(fn, ms >= 10000 ? 40 : ms, ...args);
  try {
    await run();
  } finally {
    globalThis.MessageChannel = original.MessageChannel;
    globalThis.setTimeout = original.setTimeout;
  }
}

const TIMEOUT = 'Bridge core init timeout (10000ms)';
const directBridge = (config) => new LlamaWebGpuBridge({
  disableWorker: true, allowSingleThreadCore: true, coreInitTimeoutMs: 10000, ...config,
});
const initError = (spec, config = {}) => directBridge({ coreModuleUrl: fakeCoreUrl(spec), ...config })
  ._runtime._ensureCore().then(() => assert.fail('must time out'), (error) => error);
const exportCalls = (kind) => globalThis.__fakeCores[kind].exportCalls;

await onPage({ sharedMemory: false }, async () => {
  // A pthread core that never finishes init keeps the timeout and gains the hint.
  {
    const bridge = directBridge({ coreModuleUrl: fakeCoreUrl({ kind: 'pthread-hang', pthreads: 'return 1;' }) });
    const error = await bridge._runtime._ensureCore().then(() => assert.fail('must time out'), (rejected) => rejected);
    assert.ok(error.message.startsWith(`${TIMEOUT}. This core is built with pthreads`), error.message);
    assert.match(error.message, /"_st"/);
    assert.match(error.message, /allowSingleThreadCore/);
    assert.match(error.message, /Cross-Origin-Embedder-Policy: require-corp/);
    assert.equal(error.constructor, Error);
    assert.ok(!(error instanceof CrossOriginIsolationError));
    assert.equal(error.code, undefined);
    const plain = new Error(TIMEOUT);
    assert.equal(bridge._shouldFallbackToMainThread(error), bridge._shouldFallbackToMainThread(plain));
    assert.equal(bridge._isRecoverableWorkerFsError(error), bridge._isRecoverableWorkerFsError(plain));
  }

  // Every other core keeps the timeout message unchanged.
  for (const spec of [
    { kind: 'st-hang', pthreads: 'return 0;' },
    { kind: 'no-export-hang' },
    { kind: 'throwing-export-hang', pthreads: 'throw new Error("not ready");' },
    { kind: 'pthread-detached-locator-hang', pthreads: 'return 1;', detachedLocator: true },
  ]) {
    assert.equal((await initError(spec)).message, TIMEOUT, spec.kind);
  }

  // The wasm64 fallback still records a timeout.
  {
    const bridge = directBridge({
      preferMemory64: true,
      coreModuleUrlMem64: fakeCoreUrl({ kind: 'pthread64-hang', pthreads: 'return 1;' }),
      coreModuleUrl: fakeCoreUrl({ kind: 'st32', pthreads: 'return 0;', hangs: false }),
    });
    assert.equal((await bridge._runtime._ensureCore()).kind, 'st32');
    assert.ok(bridge._runtime._runtimeNotes.includes('core_mem64_unavailable:timeout'));
  }

  // A core that finishes init is never asked for its flag.
  const finishingCores = [
    { kind: 'st', pthreads: 'return 0;', hangs: false },
    { kind: 'pthread-pool0', pthreads: 'return 1;', hangs: false },
    { kind: 'pthread-repaired', pthreads: 'return 1;', wrapperPreRun: true },
  ];
  for (const spec of finishingCores) {
    const core = await directBridge({ coreModuleUrl: fakeCoreUrl(spec) })._runtime._ensureCore();
    assert.equal(core.kind, spec.kind);
  }
  await new Promise((resolve) => setTimeout(resolve, 60));
  for (const spec of finishingCores) {
    assert.equal(exportCalls(spec.kind), 0);
  }
});

// A page that can post shared memory gets no capability probe and no hint.
await onPage({ sharedMemory: true }, async () => {
  assert.equal((await initError({ kind: 'pthread-isolated-hang', pthreads: 'return 1;' })).message, TIMEOUT);
  assert.equal(exportCalls('pthread-isolated-hang'), 0);
});
