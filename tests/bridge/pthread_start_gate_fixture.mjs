import { registerHooks } from 'node:module';

import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';

export async function withCrossOriginCore({
  readsWorkerScript, blocked = false, coreUrl = 'https://cdn.test/core.js', actualUrl = coreUrl,
}, run) {
  const source = `export default async (options) => {
    globalThis.__pthreadStartGateCalls.push(options);
    if (${readsWorkerScript}) options[Symbol.for('llamadart.pthreadScriptUrl')]?.(${JSON.stringify(actualUrl)});
    return { script: ${readsWorkerScript} ? options.mainScriptUrlOrBlob : undefined };
  };`;
  const hook = registerHooks({
    resolve(specifier, context, nextResolve) {
      return specifier === coreUrl ? { url: specifier, shortCircuit: true } : nextResolve(specifier, context);
    },
    load(url, context, nextLoad) {
      return url === coreUrl ? { format: 'module', source, shortCircuit: true } : nextLoad(url, context);
    },
  });
  const coreCalls = [];
  const state = { coreCalls, blobs: [], workers: [] };
  const fakeGlobals = {
    origin: 'https://app.test',
    __pthreadStartGateCalls: coreCalls,
    Worker: class {
      constructor(url, options) {
        state.workers.push({ url, options });
        queueMicrotask(() => {
          if (blocked) this.onerror?.({});
          else this.onmessage?.({ data: 0 });
        });
      }
      terminate() {}
    },
  };
  const originals = Object.fromEntries(Object.keys(fakeGlobals).map((key) => [key, globalThis[key]]));
  const { createObjectURL, revokeObjectURL } = URL;
  Object.assign(globalThis, fakeGlobals);
  URL.createObjectURL = (blob) => {
    state.blobs.push(blob);
    return `blob:https://app.test/${state.blobs.length}`;
  };
  URL.revokeObjectURL = () => {};
  try {
    await run(new LlamaWebGpuBridgeRuntime({
      allowSingleThreadCore: true,
      coreInitTimeoutMs: 10_000,
      coreModuleUrl: coreUrl,
    }), state);
  } finally {
    for (const [key, value] of Object.entries(originals)) {
      if (value === undefined) delete globalThis[key];
      else globalThis[key] = value;
    }
    Object.assign(URL, { createObjectURL, revokeObjectURL });
    hook.deregister();
  }
}
