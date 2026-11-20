import { registerHooks } from 'node:module';

const proxyModule = new URL('../../js/src/worker_proxy.ts', import.meta.url);
const startModule = new URL('../../js/src/internal/worker_start.ts', import.meta.url);

export async function workerStartModules(tag) {
  const proxyUrl = new URL(`?case=${tag}`, proxyModule).href;
  const startUrl = new URL(`?case=${tag}`, startModule).href;
  const hooks = registerHooks({
    resolve(specifier, context, nextResolve) {
      if (specifier === './internal/worker_start.ts' && context.parentURL === proxyUrl) {
        return { url: startUrl, shortCircuit: true };
      }
      return nextResolve(specifier, context);
    },
  });
  try {
    return { ...await import(proxyUrl), ...await import(startUrl) };
  } finally {
    hooks.deregister();
  }
}

export function stubWorkerStart(t, options = {}) {
  const state = { workers: [], blobs: [], revoked: [], events: new EventTarget(), listeners: new Set() };
  const globals = {
    origin: 'https://app.test',
    window: { location: { href: 'https://app.test/page/' } },
    addEventListener(type, listener) {
      state.listeners.add(listener);
      state.events.addEventListener(type, listener);
    },
    removeEventListener(type, listener) {
      state.listeners.delete(listener);
      state.events.removeEventListener(type, listener);
    },
    Worker: class {
      constructor(url, workerOptions) {
        this.url = url;
        this.options = workerOptions;
        this.posted = [];
        this.terminated = false;
        state.workers.push(this);
        options.construct?.(this, state.workers.length);
      }
      postMessage(message) {
        this.posted.push(message);
        if (message.method === 'dispose') {
          this.onmessage({ data: { type: 'result', id: message.id } });
        }
      }
      terminate() {
        this.terminated = true;
      }
    },
  };
  const originals = Object.fromEntries(Object.keys(globals).map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, value] of Object.entries(globals)) {
    Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
  }
  t.after(() => {
    for (const [key, descriptor] of Object.entries(originals)) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
  t.mock.method(URL, 'createObjectURL', (blob) => {
    state.blobs.push(blob);
    return `blob:https://app.test/${state.blobs.length}`;
  });
  t.mock.method(URL, 'revokeObjectURL', (url) => state.revoked.push(url));
  state.violation = () => state.events.dispatchEvent(Object.assign(new Event('securitypolicyviolation'), {
    blockedURI: 'blob', effectiveDirective: 'worker-src', disposition: 'enforce',
  }));
  return state;
}
