// New behaviour: a pthread block inside the bridge worker crosses to the page with its code and reason.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';

const BLOCKED_CORE = `data:text/javascript,export default async (options) => {
  const Worker = options[Symbol.for('llamadart.pthreadWorker')] || globalThis.Worker;
  new Worker(new URL('https://cdn.test/llama_webgpu_core.js'), { type: 'module', name: 'em-pthread' });
  await new Promise(() => {});
};`;

test('the worker host reports BLOB_WORKER_BLOCKED and its reason', async (t) => {
  const replies = [];
  const events = new EventTarget();
  const globals = {
    origin: 'https://app.test',
    self: { postMessage: (message) => replies.push(message) },
    addEventListener: (type, listener) => events.addEventListener(type, listener),
    removeEventListener: (type, listener) => events.removeEventListener(type, listener),
    Worker: class extends EventTarget {
      constructor() {
        super();
        queueMicrotask(() => {
          events.dispatchEvent(Object.assign(new Event('securitypolicyviolation'), { blockedURI: 'blob', effectiveDirective: 'worker-src', disposition: 'enforce' }));
          this.dispatchEvent(new Event('error'));
        });
      }
      terminate() {}
    },
  };
  const originals = Object.fromEntries(Object.keys(globals).map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
  t.mock.method(URL, 'createObjectURL', () => 'blob:https://app.test/1');
  t.mock.method(URL, 'revokeObjectURL', () => {});
  t.after(() => {
    for (const [key, descriptor] of Object.entries(originals)) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
    }
  });
  installBridgeWorkerHost();
  await self.onmessage({ data: { type: 'init', config: { coreModuleUrl: BLOCKED_CORE, allowSingleThreadCore: true, workerStart: 'direct' } } });
  await self.onmessage({ data: { type: 'call', id: 1, method: 'loadModelFromUrl', args: ['https://models.test/m.gguf', { nGpuLayers: 0 }] } });
  const reply = replies.find((message) => message.id === 1);
  assert.equal(reply.type, 'error');
  assert.equal(reply.errorCode, 'BLOB_WORKER_BLOCKED');
  assert.equal(reply.blobWorkerBlockReason, 'csp');
  assert.match(reply.message, /worker-src blob:/);
});
