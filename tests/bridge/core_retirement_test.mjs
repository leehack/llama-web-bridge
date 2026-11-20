import assert from 'node:assert/strict';
import test from 'node:test';
import { ModelUnloadedError } from '../../js/src/llama_webgpu_bridge.js';

import {
  EMPTY_STATE,
  LlamaWebGpuBridge,
  MODEL_STATE,
  api,
  createBridge,
  createProxy,
  healthyImpl,
  outcome,
  topology,
} from './core_retirement_fixtures.mjs';

function check(label, condition, detail = '') {
  assert.ok(condition, detail ? `${label}: ${detail}` : label);
}

// A worker whose core aborts in `method`: the host posts `core-retired` before
// the reply, which the proxy records as `coreRetired`.
function abortingProxy(name, method, message = 'memory access out of bounds') {
  const proxy = createProxy(name, (called, args) => {
    if (called === method) {
      proxy.coreRetired = true;
      throw Object.assign(new Error(message), { state: EMPTY_STATE });
    }
    return healthyImpl(called, args, proxy);
  });
  return proxy;
}

const loaded = {
  _loadedModelUrl: 'model.gguf',
  _loadedModelOptions: { nGpuLayers: 0 },
  _metadata: MODEL_STATE.metadata,
};

const wasmTrap = () => new WebAssembly.RuntimeError('unreachable');

test('direct mode end to end (fake core through coreModuleFactory)', async () => {
  const counts = { total: 0, retireHook: 0 };
  const core = { llamadartRetire: () => { counts.retireHook += 1; } };
  const strings = {
    llamadart_webgpu_last_error: '',
    llamadart_webgpu_tokenize_to_json: '[1,2]',
  };
  let aborting = false;
  const trapError = wasmTrap();
  core.ccall = (name, returnType) => {
    counts.total += 1;
    if (aborting && name.startsWith('llamadart_webgpu_tokenize')) {
      throw trapError;
    }
    return returnType === 'string' ? (strings[name] ?? '[1,2]') : 0;
  };
  const bridge = new LlamaWebGpuBridge({ disableWorker: true, coreInitTimeoutMs: 1000, coreModuleFactory: async () => core });
  const first = bridge._runtime;
  await first._ensureCore();
  first._modelBytes = 1;
  bridge._loadedModelUrl = 'model.gguf';
  bridge._loraAdapters.set(1, { owner: first, handle: 3, source: null, cacheOptions: {} });
  bridge._decisionHeads.set(1, { owner: first, handle: 2 });
  const healthy = await bridge.tokenize('hi').then((value) => ({ ok: true, value }), (error) => ({ ok: false, error }));
  check('healthy call works through the guard', healthy.ok, String(healthy.error?.message));

  aborting = true;
  const failed = await bridge.tokenize('hi').catch((error) => error);
  const callsAtAbort = counts.total;
  check('failing call rejects with the original RuntimeError', failed === trapError, String(failed?.message));
  check('runtime replaced once, retire hook once', bridge._runtime !== first && bridge._coreRetirements === 1
    && counts.retireHook === 1);
  check('notes carried to the new runtime', bridge._runtime._runtimeNotes.includes('core_retired'));
  check('metadata snapshot shows no model', !(Number(bridge.getModelMetadata()['llamadart.webgpu.model_bytes']) > 0));

  const next = await bridge.tokenize('hi').catch((error) => error);
  check('next call rejects with ModelUnloadedError', next instanceof ModelUnloadedError
    && next.code === 'MODEL_UNLOADED_AFTER_CORE_ABORT', String(next?.message));
  const lora = await bridge.setLoraAdapter(1, 1).catch((error) => error);
  check('LoRA handle is gone', lora instanceof ModelUnloadedError);
  const head = await bridge.runDecision(1, [{ tokens: [1] }]).catch((error) => error);
  check('decision head rejects head-lost', head?.code === 'head-lost', String(head?.code));
  // The retired runtime stays unusable through any stale reference.
  const stale = await first.tokenize('hi').catch((error) => error);
  check('stale runtime reference rethrows the abort, no native call', stale === trapError
    && counts.total === callsAtAbort, `${counts.total - callsAtAbort} extra calls`);
  check('zero native calls into the retired core after the abort', counts.total === callsAtAbort);
  await bridge.dispose();
  check('dispose after an abort makes no native call into the retired core', counts.total === callsAtAbort,
    `${counts.total - callsAtAbort} extra calls`);
});

test('direct mode: onAbort with no throwing call', async () => {
  const counts = { total: 0, retireHook: 0 };
  const core = {
    ccall() { counts.total += 1; return 0; },
    llamadartRetire() { counts.retireHook += 1; },
  };
  let moduleOptions = null;
  const bridge = new LlamaWebGpuBridge({
    disableWorker: true,
    coreInitTimeoutMs: 1000,
    coreModuleFactory: async (options) => {
      moduleOptions = options;
      return core;
    },
  });
  const first = bridge._runtime;
  await first._ensureCore();
  first._modelBytes = 1;
  bridge._loadedModelUrl = 'model.gguf';
  const callsBefore = counts.total;
  // A pthread's abort reaches the page only through onAbort.
  moduleOptions.onAbort('native code called abort()');
  check('onAbort alone swaps the runtime once', bridge._runtime !== first && bridge._coreRetirements === 1
    && counts.retireHook === 1);
  const next = await bridge.tokenize('hi').catch((error) => error);
  check('and the next call rejects with ModelUnloadedError without a native call',
    next instanceof ModelUnloadedError && counts.total === callsBefore);
  check('the abort note survives the swap',
    bridge._runtime._runtimeNotes.some((note) => note.startsWith('core_abort')));
  await bridge.dispose();
});

test('direct mode: callbacks of a discarded core candidate', async () => {
  const moduleUrl = (source) => `data:text/javascript,${encodeURIComponent(source)}`;
  let hookCalls = 0;
  globalThis.coreRetirementTestCore = {
    llamadartRetire() {
      hookCalls += 1;
    },
    ccall: (name, returnType) => (returnType === 'string' ? '[7]' : 0),
  };
  const bridge = new LlamaWebGpuBridge({
    disableWorker: true,
    logLevel: 0,
    coreInitTimeoutMs: 1000,
    coreModuleUrlMem64: moduleUrl(`export default async ({ onAbort }) => {
      globalThis.coreRetirementTestDiscardedAbort = onAbort;
      throw new Error('memory64 factory failed');
    }`),
    coreModuleUrl: moduleUrl(`export default async ({ onAbort }) => {
      globalThis.coreRetirementTestAdoptedAbort = onAbort;
      return globalThis.coreRetirementTestCore;
    }`),
  });
  const runtime = bridge._runtime;
  await runtime._ensureCore();
  runtime._modelBytes = 1;
  bridge._loadedModelUrl = 'model.gguf';
  check('the wasm32 fallback was adopted', runtime._coreVariant === 'wasm32');
  globalThis.coreRetirementTestDiscardedAbort('late abort from the rejected memory64 candidate');
  const after = await bridge.tokenize('hi').then((value) => value, (error) => error);
  check('an abort of the discarded candidate leaves the adopted core in use',
    Array.isArray(after) && after[0] === 7 && bridge._runtime === runtime && hookCalls === 0
    && !(bridge._coreRetirements > 0), String(after?.message ?? JSON.stringify(after)));
  check('and is still noted, as on main',
    runtime._runtimeNotes.some((note) => note.startsWith('core_abort:late_abort')));
  globalThis.coreRetirementTestAdoptedAbort('native code called abort()');
  const next = await bridge.tokenize('hi').then((value) => value, (error) => error);
  check('the adopted candidate\'s own abort still retires it', bridge._runtime !== runtime
    && bridge._coreRetirements === 1 && hookCalls === 1 && next instanceof ModelUnloadedError);
  await bridge.dispose();
});

test('retirement reply in createCompletion: replace once, restore once', async () => {
  const dead = abortingProxy('dead', 'createCompletion');
  dead.hasModel = true;
  const { bridge, created, runtimes } = createBridge({ proxy: dead, overrides: { ...loaded } });
  const failed = await outcome(bridge.createCompletion('hi', {}));
  const after = topology(bridge, dead, created, runtimes);
  check('failing call rejects with the worker error object text', failed.ok === false
    && failed.message === 'memory access out of bounds', JSON.stringify(failed));
  check('exactly one worker created', after.workersCreated === 1, JSON.stringify(after));
  check('dead worker disposed once, replacement kept', after.originalDisposed === 1
    && after.replacementsDisposed === 0 && after.worker === 'replaced(replacement-1)');
  check('no direct runtime, no fallback reason', after.directRuntimes === 0 && after.fallbackReason === null);
  check('source remembered, model marked missing', after.rememberedSource === 'model.gguf'
    && after.workerModelMissing === true);
  check('getters keep the model snapshot', after.modelBytes === '1024');
  check('no restore yet', created[0].calls.filter((name) => name === 'loadModelFromUrl').length === 0);

  const next = await outcome(bridge.createCompletion('hi', {}));
  const third = await outcome(bridge.createCompletion('hi', {}));
  check('next call succeeds on the replacement', next.ok && next.value === 'text from replacement-1');
  check('restore ran exactly once across two calls',
    created[0].calls.filter((name) => name === 'loadModelFromUrl').length === 1 && third.ok,
    created[0].calls.join(','));
  check('still one worker created', created.length === 1 && bridge._workerModelMissing === false);
});

test('every worker-backed method takes the same path', async () => {
  const cases = [
    ['createCompletion', (bridge) => bridge.createCompletion('hi', {})],
    ['getCompletionCapabilities', (bridge) => bridge.getCompletionCapabilities()],
    ['tokenize', (bridge) => bridge.tokenize('hi')],
    ['detokenize', (bridge) => bridge.detokenize([1])],
    ['applyChatTemplate', (bridge) => bridge.applyChatTemplate([{ role: 'user', content: 'hi' }])],
    ['embed', (bridge) => bridge.embed('hi')],
    ['embedBatch', (bridge) => bridge.embedBatch(['hi'])],
    ['getEmbeddingCapabilities', (bridge) => bridge.getEmbeddingCapabilities()],
    ['scoreNextToken', (bridge) => bridge.scoreNextToken('hi', {})],
    ['stateSaveBytes', (bridge) => bridge.stateSaveBytes([1])],
    ['stateLoadBytes', (bridge) => bridge.stateLoadBytes(new Uint8Array([1]), 8)],
    ['loadDraftModel', (bridge) => bridge.loadDraftModel('https://example.invalid/draft.gguf')],
    ['unloadDraftModel', (bridge) => bridge.unloadDraftModel()],
    ['loadMultimodalProjector', (bridge) => bridge.loadMultimodalProjector('https://example.invalid/mmproj.gguf')],
    ['unloadMultimodalProjector', (bridge) => bridge.unloadMultimodalProjector()],
    ['getTextToSpeechCapabilities', (bridge) => bridge.getTextToSpeechCapabilities()],
    ['synthesizeSpeech', (bridge) => bridge.synthesizeSpeech({ text: 'hi' })],
    ['getDecisionCapabilities', (bridge) => bridge.getDecisionCapabilities()],
    ['loadDecisionHead', (bridge) => bridge.loadDecisionHead(new Uint8Array([1]))],
    ['getLoraAdapterCapabilities', (bridge) => bridge.getLoraAdapterCapabilities()],
    ['loadLoraAdapter', (bridge) => bridge.loadLoraAdapter('https://example.invalid/lora.gguf')],
    ['clearLoraAdapters', (bridge) => bridge.clearLoraAdapters()],
  ];
  for (const [method, run] of cases) {
    // CPU-loaded, so decision D1's CPU rerun does not apply and the call rejects.
    const dead = abortingProxy('dead', method);
    dead.hasModel = true;
    const { bridge, created, runtimes } = createBridge({
      proxy: dead,
      overrides: { ...loaded, _multimodalWorkerCpuMode: true },
    });
    const failed = await outcome(run(bridge));
    const after = topology(bridge, dead, created, runtimes);
    const single = failed.ok === false
      && failed.message.includes('memory access out of bounds')
      && after.workersCreated === 1
      && after.originalDisposed === 1
      && after.replacementsDisposed === 0
      && after.directRuntimes === 0
      && after.fallbackReason === null
      && after.rememberedSource === 'model.gguf'
      && after.workerModelMissing === true;
    check(`${method}: rejects, one replacement, no fallback, source kept`, single,
      JSON.stringify({ failed: { ok: failed.ok, message: failed.message }, after }));
  }
});

test('a retirement reply wins over every text classifier', async () => {
  const sites = [
    ['createCompletion', (bridge) => bridge.createCompletion('hi', {})],
    ['createCompletion(media)', (bridge) => bridge.createCompletion('hi', { parts: [{ type: 'image', bytes: new Uint8Array(1) }] }), 'createCompletion'],
    ['loadModelFromUrl', (bridge) => bridge.loadModelFromUrl('model-b.gguf', {})],
    ['tokenize', (bridge) => bridge.tokenize('hi')],
    ['synthesizeSpeech', (bridge) => bridge.synthesizeSpeech({ text: 'hi' })],
    ['loadMultimodalProjector', (bridge) => bridge.loadMultimodalProjector('https://example.invalid/mmproj.gguf')],
    ['unloadMultimodalProjector', (bridge) => bridge.unloadMultimodalProjector()],
    ['getTextToSpeechCapabilities', (bridge) => bridge.getTextToSpeechCapabilities()],
    ['loadDecisionHead', (bridge) => bridge.loadDecisionHead(new Uint8Array([1]))],
    ['loadLoraAdapter', (bridge) => bridge.loadLoraAdapter('https://example.invalid/lora.gguf')],
  ];
  const messages = [
    'Bridge worker crashed',
    'FS error: no such file',
    'Worker request timeout (call, 5000ms)',
    'worker timed out',
    'ggml_webgpu: device error',
  ];
  for (const [site, run, method = site] of sites) {
    const bad = [];
    for (const message of messages) {
      const dead = abortingProxy('dead', method, message);
      dead.hasModel = true;
      const { bridge, created, runtimes } = createBridge({
        proxy: dead,
        overrides: { ...loaded, _multimodalWorkerCpuMode: true },
      });
      const failed = await outcome(run(bridge));
      const after = topology(bridge, dead, created, runtimes);
      if (!(failed.ok === false && failed.message.includes(message) && after.workersCreated === 1
        && after.originalDisposed === 1 && after.replacementsDisposed === 0 && after.directRuntimes === 0
        && after.fallbackReason === null && after.rememberedSource === 'model.gguf'
        && after.workerModelMissing === true)) {
        bad.push(`${message}: ${JSON.stringify({ ok: failed.ok, message: failed.message, after })}`);
      }
    }
    check(`${site}: one replacement whatever the reply text says`, bad.length === 0, bad.join(' || '));
  }
});

test('healthy errors never discard the worker', async () => {
  const texts = [
    'operation aborted(foo)',
    'Aborted()',
    'RuntimeError: Aborted(undefined). Build with -sASSERTIONS for more info.',
    'memory access out of bounds',
    'unreachable',
    'function signature mismatch',
    'program terminated with exit(1)',
  ];
  for (const text of texts) {
    const healthy = createProxy('healthy', (method, args) => {
      if (method === 'createCompletion') {
        throw Object.assign(new Error(text), { state: MODEL_STATE });
      }
      return healthyImpl(method, args, healthy);
    });
    healthy.hasModel = true;
    const { bridge, created, runtimes } = createBridge({ proxy: healthy, overrides: { ...loaded } });
    const failed = await outcome(bridge.createCompletion('hi', {}));
    const after = topology(bridge, healthy, created, runtimes);
    check(`"${text}" without a retirement signal keeps worker and model`,
      failed.ok === false && failed.message === text && after.worker === 'kept'
      && after.workersCreated === 0 && after.directRuntimes === 0 && after.originalDisposed === 0
      && after.rememberedSource === 'model.gguf' && after.workerModelMissing === false,
      JSON.stringify(after));
  }
  // Same-realm RuntimeError object from a fake proxy, still no signal.
  const healthy = createProxy('healthy', (method, args) => {
    if (method === 'tokenize') {
      throw new WebAssembly.RuntimeError('unreachable');
    }
    return healthyImpl(method, args, healthy);
  });
  const { bridge, created, runtimes } = createBridge({ proxy: healthy, overrides: { ...loaded } });
  const failed = await outcome(bridge.tokenize('hi'));
  check('a RuntimeError instance without the signal is rethrown as is',
    failed.ok === false && failed.error instanceof WebAssembly.RuntimeError
    && topology(bridge, healthy, created, runtimes).worker === 'kept');
});

test('"Aborted()" text alone no longer selects the CPU fallback', async () => {
  const gpu = { ...loaded, _loadedModelOptions: { nGpuLayers: 99 }, _multimodalWorkerCpuMode: true };
  const speech = createProxy('healthy', (method, args) => {
    if (method === 'synthesizeSpeech') {
      throw Object.assign(new Error('Aborted()'), { state: MODEL_STATE });
    }
    return healthyImpl(method, args, speech);
  });
  speech.hasModel = true;
  const tts = createBridge({ proxy: speech, overrides: { ...gpu } });
  const failed = await outcome(tts.bridge.synthesizeSpeech({ text: 'hi' }));
  check('speech on a GPU model: rethrown, worker kept, no main-thread reload', failed.ok === false
    && failed.message === 'Aborted()' && tts.bridge._workerProxy === speech && tts.runtimes.length === 0);
});

test('guards that are not reached through a classifier', async () => {
  // The core aborts while the worker reloads the model for media use.
  const dead = abortingProxy('dead', 'loadModelFromUrl');
  dead.hasModel = true;
  const setup = createBridge({ proxy: dead, overrides: { ...loaded, _multimodalWorkerCpuMode: false } });
  const failed = await outcome(setup.bridge.loadMultimodalProjector('https://example.invalid/mmproj.gguf'));
  check('abort during the media-mode reload: one replacement, model still missing', failed.ok === false
    && setup.created.length === 1 && setup.bridge._workerModelMissing === true
    && setup.created[0].calls.length === 0, `${setup.created.length} created; ${setup.created[0]?.calls.join(',')}`);

  // The same abort reached through a media completion (CPU-loaded, so no rerun).
  const dead2 = abortingProxy('dead', 'loadModelFromUrl');
  dead2.hasModel = true;
  const media = createBridge({ proxy: dead2, overrides: { ...loaded, _multimodalWorkerCpuMode: false } });
  const mediaFailed = await outcome(media.bridge.createCompletion('hi', {
    parts: [{ type: 'image', bytes: new Uint8Array(1) }],
  }));
  check('abort during media setup: the original error, one replacement, no fallback',
    mediaFailed.ok === false && mediaFailed.message === 'memory access out of bounds'
    && media.created.length === 1 && media.runtimes.length === 0 && media.bridge._workerModelMissing === true,
    JSON.stringify({ message: mediaFailed.message, created: media.created.length, runtimes: media.runtimes.length }));

  // The core aborts while a restore reloads the draft model.
  let abortDraft = true;
  const restoring = createProxy('restoring', (method, args) => {
    if (method === 'loadDraftModel' && abortDraft) {
      restoring.coreRetired = true;
      throw Object.assign(new Error('unreachable'), { state: EMPTY_STATE });
    }
    return healthyImpl(method, args, restoring);
  });
  const draft = createBridge({
    proxy: restoring,
    overrides: {
      ...loaded,
      _workerModelMissing: true,
      _loadedDraftModel: { url: 'https://example.invalid/draft.gguf', options: {} },
    },
  });
  const first = await outcome(draft.bridge.tokenize('hi'));
  check('abort during the draft reload: rejects, draft forgotten, model still missing', first.ok === false
    && draft.bridge._loadedDraftModel === null && draft.bridge._workerModelMissing === true
    && draft.created.length === 1, JSON.stringify({ ok: first.ok, missing: draft.bridge._workerModelMissing }));
  abortDraft = false;
  const second = await outcome(draft.bridge.tokenize('hi'));
  check('the next call restores without the draft', second.ok
    && draft.created[0].calls.join(',') === 'loadModelFromUrl,tokenize', draft.created[0].calls.join(','));

  // A retirement reported for a proxy that is no longer current changes nothing.
  const current = createProxy('current', healthyImpl);
  const stale = createProxy('stale', healthyImpl);
  const idle = createBridge({ proxy: current, overrides: { ...loaded } });
  await idle.bridge._replaceWorkerAfterCoreRetired(stale);
  check('a stale proxy never replaces the current worker', idle.bridge._workerProxy === current
    && idle.created.length === 0 && current.disposeCalls === 0);
  const directOnly = createBridge({ overrides: { ...loaded, _runtime: {} } });
  await directOnly.bridge._replaceWorkerAfterCoreRetired(stale);
  check('nor creates a worker for a facade that already fell back', directOnly.created.length === 0
    && directOnly.bridge._workerProxy === null);
});

test('a load that aborts keeps the remembered source', async () => {
  const dead = abortingProxy('dead', 'loadModelFromUrl', 'Aborted(native code called abort())');
  const { bridge, created, runtimes } = createBridge({
    proxy: dead,
    overrides: { ...loaded, _loadedModelUrl: 'model-a.gguf' },
  });
  const failed = await outcome(bridge.loadModelFromUrl('model-b.gguf', {}));
  const after = topology(bridge, dead, created, runtimes);
  check('load rejects with the abort text', failed.ok === false
    && failed.message === 'Aborted(native code called abort())');
  check('one replacement', after.workersCreated === 1 && after.worker === 'replaced(replacement-1)');
  check('model A still remembered and marked missing',
    after.rememberedSource === 'model-a.gguf' && after.workerModelMissing === true, JSON.stringify(after));
  const next = await outcome(bridge.createCompletion('hi', {}));
  check('next call restores model A once and runs', next.ok
    && created[0].calls.join(',') === 'loadModelFromUrl,createCompletion', created[0].calls.join(','));

  // First load ever aborts: nothing to restore.
  const first = abortingProxy('first', 'loadModelFromUrl', 'Aborted(native code called abort())');
  const fresh = createBridge({ proxy: first });
  await outcome(fresh.bridge.loadModelFromUrl('model-b.gguf', {}));
  check('first-ever load abort leaves nothing remembered',
    fresh.bridge._loadedModelUrl === null && fresh.created.length === 1);
  const reload = await outcome(fresh.bridge.loadModelFromUrl('model-b.gguf', {}));
  check('the caller can load again on the replacement', reload.ok
    && fresh.bridge._loadedModelUrl === 'model-b.gguf' && fresh.created.length === 1
    && fresh.bridge._workerModelMissing === false);

  // The restore itself fails without an abort: source stays, next call retries.
  let restoreFails = 1;
  const dead2 = abortingProxy('dead', 'tokenize');
  const retry = createBridge({
    proxy: dead2,
    overrides: { ...loaded },
    nextProxy: (index) => {
      const replacement = createProxy(`replacement-${index}`, (method, args) => {
        if (method === 'loadModelFromUrl' && restoreFails > 0) {
          restoreFails -= 1;
          throw Object.assign(new Error('Model shard: 503 Service Unavailable'), { state: EMPTY_STATE });
        }
        return healthyImpl(method, args, replacement);
      });
      return replacement;
    },
  });
  await outcome(retry.bridge.tokenize('hi'));
  const failedRestore = await outcome(retry.bridge.tokenize('hi'));
  check('failed restore rejects and keeps the source', failedRestore.ok === false
    && retry.bridge._loadedModelUrl === 'model.gguf' && retry.bridge._workerModelMissing === true,
    JSON.stringify(topology(retry.bridge, dead2, retry.created, retry.runtimes)));
  const recovered = await outcome(retry.bridge.tokenize('hi'));
  check('the following call restores and runs', recovered.ok && retry.created.length === 1);
});

test('typed error after a direct-mode abort', async () => {
  const { ModelUnloadedError } = api;
  check('ModelUnloadedError is exported', typeof ModelUnloadedError === 'function');
  const bridge = new LlamaWebGpuBridge({ disableWorker: true });
  const before = bridge._runtime;
  before._modelBytes = 1;
  bridge._loadedModelUrl = 'model.gguf';
  // What the guard does when a guarded call traps.
  before._handleCoreRetired(new WebAssembly.RuntimeError('unreachable'));
  check('runtime swapped exactly once', bridge._runtime !== before && bridge._coreRetirements === 1);
  before._handleCoreRetired(new WebAssembly.RuntimeError('unreachable'));
  check('a second signal does not swap again', bridge._coreRetirements === 1);
  const current = bridge._runtime;
  bridge._handleRuntimeCoreRetired(before);
  check('a runtime that is no longer current does not swap the facade', bridge._runtime === current
    && bridge._coreRetirements === 1);
  check('model forgotten', bridge._loadedModelUrl === null);
  const failed = await outcome(bridge.createCompletion('hi', {}));
  check('next call rejects with ModelUnloadedError', failed.error instanceof ModelUnloadedError
    && failed.name === 'ModelUnloadedError' && failed.code === 'MODEL_UNLOADED_AFTER_CORE_ABORT',
    JSON.stringify(failed));
  const again = await outcome(bridge.tokenize('hi'));
  check('it stays until a load succeeds', again.error instanceof ModelUnloadedError);
  const decision = await outcome(bridge.loadDecisionHead(new Uint8Array([1])));
  check('decision calls keep their own no-model code', decision.code === 'no-model', JSON.stringify(decision));
  const capabilities = await outcome(bridge.getLoraAdapterCapabilities().catch((error) => { throw error; }));
  check('capability probes are not blocked', !(capabilities.error instanceof ModelUnloadedError));
  bridge._runtime.loadModelFromUrl = async function load() { this._modelBytes = 1; };
  bridge._runtime.tokenize = async () => [7];
  await bridge.loadModelFromUrl('model.gguf', {});
  const afterLoad = await outcome(bridge.tokenize('hi'));
  check('a successful load clears it', afterLoad.ok && afterLoad.value[0] === 7);
  await bridge.dispose();

  // Kinds: exactly the calls that say "No model loaded" on a fresh direct bridge.
  const methods = {
    createCompletion: (b) => b.createCompletion('hi', {}),
    tokenize: (b) => b.tokenize('hi'),
    detokenize: (b) => b.detokenize([1]),
    embed: (b) => b.embed('hi'),
    embedBatch: (b) => b.embedBatch(['hi']),
    scoreNextToken: (b) => b.scoreNextToken('hi', {}),
    stateSaveFile: (b) => b.stateSaveFile('/s.bin', [1]),
    stateLoadFile: (b) => b.stateLoadFile('/s.bin', 8),
    stateSaveBytes: (b) => b.stateSaveBytes([1]),
    stateLoadBytes: (b) => b.stateLoadBytes(new Uint8Array([1]), 8),
    loadMultimodalProjector: (b) => b.loadMultimodalProjector('https://example.invalid/p.gguf'),
    unloadMultimodalProjector: (b) => b.unloadMultimodalProjector(),
    loadDraftModel: (b) => b.loadDraftModel('https://example.invalid/d.gguf'),
    unloadDraftModel: (b) => b.unloadDraftModel(),
    synthesizeSpeech: (b) => b.synthesizeSpeech({ text: 'hi' }),
    loadLoraAdapter: (b) => b.loadLoraAdapter('https://example.invalid/l.gguf'),
    clearLoraAdapters: (b) => b.clearLoraAdapters(),
    applyChatTemplate: (b) => b.applyChatTemplate([{ role: 'user', content: 'hi' }]),
    getCompletionCapabilities: (b) => b.getCompletionCapabilities(),
    getEmbeddingCapabilities: (b) => b.getEmbeddingCapabilities(),
    getTextToSpeechCapabilities: (b) => b.getTextToSpeechCapabilities(),
    getDecisionCapabilities: (b) => b.getDecisionCapabilities(),
    getLoraAdapterCapabilities: (b) => b.getLoraAdapterCapabilities(),
    loadDecisionHead: (b) => b.loadDecisionHead(new Uint8Array([1])),
  };
  const NO_MODEL = 'No model loaded. Call loadModelFromUrl first.';
  const mismatches = [];
  for (const [name, run] of Object.entries(methods)) {
    const freshBridge = new LlamaWebGpuBridge({ disableWorker: true });
    const fresh = await outcome(run(freshBridge));
    await freshBridge.dispose().catch(() => {});
    const abortedBridge = new LlamaWebGpuBridge({ disableWorker: true });
    abortedBridge._runtime._handleCoreRetired(new WebAssembly.RuntimeError('unreachable'));
    const aborted = await outcome(run(abortedBridge));
    await abortedBridge.dispose().catch(() => {});
    const plainNoModel = fresh.ok === false && fresh.message === NO_MODEL && fresh.code === undefined;
    const typed = aborted.error instanceof ModelUnloadedError;
    if (plainNoModel !== typed) {
      mismatches.push(name);
    }
  }
  check('typed error replaces exactly the plain "No model loaded" rejections', mismatches.length === 0,
    mismatches.join(','));
});

test('typed errors and the signal cross the worker boundary', async () => {
  for (const [label, make, test] of [
    ['ModelStagingError', () => new api.ModelStagingError('MODEL_STAGING_MOUNT_FAILED', 'mount'),
      (error) => error instanceof api.ModelStagingError && error.code === 'MODEL_STAGING_MOUNT_FAILED'],
    ['EmbeddingError', () => new api.EmbeddingError('EMBEDDING_FAILED', 'embed'),
      (error) => error instanceof api.EmbeddingError && error.code === 'EMBEDDING_FAILED'],
  ]) {
    const typed = make();
    const dead = createProxy('dead', (method, args) => {
      if (method === 'embed' || method === 'loadModelFromUrl') {
        dead.coreRetired = true;
        throw Object.assign(typed, { state: EMPTY_STATE });
      }
      return healthyImpl(method, args, dead);
    });
    const { bridge, created } = createBridge({
      proxy: dead,
      overrides: { ...loaded, _config: { jsBackedModelStaging: true } },
    });
    const failed = await outcome(label === 'EmbeddingError'
      ? bridge.embed('hi')
      : bridge.loadModelFromUrl('model-b.gguf', {}));
    check(`${label} keeps identity, type and code across the replacement`,
      failed.error === typed && test(failed.error) && created.length === 1, JSON.stringify(failed));
  }
  // Cancellation wins over the retirement, as it does over every worker error.
  const controller = new AbortController();
  const dead = createProxy('dead', (method, args) => {
    if (method === 'createCompletion') {
      controller.abort();
      dead.coreRetired = true;
      throw Object.assign(new Error('unreachable'), { state: EMPTY_STATE });
    }
    return healthyImpl(method, args, dead);
  });
  const { bridge, created } = createBridge({ proxy: dead, overrides: { ...loaded } });
  const failed = await outcome(bridge.createCompletion('hi', { signal: controller.signal }));
  check('a cancelled call still rejects with AbortError and the worker is replaced once',
    failed.name === 'AbortError' && created.length === 1 && bridge._workerModelMissing === true,
    JSON.stringify({ name: failed.name, created: created.length }));
});

test('handles across a worker-mode retirement', async () => {
  let aborting = false;
  const first = createProxy('first', (method, args) => {
    if (method === 'createCompletion' && aborting) {
      first.coreRetired = true;
      throw Object.assign(new Error('unreachable'), { state: EMPTY_STATE });
    }
    if (method === 'loadDecisionHead') {
      return { value: { handle: 5 } };
    }
    return healthyImpl(method, args, first);
  });
  const { bridge, created } = createBridge({ proxy: first });
  await bridge.loadModelFromUrl('model.gguf', {});
  const url = await bridge.loadLoraAdapter('https://example.invalid/adapter.gguf');
  await bridge.setLoraAdapter(url.handle, 0.5);
  const inactive = await bridge.loadLoraAdapter('https://example.invalid/inactive.gguf');
  const head = await bridge.loadDecisionHead(new Uint8Array([1]));
  aborting = true;
  await outcome(bridge.createCompletion('hi', {}));
  const next = await outcome(bridge.createCompletion('hi', {}));
  const replacement = created[0];
  check('one replacement; restore reloads model and the active adapter',
    created.length === 1 && next.ok
    && replacement.calls.join(',') === 'loadModelFromUrl,clearLoraAdapters,loadLoraAdapter,setLoraAdapter,createCompletion',
    replacement.calls.join(','));
  check('active LoRA facade handle stays valid and now belongs to the replacement',
    bridge._loraAdapters.get(url.handle)?.owner === replacement);
  const setAgain = await outcome(bridge.setLoraAdapter(inactive.handle, 0.25));
  check('inactive LoRA handle reloads on first use', setAgain.ok
    && bridge._loraAdapters.get(inactive.handle)?.owner === replacement, JSON.stringify(setAgain));
  const run = await outcome(bridge.runDecision(head.handle, [{ tokens: [1] }]));
  check('decision head from the retired worker rejects head-lost and never reaches the new worker',
    run.code === 'head-lost' && !replacement.calls.includes('runDecision'), JSON.stringify(run));

  // An adapter loaded from bytes with workers disabled cannot be restored: direct mode.
  const direct = new LlamaWebGpuBridge({ disableWorker: true });
  direct._runtime._modelBytes = 1;
  direct._loadedModelUrl = 'model.gguf';
  direct._loraAdapters.set(1, { owner: direct._runtime, handle: 3, source: null, cacheOptions: {} });
  direct._decisionHeads.set(1, { owner: direct._runtime, handle: 2 });
  direct._runtime._handleCoreRetired(new WebAssembly.RuntimeError('unreachable'));
  check('direct mode forgets every LoRA handle', direct._loraAdapters.size === 0);
  const lost = await outcome(direct.runDecision(1, [{ tokens: [1] }]));
  check('direct mode decision handle rejects head-lost', lost.code === 'head-lost', JSON.stringify(lost));
  await direct.dispose();
});

test('a core that retires between calls', async () => {
  const first = createProxy('first', healthyImpl);
  first.hasModel = true;
  const { bridge, created } = createBridge({ proxy: first, overrides: { ...loaded } });
  first.coreRetired = true;
  const next = await outcome(bridge.tokenize('hi'));
  check('next call replaces first, restores once and succeeds', next.ok && created.length === 1
    && created[0].calls.join(',') === 'loadModelFromUrl,tokenize' && first.calls.length === 0,
    `${created[0]?.calls.join(',')} | first: ${first.calls.join(',')}`);

  const idle = createProxy('idle', healthyImpl);
  idle.hasModel = true;
  const second = createBridge({ proxy: idle, overrides: { ...loaded, _loadedModelUrl: 'model-a.gguf' } });
  idle.coreRetired = true;
  const load = await outcome(second.bridge.loadModelFromUrl('model-b.gguf', {}));
  check('a load after an idle retirement runs on the replacement and is remembered',
    load.ok && second.created.length === 1 && idle.calls.length === 0
    && second.bridge._loadedModelUrl === 'model-b.gguf' && second.bridge._workerModelMissing === false);
});

test('a reply that succeeds although the core retired', async () => {
  const dead = createProxy('dead', (method, args) => {
    if (method === 'stateLoadBytes') {
      dead.coreRetired = true;
      return { value: { tokens: [] } };
    }
    return healthyImpl(method, args, dead);
  });
  dead.hasModel = true;
  const { bridge, created } = createBridge({ proxy: dead, overrides: { ...loaded } });
  const failed = await outcome(bridge.stateLoadBytes(new Uint8Array([1]), 8));
  check('rejects, replaces once, marks the model missing', failed.ok === false
    && /core aborted during stateLoadBytes/.test(failed.message) && created.length === 1
    && bridge._workerModelMissing === true, JSON.stringify(failed));
});

test('media completion and speech on a GPU-loaded model rerun once on CPU', async () => {
  const gpuLoaded = { ...loaded, _loadedModelOptions: { nGpuLayers: 99, nCtx: 8192 }, _multimodalWorkerCpuMode: true };
  for (const [label, run, method] of [
    ['media completion', (b) => b.createCompletion('hi', { parts: [{ type: 'image', bytes: new Uint8Array(1) }] }), 'createCompletion'],
    ['synthesizeSpeech', (b) => b.synthesizeSpeech({ text: 'hi' }), 'synthesizeSpeech'],
  ]) {
    const dead = abortingProxy('dead', method, 'Aborted()');
    dead.hasModel = true;
    const loads = [];
    const { bridge, created, runtimes } = createBridge({
      proxy: dead,
      overrides: { ...gpuLoaded },
      nextProxy: (index) => {
        const replacement = createProxy(`replacement-${index}`, (called, args) => {
          if (called === 'loadModelFromUrl') {
            loads.push(args[1]);
          }
          if (called === 'synthesizeSpeech') {
            return { value: { pcm: new Float32Array([0.25]), sampleRate: 24000, channels: 1 }, state: MODEL_STATE };
          }
          return healthyImpl(called, args, replacement);
        });
        return replacement;
      },
    });
    const result = await outcome(run(bridge));
    check(`${label}: resolves from the replacement worker`, result.ok && created.length === 1
      && runtimes.length === 0 && bridge._workerProxy === created[0], JSON.stringify(result));
    check(`${label}: one CPU-only restore`, loads.length === 1 && loads[0].nGpuLayers === 0
      && loads[0].nCtx === 4096, JSON.stringify(loads));

    // A second abort on the CPU rerun rejects; no third worker beyond that retirement's own.
    const stubborn = abortingProxy('dead', method, 'Aborted()');
    stubborn.hasModel = true;
    const again = createBridge({
      proxy: stubborn,
      overrides: { ...gpuLoaded },
      nextProxy: (index) => {
        const replacement = abortingProxy(`replacement-${index}`, method, 'Aborted()');
        return replacement;
      },
    });
    const failed = await outcome(run(again.bridge));
    check(`${label}: a rerun that aborts again rejects after exactly two replacements`,
      failed.ok === false && again.created.length === 2 && again.runtimes.length === 0,
      JSON.stringify({ ok: failed.ok, created: again.created.length }));
  }
  // Text completion never reruns.
  const dead = abortingProxy('dead', 'createCompletion', 'Aborted()');
  dead.hasModel = true;
  const text = createBridge({ proxy: dead, overrides: { ...gpuLoaded } });
  const failed = await outcome(text.bridge.createCompletion('hi', {}));
  check('text completion rejects and keeps its load options', failed.ok === false
    && text.bridge._loadedModelOptions.nGpuLayers === 99 && text.created.length === 1);
});

test('streamed speech retries a retired core only before delivering PCM', async () => {
  for (const deliverAudio of [false, true]) {
    const original = new Error('Aborted()');
    const pcm = new Float32Array([0.25]);
    const dead = abortingProxy('dead', 'synthesizeSpeech', original.message);
    dead.hasModel = true;
    dead.call = async (method, args, onEvent) => {
      assert.equal(method, 'synthesizeSpeech');
      assert.equal(args[0].streamAudio, true);
      onEvent({ event: 'progress', payload: deliverAudio ? { pcm, sampleOffset: 0 } : { state: 2 } });
      dead.coreRetired = true;
      throw original;
    };
    const loads = [];
    const { bridge, created, runtimes } = createBridge({
      proxy: dead,
      overrides: { ...loaded, _loadedModelOptions: { nGpuLayers: 99 } },
      nextProxy() {
        const replacement = createProxy('replacement', (method, args) => {
          if (method === 'loadModelFromUrl') loads.push(args[1]);
          if (method === 'synthesizeSpeech') return { value: { pcm: new Float32Array(0), sampleCount: 0 } };
          return healthyImpl(method, args, replacement);
        });
        return replacement;
      },
    });
    const chunks = [];
    const result = await outcome(bridge.synthesizeSpeech({
      text: 'Hello.',
      streamAudio: true,
      onProgress(event) { if (event.pcm) chunks.push(event.pcm); },
    }));
    assert.equal(created.length, 1);
    assert.equal(runtimes.length, 0);
    if (deliverAudio) {
      assert.equal(result.error, original);
      assert.equal(original.llamadartCoreRetired, true);
      assert.deepEqual(chunks, [pcm]);
      assert.deepEqual(created[0].calls.filter((method) => method !== 'setLogLevel'), []);
      assert.equal(bridge._loadedModelOptions.nGpuLayers, 99);
    } else {
      assert.equal(result.ok, true);
      assert.equal(loads.length, 1);
      assert.equal(loads[0].nGpuLayers, 0);
      assert.ok(created[0].calls.includes('synthesizeSpeech'));
    }
    await bridge.dispose();
  }
});

test('a streamed callback failure keeps the replacement for a retired worker that timed out', async () => {
  const failure = new Error('consumer failed');
  const timeout = new Error('Worker request timeout (synthesizeSpeech, 5000ms)');
  const dead = createProxy('dead', () => { throw timeout; });
  dead.hasModel = true;
  dead.call = async (method, _args, onEvent) => {
    if (method === 'cancel') return { value: true };
    assert.equal(method, 'synthesizeSpeech');
    onEvent({ event: 'progress', payload: { state: 1 } });
    dead.coreRetired = true;
    throw timeout;
  };
  const { bridge, created } = createBridge({ proxy: dead, overrides: { ...loaded } });
  const result = await outcome(bridge.synthesizeSpeech({
    text: 'Hello.', streamAudio: true,
    onProgress() { throw failure; },
  }));
  assert.equal(result.error, failure);
  assert.equal(timeout.llamadartCoreRetired, true);
  assert.equal(dead.disposeCalls, 1);
  assert.equal(created.length, 1);
  assert.equal(created[0].disposeCalls, 0);
  assert.equal(bridge._workerProxy, created[0]);
  assert.ok(!created[0].calls.includes('synthesizeSpeech'));
  await bridge.dispose();
});

test('no replacement worker can be created', async () => {
  const dead = abortingProxy('dead', 'tokenize');
  dead.hasModel = true;
  const { bridge, created, runtimes } = createBridge({
    proxy: dead,
    overrides: {
      ...loaded,
      _createWorkerProxy: () => {
        throw Object.assign(new Error('Worker construction violates the Content Security Policy'), {
          name: 'SecurityError',
        });
      },
    },
  });
  const failed = await outcome(bridge.tokenize('hi'));
  check('failing call still rejects with the abort text', failed.ok === false
    && failed.message === 'memory access out of bounds', JSON.stringify(failed));
  check('facade is direct, holds no model, and the dead worker is disposed', bridge._workerProxy === null
    && runtimes.length === 1 && dead.disposeCalls === 1 && bridge._loadedModelUrl === null && created.length === 0);
  const next = await outcome(bridge.tokenize('hi'));
  check('next call rejects with ModelUnloadedError', next.error instanceof api.ModelUnloadedError,
    JSON.stringify(next));
});

test('generation cleanup rethrows the original trap without re-entering native code', async () => {
  const calls = [];
  const trapError = new WebAssembly.RuntimeError('memory access out of bounds');
  const core = {
    HEAPU8: new Uint8Array(1024),
    stackSave: () => 0,
    stackRestore: () => {},
    stackAlloc: () => 8,
    _malloc: () => 8,
    _free: () => {},
    lengthBytesUTF8: (text) => text.length,
    stringToUTF8: () => {},
    FS: { unlink() {}, writeFile() {} },
    ccall(name, returnType) {
      calls.push(name);
      if (name === 'llamadart_webgpu_next_token') {
        throw trapError;
      }
      return returnType === 'string' ? '' : 0;
    },
  };
  const bridge = new LlamaWebGpuBridge({ disableWorker: true, coreInitTimeoutMs: 1000, coreModuleFactory: async () => core });
  const runtime = bridge._runtime;
  await runtime._ensureCore();
  runtime._modelBytes = 1;
  const error = await bridge.createCompletion('hi', { nPredict: 4 }).then(() => null, (thrown) => thrown);
  assert.equal(error, trapError);
  assert.equal(calls.at(-1), 'llamadart_webgpu_next_token');
  await bridge.dispose();
  assert.equal(calls.at(-1), 'llamadart_webgpu_next_token');
});
