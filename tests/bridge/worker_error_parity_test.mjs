import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';

import baseline from './worker_error_parity_7d20000.json' with { type: 'json' };
import { MODEL_STATE, api, createBridge, createProxy, healthyImpl, outcome } from './core_retirement_fixtures.mjs';

const ABORT_TEXT_ERRORS = [
  'abort-text:Aborted()',
  'abort-text:RuntimeError: Aborted(undefined). Build with -sASSERTIONS for more info.',
  'abort-text:Aborted(native code called abort())',
  'abort-text:operation aborted(foo)',
  'abort-text:memory access out of bounds',
  'abort-text:unreachable',
  'abort-text:function signature mismatch',
  'abort-text:program terminated with exit(1)',
  'abort-text:same-realm WebAssembly.RuntimeError',
];

function recordingRuntime() {
  const calls = [];
  const base = {
    _modelBytes: 0,
    _runtimeNotes: [],
    _mmProjPath: null,
    _draftModel: null,
    calls,
    getModelMetadata: () => (base._modelBytes > 0 ? MODEL_STATE.metadata : {}),
    getContextSize: () => 128,
    isGpuActive: () => false,
    getBackendName: () => 'WASM (Prototype bridge)',
    supportsVision: () => false,
    supportsAudio: () => false,
    cancel() {},
  };
  return new Proxy(base, {
    get(target, property) {
      if (property in target || typeof property !== 'string' || property.startsWith('_') || property === 'then') {
        return target[property];
      }
      return async (...args) => {
        calls.push(property === 'loadModelFromUrl' ? `${property}(nGpuLayers=${args[1]?.nGpuLayers})` : property);
        if (property === 'loadModelFromUrl') {
          target._modelBytes = 1;
        }
        return `main-thread ${property}`;
      };
    },
  });
}

const media = [{ type: 'image', bytes: new Uint8Array(1) }];
const sites = {
  createCompletion: ['createCompletion', (b) => b.createCompletion('hi', {})],
  'createCompletion(media)': ['createCompletion', (b) => b.createCompletion('hi', { parts: media })],
  getCompletionCapabilities: ['getCompletionCapabilities', (b) => b.getCompletionCapabilities()],
  loadModelFromUrl: ['loadModelFromUrl', (b) => b.loadModelFromUrl('model-b.gguf', {})],
  tokenize: ['tokenize', (b) => b.tokenize('hi')],
  detokenize: ['detokenize', (b) => b.detokenize([1])],
  applyChatTemplate: ['applyChatTemplate', (b) => b.applyChatTemplate([{ role: 'user', content: 'hi' }])],
  embed: ['embed', (b) => b.embed('hi')],
  embedBatch: ['embedBatch', (b) => b.embedBatch(['hi'])],
  getEmbeddingCapabilities: ['getEmbeddingCapabilities', (b) => b.getEmbeddingCapabilities()],
  scoreNextToken: ['scoreNextToken', (b) => b.scoreNextToken('hi', {})],
  stateSaveBytes: ['stateSaveBytes', (b) => b.stateSaveBytes([1])],
  stateLoadBytes: ['stateLoadBytes', (b) => b.stateLoadBytes(new Uint8Array([1]), 8)],
  loadDraftModel: ['loadDraftModel', (b) => b.loadDraftModel('https://example.invalid/draft.gguf')],
  unloadDraftModel: ['unloadDraftModel', (b) => b.unloadDraftModel()],
  loadMultimodalProjector: ['loadMultimodalProjector', (b) => b.loadMultimodalProjector('https://example.invalid/p.gguf')],
  unloadMultimodalProjector: ['unloadMultimodalProjector', (b) => b.unloadMultimodalProjector()],
  getTextToSpeechCapabilities: ['getTextToSpeechCapabilities', (b) => b.getTextToSpeechCapabilities()],
  synthesizeSpeech: ['synthesizeSpeech', (b) => b.synthesizeSpeech({ text: 'hi' })],
  getDecisionCapabilities: ['getDecisionCapabilities', (b) => b.getDecisionCapabilities()],
  loadDecisionHead: ['loadDecisionHead', (b) => b.loadDecisionHead(new Uint8Array([1]))],
  getLoraAdapterCapabilities: ['getLoraAdapterCapabilities', (b) => b.getLoraAdapterCapabilities()],
  loadLoraAdapter: ['loadLoraAdapter', (b) => b.loadLoraAdapter('https://example.invalid/lora.gguf')],
  clearLoraAdapters: ['clearLoraAdapters', (b) => b.clearLoraAdapters()],
};

const flagged = (message, flag) => () => Object.assign(new Error(message), { [flag]: true });
const text = (message) => () => new Error(message);
const errors = {
  'crash-text:Bridge worker crashed': text('Bridge worker crashed'),
  'crash-flag': flagged('Uncaught ReferenceError: x is not defined', 'llamadartWorkerCrash'),
  'timeout-flag': flagged('stalled', 'llamadartWorkerTimeout'),
  'text:Worker request timeout (call, 5000ms)': text('Worker request timeout (call, 5000ms)'),
  'text:Bridge worker init timeout (3000ms)': text('Bridge worker init timeout (3000ms)'),
  'text:worker timed out': text('worker timed out'),
  'text:Bridge worker disposed': text('Bridge worker disposed'),
  'text:script error': text('Script error.'),
  'text:Generation step failed': text('Generation step failed: llama_decode failed while generating tokens'),
  'text:invalid grammar': text('Failed to start generation: Failed to initialize sampler chain (invalid grammar)'),
  'text:No model loaded': text('No model loaded. Call loadModelFromUrl first.'),
  'text:Array buffer allocation failed': text('Array buffer allocation failed'),
  'text:out of memory': text('out of memory'),
  'text:FS error': text('FS error'),
  'text:Model shard: 404 Not Found': text('Model shard: 404 Not Found'),
  'text:ggml_webgpu: device error': text('ggml_webgpu: device error'),
  'text:dispatch workgroup count': text('dispatch workgroup count exceeds limit'),
  'text:operation cancelled': text('operation cancelled'),
  'typed:ModelStagingError': () => new api.ModelStagingError('MODEL_STAGING_MOUNT_FAILED', 'mount failed'),
  'typed:SingleThreadCoreError': () => new api.SingleThreadCoreError(),
  'typed:EmbeddingError': () => new api.EmbeddingError('EMBEDDING_FAILED', 'embedding failed'),
  'typed:AbortError': () => new DOMException('The operation was aborted.', 'AbortError'),
};
for (const key of ABORT_TEXT_ERRORS) {
  const message = key.slice('abort-text:'.length);
  errors[key] = message === 'same-realm WebAssembly.RuntimeError'
    ? () => new WebAssembly.RuntimeError('unreachable')
    : text(message);
}

async function recordWorkerOutcomes() {
  const table = {};
  for (const gpu of [false, true]) {
    for (const [siteName, [method, run]] of Object.entries(sites)) {
      for (const [errorName, make] of Object.entries(errors)) {
        const failing = createProxy('worker', (called, args) => {
          if (called === method) {
            throw Object.assign(make(), { state: MODEL_STATE });
          }
          return healthyImpl(called, args, failing);
        });
        failing.hasModel = true;
        const runtimes = [];
        const { bridge, created } = createBridge({
          proxy: failing,
          overrides: {
            _config: { jsBackedModelStaging: true },
            _loadedModelUrl: 'model.gguf',
            _loadedModelOptions: { nGpuLayers: gpu ? 99 : 0 },
            _metadata: MODEL_STATE.metadata,
            _multimodalWorkerCpuMode: true,
            _createRuntime: () => {
              const runtime = recordingRuntime();
              runtimes.push(runtime);
              return runtime;
            },
          },
        });
        const result = await outcome(run(bridge));
        table[`${siteName}${gpu ? ' [gpu]' : ' [cpu]'} | ${errorName}`] = {
          result: result.ok
            ? `ok:${typeof result.value === 'string' ? result.value : typeof result.value}`
            : `${result.name}${result.code ? `/${result.code}` : ''}: ${result.message.slice(0, 90)}`,
          worker: bridge._workerProxy === failing ? 'kept' : bridge._workerProxy ? 'replaced' : 'none',
          workersCreated: created.length,
          workerDisposed: failing.disposeCalls,
          replacementsDisposed: created.filter((entry) => entry.disposeCalls > 0).length,
          replacementCalls: created.map((entry) => entry.calls.join(',')),
          mainThread: runtimes.map((runtime) => runtime.calls.join(',')),
          fallbackReason: bridge._workerFallbackReason,
          rememberedSource: bridge._loadedModelUrl,
          rememberedGpuLayers: bridge._loadedModelOptions?.nGpuLayers ?? null,
          workerModelMissing: bridge._workerModelMissing,
        };
      }
    }
  }
  return table;
}

const alwaysKeep = new Set([
  'createCompletion', 'getCompletionCapabilities', 'tokenize', 'detokenize',
  'applyChatTemplate', 'embed', 'embedBatch', 'getEmbeddingCapabilities',
  'scoreNextToken', 'loadDraftModel', 'unloadDraftModel',
]);
const abortCpuFallback = new Set(ABORT_TEXT_ERRORS.slice(0, 2));
const digest = (rows) => createHash('sha256').update(JSON.stringify(rows)).digest('hex');

function partitionRows(table) {
  const stable = {};
  const changed = [];
  for (const [key, row] of Object.entries(table)) {
    const [group, errorName] = key.split(' | ');
    const [site] = group.split(' [');
    const gpu = group.endsWith('[gpu]');
    if (
      ABORT_TEXT_ERRORS.includes(errorName) && (
        alwaysKeep.has(site) || (gpu && abortCpuFallback.has(errorName)
          && (site === 'createCompletion(media)' || site === 'synthesizeSpeech'))
      )
    ) {
      changed.push({ key, site, gpu, errorName, row });
      continue;
    }
    (stable[group] ??= []).push([errorName, row]);
  }
  return {
    changed,
    stable: Object.fromEntries(Object.entries(stable).map(([group, rows]) => [group, digest(rows)])),
  };
}

function expectedChangedRow({ site, gpu, errorName }) {
  const error = errors[errorName]();
  const media = site === 'createCompletion(media)';
  return {
    result: media ? 'ok:main-thread createCompletion' : `${error.name}: ${error.message.slice(0, 90)}`,
    worker: media ? 'none' : 'kept',
    workersCreated: 0,
    workerDisposed: media ? 1 : 0,
    replacementsDisposed: 0,
    replacementCalls: [],
    mainThread: media ? ['loadModelFromUrl(nGpuLayers=99),createCompletion'] : [],
    fallbackReason: media ? error.message : null,
    rememberedSource: 'model.gguf',
    rememberedGpuLayers: gpu ? 99 : 0,
    workerModelMissing: false,
  };
}

// Run with --record in the baseline checkout to regenerate the fixture.
if (import.meta.main && process.argv.includes('--record')) {
  console.log(JSON.stringify(partitionRows(await recordWorkerOutcomes()).stable, null, 2));
} else {
  test('worker errors preserve baseline outcomes except the obsolete abort-text rules', async () => {
    const table = await recordWorkerOutcomes();
    const { stable, changed } = partitionRows(table);
    assert.equal(Object.keys(table).length, 1488);
    assert.equal(Object.keys(stable).length, 48);
    assert.equal(changed.length, 202);
    assert.deepEqual(stable, baseline);
    for (const entry of changed) {
      assert.deepEqual(entry.row, expectedChangedRow(entry), entry.key);
    }
  });
}
