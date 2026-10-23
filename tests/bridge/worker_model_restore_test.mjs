import assert from 'node:assert/strict';

import { createWorkerBridge } from './bridge_operation_queue_fixtures.mjs';

// A worker whose core answers like llama_webgpu_core.cpp: a model load frees
// the decision heads, projector and draft model, and every free succeeds
// whether or not a model is loaded.
function createWorker() {
  const worker = {
    calls: [],
    model: null,
    heads: new Set(),
    nextHead: 41,
    async call(method, args) {
      worker.calls.push(method);
      switch (method) {
        case 'loadModelFromUrl':
          worker.model = args[0];
          worker.heads.clear();
          return undefined;
        case 'loadDecisionHead':
          worker.heads.add(worker.nextHead);
          return { apiVersion: 1, handle: worker.nextHead++ };
        case 'getDecisionCapabilities':
          return worker.model
            ? { apiVersion: 1, supported: true }
            : { apiVersion: 1, supported: false, reason: 'Model is not loaded. Load a ModernBERT encoder GGUF first.' };
        case 'runDecision':
          if (!worker.model) {
            throw new Error('No model loaded. Call loadModelFromUrl first.');
          }
          if (!worker.heads.has(args[0])) {
            throw new Error('Decision run failed: The decision head is not loaded; it was freed or its model was unloaded. Load the decision head again.');
          }
          return [];
        case 'freeDecisionHead':
          worker.heads.delete(args[0]);
          return undefined;
        case 'applyChatTemplate':
          return 'prompt';
        case 'tokenize':
          if (!worker.model) {
            throw new Error('No model loaded. Call loadModelFromUrl first.');
          }
          return [1];
        default:
          return undefined;
      }
    },
  };
  return worker;
}

// The state a cancelled multimodal CPU-mode setup leaves: the facade still
// holds the model, the worker no longer does.
function createBridgeWithMissingWorkerModel(overrides = {}) {
  const worker = createWorker();
  const bridge = createWorkerBridge({
    _loadedMmProjUrl: null,
    _callWorker: (method, args) => worker.call(method, args),
    ...overrides,
  });
  worker.model = 'model.gguf';
  const loseModel = () => {
    worker.model = null;
    worker.heads.clear();
    bridge._workerModelMissing = true;
    worker.calls.length = 0;
  };
  return { bridge, worker, loseModel };
}

// A capability probe answers for the facade's model, not for an empty worker.
{
  const { bridge, worker, loseModel } = createBridgeWithMissingWorkerModel();
  loseModel();
  assert.deepEqual(await bridge.getDecisionCapabilities(), { apiVersion: 1, supported: true });
  assert.deepEqual(worker.calls, ['loadModelFromUrl', 'getDecisionCapabilities']);
  assert.equal(bridge._workerModelMissing, false);
}

// A run reports the lost head, not a missing model, and leaves the worker
// ready for the head to be loaded again.
{
  const { bridge, worker, loseModel } = createBridgeWithMissingWorkerModel();
  const info = await bridge.loadDecisionHead(new Uint8Array([1]));
  loseModel();
  await assert.rejects(
    bridge.runDecision(info.handle, []),
    /Load the decision head again/,
  );
  assert.deepEqual(worker.calls, ['loadModelFromUrl', 'runDecision']);
  assert.equal(bridge._workerModelMissing, false);

  const reloaded = await bridge.loadDecisionHead(new Uint8Array([1]));
  assert.deepEqual(await bridge.runDecision(reloaded.handle, []), []);
}

// Frees, unloads and the model-independent chat template do not reload the
// model; unloading forgets the projector and draft so the next restore skips
// them.
{
  const { bridge, worker, loseModel } = createBridgeWithMissingWorkerModel({
    _loadedMmProjUrl: 'mmproj.gguf',
    _loadedDraftModel: { url: 'https://example.test/draft.gguf', options: {} },
  });
  const info = await bridge.loadDecisionHead(new Uint8Array([1]));
  loseModel();
  assert.equal(await bridge.applyChatTemplate([{ role: 'user', content: 'hi' }]), 'prompt');
  await bridge.freeDecisionHead(info.handle);
  await bridge.unloadMultimodalProjector();
  await bridge.unloadDraftModel();
  assert.deepEqual(worker.calls, [
    'applyChatTemplate',
    'freeDecisionHead',
    'unloadMultimodalProjector',
    'unloadDraftModel',
  ]);
  assert.equal(bridge._workerModelMissing, true);
  assert.equal(bridge._loadedMmProjUrl, null);
  assert.equal(bridge._loadedDraftModel, null);

  worker.calls.length = 0;
  assert.deepEqual(await bridge.tokenize('x'), [1]);
  assert.deepEqual(worker.calls, ['loadModelFromUrl', 'tokenize']);
}
