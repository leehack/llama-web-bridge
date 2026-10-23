import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  LlamaWebGpuBridge,
  enableBridgeWorkerHost,
} from '../../js/src/llama_webgpu_bridge.js';
import {
  createDirectBridge,
  createStubCore,
  tokensFor,
} from './bridge_operation_queue_fixtures.mjs';

const TEXT = 'ab\0cd';
const BYTES_EXPORTS = [
  '_llamadart_webgpu_tokenize_bytes_to_json',
  '_llamadart_webgpu_embed_bytes_to_json',
  '_llamadart_webgpu_score_next_token_bytes_to_json',
  '_llamadart_webgpu_begin_generation_bytes',
];

function olderCoreBridge() {
  const direct = createDirectBridge({ [TEXT]: ['x'], ab: ['x'] });
  for (const name of BYTES_EXPORTS) delete direct.core[name];
  return direct;
}

test('tokenize keeps the text after U+0000', async () => {
  const { bridge } = createDirectBridge();
  assert.deepEqual(await bridge.tokenize(TEXT, false), tokensFor(TEXT));
});

test('embed keeps the text after U+0000', async () => {
  const { bridge } = createDirectBridge();
  assert.deepEqual(await bridge.embed(TEXT), tokensFor(TEXT));
  assert.deepEqual(await bridge.embedBatch([TEXT, 'ab']), [tokensFor(TEXT), tokensFor('ab')]);
});

test('scoreNextToken keeps the prompt after U+0000', async () => {
  const { bridge } = createDirectBridge();
  const scores = await bridge.scoreNextToken(TEXT, { topK: 1 });
  assert.equal(scores.promptTokens, TEXT.length);
});

test('createCompletion keeps the prompt after U+0000', async () => {
  const { bridge, core } = createDirectBridge({ [TEXT]: ['x'] });
  const usages = [];
  const text = await bridge.createCompletion(TEXT, { onUsage: (usage) => usages.push(usage) });
  assert.equal(text, 'x');
  assert.deepEqual(core.trace.find(([kind]) => kind === 'begin'), ['begin', TEXT]);
  assert.equal(usages[0].promptTokens, TEXT.length);
});

test('the bytes transport keeps non-ASCII text and an empty string', async () => {
  const { bridge } = createDirectBridge();
  assert.deepEqual(await bridge.tokenize('\0🦙\0你', false), tokensFor('\0🦙\0你'));
  assert.deepEqual(await bridge.tokenize('', false), []);
});

test('a grammar or thinking tag holding U+0000 is rejected before the core', async () => {
  const { bridge, core } = createDirectBridge({ ab: ['x'] });
  await assert.rejects(
    bridge.createCompletion('ab', { grammar: 'root ::= "a\0"' }),
    /CompletionOptions\.grammar contains U\+0000/,
  );
  for (const field of ['startTag', 'endTag', 'forcedMessage']) {
    const thinkingBudget = { maxTokens: 1, startTag: '<think>', endTag: '</think>', [field]: 'x\0y' };
    await assert.rejects(
      bridge.createCompletion('ab', { thinkingBudget }),
      new RegExp(`CompletionOptions\\.thinkingBudget\\.${field} contains U\\+0000`),
    );
  }
  assert.equal(core.trace.some(([kind]) => kind === 'begin'), false);
});

test('a core without the bytes exports keeps the C-string calls for text without U+0000', async () => {
  const { bridge, core } = olderCoreBridge();
  assert.deepEqual(await bridge.tokenize('ab', false), tokensFor('ab'));
  assert.deepEqual(await bridge.embed('ab'), tokensFor('ab'));
  assert.equal((await bridge.scoreNextToken('ab', { topK: 1 })).promptTokens, 2);
  assert.equal(await bridge.createCompletion('ab'), 'x');
  assert.deepEqual(core.trace.find(([kind]) => kind === 'begin'), ['begin', 'ab']);
});

test('a core without the bytes exports rejects text holding U+0000 instead of cutting it', async () => {
  const { bridge, core } = olderCoreBridge();
  const missing = (name) => new RegExp(`U\\+0000.*llamadart_webgpu_${name}`);
  await assert.rejects(bridge.tokenize(TEXT, false), missing('tokenize_bytes_to_json'));
  await assert.rejects(bridge.embed(TEXT), missing('embed_bytes_to_json'));
  await assert.rejects(bridge.scoreNextToken(TEXT, { topK: 1 }), missing('score_next_token_bytes_to_json'));
  await assert.rejects(bridge.createCompletion(TEXT), missing('begin_generation_bytes'));
  assert.deepEqual(core.trace.filter(([kind]) => ['tokenize', 'embed', 'score', 'begin'].includes(kind)), []);
});

test('the worker runtime keeps the text after U+0000', async () => {
  const posted = [];
  const originalSelf = globalThis.self;
  const originalTokenize = LlamaWebGpuBridge.prototype.tokenize;
  try {
    globalThis.self = { postMessage: (message) => posted.push(message) };
    LlamaWebGpuBridge.prototype.tokenize = function tokenize(text, addSpecial) {
      if (!this._runtime._core) {
        this._runtime._core = createStubCore();
        this._runtime._modelBytes = 1;
      }
      return originalTokenize.call(this, text, addSpecial);
    };
    enableBridgeWorkerHost();
    await globalThis.self.onmessage({ data: { type: 'init', config: {} } });
    await globalThis.self.onmessage({ data: { type: 'call', id: 1, method: 'tokenize', args: [TEXT, false] } });
    const result = posted.find((message) => message.id === 1);
    assert.equal(result?.type, 'result', JSON.stringify(result));
    assert.deepEqual(result.value, tokensFor(TEXT));
  } finally {
    LlamaWebGpuBridge.prototype.tokenize = originalTokenize;
    globalThis.self = originalSelf;
  }
});
