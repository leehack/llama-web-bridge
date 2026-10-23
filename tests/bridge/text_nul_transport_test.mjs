import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  LlamaWebGpuBridge,
  enableBridgeWorkerHost,
} from '../../js/src/llama_webgpu_bridge.js';
import {
  createDirectBridge,
  createStubCore,
  createWorkerBridge,
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

// The worker host installs once per module, so every worker test shares one
// worker global; each call starts a fresh bridge with `init`.
const workerPosted = [];
const workerSelf = { postMessage: (message) => workerPosted.push(message) };

async function callWorkerHost(method, args) {
  const originalSelf = globalThis.self;
  globalThis.self = workerSelf;
  try {
    enableBridgeWorkerHost();
    await workerSelf.onmessage({ data: { type: 'init', config: {} } });
    workerPosted.length = 0;
    await workerSelf.onmessage({ data: { type: 'call', id: 1, method, args } });
    const result = workerPosted.find((message) => message.id === 1 && message.type !== 'event');
    assert.equal(result?.type, 'result', JSON.stringify(result));
    return result.value;
  } finally {
    globalThis.self = originalSelf;
  }
}

// Gives the worker's bridge `core` on its first call of `method`.
async function withWorkerCore(method, createCore, run) {
  const original = LlamaWebGpuBridge.prototype[method];
  LlamaWebGpuBridge.prototype[method] = function withCore(...args) {
    if (!this._runtime._core) {
      createCore(this._runtime);
      this._runtime._modelBytes = 1;
    }
    return original.apply(this, args);
  };
  try {
    return await run();
  } finally {
    LlamaWebGpuBridge.prototype[method] = original;
  }
}

test('the worker runtime keeps the text after U+0000', async () => {
  const value = await withWorkerCore(
    'tokenize',
    (runtime) => { runtime._core = createStubCore(); },
    () => callWorkerHost('tokenize', [TEXT, false]),
  );
  assert.deepEqual(value, tokensFor(TEXT));
});

// Speech and decision exports on top of the stub core; their 'string'
// arguments reach C as NUL-terminated strings.
function withSpeechAndDecisionExports(core) {
  const ccall = core.ccall.bind(core);
  core.calls = [];
  core._llamadart_webgpu_decision_capabilities_json = () => {};
  core.ccall = (name, retType, argTypes = [], args = []) => {
    const cArgs = args.map((arg, index) => (
      argTypes[index] === 'string' && typeof arg === 'string' ? arg.split('\0')[0] : arg
    ));
    switch (name) {
      case 'llamadart_webgpu_tts_info_json':
        return JSON.stringify({ supported: true, sampleRate: 24000, channels: 1 });
      case 'llamadart_webgpu_tts_start':
        core.calls.push([name, ...cArgs]);
        core.lastError = 'stub stops after start';
        return Promise.resolve(-1);
      case 'llamadart_webgpu_decision_api_version':
        return 1;
      case 'llamadart_webgpu_decision_capabilities_json':
        return JSON.stringify({ apiVersion: 1, supported: true });
      case 'llamadart_webgpu_decision_load':
        core.calls.push([name, ...cArgs]);
        return Promise.resolve(3);
      case 'llamadart_webgpu_decision_head_info_json':
        return JSON.stringify({ apiVersion: 1, handle: cArgs[0] });
      default:
        return ccall(name, retType, argTypes, args);
    }
  };
  return core;
}

function speechAndDecisionBridge() {
  const { bridge, core } = createDirectBridge();
  withSpeechAndDecisionExports(core);
  bridge._runtime._fetchWithTimeout = async () => new Response(new Uint8Array([1, 2, 3]));
  return { bridge, core };
}

const coreCalls = (core, name) => core.calls.filter(([called]) => called === name);

test('synthesizeSpeech rejects text or language holding U+0000 before the core', async () => {
  const { bridge, core } = speechAndDecisionBridge();
  await assert.rejects(
    bridge.synthesizeSpeech({ text: TEXT }),
    { name: 'TypeError', message: /TextToSpeechOptions\.text contains U\+0000/ },
  );
  await assert.rejects(
    bridge.synthesizeSpeech({ text: 'ab', language: 'en\0x' }),
    { name: 'TypeError', message: /TextToSpeechOptions\.language contains U\+0000/ },
  );
  assert.deepEqual(coreCalls(core, 'llamadart_webgpu_tts_start'), []);

  await assert.rejects(bridge.synthesizeSpeech({ text: 'ab', language: 'en' }), /stub stops after start/);
  assert.deepEqual(coreCalls(core, 'llamadart_webgpu_tts_start')[0].slice(1, 3), ['ab', 'en']);
});

test('worker-mode synthesizeSpeech rejects U+0000 before posting to the worker', async () => {
  const posted = [];
  const bridge = createWorkerBridge({
    _restoreWorkerModelIfMissing: async () => {},
    _callWorker: async (method) => {
      posted.push(method);
      return { pcm: new Float32Array(1), sampleRate: 1, channels: 1, sampleCount: 1 };
    },
  });
  for (const options of [{ text: TEXT }, { text: 'ab', language: 'en\0x' }]) {
    await assert.rejects(bridge.synthesizeSpeech(options), { name: 'TypeError', message: /contains U\+0000/ });
  }
  assert.deepEqual(posted, []);
});

test('a decision head URL holding U+0000 keeps its whole label', async () => {
  const { bridge, core } = speechAndDecisionBridge();
  await bridge.loadDecisionHead('heads/a\0b.safetensors');
  await bridge.loadDecisionHead('https://example.test/heads/c\0d.safetensors');
  assert.deepEqual(
    coreCalls(core, 'llamadart_webgpu_decision_load').map((call) => call[2]),
    ['a%00b.safetensors', 'c%00d.safetensors'],
  );
});

test('the worker runtime keeps the whole decision head label', async () => {
  let core = null;
  await withWorkerCore(
    'loadDecisionHead',
    (runtime) => {
      core = withSpeechAndDecisionExports(createStubCore());
      runtime._core = core;
      runtime._fetchWithTimeout = async () => new Response(new Uint8Array([1]));
    },
    () => callWorkerHost('loadDecisionHead', ['heads/a\0b.safetensors', {}]),
  );
  assert.equal(coreCalls(core, 'llamadart_webgpu_decision_load')[0][2], 'a%00b.safetensors');
});
