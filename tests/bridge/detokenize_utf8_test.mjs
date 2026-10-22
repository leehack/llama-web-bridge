import assert from 'node:assert/strict';

import {
  LlamaWebGpuBridge,
  enableBridgeWorkerHost,
} from '../../js/src/llama_webgpu_bridge.js';
import {
  byteToken,
  createDirectBridge,
  createStubCore,
} from './bridge_operation_queue_fixtures.mjs';

const encoder = new TextEncoder();

function byteTokens(text, length = Infinity) {
  return [...encoder.encode(text)].slice(0, length).map(byteToken);
}

function codePointTokens(text) {
  return [...text].map((character) => character.codePointAt(0));
}

// [tokens, detokenized text] as TextDecoder decodes the bytes.
const CASES = [
  ['byte-split emoji', [...byteTokens('🦙'), ...codePointTokens('a')], '🦙a'],
  ['leading U+FEFF from byte tokens', [...byteTokens('\uFEFF'), ...codePointTokens('a')], '\uFEFFa'],
  ['leading U+FEFF from one token', codePointTokens('\uFEFFa'), '\uFEFFa'],
  ['emoji missing its last bytes', byteTokens('🦙', 2), '�'],
  ['CJK missing its last byte', byteTokens('你', 2), '�'],
  ['4-byte Han missing its last byte', byteTokens('𠜎', 3), '�'],
  ['complete 4-byte Han and CJK', [...byteTokens('𠜎'), ...byteTokens('你')], '𠜎你'],
  ['lead byte before ASCII', [byteToken(0xC3), ...codePointTokens('A')], '�A'],
  ['lone continuation byte', [byteToken(0x80)], '�'],
  [
    'incomplete character after more than 16 bytes',
    [...codePointTokens('abcdefghijklmnopq'), ...byteTokens('🦙', 2)],
    'abcdefghijklmnopq�',
  ],
];

for (const [name, tokens, expected] of CASES) {
  const { bridge } = createDirectBridge();
  assert.equal(await bridge.detokenize(tokens), expected, name);
  console.log(`ok - direct: ${name}`);
}

{
  const { bridge, core } = createDirectBridge();
  delete core._llamadart_webgpu_last_detokenized_bytes_json;
  assert.equal(await bridge.detokenize([...byteTokens('🦙'), ...codePointTokens('a')]), '🦙a');
  assert.equal(
    await bridge.detokenize(byteTokens('🦙', 2)),
    '🀀',
    'a core without the bytes export keeps the string read and its decode',
  );
  console.log('ok - a core without the bytes export keeps the string read');
}

// Worker path: the real worker host drives the real runtime over the stub core.
const posted = [];
const originalSelf = globalThis.self;
const originalDetokenize = LlamaWebGpuBridge.prototype.detokenize;
try {
  globalThis.self = { postMessage: (message) => posted.push(message) };
  LlamaWebGpuBridge.prototype.detokenize = function detokenize(tokens, special) {
    if (!this._runtime._core) {
      this._runtime._core = createStubCore();
      this._runtime._modelBytes = 1;
    }
    return originalDetokenize.call(this, tokens, special);
  };
  enableBridgeWorkerHost();
  await globalThis.self.onmessage({ data: { type: 'init', config: {} } });

  for (const [index, [name, tokens, expected]] of CASES.entries()) {
    const id = index + 1;
    await globalThis.self.onmessage({
      data: { type: 'call', id, method: 'detokenize', args: [tokens, false] },
    });
    const result = posted.find((message) => message.id === id);
    assert.equal(result?.type, 'result', JSON.stringify(result));
    assert.equal(result.value, expected, name);
    console.log(`ok - worker: ${name}`);
  }
} finally {
  LlamaWebGpuBridge.prototype.detokenize = originalDetokenize;
  globalThis.self = originalSelf;
}
