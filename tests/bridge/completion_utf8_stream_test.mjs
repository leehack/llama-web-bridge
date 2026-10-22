import assert from 'node:assert/strict';

import {
  LlamaWebGpuBridge,
  enableBridgeWorkerHost,
} from '../../js/src/llama_webgpu_bridge.js';
import {
  createDirectBridge,
  createStubCore,
  emscriptenUtf8ToString,
} from './bridge_operation_queue_fixtures.mjs';
import { readRepoText } from './bridge_js_source.mjs';
import { readNativeCoreSource } from './native_core_source.mjs';

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// Byte-fallback tokens: every UTF-8 byte of a multi-byte character is its own piece.
function bytePieces(text) {
  return [...text].flatMap((character) => {
    const bytes = encoder.encode(character);
    return bytes.length === 1 ? [character] : [...bytes].map((byte) => Uint8Array.of(byte));
  });
}

function pieceText(piece) {
  return typeof piece === 'string' ? piece : decoder.decode(piece);
}

async function stream(pieces, options = {}, { bytesExport = true } = {}) {
  const { bridge, core } = createDirectBridge({ prompt: pieces });
  if (!bytesExport) {
    delete core._llamadart_webgpu_last_output_bytes_json;
  }
  const events = [];
  const text = await bridge.createCompletion('prompt', {
    emitCurrentTextOnToken: true,
    ...options,
    onToken: (piece, currentText) => {
      assert.ok(
        typeof piece === 'string' || piece instanceof Uint8Array,
        'a streamed piece is text or bytes',
      );
      if (piece instanceof Uint8Array) {
        assert.doesNotThrow(
          () => new TextDecoder('utf-8', { fatal: true }).decode(piece),
          `streamed bytes must be complete UTF-8: ${[...piece]}`,
        );
      }
      events.push({ piece: pieceText(piece), currentText });
    },
  });
  return { text, events, pieces: events.map(({ piece }) => piece) };
}

// Streamed pieces must add up to the returned text, each currentText must be
// the text streamed so far, and no piece may repeat earlier output.
function assertConsistent({ text, events }) {
  let streamed = '';
  for (const { piece, currentText } of events) {
    assert.ok(piece.length > 0, 'no empty pieces');
    streamed += piece;
    assert.equal(currentText, streamed);
  }
  assert.equal(streamed, text);
}

const ISSUE_TEXT = '🦙 𠜎𠜱 龘靐 你好';

const CASES = [
  ['the stub core reproduces the Emscripten decode of a split character', () => {
    const llama = [...encoder.encode('🦙')];
    assert.equal(emscriptenUtf8ToString(llama.slice(0, 1)), '\u0000');
    assert.equal(emscriptenUtf8ToString(llama.slice(0, 2)), '🀀');
    assert.equal(emscriptenUtf8ToString(llama.slice(0, 3)), '🦀');
    assert.equal(emscriptenUtf8ToString(llama), '🦙');
    assert.equal(emscriptenUtf8ToString([0xC3, 0x41]), 'Á');
  }],

  ['the core exports generated and detokenized text as bytes', () => {
    const exported = readRepoText('CMakeLists.txt').match(/-sEXPORTED_FUNCTIONS=\[([^\]]*)\]/)[1];
    assert.ok(exported.includes("'_llamadart_webgpu_last_output_bytes_json'"));
    assert.ok(exported.includes("'_llamadart_webgpu_last_detokenized_bytes_json'"));
    const source = readNativeCoreSource();
    const serializer = source.slice(source.indexOf('std::string serialize_bytes_json('));
    assert.match(serializer, /static_cast<unsigned char>\(bytes\[i\]\)/);
    assert.match(source, /llamadart_webgpu_last_output_bytes_json\(\s*int32_t offset\) \{\s*g_last_output_bytes_json = serialize_bytes_json\(g_last_output, offset\);/);
    assert.match(source, /llamadart_webgpu_last_detokenized_bytes_json\(\) \{\s*g_last_detokenized_bytes_json = serialize_bytes_json\(g_last_detokenized, 0\);/);
  }],

  ['byte-split emoji, rare Han and CJK at the start of a response stream intact', async () => {
    for (const tokenEventEncoding of ['bytes', 'text']) {
      const result = await stream(bytePieces(ISSUE_TEXT), { tokenEventEncoding });
      assertConsistent(result);
      assert.equal(result.text, ISSUE_TEXT);
      assert.deepEqual(result.pieces, ['🦙', ' ', '𠜎', '𠜱', ' ', '龘', '靐', ' ', '你', '好']);
    }
  }],

  ['a character split across two multi-byte pieces streams once', async () => {
    const llama = encoder.encode('🦙');
    const han = encoder.encode('龘');
    const result = await stream([llama.slice(0, 2), llama.slice(2), han.slice(0, 1), han.slice(1), 'x']);
    assertConsistent(result);
    assert.deepEqual(result.pieces, ['🦙', '龘', 'x']);
  }],

  ['a byte-split character after more than 16 bytes streams intact', async () => {
    const prefix = 'The answer is here: ';
    const result = await stream([prefix, ...bytePieces('🦙你'), ' done']);
    assertConsistent(result);
    assert.deepEqual(result.pieces, [prefix, '🦙', '你', ' done']);
  }],

  ['invalid bytes surface as U+FFFD as soon as they cannot complete', async () => {
    const result = await stream([
      Uint8Array.of(0x80),
      Uint8Array.of(0xC3),
      'A',
      Uint8Array.of(0xF0, 0x9F),
      Uint8Array.of(0xE4),
      'B',
    ]);
    assertConsistent(result);
    assert.equal(result.text, '��A��B');
    assert.deepEqual(result.pieces, ['�', '�A', '�', '�B']);
  }],

  ['an incomplete character at the end of generation is flushed as U+FFFD', async () => {
    const llama = encoder.encode('🦙');
    for (const pieces of [[llama.slice(0, 3)], ['ok ', llama.slice(0, 2)]]) {
      const result = await stream(pieces);
      assertConsistent(result);
      assert.equal(result.text, pieces.length === 1 ? '�' : 'ok �');
    }
  }],

  ['nPredict ending inside a character returns U+FFFD, not a wrong code point', async () => {
    const result = await stream(bytePieces('🦙!'), { nPredict: 3 });
    assertConsistent(result);
    assert.equal(result.text, '�');
  }],

  ['a piece appended by a step that hits the context limit streams at the end', async () => {
    const llama = encoder.encode('🦙');
    const contextLimit = Object.assign(
      new Error('failed to find a memory slot for batch'),
      { appended: llama.slice(2) },
    );
    const result = await stream(['a', llama.slice(0, 2), contextLimit]);
    assertConsistent(result);
    assert.deepEqual(result.pieces, ['a', '🦙']);
  }],

  ['a core without the bytes export keeps the string read', async () => {
    const result = await stream(['The answer is here: ', ...bytePieces('🦙'), ' done'], {}, {
      bytesExport: false,
    });
    assertConsistent(result);
    assert.equal(result.text, 'The answer is here: 🦙 done');
    assert.deepEqual(result.pieces, ['The answer is here: ', '🦙', ' done']);
  }],

  ['a completion without onToken returns the decoded text', async () => {
    const { bridge } = createDirectBridge({ prompt: bytePieces(ISSUE_TEXT) });
    assert.equal(await bridge.createCompletion('prompt'), ISSUE_TEXT);
  }],
];

for (const [name, run] of CASES) {
  try {
    await run();
  } catch (error) {
    console.error(`FAIL: ${name}`);
    throw error;
  }
  console.log(`ok - ${name}`);
}

// Worker path: the real worker host drives the real runtime over the stub core.
const posted = [];
const originalSelf = globalThis.self;
const originalCreateCompletion = LlamaWebGpuBridge.prototype.createCompletion;
try {
  globalThis.self = { postMessage: (message) => posted.push(message) };
  LlamaWebGpuBridge.prototype.createCompletion = function createCompletion(prompt, options) {
    if (!this._runtime._core) {
      this._runtime._core = createStubCore({ prompt: bytePieces(ISSUE_TEXT) });
      this._runtime._modelBytes = 1;
      this._runtime._nCtx = 128;
    }
    return originalCreateCompletion.call(this, prompt, options);
  };
  enableBridgeWorkerHost();
  await globalThis.self.onmessage({ data: { type: 'init', config: {} } });

  const workerCases = [
    { tokenEventEncoding: 'bytes' },
    { tokenEventEncoding: 'text' },
    { tokenEventEncoding: 'bytes', tokenEventFlushMs: 50, tokenEventFlushChars: 2 },
    { tokenEventEncoding: 'text', tokenEventFlushMs: 50, tokenEventFlushChars: 2 },
  ];
  for (const [index, options] of workerCases.entries()) {
    const id = index + 1;
    await globalThis.self.onmessage({
      data: { type: 'call', id, method: 'createCompletion', args: ['prompt', options] },
    });
    const messages = posted.filter((message) => message.id === id);
    const result = messages.at(-1);
    assert.equal(result.type, 'result', JSON.stringify(result));
    assert.equal(result.value, ISSUE_TEXT);
    const pieces = messages
      .filter((message) => message.type === 'event' && message.event === 'token')
      .map(({ payload }) => (
        typeof payload.pieceText === 'string'
          ? payload.pieceText
          : new TextDecoder('utf-8', { fatal: true }).decode(Uint8Array.from(payload.piece))
      ));
    assert.ok(pieces.every((piece) => !piece.includes('�')), JSON.stringify(pieces));
    assert.equal(pieces.join(''), ISSUE_TEXT, JSON.stringify({ options, pieces }));
    console.log(`ok - worker stream intact with ${JSON.stringify(options)}`);
  }
} finally {
  LlamaWebGpuBridge.prototype.createCompletion = originalCreateCompletion;
  globalThis.self = originalSelf;
}
