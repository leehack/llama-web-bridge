import assert from 'node:assert/strict';

import {
  LlamaWebGpuBridge,
  enableBridgeWorkerHost,
} from '../js/src/llama_webgpu_bridge.js';

const encoder = new TextEncoder();
const decoder = new TextDecoder();
const posted = [];
const originalSelf = globalThis.self;
const originalCreateCompletion = LlamaWebGpuBridge.prototype.createCompletion;

function messagesFor(id) {
  return posted.filter(({ message }) => message.id === id).map(({ message }) => message);
}

async function waitForMessageCount(id, expected, timeoutMs = 500) {
  const deadline = Date.now() + timeoutMs;
  while (messagesFor(id).length < expected && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.equal(
    messagesFor(id).length,
    expected,
    `timed out waiting for ${expected} worker messages for request ${id}`,
  );
}

try {
  globalThis.self = {
    postMessage(message) {
      posted.push({ message });
    },
  };

  LlamaWebGpuBridge.prototype.createCompletion = async function createCompletion(
    prompt,
    options,
  ) {
    if (prompt === 'unicode-threshold') {
      options.onToken(encoder.encode('é'), 'é');
      options.onToken(encoder.encode('🙂'), 'é🙂');
      return 'é🙂';
    }
    if (prompt === 'completion-flush') {
      options.onToken(encoder.encode('ab'), 'ab');
      options.onToken(encoder.encode('cd'), 'abcd');
      return 'abcd';
    }
    if (prompt === 'unbatched') {
      options.onToken(encoder.encode('a'), 'a');
      options.onToken(encoder.encode('b'), 'ab');
      return 'ab';
    }
    if (prompt === 'text-threshold') {
      options.onToken('ab', 'ab');
      options.onToken('cd', 'abcd');
      return 'abcd';
    }
    if (prompt === 'error-cleanup') {
      options.onToken(encoder.encode('x'), 'x');
      throw new Error('expected generation failure');
    }
    if (prompt === 'split-unicode-threshold') {
      const smile = encoder.encode('🙂');
      options.onToken(smile.slice(0, 2), '');
      options.onToken(smile.slice(2), '🙂');
      assert.equal(
        messagesFor(6).length,
        0,
        'a split emoji must count as two JavaScript characters, not replacements',
      );
      options.onToken(encoder.encode('a'), '🙂a');
      return '🙂a';
    }
    if (prompt === 'byte-timer-flush') {
      options.onToken(encoder.encode('ab'), 'ab');
      await waitForMessageCount(7, 1);
      options.onToken(encoder.encode('cd'), 'abcd');
      await waitForMessageCount(7, 2);
      return 'abcd';
    }
    if (prompt === 'text-timer-flush') {
      options.onToken('ab', 'ab');
      await waitForMessageCount(8, 1);
      options.onToken('cd', 'abcd');
      await waitForMessageCount(8, 2);
      return 'abcd';
    }
    if (prompt === 'split-unicode-timer-boundary') {
      const smile = encoder.encode('🙂');
      options.onToken(smile.slice(0, 3), '');
      await waitForMessageCount(9, 1);
      options.onToken(smile.slice(3), '🙂');
      assert.equal(
        messagesFor(9).length,
        2,
        'decoder state must survive a timer flush and count the completed emoji',
      );
      return '🙂';
    }
    throw new Error(`Unexpected prompt: ${prompt}`);
  };

  enableBridgeWorkerHost();
  await globalThis.self.onmessage({ data: { type: 'init', config: {} } });
  assert.equal(posted.at(-1)?.message?.type, 'ready');

  await globalThis.self.onmessage({
    data: {
      type: 'call',
      id: 1,
      method: 'createCompletion',
      args: ['unicode-threshold', {
        tokenEventEncoding: 'bytes',
        tokenEventFlushMs: 100,
        tokenEventFlushChars: 3,
        emitCurrentTextOnToken: true,
      }],
    },
  });
  const unicodeMessages = messagesFor(1);
  assert.equal(unicodeMessages.length, 2);
  assert.equal(unicodeMessages[0].type, 'event');
  assert.equal(decoder.decode(Uint8Array.from(unicodeMessages[0].payload.piece)), 'é🙂');
  assert.equal(unicodeMessages[0].payload.currentText, 'é🙂');
  assert.equal(unicodeMessages[1].type, 'result');

  await globalThis.self.onmessage({
    data: {
      type: 'call',
      id: 2,
      method: 'createCompletion',
      args: ['completion-flush', {
        tokenEventEncoding: 'bytes',
        tokenEventFlushMs: 100,
        tokenEventFlushChars: 100,
      }],
    },
  });
  const completionMessages = messagesFor(2);
  assert.equal(completionMessages.length, 2);
  assert.equal(
    decoder.decode(Uint8Array.from(completionMessages[0].payload.piece)),
    'abcd',
  );
  assert.equal(completionMessages[0].payload.currentText, '');

  await globalThis.self.onmessage({
    data: {
      type: 'call',
      id: 3,
      method: 'createCompletion',
      args: ['unbatched', { tokenEventEncoding: 'bytes', tokenEventFlushMs: 0 }],
    },
  });
  const unbatchedMessages = messagesFor(3);
  assert.equal(unbatchedMessages.length, 3);
  assert.equal(decoder.decode(Uint8Array.from(unbatchedMessages[0].payload.piece)), 'a');
  assert.equal(decoder.decode(Uint8Array.from(unbatchedMessages[1].payload.piece)), 'b');

  await globalThis.self.onmessage({
    data: {
      type: 'call',
      id: 4,
      method: 'createCompletion',
      args: ['text-threshold', {
        tokenEventEncoding: 'text',
        tokenEventFlushMs: 100,
        tokenEventFlushChars: 4,
      }],
    },
  });
  const textMessages = messagesFor(4);
  assert.equal(textMessages.length, 2);
  assert.equal(textMessages[0].payload.pieceText, 'abcd');

  await globalThis.self.onmessage({
    data: {
      type: 'call',
      id: 5,
      method: 'createCompletion',
      args: ['error-cleanup', {
        tokenEventEncoding: 'bytes',
        tokenEventFlushMs: 20,
        tokenEventFlushChars: 100,
      }],
    },
  });
  const errorMessages = messagesFor(5);
  assert.equal(errorMessages.length, 2);
  assert.equal(errorMessages[0].type, 'event');
  assert.equal(decoder.decode(Uint8Array.from(errorMessages[0].payload.piece)), 'x');
  assert.equal(errorMessages[1].type, 'error');
  await new Promise((resolve) => setTimeout(resolve, 200));
  assert.equal(messagesFor(5).length, 2, 'cleared timers must not emit after an error');

  await globalThis.self.onmessage({
    data: {
      type: 'call',
      id: 6,
      method: 'createCompletion',
      args: ['split-unicode-threshold', {
        tokenEventEncoding: 'bytes',
        tokenEventFlushMs: 100,
        tokenEventFlushChars: 3,
        emitCurrentTextOnToken: true,
      }],
    },
  });
  const splitMessages = messagesFor(6);
  assert.equal(splitMessages.length, 2);
  assert.equal(
    decoder.decode(Uint8Array.from(splitMessages[0].payload.piece)),
    '🙂a',
  );
  assert.equal(splitMessages[0].payload.currentText, '🙂a');

  await globalThis.self.onmessage({
    data: {
      type: 'call',
      id: 7,
      method: 'createCompletion',
      args: ['byte-timer-flush', {
        tokenEventEncoding: 'bytes',
        tokenEventFlushMs: 10,
        tokenEventFlushChars: 100,
        emitCurrentTextOnToken: true,
      }],
    },
  });
  const byteTimerMessages = messagesFor(7);
  assert.deepEqual(byteTimerMessages.map(({ type }) => type), ['event', 'event', 'result']);
  assert.equal(
    decoder.decode(Uint8Array.from(byteTimerMessages[0].payload.piece)),
    'ab',
  );
  assert.equal(byteTimerMessages[0].payload.currentText, 'ab');
  assert.equal(
    decoder.decode(Uint8Array.from(byteTimerMessages[1].payload.piece)),
    'cd',
  );
  assert.equal(byteTimerMessages[1].payload.currentText, 'abcd');

  await globalThis.self.onmessage({
    data: {
      type: 'call',
      id: 8,
      method: 'createCompletion',
      args: ['text-timer-flush', {
        tokenEventEncoding: 'text',
        tokenEventFlushMs: 10,
        tokenEventFlushChars: 100,
        emitCurrentTextOnToken: true,
      }],
    },
  });
  const textTimerMessages = messagesFor(8);
  assert.deepEqual(textTimerMessages.map(({ type }) => type), ['event', 'event', 'result']);
  assert.deepEqual(
    textTimerMessages.slice(0, 2).map(({ payload }) => payload.pieceText),
    ['ab', 'cd'],
  );
  assert.deepEqual(
    textTimerMessages.slice(0, 2).map(({ payload }) => payload.currentText),
    ['ab', 'abcd'],
  );

  await globalThis.self.onmessage({
    data: {
      type: 'call',
      id: 9,
      method: 'createCompletion',
      args: ['split-unicode-timer-boundary', {
        tokenEventEncoding: 'bytes',
        tokenEventFlushMs: 10,
        tokenEventFlushChars: 2,
        emitCurrentTextOnToken: true,
      }],
    },
  });
  const boundaryMessages = messagesFor(9);
  assert.deepEqual(boundaryMessages.map(({ type }) => type), ['event', 'event', 'result']);
  const boundaryBytes = Uint8Array.from([
    ...boundaryMessages[0].payload.piece,
    ...boundaryMessages[1].payload.piece,
  ]);
  assert.equal(decoder.decode(boundaryBytes), '🙂');
  assert.equal(boundaryMessages[1].payload.currentText, '🙂');
} finally {
  globalThis.self = originalSelf;
  LlamaWebGpuBridge.prototype.createCompletion = originalCreateCompletion;
}

console.log('Worker token coalescing tests passed');
