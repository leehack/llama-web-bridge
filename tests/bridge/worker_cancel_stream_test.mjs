import assert from 'node:assert/strict';

import {
  LlamaWebGpuBridge,
  enableBridgeWorkerHost,
} from '../../js/src/llama_webgpu_bridge.js';
import {
  createDirectBridge,
  createRealWorkerBridge,
  createStubCore,
  withStubWorkerEnvironment,
} from './bridge_operation_queue_fixtures.mjs';

// Pieces streamed after a bare cancel() must still add up to the text the
// completion returns (#178). The page facade, a real BridgeWorkerProxy, the
// real worker host and the real runtime over the stub core exchange messages
// as separate macrotasks, so the worker keeps generating until the runtime's
// next yield lets it read the cancel.

const encoder = new TextEncoder();
const strictDecoder = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });

// Byte-fallback tokens: every UTF-8 byte of a multi-byte character is its own piece.
function bytePieces(text) {
  return [...text].flatMap((character) => {
    const bytes = encoder.encode(character);
    return bytes.length === 1 ? [character] : [...bytes].map((byte) => Uint8Array.of(byte));
  });
}

const FULL_TEXT = '🦙 𠜎𠜱 龘靐 你好 é '.repeat(4);
// Generation ends two bytes into the emoji before the worker reads the cancel,
// so the text ends in U+FFFD.
const SPLIT_TAIL_PIECES = ['abcd', 'efgh', 'ijkl', ...bytePieces('🦙').slice(0, 2)];
const SCRIPTS = {
  prompt: bytePieces(FULL_TEXT),
  'split-tail': SPLIT_TAIL_PIECES,
};

const originalSelf = globalThis.self;
const originalCreateCompletion = LlamaWebGpuBridge.prototype.createCompletion;
const macrotask = (run) => setTimeout(run, 0);
let pageWorker = null;

function pieceText(piece) {
  if (typeof piece === 'string') {
    return piece;
  }
  assert.ok(piece instanceof Uint8Array, 'a streamed piece is text or bytes');
  return strictDecoder.decode(piece);
}

// A page bridge whose worker messages reach the in-process worker host.
async function connectedWorkerBridge(workers) {
  const bridge = createRealWorkerBridge();
  const worker = workers.at(-1);
  pageWorker = worker;
  const init = worker.posted.find((message) => message.type === 'init');
  worker.postMessage = (message) => {
    worker.posted.push(message);
    const copy = structuredClone(message);
    macrotask(() => globalThis.self.onmessage({ data: copy }));
  };
  await globalThis.self.onmessage({ data: structuredClone(init) });
  return { bridge, worker };
}

async function complete(bridge, prompt, { onPiece = () => {}, ...options } = {}) {
  const events = [];
  let usage = null;
  let text = null;
  let error = null;
  let settled = false;
  try {
    text = await bridge.createCompletion(prompt, {
      emitCurrentTextOnToken: true,
      ...options,
      onToken: (piece, currentText) => {
        assert.equal(settled, false, 'no token event after the completion settles');
        events.push({ piece: pieceText(piece), currentText });
        onPiece(events.length);
      },
      onUsage: (value) => {
        usage = value;
      },
    });
  } catch (caught) {
    error = caught;
  }
  settled = true;
  return { text, error, usage, events, pieces: events.map(({ piece }) => piece) };
}

function assertStreamMatchesText({ text, error, events }) {
  assert.equal(error, null, String(error?.stack || error));
  let streamed = '';
  for (const { piece, currentText } of events) {
    assert.ok(piece.length > 0, 'no empty pieces');
    streamed += piece;
    if (currentText !== '') {
      assert.equal(currentText, streamed);
    }
  }
  assert.equal(streamed, text, JSON.stringify({ pieces: events.map(({ piece }) => piece), text }));
}

const STREAM_OPTIONS = [
  { tokenEventEncoding: 'bytes' },
  { tokenEventEncoding: 'text' },
  { tokenEventEncoding: 'bytes', tokenEventFlushMs: 1, tokenEventFlushChars: 2 },
  { tokenEventEncoding: 'text', tokenEventFlushMs: 1, tokenEventFlushChars: 2 },
];

const WORKER_CASES = [
  ['an uncancelled worker completion streams its whole text', async (bridge) => {
    for (const options of STREAM_OPTIONS) {
      const result = await complete(bridge, 'prompt', options);
      assertStreamMatchesText(result);
      assert.equal(result.text, FULL_TEXT);
      assert.equal(result.usage.finishReason, 'stop');
    }
  }],

  ['cancel() in the first onToken streams every piece the result contains', async (bridge) => {
    for (const options of STREAM_OPTIONS) {
      const result = await complete(bridge, 'prompt', {
        ...options,
        onPiece: (count) => {
          if (count === 1) {
            bridge.cancel();
          }
        },
      });
      assertStreamMatchesText(result);
      assert.equal(result.usage.finishReason, 'cancelled', JSON.stringify(options));
      assert.ok(FULL_TEXT.startsWith(result.text), 'the cancel stopped generation early');
      assert.ok(result.text.length < FULL_TEXT.length, 'the cancel stopped generation early');
      assert.ok(
        result.text.length > result.pieces[0].length,
        'the worker generated past the piece that cancelled',
      );
    }
  }],

  ['repeated cancel() calls keep the stream and result equal', async (bridge) => {
    const result = await complete(bridge, 'prompt', {
      onPiece: () => {
        bridge.cancel();
        bridge.cancel();
      },
    });
    assertStreamMatchesText(result);
    assert.equal(result.usage.finishReason, 'cancelled');
    assert.ok(result.events.length > 1, 'pieces generated before the cancel landed stream');
  }],

  ['a generation ending mid-character during a cancel streams its U+FFFD tail', async (bridge) => {
    for (const options of STREAM_OPTIONS) {
      const result = await complete(bridge, 'split-tail', {
        ...options,
        onPiece: (count) => {
          if (count === 1) {
            bridge.cancel();
          }
        },
      });
      assertStreamMatchesText(result);
      assert.equal(result.text, 'abcdefghijkl\uFFFD');
      assert.equal(result.usage.finishReason, 'stop');
    }
  }],

  ['an aborted signal rejects and stops streaming at once', async (bridge) => {
    const controller = new AbortController();
    const result = await complete(bridge, 'prompt', {
      signal: controller.signal,
      onPiece: (count) => {
        if (count === 1) {
          controller.abort();
        }
      },
    });
    assert.equal(result.error?.name, 'AbortError');
    assert.equal(result.events.length, 1, 'no piece reaches the caller after its abort');
    assert.equal(result.usage.finishReason, 'cancelled');
  }],

  ['an abort after cancel() rejects and stops streaming at once', async (bridge) => {
    const controller = new AbortController();
    const result = await complete(bridge, 'prompt', {
      signal: controller.signal,
      onPiece: (count) => {
        if (count === 1) {
          bridge.cancel();
        }
        if (count === 2) {
          controller.abort();
        }
      },
    });
    assert.equal(result.error?.name, 'AbortError');
    assert.equal(result.events.length, 2, 'no piece reaches the caller after its abort');
  }],

  ['a completion after a cancelled one streams its whole text', async (bridge) => {
    const cancelled = await complete(bridge, 'prompt', {
      onPiece: (count) => {
        if (count === 1) {
          bridge.cancel();
        }
      },
    });
    assertStreamMatchesText(cancelled);
    const next = await complete(bridge, 'prompt');
    assertStreamMatchesText(next);
    assert.equal(next.text, FULL_TEXT);
    assert.equal(next.usage.finishReason, 'stop');
  }],

  ['dispose() racing a cancelled completion settles it with a matching stream', async (bridge) => {
    let disposal = null;
    const result = await complete(bridge, 'prompt', {
      onPiece: (count) => {
        if (count === 1) {
          bridge.cancel();
          disposal = bridge.dispose();
        }
      },
    });
    await disposal;
    assertStreamMatchesText(result);
    assert.equal(result.usage.finishReason, 'cancelled');
    await assert.rejects(bridge.createCompletion('prompt'), /Bridge has been disposed/);
  }],
];

try {
  globalThis.self = {
    postMessage(message) {
      const copy = structuredClone(message);
      const target = pageWorker;
      macrotask(() => target.onmessage({ data: copy }));
    },
  };
  LlamaWebGpuBridge.prototype.createCompletion = function createCompletion(prompt, options) {
    // The worker host's facade runs the runtime directly; the page's does not.
    if (this._config?.disableWorker === true && !this._runtime._core) {
      this._runtime._core = createStubCore(SCRIPTS);
      this._runtime._modelBytes = 1;
      this._runtime._nCtx = 128;
    }
    return originalCreateCompletion.call(this, prompt, options);
  };
  enableBridgeWorkerHost();

  await withStubWorkerEnvironment(async ({ workers }) => {
    for (const [name, run] of WORKER_CASES) {
      const { bridge } = await connectedWorkerBridge(workers);
      try {
        await run(bridge);
      } catch (error) {
        console.error(`FAIL: worker ${name}`);
        throw error;
      } finally {
        await bridge.dispose();
      }
      console.log(`ok - worker ${name}`);
    }
  });
} finally {
  LlamaWebGpuBridge.prototype.createCompletion = originalCreateCompletion;
  globalThis.self = originalSelf;
}

// Direct control: the runtime streams to the caller without a transport.
for (const options of STREAM_OPTIONS.slice(0, 2)) {
  const { bridge } = createDirectBridge(SCRIPTS);
  const result = await complete(bridge, 'prompt', {
    ...options,
    onPiece: (count) => {
      if (count === 1) {
        bridge.cancel();
      }
    },
  });
  assertStreamMatchesText(result);
  assert.equal(result.usage.finishReason, 'cancelled');
  assert.equal(result.text, result.pieces[0]);
  await bridge.dispose();
  console.log(`ok - direct cancel() in onToken streams the returned text (${options.tokenEventEncoding})`);
}

