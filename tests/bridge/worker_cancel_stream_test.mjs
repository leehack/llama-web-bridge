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
// next yield lets it read the cancel. A cancel applies only while the operation
// it names has a call running in the worker (#180).

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
// Fewer pieces than the runtime's yield interval: the worker posts them and the
// result in one task, so a cancel sent from the first piece lands after the result.
const SHORT_PIECES = ['ab', 'cd'];
const SCRIPTS = {
  prompt: bytePieces(FULL_TEXT),
  'split-tail': SPLIT_TAIL_PIECES,
  short: SHORT_PIECES,
};

const originalSelf = globalThis.self;
const originalCreateCompletion = LlamaWebGpuBridge.prototype.createCompletion;
const macrotask = (run) => setTimeout(run, 0);
const nextMacrotask = () => new Promise((resolve) => macrotask(resolve));
let pageWorker = null;
// The stub core under the worker's runtime; its cancelRequests count the
// requests that set the core's cancel flag.
let workerCore = null;

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

function assertWholeCompletion(result, text = FULL_TEXT) {
  assertStreamMatchesText(result);
  assert.equal(result.text, text);
  assert.equal(result.usage.finishReason, 'stop');
}

// A raw protocol cancel, as the facade sends one, naming `operationId`.
function postWorkerCancel(operationId) {
  macrotask(() => globalThis.self.onmessage({
    data: { type: 'call', id: 1_000_000, method: 'cancel', args: [], operationId },
  }));
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

  ['a cancel that reaches the worker after its result leaves the next completion whole', async (bridge) => {
    const first = await complete(bridge, 'short', {
      onPiece: (count) => {
        if (count === 1) {
          bridge.cancel();
        }
      },
    });
    assertWholeCompletion(first, SHORT_PIECES.join(''));
    await nextMacrotask();
    assert.equal(workerCore.cancelRequests, 0, 'the late cancel must not set the core cancel flag');
    assertWholeCompletion(await complete(bridge, 'prompt'));
    assert.equal(workerCore.cancelRequests, 0);
  }],

  ['an idle cancel() leaves the next completion whole', async (bridge) => {
    assertWholeCompletion(await complete(bridge, 'prompt'));
    bridge.cancel();
    // The worker reads the cancel before the next call, which it reaches first.
    assertWholeCompletion(await complete(bridge, 'prompt'));
    assert.equal(workerCore.cancelRequests, 0, 'an idle cancel must not set the core cancel flag');
  }],

  ['cancel() right after createCompletion() cancels it before the worker starts', async (bridge) => {
    const pending = complete(bridge, 'prompt');
    const operationId = bridge._activeOperation.id;
    bridge.cancel();
    const cancelled = await pending;
    assertStreamMatchesText(cancelled);
    assert.equal(cancelled.usage.finishReason, 'cancelled');
    assert.ok(cancelled.text.length < FULL_TEXT.length, 'the cancel stopped generation early');
    assert.deepEqual(
      pageWorker.posted
        .filter((message) => message.type === 'call')
        .map(({ method, operationId: id }) => [method, id]),
      [['cancel', operationId], ['createCompletion', operationId], ['cancel', operationId]],
      'the worker drops the first cancel, so a second follows the call',
    );
    assertWholeCompletion(await complete(bridge, 'prompt'));
  }],

  ['a cancel still applies after another call of the same operation returns', async (bridge) => {
    const result = await complete(bridge, 'prompt', {
      onPiece: (count) => {
        if (count === 1) {
          bridge.setLogLevel(2);
        }
        if (count === 12) {
          bridge.cancel();
        }
      },
    });
    const setLogLevel = pageWorker.posted.find((message) => message.method === 'setLogLevel');
    const completion = pageWorker.posted.find((message) => message.method === 'createCompletion');
    assert.equal(setLogLevel.operationId, completion.operationId);
    assertStreamMatchesText(result);
    assert.equal(result.usage.finishReason, 'cancelled');
    assert.ok(result.text.length < FULL_TEXT.length);
  }],

  ['a cancel naming another operation leaves the running one alone', async (bridge) => {
    let running = null;
    const result = await complete(bridge, 'prompt', {
      onPiece: (count) => {
        if (count === 1) {
          running = bridge._activeOperation.id;
          postWorkerCancel(running + 1);
        }
      },
    });
    assertWholeCompletion(result);
    assert.equal(workerCore.cancelRequests, 0);

    const named = await complete(bridge, 'prompt', {
      onPiece: (count) => {
        if (count === 1) {
          postWorkerCancel(bridge._activeOperation.id);
        }
      },
    });
    assertStreamMatchesText(named);
    assert.equal(named.usage.finishReason, 'cancelled', 'a cancel naming the running operation applies');
    assert.ok(named.text.length < FULL_TEXT.length);

    const untagged = await complete(bridge, 'prompt', {
      onPiece: (count) => {
        if (count === 1) {
          postWorkerCancel(undefined);
        }
      },
    });
    assert.equal(untagged.usage.finishReason, 'cancelled', 'a cancel naming no operation stops the running call');
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
      workerCore = this._runtime._core;
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


// Direct: the runtime clears its cancel flag when a completion starts, so an
// idle cancel cannot cut the next completion short, and a cancel issued once the
// call owns the runtime still stops it.
{
  const { bridge } = createDirectBridge(SCRIPTS);
  assertWholeCompletion(await complete(bridge, 'prompt'));
  bridge.cancel();
  assertWholeCompletion(await complete(bridge, 'prompt'));

  const pending = complete(bridge, 'prompt');
  bridge.cancel();
  const cancelled = await pending;
  assertStreamMatchesText(cancelled);
  assert.equal(cancelled.usage.finishReason, 'cancelled');
  assert.equal(cancelled.text, '');
  assertWholeCompletion(await complete(bridge, 'prompt'));
  await bridge.dispose();
  console.log('ok - direct idle cancel() and cancel() at the start of a completion stay scoped');
}
