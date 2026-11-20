import assert from 'node:assert/strict';
import { test } from 'node:test';

import * as api from '../../js/src/llama_webgpu_bridge.js';
import {
  createDirectBridge, createRealWorkerBridge, withStubWorkerEnvironment, workerDriver,
} from './bridge_operation_queue_fixtures.mjs';

const { LlamaWebGpuBridge } = api;

const first = [0.25, 0.5];
const second = [-0.5];
const tail = [0.125, -0.25];
const full = [...first, ...second, ...tail];
const steps = [
  { state: 1 },
  { state: 2 },
  { state: 2, pcm: first },
  { state: 2, pcm: second },
  { state: 3, pcm: tail },
];

function createSpeechBridge({ capabilities = 11, supported = true, script = steps } = {}) {
  const { bridge, core } = createDirectBridge();
  const fs = core.FS;
  const files = new Set();
  const calls = [];
  const originalCcall = core.ccall;
  core.FS = {
    ...fs,
    writeFile(path, bytes) {
      files.add(path);
      fs.writeFile(path, bytes);
    },
    unlink(path) {
      files.delete(path);
      fs.unlink(path);
    },
  };
  let index = -1;
  function writePcm(path, samples) {
    const floats = new Float32Array(new SharedArrayBuffer((samples.length + 2) * 4), 4, samples.length);
    floats.set(samples);
    core.FS.writeFile(path, new Uint8Array(floats.buffer, floats.byteOffset, floats.byteLength));
  }
  core.ccall = (name, returnType, types, args, options) => {
    if (!name.startsWith('llamadart_webgpu_tts_')) {
      return originalCcall(name, returnType, types, args, options);
    }
    const method = name.slice('llamadart_webgpu_tts_'.length);
    calls.push([method, types, args]);
    switch (method) {
      case 'info_json':
        return JSON.stringify({ apiVersion: 1, supported, capabilities, sampleRate: 24000, channels: 1 });
      case 'start':
        index = -1;
        return 0;
      case 'step':
        index += 1;
        assert.ok(index < script.length);
        return script[index].status ?? 0;
      case 'progress_json': {
        const { state } = script[index];
        return JSON.stringify({ state, framesGenerated: index, truncated: false });
      }
      case 'write_pcm_chunk': {
        const pcm = script[index].pcm;
        if (!pcm) return 0;
        writePcm(args[0], pcm);
        return pcm.length;
      }
      case 'write_pcm':
        core.FS.writeFile(args[0], new Uint8Array(new Float32Array(full).buffer));
        return 0;
      case 'reset':
        return 0;
      default:
        assert.fail(method);
    }
  };
  return { bridge, core, calls, files };
}

test('incremental audio follows the streaming bit and model support', async () => {
  for (const [capabilities, supported, expected] of [[11, true, true], [3, true, false], [11, false, false]]) {
    const { bridge } = createSpeechBridge({ capabilities, supported });
    assert.equal((await bridge.getTextToSpeechCapabilities()).supportsIncrementalAudio, expected);
  }
  const uninitialized = new LlamaWebGpuBridge({ disableWorker: true });
  assert.equal((await uninitialized.getTextToSpeechCapabilities()).supportsIncrementalAudio, false);
});

test('streaming requires a callback and a compatible core before starting', async () => {
  const { bridge, calls } = createSpeechBridge({ capabilities: 3 });
  await assert.rejects(bridge.synthesizeSpeech({ text: 'Hello.', streamAudio: true }), TypeError);
  await assert.rejects(
    bridge.synthesizeSpeech({ text: 'Hello.', streamAudio: true, onProgress() {} }),
    /cannot stream audio/,
  );
  assert.equal(calls.some(([method]) => method === 'start'), false);
});

test('progress delivers standalone ordered PCM including the tail; final PCM is empty', async () => {
  const { bridge, calls, files } = createSpeechBridge();
  const progress = [];
  const result = await bridge.synthesizeSpeech({
    text: 'Hello.',
    streamAudio: true,
    onProgress(event) {
      assert.equal(files.size, 0);
      progress.push(event);
    },
  });
  const [, types, args] = calls.find(([method]) => method === 'start');
  assert.equal(types.length, 11);
  assert.equal(args.length, 11);
  assert.equal(args.at(-1), 1);
  assert.deepEqual(progress.map((event) => event.state), [1, 2, 2, 2, 3]);
  assert.ok(progress.slice(0, 2).every((event) => !('pcm' in event)));
  const chunks = progress.slice(2);
  assert.deepEqual(chunks.map((event) => event.sampleOffset), [0, 2, 3]);
  for (const event of chunks) {
    assert.ok(event.pcm instanceof Float32Array);
    assert.ok(event.pcm.buffer instanceof ArrayBuffer);
    assert.equal(event.sampleRate, 24000);
    assert.equal(event.channels, 1);
  }
  assert.deepEqual(chunks.flatMap(({ pcm }) => [...pcm]), full);
  assert.deepEqual(result.pcm, new Float32Array(0));
  assert.equal(result.sampleCount, full.length);
  assert.equal(calls.some(([method]) => method === 'write_pcm'), false);
  assert.equal(calls.at(-1)[0], 'reset');
  assert.equal(files.size, 0);
});

test('a cancelled step does not drain any remaining streamed PCM', async () => {
  const { bridge, calls, files } = createSpeechBridge({
    script: [{ state: 2, pcm: first }, { state: 4, status: -6, pcm: tail }],
  });
  const chunks = [];
  await assert.rejects(
    bridge.synthesizeSpeech({
      text: 'Hello.',
      streamAudio: true,
      onProgress(event) { if (event.pcm) chunks.push([...event.pcm]); },
    }),
    { name: 'AbortError' },
  );
  assert.deepEqual(chunks, [first]);
  assert.equal(calls.filter(([method]) => method === 'write_pcm_chunk').length, 1);
  assert.equal(calls.at(-1)[0], 'reset');
  assert.equal(files.size, 0);
});

test('default synthesis still reads full PCM and never reads a chunk', async () => {
  const { bridge, calls } = createSpeechBridge({ capabilities: 3 });
  const progress = [];
  const result = await bridge.synthesizeSpeech({ text: 'Hello.', onProgress: (event) => progress.push(event) });
  const [, types, args] = calls.find(([method]) => method === 'start');
  assert.equal(types.length, 11);
  assert.equal(args.length, 11);
  assert.equal(args.at(-1), 0);
  assert.equal(calls.some(([method]) => method === 'write_pcm_chunk'), false);
  assert.ok(progress.every((event) => !('pcm' in event)));
  assert.deepEqual(result.pcm, new Float32Array(full));
  assert.equal(result.sampleCount, full.length);
});

async function runWorkerSpeech(options, {
  legacy = false, onBridge, emitSecond = true, afterProgress, beforeResult, afterResult,
} = {}) {
  return withStubWorkerEnvironment(async ({ workers }) => {
    const bridge = createRealWorkerBridge();
    const proxy = bridge._workerProxy;
    const worker = workers[0];
    const driver = workerDriver(proxy, worker);
    const eventErrors = [];
    worker.onmessage({ data: { type: 'ready', ...(legacy ? {} : { supportsIncrementalAudio: true }) } });
    onBridge?.(bridge);
    const pending = bridge.synthesizeSpeech(options).then(
      (value) => ({ value }),
      (error) => ({ error }),
    );
    await driver.settle();
    const calls = driver.calls('synthesizeSpeech');
    for (const { id } of calls) {
      if (!legacy) {
        for (const [state, sampleOffset, pcm] of [[2, 0, first], [3, first.length, tail]]) {
          if (state === 3 && !emitSecond) break;
          try {
            worker.onmessage({ data: {
              type: 'event', id, event: 'progress', payload: { state, sampleOffset, pcm: new Float32Array(pcm) },
            } });
          } catch (error) {
            eventErrors.push(error);
          }
          afterProgress?.(state);
        }
      }
      const result = await beforeResult?.() ?? {
        pcm: new Float32Array(legacy ? [...first, ...tail] : 0),
        sampleCount: first.length + tail.length,
      };
      driver.reply(id, result);
      afterResult?.();
    }
    await driver.settle();
    const cancelCalls = driver.calls('cancel');
    for (const { id } of cancelCalls) driver.reply(id, true);
    const outcome = await pending;
    const dispose = bridge.dispose();
    await driver.settle();
    for (const { id } of driver.calls('dispose')) driver.reply(id, true);
    await dispose;
    return { ...outcome, calls, cancelCalls, eventErrors };
  });
}

const cancellationScript = [{ state: 2, pcm: first }, { state: 3, pcm: tail }];

test('a queued one-shot request keeps its admitted mode in direct synthesis', async () => {
  const { bridge, calls } = createSpeechBridge({ script: cancellationScript });
  const active = bridge.synthesizeSpeech({ text: 'First.' });
  const options = { text: 'Queued.' };
  const queued = bridge.synthesizeSpeech(options);
  options.streamAudio = true;
  await active;
  const result = await queued;
  assert.deepEqual(calls.filter(([method]) => method === 'start').map(([, , args]) => args.at(-1)), [0, 0]);
  assert.deepEqual(result.pcm, new Float32Array(full));
  assert.equal(result.sampleCount, full.length);
});

test('a queued one-shot request keeps its admitted mode on the worker', async () => {
  await withStubWorkerEnvironment(async ({ workers }) => {
    const bridge = createRealWorkerBridge();
    const worker = workers[0];
    const driver = workerDriver(bridge._workerProxy, worker);
    worker.onmessage({ data: { type: 'ready', supportsIncrementalAudio: true } });
    const active = bridge.synthesizeSpeech({ text: 'First.' });
    const options = { text: 'Queued.' };
    const queued = bridge.synthesizeSpeech(options).then((value) => ({ value }), (error) => ({ error }));
    options.streamAudio = true;
    await driver.settle();
    const [firstCall] = driver.calls('synthesizeSpeech');
    const oneShot = { pcm: new Float32Array(full), sampleCount: full.length };
    driver.reply(firstCall.id, oneShot);
    await active;
    await driver.settle();
    const [, secondCall] = driver.calls('synthesizeSpeech');
    const streamed = secondCall.args[0].streamAudio === true;
    driver.reply(secondCall.id, streamed ? { ...oneShot, pcm: new Float32Array(0) } : oneShot);
    const result = await queued;
    const dispose = bridge.dispose();
    await driver.settle();
    for (const { id } of driver.calls('dispose')) driver.reply(id, true);
    await dispose;
    assert.equal(streamed, false);
    assert.deepEqual(result, { value: oneShot });
  });
});

test('mutating streamed progress cannot truncate direct synthesis or change the result metadata', async () => {
  const { bridge } = createSpeechBridge({ script: cancellationScript });
  const chunks = [];
  const result = await bridge.synthesizeSpeech({
    text: 'Hello.', streamAudio: true,
    onProgress(event) {
      chunks.push([...event.pcm]);
      event.state = 3;
      event.framesGenerated = 500;
      event.truncated = true;
    },
  });
  assert.deepEqual(chunks, [first, tail]);
  assert.equal(result.sampleCount, first.length + tail.length);
  assert.equal(result.framesGenerated, 1);
  assert.equal(result.truncated, false);
});

test('a thrown streamed callback cancels the worker and rejects with the original error', async () => {
  const failure = new Error('ggml_webgpu: device error from the consumer');
  for (const run of [
    (options) => createSpeechBridge({ script: cancellationScript }).bridge.synthesizeSpeech(options)
      .then((value) => ({ value }), (error) => ({ error })),
    runWorkerSpeech,
  ]) {
    let callbacks = 0;
    const result = await run({
      text: 'Hello.', streamAudio: true,
      onProgress() { callbacks += 1; throw failure; },
    });
    assert.equal(result.error, failure);
    assert.equal(callbacks, 1);
    if (result.cancelCalls) {
      assert.equal(result.cancelCalls.length, 1);
      assert.equal(result.cancelCalls[0].operationId, result.calls[0].operationId);
      assert.equal(result.calls.length, 1);
      assert.deepEqual(result.eventErrors, []);
    }
  }
});

test('later cancellation cannot replace a streamed callback failure while the worker settles', async () => {
  for (const useSignal of [false, true]) {
    let bridge;
    const controller = new AbortController();
    const failure = new Error('consumer failed');
    const result = await runWorkerSpeech({
      text: 'Hello.', streamAudio: true, signal: controller.signal,
      onProgress() { throw failure; },
    }, {
      onBridge(value) { bridge = value; },
      afterProgress(state) {
        if (state !== 2) return;
        if (useSignal) controller.abort();
        else bridge.cancel();
      },
    });
    assert.equal(result.error, failure);
    assert.deepEqual(result.eventErrors, []);
    assert.equal(result.calls.length, 1);
  }
});

for (const [label, trigger, expected] of [
  ['abort from the first chunk', (event, controller) => { if (event.state === 2) controller.abort(); }, [first]],
  ['abort from the final chunk', (event, controller) => { if (event.state === 3) controller.abort(); }, [first, tail]],
  ['bare cancel from the first chunk', (event, _controller, bridge) => { if (event.state === 2) bridge.cancel(); }, [first]],
  ['abort after the last chunk before the result', (event, controller) => {
    if (event.state === 3) queueMicrotask(() => controller.abort());
  }, [first, tail]],
]) {
  test(`streamed synthesis rejects ${label} in direct and worker mode`, async () => {
    const direct = createSpeechBridge({ script: cancellationScript });
    const directController = new AbortController();
    const directChunks = [];
    const directResult = direct.bridge.synthesizeSpeech({
      text: 'Hello.',
      streamAudio: true,
      signal: directController.signal,
      onProgress(event) {
        if (event.pcm) {
          directChunks.push([...event.pcm]);
          trigger(event, directController, direct.bridge);
        }
      },
    }).then((value) => ({ value }), (error) => ({ error }));

    let bridge;
    const workerController = new AbortController();
    const workerChunks = [];
    const worker = await runWorkerSpeech({
      text: 'Hello.',
      streamAudio: true,
      signal: workerController.signal,
      onProgress(event) {
        workerChunks.push([...event.pcm]);
        trigger(event, workerController, bridge);
      },
    }, { onBridge: (value) => { bridge = value; } });
    const directOutcome = await directResult;
    assert.equal(directOutcome.error?.name, 'AbortError', 'direct');
    assert.equal(worker.error?.name, 'AbortError', 'worker');
    assert.deepEqual(directChunks, expected);
    assert.deepEqual(workerChunks, expected);
    assert.equal(direct.calls.filter(([method]) => method === 'step').length, expected.length);
  });
}

test('a signal aborted from the only PCM chunk rejects direct synthesis', async () => {
  const { bridge } = createSpeechBridge({ script: [{ state: 3, pcm: tail }] });
  const controller = new AbortController();
  await assert.rejects(bridge.synthesizeSpeech({
    text: 'Hello.',
    streamAudio: true,
    signal: controller.signal,
    onProgress(event) { if (event.pcm) controller.abort(); },
  }), { name: 'AbortError' });
});

test('worker observes cancellation before returning from the chunk event', async () => {
  let bridge;
  let afterCallback = false;
  let observed = false;
  let observedBeforeReturn = false;
  const worker = await runWorkerSpeech({
    text: 'Hello.', streamAudio: true,
    onProgress() {
      bridge.cancel();
      afterCallback = true;
    },
  }, {
    onBridge(value) {
      bridge = value;
      const isCancelled = bridge._isOperationCancellation.bind(bridge);
      bridge._isOperationCancellation = (...args) => {
        const cancelled = isCancelled(...args);
        observed ||= afterCallback && cancelled;
        return cancelled;
      };
    },
    afterProgress(state) {
      if (state === 2) observedBeforeReturn = observed;
    },
  });
  assert.equal(worker.error?.name, 'AbortError');
  assert.equal(observedBeforeReturn, true);
});

test('worker checks bare cancellation again before inspecting the successful reply', async () => {
  let bridge;
  let inspected = false;
  const worker = await runWorkerSpeech({
    text: 'Hello.', streamAudio: true, onProgress() {},
  }, {
    onBridge(value) { bridge = value; },
    beforeResult() {
      bridge.cancel();
      return {
        get pcm() {
          inspected = true;
          return new Float32Array(0);
        },
        sampleCount: first.length + tail.length,
      };
    },
  });
  assert.equal(worker.error?.name, 'AbortError');
  assert.equal(inspected, false);
});

function cancelAfterQueueRelease(bridge) {
  return new Promise((resolve, reject) => {
    let attempts = 0;
    function cancelWhenReleased() {
      if (!bridge._activeOperation) {
        bridge.cancel();
        resolve();
      } else if (++attempts >= 16) {
        reject(new Error('speech operation did not leave the queue'));
      } else {
        queueMicrotask(cancelWhenReleased);
      }
    }
    queueMicrotask(cancelWhenReleased);
  });
}

test('bare cancellation after queue release still rejects streamed direct synthesis', async () => {
  const direct = createSpeechBridge({ script: [{ state: 3, pcm: tail }] }).bridge;
  let directCancel;
  const pending = direct.synthesizeSpeech({
    text: 'Hello.', streamAudio: true,
    onProgress() { directCancel = cancelAfterQueueRelease(direct); },
  });
  const directOutcome = pending.then(() => undefined, (error) => error);
  assert.equal((await directOutcome)?.name, 'AbortError', 'direct');
  await directCancel;
});

test('bare cancellation after queue release still rejects streamed worker synthesis', async () => {
  let workerBridge;
  let workerCancel;
  const worker = await runWorkerSpeech({
    text: 'Hello.', streamAudio: true, onProgress() {},
  }, {
    onBridge(value) { workerBridge = value; },
    afterResult() { workerCancel = cancelAfterQueueRelease(workerBridge); },
  });
  await workerCancel;
  assert.equal(worker.error?.name, 'AbortError', 'worker');
});

test('direct synthesis discards PCM from a step cancelled while in flight', async () => {
  for (const cancelWithSignal of [false, true]) {
    const { bridge, core, calls } = createSpeechBridge({
      script: [{ state: 2, pcm: first }, { state: 4, status: -6 }],
    });
    const entered = Promise.withResolvers();
    const completion = Promise.withResolvers();
    const ccall = core.ccall;
    let firstStep = true;
    core.ccall = (name, ...args) => {
      const result = ccall(name, ...args);
      if (name !== 'llamadart_webgpu_tts_step' || !firstStep) return result;
      firstStep = false;
      entered.resolve();
      return completion.promise.then(() => result);
    };
    const controller = new AbortController();
    const chunks = [];
    const pending = bridge.synthesizeSpeech({
      text: 'Hello.', streamAudio: true, signal: controller.signal,
      onProgress(event) { if (event.pcm) chunks.push(event.pcm); },
    });
    const rejected = assert.rejects(pending, { name: 'AbortError' });
    await entered.promise;
    if (cancelWithSignal) controller.abort();
    else bridge.cancel();
    completion.resolve();
    await rejected;
    assert.deepEqual(chunks, []);
    assert.ok(calls.every(([method]) => method !== 'write_pcm_chunk'));
  }
});

test('a streamed request retains its delivery callback until it settles', async () => {
  for (const run of [
    async (options) => ({ value: await createSpeechBridge({ script: cancellationScript }).bridge.synthesizeSpeech(options) }),
    runWorkerSpeech,
  ]) {
    let delivered = 0;
    const options = {
      text: 'Hello.', streamAudio: true,
      onProgress(event) {
        delivered += event.pcm.length;
        delete options.onProgress;
      },
    };
    const result = await run(options);
    assert.equal(result.value.sampleCount, delivered);
    assert.equal(delivered, first.length + tail.length);
  }
});

test('streamed options retain prototype values and the original callback receiver', async () => {
  class SpeechRequest {
    get text() { return 'Hello.'; }
    get streamAudio() { return true; }
    delivered = 0;
    onProgress(event) { this.delivered += event.pcm.length; }
  }
  for (const run of [
    async (options) => ({ value: await createSpeechBridge({ script: cancellationScript }).bridge.synthesizeSpeech(options) }),
    runWorkerSpeech,
  ]) {
    const options = new SpeechRequest();
    const result = await run(options);
    assert.equal(result.value.sampleCount, options.delivered);
    assert.equal(options.delivered, first.length + tail.length);
  }
});

test('aborting a released stream does not send native cancellation to the FIFO successor', async () => {
  const { bridge, core } = createSpeechBridge({ script: [{ state: 3, pcm: tail }] });
  const controller = new AbortController();
  const ccall = core.ccall;
  let starts = 0;
  let sentCancel = false;
  core.ccall = (name, ...args) => {
    if (name === 'llamadart_webgpu_tts_start' && ++starts === 2) controller.abort();
    if (name === 'llamadart_webgpu_request_cancel') sentCancel = true;
    return ccall(name, ...args);
  };
  const first = bridge.synthesizeSpeech({
    text: 'First.', streamAudio: true, signal: controller.signal, onProgress() {},
  });
  const second = bridge.synthesizeSpeech({ text: 'Second.', streamAudio: true, onProgress() {} });
  await Promise.all([
    assert.rejects(first, { name: 'AbortError' }),
    assert.doesNotReject(second),
  ]);
  assert.equal(sentCancel, false);
});

test('worker synthesis rejects a successful result whose sample count exceeds delivered PCM', async () => {
  const worker = await runWorkerSpeech({
    text: 'Hello.', streamAudio: true, onProgress() {},
  }, { emitSecond: false });
  assert.match(worker.error?.message ?? '', /streamed PCM sample count/i);
});

test('an older worker rejects streaming with a typed error before sending synthesis', async () => {
  const worker = await runWorkerSpeech({
    text: 'Hello.', streamAudio: true, onProgress() {},
  }, { legacy: true });
  assert.equal(worker.error?.name, 'TextToSpeechStreamingUnsupportedError');
  assert.ok(worker.error instanceof api.TextToSpeechStreamingUnsupportedError);
  assert.equal(worker.error.code, 'TTS_STREAMING_UNSUPPORTED');
  assert.match(worker.error.message, /supportsIncrementalAudio.*bridge and worker scripts.*same release/i);
  assert.deepEqual(worker.calls, []);
});

test('an older worker still serves non-streamed synthesis', async () => {
  const worker = await runWorkerSpeech({ text: 'Hello.' }, { legacy: true });
  assert.equal(worker.error, undefined);
  assert.equal(worker.calls.length, 1);
  assert.deepEqual(worker.value.pcm, new Float32Array([...first, ...tail]));
  assert.equal(worker.value.sampleCount, first.length + tail.length);
});
