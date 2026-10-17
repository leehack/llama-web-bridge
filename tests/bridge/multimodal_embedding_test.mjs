import assert from 'node:assert/strict';
import { test } from 'node:test';
import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';
import { EmbeddingError, snapshotEmbedInput } from '../../js/src/internal/embedding.ts';
import { withStubWorkerEnvironment, createRealWorkerBridge, workerDriver } from './bridge_operation_queue_fixtures.mjs';
import { installBridgeWorkerHost } from '../../js/src/worker_host.ts';

function fixture({ caps = { apiVersion: 1, text: true, image: true, audio: true }, rc = 768 } = {}) {
  const bridge = new LlamaWebGpuBridge({ disableWorker: true });
  const runtime = bridge._runtime;
  const calls = [];
  runtime._modelBytes = 1;
  runtime._mmProjPath = '/projector';
  runtime.getModelMetadata = () => ({});
  runtime.getContextSize = () => 2048;
  runtime._stageMultimodalParts = async (parts) => { calls.push(['stage', parts]); };
  runtime._clearPendingMedia = () => { calls.push(['clear']); };
  runtime._core = {
    _llamadart_webgpu_embedding_capabilities_json() {},
    _llamadart_webgpu_embed_parts_to_json() {},
    ccall(name, _result, _types, args = []) {
      calls.push([name, ...args]);
      if (name === 'llamadart_webgpu_embedding_capabilities_json') return JSON.stringify(caps);
      if (name === 'llamadart_webgpu_last_embedding_json') return '[0.6,0.8]';
      if (name === 'llamadart_webgpu_last_error') return 'internal detail must not include a payload';
      return rc;
    },
  };
  return { bridge, runtime, calls };
}

test('strings keep their existing export, normalization default and batch route', async () => {
  const { bridge, calls } = fixture();
  assert.deepEqual(await bridge.embed('hello'), [0.6, 0.8]);
  await bridge.embedBatch(['one', 'two'], { normalize: false });
  assert.deepEqual(calls.filter(([n]) => n === 'llamadart_webgpu_embed_to_json'), [
    ['llamadart_webgpu_embed_to_json', 'hello', 1],
    ['llamadart_webgpu_embed_to_json', 'one', 0],
    ['llamadart_webgpu_embed_to_json', 'two', 0],
  ]);
  assert.ok(!calls.some(([n]) => n === 'llamadart_webgpu_embedding_capabilities_json'));
});

test('parts form one ordered vector with verbatim adjacent text and no prompt decoration', async () => {
  const { bridge, calls } = fixture();
  const image = { type: 'image', bytes: new Uint8Array([1, 2, 3]), width: 1, height: 1 };
  const audio = { type: 'audio', samples: new Float32Array([0]) };
  const parts = [{ type: 'text', text: ' a' }, { type: 'text', text: '\n雪' }, image,
    { type: 'text', text: '' }, { type: 'text', text: ' z ' }, audio];
  assert.deepEqual(await bridge.embed(parts, { normalize: false }), [0.6, 0.8]);
  assert.deepEqual(calls.find(([n]) => n === 'stage')[1], [image, audio]);
  assert.deepEqual(calls.find(([n]) => n === 'llamadart_webgpu_embed_parts_to_json'),
    ['llamadart_webgpu_embed_parts_to_json', ' a\n雪<__media__> z <__media__>', 0,
      new TextEncoder().encode(' a\n雪<__media__> z <__media__>').length]);
  assert.equal(calls.at(-1)[0], 'clear');
});

test('empty batch does no work; empty parts reject without inference', async () => {
  const { bridge, calls } = fixture();
  assert.deepEqual(await bridge.embedBatch([]), []);
  await assert.rejects(bridge.embed([]), { name: 'EmbeddingError', code: 'INVALID_EMBEDDING_INPUT' });
  await assert.rejects(bridge.embed(new Array(1)), { code: 'INVALID_EMBEDDING_INPUT' });
  await assert.rejects(bridge.embed([, { type: 'text', text: 'x' }]), { code: 'INVALID_EMBEDDING_INPUT' });
  assert.deepEqual(calls, []);
});

for (const [name, caps, code, input] of [
  ['unknown probe version', { apiVersion: 2, text: true, image: true, audio: true }, 'EMBEDDING_CAPABILITY_UNAVAILABLE', [{ type: 'text', text: 'x' }]],
  ['non embedding model', { apiVersion: 1, text: false, image: false, audio: false }, 'MODEL_NOT_EMBEDDING', [{ type: 'text', text: 'x' }]],
  ['unsupported image', { apiVersion: 1, text: true, image: false, audio: true }, 'EMBEDDING_MODALITY_UNSUPPORTED', [{ type: 'image', bytes: new Uint8Array([1]) }]],
  ['unsupported audio', { apiVersion: 1, text: true, image: true, audio: false }, 'EMBEDDING_MODALITY_UNSUPPORTED', [{ type: 'audio', samples: new Float32Array([0]) }]],
]) {
  test(name, async () => {
    const { bridge, calls } = fixture({ caps });
    await assert.rejects(bridge.embed(input), { name: 'EmbeddingError', code });
    assert.ok(!calls.some(([n]) => n === 'stage' || n === 'llamadart_webgpu_embed_parts_to_json'));
  });
}

test('absent probe and missing mixed export fail closed without affecting legacy strings', async () => {
  for (const symbol of ['_llamadart_webgpu_embedding_capabilities_json', '_llamadart_webgpu_embed_parts_to_json']) {
    const { bridge, runtime } = fixture();
    delete runtime._core[symbol];
    await assert.rejects(bridge.getEmbeddingCapabilities(), { code: 'EMBEDDING_CAPABILITY_UNAVAILABLE' });
    await assert.rejects(bridge.embed([{ type: 'text', text: 'x' }]), { code: 'EMBEDDING_CAPABILITY_UNAVAILABLE' });
    assert.deepEqual(await bridge.embed('x'), [0.6, 0.8]);
  }
});

test('unloaded model capabilities are all false without core IO', async () => {
  const { bridge, runtime, calls } = fixture();
  runtime._modelBytes = 0;
  assert.deepEqual(await bridge.getEmbeddingCapabilities(), { apiVersion: 1, text: false, image: false, audio: false });
  assert.deepEqual(calls, []);
});

test('projector is required before media staging', async () => {
  const { bridge, runtime, calls } = fixture();
  runtime._mmProjPath = null;
  await assert.rejects(bridge.embed([{ type: 'audio', samples: new Float32Array([0]) }]), { code: 'EMBEDDING_PROJECTOR_MISSING' });
  assert.ok(!calls.some(([n]) => n === 'stage'));
});

test('whole batch modality preflight happens before the first inference', async () => {
  const { bridge, calls } = fixture({ caps: { apiVersion: 1, text: true, image: false, audio: false } });
  await assert.rejects(bridge.embedBatch(['valid', [{ type: 'image', bytes: new Uint8Array([1]) }]]), { code: 'EMBEDDING_MODALITY_UNSUPPORTED' });
  assert.ok(!calls.some(([n]) => n.includes('to_json') || n === 'stage'));
});

for (const input of [
  [{ type: 'video', url: 'secret' }],
  [{ type: 'text', text: 123 }],
  [{ type: 'text', text: 'ambiguous <__media__>' }],
  [{ type: 'text', text: '<__' }, { type: 'text', text: 'media__>' }],
  [null],
  [{ type: 'image', path: '/secret' }],
  [{ type: 'image', path: '/secret', bytes: new Uint8Array([1]) }],
  [{ type: 'image' }],
  [{ type: 'image', bytes: [] }],
  [{ type: 'image', bytes: [1] }],
  [{ type: 'image', bytes: new Uint8Array() }],
  [{ type: 'image', url: '' }],
  [{ type: 'image', url: 123 }],
  [{ type: 'image', bytes: new Uint8Array([1]), url: 'https://secret' }],
  [{ type: 'audio', samples: [] }],
  [{ type: 'audio', samples: new Float32Array() }],
  [{ type: 'audio', samples: new Float32Array([NaN]) }],
  [{ type: 'image', samples: new Float32Array([0]) }],
]) {
  test(`malformed or unsupported part is rejected without staging: ${input[0]?.type} ${Object.keys(input[0] || {}).join(',')}`, async () => {
    const { bridge, calls } = fixture();
    assert.throws(()=>snapshotEmbedInput(input),{name:'EmbeddingError'});
    await assert.rejects(bridge.embed(input), (error) => {
      assert.equal(error.name, 'EmbeddingError');
      assert.doesNotMatch(error.message, /secret|123|NaN/);
      return true;
    });
    assert.ok(!calls.some(([n]) => n === 'stage'));
  });
}

test('staging failure cleans pending media and hides media diagnostics', async () => {
  const { bridge, runtime, calls } = fixture();
  runtime._stageMultimodalParts = async () => { throw new Error('https://secret/?token=private'); };
  await assert.rejects(bridge.embed([{ type: 'image', url: 'https://secret/?token=private' }]), (error) => {
    assert.equal(error.code, 'EMBEDDING_FAILED');
    assert.doesNotMatch(error.message, /secret|private/);
    return true;
  });
  assert.equal(calls.at(-1)[0], 'clear');
});

test('parts preserve embedded NUL and UTF-8 byte length across the core boundary', async () => {
  const { bridge, calls } = fixture();
  await bridge.embed([{type:'text',text:'a\0雪'}]);
  assert.deepEqual(calls.find(([name])=>name==='llamadart_webgpu_embed_parts_to_json'),
    ['llamadart_webgpu_embed_parts_to_json', 'a\0雪', 1, 5]);
});

test('malformed capability payloads reject every unknown protocol shape', async () => {
  for (const caps of [null, 'invalid JSON', {}, {apiVersion:1,text:1,image:false,audio:false},
    {apiVersion:1,text:true,image:1,audio:false}, {apiVersion:1,text:true,image:false,audio:1}]) {
    const {bridge}=fixture({caps});
    await assert.rejects(bridge.getEmbeddingCapabilities(), {code:'EMBEDDING_CAPABILITY_UNAVAILABLE'});
  }
});

test('cancellation while core inference runs waits for it and discards the result', async () => {
  const {bridge,runtime,calls}=fixture();
  let release, entered;
  const started=new Promise((resolve)=>{entered=resolve;});
  const original=runtime._core.ccall;
  runtime._core.ccall=(name,...args)=>{
    if(name==='llamadart_webgpu_embed_parts_to_json') {
      entered(); return new Promise((resolve)=>{release=()=>resolve(768);});
    }
    return original(name,...args);
  };
  const pending=bridge.embed([{type:'text',text:'x'}]);
  await started; bridge.cancel();
  const next=bridge.embed('after');
  assert.ok(!calls.some(([name])=>name==='llamadart_webgpu_embed_to_json'));
  release();
  await assert.rejects(pending,{name:'AbortError'});
  assert.deepEqual(await next,[0.6,0.8]);
  assert.ok(calls.some(([name])=>name==='clear'));
});

test('microbatch overflow is a typed inference error and cleans media', async () => {
  const { bridge, calls } = fixture({ rc: -23 });
  await assert.rejects(bridge.embed([{ type: 'text', text: 'x' }]), { code: 'EMBEDDING_INPUT_TOO_LARGE' });
  assert.equal(calls.at(-1)[0], 'clear');
});

test('worker structured clone preserves ordered media and caller buffer ownership', async () => {
  const { bridge } = fixture();
  const bytes = new Uint8Array([1, 2, 3]);
  const samples = new Float32Array([0, 0.25]);
  const received = [];
  bridge._restoreWorkerModelIfMissing = async () => {};
  bridge._workerProxy = {
    async call(method, args, _event, transfers) {
      assert.deepEqual(transfers, []);
      const cloned = structuredClone(args, { transfer: transfers });
      received.push([method, cloned]);
      return { value: method === 'getEmbeddingCapabilities'
        ? { apiVersion: 1, text: true, image: true, audio: true } : [0.6, 0.8] };
    },
  };
  const parts = [{ type: 'text', text: 'a' }, { type: 'image', bytes }, { type: 'audio', samples }, { type: 'text', text: 'b' }];
  await bridge.embed(parts);
  assert.deepEqual(received.at(-1), ['embed', [parts, {}]]);
  assert.notEqual(received.at(-1)[1][0][1].bytes.buffer, bytes.buffer);
  assert.deepEqual([...bytes], [1, 2, 3]);
  assert.deepEqual([...samples], [0, 0.25]);
});

test('real proxy reconstructs typed embedding errors', async () => {
  await withStubWorkerEnvironment(async ({ workers }) => {
    const bridge = createRealWorkerBridge();
    bridge._restoreWorkerModelIfMissing = async () => {};
    const driver = workerDriver(bridge._workerProxy, workers[0]);
    driver.ready();
    const pending = bridge.embed([{ type: 'text', text: 'x' }]);
    await driver.settle();
    driver.reply(driver.calls('getEmbeddingCapabilities')[0].id,
      { apiVersion: 1, text: true, image: false, audio: false });
    await driver.settle();
    const request = driver.calls('embed')[0];
    bridge._workerProxy._worker.onmessage({ data: {
      type: 'error', id: request.id, message: 'Input exceeds one microbatch.',
      embeddingErrorCode: 'EMBEDDING_INPUT_TOO_LARGE',
    } });
    await assert.rejects(pending, (error) => error instanceof EmbeddingError && error.code === 'EMBEDDING_INPUT_TOO_LARGE');
    assert.ok(bridge._workerProxy, 'application errors must keep the healthy worker');
    bridge._workerProxy._worker.terminate();
  });
});

test('worker host posts payload-free typed embedding error codes', async () => {
  const previousSelf=globalThis.self;
  const original=LlamaWebGpuBridge.prototype._embedUnlocked;
  const messages=[];
  globalThis.self={postMessage(message) {messages.push(message);}};
  LlamaWebGpuBridge.prototype._embedUnlocked=async()=>{
    throw new EmbeddingError('EMBEDDING_INPUT_TOO_LARGE','Input exceeds one microbatch.');
  };
  try {
    installBridgeWorkerHost();
    await self.onmessage({data:{type:'init',id:1,config:{disableWorker:true}}});
    await self.onmessage({data:{type:'call',id:2,method:'embed',args:['x']}});
    const error=messages.find((message)=>message.type==='error');
    assert.equal(error.embeddingErrorCode,'EMBEDDING_INPUT_TOO_LARGE');
    assert.equal(error.message,'Input exceeds one microbatch.');
    await self.onmessage({data:{type:'call',id:3,method:'dispose',args:[]}});
  } finally {
    globalThis.self=previousSelf;
    LlamaWebGpuBridge.prototype._embedUnlocked=original;
  }
});

test('old worker capability protocol fails closed without sending parts or retiring worker', async () => {
  const { bridge } = fixture();
  bridge._restoreWorkerModelIfMissing = async () => {};
  const methods = [];
  bridge._workerProxy = { async call(method) { methods.push(method); throw new Error('unknown method'); } };
  await assert.rejects(bridge.embed([{ type: 'text', text: 'x' }]), { code: 'EMBEDDING_CAPABILITY_UNAVAILABLE' });
  assert.deepEqual(methods, ['getEmbeddingCapabilities']);
  assert.ok(bridge._workerProxy);
});

test('cancellation drains staging, cleans media and permits the next text request', async () => {
  const { bridge, runtime, calls } = fixture();
  let release;
  let started;
  const entered = new Promise((resolve) => { started = resolve; });
  runtime._stageMultimodalParts = async () => {
    started(); await new Promise((resolve) => { release = resolve; });
  };
  const pending = bridge.embed([{ type: 'image', bytes: new Uint8Array([1]) }]);
  await entered;
  bridge.cancel();
  const next = bridge.embed('after cancel');
  assert.ok(!calls.some(([n]) => n === 'llamadart_webgpu_embed_to_json'));
  release();
  await assert.rejects(pending, { name: 'AbortError' });
  assert.deepEqual(await next, [0.6, 0.8]);
  assert.ok(!calls.some(([n]) => n === 'llamadart_webgpu_embed_parts_to_json'));
  assert.ok(calls.find(([n]) => n === 'clear'));
});

test('capabilities are re-probed across projector unload/reload', async () => {
  const caps = { apiVersion: 1, text: true, image: true, audio: true };
  const { bridge, runtime } = fixture({ caps });
  assert.equal((await bridge.getEmbeddingCapabilities()).image, true);
  runtime._mmProjPath = null; caps.image = caps.audio = false;
  assert.equal((await bridge.getEmbeddingCapabilities()).audio, false);
  runtime._mmProjPath = '/projector'; caps.image = caps.audio = true;
  assert.equal((await bridge.getEmbeddingCapabilities()).image, true);
});
