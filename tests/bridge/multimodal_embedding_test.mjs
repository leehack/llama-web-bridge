import assert from 'node:assert/strict';
import { test } from 'node:test';
import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';
import { LlamaWebGpuBridgeRuntime } from '../../js/src/runtime.ts';
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
    ['llamadart_webgpu_embed_parts_to_json', new TextEncoder().encode(' a\n雪<__media__> z <__media__>'), 0,
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
    ['llamadart_webgpu_embed_parts_to_json', new TextEncoder().encode('a\0雪'), 1, 5]);
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
  bridge._workerProxy = { async call(method) { methods.push(method); throw new TypeError('bridge[method] is not a function'); } };
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


for (const method of ['embed', 'embedBatch', 'getEmbeddingCapabilities']) {
  for (const failure of ['crash', 'timeout', 'core abort']) {
    test(`${method} recovers a worker ${failure} during capability probing and reloads its projector`, async () => {
      const {bridge,runtime}=fixture();
      runtime._mmProjPath=null;
      runtime.supportsVision=()=>!!runtime._mmProjPath;
      runtime.supportsAudio=()=>!!runtime._mmProjPath;
      let loads=0;
      runtime.loadMultimodalProjector=async()=>{ loads++; runtime._mmProjPath='/reloaded'; };
      bridge._loadedMmProjUrl='https://example.invalid/projector.gguf';
      bridge._restoreWorkerModelIfMissing=async()=>{};
      bridge._ensureRuntimeDraftModel=async()=>{};
      bridge._restoreLoraAdapters=async()=>{};
      bridge._disableWorkerFallback=()=>{bridge._workerProxy=null;};
      bridge._waitForWorkerDisposal=async()=>{};
      const error=failure==='core abort'?new WebAssembly.RuntimeError('Aborted()'):new Error('worker '+failure);
      if(failure==='crash') error.llamadartWorkerCrash=true;
      if(failure==='timeout') error.llamadartWorkerTimeout=true;
      bridge._loadedModelOptions={nGpuLayers:0};
      bridge._workerProxy={async call(){throw error;}};
      const input=[{type:'image',bytes:new Uint8Array([1])}];
      const result=await (method==='getEmbeddingCapabilities'?bridge[method]():bridge[method](method==='embed'?input:[input]));
      assert.deepEqual(result,method==='getEmbeddingCapabilities'?{apiVersion:1,text:true,image:true,audio:true}:method==='embed'?[0.6,0.8]:[[0.6,0.8]]);
      assert.equal(bridge._workerProxy,null);
      assert.equal(loads,1);
    });
  }
}

for (const method of ['embed','embedBatch']) {
  test(`${method} restores projector after the worker embed itself crashes`, async()=>{
    const {bridge,runtime}=fixture();
    runtime._mmProjPath=null;
    runtime.supportsVision=()=>!!runtime._mmProjPath;
    runtime.supportsAudio=()=>!!runtime._mmProjPath;
    let loads=0;
    runtime.loadMultimodalProjector=async()=>{loads++;runtime._mmProjPath='/reloaded';};
    bridge._loadedMmProjUrl='https://example.invalid/projector.gguf';
    bridge._restoreWorkerModelIfMissing=async()=>{};
    bridge._ensureRuntimeDraftModel=async()=>{};
    bridge._restoreLoraAdapters=async()=>{};
    bridge._disableWorkerFallback=()=>{bridge._workerProxy=null;};
    bridge._waitForWorkerDisposal=async()=>{};
    bridge._workerProxy={async call(name){
      if(name==='getEmbeddingCapabilities') return {value:{apiVersion:1,text:true,image:true,audio:true}};
      throw Object.assign(new Error('worker crashed'),{llamadartWorkerCrash:true});
    }};
    const input=[{type:'image',bytes:new Uint8Array([1])}];
    await bridge[method](method==='embed'?input:[input]);
    assert.equal(loads,1);
  });
}

test('parts core abort retains the worker recovery signal while staging diagnostics remain private', async()=>{
  const {bridge,runtime}=fixture();
  const original=runtime._core.ccall;
  const abort=new WebAssembly.RuntimeError('Aborted()');
  runtime._core.ccall=(name,...args)=>{if(name==='llamadart_webgpu_embed_parts_to_json') throw abort; return original(name,...args);};
  await assert.rejects(bridge.embed([{type:'text',text:'x'}]),error=>error===abort && bridge._isWorkerUnusableError(error));
});

test('parts marshal exact UTF-8 bytes for lone surrogates, surrogate before CJK and NUL', async()=>{
  for(const text of ['x\uD800','\uDC00\uD800','\uD800雪雪',('\uD800雪').repeat(50),'a\0雪']) {
    const {bridge,runtime}=fixture();
    const original=runtime._core.ccall;
    let checked=false;
    runtime._core.ccall=(name,result,types,args)=>{
      if(name==='llamadart_webgpu_embed_parts_to_json') {
        assert.deepEqual(types,['array','number','number']);
        assert.deepEqual(args[0],new TextEncoder().encode(text));
        assert.equal(args[2],args[0].byteLength); checked=true;
      }
      return original(name,result,types,args);
    };
    await bridge.embed([{type:'text',text}]); assert.ok(checked);
  }
});

test('fresh parts embed clears a prior runtime abort',async()=>{
  const {runtime}=fixture();
  runtime._abortRequested=true;
  assert.deepEqual(await runtime.embed([{type:'text',text:'fresh'}]),[0.6,0.8]);
});

test('batch cancellation between legacy string items prevents the next inference',async()=>{
  const {runtime,calls}=fixture();
  const original=runtime._core.ccall;
  runtime._core.ccall=(name,...args)=>{
    if(name==='llamadart_webgpu_embed_to_json') runtime._abortRequested=true;
    return original(name,...args);
  };
  await assert.rejects(runtime.embedBatch(['one','two']),{name:'AbortError'});
  assert.equal(calls.filter(([name])=>name==='llamadart_webgpu_embed_to_json').length,1);
});

test('non-array embedding inputs retain legacy String coercion on facade and runtime',async()=>{
  for(const owner of ['bridge','runtime']) {
    const f=fixture();
    for(const input of [42,null,undefined,true,{toString:()=> 'custom'}]) await f[owner].embed(input);
    await f[owner].embedBatch([42,null,{toString:()=> 'custom'}]);
    assert.deepEqual(f.calls.filter(([n])=>n==='llamadart_webgpu_embed_to_json').map(([,text])=>text),
      ['42','null','undefined','true','custom','42','null','custom']);
  }
});

test('empty batch still rejects on a disposed bridge',async()=>{
  const {bridge}=fixture(); await bridge.dispose();
  await assert.rejects(bridge.embedBatch([]),{message:'Bridge has been disposed.'});
});

test('missing embedding API reports loaded asset version without exposing URL secrets',async()=>{
  const {bridge,runtime}=fixture();
  runtime._config.coreModuleUrl='https://user:secret@example.invalid/pkg/v0.1.39/llama_webgpu_core.js?token=private';
  delete runtime._core._llamadart_webgpu_embedding_capabilities_json;
  await assert.rejects(bridge.getEmbeddingCapabilities(),error=>{
    assert.equal(error.code,'EMBEDDING_CAPABILITY_UNAVAILABLE');
    assert.match(error.message,/v0\.1\.39/); assert.doesNotMatch(error.message,/secret|private|user:/); return true;
  });
});


test('unrelated capability probe failures propagate without claiming old assets',async()=>{
  const {bridge}=fixture();
  bridge._restoreWorkerModelIfMissing=async()=>{};
  const error=new Error('unexpected probe failure');
  bridge._workerProxy={async call(){throw error;}};
  await assert.rejects(bridge.embed([{type:'text',text:'x'}]),e=>e===error);
});


for(const variant of ['wasm32','wasm64']) {
  test(`embedding capability error reports the active ${variant} asset version`,async()=>{
    const {bridge,runtime}=fixture();
    runtime._config.coreModuleUrl='https://example.invalid/v0.1.39/core.js';
    runtime._config.coreModuleUrlMem64='https://example.invalid/v0.1.40/core.js';
    runtime._coreVariant=variant;
    delete runtime._core._llamadart_webgpu_embedding_capabilities_json;
    await assert.rejects(bridge.getEmbeddingCapabilities(),error=>{
      assert.match(error.message,variant==='wasm64'?/v0\.1\.40/:/v0\.1\.39/);
      assert.doesNotMatch(error.message,variant==='wasm64'?/v0\.1\.39/:/v0\.1\.40/);
      return true;
    });
  });
}


// Recovery starts from a GPU-loaded worker and a fresh direct runtime. The
// fake core still runs the real bridge/runtime embedding and preflight paths.
function recoveryFixture(error, { rejectProjector = false } = {}) {
  const f = fixture();
  const { bridge, runtime } = f;
  const loads = [], warnings = [], projectors = [];
  runtime._modelBytes = 0;
  runtime._mmProjPath = null;
  runtime.supportsVision = () => !!runtime._mmProjPath;
  runtime.supportsAudio = () => !!runtime._mmProjPath;
  runtime.loadModelFromUrl = async (url, options) => {
    loads.push({ url, options });
    runtime._modelBytes = 1;
  };
  runtime.loadMultimodalProjector = async (url) => {
    projectors.push(url);
    if (rejectProjector) throw new Error('projector fetch failed');
    runtime._mmProjPath = '/restored';
  };
  bridge._loadedModelUrl = 'https://example.invalid/model.gguf';
  bridge._loadedModelOptions = { nGpuLayers: 99, nCtx: 8192, nBatch: 2048, nUbatch: 1024 };
  bridge._loadedMmProjUrl = 'https://example.invalid/projector.gguf';
  bridge._restoreWorkerModelIfMissing = async () => {};
  bridge._ensureRuntimeDraftModel = async () => {};
  bridge._restoreLoraAdapters = async () => {};
  bridge._disableWorkerFallback = () => { bridge._workerProxy = null; };
  bridge._waitForWorkerDisposal = async () => {};
  bridge._emitBridgeWarn = (message) => warnings.push(message);
  bridge._workerProxy = { async call(method) {
    if (method === 'getEmbeddingCapabilities')
      return { value: { apiVersion: 1, text: true, image: true, audio: true } };
    throw error;
  } };
  return { ...f, loads, warnings, projectors };
}

const recoveryImage = [{ type: 'image', bytes: new Uint8Array([1]), width: 1, height: 1 }];
for (const method of ['embed', 'embedBatch']) {
  for (const [label, input, message] of [
    ['image parts + Aborted()', recoveryImage, 'RuntimeError: Aborted()'],
    ['text-only parts + device error', [{ type: 'text', text: 'word '.repeat(100) }], 'ggml_webgpu: device error'],
    ['string + device error', 'word '.repeat(100), 'ggml_webgpu: device error'],
  ]) {
    test(`${method} GPU fallback preserves embedding capacity: ${label}`, async () => {
      const error = Object.assign(new Error(message), { llamadartWorkerCrash: true });
      const { bridge, loads, warnings, projectors } = recoveryFixture(error);
      const run = (value) => bridge[method](method === 'embed' ? value : [value]);
      await run(input);
      assert.equal(loads.length, 1);
      const { signal, ...loadOptions } = loads[0].options;
      assert.ok(signal instanceof AbortSignal);
      assert.deepEqual(loadOptions, { nGpuLayers: 0, nCtx: 8192, nBatch: 2048, nUbatch: 1024 });
      assert.equal(projectors.length, label.startsWith('image') ? 1 : 0);
      assert.ok(warnings.some((message) => /embedding.*CPU fallback/.test(message)));
      assert.ok(warnings.every((message) => !/generation/.test(message)));
      // The fallback must not poison capacity for later requests either.
      await run('word '.repeat(100));
      assert.equal(loads.length, 1);
    });
  }

  test(`${method} string fallback ignores a broken projector and later media restores it lazily`, async () => {
    const { bridge, runtime, projectors, loads } = recoveryFixture(
      Object.assign(new Error('worker crashed'), { llamadartWorkerCrash: true }),
      { rejectProjector: true },
    );
    const run = (value) => bridge[method](method === 'embed' ? value : [value]);
    assert.deepEqual(await run('hello'), method === 'embed' ? [0.6, 0.8] : [[0.6, 0.8]]);
    assert.equal(projectors.length, 0);
    assert.equal(loads.length, 1);
    assert.ok(bridge._loadedModelUrl);
    assert.ok(bridge._loadedMmProjUrl);
    assert.equal(runtime._mmProjPath, null);
    runtime.loadMultimodalProjector = async (url) => { projectors.push(url); runtime._mmProjPath = '/restored'; };
    assert.deepEqual(await run(recoveryImage), method === 'embed' ? [0.6, 0.8] : [[0.6, 0.8]]);
    assert.deepEqual(projectors, ['https://example.invalid/projector.gguf']);
    assert.equal(loads.length, 1);
    await run(recoveryImage);
    assert.equal(projectors.length, 1);
  });
}

test('text-only completion parts are not media', () => {
  const { bridge } = fixture();
  assert.equal(bridge._hasMediaParts({ parts: [{ type: 'text', text: 'x' }] }), false);
  assert.equal(bridge._hasMediaParts({ parts: recoveryImage }), true);
  assert.equal(bridge._hasMediaParts({ parts: [{ type: 'audio', samples: new Float32Array([0]) }] }), true);
});

test('direct capability RuntimeError propagates without entering worker recovery', async () => {
  const { bridge, runtime } = fixture();
  const error = new WebAssembly.RuntimeError('Aborted()');
  let probes = 0;
  runtime.getEmbeddingCapabilities = () => { probes++; throw error; };
  bridge._disableWorkerFallback = () => assert.fail('direct runtime must not retire a worker');
  bridge._ensureRuntimeReadyAfterWorkerFallback = () => assert.fail('direct runtime must not recover');
  await assert.rejects(bridge.getEmbeddingCapabilities(), (actual) => actual === error);
  assert.equal(probes, 1);
});

test('native -20 preserves capability-unavailable classification and active asset version', async () => {
  const { bridge, runtime, calls } = fixture({ rc: -20 });
  runtime._config.coreModuleUrl = 'https://user:secret@example.invalid/v0.1.39/core.js?token=private';
  await assert.rejects(bridge.embed([{ type: 'text', text: 'x' }]), (error) => {
    assert.ok(error instanceof EmbeddingError);
    assert.equal(error.code, 'EMBEDDING_CAPABILITY_UNAVAILABLE');
    assert.match(error.message, /v0\.1\.39/);
    assert.doesNotMatch(error.message, /secret|private|user:/);
    return true;
  });
  assert.equal(calls.at(-1)[0], 'clear');
});


for (const dimensions of [{}, { nBatch: 0, nUbatch: 0 }, { nBatch: 2048 }, { nUbatch: 512 },
  { nCtx: 0, nBatch: Infinity, nUbatch: Infinity }]) {
  test(`GPU recovery retains native batch defaults: ${JSON.stringify(dimensions)}`, async () => {
    const { bridge, runtime, loads } = recoveryFixture(
      Object.assign(new Error('ggml_webgpu: device error'), { llamadartWorkerCrash: true }),
    );
    bridge._loadedModelOptions = { nGpuLayers: 99, ...dimensions };
    bridge._contextSize = 8192;
    runtime._nCtx = 8192;
    runtime._probeBackends = async () => {};
    runtime._ensureCore = async () => runtime._core;
    runtime._releaseLoadedModel = () => {};
    runtime._syncThreadPoolSizeHintFromCore = () => {};
    runtime._coreSupportsPthreads = () => false;
    const resolved = new Error('stop before any model IO');
    runtime._resolveNativeLoadOptions = () => { throw resolved; };
    const resolve = async (options) => {
      await assert.rejects(LlamaWebGpuBridgeRuntime.prototype.loadModelFromUrl.call(
        runtime, 'https://example.invalid/model.gguf', options), (error) => error === resolved);
      return { nCtx: runtime._nCtx, nBatch: runtime._nBatch, nUbatch: runtime._nUbatch };
    };
    const gpu = await resolve(bridge._loadedModelOptions);
    // CPU loads normally choose smaller defaults; the recovery-only hint must
    // preserve the zero sentinels that select the original native defaults.
    const cpu = await resolve({ ...bridge._loadedModelOptions, nGpuLayers: 0 });
    assert.ok(cpu.nUbatch > 0);
    let recovered;
    runtime.loadModelFromUrl = async (url, options) => {
      loads.push({ url, options });
      recovered = await resolve(options);
      runtime._modelBytes = 1;
    };
    runtime._nCtx = 2048; // the replacement starts with a fresh runtime's context
    await bridge.embed('word '.repeat(100));
    assert.equal(loads[0].options.nGpuLayers, 0);
    assert.deepEqual(recovered, gpu);
    assert.equal(recovered.nCtx, 8192);
  });
}

test('capability cancellation before recovery never retires the worker or reloads the model', async () => {
  const { bridge, loads } = recoveryFixture(
    Object.assign(new Error('worker crashed'), { llamadartWorkerCrash: true }),
  );
  bridge._workerProxy.call = async () => { throw Object.assign(new Error('worker crashed'), { llamadartWorkerCrash: true }); };
  const probe = bridge._getEmbeddingCapabilitiesUnlocked.bind(bridge);
  // Cancel at the boundary after the inner probe has rejected, before the
  // public capability operation handles that rejection and attempts recovery.
  bridge._getEmbeddingCapabilitiesUnlocked = async () => {
    try { return await probe(); }
    catch (error) { bridge.cancel(); throw error; }
  };
  let retirements = 0;
  bridge._disableWorkerFallback = () => { retirements++; bridge._workerProxy = null; };
  await assert.rejects(bridge.getEmbeddingCapabilities(), { name: 'AbortError' });
  assert.equal(retirements, 0);
  assert.equal(loads.length, 0);
});
