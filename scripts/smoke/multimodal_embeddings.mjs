#!/usr/bin/env node
// Opt-in local-weight qualification. This script never downloads weights.
import fsp from 'node:fs/promises';
import path from 'node:path';
import {
  copyBridgeArtifacts, copyMemory64Artifacts, ensure, parseSmokeArgs,
  resolvePinnedFile, runMain, runPollingPlaywright, withServer, withTempDir,
} from './support.mjs';

export function renderHarness(memoryMode) {
  return `<!doctype html><meta charset="utf-8"><pre id="result">pending</pre><script type="module">
const finish = (payload) => { window.__smokeResult = payload; document.querySelector('pre').textContent = JSON.stringify(payload); };
const check = (condition, message) => { if (!condition) throw new Error(message); };
const delta = (a,b) => { check(a.length === b.length, 'embedding width mismatch'); return Math.max(...a.map((x,i)=>Math.abs(x-b[i]))); };
const stage = (value) => { window.__smokeStage = value; console.log('EMBEDDING_STAGE '+value); };
const expectCode = async (operation, code) => {
  try { await operation(); } catch (error) { check(error.code === code, 'wrong embedding error code: '+error.code); return; }
  throw new Error('expected '+code);
};
const modeResults = [];
try {
  check(crossOriginIsolated, 'cross-origin isolation required');
  const { LlamaWebGpuBridge } = await import('/llama_webgpu_bridge.js');
  const reference = await (await fetch('/reference.json')).json();
  const pixels = new Uint8Array(224*224*3);
  const black = { type: 'image', width: 224, height: 224, bytes: pixels.slice() };
  for(let i=0;i<pixels.length;++i) pixels[i] = ((Math.floor(i/3/224/16)+Math.floor((Math.floor(i/3)%224)/16))%2)?255:0;
  const checker = { type: 'image', width: 224, height: 224, bytes: pixels };
  const silence = { type: 'audio', samples: new Float32Array(16000) };
  const tone = { type: 'audio', samples: Float32Array.from({length:16000},(_,i)=>Math.fround(0.2)*Math.sin(2*Math.PI*440*i/16000)) };
  const prefix = {type:'text',text:'task: search result | text: '};
  const text = 'task: search result | title: Paris | text: The Eiffel Tower is in Paris.';
  const inputs = { text, black:[prefix,black], checker:[prefix,checker], silence:[prefix,silence], tone:[prefix,tone],
    interleaved:[prefix,{type:'text',text:'A '},black,{type:'text',text:' B'}, {type:'text',text:'\\n雪'},tone,{type:'text',text:' C '},checker] };
  const parityFailures = [];
  for(const disableWorker of [true,false]) {
    const mode = disableWorker?'direct':'worker';
    const bridge = new LlamaWebGpuBridge({disableWorker, logLevel:1, preferMemory64:${memoryMode === 'wasm64'},
      ${memoryMode === 'wasm64' ? "coreModuleUrlMem64:'/llama_webgpu_core_mem64.js'," : ''}});
    const vectors = {}, differences = {};
    const row = {mode,vectors,differences};
    modeResults.push(row);
    try {
      stage(mode+' model load');
      await bridge.loadModelFromUrl('/model.gguf',{nCtx:2048,nBatch:2048,nUbatch:2048,nGpuLayers:0,nThreads:4,nThreadsBatch:4,useCache:false,forceRemoteFetchBackend:false});
      let caps=await bridge.getEmbeddingCapabilities();
      check(caps.apiVersion===1 && caps.text && !caps.image && !caps.audio,'pre-projector capabilities');
      await expectCode(()=>bridge.embed([black]),'EMBEDDING_PROJECTOR_MISSING');
      stage(mode+' projector load');
      await bridge.loadMultimodalProjector('/projector.gguf');
      caps=await bridge.getEmbeddingCapabilities();
      check(caps.text && caps.image && caps.audio,'projector capabilities');
      check((await bridge.embedBatch([])).length===0,'empty batch');
      await expectCode(()=>bridge.embed([]),'INVALID_EMBEDDING_INPUT');
      await expectCode(()=>bridge.embed([{type:'text',text:'word '.repeat(3000)}]),'EMBEDDING_INPUT_TOO_LARGE');
      for(const [name,input] of Object.entries(inputs)) {
        stage(mode+' '+name);
        const vector=await bridge.embed(input);
        check(vector.length===768 && vector.every(Number.isFinite),'finite full pooled output');
        check(Math.abs(vector.reduce((s,v)=>s+v*v,0)-1)<1e-5,'unit L2 norm');
        const difference=delta(vector,reference[name].normalized);
        // WASM SIMD and native scalar CPU reductions are different. These are
        // component-wise absolute tolerances, not retrieval-quality claims.
        if (difference >= (name==='text'?1e-5:2e-4)) parityFailures.push(mode+' native parity '+name+' delta '+difference);
        vectors[name]=vector; differences[name]=difference;
      }
      check(delta(vectors.black,vectors.checker)>1e-4,'image content sensitivity');
      check(delta(vectors.silence,vectors.tone)>1e-4,'audio content sensitivity');
      stage(mode+' raw and adjacent text');
      const raw=await bridge.embed(inputs.black,{normalize:false});
      row.rawVectors={black:raw};
      const rawDelta=delta(raw,reference.black.raw);
      row.rawDelta=rawDelta;
      if(rawDelta>=2e-3) parityFailures.push(mode+' raw media parity '+rawDelta);
      const adjacent=await bridge.embed([{type:'text',text:'task: search result | title: Paris | text: '},{type:'text',text:''},{type:'text',text:'The Eiffel Tower is in Paris.'}]);
      check(delta(adjacent,vectors.text)<1e-6,'adjacent text equivalence');
      const textRaw=await bridge.embed(text,{normalize:false});
      row.rawVectors.text=textRaw;
      const textRawDelta=delta(textRaw,reference.text.raw);
      row.textRawDelta=textRawDelta;
      if(textRawDelta>=1e-4) parityFailures.push(mode+' raw text parity '+textRawDelta);
      const batch=await bridge.embedBatch([inputs.black,text]);
      check(delta(batch[0],vectors.black)<1e-6 && delta(batch[1],vectors.text)<1e-6,'batch order and text reuse');
      check(black.bytes.byteLength===224*224*3 && tone.samples.byteLength===64000,'caller buffers detached');
      stage(mode+' cancel and recovery');
      const pending=bridge.embed(inputs.interleaved);
      queueMicrotask(()=>bridge.cancel());
      try {
        const completed=await pending;
        check(delta(completed,vectors.interleaved)<1e-6,'best-effort cancellation result');
        row.cancellation='completed';
      } catch(error) {
        check(error.name==='AbortError','cancel result'); row.cancellation='aborted';
      }
      check(delta(await bridge.embed(text),vectors.text)<1e-6,'text recovery after cancellation');
      await bridge.unloadMultimodalProjector();
      caps=await bridge.getEmbeddingCapabilities();
      check(caps.text && !caps.image && !caps.audio,'unload capability reset');
      stage(mode+' projector reload');
      await bridge.loadMultimodalProjector('/projector.gguf');
      caps=await bridge.getEmbeddingCapabilities();
      check(caps.image && caps.audio,'reload capabilities');
      const metadata=bridge.getModelMetadata();
      const notes=JSON.stringify(metadata);
      check(notes.includes(${JSON.stringify(memoryMode)}),'memory mode fallback');
      check(disableWorker || notes.includes('worker'),'worker fallback');
      row.capabilities=caps; row.metadata=metadata;
    } finally { await bridge.dispose(); }
  }
  const directWorker={};
  for(const name of Object.keys(inputs)) { directWorker[name]=delta(modeResults[0].vectors[name],modeResults[1].vectors[name]); check(directWorker[name]<1e-6,'direct/worker parity'); }
  finish({ok:parityFailures.length===0,parityFailures,memoryMode:${JSON.stringify(memoryMode)},modeResults,directWorker});
} catch(error) { finish({ok:false,stage:window.__smokeStage,error:String(error?.message||error),modeResults}); }
</script>`;
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseSmokeArgs(argv, { prog: 'multimodal_embeddings.mjs', description: 'Local EmbeddingGemma-2 native/direct/worker qualification (no downloads).', options: [
    {flag:'--dist-dir',type:'path',default:()=> 'dist'},
    {flag:'--model-path',type:'path',default:()=>null},
    {flag:'--model-sha256',type:'string',default:()=>''},
    {flag:'--mmproj-path',type:'path',default:()=>null},
    {flag:'--mmproj-sha256',type:'string',default:()=>''},
    {flag:'--reference-path',type:'path',default:()=>null},
    {flag:'--memory-mode',type:'string',default:()=> 'wasm64',choices:['wasm32','wasm64']},
    {flag:'--artifacts-dir',type:'path',default:()=>null},
    {flag:'--timeout-ms',type:'int',default:()=>1800000},
  ]});
  ensure(args.modelPath && args.mmprojPath && args.referencePath && args.artifactsDir, 'Local model, projector, reference and artifacts paths are required.');
  const pinned = async (filePath, expectedSha256, label) => resolvePinnedFile({filePath,expectedSha256,label,cacheDir:args.artifactsDir,url:null});
  const model=await pinned(args.modelPath,args.modelSha256,'embedding model');
  const projector=await pinned(args.mmprojPath,args.mmprojSha256,'embedding projector');
  await fsp.mkdir(args.artifactsDir,{recursive:true});
  const payload=await withTempDir('embedding-browser-',async (root)=>{
    await copyBridgeArtifacts(args.distDir,root);
    if(args.memoryMode==='wasm64') await copyMemory64Artifacts(args.distDir,root);
    await fsp.symlink(model,path.join(root,'model.gguf'));
    await fsp.symlink(projector,path.join(root,'projector.gguf'));
    await fsp.copyFile(args.referencePath,path.join(root,'reference.json'));
    await fsp.writeFile(path.join(root,'index.html'),renderHarness(args.memoryMode));
    return withServer(root,(url)=>runPollingPlaywright(url,args.timeoutMs,args.artifactsDir,{
      artifactPrefix:'multimodal-embeddings',stageMarker:'EMBEDDING_STAGE',stagePrefix:'embedding',
    }));
  });
  process.stdout.write(JSON.stringify(payload,null,2)+'\n');
  return payload.ok===true?0:1;
}
if(import.meta.main) await runMain('multimodal-embeddings', main);
