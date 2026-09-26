#!/usr/bin/env node
// Browser smoke for runtime LoRA adapters.

import fsp from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import {
  copyBridgeArtifacts,
  copyMemory64Artifacts,
  ensure,
  env,
  expectedModes,
  isDict,
  isDirectory,
  modeRuntimeFailures,
  parseSmokeArgs,
  pyJson,
  pyRepr,
  resolvePath,
  resolvePinnedFile,
  runMain,
  runPlaywright,
  runPollingPlaywright,
  stageFile,
  withServer,
  withTempDir,
  writeStdout,
} from './browser_smoke_support.mjs';

const DESCRIPTION = `Browser smoke for runtime LoRA adapters.

Runs a checksum-pinned base GGUF and a LoRA adapter trained on it through the
direct and worker runtimes of the wasm32 and wasm64 cores. With a fixed seed
and greedy sampling, the adapter must change the completion and next-token
scores, scale 0 must match no adapter, an adapter loaded from bytes must match
the same adapter loaded from its URL, stacking two adapters must change the
scores again, and removing or clearing adapters must restore the base output.
A URL load must report progress and store the adapter in the Cache API. An
aLoRA adapter, a malformed file and an adapter for another base model must
reject with normal errors and leave the runtime usable. A failed download must
not expose its query string, and reloading the model must drop its adapters.

A nonzero --gpu-layers enables WebGPU in Chromium and skips the model reload:
with WebGPU enabled, reloading a model on one bridge aborts in headless
Chromium whether or not LoRA adapters were used.`;

// ggml-org/stories15M_MOE at a pinned revision (MIT): the base model and the
// Shakespeare LoRA adapter llama.cpp's server LoRA tests use.
const STORIES_REVISION = 'b6dd737497465570b5f5e962dbc9d9454ed1e0eb';
const STORIES_BASE_URL = `https://huggingface.co/ggml-org/stories15M_MOE/resolve/${STORIES_REVISION}`;
export const DEFAULT_MODEL_URL = `${STORIES_BASE_URL}/stories15M_MOE-Q8_0.gguf`;
export const DEFAULT_MODEL_SHA256 = 'c7aa6863f9a4b3cdf19716e2c95622dcbd3bd06989324bf1ac8e60486ef8e881';
export const DEFAULT_ADAPTER_URL = `${STORIES_BASE_URL}/moe_shakespeare15M.gguf`;
export const DEFAULT_ADAPTER_SHA256 = 'd1e0617d7e10de960639d18a4620ec8c6bb56343f45692830d3634a1a3e1fe1a';
const DEFAULT_MODEL_CACHE = '~/.cache/llama-web-bridge/lora-smoke-models';

const MODEL_FILENAME = 'lora-smoke-model.gguf';
const ADAPTER_FILENAME = 'lora-smoke-adapter.gguf';
const ALORA_FILENAME = 'lora-smoke-alora.gguf';
const MISMATCH_MODEL_FILENAME = 'lora-smoke-mismatch-model.gguf';
const PROMPT = 'Once upon a time';
const TOP_K = 8;
const SECRET = 'lora-smoke-secret';
export const MEMORY_MODES = Object.freeze(['wasm32', 'wasm64']);
export const RUNTIME_MODES = Object.freeze(['direct', 'worker']);

const GGUF_TYPE_UINT32 = 4;
const GGUF_TYPE_STRING = 8;
const GGUF_TYPE_ARRAY = 9;
const GGUF_SCALAR_SIZES = Object.freeze({ 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 });
const GGUF_DEFAULT_ALIGNMENT = 32;

const alignUp = (value, alignment) => Math.ceil(value / alignment) * alignment;

// The GGUF v3 file `data` with an `adapter.alora.invocation_tokens` array of
// `tokens` appended to its metadata.
export function withAloraInvocationTokens(data, tokens) {
  ensure(data.subarray(0, 4).toString('latin1') === 'GGUF', 'adapter is not a GGUF file');
  const version = data.readUInt32LE(4);
  ensure(version === 3, `unsupported GGUF version ${version}`);
  const nTensors = Number(data.readBigUInt64LE(8));
  const nKv = Number(data.readBigUInt64LE(16));
  const readString = (at) => {
    const length = Number(data.readBigUInt64LE(at));
    return [data.subarray(at + 8, at + 8 + length).toString('utf8'), at + 8 + length];
  };
  const skipValue = (type, at) => {
    if (type === GGUF_TYPE_STRING) return readString(at)[1];
    if (type === GGUF_TYPE_ARRAY) {
      const elementType = data.readUInt32LE(at);
      const count = Number(data.readBigUInt64LE(at + 4));
      let next = at + 12;
      for (let i = 0; i < count; i += 1) next = skipValue(elementType, next);
      return next;
    }
    ensure(Object.hasOwn(GGUF_SCALAR_SIZES, type), `unknown GGUF value type ${type}`);
    return at + GGUF_SCALAR_SIZES[type];
  };

  let offset = 24;
  let alignment = GGUF_DEFAULT_ALIGNMENT;
  for (let i = 0; i < nKv; i += 1) {
    let key;
    [key, offset] = readString(offset);
    const type = data.readUInt32LE(offset);
    offset += 4;
    if (key === 'general.alignment') alignment = data.readUInt32LE(offset);
    offset = skipValue(type, offset);
  }
  const kvEnd = offset;
  for (let i = 0; i < nTensors; i += 1) {
    offset = readString(offset)[1];
    const nDims = data.readUInt32LE(offset);
    offset += 4 + 8 * nDims + 4 + 8;
  }
  const tensorInfos = data.subarray(kvEnd, offset);
  const dataStart = alignUp(offset, alignment);

  const key = Buffer.from('adapter.alora.invocation_tokens', 'utf8');
  const added = Buffer.alloc(8 + key.length + 16 + 4 * tokens.length);
  let at = added.writeBigUInt64LE(BigInt(key.length), 0);
  at += key.copy(added, at);
  at = added.writeUInt32LE(GGUF_TYPE_ARRAY, at);
  at = added.writeUInt32LE(GGUF_TYPE_UINT32, at);
  at = added.writeBigUInt64LE(BigInt(tokens.length), at);
  for (const token of tokens) at = added.writeUInt32LE(token, at);

  const count = Buffer.alloc(8);
  count.writeBigUInt64LE(BigInt(nKv + 1));
  const header = Buffer.concat([data.subarray(0, 16), count, data.subarray(24, kvEnd), added, tensorInfos]);
  const padding = Buffer.alloc(alignUp(header.length, alignment) - header.length);
  return Buffer.concat([header, padding, data.subarray(dataStart)]);
}

export function renderHarness(config) {
  return `
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge LoRA adapter smoke</title>
<pre id="result">pending</pre>
<script type="module">
(async () => {
  const result = document.getElementById('result');
  const finish = (payload) => {
    result.textContent = JSON.stringify(payload);
    window.__smokeResult = payload;
  };
  const errorText = (error) => String(error && error.message ? error.message : error);
  try {
    if (!window.crossOriginIsolated) {
      throw new Error('test page is not cross-origin isolated');
    }
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    if (typeof LlamaWebGpuBridge !== 'function') {
      throw new Error('LlamaWebGpuBridge export was not registered');
    }
    const config = ${pyJson(config)};
    const adapterBytes = new Uint8Array(await (await fetch(config.adapterUrl)).arrayBuffer());
    const aloraBytes = new Uint8Array(await (await fetch(config.aloraUrl)).arrayBuffer());
    const modeResults = [];

    const runMode = async (memoryMode, runtimeMode) => {
      const mode = \`\${memoryMode} \${runtimeMode}\`;
      window.__smokeStage = mode;
      console.log(\`lora-smoke-stage:\${mode}\`);
      const useMemory64 = memoryMode === 'wasm64';
      const cacheName = \`lora-smoke-\${memoryMode}-\${runtimeMode}-\${Date.now()}\`;
      const bridgeConfig = {
        disableWorker: runtimeMode === 'direct',
        preferMemory64: useMemory64,
        coreModuleUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.js' : undefined,
        wasmUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.wasm' : undefined,
        cacheName,
      };
      const bridge = new LlamaWebGpuBridge(bridgeConfig);
      const failures = [];
      const errors = {};
      const check = (condition, message) => {
        if (!condition) {
          failures.push(message);
        }
      };
      const rejection = async (name, promise, pattern) => {
        let error = null;
        try {
          await promise;
        } catch (caught) {
          error = caught;
        }
        const text = error === null ? null : \`\${error.name}: \${errorText(error)}\`;
        errors[name] = text;
        check(text !== null && pattern.test(text), \`\${name}: expected \${pattern}, got \${text}\`);
        return text;
      };
      const loadOptions = {
        nCtx: config.nCtx,
        nThreads: 2,
        nGpuLayers: config.gpuLayers,
        useCache: false,
        forceRemoteFetchBackend: false,
      };
      const generate = () => bridge.createCompletion(config.prompt, {
        nPredict: config.nPredict,
        temp: 0,
        topK: 1,
        seed: 7,
        tokenEventEncoding: 'text',
      });
      const scores = async () => {
        const scored = await bridge.scoreNextToken(config.prompt, { topK: config.topK });
        return JSON.stringify(scored.top.map((entry) => [entry.token, entry.logprob]));
      };
      const output = async () => ({ text: await generate(), scores: await scores() });
      const same = (label, actual, expected) => {
        check(actual.text === expected.text, \`\${label}: text \${JSON.stringify(actual.text)} != \${JSON.stringify(expected.text)}\`);
        check(actual.scores === expected.scores, \`\${label}: scores \${actual.scores} != \${expected.scores}\`);
      };
      try {
        const beforeLoad = await bridge.getLoraAdapterCapabilities();
        check(beforeLoad.supported === false, 'capabilities must be unsupported before the core starts');
        await bridge.loadModelFromUrl(config.modelUrl, loadOptions);
        const capabilities = await bridge.getLoraAdapterCapabilities();
        check(
          capabilities.supported === true && capabilities.apiVersion === 1,
          \`capabilities: \${JSON.stringify(capabilities)}\`,
        );

        const base = await output();

        const aborted = new AbortController();
        aborted.abort();
        await rejection('pre-aborted load', bridge.loadLoraAdapter(config.adapterUrl, { signal: aborted.signal }), /^AbortError: /);
        const missing = await rejection(
          'missing adapter',
          bridge.loadLoraAdapter(\`/missing-adapter.gguf?token=\${config.secret}#\${config.secret}\`),
          /Failed to fetch LoRA adapter: 404/,
        );
        check(missing === null || !missing.includes(config.secret), 'a failed download exposed its query string');

        const progress = [];
        const adapter = await bridge.loadLoraAdapter(config.adapterUrl, {
          progressCallback: (event) => progress.push(event),
        });
        const lastProgress = progress.at(-1) || {};
        check(
          lastProgress.loaded === config.adapterBytes && lastProgress.total === config.adapterBytes,
          \`final progress \${JSON.stringify(lastProgress)}\`,
        );
        const cache = await caches.open(cacheName);
        check(
          (await cache.match(new URL(config.adapterUrl, location.href).href)) !== undefined,
          'the URL adapter was not stored in the Cache API',
        );

        await bridge.setLoraAdapter(adapter.handle, 1);
        const lora = await output();
        check(lora.text !== base.text, 'the adapter did not change the completion');
        check(lora.scores !== base.scores, 'the adapter did not change the scores');
        await bridge.setLoraAdapter(adapter.handle, 0);
        same('scale 0', await output(), base);
        await bridge.setLoraAdapter(adapter.handle);
        same('default scale', await output(), lora);
        await bridge.removeLoraAdapter(adapter.handle);
        same('removed', await output(), base);
        await bridge.removeLoraAdapter(adapter.handle);
        same('removed twice', await output(), base);

        const callerBytes = adapterBytes.slice();
        const fromBytes = await bridge.loadLoraAdapter(callerBytes);
        check(callerBytes.byteLength === config.adapterBytes, "loading bytes detached the caller's buffer");
        check(fromBytes.handle !== adapter.handle, 'handles must differ');
        await bridge.setLoraAdapter(fromBytes.handle, 1);
        same('bytes adapter', await output(), lora);
        await bridge.setLoraAdapter(adapter.handle, 1);
        const stacked = await output();
        check(stacked.scores !== lora.scores, 'a second adapter did not change the scores');
        await bridge.clearLoraAdapters();
        same('cleared', await output(), base);

        await rejection('aLoRA adapter', bridge.loadLoraAdapter(aloraBytes), /Failed to load LoRA adapter: the adapter is an aLoRA adapter \\(3 invocation token\\(s\\)\\)/);
        await rejection('malformed adapter', bridge.loadLoraAdapter(new Uint8Array([1, 2, 3, 4])), /Failed to load LoRA adapter: /);
        await rejection('empty bytes', bridge.loadLoraAdapter(new Uint8Array()), /LoRA adapter bytes are empty/);
        await rejection('zero handle', bridge.setLoraAdapter(0), /^TypeError: LoRA adapter handle must be a positive integer/);
        await rejection('NaN scale', bridge.setLoraAdapter(adapter.handle, Number.NaN), /^TypeError: LoRA adapter scale must be a finite number/);
        await rejection('unknown handle', bridge.setLoraAdapter(9999), /LoRA adapter 9999 is not loaded/);
        same('after rejected loads', await output(), base);
        await bridge.setLoraAdapter(adapter.handle, 1);
        same('after rejected loads with the adapter', await output(), lora);

        if (config.reloadModel) {
          await bridge.loadModelFromUrl(config.modelUrl, loadOptions);
          await rejection('handle after reload', bridge.setLoraAdapter(adapter.handle), /LoRA adapter \\d+ is not loaded; its model was unloaded or replaced/);
          same('model reloaded', await output(), base);
        }
        const metadata = bridge.getModelMetadata();

        const other = new LlamaWebGpuBridge(bridgeConfig);
        try {
          await other.loadModelFromUrl(config.mismatchModelUrl, loadOptions);
          await rejection('other base model', other.loadLoraAdapter(config.adapterUrl), /Failed to load LoRA adapter: /);
          const text = await other.createCompletion(config.prompt, { nPredict: 4, temp: 0, topK: 1, seed: 7 });
          check(typeof text === 'string', 'the runtime is unusable after an adapter for another model');
        } finally {
          await other.dispose();
        }

        return {
          mode,
          failures,
          errors,
          base: base.text,
          lora: lora.text,
          progressEvents: progress.length,
          gpuActive: bridge.isGpuActive(),
          backendName: bridge.getBackendName(),
          execution: metadata['llamadart.webgpu.execution'] || null,
          coreVariant: metadata['llamadart.webgpu.core_variant'] || null,
          workerFallbackReason: metadata['llamadart.webgpu.worker_fallback_reason'] || null,
        };
      } catch (error) {
        return { mode, failures: [...failures, \`threw: \${errorText(error)}\`], errors };
      } finally {
        await bridge.dispose();
        await caches.delete(cacheName);
      }
    };

    for (const memoryMode of config.memoryModes) {
      for (const runtimeMode of config.runtimeModes) {
        modeResults.push(await runMode(memoryMode, runtimeMode));
      }
    }
    finish({
      ok: true,
      modeResults,
      globalWorkerFallbackReason: globalThis.__llamadartBridgeWorkerFallbackReason || null,
    });
  } catch (error) {
    finish({ ok: false, error: String(error && error.stack ? error.stack : error) });
  }
})();
</script>
`;
}

// The failures of a harness payload: every mode must have run on its own core
// variant and thread with no failed check, and GPU layers must match
// `gpuLayers`.
export function validatePayload(payload, memoryModes, gpuLayers) {
  if (payload.ok !== true) return [`harness failed: ${payload.error}`];
  const modeResults = payload.modeResults;
  const expected = expectedModes(memoryModes, RUNTIME_MODES);
  if (
    !Array.isArray(modeResults)
    || JSON.stringify(modeResults.filter(isDict).map((entry) => entry.mode)) !== JSON.stringify(expected)
    || modeResults.length !== expected.length
  ) {
    return ['mode results missing'];
  }
  const failures = [];
  for (const entry of modeResults) {
    const { mode } = entry;
    if (!Array.isArray(entry.failures)) {
      failures.push(`${mode}: failures missing`);
      continue;
    }
    failures.push(...entry.failures.map((failure) => `${mode}: ${failure}`));
    if (entry.failures.length) continue;
    failures.push(...modeRuntimeFailures(mode, entry));
    if (entry.gpuActive !== (gpuLayers !== 0)) {
      failures.push(`${mode}: GPU active is ${pyRepr(entry.gpuActive ?? null)} with ${gpuLayers} GPU layers`);
    }
  }
  if (payload.globalWorkerFallbackReason != null) {
    failures.push(`worker fell back to the main thread: ${pyRepr(payload.globalWorkerFallbackReason)}`);
  }
  return failures;
}

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'lora_adapter_browser_smoke.mjs',
    description: DESCRIPTION,
    options: [
      { flag: '--dist-dir', type: 'path', default: () => env.path('BRIDGE_DIST_DIR', 'dist') },
      { flag: '--timeout-ms', type: 'int', default: () => 600000, help: 'Browser timeout in milliseconds.' },
      { flag: '--model-url', type: 'string', default: () => DEFAULT_MODEL_URL, help: 'Base GGUF URL.' },
      { flag: '--model-path', type: 'path', help: 'Local base GGUF path.' },
      { flag: '--model-sha256', type: 'string', default: () => DEFAULT_MODEL_SHA256 },
      { flag: '--adapter-url', type: 'string', default: () => DEFAULT_ADAPTER_URL, help: 'LoRA adapter GGUF URL.' },
      { flag: '--adapter-path', type: 'path', help: 'Local LoRA adapter GGUF path.' },
      { flag: '--adapter-sha256', type: 'string', default: () => DEFAULT_ADAPTER_SHA256 },
      {
        flag: '--mismatch-model-url',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_URL'),
        help: 'GGUF the adapter was not trained for; defaults to the state-persistence smoke model.',
      },
      { flag: '--mismatch-model-path', type: 'path', default: () => env.optionalPath('LLAMA_WEBGPU_SMOKE_MODEL_PATH') },
      { flag: '--mismatch-model-sha256', type: 'string', default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_SHA256') },
      { flag: '--model-cache-dir', type: 'path', default: () => DEFAULT_MODEL_CACHE },
      { flag: '--memory-mode', type: 'string', choices: ['all', ...MEMORY_MODES], default: () => 'all' },
      { flag: '--gpu-layers', type: 'int', default: () => 0, help: 'nGpuLayers; nonzero requires WebGPU.' },
      { flag: '--n-ctx', type: 'int', default: () => 512 },
      { flag: '--n-predict', type: 'int', default: () => 32 },
      { flag: '--artifacts-dir', type: 'path', help: 'Directory for JSON/console/screenshot diagnostics.' },
    ],
  });
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const distDir = resolvePath(args.distDir);
  ensure(isDirectory(distDir), `dist directory does not exist: ${distDir}`);
  const artifactsDir = args.artifactsDir !== null ? resolvePath(args.artifactsDir) : null;
  const memoryModes = args.memoryMode === 'all' ? MEMORY_MODES : [args.memoryMode];
  const pinned = (label, filePath, url, expectedSha256) => resolvePinnedFile({
    filePath, url, expectedSha256, cacheDir: args.modelCacheDir, label,
  });
  const modelPath = await pinned('model', args.modelPath, args.modelUrl, args.modelSha256);
  const adapterPath = await pinned('adapter', args.adapterPath, args.adapterUrl, args.adapterSha256);
  const mismatchPath = await pinned(
    'mismatch model', args.mismatchModelPath, args.mismatchModelUrl, args.mismatchModelSha256,
  );
  const adapter = await fsp.readFile(adapterPath);

  const payload = await withTempDir('llama-web-bridge-lora-smoke-', async (webRoot) => {
    await copyBridgeArtifacts(distDir, webRoot);
    if (memoryModes.includes('wasm64')) await copyMemory64Artifacts(distDir, webRoot);
    await stageFile(modelPath, path.join(webRoot, MODEL_FILENAME));
    await stageFile(adapterPath, path.join(webRoot, ADAPTER_FILENAME));
    await stageFile(mismatchPath, path.join(webRoot, MISMATCH_MODEL_FILENAME));
    await fsp.writeFile(path.join(webRoot, ALORA_FILENAME), withAloraInvocationTokens(adapter, [1, 2, 3]));
    await fsp.writeFile(path.join(webRoot, 'index.html'), renderHarness({
      modelUrl: `/${MODEL_FILENAME}`,
      adapterUrl: `/${ADAPTER_FILENAME}`,
      aloraUrl: `/${ALORA_FILENAME}`,
      mismatchModelUrl: `/${MISMATCH_MODEL_FILENAME}`,
      adapterBytes: adapter.length,
      prompt: PROMPT,
      topK: TOP_K,
      secret: SECRET,
      nCtx: args.nCtx,
      nPredict: args.nPredict,
      gpuLayers: args.gpuLayers,
      reloadModel: args.gpuLayers === 0,
      memoryModes,
      runtimeModes: RUNTIME_MODES,
    }), 'utf8');
    if (artifactsDir !== null) {
      await fsp.mkdir(artifactsDir, { recursive: true });
      await fsp.copyFile(path.join(webRoot, 'index.html'), path.join(artifactsDir, 'index.html'));
    }
    return withServer(webRoot, (url) => (args.gpuLayers === 0
      ? runPlaywright(url, args.timeoutMs, artifactsDir, 'lora-smoke')
      : runPollingPlaywright(url, args.timeoutMs, artifactsDir, {
        artifactPrefix: 'lora-smoke',
        stageMarker: 'lora-smoke-stage:',
      })));
  });

  writeStdout(`${pyJson(payload, { indent: 2, sortKeys: true })}\n`);
  const failures = validatePayload(payload, memoryModes, args.gpuLayers);
  for (const failure of failures) process.stderr.write(`LoRA adapter browser smoke: ${failure}\n`);
  return failures.length ? 1 : 0;
}

if (import.meta.main) {
  await runMain('LoRA adapter', main);
}
