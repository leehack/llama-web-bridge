#!/usr/bin/env node
// Browser smoke test for bridge state-persistence API wiring.
//
// Ported from state_persistence_browser_smoke.py with the same flags,
// environment variables, harness page and output.

import fsp from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import {
  DEFAULT_MODEL_CACHE,
  copyBridgeArtifacts,
  downloadToCache,
  ensure,
  env,
  expandHome,
  isDirectory,
  isFile,
  isPyInt,
  parseSmokeArgs,
  pyEquals,
  pyGet,
  pyJson,
  resolvePath,
  runMain,
  runPlaywright,
  validateHash,
  withServer,
  withTempDir,
  writeStdout,
} from './browser_smoke_support.mjs';

const DESCRIPTION = `Browser smoke test for bridge state-persistence API wiring.

The default smoke stays lightweight by verifying API shape and clean pre-load
failures in Chromium. When a tiny GGUF model is supplied, the same harness also
loads the model in both direct and worker runtimes, verifies finite embeddings,
evaluates a prompt, saves a state snapshot as bytes, mutates the context, reloads
the snapshot, and verifies restored token metadata. CI supplies an
integrity-pinned tiny model so core API regressions are caught without large
downloads.`;

export const REQUIRED_METHODS = Object.freeze([
  'stateSaveFile',
  'stateLoadFile',
  'stateSaveBytes',
  'stateLoadBytes',
]);
const MODES = Object.freeze(['direct runtime', 'worker runtime']);
const MODEL_FILENAME = 'state-smoke-model.gguf';

export function renderHarness(modelFilename) {
  const methodsJson = pyJson(REQUIRED_METHODS);
  const modelUrlJson = pyJson(modelFilename ? `/${modelFilename}` : null);
  return `
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge state smoke</title>
<pre id="result">pending</pre>
<script type="module">
(async () => {
  const result = document.getElementById('result');
  const finish = (payload) => {
    result.textContent = JSON.stringify(payload);
    window.__smokeResult = payload;
  };
  const assert = (condition, message) => {
    if (!condition) {
      throw new Error(message);
    }
  };
  const arraysEqual = (left, right) => (
    Array.isArray(left) && Array.isArray(right)
      && left.length === right.length
      && left.every((value, index) => value === right[index])
  );
  try {
    if (!window.crossOriginIsolated) {
      throw new Error('test page is not cross-origin isolated');
    }
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    if (typeof LlamaWebGpuBridge !== 'function') {
      throw new Error('LlamaWebGpuBridge export was not registered');
    }
    const methods = ${methodsJson};
    const modelUrl = ${modelUrlJson};
    const modeResults = [];

    const verifyPreloadStateApi = async (bridge, mode) => {
      const missing = methods.filter((name) => typeof bridge[name] !== 'function');
      assert(missing.length === 0, \`\${mode} missing public state methods: \${missing.join(', ')}\`);

      let saveRejected = false;
      try {
        await bridge.stateSaveBytes([]);
      } catch (error) {
        saveRejected = String(error && error.message ? error.message : error).includes('No model loaded');
      }
      assert(saveRejected, \`\${mode} stateSaveBytes did not reject cleanly before model load\`);

      let loadRejected = false;
      try {
        await bridge.stateLoadBytes(new Uint8Array([0]), 1);
      } catch (error) {
        loadRejected = String(error && error.message ? error.message : error).includes('No model loaded');
      }
      assert(loadRejected, \`\${mode} stateLoadBytes did not reject cleanly before model load\`);
    };

    const verifyModelRoundTrip = async (bridge, mode) => {
      const progress = [];
      await bridge.loadModelFromUrl(modelUrl, {
        nCtx: 64,
        nThreads: 1,
        nGpuLayers: 0,
        nBatch: 32,
        nUbatch: 32,
        useCache: false,
        forceRemoteFetchBackend: false,
        progressCallback: (event) => progress.push(event || {}),
      });

      const prompt = 'Hello';
      const mutationPrompt = 'Completely different prompt';
      const tokens = await bridge.tokenize(prompt, true);
      assert(Array.isArray(tokens) && tokens.length > 0, \`\${mode} tokenization returned no tokens\`);

      const embedding = await bridge.embed(prompt);
      assert(Array.isArray(embedding) && embedding.length > 0, \`\${mode} embedding returned no components\`);
      assert(embedding.every(Number.isFinite), \`\${mode} embedding returned non-finite components\`);
      assert(embedding.some((value) => value !== 0), \`\${mode} embedding was rounded entirely to zero\`);

      const embeddingBatch = await bridge.embedBatch([prompt, mutationPrompt]);
      assert(Array.isArray(embeddingBatch) && embeddingBatch.length === 2, \`\${mode} embedding batch shape was invalid\`);
      assert(embeddingBatch.every((values) => (
        Array.isArray(values)
          && values.length === embedding.length
          && values.every(Number.isFinite)
      )), \`\${mode} embedding batch contained invalid vectors\`);
      assert(arraysEqual(embeddingBatch[0], embedding), \`\${mode} batch embedding differed from the single embedding\`);

      const firstText = await bridge.createCompletion(prompt, {
        nPredict: 1,
        temp: 0,
        topK: 1,
        topP: 1,
        seed: 1,
        tokenEventEncoding: 'text',
      });
      assert(typeof firstText === 'string', \`\${mode} initial completion did not return text\`);

      const snapshot = await bridge.stateSaveBytes(tokens);
      assert(snapshot instanceof Uint8Array, \`\${mode} stateSaveBytes did not return Uint8Array\`);
      assert(snapshot.byteLength > 0, \`\${mode} stateSaveBytes returned empty snapshot\`);

      await bridge.createCompletion(mutationPrompt, {
        nPredict: 1,
        temp: 0,
        topK: 1,
        topP: 1,
        seed: 2,
        tokenEventEncoding: 'text',
      });

      let restored;
      let detachedAfterLoadTransfer = null;
      if (mode === 'worker runtime' && bridge._workerProxy && typeof bridge._callWorker === 'function') {
        const transferableBytes = snapshot.slice();
        restored = await bridge._callWorker(
          'stateLoadBytes',
          [transferableBytes, bridge.getContextSize()],
          null,
          [transferableBytes.buffer],
        );
        detachedAfterLoadTransfer = transferableBytes.buffer.byteLength === 0;
      } else {
        restored = await bridge.stateLoadBytes(snapshot.slice(), bridge.getContextSize());
      }

      assert(restored && arraysEqual(restored.tokens, tokens), \`\${mode} restored tokens did not match saved prompt tokens\`);

      const afterRestoreText = await bridge.createCompletion(prompt, {
        nPredict: 1,
        temp: 0,
        topK: 1,
        topP: 1,
        seed: 3,
        tokenEventEncoding: 'text',
      });
      assert(typeof afterRestoreText === 'string', \`\${mode} completion after state restore did not return text\`);

      // A second load replaces the model in place. WASMFS analyzePath('/models')
      // used to throw a bare "FS error" here, which restarted the worker.
      const workerBeforeReload = bridge._workerProxy;
      await bridge.loadModelFromUrl(modelUrl, {
        nCtx: 64,
        nThreads: 1,
        nGpuLayers: 0,
        nBatch: 32,
        nUbatch: 32,
        useCache: false,
        forceRemoteFetchBackend: false,
      });
      assert(bridge._workerProxy === workerBeforeReload, \`\${mode} model reload replaced the runtime\`);
      const reloadedTokens = await bridge.tokenize(prompt, true);
      assert(arraysEqual(reloadedTokens, tokens), \`\${mode} tokenization changed after model reload\`);

      const workerSaveSnapshotReturned = mode === 'worker runtime'
        ? snapshot instanceof Uint8Array && snapshot.byteLength > 0
        : null;
      return {
        progressEvents: progress.length,
        embedding,
        embeddingComponents: embedding.length,
        embeddingBatchSize: embeddingBatch.length,
        savedBytes: snapshot.byteLength,
        restoredTokens: restored.tokens.length,
        detachedAfterLoadTransfer,
        workerSaveSnapshotReturned,
      };
    };

    const verifyBridgeStateApi = async (bridge, mode) => {
      try {
        await verifyPreloadStateApi(bridge, mode);
        const modeResult = { mode, preload: true };
        if (modelUrl) {
          Object.assign(modeResult, await verifyModelRoundTrip(bridge, mode));
        }
        modeResults.push(modeResult);
      } finally {
        await bridge.dispose();
      }
    };

    await verifyBridgeStateApi(new LlamaWebGpuBridge({ disableWorker: true }), 'direct runtime');
    await verifyBridgeStateApi(new LlamaWebGpuBridge({ disableWorker: false }), 'worker runtime');

    if (modelUrl) {
      const direct = modeResults.find((entry) => entry.mode === 'direct runtime');
      const worker = modeResults.find((entry) => entry.mode === 'worker runtime');
      assert(direct && worker && arraysEqual(direct.embedding, worker.embedding), 'direct and worker embeddings differed');
      delete direct.embedding;
      delete worker.embedding;
      assert(worker && worker.detachedAfterLoadTransfer === true, 'worker stateLoadBytes transfer did not detach transferred buffer');
      assert(worker && worker.workerSaveSnapshotReturned === true, 'worker stateSaveBytes did not return a byte snapshot');
    }

    finish({ ok: true, methods, modes: ['direct runtime', 'worker runtime'], modelBacked: Boolean(modelUrl), modeResults });
  } catch (error) {
    finish({ ok: false, error: String(error && error.stack ? error.stack : error) });
  }
})();
</script>
`;
}

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'state_persistence_browser_smoke.mjs',
    description: DESCRIPTION,
    options: [
      {
        flag: '--dist-dir',
        type: 'path',
        default: () => env.path('BRIDGE_DIST_DIR', 'dist'),
        help: 'Directory containing built bridge artifacts.',
      },
      {
        flag: '--timeout-ms',
        type: 'int',
        default: () => env.int('LLAMA_WEBGPU_SMOKE_TIMEOUT_MS', '120000'),
        help: 'Browser operation timeout in milliseconds.',
      },
      {
        flag: '--model-url',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_URL'),
        help: 'Optional tiny GGUF URL for model-backed state round-trip smoke.',
      },
      {
        flag: '--model-path',
        type: 'path',
        default: () => env.optionalPath('LLAMA_WEBGPU_SMOKE_MODEL_PATH'),
        help: 'Optional local tiny GGUF path for model-backed state round-trip smoke.',
      },
      {
        flag: '--model-sha256',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_SHA256'),
        help: 'Expected SHA-256 for the optional smoke model.',
      },
      {
        flag: '--model-cache-dir',
        type: 'path',
        default: () => env.path('LLAMA_WEBGPU_SMOKE_MODEL_CACHE', DEFAULT_MODEL_CACHE),
        help: 'Cache directory used when --model-url is supplied.',
      },
      {
        flag: '--artifacts-dir',
        type: 'path',
        default: () => env.optionalPath('LLAMA_WEBGPU_SMOKE_ARTIFACTS_DIR'),
        help: 'Directory for JSON/console/screenshot diagnostics.',
      },
    ],
  });
}

// The model is optional here: without one the smoke checks the API only. Unlike
// the other smokes, --model-path is not `~`-expanded (as in the Python smoke).
async function resolveModel(args) {
  if (args.modelPath !== null) {
    const modelPath = resolvePath(args.modelPath);
    ensure(isFile(modelPath), `model path does not exist: ${modelPath}`);
    await validateHash(modelPath, args.modelSha256);
    return modelPath;
  }
  if (args.modelUrl) {
    return downloadToCache(args.modelUrl, resolvePath(expandHome(args.modelCacheDir)), args.modelSha256);
  }
  return null;
}

// Checks that run after the payload is printed.
export function checkPayload(payload, modelBacked) {
  ensure(pyEquals(pyGet(payload, 'methods'), REQUIRED_METHODS), 'smoke methods payload mismatch');
  ensure(pyEquals(pyGet(payload, 'modes'), MODES), 'smoke modes payload mismatch');
  if (!modelBacked) return;
  ensure(pyGet(payload, 'modelBacked') === true, 'model-backed smoke did not run');
  const modeResults = pyGet(payload, 'modeResults');
  ensure(Array.isArray(modeResults) && modeResults.length === 2, 'model-backed mode results missing');
  // isinstance(value, int) accepts a bool, as the Python checks did.
  const positiveInt = (value) => (typeof value === 'boolean' || isPyInt(value)) && Number(value) > 0;
  for (const entry of modeResults) {
    ensure(positiveInt(pyGet(entry, 'embeddingComponents')), 'embedding vector was empty');
    ensure(pyEquals(pyGet(entry, 'embeddingBatchSize'), 2), 'embedding batch result was incomplete');
    ensure(positiveInt(pyGet(entry, 'savedBytes')), 'state snapshot was empty');
    ensure(positiveInt(pyGet(entry, 'restoredTokens')), 'restored token metadata missing');
  }
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const distDir = resolvePath(args.distDir);
  ensure(isDirectory(distDir), `dist directory does not exist: ${distDir}`);
  const artifactsDir = args.artifactsDir !== null ? resolvePath(args.artifactsDir) : null;

  const modelPath = await resolveModel(args);
  const payload = await withTempDir('llama-web-bridge-smoke-', async (webRoot) => {
    await copyBridgeArtifacts(distDir, webRoot);
    let modelFilename = null;
    if (modelPath !== null) {
      await fsp.copyFile(modelPath, path.join(webRoot, MODEL_FILENAME));
      modelFilename = MODEL_FILENAME;
    }
    await fsp.writeFile(path.join(webRoot, 'index.html'), renderHarness(modelFilename), 'utf8');
    if (artifactsDir !== null) {
      await fsp.mkdir(artifactsDir, { recursive: true });
      await fsp.copyFile(path.join(webRoot, 'index.html'), path.join(artifactsDir, 'index.html'));
    }
    return withServer(webRoot, (url) => runPlaywright(url, args.timeoutMs, artifactsDir));
  });

  writeStdout(`${pyJson(payload, { indent: 2, sortKeys: true })}\n`);
  if (payload.ok !== true) return 1;
  checkPayload(payload, modelPath !== null);
  return 0;
}

if (import.meta.main) {
  await runMain('state persistence', main);
}
