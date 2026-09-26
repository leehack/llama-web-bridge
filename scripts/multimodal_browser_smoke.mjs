#!/usr/bin/env node
// Real-model browser smoke for multimodal bridge prompt ingestion.
//
// Ported from multimodal_browser_smoke.py with the same flags, environment
// variables, harness page and output.

import fsp from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import {
  copyBridgeArtifacts,
  ensure,
  env,
  expandHome,
  isDirectory,
  parseSmokeArgs,
  pyEquals,
  pyGet,
  pyJson,
  resolvePath,
  resolvePinnedFile,
  runMain,
  runPlaywright,
  withServer,
  withTempDir,
  writeStdout,
} from './browser_smoke_support.mjs';

const DESCRIPTION = 'Real-model browser smoke for multimodal bridge prompt ingestion.';
const DEFAULT_MODEL_CACHE = '~/.cache/llama-web-bridge/multimodal-smoke-models';
const MODES = Object.freeze(['direct runtime', 'worker runtime']);

export function renderHarness() {
  return `
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge multimodal smoke</title>
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
  const createImageBytes = async () => {
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 180;
    const context = canvas.getContext('2d');
    context.fillStyle = '#0f172a';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#22d3ee';
    context.fillRect(16, 16, 288, 148);
    context.fillStyle = '#111827';
    context.font = 'bold 42px sans-serif';
    context.fillText('HELLO', 80, 108);
    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob((value) => {
        if (value) {
          resolve(value);
        } else {
          reject(new Error('synthetic image encoding failed'));
        }
      }, 'image/png');
    });
    return new Uint8Array(await blob.arrayBuffer());
  };

  try {
    assert(window.crossOriginIsolated, 'test page is not cross-origin isolated');
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge =
      module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    assert(
      typeof LlamaWebGpuBridge === 'function',
      'LlamaWebGpuBridge export was not registered',
    );

    const imageBytes = await createImageBytes();
    const modeResults = [];
    const verifyMode = async (disableWorker, mode) => {
      const bridge = new LlamaWebGpuBridge({ disableWorker, logLevel: 1 });
      const startedAt = performance.now();
      try {
        await bridge.loadModelFromUrl('/multimodal-model.gguf', {
          nCtx: 4096,
          nGpuLayers: 0,
          nThreads: 4,
          nBatch: 512,
          nUbatch: 256,
          useCache: false,
          forceRemoteFetchBackend: false,
        });
        await bridge.loadMultimodalProjector('/multimodal-mmproj.gguf');
        assert(bridge.supportsVision(), \`\${mode} did not report vision support\`);

        // Normal image ingestion resizes through RGB. Also exercise the native
        // file/encoded entry points so helper API compatibility cannot hide
        // behind that independent path. The direct runtime owns this FS.
        if (disableWorker) {
          const core = bridge._runtime?._core;
          assert(core?.FS && typeof core.ccall === 'function',
            'direct runtime core is unavailable for media compatibility checks');
          const path = '/mtmd-compat.png';
          core.FS.writeFile(path, imageBytes);
          try {
            assert(
              core.ccall('llamadart_webgpu_media_add_file', 'number',
                ['string'], [path]) === 0,
              'native file helper failed to decode the real PNG',
            );
            core.ccall('llamadart_webgpu_media_clear_pending', null, [], []);
            assert(
              core.ccall('llamadart_webgpu_media_add_encoded', 'number',
                ['array', 'number'], [imageBytes, imageBytes.length]) === 0,
              'native buffer helper failed to decode the real PNG',
            );
            core.ccall('llamadart_webgpu_media_clear_pending', null, [], []);
            assert(
              core.ccall('llamadart_webgpu_media_add_encoded', 'number',
                ['array', 'number'], [imageBytes, 0]) === -3,
              'native buffer helper must reject empty input',
            );
            assert(
              core.ccall('llamadart_webgpu_media_add_encoded', 'number',
                ['array', 'number'], [imageBytes.subarray(0, 8), 8]) === -4,
              'native buffer helper must reject a truncated PNG',
            );
            assert(
              core.ccall('llamadart_webgpu_media_add_file', 'number',
                ['string'], ['/mtmd-compat-missing.png']) === -4,
              'native file helper must reject a missing file',
            );
          } finally {
            core.ccall('llamadart_webgpu_media_clear_pending', null, [], []);
            core.FS.unlink(path);
          }
        }

        const usages = [];
        const promptTextTokens = (await bridge.tokenize('what do you see?', true)).length;
        const output = await bridge.createCompletion('what do you see?', {
          onUsage: (usage) => usages.push(usage),
          nPredict: 64,
          temp: 0,
          topK: 1,
          topP: 1,
          seed: 42,
          tokenEventEncoding: 'text',
          parts: [{ type: 'image', bytes: imageBytes }],
          mediaMaxImagePixels: 50000,
          mediaMaxImageEdge: 300,
        });
        const outputText = String(output || '').trim();
        assert(outputText.length > 0, \`\${mode} returned empty multimodal output\`);
        assert(
          outputText.toLowerCase().includes('hello'),
          \`\${mode} did not recognize the synthetic HELLO image: \${outputText}\`,
        );
        assert(usages.length === 1, \`\${mode} reported usage \${usages.length} times\`);
        const [usage] = usages;
        assert(
          usage.promptTokens > promptTextTokens,
          \`\${mode} reported \${usage.promptTokens} prompt positions for an image prompt of \${promptTextTokens} text tokens\`,
        );
        assert(usage.cachedPromptTokens === 0, \`\${mode} reused \${usage.cachedPromptTokens} tokens of a multimodal prompt\`);
        assert(
          usage.completionTokens > 0 && usage.completionTokens <= 64,
          \`\${mode} reported \${usage.completionTokens} completion tokens\`,
        );
        assert(
          usage.finishReason === (usage.completionTokens === 64 ? 'length' : 'stop'),
          \`\${mode} finished with \${usage.finishReason} after \${usage.completionTokens} tokens\`,
        );
        assert(
          usage.timeToFirstTokenMs > 0 && usage.timeToFirstTokenMs <= usage.durationMs,
          \`\${mode} reported time to first token \${usage.timeToFirstTokenMs} of \${usage.durationMs} ms\`,
        );
        const metadata = bridge.getModelMetadata();
        const runtimeNotes = String(metadata['llamadart.webgpu.runtime_notes'] || '');
        assert(
          runtimeNotes.includes('media_image_resized:320x180->'),
          \`\${mode} did not expose the image-resize diagnostic: \${runtimeNotes}\`,
        );
        modeResults.push({
          mode,
          elapsedMs: Math.round(performance.now() - startedAt),
          output: outputText.slice(0, 160),
          promptTextTokens,
          usage,
          imageResizeDiagnostic: runtimeNotes
            .split(';')
            .find((note) => note.startsWith('media_image_resized:')),
        });
      } finally {
        await bridge.dispose();
      }
    };

    await verifyMode(true, 'direct runtime');
    await verifyMode(false, 'worker runtime');
    finish({
      ok: true,
      modes: ['direct runtime', 'worker runtime'],
      modeResults,
    });
  } catch (error) {
    finish({
      ok: false,
      error: String(error && error.stack ? error.stack : error),
    });
  }
})();
</script>
`;
}

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'multimodal_browser_smoke.mjs',
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
        default: () => env.int('LLAMA_WEBGPU_MULTIMODAL_TIMEOUT_MS', '420000'),
        help: 'Browser operation timeout in milliseconds.',
      },
      {
        flag: '--model-url',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_MULTIMODAL_MODEL_URL'),
        help: 'Qwen multimodal GGUF URL.',
      },
      {
        flag: '--model-path',
        type: 'path',
        default: () => env.optionalPath('LLAMA_WEBGPU_MULTIMODAL_MODEL_PATH'),
        help: 'Local Qwen multimodal GGUF path.',
      },
      {
        flag: '--model-sha256',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_MULTIMODAL_MODEL_SHA256'),
        help: 'Expected model SHA-256.',
      },
      {
        flag: '--mmproj-url',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_MULTIMODAL_MMPROJ_URL'),
        help: 'Qwen multimodal projector GGUF URL.',
      },
      {
        flag: '--mmproj-path',
        type: 'path',
        default: () => env.optionalPath('LLAMA_WEBGPU_MULTIMODAL_MMPROJ_PATH'),
        help: 'Local Qwen multimodal projector GGUF path.',
      },
      {
        flag: '--mmproj-sha256',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_MULTIMODAL_MMPROJ_SHA256'),
        help: 'Expected multimodal projector SHA-256.',
      },
      {
        flag: '--model-cache-dir',
        type: 'path',
        default: () => env.path('LLAMA_WEBGPU_MULTIMODAL_MODEL_CACHE', DEFAULT_MODEL_CACHE),
        help: 'Cache directory used for downloaded model files.',
      },
      {
        flag: '--artifacts-dir',
        type: 'path',
        default: () => env.optionalPath('LLAMA_WEBGPU_MULTIMODAL_ARTIFACTS_DIR'),
        help: 'Directory for JSON, console, and screenshot diagnostics.',
      },
    ],
  });
}

// Checks that run after the payload is printed.
export function checkPayload(payload) {
  ensure(pyEquals(pyGet(payload, 'modes'), MODES), 'multimodal smoke modes payload mismatch');
  const modeResults = pyGet(payload, 'modeResults');
  ensure(Array.isArray(modeResults) && modeResults.length === 2, 'multimodal mode results missing');
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const distDir = resolvePath(args.distDir);
  ensure(isDirectory(distDir), `dist directory does not exist: ${distDir}`);
  const cacheDir = resolvePath(expandHome(args.modelCacheDir));
  const modelPath = await resolvePinnedFile({
    filePath: args.modelPath,
    url: args.modelUrl,
    expectedSha256: args.modelSha256,
    cacheDir,
    label: 'multimodal model',
  });
  const mmprojPath = await resolvePinnedFile({
    filePath: args.mmprojPath,
    url: args.mmprojUrl,
    expectedSha256: args.mmprojSha256,
    cacheDir,
    label: 'multimodal projector',
  });
  const artifactsDir = args.artifactsDir !== null ? resolvePath(args.artifactsDir) : null;

  const payload = await withTempDir('llama-web-bridge-multimodal-', async (webRoot) => {
    await copyBridgeArtifacts(distDir, webRoot);
    await fsp.copyFile(modelPath, path.join(webRoot, 'multimodal-model.gguf'));
    await fsp.copyFile(mmprojPath, path.join(webRoot, 'multimodal-mmproj.gguf'));
    await fsp.writeFile(path.join(webRoot, 'index.html'), renderHarness(), 'utf8');
    if (artifactsDir !== null) {
      await fsp.mkdir(artifactsDir, { recursive: true });
      await fsp.copyFile(path.join(webRoot, 'index.html'), path.join(artifactsDir, 'index.html'));
    }
    return withServer(webRoot, (url) => runPlaywright(url, args.timeoutMs, artifactsDir, 'multimodal-smoke'));
  });

  writeStdout(`${pyJson(payload, { indent: 2, sortKeys: true })}\n`);
  if (payload.ok !== true) return 1;
  checkPayload(payload);
  return 0;
}

if (import.meta.main) {
  await runMain('multimodal', main);
}
