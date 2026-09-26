#!/usr/bin/env node
// Browser smoke for next-token scoring.
//
// Ported from next_token_scores_browser_smoke.py with the same flags,
// environment variables, harness page and output.

import fsp from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import {
  DEFAULT_MODEL_CACHE,
  copyBridgeArtifacts,
  copyMemory64Artifacts,
  ensure,
  env,
  expectedModes,
  isDict,
  isDirectory,
  modeOf,
  modeRuntimeFailures,
  parseSmokeArgs,
  pyEquals,
  pyGet,
  pyJson,
  pyRepr,
  pyStr,
  resolvePath,
  resolvePinnedModel,
  runMain,
  runPlaywright,
  withServer,
  withTempDir,
  writeStdout,
} from './browser_smoke_support.mjs';

const DESCRIPTION = `Browser smoke for next-token scoring.

Runs \`\`scoreNextToken\`\` through the direct and worker runtimes of the wasm32
and wasm64 cores with a checksum-pinned GGUF. Scores must be log-probabilities
in descending order, candidate scores must match the top-k scores of the same
tokens, and prompt-prefix reuse must match a fresh evaluation, for a repeated
prompt and for an extended one. Token bytes must decode like \`\`detokenize\`\`,
and a greedy one-token completion must pick a top-scoring token. A cancel
issued while idle must not abort the next score. Out-of-range
and empty requests must reject with stable messages and leave the runtime
usable, and the worker runtime must still own the model afterwards.`;

const MODEL_FILENAME = 'next-token-scores-smoke-model.gguf';
const PROMPT = 'The quick brown fox jumps over the lazy';
const PROMPT_SUFFIX = ' dog, and then the fox';
const TOP_K = 8;
// CPU logits of one position differ slightly between a whole-prompt batch and
// a single re-decoded token.
const REUSE_TOLERANCE = 1e-3;
const OUT_OF_VOCABULARY_ERROR = 'is outside the vocabulary';
const EMPTY_REQUEST_ERROR = 'Pass candidates, a positive topK, or both';
export const MEMORY_MODES = Object.freeze(['wasm32', 'wasm64']);
const RUNTIME_MODES = Object.freeze(['direct', 'worker']);

export function renderHarness(nCtx, memoryModes) {
  const config = pyJson({
    modelUrl: `/${MODEL_FILENAME}`,
    prompt: PROMPT,
    suffix: PROMPT_SUFFIX,
    topK: TOP_K,
    tolerance: REUSE_TOLERANCE,
    outOfVocabularyError: OUT_OF_VOCABULARY_ERROR,
    emptyRequestError: EMPTY_REQUEST_ERROR,
    nCtx,
    memoryModes,
    runtimeModes: RUNTIME_MODES,
  });
  return `
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge next-token scores smoke</title>
<pre id="result">pending</pre>
<script type="module">
(async () => {
  const result = document.getElementById('result');
  const finish = (payload) => {
    result.textContent = JSON.stringify(payload);
    window.__smokeResult = payload;
  };
  const errorText = (error) => String(error && error.message ? error.message : error);
  const decode = (bytes) => new TextDecoder().decode(bytes);
  try {
    if (!window.crossOriginIsolated) {
      throw new Error('test page is not cross-origin isolated');
    }
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    if (typeof LlamaWebGpuBridge !== 'function') {
      throw new Error('LlamaWebGpuBridge export was not registered');
    }
    const config = ${config};
    const modeResults = [];

    const runMode = async (memoryMode, runtimeMode) => {
      const mode = \`\${memoryMode} \${runtimeMode}\`;
      const useMemory64 = memoryMode === 'wasm64';
      const bridge = new LlamaWebGpuBridge({
        disableWorker: runtimeMode === 'direct',
        preferMemory64: useMemory64,
        coreModuleUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.js' : undefined,
        wasmUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.wasm' : undefined,
      });
      const failures = [];
      const check = (condition, message) => {
        if (!condition) {
          failures.push(message);
        }
      };
      const scoresById = (entries) => new Map(entries.map((entry) => [entry.token, entry.logprob]));
      const compareScores = (label, expected, actual) => {
        const actualById = scoresById(actual);
        for (const entry of expected) {
          const other = actualById.get(entry.token);
          check(
            other !== undefined && Math.abs(other - entry.logprob) <= config.tolerance,
            \`\${label}: token \${entry.token} scored \${other}, expected \${entry.logprob}\`,
          );
        }
      };
      const checkTop = (label, scores) => {
        check(scores.top.length === config.topK, \`\${label}: top has \${scores.top.length} entries\`);
        check(scores.promptTokens > 0, \`\${label}: promptTokens is \${scores.promptTokens}\`);
        check(new Set(scores.top.map((entry) => entry.token)).size === scores.top.length, \`\${label}: top ids repeat\`);
        let previous = 0;
        let mass = 0;
        for (const entry of scores.top) {
          check(Number.isFinite(entry.logprob) && entry.logprob <= previous, \`\${label}: logprob \${entry.logprob} out of order\`);
          check(entry.bytes instanceof Uint8Array, \`\${label}: token \${entry.token} bytes are not a Uint8Array\`);
          previous = entry.logprob;
          mass += Math.exp(entry.logprob);
        }
        check(mass <= 1 + 1e-6, \`\${label}: top probabilities sum to \${mass}\`);
      };
      try {
        await bridge.loadModelFromUrl(config.modelUrl, {
          nCtx: config.nCtx,
          nThreads: 2,
          nGpuLayers: 0,
          useCache: false,
          forceRemoteFetchBackend: false,
        });

        // A cancel with nothing running must not abort the next scoring decode.
        bridge.cancel();
        const first = await bridge.scoreNextToken(config.prompt, { topK: config.topK });
        checkTop('first', first);
        const ids = first.top.map((entry) => entry.token).reverse();

        const candidates = await bridge.scoreNextToken(config.prompt, { candidates: ids });
        check(
          JSON.stringify(candidates.candidates.map((entry) => entry.token)) === JSON.stringify(ids),
          'candidates must keep request order',
        );
        check(candidates.top.length === 0, 'topK 0 must return no top tokens');
        compareScores('candidates vs top', first.top, candidates.candidates);

        const fresh = await bridge.scoreNextToken(config.prompt, { candidates: ids, reusePromptPrefix: false });
        compareScores('repeated prompt reuse vs fresh', first.top, fresh.candidates);

        const extendedPrompt = config.prompt + config.suffix;
        const extended = await bridge.scoreNextToken(extendedPrompt, { topK: config.topK });
        checkTop('extended', extended);
        check(extended.promptTokens > first.promptTokens, 'the extended prompt must have more tokens');
        const extendedIds = extended.top.map((entry) => entry.token);
        const extendedFresh = await bridge.scoreNextToken(extendedPrompt, {
          candidates: extendedIds,
          reusePromptPrefix: false,
        });
        compareScores('extended prompt reuse vs fresh', extended.top, extendedFresh.candidates);

        for (const entry of first.top.slice(0, 4)) {
          const text = await bridge.detokenize([entry.token], true);
          check(decode(entry.bytes) === text, \`token \${entry.token} bytes decode differently from detokenize\`);
        }

        const completion = await bridge.createCompletion(config.prompt, {
          nPredict: 1,
          temp: 0,
          topK: 1,
          seed: 1,
          tokenEventEncoding: 'text',
        });
        const best = first.top[0].logprob;
        check(
          first.top.some((entry) => entry.logprob >= best - config.tolerance && decode(entry.bytes) === completion),
          \`greedy completion \${JSON.stringify(completion)} is not a top-scoring token\`,
        );

        const errors = {};
        for (const [name, options, expected] of [
          ['out-of-vocabulary candidate', { candidates: [0x7fffffff] }, config.outOfVocabularyError],
          ['negative candidate', { candidates: [-1] }, config.outOfVocabularyError],
          ['out-of-vocabulary topK', { topK: 0x7fffffff }, config.outOfVocabularyError],
          ['empty request', {}, config.emptyRequestError],
        ]) {
          let error = null;
          try {
            await bridge.scoreNextToken(config.prompt, options);
          } catch (caught) {
            error = errorText(caught);
          }
          errors[name] = error;
          check(error !== null && error.includes(expected), \`\${name}: expected \${expected}, got \${error}\`);
        }

        const after = await bridge.scoreNextToken(config.prompt, { topK: config.topK });
        compareScores('after rejected requests', first.top, after.top);

        const metadata = bridge.getModelMetadata();
        return {
          mode,
          failures,
          errors,
          top: first.top.map((entry) => [entry.token, entry.logprob]),
          promptTokens: [first.promptTokens, extended.promptTokens],
          completion,
          execution: metadata['llamadart.webgpu.execution'] || null,
          coreVariant: metadata['llamadart.webgpu.core_variant'] || null,
          workerFallbackReason: metadata['llamadart.webgpu.worker_fallback_reason'] || null,
        };
      } catch (error) {
        return { mode, failures: [...failures, \`threw: \${errorText(error)}\`] };
      } finally {
        await bridge.dispose();
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

export function validatePayload(payload, memoryModes) {
  const failures = [];
  if (payload.ok !== true) return [`harness failed: ${pyStr(pyGet(payload, 'error'))}`];
  const modeResults = pyGet(payload, 'modeResults');
  if (!Array.isArray(modeResults) || !pyEquals(
    modeResults.filter(isDict).map((entry) => pyGet(entry, 'mode')),
    expectedModes(memoryModes, RUNTIME_MODES),
  )) {
    return ['mode results missing'];
  }
  for (const entry of modeResults) {
    const mode = modeOf(entry);
    const modeFailures = pyGet(entry, 'failures');
    if (!Array.isArray(modeFailures)) {
      failures.push(`${mode}: failures missing`);
      continue;
    }
    failures.push(...modeFailures.map((failure) => `${mode}: ${pyStr(failure)}`));
    if (modeFailures.length) continue;
    failures.push(...modeRuntimeFailures(mode, entry));
  }
  if (pyGet(payload, 'globalWorkerFallbackReason') !== null) {
    failures.push(`worker fell back to the main thread: ${pyRepr(pyGet(payload, 'globalWorkerFallbackReason'))}`);
  }
  return failures;
}

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'next_token_scores_browser_smoke.mjs',
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
        default: () => env.int('LLAMA_WEBGPU_NEXT_TOKEN_SCORES_TIMEOUT_MS', '300000'),
        help: 'Browser operation timeout in milliseconds.',
      },
      {
        flag: '--model-url',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_URL'),
        help: 'GGUF URL; defaults to the state-persistence smoke model.',
      },
      {
        flag: '--model-path',
        type: 'path',
        default: () => env.optionalPath('LLAMA_WEBGPU_SMOKE_MODEL_PATH'),
        help: 'Local GGUF path.',
      },
      {
        flag: '--model-sha256',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_SHA256'),
        help: 'Expected model SHA-256.',
      },
      {
        flag: '--model-cache-dir',
        type: 'path',
        default: () => env.path('LLAMA_WEBGPU_SMOKE_MODEL_CACHE', DEFAULT_MODEL_CACHE),
        help: 'Cache directory used with --model-url.',
      },
      {
        flag: '--memory-mode',
        type: 'string',
        choices: ['all', ...MEMORY_MODES],
        default: () => 'all',
        help: 'Core memory mode to run; wasm64 needs the mem64 artifacts.',
      },
      {
        flag: '--n-ctx',
        type: 'int',
        default: () => 1024,
        help: 'Context size passed to loadModelFromUrl.',
      },
      {
        flag: '--artifacts-dir',
        type: 'path',
        default: () => env.optionalPath('LLAMA_WEBGPU_NEXT_TOKEN_SCORES_ARTIFACTS_DIR'),
        help: 'Directory for JSON/console/screenshot diagnostics.',
      },
    ],
  });
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const distDir = resolvePath(args.distDir);
  ensure(isDirectory(distDir), `dist directory does not exist: ${distDir}`);
  const artifactsDir = args.artifactsDir !== null ? resolvePath(args.artifactsDir) : null;
  const modelPath = await resolvePinnedModel(args);
  const memoryModes = args.memoryMode === 'all' ? MEMORY_MODES : [args.memoryMode];

  const payload = await withTempDir('llama-web-bridge-next-token-scores-smoke-', async (webRoot) => {
    await copyBridgeArtifacts(distDir, webRoot);
    if (memoryModes.includes('wasm64')) await copyMemory64Artifacts(distDir, webRoot);
    await fsp.copyFile(modelPath, path.join(webRoot, MODEL_FILENAME));
    await fsp.writeFile(path.join(webRoot, 'index.html'), renderHarness(args.nCtx, memoryModes), 'utf8');
    if (artifactsDir !== null) {
      await fsp.mkdir(artifactsDir, { recursive: true });
      await fsp.copyFile(path.join(webRoot, 'index.html'), path.join(artifactsDir, 'index.html'));
    }
    return withServer(
      webRoot,
      (url) => runPlaywright(url, args.timeoutMs, artifactsDir, 'next-token-scores-smoke'),
    );
  });

  writeStdout(`${pyJson(payload, { indent: 2, sortKeys: true })}\n`);
  const failures = validatePayload(payload, memoryModes);
  for (const failure of failures) process.stderr.write(`next-token scores browser smoke: ${failure}\n`);
  return failures.length ? 1 : 0;
}

if (import.meta.main) {
  await runMain('next-token scores', main);
}
