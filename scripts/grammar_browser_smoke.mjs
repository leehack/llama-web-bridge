#!/usr/bin/env node
// Browser smoke for grammar-constrained completion.
//
// Ported from grammar_browser_smoke.py with the same flags, environment
// variables, harness page and output.

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

const DESCRIPTION = `Browser smoke for grammar-constrained completion.

Runs grammar-constrained \`\`createCompletion\`\` calls through the direct and
worker runtimes of the wasm32 and wasm64 cores with a checksum-pinned GGUF and checks that every result is
text the grammar accepts. It covers greedy \`\`topK: 1\`\` decoding, which used to
truncate the candidates to a token the grammar rejects and abort the Wasm core
(issue #115), a sampled \`\`topK: 40\`\` run, top-p-only truncation, and a small
JSON grammar. It first sends invalid grammars, which used to abort the core
from the parser's throw: each must reject with an \`\`(invalid grammar)\`\` error
and leave the runtime usable for the valid cases that follow. The worker
runtime must still own the model afterwards: an abort there used to move the
bridge to the main thread for the rest of the session.`;

const MODEL_FILENAME = 'grammar-smoke-model.gguf';
// The issue #115 repro prompt. Its ChatML markers are plain text to a model
// without that template, which is fine: the grammar decides the output.
const PROMPT = '<|im_start|>user\nIs the sky blue? Answer yes or no.<|im_end|>\n'
  + '<|im_start|>assistant\n<think>\n\n</think>\n\n';
const YES_NO_GRAMMAR = 'root ::= "yes" | "no"';
const JSON_GRAMMAR = [
  String.raw`root ::= "{" ws "\"answer\"" ws ":" ws answer ws "," ws `
    + String.raw`"\"confidence\"" ws ":" ws digit ws "}"`,
  String.raw`answer ::= "\"yes\"" | "\"no\""`,
  'digit ::= [0-9]',
  'ws ::= " "?',
].join('\n');
const INVALID_GRAMMAR_ERROR = '(invalid grammar)';
// The bridge's text when llama.cpp returns no grammar without throwing.
const NON_THROWING_REJECTION = 'grammar rejected by llama.cpp';
const INVALID_OPTIONS = { nPredict: 8, temp: 0, topK: 1, seed: 1 };
// Each case keeps the Python dict's key order: the harness embeds it as JSON.
export const CASES = Object.freeze([
  // A parser syntax error, thrown and caught in the bridge's grammar wrapper.
  {
    name: 'invalid-unterminated-string',
    grammar: 'root ::= "unterminated',
    kind: 'invalid',
    throws: true,
    options: INVALID_OPTIONS,
  },
  // Thrown by the parser's undefined-rule check after parsing succeeds.
  {
    name: 'invalid-undefined-rule',
    grammar: 'root ::= answer',
    kind: 'invalid',
    throws: true,
    options: INVALID_OPTIONS,
  },
  // Rejected by llama.cpp without a throw.
  {
    name: 'invalid-left-recursion',
    grammar: 'root ::= root "a" | "a"',
    kind: 'invalid',
    throws: false,
    options: INVALID_OPTIONS,
  },
  {
    name: 'yes-no-greedy',
    grammar: YES_NO_GRAMMAR,
    kind: 'yes-no',
    options: { nPredict: 8, temp: 0, topK: 1, seed: 1 },
  },
  {
    name: 'yes-no-sampled',
    grammar: YES_NO_GRAMMAR,
    kind: 'yes-no',
    options: { nPredict: 8, temp: 0.8, topK: 40, seed: 1 },
  },
  {
    name: 'json-greedy',
    grammar: JSON_GRAMMAR,
    kind: 'json',
    options: { nPredict: 64, temp: 0, topK: 1, seed: 1 },
  },
  {
    name: 'json-top-p',
    grammar: JSON_GRAMMAR,
    kind: 'json',
    options: { nPredict: 64, temp: 0.7, topK: 0, topP: 0.1, seed: 7 },
  },
]);
export const MEMORY_MODES = Object.freeze(['wasm32', 'wasm64']);
const RUNTIME_MODES = Object.freeze(['direct', 'worker']);

export function renderHarness(nCtx, memoryModes) {
  const config = pyJson({
    modelUrl: `/${MODEL_FILENAME}`,
    prompt: PROMPT,
    cases: CASES,
    invalidGrammarError: INVALID_GRAMMAR_ERROR,
    nonThrowingRejection: NON_THROWING_REJECTION,
    nCtx,
    memoryModes,
    runtimeModes: RUNTIME_MODES,
  });
  return `
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge grammar smoke</title>
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
  const errorText = (error) => String(error && error.message ? error.message : error);
  const checkText = (kind, text) => {
    if (kind === 'yes-no') {
      return text === 'yes' || text === 'no';
    }
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (_) {
      return false;
    }
    return parsed !== null
      && typeof parsed === 'object'
      && Object.keys(parsed).join(',') === 'answer,confidence'
      && (parsed.answer === 'yes' || parsed.answer === 'no')
      && Number.isInteger(parsed.confidence)
      && parsed.confidence >= 0
      && parsed.confidence <= 9;
  };
  try {
    if (!window.crossOriginIsolated) {
      throw new Error('test page is not cross-origin isolated');
    }
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    assert(typeof LlamaWebGpuBridge === 'function', 'LlamaWebGpuBridge export was not registered');
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
      const cases = [];
      try {
        await bridge.loadModelFromUrl(config.modelUrl, {
          nCtx: config.nCtx,
          nThreads: 2,
          nGpuLayers: 0,
          useCache: false,
          forceRemoteFetchBackend: false,
        });
        for (const testCase of config.cases) {
          let text = null;
          let error = null;
          try {
            text = await bridge.createCompletion(config.prompt, {
              ...testCase.options,
              grammar: testCase.grammar,
              tokenEventEncoding: 'text',
            });
          } catch (caught) {
            error = errorText(caught);
          }
          cases.push({
            name: testCase.name,
            text,
            error,
            // A throwing case must carry the parser's message, which proves
            // the throw reached the bridge's catch.
            valid: testCase.kind === 'invalid'
              ? text === null
                && error !== null
                && error.includes(config.invalidGrammarError)
                && error.includes(config.nonThrowingRejection) === !testCase.throws
              : error === null && typeof text === 'string' && checkText(testCase.kind, text),
          });
        }

        // Generation state must be clean after the grammar runs.
        let plainError = null;
        try {
          await bridge.createCompletion(config.prompt, {
            nPredict: 4,
            temp: 0,
            topK: 1,
            seed: 1,
            tokenEventEncoding: 'text',
          });
        } catch (caught) {
          plainError = errorText(caught);
        }

        const metadata = bridge.getModelMetadata();
        return {
          mode,
          cases,
          plainCompletionError: plainError,
          execution: metadata['llamadart.webgpu.execution'] || null,
          coreVariant: metadata['llamadart.webgpu.core_variant'] || null,
          workerFallbackReason: metadata['llamadart.webgpu.worker_fallback_reason'] || null,
        };
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
  const expectedCases = CASES.map((testCase) => testCase.name);
  for (const entry of modeResults) {
    const mode = modeOf(entry);
    const cases = pyGet(entry, 'cases');
    if (!Array.isArray(cases) || !pyEquals(cases.map((testCase) => pyGet(testCase, 'name')), expectedCases)) {
      failures.push(`${mode}: case results missing`);
      continue;
    }
    for (const testCase of cases) {
      if (pyGet(testCase, 'valid') !== true) {
        const expected = pyGet(testCase, 'name').startsWith('invalid-')
          ? `an error containing ${pyRepr(INVALID_GRAMMAR_ERROR)} and the `
            + "parser's reason (or the non-throwing rejection for left recursion)"
          : 'grammar-valid text';
        failures.push(
          `${mode} ${pyStr(pyGet(testCase, 'name'))}: expected ${expected}, `
            + `got text=${pyRepr(pyGet(testCase, 'text'))} error=${pyRepr(pyGet(testCase, 'error'))}`,
        );
      }
    }
    if (pyGet(entry, 'plainCompletionError') !== null) {
      failures.push(`${mode}: completion after the grammar runs failed: ${pyRepr(pyGet(entry, 'plainCompletionError'))}`);
    }
    failures.push(...modeRuntimeFailures(mode, entry));
  }
  if (pyGet(payload, 'globalWorkerFallbackReason') !== null) {
    failures.push(`worker fell back to the main thread: ${pyRepr(pyGet(payload, 'globalWorkerFallbackReason'))}`);
  }
  return failures;
}

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'grammar_browser_smoke.mjs',
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
        default: () => env.int('LLAMA_WEBGPU_GRAMMAR_TIMEOUT_MS', '300000'),
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
        default: () => env.optionalPath('LLAMA_WEBGPU_GRAMMAR_ARTIFACTS_DIR'),
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

  const payload = await withTempDir('llama-web-bridge-grammar-smoke-', async (webRoot) => {
    await copyBridgeArtifacts(distDir, webRoot);
    if (memoryModes.includes('wasm64')) await copyMemory64Artifacts(distDir, webRoot);
    await fsp.copyFile(modelPath, path.join(webRoot, MODEL_FILENAME));
    await fsp.writeFile(path.join(webRoot, 'index.html'), renderHarness(args.nCtx, memoryModes), 'utf8');
    if (artifactsDir !== null) {
      await fsp.mkdir(artifactsDir, { recursive: true });
      await fsp.copyFile(path.join(webRoot, 'index.html'), path.join(artifactsDir, 'index.html'));
    }
    return withServer(webRoot, (url) => runPlaywright(url, args.timeoutMs, artifactsDir, 'grammar-smoke'));
  });

  writeStdout(`${pyJson(payload, { indent: 2, sortKeys: true })}\n`);
  const failures = validatePayload(payload, memoryModes);
  for (const failure of failures) process.stderr.write(`grammar browser smoke: ${failure}\n`);
  return failures.length ? 1 : 0;
}

if (import.meta.main) {
  await runMain('grammar', main);
}
