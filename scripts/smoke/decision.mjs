#!/usr/bin/env node
// Real-model browser smoke for Laya decision heads on a ModernBERT encoder.
//
// Ported from decision_browser_smoke.py with the same flags, harness page,
// output and parity checks. The fixture is read as Python's json.loads read
// it, so the page embeds its numbers exactly as the Python smoke did.

import fsp from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import {
  copyBridgeArtifacts,
  copyMemory64Artifacts,
  ensure,
  isDict,
  isDirectory,
  isFile,
  isPyInt,
  parseSmokeArgs,
  pyFloatRepr,
  pyGet,
  pyGetDefault,
  pyIndex,
  pyIter,
  pyJson,
  pyJsonLoads,
  pyLen,
  pyLessEqual,
  pyRepr,
  pyStr,
  pyTruthy,
  pyTypeName,
  readPyText,
  resolvePath,
  runMain,
  runPollingPlaywright,
  sha256File,
  stageFile,
  withServer,
  withTempDir,
  writeStdout,
} from './support.mjs';

const DESCRIPTION = 'Real-model browser smoke for Laya decision heads on a ModernBERT encoder.';
export const MEMORY_MODES = Object.freeze(['wasm32', 'wasm64']);
export const RUNTIME_MODES = Object.freeze(['direct', 'worker']);
const HEAD_SOURCES = Object.freeze(['url', 'bytes']);
const QUESTION_TYPES = Object.freeze({ choice: 0, score: 1, noul: 2 });

// isinstance(value, int), where bool is an int.
const isInt = (value) => typeof value === 'boolean' || isPyInt(value);

// `value in QUESTION_TYPES`, which raises for an unhashable list or dict.
function isQuestionType(value) {
  if (Array.isArray(value) || isDict(value)) {
    const type = pyTypeName(value);
    throw new Error(`cannot use '${type}' as a dict key (unhashable type: '${type}')`);
  }
  return typeof value === 'string' && Object.hasOwn(QUESTION_TYPES, value);
}

export async function loadFixture(file) {
  const fixture = pyJsonLoads(await readPyText(file));
  ensure(isDict(fixture), 'fixture is not a JSON object');
  const rows = pyGet(fixture, 'rows');
  ensure(Array.isArray(rows) && rows.length > 0, 'fixture has no rows');
  const special = pyGet(fixture, 'specialTokens');
  ensure(
    isDict(special) && ['cls', 'sep', 'mask'].every((name) => isInt(pyGet(special, name))),
    'fixture specialTokens must name cls, sep and mask ids',
  );
  const sequences = rows.map((row, index) => {
    const questionType = pyGet(pyGetDefault(row, 'question', {}), 'type');
    ensure(isQuestionType(questionType), `fixture row ${index} has question type ${pyRepr(questionType)}`);
    ensure(
      pyLen(pyGetDefault(row, 'markers', [])) === pyLen(pyGetDefault(row, 'rawLogits', [])),
      `fixture row ${index} has mismatched markers and logits`,
    );
    return {
      id: pyGetDefault(row, 'id', String(index)),
      tokens: pyIndex(row, 'ids'),
      markers: pyIndex(row, 'markers'),
      questionType: QUESTION_TYPES[questionType],
      rawLogits: pyIndex(row, 'rawLogits'),
      rawActLogits: pyIndex(row, 'rawActLogits'),
    };
  });
  return { specialTokens: special, sequences };
}

export function renderHarness({ fixture, configJson, memoryModes, runtimeModes, headSource, gpuLayers, contextSize }) {
  return `
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge decision smoke</title>
<pre id="result">pending</pre>
<script type="module">
(async () => {
  const resultNode = document.getElementById('result');
  const finish = (payload) => {
    resultNode.textContent = JSON.stringify(payload);
    window.__smokeResult = payload;
  };
  const setStage = (stage) => {
    window.__smokeStage = stage;
    console.log(\`decision-smoke-stage:\${stage}\`);
  };
  const assert = (condition, message) => {
    if (!condition) throw new Error(message);
  };
  const rejects = async (promise, pattern, label) => {
    try {
      await promise;
    } catch (error) {
      const text = String(error?.message || error);
      assert(pattern.test(text), \`\${label} rejected with an unexpected error: \${text}\`);
      return text;
    }
    throw new Error(\`\${label} unexpectedly succeeded\`);
  };
  const softmax = (values) => {
    const top = Math.max(...values);
    const exps = values.map((value) => Math.exp(value - top));
    const sum = exps.reduce((total, value) => total + value, 0);
    return exps.map((value) => value / sum);
  };
  const argmax = (values) => values.reduce(
    (best, value, index) => (value > values[best] ? index : best),
    0,
  );
  // A safetensors file with the given header text and no tensor data.
  const safetensorsBytes = (headerText) => {
    const header = new TextEncoder().encode(headerText);
    const bytes = new Uint8Array(8 + header.length);
    new DataView(bytes.buffer).setBigUint64(0, BigInt(header.length), true);
    bytes.set(header, 8);
    return bytes;
  };
  // Deep enough to overflow a recursive JSON copy or dump.
  const deepJson = '['.repeat(200000) + ']'.repeat(200000);
  try {
    assert(window.crossOriginIsolated, 'test page is not cross-origin isolated');
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    assert(typeof LlamaWebGpuBridge === 'function', 'bridge export was not registered');
    const fixture = ${pyJson(fixture)};
    const configJson = ${pyJson(configJson)};
    const headSource = ${pyJson(headSource)};
    const sequences = fixture.sequences.map((row) => ({
      tokens: Int32Array.from(row.tokens),
      markers: row.markers,
      questionType: row.questionType,
    }));

    const modeResults = [];
    for (const memoryMode of ${pyJson(memoryModes)}) {
      for (const runtimeMode of ${pyJson(runtimeModes)}) {
        const useMemory64 = memoryMode === 'wasm64';
        const bridge = new LlamaWebGpuBridge({
          disableWorker: runtimeMode === 'direct',
          logLevel: 2,
          preferMemory64: useMemory64,
          coreModuleUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.js' : undefined,
          wasmUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.wasm' : undefined,
        });
        const startedAt = performance.now();
        try {
          setStage(\`\${memoryMode}:\${runtimeMode}:unloaded-capabilities\`);
          const unloaded = await bridge.getDecisionCapabilities();
          assert(unloaded.apiVersion === 1, 'unexpected decision API version');
          assert(unloaded.supported === false, 'decision heads reported support before a model loaded');

          setStage(\`\${memoryMode}:\${runtimeMode}:load-model\`);
          const modelStartedAt = performance.now();
          await bridge.loadModelFromUrl('/decision-model.gguf', {
            nCtx: ${contextSize},
            nGpuLayers: ${gpuLayers},
            nThreads: 4,
            nBatch: ${contextSize},
            nUbatch: ${contextSize},
            useCache: false,
            forceRemoteFetchBackend: false,
          });
          const modelLoadMs = Math.round(performance.now() - modelStartedAt);

          setStage(\`\${memoryMode}:\${runtimeMode}:capabilities\`);
          const capabilities = await bridge.getDecisionCapabilities();
          assert(capabilities.apiVersion === 1, 'unexpected decision API version');
          assert(capabilities.supported === true, \`decision heads unsupported: \${capabilities.reason}\`);

          setStage(\`\${memoryMode}:\${runtimeMode}:reject-invalid-head\`);
          await rejects(
            bridge.loadDecisionHead(new Uint8Array([1, 2, 3])),
            /Invalid safetensors file "decision head bytes"/,
            'truncated head bytes',
          );

          setStage(\`\${memoryMode}:\${runtimeMode}:reject-deep-header\`);
          await rejects(
            bridge.loadDecisionHead(safetensorsBytes(
              \`{"x":{"dtype":"F32","shape":\${deepJson},"data_offsets":[0,0]}}\`,
            )),
            /Invalid safetensors file "decision head bytes": header nests JSON deeper than 64 levels/,
            'deeply nested head header',
          );

          setStage(\`\${memoryMode}:\${runtimeMode}:reject-deep-config\`);
          await rejects(
            bridge.loadDecisionHead(safetensorsBytes('{}'), {
              configJson: \`{"temperature":\${deepJson}}\`,
            }),
            /Decision head config nests JSON deeper than 64 levels/,
            'deeply nested configJson',
          );

          setStage(\`\${memoryMode}:\${runtimeMode}:large-config\`);
          // Past the 1 MiB wasm stack, so configJson must not travel as a
          // stack-copied string. The empty head then fails its layout check.
          await rejects(
            bridge.loadDecisionHead(safetensorsBytes('{}'), {
              configJson: JSON.stringify({ max_len: 512, pad: 'x'.repeat(3 * 1024 * 1024) }),
            }),
            /has no tensor "head[.]layers[.]0[.]linear1[.]weight"/,
            'large configJson',
          );

          setStage(\`\${memoryMode}:\${runtimeMode}:load-head\`);
          const headOptions = configJson === null ? {} : { configJson };
          const headStartedAt = performance.now();
          const source = headSource === 'bytes'
            ? new Uint8Array(await (await fetch('/decision-head.safetensors')).arrayBuffer())
            : '/decision-head.safetensors';
          const info = await bridge.loadDecisionHead(source, headOptions);
          const headLoadMs = Math.round(performance.now() - headStartedAt);
          assert(info.apiVersion === 1, 'unexpected head info API version');
          assert(Number.isInteger(info.handle) && info.handle > 0, 'head handle is not positive');
          assert(info.clsToken === fixture.specialTokens.cls, \`CLS token \${info.clsToken} does not match the fixture\`);
          assert(info.sepToken === fixture.specialTokens.sep, \`SEP token \${info.sepToken} does not match the fixture\`);
          assert(info.maskToken === fixture.specialTokens.mask, \`MASK token \${info.maskToken} does not match the fixture\`);
          assert(info.maskText.length > 0, 'MASK token text is empty');
          assert(info.configJson.length > 0, 'head info has no config');
          assert(info.deviceName.length > 0, 'head info has no device name');

          setStage(\`\${memoryMode}:\${runtimeMode}:reject-oversized-sequence\`);
          // One token past the longest fixture row, which the encoder must accept.
          const oversizedLength = Math.max(...sequences.map((row) => row.tokens.length)) + 1;
          const oversized = new Int32Array(oversizedLength).fill(sequences[0].tokens[0]);
          const tokenLimitError = await rejects(
            bridge.runDecision(info.handle, [
              sequences[0],
              { tokens: oversized, markers: [1], questionType: 0 },
            ]),
            new RegExp(\`Decision sequence 1 has \${oversizedLength} tokens; the decision encoder accepts 1 to [0-9]+[.]\`),
            'oversized sequence',
          );

          setStage(\`\${memoryMode}:\${runtimeMode}:reject-excess-markers\`);
          const firstTokens = sequences[0].tokens;
          await rejects(
            bridge.runDecision(info.handle, [{
              tokens: firstTokens,
              markers: new Array(firstTokens.length + 1).fill(1),
              questionType: sequences[0].questionType,
            }]),
            new RegExp(\`Decision sequence 0 has \${firstTokens.length + 1} markers for its \${firstTokens.length} tokens\`),
            'more markers than tokens',
          );

          setStage(\`\${memoryMode}:\${runtimeMode}:run\`);
          const runStartedAt = performance.now();
          const outputs = await bridge.runDecision(info.handle, sequences);
          const runMs = Math.round(performance.now() - runStartedAt);
          assert(outputs.length === sequences.length, 'decision output count mismatch');

          let worstLogitDiff = 0;
          let worstProbabilityDiff = 0;
          let worstActProbabilityDiff = 0;
          let worstActRelativeDiff = 0;
          const argmaxChanges = [];
          const actDecisionChanges = [];
          outputs.forEach((output, index) => {
            const row = fixture.sequences[index];
            assert(output.logits instanceof Float32Array, \`row \${row.id} logits are not Float32Array\`);
            assert(output.actLogits instanceof Float32Array, \`row \${row.id} act logits are not Float32Array\`);
            assert(output.logits.length === row.rawLogits.length, \`row \${row.id} logit count mismatch\`);
            assert(output.actLogits.length === row.rawActLogits.length, \`row \${row.id} act logit count mismatch\`);
            const logits = Array.from(output.logits);
            const actLogits = Array.from(output.actLogits);
            for (const value of [...logits, ...actLogits]) {
              assert(Number.isFinite(value), \`row \${row.id} has a non-finite output\`);
            }
            logits.forEach((value, option) => {
              worstLogitDiff = Math.max(worstLogitDiff, Math.abs(value - row.rawLogits[option]));
            });
            const probabilities = softmax(logits);
            const reference = softmax(row.rawLogits);
            probabilities.forEach((value, option) => {
              worstProbabilityDiff = Math.max(worstProbabilityDiff, Math.abs(value - reference[option]));
            });
            if (argmax(logits) !== argmax(row.rawLogits)) {
              argmaxChanges.push(row.id);
            }
            actLogits.forEach((value, action) => {
              const expected = row.rawActLogits[action];
              worstActRelativeDiff = Math.max(
                worstActRelativeDiff,
                Math.abs(value - expected) / Math.max(1, Math.abs(expected)),
              );
            });
            const actProbability = softmax(actLogits)[0];
            const referenceActProbability = softmax(row.rawActLogits)[0];
            worstActProbabilityDiff = Math.max(
              worstActProbabilityDiff,
              Math.abs(actProbability - referenceActProbability),
            );
            if ((actProbability >= 0.5) !== (referenceActProbability >= 0.5)) {
              actDecisionChanges.push(row.id);
            }
          });

          setStage(\`\${memoryMode}:\${runtimeMode}:free\`);
          await bridge.freeDecisionHead(info.handle);
          await bridge.freeDecisionHead(info.handle);
          await rejects(
            bridge.runDecision(info.handle, [sequences[0]]),
            /Decision head \\d+ is not loaded/,
            'freed head',
          );

          modeResults.push({
            memoryMode,
            runtimeMode,
            headSource,
            requestedGpuLayers: ${gpuLayers},
            gpuActive: bridge.isGpuActive(),
            backendName: bridge.getBackendName(),
            headDevice: info.deviceName,
            hiddenSize: info.hiddenSize,
            rows: outputs.length,
            totalElapsedMs: Math.round(performance.now() - startedAt),
            modelLoadMs,
            headLoadMs,
            runMs,
            msPerQuestion: runMs / outputs.length,
            worstLogitDiff,
            worstProbabilityDiff,
            worstActProbabilityDiff,
            worstActRelativeDiff,
            argmaxChanges,
            actDecisionChanges,
            tokenLimitError,
            invalidHeadRejected: true,
            hostileInputsRejected: true,
            freeTested: true,
          });
        } finally {
          await bridge.dispose();
        }
      }
    }

    finish({ ok: true, modeResults });
  } catch (error) {
    finish({ ok: false, error: String(error?.stack || error) });
  }
})();
</script>
`;
}

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'decision.mjs',
    description: DESCRIPTION,
    options: [
      { flag: '--dist-dir', type: 'path', default: () => 'dist' },
      { flag: '--model-path', type: 'path', required: true },
      { flag: '--head-path', type: 'path', required: true },
      {
        flag: '--config-path',
        type: 'path',
        help: 'rl_agent_config.json for a head without laya.config metadata',
      },
      {
        flag: '--fixture-path',
        type: 'path',
        required: true,
        help: 'Laya reference fixture with ids, markers, rawLogits and rawActLogits',
      },
      { flag: '--model-sha256', type: 'string', default: () => '' },
      { flag: '--head-sha256', type: 'string', default: () => '' },
      { flag: '--config-sha256', type: 'string', default: () => '' },
      { flag: '--gpu-layers', type: 'int', default: () => 0 },
      { flag: '--context-size', type: 'int', default: () => 512 },
      { flag: '--head-source', type: 'string', choices: HEAD_SOURCES, default: () => 'url' },
      { flag: '--max-logit-diff', type: 'float', default: () => 0.25 },
      { flag: '--max-probability-diff', type: 'float', default: () => 0.06 },
      { flag: '--max-act-probability-diff', type: 'float', default: () => 0.05 },
      // Laya act logits are thousands apart, so softmax saturates and only the
      // raw logits can show a broken act head.
      { flag: '--max-act-relative-diff', type: 'float', default: () => 0.05 },
      { flag: '--memory-mode', type: 'string', choices: ['all', ...MEMORY_MODES], default: () => 'all' },
      { flag: '--runtime-mode', type: 'string', choices: ['all', ...RUNTIME_MODES], default: () => 'all' },
      { flag: '--timeout-ms', type: 'int', default: () => 1_800_000 },
      { flag: '--artifacts-dir', type: 'path' },
    ],
  });
}

// Every mode must stay within the reference thresholds and change no decision.
export function checkParity(payload, args) {
  for (const result of pyIter(pyGetDefault(payload, 'modeResults', []))) {
    const mode = `${pyStr(pyGet(result, 'memoryMode'))}/${pyStr(pyGet(result, 'runtimeMode'))}`;
    const within = (key, limit) => pyLessEqual(pyGetDefault(result, key, Infinity), limit);
    ensure(
      within('worstLogitDiff', args.maxLogitDiff),
      `${mode}: worst logit difference ${pyStr(pyGet(result, 'worstLogitDiff'))} exceeds ${pyFloatRepr(args.maxLogitDiff)}`,
    );
    ensure(
      within('worstProbabilityDiff', args.maxProbabilityDiff),
      `${mode}: worst probability difference ${pyStr(pyGet(result, 'worstProbabilityDiff'))} `
        + `exceeds ${pyFloatRepr(args.maxProbabilityDiff)}`,
    );
    ensure(
      within('worstActProbabilityDiff', args.maxActProbabilityDiff),
      `${mode}: worst act probability difference ${pyStr(pyGet(result, 'worstActProbabilityDiff'))} `
        + `exceeds ${pyFloatRepr(args.maxActProbabilityDiff)}`,
    );
    ensure(
      within('worstActRelativeDiff', args.maxActRelativeDiff),
      `${mode}: worst act logit relative difference ${pyStr(pyGet(result, 'worstActRelativeDiff'))} `
        + `exceeds ${pyFloatRepr(args.maxActRelativeDiff)}`,
    );
    ensure(
      !pyTruthy(pyGet(result, 'argmaxChanges')) && !pyTruthy(pyGet(result, 'actDecisionChanges')),
      `${mode}: decisions changed against the reference: `
        + `options ${pyStr(pyGet(result, 'argmaxChanges'))}, act ${pyStr(pyGet(result, 'actDecisionChanges'))}`,
    );
  }
}

// The payload printed to stdout: everything but the console lines, which stay
// in the result artifact.
export function printedPayload(payload) {
  return Object.fromEntries(Object.entries(payload).filter(([key]) => key !== 'console'));
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const distDir = resolvePath(args.distDir);
  const modelPath = resolvePath(args.modelPath);
  const headPath = resolvePath(args.headPath);
  const configPath = args.configPath !== null ? resolvePath(args.configPath) : null;
  const fixturePath = resolvePath(args.fixturePath);
  ensure(isDirectory(distDir), `dist directory does not exist: ${distDir}`);
  ensure(isFile(modelPath), `model does not exist: ${modelPath}`);
  ensure(isFile(headPath), `decision head does not exist: ${headPath}`);
  ensure(isFile(fixturePath), `fixture does not exist: ${fixturePath}`);
  if (configPath !== null) ensure(isFile(configPath), `head config does not exist: ${configPath}`);
  ensure(!args.configSha256 || configPath !== null, 'config checksum requires --config-path');
  ensure(args.contextSize > 0, 'context size must be positive');
  const checksums = [
    ['model', modelPath, args.modelSha256],
    ['head', headPath, args.headSha256],
    ['config', configPath, args.configSha256],
  ];
  const digests = {};
  for (const [name, file, expected] of checksums) {
    digests[name] = file !== null ? await sha256File(file) : null;
    if (expected) ensure(digests[name] === expected.toLowerCase(), `${name} checksum mismatch`);
  }
  const fixture = await loadFixture(fixturePath);
  const configJson = configPath !== null ? await readPyText(configPath) : null;
  const memoryModes = args.memoryMode === 'all' ? MEMORY_MODES : [args.memoryMode];
  const runtimeModes = args.runtimeMode === 'all' ? RUNTIME_MODES : [args.runtimeMode];

  const payload = await withTempDir('llama-web-bridge-decision-', async (webRoot) => {
    await copyBridgeArtifacts(distDir, webRoot);
    if (memoryModes.includes('wasm64')) await copyMemory64Artifacts(distDir, webRoot);
    await stageFile(modelPath, path.join(webRoot, 'decision-model.gguf'));
    await stageFile(headPath, path.join(webRoot, 'decision-head.safetensors'));
    await fsp.writeFile(path.join(webRoot, 'index.html'), renderHarness({
      fixture,
      configJson,
      memoryModes,
      runtimeModes,
      headSource: args.headSource,
      gpuLayers: args.gpuLayers,
      contextSize: args.contextSize,
    }), 'utf8');
    const artifactsDir = args.artifactsDir !== null ? resolvePath(args.artifactsDir) : null;
    if (artifactsDir !== null) {
      await fsp.mkdir(artifactsDir, { recursive: true });
      await fsp.copyFile(path.join(webRoot, 'index.html'), path.join(artifactsDir, 'index.html'));
    }
    return withServer(webRoot, (url) => runPollingPlaywright(url, args.timeoutMs, artifactsDir, {
      artifactPrefix: 'decision-smoke',
      stageMarker: 'decision-smoke-stage:',
    }));
  });

  payload.modelSha256 = digests.model;
  payload.headSha256 = digests.head;
  payload.configSha256 = digests.config;
  writeStdout(`${pyJson(printedPayload(payload), { indent: 2, sortKeys: true })}\n`);
  if (payload.ok !== true) return 1;
  ensure(
    pyLen(pyGetDefault(payload, 'modeResults', [])) === memoryModes.length * runtimeModes.length,
    'mode results are incomplete',
  );
  checkParity(payload, args);
  return 0;
}

if (import.meta.main) {
  await runMain('decision', main);
}
