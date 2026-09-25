import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const coreSource = readFileSync(path.join(rootDir, 'src/llama_webgpu_core.cpp'), 'utf8');
const bridgeSources = {
  'js/src/runtime.js': readFileSync(
    path.join(rootDir, 'js/src/runtime.js'),
    'utf8',
  ),
  'js/llama_webgpu_bridge.js': readFileSync(
    path.join(rootDir, 'js/llama_webgpu_bridge.js'),
    'utf8',
  ),
};

const expectedSites = {
  llamadart_webgpu_load_model: 2,
  llamadart_webgpu_load_model_from_url: 2,
};
const typesSpread = '...this._nativeLoadOptionTypes()';
const valuesSpread = '...this._nativeLoadOptionValues()';

const optionParameters = [
  { name: 'n_seq_max', field: '_nSeqMax', input: 21, expected: 21 },
  { name: 'use_mmap', field: '_useMmap', input: false, expected: 0 },
  { name: 'use_mlock', field: '_useMlock', input: true, expected: 1 },
  { name: 'flash_attn_type', field: '_flashAttention', input: 24, expected: 24 },
  { name: 'type_k', field: '_cacheTypeK', input: 25, expected: 25 },
  { name: 'type_v', field: '_cacheTypeV', input: 26, expected: 26 },
  { name: 'kv_unified', field: '_kvUnified', input: 27, expected: 27 },
  { name: 'rope_freq_base', field: '_ropeFrequencyBase', input: 28.5, expected: 28.5 },
  { name: 'rope_freq_scale', field: '_ropeFrequencyScale', input: 29.5, expected: 29.5 },
  { name: 'split_mode', field: '_splitMode', input: 30, expected: 30 },
  { name: 'main_gpu', field: '_mainGpu', input: 31, expected: 31 },
];

const leadingValueTokens = {
  model_path: ['this._modelPath'],
  model_url: ['remoteFetchUrl', 'reloadUrl'],
  n_ctx: ['this._nCtx'],
  n_threads: ['this._threads'],
  n_threads_batch: ['this._threadsBatch'],
  n_batch: ['this._nBatch'],
  n_ubatch: ['this._nUbatch'],
  n_gpu_layers: ['this._nGpuLayers', 'candidateLayers'],
  chunk_size: ['chunkBytes', 'remoteFetchReloadChunkBytes'],
};

function nativeParameters(name) {
  const match = coreSource.match(
    new RegExp(`EMSCRIPTEN_KEEPALIVE\\s+int32_t\\s+${name}\\(([^)]*)\\)`),
  );
  assert.ok(match, `${name} is not exported from src/llama_webgpu_core.cpp`);
  return match[1]
    .split(',')
    .map((parameter) => parameter.trim())
    .filter(Boolean)
    .map((parameter) => ({
      name: parameter.match(/([A-Za-z_][A-Za-z0-9_]*)\s*$/)[1],
      type: parameter.includes('*') ? 'string' : 'number',
    }));
}

function readArrayLiteral(source, start) {
  const open = source.indexOf('[', start);
  assert.notEqual(open, -1, 'ccall argument array not found');
  const entries = [];
  let entry = '';
  let depth = 0;
  const pushEntry = () => {
    const text = entry.replace(/\s+/g, ' ').trim();
    if (text) entries.push(text);
    entry = '';
  };
  let index = open + 1;
  while (index < source.length) {
    const char = source[index];
    const next = source[index + 1];
    if (char === '/' && next === '/') {
      const lineEnd = source.indexOf('\n', index);
      assert.notEqual(lineEnd, -1, 'unterminated ccall argument array');
      index = lineEnd;
      continue;
    }
    if (char === '/' && next === '*') {
      const close = source.indexOf('*/', index + 2);
      assert.notEqual(close, -1, 'unterminated block comment in ccall argument array');
      index = close + 2;
      continue;
    }
    if (char === '"' || char === "'" || char === '`') {
      let close = index + 1;
      while (close < source.length && source[close] !== char) {
        close += source[close] === '\\' ? 2 : 1;
      }
      assert.ok(close < source.length, 'unterminated string in ccall argument array');
      entry += source.slice(index, close + 1);
      index = close + 1;
      continue;
    }
    if ('([{'.includes(char)) {
      depth += 1;
    } else if (')]}'.includes(char)) {
      if (depth === 0) {
        assert.equal(char, ']', 'malformed ccall argument array');
        pushEntry();
        return { entries, end: index + 1 };
      }
      depth -= 1;
    } else if (char === ',' && depth === 0) {
      pushEntry();
      index += 1;
      continue;
    }
    entry += char;
    index += 1;
  }
  throw new Error('unterminated ccall argument array');
}

function ccallSites(source, name) {
  const sites = [];
  const pattern = new RegExp(`ccall\\(\\s*["']${name}["']\\s*,\\s*["']number["']\\s*,`, 'g');
  for (const match of source.matchAll(pattern)) {
    const types = readArrayLiteral(source, match.index + match[0].length);
    const values = readArrayLiteral(source, types.end);
    sites.push({
      types: types.entries.map((entry) => entry.replace(/^["']|["']$/g, '')),
      values: values.entries,
    });
  }
  return sites;
}

function createRuntime() {
  return LlamaWebGpuBridge.prototype._createRuntime.call({ _config: {} });
}

const runtime = createRuntime();
const optionValues = runtime._nativeLoadOptionValues();
const optionTypes = runtime._nativeLoadOptionTypes();
assert.equal(optionTypes.length, optionValues.length);
assert.equal(optionValues.length, optionParameters.length);
assert.ok(optionTypes.every((type) => type === 'number'));
assert.ok(optionValues.every((value) => typeof value === 'number'));

const orderedRuntime = createRuntime();
for (const { field, input } of optionParameters) {
  assert.ok(field in orderedRuntime, `runtime has no ${field} field`);
  orderedRuntime[field] = input;
}
assert.deepEqual(
  orderedRuntime._nativeLoadOptionValues(),
  optionParameters.map(({ expected }) => expected),
  'option values are not emitted in native parameter order',
);

const derivedRuntime = createRuntime();
derivedRuntime._nativeLoadOptionValues = () => [1, 2, 3];
assert.deepEqual(
  derivedRuntime._nativeLoadOptionTypes(),
  ['number', 'number', 'number'],
  'option types must be derived from the option values',
);

for (const [name, expectedCount] of Object.entries(expectedSites)) {
  const nativeTypes = nativeParameters(name);
  const leadingCount = nativeTypes.length - optionParameters.length;
  assert.ok(leadingCount > 0, `${name} native signature is shorter than the option list`);
  assert.deepEqual(
    nativeTypes.slice(leadingCount).map(({ name: parameter }) => parameter),
    optionParameters.map(({ name: parameter }) => parameter),
    `${name} native trailing parameters do not match the option list`,
  );
  const leadingParameters = nativeTypes.slice(0, leadingCount);

  for (const [sourcePath, source] of Object.entries(bridgeSources)) {
    const sites = ccallSites(source, name);
    assert.equal(sites.length, expectedCount, `${sourcePath}: ${name} ccall site count`);
    for (const [index, site] of sites.entries()) {
      const label = `${sourcePath}: ${name} site ${index + 1}`;
      assert.equal(site.types.at(-1), typesSpread, `${label} types must end with the option spread`);
      assert.equal(site.values.at(-1), valuesSpread, `${label} values must end with the option spread`);
      const leadingTypes = site.types.slice(0, -1);
      const leadingValues = site.values.slice(0, -1);
      assert.equal(leadingTypes.length, leadingValues.length, `${label} leading arity`);
      assert.deepEqual(
        [...leadingTypes, ...optionTypes],
        nativeTypes.map(({ type }) => type),
        `${label} argument types do not match the native signature`,
      );
      assert.deepEqual(
        leadingValues.map((value, position) => {
          const parameter = leadingParameters[position].name;
          const allowed = leadingValueTokens[parameter];
          assert.ok(allowed, `${label}: no expected value token for native parameter ${parameter}`);
          return allowed.some((token) => value.includes(token))
            ? parameter
            : `${parameter}=${value}`;
        }),
        leadingParameters.map(({ name: parameter }) => parameter),
        `${label} leading values do not match the native parameter order`,
      );
    }
  }
}

console.log('Native load option arity tests passed');
