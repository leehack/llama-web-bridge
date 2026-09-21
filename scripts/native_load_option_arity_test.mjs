import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { LlamaWebGpuBridge } from '../js/src/llama_webgpu_bridge.js';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const coreSource = readFileSync(path.join(rootDir, 'src/llama_webgpu_core.cpp'), 'utf8');
const bridgeSources = {
  'js/src/llama_webgpu_bridge.js': readFileSync(
    path.join(rootDir, 'js/src/llama_webgpu_bridge.js'),
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

function nativeParameterTypes(name) {
  const match = coreSource.match(
    new RegExp(`EMSCRIPTEN_KEEPALIVE\\s+int32_t\\s+${name}\\(([^)]*)\\)`),
  );
  assert.ok(match, `${name} is not exported from src/llama_webgpu_core.cpp`);
  return match[1]
    .split(',')
    .map((parameter) => parameter.trim())
    .filter(Boolean)
    .map((parameter) => (parameter.includes('*') ? 'string' : 'number'));
}

function readArrayLiteral(source, start) {
  const open = source.indexOf('[', start);
  assert.notEqual(open, -1, 'ccall argument array not found');
  let depth = 0;
  for (let index = open; index < source.length; index += 1) {
    if (source[index] === '[') depth += 1;
    if (source[index] === ']') depth -= 1;
    if (depth === 0) {
      const entries = source
        .slice(open + 1, index)
        .split(',')
        .map((entry) => entry.trim())
        .filter(Boolean);
      return { entries, end: index + 1 };
    }
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
      types: types.entries.map((entry) => entry.replace(/["']/g, '')),
      values: values.entries,
    });
  }
  return sites;
}

const runtime = LlamaWebGpuBridge.prototype._createRuntime.call({ _config: {} });
const optionValues = runtime._nativeLoadOptionValues();
const optionTypes = runtime._nativeLoadOptionTypes();
assert.equal(optionTypes.length, optionValues.length);
assert.equal(optionValues.length, 11);
assert.ok(optionTypes.every((type) => type === 'number'));
assert.ok(optionValues.every((value) => typeof value === 'number'));

for (const [name, expectedCount] of Object.entries(expectedSites)) {
  const nativeTypes = nativeParameterTypes(name);
  const leadingCount = nativeTypes.length - optionValues.length;
  assert.ok(leadingCount > 0, `${name} native signature is shorter than the option list`);
  assert.deepEqual(
    nativeTypes.slice(leadingCount),
    optionTypes,
    `${name} native trailing parameters do not match the derived option types`,
  );

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
        nativeTypes,
        `${label} argument types do not match the native signature`,
      );
    }
  }
}

console.log('Native load option arity tests passed');
