// Static API contract checks for bridge state-persistence support.
import { bridgeJsSource, readRepoText } from './bridge_js_source.mjs';
import { readNativeCoreSource } from './native_core_source.mjs';

const SRC = readNativeCoreSource();
const JS = bridgeJsSource();
const CMAKE = readRepoText('CMakeLists.txt');
const README = readRepoText('README.md');

const REQUIRED_NATIVE = [
  'llamadart_webgpu_state_save_file',
  'llamadart_webgpu_state_load_file',
];

const REQUIRED_JS_METHODS = [
  'stateSaveFile',
  'stateLoadFile',
  'stateSaveBytes',
  'stateLoadBytes',
];

const errors = [];

function require(condition, message) {
  if (!condition) {
    errors.push(message);
  }
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function methodCount(name) {
  return (JS.match(new RegExp(String.raw`\basync\s+${escapeRegExp(name)}\s*\(`, 'g')) ?? []).length;
}

function extractNativeFunction(name) {
  const match = new RegExp(String.raw`EMSCRIPTEN_KEEPALIVE\s+int32_t\s+${escapeRegExp(name)}\s*\(`).exec(SRC);
  if (match === null) {
    return '';
  }

  const brace = SRC.indexOf('{', match.index + match[0].length);
  if (brace < 0) {
    return '';
  }

  let depth = 0;
  for (let index = brace; index < SRC.length; index += 1) {
    const char = SRC[index];
    if (char === '{') {
      depth += 1;
    } else if (char === '}') {
      depth -= 1;
      if (depth === 0) {
        return SRC.slice(brace, index + 1);
      }
    }
  }
  return '';
}

for (const symbol of REQUIRED_NATIVE) {
  require(SRC.includes(symbol), `missing native wrapper ${symbol} in src/llama_webgpu_core.cpp`);
  require(CMAKE.includes(`_${symbol}`), `missing exported function _${symbol} in CMakeLists.txt`);
}

require(
  SRC.includes('llama_state_save_file'),
  'native save wrapper must call llama_state_save_file',
);
require(
  SRC.includes('llama_state_load_file'),
  'native load wrapper must call llama_state_load_file',
);
require(
  /g_cached_prompt_tokens\s*=\s*restored_tokens/.test(SRC),
  'state load must restore g_cached_prompt_tokens for prompt-prefix reuse',
);
const loadWrapper = extractNativeFunction('llamadart_webgpu_state_load_file');
require(
  loadWrapper.includes('if (!loaded)')
    && loadWrapper.includes('llama_memory_clear(llama_get_memory(g_state.ctx), false);')
    && loadWrapper.includes('g_cached_prompt_tokens.clear();')
    && loadWrapper.includes('g_last_output.clear();')
    && loadWrapper.includes('g_last_piece.clear();')
    && loadWrapper.includes('g_last_detokenized.clear();'),
  'state load failure must clear potentially stale KV/prompt-cache/output state',
);
require(
  SRC.includes('g_generation_active')
    && SRC.includes('State cannot be saved or loaded during active generation'),
  'native wrappers must reject save/load while generation is active',
);

for (const method of REQUIRED_JS_METHODS) {
  const count = methodCount(method);
  require(
    count >= 2,
    `expected runtime and public bridge async methods for ${method}, found ${count}`,
  );
}

require(
  JS.includes('llamadart_webgpu_state_save_file'),
  'JS runtime must ccall llamadart_webgpu_state_save_file',
);
require(
  JS.includes('llamadart_webgpu_state_load_file'),
  'JS runtime must ccall llamadart_webgpu_state_load_file',
);
require(
  JS.includes('FS.readFile') && JS.includes('FS.writeFile'),
  'bytes helpers must use WASMFS readFile/writeFile',
);
require(
  JS.includes("const message = { type: 'call', id, method, args };")
    && JS.includes('this._worker.postMessage(message, transfers)'),
  'worker proxy calls must support transfer lists',
);
require(
  JS.includes("if (method === 'stateSaveBytes')")
    && JS.includes("self.postMessage({ type: 'result', id, value }, transfers)"),
  'worker stateSaveBytes must transfer the state Uint8Array buffer back to the main thread',
);
require(
  JS.includes('[transferableBytes.buffer]'),
  'worker stateLoadBytes must transfer a copied state buffer into the worker',
);
require(
  README.includes('State persistence') && README.includes('stateSaveBytes') && README.includes('stateLoadBytes'),
  'README must document state persistence semantics and bytes APIs',
);
require(
  README.includes('stateSave*` snapshots the current llama.cpp context')
    && README.includes('The `tokens` argument is stored')
    && README.includes('whose numeric conversion is greater than zero')
    && README.includes('all other values fall back'),
  'README must document snapshot timing, token metadata semantics, and tokenCapacity fallback requirements',
);

if (errors.length > 0) {
  console.error('State persistence API contract failed:');
  for (const error of errors) {
    console.error(`- ${error}`);
  }
  process.exit(1);
}

console.log('State persistence API contract passed');
