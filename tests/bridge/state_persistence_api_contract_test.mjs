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
  /g_cached_prompt_tokens\s*=\s*restored_tokens/.test(SRC),
  'state load must restore g_cached_prompt_tokens for prompt-prefix reuse',
);
const loadWrapper = extractNativeFunction('llamadart_webgpu_state_load_file');
const saveWrapper = extractNativeFunction('llamadart_webgpu_state_save_file');
const saveOpen = saveWrapper.indexOf('can_open_state_file(state_path, "wb")');
require(
  saveOpen >= 0 && saveOpen < saveWrapper.indexOf('llama_state_save_file('),
  'state save must open the file before llama.cpp does: its open error is a C++ throw that aborts the core',
);
require(
  !loadWrapper.includes('llama_state_load_file('),
  'state load must not hand a file to llama.cpp: its short reads throw past the catch compiled out of llama.cpp',
);
const loadSteps = [
  'read_state_snapshot_head(\n      state_path, restored_tokens,',
  'try_state_load_alloc(file_size, 1)',
  'try_state_load_alloc(current_size, 2)',
  'while (snapshot && !try_state_load_alloc(file_size, 4)) {',
  'llamadart_webgpu_state_set_data(\n        g_state.ctx, snapshot.get() + state_offset,',
  'llamadart_webgpu_state_read_file(',
  'llamadart_webgpu_state_set_data(\n            g_state.ctx, current_state.get(), current_size, &restore_error)',
].map((step) => loadWrapper.indexOf(step));
require(
  loadSteps.every((index, i) => index >= 0 && (i === 0 || loadSteps[i - 1] < index)),
  'state load must check the snapshot header, try to copy the snapshot and the current state, keep them only beside a restore reserve, restore through a catching wrapper, and restore the copy when llama.cpp rejects the snapshot',
);
require(
  /if \(snapshot\) \{\s*current_size = llama_state_get_size\(g_state\.ctx\);\s*current_state = try_state_load_alloc\(current_size, 2\);/.test(loadWrapper)
    && /while \(snapshot && !try_state_load_alloc\(file_size, 4\)\) \{\s*if \(current_state\) \{\s*current_state\.reset\(\);\s*\} else \{\s*snapshot\.reset\(\);\s*\}\s*\}/.test(loadWrapper)
    && !/std::vector<uint8_t>/.test(loadWrapper),
  'the copies must not crowd out the restore: the current state is copied only beside a snapshot copy, both only while a snapshot-sized reserve still fits, and without them the load matches the previous file path',
);
require(
  !/std::vector<uint8_t>\s+(snapshot|current_state)\b/.test(loadWrapper)
    && /if \(snapshot\) \{\s*loaded = llamadart_webgpu_state_set_data\(.*?\} else \{.*?loaded = llamadart_webgpu_state_read_file\(/s.test(loadWrapper)
    && /if \(current_state\s*&& llamadart_webgpu_state_set_data\(/.test(loadWrapper),
  'the snapshot and current-state copies must be optional: a load without them falls back to llama.cpp reading the file, and a rejection without a copy clears the state',
);
const tryAlloc = SRC.slice(SRC.indexOf('malloc_buffer try_state_load_alloc('), SRC.indexOf('bool read_state_file_into('));
require(
  /g_state_load_test_failures & test_failure/.test(tryAlloc)
    && /#if !LLAMADART_WEBGPU_ALLOW_MEMORY_GROWTH.*?sbrk\(0\).*?return nullptr;.*?#endif/s.test(tryAlloc)
    && /std::malloc\(/.test(tryAlloc) && !/\bnew\b|std::vector/.test(tryAlloc)
    && /LLAMADART_WEBGPU_ALLOW_MEMORY_GROWTH=\$<BOOL:\$\{LLAMADART_WEBGPU_ALLOW_MEMORY_GROWTH\}>/.test(CMAKE)
    && CMAKE.includes("'_llamadart_webgpu_test_state_load_failures'"),
  'optional state-load copies must use a null-returning malloc, stay within the heap when it cannot grow, and honour the test hook',
);
const stateClears = [...loadWrapper.matchAll(/llama_memory_clear|g_cached_prompt_tokens\.clear\(\)/g)].map(({ index }) => index);
require(
  stateClears.length === 2
    && stateClears.every((index) => index > loadSteps[loadSteps.length - 1] && index < loadWrapper.indexOf('return -6;'))
    && /llama_memory_clear\(llama_get_memory\(g_state\.ctx\), false\);\s*g_cached_prompt_tokens\.clear\(\);\s*g_last_output\.clear\(\);\s*g_last_piece\.clear\(\);\s*g_last_detokenized\.clear\(\);\s*set_error\([^]*?could not be kept[^]*?\);\s*return -6;/.test(loadWrapper),
  'only a rejection whose current-state copy is missing or fails to restore may clear the KV, prompt-cache and output state',
);
const rejections = [...loadWrapper.matchAll(/set_error\(((?:[^;"]|"[^"]*")*)\);\s*return (-[57]);/g)];
require(
  rejections.filter(([, , code]) => code === '-5').length === 2
    && (loadWrapper.match(/return -5;/g) ?? []).length === 2
    && rejections.filter(([, , code]) => code === '-7').length === 1
    && rejections.every(([, message, code]) => code === '-7' || /\bkept\b/.test(message))
    && /const std::string kept =[^;]*The current context state was kept\.";/.test(loadWrapper),
  'unreadable and rejected snapshots must reject with their own codes and say the current context state was kept',
);
const snapshotHead = SRC.slice(SRC.indexOf('int32_t read_state_snapshot_head('), SRC.indexOf('struct free_deleter'));
require(
  /file_size < sizeof\(header\).*?return -5;.*?header\[0\] != LLAMA_SESSION_MAGIC \|\| header\[1\] != LLAMA_SESSION_VERSION.*?return -5;.*?token_count > tokens\.size\(\).*?return -5;.*?file_size < state_offset \+ sizeof\(arch_size\).*?return -5;.*?read\(tokens\.data\(\), token_count \* sizeof\(llama_token\)\).*?file_size - state_offset - sizeof\(arch_size\) < arch_size.*?return -5;/s.test(snapshotHead)
    && !/std::vector<uint8_t>|resize\(/.test(snapshotHead),
  'the snapshot head check must cover the session header, the token capacity and the architecture name length llama.cpp allocates before reading, reading the tokens into the caller\'s buffer',
);
const STATE_RESTORE = readRepoText('src/llama_webgpu_state.cpp');
require(
  ['llamadart_webgpu_state_set_data', 'llamadart_webgpu_state_read_file'].every((name) => CMAKE.includes(`'${name}'`)
    && new RegExp(String.raw`extern "C" __attribute__\(\(noinline\)\) bool ${name}\(`).test(STATE_RESTORE))
    && /set_source_files_properties\([^)]*src\/llama_webgpu_state\.cpp/.test(CMAKE)
    && (STATE_RESTORE.match(/catch \(const std::exception & error\)/g) ?? []).length === 2
    && /n_read == size/.test(STATE_RESTORE),
  'state restores must catch llama.cpp throws by name, with exception catching compiled and linked, and require every byte read',
);
require(
  /bool can_open_state_file\([^)]*\) \{.*?S_ISDIR\(info\.st_mode\).*?std::fopen\(state_path, mode\).*?set_error\(.*?std::strerror\(/s.test(SRC),
  'the state file open check must reject directories and report the path and the open error',
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
