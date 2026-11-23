import assert from 'node:assert/strict';

import { readNativeCoreSource } from './native_core_source.mjs';

const source = readNativeCoreSource();
const loader = /^int32_t load_model_internal\([\s\S]*?^}/m.exec(source)?.[0];
assert.ok(loader);
const loadCall = loader.indexOf('llama_model_load_from_file(');
assert.ok(loadCall > 0);

assert.match(
  loader.slice(0, loadCall),
  /^#if SIZE_MAX <= 0xffffffffULL\n(?:  \/\/[^\n]*\n)*  mparams\.lazy_mode = LLAMA_LAZY_MODE_OFF;\n/m,
  'every wasm32 load must disable lazy reads before loading the model',
);
assert.equal(
  [...loader.matchAll(/\blazy_mode\s*=/g)].length,
  1,
  'memory64 and draft loads must not override lazy mode',
);
assert.ok(loader.indexOf('g_model_params = mparams;') > loadCall);
assert.match(source, /llamadart_webgpu_draft_model_load_file\(path, g_model_params, &load_error\)/);
