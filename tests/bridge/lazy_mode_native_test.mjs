import assert from 'node:assert/strict';

import { readNativeCoreSource } from './native_core_source.mjs';

const source = readNativeCoreSource();
const loader = /^int32_t load_model_internal\([\s\S]*?^}/m.exec(source)?.[0];
assert.ok(loader);
const loadCall = loader.indexOf('llama_model_load_from_file(');
assert.ok(loadCall > 0);

assert.match(
  loader.slice(0, loadCall),
  /^  mparams\.lazy_mode = LLAMA_LAZY_MODE_OFF;\n\s*#if SIZE_MAX <= 0xffffffffULL\n/m,
  'both wasm32 and memory64 must disable lazy reads before loading the model',
);
assert.equal(
  [...loader.matchAll(/\blazy_mode\s*=/g)].length,
  1,
  'the main model loader sets lazy mode once for both main and draft loads',
);
assert.ok(loader.indexOf('g_model_params = mparams;') > loadCall);
assert.match(source, /llamadart_webgpu_draft_model_load_file\(path, g_model_params, &load_error\)/);
