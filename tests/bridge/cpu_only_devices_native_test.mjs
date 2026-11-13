import assert from 'node:assert/strict';

import { readNativeCoreSource } from './native_core_source.mjs';

// With no layer offloaded the model and context must not list the WebGPU
// device: Firefox resolves the queue waits ggml-webgpu blocks on every sync
// on a 100 ms tick, which cuts CPU decoding by roughly 50x.
const source = readNativeCoreSource();
const loader = source.slice(source.indexOf('int32_t load_model_internal('));
const body = loader.slice(0, loader.indexOf('llama_model_load_from_file('));

assert.match(
  body,
  /if \(n_gpu_layers == 0\) \{\s*mparams\.devices = [A-Za-z_]+;/,
  'a zero-layer load must pass an empty device list',
);
assert.match(
  body,
  /static ggml_backend_dev_t [A-Za-z_]+\[\] = \{nullptr\};/,
  'the empty list outlives the call: draft models reuse g_model_params',
);

// Tensor split needs a device, and an empty list has none.
assert.match(
  body,
  /if \(n_gpu_layers == 0\) \{[^}]*LLAMA_SPLIT_MODE_TENSOR[^}]*LLAMA_SPLIT_MODE_LAYER/,
  'a zero-layer load must not ask for tensor split',
);
