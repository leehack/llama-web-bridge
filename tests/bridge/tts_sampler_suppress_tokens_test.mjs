// The TTS talker sampler must bias the GGUF's suppress tokens to -inf before any
// other stage, as upstream common_sampler does. Qwen3-TTS lists its codec control
// tokens there, and code2wav rejects a sampled control token as out of range.
import assert from 'node:assert/strict';
import test from 'node:test';

import { readRepoText } from './bridge_js_source.mjs';

const TTS = readRepoText('src/llama_webgpu_tts.cpp');

function samplerBody() {
  const start = TTS.indexOf('llama_sampler * init_sampler(');
  assert.notEqual(start, -1, 'init_sampler not found');
  const end = TTS.indexOf('\nllama_webgpu_tts_status ', start);
  assert.ok(end > start, 'init_sampler end not found');
  return TTS.slice(start, end);
}

test('TTS sampler biases suppress tokens to -inf before the first sampling stage', () => {
  const body = samplerBody();
  const suppress = body.indexOf('llama_vocab_get_suppress_tokens(');
  const bias = body.indexOf('llama_sampler_init_logit_bias(');
  const firstStage = body.indexOf('llama_sampler_init_top_k(');
  assert.ok(suppress !== -1, 'suppress tokens are not read from the vocab');
  assert.ok(bias > suppress, 'logit bias is not built from the suppress tokens');
  assert.ok(firstStage > bias, 'logit bias must be added before top_k');
  assert.match(body, /-INFINITY/);
  assert.match(body, /llama_vocab_n_tokens\(vocab\)/);
});

test('TTS sampler is built from the loaded model vocab', () => {
  assert.match(
    TTS,
    /init_sampler\(\s*llama_model_get_vocab\(llama_get_model\(tts->llama\)\),\s*request\)/,
  );
});
