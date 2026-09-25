// Exports: text generation and model queries.
// Part of llama_webgpu_core.cpp, included inside its extern "C" block;
// not a standalone translation unit.

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_generate(
    const char * prompt,
    int32_t n_predict,
    float temp,
    int32_t top_k,
    float top_p,
    float repeat_penalty,
    const char * grammar,
    uint32_t seed) {
  if (n_predict <= 0) {
    n_predict = 128;
  }

  const int32_t begin_rc = begin_generation_impl(
      prompt,
      temp,
      top_k,
      top_p,
      repeat_penalty,
      grammar,
      seed);
  if (begin_rc != 0) {
    return begin_rc;
  }

  for (int32_t i = 0; i < n_predict; ++i) {
    const int32_t step_rc = next_token_impl();
    if (step_rc == 0) {
      break;
    }

    if (step_rc < 0) {
      end_generation_state();
      return step_rc;
    }
  }

  end_generation_state();
  return 0;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_begin_generation(
    const char * prompt,
    float temp,
    int32_t top_k,
    float top_p,
    float repeat_penalty,
    const char * grammar,
    uint32_t seed) {
  return begin_generation_impl(
      prompt,
      temp,
      top_k,
      top_p,
      repeat_penalty,
      grammar,
      seed);
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_next_token() {
  return next_token_impl();
}

EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_last_piece() {
  return g_last_piece.c_str();
}

EMSCRIPTEN_KEEPALIVE void llamadart_webgpu_end_generation() {
  end_generation_state();
}

EMSCRIPTEN_KEEPALIVE void llamadart_webgpu_request_cancel() {
  g_cancel_requested = true;
  llama_webgpu_tts_cancel(g_tts);
}

EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_last_output() {
  return g_last_output.c_str();
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_get_context_size() {
  if (g_state.ctx == nullptr) {
    return 0;
  }
  return static_cast<int32_t>(llama_n_ctx(g_state.ctx));
}

EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_model_meta_json() {
  return g_model_meta_json.c_str();
}
