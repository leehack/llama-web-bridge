// Exports: decision heads.
// Part of llama_webgpu_core.cpp, included inside its extern "C" block;
// not a standalone translation unit.

EMSCRIPTEN_KEEPALIVE uint32_t llamadart_webgpu_decision_api_version() {
  return LLAMADART_WEBGPU_DECISION_API_VERSION;
}

EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_decision_capabilities_json() {
  g_decision_json = llama_webgpu_decision_capabilities_json(g_state.model);
  return g_decision_json.c_str();
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_decision_load(
    const char * head_path,
    const char * head_label,
    const char * config_path) {
  clear_error();
  if (!ensure_loaded()) {
    return LLAMADART_WEBGPU_DECISION_STATUS_INVALID_STATE;
  }
  if (g_generation_active || g_tts_active) {
    set_error("Decision heads cannot be loaded or run during active generation or text-to-speech synthesis");
    return LLAMADART_WEBGPU_DECISION_STATUS_INVALID_STATE;
  }
  if (g_next_decision_handle == std::numeric_limits<int32_t>::max()) {
    set_error("Decision head handles are exhausted; reload the model.");
    return LLAMADART_WEBGPU_DECISION_STATUS_INVALID_STATE;
  }
  const llama_webgpu_decision_load_request request{
      g_state.model,
      head_path,
      head_label,
      config_path,
      g_model_uses_gpu_ops,
      llama_n_threads(g_state.ctx),
      llama_n_threads_batch(g_state.ctx),
  };
  llama_webgpu_decision_head * head = nullptr;
  std::string error;
  const llama_webgpu_decision_status status =
      llama_webgpu_decision_head_load(request, &head, &error);
  if (status != LLAMADART_WEBGPU_DECISION_STATUS_OK) {
    set_error(error);
    return status;
  }
  const int32_t handle = g_next_decision_handle++;
  g_decision_heads[handle] = head;
  return handle;
}

EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_decision_head_info_json(
    int32_t handle) {
  const auto it = g_decision_heads.find(handle);
  g_decision_json = it == g_decision_heads.end()
      ? "{}"
      : llama_webgpu_decision_head_info_json(it->second, handle);
  return g_decision_json.c_str();
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_decision_run(
    int32_t handle,
    const char * input_path,
    const char * output_path) {
  clear_error();
  const auto it = g_decision_heads.find(handle);
  if (it == g_decision_heads.end()) {
    set_error("The decision head is not loaded; it was freed or its model was unloaded. Load the decision head again.");
    return LLAMADART_WEBGPU_DECISION_STATUS_INVALID_STATE;
  }
  if (g_generation_active || g_tts_active) {
    set_error("Decision heads cannot be loaded or run during active generation or text-to-speech synthesis");
    return LLAMADART_WEBGPU_DECISION_STATUS_INVALID_STATE;
  }
  std::vector<llama_webgpu_decision_sequence> sequences;
  std::string error;
  llama_webgpu_decision_status status =
      llama_webgpu_decision_read_sequences(input_path, &sequences, &error);
  std::vector<llama_webgpu_decision_output> outputs;
  if (status == LLAMADART_WEBGPU_DECISION_STATUS_OK) {
    status = llama_webgpu_decision_run(it->second, sequences, &outputs, &error);
  }
  if (status == LLAMADART_WEBGPU_DECISION_STATUS_OK) {
    status = llama_webgpu_decision_write_outputs(output_path, outputs, &error);
  }
  if (status != LLAMADART_WEBGPU_DECISION_STATUS_OK) {
    set_error(error);
    return status;
  }
  return static_cast<int32_t>(outputs.size());
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_decision_free(int32_t handle) {
  const auto it = g_decision_heads.find(handle);
  if (it != g_decision_heads.end()) {
    llama_webgpu_decision_head_free(it->second);
    g_decision_heads.erase(it);
  }
  return LLAMADART_WEBGPU_DECISION_STATUS_OK;
}
