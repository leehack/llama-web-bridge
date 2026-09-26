// Exports: LoRA adapters.
// Part of llama_webgpu_core.cpp, included inside its extern "C" block;
// not a standalone translation unit.

EMSCRIPTEN_KEEPALIVE uint32_t llamadart_webgpu_lora_api_version() {
  return LLAMADART_WEBGPU_LORA_API_VERSION;
}

// Returns a positive adapter handle, or a negative status.
EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_lora_load(const char * adapter_path) {
  clear_error();
  const int32_t status = check_lora_call(nullptr);
  if (status != 0) {
    return status;
  }
  if (g_next_lora_handle == std::numeric_limits<int32_t>::max()) {
    set_error("LoRA adapter handles are exhausted; reload the model.");
    return -2;
  }
  std::string error;
  llama_adapter_lora * adapter =
      llamadart_webgpu_lora_adapter_init(g_state.model, adapter_path, &error);
  if (adapter == nullptr) {
    set_error(error);
    return -4;
  }
  const uint64_t invocation_tokens =
      llama_adapter_get_alora_n_invocation_tokens(adapter);
  if (invocation_tokens > 0) {
    llama_adapter_lora_free(adapter);
    set_error(
        "the adapter is an aLoRA adapter (" + std::to_string(invocation_tokens) +
        " invocation token(s)). LoRA adapters apply from the start of "
        "generation, but an aLoRA adapter must activate only after its "
        "invocation sequence appears in the prompt. Use a standard LoRA adapter.");
    return -5;
  }
  const int32_t handle = g_next_lora_handle++;
  g_lora_adapters[handle] = adapter;
  return handle;
}

// Applies the adapter at `scale`, keeping its place if it is already applied.
EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_lora_set(int32_t handle, float scale) {
  clear_error();
  const int32_t status = check_lora_call(&handle);
  if (status != 0) {
    return status;
  }
  if (!std::isfinite(scale)) {
    set_error("LoRA adapter scale must be finite");
    return -6;
  }
  const auto active = std::find_if(
      g_active_loras.begin(),
      g_active_loras.end(),
      [handle](const auto & entry) { return entry.first == handle; });
  if (active == g_active_loras.end()) {
    g_active_loras.emplace_back(handle, scale);
  } else {
    active->second = scale;
  }
  apply_active_loras();
  return 0;
}

// Stops applying the adapter; it stays loaded. Removing one that is not
// applied still re-applies the set and drops the cached prompt.
EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_lora_remove(int32_t handle) {
  clear_error();
  const int32_t status = check_lora_call(&handle);
  if (status != 0) {
    return status;
  }
  g_active_loras.erase(
      std::remove_if(
          g_active_loras.begin(),
          g_active_loras.end(),
          [handle](const auto & entry) { return entry.first == handle; }),
      g_active_loras.end());
  apply_active_loras();
  return 0;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_lora_clear() {
  clear_error();
  const int32_t status = check_lora_call(nullptr);
  if (status != 0) {
    return status;
  }
  g_active_loras.clear();
  apply_active_loras();
  return 0;
}
