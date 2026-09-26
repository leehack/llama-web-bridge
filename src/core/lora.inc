// LoRA adapters: applying the active set to the context.
// Part of llama_webgpu_core.cpp, included inside its anonymous namespace;
// not a standalone translation unit.

// Applies g_active_loras to g_state.ctx. The cached prompt is dropped because
// its KV entries were computed with the previous adapters.
void apply_active_loras() {
  std::vector<llama_adapter_lora *> adapters;
  std::vector<float> scales;
  for (const auto & [handle, scale] : g_active_loras) {
    adapters.push_back(g_lora_adapters.at(handle));
    scales.push_back(scale);
  }
  llama_set_adapters_lora(
      g_state.ctx,
      adapters.empty() ? nullptr : adapters.data(),
      adapters.size(),
      scales.empty() ? nullptr : scales.data());
  g_cached_prompt_tokens.clear();
}

// Checks that a LoRA call may run now and that `handle`, when not null,
// names a loaded adapter. Returns 0 or the negative status to report.
int32_t check_lora_call(const int32_t * handle) {
  if (!ensure_loaded()) {
    return -1;
  }
  if (g_generation_active || g_tts_active) {
    set_error("LoRA adapters cannot change during active generation or text-to-speech synthesis");
    return -2;
  }
  if (handle != nullptr && g_lora_adapters.find(*handle) == g_lora_adapters.end()) {
    set_error(
        "LoRA adapter " + std::to_string(*handle) +
        " is not loaded; its model was unloaded or replaced. Load the adapter again.");
    return -3;
  }
  return 0;
}
