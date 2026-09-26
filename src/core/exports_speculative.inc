// Exports: speculative decoding configuration and draft models.
// Part of llama_webgpu_core.cpp, included inside its extern "C" block;
// not a standalone translation unit.

// Sets the speculative configuration that the next begin_generation or
// generate call consumes. type_names is a comma-separated list of llama.cpp
// speculative type names. Returns 0, or -1 after set_error() for an unknown
// name; the stored configuration is then inactive.
EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_set_next_speculative(
    const char * type_names,
    int32_t draft_n_max,
    int32_t draft_n_min,
    float p_min,
    float p_split,
    int32_t ngram_size_n,
    int32_t ngram_size_m,
    int32_t ngram_min_hits,
    int32_t ngram_match,
    int32_t ngram_n_min,
    int32_t ngram_n_max,
    const char * cache_static_path,
    const char * cache_dynamic_path,
    int32_t max_tokens) {
  clear_error();
  g_next_speculative = speculative_request();

  speculative_request request;
  const std::string names = type_names == nullptr ? "" : type_names;
  size_t start = 0;
  while (start <= names.size()) {
    const size_t end = std::min(names.find(',', start), names.size());
    const std::string name = names.substr(start, end - start);
    if (!name.empty()) {
      const common_speculative_type type = common_speculative_type_from_name(name);
      if (type == COMMON_SPECULATIVE_TYPE_COUNT ||
          type == COMMON_SPECULATIVE_TYPE_NONE) {
        set_error("Unknown speculative decoding strategy: " + name);
        return -1;
      }
      if (!has_type(request.types, type)) {
        request.types.push_back(type);
      }
    }
    start = end + 1;
  }

  request.active = true;
  request.draft_n_max = draft_n_max;
  request.draft_n_min = draft_n_min;
  request.p_min = p_min;
  request.p_split = p_split;
  request.ngram_size_n = ngram_size_n;
  request.ngram_size_m = ngram_size_m;
  request.ngram_min_hits = ngram_min_hits;
  request.ngram_match = ngram_match;
  request.ngram_n_min = ngram_n_min;
  request.ngram_n_max = ngram_n_max;
  request.cache_static_path = cache_static_path == nullptr ? "" : cache_static_path;
  request.cache_dynamic_path = cache_dynamic_path == nullptr ? "" : cache_dynamic_path;
  request.max_tokens = max_tokens;
  g_next_speculative = std::move(request);
  return 0;
}

// Loads a draft model with the target model's load parameters, replacing any
// loaded draft model. Returns 0, or a negative code after set_error().
EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_draft_model_load(const char * path) {
  clear_error();

  if (!ensure_loaded()) {
    return -1;
  }
  if (path == nullptr || std::strlen(path) == 0) {
    set_error("Draft model path is empty");
    return -2;
  }
  if (g_generation_active || g_tts_active) {
    set_error("The draft model cannot be replaced during active generation or text-to-speech synthesis");
    return -4;
  }

  free_draft_model();

  std::string load_error;
  llama_model * model =
      llamadart_webgpu_draft_model_load_file(path, g_model_params, &load_error);
  if (model == nullptr) {
    set_error("Failed to load the draft model: " + load_error);
    return -3;
  }

  char architecture[128] = {0};
  if (llama_model_meta_val_str(
          model, "general.architecture", architecture, sizeof(architecture)) < 0) {
    architecture[0] = '\0';
  }
  g_draft_model = model;
  g_draft_model_architecture = architecture;
  // common_speculative_types_from_gguf asserts on a non-DFlash GGUF without
  // a block count, so only DFlash-architecture files reach it.
  if (g_draft_model_architecture == "dflash") {
    const std::vector<common_speculative_type> types =
        common_speculative_types_from_gguf(path);
    if (types.size() == 1) {
      g_draft_model_block_type = types[0];
    }
  }

  g_draft_model_json =
      "{\"architecture\":\"" + escape_json(g_draft_model_architecture) + "\"";
  if (g_draft_model_block_type != COMMON_SPECULATIVE_TYPE_NONE) {
    g_draft_model_json += ",\"strategy\":\"" +
        common_speculative_type_to_str(g_draft_model_block_type) + "\"";
  }
  g_draft_model_json += "}";
  return 0;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_draft_model_free() {
  clear_error();
  if (g_generation_active || g_tts_active) {
    set_error("The draft model cannot be unloaded during active generation or text-to-speech synthesis");
    return -1;
  }
  free_draft_model();
  return 0;
}

// {"architecture":...,"strategy"?:...} for the loaded draft model, or {}.
EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_draft_model_info_json() {
  return g_draft_model_json.c_str();
}

// Bytes the heap can still allocate: free space inside the allocator plus the
// growth left before the maximum memory size.
EMSCRIPTEN_KEEPALIVE double llamadart_webgpu_heap_headroom_bytes() {
  const struct mallinfo info = mallinfo();
  const double heap_max = static_cast<double>(emscripten_get_heap_max());
  const double heap_top = static_cast<double>(
      reinterpret_cast<uintptr_t>(sbrk(0)));
  return std::max(0.0, heap_max - heap_top) + static_cast<double>(info.fordblks);
}
