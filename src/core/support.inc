// Helpers: text/JSON/file utilities, errors, logging, backend init, teardown.
// Part of llama_webgpu_core.cpp, included inside its anonymous namespace;
// not a standalone translation unit.

std::string to_lower(std::string value) {
  std::transform(
      value.begin(),
      value.end(),
      value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string escape_json(const std::string & value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (const char c : value) {
    switch (c) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

void set_error(const std::string & message) {
  g_last_error = message;
}

void clear_error() {
  g_last_error.clear();
}

void webgpu_log_callback(
    ggml_log_level level,
    const char * text,
    void * user_data) {
  (void) user_data;

  const int32_t configured = g_log_level.load(std::memory_order_relaxed);
  if (configured <= 0 || text == nullptr) {
    return;
  }

  int32_t effective = 0;
  if (level == GGML_LOG_LEVEL_CONT) {
    effective = g_last_non_cont_level.load(std::memory_order_relaxed);
  } else {
    effective = static_cast<int32_t>(level);
    g_last_non_cont_level.store(effective, std::memory_order_relaxed);
  }

  if (effective == GGML_LOG_LEVEL_NONE) {
    return;
  }

  if (effective >= configured) {
    std::fputs(text, stderr);
    std::fflush(stderr);
  }
}

void apply_log_level_callback() {
  llama_log_set(webgpu_log_callback, nullptr);
  ggml_log_set(webgpu_log_callback, nullptr);
}

void clear_pending_media() {
  for (pending_media & media : g_pending_media) {
    if (media.bitmap != nullptr) {
      mtmd_bitmap_free(media.bitmap);
    }
    if (media.video_ctx != nullptr) {
      mtmd_helper_video_free(media.video_ctx);
    }
  }
  g_pending_media.clear();
}

void ensure_backend_initialized() {
  if (!g_backend_initialized) {
    llama_backend_init();
    g_backend_initialized = true;
  }

  apply_log_level_callback();
}

// Frees the speculative session and its draft context, and turns off the
// target-context extraction that a draft implementation switched on.
void free_speculative_session() {
  speculative_session * session = g_speculative;
  if (session == nullptr) {
    return;
  }
  g_speculative = nullptr;

  if (session->spec != nullptr) {
    common_speculative_free(session->spec);
  }
  if (session->ctx_dft != nullptr) {
    llama_free(session->ctx_dft);
    if (g_state.ctx != nullptr) {
      llama_set_embeddings_nextn(g_state.ctx, false, false);
      if (session->draft_model != nullptr && g_state.model != nullptr) {
        const int32_t * layer_ids = llama_model_target_layer_ids(session->draft_model);
        const uint32_t n_layer_ids = llama_model_target_layer_ids_n(session->draft_model);
        const int32_t n_layer = llama_model_n_layer(g_state.model);
        for (uint32_t i = 0; i < n_layer_ids; ++i) {
          if (layer_ids[i] >= 0 && layer_ids[i] <= n_layer) {
            llama_set_embeddings_layer_inp(
                g_state.ctx, static_cast<uint32_t>(layer_ids[i]), false);
          }
        }
      }
    }
  }
  if (session->batch.token != nullptr) {
    llama_batch_free(session->batch);
  }
  delete session;
}

void free_draft_model() {
  free_speculative_session();
  if (g_draft_model != nullptr) {
    llama_model_free(g_draft_model);
    g_draft_model = nullptr;
  }
  g_draft_model_architecture.clear();
  g_draft_model_block_type = COMMON_SPECULATIVE_TYPE_NONE;
  g_draft_model_json = "{}";
}

void end_generation_state() {
  free_speculative_session();
  g_active_reasoning_budget = nullptr;
  if (g_active_sampler != nullptr) {
    llama_sampler_free(g_active_sampler);
    g_active_sampler = nullptr;
  }
  if (g_active_grammar != nullptr) {
    llama_sampler_free(g_active_grammar);
    g_active_grammar = nullptr;
  }
  g_generation_active = false;
  g_last_piece.clear();
  g_cancel_requested = false;
  g_generation_has_qwen3_asr_audio = false;
  g_qwen3_asr_prefix_seed_attempted = false;
  g_qwen3_asr_prefix_tokens.clear();
  g_qwen3_asr_prefix_token_index = 0;
}

void free_tts() {
  if (g_tts != nullptr) {
    llama_webgpu_tts_free(g_tts);
    g_tts = nullptr;
  }
  g_tts_active = false;
  g_tts_info_json = "{}";
  g_tts_progress_json = "{}";
}

void free_decision_heads() {
  for (auto & entry : g_decision_heads) {
    llama_webgpu_decision_head_free(entry.second);
  }
  g_decision_heads.clear();
  g_decision_json = "{}";
}

void free_lora_adapters() {
  for (auto & entry : g_lora_adapters) {
    llama_adapter_lora_free(entry.second);
  }
  g_lora_adapters.clear();
  g_active_loras.clear();
}

bool read_binary_file(
    const char * path,
    std::vector<unsigned char> & output) {
  output.clear();
  if (path == nullptr || path[0] == '\0') {
    return true;
  }
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    return false;
  }
  const std::streampos end = input.tellg();
  if (end <= 0) {
    return false;
  }
  const auto size = static_cast<uint64_t>(end);
  if (size > std::numeric_limits<size_t>::max()) {
    return false;
  }
  output.resize(static_cast<size_t>(size));
  input.seekg(0, std::ios::beg);
  return input.read(
      reinterpret_cast<char *>(output.data()),
      static_cast<std::streamsize>(output.size())).good();
}

void refresh_tts_progress_json(
    const llama_webgpu_tts_progress & progress) {
  g_tts_progress_json =
      "{\"state\":" + std::to_string(progress.state) +
      ",\"promptTokensRemaining\":" +
      std::to_string(progress.prompt_tokens_remaining) +
      ",\"framesGenerated\":" +
      std::to_string(progress.frames_generated) +
      ",\"truncated\":" + (progress.truncated ? "true" : "false") +
      "}";
}

void free_runtime() {
  end_generation_state();
  g_sampler_candidates.clear();
  g_sampler_candidates.shrink_to_fit();
  free_tts();
  // Heads hold contexts on the model, so they go before llama_model_free.
  free_decision_heads();
  free_draft_model();
  g_next_speculative = speculative_request();

  clear_pending_media();

  if (g_state.mm_ctx != nullptr) {
    mtmd_free(g_state.mm_ctx);
    g_state.mm_ctx = nullptr;
  }

  if (g_state.ctx != nullptr) {
    llama_free(g_state.ctx);
    g_state.ctx = nullptr;
  }

  // The context applies these adapters and llama_model_free deletes any still
  // registered, so they are freed between the two.
  free_lora_adapters();

  if (g_state.model != nullptr) {
    llama_model_free(g_state.model);
    g_state.model = nullptr;
  }

  g_state.vocab = nullptr;
  g_state.n_ctx = 0;
  g_last_output.clear();
  g_last_piece.clear();
  g_last_tokens_json = "[]";
  g_last_detokenized.clear();
  g_last_embedding_json = "[]";
  g_last_next_token_scores_json = "{}";
  g_model_meta_json = "{}";
  g_model_uses_gpu_ops = false;
  g_model_loaded_mtp = false;
  g_model_params = llama_model_default_params();
  g_context_params = llama_context_default_params();
  g_model_is_qwen3_asr = false;
  g_cached_prompt_tokens.clear();
}
