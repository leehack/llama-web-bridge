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

// The *_bytes exports take text as UTF-8 bytes and a length, because a C
// string ends at the first U+0000. The older C-string exports pass strlen.
bool text_from_bytes(
    const char * bytes,
    const int32_t length,
    const char * label,
    std::string & out) {
  if (bytes == nullptr) {
    set_error(std::string(label) + " is null");
    return false;
  }
  if (length < 0) {
    set_error(std::string(label) + " length is negative");
    return false;
  }
  out.assign(bytes, static_cast<size_t>(length));
  return true;
}

int32_t c_string_length(const char * text) {
  return text == nullptr ? 0 : static_cast<int32_t>(std::strlen(text));
}

// llama.cpp's state file functions catch the exception llama_file throws when
// the open or a read fails, but exception catching is compiled out of llama.cpp
// here, so that throw would abort the core. Open the file the same way first.
// WASMFS opens a directory for reading, and the first read then throws.
bool can_open_state_file(const char * state_path, const char * mode) {
  struct stat info;
  const bool is_directory = stat(state_path, &info) == 0 && S_ISDIR(info.st_mode);
  FILE * file = is_directory ? nullptr : std::fopen(state_path, mode);
  if (file == nullptr) {
    set_error(
        std::string("Cannot open state file ") + state_path + ": "
        + std::strerror(is_directory ? EISDIR : errno));
    return false;
  }
  std::fclose(file);
  return true;
}

struct file_closer {
  void operator()(FILE * file) const { std::fclose(file); }
};

// Reads and checks what llama_state_save_file writes before the state: the
// magic, the version, the token count and the tokens, then the model
// architecture's name, which opens the llama_state_get_data bytes. A short
// header is rejected without touching the context, and so is a corrupt name
// length that llama.cpp would otherwise allocate before reading. The tokens
// land in `tokens`, which holds token_capacity of them.
// Returns 0, -5 with `rejection` set, or -7 with the error set.
int32_t read_state_snapshot_head(
    const char * state_path,
    std::vector<llama_token> & tokens,
    size_t & token_count,
    size_t & state_offset,
    size_t & file_size,
    std::string & rejection) {
  if (!can_open_state_file(state_path, "rb")) {
    return -7;
  }
  struct stat info;
  std::unique_ptr<FILE, file_closer> file(
      stat(state_path, &info) == 0 ? std::fopen(state_path, "rb") : nullptr);
  if (file == nullptr) {
    set_error(std::string("Cannot read state file ") + state_path + ": " + std::strerror(errno));
    return -7;
  }
  file_size = static_cast<size_t>(info.st_size);

  uint32_t header[3] = {};
  if (file_size < sizeof(header)) {
    rejection = "State snapshot is truncated: " + std::to_string(file_size)
        + " bytes is shorter than its " + std::to_string(sizeof(header)) + "-byte header";
    return -5;
  }
  const auto read = [&](void * dst, size_t size) {
    if (std::fread(dst, 1, size, file.get()) == size) {
      return true;
    }
    set_error(std::string("Cannot read state file ") + state_path + ": " + std::strerror(errno));
    return false;
  };
  if (!read(header, sizeof(header))) {
    return -7;
  }
  if (header[0] != LLAMA_SESSION_MAGIC || header[1] != LLAMA_SESSION_VERSION) {
    rejection = "State snapshot is not a llama.cpp session of version "
        + std::to_string(LLAMA_SESSION_VERSION);
    return -5;
  }
  token_count = header[2];
  if (token_count > tokens.size()) {
    rejection = "State snapshot holds " + std::to_string(token_count)
        + " tokens, more than tokenCapacity " + std::to_string(tokens.size());
    return -5;
  }
  state_offset = sizeof(header) + token_count * sizeof(llama_token);
  uint32_t arch_size = 0;
  if (file_size < state_offset + sizeof(arch_size)) {
    rejection = "State snapshot is truncated: " + std::to_string(file_size)
        + " bytes ends inside its token list or model architecture";
    return -5;
  }
  if (!read(tokens.data(), token_count * sizeof(llama_token)) || !read(&arch_size, sizeof(arch_size))) {
    return -7;
  }
  if (file_size - state_offset - sizeof(arch_size) < arch_size) {
    rejection = "State snapshot is truncated: " + std::to_string(file_size)
        + " bytes ends inside its token list or model architecture";
    return -5;
  }
  return 0;
}

struct free_deleter {
  void operator()(void * memory) const { std::free(memory); }
};
using malloc_buffer = std::unique_ptr<uint8_t, free_deleter>;

// The snapshot and current-state copies make a state load safer but are not
// needed for it, so they must never fail a load that would otherwise succeed.
// With memory growth (the default build) malloc returns null when the heap
// cannot grow. Without growth it aborts instead, so only the space above the
// heap's break that the heap already holds is used. A test-hook bit asks for
// more than malloc ever grants, so tests take the same null path.
malloc_buffer try_state_load_alloc(size_t size, int32_t test_failure) {
  if ((g_state_load_test_failures & test_failure) != 0) {
    size = std::numeric_limits<size_t>::max();
  }
#if !LLAMADART_WEBGPU_ALLOW_MEMORY_GROWTH
  constexpr size_t allocator_slack = 1 << 20;
  const size_t heap_size = emscripten_get_heap_size();
  const size_t heap_top = reinterpret_cast<uintptr_t>(sbrk(0));
  if (heap_size < heap_top || heap_size - heap_top < allocator_slack
      || heap_size - heap_top - allocator_slack < size) {
    return nullptr;
  }
#endif
  return malloc_buffer(static_cast<uint8_t *>(std::malloc(std::max<size_t>(size, 1))));
}

size_t add_saturating(size_t a, size_t b) {
  return a > std::numeric_limits<size_t>::max() - b ? std::numeric_limits<size_t>::max() : a + b;
}

size_t multiply_saturating(size_t a, size_t b) {
  return b != 0 && a > std::numeric_limits<size_t>::max() / b ? std::numeric_limits<size_t>::max() : a * b;
}

// Everything restoring a snapshot from memory holds or allocates beyond what
// llama.cpp reading the file does: both copies, llama.cpp's per-tensor-call
// bookkeeping while it copies the current state and while it restores either
// copy (one entry per layer and K row run or V row, at most 4 * n_embd V rows
// per layer, held in a doubling vector), the cell bookkeeping (bounded by the
// context size), and slack for allocator overhead.
size_t state_load_copy_budget(size_t snapshot_size, size_t current_size) {
  const size_t layers = static_cast<size_t>(std::max(1, llama_model_n_layer(g_state.model)));
  const size_t width = static_cast<size_t>(std::max(1, llama_model_n_embd(g_state.model)));
  const size_t entries = multiply_saturating(layers, add_saturating(multiply_saturating(width, 4), 2));
  const size_t bookkeeping = multiply_saturating(entries, 4 * sizeof(size_t) * 4);
  const size_t cells = multiply_saturating(llama_n_ctx(g_state.ctx), 128);
  constexpr size_t slack = 16u << 20;
  return add_saturating(
      add_saturating(snapshot_size, current_size),
      add_saturating(bookkeeping, add_saturating(cells, slack)));
}

// A block that is only tested and freed is removed by the optimizer, and the
// probe would always pass; storing it in a volatile keeps the malloc.
void * volatile g_state_load_probe = nullptr;

bool state_load_has_room(size_t size) {
  malloc_buffer probe = try_state_load_alloc(size, 4);
  g_state_load_probe = probe.get();
  const bool fits = g_state_load_probe != nullptr;
  g_state_load_probe = nullptr;
  return fits;
}

bool read_state_file_into(const char * state_path, uint8_t * bytes, size_t size) {
  std::unique_ptr<FILE, file_closer> file(std::fopen(state_path, "rb"));
  return file != nullptr && std::fread(bytes, 1, size, file.get()) == size;
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
  g_active_grammar_lazy = false;
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
  g_next_grammar = grammar_request();

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
