// Model loading.
// Part of llama_webgpu_core.cpp, included inside its anonymous namespace;
// not a standalone translation unit.

bool is_supported_kv_cache_type(int32_t value) {
  switch (value) {
    case GGML_TYPE_F16:
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q8_0:
      return true;
    default:
      return false;
  }
}

constexpr llama_load_mode resolve_load_mode(bool use_mmap, bool use_mlock) {
  if (use_mlock) {
    return LLAMA_LOAD_MODE_MLOCK;
  }
  return use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
}

static_assert(resolve_load_mode(false, false) == LLAMA_LOAD_MODE_NONE);
static_assert(resolve_load_mode(true, false) == LLAMA_LOAD_MODE_MMAP);
static_assert(resolve_load_mode(false, true) == LLAMA_LOAD_MODE_MLOCK);
static_assert(resolve_load_mode(true, true) == LLAMA_LOAD_MODE_MLOCK);

int32_t load_model_internal(
    const char * model_path,
    int32_t n_ctx,
    int32_t n_threads,
    int32_t n_threads_batch,
    int32_t n_batch,
    int32_t n_ubatch,
    int32_t n_gpu_layers,
    int32_t n_seq_max,
    bool use_mmap,
    bool use_mlock,
    int32_t flash_attn_type,
    int32_t type_k,
    int32_t type_v,
    int32_t kv_unified,
    double rope_freq_base,
    double rope_freq_scale,
    int32_t split_mode,
    int32_t main_gpu) {
  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = n_gpu_layers;
  mparams.load_mode = resolve_load_mode(use_mmap, use_mlock);
  mparams.vocab_only = false;
  if (split_mode >= LLAMA_SPLIT_MODE_NONE &&
      split_mode <= LLAMA_SPLIT_MODE_TENSOR) {
    mparams.split_mode = static_cast<llama_split_mode>(split_mode);
  }
  if (main_gpu >= 0) {
    mparams.main_gpu = main_gpu;
  }

  g_state.model = llama_model_load_from_file(model_path, mparams);
  if (g_state.model == nullptr) {
    set_error("llama_model_load_from_file failed");
    return -2;
  }

  llama_context_params cparams = llama_context_default_params();
  if (n_ctx > 0) {
    cparams.n_ctx = static_cast<uint32_t>(n_ctx);
  }

  if (n_seq_max > 0) {
    cparams.n_seq_max = static_cast<uint32_t>(n_seq_max);
  }

  if (n_threads > 0) {
    cparams.n_threads = n_threads;
  }

  if (n_threads_batch > 0) {
    cparams.n_threads_batch = n_threads_batch;
  } else if (n_threads > 0) {
    cparams.n_threads_batch = n_threads;
  }

  if (n_batch > 0) {
    cparams.n_batch = static_cast<uint32_t>(n_batch);
  }

  if (n_ubatch > 0) {
    cparams.n_ubatch = static_cast<uint32_t>(n_ubatch);
  }

  if (cparams.n_batch == 0 || cparams.n_batch > cparams.n_ctx) {
    cparams.n_batch = std::min<uint32_t>(cparams.n_ctx, 1024U);
  }

  if (cparams.n_ubatch == 0 || cparams.n_ubatch > cparams.n_batch) {
    cparams.n_ubatch = std::min<uint32_t>(cparams.n_batch, 512U);
  }

  if (flash_attn_type >= LLAMA_FLASH_ATTN_TYPE_AUTO &&
      flash_attn_type <= LLAMA_FLASH_ATTN_TYPE_ENABLED) {
    cparams.flash_attn_type =
        static_cast<llama_flash_attn_type>(flash_attn_type);
  }
  if (is_supported_kv_cache_type(type_k)) {
    cparams.type_k = static_cast<ggml_type>(type_k);
  }
  if (is_supported_kv_cache_type(type_v)) {
    cparams.type_v = static_cast<ggml_type>(type_v);
  }
  if (kv_unified >= 0) {
    cparams.kv_unified = kv_unified != 0;
  }
  if (rope_freq_base > 0.0) {
    cparams.rope_freq_base = static_cast<float>(rope_freq_base);
  }
  if (rope_freq_scale > 0.0) {
    cparams.rope_freq_scale = static_cast<float>(rope_freq_scale);
  }

  const bool enable_gpu_ops = n_gpu_layers > 0;
  g_model_uses_gpu_ops = enable_gpu_ops;
  cparams.offload_kqv = enable_gpu_ops;
  cparams.op_offload = enable_gpu_ops;
  cparams.no_perf = true;

  g_state.ctx = llama_init_from_model(g_state.model, cparams);
  if (g_state.ctx == nullptr) {
    set_error("llama_init_from_model failed");
    free_runtime();
    return -3;
  }

  g_state.vocab = llama_model_get_vocab(g_state.model);
  g_state.n_ctx = llama_n_ctx(g_state.ctx);
  llama_set_abort_callback(g_state.ctx, should_abort_callback, nullptr);

  rebuild_model_metadata_json();
  return 0;
}
