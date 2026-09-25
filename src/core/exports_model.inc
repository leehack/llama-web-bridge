// Exports: model load and free.
// Part of llama_webgpu_core.cpp, included inside its extern "C" block;
// not a standalone translation unit.

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_load_model(
    const char * model_path,
    int32_t n_ctx,
    int32_t n_threads,
    int32_t n_threads_batch,
    int32_t n_batch,
    int32_t n_ubatch,
    int32_t n_gpu_layers,
    int32_t n_seq_max,
    int32_t use_mmap,
    int32_t use_mlock,
    int32_t flash_attn_type,
    int32_t type_k,
    int32_t type_v,
    int32_t kv_unified,
    double rope_freq_base,
    double rope_freq_scale,
    int32_t split_mode,
    int32_t main_gpu) {
  clear_error();
  g_last_output.clear();
  g_cancel_requested = false;

  if (model_path == nullptr || std::strlen(model_path) == 0) {
    set_error("Model path is empty");
    return -1;
  }

  if (g_generation_active || g_tts_active) {
    set_error("Model cannot be replaced during active generation or text-to-speech synthesis");
    return -2;
  }

  free_runtime();

  return load_model_internal(
      model_path,
      n_ctx,
      n_threads,
      n_threads_batch,
      n_batch,
      n_ubatch,
      n_gpu_layers,
      n_seq_max,
      use_mmap != 0,
      use_mlock != 0,
      flash_attn_type,
      type_k,
      type_v,
      kv_unified,
      rope_freq_base,
      rope_freq_scale,
      split_mode,
      main_gpu);
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_free_model() {
  clear_error();
  if (g_generation_active || g_tts_active) {
    set_error("Model cannot be released during active generation or text-to-speech synthesis");
    return -1;
  }

  free_runtime();
  return 0;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_load_model_from_url(
    const char * model_url,
    int32_t n_ctx,
    int32_t n_threads,
    int32_t n_threads_batch,
    int32_t n_batch,
    int32_t n_ubatch,
    int32_t n_gpu_layers,
    int32_t chunk_size,
    int32_t n_seq_max,
    int32_t use_mmap,
    int32_t use_mlock,
    int32_t flash_attn_type,
    int32_t type_k,
    int32_t type_v,
    int32_t kv_unified,
    double rope_freq_base,
    double rope_freq_scale,
    int32_t split_mode,
    int32_t main_gpu) {
  clear_error();
  g_last_output.clear();
  g_cancel_requested = false;

  if (model_url == nullptr || std::strlen(model_url) == 0) {
    set_error("Model URL is empty");
    return -1;
  }

  if (g_generation_active || g_tts_active) {
    set_error("Model cannot be replaced during active generation or text-to-speech synthesis");
    return -2;
  }

  free_runtime();

  if (emscripten_is_main_browser_thread()) {
    set_error(
        "Fetch backend requires a worker-thread bridge runtime on this build");
    return -4;
  }

  const uint32_t effective_chunk_size = chunk_size > 0
      ? static_cast<uint32_t>(chunk_size)
      : static_cast<uint32_t>(16 * 1024 * 1024);

  std::string model_url_text(model_url);
  std::string base_url = model_url_text;
  std::string fetch_file_path;

  const size_t last_slash = model_url_text.rfind('/');
  if (last_slash != std::string::npos) {
    base_url = model_url_text.substr(0, last_slash);
    fetch_file_path = model_url_text.substr(last_slash + 1);
  }

  if (fetch_file_path.empty()) {
    static uint64_t fallback_file_counter = 0;
    fetch_file_path =
        "remote_model_" + std::to_string(++fallback_file_counter) + ".gguf";
  }

  backend_t fetch_backend =
      wasmfs_create_fetch_backend(base_url.c_str(), effective_chunk_size);
  if (fetch_backend == nullptr) {
    set_error("Failed to initialize fetch-backed model loader");
    return -5;
  }

  int fd = wasmfs_create_file(
      const_cast<char *>(fetch_file_path.c_str()),
      0444,
      fetch_backend);
  if (fd < 0) {
    set_error(
        "Failed to create fetch-backed model file (errno="
        + std::to_string(-fd) + ")");
    return -6;
  }
  close(fd);

  const int32_t rc =
      load_model_internal(
          fetch_file_path.c_str(),
          n_ctx,
          n_threads,
          n_threads_batch,
          n_batch,
          n_ubatch,
          n_gpu_layers,
          n_seq_max,
          use_mmap != 0,
          use_mlock != 0,
          flash_attn_type,
          type_k,
          type_v,
          kv_unified,
          rope_freq_base,
          rope_freq_scale,
          split_mode,
          main_gpu);
  if (unlink(fetch_file_path.c_str()) != 0 && errno != ENOENT) {
    // best-effort cleanup of temporary fetch path
  }

  return rc;
}
