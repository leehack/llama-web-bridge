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
    int32_t main_gpu,
    int32_t load_mtp,
    int32_t n_rs_seq) {
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
      main_gpu,
      load_mtp != 0,
      n_rs_seq);
}

// Test hook for the staging smoke: the next model load is cancelled after
// llama.cpp has mapped the file. Not part of the API.
EMSCRIPTEN_KEEPALIVE void llamadart_webgpu_test_fail_next_model_load() {
  g_test_fail_next_model_load = true;
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
    int32_t main_gpu,
    int32_t load_mtp,
    int32_t n_rs_seq) {
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

#if !defined(__EMSCRIPTEN_PTHREADS__)
  set_error("Fetch-backed model loading requires a core built with pthreads");
  return -7;
#endif

  free_runtime();

  if (emscripten_is_main_browser_thread()) {
    set_error(
        "Fetch backend requires a worker-thread bridge runtime on this build");
    return -4;
  }

  const uint32_t effective_chunk_size = chunk_size > 0
      ? static_cast<uint32_t>(chunk_size)
      : static_cast<uint32_t>(16 * 1024 * 1024);

  // A single-file WasmFS mount fetches the backend URL verbatim.
  backend_t fetch_backend =
      wasmfs_create_fetch_backend(model_url, effective_chunk_size);
  if (fetch_backend == nullptr) {
    set_error("Failed to initialize fetch-backed model loader");
    return -5;
  }

  // Keep signed URLs out of the local filename and reuse the backend's proxy
  // worker if an application file already occupies a generated path.
  static uint64_t fetch_file_counter = 0;
  std::string fetch_file_path;
  int fd;
  do {
    fetch_file_path =
        "/remote_model_" + std::to_string(++fetch_file_counter) + ".gguf";
    fd = wasmfs_create_file(fetch_file_path.c_str(), 0444, fetch_backend);
  } while (fd == -EEXIST);
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
          main_gpu,
          load_mtp != 0,
          n_rs_seq);
  if (unlink(fetch_file_path.c_str()) != 0 && errno != ENOENT) {
    // best-effort cleanup of temporary fetch path
  }

  return rc;
}
