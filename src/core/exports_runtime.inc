// Exports: backend probing, errors, and log level.
// Part of llama_webgpu_core.cpp, included inside its extern "C" block;
// not a standalone translation unit.

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_probe() {
  refresh_backend_probe();
  return g_has_webgpu ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_supports_pthreads() {
#if defined(__EMSCRIPTEN_PTHREADS__)
  return 1;
#else
  return 0;
#endif
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_pthread_pool_size() {
#if defined(__EMSCRIPTEN_PTHREADS__)
  return LLAMADART_WEBGPU_PTHREAD_POOL_SIZE > 0
      ? LLAMADART_WEBGPU_PTHREAD_POOL_SIZE
      : 1;
#else
  return 1;
#endif
}

EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_backends_json() {
  refresh_backend_probe();
  return g_backend_json.c_str();
}

EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_last_error() {
  return g_last_error.c_str();
}

EMSCRIPTEN_KEEPALIVE void llamadart_webgpu_set_log_level(int32_t level) {
  if (level < 0) {
    level = 0;
  } else if (level > 4) {
    level = 4;
  }

  g_log_level.store(level, std::memory_order_relaxed);
  g_last_non_cont_level.store(GGML_LOG_LEVEL_NONE, std::memory_order_relaxed);
  apply_log_level_callback();
}
