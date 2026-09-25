// Exports: multimodal projector and pending media.
// Part of llama_webgpu_core.cpp, included inside its extern "C" block;
// not a standalone translation unit.

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_mmproj_load(
    const char * mmproj_path) {
  clear_error();

  if (!ensure_loaded()) {
    return -1;
  }

  if (mmproj_path == nullptr || std::strlen(mmproj_path) == 0) {
    set_error("Multimodal projector path is empty");
    return -2;
  }

  if (g_generation_active || g_tts_active) {
    set_error("Multimodal projector cannot be replaced during active generation or text-to-speech synthesis");
    return -4;
  }

  clear_pending_media();
  free_tts();

  if (g_state.mm_ctx != nullptr) {
    mtmd_free(g_state.mm_ctx);
    g_state.mm_ctx = nullptr;
  }

  mtmd_context_params params = mtmd_context_params_default();
  params.use_gpu = g_model_uses_gpu_ops;
  params.print_timings = false;
  params.n_threads = llama_n_threads(g_state.ctx);
  if (params.n_threads <= 0) {
    params.n_threads = 1;
  }

  g_state.mm_ctx = mtmd_init_from_file(mmproj_path, g_state.model, params);
  if (g_state.mm_ctx == nullptr) {
    set_error("Failed to load multimodal projector");
    return -3;
  }

  return 0;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_mmproj_free() {
  clear_error();
  if (g_generation_active || g_tts_active) {
    set_error("Multimodal projector cannot be unloaded during active generation or text-to-speech synthesis");
    return -1;
  }

  clear_pending_media();
  free_tts();

  if (g_state.mm_ctx != nullptr) {
    mtmd_free(g_state.mm_ctx);
    g_state.mm_ctx = nullptr;
  }
  return 0;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_mmproj_supports_vision() {
  if (g_state.mm_ctx == nullptr) {
    return 0;
  }
  return mtmd_support_vision(g_state.mm_ctx) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_mmproj_supports_audio() {
  if (g_state.mm_ctx == nullptr) {
    return 0;
  }
  return mtmd_support_audio(g_state.mm_ctx) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE void llamadart_webgpu_media_clear_pending() {
  clear_error();
  clear_pending_media();
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_media_add_file(
    const char * media_path) {
  clear_error();

  if (!ensure_loaded()) {
    return -1;
  }

  if (g_state.mm_ctx == nullptr) {
    set_error("Multimodal projector is not loaded");
    return -2;
  }

  if (media_path == nullptr || std::strlen(media_path) == 0) {
    set_error("Media file path is empty");
    return -3;
  }

  mtmd_helper_bitmap_wrapper media =
      llama_webgpu_bitmap_from_file(g_state.mm_ctx, media_path);
  if (media.bitmap == nullptr) {
    set_error("Failed to decode media file content");
    return -4;
  }

  g_pending_media.push_back({media.bitmap, media.video_ctx});
  return 0;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_media_add_encoded(
    const uint8_t * bytes,
    int32_t length) {
  clear_error();

  if (!ensure_loaded()) {
    return -1;
  }

  if (g_state.mm_ctx == nullptr) {
    set_error("Multimodal projector is not loaded");
    return -2;
  }

  if (bytes == nullptr || length <= 0) {
    set_error("Encoded media bytes are empty");
    return -3;
  }

  mtmd_helper_bitmap_wrapper media = llama_webgpu_bitmap_from_buffer(
      g_state.mm_ctx,
      bytes,
      static_cast<size_t>(length));
  if (media.bitmap == nullptr) {
    set_error("Failed to decode encoded media bytes");
    return -4;
  }

  g_pending_media.push_back({media.bitmap, media.video_ctx});
  return 0;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_media_add_rgb(
    uint32_t width,
    uint32_t height,
    const uint8_t * bytes,
    int32_t length) {
  clear_error();

  if (!ensure_loaded()) {
    return -1;
  }

  if (g_state.mm_ctx == nullptr) {
    set_error("Multimodal projector is not loaded");
    return -2;
  }

  if (width == 0 || height == 0 || bytes == nullptr || length <= 0) {
    set_error("Invalid raw RGB media payload");
    return -3;
  }

  const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
  if (expected != static_cast<size_t>(length)) {
    set_error("Raw RGB bytes do not match width*height*3");
    return -4;
  }

  mtmd_bitmap * bitmap = mtmd_bitmap_init(width, height, bytes);
  if (bitmap == nullptr) {
    set_error("Failed to initialize RGB media bitmap");
    return -5;
  }

  g_pending_media.push_back({bitmap, nullptr});
  return 0;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_media_add_audio_f32(
    const float * samples,
    int32_t sample_count) {
  clear_error();

  if (!ensure_loaded()) {
    return -1;
  }

  if (g_state.mm_ctx == nullptr) {
    set_error("Multimodal projector is not loaded");
    return -2;
  }

  if (samples == nullptr || sample_count <= 0) {
    set_error("Audio samples are empty");
    return -3;
  }

  mtmd_bitmap * bitmap =
      mtmd_bitmap_init_from_audio(static_cast<size_t>(sample_count), samples);
  if (bitmap == nullptr) {
    set_error("Failed to initialize audio bitmap");
    return -4;
  }

  g_pending_media.push_back({bitmap, nullptr});
  return 0;
}
