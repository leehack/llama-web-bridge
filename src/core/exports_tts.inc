// Exports: text-to-speech.
// Part of llama_webgpu_core.cpp, included inside its extern "C" block;
// not a standalone translation unit.

EMSCRIPTEN_KEEPALIVE uint32_t llamadart_webgpu_tts_api_version() {
  return LLAMADART_WEBGPU_TTS_API_VERSION;
}

EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_tts_info_json() {
  const llama_webgpu_tts_info info =
      llama_webgpu_tts_get_info(g_state.mm_ctx);
  const bool supported =
      g_state.ctx != nullptr && g_state.mm_ctx != nullptr &&
      info.model_type == LLAMADART_WEBGPU_TTS_MODEL_TYPE_QWEN3;
  std::string reason;
  if (g_state.ctx == nullptr) {
    reason = "Model is not loaded";
  } else if (g_state.mm_ctx == nullptr) {
    reason = "Multimodal projector is not loaded";
  } else if (!supported) {
    reason = "Loaded projector does not support Qwen3 text-to-speech";
  }
  g_tts_info_json =
      "{\"apiVersion\":" + std::to_string(info.api_version) +
      ",\"supported\":" + (supported ? "true" : "false") +
      ",\"modelType\":" + std::to_string(info.model_type) +
      ",\"capabilities\":" + std::to_string(info.capabilities) +
      ",\"supportsLanguage\":" +
      ((info.capabilities & LLAMADART_WEBGPU_TTS_CAPABILITY_LANGUAGE) != 0
           ? "true"
           : "false") +
      ",\"supportsSpeakerReference\":" +
      ((info.capabilities &
        LLAMADART_WEBGPU_TTS_CAPABILITY_SPEAKER_REFERENCE) != 0
           ? "true"
           : "false") +
      ",\"sampleRate\":" + std::to_string(info.sample_rate) +
      ",\"channels\":" + std::to_string(info.channels) +
      ",\"reason\":\"" + escape_json(reason) + "\"}";
  return g_tts_info_json.c_str();
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_tts_start(
    const char * text,
    const char * language,
    const char * speaker_path,
    int32_t prompt_batch_size,
    int32_t max_frames,
    int32_t top_k,
    float top_p,
    float min_p,
    float temperature,
    uint32_t seed) {
  clear_error();
  if (!ensure_loaded()) {
    return LLAMADART_WEBGPU_TTS_STATUS_INVALID_STATE;
  }
  if (g_generation_active) {
    set_error("Token generation is active");
    return LLAMADART_WEBGPU_TTS_STATUS_INVALID_STATE;
  }
  if (g_state.mm_ctx == nullptr) {
    set_error("Multimodal projector is not loaded");
    return LLAMADART_WEBGPU_TTS_STATUS_UNSUPPORTED;
  }
  if (g_tts == nullptr) {
    llama_webgpu_tts_status init_status;
    g_tts = llama_webgpu_tts_init(
        g_state.ctx, g_state.mm_ctx, &init_status);
    if (g_tts == nullptr) {
      set_error(init_status == LLAMADART_WEBGPU_TTS_STATUS_UNSUPPORTED
          ? "Loaded projector does not support Qwen3 text-to-speech"
          : "Text-to-speech task initialization failed");
      return init_status;
    }
  }

  std::vector<unsigned char> speaker_audio;
  if (!read_binary_file(speaker_path, speaker_audio)) {
    set_error("Speaker reference audio could not be read");
    return LLAMADART_WEBGPU_TTS_STATUS_SPEAKER_DECODE_FAILED;
  }

  const llama_webgpu_tts_request request{
      text,
      text != nullptr ? std::strlen(text) : 0,
      speaker_audio.empty() ? nullptr : speaker_audio.data(),
      speaker_audio.size(),
      language,
      0,
      prompt_batch_size,
      max_frames,
      top_k,
      top_p,
      min_p,
      temperature,
      seed,
  };
  // Cancellation is shared with token generation through the llama.cpp abort
  // callback. A completed TTS cancellation must not poison the next task.
  g_cancel_requested = false;
  // TTS currently shares sequence 0 with normal generation. Its upstream
  // helper clears that sequence before loading the speech prompt, so the
  // cached-token mirror must be invalidated before any later chat generation
  // attempts prefix reuse. Using a second sequence would require every caller
  // to allocate n_seq_max >= 2 and would reduce the per-sequence context size.
  g_cached_prompt_tokens.clear();
  const llama_webgpu_tts_status status =
      llama_webgpu_tts_start(g_tts, request);
  if (status != LLAMADART_WEBGPU_TTS_STATUS_OK) {
    set_error(llama_webgpu_tts_last_error(g_tts));
    g_tts_active = false;
    return status;
  }
  g_tts_active = true;
  g_tts_progress_json =
      "{\"state\":1,\"promptTokensRemaining\":0,"
      "\"framesGenerated\":0,\"truncated\":false}";
  return LLAMADART_WEBGPU_TTS_STATUS_OK;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_tts_step() {
  clear_error();
  if (g_tts == nullptr || !g_tts_active) {
    set_error("Text-to-speech synthesis is not active");
    return LLAMADART_WEBGPU_TTS_STATUS_INVALID_STATE;
  }
  llama_webgpu_tts_progress progress{};
  const llama_webgpu_tts_status status =
      llama_webgpu_tts_step(g_tts, &progress);
  refresh_tts_progress_json(progress);
  if (status != LLAMADART_WEBGPU_TTS_STATUS_OK) {
    set_error(llama_webgpu_tts_last_error(g_tts));
  }
  if (progress.state == LLAMADART_WEBGPU_TTS_STATE_COMPLETED ||
      progress.state == LLAMADART_WEBGPU_TTS_STATE_CANCELLED ||
      progress.state == LLAMADART_WEBGPU_TTS_STATE_FAILED) {
    g_tts_active = false;
  }
  return status;
}

EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_tts_progress_json() {
  return g_tts_progress_json.c_str();
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_tts_write_pcm(
    const char * output_path) {
  clear_error();
  const llama_webgpu_tts_status status =
      llama_webgpu_tts_write_pcm(g_tts, output_path);
  if (status != LLAMADART_WEBGPU_TTS_STATUS_OK) {
    set_error(g_tts != nullptr
        ? llama_webgpu_tts_last_error(g_tts)
        : "Text-to-speech task is unavailable");
  }
  return status;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_tts_reset() {
  clear_error();
  g_cancel_requested = false;
  if (g_tts == nullptr) {
    g_tts_active = false;
    g_tts_progress_json = "{}";
    return LLAMADART_WEBGPU_TTS_STATUS_OK;
  }
  const llama_webgpu_tts_status status = llama_webgpu_tts_reset(g_tts);
  g_tts_active = false;
  g_tts_progress_json = "{}";
  if (status != LLAMADART_WEBGPU_TTS_STATUS_OK) {
    set_error(llama_webgpu_tts_last_error(g_tts));
  }
  return status;
}
