// Versioned parts embedding API. The legacy string export stays unchanged.
// Part of llama_webgpu_core.cpp, included inside its extern "C" block.

EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_embedding_capabilities_json() {
  static std::string json;
  const bool text = llamadart_webgpu_detail::supports_pooled_embeddings(g_state.ctx, g_state.model);
  bool image = false;
  bool audio = false;
#if LLAMADART_WEBGPU_HAS_MIXED_EMBEDDINGS
  if (g_state.mm_ctx && llamadart_webgpu_detail::supports_mixed_embeddings(g_state.ctx, g_state.model)) {
    image = mtmd_support_vision(g_state.mm_ctx);
    audio = mtmd_support_audio(g_state.mm_ctx) && mtmd_get_audio_sample_rate(g_state.mm_ctx) == 16000;
  }
#endif
  json = std::string("{\"apiVersion\":1,\"text\":") + (text ? "true" : "false") +
      ",\"image\":" + (image ? "true" : "false") + ",\"audio\":" + (audio ? "true" : "false") + "}";
  return json.c_str();
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_embed_parts_to_json(const char * prompt, int32_t normalize, int32_t text_length) {
  clear_error();
  g_last_embedding_json = "[]";
  struct media_cleanup { ~media_cleanup() { clear_pending_media(); } } cleanup;
  if (!ensure_loaded() || g_generation_active || g_tts_active || !prompt || text_length < 0) return -25;
  g_cancel_requested = false;
  struct cancel_cleanup { ~cancel_cleanup() { g_cancel_requested = false; } } cancel_reset;
  g_cached_prompt_tokens.clear();
  std::vector<const mtmd_bitmap *> bitmaps;
  for (const auto & media : g_pending_media) bitmaps.push_back(media.bitmap);
  std::vector<float> output;
  const int rc = llamadart_webgpu_detail::embed_parts(g_state.ctx, g_state.model, g_state.mm_ctx,
      std::string(prompt, static_cast<size_t>(text_length)), bitmaps, output);
  if (rc < 0) return rc;
  if (normalize != 0) normalize_embedding_inplace(output);
  g_last_embedding_json = llamadart_webgpu_detail::serialize_embedding_json(output);
  return rc;
}
