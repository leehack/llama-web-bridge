// Multimodal prompt markers and media decoding.
// Part of llama_webgpu_core.cpp, included inside its anonymous namespace;
// not a standalone translation unit.

std::string normalize_media_markers(const std::string & prompt, const size_t media_count) {
  const char * marker_ptr = mtmd_default_marker();
  const std::string marker = marker_ptr == nullptr
      ? std::string("<__media__>")
      : std::string(marker_ptr);

  std::string normalized = prompt;

  replace_all_inplace(normalized, "<image>", marker);
  replace_all_inplace(normalized, "[IMG]", marker);
  replace_all_inplace(normalized, "<|image|>", marker);
  replace_all_inplace(normalized, "<img>", marker);
  replace_all_inplace(normalized, "<|img|>", marker);
  replace_all_inplace(
      normalized,
      "<|vision_start|><|image_pad|><|vision_end|>",
      marker);
  replace_all_inplace(
      normalized,
      "<|vision_start|><|video_pad|><|vision_end|>",
      marker);
  replace_all_inplace(normalized, "<audio>", marker);
  replace_all_inplace(normalized, "<|audio|>", marker);

  normalized = std::regex_replace(normalized, std::regex("<\\|image_\\d+\\|>"), marker);
  normalized = std::regex_replace(normalized, std::regex("<\\|audio_\\d+\\|>"), marker);

  if (media_count == 0) {
    return normalized;
  }

  const size_t marker_count = count_occurrences(normalized, marker);
  if (marker_count >= media_count) {
    return normalized;
  }

  const size_t missing = media_count - marker_count;
  std::string marker_block;
  for (size_t i = 0; i < missing; ++i) {
    if (!marker_block.empty()) {
      marker_block += ' ';
    }
    marker_block += marker;
  }

  const size_t user_cap_pos = normalized.find("User:");
  if (user_cap_pos != std::string::npos) {
    normalized.replace(user_cap_pos, 5, "User: " + marker_block + " ");
    return normalized;
  }

  const size_t user_pos = normalized.find("user:");
  if (user_pos != std::string::npos) {
    normalized.replace(user_pos, 5, "user: " + marker_block + " ");
    return normalized;
  }

  return marker_block + "\n" + normalized;
}

bool decode_multimodal_prompt(const std::string & prompt, llama_pos * n_past) {
  if (g_state.mm_ctx == nullptr) {
    set_error(
        "Multimodal projector is not loaded. Call loadMultimodalProjector first.");
    clear_pending_media();
    return false;
  }

  mtmd_input_chunks * chunks = mtmd_input_chunks_init();
  if (chunks == nullptr) {
    set_error("Failed to allocate multimodal input chunks");
    clear_pending_media();
    return false;
  }

  const std::string normalized_prompt =
      normalize_media_markers(prompt, g_pending_media.size());

  mtmd_input_text input_text {};
  input_text.text = normalized_prompt.c_str();
  // llama.cpp b9979 stopped deriving input length from the C string.
  input_text.text_len = normalized_prompt.size();
  const llama_token bos = llama_vocab_bos(g_state.vocab);
  const llama_token eos = llama_vocab_eos(g_state.vocab);
  input_text.add_special = bos != eos && bos != -1;
  input_text.parse_special = true;

  std::vector<const mtmd_bitmap *> bitmaps;
  bitmaps.reserve(g_pending_media.size());
  for (const pending_media & media : g_pending_media) {
    bitmaps.push_back(media.bitmap);
  }

  const int32_t tokenize_rc = mtmd_tokenize(
      g_state.mm_ctx,
      chunks,
      &input_text,
      bitmaps.data(),
      bitmaps.size());

  if (tokenize_rc != 0) {
    if (tokenize_rc == 1) {
      set_error(
          "Multimodal marker count does not match number of provided media parts");
    } else if (tokenize_rc == 2) {
      set_error("Failed to preprocess multimodal media content");
    } else {
      set_error("mtmd_tokenize failed while processing multimodal prompt");
    }
    mtmd_input_chunks_free(chunks);
    clear_pending_media();
    return false;
  }

  llama_pos new_n_past = 0;
  int32_t n_batch = static_cast<int32_t>(llama_n_batch(g_state.ctx));
  if (n_batch <= 0) {
    n_batch = 512;
  }

  const int32_t eval_rc = mtmd_helper_eval_chunks(
      g_state.mm_ctx,
      g_state.ctx,
      chunks,
      0,
      0,
      n_batch,
      true,
      &new_n_past);

  mtmd_input_chunks_free(chunks);
  clear_pending_media();

  if (eval_rc != 0) {
    set_error("mtmd_helper_eval_chunks failed while ingesting multimodal prompt");
    return false;
  }

  *n_past = new_n_past;
  return true;
}
