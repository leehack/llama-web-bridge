// Exports: tokenization, context state, embeddings, and next-token scores.
// Part of llama_webgpu_core.cpp, included inside its extern "C" block;
// not a standalone translation unit.

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_tokenize_to_json(
    const char * text,
    int32_t add_special) {
  clear_error();
  g_last_tokens_json = "[]";

  if (text == nullptr) {
    set_error("Text is null");
    return -1;
  }

  std::vector<llama_token> tokens;
  if (!tokenize_text(text, add_special != 0, tokens)) {
    return -2;
  }

  g_last_tokens_json = serialize_tokens_json(tokens);
  return static_cast<int32_t>(tokens.size());
}

EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_last_tokens_json() {
  return g_last_tokens_json.c_str();
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_state_save_file(
    const char * state_path,
    const char * token_text) {
  clear_error();

  if (!ensure_loaded()) {
    return -1;
  }

  if (g_generation_active || g_tts_active) {
    set_error("State cannot be saved or loaded during active generation or text-to-speech synthesis");
    return -2;
  }

  if (state_path == nullptr || std::strlen(state_path) == 0) {
    set_error("State file path is empty");
    return -3;
  }

  std::vector<llama_token> tokens;
  parse_token_list(token_text, tokens);

  llama_synchronize(g_state.ctx);
  const bool saved = llama_state_save_file(
      g_state.ctx,
      state_path,
      tokens.empty() ? nullptr : tokens.data(),
      tokens.size());
  if (!saved) {
    set_error("Failed to save llama.cpp state file");
    return -4;
  }

  return 0;
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_state_load_file(
    const char * state_path,
    int32_t token_capacity) {
  clear_error();
  g_last_tokens_json = "[]";

  if (!ensure_loaded()) {
    return -1;
  }

  if (g_generation_active || g_tts_active) {
    set_error("State cannot be saved or loaded during active generation or text-to-speech synthesis");
    return -2;
  }

  if (state_path == nullptr || std::strlen(state_path) == 0) {
    set_error("State file path is empty");
    return -3;
  }

  const int32_t n_ctx = static_cast<int32_t>(llama_n_ctx(g_state.ctx));
  if (token_capacity <= 0 || token_capacity > n_ctx) {
    set_error("State token capacity must be positive and not exceed context size");
    return -4;
  }

  std::vector<llama_token> restored_tokens(static_cast<size_t>(token_capacity));
  size_t restored_count = 0;
  llama_synchronize(g_state.ctx);
  const bool loaded = llama_state_load_file(
      g_state.ctx,
      state_path,
      restored_tokens.data(),
      restored_tokens.size(),
      &restored_count);
  if (!loaded) {
    llama_memory_clear(llama_get_memory(g_state.ctx), false);
    g_cached_prompt_tokens.clear();
    g_last_output.clear();
    g_last_piece.clear();
    g_last_detokenized.clear();
    set_error(
        "Failed to load llama.cpp state file. The file may be corrupt, from a different model/build, or larger than tokenCapacity.");
    return -5;
  }

  if (restored_count > restored_tokens.size()) {
    llama_memory_clear(llama_get_memory(g_state.ctx), false);
    g_cached_prompt_tokens.clear();
    g_last_output.clear();
    g_last_piece.clear();
    g_last_detokenized.clear();
    set_error("Loaded state token count exceeds token capacity");
    return -6;
  }

  restored_tokens.resize(restored_count);
  g_cached_prompt_tokens = restored_tokens;
  g_last_tokens_json = serialize_tokens_json(restored_tokens);
  g_last_output.clear();
  g_last_piece.clear();
  g_last_detokenized.clear();

  return static_cast<int32_t>(restored_count);
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_detokenize_from_json(
    const char * token_text,
    int32_t special) {
  clear_error();
  g_last_detokenized.clear();

  if (!ensure_loaded()) {
    return -1;
  }

  std::vector<llama_token> tokens;
  parse_token_list(token_text, tokens);

  if (tokens.empty()) {
    return 0;
  }

  g_last_detokenized.reserve(tokens.size() * 4);
  for (const llama_token token : tokens) {
    g_last_detokenized += token_to_piece(token, special != 0);
  }

  return static_cast<int32_t>(g_last_detokenized.size());
}

EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_last_detokenized() {
  return g_last_detokenized.c_str();
}

EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_embed_to_json(
    const char * text,
    int32_t normalize) {
  clear_error();
  g_last_embedding_json = "[]";

  if (!ensure_loaded()) {
    return -1;
  }

  if (g_generation_active || g_tts_active) {
    set_error("Embeddings cannot be generated during active generation or text-to-speech synthesis");
    return -10;
  }

  if (text == nullptr) {
    set_error("Text is null");
    return -2;
  }

  const bool has_encoder = llama_model_has_encoder(g_state.model);
  const bool has_decoder = llama_model_has_decoder(g_state.model);
  if (has_encoder && has_decoder) {
    set_error("Embedding extraction for encoder-decoder models is not supported");
    return -3;
  }
  const bool use_encoder_path = has_encoder && !has_decoder;

  std::vector<llama_token> tokens;
  if (!tokenize_text(std::string(text), true, tokens)) {
    return -4;
  }

  if (tokens.empty()) {
    set_error("Embedding input tokenized to an empty sequence");
    return -5;
  }

  int32_t embedding_size = llama_model_n_embd_out(g_state.model);
  if (embedding_size <= 0) {
    embedding_size = llama_model_n_embd(g_state.model);
  }
  if (embedding_size <= 0) {
    set_error("Failed to resolve embedding dimension");
    return -6;
  }

  int32_t max_batch = static_cast<int32_t>(llama_n_batch(g_state.ctx));
  if (max_batch <= 0) {
    max_batch = static_cast<int32_t>(tokens.size());
  }
  max_batch = std::max<int32_t>(1, std::min<int32_t>(max_batch, static_cast<int32_t>(tokens.size())));

  llama_batch batch = llama_batch_init(max_batch, 0, 1);
  if (batch.token == nullptr || batch.pos == nullptr ||
      batch.n_seq_id == nullptr || batch.seq_id == nullptr ||
      batch.logits == nullptr) {
    llama_batch_free(batch);
    set_error("Failed to allocate embedding batch buffers");
    return -7;
  }

  int32_t rc = embedding_size;

  llama_synchronize(g_state.ctx);
  auto * memory = llama_get_memory(g_state.ctx);
  if (memory != nullptr) {
    llama_memory_clear(memory, false);
  }
  g_cached_prompt_tokens.clear();
  llama_set_embeddings(g_state.ctx, true);

  int32_t decoded_tokens = 0;
  while (decoded_tokens < static_cast<int32_t>(tokens.size())) {
    const int32_t remaining = static_cast<int32_t>(tokens.size()) - decoded_tokens;
    const int32_t chunk_token_count = std::min(max_batch, remaining);
    batch.n_tokens = chunk_token_count;

    for (int32_t i = 0; i < chunk_token_count; ++i) {
      const int32_t token_index = decoded_tokens + i;
      batch.token[i] = tokens[static_cast<size_t>(token_index)];
      batch.pos[i] = token_index;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = 1;
    }

    const int status = use_encoder_path
        ? llama_encode(g_state.ctx, batch)
        : llama_decode(g_state.ctx, batch);
    if (status != 0) {
      set_error("Embedding forward pass failed");
      rc = -8;
      break;
    }

    decoded_tokens += chunk_token_count;
  }

  if (rc > 0) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(g_state.ctx);
    float * embedding_ptr = nullptr;
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
      embedding_ptr = llama_get_embeddings_ith(g_state.ctx, batch.n_tokens - 1);
      if (embedding_ptr == nullptr) {
        embedding_ptr = llama_get_embeddings(g_state.ctx);
      }
    } else {
      embedding_ptr = llama_get_embeddings_seq(g_state.ctx, 0);
      if (embedding_ptr == nullptr) {
        embedding_ptr = llama_get_embeddings(g_state.ctx);
      }
    }

    if (embedding_ptr == nullptr) {
      set_error("Embedding output is unavailable");
      rc = -9;
    } else {
      std::vector<float> embedding(
          embedding_ptr,
          embedding_ptr + static_cast<size_t>(embedding_size));
      if (normalize != 0) {
        normalize_embedding_inplace(embedding);
      }

      g_last_embedding_json =
          llamadart_webgpu_detail::serialize_embedding_json(embedding);
    }
  }

  {
    llama_set_embeddings(g_state.ctx, false);
    llama_batch_free(batch);
  }

  return rc;
}

EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_last_embedding_json() {
  return g_last_embedding_json.c_str();
}

// Scores the position after prompt with a softmax over the raw logits, like
// llama-server n_probs; sampling settings do not apply. -3 means a candidate
// or top_k is outside the vocabulary.
EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_score_next_token_to_json(
    const char * prompt,
    const char * candidates_json,
    int32_t top_k,
    int32_t reuse_prompt_prefix) {
  clear_error();
  g_last_next_token_scores_json = "{}";

  if (!ensure_loaded()) {
    return -1;
  }

  if (g_generation_active || g_tts_active) {
    set_error("Next-token scoring cannot run during active generation or text-to-speech synthesis");
    return -10;
  }

  if (prompt == nullptr) {
    set_error("Prompt is null");
    return -2;
  }

  if (!llama_model_has_decoder(g_state.model) ||
      llama_model_has_encoder(g_state.model)) {
    set_error("Next-token scoring needs a decoder-only model");
    return -5;
  }

  if (!g_pending_media.empty()) {
    set_error("Next-token scoring does not accept media; clear pending media first");
    return -4;
  }

  const int32_t n_vocab = llama_vocab_n_tokens(g_state.vocab);
  std::vector<llama_token> candidates;
  parse_token_list(candidates_json, candidates);
  for (const llama_token token : candidates) {
    if (token < 0 || token >= n_vocab) {
      set_error(
          "Token id " + std::to_string(token) +
          " is outside the vocabulary of " + std::to_string(n_vocab) +
          " tokens");
      return -3;
    }
  }
  if (top_k < 0 || top_k > n_vocab) {
    set_error(
        "topK " + std::to_string(top_k) +
        " is outside the vocabulary of " + std::to_string(n_vocab) +
        " tokens");
    return -3;
  }
  if (candidates.empty() && top_k == 0) {
    set_error("Pass candidates, a positive topK, or both");
    return -4;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_text(std::string(prompt), true, prompt_tokens)) {
    return -6;
  }
  if (prompt_tokens.empty()) {
    set_error("Prompt tokenized to an empty sequence");
    return -6;
  }
  const uint32_t n_ctx = llama_n_ctx(g_state.ctx);
  if (prompt_tokens.size() > n_ctx) {
    set_error(
        "Prompt has " + std::to_string(prompt_tokens.size()) +
        " tokens; the context holds " + std::to_string(n_ctx));
    return -6;
  }

  // An idle cancel() leaves the flag set, and it would abort this decode.
  g_cancel_requested = false;
  if (!evaluate_prompt_tokens(prompt_tokens, reuse_prompt_prefix != 0)) {
    return -7;
  }

  const float * logits = llama_get_logits_ith(g_state.ctx, -1);
  if (logits == nullptr) {
    set_error("Prompt evaluation produced no logits for the last token");
    return -8;
  }

  const double log_sum = llamadart_webgpu_detail::log_sum_exp(logits, n_vocab);
  auto scored = [&](const llama_token token) {
    return llamadart_webgpu_detail::ScoredToken{
        token,
        token_to_piece(token, true),
        llamadart_webgpu_detail::rank_logit(logits[token]) - log_sum,
    };
  };

  std::vector<llamadart_webgpu_detail::ScoredToken> scored_candidates;
  scored_candidates.reserve(candidates.size());
  for (const llama_token token : candidates) {
    scored_candidates.push_back(scored(token));
  }

  std::vector<llamadart_webgpu_detail::ScoredToken> scored_top;
  for (const int32_t token :
       llamadart_webgpu_detail::top_token_ids(logits, n_vocab, top_k)) {
    scored_top.push_back(scored(token));
  }

  g_last_next_token_scores_json =
      llamadart_webgpu_detail::serialize_next_token_scores_json(
          scored_candidates,
          scored_top,
          static_cast<int32_t>(prompt_tokens.size()));
  return 0;
}

EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_last_next_token_scores_json() {
  return g_last_next_token_scores_json.c_str();
}
