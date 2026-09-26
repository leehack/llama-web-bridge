// Samplers, grammar-constrained sampling, and token generation.
// Part of llama_webgpu_core.cpp, included inside its anonymous namespace;
// not a standalone translation unit.

llama_sampler * create_sampler(
    const float temp,
    const int32_t top_k,
    const float top_p,
    const float min_p,
    const float repeat_penalty,
    const float presence_penalty,
    const uint32_t seed) {
  llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
  llama_sampler * sampler = llama_sampler_chain_init(sparams);
  if (sampler == nullptr) {
    return nullptr;
  }

  if (repeat_penalty != 1.0f || presence_penalty != 0.0f) {
    const int32_t vocab_size = llama_vocab_n_tokens(g_state.vocab);
    llama_sampler_chain_add(
        sampler,
        llama_sampler_init_penalties(
            vocab_size, 64, repeat_penalty, 0.0f, presence_penalty));
  }

  if (top_k > 0) {
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
  }

  if (top_p < 1.0f) {
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
  }

  if (min_p > 0.0f) {
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(min_p, 1));
  }

  llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
  llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));

  return sampler;
}

bool grammar_accepts_token(const llama_token token) {
  llama_token_data single_token = {token, 1.0f, 0.0f};
  llama_token_data_array single_token_array = {&single_token, 1, -1, false};
  llama_sampler_apply(g_active_grammar, &single_token_array);
  return single_token_array.data[0].logit != -INFINITY;
}

// Fills cur_p from the last logits and runs the samplers over it. Returns
// false after set_error() when the logits are missing or nothing is selected.
bool apply_generation_samplers(
    const bool grammar_first,
    llama_token_data_array & cur_p) {
  const float * logits = llama_get_logits_ith(g_state.ctx, -1);
  if (logits == nullptr) {
    set_error("Failed to read logits for sampling");
    return false;
  }

  const int32_t n_vocab = llama_vocab_n_tokens(g_state.vocab);
  g_sampler_candidates.resize(static_cast<size_t>(n_vocab));
  for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
    g_sampler_candidates[token_id] =
        llama_token_data{token_id, logits[token_id], 0.0f};
  }
  cur_p = {
      g_sampler_candidates.data(),
      g_sampler_candidates.size(),
      -1,
      false,
  };

  if (grammar_first) {
    llama_sampler_apply(g_active_grammar, &cur_p);
  }
  llama_sampler_apply(g_active_sampler, &cur_p);

  if (cur_p.selected < 0 ||
      cur_p.selected >= static_cast<int64_t>(cur_p.size)) {
    set_error("Sampler selected no token");
    return false;
  }
  return true;
}

// Mirrors llama.cpp common_sampler_sample(): sample with the chain alone,
// check the pick against the grammar, and on rejection resample with the
// grammar applied before truncation. Running the grammar after top-k/top-p
// can leave only rejected tokens, and llama_grammar_accept on a rejected
// token aborts the Wasm core. Returns LLAMA_TOKEN_NULL after set_error().
llama_token sample_next_token() {
  llama_token_data_array cur_p = {};
  if (!apply_generation_samplers(false, cur_p)) {
    return LLAMA_TOKEN_NULL;
  }
  llama_token token = cur_p.data[cur_p.selected].id;

  if (g_active_grammar != nullptr && !grammar_accepts_token(token)) {
    if (!apply_generation_samplers(true, cur_p)) {
      return LLAMA_TOKEN_NULL;
    }
    token = cur_p.data[cur_p.selected].id;
    // Every candidate is rejected only when the grammar can't be continued
    // with this vocabulary; accepting the token would abort.
    if (!grammar_accepts_token(token)) {
      set_error("Grammar rejected every candidate token");
      return LLAMA_TOKEN_NULL;
    }
  }

  if (g_active_grammar != nullptr) {
    llama_sampler_accept(g_active_grammar, token);
  }
  llama_sampler_accept(g_active_sampler, token);
  return token;
}

// Decodes prompt_tokens into sequence 0, keeping the longest prefix shared
// with the previous prompt when reuse_prefix is true. At least one token is
// decoded, so the last position always has logits. Sets *reused_tokens, when
// given, to the number of leading tokens kept rather than decoded. Returns
// false after decode_tokens() records the error.
bool evaluate_prompt_tokens(
    const std::vector<llama_token> & prompt_tokens,
    const bool reuse_prefix,
    size_t * reused_tokens = nullptr) {
  size_t prefix = 0;
  if (reuse_prefix) {
    const size_t max_prefix =
        std::min(g_cached_prompt_tokens.size(), prompt_tokens.size());
    while (prefix < max_prefix &&
           g_cached_prompt_tokens[prefix] == prompt_tokens[prefix]) {
      prefix++;
    }
  }

  if (prefix == prompt_tokens.size() && prefix > 0) {
    prefix--;
  }

  if (prefix == 0) {
    llama_memory_clear(llama_get_memory(g_state.ctx), false);
  } else {
    const bool removed = llama_memory_seq_rm(
        llama_get_memory(g_state.ctx),
        0,
        static_cast<llama_pos>(prefix),
        -1);
    if (!removed) {
      prefix = 0;
      llama_memory_clear(llama_get_memory(g_state.ctx), false);
    }
  }

  if (prefix < prompt_tokens.size()) {
    std::vector<llama_token> eval_tokens(
        prompt_tokens.begin() + prefix,
        prompt_tokens.end());
    if (!decode_tokens(eval_tokens)) {
      g_cached_prompt_tokens.clear();
      return false;
    }
  }

  g_cached_prompt_tokens = prompt_tokens;
  if (reused_tokens != nullptr) {
    *reused_tokens = prefix;
  }
  return true;
}

int32_t begin_generation_impl(
    const char * prompt,
    float temp,
    int32_t top_k,
    float top_p,
    float repeat_penalty,
    const char * grammar,
    uint32_t seed,
    const float min_p,
    const float presence_penalty) {
  // Defence in depth behind the bridge's single-writer queue: reject before
  // touching any shared state, so a stray second begin cannot destroy the
  // in-flight generation's output, sampler, or g_last_error. The caller maps
  // -7 to a fixed message instead of reading g_last_error.
  if (g_generation_active) {
    return -7;
  }

  clear_error();
  g_last_output.clear();
  g_last_piece.clear();
  g_usage_prompt_tokens = 0;
  g_usage_cached_prompt_tokens = 0;
  g_usage_completion_tokens = 0;

  if (!ensure_loaded()) {
    return -1;
  }

  if (g_tts_active) {
    set_error("Text-to-speech synthesis is active");
    return -6;
  }

  if (prompt == nullptr) {
    set_error("Prompt is null");
    return -2;
  }

  if (temp < 0.0f) {
    temp = 0.0f;
  }

  if (top_k < 0) {
    top_k = 0;
  }

  if (top_p <= 0.0f || top_p > 1.0f) {
    top_p = 1.0f;
  }

  if (repeat_penalty <= 0.0f) {
    repeat_penalty = 1.0f;
  }

  if (!std::isfinite(min_p) || min_p < 0.0f || min_p > 1.0f) {
    set_error("min_p must be a finite number from 0 to 1");
    return -2;
  }

  if (!std::isfinite(presence_penalty)) {
    set_error("presence_penalty must be a finite number");
    return -2;
  }

  end_generation_state();

  // Parse the grammar before the prompt touches the KV cache, so an invalid
  // grammar rejects without side effects.
  llama_sampler_ptr grammar_sampler;
  if (grammar != nullptr && grammar[0] != '\0') {
    std::string grammar_error;
    grammar_sampler.reset(llamadart_webgpu_grammar_sampler_init(
        g_state.vocab, grammar, "root", &grammar_error));
    if (!grammar_sampler) {
      set_error(
          "Failed to initialize sampler chain (invalid grammar): " +
          grammar_error);
      return -5;
    }
  }

  g_cancel_requested = false;
  g_generation_has_qwen3_asr_audio =
      g_model_is_qwen3_asr &&
      g_state.mm_ctx != nullptr &&
      mtmd_support_audio(g_state.mm_ctx) &&
      !mtmd_support_vision(g_state.mm_ctx) &&
      !g_pending_media.empty();

  const std::string prompt_text = prompt;
  if (!g_pending_media.empty()) {
    llama_memory_clear(llama_get_memory(g_state.ctx), false);
    g_cached_prompt_tokens.clear();

    llama_pos n_past = 0;
    if (!decode_multimodal_prompt(prompt_text, &n_past)) {
      return -3;
    }
    g_usage_prompt_tokens = static_cast<int32_t>(n_past);
  } else {
    std::vector<llama_token> prompt_tokens;
    if (!tokenize_text(prompt_text, true, prompt_tokens)) {
      return -3;
    }

    size_t reused_tokens = 0;
    if (!evaluate_prompt_tokens(prompt_tokens, true, &reused_tokens)) {
      return -4;
    }
    g_usage_prompt_tokens = static_cast<int32_t>(prompt_tokens.size());
    g_usage_cached_prompt_tokens = static_cast<int32_t>(reused_tokens);
  }

  g_active_sampler =
      create_sampler(
          temp, top_k, top_p, min_p, repeat_penalty, presence_penalty, seed);
  if (g_active_sampler == nullptr) {
    set_error("Failed to initialize sampler chain");
    return -5;
  }

  g_active_grammar = grammar_sampler.release();
  g_generation_active = true;
  return 0;
}

int32_t next_token_impl() {
  clear_error();

  if (!ensure_loaded()) {
    return -1;
  }

  if (!g_generation_active || g_active_sampler == nullptr) {
    set_error("Generation is not active");
    return -2;
  }

  if (g_cancel_requested) {
    end_generation_state();
    return 0;
  }

  while (true) {
    const bool is_prefix_token =
        g_qwen3_asr_prefix_token_index < g_qwen3_asr_prefix_tokens.size();
    llama_token token = is_prefix_token
        ? g_qwen3_asr_prefix_tokens[g_qwen3_asr_prefix_token_index++]
        : sample_next_token();
    if (token == LLAMA_TOKEN_NULL) {
      // sample_next_token() has already recorded the error.
      end_generation_state();
      return -3;
    }

    if (!is_prefix_token &&
        llama_vocab_is_eog(g_state.vocab, token) &&
        g_last_output.empty() &&
        g_generation_has_qwen3_asr_audio &&
        !g_qwen3_asr_prefix_seed_attempted) {
      // The WASM path can select EOG at the first decoding step for short,
      // clearly decodable speech. Seed Qwen3-ASR's output boundary once and
      // let the model decide whether transcript text follows. The prefix is
      // consumed internally, so silence can still terminate with no output.
      g_qwen3_asr_prefix_seed_attempted = true;
      if (tokenize_text(
              "<asr_text>", false, g_qwen3_asr_prefix_tokens) &&
          !g_qwen3_asr_prefix_tokens.empty()) {
        g_qwen3_asr_prefix_token_index = 0;
        continue;
      }
    }

    if (llama_vocab_is_eog(g_state.vocab, token)) {
      end_generation_state();
      return 0;
    }

    g_last_piece = is_prefix_token ? "" : token_to_piece(token, true);
    if (!is_prefix_token) {
      g_last_output += g_last_piece;
      g_usage_completion_tokens++;
    }

    llama_token token_for_decode = token;
    const int rc =
        llama_decode(g_state.ctx, llama_batch_get_one(&token_for_decode, 1));
    if (rc != 0) {
      if (g_cancel_requested) {
        end_generation_state();
        return 0;
      }
      set_error("llama_decode failed while generating tokens");
      end_generation_state();
      return -4;
    }

    if (is_prefix_token) {
      // The prefix is internal, not generated output, so like llama.cpp's
      // common_sampler_accept(is_generated=false) it bypasses the grammar.
      llama_sampler_accept(g_active_sampler, token);
    } else {
      return 1;
    }
  }
}
