// Samplers, grammar-constrained sampling, and token generation.
// Part of llama_webgpu_core.cpp, included inside its anonymous namespace;
// not a standalone translation unit.

// Takes ownership of reasoning_budget, which may be null.
llama_sampler * create_sampler(
    const float temp,
    const int32_t top_k,
    const float top_p,
    const float min_p,
    const float repeat_penalty,
    const float presence_penalty,
    const uint32_t seed,
    llama_sampler * reasoning_budget) {
  llama_sampler_ptr owned_reasoning_budget(reasoning_budget);
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

  if (owned_reasoning_budget) {
    llama_sampler_chain_add(sampler, owned_reasoning_budget.release());
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

// Mirrors llamadart-native's reasoning-budget sampler: the grammar pauses
// while a reasoning block is open or its end is being forced.
bool grammar_is_active() {
  if (g_active_grammar == nullptr) {
    return false;
  }
  if (g_active_reasoning_budget == nullptr) {
    return true;
  }
  const common_reasoning_budget_state state =
      common_reasoning_budget_get_state(g_active_reasoning_budget);
  return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
}

bool grammar_accepts_token(const llama_token token) {
  llama_token_data single_token = {token, 1.0f, 0.0f};
  llama_token_data_array single_token_array = {&single_token, 1, -1, false};
  llama_sampler_apply(g_active_grammar, &single_token_array);
  return single_token_array.data[0].logit != -INFINITY;
}

// Fills cur_p from the logits of batch_index (-1: the last output) and runs
// the samplers over it. Returns false after set_error() when the logits are
// missing or nothing is selected.
bool apply_generation_samplers(
    const bool grammar_first,
    llama_token_data_array & cur_p,
    const int32_t batch_index) {
  const float * logits = llama_get_logits_ith(g_state.ctx, batch_index);
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
llama_token sample_next_token(const int32_t batch_index = -1) {
  const bool use_grammar = grammar_is_active();
  llama_token_data_array cur_p = {};
  if (!apply_generation_samplers(false, cur_p, batch_index)) {
    return LLAMA_TOKEN_NULL;
  }
  llama_token token = cur_p.data[cur_p.selected].id;

  if (use_grammar && !grammar_accepts_token(token)) {
    if (!apply_generation_samplers(true, cur_p, batch_index)) {
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

  if (use_grammar) {
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

// Tokenized reasoning-budget tags, as llamadart-native's
// llama_dart_sampler_init_reasoning_budget tokenizes them.
struct reasoning_budget_config {
  int32_t max_tokens = 0;
  llama_tokens start_tokens;
  llama_tokens end_tokens;
  llama_tokens forced_tokens;
};

// Returns false after set_error() when a tag is empty or yields no tokens.
bool resolve_reasoning_budget(
    const int32_t max_tokens,
    const char * start_tag,
    const char * end_tag,
    const char * forced_message,
    reasoning_budget_config & out) {
  if (max_tokens < 0) {
    set_error("thinking budget max_tokens must be non-negative");
    return false;
  }
  if (end_tag == nullptr || start_tag[0] == '\0' || end_tag[0] == '\0') {
    set_error("thinking budget start and end tags must be non-empty");
    return false;
  }
  out.max_tokens = max_tokens;
  out.start_tokens = common_tokenize(g_state.vocab, start_tag, false, true);
  out.end_tokens = common_tokenize(g_state.vocab, end_tag, false, true);
  std::string forced_text = forced_message == nullptr ? "" : forced_message;
  forced_text += end_tag;
  out.forced_tokens = common_tokenize(g_state.vocab, forced_text, false, true);
  if (out.start_tokens.empty() || out.end_tokens.empty() ||
      out.forced_tokens.empty()) {
    set_error("thinking budget tags produce no tokens for this vocabulary");
    return false;
  }
  return true;
}

bool token_sequence_at(
    const std::vector<llama_token> & tokens,
    const size_t start,
    const llama_tokens & sequence) {
  return sequence.size() <= tokens.size() - start &&
      std::equal(sequence.begin(), sequence.end(), tokens.begin() + start);
}

// A prompt that ends inside an opened reasoning block starts the budget
// counting, as llamadart-native derives it.
common_reasoning_budget_state reasoning_budget_initial_state(
    const reasoning_budget_config & config,
    const std::vector<llama_token> & prompt_tokens) {
  common_reasoning_budget_state state = REASONING_BUDGET_IDLE;
  for (size_t index = 0; index < prompt_tokens.size();) {
    if (token_sequence_at(prompt_tokens, index, config.start_tokens)) {
      state = REASONING_BUDGET_COUNTING;
      index += config.start_tokens.size();
    } else if (token_sequence_at(prompt_tokens, index, config.end_tokens)) {
      state = REASONING_BUDGET_IDLE;
      index += config.end_tokens.size();
    } else {
      index++;
    }
  }
  return state;
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
    const float presence_penalty,
    const int32_t thinking_budget_tokens,
    const char * thinking_start_tag,
    const char * thinking_end_tag,
    const char * thinking_forced_message) {
  // Defence in depth behind the bridge's single-writer queue: reject before
  // touching any shared state, so a stray second begin cannot destroy the
  // in-flight generation's output, sampler, or g_last_error. The caller maps
  // -7 to a fixed message instead of reading g_last_error.
  // The speculative request belongs to this call, even a rejected one.
  const speculative_request speculative = std::move(g_next_speculative);
  g_next_speculative = speculative_request();
  if (g_generation_active) {
    return -7;
  }

  clear_error();
  g_last_output.clear();
  g_last_piece.clear();
  g_usage_prompt_tokens = 0;
  g_usage_cached_prompt_tokens = 0;
  g_usage_completion_tokens = 0;
  g_speculative_usage = speculative_usage();

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

  const bool has_reasoning_budget = thinking_start_tag != nullptr;
  reasoning_budget_config reasoning_budget;
  if (has_reasoning_budget) {
    if (!g_pending_media.empty()) {
      set_error("thinking budget supports text-only prompts");
      return -2;
    }
    if (!resolve_reasoning_budget(
            thinking_budget_tokens,
            thinking_start_tag,
            thinking_end_tag,
            thinking_forced_message,
            reasoning_budget)) {
      return -2;
    }
  }

  if (speculative.active &&
      !validate_speculative_request(
          speculative,
          grammar != nullptr && grammar[0] != '\0',
          has_reasoning_budget)) {
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
  llama_sampler_ptr reasoning_budget_sampler;
  if (!g_pending_media.empty()) {
    llama_memory_clear(llama_get_memory(g_state.ctx), false);
    g_cached_prompt_tokens.clear();

    llama_pos n_past = 0;
    if (!decode_multimodal_prompt(prompt_text, &n_past)) {
      return -3;
    }
    g_usage_prompt_tokens = static_cast<int32_t>(n_past);
  } else if (speculative.active) {
    // As native llamadart does, speculative generation evaluates the whole
    // prompt, so every draft state sees it.
    llama_memory_clear(llama_get_memory(g_state.ctx), false);
    g_cached_prompt_tokens.clear();

    std::vector<llama_token> prompt_tokens;
    if (!tokenize_text(prompt_text, true, prompt_tokens)) {
      return -3;
    }
    if (prompt_tokens.empty()) {
      set_error("Prompt produced no tokens");
      return -3;
    }
    if (!begin_speculative_session(speculative)) {
      return -5;
    }
    if (!evaluate_speculative_prompt(prompt_tokens)) {
      free_speculative_session();
      return -4;
    }
    g_cached_prompt_tokens = prompt_tokens;
    g_usage_prompt_tokens = static_cast<int32_t>(prompt_tokens.size());
    g_speculative_usage.ran = true;
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

    if (has_reasoning_budget) {
      reasoning_budget_sampler.reset(common_reasoning_budget_init(
          g_state.vocab,
          {reasoning_budget.start_tokens},
          {reasoning_budget.end_tokens},
          reasoning_budget.forced_tokens,
          reasoning_budget.max_tokens,
          reasoning_budget_initial_state(reasoning_budget, prompt_tokens)));
      if (!reasoning_budget_sampler) {
        set_error("Failed to initialize the thinking budget sampler");
        return -5;
      }
    }
  }

  llama_sampler * reasoning_budget_ptr = reasoning_budget_sampler.get();
  g_active_sampler = create_sampler(
      temp,
      top_k,
      top_p,
      min_p,
      repeat_penalty,
      presence_penalty,
      seed,
      reasoning_budget_sampler.release());
  if (g_active_sampler == nullptr) {
    free_speculative_session();
    set_error("Failed to initialize sampler chain");
    return -5;
  }

  g_active_reasoning_budget = reasoning_budget_ptr;
  g_active_grammar = grammar_sampler.release();
  g_generation_active = true;
  return 0;
}

// Decodes [id_last, draft...] at the end of the target memory and returns the
// target's picks, as llama.cpp's common_sampler_sample_and_accept_n does: one
// sample per position until a pick differs from the draft, plus one after a
// fully accepted draft. Returns false after set_error(); *rc is the decode
// result when decoding failed.
bool verify_speculative_draft(
    std::vector<llama_token> & accepted,
    int32_t * rc) {
  speculative_session & session = *g_speculative;
  const llama_pos n_past = static_cast<llama_pos>(session.history.size());
  common_batch_clear(session.batch);
  common_batch_add(session.batch, session.id_last, n_past, {0}, true);
  for (size_t i = 0; i < session.draft.size(); ++i) {
    common_batch_add(
        session.batch,
        session.draft[i],
        n_past + 1 + static_cast<llama_pos>(i),
        {0},
        true);
  }
  *rc = decode_speculative_batch();
  if (*rc != 0) {
    return false;
  }
  g_speculative_usage.verify_tokens += session.batch.n_tokens;

  accepted.clear();
  for (size_t i = 0; i <= session.draft.size(); ++i) {
    const llama_token token = sample_next_token(static_cast<int32_t>(i));
    if (token == LLAMA_TOKEN_NULL) {
      return false;
    }
    accepted.push_back(token);
    if (i < session.draft.size() && token != session.draft[i]) {
      break;
    }
  }
  return true;
}

// Drops the rejected draft tokens from the target memory. A recurrent target
// without rollback snapshots restores the state saved before verification and
// decodes the accepted tokens again, as native llamadart does. Returns false
// after set_error(); *rc is the decode result when replaying failed.
bool roll_back_speculative_target(
    const std::vector<llama_token> & accepted,
    int32_t * rc) {
  speculative_session & session = *g_speculative;
  const llama_pos n_past = static_cast<llama_pos>(session.history.size());
  const llama_pos keep = n_past + static_cast<llama_pos>(accepted.size());
  llama_memory_t memory = llama_get_memory(g_state.ctx);
  if (llama_memory_seq_rm(memory, 0, keep, -1)) {
    return true;
  }
  if (!session.needs_checkpoint || session.checkpoint.empty()) {
    set_error(
        "Speculative decoding could not roll back the target memory; reload "
        "the model with speculativeRollbackTokenMax of at least the draft "
        "length");
    return false;
  }
  const size_t restored = llama_state_seq_set_data_ext(
      g_state.ctx,
      session.checkpoint.data(),
      session.checkpoint.size(),
      0,
      LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
  if (restored != session.checkpoint.size() ||
      !llama_memory_seq_rm(memory, 0, n_past, -1)) {
    set_error("Speculative decoding could not restore the target state");
    return false;
  }

  common_batch_clear(session.batch);
  common_batch_add(session.batch, session.id_last, n_past, {0}, false);
  for (size_t i = 0; i + 1 < accepted.size(); ++i) {
    common_batch_add(
        session.batch,
        accepted[i],
        n_past + 1 + static_cast<llama_pos>(i),
        {0},
        false);
  }
  *rc = decode_speculative_batch();
  if (*rc != 0) {
    return false;
  }
  g_speculative_usage.replay_tokens += session.batch.n_tokens;
  return true;
}

// Samples at least one token into the session output: the first token after
// the prompt, or id_last decoded with a verified draft. Returns 1, 0 when the
// generation was cancelled, or a negative code after set_error().
int32_t speculative_step() {
  speculative_session & session = *g_speculative;
  const auto decode_failure = [](const int32_t rc) -> int32_t {
    if (rc == -100) {
      return -4;
    }
    if (g_cancel_requested) {
      return 0;
    }
    set_error("llama_decode failed while generating tokens");
    return -4;
  };

  if (session.id_last == LLAMA_TOKEN_NULL) {
    const llama_token token = sample_next_token();
    if (token == LLAMA_TOKEN_NULL) {
      return -3;
    }
    session.id_last = token;
    session.output.push_back(token);
    return 1;
  }

  const int32_t n_past = static_cast<int32_t>(session.history.size());
  int32_t n_draft_max = std::min(
      session.draft_max,
      std::min(
          static_cast<int32_t>(g_state.n_ctx) - n_past - 2,
          session.batch_capacity - 1));
  const int32_t n_rs_seq = static_cast<int32_t>(llama_n_rs_seq(g_state.ctx));
  if (n_rs_seq > 0) {
    n_draft_max = std::min(n_draft_max, n_rs_seq);
  }
  if (session.max_tokens > 0) {
    n_draft_max = std::min(
        n_draft_max,
        session.max_tokens - static_cast<int32_t>(session.output.size()) - 1);
  }

  session.draft.clear();
  if (n_draft_max > 0) {
    g_speculative_usage.draft_attempts++;
    common_speculative_get_draft_params(session.spec, 0) = {
        /* .drafting = */ true,
        /* .n_max    = */ n_draft_max,
        /* .pos0     = */ n_past,
        /* .id_last  = */ session.id_last,
        /* .prompt   = */ &session.history,
        /* .result   = */ &session.draft,
    };
    common_speculative_draft(session.spec);
    if (session.ctx_dft != nullptr &&
        !llama_memory_seq_rm(llama_get_memory(session.ctx_dft), 0, n_past, -1)) {
      set_error("Speculative decoding could not roll back the draft context");
      return -4;
    }
    // A draft vocabulary may be larger than the target's.
    const int32_t n_vocab = llama_vocab_n_tokens(g_state.vocab);
    for (size_t i = 0; i < session.draft.size(); ++i) {
      if (session.draft[i] < 0 || session.draft[i] >= n_vocab) {
        session.draft.resize(i);
        break;
      }
    }
    g_speculative_usage.draft_tokens +=
        static_cast<int64_t>(session.draft.size());
  }

  if (session.draft.empty()) {
    common_batch_clear(session.batch);
    common_batch_add(session.batch, session.id_last, n_past, {0}, true);
    const int32_t rc = decode_speculative_batch();
    if (rc != 0) {
      return decode_failure(rc);
    }
    session.history.push_back(session.id_last);
    const llama_token token = sample_next_token();
    if (token == LLAMA_TOKEN_NULL) {
      return -3;
    }
    session.id_last = token;
    session.output.push_back(token);
    return 1;
  }

  if (session.needs_checkpoint) {
    llama_synchronize(g_state.ctx);
    session.checkpoint.resize(llama_state_seq_get_size_ext(
        g_state.ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    const size_t written = llama_state_seq_get_data_ext(
        g_state.ctx,
        session.checkpoint.data(),
        session.checkpoint.size(),
        0,
        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (written != session.checkpoint.size()) {
      set_error("Speculative decoding could not save the target state");
      return -4;
    }
  }

  std::vector<llama_token> accepted;
  int32_t rc = 0;
  if (!verify_speculative_draft(accepted, &rc)) {
    return rc != 0 ? decode_failure(rc) : -3;
  }
  const size_t n_accepted = accepted.size() - 1;
  if (n_accepted < session.draft.size() &&
      !roll_back_speculative_target(accepted, &rc)) {
    return rc != 0 ? decode_failure(rc) : -4;
  }
  common_speculative_accept(session.spec, 0, static_cast<uint16_t>(n_accepted));
  g_speculative_usage.accepted_draft_tokens += static_cast<int64_t>(n_accepted);

  const llama_pos keep = static_cast<llama_pos>(n_past) +
      static_cast<llama_pos>(accepted.size());
  if (session.ctx_dft != nullptr &&
      !llama_memory_seq_rm(llama_get_memory(session.ctx_dft), 0, keep, -1)) {
    set_error("Speculative decoding could not roll back the draft context");
    return -4;
  }

  session.history.push_back(session.id_last);
  session.history.insert(
      session.history.end(), accepted.begin(), accepted.end() - 1);
  session.id_last = accepted.back();
  session.output.insert(session.output.end(), accepted.begin(), accepted.end());
  return 1;
}

// Returns the session's next sampled token as next_token_impl() does.
int32_t speculative_next_token() {
  speculative_session & session = *g_speculative;
  if (session.output_index == session.output.size()) {
    const int32_t rc = speculative_step();
    if (rc <= 0) {
      end_generation_state();
      return rc;
    }
  }

  const llama_token token = session.output[session.output_index++];
  if (llama_vocab_is_eog(g_state.vocab, token)) {
    end_generation_state();
    return 0;
  }
  g_last_piece = token_to_piece(token, true);
  g_last_output += g_last_piece;
  g_usage_completion_tokens++;
  return 1;
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

  if (g_speculative != nullptr) {
    return speculative_next_token();
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
