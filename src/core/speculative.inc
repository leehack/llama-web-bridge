// Speculative decoding: request checks, the per-generation session with its
// draft context, and target decoding that keeps the draft state in step.
// Part of llama_webgpu_core.cpp, included inside its anonymous namespace;
// not a standalone translation unit.

bool is_draft_model_type(const common_speculative_type type) {
  switch (type) {
    case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
    case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
    case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
    case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
      return true;
    default:
      return false;
  }
}

bool needs_draft_context(const common_speculative_type type) {
  return is_draft_model_type(type) || type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
}

bool has_type(
    const std::vector<common_speculative_type> & types,
    const common_speculative_type type) {
  return std::find(types.begin(), types.end(), type) != types.end();
}

bool model_has_recurrent_state(const llama_model * model) {
  return llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
}

bool target_has_mtp_head() {
  return g_state.model != nullptr && g_model_loaded_mtp &&
      llama_model_n_layer_nextn(g_state.model) > 0;
}

// Upstream enables extraction of these target layers, and llama.cpp asserts
// that each is at most the target's layer count.
bool draft_layer_ids_fit_target(const llama_model * draft_model) {
  const int32_t * layer_ids = llama_model_target_layer_ids(draft_model);
  const uint32_t n_layer_ids = llama_model_target_layer_ids_n(draft_model);
  const int32_t n_layer = llama_model_n_layer(g_state.model);
  if (n_layer_ids == 0) {
    return false;
  }
  for (uint32_t i = 0; i < n_layer_ids; ++i) {
    if (layer_ids[i] < 0 || layer_ids[i] > n_layer) {
      return false;
    }
  }
  return true;
}

// common_ngram_cache_load asserts on a malformed file, which would abort the
// core. Walks the same record layout first, and rejects draft tokens outside
// the target vocabulary.
bool validate_ngram_cache_file(const std::string & path, const char * label) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    set_error(std::string("Cannot open the ") + label + " n-gram cache");
    return false;
  }
  const int32_t n_vocab = llama_vocab_n_tokens(g_state.vocab);
  common_ngram ngram;
  while (input.read(reinterpret_cast<char *>(&ngram), sizeof(ngram))) {
    int32_t n_tokens = 0;
    if (!input.read(reinterpret_cast<char *>(&n_tokens), sizeof(n_tokens)) ||
        n_tokens <= 0) {
      set_error(std::string("The ") + label + " n-gram cache is malformed");
      return false;
    }
    for (int32_t i = 0; i < n_tokens; ++i) {
      llama_token token = 0;
      int32_t count = 0;
      if (!input.read(reinterpret_cast<char *>(&token), sizeof(token)) ||
          !input.read(reinterpret_cast<char *>(&count), sizeof(count)) ||
          count <= 0) {
        set_error(std::string("The ") + label + " n-gram cache is malformed");
        return false;
      }
      if (token < 0 || token >= n_vocab) {
        set_error(
            std::string("The ") + label +
            " n-gram cache holds a token outside this model's vocabulary");
        return false;
      }
    }
  }
  return true;
}

// Returns false after set_error() when the request cannot run on the loaded
// model. Changes no state.
bool validate_speculative_request(
    const speculative_request & request,
    const bool has_grammar,
    const bool has_reasoning_budget) {
  if (has_grammar) {
    set_error("speculative decoding does not support grammar-constrained sampling");
    return false;
  }
  if (has_reasoning_budget) {
    set_error("speculative decoding cannot be combined with a thinking budget");
    return false;
  }
  if (!g_pending_media.empty()) {
    set_error("speculative decoding supports text-only prompts");
    return false;
  }
  if (request.types.empty()) {
    set_error("speculative decoding needs at least one strategy");
    return false;
  }

  int32_t n_context_types = 0;
  for (const common_speculative_type type : request.types) {
    if (needs_draft_context(type)) {
      n_context_types++;
    }
  }
  if (n_context_types > 1) {
    set_error(
        "speculative decoding can mix n-gram strategies with at most one "
        "draft-model or MTP strategy");
    return false;
  }

  for (const common_speculative_type type : request.types) {
    if (type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP && !target_has_mtp_head()) {
      set_error(
          "draft-mtp needs the model's MTP head: load a model with MTP layers "
          "with loadMtp: true");
      return false;
    }
    if (!is_draft_model_type(type)) {
      continue;
    }
    const std::string name = common_speculative_type_to_str(type);
    if (g_draft_model == nullptr) {
      set_error(name + " needs a draft model: call loadDraftModel first");
      return false;
    }
    const bool is_block_type = type == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH ||
        type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
    if (is_block_type && g_draft_model_block_type != type) {
      set_error(
          name + " needs a " + name.substr(6) + " draft model; the loaded "
          "draft model's architecture is " + g_draft_model_architecture +
          (g_draft_model_block_type == COMMON_SPECULATIVE_TYPE_NONE
              ? ""
              : " (" + common_speculative_type_to_str(g_draft_model_block_type) + ")"));
      return false;
    }
    if (type == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 &&
        g_draft_model_architecture != "eagle3") {
      set_error(
          "draft-eagle3 needs an EAGLE3 draft model; the loaded draft model's "
          "architecture is " + g_draft_model_architecture);
      return false;
    }
    if (type == COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE &&
        (g_draft_model_architecture == "eagle3" ||
         g_draft_model_architecture == "dflash")) {
      set_error(
          "draft-simple needs a standalone language model; the loaded draft "
          "model's architecture is " + g_draft_model_architecture);
      return false;
    }
    if (type == COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE &&
        model_has_recurrent_state(g_draft_model)) {
      set_error(
          "draft-simple does not support a draft model with recurrent state, "
          "which cannot roll back rejected draft tokens");
      return false;
    }
    if (type != COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE &&
        !draft_layer_ids_fit_target(g_draft_model)) {
      set_error(
          name + ": the draft model's target layers do not match the loaded "
          "model");
      return false;
    }
  }

  if (n_context_types > 0 && model_has_recurrent_state(g_state.model) &&
      llama_n_rs_seq(g_state.ctx) == 0) {
    set_error(
        "this model has recurrent state, so draft-model and MTP speculative "
        "decoding need rollback snapshots: reload it with "
        "speculativeRollbackTokenMax of at least the draft length");
    return false;
  }

  if (request.draft_n_max <= 0 || request.draft_n_min < 0 ||
      request.draft_n_min > request.draft_n_max) {
    set_error("speculative draft token limits are out of range");
    return false;
  }
  const int64_t n_ctx = llama_n_ctx(g_state.ctx);
  if (request.draft_n_max > n_ctx || request.ngram_n_max > n_ctx) {
    set_error(
        "speculative draft token limits exceed the context size of " +
        std::to_string(n_ctx) + " tokens");
    return false;
  }
  const auto is_probability = [](const float value) {
    return value == -1.0f || (value >= 0.0f && value <= 1.0f);
  };
  if (!is_probability(request.p_min) || !is_probability(request.p_split)) {
    set_error("speculative draft probabilities must be from 0 to 1");
    return false;
  }
  const auto is_ngram_size = [](const int32_t value) {
    return value >= 0 && value <= std::numeric_limits<uint16_t>::max();
  };
  if (!is_ngram_size(request.ngram_size_n) ||
      !is_ngram_size(request.ngram_size_m) ||
      !is_ngram_size(request.ngram_min_hits) ||
      !is_ngram_size(request.ngram_match) ||
      request.ngram_n_min < -1 || request.ngram_n_max < 0 ||
      (request.ngram_n_min >= 0 && request.ngram_n_max > 0 &&
       request.ngram_n_min > request.ngram_n_max)) {
    set_error("speculative n-gram settings are out of range");
    return false;
  }

  if (!request.cache_static_path.empty() &&
      !validate_ngram_cache_file(request.cache_static_path, "static")) {
    return false;
  }
  if (!request.cache_dynamic_path.empty() &&
      !validate_ngram_cache_file(request.cache_dynamic_path, "dynamic")) {
    return false;
  }
  return true;
}

// Mirrors llama.cpp's common_speculative_init_result and
// common_base_params_to_speculative for one sequence.
llama_context * create_draft_context(
    const speculative_request & request,
    llama_model * model,
    const bool is_mtp) {
  llama_context_params cparams = g_context_params;
  cparams.n_ctx = llama_n_ctx(g_state.ctx);
  cparams.n_seq_max = 1;
  cparams.n_rs_seq = 0;
  cparams.n_outputs_max = 1;
  cparams.n_outputs_max_per_seq = 1;
  if (has_type(request.types, COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) ||
      has_type(request.types, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)) {
    const uint32_t per_seq =
        static_cast<uint32_t>(std::max(1, request.draft_n_max + 1));
    cparams.n_outputs_max = per_seq;
    cparams.n_outputs_max_per_seq = per_seq;
  }
  cparams.embeddings = false;
  cparams.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
  cparams.type_k = GGML_TYPE_F16;
  cparams.type_v = GGML_TYPE_F16;
  cparams.ctx_type = is_mtp ? LLAMA_CONTEXT_TYPE_MTP : LLAMA_CONTEXT_TYPE_DEFAULT;
  cparams.ctx_other = g_state.ctx;
  cparams.samplers = nullptr;
  cparams.n_samplers = 0;

  std::string init_error;
  llama_context * ctx =
      llamadart_webgpu_draft_context_init(model, cparams, &init_error);
  if (ctx == nullptr) {
    set_error("Failed to create the speculative draft context: " + init_error);
    return nullptr;
  }
  llama_set_abort_callback(ctx, should_abort_callback, nullptr);
  return ctx;
}

// Returns false after set_error(). On success g_speculative holds a session
// with no prompt evaluated.
bool begin_speculative_session(const speculative_request & request) {
  auto * session = new speculative_session();
  g_speculative = session;

  const bool uses_mtp = has_type(request.types, COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
  bool uses_draft_model = false;
  for (const common_speculative_type type : request.types) {
    uses_draft_model = uses_draft_model || is_draft_model_type(type);
  }

  if (uses_draft_model || uses_mtp) {
    llama_model * model = uses_draft_model ? g_draft_model : g_state.model;
    session->ctx_dft = create_draft_context(request, model, uses_mtp);
    if (session->ctx_dft == nullptr) {
      free_speculative_session();
      return false;
    }
    if (uses_draft_model) {
      session->draft_model = g_draft_model;
    }
  }

  common_params_speculative params;
  params.types = request.types;
  params.draft.n_max = request.draft_n_max;
  params.draft.n_min = request.draft_n_min;
  if (request.p_min >= 0.0f) {
    params.draft.p_min = request.p_min;
  }
  if (request.p_split >= 0.0f) {
    params.draft.p_split = request.p_split;
  }
  params.draft.ctx_tgt = g_state.ctx;
  params.draft.ctx_dft = session->ctx_dft;
  for (common_params_speculative_ngram_map * map :
       {&params.ngram_simple, &params.ngram_map_k, &params.ngram_map_k4v}) {
    if (request.ngram_size_n > 0) {
      map->size_n = static_cast<uint16_t>(request.ngram_size_n);
    }
    if (request.ngram_size_m > 0) {
      map->size_m = static_cast<uint16_t>(request.ngram_size_m);
    }
    if (request.ngram_min_hits > 0) {
      map->min_hits = static_cast<uint16_t>(request.ngram_min_hits);
    }
  }
  if (request.ngram_match > 0) {
    params.ngram_mod.n_match = request.ngram_match;
  }
  if (request.ngram_n_min >= 0) {
    params.ngram_mod.n_min = request.ngram_n_min;
  }
  if (request.ngram_n_max > 0) {
    params.ngram_mod.n_max = request.ngram_n_max;
  }
  params.ngram_cache.lookup_cache_static = request.cache_static_path;
  params.ngram_cache.lookup_cache_dynamic = request.cache_dynamic_path;

  std::string init_error;
  session->spec = llamadart_webgpu_speculative_init(&params, &init_error);
  if (session->spec == nullptr) {
    set_error("Failed to initialize speculative decoding: " + init_error);
    free_speculative_session();
    return false;
  }

  int32_t n_batch = static_cast<int32_t>(llama_n_batch(g_state.ctx));
  if (n_batch <= 0) {
    n_batch = 512;
  }
  session->batch = llama_batch_init(n_batch, 0, 1);
  session->batch_capacity = n_batch;
  session->draft_max = request.draft_n_max;
  session->max_tokens = request.max_tokens;
  // A recurrent target without rollback snapshots cannot drop rejected draft
  // tokens; its recurrent state is saved before each verification instead.
  session->needs_checkpoint = model_has_recurrent_state(g_state.model) &&
      llama_n_rs_seq(g_state.ctx) == 0;
  return true;
}

// Decodes `count` tokens of the session batch into sequence 0 of the target
// and passes the batch to the speculative state, as llama-server does after
// every target decode. Returns the llama_decode result, or -100 after
// set_error() when the speculative state rejects the batch.
int32_t decode_speculative_batch() {
  speculative_session & session = *g_speculative;
  const int32_t rc = llama_decode(g_state.ctx, session.batch);
  if (rc != 0) {
    return rc;
  }
  if (!common_speculative_process(session.spec, session.batch)) {
    set_error("The speculative draft state failed to process a target batch");
    return -100;
  }
  return 0;
}

// Decodes the prompt into an empty target memory in n_batch chunks, logits
// for its last token only. Returns false after set_error().
bool evaluate_speculative_prompt(const std::vector<llama_token> & tokens) {
  speculative_session & session = *g_speculative;
  const size_t n_batch = static_cast<size_t>(session.batch_capacity);
  for (size_t offset = 0; offset < tokens.size(); offset += n_batch) {
    const size_t count = std::min(tokens.size() - offset, n_batch);
    common_batch_clear(session.batch);
    for (size_t i = 0; i < count; ++i) {
      const size_t index = offset + i;
      common_batch_add(
          session.batch,
          tokens[index],
          static_cast<llama_pos>(index),
          {0},
          index + 1 == tokens.size());
    }
    const int32_t rc = decode_speculative_batch();
    if (rc == -100) {
      return false;
    }
    if (rc != 0) {
      set_error("llama_decode failed while processing prompt");
      return false;
    }
  }
  common_speculative_begin(session.spec, 0, tokens);
  session.history = tokens;
  return true;
}
