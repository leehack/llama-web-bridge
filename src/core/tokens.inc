// Tokenization, batch decoding, and token and text helpers.
// Part of llama_webgpu_core.cpp, included inside its anonymous namespace;
// not a standalone translation unit.

bool ensure_loaded() {
  if (g_state.model == nullptr || g_state.ctx == nullptr || g_state.vocab == nullptr) {
    set_error("Model is not loaded");
    return false;
  }
  return true;
}

bool tokenize_text(
    const std::string & text,
    const bool add_special,
    std::vector<llama_token> & out) {
  if (!ensure_loaded()) {
    return false;
  }

  int32_t capacity = static_cast<int32_t>(text.size()) + 8;
  if (capacity < 32) {
    capacity = 32;
  }

  out.assign(static_cast<size_t>(capacity), 0);
  int32_t n_tokens = llama_tokenize(
      g_state.vocab,
      text.c_str(),
      static_cast<int32_t>(text.size()),
      out.data(),
      static_cast<int32_t>(out.size()),
      add_special,
      true);

  if (n_tokens < 0) {
    const int32_t required = -n_tokens;
    out.assign(static_cast<size_t>(required), 0);
    n_tokens = llama_tokenize(
        g_state.vocab,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        out.data(),
        static_cast<int32_t>(out.size()),
        add_special,
        true);
  }

  if (n_tokens < 0) {
    set_error("Prompt tokenization failed");
    return false;
  }

  out.resize(static_cast<size_t>(n_tokens));
  return true;
}

bool decode_tokens(const std::vector<llama_token> & tokens) {
  if (!ensure_loaded()) {
    return false;
  }

  if (tokens.empty()) {
    set_error("Cannot decode empty token sequence");
    return false;
  }

  int32_t max_batch = static_cast<int32_t>(llama_n_batch(g_state.ctx));
  if (max_batch <= 0) {
    max_batch = 512;
  }

  for (size_t offset = 0; offset < tokens.size(); offset += static_cast<size_t>(max_batch)) {
    const int32_t count = static_cast<int32_t>(
        std::min(tokens.size() - offset, static_cast<size_t>(max_batch)));

    llama_token * ptr = const_cast<llama_token *>(tokens.data() + offset);
    const int rc = llama_decode(g_state.ctx, llama_batch_get_one(ptr, count));
    if (rc != 0) {
      set_error("llama_decode failed while processing prompt");
      return false;
    }
  }

  return true;
}

std::string token_to_piece(const llama_token token, const bool special) {
  if (!ensure_loaded()) {
    return "";
  }

  std::vector<char> buf(256, '\0');
  int32_t n =
      llama_token_to_piece(g_state.vocab, token, buf.data(), buf.size(), 0, special);

  if (n < 0) {
    buf.assign(static_cast<size_t>(-n) + 8, '\0');
    n = llama_token_to_piece(g_state.vocab, token, buf.data(), buf.size(), 0, special);
  }

  if (n < 0) {
    return "";
  }

  return std::string(buf.data(), static_cast<size_t>(n));
}

bool should_abort_callback(void * /*data*/) {
  return g_cancel_requested;
}

std::string serialize_tokens_json(const std::vector<llama_token> & tokens) {
  std::string json = "[";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0) {
      json += ",";
    }
    json += std::to_string(tokens[i]);
  }
  json += "]";
  return json;
}

void normalize_embedding_inplace(std::vector<float> & embedding) {
  double norm_squared = 0.0;
  for (const float value : embedding) {
    const double dv = static_cast<double>(value);
    norm_squared += dv * dv;
  }

  if (norm_squared <= 0.0) {
    return;
  }

  const double scale = 1.0 / std::sqrt(norm_squared);
  for (float & value : embedding) {
    value = static_cast<float>(static_cast<double>(value) * scale);
  }
}

void parse_token_list(const char * token_text, std::vector<llama_token> & out_tokens) {
  out_tokens.clear();
  if (token_text == nullptr) {
    return;
  }

  const char * p = token_text;
  while (*p != '\0') {
    while (*p != '\0' &&
           !std::isdigit(static_cast<unsigned char>(*p)) &&
           *p != '-' &&
           *p != '+') {
      ++p;
    }

    if (*p == '\0') {
      break;
    }

    char * end = nullptr;
    const long value = std::strtol(p, &end, 10);
    if (end == p) {
      ++p;
      continue;
    }

    out_tokens.push_back(static_cast<llama_token>(value));
    p = end;
  }
}

void replace_all_inplace(
    std::string & text,
    const std::string & from,
    const std::string & to) {
  if (from.empty()) {
    return;
  }

  size_t start = 0;
  while (true) {
    const size_t pos = text.find(from, start);
    if (pos == std::string::npos) {
      break;
    }

    text.replace(pos, from.size(), to);
    start = pos + to.size();
  }
}

size_t count_occurrences(const std::string & text, const std::string & pattern) {
  if (pattern.empty()) {
    return 0;
  }

  size_t count = 0;
  size_t start = 0;
  while (true) {
    const size_t pos = text.find(pattern, start);
    if (pos == std::string::npos) {
      break;
    }
    ++count;
    start = pos + pattern.size();
  }
  return count;
}
