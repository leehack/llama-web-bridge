#ifndef LLAMADART_WEBGPU_NEXT_TOKEN_SCORES_H
#define LLAMADART_WEBGPU_NEXT_TOKEN_SCORES_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace llamadart_webgpu_detail {

struct ScoredToken {
  int32_t token;
  std::string bytes;
  double logprob;
};

// NaN logits rank below every other logit, so sorting stays well defined.
inline double rank_logit(const float logit) {
  return std::isnan(logit) ? -std::numeric_limits<double>::infinity()
                           : static_cast<double>(logit);
}

// log(sum(exp(logits))), computed from the largest logit for stability.
inline double log_sum_exp(const float * logits, const int32_t n_vocab) {
  double max_logit = -std::numeric_limits<double>::infinity();
  for (int32_t i = 0; i < n_vocab; ++i) {
    max_logit = std::max(max_logit, rank_logit(logits[i]));
  }
  if (!std::isfinite(max_logit)) {
    return max_logit;
  }
  double sum = 0.0;
  for (int32_t i = 0; i < n_vocab; ++i) {
    sum += std::exp(rank_logit(logits[i]) - max_logit);
  }
  return max_logit + std::log(sum);
}

// Ids of the k largest logits, largest first; ties keep the lower id.
inline std::vector<int32_t> top_token_ids(
    const float * logits,
    const int32_t n_vocab,
    const int32_t k) {
  const int32_t count = std::max(0, std::min(k, n_vocab));
  std::vector<int32_t> ids(static_cast<size_t>(n_vocab));
  std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(
      ids.begin(),
      ids.begin() + count,
      ids.end(),
      [logits](const int32_t a, const int32_t b) {
        const double la = rank_logit(logits[a]);
        const double lb = rank_logit(logits[b]);
        return la > lb || (la == lb && a < b);
      });
  ids.resize(static_cast<size_t>(count));
  return ids;
}

// A finite log-probability as a JSON number; anything else as null.
inline std::string json_logprob(const double value) {
  if (!std::isfinite(value)) {
    return "null";
  }
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.17g", value);
  return buffer;
}

inline void append_scored_tokens_json(
    std::string & json,
    const std::vector<ScoredToken> & tokens) {
  json += "[";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0) {
      json += ",";
    }
    json += "{\"token\":";
    json += std::to_string(tokens[i].token);
    json += ",\"bytes\":[";
    for (size_t b = 0; b < tokens[i].bytes.size(); ++b) {
      if (b > 0) {
        json += ",";
      }
      json += std::to_string(static_cast<unsigned char>(tokens[i].bytes[b]));
    }
    json += "],\"logprob\":";
    json += json_logprob(tokens[i].logprob);
    json += "}";
  }
  json += "]";
}

inline std::string serialize_next_token_scores_json(
    const std::vector<ScoredToken> & candidates,
    const std::vector<ScoredToken> & top,
    const int32_t prompt_tokens) {
  std::string json = "{\"candidates\":";
  append_scored_tokens_json(json, candidates);
  json += ",\"top\":";
  append_scored_tokens_json(json, top);
  json += ",\"promptTokens\":";
  json += std::to_string(prompt_tokens);
  json += "}";
  return json;
}

}  // namespace llamadart_webgpu_detail

#endif  // LLAMADART_WEBGPU_NEXT_TOKEN_SCORES_H
