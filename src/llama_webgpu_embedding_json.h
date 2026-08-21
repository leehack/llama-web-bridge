#ifndef LLAMADART_WEBGPU_EMBEDDING_JSON_H
#define LLAMADART_WEBGPU_EMBEDDING_JSON_H

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace llamadart_webgpu_detail {

inline std::string json_number(const float value) {
  if (!std::isfinite(value)) {
    return "0";
  }

  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.9g", static_cast<double>(value));
  return buffer;
}

inline std::string serialize_embedding_json(
    const std::vector<float> & embedding) {
  std::string json = "[";
  for (std::size_t i = 0; i < embedding.size(); ++i) {
    if (i > 0) {
      json += ",";
    }
    json += json_number(embedding[i]);
  }
  json += "]";
  return json;
}

}  // namespace llamadart_webgpu_detail

#endif  // LLAMADART_WEBGPU_EMBEDDING_JSON_H
