#include "llama_webgpu_decision.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "nlohmann/json.hpp"

// The bridge must not depend on C++ exception catching, so every nlohmann call
// below uses a non-throwing form: parse without exceptions, typed accessors
// only after a type check, and dump with invalid UTF-8 replaced.
using ordered_json = nlohmann::ordered_json;

struct llama_webgpu_decision_layer {
  ggml_tensor * norm1_weight = nullptr;
  ggml_tensor * norm1_bias = nullptr;
  ggml_tensor * query_weight = nullptr;
  ggml_tensor * query_bias = nullptr;
  ggml_tensor * key_weight = nullptr;
  ggml_tensor * key_bias = nullptr;
  ggml_tensor * value_weight = nullptr;
  ggml_tensor * value_bias = nullptr;
  ggml_tensor * out_weight = nullptr;
  ggml_tensor * out_bias = nullptr;
  ggml_tensor * norm2_weight = nullptr;
  ggml_tensor * norm2_bias = nullptr;
  ggml_tensor * linear1_weight = nullptr;
  ggml_tensor * linear1_bias = nullptr;
  ggml_tensor * linear2_weight = nullptr;
  ggml_tensor * linear2_bias = nullptr;
};

struct llama_webgpu_decision_head {
  llama_context * context = nullptr;
  ggml_backend_t cpu_backend = nullptr;
  ggml_backend_t device_backend = nullptr;
  ggml_context * weights_context = nullptr;
  ggml_backend_buffer_t weights_buffer = nullptr;
  ggml_backend_sched_t sched = nullptr;
  std::vector<llama_webgpu_decision_layer> layers;
  ggml_tensor * scorer_norm_weight = nullptr;
  ggml_tensor * scorer_norm_bias = nullptr;
  ggml_tensor * scorer_hidden_weight = nullptr;
  ggml_tensor * scorer_hidden_bias = nullptr;
  ggml_tensor * scorer_out_weight = nullptr;
  ggml_tensor * scorer_out_bias = nullptr;
  std::vector<float> type_embedding;
  std::vector<float> act_hidden_weight;
  std::vector<float> act_hidden_bias;
  std::vector<float> act_out_weight;
  std::vector<float> act_out_bias;
  int64_t hidden_size = 0;
  int64_t heads = 0;
  size_t graph_size = 0;
  int32_t token_limit = 0;
  int32_t vocab_size = 0;
  llama_webgpu_decision_head_info info;
};

namespace {

constexpr float kLayerNormEpsilon = 1e-5f;
constexpr uint64_t kMaxHeaderBytes = 100ull * 1024ull * 1024ull;
// nlohmann copies and dumps JSON recursively, so untrusted text nested deeper
// than this is rejected before parsing. Real headers and configs nest 3 deep.
constexpr int kMaxJsonDepth = 64;
constexpr const char * kEncoderArchitecture = "modern-bert";

static_assert(sizeof(off_t) == 8, "safetensors reads need 64-bit file offsets");

std::string quote_name(const std::string & value) {
  return "\"" + value + "\"";
}

std::string format_shape(const std::vector<int64_t> & shape) {
  std::string text = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      text += ", ";
    }
    text += std::to_string(shape[i]);
  }
  return text + "]";
}

// Linear scan for arrays and objects nested deeper than kMaxJsonDepth,
// skipping string contents and their escapes.
bool json_depth_within_limit(const std::string & text) {
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (const char c : text) {
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
    } else if (c == '"') {
      in_string = true;
    } else if (c == '[' || c == '{') {
      if (++depth > kMaxJsonDepth) {
        return false;
      }
    } else if ((c == ']' || c == '}') && depth > 0) {
      --depth;
    }
  }
  return true;
}

std::string dump_json(const ordered_json & value) {
  return value.dump(-1, ' ', false, ordered_json::error_handler_t::replace);
}

std::string describe_json(const ordered_json & value) {
  if (value.is_string()) {
    return value.get_ref<const std::string &>();
  }
  return dump_json(value);
}

bool json_int64(const ordered_json & value, int64_t & out) {
  if (value.is_number_unsigned()) {
    const uint64_t unsigned_value = value.get<uint64_t>();
    if (unsigned_value >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return false;
    }
    out = static_cast<int64_t>(unsigned_value);
    return true;
  }
  if (value.is_number_integer()) {
    out = value.get<int64_t>();
    return true;
  }
  return false;
}

size_t dtype_bytes(const std::string & dtype) {
  static const std::map<std::string, size_t> table = {
      {"BOOL", 1}, {"U8", 1},   {"I8", 1},   {"F8_E5M2", 1}, {"F8_E4M3", 1},
      {"I16", 2},  {"U16", 2},  {"F16", 2},  {"BF16", 2},    {"I32", 4},
      {"U32", 4},  {"F32", 4},  {"I64", 8},  {"U64", 8},     {"F64", 8},
  };
  const auto it = table.find(dtype);
  return it == table.end() ? 0 : it->second;
}

uint32_t half_bits_to_float_bits(uint32_t half) {
  const uint32_t sign = (half & 0x8000u) << 16;
  const uint32_t exponent = (half >> 10) & 0x1fu;
  uint32_t mantissa = half & 0x3ffu;
  if (exponent == 0x1fu) {
    return sign | 0x7f800000u | (mantissa << 13);
  }
  if (exponent != 0) {
    return sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  if (mantissa == 0) {
    return sign;
  }
  uint32_t float_exponent = 113;
  while ((mantissa & 0x400u) == 0) {
    mantissa <<= 1;
    --float_exponent;
  }
  return sign | (float_exponent << 23) | ((mantissa & 0x3ffu) << 13);
}

float bits_to_float(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

struct safetensors_tensor {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t begin = 0;
  uint64_t end = 0;
};

// A safetensors file whose header is parsed up front and whose tensor bytes
// are read on demand, with the bounds checks of llamadart's native parser.
class safetensors_file {
 public:
  safetensors_file() = default;
  safetensors_file(const safetensors_file &) = delete;
  safetensors_file & operator=(const safetensors_file &) = delete;
  ~safetensors_file() {
    if (file_ != nullptr) {
      std::fclose(file_);
    }
  }

  bool open(const char * path, const std::string & label, std::string & error) {
    label_ = label;
    file_ = std::fopen(path, "rb");
    if (file_ == nullptr) {
      error = "Cannot open safetensors file " + quote_name(label_) + ": " +
          std::strerror(errno) + ".";
      return false;
    }
    if (fseeko(file_, 0, SEEK_END) != 0) {
      return io_error(error);
    }
    const off_t size = ftello(file_);
    if (size < 0) {
      return io_error(error);
    }
    const uint64_t file_length = static_cast<uint64_t>(size);
    const auto malformed = [&](const std::string & reason) {
      error = "Invalid safetensors file " + quote_name(label_) + ": " + reason + ".";
      return false;
    };
    if (file_length < 8) {
      return malformed(
          std::to_string(file_length) +
          " bytes is too short for the 8-byte header length");
    }
    unsigned char prefix[8];
    if (!read_exact(0, prefix, sizeof(prefix), error)) {
      return false;
    }
    uint64_t header_length = 0;
    for (int i = 7; i >= 0; --i) {
      header_length = (header_length << 8) | prefix[i];
    }
    if (header_length < 2 || header_length > file_length - 8 ||
        header_length > kMaxHeaderBytes) {
      return malformed(
          "header length " + std::to_string(header_length) +
          " does not fit a " + std::to_string(file_length) +
          "-byte file (at most " + std::to_string(kMaxHeaderBytes) + " bytes)");
    }
    std::string header_text(static_cast<size_t>(header_length), '\0');
    if (!read_exact(8, header_text.data(), header_length, error)) {
      return false;
    }
    if (!json_depth_within_limit(header_text)) {
      return malformed(
          "header nests JSON deeper than " + std::to_string(kMaxJsonDepth) +
          " levels");
    }
    const ordered_json header = ordered_json::parse(header_text, nullptr, false);
    if (header.is_discarded()) {
      return malformed("header is not UTF-8 JSON");
    }
    if (!header.is_object()) {
      return malformed("header is not a JSON object");
    }

    data_start_ = 8 + header_length;
    const uint64_t data_length = file_length - data_start_;
    for (auto it = header.begin(); it != header.end(); ++it) {
      const std::string & name = it.key();
      const ordered_json & value = it.value();
      if (name == "__metadata__") {
        if (!value.is_object()) {
          return malformed("__metadata__ is not a map of strings");
        }
        for (auto entry = value.begin(); entry != value.end(); ++entry) {
          if (!entry.value().is_string()) {
            return malformed("__metadata__ is not a map of strings");
          }
        }
        metadata.clear();
        for (auto entry = value.begin(); entry != value.end(); ++entry) {
          metadata[entry.key()] =
              entry.value().get_ref<const std::string &>();
        }
        continue;
      }
      const std::string tensor = "tensor " + quote_name(name);
      if (!value.is_object()) {
        return malformed(tensor + " is not a JSON object");
      }
      const auto dtype = value.find("dtype");
      if (dtype == value.end() || !dtype->is_string()) {
        return malformed(tensor + " has no string dtype");
      }
      const auto shape = value.find("shape");
      const std::string shape_text =
          shape == value.end() ? "null" : dump_json(*shape);
      if (shape == value.end() || !shape->is_array()) {
        return malformed(tensor + " shape " + shape_text + " is not a list of sizes");
      }
      std::vector<int64_t> dims;
      dims.reserve(shape->size());
      for (const auto & dim_value : *shape) {
        int64_t dim = 0;
        if (!json_int64(dim_value, dim) || dim < 0) {
          return malformed(
              tensor + " shape " + shape_text + " is not a list of sizes");
        }
        dims.push_back(dim);
      }
      const auto offsets = value.find("data_offsets");
      int64_t begin = 0;
      int64_t end = 0;
      if (offsets == value.end() || !offsets->is_array() ||
          offsets->size() != 2 || !json_int64((*offsets)[0], begin) ||
          !json_int64((*offsets)[1], end)) {
        return malformed(
            tensor + " data_offsets " +
            (offsets == value.end() ? std::string("null") : dump_json(*offsets)) +
            " is not [begin, end]");
      }
      if (begin < 0 || begin > end ||
          static_cast<uint64_t>(end) > data_length) {
        return malformed(
            tensor + " data_offsets [" + std::to_string(begin) + ", " +
            std::to_string(end) + "] fall outside the " +
            std::to_string(data_length) + "-byte data section");
      }
      const std::string & dtype_name = dtype->get_ref<const std::string &>();
      const size_t element_bytes = dtype_bytes(dtype_name);
      if (element_bytes != 0) {
        uint64_t elements =
            std::find(dims.begin(), dims.end(), 0) != dims.end() ? 0 : 1;
        for (const int64_t dim : dims) {
          const uint64_t divisor = static_cast<uint64_t>(std::max<int64_t>(dim, 1));
          if (elements > data_length / divisor) {
            elements = data_length + 1;
            break;
          }
          elements *= static_cast<uint64_t>(dim);
        }
        if (elements > data_length ||
            elements * element_bytes != static_cast<uint64_t>(end - begin)) {
          return malformed(
              tensor + " is " + dtype_name + " " + format_shape(dims) +
              " but spans " + std::to_string(end - begin) + " bytes");
        }
      }
      tensors[name] = safetensors_tensor{
          dtype_name,
          dims,
          static_cast<uint64_t>(begin),
          static_cast<uint64_t>(end),
      };
    }
    return true;
  }

  const safetensors_tensor * find(const std::string & name) const {
    const auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : &it->second;
  }

  static bool converts_to_f32(const std::string & dtype) {
    return dtype == "F32" || dtype == "F16" || dtype == "BF16";
  }

  // Reads tensor `name` and converts F32, F16 or BF16 to F32.
  bool read_f32(const std::string & name, std::vector<float> & out, std::string & error) {
    const safetensors_tensor * tensor = find(name);
    if (tensor == nullptr) {
      error = "Safetensors file " + quote_name(label_) + " has no tensor " +
          quote_name(name) + ".";
      return false;
    }
    if (!converts_to_f32(tensor->dtype)) {
      error = "Tensor " + quote_name(name) + " in " + quote_name(label_) + " is " +
          tensor->dtype + "; only F32, F16 and BF16 tensors convert to F32.";
      return false;
    }
    const uint64_t length = tensor->end - tensor->begin;
    const uint64_t position = data_start_ + tensor->begin;
    if (tensor->dtype == "F32") {
      out.resize(static_cast<size_t>(length / sizeof(float)));
      return read_exact(position, out.data(), length, error);
    }
    std::vector<uint16_t> halves(static_cast<size_t>(length / sizeof(uint16_t)));
    if (!read_exact(position, halves.data(), length, error)) {
      return false;
    }
    out.resize(halves.size());
    const bool is_f16 = tensor->dtype == "F16";
    for (size_t i = 0; i < halves.size(); ++i) {
      out[i] = bits_to_float(
          is_f16 ? half_bits_to_float_bits(halves[i])
                 : static_cast<uint32_t>(halves[i]) << 16);
    }
    return true;
  }

  std::map<std::string, std::string> metadata;
  std::map<std::string, safetensors_tensor> tensors;

 private:
  bool io_error(std::string & error) {
    error = "Cannot read safetensors file " + quote_name(label_) + ": " +
        std::strerror(errno) + ".";
    return false;
  }

  bool read_exact(uint64_t position, void * destination, uint64_t length, std::string & error) {
    if (position > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
        fseeko(file_, static_cast<off_t>(position), SEEK_SET) != 0) {
      return io_error(error);
    }
    const size_t read = std::fread(destination, 1, static_cast<size_t>(length), file_);
    if (read != length) {
      error = "Safetensors file " + quote_name(label_) + " ended " +
          std::to_string(read) + " bytes into a " + std::to_string(length) +
          "-byte read at offset " + std::to_string(position) + ".";
      return false;
    }
    return true;
  }

  FILE * file_ = nullptr;
  std::string label_;
  uint64_t data_start_ = 0;
};

// Mirrors llamadart's DecisionHeadConfig.fromJson checks; only max_len is kept.
bool read_positive_int(
    const ordered_json & config,
    const char * key,
    int64_t fallback,
    int64_t & out,
    std::string & error) {
  const auto it = config.find(key);
  if (it == config.end() || it->is_null()) {
    out = fallback;
    return true;
  }
  if (json_int64(*it, out) && out > 0) {
    return true;
  }
  error = std::string("Decision head \"") + key +
      "\" must be a positive integer, got " + describe_json(*it) + ".";
  return false;
}

bool parse_config(
    const std::string & text,
    const std::string & source,
    ordered_json & config,
    int64_t & max_tokens,
    std::string & error) {
  const auto invalid = [&](const std::string & reason) {
    error = "The decision head config in " + source + " is invalid: " + reason;
    return false;
  };
  if (!json_depth_within_limit(text)) {
    return invalid(
        "Decision head config nests JSON deeper than " +
        std::to_string(kMaxJsonDepth) + " levels.");
  }
  config = ordered_json::parse(text, nullptr, false);
  if (config.is_discarded()) {
    return invalid("Decision head config is not valid JSON.");
  }
  if (!config.is_object()) {
    return invalid("Decision head config is not a JSON object.");
  }
  const auto temperature = config.find("temperature");
  if (temperature != config.end() && !temperature->is_null() &&
      (!temperature->is_array() || temperature->size() < 3)) {
    return invalid(
        "Decision head \"temperature\" must be a list of at least 3 values, got " +
        describe_json(*temperature) + ".");
  }
  const auto by_options = config.find("temperature_by_options");
  if (by_options != config.end() && !by_options->is_null() &&
      !by_options->is_object()) {
    return invalid(
        "Decision head \"temperature_by_options\" must be a map, got " +
        describe_json(*by_options) + ".");
  }
  std::string field_error;
  int64_t head_max_tokens = 0;
  if (!read_positive_int(config, "max_len", 512, max_tokens, field_error) ||
      !read_positive_int(config, "head_max_len", 192, head_max_tokens, field_error)) {
    return invalid(field_error);
  }
  return true;
}

// Checks a shape against expectations and names the tensor on failure.
class shape_check {
 public:
  shape_check(const safetensors_file & file, const std::string & label)
      : file_(file), label_(label) {}

  const std::vector<int64_t> * shape(const std::string & name, std::string & error) const {
    const safetensors_tensor * tensor = file_.find(name);
    if (tensor == nullptr) {
      error = "Decision head file " + quote_name(label_) + " has no tensor " +
          quote_name(name) + ".";
      return nullptr;
    }
    return &tensor->shape;
  }

  bool expect(const std::string & name, const std::vector<int64_t> & expected, std::string & error) const {
    const std::vector<int64_t> * found = shape(name, error);
    if (found == nullptr) {
      return false;
    }
    if (*found != expected) {
      error = "Decision head tensor " + quote_name(name) + " in " + quote_name(label_) +
          " has shape " + format_shape(*found) + "; expected " +
          format_shape(expected) + ".";
      return false;
    }
    return true;
  }

  bool rows(
      const std::string & name,
      int64_t columns,
      const std::string & row_name,
      int64_t & out_rows,
      std::string & error) const {
    const std::vector<int64_t> * found = shape(name, error);
    if (found == nullptr) {
      return false;
    }
    if (found->size() != 2 || (*found)[0] < 1 || (*found)[1] != columns) {
      error = "Decision head tensor " + quote_name(name) + " in " + quote_name(label_) +
          " has shape " + format_shape(*found) + "; expected [" + row_name +
          ", " + std::to_string(columns) + "] with " + row_name + " >= 1.";
      return false;
    }
    out_rows = (*found)[0];
    return true;
  }

 private:
  const safetensors_file & file_;
  std::string label_;
};

struct head_layout {
  int64_t hidden = 0;
  int64_t heads = 0;
  int64_t layers = 0;
  int64_t ffn = 0;
  int64_t act_hidden = 0;
  int64_t act_classes = 0;
};

std::string layer_prefix(int64_t index) {
  return "head.layers." + std::to_string(index);
}

// Mirrors DecisionHeadWeights.read: config head_layers, then every tensor's
// exact shape, then the dtype of every tensor the head reads.
bool check_head_layout(
    const safetensors_file & file,
    const std::string & label,
    const ordered_json & config,
    int64_t hidden,
    head_layout & layout,
    std::string & error) {
  if (hidden < 1) {
    error = "Decision head hidden size must be positive, got " +
        std::to_string(hidden) + ".";
    return false;
  }
  const int64_t heads = std::max<int64_t>(1, hidden / 64);
  if (hidden % heads != 0) {
    error = "Decision head hidden size " + std::to_string(hidden) +
        " is not divisible by its " + std::to_string(heads) +
        " attention heads.";
    return false;
  }
  int64_t layers = 2;
  const auto layers_value = config.find("head_layers");
  if (layers_value != config.end() && !layers_value->is_null() &&
      (!json_int64(*layers_value, layers) || layers < 1)) {
    error = "Decision head config \"head_layers\" must be a positive integer, got " +
        describe_json(*layers_value) + ".";
    return false;
  }
  const std::string extra_layer = layer_prefix(layers) + ".";
  for (const auto & entry : file.tensors) {
    if (entry.first.compare(0, extra_layer.size(), extra_layer) == 0) {
      error = "Decision head file " + quote_name(label) +
          " has tensors for more than the " + std::to_string(layers) +
          " layers its config \"head_layers\" names.";
      return false;
    }
  }

  const shape_check shapes(file, label);
  const int64_t d = hidden;
  int64_t ffn = 0;
  if (!shapes.rows("head.layers.0.linear1.weight", d, "ffn", ffn, error) ||
      !shapes.expect("type_emb.weight", {3, d}, error)) {
    return false;
  }
  for (int64_t i = 0; i < layers; ++i) {
    const std::string p = layer_prefix(i);
    if (!shapes.expect(p + ".self_attn.in_proj_weight", {3 * d, d}, error) ||
        !shapes.expect(p + ".self_attn.in_proj_bias", {3 * d}, error) ||
        !shapes.expect(p + ".self_attn.out_proj.weight", {d, d}, error) ||
        !shapes.expect(p + ".self_attn.out_proj.bias", {d}, error) ||
        !shapes.expect(p + ".linear1.weight", {ffn, d}, error) ||
        !shapes.expect(p + ".linear1.bias", {ffn}, error) ||
        !shapes.expect(p + ".linear2.weight", {d, ffn}, error) ||
        !shapes.expect(p + ".linear2.bias", {d}, error) ||
        !shapes.expect(p + ".norm1.weight", {d}, error) ||
        !shapes.expect(p + ".norm1.bias", {d}, error) ||
        !shapes.expect(p + ".norm2.weight", {d}, error) ||
        !shapes.expect(p + ".norm2.bias", {d}, error)) {
      return false;
    }
  }
  int64_t act_hidden = 0;
  int64_t act_classes = 0;
  if (!shapes.expect("scorer.0.weight", {d}, error) ||
      !shapes.expect("scorer.0.bias", {d}, error) ||
      !shapes.expect("scorer.1.weight", {d, d}, error) ||
      !shapes.expect("scorer.1.bias", {d}, error) ||
      !shapes.expect("scorer.3.weight", {1, d}, error) ||
      !shapes.expect("scorer.3.bias", {1}, error) ||
      !shapes.rows("act_head.0.weight", d + 4, "act hidden", act_hidden, error) ||
      !shapes.expect("act_head.0.bias", {act_hidden}, error) ||
      !shapes.rows("act_head.2.weight", act_hidden, "classes", act_classes, error) ||
      !shapes.expect("act_head.2.bias", {act_classes}, error)) {
    return false;
  }

  std::vector<std::string> names = {"type_emb.weight"};
  for (int64_t i = 0; i < layers; ++i) {
    const std::string p = layer_prefix(i);
    for (const char * suffix : {
             ".self_attn.in_proj_weight", ".self_attn.in_proj_bias",
             ".self_attn.out_proj.weight", ".self_attn.out_proj.bias",
             ".linear1.weight", ".linear1.bias", ".linear2.weight",
             ".linear2.bias", ".norm1.weight", ".norm1.bias",
             ".norm2.weight", ".norm2.bias"}) {
      names.push_back(p + suffix);
    }
  }
  for (const char * name : {
           "scorer.0.weight", "scorer.0.bias", "scorer.1.weight",
           "scorer.1.bias", "scorer.3.weight", "scorer.3.bias",
           "act_head.0.weight", "act_head.0.bias", "act_head.2.weight",
           "act_head.2.bias"}) {
    names.push_back(name);
  }
  for (const std::string & name : names) {
    const safetensors_tensor * tensor = file.find(name);
    if (!safetensors_file::converts_to_f32(tensor->dtype)) {
      error = "Tensor " + quote_name(name) + " in " + quote_name(label) + " is " +
          tensor->dtype + "; only F32, F16 and BF16 tensors convert to F32.";
      return false;
    }
  }

  layout.hidden = d;
  layout.heads = heads;
  layout.layers = layers;
  layout.ffn = ffn;
  layout.act_hidden = act_hidden;
  layout.act_classes = act_classes;
  return true;
}

bool read_text_file(const char * path, std::string & out) {
  FILE * file = std::fopen(path, "rb");
  if (file == nullptr) {
    return false;
  }
  bool ok = fseeko(file, 0, SEEK_END) == 0;
  const off_t size = ok ? ftello(file) : -1;
  ok = ok && size >= 0 && fseeko(file, 0, SEEK_SET) == 0;
  if (ok) {
    out.assign(static_cast<size_t>(size), '\0');
    ok = std::fread(out.data(), 1, out.size(), file) == out.size();
  }
  std::fclose(file);
  return ok;
}

std::string vocab_text(const llama_vocab * vocab, llama_token token) {
  const char * text = llama_vocab_get_text(vocab, token);
  return text == nullptr ? std::string() : std::string(text);
}

bool model_architecture(const llama_model * model, std::string & out) {
  std::vector<char> buffer(256, '\0');
  int32_t length = llama_model_meta_val_str(
      model, "general.architecture", buffer.data(), buffer.size());
  if (length < 0) {
    return false;
  }
  if (static_cast<size_t>(length) >= buffer.size()) {
    buffer.assign(static_cast<size_t>(length) + 1, '\0');
    length = llama_model_meta_val_str(
        model, "general.architecture", buffer.data(), buffer.size());
    if (length < 0) {
      return false;
    }
  }
  out.assign(buffer.data(), static_cast<size_t>(length));
  return true;
}

void free_head(llama_webgpu_decision_head * head) {
  if (head == nullptr) {
    return;
  }
  // Same teardown order as llamadart's native head: scheduler, weights, then
  // backends, and the private encoder context last.
  if (head->sched != nullptr) {
    ggml_backend_sched_synchronize(head->sched);
    ggml_backend_sched_free(head->sched);
    head->sched = nullptr;
  }
  if (head->weights_buffer != nullptr) {
    ggml_backend_buffer_free(head->weights_buffer);
    head->weights_buffer = nullptr;
  }
  if (head->weights_context != nullptr) {
    ggml_free(head->weights_context);
    head->weights_context = nullptr;
  }
  if (head->device_backend != nullptr) {
    ggml_backend_free(head->device_backend);
    head->device_backend = nullptr;
  }
  if (head->cpu_backend != nullptr) {
    ggml_backend_free(head->cpu_backend);
    head->cpu_backend = nullptr;
  }
  if (head->context != nullptr) {
    llama_free(head->context);
    head->context = nullptr;
  }
  delete head;
}

struct head_deleter {
  void operator()(llama_webgpu_decision_head * head) const {
    free_head(head);
  }
};

using head_ptr = std::unique_ptr<llama_webgpu_decision_head, head_deleter>;

ggml_backend_dev_t find_gpu_device() {
  const size_t count = ggml_backend_dev_count();
  for (size_t i = 0; i < count; ++i) {
    ggml_backend_dev_t device = ggml_backend_dev_get(i);
    if (device == nullptr) {
      continue;
    }
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
    if (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
        type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
      return device;
    }
  }
  return nullptr;
}

struct weight_upload {
  ggml_tensor * tensor;
  std::string source;
  size_t offset;
  size_t count;
};

// Places the head's weights in one backend buffer on the model's device (or
// the CPU) and creates a scheduler that lists the CPU backend last.
bool init_head_runtime(
    llama_webgpu_decision_head * head,
    safetensors_file & file,
    const head_layout & layout,
    bool use_gpu,
    int32_t cpu_threads,
    std::string & error) {
  ggml_backend_dev_t cpu_device =
      ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  if (cpu_device == nullptr) {
    error = "No ggml CPU device is registered; initialize the llama.cpp "
        "backend before loading a decision head.";
    return false;
  }
  head->cpu_backend = ggml_backend_dev_init(cpu_device, nullptr);
  if (head->cpu_backend == nullptr) {
    error = "Could not start the ggml CPU backend for the decision head.";
    return false;
  }
  ggml_backend_reg_t cpu_registry =
      ggml_backend_dev_backend_reg(ggml_backend_get_device(head->cpu_backend));
  if (cpu_registry != nullptr && cpu_threads > 0) {
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(cpu_registry, "ggml_backend_set_n_threads"));
    if (set_threads != nullptr) {
      set_threads(head->cpu_backend, cpu_threads);
    }
  }
  if (use_gpu) {
    ggml_backend_dev_t device = find_gpu_device();
    if (device != nullptr && device != cpu_device) {
      head->device_backend = ggml_backend_dev_init(device, nullptr);
      if (head->device_backend == nullptr) {
        error = "Could not start the ggml backend of the model device for "
            "the decision head.";
        return false;
      }
    }
  }
  ggml_backend_t primary =
      head->device_backend != nullptr ? head->device_backend : head->cpu_backend;
  const char * primary_name = ggml_backend_name(primary);
  head->info.device_name = primary_name != nullptr ? primary_name : "";

  const int64_t d = layout.hidden;
  const size_t tensor_count = static_cast<size_t>(16 * layout.layers + 6);
  ggml_init_params params{};
  params.mem_size = ggml_tensor_overhead() * tensor_count;
  params.mem_buffer = nullptr;
  params.no_alloc = true;
  head->weights_context = ggml_init(params);
  if (head->weights_context == nullptr) {
    error = "Could not create the ggml context for decision head weights.";
    return false;
  }

  std::vector<weight_upload> uploads;
  uploads.reserve(tensor_count);
  const auto vector = [&](const std::string & source, size_t offset, int64_t size) {
    ggml_tensor * tensor =
        ggml_new_tensor_1d(head->weights_context, GGML_TYPE_F32, size);
    uploads.push_back({tensor, source, offset, static_cast<size_t>(size)});
    return tensor;
  };
  const auto matrix = [&](const std::string & source, size_t offset, int64_t columns, int64_t rows) {
    ggml_tensor * tensor =
        ggml_new_tensor_2d(head->weights_context, GGML_TYPE_F32, columns, rows);
    uploads.push_back({tensor, source, offset, static_cast<size_t>(columns * rows)});
    return tensor;
  };

  const size_t dd = static_cast<size_t>(d * d);
  const size_t du = static_cast<size_t>(d);
  for (int64_t i = 0; i < layout.layers; ++i) {
    const std::string p = layer_prefix(i);
    llama_webgpu_decision_layer layer;
    layer.norm1_weight = vector(p + ".norm1.weight", 0, d);
    layer.norm1_bias = vector(p + ".norm1.bias", 0, d);
    const std::string in_weight = p + ".self_attn.in_proj_weight";
    layer.query_weight = matrix(in_weight, 0, d, d);
    layer.key_weight = matrix(in_weight, dd, d, d);
    layer.value_weight = matrix(in_weight, 2 * dd, d, d);
    const std::string in_bias = p + ".self_attn.in_proj_bias";
    layer.query_bias = vector(in_bias, 0, d);
    layer.key_bias = vector(in_bias, du, d);
    layer.value_bias = vector(in_bias, 2 * du, d);
    layer.out_weight = matrix(p + ".self_attn.out_proj.weight", 0, d, d);
    layer.out_bias = vector(p + ".self_attn.out_proj.bias", 0, d);
    layer.norm2_weight = vector(p + ".norm2.weight", 0, d);
    layer.norm2_bias = vector(p + ".norm2.bias", 0, d);
    layer.linear1_weight = matrix(p + ".linear1.weight", 0, d, layout.ffn);
    layer.linear1_bias = vector(p + ".linear1.bias", 0, layout.ffn);
    layer.linear2_weight = matrix(p + ".linear2.weight", 0, layout.ffn, d);
    layer.linear2_bias = vector(p + ".linear2.bias", 0, d);
    head->layers.push_back(layer);
  }
  head->scorer_norm_weight = vector("scorer.0.weight", 0, d);
  head->scorer_norm_bias = vector("scorer.0.bias", 0, d);
  head->scorer_hidden_weight = matrix("scorer.1.weight", 0, d, d);
  head->scorer_hidden_bias = vector("scorer.1.bias", 0, d);
  head->scorer_out_weight = matrix("scorer.3.weight", 0, d, 1);
  head->scorer_out_bias = vector("scorer.3.bias", 0, 1);

  ggml_backend_buffer_type_t buffer_type = ggml_backend_get_default_buffer_type(primary);
  head->weights_buffer =
      ggml_backend_alloc_ctx_tensors_from_buft(head->weights_context, buffer_type);
  if (head->weights_buffer == nullptr) {
    error = "Could not allocate " +
        std::to_string(ggml_backend_alloc_ctx_tensors_from_buft_size(
            head->weights_context, buffer_type)) +
        " bytes for decision head weights on " + head->info.device_name + ".";
    return false;
  }
  ggml_backend_buffer_set_usage(
      head->weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

  std::vector<float> staging;
  std::string staged_source;
  for (const weight_upload & upload : uploads) {
    if (upload.source != staged_source) {
      if (!file.read_f32(upload.source, staging, error)) {
        return false;
      }
      staged_source = upload.source;
    }
    ggml_backend_tensor_set(
        upload.tensor,
        staging.data() + upload.offset,
        0,
        upload.count * sizeof(float));
  }
  staging.clear();
  staging.shrink_to_fit();

  if (!file.read_f32("type_emb.weight", head->type_embedding, error) ||
      !file.read_f32("act_head.0.weight", head->act_hidden_weight, error) ||
      !file.read_f32("act_head.0.bias", head->act_hidden_bias, error) ||
      !file.read_f32("act_head.2.weight", head->act_out_weight, error) ||
      !file.read_f32("act_head.2.bias", head->act_out_bias, error)) {
    return false;
  }

  head->graph_size = static_cast<size_t>(64 + 64 * layout.layers);
  ggml_backend_t backends[2];
  int backend_count = 0;
  if (head->device_backend != nullptr) {
    backends[backend_count++] = head->device_backend;
  }
  backends[backend_count++] = head->cpu_backend;
  head->sched = ggml_backend_sched_new(
      backends,
      nullptr,
      backend_count,
      std::max<size_t>(2048, head->graph_size),
      false,
      head->device_backend != nullptr);
  if (head->sched == nullptr) {
    error = "Could not create the ggml scheduler for the decision head on " +
        head->info.device_name + ".";
    return false;
  }
  return true;
}

struct graph_context_deleter {
  void operator()(ggml_context * context) const {
    ggml_free(context);
  }
};

// Runs the head graph on one sequence's encoder output. Returns one raw logit
// per marker and the CLS row after the head layers.
bool compute_head_graph(
    llama_webgpu_decision_head * head,
    const float * hidden,
    int64_t token_count,
    int32_t question_type,
    const std::vector<int32_t> & markers,
    std::vector<float> & logits,
    std::vector<float> & cls,
    std::string & error) {
  const int64_t d = head->hidden_size;
  const int64_t n = token_count;
  const int64_t head_size = d / head->heads;
  const int64_t row_count = static_cast<int64_t>(markers.size()) + 1;
  ggml_init_params params{};
  params.mem_size = ggml_tensor_overhead() * head->graph_size +
      ggml_graph_overhead_custom(head->graph_size, false);
  params.mem_buffer = nullptr;
  params.no_alloc = true;
  std::unique_ptr<ggml_context, graph_context_deleter> graph_context(ggml_init(params));
  ggml_context * g = graph_context.get();
  if (g == nullptr) {
    error = "Could not create the decision head graph context.";
    return false;
  }

  const auto norm = [&](ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias) {
    return ggml_add(g, ggml_mul(g, ggml_norm(g, x, kLayerNormEpsilon), weight), bias);
  };
  const auto linear = [&](ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias) {
    return ggml_add(g, ggml_mul_mat(g, weight, x), bias);
  };
  const auto split_heads = [&](ggml_tensor * x) {
    return ggml_permute(g, ggml_reshape_3d(g, x, head_size, head->heads, n), 0, 2, 1, 3);
  };

  ggml_tensor * hidden_input = ggml_new_tensor_2d(g, GGML_TYPE_F32, d, n);
  ggml_tensor * type_input = ggml_new_tensor_1d(g, GGML_TYPE_F32, d);
  ggml_tensor * rows_input = ggml_new_tensor_1d(g, GGML_TYPE_I32, row_count);
  ggml_set_input(hidden_input);
  ggml_set_input(type_input);
  ggml_set_input(rows_input);

  ggml_tensor * x = ggml_add(g, hidden_input, type_input);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  for (const llama_webgpu_decision_layer & layer : head->layers) {
    ggml_tensor * a = norm(x, layer.norm1_weight, layer.norm1_bias);
    ggml_tensor * q = split_heads(linear(a, layer.query_weight, layer.query_bias));
    ggml_tensor * k = split_heads(linear(a, layer.key_weight, layer.key_bias));
    ggml_tensor * v = split_heads(linear(a, layer.value_weight, layer.value_bias));
    ggml_tensor * scores =
        ggml_soft_max_ext(g, ggml_mul_mat(g, k, q), nullptr, scale, 0.0f);
    ggml_tensor * attended =
        ggml_mul_mat(g, ggml_cont(g, ggml_transpose(g, v)), scores);
    ggml_tensor * merged =
        ggml_cont_2d(g, ggml_permute(g, attended, 0, 2, 1, 3), d, n);
    x = ggml_add(g, x, linear(merged, layer.out_weight, layer.out_bias));
    ggml_tensor * ff = norm(x, layer.norm2_weight, layer.norm2_bias);
    x = ggml_add(
        g,
        x,
        linear(
            ggml_relu(g, linear(ff, layer.linear1_weight, layer.linear1_bias)),
            layer.linear2_weight,
            layer.linear2_bias));
  }
  ggml_tensor * rows = ggml_get_rows(g, x, rows_input);
  ggml_set_output(rows);
  ggml_tensor * scores = norm(rows, head->scorer_norm_weight, head->scorer_norm_bias);
  scores = ggml_gelu_erf(
      g, linear(scores, head->scorer_hidden_weight, head->scorer_hidden_bias));
  scores = linear(scores, head->scorer_out_weight, head->scorer_out_bias);
  ggml_set_output(scores);

  ggml_cgraph * graph = ggml_new_graph_custom(g, head->graph_size, false);
  ggml_build_forward_expand(graph, scores);
  ggml_build_forward_expand(graph, rows);
  ggml_backend_sched_reset(head->sched);
  if (!ggml_backend_sched_alloc_graph(head->sched, graph)) {
    error = "Could not allocate decision head compute buffers on " +
        head->info.device_name + " for " + std::to_string(n) + " tokens.";
    return false;
  }

  ggml_backend_tensor_set(
      hidden_input, hidden, 0, static_cast<size_t>(n * d) * sizeof(float));
  ggml_backend_tensor_set(
      type_input,
      head->type_embedding.data() + static_cast<size_t>(question_type * d),
      0,
      static_cast<size_t>(d) * sizeof(float));
  std::vector<int32_t> row_ids(static_cast<size_t>(row_count));
  row_ids[0] = 0;
  std::copy(markers.begin(), markers.end(), row_ids.begin() + 1);
  ggml_backend_tensor_set(
      rows_input, row_ids.data(), 0, row_ids.size() * sizeof(int32_t));

  const ggml_status status = ggml_backend_sched_graph_compute(head->sched, graph);
  if (status != GGML_STATUS_SUCCESS) {
    error = "Decision head compute failed on " + head->info.device_name +
        " (ggml status " + std::to_string(static_cast<int>(status)) + ").";
    return false;
  }
  std::vector<float> all_scores(static_cast<size_t>(row_count));
  ggml_backend_tensor_get(scores, all_scores.data(), 0, all_scores.size() * sizeof(float));
  logits.assign(all_scores.begin() + 1, all_scores.end());
  cls.resize(static_cast<size_t>(d));
  ggml_backend_tensor_get(rows, cls.data(), 0, cls.size() * sizeof(float));
  return true;
}

// Act-head features from untempered marker logits: top1, top1 - top2 (top2 is
// 0 for one option), entropy over ln(max(K, 2)) with p clipped to at least 1e-9
// inside the log, and max(K, 2) / 255.
std::array<double, 4> act_features(const std::vector<float> & raw_logits) {
  double top = -std::numeric_limits<double>::infinity();
  for (const float logit : raw_logits) {
    top = std::max(top, static_cast<double>(logit));
  }
  std::vector<double> probabilities(raw_logits.size());
  double sum = 0.0;
  for (size_t i = 0; i < raw_logits.size(); ++i) {
    probabilities[i] = std::exp(static_cast<double>(raw_logits[i]) - top);
    sum += probabilities[i];
  }
  for (double & probability : probabilities) {
    probability /= sum;
  }
  const double k = static_cast<double>(std::max<size_t>(probabilities.size(), 2));
  std::vector<double> sorted = probabilities;
  std::sort(sorted.begin(), sorted.end(), [](double a, double b) { return a > b; });
  const double top1 = sorted[0];
  const double top2 = sorted.size() > 1 ? sorted[1] : 0.0;
  double entropy = 0.0;
  for (const double probability : probabilities) {
    entropy -= probability * std::log(std::max(probability, 1e-9));
  }
  return {top1, top1 - top2, entropy / std::log(k), k / 255.0};
}

// Linear -> GELU(erf) -> Linear on the host in double precision.
std::vector<float> act_logits(
    const llama_webgpu_decision_head * head,
    const std::vector<float> & cls,
    const std::vector<float> & logits) {
  const size_t d = cls.size();
  std::vector<double> inputs(d + 4);
  for (size_t i = 0; i < d; ++i) {
    inputs[i] = cls[i];
  }
  const std::array<double, 4> features = act_features(logits);
  std::copy(features.begin(), features.end(), inputs.begin() + static_cast<long>(d));

  const size_t hidden_count = head->act_hidden_bias.size();
  std::vector<double> hidden(hidden_count);
  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  for (size_t j = 0; j < hidden_count; ++j) {
    double sum = head->act_hidden_bias[j];
    const float * row = head->act_hidden_weight.data() + j * inputs.size();
    for (size_t i = 0; i < inputs.size(); ++i) {
      sum += static_cast<double>(row[i]) * inputs[i];
    }
    hidden[j] = 0.5 * sum * (1.0 + std::erf(sum * inv_sqrt2));
  }
  const size_t class_count = head->act_out_bias.size();
  std::vector<float> result(class_count);
  for (size_t c = 0; c < class_count; ++c) {
    double sum = head->act_out_bias[c];
    const float * row = head->act_out_weight.data() + c * hidden_count;
    for (size_t j = 0; j < hidden_count; ++j) {
      sum += static_cast<double>(row[j]) * hidden[j];
    }
    result[c] = static_cast<float>(sum);
  }
  return result;
}

}  // namespace

std::string llama_webgpu_decision_unsupported_reason(const llama_model * model) {
  if (model == nullptr) {
    return "Model is not loaded. Load a ModernBERT encoder GGUF first.";
  }
  std::string architecture;
  const bool has_architecture = model_architecture(model, architecture);
  if (!has_architecture || architecture != kEncoderArchitecture) {
    const std::string reported = has_architecture
        ? "architecture " + quote_name(architecture)
        : std::string("no architecture");
    return std::string("Decision heads need a ModernBERT encoder GGUF "
                       "(general.architecture \"") +
        kEncoderArchitecture + "\"); the loaded model reports " + reported + ".";
  }
  const llama_vocab * vocab = llama_model_get_vocab(model);
  const int32_t vocab_size = llama_vocab_n_tokens(vocab);
  const std::pair<const char *, llama_token> specials[] = {
      {"CLS", llama_vocab_bos(vocab)},
      {"SEP", llama_vocab_sep(vocab)},
      {"MASK", llama_vocab_mask(vocab)},
  };
  for (const auto & special : specials) {
    if (special.second < 0 || special.second >= vocab_size) {
      return std::string("The loaded encoder has no ") + special.first +
          " token, which decision sequences need. Convert the GGUF with its "
          "tokenizer's special tokens.";
    }
  }
  if (vocab_text(vocab, llama_vocab_mask(vocab)).empty()) {
    return "The loaded encoder's MASK token has no text, which decision "
        "prompts need to strip it from user text.";
  }
  const int32_t hidden_size = llama_model_n_embd(model);
  const int32_t output_size = llama_model_n_embd_out(model);
  if (output_size > 0 && output_size != hidden_size) {
    return "The encoder outputs " + std::to_string(output_size) +
        " values per token but its hidden size is " +
        std::to_string(hidden_size) +
        "; decision heads need the last hidden state.";
  }
  return "";
}

std::string llama_webgpu_decision_capabilities_json(const llama_model * model) {
  const std::string reason = llama_webgpu_decision_unsupported_reason(model);
  ordered_json json = ordered_json::object();
  json["apiVersion"] = LLAMADART_WEBGPU_DECISION_API_VERSION;
  json["supported"] = reason.empty();
  if (!reason.empty()) {
    json["reason"] = reason;
  }
  return dump_json(json);
}

std::string llama_webgpu_decision_head_info_json(
    const llama_webgpu_decision_head * head,
    int32_t handle) {
  const llama_webgpu_decision_head_info & info = head->info;
  ordered_json json = ordered_json::object();
  json["apiVersion"] = LLAMADART_WEBGPU_DECISION_API_VERSION;
  json["handle"] = handle;
  json["hiddenSize"] = info.hidden_size;
  json["clsToken"] = info.cls_token;
  json["sepToken"] = info.sep_token;
  json["maskToken"] = info.mask_token;
  json["maskText"] = info.mask_text;
  json["configJson"] = info.config_json;
  json["deviceName"] = info.device_name;
  return dump_json(json);
}

llama_webgpu_decision_status llama_webgpu_decision_head_load(
    const llama_webgpu_decision_load_request & request,
    llama_webgpu_decision_head ** out_head,
    std::string * out_error) {
  std::string scratch_error;
  std::string & error = out_error != nullptr ? *out_error : scratch_error;
  if (out_head == nullptr) {
    error = "Decision head output is missing.";
    return LLAMADART_WEBGPU_DECISION_STATUS_INVALID_ARGUMENT;
  }
  *out_head = nullptr;
  if (request.model == nullptr) {
    error = "Model is not loaded. Load the decision encoder before its head.";
    return LLAMADART_WEBGPU_DECISION_STATUS_INVALID_STATE;
  }
  if (request.head_path == nullptr || request.head_path[0] == '\0') {
    error = "Decision head path is empty.";
    return LLAMADART_WEBGPU_DECISION_STATUS_INVALID_ARGUMENT;
  }
  const std::string unsupported =
      llama_webgpu_decision_unsupported_reason(request.model);
  if (!unsupported.empty()) {
    error = unsupported;
    return LLAMADART_WEBGPU_DECISION_STATUS_UNSUPPORTED;
  }

  llama_model * model = request.model;
  const llama_vocab * vocab = llama_model_get_vocab(model);
  const std::string label =
      request.head_label != nullptr && request.head_label[0] != '\0'
          ? request.head_label
          : "decision head";
  const int32_t hidden_size = llama_model_n_embd(model);

  safetensors_file file;
  if (!file.open(request.head_path, label, error)) {
    return LLAMADART_WEBGPU_DECISION_STATUS_MODEL_ERROR;
  }

  std::string config_text;
  std::string config_source;
  if (request.config_path != nullptr) {
    if (!read_text_file(request.config_path, config_text)) {
      error = std::string("Decision head configJson could not be read: ") +
          std::strerror(errno) + ".";
      return LLAMADART_WEBGPU_DECISION_STATUS_INVALID_ARGUMENT;
    }
    config_source = "configJson";
  } else {
    const auto metadata = file.metadata.find("laya.config");
    if (metadata == file.metadata.end()) {
      error = "The decision head at " + quote_name(label) +
          " has no \"laya.config\" metadata. Pass configJson with the head's "
          "rl_agent_config.json.";
      return LLAMADART_WEBGPU_DECISION_STATUS_MODEL_ERROR;
    }
    config_text = metadata->second;
    config_source = quote_name(label) + " (laya.config metadata)";
  }
  ordered_json config;
  int64_t max_tokens = 0;
  if (!parse_config(config_text, config_source, config, max_tokens, error)) {
    return LLAMADART_WEBGPU_DECISION_STATUS_MODEL_ERROR;
  }

  const safetensors_tensor * type_embedding = file.find("type_emb.weight");
  if (type_embedding != nullptr && type_embedding->shape.size() == 2 &&
      type_embedding->shape[1] != hidden_size) {
    error = "The decision head at " + quote_name(label) + " is " +
        std::to_string(type_embedding->shape[1]) +
        " wide but the loaded encoder has hidden size " +
        std::to_string(hidden_size) + ". Use the head trained for this encoder.";
    return LLAMADART_WEBGPU_DECISION_STATUS_MODEL_ERROR;
  }
  const int32_t trained_context = llama_model_n_ctx_train(model);
  if (trained_context < max_tokens) {
    error = "The decision head config sets max_len " + std::to_string(max_tokens) +
        ", but the loaded encoder was trained for " +
        std::to_string(trained_context) + " tokens.";
    return LLAMADART_WEBGPU_DECISION_STATUS_MODEL_ERROR;
  }

  head_layout layout;
  if (!check_head_layout(file, label, config, hidden_size, layout, error)) {
    return LLAMADART_WEBGPU_DECISION_STATUS_MODEL_ERROR;
  }

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = static_cast<uint32_t>(max_tokens);
  context_params.n_batch = static_cast<uint32_t>(max_tokens);
  context_params.n_ubatch = static_cast<uint32_t>(max_tokens);
  context_params.n_seq_max = 1;
  context_params.embeddings = true;
  context_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
  if (request.n_threads > 0) {
    context_params.n_threads = request.n_threads;
  }
  if (request.n_threads_batch > 0) {
    context_params.n_threads_batch = request.n_threads_batch;
  }
  if (request.use_gpu) {
    context_params.offload_kqv = true;
    context_params.op_offload = true;
  } else {
    context_params.offload_kqv = false;
    context_params.op_offload = false;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  }
  context_params.no_perf = true;

  head_ptr head(new llama_webgpu_decision_head());
  head->context = llama_init_from_model(model, context_params);
  if (head->context == nullptr) {
    error = "Failed to create the decision encoder context of " +
        std::to_string(max_tokens) + " tokens.";
    return LLAMADART_WEBGPU_DECISION_STATUS_CONTEXT_ERROR;
  }
  if (llama_pooling_type(head->context) != LLAMA_POOLING_TYPE_NONE) {
    error = "The decision encoder context does not return per-token hidden "
        "states (pooling is not NONE).";
    return LLAMADART_WEBGPU_DECISION_STATUS_CONTEXT_ERROR;
  }
  const int64_t token_limit = static_cast<int64_t>(llama_n_ubatch(head->context));
  if (token_limit < max_tokens) {
    error = "The decision encoder context accepts " + std::to_string(token_limit) +
        " tokens per pass, fewer than the head config max_len " +
        std::to_string(max_tokens) + ".";
    return LLAMADART_WEBGPU_DECISION_STATUS_CONTEXT_ERROR;
  }

  head->hidden_size = layout.hidden;
  head->heads = layout.heads;
  head->token_limit = static_cast<int32_t>(token_limit);
  head->vocab_size = llama_vocab_n_tokens(vocab);
  if (!init_head_runtime(
          head.get(),
          file,
          layout,
          request.use_gpu,
          llama_n_threads_batch(head->context),
          error)) {
    return LLAMADART_WEBGPU_DECISION_STATUS_MODEL_ERROR;
  }

  const llama_token mask_token = llama_vocab_mask(vocab);
  head->info.hidden_size = hidden_size;
  head->info.cls_token = llama_vocab_bos(vocab);
  head->info.sep_token = llama_vocab_sep(vocab);
  head->info.mask_token = mask_token;
  head->info.mask_text = vocab_text(vocab, mask_token);
  head->info.config_json = config_text;
  *out_head = head.release();
  return LLAMADART_WEBGPU_DECISION_STATUS_OK;
}

void llama_webgpu_decision_head_free(llama_webgpu_decision_head * head) {
  free_head(head);
}

const llama_webgpu_decision_head_info & llama_webgpu_decision_head_get_info(
    const llama_webgpu_decision_head * head) {
  return head->info;
}

llama_webgpu_decision_status llama_webgpu_decision_validate_sequences(
    const std::vector<llama_webgpu_decision_sequence> & sequences,
    int32_t token_limit,
    int32_t vocab_size,
    std::string * out_error) {
  std::string scratch_error;
  std::string & error = out_error != nullptr ? *out_error : scratch_error;
  for (size_t i = 0; i < sequences.size(); ++i) {
    const llama_webgpu_decision_sequence & sequence = sequences[i];
    const std::string prefix = "Decision sequence " + std::to_string(i);
    const size_t token_count = sequence.tokens.size();
    if (token_count == 0 || token_count > static_cast<size_t>(std::max(token_limit, 0))) {
      error = prefix + " has " + std::to_string(token_count) +
          " tokens; the decision encoder accepts 1 to " +
          std::to_string(token_limit) + ".";
      return LLAMADART_WEBGPU_DECISION_STATUS_INFERENCE_ERROR;
    }
    for (const int32_t token : sequence.tokens) {
      if (token < 0 || token >= vocab_size) {
        error = prefix + " contains token " + std::to_string(token) +
            ", outside the encoder vocabulary of " + std::to_string(vocab_size) +
            " tokens.";
        return LLAMADART_WEBGPU_DECISION_STATUS_INFERENCE_ERROR;
      }
    }
    if (sequence.markers.empty()) {
      error = prefix + " has no option markers.";
      return LLAMADART_WEBGPU_DECISION_STATUS_INFERENCE_ERROR;
    }
    // The head graph holds one row per marker, so the count must stay bounded
    // by the token limit or its buffers outgrow what ggml can allocate.
    if (sequence.markers.size() > token_count) {
      error = prefix + " has " + std::to_string(sequence.markers.size()) +
          " markers for its " + std::to_string(token_count) +
          " tokens; a sequence holds at most one marker per token.";
      return LLAMADART_WEBGPU_DECISION_STATUS_INFERENCE_ERROR;
    }
    for (const int32_t marker : sequence.markers) {
      if (marker < 0 || static_cast<size_t>(marker) >= token_count) {
        error = prefix + " has marker " + std::to_string(marker) +
            " outside its " + std::to_string(token_count) + " tokens.";
        return LLAMADART_WEBGPU_DECISION_STATUS_INFERENCE_ERROR;
      }
    }
    if (sequence.question_type < 0 || sequence.question_type > 2) {
      error = prefix + " has question type " +
          std::to_string(sequence.question_type) +
          "; expected 0 (choice), 1 (score) or 2 (noul).";
      return LLAMADART_WEBGPU_DECISION_STATUS_INFERENCE_ERROR;
    }
  }
  return LLAMADART_WEBGPU_DECISION_STATUS_OK;
}

llama_webgpu_decision_status llama_webgpu_decision_run(
    llama_webgpu_decision_head * head,
    const std::vector<llama_webgpu_decision_sequence> & sequences,
    std::vector<llama_webgpu_decision_output> * out_outputs,
    std::string * out_error) {
  std::string scratch_error;
  std::string & error = out_error != nullptr ? *out_error : scratch_error;
  if (head == nullptr || out_outputs == nullptr) {
    error = "The decision head is not loaded; it was freed or its model was "
        "unloaded. Load the decision head again.";
    return LLAMADART_WEBGPU_DECISION_STATUS_INVALID_STATE;
  }
  out_outputs->clear();
  // Reject every invalid sequence before the first llama_encode, whose
  // n_ubatch assertion would otherwise abort the runtime.
  const llama_webgpu_decision_status valid = llama_webgpu_decision_validate_sequences(
      sequences, head->token_limit, head->vocab_size, &error);
  if (valid != LLAMADART_WEBGPU_DECISION_STATUS_OK) {
    return valid;
  }
  if (sequences.empty()) {
    return LLAMADART_WEBGPU_DECISION_STATUS_OK;
  }

  llama_batch batch = llama_batch_init(head->token_limit, 0, 1);
  std::vector<llama_webgpu_decision_output> outputs;
  outputs.reserve(sequences.size());
  llama_webgpu_decision_status status = LLAMADART_WEBGPU_DECISION_STATUS_OK;
  for (const llama_webgpu_decision_sequence & sequence : sequences) {
    const int32_t token_count = static_cast<int32_t>(sequence.tokens.size());
    batch.n_tokens = token_count;
    for (int32_t i = 0; i < token_count; ++i) {
      batch.token[i] = sequence.tokens[static_cast<size_t>(i)];
      batch.pos[i] = i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = 1;
    }
    const int32_t encode_status = llama_encode(head->context, batch);
    if (encode_status != 0) {
      error = "The decision encoder pass failed (llama_encode returned " +
          std::to_string(encode_status) + ").";
      status = LLAMADART_WEBGPU_DECISION_STATUS_INFERENCE_ERROR;
      break;
    }
    const float * hidden = llama_get_embeddings(head->context);
    if (hidden == nullptr) {
      error = "The decision encoder returned no per-token hidden states.";
      status = LLAMADART_WEBGPU_DECISION_STATUS_INFERENCE_ERROR;
      break;
    }
    llama_webgpu_decision_output output;
    std::vector<float> cls;
    if (!compute_head_graph(
            head,
            hidden,
            token_count,
            sequence.question_type,
            sequence.markers,
            output.logits,
            cls,
            error)) {
      status = LLAMADART_WEBGPU_DECISION_STATUS_INFERENCE_ERROR;
      break;
    }
    output.act_logits = act_logits(head, cls, output.logits);
    outputs.push_back(std::move(output));
  }
  llama_batch_free(batch);
  if (status == LLAMADART_WEBGPU_DECISION_STATUS_OK) {
    *out_outputs = std::move(outputs);
  }
  return status;
}

llama_webgpu_decision_status llama_webgpu_decision_read_sequences(
    const char * path,
    std::vector<llama_webgpu_decision_sequence> * out_sequences,
    std::string * out_error) {
  std::string scratch_error;
  std::string & error = out_error != nullptr ? *out_error : scratch_error;
  const auto malformed = [&]() {
    error = "Decision input is malformed.";
    return LLAMADART_WEBGPU_DECISION_STATUS_INVALID_ARGUMENT;
  };
  if (path == nullptr || path[0] == '\0' || out_sequences == nullptr) {
    return malformed();
  }
  out_sequences->clear();
  FILE * file = std::fopen(path, "rb");
  if (file == nullptr) {
    error = std::string("Decision input could not be read: ") +
        std::strerror(errno) + ".";
    return LLAMADART_WEBGPU_DECISION_STATUS_INVALID_ARGUMENT;
  }
  std::vector<int32_t> words;
  bool read_ok = fseeko(file, 0, SEEK_END) == 0;
  const off_t size = read_ok ? ftello(file) : -1;
  read_ok = read_ok && size >= 0 && size % 4 == 0 &&
      fseeko(file, 0, SEEK_SET) == 0;
  if (read_ok) {
    words.resize(static_cast<size_t>(size / 4));
    read_ok = std::fread(words.data(), sizeof(int32_t), words.size(), file) ==
        words.size();
  }
  std::fclose(file);
  if (!read_ok || words.empty()) {
    return malformed();
  }

  const int32_t count = words[0];
  if (count < 0) {
    return malformed();
  }
  size_t cursor = 1;
  std::vector<llama_webgpu_decision_sequence> sequences;
  sequences.reserve(std::min<size_t>(static_cast<size_t>(count), words.size()));
  for (int32_t index = 0; index < count; ++index) {
    if (words.size() - cursor < 3) {
      return malformed();
    }
    const int32_t question_type = words[cursor];
    const int32_t token_count = words[cursor + 1];
    const int32_t marker_count = words[cursor + 2];
    cursor += 3;
    if (token_count < 0 || marker_count < 0 ||
        static_cast<uint64_t>(token_count) + static_cast<uint64_t>(marker_count) >
            words.size() - cursor) {
      return malformed();
    }
    llama_webgpu_decision_sequence sequence;
    sequence.question_type = question_type;
    sequence.tokens.assign(
        words.begin() + static_cast<long>(cursor),
        words.begin() + static_cast<long>(cursor + static_cast<size_t>(token_count)));
    cursor += static_cast<size_t>(token_count);
    sequence.markers.assign(
        words.begin() + static_cast<long>(cursor),
        words.begin() + static_cast<long>(cursor + static_cast<size_t>(marker_count)));
    cursor += static_cast<size_t>(marker_count);
    sequences.push_back(std::move(sequence));
  }
  if (cursor != words.size()) {
    return malformed();
  }
  *out_sequences = std::move(sequences);
  return LLAMADART_WEBGPU_DECISION_STATUS_OK;
}

llama_webgpu_decision_status llama_webgpu_decision_write_outputs(
    const char * path,
    const std::vector<llama_webgpu_decision_output> & outputs,
    std::string * out_error) {
  std::string scratch_error;
  std::string & error = out_error != nullptr ? *out_error : scratch_error;
  if (path == nullptr || path[0] == '\0') {
    error = "Decision output path is empty.";
    return LLAMADART_WEBGPU_DECISION_STATUS_INVALID_ARGUMENT;
  }
  FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    error = std::string("Decision output could not be written: ") +
        std::strerror(errno) + ".";
    return LLAMADART_WEBGPU_DECISION_STATUS_INFERENCE_ERROR;
  }
  bool ok = true;
  for (const llama_webgpu_decision_output & output : outputs) {
    const int32_t counts[2] = {
        static_cast<int32_t>(output.logits.size()),
        static_cast<int32_t>(output.act_logits.size()),
    };
    ok = ok && std::fwrite(counts, sizeof(int32_t), 2, file) == 2;
    ok = ok &&
        std::fwrite(output.logits.data(), sizeof(float), output.logits.size(), file) ==
            output.logits.size();
    ok = ok &&
        std::fwrite(
            output.act_logits.data(), sizeof(float), output.act_logits.size(), file) ==
            output.act_logits.size();
  }
  const int close_result = std::fclose(file);
  if (!ok || close_result != 0) {
    error = "Decision output could not be written.";
    return LLAMADART_WEBGPU_DECISION_STATUS_INFERENCE_ERROR;
  }
  return LLAMADART_WEBGPU_DECISION_STATUS_OK;
}
