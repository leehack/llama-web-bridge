// Backend probing and model metadata.
// Part of llama_webgpu_core.cpp, included inside its anonymous namespace;
// not a standalone translation unit.

std::vector<std::string> collect_backend_labels() {
  std::vector<std::string> labels;

  const size_t count = ggml_backend_dev_count();
  labels.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    if (dev == nullptr) {
      continue;
    }

    const char * dev_name = ggml_backend_dev_name(dev);
    if (dev_name == nullptr) {
      continue;
    }

    std::string label = dev_name;

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg != nullptr) {
      const char * reg_name = ggml_backend_reg_name(reg);
      if (reg_name != nullptr && std::strlen(reg_name) > 0) {
        const std::string reg_str = reg_name;
        const std::string dev_str = dev_name;
        if (to_lower(reg_str) == to_lower(dev_str)) {
          label = reg_str;
        } else {
          label = reg_str + " (" + dev_str + ")";
        }
      }
    }

    labels.push_back(label);
  }

  return labels;
}

void refresh_backend_probe() {
  clear_error();
  ensure_backend_initialized();

  ggml_backend_load_all();

  const std::vector<std::string> labels = collect_backend_labels();

  std::string json = "[";
  for (size_t i = 0; i < labels.size(); ++i) {
    if (i > 0) {
      json += ",";
    }
    json += '"';
    json += escape_json(labels[i]);
    json += '"';
  }
  json += "]";
  g_backend_json = json;

  g_has_webgpu = false;
  for (const std::string & label : labels) {
    const std::string lowered = to_lower(label);
    if (lowered.find("webgpu") != std::string::npos ||
        lowered.find("wgpu") != std::string::npos) {
      g_has_webgpu = true;
      break;
    }
  }
}

std::string read_model_meta_string(
    const llama_model * model,
    const int32_t index,
    const bool read_key) {
  size_t buf_size = read_key ? 1024 : 65536;

  for (int attempt = 0; attempt < 6; ++attempt) {
    std::vector<char> buf(buf_size, '\0');
    const int32_t rc = read_key
        ? llama_model_meta_key_by_index(model, index, buf.data(), buf.size())
        : llama_model_meta_val_str_by_index(model, index, buf.data(), buf.size());

    if (rc < 0) {
      buf_size *= 2;
      continue;
    }

    if (static_cast<size_t>(rc) >= buf_size) {
      buf_size = static_cast<size_t>(rc) + 1;
      continue;
    }

    return std::string(buf.data());
  }

  return "";
}

void rebuild_model_metadata_json() {
  if (g_state.model == nullptr) {
    g_model_meta_json = "{}";
    return;
  }

  const int32_t count = llama_model_meta_count(g_state.model);
  if (count <= 0) {
    g_model_meta_json = "{}";
    return;
  }

  std::string json = "{";
  bool wrote_any = false;

  for (int32_t i = 0; i < count; ++i) {
    const std::string key = read_model_meta_string(g_state.model, i, true);
    const std::string value = read_model_meta_string(g_state.model, i, false);

    if (key.empty()) {
      continue;
    }

    if (wrote_any) {
      json += ",";
    }
    wrote_any = true;

    json += '"';
    json += escape_json(key);
    json += "\":";
    json += '"';
    json += escape_json(value);
    json += '"';
  }

  json += "}";
  g_model_meta_json = json;

  const std::string normalized = to_lower(json);
  // llama.cpp's string metadata API omits the general.tags array. Combine the
  // Qwen3-VL backbone with Qwen3-ASR's published sampling temperature here,
  // then require an audio-only projector when generation begins.
  g_model_is_qwen3_asr =
      normalized.find("\"general.architecture\":\"qwen3vl\"") !=
          std::string::npos &&
      normalized.find("\"general.sampling.temp\":\"0.000001\"") !=
          std::string::npos;
}
