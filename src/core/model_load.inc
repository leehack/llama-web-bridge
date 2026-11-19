// Model loading.
// Part of llama_webgpu_core.cpp, included inside its anonymous namespace;
// not a standalone translation unit.

bool is_supported_kv_cache_type(int32_t value) {
  switch (value) {
    case GGML_TYPE_F16:
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q8_0:
      return true;
    default:
      return false;
  }
}

constexpr llama_load_mode resolve_load_mode(bool use_mmap, bool use_mlock) {
  if (use_mlock) {
    return LLAMA_LOAD_MODE_MLOCK;
  }
  return use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
}

static_assert(resolve_load_mode(false, false) == LLAMA_LOAD_MODE_NONE);
static_assert(resolve_load_mode(true, false) == LLAMA_LOAD_MODE_MMAP);
static_assert(resolve_load_mode(false, true) == LLAMA_LOAD_MODE_MLOCK);
static_assert(resolve_load_mode(true, true) == LLAMA_LOAD_MODE_MLOCK);

#if SIZE_MAX <= 0xffffffffULL
class BufferedFileReader {
  std::FILE * fp_ = nullptr;
  std::vector<uint8_t> buf_;
  size_t pos_ = 0;
  size_t end_ = 0;

 public:
  explicit BufferedFileReader(std::FILE * fp, size_t capacity = 65536)
      : fp_(fp), buf_(capacity) {}

  bool read_bytes(void * dst, size_t len) {
    auto * out = static_cast<uint8_t *>(dst);
    while (len > 0) {
      if (pos_ < end_) {
        const size_t avail = end_ - pos_;
        const size_t take = std::min(len, avail);
        std::memcpy(out, buf_.data() + pos_, take);
        pos_ += take;
        out += take;
        len -= take;
        continue;
      }
      if (len >= buf_.size()) {
        return std::fread(out, 1, len, fp_) == len;
      }
      pos_ = 0;
      end_ = std::fread(buf_.data(), 1, buf_.size(), fp_);
      if (end_ == 0) {
        return false;
      }
    }
    return true;
  }

  bool skip_bytes(uint64_t count) {
    while (count > 0) {
      const size_t avail = end_ - pos_;
      if (count <= avail) {
        pos_ += static_cast<size_t>(count);
        return true;
      }
      count -= avail;
      pos_ = 0;
      end_ = 0;
      if (count < buf_.size()) {
        // Refill the buffer once rather than issuing an fseeko across block boundaries.
        end_ = std::fread(buf_.data(), 1, buf_.size(), fp_);
        if (end_ == 0) {
          return false;
        }
        continue;
      }
      if (count <= static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
        return fseeko(fp_, static_cast<off_t>(count), SEEK_CUR) == 0;
      }
      return false;
    }
    return true;
  }
};

size_t gguf_scalar_byte_size(uint32_t gguf_type) {
  switch (gguf_type) {
    case GGUF_TYPE_UINT8:
    case GGUF_TYPE_INT8:
    case GGUF_TYPE_BOOL:
      return 1;
    case GGUF_TYPE_UINT16:
    case GGUF_TYPE_INT16:
      return 2;
    case GGUF_TYPE_UINT32:
    case GGUF_TYPE_INT32:
    case GGUF_TYPE_FLOAT32:
      return 4;
    case GGUF_TYPE_UINT64:
    case GGUF_TYPE_INT64:
    case GGUF_TYPE_FLOAT64:
      return 8;
    default:
      return 0;
  }
}

bool read_or_skip_gguf_value(
    BufferedFileReader & reader,
    uint32_t gguf_type,
    uint64_t * out_uint) {
  if (out_uint != nullptr) {
    if (gguf_type == GGUF_TYPE_UINT16) {
      uint16_t v = 0;
      const bool ok = reader.read_bytes(&v, sizeof(v));
      *out_uint = v;
      return ok;
    }
    if (gguf_type == GGUF_TYPE_UINT32) {
      uint32_t v = 0;
      const bool ok = reader.read_bytes(&v, sizeof(v));
      *out_uint = v;
      return ok;
    }
    if (gguf_type == GGUF_TYPE_UINT64) {
      uint64_t v = 0;
      const bool ok = reader.read_bytes(&v, sizeof(v));
      *out_uint = v;
      return ok;
    }
  }
  if (gguf_type == GGUF_TYPE_STRING) {
    uint64_t len = 0;
    if (!reader.read_bytes(&len, sizeof(len))) {
      return false;
    }
    return reader.skip_bytes(len);
  }
  if (gguf_type == GGUF_TYPE_ARRAY) {
    uint32_t elem_type = 0;
    uint64_t count = 0;
    if (!reader.read_bytes(&elem_type, sizeof(elem_type)) ||
        !reader.read_bytes(&count, sizeof(count))) {
      return false;
    }
    if (elem_type == GGUF_TYPE_STRING) {
      if (count > (uint64_t(1) << 24)) {
        return false;
      }
      for (uint64_t i = 0; i < count; ++i) {
        if (!read_or_skip_gguf_value(reader, GGUF_TYPE_STRING, nullptr)) {
          return false;
        }
      }
      return true;
    }
    const size_t elem_size = gguf_scalar_byte_size(elem_type);
    if (elem_size == 0 || count > std::numeric_limits<uint64_t>::max() / elem_size) {
      return false;
    }
    return reader.skip_bytes(count * elem_size);
  }
  const size_t scalar_size = gguf_scalar_byte_size(gguf_type);
  if (scalar_size == 0) {
    return false;
  }
  return reader.skip_bytes(scalar_size);
}

bool check_wasm32_gguf_shard(
    const char * shard_path,
    uint64_t * total_accumulated,
    uint16_t * out_split_count,
    uint16_t * out_split_no) {
  if (shard_path == nullptr || std::strlen(shard_path) == 0) {
    return true;
  }
  std::unique_ptr<std::FILE, decltype(&std::fclose)> fp(
      std::fopen(shard_path, "rb"), &std::fclose);
  if (!fp) {
    return true;
  }
  BufferedFileReader reader(fp.get());

  char magic[4] = {};
  uint32_t version = 0;
  uint64_t n_tensors = 0;
  uint64_t n_kv = 0;
  if (!reader.read_bytes(magic, sizeof(magic)) ||
      std::memcmp(magic, "GGUF", 4) != 0 ||
      !reader.read_bytes(&version, sizeof(version)) ||
      (version != 2 && version != 3) ||
      !reader.read_bytes(&n_tensors, sizeof(n_tensors)) ||
      !reader.read_bytes(&n_kv, sizeof(n_kv))) {
    return true;
  }
  if (n_tensors > (uint64_t(1) << 24) || n_kv > (uint64_t(1) << 24)) {
    return true;
  }

  uint64_t alignment = GGUF_DEFAULT_ALIGNMENT;
  constexpr const char kAlignmentKey[] = "general.alignment";
  constexpr const char kSplitCountKey[] = "split.count";
  constexpr const char kSplitNoKey[] = "split.no";
  constexpr uint64_t kMaxMatchedKeyLen = 17;

  for (uint64_t i = 0; i < n_kv; ++i) {
    uint64_t key_len = 0;
    if (!reader.read_bytes(&key_len, sizeof(key_len))) {
      return true;
    }
    enum KvMatch { kNone, kAlignment, kSplitCount, kSplitNo };
    KvMatch match = kNone;
    if (key_len <= kMaxMatchedKeyLen) {
      char key_buf[kMaxMatchedKeyLen] = {};
      if (!reader.read_bytes(key_buf, static_cast<size_t>(key_len))) {
        return true;
      }
      if (key_len == sizeof(kAlignmentKey) - 1 &&
          std::memcmp(key_buf, kAlignmentKey, key_len) == 0) {
        match = kAlignment;
      } else if (out_split_count != nullptr &&
                 key_len == sizeof(kSplitCountKey) - 1 &&
                 std::memcmp(key_buf, kSplitCountKey, key_len) == 0) {
        match = kSplitCount;
      } else if (out_split_no != nullptr &&
                 key_len == sizeof(kSplitNoKey) - 1 &&
                 std::memcmp(key_buf, kSplitNoKey, key_len) == 0) {
        match = kSplitNo;
      }
    } else if (!reader.skip_bytes(key_len)) {
      return true;
    }

    uint32_t kv_type = 0;
    if (!reader.read_bytes(&kv_type, sizeof(kv_type))) {
      return true;
    }
    uint64_t matched_val = 0;
    if (!read_or_skip_gguf_value(
            reader, kv_type, match != kNone ? &matched_val : nullptr)) {
      return true;
    }
    if (match == kAlignment && kv_type == GGUF_TYPE_UINT32 &&
        matched_val > 0 && (matched_val & (matched_val - 1)) == 0) {
      alignment = matched_val;
    } else if (match == kSplitCount && matched_val > 1 &&
               matched_val <= std::numeric_limits<uint16_t>::max()) {
      *out_split_count = static_cast<uint16_t>(matched_val);
    } else if (match == kSplitNo &&
               matched_val <= std::numeric_limits<uint16_t>::max()) {
      *out_split_no = static_cast<uint16_t>(matched_val);
    }
  }

  constexpr uint64_t kWasm32SizeMax = static_cast<uint64_t>(SIZE_MAX);
  uint64_t shard_accumulated = 0;

  for (uint64_t i = 0; i < n_tensors; ++i) {
    uint64_t name_len = 0;
    if (!reader.read_bytes(&name_len, sizeof(name_len))) {
      return true;
    }
    char name_buf[65] = {};
    const size_t capture = static_cast<size_t>(std::min<uint64_t>(name_len, 64));
    if (capture > 0 && !reader.read_bytes(name_buf, capture)) {
      return true;
    }
    name_buf[capture] = '\0';
    if (name_len > capture && !reader.skip_bytes(name_len - capture)) {
      return true;
    }

    uint32_t n_dims = 0;
    if (!reader.read_bytes(&n_dims, sizeof(n_dims)) || n_dims > GGML_MAX_DIMS) {
      return true;
    }
    uint64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (uint32_t d = 0; d < n_dims; ++d) {
      int64_t dim = 0;
      if (!reader.read_bytes(&dim, sizeof(dim)) || dim < 0) {
        return true;
      }
      ne[d] = static_cast<uint64_t>(dim);
    }

    int32_t tensor_type = -1;
    uint64_t tensor_offset = 0;
    if (!reader.read_bytes(&tensor_type, sizeof(tensor_type)) ||
        !reader.read_bytes(&tensor_offset, sizeof(tensor_offset)) ||
        tensor_type < 0 || tensor_type >= GGML_TYPE_COUNT) {
      return true;
    }

    const ggml_type gtype = static_cast<ggml_type>(tensor_type);
    const uint64_t type_size = static_cast<uint64_t>(ggml_type_size(gtype));
    const int64_t blck_raw = ggml_blck_size(gtype);
    if (blck_raw <= 0) {
      return true;
    }
    const uint64_t blck_size = static_cast<uint64_t>(blck_raw);
    if (ne[0] % blck_size != 0) {
      return true;
    }

    uint64_t nbytes = ne[0] / blck_size;
    if (type_size > 0 && nbytes > std::numeric_limits<uint64_t>::max() / type_size) {
      nbytes = std::numeric_limits<uint64_t>::max();
    } else {
      nbytes *= type_size;
      for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (ne[d] > 0 && nbytes > std::numeric_limits<uint64_t>::max() / ne[d]) {
          nbytes = std::numeric_limits<uint64_t>::max();
          break;
        }
        nbytes *= ne[d];
      }
    }

    const bool nbytes_overflows = nbytes > kWasm32SizeMax - (alignment - 1);
    const uint64_t padded_size = nbytes_overflows
        ? kWasm32SizeMax
        : ((nbytes + alignment - 1) / alignment) * alignment;
    const uint64_t prior_total = *total_accumulated;
    if (nbytes_overflows || tensor_offset > kWasm32SizeMax ||
        kWasm32SizeMax - shard_accumulated < padded_size ||
        kWasm32SizeMax - prior_total < padded_size) {
      set_error(
          "GGUF tensor payload exceeds the wasm32 4 GiB address space limit (tensor '" +
          std::string(name_buf) + "', accumulated " + std::to_string(prior_total) +
          " + " + std::to_string(padded_size) + " > " +
          std::to_string(kWasm32SizeMax) +
          " bytes); use the memory64 core or a smaller quantization");
      return false;
    }
    shard_accumulated += padded_size;
    *total_accumulated += padded_size;
  }

  return true;
}

// Fail fast with a clean status before gguf_init_from_reader logs overflow and
// unwinds across un-catchable llama.cpp C++ frames into a raw wasm abort().
bool check_wasm32_gguf_tensor_size_limit(const char * model_path) {
  if (model_path == nullptr || std::strlen(model_path) == 0) {
    return true;
  }
  uint64_t total_accumulated = 0;
  uint16_t split_count = 1;
  uint16_t split_no = 0;
  if (!check_wasm32_gguf_shard(
          model_path, &total_accumulated, &split_count, &split_no)) {
    return false;
  }

  if (split_count > 1) {
    const size_t path_buf_len = std::strlen(model_path) + 64;
    std::vector<char> prefix_buf(path_buf_len, 0);
    const int32_t prefix_len = llama_split_prefix(
        prefix_buf.data(),
        prefix_buf.size(),
        model_path,
        static_cast<int32_t>(split_no),
        static_cast<int32_t>(split_count));
    if (prefix_len > 0) {
      std::vector<char> shard_buf(static_cast<size_t>(prefix_len) + 64, 0);
      for (uint16_t s = 0; s < split_count; ++s) {
        if (s == split_no) {
          continue;
        }
        const int32_t written = llama_split_path(
            shard_buf.data(),
            shard_buf.size(),
            prefix_buf.data(),
            static_cast<int32_t>(s),
            static_cast<int32_t>(split_count));
        if (written > 0 &&
            !check_wasm32_gguf_shard(
                shard_buf.data(), &total_accumulated, nullptr, nullptr)) {
          return false;
        }
      }
    }
  }

  return true;
}
#endif

int32_t load_model_internal(
    const char * model_path,
    int32_t n_ctx,
    int32_t n_threads,
    int32_t n_threads_batch,
    int32_t n_batch,
    int32_t n_ubatch,
    int32_t n_gpu_layers,
    int32_t n_seq_max,
    bool use_mmap,
    bool use_mlock,
    int32_t flash_attn_type,
    int32_t type_k,
    int32_t type_v,
    int32_t kv_unified,
    double rope_freq_base,
    double rope_freq_scale,
    int32_t split_mode,
    int32_t main_gpu,
    bool load_mtp,
    int32_t n_rs_seq) {
  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = n_gpu_layers;
  mparams.load_mode = resolve_load_mode(use_mmap, use_mlock);
  mparams.vocab_only = false;
  mparams.load_mtp = load_mtp;
  if (split_mode >= LLAMA_SPLIT_MODE_NONE &&
      split_mode <= LLAMA_SPLIT_MODE_TENSOR) {
    mparams.split_mode = static_cast<llama_split_mode>(split_mode);
  }
  if (main_gpu >= 0) {
    mparams.main_gpu = main_gpu;
  }

  // With no layer offloaded, keep WebGPU out of the model and context backends:
  // ggml-webgpu waits on queue work at every sync, and Firefox resolves those
  // waits on a 100 ms tick. Static because draft models reuse g_model_params.
  static ggml_backend_dev_t cpu_only_devices[] = {nullptr};
  if (n_gpu_layers == 0) {
    mparams.devices = cpu_only_devices;
    // Tensor split needs at least one device, and the empty list has none.
    if (mparams.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
      mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
    }
  }

#if SIZE_MAX <= 0xffffffffULL
  if (llamadart_webgpu_is_js_model_path(model_path) &&
      !check_wasm32_gguf_tensor_size_limit(model_path)) {
    return -4;
  }
#endif

  g_state.model = llama_model_load_from_file(model_path, mparams);
  if (g_state.model == nullptr) {
    set_error("llama_model_load_from_file failed");
    return -2;
  }

  llama_context_params cparams = llama_context_default_params();
  if (n_ctx > 0) {
    cparams.n_ctx = static_cast<uint32_t>(n_ctx);
  }

  if (n_seq_max > 0) {
    cparams.n_seq_max = static_cast<uint32_t>(n_seq_max);
  }

  if (n_rs_seq > 0) {
    cparams.n_rs_seq = static_cast<uint32_t>(n_rs_seq);
  }

  if (n_threads > 0) {
    cparams.n_threads = n_threads;
  }

  if (n_threads_batch > 0) {
    cparams.n_threads_batch = n_threads_batch;
  } else if (n_threads > 0) {
    cparams.n_threads_batch = n_threads;
  }

  if (n_batch > 0) {
    cparams.n_batch = static_cast<uint32_t>(n_batch);
  }

  if (n_ubatch > 0) {
    cparams.n_ubatch = static_cast<uint32_t>(n_ubatch);
  }

  if (cparams.n_batch == 0 || cparams.n_batch > cparams.n_ctx) {
    cparams.n_batch = std::min<uint32_t>(cparams.n_ctx, 1024U);
  }

  if (cparams.n_ubatch == 0 || cparams.n_ubatch > cparams.n_batch) {
    cparams.n_ubatch = std::min<uint32_t>(cparams.n_batch, 512U);
  }

  if (flash_attn_type >= LLAMA_FLASH_ATTN_TYPE_AUTO &&
      flash_attn_type <= LLAMA_FLASH_ATTN_TYPE_ENABLED) {
    cparams.flash_attn_type =
        static_cast<llama_flash_attn_type>(flash_attn_type);
  }
  if (is_supported_kv_cache_type(type_k)) {
    cparams.type_k = static_cast<ggml_type>(type_k);
  }
  if (is_supported_kv_cache_type(type_v)) {
    cparams.type_v = static_cast<ggml_type>(type_v);
  }
  if (kv_unified >= 0) {
    cparams.kv_unified = kv_unified != 0;
  }
  if (rope_freq_base > 0.0) {
    cparams.rope_freq_base = static_cast<float>(rope_freq_base);
  }
  if (rope_freq_scale > 0.0) {
    cparams.rope_freq_scale = static_cast<float>(rope_freq_scale);
  }

  const bool enable_gpu_ops = n_gpu_layers > 0;
  g_model_uses_gpu_ops = enable_gpu_ops;
  cparams.offload_kqv = enable_gpu_ops;
  cparams.op_offload = enable_gpu_ops;
  cparams.no_perf = true;

  g_state.ctx = llama_init_from_model(g_state.model, cparams);
  if (g_state.ctx == nullptr) {
    set_error("llama_init_from_model failed");
    free_runtime();
    return -3;
  }

  g_state.vocab = llama_model_get_vocab(g_state.model);
  g_state.n_ctx = llama_n_ctx(g_state.ctx);
  g_model_params = mparams;
  g_context_params = cparams;
  g_model_loaded_mtp = load_mtp;
  llama_set_abort_callback(g_state.ctx, should_abort_callback, nullptr);

  rebuild_model_metadata_json();
  return 0;
}
