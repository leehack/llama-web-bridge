// Runtime state shared by the core's helpers and exports.
// Part of llama_webgpu_core.cpp, included inside its anonymous namespace;
// not a standalone translation unit.

struct runtime_state {
  llama_model * model = nullptr;
  llama_context * ctx = nullptr;
  const llama_vocab * vocab = nullptr;
  mtmd_context * mm_ctx = nullptr;
  uint32_t n_ctx = 0;
};

runtime_state g_state;

struct pending_media {
  mtmd_bitmap * bitmap = nullptr;
  mtmd_helper_video * video_ctx = nullptr;
};

bool g_backend_initialized = false;
bool g_has_webgpu = false;
bool g_generation_active = false;
bool g_cancel_requested = false;
llama_sampler * g_active_sampler = nullptr;
// Kept out of g_active_sampler's chain so sample_next_token() can apply it
// before or after truncation, as llama.cpp's common_sampler does.
llama_sampler * g_active_grammar = nullptr;
std::vector<llama_token_data> g_sampler_candidates;
std::atomic<int32_t> g_log_level{3};
std::atomic<int32_t> g_last_non_cont_level{GGML_LOG_LEVEL_NONE};

std::string g_last_error;
std::string g_last_output;
std::string g_last_piece;
std::string g_last_tokens_json = "[]";
std::string g_last_detokenized;
std::string g_last_embedding_json = "[]";
std::string g_last_next_token_scores_json = "{}";
std::string g_backend_json = "[]";
std::string g_model_meta_json = "{}";
bool g_model_uses_gpu_ops = false;
std::vector<llama_token> g_cached_prompt_tokens;
// Token counts of the latest generation, reset when it begins.
int32_t g_usage_prompt_tokens = 0;
int32_t g_usage_cached_prompt_tokens = 0;
int32_t g_usage_completion_tokens = 0;
std::string g_generation_usage_json = "{}";
bool g_model_is_qwen3_asr = false;
bool g_generation_has_qwen3_asr_audio = false;
bool g_qwen3_asr_prefix_seed_attempted = false;
std::vector<llama_token> g_qwen3_asr_prefix_tokens;
size_t g_qwen3_asr_prefix_token_index = 0;
llama_webgpu_tts * g_tts = nullptr;
bool g_tts_active = false;
std::string g_tts_info_json = "{}";
std::string g_tts_progress_json = "{}";
// Decision heads own private encoder contexts on g_state.model. Handles are
// never reused within one runtime, so a stale handle cannot reach a newer head.
std::map<int32_t, llama_webgpu_decision_head *> g_decision_heads;
int32_t g_next_decision_handle = 1;
std::string g_decision_json = "{}";
// LoRA adapters loaded on g_state.model. Handles are never reused within one
// runtime, so a stale handle cannot reach a newer adapter.
std::map<int32_t, llama_adapter_lora *> g_lora_adapters;
int32_t g_next_lora_handle = 1;
// The adapters applied to g_state.ctx and their scales, in the order first set.
std::vector<std::pair<int32_t, float>> g_active_loras;

std::vector<pending_media> g_pending_media;
