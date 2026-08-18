#ifndef LLAMADART_WEBGPU_TTS_H
#define LLAMADART_WEBGPU_TTS_H

#include <cstddef>
#include <cstdint>

#include "llama.h"

struct mtmd_context;
struct llama_webgpu_tts;

constexpr uint32_t LLAMADART_WEBGPU_TTS_API_VERSION = 1;

enum llama_webgpu_tts_status {
  LLAMADART_WEBGPU_TTS_STATUS_OK = 0,
  LLAMADART_WEBGPU_TTS_STATUS_INVALID_ARGUMENT = -1,
  LLAMADART_WEBGPU_TTS_STATUS_UNSUPPORTED = -2,
  LLAMADART_WEBGPU_TTS_STATUS_INVALID_STATE = -3,
  LLAMADART_WEBGPU_TTS_STATUS_SPEAKER_DECODE_FAILED = -4,
  LLAMADART_WEBGPU_TTS_STATUS_UPSTREAM_ERROR = -5,
  LLAMADART_WEBGPU_TTS_STATUS_CANCELLED = -6,
};

enum llama_webgpu_tts_model_type {
  LLAMADART_WEBGPU_TTS_MODEL_TYPE_NONE = 0,
  LLAMADART_WEBGPU_TTS_MODEL_TYPE_QWEN3 = 1,
  LLAMADART_WEBGPU_TTS_MODEL_TYPE_UNKNOWN = 255,
};

enum llama_webgpu_tts_capability {
  LLAMADART_WEBGPU_TTS_CAPABILITY_LANGUAGE = 1u << 0,
  LLAMADART_WEBGPU_TTS_CAPABILITY_SPEAKER_REFERENCE = 1u << 1,
};

enum llama_webgpu_tts_state {
  LLAMADART_WEBGPU_TTS_STATE_IDLE = 0,
  LLAMADART_WEBGPU_TTS_STATE_PROCESSING_PROMPT = 1,
  LLAMADART_WEBGPU_TTS_STATE_GENERATING = 2,
  LLAMADART_WEBGPU_TTS_STATE_COMPLETED = 3,
  LLAMADART_WEBGPU_TTS_STATE_CANCELLED = 4,
  LLAMADART_WEBGPU_TTS_STATE_FAILED = 5,
};

struct llama_webgpu_tts_info {
  uint32_t api_version;
  int32_t model_type;
  uint32_t capabilities;
  int32_t sample_rate;
  int32_t channels;
};

struct llama_webgpu_tts_request {
  const char * text;
  size_t text_length;
  const unsigned char * speaker_audio;
  size_t speaker_audio_length;
  const char * language;
  llama_seq_id sequence_id;
  int32_t prompt_batch_size;
  int32_t max_frames;
  int32_t top_k;
  float top_p;
  float min_p;
  float temperature;
  uint32_t seed;
};

struct llama_webgpu_tts_progress {
  int32_t state;
  int32_t prompt_tokens_remaining;
  int32_t frames_generated;
  bool truncated;
};

llama_webgpu_tts_info llama_webgpu_tts_get_info(const mtmd_context * mtmd);

llama_webgpu_tts * llama_webgpu_tts_init(
    llama_context * llama,
    mtmd_context * mtmd,
    llama_webgpu_tts_status * out_status);

void llama_webgpu_tts_free(llama_webgpu_tts * tts);

llama_webgpu_tts_status llama_webgpu_tts_start(
    llama_webgpu_tts * tts,
    const llama_webgpu_tts_request & request);

llama_webgpu_tts_status llama_webgpu_tts_step(
    llama_webgpu_tts * tts,
    llama_webgpu_tts_progress * out_progress);

void llama_webgpu_tts_cancel(llama_webgpu_tts * tts);

llama_webgpu_tts_status llama_webgpu_tts_reset(llama_webgpu_tts * tts);

llama_webgpu_tts_status llama_webgpu_tts_write_pcm(
    const llama_webgpu_tts * tts,
    const char * output_path);

int32_t llama_webgpu_tts_sample_rate(const llama_webgpu_tts * tts);
int64_t llama_webgpu_tts_sample_count(const llama_webgpu_tts * tts);
const char * llama_webgpu_tts_last_error(const llama_webgpu_tts * tts);

#endif
