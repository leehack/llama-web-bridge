#include "llama_webgpu_tts.h"
#include "llama_webgpu_mtmd_compat.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "mtmd-helper.h"
#include "mtmd.h"

struct llama_webgpu_tts {
  llama_context * llama = nullptr;
  mtmd_context * mtmd = nullptr;
  mtmd_helper_gen_audio * generator = nullptr;
  llama_sampler * sampler = nullptr;
  mtmd_bitmap * speaker = nullptr;
  std::atomic<bool> cancel_requested{false};
  llama_webgpu_tts_state state = LLAMADART_WEBGPU_TTS_STATE_IDLE;
  llama_seq_id sequence_id = 0;
  bool owns_sequence = false;
  bool owns_embedding_mode = false;
  int32_t prompt_batch_size = 512;
  int32_t max_frames = 512;
  int32_t prompt_tokens_remaining = 0;
  int32_t frames_generated = 0;
  bool truncated = false;
  int32_t sample_rate = 0;
  int64_t sample_count = 0;
  std::vector<float> pcm;
  std::string language;
  std::string error;
};

namespace {

void release_task_resources(llama_webgpu_tts * tts) {
  if (tts == nullptr) {
    return;
  }
  if (tts->speaker != nullptr) {
    mtmd_bitmap_free(tts->speaker);
    tts->speaker = nullptr;
  }
  if (tts->sampler != nullptr) {
    llama_sampler_free(tts->sampler);
    tts->sampler = nullptr;
  }
  if (tts->generator != nullptr) {
    mtmd_helper_gen_audio_reset(tts->generator);
  }
  if (tts->owns_embedding_mode && tts->llama != nullptr) {
    llama_set_embeddings(tts->llama, false);
    tts->owns_embedding_mode = false;
  }
  if (tts->owns_sequence && tts->llama != nullptr) {
    llama_memory_seq_rm(
        llama_get_memory(tts->llama), tts->sequence_id, 0, -1);
    tts->owns_sequence = false;
  }
}

llama_webgpu_tts_status fail(
    llama_webgpu_tts * tts,
    llama_webgpu_tts_status status,
    const char * message) {
  if (tts != nullptr) {
    tts->state = LLAMADART_WEBGPU_TTS_STATE_FAILED;
    tts->error = message != nullptr ? message : "unknown TTS error";
    release_task_resources(tts);
  }
  return status;
}

llama_webgpu_tts_status error(
    llama_webgpu_tts * tts,
    llama_webgpu_tts_status status,
    const char * message) {
  if (tts != nullptr) {
    tts->error = message != nullptr ? message : "unknown TTS error";
  }
  return status;
}

llama_webgpu_tts_model_type model_type_from_upstream(
    mtmd_gen_audio_type type) {
  switch (type) {
    case MTMD_GEN_AUDIO_TYPE_NONE:
      return LLAMADART_WEBGPU_TTS_MODEL_TYPE_NONE;
    case MTMD_GEN_AUDIO_TYPE_QWEN3TTS:
      return LLAMADART_WEBGPU_TTS_MODEL_TYPE_QWEN3;
    default:
      return LLAMADART_WEBGPU_TTS_MODEL_TYPE_UNKNOWN;
  }
}

uint32_t capabilities_for(
    const mtmd_context * mtmd,
    mtmd_gen_audio_type type) {
  if (type != MTMD_GEN_AUDIO_TYPE_QWEN3TTS) {
    return 0;
  }
  return LLAMADART_WEBGPU_TTS_CAPABILITY_LANGUAGE |
      (mtmd_support_audio(mtmd)
           ? LLAMADART_WEBGPU_TTS_CAPABILITY_SPEAKER_REFERENCE
           : 0u);
}

llama_sampler * init_sampler(const llama_webgpu_tts_request & request) {
  llama_sampler * sampler =
      llama_sampler_chain_init(llama_sampler_chain_default_params());
  if (sampler == nullptr) {
    return nullptr;
  }
  llama_sampler_chain_add(sampler, llama_sampler_init_top_k(request.top_k));
  llama_sampler_chain_add(sampler, llama_sampler_init_top_p(request.top_p, 1));
  llama_sampler_chain_add(sampler, llama_sampler_init_min_p(request.min_p, 1));
  llama_sampler_chain_add(sampler, llama_sampler_init_temp(request.temperature));
  llama_sampler_chain_add(sampler, llama_sampler_init_dist(request.seed));
  return sampler;
}

llama_webgpu_tts_status finish_output(llama_webgpu_tts * tts) {
  int32_t sample_rate = 0;
  const char * data = nullptr;
  size_t data_len = 0;
  int64_t sample_count = 0;
  if (mtmd_helper_gen_audio_get_output(
          tts->generator,
          &sample_rate,
          &data,
          &data_len,
          &sample_count) != 0) {
    return fail(
        tts,
        LLAMADART_WEBGPU_TTS_STATUS_UPSTREAM_ERROR,
        "audio output conversion failed");
  }

  const bool sample_count_overflows =
      sample_count > 0 &&
      static_cast<uint64_t>(sample_count) >
          std::numeric_limits<size_t>::max() / sizeof(float);
  if (sample_rate <= 0 || sample_count <= 0 || sample_count_overflows ||
      data_len != static_cast<size_t>(sample_count) * sizeof(float) ||
      (data_len > 0 && data == nullptr)) {
    return fail(
        tts,
        LLAMADART_WEBGPU_TTS_STATUS_UPSTREAM_ERROR,
        "audio output metadata is invalid");
  }

  tts->pcm.resize(static_cast<size_t>(sample_count));
  std::memcpy(tts->pcm.data(), data, data_len);
  tts->sample_rate = sample_rate;
  tts->sample_count = sample_count;
  tts->state = LLAMADART_WEBGPU_TTS_STATE_COMPLETED;
  release_task_resources(tts);
  return LLAMADART_WEBGPU_TTS_STATUS_OK;
}

void write_progress(
    const llama_webgpu_tts * tts,
    llama_webgpu_tts_progress * out_progress) {
  out_progress->state = tts->state;
  out_progress->prompt_tokens_remaining = tts->prompt_tokens_remaining;
  out_progress->frames_generated = tts->frames_generated;
  out_progress->truncated = tts->truncated;
}

}  // namespace

llama_webgpu_tts_info llama_webgpu_tts_get_info(const mtmd_context * mtmd) {
  llama_webgpu_tts_info info{
      LLAMADART_WEBGPU_TTS_API_VERSION,
      LLAMADART_WEBGPU_TTS_MODEL_TYPE_NONE,
      0,
      0,
      0,
  };
  if (mtmd == nullptr) {
    return info;
  }

  const mtmd_gen_audio_info upstream = mtmd_gen_audio_get_info(mtmd);
  info.model_type = model_type_from_upstream(upstream.type);
  info.capabilities = capabilities_for(mtmd, upstream.type);
  if (upstream.type == MTMD_GEN_AUDIO_TYPE_QWEN3TTS) {
    info.sample_rate = upstream.sample_rate;
    info.channels = 1;
  }
  return info;
}

llama_webgpu_tts * llama_webgpu_tts_init(
    llama_context * llama,
    mtmd_context * mtmd,
    llama_webgpu_tts_status * out_status) {
  if (out_status != nullptr) {
    *out_status = LLAMADART_WEBGPU_TTS_STATUS_INVALID_ARGUMENT;
  }
  const auto info = llama_webgpu_tts_get_info(mtmd);
  if (llama == nullptr || mtmd == nullptr ||
      info.model_type != LLAMADART_WEBGPU_TTS_MODEL_TYPE_QWEN3) {
    if (out_status != nullptr && llama != nullptr && mtmd != nullptr) {
      *out_status = LLAMADART_WEBGPU_TTS_STATUS_UNSUPPORTED;
    }
    return nullptr;
  }

  mtmd_helper_gen_audio * generator =
      mtmd_helper_gen_audio_init(llama, mtmd);
  if (generator == nullptr) {
    if (out_status != nullptr) {
      *out_status = LLAMADART_WEBGPU_TTS_STATUS_UPSTREAM_ERROR;
    }
    return nullptr;
  }

  auto * tts = new llama_webgpu_tts();
  tts->llama = llama;
  tts->mtmd = mtmd;
  tts->generator = generator;
  if (out_status != nullptr) {
    *out_status = LLAMADART_WEBGPU_TTS_STATUS_OK;
  }
  return tts;
}

void llama_webgpu_tts_free(llama_webgpu_tts * tts) {
  if (tts == nullptr) {
    return;
  }
  release_task_resources(tts);
  mtmd_helper_gen_audio_free(tts->generator);
  delete tts;
}

llama_webgpu_tts_status llama_webgpu_tts_start(
    llama_webgpu_tts * tts,
    const llama_webgpu_tts_request & request) {
  if (tts == nullptr) {
    return LLAMADART_WEBGPU_TTS_STATUS_INVALID_ARGUMENT;
  }
  if (tts->state == LLAMADART_WEBGPU_TTS_STATE_PROCESSING_PROMPT ||
      tts->state == LLAMADART_WEBGPU_TTS_STATE_GENERATING) {
    return error(
        tts,
        LLAMADART_WEBGPU_TTS_STATUS_INVALID_STATE,
        "a TTS task is already active");
  }
  if (request.text == nullptr || request.text_length == 0 ||
      request.prompt_batch_size <= 0 || request.max_frames <= 0 ||
      request.sequence_id < 0 || request.top_k < 0 ||
      !std::isfinite(request.top_p) || request.top_p < 0.0f ||
      request.top_p > 1.0f || !std::isfinite(request.min_p) ||
      request.min_p < 0.0f || request.min_p > 1.0f ||
      !std::isfinite(request.temperature) || request.temperature < 0.0f ||
      (request.speaker_audio_length > 0 && request.speaker_audio == nullptr)) {
    return error(
        tts,
        LLAMADART_WEBGPU_TTS_STATUS_INVALID_ARGUMENT,
        "invalid TTS request");
  }

  release_task_resources(tts);
  tts->pcm.clear();
  tts->sample_rate = 0;
  tts->sample_count = 0;
  tts->frames_generated = 0;
  tts->prompt_tokens_remaining = 0;
  tts->truncated = false;
  tts->error.clear();
  tts->cancel_requested.store(false, std::memory_order_release);
  tts->sequence_id = request.sequence_id;
  tts->prompt_batch_size = request.prompt_batch_size;
  tts->max_frames = request.max_frames;
  tts->language = request.language != nullptr ? request.language : "";

  llama_memory_seq_rm(
      llama_get_memory(tts->llama), tts->sequence_id, 0, -1);
  tts->owns_sequence = true;
  // The audio generator consumes the backbone's last hidden state. llama.cpp's
  // TTS tool enables embedding outputs for the full context; the bridge keeps
  // its shared context in normal chat mode and scopes that setting to this
  // task instead.
  llama_set_embeddings(tts->llama, true);
  tts->owns_embedding_mode = true;

  if (request.speaker_audio_length > 0) {
    mtmd_helper_bitmap_wrapper wrapper = llama_webgpu_bitmap_from_buffer(
        tts->mtmd,
        request.speaker_audio,
        request.speaker_audio_length);
    if (wrapper.bitmap == nullptr || !mtmd_bitmap_is_audio(wrapper.bitmap)) {
      if (wrapper.bitmap != nullptr) {
        mtmd_bitmap_free(wrapper.bitmap);
      }
      return fail(
          tts,
          LLAMADART_WEBGPU_TTS_STATUS_SPEAKER_DECODE_FAILED,
          "speaker reference audio could not be decoded");
    }
    tts->speaker = wrapper.bitmap;
  }

  tts->sampler = init_sampler(request);
  if (tts->sampler == nullptr) {
    return fail(
        tts,
        LLAMADART_WEBGPU_TTS_STATUS_UPSTREAM_ERROR,
        "sampler initialization failed");
  }

  mtmd_helper_gen_audio_inp input{};
  input.seq_id = tts->sequence_id;
  input.prompt = request.text;
  input.prompt_len = request.text_length;
  input.speaker_ref = tts->speaker;
  input.lang = tts->language.empty() ? nullptr : tts->language.c_str();
  input.top_k = request.top_k;
  input.top_p = request.top_p;
  input.seed = request.seed;
  input.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
  if (mtmd_helper_gen_audio_set_input(tts->generator, &input) != 0) {
    return fail(
        tts,
        LLAMADART_WEBGPU_TTS_STATUS_UPSTREAM_ERROR,
        "TTS input setup failed");
  }

  tts->state = LLAMADART_WEBGPU_TTS_STATE_PROCESSING_PROMPT;
  return LLAMADART_WEBGPU_TTS_STATUS_OK;
}

llama_webgpu_tts_status llama_webgpu_tts_step(
    llama_webgpu_tts * tts,
    llama_webgpu_tts_progress * out_progress) {
  if (tts == nullptr || out_progress == nullptr) {
    return LLAMADART_WEBGPU_TTS_STATUS_INVALID_ARGUMENT;
  }

  const bool active =
      tts->state == LLAMADART_WEBGPU_TTS_STATE_PROCESSING_PROMPT ||
      tts->state == LLAMADART_WEBGPU_TTS_STATE_GENERATING;
  if (active && tts->cancel_requested.load(std::memory_order_acquire)) {
    tts->state = LLAMADART_WEBGPU_TTS_STATE_CANCELLED;
    tts->error = "TTS task cancelled";
    release_task_resources(tts);
    write_progress(tts, out_progress);
    return LLAMADART_WEBGPU_TTS_STATUS_CANCELLED;
  }

  if (tts->state == LLAMADART_WEBGPU_TTS_STATE_PROCESSING_PROMPT) {
    const int32_t remaining = mtmd_helper_gen_audio_step_prompt(
        tts->generator, tts->prompt_batch_size);
    if (remaining < 0) {
      const auto status = fail(
          tts,
          LLAMADART_WEBGPU_TTS_STATUS_UPSTREAM_ERROR,
          "TTS prompt processing failed");
      write_progress(tts, out_progress);
      return status;
    }
    tts->prompt_tokens_remaining = remaining;
    if (remaining == 0) {
      tts->state = LLAMADART_WEBGPU_TTS_STATE_GENERATING;
    }
    write_progress(tts, out_progress);
    return LLAMADART_WEBGPU_TTS_STATUS_OK;
  }

  if (tts->state == LLAMADART_WEBGPU_TTS_STATE_GENERATING) {
    if (tts->frames_generated >= tts->max_frames) {
      tts->truncated = true;
      const auto status = finish_output(tts);
      write_progress(tts, out_progress);
      return status;
    }

    const llama_token sampled =
        llama_sampler_sample(tts->sampler, tts->llama, -1);
    if (sampled == LLAMA_TOKEN_NULL) {
      const auto status = fail(
          tts,
          LLAMADART_WEBGPU_TTS_STATUS_UPSTREAM_ERROR,
          "TTS token sampling failed");
      write_progress(tts, out_progress);
      return status;
    }
    const llama_vocab * vocab =
        llama_model_get_vocab(llama_get_model(tts->llama));
    if (vocab == nullptr) {
      const auto status = fail(
          tts,
          LLAMADART_WEBGPU_TTS_STATUS_UPSTREAM_ERROR,
          "TTS vocabulary is unavailable");
      write_progress(tts, out_progress);
      return status;
    }
    if (llama_vocab_is_eog(vocab, sampled)) {
      const auto status = finish_output(tts);
      write_progress(tts, out_progress);
      return status;
    }

    const float * state = llama_get_embeddings_ith(tts->llama, -1);
    const float * next_state = nullptr;
    bool stop = false;
    if (state == nullptr ||
        mtmd_helper_gen_audio_step_gen(
            tts->generator,
            sampled,
            state,
            &next_state,
            &stop) != 0) {
      const auto status = fail(
          tts,
          LLAMADART_WEBGPU_TTS_STATUS_UPSTREAM_ERROR,
          "TTS generation step failed");
      write_progress(tts, out_progress);
      return status;
    }
    if (next_state != nullptr) {
      ++tts->frames_generated;
    }
    if (stop || next_state == nullptr) {
      const auto status = finish_output(tts);
      write_progress(tts, out_progress);
      return status;
    }

    write_progress(tts, out_progress);
    return LLAMADART_WEBGPU_TTS_STATUS_OK;
  }

  write_progress(tts, out_progress);
  return tts->state == LLAMADART_WEBGPU_TTS_STATE_CANCELLED
      ? LLAMADART_WEBGPU_TTS_STATUS_CANCELLED
      : LLAMADART_WEBGPU_TTS_STATUS_INVALID_STATE;
}

void llama_webgpu_tts_cancel(llama_webgpu_tts * tts) {
  if (tts != nullptr) {
    tts->cancel_requested.store(true, std::memory_order_release);
  }
}

llama_webgpu_tts_status llama_webgpu_tts_reset(llama_webgpu_tts * tts) {
  if (tts == nullptr) {
    return LLAMADART_WEBGPU_TTS_STATUS_INVALID_ARGUMENT;
  }
  release_task_resources(tts);
  tts->cancel_requested.store(false, std::memory_order_release);
  tts->state = LLAMADART_WEBGPU_TTS_STATE_IDLE;
  tts->prompt_tokens_remaining = 0;
  tts->frames_generated = 0;
  tts->truncated = false;
  tts->sample_rate = 0;
  tts->sample_count = 0;
  tts->pcm.clear();
  tts->language.clear();
  tts->error.clear();
  return LLAMADART_WEBGPU_TTS_STATUS_OK;
}

llama_webgpu_tts_status llama_webgpu_tts_write_pcm(
    const llama_webgpu_tts * tts,
    const char * output_path) {
  if (tts == nullptr || output_path == nullptr || output_path[0] == '\0') {
    return LLAMADART_WEBGPU_TTS_STATUS_INVALID_ARGUMENT;
  }
  if (tts->state != LLAMADART_WEBGPU_TTS_STATE_COMPLETED || tts->pcm.empty()) {
    return LLAMADART_WEBGPU_TTS_STATUS_INVALID_STATE;
  }

  FILE * output = std::fopen(output_path, "wb");
  if (output == nullptr) {
    return LLAMADART_WEBGPU_TTS_STATUS_UPSTREAM_ERROR;
  }
  const size_t written =
      std::fwrite(tts->pcm.data(), sizeof(float), tts->pcm.size(), output);
  const int close_result = std::fclose(output);
  if (written != tts->pcm.size() || close_result != 0) {
    return LLAMADART_WEBGPU_TTS_STATUS_UPSTREAM_ERROR;
  }
  return LLAMADART_WEBGPU_TTS_STATUS_OK;
}

int32_t llama_webgpu_tts_sample_rate(const llama_webgpu_tts * tts) {
  return tts != nullptr ? tts->sample_rate : 0;
}

int64_t llama_webgpu_tts_sample_count(const llama_webgpu_tts * tts) {
  return tts != nullptr ? tts->sample_count : 0;
}

const char * llama_webgpu_tts_last_error(const llama_webgpu_tts * tts) {
  return tts != nullptr ? tts->error.c_str() : "TTS task is unavailable";
}
