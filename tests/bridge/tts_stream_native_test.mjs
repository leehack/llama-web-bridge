import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

import { repoRoot } from './bridge_js_source.mjs';

const llama = String.raw`
#pragma once
#include <cstdint>

using llama_token = int32_t;
using llama_seq_id = int32_t;
constexpr llama_token LLAMA_TOKEN_NULL = -1;
struct llama_context {};
struct llama_model {};
struct llama_vocab {};
struct llama_sampler {};

inline int eos_frame = 187;
inline int generated_frames = 0;

inline const llama_token * llama_vocab_get_suppress_tokens(const llama_vocab *, int32_t *) { return nullptr; }
struct llama_logit_bias { llama_token token; float bias; };
inline int llama_vocab_n_tokens(const llama_vocab *) { return 2; }
inline llama_sampler * llama_sampler_init_logit_bias(int, int, const llama_logit_bias *) { return nullptr; }
inline int llama_sampler_chain_default_params() { return 0; }
inline llama_sampler * llama_sampler_chain_init(int) { return new llama_sampler; }
inline void llama_sampler_chain_add(llama_sampler *, llama_sampler *) {}
inline llama_sampler * llama_sampler_init_top_k(int) { return nullptr; }
inline llama_sampler * llama_sampler_init_top_p(float, int) { return nullptr; }
inline llama_sampler * llama_sampler_init_min_p(float, int) { return nullptr; }
inline llama_sampler * llama_sampler_init_temp(float) { return nullptr; }
inline llama_sampler * llama_sampler_init_dist(uint32_t) { return nullptr; }
inline void llama_sampler_free(llama_sampler * sampler) { delete sampler; }
inline llama_token llama_sampler_sample(llama_sampler *, llama_context *, int) { return generated_frames >= eos_frame; }
inline bool llama_vocab_is_eog(const llama_vocab *, llama_token token) { return token == 1; }
inline const llama_model * llama_get_model(llama_context *) { static llama_model model; return &model; }
inline const llama_vocab * llama_model_get_vocab(const llama_model *) { static llama_vocab vocab; return &vocab; }
inline const float * llama_get_embeddings_ith(llama_context *, int) { static const float state = 1; return &state; }
inline void llama_set_embeddings(llama_context *, bool) {}
inline void * llama_get_memory(llama_context *) { return nullptr; }
inline void llama_memory_seq_rm(void *, llama_seq_id, int, int) {}
`;

const mtmd = String.raw`
#pragma once

struct mtmd_context {};
struct mtmd_bitmap {};
enum mtmd_gen_audio_type { MTMD_GEN_AUDIO_TYPE_NONE, MTMD_GEN_AUDIO_TYPE_QWEN3TTS };
struct mtmd_gen_audio_info { mtmd_gen_audio_type type; int sample_rate; };
inline mtmd_gen_audio_info mtmd_gen_audio_get_info(const mtmd_context *) { return { MTMD_GEN_AUDIO_TYPE_QWEN3TTS, 24000 }; }
inline bool mtmd_support_audio(const mtmd_context *) { return true; }
inline bool mtmd_bitmap_is_audio(const mtmd_bitmap *) { return true; }
inline void mtmd_bitmap_free(mtmd_bitmap * bitmap) { delete bitmap; }
`;

// The sample values depend on window boundaries so a premature one-shot read
// changes the output even when the total number of samples stays the same.
const helper = String.raw`
#pragma once
#include "llama.h"
#include "mtmd.h"
#include <cstddef>
#include <vector>

struct mtmd_helper_bitmap_wrapper { mtmd_bitmap * bitmap; };
inline mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(mtmd_context *, const unsigned char *, size_t, bool) { return {nullptr}; }
inline mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_file(mtmd_context *, const char *, bool) { return {nullptr}; }

constexpr int MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM = 0;
struct mtmd_helper_gen_audio_inp {
  llama_seq_id seq_id;
  const char * prompt;
  size_t prompt_len;
  mtmd_bitmap * speaker_ref;
  const char * lang;
  int top_k;
  float top_p;
  uint32_t seed;
  int out_type;
  bool stream;
};
struct mtmd_helper_gen_audio {
  bool stream = false;
  int pending_frames = 0;
  size_t sent = 0;
  std::vector<float> pcm;
};
constexpr int window_frames = 72;
constexpr int samples_per_frame = 1920;
inline bool fail_flush = false;
inline bool fail_output = false;
inline mtmd_helper_gen_audio * mtmd_helper_gen_audio_init(llama_context *, mtmd_context *) {
  return new mtmd_helper_gen_audio;
}
inline void mtmd_helper_gen_audio_free(mtmd_helper_gen_audio * gen) { delete gen; }
inline void mtmd_helper_gen_audio_reset(mtmd_helper_gen_audio * gen) { *gen = {}; }
inline int mtmd_helper_gen_audio_set_input(mtmd_helper_gen_audio * gen, const mtmd_helper_gen_audio_inp * input) {
  generated_frames = 0;
  gen->stream = input->stream;
  return 0;
}
inline int mtmd_helper_gen_audio_step_prompt(mtmd_helper_gen_audio *, int) { return 0; }
inline int mtmd_helper_gen_audio_flush(mtmd_helper_gen_audio * gen) {
  if (fail_flush) return 1;
  for (int i = 0; i < gen->pending_frames * samples_per_frame; ++i) {
    gen->pcm.push_back(float(generated_frames * 256 + gen->pending_frames) + float(i % 32) / 32);
  }
  gen->pending_frames = 0;
  return 0;
}
inline int mtmd_helper_gen_audio_step_gen(mtmd_helper_gen_audio * gen, llama_token, const float * state, const float ** next, bool * stop) {
  ++generated_frames;
  *next = state;
  *stop = false;
  if (++gen->pending_frames == window_frames) return mtmd_helper_gen_audio_flush(gen);
  return 0;
}
inline int mtmd_helper_gen_audio_get_output(mtmd_helper_gen_audio * gen, int32_t * rate, const char ** data, size_t * len, int64_t * count) {
  if (fail_output || (!gen->stream && mtmd_helper_gen_audio_flush(gen))) return 1;
  *rate = 24000;
  *count = gen->pcm.size() - gen->sent;
  *len = *count * sizeof(float);
  *data = *count ? reinterpret_cast<const char *>(gen->pcm.data() + gen->sent) : nullptr;
  if (gen->stream) gen->sent = gen->pcm.size();
  return 0;
}
`;

const fixture = String.raw`
#include "src/llama_webgpu_tts.cpp"

#include <cassert>
#include <fstream>
#include <iterator>

constexpr char pcm_path[] = "tts.pcm";
constexpr auto ok = LLAMADART_WEBGPU_TTS_STATUS_OK;
constexpr auto invalid_state = LLAMADART_WEBGPU_TTS_STATUS_INVALID_STATE;

void start(llama_webgpu_tts * tts, bool stream, int max_frames) {
  llama_webgpu_tts_request request{};
  request.text = "test";
  request.text_length = 4;
  request.prompt_batch_size = 512;
  request.max_frames = max_frames;
  request.top_p = 1;
  request.stream_audio = stream;
  assert(llama_webgpu_tts_start(tts, request) == ok);
}

std::vector<char> read_pcm() {
  std::ifstream file(pcm_path, std::ios::binary);
  assert(file);
  std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();
  assert(std::remove(pcm_path) == 0);
  return bytes;
}

struct output {
  std::vector<char> bytes;
  std::vector<int64_t> chunks;
};

output generate(llama_webgpu_tts * tts, bool stream, int max_frames, int interval) {
  start(tts, stream, max_frames);
  output actual;
  llama_webgpu_tts_progress progress{};
  int steps = 0;
  do {
    assert(llama_webgpu_tts_step(tts, &progress) == ok);
    if (stream && (++steps % interval == 0 || progress.state == LLAMADART_WEBGPU_TTS_STATE_COMPLETED)) {
      int64_t samples = -1;
      assert(llama_webgpu_tts_write_stream_chunk(tts, pcm_path, &samples) == ok);
      if (samples) {
        const auto bytes = read_pcm();
        assert(bytes.size() == samples * sizeof(float));
        actual.chunks.push_back(samples);
        actual.bytes.insert(actual.bytes.end(), bytes.begin(), bytes.end());
      }
    }
  } while (progress.state != LLAMADART_WEBGPU_TTS_STATE_COMPLETED);
  int64_t samples = -1;
  if (stream) {
    assert(llama_webgpu_tts_write_pcm(tts, pcm_path) == invalid_state);
    assert(llama_webgpu_tts_write_stream_chunk(tts, pcm_path, &samples) == ok);
    assert(samples == 0);
    assert(!std::ifstream(pcm_path));
  } else {
    assert(llama_webgpu_tts_write_stream_chunk(tts, pcm_path, &samples) == invalid_state);
    assert(samples == 0);
    assert(llama_webgpu_tts_write_pcm(tts, pcm_path) == ok);
    actual.bytes = read_pcm();
  }
  return actual;
}

int main() {
  llama_context llama;
  mtmd_context mtmd;
  llama_webgpu_tts * tts = llama_webgpu_tts_init(&llama, &mtmd, nullptr);
  assert(tts);
  assert(llama_webgpu_tts_get_info(&mtmd).capabilities == 11);
  for (const int frames : {187, 72, 73, 144}) {
    const int max_frames = frames == eos_frame ? 200 : frames;
    const auto nonstream = generate(tts, false, max_frames, 1);
    const auto streamed = generate(tts, true, max_frames, 1);
    assert(nonstream.bytes.size() == frames * samples_per_frame * sizeof(float));
    assert(streamed.bytes == nonstream.bytes);
    std::vector<int64_t> expected(frames / window_frames, window_frames * samples_per_frame);
    if (frames % window_frames) expected.push_back((frames % window_frames) * samples_per_frame);
    assert(streamed.chunks == expected);
    const auto deferred = generate(tts, true, max_frames, 3);
    assert(deferred.bytes == nonstream.bytes);
    if (frames == 73) assert(deferred.chunks == std::vector<int64_t>{73 * samples_per_frame});
    assert(generate(tts, false, max_frames, 1).bytes == nonstream.bytes);
  }

  start(tts, true, 187);
  llama_webgpu_tts_progress progress{};
  while (progress.frames_generated < 72) assert(llama_webgpu_tts_step(tts, &progress) == ok);
  int64_t samples = -1;
  assert(llama_webgpu_tts_write_stream_chunk(tts, pcm_path, &samples) == ok);
  assert(samples == 72 * samples_per_frame);
  read_pcm();
  while (progress.frames_generated < 144) assert(llama_webgpu_tts_step(tts, &progress) == ok);
  assert(!tts->stream_chunk.empty());
  llama_webgpu_tts_cancel(tts);
  assert(llama_webgpu_tts_step(tts, &progress) == LLAMADART_WEBGPU_TTS_STATUS_CANCELLED);
  assert(progress.state == LLAMADART_WEBGPU_TTS_STATE_CANCELLED);
  assert(tts->stream_chunk.empty());
  assert(llama_webgpu_tts_write_stream_chunk(tts, pcm_path, &samples) == invalid_state);
  assert(samples == 0);

  start(tts, true, 1);
  assert(llama_webgpu_tts_step(tts, &progress) == ok);
  assert(llama_webgpu_tts_step(tts, &progress) == ok);
  fail_flush = true;
  assert(llama_webgpu_tts_step(tts, &progress) == LLAMADART_WEBGPU_TTS_STATUS_UPSTREAM_ERROR);
  assert(std::string(llama_webgpu_tts_last_error(tts)) == "audio flush failed");
  fail_flush = false;

  start(tts, true, 10);
  eos_frame = 0;
  assert(llama_webgpu_tts_step(tts, &progress) == ok);
  assert(llama_webgpu_tts_step(tts, &progress) == LLAMADART_WEBGPU_TTS_STATUS_UPSTREAM_ERROR);
  assert(std::string(llama_webgpu_tts_last_error(tts)) == "audio output metadata is invalid");
  eos_frame = 187;

  start(tts, true, 10);
  assert(llama_webgpu_tts_step(tts, &progress) == ok);
  fail_output = true;
  assert(llama_webgpu_tts_step(tts, &progress) == LLAMADART_WEBGPU_TTS_STATUS_UPSTREAM_ERROR);
  assert(std::string(llama_webgpu_tts_last_error(tts)) == "audio output metadata is invalid");
  fail_output = false;
  const auto fresh = generate(tts, true, 1, 1);
  assert(fresh.chunks == std::vector<int64_t>{samples_per_frame});
  llama_webgpu_tts_free(tts);
}
`;

test('native streamed TTS drains only pending chunks and keeps one-shot PCM byte-identical', () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'lane220-tts-native-'));
  try {
    for (const [name, source] of Object.entries({
      'llama.h': llama,
      'mtmd.h': mtmd,
      'mtmd-helper.h': helper,
      'fixture.cpp': fixture,
    })) {
      writeFileSync(path.join(dir, name), source);
    }
    execFileSync(process.env.CXX || 'c++', [
      '-std=c++17', '-I', dir, '-I', repoRoot, 'fixture.cpp', '-o', 'fixture',
    ], { cwd: dir, stdio: 'pipe' });
    execFileSync(path.join(dir, 'fixture'), [], { cwd: dir, stdio: 'pipe' });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
