#pragma once

#include "llama.h"
#include "mtmd.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llamadart_webgpu_detail {

inline bool supports_pooled_embeddings(llama_context * ctx, llama_model * model) {
  return ctx && model && llama_model_n_embd_out(model) > 0 &&
      !(llama_model_has_encoder(model) && llama_model_has_decoder(model)) &&
      llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE &&
      llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_RANK;
}

inline bool embedding_input_fits(llama_context * ctx, size_t count) {
  return count <= llama_n_ctx(ctx) && count <= llama_n_ctx_seq(ctx) &&
      count <= llama_n_batch(ctx) && count <= llama_n_ubatch(ctx) && llama_n_seq_max(ctx) >= 1;
}

struct embedding_scope {
  llama_context * ctx;
  explicit embedding_scope(llama_context * context) : ctx(context) {
    llama_synchronize(ctx);
    if (auto * memory = llama_get_memory(ctx)) llama_memory_clear(memory, false);
    llama_set_embeddings(ctx, true);
  }
  ~embedding_scope() { llama_set_embeddings(ctx, false); }
};

inline int pooled_embedding_output(llama_context * ctx, llama_model * model, std::vector<float> & output) {
  const float * pooled = llama_get_embeddings_seq(ctx, 0);
  if (!pooled) return -25;
  const int width = llama_model_n_embd_out(model);
  output.assign(pooled, pooled + width);
  return width;
}

// The current production upstream has no extended batches. Text-only parts
// still use one complete physical batch, never the legacy string chunk loop.
inline int embed_text_parts(llama_context * ctx, llama_model * model,
    const std::string & prompt, std::vector<float> & output) {
  const auto * vocab = llama_model_get_vocab(model);
  const int needed = llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
  if (needed >= 0 || needed == INT32_MIN) return -25;
  std::vector<llama_token> tokens(static_cast<size_t>(-needed));
  const int n = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true);
  if (n <= 0 || n > static_cast<int>(tokens.size())) return -25;
  if (!embedding_input_fits(ctx, static_cast<size_t>(n))) return -23;
  auto batch = llama_batch_init(n, 0, 1);
  struct batch_cleanup { llama_batch batch; ~batch_cleanup() { llama_batch_free(batch); } } cleanup{batch};
  if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) return -25;
  batch.n_tokens = n;
  for (int i = 0; i < n; ++i) {
    batch.token[i] = tokens[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0; batch.logits[i] = 1;
  }
  embedding_scope scope(ctx);
  const int status = llama_model_has_encoder(model) ? llama_encode(ctx, batch) : llama_decode(ctx, batch);
  if (status != 0) return -25;
  return pooled_embedding_output(ctx, model, output);
}

#if LLAMADART_WEBGPU_HAS_MIXED_EMBEDDINGS
inline bool supports_mixed_embeddings(llama_context * ctx, llama_model * model) {
  return supports_pooled_embeddings(ctx, model) &&
      !llama_model_has_encoder(model) &&
      llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_MEAN &&
      !llama_get_causal_attn(ctx);
}

inline int embed_mixed_parts(llama_context * ctx, llama_model * model,
    mtmd_context * projector, const std::string & prompt,
    const std::vector<const mtmd_bitmap *> & bitmaps, std::vector<float> & output) {
  if (!supports_mixed_embeddings(ctx, model)) return -21;
  for (const auto * bitmap : bitmaps) {
    if ((mtmd_bitmap_is_audio(bitmap) &&
          (!mtmd_support_audio(projector) || mtmd_get_audio_sample_rate(projector) != 16000)) ||
        (!mtmd_bitmap_is_audio(bitmap) && !mtmd_support_vision(projector))) return -24;
  }
  std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>
      chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
  if (!chunks) return -25;
  mtmd_input_text text{prompt.data(), prompt.size(), true, true};
  if (mtmd_tokenize(projector, chunks.get(), &text, bitmaps.data(), bitmaps.size()) != 0) return -25;
  size_t count = 0;
  for (size_t c = 0; c < mtmd_input_chunks_size(chunks.get()); ++c) {
    const size_t n = mtmd_input_chunk_get_n_tokens(mtmd_input_chunks_get(chunks.get(), c));
    if (n > SIZE_MAX - count) return -23;
    count += n;
  }
  if (count == 0) return -25;
  if (!embedding_input_fits(ctx, count)) return -23;

  const int width = llama_model_n_embd_inp(model);
  if (width <= 0) return -25;
  std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)>
      batch(llama_batch_ext_init(ctx), llama_batch_ext_free);
  if (!batch) return -25;
  std::vector<std::vector<float>> encoded_media;
  llama_pos position = 0;
  const auto append = [&](const llama_token * tokens, const float * embeddings, size_t n) {
    for (size_t i = 0; i < n; ++i, ++position) {
      const int index = tokens
          ? llama_batch_ext_add_token(batch.get(), 0, tokens[i])
          : llama_batch_ext_add_embd(batch.get(), 0, {embeddings + i * width, 1, size_t(width)});
      if (index < 0 || !llama_batch_ext_set_pos(batch.get(), index, &position) ||
          !llama_batch_ext_set_output_embd(batch.get(), index, true)) return false;
    }
    return true;
  };
  for (size_t c = 0; c < mtmd_input_chunks_size(chunks.get()); ++c) {
    const auto * chunk = mtmd_input_chunks_get(chunks.get(), c);
    size_t n = mtmd_input_chunk_get_n_tokens(chunk);
    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      const auto * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n);
      if (!tokens || !append(tokens, nullptr, n)) return -25;
    } else {
      if (mtmd_encode_chunk(projector, chunk) != 0) return -25;
      const float * encoded = mtmd_get_output_embd(projector);
      if (!encoded || n > SIZE_MAX / size_t(width)) return -25;
      // mtmd reuses this buffer on every encode; keep each chunk alive until process settles.
      encoded_media.emplace_back(encoded, encoded + n * size_t(width));
      if (!append(nullptr, encoded_media.back().data(), n)) return -25;
    }
  }
  embedding_scope scope(ctx);
  if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) return -25;
  return pooled_embedding_output(ctx, model, output);
}
#endif

// Payload-free status codes; the export owns pending-media/cancel cleanup.
inline int embed_parts(llama_context * ctx, llama_model * model,
    mtmd_context * projector, const std::string & prompt,
    const std::vector<const mtmd_bitmap *> & bitmaps, std::vector<float> & output) {
  if (!supports_pooled_embeddings(ctx, model)) return -21;
  if (bitmaps.empty()) return embed_text_parts(ctx, model, prompt, output);
  if (!projector) return -22;
#if LLAMADART_WEBGPU_HAS_MIXED_EMBEDDINGS
  return embed_mixed_parts(ctx, model, projector, prompt, bitmaps, output);
#else
  return -20;
#endif
}
} // namespace llamadart_webgpu_detail
