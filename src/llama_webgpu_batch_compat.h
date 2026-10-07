#pragma once

#include "llama.h"
#include "speculative.h"

// Bridge speculative batches are allocated text batches for sequence zero.
// Keep construction independent of the helpers removed by upstream v0.6.0.
inline void llama_webgpu_batch_clear(llama_batch &batch) {
  batch.n_tokens = 0;
}

inline void llama_webgpu_batch_add(llama_batch &batch, llama_token token,
                                  llama_pos position, bool output) {
  GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");
  const int32_t index = batch.n_tokens++;
  batch.token[index] = token;
  batch.pos[index] = position;
  batch.n_seq_id[index] = 1;
  batch.seq_id[index][0] = 0;
  batch.logits[index] = output;
}

inline bool llama_webgpu_speculative_process(common_speculative *spec,
                                              llama_batch batch) {
#if LLAMADART_WEBGPU_SPECULATIVE_HAS_COMMON_BATCH
  // All metadata is explicit in the bridge-owned batch; reject unsupported
  // layouts before constructing upstream's non-owning process view.
  if (batch.n_tokens < 0 || batch.embd ||
      (batch.n_tokens > 0 && (!batch.token || !batch.pos || !batch.n_seq_id ||
                             !batch.seq_id || !batch.logits))) {
    return false;
  }
  common_batch converted;
  converted.tokens.reserve(batch.n_tokens);
  for (int32_t i = 0; i < batch.n_tokens; ++i) {
    if (batch.n_seq_id[i] != 1 || !batch.seq_id[i] || batch.seq_id[i][0] != 0) {
      return false;
    }
    converted.add(batch.token[i], batch.pos[i], 0, batch.logits[i] != 0);
  }
  return common_speculative_process(spec, converted);
#else
  return common_speculative_process(spec, batch);
#endif
}
