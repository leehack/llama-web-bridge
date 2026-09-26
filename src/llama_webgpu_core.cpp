#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include <unistd.h>

#include <malloc.h>

#include <emscripten/emscripten.h>
#include <emscripten/heap.h>
#include <emscripten/threading.h>
#include <emscripten/wasmfs.h>

#include "ggml-backend.h"
#include "llama-cpp.h"
#include "llama-ext.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "ngram-cache.h"
#include "reasoning-budget.h"
#include "speculative.h"

#include "llama_webgpu_decision.h"
#include "llama_webgpu_embedding_json.h"
#include "llama_webgpu_grammar.h"
#include "llama_webgpu_lora.h"
#include "llama_webgpu_mtmd_compat.h"
#include "llama_webgpu_next_token_scores.h"
#include "llama_webgpu_speculative.h"
#include "llama_webgpu_tts.h"

namespace {

#include "core/state.inc"

#include "core/support.inc"

#include "core/model_info.inc"

#include "core/tokens.inc"

#include "core/multimodal.inc"

#include "core/speculative.inc"

#include "core/generation.inc"

#include "core/model_load.inc"

#include "core/lora.inc"

}  // namespace

extern "C" {

#include "core/exports_runtime.inc"

#include "core/exports_model.inc"

#include "core/exports_multimodal.inc"

#include "core/exports_context.inc"

#include "core/exports_generation.inc"

#include "core/exports_speculative.inc"

#include "core/exports_tts.inc"

#include "core/exports_decision.inc"

#include "core/exports_lora.inc"

EMSCRIPTEN_KEEPALIVE void llamadart_webgpu_shutdown() {
  free_runtime();

  if (g_backend_initialized) {
    llama_backend_free();
    g_backend_initialized = false;
  }

  g_has_webgpu = false;
  g_backend_json = "[]";
}

}  // extern "C"

int main() {
  return 0;
}
