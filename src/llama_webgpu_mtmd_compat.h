#pragma once

#include "mtmd-helper.h"

// v0.4.0 added helper options. Preserve older upstream builds and use the
// upstream defaults without changing the bridge's non-placeholder behavior.
static inline mtmd_helper_bitmap_wrapper llama_webgpu_bitmap_from_buffer(
    mtmd_context * ctx, const unsigned char * bytes, size_t length) {
#if LLAMADART_MTMD_HELPER_HAS_OPTIONS
  return mtmd_helper_bitmap_init_from_buf(
      ctx, bytes, length, false, mtmd_helper_init_opt_default());
#else
  return mtmd_helper_bitmap_init_from_buf(ctx, bytes, length, false);
#endif
}

static inline mtmd_helper_bitmap_wrapper llama_webgpu_bitmap_from_file(
    mtmd_context * ctx, const char * path) {
#if LLAMADART_MTMD_HELPER_HAS_OPTIONS
  return mtmd_helper_bitmap_init_from_file(
      ctx, path, false, mtmd_helper_init_opt_default());
#else
  return mtmd_helper_bitmap_init_from_file(ctx, path, false);
#endif
}
