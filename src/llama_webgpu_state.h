#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "llama.h"

// Restores context state from llama_state_get_data bytes without aborting on
// bytes llama.cpp rejects.
//
// llama.cpp reports truncated or mismatched state, and state for another
// architecture or cache layout, by throwing inside llama_state_set_data and
// catching it there. The core is built without exception catching, so that
// throw would abort the runtime. This function is linked with catching enabled
// (EXCEPTION_CATCHING_ALLOWED in CMakeLists.txt). The throw unwinds straight to
// it without running upstream destructors, so a rejected restore leaks
// llama.cpp's bookkeeping for it, and the context memory is left partly
// replaced: restore a saved copy afterwards. Returns false and sets error_out
// unless llama.cpp read exactly `size` bytes; std::bad_alloc still aborts.
extern "C" bool llamadart_webgpu_state_set_data(
    llama_context * ctx,
    const uint8_t * data,
    size_t size,
    std::string * error_out);
