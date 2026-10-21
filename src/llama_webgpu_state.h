#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "llama.h"

// Restore context state without aborting on state llama.cpp rejects.
//
// llama.cpp reports truncated or mismatched state, and state for another
// architecture or cache layout, by throwing inside llama_state_set_data and
// llama_state_load_file and catching it there. The core is built without
// exception catching, so that throw would abort the runtime. These functions
// are linked with catching enabled (EXCEPTION_CATCHING_ALLOWED in
// CMakeLists.txt). The throw unwinds straight to them without running upstream
// destructors, so a rejected restore leaks llama.cpp's bookkeeping for it (for
// a file, also its open handle), and the context memory is left partly
// replaced: restore a saved copy afterwards. Both return false and set
// error_out on failure; std::bad_alloc still aborts.

// Restores llama_state_get_data bytes; fails unless llama.cpp read all `size`.
extern "C" bool llamadart_webgpu_state_set_data(
    llama_context * ctx,
    const uint8_t * data,
    size_t size,
    std::string * error_out);

// Restores a llama_state_save_file file without reading it into memory first.
extern "C" bool llamadart_webgpu_state_read_file(
    llama_context * ctx,
    const char * path,
    llama_token * tokens,
    size_t token_capacity,
    std::string * error_out);
