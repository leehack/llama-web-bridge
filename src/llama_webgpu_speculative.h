#pragma once

#include <string>

#include "llama.h"
#include "speculative.h"

// Speculative decoding setup that must not abort the core.
//
// llama.cpp reports a draft model it cannot load (an unknown architecture, a
// malformed file, a buffer it cannot allocate) and an incompatible draft (a
// vocabulary mismatch, a draft that is not EAGLE3, a DSpark draft without a
// confidence head) by throwing std::exception, and context creation throws
// when a buffer cannot be allocated. The core is built without exception
// catching, so those throws would abort the runtime. These functions are
// linked with catching enabled (EXCEPTION_CATCHING_ALLOWED in
// CMakeLists.txt). A throw unwinds straight to them without running upstream
// destructors, so a rejected load or configuration leaks what upstream built
// before it. Each returns nullptr and sets error_out on failure.

// Loads a draft model. std::bad_alloc is reported as not fitting in memory.
extern "C" llama_model * llamadart_webgpu_draft_model_load_file(
    const char * path,
    llama_model_params params,
    std::string * error_out);

// Creates a draft context. std::bad_alloc is reported as not fitting in
// memory.
extern "C" llama_context * llamadart_webgpu_draft_context_init(
    llama_model * model,
    llama_context_params params,
    std::string * error_out);

// Initializes speculative decoding for one sequence. std::bad_alloc aborts.
extern "C" common_speculative * llamadart_webgpu_speculative_init(
    common_params_speculative * params,
    std::string * error_out);
