#pragma once

#include <cstdint>
#include <string>

#include "llama.h"

constexpr uint32_t LLAMADART_WEBGPU_LORA_API_VERSION = 1;

// Loads a LoRA adapter GGUF without aborting on an adapter llama.cpp rejects.
//
// llama.cpp reports a malformed adapter, or one made for another base model,
// by throwing std::runtime_error inside llama_adapter_lora_init_from_file_ptr
// and catching it there. The core is built without exception catching, so that
// throw would abort the runtime. This function is linked with catching enabled
// (EXCEPTION_CATCHING_ALLOWED in CMakeLists.txt). The throw unwinds straight to
// it without running upstream destructors, so a rejected adapter leaks what
// llama.cpp had built for it. Returns nullptr and sets error_out on failure;
// std::bad_alloc still aborts.
extern "C" llama_adapter_lora * llamadart_webgpu_lora_adapter_init(
    llama_model * model,
    const char * path,
    std::string * error_out);
