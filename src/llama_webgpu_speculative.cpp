#include "llama_webgpu_speculative.h"

#include <cstdlib>
#include <exception>
#include <new>

// noinline: EXCEPTION_CATCHING_ALLOWED matches the function that holds the
// catch by name, so inlining it into a caller would silently drop the catch.
extern "C" __attribute__((noinline)) llama_model *
llamadart_webgpu_draft_model_load_file(
    const char * path,
    llama_model_params params,
    std::string * error_out) {
  try {
    llama_model * model = llama_model_load_from_file(path, params);
    if (model == nullptr && error_out != nullptr) {
      *error_out = "llama.cpp could not load the draft model";
    }
    return model;
  } catch (const std::bad_alloc &) {
    if (error_out != nullptr) {
      *error_out = "the draft model does not fit in memory";
    }
    return nullptr;
  } catch (const std::exception & error) {
    if (error_out != nullptr) {
      *error_out = error.what();
    }
    return nullptr;
  }
}

extern "C" __attribute__((noinline)) llama_context *
llamadart_webgpu_draft_context_init(
    llama_model * model,
    llama_context_params params,
    std::string * error_out) {
  try {
    llama_context * ctx = llama_init_from_model(model, params);
    if (ctx == nullptr && error_out != nullptr) {
      *error_out = "llama.cpp could not create the draft context";
    }
    return ctx;
  } catch (const std::bad_alloc &) {
    if (error_out != nullptr) {
      *error_out = "the draft context does not fit in memory";
    }
    return nullptr;
  } catch (const std::exception & error) {
    if (error_out != nullptr) {
      *error_out = error.what();
    }
    return nullptr;
  }
}

extern "C" __attribute__((noinline)) common_speculative *
llamadart_webgpu_speculative_init(
    common_params_speculative * params,
    std::string * error_out) {
  try {
    common_speculative * spec = common_speculative_init(*params, 1);
    if (spec == nullptr && error_out != nullptr) {
      *error_out = "no speculative implementation was selected";
    }
    return spec;
  } catch (const std::bad_alloc &) {
    std::abort();
  } catch (const std::exception & error) {
    if (error_out != nullptr) {
      *error_out = error.what();
    }
    return nullptr;
  }
}
