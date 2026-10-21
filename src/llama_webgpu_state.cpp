#include "llama_webgpu_state.h"

#include <cstdlib>
#include <exception>
#include <new>

// noinline: EXCEPTION_CATCHING_ALLOWED matches the function that holds the
// catch by name, so inlining it into a caller would silently drop the catch.
extern "C" __attribute__((noinline)) bool llamadart_webgpu_state_set_data(
    llama_context * ctx,
    const uint8_t * data,
    size_t size,
    std::string * error_out) {
  try {
    const size_t n_read = llama_state_set_data(ctx, data, size);
    if (n_read == size) {
      return true;
    }
    if (error_out != nullptr) {
      *error_out = n_read == 0
          ? std::string("rejected by llama.cpp")
          : "llama.cpp read " + std::to_string(n_read) + " of its " + std::to_string(size) + " bytes";
    }
  } catch (const std::bad_alloc &) {
    // Out of memory is not a corrupt snapshot. Abort as a build without
    // exception catching does.
    std::abort();
  } catch (const std::exception & error) {
    if (error_out != nullptr) {
      *error_out = error.what();
    }
  }
  return false;
}
