#include "llama_webgpu_lora.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <new>

// noinline: EXCEPTION_CATCHING_ALLOWED matches the function that holds the
// catch by name, so inlining it into a caller would silently drop the catch.
extern "C" __attribute__((noinline)) llama_adapter_lora *
llamadart_webgpu_lora_adapter_init(
    llama_model * model,
    const char * path,
    std::string * error_out) {
  FILE * file = std::fopen(path, "rb");
  if (file == nullptr) {
    if (error_out != nullptr) {
      *error_out = "the staged adapter file cannot be opened";
    }
    return nullptr;
  }
  llama_adapter_lora * adapter = nullptr;
  try {
    adapter = llama_adapter_lora_init_from_file_ptr(model, file);
    if (adapter == nullptr && error_out != nullptr) {
      *error_out = "rejected by llama.cpp";
    }
  } catch (const std::bad_alloc &) {
    // Out of memory is not an invalid adapter. Abort as a build without
    // exception catching does.
    std::abort();
  } catch (const std::exception & error) {
    if (error_out != nullptr) {
      *error_out = error.what();
    }
  }
  std::fclose(file);
  return adapter;
}
