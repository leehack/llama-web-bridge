#include "llama_webgpu_grammar.h"

#include <cstdlib>
#include <exception>
#include <new>

// noinline: EXCEPTION_CATCHING_ALLOWED matches the function that holds the
// catch by name, so inlining it into a caller would silently drop the catch.
extern "C" __attribute__((noinline)) llama_sampler *
llamadart_webgpu_grammar_sampler_init(
    const llama_vocab * vocab,
    const char * grammar,
    const char * root,
    std::string * error_out) {
  try {
    llama_sampler * sampler = llama_sampler_init_grammar(vocab, grammar, root);
    if (sampler == nullptr && error_out != nullptr) {
      // Parsed but rejected without throwing, e.g. left recursion or a
      // missing root rule. llama.cpp logs the reason.
      *error_out = "grammar rejected by llama.cpp";
    }
    return sampler;
  } catch (const std::bad_alloc &) {
    // Out of memory is not an invalid grammar. Abort as a build without
    // exception catching does.
    std::abort();
  } catch (const std::exception & error) {
    if (error_out != nullptr) {
      *error_out = error.what();
    }
    return nullptr;
  }
}
