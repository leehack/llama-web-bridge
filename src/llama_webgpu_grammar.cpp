#include "llama_webgpu_grammar.h"

#include <cstdlib>
#include <exception>
#include <new>
#include <regex>

// noinline: EXCEPTION_CATCHING_ALLOWED matches the function that holds the
// catch by name, so inlining it into a caller would silently drop the catch.
extern "C" __attribute__((noinline)) llama_sampler *
llamadart_webgpu_grammar_sampler_init(
    const llama_vocab * vocab,
    const char * grammar,
    const char * root,
    const bool lazy,
    const char ** trigger_patterns,
    const size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    const size_t num_trigger_tokens,
    std::string * error_out) {
  try {
    // llama.cpp compiles the patterns only after building the grammar, and a
    // bad pattern's std::regex_error does not say which pattern it was.
    for (size_t i = 0; lazy && i < num_trigger_patterns; ++i) {
      try {
        std::regex pattern(trigger_patterns[i]);
      } catch (const std::regex_error & error) {
        if (error_out != nullptr) {
          *error_out = std::string("grammar trigger pattern ") +
              trigger_patterns[i] + " is not a valid regular expression: " +
              error.what();
        }
        return nullptr;
      }
    }
    llama_sampler * sampler = lazy
        ? llama_sampler_init_grammar_lazy_patterns(
              vocab,
              grammar,
              root,
              trigger_patterns,
              num_trigger_patterns,
              trigger_tokens,
              num_trigger_tokens)
        : llama_sampler_init_grammar(vocab, grammar, root);
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

extern "C" __attribute__((noinline)) bool llamadart_webgpu_grammar_sampler_accept(
    llama_sampler * grammar,
    const llama_token token,
    std::string * error_out) {
  try {
    llama_sampler_accept(grammar, token);
    return true;
  } catch (const std::bad_alloc &) {
    std::abort();
  } catch (const std::exception & error) {
    if (error_out != nullptr) {
      *error_out = error.what();
    }
    return false;
  }
}
