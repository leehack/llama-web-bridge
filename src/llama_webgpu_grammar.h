#pragma once

#include <cstddef>
#include <string>

#include "llama.h"

// Initializes a GBNF grammar sampler without aborting on invalid grammar.
//
// llama.cpp's grammar parser reports syntax errors by throwing
// std::runtime_error and catching it inside llama_grammar_parser::parse. The
// core is built without exception catching, so that throw would abort the
// runtime. This function is one of the few linked with catching enabled
// (EXCEPTION_CATCHING_ALLOWED in CMakeLists.txt), which keeps the cost off
// every other code path, grammar sampling included. The throw unwinds straight
// to it without running upstream destructors, so an invalid grammar leaks the
// partly built parser state. Returns nullptr and sets error_out on failure;
// std::bad_alloc still aborts.
//
// With lazy set, it builds llama.cpp's lazy grammar, which waits for one of
// trigger_patterns (ECMAScript regular expressions, which throw
// std::regex_error when invalid) or trigger_tokens.
extern "C" llama_sampler * llamadart_webgpu_grammar_sampler_init(
    const llama_vocab * vocab,
    const char * grammar,
    const char * root,
    bool lazy,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens,
    std::string * error_out);

// Accepts token into a lazy grammar sampler without aborting. When the token
// fires a trigger, llama.cpp replays the trigger text into the grammar and
// throws std::runtime_error if the grammar rejects it; sampling checks only the
// token, not that text. Returns false and sets error_out on that throw, after
// which the grammar must be freed; std::bad_alloc still aborts.
extern "C" bool llamadart_webgpu_grammar_sampler_accept(
    llama_sampler * grammar,
    llama_token token,
    std::string * error_out);
