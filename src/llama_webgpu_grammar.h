#pragma once

#include <string>

#include "llama.h"

// Initializes a GBNF grammar sampler without aborting on invalid grammar.
//
// llama.cpp's grammar parser reports syntax errors by throwing
// std::runtime_error and catching it inside llama_grammar_parser::parse. The
// core is built without exception catching, so that throw would abort the
// runtime. This is the only function linked with catching enabled
// (EXCEPTION_CATCHING_ALLOWED in CMakeLists.txt), which keeps the cost off
// every other code path, grammar sampling included. The throw unwinds straight
// to it without running upstream destructors, so an invalid grammar leaks the
// partly built parser state. Returns nullptr and sets error_out on failure;
// std::bad_alloc still aborts.
extern "C" llama_sampler * llamadart_webgpu_grammar_sampler_init(
    const llama_vocab * vocab,
    const char * grammar,
    const char * root,
    std::string * error_out);
