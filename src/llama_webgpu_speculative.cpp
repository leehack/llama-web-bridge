#include "llama_webgpu_speculative.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>

namespace {

constexpr const char * kContextInitFailure = "failed to initialize the context: ";

// llama_init_from_model frees a context it cannot create and reports why only
// in its log. While alive, this keeps that reason and forwards every message.
class context_init_log {
 public:
  context_init_log() {
    llama_log_get(&forward_, &forward_data_);
    llama_log_set(capture, this);
  }
  ~context_init_log() { llama_log_set(forward_, forward_data_); }
  context_init_log(const context_init_log &) = delete;
  context_init_log & operator=(const context_init_log &) = delete;

  const std::string & reason() const { return reason_; }

 private:
  static void capture(ggml_log_level level, const char * text, void * user_data) {
    auto * log = static_cast<context_init_log *>(user_data);
    const char * reason = text == nullptr ? nullptr : std::strstr(text, kContextInitFailure);
    if (level == GGML_LOG_LEVEL_ERROR && reason != nullptr) {
      log->reason_ = reason + std::strlen(kContextInitFailure);
      while (!log->reason_.empty() && (log->reason_.back() == '\n' || log->reason_.back() == '\r')) {
        log->reason_.pop_back();
      }
    }
    if (log->forward_ != nullptr) {
      log->forward_(level, text, log->forward_data_);
    }
  }

  ggml_log_callback forward_ = nullptr;
  void * forward_data_ = nullptr;
  std::string reason_;
};

}  // namespace

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
    const context_init_log log;
    llama_context * ctx = llama_init_from_model(model, params);
    if (ctx == nullptr && error_out != nullptr) {
      *error_out = log.reason().empty() ? "llama.cpp could not create the draft context" : log.reason();
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
