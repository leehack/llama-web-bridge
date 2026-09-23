#ifndef LLAMADART_WEBGPU_DECISION_H
#define LLAMADART_WEBGPU_DECISION_H

#include <cstdint>
#include <string>
#include <vector>

#include "llama.h"

struct llama_webgpu_decision_head;

constexpr uint32_t LLAMADART_WEBGPU_DECISION_API_VERSION = 1;

enum llama_webgpu_decision_status {
  LLAMADART_WEBGPU_DECISION_STATUS_OK = 0,
  LLAMADART_WEBGPU_DECISION_STATUS_INVALID_ARGUMENT = -1,
  LLAMADART_WEBGPU_DECISION_STATUS_UNSUPPORTED = -2,
  LLAMADART_WEBGPU_DECISION_STATUS_INVALID_STATE = -3,
  LLAMADART_WEBGPU_DECISION_STATUS_MODEL_ERROR = -4,
  LLAMADART_WEBGPU_DECISION_STATUS_CONTEXT_ERROR = -5,
  LLAMADART_WEBGPU_DECISION_STATUS_INFERENCE_ERROR = -6,
};

struct llama_webgpu_decision_load_request {
  llama_model * model;
  // WasmFS path of the staged safetensors file.
  const char * head_path;
  // Name used in error messages instead of the staging path.
  const char * head_label;
  // WasmFS path of the staged Laya rl_agent_config.json text; null reads the
  // head's laya.config metadata.
  const char * config_path;
  bool use_gpu;
  int32_t n_threads;
  int32_t n_threads_batch;
};

struct llama_webgpu_decision_head_info {
  int32_t hidden_size = 0;
  llama_token cls_token = -1;
  llama_token sep_token = -1;
  llama_token mask_token = -1;
  std::string mask_text;
  std::string config_json;
  std::string device_name;
};

struct llama_webgpu_decision_sequence {
  std::vector<int32_t> tokens;
  std::vector<int32_t> markers;
  int32_t question_type = 0;
};

struct llama_webgpu_decision_output {
  std::vector<float> logits;
  std::vector<float> act_logits;
};

// Returns why `model` cannot run decision heads, or an empty string when it can.
std::string llama_webgpu_decision_unsupported_reason(const llama_model * model);

// JSON `{"apiVersion", "supported", "reason"?}` for `model` (null when unloaded).
std::string llama_webgpu_decision_capabilities_json(const llama_model * model);

// JSON description of a loaded head, including its bridge `handle`.
std::string llama_webgpu_decision_head_info_json(
    const llama_webgpu_decision_head * head,
    int32_t handle);

llama_webgpu_decision_status llama_webgpu_decision_head_load(
    const llama_webgpu_decision_load_request & request,
    llama_webgpu_decision_head ** out_head,
    std::string * out_error);

void llama_webgpu_decision_head_free(llama_webgpu_decision_head * head);

const llama_webgpu_decision_head_info & llama_webgpu_decision_head_get_info(
    const llama_webgpu_decision_head * head);

// Validates every sequence against the head's limits before any encoder pass.
llama_webgpu_decision_status llama_webgpu_decision_validate_sequences(
    const std::vector<llama_webgpu_decision_sequence> & sequences,
    int32_t token_limit,
    int32_t vocab_size,
    std::string * out_error);

// Validates all sequences, then runs each through the encoder and the head.
llama_webgpu_decision_status llama_webgpu_decision_run(
    llama_webgpu_decision_head * head,
    const std::vector<llama_webgpu_decision_sequence> & sequences,
    std::vector<llama_webgpu_decision_output> * out_outputs,
    std::string * out_error);

// Input file: little-endian int32 count, then per sequence question type,
// token count, marker count, tokens and markers.
llama_webgpu_decision_status llama_webgpu_decision_read_sequences(
    const char * path,
    std::vector<llama_webgpu_decision_sequence> * out_sequences,
    std::string * out_error);

// Output file: per sequence little-endian int32 logit count and act count,
// then the float32 logits and act logits.
llama_webgpu_decision_status llama_webgpu_decision_write_outputs(
    const char * path,
    const std::vector<llama_webgpu_decision_output> & outputs,
    std::string * out_error);

#endif
