import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { readNativeCoreSource } from './native_core_source.mjs';

const source = readNativeCoreSource();
function body(signature) {
  const start = source.indexOf(signature);
  assert.ok(start >= 0, signature);
  return source.slice(start, source.indexOf('\n}\n', start) + 2);
}
const state = source.slice(
  source.indexOf('llama_model * g_draft_model = nullptr;'),
  source.indexOf('speculative_request g_next_speculative;'),
);

const fixture = [
  String.raw`
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#define EMSCRIPTEN_KEEPALIVE
enum llama_dflash_variant : int {
  LLAMA_DFLASH_VARIANT_NONE = -1,
  LLAMA_DFLASH_VARIANT_DFLASH,
  LLAMA_DFLASH_VARIANT_DFLASH2,
  LLAMA_DFLASH_VARIANT_DSPARK,
  LLAMA_DFLASH_VARIANT_LILICORR,
};
enum common_speculative_type {
  COMMON_SPECULATIVE_TYPE_NONE,
  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE,
  COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3,
  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,
  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK,
};
struct llama_model { llama_dflash_variant variant; };
static llama_model target{LLAMA_DFLASH_VARIANT_NONE};
struct { llama_model * model = &target; int ctx = 0; } g_state;
static const char * architecture = "dflash";
static std::string error;
static int draft_allocations = 0;
static int g_model_params = 0;
static bool g_generation_active = false, g_tts_active = false;
static std::vector<int> g_pending_media;

static void clear_error() { error.clear(); }
static void set_error(const std::string & message) { error = message; }
static bool ensure_loaded() { return g_state.model != nullptr; }
static void free_speculative_session() {}
static bool target_has_mtp_head() { return false; }
static bool draft_layer_ids_fit_target(const llama_model *) { return true; }
static bool model_has_recurrent_state(const llama_model *) { return false; }
static int llama_model_n_embd(const llama_model *) { return 1024; }
static int draft_target_hidden_size(const llama_model *, const std::string &) { return 1024; }
static bool validate_ngram_cache_file(const std::string &, const char *) { return true; }
static int llama_n_rs_seq(int) { return 16; }
static int llama_n_ctx(int) { return 256; }
static std::string escape_json(const std::string & value) { return value; }
static llama_dflash_variant llama_model_dflash_variant(const llama_model * model) { return model->variant; }
static int llama_model_meta_val_str(const llama_model *, const char *, char * out, size_t size) {
  assert(std::strlen(architecture) < size);
  std::strcpy(out, architecture);
  return std::strlen(out);
}
static llama_model * llamadart_webgpu_draft_model_load_file(const char * path, int, std::string *) {
  ++draft_allocations;
  return new llama_model{static_cast<llama_dflash_variant>(std::atoi(path))};
}
static void llama_model_free(llama_model * model) { --draft_allocations; delete model; }
static std::vector<common_speculative_type> common_speculative_types_from_gguf(const char *) {
  return {COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
}
static std::string common_speculative_type_to_str(common_speculative_type type) {
  switch (type) {
    case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: return "draft-dflash";
    case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: return "draft-dspark";
    case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: return "draft-eagle3";
    case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: return "draft-simple";
    default: return "none";
  }
}
static std::string g_completion_capabilities_json;
`,
  state,
  body('void free_draft_model() {'),
  body('bool is_draft_model_type('),
  body('bool needs_draft_context('),
  body('bool validate_speculative_request('),
  body('EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_draft_model_load('),
  body('EMSCRIPTEN_KEEPALIVE const char * llamadart_webgpu_completion_capabilities_json('),
  String.raw`
int main(int argc, const char * argv[]) {
  assert(argc == 3);
  architecture = argv[2];
  const auto rc = llamadart_webgpu_draft_model_load(argv[1]);
  std::cout << rc << '\n' << error << '\n' << g_draft_model_json << '\n'
            << llamadart_webgpu_completion_capabilities_json() << '\n';
  for (const auto type : {COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK}) {
    speculative_request request{};
    request.draft_n_max = 7;
    request.types = {type};
    clear_error();
    const auto valid = validate_speculative_request(request, false, false);
    std::cout << (valid ? "accepted" : error) << '\n';
  }
  if (rc != 0) assert(draft_allocations == 0);
  free_draft_model();
  assert(draft_allocations == 0);
  speculative_request unloaded{};
  unloaded.draft_n_max = 7;
  unloaded.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK};
  assert(!validate_speculative_request(unloaded, false, false));
  assert(error == "draft-dspark needs a draft model: call loadDraftModel first");
}
`,
].join('\n');

test('loaded draft variants determine the bridge strategy, capabilities and request validation', () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'b1-draft-variant-'));
  try {
    const cpp = path.join(dir, 'fixture.cpp');
    const bin = path.join(dir, 'fixture');
    writeFileSync(cpp, fixture);
    execFileSync(process.env.CXX || 'c++', ['-std=c++17', cpp, '-o', bin], { stdio: 'pipe' });
    for (const [variant, architecture, strategy, unsupported] of [
      ['0', 'dflash', 'draft-dflash', null],
      ['2', 'dflash', 'draft-dspark', null],
      ['1', 'dflash', null, 'DFlash2'],
      ['3', 'dflash', null, 'LiliCorr'],
      ['-1', 'dflash', null, 'Unknown DFlash variant (-1)'],
      ['42', 'dflash', null, 'Unknown DFlash variant (42)'],
      ['-1', 'llama', null, null],
    ]) {
      const [rc, loadError, info, capabilities, dflash, dspark] =
        execFileSync(bin, [variant, architecture], { encoding: 'utf8' }).split('\n');
      assert.equal(Number(rc), unsupported ? -6 : 0, variant);
      assert.equal(loadError, unsupported ? unsupported + ' is unsupported by this bridge version' : '');
      assert.deepEqual(JSON.parse(info), unsupported ? {} : {
        architecture, ...(strategy ? { strategy } : {}),
      });
      const flags = JSON.parse(capabilities).speculativeDecoding;
      assert.equal(flags['draft-dflash'], strategy === 'draft-dflash');
      assert.equal(flags['draft-dspark'], strategy === 'draft-dspark');
      for (const [name, result] of [['draft-dflash', dflash], ['draft-dspark', dspark]]) {
        if (unsupported) assert.equal(result, loadError);
        else if (name === strategy) assert.equal(result, 'accepted');
        else assert.match(result, new RegExp('^' + name + ' needs a ' + name.slice(6) + ' draft model'));
      }
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
