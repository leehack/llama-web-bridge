import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { readNativeCoreSource } from './native_core_source.mjs';

const source = readNativeCoreSource();
const body = (signature) => {
  const start = source.indexOf(signature);
  assert.ok(start >= 0);
  return source.slice(start, source.indexOf('\n}\n', start) + 2);
};
const sampler = body('llama_sampler * create_sampler(');
const begin = body('int32_t begin_generation_impl(');
const gate = begin.match(/  if \(\(grammar != nullptr[^]*?penalty_prompt_tokens\.clear\(\);\n  }/)?.[0];
const validation = begin.match(/  if \(prompt_penalty_mode < 0[^]*?return -2;\n  }/)?.[0];
assert.ok(gate && validation, 'actual production mode selection and validation must be compiled');
const fixture = `
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>
struct llama_sampler {
  int kind;
  std::vector<int> accepted;
  std::vector<llama_sampler *> children;
  ~llama_sampler() { for (auto * child : children) delete child; }
};
using llama_token = int;
using llama_sampler_ptr = std::unique_ptr<llama_sampler>;
struct llama_sampler_chain_params {};
struct { void * vocab = nullptr; } g_state;
std::vector<int> g_pending_media;
void set_error(const char *) {}
llama_sampler_chain_params llama_sampler_chain_default_params() { return {}; }
llama_sampler * llama_sampler_chain_init(llama_sampler_chain_params) { return new llama_sampler{0, {}, {}}; }
int llama_vocab_n_tokens(void *) { return 100; }
llama_sampler * llama_sampler_init_penalties(int, int, float, float, float) { return new llama_sampler{1, {}, {}}; }
void llama_sampler_accept(llama_sampler * s, int token) { s->accepted.push_back(token); }
void llama_sampler_chain_add(llama_sampler * s, llama_sampler * child) { s->children.push_back(child); }
llama_sampler * llama_sampler_init_top_k(int) { return new llama_sampler{3, {}, {}}; }
llama_sampler * llama_sampler_init_top_p(float, int) { return new llama_sampler{4, {}, {}}; }
llama_sampler * llama_sampler_init_min_p(float, int) { return new llama_sampler{5, {}, {}}; }
llama_sampler * llama_sampler_init_temp(float) { return new llama_sampler{6, {}, {}}; }
llama_sampler * llama_sampler_init_dist(uint32_t) { return new llama_sampler{7, {}, {}}; }
${sampler}
int validate(const char * grammar, int prompt_penalty_mode) {
${validation}
  return 0;
}
void check(const char * grammar, int prompt_penalty_mode, bool seeds) {
  std::vector<int> penalty_prompt_tokens{7, 42, 99};
${gate}
  llama_sampler grammar_sampler{8, {}, {}};
  llama_sampler_ptr chain(create_sampler(0.8f, 40, 0.95f, 0.1f, 1.1f, 0.5f, 1,
      new llama_sampler{2, {}, {}}, penalty_prompt_tokens));
  bool saw_penalties = false, saw_reasoning = false;
  for (auto * child : chain->children) {
    if (child->kind == 1) {
      saw_penalties = true;
      assert(child->accepted == (seeds ? std::vector<int>{7,42,99} : std::vector<int>{}));
    } else {
      saw_reasoning |= child->kind == 2;
      assert(child->accepted.empty());
    }
  }
  assert(saw_penalties && saw_reasoning && grammar_sampler.accepted.empty());
}
int main() {
  check(nullptr, 0, true);
  check(nullptr, 1, false);
  check("root", 0, false);
  check("", 0, false);
  check("root", 1, false);
  check("root", 2, true);
  check("", 2, true);
  assert(validate("root", -1) == -2);
  assert(validate("root", 3) == -2);
  assert(validate(nullptr, 2) == -2);
  assert(validate("root", 0) == 0 && validate("root", 1) == 0 && validate("root", 2) == 0);
  g_pending_media.push_back(1);
  assert(validate("root", 2) == -2);
}
`;
const directory = mkdtempSync(path.join(tmpdir(), 'bridge-grammar-penalties-'));
try {
  const cpp = path.join(directory, 'contract.cpp');
  const executable = path.join(directory, 'contract');
  writeFileSync(cpp, fixture);
  const compiled = spawnSync(process.env.CXX || 'c++', ['-std=c++17', cpp, '-o', executable], { encoding: 'utf8' });
  assert.equal(compiled.status, 0, compiled.stderr);
  const ran = spawnSync(executable, [], { encoding: 'utf8' });
  assert.equal(ran.status, 0, `${ran.signal}: ${ran.stderr}`);
} finally {
  rmSync(directory, { recursive: true, force: true });
}
