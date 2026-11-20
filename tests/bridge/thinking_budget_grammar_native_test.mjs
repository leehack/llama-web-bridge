import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { readNativeCoreSource } from './native_core_source.mjs';

const source = readNativeCoreSource();
const body = (signature) => {
  const start = source.indexOf(signature);
  assert.ok(start >= 0, `${signature} not found in the native core`);
  return source.slice(start, source.indexOf('\n}\n', start) + 2);
};
const isActive = body('bool grammar_is_active() {');
const begin = body('int32_t begin_generation_impl(');
const end = body('void end_generation_state() {');
const replayAssignment = begin.indexOf('grammar_replay_end = reasoning_budget.end_tokens[');
assert.ok(replayAssignment >= 0, 'the request must bind grammar replay before generation');
const configureReplay = begin.slice(
  begin.lastIndexOf('if (', replayAssignment),
  begin.indexOf('\n        }', replayAssignment) + '\n        }'.length,
);

const adapter = source.slice(
  source.indexOf('struct multi_end_reasoning_budget {'),
  source.indexOf('// Takes ownership of reasoning_budget,'),
);
const replay = body('bool replay_natural_reasoning_end(');
const sample = body('llama_token sample_next_token(');
const promptConfig = source.slice(
  source.indexOf('struct reasoning_budget_config {'),
  source.indexOf('struct reasoning_budget_ends {'),
);
const at = body('bool token_sequence_at(');
const initialState = body('common_reasoning_budget_state reasoning_budget_initial_state(');

// A deterministic stock sampler double includes v1.0.0's early exit in
// FORCING. The adapter and replay code under test are the production bodies.
const fixture = `
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>
enum common_reasoning_budget_state {
  REASONING_BUDGET_IDLE,
  REASONING_BUDGET_COUNTING,
  REASONING_BUDGET_FORCING,
  REASONING_BUDGET_WAITING_UTF8,
  REASONING_BUDGET_DONE,
};
struct llama_vocab {};
using llama_token = int32_t;
using llama_tokens = std::vector<llama_token>;
struct llama_token_data { llama_token id; float logit; };
struct llama_token_data_array {
  llama_token_data * data;
  int selected;
  llama_token forced = -1;
};
constexpr llama_token LLAMA_TOKEN_NULL = -1;
struct llama_sampler;
struct llama_sampler_i {
  const char * (*name)(const llama_sampler *);
  void (*accept)(llama_sampler *, llama_token);
  void (*apply)(llama_sampler *, llama_token_data_array *);
  void (*reset)(llama_sampler *);
  llama_sampler * (*clone)(const llama_sampler *);
  void (*free)(llama_sampler *);
};
struct llama_sampler {
  llama_sampler_i * iface;
  void * ctx;
};
llama_sampler * llama_sampler_init(llama_sampler_i * iface, void * ctx) {
  return new llama_sampler{iface, ctx};
}
void llama_sampler_free(llama_sampler * sampler) {
  if (sampler) {
    sampler->iface->free(sampler);
    delete sampler;
  }
}
struct sampler_deleter {
  void operator()(llama_sampler * p) const { llama_sampler_free(p); }
};
using llama_sampler_ptr = std::unique_ptr<llama_sampler, sampler_deleter>;
void llama_sampler_apply(llama_sampler * sampler, llama_token_data_array * candidates) {
  sampler->iface->apply(sampler, candidates);
}
void llama_sampler_accept(llama_sampler * sampler, llama_token token) {
  sampler->iface->accept(sampler, token);
}
llama_sampler * llama_sampler_clone(const llama_sampler * sampler) {
  return sampler->iface->clone(sampler);
}
struct stock_sampler {
  std::vector<llama_tokens> starts, ends;
  llama_tokens forced;
  int remaining;
  common_reasoning_budget_state state;
  size_t cursor = 0;
  llama_tokens accepted;
  const llama_tokens * matched = nullptr;

  void accept(llama_token token) {
    accepted.push_back(token);
    auto suffix = [&](const llama_tokens & seq) {
      return seq.size() <= accepted.size() && std::equal(seq.rbegin(), seq.rend(), accepted.rbegin());
    };
    if (state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE) {
      if (std::any_of(starts.begin(), starts.end(), suffix)) {
        state = remaining > 0 ? REASONING_BUDGET_COUNTING : REASONING_BUDGET_FORCING;
      }
      return;
    }
    const auto end = std::find_if(ends.begin(), ends.end(), suffix);
    if (end != ends.end()) {
      state = REASONING_BUDGET_DONE;
      matched = &*end;
    } else if (state == REASONING_BUDGET_FORCING) {
      if (++cursor == forced.size()) {
        state = REASONING_BUDGET_DONE;
      }
    } else if (--remaining == 0) {
      state = REASONING_BUDGET_FORCING;
    }
  }
};
stock_sampler & stock(const llama_sampler * sampler) {
  return *static_cast<stock_sampler *>(sampler->ctx);
}
llama_sampler_i stock_interface = {
  nullptr,
  [](llama_sampler * sampler, llama_token token) { stock(sampler).accept(token); },
  [](llama_sampler * sampler, llama_token_data_array * candidates) {
    const auto & ctx = stock(sampler);
    if (ctx.state == REASONING_BUDGET_FORCING) candidates->forced = ctx.forced[ctx.cursor];
  },
  nullptr,
  [](const llama_sampler * sampler) {
    return llama_sampler_init(sampler->iface, new stock_sampler(stock(sampler)));
  },
  [](llama_sampler * sampler) { delete &stock(sampler); },
};
llama_sampler * common_reasoning_budget_init(
    const llama_vocab *, const std::vector<llama_tokens> & starts,
    const std::vector<llama_tokens> & ends, const llama_tokens & forced,
    int32_t max_tokens, common_reasoning_budget_state state) {
  if (state == REASONING_BUDGET_COUNTING && max_tokens == 0) state = REASONING_BUDGET_FORCING;
  return llama_sampler_init(&stock_interface, new stock_sampler{starts, ends, forced, max_tokens, state});
}
common_reasoning_budget_state common_reasoning_budget_get_state(const llama_sampler * sampler) {
  return stock(sampler).state;
}
const llama_tokens * common_reasoning_budget_get_end_match(const llama_sampler * sampler) {
  return stock(sampler).matched;
}
llama_sampler * g_active_grammar = nullptr;
llama_sampler * g_active_sampler = nullptr;
llama_sampler * g_active_reasoning_budget = nullptr;
bool g_active_grammar_pauses_while_reasoning = true;
bool g_active_grammar_lazy = false;
llama_tokens g_active_grammar_replay_end = {51, 52};
llama_tokens replayed;
bool grammar_accepts_token(llama_token) { return true; }
bool accept_generation_grammar(llama_token token) {
  replayed.push_back(token);
  return true;
}
void set_error(const std::string &) { assert(false); }
llama_token_data next_candidate;
bool apply_generation_samplers(bool, llama_token_data_array & out, int32_t) {
  out.data = &next_candidate;
  out.selected = 0;
  return true;
}
${adapter}
${isActive}
${replay}
${sample}
${promptConfig}
${at}
${initialState}
llama_tokens configure_grammar_replay(
    const llama_sampler * grammar_sampler,
    bool pause,
    const reasoning_budget_config & reasoning_budget,
    int32_t replay_index) {
  const struct { bool pause_while_reasoning; } grammar_options{pause};
  const struct { int32_t grammar_replay_index; } natural_ends{replay_index};
  llama_tokens grammar_replay_end;
  ${configureReplay}
  return grammar_replay_end;
}
int main() {
  const reasoning_budget_config config{0, {11}, {{70, 71}, {71}, {51, 52}}, {71}};
  assert(reasoning_budget_initial_state(config, {11, 51, 52, 99}) == REASONING_BUDGET_IDLE);
  assert(reasoning_budget_initial_state(config, {11, 51, 43}) == REASONING_BUDGET_COUNTING);
  assert(reasoning_budget_initial_state(config, {11, 70, 71, 11}) == REASONING_BUDGET_COUNTING);
  assert(reasoning_budget_initial_state(config, {11, 71}) == REASONING_BUDGET_IDLE);

  auto make_sampler = [](int limit, common_reasoning_budget_state state, const llama_tokens & forced) {
    auto * budget = new multi_end_reasoning_budget{nullptr, {11}, {{70, 71}, {71}, {51, 52}}, forced, limit, nullptr};
    budget->set_natural_child(state);
    return llama_sampler_ptr(llama_sampler_init(&multi_end_reasoning_budget_interface, budget));
  };
  auto accept = [](llama_sampler * sampler, llama_token token) {
    g_active_reasoning_budget = sampler;
    g_active_sampler = sampler;
    next_candidate = {token, 1.0f};
    assert(sample_next_token() == token);
  };
  auto next_forced = [](llama_sampler * sampler) {
    llama_token_data_array candidates{};
    llama_sampler_apply(sampler, &candidates);
    return candidates.forced;
  };
  auto check_forced = [&](llama_sampler * sampler, const llama_tokens & tokens) {
    for (const auto token : tokens) {
      assert(next_forced(sampler) == token);
      assert(next_forced(sampler) == token);
      accept(sampler, token);
    }
    assert(next_forced(sampler) == -1);
    assert(common_reasoning_budget_get_state(active_reasoning_budget_child()) == REASONING_BUDGET_DONE);
    assert(common_reasoning_budget_get_end_match(active_reasoning_budget_child()) == nullptr);
    assert(replayed.empty());
  };

  auto grammar = llama_sampler_ptr(llama_sampler_init(nullptr, nullptr));
  g_active_grammar = grammar.get();
  auto zero = make_sampler(0, REASONING_BUDGET_COUNTING, {51, 52, 70, 71});
  check_forced(zero.get(), {51, 52, 70, 71});
  const llama_tokens embedded = {71, 11, 51, 52, 70, 71};
  auto positive = make_sampler(1, REASONING_BUDGET_COUNTING, embedded);
  accept(positive.get(), 42);
  accept(positive.get(), 71);
  auto clone = llama_sampler_ptr(llama_sampler_clone(positive.get()));
  check_forced(positive.get(), {11, 51, 52, 70, 71});
  check_forced(clone.get(), {11, 51, 52, 70, 71});

  g_active_grammar = nullptr;
  accept(zero.get(), 11);
  assert(next_forced(zero.get()) == 51);
  zero->iface->reset(zero.get());
  assert(next_forced(zero.get()) == -1);
  accept(zero.get(), 11);
  assert(next_forced(zero.get()) == 51);

  g_active_grammar = grammar.get();
  auto natural = make_sampler(2, REASONING_BUDGET_COUNTING, embedded);
  accept(natural.get(), 51);
  assert(replayed.empty());
  accept(natural.get(), 52);
  assert((replayed == llama_tokens{51, 52}));
  accept(natural.get(), 43);
  assert((replayed == llama_tokens{51, 52, 43}));

  replayed.clear();
  // A normal close completes the budget while replay is reserved for the opener.
  auto normal = make_sampler(2, REASONING_BUDGET_COUNTING, embedded);
  accept(normal.get(), 70);
  accept(normal.get(), 71);
  assert(common_reasoning_budget_get_state(active_reasoning_budget_child()) == REASONING_BUDGET_DONE);
  assert(replayed.empty());

  // The request can name an eligible end without attaching a grammar.
  g_active_grammar = nullptr;
  g_active_grammar_replay_end = configure_grammar_replay(nullptr, true, config, 2);
  assert(g_active_grammar_replay_end.empty());
  auto no_grammar = make_sampler(2, REASONING_BUDGET_COUNTING, embedded);
  accept(no_grammar.get(), 51);
  accept(no_grammar.get(), 52);
  assert(common_reasoning_budget_get_state(active_reasoning_budget_child()) == REASONING_BUDGET_DONE);
  assert(replayed.empty());
  g_active_grammar = grammar.get();
  g_active_grammar_replay_end = configure_grammar_replay(grammar.get(), true, config, 2);
  assert((g_active_grammar_replay_end == llama_tokens{51, 52}));
  assert(configure_grammar_replay(grammar.get(), true, config, -1).empty());

  auto partial = make_sampler(2, REASONING_BUDGET_COUNTING, embedded);
  accept(partial.get(), 51);
  accept(partial.get(), 43);
  assert(next_forced(partial.get()) == 71);
  assert(replayed.empty());

  g_active_grammar_pauses_while_reasoning = false;
  auto active = make_sampler(2, REASONING_BUDGET_COUNTING, embedded);
  accept(active.get(), 51);
  accept(active.get(), 52);
  assert((replayed == llama_tokens{51, 52}));
  g_active_grammar = nullptr;
  delete grammar.release();
}
`;
const directory = mkdtempSync(path.join(tmpdir(), 'bridge-thinking-budget-grammar-'));
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

// A request sets the flag for its own generation only.
assert.match(begin, /g_active_grammar_pauses_while_reasoning = grammar_options\.pause_while_reasoning;/);
assert.match(end, /g_active_grammar_pauses_while_reasoning = true;/);
assert.match(end, /g_active_grammar_replay_end\.clear\(\);/);

const cmake = readFileSync(new URL('../../CMakeLists.txt', import.meta.url), 'utf8');
const exported = cmake.match(/-sEXPORTED_FUNCTIONS=\[([^\]]*)\]/)[1];
assert.ok(
  exported.includes("'_llamadart_webgpu_set_next_grammar_reasoning_pause'"),
  'llamadart_webgpu_set_next_grammar_reasoning_pause is not in EXPORTED_FUNCTIONS',
);
assert.ok(
  exported.includes("'_llamadart_webgpu_begin_generation_multi_end_bytes'"),
  'llamadart_webgpu_begin_generation_multi_end_bytes is not in EXPORTED_FUNCTIONS',
);

// A grammar that rejects the budget's forced token leaves every logit at -inf
// after the grammar-first resample; the sampler must fail instead of letting
// the budget finish the block without its end tag.
assert.match(sample, /cur_p\.data\[cur_p\.selected\]\.logit == -INFINITY/);
assert.ok(
  sample.indexOf('.logit == -INFINITY') < sample.indexOf('Grammar rejected every candidate token'),
  'the -inf pick is rejected with the grammar-rejection error',
);
