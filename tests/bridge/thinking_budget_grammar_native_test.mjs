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

// The production predicate, compiled against llama.cpp's reasoning-budget
// states, decides per state whether the grammar is applied.
const fixture = `
#include <cassert>
#include <initializer_list>
enum common_reasoning_budget_state {
  REASONING_BUDGET_IDLE,
  REASONING_BUDGET_COUNTING,
  REASONING_BUDGET_FORCING,
  REASONING_BUDGET_WAITING_UTF8,
  REASONING_BUDGET_DONE,
};
struct llama_sampler { common_reasoning_budget_state state; };
common_reasoning_budget_state common_reasoning_budget_get_state(const llama_sampler * sampler) {
  return sampler->state;
}
llama_sampler * g_active_grammar = nullptr;
llama_sampler * g_active_reasoning_budget = nullptr;
bool g_active_grammar_pauses_while_reasoning = true;
${isActive}
int main() {
  llama_sampler grammar{REASONING_BUDGET_IDLE};
  llama_sampler budget{REASONING_BUDGET_IDLE};
  const common_reasoning_budget_state states[] = {
      REASONING_BUDGET_IDLE, REASONING_BUDGET_COUNTING, REASONING_BUDGET_FORCING,
      REASONING_BUDGET_WAITING_UTF8, REASONING_BUDGET_DONE};
  for (const bool pauses : {true, false}) {
    g_active_grammar_pauses_while_reasoning = pauses;
    g_active_grammar = nullptr;
    g_active_reasoning_budget = &budget;
    assert(!grammar_is_active());
    g_active_grammar = &grammar;
    g_active_reasoning_budget = nullptr;
    assert(grammar_is_active());
    g_active_reasoning_budget = &budget;
    for (const common_reasoning_budget_state state : states) {
      budget.state = state;
      const bool outside = state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
      assert(grammar_is_active() == (outside || !pauses));
    }
  }
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

const cmake = readFileSync(new URL('../../CMakeLists.txt', import.meta.url), 'utf8');
const exported = cmake.match(/-sEXPORTED_FUNCTIONS=\[([^\]]*)\]/)[1];
assert.ok(
  exported.includes("'_llamadart_webgpu_set_next_grammar_reasoning_pause'"),
  'llamadart_webgpu_set_next_grammar_reasoning_pause is not in EXPORTED_FUNCTIONS',
);

// A grammar that rejects the budget's forced token leaves every logit at -inf
// after the grammar-first resample; the sampler must fail instead of letting
// the budget finish the block without its end tag.
const sample = body('llama_token sample_next_token(');
assert.match(sample, /cur_p\.data\[cur_p\.selected\]\.logit == -INFINITY/);
assert.ok(
  sample.indexOf('.logit == -INFINITY') < sample.indexOf('Grammar rejected every candidate token'),
  'the -inf pick is rejected with the grammar-rejection error',
);
