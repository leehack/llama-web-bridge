import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { readNativeCoreSource } from './native_core_source.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const dir = mkdtempSync(path.join(tmpdir(), 'web-batch-compat-'));
const core = readNativeCoreSource();
assert.match(core, /llama_webgpu_speculative_process\(session.spec, session.batch\)/);
assert.doesNotMatch(core, /common_batch_(add|clear)\(/);
const cmake = readFileSync(path.join(root, 'CMakeLists.txt'), 'utf8');
assert.match(cmake, /unset\(LLAMADART_WEBGPU_SPECULATIVE_HAS_COMMON_BATCH CACHE\)/);
assert.match(cmake, /LLAMADART_WEBGPU_SPECULATIVE_HAS_COMMON_BATCH=\$<BOOL:/);

try {
  writeFileSync(path.join(dir, 'llama.h'), String.raw`
#pragma once
#include <cstdint>
#include <stdexcept>
#define GGML_ASSERT(condition) do { if (!(condition)) throw std::runtime_error("capacity exceeded"); } while (0)
using llama_token = int32_t;
using llama_pos = int32_t;
using llama_seq_id = int32_t;
struct llama_batch {
 int32_t n_tokens;
 llama_token *token;
 float *embd;
 llama_pos *pos;
 int32_t *n_seq_id;
 llama_seq_id **seq_id;
 int8_t *logits;
};
`);
  writeFileSync(path.join(dir, 'speculative.h'), String.raw`
#pragma once
#include "llama.h"
#include <vector>
struct common_speculative {};
inline bool process_result = true;
inline int process_calls = 0;
#if LLAMADART_WEBGPU_SPECULATIVE_HAS_COMMON_BATCH
struct common_batch {
 struct token { llama_token id; llama_pos pos; llama_seq_id seq; bool output; };
 std::vector<token> tokens;
 void add(llama_token id, llama_pos pos, llama_seq_id seq, bool output) {
  tokens.push_back({id, pos, seq, output});
 }
};
inline std::vector<common_batch::token> captured;
inline bool common_speculative_process(common_speculative *, const common_batch &b) {
 ++process_calls; captured = b.tokens; return process_result;
}
#else
inline llama_batch captured{};
inline bool common_speculative_process(common_speculative *, llama_batch b) {
 ++process_calls; captured = b; return process_result;
}
#endif
`);
  writeFileSync(path.join(dir, 'test.cpp'), String.raw`
#include <cassert>
#include "src/llama_webgpu_batch_compat.h"
int main() {
 llama_token tokens[3]{}; llama_pos positions[3]{};
 int32_t counts[3]{}; llama_seq_id ids[3]{};
 llama_seq_id *sequences[]{&ids[0], &ids[1], nullptr};
 int8_t outputs[3]{};
 llama_batch b{0,tokens,nullptr,positions,counts,sequences,outputs};
 common_speculative spec;
 llama_webgpu_batch_add(b, 7, 40, false);
 llama_webgpu_batch_add(b, 9, 41, true);
 assert(b.n_tokens==2 && tokens[0]==7 && tokens[1]==9);
 assert(positions[0]==40 && positions[1]==41);
 assert(counts[0]==1 && counts[1]==1 && ids[0]==0 && ids[1]==0);
 assert(outputs[0]==0 && outputs[1]==1);
 tokens[2]=123; positions[2]=456; counts[2]=789; outputs[2]=1;
 bool overflow_rejected=false;
 try { llama_webgpu_batch_add(b, 99, 99, false); }
 catch (const std::runtime_error &) { overflow_rejected=true; }
 assert(overflow_rejected && b.n_tokens==2);
 assert(tokens[2]==123 && positions[2]==456 && counts[2]==789 && outputs[2]==1);
 assert(llama_webgpu_speculative_process(&spec,b) && process_calls==1);
#if LLAMADART_WEBGPU_SPECULATIVE_HAS_COMMON_BATCH
 assert(captured.size()==2 && captured[0].id==7 && captured[1].id==9);
 assert(captured[0].pos==40 && captured[1].pos==41);
 assert(!captured[0].output && captured[1].output);
 assert(captured[0].seq==0 && captured[1].seq==0);
 auto reject = [&](llama_batch invalid) {
  const int previous=process_calls;
  assert(!llama_webgpu_speculative_process(&spec, invalid));
  assert(process_calls==previous);
 };
 auto invalid=b; invalid.n_tokens=-1; reject(invalid);
 invalid=b; invalid.token=nullptr; reject(invalid);
 invalid=b; invalid.pos=nullptr; reject(invalid);
 invalid=b; invalid.n_seq_id=nullptr; reject(invalid);
 invalid=b; invalid.seq_id=nullptr; reject(invalid);
 invalid=b; invalid.logits=nullptr; reject(invalid);
 float embedding=0; invalid=b; invalid.embd=&embedding; reject(invalid);
 counts[1]=0; reject(b); counts[1]=2; reject(b); counts[1]=1;
 sequences[1]=nullptr; reject(b); sequences[1]=&ids[1];
 ids[1]=1; reject(b); ids[1]=0;
#else
 assert(captured.token==tokens && captured.pos==positions && captured.logits==outputs);
#endif
 process_result=false;
 assert(!llama_webgpu_speculative_process(&spec,b));
 llama_webgpu_batch_clear(b);
 assert(b.n_tokens==0);
 process_result=true;
 assert(llama_webgpu_speculative_process(&spec,b));
#if LLAMADART_WEBGPU_SPECULATIVE_HAS_COMMON_BATCH
 assert(captured.empty());
#endif
 llama_webgpu_batch_add(b, 11, 42, false);
 assert(b.n_tokens==1 && tokens[0]==11 && positions[0]==42 && outputs[0]==0);
}
`);
  for (const mode of [0, 1]) {
    const executable = path.join(dir, `test-${mode}`);
    execFileSync(process.env.CXX || 'c++', ['-std=c++17', '-Wall', '-Wextra', '-Werror',
      `-DLLAMADART_WEBGPU_SPECULATIVE_HAS_COMMON_BATCH=${mode}`, '-I', dir,
      '-I', root, path.join(dir, 'test.cpp'), '-o', executable], { stdio: 'inherit' });
    execFileSync(executable, [], { stdio: 'inherit' });
  }
} finally {
  rmSync(dir, { recursive: true, force: true });
}
