import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const temporaryDir = mkdtempSync(path.join(tmpdir(), 'llama-web-bridge-next-token-scores-'));
const sourcePath = path.join(temporaryDir, 'next_token_scores_contract_test.cpp');
const executablePath = path.join(temporaryDir, 'next_token_scores_contract_test');

const source = String.raw`
#include <iostream>
#include <limits>
#include <vector>

#include "src/llama_webgpu_next_token_scores.h"

using namespace llamadart_webgpu_detail;

int main() {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> logits = {2.0f, 1.0f, nan, 2.0f, -inf};
  const int32_t n = static_cast<int32_t>(logits.size());
  const double log_sum = log_sum_exp(logits.data(), n);

  std::vector<ScoredToken> candidates;
  for (const int32_t token : {4, 1, 2}) {
    candidates.push_back({token, token == 1 ? "\xE2\x82" : "", rank_logit(logits[token]) - log_sum});
  }
  std::vector<ScoredToken> top;
  for (const int32_t token : top_token_ids(logits.data(), n, 3)) {
    top.push_back({token, "a\"", rank_logit(logits[token]) - log_sum});
  }

  const std::vector<float> dead = {-inf, nan};
  const std::vector<float> infinite = {inf, 0.0f};
  const double infinite_sum = log_sum_exp(infinite.data(), 2);
  std::cout << serialize_next_token_scores_json(candidates, top, 7) << "\n"
            << top_token_ids(logits.data(), n, 0).size() << " "
            << top_token_ids(logits.data(), n, 99).size() << " "
            << json_logprob(log_sum_exp(dead.data(), 2)) << " "
            << json_logprob(rank_logit(infinite[0]) - infinite_sum) << " "
            << json_logprob(rank_logit(infinite[1]) - infinite_sum) << "\n";
  return 0;
}
`;

try {
  writeFileSync(sourcePath, source);
  execFileSync(process.env.CXX || 'c++', [
    '-std=c++17',
    '-Wall',
    '-Wextra',
    '-Werror',
    '-I',
    rootDir,
    sourcePath,
    '-o',
    executablePath,
  ], { stdio: 'inherit' });

  const [json, counts] = execFileSync(executablePath, { encoding: 'utf8' }).trim().split('\n');
  const scores = JSON.parse(json);
  const logSum = Math.log(2 * Math.exp(2) + Math.exp(1));

  assert.deepEqual(scores.candidates.map((entry) => entry.token), [4, 1, 2], 'candidates keep request order');
  assert.equal(scores.candidates[0].logprob, null, 'a -inf logit serializes as null');
  assert.equal(scores.candidates[2].logprob, null, 'a NaN logit serializes as null');
  assert.ok(Math.abs(scores.candidates[1].logprob - (1 - logSum)) < 1e-12);
  assert.deepEqual(scores.candidates[1].bytes, [0xe2, 0x82], 'bytes are raw and unsigned');

  assert.deepEqual(scores.top.map((entry) => entry.token), [0, 3, 1], 'ties rank the lower id first');
  assert.deepEqual(scores.top[0].bytes, [0x61, 0x22], 'bytes carry quotes without escaping');
  const total = scores.top.reduce((sum, entry) => sum + Math.exp(entry.logprob), 0);
  assert.ok(Math.abs(total - 1) < 1e-12, `finite probabilities sum to 1, got ${total}`);

  assert.equal(scores.promptTokens, 7);
  assert.equal(
    counts,
    '0 5 null null null',
    'topK clamps to the vocabulary; all-dead and infinite logits give null',
  );
} finally {
  rmSync(temporaryDir, { recursive: true, force: true });
}

const scoringCalls = (calls) =>
  calls.filter(([name]) => name === 'llamadart_webgpu_score_next_token_to_json');

function createScoringBridge({ rc = 0, lastError = '', json = '{}' } = {}) {
  const bridge = new LlamaWebGpuBridge({ disableWorker: true });
  const calls = [];
  bridge._runtime._core = {
    ccall(name, _returnType, _argTypes, args = []) {
      calls.push([name, ...args]);
      switch (name) {
        case 'llamadart_webgpu_score_next_token_to_json':
          return Promise.resolve(rc);
        case 'llamadart_webgpu_last_next_token_scores_json':
          return json;
        case 'llamadart_webgpu_last_error':
          return lastError;
        case 'llamadart_webgpu_model_meta_json':
          return '{}';
        case 'llamadart_webgpu_get_context_size':
          return 128;
        case 'llamadart_webgpu_media_clear_pending':
        case 'llamadart_webgpu_mmproj_free':
        case 'llamadart_webgpu_shutdown':
          return 0;
        default:
          throw new Error(`Unexpected ccall: ${name}`);
      }
    },
  };
  bridge._runtime._modelBytes = 1;
  return { bridge, calls };
}

{
  const { bridge, calls } = createScoringBridge({
    json: JSON.stringify({
      candidates: [{ token: 5, bytes: [0xe2, 0x82], logprob: -0.25 }, { token: 6, bytes: [], logprob: null }],
      top: [{ token: 5, bytes: [0x41], logprob: -0.25 }],
      promptTokens: 3,
    }),
  });
  const scores = await bridge.scoreNextToken('hi', { candidates: new Int32Array([5, 6]), topK: 1 });
  assert.deepEqual(scoringCalls(calls), [['llamadart_webgpu_score_next_token_to_json', 'hi', '[5,6]', 1, 1]]);
  assert.ok(scores.candidates[0].bytes instanceof Uint8Array);
  assert.deepEqual(Array.from(scores.candidates[0].bytes), [0xe2, 0x82]);
  assert.equal(scores.candidates[1].logprob, -Infinity, 'null logprob maps to -Infinity');
  assert.deepEqual(scores.top.map((entry) => entry.token), [5]);
  assert.equal(scores.promptTokens, 3);

  calls.length = 0;
  await bridge.scoreNextToken('hi', { topK: 2, reusePromptPrefix: false });
  assert.deepEqual(scoringCalls(calls), [['llamadart_webgpu_score_next_token_to_json', 'hi', '[]', 2, 0]]);

  calls.length = 0;
  for (const [prompt, options, pattern] of [
    ['hi', { candidates: [1.5] }, /candidates\[0\] is 1\.5; expected a 32-bit integer/],
    ['hi', { candidates: ['1'] }, /candidates\[0\] is 1;/],
    ['hi', { candidates: 1 }, /must be an array or typed array/],
    ['hi', { topK: 2 ** 31 }, /topK is 2147483648/],
    [null, {}, /prompt must be a string/],
  ]) {
    await assert.rejects(bridge.scoreNextToken(prompt, options), pattern);
  }
  assert.equal(scoringCalls(calls).length, 0, 'malformed input never reaches the core');
  await bridge.dispose();
}

{
  const { bridge } = createScoringBridge({
    rc: -3,
    lastError: 'Token id 99 is outside the vocabulary of 32 tokens',
  });
  await assert.rejects(
    bridge.scoreNextToken('hi', { candidates: [99] }),
    /^Error: Next-token scoring failed: Token id 99 is outside the vocabulary of 32 tokens$/,
  );
  await bridge.dispose();
}

{
  const { bridge, calls } = createScoringBridge();
  bridge._runtime._modelBytes = 0;
  await assert.rejects(bridge.scoreNextToken('hi', { topK: 1 }), /No model loaded/);
  assert.equal(scoringCalls(calls).length, 0);
}

console.log('next-token scores contract: ok');
