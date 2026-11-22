#!/usr/bin/env node
// Browser smoke for grammar-constrained completion and sampler options.
//
// Ported from grammar_browser_smoke.py with the same flags, environment
// variables, harness page and output.

import fsp from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import {
  DEFAULT_MODEL_CACHE,
  copyBridgeArtifacts,
  copyMemory64Artifacts,
  ensure,
  env,
  expectedModes,
  isDict,
  isDirectory,
  modeOf,
  modeRuntimeFailures,
  parseSmokeArgs,
  pyEquals,
  pyGet,
  pyJson,
  pyRepr,
  pyStr,
  resolvePath,
  resolvePinnedModel,
  runMain,
  runPlaywright,
  stageFile,
  withServer,
  withTempDir,
  writeStdout,
} from './support.mjs';

const DESCRIPTION = `Browser smoke for grammar-constrained completion and sampler options.

Runs grammar-constrained \`\`createCompletion\`\` calls through the direct and
worker runtimes of the wasm32 and wasm64 cores with a checksum-pinned GGUF and checks that every result is
text the grammar accepts. It covers greedy \`\`topK: 1\`\` decoding, which used to
truncate the candidates to a token the grammar rejects and abort the Wasm core
(issue #115), a sampled \`\`topK: 40\`\` run, top-p-only truncation, and a small
JSON grammar. It first sends invalid grammars, which used to abort the core
from the parser's throw: each must reject with an \`\`(invalid grammar)\`\` error
and leave the runtime usable for the valid cases that follow. The worker
runtime must still own the model afterwards: an abort there used to move the
bridge to the main thread for the rest of the session.

With a thinking budget it checks that a grammar pauses inside the reasoning
block by default; that \`\`pauseGrammar: false\`\` keeps a grammar that takes the
block and its end tag itself applied through the forced end tag; and that a
grammar rejecting the forced end tag fails with the grammar-rejection error
instead of finishing the block without it. Only the Qwen3.5 run reproduces the
original bug (the end-tag-rejecting grammar finishing the block without its end
tag); the tiny-model run passes with or without the fix, so CI must keep the
Qwen3.5 run.

It then checks \`\`minP\`\` and \`\`presencePenalty\`\` with a fixed seed and a shared
prompt. The comparisons start with eight tokens and expand to 20 if the
controls need more tokens to diverge. \`\`minP: 1\`\` keeps only the most probable
token and matches greedy decoding. Plain sampling produces different text,
explicit zeros match the defaults, and a presence penalty changes the greedy
output. After invalid values are rejected, the seeded sample repeats exactly.
\`\`getCompletionCapabilities\`\` reports no option before a model load and
every option after it.

As native llamadart does, the penalties count the prompt's last 64 tokens
unless a grammar is set. The smoke extends the sampler prompt with its most
probable next token until that token is among the prompt's last 64 tokens.
A greedy token under a presence penalty far above any logit gap must then be
the most probable token outside them. It stays the most probable token with a
grammar that accepts any text, with an empty grammar, and with
\`\`penaltyPromptTokens: false\`\`.

Last, it checks \`\`thinkingBudget\`\` with greedy decoding and a prompt that
leaves a reasoning block open. With \`\`maxTokens: 8\`\` the output is the
unbudgeted 8-token output, then the forced end tag. With \`\`maxTokens: 0\`\` it
starts with the forced message and end tag. A grammar pauses inside the block
and constrains the text after the forced end. A budget has no effect when the
prompt closes the block.

Then it checks lazy grammars with greedy decoding. The text of the first two
tokens, after at least three, that an unconstrained completion has not produced
before becomes the trigger of a grammar that accepts
it followed by "yes" or "no". A word, pattern and full-pattern trigger must
each leave the tokens before it unconstrained, then constrain the rest; the
same grammar applied strictly starts with the trigger text instead. A token
trigger on the first greedy token constrains everything after it, and one on
the runner-up token leaves the output unconstrained. A grammar
starting at a rule other than \`\`root\`\` must apply. An invalid trigger
pattern, a token outside the vocabulary, a missing start rule and a grammar
that rejects its own trigger text must each reject, after which the
unconstrained completion repeats exactly.`;

const MODEL_FILENAME = 'grammar-smoke-model.gguf';
export const MAX_THREADS = 4;
// The issue #115 repro prompt. Its ChatML markers are plain text to a model
// without that template, which is fine: the grammar decides the output.
const PROMPT = '<|im_start|>user\nIs the sky blue? Answer yes or no.<|im_end|>\n'
  + '<|im_start|>assistant\n<think>\n\n</think>\n\n';
const YES_NO_GRAMMAR = 'root ::= "yes" | "no"';
const JSON_GRAMMAR = [
  String.raw`root ::= "{" ws "\"answer\"" ws ":" ws answer ws "," ws `
    + String.raw`"\"confidence\"" ws ":" ws digit ws "}"`,
  String.raw`answer ::= "\"yes\"" | "\"no\""`,
  'digit ::= [0-9]',
  'ws ::= " "?',
].join('\n');
const INVALID_GRAMMAR_ERROR = '(invalid grammar)';
// The bridge's text when llama.cpp returns no grammar without throwing.
const NON_THROWING_REJECTION = 'grammar rejected by llama.cpp';
const INVALID_OPTIONS = { nPredict: 8, temp: 0, topK: 1, seed: 1 };
// Each case keeps the Python dict's key order: the harness embeds it as JSON.
export const CASES = Object.freeze([
  // A parser syntax error, thrown and caught in the bridge's grammar wrapper.
  {
    name: 'invalid-unterminated-string',
    grammar: 'root ::= "unterminated',
    kind: 'invalid',
    throws: true,
    options: INVALID_OPTIONS,
  },
  // Thrown by the parser's undefined-rule check after parsing succeeds.
  {
    name: 'invalid-undefined-rule',
    grammar: 'root ::= answer',
    kind: 'invalid',
    throws: true,
    options: INVALID_OPTIONS,
  },
  // Rejected by llama.cpp without a throw.
  {
    name: 'invalid-left-recursion',
    grammar: 'root ::= root "a" | "a"',
    kind: 'invalid',
    throws: false,
    options: INVALID_OPTIONS,
  },
  {
    name: 'yes-no-greedy',
    grammar: YES_NO_GRAMMAR,
    kind: 'yes-no',
    options: { nPredict: 8, temp: 0, topK: 1, seed: 1 },
  },
  {
    name: 'yes-no-sampled',
    grammar: YES_NO_GRAMMAR,
    kind: 'yes-no',
    options: { nPredict: 8, temp: 0.8, topK: 40, seed: 1 },
  },
  {
    name: 'json-greedy',
    grammar: JSON_GRAMMAR,
    kind: 'json',
    options: { nPredict: 64, temp: 0, topK: 1, seed: 1 },
  },
  {
    name: 'json-top-p',
    grammar: JSON_GRAMMAR,
    kind: 'json',
    options: { nPredict: 64, temp: 0.7, topK: 0, topP: 0.1, seed: 7 },
  },
]);
const SAMPLER_PROMPT = 'Here is a long list of animals, colors, and cities:';
export const SAMPLER_SHORT_PREDICT = 8;
const SAMPLED = { nPredict: 20, temp: 1, topK: 0, topP: 1, penalty: 1, seed: 11 };
const GREEDY = { nPredict: 20, temp: 0, topK: 1, penalty: 1, seed: 11 };
// Each run's options; "first" warms the prompt cache and is not compared.
export const SAMPLER_RUNS = Object.freeze([
  ['first', { ...SAMPLED, nPredict: 1 }],
  ['sampled', SAMPLED],
  ['sampled-explicit-zeros', { ...SAMPLED, minP: 0, presencePenalty: 0 }],
  ['greedy', GREEDY],
  ['min-p-one', { ...SAMPLED, minP: 1 }],
  ['greedy-presence', { ...GREEDY, presencePenalty: 20 }],
]);
// The penalty window, a topK above it so some scored token lies outside it, a
// presence penalty larger than any logit gap among those tokens, and the least
// logprob gap a compared pick needs over the next token: CPU logits of a
// re-decoded position drift by up to 1e-3 (next_token_scores.mjs).
export const PROMPT_PENALTY = Object.freeze({
  window: 64,
  topK: 80,
  maxSteps: 32,
  minGap: 0.005,
  presencePenalty: 1000,
  options: { ...GREEDY, nPredict: 1 },
  grammar: String.raw`root ::= [^\x00]*`,
});
const INVALID_SAMPLER_OPTIONS = Object.freeze([
  { minP: 1.5 },
  { minP: -0.1 },
  { presencePenalty: Number.POSITIVE_INFINITY },
]);
// speculative.mjs checks the speculativeDecoding capabilities.
export const COMPLETION_CAPABILITIES = Object.freeze([
  'presencePenalty', 'minP', 'thinkingBudget', 'thinkingBudgetPauseGrammar', 'thinkingBudgetMultiEnd', 'penaltyPromptTokens', 'penaltyPromptTokensWithGrammar', 'lazyGrammar',
]);
// Each tag follows a space: SentencePiece vocabularies tokenize a tag at the
// start of text with a space prefix, so only a spaced tag in the prompt
// matches the tokenized start and end tags.
const THINK_PROMPT = '<|im_start|>user\nWhat is 17 times 23?<|im_end|>\n<|im_start|>assistant\n <think>\n';
const CLOSED_THINK_PROMPT = `${THINK_PROMPT}\n </think>\n\n`;
const THINK_GREEDY = { temp: 0, topK: 1, penalty: 1, seed: 3 };
const THINK_TAGS = { startTag: '<think>', endTag: '</think>' };
const THINK_BUDGET_TOKENS = 8;
const FORCED_MESSAGE = 'Done.';
// Takes the reasoning block and its end tag itself, as one written for a prompt
// that ends inside the block does.
const THINK_BODY_GRAMMAR = String.raw`root ::= think-body "</think>" ("yes" | "no")
think-body ::= ( [^<] | "<" [^/] )*`;
// Each run's prompt and options; each "-first" run warms the prompt cache.
export const THINKING_RUNS = Object.freeze([
  ['open-first', THINK_PROMPT, { ...THINK_GREEDY, nPredict: 1 }],
  ['open-prefix', THINK_PROMPT, { ...THINK_GREEDY, nPredict: THINK_BUDGET_TOKENS }],
  ['open-budget', THINK_PROMPT, {
    ...THINK_GREEDY,
    nPredict: 12,
    thinkingBudget: { ...THINK_TAGS, maxTokens: THINK_BUDGET_TOKENS },
  }],
  ['open-budget-zero', THINK_PROMPT, {
    ...THINK_GREEDY,
    nPredict: 6,
    thinkingBudget: { ...THINK_TAGS, maxTokens: 0, forcedMessage: FORCED_MESSAGE },
  }],
  ['open-budget-grammar', THINK_PROMPT, {
    ...THINK_GREEDY,
    nPredict: 16,
    grammar: YES_NO_GRAMMAR,
    thinkingBudget: { ...THINK_TAGS, maxTokens: THINK_BUDGET_TOKENS },
  }],
  ['open-budget-taking-grammar', THINK_PROMPT, {
    ...THINK_GREEDY,
    nPredict: 24,
    grammar: THINK_BODY_GRAMMAR,
    thinkingBudget: { ...THINK_TAGS, maxTokens: THINK_BUDGET_TOKENS, pauseGrammar: false },
  }],
  ['open-budget-rejecting-grammar', THINK_PROMPT, {
    ...THINK_GREEDY,
    nPredict: 24,
    grammar: String.raw`root ::= [^<]*`,
    thinkingBudget: { ...THINK_TAGS, maxTokens: THINK_BUDGET_TOKENS, pauseGrammar: false },
  }],
  ['open-budget-after-rejecting', THINK_PROMPT, {
    ...THINK_GREEDY,
    nPredict: 16,
    grammar: YES_NO_GRAMMAR,
    thinkingBudget: { ...THINK_TAGS, maxTokens: THINK_BUDGET_TOKENS },
  }],
  ['closed-first', CLOSED_THINK_PROMPT, { ...THINK_GREEDY, nPredict: 1 }],
  ['closed-plain', CLOSED_THINK_PROMPT, { ...THINK_GREEDY, nPredict: 4 }],
  ['closed-budget-zero', CLOSED_THINK_PROMPT, {
    ...THINK_GREEDY,
    nPredict: 4,
    thinkingBudget: { ...THINK_TAGS, maxTokens: 0 },
  }],
]);
const INVALID_THINKING_BUDGET = { ...THINK_TAGS, maxTokens: -1 };
export const INVALID_NATIVE_MULTI_ENDS = Object.freeze([
  { name: 'past-end index', json: '["</think>"]', index: 1, error: 'replay index' },
  { name: 'negative index', json: '["</think>"]', index: -2, error: 'replay index' },
  { name: 'non-string entry', json: '[42]', index: -1, error: 'tags must be strings' },
  { name: 'NUL in entry', json: JSON.stringify(['before\0after']), index: -1, error: 'without NUL bytes' },
  { name: 'nested array', json: '[["</think>"]]', index: -1, error: 'non-empty JSON array' },
  { name: 'empty array', json: '[]', index: -1, error: 'non-empty JSON array' },
  { name: 'null JSON pointer', json: null, index: -1, error: 'non-empty JSON array' },
]);
// Greedy runs whose first tokens are unconstrained text. The first window of
// triggerTokens tokens after at least minBeforeTokens tokens whose text does
// not occur earlier, searched up to maxBeforeTokens, becomes the trigger.
export const LAZY_GRAMMAR = Object.freeze({
  prompt: SAMPLER_PROMPT,
  options: { temp: 0, topK: 1, penalty: 1, seed: 5 },
  minBeforeTokens: 3,
  maxBeforeTokens: 12,
  triggerTokens: 2,
  answerTokens: 6,
  minGap: 0.005,
  answers: ['yes', 'no'],
  rootGrammar: 'root ::= "unused"\nanswer ::= "yes" | "no"',
});
export const LAZY_TRIGGER_RUNS = Object.freeze(['word', 'pattern', 'pattern-full']);
const LAZY_GRAMMAR_ERRORS = Object.freeze({
  invalidPattern: ['(invalid grammar)', 'is not a valid regular expression'],
  outOfVocabulary: ['(invalid grammar)', 'is outside the vocabulary'],
  missingRoot: ['(invalid grammar)', NON_THROWING_REJECTION],
  rejectedTrigger: ['The lazy grammar rejected the text that triggered it'],
});
export const MEMORY_MODES = Object.freeze(['wasm32', 'wasm64']);
const RUNTIME_MODES = Object.freeze(['direct', 'worker']);

// JSON has no Infinity, so the invalid options are written as a JS literal.
const invalidSamplerOptionsLiteral = () => `[${INVALID_SAMPLER_OPTIONS.map((options) => `{ ${
  Object.entries(options).map(([key, value]) => `${key}: ${String(value)}`).join(', ')
} }`).join(', ')}]`;

export function renderHarness(nCtx, memoryModes, runtimeModes = RUNTIME_MODES) {
  const config = pyJson({
    modelUrl: `/${MODEL_FILENAME}`,
    prompt: PROMPT,
    cases: CASES,
    invalidGrammarError: INVALID_GRAMMAR_ERROR,
    nonThrowingRejection: NON_THROWING_REJECTION,
    samplerPrompt: SAMPLER_PROMPT,
    samplerRuns: SAMPLER_RUNS,
    samplerShortPredict: SAMPLER_SHORT_PREDICT,
    thinkingRuns: THINKING_RUNS,
    promptPenalty: PROMPT_PENALTY,
    invalidThinkingBudget: INVALID_THINKING_BUDGET,
    invalidNativeMultiEnds: INVALID_NATIVE_MULTI_ENDS,
    lazyGrammar: LAZY_GRAMMAR,
    lazyTriggerRuns: LAZY_TRIGGER_RUNS,
    completionCapabilities: COMPLETION_CAPABILITIES,
    maxThreads: MAX_THREADS,
    nCtx,
    memoryModes,
    runtimeModes,
  });
  const invalidSamplerOptions = invalidSamplerOptionsLiteral();
  return `
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge grammar smoke</title>
<pre id="result">pending</pre>
<script type="module">
(async () => {
  const result = document.getElementById('result');
  const setStage = (stage) => {
    window.__smokeStage = stage;
    result.textContent = \`pending: \${stage}\`;
    console.log(\`GRAMMAR_STAGE \${stage}\`);
  };
  const finish = (payload) => {
    result.textContent = JSON.stringify(payload);
    window.__smokeResult = payload;
  };
  const assert = (condition, message) => {
    if (!condition) {
      throw new Error(message);
    }
  };
  const errorText = (error) => String(error && error.message ? error.message : error);
  const checkText = (kind, text) => {
    if (kind === 'yes-no') {
      return text === 'yes' || text === 'no';
    }
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (_) {
      return false;
    }
    return parsed !== null
      && typeof parsed === 'object'
      && Object.keys(parsed).join(',') === 'answer,confidence'
      && (parsed.answer === 'yes' || parsed.answer === 'no')
      && Number.isInteger(parsed.confidence)
      && parsed.confidence >= 0
      && parsed.confidence <= 9;
  };
  try {
    if (!window.crossOriginIsolated) {
      throw new Error('test page is not cross-origin isolated');
    }
    setStage('startup:bridge-import');
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    assert(typeof LlamaWebGpuBridge === 'function', 'LlamaWebGpuBridge export was not registered');
    const config = ${config};
    const samplerCapabilities = (capabilities) => Object.fromEntries(
      config.completionCapabilities.map((name) => [name, capabilities[name]]),
    );
    const modeResults = [];

    const createPrefixSnapshotHelper = (bridge) => {
      const snapshotPath = '/tmp/grammar-smoke-prefix.bin';
      const stats = { enabled: false, saves: 0, restores: 0, saveMs: 0, restoreMs: 0, cachedPromptTokens: 0, restoresByPhase: {} };
      let active = null;
      let phase;
      const startsWithTokens = (tokens, prefix) => prefix.every((token, i) => tokens[i] === token);
      const invalidate = () => { active = null; };
      const detect = async () => {
        if (typeof bridge.stateSaveFile !== 'function' || typeof bridge.stateLoadFile !== 'function'
            || typeof bridge.detokenize !== 'function') return;
        let usage;
        for (let i = 0; i < 2; i++) {
          usage = undefined;
          await bridge.createCompletion('a b', {
            nPredict: 1, temp: 0, topK: 1, seed: 1, onUsage: (value) => { usage = value; },
          });
        }
        stats.enabled = usage?.promptTokens > 1 && usage.cachedPromptTokens === 0;
      };
      const saveCurrentContextAs = async (prompt, tokens) => {
        const startedAt = performance.now();
        await bridge.stateSaveFile(snapshotPath, tokens);
        stats.saveMs += performance.now() - startedAt;
        stats.saves += 1;
        active = { prompt, tokens };
      };
      const load = async (run) => {
        const startedAt = performance.now();
        await bridge.stateLoadFile(snapshotPath);
        const elapsed = performance.now() - startedAt;
        stats.restoreMs += elapsed;
        stats.restores += 1;
        (stats.restoresByPhase[phase] ??= []).push({ run, restoreMs: Math.round(elapsed * 1_000) / 1_000 });
      };
      const prepare = async (prompt, run) => {
        if (!stats.enabled) return false;
        if (active?.prompt === prompt) return true;
        // Leaving the final token to decode lets recurrent memory keep the snapshot and still produce logits.
        const prefix = (await bridge.tokenize(prompt, true)).slice(0, -1);
        if (prefix.length === 0) {
          invalidate();
          return false;
        }
        if (active?.tokens.length === prefix.length && startsWithTokens(prefix, active.tokens)) {
          active.prompt = prompt;
          return true;
        }
        const prefixText = await bridge.detokenize(prefix, true);
        const roundTrip = await bridge.tokenize(prefixText, true);
        if (roundTrip.length !== prefix.length || !startsWithTokens(roundTrip, prefix)) {
          invalidate();
          return false;
        }
        if (active && active.tokens.length < prefix.length && startsWithTokens(prefix, active.tokens)) {
          await load(run + ':extend-prefix');
        }
        await bridge.scoreNextToken(prefixText, { topK: 1 });
        await saveCurrentContextAs(prompt, prefix);
        return true;
      };
      const restore = async (prompt, run) => {
        if (!stats.enabled || active?.prompt !== prompt) return false;
        await load(run);
        return true;
      };
      const complete = async (prompt, options, run) => {
        if (!await restore(prompt, run)) return bridge.createCompletion(prompt, options);
        return bridge.createCompletion(prompt, {
          ...options,
          onUsage: (usage) => {
            stats.cachedPromptTokens += usage.cachedPromptTokens ?? 0;
            options.onUsage?.(usage);
          },
        });
      };
      return { detect, prepare, restore, complete, invalidate, saveCurrentContextAs, stats, setPhase: (key) => { phase = key; } };
    };

    // Appends the most probable next token's text, a greedy continuation,
    // until that token is among the prompt's last tokens, then runs the
    // greedy token under a large presence penalty with and without a grammar.
    // A step whose picks are near ties, or whose penalized pick ends
    // generation and so returns no text, is skipped.
    const probePromptPenalty = async (bridge, snap) => {
      const probe = config.promptPenalty;
      const text = (bytes) => new TextDecoder().decode(bytes);
      const usable = (piece) => piece.length > 0 && !piece.includes('\uFFFD');
      const clear = ([first, second]) => second === undefined || first.logprob - second.logprob > probe.minGap;
      let prompt = config.samplerPrompt;
      const rolling = await snap.prepare(prompt, 'score-0');
      if (rolling) await snap.restore(prompt, 'score-0');
      for (let step = 0; step < probe.maxSteps; step++) {
        const { top } = await bridge.scoreNextToken(prompt, { topK: probe.topK });
        const fullTokens = await bridge.tokenize(prompt, true);
        const window = fullTokens.slice(-probe.window);
        let contextMatchesPrompt = true;
        const outside = top.filter((entry) => !window.includes(entry.token));
        const best = text(top[0].bytes);
        const unpenalized = outside.length > 0 ? text(outside[0].bytes) : '';
        if (window.includes(top[0].token) && clear(top) && clear(outside)
            && usable(best) && usable(unpenalized) && best !== unpenalized) {
          const completionOptions = (options) => ({
            ...probe.options,
            ...options,
            tokenEventEncoding: 'text',
          });
          contextMatchesPrompt = false;
          const penalized = await bridge.createCompletion(prompt, completionOptions({ presencePenalty: probe.presencePenalty }));
          if (penalized !== '') {
            await snap.prepare(prompt, 'penalty-comparisons');
            const run = (name, options) => snap.complete(prompt, completionOptions(options), name);
            return {
              steps: step,
              best,
              unpenalized,
              neutral: await run('neutral', {}),
              penalized,
              grammar: await run('grammar', { presencePenalty: probe.presencePenalty, grammar: probe.grammar }),
              emptyGrammar: await run('empty-grammar', { presencePenalty: probe.presencePenalty, grammar: '' }),
              grammarOptIn: await run('grammar-opt-in', { presencePenalty: probe.presencePenalty, grammar: probe.grammar, penaltyPromptTokensWithGrammar: true }),
              emptyGrammarOptIn: await run('empty-grammar-opt-in', { presencePenalty: probe.presencePenalty, grammar: '', penaltyPromptTokensWithGrammar: true }),
              // The trigger appears in the prompt, and its grammar allows only
              // a different suffix after it. Seeding must not fire it.
              promptTriggerOptIn: await run('prompt-trigger-opt-in', {
                presencePenalty: probe.presencePenalty,
                grammar: 'root ::= ' + gbnfLiteral(prompt) + ' "__prompt_trigger_fired__"',
                grammarLazy: true,
                grammarTriggers: [{ type: 'word', value: prompt }],
                penaltyPromptTokensWithGrammar: true,
              }),
              optOut: await run('opt-out', { presencePenalty: probe.presencePenalty, penaltyPromptTokens: false }),
            };
          }
          snap.invalidate();
        }
        if (rolling && contextMatchesPrompt) await snap.saveCurrentContextAs(null, fullTokens);
        prompt += best;
      }
      return { steps: probe.maxSteps, best: null };
    };

    // Every character but ASCII letters, digits and spaces as a \U escape.
    const gbnfLiteral = (text) => '"' + Array.from(text, (ch) => (/[A-Za-z0-9 ]/.test(ch)
      ? ch
      : '\\\\U' + ch.codePointAt(0).toString(16).padStart(8, '0'))).join('') + '"';
    const regexLiteral = (text) => text.replace(/[.*+?^$()|[\\]{}\\\\]/g, '\\\\$&');
    const errorOf = async (promise) => {
      try {
        await promise;
        return null;
      } catch (caught) {
        return errorText(caught);
      }
    };

    // Runs lazy grammars triggered by text the unconstrained greedy
    // completion produces, and the rejections that must leave the core usable.
    const probeLazyGrammar = async (bridge, snap) => {
      const lazy = config.lazyGrammar;
      // Room for the trigger window, an answer and the end of generation.
      let answeredTokens = 0;
      const completionOptions = (options, nPredict = answeredTokens) => ({
        ...lazy.options,
        nPredict,
        ...options,
        tokenEventEncoding: 'text',
      });
      const raw = (options, nPredict) => bridge.createCompletion(lazy.prompt, completionOptions(options, nPredict));
      const run = (name, options, nPredict) => snap.complete(lazy.prompt, completionOptions(options, nPredict), name);
      const usableWindow = (before, through) => {
        const word = through.startsWith(before) ? through.slice(before.length) : '';
        return word.length > 0 && through.indexOf(word) === before.length && !through.includes('\uFFFD');
      };
      let beforeTokens = lazy.minBeforeTokens;
      let before = await raw({}, beforeTokens);
      await snap.prepare(lazy.prompt, 'through');
      let through = await run('through', {}, beforeTokens + lazy.triggerTokens);
      while (!usableWindow(before, through) && beforeTokens < lazy.maxBeforeTokens) {
        beforeTokens += 1;
        before = await run('before', {}, beforeTokens);
        through = await run('through', {}, beforeTokens + lazy.triggerTokens);
      }
      const throughTokens = beforeTokens + lazy.triggerTokens;
      answeredTokens = throughTokens + lazy.answerTokens;
      const result = { beforeTokens, before, through, word: through.slice(before.length) };
      if (!usableWindow(before, through)) {
        result.word = null;
        return result;
      }
      const word = result.word;
      const answers = lazy.answers.map(gbnfLiteral).join(' | ');
      const grammar = \`root ::= \${gbnfLiteral(word)} (\${answers})\`;
      const triggerOptions = (trigger) => ({ grammar, grammarLazy: true, grammarTriggers: [trigger] });
      const triggers = {
        word: { type: 'word', value: word },
        pattern: { type: 'pattern', value: regexLiteral(word) },
        'pattern-full': { type: 'pattern-full', value: \`[\\\\s\\\\S]*?(\${regexLiteral(word)})\` },
      };
      result.triggered = {};
      for (const name of config.lazyTriggerRuns) {
        result.triggered[name] = await run(name, triggerOptions(triggers[name]));
      }
      result.triggeredWithPromptPenalties = await run('triggered-with-prompt-penalties', {
        grammar, grammarLazy: true, grammarTriggers: [triggers.word],
        penaltyPromptTokensWithGrammar: true,
      });
      result.strict = await run('strict', { grammar });

      await snap.restore(lazy.prompt, 'score');
      const { top } = await bridge.scoreNextToken(lazy.prompt, { topK: 2 });
      const piece = new TextDecoder().decode(top[0].bytes);
      if (top[0].logprob - top[1].logprob > lazy.minGap && piece.length > 0 && !piece.includes('\uFFFD')) {
        const tokenRun = (name, token, nPredict) => run(name, {
          grammar: \`root ::= \${gbnfLiteral(piece)} (\${answers})\`,
          grammarLazy: true,
          grammarTriggers: [{ type: 'token', token }],
        }, nPredict);
        result.tokenPiece = piece;
        result.tokenTriggered = await tokenRun('token-triggered', top[0].token);
        result.tokenNotSampled = await tokenRun('token-not-sampled', top[1].token, throughTokens);
      }
      result.root = await run('root', { grammar: lazy.rootGrammar, grammarRoot: 'answer' });
      result.errors = {
        invalidPattern: await errorOf(raw(triggerOptions({ type: 'pattern', value: '(' }))),
        outOfVocabulary: await errorOf(raw(triggerOptions({ type: 'token', token: 2147483647 }))),
        missingRoot: await errorOf(raw({ grammar, grammarRoot: 'missing' })),
        rejectedTrigger: await errorOf(raw({
          grammar: 'root ::= ' + answers,
          grammarLazy: true,
          grammarTriggers: [triggers.word],
        })),
      };
      result.afterErrors = await raw({}, throughTokens);
      return result;
    };

    const probeNativeMultiEndValidation = async (bridge, prompt) => {
      const core = bridge._runtime._core;
      const bytes = new TextEncoder().encode(prompt);
      const rejected = [];
      for (const test of config.invalidNativeMultiEnds) {
        const rc = await core.ccall(
          'llamadart_webgpu_begin_generation_multi_end_bytes', 'number',
          ['array', 'number', 'number', 'number', 'number', 'number', 'string', 'number',
            'number', 'number', 'number', 'string', 'string', 'string', 'number', 'string', 'number'],
          [bytes, bytes.length, 0, 1, 1, 1, null, 1, 0, 0, 0, '<think>', '</think>', null, 0, test.json, test.index],
          { async: true },
        );
        const error = core.ccall('llamadart_webgpu_last_error', 'string', [], []);
        if (rc === 0) {
          await core.ccall('llamadart_webgpu_end_generation', null, [], [], { async: true });
        }
        assert(rc === -2 && error?.includes(test.error),
          'native multi-end validation failed for ' + test.name + ' (rc=' + rc + ', error=' + error + ')');
        rejected.push(test.name);
      }
      return rejected;
    };

    const probeMultiEndBudget = async (bridge, runtimeMode, snap) => {
      const prompt = '<|im_start|>user\\nCall get_weather for Seoul.<|im_end|>\\n<|im_start|>assistant\\n<think>\\n';
      const opener = '<tool_call>';
      const endTag = '</think>';
      const forcedClose = opener + '\\n' + endTag;
      const embeddedClose = endTag + ' then <think> ' + forcedClose;
      const options = { temp: 0, topK: 1, penalty: 1, seed: 1, nPredict: 16 };
      const budget = { maxTokens: 0, startTag: '<think>', endTag };
      const naturalEndTags = ['\\n' + endTag, endTag, opener];
      const multi = { ...budget, naturalEndTags, grammarReplayEndIndex: 2 };
      const neverReplayGrammar = 'root ::= "yes"';
      const raw = (thinkingBudget, overrides = {}, input = prompt) =>
        bridge.createCompletion(input, { ...options, ...overrides, thinkingBudget });
      const run = (name, thinkingBudget, overrides = {}, input = prompt) =>
        snap.complete(input, { ...options, ...overrides, thinkingBudget }, name);
      const tokens = await bridge.tokenize(forcedClose, false);
      assert(tokens.join(',') === '248058,198,248069', 'Qwen3.5 forced close tokenization changed');

      const invalidNativeMultiEnds = runtimeMode === 'direct'
        ? await probeNativeMultiEndValidation(bridge, prompt) : null;
      const forced = { forcedMessage: opener + '\\n' };
      const legacy = await raw({ ...budget, ...forced }, { nPredict: tokens.length });
      await snap.prepare(prompt, 'full');
      const full = await run('full', { ...multi, ...forced }, { grammar: neverReplayGrammar });
      assert(legacy === forcedClose, 'legacy singleton changed the forced close: ' + JSON.stringify(legacy));
      assert(full === forcedClose + 'yes', 'multi-end did not emit the full close then resume grammar: ' + JSON.stringify(full));
      const embedded = await run(
        'embedded',
        { ...multi, forcedMessage: embeddedClose.slice(0, -endTag.length) },
        { grammar: neverReplayGrammar },
      );
      assert(embedded === embeddedClose + 'yes', 'embedded delimiters truncated forcing: ' + JSON.stringify(embedded));

      const activeGrammar = 'root ::= <[248058]> "weather"';
      const activeControl = await run('active-control', null, { grammar: activeGrammar });
      const active = await run(
        'active',
        { ...multi, maxTokens: 1, forcedMessage: 'BUDGET1949', pauseGrammar: false },
        { grammar: activeGrammar },
      );
      assert(activeControl === opener + 'weather' && active === activeControl,
        'a generated XML opener at the final allowed token changed the active grammar: ' + JSON.stringify(active));

      await snap.restore(prompt, 'score');
      const { top } = await bridge.scoreNextToken(prompt, { topK: 1 });
      const greedyToken = top[0].token;
      const greedyText = await bridge.detokenize([greedyToken], true);
      assert((await bridge.tokenize(greedyText, false)).join(',') === String(greedyToken),
        'the greedy marker did not round-trip to the same token');
      const closing = { ...multi, maxTokens: 1, naturalEndTags: [endTag, greedyText], grammarReplayEndIndex: 1 };
      const greedyGrammar = 'root ::= <[' + greedyToken + ']> "weather"';
      const eager = await run('eager', closing, { grammar: greedyGrammar });
      const lazy = await run('lazy', closing, {
        grammar: greedyGrammar,
        grammarLazy: true,
        grammarTriggers: [{ type: 'token', token: greedyToken }],
      });
      const unrelated = await run('unrelated', { ...closing, grammarReplayEndIndex: -1 }, { grammar: 'root ::= "weather"' });
      const nonSelected = await run(
        'non-selected',
        { ...closing, naturalEndTags: [endTag, greedyText, opener], grammarReplayEndIndex: 2 },
        { grammar: 'root ::= "weather"' },
      );
      assert([eager, lazy, unrelated, nonSelected].every((text) => text === greedyText + 'weather'),
        'natural replay routed the marker to the wrong grammar: ' + JSON.stringify({ eager, lazy, unrelated, nonSelected }));

      const noGrammar = await run('no-grammar', closing, { nPredict: 1 });
      assert(noGrammar === greedyText, 'a natural end tried to replay without a grammar: ' + JSON.stringify(noGrammar));

      const prefilled = prompt + opener + '\\n<function=get_weather>\\n';
      const prefilledControl = await raw(null, { nPredict: 6 }, prefilled);
      await snap.prepare(prefilled, 'prefilled-budgeted');
      const prefilledBudgeted = await run('prefilled-budgeted', { ...multi, forcedMessage: 'BUDGET1949' },
        { nPredict: 6 }, prefilled);
      assert(prefilledBudgeted === prefilledControl && !prefilledBudgeted.includes('BUDGET1949'),
        'the prefilled opener did not close the budget: ' + JSON.stringify(prefilledBudgeted));

      let rejected = false;
      try {
        await raw(closing, { grammar: neverReplayGrammar });
      } catch (error) {
        rejected = errorText(error).includes("paused grammar rejected the thinking budget's natural end");
      }
      assert(rejected, 'eager grammar accepted an invalid replay');

      const abort = new AbortController();
      try {
        await raw({ ...multi, ...forced }, {
          signal: abort.signal,
          tokenEventEncoding: 'text',
          onToken: () => abort.abort(),
        });
      } catch (error) {
        if (error.name !== 'AbortError') throw error;
      }
      assert(abort.signal.aborted, 'multi-end generation was not cancelled during forcing');
      const singletonAfterCancel = await raw({ ...budget, forcedMessage: 'BUDGET1949' }, { nPredict: 8 }, prefilled);
      assert(singletonAfterCancel.startsWith('BUDGET1949' + endTag),
        'multi-end state leaked into the next singleton request: ' + JSON.stringify(singletonAfterCancel));
      return {
        tokens, legacy, full, embedded, prefilledBudgeted, active, eager, lazy, unrelated, nonSelected, noGrammar,
        singletonAfterCancel, invalidNativeMultiEnds,
      };
    };

    const runMode = async (memoryMode, runtimeMode) => {
      const startedAt = performance.now();
      const mode = \`\${memoryMode} \${runtimeMode}\`;
      const useMemory64 = memoryMode === 'wasm64';
      const bridge = new LlamaWebGpuBridge({
        disableWorker: runtimeMode === 'direct',
        preferMemory64: useMemory64,
        coreModuleUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.js' : undefined,
        wasmUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.wasm' : undefined,
      });
      const availableThreads = Math.trunc(Number(navigator.hardwareConcurrency) || config.maxThreads);
      const nThreads = Math.max(1, Math.min(config.maxThreads, availableThreads));
      const phaseTimingsMs = {};
      const snap = createPrefixSnapshotHelper(bridge);
      const timePhase = async (key, stage, run) => {
        setStage(\`\${mode}:\${stage}\`);
        snap.setPhase(key);
        const phaseStartedAt = performance.now();
        try {
          return await run();
        } finally {
          phaseTimingsMs[key] = Math.round(performance.now() - phaseStartedAt);
        }
      };
      try {
        const capabilitiesBeforeLoad = await timePhase('loadModelMs', 'load-model', async () => {
          const capabilities = samplerCapabilities(await bridge.getCompletionCapabilities());
          await bridge.loadModelFromUrl(config.modelUrl, {
            nCtx: config.nCtx,
            nThreads,
            nGpuLayers: 0,
            useCache: false,
            forceRemoteFetchBackend: false,
          });
          return capabilities;
        });
        const cases = await timePhase('casesMs', 'cases', async () => {
          const cases = [];
          for (const testCase of config.cases) {
            const useSnapshot = testCase.kind !== 'invalid' && testCase.name !== 'yes-no-greedy';
            if (useSnapshot) await snap.prepare(config.prompt, testCase.name);
            let text = null;
            let error = null;
            try {
              const options = {
                ...testCase.options,
                grammar: testCase.grammar,
                tokenEventEncoding: 'text',
              };
              text = useSnapshot ? await snap.complete(config.prompt, options, testCase.name)
                : await bridge.createCompletion(config.prompt, options);
            } catch (caught) {
              error = errorText(caught);
            }
            cases.push({
              name: testCase.name,
              text,
              error,
              // A throwing case must carry the parser's message, which proves
              // the throw reached the bridge's catch.
              valid: testCase.kind === 'invalid'
                ? text === null
                  && error !== null
                  && error.includes(config.invalidGrammarError)
                  && error.includes(config.nonThrowingRejection) === !testCase.throws
                : error === null && typeof text === 'string' && checkText(testCase.kind, text),
            });
            if (testCase.name === 'yes-no-greedy') await snap.detect();
          }
          return cases;
        });

        // Generation state must be clean after the grammar runs.
        const plainError = await timePhase('plainCompletionMs', 'plain-completion', async () => {
          try {
            await bridge.createCompletion(config.prompt, {
              nPredict: 4,
              temp: 0,
              topK: 1,
              seed: 1,
              tokenEventEncoding: 'text',
            });
            return null;
          } catch (caught) {
            return errorText(caught);
          }
        });

        const { capabilitiesAfterLoad, samplerRuns, samplerPredictLimit, invalidSamplerErrors, samplerAfterInvalid } =
          await timePhase('samplerRunsMs', 'sampler-runs', async () => {
            const capabilitiesAfterLoad = samplerCapabilities(await bridge.getCompletionCapabilities());
            const prompt = config.samplerPrompt;
            const invalidSamplerErrors = [];
            const runSamplerBatch = async (limit) => {
              const runs = {};
              for (const [name, options] of config.samplerRuns) {
                const completionOptions = { ...options, nPredict: Math.min(options.nPredict, limit), tokenEventEncoding: 'text' };
                if (name === 'first') {
                  runs[name] = await bridge.createCompletion(prompt, completionOptions);
                } else {
                  const run = name + '-' + limit;
                  await snap.prepare(prompt, run);
                  runs[name] = await snap.complete(prompt, completionOptions, run);
                }
              }
              return runs;
            };
            const sampledOptions = Object.fromEntries(config.samplerRuns).sampled;
            let samplerPredictLimit = config.samplerShortPredict;
            let samplerRuns = await runSamplerBatch(samplerPredictLimit);
            if (samplerRuns.sampled === samplerRuns.greedy || samplerRuns['greedy-presence'] === samplerRuns.greedy) {
              samplerPredictLimit = sampledOptions.nPredict;
              samplerRuns = await runSamplerBatch(samplerPredictLimit);
            }
            const sampled = { ...sampledOptions, nPredict: samplerPredictLimit };
            for (const options of ${invalidSamplerOptions}) {
              try {
                await bridge.createCompletion(config.samplerPrompt, { ...sampled, ...options });
                invalidSamplerErrors.push(null);
              } catch (caught) {
                invalidSamplerErrors.push(\`\${caught && caught.name}: \${errorText(caught)}\`);
              }
            }
            // Repeats "sampled" after the rejections.
            const samplerAfterInvalid = await bridge.createCompletion(config.samplerPrompt, {
              ...sampled,
              tokenEventEncoding: 'text',
            });
            return { capabilitiesAfterLoad, samplerRuns, samplerPredictLimit, invalidSamplerErrors, samplerAfterInvalid };
          });

        const promptPenalty = await timePhase('promptPenaltyMs', 'prompt-penalty', () => probePromptPenalty(bridge, snap));
        const { thinkingRuns, invalidThinkingBudgetError } = await timePhase('thinkingRunsMs', 'thinking-runs', async () => {
          const thinkingRuns = {};
          for (const [name, prompt, options] of config.thinkingRuns) {
            const raw = name.endsWith('-first') || name === 'open-budget-rejecting-grammar' || name === 'open-budget-after-rejecting';
            if (!raw) await snap.prepare(prompt, name);
            try {
              const completionOptions = { ...options, tokenEventEncoding: 'text' };
              thinkingRuns[name] = raw ? await bridge.createCompletion(prompt, completionOptions)
                : await snap.complete(prompt, completionOptions, name);
            } catch (caught) {
              if (name !== 'open-budget-rejecting-grammar') throw caught;
              thinkingRuns[name] = \`ERR \${caught && caught.name}: \${errorText(caught)}\`;
            }
          }
          let invalidThinkingBudgetError = null;
          try {
            await bridge.createCompletion(config.thinkingRuns[0][1], {
              ...config.thinkingRuns[0][2],
              thinkingBudget: config.invalidThinkingBudget,
            });
          } catch (caught) {
            invalidThinkingBudgetError = \`\${caught && caught.name}: \${errorText(caught)}\`;
          }
          return { thinkingRuns, invalidThinkingBudgetError };
        });
        const lazyGrammar = await timePhase('lazyGrammarMs', 'lazy-grammar', () => probeLazyGrammar(bridge, snap));

        const metadata = bridge.getModelMetadata();
        const multiEndBudget = await timePhase('multiEndBudgetMs', 'multi-end-budget', () => (
          metadata['general.architecture'] === 'qwen35' ? probeMultiEndBudget(bridge, runtimeMode, snap) : null
        ));
        return {
          mode,
          cases,
          capabilitiesBeforeLoad,
          capabilitiesAfterLoad,
          samplerRuns,
          samplerPredictLimit,
          invalidSamplerErrors,
          samplerAfterInvalid,
          promptPenalty,
          thinkingRuns,
          multiEndBudget,
          invalidThinkingBudgetError,
          lazyGrammar,
          plainCompletionError: plainError,
          nThreads: Number(metadata['llamadart.webgpu.n_threads']) || null,
          phaseTimingsMs,
          prefixSnapshotReuse: snap.stats,
          execution: metadata['llamadart.webgpu.execution'] || null,
          coreVariant: metadata['llamadart.webgpu.core_variant'] || null,
          workerFallbackReason: metadata['llamadart.webgpu.worker_fallback_reason'] || null,
        };
      } finally {
        await timePhase('disposeMs', 'dispose', () => bridge.dispose());
        phaseTimingsMs.totalMs = Math.round(performance.now() - startedAt);
        setStage(\`\${mode}:done\`);
      }
    };

    for (const memoryMode of config.memoryModes) {
      for (const runtimeMode of config.runtimeModes) {
        modeResults.push(await runMode(memoryMode, runtimeMode));
      }
    }
    finish({
      ok: true,
      modeResults,
      globalWorkerFallbackReason: globalThis.__llamadartBridgeWorkerFallbackReason || null,
    });
  } catch (error) {
    finish({ ok: false, error: String(error && error.stack ? error.stack : error) });
  }
})();
</script>
`;
}

export function validatePayload(payload, memoryModes, runtimeModes = RUNTIME_MODES) {
  const failures = [];
  if (payload.ok !== true) return [`harness failed: ${pyStr(pyGet(payload, 'error'))}`];
  const modeResults = pyGet(payload, 'modeResults');
  if (!Array.isArray(modeResults) || !pyEquals(
    modeResults.filter(isDict).map((entry) => pyGet(entry, 'mode')),
    expectedModes(memoryModes, runtimeModes),
  )) {
    return ['mode results missing'];
  }
  const expectedCases = CASES.map((testCase) => testCase.name);
  for (const entry of modeResults) {
    const mode = modeOf(entry);
    const cases = pyGet(entry, 'cases');
    if (!Array.isArray(cases) || !pyEquals(cases.map((testCase) => pyGet(testCase, 'name')), expectedCases)) {
      failures.push(`${mode}: case results missing`);
      continue;
    }
    for (const testCase of cases) {
      if (pyGet(testCase, 'valid') !== true) {
        const expected = pyGet(testCase, 'name').startsWith('invalid-')
          ? `an error containing ${pyRepr(INVALID_GRAMMAR_ERROR)} and the `
            + "parser's reason (or the non-throwing rejection for left recursion)"
          : 'grammar-valid text';
        failures.push(
          `${mode} ${pyStr(pyGet(testCase, 'name'))}: expected ${expected}, `
            + `got text=${pyRepr(pyGet(testCase, 'text'))} error=${pyRepr(pyGet(testCase, 'error'))}`,
        );
      }
    }
    failures.push(...validateSampler(entry).map((failure) => `${mode}: ${failure}`));
    if (pyGet(entry, 'plainCompletionError') !== null) {
      failures.push(`${mode}: completion after the grammar runs failed: ${pyRepr(pyGet(entry, 'plainCompletionError'))}`);
    }
    failures.push(...modeRuntimeFailures(mode, entry));
  }
  if (pyGet(payload, 'globalWorkerFallbackReason') !== null) {
    failures.push(`worker fell back to the main thread: ${pyRepr(pyGet(payload, 'globalWorkerFallbackReason'))}`);
  }
  return failures;
}

const capabilities = (supported) => Object.fromEntries(COMPLETION_CAPABILITIES.map((name) => [name, supported]));

export function validateSampler(entry) {
  const failures = [];
  const before = pyGet(entry, 'capabilitiesBeforeLoad');
  if (!pyEquals(before, capabilities(false))) {
    failures.push(`capabilities before the load: expected none, got ${pyRepr(before)}`);
  }
  const after = pyGet(entry, 'capabilitiesAfterLoad');
  if (!pyEquals(after, capabilities(true))) {
    failures.push(`capabilities after the load: expected all, got ${pyRepr(after)}`);
  }
  const runs = pyGet(entry, 'samplerRuns');
  const names = SAMPLER_RUNS.map(([name]) => name);
  if (!isDict(runs) || !names.every((name) => typeof runs[name] === 'string' && runs[name].length > 0)) {
    return [...failures, `sampler runs missing or empty: ${pyRepr(runs)}`];
  }
  const expect = (condition, message) => {
    if (!condition) failures.push(message);
  };
  expect(runs['sampled-explicit-zeros'] === runs.sampled, 'minP: 0 and presencePenalty: 0 changed the default output');
  expect(runs.sampled !== runs.greedy, 'plain sampling matched greedy, so min-p-one proves nothing');
  expect(runs['min-p-one'] === runs.greedy, 'minP: 1 did not reduce sampling to the most probable token');
  expect(runs['greedy-presence'] !== runs.greedy, 'presencePenalty did not change greedy output');
  const errors = pyGet(entry, 'invalidSamplerErrors');
  if (!Array.isArray(errors) || errors.length !== INVALID_SAMPLER_OPTIONS.length || !errors.every(
    (error) => typeof error === 'string' && error.startsWith('RangeError: CompletionOptions.'),
  )) {
    failures.push(`invalid sampler options: expected RangeErrors, got ${pyRepr(errors)}`);
  }
  expect(
    pyGet(entry, 'samplerAfterInvalid') === runs.sampled,
    'a seeded sample after the rejected options did not repeat the earlier one',
  );
  if (failures.length > 0) failures.push(`sampler outputs: ${pyRepr(runs)}`);
  return [
    ...failures,
    ...validatePromptPenalty(entry),
    ...validateThinkingBudget(entry),
    ...validateLazyGrammar(entry),
  ];
}

export function validateLazyGrammar(entry) {
  const probe = pyGet(entry, 'lazyGrammar');
  if (!isDict(probe) || typeof pyGet(probe, 'word') !== 'string' || !isDict(pyGet(probe, 'triggered'))) {
    return [
      `lazy grammar: no ${LAZY_GRAMMAR.triggerTokens}-token window of the greedy completion after token `
        + `${LAZY_GRAMMAR.minBeforeTokens} to ${LAZY_GRAMMAR.maxBeforeTokens} is new text: ${pyRepr(probe)}`,
    ];
  }
  const failures = [];
  const expect = (condition, message) => {
    if (!condition) failures.push(`lazy grammar: ${message}`);
  };
  const { before, word } = probe;
  const answered = (text, prefix) => typeof text === 'string'
    && text.startsWith(prefix)
    && LAZY_GRAMMAR.answers.includes(text.slice(prefix.length));
  expect(before.length > 0, 'the text before the trigger is empty, so it proves nothing');
  for (const name of LAZY_TRIGGER_RUNS) {
    expect(
      answered(probe.triggered[name], before + word),
      `a ${name} trigger did not leave the text before it unconstrained and constrain the rest`,
    );
  }
  expect(
    answered(probe.triggeredWithPromptPenalties, before + word),
    'the opt-in changed the pre-trigger prefix or failed to constrain the generated trigger',
  );
  expect(answered(probe.strict, word), 'the strict grammar did not constrain the output from the first token');
  if (typeof pyGet(probe, 'tokenPiece') === 'string') {
    expect(answered(probe.tokenTriggered, probe.tokenPiece), 'a token trigger did not constrain the text after the token');
    expect(probe.tokenNotSampled === probe.through, 'a token trigger that is never sampled constrained the output');
  } else {
    failures.push('lazy grammar: the first greedy token is a near tie or no usable text, so the token trigger was not run');
  }
  expect(LAZY_GRAMMAR.answers.includes(probe.root), 'the grammar did not start at grammarRoot');
  const errors = isDict(pyGet(probe, 'errors')) ? probe.errors : {};
  for (const [name, parts] of Object.entries(LAZY_GRAMMAR_ERRORS)) {
    const error = pyGet(errors, name);
    expect(
      typeof error === 'string' && parts.every((part) => error.includes(part)),
      `${name}: expected an error containing ${parts.map(pyRepr).join(' and ')}, got ${pyRepr(error)}`,
    );
  }
  expect(probe.afterErrors === probe.through, 'the unconstrained completion changed after the rejected grammars');
  if (failures.length > 0) failures.push(`lazy grammar outputs: ${pyRepr(probe)}`);
  return failures;
}

export function validatePromptPenalty(entry) {
  const probe = pyGet(entry, 'promptPenalty');
  if (!isDict(probe) || typeof pyGet(probe, 'best') !== 'string') {
    return [
      `prompt penalty: no prompt within ${PROMPT_PENALTY.maxSteps} steps had its most probable token `
        + `among its last ${PROMPT_PENALTY.window} tokens: ${pyRepr(probe)}`,
    ];
  }
  const failures = [];
  const expect = (condition, message) => {
    if (!condition) failures.push(`prompt penalty: ${message}`);
  };
  const { best, unpenalized } = probe;
  expect(probe.neutral === best, 'the greedy token is not the most probable scored token');
  expect(
    probe.penalized === unpenalized,
    'the presence penalty did not count the prompt, so a prompt token stayed the greedy pick',
  );
  expect(probe.grammar === best, 'with a grammar the presence penalty counted the prompt, unlike native llamadart');
  expect(probe.emptyGrammar === best, 'with an empty grammar the presence penalty counted the prompt, unlike native llamadart');
  expect(probe.optOut === best, 'with penaltyPromptTokens: false the presence penalty counted the prompt');
  for (const field of ['grammarOptIn', 'emptyGrammarOptIn', 'promptTriggerOptIn']) {
    expect(probe[field] === probe.penalized, field + ' did not preserve the no-grammar prompt-penalty baseline');
  }
  if (failures.length > 0) failures.push(`prompt penalty outputs: ${pyRepr(probe)}`);
  return failures;
}

export function validateThinkingBudget(entry) {
  const runs = pyGet(entry, 'thinkingRuns');
  const names = THINKING_RUNS.map(([name]) => name);
  if (!isDict(runs) || !names.every((name) => typeof runs[name] === 'string')) {
    return [`thinking-budget runs missing: ${pyRepr(runs)}`];
  }
  const failures = [];
  const expect = (condition, message) => {
    if (!condition) failures.push(message);
  };
  const endTag = THINK_TAGS.endTag;
  const prefix = runs['open-prefix'];
  const afterPrefix = (text) => (text.startsWith(prefix) ? text.slice(prefix.length).trimStart() : null);
  expect(prefix.length > 0, 'the unbudgeted reasoning prefix is empty');
  expect(!prefix.includes(endTag), 'the model closed the block itself, so the budget proves nothing');
  expect(
    afterPrefix(runs['open-budget'])?.startsWith(endTag) === true,
    `maxTokens ${THINK_BUDGET_TOKENS} did not force ${pyRepr(endTag)} after the unbudgeted prefix`,
  );
  expect(
    runs['open-budget-zero'].trimStart().startsWith(FORCED_MESSAGE + endTag),
    `maxTokens 0 did not start with the forced ${pyRepr(FORCED_MESSAGE + endTag)}`,
  );
  expect(
    [`${endTag}yes`, `${endTag}no`].includes(afterPrefix(runs['open-budget-grammar'])),
    'the grammar did not pause inside the block and constrain the text after it',
  );
  expect(
    [`${endTag}yes`, `${endTag}no`].includes(afterPrefix(runs['open-budget-taking-grammar'])),
    'the grammar did not stay applied inside the block through the forced end tag',
  );
  expect(
    /^ERR .*Grammar rejected every candidate token/.test(runs['open-budget-rejecting-grammar']),
    'a grammar rejecting the forced end tag did not fail with the grammar-rejection error',
  );
  expect(
    [`${endTag}yes`, `${endTag}no`].includes(afterPrefix(runs['open-budget-after-rejecting'])),
    'the default pausing grammar changed after a rejected forced end tag',
  );
  expect(runs['closed-budget-zero'] === runs['closed-plain'], 'a budget changed output after the prompt closed the block');
  const error = pyGet(entry, 'invalidThinkingBudgetError');
  if (typeof error !== 'string' || !error.startsWith('RangeError: CompletionOptions.thinkingBudget.maxTokens')) {
    failures.push(`invalid thinking budget: expected a RangeError, got ${pyRepr(error)}`);
  }
  if (failures.length > 0) failures.push(`thinking-budget outputs: ${pyRepr(runs)}`);
  return failures;
}

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'grammar.mjs',
    description: DESCRIPTION,
    options: [
      {
        flag: '--dist-dir',
        type: 'path',
        default: () => env.path('BRIDGE_DIST_DIR', 'dist'),
        help: 'Directory containing built bridge artifacts.',
      },
      {
        flag: '--timeout-ms',
        type: 'int',
        default: () => env.int('LLAMA_WEBGPU_GRAMMAR_TIMEOUT_MS', '300000'),
        help: 'Browser operation timeout in milliseconds.',
      },
      {
        flag: '--model-url',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_URL'),
        help: 'GGUF URL; defaults to the state-persistence smoke model.',
      },
      {
        flag: '--model-path',
        type: 'path',
        default: () => env.optionalPath('LLAMA_WEBGPU_SMOKE_MODEL_PATH'),
        help: 'Local GGUF path.',
      },
      {
        flag: '--model-sha256',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SMOKE_MODEL_SHA256'),
        help: 'Expected model SHA-256.',
      },
      {
        flag: '--model-cache-dir',
        type: 'path',
        default: () => env.path('LLAMA_WEBGPU_SMOKE_MODEL_CACHE', DEFAULT_MODEL_CACHE),
        help: 'Cache directory used with --model-url.',
      },
      {
        flag: '--memory-mode',
        type: 'string',
        choices: ['all', ...MEMORY_MODES],
        default: () => 'all',
        help: 'Core memory mode to run; wasm64 needs the mem64 artifacts.',
      },
      {
        flag: '--runtime-mode',
        type: 'string',
        choices: ['all', ...RUNTIME_MODES],
        default: () => 'all',
        help: 'Bridge runtime to run, or all runtimes.',
      },
      {
        flag: '--n-ctx',
        type: 'int',
        default: () => 1024,
        help: 'Context size passed to loadModelFromUrl.',
      },
      {
        flag: '--artifacts-dir',
        type: 'path',
        default: () => env.optionalPath('LLAMA_WEBGPU_GRAMMAR_ARTIFACTS_DIR'),
        help: 'Directory for JSON/console/screenshot diagnostics.',
      },
    ],
  });
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const distDir = resolvePath(args.distDir);
  ensure(isDirectory(distDir), `dist directory does not exist: ${distDir}`);
  const artifactsDir = args.artifactsDir !== null ? resolvePath(args.artifactsDir) : null;
  const modelPath = await resolvePinnedModel(args);
  const memoryModes = args.memoryMode === 'all' ? MEMORY_MODES : [args.memoryMode];
  const runtimeModes = args.runtimeMode === 'all' ? RUNTIME_MODES : [args.runtimeMode];

  const payload = await withTempDir('llama-web-bridge-grammar-smoke-', async (webRoot) => {
    await copyBridgeArtifacts(distDir, webRoot);
    if (memoryModes.includes('wasm64')) await copyMemory64Artifacts(distDir, webRoot);
    await stageFile(modelPath, path.join(webRoot, MODEL_FILENAME));
    await fsp.writeFile(path.join(webRoot, 'index.html'), renderHarness(args.nCtx, memoryModes, runtimeModes), 'utf8');
    if (artifactsDir !== null) {
      await fsp.mkdir(artifactsDir, { recursive: true });
      await fsp.copyFile(path.join(webRoot, 'index.html'), path.join(artifactsDir, 'index.html'));
    }
    return withServer(webRoot, (url) => runPlaywright(url, args.timeoutMs, artifactsDir, 'grammar-smoke', {
      stageMarker: 'GRAMMAR_STAGE ',
      stagePrefix: 'grammar-smoke',
    }));
  });

  writeStdout(`${pyJson(payload, { indent: 2, sortKeys: true })}\n`);
  const failures = validatePayload(payload, memoryModes, runtimeModes);
  for (const failure of failures) process.stderr.write(`grammar browser smoke: ${failure}\n`);
  return failures.length ? 1 : 0;
}

if (import.meta.main) {
  await runMain('grammar', main);
}
