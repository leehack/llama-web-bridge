import assert from 'node:assert/strict';

import { completionCapabilitiesFrom } from '../../js/src/internal/completion_options.ts';
import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';
import { createDirectBridge, createWorkerBridge } from './bridge_operation_queue_fixtures.mjs';
import { readNativeCoreSource } from './native_core_source.mjs';

const coreSource = readNativeCoreSource();
const SUPPORTED = { presencePenalty: true, minP: true, thinkingBudget: true };
const NONE = { presencePenalty: false, minP: false, thinkingBudget: false };
const BUDGET = { maxTokens: 8, startTag: '<think>', endTag: '</think>' };
const BUDGET_PARAMETERS = [
  'thinking_budget_tokens',
  'thinking_start_tag',
  'thinking_end_tag',
  'thinking_forced_message',
];

function functionBody(signature) {
  const start = coreSource.indexOf(signature);
  assert.ok(start >= 0, `${signature} not found in the native core`);
  return coreSource.slice(start, coreSource.indexOf('\n}\n', start));
}

function nativeParameterNames(name) {
  const match = coreSource.match(
    new RegExp(`EMSCRIPTEN_KEEPALIVE\\s+int32_t\\s+${name}\\(([^)]*)\\)`),
  );
  assert.ok(match, `${name} is not exported`);
  return match[1].split(',').map((parameter) => parameter.trim().match(/(\w+)$/)[1]);
}

// The capability string the core returns, as the C++ literal decodes it.
function coreCapabilityJson() {
  const body = functionBody('llamadart_webgpu_completion_capabilities_json()');
  const literal = body.match(/return\s+("(?:[^"\\]|\\.)*");/);
  assert.ok(literal, 'the capability export must return one string literal');
  return JSON.parse(literal[1]);
}

// A direct bridge whose stub core records begin_generation's arguments and,
// when capabilityJson is given, exports the completion capability probe.
function directBridge(capabilityJson) {
  const { bridge, core } = createDirectBridge({ hello: ['a'] });
  const begins = [];
  const ccall = core.ccall.bind(core);
  core.ccall = (name, returnType, argTypes, args) => {
    if (name === 'llamadart_webgpu_begin_generation') {
      begins.push({ argTypes, args });
    }
    if (name === 'llamadart_webgpu_completion_capabilities_json') {
      return capabilityJson;
    }
    return ccall(name, returnType, argTypes, args);
  };
  if (capabilityJson !== undefined) {
    core._llamadart_webgpu_completion_capabilities_json = () => 0;
  }
  return { bridge, core, begins };
}

const INVALID = [
  [{ minP: -0.01 }, /CompletionOptions\.minP must be a finite number from 0 to 1; got -0\.01/],
  [{ minP: 1.01 }, /CompletionOptions\.minP/],
  [{ minP: Number.NaN }, /CompletionOptions\.minP/],
  [{ minP: Number.POSITIVE_INFINITY }, /CompletionOptions\.minP/],
  [{ minP: '0.1' }, /CompletionOptions\.minP/],
  [{ presencePenalty: Number.NaN }, /CompletionOptions\.presencePenalty must be a finite number; got NaN/],
  [{ presencePenalty: Number.NEGATIVE_INFINITY }, /CompletionOptions\.presencePenalty/],
  [{ presencePenalty: '1' }, /CompletionOptions\.presencePenalty/],
  [{ presencePenalty: 1e39 }, /CompletionOptions\.presencePenalty must be a finite number; got 1e\+39/],
  [{ thinkingBudget: { ...BUDGET, maxTokens: -1 } }, /thinkingBudget\.maxTokens must be an integer from 0 to 2147483647; got -1/],
  [{ thinkingBudget: { ...BUDGET, maxTokens: 1.5 } }, /thinkingBudget\.maxTokens/],
  [{ thinkingBudget: { ...BUDGET, maxTokens: 2147483648 } }, /thinkingBudget\.maxTokens/],
  [{ thinkingBudget: { ...BUDGET, maxTokens: '8' } }, /thinkingBudget\.maxTokens/],
  [{ thinkingBudget: { ...BUDGET, maxTokens: undefined } }, /thinkingBudget\.maxTokens/],
];

const INVALID_TYPES = [
  [{ thinkingBudget: 8 }, /CompletionOptions\.thinkingBudget must be an object/],
  [{ thinkingBudget: { ...BUDGET, startTag: '' } }, /thinkingBudget\.startTag must be a non-empty string/],
  [{ thinkingBudget: { ...BUDGET, startTag: ' \n' } }, /thinkingBudget\.startTag/],
  [{ thinkingBudget: { maxTokens: 1, startTag: '<think>' } }, /thinkingBudget\.endTag must be a non-empty string/],
  [{ thinkingBudget: { ...BUDGET, forcedMessage: 3 } }, /thinkingBudget\.forcedMessage must be a string/],
];

const MEDIA_BUDGET = {
  thinkingBudget: BUDGET,
  parts: [{ type: 'image', bytes: new Uint8Array([1]) }],
};

const CASES = [
  ['begin_generation receives the sampling options after the existing arguments', async () => {
    const { bridge, begins } = directBridge(coreCapabilityJson());
    assert.equal(
      await bridge.createCompletion('hello', {
        minP: 0.05,
        presencePenalty: -1.5,
        seed: 3,
        thinkingBudget: { ...BUDGET, forcedMessage: 'Done.' },
      }),
      'a',
    );

    const names = nativeParameterNames('llamadart_webgpu_begin_generation');
    assert.deepEqual(names.slice(-6), ['min_p', 'presence_penalty', ...BUDGET_PARAMETERS]);
    const [{ argTypes, args }] = begins;
    assert.equal(argTypes.length, names.length);
    assert.equal(args.length, names.length);
    const arg = (name) => args[names.indexOf(name)];
    const type = (name) => argTypes[names.indexOf(name)];
    assert.equal(arg('min_p'), 0.05);
    assert.equal(arg('presence_penalty'), -1.5);
    assert.equal(arg('seed'), 3);
    assert.deepEqual(BUDGET_PARAMETERS.map(arg), [8, '<think>', '</think>', 'Done.']);
    assert.deepEqual(
      ['min_p', 'presence_penalty', ...BUDGET_PARAMETERS].map(type),
      ['number', 'number', 'number', 'string', 'string', 'string'],
    );
  }],

  ['omitted, null and zero options pass defaults and need no capability', async () => {
    const { bridge, begins } = directBridge(undefined);
    await bridge.createCompletion('hello');
    await bridge.createCompletion('hello', { minP: null, presencePenalty: null, thinkingBudget: null });
    await bridge.createCompletion('hello', { minP: 0, presencePenalty: 0 });
    const disabled = [0, 0, 0, null, null, null];
    assert.deepEqual(begins.map(({ args }) => args.slice(-6)), [disabled, disabled, disabled]);
  }],

  ['the range ends are accepted and forcedMessage defaults to empty', async () => {
    const { bridge, begins } = directBridge(JSON.stringify(SUPPORTED));
    await bridge.createCompletion('hello', { minP: 1, presencePenalty: 1e6, thinkingBudget: { ...BUDGET, maxTokens: 0 } });
    await bridge.createCompletion('hello', { thinkingBudget: { ...BUDGET, maxTokens: 2147483647 } });
    assert.deepEqual(begins[0].args.slice(-6), [1, 1e6, 0, '<think>', '</think>', '']);
    assert.equal(begins[1].args.at(-4), 2147483647);
  }],

  ['an invalid value rejects before the direct runtime touches the core', async () => {
    const { bridge, core, begins } = directBridge(JSON.stringify(SUPPORTED));
    for (const [options, message] of INVALID) {
      await assert.rejects(bridge.createCompletion('hello', options), (error) => {
        assert.equal(error.name, 'RangeError');
        assert.match(error.message, message);
        return true;
      });
    }
    for (const [options, message] of INVALID_TYPES) {
      await assert.rejects(bridge.createCompletion('hello', options), (error) => {
        assert.equal(error.name, 'TypeError');
        assert.match(error.message, message);
        return true;
      });
    }
    await assert.rejects(
      bridge.createCompletion('hello', MEDIA_BUDGET),
      /CompletionOptions\.thinkingBudget supports text-only prompts/,
    );
    assert.deepEqual(begins, []);
    assert.deepEqual(core.trace, []);
    assert.equal(await bridge.createCompletion('hello'), 'a', 'the bridge stays usable');
  }],

  ['an invalid value rejects before a worker request', async () => {
    const calls = [];
    const bridge = createWorkerBridge({
      _callWorker: async (method) => {
        calls.push(method);
        return 'x';
      },
    });
    for (const [options] of INVALID) {
      await assert.rejects(bridge.createCompletion('hello', options), RangeError);
    }
    for (const [options] of INVALID_TYPES) {
      await assert.rejects(bridge.createCompletion('hello', options), TypeError);
    }
    // A worker media failure would otherwise move the bridge to the main thread.
    await assert.rejects(bridge.createCompletion('hello', MEDIA_BUDGET), /text-only prompts/);
    assert.deepEqual(calls, []);
  }],

  ['a worker completion forwards the options', async () => {
    let sent = null;
    const bridge = createWorkerBridge({
      _callWorker: async (method, args) => {
        assert.equal(method, 'createCompletion');
        sent = args[1];
        return 'x';
      },
    });
    await bridge.createCompletion('hello', { minP: 0.1, presencePenalty: 0.5, thinkingBudget: BUDGET });
    assert.equal(structuredClone(sent).minP, 0.1);
    assert.equal(structuredClone(sent).presencePenalty, 0.5);
    assert.deepEqual(structuredClone(sent).thinkingBudget, BUDGET);
  }],

  ['a core without the probe reports nothing and rejects a nonzero option', async () => {
    const { bridge, begins } = directBridge(undefined);
    assert.deepEqual(await bridge.getCompletionCapabilities(), NONE);
    await assert.rejects(
      bridge.createCompletion('hello', { minP: 0.1 }),
      /does not support CompletionOptions\.minP\.$/,
    );
    await assert.rejects(
      bridge.createCompletion('hello', { minP: 0.1, presencePenalty: 1 }),
      /does not support CompletionOptions\.minP or CompletionOptions\.presencePenalty\.$/,
    );
    await assert.rejects(
      bridge.createCompletion('hello', { thinkingBudget: BUDGET }),
      /does not support CompletionOptions\.thinkingBudget\.$/,
    );
    assert.deepEqual(begins, []);
  }],

  ['a core reporting one option rejects only the other', async () => {
    const { bridge, begins } = directBridge(JSON.stringify({ minP: true }));
    assert.deepEqual(
      await bridge.getCompletionCapabilities(),
      { presencePenalty: false, minP: true, thinkingBudget: false },
    );
    await bridge.createCompletion('hello', { minP: 0.1 });
    await assert.rejects(
      bridge.createCompletion('hello', { presencePenalty: 1 }),
      /does not support CompletionOptions\.presencePenalty\.$/,
    );
    assert.equal(begins.length, 1);
  }],

  ['the core literal reports every option', async () => {
    assert.deepEqual(completionCapabilitiesFrom(coreCapabilityJson()), SUPPORTED);
    const { bridge } = directBridge(coreCapabilityJson());
    assert.deepEqual(await bridge.getCompletionCapabilities(), SUPPORTED);
  }],

  ['a malformed or non-boolean capability response reports nothing', async () => {
    assert.deepEqual(completionCapabilitiesFrom('{'), NONE);
    assert.deepEqual(completionCapabilitiesFrom(null), NONE);
    assert.deepEqual(completionCapabilitiesFrom('{"minP":1,"presencePenalty":"true","thinkingBudget":{}}'), NONE);
  }],

  ['capabilities before a core exists report nothing', async () => {
    const bridge = new LlamaWebGpuBridge({ disableWorker: true });
    assert.deepEqual(await bridge.getCompletionCapabilities(), NONE);
    const unloaded = createWorkerBridge({ _workerProxy: null });
    assert.deepEqual(await unloaded.getCompletionCapabilities(), NONE);
  }],

  ['a worker answers the capability probe', async () => {
    const calls = [];
    const bridge = createWorkerBridge({
      _callWorker: async (method, args) => {
        calls.push([method, args]);
        return SUPPORTED;
      },
    });
    assert.deepEqual(await bridge.getCompletionCapabilities(), SUPPORTED);
    assert.deepEqual(calls, [['getCompletionCapabilities', []]]);
  }],

  ['the native chain adds penalties for either penalty and orders min-p after top-p', async () => {
    const body = functionBody('llama_sampler * create_sampler(');
    assert.match(body, /if \(repeat_penalty != 1\.0f \|\| presence_penalty != 0\.0f\)/);
    assert.match(
      body,
      /llama_sampler_init_penalties\(\s*vocab_size, 64, repeat_penalty, 0\.0f, presence_penalty\)/,
    );
    assert.match(body, /if \(min_p > 0\.0f\) \{\s*llama_sampler_chain_add\(sampler, llama_sampler_init_min_p\(min_p, 1\)\);/);
    const order = [
      'llama_sampler_init_penalties',
      'owned_reasoning_budget.release()',
      'llama_sampler_init_top_k',
      'llama_sampler_init_top_p',
      'llama_sampler_init_min_p',
      'llama_sampler_init_temp',
      'llama_sampler_init_dist',
    ].map((name) => body.indexOf(name));
    assert.ok(order.every((index) => index >= 0), 'every sampler is in the chain');
    assert.deepEqual([...order].sort((a, b) => a - b), order);
  }],

  ['the native core rejects invalid values before it changes generation state', async () => {
    const body = functionBody('int32_t begin_generation_impl(');
    const checks = [
      body.indexOf('!std::isfinite(min_p) || min_p < 0.0f || min_p > 1.0f'),
      body.indexOf('!std::isfinite(presence_penalty)'),
      body.indexOf('set_error("thinking budget supports text-only prompts");'),
      body.indexOf('!resolve_reasoning_budget('),
    ];
    const firstSideEffect = body.indexOf('end_generation_state();');
    assert.ok(checks.every((index) => index > 0 && index < firstSideEffect));
  }],

  ['the native grammar pauses while the reasoning budget is inside a block', async () => {
    const gate = functionBody('bool grammar_is_active()');
    assert.match(gate, /return state == REASONING_BUDGET_IDLE \|\| state == REASONING_BUDGET_DONE;/);
    const sample = functionBody('llama_token sample_next_token()');
    const gateIndex = sample.indexOf('const bool use_grammar = grammar_is_active();');
    assert.ok(gateIndex >= 0, 'the gate is read once per token');
    assert.ok(gateIndex < sample.indexOf('llama_sampler_accept(g_active_sampler, token);'),
      'the gate is read before the budget accepts the token');
    assert.doesNotMatch(sample, /g_active_grammar != nullptr/);
  }],
];

for (const [name, run] of CASES) {
  try {
    await run();
  } catch (error) {
    console.error(`FAIL: ${name}`);
    throw error;
  }
  console.log(`ok - ${name}`);
}
