import assert from 'node:assert/strict';

import { completionCapabilitiesFrom } from '../../js/src/internal/completion_options.ts';
import { LlamaWebGpuBridge } from '../../js/src/llama_webgpu_bridge.js';
import { createDirectBridge, createWorkerBridge } from './bridge_operation_queue_fixtures.mjs';
import { readNativeCoreSource } from './native_core_source.mjs';

const coreSource = readNativeCoreSource();
const SUPPORTED = { presencePenalty: true, minP: true };
const NONE = { presencePenalty: false, minP: false };

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
];

const CASES = [
  ['begin_generation receives minP and presencePenalty after the existing arguments', async () => {
    const { bridge, begins } = directBridge(coreCapabilityJson());
    assert.equal(await bridge.createCompletion('hello', { minP: 0.05, presencePenalty: -1.5, seed: 3 }), 'a');

    const names = nativeParameterNames('llamadart_webgpu_begin_generation');
    assert.deepEqual(names.slice(-2), ['min_p', 'presence_penalty']);
    const [{ argTypes, args }] = begins;
    assert.equal(argTypes.length, names.length);
    assert.equal(args.length, names.length);
    assert.equal(args[names.indexOf('min_p')], 0.05);
    assert.equal(args[names.indexOf('presence_penalty')], -1.5);
    assert.equal(argTypes[names.indexOf('min_p')], 'number');
    assert.equal(argTypes[names.indexOf('presence_penalty')], 'number');
    assert.equal(args[names.indexOf('seed')], 3);
  }],

  ['omitted, null and zero options pass 0 and need no capability', async () => {
    const { bridge, begins } = directBridge(undefined);
    await bridge.createCompletion('hello');
    await bridge.createCompletion('hello', { minP: null, presencePenalty: null });
    await bridge.createCompletion('hello', { minP: 0, presencePenalty: 0 });
    assert.deepEqual(begins.map(({ args }) => args.slice(-2)), [[0, 0], [0, 0], [0, 0]]);
  }],

  ['the range ends are accepted', async () => {
    const { bridge, begins } = directBridge(JSON.stringify(SUPPORTED));
    await bridge.createCompletion('hello', { minP: 1, presencePenalty: 1e6 });
    assert.deepEqual(begins[0].args.slice(-2), [1, 1e6]);
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
    await bridge.createCompletion('hello', { minP: 0.1, presencePenalty: 0.5 });
    assert.equal(structuredClone(sent).minP, 0.1);
    assert.equal(structuredClone(sent).presencePenalty, 0.5);
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
    assert.deepEqual(begins, []);
  }],

  ['a core reporting one option rejects only the other', async () => {
    const { bridge, begins } = directBridge(JSON.stringify({ minP: true }));
    assert.deepEqual(await bridge.getCompletionCapabilities(), { presencePenalty: false, minP: true });
    await bridge.createCompletion('hello', { minP: 0.1 });
    await assert.rejects(
      bridge.createCompletion('hello', { presencePenalty: 1 }),
      /does not support CompletionOptions\.presencePenalty\.$/,
    );
    assert.equal(begins.length, 1);
  }],

  ['the core literal reports both options', async () => {
    assert.deepEqual(completionCapabilitiesFrom(coreCapabilityJson()), SUPPORTED);
    const { bridge } = directBridge(coreCapabilityJson());
    assert.deepEqual(await bridge.getCompletionCapabilities(), SUPPORTED);
  }],

  ['a malformed or non-boolean capability response reports nothing', async () => {
    assert.deepEqual(completionCapabilitiesFrom('{'), NONE);
    assert.deepEqual(completionCapabilitiesFrom(null), NONE);
    assert.deepEqual(completionCapabilitiesFrom('{"minP":1,"presencePenalty":"true"}'), NONE);
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
    ];
    const firstSideEffect = body.indexOf('end_generation_state();');
    assert.ok(checks.every((index) => index > 0 && index < firstSideEffect));
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
