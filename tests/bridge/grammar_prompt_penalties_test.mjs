import assert from 'node:assert/strict';
import { test } from 'node:test';
import { completionCapabilitiesFrom, NO_COMPLETION_CAPABILITIES } from '../../js/src/internal/completion_options.ts';
import { createDirectBridge, createWorkerBridge } from './bridge_operation_queue_fixtures.mjs';

const GRAMMAR = 'root ::= "yes"';
const OPT_IN = { grammar: GRAMMAR, penaltyPromptTokensWithGrammar: true };
function direct(capability) {
  const { bridge, core } = createDirectBridge({ hello: ['a'] });
  const begins = [];
  const call = core.ccall.bind(core);
  if (capability !== undefined) core._llamadart_webgpu_completion_capabilities_json = () => 0;
  core.ccall = (name, returns, types, args, options) => {
    if (name === 'llamadart_webgpu_completion_capabilities_json') return JSON.stringify(capability);
    if (name === 'llamadart_webgpu_begin_generation_bytes') begins.push(args);
    return call(name, returns, types, args, options);
  };
  return { bridge, core, begins };
}

for (const worker of [false, true]) {
  function setup(capability) {
    const result = direct(capability);
    if (worker) {
      const runtime = result.bridge._runtime;
      result.bridge = createWorkerBridge({
        _callWorker: async (method, args) => runtime[method](...structuredClone(args)),
      });
    }
    return result;
  }
  test(`${worker ? 'worker' : 'direct'} opt-in maps only to the existing prompt-penalty mode slot`, async () => {
    const { bridge, begins } = setup({ penaltyPromptTokensWithGrammar: true });
    await bridge.createCompletion('hello', OPT_IN);
    await bridge.createCompletion('hello', { grammar: GRAMMAR });
    await bridge.createCompletion('hello', { penaltyPromptTokens: false });
    await bridge.createCompletion('hello', { grammar: '', penaltyPromptTokensWithGrammar: true });
    assert.deepEqual(begins.map((args) => args.at(-1)), [2, 0, 1, 2]);
    assert.ok(begins.every((args) => args.length === 15), 'existing C export arity is unchanged');
  });
  for (const capability of [undefined, {}, { penaltyPromptTokensWithGrammar: false }, { penaltyPromptTokensWithGrammar: 1 }, { penaltyPromptTokensWithGrammar: 'true' }]) {
    test(`${worker ? 'worker' : 'direct'} refuses unsupported ${JSON.stringify(capability)} before generation`, async () => {
      const { bridge, begins } = setup(capability);
      await assert.rejects(bridge.createCompletion('hello', OPT_IN), /does not support CompletionOptions\.penaltyPromptTokensWithGrammar/);
      assert.deepEqual(begins, []);
      assert.equal(await bridge.createCompletion('hello'), 'a');
    });
  }
  test(`${worker ? 'worker' : 'direct'} rejects invalid opt-ins and remains usable`, async () => {
    const { bridge, begins } = setup({ penaltyPromptTokensWithGrammar: true });
    for (const value of [1, 'true', {}, []]) {
      await assert.rejects(bridge.createCompletion('hello', { ...OPT_IN, penaltyPromptTokensWithGrammar: value }), /must be a boolean/);
    }
    await assert.rejects(bridge.createCompletion('hello', { penaltyPromptTokensWithGrammar: true }), /needs a CompletionOptions\.grammar string/);
    await assert.rejects(bridge.createCompletion('hello', { ...OPT_IN, penaltyPromptTokens: false }), /requires penaltyPromptTokens/);
    await assert.rejects(bridge.createCompletion('hello', { ...OPT_IN, parts: [{ type: 'image', bytes: new Uint8Array([1]) }] }), /text-only prompts/);
    assert.deepEqual(begins, []);
    assert.equal(await bridge.createCompletion('hello'), 'a');
  });
}

test('only the loaded core literal true advertises grammar prompt penalties', () => {
  assert.equal(NO_COMPLETION_CAPABILITIES.penaltyPromptTokensWithGrammar, false);
  assert.equal(completionCapabilitiesFrom('{"penaltyPromptTokensWithGrammar":true}').penaltyPromptTokensWithGrammar, true);
  for (const raw of [null, '{', '{}', '{"penaltyPromptTokensWithGrammar":false}', '{"penaltyPromptTokensWithGrammar":1}']) {
    assert.equal(completionCapabilitiesFrom(raw).penaltyPromptTokensWithGrammar, false);
  }
});

for (const flag of [undefined, false, 1, 'true']) {
  test(`an older worker wrapper cannot silently ignore opt-in with flag ${String(flag)}`, async () => {
    const calls = [];
    const bridge = createWorkerBridge({
      _callWorker: async (method) => {
        calls.push(method);
        return method === 'getCompletionCapabilities'
          ? { penaltyPromptTokens: true, penaltyPromptTokensWithGrammar: flag }
          : 'ignored';
      },
    });
    assert.equal((await bridge.getCompletionCapabilities()).penaltyPromptTokensWithGrammar, false);
    calls.length = 0;
    await assert.rejects(bridge.createCompletion('hello', OPT_IN), /penaltyPromptTokensWithGrammar/);
    assert.deepEqual(calls, ['getCompletionCapabilities']);
  });
}
