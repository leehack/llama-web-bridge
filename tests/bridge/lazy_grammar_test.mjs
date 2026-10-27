import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { GRAMMAR_TRIGGER_TYPES } from '../../js/src/internal/completion_options.ts';
import { LAZY_GRAMMAR_REJECTED_ERROR_TEXT } from '../../js/src/internal/constants.ts';
import { createDirectBridge, createWorkerBridge } from './bridge_operation_queue_fixtures.mjs';
import { readNativeCoreSource } from './native_core_source.mjs';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const coreSource = readNativeCoreSource();
const cmake = readFileSync(path.join(rootDir, 'CMakeLists.txt'), 'utf8');
const wrappers = readFileSync(path.join(rootDir, 'src/llama_webgpu_grammar.cpp'), 'utf8');
const dts = readFileSync(path.join(rootDir, 'js/src/llama_webgpu_bridge.d.ts'), 'utf8');

const GRAMMAR = 'root ::= "<tool_call>" call\ncall ::= "yes" | "no"';
const LAZY = {
  grammar: GRAMMAR,
  grammarLazy: true,
  grammarTriggers: [{ type: 'word', value: '<tool_call>' }],
};
const GRAMMAR_CALLS = new Set([
  'llamadart_webgpu_set_next_grammar',
  'llamadart_webgpu_add_next_grammar_trigger',
]);

function functionBody(signature) {
  const start = coreSource.indexOf(signature);
  assert.ok(start >= 0, `${signature} not found in the native core`);
  return coreSource.slice(start, coreSource.indexOf('\n}\n', start));
}

// A direct bridge whose stub core reports capabilityJson and records the
// grammar and begin calls in order. coreErrors maps an export name to the
// return code and last error it fails with.
function directBridge(capabilityJson = '{"lazyGrammar":true}', coreErrors = {}) {
  const { bridge, core } = createDirectBridge({ hello: ['a'] });
  const calls = [];
  const ccall = core.ccall.bind(core);
  core._llamadart_webgpu_completion_capabilities_json = () => 0;
  core.ccall = (name, returnType, argTypes, args, options) => {
    if (name === 'llamadart_webgpu_completion_capabilities_json') {
      return capabilityJson;
    }
    if (GRAMMAR_CALLS.has(name)) {
      calls.push([name, argTypes, args]);
      if (coreErrors[name]) {
        core.lastError = coreErrors[name];
        return -1;
      }
      return 0;
    }
    if (name === 'llamadart_webgpu_begin_generation_bytes') {
      calls.push([name, args[6]]);
    }
    return ccall(name, returnType, argTypes, args, options);
  };
  return { bridge, core, calls };
}

const INVALID_TYPES = [
  [{ grammarLazy: 'true' }, /CompletionOptions\.grammarLazy must be a boolean; got true\./],
  [{ grammarLazy: 1 }, /CompletionOptions\.grammarLazy must be a boolean/],
  [{ grammarTriggers: { type: 'word', value: 'a' } }, /CompletionOptions\.grammarTriggers must be an array\./],
  [{ grammarTriggers: [null] }, /CompletionOptions\.grammarTriggers\[0\] must be an object\./],
  [{ grammarTriggers: ['<tool_call>'] }, /CompletionOptions\.grammarTriggers\[0\] must be an object\./],
  [{ grammarTriggers: [{ type: 'regex', value: 'a' }] },
    /CompletionOptions\.grammarTriggers\[0\]\.type must be one of token, word, pattern, pattern-full; got "regex"\./],
  [{ grammarTriggers: [{ type: 1, value: 'a' }] }, /grammarTriggers\[0\]\.type must be one of/],
  [{ grammarTriggers: [{ type: 'toString', value: 'a' }] }, /grammarTriggers\[0\]\.type must be one of/],
  [{ grammarTriggers: [{ type: 'word', value: 'a' }, { type: 'word', value: '' }] },
    /CompletionOptions\.grammarTriggers\[1\]\.value must be a non-empty string\./],
  [{ grammarTriggers: [{ type: 'pattern' }] }, /grammarTriggers\[0\]\.value must be a non-empty string/],
  [{ grammarTriggers: [{ type: 'pattern-full', value: 3 }] }, /grammarTriggers\[0\]\.value must be a non-empty string/],
  [{ grammarTriggers: [{ type: 'word', value: 'a\0' }] },
    /CompletionOptions\.grammarTriggers\[0\]\.value contains U\+0000/],
  [{ grammarRoot: '' }, /CompletionOptions\.grammarRoot must be a non-empty string\./],
  [{ grammarRoot: 7 }, /CompletionOptions\.grammarRoot must be a non-empty string\./],
  [{ grammarRoot: 'ro\0ot' }, /CompletionOptions\.grammarRoot contains U\+0000/],
];

const INVALID_RANGES = [
  [{ grammarTriggers: [{ type: 'token', token: -1 }] },
    /CompletionOptions\.grammarTriggers\[0\]\.token must be an integer from 0 to 2147483647; got -1\./],
  [{ grammarTriggers: [{ type: 'token', token: 1.5 }] }, /grammarTriggers\[0\]\.token must be an integer/],
  [{ grammarTriggers: [{ type: 'token', token: 2147483648 }] }, /grammarTriggers\[0\]\.token must be an integer/],
  [{ grammarTriggers: [{ type: 'token', token: '5' }] }, /grammarTriggers\[0\]\.token must be an integer/],
  [{ grammarTriggers: [{ type: 'token', value: '<tool_call>' }] }, /grammarTriggers\[0\]\.token must be an integer/],
];

const INVALID_COMBINATIONS = [
  [{ grammarLazy: true, grammarTriggers: LAZY.grammarTriggers },
    /^CompletionOptions\.grammarLazy needs a non-empty CompletionOptions\.grammar\.$/],
  [{ ...LAZY, grammar: '' }, /^CompletionOptions\.grammarLazy needs a non-empty CompletionOptions\.grammar\.$/],
  [{ grammarRoot: 'call' }, /^CompletionOptions\.grammarRoot needs a non-empty CompletionOptions\.grammar\.$/],
  [{ ...LAZY, grammarTriggers: [] }, /^CompletionOptions\.grammarLazy needs at least one CompletionOptions\.grammarTriggers entry\.$/],
  [{ ...LAZY, grammarTriggers: null }, /grammarLazy needs at least one/],
];

const errorName = (error) => error.constructor.name;

const CASES = [
  ['a lazy grammar reaches the core before begin_generation with llama.cpp trigger types', async () => {
    const { bridge, calls } = directBridge();
    assert.equal(
      await bridge.createCompletion('hello', {
        grammar: GRAMMAR,
        grammarLazy: true,
        grammarRoot: 'call',
        grammarTriggers: [
          { type: 'word', value: '<tool_call>' },
          { type: 'pattern', value: '[\\s\\S]*?(<tool_call>)' },
          { type: 'pattern-full', value: '.*<tool_call>' },
          { type: 'token', token: 7 },
        ],
      }),
      'a',
    );
    const add = 'llamadart_webgpu_add_next_grammar_trigger';
    const types = ['number', 'string', 'number'];
    assert.deepEqual(calls, [
      ['llamadart_webgpu_set_next_grammar', ['string', 'number'], ['call', 1]],
      [add, types, [1, '<tool_call>', -1]],
      [add, types, [2, '[\\s\\S]*?(<tool_call>)', -1]],
      [add, types, [3, '.*<tool_call>', -1]],
      [add, types, [0, '', 7]],
      ['llamadart_webgpu_begin_generation_bytes', GRAMMAR],
    ]);
  }],

  ['trigger wire values are common_grammar_trigger_type, which the core takes as is', async () => {
    // llama.cpp common/common.h: TOKEN, WORD, PATTERN, PATTERN_FULL.
    assert.deepEqual(GRAMMAR_TRIGGER_TYPES, { token: 0, word: 1, pattern: 2, 'pattern-full': 3 });
    const add = functionBody('EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_add_next_grammar_trigger(');
    assert.match(add, /trigger\.type = static_cast<common_grammar_trigger_type>\(type\);/);
    assert.deepEqual(
      [...add.matchAll(/case (COMMON_GRAMMAR_TRIGGER_TYPE_\w+):/g)].map(([, name]) => name),
      ['COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN', 'COMMON_GRAMMAR_TRIGGER_TYPE_WORD',
        'COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN', 'COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL'],
    );
    assert.match(add, /default:\s*g_next_grammar = grammar_request\(\);\s*set_error\("Unknown grammar trigger type: "/);
    const union = dts.match(/export type GrammarTrigger =([\s\S]+?);\n\n/)[1];
    assert.deepEqual(
      [...union.matchAll(/'([a-z-]+)'/g)].map(([, name]) => name).sort(),
      Object.keys(GRAMMAR_TRIGGER_TYPES).sort(),
    );
  }],

  ['a strict grammar starting at root keeps the old call', async () => {
    const { bridge, calls } = directBridge();
    await bridge.createCompletion('hello', { grammar: GRAMMAR });
    await bridge.createCompletion('hello', { grammar: GRAMMAR, grammarLazy: false, grammarRoot: 'root' });
    await bridge.createCompletion('hello', { grammar: GRAMMAR, grammarLazy: null, grammarTriggers: null, grammarRoot: null });
    // Triggers only activate a lazy grammar, as in llama.cpp.
    await bridge.createCompletion('hello', { grammar: GRAMMAR, grammarTriggers: LAZY.grammarTriggers });
    await bridge.createCompletion('hello');
    assert.deepEqual(calls.map(([name]) => name), Array(5).fill('llamadart_webgpu_begin_generation_bytes'));
  }],

  ['a strict grammar with another root sends the root without triggers', async () => {
    const { bridge, calls } = directBridge();
    await bridge.createCompletion('hello', { grammar: GRAMMAR, grammarRoot: 'call', grammarTriggers: LAZY.grammarTriggers });
    assert.deepEqual(calls, [
      ['llamadart_webgpu_set_next_grammar', ['string', 'number'], ['call', 0]],
      ['llamadart_webgpu_begin_generation_bytes', GRAMMAR],
    ]);
  }],

  ['invalid options reject with typed errors before the direct runtime touches the core', async () => {
    const { bridge, core, calls } = directBridge();
    const cases = [
      ...INVALID_TYPES.map(([options, message]) => [{ ...LAZY, ...options }, message, 'TypeError']),
      ...INVALID_RANGES.map(([options, message]) => [{ ...LAZY, ...options }, message, 'RangeError']),
      ...INVALID_COMBINATIONS.map(([options, message]) => [options, message, 'Error']),
    ];
    for (const [options, message, name] of cases) {
      await assert.rejects(bridge.createCompletion('hello', options), (error) => {
        assert.equal(errorName(error), name, message.source);
        assert.match(error.message, message);
        return true;
      });
    }
    assert.deepEqual(calls, []);
    assert.deepEqual(core.trace, []);
    assert.equal(await bridge.createCompletion('hello', LAZY), 'a', 'the bridge stays usable');
  }],

  ['invalid options reject before a worker request', async () => {
    const sent = [];
    const bridge = createWorkerBridge({
      _callWorker: async (method) => {
        sent.push(method);
        return 'x';
      },
    });
    for (const [options, , name] of [
      ...INVALID_TYPES.map(([options]) => [{ ...LAZY, ...options }, null, TypeError]),
      ...INVALID_RANGES.map(([options]) => [{ ...LAZY, ...options }, null, RangeError]),
      ...INVALID_COMBINATIONS.map(([options]) => [options, null, Error]),
    ]) {
      await assert.rejects(bridge.createCompletion('hello', options), name);
    }
    assert.deepEqual(sent, []);
  }],

  ['a worker completion forwards the grammar options', async () => {
    let sent = null;
    const bridge = createWorkerBridge({
      _callWorker: async (method, args) => {
        assert.equal(method, 'createCompletion');
        sent = structuredClone(args[1]);
        return 'x';
      },
    });
    const triggers = [{ type: 'word', value: '<tool_call>' }, { type: 'token', token: 3 }];
    await bridge.createCompletion('hello', { grammar: GRAMMAR, grammarLazy: true, grammarTriggers: triggers, grammarRoot: 'call' });
    assert.equal(sent.grammarLazy, true);
    assert.deepEqual(sent.grammarTriggers, triggers);
    assert.equal(sent.grammarRoot, 'call');
  }],

  ['a core without lazyGrammar rejects grammarLazy and grammarRoot but keeps strict grammars', async () => {
    for (const capabilities of [null, '{"lazyGrammar":1}', '{"minP":true}']) {
      const { bridge, calls } = directBridge(capabilities);
      await assert.rejects(
        bridge.createCompletion('hello', LAZY),
        /^Error: The loaded WebGPU core does not support CompletionOptions\.grammarLazy\.$/,
      );
      await assert.rejects(
        bridge.createCompletion('hello', { grammar: GRAMMAR, grammarRoot: 'call' }),
        /does not support CompletionOptions\.grammarRoot\.$/,
      );
      await assert.rejects(
        bridge.createCompletion('hello', { ...LAZY, grammarRoot: 'call' }),
        /does not support CompletionOptions\.grammarLazy or CompletionOptions\.grammarRoot\.$/,
      );
      assert.equal(await bridge.createCompletion('hello', { grammar: GRAMMAR, grammarRoot: 'root' }), 'a');
      assert.deepEqual(calls.map(([name]) => name), ['llamadart_webgpu_begin_generation_bytes']);
    }
  }],

  ['a core rejection of the grammar options fails the completion before begin_generation', async () => {
    for (const name of GRAMMAR_CALLS) {
      const { bridge, calls } = directBridge(undefined, { [name]: 'nope' });
      await assert.rejects(
        bridge.createCompletion('hello', LAZY),
        /^Error: Failed to configure the grammar: nope$/,
      );
      assert.ok(!calls.some(([called]) => called === 'llamadart_webgpu_begin_generation_bytes'), name);
    }
  }],

  ['the core consumes the grammar request in every begin and resets it with the runtime', async () => {
    const begin = functionBody('int32_t begin_generation_impl(');
    const consume = begin.indexOf('const grammar_request grammar_options = std::move(g_next_grammar);\n  g_next_grammar = grammar_request();');
    assert.ok(consume >= 0 && consume < begin.indexOf('if (g_generation_active) {'),
      'a rejected begin still consumes the request');
    assert.match(functionBody('void free_runtime()'), /g_next_grammar = grammar_request\(\);/);
    assert.match(functionBody('void end_generation_state()'), /g_active_grammar_lazy = false;/);
    const set = functionBody('EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_set_next_grammar(');
    assert.match(set, /g_next_grammar = grammar_request\(\);\s*if \(root == nullptr \|\| root\[0\] == '\\0'\) \{/);
    const exported = cmake.match(/-sEXPORTED_FUNCTIONS=\[([^\]]*)\]/)[1];
    for (const name of GRAMMAR_CALLS) {
      assert.ok(exported.includes(`'_${name}'`), `${name} is not in EXPORTED_FUNCTIONS`);
    }
  }],

  ['the core builds a lazy grammar from triggers as common_sampler_init does', async () => {
    const begin = functionBody('int32_t begin_generation_impl(');
    assert.match(
      begin,
      /if \(grammar_options\.lazy &&\s*!lazy_grammar_triggers\([^;]*\)\) \{\s*set_error\(\s*"Failed to initialize sampler chain \(invalid grammar\): " \+\s*grammar_error\);\s*return -5;/,
      'a bad trigger is an invalid grammar, which the facade never retries on the main thread',
    );
    assert.match(
      begin,
      /llamadart_webgpu_grammar_sampler_init\(\s*g_state\.vocab,\s*grammar,\s*grammar_options\.root\.c_str\(\),\s*grammar_options\.lazy,/,
    );
    assert.match(begin, /g_active_grammar_lazy = g_active_grammar != nullptr && grammar_options\.lazy;/);
    const triggers = functionBody('bool lazy_grammar_triggers(');
    assert.match(triggers, /if \(triggers\.empty\(\)\) \{/);
    assert.match(triggers, /case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:\s*patterns\.push_back\(regex_escape\(trigger\.value\)\);/);
    assert.match(triggers, /case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:\s*patterns\.push_back\(trigger\.value\);/);
    assert.match(
      triggers,
      /std::string anchored = "\^\$";\s*if \(!pattern\.empty\(\)\) \{\s*anchored = \(pattern\.front\(\) != '\^' \? "\^" : ""\) \+ pattern \+\s*\(pattern\.back\(\) != '\$' \? "\$" : ""\);/,
    );
    assert.match(triggers, /if \(trigger\.token < 0 \|\| trigger\.token >= n_vocab\) \{/);
  }],

  ['only a lazy grammar accepts through the catching wrapper', async () => {
    const sample = functionBody('llama_token sample_next_token(');
    assert.match(
      sample,
      /if \(!g_active_grammar_lazy\) \{\s*llama_sampler_accept\(g_active_grammar, token\);\s*\} else \{\s*std::string grammar_error;\s*if \(!llamadart_webgpu_grammar_sampler_accept\(\s*g_active_grammar, token, &grammar_error\)\) \{\s*set_error\([^;]*\);\s*return LLAMA_TOKEN_NULL;/,
    );
    assert.ok(
      sample.includes(`"${LAZY_GRAMMAR_REJECTED_ERROR_TEXT}: "`),
      'the facade recognizes the core error and keeps the worker',
    );
    for (const name of ['llamadart_webgpu_grammar_sampler_init', 'llamadart_webgpu_grammar_sampler_accept']) {
      assert.match(wrappers, new RegExp(`extern "C" __attribute__\\(\\(noinline\\)\\)[^\\n]*\\n?${name}\\(`), `${name} must stay noinline`);
      assert.match(cmake, new RegExp(`-sEXCEPTION_CATCHING_ALLOWED=\\[[^\\]]*'${name}'`), `${name} must catch`);
    }
    assert.match(
      wrappers,
      /std::regex pattern\(trigger_patterns\[i\]\);\s*\} catch \(const std::regex_error & error\) \{/,
      'a bad pattern is named before llama.cpp compiles it',
    );
  }],

  ['the core reports lazyGrammar', async () => {
    const body = functionBody('llamadart_webgpu_completion_capabilities_json()');
    assert.match(body, /\\"lazyGrammar\\":true,/);
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
