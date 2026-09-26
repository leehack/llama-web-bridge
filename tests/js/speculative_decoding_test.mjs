import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  SPECULATIVE_DECODING_STRATEGIES,
  resolveCompletionSamplingOptions,
} from '../../js/src/internal/completion_options.ts';
import * as smoke from '../../scripts/speculative_browser_smoke.mjs';
import { createDirectBridge, createWorkerBridge } from './bridge_operation_queue_fixtures.mjs';
import { readNativeCoreSource } from './native_core_source.mjs';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const coreSource = readNativeCoreSource();
const cmake = readFileSync(path.join(rootDir, 'CMakeLists.txt'), 'utf8');
const wrappers = readFileSync(path.join(rootDir, 'src/llama_webgpu_speculative.cpp'), 'utf8');
const dts = readFileSync(path.join(rootDir, 'js/src/llama_webgpu_bridge.d.ts'), 'utf8');

const ALL_STRATEGIES = Object.fromEntries(SPECULATIVE_DECODING_STRATEGIES.map((name) => [name, true]));
const CAPABILITIES = JSON.stringify({ minP: true, presencePenalty: true, thinkingBudget: true, speculativeDecoding: ALL_STRATEGIES });
const SPECULATIVE_USAGE = { draftTokens: 9, acceptedDraftTokens: 6, draftAttempts: 4, verifyTokens: 13, replayTokens: 2 };

function resolve(speculativeDecoding, extra = {}) {
  return resolveCompletionSamplingOptions({ speculativeDecoding, ...extra }).speculative;
}

function functionBody(signature) {
  const start = coreSource.indexOf(signature);
  assert.ok(start >= 0, `${signature} not found in the native core`);
  return coreSource.slice(start, coreSource.indexOf('\n}\n', start));
}

function nativeParameterNames(name) {
  const match = coreSource.match(new RegExp(`EMSCRIPTEN_KEEPALIVE\\s+int32_t\\s+${name}\\(([^)]*)\\)`));
  assert.ok(match, `${name} is not exported`);
  return match[1].split(',').map((parameter) => parameter.trim().match(/(\w+)$/)[1]);
}

// A direct bridge whose stub core reports every strategy, records the
// speculative setter and begin calls in order, and reports `usage`.
function directBridge({ capabilities = CAPABILITIES, usage = null, setterRc = 0 } = {}) {
  const { bridge, core } = createDirectBridge({ hello: ['a', 'b'] });
  const calls = [];
  const ccall = core.ccall.bind(core);
  core.ccall = (name, returnType, argTypes, args) => {
    if (name === 'llamadart_webgpu_completion_capabilities_json') {
      return capabilities;
    }
    if (name === 'llamadart_webgpu_set_next_speculative') {
      calls.push({ name, argTypes, args, files: [...core.files()] });
      if (setterRc !== 0) core.lastError = 'Unknown speculative decoding strategy: x';
      return setterRc;
    }
    if (name === 'llamadart_webgpu_begin_generation') {
      calls.push({ name, files: [...core.files()] });
    }
    if (name === 'llamadart_webgpu_last_generation_usage_json' && usage) {
      return JSON.stringify({ ...core.usage, speculative: usage });
    }
    return ccall(name, returnType, argTypes, args);
  };
  const files = new Map();
  core.FS.writeFile = (file, bytes) => files.set(file, bytes);
  core.FS.unlink = (file) => files.delete(file);
  core.files = () => files.keys();
  core.fileBytes = files;
  core._llamadart_webgpu_completion_capabilities_json = () => 0;
  return { bridge, core, calls };
}

const CASES = [
  ['draftTokenMax resolves per strategy as native llamadart resolves it', () => {
    assert.equal(resolve({ strategies: ['draft-simple'] }).draftTokenMax, 3);
    assert.equal(resolve({ strategies: ['draft-mtp'], draftTokenMax: 5 }).draftTokenMax, 5);
    assert.equal(resolve({ strategies: ['ngram-simple'] }).draftTokenMax, 48);
    assert.equal(resolve({ strategies: ['ngram-map-k'], ngramSizeM: 8 }).draftTokenMax, 8);
    assert.equal(resolve({ strategies: ['ngram-cache'] }).draftTokenMax, 8);
    const mod = resolve({ strategies: ['ngram-mod'] });
    assert.equal(mod.draftTokenMax, 64);
    assert.equal(mod.ngramTokenMax, 0);
    const modWithMax = resolve({ strategies: ['ngram-mod'], draftTokenMax: 10 });
    assert.equal(modWithMax.draftTokenMax, 10);
    assert.equal(modWithMax.ngramTokenMax, 10, 'ngram-mod inherits draftTokenMax');
    assert.equal(resolve({ strategies: ['ngram-mod'], ngramTokenMax: 16, draftTokenMax: 4 }).draftTokenMax, 16);
    assert.equal(resolve({ strategies: ['ngram-mod'], ngramTokenMax: 0 }).draftTokenMax, 64, 'a zero maximum falls back to 64');
    assert.equal(resolve({ strategies: ['draft-simple'], draftTokenMax: 0 }).draftTokenMax, 64, 'as native llamadart resolves it');
    assert.equal(resolve({ strategies: ['ngram-simple'], draftTokenMax: 4 }).ngramTokenMax, 0);
    assert.equal(resolve({ strategies: ['ngram-mod', 'draft-simple'] }).draftTokenMax, 64);
  }],

  ['strategies keep their order without duplicates and unset knobs keep llama.cpp defaults', () => {
    assert.deepEqual(resolve({ strategies: ['ngram-mod', 'draft-eagle3', 'ngram-mod'] }), {
      strategies: ['ngram-mod', 'draft-eagle3'],
      draftTokenMax: 64,
      draftTokenMin: 0,
      minProbability: -1,
      draftSplitProbability: -1,
      ngramSizeN: 0,
      ngramSizeM: 0,
      ngramMinHits: 0,
      ngramMatch: 0,
      ngramTokenMin: -1,
      ngramTokenMax: 0,
      ngramCacheStatic: null,
      ngramCacheDynamic: null,
    });
    assert.equal(resolve(null), null);
    assert.equal(resolve(undefined), null);
  }],

  ['invalid configurations reject with the native llamadart rules', () => {
    const invalid = [
      [{ strategies: [] }, TypeError, /strategies must be a non-empty array/],
      [{ strategies: 'ngram-mod' }, TypeError, /strategies must be a non-empty array/],
      [{ strategies: ['mtp'] }, TypeError, /unknown strategy "mtp"/],
      [{ strategies: ['ngram-mod'], draftTokenMax: -1 }, RangeError, /draftTokenMax must be an integer from 0/],
      [{ strategies: ['ngram-mod'], draftTokenMax: 1.5 }, RangeError, /draftTokenMax/],
      [{ strategies: ['draft-simple'], minProbability: 1.5 }, RangeError, /minProbability must be a number from 0 to 1/],
      [{ strategies: ['draft-simple'], draftSplitProbability: Number.NaN }, RangeError, /draftSplitProbability/],
      [{ strategies: ['ngram-simple'], ngramSizeN: 65536 }, RangeError, /ngramSizeN must be an integer from 1 to 65535/],
      [{ strategies: ['ngram-mod'], ngramTokenMin: 9, ngramTokenMax: 8 }, RangeError, /ngramTokenMin \(9\) must not exceed ngramTokenMax \(8\)/],
      [{ strategies: ['draft-simple'], draftTokenMin: 4 }, RangeError, /draftTokenMin \(4\) must not exceed the draft token maximum \(3\)/],
      [{ strategies: ['ngram-cache'], ngramCacheStatic: 3 }, TypeError, /ngramCacheStatic must be a URL string/],
      [{ strategies: ['ngram-cache'], ngramCacheStatic: new Uint8Array() }, TypeError, /ngramCacheStatic is empty/],
      [{ strategies: ['draft-simple', 'draft-mtp'] }, Error, /at most one draft-\* strategy/],
      [{ strategies: ['ngram-mod'], draftTokenMin: 1 }, Error, /do not support draftTokenMin/],
      [{ strategies: ['ngram-mod'], minProbability: 0.5 }, Error, /minProbability/],
      [{ strategies: ['ngram-mod'], ngramCacheStatic: 'cache.lcs' }, Error, /need the ngram-cache strategy/],
    ];
    for (const [config, type, message] of invalid) {
      assert.throws(() => resolve(config), (error) => {
        assert.equal(error.constructor, type, `${JSON.stringify(config)} threw ${error.name}`);
        assert.match(error.message, message);
        return true;
      });
    }
    assert.throws(() => resolve(8), TypeError);
    const mtp = { strategies: ['draft-mtp'] };
    assert.throws(() => resolve(mtp, { parts: [{ type: 'image', bytes: new Uint8Array([1]) }] }), /text-only prompts/);
    assert.throws(
      () => resolve(mtp, { thinkingBudget: { maxTokens: 1, startTag: '<think>', endTag: '</think>' } }),
      /cannot be combined with CompletionOptions\.thinkingBudget/,
    );
    assert.throws(() => resolve(mtp, { grammar: 'root ::= "a"' }), /does not support CompletionOptions\.grammar/);
    assert.equal(resolve(mtp, { grammar: '' }).strategies[0], 'draft-mtp');
  }],

  ['the core consumes the settings in the begin call that follows them', async () => {
    const { bridge, core, calls } = directBridge({ usage: SPECULATIVE_USAGE });
    let usage = null;
    const cache = new Int32Array([1, 2, -1, -1, 1, 3, 1]);
    const text = await bridge.createCompletion('hello', {
      nPredict: 7,
      speculativeDecoding: {
        strategies: ['ngram-cache', 'draft-simple'],
        draftTokenMin: 1,
        minProbability: 0.25,
        draftSplitProbability: 0.5,
        ngramSizeN: 3,
        ngramSizeM: 5,
        ngramMinHits: 2,
        ngramMatch: 4,
        ngramTokenMin: 1,
        ngramTokenMax: 6,
        ngramCacheStatic: cache,
        ngramCacheDynamic: cache.buffer,
      },
      onUsage: (value) => {
        usage = value;
      },
    });
    assert.equal(text, 'ab');
    assert.deepEqual(calls.map(({ name }) => name), [
      'llamadart_webgpu_set_next_speculative',
      'llamadart_webgpu_begin_generation',
    ]);

    const [setter, begin] = calls;
    const names = nativeParameterNames('llamadart_webgpu_set_next_speculative');
    assert.equal(setter.args.length, names.length);
    assert.equal(setter.argTypes.length, names.length);
    const arg = (name) => setter.args[names.indexOf(name)];
    assert.equal(arg('type_names'), 'ngram-cache,draft-simple');
    assert.deepEqual(
      ['draft_n_max', 'draft_n_min', 'p_min', 'p_split', 'ngram_size_n', 'ngram_size_m', 'ngram_min_hits',
        'ngram_match', 'ngram_n_min', 'ngram_n_max', 'max_tokens'].map(arg),
      [8, 1, 0.25, 0.5, 3, 5, 2, 4, 1, 6, 7],
    );
    assert.deepEqual(
      names.map((name, index) => [name, setter.argTypes[index]]).filter(([, type]) => type === 'string').map(([name]) => name),
      ['type_names', 'cache_static_path', 'cache_dynamic_path'],
    );
    assert.deepEqual(setter.files, [arg('cache_static_path'), arg('cache_dynamic_path')]);
    assert.deepEqual([...begin.files], setter.files, 'the caches stay staged until the begin call');
    assert.deepEqual([...core.files()], [], 'the staged caches are deleted');
    assert.deepEqual(usage.speculative, SPECULATIVE_USAGE);
  }],

  ['n-gram cache bytes reach the core unchanged', async () => {
    const { bridge, core } = directBridge();
    const written = [];
    const writeFile = core.FS.writeFile;
    core.FS.writeFile = (file, bytes) => {
      written.push([...bytes]);
      writeFile(file, bytes);
    };
    const words = new Int32Array([7, 8, -1, -1, 1, 9, 2]);
    const view = new Uint8Array(words.buffer, 4, 8);
    await bridge.createCompletion('hello', { speculativeDecoding: { strategies: ['ngram-cache'], ngramCacheStatic: view } });
    assert.deepEqual(written, [[...view]]);
  }],

  ['a completion without speculative decoding reports no speculative usage and never calls the setter', async () => {
    const { bridge, calls } = directBridge();
    let usage = null;
    await bridge.createCompletion('hello', { onUsage: (value) => { usage = value; } });
    assert.deepEqual(calls.map(({ name }) => name), ['llamadart_webgpu_begin_generation']);
    assert.equal('speculative' in usage, false);
  }],

  ['a rejected setter call rejects before generation and deletes staged caches', async () => {
    const { bridge, core, calls } = directBridge({ setterRc: -1 });
    await assert.rejects(
      bridge.createCompletion('hello', {
        speculativeDecoding: { strategies: ['ngram-cache'], ngramCacheStatic: new Uint8Array([1]) },
      }),
      /Failed to configure speculative decoding: Unknown speculative decoding strategy/,
    );
    assert.deepEqual(calls.map(({ name }) => name), ['llamadart_webgpu_set_next_speculative']);
    assert.deepEqual([...core.files()], []);
  }],

  ['a core without speculative strategies rejects speculative decoding', async () => {
    const { bridge, calls } = directBridge({ capabilities: JSON.stringify({ minP: true }) });
    await assert.rejects(
      bridge.createCompletion('hello', { speculativeDecoding: { strategies: ['ngram-mod'] } }),
      /does not support CompletionOptions\.speculativeDecoding\.$/,
    );
    assert.deepEqual(calls, []);
  }],

  ['a speculative generation failure never reloads the model on the CPU', async () => {
    const { bridge, core } = directBridge();
    core.scripts.hello = [new Error('failed to decode')];
    let recoveries = 0;
    bridge._runtime._shouldAttemptGenerationRecovery = () => true;
    bridge._runtime._recoverGenerationWithCpuFallback = async () => {
      recoveries += 1;
      return false;
    };
    await assert.rejects(
      bridge.createCompletion('hello', { speculativeDecoding: { strategies: ['ngram-mod'] } }),
      /failed to decode/,
    );
    assert.equal(recoveries, 0);
    await assert.rejects(bridge.createCompletion('hello'), /failed to decode/);
    assert.equal(recoveries, 1, 'a plain completion still recovers');
  }],

  ['an invalid configuration rejects before a worker request and a valid one is forwarded', async () => {
    const calls = [];
    const bridge = createWorkerBridge({
      _callWorker: async (method, args) => {
        calls.push([method, structuredClone(args)]);
        return 'x';
      },
    });
    await assert.rejects(bridge.createCompletion('hello', { speculativeDecoding: { strategies: [] } }), TypeError);
    await assert.rejects(
      bridge.createCompletion('hello', { grammar: 'root ::= "a"', speculativeDecoding: { strategies: ['ngram-mod'] } }),
      /does not support CompletionOptions\.grammar/,
    );
    assert.deepEqual(calls, []);
    const bytes = new Uint8Array([1, 2, 3]);
    await bridge.createCompletion('hello', {
      speculativeDecoding: { strategies: ['ngram-cache'], ngramCacheStatic: bytes },
    });
    assert.equal(calls[0][0], 'createCompletion');
    assert.deepEqual([...calls[0][1][1].speculativeDecoding.ngramCacheStatic], [1, 2, 3]);
  }],

  ['a worker draft load sends an absolute URL, forwards progress and is remembered until the model changes', async () => {
    const calls = [];
    const bridge = createWorkerBridge({
      _callWorker: async (method, args, onEvent) => {
        calls.push([method, structuredClone(args)]);
        onEvent?.({ event: 'progress', payload: { loaded: 1, total: 2 } });
        return { architecture: 'llama' };
      },
    });
    const progress = [];
    const controller = new AbortController();
    const info = await bridge.loadDraftModel('https://example.invalid/draft.gguf', {
      useCache: false,
      signal: controller.signal,
      progressCallback: (event) => progress.push(event),
    });
    assert.deepEqual(info, { architecture: 'llama' });
    assert.deepEqual(calls, [['loadDraftModel', ['https://example.invalid/draft.gguf', { useCache: false }]]]);
    assert.deepEqual(progress, [{ loaded: 1, total: 2 }]);
    assert.deepEqual(bridge._loadedDraftModel, { url: 'https://example.invalid/draft.gguf', options: { useCache: false } });

    calls.length = 0;
    bridge._workerModelMissing = false;
    await bridge._loadRememberedModelIntoWorker({ nGpuLayers: 0 });
    assert.deepEqual(calls.map(([method]) => method), ['loadModelFromUrl', 'loadMultimodalProjector', 'loadDraftModel']);
    assert.deepEqual(calls[2][1], ['https://example.invalid/draft.gguf', { useCache: false }]);

    bridge._rememberLoadedModel('other.gguf', {});
    assert.equal(bridge._loadedDraftModel, null, 'a new target model drops the draft');
  }],

  ['a failed draft reload after a worker restart forgets the draft instead of failing the model reload', async () => {
    const warnings = [];
    const bridge = createWorkerBridge({
      _loadedDraftModel: { url: 'https://example.invalid/draft.gguf', options: {} },
      _emitBridgeWarn: (message) => warnings.push(message),
      _callWorker: async (method) => {
        if (method === 'loadDraftModel') throw new Error('Failed to fetch draft model: 404 Not Found');
        return 1;
      },
    });
    await bridge._loadRememberedModelIntoWorker({ nGpuLayers: 0 });
    assert.equal(bridge._loadedDraftModel, null);
    assert.match(warnings.join('\n'), /draft model reload failed .*404 Not Found.*call loadDraftModel again/);
  }],

  ['unloading a draft forgets it', async () => {
    const calls = [];
    const bridge = createWorkerBridge({
      _loadedDraftModel: { url: 'https://example.invalid/draft.gguf', options: {} },
      _callWorker: async (method) => {
        calls.push(method);
      },
    });
    await bridge.unloadDraftModel();
    assert.deepEqual(calls, ['unloadDraftModel']);
    assert.equal(bridge._loadedDraftModel, null);
  }],

  ['a draft that cannot fit names the memory64 core on wasm32', async () => {
    for (const [variant, pattern] of [
      ['wasm32', /The draft model needs about 1 MiB but only 0 MiB of WebAssembly memory is left; the wasm32 core addresses at most 4 GiB; memory64 is required/],
      ['wasm64', /only 0 MiB of WebAssembly memory is left; free memory by unloading other models or use a smaller draft model\.$/],
    ]) {
      const { bridge, core } = directBridge();
      const runtime = bridge._runtime;
      runtime._coreVariant = variant;
      const ccall = core.ccall;
      const trace = [];
      core.ccall = (name, ...rest) => {
        trace.push(name);
        if (name === 'llamadart_webgpu_heap_headroom_bytes') return 1000;
        if (name === 'llamadart_webgpu_draft_model_free') return 0;
        return ccall(name, ...rest);
      };
      runtime._getCachedModelResponse = async () => new Response(new Uint8Array(5000), {
        headers: { 'content-length': '5000' },
      });
      await assert.rejects(runtime.loadDraftModel('https://example.invalid/draft.gguf'), pattern);
      assert.equal(trace.includes('llamadart_webgpu_draft_model_load'), false);
      assert.equal(runtime._draftModel, null);
    }
  }],

  ['a direct draft load stages the file, loads it, deletes it and keeps the target cache fields', async () => {
    const { bridge, core } = directBridge();
    const runtime = bridge._runtime;
    runtime._coreVariant = 'wasm32';
    runtime._modelSource = 'cache';
    runtime._modelCacheState = 'hit';
    runtime._modelCacheName = 'target-cache';
    const opened = [];
    core.FS.open = (file) => {
      opened.push(file);
      return 1;
    };
    const ccall = core.ccall;
    const loads = [];
    core.ccall = (name, returnType, argTypes, args, ...rest) => {
      if (name === 'llamadart_webgpu_heap_headroom_bytes') return 1 << 30;
      if (name === 'llamadart_webgpu_draft_model_free') return 0;
      if (name === 'llamadart_webgpu_draft_model_load') {
        loads.push(args[0]);
        return Promise.resolve(0);
      }
      if (name === 'llamadart_webgpu_draft_model_info_json') return '{"architecture":"dflash","strategy":"draft-dspark"}';
      return ccall(name, returnType, argTypes, args, ...rest);
    };
    runtime._getCachedModelResponse = async (_url, options) => {
      assert.equal(options.useCache, false);
      assert.equal(options.requireReadableStream, true);
      runtime._modelSource = 'network';
      runtime._modelCacheState = 'disabled';
      runtime._modelCacheName = 'other';
      return new Response(new Uint8Array(16), { headers: { 'content-length': '16' } });
    };
    const info = await bridge.loadDraftModel('https://example.invalid/models/draft.gguf', { useCache: false });
    assert.deepEqual(info, { architecture: 'dflash', strategy: 'draft-dspark' });
    assert.deepEqual(loads, ['/draft/draft.gguf']);
    assert.deepEqual(opened, ['/draft/draft.gguf']);
    assert.deepEqual(
      [runtime._modelSource, runtime._modelCacheState, runtime._modelCacheName],
      ['cache', 'hit', 'target-cache'],
    );
    assert.equal(runtime._draftModel.url, 'https://example.invalid/models/draft.gguf');
    assert.deepEqual(bridge._loadedDraftModel, { url: 'https://example.invalid/models/draft.gguf', options: { useCache: false } });
  }],

  ['the browser smoke covers every strategy with pinned files and fails a mismatch', () => {
    const covered = new Set(Object.values(smoke.GROUPS).flatMap((group) => group.runs.flatMap((run) => run.strategies)));
    assert.deepEqual([...covered].sort(), [...SPECULATIVE_DECODING_STRATEGIES].sort());
    for (const group of Object.values(smoke.GROUPS)) {
      for (const key of smoke.groupFiles(group)) {
        assert.ok(key === 'smoke_model' || /^[0-9a-f]{64}$/.test(smoke.FILES[key]?.sha256), `${key} is not pinned`);
      }
    }
    const entry = {
      group: 'tiny',
      mode: 'wasm32 direct',
      failures: [],
      coreVariant: 'wasm32',
      execution: 'main-thread',
    };
    assert.deepEqual(smoke.validatePayload({ ok: true, modeResults: [entry] }, ['tiny'], ['wasm32'], ['direct']), []);
    assert.deepEqual(
      smoke.validatePayload(
        { ok: true, modeResults: [{ ...entry, failures: ['ngram-mod: output differs from the baseline'] }] },
        ['tiny'],
        ['wasm32'],
        ['direct'],
      ),
      ['tiny wasm32 direct: ngram-mod: output differs from the baseline'],
    );
    assert.match(
      smoke.validatePayload({ ok: true, modeResults: [] }, ['tiny'], ['wasm32'], ['direct'])[0],
      /mode results missing/,
    );
    const html = smoke.renderHarness([{ name: 'tiny', ...smoke.GROUPS.tiny }], 256, ['wasm32'], ['direct']);
    assert.match(html, /speculativeDecoding\[strategy\] === true/);
    assert.equal(smoke.parseArgs([]).group, 'tiny');
  }],

  ['the declared strategies, capability flags and core strategy names agree', () => {
    const union = dts.match(/export type SpeculativeDecodingStrategy =([^;]+);/)[1];
    assert.deepEqual([...union.matchAll(/'([a-z0-9-]+)'/g)].map(([, name]) => name), SPECULATIVE_DECODING_STRATEGIES);
    const body = functionBody('llamadart_webgpu_completion_capabilities_json()');
    assert.deepEqual([...body.matchAll(/\{"([a-z0-9-]+)",/g)].map(([, name]) => name), SPECULATIVE_DECODING_STRATEGIES);
    const upstream = readFileSync(
      path.join(rootDir, 'src/core/exports_speculative.inc'),
      'utf8',
    );
    assert.match(upstream, /common_speculative_type_from_name\(name\)/, 'the core maps names through llama.cpp');
  }],

  ['only the grammar, LoRA and speculative wrappers catch exceptions and every new export is linked', () => {
    const allowed = cmake.match(/-sEXCEPTION_CATCHING_ALLOWED=\[([^\]]*)\]/)[1];
    const names = allowed.split(',').map((name) => name.trim().replace(/^'|'$/g, ''));
    assert.deepEqual(names, [
      'llamadart_webgpu_grammar_sampler_init',
      'llamadart_webgpu_lora_adapter_init',
      'llamadart_webgpu_draft_model_load_file',
      'llamadart_webgpu_draft_context_init',
      'llamadart_webgpu_speculative_init',
    ]);
    for (const name of names.slice(2)) {
      assert.match(wrappers, new RegExp(`extern "C" __attribute__\\(\\(noinline\\)\\)[^\\n]*\\n${name}\\(`), `${name} must stay noinline`);
    }
    const exported = cmake.match(/-sEXPORTED_FUNCTIONS=\[([^\]]*)\]/)[1];
    const speculativeExports = readFileSync(path.join(rootDir, 'src/core/exports_speculative.inc'), 'utf8');
    for (const [, name] of speculativeExports.matchAll(/EMSCRIPTEN_KEEPALIVE [^(]*?\b(llamadart_webgpu_\w+)\(/g)) {
      assert.ok(exported.includes(`'_${name}'`), `${name} is not in EXPORTED_FUNCTIONS`);
    }
  }],

  ['the core validates a speculative request before it changes generation state', () => {
    const body = functionBody('int32_t begin_generation_impl(');
    const consume = body.indexOf('const speculative_request speculative = std::move(g_next_speculative);');
    const activeCheck = body.indexOf('if (g_generation_active) {');
    const validate = body.indexOf('!validate_speculative_request(');
    const firstSideEffect = body.indexOf('end_generation_state();');
    assert.ok(consume >= 0 && consume < activeCheck, 'a rejected begin still consumes the request');
    assert.ok(validate > 0 && validate < firstSideEffect);
    const draftLoad = functionBody('EMSCRIPTEN_KEEPALIVE int32_t llamadart_webgpu_draft_model_load(');
    assert.ok(
      draftLoad.indexOf('if (g_draft_model_architecture == "dflash") {')
        < draftLoad.indexOf('common_speculative_types_from_gguf(path)'),
      'types_from_gguf asserts on a non-DFlash GGUF without a block count',
    );
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
