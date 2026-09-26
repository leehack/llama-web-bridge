// Contract tests for the browser smoke helpers (scripts/browser_smoke_support.mjs)
// and the checks the Node smokes run on their result, including the in-page
// checks of the state and multimodal harnesses, run against fake bridges.
// Expected values were produced by the Python smokes and Python's json/repr,
// which the Node smokes replace byte for byte. No browser is started here.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  ISOLATION_HEADERS,
  UsageError,
  cachedModelName,
  contentType,
  downloadToCache,
  env,
  expandHome,
  parseSmokeArgs,
  pyFloat,
  pyInt,
  pyIter,
  pyJson,
  pyJsonLoads,
  pyLen,
  pyLessEqual,
  pyRepr,
  pyStrip,
  pyTruthy,
  redactLocation,
  resolvePath,
  resolvePinnedFile,
  resolvePinnedModel,
  serveIsolated,
  stageFile,
  translatePath,
  validateHash,
  webGpuLaunchArgs,
} from '../../scripts/browser_smoke_support.mjs';
import * as grammarSmoke from '../../scripts/grammar_browser_smoke.mjs';
import * as multimodalSmoke from '../../scripts/multimodal_browser_smoke.mjs';
import * as nextTokenSmoke from '../../scripts/next_token_scores_browser_smoke.mjs';
import * as stateSmoke from '../../scripts/state_persistence_browser_smoke.mjs';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const supportUrl = pathToFileURL(path.join(rootDir, 'scripts/browser_smoke_support.mjs')).href;
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'browser-smoke-support-test-'));
process.on('exit', () => fs.rmSync(tmp, { recursive: true, force: true }));

// pyJson writes what json.dumps writes for the value Playwright's Python client
// returns: undefined is None, NaN and the infinities are literals, -0 is -0.0,
// integral numbers are ints, floats use repr, and strings are ASCII-escaped.
{
  const value = {
    b: 1,
    a: [1, 2.5, -0, NaN, Infinity, -Infinity, undefined, true, false, 'é\u2028\n"\\\x7f😀\t\x01',
      0.00001, 0.0001, 1.5e16, 1e21, 123456789.125, 0.1, 1e-7, 5e-324, 1.7976931348623157e308, 1e16, -3],
    '': {},
    z: [],
    '\ue000': 1,
    '😀': 2,
    nested: { y: [[]], x: { k: null } },
  };
  assert.equal(pyJson(value), "{\"b\": 1, \"a\": [1, 2.5, -0.0, NaN, Infinity, -Infinity, null, true, false, \"\\u00e9\\u2028\\n\\\"\\\\\\u007f\\ud83d\\ude00\\t\\u0001\", 1e-05, 0.0001, 15000000000000000, 1e+21, 123456789.125, 0.1, 1e-07, 5e-324, 1.7976931348623157e+308, 10000000000000000, -3], \"\": {}, \"z\": [], \"\\ue000\": 1, \"\\ud83d\\ude00\": 2, \"nested\": {\"y\": [[]], \"x\": {\"k\": null}}}");
  assert.equal(pyJson(value, { indent: 2, sortKeys: true }), "{\n  \"\": {},\n  \"a\": [\n    1,\n    2.5,\n    -0.0,\n    NaN,\n    Infinity,\n    -Infinity,\n    null,\n    true,\n    false,\n    \"\\u00e9\\u2028\\n\\\"\\\\\\u007f\\ud83d\\ude00\\t\\u0001\",\n    1e-05,\n    0.0001,\n    15000000000000000,\n    1e+21,\n    123456789.125,\n    0.1,\n    1e-07,\n    5e-324,\n    1.7976931348623157e+308,\n    10000000000000000,\n    -3\n  ],\n  \"b\": 1,\n  \"nested\": {\n    \"x\": {\n      \"k\": null\n    },\n    \"y\": [\n      []\n    ]\n  },\n  \"z\": [],\n  \"\\ue000\": 1,\n  \"\\ud83d\\ude00\": 2\n}");
  assert.equal(pyJson({ missing: undefined }, { indent: 2, sortKeys: true }), '{\n  "missing": null\n}');
  assert.throws(() => pyJson(new Date(0)), /Object of type datetime is not JSON serializable/);
}

// pyJsonLoads keeps what json.loads keeps: float literals stay floats, big ints
// stay exact, -0 is the int 0 and a duplicate key keeps its last value, so
// pyJson writes the text json.dumps(json.loads(text)) writes.
{
  const text = '{"n": [1.0, -0, -0.0, 1e2, 1E-7, 12345678901234567890, 9007199254740993, 2.5e16, 0.1, 100, -3], '
    + '"b": {"z": 1e400, "y": 1}, "a": 1, "a": "dup"}';
  assert.equal(pyJson(pyJsonLoads(text)), '{"n": [1.0, 0, -0.0, 100.0, 1e-07, 12345678901234567890, 9007199254740993, '
    + '2.5e+16, 0.1, 100, -3], "b": {"z": Infinity, "y": 1}, "a": "dup"}');
  assert.equal(pyRepr(pyJsonLoads('[2.0, 12345678901234567890]')), '[2.0, 12345678901234567890]');
  assert.throws(() => pyJsonLoads('[NaN]'));
  // float(), str.strip(), bool(), len(), iter() and `<=` as the smokes use them.
  assert.deepEqual(['1', ' 2_5 ', '.5', '-1e-3', '+inf', 'NaN'].map(pyFloat).map(String), ['1', '25', '0.5', '-0.001', 'Infinity', 'NaN']);
  assert.throws(() => pyFloat('1__0'), { message: "could not convert string to float: '1__0'" });
  assert.equal(pyStrip(' 　x\x1c\x85 '), 'x');
  assert.equal(pyStrip('﻿z'), '﻿z');
  assert.deepEqual([null, 0, '', [], {}, false, 0n].map(pyTruthy), [false, false, false, false, false, false, false]);
  assert.deepEqual([1, 'a', [0], { a: 0 }, true].map(pyTruthy), [true, true, true, true, true]);
  assert.deepEqual([[1, 2], { a: 1 }, 'hé😀'].map(pyLen), [2, 1, 3]);
  assert.throws(() => pyLen(null), { message: "object of type 'NoneType' has no len()" });
  assert.deepEqual(pyIter({ a: 1, b: 2 }), ['a', 'b']);
  assert.throws(() => pyIter(3.5), { message: "'float' object is not iterable" });
  assert.equal(pyLessEqual(true, 1), true);
  assert.equal(pyLessEqual(NaN, 1), false);
  assert.throws(() => pyLessEqual(null, 0.25), { message: "'<=' not supported between instances of 'NoneType' and 'float'" });
}

// The text-to-speech and decision smokes launch Chromium with WebGPU enabled.
assert.deepEqual(webGpuLaunchArgs('darwin'), [
  '--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-webgpu', '--use-angle=metal', '--enable-features=SharedArrayBuffer',
]);
assert.deepEqual(webGpuLaunchArgs('linux'), [
  '--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-webgpu', '--disable-vulkan-surface',
  '--enable-features=SharedArrayBuffer,Vulkan',
]);

// repr() for failure messages.
{
  const cases = [
    ['abc', "'abc'"],
    ["it's", '"it\'s"'],
    ['say "hi"', '\'say "hi"\''],
    ['both \' and "', "'both \\' and \"'"],
    ['tab\there\n', "'tab\\there\\n'"],
    ['\x00\x7f\x80\xa0\u00e9\u2028\u{1F600}\u{E0001}', "'\\x00\\x7f\\x80\\xa0é\\u2028😀\\U000e0001'"],
    ['back\\slash', "'back\\\\slash'"],
    [[null, true, 1, 2.5, -0, NaN, Infinity, 'x', { k: [1] }], "[None, True, 1, 2.5, -0.0, nan, inf, 'x', {'k': [1]}]"],
    [undefined, 'None'],
  ];
  for (const [value, expected] of cases) assert.equal(pyRepr(value), expected, expected);
}

// Redaction keeps scheme, lowercased host, an explicit non-zero port and the
// path; cache names are sha256(url)[:12] plus the URL's file name.
{
  const cases = [
    ['https://huggingface.co/aladar/llama-2-tiny-random-GGUF/resolve/main/llama-2-tiny-random.gguf',
      'https://huggingface.co/aladar/llama-2-tiny-random-GGUF/resolve/main/llama-2-tiny-random.gguf',
      '20cb00cf64ff-llama-2-tiny-random.gguf'],
    ['https://user:secret@Example.COM:443/a/b.gguf;p=1?token=abc#frag', 'https://example.com:443/a/b.gguf', 'b3516c679caa-b.gguf'],
    ['http://user@host:8080/x?y', 'http://host:8080/x', '53146f57d74c-x'],
    ['HTTPS://[::1]:9000/m.gguf', 'https://::1:9000/m.gguf', 'dcde5233ff4b-m.gguf'],
    ['http://[fe80::1%25EN0]/m', 'http://fe80::1%25EN0/m', 'fc1a09a35071-m'],
    ['ftp://host/dir/;type=a', 'ftp://host/dir/', '8e5601e43dfa-dir'],
    ['s3://bucket/key;v=1', 's3://bucket/key;v=1', 'cf61e2f4e059-key;v=1'],
    ['not a url', '[invalid-url]', 'd8b5bf9b9fd4-not a url'],
    ['/local/path.gguf', '[invalid-url]', 'c269f8de1941-path.gguf'],
    ['file:///tmp/x.gguf', '[invalid-url]', '97b9d200a418-x.gguf'],
    ['http://host:0/x', 'http://host/x', '8696bbc00ad5-x'],
    ['http://host/', 'http://host/', 'e0a05c858d79-state-smoke-model.gguf'],
    ['http://host', 'http://host', 'a7ea5c455a69-state-smoke-model.gguf'],
    ['  http://host/sp', 'http://host/sp', 'd29e674a0ee8-sp'],
    ['http://ho\tst/a\nb', 'http://host/ab', 'be5455785872-ab'],
  ];
  for (const [url, redacted, cacheName] of cases) {
    assert.equal(redactLocation(url), redacted, url);
    assert.equal(cachedModelName(url), cacheName, url);
  }
  assert.throws(() => redactLocation('http://host:99999/x'), /Port out of range/);
}

// Paths: `~` expands only where the Python smokes called expanduser, and
// resolution follows symlinks for the part that exists.
{
  const home = process.env.HOME;
  assert.equal(expandHome('~'), home);
  assert.equal(expandHome('~/.cache/x'), path.join(home, '.cache/x'));
  assert.equal(expandHome('a/~/b'), 'a/~/b');
  assert.equal(expandHome('~other/x'), '~other/x');
  const real = fs.realpathSync(tmp);
  fs.symlinkSync(real, path.join(tmp, 'link'));
  assert.equal(resolvePath(path.join(tmp, 'link', 'missing', 'file')), path.join(real, 'missing', 'file'));
  // Python resolves `link/..` after following the link, not lexically.
  assert.equal(resolvePath(`${tmp}/link/../x`), path.join(path.dirname(real), 'x'));
  assert.equal(resolvePath(`${tmp}/nope/../x`), path.join(real, 'x'));
}

// Argument parsing: argparse's `--flag=value`, unique prefixes, negative
// numbers, int() literals, choices, and exit status 2 (UsageError) on misuse;
// env defaults are read at parse time and a bad int env value is a plain error.
{
  const spec = {
    prog: 'example.mjs',
    description: 'd',
    options: [
      { flag: '--dist-dir', type: 'path', default: () => env.path('TEST_SMOKE_DIST', 'dist'), help: 'h' },
      { flag: '--timeout-ms', type: 'int', default: () => env.int('TEST_SMOKE_TIMEOUT', '100'), help: 'h' },
      { flag: '--model-url', type: 'string', default: () => env.string('TEST_SMOKE_URL'), help: 'h' },
      { flag: '--model-path', type: 'path', default: () => env.optionalPath('TEST_SMOKE_PATH'), help: 'h' },
      { flag: '--memory-mode', type: 'string', choices: ['all', 'wasm32'], default: () => 'all', help: 'h' },
    ],
  };
  delete process.env.TEST_SMOKE_TIMEOUT;
  process.env.TEST_SMOKE_DIST = '~/dist';
  process.env.TEST_SMOKE_PATH = '';
  assert.deepEqual(parseSmokeArgs([], spec), {
    distDir: '~/dist', timeoutMs: 100, modelUrl: '', modelPath: null, memoryMode: 'all',
  });
  assert.deepEqual(
    parseSmokeArgs(['--timeout-ms=-5', '--model-u', '-', '--model-path', '', '--mem', 'wasm32', '--dist-dir', 'x'], spec),
    { distDir: 'x', timeoutMs: -5, modelUrl: '-', modelPath: '.', memoryMode: 'wasm32' },
  );
  assert.equal(parseSmokeArgs(['--timeout-ms', ' 1_000 '], spec).timeoutMs, 1000);
  for (const [argv, message] of [
    [['--timeout-ms', 'abc'], "argument --timeout-ms: invalid int value: 'abc'"],
    [['--timeout-ms'], 'argument --timeout-ms: expected one argument'],
    [['--model-url', '--dist-dir', 'x'], 'argument --model-url: expected one argument'],
    [['--model'], 'ambiguous option: --model could match --model-url, --model-path'],
    [['--memory-mode', 'wasm64'], "argument --memory-mode: invalid choice: 'wasm64' (choose from 'all', 'wasm32')"],
    [['--nope', 'x'], 'unrecognized arguments: --nope x'],
  ]) {
    assert.throws(() => parseSmokeArgs(argv, spec), (error) => error instanceof UsageError && error.message === message, message);
  }
  process.env.TEST_SMOKE_TIMEOUT = '12x';
  assert.throws(() => parseSmokeArgs(['--timeout-ms', '5'], spec), (error) => !(error instanceof UsageError)
    && error.message === "invalid literal for int() with base 10: '12x'");
  delete process.env.TEST_SMOKE_TIMEOUT;
  assert.equal(pyInt('+7'), 7);
}

// The smokes read the same flags and environment variables as the Python ones.
{
  const saved = { ...process.env };
  try {
    process.env.BRIDGE_DIST_DIR = '/d';
    process.env.LLAMA_WEBGPU_SMOKE_MODEL_URL = 'https://h/m.gguf';
    process.env.LLAMA_WEBGPU_SMOKE_MODEL_SHA256 = 'ab';
    process.env.LLAMA_WEBGPU_SMOKE_MODEL_CACHE = '~/.cache/llama-web-bridge/state-smoke-models';
    process.env.LLAMA_WEBGPU_SMOKE_ARTIFACTS_DIR = '/a/state';
    process.env.LLAMA_WEBGPU_GRAMMAR_ARTIFACTS_DIR = '/a/grammar';
    process.env.LLAMA_WEBGPU_NEXT_TOKEN_SCORES_ARTIFACTS_DIR = '/a/nts';
    process.env.LLAMA_WEBGPU_GRAMMAR_TIMEOUT_MS = '7';
    delete process.env.LLAMA_WEBGPU_SMOKE_MODEL_PATH;
    delete process.env.LLAMA_WEBGPU_SMOKE_TIMEOUT_MS;
    delete process.env.LLAMA_WEBGPU_NEXT_TOKEN_SCORES_TIMEOUT_MS;
    const shared = {
      distDir: '/d', modelUrl: 'https://h/m.gguf', modelPath: null, modelSha256: 'ab',
      modelCacheDir: '~/.cache/llama-web-bridge/state-smoke-models',
    };
    assert.deepEqual(stateSmoke.parseArgs([]), { ...shared, timeoutMs: 120000, artifactsDir: '/a/state' });
    assert.deepEqual(grammarSmoke.parseArgs([]), {
      ...shared, timeoutMs: 7, memoryMode: 'all', nCtx: 1024, artifactsDir: '/a/grammar',
    });
    assert.deepEqual(nextTokenSmoke.parseArgs(['--memory-mode', 'wasm64', '--n-ctx', '64']), {
      ...shared, timeoutMs: 300000, memoryMode: 'wasm64', nCtx: 64, artifactsDir: '/a/nts',
    });

    // The multimodal smoke reads its own LLAMA_WEBGPU_MULTIMODAL_* variables.
    for (const key of Object.keys(process.env)) if (key.startsWith('LLAMA_WEBGPU_MULTIMODAL_')) delete process.env[key];
    assert.deepEqual(multimodalSmoke.parseArgs([]), {
      distDir: '/d', timeoutMs: 420000, modelUrl: '', modelPath: null, modelSha256: '', mmprojUrl: '', mmprojPath: null,
      mmprojSha256: '', modelCacheDir: '~/.cache/llama-web-bridge/multimodal-smoke-models', artifactsDir: null,
    });
    Object.assign(process.env, {
      LLAMA_WEBGPU_MULTIMODAL_TIMEOUT_MS: '9',
      LLAMA_WEBGPU_MULTIMODAL_MODEL_URL: 'https://h/q.gguf',
      LLAMA_WEBGPU_MULTIMODAL_MODEL_PATH: '/m/q.gguf',
      LLAMA_WEBGPU_MULTIMODAL_MODEL_SHA256: 'cd',
      LLAMA_WEBGPU_MULTIMODAL_MMPROJ_URL: 'https://h/p.gguf',
      LLAMA_WEBGPU_MULTIMODAL_MMPROJ_PATH: '/m/p.gguf',
      LLAMA_WEBGPU_MULTIMODAL_MMPROJ_SHA256: 'ef',
      LLAMA_WEBGPU_MULTIMODAL_MODEL_CACHE: '~/mm',
      LLAMA_WEBGPU_MULTIMODAL_ARTIFACTS_DIR: '/a/mm',
    });
    assert.deepEqual(multimodalSmoke.parseArgs(['--mmproj-path', '/x/p.gguf', '--timeout-ms=3']), {
      distDir: '/d', timeoutMs: 3, modelUrl: 'https://h/q.gguf', modelPath: '/m/q.gguf', modelSha256: 'cd',
      mmprojUrl: 'https://h/p.gguf', mmprojPath: '/x/p.gguf', mmprojSha256: 'ef', modelCacheDir: '~/mm',
      artifactsDir: '/a/mm',
    });
  } finally {
    for (const key of Object.keys(process.env)) if (!(key in saved)) delete process.env[key];
    Object.assign(process.env, saved);
  }
}

// The checks the smokes run on the printed result give the Python messages.
{
  const MODES = ['wasm32 direct', 'wasm32 worker', 'wasm64 direct', 'wasm64 worker'];
  const GRAMMAR_CASES = grammarSmoke.CASES.map((testCase) => testCase.name);
  const modeEntry = (mode, extra) => ({
    mode,
    coreVariant: mode.split(' ')[0],
    execution: mode.endsWith(' worker') ? 'worker' : 'main-thread',
    workerFallbackReason: null,
    ...extra,
  });
  const grammarPayload = (modes = MODES) => ({
    ok: true,
    modeResults: modes.map((mode) => modeEntry(mode, {
      cases: GRAMMAR_CASES.map((name) => ({ name, valid: true, text: name.startsWith('invalid-') ? null : 'yes', error: null })),
      plainCompletionError: null,
    })),
    globalWorkerFallbackReason: null,
  });
  const ntsPayload = () => ({
    ok: true,
    modeResults: MODES.map((mode) => modeEntry(mode, { failures: [], top: [[1, -0.5]] })),
    globalWorkerFallbackReason: null,
  });
  const mutate = (payload, fn) => {
    fn(payload);
    return payload;
  };
  const both = ['wasm32', 'wasm64'];
  const grammarCases = [
    [grammarPayload(), both, []],
    [grammarPayload(MODES.slice(0, 2)), ['wasm32'], []],
    [{ ok: false, error: 'boom' }, both, ['harness failed: boom']],
    [{ error: null }, both, ['harness failed: None']],
    [{ ok: true }, both, ['mode results missing']],
    [mutate(grammarPayload(), (p) => p.modeResults.reverse()), both, ['mode results missing']],
    [grammarPayload(), ['wasm32'], ['mode results missing']],
    [mutate(grammarPayload(), (p) => Object.assign(p.modeResults[1].cases[0], { valid: false, text: null, error: "it's \"bad\"" })), both,
      ["wasm32 worker invalid-unterminated-string: expected an error containing '(invalid grammar)' and the parser's reason (or the non-throwing rejection for left recursion), got text=None error='it\\'s \"bad\"'"]],
    [mutate(grammarPayload(), (p) => Object.assign(p.modeResults[2].cases[4], { valid: false, text: 'maybe\n' })), both,
      ["wasm64 direct yes-no-sampled: expected grammar-valid text, got text='maybe\\n' error=None"]],
    [mutate(grammarPayload(), (p) => { p.modeResults[0].cases[3].valid = 1; }), both,
      ["wasm32 direct yes-no-greedy: expected grammar-valid text, got text='yes' error=None"]],
    [mutate(grammarPayload(), (p) => p.modeResults[3].cases.pop()), both, ['wasm64 worker: case results missing']],
    [mutate(grammarPayload(), (p) => { p.modeResults[0].plainCompletionError = 'bad'; }), both,
      ["wasm32 direct: completion after the grammar runs failed: 'bad'"]],
    [mutate(grammarPayload(), (p) => { delete p.modeResults[0].plainCompletionError; }), both, []],
    [mutate(grammarPayload(), (p) => { p.modeResults[3].coreVariant = 'wasm32'; }), both,
      ["wasm64 worker: expected the wasm64 core, got 'wasm32'"]],
    [mutate(grammarPayload(), (p) => Object.assign(p.modeResults[1], { execution: 'main-thread', workerFallbackReason: 'why' })), both,
      ["wasm32 worker: expected worker execution, got 'main-thread' (worker fallback reason: 'why')"]],
    [mutate(grammarPayload(), (p) => { p.globalWorkerFallbackReason = 'gone'; }), both, ["worker fell back to the main thread: 'gone'"]],
  ];
  for (const [payload, memoryModes, expected] of grammarCases) {
    assert.deepEqual(grammarSmoke.validatePayload(payload, memoryModes), expected, expected.join('\n'));
  }
  const ntsCases = [
    [ntsPayload(), []],
    [{ ok: false, error: 'Error: x\n    at y' }, ['harness failed: Error: x\n    at y']],
    [mutate(ntsPayload(), (p) => Object.assign(p.modeResults[2], { failures: ['first: bad', 'threw: x'], coreVariant: null })),
      ['wasm64 direct: first: bad', 'wasm64 direct: threw: x']],
    [mutate(ntsPayload(), (p) => { delete p.modeResults[0].failures; }), ['wasm32 direct: failures missing']],
    [mutate(ntsPayload(), (p) => { p.modeResults[2].coreVariant = 'wasm32'; }), ["wasm64 direct: expected the wasm64 core, got 'wasm32'"]],
    [mutate(ntsPayload(), (p) => { p.modeResults[3].execution = null; }),
      ['wasm64 worker: expected worker execution, got None (worker fallback reason: None)']],
    [mutate(ntsPayload(), (p) => { p.globalWorkerFallbackReason = 'gone'; }), ["worker fell back to the main thread: 'gone'"]],
  ];
  for (const [payload, expected] of ntsCases) {
    assert.deepEqual(nextTokenSmoke.validatePayload(payload, both), expected, expected.join('\n'));
  }
  // A malformed mode entry fails instead of passing, as Python raised.
  assert.throws(() => grammarSmoke.validatePayload(mutate(grammarPayload(), (p) => p.modeResults.push(5)), both));

  const stateEntry = (mode) => ({
    mode, preload: true, progressEvents: 3, embeddingComponents: 32, embeddingBatchSize: 2, savedBytes: 1000, restoredTokens: 2,
  });
  const statePayload = () => ({
    ok: true,
    methods: ['stateSaveFile', 'stateLoadFile', 'stateSaveBytes', 'stateLoadBytes'],
    modes: ['direct runtime', 'worker runtime'],
    modelBacked: true,
    modeResults: [stateEntry('direct runtime'), stateEntry('worker runtime')],
  });
  const stateCases = [
    [statePayload(), true, null],
    [{ ...statePayload(), modelBacked: false, modeResults: [] }, false, null],
    [{ ...statePayload(), methods: ['stateSaveFile'] }, true, 'smoke methods payload mismatch'],
    [{ ...statePayload(), modes: ['worker runtime', 'direct runtime'] }, true, 'smoke modes payload mismatch'],
    [{ ...statePayload(), modelBacked: false }, true, 'model-backed smoke did not run'],
    [{ ...statePayload(), modeResults: [stateEntry('direct runtime')] }, true, 'model-backed mode results missing'],
    [mutate(statePayload(), (p) => { p.modeResults[1].embeddingComponents = 0; }), true, 'embedding vector was empty'],
    // isinstance(True, int) held in Python.
    [mutate(statePayload(), (p) => { p.modeResults[1].embeddingComponents = true; }), true, null],
    [mutate(statePayload(), (p) => { p.modeResults[0].savedBytes = 1.5; }), true, 'state snapshot was empty'],
    [mutate(statePayload(), (p) => { p.modeResults[0].embeddingBatchSize = 1; }), true, 'embedding batch result was incomplete'],
    [mutate(statePayload(), (p) => { delete p.modeResults[1].restoredTokens; }), true, 'restored token metadata missing'],
  ];
  for (const [payload, modelBacked, message] of stateCases) {
    if (message === null) stateSmoke.checkPayload(payload, modelBacked);
    else assert.throws(() => stateSmoke.checkPayload(payload, modelBacked), { message });
  }

  const multimodalPayload = () => ({
    ok: true,
    modes: ['direct runtime', 'worker runtime'],
    modeResults: [
      { mode: 'direct runtime', elapsedMs: 1, output: 'HELLO', imageResizeDiagnostic: 'media_image_resized:320x180->298x168' },
      { mode: 'worker runtime', elapsedMs: 1, output: 'HELLO', imageResizeDiagnostic: undefined },
    ],
  });
  const multimodalCases = [
    [multimodalPayload(), null],
    [{ ...multimodalPayload(), modes: ['direct runtime'] }, 'multimodal smoke modes payload mismatch'],
    [{ ...multimodalPayload(), modes: undefined }, 'multimodal smoke modes payload mismatch'],
    [mutate(multimodalPayload(), (p) => p.modeResults.pop()), 'multimodal mode results missing'],
    [{ ...multimodalPayload(), modeResults: { length: 2 } }, 'multimodal mode results missing'],
  ];
  for (const [payload, message] of multimodalCases) {
    if (message === null) multimodalSmoke.checkPayload(payload);
    else assert.throws(() => multimodalSmoke.checkPayload(payload), { message });
  }
}

// The harness pages embed their configuration as Python's json.dumps did.
{
  const state = stateSmoke.renderHarness('state-smoke-model.gguf');
  assert.ok(state.includes('    const methods = ["stateSaveFile", "stateLoadFile", "stateSaveBytes", "stateLoadBytes"];\n'));
  assert.ok(state.includes('    const modelUrl = "/state-smoke-model.gguf";\n'));
  assert.ok(stateSmoke.renderHarness(null).includes('    const modelUrl = null;\n'));
  assert.ok(state.includes('assert(missing.length === 0, `${mode} missing public state methods: ${missing.join(\', \')}`);'));
  const grammar = grammarSmoke.renderHarness(1024, ['wasm64']);
  assert.ok(grammar.includes('"prompt": "<|im_start|>user\\nIs the sky blue? Answer yes or no.<|im_end|>\\n'));
  assert.ok(grammar.includes('"temp": 0.8, "topK": 40'));
  assert.ok(grammar.includes('"nCtx": 1024, "memoryModes": ["wasm64"], "runtimeModes": ["direct", "worker"]}'));
  const scores = nextTokenSmoke.renderHarness(64, ['wasm32', 'wasm64']);
  assert.ok(scores.includes('"topK": 8, "tolerance": 0.001, '));
  const multimodal = multimodalSmoke.renderHarness();
  assert.ok(multimodal.includes("await bridge.loadModelFromUrl('/multimodal-model.gguf', {"));
  assert.ok(multimodal.includes("await bridge.loadMultimodalProjector('/multimodal-mmproj.gguf');"));
  assert.ok(multimodal.includes('assert(bridge.supportsVision(), `${mode} did not report vision support`);'));
  for (const page of [state, grammar, scores, multimodal]) {
    assert.ok(page.startsWith('\n<!doctype html>\n<meta charset="utf-8">\n'));
    assert.ok(page.endsWith('})();\n</script>\n'));
  }
}

// --- Harness pages against a fake bridge --------------------------------------

// Runs a harness page's module script in Node against a fake bridge class, so the
// in-page usage assertions are exercised without a browser. The page imports the
// bridge from '/llama_webgpu_bridge.js'; that one call is redirected to the fake.
async function runHarnessPage(page, Bridge, extraDocument = {}) {
  const body = page.slice(page.indexOf('<script type="module">\n') + '<script type="module">\n'.length, page.lastIndexOf('</script>'));
  const importCall = "await import('/llama_webgpu_bridge.js')";
  assert.equal(body.split(importCall).length, 2, 'the harness must import the bridge exactly once');
  const script = body.replace(importCall, '({ LlamaWebGpuBridge: window.__FakeBridge })').trim();
  const result = { textContent: 'pending' };
  const window = { crossOriginIsolated: true, __FakeBridge: Bridge };
  const document = { getElementById: (id) => (id === 'result' ? result : null), ...extraDocument };
  await new Function('window', 'document', `'use strict'; return ${script}`)(window, document);
  assert.equal(result.textContent, JSON.stringify(window.__smokeResult));
  return window.__smokeResult;
}

const firstErrorLine = (payload) => {
  assert.equal(payload.ok, false, JSON.stringify(payload));
  return payload.error.split('\n')[0];
};

// Words as tokens after a BOS token, like a tokenizer that adds special tokens.
const fakeTokenize = (text) => [1, ...text.split(/\s+/).filter(Boolean).map((word) => 100 + word.length)];

// A fake bridge for the state harness. The KV cache keeps the prompt and the
// generated token, a repeated prompt keeps all but its last token, and an abort
// during generation resolves in the direct runtime and rejects in the worker.
// `tweak.usage(context)` returns the usages to report for one completion.
const stateBridge = (tweak = {}) => class FakeStateBridge {
  constructor({ disableWorker }) {
    this.mode = disableWorker ? 'direct runtime' : 'worker runtime';
    this.loaded = false;
    this.cache = [];
    this.calls = 0;
    if (!disableWorker) this._workerProxy = {};
  }

  async loadModelFromUrl(url, options) {
    assert.equal(url, '/state-smoke-model.gguf');
    for (let loaded = 1; loaded <= 3; loaded += 1) options.progressCallback?.({ loaded, total: 3 });
    this.loaded = true;
    this.cache = [];
  }

  requireModel() {
    if (!this.loaded) throw new Error('No model loaded');
  }

  async tokenize(text) { this.requireModel(); return fakeTokenize(text); }

  async embed(text) { this.requireModel(); return [text.length, 0.5, -0.25]; }

  async embedBatch(texts) { return Promise.all(texts.map((text) => this.embed(text))); }

  getContextSize() { return 64; }

  async stateSaveBytes(tokens) {
    this.requireModel();
    return new Uint8Array(tokens.flatMap((token) => [token & 0xff, token >> 8]));
  }

  async stateLoadBytes(bytes) {
    this.requireModel();
    const tokens = [];
    for (let index = 0; index < bytes.length; index += 2) tokens.push(bytes[index] | (bytes[index + 1] << 8));
    this.cache = tokens;
    return { tokens };
  }

  async stateSaveFile() { this.requireModel(); }

  async stateLoadFile() { this.requireModel(); }

  async _callWorker(name, args, _signal, transfer) {
    // Like postMessage: the worker gets copies and the transferred buffers detach.
    return this[name](...structuredClone(args, { transfer }));
  }

  async createCompletion(prompt, options) {
    this.requireModel();
    const tokens = fakeTokenize(prompt);
    let cached = 0;
    while (cached < tokens.length - 1 && this.cache[cached] === tokens[cached]) cached += 1;
    let completionTokens = 0;
    let finishReason = 'length';
    while (completionTokens < options.nPredict) {
      completionTokens += 1;
      options.onToken?.(new Uint8Array([120]), 'x'.repeat(completionTokens));
      if (options.signal?.aborted) {
        finishReason = 'cancelled';
        break;
      }
    }
    this.cache = [...tokens, 120];
    const usage = {
      promptTokens: tokens.length,
      cachedPromptTokens: cached,
      completionTokens,
      timeToFirstTokenMs: 2,
      durationMs: 5,
      finishReason,
    };
    this.calls += 1;
    const context = { mode: this.mode, call: this.calls, prompt, usage };
    for (const reported of tweak.usage ? tweak.usage(context) : [usage]) options.onUsage?.(reported);
    const outcome = tweak.abort?.(context)
      ?? (finishReason === 'cancelled' && this.mode === 'worker runtime' ? 'reject' : 'resolve');
    if (outcome === 'reject') throw new DOMException('The operation was aborted.', 'AbortError');
    if (outcome instanceof Error) throw outcome;
    return 'x'.repeat(completionTokens);
  }

  async dispose() {}
};

{
  const page = stateSmoke.renderHarness('state-smoke-model.gguf');
  const onCall = (mode, call, change) => ({
    usage: (context) => (context.mode === mode && context.call === call ? change(context.usage) : [context.usage]),
  });
  const passed = await runHarnessPage(page, stateBridge());
  assert.equal(passed.ok, true, passed.error);
  stateSmoke.checkPayload(passed, true);
  const [direct, worker] = passed.modeResults;
  assert.deepEqual(direct.initialUsage,
    { promptTokens: 2, cachedPromptTokens: 0, completionTokens: 1, timeToFirstTokenMs: 2, durationMs: 5, finishReason: 'length' });
  assert.equal(direct.restoredUsage.cachedPromptTokens, 1);
  assert.deepEqual([direct.repeatedUsage.promptTokens, direct.repeatedUsage.cachedPromptTokens], [10, 9]);
  assert.deepEqual([direct.abortOutcome, worker.abortOutcome], ['resolved', 'rejected']);
  assert.deepEqual([direct.abortedUsage.finishReason, direct.abortedUsage.completionTokens], ['cancelled', 1]);
  assert.equal(worker.detachedAfterLoadTransfer, true);
  // A completion that stops before streaming anything has no first-token time.
  assert.equal((await runHarnessPage(page, stateBridge(onCall('worker runtime', 1,
    (usage) => [{ ...usage, completionTokens: 0, finishReason: 'stop', timeToFirstTokenMs: null }])))).ok, true);
  // No model: the harness never completes, so usage is not checked.
  assert.equal((await runHarnessPage(stateSmoke.renderHarness(null), stateBridge({ usage: () => [] }))).ok, true);

  const edit = (fields) => (usage) => [{ ...usage, ...fields }];
  // Completions per mode: 1 initial, 2 mutation, 3 after restore, 4 and 5 repeated prompt, 6 aborted.
  const cases = [
    [onCall('direct runtime', 1, () => []), 'direct runtime initial completion reported usage 0 times'],
    [onCall('worker runtime', 1, (usage) => [usage, usage]), 'worker runtime initial completion reported usage 2 times'],
    [onCall('direct runtime', 1, edit({ promptTokens: 3 })), 'direct runtime initial completion reported 3 prompt tokens, expected 2'],
    [onCall('direct runtime', 1, edit({ cachedPromptTokens: 2 })), 'direct runtime initial completion reported 2 cached prompt tokens'],
    [onCall('direct runtime', 1, edit({ cachedPromptTokens: 0.5 })), 'direct runtime initial completion reported 0.5 cached prompt tokens'],
    [onCall('direct runtime', 1, edit({ cachedPromptTokens: -1 })), 'direct runtime initial completion reported -1 cached prompt tokens'],
    [onCall('direct runtime', 1, edit({ completionTokens: 2 })), 'direct runtime initial completion reported 2 completion tokens'],
    [onCall('direct runtime', 1, edit({ finishReason: 'stop' })), 'direct runtime initial completion finished with stop after 1 tokens'],
    [onCall('direct runtime', 1, edit({ completionTokens: 0 })), 'direct runtime initial completion finished with length after 0 tokens'],
    [onCall('direct runtime', 1, edit({ durationMs: undefined })), 'direct runtime initial completion reported duration undefined'],
    [onCall('direct runtime', 1, edit({ durationMs: -1 })), 'direct runtime initial completion reported duration -1'],
    [onCall('direct runtime', 1, edit({ timeToFirstTokenMs: 6 })), 'direct runtime initial completion reported time to first token 6'],
    [onCall('direct runtime', 1, edit({ timeToFirstTokenMs: -1 })), 'direct runtime initial completion reported time to first token -1'],
    [onCall('worker runtime', 3, () => []), 'worker runtime completion after state restore reported usage 0 times'],
    [onCall('worker runtime', 3, edit({ promptTokens: 1 })), 'worker runtime completion after state restore reported 1 prompt tokens, expected 2'],
    [onCall('direct runtime', 3, edit({ cachedPromptTokens: 0 })),
      'direct runtime completion after state restore reused 0 of 2 prompt tokens'],
    [onCall('direct runtime', 4, () => []), 'direct runtime repeated prompt reported usage 1 times'],
    [onCall('direct runtime', 5, edit({ promptTokens: 11 })), 'direct runtime repeated prompt reported 10,11 prompt tokens, expected 10'],
    [onCall('worker runtime', 5, edit({ cachedPromptTokens: 1 })), 'worker runtime repeated prompt reused 1 of 10 prompt tokens'],
    [{ abort: (context) => (context.mode === 'worker runtime' && context.call === 6 ? 'resolve' : undefined) },
      'worker runtime aborted completion resolved'],
    [{ abort: (context) => (context.mode === 'direct runtime' && context.call === 6 ? 'reject' : undefined) },
      'direct runtime aborted completion rejected'],
    [{ abort: (context) => (context.call === 6 ? new Error('boom') : undefined) }, 'direct runtime aborted completion failed: Error: boom'],
    [onCall('worker runtime', 6, () => []), 'worker runtime aborted completion reported usage 0 times'],
    [onCall('direct runtime', 6, (usage) => [usage, usage]), 'direct runtime aborted completion reported usage 2 times'],
    [onCall('direct runtime', 6, edit({ finishReason: 'length' })), 'direct runtime aborted completion finished with length'],
    [onCall('direct runtime', 6, edit({ completionTokens: 0 })), 'direct runtime aborted completion reported 0 completion tokens'],
    [onCall('worker runtime', 6, edit({ completionTokens: 8 })), 'worker runtime aborted completion reported 8 completion tokens'],
  ];
  for (const [tweak, message] of cases) {
    assert.equal(firstErrorLine(await runHarnessPage(page, stateBridge(tweak))), `Error: ${message}`, message);
  }
}

// A fake bridge for the multimodal harness: the image adds 64 prompt positions
// and the direct runtime exposes the native media helpers the page calls.
const multimodalBridge = (tweak = {}) => class FakeMultimodalBridge {
  constructor({ disableWorker }) {
    this.mode = disableWorker ? 'direct runtime' : 'worker runtime';
    const files = new Map();
    const helpers = {
      llamadart_webgpu_media_add_file: ([file]) => (files.has(file) ? 0 : -4),
      llamadart_webgpu_media_add_encoded: ([bytes, length]) => {
        if (length === 0) return -3;
        return length === bytes.length && length > 8 ? 0 : -4;
      },
      llamadart_webgpu_media_clear_pending: () => null,
    };
    if (disableWorker) {
      this._runtime = {
        _core: {
          FS: { writeFile: (file, bytes) => files.set(file, bytes), unlink: (file) => files.delete(file) },
          ccall: (name, _returnType, _argTypes, args) => helpers[name](args),
        },
      };
    }
  }

  async loadModelFromUrl(url) { assert.equal(url, '/multimodal-model.gguf'); }

  async loadMultimodalProjector(url) { assert.equal(url, '/multimodal-mmproj.gguf'); }

  supportsVision() { return true; }

  async tokenize(text) { return fakeTokenize(text); }

  async createCompletion(prompt, options) {
    assert.equal(options.parts[0].type, 'image');
    const usage = {
      promptTokens: fakeTokenize(prompt).length + 64,
      cachedPromptTokens: 0,
      completionTokens: 12,
      timeToFirstTokenMs: 40,
      durationMs: 90,
      finishReason: 'stop',
    };
    for (const reported of tweak.usage ? tweak.usage({ mode: this.mode, usage }) : [usage]) options.onUsage?.(reported);
    return 'I see a box with the word HELLO.';
  }

  getModelMetadata() { return { 'llamadart.webgpu.runtime_notes': 'x;media_image_resized:320x180->300x169' }; }

  async dispose() {}
};

{
  const canvas = {
    getContext: () => ({ fillRect() {}, fillText() {} }),
    toBlob: (callback) => callback({ arrayBuffer: async () => new Uint8Array(32).fill(7).buffer }),
  };
  const page = multimodalSmoke.renderHarness();
  const run = (tweak) => runHarnessPage(page, multimodalBridge(tweak), { createElement: () => canvas });
  const passed = await run();
  assert.equal(passed.ok, true, passed.error);
  multimodalSmoke.checkPayload(passed);
  assert.deepEqual(passed.modeResults.map((entry) => [entry.mode, entry.promptTextTokens, entry.usage.promptTokens]),
    [['direct runtime', 5, 69], ['worker runtime', 5, 69]]);
  assert.equal(passed.modeResults[0].imageResizeDiagnostic, 'media_image_resized:320x180->300x169');

  const onMode = (mode, change) => ({ usage: (context) => (context.mode === mode ? change(context.usage) : [context.usage]) });
  const edit = (fields) => (usage) => [{ ...usage, ...fields }];
  // nPredict (64) tokens end with 'length'.
  assert.equal((await run(onMode('direct runtime', edit({ completionTokens: 64, finishReason: 'length' })))).ok, true);
  const cases = [
    [onMode('direct runtime', () => []), 'direct runtime reported usage 0 times'],
    [onMode('worker runtime', (usage) => [usage, usage]), 'worker runtime reported usage 2 times'],
    [onMode('direct runtime', edit({ promptTokens: 5 })),
      'direct runtime reported 5 prompt positions for an image prompt of 5 text tokens'],
    [onMode('worker runtime', edit({ cachedPromptTokens: 4 })), 'worker runtime reused 4 tokens of a multimodal prompt'],
    [onMode('direct runtime', edit({ completionTokens: 0 })), 'direct runtime reported 0 completion tokens'],
    [onMode('direct runtime', edit({ completionTokens: 65 })), 'direct runtime reported 65 completion tokens'],
    [onMode('direct runtime', edit({ finishReason: 'length' })), 'direct runtime finished with length after 12 tokens'],
    [onMode('direct runtime', edit({ completionTokens: 64 })), 'direct runtime finished with stop after 64 tokens'],
    [onMode('worker runtime', edit({ timeToFirstTokenMs: null })), 'worker runtime reported time to first token null of 90 ms'],
    [onMode('direct runtime', edit({ timeToFirstTokenMs: 0 })), 'direct runtime reported time to first token 0 of 90 ms'],
    [onMode('direct runtime', edit({ timeToFirstTokenMs: 91 })), 'direct runtime reported time to first token 91 of 90 ms'],
  ];
  for (const [tweak, message] of cases) {
    assert.equal(firstErrorLine(await run(tweak)), `Error: ${message}`, message);
  }
}

// --- HTTP server -------------------------------------------------------------

async function rawRequest(url, request) {
  const { port } = new URL(url);
  return new Promise((resolve, reject) => {
    const socket = net.connect(Number(port), '127.0.0.1', () => socket.write(request));
    const chunks = [];
    socket.on('data', (chunk) => chunks.push(chunk));
    socket.on('end', () => resolve(Buffer.concat(chunks)));
    socket.on('error', reject);
  });
}

function parseResponse(buffer) {
  const split = buffer.indexOf('\r\n\r\n');
  const [status, ...lines] = buffer.subarray(0, split).toString('latin1').split('\r\n');
  const headers = lines.map((line) => [line.slice(0, line.indexOf(':')), line.slice(line.indexOf(':') + 2)]);
  return { status, headers, body: buffer.subarray(split + 4) };
}

{
  const webRoot = path.join(tmp, 'web');
  fs.mkdirSync(path.join(webRoot, 'sub'), { recursive: true });
  fs.writeFileSync(path.join(webRoot, 'index.html'), '<p>hi</p>');
  fs.writeFileSync(path.join(webRoot, 'model.gguf'), Buffer.alloc(3 * 1024 * 1024, 7));
  fs.writeFileSync(path.join(tmp, 'secret.txt'), 'secret');
  const mtime = new Date('2026-01-02T03:04:05.678Z');
  fs.utimesSync(path.join(webRoot, 'model.gguf'), mtime, mtime);
  const server = await serveIsolated(webRoot);
  try {
    assert.match(server.url, /^http:\/\/127\.0\.0\.1:\d+\/index\.html$/);
    const isolation = ISOLATION_HEADERS.map(([name, value]) => `${name}: ${value}`);

    // Range is ignored: the probe gets a full 200, and the connection closes
    // after one response like Python's HTTP/1.0 server.
    const ranged = parseResponse(await rawRequest(server.url,
      'GET /model.gguf?x=1#f HTTP/1.1\r\nHost: a\r\nRange: bytes=0-0\r\nConnection: keep-alive\r\n\r\n'));
    assert.equal(ranged.status, 'HTTP/1.1 200 OK');
    const headerLines = ranged.headers.map(([name, value]) => `${name}: ${value}`);
    assert.deepEqual(headerLines.slice(0, 8), [
      'Content-type: application/octet-stream',
      `Content-Length: ${3 * 1024 * 1024}`,
      'Last-Modified: Fri, 02 Jan 2026 03:04:05 GMT',
      ...isolation,
    ]);
    assert.ok(headerLines.includes('Connection: close'));
    assert.ok(!headerLines.some((line) => /^(Content-Range|Accept-Ranges):/i.test(line)));
    assert.equal(ranged.body.length, 3 * 1024 * 1024);

    const head = parseResponse(await rawRequest(server.url, 'HEAD /index.html HTTP/1.0\r\n\r\n'));
    assert.equal(head.status, 'HTTP/1.1 200 OK');
    assert.deepEqual(head.headers.slice(0, 2), [['Content-type', 'text/html'], ['Content-Length', '9']]);
    assert.equal(head.body.length, 0);

    const root = parseResponse(await rawRequest(server.url, 'GET / HTTP/1.0\r\n\r\n'));
    assert.equal(root.body.toString(), '<p>hi</p>');

    const redirect = parseResponse(await rawRequest(server.url, 'GET /sub?q=1 HTTP/1.0\r\n\r\n'));
    assert.equal(redirect.status, 'HTTP/1.1 301 Moved Permanently');
    assert.deepEqual(redirect.headers.slice(0, 2), [['Location', '/sub/?q=1'], ['Content-Length', '0']]);

    // Traversal stays inside the web root.
    for (const target of ['/../secret.txt', '/%2e%2e/secret.txt', '/sub/../../secret.txt', '/nope', '/index.html/', '/sub/']) {
      const missing = parseResponse(await rawRequest(server.url, `GET ${target} HTTP/1.0\r\n\r\n`));
      assert.equal(missing.status, 'HTTP/1.1 404 File not found', target);
      assert.ok(missing.body.includes('<p>Error code explanation: 404 - Nothing matches the given URI.</p>'), target);
      assert.deepEqual(missing.headers.slice(0, 3).map(([name]) => name), ['Connection', 'Content-Type', 'Content-Length']);
      assert.deepEqual(missing.headers.slice(3, 8).map(([name, value]) => `${name}: ${value}`), isolation);
    }
    assert.equal(parseResponse(await rawRequest(server.url, 'GET /sub/../index.html HTTP/1.0\r\n\r\n')).body.toString(), '<p>hi</p>');
    const post = parseResponse(await rawRequest(server.url, 'POST /index.html HTTP/1.0\r\nContent-Length: 0\r\n\r\n'));
    assert.equal(post.status, "HTTP/1.1 501 Unsupported method ('POST')");

    const notModified = parseResponse(await rawRequest(server.url,
      `GET /model.gguf HTTP/1.0\r\nIf-Modified-Since: ${new Date('2026-01-02T03:04:05Z').toUTCString()}\r\n\r\n`));
    assert.equal(notModified.status, 'HTTP/1.1 304 Not Modified');

    // A client that aborts a large body does not stop the server.
    await new Promise((resolve) => {
      const socket = net.connect(Number(new URL(server.url).port), '127.0.0.1', () => {
        socket.write('GET /model.gguf HTTP/1.0\r\n\r\n');
      });
      socket.once('data', () => {
        socket.destroy();
        resolve();
      });
    });
    const after = await fetch(server.url);
    assert.equal(after.status, 200);
    assert.equal(after.headers.get('cross-origin-embedder-policy'), 'require-corp');
    assert.equal(await after.text(), '<p>hi</p>');
  } finally {
    await server.close();
  }
  assert.equal(translatePath('/r', '/a/./b/../c?x#y'), '/r/a/c');
  assert.equal(contentType('x.wasm'), 'application/wasm');
  assert.equal(contentType('x.JS'), 'text/javascript');
  assert.equal(contentType('x.gguf'), 'application/octet-stream');
}

// --- Model download and cache ------------------------------------------------

{
  const origin = path.join(tmp, 'origin');
  fs.mkdirSync(origin);
  const body = Buffer.from('tiny model bytes');
  fs.writeFileSync(path.join(origin, 'model.gguf'), body);
  const sha = createHash('sha256').update(body).digest('hex');
  const cacheDir = path.join(tmp, 'cache');
  const server = await serveIsolated(origin);
  const base = server.url.replace('/index.html', '');
  try {
    const url = `${base}/model.gguf?download=1`;
    const target = await downloadToCache(url, cacheDir, sha.toUpperCase());
    assert.equal(target, path.join(cacheDir, cachedModelName(url)));
    assert.deepEqual(fs.readFileSync(target), body);
    assert.deepEqual(fs.readdirSync(cacheDir), [path.basename(target)]);

    // A 404 names only the redacted location and leaves no .tmp file.
    const missing = `${base.replace('http://', 'http://user:hunter2@')}/missing.gguf?token=secret#frag`;
    await assert.rejects(downloadToCache(`${base}/missing.gguf?token=secret`, cacheDir, sha), (error) => {
      assert.equal(error.message, `failed to download smoke model from ${base}/missing.gguf: HTTP Error 404: File not found`);
      return true;
    });
    await assert.rejects(downloadToCache(missing, cacheDir, sha), (error) => {
      assert.ok(error.message.startsWith(`failed to download smoke model from ${base}/missing.gguf: `), error.message);
      assert.ok(!/hunter2|user|secret|frag/.test(error.message), error.message);
      return true;
    });
    assert.deepEqual(fs.readdirSync(cacheDir), [path.basename(target)]);

    // A downloaded file with the wrong checksum stays as .tmp and fails.
    const wrong = `${base}/model.gguf?wrong=1`;
    await assert.rejects(downloadToCache(wrong, cacheDir, '0'.repeat(64)), {
      message: `model checksum mismatch for ${path.join(cacheDir, cachedModelName(wrong))}.tmp: expected ${'0'.repeat(64)}, got ${sha}`,
    });
  } finally {
    await server.close();
  }
  // A cache hit is checked and never downloaded again, even with the origin gone.
  const url = `${base}/model.gguf?download=1`;
  assert.equal(await downloadToCache(url, cacheDir, sha), path.join(cacheDir, cachedModelName(url)));
  fs.writeFileSync(path.join(cacheDir, cachedModelName(url)), 'corrupt');
  await assert.rejects(downloadToCache(url, cacheDir, sha), /^Error: model checksum mismatch for /);
  await validateHash(path.join(origin, 'model.gguf'), '');

  // The pinned-model smokes require a checksum and a location, and `~`-expand
  // --model-path.
  await assert.rejects(resolvePinnedModel({ modelPath: null, modelUrl: 'x', modelSha256: '', modelCacheDir: cacheDir }),
    { message: 'model SHA-256 is required' });
  await assert.rejects(resolvePinnedModel({ modelPath: null, modelUrl: '', modelSha256: sha, modelCacheDir: cacheDir }),
    { message: '--model-url or --model-path is required' });
  await assert.rejects(resolvePinnedModel({ modelPath: '~/definitely-missing.gguf', modelUrl: '', modelSha256: sha, modelCacheDir: cacheDir }),
    { message: `model path does not exist: ${resolvePath(path.join(process.env.HOME, 'definitely-missing.gguf'))}` });
  assert.equal(
    await resolvePinnedModel({ modelPath: path.join(origin, 'model.gguf'), modelUrl: '', modelSha256: sha, modelCacheDir: cacheDir }),
    resolvePath(path.join(origin, 'model.gguf')),
  );

  // The multimodal smoke's resolve_file: the same rules, failures named by label.
  const pinned = (fields) => resolvePinnedFile({ filePath: null, url: '', expectedSha256: sha, cacheDir, label: 'multimodal projector', ...fields });
  await assert.rejects(pinned({ url: 'x', expectedSha256: '' }), { message: 'multimodal projector SHA-256 is required' });
  await assert.rejects(pinned({}), { message: 'multimodal projector URL or local path is required' });
  await assert.rejects(pinned({ filePath: '~/definitely-missing.gguf' }),
    { message: `multimodal projector path does not exist: ${resolvePath(path.join(process.env.HOME, 'definitely-missing.gguf'))}` });
  await assert.rejects(pinned({ filePath: path.join(origin, 'model.gguf'), expectedSha256: '0'.repeat(64) }),
    { message: `model checksum mismatch for ${resolvePath(path.join(origin, 'model.gguf'))}: expected ${'0'.repeat(64)}, got ${sha}` });
  assert.equal(await pinned({ filePath: path.join(origin, 'model.gguf'), url: 'ignored' }), resolvePath(path.join(origin, 'model.gguf')));
  // A --mmproj-url goes through the cache (`~`-expanded), where a hit is checked
  // and never downloaded again.
  const mmprojUrl = `${base}/mmproj.gguf`;
  const homeCache = path.join(tmp, 'home', 'mm-cache');
  fs.mkdirSync(homeCache, { recursive: true });
  fs.writeFileSync(path.join(homeCache, cachedModelName(mmprojUrl)), body);
  const savedHome = process.env.HOME;
  process.env.HOME = path.join(tmp, 'home');
  try {
    assert.equal(await pinned({ url: mmprojUrl, cacheDir: '~/mm-cache' }), path.join(resolvePath(homeCache), cachedModelName(mmprojUrl)));
  } finally {
    process.env.HOME = savedHome;
  }
}

// --- Exit status and stdout ----------------------------------------------------

// runMain sets the exit status without process.exit(), so a large result piped
// to stdout arrives whole; errors map to 1 with the Python message, and usage
// errors to 2.
{
  const run = (body) => spawnSync(process.execPath, ['--input-type=module', '-e', `
    import { runMain, parseSmokeArgs, writeStdout } from ${JSON.stringify(supportUrl)};
    ${body}
  `], { encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
  const big = run(`await runMain('big', async () => { writeStdout('x'.repeat(4 * 1024 * 1024) + '\\n'); return 1; });`);
  assert.equal(big.status, 1);
  assert.equal(big.stdout.length, 4 * 1024 * 1024 + 1);
  const failed = run(`await runMain('grammar', async () => { throw new Error('dist directory does not exist: /x'); });`);
  assert.equal(failed.status, 1);
  assert.equal(failed.stdout, '');
  assert.equal(failed.stderr, 'grammar browser smoke failed: dist directory does not exist: /x\n');
  const usage = run(`await runMain('grammar', async () => parseSmokeArgs(['--bad'], {
    prog: 'grammar_browser_smoke.mjs', description: 'd',
    options: [{ flag: '--n-ctx', type: 'int', default: () => 1, help: 'h' }],
  }));`);
  assert.equal(usage.status, 2);
  assert.equal(usage.stderr, 'usage: grammar_browser_smoke.mjs [-h] [--n-ctx N_CTX]\ngrammar_browser_smoke.mjs: error: unrecognized arguments: --bad\n');
  const help = run(`await runMain('grammar', async () => parseSmokeArgs(['--he'], {
    prog: 'grammar_browser_smoke.mjs', description: 'd',
    options: [{ flag: '--n-ctx', type: 'int', default: () => 1, help: 'h' }],
  }));`);
  assert.equal(help.status, 0);
  assert.ok(help.stdout.startsWith('usage: grammar_browser_smoke.mjs [-h] [--n-ctx N_CTX]\n'));
}

// The smokes fail before the browser for bad inputs, with the Python messages.
{
  const smoke = (name, args, extraEnv = {}) => spawnSync(process.execPath, [path.join(rootDir, 'scripts', name), ...args], {
    encoding: 'utf8',
    env: { PATH: process.env.PATH, HOME: process.env.HOME, ...extraEnv },
  });
  const missingDist = path.join(tmp, 'no-dist');
  for (const [name, label] of [
    ['state_persistence_browser_smoke.mjs', 'state persistence'],
    ['grammar_browser_smoke.mjs', 'grammar'],
    ['next_token_scores_browser_smoke.mjs', 'next-token scores'],
    ['multimodal_browser_smoke.mjs', 'multimodal'],
  ]) {
    const result = smoke(name, ['--dist-dir', missingDist]);
    assert.equal(result.status, 1, name);
    assert.equal(result.stdout, '');
    assert.equal(result.stderr, `${label} browser smoke failed: dist directory does not exist: ${resolvePath(missingDist)}\n`);
    assert.equal(smoke(name, ['--timeout-ms', 'x']).status, 2, name);
  }
  const emptyDist = path.join(tmp, 'empty-dist');
  fs.mkdirSync(emptyDist);
  const noArtifacts = smoke('state_persistence_browser_smoke.mjs', ['--dist-dir', emptyDist]);
  assert.equal(noArtifacts.status, 1);
  assert.equal(noArtifacts.stderr,
    `state persistence browser smoke failed: missing bridge artifact: ${path.join(resolvePath(emptyDist), 'llama_webgpu_bridge.js')}\n`);
  const noSha = smoke('grammar_browser_smoke.mjs', ['--dist-dir', emptyDist]);
  assert.equal(noSha.stderr, 'grammar browser smoke failed: model SHA-256 is required\n');
  const badEnv = smoke('next_token_scores_browser_smoke.mjs', [], { LLAMA_WEBGPU_NEXT_TOKEN_SCORES_TIMEOUT_MS: 'soon' });
  assert.equal(badEnv.status, 1);
  assert.equal(badEnv.stderr, "next-token scores browser smoke failed: invalid literal for int() with base 10: 'soon'\n");
  // The multimodal smoke checks the model, then the projector, before the dist files.
  const modelFile = path.join(tmp, 'origin', 'model.gguf');
  const modelSha = createHash('sha256').update(fs.readFileSync(modelFile)).digest('hex');
  const multimodal = (args, extraEnv) => smoke('multimodal_browser_smoke.mjs', ['--dist-dir', emptyDist, ...args], extraEnv);
  assert.equal(multimodal([]).stderr, 'multimodal browser smoke failed: multimodal model SHA-256 is required\n');
  assert.equal(multimodal(['--model-sha256', modelSha]).stderr,
    'multimodal browser smoke failed: multimodal model URL or local path is required\n');
  assert.equal(multimodal(['--model-sha256', modelSha, '--model-path', modelFile]).stderr,
    'multimodal browser smoke failed: multimodal projector SHA-256 is required\n');
  assert.equal(multimodal(['--model-sha256', modelSha, '--model-path', modelFile, '--mmproj-sha256', modelSha]).stderr,
    'multimodal browser smoke failed: multimodal projector URL or local path is required\n');
  const noBridge = multimodal([], {
    LLAMA_WEBGPU_MULTIMODAL_MODEL_PATH: modelFile, LLAMA_WEBGPU_MULTIMODAL_MODEL_SHA256: modelSha,
    LLAMA_WEBGPU_MULTIMODAL_MMPROJ_PATH: modelFile, LLAMA_WEBGPU_MULTIMODAL_MMPROJ_SHA256: modelSha,
  });
  assert.equal(noBridge.status, 1);
  assert.equal(noBridge.stdout, '');
  assert.equal(noBridge.stderr,
    `multimodal browser smoke failed: missing bridge artifact: ${path.join(resolvePath(emptyDist), 'llama_webgpu_bridge.js')}\n`);
}

// stageFile hard-links when it can, like os.link, and copies otherwise (here
// because the target exists, as os.link raised FileExistsError).
{
  const source = path.join(tmp, 'stage-source.bin');
  fs.writeFileSync(source, 'stage');
  const linked = path.join(tmp, 'stage-linked.bin');
  await stageFile(source, linked);
  assert.equal(fs.statSync(linked).ino, fs.statSync(source).ino);
  const copied = path.join(tmp, 'stage-copied.bin');
  fs.writeFileSync(copied, 'old');
  await stageFile(source, copied);
  assert.equal(fs.readFileSync(copied, 'utf8'), 'stage');
  assert.notEqual(fs.statSync(copied).ino, fs.statSync(source).ino);
}

console.log('Browser smoke support contract passed');
