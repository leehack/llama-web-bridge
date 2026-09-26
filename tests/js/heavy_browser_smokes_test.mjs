// Contract tests for the Node speech-to-text, text-to-speech and decision
// smokes: their command lines, the fixture and payload checks, the harness pages
// and the files they write. Expected values were produced by the Python smokes
// they replace and Python's json/repr. No browser is started here.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { pyB64Decode, pyJson, pyJsonLoads, readPyText, resolvePath } from '../../scripts/browser_smoke_support.mjs';
import * as decisionSmoke from '../../scripts/decision_browser_smoke.mjs';
import * as speechSmoke from '../../scripts/speech_to_text_browser_smoke.mjs';
import * as ttsSmoke from '../../scripts/text_to_speech_browser_smoke.mjs';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'heavy-browser-smokes-test-'));
process.on('exit', () => fs.rmSync(tmp, { recursive: true, force: true }));
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

// --- Speech fixture -------------------------------------------------------------

// The speech smoke's defaults are speech_to_text_fixture.json, which
// release_qualification.py reads too.
{
  const fixture = JSON.parse(fs.readFileSync(path.join(rootDir, 'scripts/speech_to_text_fixture.json'), 'utf8'));
  assert.deepEqual(Object.keys(fixture).sort(), ['audio_sha256', 'audio_url', 'expected_text']);
  assert.equal(speechSmoke.DEFAULT_AUDIO_URL, fixture.audio_url);
  assert.equal(speechSmoke.DEFAULT_AUDIO_SHA256, fixture.audio_sha256);
  assert.equal(speechSmoke.DEFAULT_EXPECTED_TEXT, fixture.expected_text);
  // wave.open(..., "wb") with 4 s of mono 16 kHz PCM16 silence.
  const silence = speechSmoke.silenceWav();
  assert.equal(silence.length, 128044);
  assert.equal(sha256(silence), 'f751fd7094be088dc785125e4a1a0222a17c2bd20469aa31728ca5e515ed5809');
}

// --- Command lines ----------------------------------------------------------------

{
  const saved = { ...process.env };
  try {
    for (const key of Object.keys(process.env)) if (key.startsWith('LLAMA_WEBGPU_SPEECH_')) delete process.env[key];
    process.env.BRIDGE_DIST_DIR = '/d';
    assert.deepEqual(speechSmoke.parseArgs([]), {
      distDir: '/d', timeoutMs: 900000, modelUrl: '', modelPath: null, modelSha256: '', mmprojUrl: '', mmprojPath: null,
      mmprojSha256: '', audioUrl: speechSmoke.DEFAULT_AUDIO_URL, audioPath: null, audioSha256: speechSmoke.DEFAULT_AUDIO_SHA256,
      expect: speechSmoke.DEFAULT_EXPECTED_TEXT, memoryMode: 'all', modelCacheDir: '~/.cache/llama-web-bridge/speech-smoke-models',
      artifactsDir: null,
    });
    Object.assign(process.env, {
      LLAMA_WEBGPU_SPEECH_TIMEOUT_MS: '5', LLAMA_WEBGPU_SPEECH_MODEL_URL: 'https://h/m.gguf', LLAMA_WEBGPU_SPEECH_MODEL_PATH: '/m',
      LLAMA_WEBGPU_SPEECH_MODEL_SHA256: 'a', LLAMA_WEBGPU_SPEECH_MMPROJ_URL: 'https://h/p.gguf', LLAMA_WEBGPU_SPEECH_MMPROJ_PATH: '/p',
      LLAMA_WEBGPU_SPEECH_MMPROJ_SHA256: 'b', LLAMA_WEBGPU_SPEECH_AUDIO_URL: 'https://h/a.wav', LLAMA_WEBGPU_SPEECH_AUDIO_PATH: '/a',
      LLAMA_WEBGPU_SPEECH_AUDIO_SHA256: 'c', LLAMA_WEBGPU_SPEECH_EXPECTED_TEXT: 'hi', LLAMA_WEBGPU_SPEECH_MEMORY_MODE: 'wasm64',
      LLAMA_WEBGPU_SPEECH_MODEL_CACHE: '~/c', LLAMA_WEBGPU_SPEECH_ARTIFACTS_DIR: '/art',
    });
    assert.deepEqual(speechSmoke.parseArgs(['--memory-mode', 'wasm32', '--audio-path=/x.wav']), {
      distDir: '/d', timeoutMs: 5, modelUrl: 'https://h/m.gguf', modelPath: '/m', modelSha256: 'a', mmprojUrl: 'https://h/p.gguf',
      mmprojPath: '/p', mmprojSha256: 'b', audioUrl: 'https://h/a.wav', audioPath: '/x.wav', audioSha256: 'c', expect: 'hi',
      memoryMode: 'wasm32', modelCacheDir: '~/c', artifactsDir: '/art',
    });
    assert.equal(speechSmoke.parseArgs([]).audioPath, '/a');
  } finally {
    for (const key of Object.keys(process.env)) if (!(key in saved)) delete process.env[key];
    Object.assign(process.env, saved);
  }

  // text-to-speech and decision read no environment; argparse defaults.
  assert.deepEqual(ttsSmoke.parseArgs(['--model-path', '/m', '--mmproj-path', '/p']), {
    distDir: 'dist', modelPath: '/m', mmprojPath: '/p', modelSha256: '', mmprojSha256: '', speakerAudioPath: null,
    speakerAudioSha256: '', prompt: 'Hello from llamadart.', maxFrames: 96, gpuLayers: 0, memoryMode: 'all', runtimeMode: 'all',
    timeoutMs: 1200000, skipCancellation: false, artifactsDir: null,
  });
  assert.equal(ttsSmoke.parseArgs(['--model-path', '/m', '--mmproj-path', '/p', '--skip-c']).skipCancellation, true);
  assert.deepEqual(decisionSmoke.parseArgs(['--model-path', '/m', '--head-path', '/h', '--fixture-path', '/f', '--max-logit-diff', '-1']), {
    distDir: 'dist', modelPath: '/m', headPath: '/h', configPath: null, fixturePath: '/f', modelSha256: '', headSha256: '',
    configSha256: '', gpuLayers: 0, contextSize: 512, headSource: 'url', maxLogitDiff: -1, maxProbabilityDiff: 0.06,
    maxActProbabilityDiff: 0.05, maxActRelativeDiff: 0.05, memoryMode: 'all', runtimeMode: 'all', timeoutMs: 1800000,
    artifactsDir: null,
  });
  assert.equal(decisionSmoke.parseArgs(['--model-path', '/m', '--head-path', '/h', '--fixture-path', '/f', '--max-act-r', ' 1_0 '])
    .maxActRelativeDiff, 10);
  const usageError = (parse, argv) => {
    try {
      parse(argv);
    } catch (error) {
      return error.message;
    }
    return null;
  };
  assert.equal(usageError(ttsSmoke.parseArgs, ['--mmproj-path', '/p', '--bogus']),
    'the following arguments are required: --model-path');
  assert.equal(usageError(ttsSmoke.parseArgs, ['--model-path', '/m', '--mmproj-path', '/p', '--skip-cancellation=1']),
    "argument --skip-cancellation: ignored explicit argument '1'");
  assert.equal(usageError(ttsSmoke.parseArgs, ['--model-path', '/m', '--mmproj-path', '/p', '--runtime-mode', 'both']),
    "argument --runtime-mode: invalid choice: 'both' (choose from 'all', 'direct', 'worker')");
  assert.equal(usageError(decisionSmoke.parseArgs, []),
    'the following arguments are required: --model-path, --head-path, --fixture-path');
  assert.equal(usageError(decisionSmoke.parseArgs, ['--model-path', '/m', '--head-path', '/h', '--fixture-path', '/f', '--max-logit-diff', 'x']),
    "argument --max-logit-diff: invalid float value: 'x'");
}

// --- Decision fixture ----------------------------------------------------------------

// load_fixture as Python ran it: json.loads keeps 2.0 a float and big ints
// exact, str() of a KeyError is the key's repr, and bool is an int.
{
  const cases = [
  ["[1]", { error: "fixture is not a JSON object" }],
  ["{\"rows\": []}", { error: "fixture has no rows" }],
  ["{\"rows\": {\"a\": 1}}", { error: "fixture has no rows" }],
  ["{\"rows\": [1], \"specialTokens\": {\"cls\": 1, \"sep\": 2}}", { error: "fixture specialTokens must name cls, sep and mask ids" }],
  ["{\"rows\": [1], \"specialTokens\": {\"cls\": 1.0, \"sep\": 2, \"mask\": 3}}", { error: "fixture specialTokens must name cls, sep and mask ids" }],
  ["{\"rows\": [[1]], \"specialTokens\": {\"cls\": true, \"sep\": 2, \"mask\": 3}}", { error: "'list' object has no attribute 'get'" }],
  ["{\"rows\": [{\"question\": null}], \"specialTokens\": {\"cls\": 1, \"sep\": 2, \"mask\": 3}}", { error: "'NoneType' object has no attribute 'get'" }],
  ["{\"rows\": [{\"question\": {\"type\": \"multi\"}}], \"specialTokens\": {\"cls\": 1, \"sep\": 2, \"mask\": 3}}", { error: "fixture row 0 has question type 'multi'" }],
  ["{\"rows\": [{\"question\": {\"type\": 1.5}}], \"specialTokens\": {\"cls\": 1, \"sep\": 2, \"mask\": 3}}", { error: "fixture row 0 has question type 1.5" }],
  ["{\"rows\": [{\"question\": {\"type\": [\"choice\"]}}], \"specialTokens\": {\"cls\": 1, \"sep\": 2, \"mask\": 3}}", { error: "cannot use 'list' as a dict key (unhashable type: 'list')" }],
  ["{\"rows\": [{\"ids\": [1]}], \"specialTokens\": {\"cls\": 1, \"sep\": 2, \"mask\": 3}}", { error: "fixture row 0 has question type None" }],
  ["{\"rows\": [{\"question\": {\"type\": \"choice\"}, \"markers\": [1, 2], \"rawLogits\": [0.5]}], \"specialTokens\": {\"cls\": 1, \"sep\": 2, \"mask\": 3}}", { error: "fixture row 0 has mismatched markers and logits" }],
  ["{\"rows\": [{\"question\": {\"type\": \"choice\"}, \"markers\": null, \"rawLogits\": [0.5]}], \"specialTokens\": {\"cls\": 1, \"sep\": 2, \"mask\": 3}}", { error: "object of type 'NoneType' has no len()" }],
  ["{\"rows\": [{\"question\": {\"type\": \"choice\"}, \"markers\": [1], \"rawLogits\": [0.5]}], \"specialTokens\": {\"cls\": 1, \"sep\": 2, \"mask\": 3}}", { error: "'ids'" }],
  ["{\"rows\": [{\"question\": {\"type\": \"score\"}, \"ids\": [1], \"markers\": [1], \"rawLogits\": [0.5]}], \"specialTokens\": {\"cls\": 1, \"sep\": 2, \"mask\": 3}}", { error: "'rawActLogits'" }],
  ["{\"rows\": [{\"id\": 7, \"question\": {\"type\": \"noul\"}, \"ids\": [1, 2.0, 12345678901234567890], \"markers\": \"ab\", \"rawLogits\": {\"x\": 1, \"y\": 2}, \"rawActLogits\": [1e-05, -0.0, -0, 1E400, 0.1]}, {\"question\": {\"type\": \"score\", \"extra\": 1}, \"ids\": [], \"markers\": [], \"rawLogits\": [], \"rawActLogits\": []}], \"specialTokens\": {\"mask\": 3, \"cls\": 1, \"sep\": true, \"pad\": 4.0}}", { page: "{\"specialTokens\": {\"mask\": 3, \"cls\": 1, \"sep\": true, \"pad\": 4.0}, \"sequences\": [{\"id\": 7, \"tokens\": [1, 2.0, 12345678901234567890], \"markers\": \"ab\", \"questionType\": 2, \"rawLogits\": {\"x\": 1, \"y\": 2}, \"rawActLogits\": [1e-05, -0.0, 0, Infinity, 0.1]}, {\"id\": \"1\", \"tokens\": [], \"markers\": [], \"questionType\": 1, \"rawLogits\": [], \"rawActLogits\": []}]}" }],
  ];
  for (const [index, [text, expected]] of cases.entries()) {
    const file = path.join(tmp, `fixture-${index}.json`);
    fs.writeFileSync(file, text);
    if (expected.error !== undefined) {
      await assert.rejects(decisionSmoke.loadFixture(file), { message: expected.error }, text);
    } else {
      assert.equal(pyJson(await decisionSmoke.loadFixture(file)), expected.page, text);
    }
  }
  // Path.read_text: universal newlines, a BOM kept, and strict UTF-8.
  const crlf = path.join(tmp, 'crlf.json');
  fs.writeFileSync(crlf, '{"a": 1}\r\n\r');
  assert.equal(await readPyText(crlf), '{"a": 1}\n\n');
  fs.writeFileSync(crlf, '﻿{}');
  assert.equal(await readPyText(crlf), '﻿{}');
  fs.writeFileSync(crlf, Buffer.from([0x7b, 0xff, 0x7d]));
  await assert.rejects(readPyText(crlf), /'utf-8' codec can't decode/);
  assert.throws(() => pyJsonLoads('[NaN]'));
}

// --- Decision parity ----------------------------------------------------------------

// check_parity's messages, with the thresholds printed as Python floats.
{
  const args = { maxLogitDiff: 0.25, maxProbabilityDiff: 0.06, maxActProbabilityDiff: 0.05, maxActRelativeDiff: 1 };
  const cases = [
  ["good", [{"memoryMode": "wasm32", "runtimeMode": "direct", "worstLogitDiff": 0.1, "worstProbabilityDiff": 0, "worstActProbabilityDiff": 0.01, "worstActRelativeDiff": 1, "argmaxChanges": [], "actDecisionChanges": []}], null],
  ["logit", [{"memoryMode": "wasm32", "runtimeMode": "direct", "worstLogitDiff": 0.30000001192092896, "worstProbabilityDiff": 0, "worstActProbabilityDiff": 0.01, "worstActRelativeDiff": 1, "argmaxChanges": [], "actDecisionChanges": []}], "wasm32/direct: worst logit difference 0.30000001192092896 exceeds 0.25"],
  ["prob", [{"memoryMode": null, "runtimeMode": "direct", "worstLogitDiff": 0.1, "worstProbabilityDiff": 1e-05, "worstActProbabilityDiff": 0.01, "worstActRelativeDiff": 1, "argmaxChanges": [], "actDecisionChanges": []}], null],
  ["prob_over", [{"memoryMode": "wasm32", "runtimeMode": "direct", "worstLogitDiff": 0.1, "worstProbabilityDiff": 0.5, "worstActProbabilityDiff": 0.01, "worstActRelativeDiff": 1, "argmaxChanges": [], "actDecisionChanges": []}], "wasm32/direct: worst probability difference 0.5 exceeds 0.06"],
  ["act_prob", [{"memoryMode": "wasm32", "runtimeMode": "direct", "worstLogitDiff": 0.1, "worstProbabilityDiff": 0, "worstActProbabilityDiff": 2, "worstActRelativeDiff": 1, "argmaxChanges": [], "actDecisionChanges": []}], "wasm32/direct: worst act probability difference 2 exceeds 0.05"],
  ["act_rel", [{"memoryMode": "wasm32", "runtimeMode": "direct", "worstLogitDiff": 0.1, "worstProbabilityDiff": 0, "worstActProbabilityDiff": 0.01, "worstActRelativeDiff": 1.5, "argmaxChanges": [], "actDecisionChanges": []}], "wasm32/direct: worst act logit relative difference 1.5 exceeds 1.0"],
  ["missing", [{"memoryMode": "wasm32", "runtimeMode": "direct", "worstProbabilityDiff": 0, "worstActProbabilityDiff": 0.01, "worstActRelativeDiff": 1, "argmaxChanges": [], "actDecisionChanges": []}], "wasm32/direct: worst logit difference None exceeds 0.25"],
  ["none", [{"memoryMode": "wasm32", "runtimeMode": "direct", "worstLogitDiff": null, "worstProbabilityDiff": 0, "worstActProbabilityDiff": 0.01, "worstActRelativeDiff": 1, "argmaxChanges": [], "actDecisionChanges": []}], "'<=' not supported between instances of 'NoneType' and 'float'"],
  ["string", [{"memoryMode": "wasm32", "runtimeMode": "direct", "worstLogitDiff": "0.1", "worstProbabilityDiff": 0, "worstActProbabilityDiff": 0.01, "worstActRelativeDiff": 1, "argmaxChanges": [], "actDecisionChanges": []}], "'<=' not supported between instances of 'str' and 'float'"],
  ["bool", [{"memoryMode": "wasm32", "runtimeMode": "direct", "worstLogitDiff": true, "worstProbabilityDiff": 0, "worstActProbabilityDiff": 0.01, "worstActRelativeDiff": 1, "argmaxChanges": [], "actDecisionChanges": []}], "wasm32/direct: worst logit difference True exceeds 0.25"],
  ["changes", [{"memoryMode": "wasm32", "runtimeMode": "direct", "worstLogitDiff": 0.1, "worstProbabilityDiff": 0, "worstActProbabilityDiff": 0.01, "worstActRelativeDiff": 1, "argmaxChanges": ["row-1", "row 'two'"], "actDecisionChanges": []}], "wasm32/direct: decisions changed against the reference: options ['row-1', \"row 'two'\"], act []"],
  ["changes_act", [{"memoryMode": "wasm32", "runtimeMode": "direct", "worstLogitDiff": 0.1, "worstProbabilityDiff": 0, "worstActProbabilityDiff": 0.01, "worstActRelativeDiff": 1, "argmaxChanges": null, "actDecisionChanges": [3]}], "wasm32/direct: decisions changed against the reference: options None, act [3]"],
  ["changes_falsy", [{"memoryMode": "wasm32", "runtimeMode": "direct", "worstLogitDiff": 0.1, "worstProbabilityDiff": 0, "worstActProbabilityDiff": 0.01, "worstActRelativeDiff": 1, "argmaxChanges": 0, "actDecisionChanges": ""}], null],
  ["not_dict", [5], "'int' object has no attribute 'get'"],
  ];
  for (const [name, modeResults, expected] of cases) {
    if (expected === null) decisionSmoke.checkParity({ modeResults }, args);
    else assert.throws(() => decisionSmoke.checkParity({ modeResults }, args), { message: expected }, name);
  }
  // Integral thresholds print as Python floats ("1.0"), not JS numbers ("1").
  const integral = { maxLogitDiff: 1, maxProbabilityDiff: 2, maxActProbabilityDiff: 3, maxActRelativeDiff: 4 };
  const fine = { memoryMode: 'wasm64', runtimeMode: 'worker', worstLogitDiff: 0, worstProbabilityDiff: 0,
    worstActProbabilityDiff: 0, worstActRelativeDiff: 0, argmaxChanges: [], actDecisionChanges: [] };
  for (const [key, message] of [
    ['worstLogitDiff', 'worst logit difference 9 exceeds 1.0'],
    ['worstProbabilityDiff', 'worst probability difference 9 exceeds 2.0'],
    ['worstActProbabilityDiff', 'worst act probability difference 9 exceeds 3.0'],
    ['worstActRelativeDiff', 'worst act logit relative difference 9 exceeds 4.0'],
  ]) {
    assert.throws(() => decisionSmoke.checkParity({ modeResults: [{ ...fine, [key]: 9 }] }, integral),
      { message: `wasm64/worker: ${message}` }, key);
  }
  // No modeResults is no modes; the printed payload drops only the console.
  decisionSmoke.checkParity({ ok: true }, args);
  assert.deepEqual(decisionSmoke.printedPayload({ ok: true, console: ['x'], modeResults: [] }), { ok: true, modeResults: [] });
}

// --- Text-to-speech WAV artifacts --------------------------------------------------

// Each mode's page-encoded WAV is written byte for byte under a name built from
// its modes, and _wavBase64 never reaches the printed payload.
{
  const wav = Buffer.concat([Buffer.from('RIFF'), Buffer.alloc(40, 1), Buffer.from([0xff, 0x7f, 0x00, 0x80])]);
  const payload = {
    ok: true,
    modeResults: [
      { memoryMode: 'wasm64', runtimeMode: 'direct', _wavBase64: wav.toString('base64') },
      { runtimeMode: 'worker', _wavBase64: wav.toString('base64') },
      { memoryMode: undefined, runtimeMode: null, _wavBase64: wav.toString('base64') },
      { memoryMode: 'wasm64', runtimeMode: 'worker', _wavBase64: '' },
      'not a mode',
    ],
  };
  const artifacts = path.join(tmp, 'tts-artifacts');
  fs.mkdirSync(artifacts);
  await ttsSmoke.extractWavArtifacts(payload, artifacts);
  assert.deepEqual(payload.modeResults, [
    { memoryMode: 'wasm64', runtimeMode: 'direct', audioArtifact: 'text-to-speech-wasm64-direct.wav' },
    { runtimeMode: 'worker', audioArtifact: 'text-to-speech-unknown-worker.wav' },
    { memoryMode: undefined, runtimeMode: null, audioArtifact: 'text-to-speech-None-None.wav' },
    { memoryMode: 'wasm64', runtimeMode: 'worker' },
    'not a mode',
  ]);
  for (const name of ['text-to-speech-wasm64-direct.wav', 'text-to-speech-unknown-worker.wav', 'text-to-speech-None-None.wav']) {
    assert.deepEqual(fs.readFileSync(path.join(artifacts, name)), wav);
  }
  const dropped = { ok: true, modeResults: [{ memoryMode: 'wasm64', runtimeMode: 'direct', _wavBase64: 'AAAA' }] };
  await ttsSmoke.extractWavArtifacts(dropped, null);
  assert.deepEqual(dropped.modeResults, [{ memoryMode: 'wasm64', runtimeMode: 'direct' }]);
  await ttsSmoke.extractWavArtifacts({ ok: false, error: 'x' }, artifacts);
  await assert.rejects(ttsSmoke.extractWavArtifacts({ ok: true, modeResults: null }, artifacts), {
    message: "'NoneType' object is not iterable",
  });
  assert.throws(() => pyB64Decode('AAA'), { message: 'base64 data is not canonical' });
  assert.throws(() => pyB64Decode(5), { message: "argument should be a bytes-like object or ASCII string, not 'int'" });

  ttsSmoke.checkPayload({ modeResults: [{}, {}] }, ['wasm64'], ['direct', 'worker']);
  assert.throws(() => ttsSmoke.checkPayload({ modeResults: [{}] }, ['wasm64'], ['direct', 'worker']), { message: 'mode results are incomplete' });
  assert.throws(() => ttsSmoke.checkPayload({}, ['wasm64'], ['direct']), { message: 'mode results are incomplete' });
  speechSmoke.checkPayload({ modeResults: [1, 2, 3, 4] }, ['wasm32', 'wasm64']);
  assert.throws(() => speechSmoke.checkPayload({ modeResults: [1, 2] }, ['wasm32', 'wasm64']), {
    message: 'speech-to-text mode results are incomplete',
  });
}

// --- Harness pages ------------------------------------------------------------------

// The pages embed their configuration as Python's json.dumps did, and the
// page code keeps its escapes (a regex's \s, a template's ${}).
{
  const speech = speechSmoke.renderHarness({ expectedText: 'café "x"', audioSha256: 'ab', memoryModes: ['wasm64'] });
  assert.ok(speech.includes('    const expected = normalizeTranscript("caf\\u00e9 \\"x\\"");\n'));
  assert.ok(speech.includes('    const memoryModes = ["wasm64"];\n'));
  assert.ok(speech.includes("    .replace(/^\\s*language\\s+[^<\\r\\n]+?\\s*<asr_text>\\s*/i, '')\n"));
  assert.ok(speech.includes('        sha256: "ab",\n'));
  const tts = ttsSmoke.renderHarness({
    prompt: 'Hi', modelSha256: 'm', mmprojSha256: 'p', speakerAudioSha256: null, memoryModes: ['wasm64'],
    runtimeModes: ['direct', 'worker'], maxFrames: 24, gpuLayers: 99, testCancellation: false,
  });
  assert.ok(tts.includes('    const speakerAudio = false\n'));
  assert.ok(tts.includes('            nGpuLayers: 99,\n'));
  assert.ok(tts.includes('            maxFrames: 24,\n'));
  assert.ok(tts.includes('          if (false) {\n'));
  assert.ok(tts.includes('      speakerAudioSha256: null,\n'));
  assert.ok(tts.includes('    console.log(`tts-smoke-stage:${stage}`);\n'));
  const fixtureFile = path.join(tmp, 'page-fixture.json');
  fs.writeFileSync(fixtureFile, '{"specialTokens": {"cls": 1, "sep": 2, "mask": 3}, "rows": [{"question": {"type": "choice"}, '
    + '"ids": [1, 2], "markers": [1, 1, 1], "rawLogits": [1.0, -0.0, 5e-05], "rawActLogits": [3]}]}');
  const decision = decisionSmoke.renderHarness({
    fixture: await decisionSmoke.loadFixture(fixtureFile), configJson: null, memoryModes: ['wasm32'], runtimeModes: ['worker'],
    headSource: 'bytes', gpuLayers: 0, contextSize: 64,
  });
  assert.ok(decision.includes('"rawLogits": [1.0, -0.0, 5e-05], "rawActLogits": [3]}]};\n'));
  assert.ok(decision.includes('    const configJson = null;\n    const headSource = "bytes";\n'));
  assert.ok(decision.includes('            nCtx: 64,\n'));
  assert.ok(decision.includes('            /Decision head \\d+ is not loaded/,\n'));
  for (const page of [speech, tts, decision]) {
    assert.ok(page.startsWith('\n<!doctype html>\n<meta charset="utf-8">\n'));
    assert.ok(page.endsWith('})();\n</script>\n'));
  }
}

// --- Failures before the browser --------------------------------------------------

// The smokes fail before the browser for bad inputs, with the Python messages.
{
  const smoke = (name, args) => spawnSync(process.execPath, [path.join(rootDir, 'scripts', name), ...args], {
    encoding: 'utf8',
    env: { PATH: process.env.PATH, HOME: process.env.HOME },
  });
  const dist = path.join(tmp, 'dist');
  fs.mkdirSync(dist);
  const model = path.join(tmp, 'model.gguf');
  fs.writeFileSync(model, 'model');
  const modelSha = sha256('model');
  const failed = (result, label, message) => {
    assert.equal(result.status, 1, result.stderr);
    assert.equal(result.stdout, '');
    assert.equal(result.stderr, `${label} browser smoke failed: ${message}\n`);
  };
  const missing = path.join(tmp, 'missing');
  failed(smoke('speech_to_text_browser_smoke.mjs', ['--dist-dir', missing]), 'speech-to-text',
    `dist directory does not exist: ${resolvePath(missing)}`);
  failed(smoke('speech_to_text_browser_smoke.mjs', ['--dist-dir', dist]), 'speech-to-text', 'Qwen3-ASR model SHA-256 is required');
  const speechInputs = ['--dist-dir', dist, '--model-path', model, '--model-sha256', modelSha, '--mmproj-path', model,
    '--mmproj-sha256', modelSha.toUpperCase(), '--audio-path', model, '--audio-sha256', modelSha];
  failed(smoke('speech_to_text_browser_smoke.mjs', [...speechInputs.slice(0, 6), '--mmproj-sha256', modelSha]), 'speech-to-text',
    'Qwen3-ASR projector URL or local path is required');
  failed(smoke('speech_to_text_browser_smoke.mjs', [...speechInputs, '--expect', ' 　\n']), 'speech-to-text',
    'expected transcript is required');
  failed(smoke('speech_to_text_browser_smoke.mjs', speechInputs), 'speech-to-text',
    `missing bridge artifact: ${path.join(resolvePath(dist), 'llama_webgpu_bridge.js')}`);
  assert.equal(smoke('speech_to_text_browser_smoke.mjs', ['--memory-mode', 'wasm16']).status, 2);

  const ttsInputs = ['--dist-dir', dist, '--model-path', model, '--mmproj-path', model];
  failed(smoke('text_to_speech_browser_smoke.mjs', [...ttsInputs.slice(0, 4), '--mmproj-path', missing]), 'text-to-speech',
    `projector does not exist: ${resolvePath(missing)}`);
  failed(smoke('text_to_speech_browser_smoke.mjs', [...ttsInputs, '--speaker-audio-sha256', 'ab']), 'text-to-speech',
    'speaker audio checksum requires --speaker-audio-path');
  failed(smoke('text_to_speech_browser_smoke.mjs', [...ttsInputs, '--max-frames', '0']), 'text-to-speech', 'max frames must be positive');
  failed(smoke('text_to_speech_browser_smoke.mjs', [...ttsInputs, '--mmproj-sha256', '00']), 'text-to-speech', 'projector checksum mismatch');
  failed(smoke('text_to_speech_browser_smoke.mjs', [...ttsInputs, '--model-sha256', modelSha.toUpperCase()]), 'text-to-speech',
    `missing bridge artifact: ${path.join(resolvePath(dist), 'llama_webgpu_bridge.js')}`);
  const ttsUsage = smoke('text_to_speech_browser_smoke.mjs', []);
  assert.equal(ttsUsage.status, 2);
  assert.ok(ttsUsage.stderr.endsWith('text_to_speech_browser_smoke.mjs: error: the following arguments are required: --model-path, --mmproj-path\n'));

  const decisionInputs = ['--dist-dir', dist, '--model-path', model, '--head-path', model, '--fixture-path', model];
  failed(smoke('decision_browser_smoke.mjs', [...decisionInputs.slice(0, 6), '--fixture-path', missing]), 'decision',
    `fixture does not exist: ${resolvePath(missing)}`);
  failed(smoke('decision_browser_smoke.mjs', [...decisionInputs, '--config-sha256', 'ab']), 'decision', 'config checksum requires --config-path');
  failed(smoke('decision_browser_smoke.mjs', [...decisionInputs, '--context-size', '0']), 'decision', 'context size must be positive');
  failed(smoke('decision_browser_smoke.mjs', [...decisionInputs, '--head-sha256', '00']), 'decision', 'head checksum mismatch');
  const listFixture = path.join(tmp, 'list-fixture.json');
  fs.writeFileSync(listFixture, '[1]');
  failed(smoke('decision_browser_smoke.mjs', [...decisionInputs.slice(0, 6), '--fixture-path', listFixture]), 'decision',
    'fixture is not a JSON object');
}

console.log('Heavy browser smoke contract passed');
