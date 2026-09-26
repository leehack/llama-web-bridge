// Static API contract checks for versioned Web text-to-speech support.
import { bridgeJsSource, methodBody, readRepoText } from './bridge_js_source.mjs';
import { readNativeCoreSource } from './native_core_source.mjs';

const CORE = readNativeCoreSource();
const TTS = readRepoText('src/llama_webgpu_tts.cpp');
const HEADER = readRepoText('src/llama_webgpu_tts.h');
const JS = bridgeJsSource();
const RUNTIME_JS = readRepoText('js/src/runtime.ts');
const DTS = readRepoText('js/src/llama_webgpu_bridge.d.ts');
const CMAKE = readRepoText('CMakeLists.txt');
const README = readRepoText('README.md');
const API_DOCS = readRepoText('docs/api.md');
// Python's " ".join(text.split()): collapse runs of Python str whitespace.
const API_DOCS_FLAT = API_DOCS
  .split(/[\t\n\v\f\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/)
  .filter(Boolean)
  .join(' ');
const SMOKE = readRepoText('scripts/smoke/text_to_speech.mjs');
const RECOVERY_TEST = readRepoText('tests/bridge/text_to_speech_recovery_test.mjs');

const NATIVE_EXPORTS = [
  'llamadart_webgpu_tts_api_version',
  'llamadart_webgpu_tts_info_json',
  'llamadart_webgpu_tts_start',
  'llamadart_webgpu_tts_step',
  'llamadart_webgpu_tts_progress_json',
  'llamadart_webgpu_tts_write_pcm',
  'llamadart_webgpu_tts_reset',
];

const errors = [];

function require(condition, message) {
  if (!condition) {
    errors.push(message);
  }
}

function includesAll(text, ...needles) {
  return needles.every((needle) => text.includes(needle));
}

require(
  HEADER.includes('LLAMADART_WEBGPU_TTS_API_VERSION = 1'),
  'TTS wrapper API must have an explicit version',
);
for (const symbol of NATIVE_EXPORTS) {
  require(CORE.includes(symbol), `missing core wrapper ${symbol}`);
  require(CMAKE.includes(`_${symbol}`), `missing exported symbol _${symbol}`);
}

require(
  includesAll(
    TTS,
    'mtmd_helper_gen_audio_init',
    'mtmd_helper_gen_audio_step_prompt',
    'mtmd_helper_gen_audio_step_gen',
    'mtmd_helper_gen_audio_get_output',
  ),
  'TTS wrapper must use the upstream stepwise generated-audio API',
);
require(
  includesAll(TTS, 'llama_set_embeddings(tts->llama, true)', 'llama_set_embeddings(tts->llama, false)'),
  'TTS must scope embedding-output mode to the active task',
);
require(
  CORE.includes('g_generation_active || g_tts_active'),
  'shared llama.cpp context operations must reject TTS overlap',
);
require(
  /llamadart_webgpu_tts_start\(.*?g_cancel_requested = false;.*?g_cached_prompt_tokens\.clear\(\);.*?llama_webgpu_tts_start\(g_tts, request\)/s.test(CORE)
    && /llamadart_webgpu_tts_reset\(\) \{.*?g_cancel_requested = false;/s.test(CORE),
  'TTS start must invalidate shared prompt-cache metadata and start/reset must clear the cancellation latch',
);

require(
  (JS.match(/\basync\s+synthesizeSpeech\s*\(/g) ?? []).length >= 2,
  'expected direct-runtime and public synthesizeSpeech methods',
);
require(
  includesAll(JS, "method === 'synthesizeSpeech'", "event: 'progress'", '[value.pcm.buffer]'),
  'worker TTS must forward progress and transfer PCM without copying it back',
);
require(
  includesAll(
    JS,
    '_llamadartTextToSpeech: true',
    'retrying text-to-speech once with CPU fallback',
    'worker_fallback_cpu_text_to_speech',
  ),
  'worker synthesis WebGPU failures must retry once through the CPU recovery path',
);
require(
  includesAll(
    RECOVERY_TEST,
    'nGpuLayers, 0',
    'worker_fallback_cpu_text_to_speech',
    'gpuLayers: 0',
    'AbortError',
    'Worker request timeout',
    'worker timed out',
    'beforeWorkerError: () => controller.abort()',
    'beforeWorkerError: () => bridge.cancel()',
    'CPU recovery must run at most once',
  ),
  'TTS recovery tests must cover GPU failure, worker request timeout, the '
    + 'single-retry limit, CPU no-retry, and cancellation',
);
require(
  includesAll(
    API_DOCS_FLAT,
    'at most one main-thread retry',
    'dispatch workgroup count',
    'worker request timeout',
    'not already loaded CPU-only',
    'preserve the original GPU offload settings',
    'Cancellation is classified before recovery',
  ),
  'public API docs must state the worker TTS retry triggers, its single-retry '
    + 'limit, and that cancellation is never retried',
);
require(
  includesAll(
    DTS,
    'interface TextToSpeechCapabilities',
    'interface TextToSpeechOptions',
    'interface TextToSpeechResult',
  ),
  'TypeScript declarations must expose TTS capabilities, options, and result',
);
require(
  includesAll(API_DOCS, 'getTextToSpeechCapabilities', 'synthesizeSpeech', 'wasm64'),
  'public API docs must document TTS and its memory64 requirement',
);
require(
  includesAll(README, 'Text-to-speech', 'scripts/smoke/text_to_speech.mjs'),
  'README must document TTS and its real-model smoke',
);
require(
  includesAll(
    SMOKE,
    "RUNTIME_MODES = Object.freeze(['direct', 'worker'])",
    "memoryModes.includes('wasm64')",
    'sha256File(modelPath)',
    'sha256File(mmprojPath)',
    "flag: '--gpu-layers'",
    "flag: '--speaker-audio-path'",
    'speakerReferenceTested',
    'output.pcm instanceof Float32Array',
  ),
  'real-model smoke must cover direct/worker memory64 PCM with pinned assets, optional speaker reference, and selectable GPU offload',
);
require(
  includesAll(
    SMOKE,
    'AbortController',
    'preAbortedTested',
    'cancellationTested',
    'unloadTested',
    'runtime reuse after cancellation',
  ),
  'real-model smoke must validate pre-abort, active cancellation, warm reuse, and projector unload',
);
require(
  includesAll(JS, 'const mmprojPath = this._mmProjPath;', 'this._deleteFsFile(mmprojPath);')
    && /if \(rc !== 0\) \{[^}]*?\bthrow\b.*?this\._deleteFsFile\(mmprojPath\);/s.test(
      methodBody(RUNTIME_JS, 'async unloadMultimodalProjector() {'),
    ),
  'projector unload must preserve state on native rejection and release the WasmFS file after success',
);
require(
  new RegExp(
    String.raw`const mmprojPath = this\._mmProjPath;.*?`
      + String.raw`const mmprojFreeRc = Number\(\s*`
      + String.raw`this\._core\.ccall\('llamadart_webgpu_mmproj_free', 'number'.*?`
      + String.raw`llamadart_webgpu_shutdown'.*?this\._deleteFsFile\(mmprojPath\);.*?`
      + String.raw`if \(mmprojFreeRc !== 0\)`,
    's',
  ).test(methodBody(RUNTIME_JS, 'async dispose() {')),
  'runtime dispose must inspect projector unload status and release the staged WasmFS file after shutdown',
);
require(
  /async synthesizeSpeech\(options(?::[^=]*)? = \{\}\).*?if \(this\._textToSpeechActive\).*?this\._ensureTextToSpeechDir\(\)/s.test(JS),
  'direct TTS must reserve the operation before writing temporary speaker input',
);

if (errors.length > 0) {
  console.error('Text-to-speech API contract failed:');
  for (const item of errors) {
    console.error(`- ${item}`);
  }
  process.exit(1);
}

console.log('Text-to-speech API contract passed');
