#!/usr/bin/env node
// Real-model browser smoke for Qwen3-ASR audio ingestion.
//
// Ported from speech_to_text_browser_smoke.py with the same flags, environment
// variables, harness page and output. release_qualification.py runs it as the
// speech-to-text gate, and both read the fixture audio pin and the expected
// transcript from speech_to_text_fixture.json.

import fsp from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import {
  copyBridgeArtifacts,
  copyMemory64Artifacts,
  ensure,
  env,
  expandHome,
  isDirectory,
  parseSmokeArgs,
  pyGet,
  pyJson,
  pyStrip,
  resolvePath,
  resolvePinnedFile,
  runMain,
  runPlaywright,
  sha256File,
  stageFile,
  withServer,
  withTempDir,
  writeStdout,
} from './support.mjs';
import SPEECH_FIXTURE from './speech_to_text_fixture.json' with { type: 'json' };

for (const key of ['audio_url', 'audio_sha256', 'expected_text']) {
  if (typeof SPEECH_FIXTURE[key] !== 'string' || SPEECH_FIXTURE[key] === '') {
    throw new Error(`speech_to_text_fixture.json must hold a non-empty ${key}`);
  }
}

const DESCRIPTION = 'Real-model browser smoke for Qwen3-ASR audio ingestion.';
const DEFAULT_MODEL_CACHE = '~/.cache/llama-web-bridge/speech-smoke-models';
export const DEFAULT_AUDIO_URL = SPEECH_FIXTURE.audio_url;
export const DEFAULT_AUDIO_SHA256 = SPEECH_FIXTURE.audio_sha256;
export const DEFAULT_EXPECTED_TEXT = SPEECH_FIXTURE.expected_text;
export const MEMORY_MODES = Object.freeze(['wasm32', 'wasm64']);
const SILENCE_DURATION_SECONDS = 4;
const SILENCE_SAMPLE_RATE = 16000;

// The deterministic PCM16 WAV used to reject ASR hallucinations: what Python's
// wave module writes for 4 s of mono 16 kHz silence.
export function silenceWav() {
  const dataLength = SILENCE_DURATION_SECONDS * SILENCE_SAMPLE_RATE * 2;
  const bytes = Buffer.alloc(44 + dataLength);
  bytes.write('RIFF', 0, 'latin1');
  bytes.writeUInt32LE(36 + dataLength, 4);
  bytes.write('WAVEfmt ', 8, 'latin1');
  bytes.writeUInt32LE(16, 16);
  bytes.writeUInt16LE(1, 20);
  bytes.writeUInt16LE(1, 22);
  bytes.writeUInt32LE(SILENCE_SAMPLE_RATE, 24);
  bytes.writeUInt32LE(SILENCE_SAMPLE_RATE * 2, 28);
  bytes.writeUInt16LE(2, 32);
  bytes.writeUInt16LE(16, 34);
  bytes.write('data', 36, 'latin1');
  bytes.writeUInt32LE(dataLength, 40);
  return bytes;
}

export function renderHarness({ expectedText, audioSha256, memoryModes }) {
  return `
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge Qwen3-ASR smoke</title>
<pre id="result">pending</pre>
<script type="module">
(async () => {
  const result = document.getElementById('result');
  const finish = (payload) => {
    result.textContent = JSON.stringify(payload);
    window.__smokeResult = payload;
  };
  const assert = (condition, message) => {
    if (!condition) throw new Error(message);
  };
  const normalizeTranscript = (value) => String(value || '')
    .replace(/^\\s*language\\s+[^<\\r\\n]+?\\s*<asr_text>\\s*/i, '')
    .replace(/^\\s*<asr_text>\\s*/i, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();

  try {
    assert(window.crossOriginIsolated, 'test page is not cross-origin isolated');
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    assert(
      typeof LlamaWebGpuBridge === 'function',
      'LlamaWebGpuBridge export was not registered',
    );

    const audioResponse = await fetch('/speech.wav', { cache: 'no-store' });
    assert(audioResponse.ok, \`audio fetch failed: \${audioResponse.status}\`);
    const audioBytes = new Uint8Array(await audioResponse.arrayBuffer());
    assert(audioBytes.byteLength > 44, 'audio fixture is empty');
    const silenceResponse = await fetch('/silence.wav', { cache: 'no-store' });
    assert(silenceResponse.ok, \`silence fetch failed: \${silenceResponse.status}\`);
    const silenceBytes = new Uint8Array(await silenceResponse.arrayBuffer());
    assert(silenceBytes.byteLength > 44, 'silence fixture is empty');

    const expected = normalizeTranscript(${pyJson(expectedText)});
    assert(expected.length > 0, 'expected transcript is empty');
    const memoryModes = ${pyJson(memoryModes)};
    const modeResults = [];

    const verifyMode = async (memoryMode, disableWorker, runtimeMode) => {
      const useMemory64 = memoryMode === 'wasm64';
      const bridge = new LlamaWebGpuBridge({
        disableWorker,
        logLevel: 1,
        preferMemory64: useMemory64,
        coreModuleUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.js' : undefined,
        wasmUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.wasm' : undefined,
      });
      const startedAt = performance.now();
      // Local release qualification records each phase separately, so a
      // regression in model load, decode, cancellation, or silence rejection is
      // attributable instead of hidden inside one aggregate duration.
      const phaseTimingsMs = {
        modelLoadMs: 0,
        projectorLoadMs: 0,
        coldTranscriptMs: 0,
        cancellationMs: 0,
        warmTranscriptMs: 0,
        silenceMs: 0,
      };
      const timePhase = async (name, run) => {
        const phaseStartedAt = performance.now();
        try {
          return await run();
        } finally {
          phaseTimingsMs[name] = Math.round(performance.now() - phaseStartedAt);
        }
      };
      try {
        await timePhase('modelLoadMs', () => bridge.loadModelFromUrl('/qwen3-asr-model.gguf', {
          nCtx: 4096,
          nGpuLayers: 0,
          nThreads: 4,
          nBatch: 512,
          nUbatch: 256,
          useCache: false,
          forceRemoteFetchBackend: false,
        }));
        await timePhase(
          'projectorLoadMs',
          () => bridge.loadMultimodalProjector('/qwen3-asr-mmproj.gguf'),
        );
        assert(bridge.supportsAudio(), \`\${memoryMode} \${runtimeMode} did not report audio support\`);

        const transcribe = (
          bytes = audioBytes,
          signal = undefined,
          onToken = undefined,
        ) => bridge.createCompletion(
            'Transcribe this audio accurately.',
            {
              nPredict: 512,
              temp: 0,
              topK: 1,
              topP: 1,
              penalty: 1,
              seed: 1,
              tokenEventEncoding: 'text',
              signal,
              onToken,
              parts: [{ type: 'audio', bytes }],
            },
          );
        const transcripts = [];
        for (let attempt = 0; attempt < 2; attempt += 1) {
          const output = await timePhase(
            attempt === 0 ? 'coldTranscriptMs' : 'warmTranscriptMs',
            () => transcribe(audioBytes),
          );
          const outputText = String(output || '').trim();
          const normalized = normalizeTranscript(outputText);
          assert(normalized.length > 0, \`\${memoryMode} \${runtimeMode} returned an empty transcript\`);
          assert(
            normalized === expected,
            \`\${memoryMode} \${runtimeMode} transcript mismatch: \${outputText}\`,
          );
          transcripts.push(outputText.slice(0, 512));

          if (attempt === 0) {
            const cancellationStartedAt = performance.now();
            const controller = new AbortController();
            let cancelledOutput = '';
            let cancellationState = 'resolved';
            let cancellationRequested = false;
            let cancellationWatchdog;
            let tokenWatchdog;
            let rejectWatchdog;
            const watchdogPromise = new Promise((_, reject) => {
              rejectWatchdog = reject;
              tokenWatchdog = setTimeout(
                () => reject(new Error('speech did not emit a cancellable token within 180 seconds')),
                180000,
              );
            });
            try {
              cancelledOutput = String(await Promise.race([
                transcribe(audioBytes, controller.signal, () => {
                  if (cancellationRequested) return;
                  cancellationRequested = true;
                  clearTimeout(tokenWatchdog);
                  cancellationWatchdog = setTimeout(
                    () => rejectWatchdog(new Error('speech cancellation did not settle within 30 seconds')),
                    30000,
                  );
                  controller.abort();
                  bridge.cancel();
                }),
                watchdogPromise,
              ]) || '');
            } catch (error) {
              const message = String(error?.message || error || '');
              assert(
                /abort|cancel|interrupt/i.test(message),
                \`\${memoryMode} \${runtimeMode} cancellation failed: \${message}\`,
              );
              cancellationState = 'rejected';
            } finally {
              clearTimeout(tokenWatchdog);
              clearTimeout(cancellationWatchdog);
              phaseTimingsMs.cancellationMs = Math.round(
                performance.now() - cancellationStartedAt,
              );
            }
            assert(cancellationRequested, \`\${memoryMode} \${runtimeMode} did not emit a token to cancel\`);
            assert(controller.signal.aborted, \`\${memoryMode} \${runtimeMode} cancellation was not requested\`);
            assert(
              normalizeTranscript(cancelledOutput) !== expected,
              \`\${memoryMode} \${runtimeMode} ignored cancellation and returned a full transcript\`,
            );
            transcripts.push(\`cancel:\${cancellationState}:\${cancelledOutput.length}\`);
          }
        }

        const silenceOutput = String(await timePhase(
          'silenceMs',
          () => transcribe(silenceBytes),
        ) || '').trim();
        assert(
          normalizeTranscript(silenceOutput).length === 0,
          \`\${memoryMode} \${runtimeMode} hallucinated speech from silence: \${silenceOutput}\`,
        );

        modeResults.push({
          memoryMode,
          runtimeMode,
          elapsedMs: Math.round(performance.now() - startedAt),
          phaseTimingsMs,
          coldTranscript: transcripts[0],
          cancellation: transcripts[1],
          warmTranscript: transcripts[2],
          silenceTranscript: silenceOutput,
        });
      } finally {
        await bridge.dispose();
      }
    };

    for (const memoryMode of memoryModes) {
      await verifyMode(memoryMode, true, 'direct');
      await verifyMode(memoryMode, false, 'worker');
    }

    finish({
      ok: true,
      fixture: {
        sha256: ${pyJson(audioSha256)},
        encodedByteLength: audioBytes.byteLength,
      },
      modeResults,
    });
  } catch (error) {
    finish({
      ok: false,
      error: String(error && error.stack ? error.stack : error),
    });
  }
})();
</script>
`;
}

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'speech_to_text.mjs',
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
        default: () => env.int('LLAMA_WEBGPU_SPEECH_TIMEOUT_MS', '900000'),
        help: 'Browser operation timeout in milliseconds.',
      },
      {
        flag: '--model-url',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SPEECH_MODEL_URL'),
        help: 'Qwen3-ASR GGUF URL.',
      },
      {
        flag: '--model-path',
        type: 'path',
        default: () => env.optionalPath('LLAMA_WEBGPU_SPEECH_MODEL_PATH'),
        help: 'Local Qwen3-ASR GGUF path.',
      },
      {
        flag: '--model-sha256',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SPEECH_MODEL_SHA256'),
        help: 'Expected model SHA-256.',
      },
      {
        flag: '--mmproj-url',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SPEECH_MMPROJ_URL'),
        help: 'Qwen3-ASR projector GGUF URL.',
      },
      {
        flag: '--mmproj-path',
        type: 'path',
        default: () => env.optionalPath('LLAMA_WEBGPU_SPEECH_MMPROJ_PATH'),
        help: 'Local Qwen3-ASR projector GGUF path.',
      },
      {
        flag: '--mmproj-sha256',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SPEECH_MMPROJ_SHA256'),
        help: 'Expected projector SHA-256.',
      },
      {
        flag: '--audio-url',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SPEECH_AUDIO_URL', DEFAULT_AUDIO_URL),
        help: 'WAV fixture URL used when --audio-path is omitted.',
      },
      {
        flag: '--audio-path',
        type: 'path',
        default: () => env.optionalPath('LLAMA_WEBGPU_SPEECH_AUDIO_PATH'),
        help: 'Local WAV fixture path.',
      },
      {
        flag: '--audio-sha256',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SPEECH_AUDIO_SHA256', DEFAULT_AUDIO_SHA256),
        help: 'Expected WAV SHA-256.',
      },
      {
        flag: '--expect',
        type: 'string',
        default: () => env.string('LLAMA_WEBGPU_SPEECH_EXPECTED_TEXT', DEFAULT_EXPECTED_TEXT),
        help: 'Expected normalized transcript.',
      },
      {
        flag: '--memory-mode',
        type: 'string',
        choices: ['all', ...MEMORY_MODES],
        default: () => env.string('LLAMA_WEBGPU_SPEECH_MEMORY_MODE', 'all'),
        help: 'WASM memory variant to validate (default: both).',
      },
      {
        flag: '--model-cache-dir',
        type: 'path',
        default: () => env.path('LLAMA_WEBGPU_SPEECH_MODEL_CACHE', DEFAULT_MODEL_CACHE),
        help: 'Cache directory used for downloaded model files.',
      },
      {
        flag: '--artifacts-dir',
        type: 'path',
        default: () => env.optionalPath('LLAMA_WEBGPU_SPEECH_ARTIFACTS_DIR'),
        help: 'Directory for JSON, console, and screenshot diagnostics.',
      },
    ],
  });
}

// The check that runs after the payload is printed.
export function checkPayload(payload, memoryModes) {
  const modeResults = pyGet(payload, 'modeResults');
  ensure(
    Array.isArray(modeResults) && modeResults.length === memoryModes.length * 2,
    'speech-to-text mode results are incomplete',
  );
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const distDir = resolvePath(args.distDir);
  ensure(isDirectory(distDir), `dist directory does not exist: ${distDir}`);
  const cacheDir = resolvePath(expandHome(args.modelCacheDir));
  const modelPath = await resolvePinnedFile({
    filePath: args.modelPath,
    url: args.modelUrl,
    expectedSha256: args.modelSha256,
    cacheDir,
    label: 'Qwen3-ASR model',
  });
  const mmprojPath = await resolvePinnedFile({
    filePath: args.mmprojPath,
    url: args.mmprojUrl,
    expectedSha256: args.mmprojSha256,
    cacheDir,
    label: 'Qwen3-ASR projector',
  });
  const audioPath = await resolvePinnedFile({
    filePath: args.audioPath,
    url: args.audioUrl,
    expectedSha256: args.audioSha256,
    cacheDir,
    label: 'speech fixture',
  });
  const actualAudioSha256 = await sha256File(audioPath);
  ensure(pyStrip(args.expect) !== '', 'expected transcript is required');
  const memoryModes = args.memoryMode === 'all' ? MEMORY_MODES : [args.memoryMode];
  const artifactsDir = args.artifactsDir !== null ? resolvePath(args.artifactsDir) : null;

  const payload = await withTempDir('llama-web-bridge-speech-', async (webRoot) => {
    await copyBridgeArtifacts(distDir, webRoot);
    if (memoryModes.includes('wasm64')) await copyMemory64Artifacts(distDir, webRoot);
    await stageFile(modelPath, path.join(webRoot, 'qwen3-asr-model.gguf'));
    await stageFile(mmprojPath, path.join(webRoot, 'qwen3-asr-mmproj.gguf'));
    await stageFile(audioPath, path.join(webRoot, 'speech.wav'));
    await fsp.writeFile(path.join(webRoot, 'silence.wav'), silenceWav());
    await fsp.writeFile(
      path.join(webRoot, 'index.html'),
      renderHarness({ expectedText: args.expect, audioSha256: actualAudioSha256, memoryModes }),
      'utf8',
    );
    if (artifactsDir !== null) {
      await fsp.mkdir(artifactsDir, { recursive: true });
      await fsp.copyFile(path.join(webRoot, 'index.html'), path.join(artifactsDir, 'index.html'));
    }
    return withServer(webRoot, (url) => runPlaywright(url, args.timeoutMs, artifactsDir, 'speech-to-text-smoke'));
  });

  writeStdout(`${pyJson(payload, { indent: 2, sortKeys: true })}\n`);
  if (payload.ok !== true) return 1;
  checkPayload(payload, memoryModes);
  return 0;
}

if (import.meta.main) {
  await runMain('speech-to-text', main);
}
