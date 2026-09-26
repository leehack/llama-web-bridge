#!/usr/bin/env node
// Real-model browser smoke for Qwen3-TTS audio generation.
//
// Ported from text_to_speech_browser_smoke.py with the same flags, harness
// page and output. release_qualification.py runs it as the text-to-speech gate
// and checks each WAV it writes against the peak and RMS the page reported, so
// the WAV bytes are exactly the ones the page encoded.

import fsp from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import {
  copyBridgeArtifacts,
  copyMemory64Artifacts,
  ensure,
  isDict,
  isDirectory,
  isFile,
  parseSmokeArgs,
  pyB64Decode,
  pyGetDefault,
  pyIter,
  pyJson,
  pyLen,
  pyPop,
  pyStr,
  pyTruthy,
  resolvePath,
  runMain,
  runPollingPlaywright,
  sha256File,
  stageFile,
  withServer,
  withTempDir,
  writeStdout,
} from './support.mjs';

const DESCRIPTION = 'Real-model browser smoke for Qwen3-TTS audio generation.';
export const MEMORY_MODES = Object.freeze(['wasm32', 'wasm64']);
export const RUNTIME_MODES = Object.freeze(['direct', 'worker']);

export function renderHarness({
  prompt, modelSha256, mmprojSha256, speakerAudioSha256, memoryModes, runtimeModes, maxFrames, gpuLayers,
  testCancellation,
}) {
  return `
<!doctype html>
<meta charset="utf-8">
<title>llama-web-bridge Qwen3-TTS smoke</title>
<pre id="result">pending</pre>
<script type="module">
(async () => {
  const resultNode = document.getElementById('result');
  const finish = (payload) => {
    resultNode.textContent = JSON.stringify(payload);
    window.__smokeResult = payload;
  };
  const setStage = (stage) => {
    window.__smokeStage = stage;
    console.log(\`tts-smoke-stage:\${stage}\`);
  };
    const assert = (condition, message) => {
      if (!condition) throw new Error(message);
    };
    const pcmToWavBase64 = (pcm, sampleRate) => {
      const bytes = new Uint8Array(44 + pcm.length * 2);
      const view = new DataView(bytes.buffer);
      const writeText = (offset, value) => {
        for (let index = 0; index < value.length; index += 1) {
          view.setUint8(offset + index, value.charCodeAt(index));
        }
      };
      writeText(0, 'RIFF');
      view.setUint32(4, bytes.length - 8, true);
      writeText(8, 'WAVE');
      writeText(12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeText(36, 'data');
      view.setUint32(40, pcm.length * 2, true);
      for (let index = 0; index < pcm.length; index += 1) {
        const sample = Math.max(-1, Math.min(1, pcm[index]));
        view.setInt16(
          44 + index * 2,
          sample < 0 ? Math.round(sample * 32768) : Math.round(sample * 32767),
          true,
        );
      }
      let binary = '';
      for (let offset = 0; offset < bytes.length; offset += 0x8000) {
        binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
      }
      return btoa(binary);
    };
  try {
    assert(window.crossOriginIsolated, 'test page is not cross-origin isolated');
    const module = await import('/llama_webgpu_bridge.js');
    const LlamaWebGpuBridge = module.LlamaWebGpuBridge || window.LlamaWebGpuBridge;
    assert(typeof LlamaWebGpuBridge === 'function', 'bridge export was not registered');
    const speakerAudio = ${pyJson(speakerAudioSha256 !== null)}
      ? new Uint8Array(await (await fetch('/speaker-reference.wav')).arrayBuffer())
      : null;
    if (speakerAudio) {
      assert(speakerAudio.byteLength > 44, 'speaker-reference WAV is empty');
    }

    const modeResults = [];
    const memoryModes = ${pyJson(memoryModes)};
    const runtimeModes = ${pyJson(runtimeModes)};
    for (const memoryMode of memoryModes) {
      for (const runtimeMode of runtimeModes) {
        const useMemory64 = memoryMode === 'wasm64';
        const bridge = new LlamaWebGpuBridge({
          disableWorker: runtimeMode === 'direct',
          logLevel: 2,
          preferMemory64: useMemory64,
          coreModuleUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.js' : undefined,
          wasmUrlMem64: useMemory64 ? '/llama_webgpu_core_mem64.wasm' : undefined,
          workerTextToSpeechTimeoutMs: 1200000,
        });
        const startedAt = performance.now();
        try {
          setStage(\`\${memoryMode}:\${runtimeMode}:load-model\`);
          const modelStartedAt = performance.now();
          await bridge.loadModelFromUrl('/qwen3-tts-model.gguf', {
            nCtx: 4096,
            nGpuLayers: ${gpuLayers},
            nThreads: 4,
            nBatch: 512,
            nUbatch: 256,
            useCache: false,
            forceRemoteFetchBackend: false,
            progressCallback: (event) => {
              const loaded = Number(event.loaded) || 0;
              const total = Number(event.total) || 0;
              window.__smokeStage = \`\${memoryMode}:\${runtimeMode}:load-model:\${loaded}/\${total}\`;
            },
          });
          const modelLoadMs = Math.round(performance.now() - modelStartedAt);
          setStage(\`\${memoryMode}:\${runtimeMode}:load-projector\`);
          const projectorStartedAt = performance.now();
          await bridge.loadMultimodalProjector('/qwen3-tts-mmproj.gguf');
          const projectorLoadMs = Math.round(performance.now() - projectorStartedAt);
          setStage(\`\${memoryMode}:\${runtimeMode}:capabilities\`);
          const capabilities = await bridge.getTextToSpeechCapabilities();
          assert(capabilities.apiVersion === 1, 'unexpected TTS API version');
          assert(capabilities.supported === true, \`TTS unsupported: \${capabilities.reason}\`);
          assert(capabilities.sampleRate === 24000, 'unexpected sample rate');
          assert(capabilities.channels === 1, 'unexpected channel count');
          if (speakerAudio) {
            assert(
              capabilities.supportsSpeakerReference === true,
              'loaded projector does not report speaker-reference support',
            );
          }

          const preAbortedController = new AbortController();
          preAbortedController.abort();
          try {
            await bridge.synthesizeSpeech({
              text: 'This pre-aborted task must not start.',
              signal: preAbortedController.signal,
            });
            throw new Error('pre-aborted synthesis unexpectedly completed');
          } catch (error) {
            assert(error?.name === 'AbortError', \`unexpected pre-abort error: \${error}\`);
          }

          const progress = [];
          setStage(\`\${memoryMode}:\${runtimeMode}:synthesize\`);
          const synthesisStartedAt = performance.now();
          const output = await bridge.synthesizeSpeech({
            text: ${pyJson(prompt)},
            language: 'en',
            speakerAudio: speakerAudio || undefined,
            promptBatchSize: 512,
            maxFrames: ${maxFrames},
            topK: 40,
            topP: 0.95,
            minP: 0,
            temperature: 0.8,
            seed: 1,
            onProgress: (event) => progress.push({
              state: event.state,
              promptTokensRemaining: event.promptTokensRemaining,
              framesGenerated: event.framesGenerated,
            }),
          });
          const synthesisMs = Math.round(performance.now() - synthesisStartedAt);
          setStage(\`\${memoryMode}:\${runtimeMode}:validate-output\`);
          assert(output.pcm instanceof Float32Array, 'PCM output is not Float32Array');
          assert(output.sampleRate === 24000, 'result sample rate mismatch');
          assert(output.channels === 1, 'result channel mismatch');
          assert(output.sampleCount === output.pcm.length, 'sample count mismatch');
          assert(output.sampleCount >= 2400, 'synthesized audio is too short');
          let peak = 0;
          let energy = 0;
          for (const sample of output.pcm) {
            assert(Number.isFinite(sample), 'PCM contains a non-finite sample');
            peak = Math.max(peak, Math.abs(sample));
            energy += sample * sample;
          }
          const rms = Math.sqrt(energy / output.pcm.length);
          assert(peak > 0.001, \`synthesized audio peak is too low: \${peak}\`);
          assert(rms > 0.0001, \`synthesized audio RMS is too low: \${rms}\`);
          assert(progress.some((event) => event.state === 2), 'generation progress was not observed');
          assert(progress.some((event) => event.state === 3), 'completion progress was not observed');

          let cancellationTested = false;
          let reuseSampleCount = 0;
          if (${pyJson(testCancellation)}) {
            setStage(\`\${memoryMode}:\${runtimeMode}:cancel\`);
            const controller = new AbortController();
            let abortRequested = false;
            try {
              await bridge.synthesizeSpeech({
                text: 'Cancel this browser speech task.',
                language: 'en',
                maxFrames: 24,
                seed: 2,
                signal: controller.signal,
                onProgress: (event) => {
                  if (!abortRequested && event.state === 2 && event.framesGenerated >= 1) {
                    abortRequested = true;
                    controller.abort();
                  }
                },
              });
              throw new Error('cancelled synthesis unexpectedly completed');
            } catch (error) {
              assert(error?.name === 'AbortError', \`unexpected cancellation error: \${error}\`);
            }
            assert(abortRequested, 'cancellation was not requested during audio generation');

            setStage(\`\${memoryMode}:\${runtimeMode}:reuse\`);
            const reuse = await bridge.synthesizeSpeech({
              text: 'Ready.',
              language: 'en',
              maxFrames: 1,
              seed: 3,
            });
            assert(reuse.pcm instanceof Float32Array, 'reuse PCM output is invalid');
            assert(reuse.sampleCount > 0, 'runtime reuse after cancellation returned no audio');
            cancellationTested = true;
            reuseSampleCount = reuse.sampleCount;
          }
          setStage(\`\${memoryMode}:\${runtimeMode}:unload-projector\`);
          await bridge.unloadMultimodalProjector();
          const unloadedCapabilities = await bridge.getTextToSpeechCapabilities();
          assert(
            unloadedCapabilities.supported === false,
            'TTS capability remained enabled after projector unload',
          );
          modeResults.push({
            memoryMode,
            runtimeMode,
            requestedGpuLayers: ${gpuLayers},
            gpuActive: bridge.isGpuActive(),
            backendName: bridge.getBackendName(),
            totalElapsedMs: Math.round(performance.now() - startedAt),
            modelLoadMs,
            projectorLoadMs,
            synthesisMs,
            sampleRate: output.sampleRate,
            sampleCount: output.sampleCount,
            durationSeconds: output.sampleCount / output.sampleRate,
            framesGenerated: output.framesGenerated,
            truncated: output.truncated,
            peak,
            rms,
            cancellationTested,
            preAbortedTested: true,
            reuseSampleCount,
            speakerReferenceTested: speakerAudio !== null,
            unloadTested: true,
            _wavBase64: pcmToWavBase64(output.pcm, output.sampleRate),
          });
        } finally {
          await bridge.dispose();
        }
      }
    }

    finish({
      ok: true,
      modelSha256: ${pyJson(modelSha256)},
      mmprojSha256: ${pyJson(mmprojSha256)},
      speakerAudioSha256: ${pyJson(speakerAudioSha256)},
      modeResults,
    });
  } catch (error) {
    finish({ ok: false, error: String(error?.stack || error) });
  }
})();
</script>
`;
}

export function parseArgs(argv) {
  return parseSmokeArgs(argv, {
    prog: 'text_to_speech.mjs',
    description: DESCRIPTION,
    options: [
      { flag: '--dist-dir', type: 'path', default: () => 'dist' },
      { flag: '--model-path', type: 'path', required: true },
      { flag: '--mmproj-path', type: 'path', required: true },
      { flag: '--model-sha256', type: 'string', default: () => '' },
      { flag: '--mmproj-sha256', type: 'string', default: () => '' },
      { flag: '--speaker-audio-path', type: 'path' },
      { flag: '--speaker-audio-sha256', type: 'string', default: () => '' },
      { flag: '--prompt', type: 'string', default: () => 'Hello from llamadart.' },
      { flag: '--max-frames', type: 'int', default: () => 96 },
      { flag: '--gpu-layers', type: 'int', default: () => 0 },
      { flag: '--memory-mode', type: 'string', choices: ['all', ...MEMORY_MODES], default: () => 'all' },
      { flag: '--runtime-mode', type: 'string', choices: ['all', ...RUNTIME_MODES], default: () => 'all' },
      { flag: '--timeout-ms', type: 'int', default: () => 1_200_000 },
      { flag: '--skip-cancellation', type: 'flag', default: () => false },
      { flag: '--artifacts-dir', type: 'path' },
    ],
  });
}

// Move each mode's page-encoded WAV into `text-to-speech-<memory>-<runtime>.wav`
// in the artifacts directory and name it in `audioArtifact`; without an
// artifacts directory the WAV is dropped.
export async function extractWavArtifacts(payload, artifactsDir) {
  for (const modeResult of pyIter(pyGetDefault(payload, 'modeResults', []))) {
    if (!isDict(modeResult)) continue;
    const encodedWav = pyPop(modeResult, '_wavBase64', '');
    if (!pyTruthy(encodedWav) || artifactsDir === null) continue;
    const memoryMode = pyStr(pyGetDefault(modeResult, 'memoryMode', 'unknown'));
    const runtimeMode = pyStr(pyGetDefault(modeResult, 'runtimeMode', 'unknown'));
    const audioName = `text-to-speech-${memoryMode}-${runtimeMode}.wav`;
    await fsp.writeFile(path.join(artifactsDir, audioName), pyB64Decode(encodedWav));
    modeResult.audioArtifact = audioName;
  }
}

// The check that runs after the payload is printed.
export function checkPayload(payload, memoryModes, runtimeModes) {
  ensure(
    pyLen(pyGetDefault(payload, 'modeResults', [])) === memoryModes.length * runtimeModes.length,
    'mode results are incomplete',
  );
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const distDir = resolvePath(args.distDir);
  const modelPath = resolvePath(args.modelPath);
  const mmprojPath = resolvePath(args.mmprojPath);
  const speakerAudioPath = args.speakerAudioPath !== null ? resolvePath(args.speakerAudioPath) : null;
  ensure(isDirectory(distDir), `dist directory does not exist: ${distDir}`);
  ensure(isFile(modelPath), `model does not exist: ${modelPath}`);
  ensure(isFile(mmprojPath), `projector does not exist: ${mmprojPath}`);
  if (speakerAudioPath !== null) {
    ensure(isFile(speakerAudioPath), `speaker-reference audio does not exist: ${speakerAudioPath}`);
  }
  ensure(!args.speakerAudioSha256 || speakerAudioPath !== null, 'speaker audio checksum requires --speaker-audio-path');
  ensure(args.maxFrames > 0, 'max frames must be positive');
  const modelSha256 = await sha256File(modelPath);
  const mmprojSha256 = await sha256File(mmprojPath);
  const speakerAudioSha256 = speakerAudioPath !== null ? await sha256File(speakerAudioPath) : null;
  if (args.modelSha256) ensure(modelSha256 === args.modelSha256.toLowerCase(), 'model checksum mismatch');
  if (args.mmprojSha256) ensure(mmprojSha256 === args.mmprojSha256.toLowerCase(), 'projector checksum mismatch');
  if (args.speakerAudioSha256) {
    ensure(
      speakerAudioSha256 === args.speakerAudioSha256.toLowerCase(),
      'speaker-reference audio checksum mismatch',
    );
  }
  const memoryModes = args.memoryMode === 'all' ? MEMORY_MODES : [args.memoryMode];
  const runtimeModes = args.runtimeMode === 'all' ? RUNTIME_MODES : [args.runtimeMode];

  const payload = await withTempDir('llama-web-bridge-tts-', async (webRoot) => {
    await copyBridgeArtifacts(distDir, webRoot);
    if (memoryModes.includes('wasm64')) await copyMemory64Artifacts(distDir, webRoot);
    await stageFile(modelPath, path.join(webRoot, 'qwen3-tts-model.gguf'));
    await stageFile(mmprojPath, path.join(webRoot, 'qwen3-tts-mmproj.gguf'));
    if (speakerAudioPath !== null) await stageFile(speakerAudioPath, path.join(webRoot, 'speaker-reference.wav'));
    await fsp.writeFile(path.join(webRoot, 'index.html'), renderHarness({
      prompt: args.prompt,
      modelSha256,
      mmprojSha256,
      speakerAudioSha256,
      memoryModes,
      runtimeModes,
      maxFrames: args.maxFrames,
      gpuLayers: args.gpuLayers,
      testCancellation: !args.skipCancellation,
    }), 'utf8');
    const artifactsDir = args.artifactsDir !== null ? resolvePath(args.artifactsDir) : null;
    if (artifactsDir !== null) {
      await fsp.mkdir(artifactsDir, { recursive: true });
      await fsp.copyFile(path.join(webRoot, 'index.html'), path.join(artifactsDir, 'index.html'));
    }
    return withServer(webRoot, (url) => runPollingPlaywright(url, args.timeoutMs, artifactsDir, {
      artifactPrefix: 'text-to-speech-smoke',
      stageMarker: 'tts-smoke-stage:',
      stagePrefix: 'tts smoke',
      transform: (result) => extractWavArtifacts(result, artifactsDir),
    }));
  });

  writeStdout(`${pyJson(payload, { indent: 2, sortKeys: true })}\n`);
  if (payload.ok !== true) return 1;
  checkPayload(payload, memoryModes, runtimeModes);
  return 0;
}

if (import.meta.main) {
  await runMain('text-to-speech', main);
}
