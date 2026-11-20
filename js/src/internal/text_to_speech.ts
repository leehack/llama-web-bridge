import type { TextToSpeechCapabilities, TextToSpeechProgress } from '../llama_webgpu_bridge.d.ts';

const STREAMING_PCM = 1 << 3;

export const TEXT_TO_SPEECH_CANCELLED = 'Text-to-speech synthesis was cancelled.';

export class TextToSpeechStreamingUnsupportedError extends Error {
  readonly code = 'TTS_STREAMING_UNSUPPORTED';

  constructor() {
    super('The worker did not declare supportsIncrementalAudio. The bridge and worker scripts must come from the same release.');
    this.name = 'TextToSpeechStreamingUnsupportedError';
  }
}

export function withTextToSpeechStreaming(
  capabilities: Omit<TextToSpeechCapabilities, 'supportsIncrementalAudio'>,
): TextToSpeechCapabilities {
  return {
    ...capabilities,
    supportsIncrementalAudio: capabilities.supported === true
      && (capabilities.capabilities & STREAMING_PCM) !== 0,
  };
}

export function textToSpeechProgressTransfers({ pcm }: TextToSpeechProgress): ArrayBuffer[] {
  return pcm?.buffer instanceof ArrayBuffer ? [pcm.buffer] : [];
}
