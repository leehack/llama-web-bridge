import type { EmbedInput, EmbedPart, EmbeddingCapabilities, EmbeddingErrorCode } from '../llama_webgpu_bridge.d.ts';

// Public, payload-free error codes survive the worker protocol.
export class EmbeddingError extends Error {
  declare code: EmbeddingErrorCode;
  constructor(code: EmbeddingErrorCode, message: string) {
    super(message);
    this.name = 'EmbeddingError';
    this.code = code;
  }
}

export const EMPTY_EMBEDDING_CAPABILITIES: EmbeddingCapabilities = {
  apiVersion: 1, text: false, image: false, audio: false,
};
export const EMBEDDING_MEDIA_MARKER = '<__media__>';
// One CPU/WASM media embedding can run for minutes, far past the generic
// worker request budget.
export const EMBEDDING_MEDIA_WORKER_TIMEOUT_MS = 10 * 60 * 1000;
// The worker host's embedBatch hook. Each finished input posts a progress event
// that re-arms the proxy's request deadline, so the deadline bounds one input
// rather than the whole batch. Functions cannot cross postMessage, so a
// worker-mode caller's options never carry it.
export const EMBED_BATCH_PROGRESS_OPTION = '__llamadartEmbedBatchProgress';
export type EmbedBatchProgressHook = (progress: { completed: number; total: number }) => void;

export function hasEmbeddingMediaParts(input: unknown): boolean {
  return Array.isArray(input) && input.some((part) => part?.type !== 'text');
}

// Versioned CDN paths identify the loaded assets. Never include URL credentials,
// query strings, fragments or arbitrary path text in public diagnostics.
export function embeddingAssetVersion(...sources: (string | undefined)[]): string {
  for (const source of [...sources, import.meta.url]) {
    try {
      const version = new URL(source || '', import.meta.url).pathname.match(/(?:^|[/@])(v?\d+\.\d+\.\d+(?:-\d+)?)(?:[/]|$)/)?.[1];
      if (version) return version;
    } catch (_) { /* Local or unversioned assets. */ }
  }
  return 'unversioned/local';
}

export function missingEmbeddingCapability(version = embeddingAssetVersion()): EmbeddingError {
  return new EmbeddingError('EMBEDDING_CAPABILITY_UNAVAILABLE',
    `Loaded bridge/assets version ${version} has no known media-embedding capability API version 1. Upgrade the loaded bridge assets.`);
}

export function embeddingCapabilitiesFrom(value: unknown, version = embeddingAssetVersion()): EmbeddingCapabilities {
  let parsed: unknown = value;
  if (typeof value === 'string') {
    try { parsed = JSON.parse(value); } catch (_) { throw missingEmbeddingCapability(version); }
  }
  const caps = parsed as Partial<EmbeddingCapabilities> | null;
  if (!caps || caps.apiVersion !== 1
    || typeof caps.text !== 'boolean' || typeof caps.image !== 'boolean' || typeof caps.audio !== 'boolean') {
    throw missingEmbeddingCapability(version);
  }
  return { apiVersion: 1, text: caps.text, image: caps.image, audio: caps.audio };
}

// Snapshot the ordered list/part descriptors; caller media buffers remain owned
// by the caller. Worker postMessage uses structured clone with no transfer list.
export function snapshotEmbedInput(input: EmbedInput): EmbedInput {
  if (!Array.isArray(input)) return String(input);
  if (!Array.isArray(input) || input.length === 0) {
    throw new EmbeddingError('INVALID_EMBEDDING_INPUT', 'Embedding parts must be a nonempty array.');
  }
  const parts = Array.from(input, (part): EmbedPart => {
    if (!part || typeof part !== 'object') {
      throw new EmbeddingError('INVALID_EMBEDDING_INPUT', 'Invalid embedding part.');
    }
    if (part.type === 'text') {
      if (typeof part.text !== 'string') {
        throw new EmbeddingError('INVALID_EMBEDDING_INPUT', 'Embedding text part must contain a string.');
      }
      return { type: 'text', text: part.text };
    }
    if (part.type !== 'image' && part.type !== 'audio') {
      throw new EmbeddingError('EMBEDDING_MODALITY_UNSUPPORTED', 'Embedding part modality is unsupported.');
    }
    if ('path' in part) {
      throw new EmbeddingError('EMBEDDING_MODALITY_UNSUPPORTED', 'Browser embeddings require media bytes, URL or PCM; filesystem paths are unsupported.');
    }
    const sourceCount = Number(part.bytes != null) + Number(part.url != null)
      + Number('samples' in part && part.samples != null);
    if (sourceCount !== 1
      || (part.bytes != null && (!(part.bytes instanceof Uint8Array) || part.bytes.length === 0))
      || (part.url != null && (typeof part.url !== 'string' || part.url.length === 0))) {
      throw new EmbeddingError('INVALID_EMBEDDING_INPUT', 'Embedding media requires exactly one nonempty supported source.');
    }
    if ('samples' in part && part.samples != null) {
      if (part.type !== 'audio' || !(part.samples instanceof Float32Array)
        || part.samples.length === 0 || !part.samples.every(Number.isFinite)) {
        throw new EmbeddingError('INVALID_EMBEDDING_INPUT', 'Embedding PCM must be finite mono 16 kHz Float32 samples.');
      }
    }
    return { ...part };
  });
  const merged: EmbedPart[] = [];
  for (const part of parts) {
    const previous = merged.at(-1);
    if (part.type === 'text' && previous?.type === 'text') previous.text += part.text;
    else merged.push(part);
  }
  // Check after coalescing: a marker can span adjacent text parts.
  if (merged.some((part) => part.type === 'text' && part.text.includes(EMBEDDING_MEDIA_MARKER))) {
    throw new EmbeddingError('INVALID_EMBEDDING_INPUT', 'Reserved media marker in embedding text.');
  }
  return merged;
}

export function requireEmbeddingModalities(input: EmbedInput, caps: EmbeddingCapabilities, hasProjector: boolean) {
  if (typeof input === 'string') return;
  if (!caps.text) {
    throw new EmbeddingError('MODEL_NOT_EMBEDDING', 'Loaded model does not support pooled embeddings.');
  }
  for (const part of input) {
    if (part.type === 'text') continue;
    if (!hasProjector) {
      throw new EmbeddingError('EMBEDDING_PROJECTOR_MISSING', 'Media embeddings require a loaded projector.');
    }
    if (!caps[part.type]) {
      throw new EmbeddingError('EMBEDDING_MODALITY_UNSUPPORTED', 'Loaded projector/model does not support the requested embedding modality.');
    }
  }
}

export function embeddingCoreError(rc: number, version = embeddingAssetVersion()): EmbeddingError {
  if (rc === -20) return missingEmbeddingCapability(version);
  const errors: Record<number, [EmbeddingErrorCode, string]> = {
    [-21]: ['MODEL_NOT_EMBEDDING', 'Loaded model does not support this pooled embedding operation.'],
    [-22]: ['EMBEDDING_PROJECTOR_MISSING', 'Media embeddings require a loaded projector.'],
    [-23]: ['EMBEDDING_INPUT_TOO_LARGE', 'Expanded embedding input exceeds the allocated context, batch, sequence or single microbatch capacity.'],
    [-24]: ['EMBEDDING_MODALITY_UNSUPPORTED', 'Loaded projector does not support the requested embedding modality or PCM rate.'],
  };
  const [code, message] = errors[rc] || ['EMBEDDING_FAILED', 'Embedding execution failed.'];
  return new EmbeddingError(code, message);
}
