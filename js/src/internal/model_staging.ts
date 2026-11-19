import type { ModelStagingErrorCode } from '../llama_webgpu_bridge.d.ts';

// Typed, worker-transportable failures raised when JS-backed model staging
// runs out of JS heap memory or when a GGUF exceeds the wasm32 4 GiB limit.
export class ModelStagingError extends Error {
  declare code: ModelStagingErrorCode;

  constructor(code: ModelStagingErrorCode, message: string) {
    super(message);
    this.name = 'ModelStagingError';
    this.code = code;
  }
}

export function isModelStagingErrorCode(value: unknown): value is ModelStagingErrorCode {
  return value === 'MODEL_STAGING_ALLOCATION_FAILED'
    || value === 'MODEL_TENSOR_SIZE_LIMIT_EXCEEDED';
}

export function isJsAllocationError(error: unknown): boolean {
  if (!error || typeof error !== 'object') {
    const text = String(error || '').toLowerCase();
    return text.includes('array buffer allocation failed')
      || text.includes('out of memory')
      || text.includes('enomem');
  }

  const candidate = error as { name?: unknown; message?: unknown; errno?: unknown; code?: unknown };
  if (candidate.code === 'ENOMEM' || candidate.errno === 12 || candidate.errno === 48) {
    return true;
  }

  const text = `${String(candidate.name || '')} ${String(candidate.message || '')}`.toLowerCase();
  return text.includes('array buffer allocation failed')
    || text.includes('cannot allocate')
    || text.includes('allocation failed')
    || text.includes('out of memory')
    || text.includes('enomem')
    || (candidate.name === 'RangeError' && text.includes('buffer'));
}

export function isWasm32TensorLimitText(text: string): boolean {
  const normalized = text.toLowerCase();
  return normalized.includes('wasm32 4 gib')
    || (normalized.includes('gguf_init_from_reader') && normalized.includes('size overflow'));
}
