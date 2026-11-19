import type { ModelStagingErrorCode } from '../llama_webgpu_bridge.d.ts';
import type { LlamaCoreModule } from './types.ts';

export const REQUIRED_JS_MODEL_STAGING_EXPORTS = [
  '_llamadart_webgpu_mount_js_model_fs',
  '_llamadart_webgpu_unmount_js_model_fs',
  '_llamadart_webgpu_js_model_fs_prealloc_next',
  '_llamadart_webgpu_js_model_fs_open_direct',
  '_llamadart_webgpu_js_model_fs_close_direct',
  '_llamadart_webgpu_js_model_fs_consume_oom_bytes',
  '_llamadart_webgpu_js_model_fs_allocated_bytes',
  'llamadartJsModelFsWriteDirect',
] as const;

// Typed, worker-transportable failures raised when JS-backed model staging
// fails, is unsupported by the core, collides on mount, or hits the wasm32 4 GiB limit.
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
    || value === 'MODEL_TENSOR_SIZE_LIMIT_EXCEEDED'
    || value === 'MODEL_STAGING_MOUNT_FAILED'
    || value === 'MODEL_STAGING_UNSUPPORTED';
}

export function missingJsModelStagingExports(core: LlamaCoreModule | null | undefined): string[] {
  const missing: string[] = [];
  for (const name of REQUIRED_JS_MODEL_STAGING_EXPORTS) {
    if (typeof core?.[name] !== 'function') {
      missing.push(name);
    }
  }
  return missing;
}

export function isGenericCoreAbortText(value: unknown): boolean {
  const lowered = String(value ?? '').trim().toLowerCase();
  return lowered === 'aborted'
    || lowered === 'native code called abort()'
    || lowered === 'core abort'
    || lowered.startsWith('core abort:')
    || /^aborted\([^)]*\)(?:\.\s*build with -sassertions.*)?$/i.test(lowered);
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
