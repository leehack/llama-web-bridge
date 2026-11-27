import type { ModelStagingErrorCode } from '../llama_webgpu_bridge.d.ts';
import { isWebKitUserAgent } from './environment.ts';
import type { LlamaCoreModule } from './types.ts';

export const REQUIRED_JS_MODEL_STAGING_EXPORTS = [
  '_llamadart_webgpu_js_model_fs_api_version',
  '_llamadart_webgpu_js_model_fs_mount_id',
  '_llamadart_webgpu_mount_js_model_fs',
  '_llamadart_webgpu_unmount_js_model_fs',
  '_llamadart_webgpu_js_model_fs_prealloc_next',
  '_llamadart_webgpu_js_model_fs_open_direct',
  '_llamadart_webgpu_js_model_fs_close_direct',
  '_llamadart_webgpu_js_model_fs_consume_oom_bytes',
  '_llamadart_webgpu_js_model_fs_allocated_bytes',
  '_llamadart_webgpu_js_model_fs_release_mount_chunks',
  'llamadartJsModelFsWriteDirect',
] as const;

const JS_MODEL_STAGING_API_VERSION = 3;

export type JsBackedStagingMode = 'on' | 'off' | 'auto';

// WebKit charges staged file copies, the wasm heap and GPU buffers against
// the WebContent process memory limit.
export function jsBackedStagingMode(
  config: { jsBackedModelStaging?: unknown; userAgent?: unknown } | null | undefined,
): JsBackedStagingMode {
  if (config?.jsBackedModelStaging === true) return 'on';
  if (config?.jsBackedModelStaging === false) return 'off';
  return isWebKitUserAgent(config?.userAgent ?? globalThis.navigator?.userAgent ?? '') ? 'auto' : 'off';
}

export function supportsJsBackedStaging(core: LlamaCoreModule | null | undefined): boolean {
  if (!core || missingJsModelStagingExports(core).length > 0) return false;
  const apiVersion = Number(core.ccall('llamadart_webgpu_js_model_fs_api_version', 'number', [], []));
  return isSupportedJsModelStagingApiVersion(apiVersion);
}

export function isSupportedJsModelStagingApiVersion(apiVersion: number): boolean {
  return Number.isInteger(apiVersion) && apiVersion >= JS_MODEL_STAGING_API_VERSION;
}

const EXTERNAL_MODEL_READER_API_VERSION = 5;

// A core that cannot read a model on demand keeps staging it in JS chunks.
export function supportsExternalModelReader(
  core: LlamaCoreModule | null | undefined,
  apiVersion: number,
): boolean {
  return Number.isInteger(apiVersion)
    && apiVersion >= EXTERNAL_MODEL_READER_API_VERSION
    && typeof core?._llamadart_webgpu_js_model_fs_open_external === 'function'
    && typeof core?.llamadartJsModelFsBindExternal === 'function';
}

// Optional: a core without these keeps staging on JS chunks.
const HEAP_RESIDENT_STAGING_EXPORTS = [
  '_llamadart_webgpu_js_model_fs_supports_heap_resident',
  '_llamadart_webgpu_js_model_fs_open_direct_heap',
  '_llamadart_webgpu_js_model_fs_heap_resident_bytes',
] as const;

const HEAP_RESIDENT_STAGING_API_VERSION = 4;

export function supportsHeapResidentStaging(
  core: LlamaCoreModule | null | undefined,
  apiVersion: number,
): boolean {
  return Number.isInteger(apiVersion)
    && apiVersion >= HEAP_RESIDENT_STAGING_API_VERSION
    && HEAP_RESIDENT_STAGING_EXPORTS.every((name) => typeof core?.[name] === 'function')
    && core?.ccall('llamadart_webgpu_js_model_fs_supports_heap_resident', 'number', [], []) === 1;
}

// The layers llama.cpp offloads for a requested count: it expands a negative
// count to every layer, so the default (-1) is full offload, not zero.
export function effectiveOffloadedGpuLayers(requestedGpuLayers: number): number {
  if (!Number.isFinite(requestedGpuLayers) || requestedGpuLayers < 0) {
    return Infinity;
  }
  return Math.trunc(requestedGpuLayers);
}

export interface HeapResidentStagingInputs {
  jsBackedStaging: boolean;
  // The count the native load receives, after the Safari cap.
  nativeGpuLayers: number;
  useMmapDisabled: boolean;
  useMlock: boolean;
  shardCount: number;
}

// The mapped block stays for the model's lifetime, so it only pays off when at
// most one layer's weights are also copied to the GPU.
export function shouldUseHeapResidentStaging(inputs: HeapResidentStagingInputs): boolean {
  return inputs.jsBackedStaging === true
    && effectiveOffloadedGpuLayers(inputs.nativeGpuLayers) <= 1
    && inputs.useMmapDisabled !== true
    && inputs.useMlock !== true
    && inputs.shardCount === 1;
}

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
    || value === 'MODEL_STAGING_UNSUPPORTED'
    || value === 'MODEL_READ_FAILED'
    || value === 'MODEL_SOURCE_INVALID'
    || value === 'MODEL_SOURCE_UNSUPPORTED'
    || value === 'MODEL_SOURCE_SPLIT_UNSUPPORTED'
    || value === 'MODEL_SOURCE_CHANGED'
    || value === 'MODEL_SOURCE_BUSY';
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
