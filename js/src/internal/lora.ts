// LoRA adapter API constants and argument checks.

import { isInt32 } from './typed_values.ts';

// LoRA adapter API version shared with llama_webgpu_lora.h.
export const LORA_API_VERSION = 1;

export const LORA_LOAD_ABORT_MESSAGE = 'LoRA adapter load was cancelled.';

export function loraHandleFrom(handle: unknown): number {
  if (!isInt32(handle) || (handle as number) <= 0) {
    throw new TypeError(`LoRA adapter handle must be a positive integer, got ${String(handle)}.`);
  }
  return handle as number;
}

export function loraScaleFrom(scale: unknown): number {
  if (typeof scale !== 'number' || !Number.isFinite(scale)) {
    throw new TypeError(`LoRA adapter scale must be a finite number, got ${String(scale)}.`);
  }
  if (!Number.isFinite(Math.fround(scale))) {
    throw new RangeError(`LoRA adapter scale ${String(scale)} is outside the 32-bit float range.`);
  }
  return scale;
}

export function staleLoraAdapterError(handle: number): Error {
  return new Error(
    `LoRA adapter ${handle} is not loaded; its model was unloaded or replaced. `
    + 'Load the adapter again.',
  );
}
