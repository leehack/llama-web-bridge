// Next-token scoring request and result helpers.

import type { ScoredToken } from '../llama_webgpu_bridge.d.ts';
import { isInt32 } from './typed_values.ts';

/**
 * Copies next-token candidate ids. Only the integer type is checked here; the
 * core rejects ids outside the loaded vocabulary.
 */
export function nextTokenCandidateIds(value: unknown): number[] {
  if (value == null) {
    return [];
  }
  const isList = Array.isArray(value)
    || (ArrayBuffer.isView(value) && !(value instanceof DataView));
  if (!isList) {
    throw new TypeError('Next-token candidates must be an array or typed array of token ids.');
  }
  return Array.from(value as ArrayLike<unknown>, (item, index) => {
    if (!isInt32(item)) {
      throw new TypeError(`Next-token candidates[${index}] is ${String(item)}; expected a 32-bit integer.`);
    }
    return item as number;
  });
}

export function scoredTokensFrom(entries: unknown): ScoredToken[] {
  if (!Array.isArray(entries)) {
    return [];
  }
  return entries.map((entry) => ({
    token: Number(entry?.token),
    bytes: Uint8Array.from(Array.isArray(entry?.bytes) ? entry.bytes : []),
    logprob: typeof entry?.logprob === 'number' ? entry.logprob : -Infinity,
  }));
}
