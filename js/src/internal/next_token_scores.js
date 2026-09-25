// Next-token scoring request and result helpers.

import { isInt32 } from './typed_values.js';

/**
 * Copies next-token candidate ids. Only the integer type is checked here; the
 * core rejects ids outside the loaded vocabulary.
 *
 * @param {unknown} value
 * @returns {number[]}
 */
export function nextTokenCandidateIds(value) {
  if (value == null) {
    return [];
  }
  const isList = Array.isArray(value)
    || (ArrayBuffer.isView(value) && !(value instanceof DataView));
  if (!isList) {
    throw new TypeError('Next-token candidates must be an array or typed array of token ids.');
  }
  return Array.from(/** @type {ArrayLike<unknown>} */ (value), (item, index) => {
    if (!isInt32(item)) {
      throw new TypeError(`Next-token candidates[${index}] is ${String(item)}; expected a 32-bit integer.`);
    }
    return /** @type {number} */ (item);
  });
}

/**
 * @param {unknown} entries
 * @returns {{ token: number, bytes: Uint8Array, logprob: number }[]}
 */
export function scoredTokensFrom(entries) {
  if (!Array.isArray(entries)) {
    return [];
  }
  return entries.map((entry) => ({
    token: Number(entry?.token),
    bytes: Uint8Array.from(Array.isArray(entry?.bytes) ? entry.bytes : []),
    logprob: typeof entry?.logprob === 'number' ? entry.logprob : -Infinity,
  }));
}
