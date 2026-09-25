// Decision-head API constants and sequence/output codecs.

import { isInt32 } from './typed_values.js';

// Decision head API version shared with llama_webgpu_decision.h.
export const DECISION_API_VERSION = 1;

// Worker runDecision budget per sequence on top of a 10-minute base; one
// 512-token sequence takes a few seconds on the WASM CPU backend.
export const DECISION_WORKER_TIMEOUT_PER_SEQUENCE_MS = 60 * 1000;

export function decisionHandleFrom(handle) {
  if (!isInt32(handle) || handle <= 0) {
    throw new TypeError(`Decision head handle must be a positive integer, got ${String(handle)}.`);
  }
  return handle;
}

export function decisionHeadBytes(source) {
  if (source instanceof ArrayBuffer) {
    return new Uint8Array(source);
  }
  if (ArrayBuffer.isView(source)) {
    return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
  }
  return null;
}

function decisionIntegerList(value, label) {
  const isList = Array.isArray(value)
    || (ArrayBuffer.isView(value) && !(value instanceof DataView));
  if (!isList) {
    throw new TypeError(`${label} must be an array or typed array of integers.`);
  }
  const list = /** @type {ArrayLike<unknown>} */ (value);
  const out = new Int32Array(list.length);
  for (let index = 0; index < list.length; index += 1) {
    const item = list[index];
    if (!isInt32(item)) {
      throw new TypeError(`${label}[${index}] is ${String(item)}; expected a 32-bit integer.`);
    }
    out[index] = /** @type {number} */ (item);
  }
  return out;
}

/**
 * Copies decision sequences into plain `{tokens, markers, questionType}`
 * objects with `Int32Array` lists. Only integer types are checked here; the
 * core validates ranges against the loaded encoder before its first encoder
 * pass.
 *
 * @param {unknown} sequences
 * @returns {{ tokens: Int32Array, markers: Int32Array, questionType: number }[]}
 */
export function normalizeDecisionSequences(sequences) {
  if (!Array.isArray(sequences)) {
    throw new TypeError('Decision sequences must be an array.');
  }
  return sequences.map((sequence, index) => {
    const label = `Decision sequence ${index}`;
    if (!sequence || typeof sequence !== 'object') {
      throw new TypeError(`${label} must be an object with tokens, markers and questionType.`);
    }
    const tokens = decisionIntegerList(sequence.tokens, `${label} tokens`);
    const markers = decisionIntegerList(sequence.markers, `${label} markers`);
    const questionType = sequence.questionType;
    if (!isInt32(questionType)) {
      throw new TypeError(
        `${label} questionType is ${String(questionType)}; expected 0 (choice), 1 (score) or 2 (noul).`,
      );
    }
    return { tokens, markers, questionType: /** @type {number} */ (questionType) };
  });
}

/**
 * Encodes decision sequences as the core's input file: little-endian int32
 * count, then per sequence question type, token count, marker count, tokens
 * and markers.
 *
 * @param {unknown} sequences
 * @returns {Uint8Array}
 */
export function encodeDecisionSequences(sequences) {
  const encoded = normalizeDecisionSequences(sequences);
  let words = 1;
  for (const sequence of encoded) {
    words += 3 + sequence.tokens.length + sequence.markers.length;
  }
  const bytes = new Uint8Array(words * 4);
  const view = new DataView(bytes.buffer);
  let offset = 0;
  const put = (value) => {
    view.setInt32(offset, value, true);
    offset += 4;
  };
  put(encoded.length);
  for (const sequence of encoded) {
    put(sequence.questionType);
    put(sequence.tokens.length);
    put(sequence.markers.length);
    sequence.tokens.forEach(put);
    sequence.markers.forEach(put);
  }
  return bytes;
}

/**
 * Decodes the core's output file: per sequence little-endian int32 logit and
 * act counts, then the float32 logits and act logits.
 *
 * @param {Uint8Array} bytes
 * @param {number} count
 * @returns {{ logits: Float32Array, actLogits: Float32Array }[]}
 */
export function decodeDecisionOutputs(bytes, count) {
  const malformed = () => new Error('Decision output from the WebGPU core is malformed.');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;
  const floats = (length) => {
    const values = new Float32Array(length);
    for (let index = 0; index < length; index += 1) {
      values[index] = view.getFloat32(offset, true);
      offset += 4;
    }
    return values;
  };
  const outputs = [];
  for (let index = 0; index < count; index += 1) {
    if (bytes.byteLength - offset < 8) {
      throw malformed();
    }
    const logitCount = view.getInt32(offset, true);
    const actCount = view.getInt32(offset + 4, true);
    offset += 8;
    if (logitCount < 0 || actCount < 0 || (logitCount + actCount) * 4 > bytes.byteLength - offset) {
      throw malformed();
    }
    outputs.push({ logits: floats(logitCount), actLogits: floats(actCount) });
  }
  if (offset !== bytes.byteLength) {
    throw malformed();
  }
  return outputs;
}
