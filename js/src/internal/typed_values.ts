// Typed-array and integer coercion helpers.

export function toUint8Array(value: unknown): Uint8Array | null {
  if (!value) {
    return null;
  }

  if (value instanceof Uint8Array) {
    return value;
  }

  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }

  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value);
  }

  if (Array.isArray(value)) {
    return Uint8Array.from(value.map((v) => Number(v) & 0xff));
  }

  return null;
}

export function toFloat32Array(value: unknown): Float32Array | null {
  if (!value) {
    return null;
  }

  if (value instanceof Float32Array) {
    return value;
  }

  if (ArrayBuffer.isView(value)) {
    return new Float32Array(
      value.buffer,
      value.byteOffset,
      Math.floor(value.byteLength / Float32Array.BYTES_PER_ELEMENT),
    );
  }

  if (value instanceof ArrayBuffer) {
    return new Float32Array(value);
  }

  if (Array.isArray(value)) {
    return Float32Array.from(value.map((v) => Number(v) || 0));
  }

  return null;
}

// Not a type predicate: a false result must not narrow a number argument to
// never, since non-integers and out-of-range numbers land there too.
export function isInt32(value: unknown): boolean {
  return typeof value === 'number'
    && Number.isInteger(value)
    && value >= -0x80000000
    && value <= 0x7fffffff;
}
