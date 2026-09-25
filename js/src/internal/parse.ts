// Lenient parsing of numeric, boolean, and enum configuration values.

export function parsePositiveInteger(value: unknown): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return 0;
  }
  return Math.trunc(numeric);
}

export function parseInteger(value: unknown, fallback = 0): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.trunc(numeric);
}

export function parseBooleanFlag(value: unknown, fallback = false): boolean {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value !== 0;
  }
  return fallback;
}

// 1 or 0 for an explicit flag, -1 to leave the native default in place.
export function parseOptionalBooleanFlag(value: unknown): 1 | 0 | -1 {
  if (typeof value === 'boolean') {
    return value ? 1 : 0;
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value !== 0 ? 1 : 0;
  }
  return -1;
}

export function parseEnumValue(value: unknown, allowed: readonly number[], fallback: number): number {
  const parsed = parseInteger(value, fallback);
  return allowed.includes(parsed) ? parsed : fallback;
}

export function parsePositiveNumber(value: unknown): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : 0;
}
