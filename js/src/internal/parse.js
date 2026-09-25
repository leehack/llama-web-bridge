// Lenient parsing of numeric, boolean, and enum configuration values.

export function parsePositiveInteger(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return 0;
  }
  return Math.trunc(numeric);
}

export function parseInteger(value, fallback = 0) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.trunc(numeric);
}

export function parseBooleanFlag(value, fallback = false) {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value !== 0;
  }
  return fallback;
}

export function parseOptionalBooleanFlag(value) {
  if (typeof value === 'boolean') {
    return value ? 1 : 0;
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value !== 0 ? 1 : 0;
  }
  return -1;
}

export function parseEnumValue(value, allowed, fallback) {
  const parsed = parseInteger(value, fallback);
  return allowed.includes(parsed) ? parsed : fallback;
}

export function parsePositiveNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : 0;
}
