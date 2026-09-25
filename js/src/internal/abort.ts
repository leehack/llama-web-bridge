// AbortSignal helpers.

export function createAbortError(message: string): Error {
  if (typeof DOMException === 'function') {
    return new DOMException(message, 'AbortError');
  }

  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

export function throwIfAborted(
  signal: AbortSignal | null | undefined,
  message = 'Bridge operation was cancelled.',
): void {
  if (signal?.aborted) {
    throw createAbortError(message);
  }
}
