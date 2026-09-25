// AbortSignal helpers.

export function createAbortError(message) {
  if (typeof DOMException === 'function') {
    return new DOMException(message, 'AbortError');
  }

  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

export function throwIfAborted(signal, message = 'Bridge operation was cancelled.') {
  if (signal?.aborted) {
    throw createAbortError(message);
  }
}
