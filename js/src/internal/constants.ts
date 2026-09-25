// Shared status codes and user-facing error messages.

export const defaultModelCacheName = 'llamadart-webgpu-model-cache-v1';

// Stable lifecycle error; consumers match on it to distinguish a disposed
// bridge from a transient runtime failure.
export const BRIDGE_DISPOSED_MESSAGE = 'Bridge has been disposed.';

// llamadart_webgpu_begin_generation returns this when a generation is already
// active. The core leaves g_last_error untouched in that case, so the message is
// fixed here rather than read from the shared buffer.
export const GENERATION_ALREADY_ACTIVE_RC = -7;

export const GENERATION_ALREADY_ACTIVE_MESSAGE =
  'Generation is already active on this bridge runtime.';

// The core puts this in the error when it rejects a grammar before generation
// starts. The worker that reported it is still healthy and the main thread
// would reject the grammar the same way, so the facade rethrows it even on the
// media-parts path, which otherwise falls back for any worker error.
export const INVALID_GRAMMAR_ERROR_TEXT =
  'Failed to initialize sampler chain (invalid grammar)';
