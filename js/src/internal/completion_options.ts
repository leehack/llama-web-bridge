// Completion sampling options that are rejected rather than clamped.

import type { CompletionCapabilities } from '../llama_webgpu_bridge.d.ts';

export interface CompletionSamplingOptions {
  minP: number;
  presencePenalty: number;
}

export const NO_COMPLETION_CAPABILITIES: Readonly<CompletionCapabilities> = Object.freeze({
  presencePenalty: false,
  minP: false,
});

function optionalNumber(value: unknown, name: string, isValid: (value: number) => boolean, range: string): number | null {
  if (value == null) {
    return null;
  }
  if (typeof value !== 'number' || !Number.isFinite(Math.fround(value)) || !isValid(value)) {
    throw new RangeError(`CompletionOptions.${name} must be a finite number${range}; got ${String(value)}.`);
  }
  return value;
}

/** Resolves `minP` and `presencePenalty`, throwing a RangeError for a value that is invalid or not finite as the core's 32-bit float. */
export function resolveCompletionSamplingOptions(
  options: { minP?: unknown; presencePenalty?: unknown } | null | undefined,
): CompletionSamplingOptions {
  return {
    minP: optionalNumber(options?.minP, 'minP', (value) => value >= 0 && value <= 1, ' from 0 to 1') ?? 0,
    presencePenalty: optionalNumber(options?.presencePenalty, 'presencePenalty', () => true, '') ?? 0,
  };
}

/** Parses the core's capability JSON; a field that is not `true` is unsupported. */
export function completionCapabilitiesFrom(raw: string | null): CompletionCapabilities {
  let parsed: Record<string, unknown> | null = null;
  try {
    parsed = JSON.parse(raw || '{}');
  } catch (_) {
    return { ...NO_COMPLETION_CAPABILITIES };
  }
  return {
    presencePenalty: parsed?.presencePenalty === true,
    minP: parsed?.minP === true,
  };
}

/** Throws when a non-default option needs a capability the core lacks. */
export function requireCompletionCapabilities(
  sampling: CompletionSamplingOptions,
  capabilities: CompletionCapabilities,
): void {
  const missing = [
    sampling.minP !== 0 && !capabilities.minP ? 'minP' : null,
    sampling.presencePenalty !== 0 && !capabilities.presencePenalty ? 'presencePenalty' : null,
  ].filter((name) => name != null);
  if (missing.length > 0) {
    throw new Error(
      `The loaded WebGPU core does not support CompletionOptions.${missing.join(' or CompletionOptions.')}.`,
    );
  }
}
