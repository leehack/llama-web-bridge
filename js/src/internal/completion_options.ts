// Completion sampling options that are rejected rather than clamped.

import type { CompletionCapabilities, ThinkingBudgetOptions } from '../llama_webgpu_bridge.d.ts';

export interface CompletionSamplingOptions {
  minP: number;
  presencePenalty: number;
  thinkingBudget: Required<ThinkingBudgetOptions> | null;
}

export const NO_COMPLETION_CAPABILITIES: Readonly<CompletionCapabilities> = Object.freeze({
  presencePenalty: false,
  minP: false,
  thinkingBudget: false,
});

const MAX_THINKING_BUDGET_TOKENS = 0x7fffffff;

function optionalNumber(value: unknown, name: string, isValid: (value: number) => boolean, range: string): number | null {
  if (value == null) {
    return null;
  }
  if (typeof value !== 'number' || !Number.isFinite(Math.fround(value)) || !isValid(value)) {
    throw new RangeError(`CompletionOptions.${name} must be a finite number${range}; got ${String(value)}.`);
  }
  return value;
}

function tag(budget: Record<string, unknown>, name: 'startTag' | 'endTag'): string {
  const value = budget[name];
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new TypeError(`CompletionOptions.thinkingBudget.${name} must be a non-empty string.`);
  }
  return value;
}

function thinkingBudget(value: unknown, hasMediaParts: boolean): Required<ThinkingBudgetOptions> | null {
  if (value == null) {
    return null;
  }
  if (typeof value !== 'object') {
    throw new TypeError('CompletionOptions.thinkingBudget must be an object.');
  }
  const budget = value as Record<string, unknown>;
  const maxTokens = budget.maxTokens;
  if (!Number.isInteger(maxTokens) || (maxTokens as number) < 0 || (maxTokens as number) > MAX_THINKING_BUDGET_TOKENS) {
    throw new RangeError(
      `CompletionOptions.thinkingBudget.maxTokens must be an integer from 0 to ${MAX_THINKING_BUDGET_TOKENS}; `
      + `got ${String(maxTokens)}.`,
    );
  }
  const startTag = tag(budget, 'startTag');
  const endTag = tag(budget, 'endTag');
  const forcedMessage = budget.forcedMessage ?? '';
  if (typeof forcedMessage !== 'string') {
    throw new TypeError('CompletionOptions.thinkingBudget.forcedMessage must be a string.');
  }
  if (hasMediaParts) {
    throw new Error('CompletionOptions.thinkingBudget supports text-only prompts; remove the media parts.');
  }
  return { maxTokens: maxTokens as number, startTag, endTag, forcedMessage };
}

/**
 * Resolves `minP`, `presencePenalty` and `thinkingBudget`. Throws a RangeError
 * or TypeError for an invalid value, including a number that is not finite as
 * the core's 32-bit float, and an Error for a thinking budget with media parts.
 */
export function resolveCompletionSamplingOptions(
  options: { minP?: unknown; presencePenalty?: unknown; thinkingBudget?: unknown; parts?: unknown } | null | undefined,
): CompletionSamplingOptions {
  return {
    minP: optionalNumber(options?.minP, 'minP', (value) => value >= 0 && value <= 1, ' from 0 to 1') ?? 0,
    presencePenalty: optionalNumber(options?.presencePenalty, 'presencePenalty', () => true, '') ?? 0,
    thinkingBudget: thinkingBudget(
      options?.thinkingBudget,
      Array.isArray(options?.parts) && options.parts.length > 0,
    ),
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
    thinkingBudget: parsed?.thinkingBudget === true,
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
    sampling.thinkingBudget != null && !capabilities.thinkingBudget ? 'thinkingBudget' : null,
  ].filter((name) => name != null);
  if (missing.length > 0) {
    throw new Error(
      `The loaded WebGPU core does not support CompletionOptions.${missing.join(' or CompletionOptions.')}.`,
    );
  }
}
