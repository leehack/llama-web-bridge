// Completion sampling options that are rejected rather than clamped.

import type {
  CompletionCapabilities,
  NgramCacheSource,
  SpeculativeDecodingStrategy,
  ThinkingBudgetOptions,
} from '../llama_webgpu_bridge.d.ts';

export interface CompletionSamplingOptions {
  minP: number;
  presencePenalty: number;
  thinkingBudget: Required<ThinkingBudgetOptions> | null;
  speculative: ResolvedSpeculativeOptions | null;
}

/**
 * Speculative settings as the core takes them, resolved as native llamadart
 * resolves SpeculativeDecodingConfig. -1 or 0 leaves llama.cpp's default.
 */
export interface ResolvedSpeculativeOptions {
  strategies: SpeculativeDecodingStrategy[];
  draftTokenMax: number;
  draftTokenMin: number;
  minProbability: number;
  draftSplitProbability: number;
  ngramSizeN: number;
  ngramSizeM: number;
  ngramMinHits: number;
  ngramMatch: number;
  ngramTokenMin: number;
  ngramTokenMax: number;
  ngramCacheStatic: NgramCacheSource | null;
  ngramCacheDynamic: NgramCacheSource | null;
}

export const SPECULATIVE_DECODING_STRATEGIES: readonly SpeculativeDecodingStrategy[] = Object.freeze([
  'draft-simple',
  'draft-eagle3',
  'draft-mtp',
  'draft-dflash',
  'draft-dspark',
  'ngram-simple',
  'ngram-map-k',
  'ngram-map-k4v',
  'ngram-mod',
  'ngram-cache',
]);

function speculativeFlags(isSupported: (strategy: SpeculativeDecodingStrategy) => boolean) {
  return Object.fromEntries(
    SPECULATIVE_DECODING_STRATEGIES.map((strategy) => [strategy, isSupported(strategy)]),
  ) as Record<SpeculativeDecodingStrategy, boolean>;
}

export const NO_COMPLETION_CAPABILITIES: Readonly<CompletionCapabilities> = Object.freeze({
  presencePenalty: false,
  minP: false,
  thinkingBudget: false,
  speculativeDecoding: Object.freeze(speculativeFlags(() => false)),
});

const MAX_NGRAM_SIZE = 0xffff;
const MAX_INT32 = 0x7fffffff;

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

function speculativeInteger(
  config: Record<string, unknown>,
  name: string,
  min: number,
  max: number,
): number | null {
  const value = config[name];
  if (value == null) {
    return null;
  }
  if (!Number.isInteger(value) || (value as number) < min || (value as number) > max) {
    throw new RangeError(
      `CompletionOptions.speculativeDecoding.${name} must be an integer from ${min} to ${max}; got ${String(value)}.`,
    );
  }
  return value as number;
}

function speculativeProbability(config: Record<string, unknown>, name: string): number | null {
  const value = config[name];
  if (value == null) {
    return null;
  }
  if (typeof value !== 'number' || !(value >= 0 && value <= 1)) {
    throw new RangeError(
      `CompletionOptions.speculativeDecoding.${name} must be a number from 0 to 1; got ${String(value)}.`,
    );
  }
  return value;
}

function ngramCacheSource(config: Record<string, unknown>, name: string): NgramCacheSource | null {
  const value = config[name];
  if (value == null) {
    return null;
  }
  const isBytes = value instanceof ArrayBuffer || ArrayBuffer.isView(value);
  if (!(typeof value === 'string' || isBytes)) {
    throw new TypeError(
      `CompletionOptions.speculativeDecoding.${name} must be a URL string, an ArrayBuffer or a typed array.`,
    );
  }
  if ((typeof value === 'string' ? value.length : (value as ArrayBuffer | ArrayBufferView).byteLength) === 0) {
    throw new TypeError(`CompletionOptions.speculativeDecoding.${name} is empty.`);
  }
  return value as NgramCacheSource;
}

function isDraftStrategy(strategy: SpeculativeDecodingStrategy): boolean {
  return strategy.startsWith('draft-');
}

function speculativeDecoding(
  value: unknown,
  conflicts: { hasMediaParts: boolean; hasGrammar: boolean; hasThinkingBudget: boolean },
): ResolvedSpeculativeOptions | null {
  if (value == null) {
    return null;
  }
  if (typeof value !== 'object') {
    throw new TypeError('CompletionOptions.speculativeDecoding must be an object.');
  }
  const config = value as Record<string, unknown>;
  const requested = config.strategies;
  if (!Array.isArray(requested) || requested.length === 0) {
    throw new TypeError('CompletionOptions.speculativeDecoding.strategies must be a non-empty array.');
  }
  const strategies: SpeculativeDecodingStrategy[] = [];
  for (const strategy of requested) {
    if (!SPECULATIVE_DECODING_STRATEGIES.includes(strategy)) {
      throw new TypeError(
        `CompletionOptions.speculativeDecoding.strategies has an unknown strategy ${JSON.stringify(strategy)}; `
        + `expected one of ${SPECULATIVE_DECODING_STRATEGIES.join(', ')}.`,
      );
    }
    if (!strategies.includes(strategy)) {
      strategies.push(strategy);
    }
  }

  const draftTokenMax = speculativeInteger(config, 'draftTokenMax', 0, MAX_INT32);
  const draftTokenMin = speculativeInteger(config, 'draftTokenMin', 0, MAX_INT32);
  const minProbability = speculativeProbability(config, 'minProbability');
  const draftSplitProbability = speculativeProbability(config, 'draftSplitProbability');
  const ngramSizeN = speculativeInteger(config, 'ngramSizeN', 1, MAX_NGRAM_SIZE);
  const ngramSizeM = speculativeInteger(config, 'ngramSizeM', 1, MAX_NGRAM_SIZE);
  const ngramMinHits = speculativeInteger(config, 'ngramMinHits', 1, MAX_NGRAM_SIZE);
  const ngramMatch = speculativeInteger(config, 'ngramMatch', 1, MAX_NGRAM_SIZE);
  const ngramTokenMin = speculativeInteger(config, 'ngramTokenMin', 0, MAX_INT32);
  const ngramTokenMax = speculativeInteger(config, 'ngramTokenMax', 0, MAX_INT32);
  const ngramCacheStatic = ngramCacheSource(config, 'ngramCacheStatic');
  const ngramCacheDynamic = ngramCacheSource(config, 'ngramCacheDynamic');

  if (strategies.filter(isDraftStrategy).length > 1) {
    throw new Error(
      'CompletionOptions.speculativeDecoding can mix n-gram strategies with at most one draft-* strategy.',
    );
  }
  if (
    !strategies.some(isDraftStrategy)
    && (draftTokenMin != null || minProbability != null || draftSplitProbability != null)
  ) {
    throw new Error(
      'CompletionOptions.speculativeDecoding: n-gram strategies use token history and do not support '
      + 'draftTokenMin, minProbability or draftSplitProbability unless a draft-* strategy is also enabled.',
    );
  }
  if ((ngramCacheStatic != null || ngramCacheDynamic != null) && !strategies.includes('ngram-cache')) {
    throw new Error(
      'CompletionOptions.speculativeDecoding: ngramCacheStatic and ngramCacheDynamic need the ngram-cache strategy.',
    );
  }
  if (ngramTokenMin != null && ngramTokenMax != null && ngramTokenMin > ngramTokenMax) {
    throw new RangeError(
      `CompletionOptions.speculativeDecoding.ngramTokenMin (${ngramTokenMin}) must not exceed ngramTokenMax (${ngramTokenMax}).`,
    );
  }

  let resolvedDraftTokenMax = 0;
  for (const strategy of strategies) {
    let strategyMax: number;
    if (isDraftStrategy(strategy)) {
      strategyMax = draftTokenMax ?? 3;
    } else if (strategy === 'ngram-mod') {
      strategyMax = ngramTokenMax ?? draftTokenMax ?? 64;
    } else if (strategy === 'ngram-cache') {
      strategyMax = draftTokenMax ?? 8;
    } else {
      strategyMax = ngramSizeM ?? 48;
    }
    resolvedDraftTokenMax = Math.max(resolvedDraftTokenMax, strategyMax);
  }
  if (resolvedDraftTokenMax === 0) {
    resolvedDraftTokenMax = 64;
  }
  if ((draftTokenMin ?? 0) > resolvedDraftTokenMax) {
    throw new RangeError(
      `CompletionOptions.speculativeDecoding.draftTokenMin (${draftTokenMin}) must not exceed the draft token maximum `
      + `(${resolvedDraftTokenMax}).`,
    );
  }

  if (conflicts.hasMediaParts) {
    throw new Error('CompletionOptions.speculativeDecoding supports text-only prompts; remove the media parts.');
  }
  if (conflicts.hasThinkingBudget) {
    throw new Error('CompletionOptions.speculativeDecoding cannot be combined with CompletionOptions.thinkingBudget.');
  }
  if (conflicts.hasGrammar) {
    throw new Error('CompletionOptions.speculativeDecoding does not support CompletionOptions.grammar.');
  }

  return {
    strategies,
    draftTokenMax: resolvedDraftTokenMax,
    draftTokenMin: draftTokenMin ?? 0,
    minProbability: minProbability ?? -1,
    draftSplitProbability: draftSplitProbability ?? -1,
    ngramSizeN: ngramSizeN ?? 0,
    ngramSizeM: ngramSizeM ?? 0,
    ngramMinHits: ngramMinHits ?? 0,
    ngramMatch: ngramMatch ?? 0,
    ngramTokenMin: ngramTokenMin ?? -1,
    ngramTokenMax: ngramTokenMax ?? (strategies.includes('ngram-mod') ? draftTokenMax ?? 0 : 0),
    ngramCacheStatic,
    ngramCacheDynamic,
  };
}

/**
 * Resolves `minP`, `presencePenalty`, `thinkingBudget` and `speculativeDecoding`. Throws a RangeError
 * or TypeError for an invalid value, including a number that is not finite as
 * the core's 32-bit float, and an Error for a thinking budget with media parts
 * or a speculative configuration native llamadart rejects.
 */
export function resolveCompletionSamplingOptions(
  options: {
    minP?: unknown;
    presencePenalty?: unknown;
    thinkingBudget?: unknown;
    speculativeDecoding?: unknown;
    grammar?: unknown;
    parts?: unknown;
  } | null | undefined,
): CompletionSamplingOptions {
  const hasMediaParts = Array.isArray(options?.parts) && options.parts.length > 0;
  const budget = thinkingBudget(options?.thinkingBudget, hasMediaParts);
  return {
    minP: optionalNumber(options?.minP, 'minP', (value) => value >= 0 && value <= 1, ' from 0 to 1') ?? 0,
    presencePenalty: optionalNumber(options?.presencePenalty, 'presencePenalty', () => true, '') ?? 0,
    thinkingBudget: budget,
    speculative: speculativeDecoding(options?.speculativeDecoding, {
      hasMediaParts,
      hasGrammar: typeof options?.grammar === 'string' && options.grammar.length > 0,
      hasThinkingBudget: budget != null,
    }),
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
  const speculative = parsed?.speculativeDecoding as Record<string, unknown> | null | undefined;
  return {
    presencePenalty: parsed?.presencePenalty === true,
    minP: parsed?.minP === true,
    thinkingBudget: parsed?.thinkingBudget === true,
    speculativeDecoding: speculativeFlags((strategy) => speculative?.[strategy] === true),
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
    sampling.speculative != null && !Object.values(capabilities.speculativeDecoding).includes(true)
      ? 'speculativeDecoding'
      : null,
  ].filter((name) => name != null);
  if (missing.length > 0) {
    throw new Error(
      `The loaded WebGPU core does not support CompletionOptions.${missing.join(' or CompletionOptions.')}.`,
    );
  }
}
