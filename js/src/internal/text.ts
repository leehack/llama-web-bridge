// Prompt assembly and generated-text sanitation.

// A chat message as callers pass it; role and content are stringified.
interface PromptMessage {
  role?: unknown;
  content?: unknown;
}

export function buildPromptFromMessages(
  messages: readonly (PromptMessage | null | undefined)[] | null | undefined,
  addAssistant: unknown,
): string {
  const lines = [];
  for (const msg of messages || []) {
    const role = String(msg?.role ?? 'user');
    const content = String(msg?.content ?? '');
    lines.push(`${role}: ${content}`);
  }
  if (addAssistant) {
    lines.push('assistant: ');
  }
  return lines.join('\n');
}

export function looksLikeCorruptedGeneration(text: unknown): boolean {
  if (typeof text !== 'string' || text.length === 0) {
    return false;
  }

  const normalized = text.trim();
  if (normalized.length === 0) {
    return false;
  }

  const unusedTokens = text.match(/<unused\d+>/g) || [];
  if (unusedTokens.length >= 4) {
    return true;
  }

  const tokenLikeTags = text.match(/<[^>]{1,40}>/g) || [];
  if (tokenLikeTags.length >= 8) {
    return true;
  }

  const compact = text.replace(/\s+/g, '');
  if (compact.length === 0) {
    return false;
  }

  const tagRun = compact.match(/(?:<[^>]{2,32}>){6,}/);
  if (tagRun) {
    return true;
  }

  const alphaNum = (normalized.match(/[A-Za-z0-9]/g) || []).length;
  const printable = (normalized.match(/[\x20-\x7E]/g) || []).length;
  const angleBrackets = (normalized.match(/[<>]/g) || []).length;

  const alphaNumRatio = alphaNum / normalized.length;
  const printableRatio = printable / normalized.length;
  const bracketRatio = angleBrackets / normalized.length;

  if (normalized.length >= 24 && printableRatio > 0.95 && alphaNumRatio < 0.18) {
    return true;
  }

  if (normalized.length >= 24 && bracketRatio > 0.25) {
    return true;
  }

  return false;
}

export function trimUnstableUtf8Tail(text: unknown): string {
  if (typeof text !== 'string' || text.length === 0) {
    return '';
  }

  let end = text.length;
  while (end > 0 && text.charCodeAt(end - 1) === 0xFFFD) {
    end -= 1;
  }

  if (end > 0) {
    const tail = text.charCodeAt(end - 1);
    if (tail >= 0xD800 && tail <= 0xDBFF) {
      end -= 1;
    }
  }

  return end === text.length ? text : text.slice(0, end);
}
