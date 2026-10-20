// Model source descriptors, URLs, and split-GGUF shard expansion.

// One model URL, or the caller's explicit ordered shard list kept as given.
export type ModelSource = string | unknown[];

interface SplitShardPattern {
  prefix: string;
  width: number;
  total: number;
}

export function cloneModelSource(source: unknown): ModelSource {
  if (Array.isArray(source)) {
    // Preserve the caller's explicit shard order and values. The loader may
    // normalize the URLs for fetches, but recovery must never turn the array
    // into a comma-joined string or silently drop a shard.
    return source.slice();
  }

  return String(source || '').trim();
}

export function hasModelSource(source: unknown): boolean {
  return Array.isArray(source)
    ? source.length > 0
    : typeof source === 'string' && source.length > 0;
}

// The page's base URL, so a `<base href>` applies as it does to a page fetch.
// A worker has no page: callers resolve URLs on the main thread before posting
// them, because a worker would resolve them against its own script URL.
function pageBaseUrl(): string | undefined {
  if (typeof document !== 'undefined' && typeof document.baseURI === 'string' && document.baseURI.length > 0) {
    return document.baseURI;
  }
  return typeof window !== 'undefined' ? window.location.href : undefined;
}

export function basenameFromUrl(url: unknown): string {
  try {
    const parsed = new URL(url as string, pageBaseUrl());
    const pathname = parsed.pathname || '';
    const name = pathname.split('/').pop() || 'model.gguf';
    return name.includes('?') ? name.split('?')[0] : name;
  } catch (_) {
    const parts = String(url).split('/');
    return parts[parts.length - 1] || 'model.gguf';
  }
}

export function normalizeAbsoluteUrl(url: unknown): string {
  try {
    return new URL(url as string, pageBaseUrl()).toString();
  } catch (_) {
    return String(url);
  }
}

// Resolves each URL of a model source against the page, keeping a single URL
// a string so the loader still expands a split-GGUF name into its shards.
export function resolveModelSource(source: unknown): ModelSource {
  const cloned = cloneModelSource(source);
  if (Array.isArray(cloned)) {
    return cloned.map((value) => (
      typeof value === 'string' && value.trim().length > 0 ? normalizeAbsoluteUrl(value.trim()) : value
    ));
  }
  return cloned.length > 0 ? normalizeAbsoluteUrl(cloned) : cloned;
}

// Copies media parts with each `url` resolved against the page; other parts
// are kept as given.
export function resolvePartUrls<T>(parts: readonly T[]): T[] {
  return parts.map((part) => {
    const url = part != null && typeof part === 'object' ? (part as { url?: unknown }).url : undefined;
    return typeof url === 'string' && url.length > 0 ? { ...part, url: normalizeAbsoluteUrl(url) } : part;
  });
}

function parseSplitShardPattern(fileName: unknown): SplitShardPattern | null {
  if (typeof fileName !== 'string' || fileName.length === 0) {
    return null;
  }

  const match = fileName.match(/^(.*)-(\d{4,6})-of-(\d{4,6})\.gguf$/i);
  if (!match) {
    return null;
  }

  const total = Number(match[3]);
  if (!Number.isInteger(total) || total < 2 || total > 512) {
    return null;
  }

  return {
    prefix: match[1],
    width: Math.max(match[2].length, match[3].length),
    total,
  };
}

export function expandModelShardUrls(modelUrlOrUrls: unknown): string[] {
  if (Array.isArray(modelUrlOrUrls)) {
    return modelUrlOrUrls
      .map((value) => String(value || '').trim())
      .filter((value) => value.length > 0)
      .map((value) => normalizeAbsoluteUrl(value));
  }

  const source = String(modelUrlOrUrls || '').trim();
  if (source.length === 0) {
    return [];
  }

  try {
    const parsed = new URL(source, pageBaseUrl());
    const pathname = parsed.pathname || '';
    const slash = pathname.lastIndexOf('/');
    const dirPath = slash >= 0 ? pathname.slice(0, slash + 1) : '';
    const fileName = slash >= 0 ? pathname.slice(slash + 1) : pathname;
    const split = parseSplitShardPattern(fileName);
    if (!split) {
      return [parsed.toString()];
    }

    const totalShardId = String(split.total).padStart(split.width, '0');
    const urls: string[] = [];
    for (let shardIndex = 1; shardIndex <= split.total; shardIndex += 1) {
      const shardId = String(shardIndex).padStart(split.width, '0');
      parsed.pathname = `${dirPath}${split.prefix}-${shardId}-of-${totalShardId}.gguf`;
      urls.push(parsed.toString());
    }
    return urls;
  } catch (_) {
    return [source];
  }
}
