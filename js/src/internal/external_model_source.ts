import { openModelCacheParts } from './download.ts';
import {
  blobModelReader,
  cachePartsModelReader,
} from './external_model_reader.ts';
import type { ExternalModelReader } from './external_model_reader.ts';
import { openOpfsModel } from './opfs_model_store.ts';

export type ExternalReaderKind = 'opfs_sync' | 'opfs_blob' | 'cache_parts' | 'blob_cache';

export interface CachedExternalModel {
  reader: ExternalModelReader;
  kind: ExternalReaderKind;
}

// A committed OPFS entry, a part-stored entry or a whole-body Cache Storage
// entry for the model, as a reader the core reads on demand. WebKit stores
// models in parts because it keeps a whole-body response resident, so it never
// reads one as a Blob.
export async function openCachedExternalModel(
  cacheName: string,
  cacheKey: string,
  options: { worker: boolean; wholeBody: boolean; signal: AbortSignal | null },
): Promise<CachedExternalModel | null> {
  const opfs = await openOpfsModel(cacheName, cacheKey, options.worker, options.signal);
  if (opfs) return { reader: opfs, kind: opfs.mode === 0 ? 'opfs_sync' : 'opfs_blob' };
  if (typeof globalThis.caches?.open !== 'function') return null;
  try {
    const cache = await globalThis.caches.open(cacheName);
    const parts = await openModelCacheParts(cache, cacheKey, cacheName);
    if (parts) {
      // Parts are read one at a time through readPart, never through the stream.
      void parts.response.body?.cancel().catch(() => {});
      return { reader: cachePartsModelReader(parts, options.signal), kind: 'cache_parts' };
    }
    if (!options.wholeBody) return null;
    const cached = await cache.match(cacheKey);
    if (!cached) return null;
    const blob = await cached.blob();
    if (blob.size <= 0) return null;
    return { reader: blobModelReader(blob, options.signal), kind: 'blob_cache' };
  } catch (_) {
    return null;
  }
}

// The reason a native read failed, once the native load has returned.
export function externalReadFailure(
  readers: Iterable<ExternalModelReader>,
): { code: number; error: unknown } | null {
  for (const reader of readers) {
    if (reader.failure) return reader.failure;
  }
  return null;
}

// Thrown inside the model load to restage from the network after a cached
// model could not be read; the load catches it and retries once.
export class ExternalReadRetry extends Error {
  constructor() {
    super('A cached model could not be read.');
    this.name = 'ExternalReadRetry';
  }
}
