import { throwIfAborted } from './abort.ts';
import { blobModelReader, opfsModelReader } from './external_model_reader.ts';
import type { ExternalModelReader, SyncModelFile } from './external_model_reader.ts';

interface SyncAccessHandle extends SyncModelFile {
  write(buffer: Uint8Array<ArrayBuffer>, options: { at: number }): number;
  truncate(size: number): void;
  getSize(): number;
  flush(): void;
}

type OpfsFile = FileSystemFileHandle & {
  createSyncAccessHandle?: (options?: { mode: 'read-only' }) => Promise<SyncAccessHandle>;
};
type IterableDirectory = FileSystemDirectoryHandle & { keys(): AsyncIterable<string> };

const OPFS_ROOT = 'llamadart-model-cache-v1';
const LOCK_PREFIX = 'llamadart-opfs-cache:';
const DATA_NAME = /^([a-f0-9]{64})--([a-f0-9-]{36})\.bin$/;

type ReleaseLock = () => void;

async function acquireLock(key: string, mode: 'shared' | 'exclusive'): Promise<ReleaseLock | null> {
  const locks = globalThis.navigator?.locks;
  if (!locks) return null;
  return new Promise((resolve, reject) => {
    void locks.request(LOCK_PREFIX + key, { ifAvailable: true, mode }, (lock) => {
      if (!lock) {
        resolve(null);
        return;
      }
      return new Promise<void>((release) => resolve(release));
    }).catch(reject);
  });
}

async function keyFor(cacheName: string, cacheKey: string): Promise<string> {
  const value = new TextEncoder().encode(cacheName + '\0' + cacheKey);
  const digest = await crypto.subtle.digest('SHA-256', value);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function remove(directory: FileSystemDirectoryHandle, name: string): Promise<boolean> {
  try {
    await directory.removeEntry(name);
    return true;
  } catch (_) {
    return false;
  }
}

async function directories(create: boolean) {
  if (typeof globalThis.navigator?.storage?.getDirectory !== 'function' || !globalThis.navigator.locks) return null;
  try {
    const root = await navigator.storage.getDirectory();
    const cache = await root.getDirectoryHandle(OPFS_ROOT, { create });
    const files = await cache.getDirectoryHandle('files', { create }) as IterableDirectory;
    const manifests = await cache.getDirectoryHandle('manifests', { create });
    return { files, manifests };
  } catch (_) {
    return null;
  }
}

type Store = NonNullable<Awaited<ReturnType<typeof directories>>>;
type Manifest = { record: string; decodedBytes: number };

async function manifestFor(store: Store, key: string): Promise<Manifest | null> {
  try {
    const handle = await store.manifests.getFileHandle(key + '.json');
    const file = await handle.getFile();
    if (file.size > 1024) return null;
    const value = JSON.parse(await file.text()) as Partial<Manifest> & { version?: unknown };
    if (value.version !== 1 || typeof value.record !== 'string' || !/^[a-f0-9-]{36}$/.test(value.record)
        || !Number.isSafeInteger(value.decodedBytes) || value.decodedBytes! <= 0) return null;
    return value as Manifest;
  } catch (_) {
    return null;
  }
}

function dataName(key: string, record: string): string {
  return key + '--' + record + '.bin';
}

let sweepStarted = false;
async function sweepOrphans(store: Store): Promise<void> {
  if (typeof store.files.keys !== 'function') return;
  for await (const name of store.files.keys()) {
    const match = DATA_NAME.exec(name);
    if (!match) continue;
    const [, key, record] = match;
    const release = await acquireLock(key, 'exclusive');
    if (!release) continue;
    try {
      if ((await manifestFor(store, key))?.record !== record) await remove(store.files, name);
    } finally {
      release();
    }
  }
}

export async function openOpfsModel(
  cacheName: string, cacheKey: string, worker: boolean, signal: AbortSignal | null,
): Promise<ExternalModelReader | null> {
  const store = await directories(false);
  if (!store) return null;
  const key = await keyFor(cacheName, cacheKey);
  const release = await acquireLock(key, 'shared');
  if (!release) return null;
  let reader: ExternalModelReader | null = null;
  try {
    const manifest = await manifestFor(store, key);
    if (!manifest) return null;
    const handle = await store.files.getFileHandle(dataName(key, manifest.record)) as OpfsFile;
    if (worker && handle.createSyncAccessHandle) {
      let sync: SyncAccessHandle | null = null;
      try {
        sync = await handle.createSyncAccessHandle({ mode: 'read-only' });
        if (sync.getSize() === manifest.decodedBytes) {
          reader = opfsModelReader(sync, manifest.decodedBytes, signal);
          sync = null;
        }
      } catch (_) {
        // A browser without a concurrent sync handle can still read a Blob snapshot.
      } finally {
        sync?.close();
      }
    }
    if (!reader) {
      const file = await handle.getFile();
      if (file.size !== manifest.decodedBytes) return null;
      reader = blobModelReader(file, signal);
    }
    const close = reader.close;
    reader.close = () => {
      try { close(); } finally { release(); }
    };
    return reader;
  } catch (_) {
    return null;
  } finally {
    if (!reader) release();
  }
}

export async function hasOpfsModel(cacheName: string, cacheKey: string): Promise<boolean> {
  const store = await directories(false);
  if (!store) return false;
  const manifest = await manifestFor(store, await keyFor(cacheName, cacheKey));
  if (!manifest) return false;
  try {
    await store.files.getFileHandle(dataName(await keyFor(cacheName, cacheKey), manifest.record));
    return true;
  } catch (_) {
    return false;
  }
}

export async function evictOpfsModel(cacheName: string, cacheKey: string): Promise<boolean> {
  const store = await directories(false);
  if (!store) return false;
  const key = await keyFor(cacheName, cacheKey);
  // A reader this context just closed releases its lock asynchronously.
  let release: ReleaseLock | null = null;
  for (let attempt = 0; attempt < 5 && !release; ++attempt) {
    if (attempt > 0) await new Promise((resolve) => setTimeout(resolve, 20));
    release = await acquireLock(key, 'exclusive');
  }
  if (!release) return false;
  try {
    const manifest = await manifestFor(store, key);
    const removed = await remove(store.manifests, key + '.json');
    if (removed && manifest) await remove(store.files, dataName(key, manifest.record));
    return removed;
  } finally {
    release();
  }
}

export class OpfsWriteError extends Error {
  constructor() { super('OPFS model storage could not accept the download.'); }
}

export interface OpfsModelDownload {
  // Whether finish() recorded the file as a cache entry.
  readonly committed: boolean;
  writeChunk(chunk: Uint8Array, position: number): number;
  reset(): void;
  finish(size: number): Promise<ExternalModelReader>;
  dispose(): Promise<void>;
}

export async function createOpfsModelDownload(
  cacheName: string,
  cacheKey: string,
  signal: AbortSignal | null,
): Promise<OpfsModelDownload | null> {
  const store = await directories(true);
  if (!store) return null;
  if (!sweepStarted) {
    sweepStarted = true;
    void sweepOrphans(store).catch(() => {});
  }
  const key = await keyFor(cacheName, cacheKey);
  const release = await acquireLock(key, 'exclusive');
  if (!release) return null;
  const record = crypto.randomUUID();
  const name = dataName(key, record);
  let sync: SyncAccessHandle | null = null;
  let committed = false;
  let disposed = false;
  let handedOff = false;
  const cleanUp = async () => {
    if (disposed) return;
    disposed = true;
    try { sync?.close(); } catch (_) {}
    sync = null;
    try {
      if (!committed) await remove(store.files, name);
    } finally {
      release();
    }
  };
  try {
    throwIfAborted(signal, 'Model download was cancelled.');
    const file = await store.files.getFileHandle(name, { create: true }) as OpfsFile;
    sync = await file.createSyncAccessHandle?.() ?? null;
    if (!sync) {
      await cleanUp();
      return null;
    }
  } catch (_) {
    await cleanUp();
    throwIfAborted(signal, 'Model download was cancelled.');
    return null;
  }

  return {
    get committed() {
      return committed;
    },
    writeChunk(chunk, position) {
      if (!sync) throw new OpfsWriteError();
      try {
        const bytes = chunk as Uint8Array<ArrayBuffer>;
        const count = sync.write(bytes, { at: position });
        if (count <= 0 || !Number.isInteger(count) || count > bytes.byteLength) throw new OpfsWriteError();
        return count;
      } catch (_) {
        throw new OpfsWriteError();
      }
    },
    reset() {
      try { sync!.truncate(0); } catch (_) { throw new OpfsWriteError(); }
    },
    async finish(size) {
      const handle = sync;
      if (!handle || !Number.isSafeInteger(size) || size <= 0 || handle.getSize() !== size) {
        throw new OpfsWriteError();
      }
      throwIfAborted(signal, 'Model download was cancelled.');
      let manifestWriter: FileSystemWritableFileStream | null = null;
      try {
        handle.flush();
        const old = await manifestFor(store, key);
        const manifest = await store.manifests.getFileHandle(key + '.json', { create: true });
        manifestWriter = await manifest.createWritable();
        await manifestWriter.write(JSON.stringify({ version: 1, record, decodedBytes: size }));
        throwIfAborted(signal, 'Model download was cancelled.');
        await manifestWriter.close();
        manifestWriter = null;
        committed = true;
        if (old) await remove(store.files, dataName(key, old.record));
      } catch (_) {
        try { await manifestWriter?.abort(); } catch (_) {}
        throwIfAborted(signal, 'Model download was cancelled.');
        // The download is complete; without a manifest it is only read once.
      }

      const reader = opfsModelReader(handle, size, signal);
      const close = reader.close;
      reader.close = () => {
        sync = null;
        try { close(); } finally { void cleanUp(); }
      };
      handedOff = true;
      return reader;
    },
    async dispose() {
      if (!handedOff) await cleanUp();
    },
  };
}
