const textEncoder = new TextEncoder();
const defaultModelCacheName = 'llamadart-webgpu-model-cache-v1';

function basenameFromUrl(url) {
  try {
    const parsed = new URL(url, typeof window !== 'undefined' ? window.location.href : undefined);
    const pathname = parsed.pathname || '';
    const name = pathname.split('/').pop() || 'model.gguf';
    return name.includes('?') ? name.split('?')[0] : name;
  } catch (_) {
    const parts = String(url).split('/');
    return parts[parts.length - 1] || 'model.gguf';
  }
}

function normalizeAbsoluteUrl(url) {
  try {
    return new URL(url, typeof window !== 'undefined' ? window.location.href : undefined).toString();
  } catch (_) {
    return String(url);
  }
}

function hasReadableResponseStream(response) {
  return !!(
    response
    && response.body
    && typeof response.body.getReader === 'function'
  );
}

function parseSplitShardPattern(fileName) {
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

function expandModelShardUrls(modelUrlOrUrls) {
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
    const parsed = new URL(source, typeof window !== 'undefined' ? window.location.href : undefined);
    const pathname = parsed.pathname || '';
    const slash = pathname.lastIndexOf('/');
    const dirPath = slash >= 0 ? pathname.slice(0, slash + 1) : '';
    const fileName = slash >= 0 ? pathname.slice(slash + 1) : pathname;
    const split = parseSplitShardPattern(fileName);
    if (!split) {
      return [parsed.toString()];
    }

    const totalShardId = String(split.total).padStart(split.width, '0');
    const urls = [];
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

function sumProgressValues(values) {
  let total = 0;
  for (const value of values || []) {
    const numeric = Number(value);
    if (Number.isFinite(numeric) && numeric > 0) {
      total += numeric;
    }
  }
  return total;
}

function parsePositiveInteger(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return 0;
  }
  return Math.trunc(numeric);
}

function parseTotalFromContentRangeHeader(contentRangeHeader) {
  if (typeof contentRangeHeader !== 'string' || contentRangeHeader.length === 0) {
    return 0;
  }

  const slash = contentRangeHeader.lastIndexOf('/');
  if (slash < 0 || slash + 1 >= contentRangeHeader.length) {
    return 0;
  }

  return parsePositiveInteger(contentRangeHeader.slice(slash + 1));
}

function inferResponseTotalBytes(response, loadedFallback = 0) {
  if (!response || !response.headers) {
    return parsePositiveInteger(loadedFallback);
  }

  const linkedSize = parsePositiveInteger(response.headers.get('x-linked-size'));
  if (linkedSize > 0) {
    return linkedSize;
  }

  const contentRangeTotal = parseTotalFromContentRangeHeader(
    response.headers.get('content-range'),
  );
  if (contentRangeTotal > 0) {
    return contentRangeTotal;
  }

  const contentLength = parsePositiveInteger(response.headers.get('content-length'));
  if (contentLength > 0) {
    return contentLength;
  }

  return parsePositiveInteger(loadedFallback);
}

function isRetryableStreamNetworkError(error) {
  const text = String(error || '').toLowerCase();
  return text.includes('network error')
    || text.includes('failed to fetch')
    || text.includes('networkerror')
    || text.includes('err_network_io_suspended')
    || text.includes('the network connection was lost')
    || text.includes('connection reset');
}

function normalizeFactory(moduleExport) {
  if (typeof moduleExport === 'function') {
    return moduleExport;
  }

  if (moduleExport && typeof moduleExport.default === 'function') {
    return moduleExport.default;
  }

  if (moduleExport && typeof moduleExport.createLlamaWebGpuCoreModule === 'function') {
    return moduleExport.createLlamaWebGpuCoreModule;
  }

  throw new Error('Unable to resolve llama_webgpu_core factory function');
}

async function importCoreFactory(moduleUrl) {
  const exportedModule = await import(moduleUrl);
  return normalizeFactory(exportedModule);
}

function buildPromptFromMessages(messages, addAssistant) {
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

function isSafariUserAgent(userAgent) {
  if (typeof userAgent !== 'string' || userAgent.length === 0) {
    return false;
  }

  const hasSafariToken = /Safari\//.test(userAgent);
  const hasOtherBrowserToken = /(Chrome|Chromium|CriOS|Edg|OPR|Firefox|FxiOS)\//.test(userAgent);
  return hasSafariToken && !hasOtherBrowserToken;
}

function looksLikeCorruptedGeneration(text) {
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

function isCrossOriginIsolatedRuntime() {
  try {
    if (typeof globalThis.crossOriginIsolated === 'boolean') {
      return globalThis.crossOriginIsolated;
    }
  } catch (_) {
    // ignore environment probing failures
  }

  // Assume isolated in runtimes that do not expose the signal.
  return true;
}

async function readResponseBytesWithProgress(response, progressCallback) {
  const total = Number(response.headers.get('content-length')) || 0;

  if (!response.body || typeof response.body.getReader !== 'function') {
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (typeof progressCallback === 'function') {
      progressCallback({ loaded: bytes.byteLength, total: total || bytes.byteLength });
    }
    return bytes;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  let lastBucket = -1;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    if (!value || value.length === 0) {
      continue;
    }

    // Some browsers may reuse the same Uint8Array backing store across reads.
    // Clone each chunk before storing so reassembly is deterministic.
    const chunk = value.slice ? value.slice() : new Uint8Array(value);
    chunks.push(chunk);
    loaded += chunk.length;

    if (typeof progressCallback === 'function') {
      const effectiveTotal = total || loaded;
      const bucket = effectiveTotal > 0
        ? Math.floor((loaded / effectiveTotal) * 100)
        : -1;
      if (bucket > lastBucket) {
        lastBucket = bucket;
        progressCallback({ loaded, total: effectiveTotal });
      }
    }
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }

  if (typeof progressCallback === 'function') {
    progressCallback({ loaded, total: total || loaded });
  }

  return bytes;
}

async function drainResponseWithProgress(response, progressCallback) {
  const total = Number(response.headers.get('content-length')) || 0;

  if (!response.body || typeof response.body.getReader !== 'function') {
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (typeof progressCallback === 'function') {
      progressCallback({ loaded: bytes.byteLength, total: total || bytes.byteLength });
    }
    return bytes.byteLength;
  }

  const reader = response.body.getReader();
  let loaded = 0;
  let lastBucket = -1;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    if (!value || value.length === 0) {
      continue;
    }

    loaded += value.length;
    if (typeof progressCallback === 'function') {
      const effectiveTotal = total || loaded;
      const bucket = effectiveTotal > 0
        ? Math.floor((loaded / effectiveTotal) * 100)
        : -1;
      if (bucket > lastBucket) {
        lastBucket = bucket;
        progressCallback({ loaded, total: effectiveTotal });
      }
    }
  }

  if (typeof progressCallback === 'function') {
    progressCallback({ loaded, total: total || loaded });
  }

  return loaded;
}

async function writeResponseToFsFileWithProgress(
  response,
  fs,
  filePath,
  progressCallback,
  writeOptions = {},
) {
  const total = parsePositiveInteger(writeOptions.totalBytes)
    || inferResponseTotalBytes(response, 0);
  const useBigIntPosition = writeOptions.useBigIntPosition === true;
  const startOffset = parsePositiveInteger(writeOptions.startOffset);
  const preservePartialOnError = writeOptions.preservePartialOnError === true;
  const allowAppend = writeOptions.allowAppend === true || startOffset > 0;
  const appendMode = allowAppend && startOffset > 0;

  if (!appendMode) {
    try {
      if (fs.analyzePath(filePath).exists) {
        fs.unlink(filePath);
      }
    } catch (_) {
      // best-effort replacement of stale temp files
    }
  }

  if (!response.body || typeof response.body.getReader !== 'function') {
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (appendMode) {
      const stream = fs.open(filePath, 'r+');
      try {
        const position = useBigIntPosition ? BigInt(startOffset) : startOffset;
        fs.write(stream, bytes, 0, bytes.length, position);
      } finally {
        fs.close(stream);
      }
    } else {
      fs.writeFile(filePath, bytes);
    }

    const finalLoaded = startOffset + bytes.byteLength;
    if (typeof progressCallback === 'function') {
      progressCallback({ loaded: finalLoaded, total: total || finalLoaded });
    }
    return finalLoaded;
  }

  const reader = response.body.getReader();
  const openMode = appendMode ? 'r+' : 'w';
  const stream = fs.open(filePath, openMode);
  let loaded = 0;
  let lastBucket = -1;
  let writePosition = useBigIntPosition ? BigInt(startOffset) : startOffset;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      if (!value || value.length === 0) {
        continue;
      }

      // Some browsers may reuse the same Uint8Array backing store across reads.
      // Clone each chunk before writing to avoid transient buffer aliasing.
      const chunk = value.slice ? value.slice() : new Uint8Array(value);
      if (useBigIntPosition) {
        fs.write(stream, chunk, 0, chunk.length, writePosition);
        writePosition += BigInt(chunk.length);
      } else {
        fs.write(stream, chunk, 0, chunk.length, writePosition);
        writePosition += chunk.length;
      }
      loaded += chunk.length;

      if (typeof progressCallback === 'function') {
        const effectiveLoaded = startOffset + loaded;
        const effectiveTotal = total || effectiveLoaded;
        const bucket = effectiveTotal > 0
          ? Math.floor((effectiveLoaded / effectiveTotal) * 100)
          : -1;
        if (bucket > lastBucket) {
          lastBucket = bucket;
          progressCallback({ loaded: effectiveLoaded, total: effectiveTotal });
        }
      }
    }
  } catch (error) {
    try {
      if (error && typeof error === 'object') {
        error.llamadartLoadedBytes = startOffset + loaded;
        error.llamadartFilePath = filePath;
      }
    } catch (_) {
      // ignore metadata attachment failures
    }

    try {
      fs.close(stream);
    } catch (_) {
      // ignore close failures during abort/error
    }

    if (!preservePartialOnError) {
      try {
        fs.unlink(filePath);
      } catch (_) {
        // ignore best-effort cleanup failures
      }
    }

    throw error;
  }

  fs.close(stream);

  const finalLoaded = startOffset + loaded;
  if (typeof progressCallback === 'function') {
    progressCallback({ loaded: finalLoaded, total: total || finalLoaded });
  }

  return finalLoaded;
}

function toUint8Array(value) {
  if (!value) {
    return null;
  }

  if (value instanceof Uint8Array) {
    return value;
  }

  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }

  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value);
  }

  if (Array.isArray(value)) {
    return Uint8Array.from(value.map((v) => Number(v) & 0xff));
  }

  return null;
}

function toFloat32Array(value) {
  if (!value) {
    return null;
  }

  if (value instanceof Float32Array) {
    return value;
  }

  if (ArrayBuffer.isView(value)) {
    return new Float32Array(
      value.buffer,
      value.byteOffset,
      Math.floor(value.byteLength / Float32Array.BYTES_PER_ELEMENT),
    );
  }

  if (value instanceof ArrayBuffer) {
    return new Float32Array(value);
  }

  if (Array.isArray(value)) {
    return Float32Array.from(value.map((v) => Number(v) || 0));
  }

  return null;
}

function serializeWorkerError(error) {
  if (!error) {
    return 'Unknown worker error';
  }

  if (typeof error === 'string') {
    return error;
  }

  if (typeof error.message === 'string' && error.message.length > 0) {
    return error.message;
  }

  try {
    return JSON.stringify(error);
  } catch (_) {
    return String(error);
  }
}

const bridgeWorkerModeParam = '__llamadartBridgeWorker';
let bridgeWorkerHostInstalled = false;

function snapshotBridgeState(target) {
  return {
    metadata: target.getModelMetadata(),
    contextSize: target.getContextSize(),
    gpuActive: target.isGpuActive(),
    backendName: target.getBackendName(),
    supportsVision: target.supportsVision(),
    supportsAudio: target.supportsAudio(),
  };
}

function installBridgeWorkerHost() {
  if (bridgeWorkerHostInstalled) {
    return;
  }

  if (typeof self === 'undefined' || typeof self.postMessage !== 'function') {
    throw new Error('Bridge worker host can only run inside a worker context');
  }

  bridgeWorkerHostInstalled = true;
  let bridge = null;

  const postError = (id, error) => {
    let state;
    try {
      state = bridge ? snapshotBridgeState(bridge) : undefined;
    } catch (_) {
      state = undefined;
    }

    self.postMessage({
      type: 'error',
      id,
      message: serializeWorkerError(error),
      state,
    });
  };

  self.onmessage = async (event) => {
    const message = event.data || {};
    const type = message.type;
    const id = message.id ?? 0;

    try {
      if (type === 'init') {
        bridge = new LlamaWebGpuBridge({
          ...(message.config || {}),
          disableWorker: true,
        });
        self.postMessage({ type: 'ready' });
        return;
      }

      if (type !== 'call') {
        return;
      }

      if (!bridge) {
        throw new Error('Bridge worker is not initialized');
      }

      const method = String(message.method || '');
      const args = Array.isArray(message.args) ? message.args : [];

      if (method === 'loadModelFromUrl') {
        const url = args[0];
        const options = (args[1] && typeof args[1] === 'object') ? { ...args[1] } : {};
        options.progressCallback = (progress) => {
          self.postMessage({ type: 'event', id, event: 'progress', payload: progress || {} });
        };

        const value = await bridge.loadModelFromUrl(url, options);
        self.postMessage({ type: 'result', id, value, state: snapshotBridgeState(bridge) });
        return;
      }

      if (method === 'createCompletion') {
        const prompt = args[0];
        const options = (args[1] && typeof args[1] === 'object') ? { ...args[1] } : {};
        delete options.signal;
        options.onToken = (piece, currentText) => {
          self.postMessage({
            type: 'event',
            id,
            event: 'token',
            payload: {
              piece: Array.from(piece || []),
              currentText: String(currentText || ''),
            },
          });
        };

        const value = await bridge.createCompletion(prompt, options);
        self.postMessage({ type: 'result', id, value });
        return;
      }

      if (method === 'loadMultimodalProjector') {
        const value = await bridge.loadMultimodalProjector(args[0]);
        self.postMessage({ type: 'result', id, value, state: snapshotBridgeState(bridge) });
        return;
      }

      if (method === 'unloadMultimodalProjector') {
        const value = await bridge.unloadMultimodalProjector();
        self.postMessage({ type: 'result', id, value, state: snapshotBridgeState(bridge) });
        return;
      }

      if (method === 'dispose') {
        const value = await bridge.dispose();
        self.postMessage({
          type: 'result',
          id,
          value,
          state: {
            metadata: {},
            contextSize: 0,
            gpuActive: false,
            backendName: 'WASM (Prototype bridge)',
            supportsVision: false,
            supportsAudio: false,
          },
        });
        return;
      }

      const value = await bridge[method](...(args || []));
      self.postMessage({ type: 'result', id, value });
    } catch (error) {
      postError(id, error);
    }
  };
}

function shouldAutoBootBridgeWorkerHost() {
  if (typeof WorkerGlobalScope === 'undefined' || !(globalThis instanceof WorkerGlobalScope)) {
    return false;
  }

  try {
    const href = String(globalThis.location?.href || '');
    if (!href) {
      return false;
    }
    const url = new URL(href);
    return url.searchParams.get(bridgeWorkerModeParam) === '1';
  } catch (_) {
    return false;
  }
}

export function enableBridgeWorkerHost() {
  installBridgeWorkerHost();
}

if (shouldAutoBootBridgeWorkerHost()) {
  installBridgeWorkerHost();
}

function createBridgeWorkerSource(moduleUrl) {
  return `import * as workerModule from ${JSON.stringify(moduleUrl)};\nif (workerModule && typeof workerModule.enableBridgeWorkerHost === 'function') { workerModule.enableBridgeWorkerHost(); }\n`;
}

function resolveWorkerEntryUrl(moduleUrl) {
  if (typeof moduleUrl !== 'string' || moduleUrl.length === 0) {
    return null;
  }

  try {
    const base = (typeof window !== 'undefined' && window.location?.href)
      ? window.location.href
      : undefined;
    const url = new URL(moduleUrl, base);
    const path = url.pathname || '';
    const usesDedicatedWorkerEntry = path.endsWith('_worker.js');
    if (!usesDedicatedWorkerEntry) {
      url.searchParams.set(bridgeWorkerModeParam, '1');
    }
    return url.toString();
  } catch (_) {
    return null;
  }
}

function deriveBridgeModuleUrlFromWorkerEntry(moduleUrl) {
  if (typeof moduleUrl !== 'string' || moduleUrl.length === 0) {
    return null;
  }

  try {
    const base = (typeof window !== 'undefined' && window.location?.href)
      ? window.location.href
      : undefined;
    const url = new URL(moduleUrl, base);
    const path = url.pathname || '';
    if (!path.endsWith('_worker.js')) {
      return null;
    }

    url.pathname = path.replace(/_worker\.js$/, '.js');
    url.searchParams.delete(bridgeWorkerModeParam);
    return url.toString();
  } catch (_) {
    return null;
  }
}

class BridgeWorkerProxy {
  constructor({ moduleUrl, config }) {
    this._nextId = 1;
    this._pending = new Map();
    this._workerBlobUrl = null;

    let workerInitError = null;
    const moduleCandidates = [moduleUrl];
    const bridgeModuleFallback = deriveBridgeModuleUrlFromWorkerEntry(moduleUrl);
    if (bridgeModuleFallback && bridgeModuleFallback !== moduleUrl) {
      moduleCandidates.push(bridgeModuleFallback);
    }

    for (const candidate of moduleCandidates) {
      if (this._worker) {
        break;
      }

      const directWorkerUrl = resolveWorkerEntryUrl(candidate);
      if (!directWorkerUrl) {
        continue;
      }

      try {
        this._worker = new Worker(directWorkerUrl, { type: 'module' });
      } catch (error) {
        workerInitError = error;
      }
    }

    for (const candidate of moduleCandidates) {
      if (this._worker) {
        break;
      }

      const source = createBridgeWorkerSource(candidate);
      this._workerBlobUrl = URL.createObjectURL(
        new Blob([source], { type: 'text/javascript' }),
      );

      try {
        this._worker = new Worker(this._workerBlobUrl, { type: 'module' });
      } catch (error) {
        workerInitError = error;
        URL.revokeObjectURL(this._workerBlobUrl);
        this._workerBlobUrl = null;
      }
    }

    if (!this._worker) {
      throw workerInitError || new Error('Failed to initialize bridge worker');
    }

    this._ready = new Promise((resolve, reject) => {
      this._readyResolve = resolve;
      this._readyReject = reject;
    });

    this._worker.onmessage = (event) => {
      const message = event.data || {};
      const type = message.type;
      if (type === 'ready') {
        this._readyResolve?.();
        return;
      }

      const id = Number(message.id || 0);
      const pending = this._pending.get(id);
      if (!pending) {
        return;
      }

      if (type === 'event') {
        pending.onEvent?.(message);
        return;
      }

      this._pending.delete(id);
      if (type === 'error') {
        const workerError = new Error(String(message.message || 'Worker request failed'));
        if (message.state && typeof message.state === 'object') {
          workerError.state = message.state;
        }
        pending.reject(workerError);
        return;
      }

      pending.resolve(message);
    };

    this._worker.onerror = (event) => {
      const message = event?.message || 'Bridge worker crashed';
      const error = new Error(String(message));

      this._readyReject?.(error);

      for (const pending of this._pending.values()) {
        pending.reject(error);
      }
      this._pending.clear();
    };

    this._worker.postMessage({ type: 'init', config });
  }

  async call(method, args, onEvent) {
    await this._ready;
    const id = this._nextId++;

    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject, onEvent });
      this._worker.postMessage({ type: 'call', id, method, args });
    });
  }

  async dispose() {
    try {
      await this.call('dispose', []);
    } catch (_) {
      // best-effort disposal
    }

    for (const pending of this._pending.values()) {
      pending.reject(new Error('Bridge worker disposed'));
    }
    this._pending.clear();

    this._worker.terminate();
    if (this._workerBlobUrl) {
      URL.revokeObjectURL(this._workerBlobUrl);
      this._workerBlobUrl = null;
    }
  }
}

class LlamaWebGpuBridgeRuntime {
  constructor(config = {}) {
    this._config = config;
    this._core = null;
    this._backendLabels = [];
    this._gpuActive = false;
    this._modelPath = null;
    this._modelPaths = [];
    this._modelBytes = 0;
    this._mmProjPath = null;
    this._mmSupportsVision = false;
    this._mmSupportsAudio = false;
    this._mediaFileCounter = 0;
    this._stagedMediaPaths = [];
    this._nCtx = 4096;
    this._abortRequested = false;
    this._threads = Number(config.threads) > 0
      ? Number(config.threads)
      : Math.max(1, Math.min(8, Number(globalThis.navigator?.hardwareConcurrency) || 4));
    this._nGpuLayers = Number.isFinite(config.nGpuLayers)
      ? Number(config.nGpuLayers)
      : -1;
    this._runtimeNotes = [];
    this._isSafari = isSafariUserAgent(this._config.userAgent ?? globalThis.navigator?.userAgent ?? '');
    this._coreVariant = 'uninitialized';
    this._preferMemory64 = this._config.preferMemory64 !== false;
    this._modelSource = 'network';
    this._modelCacheState = 'disabled';
    this._modelCacheName = defaultModelCacheName;
    this._remoteFetchThresholdBytes = Number(config.remoteFetchThresholdBytes) > 0
      ? Number(config.remoteFetchThresholdBytes)
      : 1900 * 1024 * 1024;
    this._remoteFetchChunkBytes = Number(config.remoteFetchChunkBytes) > 0
      ? Number(config.remoteFetchChunkBytes)
      : 16 * 1024 * 1024;
    this._activeTransferAbortController = null;
    this._lastCoreErrorText = '';
    this._lastCoreErrorHint = '';
    this._logLevel = Number.isFinite(config.logLevel)
      ? Math.max(0, Math.min(4, Math.trunc(config.logLevel)))
      : 2;
  }

  static supportsSafariAdaptiveGpu = true;

  _loggerFor(level) {
    const logger = this._config?.logger;
    const fallback = (typeof console !== 'undefined') ? console : null;

    if (logger && typeof logger[level] === 'function') {
      return logger[level].bind(logger);
    }

    if (!fallback) {
      return () => {};
    }

    if (typeof fallback[level] === 'function') {
      return fallback[level].bind(fallback);
    }

    if (typeof fallback.log === 'function') {
      return fallback.log.bind(fallback);
    }

    return () => {};
  }

  _emitLogger(level, message) {
    try {
      this._loggerFor(level)(message);
    } catch (_) {
      // Logger callbacks are best-effort only.
    }
  }

  _applyCoreLogLevel() {
    if (!this._core) {
      return;
    }

    try {
      this._core.ccall(
        'llamadart_webgpu_set_log_level',
        null,
        ['number'],
        [this._logLevel],
      );
    } catch (_) {
      // Older core builds may not expose log-level setter.
    }
  }

  _coreErrorMessage(prefix, fallbackCode = 0) {
    try {
      const err = this._core?.ccall('llamadart_webgpu_last_error', 'string', [], []);
      if (err) {
        return `${prefix}: ${err}`;
      }
    } catch (_) {
      // Ignore nested error retrieval failures.
    }
    return `${prefix} (code=${fallbackCode})`;
  }

  _resolveCacheName(options = {}) {
    if (typeof options.cacheName === 'string' && options.cacheName.trim().length > 0) {
      return options.cacheName.trim();
    }

    if (typeof this._config.cacheName === 'string' && this._config.cacheName.trim().length > 0) {
      return this._config.cacheName.trim();
    }

    return defaultModelCacheName;
  }

  _isWorkerRuntime() {
    return typeof WorkerGlobalScope !== 'undefined'
      && globalThis instanceof WorkerGlobalScope;
  }

  _resolveRemoteFetchThresholdBytes(options = {}) {
    const candidate = Number(options.remoteFetchThresholdBytes);
    if (Number.isFinite(candidate) && candidate > 0) {
      return Math.trunc(candidate);
    }
    return Math.trunc(this._remoteFetchThresholdBytes);
  }

  _resolveRemoteFetchChunkBytes(options = {}) {
    const candidate = Number(options.remoteFetchChunkBytes);
    if (Number.isFinite(candidate) && candidate > 0) {
      return Math.max(16 * 1024, Math.trunc(candidate));
    }
    return Math.max(16 * 1024, Math.trunc(this._remoteFetchChunkBytes));
  }

  _canUseRemoteFetchBackend(options = {}) {
    if (options.forceRemoteFetchBackend === false) {
      return false;
    }

    if (!this._isWorkerRuntime()) {
      return false;
    }

    if (options.forceRemoteFetchBackend === true) {
      return true;
    }

    return this._config.allowAutoRemoteFetchBackend === true;
  }

  async _tryHeadContentLength(url) {
    function parseSizeFromHeaders(headers) {
      if (!headers) {
        return 0;
      }

      const length = Number(headers.get('content-length')) || 0;
      if (length > 0) {
        return length;
      }

      const linkedSize = Number(headers.get('x-linked-size')) || 0;
      if (linkedSize > 0) {
        return linkedSize;
      }

      const contentRange = String(headers.get('content-range') || '');
      const slash = contentRange.lastIndexOf('/');
      if (slash >= 0 && slash + 1 < contentRange.length) {
        const total = Number(contentRange.slice(slash + 1)) || 0;
        if (total > 0) {
          return total;
        }
      }

      return 0;
    }

    try {
      const response = await fetch(url, {
        method: 'HEAD',
        cache: 'no-store',
      });
      if (!response.ok) {
        throw new Error('HEAD request failed');
      }

      const size = parseSizeFromHeaders(response.headers);
      if (size > 0) {
        return size;
      }
    } catch (_) {
      // ignore best-effort HEAD failures
    }

    try {
      const probe = await fetch(url, {
        headers: {
          Range: 'bytes=0-0',
        },
        cache: 'no-store',
      });

      if (!(probe.ok || probe.status === 206)) {
        return null;
      }

      const size = parseSizeFromHeaders(probe.headers);
      if (size > 0) {
        this._runtimeNotes.push('model_fetch_size_probe');
        return size;
      }
    } catch (_) {
      // ignore best-effort range probe failures
    }

    return null;
  }

  async _resolveRemoteFetchUrl(url) {
    function parseSizeFromHeaders(headers) {
      if (!headers) {
        return 0;
      }

      const contentRange = String(headers.get('content-range') || '');
      const slash = contentRange.lastIndexOf('/');
      if (slash >= 0 && slash + 1 < contentRange.length) {
        const total = Number(contentRange.slice(slash + 1)) || 0;
        if (total > 0) {
          return total;
        }
      }

      const linkedSize = Number(headers.get('x-linked-size')) || 0;
      if (linkedSize > 0) {
        return linkedSize;
      }

      const length = Number(headers.get('content-length')) || 0;
      if (length > 0) {
        return length;
      }

      return 0;
    }

    try {
      const probe = await fetch(url, {
        headers: {
          Range: 'bytes=0-0',
        },
        cache: 'no-store',
      });

      if (!(probe.ok || probe.status === 206)) {
        return null;
      }

      const resolvedUrl =
        typeof probe.url === 'string' && probe.url.length > 0
          ? probe.url
          : null;
      const sizeBytes = parseSizeFromHeaders(probe.headers);

      return {
        resolvedUrl,
        sizeBytes: sizeBytes > 0 ? sizeBytes : null,
      };
    } catch (_) {
      return null;
    }
  }

  async _tryLoadModelFromRemoteFetchBackend(core, url, options = {}) {
    if (!this._canUseRemoteFetchBackend(options)) {
      return { loaded: false, sizeBytes: null };
    }

    const thresholdBytes = this._resolveRemoteFetchThresholdBytes(options);
    const chunkBytes = this._resolveRemoteFetchChunkBytes(options);
    const forceRemote = options.forceRemoteFetchBackend === true;

    let sizeBytes = Number(options.modelBytesHint);
    if (!Number.isFinite(sizeBytes) || sizeBytes <= 0) {
      sizeBytes = await this._tryHeadContentLength(url);
    }

    if (!forceRemote) {
      if (Number.isFinite(sizeBytes) && sizeBytes > 0 && sizeBytes < thresholdBytes) {
        this._runtimeNotes.push('model_fetch_backend_skipped_small');
        return { loaded: false, sizeBytes: sizeBytes };
      }

      if (!Number.isFinite(sizeBytes) || sizeBytes <= 0) {
        this._runtimeNotes.push('model_fetch_backend_size_unknown');
        this._runtimeNotes.push('model_fetch_backend_unknown_size_attempt');
      }
    }

    this._runtimeNotes.push('model_fetch_backend_attempt');
    this._runtimeNotes.push(`model_fetch_chunk:${chunkBytes}`);
    this._lastCoreErrorHint = '';

    let remoteFetchUrl = url;
    const resolvedProbe = await this._resolveRemoteFetchUrl(url);
    if (resolvedProbe?.resolvedUrl) {
      remoteFetchUrl = resolvedProbe.resolvedUrl;
      if (remoteFetchUrl !== url) {
        this._runtimeNotes.push('model_fetch_backend_resolved_url');
      }
    }
    if ((!Number.isFinite(sizeBytes) || sizeBytes <= 0) &&
        Number.isFinite(resolvedProbe?.sizeBytes) &&
        resolvedProbe.sizeBytes > 0) {
      sizeBytes = resolvedProbe.sizeBytes;
    }

    try {
      globalThis.__llamadartFetchBackendLastError = null;
    } catch (_) {
      // ignore debug-state reset failures
    }

    if (typeof options.progressCallback === 'function') {
      options.progressCallback({ loaded: 0, total: Number.isFinite(sizeBytes) ? sizeBytes : 0 });
    }

    try {
      const rc = Number(
        await core.ccall(
          'llamadart_webgpu_load_model_from_url',
          'number',
          ['string', 'number', 'number', 'number', 'number'],
          [remoteFetchUrl, this._nCtx, this._threads, this._nGpuLayers, chunkBytes],
          { async: true },
        ),
      );

      if (rc !== 0) {
        throw new Error(this._coreErrorMessage('Fetch-backed model load failed', rc));
      }

      this._modelSource = 'network-fetch';
      this._modelPath = null;
      this._modelBytes = Number.isFinite(sizeBytes) && sizeBytes > 0 ? Math.trunc(sizeBytes) : 1;
      this._runtimeNotes.push('model_source_fetch_backend');

      if (typeof options.progressCallback === 'function') {
        const resolved = Number.isFinite(sizeBytes) && sizeBytes > 0 ? sizeBytes : 1;
        options.progressCallback({ loaded: resolved, total: resolved });
      }

      return {
        loaded: true,
        sizeBytes: Number.isFinite(sizeBytes) ? sizeBytes : null,
      };
    } catch (error) {
      const text = String(error || '').toLowerCase();

      if (text.includes('aborted(native code called abort())')) {
        try {
          const stderrText = String(
            this._lastCoreErrorHint || this._lastCoreErrorText || '',
          ).trim();
          if (stderrText.length > 0) {
            const token = stderrText
              .slice(0, 120)
              .replace(/[\s;=]+/g, '_')
              .replace(/[^a-zA-Z0-9._:-]/g, '');
            if (token.length > 0) {
              this._runtimeNotes.push(`model_core_stderr:${token}`);
            }
          }
        } catch (_) {
          // ignore stderr capture failures on abort path
        }

        try {
          const fetchError = String(globalThis.__llamadartFetchBackendLastError || '').trim();
          if (fetchError.length > 0) {
            const token = fetchError
              .slice(0, 120)
              .replace(/[\s;=]+/g, '_')
              .replace(/[^a-zA-Z0-9._:-]/g, '');
            if (token.length > 0) {
              this._runtimeNotes.push(`model_fetch_js_error:${token}`);
            }
          }
        } catch (_) {
          // ignore fetch-backend debug-state probe failures
        }

        try {
          const stats = globalThis.__llamadartFetchBackendStats;
          if (stats && typeof stats === 'object') {
            const reads = Number(stats.reads) || 0;
            const getSize = Number(stats.getSize) || 0;
            const ranges = Number(stats.ranges) || 0;
            const fallbacks = Number(stats.wholeFileFallbacks) || 0;
            const errors = Number(stats.errors) || 0;
            this._runtimeNotes.push(
              `model_fetch_stats:r${reads}_s${getSize}_q${ranges}_f${fallbacks}_e${errors}`,
            );
          }
        } catch (_) {
          // ignore fetch stats probe failures
        }

        try {
          const coreError = String(
            this._core?.ccall('llamadart_webgpu_last_error', 'string', [], []) || '',
          ).trim();
          if (coreError.length > 0) {
            const token = coreError
              .slice(0, 120)
              .replace(/[\s;=]+/g, '_')
              .replace(/[^a-zA-Z0-9._:-]/g, '');
            if (token.length > 0) {
              this._runtimeNotes.push(`model_fetch_core_error:${token}`);
            }
          }
        } catch (_) {
          // ignore last-error probing failures on abort path
        }

        this._runtimeNotes.push('model_fetch_backend_abort');
        throw error;
      }

      if (text.includes('fetch-backed model load failed')) {
        this._runtimeNotes.push('model_fetch_backend_failed');
        return {
          loaded: false,
          sizeBytes: Number.isFinite(sizeBytes) ? sizeBytes : null,
        };
      }

      if (
        text.includes('load_model_from_url')
        && (text.includes('not found')
          || text.includes('undefined symbol')
          || text.includes('missing function'))
      ) {
        this._runtimeNotes.push('model_fetch_backend_unavailable');
        return {
          loaded: false,
          sizeBytes: Number.isFinite(sizeBytes) ? sizeBytes : null,
        };
      }

      if (text.includes('worker-thread bridge runtime')) {
        this._runtimeNotes.push('model_fetch_backend_requires_worker');
        return {
          loaded: false,
          sizeBytes: Number.isFinite(sizeBytes) ? sizeBytes : null,
        };
      }

      throw error;
    }
  }

  _beginTransferAbortController() {
    if (typeof AbortController !== 'function') {
      this._activeTransferAbortController = null;
      return null;
    }

    if (this._activeTransferAbortController) {
      try {
        this._activeTransferAbortController.abort();
      } catch (_) {
        // ignore abort failures on stale controllers
      }
    }

    const controller = new AbortController();
    this._activeTransferAbortController = controller;
    return controller;
  }

  _clearTransferAbortController(controller) {
    if (this._activeTransferAbortController === controller) {
      this._activeTransferAbortController = null;
    }
  }

  _deleteFsFile(path) {
    if (!this._core || typeof path !== 'string' || path.length === 0) {
      return false;
    }

    try {
      if (this._core.FS.analyzePath(path).exists) {
        this._core.FS.unlink(path);
        return true;
      }
    } catch (_) {
      // ignore best-effort cleanup failures
    }

    return false;
  }

  _releaseModelFiles() {
    const paths = Array.isArray(this._modelPaths) && this._modelPaths.length > 0
      ? [...this._modelPaths]
      : (this._modelPath ? [this._modelPath] : []);

    let removed = 0;
    for (const path of paths) {
      if (this._deleteFsFile(path)) {
        removed += 1;
      }
    }

    this._modelPaths = [];
    return removed;
  }

  async _getCachedModelResponse(url, options = {}) {
    let useCache = options.useCache !== false;
    const forceRefresh = options.force === true;
    const requireReadableStream = options.requireReadableStream === true;
    const requestHeaders =
      options.requestHeaders && typeof options.requestHeaders === 'object'
        ? options.requestHeaders
        : null;

    if (requestHeaders && Object.keys(requestHeaders).length > 0) {
      useCache = false;
    }

    const fetchOptions = {
      ...(options.signal ? { signal: options.signal } : {}),
      ...(requestHeaders ? { headers: requestHeaders } : {}),
    };
    this._modelSource = 'network';
    this._modelCacheState = useCache ? 'unavailable' : 'disabled';
    this._modelCacheName = this._resolveCacheName(options);

    if (!useCache) {
      const response = await fetch(url, {
        cache: 'no-store',
        ...fetchOptions,
      });
      this._modelCacheState = 'disabled';
      if (requireReadableStream) {
        this._runtimeNotes.push(
          hasReadableResponseStream(response)
            ? 'model_network_stream'
            : 'model_network_no_stream',
        );
      }
      return response;
    }

    if (!globalThis.caches || typeof globalThis.caches.open !== 'function') {
      this._modelCacheState = 'unavailable';
      const response = await fetch(url, fetchOptions);
      if (requireReadableStream) {
        this._runtimeNotes.push(
          hasReadableResponseStream(response)
            ? 'model_network_stream'
            : 'model_network_no_stream',
        );
      }
      return response;
    }

    const cacheKey = normalizeAbsoluteUrl(url);

    try {
      const cache = await globalThis.caches.open(this._modelCacheName);
      const cached = forceRefresh ? null : await cache.match(cacheKey);
      if (cached) {
        const cacheHasStream = hasReadableResponseStream(cached);
        if (!requireReadableStream || cacheHasStream) {
          this._modelSource = 'cache';
          this._modelCacheState = 'hit';
          this._runtimeNotes.push('model_cache_hit');
          if (requireReadableStream) {
            this._runtimeNotes.push('model_cache_stream');
          }
          return cached;
        }

        this._runtimeNotes.push('model_cache_hit_no_stream');
        this._modelSource = 'network';
        this._modelCacheState = 'refresh';
        const refreshed = await fetch(url, {
          cache: 'no-store',
          ...fetchOptions,
        });

        if (refreshed.ok) {
          try {
            await cache.put(cacheKey, refreshed.clone());
            this._modelCacheState = 'stored';
            this._runtimeNotes.push('model_cache_stored');
          } catch (_) {
            this._modelCacheState = 'store_failed';
            this._runtimeNotes.push('model_cache_store_failed');
          }
        }

        this._runtimeNotes.push(
          hasReadableResponseStream(refreshed)
            ? 'model_network_stream'
            : 'model_network_no_stream',
        );
        return refreshed;
      }

      this._modelCacheState = forceRefresh ? 'refresh' : 'miss';
      const response = await fetch(url, fetchOptions);

      if (response.ok) {
        try {
          await cache.put(cacheKey, response.clone());
          this._modelCacheState = 'stored';
          this._runtimeNotes.push('model_cache_stored');
        } catch (_) {
          this._modelCacheState = 'store_failed';
          this._runtimeNotes.push('model_cache_store_failed');
        }
      }

      if (requireReadableStream) {
        this._runtimeNotes.push(
          hasReadableResponseStream(response)
            ? 'model_network_stream'
            : 'model_network_no_stream',
        );
      }

      return response;
    } catch (_) {
      this._modelCacheState = 'error';
      this._runtimeNotes.push('model_cache_error');
      const response = await fetch(url, fetchOptions);
      if (requireReadableStream) {
        this._runtimeNotes.push(
          hasReadableResponseStream(response)
            ? 'model_network_stream'
            : 'model_network_no_stream',
        );
      }
      return response;
    }
  }

  async prefetchModelToCache(url, options = {}) {
    const useCache = options.useCache !== false;
    this._modelSource = 'network';
    this._modelCacheState = useCache ? 'unavailable' : 'disabled';
    this._modelCacheName = this._resolveCacheName(options);

    const progressCallback = typeof options.progressCallback === 'function'
      ? options.progressCallback
      : null;

    const modelUrls = expandModelShardUrls(url);
    if (modelUrls.length === 0) {
      throw new Error('Model URL is empty.');
    }

    if (modelUrls.length > 1) {
      this._runtimeNotes.push(`model_split_cache_prefetch:${modelUrls.length}`);
    }

    const shardLoaded = new Array(modelUrls.length).fill(0);
    const shardTotals = new Array(modelUrls.length).fill(0);
    const emitAggregateProgress = () => {
      if (!progressCallback) {
        return;
      }

      const loaded = sumProgressValues(shardLoaded);
      const total = sumProgressValues(shardTotals);
      progressCallback({
        loaded,
        total: total > 0 ? total : loaded,
      });
    };

    const controller = this._beginTransferAbortController();

    try {
      const fetchOptions = {
        cache: 'no-store',
        ...(controller?.signal ? { signal: controller.signal } : {}),
      };

      const cache = (useCache && globalThis.caches && typeof globalThis.caches.open === 'function')
        ? await globalThis.caches.open(this._modelCacheName)
        : null;

      if (!useCache || !cache) {
        this._modelCacheState = useCache ? 'unavailable' : 'disabled';
      }

      for (let shardIndex = 0; shardIndex < modelUrls.length; shardIndex += 1) {
        const shardUrl = modelUrls[shardIndex];
        const cacheKey = normalizeAbsoluteUrl(shardUrl);
        let response;
        let headerTotal = 0;

        if (cache) {
          const cached = options.force === true ? null : await cache.match(cacheKey);
          if (cached) {
            this._modelSource = 'cache';
            this._modelCacheState = 'hit';
            this._runtimeNotes.push('model_cache_hit');

            headerTotal = Number(cached.headers.get('content-length')) || 0;
            const resolved = headerTotal > 0 ? headerTotal : 1;
            shardLoaded[shardIndex] = resolved;
            shardTotals[shardIndex] = resolved;
            emitAggregateProgress();
            continue;
          }

          this._modelSource = 'network';
          this._modelCacheState = options.force === true ? 'refresh' : 'miss';
        }

        response = await fetch(shardUrl, fetchOptions);
        if (!response.ok) {
          throw new Error(
            `Failed to prefetch model shard: ${response.status} ${response.statusText}`,
          );
        }

        headerTotal = Number(response.headers.get('content-length')) || 0;
        if (headerTotal > 0) {
          shardTotals[shardIndex] = headerTotal;
        }

        const putPromise = cache ? cache.put(cacheKey, response.clone()) : null;

        await drainResponseWithProgress(
          response,
          progressCallback
              ? (progress) => {
                  const loaded = Number(progress?.loaded) || 0;
                  const total = Number(progress?.total) || 0;
                  shardLoaded[shardIndex] = loaded;
                  if (total > 0) {
                    shardTotals[shardIndex] = total;
                  }
                  emitAggregateProgress();
                }
              : null,
        );

        const finalLoaded = shardLoaded[shardIndex] > 0
          ? shardLoaded[shardIndex]
          : (shardTotals[shardIndex] > 0 ? shardTotals[shardIndex] : headerTotal);
        shardLoaded[shardIndex] = finalLoaded;
        if (finalLoaded > 0 && shardTotals[shardIndex] < finalLoaded) {
          shardTotals[shardIndex] = finalLoaded;
        }
        emitAggregateProgress();

        if (putPromise) {
          try {
            await putPromise;
            this._modelCacheState = 'stored';
            this._runtimeNotes.push('model_cache_stored');
          } catch (_) {
            this._modelCacheState = 'store_failed';
            this._runtimeNotes.push('model_cache_store_failed');
            throw new Error('Failed to store prefetched model in browser cache.');
          }
        }
      }

      if (modelUrls.length > 1) {
        this._runtimeNotes.push('model_split_cache_prefetched');
      }
      return 1;
    } catch (error) {
      this._modelCacheState = 'error';

      const text = String(error || '').toLowerCase();
      if (text.includes('abort')) {
        this._runtimeNotes.push('model_cache_prefetch_aborted');
      } else if (text.includes('quota') || text.includes('storage')) {
        this._runtimeNotes.push('model_cache_quota_exceeded');
      } else {
        this._runtimeNotes.push('model_cache_error');
      }

      throw error;
    } finally {
      this._clearTransferAbortController(controller);
    }
  }

  async evictModelFromCache(url, options = {}) {
    this._modelCacheName = this._resolveCacheName(options);

    const modelUrls = expandModelShardUrls(url);
    if (modelUrls.length === 0) {
      return false;
    }

    if (!globalThis.caches || typeof globalThis.caches.open !== 'function') {
      this._modelCacheState = 'unavailable';
      return false;
    }

    try {
      const cache = await globalThis.caches.open(this._modelCacheName);
      let removedCount = 0;
      for (const modelUrl of modelUrls) {
        const cacheKey = normalizeAbsoluteUrl(modelUrl);
        const removed = await cache.delete(cacheKey);
        if (removed) {
          removedCount += 1;
        }
      }

      const removedAny = removedCount > 0;
      this._modelCacheState = removedAny ? 'evicted' : 'miss';
      if (removedAny) {
        this._runtimeNotes.push('model_cache_evicted');
      }
      if (modelUrls.length > 1) {
        this._runtimeNotes.push(`model_split_cache_evicted:${removedCount}/${modelUrls.length}`);
      }
      return removedAny;
    } catch (_) {
      this._modelCacheState = 'error';
      this._runtimeNotes.push('model_cache_error');
      return false;
    }
  }

  async _ensureCore() {
    if (this._core) {
      this._applyCoreLogLevel();
      return this._core;
    }

    const candidates = [];
    if (this._config.coreModuleFactory) {
      candidates.push({
        variant: 'custom',
        factoryPromise: Promise.resolve(this._config.coreModuleFactory),
        wasmUrl: this._config.wasmUrl,
      });
    } else {
      if (
        this._preferMemory64
        && typeof this._config.coreModuleUrlMem64 === 'string'
        && this._config.coreModuleUrlMem64.length > 0
      ) {
        candidates.push({
          variant: 'wasm64',
          factoryPromise: importCoreFactory(this._config.coreModuleUrlMem64),
          wasmUrl: this._config.wasmUrlMem64 || this._config.wasmUrl,
        });
      }

      candidates.push({
        variant: 'wasm32',
        factoryPromise: importCoreFactory(this._config.coreModuleUrl ?? './llama_webgpu_core.js'),
        wasmUrl: this._config.wasmUrl,
      });
    }

    let lastError = null;
    for (const candidate of candidates) {
      if (candidate.variant === 'wasm64') {
        this._runtimeNotes.push('core_mem64_attempt');
      }

      try {
        const moduleFactory = await candidate.factoryPromise;
        this._core = await moduleFactory({
          locateFile: (path, prefix) => {
            if (path.endsWith('.wasm') && candidate.wasmUrl) {
              return candidate.wasmUrl;
            }
            return `${prefix}${path}`;
          },
          print: (msg) => {
            this._emitLogger('log', msg);
          },
          printErr: (msg) => {
            const text = String(msg ?? '');
            this._lastCoreErrorText = text;
            const trimmed = text.trim();
            if (trimmed.length > 0) {
              const loweredTrimmed = trimmed.toLowerCase();
              const isGenericAbort =
                loweredTrimmed === 'aborted(native code called abort())'
                || loweredTrimmed === 'native code called abort()'
                || loweredTrimmed === 'aborted';
              if (!isGenericAbort) {
                this._lastCoreErrorHint = trimmed;
              }
            }
            const lowered = text.toLowerCase();
            if (lowered.startsWith('warning')) {
              this._emitLogger('warn', text);
              return;
            }
            this._emitLogger('error', text);
          },
          onAbort: (reason) => {
            const text = String(reason ?? '').trim();
            if (text.length > 0) {
              const token = text
                .slice(0, 120)
                .replace(/[\s;=]+/g, '_')
                .replace(/[^a-zA-Z0-9._:-]/g, '');
              if (token.length > 0) {
                this._runtimeNotes.push(`core_abort:${token}`);
              }
              this._emitLogger('error', `core abort: ${text}`);
            } else {
              this._runtimeNotes.push('core_abort');
              this._emitLogger('error', 'core abort');
            }
          },
        });

        this._coreVariant = candidate.variant === 'wasm64' ? 'wasm64' : 'wasm32';
        if (candidate.variant === 'wasm64') {
          this._runtimeNotes.push('core_mem64_active');
        } else if (candidate.variant === 'wasm32') {
          this._runtimeNotes.push('core_wasm32_active');
        }

        break;
      } catch (error) {
        lastError = error;
        if (candidate.variant === 'wasm64') {
          this._runtimeNotes.push('core_mem64_unavailable');
          continue;
        }
        throw error;
      }
    }

    if (!this._core) {
      throw lastError || new Error('Failed to initialize bridge core module');
    }

    this._applyCoreLogLevel();

    return this._core;
  }

  async _probeBackends() {
    try {
      const core = await this._ensureCore();
      const probeResult = Number(
        await core.ccall('llamadart_webgpu_probe', 'number', [], [], { async: true }),
      );
      const json = core.ccall('llamadart_webgpu_backends_json', 'string', [], []);

      let parsed = [];
      try {
        parsed = JSON.parse(json || '[]');
      } catch (_) {
        parsed = [];
      }

      this._backendLabels = Array.isArray(parsed)
        ? parsed.map((v) => String(v))
        : [];
      this._gpuActive = probeResult === 1;
    } catch (_err) {
      this._backendLabels = [];
      this._gpuActive = false;
    }

    return this._gpuActive;
  }

  async loadModelFromUrl(url, options = {}) {
    this._abortRequested = false;
    this._runtimeNotes = [];
    await this._probeBackends();

    const core = await this._ensureCore();

    this._nCtx = Number(options.nCtx) > 0 ? Number(options.nCtx) : this._nCtx;

    const requestedThreads = Number(options.nThreads);
    if (Number.isFinite(requestedThreads) && requestedThreads > 0) {
      this._threads = Math.trunc(requestedThreads);
    }

    const requestedGpuLayers = Number(options.nGpuLayers);
    if (Number.isFinite(requestedGpuLayers)) {
      this._nGpuLayers = Math.trunc(requestedGpuLayers);
    }

    if (!isCrossOriginIsolatedRuntime() && this._threads > 1) {
      this._threads = 1;
      this._runtimeNotes.push('threads_capped_no_coi');
    }

    if (this._isSafari && this._nGpuLayers > 0) {
      const requestedSafariMaxLayers = Number(options.safariMaxGpuLayers);
      const safariMaxGpuLayers = Number.isFinite(requestedSafariMaxLayers)
        ? Math.max(1, Math.trunc(requestedSafariMaxLayers))
        : 1;

      if (this._nGpuLayers > safariMaxGpuLayers) {
        this._nGpuLayers = safariMaxGpuLayers;
        this._runtimeNotes.push(`safari_gpu_layers_capped:${safariMaxGpuLayers}`);
      }
    }

    const modelUrls = expandModelShardUrls(url);
    if (modelUrls.length === 0) {
      throw new Error('Model URL is empty.');
    }
    if (modelUrls.length > 1) {
      this._runtimeNotes.push(`model_split_detected:${modelUrls.length}`);
    }

    let loadedViaRemoteFetch = false;
    if (modelUrls.length === 1) {
      const remoteResult = await this._tryLoadModelFromRemoteFetchBackend(
        core,
        modelUrls[0],
        options,
      );
      loadedViaRemoteFetch = remoteResult.loaded === true;
    } else {
      this._runtimeNotes.push('model_fetch_backend_skipped_split');
    }

    if (!loadedViaRemoteFetch) {
      if (!core.FS.analyzePath('/models').exists) {
        core.FS.mkdir('/models');
      }

      const progressCallback = typeof options.progressCallback === 'function'
        ? options.progressCallback
        : null;
      const shardLoaded = new Array(modelUrls.length).fill(0);
      const shardTotals = new Array(modelUrls.length).fill(0);
      const emitAggregateProgress = () => {
        if (!progressCallback) {
          return;
        }

        const loaded = sumProgressValues(shardLoaded);
        const total = sumProgressValues(shardTotals);
        progressCallback({
          loaded,
          total: total > 0 ? total : loaded,
        });
      };

      const modelPaths = [];
      let totalModelBytes = 0;
      const maxStreamResumeRetries = Number.isFinite(options.streamResumeRetries)
        ? Math.max(0, Math.trunc(options.streamResumeRetries))
        : 8;

      try {
        for (let shardIndex = 0; shardIndex < modelUrls.length; shardIndex += 1) {
          const shardUrl = modelUrls[shardIndex];
          const fileName = basenameFromUrl(shardUrl);
          const modelPath = `/models/${fileName}`;
          modelPaths.push(modelPath);

          let shardBytes = 0;
          let resumeOffset = 0;
          let resumeAttempt = 0;
          let activeShardUrl = shardUrl;
          let knownTotalBytes = 0;

          while (true) {
            const requestHeaders = resumeOffset > 0
              ? { Range: `bytes=${resumeOffset}-` }
              : null;

            const response = await this._getCachedModelResponse(activeShardUrl, {
              ...options,
              requireReadableStream: true,
              useCache: requestHeaders ? false : options.useCache,
              force: requestHeaders ? true : options.force,
              requestHeaders,
            });
            if (!response.ok) {
              throw new Error(
                `Failed to fetch model shard: ${response.status} ${response.statusText}`,
              );
            }

            const responseUrl =
              typeof response.url === 'string' && response.url.length > 0
                ? response.url
                : activeShardUrl;
            if (responseUrl !== activeShardUrl) {
              activeShardUrl = responseUrl;
              if (resumeOffset > 0) {
                this._runtimeNotes.push('model_stream_resume_redirect');
              }
            }

            if (resumeOffset > 0 && response.status !== 206) {
              this._runtimeNotes.push(`model_stream_resume_status:${response.status}`);
              throw new Error(
                `Range resume not honored for model shard: ${response.status} ${response.statusText}`,
              );
            }

            if (!hasReadableResponseStream(response)) {
              this._runtimeNotes.push('model_response_nostream');
              const declaredBytes = inferResponseTotalBytes(response, 0);
              const declaredText = declaredBytes > 0 ? `${declaredBytes} bytes` : 'unknown size';
              throw new Error(
                `Model response did not expose a readable stream (${declaredText}).`,
              );
            }
            this._runtimeNotes.push('model_response_stream');

            const responseTotal = inferResponseTotalBytes(response, knownTotalBytes);
            if (responseTotal > 0) {
              knownTotalBytes = responseTotal;
              shardTotals[shardIndex] = responseTotal;
            }

            try {
              shardBytes = await writeResponseToFsFileWithProgress(
                response,
                core.FS,
                modelPath,
                progressCallback
                  ? (progress) => {
                      const loaded = Number(progress?.loaded) || 0;
                      const total = Number(progress?.total) || 0;
                      shardLoaded[shardIndex] = loaded;
                      if (total > 0) {
                        shardTotals[shardIndex] = total;
                      }
                      emitAggregateProgress();
                    }
                  : null,
                {
                  useBigIntPosition: this._coreVariant === 'wasm64',
                  startOffset: resumeOffset,
                  allowAppend: resumeOffset > 0,
                  preservePartialOnError: true,
                  totalBytes: knownTotalBytes,
                },
              );
              break;
            } catch (error) {
              const text = String(error || '').toLowerCase();
              const loadedBytes = Number(error?.llamadartLoadedBytes);
              if (Number.isFinite(loadedBytes) && loadedBytes >= 0) {
                const normalizedLoaded = Math.trunc(loadedBytes);
                this._runtimeNotes.push(`model_fs_write_loaded:${normalizedLoaded}`);
                if (normalizedLoaded > resumeOffset) {
                  resumeOffset = normalizedLoaded;
                  shardLoaded[shardIndex] = normalizedLoaded;
                  if (knownTotalBytes > 0 && shardTotals[shardIndex] < knownTotalBytes) {
                    shardTotals[shardIndex] = knownTotalBytes;
                  }
                  emitAggregateProgress();
                }
              }

              const shouldRetryResume =
                isRetryableStreamNetworkError(error)
                && resumeOffset > 0
                && resumeAttempt < maxStreamResumeRetries;
              if (shouldRetryResume) {
                resumeAttempt += 1;
                this._runtimeNotes.push(`model_stream_resume_retry:${resumeAttempt}`);
                this._runtimeNotes.push(`model_stream_resume_offset:${resumeOffset}`);
                continue;
              }

              if (text.includes('bigint')) {
                this._runtimeNotes.push('model_fs_write_bigint_error');
              } else if (text.includes('abort')) {
                this._runtimeNotes.push('model_fs_write_abort');
              } else if (text.includes('array buffer allocation failed')) {
                this._runtimeNotes.push('model_fs_write_arraybuffer_oom');
              } else if (isRetryableStreamNetworkError(error)) {
                this._runtimeNotes.push('model_fs_write_network_error');
                this._runtimeNotes.push('model_fs_write_failed');
              } else {
                this._runtimeNotes.push('model_fs_write_failed');
              }
              throw error;
            }
          }

          shardLoaded[shardIndex] = shardBytes;
          if (shardTotals[shardIndex] < shardBytes) {
            shardTotals[shardIndex] = shardBytes;
          }
          totalModelBytes += shardBytes;
          emitAggregateProgress();
        }

        this._modelPaths = modelPaths;
        this._modelPath = modelPaths[0] || null;
        this._modelBytes = totalModelBytes;

        let rc = 0;
        try {
          rc = Number(
            await core.ccall(
              'llamadart_webgpu_load_model',
              'number',
              ['string', 'number', 'number', 'number'],
              [this._modelPath, this._nCtx, this._threads, this._nGpuLayers],
              { async: true },
            ),
          );
        } catch (error) {
          const text = String(error || '').toLowerCase();
          if (text.includes('bigint')) {
            this._runtimeNotes.push('model_load_ccall_bigint_error');
          } else if (text.includes('abort')) {
            this._runtimeNotes.push('model_load_ccall_abort');
          } else {
            this._runtimeNotes.push('model_load_ccall_failed');
          }
          throw error;
        }

        if (rc !== 0) {
          throw new Error(this._coreErrorMessage('Failed to load GGUF model', rc));
        }

        if (modelUrls.length > 1) {
          this._runtimeNotes.push(`model_split_loaded:${modelUrls.length}`);
        }
      } catch (error) {
        this._modelBytes = 0;
        this._modelPath = null;
        this._modelPaths = [];
        for (const modelPath of modelPaths) {
          this._deleteFsFile(modelPath);
        }
        throw error;
      }
    }

    const shouldProbeSafariGpu = this._isSafari
      && this._nGpuLayers > 0
      && options.safariGpuProbe !== false
      && !loadedViaRemoteFetch;

    if (shouldProbeSafariGpu) {
      const defaultProbePrompts = [
        'user: hi\nassistant:',
        'user: say hello in one short sentence\nassistant:',
      ];

      const probePrompts = Array.isArray(options.safariProbePrompts)
        ? options.safariProbePrompts
          .map((v) => String(v || '').trim())
          .filter((v) => v.length > 0)
        : (typeof options.safariProbePrompt === 'string' && options.safariProbePrompt.trim().length > 0
            ? [options.safariProbePrompt.trim()]
            : defaultProbePrompts);

      const probeTokensRaw = Number(options.safariProbeTokens);
      const probeTokens = Number.isFinite(probeTokensRaw) && probeTokensRaw > 0
        ? Math.min(Math.trunc(probeTokensRaw), 96)
        : 48;

      const runProbe = async (probePrompt, probeSeed) => {
        try {
          const probeOutput = await this.createCompletion(probePrompt, {
            nPredict: probeTokens,
            temp: 0,
            topK: 1,
            topP: 1,
            penalty: 1,
            seed: probeSeed,
          });
          return !looksLikeCorruptedGeneration(probeOutput);
        } catch (_) {
          return false;
        }
      };

      let initialProbePassed = true;
      for (let i = 0; i < probePrompts.length; i += 1) {
        const ok = await runProbe(probePrompts[i], i + 1);
        if (!ok) {
          initialProbePassed = false;
          break;
        }
      }

      if (!initialProbePassed) {
        this._runtimeNotes.push('safari_gpu_probe_failed');

        const retryCandidates = [];
        if (this._nGpuLayers > 1) {
          retryCandidates.push(1);
        }
        retryCandidates.push(0);

        let stabilized = false;
        for (const candidateLayers of retryCandidates) {
          try {
            core.ccall('llamadart_webgpu_shutdown', null, [], []);
          } catch (_) {
            // ignore shutdown retries
          }

          const retryRc = Number(
            await core.ccall(
              'llamadart_webgpu_load_model',
              'number',
              ['string', 'number', 'number', 'number'],
              [this._modelPath, this._nCtx, this._threads, candidateLayers],
              { async: true },
            ),
          );

          if (retryRc !== 0) {
            continue;
          }

          this._nGpuLayers = candidateLayers;
          if (candidateLayers === 0) {
            this._runtimeNotes.push('safari_fallback_cpu');
            stabilized = true;
            break;
          }

          let retryProbePassed = true;
          for (let i = 0; i < probePrompts.length; i += 1) {
            const ok = await runProbe(probePrompts[i], i + 11);
            if (!ok) {
              retryProbePassed = false;
              break;
            }
          }

          if (retryProbePassed) {
            this._runtimeNotes.push(`safari_gpu_layers_capped:${candidateLayers}`);
            stabilized = true;
            break;
          }
        }

        if (!stabilized) {
          throw new Error('Safari GPU probe failed and fallback attempts were unsuccessful.');
        }
      } else {
        this._runtimeNotes.push('safari_gpu_probe_passed');
      }
    }

    try {
      const effectiveNctx = Number(core.ccall('llamadart_webgpu_get_context_size', 'number', [], []));
      if (effectiveNctx > 0) {
        this._nCtx = effectiveNctx;
      }
    } catch (_) {
      // Keep requested nCtx if runtime query is unavailable.
    }

    this._mmProjPath = null;
    this._mmSupportsVision = false;
    this._mmSupportsAudio = false;
    this._mediaFileCounter = 0;
    this._stagedMediaPaths = [];
    this._gpuActive = this._gpuActive && this._nGpuLayers > 0;

    if (this._releaseModelFiles() > 0) {
      this._runtimeNotes.push('model_file_released');
    }

    return 1;
  }

  async loadMultimodalProjector(url) {
    if (!this._core || this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }

    if (typeof url !== 'string' || url.length === 0) {
      throw new Error('Multimodal projector URL/path is empty.');
    }

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch multimodal projector: ${response.status} ${response.statusText}`,
      );
    }

    const core = await this._ensureCore();

    if (!core.FS.analyzePath('/mmproj').exists) {
      core.FS.mkdir('/mmproj');
    }

    const fileName = basenameFromUrl(url);
    this._mmProjPath = `/mmproj/${fileName}`;
    await writeResponseToFsFileWithProgress(
      response,
      core.FS,
      this._mmProjPath,
      null,
      { useBigIntPosition: this._coreVariant === 'wasm64' },
    );

    const rc = Number(
      await core.ccall(
        'llamadart_webgpu_mmproj_load',
        'number',
        ['string'],
        [this._mmProjPath],
        { async: true },
      ),
    );
    if (rc !== 0) {
      this._mmProjPath = null;
      throw new Error(this._coreErrorMessage('Failed to load multimodal projector', rc));
    }

    this._mmSupportsVision = Number(
      core.ccall('llamadart_webgpu_mmproj_supports_vision', 'number', [], []),
    ) === 1;
    this._mmSupportsAudio = Number(
      core.ccall('llamadart_webgpu_mmproj_supports_audio', 'number', [], []),
    ) === 1;
    return 1;
  }

  async unloadMultimodalProjector() {
    if (!this._core) {
      this._mmProjPath = null;
      this._mmSupportsVision = false;
      this._mmSupportsAudio = false;
      return;
    }

    try {
      this._core.ccall('llamadart_webgpu_mmproj_free', null, [], []);
    } finally {
      this._mmProjPath = null;
      this._mmSupportsVision = false;
      this._mmSupportsAudio = false;
    }
  }

  supportsVision() {
    return this._mmSupportsVision;
  }

  supportsAudio() {
    return this._mmSupportsAudio;
  }

  _clearStagedMediaFiles() {
    if (!this._core || this._stagedMediaPaths.length === 0) {
      this._stagedMediaPaths = [];
      return;
    }

    for (const mediaPath of this._stagedMediaPaths) {
      try {
        this._core.FS.unlink(mediaPath);
      } catch (_) {
        // ignore best-effort cleanup failures
      }
    }

    this._stagedMediaPaths = [];
  }

  _clearPendingMedia() {
    this._core?.ccall('llamadart_webgpu_media_clear_pending', null, [], []);
    this._clearStagedMediaFiles();
  }

  _persistMediaBytes(bytes, extension = '.bin') {
    if (!this._core) {
      throw new Error('WebGPU core is not initialized.');
    }

    if (!this._core.FS.analyzePath('/media').exists) {
      this._core.FS.mkdir('/media');
    }

    this._mediaFileCounter += 1;
    const suffix = typeof extension === 'string' && extension.startsWith('.')
      ? extension
      : '.bin';
    const mediaPath = `/media/input_${Date.now()}_${this._mediaFileCounter}${suffix}`;
    this._core.FS.writeFile(mediaPath, bytes);
    this._stagedMediaPaths.push(mediaPath);
    return mediaPath;
  }

  _addMediaFile(mediaPath) {
    const rc = Number(
      this._core.ccall(
        'llamadart_webgpu_media_add_file',
        'number',
        ['string'],
        [mediaPath],
      ),
    );
    if (rc !== 0) {
      throw new Error(this._coreErrorMessage('Failed to add media file', rc));
    }
  }

  _addRawRgbMediaBytes(bytes, width, height) {
    const rc = Number(
      this._core.ccall(
        'llamadart_webgpu_media_add_rgb',
        'number',
        ['number', 'number', 'array', 'number'],
        [width, height, bytes, bytes.length],
      ),
    );
    if (rc !== 0) {
      throw new Error(this._coreErrorMessage('Failed to add raw RGB media bytes', rc));
    }
  }

  _addAudioSamples(samples) {
    const sampleBytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
    const rc = Number(
      this._core.ccall(
        'llamadart_webgpu_media_add_audio_f32',
        'number',
        ['array', 'number'],
        [sampleBytes, samples.length],
      ),
    );
    if (rc !== 0) {
      throw new Error(this._coreErrorMessage('Failed to add audio samples', rc));
    }
  }

  async _fetchMediaBytes(url) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch media: ${response.status} ${response.statusText}`);
    }

    return new Uint8Array(await response.arrayBuffer());
  }

  async _stageMultimodalParts(parts) {
    this._clearPendingMedia();

    const mediaParts = Array.isArray(parts) ? parts : [];
    if (mediaParts.length === 0) {
      return;
    }

    if (!this._mmProjPath) {
      throw new Error(
        'Multimodal input requires a loaded projector. Call loadMultimodalProjector first.',
      );
    }

    for (const rawPart of mediaParts) {
      const part = rawPart && typeof rawPart === 'object' ? rawPart : {};
      const type = String(part.type || '').toLowerCase();

      if (type === 'image') {
        const bytes = toUint8Array(part.bytes);
        if (bytes && bytes.length > 0) {
          const width = Number(part.width);
          const height = Number(part.height);
          const isRawRgb = Number.isInteger(width)
            && Number.isInteger(height)
            && width > 0
            && height > 0
            && bytes.length === (width * height * 3);

          if (isRawRgb) {
            this._addRawRgbMediaBytes(bytes, width, height);
          } else {
            const mediaPath = this._persistMediaBytes(bytes, '.img');
            this._addMediaFile(mediaPath);
          }
          continue;
        }

        if (typeof part.url !== 'string' || part.url.length === 0) {
          throw new Error('Image part must provide bytes or url.');
        }

        const fetched = await this._fetchMediaBytes(part.url);
        const mediaPath = this._persistMediaBytes(fetched, '.img');
        this._addMediaFile(mediaPath);
        continue;
      }

      if (type === 'audio') {
        const samples = toFloat32Array(part.samples);
        if (samples && samples.length > 0) {
          this._addAudioSamples(samples);
          continue;
        }

        const bytes = toUint8Array(part.bytes);
        if (bytes && bytes.length > 0) {
          const mediaPath = this._persistMediaBytes(bytes, '.aud');
          this._addMediaFile(mediaPath);
          continue;
        }

        if (typeof part.url !== 'string' || part.url.length === 0) {
          throw new Error('Audio part must provide samples, bytes, or url.');
        }

        const fetched = await this._fetchMediaBytes(part.url);
        const mediaPath = this._persistMediaBytes(fetched, '.aud');
        this._addMediaFile(mediaPath);
      }
    }
  }

  async createCompletion(prompt, options = {}) {
    if (this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }

    this._abortRequested = false;

    const nPredict = Number(options.nPredict) > 0 ? Number(options.nPredict) : 256;
    const temp = Number.isFinite(options.temp) ? Number(options.temp) : 0.8;
    const topK = Number.isFinite(options.topK) ? Number(options.topK) : 40;
    const topP = Number.isFinite(options.topP) ? Number(options.topP) : 0.95;
    const penalty = Number.isFinite(options.penalty) ? Number(options.penalty) : 1.1;
    const grammar = typeof options.grammar === 'string' && options.grammar.length > 0
      ? options.grammar
      : null;
    const seed = Number.isInteger(options.seed)
      ? Number(options.seed)
      : Math.floor(Math.random() * 0xffffffff);

    await this._stageMultimodalParts(options.parts);

    let generationStarted = false;

    try {
      const beginRc = Number(
        await this._core.ccall(
          'llamadart_webgpu_begin_generation',
          'number',
          ['string', 'number', 'number', 'number', 'number', 'string', 'number'],
          [
            String(prompt),
            temp,
            topK,
            topP,
            penalty,
            grammar,
            seed >>> 0,
          ],
          { async: true },
        ),
      );

      if (beginRc !== 0) {
        throw new Error(this._coreErrorMessage('Failed to start generation', beginRc));
      }

      generationStarted = true;

      let generated = 0;
      let streamed = '';

      while (generated < nPredict) {
        if (this._abortRequested || options.signal?.aborted) {
          break;
        }

        const stepRc = Number(
          await this._core.ccall(
            'llamadart_webgpu_next_token',
            'number',
            [],
            [],
            { async: true },
          ),
        );
        if (stepRc === 0) {
          break;
        }

        if (stepRc < 0) {
          throw new Error(this._coreErrorMessage('Generation step failed', stepRc));
        }

        generated += 1;
        const piece = this._core.ccall('llamadart_webgpu_last_piece', 'string', [], []) || '';
        if (piece.length === 0) {
          continue;
        }

        streamed += piece;
        if (typeof options.onToken === 'function') {
          options.onToken(textEncoder.encode(piece), streamed);
        }

        if ((generated % 4) === 0) {
          await new Promise((resolve) => setTimeout(resolve, 0));
        }
      }

      const text = this._core.ccall('llamadart_webgpu_last_output', 'string', [], []) || streamed;
      return text;
    } finally {
      if (generationStarted) {
        this._core.ccall('llamadart_webgpu_end_generation', null, [], []);
      }
      this._clearPendingMedia();
    }
  }

  async tokenize(text, _addSpecial = true) {
    if (this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }

    const rc = Number(
      await this._core.ccall(
        'llamadart_webgpu_tokenize_to_json',
        'number',
        ['string', 'number'],
        [String(text), _addSpecial ? 1 : 0],
        { async: true },
      ),
    );

    if (rc < 0) {
      throw new Error(this._coreErrorMessage('Tokenization failed', rc));
    }

    const raw = this._core.ccall('llamadart_webgpu_last_tokens_json', 'string', [], []) || '[]';
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.map((v) => Number(v) | 0)
      : [];
  }

  async detokenize(tokens, _special = false) {
    if (this._modelBytes <= 0) {
      throw new Error('No model loaded. Call loadModelFromUrl first.');
    }

    const normalized = Array.isArray(tokens)
      ? tokens
      : Array.from(tokens || []);
    const tokenText = JSON.stringify(normalized.map((v) => Number(v) | 0));

    const rc = Number(
      await this._core.ccall(
        'llamadart_webgpu_detokenize_from_json',
        'number',
        ['string', 'number'],
        [tokenText, _special ? 1 : 0],
        { async: true },
      ),
    );

    if (rc < 0) {
      throw new Error(this._coreErrorMessage('Detokenization failed', rc));
    }

    return this._core.ccall('llamadart_webgpu_last_detokenized', 'string', [], []) || '';
  }

  getModelMetadata() {
    let modelMetadata = {};

    try {
      const raw = this._core?.ccall('llamadart_webgpu_model_meta_json', 'string', [], []);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object') {
          modelMetadata = parsed;
        }
      }
    } catch (_) {
      // Keep fallback metadata only.
    }

    return {
      ...modelMetadata,
      'llamadart.webgpu.prototype': '1',
      'llamadart.webgpu.backends': this._backendLabels.join(','),
      'llamadart.webgpu.model_bytes': String(this._modelBytes),
      'llamadart.webgpu.n_threads': String(this._threads),
      'llamadart.webgpu.n_gpu_layers': String(this._nGpuLayers),
      'llamadart.webgpu.core_variant': this._coreVariant,
      'llamadart.webgpu.model_source': this._modelSource,
      'llamadart.webgpu.model_cache_state': this._modelCacheState,
      'llamadart.webgpu.model_cache_name': this._modelCacheName,
      'llamadart.webgpu.runtime_notes': this._runtimeNotes.join(';'),
      'llamadart.webgpu.mmproj_loaded': this._mmProjPath ? '1' : '0',
      'llamadart.webgpu.supports_vision': this._mmSupportsVision ? '1' : '0',
      'llamadart.webgpu.supports_audio': this._mmSupportsAudio ? '1' : '0',
    };
  }

  getContextSize() {
    try {
      const nctx = Number(this._core?.ccall('llamadart_webgpu_get_context_size', 'number', [], []));
      if (nctx > 0) {
        return nctx;
      }
    } catch (_) {
      // fall through to cached value
    }

    return this._nCtx;
  }

  isGpuActive() {
    return this._gpuActive;
  }

  getBackendName() {
    if (this._nGpuLayers === 0) {
      return 'WASM (Prototype bridge)';
    }

    if (this._backendLabels.length > 0) {
      return this._backendLabels.join(', ');
    }
    return this._gpuActive
      ? 'WebGPU (Prototype bridge)'
      : 'WASM (Prototype bridge)';
  }

  setLogLevel(level) {
    if (Number.isFinite(level)) {
      this._logLevel = Math.max(0, Math.min(4, Math.trunc(level)));
    }
    this._applyCoreLogLevel();
  }

  cancel() {
    this._abortRequested = true;

    if (this._activeTransferAbortController) {
      try {
        this._activeTransferAbortController.abort();
      } catch (_) {
        // ignore best-effort transfer abort failures
      }
      this._activeTransferAbortController = null;
    }

    try {
      this._core?.ccall('llamadart_webgpu_request_cancel', null, [], []);
    } catch (_) {
      // ignore best-effort cancel failures
    }
  }

  async dispose() {
    if (this._activeTransferAbortController) {
      try {
        this._activeTransferAbortController.abort();
      } catch (_) {
        // ignore best-effort transfer abort failures
      }
      this._activeTransferAbortController = null;
    }

    if (this._core) {
      this._clearPendingMedia();
      this._core.ccall('llamadart_webgpu_mmproj_free', null, [], []);
      this._core.ccall('llamadart_webgpu_shutdown', null, [], []);
    }
    this._modelPath = null;
    this._modelPaths = [];
    this._modelBytes = 0;
    this._modelSource = 'network';
    this._modelCacheState = 'disabled';
    this._mmProjPath = null;
    this._mmSupportsVision = false;
    this._mmSupportsAudio = false;
    this._abortRequested = false;
    this._activeTransferAbortController = null;
  }

  async applyChatTemplate(messages, addAssistant = true, _customTemplate = null) {
    return buildPromptFromMessages(messages, addAssistant);
  }
}

export class LlamaWebGpuBridge {
  static supportsSafariAdaptiveGpu =
    LlamaWebGpuBridgeRuntime.supportsSafariAdaptiveGpu === true;

  constructor(config = {}) {
    this._config = config;
    this._runtime = null;
    this._workerProxy = null;
    this._workerFallbackReason = null;

    this._metadata = {};
    this._contextSize = 0;
    this._gpuActive = false;
    this._backendName = 'WASM (Prototype bridge)';
    this._supportsVision = false;
    this._supportsAudio = false;

    if (this._shouldUseWorker()) {
      try {
        this._workerProxy = new BridgeWorkerProxy({
          moduleUrl: this._workerModuleUrl(),
          config: this._workerConfig(),
        });
      } catch (error) {
        this._disableWorkerFallback(error);
      }
    }

    if (!this._workerProxy) {
      this._runtime = this._createRuntime();
    }
  }

  _createRuntime() {
    return new LlamaWebGpuBridgeRuntime({
      ...this._config,
      disableWorker: true,
    });
  }

  _shouldUseWorker() {
    if (this._config?.disableWorker === true) {
      return false;
    }

    if (typeof Worker === 'undefined' ||
        typeof Blob === 'undefined' ||
        typeof URL === 'undefined' ||
        typeof URL.createObjectURL !== 'function') {
      return false;
    }

    if (typeof this._config?.coreModuleFactory === 'function') {
      return false;
    }

    return true;
  }

  _workerModuleUrl() {
    const candidate = this._config?.workerUrl;
    if (typeof candidate === 'string' && candidate.trim().length > 0) {
      return candidate.trim();
    }

    try {
      return new URL('./llama_webgpu_bridge_worker.js', import.meta.url).toString();
    } catch (_) {
      return import.meta.url;
    }
  }

  _workerConfig() {
    const config = this._config || {};
    return {
      wasmUrl: typeof config.wasmUrl === 'string' ? config.wasmUrl : undefined,
      wasmUrlMem64: typeof config.wasmUrlMem64 === 'string'
        ? config.wasmUrlMem64
        : undefined,
      coreModuleUrl: typeof config.coreModuleUrl === 'string'
        ? config.coreModuleUrl
        : undefined,
      coreModuleUrlMem64: typeof config.coreModuleUrlMem64 === 'string'
        ? config.coreModuleUrlMem64
        : undefined,
      preferMemory64: typeof config.preferMemory64 === 'boolean'
        ? config.preferMemory64
        : undefined,
      threads: Number(config.threads) > 0 ? Number(config.threads) : undefined,
      nGpuLayers: Number.isFinite(config.nGpuLayers)
        ? Number(config.nGpuLayers)
        : undefined,
      userAgent: typeof config.userAgent === 'string' ? config.userAgent : undefined,
      cacheName: typeof config.cacheName === 'string' ? config.cacheName : undefined,
      remoteFetchThresholdBytes: Number(config.remoteFetchThresholdBytes) > 0
        ? Number(config.remoteFetchThresholdBytes)
        : undefined,
      remoteFetchChunkBytes: Number(config.remoteFetchChunkBytes) > 0
        ? Number(config.remoteFetchChunkBytes)
        : undefined,
      logLevel: Number.isFinite(config.logLevel) ? Number(config.logLevel) : 2,
    };
  }

  _applyShadowState(state) {
    if (!state || typeof state !== 'object') {
      return;
    }

    if (state.metadata && typeof state.metadata === 'object') {
      this._metadata = state.metadata;
    }
    if (Number.isFinite(state.contextSize)) {
      this._contextSize = Number(state.contextSize);
    }
    if (typeof state.gpuActive === 'boolean') {
      this._gpuActive = state.gpuActive;
    }
    if (typeof state.backendName === 'string' && state.backendName.length > 0) {
      this._backendName = state.backendName;
    }
    if (typeof state.supportsVision === 'boolean') {
      this._supportsVision = state.supportsVision;
    }
    if (typeof state.supportsAudio === 'boolean') {
      this._supportsAudio = state.supportsAudio;
    }
  }

  _shouldFallbackToMainThread(error) {
    const text = serializeWorkerError(error).toLowerCase();

    if (text.includes('aborted(native code called abort())')) {
      return false;
    }
    if (text.includes('array buffer allocation failed')) {
      return false;
    }
    if (text.includes('bad_alloc')) {
      return false;
    }
    if (text.includes('out of memory')) {
      return false;
    }
    if (text.includes('memory access out of bounds')) {
      return false;
    }

    if (text.includes('bridge worker')) {
      return true;
    }
    if (text.includes('worker request failed')) {
      return true;
    }
    if (text.includes('worker proxy is not available')) {
      return true;
    }
    if (text.includes('worker is not initialized')) {
      return true;
    }
    if (text.includes('failed to initialize bridge worker')) {
      return true;
    }
    if (text.includes('script error')) {
      return true;
    }

    return false;
  }

  _disableWorkerFallback(error) {
    const reason = serializeWorkerError(error);
    this._workerFallbackReason = reason;

    if (typeof globalThis !== 'undefined') {
      globalThis.__llamadartBridgeWorkerFallbackReason = reason;
    }

    if (typeof console !== 'undefined' && typeof console.warn === 'function') {
      console.warn(
        `llamadart: bridge worker unavailable, falling back to main thread (${reason})`,
      );
    }

    if (this._workerProxy) {
      this._workerProxy.dispose().catch(() => {});
      this._workerProxy = null;
    }

    if (!this._runtime) {
      this._runtime = this._createRuntime();
    }

    if (
      this._runtime
      && Array.isArray(this._runtime._runtimeNotes)
      && typeof reason === 'string'
      && reason.length > 0
    ) {
      this._runtime._runtimeNotes.push(`worker_fallback:${reason}`);
    }
  }

  async _callWorker(method, args, onEvent) {
    if (!this._workerProxy) {
      throw new Error('Bridge worker proxy is not available');
    }

    try {
      const response = await this._workerProxy.call(method, args, onEvent);
      this._applyShadowState(response.state);
      return response.value;
    } catch (error) {
      if (error && typeof error === 'object' && error.state) {
        this._applyShadowState(error.state);
      }
      throw error;
    }
  }

  async loadModelFromUrl(url, options = {}) {
    if (!this._workerProxy) {
      return this._runtime.loadModelFromUrl(url, options);
    }

    try {
      const workerOptions = { ...options };
      delete workerOptions.progressCallback;

      return await this._callWorker(
        'loadModelFromUrl',
        [url, workerOptions],
        (event) => {
          if (event.event !== 'progress') {
            return;
          }
          if (typeof options.progressCallback !== 'function') {
            return;
          }
          options.progressCallback(event.payload || {});
        },
      );
    } catch (error) {
      if (!this._shouldFallbackToMainThread(error)) {
        throw error;
      }

      this._disableWorkerFallback(error);
      return this._runtime.loadModelFromUrl(url, options);
    }
  }

  async prefetchModelToCache(url, options = {}) {
    if (!this._runtime) {
      this._runtime = this._createRuntime();
    }
    return this._runtime.prefetchModelToCache(url, options);
  }

  async evictModelFromCache(url, options = {}) {
    if (!this._runtime) {
      this._runtime = this._createRuntime();
    }
    return this._runtime.evictModelFromCache(url, options);
  }

  async createCompletion(prompt, options = {}) {
    if (!this._workerProxy) {
      return this._runtime.createCompletion(prompt, options);
    }

    let removeAbortListener = null;
    try {
      if (options?.signal && typeof options.signal.addEventListener === 'function') {
        const onAbort = () => {
          this.cancel();
        };
        options.signal.addEventListener('abort', onAbort, { once: true });
        removeAbortListener = () => {
          options.signal.removeEventListener('abort', onAbort);
        };
      }

      const workerOptions = { ...options };
      delete workerOptions.onToken;
      delete workerOptions.signal;

      return await this._callWorker(
        'createCompletion',
        [prompt, workerOptions],
        (event) => {
          if (event.event !== 'token') {
            return;
          }
          if (typeof options.onToken !== 'function') {
            return;
          }

          const payload = event.payload || {};
          const piece = Uint8Array.from(Array.isArray(payload.piece) ? payload.piece : []);
          options.onToken(piece, String(payload.currentText || ''));
        },
      );
    } catch (error) {
      this._disableWorkerFallback(error);
      return this._runtime.createCompletion(prompt, options);
    } finally {
      removeAbortListener?.();
    }
  }

  async loadMultimodalProjector(url) {
    if (!this._workerProxy) {
      return this._runtime.loadMultimodalProjector(url);
    }

    try {
      return await this._callWorker('loadMultimodalProjector', [url]);
    } catch (error) {
      this._disableWorkerFallback(error);
      return this._runtime.loadMultimodalProjector(url);
    }
  }

  async unloadMultimodalProjector() {
    if (!this._workerProxy) {
      return this._runtime.unloadMultimodalProjector();
    }

    try {
      return await this._callWorker('unloadMultimodalProjector', []);
    } catch (error) {
      this._disableWorkerFallback(error);
      return this._runtime.unloadMultimodalProjector();
    }
  }

  supportsVision() {
    if (this._workerProxy) {
      return this._supportsVision;
    }
    return this._runtime.supportsVision();
  }

  supportsAudio() {
    if (this._workerProxy) {
      return this._supportsAudio;
    }
    return this._runtime.supportsAudio();
  }

  async tokenize(text, addSpecial = true) {
    if (!this._workerProxy) {
      return this._runtime.tokenize(text, addSpecial);
    }

    try {
      return await this._callWorker('tokenize', [text, addSpecial]);
    } catch (error) {
      this._disableWorkerFallback(error);
      return this._runtime.tokenize(text, addSpecial);
    }
  }

  async detokenize(tokens, special = false) {
    if (!this._workerProxy) {
      return this._runtime.detokenize(tokens, special);
    }

    const normalized = Array.isArray(tokens)
      ? tokens
      : Array.from(tokens || []);

    try {
      return await this._callWorker('detokenize', [normalized, special]);
    } catch (error) {
      this._disableWorkerFallback(error);
      return this._runtime.detokenize(normalized, special);
    }
  }

  getModelMetadata() {
    if (this._workerProxy) {
      return {
        ...(this._metadata || {}),
        'llamadart.webgpu.execution': 'worker',
      };
    }

    const workerReason =
      typeof this._workerFallbackReason === 'string' && this._workerFallbackReason.length > 0
        ? this._workerFallbackReason
        : null;

    return {
      ...this._runtime.getModelMetadata(),
      'llamadart.webgpu.execution': 'main-thread',
      ...(workerReason == null
        ? {}
        : { 'llamadart.webgpu.worker_fallback_reason': workerReason }),
    };
  }

  getContextSize() {
    if (this._workerProxy) {
      return this._contextSize || 0;
    }
    return this._runtime.getContextSize();
  }

  isGpuActive() {
    if (this._workerProxy) {
      return this._gpuActive;
    }
    return this._runtime.isGpuActive();
  }

  getBackendName() {
    if (this._workerProxy) {
      return this._backendName;
    }
    return this._runtime.getBackendName();
  }

  setLogLevel(level) {
    if (this._workerProxy) {
      this._callWorker('setLogLevel', [level]).catch((error) => {
        this._disableWorkerFallback(error);
      });
      return;
    }
    this._runtime.setLogLevel(level);
  }

  cancel() {
    if (this._workerProxy) {
      this._callWorker('cancel', []).catch(() => {});
    }

    if (this._runtime) {
      this._runtime.cancel();
    }
  }

  async dispose() {
    if (this._workerProxy) {
      await this._workerProxy.dispose();
      this._workerProxy = null;
      this._metadata = {};
      this._contextSize = 0;
      this._gpuActive = false;
      this._backendName = 'WASM (Prototype bridge)';
      this._supportsVision = false;
      this._supportsAudio = false;
    }

    if (this._runtime) {
      await this._runtime.dispose();
      this._runtime = null;
    }
  }

  async applyChatTemplate(messages, addAssistant = true, customTemplate = null) {
    if (!this._workerProxy) {
      return this._runtime.applyChatTemplate(messages, addAssistant, customTemplate);
    }

    try {
      return await this._callWorker('applyChatTemplate', [messages, addAssistant, customTemplate]);
    } catch (error) {
      this._disableWorkerFallback(error);
      return this._runtime.applyChatTemplate(messages, addAssistant, customTemplate);
    }
  }
}

if (typeof window !== 'undefined' && !window.LlamaWebGpuBridge) {
  window.LlamaWebGpuBridge = LlamaWebGpuBridge;
}
