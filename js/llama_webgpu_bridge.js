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

export class LlamaWebGpuBridge {
  constructor(config = {}) {
    this._config = config;
    this._core = null;
    this._backendLabels = [];
    this._gpuActive = false;
    this._modelPath = null;
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
    this._modelSource = 'network';
    this._modelCacheState = 'disabled';
    this._modelCacheName = defaultModelCacheName;
  }

  static supportsSafariAdaptiveGpu = true;

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

  async _getCachedModelResponse(url, options = {}) {
    const useCache = options.useCache !== false;
    this._modelSource = 'network';
    this._modelCacheState = useCache ? 'unavailable' : 'disabled';
    this._modelCacheName = this._resolveCacheName(options);

    if (!useCache) {
      const response = await fetch(url, { cache: 'no-store' });
      this._modelCacheState = 'disabled';
      return response;
    }

    if (!globalThis.caches || typeof globalThis.caches.open !== 'function') {
      this._modelCacheState = 'unavailable';
      return fetch(url);
    }

    const cacheKey = normalizeAbsoluteUrl(url);

    try {
      const cache = await globalThis.caches.open(this._modelCacheName);
      const cached = await cache.match(cacheKey);
      if (cached) {
        this._modelSource = 'cache';
        this._modelCacheState = 'hit';
        this._runtimeNotes.push('model_cache_hit');
        return cached;
      }

      this._modelCacheState = 'miss';
      const response = await fetch(url);

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

      return response;
    } catch (_) {
      this._modelCacheState = 'error';
      this._runtimeNotes.push('model_cache_error');
      return fetch(url);
    }
  }

  async _ensureCore() {
    if (this._core) {
      return this._core;
    }

    const moduleFactory = this._config.coreModuleFactory
      ? this._config.coreModuleFactory
      : await importCoreFactory(this._config.coreModuleUrl ?? './llama_webgpu_core.js');

    this._core = await moduleFactory({
      locateFile: (path, prefix) => {
        if (path.endsWith('.wasm') && this._config.wasmUrl) {
          return this._config.wasmUrl;
        }
        return `${prefix}${path}`;
      },
    });

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

    const response = await this._getCachedModelResponse(url, options);
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
    }

    const bytes = await readResponseBytesWithProgress(
      response,
      options.progressCallback,
    );

    const core = await this._ensureCore();
    if (!core.FS.analyzePath('/models').exists) {
      core.FS.mkdir('/models');
    }

    const fileName = basenameFromUrl(url);
    this._modelPath = `/models/${fileName}`;
    core.FS.writeFile(this._modelPath, bytes);
    this._modelBytes = bytes.byteLength;
    this._nCtx = Number(options.nCtx) > 0 ? Number(options.nCtx) : this._nCtx;

    const requestedThreads = Number(options.nThreads);
    if (Number.isFinite(requestedThreads) && requestedThreads > 0) {
      this._threads = Math.trunc(requestedThreads);
    }

    const requestedGpuLayers = Number(options.nGpuLayers);
    if (Number.isFinite(requestedGpuLayers)) {
      this._nGpuLayers = Math.trunc(requestedGpuLayers);
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

    const rc = Number(
      await core.ccall(
        'llamadart_webgpu_load_model',
        'number',
        ['string', 'number', 'number', 'number'],
        [this._modelPath, this._nCtx, this._threads, this._nGpuLayers],
        { async: true },
      ),
    );

    if (rc !== 0) {
      throw new Error(this._coreErrorMessage('Failed to load GGUF model', rc));
    }

    const shouldProbeSafariGpu = this._isSafari
      && this._nGpuLayers > 0
      && options.safariGpuProbe !== false;

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

    return 1;
  }

  async loadMultimodalProjector(url) {
    if (!this._modelPath) {
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

    const bytes = await readResponseBytesWithProgress(response, null);
    const core = await this._ensureCore();

    if (!core.FS.analyzePath('/mmproj').exists) {
      core.FS.mkdir('/mmproj');
    }

    const fileName = basenameFromUrl(url);
    this._mmProjPath = `/mmproj/${fileName}`;
    core.FS.writeFile(this._mmProjPath, bytes);

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
    if (!this._modelPath) {
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
    if (!this._modelPath) {
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
    if (!this._modelPath) {
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

  cancel() {
    this._abortRequested = true;
    try {
      this._core?.ccall('llamadart_webgpu_request_cancel', null, [], []);
    } catch (_) {
      // ignore best-effort cancel failures
    }
  }

  async dispose() {
    if (this._core) {
      this._clearPendingMedia();
      this._core.ccall('llamadart_webgpu_mmproj_free', null, [], []);
      this._core.ccall('llamadart_webgpu_shutdown', null, [], []);
    }
    this._modelPath = null;
    this._modelBytes = 0;
    this._modelSource = 'network';
    this._modelCacheState = 'disabled';
    this._mmProjPath = null;
    this._mmSupportsVision = false;
    this._mmSupportsAudio = false;
    this._abortRequested = false;
  }

  async applyChatTemplate(messages, addAssistant = true, _customTemplate = null) {
    return buildPromptFromMessages(messages, addAssistant);
  }
}

if (typeof window !== 'undefined' && !window.LlamaWebGpuBridge) {
  window.LlamaWebGpuBridge = LlamaWebGpuBridge;
}
