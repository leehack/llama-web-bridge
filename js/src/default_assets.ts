import { normalizeAbsoluteUrl } from './internal/model_source.ts';
import type { LlamaWebGpuBridgeConfig } from './llama_webgpu_bridge.d.ts';

type AssetSource = 'explicit' | 'base' | 'default';
type ResolvedAsset = { url: string; source: AssetSource };
type CoreAssetSources = { wasm32?: AssetSource; wasm64?: AssetSource };
type CoreUrls = Pick<LlamaWebGpuBridgeConfig, 'coreModuleUrl' | 'coreModuleUrlMem64' | 'wasmUrl' | 'wasmUrlMem64'>;

export interface BridgeConfig extends LlamaWebGpuBridgeConfig {
  _coreAssetSources?: CoreAssetSources;
  _coreWorkerUrls?: string[];
}

const workerAsset = ['llama_webgpu_bridge_worker.js', () => new URL('./llama_webgpu_bridge_worker.js', import.meta.url).href] as const;
const threadedAssets = [
  ['llama_webgpu_core.js', () => new URL('./llama_webgpu_core.js', import.meta.url).href],
  ['llama_webgpu_core.wasm', () => new URL('./llama_webgpu_core.wasm', import.meta.url).href],
  ['llama_webgpu_core_mem64.js', () => new URL('./llama_webgpu_core_mem64.js', import.meta.url).href],
  ['llama_webgpu_core_mem64.wasm', () => new URL('./llama_webgpu_core_mem64.wasm', import.meta.url).href],
] as const;
const singleThreadAssets = [
  ['llama_webgpu_core_st.js', () => new URL('./llama_webgpu_core_st.js', import.meta.url).href],
  ['llama_webgpu_core_st.wasm', () => new URL('./llama_webgpu_core_st.wasm', import.meta.url).href],
  ['llama_webgpu_core_mem64_st.js', () => new URL('./llama_webgpu_core_mem64_st.js', import.meta.url).href],
  ['llama_webgpu_core_mem64_st.wasm', () => new URL('./llama_webgpu_core_mem64_st.wasm', import.meta.url).href],
] as const;

function assetBase(config: LlamaWebGpuBridgeConfig): URL | undefined {
  const value = config.assetBaseUrl?.trim();
  if (!value) return undefined;
  // Bundlers treat a direct import.meta.url argument as an asset reference.
  const moduleUrl = import.meta.url;
  const url = new URL(normalizeAbsoluteUrl(value), moduleUrl);
  if (!url.pathname.endsWith('/')) url.pathname += '/';
  return url;
}

function explicitWorkerUrl(config: LlamaWebGpuBridgeConfig): string | undefined {
  return typeof config.workerUrl === 'string' ? config.workerUrl.trim() || undefined : undefined;
}

function resolveAsset(
  [file, defaultUrl]: readonly [string, () => string],
  explicitUrl: string | undefined,
  base: URL | undefined,
  fallbackUrl = `./${file}`,
): ResolvedAsset {
  if (explicitUrl != null) return { url: explicitUrl, source: 'explicit' };
  if (base) return { url: new URL(file, base).href, source: 'base' };
  try {
    return { url: defaultUrl(), source: 'default' };
  } catch (_) {
    return { url: fallbackUrl, source: 'default' };
  }
}

export function resolveWorkerAsset(config: LlamaWebGpuBridgeConfig): ResolvedAsset {
  const explicitUrl = explicitWorkerUrl(config);
  return resolveAsset(workerAsset, explicitUrl, explicitUrl ? undefined : assetBase(config), import.meta.url);
}

export function isBridgeResolvedAsset(source: AssetSource | undefined): boolean {
  return source === 'base' || source === 'default';
}

export function resolveCoreAssets(
  config: BridgeConfig,
  useDefaults = explicitWorkerUrl(config) === undefined,
): CoreUrls & { _coreAssetSources: CoreAssetSources } {
  const { coreModuleUrl, coreModuleUrlMem64, wasmUrl, wasmUrlMem64 } = config;
  const sources: CoreAssetSources = {
    wasm32: coreModuleUrl != null ? 'explicit' : undefined,
    wasm64: coreModuleUrlMem64 != null ? 'explicit' : undefined,
  };
  const provided = { coreModuleUrl, coreModuleUrlMem64, wasmUrl, wasmUrlMem64 };
  // A worker must use the page's resolution, including intentionally absent URLs.
  if (config._coreAssetSources) return { ...provided, _coreAssetSources: config._coreAssetSources };
  if (config.coreModuleFactory) return { ...provided, _coreAssetSources: sources };

  const base = assetBase(config);
  if (!base && (!useDefaults || coreModuleUrl != null || coreModuleUrlMem64 != null)) {
    return { ...provided, _coreAssetSources: sources };
  }

  const [core32, binary32, core64, binary64] = config.allowSingleThreadCore === true
    ? singleThreadAssets
    : threadedAssets;
  const module32 = resolveAsset(core32, coreModuleUrl, base);
  const include64 = config.preferMemory64 !== false;
  const module64 = include64 ? resolveAsset(core64, coreModuleUrlMem64, base) : undefined;
  return {
    coreModuleUrl: module32.url,
    wasmUrl: resolveAsset(binary32, wasmUrl, base).url,
    coreModuleUrlMem64: module64?.url ?? coreModuleUrlMem64,
    wasmUrlMem64: include64 ? resolveAsset(binary64, wasmUrlMem64, base).url : wasmUrlMem64,
    _coreAssetSources: { wasm32: module32.source, wasm64: module64?.source ?? sources.wasm64 },
  };
}
