#!/usr/bin/env node

import fsp from 'node:fs/promises';
import path from 'node:path';

import {
  copyBridgeArtifacts,
  ensure,
  runMain,
  runPlaywright,
  validateHash,
  withServer,
  withTempDir,
} from './support.mjs';

async function checkRemount(nativeOnly) {
  const { LlamaWebGpuBridge } = await import('/llama_webgpu_bridge.js');
  const { default: createCore } = await import('/llama_webgpu_core.js');
  const core = await createCore();
  const mountPath = '/jsmodels-1';
  const modelPath = '/tiny.gguf';
  const markerPath = `${mountPath}/replacement.bin`;
  const mountId = () => core.ccall('llamadart_webgpu_js_model_fs_mount_id', 'number', ['string'], [mountPath]);
  const allocatedBytes = () => core.ccall('llamadart_webgpu_js_model_fs_allocated_bytes', 'number', [], []);
  const newBridge = () => new LlamaWebGpuBridge({
    disableWorker: true,
    jsBackedModelStaging: true,
    coreModuleFactory: async () => core,
  });
  const load = (bridge) => bridge.loadModelFromUrl(modelPath, {
    nGpuLayers: 0,
    nCtx: 128,
    nThreads: 1,
    useCache: false,
    forceRemoteFetchBackend: false,
  });

  const ccall = core.ccall.bind(core);
  const mountResults = [];
  core.ccall = (name, ...args) => {
    const result = ccall(name, ...args);
    if (name === 'llamadart_webgpu_mount_js_model_fs') mountResults.push(result);
    return result;
  };

  const assert = (condition, message) => {
    if (!condition) throw new Error(message);
  };
  const first = newBridge();
  const unlink = core.FS.unlink.bind(core.FS);
  let originalId = 0;
  core.FS.unlink = (filename) => {
    unlink(filename);
    if (filename !== `${mountPath}${modelPath}` || originalId !== 0) return;
    originalId = mountId();
    core.FS.rmdir(mountPath);
    core.FS.mkdir(mountPath);
    core.FS.writeFile(markerPath, new Uint8Array([42]));
  };

  try {
    assert(await load(first) === 1, 'first bridge did not load the model');
  } finally {
    core.FS.unlink = unlink;
    await first.dispose();
  }
  assert(originalId > 0, 'the staging directory was not replaced during cleanup');
  assert(core.FS.readFile(markerPath)[0] === 42, 'cleanup changed the replacement directory');
  const allocatedAfterDispose = allocatedBytes();
  assert(allocatedAfterDispose === 0, 'old mount retained staged chunks');
  unlink(markerPath);
  core.FS.rmdir(mountPath);
  assert(mountId() === 0 && !core.FS.readdir('/').includes('jsmodels-1'), 'staging path is still present');

  if (nativeOnly) {
    // A later bridge can choose a different path and miss the stale registration.
    const rc = ccall('llamadart_webgpu_mount_js_model_fs', 'number', ['string'], [mountPath]);
    assert(rc === 0, `native remount of ${mountPath} returned ${rc}`);
    const unmountRc = ccall('llamadart_webgpu_unmount_js_model_fs', 'number', ['string'], [mountPath]);
    assert(unmountRc === 0, `native unmount of ${mountPath} returned ${unmountRc}`);
    return rc;
  }

  const second = newBridge();
  try {
    assert(await load(second) === 1, 'second bridge did not load the model');
    assert((await second.tokenize('hello')).length > 0, 'second bridge did not tokenize');
  } catch (error) {
    throw new Error(`${error.code ?? error.name}: ${error.message} (native mount results: ${mountResults})`);
  } finally {
    await second.dispose();
  }
  assert(mountResults.length === 2 && mountResults.every((rc) => rc === 0), 'expected two successful native mounts');
  const allocatedAfterRemount = allocatedBytes();
  assert(allocatedAfterRemount === 0, 'new mount retained staged chunks');
  return { originalId, mountResults, allocatedAfterDispose, allocatedAfterRemount };
}

async function main([distDir, modelPath, modelSha256] = process.argv.slice(2)) {
  ensure(distDir && modelPath && modelSha256, 'usage: js_model_staging_remount.mjs DIST_DIR MODEL_PATH MODEL_SHA256');
  await validateHash(modelPath, modelSha256);
  const payload = await withTempDir('lane230-remount-', async (webRoot) => {
    await copyBridgeArtifacts(distDir, webRoot);
    await fsp.copyFile(modelPath, path.join(webRoot, 'tiny.gguf'));
    await fsp.writeFile(path.join(webRoot, 'index.html'), `<script type="module">
      const checkRemount = ${checkRemount.toString()};
      try {
        const nativeRemountRc = await checkRemount(true);
        const bridgeResult = await checkRemount(false);
        window.__smokeResult = { ok: true, nativeRemountRc, ...bridgeResult };
      } catch (error) {
        window.__smokeResult = { ok: false, error: error.stack };
      }
    </script>`);
    return withServer(webRoot, (url) => runPlaywright(url, 60000, null));
  });
  ensure(payload.ok, payload.error);
  const { originalId, nativeRemountRc, mountResults, allocatedAfterDispose, allocatedAfterRemount } = payload;
  console.log(JSON.stringify({ originalId, nativeRemountRc, mountResults, allocatedAfterDispose, allocatedAfterRemount }));
  return 0;
}

if (import.meta.main) await runMain('JS staging remount', main);
