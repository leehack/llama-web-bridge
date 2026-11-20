import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';

import {
  DEFAULT_MODEL_CACHE,
  resolvePinnedModel,
  stageFile,
  webGpuLaunchArgs,
  withServer,
  withTempDir,
} from '../../scripts/smoke/support.mjs';

async function loadOnSharedCore({ coreName, isolated, replaceMount }) {
  const check = (condition, message) => {
    if (!condition) throw new Error(message);
  };
  check(crossOriginIsolated === isolated, 'unexpected cross-origin isolation');
  check((typeof SharedArrayBuffer !== 'undefined') === isolated, 'unexpected SharedArrayBuffer availability');
  const { LlamaWebGpuBridge } = await import('/llama_webgpu_bridge.js');
  const { default: createCore } = await import(`/${coreName}.js`);
  const core = await createCore();
  const makeBridge = () => new LlamaWebGpuBridge({
    disableWorker: true,
    allowSingleThreadCore: true,
    jsBackedModelStaging: true,
    coreModuleFactory: async () => core,
  });
  const modelPath = '/tiny.gguf';
  const mountPath = '/jsmodels-1';
  const markerPath = `${mountPath}/replacement.bin`;
  const { FS } = core;
  const ccall = core.ccall.bind(core);
  const mountId = () => ccall('llamadart_webgpu_js_model_fs_mount_id', 'number', ['string'], [mountPath]);
  const allocatedBytes = () => ccall('llamadart_webgpu_js_model_fs_allocated_bytes', 'number', [], []);
  const mountPaths = [];
  const unmountPaths = [];
  let busy = !replaceMount;
  // An intact WasmFS mount needs fault injection to return EBUSY.
  core.ccall = (name, returnType, argTypes, args, options) => {
    if (name === 'llamadart_webgpu_mount_js_model_fs') {
      mountPaths.push(args[0]);
    }
    if (name === 'llamadart_webgpu_unmount_js_model_fs') {
      unmountPaths.push(args[0]);
      if (busy && args[0] === mountPath) return -10;
    }
    return ccall(name, returnType, argTypes, args, options);
  };
  const unlink = FS.unlink.bind(FS);
  let originalId = 0;
  if (replaceMount) {
    FS.unlink = (filename) => {
      unlink(filename);
      if (filename !== mountPath + modelPath || originalId !== 0) return;
      originalId = mountId();
      FS.rmdir(mountPath);
      FS.mkdir(mountPath);
      FS.writeFile(markerPath, new Uint8Array([42]));
    };
  }
  const load = (bridge) => bridge.loadModelFromUrl(modelPath, {
    nCtx: 128,
    nThreads: 1,
    nGpuLayers: 0,
    useCache: false,
    forceRemoteFetchBackend: false,
  });

  const first = makeBridge();
  try {
    check(await load(first) === 1, 'first bridge did not load the model');
    check(mountPaths[0] === mountPath, `new core started at ${mountPaths[0]}`);
    if (!replaceMount) {
      originalId = mountId();
      check(unmountPaths.length === 2 && unmountPaths.every((entry) => entry === mountPath),
        'first bridge did not hit the injected busy condition twice');
    }
    check(originalId > 0, 'first bridge did not stage on a native mount');
  } finally {
    FS.unlink = unlink;
    await first.dispose();
  }
  check(allocatedBytes() === 0, 'first bridge left staged chunks after disposal');

  if (replaceMount) {
    check(FS.readFile(markerPath)[0] === 42, 'cleanup changed the replacement directory');
    unlink(markerPath);
    FS.rmdir(mountPath);
    check(mountId() === 0 && !FS.readdir('/').includes('jsmodels-1'), 'replacement path still exists');
  } else {
    busy = false;
    check(mountId() === originalId, 'the busy mount did not survive disposal');
  }

  const previousUnmounts = unmountPaths.length;
  const second = makeBridge();
  try {
    check(await load(second) === 1, 'second bridge did not load the model');
    const tokens = await second.tokenize('hello');
    return {
      mountPaths,
      tokens,
      unmountPaths: unmountPaths.slice(previousUnmounts),
      remainingPaths: FS.readdir('/').filter((entry) => entry.startsWith('jsmodels-')),
      remainingId: mountId(),
      originalId,
      allocatedBytes: allocatedBytes(),
    };
  } finally {
    await second.dispose();
  }
}

test('shared-core staging survives leftover mount paths', {
  skip: process.env.LLAMADART_BROWSER_STAGING_MOUNT_SMOKE !== '1',
  timeout: 120_000,
}, async (t) => {
  const distDir = path.resolve(process.env.BRIDGE_DIST_DIR || 'dist');
  const model = await resolvePinnedModel({
    modelPath: process.env.LLAMA_WEBGPU_SMOKE_MODEL_PATH || null,
    modelUrl: process.env.LLAMA_WEBGPU_SMOKE_MODEL_URL,
    modelSha256: process.env.LLAMA_WEBGPU_SMOKE_MODEL_SHA256,
    modelCacheDir: process.env.LLAMA_WEBGPU_SMOKE_MODEL_CACHE || DEFAULT_MODEL_CACHE,
  });
  const { chromium } = await import('playwright');
  await withTempDir('shared-core-staging-', async (webRoot) => {
    const cores = [
      'llama_webgpu_core',
      'llama_webgpu_core_mem64',
      'llama_webgpu_core_st',
      'llama_webgpu_core_mem64_st',
    ];
    const artifacts = ['llama_webgpu_bridge.js', ...cores.flatMap((core) => [`${core}.js`, `${core}.wasm`])];
    for (const artifact of artifacts) {
      await stageFile(path.join(distDir, artifact), path.join(webRoot, artifact));
    }
    await stageFile(model, path.join(webRoot, 'tiny.gguf'));
    await fs.writeFile(path.join(webRoot, 'index.html'), '<!doctype html>');

    const browser = await chromium.launch({
      args: [...webGpuLaunchArgs(process.platform, { sharedArrayBuffer: false }), '--enable-automation'],
    });
    try {
      const session = await browser.newBrowserCDPSession();
      const commandLine = await session.send('Browser.getBrowserCommandLine');
      assert.ok(!commandLine.arguments.some((arg) => arg.includes('SharedArrayBuffer')));
      for (const coreName of cores) {
        const isolated = !coreName.endsWith('_st');
        await withServer(webRoot, async (url) => {
          const page = await browser.newPage();
          try {
            await page.goto(url);
            for (const [issue, replaceMount] of [[236, false], [230, true]]) {
              await t.test(`#${issue} on ${coreName}`, async () => {
                const result = await page.evaluate(loadOnSharedCore, { coreName, isolated, replaceMount });
                assert.deepEqual(result.mountPaths, ['/jsmodels-1', '/jsmodels-2']);
                assert.ok(result.tokens.length > 0);
                assert.deepEqual(result.unmountPaths, ['/jsmodels-2']);
                assert.deepEqual(result.remainingPaths, replaceMount ? [] : ['jsmodels-1']);
                assert.equal(result.remainingId, replaceMount ? 0 : result.originalId);
                assert.equal(result.allocatedBytes, 0);
              });
            }
          } finally {
            await page.close();
          }
        }, { isolated });
      }
    } finally {
      await browser.close();
    }
  });
});
