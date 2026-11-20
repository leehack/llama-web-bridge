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

async function loadOnSharedCore({ coreName, isolated, scenario }) {
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
  const appPath = '/jsmodels-2';
  const appFile = `${appPath}/application.bin`;
  const movedAppPath = '/application-model';
  const replaceMount = scenario === 'replaced';
  const leaveBusy = scenario === 'busy';
  const appOwned = !replaceMount && !leaveBusy;
  const removedAppMount = scenario === 'removed application mount';
  const renamedAppMount = scenario === 'renamed application mount';
  const absentAppMount = removedAppMount || renamedAppMount;
  const nativeAppMount = scenario === 'application mount' || absentAppMount;
  const { FS } = core;
  const ccall = core.ccall.bind(core);
  const mountId = (path = mountPath) => ccall('llamadart_webgpu_js_model_fs_mount_id', 'number', ['string'], [path]);
  const allocatedBytes = () => ccall('llamadart_webgpu_js_model_fs_allocated_bytes', 'number', [], []);
  const mountAttempts = [];
  const unmountPaths = [];
  let busy = leaveBusy;
  // An intact WasmFS mount needs fault injection to return EBUSY.
  core.ccall = (name, returnType, argTypes, args, options) => {
    if (name === 'llamadart_webgpu_mount_js_model_fs') {
      const rc = ccall(name, returnType, argTypes, args, options);
      mountAttempts.push([args[0], rc]);
      return rc;
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
    check(mountAttempts[0]?.[0] === mountPath, `new core started at ${mountAttempts[0]?.[0]}`);
    if (leaveBusy) {
      originalId = mountId();
      check(unmountPaths.length === 2 && unmountPaths.every((entry) => entry === mountPath),
        'first bridge did not hit the injected busy condition twice');
    }
    if (!appOwned) check(originalId > 0, 'first bridge did not stage on a native mount');
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
  } else if (leaveBusy) {
    busy = false;
    check(mountId() === originalId, 'the busy mount did not survive disposal');
  } else {
    check(mountId() === 0 && !FS.readdir('/').includes('jsmodels-1'), 'first staging path was not removed');
    if (nativeAppMount) {
      check(ccall('llamadart_webgpu_mount_js_model_fs', 'number', ['string'], [appPath]) === 0,
        'failed to create the application mount');
    } else {
      FS.mkdir(appPath);
    }
    if (!removedAppMount) FS.writeFile(appFile, new Uint8Array([12, 34, 56, 78]));
  }

  const appSnapshot = () => appOwned && !removedAppMount ? {
    id: mountId(appPath),
    bytes: [...FS.readFile(appFile)],
    allocated: allocatedBytes(),
  } : null;
  const appBefore = appSnapshot();
  if (renamedAppMount) FS.rename(appPath, movedAppPath);
  if (removedAppMount) FS.rmdir(appPath);
  if (absentAppMount) {
    check(mountId(appPath) === 0 && !FS.readdir('/').includes('jsmodels-2'), 'application mount path is still visible');
  }
  const previousUnmounts = unmountPaths.length;
  const second = makeBridge();
  let tokens;
  try {
    check(await load(second) === 1, 'second bridge did not load the model');
    tokens = await second.tokenize('hello');
  } finally {
    await second.dispose();
  }
  const remainingPaths = FS.readdir('/').filter((entry) => entry.startsWith('jsmodels-'));
  if (renamedAppMount) FS.rename(movedAppPath, appPath);
  return {
    mountAttempts,
    tokens,
    unmountPaths: unmountPaths.slice(previousUnmounts),
    remainingPaths,
    remainingId: mountId(),
    originalId,
    allocatedBytes: allocatedBytes(),
    appBefore,
    appAfter: appSnapshot(),
  };
}

async function loadWithUnreadableRoot({ coreName, isolated, worker }) {
  if (crossOriginIsolated !== isolated || (typeof SharedArrayBuffer !== 'undefined') !== isolated) {
    throw new Error('unexpected isolation');
  }
  const { LlamaWebGpuBridge } = await import('/llama_webgpu_bridge.js');
  const coreModuleUrl = `${location.origin}/${coreName}-unreadable-root.mjs`;
  const wasmUrl = `${location.origin}/${coreName}.wasm`;
  const bridge = new LlamaWebGpuBridge({
    disableWorker: !worker,
    allowSingleThreadCore: !isolated,
    jsBackedModelStaging: true,
    preferMemory64: coreName.includes('_mem64'),
    coreModuleUrl,
    coreModuleUrlMem64: coreModuleUrl,
    wasmUrl,
    wasmUrlMem64: wasmUrl,
  });
  try {
    const result = await bridge.loadModelFromUrl('/tiny.gguf', {
      nCtx: 128, nGpuLayers: 0, nThreads: 1, useCache: false, forceRemoteFetchBackend: false,
    });
    return {
      result,
      tokens: await bridge.tokenize('hello'),
      execution: bridge.getModelMetadata()['llamadart.webgpu.execution'],
    };
  } finally {
    await bridge.dispose();
  }
}

test('shared-core staging survives leftover and application-owned mount paths', {
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
    const artifacts = [
      'llama_webgpu_bridge.js', 'llama_webgpu_bridge_worker.js',
      ...cores.flatMap((core) => [`${core}.js`, `${core}.wasm`]),
    ];
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
        await fs.writeFile(path.join(webRoot, `${coreName}-unreadable-root.mjs`), `
          import createCore from '/${coreName}.js';
          export default async (options) => {
            const core = await createCore(options);
            core.FS.chmod('/', 0o333);
            try {
              core.FS.readdir('/');
            } catch (error) {
              if (error.message === 'No such directory') return core;
              throw error;
            }
            throw new Error('root directory is still readable');
          };
        `);
        await withServer(webRoot, async (url) => {
          const page = await browser.newPage();
          try {
            await page.goto(url);
            const scenarios = [
              ['#236', 'busy', '/jsmodels-2', ['jsmodels-1']],
              ['#230', 'replaced', '/jsmodels-2', []],
              ['application directory', 'application directory', '/jsmodels-3', ['jsmodels-2']],
              ['application mount', 'application mount', '/jsmodels-3', ['jsmodels-2']],
              ['removed application mount', 'removed application mount', '/jsmodels-3', []],
              ['renamed application mount', 'renamed application mount', '/jsmodels-3', []],
            ];
            for (const [label, scenario, nextPath, remainingPaths] of scenarios) {
              await t.test(`${label} on ${coreName}`, async () => {
                const result = await page.evaluate(loadOnSharedCore, { coreName, isolated, scenario });
                const attempts = [['/jsmodels-1', 0], [nextPath, 0]];
                if (scenario === 'removed application mount' || scenario === 'renamed application mount') {
                  attempts.splice(1, 0, ['/jsmodels-2', -20]);
                }
                assert.deepEqual(result.mountAttempts, attempts);
                assert.ok(result.tokens.length > 0);
                assert.deepEqual(result.unmountPaths, [nextPath]);
                assert.deepEqual(result.remainingPaths, remainingPaths);
                assert.equal(result.remainingId, scenario === 'busy' ? result.originalId : 0);
                if (result.appBefore) {
                  assert.deepEqual(result.appBefore.bytes, [12, 34, 56, 78]);
                  assert.equal(result.appBefore.id > 0, scenario !== 'application directory');
                  assert.equal(result.appBefore.allocated > 0, scenario !== 'application directory');
                }
                assert.deepEqual(result.appAfter, result.appBefore);
                assert.equal(result.allocatedBytes, result.appBefore?.allocated ?? 0);
              });
            }
            for (const worker of [false, true]) {
              const execution = worker ? 'worker' : 'main-thread';
              await t.test(`unreadable root on ${coreName} (${execution})`, async () => {
                const result = await page.evaluate(loadWithUnreadableRoot, { coreName, isolated, worker });
                assert.deepEqual(result, { result: 1, tokens: [1, 22172], execution });
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
