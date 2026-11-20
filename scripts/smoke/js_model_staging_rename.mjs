#!/usr/bin/env node

import assert from 'node:assert/strict';
import fsp from 'node:fs/promises';
import path from 'node:path';

import {
  copyBridgeArtifacts,
  copyMemory64Artifacts,
  copySingleThreadArtifacts,
  ensure,
  runMain,
  runPlaywright,
  validateHash,
  withServer,
  withTempDir,
} from './support.mjs';

async function checkRename(coreModule, scenario) {
  const { LlamaWebGpuBridge } = await import('/llama_webgpu_bridge.js');
  const { default: createCore } = await import(coreModule);
  const core = await createCore();
  const newBridge = () => new LlamaWebGpuBridge({
    disableWorker: true,
    jsBackedModelStaging: true,
    coreModuleFactory: async () => core,
  });
  const load = async (bridge, url) => {
    try {
      return await bridge.loadModelFromUrl(url, {
        nGpuLayers: 0,
        nCtx: 128,
        nThreads: 1,
        useCache: false,
        forceRemoteFetchBackend: false,
      });
    } catch (error) {
      return error.code ?? error.message;
    }
  };
  const original = '/jsmodels-1';
  const moved = '/lane230-moved';
  const restoresOwner = scenario === 'restore';
  const ordinaryFile = scenario === 'file';
  const ownerModel = restoresOwner ? '/owner.gguf' : '/tiny-00001-of-00002.gguf';
  const pausedModel = restoresOwner ? ownerModel : '/tiny-00002-of-00002.gguf';
  const { FS } = core;
  const native = (name, mountPath = original) => core.ccall(`llamadart_webgpu_${name}`, 'number', ['string'], [mountPath]);
  const allocated = () => core.ccall('llamadart_webgpu_js_model_fs_allocated_bytes', 'number', [], []);
  const shardContents = async (directory) => {
    const file = directory + ownerModel;
    try {
      const bytes = FS.readFile(file);
      const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)).toHex();
      return { size: bytes.length, digest };
    } catch (error) {
      return { size: FS.stat(file).size, errno: error.errno, error: error.message };
    }
  };

  const owner = newBridge();
  const other = newBridge();
  const waiting = Promise.withResolvers();
  const resume = Promise.withResolvers();
  const fetch = window.fetch;
  window.fetch = async (url, options) => {
    if (String(url).endsWith(pausedModel)) {
      waiting.resolve();
      await resume.promise;
    }
    return fetch(url, options);
  };

  const ownerLoad = load(owner, ownerModel);
  try {
    await Promise.race([waiting.promise, ownerLoad.then((result) => {
      throw new Error(`owner finished before the paused fetch: ${result}`);
    })]);
    if (restoresOwner) {
      const originalId = native('js_model_fs_mount_id');
      const empty = FS.readdir(original).every((name) => name === '.' || name === '..');
      FS.rename(original, moved);
      FS.mkdir(original);
      const otherResult = await load(other, '/tiny.gguf');
      FS.rmdir(original);
      FS.rename(moved, original);
      const restoredId = native('js_model_fs_mount_id');
      resume.resolve();
      const ownerResult = await ownerLoad;

      const roundtrip = '/lane230-roundtrip';
      const nativeMount = native('mount_js_model_fs', roundtrip);
      FS.rename(roundtrip, moved);
      FS.mkdir(roundtrip);
      const nativeConflict = native('mount_js_model_fs', roundtrip);
      FS.rmdir(roundtrip);
      FS.rename(moved, roundtrip);
      const nativeUnmount = native('unmount_js_model_fs', roundtrip);
      return { scenario, originalId, empty, otherResult, restoredId, ownerResult,
        nativeMount, nativeConflict, nativeUnmount, allocatedAfterOwner: allocated() };
    }

    const before = await shardContents(original);
    FS.rename(original, moved);
    FS.writeFile(`${moved}/application.bin`, new Uint8Array([12, 34, 56, 78]));
    if (ordinaryFile) FS.writeFile(original, new Uint8Array([42]));
    const allocatedBefore = allocated();

    const otherResult = await load(other, '/tiny.gguf');
    const after = await shardContents(moved);
    const allocatedAfter = allocated();
    let applicationBytes;
    try {
      applicationBytes = [...FS.readFile(`${moved}/application.bin`)];
    } catch (error) {
      applicationBytes = { errno: error.errno };
    }
    resume.resolve();
    const ownerResult = await ownerLoad;
    const replacementBytes = ordinaryFile ? [...FS.readFile(original)] : null;
    return { scenario, ordinaryFile, before, after, otherResult, applicationBytes, allocatedBefore,
      allocatedAfter, ownerResult, replacementBytes, allocatedAfterOwner: allocated() };
  } finally {
    window.fetch = fetch;
    resume.resolve();
    await ownerLoad;
    await owner.dispose();
    await other.dispose();
  }
}

async function main([dist, model, checksum, splitPrefix] = process.argv.slice(2)) {
  ensure(dist && model && checksum && splitPrefix,
    'usage: js_model_staging_rename.mjs DIST_DIR MODEL_PATH MODEL_SHA256 SPLIT_PREFIX');
  await validateHash(model, checksum);
  const results = await withTempDir('lane230-rename-', async (root) => {
    await copyBridgeArtifacts(dist, root);
    await copyMemory64Artifacts(dist, root);
    await copySingleThreadArtifacts(dist, root, { memory64: true });
    await fsp.copyFile(model, path.join(root, 'tiny.gguf'));
    await fsp.copyFile(model, path.join(root, 'owner.gguf'));
    for (const suffix of ['-00001-of-00002.gguf', '-00002-of-00002.gguf']) {
      await fsp.copyFile(`${splitPrefix}${suffix}`, path.join(root, `tiny${suffix}`));
    }

    const results = [];
    for (const variant of ['core', 'core_mem64', 'core_st', 'core_mem64_st']) {
      const coreModule = JSON.stringify(`/llama_webgpu_${variant}.js`);
      await fsp.writeFile(path.join(root, 'index.html'), `<script type="module">
        const checkRename = ${checkRename.toString()};
        try {
          const cases = [];
          for (const scenario of ['absent', 'file', 'restore']) {
            cases.push(await checkRename(${coreModule}, scenario));
          }
          window.__smokeResult = { ok: true, cases, isolated: crossOriginIsolated };
        } catch (error) {
          window.__smokeResult = { ok: false, error: error.stack };
        }
      </script>`);
      const isolated = !variant.endsWith('_st');
      const result = await withServer(root, (url) => runPlaywright(url, 60000, null), { isolated });
      ensure(result.ok, `${variant}: ${result.error}`);
      assert.equal(result.isolated, isolated);
      for (const scenario of result.cases) {
        results.push({ variant, ...scenario });
      }
    }
    return results;
  });
  for (const result of results) console.log(JSON.stringify(result));
  for (const result of results) {
    const label = `${result.variant}, ${result.scenario}`;
    if (result.scenario === 'restore') {
      assert.ok(result.empty && result.originalId > 0, label);
      assert.equal(result.otherResult, 'MODEL_STAGING_MOUNT_FAILED', label);
      assert.equal(result.restoredId, result.originalId, label);
      assert.equal(result.ownerResult, 1, label);
      assert.deepEqual([result.nativeMount, result.nativeConflict, result.nativeUnmount], [0, -20, 0], label);
      assert.equal(result.allocatedAfterOwner, 0, label);
      continue;
    }
    assert.equal(result.otherResult, result.ordinaryFile ? 'MODEL_STAGING_MOUNT_FAILED' : 1, label);
    assert.ok(result.before.size > 0 && result.before.digest, label);
    assert.deepEqual(result.after, result.before, label);
    assert.deepEqual(result.applicationBytes, [12, 34, 56, 78], label);
    assert.deepEqual(result.replacementBytes, result.ordinaryFile ? [42] : null, label);
    assert.ok(result.allocatedBefore > 0, label);
    assert.equal(result.allocatedAfter, result.allocatedBefore, label);
    assert.equal(result.ownerResult, 'MODEL_STAGING_MOUNT_FAILED', label);
    assert.equal(result.allocatedAfterOwner, 0, label);
  }
  return 0;
}

if (import.meta.main) await runMain('JS staging rename', main);
