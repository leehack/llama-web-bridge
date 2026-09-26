#!/usr/bin/env node
// Generate the checksummed bridge asset manifest from validated inputs,
// ported from scripts/generate_release_manifest.py (deleted at the harness
// 5.0.0 cutover).
//
// manifest.json and sha256sums.txt are published artifact bytes, so they stay
// byte-identical to what the Python generator wrote for the same inputs:
// manifest.json is json.dumps(manifest, indent=2, sort_keys=True) + "\n" and
// sha256sums.txt is "<sha256>  <name>" per artifact, newline-joined, plus "\n".

import { createHash } from 'node:crypto';

import {
  SystemExitError, parseArguments, progName, runCli,
} from './cli.mjs';
import {
  ASSETS_REPOSITORY,
  AUTOMATED_QUALIFICATION_REQUIRED,
  BRIDGE_REPOSITORY,
  ContractError,
  NATIVE_REPOSITORY,
  UNPROVEN_CAPABILITIES,
  requireCorrelationId,
  requireRepository,
  requireSha256,
  validateReleaseIdentity,
} from './contract.mjs';
import {
  PyException, isPyException, pyIsFile, pyJoinPath, pyJsonDumps, pyPath, pyReadBytes, pyTypeName,
  pyWriteText,
} from './json.mjs';

// generate_release_manifest re-exports these from release_contract.
export { AUTOMATED_QUALIFICATION_REQUIRED, UNPROVEN_CAPABILITIES };

export const ARTIFACTS = Object.freeze([
  'llama_webgpu_bridge.js',
  'llama_webgpu_bridge_worker.js',
  'llama_webgpu_bridge.d.ts',
  'llama_webgpu_core.js',
  'llama_webgpu_core.wasm',
  'llama_webgpu_core_mem64.js',
  'llama_webgpu_core_mem64.wasm',
]);

export const CAPABILITIES = Object.freeze({
  wasm32: true,
  memory64: true,
  state_persistence: Object.freeze({ direct: true, worker: true }),
  multimodal: Object.freeze({ direct: true, worker: true }),
  speech_to_text: Object.freeze({
    advertised: true,
    direct: true,
    worker: true,
    wasm32: true,
    memory64: true,
  }),
  text_to_speech: Object.freeze({
    advertised: true,
    direct: true,
    worker: true,
    wasm32: false,
    memory64: true,
  }),
});

// The candidate build never runs the heavy real-model ASR/TTS gates, so its
// manifest records the requirement its own run did not satisfy. Publication
// refuses to publish this artifact unless a verified attestation from the hosted
// qualification run binds its exact digest.
export const QUALIFICATION_GATES = Object.freeze({
  state_persistence: 'passed',
  multimodal: 'passed',
  speech_to_text: AUTOMATED_QUALIFICATION_REQUIRED,
  text_to_speech: AUTOMATED_QUALIFICATION_REQUIRED,
});

const RUN_ID_RE = /^[1-9][0-9]*$/u;
const HEX_DIGITS = '0123456789abcdef';

// len(value) != 40 or any(character not in "0123456789abcdef" ...), with
// Python's TypeError for a value without len() or a non-str character.
function commit(value, field) {
  const invalid = () => new ContractError(`${field} must be a lowercase full 40-character commit SHA`);
  if (typeof value === 'string') {
    const characters = Array.from(value);
    if (characters.length !== 40 || characters.some((character) => !HEX_DIGITS.includes(character))) throw invalid();
    return value;
  }
  if (!Array.isArray(value)) throw new PyException('TypeError', `object of type '${pyTypeName(value)}' has no len()`);
  if (value.length !== 40) throw invalid();
  for (const character of value) {
    if (typeof character !== 'string') {
      throw new PyException('TypeError', `'in <string>' requires string as left operand, not ${pyTypeName(character)}`);
    }
    if (!HEX_DIGITS.includes(character)) throw invalid();
  }
  return value;
}

function sha256Hex(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

// The validated identity fields of the parsed command line, raising what
// generate_release_manifest.main raises before it reads any artifact.
function validateArgs(args) {
  const release = validateReleaseIdentity(args.releaseTag, args.releaseRebuild, args.upstreamTag);
  requireRepository(args.assetsRepo, 'assets_repo');
  requireRepository(args.bridgeRepo, 'bridge_repo');
  requireRepository(args.upstreamRepo, 'upstream_repo');
  requireRepository(args.nativeRepo, 'native_repo');
  if (args.assetsRepo !== ASSETS_REPOSITORY) throw new ContractError(`assets_repo must be exactly ${ASSETS_REPOSITORY}`);
  if (args.bridgeRepo !== BRIDGE_REPOSITORY) throw new ContractError(`bridge_repo must be exactly ${BRIDGE_REPOSITORY}`);
  if (args.nativeRepo !== NATIVE_REPOSITORY) throw new ContractError(`native_repo must be exactly ${NATIVE_REPOSITORY}`);
  requireSha256(args.nativeManifestSha256, 'native_manifest_sha256');
  const correlationId = requireCorrelationId(args.orchestratorCorrelationId);
  if (typeof args.githubRunId !== 'string' || RUN_ID_RE.exec(args.githubRunId) === null) {
    throw new ContractError('github_run_id must be a positive decimal string');
  }
  const expectedRunUrl = `https://github.com/${BRIDGE_REPOSITORY}/actions/runs/${args.githubRunId}`;
  if (args.githubRunUrl !== expectedRunUrl) throw new ContractError(`github_run_url must be exactly ${expectedRunUrl}`);
  return {
    release,
    correlationId,
    bridgeCommit: commit(args.bridgeCommit, 'bridge_commit'),
    upstreamCommit: commit(args.upstreamCommit, 'upstream_commit'),
    nativeCommit: commit(args.nativeCommit, 'native_commit'),
  };
}

// The manifest and the two files' text from validated args and each
// artifact's { name, sha256, sizeBytes }, in ARTIFACTS order.
function render(args, identity, artifacts) {
  const { release, correlationId, bridgeCommit, upstreamCommit, nativeCommit } = identity;
  const files = {};
  const checksums = [];
  for (const { name, sha256, sizeBytes } of artifacts) {
    files[name] = { size_bytes: sizeBytes, sha256 };
    checksums.push(`${sha256}  ${name}`);
  }
  const manifest = {
    schema_version: 2,
    release_tag: release.tag,
    release_channel: release.channel,
    release_rebuild: release.rebuild,
    assets_repository: args.assetsRepo,
    bridge_repository: args.bridgeRepo,
    bridge_commit: bridgeCommit,
    upstream_repository: args.upstreamRepo,
    upstream_tag: args.upstreamTag,
    upstream_commit: upstreamCommit,
    native_repository: args.nativeRepo,
    native_release_tag: args.nativeReleaseTag,
    native_manifest_sha256: args.nativeManifestSha256,
    native_commit: nativeCommit,
    emscripten_version: args.emscriptenVersion,
    orchestrator_correlation_id: correlationId,
    github_run_id: args.githubRunId,
    github_run_url: args.githubRunUrl,
    qualification_gates: QUALIFICATION_GATES,
    unproven_capabilities: UNPROVEN_CAPABILITIES,
    capabilities: CAPABILITIES,
    artifacts: files,
    // Compatibility aliases are read by existing consumers. New tooling must
    // use the explicit schema-v2 names above.
    bridge_assets_tag: release.tag,
    source_repository: args.bridgeRepo,
    source_commit: bridgeCommit,
    llama_cpp_tag: args.upstreamTag,
    llama_cpp_commit: upstreamCommit,
    files,
  };
  return {
    manifest,
    'manifest.json': `${pyJsonDumps(manifest, { indent: 2, sortKeys: true })}\n`,
    'sha256sums.txt': `${checksums.join('\n')}\n`,
  };
}

// The manifest.json and sha256sums.txt text generate() writes for `args` and
// the recorded digest and size of each artifact ({ name, sha256, sizeBytes },
// in ARTIFACTS order), with the same validation, without reading any file.
export function renderManifestFiles(args, artifacts) {
  return render(args, validateArgs(args), artifacts);
}

// generate(args): `args` holds the parsed command line with camelCase keys
// (outDir, releaseTag, releaseRebuild, ...), as parseArguments returns it.
// Writes manifest.json and sha256sums.txt into outDir and returns the manifest.
export function generate(args) {
  const identity = validateArgs(args);
  const outDir = pyPath(String(args.outDir));
  const artifacts = [];
  for (const name of ARTIFACTS) {
    const path = pyJoinPath(outDir, name);
    if (!pyIsFile(path)) throw new ContractError(`required artifact is missing: ${name}`);
    const data = pyReadBytes(path);
    artifacts.push({ name, sha256: sha256Hex(data), sizeBytes: data.length });
  }
  const rendered = render(args, identity, artifacts);
  pyWriteText(pyJoinPath(outDir, 'manifest.json'), rendered['manifest.json']);
  pyWriteText(pyJoinPath(outDir, 'sha256sums.txt'), rendered['sha256sums.txt']);
  return rendered.manifest;
}

const required = (flag, type = 'str') => ({ flag, required: true, type });

export const OPTIONS = Object.freeze([
  required('--out-dir', 'path'),
  required('--release-tag'),
  required('--release-rebuild', 'int'),
  required('--assets-repo'),
  required('--bridge-repo'),
  required('--bridge-commit'),
  { flag: '--upstream-repo', required: false, type: 'str', default: 'ggml-org/llama.cpp' },
  required('--upstream-tag'),
  required('--upstream-commit'),
  required('--native-repo'),
  required('--native-release-tag'),
  required('--native-manifest-sha256'),
  required('--native-commit'),
  required('--emscripten-version'),
  required('--orchestrator-correlation-id'),
  required('--github-run-id'),
  required('--github-run-url'),
]);

// The CLI: `main(argv, write)` parses argv like generate_release_manifest.py's
// argparse parser; a contract or file error becomes `error: <message>` with
// exit status 1. It prints nothing on success.
export function main(argv, _write) {
  const args = parseArguments(argv, { prog: progName(import.meta.url), options: OPTIONS });
  try {
    generate(args);
  } catch (error) {
    if (error instanceof ContractError || isPyException(error, 'OSError')) throw new SystemExitError(`error: ${error.message}`);
    throw error;
  }
  return 0;
}

if (import.meta.main) runCli(main);
