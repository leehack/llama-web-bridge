// The real v0.1.53 candidate manifest, for tests/release/manifest_test.mjs.
//
// Source: candidate run 36233412322, artifact exact-webgpu-bridge-dist (id
// 10903558022), fetched read-only with
//   gh api repos/leehack/llama-web-bridge/actions/artifacts/10903558022/zip
// MANIFEST_JSON and SHA256SUMS are its manifest.json and sha256sums.txt as
// scripts/generate_release_manifest.py wrote them (their SHA-256 digests are
// below, so the text here is provably those bytes). ARGV is the generator
// command line that run passed, --out-dir aside; ARTIFACTS holds the SHA-256
// and size measured from each built file in the artifact, in ARTIFACTS order.
// The binaries themselves are not checked in.

export const MANIFEST_JSON_SHA256 = '2777ac575d4a93ea8b5a6f086bf9132f7333fe50f12cc2a507423b7ba3f1c072';
export const SHA256SUMS_SHA256 = '0e0f29dd33f05da5976fc3445bc724a58612d8b63364b9a1af92179f840395de';

export const ARGV = Object.freeze([
  '--release-tag', 'v0.1.53',
  '--release-rebuild', '0',
  '--assets-repo', 'leehack/llama-web-bridge-assets',
  '--bridge-repo', 'leehack/llama-web-bridge',
  '--bridge-commit', '6a3054a2b1140eff593c7bce2fb0cc90f12323a7',
  '--upstream-tag', 'v0.5.0',
  '--upstream-commit', '7fe450e19305b828c199d602c23a8337aaa1f03b',
  '--native-repo', 'leehack/llamadart-native',
  '--native-release-tag', 'v0.5.0',
  '--native-manifest-sha256', '1ded54a5b3d687dfddde33586acd89bcd5b774ab4301197194b63a7e05541c4b',
  '--native-commit', 'aa8ceb95a99f299587a9b0ba5c0e9b6f829bcd64',
  '--emscripten-version', '6.0.8',
  '--orchestrator-correlation-id', 'auto-stable-v0.5.0-1ded54a5b3d687df-build-6a3054a2b1140eff',
  '--github-run-id', '36233412322',
  '--github-run-url', 'https://github.com/leehack/llama-web-bridge/actions/runs/36233412322',
]);

export const ARTIFACTS = Object.freeze([
  { name: 'llama_webgpu_bridge.js', sha256: 'b61b900137454ed67698a3b2f2f7dfeb5d97b0b704f5df33f1de1531b3d06703', sizeBytes: 238606 },
  { name: 'llama_webgpu_bridge_worker.js', sha256: '47bbfa0fe897e708455b9497e9aa41d2e94489c329300c33b12936fb3169deca', sizeBytes: 257 },
  { name: 'llama_webgpu_bridge.d.ts', sha256: 'e8076a2286fce0a9cbc0daa2fb5034b605b339dd580fcbf1dddbe3c03400bfc6', sizeBytes: 10036 },
  { name: 'llama_webgpu_core.js', sha256: '9dc4dd9bbc96897a9140b5dc3616da0e57f4daa0f51aa7feb135529d9f74a2dd', sizeBytes: 119625 },
  { name: 'llama_webgpu_core.wasm', sha256: '7037ab387d2997727ba72b8ee5776e3c18950c568223e7bae8e329b723cbbb6e', sizeBytes: 9347052 },
  { name: 'llama_webgpu_core_mem64.js', sha256: '7c7a0290ca4baa919111a2452e3669f42b6d7e95247af4c6621f06ea99b83489', sizeBytes: 141012 },
  { name: 'llama_webgpu_core_mem64.wasm', sha256: 'e4221033b29c1113a6e575d55ad3bfc83f37ef48a555b4db43c1d9c24f4bc902', sizeBytes: 9598517 },
]);

export const MANIFEST_JSON = `{
  "artifacts": {
    "llama_webgpu_bridge.d.ts": {
      "sha256": "e8076a2286fce0a9cbc0daa2fb5034b605b339dd580fcbf1dddbe3c03400bfc6",
      "size_bytes": 10036
    },
    "llama_webgpu_bridge.js": {
      "sha256": "b61b900137454ed67698a3b2f2f7dfeb5d97b0b704f5df33f1de1531b3d06703",
      "size_bytes": 238606
    },
    "llama_webgpu_bridge_worker.js": {
      "sha256": "47bbfa0fe897e708455b9497e9aa41d2e94489c329300c33b12936fb3169deca",
      "size_bytes": 257
    },
    "llama_webgpu_core.js": {
      "sha256": "9dc4dd9bbc96897a9140b5dc3616da0e57f4daa0f51aa7feb135529d9f74a2dd",
      "size_bytes": 119625
    },
    "llama_webgpu_core.wasm": {
      "sha256": "7037ab387d2997727ba72b8ee5776e3c18950c568223e7bae8e329b723cbbb6e",
      "size_bytes": 9347052
    },
    "llama_webgpu_core_mem64.js": {
      "sha256": "7c7a0290ca4baa919111a2452e3669f42b6d7e95247af4c6621f06ea99b83489",
      "size_bytes": 141012
    },
    "llama_webgpu_core_mem64.wasm": {
      "sha256": "e4221033b29c1113a6e575d55ad3bfc83f37ef48a555b4db43c1d9c24f4bc902",
      "size_bytes": 9598517
    }
  },
  "assets_repository": "leehack/llama-web-bridge-assets",
  "bridge_assets_tag": "v0.1.53",
  "bridge_commit": "6a3054a2b1140eff593c7bce2fb0cc90f12323a7",
  "bridge_repository": "leehack/llama-web-bridge",
  "capabilities": {
    "memory64": true,
    "multimodal": {
      "direct": true,
      "worker": true
    },
    "speech_to_text": {
      "advertised": true,
      "direct": true,
      "memory64": true,
      "wasm32": true,
      "worker": true
    },
    "state_persistence": {
      "direct": true,
      "worker": true
    },
    "text_to_speech": {
      "advertised": true,
      "direct": true,
      "memory64": true,
      "wasm32": false,
      "worker": true
    },
    "wasm32": true
  },
  "emscripten_version": "6.0.8",
  "files": {
    "llama_webgpu_bridge.d.ts": {
      "sha256": "e8076a2286fce0a9cbc0daa2fb5034b605b339dd580fcbf1dddbe3c03400bfc6",
      "size_bytes": 10036
    },
    "llama_webgpu_bridge.js": {
      "sha256": "b61b900137454ed67698a3b2f2f7dfeb5d97b0b704f5df33f1de1531b3d06703",
      "size_bytes": 238606
    },
    "llama_webgpu_bridge_worker.js": {
      "sha256": "47bbfa0fe897e708455b9497e9aa41d2e94489c329300c33b12936fb3169deca",
      "size_bytes": 257
    },
    "llama_webgpu_core.js": {
      "sha256": "9dc4dd9bbc96897a9140b5dc3616da0e57f4daa0f51aa7feb135529d9f74a2dd",
      "size_bytes": 119625
    },
    "llama_webgpu_core.wasm": {
      "sha256": "7037ab387d2997727ba72b8ee5776e3c18950c568223e7bae8e329b723cbbb6e",
      "size_bytes": 9347052
    },
    "llama_webgpu_core_mem64.js": {
      "sha256": "7c7a0290ca4baa919111a2452e3669f42b6d7e95247af4c6621f06ea99b83489",
      "size_bytes": 141012
    },
    "llama_webgpu_core_mem64.wasm": {
      "sha256": "e4221033b29c1113a6e575d55ad3bfc83f37ef48a555b4db43c1d9c24f4bc902",
      "size_bytes": 9598517
    }
  },
  "github_run_id": "36233412322",
  "github_run_url": "https://github.com/leehack/llama-web-bridge/actions/runs/36233412322",
  "llama_cpp_commit": "7fe450e19305b828c199d602c23a8337aaa1f03b",
  "llama_cpp_tag": "v0.5.0",
  "native_commit": "aa8ceb95a99f299587a9b0ba5c0e9b6f829bcd64",
  "native_manifest_sha256": "1ded54a5b3d687dfddde33586acd89bcd5b774ab4301197194b63a7e05541c4b",
  "native_release_tag": "v0.5.0",
  "native_repository": "leehack/llamadart-native",
  "orchestrator_correlation_id": "auto-stable-v0.5.0-1ded54a5b3d687df-build-6a3054a2b1140eff",
  "qualification_gates": {
    "multimodal": "passed",
    "speech_to_text": "required-automated-qualification",
    "state_persistence": "passed",
    "text_to_speech": "required-automated-qualification"
  },
  "release_channel": "stable",
  "release_rebuild": 0,
  "release_tag": "v0.1.53",
  "schema_version": 2,
  "source_commit": "6a3054a2b1140eff593c7bce2fb0cc90f12323a7",
  "source_repository": "leehack/llama-web-bridge",
  "unproven_capabilities": {
    "hardware_gpu_acceleration": "unavailable-on-hosted-runners",
    "real_device_intelligibility": "unproven",
    "real_device_playback": "unproven",
    "speaker_reference_fidelity": "unproven",
    "wasm32_text_to_speech": "unsupported"
  },
  "upstream_commit": "7fe450e19305b828c199d602c23a8337aaa1f03b",
  "upstream_repository": "ggml-org/llama.cpp",
  "upstream_tag": "v0.5.0"
}
`;

export const SHA256SUMS = `b61b900137454ed67698a3b2f2f7dfeb5d97b0b704f5df33f1de1531b3d06703  llama_webgpu_bridge.js
47bbfa0fe897e708455b9497e9aa41d2e94489c329300c33b12936fb3169deca  llama_webgpu_bridge_worker.js
e8076a2286fce0a9cbc0daa2fb5034b605b339dd580fcbf1dddbe3c03400bfc6  llama_webgpu_bridge.d.ts
9dc4dd9bbc96897a9140b5dc3616da0e57f4daa0f51aa7feb135529d9f74a2dd  llama_webgpu_core.js
7037ab387d2997727ba72b8ee5776e3c18950c568223e7bae8e329b723cbbb6e  llama_webgpu_core.wasm
7c7a0290ca4baa919111a2452e3669f42b6d7e95247af4c6621f06ea99b83489  llama_webgpu_core_mem64.js
e4221033b29c1113a6e575d55ad3bfc83f37ef48a555b4db43c1d9c24f4bc902  llama_webgpu_core_mem64.wasm
`;
