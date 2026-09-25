import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';

import {
  CANONICAL_MODEL_PIN_NAMES,
  EXPECTED_MODEL_PINS,
  PUBLICATION_PAT_GUARD_ERROR,
  PUBLICATION_PAT_NAME,
  ROOT,
  extractModelShaPinRoles,
  extractModelUrls,
  modelFileNameRoles,
  parseExpectedModelPins,
  requireCanonicalModelShaPins,
  requireIdenticalModelUrls,
  requireMarkdownModelPinRoles,
  requirePairedModelUrlsAndPins,
  requirePinnedModelUrlRevisions,
  requireQualificationModelShaPinRoles,
  validatePublicationPatContract,
} from '../../scripts/verify_ci_reliability.mjs';

// Contract tests for the CI reliability verifier: the role-aware model pin,
// URL and revision parity checks, and the publication PAT validator's
// adversarial fixtures.

const read = (relativePath) => readFileSync(path.join(ROOT, relativePath), 'utf8');

// EXPECTED_MODEL_PINS is read from release_qualification.py and fails closed.
{
  assert.equal(Object.keys(EXPECTED_MODEL_PINS).length, 8);
  for (const pin of Object.values(EXPECTED_MODEL_PINS)) assert.match(pin, /^[0-9a-f]{64}$/);
  assert.deepEqual(parseExpectedModelPins(read('scripts/release_qualification.py')), EXPECTED_MODEL_PINS);
  const source = read('scripts/release_qualification.py');
  assert.throws(
    () => parseExpectedModelPins(source.replace('    "tts_model_sha256": TTS_MODEL_SHA256,\n', '')),
    /declares 7 pins, expected 8/,
  );
  assert.throws(
    () => parseExpectedModelPins(source.replace('"tts_model_sha256": TTS_MODEL_SHA256', '"tts_model_sha256": UNKNOWN_SHA256')),
    /not a 64-hex constant/,
  );
  assert.throws(
    () => parseExpectedModelPins(source.replace('"tts_model_sha256": TTS_MODEL_SHA256', '"tts_model_sha256": TTS_MODEL_SHA256 + ""')),
    /cannot resolve/,
  );
  assert.throws(() => parseExpectedModelPins(''), /no EXPECTED_MODEL_PINS/);
}

// --- Model pin parity ------------------------------------------------------

const SPEECH_REVISION = '928ab958557df9aa2ef1c93e0e83c7ad0933fae2';
const TTS_REVISION = 'ca27d74bc954b73dadab5b71ca265d87fc861a7c';
const ASR_REPO = `https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF/resolve/${SPEECH_REVISION}`;
const TTS_REPO = `https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/${TTS_REVISION}`;
const MODEL_URLS = {
  LLAMA_WEBGPU_SMOKE_MODEL: 'https://huggingface.co/aladar/llama-2-tiny-random-GGUF/resolve/main/llama-2-tiny-random.gguf',
  LLAMA_WEBGPU_MULTIMODAL_MODEL: 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf',
  LLAMA_WEBGPU_MULTIMODAL_MMPROJ: 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-F16.gguf',
  LLAMA_WEBGPU_SPEECH_MODEL: `${ASR_REPO}/Qwen3-ASR-0.6B-Q8_0.gguf?download=true`,
  LLAMA_WEBGPU_SPEECH_MMPROJ: `${ASR_REPO}/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf?download=true`,
  LLAMA_WEBGPU_SPEECH_AUDIO: 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav',
  LLAMA_WEBGPU_TTS_MODEL: `${TTS_REPO}/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf?download=true`,
  LLAMA_WEBGPU_TTS_MMPROJ: `${TTS_REPO}/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf?download=true`,
};
const BUILD_ROLES = [
  'LLAMA_WEBGPU_SMOKE_MODEL',
  'LLAMA_WEBGPU_MULTIMODAL_MODEL',
  'LLAMA_WEBGPU_MULTIMODAL_MMPROJ',
  'LLAMA_WEBGPU_SPEECH_MODEL',
  'LLAMA_WEBGPU_SPEECH_MMPROJ',
  'LLAMA_WEBGPU_TTS_MODEL',
  'LLAMA_WEBGPU_TTS_MMPROJ',
];
const QUALIFICATION_ROLES = [
  'LLAMA_WEBGPU_SPEECH_MODEL',
  'LLAMA_WEBGPU_SPEECH_MMPROJ',
  'LLAMA_WEBGPU_SPEECH_AUDIO',
  'LLAMA_WEBGPU_TTS_MODEL',
  'LLAMA_WEBGPU_TTS_MMPROJ',
];
const DOCUMENTED_ROLES = [
  ['--model-url', MODEL_URLS.LLAMA_WEBGPU_SMOKE_MODEL, '--model-sha256', 'state_smoke_model_sha256'],
  ['--model-path', '/path/to/Qwen3.5-0.8B-Q4_K_M.gguf', '--model-sha256', 'multimodal_model_sha256'],
  ['--mmproj-path', '/path/to/mmproj-F16.gguf', '--mmproj-sha256', 'multimodal_mmproj_sha256'],
  ['--model-path', '/path/to/Qwen3-ASR-0.6B-Q8_0.gguf', '--model-sha256', 'speech_model_sha256'],
  ['--mmproj-path', '/path/to/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf', '--mmproj-sha256', 'speech_mmproj_sha256'],
  ['--model-path', '/path/to/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf', '--model-sha256', 'tts_model_sha256'],
  ['--mmproj-path', '/path/to/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf', '--mmproj-sha256', 'tts_mmproj_sha256'],
];
const CI_PATH = '.github/workflows/ci.yml';
const CANDIDATE_PATH = '.github/workflows/bridge_candidate.yml';
const QUALIFICATION_PATH = '.github/workflows/bridge_qualification.yml';

function workflow(roles, { urls = {}, pins = {} } = {}) {
  const resolvedUrls = { ...MODEL_URLS, ...urls };
  const resolvedPins = { ...EXPECTED_MODEL_PINS, ...pins };
  const lines = ['name: fixture', 'on: push', 'env:'];
  for (const role of roles) {
    const name = CANONICAL_MODEL_PIN_NAMES[`${role}_SHA256`];
    lines.push(`      ${role}_URL: ${resolvedUrls[role]}`);
    lines.push(`      ${role}_SHA256: ${resolvedPins[name]}`);
  }
  return `${lines.join('\n')}\n`;
}

function markdown({ pins = {}, roles = DOCUMENTED_ROLES } = {}) {
  const resolved = { ...EXPECTED_MODEL_PINS, ...pins };
  const lines = ['```bash', 'python3 scripts/example_browser_smoke.py \\'];
  for (const [roleFlag, value, pinFlag, name] of roles) {
    lines.push(`  ${roleFlag} ${value} \\`);
    lines.push(`  ${pinFlag} ${resolved[name]} \\`);
  }
  lines.push('  --artifacts-dir /tmp/example');
  lines.push('```');
  return `${lines.join('\n')}\n`;
}

function collectErrors({ ci, candidate, qualification, contributing } = {}) {
  const contents = {
    [CI_PATH]: ci ?? workflow(BUILD_ROLES),
    [CANDIDATE_PATH]: candidate ?? workflow(BUILD_ROLES),
    [QUALIFICATION_PATH]: qualification ?? workflow(QUALIFICATION_ROLES),
  };
  const errors = [];
  const roles = {};
  const urls = {};
  for (const [file, content] of Object.entries(contents)) roles[file] = extractModelShaPinRoles(file, content, errors);
  for (const [file, content] of Object.entries(contents)) urls[file] = extractModelUrls(file, content, errors);
  requireQualificationModelShaPinRoles(roles[QUALIFICATION_PATH], errors);
  requireCanonicalModelShaPins(roles, errors);
  requirePairedModelUrlsAndPins(urls, roles, errors);
  requireIdenticalModelUrls(urls, errors);
  requirePinnedModelUrlRevisions(urls, errors);
  requireMarkdownModelPinRoles('CONTRIBUTING.md', contributing ?? markdown(), modelFileNameRoles(urls, errors), errors);
  return errors;
}

function assertRejected(errors, fragment) {
  assert.ok(errors.length > 0, 'expected at least one error');
  assert.ok(errors.some((error) => error.includes(fragment)), `no error contained ${JSON.stringify(fragment)}: ${errors.join('\n')}`);
}

const without = (roles, dropped) => roles.filter((role) => role !== dropped);

// fixture baseline accepted
assert.deepEqual(collectErrors(), []);

// repository tree accepted
assert.deepEqual(collectErrors({
  ci: read(CI_PATH),
  candidate: read(CANDIDATE_PATH),
  qualification: read(QUALIFICATION_PATH),
  contributing: read('CONTRIBUTING.md'),
}), []);

// markdown-only role swap rejected
{
  const errors = collectErrors({ contributing: markdown({ pins: {
    speech_model_sha256: EXPECTED_MODEL_PINS.speech_mmproj_sha256,
    speech_mmproj_sha256: EXPECTED_MODEL_PINS.speech_model_sha256,
  } }) });
  assertRejected(errors, 'canonical speech_model_sha256 is');
  assertRejected(errors, 'canonical speech_mmproj_sha256 is');
}

// identical cross-workflow swap rejected
{
  const pins = {
    speech_model_sha256: EXPECTED_MODEL_PINS.tts_model_sha256,
    tts_model_sha256: EXPECTED_MODEL_PINS.speech_model_sha256,
  };
  const errors = collectErrors({ ci: workflow(BUILD_ROLES, { pins }), candidate: workflow(BUILD_ROLES, { pins }) });
  assertRejected(errors, 'binds LLAMA_WEBGPU_SPEECH_MODEL_SHA256');
  assertRejected(errors, 'binds LLAMA_WEBGPU_TTS_MODEL_SHA256');
}

// bridge_qualification pin drift rejected
assertRejected(
  collectErrors({ qualification: workflow(QUALIFICATION_ROLES, { pins: { speech_audio_sha256: '0'.repeat(64) } }) }),
  'canonical speech_audio_sha256 is',
);

// URL drift rejected
assertRejected(
  collectErrors({ candidate: workflow(BUILD_ROLES, { urls: {
    LLAMA_WEBGPU_MULTIMODAL_MMPROJ: 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-F32.gguf',
  } }) }),
  'LLAMA_WEBGPU_MULTIMODAL_MMPROJ_URL requests different bytes',
);

// revision drift rejected
assertRejected(
  collectErrors({ qualification: workflow(QUALIFICATION_ROLES, { urls: {
    LLAMA_WEBGPU_TTS_MODEL: MODEL_URLS.LLAMA_WEBGPU_TTS_MODEL.replace(TTS_REVISION, 'f'.repeat(40)),
  } }) }),
  'LLAMA_WEBGPU_TTS_MODEL_URL requests different bytes',
);

// dropped qualification role rejected
assertRejected(
  collectErrors({ qualification: workflow(without(QUALIFICATION_ROLES, 'LLAMA_WEBGPU_TTS_MMPROJ')) }),
  'declares model SHA-256 env keys',
);

// extra qualification role rejected
assertRejected(
  collectErrors({ qualification: workflow([...QUALIFICATION_ROLES, 'LLAMA_WEBGPU_SMOKE_MODEL']) }),
  'declares model SHA-256 env keys',
);

// newly unversioned URL rejected
{
  const urls = { LLAMA_WEBGPU_TTS_MODEL: 'https://mirror.example/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf' };
  assertRejected(collectErrors({
    ci: workflow(BUILD_ROLES, { urls }),
    candidate: workflow(BUILD_ROLES, { urls }),
    qualification: workflow(QUALIFICATION_ROLES, { urls }),
  }), 'carrying no revision segment');
}

// duplicate model URL rejected
assertRejected(
  collectErrors({ qualification: `${workflow(QUALIFICATION_ROLES)}      LLAMA_WEBGPU_SPEECH_AUDIO_URL: https://mirror.example/asr_en.wav\n` }),
  'redefines model URL env key',
);

// shared model file name rejected
{
  const urls = { LLAMA_WEBGPU_MULTIMODAL_MMPROJ: `${ASR_REPO}/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf?download=true` };
  assertRejected(
    collectErrors({ ci: workflow(BUILD_ROLES, { urls }), candidate: workflow(BUILD_ROLES, { urls }) }),
    'also downloads',
  );
}

// markdown missing documented pin rejected
assertRejected(
  collectErrors({ contributing: markdown({ roles: DOCUMENTED_ROLES.slice(0, -1) }) }),
  'documented model SHA-256 pins with a role',
);

// markdown duplicate role rejected
assertRejected(
  collectErrors({ contributing: markdown({ roles: [...DOCUMENTED_ROLES, DOCUMENTED_ROLES[0]] }) }),
  'pins canonical state_smoke_model_sha256 again',
);

// uncanonical pin name rejected
{
  CANONICAL_MODEL_PIN_NAMES.LLAMA_WEBGPU_ABSENT_MODEL_SHA256 = 'absent_model_sha256';
  let errors;
  try {
    errors = collectErrors();
  } finally {
    delete CANONICAL_MODEL_PIN_NAMES.LLAMA_WEBGPU_ABSENT_MODEL_SHA256;
  }
  assertRejected(errors, 'EXPECTED_MODEL_PINS does not declare');
}

// URL trailing comment accepted
{
  const audio = `      LLAMA_WEBGPU_SPEECH_AUDIO_URL: ${MODEL_URLS.LLAMA_WEBGPU_SPEECH_AUDIO}`;
  const content = workflow(QUALIFICATION_ROLES).replace(`${audio}\n`, `${audio}  # not a Hugging Face object\n`);
  assert.notEqual(content, workflow(QUALIFICATION_ROLES));
  assert.deepEqual(collectErrors({ qualification: content }), []);
}

// newly mutable revision rejected
{
  const urls = Object.fromEntries(['LLAMA_WEBGPU_SPEECH_MODEL', 'LLAMA_WEBGPU_SPEECH_MMPROJ']
    .map((role) => [role, MODEL_URLS[role].replace(SPEECH_REVISION, 'main')]));
  assertRejected(collectErrors({
    ci: workflow(BUILD_ROLES, { urls }),
    candidate: workflow(BUILD_ROLES, { urls }),
    qualification: workflow(QUALIFICATION_ROLES, { urls }),
  }), 'resolving through a mutable revision');
}

// missing canonical role rejected
assertRejected(
  collectErrors({ qualification: workflow(without(QUALIFICATION_ROLES, 'LLAMA_WEBGPU_SPEECH_AUDIO')) }),
  'speech_audio_sha256',
);

// unnamed role key rejected
assertRejected(
  collectErrors({ qualification: workflow(QUALIFICATION_ROLES).replace('LLAMA_WEBGPU_SPEECH_AUDIO_SHA256', 'LLAMA_WEBGPU_SPEECH_SAMPLE_SHA256') }),
  'CANONICAL_MODEL_PIN_NAMES does not name',
);

// URL without pin rejected
{
  const pin = `      LLAMA_WEBGPU_SPEECH_AUDIO_SHA256: ${EXPECTED_MODEL_PINS.speech_audio_sha256}\n`;
  assertRejected(
    collectErrors({ qualification: workflow(QUALIFICATION_ROLES).replace(pin, '') }),
    'with no matching <ROLE>_SHA256 pin',
  );
}

// pin without URL rejected
{
  const url = `      LLAMA_WEBGPU_SPEECH_AUDIO_URL: ${MODEL_URLS.LLAMA_WEBGPU_SPEECH_AUDIO}\n`;
  assertRejected(
    collectErrors({ qualification: workflow(QUALIFICATION_ROLES).replace(url, '') }),
    'with no matching <ROLE>_URL',
  );
}

// markdown pin without named file rejected
assertRejected(
  collectErrors({ contributing: markdown().replace('  --model-path /path/to/Qwen3-ASR-0.6B-Q8_0.gguf \\\n', '') }),
  'no preceding --model-url',
);

// markdown unknown file rejected
assertRejected(
  collectErrors({ contributing: markdown().replace('/path/to/Qwen3-ASR-0.6B-Q8_0.gguf', '/path/to/Qwen3-ASR-0.6B-Q4_K_M.gguf') }),
  'Qwen3-ASR-0.6B-Q4_K_M.gguf, which',
);

// --- Publication PAT contract ------------------------------------------------

const EXPECTED_STEPS = [['publish-assets', 'Safe publication']];

// The safe fixture binds the PAT through a YAML merge key to another variable.
assert.deepEqual(validatePublicationPatContract(`
env: &publication-env
  RELEASE_CREDENTIAL: '\${{ secrets.WEBGPU_BRIDGE_ASSETS_PAT }}'
jobs:
  publish-assets:
    steps:
      - name: Safe publication
        env:
          <<: *publication-env
        run: |
          set -euo pipefail
          # The credential guard must be the first executable block after setup.
          if [ -z "\${RELEASE_CREDENTIAL}" ]; then
            echo "error: WEBGPU_BRIDGE_ASSETS_PAT is required for asset publication"
            exit 1
          fi
          git -C assets push origin main
`, EXPECTED_STEPS), []);

{
  const canonicalGuard = `if [ -z "\${TOKEN_ALIAS}" ]; then\n  echo "${PUBLICATION_PAT_GUARD_ERROR}"\n  exit 1\nfi`;
  const canonicalScript = `set -euo pipefail\n${canonicalGuard}`;
  const guardError = 'canonical executable empty-token guard for TOKEN_ALIAS';
  const unsafeScripts = {
    'missing guard': ['set -euo pipefail\ntrue', guardError],
    'non-executing false-and-exit empty-token block': [`set -euo pipefail\nfalse && ${canonicalGuard}`, guardError],
    'single-quoted non-expanding parameter guard': [
      "set -euo pipefail\n: '${TOKEN_ALIAS:?WEBGPU_BRIDGE_ASSETS_PAT is required}'\ngit -C assets push origin main",
      guardError,
    ],
    'indirect gh invocation before guard': [
      `set -euo pipefail\nclient=gh\n"\${client}" api repos/example/assets\n${canonicalGuard}`,
      guardError,
    ],
    'set -x': [`${canonicalScript}\nset -x`, 'set -x/xtrace'],
    printenv: [`${canonicalScript}\nprintenv`, 'through printenv'],
    'bare env': [`${canonicalScript}\nenv`, 'through bare env'],
    'absolute-path env': [`${canonicalScript}\n/usr/bin/env`, 'through bare env'],
    'echo leakage': [`${canonicalScript}\necho "\${TOKEN_ALIAS}"`, 'echo/printf of TOKEN_ALIAS'],
    'printf leakage': [`${canonicalScript}\nprintf '%s' "$TOKEN_ALIAS"`, 'echo/printf of TOKEN_ALIAS'],
    'secret expansion in guard error': [
      'set -euo pipefail\nif [ -z "${TOKEN_ALIAS}" ]; then\n  echo "error: ${TOKEN_ALIAS} is required"\n  exit 1\nfi',
      guardError,
    ],
  };
  for (const [fixtureName, [script, expectedError]] of Object.entries(unsafeScripts)) {
    const indented = script.split('\n').map((line) => `          ${line}\n`).join('');
    const fixture = `
jobs:
  publish-assets:
    steps:
      - name: Safe publication
        env:
          TOKEN_ALIAS: "\${{secrets.WEBGPU_BRIDGE_ASSETS_PAT}}"
        run: |-
${indented}`;
    const errors = validatePublicationPatContract(fixture, EXPECTED_STEPS);
    assert.ok(
      errors.some((error) => error.includes(expectedError)),
      `the PAT contract did not reject ${fixtureName} for the expected reason (${expectedError}): ${errors.join('\n')}`,
    );
  }
}

// A job-level container credential is a PAT reference outside any step env.
{
  const errors = validatePublicationPatContract(`
jobs:
  publish-assets:
    container:
      image: example.invalid/publisher:latest
      credentials:
        username: publisher
        password: \${{ secrets.WEBGPU_BRIDGE_ASSETS_PAT }}
    steps:
      - name: Safe publication
        run: echo safe
`, EXPECTED_STEPS);
  assertRejected(errors, `job publish-assets properties references ${PUBLICATION_PAT_NAME} outside resolved env`);
}

// A PAT-bearing step outside the expected set is rejected, and so is a
// workflow the resolver cannot parse.
assertRejected(validatePublicationPatContract(`
jobs:
  publish-assets:
    steps:
      - name: Unexpected publication
        env:
          TOKEN: \${{ secrets.WEBGPU_BRIDGE_ASSETS_PAT }}
        run: |
          set -euo pipefail
          if [ -z "\${TOKEN}" ]; then
            echo "${PUBLICATION_PAT_GUARD_ERROR}"
            exit 1
          fi
`, EXPECTED_STEPS), 'PAT-bearing steps are');
assertRejected(validatePublicationPatContract('jobs: [unterminated', EXPECTED_STEPS), 'workflow YAML cannot be resolved');

console.log('CI reliability verifier tests passed');
