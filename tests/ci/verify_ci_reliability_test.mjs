import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';

import { EXPECTED_MODEL_PINS as QUALIFICATION_MODEL_PINS } from '../../scripts/release/qualification.mjs';
import {
  CANONICAL_MODEL_PIN_NAMES,
  EMSCRIPTEN_VERIFIER,
  EXPECTED_MODEL_PINS,
  QUALIFICATION_MODULE,
  QUALIFY_COMMAND,
  RELEASE_CONTRACT_SUITES,
  RELEASE_MANIFEST_MODULE,
  collectErrors as collectRepositoryErrors,
  ORCHESTRATOR_COMMAND,
  ORCHESTRATOR_DIRECTORY,
  ORCHESTRATOR_ENTRY,
  PUBLICATION_PAT_GUARD_ERROR,
  PUBLICATION_PAT_NAME,
  RELEASE_CONTRACT_COMMAND,
  RELEASE_QUALIFICATION_COMMAND,
  ROOT,
  SANCTIONED_APPROVAL,
  TEST_COMMAND,
  Workflow,
  checkJsContractTestsRegistered,
  checkOrchestration,
  extractModelShaPinRoles,
  extractModelUrls,
  literalDispatchBooleans,
  modelFileNameRoles,
  orchestratorModules,
  parseExpectedModelPins,
  requireCanonicalModelShaPins,
  requireIdenticalModelUrls,
  requireMarkdownModelPinRoles,
  requirePairedModelUrlsAndPins,
  requirePinnedModelUrlRevisions,
  requireQualificationModelShaPinRoles,
  requireQualificationNodeHarness,
  validatePublicationPatContract,
} from '../../scripts/ci/verify_ci_reliability.mjs';

// Contract tests for the CI reliability verifier: the role-aware model pin,
// URL and revision parity checks, and the publication PAT validator's
// adversarial fixtures.

const read = (relativePath) => readFileSync(path.join(ROOT, relativePath), 'utf8');

// EXPECTED_MODEL_PINS is parsed from the JavaScript source of
// scripts/release/qualification.mjs, equals what that module exports, and
// fails closed.
{
  assert.equal(Object.keys(EXPECTED_MODEL_PINS).length, 8);
  for (const pin of Object.values(EXPECTED_MODEL_PINS)) assert.match(pin, /^[0-9a-f]{64}$/);
  const source = read(QUALIFICATION_MODULE);
  assert.deepEqual(parseExpectedModelPins(source), EXPECTED_MODEL_PINS);
  assert.deepEqual(EXPECTED_MODEL_PINS, { ...QUALIFICATION_MODEL_PINS });
  const entry = '  tts_model_sha256: TTS_MODEL_SHA256,\n';
  assert.ok(source.includes(entry));
  assert.throws(() => parseExpectedModelPins(source.replace(entry, '')), /declares 7 pins, expected 8/);
  assert.throws(
    () => parseExpectedModelPins(source.replace(entry, '  tts_model_sha256: UNKNOWN_SHA256,\n')),
    /maps tts_model_sha256 to UNKNOWN_SHA256, which is not a 64-hex constant/,
  );
  assert.throws(
    () => parseExpectedModelPins(source.replace(entry, "  tts_model_sha256: TTS_MODEL_SHA256 + '',\n")),
    /cannot resolve/,
  );
  assert.throws(
    () => parseExpectedModelPins(source.replace(entry, `${entry}  tts_model_sha256: TTS_MODEL_SHA256,\n`)),
    /names tts_model_sha256 twice/,
  );
  // A quoted key and a trailing comment are JavaScript the reader follows.
  assert.deepEqual(
    parseExpectedModelPins(source.replace(entry, "  'tts_model_sha256': TTS_MODEL_SHA256, // the TTS model\n")),
    EXPECTED_MODEL_PINS,
  );
  // A role mapped to another role's constant is the canonical value the
  // workflows are checked against, so the pin parity checks see the swap.
  const swapped = parseExpectedModelPins(source.replace(entry, '  tts_model_sha256: SPEECH_MODEL_SHA256,\n'));
  assert.equal(swapped.tts_model_sha256, EXPECTED_MODEL_PINS.speech_model_sha256);
  const constant = `export const TTS_MODEL_SHA256 = '${EXPECTED_MODEL_PINS.tts_model_sha256}';\n`;
  assert.ok(source.includes(constant));
  assert.throws(() => parseExpectedModelPins(source.replace(constant, `${constant}${constant}`)), /defines TTS_MODEL_SHA256 twice/);
  assert.throws(() => parseExpectedModelPins(source.replace(constant, constant.replace(/'[0-9a-f]{64}'/, "'abc'"))),
    /TTS_MODEL_SHA256, which is not a 64-hex constant/);
  assert.throws(() => parseExpectedModelPins(source.replace(constant, constant.replace('export const', 'const'))),
    /TTS_MODEL_SHA256, which is not a 64-hex constant/);
  assert.throws(() => parseExpectedModelPins(source.replace('export const EXPECTED_MODEL_PINS = Object.freeze({', 'export const EXPECTED_MODEL_PINS = ({')),
    /no export const EXPECTED_MODEL_PINS = Object.freeze/);
  assert.throws(() => parseExpectedModelPins(''), /no export const EXPECTED_MODEL_PINS/);
  // The retired Python syntax is not read.
  assert.throws(() => parseExpectedModelPins('TTS_MODEL_SHA256 = (\n    "' + 'a'.repeat(64) + '"\n)\nEXPECTED_MODEL_PINS = {\n    "tts_model_sha256": TTS_MODEL_SHA256,\n}\n'),
    /no export const EXPECTED_MODEL_PINS/);
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
  const lines = ['```bash', 'node scripts/smoke/example.mjs \\'];
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
  assertRejected(errors, 'that EXPECTED_MODEL_PINS in scripts/release/qualification.mjs does not declare');
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

// --- Qualification harness setup --------------------------------------------

// qualify runs the candidate's Node smokes: Node.js 24, npm ci and the
// Playwright Chromium in candidate-source come first, and pip Playwright never.
{
  const qualificationErrors = (text) => {
    const errors = [];
    requireQualificationNodeHarness(text, errors);
    return errors;
  };
  const real = read(QUALIFICATION_PATH);
  assert.deepEqual(qualificationErrors(real), []);
  const setupNode = '        uses: actions/setup-node@v4\n        with:\n          node-version: 24\n';
  const npmCi = '      - name: Install the candidate\'s locked npm dependencies\n        working-directory: candidate-source\n        run: npm ci --ignore-scripts\n';
  const chromium = '        working-directory: candidate-source\n        run: npx --no-install playwright install --only-shell chromium\n';
  for (const needle of [setupNode, npmCi, chromium, QUALIFY_COMMAND]) assert.ok(real.includes(needle), needle);
  assert.equal(real.split(setupNode).length - 1, 2);
  const qualifyStep = real.slice(real.indexOf('      - name: Run the heavy Qwen3-ASR and Qwen3-TTS gates'), real.indexOf('      - name: Re-verify the attestation'));
  const setupMessage = 'must set up Node.js 24, run npm ci --ignore-scripts and install the Playwright Chromium in candidate-source';
  for (const [label, text] of [
    ['no setup-node', real.replaceAll(setupNode, '        uses: actions/cache@v4\n        with:\n          node-version: 24\n')],
    ['Node 22', real.replaceAll(setupNode, setupNode.replace('24', '22'))],
    ['npm ci in the trusted checkout', real.replace(npmCi, npmCi.replace('        working-directory: candidate-source\n', ''))],
    ['npm ci with lifecycle scripts', real.replace(npmCi, npmCi.replace('npm ci --ignore-scripts', 'npm ci'))],
    ['Chromium from the trusted checkout', real.replace(chromium, chromium.replace('        working-directory: candidate-source\n', ''))],
    ['setup after qualify', real.replace(qualifyStep, '').replace('      - name: Re-verify the attestation', `${qualifyStep}      - name: Re-verify the attestation`)
      .replace(npmCi, '').replace('      - name: Upload verified', `${npmCi}\n      - name: Upload verified`)],
    ['no qualify', real.replace(QUALIFY_COMMAND, 'node candidate-source/scripts/release/qualification.mjs run-gates')],
    ['qualify from the trusted checkout', real.replace(QUALIFY_COMMAND, 'node scripts/release/qualification.mjs qualify')],
  ]) {
    assert.ok(qualificationErrors(text).some((error) => error.includes(setupMessage)), label);
  }
  assertRejected(
    qualificationErrors(real.replace(chromium, `${chromium}      - run: python3 -m pip install --user playwright==1.63.0\n`)),
    'must never install Python Playwright; found: "pip install"',
  );
}

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

// R1: npm test runs one glob over tests/, check:js runs npm test, and every
// other file under tests/ is a helper that a test imports.
{
  const packageJson = (scripts) => JSON.stringify({ scripts });
  const good = { 'check:js': 'npm run typecheck:js && npm test', test: TEST_COMMAND };
  const files = {
    'tests/bridge/a_test.mjs': "import { x } from './helper.mjs';\nconst late = await import(`./late.mjs`);",
    'tests/bridge/helper.mjs': "export { y as x } from './nested/deep_helper.mjs';",
    'tests/bridge/late.mjs': 'export {};',
    'tests/bridge/nested/deep_helper.mjs': "import '../../shared/bare.mjs';\nexport const y = 1;",
    'tests/bridge/nested/b_test.mjs': '',
    'tests/shared/bare.mjs': '',
  };
  const errorsFor = (json, testFiles) => {
    const errors = [];
    checkJsContractTestsRegistered(json, testFiles, errors);
    return errors;
  };
  assert.deepEqual(errorsFor(packageJson(good), files), []);
  assert.deepEqual(errorsFor(packageJson({ ...good, 'check:js': 'npm run test' }), files), []);
  assert.match(errorsFor(packageJson({ ...good, test: "node --test 'tests/bridge/*_test.mjs'" }), files).join('\n'), /npm test must be exactly/);
  assert.match(errorsFor(packageJson({ ...good, test: `${TEST_COMMAND} || true` }), files).join('\n'), /npm test must be exactly/);
  assert.match(errorsFor(packageJson({ ...good, 'check:js': 'npm run typecheck:js' }), files).join('\n'), /check:js must run npm test/);
  assert.match(errorsFor(packageJson({ ...good, 'check:js': 'npm test || true' }), files).join('\n'), /check:js must run npm test/);
  assert.match(errorsFor('{', files).join('\n'), /package.json is not valid JSON/);
  assert.match(errorsFor(packageJson(good), {}).join('\n'), /no \*_test\.mjs contract tests/);
  const unreached = (extra) => errorsFor(packageJson(good), { ...files, ...extra })
    .map((error) => /^(\S+) is neither/.exec(error)?.[1]);
  // A misnamed test, and a helper imported only by an unrun file, never run.
  assert.deepEqual(unreached({
    'tests/bridge/c.test.mjs': "import './orphan_helper.mjs';",
    'tests/bridge/orphan_helper.mjs': '',
  }), ['tests/bridge/c.test.mjs', 'tests/bridge/orphan_helper.mjs']);
  // Naming a file in a string or path list, or importing a same-named file in
  // another directory, does not reach it.
  assert.deepEqual(unreached({
    'tests/bridge/d_test.mjs': "const listed = ['tests/bridge/e.test.mjs', './e.test.mjs'];",
    'tests/bridge/e.test.mjs': '',
    'tests/other/helper.mjs': '',
  }), ['tests/bridge/e.test.mjs', 'tests/other/helper.mjs']);
  // An interpolated dynamic import cannot be resolved statically.
  assert.deepEqual(unreached({
    'tests/bridge/f_test.mjs': 'await import(`./${name}.mjs`);',
    'tests/bridge/g.mjs': '',
  }), ['tests/bridge/g.mjs']);
}

// --- Orchestration ---------------------------------------------------------

// The scan workflow runs the Node orchestrator and release CLIs, and the
// orchestrator source is the entry plus every module it imports. Each check
// is proven against a mutated copy of the real workflow and sources.
{
  const AUTO_UPDATE = '.github/workflows/auto_llama_cpp_update.yml';
  const workflowText = read(AUTO_UPDATE);
  const sourceFiles = Object.fromEntries(
    readdirSync(path.join(ROOT, ORCHESTRATOR_DIRECTORY))
      .filter((name) => name.endsWith('.mjs'))
      .map((name) => [`${ORCHESTRATOR_DIRECTORY}/${name}`, read(`${ORCHESTRATOR_DIRECTORY}/${name}`)]),
  );
  const sources = (files = sourceFiles) => {
    const errors = [];
    const modules = orchestratorModules(files, errors);
    return { modules, errors };
  };
  const orchestrationErrors = (text = workflowText, modules = sources().modules) => {
    const errors = [];
    const workflow = new Workflow(AUTO_UPDATE, text, errors);
    checkOrchestration(workflow, modules, errors);
    return errors;
  };
  const mutate = (old, replacement, text = workflowText) => {
    assert.ok(text.includes(old), `the workflow no longer contains ${JSON.stringify(old)}`);
    return text.split(old).join(replacement);
  };

  // The real workflow and sources pass, and the source is the entry first,
  // then every module by path.
  assert.deepEqual(sources().errors, []);
  assert.deepEqual(Object.keys(sources().modules), [
    ORCHESTRATOR_ENTRY,
    ...Object.keys(sourceFiles).filter((file) => file !== ORCHESTRATOR_ENTRY).sort(),
  ]);
  assert.deepEqual(orchestrationErrors(), []);

  // Each command needle: the Python command it replaced is rejected.
  for (const [nodeCommand, pythonCommand, fragment] of [
    [`${RELEASE_QUALIFICATION_COMMAND} verify-run`, 'python3 scripts/release_qualification.py verify-run',
      `with ${RELEASE_QUALIFICATION_COMMAND} verify-run`],
    [`${RELEASE_CONTRACT_COMMAND} validate-environment`, 'python3 scripts/release_contract.py validate-environment',
      `with ${RELEASE_CONTRACT_COMMAND} validate-environment`],
    [`${RELEASE_CONTRACT_COMMAND} resolve-tag-commit`, 'python3 scripts/release_contract.py resolve-tag-commit',
      `"${RELEASE_CONTRACT_COMMAND} resolve-tag-commit"`],
    [`${RELEASE_CONTRACT_COMMAND} validate-native-release`, 'python3 scripts/release_contract.py validate-native-release',
      `"${RELEASE_CONTRACT_COMMAND} validate-native-release"`],
    [`${ORCHESTRATOR_COMMAND} resolve-bridge-source`, 'python3 scripts/stable_release_orchestrator.py resolve-bridge-source',
      `"${ORCHESTRATOR_COMMAND} resolve-bridge-source"`],
    [`${ORCHESTRATOR_COMMAND} scan-native`, 'python3 scripts/stable_release_orchestrator.py scan-native',
      `"${ORCHESTRATOR_COMMAND} scan-native"`],
    [`${ORCHESTRATOR_COMMAND} orchestrate-backlog`, 'python3 scripts/stable_release_orchestrator.py orchestrate-backlog',
      `may hand the PAT only to ${ORCHESTRATOR_COMMAND} orchestrate-backlog`],
  ]) {
    assertRejected(orchestrationErrors(mutate(nodeCommand, pythonCommand)), fragment);
  }
  assertRejected(orchestrationErrors(mutate(`${ORCHESTRATOR_COMMAND} \\\n                select-stable-native-backlog`,
    'python3 scripts/stable_release_orchestrator.py \\\n                select-stable-native-backlog')),
  `"${ORCHESTRATOR_COMMAND} select-stable-native-backlog"`);
  // The Python orchestrator is never named again, even beside the Node commands.
  assertRejected(orchestrationErrors(mutate('set -euo pipefail\n          dry_run_flag=()',
    'set -euo pipefail\n          python3 scripts/stable_release_orchestrator.py --help\n          dry_run_flag=()')),
  'never run the deleted Python orchestrator');
  assertRejected(orchestrationErrors(`${workflowText}# scripts/release_orchestrator_driver.py\n`),
    'never run the deleted Python orchestrator');
  // Dispatch goes only through the orchestrator entry.
  assertRejected(orchestrationErrors(mutate('set -euo pipefail\n          dry_run_flag=()',
    'set -euo pipefail\n          gh workflow run bridge_candidate.yml\n          dry_run_flag=()')),
  'dispatch only through scripts/release/orchestrator/cli.mjs');
  assertRejected(orchestrationErrors(mutate(`${ORCHESTRATOR_COMMAND} orchestrate-backlog`, 'node scripts/release/other.mjs orchestrate-backlog')),
    `may hand the PAT only to ${ORCHESTRATOR_COMMAND} orchestrate-backlog`);
  // The environment is validated with the job token before the PAT step.
  assertRejected(orchestrationErrors(mutate('          GH_TOKEN: ${{ github.token }}\n        run: |\n          set -euo pipefail\n          gh api "repos/${BRIDGE_REPO}/environments',
    '          GH_TOKEN: ${{ secrets.OTHER }}\n        run: |\n          set -euo pipefail\n          gh api "repos/${BRIDGE_REPO}/environments')),
  'may use the PAT only inside the bridge-assets-publication environment');
  assertRejected(orchestrationErrors(mutate('    environment:\n      name: bridge-assets-publication\n', '')),
    'may use the PAT only inside the bridge-assets-publication environment');
  assertRejected(orchestrationErrors(mutate('WEBGPU_BRIDGE_ASSETS_PAT: ${{ secrets.WEBGPU_BRIDGE_ASSETS_PAT }}',
    'RELEASE_CREDENTIAL: ${{ secrets.WEBGPU_BRIDGE_ASSETS_PAT }}')), `bind the publication PAT only as ${PUBLICATION_PAT_NAME}`);

  // Node.js 24 is set up without a cache before each job's first node
  // command, and nothing installs packages.
  const setupNode = '      - uses: actions/setup-node@v4\n        with:\n          node-version: 24\n';
  const setupCount = workflowText.split(setupNode).length - 1;
  assert.equal(setupCount, 2);
  const lastSetup = workflowText.lastIndexOf(setupNode);
  const withoutAdvanceSetup = workflowText.slice(0, lastSetup) + workflowText.slice(lastSetup + setupNode.length);
  assertRejected(orchestrationErrors(withoutAdvanceSetup), 'job advance_stable_release must set up Node.js 24');
  assertRejected(orchestrationErrors(mutate(setupNode, '')), 'job prepare_release_candidate must set up Node.js 24');
  assertRejected(orchestrationErrors(mutate('node-version: 24', 'node-version: 22')), 'must set up Node.js 24');
  assertRejected(orchestrationErrors(mutate(setupNode, `${setupNode}          cache: npm\n`)), 'without a package cache');
  // Set up only after the first node command.
  assertRejected(orchestrationErrors(mutate(`${setupNode}\n      - name: Prove the exact workflow continuation before environment use\n`,
    '      - name: Prove the exact workflow continuation before environment use\n')
    .replace('      - name: Resolve native-aligned candidate\n', `${setupNode}\n      - name: Resolve native-aligned candidate\n`)),
  'job prepare_release_candidate must set up Node.js 24 with actions/setup-node@v4 before its first node command');
  // Another major version could turn on package caching without a cache input.
  assertRejected(orchestrationErrors(mutate('actions/setup-node@v4', 'actions/setup-node@v5')), 'with actions/setup-node@v4');
  for (const install of ['npm ci', 'npm install', 'npx --no-install playwright', 'pip install yaml', 'corepack enable']) {
    assertRejected(orchestrationErrors(mutate('set -euo pipefail\n          dry_run_flag=()', `set -euo pipefail\n          ${install}\n          dry_run_flag=()`)),
      'must never install or run packages');
  }

  // The orchestrator source: an unimported module, an import of a missing
  // one, an import outside the orchestrator and the shared release modules,
  // or a non-literal import() fails; so does a missing entry.
  const extra = `${ORCHESTRATOR_DIRECTORY}/unimported.mjs`;
  assertRejected(sources({ ...sourceFiles, [extra]: 'export const x = 1;\n' }).errors, `unreached: ${extra}`);
  const driver = `${ORCHESTRATOR_DIRECTORY}/driver.mjs`;
  assertRejected(sources({ ...sourceFiles, [driver]: `import './missing.mjs';\n${sourceFiles[driver]}` }).errors,
    `imports ${ORCHESTRATOR_DIRECTORY}/missing.mjs, which does not exist`);
  assertRejected(sources({ ...sourceFiles, [driver]: `import '../../ci/ci_scope.mjs';\n${sourceFiles[driver]}` }).errors,
    'imports scripts/ci/ci_scope.mjs, which is neither');
  assertRejected(sources({ ...sourceFiles, [driver]: `import '../qualify.mjs';\n${sourceFiles[driver]}` }).errors,
    'imports scripts/release/qualify.mjs, which is neither');
  assertRejected(sources({ ...sourceFiles, [driver]: `const m = await import(name);\n${sourceFiles[driver]}` }).errors,
    'has an import() whose target is not a string literal');
  // A module reached only through an unreached one is itself unreached.
  const cut = { ...sourceFiles, [ORCHESTRATOR_ENTRY]: sourceFiles[ORCHESTRATOR_ENTRY].replace("import { advancePipeline } from './driver.mjs';\n", '') };
  assert.notEqual(cut[ORCHESTRATOR_ENTRY], sourceFiles[ORCHESTRATOR_ENTRY]);
  assert.ok(sources(cut).errors.some((error) => /unreached: /.test(error) && error.split('unreached: ')[1].split(', ').includes(driver)),
    sources(cut).errors.join('\n'));
  const withoutEntry = { ...sourceFiles };
  delete withoutEntry[ORCHESTRATOR_ENTRY];
  assertRejected(sources(withoutEntry).errors, `${ORCHESTRATOR_ENTRY} is missing`);
  assertRejected(orchestrationErrors(workflowText, {}), `must start at ${ORCHESTRATOR_ENTRY}`);

  // Row 53: every JavaScript spelling of a literal 'true' governance or
  // approval boolean is found, in any orchestrator module.
  for (const key of ['assets_immutable_releases_enabled', 'publish_approved']) {
    for (const literal of [
      `inputs.${key} = 'true';`,
      `inputs.${key}='true'`,
      `inputs . ${key} = "true"`,
      `inputs['${key}'] = 'true';`,
      `inputs["${key}"] = \`true\``,
      `const inputs = { ${key}: 'true' };`,
      `{${key}:"true"}`,
      `"${key}": "true"`,
      `'${key}': 'true'`,
      `new Map([['${key}', 'true']])`,
      `inputs.set("${key}", 'true')`,
    ]) {
      assert.notDeepEqual(literalDispatchBooleans(literal), [], literal);
      const errors = orchestrationErrors(workflowText, { ...sources().modules, [driver]: `${sourceFiles[driver]}\n${literal}\n` });
      assertRejected(errors, `${driver} must derive the governance and approval booleans only from live proofs`);
    }
    for (const allowed of [
      `inputs.${key} = governance.enabled === true ? 'true' : 'false';`,
      `if (inputs.${key} === 'true') {}`,
      `inputs.${key} == 'true'`,
      `other_${key}: 'true'`,
      `inputs.${key}_extra = 'true'`,
      `'${key}',`,
      `${key}: 'false'`,
    ]) {
      assert.deepEqual(literalDispatchBooleans(allowed), [], allowed);
    }
  }
  // The approval is sanctioned only in the driver, exactly once, on the
  // statement right after the live environment proof.
  const proof = "    requirePublicationEnvironment(gateway);\n    inputs.publish_approved = 'true';";
  assert.ok(sourceFiles[driver].includes(proof));
  assert.equal(sourceFiles[driver].match(SANCTIONED_APPROVAL).length, 1);
  assert.deepEqual(literalDispatchBooleans(sourceFiles[driver], { sanctioned: true }), []);
  assert.deepEqual(literalDispatchBooleans(sourceFiles[driver]), [".publish_approved = 'true'"]);
  assert.deepEqual(orchestrationErrors(workflowText, sources().modules), []);
  const unproven = sourceFiles[driver].replace(proof, "    inputs.publish_approved = 'true';");
  assert.deepEqual(literalDispatchBooleans(unproven, { sanctioned: true }), [".publish_approved = 'true'"]);
  for (const [label, driverText, expected] of [
    ['a commented-out proof', sourceFiles[driver].replace(proof, "    // requirePublicationEnvironment(gateway);\n    inputs.publish_approved = 'true';"),
      'must derive the governance and approval booleans only from live proofs'],
    ['a conditional proof', sourceFiles[driver].replace(proof, "    if (false) requirePublicationEnvironment(gateway);\n    inputs.publish_approved = 'true';"),
      'must derive the governance and approval booleans only from live proofs'],
    ['a second sanctioned approval', `${sourceFiles[driver]}\n${proof}\n`, 'must assert publish_approved exactly once'],
    ['no approval at all', sourceFiles[driver].replace(proof, '    requirePublicationEnvironment(gateway);'), 'must assert publish_approved exactly once'],
  ]) {
    assert.notEqual(driverText, sourceFiles[driver], label);
    assertRejected(orchestrationErrors(workflowText, { ...sources().modules, [driver]: driverText }), expected);
  }
  // The same two statements in any other module are not sanctioned.
  const planner = 'scripts/release/orchestrator/planner.mjs';
  assertRejected(
    orchestrationErrors(workflowText, { ...sources().modules, [planner]: `${sourceFiles[planner]}\n${proof}\n` }),
    `${planner} must derive the governance and approval booleans only from live proofs`,
  );
  // The import walk reads every module the orchestrator can load.
  for (const [label, line, expected] of [
    ['a package import', "import yaml from 'yaml';", 'may import only node: builtins and relative modules'],
    ['a subpath import', "import x from '#internal';", 'may import only node: builtins and relative modules'],
    ['an absolute URL import', "const x = await import('file:///tmp/x.mjs');", 'may import only node: builtins and relative modules'],
    ['createRequire', "import { createRequire } from 'node:module';\nconst load = createRequire(import.meta.url);", 'loads modules through require'],
    ['a bare require', "const x = require('./x.cjs');", 'loads modules through require'],
  ]) {
    const { errors } = sources({ ...sourceFiles, [planner]: `${sourceFiles[planner]}\n${line}\n` });
    assertRejected(errors, expected);
    assert.ok(errors.every((error) => error.startsWith(planner)), label);
  }
}

// --- Release workflow commands ---------------------------------------------

// Every command needle of the candidate, qualification, publish and CI
// checks, proven against a mutated copy of the real file: the tree passes,
// and each mutation (most of them the Python command the Node one replaced)
// is rejected for its own reason.
{
  const PUBLISH_PATH = '.github/workflows/publish_assets.yml';
  const BUILD_SCRIPT = 'scripts/build/build_bridge.sh';
  assert.deepEqual(collectRepositoryErrors(), []);
  // Replace every occurrence of `old` (or only the first, with once) in the
  // real file and return the verifier's errors.
  const mutated = (relativePath, old, replacement, { once = false } = {}) => {
    const text = read(relativePath);
    assert.ok(text.includes(old), `${relativePath} no longer contains ${JSON.stringify(old)}`);
    const changed = once ? text.replace(old, () => replacement) : text.split(old).join(replacement);
    return collectRepositoryErrors({ [relativePath]: changed });
  };
  const cases = [
    // No workflow runs Python (the R2 rule that replaced "CI runs the Python
    // unittest suite").
    [CANDIDATE_PATH, 'node scripts/release/contract.mjs validate-release', 'python3 scripts/release_contract.py validate-release',
      'runs Python ("python3")'],
    [CI_PATH, 'run: node scripts/ci/verify_ci_reliability.mjs', "run: python3 -m unittest discover -s scripts -p '*_test.py'",
      'runs Python ("python3")'],
    [CI_PATH, 'run: node scripts/ci/verify_ci_reliability.mjs', 'run: python -m py_compile x',
      'runs Python ("python")'],
    [PUBLISH_PATH, '          npm ci --ignore-scripts\n', '          npm ci --ignore-scripts\n          pip install -r requirements.txt\n',
      'runs Python ("pip")'],
    [CANDIDATE_PATH, './scripts/build/build_bridge.sh', './scripts/build/build_bridge.py', 'runs Python ("./scripts/build/build_bridge.py")'],
    [CI_PATH, '      - uses: actions/setup-node@v4\n        with:\n          node-version: 24\n      - id: scope',
      '      - uses: actions/setup-python@v5\n        with:\n          node-version: 24\n      - id: scope', 'runs Python (actions/setup-python@v5)'],
    // R2: CI runs this contract.
    [CI_PATH, 'run: node scripts/ci/verify_ci_reliability.mjs', 'run: true', 'must run the CI reliability contract'],
    // R3: candidate and publish run the release suites through check:js.
    [CANDIDATE_PATH, '          npm run check:js\n', '', 'run the release contract suites (through npm run check:js)'],
    [PUBLISH_PATH, '          node scripts/ci/verify_ci_reliability.mjs\n', '', 'run the release contract suites (through npm run check:js)'],
    // Rows 44-46: the Emscripten pin, its verifier and the manifest record.
    [CI_PATH, 'node scripts/build/verify_emscripten_version.mjs --print-pin', 'python3 scripts/verify_emscripten_version.py --print-pin',
      'resolve emsdk.version and verify the resolved emcc version before building'],
    [CANDIDATE_PATH, 'node scripts/build/verify_emscripten_version.mjs --emit-github-env', 'python3 scripts/verify_emscripten_version.py --emit-github-env',
      'resolve emsdk.version and verify the resolved emcc version before building'],
    [EMSCRIPTEN_VERIFIER, "spawnSync('emcc', ['--version']", "spawnSync('emcc', ['-v']", 'the Emscripten verifier must compare emcc'],
    [EMSCRIPTEN_VERIFIER, 'if (resolved !== expected)', 'if (false)', 'the Emscripten verifier must compare emcc'],
    [EMSCRIPTEN_VERIFIER, '`EMSCRIPTEN_VERSION=${resolved}\\n`', '`EMSCRIPTEN_VERSION=${expected}\\n`', 'the Emscripten verifier must compare emcc'],
    [EMSCRIPTEN_VERIFIER, "const PIN_NAME = 'emsdk.version';", "const PIN_NAME = 'emsdk.txt';", 'the Emscripten verifier must compare emcc'],
    [BUILD_SCRIPT, 'node "$BRIDGE_DIR/scripts/build/verify_emscripten_version.mjs"', 'python3 "$BRIDGE_DIR/scripts/verify_emscripten_version.py"',
      'and gate direct builds'],
    [BUILD_SCRIPT, 'node "$BRIDGE_DIR/scripts/build/verify_emscripten_version.mjs"\n', '', 'and gate direct builds'],
    // configure echoed before the gate: the gate no longer runs first
    [BUILD_SCRIPT, 'node "$BRIDGE_DIR/scripts/build/verify_emscripten_version.mjs"\n',
      'echo "[bridge] configuring with emcmake"\nnode "$BRIDGE_DIR/scripts/build/verify_emscripten_version.mjs"\n', 'and gate direct builds'],
    [RELEASE_MANIFEST_MODULE, 'emscripten_version: args.emscriptenVersion,', "emscripten_version: '6.0.8',",
      'must record the runtime-verified Emscripten compiler version'],
    [CANDIDATE_PATH, 'node scripts/release/manifest.mjs \\', 'python3 scripts/generate_release_manifest.py \\',
      'must record the runtime-verified Emscripten compiler version'],
    // The environment validators: the trusted checkout before approval, the
    // publication-policy checkout after it.
    [PUBLISH_PATH, 'node scripts/release/contract.mjs validate-environment', 'python3 scripts/release_contract.py validate-environment',
      'job verify-publication-environment must run after the approval check'],
    [PUBLISH_PATH, 'node publication-policy/scripts/release/contract.mjs validate-environment', 'python3 publication-policy/scripts/release_contract.py validate-environment',
      'revalidate the environment policy with github.token and the trusted publication-policy validator'],
    [PUBLISH_PATH, 'node publication-policy/scripts/release/contract.mjs validate-environment', 'node scripts/release/contract.mjs validate-environment',
      'revalidate the environment policy with github.token and the trusted publication-policy validator'],
    // Rows 73-76: exact provenance.
    [PUBLISH_PATH, 'node scripts/release/contract.mjs resolve-tag-commit', 'python3 scripts/release_contract.py resolve-tag-commit',
      'must resolve the upstream and native tag commits itself', { once: true }],
    [PUBLISH_PATH, 'node scripts/release/contract.mjs validate-native-request', 'python3 scripts/release_contract.py validate-native-request',
      'must validate native tag, upstream commit, and manifest SHA-256 inputs before the first native network request'],
    [PUBLISH_PATH, 'node scripts/release/contract.mjs validate-native-release', 'python3 scripts/release_contract.py validate-native-release',
      `missing: "${RELEASE_CONTRACT_COMMAND} validate-native-release"`],
    [PUBLISH_PATH, 'node scripts/release/contract.mjs require-correlation-id', 'true',
      'the correlation id and both repositories'],
    [PUBLISH_PATH, '--repository "${NATIVE_REPO}" --field native_repo', '--repository "${NATIVE_REPO}" --field assets_repo',
      'the correlation id and both repositories'],
    [PUBLISH_PATH, 'node publication-policy/scripts/release/contract.mjs resolve-tag-commit', 'python3 publication-policy/scripts/release_contract.py resolve-tag-commit',
      'must resolve the existing asset tag with the strict trusted parser', { once: true }],
    // Rows 78-80: classification, governance and readback, all with the
    // trusted publication-policy CLIs.
    [PUBLISH_PATH, 'node publication-policy/scripts/release/publication_state.mjs classify', 'python3 publication-policy/scripts/release_publication_state.py classify',
      'must re-query the remote state after a failed ref mutation'],
    [PUBLISH_PATH, 'node publication-policy/scripts/release/publication_state.mjs classify', 'node bridge-source/scripts/release/publication_state.mjs classify',
      'never rebuild the candidate or run a validator from the historical build source'],
    [PUBLISH_PATH, 'node publication-policy/scripts/release/publication_state.mjs state-changed', 'python3 publication-policy/scripts/release_publication_state.py state-changed',
      'publication_state.mjs state-changed"'],
    [PUBLISH_PATH, 'node publication-policy/scripts/release/publication_state.mjs mutation-unknown', 'python3 publication-policy/scripts/release_publication_state.py mutation-unknown',
      'publication_state.mjs mutation-unknown"'],
    [PUBLISH_PATH, 'node publication-policy/scripts/release/publication_state.mjs validate-target', 'python3 publication-policy/scripts/release_publication_state.py validate-target',
      'publication_state.mjs validate-target"'],
    [PUBLISH_PATH, '&& node publication-policy/scripts/release/publication_state.mjs \\\n              verify-immutable-publication',
      '&& python3 publication-policy/scripts/release_publication_state.py \\\n              verify-immutable-publication',
      'publication_state.mjs verify-immutable-publication"'],
    [PUBLISH_PATH, 'node publication-policy/scripts/release/contract.mjs \\\n            validate-immutable-release-governance',
      'python3 publication-policy/scripts/release_contract.py \\\n            validate-immutable-release-governance',
      'must prove immutable-release governance through the repository API before the first ref mutation'],
    [PUBLISH_PATH, 'node publication-policy/scripts/release/contract.mjs \\\n                 validate-immutable-release-governance',
      'python3 publication-policy/scripts/release_contract.py \\\n                 validate-immutable-release-governance',
      'must prove immutable-release governance through the repository API before the first ref mutation'],
    // Rows 87-91: verify-run, verify-attestation, extraction and qualify.
    [PUBLISH_PATH, 'node scripts/release/qualification.mjs verify-attestation', 'python3 scripts/release_qualification.py verify-attestation',
      'job verify-candidate-and-qualification must verify the attestation with node scripts/release/qualification.mjs verify-attestation'],
    [PUBLISH_PATH, 'node publication-policy/scripts/release/qualification.mjs verify-attestation', 'node bridge-source/scripts/release/qualification.mjs verify-attestation',
      'job publish-assets must verify the attestation with node publication-policy/scripts/release/qualification.mjs verify-attestation'],
    [PUBLISH_PATH, 'node scripts/release/qualification.mjs verify-run', 'python3 scripts/release_qualification.py verify-run',
      `missing: "${RELEASE_QUALIFICATION_COMMAND} verify-run"`],
    [PUBLISH_PATH, 'node scripts/release/qualification.mjs extract-artifact --type candidate', 'node scripts/release/archive.mjs --type candidate',
      'download all three artifacts by immutable id'],
    [PUBLISH_PATH, 'extract-artifact --type attestation', 'extract-artifact --type candidate', 'download all three artifacts by immutable id'],
    [PUBLISH_PATH, 'extract-artifact --type prequalification', 'extract-artifact --type attestation', 'download all three artifacts by immutable id'],
    [PUBLISH_PATH, '--harness-dir bridge-source/scripts)"', '--harness-dir scripts)"', 'download all three artifacts by immutable id'],
    [PUBLISH_PATH, 'node scripts/release/contract.mjs validate-candidate-prequalification', 'true', 'download all three artifacts by immutable id'],
    [PUBLISH_PATH, 'node bridge-source/scripts/build/verify_emscripten_version.mjs --print-pin', 'node scripts/build/verify_emscripten_version.mjs --print-pin',
      'may run only node bridge-source/scripts/build/verify_emscripten_version.mjs --print-pin'],
    [PUBLISH_PATH, 'node scripts/release/contract.mjs validate-release', 'node bridge-source/scripts/release/contract.mjs validate-release',
      'never rebuild the candidate or run a validator from the historical build source'],
    [PUBLISH_PATH, 'node bridge-source/scripts/build/verify_emscripten_version.mjs --print-pin', 'node bridge-source/scripts/release/manifest.mjs --print-pin',
      'never rebuild the candidate or run a validator from the historical build source'],
    [QUALIFICATION_PATH, 'node scripts/release/qualification.mjs verify-run', 'python3 scripts/release_qualification.py verify-run',
      `missing: "${RELEASE_QUALIFICATION_COMMAND} verify-run"`],
    [QUALIFICATION_PATH, 'node scripts/release/qualification.mjs extract-artifact --type candidate', 'node scripts/release/qualification.mjs extract-artifact --type attestation',
      'bind the attestation to the exact candidate artifact id'],
    [QUALIFICATION_PATH, 'node scripts/release/qualification.mjs verify-attestation', 'node candidate-source/scripts/release/qualification.mjs verify-attestation',
      `missing: "${RELEASE_QUALIFICATION_COMMAND} verify-attestation"`],
    [QUALIFICATION_PATH, 'node scripts/release/qualification.mjs candidate-fingerprint', 'python3 scripts/release_qualification.py candidate-fingerprint',
      `missing: "${RELEASE_QUALIFICATION_COMMAND} candidate-fingerprint"`],
    [QUALIFICATION_PATH, QUALIFY_COMMAND, 'python3 candidate-source/scripts/release_qualification.py qualify', `missing: "${QUALIFY_COMMAND}`],
    [QUALIFICATION_PATH, '--harness-dir candidate-source/scripts', '--harness-dir scripts', 'missing: "--harness-dir candidate-source/scripts"'],
    // Node.js 24 before each job's first node command; the privileged publish
    // jobs without a package cache and without packages.
    [QUALIFICATION_PATH, '      - name: Setup Node.js for the trusted contract\n        uses: actions/setup-node@v4\n        with:\n          node-version: 24\n\n', '',
      `${QUALIFICATION_PATH} job qualify must set up Node.js 24 with actions/setup-node@v4 before its first node command`],
    [CANDIDATE_PATH, 'uses: actions/setup-node@v4', 'uses: actions/setup-node@v5',
      `${CANDIDATE_PATH} job build-candidate must set up Node.js 24`],
    [CI_PATH, '      - uses: actions/setup-node@v4\n        with:\n          node-version: 24\n      - id: scope', '      - id: scope',
      `${CI_PATH} job changes must set up Node.js 24`],
    [PUBLISH_PATH, '      - name: Setup Node.js\n        uses: actions/setup-node@v4\n        with:\n          node-version: 24\n      - name: Fail closed', '      - name: Fail closed',
      `${PUBLISH_PATH} job verify-publication-environment must set up Node.js 24`],
    [PUBLISH_PATH, '      - name: Setup Node.js\n        uses: actions/setup-node@v4\n        with:\n          node-version: 24\n\n      - name: Download gated', '      - name: Download gated',
      `${PUBLISH_PATH} job publish-assets must set up Node.js 24 with actions/setup-node@v4 before its first node command, without a package cache`],
    [PUBLISH_PATH, '        with:\n          node-version: 24\n\n      - name: Download gated', '        with:\n          node-version: 24\n          cache: npm\n\n      - name: Download gated',
      `${PUBLISH_PATH} job publish-assets must set up Node.js 24 with actions/setup-node@v4 before its first node command, without a package cache`],
    [PUBLISH_PATH, '          node-version: 24\n      - name: Fail closed', '          node-version: 22\n      - name: Fail closed',
      `${PUBLISH_PATH} job verify-publication-environment must set up Node.js 24`],
    [PUBLISH_PATH, '          set -euo pipefail\n          node publication-policy/scripts/release/qualification.mjs verify-attestation',
      '          set -euo pipefail\n          npm ci --ignore-scripts\n          node publication-policy/scripts/release/qualification.mjs verify-attestation',
      `${PUBLISH_PATH} job publish-assets must never install or run packages`],
    [PUBLISH_PATH, '          set -euo pipefail\n          gh api "repos/${BRIDGE_REPO}/environments/bridge-assets-publication" \\\n            > "${RUNNER_TEMP}/publication-environment.json"',
      '          set -euo pipefail\n          npx --yes some-package\n          gh api "repos/${BRIDGE_REPO}/environments/bridge-assets-publication" \\\n            > "${RUNNER_TEMP}/publication-environment.json"',
      `${PUBLISH_PATH} job verify-publication-environment must never install or run packages`],
  ];
  for (const [relativePath, old, replacement, fragment, options] of cases) {
    const errors = mutated(relativePath, old, replacement, options);
    const label = `${relativePath}: ${JSON.stringify(old)} -> ${JSON.stringify(replacement)}`;
    assert.ok(
      errors.some((error) => error.includes(fragment)),
      `${label}: no error contained ${JSON.stringify(fragment)}: ${errors.join('\n') || '(no errors)'}`,
    );
  }
  // R3: each release contract suite must exist under tests/, where npm test
  // runs it.
  assert.ok(RELEASE_CONTRACT_SUITES.length >= 9);
  for (const suite of RELEASE_CONTRACT_SUITES) {
    assertRejected(collectRepositoryErrors({ [suite]: null }), `the release contract suites npm run check:js must run are missing: ${suite}`);
  }
}

console.log('CI reliability verifier tests passed');
