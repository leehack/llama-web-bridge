import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { copyFileSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporaryDir = mkdtempSync(path.join(tmpdir(), 'llama-web-bridge-types-'));

try {
  copyFileSync(
    path.join(rootDir, 'js/src/llama_webgpu_bridge.d.ts'),
    path.join(temporaryDir, 'bridge.d.ts'),
  );
  writeFileSync(
    path.join(temporaryDir, 'tsconfig.json'),
    JSON.stringify({
      compilerOptions: {
        target: 'ES2022',
        module: 'ESNext',
        moduleResolution: 'Bundler',
        strict: true,
        noEmit: true,
        lib: ['ES2022', 'DOM'],
        types: [],
      },
      include: ['consumer.ts'],
    }),
  );
  writeFileSync(
    path.join(temporaryDir, 'consumer.ts'),
    [
      "import type { CompletionOptions, EmbedOptions } from './bridge.js';",
      '',
      'export const accepted: CompletionOptions = {',
      '  nPredict: 64,',
      '  mediaMaxPredict: 128,',
      "  parts: [{ type: 'image', url: 'a.png' }, { type: 'image', bytes: new DataView(new ArrayBuffer(3)) }, { type: 'audio', samples: new Float32Array(1) }],",
      '  temp: 0.7,',
      '  topK: 40,',
      '  topP: 0.95,',
      '  penalty: 1.1,',
      "  grammar: 'root ::= \"a\"',",
      '  seed: 7,',
      '};',
      '',
      'export const acceptedEmbed: EmbedOptions = { normalize: false };',
      '',
      '// @ts-expect-error temperature is not a supported completion option.',
      'export const rejectedTemperature: CompletionOptions = { temperature: 0.7 };',
      '// @ts-expect-error normalise is not a supported embedding option.',
      'export const rejectedNormalise: EmbedOptions = { normalise: false };',
      '// @ts-expect-error video parts are not supported.',
      "export const rejectedPart: CompletionOptions = { parts: [{ type: 'video' }] };",
      '// @ts-expect-error generic array-like objects are not supported media data.',
      "export const rejectedArrayLike: CompletionOptions = { parts: [{ type: 'image', bytes: { 0: 255, length: 1 } }] };",
      '// @ts-expect-error image parts require bytes or a URL.',
      "export const rejectedImageSource: CompletionOptions = { parts: [{ type: 'image' }] };",
      '// @ts-expect-error audio parts require samples, bytes, or a URL.',
      "export const rejectedAudioSource: CompletionOptions = { parts: [{ type: 'audio' }] };",
      '',
    ].join('\n'),
  );

  const compilation = spawnSync(
    process.execPath,
    [path.join(rootDir, 'node_modules/typescript/bin/tsc'), '-p', '.'],
    { cwd: temporaryDir, encoding: 'utf8' },
  );
  assert.equal(compilation.error, undefined, `failed to run tsc: ${compilation.error}`);
  assert.equal(
    compilation.status,
    0,
    `bridge declarations failed consumer compilation:\n${compilation.stdout}${compilation.stderr}`,
  );
} finally {
  rmSync(temporaryDir, { recursive: true, force: true });
}

console.log('Bridge type declaration contract tests passed');
