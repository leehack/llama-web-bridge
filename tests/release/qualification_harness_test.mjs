// Tests of scripts/release/qualification.mjs's harness digest and speech
// fixture, one test per test method of scripts/release_qualification_test.py
// with the same name and assertions (see qualification_fixtures.mjs for the
// mapping), plus digests over sub-path source names and Path.resolve().

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';

import { ContractError } from '../../scripts/release/contract.mjs';
import { pyJsonDumps } from '../../scripts/release/json.mjs';
import { pyExpanduser, pyNormpath, pyResolve } from '../../scripts/release/python_compat.mjs';
import {
  EXPECTED_MODEL_PINS,
  EXPECTED_SPEECH_TRANSCRIPT,
  HARNESS_SOURCES,
  HARNESS_VERSION,
  QUALIFICATION_SMOKES,
  SPEECH_AUDIO_SHA256,
  SPEECH_FIXTURE,
  SPEECH_FIXTURE_FILE,
  harnessSourceSha256,
  harnessSourceSha256AtCommit,
  loadSpeechFixture,
  normalizeTranscript,
  requireHarnessMatchesBridgeSource,
} from '../../scripts/release/qualification.mjs';
import { SCRIPTS_DIR, makeTempDir } from './qualification_fixtures.mjs';

function raises(fn, message = undefined) {
  let caught;
  assert.throws(fn, (error) => {
    caught = error;
    return error instanceof ContractError;
  }, message);
  return caught.message;
}

function withTmp(fn) {
  const tmp = makeTempDir();
  try {
    return fn(tmp);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

function git(repository, ...args) {
  const result = spawnSync('git', ['-C', repository, ...args], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  return result.stdout;
}

// A git repository whose single commit holds `files` ({ 'scripts/x': data }).
function commitRepository(repository, files) {
  for (const [name, data] of Object.entries(files)) {
    fs.mkdirSync(path.dirname(path.join(repository, name)), { recursive: true });
    fs.writeFileSync(path.join(repository, name), data);
  }
  git(repository, 'init', '-q');
  git(repository, 'add', 'scripts');
  git(repository, '-c', 'user.name=test', '-c', 'user.email=test@example.com', '-c', 'commit.gpgsign=false', 'commit', '-q', '-m', 'harness');
  return git(repository, 'rev-parse', 'HEAD').trim();
}

test('test_harness_digest_covers_every_heavy_gate_source', () => withTmp((tmp) => {
  assert.ok(HARNESS_SOURCES.includes('release/contract.mjs'));
  assert.ok(HARNESS_SOURCES.includes('release/publication_state.mjs'));
  const baseline = harnessSourceSha256(SCRIPTS_DIR);
  for (const name of HARNESS_SOURCES) {
    const mirror = path.join(tmp, `mirror-${name.replaceAll('/', '-')}`);
    fs.mkdirSync(mirror);
    for (const other of HARNESS_SOURCES) {
      fs.mkdirSync(path.dirname(path.join(mirror, other)), { recursive: true });
      fs.copyFileSync(path.join(SCRIPTS_DIR, other), path.join(mirror, other));
    }
    fs.writeFileSync(path.join(mirror, name), Buffer.concat([fs.readFileSync(path.join(SCRIPTS_DIR, name)), Buffer.from('\n# drift\n')]));
    assert.notEqual(baseline, harnessSourceSha256(mirror), name);
  }
}));

test('test_speech_fixture_holds_the_pinned_audio_and_transcript', () => {
  const fixture = JSON.parse(fs.readFileSync(path.join(SCRIPTS_DIR, SPEECH_FIXTURE_FILE), 'utf8'));
  assert.deepEqual({ ...SPEECH_FIXTURE }, fixture);
  // The smoke's default audio is the fixture qualification pins.
  assert.equal(fixture.audio_sha256, SPEECH_AUDIO_SHA256);
  assert.equal(EXPECTED_MODEL_PINS.speech_audio_sha256, fixture.audio_sha256);
  assert.equal(EXPECTED_SPEECH_TRANSCRIPT, normalizeTranscript(fixture.expected_text));
  assert.ok(EXPECTED_SPEECH_TRANSCRIPT);
});

test('test_speech_fixture_fails_closed', () => withTmp((tmp) => {
  const good = { audio_sha256: 'a'.repeat(64), audio_url: 'https://example.com/a.wav', expected_text: 'hello' };
  const file = path.join(tmp, SPEECH_FIXTURE_FILE);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, pyJsonDumps(good));
  assert.deepEqual(loadSpeechFixture(file), good);
  const { expected_text: _dropped, ...missing } = good;
  for (const [label, raw] of [
    ['missing', pyJsonDumps(missing)],
    ['extra', pyJsonDumps({ ...good, extra: 'x' })],
    ['empty', pyJsonDumps({ ...good, expected_text: ' ' })],
    ['not a string', pyJsonDumps({ ...good, audio_url: 1 })],
    ['duplicate', `${pyJsonDumps(good).slice(0, -1)}, "expected_text": "x"}`],
    ['nan', `${pyJsonDumps(good).slice(0, -1)}, "n": NaN}`],
    ['list', '[]'],
    ['malformed', '{'],
  ]) {
    fs.writeFileSync(file, raw);
    raises(() => loadSpeechFixture(file), label);
  }
  fs.unlinkSync(file);
  raises(() => loadSpeechFixture(file));
}));

// The module specifiers of `text`: static imports and re-exports (which
// may span lines), bare imports, and dynamic import() calls, whose target
// must be a plain string literal. Comment lines are prose, not code.
const STATIC_SPECIFIER = /^[ \t]*(?:import|export)\b[^;'"`]*?\bfrom\s*(['"])([^'"]+)\1/gm;
const BARE_SPECIFIER = /^[ \t]*import\s*(['"])([^'"]+)\1/gm;
const DYNAMIC_IMPORT = /(?<![\w$.])import\s*\(/g;
const LITERAL_ARGUMENT = /^\s*(['"])([^'"`$\\]+)\1\s*\)/;

function moduleSpecifiers(name, text) {
  const specifiers = [...text.matchAll(STATIC_SPECIFIER), ...text.matchAll(BARE_SPECIFIER)].map((match) => match[2]);
  for (const match of text.matchAll(DYNAMIC_IMPORT)) {
    const lineStart = text.lastIndexOf('\n', match.index) + 1;
    if (/^\s*(?:\/\/|\*|\/\*)/.test(text.slice(lineStart, match.index))) continue;
    const literal = LITERAL_ARGUMENT.exec(text.slice(match.index + match[0].length));
    assert.ok(literal, `${name} has an import() whose target is not a string literal, which the closure cannot follow`);
    specifiers.push(literal[2]);
  }
  return specifiers;
}

// Parse the imports of the harness entries and return the sorted closure of
// scripts/ files they reach, plus `extraFiles` (the speech fixture, which
// qualification.mjs reads by path). Only node: builtins, the smokes'
// 'playwright' (the candidate's locked dependency) and the '/...' web-root
// URLs the smokes' page code imports may be non-relative.
export function harnessClosure(scriptsDir, entries, extraFiles = []) {
  const seen = new Set();
  const visit = (name) => {
    if (seen.has(name)) return;
    seen.add(name);
    if (!name.endsWith('.mjs')) return;
    const text = fs.readFileSync(path.join(scriptsDir, name), 'utf8');
    assert.ok(!/\bcreateRequire\b|(?<![\w$.])require\s*\(/.test(text), `${name} loads modules through require`);
    for (const spec of moduleSpecifiers(name, text)) {
      if (spec.startsWith('./') || spec.startsWith('../')) {
        const target = path.posix.normalize(path.posix.join(path.posix.dirname(name), spec));
        assert.ok(!target.startsWith('../'), `${name} imports ${spec} from outside scripts/`);
        visit(target);
      } else {
        assert.ok(spec.startsWith('node:') || spec.startsWith('/') || spec === 'playwright', `${name} imports ${JSON.stringify(spec)}`);
      }
    }
  };
  for (const entry of entries) visit(entry);
  for (const file of extraFiles) seen.add(file);
  return [...seen].sort();
}

// The harness entries: the qualification CLI and the module it runs the
// gates from, the smokes qualify runs, and the candidate build's hosted gate
// smokes (read from bridge_candidate.yml).
function harnessEntries() {
  const workflow = fs.readFileSync(path.join(SCRIPTS_DIR, '..', '.github', 'workflows', 'bridge_candidate.yml'), 'utf8');
  const candidateGates = [...workflow.matchAll(/run: node scripts\/(\S+\.mjs)\s*$/gm)].map((match) => match[1]);
  assert.deepEqual([...candidateGates].sort(), ['smoke/multimodal.mjs', 'smoke/state_persistence.mjs']);
  return ['release/qualification.mjs', 'release/qualify.mjs', ...QUALIFICATION_SMOKES, ...candidateGates];
}

test('test_harness_sources_are_exactly_what_the_gates_execute_or_read', () => {
  const closure = harnessClosure(SCRIPTS_DIR, harnessEntries(), [SPEECH_FIXTURE_FILE]);
  assert.deepEqual([...HARNESS_SOURCES].sort(), closure);
  assert.deepEqual([...HARNESS_SOURCES], closure, 'HARNESS_SOURCES must be listed sorted');
  assert.equal(new Set(HARNESS_SOURCES).size, HARNESS_SOURCES.length);
  // The release modules the entries import are harness code, as their .py
  // originals were.
  for (const name of ['release/contract.mjs', 'release/manifest.mjs', 'release/publication_state.mjs']) {
    assert.ok(closure.includes(name), name);
  }
  // Nothing outside the closure locates a file through import.meta: only the
  // entry-point checks, the program name, and the two reviewed paths into
  // scripts/ (qualify's scripts directory and the speech fixture), which
  // both resolve inside the digested tree.
  const reviewed = new Set([
    'release/qualify.mjs:export const SCRIPTS_DIR = path.dirname(path.dirname(fs.realpathSync(fileURLToPath(import.meta.url))));',
    "release/qualification.mjs:export const SPEECH_FIXTURE = Object.freeze(loadSpeechFixture(path.join(import.meta.dirname, '..', SPEECH_FIXTURE_FILE)));",
  ]);
  for (const name of closure.filter((file) => file.endsWith('.mjs'))) {
    const text = fs.readFileSync(path.join(SCRIPTS_DIR, name), 'utf8');
    assert.ok(!text.includes('__dirname') && !text.includes('__filename'), name);
    for (const line of text.split('\n')) {
      const uses = [...line.matchAll(/import\.meta\.(\w+)/g)].map((match) => match[1]);
      if (uses.length === 0 || uses.every((use) => use === 'main')) continue;
      if (/progName\(import\.meta\.url\)/.test(line) && uses.every((use) => use === 'url')) continue;
      assert.ok(reviewed.has(`${name}:${line.trim()}`), `${name} uses import.meta outside the reviewed paths: ${line.trim()}`);
    }
  }
});

test('the harness closure follows every relative import form and fails on a new one', () => withTmp((tmp) => {
  const files = {
    'release/entry.mjs': [
      "import { a } from './a.mjs';",
      "export { b } from './b.mjs';",
      "import './side.mjs';",
      "const lazy = await import('./lazy.mjs');",
      "import fs from 'node:fs';",
      "import data from '../smoke/data.json' with { type: 'json' };",
      'import {',
      '  multi,',
      "} from './multi.mjs';",
      "// A comment naming import('./commented.mjs') or from './commented.mjs' is prose.",
    ].join('\n'),
    'release/a.mjs': "import { c } from '../smoke/c.mjs';",
    'release/b.mjs': '',
    'release/multi.mjs': '',
    'release/side.mjs': '',
    'release/lazy.mjs': '',
    'release/unlisted.mjs': '',
    'smoke/c.mjs': "const page = () => import('/llama_webgpu_bridge.js');\nimport('playwright');",
    'smoke/data.json': '{}',
  };
  for (const [name, text] of Object.entries(files)) {
    fs.mkdirSync(path.dirname(path.join(tmp, name)), { recursive: true });
    fs.writeFileSync(path.join(tmp, name), text);
  }
  assert.deepEqual(harnessClosure(tmp, ['release/entry.mjs'], ['fixture.json']), [
    'fixture.json', 'release/a.mjs', 'release/b.mjs', 'release/entry.mjs', 'release/lazy.mjs', 'release/multi.mjs', 'release/side.mjs',
    'smoke/c.mjs', 'smoke/data.json',
  ]);
  // A new relative import widens the closure, so HARNESS_SOURCES no longer
  // equals it.
  fs.appendFileSync(path.join(tmp, 'release/b.mjs'), "\nimport './unlisted.mjs';\n");
  assert.ok(harnessClosure(tmp, ['release/entry.mjs']).includes('release/unlisted.mjs'));
  for (const [label, text] of [
    ['a package', "import yaml from 'yaml';"],
    ['a computed import()', 'const m = await import(name);'],
    ['a template literal import()', 'const m = await import(`./x.mjs`);'],
    ['require', "const x = require('./x.cjs');"],
    ['an import from outside scripts/', "import '../../x.mjs';"],
  ]) {
    fs.writeFileSync(path.join(tmp, 'release/b.mjs'), text);
    assert.throws(() => harnessClosure(tmp, ['release/entry.mjs']), assert.AssertionError, label);
  }
}));

test('test_harness_version_moves_with_the_harness_sources', () => {
  // A new harness file list is a new harness: bump HARNESS_VERSION with it,
  // so an attestation from the old list fails on its version too.
  assert.deepEqual([HARNESS_VERSION, [...HARNESS_SOURCES].sort()], [
    '5.0.0',
    [
      'release/archive.mjs',
      'release/cli.mjs',
      'release/contract.mjs',
      'release/errors.mjs',
      'release/json.mjs',
      'release/manifest.mjs',
      'release/publication_state.mjs',
      'release/python_compat.mjs',
      'release/qualification.mjs',
      'release/qualify.mjs',
      'release/unicode15.mjs',
      'release/wav.mjs',
      'smoke/multimodal.mjs',
      'smoke/speech_to_text.mjs',
      'smoke/speech_to_text_fixture.json',
      'smoke/state_persistence.mjs',
      'smoke/support.mjs',
      'smoke/text_to_speech.mjs',
    ],
  ]);
});

test('test_local_harness_must_match_the_exact_bridge_source', () => withTmp((tmp) => {
  const repository = path.join(tmp, 'harness-repository');
  const scriptsDir = path.join(repository, 'scripts');
  const files = Object.fromEntries(HARNESS_SOURCES.map((name) => [`scripts/${name}`, fs.readFileSync(path.join(SCRIPTS_DIR, name))]));
  const bridgeSha = commitRepository(repository, files);
  requireHarnessMatchesBridgeSource(scriptsDir, bridgeSha);
  fs.writeFileSync(path.join(scriptsDir, HARNESS_SOURCES[0]), 'drift\n');
  assert.ok(raises(() => requireHarnessMatchesBridgeSource(scriptsDir, bridgeSha)).includes('does not match'));
}));

// --- Node-only ----------------------------------------------------------------------------

// The digest scheme, spelled out: sorted names, each as name NUL, u64 BE
// length, bytes.
function referenceDigest(entries) {
  const digest = createHash('sha256');
  for (const [name, data] of [...entries].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) {
    const length = Buffer.alloc(8);
    length.writeBigUInt64BE(BigInt(data.length));
    digest.update(Buffer.concat([Buffer.from(`${name}\0`), length, Buffer.from(data)]));
  }
  return digest.digest('hex');
}

test('harness digests take source names with sub-paths', () => withTmp((tmp) => {
  const sources = ['release/qualification.mjs', 'smoke/support.mjs', 'release_contract.txt', 'release.mjs', 'fixture.json'];
  const files = {
    'scripts/release/qualification.mjs': 'q',
    'scripts/smoke/support.mjs': 's',
    'scripts/release_contract.txt': 'c',
    'scripts/release.mjs': 'r',
    'scripts/fixture.json': '{}',
  };
  const repository = path.join(tmp, 'repo');
  const bridgeSha = commitRepository(repository, files);
  const scriptsDir = path.join(repository, 'scripts');
  const expected = referenceDigest(sources.map((name) => [name, files[`scripts/${name}`]]));
  // '.' (0x2e) sorts before '/' (0x2f), which sorts before '_' (0x5f).
  assert.equal(harnessSourceSha256(scriptsDir, { sources }), expected);
  assert.equal(harnessSourceSha256AtCommit(repository, bridgeSha, { sources }), expected);
  assert.equal(requireHarnessMatchesBridgeSource(`${scriptsDir}/`, bridgeSha, { sources }), expected);
  fs.writeFileSync(path.join(scriptsDir, 'smoke', 'support.mjs'), 'drift');
  assert.ok(raises(() => requireHarnessMatchesBridgeSource(scriptsDir, bridgeSha, { sources })).includes('does not match'));
  fs.rmSync(path.join(scriptsDir, 'smoke'), { recursive: true });
  assert.equal(raises(() => harnessSourceSha256(scriptsDir, { sources })), 'harness source is missing: smoke/support.mjs');
  const missing = raises(() => harnessSourceSha256AtCommit(repository, bridgeSha, { sources: ['absent/x.mjs'] }));
  assert.ok(missing.startsWith(`could not read harness source 'absent/x.mjs' at ${bridgeSha}: fatal: path 'scripts/absent/x.mjs' does not exist`), missing);
  assert.equal(raises(() => harnessSourceSha256AtCommit(repository, 'HEAD')), 'candidate bridge source must be a lowercase 40-hex SHA');
  // The real list digests today's scripts/ exactly as the scheme says.
  assert.equal(
    harnessSourceSha256(SCRIPTS_DIR),
    referenceDigest(HARNESS_SOURCES.map((name) => [name, fs.readFileSync(path.join(SCRIPTS_DIR, name))])),
  );
}));

test('Path.resolve() and expanduser() follow pathlib', () => withTmp((tmp) => {
  fs.mkdirSync(path.join(tmp, 'a', 'b'), { recursive: true });
  fs.symlinkSync(path.join(tmp, 'a', 'b'), path.join(tmp, 'link'));
  fs.symlinkSync('loop', path.join(tmp, 'loop'));
  // A symlink is resolved before a following '..' applies.
  assert.equal(pyResolve(`${tmp}/link/../x`), path.join(tmp, 'a', 'x'));
  assert.equal(pyResolve(`${tmp}/missing/../a/./b/`), path.join(tmp, 'a', 'b'));
  assert.equal(pyResolve('.'), process.cwd());
  assert.throws(() => pyResolve(path.join(tmp, 'loop')), (error) => error.pyType === 'RuntimeError'
    && error.message === `Symlink loop from '${path.join(tmp, 'loop')}'`);
  assert.equal(pyNormpath('//a/../b'), '//b');
  assert.equal(pyNormpath('../a/..'), '..');
  assert.equal(pyNormpath('/../a'), '/a');
  assert.equal(pyExpanduser('x/~'), 'x/~');
  assert.equal(pyExpanduser('~/d'), path.join(process.env.HOME, 'd'));
}));
