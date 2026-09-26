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
  assert.ok(HARNESS_SOURCES.includes('release_contract.py'));
  assert.ok(HARNESS_SOURCES.includes('release_publication_state.py'));
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

test('test_harness_sources_are_exactly_what_the_gates_execute_or_read', () => {
  // The Python import closure of release_qualification.py: every top-level
  // `import x` / `from x import` (at any indentation, as ast.walk sees them)
  // that names a scripts/*.py file.
  const pythonClosure = (name, seen) => {
    if (seen.has(name)) return;
    seen.add(name);
    const text = fs.readFileSync(path.join(SCRIPTS_DIR, name), 'utf8');
    const modules = [];
    for (const match of text.matchAll(/^[ \t]*import[ \t]+([\w.]+(?:[ \t]+as[ \t]+\w+)?(?:[ \t]*,[ \t]*[\w.]+(?:[ \t]+as[ \t]+\w+)?)*)/gm)) {
      for (const part of match[1].split(',')) modules.push(part.trim().split(/\s+/)[0]);
    }
    for (const match of text.matchAll(/^[ \t]*from[ \t]+(\w[\w.]*)[ \t]+import\b/gm)) modules.push(match[1]);
    for (const module of modules) {
      const candidate = `${module.split('.')[0]}.py`;
      if (fs.existsSync(path.join(SCRIPTS_DIR, candidate)) && fs.statSync(path.join(SCRIPTS_DIR, candidate)).isFile()) {
        pythonClosure(candidate, seen);
      }
    }
  };

  // Static and dynamic imports and re-exports; a relative specifier is a
  // file relative to the importer, inside scripts/. Harness page code imports
  // '/...' URLs from the web root, which are not scripts/ files.
  const specifier = /(?:\bfrom\s*|\bimport\s*\(?\s*)(['"])([^'"]+)\1/g;
  const nodeClosure = (name, seen) => {
    if (seen.has(name)) return;
    seen.add(name);
    if (!name.endsWith('.mjs')) return;
    const text = fs.readFileSync(path.join(SCRIPTS_DIR, name), 'utf8');
    // Any other file a module reads would have to be located through
    // import.meta; only the entry-point check may use it.
    assert.deepEqual([...text.matchAll(/import\.meta\.(?!main\b)\w+/g)].map((match) => match[0]), [], name);
    assert.ok(!text.includes('__dirname'), name);
    for (const [, , spec] of text.matchAll(specifier)) {
      if (spec.startsWith('./') || spec.startsWith('../')) {
        const target = path.resolve(SCRIPTS_DIR, path.dirname(name), spec);
        const relative = path.relative(SCRIPTS_DIR, target);
        assert.ok(!relative.startsWith('..') && !path.isAbsolute(relative), spec);
        nodeClosure(relative.split(path.sep).join('/'), seen);
      } else {
        assert.ok(spec.startsWith('node:') || spec.startsWith('/') || spec === 'playwright', `${name} imports ${JSON.stringify(spec)}`);
      }
    }
  };

  const workflow = fs.readFileSync(path.join(SCRIPTS_DIR, '..', '.github', 'workflows', 'bridge_candidate.yml'), 'utf8');
  const candidateGates = [...workflow.matchAll(/run: node scripts\/(\S+\.mjs)\s*$/gm)].map((match) => match[1]);
  assert.deepEqual([...candidateGates].sort(), ['smoke/multimodal.mjs', 'smoke/state_persistence.mjs']);
  const closure = new Set();
  pythonClosure('release_qualification.py', closure);
  closure.add(SPEECH_FIXTURE_FILE);
  for (const smoke of [...QUALIFICATION_SMOKES, ...candidateGates]) nodeClosure(smoke, closure);
  assert.deepEqual([...closure].sort(), [...HARNESS_SOURCES].sort());
  assert.equal(new Set(HARNESS_SOURCES).size, HARNESS_SOURCES.length);
});

test('test_harness_version_moves_with_the_harness_sources', () => {
  // A new harness file list is a new harness: bump HARNESS_VERSION with it,
  // so an attestation from the old list fails on its version too.
  assert.deepEqual([HARNESS_VERSION, [...HARNESS_SOURCES].sort()], [
    '4.0.0',
    [
      'generate_release_manifest.py',
      'release_contract.py',
      'release_publication_state.py',
      'release_qualification.py',
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
  const sources = ['release/qualification.mjs', 'smoke/support.mjs', 'release_contract.py', 'release.mjs', 'fixture.json'];
  const files = {
    'scripts/release/qualification.mjs': 'q',
    'scripts/smoke/support.mjs': 's',
    'scripts/release_contract.py': 'c',
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
