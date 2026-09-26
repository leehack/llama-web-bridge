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

import { build, transformSync } from 'esbuild';

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

async function withTmpAsync(fn) {
  const tmp = makeTempDir();
  try {
    return await fn(tmp);
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

// The only non-relative imports the harness may make: node: builtins, the
// smokes' 'playwright' (the candidate's locked dependency), and the bridge
// module the smokes' page code imports from the served web root.
const PAGE_BRIDGE = '/llama_webgpu_bridge.js';
const EXTERNAL_IMPORT = /import\s*\(\s*(["'])(?:node:[\w/]+|playwright|\/llama_webgpu_bridge\.js)\1\s*\)/y;

// Bundle the harness entries with esbuild, the parser the JS bridge build
// already uses, and return the sorted scripts/ files the bundle read, plus
// `extraFiles` (the speech fixture, which qualification.mjs reads by path).
// esbuild parses every import form (string-named bindings, imports after
// other statements, .js modules, JSON with import attributes); a bare
// package, a path outside scripts/, or an import() the bundle cannot follow
// fails the closure.
export async function harnessClosure(scriptsDir, entries, extraFiles = []) {
  const rejectUnknownImports = {
    name: 'harness-imports',
    setup(pluginBuild) {
      pluginBuild.onResolve({ filter: /.*/ }, (args) => {
        if (args.kind === 'entry-point' || args.path.startsWith('./') || args.path.startsWith('../')) return undefined;
        if (args.path.startsWith('node:') || args.path === 'playwright' || args.path === PAGE_BRIDGE) {
          return { path: args.path, external: true };
        }
        return { errors: [{ text: `${args.importer} imports ${JSON.stringify(args.path)}` }] };
      });
    },
  };
  const result = await build({
    absWorkingDir: path.resolve(scriptsDir),
    entryPoints: entries,
    bundle: true,
    write: false,
    metafile: true,
    platform: 'node',
    format: 'esm',
    outdir: 'bundle',
    logLevel: 'silent',
    // Tree shaking must not drop code the scan below has to see, whatever a
    // package.json sideEffects field or a /* @__PURE__ */ comment says.
    ignoreAnnotations: true,
    // List a symlink under its own name, so the check below sees it.
    preserveSymlinks: true,
    plugins: [rejectUnknownImports],
  });
  assert.deepEqual(result.warnings.map((warning) => warning.text), []);
  // An import() the bundle kept is external or computed; only the external
  // ones above are allowed, so a computed target cannot hide a module.
  for (const output of result.outputFiles) {
    for (const match of output.text.matchAll(/(?<![\w$.])import\s*\(/g)) {
      EXTERNAL_IMPORT.lastIndex = match.index;
      assert.ok(EXTERNAL_IMPORT.test(output.text), `the harness keeps an import() it cannot follow: ${output.text.slice(match.index, match.index + 60)}`);
    }
  }
  const files = Object.keys(result.metafile.inputs);
  for (const name of files) {
    assert.ok(!name.startsWith('../') && !path.isAbsolute(name), `the harness reads ${name} from outside scripts/`);
    // ES modules and JSON only: a .js module's meaning depends on an undigested
    // package.json, and a symlink's target is digested under another name.
    assert.ok(/\.(?:mjs|json)$/.test(name), `the harness imports ${name}, which is neither .mjs nor .json`);
    assert.ok(!fs.lstatSync(path.join(scriptsDir, name)).isSymbolicLink(), `the harness imports ${name}, a symlink`);
    const text = fs.readFileSync(path.join(scriptsDir, name), 'utf8');
    assert.ok(!/\bcreateRequire\b|(?<![\w$.])require\s*\(/.test(text), `${name} loads modules through require`);
  }
  return [...new Set([...files, ...extraFiles])].sort();
}

// Nothing in the closure locates a file it runs or reads except through
// reviewed lines: the entry-point checks, the program name, qualify's
// scripts directory and the two smokes it joins onto it, the speech fixture,
// and the harness digest's own reads, all of which resolve inside the
// digested tree. Command-line arguments are only ever the user's
// (argv.slice(2)), never the script's own path, nothing starts a Worker, and
// no string names a scripts/ path relative to the working directory.
//
// The lines are checked as esbuild prints them, so comments are gone and
// spacing is normal. This catches accidental file locations, not deliberate
// evasion: an alias of a reviewed value (qualification.mjs's `directory`), a
// computed property name or an eval'd string can still reach a file.
const REVIEWED_LINES = new Set([
  'release/qualify.mjs:const SCRIPTS_DIR = path.dirname(path.dirname(fs.realpathSync(fileURLToPath(import.meta.url))));',
  'release/qualify.mjs:const scriptsDir = deps.scriptsDir ?? SCRIPTS_DIR;',
  'release/qualify.mjs:scriptsDir,',
  'release/qualify.mjs:pyJoinPath(scriptsDir, SPEECH_SMOKE),',
  'release/qualify.mjs:pyJoinPath(scriptsDir, TTS_SMOKE),',
  'release/qualify.mjs:SCRIPTS_DIR,',
  'release/qualification.mjs:const SPEECH_FIXTURE = Object.freeze(loadSpeechFixture(path.join(import.meta.dirname, "..", SPEECH_FIXTURE_FILE)));',
  'release/qualification.mjs:function harnessSourceSha256(scriptsDir, { sources = HARNESS_SOURCES } = {}) {',
  'release/qualification.mjs:function requireHarnessMatchesBridgeSource(scriptsDir, bridgeSha, { sources = HARNESS_SOURCES } = {}) {',
  'release/qualification.mjs:const directory = pyPath(String(scriptsDir));',
  'release/qualification.mjs:const result = spawnSync("git", ["-C", repositoryPath, "show", `${bridgeSha}:scripts/${name}`], {',
]);
const PROG_NAME = /progName\(import\.meta\.url\)/g;

export function unreviewedFileLocations(name, text) {
  const { code } = transformSync(text, { format: 'esm', loader: 'js', legalComments: 'none', ignoreAnnotations: true });
  const found = [];
  for (const rawLine of code.split('\n')) {
    const line = rawLine.trim();
    const withoutProg = line.replace(PROG_NAME, '');
    const locates = /\bimport\.meta\b(?!\.main\b)/.test(withoutProg)
      || /\b(?:scriptsDir|SCRIPTS_DIR|__dirname|__filename|Worker)\b|scripts\//.test(line)
      || /\bprocess\.argv\b(?!\.slice\(2\))|\bprocess\s*\[/.test(line);
    if (locates && !REVIEWED_LINES.has(`${name}:${line}`)) found.push(line);
  }
  return found;
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

test('test_harness_sources_are_exactly_what_the_gates_execute_or_read', async () => {
  const closure = await harnessClosure(SCRIPTS_DIR, harnessEntries(), [SPEECH_FIXTURE_FILE]);
  assert.deepEqual([...HARNESS_SOURCES].sort(), closure);
  assert.deepEqual([...HARNESS_SOURCES], closure, 'HARNESS_SOURCES must be listed sorted');
  assert.equal(new Set(HARNESS_SOURCES).size, HARNESS_SOURCES.length);
  // The release modules the entries import are harness code, as their .py
  // originals were.
  for (const name of ['release/contract.mjs', 'release/manifest.mjs', 'release/publication_state.mjs']) {
    assert.ok(closure.includes(name), name);
  }
  for (const name of closure.filter((file) => file.endsWith('.mjs'))) {
    assert.deepEqual(unreviewedFileLocations(name, fs.readFileSync(path.join(SCRIPTS_DIR, name), 'utf8')), [], name);
  }
});

test('the harness closure follows every import form and fails on one it cannot follow', () => withTmpAsync(async (tmp) => {
  // The scripts tree is a subdirectory, so a file outside it can exist.
  const root = path.join(tmp, 'scripts');
  const write = (files) => {
    for (const [name, text] of Object.entries(files)) {
      fs.mkdirSync(path.dirname(path.join(root, name)), { recursive: true });
      fs.writeFileSync(path.join(root, name), text);
    }
  };
  write({
    'release/entry.mjs': [
      "import { a } from './a.mjs';",
      "export { b } from './b.mjs';",
      "import './side.mjs';",
      "const lazy = await import('./lazy.mjs');",
      'const template = await import(`./template.mjs`);',
      "import fs from 'node:fs';",
      "import data from '../smoke/data.json' with { type: 'json' };",
      'import {',
      '  multi,',
      "} from './multi.mjs';",
      "const later = 1; import './later.mjs';",
      "/* c */ import { \"named\" as named } from './named.mjs';",
      "import './helper.mjs';",
      "// A comment naming import('./commented.mjs') or from './commented.mjs' is prose.",
    ].join('\n'),
    'release/a.mjs': "export { c as a } from '../smoke/c.mjs';",
    'release/b.mjs': 'export const b = 1;',
    'release/multi.mjs': 'export const multi = 1;',
    'release/side.mjs': '',
    'release/lazy.mjs': '',
    'release/template.mjs': '',
    'release/later.mjs': '',
    'release/named.mjs': 'const n = 1; export { n as "named" };',
    'release/helper.mjs': "import './helper_dependency.mjs';",
    'release/helper_dependency.mjs': '',
    'release/unlisted.mjs': '',
    'release/required.mjs': '',
    '../outside.mjs': '',
    'smoke/c.mjs': "export const c = () => import('/llama_webgpu_bridge.js');\nexport const p = () => import('playwright');",
    'smoke/data.json': '{}',
  });
  assert.deepEqual(await harnessClosure(root, ['release/entry.mjs'], ['fixture.json']), [
    'fixture.json', 'release/a.mjs', 'release/b.mjs', 'release/entry.mjs', 'release/helper.mjs', 'release/helper_dependency.mjs',
    'release/later.mjs', 'release/lazy.mjs', 'release/multi.mjs', 'release/named.mjs', 'release/side.mjs', 'release/template.mjs',
    'smoke/c.mjs', 'smoke/data.json',
  ]);
  // A new relative import widens the closure, so HARNESS_SOURCES no longer
  // equals it.
  fs.appendFileSync(path.join(root, 'release/b.mjs'), "\nimport './unlisted.mjs';\n");
  assert.ok((await harnessClosure(root, ['release/entry.mjs'])).includes('release/unlisted.mjs'));
  for (const [label, text] of [
    ['a package', "import yaml from 'yaml';"],
    ['a bare builtin', "import fs from 'fs';"],
    ['a computed import()', 'export const m = await import(name);'],
    ['a computed import() after a line-leading operator', 'export const m = 2\n  * await import(name);'],
    ['an absolute import()', "export const m = await import('/abs/evil.mjs');"],
    ['require', "const x = require('./required.mjs');"],
    ['createRequire', "import { createRequire } from 'node:module';"],
    ['an import from outside scripts/', "import '../../outside.mjs';"],
    ['a .js module', "import './plain.js';"],
    ['a symlink', "import './link.mjs';"],
    ['a computed import() in code tree shaking would drop', "import { unused } from './pure/pure.mjs';"],
  ]) {
    write({
      'release/plain.js': '',
      'release/pure/pure.mjs': 'export const unused = 1;\nawait import(name);\n',
      'release/pure/package.json': '{"sideEffects": false}',
    });
    fs.rmSync(path.join(root, 'release/link.mjs'), { force: true });
    fs.symlinkSync('side.mjs', path.join(root, 'release/link.mjs'));
    fs.writeFileSync(path.join(root, 'release/b.mjs'), `${text}\nexport const b = 1;\n`);
    await assert.rejects(harnessClosure(root, ['release/entry.mjs']), label);
  }
}));

test('the reviewed-lines check sees file locations however they are spelled', () => {
  // The real reviewed lines pass as written.
  assert.deepEqual(unreviewedFileLocations('release/qualify.mjs', 'const scriptsDir = deps.scriptsDir ?? SCRIPTS_DIR;\n'), []);
  assert.deepEqual(unreviewedFileLocations('release/x.mjs', [
    'if (import.meta.main) runCli(main);',
    'const PROG = progName(import.meta.url);',
    'export async function main(argv = process.argv.slice(2)) {}',
    '// import.meta.dirname and scripts/x.mjs in a comment are prose',
  ].join('\n')), []);
  for (const text of [
    'const { dirname } = import.meta;',
    'const m = import.meta; use(m);',
    'use(import.meta["dirname"]);',
    'use(import . meta.dirname);',
    'const x = 2\n  * read(path.join(import.meta.dirname, "x"));',
    'const u = "file://" + import.meta.dirname;',
    'const p = progName(import.meta.url), d = path.dirname(fileURLToPath(import.meta.url));',
    'export function harnessSourceSha256(scriptsDir, { sources = HARNESS_SOURCES } = {}) { return read(scriptsDir); }',
    'const scriptsDir = deps.scriptsDir ?? SCRIPTS_DIR; read(scriptsDir);',
    'spawn(node, ["scripts/smoke/x.mjs"]);',
    'const self = process.argv[1];',
    'const a = process["argv"];',
    'new Worker("./x.mjs");',
    'const d = __dirname;',
  ]) {
    assert.notDeepEqual(unreviewedFileLocations('release/qualify.mjs', text), [], text);
  }
});

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
