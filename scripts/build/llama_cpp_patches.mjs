import { mkdirSync, readFileSync, readdirSync, realpathSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const patchDir = fileURLToPath(new URL('../../patches/llama.cpp/', import.meta.url));
const patchName = /^\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*\.patch$/;
const sourcePath = /^(?:[a-zA-Z0-9_-][a-zA-Z0-9_.+-]*\/)*[a-zA-Z0-9_-][a-zA-Z0-9_.+-]*\.(?:c|cpp)$/;
const keys = new Set(['file', 'target', 'tracking', 'upstream', 'wasm_marker', 'upstream_fix']);
const trackingLink = /^https:\/\/github\.com\/leehack\/llama-web-bridge\/issues\/(?:[1-9]\d*|TRACKING-ISSUE-PENDING)$/;
const upstreamLink = /^https:\/\/github\.com\/ggml-org\/llama\.cpp\/(?:issues\/(?:[1-9]\d*|UPSTREAM-ISSUE-PENDING)|pull\/[1-9]\d*)$/;

const readText = (file) => readFileSync(file, 'utf8').replaceAll('\r\n', '\n');
const nonempty = (value) => typeof value === 'string' && value.trim() !== '' && !value.includes('\0');

export function loadSeries(patches = patchDir) {
  try {
    const series = JSON.parse(readText(path.join(patches, 'series.json')));
    if (!series || Object.keys(series).length !== 1 || !Array.isArray(series.patches)) {
      throw new Error('expected {"patches": [...]}');
    }
    const names = [];
    const markers = new Set();
    for (const entry of series.patches) {
      if (!entry || typeof entry.file !== 'string' || !patchName.test(entry.file)) {
        throw new Error('each patch needs a NNNN-name.patch file');
      }
      if (Object.keys(entry).some((key) => !keys.has(key))
          || typeof entry.target !== 'string' || !/^[A-Za-z0-9_-]+$/.test(entry.target)
          || typeof entry.tracking !== 'string' || !trackingLink.test(entry.tracking)
          || typeof entry.upstream !== 'string' || !upstreamLink.test(entry.upstream)
          || !nonempty(entry.wasm_marker) || markers.has(entry.wasm_marker)
          || (Object.hasOwn(entry, 'upstream_fix') && (!nonempty(entry.upstream_fix)
            || entry.upstream_fix.includes(entry.wasm_marker)))) {
        throw new Error(entry.file + ': expected a CMake target, owner tracking issue, upstream issue or PR, '
          + 'unique Wasm marker and optional upstream fix text');
      }
      names.push(entry.file);
      markers.add(entry.wasm_marker);
    }
    const disk = readdirSync(patches).filter((file) => file.endsWith('.patch')).sort();
    if (names.length !== disk.length || names.some((name, i) => name !== disk[i])) {
      throw new Error('patches must list every .patch file exactly once in filename order');
    }
    return series.patches;
  } catch (error) {
    throw new Error('series.json: ' + error.message, { cause: error });
  }
}

export function verifyReleaseLinks(patches = patchDir) {
  for (const { file, tracking, upstream } of loadSeries(patches)) {
    for (const [name, url] of Object.entries({ tracking, upstream })) {
      if (url.endsWith('-PENDING')) {
        throw new Error(file + ': replace the pending ' + name + ' issue URL before building a release candidate');
      }
    }
  }
}

function parsePatch(file, text) {
  const lines = text.split('\n');
  if (lines.at(-1) === '') lines.pop();
  const gitHeader = lines[0]?.startsWith('diff --git ') ? lines.shift() : null;
  if (gitHeader && /^index [0-9a-f]+\.\.[0-9a-f]+(?: 100(?:644|755))?$/.test(lines[0])) lines.shift();
  const oldPath = /^--- a\/(.+)$/.exec(lines.shift())?.[1];
  if (!oldPath || !sourcePath.test(oldPath) || lines.shift() !== '+++ b/' + oldPath
      || (gitHeader && gitHeader !== 'diff --git a/' + oldPath + ' b/' + oldPath)) {
    throw new Error(file + ': expected a unified diff of one existing .c or .cpp source');
  }

  const hunks = [];
  for (let i = 0; i < lines.length;) {
    const header = lines[i++];
    const range = /^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@(?: .*)?$/.exec(header);
    if (!range) throw new Error(file + ': invalid hunk header ' + header);
    const before = [];
    const after = [];
    while (i < lines.length && !lines[i].startsWith('@@ ')) {
      const line = lines[i++];
      if (!/^[ +\-]/.test(line)) throw new Error(file + ': invalid hunk line ' + line);
      if (line[0] !== '+') before.push(line.slice(1));
      if (line[0] !== '-') after.push(line.slice(1));
    }
    if (before.length === 0 || before.length !== Number(range[1] ?? 1)
        || after.length !== Number(range[2] ?? 1) || before.join('\n') === after.join('\n')) {
      throw new Error(file + ': invalid hunk range or empty change in ' + header);
    }
    hunks.push({
      header,
      before: before.join('\n') + '\n',
      after: after.length === 0 ? '' : after.join('\n') + '\n',
    });
  }
  if (hunks.length === 0) throw new Error(file + ': expected at least one hunk');
  return { relative: oldPath, hunks };
}

function applyHunks(entry, source) {
  const indexedSource = '\n' + source;
  let cursor = 0;
  let result = '';
  for (const { header, before, after } of entry.hunks) {
    const needle = '\n' + before;
    const offset = indexedSource.indexOf(needle);
    let count = 0;
    for (let match = offset; match !== -1; match = indexedSource.indexOf(needle, match + 1)) count++;
    if (count !== 1) return { reason: header + ' matches ' + count + ' times, expected 1' };
    if (offset < cursor) return { reason: header + ' overlaps or precedes another hunk' };
    result += source.slice(cursor, offset) + after;
    cursor = offset + before.length;
  }
  return { text: result + source.slice(cursor) };
}

export function applySeries(upstream, patches = patchDir, { qualification = false } = {}) {
  const root = realpathSync(upstream);
  const sources = new Map();
  const applied = [];
  const skipped = [];
  for (const entry of loadSeries(patches)) {
    const parsed = parsePatch(entry.file, readText(path.join(patches, entry.file)));
    const relative = parsed.relative;
    let source = sources.get(relative);
    if (!source) {
      const originalPath = path.join(root, relative);
      let original;
      try {
        if (realpathSync(originalPath) !== originalPath) {
          throw new Error(entry.file + ': patched source must not resolve through a symlink');
        }
        original = readText(originalPath);
      } catch (error) {
        if (error.code !== 'ENOENT') throw error;
      }
      source = { target: entry.target, original, current: original };
      sources.set(relative, source);
    }
    if (source.target !== entry.target) throw new Error(entry.file + ': source has conflicting targets');
    if (entry.upstream_fix && source.original?.includes(entry.upstream_fix)) continue;
    if (source.current?.includes(entry.wasm_marker)) {
      throw new Error(entry.file + ': marker already present in ' + relative);
    }
    const result = source.current === undefined
      ? { reason: 'source does not exist' }
      : applyHunks(parsed, source.current);
    if (result.reason) {
      const message = entry.file + ': ' + relative + ': ' + result.reason;
      if (!qualification) throw new Error(message);
      skipped.push(message);
      continue;
    }
    if (!result.text.includes(entry.wasm_marker)) {
      throw new Error(entry.file + ': patched source lacks its Wasm marker');
    }
    applied.push({ ...entry, relative, text: result.text });
    source.current = result.text;
  }
  return { sources: [...sources.keys()], applied, skipped };
}

export function verifyMarkers(applied, wasmFiles) {
  if (wasmFiles.length === 0) throw new Error('expected at least one Wasm file');
  for (const file of wasmFiles) {
    const bytes = readFileSync(file);
    if (bytes.subarray(0, 8).toString('hex') !== '0061736d01000000') {
      throw new Error(file + ': expected a Wasm binary');
    }
    for (const entry of applied) {
      if (!bytes.includes(Buffer.from(entry.wasm_marker))) {
        throw new Error(file + ': built without ' + entry.file + ' (missing "' + entry.wasm_marker + '")');
      }
    }
  }
}

function configure(upstream, patches, output, qualification) {
  const { sources, applied, skipped } = applySeries(upstream, patches, { qualification });
  const outputs = new Map(applied.map(({ relative, target, text }) => [relative, { relative, target, text }]));
  for (const [relative, { text }] of outputs) {
    const file = path.join(output, relative);
    mkdirSync(path.dirname(file), { recursive: true });
    try {
      if (readFileSync(file, 'utf8') === text) continue;
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    writeFileSync(file, text);
  }
  for (const message of skipped) console.error('Skipped ' + message);
  console.log(JSON.stringify({
    sources,
    replacements: [...outputs.values()].map(({ relative, target }) => ({ relative, target })),
  }));
}

function main(args) {
  const qualification = args[0] === '--qualification';
  if (qualification) args.shift();
  const [action, upstream, ...rest] = args;
  if (action === 'configure' && upstream && rest.length === 2) {
    configure(upstream, rest[0], rest[1], qualification);
  } else if (action === 'verify' && upstream && rest.length > 0) {
    verifyMarkers(applySeries(upstream, patchDir, { qualification }).applied, rest);
  } else if (action === 'verify-release' && args.length === 1 && !qualification) {
    verifyReleaseLinks();
  } else {
    throw new Error('expected [--qualification] configure UPSTREAM PATCH_DIR OUTPUT, verify UPSTREAM WASM..., or verify-release');
  }
}

if (import.meta.main) {
  try {
    main(process.argv.slice(2));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
