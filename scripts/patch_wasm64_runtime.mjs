#!/usr/bin/env node
// Patch the Emscripten wasm64 WASMFS JavaScript boundary: wrap the pointer,
// length and offset arguments the generated JS passes to the five __wasmfs_*
// imports in BigInt(), in place. Idempotent; fails without writing when any
// of the five symbols is missing from the generated output.
//
// Ported from scripts/patch_wasm64_runtime.py with byte-identical output: the
// file is read as UTF-8 dropping undecodable bytes and with universal
// newlines, and `\s` is Python's whitespace class, not JavaScript's.

import fs from 'node:fs';
import process from 'node:process';

// Python's `re` \s for str patterns (str.isspace()); JavaScript's \s lacks
// U+001C-U+001F and U+0085 and adds U+FEFF.
const SPACES = '[\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]*';

const bigintOrName = (name) => `(?:BigInt\\(${SPACES}${name}${SPACES}\\)|${name})`;

const DATA_BUFFER = bigintOrName('dataBuffer');
const LENGTH = bigintOrName('length');
const POSITION = bigintOrName('position');
const OFFSET = bigintOrName('offset');
const call = (name, args) => `${name}\\(${SPACES}${args.join(`${SPACES},${SPACES}`)}${SPACES}\\)`;

export const PATCHES = [
  [
    '__wasmfs_read',
    call('__wasmfs_read', ['stream\\.fd', DATA_BUFFER, LENGTH]),
    '__wasmfs_read(stream.fd,BigInt(dataBuffer),BigInt(length))',
  ],
  [
    '__wasmfs_pread',
    call('__wasmfs_pread', ['stream\\.fd', DATA_BUFFER, LENGTH, POSITION]),
    '__wasmfs_pread(stream.fd,BigInt(dataBuffer),BigInt(length),BigInt(position))',
  ],
  [
    '__wasmfs_write',
    call('__wasmfs_write', ['stream\\.fd', DATA_BUFFER, LENGTH]),
    '__wasmfs_write(stream.fd,BigInt(dataBuffer),BigInt(length))',
  ],
  [
    '__wasmfs_pwrite',
    call('__wasmfs_pwrite', ['stream\\.fd', DATA_BUFFER, LENGTH, POSITION]),
    '__wasmfs_pwrite(stream.fd,BigInt(dataBuffer),BigInt(length),BigInt(position))',
  ],
  [
    '__wasmfs_mmap',
    call('__wasmfs_mmap', [LENGTH, 'prot', 'flags', 'stream\\.fd', OFFSET]),
    '__wasmfs_mmap(BigInt(length),prot,flags,stream.fd,BigInt(offset))',
  ],
];

export class PatchError extends Error {}

export function patchWasm64Runtime(text) {
  const counts = {};
  for (const [name, pattern, replacement] of PATCHES) {
    let count = 0;
    text = text.replace(new RegExp(pattern, 'g'), () => {
      count += 1;
      return replacement;
    });
    counts[name] = count;
  }
  const missing = Object.keys(counts).filter((name) => counts[name] === 0);
  if (missing.length > 0) {
    throw new PatchError(
      'wasm64 runtime patch did not match required generated-JS symbols: '
      + `${missing.join(', ')}; inspect the pinned Emscripten output before `
      + 'changing the expected symbol set',
    );
  }
  return { text, counts };
}

// UTF-8 decoding that drops each maximal ill-formed subsequence, as Python's
// errors="ignore" does; the BOM is kept.
function decodeUtf8DroppingInvalid(bytes) {
  try {
    return new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes);
  } catch {
    // Fall through to the dropping decoder below.
  }
  const codePoints = [];
  let needed = 0;
  let codePoint = 0;
  let lower = 0x80;
  let upper = 0xbf;
  for (let i = 0; i < bytes.length; i += 1) {
    const byte = bytes[i];
    if (needed === 0) {
      if (byte <= 0x7f) {
        codePoints.push(byte);
      } else if (byte >= 0xc2 && byte <= 0xdf) {
        needed = 1;
        codePoint = byte & 0x1f;
      } else if (byte >= 0xe0 && byte <= 0xef) {
        if (byte === 0xe0) lower = 0xa0;
        if (byte === 0xed) upper = 0x9f;
        needed = 2;
        codePoint = byte & 0x0f;
      } else if (byte >= 0xf0 && byte <= 0xf4) {
        if (byte === 0xf0) lower = 0x90;
        if (byte === 0xf4) upper = 0x8f;
        needed = 3;
        codePoint = byte & 0x07;
      }
      continue;
    }
    if (byte < lower || byte > upper) {
      // Drop the incomplete sequence and decode this byte afresh.
      needed = 0;
      lower = 0x80;
      upper = 0xbf;
      i -= 1;
      continue;
    }
    lower = 0x80;
    upper = 0xbf;
    codePoint = (codePoint << 6) | (byte & 0x3f);
    needed -= 1;
    if (needed === 0) codePoints.push(codePoint);
  }
  let text = '';
  for (let i = 0; i < codePoints.length; i += 8192) {
    text += String.fromCodePoint(...codePoints.slice(i, i + 8192));
  }
  return text;
}

// Python's Path.read_text(encoding="utf-8", errors="ignore"): universal
// newlines translate \r\n and \r to \n.
export function readText(target) {
  return decodeUtf8DroppingInvalid(fs.readFileSync(target)).replace(/\r\n?/g, '\n');
}

const USAGE = 'usage: patch_wasm64_runtime.mjs [-h] target';

function parseArgs(argv) {
  const positional = [];
  let options = true;
  for (const arg of argv) {
    if (options && arg === '--') {
      options = false;
    } else if (options && (arg === '-h' || arg === '--help')) {
      return { help: true };
    } else if (options && arg.startsWith('-') && arg !== '-') {
      return { error: `unrecognized arguments: ${arg}` };
    } else {
      positional.push(arg);
    }
  }
  if (positional.length === 0) return { error: 'the following arguments are required: target' };
  if (positional.length > 1) return { error: `unrecognized arguments: ${positional.slice(1).join(' ')}` };
  return { target: positional[0] };
}

export function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args.help) {
    console.log(`${USAGE}\n\npositional arguments:\n  target      generated wasm64 JavaScript file`);
    return 0;
  }
  if (args.error) {
    console.error(`${USAGE}\npatch_wasm64_runtime.mjs: error: ${args.error}`);
    return 2;
  }
  let counts;
  try {
    const patched = patchWasm64Runtime(readText(args.target));
    counts = patched.counts;
    fs.writeFileSync(args.target, patched.text, 'utf8');
  } catch (error) {
    console.error(`error: ${error.message}`);
    return 1;
  }
  const summary = Object.entries(counts).map(([name, count]) => `${name}=${count}`).join(', ');
  console.log(`Patched wasm64 generated JavaScript: ${summary}`);
  return 0;
}

if (import.meta.main) {
  process.exitCode = main();
}
