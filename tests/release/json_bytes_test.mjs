// Tests of the bytes and pathlib primitives of scripts/release/json.mjs that
// the manifest and publication-state ports use. Expected values are CPython
// 3.12's output for the same input.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

import {
  PyException,
  pyBytesStrip,
  pyDecodeJsonBytes,
  pyFsDecode,
  pyIsDir,
  pyIsFile,
  pyIsSymlink,
  pyJoinPath,
  pyJsonDetectEncoding,
  pyJsonLoadsBytes,
  pyListdir,
  pyRepr,
  pyWriteText,
} from '../../scripts/release/json.mjs';

function pyError(fn) {
  try {
    fn();
  } catch (error) {
    if (error instanceof PyException) return `${error.pyType}: ${error.message}`;
    throw error;
  }
  return null;
}

function bytes(...values) {
  return Buffer.from(values.map((value) => (typeof value === 'string' ? Buffer.from(value, 'latin1') : Buffer.from([value]))).reduce((all, part) => Buffer.concat([all, part]), Buffer.alloc(0)));
}

function withTemporaryDirectory(fn) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'release-json-bytes-'));
  try {
    return fn(directory);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
}

test('json.detect_encoding picks the BOM or NUL-pattern encoding', () => {
  const cases = [
    [bytes(0x00, 0x00, 0xfe, 0xff), 'utf-32'],
    [bytes(0xff, 0xfe, 0x00, 0x00), 'utf-32'],
    [bytes(0xff, 0xfe), 'utf-16'],
    [bytes(0xfe, 0xff, 0x00), 'utf-16'],
    [bytes(0xef, 0xbb, 0xbf), 'utf-8-sig'],
    [bytes('\x00\x00\x00{'), 'utf-32-be'],
    [bytes('\x00{\x00}'), 'utf-16-be'],
    [bytes('{\x00\x00\x00'), 'utf-32-le'],
    [bytes('{\x00}\x00'), 'utf-16-le'],
    [bytes('{\x00\x00\xd8'), 'utf-16-le'],
    [bytes('\x00{'), 'utf-16-be'],
    [bytes('{\x00'), 'utf-16-le'],
    [bytes('\x00\x00\x00'), 'utf-8'],
    [bytes('{}'), 'utf-8'],
  ];
  for (const [input, expected] of cases) assert.equal(pyJsonDetectEncoding(input), expected, input.toString('hex'));
});

test('json.loads(bytes) decodes with surrogatepass and reports CPython errors', () => {
  const cases = [
    [bytes('\xff\xfe{\x00}\x00x'), "UnicodeDecodeError: 'utf-16-le' codec can't decode byte 0x78 in position 6: truncated data"],
    [bytes('{\x00}\x00x'), "UnicodeDecodeError: 'utf-16-le' codec can't decode byte 0x78 in position 4: truncated data"],
    [bytes('\x00{\x00}\x00'), "UnicodeDecodeError: 'utf-16-be' codec can't decode byte 0x00 in position 4: truncated data"],
    [bytes('\xff\xfe\x00\xd8'), 'JSONDecodeError: Expecting value: line 1 column 1 (char 0)'],
    [bytes('\xef\xbb\xbf\xff'), "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte"],
    [bytes('"\xed\xa0A"'), "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xed in position 1: invalid continuation byte"],
    [bytes('\xed\xa0'), "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xed in position 0: invalid continuation byte"],
    [bytes('"\xf0\x90\x80"'), "UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 1-3: invalid continuation byte"],
    [bytes('\xc3'), "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 0: unexpected end of data"],
    [bytes('\xff\xfe\x00\x00{\x00\x00\x00}'), "UnicodeDecodeError: 'utf-32-le' codec can't decode byte 0x7d in position 8: truncated data"],
    [bytes('\x00\x00\x00{\x00\x00\x00}\x00\x00'), "UnicodeDecodeError: 'utf-32-be' codec can't decode bytes in position 8-9: truncated data"],
    [bytes('\x00\x00\x00{\x00\x11\x00\x00\x00'), "UnicodeDecodeError: 'utf-32-be' codec can't decode bytes in position 4-7: code point not in range(0x110000)"],
    [bytes('\xff\xfe\x00\x00\x00\x00\x00\x01'), "UnicodeDecodeError: 'utf-32-le' codec can't decode bytes in position 4-7: code point not in range(0x110000)"],
    [bytes('\x00\x00\x00{\x00\x00\xd8\x00'), 'JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)'],
    [bytes('\xef\xbb\xbf\xef\xbb\xbf{}'), 'JSONDecodeError: Expecting value: line 1 column 1 (char 0)'],
    [bytes('\xff\xfe'), 'JSONDecodeError: Expecting value: line 1 column 1 (char 0)'],
    [bytes('\x00'), 'JSONDecodeError: Expecting value: line 1 column 1 (char 0)'],
  ];
  for (const [input, expected] of cases) assert.equal(pyError(() => pyJsonLoadsBytes(input)), expected, input.toString('hex'));

  assert.equal(pyRepr(pyJsonLoadsBytes(bytes('"\xed\xa0\x80"'))), "'\\ud800'");
  assert.deepEqual(pyJsonLoadsBytes(bytes('\x00\x00\xfe\xff\x00\x00\x00{\x00\x00\x00}')), {});
  assert.deepEqual(pyJsonLoadsBytes(bytes('\xef\xbb\xbf{"a": 1}')), { a: 1 });
  assert.equal(pyJsonLoadsBytes(bytes('"\x00\x00\xd8\x00\xdc"\x00')), '\u{10000}');
  assert.equal(pyDecodeJsonBytes(bytes('{\x00\x00\xd8')), '{\ud800');
});

test('bytes.strip removes ASCII whitespace only', () => {
  assert.deepEqual(pyBytesStrip(bytes(' \t\x0b\x0c\r\nab \x85')), bytes('ab \x85'));
  assert.equal(pyBytesStrip(bytes(' \n')).length, 0);
});

test('os.fsdecode escapes undecodable bytes as surrogates', () => {
  assert.equal(
    pyRepr(pyFsDecode(bytes('a\xff\xc3\xa9\xed\xa0\x80\xf0\x90\x80'))),
    "'a\\udcffé\\udced\\udca0\\udc80\\udcf0\\udc90\\udc80'",
  );
});

test('Path / name joins like pathlib', () => {
  assert.equal(pyJoinPath('/', 'n'), '/n');
  assert.equal(pyJoinPath('//', 'n'), '//n');
  assert.equal(pyJoinPath('x', 'n'), 'x/n');
});

test('pathlib predicates, listdir and write_text report OSError like Python', () => {
  withTemporaryDirectory((directory) => {
    const file = path.join(directory, 'file');
    fs.writeFileSync(file, 'x');
    fs.symlinkSync(file, path.join(directory, 'link'));
    fs.symlinkSync(path.join(directory, 'nowhere'), path.join(directory, 'dangling'));
    fs.mkdirSync(path.join(directory, 'sub'));
    assert.equal(pyIsDir(directory), true);
    assert.equal(pyIsDir(file), false);
    assert.equal(pyIsFile(file), true);
    assert.equal(pyIsFile(path.join(directory, 'link')), true);
    assert.equal(pyIsSymlink(path.join(directory, 'link')), true);
    assert.equal(pyIsFile(path.join(directory, 'dangling')), false);
    assert.equal(pyIsSymlink(path.join(directory, 'dangling')), true);
    assert.equal(pyIsDir(path.join(file, 'below')), false);
    assert.equal(pyIsDir(path.join(directory, 'missing')), false);
    assert.deepEqual(pyListdir(directory).sort(), ['dangling', 'file', 'link', 'sub']);
    assert.equal(pyError(() => pyListdir(file)), `NotADirectoryError: [Errno 20] Not a directory: ${pyRepr(file)}`);
    const missing = path.join(directory, 'missing');
    assert.equal(pyError(() => pyListdir(missing)), `FileNotFoundError: [Errno 2] No such file or directory: ${pyRepr(missing)}`);
    const target = path.join(directory, 'sub');
    assert.equal(pyError(() => pyWriteText(target, 'x')), `IsADirectoryError: [Errno 21] Is a directory: ${pyRepr(target)}`);
    pyWriteText(path.join(directory, 'written'), 'text\n');
    assert.equal(fs.readFileSync(path.join(directory, 'written'), 'utf8'), 'text\n');
  });
});
