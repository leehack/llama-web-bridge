// Tests of scripts/release/wav.mjs, the port of release_qualification.py's
// read_wav_identity. The first two tests are the Python tests of the same
// name; the rest pin the wave-module behaviour the port reproduces. WAV files
// are written byte by byte. Expected peak and rms values are CPython 3.12.3's
// (ubuntu-24.04) for the same files.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, test } from 'node:test';

import { ContractError } from '../../scripts/release/errors.mjs';
import { isPyException } from '../../scripts/release/json.mjs';
import { internals, readWavIdentity } from '../../scripts/release/wav.mjs';

let tmp;
beforeEach(() => {
  tmp = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'wav-test-')));
});
afterEach(() => {
  fs.rmSync(tmp, { recursive: true, force: true });
});

function chunk(id, body) {
  const header = Buffer.alloc(8);
  header.write(id, 0, 'latin1');
  header.writeUInt32LE(body.length, 4);
  return Buffer.concat([header, body, Buffer.alloc(body.length & 1)]);
}

function fmtBody({ formatTag = 1, channels = 1, sampleRate = 24000, bitsPerSample = 16 } = {}) {
  const blockAlign = channels * Math.ceil(bitsPerSample / 8);
  const body = Buffer.alloc(16);
  body.writeUInt16LE(formatTag, 0);
  body.writeUInt16LE(channels, 2);
  body.writeUInt32LE(sampleRate, 4);
  body.writeUInt32LE(sampleRate * blockAlign, 8);
  body.writeUInt16LE(blockAlign, 12);
  body.writeUInt16LE(bitsPerSample, 14);
  return body;
}

function pcm16(samples) {
  const data = Buffer.alloc(samples.length * 2);
  samples.forEach((sample, index) => data.writeInt16LE(sample, index * 2));
  return data;
}

function riff(...chunks) {
  const body = Buffer.concat([Buffer.from('WAVE', 'latin1'), ...chunks]);
  return Buffer.concat([chunk('RIFF', body).subarray(0, 8), body]);
}

// What write_wav() in release_qualification_test.py writes: wave.open(..., "wb")
// with 16-bit frames 0..7 repeated per channel.
function writeWav(filePath, { sampleRate = 24000, channels = 1 } = {}) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const samples = Array.from({ length: channels }, () => [0, 1, 2, 3, 4, 5, 6, 7]).flat();
  fs.writeFileSync(filePath, riff(chunk('fmt ', fmtBody({ channels, sampleRate })), chunk('data', pcm16(samples))));
  return filePath;
}

function writeBytes(name, bytes) {
  const filePath = path.join(tmp, name);
  fs.writeFileSync(filePath, bytes);
  return filePath;
}

function assertNotReadable(filePath, reason) {
  assert.throws(() => readWavIdentity(filePath), (error) => error instanceof ContractError
    && error.message === `generated audio is not a readable WAV: ${reason}`);
}

test('test_generated_wav_identity_measured_from_the_file', () => {
  const wavPath = writeWav(path.join(tmp, 'audio', 'tts.wav'));
  const identity = readWavIdentity(wavPath);
  assert.equal(identity.sample_rate, 24000);
  assert.equal(identity.channels, 1);
  assert.equal(identity.bits_per_sample, 16);
  assert.equal(identity.frame_count, 8);
  assert.ok(identity.peak > 0);
  assert.ok(identity.rms > 0);
  assert.equal(identity.sha256, createHash('sha256').update(fs.readFileSync(wavPath)).digest('hex'));
});

test('test_invalid_generated_wav_rejected', () => {
  const broken = writeBytes('broken.wav', Buffer.from('not a wav at all'));
  assert.throws(() => readWavIdentity(broken), ContractError);
  const stereo = writeWav(path.join(tmp, 'stereo.wav'), { channels: 2 });
  assert.throws(() => readWavIdentity(stereo), ContractError);
  const resampled = writeWav(path.join(tmp, 'resampled.wav'), { sampleRate: 16000 });
  assert.throws(() => readWavIdentity(resampled), ContractError);
});

// --- wave-module parity --------------------------------------------------------

test('the identity is the file wave.open() writes, with CPython peak and rms bits', () => {
  const ramp = writeWav(path.join(tmp, 'ramp.wav'));
  const bytes = fs.readFileSync(ramp);
  assert.equal(
    bytes.subarray(0, 44).toString('hex'),
    '524946463400000057415645666d74201000000001000100c05d000080bb0000020010006461746110000000',
  );
  assert.deepEqual(readWavIdentity(ramp), {
    sha256: createHash('sha256').update(bytes).digest('hex'),
    byte_length: 60,
    channels: 1,
    bits_per_sample: 16,
    sample_rate: 24000,
    frame_count: 8,
    peak: 0.000213623046875,
    rms: 0.00012766418861909112,
  });
  const extremes = writeBytes('extremes.wav', riff(
    chunk('fmt ', fmtBody()),
    chunk('data', pcm16([-32768, 32767, 1, -1, 12345, -23456, 0, 7])),
  ));
  const identity = readWavIdentity(extremes);
  assert.equal(identity.peak, 1.0);
  assert.equal(identity.rms, 0.5760067692084888);
});

test('WAVE_FORMAT_EXTENSIBLE with the PCM subformat is accepted', () => {
  const pcmGuid = Buffer.from('0100000000001000800000aa00389b71', 'hex');
  const extension = Buffer.alloc(8);
  extension.writeUInt16LE(22, 0);
  extension.writeUInt16LE(16, 2);
  extension.writeUInt32LE(4, 4);
  const fmt = Buffer.concat([fmtBody({ formatTag: 0xfffe }), extension, pcmGuid]);
  const wavPath = writeBytes('extensible.wav', riff(chunk('fmt ', fmt), chunk('data', pcm16([1, 2, 3]))));
  assert.equal(readWavIdentity(wavPath).frame_count, 3);

  const floatGuid = Buffer.from(pcmGuid);
  floatGuid[0] = 3;
  const other = writeBytes('extensible-float.wav', riff(
    chunk('fmt ', Buffer.concat([fmtBody({ formatTag: 0xfffe }), extension, floatGuid])),
    chunk('data', pcm16([1])),
  ));
  assertNotReadable(other, 'unknown extended format: 00000003-0000-0010-8000-00aa00389b71');
});

test('chunks are word aligned and unknown ones skipped', () => {
  const wavPath = writeBytes('junk.wav', riff(
    chunk('LIST', Buffer.from('odd')),
    chunk('fmt ', fmtBody()),
    chunk('data', pcm16([5, -5])),
  ));
  assert.equal(readWavIdentity(wavPath).frame_count, 2);
});

test('wave.Error and EOFError become ContractError with wave\'s text', () => {
  assertNotReadable(writeBytes('rifx.wav', Buffer.from('RIFX\x04\x00\x00\x00WAVE', 'latin1')), 'file does not start with RIFF id');
  assertNotReadable(writeBytes('avi.wav', Buffer.from('RIFF\x04\x00\x00\x00AVI ', 'latin1')), 'not a WAVE file');
  assertNotReadable(writeBytes('data-first.wav', riff(chunk('data', pcm16([1])), chunk('fmt ', fmtBody()))), 'data chunk before fmt chunk');
  assertNotReadable(writeBytes('no-data.wav', riff(chunk('fmt ', fmtBody()))), 'fmt chunk and/or data chunk missing');
  assertNotReadable(writeBytes('float.wav', riff(chunk('fmt ', fmtBody({ formatTag: 3 })), chunk('data', pcm16([1])))), 'unknown format: 3');
  assertNotReadable(writeBytes('zero-channels.wav', riff(chunk('fmt ', fmtBody({ channels: 0 })), chunk('data', pcm16([1])))), 'bad # of channels');
  assertNotReadable(writeBytes('zero-width.wav', riff(chunk('fmt ', fmtBody({ bitsPerSample: 0 })), chunk('data', pcm16([1])))), 'bad sample width');
  // EOFError has no message, so the reason is empty.
  assertNotReadable(writeBytes('short.wav', Buffer.from('RIF', 'latin1')), '');
  assertNotReadable(writeBytes('short-fmt.wav', riff(chunk('fmt ', Buffer.alloc(10)), chunk('data', pcm16([1])))), '');
});

test('the audio checks reject what Python rejects, with its messages', () => {
  const cases = [
    [riff(chunk('fmt ', fmtBody({ bitsPerSample: 8 })), chunk('data', Buffer.from([1, 2]))), 'generated audio must be mono PCM16'],
    [riff(chunk('fmt ', fmtBody({ sampleRate: 44100 })), chunk('data', pcm16([1]))), 'generated audio sample rate must be 24000, got 44100'],
    [riff(chunk('fmt ', fmtBody()), chunk('data', Buffer.from([1]))), 'generated audio contains no frames'],
  ];
  for (const [index, [bytes, message]] of cases.entries()) {
    assert.throws(() => readWavIdentity(writeBytes(`case-${index}.wav`, bytes)), { name: 'ContractError', message });
  }
  // A data chunk that claims more bytes than the file holds.
  const truncated = Buffer.concat([riff(chunk('fmt ', fmtBody())), Buffer.from('data\x10\x00\x00\x00\x01\x00', 'latin1')]);
  truncated.writeUInt32LE(truncated.length - 8, 4);
  assert.throws(() => readWavIdentity(writeBytes('truncated.wav', truncated)), {
    name: 'ContractError',
    message: 'generated audio PCM payload is truncated',
  });
});

test('a chunk that overruns the RIFF chunk raises RuntimeError, as wave does', () => {
  const bytes = riff(chunk('fmt ', fmtBody()), chunk('LIST', Buffer.alloc(4)), chunk('data', pcm16([1])));
  // Grow LIST's size past the end of the RIFF chunk.
  bytes.writeUInt32LE(1000, 12 + 8 + 16 + 4);
  assert.throws(() => readWavIdentity(writeBytes('overrun.wav', bytes)), (error) => isPyException(error, 'RuntimeError') && error.message === '');
});

test('an unreadable path raises OSError with Python\'s text', () => {
  const missing = path.join(tmp, 'missing.wav');
  assert.throws(() => readWavIdentity(missing), (error) => isPyException(error, 'OSError')
    && error.message === `[Errno 2] No such file or directory: '${missing}'`);
});

test('pySumFloats is Python 3.12\'s compensated sum()', () => {
  const { pySumFloats } = internals;
  assert.equal(pySumFloats(new Float64Array(10).fill(0.1)), 1.0);
  assert.equal(pySumFloats([1e100, 1.0, -1e100]), 1.0);
  assert.equal(pySumFloats([0.5, 0.25]), 0.75);
  assert.ok(Number.isNaN(pySumFloats([Infinity, -Infinity])));
});
