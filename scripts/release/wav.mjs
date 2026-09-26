// Generated-WAV identity: the port of release_qualification.py's
// read_wav_identity and the part of CPython 3.12's wave module it relies on.
//
// The GitHub runners qualify with the system Python 3.12, so this accepts and
// rejects exactly what 3.12's wave.open() does: the same RIFF chunk walk
// (wave._Chunk, including word alignment, reads bounded by the enclosing RIFF
// chunk, and the RuntimeError a chunk that overruns the RIFF chunk raises
// when skipped), WAVE_FORMAT_PCM and WAVE_FORMAT_EXTENSIBLE with the PCM
// subformat, and the same error text. peak and rms are computed with the
// same float operations, including 3.12's Neumaier-compensated sum(), so the
// results are bit-identical.

import { createHash } from 'node:crypto';

import { ContractError } from './errors.mjs';
import { PyException, pyReadBytes } from './json.mjs';

export const TTS_WAV_SAMPLE_RATE = 24000;
export const TTS_WAV_CHANNELS = 1;
export const TTS_WAV_BITS_PER_SAMPLE = 16;

const WAVE_FORMAT_PCM = 0x0001;
const WAVE_FORMAT_EXTENSIBLE = 0xfffe;
const KSDATAFORMAT_SUBTYPE_PCM = Buffer.from('0100000000001000800000aa00389b71', 'hex');

// wave.Error
class WaveError extends Error {}
// EOFError
class WaveEOFError extends Error {}

// A read-only file object over the bytes wave.open() would read.
class BytesFile {
  constructor(data) {
    this.data = data;
    this.position = 0;
  }

  read(size) {
    const start = Math.min(this.position, this.data.length);
    const chunk = this.data.subarray(start, Math.min(start + size, this.data.length));
    this.position += chunk.length;
    return chunk;
  }

  seek(position) {
    this.position = position;
  }

  tell() {
    return this.position;
  }
}

// wave._Chunk (little-endian, word-aligned).
class Chunk {
  constructor(file) {
    this.file = file;
    this.chunkname = file.read(4);
    if (this.chunkname.length < 4) throw new WaveEOFError('');
    const size = file.read(4);
    if (size.length < 4) throw new WaveEOFError('');
    this.chunksize = size.readUInt32LE(0);
    this.sizeRead = 0;
    this.offset = file.tell();
  }

  name() {
    return this.chunkname.toString('latin1');
  }

  seek(position, whence = 0) {
    let target = position;
    if (whence === 1) target += this.sizeRead;
    else if (whence === 2) target += this.chunksize;
    if (target < 0 || target > this.chunksize) throw new PyException('RuntimeError', '');
    this.file.seek(this.offset + target, 0);
    this.sizeRead = target;
  }

  tell() {
    return this.sizeRead;
  }

  read(requested = -1) {
    if (this.sizeRead >= this.chunksize) return Buffer.alloc(0);
    let size = requested;
    if (size < 0) size = this.chunksize - this.sizeRead;
    if (size > this.chunksize - this.sizeRead) size = this.chunksize - this.sizeRead;
    const data = this.file.read(size);
    this.sizeRead += data.length;
    if (this.sizeRead === this.chunksize && (this.chunksize & 1)) {
      const dummy = this.file.read(1);
      this.sizeRead += dummy.length;
    }
    return data;
  }

  // Seekable skip; a seek past the enclosing chunk raises RuntimeError,
  // which _Chunk.skip() does not catch.
  skip() {
    let n = this.chunksize - this.sizeRead;
    if (this.chunksize & 1) n += 1;
    this.file.seek(n, 1);
    this.sizeRead += n;
  }
}

// str(uuid.UUID(bytes_le=value))
function uuidFromBytesLe(value) {
  const bytes = Buffer.concat([
    Buffer.from(value.subarray(0, 4)).reverse(),
    Buffer.from(value.subarray(4, 6)).reverse(),
    Buffer.from(value.subarray(6, 8)).reverse(),
    value.subarray(8, 16),
  ]).toString('hex');
  return `${bytes.slice(0, 8)}-${bytes.slice(8, 12)}-${bytes.slice(12, 16)}-${bytes.slice(16, 20)}-${bytes.slice(20)}`;
}

function readFmtChunk(chunk) {
  const header = chunk.read(14);
  if (header.length < 14) throw new WaveEOFError('');
  const formatTag = header.readUInt16LE(0);
  const nchannels = header.readUInt16LE(2);
  const framerate = header.readUInt32LE(4);
  if (formatTag !== WAVE_FORMAT_PCM && formatTag !== WAVE_FORMAT_EXTENSIBLE) {
    throw new WaveError(`unknown format: ${formatTag}`);
  }
  const width = chunk.read(2);
  if (width.length < 2) throw new WaveEOFError('');
  const sampleBits = width.readUInt16LE(0);
  if (formatTag === WAVE_FORMAT_EXTENSIBLE) {
    const extension = chunk.read(8);
    if (extension.length < 8) throw new WaveEOFError('');
    const subFormat = chunk.read(16);
    if (subFormat.length < 16) throw new WaveEOFError('');
    if (!subFormat.equals(KSDATAFORMAT_SUBTYPE_PCM)) {
      throw new WaveError(`unknown extended format: ${uuidFromBytesLe(subFormat)}`);
    }
  }
  const sampwidth = Math.floor((sampleBits + 7) / 8);
  if (!sampwidth) throw new WaveError('bad sample width');
  if (!nchannels) throw new WaveError('bad # of channels');
  return { nchannels, framerate, sampwidth, framesize: nchannels * sampwidth };
}

// wave.open(path, 'rb') followed by the getters and readframes(nframes).
function readWave(data) {
  const riff = new Chunk(new BytesFile(data));
  if (riff.name() !== 'RIFF') throw new WaveError('file does not start with RIFF id');
  if (riff.read(4).toString('latin1') !== 'WAVE') throw new WaveError('not a WAVE file');
  let format = null;
  let dataChunk = null;
  let nframes = 0;
  for (;;) {
    let chunk;
    try {
      chunk = new Chunk(riff);
    } catch (error) {
      if (error instanceof WaveEOFError) break;
      throw error;
    }
    const name = chunk.name();
    if (name === 'fmt ') {
      format = readFmtChunk(chunk);
    } else if (name === 'data') {
      if (!format) throw new WaveError('data chunk before fmt chunk');
      dataChunk = chunk;
      nframes = Math.floor(chunk.chunksize / format.framesize);
      break;
    }
    chunk.skip();
  }
  if (!format || !dataChunk) throw new WaveError('fmt chunk and/or data chunk missing');
  const pcm = nframes === 0 ? Buffer.alloc(0) : dataChunk.read(nframes * format.framesize);
  return { ...format, nframes, pcm };
}

// Python 3.12's sum() over floats, one term at a time: Neumaier compensated
// summation, the compensation added once at the end unless it is 0 or not
// finite.
class PyFloatSum {
  #result = 0;
  #compensation = 0;
  #count = 0;

  add(x) {
    if (this.#count === 0) {
      this.#result = x;
    } else {
      const t = this.#result + x;
      if (Math.abs(this.#result) >= Math.abs(x)) this.#compensation += (this.#result - t) + x;
      else this.#compensation += (x - t) + this.#result;
      this.#result = t;
    }
    this.#count += 1;
  }

  get value() {
    const compensation = this.#compensation;
    return compensation && Number.isFinite(compensation) ? this.#result + compensation : this.#result;
  }
}

// sum(values) over a non-empty sequence of floats.
function pySumFloats(values) {
  const sum = new PyFloatSum();
  for (const value of values) sum.add(value);
  return sum.value;
}

// Verify a generated WAV container and return its measured identity. The
// keys are the attestation's. peak and rms are plain numbers; they are Python
// floats in the attestation, so whoever serializes them must keep them floats
// (1.0, not 1).
export function readWavIdentity(wavPath) {
  const data = pyReadBytes(String(wavPath));
  let wave;
  try {
    wave = readWave(data);
  } catch (error) {
    if (error instanceof WaveError || error instanceof WaveEOFError) {
      throw new ContractError(`generated audio is not a readable WAV: ${error.message}`, { cause: error });
    }
    throw error;
  }
  const { nchannels: channels, sampwidth: sampleWidth, framerate: sampleRate, nframes: frameCount, pcm } = wave;
  if (channels !== TTS_WAV_CHANNELS || sampleWidth * 8 !== TTS_WAV_BITS_PER_SAMPLE) {
    throw new ContractError('generated audio must be mono PCM16');
  }
  if (sampleRate !== TTS_WAV_SAMPLE_RATE) {
    throw new ContractError(`generated audio sample rate must be ${TTS_WAV_SAMPLE_RATE}, got ${sampleRate}`);
  }
  if (frameCount <= 0) throw new ContractError('generated audio contains no frames');
  const expectedPcmBytes = frameCount * channels * sampleWidth;
  if (pcm.length !== expectedPcmBytes) throw new ContractError('generated audio PCM payload is truncated');
  const count = frameCount * channels;
  let peak = -Infinity;
  const squares = new PyFloatSum();
  for (let index = 0; index < count; index += 1) {
    const sample = pcm.readInt16LE(index * 2);
    const magnitude = Math.abs(sample) / 32768.0;
    if (magnitude > peak) peak = magnitude;
    const scaled = sample / 32768.0;
    squares.add(scaled * scaled);
  }
  const rms = Math.sqrt(squares.value / count);
  return {
    sha256: createHash('sha256').update(data).digest('hex'),
    byte_length: data.length,
    channels,
    bits_per_sample: sampleWidth * 8,
    sample_rate: sampleRate,
    frame_count: frameCount,
    peak,
    rms,
  };
}

// Exposed for the tests only.
export const internals = { pySumFloats };
