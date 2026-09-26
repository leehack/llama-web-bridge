// Fail-closed extraction of GitHub artifact archives: the port of
// release_qualification.py's _extract_flat_artifact_archive and the checks it
// relies on.
//
// Parity with the Python extractor covers the accept/reject decision, the
// error text, and the extracted bytes and file set. Three layers make that
// hold:
//   * the hand-written checks (end record, local headers, data descriptors,
//     names, bounds, compression ratio) are a line-for-line port;
//   * readZipFile() and ZipExtFile reproduce what CPython 3.12's zipfile does
//     with the central directory and while streaming a member, including the
//     errors it raises and the ones it lets escape (zlib.error,
//     UnicodeDecodeError, NotImplementedError surface as PyException);
//   * DEFLATE goes through node:zlib in one pass per member, bounded to the
//     member's declared size. Python streams the member through
//     zlib.decompressobj and stops once it has file_size bytes, so it never
//     sees what the stream holds past its read window. Here any zlib data
//     error anywhere in the member's compressed payload is a zlib.error, and
//     a stream that decodes past file_size is rejected as a BadZipFile;
//     Python truncates such a stream, then reports a Bad CRC-32 (or accepts
//     it when the truncated bytes happen to match the CRC). Both only ever
//     reject input Python also rejects or cannot tell from a malformed member.
//
// Filesystem hardening. Python pins the destination with
// os.open(O_NOFOLLOW|O_DIRECTORY) and then works relative to that descriptor
// (os.listdir(fd), os.replace(dst_dir_fd=), os.unlink(dir_fd=)), so a
// destination swapped for a symlink mid-extraction can never redirect a
// write. Node has no *at() calls. The closest equivalent used here:
//   * the destination is opened with O_RDONLY|O_DIRECTORY|O_NOFOLLOW and its
//     (dev, ino) recorded, exactly as Python does;
//   * on Linux, every later operation goes through /proc/self/fd/<fd>/<name>.
//     The kernel resolves that magic link to the pinned directory itself, so
//     listing, placing and unlinking hit the same directory Python's dir_fd
//     calls hit, even after a swap. This is the mode the GitHub runners use.
//     It is used only after stat(/proc/self/fd/<fd>) proves it names the
//     pinned directory; on Linux, extraction is refused when that proof fails
//     (no /proc), never downgraded to path mode;
//   * elsewhere (macOS) operations go by path, each
//     bracketed by lstat() (dev, ino) checks of the destination against the
//     pinned identity. A swap that lands between a check and the rename can
//     place a member through the attacker's path for a moment. The final
//     verification then fails exactly as in Python, and cleanup removes the
//     member from wherever the path led, only if its (dev, ino) is the one
//     staged here; a member that cannot be found that way is reported as a
//     cleanup error instead of silently left behind.
//   * members are staged in a private 0700 directory inside the destination's
//     parent (fs.mkdtempSync, like tempfile.mkdtemp), opened for writing with
//     O_NOFOLLOW, moved in with rename(2), and the destination is re-verified
//     afterwards: still a directory, same (dev, ino), exactly the placed names.

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import zlib from 'node:zlib';

import { ContractError } from './errors.mjs';
import {
  PyException,
  compareCodePoints,
  isPyException,
  pyDecodeUtf8,
  pyFloatRepr,
  pyPath,
  pyRepr,
} from './json.mjs';
import { PUBLICATION_FILES } from './publication_state.mjs';
import { isOSError, osErrorString, pyDecodeCp437, pyReprBytes } from './python_compat.mjs';

// generate_release_manifest.ARTIFACTS plus the two generated files, the one
// definition publication validates candidates against.
export { PUBLICATION_FILES };

export const MAX_ATTESTATION_BYTES = 32768;
export const MAX_COMPRESSION_RATIO = 100.0;
export const MAX_CANDIDATE_MEMBER_BYTES = 64 * 1024 * 1024;
export const MAX_CANDIDATE_TOTAL_BYTES = 256 * 1024 * 1024;
export const CANDIDATE_ALLOWED_MEMBERS = Object.freeze(new Set(PUBLICATION_FILES));
export const MAX_ATTESTATION_MEMBER_BYTES = MAX_ATTESTATION_BYTES;
export const MAX_ATTESTATION_TOTAL_BYTES = MAX_ATTESTATION_BYTES;
export const ATTESTATION_ALLOWED_MEMBERS = Object.freeze(new Set(['qualification-attestation.json']));

const ZIP_STORED = 0;
const ZIP_DEFLATED = 8;
export const ALLOWED_COMPRESS_TYPES = Object.freeze(new Set([ZIP_STORED, ZIP_DEFLATED]));
const EXTRACT_CHUNK_BYTES = 64 * 1024;
const LOCAL_HEADER_SIGNATURE = Buffer.from('PK\x03\x04', 'latin1');
const LOCAL_HEADER_SIZE = 30;
const DATA_DESCRIPTOR_SIGNATURE = Buffer.from('PK\x07\x08', 'latin1');
const DATA_DESCRIPTOR_SIZE = 16;
const END_OF_CENTRAL_DIRECTORY_SIGNATURE = Buffer.from('PK\x05\x06', 'latin1');
const END_OF_CENTRAL_DIRECTORY_SIZE = 22;
const MAX_CENTRAL_DIRECTORY_BYTES = 2 * 1024 * 1024;
const ZIP_ENCRYPTED_FLAG = 0x1;
const ZIP_STRONG_ENCRYPTION_FLAG = 0x40;
const ZIP_MASKED_HEADER_FLAG = 0x2000;
const ZIP_ENCRYPTION_FLAGS = ZIP_ENCRYPTED_FLAG | ZIP_STRONG_ENCRYPTION_FLAG | ZIP_MASKED_HEADER_FLAG;
const ZIP_DATA_DESCRIPTOR_FLAG = 0x8;
const ZIP_UTF8_NAME_FLAG = 0x800;
const S_IFMT = 0o170000;
const S_IFREG = 0o100000;

// zipfile.BadZipFile. Only raised inside the archive reader, which turns it
// into a ContractError like Python's `except zipfile.BadZipFile`.
class BadZipFile extends Error {
  constructor(message) {
    super(message);
    this.name = 'BadZipFile';
  }
}

// Attach the filename Python's OSError would carry.
function withFilename(filename, operation) {
  try {
    return operation();
  } catch (error) {
    if (isOSError(error) && error.pyFilename === undefined) error.pyFilename = filename;
    throw error;
  }
}

// f"{value:.1f}": JavaScript rounds ties up where Python rounds half to even.
// A double has a one-decimal tie only at .25 or .75, and only .25 differs.
function pyFixed1(value) {
  const quarters = value * 4;
  if (Number.isInteger(quarters) && ((quarters % 4) + 4) % 4 === 1) {
    return `${Math.trunc(value)}.2`;
  }
  return value.toFixed(1);
}

export function artifactArchiveBounds(artifactType) {
  if (artifactType === 'candidate') {
    return [CANDIDATE_ALLOWED_MEMBERS, MAX_CANDIDATE_MEMBER_BYTES, MAX_CANDIDATE_TOTAL_BYTES];
  }
  if (artifactType === 'attestation') {
    return [ATTESTATION_ALLOWED_MEMBERS, MAX_ATTESTATION_MEMBER_BYTES, MAX_ATTESTATION_TOTAL_BYTES];
  }
  throw new ContractError(`unknown artifact type: ${pyRepr(artifactType)}`);
}

// A positioned reader over the open archive, like Python's seek() + read().
class ArchiveReader {
  constructor(fd) {
    this.fd = fd;
  }

  size() {
    return fs.fstatSync(this.fd).size;
  }

  // Read up to `length` bytes at `position` (a Number or BigInt); short at EOF.
  read(position, length) {
    const start = typeof position === 'bigint' ? position : BigInt(position);
    if (start < 0n) throw new PyException('ValueError', `negative seek value ${start}`);
    if (length <= 0) return Buffer.alloc(0);
    if (start > BigInt(Number.MAX_SAFE_INTEGER)) return Buffer.alloc(0);
    const buffer = Buffer.alloc(length);
    let filled = 0;
    while (filled < length) {
      const count = fs.readSync(this.fd, buffer, filled, length - filled, Number(start) + filled);
      if (count === 0) break;
      filled += count;
    }
    return buffer.subarray(0, filled);
  }
}

function preflightZipEndRecord(reader, archiveSize, expectedMemberCount) {
  const fixedSize = END_OF_CENTRAL_DIRECTORY_SIZE;
  if (archiveSize < fixedSize) {
    throw new ContractError('artifact archive has no complete end-of-central-directory record');
  }
  const fixed = reader.read(archiveSize - fixedSize, fixedSize);
  if (fixed.length !== fixedSize) {
    throw new ContractError('artifact archive has a truncated end-of-central-directory record');
  }
  const signature = fixed.subarray(0, 4);
  const diskNumber = fixed.readUInt16LE(4);
  const centralDirectoryDisk = fixed.readUInt16LE(6);
  const entriesOnDisk = fixed.readUInt16LE(8);
  const totalEntries = fixed.readUInt16LE(10);
  const centralDirectorySize = fixed.readUInt32LE(12);
  const centralDirectoryOffset = fixed.readUInt32LE(16);
  const commentLength = fixed.readUInt16LE(20);
  if (!signature.equals(END_OF_CENTRAL_DIRECTORY_SIGNATURE) || commentLength !== 0) {
    throw new ContractError(
      'artifact archive must end with an uncommented end-of-central-directory record',
    );
  }
  if (diskNumber !== 0 || centralDirectoryDisk !== 0 || entriesOnDisk !== totalEntries) {
    throw new ContractError('multi-disk artifact archives are unsupported');
  }
  if (totalEntries !== expectedMemberCount) {
    throw new ContractError(
      'artifact archive end-of-central-directory member count must be exactly '
      + `${expectedMemberCount}, got ${totalEntries}`,
    );
  }
  if (centralDirectorySize > MAX_CENTRAL_DIRECTORY_BYTES) {
    throw new ContractError('artifact archive central directory exceeds its metadata byte bound');
  }
  if (centralDirectoryOffset + centralDirectorySize !== archiveSize - fixedSize) {
    throw new ContractError('artifact archive central directory location disagrees with its end record');
  }
  return centralDirectoryOffset;
}

// zipfile._sanitize_filename on POSIX: cut at the first NUL.
function sanitizeFilename(filename) {
  const nul = filename.indexOf('\0');
  return nul >= 0 ? filename.slice(0, nul) : filename;
}

function decodeMemberName(raw, flagBits) {
  return flagBits & ZIP_UTF8_NAME_FLAG ? pyDecodeUtf8(raw) : pyDecodeCp437(raw);
}

// ZipInfo._decodeExtra(filename_crc) as CPython 3.12 runs it.
function decodeExtra(info, filenameCrc) {
  let extra = info.extra;
  while (extra.length >= 4) {
    const type = extra.readUInt16LE(0);
    const length = extra.readUInt16LE(2);
    if (length + 4 > extra.length) {
      throw new BadZipFile(`Corrupt extra field ${type.toString(16).padStart(4, '0')} (size=${length})`);
    }
    if (type === 0x0001) {
      let data = extra.subarray(4, length + 4);
      const take = (field) => {
        if (data.length < 8) throw new BadZipFile(`Corrupt zip64 extra field. ${field} not found.`);
        const value = data.readBigUInt64LE(0);
        data = data.subarray(8);
        return value;
      };
      if (info.fileSize === 0xffffffffffffffffn || info.fileSize === 0xffffffffn) {
        info.fileSize = take('File size');
      }
      if (info.compressSize === 0xffffffffn) info.compressSize = take('Compress size');
      if (info.headerOffset === 0xffffffffn) info.headerOffset = take('Header offset');
    } else if (type === 0x7075) {
      const data = extra.subarray(4, length + 4);
      if (data.length < 5) throw new BadZipFile('Corrupt unicode path extra field (0x7075)');
      const version = data[0];
      const nameCrc = data.readUInt32LE(1);
      if (version === 1 && nameCrc === filenameCrc) {
        let unicodeName;
        try {
          unicodeName = pyDecodeUtf8(data.subarray(5));
        } catch (error) {
          if (isPyException(error, 'UnicodeDecodeError')) {
            throw new BadZipFile('Corrupt unicode path extra field (0x7075): invalid utf-8 bytes');
          }
          throw error;
        }
        // An empty name only warns in CPython.
        if (unicodeName) info.filename = sanitizeFilename(unicodeName);
      }
    }
    extra = extra.subarray(length + 4);
  }
}

// zipfile._EndRecData and _EndRecData64 for an archive whose last 22 bytes
// the preflight already proved to be an uncommented end record.
function endRecord(reader) {
  const fileSize = reader.size();
  if (fileSize < END_OF_CENTRAL_DIRECTORY_SIZE) return null;
  const data = reader.read(fileSize - END_OF_CENTRAL_DIRECTORY_SIZE, END_OF_CENTRAL_DIRECTORY_SIZE);
  if (
    data.length !== END_OF_CENTRAL_DIRECTORY_SIZE
    || !data.subarray(0, 4).equals(END_OF_CENTRAL_DIRECTORY_SIGNATURE)
    || data[20] !== 0
    || data[21] !== 0
  ) {
    // CPython would go on to search for a commented record; the preflight
    // has already refused any archive that reaches this.
    return null;
  }
  const record = {
    size: BigInt(data.readUInt32LE(12)),
    offset: BigInt(data.readUInt32LE(16)),
    location: BigInt(fileSize - END_OF_CENTRAL_DIRECTORY_SIZE),
  };
  return endRecord64(reader, fileSize - END_OF_CENTRAL_DIRECTORY_SIZE, record);
}

// _EndRecData64 with the CVE-2025-8291 consistency checks, as shipped in the
// runners' Python (Ubuntu 3.12.3-1ubuntu0.17; upstream 3.12.12+). Upstream
// 3.12.11 and older skip these checks and subtract the ZIP64 record sizes
// from `concat` instead. Its OSError("Unknown I/O error") reaches
// _RealGetContents, which reports "File is not a zip file".
function endRecord64(reader, endOffset, record) {
  let offset = BigInt(endOffset) - 20n;
  if (offset < 0n) return record;
  const locator = reader.read(offset, 20);
  if (locator.length !== 20) throw new BadZipFile('File is not a zip file');
  if (!locator.subarray(0, 4).equals(Buffer.from('PK\x06\x07', 'latin1'))) return record;
  const diskNumber = locator.readUInt32LE(4);
  const relativeOffset = locator.readBigUInt64LE(8);
  const disks = locator.readUInt32LE(16);
  if (diskNumber !== 0 || disks > 1) {
    throw new BadZipFile('zipfiles that span multiple disks are not supported');
  }
  offset -= 56n;
  if (relativeOffset > offset) throw new BadZipFile('Corrupt zip64 end of central directory locator');
  let extraSize = offset - relativeOffset;
  let zip64 = reader.read(relativeOffset, 56);
  if (zip64.length !== 56) throw new BadZipFile('File is not a zip file');
  const zip64Signature = Buffer.from('PK\x06\x06', 'latin1');
  if (!zip64.subarray(0, 4).equals(zip64Signature) && relativeOffset !== offset) {
    extraSize = 0n;
    zip64 = reader.read(offset, 56);
    if (zip64.length !== 56) throw new BadZipFile('File is not a zip file');
  }
  if (!zip64.subarray(0, 4).equals(zip64Signature)) {
    throw new BadZipFile('Zip64 end of central directory record not found');
  }
  const recordSize = zip64.readBigUInt64LE(4);
  const directorySize = zip64.readBigUInt64LE(40);
  const directoryOffset = zip64.readBigUInt64LE(48);
  if (directoryOffset + directorySize !== relativeOffset || recordSize + 12n !== 56n + extraSize) {
    throw new BadZipFile('Corrupt zip64 end of central directory record');
  }
  return { size: directorySize, offset: directoryOffset, location: offset - extraSize };
}

// zipfile.ZipFile(raw) in read mode: parse the central directory the way
// the runners' CPython 3.12 _RealGetContents does. Offsets stay BigInt
// because ZIP64 records can carry 64-bit values.
export function readZipFile(reader) {
  const record = endRecord(reader);
  if (!record) throw new BadZipFile('File is not a zip file');
  const sizeCd = record.size;
  const concat = record.location - sizeCd - record.offset;
  const startDir = record.offset + concat;
  if (startDir < 0n) throw new BadZipFile('Bad offset for central directory');
  const directory = reader.read(startDir, Number(sizeCd));
  let position = 0;
  const readDirectory = (length) => {
    const chunk = directory.subarray(position, position + length);
    position += chunk.length;
    return chunk;
  };
  const members = [];
  let total = 0n;
  while (total < sizeCd) {
    const centdir = readDirectory(46);
    if (centdir.length !== 46) throw new BadZipFile('Truncated central directory');
    if (!centdir.subarray(0, 4).equals(Buffer.from('PK\x01\x02', 'latin1'))) {
      throw new BadZipFile('Bad magic number for central directory');
    }
    const filenameLength = centdir.readUInt16LE(28);
    const extraLength = centdir.readUInt16LE(30);
    const commentLength = centdir.readUInt16LE(32);
    const rawName = readDirectory(filenameLength);
    const filenameCrc = zlib.crc32(rawName);
    const flagBits = centdir.readUInt16LE(8);
    const origFilename = decodeMemberName(rawName, flagBits);
    const info = {
      origFilename,
      filename: sanitizeFilename(origFilename),
      extra: readDirectory(extraLength),
      comment: readDirectory(commentLength),
      headerOffset: BigInt(centdir.readUInt32LE(42)),
      createVersion: centdir[4],
      createSystem: centdir[5],
      extractVersion: centdir[6],
      reserved: centdir[7],
      flagBits,
      compressType: centdir.readUInt16LE(10),
      CRC: centdir.readUInt32LE(16),
      compressSize: BigInt(centdir.readUInt32LE(20)),
      fileSize: BigInt(centdir.readUInt32LE(24)),
      externalAttr: centdir.readUInt32LE(38),
      endOffset: null,
    };
    if (info.extractVersion > 63) {
      throw new PyException('NotImplementedError', `zip file version ${(info.extractVersion / 10).toFixed(1)}`);
    }
    decodeExtra(info, filenameCrc);
    info.headerOffset += concat;
    members.push(info);
    total += BigInt(46 + filenameLength + extraLength + commentLength);
  }
  // sorted(filelist, key=header_offset, reverse=True): stable, so equal
  // offsets keep their directory order.
  let endOffset = startDir;
  const byOffset = members.map((info, index) => ({ info, index }))
    .sort((a, b) => (a.info.headerOffset > b.info.headerOffset ? -1
      : a.info.headerOffset < b.info.headerOffset ? 1 : a.index - b.index));
  for (const { info } of byOffset) {
    info.endOffset = endOffset;
    endOffset = info.headerOffset;
  }
  return { startDir, members, reader };
}

// ZipFile.open(zinfo) followed by ZipExtFile reads, as in CPython 3.12.
class ZipExtFile {
  constructor(reader, info, dataStart) {
    this.reader = reader;
    this.position = dataStart;
    this.compressType = info.compressType;
    this.compressLeft = Number(info.compressSize);
    this.left = Number(info.fileSize);
    this.dataStart = dataStart;
    this.inflated = null;
    this.eof = false;
    this.readbuffer = Buffer.alloc(0);
    this.offset = 0;
    this.name = info.filename;
    this.expectedCrc = info.CRC;
    this.runningCrc = 0;
  }

  // Only called for a member stageArtifactArchive has validated, which
  // already refused what CPython checks here after the name: encryption
  // (flag bits 0x1, 0x40, 0x2000) and any method but STORED and DEFLATED, so
  // those checks are left out. The header, name and overlap checks re-read the
  // archive; they repeat what validateLocalHeader proved, and fire only if the
  // archive bytes change after that read, as Python's would. Flag bit 0x20 is
  // not refused earlier and raises NotImplementedError, as in Python.
  static open(zipFile, info) {
    const { reader } = zipFile;
    let position = info.headerOffset;
    const fheader = reader.read(position, LOCAL_HEADER_SIZE);
    position += BigInt(fheader.length);
    if (fheader.length !== LOCAL_HEADER_SIZE) throw new BadZipFile('Truncated file header');
    if (!fheader.subarray(0, 4).equals(LOCAL_HEADER_SIGNATURE)) {
      throw new BadZipFile('Bad magic number for file header');
    }
    const localFlags = fheader.readUInt16LE(6);
    const nameLength = fheader.readUInt16LE(26);
    const extraLength = fheader.readUInt16LE(28);
    const fname = reader.read(position, nameLength);
    position += BigInt(fname.length);
    if (extraLength) position += BigInt(extraLength);
    if (info.flagBits & 0x20) throw new PyException('NotImplementedError', 'compressed patched data (flag bit 5)');
    const fnameStr = decodeMemberName(fname, localFlags);
    if (fnameStr !== info.origFilename) {
      throw new BadZipFile(
        `File name in directory ${pyRepr(info.origFilename)} and header ${pyReprBytes(fname)} differ.`,
      );
    }
    if (info.endOffset !== null && position + info.compressSize > info.endOffset) {
      throw new BadZipFile(`Overlapped entries: ${pyRepr(info.origFilename)} (possible zip bomb)`);
    }
    return new ZipExtFile(reader, info, position);
  }

  read(n) {
    const end = n + this.offset;
    if (end < this.readbuffer.length) {
      const chunk = this.readbuffer.subarray(this.offset, end);
      this.offset = end;
      return chunk;
    }
    let remaining = end - this.readbuffer.length;
    const parts = [this.readbuffer.subarray(this.offset)];
    this.readbuffer = Buffer.alloc(0);
    this.offset = 0;
    while (remaining > 0 && !this.eof) {
      const data = this.#read1(remaining);
      if (remaining < data.length) {
        this.readbuffer = data;
        this.offset = remaining;
        parts.push(data.subarray(0, remaining));
        break;
      }
      parts.push(data);
      remaining -= data.length;
    }
    return Buffer.concat(parts);
  }

  #read1(n) {
    if (this.eof || n <= 0) return Buffer.alloc(0);
    let data;
    if (this.compressType === ZIP_DEFLATED) {
      this.inflated ??= this.#inflate();
      data = this.inflated.subarray(0, n);
      this.inflated = this.inflated.subarray(data.length);
      this.eof = this.inflated.length === 0;
    } else {
      data = this.#read2(n);
      this.eof = this.compressLeft <= 0;
    }
    data = data.subarray(0, Math.max(this.left, 0));
    this.left -= data.length;
    if (this.left <= 0) this.eof = true;
    this.#updateCrc(data);
    return data;
  }

  // The member's whole DEFLATE payload through node:zlib. Python decodes in
  // windows of at most one extraction read (64 KiB) and drops what lies past
  // file_size, so it never decodes more than file_size + 64 KiB; the same
  // bound applies here, and #read1 drops the same excess. Z_SYNC_FLUSH returns
  // what a truncated stream decodes to, as Python's decompressobj does; the
  // CRC and size checks then reject it.
  #inflate() {
    const payload = this.reader.read(this.dataStart, this.compressLeft);
    this.compressLeft -= payload.length;
    if (this.compressLeft > 0) throw new PyException('EOFError', '');
    let output;
    try {
      output = zlib.inflateRawSync(payload, {
        finishFlush: zlib.constants.Z_SYNC_FLUSH,
        maxOutputLength: this.left + EXTRACT_CHUNK_BYTES,
      });
    } catch (error) {
      if (error?.code === 'ERR_BUFFER_TOO_LARGE') {
        throw new BadZipFile(
          `DEFLATE stream for file ${pyRepr(this.name)} decodes more than ${EXTRACT_CHUNK_BYTES} bytes past its declared size ${this.left}`,
        );
      }
      if (error?.errno === zlib.constants.Z_DATA_ERROR) {
        throw new PyException('zlib.error', `Error -3 while decompressing data: ${error.message}`);
      }
      throw error;
    }
    return output;
  }

  #read2(n) {
    if (this.compressLeft <= 0) return Buffer.alloc(0);
    const count = Math.min(Math.max(n, 4096), this.compressLeft);
    const data = this.reader.read(this.position, count);
    this.position += BigInt(data.length);
    this.compressLeft -= data.length;
    if (data.length === 0) throw new PyException('EOFError', '');
    return data;
  }

  #updateCrc(data) {
    this.runningCrc = zlib.crc32(data, this.runningCrc);
    if (this.eof && this.runningCrc !== this.expectedCrc) {
      throw new BadZipFile(`Bad CRC-32 for file ${pyRepr(this.name)}`);
    }
  }
}

function validateLocalHeader(reader, member, archiveSize, centralDirectoryOffset) {
  const name = pyRepr(member.filename);
  if (member.headerOffset < 0n || member.headerOffset >= BigInt(archiveSize)) {
    throw new ContractError(`artifact archive member ${name} has an out-of-range header offset`);
  }
  const headerOffset = Number(member.headerOffset);
  const fixed = reader.read(headerOffset, LOCAL_HEADER_SIZE);
  if (fixed.length !== LOCAL_HEADER_SIZE) {
    throw new ContractError(`artifact archive member ${name} has a truncated local header`);
  }
  const signature = fixed.subarray(0, 4);
  const flagBits = fixed.readUInt16LE(6);
  const compressType = fixed.readUInt16LE(8);
  const crc = fixed.readUInt32LE(14);
  const compressSize = fixed.readUInt32LE(18);
  const fileSize = fixed.readUInt32LE(22);
  const nameLength = fixed.readUInt16LE(26);
  const extraLength = fixed.readUInt16LE(28);
  if (!signature.equals(LOCAL_HEADER_SIGNATURE)) {
    throw new ContractError(`artifact archive member ${name} has no local file header`);
  }
  if (flagBits !== member.flagBits) {
    throw new ContractError(
      `artifact archive member ${name} local flags 0x${flagBits.toString(16)} `
      + `disagree with central flags 0x${member.flagBits.toString(16)}`,
    );
  }
  if (flagBits & ZIP_ENCRYPTION_FLAGS) {
    throw new ContractError(`artifact archive member is encrypted: ${name}`);
  }
  if (compressType !== member.compressType) {
    throw new ContractError(
      `artifact archive member ${name} local compression method `
      + `${compressType} disagrees with the central directory`,
    );
  }
  const rawName = reader.read(headerOffset + LOCAL_HEADER_SIZE, nameLength);
  if (rawName.length !== nameLength) {
    throw new ContractError(`artifact archive member ${name} has a truncated local name`);
  }
  if (extraLength !== 0) {
    throw new ContractError(`artifact archive member ${name} carries hidden local metadata`);
  }
  let localName;
  try {
    localName = decodeMemberName(rawName, flagBits);
  } catch (error) {
    if (isPyException(error, 'UnicodeDecodeError')) {
      throw new ContractError(`artifact archive member ${name} has an undecodable local name`, { cause: error });
    }
    throw error;
  }
  if (localName !== member.origFilename) {
    throw new ContractError(
      `artifact archive member ${name} local name ${pyRepr(localName)} `
      + 'disagrees with the central directory',
    );
  }
  const dataOffset = headerOffset + LOCAL_HEADER_SIZE + nameLength + extraLength;
  const memberCompressSize = Number(member.compressSize);
  const payloadEnd = dataOffset + memberCompressSize;
  let memberEnd;
  if (flagBits & ZIP_DATA_DESCRIPTOR_FLAG) {
    if (crc !== 0 || compressSize !== 0 || fileSize !== 0) {
      throw new ContractError(
        `artifact archive member ${name} data descriptor mode requires zero local CRC and sizes`,
      );
    }
    memberEnd = payloadEnd + DATA_DESCRIPTOR_SIZE;
    if (memberEnd > archiveSize || memberEnd > centralDirectoryOffset) {
      throw new ContractError(
        `artifact archive member ${name} data descriptor overlaps the archive boundary or central directory`,
      );
    }
    const descriptor = reader.read(payloadEnd, DATA_DESCRIPTOR_SIZE);
    if (descriptor.length !== DATA_DESCRIPTOR_SIZE) {
      throw new ContractError(`artifact archive member ${name} has a truncated data descriptor`);
    }
    if (
      !descriptor.subarray(0, 4).equals(DATA_DESCRIPTOR_SIGNATURE)
      || descriptor.readUInt32LE(4) !== member.CRC
      || BigInt(descriptor.readUInt32LE(8)) !== member.compressSize
      || BigInt(descriptor.readUInt32LE(12)) !== member.fileSize
    ) {
      throw new ContractError(
        `artifact archive member ${name} data descriptor disagrees with the central directory`,
      );
    }
  } else {
    if (crc !== member.CRC || BigInt(compressSize) !== member.compressSize || BigInt(fileSize) !== member.fileSize) {
      throw new ContractError(`artifact archive member ${name} local header disagrees with the central directory`);
    }
    memberEnd = payloadEnd;
  }
  if (memberEnd > archiveSize) {
    throw new ContractError(`artifact archive member ${name} payload extends past the archive boundary`);
  }
  if (memberEnd > centralDirectoryOffset) {
    throw new ContractError(`artifact archive member ${name} payload overlaps the central directory`);
  }
  return memberEnd;
}

// Prove the whole inventory, then stream each member under hard byte caps.
// Returns the sorted member names. `zipFile` is the dependency-injection seam
// for zipfile.ZipFile.
function stageArtifactArchive(archivePath, staging, { allowedMembers, perMemberCap, totalCap, zipFile }) {
  const archive = pyPath(String(archivePath));
  try {
    const fd = withFilename(archive, () => fs.openSync(archive, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW));
    try {
      const archiveStat = fs.fstatSync(fd);
      if (!archiveStat.isFile()) throw new ContractError(`artifact archive must be a regular file: ${archive}`);
      const archiveSize = archiveStat.size;
      if (archiveSize <= 0) throw new ContractError(`artifact archive is empty: ${archive}`);
      const reader = new ArchiveReader(fd);
      const preflightCentralDirectoryOffset = preflightZipEndRecord(reader, archiveSize, allowedMembers.size);
      const parsed = zipFile(reader);
      const centralDirectoryOffset = parsed.startDir;
      if (
        typeof centralDirectoryOffset !== 'bigint'
        || centralDirectoryOffset < 0n
        || centralDirectoryOffset > BigInt(archiveSize)
        || centralDirectoryOffset !== BigInt(preflightCentralDirectoryOffset)
      ) {
        throw new ContractError('artifact archive central directory offset is out of range or ambiguous');
      }
      const directoryOffset = Number(centralDirectoryOffset);
      const { members } = parsed;
      const allowedSorted = [...allowedMembers].sort(compareCodePoints);
      if (members.length !== allowedMembers.size) {
        throw new ContractError(
          'artifact archive must contain exactly the '
          + `${allowedMembers.size} expected members `
          + `${pyRepr(allowedSorted)}, found ${members.length}`,
        );
      }
      const lowest = members.reduce((low, member) => (member.headerOffset < low ? member.headerOffset : low), members[0].headerOffset);
      if (lowest !== 0n) {
        throw new ContractError('artifact archive members must start at byte zero with no preamble');
      }
      const seen = new Set();
      let totalUncompressed = 0n;
      for (const member of members) {
        const name = pyRepr(member.filename);
        if (member.extra.length || member.comment.length) {
          throw new ContractError(`artifact archive member ${name} carries hidden metadata`);
        }
        if (member.origFilename !== member.filename) {
          throw new ContractError(`artifact archive member name is ambiguous: ${pyRepr(member.origFilename)}`);
        }
        if (member.flagBits & ZIP_ENCRYPTION_FLAGS) {
          throw new ContractError(`artifact archive member is encrypted: ${name}`);
        }
        const modeType = ((member.externalAttr >>> 16) & S_IFMT);
        if (
          member.filename.endsWith('/')
          || member.filename === ''
          || member.filename === '.'
          || member.filename === '..'
          || member.filename.includes('\\')
          || member.filename.includes('/')
          || (modeType !== 0 && modeType !== S_IFREG)
        ) {
          throw new ContractError(`artifact archive member is not a flat regular file: ${name}`);
        }
        if (!allowedMembers.has(member.filename)) {
          throw new ContractError(`artifact archive contains unauthorized member: ${name}`);
        }
        if (seen.has(member.filename)) {
          throw new ContractError(`artifact archive repeats member: ${name}`);
        }
        seen.add(member.filename);
        if (!ALLOWED_COMPRESS_TYPES.has(member.compressType)) {
          throw new ContractError(
            `artifact archive member ${name} uses unsupported compression method ${member.compressType}`,
          );
        }
        if (member.fileSize < 0n || member.fileSize > BigInt(perMemberCap)) {
          throw new ContractError(
            `artifact archive member ${name} uncompressed size ${member.fileSize} exceeds bound ${perMemberCap}`,
          );
        }
        if (member.compressSize > 0n) {
          const ratio = Number(member.fileSize) / Number(member.compressSize);
          if (ratio > MAX_COMPRESSION_RATIO) {
            throw new ContractError(
              `artifact archive member ${name} compression ratio `
              + `${pyFixed1(ratio)} exceeds maximum allowed ratio ${pyFloatRepr(MAX_COMPRESSION_RATIO)}`,
            );
          }
        } else if (member.fileSize > 0n) {
          throw new ContractError(
            `artifact archive member ${name} has zero compressed size for non-empty content`,
          );
        }
        totalUncompressed += member.fileSize;
      }
      if (totalUncompressed > BigInt(totalCap)) {
        throw new ContractError(
          `artifact archive total uncompressed size ${totalUncompressed} exceeds bound ${totalCap}`,
        );
      }

      let payloadEnd = 0;
      const byOffset = members.map((member, index) => ({ member, index }))
        .sort((a, b) => (a.member.headerOffset < b.member.headerOffset ? -1
          : a.member.headerOffset > b.member.headerOffset ? 1 : a.index - b.index));
      for (const { member } of byOffset) {
        if (member.headerOffset < BigInt(payloadEnd)) {
          throw new ContractError(`overlapping members detected in artifact archive: ${pyRepr(member.filename)}`);
        }
        if (member.headerOffset > BigInt(payloadEnd)) {
          throw new ContractError(
            `unclaimed gap detected between artifact archive members before ${pyRepr(member.filename)}`,
          );
        }
        payloadEnd = validateLocalHeader(reader, member, archiveSize, directoryOffset);
      }
      if (payloadEnd !== directoryOffset) {
        throw new ContractError('unclaimed gap detected before the artifact archive central directory');
      }

      let totalWritten = 0;
      for (const member of members) {
        const target = path.join(staging, member.filename);
        let written = 0;
        const source = ZipExtFile.open(parsed, member);
        const output = withFilename(target, () => fs.openSync(
          target,
          fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_TRUNC | fs.constants.O_NOFOLLOW,
          0o666,
        ));
        try {
          for (;;) {
            const chunk = source.read(EXTRACT_CHUNK_BYTES);
            if (chunk.length === 0) break;
            written += chunk.length;
            totalWritten += chunk.length;
            if (written > perMemberCap) {
              throw new ContractError(`member ${pyRepr(member.filename)} exceeded per-member byte cap during extraction`);
            }
            if (totalWritten > totalCap) {
              throw new ContractError('total decompressed bytes exceeded total byte cap during extraction');
            }
            let offset = 0;
            while (offset < chunk.length) offset += fs.writeSync(output, chunk, offset, chunk.length - offset);
          }
        } finally {
          fs.closeSync(output);
        }
        if (BigInt(written) !== member.fileSize) {
          throw new ContractError(
            `member ${pyRepr(member.filename)} decompressed size mismatch: ${written} != ${member.fileSize}`,
          );
        }
      }
      return [...seen].sort(compareCodePoints);
    } finally {
      fs.closeSync(fd);
    }
  } catch (error) {
    if (error instanceof BadZipFile) {
      throw new ContractError(`could not read artifact archive: ${error.message}`, { cause: error });
    }
    if (isOSError(error)) {
      throw new ContractError(`could not read artifact archive: ${osErrorString(error, error.pyFilename)}`, { cause: error });
    }
    throw error;
  }
}

// Path.mkdir(parents=True, exist_ok=True).
function pyMakeDirectories(target) {
  try {
    fs.mkdirSync(target);
  } catch (error) {
    if (!isOSError(error)) throw error;
    if (error.code === 'ENOENT') {
      const parent = pyPathParent(target);
      if (parent === target) {
        error.pyFilename = target;
        throw error;
      }
      pyMakeDirectories(parent);
      try {
        fs.mkdirSync(target);
      } catch (retryError) {
        if (!isOSError(retryError)) throw retryError;
        if (!pyIsDir(target)) {
          retryError.pyFilename = target;
          throw retryError;
        }
      }
      return;
    }
    if (!pyIsDir(target)) {
      error.pyFilename = target;
      throw error;
    }
  }
}

function pyIsDir(target) {
  try {
    return fs.statSync(target).isDirectory();
  } catch {
    return false;
  }
}

// str(PurePosixPath(text).parent) for an already normalized path.
function pyPathParent(normalized) {
  if (normalized === '/' || normalized === '//' || normalized === '.') return normalized;
  const cut = normalized.lastIndexOf('/');
  if (cut < 0) return '.';
  if (cut === 0) return '/';
  if (cut === 1 && normalized.startsWith('//')) return '//';
  return normalized.slice(0, cut);
}

const identityOf = (stat) => `${stat.dev}:${stat.ino}`;

// The destination directory pinned by descriptor, and every operation Python
// performs relative to that descriptor.
//
// pinMode 'auto' is 'proc' on Linux and 'path' elsewhere. Proc mode fails
// closed when /proc/self/fd/<fd> cannot be proved to name the pinned
// directory; it never falls back to path mode. `platform` and `procFdRoot`
// are seams for the tests.
class PinnedDirectory {
  constructor(pathText, fd, identity, pinMode, { platform = process.platform, procFdRoot = '/proc/self/fd' } = {}) {
    this.path = pathText;
    this.fd = fd;
    this.identity = identity;
    this.procPath = null;
    const mode = pinMode === 'auto' ? (platform === 'linux' ? 'proc' : 'path') : pinMode;
    if (mode === 'proc') {
      const candidate = `${procFdRoot}/${fd}`;
      let stat = null;
      try {
        stat = fs.statSync(candidate, { bigint: true });
      } catch {
        // reported below
      }
      if (stat === null || identityOf(stat) !== identity) {
        throw new ContractError(`artifact destination cannot be pinned: ${candidate} does not name the pinned directory`);
      }
      this.procPath = candidate;
    } else if (mode !== 'path') {
      throw new ContractError(`unknown artifact destination pin mode: ${pyRepr(pinMode)}`);
    }
  }

  get mode() {
    return this.procPath === null ? 'path' : 'proc';
  }

  // lstat(destination) still names the pinned directory.
  #assertSamePath() {
    let current;
    try {
      current = fs.lstatSync(this.path, { bigint: true });
    } catch (error) {
      if (!isOSError(error)) throw error;
      throw new ContractError(`artifact destination changed during extraction: ${osErrorString(error, this.path)}`, { cause: error });
    }
    if (!current.isDirectory() || identityOf(current) !== this.identity) {
      throw new ContractError('artifact destination changed during extraction');
    }
  }

  // Path of `name` inside the pinned directory.
  entry(name) {
    if (this.procPath !== null) return `${this.procPath}/${name}`;
    this.#assertSamePath();
    return path.join(this.path, name);
  }

  // os.listdir(fd)
  list() {
    if (this.procPath !== null) return fs.readdirSync(this.procPath);
    this.#assertSamePath();
    const names = fs.readdirSync(this.path);
    this.#assertSamePath();
    return names;
  }

  // _unlink_placed_members: remove only the members this extraction placed.
  unlinkPlaced(names, stagedIdentities) {
    const errors = [];
    for (const name of names) {
      const target = this.procPath !== null ? `${this.procPath}/${name}` : path.join(this.path, name);
      try {
        if (this.procPath === null) {
          const current = fs.lstatSync(target, { bigint: true });
          if (identityOf(current) !== stagedIdentities.get(name)) {
            errors.push(`${name}: placed member is no longer at its destination path`);
            continue;
          }
        }
        fs.unlinkSync(target);
      } catch (error) {
        if (!isOSError(error)) throw error;
        if (error.code === 'ENOENT') continue;
        errors.push(`${name}: ${osErrorString(error, name)}`);
      }
    }
    return errors;
  }
}

// Extract a GitHub artifact only when every member is a unique flat allowed
// file. Members are staged outside the destination and moved in only once
// the whole archive has passed, so a rejected archive never leaves partially
// written files behind.
//
// options:
//   artifactType: 'candidate' (default) or 'attestation'
//   replace(source, target): the rename primitive (os.replace); a seam for
//     injecting placement failures and destination swaps
//   zipFile(reader): the central directory reader (zipfile.ZipFile); a seam
//     for proving it is never reached
//   pinMode: 'auto' (default: 'proc' on Linux, 'path' elsewhere), 'proc' or
//     'path'
//   platform, procFdRoot: what 'auto' takes for process.platform and
//     /proc/self/fd; seams for proving that proc mode fails closed
export function extractFlatArtifactArchive(archivePath, destination, options = {}) {
  const {
    artifactType = 'candidate',
    replace = (source, target) => fs.renameSync(source, target),
    zipFile = readZipFile,
    pinMode = 'auto',
    platform = process.platform,
    procFdRoot = '/proc/self/fd',
  } = options;
  const [allowedMembers, perMemberCap, totalCap] = artifactArchiveBounds(artifactType);
  const destinationPath = pyPath(String(destination));

  let fd;
  try {
    pyMakeDirectories(destinationPath);
    fd = withFilename(destinationPath, () => fs.openSync(
      destinationPath,
      fs.constants.O_RDONLY | fs.constants.O_DIRECTORY | fs.constants.O_NOFOLLOW,
    ));
  } catch (error) {
    if (!isOSError(error)) throw error;
    throw new ContractError(
      `artifact destination must be a real directory: ${destinationPath}: ${osErrorString(error, error.pyFilename)}`,
      { cause: error },
    );
  }
  let pinned;
  try {
    const destinationStat = fs.fstatSync(fd, { bigint: true });
    if (!destinationStat.isDirectory()) {
      throw new ContractError(`artifact destination is not a directory: ${destinationPath}`);
    }
    pinned = new PinnedDirectory(destinationPath, fd, identityOf(destinationStat), pinMode, { platform, procFdRoot });
    if (pinned.list().length) throw new ContractError(`artifact destination is not empty: ${destinationPath}`);
  } catch (error) {
    fs.closeSync(fd);
    throw error;
  }

  let staging;
  try {
    staging = path.resolve(fs.mkdtempSync(path.join(pyPathParent(destinationPath), '.artifact-extract-')));
  } catch (error) {
    fs.closeSync(fd);
    if (!isOSError(error)) throw error;
    throw new ContractError(`could not create artifact staging directory: ${osErrorString(error, error.path)}`, { cause: error });
  }
  const placed = [];
  const stagedIdentities = new Map();
  try {
    const names = stageArtifactArchive(archivePath, staging, { allowedMembers, perMemberCap, totalCap, zipFile });
    for (const name of names) stagedIdentities.set(name, identityOf(fs.lstatSync(path.join(staging, name), { bigint: true })));
    let current = null;
    try {
      for (const name of names) {
        current = name;
        replace(path.join(staging, name), pinned.entry(name));
        placed.push(name);
      }
    } catch (error) {
      if (!isOSError(error)) throw error;
      throw new ContractError(
        `could not place verified artifact members: ${osErrorString(error, path.join(staging, current), current)}`,
        { cause: error },
      );
    }
    let currentStat;
    try {
      currentStat = fs.lstatSync(destinationPath, { bigint: true });
    } catch (error) {
      if (!isOSError(error)) throw error;
      throw new ContractError(`artifact destination changed during extraction: ${osErrorString(error, destinationPath)}`, { cause: error });
    }
    if (!currentStat.isDirectory() || identityOf(currentStat) !== pinned.identity) {
      throw new ContractError('artifact destination changed during extraction');
    }
    const listed = new Set(pinned.list());
    if (listed.size !== names.length || names.some((name) => !listed.has(name))) {
      throw new ContractError('artifact destination changed during extraction');
    }
  } catch (error) {
    const cleanupErrors = pinned.unlinkPlaced(placed, stagedIdentities);
    if (cleanupErrors.length) {
      throw new ContractError(`could not remove partially placed artifact members: ${cleanupErrors.join('; ')}`, { cause: error });
    }
    throw error;
  } finally {
    fs.closeSync(fd);
    try {
      fs.rmSync(staging, { recursive: true, force: true });
    } catch {
      // shutil.rmtree(staging, ignore_errors=True)
    }
  }
}

// Exposed for the tests only.
export const internals = { PinnedDirectory, pyPathParent, pyFixed1 };

if (import.meta.main) {
  // Diagnostic entry: node archive.mjs <archive> <destination> [candidate|attestation]
  const [archive, destination, artifactType = 'candidate'] = process.argv.slice(2);
  try {
    extractFlatArtifactArchive(archive, destination, { artifactType });
  } catch (error) {
    process.stderr.write(`${error.name}: ${error.message}\n`);
    process.exitCode = 1;
  }
}
