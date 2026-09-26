// A zero-dependency ZIP writer for the archive tests. It emits the bytes
// CPython 3.12's zipfile.ZipFile(path, "w").writestr() writes on POSIX, so
// the tests can patch offsets exactly where the Python tests patch them:
//   * one local header per member (version 20, flags 0 or 0x800 for a
//     non-ASCII name, the final CRC and sizes, the member's extra field),
//     then its data;
//   * a central directory entry per member (create_version 20,
//     create_system 3, extra and comment, external_attr defaulting to
//     0o600 << 16 for a name and 0 for a ZipInfo, as writestr() does);
//   * a plain end-of-central-directory record with an empty comment.
// DEFLATE members use raw deflate at the default level, as zlib.compressobj
// (level -1, wbits -15) does; the compressed bytes may differ from CPython's
// zlib build, which none of the tests depend on.

import fs from 'node:fs';
import zlib from 'node:zlib';

export const ZIP_STORED = 0;
export const ZIP_DEFLATED = 8;
export const ZIP_BZIP2 = 12;

const VERSION = 20;
const CREATE_SYSTEM_UNIX = 3;
// zipfile.ZipInfo's default date_time, used for every member so the bytes
// are reproducible.
const DOS_TIME = 0;
const DOS_DATE = (0 << 9) | (1 << 5) | 1;

// A member: { name, data, compressType?, extra?, comment?, externalAttr? }.
// A [name, data] pair is writestr(name, data); an object with
// `zipInfo: true` is writestr(ZipInfo(name), data), whose external_attr
// defaults to 0.
function normalize(entry, compression) {
  if (Array.isArray(entry)) {
    const [name, data] = entry;
    return { name, data, compressType: compression, extra: Buffer.alloc(0), comment: Buffer.alloc(0), externalAttr: 0o600 << 16 };
  }
  return {
    name: entry.name,
    data: entry.data,
    compressType: entry.compressType ?? compression,
    extra: Buffer.from(entry.extra ?? []),
    comment: Buffer.from(entry.comment ?? []),
    externalAttr: entry.externalAttr ?? (entry.zipInfo ? 0 : 0o600 << 16),
  };
}

// The bytes of an archive holding `entries` in order, as ZipFile(..., "w",
// compression) with one writestr() per entry would write them.
export function buildZip(entries, { compression = ZIP_STORED } = {}) {
  const parts = [];
  const central = [];
  let offset = 0;
  for (const entry of entries.map((item) => normalize(item, compression))) {
    const data = Buffer.from(entry.data);
    const name = Buffer.from(entry.name, 'utf8');
    const flags = /^[\x00-\x7f]*$/.test(entry.name) ? 0 : 0x800;
    const payload = entry.compressType === ZIP_DEFLATED ? zlib.deflateRawSync(data) : data;
    const crc = zlib.crc32(data);
    const local = Buffer.alloc(30);
    local.write('PK\x03\x04', 0, 'latin1');
    local.writeUInt8(VERSION, 4);
    local.writeUInt8(0, 5);
    local.writeUInt16LE(flags, 6);
    local.writeUInt16LE(entry.compressType, 8);
    local.writeUInt16LE(DOS_TIME, 10);
    local.writeUInt16LE(DOS_DATE, 12);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(payload.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(name.length, 26);
    local.writeUInt16LE(entry.extra.length, 28);
    parts.push(local, name, entry.extra, payload);

    const record = Buffer.alloc(46);
    record.write('PK\x01\x02', 0, 'latin1');
    record.writeUInt8(VERSION, 4);
    record.writeUInt8(CREATE_SYSTEM_UNIX, 5);
    record.writeUInt8(VERSION, 6);
    record.writeUInt8(0, 7);
    record.writeUInt16LE(flags, 8);
    record.writeUInt16LE(entry.compressType, 10);
    record.writeUInt16LE(DOS_TIME, 12);
    record.writeUInt16LE(DOS_DATE, 14);
    record.writeUInt32LE(crc, 16);
    record.writeUInt32LE(payload.length, 20);
    record.writeUInt32LE(data.length, 24);
    record.writeUInt16LE(name.length, 28);
    record.writeUInt16LE(entry.extra.length, 30);
    record.writeUInt16LE(entry.comment.length, 32);
    record.writeUInt16LE(0, 34);
    record.writeUInt16LE(0, 36);
    record.writeUInt32LE(entry.externalAttr >>> 0, 38);
    record.writeUInt32LE(offset, 42);
    central.push(record, name, entry.extra, entry.comment);

    offset += local.length + name.length + entry.extra.length + payload.length;
  }
  const directory = Buffer.concat(central);
  const end = Buffer.alloc(22);
  end.write('PK\x05\x06', 0, 'latin1');
  end.writeUInt16LE(0, 4);
  end.writeUInt16LE(0, 6);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(directory.length, 12);
  end.writeUInt32LE(offset, 16);
  end.writeUInt16LE(0, 20);
  return Buffer.concat([...parts, directory, end]);
}

// Write buildZip(entries, options) to `filePath` and return the path.
export function writeZip(filePath, entries, options) {
  fs.writeFileSync(filePath, buildZip(entries, options));
  return filePath;
}
