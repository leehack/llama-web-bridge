// Tests of scripts/release/archive.mjs, the port of release_qualification.py's
// _extract_flat_artifact_archive. The first block is one test per archive
// test in scripts/release_qualification_test.py, with the same name and the
// same archive bytes (tests/release/zip_fixture.mjs writes what
// zipfile.writestr() writes). Where Python patches zipfile.ZipFile or
// os.replace with mock, these pass the extractor's `zipFile` and `replace`
// seams. The rest cover what only the Node port has: node:zlib inflation,
// the destination pinning modes, and the Python renderings it relies on.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, test } from 'node:test';
import process from 'node:process';
import zlib from 'node:zlib';

import {
  CANDIDATE_ALLOWED_MEMBERS,
  MAX_ATTESTATION_MEMBER_BYTES,
  MAX_CANDIDATE_MEMBER_BYTES,
  MAX_PREQUALIFICATION_ARCHIVE_BYTES,
  MAX_PREQUALIFICATION_MEMBER_BYTES,
  MAX_PREQUALIFICATION_TOTAL_BYTES,
  PREQUALIFICATION_ALLOWED_MEMBERS,
  PUBLICATION_FILES,
  artifactArchiveBounds,
  artifactArchiveSizeBound,
  extractFlatArtifactArchive,
  internals,
} from '../../scripts/release/archive.mjs';
import { ARTIFACTS as MANIFEST_ARTIFACTS } from '../../scripts/release/manifest.mjs';
import { PUBLICATION_FILES as STATE_PUBLICATION_FILES } from '../../scripts/release/publication_state.mjs';
import { ContractError } from '../../scripts/release/errors.mjs';
import { PyException, isPyException } from '../../scripts/release/json.mjs';
import { osErrorString, pyDecodeCp437, pyReprBytes } from '../../scripts/release/python_compat.mjs';
import { ZIP_BZIP2, ZIP_DEFLATED, buildZip, writeZip } from './zip_fixture.mjs';

const ARTIFACTS = PUBLICATION_FILES.slice(0, -2);
const S_IFLNK = 0o120000;
const HAS_PROC_FD = fs.existsSync('/proc/self/fd');

let tmp;
beforeEach(() => {
  tmp = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'archive-test-')));
});
afterEach(() => {
  fs.rmSync(tmp, { recursive: true, force: true });
});

// Every member a legitimate candidate archive must carry, and nothing else.
function candidateArchiveMembers() {
  const members = new Map(ARTIFACTS.map((name) => [name, 'data\n']));
  members.set('manifest.json', '{}\n');
  members.set('sha256sums.txt', 'sums\n');
  return members;
}

// A member-complete candidate archive, so rejections are on merit; `replace`
// renames members (their data becomes "bad").
function writeCandidateArchive(name, { replace = {} } = {}) {
  const members = candidateArchiveMembers();
  for (const original of Object.keys(replace)) members.delete(original);
  for (const renamed of Object.values(replace)) members.set(renamed, 'bad');
  return writeZip(path.join(tmp, name), [...members]);
}

function extract(archive, destination, artifactType, options = {}) {
  return extractFlatArtifactArchive(archive, destination, { artifactType, ...options });
}

// assertRaises(ContractError) followed by assertIn(expected, str(exc)).
function assertContractError(operation, expected) {
  let caught;
  assert.throws(operation, (error) => {
    caught = error;
    return error instanceof ContractError;
  });
  if (expected !== undefined) assert.ok(caught.message.includes(expected), `${JSON.stringify(expected)} not in ${JSON.stringify(caught.message)}`);
  return caught;
}

const listDir = (directory) => fs.readdirSync(directory);

// The offsets of every central directory record, found the way the Python
// tests find them (data.find(b"PK\x01\x02") from the start).
function centralRecords(data) {
  const records = [];
  for (let offset = data.indexOf('PK\x01\x02'); offset !== -1; offset = data.indexOf('PK\x01\x02', offset + 4)) {
    const nameLength = data.readUInt16LE(offset + 28);
    records.push({
      offset,
      name: data.subarray(offset + 46, offset + 46 + nameLength).toString('latin1'),
      nameLength,
      localOffset: data.readUInt32LE(offset + 42),
    });
  }
  return records;
}

function centralRecord(data, name) {
  const record = centralRecords(data).find((entry) => entry.name === name);
  assert.ok(record, `no central directory record for ${name}`);
  return record;
}

function patchArchive(archive, patch) {
  const data = fs.readFileSync(archive);
  patch(data);
  fs.writeFileSync(archive, data);
}

function insertBytes(data, offset, bytes) {
  return Buffer.concat([data.subarray(0, offset), Buffer.from(bytes), data.subarray(offset)]);
}

test('test_artifact_archive_rejects_path_escape_and_duplicate_members', () => {
  for (const [name, swapped, expected] of [
    ['escape.zip', '../escape', 'not a flat regular file'],
    ['nested.zip', 'nested/file', 'not a flat regular file'],
    ['unauthorized.zip', 'evil.sh', 'unauthorized member'],
  ]) {
    const archive = writeCandidateArchive(name, { replace: { 'sha256sums.txt': swapped } });
    assertContractError(() => extract(archive, path.join(tmp, `extract-${name}`), 'candidate'), expected);
  }

  const members = candidateArchiveMembers();
  members.delete('sha256sums.txt');
  const duplicate = writeZip(path.join(tmp, 'duplicate.zip'), [...members, ['manifest.json', 'second\n']]);
  assertContractError(() => extract(duplicate, path.join(tmp, 'extract-duplicate'), 'candidate'), 'repeats member');
});

test('test_artifact_archive_extracts_only_flat_regular_files', () => {
  const archive = writeCandidateArchive('valid-candidate.zip');
  const destination = path.join(tmp, 'valid-candidate-extract');
  extract(archive, destination, 'candidate');
  assert.equal(fs.readFileSync(path.join(destination, 'manifest.json'), 'latin1'), '{}\n');
  assert.deepEqual(new Set(listDir(destination)), new Set(candidateArchiveMembers().keys()));

  const realDestination = path.join(tmp, 'real-artifact-destination');
  fs.mkdirSync(realDestination);
  const linkedDestination = path.join(tmp, 'linked-artifact-destination');
  fs.symlinkSync(realDestination, linkedDestination, 'dir');
  assertContractError(() => extract(archive, linkedDestination, 'candidate'));
});

test('test_attestation_zip_bomb_rejected_on_compression_ratio', () => {
  // Small enough to clear the per-member byte cap, so the ratio bound is the
  // check that has to refuse it.
  const archive = writeZip(
    path.join(tmp, 'attestation-bomb.zip'),
    [['qualification-attestation.json', Buffer.alloc(32000)]],
    { compression: ZIP_DEFLATED },
  );
  assert.ok(fs.statSync(archive).size < MAX_ATTESTATION_MEMBER_BYTES);
  const destination = path.join(tmp, 'attestation-bomb-extract');
  assertContractError(() => extract(archive, destination, 'attestation'), 'compression ratio');
  assert.deepEqual(listDir(destination), []);
});

test('test_candidate_zip_bomb_rejected_on_compression_ratio', () => {
  const members = candidateArchiveMembers();
  members.set('sha256sums.txt', Buffer.alloc(1024 * 1024));
  const archive = writeZip(path.join(tmp, 'candidate-bomb.zip'), [...members], { compression: ZIP_DEFLATED });
  const destination = path.join(tmp, 'candidate-bomb-extract');
  assertContractError(() => extract(archive, destination, 'candidate'), 'compression ratio');
  assert.deepEqual(listDir(destination), []);
});

test('test_oversized_candidate_member_rejected_before_extraction', () => {
  const archive = writeCandidateArchive('oversize-candidate.zip');
  const oversize = MAX_CANDIDATE_MEMBER_BYTES + 1;
  patchArchive(archive, (data) => {
    const { offset, localOffset } = centralRecord(data, 'sha256sums.txt');
    data.writeUInt32LE(oversize, offset + 20);
    data.writeUInt32LE(oversize, offset + 24);
    data.writeUInt32LE(oversize, localOffset + 18);
    data.writeUInt32LE(oversize, localOffset + 22);
  });
  const destination = path.join(tmp, 'oversize-candidate-extract');
  assertContractError(() => extract(archive, destination, 'candidate'), 'exceeds bound');
  assert.deepEqual(listDir(destination), []);
});

test('test_oversized_attestation_member_rejected_before_extraction', () => {
  const archive = writeZip(path.join(tmp, 'oversize-attestation.zip'), [
    ['qualification-attestation.json', Buffer.alloc(MAX_ATTESTATION_MEMBER_BYTES + 1, 'x')],
  ]);
  const destination = path.join(tmp, 'oversize-attestation-extract');
  assertContractError(() => extract(archive, destination, 'attestation'), 'exceeds bound');
  assert.deepEqual(listDir(destination), []);
});

test('test_total_uncompressed_size_is_bounded_before_extraction', () => {
  const archive = writeCandidateArchive('oversize-total-candidate.zip');
  const claimedSize = 32 * 1024 * 1024;
  let rewritten = 0;
  patchArchive(archive, (data) => {
    for (const { offset, localOffset } of centralRecords(data)) {
      data.writeUInt32LE(claimedSize, offset + 20);
      data.writeUInt32LE(claimedSize, offset + 24);
      data.writeUInt32LE(claimedSize, localOffset + 18);
      data.writeUInt32LE(claimedSize, localOffset + 22);
      rewritten += 1;
    }
  });
  assert.equal(rewritten, candidateArchiveMembers().size);
  const destination = path.join(tmp, 'oversize-total-extract');
  assertContractError(() => extract(archive, destination, 'candidate'), 'total uncompressed size');
  assert.deepEqual(listDir(destination), []);
});

test('test_symlink_archive_member_rejected', () => {
  const archive = writeZip(path.join(tmp, 'symlink-attestation.zip'), [
    { name: 'qualification-attestation.json', data: 'target.json', zipInfo: true, externalAttr: ((S_IFLNK | 0o777) << 16) >>> 0 },
  ]);
  const destination = path.join(tmp, 'symlink-attestation-extract');
  assertContractError(() => extract(archive, destination, 'attestation'), 'not a flat regular file');
  assert.deepEqual(listDir(destination), []);
});

test('test_unauthorized_artifact_member_rejected', () => {
  const archive = writeZip(path.join(tmp, 'unauthorized-attestation.zip'), [['evil.sh', 'echo evil\n']]);
  assertContractError(() => extract(archive, path.join(tmp, 'unauthorized-extract'), 'attestation'), 'unauthorized member');
});

test('test_short_candidate_member_inventory_rejected', () => {
  const archive = writeZip(path.join(tmp, 'incomplete-candidate.zip'), ARTIFACTS.map((name) => [name, 'data\n']));
  const error = assertContractError(
    () => extract(archive, path.join(tmp, 'incomplete-candidate-extract'), 'candidate'),
    'member count must be exactly',
  );
  assert.ok(error.message.includes(String(candidateArchiveMembers().size)));
});

test('test_eocd_count_is_bounded_before_zipfile_parses_members', () => {
  const archive = writeZip(path.join(tmp, 'too-many-attestation-members.zip'), [
    ['qualification-attestation.json', 'evidence\n'],
    ['evil.txt', 'extra\n'],
  ]);
  const destination = path.join(tmp, 'too-many-attestation-members-extract');
  const zipFile = () => {
    throw new assert.AssertionError({ message: 'ZipFile must not parse an over-count archive' });
  };
  assertContractError(() => extract(archive, destination, 'attestation', { zipFile }), 'end-of-central-directory member count');
  assert.deepEqual(listDir(destination), []);
});

test('test_archive_preamble_cannot_hide_outside_the_member_inventory', () => {
  const archive = writeZip(path.join(tmp, 'prefixed-attestation.zip'), [['qualification-attestation.json', 'evidence\n']]);
  const original = fs.readFileSync(archive);
  const oldCdOffset = original.indexOf('PK\x01\x02');
  const oldEocdOffset = original.indexOf('PK\x05\x06');
  assert.notEqual(oldCdOffset, -1);
  assert.notEqual(oldEocdOffset, -1);
  const prefix = Buffer.from('JUNK');
  const data = Buffer.concat([prefix, original]);
  const cdOffset = oldCdOffset + prefix.length;
  const eocdOffset = oldEocdOffset + prefix.length;
  data.writeUInt32LE(data.readUInt32LE(cdOffset + 42) + prefix.length, cdOffset + 42);
  data.writeUInt32LE(cdOffset, eocdOffset + 16);
  fs.writeFileSync(archive, data);
  const destination = path.join(tmp, 'prefixed-attestation-extract');
  assertContractError(() => extract(archive, destination, 'attestation'), 'start at byte zero');
  assert.deepEqual(listDir(destination), []);
});

test('test_encrypted_zip_member_rejected', () => {
  for (const flag of [0x1, 0x40, 0x2000]) {
    const archive = writeZip(path.join(tmp, `encrypted-${flag.toString(16)}.zip`), [['qualification-attestation.json', 'secret']]);
    patchArchive(archive, (data) => {
      data.writeUInt16LE(data.readUInt16LE(6) | flag, 6);
      const cdOffset = data.indexOf('PK\x01\x02');
      assert.notEqual(cdOffset, -1);
      data.writeUInt16LE(data.readUInt16LE(cdOffset + 8) | flag, cdOffset + 8);
    });
    assertContractError(() => extract(archive, path.join(tmp, `encrypted-${flag.toString(16)}-extract`), 'attestation'), 'encrypted');
  }
});

test('test_data_descriptor_zip_member_rejected', () => {
  const archive = writeZip(path.join(tmp, 'data-descriptor.zip'), [['qualification-attestation.json', 'evidence\n']]);
  patchArchive(archive, (data) => {
    data.writeUInt16LE(data.readUInt16LE(6) | 0x8, 6);
    data.fill(0, 14, 26);
    const cdOffset = data.indexOf('PK\x01\x02');
    assert.notEqual(cdOffset, -1);
    data.writeUInt16LE(data.readUInt16LE(cdOffset + 8) | 0x8, cdOffset + 8);
  });
  const destination = path.join(tmp, 'data-descriptor-extract');
  assertContractError(() => extract(archive, destination, 'attestation'), 'data descriptor');
  assert.deepEqual(listDir(destination), []);
});

test('test_signed_github_style_data_descriptor_is_exactly_validated', () => {
  const archive = writeZip(path.join(tmp, 'signed-data-descriptor.zip'), [['qualification-attestation.json', 'evidence\n']]);
  let data = fs.readFileSync(archive);
  const cdOffset = data.indexOf('PK\x01\x02');
  assert.notEqual(cdOffset, -1);
  const crc = data.readUInt32LE(cdOffset + 16);
  const compressedSize = data.readUInt32LE(cdOffset + 20);
  const fileSize = data.readUInt32LE(cdOffset + 24);
  data.writeUInt16LE(data.readUInt16LE(6) | 0x8, 6);
  data.fill(0, 14, 26);
  data.writeUInt16LE(data.readUInt16LE(cdOffset + 8) | 0x8, cdOffset + 8);
  const descriptor = Buffer.alloc(16);
  descriptor.write('PK\x07\x08', 0, 'latin1');
  descriptor.writeUInt32LE(crc, 4);
  descriptor.writeUInt32LE(compressedSize, 8);
  descriptor.writeUInt32LE(fileSize, 12);
  data = insertBytes(data, cdOffset, descriptor);
  const eocdOffset = data.indexOf('PK\x05\x06', cdOffset + descriptor.length);
  assert.notEqual(eocdOffset, -1);
  data.writeUInt32LE(cdOffset + descriptor.length, eocdOffset + 16);
  fs.writeFileSync(archive, data);

  const destination = path.join(tmp, 'signed-data-descriptor-extract');
  extract(archive, destination, 'attestation');
  assert.equal(fs.readFileSync(path.join(destination, 'qualification-attestation.json'), 'latin1'), 'evidence\n');

  patchArchive(archive, (mismatched) => {
    const descriptorOffset = mismatched.indexOf('PK\x07\x08');
    assert.notEqual(descriptorOffset, -1);
    mismatched.writeUInt32LE((crc ^ 0xffffffff) >>> 0, descriptorOffset + 4);
  });
  const mismatchedDestination = path.join(tmp, 'mismatched-data-descriptor-extract');
  assertContractError(() => extract(archive, mismatchedDestination, 'attestation'), 'data descriptor');
  assert.deepEqual(listDir(mismatchedDestination), []);
});

test('test_local_and_central_flag_mismatch_rejected', () => {
  const archive = writeZip(path.join(tmp, 'flag-mismatch.zip'), [['qualification-attestation.json', 'evidence\n']]);
  patchArchive(archive, (data) => data.writeUInt16LE(data.readUInt16LE(6) ^ 0x800, 6));
  const destination = path.join(tmp, 'flag-mismatch-extract');
  assertContractError(() => extract(archive, destination, 'attestation'), 'flags');
  assert.deepEqual(listDir(destination), []);
});

test('test_nul_truncated_member_name_cannot_masquerade_as_allowlisted', () => {
  const archive = writeCandidateArchive('nul-name-candidate.zip', { replace: { 'manifest.json': 'manifest.jsonX' } });
  patchArchive(archive, (data) => {
    const { offset, nameLength, localOffset } = centralRecord(data, 'manifest.jsonX');
    data[localOffset + 30 + nameLength - 1] = 0;
    data[offset + 46 + nameLength - 1] = 0;
  });
  const destination = path.join(tmp, 'nul-name-extract');
  assertContractError(() => extract(archive, destination, 'candidate'), 'ambiguous');
  assert.deepEqual(listDir(destination), []);
});

test('test_member_extra_fields_and_comments_cannot_hide_metadata', () => {
  for (const kind of ['extra', 'comment']) {
    const member = { name: 'qualification-attestation.json', data: 'evidence\n', zipInfo: true };
    if (kind === 'extra') member.extra = Buffer.from([0xfe, 0xca, 0, 0]);
    else member.comment = Buffer.from('hidden');
    const archive = writeZip(path.join(tmp, `hidden-${kind}-attestation.zip`), [member]);
    const destination = path.join(tmp, `hidden-${kind}-extract`);
    assertContractError(() => extract(archive, destination, 'attestation'), 'metadata');
    assert.deepEqual(listDir(destination), []);
  }
});

test('test_local_only_extra_field_cannot_hide_metadata', () => {
  const archive = writeZip(path.join(tmp, 'local-extra-attestation.zip'), [['qualification-attestation.json', 'evidence\n']]);
  let data = fs.readFileSync(archive);
  const payloadOffset = 30 + data.readUInt16LE(26);
  const localExtra = Buffer.from([0xfe, 0xca, 0, 0]);
  data = insertBytes(data, payloadOffset, localExtra);
  data.writeUInt16LE(localExtra.length, 28);
  const cdOffset = data.indexOf('PK\x01\x02');
  const eocdOffset = data.indexOf('PK\x05\x06');
  assert.notEqual(cdOffset, -1);
  assert.notEqual(eocdOffset, -1);
  data.writeUInt32LE(cdOffset, eocdOffset + 16);
  fs.writeFileSync(archive, data);
  const destination = path.join(tmp, 'local-extra-extract');
  assertContractError(() => extract(archive, destination, 'attestation'), 'metadata');
  assert.deepEqual(listDir(destination), []);
});

test('test_unsupported_compression_method_rejected', () => {
  const archive = writeCandidateArchive('bzip2-candidate.zip');
  patchArchive(archive, (data) => {
    const { offset, localOffset } = centralRecord(data, 'manifest.json');
    data.writeUInt16LE(ZIP_BZIP2, offset + 10);
    data.writeUInt16LE(ZIP_BZIP2, localOffset + 8);
  });
  const destination = path.join(tmp, 'bzip2-candidate-extract');
  assertContractError(() => extract(archive, destination, 'candidate'), 'unsupported compression method');
  assert.deepEqual(listDir(destination), []);
});

test('test_local_header_disagreeing_with_central_directory_rejected', () => {
  // Only the local header is rewritten, so the central directory still looks
  // benign; the archive is ambiguous and must be refused.
  const archive = writeCandidateArchive('local-mismatch-candidate.zip');
  patchArchive(archive, (data) => {
    const cdOffset = data.indexOf('PK\x01\x02');
    data.writeUInt32LE(4096, data.readUInt32LE(cdOffset + 42) + 22);
  });
  const destination = path.join(tmp, 'local-mismatch-extract');
  assertContractError(() => extract(archive, destination, 'candidate'), 'disagrees with the central directory');
  assert.deepEqual(listDir(destination), []);
});

test('test_local_header_name_disagreeing_with_central_directory_rejected', () => {
  const archive = writeCandidateArchive('local-name-candidate.zip');
  patchArchive(archive, (data) => {
    const [{ offset, nameLength, localOffset }] = centralRecords(data);
    const start = localOffset + 30;
    assert.deepEqual(data.subarray(start, start + nameLength), data.subarray(offset + 46, offset + 46 + nameLength));
    data[start] = 'Z'.charCodeAt(0);
  });
  const destination = path.join(tmp, 'local-name-extract');
  assertContractError(() => extract(archive, destination, 'candidate'), 'disagrees with the central directory');
  assert.deepEqual(listDir(destination), []);
});

test('test_overlapping_members_rejected', () => {
  const archive = writeCandidateArchive('overlapping-candidate.zip');
  patchArchive(archive, (data) => {
    const records = centralRecords(data);
    assert.ok(records.length > 1);
    // Point a second member's local header back inside the first member's
    // payload so the two entries claim overlapping bytes.
    data.writeUInt32LE(records[0].localOffset, records[1].offset + 42);
  });
  const destination = path.join(tmp, 'overlapping-extract');
  assertContractError(() => extract(archive, destination, 'candidate'), 'overlapping members');
  assert.deepEqual(listDir(destination), []);
});

test('test_member_payload_cannot_overlap_the_central_directory', () => {
  const archive = writeCandidateArchive('central-overlap-candidate.zip');
  patchArchive(archive, (data) => {
    const { offset, localOffset } = centralRecord(data, 'sha256sums.txt');
    const compressedSize = data.readUInt32LE(offset + 20);
    const fileSize = data.readUInt32LE(offset + 24);
    data.writeUInt32LE(compressedSize + 1, offset + 20);
    data.writeUInt32LE(fileSize + 1, offset + 24);
    data.writeUInt32LE(compressedSize + 1, localOffset + 18);
    data.writeUInt32LE(fileSize + 1, localOffset + 22);
  });
  const destination = path.join(tmp, 'central-overlap-extract');
  assertContractError(() => extract(archive, destination, 'candidate'), 'central directory');
  assert.deepEqual(listDir(destination), []);
});

test('test_unclaimed_gap_before_central_directory_rejected', () => {
  const archive = writeZip(path.join(tmp, 'central-gap-attestation.zip'), [['qualification-attestation.json', 'evidence\n']]);
  let data = fs.readFileSync(archive);
  const cdOffset = data.indexOf('PK\x01\x02');
  let eocdOffset = data.indexOf('PK\x05\x06');
  assert.notEqual(cdOffset, -1);
  assert.notEqual(eocdOffset, -1);
  const gap = Buffer.from('JUNK');
  data = insertBytes(data, cdOffset, gap);
  eocdOffset += gap.length;
  data.writeUInt32LE(cdOffset + gap.length, eocdOffset + 16);
  fs.writeFileSync(archive, data);
  const destination = path.join(tmp, 'central-gap-extract');
  assertContractError(() => extract(archive, destination, 'attestation'), 'unclaimed gap');
  assert.deepEqual(listDir(destination), []);
});

test('test_failed_extraction_leaves_no_partial_trusted_output', () => {
  // The corrupted member sorts last, so earlier members extract cleanly
  // before the failure and would survive without staged extraction.
  const archive = writeCandidateArchive('corrupt-candidate.zip');
  patchArchive(archive, (data) => {
    const { nameLength, localOffset } = centralRecord(data, 'sha256sums.txt');
    data[localOffset + 30 + nameLength] ^= 0xff;
  });
  const destination = path.join(tmp, 'corrupt-candidate-extract');
  assertContractError(() => extract(archive, destination, 'candidate'));
  assert.ok(fs.statSync(destination).isDirectory());
  assert.deepEqual(listDir(destination), []);
  assert.deepEqual(listDir(path.dirname(destination)).filter((name) => name.startsWith('.artifact-extract-')), []);
});

test('test_interrupted_placement_leaves_no_partial_trusted_output', () => {
  // Every member has already passed, so the failure lands while verified
  // files are being moved in; the destination must still come back empty.
  const archive = writeCandidateArchive('interrupted-candidate.zip');
  const destination = path.join(tmp, 'interrupted-extract');
  let calls = 0;
  const replace = (source, target) => {
    calls += 1;
    if (calls > 3) throw new PyException('OSError', 'simulated placement failure');
    fs.renameSync(source, target);
  };
  assertContractError(() => extract(archive, destination, 'candidate', { replace }), 'could not place verified artifact members');
  assert.ok(calls > 3);
  assert.ok(fs.statSync(destination).isDirectory());
  assert.deepEqual(listDir(destination), []);
});

test('test_destination_swap_cannot_redirect_or_preserve_verified_output', () => {
  // Python pins the destination with a dir_fd. Linux runs this in both of
  // the Node port's pinning modes; elsewhere only the path mode exists.
  for (const pinMode of HAS_PROC_FD ? ['proc', 'path'] : ['path']) {
    const base = path.join(tmp, pinMode);
    fs.mkdirSync(base);
    const archive = writeCandidateArchive(`${pinMode}/destination-swap-candidate.zip`);
    const destination = path.join(base, 'destination-swap-extract');
    const displaced = path.join(base, 'displaced-destination');
    const attackerTarget = path.join(base, 'attacker-target');
    fs.mkdirSync(attackerTarget);
    let swapped = false;
    const replace = (source, target) => {
      if (!swapped) {
        fs.renameSync(destination, displaced);
        fs.symlinkSync(attackerTarget, destination, 'dir');
        swapped = true;
      }
      fs.renameSync(source, target);
    };
    assertContractError(() => extract(archive, destination, 'candidate', { replace, pinMode }), 'changed during extraction');
    assert.deepEqual(listDir(attackerTarget), [], pinMode);
    assert.ok(fs.statSync(displaced).isDirectory(), pinMode);
    assert.deepEqual(listDir(displaced), [], pinMode);
  }
});

test('test_unknown_artifact_type_rejected', () => {
  const archive = writeCandidateArchive('unknown-type-candidate.zip');
  assertContractError(() => extract(archive, path.join(tmp, 'unknown-extract'), 'invalid-type'), 'unknown artifact type');
});

// --- Node port only ----------------------------------------------------------

test('the candidate allowlist is publication_state.PUBLICATION_FILES', () => {
  // manifest.mjs's ARTIFACTS plus the two generated files, in that order.
  assert.deepEqual(PUBLICATION_FILES, [...MANIFEST_ARTIFACTS, 'manifest.json', 'sha256sums.txt']);
  assert.deepEqual(PUBLICATION_FILES, STATE_PUBLICATION_FILES);
  assert.deepEqual([...CANDIDATE_ALLOWED_MEMBERS], PUBLICATION_FILES);
});

test('the prequalification record has its own bounds', () => {
  assert.deepEqual(artifactArchiveBounds('prequalification'), [
    PREQUALIFICATION_ALLOWED_MEMBERS, MAX_PREQUALIFICATION_MEMBER_BYTES, MAX_PREQUALIFICATION_TOTAL_BYTES,
  ]);
  assert.deepEqual([...PREQUALIFICATION_ALLOWED_MEMBERS], ['candidate-prequalification.json']);
  assert.equal(MAX_PREQUALIFICATION_MEMBER_BYTES, 128 * 1024);
  assert.equal(MAX_PREQUALIFICATION_TOTAL_BYTES, 128 * 1024);
  assert.equal(MAX_PREQUALIFICATION_ARCHIVE_BYTES, 1024 * 1024);
  // Only the prequalification archive is bounded as a file.
  assert.equal(artifactArchiveSizeBound('prequalification'), MAX_PREQUALIFICATION_ARCHIVE_BYTES);
  assert.equal(artifactArchiveSizeBound('candidate'), null);
  assert.equal(artifactArchiveSizeBound('attestation'), null);
  assert.throws(() => artifactArchiveSizeBound('x'), { message: "unknown artifact type: 'x'" });
  const archive = writeZip(path.join(tmp, 'prequalification.zip'), [['candidate-prequalification.json', '{}\n']]);
  extract(archive, path.join(tmp, 'out'), 'prequalification');
  assert.deepEqual(listDir(path.join(tmp, 'out')), ['candidate-prequalification.json']);
  // An archive one byte over its bound is refused before it is read.
  const padded = writeZip(path.join(tmp, 'padded.zip'), [['candidate-prequalification.json', Buffer.alloc(MAX_PREQUALIFICATION_ARCHIVE_BYTES)]]);
  const size = fs.statSync(padded).size;
  assert.ok(size > MAX_PREQUALIFICATION_ARCHIVE_BYTES);
  let reads = 0;
  assertContractError(
    () => extract(padded, path.join(tmp, 'padded'), 'prequalification', { zipFile: () => { reads += 1; } }),
    `artifact archive size ${size} exceeds bound ${MAX_PREQUALIFICATION_ARCHIVE_BYTES}`,
  );
  assert.equal(reads, 0);
  assert.deepEqual(listDir(path.join(tmp, 'padded')), []);
});

test('unknown artifact types render as Python repr()', () => {
  for (const [value, rendered] of [['x', "'x'"], [null, 'None'], [undefined, 'None'], [1, '1']]) {
    assert.throws(() => artifactArchiveBounds(value), { message: `unknown artifact type: ${rendered}` });
  }
});

// A single-member attestation archive whose payload is `payload`, declared to
// hold `content` (its CRC and size).
function deflatedAttestation(name, payload, content) {
  const data = buildZip([['qualification-attestation.json', content]]);
  const nameLength = data.readUInt16LE(26);
  const cdOffset = data.indexOf('PK\x01\x02');
  const head = Buffer.from(data.subarray(0, 30 + nameLength));
  head.writeUInt16LE(ZIP_DEFLATED, 8);
  head.writeUInt32LE(payload.length, 18);
  const central = Buffer.from(data.subarray(cdOffset));
  central.writeUInt16LE(ZIP_DEFLATED, 10);
  central.writeUInt32LE(payload.length, 20);
  central.writeUInt32LE(head.length + payload.length, central.length - 22 + 16);
  const archive = path.join(tmp, name);
  fs.writeFileSync(archive, Buffer.concat([head, payload, central]));
  return archive;
}

test('DEFLATE members extract through node:zlib', () => {
  const content = Buffer.from('{"evidence": true}\n'.repeat(40));
  const archive = deflatedAttestation('deflated.zip', zlib.deflateRawSync(content), content);
  const destination = path.join(tmp, 'deflated-extract');
  extract(archive, destination, 'attestation');
  assert.deepEqual(fs.readFileSync(path.join(destination, 'qualification-attestation.json')), content);
});

test('a zlib data error escapes as zlib.error, as it does in Python', () => {
  const content = Buffer.from('evidence\n'.repeat(20));
  // A block with the reserved block type 3.
  const archive = deflatedAttestation('bad-block.zip', Buffer.from([0x07, 0x00]), content);
  const destination = path.join(tmp, 'bad-block-extract');
  assert.throws(() => extract(archive, destination, 'attestation'), (error) => isPyException(error, 'zlib.error')
    && error.message === 'Error -3 while decompressing data: invalid block type');
  assert.deepEqual(listDir(destination), []);
});

test('a truncated DEFLATE stream fails the CRC check', () => {
  const content = Buffer.from('evidence\n'.repeat(20));
  const full = zlib.deflateRawSync(content, { level: 0 });
  const archive = deflatedAttestation('truncated.zip', full.subarray(0, full.length - 10), content);
  assertContractError(
    () => extract(archive, path.join(tmp, 'truncated-extract'), 'attestation'),
    "could not read artifact archive: Bad CRC-32 for file 'qualification-attestation.json'",
  );
});

test('a stream that decodes past its declared size is truncated like Python, then fails the CRC', () => {
  const declared = Buffer.from('evidence\n'.repeat(20));
  const longer = Buffer.concat([declared, Buffer.from('surplus')]);
  // The declared CRC is the full stream's, so the truncated bytes cannot match.
  const archive = deflatedAttestation('surplus.zip', zlib.deflateRawSync(longer), declared);
  patchArchive(archive, (data) => {
    const cdOffset = data.indexOf('PK\x01\x02');
    data.writeUInt32LE(zlib.crc32(longer), 14);
    data.writeUInt32LE(zlib.crc32(longer), cdOffset + 16);
  });
  assertContractError(
    () => extract(archive, path.join(tmp, 'surplus-extract'), 'attestation'),
    "could not read artifact archive: Bad CRC-32 for file 'qualification-attestation.json'",
  );
});

// The one place the port deliberately departs from Python: Python stops
// decoding once it has file_size bytes, and the CRC covers only those, so it
// ACCEPTS this archive and never looks at the 200 KiB behind them. node:zlib
// cannot stop there, so the port bounds decoding at file_size + 64 KiB (more
// than Python ever decodes) and refuses a stream that goes further.
test('a stream that decodes more than 64 KiB past its declared size is refused', () => {
  const declared = Buffer.from('evidence\n'.repeat(20));
  const bomb = Buffer.concat([declared, Buffer.alloc(200 * 1024, 'x')]);
  const payload = zlib.deflateRawSync(bomb);
  // The declared size and CRC are the first 180 bytes', and the ratio stays
  // under the bound.
  const archive = deflatedAttestation('overlong.zip', payload, declared);
  assertContractError(
    () => extract(archive, path.join(tmp, 'overlong-extract'), 'attestation'),
    `could not read artifact archive: DEFLATE stream for file 'qualification-attestation.json' decodes more than 65536 bytes past its declared size ${declared.length}`,
  );
});

test('pinMode proc is refused where /proc/self/fd cannot name the pinned directory', { skip: HAS_PROC_FD }, () => {
  const archive = writeCandidateArchive('proc.zip');
  assert.throws(() => extract(archive, path.join(tmp, 'proc-extract'), 'candidate', { pinMode: 'proc' }));
});

test('the extracted destination holds exactly the members in path mode', () => {
  const archive = writeCandidateArchive('path-mode.zip');
  const destination = path.join(tmp, 'path-mode-extract');
  assert.equal(extract(archive, destination, 'candidate', { pinMode: 'path' }), undefined);
  assert.deepEqual(new Set(listDir(destination)), new Set(PUBLICATION_FILES));
});

test('pyFixed1 rounds one-decimal ties to even, as f"{x:.1f}" does', () => {
  const { pyFixed1 } = internals;
  assert.equal(pyFixed1(0.25), '0.2');
  assert.equal(pyFixed1(0.75), '0.8');
  assert.equal(pyFixed1(2.25), '2.2');
  assert.equal(pyFixed1(100.05), '100.0');
  assert.equal(pyFixed1(1234.5678), '1234.6');
});

test('python_compat renders OSError, bytes and cp437 as CPython does', () => {
  const missing = Object.assign(new Error('x'), { code: 'ENOENT', errno: -2 });
  assert.equal(osErrorString(missing), '[Errno 2] No such file or directory');
  assert.equal(osErrorString(missing, "it's"), `[Errno 2] No such file or directory: "it's"`);
  assert.equal(osErrorString(missing, 'a', 'b'), "[Errno 2] No such file or directory: 'a' -> 'b'");
  assert.equal(osErrorString(new PyException('OSError', 'simulated')), 'simulated');
  assert.equal(pyReprBytes(Buffer.from("a'\t\x00\xff\\", 'latin1')), `b"a'\\t\\x00\\xff\\\\"`);
  assert.equal(pyReprBytes(Buffer.from(`'"`)), `b'\\'"'`);
  assert.equal(pyDecodeCp437(Buffer.from([0x41, 0x80, 0xe1, 0xff])), 'AÇß\u{a0}');
});

// --- Byte-level regressions against CPython 3.12.3 ------------------------------
//
// Single-member attestation archives built byte for byte like the review's
// craft.py recipes. Each expected outcome is what release_qualification.py's
// _extract_flat_artifact_archive does with the same bytes on the ubuntu-24.04
// runners' Python 3.12.3.

const ATTESTATION = Buffer.from('qualification-attestation.json');
const EVIDENCE = Buffer.from('evidence\n');

function packFields(fields) {
  const parts = fields.map(([kind, value]) => {
    if (Buffer.isBuffer(value)) return value;
    const size = { B: 1, H: 2, L: 4, Q: 8 }[kind];
    const buffer = Buffer.alloc(size);
    if (kind === 'B') buffer.writeUInt8(value);
    else if (kind === 'H') buffer.writeUInt16LE(value);
    else if (kind === 'L') buffer.writeUInt32LE(value >>> 0);
    else buffer.writeBigUInt64LE(BigInt(value));
    return buffer;
  });
  return Buffer.concat(parts);
}

// craft.py one(): a STORED member with a local header, one central record and
// a plain end record; `name`/`lname` are the central and local name bytes.
function craftAttestation({
  name = ATTESTATION, lname = name, flags = 0, lflags = flags, extractVersion = 20,
  externalAttr = 0o100644 << 16, cdExtra = Buffer.alloc(0),
} = {}) {
  const crc = zlib.crc32(EVIDENCE);
  const local = packFields([
    ['s', Buffer.from('PK\x03\x04', 'latin1')], ['H', 20], ['H', lflags], ['H', 0], ['H', 0], ['H', 33],
    ['L', crc], ['L', EVIDENCE.length], ['L', EVIDENCE.length], ['H', lname.length], ['H', 0],
  ]);
  const body = Buffer.concat([local, lname, EVIDENCE]);
  const central = Buffer.concat([packFields([
    ['s', Buffer.from('PK\x01\x02', 'latin1')], ['B', 20], ['B', 3], ['B', extractVersion], ['B', 0],
    ['H', flags], ['H', 0], ['H', 0], ['H', 33], ['L', crc], ['L', EVIDENCE.length], ['L', EVIDENCE.length],
    ['H', name.length], ['H', cdExtra.length], ['H', 0], ['H', 0], ['H', 0], ['L', externalAttr], ['L', 0],
  ]), name, cdExtra]);
  return { body, central };
}

function endRecord(directoryLength, directoryOffset) {
  return packFields([
    ['s', Buffer.from('PK\x05\x06', 'latin1')], ['H', 0], ['H', 0], ['H', 1], ['H', 1],
    ['L', directoryLength], ['L', directoryOffset], ['H', 0],
  ]);
}

function writeCrafted(name, options) {
  const { body, central } = craftAttestation(options);
  const file = path.join(tmp, name);
  fs.writeFileSync(file, Buffer.concat([body, central, endRecord(central.length, body.length)]));
  return file;
}

// craft.py zip64_case(): the central directory is followed by a ZIP64 end
// record and locator that the plain end record counts as directory bytes.
function writeZip64(name, { recordSize = 44, extra = Buffer.alloc(0), disks = 1, disk = 0, relativeOffset = null, directorySize = null, directoryOffset = null } = {}) {
  const { body, central } = craftAttestation();
  const record = Buffer.concat([packFields([
    ['s', Buffer.from('PK\x06\x06', 'latin1')], ['Q', recordSize], ['H', 45], ['H', 45], ['L', 0], ['L', 0],
    ['Q', 1], ['Q', 1], ['Q', directorySize ?? central.length], ['Q', directoryOffset ?? body.length],
  ]), extra]);
  const locator = packFields([
    ['s', Buffer.from('PK\x06\x07', 'latin1')], ['L', disk], ['Q', relativeOffset ?? body.length + central.length], ['L', disks],
  ]);
  const directory = Buffer.concat([central, record, locator]);
  const file = path.join(tmp, name);
  fs.writeFileSync(file, Buffer.concat([body, directory, endRecord(directory.length, body.length)]));
  return file;
}

// craft.py's locator-only archives: a ZIP64 locator and no ZIP64 record.
function writeLocatorOnly(name, disks) {
  const { body, central } = craftAttestation();
  const locator = packFields([['s', Buffer.from('PK\x06\x07', 'latin1')], ['L', 0], ['Q', 0], ['L', disks]]);
  const directory = Buffer.concat([central, locator]);
  const file = path.join(tmp, name);
  fs.writeFileSync(file, Buffer.concat([body, directory, endRecord(directory.length, body.length)]));
  return file;
}

function assertExtracted(archive, label) {
  const destination = path.join(tmp, `${label}-extract`);
  assert.equal(extract(archive, destination, 'attestation'), undefined, label);
  assert.deepEqual(listDir(destination), ['qualification-attestation.json'], label);
  assert.deepEqual(fs.readFileSync(path.join(destination, 'qualification-attestation.json')), EVIDENCE, label);
}

function assertRefused(archive, label, check) {
  const destination = path.join(tmp, `${label}-extract`);
  assert.throws(() => extract(archive, destination, 'attestation'), (error) => {
    check(error);
    return true;
  }, label);
  assert.deepEqual(listDir(destination), [], label);
}

const contractError = (message) => (error) => {
  assert.ok(error instanceof ContractError, String(error));
  assert.equal(error.message, message);
};
const pyError = (type, message) => (error) => {
  assert.ok(isPyException(error, type), String(error));
  assert.equal(error.pyType, type);
  assert.equal(error.message, message);
};

test('ZIP64 end records are read, and their inconsistencies refused, as CPython 3.12.3 does', () => {
  assertExtracted(writeZip64('z64-plain.zip'), 'z64-plain');
  assertExtracted(writeZip64('z64-extra.zip', { recordSize: 52, extra: Buffer.from('12345678') }), 'z64-extra');
  const corruptRecord = 'could not read artifact archive: Corrupt zip64 end of central directory record';
  for (const [label, options] of [
    ['z64-badsize', { recordSize: 40 }],
    ['z64-badrel', { relativeOffset: 3 }],
    ['z64-cd-size-0', { directorySize: 0 }],
    ['z64-cd-off-0', { directorySize: 0, directoryOffset: 0 }],
  ]) {
    assertRefused(writeZip64(`${label}.zip`, options), label, contractError(corruptRecord));
  }
  for (const [label, options] of [['z64-disks2', { disks: 2 }], ['z64-disk1', { disk: 1 }]]) {
    assertRefused(writeZip64(`${label}.zip`, options), label,
      contractError('could not read artifact archive: zipfiles that span multiple disks are not supported'));
  }
  assertRefused(writeZip64('z64-rel-big.zip', { relativeOffset: 2 ** 40 }), 'z64-rel-big',
    contractError('could not read artifact archive: Corrupt zip64 end of central directory locator'));
  for (const disks of [1, 0]) {
    assertRefused(writeLocatorOnly(`z64-locator-${disks}.zip`, disks), `z64-locator-${disks}`,
      contractError('could not read artifact archive: Zip64 end of central directory record not found'));
  }
});

test('central extra fields are decoded as CPython 3.12.3 decodes them before the metadata check', () => {
  const unicodePath = (nameCrc, text) => Buffer.concat([
    packFields([['H', 0x7075], ['H', 5 + text.length], ['B', 1], ['L', nameCrc]]), text,
  ]);
  // A valid 0x7075 unicode path decodes, then counts as hidden metadata.
  assertRefused(writeCrafted('upath.zip', { cdExtra: unicodePath(zlib.crc32(ATTESTATION), ATTESTATION) }), 'upath',
    contractError("artifact archive member 'qualification-attestation.json' carries hidden metadata"));
  assertRefused(writeCrafted('upath-bad.zip', { cdExtra: unicodePath(zlib.crc32(ATTESTATION), Buffer.from([0xff, 0xfe])) }), 'upath-bad',
    contractError('could not read artifact archive: Corrupt unicode path extra field (0x7075): invalid utf-8 bytes'));
  assertRefused(writeCrafted('corrupt-extra.zip', { cdExtra: packFields([['H', 1], ['H', 50]]) }), 'corrupt-extra',
    contractError('could not read artifact archive: Corrupt extra field 0001 (size=50)'));
  // A ZIP64 extra field with nothing to replace, and a 3-byte tail, are only
  // hidden metadata.
  for (const [label, cdExtra] of [['zip64-extra', packFields([['H', 1], ['H', 8], ['Q', 5]])], ['extra-short', Buffer.from([1, 0, 0])]]) {
    assertRefused(writeCrafted(`${label}.zip`, { cdExtra }), label,
      contractError("artifact archive member 'qualification-attestation.json' carries hidden metadata"));
  }
});

test('an extract version above 6.3 and flag bit 5 raise NotImplementedError, as zipfile does', () => {
  assertRefused(writeCrafted('ext-ver-64.zip', { extractVersion: 64 }), 'ext-ver-64', pyError('NotImplementedError', 'zip file version 6.4'));
  assertExtracted(writeCrafted('ext-ver-63.zip', { extractVersion: 63 }), 'ext-ver-63');
  assertRefused(writeCrafted('flag-0x20.zip', { flags: 0x20 }), 'flag-0x20', pyError('NotImplementedError', 'compressed patched data (flag bit 5)'));
  for (const flags of [0x10, 0x4000]) assertExtracted(writeCrafted(`flag-${flags}.zip`, { flags }), `flag-${flags}`);
});

test('member names decode as UTF-8 under flag 0x800 and as cp437 otherwise', () => {
  assertExtracted(writeCrafted('utf8-name.zip', { flags: 0x800 }), 'utf8-name');
  const invalid = Buffer.concat([ATTESTATION.subarray(0, -1), Buffer.from([0xff])]);
  // An undecodable central name escapes as zipfile raises it.
  assertRefused(writeCrafted('utf8-invalid.zip', { name: invalid, flags: 0x800 }), 'utf8-invalid',
    pyError('UnicodeDecodeError', "'utf-8' codec can't decode byte 0xff in position 29: invalid start byte"));
  assertRefused(writeCrafted('utf8-invalid-local.zip', { lname: invalid, flags: 0x800 }), 'utf8-invalid-local',
    contractError("artifact archive member 'qualification-attestation.json' has an undecodable local name"));
  assertRefused(writeCrafted('cp437.zip', { name: Buffer.concat([ATTESTATION.subarray(0, -1), Buffer.from([0x80])]) }), 'cp437',
    contractError("artifact archive contains unauthorized member: 'qualification-attestation.jso\u{c7}'"));
  const utf8Name = Buffer.from('qualification-attestation.jso\u{f1}');
  const cp437Name = Buffer.concat([ATTESTATION.subarray(0, -1), Buffer.from([0xa4])]);
  assertRefused(writeCrafted('cp437-vs-utf8.zip', { name: utf8Name, lname: cp437Name, flags: 0x800 }), 'cp437-vs-utf8',
    contractError("artifact archive contains unauthorized member: 'qualification-attestation.jso\u{f1}'"));
});

// --- Destination pinning -------------------------------------------------------------

test('pinMode auto pins through /proc/self/fd on Linux', { skip: process.platform !== 'linux' && 'Linux only' }, () => {
  const destination = path.join(tmp, 'auto-pin');
  fs.mkdirSync(destination);
  const fd = fs.openSync(destination, fs.constants.O_RDONLY | fs.constants.O_DIRECTORY);
  try {
    const stat = fs.fstatSync(fd, { bigint: true });
    const pinned = new internals.PinnedDirectory(destination, fd, `${stat.dev}:${stat.ino}`, 'auto');
    assert.equal(pinned.mode, 'proc');
  } finally {
    fs.closeSync(fd);
  }
  const archive = writeCandidateArchive('auto-pin.zip');
  assert.equal(extract(archive, path.join(tmp, 'auto-extract'), 'candidate'), undefined);
});

test('pinMode auto fails closed on Linux when /proc/self/fd cannot pin the destination', () => {
  const archive = writeCandidateArchive('no-proc.zip');
  const destination = path.join(tmp, 'no-proc-extract');
  const procFdRoot = path.join(tmp, 'no-proc');
  assertContractError(
    () => extract(archive, destination, 'candidate', { platform: 'linux', procFdRoot }),
    `artifact destination cannot be pinned: ${procFdRoot}/`,
  );
  assert.deepEqual(listDir(destination), []);
  // A /proc entry that names another directory is refused as well.
  fs.mkdirSync(procFdRoot);
  const elsewhere = path.join(tmp, 'elsewhere');
  fs.mkdirSync(elsewhere);
  for (let fd = 0; fd < 1024; fd += 1) fs.symlinkSync(elsewhere, path.join(procFdRoot, String(fd)));
  assertContractError(() => extract(archive, destination, 'candidate', { platform: 'linux', procFdRoot }), 'does not name the pinned directory');
  assert.deepEqual(listDir(destination), []);
  assert.deepEqual(listDir(elsewhere), []);
  // Off Linux, auto is path mode.
  assert.equal(extract(archive, destination, 'candidate', { platform: 'darwin' }), undefined);
  assert.deepEqual(new Set(listDir(destination)), new Set(PUBLICATION_FILES));
});

test('path mode reports a placed member that another file replaced, and leaves that file', () => {
  const archive = writeCandidateArchive('replaced-member.zip');
  const destination = path.join(tmp, 'replaced-member-extract');
  let first = null;
  const replace = (source, target) => {
    fs.renameSync(source, target);
    if (first === null) {
      first = target;
      // Another file takes the placed member's name, and a stray file makes
      // the final listing check fail.
      const impostor = path.join(tmp, 'impostor');
      fs.writeFileSync(impostor, 'impostor');
      fs.renameSync(impostor, target);
      fs.writeFileSync(path.join(destination, 'stray'), 'stray');
    }
  };
  const error = assertContractError(() => extract(archive, destination, 'candidate', { replace, pinMode: 'path' }));
  const name = path.basename(first);
  assert.ok(error.message.startsWith('could not remove partially placed artifact members: '), error.message);
  assert.ok(error.message.includes(`${name}: placed member is no longer at its destination path`), error.message);
  assert.equal(error.cause.message, 'artifact destination changed during extraction');
  assert.equal(fs.readFileSync(first, 'utf8'), 'impostor');
  assert.deepEqual(new Set(listDir(destination)), new Set([name, 'stray']));
});
