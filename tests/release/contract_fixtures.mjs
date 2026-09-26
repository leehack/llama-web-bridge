// Fixtures shared by the release contract tests: the port of
// release_contract_test.py's release_attestation() builder.

import { IMMUTABLE_RELEASE_ATTESTATION_PREDICATE_TYPE, IMMUTABLE_RELEASE_ATTESTATION_SIGNER } from '../../scripts/release/contract.mjs';
import { isDict, pyJsonDumps } from '../../scripts/release/json.mjs';

export const ASSETS_REPO = 'leehack/llama-web-bridge-assets';
export const TAG_COMMIT = 'a'.repeat(40);
export const RUN_ID = '123456789';
export const RUN_URL = `https://github.com/leehack/llama-web-bridge/actions/runs/${RUN_ID}`;

// Build a `gh release verify --format json` payload shaped like GitHub's.
export function releaseAttestation({
  releaseTag = 'v0.1.39',
  assetsRepo = ASSETS_REPO,
  tagCommit = TAG_COMMIT,
  releaseId = 1,
  assets = null,
  statementOverrides = null,
  resultOverrides = null,
} = {}) {
  const purl = `pkg:github/${assetsRepo}@${releaseTag}`;
  const subjects = [{ uri: purl, digest: { sha1: tagCommit } }];
  for (const [name, digest] of Object.entries(assets ?? {})) subjects.push({ name, digest: { sha256: digest } });
  const statement = {
    _type: 'https://in-toto.io/Statement/v1',
    subject: subjects,
    predicateType: IMMUTABLE_RELEASE_ATTESTATION_PREDICATE_TYPE,
    predicate: {
      databaseId: String(releaseId),
      ownerId: '2',
      packageId: '3',
      purl,
      repository: assetsRepo,
      repositoryId: '3',
      tag: releaseTag,
    },
  };
  Object.assign(statement, statementOverrides ?? {});
  const result = {
    mediaType: 'application/vnd.dev.sigstore.verificationresult+json;version=0.1',
    signature: {
      certificate: {
        certificateIssuer: 'CN=Fulcio Intermediate l1,O=GitHub\\, Inc.',
        subjectAlternativeName: IMMUTABLE_RELEASE_ATTESTATION_SIGNER,
      },
    },
    verifiedTimestamps: [
      { type: 'TimestampAuthority', uri: 'timestamp.githubapp.com', timestamp: '2026-08-20T22:15:59Z' },
    ],
    verifiedIdentity: {
      subjectAlternativeName: { subjectAlternativeName: '', regexp: '^https://dotcom\\.releases\\.github\\.com$' },
      issuer: { issuer: '', regexp: '.*' },
    },
    statement,
  };
  Object.assign(result, resultOverrides ?? {});
  const signed = isDict(result.statement) ? result.statement : statement;
  return {
    attestation: {
      bundle: {
        mediaType: 'application/vnd.dev.sigstore.bundle.v0.3+json',
        verificationMaterial: {
          certificate: { rawBytes: 'Zm9v' },
          timestampVerificationData: { rfc3161Timestamps: [{ signedTimestamp: 'Zm9v' }] },
        },
        dsseEnvelope: {
          payloadType: 'application/vnd.in-toto+json',
          payload: Buffer.from(pyJsonDumps(signed), 'utf8').toString('base64'),
          signatures: [{ sig: 'Zm9v' }],
        },
      },
      bundle_url: '',
      initiator: '',
    },
    verificationResult: result,
  };
}
