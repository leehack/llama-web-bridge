// Node-only helpers the driver and CLI suites share, for idioms the Python
// suites write inline: a run-listing route key, wrapping the fake gateway's
// dispatch (Python reassigns gateway.dispatch_workflow), dataclasses.replace
// of a provenance, and the hosted qualification attestation the Python suites
// build with release_qualification_test's fixtures.

import path from 'node:path';

import * as model from '../../../scripts/release/orchestrator/model.mjs';
import * as workflowRuns from '../../../scripts/release/orchestrator/workflow_runs.mjs';
import { buildAttestation, canonicalJson, harnessSourceSha256, loadCandidate } from '../../../scripts/release/qualification.mjs';
import {
  qualificationEnvironment, qualificationIdentity, speechPhase, ttsPhase,
} from '../qualification_fixtures.mjs';
import {
  CANDIDATE_ARTIFACT_ID, CANDIDATE_RUN_ID, DEFAULT_BRANCH, HEAD_SHA, NATIVE_PUBLISHED_AT, QUALIFICATION_RUN_ID,
  provenanceToDict,
} from './fixtures.mjs';

// The scripts/ directory, whose harness sources the attestation digests
// (Python's Path(__file__).resolve().parent).
export const SCRIPTS_DIR = path.resolve(import.meta.dirname, '..', '..', '..', 'scripts');

// workflow_runs._workflow_runs_path(workflow_file=..., default_branch=main,
// created_since=...).
export function runsKey(workflowFile, createdSince = NATIVE_PUBLISHED_AT) {
  return workflowRuns.workflowRunsPath({ workflowFile, defaultBranch: DEFAULT_BRANCH, createdSince });
}

// Wrap gateway.dispatchWorkflow so each dispatch is recorded, then runs
// after(options).
export function onDispatch(gateway, after) {
  const original = gateway.dispatchWorkflow.bind(gateway);
  gateway.dispatchWorkflow = (options) => {
    original(options);
    after(options);
  };
}

// dataclasses.replace(provenance, **overrides) with camelCase keys.
export function replaceProvenance(provenance, overrides) {
  const fields = Object.fromEntries(Object.entries(provenanceToDict(provenance))
    .map(([key, value]) => [key.replace(/_([a-z0-9])/gu, (_, char) => char.toUpperCase()), value]));
  return new model.NativeProvenance({ ...fields, ...overrides });
}

// rq.build_attestation(...) of the candidate in `candidateDir`, qualified by
// `qualificationRunId` from HEAD_SHA, as canonical JSON bytes.
export function attestationBytes(candidateDir, {
  candidateRunId = CANDIDATE_RUN_ID,
  candidateArtifactId = CANDIDATE_ARTIFACT_ID,
  qualificationRunId = QUALIFICATION_RUN_ID,
  manifestDir = candidateDir,
} = {}) {
  const [manifest, fingerprint] = loadCandidate(manifestDir);
  const attestation = buildAttestation({
    manifest,
    candidateFingerprint: fingerprint,
    candidateRunId,
    candidateArtifactId,
    candidateRunAttempt: 1,
    ...qualificationIdentity({ qualificationRunId, qualificationSourceSha: HEAD_SHA }),
    harnessDigest: harnessSourceSha256(SCRIPTS_DIR),
    environment: qualificationEnvironment(),
    speechPhase: speechPhase(),
    ttsPhase: ttsPhase(),
  });
  return Buffer.from(canonicalJson(attestation), 'utf8');
}
