// Fixtures shared by the manifest and publication-state tests.
//
// loadCandidate is release_qualification.load_candidate, ported in
// scripts/release/qualification.mjs: it reads the candidate manifest strictly,
// rebuilds the CandidateIdentity from its fields, and validates the directory
// with the same validateCandidate publication uses.

export { loadCandidate } from '../../scripts/release/qualification.mjs';
