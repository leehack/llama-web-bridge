// Shared error type of the Node release tooling, the port of
// release_contract.py's ContractError.

// Raised when release provenance is ambiguous or unsafe.
export class ContractError extends Error {
  constructor(message, options) {
    super(message, options);
    this.name = 'ContractError';
  }
}
