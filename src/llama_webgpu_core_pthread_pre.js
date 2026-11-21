// An embedder may supply the constructor for this core's pthread workers, for
// pages that may not construct a Worker from the core's own URL.
var Worker = Module[Symbol.for('llamadart.pthreadWorker')] || globalThis.Worker;
