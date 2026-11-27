// Linked with --post-js. Exception catching is enabled only for the functions
// in EXCEPTION_CATCHING_ALLOWED (CMakeLists.txt), but that link setting also
// makes every other C++ throw escape ccall as a CppException instead of
// aborting. Restore the abort, so an uncaught throw still reaches onAbort and
// rejects with "Aborted(...)" as it does in a build without exception
// catching. This covers every bridge call, since the bridge enters the core
// only through ccall; a throw on a pthread or in main() is outside it.
{
  const catchingCcall = ccall;
  const abortOnUncaughtCppException = (error) => {
    if (typeof CppException === 'function' && error instanceof CppException) {
      abort();
    }
    throw error;
  };
  Module['ccall'] = (...args) => {
    let result;
    try {
      result = catchingCcall(...args);
    } catch (error) {
      abortOnUncaughtCppException(error);
    }
    return result instanceof Promise
      ? result.catch(abortOnUncaughtCppException)
      : result;
  };
}

Module['llamadartWasmRuntimeError'] = WebAssembly.RuntimeError;

Module['llamadartRetire'] = () => {
  let threads = -1;
  if (typeof PThread !== 'undefined') {
    threads = Object.keys(PThread.pthreads).length + PThread.unusedWorkers.length;
    // Besides the pool, this wakes the calling thread's mailbox
    // Atomics.waitAsync. The browser keeps a pending wait alive, and with it
    // this module and its memory.
    try {
      PThread.terminateRuntime();
    } catch (_) {
      PThread.terminateAllThreads();
    }
  }
  let stagedBytes = 0;
  const files = llamadartJsModelFsFiles();
  for (const file of files.values()) {
    if (file.externalMode === undefined) stagedBytes += file.size;
    llamadartJsModelFsDiscardFile(file);
  }
  files.clear();
  llamadartJsModelFsHandles().clear();
  return { threads, stagedBytes };
};
