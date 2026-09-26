// Linked with --post-js. Exception catching is enabled only for
// llamadart_webgpu_grammar_sampler_init and llamadart_webgpu_lora_adapter_init
// (EXCEPTION_CATCHING_ALLOWED in CMakeLists.txt), but that link setting also
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
