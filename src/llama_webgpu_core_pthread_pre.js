// Runs before PThread.init(), where Emscripten first reads mainScriptUrlOrBlob.
Module[Symbol.for('llamadart.pthreadScriptUrl')]?.(import.meta.url);
