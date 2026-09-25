// Types shared across bridge modules. Types only: this module has no runtime code.

import type { EmscriptenFs } from './download.ts';

// Logger method names the runtime routes core and bridge output through.
export type LogMethod = 'log' | 'warn' | 'error';

// Emscripten `ccall` argument type names the runtime passes.
export type CcallArgType = 'number' | 'string' | 'array';

// Emscripten `ccall` argument values: 'array' takes bytes, 'string' may be null.
export type CcallArg = number | string | Uint8Array | null;

// The Emscripten FS subset the runtime uses, on top of the download helpers'.
export interface LlamaCoreFs extends EmscriptenFs {
  readFile(path: string): Uint8Array;
}

// Emscripten pthread pool internals, inspected only as a fallback thread-count
// probe.
export interface LlamaCorePThread {
  unusedWorkers?: unknown;
  runningWorkers?: unknown;
  allocateUnusedWorker?: unknown;
}

// The instantiated llama_webgpu_core module surface the runtime uses. The
// build exports only `FS`, `ccall`, and the bridge's C entry points; the
// optional members are probed before use.
export interface LlamaCoreModule {
  ccall(
    ident: string,
    returnType: 'number',
    argTypes: readonly CcallArgType[],
    args: readonly CcallArg[],
    opts: { async: true },
  ): Promise<number>;
  ccall(
    ident: string,
    returnType: 'number',
    argTypes: readonly CcallArgType[],
    args: readonly CcallArg[],
  ): number;
  ccall(
    ident: string,
    returnType: 'string',
    argTypes: readonly CcallArgType[],
    args: readonly CcallArg[],
  ): string | null;
  ccall(
    ident: string,
    returnType: null,
    argTypes: readonly CcallArgType[],
    args: readonly CcallArg[],
  ): void;
  FS: LlamaCoreFs;
  HEAP8?: Int8Array;
  HEAPU8?: Uint8Array;
  wasmMemory?: WebAssembly.Memory;
  PThread?: LlamaCorePThread;
  _malloc?: (size: number) => number;
  _free?: (ptr: number) => void;
  // Present only in builds that include decision heads.
  _llamadart_webgpu_decision_capabilities_json?: unknown;
}
