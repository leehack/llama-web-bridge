// Funnel for every call into a Wasm core, and the errors of a core that aborted.

export const MODEL_UNLOADED_AFTER_CORE_ABORT = 'MODEL_UNLOADED_AFTER_CORE_ABORT';

export const MODEL_UNLOADED_AFTER_CORE_ABORT_MESSAGE =
  'The native core aborted and was replaced, which unloaded the model. Call loadModelFromUrl again.';

export class ModelUnloadedError extends Error {
  readonly code = MODEL_UNLOADED_AFTER_CORE_ABORT;

  constructor(message = MODEL_UNLOADED_AFTER_CORE_ABORT_MESSAGE) {
    super(message);
    this.name = 'ModelUnloadedError';
  }
}

/**
 * A trap or abort() surfaces as a WebAssembly.RuntimeError of the realm that
 * instantiated the core, which is not this realm for an iframe core. Such a
 * core hands over its own realm's class as `llamadartWasmRuntimeError`; an
 * error is never classified by its name.
 */
export function isWasmRuntimeError(error: unknown, coreRuntimeError: unknown = null): boolean {
  if (typeof WebAssembly !== 'undefined' && error instanceof WebAssembly.RuntimeError) {
    return true;
  }
  if (typeof coreRuntimeError !== 'function') {
    return false;
  }
  try {
    return error instanceof coreRuntimeError;
  } catch (_) {
    // Not a constructor `instanceof` accepts.
    return false;
  }
}

interface CoreGuardState {
  retired: boolean;
  cause: unknown;
  retire: (cause: unknown, fromThrow: boolean) => void;
  release: () => void;
}

interface CoreRetirementStats {
  threads: number;
  stagedBytes: number;
}

const guardStates = new WeakMap<object, CoreGuardState>();
const rawCores = new WeakMap<object, object>();

// Per-core state must survive a second bridge guarding the same core.
export function unguardedCore<T extends object>(core: T): T {
  return (rawCores.get(core) as T | undefined) ?? core;
}

export function isCoreRetired(core: unknown): boolean {
  return !!core && typeof core === 'object' && guardStates.get(core)?.retired === true;
}

// Disposal uses the retirement hook without triggering abort recovery.
export function releaseCore(core: unknown): void {
  if (core && typeof core === 'object') {
    guardStates.get(core)?.release();
  }
}

// Retires a guarded core for an abort no guarded call observed (onAbort).
export function retireCore(core: unknown, reason: unknown): void {
  if (core && typeof core === 'object') {
    guardStates.get(core)?.retire(new Error(`Aborted(${String(reason ?? '')})`), false);
  }
}

/**
 * Wraps a core module so every function read from it, from its `FS`, and every
 * function such a function returns (`cwrap`) goes through one check: a retired
 * core is never called again, and a call that traps retires it.
 */
export function guardCore<T extends object>(
  rawModule: T,
  onRetired: (cause: unknown, stats: CoreRetirementStats | undefined) => void,
): T {
  const wrappers = new WeakMap<object, unknown>();
  const coreRuntimeError = (rawModule as { llamadartWasmRuntimeError?: unknown }).llamadartWasmRuntimeError;
  const isTrap = (error: unknown) => isWasmRuntimeError(error, coreRuntimeError);
  const runRetireHook = () =>
    (rawModule as { llamadartRetire?: () => CoreRetirementStats }).llamadartRetire?.();
  const state: CoreGuardState = {
    retired: false,
    cause: null,
    retire(cause, fromThrow) {
      if (state.retired) {
        // onAbort runs before abort() throws; keep the error the caller saw.
        if (fromThrow && !isTrap(state.cause)) {
          state.cause = cause;
        }
        return;
      }
      state.retired = true;
      state.cause = cause;
      let stats: CoreRetirementStats | undefined;
      try {
        stats = runRetireHook();
      } catch (_) {
        // A release failure must not hide the abort.
      }
      onRetired(cause, stats);
    },
    release() {
      if (state.retired) {
        return;
      }
      state.retired = true;
      state.cause = new Error('The bridge core was disposed');
      runRetireHook();
    },
  };

  const observe = (error: unknown) => {
    if (isTrap(error)) {
      state.retire(error, true);
    }
  };

  const wrapValue = (value: unknown, owner: object): unknown => {
    if (typeof value !== 'function') {
      return value;
    }
    const cached = wrappers.get(value);
    if (cached) {
      return cached;
    }
    const wrapped = function guarded(this: unknown, ...args: unknown[]) {
      if (state.retired) {
        throw state.cause;
      }
      let result: unknown;
      try {
        result = Reflect.apply(value, this === proxy || this === fsProxy ? owner : this, args);
      } catch (error) {
        observe(error);
        throw error;
      }
      if (typeof result === 'function') {
        return wrapValue(result, owner);
      }
      if (result && typeof (result as PromiseLike<unknown>).then === 'function') {
        return (result as PromiseLike<unknown>).then(
          (settled) => settled,
          (error) => {
            observe(error);
            throw error;
          },
        );
      }
      return result;
    };
    wrappers.set(value, wrapped);
    return wrapped;
  };

  let fsProxy: object | null = null;
  const proxy: T = new Proxy(rawModule, {
    get(target, property) {
      const value = Reflect.get(target, property, target);
      if (property === 'FS' && value && typeof value === 'object') {
        fsProxy ??= new Proxy(value, {
          get(fsTarget, fsProperty) {
            return wrapValue(Reflect.get(fsTarget, fsProperty, fsTarget), fsTarget);
          },
        });
        return fsProxy;
      }
      return wrapValue(value, target);
    },
  });
  guardStates.set(proxy, state);
  rawCores.set(proxy, rawModule);
  return proxy;
}
