// Compile-time proof that the implementation matches the published API.
// Types only: nothing imports this module, so it never reaches the bundle.
//
// `implements` alone compares method parameters bivariantly, so a method that
// narrows a parameter (`tokenize(text: 'x')`) would still pass. Rewriting every
// public method as a function-typed property makes `strictFunctionTypes`
// compare parameters contravariantly, so the implementation must accept at
// least what the declaration promises and return no more than it declares.

import type { LlamaWebGpuBridge as PublicLlamaWebGpuBridge } from './llama_webgpu_bridge.d.ts';
import type { LlamaWebGpuBridge } from './bridge.ts';

type StrictMethods<T> = {
  [K in keyof T]: T[K] extends (...args: infer A) => infer R ? (...args: A) => R : T[K];
};

type Assert<T extends true> = T;

export type ImplementationMatchesPublicApi = Assert<
  LlamaWebGpuBridge extends StrictMethods<PublicLlamaWebGpuBridge> ? true : false
>;
