// Release fixtures' stand-ins for the built artifacts. Manifest generation
// reads each core's memory address type, so the two .wasm artifacts are tiny
// wasm modules; `marker` rides in a custom section to keep fixtures distinct.

const u32 = (value) => {
  const out = [];
  do {
    out.push((value & 0x7f) | (value > 0x7f ? 0x80 : 0));
    value >>>= 7;
  } while (value > 0);
  return out;
};

const sectionBytes = (id, body) => [id, ...u32(body.length), ...body];

// A module defining one memory per { addressType, shared } spec or address type ('i32' or 'i64').
export function coreWasm(specs, marker = '', { shared = false } = {}) {
  const name = [...Buffer.from('fixture')];
  const normalized = [specs].flat().map((spec) => (typeof spec === 'string'
    ? { addressType: spec, shared }
    : { addressType: spec.addressType, shared: Boolean(spec.shared) }));
  const memories = normalized.flatMap(({ addressType, shared: isShared }) => {
    if (isShared) {
      const flags = (addressType === 'i64' ? 0x04 : 0x00) | 0x03;
      return [flags, 0x01, 0x01];
    }
    return [addressType === 'i64' ? 0x04 : 0x00, 0x01];
  });
  return Buffer.from([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...sectionBytes(0, [...u32(name.length), ...name, ...Buffer.from(marker)]),
    ...sectionBytes(5, [...u32(normalized.length), ...memories]),
  ]);
}

const CORE_MEMORY_SPECS = Object.freeze({
  'llama_webgpu_core.wasm': Object.freeze({ addressType: 'i32', shared: true }),
  'llama_webgpu_core_mem64.wasm': Object.freeze({ addressType: 'i64', shared: true }),
  'llama_webgpu_core_st.wasm': Object.freeze({ addressType: 'i32', shared: false }),
  'llama_webgpu_core_mem64_st.wasm': Object.freeze({ addressType: 'i64', shared: false }),
});

export function artifactBytes(name, marker) {
  const spec = CORE_MEMORY_SPECS[name];
  return spec === undefined ? Buffer.from(marker) : coreWasm(spec, marker);
}
