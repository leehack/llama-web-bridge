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

// A module defining one memory per address type ('i32' or 'i64').
export function coreWasm(addressTypes, marker = '') {
  const name = [...Buffer.from('fixture')];
  const memories = [addressTypes].flat().flatMap((type) => [type === 'i64' ? 0x04 : 0x00, 0x01]);
  return Buffer.from([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...sectionBytes(0, [...u32(name.length), ...name, ...Buffer.from(marker)]),
    ...sectionBytes(5, [...u32(memories.length / 2), ...memories]),
  ]);
}

const CORE_ADDRESS_TYPES = Object.freeze({
  'llama_webgpu_core.wasm': 'i32',
  'llama_webgpu_core_mem64.wasm': 'i64',
});

export function artifactBytes(name, marker) {
  const addressType = CORE_ADDRESS_TYPES[name];
  return addressType === undefined ? Buffer.from(marker) : coreWasm(addressType, marker);
}
