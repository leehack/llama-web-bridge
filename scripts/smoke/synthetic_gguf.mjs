// A 41 MB llama GGUF with random weights whose two 20 MB tensors each span more
// than one 16 MiB read step, so reader faults can land after the first step.
const VOCAB = 80000;
const EMBED = 64;
const FFN = 128;

const string = (value) => {
  const data = Buffer.from(value);
  const out = Buffer.alloc(8 + data.length);
  out.writeBigUInt64LE(BigInt(data.length));
  data.copy(out, 8);
  return out;
};
const u32 = (value) => { const b = Buffer.alloc(4); b.writeUInt32LE(value); return b; };
const u64 = (value) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(value)); return b; };
const f32 = (value) => { const b = Buffer.alloc(4); b.writeFloatLE(value); return b; };
const pad = (buffer) => Buffer.concat([buffer, Buffer.alloc((32 - (buffer.length % 32)) % 32)]);

function random(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function buildLargeOutputGguf() {
  const tokens = ['<unk>', '<s>', '</s>', '▁', 'a', 'b', 'c', '<0x0A>'];
  for (let i = 0; tokens.length < VOCAB; i += 1) tokens.push(`t${i}`);
  const types = tokens.map((_, i) => (i === 0 ? 2 : i < 3 ? 3 : i === 7 ? 6 : 1));
  const arrayOf = (elementType, items) => Buffer.concat([u32(elementType), u64(items.length), ...items]);
  const metadata = [
    ['general.architecture', 8, string('llama')],
    ['general.name', 8, string('large output fixture')],
    ['llama.attention.layer_norm_rms_epsilon', 6, f32(1e-5)],
    ['tokenizer.ggml.model', 8, string('llama')],
    ['tokenizer.ggml.tokens', 9, arrayOf(8, tokens.map(string))],
    ['tokenizer.ggml.scores', 9, arrayOf(6, tokens.map(() => f32(0)))],
    ['tokenizer.ggml.token_type', 9, arrayOf(5, types.map((t) => { const b = Buffer.alloc(4); b.writeInt32LE(t); return b; }))],
    ...Object.entries({
      'llama.context_length': 64, 'llama.embedding_length': EMBED, 'llama.block_count': 1,
      'llama.feed_forward_length': FFN, 'llama.attention.head_count': 4, 'llama.attention.head_count_kv': 4,
      'llama.rope.dimension_count': 16, 'tokenizer.ggml.bos_token_id': 1, 'tokenizer.ggml.eos_token_id': 2,
    }).map(([key, value]) => [key, 4, u32(value)]),
  ];
  const tensors = [
    ['token_embd.weight', [EMBED, VOCAB]], ['output_norm.weight', [EMBED]], ['output.weight', [EMBED, VOCAB]],
    ['blk.0.attn_norm.weight', [EMBED]], ['blk.0.attn_q.weight', [EMBED, EMBED]],
    ['blk.0.attn_k.weight', [EMBED, EMBED]], ['blk.0.attn_v.weight', [EMBED, EMBED]],
    ['blk.0.attn_output.weight', [EMBED, EMBED]], ['blk.0.ffn_norm.weight', [EMBED]],
    ['blk.0.ffn_gate.weight', [EMBED, FFN]], ['blk.0.ffn_down.weight', [FFN, EMBED]],
    ['blk.0.ffn_up.weight', [EMBED, FFN]],
  ];
  const next = random(346);
  const header = [Buffer.from('GGUF'), u32(3), u64(tensors.length), u64(metadata.length)];
  for (const [key, type, value] of metadata) header.push(string(key), u32(type), value);
  const data = [];
  let offset = 0;
  for (const [name, shape] of tensors) {
    const count = shape.reduce((a, b) => a * b, 1);
    const values = new Float32Array(count);
    for (let i = 0; i < count; i += 1) values[i] = name.endsWith('norm.weight') ? 1 : next() - 0.5;
    const body = pad(Buffer.from(values.buffer));
    header.push(string(name), u32(shape.length), ...shape.map(u64), u32(0), u64(offset));
    data.push(body);
    offset += body.length;
  }
  return Buffer.concat([pad(Buffer.concat(header)), ...data]);
}
