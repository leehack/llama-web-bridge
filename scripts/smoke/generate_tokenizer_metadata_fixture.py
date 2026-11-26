#!/usr/bin/env python3
"""Write the synthetic GGUF used by the state-persistence browser smoke."""

from argparse import ArgumentParser
from math import prod
from pathlib import Path
from struct import pack


def gguf_string(value):
    data = value.encode('utf-8')
    return pack('<Q', len(data)) + data


def padded(data):
    return data + bytes(-len(data) % 32)


def generate_fixture():
    tokens = ['<unk>', '<fixture-bos>', '<fixture-\x0beos>', '▁', 'a', 'b', 'c', '<0x0A>']
    kv = [
        ('general.architecture', 8, gguf_string('llama')),
        ('general.name', 8, gguf_string('Synthetic tokenizer metadata fixture')),
        ('llama.attention.layer_norm_rms_epsilon', 6, pack('<f', 1e-5)),
        ('tokenizer.ggml.model', 8, gguf_string('llama')),
        ('tokenizer.ggml.tokens', 9, pack('<IQ', 8, len(tokens)) + b''.join(map(gguf_string, tokens))),
        ('tokenizer.ggml.scores', 9, pack('<IQ', 6, len(tokens)) + bytes(4 * len(tokens))),
        ('tokenizer.ggml.token_type', 9, pack('<IQ8i', 5, len(tokens), 2, 3, 3, 1, 1, 1, 1, 6)),
        ('tokenizer.ggml.bos_token', 8, gguf_string('raw-bos-placeholder')),
        ('tokenizer.ggml.eos_token', 8, gguf_string('raw-eos-placeholder')),
        ('tokenizer.ggml.add_bos_token', 7, pack('<?', False)),
    ]
    for key, value in {
        'llama.context_length': 64, 'llama.embedding_length': 8, 'llama.block_count': 1,
        'llama.feed_forward_length': 16, 'llama.attention.head_count': 1,
        'llama.attention.head_count_kv': 1, 'llama.rope.dimension_count': 8,
        'tokenizer.ggml.bos_token_id': 1, 'tokenizer.ggml.eos_token_id': 2,
    }.items():
        kv.append((key, 4, pack('<I', value)))

    tensors = {
        'token_embd.weight': (8, len(tokens)), 'output_norm.weight': (8,),
        'blk.0.attn_norm.weight': (8,), 'blk.0.attn_q.weight': (8, 8),
        'blk.0.attn_k.weight': (8, 8), 'blk.0.attn_v.weight': (8, 8),
        'blk.0.attn_output.weight': (8, 8), 'blk.0.ffn_norm.weight': (8,),
        'blk.0.ffn_gate.weight': (8, 16), 'blk.0.ffn_down.weight': (16, 8),
        'blk.0.ffn_up.weight': (8, 16),
    }
    header = b'GGUF' + pack('<IQQ', 3, len(tensors), len(kv))
    header += b''.join(gguf_string(key) + pack('<I', kind) + value for key, kind, value in kv)
    data = b''
    for name, shape in tensors.items():
        header += gguf_string(name) + pack('<I', len(shape))
        header += b''.join(pack('<Q', size) for size in shape) + pack('<IQ', 0, len(data))
        data += padded(bytes(4 * prod(shape)))
    return padded(header) + data


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.parse_args().output.write_bytes(generate_fixture())
