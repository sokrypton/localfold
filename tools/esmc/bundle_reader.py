"""Read a LocalFold weight bundle from Python, so a probe can price the artefact.

The compression study quantises ESM-C's tensors in memory, in the layout the
safetensors file has. The BUNDLE is a different thing: `export_esmc_model.py`
transposes the projections into (inner, outer) for the GPU, and
`quantize_af3.py` then groups over THAT flattened order. So the two schemes put
their group boundaries on different sets of weights and are not the same
quantisation, however alike their statistics.

🔴 WHICH MEANS A NUMBER MEASURED ON ONE IS NOT A NUMBER ABOUT THE OTHER. int3 at
group 128 was measured folding at float32's median crystal RMSD through the
in-memory path; this exists so the claim can be made about the thing that would
ship instead. Same reason the GPU checkers read the bundle rather than a
float32 stand-in for it.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

# The bundle's name for each ESM-C tensor, and whether the export transposed it.
BLOCK_NAMES = {
    'attn.layernorm_qkv.layer_norm_weight': ('attn_norm/scale', False),
    'attn.layernorm_qkv.layer_norm_bias': ('attn_norm/offset', False),
    'attn.layernorm_qkv.weight': ('qkv/weights', True),
    'attn.q_ln.weight': ('q_norm/scale', False),
    'attn.k_ln.weight': ('k_norm/scale', False),
    'attn.out_proj.weight': ('attn_out/weights', True),
    'ffn.layer_norm_weight': ('ffn_norm/scale', False),
    'ffn.layer_norm_bias': ('ffn_norm/offset', False),
    'ffn.fc1_weight': ('fc1/weights', True),
    'ffn.fc2_weight': ('fc2/weights', True),
}


def _decode(record, blob):
    """One tensor, dequantised. Mirrors src/reference/dtype.js."""
    shape = tuple(record['shape'])
    count = int(np.prod(shape))
    start = record.get('byteOffset', 0)
    dtype = record['dtype']
    if dtype == 'float32':
        return np.frombuffer(blob, '<f4', count, start).reshape(shape)

    bits = int(dtype[3:])
    group = record['block']
    groups = -(-count // group)
    group_bytes = group * bits // 8
    codes = np.frombuffer(blob, np.uint8, groups * group_bytes, start)
    scales = np.frombuffer(blob, '<f2', groups, record['scaleOffset']).astype(np.float32)
    zeros = np.frombuffer(blob, '<f2', groups, record['zeroOffset']).astype(np.float32)
    # 🔴 LEAST SIGNIFICANT BIT FIRST, which is what np.packbits was told and what
    # the shader assumes. Read the other way every code is a different number
    # and the tensor still has the right shape.
    stream = np.unpackbits(codes.reshape(groups, group_bytes), axis=1,
                           bitorder='little')
    values = stream.reshape(groups, group, bits)
    weights = (1 << np.arange(bits, dtype=np.uint32))
    decoded = (values * weights).sum(-1).astype(np.float32)
    out = (decoded * scales[:, None] + zeros[:, None]).reshape(-1)[:count]
    return out.reshape(shape)


class Bundle:
    """A manifest plus its shards, addressed by ESM-C's own tensor names."""

    def __init__(self, directory):
        self.root = pathlib.Path(directory)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.tensors = self.manifest['tensors']
        self._blobs = {}

    def _blob(self, name):
        if name not in self._blobs:
            self._blobs[name] = (self.root / name).read_bytes()
        return self._blobs[name]

    def bundle_name(self, key):
        """`esmc.transformer.blocks.7.ffn.fc1_weight` -> (name, transposed)."""
        if key == 'esmc.embed.weight':
            return 'embed/weights', False
        if key == 'esmc.transformer.norm.weight':
            return 'final_norm/scale', False
        parts = key.split('.')
        if len(parts) > 4 and parts[2] == 'blocks':
            leaf = '.'.join(parts[4:])
            name, transposed = BLOCK_NAMES[leaf]
            return 'blocks/%s/%s' % (parts[3], name), transposed
        raise KeyError(key)

    def __getitem__(self, key):
        """-> the tensor in ESM-C's OWN layout, transposed back where needed."""
        name, transposed = self.bundle_name(key)
        record = self.tensors[name]
        values = _decode(record, self._blob(record['file']))
        return np.ascontiguousarray(values.T) if transposed else values

    @property
    def encoding(self):
        return self.tensors['blocks/0/fc1/weights']['dtype']

    @property
    def group(self):
        return self.tensors['blocks/0/fc1/weights'].get('block')
