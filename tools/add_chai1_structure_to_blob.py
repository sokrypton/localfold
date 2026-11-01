#!/usr/bin/env python3
"""Add chai-lab's structure token-pair weights to af3-any-model's published chai1 blobs - the weights its converter drops.

  ~/.venv-lfjax/bin/python tools/add_chai1_structure_to_blob.py --bundle model-chai1-f32 \
      --reference /tmp/claude-1000/af3up/src chai1/chai1.bin.zst chai1/chai1.fp16.bin.zst chai1/chai1.int8.bin.zst

chai-lab's diffusion module reads a pair input of its own, its STRUCTURE token-pair features, through the second
halves of `input_projs.TOKEN_PAIR` and `bond_loss_input_proj` (tools/export_chai1_structure_pair.py, which puts
them in LocalFold's bundle). af3-any-model's converters/chai1.py keeps only the trunk halves, so its blobs lack
them - and LocalFold's native port, which reads those blobs as they are published, needs them:

  diffuser/chai1_structure_token_pair/weights  [163, 256]
  diffuser/chai1_structure_token_pair/bias     [256]
  diffuser/chai1_structure_bond/weights        [1, 256]

Each blob is rewritten in place with the three records appended (replaced if present), in the precision its
other unquantised tensors use: float32 in the float32 blob, float16 in the fp16 and int8 ones. Every other
record is copied byte for byte. The records go through af3-any-model's own encoder (--reference: its src/).
"""
import argparse
import io
import json
import pathlib
import sys

import numpy as np

NAMES = {
    ('diffuser/chai1_structure_token_pair', 'weights'): 'diffuser/chai1_structure_token_pair/weights',
    ('diffuser/chai1_structure_token_pair', 'bias'): 'diffuser/chai1_structure_token_pair/bias',
    ('diffuser/chai1_structure_bond', 'weights'): 'diffuser/chai1_structure_bond/weights',
}


def bundle_tensor(root, name):
    manifest = json.loads((root / 'manifest.json').read_text())['tensors'][name]
    if manifest['dtype'] != 'float32':
        raise SystemExit('%s is %s in %s; want the float32 bundle' % (name, manifest['dtype'], root))
    n = int(np.prod(manifest['shape']))
    return np.fromfile(root / manifest['file'], '<f4', count=n, offset=manifest['byteOffset']).reshape(manifest['shape'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True, help='LocalFold\'s float32 chai1 bundle (with the structure shard)')
    parser.add_argument('--reference', required=True, help="af3-any-model's src/ (its params.encode_record)")
    parser.add_argument('blobs', nargs='+')
    arguments = parser.parse_args()
    sys.path.insert(0, arguments.reference)
    import zstandard
    from alphafold3.model import params
    root = pathlib.Path(arguments.bundle)
    values = {key: bundle_tensor(root, name) for key, name in NAMES.items()}
    for path in map(pathlib.Path, arguments.blobs):
        with open(path, 'rb') as handle:
            raw = zstandard.ZstdDecompressor().stream_reader(handle).read()
        records = list(params.read_records(io.BytesIO(raw)))
        kept = [(s, n, a) for s, n, a in records if (s, n) not in values]
        # the precision of the blob's other unquantised tensors (biases and norms: never int8)
        dtypes = {str(a.dtype) for s, n, a in kept
                  if n.endswith(('bias', 'scale', 'offset')) and not n.endswith('__q_scale') and a.dtype != np.int8}
        if dtypes not in ({'float32'}, {'float16'}):
            raise SystemExit('%s: its unquantised tensors are %s' % (path, sorted(dtypes)))
        dtype = np.float32 if dtypes == {'float32'} else np.float16
        out = io.BytesIO()
        for s, n, a in kept:
            out.write(params.encode_record(s, n, a))
        for (s, n), v in values.items():
            out.write(params.encode_record(s, n, v.astype(dtype)))
        data = zstandard.ZstdCompressor(level=19, threads=-1).compress(out.getvalue())
        path.write_bytes(data)
        check = list(params.read_records(io.BytesIO(zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read())))
        assert len(check) == len(kept) + len(values)
        print('%s: %d records kept, %d added as %s, %.1f MB' % (path, len(kept), len(values), np.dtype(dtype).name,
                                                               len(data) / 1e6))


if __name__ == '__main__':
    main()
