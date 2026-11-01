#!/usr/bin/env python3
"""Add chai-lab's STRUCTURE token-pair weights to a Chai-1 bundle - the half af3-any-model's converter drops.

  ~/venv_chai/bin/python tools/export_chai1_structure_pair.py --assets <chai-lab models_v2 dir> --bundle model-chai1-f32

chai-lab (chai1.py, run_folding_on_context) chunks its 512-wide TOKEN_PAIR projection and its bond projection
in two: the first halves build the trunk's z_init, the second halves are the diffusion module's pair input
(`token_pair_structure_input_feats`). af3-any-model's converters/chai1.py keeps only the trunk halves and
conditions the diffusion on the trunk's z_init instead - measured on chai-lab's own tensors for 6MRR, the two
are relRMS 9.2 apart. So this reads the structure halves out of chai-lab's TorchScript checkpoints, unchanged:

  diffuser/chai1_structure_token_pair/weights  [163, 256]  input_projs.TOKEN_PAIR.0.weight rows 256:512, transposed
  diffuser/chai1_structure_token_pair/bias     [256]       its bias, same rows
  diffuser/chai1_structure_bond/weights        [1, 256]    bond_loss_input_proj weight rows 256:512, transposed

The 163 inputs are chai's own column order (alphabetical generators): docking 0:6, relative chain 6:12, relative
entity 12:15, relative sequence separation 15:82, relative token separation 82:149, token distance restraint
149:156, pocket restraint 156:163.
"""
import argparse
import json
import pathlib

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', required=True, help="chai-lab's models_v2 directory (its .pt files)")
    parser.add_argument('--bundle', required=True)
    arguments = parser.parse_args()
    assets, bundle = pathlib.Path(arguments.assets), pathlib.Path(arguments.bundle)
    fe = torch.jit.load(str(assets / 'feature_embedding.pt'), map_location='cpu').state_dict()
    bond = torch.jit.load(str(assets / 'bond_loss_input_proj.pt'), map_location='cpu').state_dict()
    w = fe['input_projs.TOKEN_PAIR.0.weight'].numpy()          # [512, 163]
    b = fe['input_projs.TOKEN_PAIR.0.bias'].numpy()            # [512]
    wb = bond['weight'].numpy()                                 # [512, 1]
    if w.shape != (512, 163) or wb.shape != (512, 1):
        raise SystemExit('unexpected shapes %s %s' % (w.shape, wb.shape))
    half = w.shape[0] // 2
    tensors = {
        'diffuser/chai1_structure_token_pair/weights': np.ascontiguousarray(w[half:].T, '<f4'),
        'diffuser/chai1_structure_token_pair/bias': np.ascontiguousarray(b[half:], '<f4'),
        'diffuser/chai1_structure_bond/weights': np.ascontiguousarray(wb[half:].T, '<f4'),
    }
    manifest = json.loads((bundle / 'manifest.json').read_text())
    shard = 'weights-chai1-structure.f32.bin'
    offset, records = 0, manifest['tensors']
    with open(bundle / shard, 'wb') as f:
        for name, values in tensors.items():
            records[name] = {'file': shard, 'shape': list(values.shape), 'byteOffset': offset, 'dtype': 'float32'}
            f.write(values.tobytes()); offset += values.nbytes
    keep = set(manifest.get('float32Tensors', []))
    keep.add('diffuser/chai1_structure_token_pair/bias')
    manifest['float32Tensors'] = sorted(keep)
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('%s: + %s' % (bundle.name, ', '.join('%s %s' % (k, tuple(v.shape)) for k, v in tensors.items())))


if __name__ == '__main__':
    main()
