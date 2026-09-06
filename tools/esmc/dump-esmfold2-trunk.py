"""Dump ESMFold2's folding trunk, so a WebGPU port has something to be wrong against.

    .venv-esm/bin/python tools/esmc/dump-esmfold2-trunk.py --sequence-length 40

Writes `oracle-dumps/esmfold2-trunk-<n>.json`.

🔴 THE TRUNK IS DUMPED FROM z_init, NOT FROM A SEQUENCE. Everything upstream -
the featuriser, the relative position encoding, the language-model shim - is a
separate port with its own failure modes, and a trunk checked end-to-end would
be checking all of them at once. Given z_init and the pair mask, the trunk is a
pure function; that is what this records.

🔴 AND IT RECORDS EVERY LOOP, BECAUSE THE RECURRENCE IS WHERE A PORT DIVERGES
SLOWLY. `z = z_init + pair_loop_proj(z)` then 24 blocks, four times over - a
port that is slightly wrong in one block reads as slightly wrong after the
first loop and badly wrong after the fourth, and only the per-loop values say
which.

🔴 AND THE LOOP RUNS n_loops + 1 TIMES. The config says `num_loops: 3` and the
model runs four iterations. Reading that as three is a silent quarter less
trunk, which still folds.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

SEQUENCE = ('MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYN'
            'IQKESTLHLVLRLRGGMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGT')


def tolist(tensor):
    return np.asarray(tensor.detach().cpu(), np.float32).reshape(-1).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esmfold2', default='esmfold2-fast-600m')
    parser.add_argument('--sequence-length', type=int, default=40)
    parser.add_argument('--out', default=None)
    arguments = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    from esm.models.esmfold2 import EsmFold2ExperimentalModel
    from esm.models.esmfold2.protein_utils import prepare_protein_features

    sequence = SEQUENCE[:arguments.sequence_length]
    model = EsmFold2ExperimentalModel.from_pretrained(
        str(ROOT / arguments.esmfold2), load_esmc=False).eval()

    captured = {'loops': [], 'projected': []}

    def after_projection(_module, _inputs, output):
        captured['projected'].append(output.detach().clone())

    def after_trunk(_module, _inputs, output):
        pair = output[0] if isinstance(output, (tuple, list)) else output
        captured['loops'].append(pair.detach().clone())

    handles = [model.pair_loop_proj.register_forward_hook(after_projection),
               model.folding_trunk.register_forward_hook(after_trunk)]

    features = prepare_protein_features(sequence)
    # 🔴 NO LANGUAGE MODEL, DELIBERATELY. lm_z is added to z_init once and is
    # this port's OTHER half; leaving it out makes the trunk's own arithmetic
    # the only thing recorded, and the tower already has its own oracle.
    output = model(**features, num_diffusion_samples=1, seed=0)
    for handle in handles:
        handle.remove()

    if not captured['loops']:
        raise SystemExit('the trunk hook never fired; the module name has moved')

    trunk = model.folding_trunk
    blocks = len(trunk.blocks) if hasattr(trunk, 'blocks') else None
    z_init = captured['projected'][0] - torch.zeros_like(captured['projected'][0])
    payload = {
        'sequence': sequence,
        'esmfold2': arguments.esmfold2,
        'loops': len(captured['loops']),
        'blocks': blocks,
        'shapes': {'pair': list(captured['loops'][0].shape)},
        # The first projection sees a zero z, so its output is
        # pair_loop_proj(0) - the bias path alone - and z_init is what the
        # model added to it. Both are recorded rather than reconstructed.
        'projectionOfZero': tolist(captured['projected'][0]),
        'afterLoop': {str(i): tolist(v) for i, v in enumerate(captured['loops'])},
        'distogram': tolist(output['distogram_logits'])
        if 'distogram_logits' in output else None,
    }

    out = pathlib.Path(arguments.out) if arguments.out else (
        ROOT / 'oracle-dumps' / ('esmfold2-trunk-%d.json' % len(sequence)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print('wrote %s  (%d residues, %d loops, %s blocks, %.1f MB)'
          % (out, len(sequence), len(captured['loops']), blocks,
             out.stat().st_size / 1e6))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
