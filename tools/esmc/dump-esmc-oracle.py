"""Dump ESM-C's answers, so a WebGPU tower has something to be wrong against.

    python3 tools/esmc/dump-esmc-oracle.py --sequence-length 59

Writes `oracle-dumps/esmc-<n>.json`, which `tools/gpu/check-esmc-*.js` fetch the
way every other checker in this repository fetches its oracle.

🔴 THE REFERENCE IS GATED, WHICH IS WHY IT CAN BE AN ORACLE. `esmc_forward.py`
is written from the block layout `../alphafold3/converters/esmc.py` documents
rather than by importing the `esm` package, and it agrees with `transformers`'
own `EsmcForMaskedLM` to **2.1e-6** on every one of the 37 hidden states and
with the `esm` package's own `language_model` module to **3.2e-7** on the pair
representation. A dump from an unchecked reference is not an oracle, it is a
second opinion.

🔴 AND IT CARRIES ONE BLOCK'S INTERMEDIATES, NOT ONLY THE ENDS. A tower that
disagrees at state 36 tells you nothing about where; the four matmul inputs of
block 0 and the two residual points let a checker localise before it bisects.
That is the ladder the AF3 port had to build twice.

🔴 AND THE SHIM IS TAKEN FROM THE FOLDING CHECKPOINT, NEVER FROM A CACHE. Every
ESMFold2 release trains its own `language_model.*` and they share only the
tower; ../alphafold3 fed one variant another's and read corr 0.026 where its own
reads 0.999998, costing 8.798 A against 0.812. The dump records which folding
model it came from so a checker can refuse a mismatched pair.
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

import esmc_forward as E          # noqa: E402

# A fixed, arbitrary sequence: deterministic, not a protein anyone is attached
# to, and long enough that a wrong RoPE convention cannot hide in the first few
# positions. Trimmed to --sequence-length.
SEQUENCE = ('MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYN'
            'IQKESTLHLVLRLRGGMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGT'
            'QDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDED')


def tolist(tensor):
    return np.asarray(tensor.detach().cpu(), np.float32).reshape(-1).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esmc', default='esmc-600m')
    parser.add_argument('--esmfold2', default='esmfold2-fast-600m')
    parser.add_argument('--sequence-length', type=int, default=59)
    parser.add_argument('--out', default=None)
    # The pair representation is L^2 x 256 floats: 36 MB of JSON at 59 residues
    # and 1.6 GB at 400. The tower checker never reads it - only the reference
    # checker builds a pair - so a length sweep skips it.
    parser.add_argument('--skip-pair', action='store_true')
    arguments = parser.parse_args()

    torch.set_grad_enabled(False)
    sequence = SEQUENCE[:arguments.sequence_length]
    checkpoint = E.Checkpoint(ROOT / arguments.esmc)
    tower = E.Tower(checkpoint)
    shim = E.Shim(ROOT / arguments.esmfold2)

    ids = E.sequence_ids(sequence)
    embedded = tower.embed(ids)
    positions = tower.positions(embedded.shape[0])

    # Block 0's four matmul inputs and its output, for localisation.
    record = {}
    first = tower.block(embedded, 0, positions, record=record)

    states = tower.hidden_states(ids)
    single = shim.single(states[:, 1:-1, :])
    pair = None if arguments.skip_pair else shim.pair(single)

    payload = {
        'sequence': sequence,
        'tokens': [int(i) for i in ids],
        'esmc': arguments.esmc,
        'esmfold2': arguments.esmfold2,
        'dims': {
            'layers': checkpoint.n_layers, 'model': checkpoint.d_model,
            'heads': checkpoint.n_heads, 'vocab': checkpoint.vocab,
            'residualScale': checkpoint.residual_scale,
        },
        'embedded': tolist(embedded),
        'block0': {name: tolist(value) for name, value in record.items()},
        'block0Output': tolist(first),
        # Every state would be 37 x L x 1152; the ones a checker actually needs
        # are the first, the last, and the two the layer mix weighs most.
        'states': {str(k): tolist(states[k]) for k in (0, 1, 18, 35, 36)},
        'mix': tolist(shim.combine),
        'single': tolist(single),
        **({} if pair is None else {'pair': tolist(pair)}),
        'shapes': {
            'embedded': list(embedded.shape), 'block0Output': list(first.shape),
            'state': list(states[0].shape), 'single': list(single.shape),
            **({} if pair is None else {'pair': list(pair.shape)}),
            **{name: list(value.shape) for name, value in record.items()},
        },
    }

    out = pathlib.Path(arguments.out) if arguments.out else (
        ROOT / 'oracle-dumps' / ('esmc-%d.json' % len(sequence)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print('wrote %s  (%d tokens with BOS/EOS, %.1f MB)'
          % (out, len(ids), out.stat().st_size / 1e6))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
