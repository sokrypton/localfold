"""How far can ESM-C 600M be compressed before ESMFold2 stops recognising it?

    python3 tools/esmc/probe-esmc-compression.py
    python3 tools/esmc/probe-esmc-compression.py --schemes "int5 g32 asym,int4 g32 asym"

WHAT IS MEASURED, AND WHY NOT PERPLEXITY. ESMFold2 does not read ESM-C's
logits. It reads all 37 hidden states, mixes them with a learned softmax and
projects the result into a pair representation - so the number that decides
whether a scheme is usable is the error in THAT, not in the language model's own
output. The masked-LM recovery column is carried anyway, because it is the one
column a human can read without a reference: it says whether the tower is still
a protein language model at all.

🔴 THE MIX IS NOT UNIFORM AND WEIGHTING BY IT CHANGES THE ANSWER. States 12-17
carry 0.01% of the softmax between them; a scheme that wrecks them and leaves
the last three alone would read as a disaster on a flat per-layer average and as
nothing at all on the mix. Both columns are printed for that reason.

🔴 THE CHECKPOINT IS bfloat16 STORED AS float32, SO THE FIRST 2x IS FREE AND IS
NOT COMPRESSION. Not one of the 95.6M weights sampled has any of its low sixteen
mantissa bits set - the tower was trained in bfloat16 and widened on the way
out, which is also how transformers runs it (`torch.autocast(bfloat16)` around
the ESM-C call in ESMFold2's own forward). So the bfloat16 arm reads relRMS
EXACTLY ZERO on weights and on every downstream column, and float16 reads 3e-9,
which is the tiny-weight tail flushing below float16's subnormal floor rather
than any rounding. Quote a scheme against the 16 bits that are really there,
not against the 32 the file is written in.

🔴 AND A HIDDEN STATE IS NOT THE OUTPUT EITHER. The shim LayerNorms its input
and the pair MLP LayerNorms its output, so a relative error in the residual
stream is renormalised twice before the folding trunk sees it. `pair` is the
column that prices a scheme; `single` and the per-layer figures say where the
damage is.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import esmc_forward as E          # noqa: E402
import quantisation as Q          # noqa: E402

# Three lengths, three folds. Ubiquitin is the model card's own example; the
# other two are READ FROM the crystal structures the AF3 side of this repository
# already scores against, so a later end-to-end check has the same targets and
# no sequence is retyped from memory - the first version of this file guessed
# 6MRR's and got a different protein.
UBIQUITIN = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'
FIXTURES = ROOT / 'tools' / 'fixtures'


def sequences_from(path):
    """Read one-letter chain sequences out of a PDB, for --targets-from."""
    three = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
             'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
             'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'MSE': 'M', 'PHE': 'F',
             'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y',
             'VAL': 'V'}
    seen, out = set(), []
    for line in pathlib.Path(path).read_text().splitlines():
        if line[:4] not in ('ATOM', 'HETA') or line[12:16].strip() != 'CA':
            continue
        key = (line[21], line[22:27])
        if key in seen:
            continue
        seen.add(key)
        out.append(three.get(line[17:20].strip(), 'X'))
    return ''.join(out)


def targets():
    return {
        '1ubq': UBIQUITIN,
        '6mrr': sequences_from(FIXTURES / '6mrr-crystal.pdb'),
        '1qys': sequences_from(FIXTURES / '1qys-crystal.pdb'),
    }


class Reference:
    """The float32 tower's answers, computed once and kept for every arm."""

    def __init__(self, checkpoint, shim, sequences, device):
        self.states, self.single, self.pair, self.ids = {}, {}, {}, {}
        tower = E.Tower(checkpoint, device=device)
        for name, sequence in sequences.items():
            ids = E.sequence_ids(sequence)
            states = tower.hidden_states(ids)
            self.ids[name] = ids
            self.states[name] = states
            self.single[name] = shim.single(states)
            self.pair[name] = shim.pair(self.single[name])
        self.logits = {n: tower.logits(self.ids[n]) for n in sequences}


def masked_recovery(tower, ids, stride=5):
    """Fraction of masked residues the head puts back. Chance is about 5%."""
    positions = np.arange(1, len(ids) - 1, stride)
    masked = ids.copy()
    masked[positions] = E.MASK
    predicted = tower.logits(masked)[positions].argmax(-1).cpu().numpy()
    return float((predicted == ids[positions]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esmc', default='esmc-600m')
    parser.add_argument('--esmfold2', default='esmfold2-fast-600m')
    parser.add_argument('--schemes', default='',
                        help='comma-separated subset of the catalogue')
    parser.add_argument('--targets', default='',
                        help='comma-separated subset of 1ubq,6mrr,1qys')
    parser.add_argument('--mixed', default='',
                        help='per-layer allocation arms, e.g. 4/6@6,3/6@6')
    parser.add_argument('--per-layer', action='store_true',
                        help='print the per-state error of every arm')
    parser.add_argument('--device', default='cpu')
    arguments = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    catalogue = Q.catalogue()
    extra = []
    for spec in (a for a in arguments.mixed.split(',') if a.strip()):
        low_high, _, count = spec.partition('@')
        low, _, high = low_high.partition('/')
        extra.append((int(low), int(high), int(count)))
    names = ([n.strip() for n in arguments.schemes.split(',') if n.strip()]
             or list(catalogue))
    sequences = {k: v for k, v in targets().items()
                 if not arguments.targets or k in arguments.targets.split(',')}

    checkpoint = E.Checkpoint(ROOT / arguments.esmc)
    for spec in extra:
        scheme = Q.mixed(spec, checkpoint.n_layers)
        catalogue[scheme.name] = scheme
        names.append(scheme.name)
    shim = E.Shim(ROOT / arguments.esmfold2, device=arguments.device)
    matrices = checkpoint.matrices()
    quantisable = sum(int(np.prod(shape)) for _, shape in matrices)
    total = quantisable + sum(
        int(np.prod(checkpoint.file.shape(k))) for k in checkpoint.file.keys()
        if k.startswith('esmc.') and not any(k == m for m, _ in matrices))
    print('ESM-C %d layers x %d, %.1fM parameters, %.1fM of them in the %d '
          'matrices a scheme touches (%.2f%%)'
          % (checkpoint.n_layers, checkpoint.d_model, total / 1e6,
             quantisable / 1e6, len(matrices), 100 * quantisable / total))

    # The layer mix, which is what makes a weighted per-state error meaningful.
    mix = shim.combine.cpu().numpy()
    print('layer mix: last three states carry %.1f%%, states 12-17 carry %.3f%%'
          % (100 * mix[-3:].sum(), 100 * mix[12:18].sum()))

    reference = Reference(checkpoint, shim, sequences, arguments.device)
    print('\ntargets: ' + ', '.join('%s (%d)' % (n, len(s))
                                    for n, s in sequences.items()))

    header = ('%-26s %6s %8s %10s %10s %10s %10s %7s'
              % ('scheme', 'bits/w', 'MiB', 'weights', 'states', 'mixed',
                 'pair', 'mask%'))
    print('\n' + header)
    print('-' * len(header))
    rows = []
    for name in names:
        scheme = catalogue[name]
        started = time.time()
        cache, weight_error = {}, [0.0, 0.0]

        def supply(key, _scheme=scheme, _cache=cache, _e=weight_error):
            if key not in _cache:
                original = checkpoint[key]
                approximate = _scheme(original, key)
                _e[0] += float(((approximate - original) ** 2).sum())
                _e[1] += float((original ** 2).sum())
                _cache[key] = approximate
            return _cache[key]

        tower = E.Tower(checkpoint, weights=supply, device=arguments.device)
        flat, mixed, single, pair, mask = [], [], [], [], []
        for target in sequences:
            ids = reference.ids[target]
            states = tower.hidden_states(ids)
            per_state = np.array([
                E.relative_rms(states[k], reference.states[target][k])
                for k in range(states.shape[0])])
            flat.append(per_state.mean())
            mixed.append(float((mix * per_state).sum()))
            one = shim.single(states)
            single.append(E.relative_rms(one, reference.single[target]))
            pair.append(E.relative_rms(shim.pair(one), reference.pair[target]))
            mask.append(masked_recovery(tower, ids))
            if arguments.per_layer:
                print('    %s %s: ' % (name, target)
                      + ' '.join('%d:%.1e' % (k, v)
                                 for k, v in enumerate(per_state)))
        weights = float(np.sqrt(weight_error[0] / weight_error[1]))
        mib = quantisable * scheme.bits_per_weight / 8 / 2 ** 20
        row = (name, scheme.bits_per_weight, mib, weights, float(np.mean(flat)),
               float(np.mean(mixed)), float(np.mean(single)),
               float(np.mean(pair)), float(np.mean(mask)))
        rows.append(row)
        print('%-26s %6.2f %8.1f %10.2e %10.2e %10.2e %10.2e %6.1f%%   (%.0fs)'
              % (row[0], row[1], row[2], row[3], row[4], row[5], row[7],
                 100 * row[8], time.time() - started))
        del tower, cache

    print('\nThe `pair` column is the one that prices a scheme: it is what the '
          'folding trunk\nreceives. `weights` is what a weight-space study '
          'would have reported instead.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
