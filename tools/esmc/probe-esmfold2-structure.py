"""What a compressed ESM-C costs the STRUCTURE, not the pair representation.

    .venv-esm/bin/python tools/esmc/probe-esmfold2-structure.py

Needs the `esm` package, which is why it is not run by the plain interpreter:
`probe-esmc-compression.py` measures the tensor ESMFold2 receives and this
measures what ESMFold2 does with it. The two are different questions and the
AF3 side of this repository has the scar to prove it - forty-eight pairformer
blocks swallowed a factor of 1200 in the term that fed them, so an error
measured on a stage's input is an upper bound on the damage and often a wild
one.

🔴 THE FOLDING MODEL NEVER MOVES, AND NEITHER DOES THE SEED. `forward` takes
`lm_hidden_states` directly, so an arm replaces exactly the tensor whose error
the other probe reports and nothing else - not the weights, not the diffusion
noise, not the number of loops. Anything that differs between two arms here is
the language model's quantisation and can be nothing else.

🔴 AND `no language model` IS THE CONTROL THAT GIVES EVERY OTHER ROW A SCALE.
"1.4 A from the float32 fold" means nothing until you know what dropping the
language model entirely costs; ESMFold2 folds without one, so that arm exists.
A scheme whose damage is a small fraction of that is a scheme that works.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import esmc_forward as E          # noqa: E402
import quantisation as Q          # noqa: E402

FIXTURES = ROOT / 'tools' / 'fixtures'
UBIQUITIN = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'


def parse_crystal(text):
    """-> (one-letter sequence, CA coordinates), from ONE parse of the file.

    🔴 TWO PARSERS OF THE SAME PDB DISAGREE, AND THE DISAGREEMENT IS SILENT
    UNTIL AN RMSD REFUSES IT. The first version read the sequence with a
    (chain, residue) dedupe over ATOM and HETATM and the coordinates with a
    plain scan of ATOM CA lines: 6MRR came out as 68 residues and 71
    coordinates, because an alternate conformation writes its alpha carbon
    twice. One pass, one dedupe, one length.
    """
    three = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
             'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
             'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'MSE': 'M', 'PHE': 'F',
             'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y',
             'VAL': 'V'}
    seen, sequence, coordinates = set(), [], []
    for line in text.splitlines():
        if line[:4] not in ('ATOM', 'HETA') or line[12:16].strip() != 'CA':
            continue
        key = (line[21], line[22:27])
        if key in seen:
            continue
        seen.add(key)
        sequence.append(three.get(line[17:20].strip(), 'X'))
        coordinates.append([float(line[30:38]), float(line[38:46]),
                            float(line[46:54])])
    return ''.join(sequence), np.array(coordinates, np.float64)


PROTEIN_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',
    'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
    'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
    'Y': 'TYR', 'V': 'VAL',
}


def write_ca_pdb(path, coordinates, sequence):
    """A trace, so a fold can be looked at. Not a deliverable, an aid."""
    lines = []
    for i, (x, y, z) in enumerate(coordinates):
        lines.append('ATOM  %5d  CA  %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C'
                     % (i + 1, PROTEIN_1TO3.get(sequence[i], 'UNK'), i + 1, x, y, z))
    path.write_text('\n'.join(lines) + '\nEND\n')


def rmsd(a, b):
    """Kabsch-superposed RMSD. Raises rather than truncating a length mismatch."""
    if a.shape != b.shape:
        raise ValueError('%s against %s' % (a.shape, b.shape))
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((a @ rotation - b) ** 2).sum() / len(a)))


def tm_score(a, b):
    """TM-score of a against b, superposed by Kabsch over all of it."""
    n = len(a)
    d0 = 1.24 * (n - 15) ** (1 / 3) - 1.8 if n > 21 else 0.5
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    moved = a @ (u @ np.diag([1.0, 1.0, d]) @ vt)
    distance = np.sqrt(((moved - b) ** 2).sum(1))
    return float((1.0 / (1.0 + (distance / d0) ** 2)).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esmc', default='esmc-600m')
    parser.add_argument('--esmfold2', default='esmfold2-fast-600m')
    parser.add_argument('--schemes',
                        default='int8 g64 sym,int5 g32 asym,int4 g32 asym,int3 g32 asym')
    parser.add_argument('--targets', default='1ubq,6mrr,1qys')
    parser.add_argument('--pdb-dir', default=None,
                        help='fold every .pdb here instead, scored against it')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--calibrated', default='',
                        help='label=path.npz, comma separated - arms whose '
                             'codes were chosen against real activations')
    parser.add_argument('--fold-bits', type=int, default=0,
                        help='also quantise the FOLDING model at this many bits')
    parser.add_argument('--fold-group', type=int, default=32)
    parser.add_argument('--fold-keep', type=int, default=1 << 16,
                        help='tensors smaller than this stay float32')
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed-spread', type=int, default=0,
                        help='extra float32 seeds, to say what "free" means')
    parser.add_argument('--out', default=None, help='write every fold as a PDB here')
    parser.add_argument('--csv', default=None,
                        help='every (arm, target) row, so a mean is not the '
                             'only summary available afterwards')
    arguments = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    from esm.models.esmfold2 import EsmFold2ExperimentalModel
    from esm.models.esmfold2.protein_utils import prepare_protein_features

    sequences, reference_ca = {}, {}
    if arguments.pdb_dir:
        for path in sorted(pathlib.Path(arguments.pdb_dir).glob('*.pdb')):
            sequences[path.stem], reference_ca[path.stem] = parse_crystal(
                path.read_text())
    else:
        crystals = {'6mrr': FIXTURES / '6mrr-crystal.pdb',
                    '1qys': FIXTURES / '1qys-crystal.pdb'}
        sequences = {'1ubq': UBIQUITIN}
        for name, path in crystals.items():
            sequences[name], reference_ca[name] = parse_crystal(path.read_text())
        sequences = {k: v for k, v in sequences.items()
                     if k in arguments.targets.split(',')}

    print('loading the folding model (the language model is supplied per arm)',
          flush=True)
    model = EsmFold2ExperimentalModel.from_pretrained(
        str(ROOT / arguments.esmfold2), load_esmc=False,
        device=arguments.device).eval()
    if arguments.fold_bits:
        # 🔴 THE BUNDLE IS TWO MODELS AND ONLY ONE OF THEM WAS EVER PRICED.
        # The folding model is 171 M parameters - 34% of the 600M bundle and
        # 41% of the 300M one at equal bits - so a size target that quantises
        # only the tower is a size target for two thirds of the download. Same
        # rule as tools/quantize_af3.py: anything 1-D, or smaller than
        # --fold-keep, stays float32, because norms and biases are a rounding
        # error of the bytes and sit where an error is not averaged away.
        touched = kept = 0
        for name, parameter in model.named_parameters():
            values = parameter.detach().cpu().numpy().reshape(-1)
            if parameter.dim() < 2 or values.size < arguments.fold_keep:
                kept += 1
                continue
            approximate = Q.asymmetric(values.astype(np.float32),
                                       arguments.fold_bits, arguments.fold_group)
            parameter.data.copy_(torch.as_tensor(
                approximate.reshape(parameter.shape), dtype=parameter.dtype,
                device=parameter.device))
            touched += 1
        print('folding model: %d tensors at int%d group %d, %d kept float32'
              % (touched, arguments.fold_bits, arguments.fold_group, kept),
              flush=True)

    checkpoint = E.Checkpoint(ROOT / arguments.esmc)

    features = {}
    for name, sequence in sequences.items():
        features[name] = prepare_protein_features(sequence)

    def fold(name, hidden, seed=None):
        """One fold -> its representative-atom coordinates, seed pinned.

        🔴 NOT THROUGH `output_to_pdb`, WHICH WANTS A pLDDT THIS CHECKPOINT
        CANNOT PRODUCE. The `-step1500k` ablation checkpoints ship with
        `confidence_head.enabled: false`, so the fold succeeds and the PDB
        writer raises KeyError on a field the model does not have.
        `distogram_atom_idx` is the per-token representative atom - the alpha
        carbon for a standard residue - which is what an RMSD wants anyway.
        """
        seed = arguments.seed if seed is None else seed
        torch.manual_seed(seed)
        moved = {k: (v.to(arguments.device) if hasattr(v, 'to') else v)
                 for k, v in features[name].items()}
        output = model(**moved, lm_hidden_states=hidden,
                       num_diffusion_samples=arguments.samples,
                       seed=seed)
        coordinates = output['sample_atom_coords']
        if coordinates.dim() == 4:
            coordinates = coordinates[:, 0]
        index = moved['distogram_atom_idx'][0].long()
        return coordinates[0][index].to(torch.float64).cpu().numpy()

    def tower_for(scheme):
        """One quantised tower per ARM, not per target - the weights do not
        depend on the sequence and requantising 573M of them three times is
        twenty seconds each of nothing."""
        cache = {}
        supply = (lambda key: cache.setdefault(
            key, scheme(checkpoint[key], key) if scheme.keyed
            else scheme(checkpoint[key])))
        return E.Tower(checkpoint, weights=supply, device=arguments.device)

    def hidden_for(tower, name):
        states = tower.hidden_states(E.sequence_ids(sequences[name]))
        return states[:, 1:-1, :].permute(1, 0, 2)[None]

    catalogue = Q.catalogue()
    names = [n.strip() for n in arguments.schemes.split(',') if n.strip()]
    for entry in (e for e in arguments.calibrated.split(',') if e.strip()):
        label, _, where = entry.partition('=')
        scheme = Q.calibrated(where, label)
        catalogue[scheme.name] = scheme
        names.append(scheme.name)
    outdir = pathlib.Path(arguments.out) if arguments.out else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    exact = tower_for(catalogue['float32'])
    print('%d targets, %d-%d residues, on %s'
          % (len(sequences), min(len(v) for v in sequences.values()),
             max(len(v) for v in sequences.values()), arguments.device),
          flush=True)
    baseline, results, rows_csv = {}, {}, []
    print('\nfloat32 language model (the reference every arm is measured '
          'against)', flush=True)
    for name in sequences:
        baseline[name] = fold(name, hidden_for(exact, name))
        if outdir:
            write_ca_pdb(outdir / ('%s.float32.pdb' % name),
                         baseline[name], sequences[name])
        if name in reference_ca:
            rows_csv.append(('float32', name, len(sequences[name]), 0.0, 1.0,
                             rmsd(baseline[name], reference_ca[name])))
        note = ('  vs crystal %.2f A, TM %.3f'
                % (rmsd(baseline[name], reference_ca[name]),
                   tm_score(baseline[name], reference_ca[name]))
                if name in reference_ca else '')
        print('  %-6s %3d residues%s' % (name, len(baseline[name]), note),
              flush=True)

    if arguments.seed_spread:
        # 🔴 "0.06 A from the float32 fold" MEANS NOTHING UNTIL THE SAMPLER'S
        # OWN SPREAD IS KNOWN. Two seeds of the SAME weights differ, and a
        # scheme that moves the structure by less than that has not moved it.
        print('\nthe same float32 weights at %d more seeds (the yardstick)'
              % arguments.seed_spread, flush=True)
        spread = []
        for name in sequences:
            for seed in range(1, arguments.seed_spread + 1):
                other = fold(name, hidden_for(exact, name), seed=seed)
                spread.append(rmsd(other, baseline[name]))
        print('  seed-to-seed RMSD: mean %.2f A, worst %.2f A over %d folds'
              % (np.mean(spread), np.max(spread), len(spread)), flush=True)

    print('\nno language model at all (the control that gives the rest a scale)',
          flush=True)
    for name in sequences:
        coordinates = fold(name, None)
        results.setdefault('no language model', {})[name] = coordinates
        if outdir:
            write_ca_pdb(outdir / ('%s.no-lm.pdb' % name), coordinates,
                         sequences[name])
        print('  %-6s %5.2f A from the float32 fold, TM %.3f'
              % (name, rmsd(coordinates, baseline[name]),
                 tm_score(coordinates, baseline[name])), flush=True)

    header = '%-24s %6s %8s %8s %8s' % ('scheme', 'bits/w', 'RMSD', 'TM',
                                        'crystal~')
    print('\n' + header)
    print('-' * len(header))
    for label in ['no language model'] + names:
        scheme = catalogue.get(label)
        rows, crystal_rows = [], []
        tower = None if scheme is None else tower_for(scheme)
        for name in sequences:
            if label in results and name in results[label]:
                coordinates = results[label][name]
            else:
                coordinates = fold(name, hidden_for(tower, name))
                if outdir:
                    write_ca_pdb(outdir / ('%s.%s.pdb'
                                           % (name, label.replace(' ', '_'))),
                                 coordinates, sequences[name])
            moved_by = rmsd(coordinates, baseline[name])
            rows.append((moved_by, tm_score(coordinates, baseline[name])))
            against = rmsd(coordinates, reference_ca[name]) if name in reference_ca else None
            if against is not None:
                crystal_rows.append(against)
            rows_csv.append((label, name, len(sequences[name]), moved_by,
                             tm_score(coordinates, baseline[name]), against))
        worst = max(rows, key=lambda r: r[0])
        print('%-24s %6s %7.2fA %8.3f %8s   median %5.2fA  worst %.2fA'
              % (label, '-' if scheme is None else '%.2f' % scheme.bits_per_weight,
                 np.mean([r for r, _ in rows]), np.mean([t for _, t in rows]),
                 '%.2fA' % np.median(crystal_rows) if crystal_rows else '-',
                 np.median([r for r, _ in rows]), worst[0]), flush=True)
    if arguments.csv:
        import csv as csv_module
        with open(arguments.csv, 'w', newline='') as handle:
            writer = csv_module.writer(handle)
            writer.writerow(['scheme', 'target', 'residues', 'rmsd_from_float32',
                             'tm_to_float32', 'rmsd_to_crystal'])
            writer.writerows(rows_csv)
        print('\nper-target rows in %s' % arguments.csv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
