"""Synthyra's ESMFold2 ConfidenceHead, inputs and outputs, from THEIR code.

    ~/venv_ef2/bin/python tools/oracle/dump_esmfold2_confidence.py

🔴 THE HEAD IS THE ONLY PART OF THIS CHECKPOINT WE DID NOT ALREADY HAVE.
biohub ships ESMFold2 with `confidence_head.enabled: false` and zero confidence
tensors; Synthyra froze that trunk and trained one on it. The trunk underneath
is ours byte for byte (see tools/export_esmfold2_trunk.py's note), so the only
thing that needs an oracle is the head's arithmetic.

🔴 AND IT RUNS THEIR MODULE, NOT A READING OF IT. `fastplms_bundle.py` in the
Synthyra repo is base85 of a zip of readable sources; this unpacks that archive
and imports `ConfidenceHead` out of it. A reference written from reading their
forward pass and compared against a port written from the same reading is two
copies of one misunderstanding agreeing perfectly - which is the trap
docs/SMILES.md records for a benzene written and read back as a CCD code.

🔴 THE COORDINATES ARE A REAL STRUCTURE AND THE REPRESENTATIONS ARE NOT.
`x_pred` comes from a PDB, because the head buckets rep-atom distances into 128
bins over [2, 52] and a random cloud puts every pair in one column of that
embedding - the failure `dump_af3_opendde_confidence.py` records as POS_SCALE.
`z` and `s_inputs` are seeded normals: this gates the head's ARITHMETIC, and a
run on a real trunk's pair is a second arm that wants our own fold to dump one.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import re
import sys
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def unpack_runtime(bundle: pathlib.Path) -> pathlib.Path:
    """Their `fastplms` package, out of the base85 zip they ship it in."""
    text = bundle.read_text()
    blob = ''.join(re.findall(r"^\s*'([^']*)'", text, re.M))
    archive = base64.b85decode(blob)
    if archive[:2] != b'PK':
        raise SystemExit('%s did not decode to a zip' % bundle)
    out = pathlib.Path(tempfile.mkdtemp(prefix='fastplms-'))
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        zf.extractall(out)
    return out


def alpha_carbons(pdb: pathlib.Path):
    """(residue name, [(atom name, x, y, z)]) per residue, in file order."""
    residues, order = {}, []
    for line in pdb.read_text().splitlines():
        if not line.startswith('ATOM'):
            continue
        key = (line[21], int(line[22:26]))
        if key not in residues:
            residues[key] = []
            order.append(key)
        residues[key].append((line[12:16].strip(),
                              float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return [residues[k] for k in order]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', default='', help='fastplms_bundle.py from the Synthyra repo')
    parser.add_argument('--config', default='', help='its config.json')
    parser.add_argument('--head', default='', help='.npz of its confidence_head.* tensors')
    parser.add_argument('--pdb', default=str(ROOT / 'tools' / 'fixtures' / '6mrr-crystal.pdb'))
    parser.add_argument('--tokens', type=int, default=0, help='0 means every residue in the PDB')
    parser.add_argument('--seed', type=int, default=0)
    # 🔴 WITHOUT THIS THE ipTM ARM CANNOT FAIL. A monomer has no inter-chain
    # pair, so their `iptm` is 0 and any port returning 0 agrees with it - a
    # gate that passes by having nothing to compare. `--split N` puts tokens
    # from N on a second `asym_id`, which is the only thing ipTM reads.
    parser.add_argument('--split', type=int, default=0,
                        help='first token of a second chain; 0 means one chain')
    parser.add_argument('--out', default=str(ROOT / 'oracle-dumps' /
                                             'esmfold2-confidence-600.json'))
    arguments = parser.parse_args()
    for needed in ('bundle', 'config', 'head'):
        if not getattr(arguments, needed):
            raise SystemExit('--%s is required; see the header' % needed)

    sys.path.insert(0, str(unpack_runtime(pathlib.Path(arguments.bundle))))
    import numpy as np
    import torch
    from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from fastplms.models.esmfold2.modeling_esmfold2_experimental import ConfidenceHead

    config = ESMFold2Config(**json.loads(pathlib.Path(arguments.config).read_text()))
    torch.manual_seed(arguments.seed)
    head = ConfidenceHead(config).eval()

    weights = np.load(arguments.head)
    state = {name: torch.from_numpy(weights[name].copy()) for name in weights.files}
    missing, unexpected = head.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit('the head has tensors this module does not: %s' % sorted(unexpected))
    print('loaded %d tensors; module wanted %d it did not get: %s'
          % (len(state), len(missing), sorted(missing)), flush=True)

    residues = alpha_carbons(pathlib.Path(arguments.pdb))
    if arguments.tokens:
        residues = residues[:arguments.tokens]
    tokens = len(residues)
    coords, atom_to_token, rep_idx = [], [], []
    for index, atoms in enumerate(residues):
        rep = next((a for a in range(len(atoms)) if atoms[a][0] == 'CA'), 0)
        rep_idx.append(len(coords) + rep)
        for _, x, y, z in atoms:
            coords.append((x, y, z))
            atom_to_token.append(index)
    atoms_total = len(coords)

    generator = torch.Generator().manual_seed(arguments.seed + 1)
    d_inputs = config.inputs.d_inputs
    d_pair = config.d_pair
    s_inputs = torch.randn(1, tokens, d_inputs, generator=generator) * 0.5
    pair = torch.randn(1, tokens, tokens, d_pair, generator=generator) * 0.5
    inputs = {
        's_inputs': s_inputs,
        'z': pair,
        'x_pred': torch.tensor(coords, dtype=torch.float32).unsqueeze(0),
        'distogram_atom_idx': torch.tensor(rep_idx).unsqueeze(0),
        'token_attention_mask': torch.ones(1, tokens, dtype=torch.long),
        'atom_to_token': torch.tensor(atom_to_token).unsqueeze(0),
        'atom_attention_mask': torch.ones(1, atoms_total, dtype=torch.long),
        'asym_id': (torch.arange(tokens) >= arguments.split).long().unsqueeze(0)
                   if arguments.split else torch.zeros(1, tokens, dtype=torch.long),
        'mol_type': torch.zeros(1, tokens, dtype=torch.long),
    }
    with torch.no_grad():
        out = head(**inputs)

    stages = {}
    def record(name, tensor):
        array = np.asarray(tensor.detach().to(torch.float32).cpu()).reshape(-1)
        stages[name] = {'shape': list(tensor.shape),
                        'rms': float(np.sqrt((array.astype(np.float64) ** 2).mean())),
                        'data': [float(v) for v in array]}
    for name, tensor in inputs.items():
        record('in.%s' % name, tensor.to(torch.float32))
    for name, tensor in sorted(out.items()):
        if torch.is_tensor(tensor):
            record('out.%s' % name, tensor)
            print('  out.%-22s %-18s rms %.6f'
                  % (name, tuple(tensor.shape), stages['out.%s' % name]['rms']), flush=True)

    target = pathlib.Path(arguments.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        'model': 'esmfold2-confidence', 'tokens': tokens, 'atoms': atoms_total,
        'pdb': pathlib.Path(arguments.pdb).name, 'seed': arguments.seed,
        'split': arguments.split,
        'dPair': d_pair, 'dSingle': config.d_single, 'dInputs': d_inputs,
        'stages': stages}))
    print('wrote %s (%.1f MiB)' % (target, target.stat().st_size / 2 ** 20))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
