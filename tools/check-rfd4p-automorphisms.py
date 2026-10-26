#!/usr/bin/env python3
"""Hold shared/rfd4p/symmetry.js's automorphisms to atomworks' own, over the component dictionary.

    ~/venv_rfd4p/bin/python tools/check-rfd4p-automorphisms.py [--sample=N] [--seed=S] [CODE ...]

A motif RMSD resolves chemically equivalent atoms first (upstream's `permute_gt_atoms`), and which
atoms are equivalent is atomworks' `generate_automorphisms_from_atom_array_with_networkx`: the bond
graph's automorphisms, nodes coloured by element and heavy-atom degree, edges by a bond order
RDKit has normalised for resonance. For every component - the twenty amino acids, the
nucleotides, and a seeded sample of biotite's dictionary - this builds the heavy-atom array a
design would carry, asks atomworks for its automorphisms, and requires the SAME SET from
tools/rfd4p-automorphisms.mjs, which runs this port's. No gate design holds a coordinate-
conditioned ligand, so this is where a ligand's symmetry is checked at all.
"""
from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STANDARD = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
            "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "A", "C", "G", "U", "DA", "DC", "DG", "DT",
            "SAM", "HEM", "NAG", "ATP", "FAD", "GOL", "SO4", "PO4", "BEN", "CIT"]
MAX_AUTOMORPHS = 1000


def main() -> int:
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    import biotite.structure.info as info
    import numpy as np
    from atomworks.ml.transforms.symmetry import (count_heavy_neighbors,
                                                  generate_automorphisms_from_atom_array_with_networkx)

    args = sys.argv[1:]
    opt = {a.split("=", 1)[0][2:]: a.split("=", 1)[1] for a in args if a.startswith("--")}
    codes = [a for a in args if not a.startswith("--")]
    if not codes:
        rng = random.Random(int(opt.get("seed", "0")))
        codes = STANDARD + rng.sample(sorted(info.all_residues()), int(opt.get("sample", "1500")))
    cases = []
    for code in codes:
        try:
            arr = info.residue(code)
        except Exception:
            continue
        arr = arr[arr.element != "H"]
        if arr.array_length() == 0 or arr.array_length() > 120:
            continue
        arr.set_annotation("n_heavy_neighbors", count_heavy_neighbors(arr))
        perms = generate_automorphisms_from_atom_array_with_networkx(
            arr, node_features=["element", "n_heavy_neighbors"], normalize_bond_orders=True)
        cases.append({
            "code": code,
            "element": [str(e).upper() for e in arr.element],
            "charge": [int(c) for c in arr.charge],
            "bonds": [[int(i), int(j), int(t)] for i, j, t in arr.bonds.as_array()],
            "perms": np.asarray(perms).tolist(),
        })
    r = subprocess.run(["node", REPO / "tools/rfd4p-automorphisms.mjs"], input=json.dumps(cases),
                       capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return 1
    ours = json.loads(r.stdout)
    bad, truncated = [], []
    for case, got in zip(cases, ours, strict=True):
        want = {tuple(p) for p in case["perms"]}
        have = {tuple(p) for p in got}
        if tuple(got[0]) != tuple(range(len(case["element"]))):
            bad.append(f"{case['code']}: the identity is not first")
        elif len(want) == len(have) == MAX_AUTOMORPHS:
            # Both stop at the cap; WHICH thousand is networkx's VF2 visiting order (down to
            # Python's set iteration), which is not reproduced. It can matter only on an exact
            # tie between labellings of a component this symmetric.
            truncated.append(case["code"])
        elif want != have:
            bad.append(f"{case['code']}: atomworks {len(want)}, here {len(have)}")
    for b in bad[:20]:
        print(f"  {b}")
    print(f"{len(cases)} components, {sum(len(c['perms']) > 1 for c in cases)} with a symmetry, "
          f"{len(bad)} differing; {len(truncated)} past {MAX_AUTOMORPHS} automorphisms, where both "
          f"keep {MAX_AUTOMORPHS} but not the same ones ({', '.join(truncated)})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
