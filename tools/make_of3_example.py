#!/usr/bin/env python3
"""A training example for cuda/grad/train-of3 from a deposited structure: the AF3 job its chains are, the truth its
atoms are, and the featurised input the trainer reads.

    ~/venv/bin/python tools/make_of3_example.py <entry.cif> <out dir> --family=openbind0 [--a3m-dir=<dir>] [--search]

Writes <out dir>/job.json, <out dir>/truth.txt and <out dir>/input/ (cuda/featurise/af3-featurise's output), and
<out dir>/example.txt - the trainer's one line, `input truth [resolution=X]`.

The job: the first model's chains by label_asym_id, waters and hydrogens left out. A polymer is its entity's FULL
sequence (_entity_poly_seq, so an unresolved residue is in the job and masked in the truth, as OpenFold3 trains), a
modified residue its parent's letter with the component as a modification; every other chain a ligand of its
components in order. The chain ids in the job are A, B, ... in that order.

The truth: one line an atom, `chain residue name x y z` - chain the job's chain index from 1 (the batch's asymId),
residue the label_seq_id (a polymer) or the component's place in its ligand chain from 1 - the first altloc, and only
atoms with occupancy above zero (OpenFold3's resolved mask).

Without --a3m-dir or --search every chain folds from its own sequence (an empty alignment); --a3m-dir=<dir> reads
<dir>/<label_asym_id>.a3m for a protein chain where one exists; --search lets the featuriser run its MMseqs2 search.
"""
import argparse
import json
import os
import re
import string
import subprocess
import sys

import gemmi
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def chain_ids():
    for c in string.ascii_uppercase:
        yield c
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            yield a + b


def ccd_blocks(path, codes):
    """the CCD's data blocks for `codes` (a scan for each `data_X` line rather than a parse of 50,000 components)"""
    text = open(path).read()
    out = {}
    for code in codes:
        m = re.search(r"^data_" + re.escape(code) + r"\s*$", text, re.M)
        if not m: continue
        end = text.find("\ndata_", m.end())
        out[code] = gemmi.cif.read_string(text[m.start(): end if end >= 0 else len(text)]).sole_block()
    return out


def symmetry_permutations(block, max_permutations=1000):
    """OpenFold3's conformer permutations (core/data/primitives/structure/conformer.py get_cropped_permutations): RDKit's
    GetSubstructMatches of the component against itself, uniquify off, chirality ignored, at most 1000 - over its heavy
    atoms as the CCD gives them (elements, charges, bond orders, aromaticity). Each permutation as atom names: perm[i]
    the name atom i maps to. None when only the identity."""
    atoms = block.find("_chem_comp_atom.", ["atom_id", "type_symbol", "charge", "pdbx_aromatic_flag"])
    bonds = block.find("_chem_comp_bond.", ["atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag"])
    names, idx = [], {}
    mol = Chem.RWMol()
    for row in atoms:
        el = row[1].strip().capitalize()
        if el in ("H", "D"): continue
        a = Chem.Atom(el if el != "X" else "C")
        try: a.SetFormalCharge(int(row[2]))
        except ValueError: pass
        a.SetIsAromatic(row[3] == "Y")
        idx[row[0].strip('"')] = mol.AddAtom(a); names.append(row[0].strip('"'))
    order = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE, "TRIP": Chem.BondType.TRIPLE}
    for row in bonds if bonds else []:
        a, b = row[0].strip('"'), row[1].strip('"')
        if a not in idx or b not in idx: continue
        bt = Chem.BondType.AROMATIC if row[3] == "Y" else order.get(row[2].upper(), Chem.BondType.SINGLE)
        mol.AddBond(idx[a], idx[b], bt)
        if bt == Chem.BondType.AROMATIC: mol.GetBondBetweenAtoms(idx[a], idx[b]).SetIsAromatic(True)
    m = mol.GetMol()
    try: m.UpdatePropertyCache(strict=False)
    except Exception: pass
    perms = m.GetSubstructMatches(m, uniquify=False, maxMatches=max_permutations, useChirality=False)
    perms = [p for p in perms if list(p) != list(range(len(names)))]
    if not perms: return None
    return [names] + [[names[j] for j in p] for p in perms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cif"); ap.add_argument("out")
    ap.add_argument("--family", default="openbind0")
    ap.add_argument("--a3m-dir", default="")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--ccd", default=os.path.join(REPO, "ccd", "components.cif"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    st = gemmi.read_structure(a.cif)
    st.setup_entities()
    st.remove_hydrogens()
    st.remove_waters()
    model = st[0]
    doc = gemmi.cif.read(a.cif)
    block = doc.sole_block()
    # each polymer entity's full sequence (_entity_poly_seq), the first of any microheterogeneity
    seq_of = {}
    eps = block.find("_entity_poly_seq.", ["entity_id", "num", "mon_id"])
    for row in eps:
        ent, num, mon = row[0], int(row[1]), row[2]
        seq_of.setdefault(ent, {}).setdefault(num, mon)
    kind_of = {e.name: e.polymer_type for e in st.entities}

    sequences, truth = [], []
    components = []          # (chain index, residue index, component code): what the permutations are written for
    ids = chain_ids()
    # label_asym_id chains (a gemmi subchain is one), polymers first and then the rest, each in file order: the
    # featuriser orders a job's chains that way, and the truth's chain index is the batch's asymId
    subs = [(sub, st.get_entity_of(sub)) for chain in model for sub in chain.subchains()]
    subs = [x for x in subs if x[1] is not None]
    subs.sort(key=lambda x: x[1].entity_type != gemmi.EntityType.Polymer)
    for sub, ent in subs:
            label = sub.subchain_id()
            jid = next(ids)
            index = len(sequences) + 1
            if ent.entity_type == gemmi.EntityType.Polymer:
                pt = ent.polymer_type
                kind = ("protein" if pt in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD)
                        else "dna" if pt == gemmi.PolymerType.Dna else "rna" if pt == gemmi.PolymerType.Rna else None)
                if kind is None:
                    sys.exit(f"{label}: polymer type {pt} is not one AF3 folds")
                mons = seq_of[ent.name]
                letters, mods = [], []
                for pos in sorted(mons):
                    mon = mons[pos]
                    info = gemmi.find_tabulated_residue(mon)
                    if info is not None and info.is_standard():
                        letters.append(info.one_letter_code.upper())
                        continue
                    parent = info.one_letter_code.upper() if info is not None and info.one_letter_code.strip() not in ("", "?") else ("X" if kind == "protein" else "N")
                    letters.append(parent)
                    mods.append({"ptmType": mon, "ptmPosition": pos} if kind == "protein" else {"modificationType": mon, "basePosition": pos})
                entry = {"id": jid, "sequence": "".join(letters)}
                if mods: entry["modifications"] = mods
                if kind == "protein":
                    a3m = os.path.join(a.a3m_dir, f"{label}.a3m") if a.a3m_dir else ""
                    if a3m and os.path.exists(a3m):
                        entry["unpairedMsa"] = open(a3m).read(); entry["pairedMsa"] = ""; entry["templates"] = []
                    elif not a.search:
                        entry["unpairedMsa"] = ""; entry["pairedMsa"] = ""; entry["templates"] = []
                elif kind == "rna" and not a.search:
                    entry["unpairedMsa"] = ""
                sequences.append({kind: entry})
                for pos in sorted(mons): components.append((index, pos, mons[pos]))
                for res in sub:
                    if res.label_seq is None: continue
                    for atom in res:
                        if atom.altloc not in ("\0", "A") or atom.occ <= 0: continue
                        p = atom.pos
                        truth.append(f"{index} {res.label_seq} {atom.name} {p.x:.3f} {p.y:.3f} {p.z:.3f}")
            else:
                codes = [res.name for res in sub]
                sequences.append({"ligand": {"id": jid, "ccdCodes": codes}})
                for k, res in enumerate(sub, start=1):
                    components.append((index, k, res.name))
                    for atom in res:
                        if atom.altloc not in ("\0", "A") or atom.occ <= 0: continue
                        p = atom.pos
                        truth.append(f"{index} {k} {atom.name} {p.x:.3f} {p.y:.3f} {p.z:.3f}")
    job = {"name": os.path.basename(a.cif).split(".")[0], "modelSeeds": [1], "sequences": sequences,
           "dialect": "alphafold3", "version": 2}
    with open(os.path.join(a.out, "job.json"), "w") as f: json.dump(job, f, indent=1)
    with open(os.path.join(a.out, "truth.txt"), "w") as f: f.write("\n".join(truth) + "\n")
    cmd = [os.path.join(REPO, "cuda", "featurise", "af3-featurise"), os.path.join(a.out, "input"), "--no-weights",
           f"--family={a.family}", f"--job={os.path.join(a.out, 'job.json')}"]
    if os.path.exists(a.ccd): cmd.append(f"--ccd={a.ccd}")
    if a.search: cmd.append("--network")
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    if r.returncode: sys.exit(r.returncode)
    # the symmetric-atom permutations of every component that has any (AF3 SI 4.2: the truth's atoms renamed to the
    # prediction's best match), `chain residue identity-names | permuted-names | ...`
    blocks = ccd_blocks(a.ccd, sorted({c for _, _, c in components})) if os.path.exists(a.ccd) else {}
    perm_of = {code: symmetry_permutations(b) for code, b in blocks.items()}
    with open(os.path.join(a.out, "perms.txt"), "w") as f:
        for chain, res, code in components:
            p = perm_of.get(code)
            if p: f.write(f"{chain} {res} " + " | ".join(",".join(x) for x in p) + "\n")
    # the resolution, which gates the confidence losses (OpenFold3: 0.1 to 4.0 A, none outside or without one)
    res = st.resolution if st.resolution and st.resolution > 0 else None
    with open(os.path.join(a.out, "example.txt"), "w") as f:
        f.write(f"{os.path.abspath(os.path.join(a.out, 'input'))} {os.path.abspath(os.path.join(a.out, 'truth.txt'))}"
                + (f" resolution={res:.2f}" if res else "") + f" perms={os.path.abspath(os.path.join(a.out, 'perms.txt'))}\n")
    print(f"{len(sequences)} chains, {len(truth)} resolved atoms -> {a.out}")


if __name__ == "__main__":
    main()
