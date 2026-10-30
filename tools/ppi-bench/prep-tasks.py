# RFD4-Proteina's DNA- and small-molecule-binder benchmark targets (their repo, BSD-3) -> tasks.json
#   dna: data/benchmarks/dna_binder/dna_binder_denovo_val/<id>_sequence_context (the paper's "unfixed" DNA: the
#        duplex is given as sequence only and co-folded), binder 120-150
#   sm:  data/benchmarks/small_molecule_binder/disco_figure_2c, the 77 keys of configs/.../disco_figure_2c.yaml; each
#        ligand chain written as SMILES from the file's own atoms, bonds and formal charges (no CCD code exists for
#        most), binder over the file's own expansion range (100-155)
# usage: ~/.venv-rdkit/bin/python prep-tasks.py <RFD4-Proteina checkout>
import json, os, re, subprocess, sys
from rdkit import Chem
SRC = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
DNA = {"DA": "A", "DC": "C", "DG": "G", "DT": "T"}
ORDER = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE, "TRIP": Chem.BondType.TRIPLE,
         "AROM": Chem.BondType.AROMATIC}

def loops(path):
    """Every mmCIF loop as (columns, rows); quoted tokens kept whole."""
    text = subprocess.run(["zstd", "-dc", path], capture_output=True, text=True, check=True).stdout.split("\n")
    out, i = [], 0
    while i < len(text):
        if text[i].startswith("loop_"):
            i += 1; cols = []
            while text[i].startswith("_"):
                cols.append(text[i].strip()); i += 1
            rows = []
            while i < len(text) and text[i].strip() and not text[i].startswith(("loop_", "#", "_")):
                rows.append(re.findall(r'"[^"]*"|\S+', text[i])); i += 1
            out.append((cols, [[t.strip('"') for t in r] for r in rows]))
        else:
            i += 1
    return out

def table(path, prefix, optional=False):
    for cols, rows in loops(path):
        if cols[0].startswith(prefix):
            c = {k.split(".", 1)[1]: j for j, k in enumerate(cols)}
            return c, rows
    if optional: return {}, []  # 🔴 an ion's file has no bond table at all
    raise ValueError(f"{path}: no {prefix} loop")

def binder_range(c, rows):
    r = next(r for r in rows if r[c["label_atom_id"]] == "UNK" or r[c["type_symbol"]] == "X")
    return [int(r[c["annotation_expsegmin_1_residue"]]), int(r[c["annotation_expsegmax_1_residue"]])]

HEME = "C=CC1=C(C)C2=CC3=[N]4->[Fe]56<-[N]2=C1C=C1C(C)=C(C=C)C(=[N]->51)C=C1C(C)=C(CCC(=O)O)C(=[N]->61)C=C4C(CCC(=O)O)=C3C"
CCD = {HEME: ["HEM"], "[Cu+2]": ["CU"], "[Zn+2]": ["ZN"], "[Mn+2]": ["MN"], "[Ni+2]": ["NI"], "[Cu].[Cu]": ["CU", "CU"],
       "[SH]12[Fe]3[SH]4[Fe]1[SH]1[Fe]2[SH]3[Fe]41": ["SF4"], "[S]1[Fe][S][Fe]1": ["FES"],
       "O=C(O)CCC1=[O]->[Fe]<-[O]=C1O": ["FE2", "AKG"],
       "CC1=C2C(CCC(N)=O)C(C)(C)C3=CC4=[N]5->[Co+2]6([N]32)<-[N]2=C1C(C)(CCC(=O)NCC(C)OP(=O)(O)OC1C(CO)OC(n3cnc7cc(C)c(C)"
       "cc73)C1O)C(CC(N)=O)C2C1(C)[N]->6=C(C(C)=C5C(C)(CC(N)=O)C4CCC(N)=O)C(CCC(N)=O)C1(C)CC(N)=O": ["B12"]}
tasks = {}
d = os.path.join(SRC, "data/benchmarks/dna_binder/dna_binder_denovo_val")
for pdb in ("7m5w", "7n5u", "7rte"):
    f = os.path.join(d, f"{pdb}_sequence_context.cif.zst")
    c, rows = table(f, "_atom_site.")
    chains = {}
    for r in rows:
        comp = r[c["label_comp_id"]]
        if comp in DNA:
            chains.setdefault(r[c["label_asym_id"]], {})[int(r[c["label_seq_id"]])] = DNA[comp]
    tasks[pdb] = {"task": "dna", "binder": binder_range(c, rows),
                  "chains": [{"kind": "dna", "sequence": "".join(s[k] for k in sorted(s))} for s in chains.values()]}

d = os.path.join(SRC, "data/benchmarks/small_molecule_binder/disco_figure_2c")
cfg = open(os.path.join(SRC, "configs/datasets/val/small_molecule_binder/disco_figure_2c.yaml")).read()
keys = re.findall(r"^\s+- (\S+)\s*$", cfg.split("subset_to_keys:")[1].split("num_replicates")[0], re.M)
assert len(keys) == 77, len(keys)
for key in keys:
    f = os.path.join(d, key + ".cif.zst")
    c, rows = table(f, "_atom_site.")
    bc, bonds = table(f, "_chem_comp_bond.", optional=True)
    ligands = {}
    for r in rows:
        if r[c["type_symbol"]] == "X": continue
        ligands.setdefault(r[c["label_asym_id"]], []).append(r)
    chains = []
    for asym, atoms in ligands.items():
        comps = {a[c["label_comp_id"]] for a in atoms}
        assert len(comps) == 1, (key, comps)
        comp = comps.pop()
        mol = Chem.RWMol(); index = {}
        for a in atoms:
            atom = Chem.Atom(a[c["type_symbol"]].capitalize())
            q = a[c["pdbx_formal_charge"]]
            atom.SetFormalCharge(0 if q in ("?", ".") else int(q))
            index[a[c["label_atom_id"]]] = mol.AddAtom(atom)
        for b in bonds:
            if b[bc["comp_id"]] != comp or b[bc["atom_id_1"]] not in index or b[bc["atom_id_2"]] not in index: continue
            mol.AddBond(index[b[bc["atom_id_1"]]], index[b[bc["atom_id_2"]]], ORDER[b[bc["value_order"]]])
        m = mol.GetMol()
        Chem.SanitizeMol(m)
        smiles = Chem.MolToSmiles(m)
        # 🔴 A METAL COMPLEX IS GIVEN BY ITS CCD CODE: RDKit writes its coordination as dative bonds, which a SMILES
        # reader need not take, and the dictionary has every one of them
        metals = {a.GetSymbol() for a in m.GetAtoms()} & {"Fe", "Co", "Cu", "Zn", "Mn", "Ni"}
        if metals:
            codes = CCD[smiles]
            chains.extend({"kind": "ligand", "ccd": code, "atoms": len(atoms)} for code in codes)
            continue
        chains.append({"kind": "ligand", "smiles": smiles, "atoms": len(atoms)})
    tasks[key] = {"task": "sm", "binder": binder_range(c, rows), "chains": chains}

json.dump(tasks, open(os.path.join(HERE, "tasks.json"), "w"))
for k, t in tasks.items():
    print(k, t["task"], t["binder"], [ch.get("sequence") or ch.get("smiles") or ch["ccd"] for ch in t["chains"]])
