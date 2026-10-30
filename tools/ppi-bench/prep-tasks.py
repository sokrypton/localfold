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
    # "F-<pdb>": the paper's "dna_fixed" - the same duplex from *_structure_context, its coordinates given: every heavy
    # atom as [chain index, residue (1-based within the chain), atom name, x, y, z], for the design folds to hold
    f = os.path.join(d, f"{pdb}_structure_context.cif.zst")
    c, rows = table(f, "_atom_site.")
    order, atoms, chains = [], [], {}
    for r in rows:
        comp = r[c["label_comp_id"]]
        if comp not in DNA or r[c["type_symbol"]] == "H": continue
        asym = r[c["label_asym_id"]]
        if asym not in order: order.append(asym)
        seq = int(r[c["label_seq_id"]])
        chains.setdefault(asym, {})[seq] = DNA[comp]
        atoms.append([order.index(asym), seq, r[c["label_atom_id"]],
                      float(r[c["Cartn_x"]]), float(r[c["Cartn_y"]]), float(r[c["Cartn_z"]])])
    first = {a: min(chains[a]) for a in order}
    tasks["F-" + pdb] = {"task": "dna", "binder": binder_range(c, rows),
                         "chains": [{"kind": "dna", "sequence": "".join(chains[a][k] for k in sorted(chains[a]))} for a in order],
                         "dnaAtoms": [[i, seq - first[order[i]] + 1, n, x, y, z] for i, seq, n, x, y, z in atoms]}
    assert tasks["F-" + pdb]["chains"] == tasks[pdb]["chains"], pdb

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
        xyz = [[float(a[c["Cartn_x"]]), float(a[c["Cartn_y"]]), float(a[c["Cartn_z"]])] for a in atoms]
        metals = {a.GetSymbol() for a in m.GetAtoms()} & {"Fe", "Co", "Cu", "Zn", "Mn", "Ni"}
        if metals:
            codes = CCD[smiles]
            # (the file's conformer by atom NAME: the harness gives each name to the code's chain that builds it)
            named = [[a[c["label_atom_id"]], *p] for a, p in zip(atoms, xyz)]
            chains.extend({"kind": "ligand", "ccd": code, "atoms": len(atoms), "conformerNames": named} for code in codes)
            continue
        # `conformer`: the file's coordinates in the SMILES's own atom order (RDKit's _smilesAtomOutputOrder), which
        # is the order a SMILES ligand's atoms become tokens - each with its element, which the harness checks
        out = json.loads(m.GetProp("_smilesAtomOutputOrder").replace(",]", "]"))
        chains.append({"kind": "ligand", "smiles": smiles, "atoms": len(atoms),
                       "conformer": [[m.GetAtomWithIdx(k).GetSymbol(), *xyz[k]] for k in out]})
    tasks[key] = {"task": "sm", "binder": binder_range(c, rows), "chains": chains}

# ame: data/benchmarks/mcsa/mcsa_37_fixed_ligand (the paper's "mcsa_fixed"), unindexed catalytic motifs: each motif
#      residue's type and its COORDINATE-CONDITIONED atoms (the non-NaN rows of _condition_coordinate_1_atom - the
#      tips; the backbone atoms recovered from the PDB entry are sequence-conditioned only), each ligand chain by CCD
#      code with its conditioned atoms, and the scaffold length. Every conditioned atom is one absolute group, so the
#      motif RMSD is one superposition over all of them (configs/metrics/refolding_metrics, refolded_motif_allatom_rmsd)
#   🔴 inter-ligand bonds (M0110's FAD-PER-DAL) are not carried: each ligand chain is folded as its own molecule
#   ...and "U-<key>": the same 37 from mcsa_37 (the paper's "mcsa_unfixed"), where the ligands' atoms are NOT
#   coordinate-conditioned - only the motif's tips are, so only those are inpainted
for d, prefix in ((os.path.join(SRC, "data/benchmarks/mcsa/mcsa_37_fixed_ligand"), ""), (os.path.join(SRC, "data/benchmarks/mcsa/mcsa_37"), "U-")):
  for f in sorted(os.listdir(d)):
    key = prefix + f.split(".")[0]
    c, rows = table(os.path.join(d, f), "_atom_site.")
    cc, crow = table(os.path.join(d, f), "_condition_coordinate_1_atom.")
    conditioned = {int(r[0]) for r in crow if r[1] != "nan"}
    motif, ligands = {}, {}
    for r in rows:
        if r[c["type_symbol"]] in ("X", "H"): continue
        on = int(r[c["atom_array_index"]]) in conditioned
        xyz = [float(r[c["Cartn_x"]]), float(r[c["Cartn_y"]]), float(r[c["Cartn_z"]])]
        if r[c["group_PDB"]] == "ATOM":
            m = motif.setdefault(int(r[c["label_seq_id"]]), {"res": r[c["label_comp_id"]], "atoms": []})
            if on: m["atoms"].append([r[c["label_atom_id"]], *xyz])
        else:
            lig = ligands.setdefault(r[c["label_asym_id"]], {"comps": [], "atoms": []})
            if r[c["label_comp_id"]] not in lig["comps"]: lig["comps"].append(r[c["label_comp_id"]])
            if on: lig["atoms"].append([r[c["label_atom_id"]], *xyz])
    assert all(len(l["comps"]) == 1 for l in ligands.values()), (key, ligands)
    tasks[key] = {"task": "ame", "binder": binder_range(c, rows),
                  "chains": [{"kind": "ligand", "ccd": l["comps"][0], "atoms": len(l["atoms"]), "conditioned": l["atoms"]}
                             for _, l in sorted(ligands.items())],
                  "motif": [motif[k] for k in sorted(motif)]}

json.dump(tasks, open(os.path.join(HERE, "tasks.json"), "w"))
for k, t in tasks.items():
    print(k, t["task"], t["binder"], [ch.get("sequence") or ch.get("smiles") or ch["ccd"] for ch in t["chains"]],
          *([[m["res"], len(m["atoms"])] for m in t["motif"]] if "motif" in t else []))
