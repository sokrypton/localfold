#!/usr/bin/env python3
"""A training example for cuda/grad/train-of3 from a deposited structure: the AF3 job its chains are, the truth its
atoms are, and the featurised input the trainer reads.

    ~/venv/bin/python cuda/grad/make_of3_example.py <entry.cif> <out dir> --family=openbind0 [--a3m-dir=<dir>] [--search]

Writes <out dir>/job.json, <out dir>/truth.txt and <out dir>/input/ (cuda/featurise/af3-featurise's output), and
<out dir>/example.txt - the trainer's lines, `input truth [resolution=X] perms=... set=... preferred=... weight=...`: one
a chain and one an interface (chains within 5 A), weighted as AF3 SI 2.5.1 weighs them (--whole: one line, none preferred).

The job: the first model's chains by label_asym_id, waters and hydrogens left out. A polymer is its entity's FULL
sequence (_entity_poly_seq, so an unresolved residue is in the job and masked in the truth, as OpenFold3 trains), a
modified residue its parent's letter with the component as a modification; every other chain a ligand of its
components in order. The chain ids in the job are A, B, ... in that order.

The truth: one line an atom, `chain residue name x y z` - chain the job's chain index from 1 (the batch's asymId),
residue the label_seq_id (a polymer) or the component's place in its ligand chain from 1 - the first altloc, and only
atoms with occupancy above zero (OpenFold3's resolved mask).

Without --a3m-dir or --search every chain folds from its own sequence (an empty alignment); --a3m-dir=<dir> reads
<dir>/<label_asym_id>.a3m for a protein chain where one exists; --search lets the featuriser run its MMseqs2 search (alignments and template hits).
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def align_pairs(q, t):
    """a semi-global affine-gap alignment of q onto t (match +2, mismatch -1, gap open -5, extend -1, end gaps free):
    the aligned (q index, t index) pairs - a template's residue mapping"""
    n, m = len(q), len(t)
    NEG = -1e9
    M = [[NEG] * (m + 1) for _ in range(n + 1)]; X = [[NEG] * (m + 1) for _ in range(n + 1)]; Y = [[NEG] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): M[i][0] = 0
    for j in range(m + 1): M[0][j] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            M[i][j] = max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]) + (2 if q[i - 1] == t[j - 1] else -1)
            X[i][j] = max(M[i - 1][j] - 5, X[i - 1][j] - 1)
            Y[i][j] = max(M[i][j - 1] - 5, Y[i][j - 1] - 1)
    # (free trailing gaps: start the traceback at the best cell of the last row or column)
    best, bi, bj = NEG, n, m
    for j in range(m + 1):
        v = max(M[n][j], X[n][j], Y[n][j])
        if v > best: best, bi, bj = v, n, j
    for i in range(n + 1):
        v = max(M[i][m], X[i][m], Y[i][m])
        if v > best: best, bi, bj = v, i, m
    i, j = bi, bj
    state = max((M[i][j], 0), (X[i][j], 1), (Y[i][j], 2))[1]
    pairs = []
    while i > 0 and j > 0:
        if state == 0:
            pairs.append((i - 1, j - 1))
            prev = M[i][j] - (2 if q[i - 1] == t[j - 1] else -1)
            state = 0 if M[i - 1][j - 1] == prev else 1 if X[i - 1][j - 1] == prev else 2
            i -= 1; j -= 1
        elif state == 1:
            state = 0 if X[i][j] == M[i - 1][j] - 5 else 1
            i -= 1
        else:
            state = 0 if Y[i][j] == M[i][j - 1] - 5 else 2
            j -= 1
    return pairs[::-1]


def rcsb_templates(seq, before, exclude, raw, rows=4):
    """AF3 SI 2.4's training templates for one protein chain: RCSB's sequence search (MMseqs2, e-value 0.1) over
    entries released before `before` (the target's release less 60 days), the target itself excluded; each hit as
    the job's own template - its chain alone as an mmCIF, and the alignment's residue mapping"""
    import urllib.request, gzip
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "sequence", "parameters": {"evalue_cutoff": 0.1, "identity_cutoff": 0.0,
                                                                     "sequence_type": "protein", "value": seq}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_accession_info.initial_release_date",
                                                               "operator": "less", "value": before}}]},
        "return_type": "polymer_instance",
        "request_options": {"paginate": {"start": 0, "rows": 4 * rows}, "scoring_strategy": "sequence"}}
    req = urllib.request.Request("https://search.rcsb.org/rcsbsearch/v2/query", data=json.dumps(q).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r: body = r.read()
    except Exception as e:
        print(f"template search: {e} - no templates for this chain"); return []
    hits = [h["identifier"] for h in (json.loads(body).get("result_set") or []) if body] if body else []
    out = []
    for ident in hits:
        pdb, asym = ident.split(".")
        if pdb.upper() == exclude.upper(): continue
        path = os.path.join(raw, f"{pdb.lower()}.cif")
        if not os.path.exists(path):
            with urllib.request.urlopen(f"https://files.rcsb.org/download/{pdb.upper()}.cif.gz", timeout=120) as r:
                open(path, "wb").write(gzip.decompress(r.read()))
        st = gemmi.read_structure(path); st.setup_entities(); st.remove_hydrogens(); st.remove_waters()
        st.remove_ligands_and_waters()
        sub = next((sc for ch in st[0] for sc in ch.subchains() if sc.subchain_id() == asym), None)
        ent = st.get_entity_of(sub) if sub is not None else None
        if sub is None or ent is None or ent.polymer_type != gemmi.PolymerType.PeptideL: continue
        tseq = gemmi.one_letter_code([m.split(",")[0] for m in ent.full_sequence]).upper()
        pairs = align_pairs(seq, tseq)
        if len(pairs) < 10: continue
        # the one chain alone, its entity's sequence kept (the mapping indexes it)
        keep = gemmi.Structure(); keep.cell = st.cell; keep.spacegroup_hm = st.spacegroup_hm
        model = gemmi.Model("1"); chain = gemmi.Chain("A")
        for res in sub: chain.add_residue(res)
        model.add_chain(chain); keep.add_model(model)
        e2 = gemmi.Entity("1"); e2.entity_type = gemmi.EntityType.Polymer; e2.polymer_type = ent.polymer_type
        e2.full_sequence = list(ent.full_sequence); e2.subchains = ["A"]; keep.entities.append(e2)
        for res in keep[0]["A"]: res.subchain = "A"; res.entity_id = "1"
        keep.setup_entities()
        out.append({"mmcif": keep.make_mmcif_document().as_string(),
                    "queryIndices": [p_[0] for p_ in pairs], "templateIndices": [p_[1] for p_ in pairs]})
        if len(out) == rows: break
    return out


def etkdg_conformers(block, k=8, seed=0):
    """OpenFold3's fresh conformers (core/data/primitives/structure/conformer.py compute_conformer): RDKit ETKDGv3
    with hydrogens, random seeds, the hydrogens then removed - k runs of the component, each heavy atom by name. []
    when the embedding fails (the featuriser's conformer then stands, as OpenFold3's stored fallback does)."""
    from rdkit.Chem import AllChem
    atoms = block.find("_chem_comp_atom.", ["atom_id", "type_symbol", "charge"])
    bonds = block.find("_chem_comp_bond.", ["atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag"])
    mol = Chem.RWMol(); idx, names = {}, []
    for row in atoms:
        el = row[1].strip().capitalize()
        if el == "X": return []
        at = Chem.Atom(el if el != "D" else "H")
        try: at.SetFormalCharge(int(row[2]))
        except ValueError: pass
        at.SetNoImplicit(True)
        idx[row[0].strip('"')] = mol.AddAtom(at); names.append(row[0].strip('"'))
    order = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE, "TRIP": Chem.BondType.TRIPLE}
    for row in bonds if bonds else []:
        a, b = row[0].strip('"'), row[1].strip('"')
        if a in idx and b in idx: mol.AddBond(idx[a], idx[b], order.get(row[2].upper(), Chem.BondType.SINGLE))
    m = mol.GetMol()
    try: Chem.SanitizeMol(m)
    except Exception: return []
    params = AllChem.ETKDGv3(); params.randomSeed = seed
    ids = AllChem.EmbedMultipleConfs(m, numConfs=k, params=params)
    out = []
    for cid in ids:
        conf = m.GetConformer(cid)
        out.append([(names[i], conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms()) if m.GetAtomWithIdx(i).GetSymbol() != "H"])
    return out


# AF3 SI table 9's crystallisation aids, removed from training structures
CRYSTALLISATION_AIDS = {"SO4", "GOL", "EDO", "PO4", "ACT", "PEG", "DMS", "TRS", "PGE", "PG4", "FMT", "EPE", "MPD", "MES", "CD", "IOD"}


def af3_filter(subs):
    """AF3 SI 2.5.4 (OpenFold3's structure cleanup): crystallisation aids out; a polymer chain with fewer than 4
    resolved residues out, and one whose residues are all unknown; of two chains that clash (more than 30% of either's
    atoms within 1.7 A of the other's), the one with the larger clashing fraction out (ties: the larger chain index)."""
    import numpy as np
    keep = []
    for sub, ent in subs:
        if ent.entity_type != gemmi.EntityType.Polymer:
            if all(res.name in CRYSTALLISATION_AIDS for res in sub): continue
        else:
            resolved = [res for res in sub if res.label_seq is not None and len(res) > 0]
            if len(resolved) < 4: continue
            if all(res.name in ("UNK", "N", "DN") for res in resolved): continue
        keep.append((sub, ent))
    xyz = [np.array([[at.pos.x, at.pos.y, at.pos.z] for res in sub for at in res]) for sub, _ in keep]
    drop = set()
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            if i in drop or j in drop or not len(xyz[i]) or not len(xyz[j]): continue
            d = np.sqrt(((xyz[i][:, None, :] - xyz[j][None, :, :]) ** 2).sum(-1))
            close = d < 1.7
            fi, fj = close.any(1).mean(), close.any(0).mean()
            if fi > 0.3 or fj > 0.3:
                drop.add(i if fi > fj else j if fj > fi else max(i, j))
    return [x for k, x in enumerate(keep) if k not in drop]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cif"); ap.add_argument("out")
    ap.add_argument("--family", default="openbind0")
    ap.add_argument("--a3m-dir", default="")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--rcsb-templates", action="store_true",
                    help="the protein chains' templates from RCSB's sequence search, released at least 60 days before the "
                         "entry and never the entry itself (AF3 SI 2.4), with --search for the alignments")
    ap.add_argument("--ccd", default=os.path.join(REPO, "ccd", "components.cif"))
    ap.add_argument("--set", default="pdb", choices=["pdb", "monomer", "rna", "disordered"],
                    help="OpenFold3's dataset the entry belongs to: its loss weights and crops (dataset_configs.py)")
    ap.add_argument("--n-clust", type=float, default=1.0,
                    help="the entry's cluster size, AF3 SI 2.5.1's N_clust (OpenFold3 counts the datapoints in a 40%%-identity cluster)")
    ap.add_argument("--whole", action="store_true", help="one line for the entry, no preferred chain or interface")
    ap.add_argument("--af3-filters", action="store_true",
                    help="AF3 SI 2.5.4's training-set filters: crystallisation aids, polymer chains of under 4 resolved "
                         "residues, all-unknown chains and clashing chains removed (as OpenFold3's cleanup does)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    st = gemmi.read_structure(a.cif)
    # (--rcsb-templates: the cutoff, the entry's first release less 60 days)
    tmpl_before, entry_id = None, os.path.basename(a.cif).split(".")[0]
    if a.rcsb_templates:
        import datetime
        dates = gemmi.cif.read(a.cif).sole_block().find_values("_pdbx_audit_revision_history.revision_date")
        if not dates: sys.exit(f"{a.cif}: no release date, so no template cutoff")
        tmpl_before = (datetime.date.fromisoformat(min(dates)) - datetime.timedelta(days=60)).isoformat()
        tmpl_raw = os.path.join(a.out, "templates-raw"); os.makedirs(tmpl_raw, exist_ok=True)
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
    kinds, heavy = [], []    # each job chain's molecule type, and its resolved heavy atoms (for the interfaces)
    components = []          # (chain index, residue index, component code): what the permutations are written for
    ids = chain_ids()
    # label_asym_id chains (a gemmi subchain is one), polymers first and then the rest, each in file order: the
    # featuriser orders a job's chains that way, and the truth's chain index is the batch's asymId
    subs = [(sub, st.get_entity_of(sub)) for chain in model for sub in chain.subchains()]
    subs = [x for x in subs if x[1] is not None]
    subs.sort(key=lambda x: x[1].entity_type != gemmi.EntityType.Polymer)
    if a.af3_filters:
        subs = af3_filter(subs)
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
                    if tmpl_before:
                        entry["templates"] = rcsb_templates(entry["sequence"], tmpl_before, entry_id, tmpl_raw)
                        print(f"chain {jid}: {len(entry['templates'])} templates released before {tmpl_before}")
                elif kind == "rna" and not a.search:
                    entry["unpairedMsa"] = ""
                sequences.append({kind: entry})
                kinds.append("nuc" if kind in ("dna", "rna") else "prot"); heavy.append([])
                for pos in sorted(mons): components.append((index, pos, mons[pos]))
                for res in sub:
                    if res.label_seq is None: continue
                    for atom in res:
                        if atom.altloc not in ("\0", "A") or atom.occ <= 0: continue
                        p = atom.pos
                        truth.append(f"{index} {res.label_seq} {atom.name} {p.x:.3f} {p.y:.3f} {p.z:.3f}")
                        heavy[-1].append((p.x, p.y, p.z))
            else:
                codes = [res.name for res in sub]
                sequences.append({"ligand": {"id": jid, "ccdCodes": codes}})
                kinds.append("ligand"); heavy.append([])
                for k, res in enumerate(sub, start=1):
                    components.append((index, k, res.name))
                    for atom in res:
                        if atom.altloc not in ("\0", "A") or atom.occ <= 0: continue
                        p = atom.pos
                        truth.append(f"{index} {k} {atom.name} {p.x:.3f} {p.y:.3f} {p.z:.3f}")
                        heavy[-1].append((p.x, p.y, p.z))
    job = {"name": os.path.basename(a.cif).split(".")[0], "modelSeeds": [1], "sequences": sequences,
           "dialect": "alphafold3", "version": 2}
    with open(os.path.join(a.out, "job.json"), "w") as f: json.dump(job, f, indent=1)
    with open(os.path.join(a.out, "truth.txt"), "w") as f: f.write("\n".join(truth) + "\n")
    cmd = [os.path.join(REPO, "cuda", "featurise", "af3-featurise"), os.path.join(a.out, "input"), "--no-weights",
           f"--family={a.family}", f"--job={os.path.join(a.out, 'job.json')}"]
    if os.path.exists(a.ccd): cmd.append(f"--ccd={a.ccd}")
    # (the featuriser's ColabFold MMseqs2 search: the protein chains' alignments, and their template hits - every
    # chain's best four, which train-of3 then draws from per step; an entry with no protein chain has nothing to search)
    if a.search and "prot" in kinds: cmd += ["--search"] + ([] if a.rcsb_templates else ["--search-templates"])
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    if r.returncode: sys.exit(r.returncode)
    # the symmetric-atom permutations of every component that has any (AF3 SI 4.2: the truth's atoms renamed to the
    # prediction's best match), `chain residue identity-names | permuted-names | ...`
    blocks = ccd_blocks(a.ccd, sorted({c for _, _, c in components})) if os.path.exists(a.ccd) else {}
    perm_of = {code: symmetry_permutations(b) for code, b in blocks.items()}
    # the conformer pool (OpenFold3 embeds a fresh ETKDGv3 conformer per featurisation; train-of3 draws one of these)
    import random
    with open(os.path.join(a.out, "conformers.txt"), "w") as f:
        for chain, res_i, code in components: f.write(f"component {chain} {res_i} {code}\n")
        for code, blk in blocks.items():
            for k, conf in enumerate(etkdg_conformers(blk, 8, random.randint(0, 10 ** 9))):
                f.write(f"conformer {code} {k} {len(conf)}\n")
                for nm, p in conf: f.write(f"{nm} {p.x:.3f} {p.y:.3f} {p.z:.3f}\n")
    with open(os.path.join(a.out, "perms.txt"), "w") as f:
        for chain, res, code in components:
            p = perm_of.get(code)
            if p: f.write(f"{chain} {res} " + " | ".join(",".join(x) for x in p) + "\n")
    # the resolution, which gates the confidence losses (OpenFold3: 0.1 to 4.0 A, none outside or without one)
    res = st.resolution if st.resolution and st.resolution > 0 else None
    # the datapoints (AF3 SI 2.5.1, OpenFold3's WeightedPDBDataset): every chain, and every pair of chains with heavy
    # atoms within 5 A, each weighted (w_type / N_clust) (3 n_prot + 3 n_nuc + 1 n_ligand) and preferred by the crop
    import numpy as np
    base = (f"{os.path.abspath(os.path.join(a.out, 'input'))} {os.path.abspath(os.path.join(a.out, 'truth.txt'))}"
            + (f" resolution={res:.2f}" if res else "") + f" perms={os.path.abspath(os.path.join(a.out, 'perms.txt'))} conformers={os.path.abspath(os.path.join(a.out, 'conformers.txt'))}"
            + f" set={a.set}")
    a_of = {"prot": 3.0, "nuc": 3.0, "ligand": 1.0}
    lines = []
    if a.whole:
        lines.append(base)
    else:
        for i, k in enumerate(kinds):
            if heavy[i]: lines.append(f"{base} preferred={i + 1} weight={0.5 / a.n_clust * a_of[k]:.4g}")
        xyz = [np.array(h) if h else np.zeros((0, 3)) for h in heavy]
        for i in range(len(kinds)):
            for j in range(i + 1, len(kinds)):
                if not len(xyz[i]) or not len(xyz[j]): continue
                d = np.sqrt(((xyz[i][:, None, :] - xyz[j][None, :, :]) ** 2).sum(-1)).min()
                if d < 5.0:
                    lines.append(f"{base} preferred={i + 1},{j + 1} weight={1.0 / a.n_clust * (a_of[kinds[i]] + a_of[kinds[j]]):.4g}")
    with open(os.path.join(a.out, "example.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(sequences)} chains, {len(truth)} resolved atoms -> {a.out}")


if __name__ == "__main__":
    main()
