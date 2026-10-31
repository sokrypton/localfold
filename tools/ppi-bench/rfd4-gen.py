# rfd4-gen.py: RFD4-Proteina's OWN designs for a task, from the benchmark's own annotated inputs, through the native
# port (cuda/rfd4p, "high" recycling), written as rows `task-bench.mjs --rescore` judges exactly as it judges ours.
#
#   python3 rfd4-gen.py <input dir> <name prefix> <designs a target> <work dir> <out.jsonl> [<targets.jsonl>]
#
# <targets.jsonl>: a run of ours whose targets to take (the rows' `target`, prefix included), so both sides cover the
# same set. A design PDB is rewritten into our refold's layout, because the verdict pairs atoms by chain:residue:name:
#   - the binder (the one protein chain) becomes chain A, numbered from 1;
#   - every other chain takes the letter of its entry in tasks.json (B, C, ...), numbered from 1 - a DNA strand matched
#     by its sequence, a CCD ligand by its code, a SMILES ligand by its element order against the task's conformer;
#   - a SMILES ligand's atoms are renamed element + per-element counter from 1 (N1 C1 N2 ...), our featuriser's names;
#   - hydrogens are dropped.
# 🔴 Anything that does not match the task raises: a design paired against the wrong chain would judge nothing.
import json, os, subprocess, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TASKS = json.load(open(os.path.join(HERE, "tasks.json")))
AA = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
      'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
NA = {'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T'}


def element(line):
    e = line[76:78].strip()
    return e if e else line[12:16].strip()[0]


RADII = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05, 'P': 1.07, 'F': 0.57, 'CL': 1.02, 'BR': 1.2, 'I': 1.39,
         'B': 0.84, 'SE': 1.2, 'SI': 1.11, 'FE': 1.32, 'CU': 1.32, 'ZN': 1.22, 'MN': 1.39, 'NI': 1.24, 'CO': 1.26, 'MG': 1.41}


def graph(atoms, slack=0.45, split_metals=False):
    """Bonds from distances (covalent radii + `slack` A) over [(element, xyz)]; `split_metals` drops a metal's."""
    n = len(atoms)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            (a, p), (b, q) = atoms[i], atoms[j]
            if sum((x - y) ** 2 for x, y in zip(p, q)) ** 0.5 < RADII[a.upper()] + RADII[b.upper()] + slack:
                adj[i].add(j); adj[j].add(i)
    if split_metals:
        metal = [a.upper() in METALS for a, _ in atoms]
        adj = [{j for j in adj[i] if not (metal[i] or metal[j])} for i in range(n)]
    colour = [a.upper() for a, _ in atoms]
    for _ in range(6):
        colour = [f"{colour[i]}|{','.join(sorted(colour[j] for j in adj[i]))}" for i in range(n)]
        names = {c: str(k) for k, c in enumerate(sorted(set(colour)))}
        colour = [names[c] for c in colour]
    return adj, colour


def isomorphism(design, reference):
    """design atom index -> reference atom index, over element-labelled bond graphs (backtracking, colour-pruned)."""
    if sorted(a.upper() for a, _ in design) != sorted(a.upper() for a, _ in reference):
        return None
    # a design's ligand is a generated pose and a conformer an idealised one: a bond stretched past the distance rule,
    # or a contact inside it, and the two graphs differ by an edge. So both sides are read at a few slacks, and with a
    # metal's coordination dropped (a metal is then placed by its element alone)
    for metals in (False, True):
        for ref_slack in (0.45, 0.35):
            ra, rc = graph(reference, ref_slack, metals)
            for slack in (0.45, 0.55, 0.65, 0.35, 0.75):
                da, dc = graph(design, slack, metals)
                if sum(map(len, da)) == sum(map(len, ra)):
                    mapped = _isomorphism(design, reference, da, dc, ra, rc)
                    if mapped is not None:
                        return mapped
    return None


def _isomorphism(design, reference, da, dc, ra, rc):
    order, seen = [], set()
    for start in range(len(design)):
        stack = [start]
        while stack:
            i = stack.pop(0)
            if i in seen: continue
            seen.add(i); order.append(i); stack.extend(sorted(da[i] - seen))
    rmap, used = {}, set()
    def go(k):
        if k == len(order): return True
        i = order[k]
        for j in range(len(reference)):
            if j in used or rc[j] != dc[i] or len(ra[j]) != len(da[i]): continue
            if any((m in rmap) and (rmap[m] not in ra[j]) for m in da[i]): continue
            rmap[i] = j; used.add(j)
            if go(k + 1): return True
            del rmap[i]; used.discard(j)
        return False
    return dict(rmap) if go(0) else None


def chains_of(pdb):
    """Chain id -> list of (line, resnum key) in file order, heavy atoms only."""
    out = {}
    for line in pdb.split("\n"):
        if not line.startswith(("ATOM", "HETATM")) or element(line).upper() == "H":
            continue
        out.setdefault(line[21], []).append(line)
    return out


def residues(lines):
    seen, order = {}, []
    for l in lines:
        k = (l[22:27], l[17:20].strip())
        if k not in seen:
            seen[k] = len(order)
            order.append(k)
    return seen, order


METALS = {'FE', 'CU', 'ZN', 'MN', 'NI', 'CO', 'MG'}


def xyz(l):
    return [float(l[30:38]), float(l[38:46]), float(l[46:54])]


def components(lines, split_metals):
    """Bonded components of a chain's atoms; `split_metals` drops a metal's contacts (coordination, not bonds)."""
    adj, _ = graph([(element(l), xyz(l)) for l in lines], split_metals=split_metals)
    seen, out = set(), []
    for start in range(len(lines)):
        if start in seen:
            continue
        group, stack = [], [start]
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i); group.append(i); stack.extend(adj[i])
        out.append([lines[i] for i in sorted(group)])
    return out


def units(chains, binder):
    """The design's non-binder molecules: a DNA strand whole, a ligand chain as its bonded components (RFD4 writes
    several ions or molecules as one chain where the task lists each). Each comes with the finer split to try when
    it matches nothing whole: two copper ions 2.3 A apart bond by distance and are two molecules, where a heme's iron
    is part of the heme."""
    out = []
    for c, ls in chains.items():
        if c == binder:
            continue
        if all(l[17:20].strip() in NA for l in ls):
            out.append((ls, [ls]))
            continue
        for whole in components(ls, False):
            out.append((whole, components(whole, True)))
    return out


def reference_element(name):
    """A CCD atom name's element: a metal when the name, digits stripped, is one (FE1, CO), else its first letter."""
    stem = name.rstrip("0123456789").upper()
    return stem if stem in METALS else stem[0]


def match(unit, entry):
    """Our atom names for a unit if it is this task entry, else None."""
    names = [r for _, r in residues(unit)[1]]
    if entry["kind"] == "dna":
        ok = all(n in NA for n in names) and "".join(NA[n] for n in names) == entry["sequence"]
        return [l[12:16] for l in unit] if ok else None
    if entry.get("ccd"):
        if names == [entry["ccd"]]:
            return [l[12:16] for l in unit]
        # a one-atom ion written as a generic ligand: its CCD atom name is its element
        if len(unit) == 1 and element(unit[0]).upper() == entry["ccd"].upper():
            return [f" {entry['ccd'].upper()}".ljust(4)]
        # a dictionary ligand written as a generic one: matched by bond graph against its reference conformer
        reference = entry.get("conformerNames")
        if not reference or len(reference) != len(unit):
            return None
        mapped = isomorphism([(element(l), xyz(l)) for l in unit], [(reference_element(a[0]), a[1:]) for a in reference])
        return None if mapped is None else [(" " + reference[mapped[k]][0] if len(reference[mapped[k]][0]) < 4 else reference[mapped[k]][0]).ljust(4) for k in range(len(unit))]
    mapped = isomorphism([(element(l), xyz(l)) for l in unit], [(a[0], a[1:]) for a in entry["conformer"]])
    if mapped is None:
        return None
    # our featuriser names a SMILES ligand's atoms element + per-element counter, in the conformer's order
    conformer = [a[0] for a in entry["conformer"]]
    ours = [f" {e}{conformer[:k + 1].count(e)}".ljust(4) for k, e in enumerate(conformer)]
    return [ours[mapped[k]] for k in range(len(unit))]


def relabel(pdb, task):
    t = TASKS[task]
    chains = chains_of(pdb)
    protein = [c for c, ls in chains.items() if any(l.startswith("ATOM") and l[17:20] in AA for l in ls)]
    if len(protein) != 1:
        raise ValueError(f"{task}: {len(protein)} protein chains in the design")
    binder = protein[0]
    index, order = residues(chains[binder])
    out = [l[:21] + "A" + f"{index[(l[22:27], l[17:20].strip())] + 1:4d} " + l[27:] for l in chains[binder]]
    sequence = "".join(AA[r] for _, r in order)
    taken = set()
    def place(unit):
        for i, e in enumerate(t["chains"]):
            names = None if i in taken else match(unit, e)
            if names is not None:
                return i, names
        return None
    placed = []
    for whole, pieces in units(chains, binder):
        found = place(whole)
        if found is not None:
            taken.add(found[0]); placed.append((whole, *found))
            continue
        for unit in pieces if len(pieces) > 1 else []:
            found = place(unit)
            if found is None:
                break
            taken.add(found[0]); placed.append((unit, *found))
        else:
            if len(pieces) > 1:
                continue
        raise ValueError(f"{task}: a design molecule ({whole[0][17:20]}, {len(whole)} atoms) matches no entry of the task")
    for unit, i, names in placed:
        index, _ = residues(unit)
        out += [l[:12] + n + l[16:21] + chr(66 + i) + f"{index[(l[22:27], l[17:20].strip())] + 1:4d} " + l[27:] for l, n in zip(unit, names)]
    if len(taken) != len(t["chains"]):
        raise ValueError(f"{task}: the design has {len(taken)} of the task's {len(t['chains'])} other molecules")
    out.sort(key=lambda l: l[21])
    return sequence, "\n".join(out) + "\n"


def main():
    src, prefix, designs, work, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    files = {f[: -len(".cif.zst")]: os.path.join(src, f) for f in os.listdir(src) if f.endswith(".cif.zst")}
    if len(sys.argv) > 6:
        wanted = sorted({json.loads(l)["target"] for l in open(sys.argv[6])})
    else:
        wanted = sorted(prefix + k.replace("_sequence_context", "").replace("_structure_context", "") for k in files)
    os.makedirs(work, exist_ok=True)
    done = {(r["target"], r["attempt"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    for target in wanted:
        if all((target, k) in done for k in range(designs)):
            continue
        key = target[len(prefix):]
        source = files.get(key) or files.get(key + ("_structure_context" if prefix == "F-" else "_sequence_context"))
        if source is None:
            raise ValueError(f"no input for {target} in {src}")
        job = os.path.join(work, f"{target}.json")
        json.dump({"input": source, "input_as_annotated_cif": True, "recycleSchedule": "high", "nsteps": 60}, open(job, "w"))
        pdb = os.path.join(work, f"{target}.pdb")
        subprocess.run([os.path.join(ROOT, "cuda/rfd4p/localfold-rfd4p"), f"--job={job}", f"--int8={ROOT}/model-rfd4p-int8",
                        f"--out={pdb}", f"--designs={designs}", f"--seed={zlib.crc32(target.encode()) % 100000}", "--fast"],
                       check=True, stdout=open(pdb + ".log", "w"), stderr=subprocess.STDOUT)
        with open(out, "a") as f:
            for k in range(designs):
                path = pdb if k == 0 else pdb[:-4] + f"_{k + 1}.pdb"
                try:
                    sequence, design = relabel(open(path).read(), target)
                except ValueError as error:
                    # 🔴 recorded, never dropped: an unreadable design is counted as a failed attempt
                    f.write(json.dumps({"target": target, "task": TASKS[target]["task"], "attempt": k, "config": "RFD4P-high",
                                        "unreadable": str(error)}) + "\n")
                    print(target, k, "unreadable:", error, flush=True)
                    continue
                f.write(json.dumps({"target": target, "task": TASKS[target]["task"], "attempt": k, "config": "RFD4P-high",
                                    "binder": sequence, "designPdb": design}) + "\n")
        print(target, "ok", flush=True)


if __name__ == "__main__":
    main()
