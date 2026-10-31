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


def graph(atoms):
    """Bonds from distances (covalent radii + 0.45 A) over [(element, xyz)]: one rule for both sides of a match."""
    n = len(atoms)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            (a, p), (b, q) = atoms[i], atoms[j]
            if sum((x - y) ** 2 for x, y in zip(p, q)) ** 0.5 < RADII[a.upper()] + RADII[b.upper()] + 0.45:
                adj[i].add(j); adj[j].add(i)
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
    (da, dc), (ra, rc) = graph(design), graph(reference)
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


def relabel(pdb, task):
    t = TASKS[task]
    chains = chains_of(pdb)
    protein = [c for c, ls in chains.items() if any(l.startswith("ATOM") and l[17:20] in AA for l in ls)]
    if len(protein) != 1:
        raise ValueError(f"{task}: {len(protein)} protein chains in the design")
    binder = protein[0]
    taken, letter, atom_maps = set(), {binder: "A"}, {}
    for c, ls in chains.items():
        if c == binder:
            continue
        _, order = residues(ls)
        names = [r for _, r in order]
        for i, e in enumerate(t["chains"]):
            if i in taken:
                continue
            if e["kind"] == "dna" and all(n in NA for n in names) and "".join(NA[n] for n in names) == e["sequence"]:
                break
            if e["kind"] == "ligand" and e.get("ccd") and names == [e["ccd"]]:
                break
            if e["kind"] == "ligand" and e.get("smiles"):
                mapped = isomorphism([(element(l), [float(l[30:38]), float(l[38:46]), float(l[46:54])]) for l in ls],
                                     [(a[0], a[1:]) for a in e["conformer"]])
                if mapped is not None:
                    atom_maps[c] = mapped
                    break
        else:
            raise ValueError(f"{task}: design chain {c} ({names[:3]}, {len(ls)} atoms) matches no entry of the task")
        taken.add(i)
        letter[c] = chr(66 + i)
    if len(taken) != len(t["chains"]):
        raise ValueError(f"{task}: the design has {len(taken)} of the task's {len(t['chains'])} other chains")
    out, sequence = [], ""
    for c, ls in chains.items():
        index, order = residues(ls)
        if c in atom_maps:
            # our featuriser names a SMILES ligand's atoms element + per-element counter, in the conformer's order
            conformer = [a[0] for a in t["chains"][ord(letter[c]) - 66]["conformer"]]
            ours = [f" {e}{conformer[:k + 1].count(e)}".ljust(4) for k, e in enumerate(conformer)]
        for k, l in enumerate(ls):
            number = index[(l[22:27], l[17:20].strip())] + 1
            name = ours[atom_maps[c][k]] if c in atom_maps else l[12:16]
            out.append(l[:12] + name + l[16:21] + letter[c] + f"{number:4d} " + l[27:])
        if letter[c] == "A":
            sequence = "".join(AA[r] for _, r in order)
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
                sequence, design = relabel(open(path).read(), target)
                f.write(json.dumps({"target": target, "task": TASKS[target]["task"], "attempt": k, "config": "RFD4P-high",
                                    "binder": sequence, "designPdb": design}) + "\n")
        print(target, "ok", flush=True)


if __name__ == "__main__":
    main()
