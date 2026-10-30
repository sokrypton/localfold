# disco-score.py <run.jsonl> ...: small-molecule rows (task-bench.mjs) under the paper's OTHER small-molecule criterion,
# `disco` (configs/metrics/pass_criteria/all_pass_criteria.yaml): binder backbone RMSD <= 2 A, and the ligand centroid
# - per ligand chain, after fitting the binder's backbone, the largest over ligands - within 2 A of the design's. It
# reads no confidence at all. Rescored from each row's own design and refold PDBs; beside rfd4_proteina_high's
# `disco` rows on the same targets.
#   🔴 the paper groups a ligand by connectivity including coordination bonds, so FE2 + AKG is one molecule there and
#   two chains here; the max over two centroids is never smaller than over their union's, so this can only be stricter
import json, os, sys, collections, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))

def parse(pdb):
    bb, lig = {}, collections.defaultdict(list)
    for line in pdb.split("\n"):
        if not line.startswith(("ATOM", "HETATM")): continue
        name, chain = line[12:16].strip(), line[21]
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        if chain == "A":
            if name in ("N", "CA", "C", "O"): bb[(line[22:26].strip(), name)] = xyz
        elif line[76:78].strip() != "H":
            lig[chain].append(xyz)
    return bb, {c: np.array(v) for c, v in lig.items()}

def fit(p, q):
    """R, t taking p onto q (Kabsch)."""
    pc, qc = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return r, qc - pc @ r

def disco(design, refold):
    db, dl = parse(design); fb, fl = parse(refold)
    keys = [k for k in db if k in fb]
    p, q = np.array([fb[k] for k in keys]), np.array([db[k] for k in keys])
    r, t = fit(p, q)
    rmsd = float(np.sqrt(((p @ r + t - q) ** 2).sum(1).mean()))
    if set(dl) != set(fl): raise ValueError(f"ligand chains differ: {sorted(dl)} vs {sorted(fl)}")
    cent = max(float(np.linalg.norm((fl[c] @ r + t).mean(0) - dl[c].mean(0))) for c in dl)
    return rmsd, cent

ref = json.load(open(os.path.join(HERE, "rfd4-paper-data.json")))
paper = collections.defaultdict(dict)
for row in ref["rows"]:
    if row.get("figure2") and row["criterion"] == "disco": paper[row["model"]][row["target"]] = row["pass_percent"]
for f in sys.argv[1:]:
    by = collections.defaultdict(list)
    for l in open(f):
        d = json.loads(l)
        if d.get("error") or d.get("task") != "sm" or not d.get("designPdb"): continue
        rmsd, cent = disco(d["designPdb"], d["refoldPdb"])
        by[d["target"]].append(rmsd <= 2.0 and cent <= 2.0)
    m = sum(sum(v) / len(v) for v in by.values()) / len(by)
    rng = random.Random(0)
    s = sorted(sum(sum(rng.choice(v) for _ in v) / len(v) for v in by.values()) / len(by) for _ in range(2000))
    print(f"{os.path.basename(f)}: disco {len(by)} targets, {sum(map(len, by.values()))} attempts, mean {100 * m:.1f}% "
          f"(CI {100 * s[50]:.1f}-{100 * s[1949]:.1f}), targets with a pass {sum(any(v) for v in by.values())}")
    for model in ("rfd4_proteina_high", "disco", "protein_hunter"):
        same = [paper[model][t] for t in by if t in paper[model]]
        print(f"   {model:20s} on these targets: {sum(same) / max(len(same), 1):.1f}%")
