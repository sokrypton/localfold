# Cross-evaluation: re-predicts benchmark designs with a model that took no part in designing them, under the
# benchmark's criteria - the answer to "AF3 designed, AF3 checked, AF3 scored".
# usage: xeval.py <model> <in.jsonl> <out.jsonl> [seed]
#   <model>: boltz2 | chai1 | protenix2 | openbind0 (cuda/af3, the target templated as the benchmark's AF3 is,
#            single sequence, 5 recycles, 200 steps, one sample), or ef2 (cuda/ef2, ESMFold2-fast 600M with
#            Synthyra's confidence head: no template and no alignment by design, and no per-chain pTM - so its
#            verdict is interface PAE and RMSD alone, and `binderPtm` is null)
import json, os, subprocess, sys
import numpy as np
ROOT = os.path.expanduser("~/localfold")
HERE = os.path.dirname(os.path.abspath(__file__))
model, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 20261029
targets = json.load(open(os.path.join(HERE, "targets.json")))
work = os.path.join(HERE, f"xeval-{model}-" + os.path.basename(dst).replace(".jsonl", ""))
os.makedirs(os.path.join(work, "t"), exist_ok=True)
done = {(d["target"], d["attempt"]) for d in map(json.loads, open(dst))} if os.path.exists(dst) else set()
rows = [d for d in map(json.loads, open(src))
        if d.get("binder") and d.get("designPdb") and (d["target"], d["attempt"]) not in done]

def out_of(d): return os.path.join(work, f"{d['target']}_{d['attempt']}.pdb")
def sequence_of(d): return ":".join([d["binder"]] + targets[d["target"]]["sequences"])

if model == "ef2":
    for d in rows:
        if not os.path.exists(out_of(d)):
            subprocess.run([os.path.join(ROOT, "cuda/ef2/fold"), out_of(d), f"--sequence={sequence_of(d)}",
                            "--", f"--seed={seed}"], check=True, stdout=subprocess.DEVNULL)
else:
    lines = []
    for d in rows:
        t = targets[d["target"]]
        tp = os.path.join(work, "t", d["target"] + ".pdb")
        if not os.path.exists(tp): open(tp, "w").write(t["pdb"])
        slots = ",".join(f"{tp}:{'BCDEFG'[i]}@{i + 1}" for i in range(len(t["sequences"])))
        lines.append(f"{out_of(d)} --sequence={sequence_of(d)} --template={slots}")
    batch = os.path.join(work, "batch.txt")
    open(batch, "w").write("\n".join(lines) + "\n")
    if lines:
        subprocess.run([os.path.join(ROOT, "cuda/af3/fold"), f"--batch={batch}", f"--model={model}", "--",
                        "--recycles=5", "--steps=200", f"--seed={seed}"], check=True, stdout=subprocess.DEVNULL)

def backbone(text):
    m = {}
    for l in text.split("\n"):
        if l.startswith("ATOM") and l[12:16].strip() in ("N", "CA", "C", "O"):
            m[(l[21], l[22:26].strip(), l[12:16].strip())] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
    return m

def binder_rmsd(design, refold):
    """Binder backbone RMSD after fitting the target's backbone (the benchmark's), design against refold."""
    a, b = backbone(design), backbone(refold)
    keys = [k for k in a if k in b]
    tk = [k for k in keys if k[0] != "A"]; bk = [k for k in keys if k[0] == "A"]
    x = np.array([b[k] for k in tk]); y = np.array([a[k] for k in tk])
    cx, cy = x.mean(0), y.mean(0)
    u, s, vt = np.linalg.svd((x - cx).T @ (y - cy))
    r = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    xb = (np.array([b[k] for k in bk]) - cx) @ r.T + cy
    return float(np.sqrt(((xb - np.array([a[k] for k in bk])) ** 2).sum(1).mean()))

def interface_pae(stem):
    """AF3's chain_pair_pae_min for the binder, both directions averaged, min over partners - from the summary
    where the model writes one, else from the PAE matrix."""
    s = json.load(open(stem + "_summary_confidences.json"))
    if "chain_pair_pae_min" in s:
        pm = s["chain_pair_pae_min"]
        return min((pm[0][j] + pm[j][0]) / 2 for j in range(1, len(pm))), s.get("chain_ptm", [None])[0]
    c = json.load(open(stem + "_confidences.json"))
    pae = np.array(c["pae"]); ids = np.array(c["token_chain_ids"])
    binder = ids == ids[0]
    best = min((pae[np.ix_(binder, ids == other)].min() + pae[np.ix_(ids == other, binder)].min()) / 2
               for other in dict.fromkeys(ids[~binder]))
    return float(best), None

with open(dst, "a") as f:
    for d in rows:
        stem = out_of(d)[:-4]
        ipae, ptm = interface_pae(stem)
        f.write(json.dumps({"target": d["target"], "attempt": d["attempt"], "binder": d["binder"], "model": model,
                            "binderPtm": ptm, "ipae": ipae, "rmsd": binder_rmsd(d["designPdb"], open(out_of(d)).read()),
                            "designPdb": d["designPdb"]}) + "\n")
print("cross-evaluated", len(rows), "with", model)
