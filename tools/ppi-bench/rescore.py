# Re-scores benchmark designs with the native CUDA AlphaFold 3 (cuda/af3, af3-any-model's int8 weights) instead
# of the page's int5 WebGPU fold - an independent evaluator for the same criteria: single sequence, target
# templated, 5 recycles, 200 steps, one sample.
# usage: rescore.py <in.jsonl> <out.jsonl> [seed]
import json, os, subprocess, sys
import numpy as np
ROOT = os.path.expanduser("~/localfold")
HERE = os.path.dirname(os.path.abspath(__file__))
src, dst = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261029
targets = json.load(open(os.path.join(HERE, "targets.json")))
work = os.path.join(HERE, "rescore-" + os.path.basename(dst).replace(".jsonl", ""))
os.makedirs(os.path.join(work, "t"), exist_ok=True)
done = set()
if os.path.exists(dst):
    done = {(d["target"], d["attempt"]) for d in map(json.loads, open(dst))}
every = [d for d in map(json.loads, open(src)) if (d["target"], d["attempt"]) not in done]
rows = [d for d in every if d.get("binder")]
lines = []
for d in rows:
    t = targets[d["target"]]
    tp = os.path.join(work, "t", d["target"] + ".pdb")
    if not os.path.exists(tp): open(tp, "w").write(t["pdb"])
    slots = ",".join(f"{tp}:{'BCDEFG'[i]}@{i + 1}" for i in range(len(t["sequences"])))
    out = os.path.join(work, f"{d['target']}_{d['attempt']}.pdb")
    lines.append(f"{out} --sequence={':'.join([d['binder']] + t['sequences'])} --template={slots}")
batch = os.path.join(work, "batch.txt")
open(batch, "w").write("\n".join(lines) + "\n")
if lines:
    subprocess.run([os.path.join(ROOT, "cuda/af3/fold"), f"--batch={batch}", "--",
                    "--recycles=5", "--steps=200", f"--seed={seed}"], check=True)

def backbone(pdb_text):
    m = {}
    for l in pdb_text.split("\n"):
        if l.startswith("ATOM") and l[12:16].strip() in ("N", "CA", "C", "O"):
            m[(l[21], l[22:26].strip(), l[12:16].strip())] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
    return m

def binder_rmsd(design, refold):
    a, b = backbone(design), backbone(refold)
    keys = [k for k in a if k in b]
    tk = [k for k in keys if k[0] != "A"]; bk = [k for k in keys if k[0] == "A"]
    x = np.array([b[k] for k in tk]); y = np.array([a[k] for k in tk])
    cx, cy = x.mean(0), y.mean(0)
    u, s, vt = np.linalg.svd((x - cx).T @ (y - cy))
    r = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    xb = (np.array([b[k] for k in bk]) - cx) @ r.T + cy
    return float(np.sqrt(((xb - np.array([a[k] for k in bk])) ** 2).sum(1).mean()))

with open(dst, "a") as f:
    for d in rows:
        out = os.path.join(work, f"{d['target']}_{d['attempt']}.pdb")
        summary = json.load(open(out.replace(".pdb", "_summary_confidences.json")))
        pm = summary["chain_pair_pae_min"]; k = len(pm)
        ipae = min((pm[0][j] + pm[j][0]) / 2 for j in range(1, k))
        rmsd = binder_rmsd(d["designPdb"], open(out).read())
        f.write(json.dumps({"target": d["target"], "attempt": d["attempt"], "binder": d["binder"],
                            "binderPtm": summary["chain_ptm"][0], "ipae": ipae, "rmsd": rmsd,
                            "designPdb": d["designPdb"], "pageBinderPtm": d.get("binderPtm"),
                            "pageIpae": d.get("ipae"), "pageRmsd": d.get("rmsd")}) + "\n")
    # an attempt that produced no design is a failure under either evaluator
    for d in every:
        if not d.get("binder") and not d.get("error"):
            f.write(json.dumps({"target": d["target"], "attempt": d["attempt"], "binderPtm": None}) + "\n")
print("rescored", len(rows))
