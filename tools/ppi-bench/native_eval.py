# The benchmark's evaluation on the native CUDA AlphaFold 3 (cuda/af3, af3-any-model's int8 weights): [binder,
# ...target] single sequence, the target templated, 5 recycles, 200 steps, one sample - and the paper's terms
# (binder-chain pTM, min interface PAE averaged both ways, target-aligned binder backbone RMSD). rescore.py and
# bench.py both score through here, so a page run and a native run are judged by the same model.
import json, os, subprocess
import numpy as np
ROOT = os.path.expanduser("~/localfold")

def backbone(pdb_text):
    m = {}
    for l in pdb_text.split("\n"):
        if l.startswith("ATOM") and l[12:16].strip() in ("N", "CA", "C", "O"):
            m[(l[21], l[22:26].strip(), l[12:16].strip())] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
    return m

def binder_rmsd(design, refold):
    """The binder's backbone RMSD, design against refold, after fitting the target's backbone."""
    a, b = backbone(design), backbone(refold)
    keys = [k for k in a if k in b]
    tk = [k for k in keys if k[0] != "A"]; bk = [k for k in keys if k[0] == "A"]
    x = np.array([b[k] for k in tk]); y = np.array([a[k] for k in tk])
    cx, cy = x.mean(0), y.mean(0)
    u, s, vt = np.linalg.svd((x - cx).T @ (y - cy))
    r = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    xb = (np.array([b[k] for k in bk]) - cx) @ r.T + cy
    return float(np.sqrt(((xb - np.array([a[k] for k in bk])) ** 2).sum(1).mean()))

def fold_line(target, binder, template_path, out):
    """One `cuda/af3/fold` input for [binder, ...target], each target chain templated from the target's PDB."""
    slots = ",".join(f"{template_path}:{'BCDEFG'[i]}@{i + 1}" for i in range(len(target["sequences"])))
    return [out, f"--sequence={':'.join([binder] + target['sequences'])}", f"--template={slots}"]

def terms(out, design_pdb):
    """The paper's terms for a finished fold at `out`, against the design it refolds."""
    summary = json.load(open(out.replace(".pdb", "_summary_confidences.json")))
    pm = summary["chain_pair_pae_min"]; k = len(pm)
    ipae = min((pm[0][j] + pm[j][0]) / 2 for j in range(1, k))
    ptm = summary["chain_ptm"][0]
    rmsd = binder_rmsd(design_pdb, open(out).read())
    return {"binderPtm": ptm, "ipae": ipae, "rmsd": rmsd, "pass": ptm >= 0.8 and ipae <= 1.5 and rmsd <= 2.5}

def evaluate(target, binder, design_pdb, seed, work):
    """One design evaluated natively at `seed` (a resident `cuda/af3/fold --serve` if one is up)."""
    os.makedirs(work, exist_ok=True)
    tp = os.path.join(work, "target.pdb"); open(tp, "w").write(target["pdb"])
    out = os.path.join(work, "eval.pdb")
    subprocess.run([os.path.join(ROOT, "cuda/af3/fold"), *fold_line(target, binder, tp, out), "--",
                    "--recycles=5", "--steps=200", f"--seed={seed}"], check=True, capture_output=True)
    return terms(out, design_pdb)
