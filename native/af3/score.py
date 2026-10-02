"""CA RMSD of a predicted PDB against a reference, after Kabsch superposition, plus the chain
check every fold tool here asserts (consecutive CA about 3.8 A).

    python3 native/af3/score.py <predicted.pdb> [reference.pdb]
"""
import sys
import numpy as np


def ca(path):
    seen, out = set(), []
    for line in open(path):
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        key = (line[21], int(line[22:26]))
        if key in seen:
            continue          # an alternate location: the first one counts
        seen.add(key)
        out.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(out)


def kabsch_rmsd(p, q):
    p = p - p.mean(0)
    q = q - q.mean(0)
    u, s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))


pred = ca(sys.argv[1])
ref = ca(sys.argv[2] if len(sys.argv) > 2 else "tools/fixtures/6mrr-crystal.pdb")
m = min(len(pred), len(ref))
steps = np.linalg.norm(np.diff(pred, axis=0), axis=1)
print(f"CA RMSD {kabsch_rmsd(pred[:m], ref[:m]):.3f} A over {m}   "
      f"CA-CA median {np.median(steps):.3f} min {steps.min():.3f} max {steps.max():.3f}")
