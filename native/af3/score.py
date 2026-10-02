"""CA RMSD of a predicted PDB against a reference, after Kabsch superposition, plus the chain
check every fold tool here asserts (consecutive CA about 3.8 A).

    python3 native/af3/score.py <predicted.pdb> [reference.pdb]
"""
import sys
import numpy as np


def ca(path, chain=None):
    """{residue number: (name, xyz)} for one chain (the first one by default)."""
    out = {}
    for line in open(path):
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        if chain is None:
            chain = line[21]
        if line[21] != chain:
            continue
        number = int(line[22:26])
        if number in out:
            continue          # an alternate location: the first one counts
        out[number] = (line[17:20], [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return out


def paired(pred, ref):
    """The residue-number offset that matches the most residue names, and the CA pairs."""
    best = max(range(-400, 401), key=lambda o: sum(
        1 for k, (name, _) in pred.items() if k + o in ref and ref[k + o][0] == name))
    keys = [k for k in pred if k + best in ref and ref[k + best][0] == pred[k][0]]
    return (np.array([pred[k][1] for k in keys]), np.array([ref[k + best][1] for k in keys]),
            [pred[k][1] for k in sorted(pred)])


def kabsch_rmsd(p, q):
    p = p - p.mean(0)
    q = q - q.mean(0)
    u, s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))


p, r, chainOrder = paired(ca(sys.argv[1]), ca(sys.argv[2] if len(sys.argv) > 2 else "tools/fixtures/6mrr-crystal.pdb",
                                              sys.argv[3] if len(sys.argv) > 3 else None))
steps = np.linalg.norm(np.diff(np.array(chainOrder), axis=0), axis=1)
print(f"CA RMSD {kabsch_rmsd(p, r):.3f} A over {len(p)}   "
      f"CA-CA median {np.median(steps):.3f} min {steps.min():.3f} max {steps.max():.3f}")
