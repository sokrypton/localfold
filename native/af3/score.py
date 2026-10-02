"""CA RMSD of a predicted PDB against a reference, after Kabsch superposition, plus the chain
check every fold tool here asserts (consecutive CA about 3.8 A).

    python3 native/af3/score.py <predicted.pdb> [reference.pdb] [reference chain]
    python3 native/af3/score.py <predicted.pdb> <reference.pdb> A,D    # a complex

For a complex the predicted chains A, B, ... pair with the listed reference chains in order, one
superposition over all of them (two chains can each be right and placed wrongly), each chain
also re-fitted alone.
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
    """CA pairs by aligning the two chains' residue-name sequences (a gap or a numbering jump in
    the reference shifts no residue), and the predicted chain's CAs in order."""
    import difflib
    pk, rk = sorted(pred), sorted(ref)
    match = difflib.SequenceMatcher(None, [pred[k][0] for k in pk], [ref[k][0] for k in rk], autojunk=False)
    pairs = [(pk[a + i], rk[b + i]) for a, b, size in match.get_matching_blocks() for i in range(size)]
    return (np.array([pred[i][1] for i, _ in pairs]), np.array([ref[j][1] for _, j in pairs]),
            [pred[k][1] for k in pk])


def kabsch_rmsd(p, q):
    p = p - p.mean(0)
    q = q - q.mean(0)
    u, s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))


refChains = sys.argv[3].split(",") if len(sys.argv) > 3 else [None]
if len(refChains) > 1:
    ps, rs, alone = [], [], []
    for k, rc in enumerate(refChains):
        pk, rk, order = paired(ca(sys.argv[1], "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[k]), ca(sys.argv[2], rc))
        steps = np.linalg.norm(np.diff(np.array(order), axis=0), axis=1)
        alone.append(f"{rc} {kabsch_rmsd(pk, rk):.3f} A over {len(pk)} (CA-CA median {np.median(steps):.3f})")
        ps.append(pk); rs.append(rk)
    print(f"complex CA RMSD {kabsch_rmsd(np.concatenate(ps), np.concatenate(rs)):.3f} A over "
          f"{sum(len(x) for x in ps)}; alone: " + ", ".join(alone))
    sys.exit(0)
p, r, chainOrder = paired(ca(sys.argv[1]), ca(sys.argv[2] if len(sys.argv) > 2 else "tools/fixtures/6mrr-crystal.pdb",
                                              refChains[0]))
steps = np.linalg.norm(np.diff(np.array(chainOrder), axis=0), axis=1)
print(f"CA RMSD {kabsch_rmsd(p, r):.3f} A over {len(p)}   "
      f"CA-CA median {np.median(steps):.3f} min {steps.min():.3f} max {steps.max():.3f}")
