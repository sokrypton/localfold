"""CA RMSD of a predicted PDB against a reference, after Kabsch superposition, plus the chain
check every fold tool here asserts (consecutive CA about 3.8 A).

    python3 cuda/af3/score.py <predicted.pdb> [reference.pdb] [reference chain]
    python3 cuda/af3/score.py <predicted.pdb> <reference.pdb> A,D    # a complex

For a complex the predicted chains A, B, ... pair with the listed reference chains in order, one
superposition over all of them (two chains can each be right and placed wrongly), each chain
also re-fitted alone.
"""
import math
import sys

# 🔴 NUMPY WHERE THERE IS ONE, AND THE SAME ARITHMETIC BY HAND WHERE THERE IS NOT. Every Mac in this tree runs
# the system python3 from the Command Line Tools, which has no numpy and cannot be given one without installing
# into a managed interpreter - so this script raised `ModuleNotFoundError` there and took two whole gates with
# it: `metal/af2/gate.py` reported "unscored: No module named numpy" on all seven cases, and
# `metal/af3/gate.py` printed `RMSD   -` on nineteen and called every one of them MOVED, which reads as a moved
# fold rather than an absent scorer. The arrays here are a few hundred alpha carbons; the only real work is a
# 3 x 3 SVD, and Kabsch needs its singular VALUES alone (the rotation never appears in the RMSD), which are the
# square roots of the eigenvalues of a symmetric 3 x 3 - closed form. Same precedent as
# `cuda/featurise/jsnum.h`'s `__APPLE__` branch for `to_chars`.
try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def _centre(v):
    n = len(v)
    c = [sum(r[k] for r in v) / n for k in range(3)]
    return [[r[k] - c[k] for k in range(3)] for r in v]


def _eig3(a):
    """The eigenvalues of a symmetric 3 x 3, descending (Smith's closed form)."""
    p1 = a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2
    if p1 == 0:
        return sorted([a[0][0], a[1][1], a[2][2]], reverse=True)
    q = (a[0][0] + a[1][1] + a[2][2]) / 3
    p2 = (a[0][0] - q) ** 2 + (a[1][1] - q) ** 2 + (a[2][2] - q) ** 2 + 2 * p1
    p = math.sqrt(p2 / 6)
    b = [[(a[i][j] - (q if i == j else 0)) / p for j in range(3)] for i in range(3)]
    r = _det3(b) / 2
    phi = math.acos(max(-1.0, min(1.0, r))) / 3
    e1 = q + 2 * p * math.cos(phi)
    e3 = q + 2 * p * math.cos(phi + 2 * math.pi / 3)
    return [e1, 3 * q - e1 - e3, e3]


def _det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _median(xs):
    v = sorted(xs)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def _steps(order):
    return [math.dist(order[i], order[i + 1]) for i in range(len(order) - 1)]


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
    p = [pred[i][1] for i, _ in pairs]
    r = [ref[j][1] for _, j in pairs]
    return (np.array(p) if np is not None else p, np.array(r) if np is not None else r,
            [pred[k][1] for k in pk])


def kabsch_rmsd(p, q):
    if np is not None:
        p = p - p.mean(0)
        q = q - q.mean(0)
        u, s, vt = np.linalg.svd(p.T @ q)
        d = np.sign(np.linalg.det(u @ vt))
        r = u @ np.diag([1, 1, d]) @ vt
        return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))
    # 🔴 THE SINGULAR VALUES ALONE, because the rotation cancels out of the residual:
    # `N rmsd^2 = sum|p|^2 + sum|q|^2 - 2 (s1 + s2 + sign(det H) s3)` for `H = p^T q`. So there is no SVD to
    # write - the singular values are the square roots of the eigenvalues of `H^T H`, which is symmetric.
    p, q = _centre(p), _centre(q)
    n = len(p)
    h = [[sum(p[t][i] * q[t][j] for t in range(n)) for j in range(3)] for i in range(3)]
    hth = [[sum(h[t][i] * h[t][j] for t in range(3)) for j in range(3)] for i in range(3)]
    sv = [math.sqrt(max(0.0, e)) for e in _eig3(hth)]
    if _det3(h) < 0:
        sv[2] = -sv[2]
    sq = sum(v * v for r in p for v in r) + sum(v * v for r in q for v in r)
    return math.sqrt(max(0.0, (sq - 2 * sum(sv)) / n))


refChains = sys.argv[3].split(",") if len(sys.argv) > 3 else [None]
if len(refChains) > 1:
    ps, rs, alone = [], [], []
    for k, rc in enumerate(refChains):
        pk, rk, order = paired(ca(sys.argv[1], "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[k]), ca(sys.argv[2], rc))
        steps = _steps(order)
        alone.append(f"{rc} {kabsch_rmsd(pk, rk):.3f} A over {len(pk)} (CA-CA median {_median(steps):.3f})")
        ps.append(pk); rs.append(rk)
    cat = (lambda xs: np.concatenate(xs)) if np is not None else (lambda xs: [r for x in xs for r in x])
    print(f"complex CA RMSD {kabsch_rmsd(cat(ps), cat(rs)):.3f} A over "
          f"{sum(len(x) for x in ps)}; alone: " + ", ".join(alone))
    sys.exit(0)
p, r, chainOrder = paired(ca(sys.argv[1]), ca(sys.argv[2] if len(sys.argv) > 2 else "tools/fixtures/6mrr-crystal.pdb",
                                              refChains[0]))
steps = _steps(chainOrder)
print(f"CA RMSD {kabsch_rmsd(p, r):.3f} A over {len(p)}   "
      f"CA-CA median {_median(steps):.3f} min {min(steps):.3f} max {max(steps):.3f}")
