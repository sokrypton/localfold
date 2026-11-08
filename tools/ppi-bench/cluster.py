# Structural clusters of passing binders, standing in for the paper's Foldseek (TM-score 0.5) clustering,
# which could not be installed here. A small TM-align on CA traces: gapless-threading seeds, then
# Needleman-Wunsch on the TM-score similarity of the current superposition, a few rounds. Two binders
# join when the TM-score normalised by the SHORTER is >= 0.5 - the generous side, so clusters come out
# no more numerous than Foldseek's would (the conservative direction for our count).
import numpy as np

def ca_of(pdb, chain="A"):
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in pdb.split("\n")
                     if l.startswith("ATOM") and l[12:16].strip() == "CA" and l[21] == chain])

def kabsch(x, y):
    cx, cy = x.mean(0), y.mean(0)
    h = (x - cx).T @ (y - cy)
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, cy - cx @ r.T

def d0(n):
    return max(0.5, 1.24 * np.cbrt(max(n, 19) - 15) - 1.8)

def tm_of(a, b, pairs, norm):
    """Best TM-score over superpositions fitted on subsets of the aligned pairs (TM-score's search)."""
    if len(pairs) < 3: return 0.0, None
    i, j = np.array(pairs).T
    x, y = a[i], b[j]
    dz = d0(norm)
    best, best_rt = 0.0, None
    for frag in sorted({len(pairs), max(4, len(pairs) // 2), max(4, len(pairs) // 4), 8}):
        for start in range(0, len(pairs) - frag + 1, max(1, frag // 2)):
            sel = np.arange(start, start + frag)
            for _ in range(10):
                r, t = kabsch(x[sel], y[sel])
                dist = np.linalg.norm(x @ r.T + t - y, axis=1)
                score = np.sum(1 / (1 + (dist / dz) ** 2)) / norm
                if score > best: best, best_rt = score, (r, t)
                new = np.where(dist < max(dz, 4.0))[0]
                if len(new) < 3 or np.array_equal(new, sel): break
                sel = new
    return best, best_rt

def align(a, b, rt, norm, gap=-0.6):
    r, t = rt
    dist = np.linalg.norm((a @ r.T + t)[:, None, :] - b[None, :, :], axis=2)
    s = 1 / (1 + (dist / d0(norm)) ** 2)
    n, m = s.shape
    f = np.zeros((n + 1, m + 1)); tb = np.zeros((n + 1, m + 1), dtype=np.int8)
    for p in range(1, n + 1):
        for q in range(1, m + 1):
            opts = (f[p - 1, q - 1] + s[p - 1, q - 1], f[p - 1, q] + gap, f[p, q - 1] + gap)
            k = int(np.argmax(opts)); f[p, q] = opts[k]; tb[p, q] = k
    pairs, p, q = [], n, m
    while p > 0 and q > 0:
        if tb[p, q] == 0: pairs.append((p - 1, q - 1)); p -= 1; q -= 1
        elif tb[p, q] == 1: p -= 1
        else: q -= 1
    return pairs[::-1]

def tm_score(a, b):
    norm = min(len(a), len(b))
    best, best_rt = 0.0, None
    for shift in range(-len(a) + 8, len(b) - 8, max(1, norm // 10)):
        pairs = [(i, i + shift) for i in range(len(a)) if 0 <= i + shift < len(b)]
        score, rt = tm_of(a, b, pairs, norm)
        if score > best: best, best_rt = score, rt
    for _ in range(3):
        if best_rt is None: break
        score, rt = tm_of(a, b, align(a, b, best_rt, norm), norm)
        if score <= best + 1e-4: break
        best, best_rt = score, rt
    return best

def clusters(cas, threshold=0.5):
    n = len(cas)
    link = [[i] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if tm_score(cas[i], cas[j]) >= threshold:
                link[i].append(j); link[j].append(i)
    left, count = set(range(n)), 0
    while left:   # greedy set cover, as Foldseek's default cluster mode
        rep = max(left, key=lambda i: len(set(link[i]) & left))
        left -= set(link[rep]); count += 1
    return count

if __name__ == "__main__":
    import sys, json
    rng = np.random.default_rng(0)
    helix = np.array([[2.3 * np.cos(k * 1.745), 2.3 * np.sin(k * 1.745), 1.5 * k] for k in range(40)])
    rot, _ = kabsch(rng.normal(size=(5, 3)), rng.normal(size=(5, 3)))
    print("self", round(tm_score(helix, helix @ rot.T + 3), 3),
          "random", round(tm_score(helix, np.cumsum(rng.normal(size=(40, 3)) * 2.2, 0)), 3))
