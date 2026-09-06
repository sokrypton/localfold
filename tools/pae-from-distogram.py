#!/usr/bin/env python3
"""Can a PAE matrix be estimated from a distogram and a structure?

    node tools/gpu-chrome.mjs tools/gpu/probe-pae-from-distogram.js \
      --sequence=<SEQ> > oracle-dumps/pae/<name>.json      # collect
    python3 tools/pae-from-distogram.py oracle-dumps/pae/*.json   # score

WHY IT IS WORTH ASKING. EF2-fast has no confidence head - no pLDDT, no PAE - and
its distogram is the only thing it says about its own reliability. PAE is the
score people read off a complex, because it is what says whether two parts are
placed correctly relative to each other.

WHY IT CAN BE CHECKED. AlphaFold 3 produces BOTH out of one fold, so an
estimator can be scored against a real PAE before being carried to the model
that has none.

WHAT PAE IS, WHICH DECIDES THE SHAPE OF THE ESTIMATOR. Aligning on token i's
frame, let Delta_ij = dx_j - dx_i be the relative displacement error. Then

    PAE(i,j)^2  ~  E||Delta_ij||^2        the full 3-D magnitude
    sigma_ij^2  =  E[(u_ij . Delta_ij)^2]  ONE radial projection of it

so a distogram gives one scalar projection, along a known direction, of a
3-vector. That is exactly what it is missing: a domain rotation displaces things
TANGENTIALLY, which is the direction u_ij cannot see, so sigma systematically
under-reads inter-domain error.

AND E||Delta_ij||^2 = g_ii + g_jj - 2 g_ij for g the Gram matrix of displacement
covariances - so PAE^2 is a SQUARED-DISTANCE MATRIX, and double-centring it
gives a Gram matrix whose rank is the number of collective modes. Measured, on
these thirteen targets:

  * PAE needs 6-10 components for 90% of its energy; sigma needs 13-33 at the
    same sizes. The distogram is a high-rank per-pair signal and PAE is a
    low-rank collective one.
  * PAE is ~88% symmetric (|asym| / |sym| ~ 0.12), so the frame-rotation term
    is a correction rather than the substance.
  * double-centred PAE^2 puts 61-88% of its energy in THREE eigenvalues.

That says the features should be `mobility_i + mobility_j` (the g_ii terms) plus
a pair coupling (g_ij), which is what the fit is given.

🔴 EVERY NUMBER HERE IS LEAVE-ONE-TARGET-OUT. The fit sees the other twelve
targets and never the one it is scored on. Fitting and scoring on the same rows
is what "twenty free parameters beat one" manufactures, and this repository has
already been caught by it once.

🔴 AND THE BASELINE IS THE GEOMETRY, NOT ZERO. PAE grows with distance whatever
the model believes, so `d_ij` alone already scores 0.658. The claim is only ever
what the DISTOGRAM adds on top of that.
"""
import glob
import json
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

BASE = ["sigma", "d", "entropy", "|E[d]-d|", "min(d,22)"]
# 🔴 THE GRAM TERMS. `PAE^2 = g_ii + g_jj - 2 g_ij` splits into a per-token
# mobility and a pair coupling; sigma_ij is the coupling and these are the
# mobility, read off the distogram as how uncertain a token's distances are in
# general. Worth 0.745 on their own, against the geometry's 0.658.
MOBILITY = ["mob_i+mob_j", "mobfar_i+mobfar_j", "max(mob)"]


def load(path):
    d = json.load(open(path))
    n = d["tokens"]
    D = d["data"]
    g = lambda k: np.array(D[k], dtype=np.float64).reshape(n, n)
    keep = ~np.eye(n, dtype=bool)
    obs, mean, sigma = g("observed"), g("mean"), g("sigma")
    mob = sigma.mean(axis=1)
    # ...and the same read over FAR pairs only, which is where a collective
    # motion shows and a bond length does not.
    far = obs > 12
    mobfar = np.array([sigma[i][far[i]].mean() if far[i].any() else mob[i]
                       for i in range(n)])
    columns = {
        "sigma": sigma, "d": obs, "entropy": g("entropy"),
        "|E[d]-d|": np.abs(mean - obs),
        # 🔴 `min(d, 22)` IS A FEATURE BECAUSE THE DISTOGRAM STOPS AT 22 A.
        # Its last bin is open-ended, so it cannot tell 30 A from 60 - while PAE
        # runs to 32. Beyond the grid the distogram is uninformative BY
        # CONSTRUCTION, and saying so is more honest than letting a fit find it.
        "min(d,22)": np.minimum(obs, 22),
        "mob_i+mob_j": mob[:, None] + mob[None, :],
        "mobfar_i+mobfar_j": mobfar[:, None] + mobfar[None, :],
        "max(mob)": np.maximum(mob[:, None], mob[None, :]),
    }
    return {
        "name": path.split("/")[-1][: -len(".json")],
        "n": n, "keep": keep, "y": g("pae")[keep], "sigma": sigma, "obs": obs,
        "columns": columns,
        "X": lambda cols, c=columns, k=keep: np.stack([c[n2][k] for n2 in cols], axis=1),
    }


def spectrum(t):
    """How low-rank the true PAE is, and how low-rank the distogram is not."""
    P = np.zeros((t["n"], t["n"]))
    P[t["keep"]] = t["y"]
    sym = (P + P.T) / 2
    rank90 = lambda M: int(np.searchsorted(
        np.cumsum(np.linalg.svd(M, compute_uv=False))
        / np.linalg.svd(M, compute_uv=False).sum(), 0.90) + 1)
    J = np.eye(t["n"]) - np.ones((t["n"], t["n"])) / t["n"]
    w = np.abs(np.linalg.eigvalsh(-0.5 * J @ (sym ** 2) @ J))
    w = np.sort(w)[::-1]
    return {
        "asym/sym": np.abs((P - P.T) / 2).mean() / max(np.abs(sym).mean(), 1e-9),
        "rank90 PAE": rank90(sym), "rank90 sigma": rank90(t["sigma"]),
        "gram top3": w[:3].sum() / w.sum(),
    }


def fit(train, test, cols):
    X = np.concatenate([t["X"](cols) for t in train])
    y = np.concatenate([t["y"] for t in train])
    beta, *_ = np.linalg.lstsq(np.c_[X, np.ones(len(X))], y, rcond=None)
    Xt = test["X"](cols)
    return np.c_[Xt, np.ones(len(Xt))] @ beta


def main(paths):
    targets = []
    for path in sorted(paths):
        t = load(path)
        # A target whose PAE barely varies cannot rank anything.
        if t["y"].std() < 0.2:
            print(f"skipping {t['name']}: PAE sd {t['y'].std():.2f}, nothing to rank")
            continue
        targets.append(t)
    if len(targets) < 3:
        raise SystemExit("need at least three targets to hold one out")

    print("What the target IS: low-rank and nearly symmetric, where the "
          "distogram is neither.")
    print(f"  {'target':12s} {'n':>4s} {'asym/sym':>9s} {'rank90 PAE':>11s}"
          f" {'rank90 sigma':>13s} {'gram top3':>10s}")
    for t in targets:
        d = spectrum(t)
        print(f"  {t['name']:12s} {t['n']:4d} {d['asym/sym']:9.3f}"
              f" {d['rank90 PAE']:11d} {d['rank90 sigma']:13d} {d['gram top3']:10.3f}")

    rng = np.random.default_rng(1)
    arms = {
        "geometry only (baseline)": ["d", "min(d,22)"],
        "mobility, no pair term": ["d", "min(d,22)"] + MOBILITY[:2],
        "the distogram, per pair": BASE,
        "both (mobility + coupling)": BASE + MOBILITY,
    }
    print(f"\n{'target':12s} {'PAEsd':>6s} | "
          + " ".join(f"{k.split(' (')[0][:22]:>22s}" for k in arms)
          + f" | {'shuffled':>9s} {'d_ij':>7s} | {'RMSE A':>7s}")
    rows = []
    for t in targets:
        train = [o for o in targets if o is not t]
        y = t["y"]
        preds = {name: fit(train, t, cols) for name, cols in arms.items()}
        # 🔴 THE CONTROL THAT SEPARATES "USED" FROM "ADDED". The same sigma
        # values on the WRONG pairs: identical marginal distribution, no
        # correspondence. It lands BELOW using no distogram at all, which is
        # what says the pairing carries the signal rather than the column.
        shuffled = t["sigma"].copy()
        upper = np.triu_indices(t["n"], 1)
        v = shuffled[upper]
        rng.shuffle(v)
        shuffled[upper] = v
        shuffled.T[upper] = v
        ranked = rankdata(shuffled[t["keep"]]) + rankdata(t["obs"][t["keep"]])
        best = preds["both (mobility + coupling)"]
        row = [spearmanr(preds[k], y)[0] for k in arms] + [
            spearmanr(ranked, y)[0], spearmanr(t["obs"][t["keep"]], y)[0],
            float(np.sqrt(((best - y) ** 2).mean()))]
        rows.append(row)
        print(f"{t['name']:12s} {y.std():6.2f} | "
              + " ".join(f"{row[i]:22.3f}" for i in range(len(arms)))
              + f" | {row[-3]:9.3f} {row[-2]:7.3f} | {row[-1]:7.2f}")

    a = np.array(rows)
    labels = list(arms) + ["sigma SHUFFLED", "d_ij alone"]
    print(f"\n{'':30s} {'median':>8s} {'worst':>8s}")
    for i, label in enumerate(labels):
        print(f"  {label:28s} {np.median(a[:, i]):8.3f} {a[:, i].min():8.3f}")
    print(f"  {'RMSE of the best arm':28s} {np.median(a[:, -1]):8.2f} A")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(glob.glob("oracle-dumps/pae/*.json"))
    raise SystemExit(main(args))
