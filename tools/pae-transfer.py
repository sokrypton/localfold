#!/usr/bin/env python3
"""Does EF2-fast's distogram carry the same aligned-error information?

    python3 tools/pae-transfer.py

Reads the AF3 dumps (which carry a real PAE) and the EF2-fast dumps of the SAME
sequences (which carry none, this checkpoint having no confidence head), and
asks whether an estimator built on EF2-fast's distogram recovers AlphaFold 3's
PAE.

🔴 EVERY FEATURE IS IN ANGSTROMS, NOT IN BINS. AF3's distogram is 64 bins over
2-22 A and EF2-fast's is 128 over a borrowed 2-52, so a sigma read in bin
indices differs by a factor of two between them for the same physical spread.
Entropy is worse than that - it is not even the same UNIT, since a uniform
distribution over 128 bins carries log 2 more nats than one over 64 for free -
so it enters as `exp(H) * binWidth`, an effective width in angstroms.

🔴 AND THE TARGET IS ANOTHER MODEL'S PAE, WHICH HAS TO BE SAID. AF3's PAE is
about AF3's OWN structure; an estimate off EF2-fast is about EF2-fast's. Where
the two models fold a target the same way that difference is small and the
comparison is fair; where they disagree it is not, and a low score means "a
different fold" rather than "a bad estimator". So the agreement between the two
structures is measured per target and printed beside the score, rather than
left for the reader to wonder about.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr


def features_of(d, name):
    n = d["tokens"]
    D = d["data"]
    g = lambda k: np.array(D[k], dtype=np.float64).reshape(n, n)
    bins = d["bins"]
    # AF3 writes 63 breaks from 2.3125; EF2-fast's grid is uniform over 2-52.
    width = (22.0 - 2.0) / bins if bins == 64 else (52.0 - 2.0) / bins
    obs, mean, sigma = g("observed"), g("mean"), g("sigma")
    mob = sigma.mean(axis=1)
    far = obs > 12
    mobfar = np.array([sigma[i][far[i]].mean() if far[i].any() else mob[i]
                       for i in range(n)])
    keep = ~np.eye(n, dtype=bool)
    columns = {
        "sigma": sigma,
        "d": obs,
        "effWidth": np.exp(g("entropy")) * width,
        "|E[d]-d|": np.abs(mean - obs),
        "min(d,22)": np.minimum(obs, 22),
        "mob_i+mob_j": mob[:, None] + mob[None, :],
        "mobfar_i+mobfar_j": mobfar[:, None] + mobfar[None, :],
        "max(mob)": np.maximum(mob[:, None], mob[None, :]),
    }
    return {"name": name, "n": n, "keep": keep,
            "pae": g("pae")[keep], "obs": obs, "sigma": sigma, "columns": columns}


def load_af3(path):
    """One AF3 dump, one target."""
    return features_of(json.load(open(path)),
                       os.path.basename(path)[: -len(".json")])


def load_ef2(path):
    """One EF2-fast run, every target - see the probe's --targets."""
    blob = json.load(open(path))
    if "targets" in blob:
        return {t["name"]: features_of(t, t["name"]) for t in blob["targets"]}
    name = os.path.basename(path)[: -len(".json")]
    return {name: features_of(blob, name)}


COLS = ["sigma", "d", "effWidth", "|E[d]-d|", "min(d,22)",
        "mob_i+mob_j", "mobfar_i+mobfar_j", "max(mob)"]


def design(t, cols=COLS):
    return np.stack([t["columns"][c][t["keep"]] for c in cols], axis=1)


def loo(sources, targets, cols=COLS):
    """Fit on every OTHER target's features, predict this one's."""
    out = []
    for i, t in enumerate(targets):
        train = [(sources[j], targets[j]) for j in range(len(targets)) if j != i]
        X = np.concatenate([design(s, cols) for s, _ in train])
        y = np.concatenate([g["pae"] for _, g in train])
        beta, *_ = np.linalg.lstsq(np.c_[X, np.ones(len(X))], y, rcond=None)
        Xt = design(sources[i], cols)
        out.append(np.c_[Xt, np.ones(len(Xt))] @ beta)
    return out


def main():
    af3 = {os.path.basename(p)[: -len(".json")]: load_af3(p)
           for p in glob.glob("oracle-dumps/pae/*.json")}
    ef2 = {}
    for p in glob.glob("oracle-dumps/pae-ef2/*.json"):
        ef2.update(load_ef2(p))
    names = sorted(set(af3) & set(ef2))
    print(f"{len(af3)} AF3 dumps, {len(ef2)} EF2-fast folds, {len(names)} matched\n")
    pairs = []
    for name in names:
        a, e = af3[name], ef2[name]
        if a["n"] != e["n"]:
            print(f"skipping {name}: {a['n']} tokens in AF3, {e['n']} in EF2-fast")
            continue
        if a["pae"].std() < 0.2:
            continue
        pairs.append((a, e))
    if len(pairs) < 3:
        raise SystemExit("need at least three matched targets")

    af3s = [a for a, _ in pairs]
    ef2s = [e for _, e in pairs]
    truth = [a["pae"] for a in af3s]

    from_af3 = loo(af3s, af3s)
    from_ef2 = loo(ef2s, af3s)
    # ...the estimator with the distogram taken out, which is the geometry the
    # two models very nearly share.
    geom = loo(ef2s, af3s, ["d", "min(d,22)"])

    print(f"{'target':10s} {'n':>4s} {'folds agree':>11s} | {'from AF3':>9s}"
          f" {'from EF2':>9s} {'geometry':>9s} | {'EF2 RMSE':>9s}")
    rows = []
    for (a, e), y, pa, pe, pg in zip(pairs, truth, from_af3, from_ef2, geom):
        # how similar the two models' structures are, on the shared token set
        agree = pearsonr(a["obs"][a["keep"]], e["obs"][e["keep"]])[0]
        r = [spearmanr(pa, y)[0], spearmanr(pe, y)[0], spearmanr(pg, y)[0],
             float(np.sqrt(((pe - y) ** 2).mean())), agree]
        rows.append(r)
        print(f"{a['name']:10s} {a['n']:4d} {agree:11.3f} | {r[0]:9.3f}"
              f" {r[1]:9.3f} {r[2]:9.3f} | {r[3]:9.2f}")
    m = np.array(rows)
    print(f"\n{'median':10s} {'':4s} {np.median(m[:, 4]):11.3f} | "
          f"{np.median(m[:, 0]):9.3f} {np.median(m[:, 1]):9.3f}"
          f" {np.median(m[:, 2]):9.3f} | {np.median(m[:, 3]):9.2f}")
    print(f"{'worst':10s} {'':4s} {m[:, 4].min():11.3f} | "
          f"{m[:, 0].min():9.3f} {m[:, 1].min():9.3f} {m[:, 2].min():9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
