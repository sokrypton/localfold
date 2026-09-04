"""Fit a distogram -> pLDDT map on AF2's own per-recycle confidence.

    node tools/gpu-chrome.mjs tools/gpu/probe-af2-dgram-plddt.js \
        --from=0 --to=12 --recycles=3 > af2/chunk-0.json     # collect
    python3 tools/fit-distogram-plddt.py features af2/chunk-*.json
    python3 tools/fit-distogram-plddt.py fit                 # LOTO scores

🔴 WHY AF2 ANSWERS A QUESTION ASKED ABOUT AF3. The page draws trunk previews
and sampler frames before AF3's confidence head has run, so it has to colour
them with an estimate. AF2's head runs on EVERY recycle, which makes it the
only source of (distogram, structure, pLDDT) triples that are labelled and
real. The map is fitted there and read as evidence about the SHAPE of the
estimator; the two calibration constants do not transfer - AF2 wants
1.33 + 1.20x and AF3 41.29 + 0.578x for the same feature.

🔴 SINGLE SEQUENCE, DELIBERATELY. An alignment folds almost everything and a
set where every label is 85 teaches that the answer is always 85. Query-only
over 108 natives gives per-target mean pLDDT from 29.7 to 91.5, 28 targets
below 40 and 16 above 70.

WHAT IT FOUND, over 44,740 residue-recycle rows, leave-one-TARGET-out:

    shipped estimator, uncalibrated   RMSE 10.44
    the same + a two-number affine          4.53
    ten features, ridge                     4.18
    all 89 features, ridge                  3.91

🔴 SO ALMOST ALL OF IT IS CALIBRATION. Two numbers take 10.44 to 4.53 and
eighty-nine features buy 0.6 more - which is the argument for leaving the live
estimator lDDT-shaped rather than shipping a fitted vector whose constants are
per-model anyway. For scale, this predicts AF2's pLDDT to 4.53 where AF2's
pLDDT predicts a crystal structure to 13.44.
"""
import base64
import json
import sys

import numpy as np

EBINS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
NEAR = 15.0
CACHE = "features.npz"

NAMES = ([f"mass{b}" for b in range(64)]
         + [f"agree{t}" for t in EBINS] + ["within"]
         + [f"near{t}" for t in EBINS]
         + [f"ragree{t}" for t in EBINS]
         + [f"rnear{t}" for t in EBINS])
# the four lDDT thresholds as a fraction of the mass inside the radius: the
# shipped estimator, up to its affine
LDDT4 = [NAMES.index(f"rnear{t}") for t in (0.5, 1.0, 2.0, 4.0)]


def bin_centres(entry):
    bins = entry["bins"]
    breaks = np.linspace(entry["firstBreak"], entry["lastBreak"], bins - 1)
    step = breaks[1] - breaks[0]
    return np.concatenate([[breaks[0] - step / 2],
                           (breaks[:-1] + breaks[1:]) / 2,
                           [breaks[-1] + step / 2]])


def features(entry, key, centres):
    """Raw per-residue sums over the distogram and the model's own distances.

    🔴 ONE BIN AT A TIME. The dense form is tokens^2 x 64 twice over - the
    distogram and the error against every bin centre - which is gigabytes on a
    300-residue target. Accumulating bin by bin is the same arithmetic in
    tokens^2 floats.
    """
    n = len(entry["sequence"])
    bins = entry["bins"]
    raw = np.frombuffer(base64.b64decode(entry["distograms"][key]), np.uint8)
    probability = raw.reshape(-1, bins).astype(np.float32)
    probability /= np.maximum(probability.sum(1, keepdims=True), 1e-6)
    upper = np.triu_indices(n)

    model = np.asarray(entry["models"][key], np.float32)
    distances = np.linalg.norm(model[:, None] - model[None], axis=-1)
    off = ~np.eye(n, dtype=bool)

    mass = np.zeros((n, bins), np.float32)
    agree = np.zeros((n, len(EBINS)), np.float32)
    agree_near = np.zeros((n, len(EBINS)), np.float32)
    within = np.zeros(n, np.float32)
    pair = np.zeros((n, n), np.float32)
    for b in range(bins):
        pair[:] = 0
        pair[upper] = probability[:, b]
        pair += pair.T
        pair[np.arange(n), np.arange(n)] /= 2
        pair *= off
        mass[:, b] = pair.sum(1)
        error = np.abs(distances - centres[b])
        near = centres[b] < NEAR
        if near:
            within += mass[:, b]
        for t, threshold in enumerate(EBINS):
            hit = (pair * (error < threshold)).sum(1)
            agree[:, t] += hit
            if near:
                agree_near[:, t] += hit
    denominator = np.maximum(within[:, None], 1e-6)
    return np.c_[mass, agree, within, agree_near,
                 agree / denominator, agree_near / denominator]


def build(paths):
    X, plddt, true, groups, recycles = [], [], [], [], []
    for path in paths:
        text = open(path).read()
        # 🔴 A CRASHED CHUNK IS A LOG, NOT JSON. Chrome dies somewhere above
        # fifteen folds in one process and says so on stdout, so a chunk that
        # never reported has to be skipped rather than parsed.
        if not text.lstrip().startswith("{"):
            print("skip (crashed)", path)
            continue
        dump = json.loads(text)
        label = {}
        for row in dump["rows"]:
            label.setdefault((row[0], row[1]), {})[row[2]] = (row[3], row[4])
        for name, entry in dump["data"].items():
            centres = bin_centres(entry)
            for key in sorted(entry["models"], key=int):
                seen = label.get((name, int(key)))
                length = len(entry["sequence"])
                if seen is None or len(seen) != length:
                    continue
                X.append(features(entry, key, centres))
                plddt.append([seen[i][0] for i in range(length)])
                true.append([seen[i][1] for i in range(length)])
                groups += [name] * length
                recycles += [int(key)] * length
        print("%-22s %3d targets  %6d rows"
              % (path.split("/")[-1], len(dump["data"]),
                 sum(a.shape[0] for a in X)), flush=True)
    np.savez_compressed(CACHE, X=np.vstack(X).astype(np.float32),
                        plddt=np.concatenate(plddt), true=np.concatenate(true),
                        groups=np.array(groups), recycle=np.array(recycles))
    print("design matrix", np.vstack(X).shape, "->", CACHE)


def fit(column):
    cached = np.load(CACHE, allow_pickle=False)
    X, groups = cached["X"].astype(np.float64), cached["groups"]
    y = cached[column].astype(np.float64)
    targets = np.unique(groups)
    mean, spread = X.mean(0), np.maximum(X.std(0), 1e-9)
    Z = np.c_[np.ones(len(y)), (X - mean) / spread]
    print("rows %d  targets %d  target %s  mean %.1f sd %.1f"
          % (len(y), len(targets), column, y.mean(), y.std()))

    # 🔴 LOTO BY GRAM DOWNDATE. Greedy selection is 108 folds x 89 candidates x
    # 12 steps of least squares; accumulating each target's own A^T A once and
    # subtracting it makes every fold a 90x90 solve.
    where = {t: np.flatnonzero(groups == t) for t in targets}
    gram = {t: Z[i].T @ Z[i] for t, i in where.items()}
    moment = {t: Z[i].T @ y[i] for t, i in where.items()}
    total, rhs = sum(gram.values()), sum(moment.values())

    def loto(columns, alpha=10.0):
        keep = np.array([0] + [c + 1 for c in columns])
        ridge = alpha * np.eye(len(keep))
        ridge[0, 0] = 0
        out = np.empty_like(y)
        for t in targets:
            left = total[np.ix_(keep, keep)] - gram[t][np.ix_(keep, keep)]
            w = np.linalg.solve(left + ridge, rhs[keep] - moment[t][keep])
            out[where[t]] = Z[np.ix_(where[t], keep)] @ w
        return np.clip(out, 0, 100)

    def rmse(p):
        return float(np.sqrt(np.mean((p - y) ** 2)))

    def score(name, p):
        worst = max(np.sqrt(np.mean((p[where[t]] - y[where[t]]) ** 2))
                    for t in targets)
        print("%-34s RMSE %6.2f  MAE %6.2f  worst target %6.2f"
              % (name, rmse(p), np.mean(np.abs(p - y)), worst))

    every = list(range(len(NAMES)))
    score("shipped shape, uncalibrated", 100 * X[:, LDDT4].mean(1))
    score("shipped shape + affine (LOTO)", loto(LDDT4))
    for alpha in (1.0, 10.0, 100.0):
        score("all %d features  a=%-6g" % (len(NAMES), alpha), loto(every, alpha))

    chosen = []
    for _ in range(12):
        pick = min((c for c in every if c not in chosen),
                   key=lambda c: rmse(loto(chosen + [c])))
        chosen.append(pick)
        print("%2d  +%-10s RMSE %6.2f" % (len(chosen), NAMES[pick], rmse(loto(chosen))),
              flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "features":
        build(sys.argv[2:])
    else:
        fit(sys.argv[2] if len(sys.argv) > 2 else "plddt")
