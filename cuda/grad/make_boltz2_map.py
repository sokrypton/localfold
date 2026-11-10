"""Boltz-2's own parameters behind the bundle the port reads, for train-of3 --source.

The bundle (model-boltz2-f32) is af3-any-model's converter (converters/boltz2.py) applied to Boltz-2's checkpoint, and
the converter is not a renaming: it copies one tensor into several (the trunk's relative encoding into the diffusion
conditioning's, the atom transformers' one adaLN into a query and a key copy, the atom pair embeddings into their `_1`
twins), negates the pair offsets, halves the distogram's bias, permutes the MSA columns and zeroes one, writes zeros
for a reference-mask feature Boltz-2 does not have and ones for the LayerNorm scales it folds into the atom
transformers' pair-logit projections (each block's LN scale times its projection). Training the bundle's tensors as
free parameters would train a different, larger model; training Boltz-2 means its own tensors are the parameters and
the bundle's are a function of them.

This finds that function element by element, by running the converter itself on stand-in state dicts: every
tensor filled with its own number (which tensor), then with each element's position (which element), then all ones
(the scale), then random values twice (the check). A bundle element is then
  - a constant (the same in every run),
  - k * m[a]           a scaled copy of one Boltz element (k = 1, -1, 0.5), or
  - k * m[a] * m[b]    a product (a folded LayerNorm scale; found by holding one factor's tensors at one),
and every element so explained is checked against both random runs exactly. Anything else is refused.

Written to <out>/model.idx + model.bin (the trainer's loader format):
  i boltz2map.runs      the runs, 9 int32 each: bundle tensor, first element, count, kind (0 const, 1 copy, 2 product),
                        a source, a first, a step, b source (-1 none), b first   (b's step: see `bstep`)
  i boltz2map.bstep     each run's b step
  t boltz2map.k         each run's k (or its constant)
  t boltz2source.<i>    the values of the sources the port cannot give back: those no bundle tensor copies whole
                        (rows the MSA permutation drops, the LayerNorm offsets a softmax cancels, the B-factor head)
                        and every factor of a product (the folded LayerNorm scales and the projections they fold into)
  m boltz2source.count, and per source `m boltz2source.<i>.n <size>`; names.txt beside them, one per line: the
  bundle tensors, then the sources' Boltz keys. (Tied keys - Boltz-2's one parameter under two names - and the buffers
  are not sources; aliases.txt lists the tied ones.)

  ~/.venv-lfjax/bin/python cuda/grad/make_boltz2_map.py --ckpt ~/.boltz/boltz2_conf.ckpt \
      --af3am ~/beta/af3am-src --bundle model-boltz2-f32 --out cuda/grad/data/boltz2-map
(--af3am: a checkout of sokrypton/alphafold3, whose converters/boltz2.py made the bundle - checked first: the
converter on the real checkpoint must reproduce every bundle tensor exactly.)
"""
import argparse
import json
import math
import os
import pickle
import re
import sys
import types

import numpy as np

BUFFERS = {"confidence_module.boundaries"}


def load_state_dict(path):
    """Boltz-2's state dict, float32, with tied keys (one storage under two names) given one source: the checkpoint
    pickles omegaconf objects in its hyperparameters, so their classes are stubbed rather than installed."""
    import torch

    class Stub:
        def __init__(self, *a, **k): pass
        def __setstate__(self, s): pass

    class Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.startswith("omegaconf") or mod.startswith("boltz"):
                return Stub
            return super().find_class(mod, name)

    mod = types.ModuleType("stub_pickle"); mod.Unpickler = Unpickler; mod.load = pickle.load
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False, pickle_module=mod)
    sd = ck["state_dict"]
    # tied keys: Boltz-2's one parameter under two names - its transformer layers' gate is
    # `output_projection = Sequential(output_projection_linear, Sigmoid())`, the only sharing in the model
    # (Boltz2(**hparams).named_parameters(remove_duplicate=False): 33 such pairs, nothing else); the checkpoint wrote
    # the two as equal copies. And the buffers, which are not trained (confidence_module.boundaries).
    canon, alias = {}, {}
    for k, v in sd.items():
        if k in BUFFERS: continue
        m = re.match(r"(.*)\.output_projection\.0\.(weight|bias)$", k)
        if m and f"{m[1]}.output_projection_linear.{m[2]}" in sd:
            c = f"{m[1]}.output_projection_linear.{m[2]}"
            if not torch.equal(v, sd[c]): sys.exit(f"{k} is tied to {c} in Boltz-2 and differs from it here")
            alias[k] = c
        else: canon[k] = v.float().numpy()
    return canon, alias, {b: sd[b].float().numpy() for b in BUFFERS}


def toRuns(kind, kk, a, ai, b, bi):
    """Elements as runs: the same kind, k and sources, each source's element stepping by a constant. A run starts at
    an element whose kind, k or sources differ from the last's (a hard start), or whose step differs from the last
    one's - but a run's second element sets its step, so of a chain of consecutive step changes only every other one
    starts a run. Checked against the elements before returning."""
    N = kind.size
    hard = np.ones(N, bool)
    if N > 1:
        hard[1:] = (kind[1:] != kind[:-1]) | (kk[1:] != kk[:-1]) | (a[1:] != a[:-1]) | (b[1:] != b[:-1])
    da = np.zeros(N, np.int64); db = np.zeros(N, np.int64)
    da[1:] = ai[1:] - ai[:-1]; db[1:] = bi[1:] - bi[:-1]
    soft = np.zeros(N, bool)
    if N > 2:
        soft[2:] = ~hard[2:] & ~hard[1:-1] & ((da[2:] != da[1:-1]) | (db[2:] != db[1:-1]))
    start = hard.copy()
    # the alternation, chain by chain (a chain: consecutive soft breaks; it is short - a transpose's row ends are
    # isolated - so a loop over chains, not elements)
    js = np.flatnonzero(soft)
    if js.size:
        cut = np.flatnonzero(np.diff(js) != 1) + 1
        for chain in np.split(js, cut):
            first = chain[0]
            # the chain's first is a start unless the element before it began a run (then it is that run's second)
            prevStart = start[first - 1]
            for t, j in enumerate(chain):
                start[j] = (t % 2 == 0) != prevStart
    st = np.flatnonzero(start); en = np.append(st[1:], N)
    cnt = en - st
    astep = np.where(cnt > 1, ai[np.minimum(st + 1, N - 1)] - ai[st], 0)
    bstep = np.where(cnt > 1, bi[np.minimum(st + 1, N - 1)] - bi[st], 0)
    # the check: every element from its run
    rid = np.repeat(np.arange(st.size), cnt); off = np.arange(N) - st[rid]
    ok = (ai == ai[st][rid] + astep[rid] * off) & (bi == bi[st][rid] + bstep[rid] * off) & (a == a[st][rid]) & \
         (b == b[st][rid]) & (kind == kind[st][rid]) & (kk == kk[st][rid])
    if not ok.all(): sys.exit(f"the runs do not reproduce element {np.flatnonzero(~ok)[0]}")
    return [(int(s), int(c), int(kind[s]), int(a[s]), int(ai[s]), int(sa), int(b[s]), int(bi[s]), float(kk[s]), int(sb))
            for s, c, sa, sb in zip(st, cnt, astep, bstep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--af3am", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.af3am); sys.path.insert(0, os.path.join(args.af3am, "src"))
    from converters import boltz2 as B

    sd, alias, buffers = load_state_dict(os.path.expanduser(args.ckpt))
    keys = list(sd); shapes = {k: sd[k].shape for k in keys}; sizes = [sd[k].size for k in keys]
    manifest = json.load(open(os.path.join(args.bundle, "manifest.json")))["tensors"]
    names = list(manifest)

    def run(values):
        full = dict(values); full.update({a: values[c] for a, c in alias.items()})
        full.update({b: buffers[b] for b in BUFFERS})
        P = B.map_boltz2_to_af3(full)
        flat = {s + "/" + k: np.asarray(v, np.float64).ravel() for s, d in P.items() for k, v in d.items()}
        missing = [n for n in names if n not in flat]
        if missing: sys.exit(f"the converter does not write {len(missing)} bundle tensors, e.g. {missing[0]}")
        return {n: flat[n] for n in names}

    # the converter on the real checkpoint must be the bundle
    real = run(sd)
    files = {}
    for n in names:
        t = manifest[n]
        if t["file"] not in files: files[t["file"]] = np.memmap(os.path.join(args.bundle, t["file"]), dtype=np.float32, mode="r")
        a = np.asarray(files[t["file"]][t["byteOffset"] // 4: t["byteOffset"] // 4 + math.prod(t["shape"])], np.float64)
        if a.size != real[n].size or not np.array_equal(a, real[n].astype(np.float32).astype(np.float64)):
            sys.exit(f"{n}: the converter on {args.ckpt} is not {args.bundle} - another export, or another converter")
    del real
    print(f"the converter reproduces all {len(names)} bundle tensors from {len(keys)} Boltz-2 tensors ({len(alias)} tied keys)")

    fold = re.compile(r"atom_(enc|dec)_proj_z\.\d+\.0\.weight$")    # (the LN scales the converter folds)
    ident = run({k: np.full(shapes[k], i + 1, np.float32) for i, k in enumerate(keys)})
    index = run({k: (np.arange(sizes[i], dtype=np.float32) + 1).reshape(shapes[k]) for i, k in enumerate(keys)})
    ones = run({k: np.ones(shapes[k], np.float32) for k in keys})
    # a product's two factors: the folded scales held at one (the other factor's id and index), then the rest
    identL = run({k: np.ones(shapes[k], np.float32) if fold.search(k) else np.full(shapes[k], i + 1, np.float32) for i, k in enumerate(keys)})
    indexL = run({k: np.ones(shapes[k], np.float32) if fold.search(k) else (np.arange(sizes[i], dtype=np.float32) + 1).reshape(shapes[k]) for i, k in enumerate(keys)})
    identS = run({k: np.full(shapes[k], i + 1, np.float32) if fold.search(k) else np.ones(shapes[k], np.float32) for i, k in enumerate(keys)})
    indexS = run({k: (np.arange(sizes[i], dtype=np.float32) + 1).reshape(shapes[k]) if fold.search(k) else np.ones(shapes[k], np.float32) for i, k in enumerate(keys)})
    rng = np.random.default_rng(0)
    R1 = {k: rng.standard_normal(shapes[k]).astype(np.float32) for k in keys}
    R2 = {k: (rng.standard_normal(shapes[k]) * 3 + 1).astype(np.float32) for k in keys}
    c1, c2 = run(R1), run(R2)
    flat1 = [R1[k].ravel().astype(np.float64) for k in keys]; flat2 = [R2[k].ravel().astype(np.float64) for k in keys]

    def decode(idv, ixv, kv):
        """(source, element) from an id run and an index run of k * m[a]; -1 where not a whole number"""
        with np.errstate(divide="ignore", invalid="ignore"):
            s, i = idv / kv - 1, ixv / kv - 1
        good = (kv != 0) & (s == np.rint(s)) & (i == np.rint(i)) & (s >= 0) & (s < len(keys)) & (i >= 0)
        s = np.where(good, np.rint(s), -1).astype(np.int64); i = np.where(good, np.rint(i), -1).astype(np.int64)
        ok = good.copy()
        for m in np.unique(s[good]):
            sel = good & (s == m); ok[sel] &= i[sel] < sizes[m]
        return np.where(ok, s, -1), np.where(ok, i, -1)

    def value(src, idx, flats):
        out = np.zeros(src.size)
        for m in np.unique(src[src >= 0]):
            sel = src == m; out[sel] = flats[m][idx[sel]]
        return out

    runs, kvals, bsteps = [], [], []
    tally = {"const": 0, "copy": 0, "product": 0}
    for ti, n in enumerate(names):
        k = ones[n]
        a, ai = decode(ident[n], index[n], k)
        copy = (a >= 0) & (value(a, ai, flat1) * k == c1[n]) & (value(a, ai, flat2) * k == c2[n])
        const = ~copy & (ident[n] == index[n]) & (index[n] == c1[n]) & (c1[n] == c2[n])
        rest = ~copy & ~const
        b, bi = np.full(k.size, -1, np.int64), np.full(k.size, -1, np.int64)
        if rest.any():
            la, lai = decode(identL[n], indexL[n], k)
            sa, sai = decode(identS[n], indexS[n], k)
            prod = rest & (la >= 0) & (sa >= 0)
            v1 = value(la, lai, flat1) * value(sa, sai, flat1) * k; v2 = value(la, lai, flat2) * value(sa, sai, flat2) * k
            # (the converter multiplies in float32: a product is held to its rounding)
            prod &= (np.abs(v1 - c1[n]) <= 1e-6 * np.abs(v1) + 1e-30) & (np.abs(v2 - c2[n]) <= 1e-6 * np.abs(v2) + 1e-30)
            if (rest & ~prod).any():
                j = np.flatnonzero(rest & ~prod)[0]
                sys.exit(f"{n}[{j}]: neither a constant, a copy nor a product of two Boltz-2 elements")
            a = np.where(prod, la, a); ai = np.where(prod, lai, ai); b = np.where(prod, sa, -1); bi = np.where(prod, sai, -1)
        kind = np.where(const, 0, np.where(b >= 0, 2, 1))
        kk = np.where(const, c1[n], k).astype(np.float32)
        tally["const"] += int(const.sum()); tally["copy"] += int((kind == 1).sum()); tally["product"] += int((kind == 2).sum())
        for r in toRuns(kind, kk, a, ai, b, bi):
            runs.append((ti,) + r[:8]); kvals.append(r[8]); bsteps.append(r[9])
    R = np.array(runs, np.int64)
    if (np.abs(R) > 2**31 - 1).any(): sys.exit("a run does not fit int32")
    print(f"{tally['copy']} copied, {tally['product']} products, {tally['const']} constants: {len(runs)} runs")

    # the sources no bundle tensor holds as a plain copy of k != 0: their values come with the map
    # (per element: a source the bundle copies only in part - rows the converter's column permutation drops - is
    # carried whole, and its unread elements reported, since the port's graph can give them no gradient)
    seen = [np.zeros(sz, bool) for sz in sizes]
    for r, kv in zip(runs, kvals):
        if r[3] == 1 and kv != 0: seen[r[4]][r[5] + r[6] * np.arange(r[2])] = True
    covered = np.array([sv.all() for sv in seen])
    for i, sv in enumerate(seen):
        if sv.any() and not sv.all():
            un = np.flatnonzero(~sv); print(f"{keys[i]} {tuple(shapes[keys[i]])}: {un.size} elements no bundle tensor copies (flat {un[:4].tolist()}...)")
    # (a factor of a product is carried even where a bundle tensor copies it: the port may read the product alone - the
    # token transformer's pair bias, whose LN scale and projection the weight walk multiplies - and a product does
    # not give back its factors)
    for r in runs:
        if r[3] == 2: covered[r[4]] = covered[r[7]] = False
    # ...and the weight walk (cuda/featurise/af3_weights.h) folds the token transformer's pair LayerNorm scales into its
    # pair-logit projections the same way, a step this script does not see: those two bundle tensors' sources carried
    # too (the trainer refuses to start if a source is still set by nothing)
    for r in runs:
        if r[3] >= 1 and re.search(r"pair_input_layer_norm/scale$|pair_logits_projection/weights$", names[r[0]]): covered[r[4]] = False
    os.makedirs(args.out, exist_ok=True)
    blob = bytearray(); idx = []
    def put(kind, name, arr):
        arr = np.ascontiguousarray(arr); off = len(blob) // 4
        blob.extend(arr.astype(np.int32 if kind == "i" else np.float32).tobytes()); idx.append(f"{kind} {name} {off} {arr.size}")
    put("i", "boltz2map.runs", R.astype(np.int32).ravel()); put("i", "boltz2map.bstep", np.array(bsteps, np.int32))
    put("t", "boltz2map.k", np.array(kvals, np.float32))
    idx.append(f"m boltz2map.bundleTensors {len(names)}"); idx.append(f"m boltz2source.count {len(keys)}")
    for i, k in enumerate(keys):
        idx.append(f"m boltz2source.{i}.n {sizes[i]}")
        if not covered[i]: put("t", f"boltz2source.{i}", sd[k].ravel())
    with open(os.path.join(args.out, "model.bin"), "wb") as f: f.write(blob)
    with open(os.path.join(args.out, "model.idx"), "w") as f: f.write("\n".join(idx) + "\n")
    with open(os.path.join(args.out, "names.txt"), "w") as f: f.write("\n".join(names + keys) + "\n")
    print(f"{int((~covered).sum())} sources carried in the map ({sum(sizes[i] for i in range(len(keys)) if not covered[i])} values);"
          f" {len(alias)} tied keys: {sorted(alias.items())[:3]}...")
    with open(os.path.join(args.out, "aliases.txt"), "w") as f: f.write("".join(f"{a} {c}\n" for a, c in sorted(alias.items())))


if __name__ == "__main__":
    main()
