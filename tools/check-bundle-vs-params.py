#!/usr/bin/env python3
"""Is this bundle the weights af3-any-model loads, tensor by tensor?

    python3 tools/check-bundle-vs-params.py --bundle=model-boltz2-f32 \
      --digest=oracle-dumps/af3-params-digest-boltz2.json

🔴 THE ONE QUESTION EVERY OTHER CHECKER HERE ASSUMES THE ANSWER TO. Every AF3
parity tool feeds ONE bundle to both sides and compares two forwards, so a
tensor that is the wrong SIGN in the export is invisible to all of them - both
sides read it and both are wrong together. boltz2's
`diffusion_embed_pair_offsets` was exactly that: negated relative to the
reference's own params by a converter fix that landed after this bundle was
built, worth relRMS 2.02e-1 on a whole denoise step, and found only by fitting
the weight back out of a captured activation.

Reads the shards directly rather than through the loader, because the loader is
part of what this is checking. Reports, per tensor: MISSING, EXTRA, a shape
disagreement, NEGATED (rms equal, sum equal and opposite), or a value residual.
"""
import argparse, json, os
import numpy as np

DTYPES = {"float32": np.float32, "float16": np.float16, "bfloat16": None}


def read(bundle, record):
    dtype = record.get("dtype", "float32")
    count = int(np.prod(record["shape"])) if record["shape"] else 1
    path = os.path.join(bundle, record["file"])
    if dtype == "float32":
        a = np.fromfile(path, np.float32, count=count, offset=record["byteOffset"])
    elif dtype == "float16":
        a = np.fromfile(path, np.float16, count=count,
                        offset=record["byteOffset"]).astype(np.float32)
    elif dtype == "bfloat16":
        raw = np.fromfile(path, np.uint16, count=count, offset=record["byteOffset"])
        a = (raw.astype(np.uint32) << 16).view(np.float32)
    else:
        return None          # int5/int3/int8 need the block decoder; skip with a note
    return a.reshape(record["shape"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--digest", required=True)
    ap.add_argument("--bound", type=float, default=1e-5)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.bundle, "manifest.json")))
    tensors = manifest["tensors"]
    if not isinstance(tensors, dict):
        tensors = {t["name"]: t for t in tensors}
    want = json.load(open(args.digest))["tensors"]
    # `__meta__/*` is the params file's own provenance, not a weight: the bundle
    # carries that in its manifest instead.
    want = {k: v for k, v in want.items() if not k.startswith("__meta__/")}

    missing = sorted(set(want) - set(tensors))
    extra = sorted(set(tensors) - set(want))
    problems, skipped, ok = [], 0, 0
    for name in sorted(set(want) & set(tensors)):
        record, expect = tensors[name], want[name]
        if list(record["shape"]) != list(expect["shape"]):
            problems.append((9e9, name, "SHAPE %s vs %s"
                             % (record["shape"], expect["shape"])))
            continue
        got = read(args.bundle, record)
        if got is None:
            skipped += 1
            continue
        rms = float(np.sqrt((got.astype(np.float64) ** 2).mean()))
        total = float(got.astype(np.float64).sum())
        scale = max(expect["rms"], 1e-12)
        # 🔴 NEGATION IS ITS OWN VERDICT, not a large residual. It is what a
        # converter's sign convention looks like, and saying so names the fix.
        if (abs(rms - expect["rms"]) <= 1e-4 * scale
                and abs(total + expect["sum"]) <= 1e-3 * max(abs(expect["sum"]), 1.0)
                and abs(total - expect["sum"]) > 1e-3 * max(abs(expect["sum"]), 1.0)):
            problems.append((8e9, name, "NEGATED (rms %.6g both, sum %+.6g vs %+.6g)"
                             % (rms, total, expect["sum"])))
            continue
        head = np.asarray(expect["head"], np.float32)
        mine = got.ravel()[:len(head)]
        residual = float(np.sqrt(((mine - head) ** 2).sum()
                                 / max((head ** 2).sum(), 1e-30)))
        if residual > args.bound or abs(rms - expect["rms"]) > 1e-3 * scale:
            problems.append((residual, name,
                             "residual %.3e (rms %.6g vs %.6g)" % (residual, rms, expect["rms"])))
        else:
            ok += 1

    print("bundle   %s" % args.bundle)
    print("digest   %s (%d tensors)" % (args.digest, len(want)))
    print("agree    %d, differ %d, skipped (quantised) %d" % (ok, len(problems), skipped))
    if missing:
        print("MISSING from the bundle: %d, e.g. %s" % (len(missing), missing[:4]))
    if extra:
        print("EXTRA in the bundle:     %d, e.g. %s" % (len(extra), extra[:4]))
    for _, name, why in sorted(problems, reverse=True)[:args.top]:
        print("  %-88s %s" % (name, why))
    raise SystemExit(1 if problems or missing else 0)


if __name__ == "__main__":
    main()
