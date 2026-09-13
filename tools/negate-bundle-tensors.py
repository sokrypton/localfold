#!/usr/bin/env python3
"""Negate NAMED tensors in a float32 bundle, in place.

    python3 tools/negate-bundle-tensors.py --bundle=model-boltz2-f32 \
      --only=diffuser/~/diffusion_head/diffusion_embed_pair_offsets/weights,...

🔴 FOR A CONVERTER SIGN THAT LANDED AFTER AN EXPORT, AND NOTHING ELSE. boltz2's
four `embed_pair_offsets` weights are the negative of the ones af3-any-model
loads: its converter now writes `ofs = -C.t(...)` because boltz computes
`d = keys - queries` where AF3 computes `queries - keys`, and this repository's
1.9 GB bundle predates that line. `check-bundle-vs-params.py` says the other 438
tensors agree exactly, so negating these four IS the re-export, and it can be
verified rather than trusted - run that checker before and after.

🔴 `--only` IS REQUIRED AND EXACT. A tool that reads a digest and rewrites
whatever disagrees would turn a wrong digest into wrong weights silently; the
names are typed by the caller and every one must be found.
"""
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--only", required=True, help="comma-separated tensor names")
    ap.add_argument("--apply", action="store_true", help="without this, only report")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.bundle, "manifest.json")))
    tensors = manifest["tensors"]
    if not isinstance(tensors, dict):
        tensors = {t["name"]: t for t in tensors}
    names = [n for n in args.only.split(",") if n]
    for name in names:
        if name not in tensors:
            raise SystemExit("not in the bundle: %s" % name)
        record = tensors[name]
        if record.get("dtype", "float32") != "float32":
            raise SystemExit("%s is %s, not float32" % (name, record.get("dtype")))

    for name in names:
        record = tensors[name]
        count = int(np.prod(record["shape"]))
        path = os.path.join(args.bundle, record["file"])
        before = np.fromfile(path, np.float32, count=count, offset=record["byteOffset"])
        print("%-88s %s sum %+.6g" % (name, record["shape"], float(before.sum())), end="")
        if not args.apply:
            print("   (dry run)")
            continue
        with open(path, "r+b") as handle:
            handle.seek(record["byteOffset"])
            handle.write((-before).astype(np.float32).tobytes())
        after = np.fromfile(path, np.float32, count=count, offset=record["byteOffset"])
        assert np.array_equal(after, -before), name
        print("   -> %+.6g" % float(after.sum()))


if __name__ == "__main__":
    main()
