"""Add AlphaFold 2's distogram head to an already-published bundle.

    python3 tools/add_distogram_head.py --model model \\
        --params ~/Documents/GitHub/af-params/params_model_1_ptm.npz
    python3 tools/add_distogram_head.py --model model-multimer \\
        --params ~/Documents/GitHub/af-params/params_model_1_multimer_v3.npz

WHY IT IS NOT PART OF THE EXPORT. tools/export-web-model.js builds a bundle by
re-sharding every tensor from the source manifest, so adding one head that way
rewrites all eight shards - 227 MB for the monomer - and everything published
has to be uploaded again. The head is 33 KB.

🔴 SO THIS IS APPEND-ONLY, AND THE BUNDLE FORMAT ALREADY ALLOWS IT. Every
tensor entry names its own `file` and `byteOffset`, so a tensor in a NEW shard
is reachable without touching the existing ones: the published shards keep
their bytes and their digests, and what has to be uploaded is one small file
plus the manifest.

WHAT THE HEAD IS. One linear projection of the 128-channel pair representation
to 64 distance bins, symmetrised as `logits + logits^T`, over breaks that run
2 to 22 A. AlphaFold 2 has always had it - it is what the contact map in every
AlphaFold figure is drawn from - and this repo simply never converted it,
because the bundle was built for the structure and the confidence heads.

🔴 FLOAT32, NOT QUANTISED. The rest of the bundle is int8 with per-block
scales; 33 KB is not worth a codec, and the store already reads float32
tensors - the PAE bin edges and the residue geometry tables are float32 in
these same shards.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

WEIGHTS = "alphafold/alphafold_iteration/distogram_head/half_logits//weights"
BIAS = "alphafold/alphafold_iteration/distogram_head/half_logits//bias"

# AlphaFold 2's distogram: 64 bins with 63 breaks from 2 to 22 A. The head
# itself carries no bin edges, so they are written into the manifest here -
# the same shape the PAE breaks are stored in, so nothing downstream has to
# know a second convention.
FIRST_BREAK, LAST_BREAK, BINS = 2.0, 22.0, 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="the bundle directory to extend, e.g. model/")
    parser.add_argument("--params",
                        help="the AlphaFold .npz the head is taken from;"
                             " not needed with --check")
    parser.add_argument("--check", action="store_true",
                        help="verify an already-added head instead of writing")
    args = parser.parse_args()

    root = pathlib.Path(args.model)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    if args.check:
        section = manifest.get("distogramHead")
        if section is None:
            print("FAIL: no distogramHead section in %s" % manifest_path)
            return 1
        for name in section["parameters"].values():
            entry = manifest["tensors"][name]
            blob = (root / entry["file"]).read_bytes()
            end = entry["byteOffset"] + 4 * int(np.prod(entry["shape"]))
            if end > len(blob):
                print("FAIL: %s runs past the end of %s" % (name, entry["file"]))
                return 1
        print("ok: distogramHead present and in range")
        return 0

    if args.params is None:
        print("FAIL: --params is required unless --check")
        return 1
    params = np.load(args.params)
    if WEIGHTS not in params.files:
        print("FAIL: %s has no distogram head" % args.params)
        return 1
    weights = np.asarray(params[WEIGHTS], dtype="<f4")
    bias = np.asarray(params[BIAS], dtype="<f4")
    if weights.shape != (128, BINS) or bias.shape != (BINS,):
        print("FAIL: unexpected shapes %s %s" % (weights.shape, bias.shape))
        return 1

    # 🔴 A NEW SHARD, NUMBERED PAST THE LAST ONE. Writing into an existing
    # shard would change bytes that are already published and already
    # digested, which is the one thing this script exists to avoid.
    shard_index = int(manifest["bundle"]["shards"])
    shard_name = "weights-%02d.bin" % shard_index
    payload = weights.tobytes() + bias.tobytes()
    (root / shard_name).write_bytes(payload)

    manifest["tensors"]["af2DistogramHalfLogitsWeights"] = {
        "file": shard_name, "shape": list(weights.shape),
        "byteOffset": 0, "dtype": "float32",
    }
    manifest["tensors"]["af2DistogramHalfLogitsBias"] = {
        "file": shard_name, "shape": list(bias.shape),
        "byteOffset": weights.nbytes, "dtype": "float32",
    }
    manifest["distogramHead"] = {
        "parameters": {
            "halfLogitsWeights": "af2DistogramHalfLogitsWeights",
            "halfLogitsBias": "af2DistogramHalfLogitsBias",
        },
        "bins": BINS,
        "firstBreak": FIRST_BREAK,
        "lastBreak": LAST_BREAK,
        "note": "logits = half + half^T; breaks are linspace(firstBreak,"
                " lastBreak, bins - 1)",
    }
    manifest["bundle"]["shards"] = shard_index + 1
    manifest["bundle"]["bytes"] = int(manifest["bundle"].get("bytes", 0)) + len(payload)
    manifest["bundle"]["tensors"] = len(manifest["tensors"])
    # 🔴 NOT EVERY BUNDLE HAS DIGESTS. The monomer manifest carries
    # shardDigests and tools/build_site.py checks them; the multimer's does
    # not. Adding the key to a bundle that never had it would make the site
    # build start checking shards it has no digests for.
    if "shardDigests" in manifest:
        manifest["shardDigests"][shard_name] = hashlib.sha256(payload).hexdigest()

    manifest_path.write_text(json.dumps(manifest))
    print("wrote %s (%d bytes) and extended %s"
          % (root / shard_name, len(payload), manifest_path))
    print("shards now %d; re-run tools/write_manifest_module.py for the"
          " committed copy" % manifest["bundle"]["shards"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
