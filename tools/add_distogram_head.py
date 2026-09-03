"""Add AlphaFold 2's distogram head to an already-published bundle.

    python3 tools/add_distogram_head.py --model model \\
        --params ~/Documents/GitHub/af-params/params_model_1_ptm.npz
    python3 tools/add_distogram_head.py --model model-multimer \\
        --params ~/Documents/GitHub/af-params/params_model_1_multimer_v3.npz

WHY IT IS NOT PART OF THE EXPORT. tools/export-web-model.js builds a bundle by
re-sharding every tensor from the source manifest, so adding one head that way
rewrites all eight shards - 227 MB for the monomer - and everything published
has to be uploaded again. The head is 33 KB.

🔴 SO THE HEAD IS EMBEDDED IN THE MANIFEST, NOT SHARDED. It was a new shard
first, which is append-only and leaves the published bytes alone - and that
still broke every fold: the manifests are COMPILED INTO the page
(src/reference/manifests/), but the shards are fetched from a pinned remote,
so the moment the manifest declared a shard the remote did not have, every AF2
load asked for a 404 and the rejection took the model load down with it.
Base64 in the manifest has no such gap: 44 KB of text per model, carried by
the same file that declares the head, published the moment the page is.

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
import base64
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
        w = np.frombuffer(base64.b64decode(section["weights"]), dtype="<f4")
        b = np.frombuffer(base64.b64decode(section["bias"]), dtype="<f4")
        if list(w.shape) != [int(np.prod(section["weightsShape"]))]:
            print("FAIL: weights decode to %s, not %s" % (w.shape, section["weightsShape"]))
            return 1
        if list(b.shape) != section["biasShape"]:
            print("FAIL: bias decodes to %s, not %s" % (b.shape, section["biasShape"]))
            return 1
        print("ok: distogramHead embedded, %d weights and %d bias"
              % (w.size, b.size))
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

    # 🔴 REMOVE AN EARLIER SHARDED ATTEMPT, so re-running is idempotent and a
    # tree that tried the shard route does not keep a file nothing references.
    for stale in ("af2DistogramHalfLogitsWeights", "af2DistogramHalfLogitsBias"):
        entry = manifest["tensors"].pop(stale, None)
        if entry is not None and entry.get("file", "").startswith("weights-"):
            shard = root / entry["file"]
            index = int(entry["file"][len("weights-"):-len(".bin")])
            if shard.exists() and index >= int(manifest["bundle"]["shards"]) - 1:
                shard.unlink()
                manifest["bundle"]["shards"] = index
                manifest.get("shardDigests", {}).pop(entry["file"], None)

    payload = weights.tobytes() + bias.tobytes()
    manifest["distogramHead"] = {
        "encoding": "base64-float32-le",
        "weights": base64.b64encode(weights.tobytes()).decode("ascii"),
        "bias": base64.b64encode(bias.tobytes()).decode("ascii"),
        "weightsShape": list(weights.shape),
        "biasShape": list(bias.shape),
        "bins": BINS,
        "firstBreak": FIRST_BREAK,
        "lastBreak": LAST_BREAK,
        "note": "logits = half + half^T; breaks are linspace(firstBreak,"
                " lastBreak, bins - 1)",
    }
    manifest["bundle"]["tensors"] = len(manifest["tensors"])

    manifest_path.write_text(json.dumps(manifest))
    print("embedded the head in %s (%d bytes of weights, %d of base64)"
          % (manifest_path, len(payload),
             len(manifest["distogramHead"]["weights"])
             + len(manifest["distogramHead"]["bias"])))
    print("re-run tools/write_manifest_module.py for the committed copy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
