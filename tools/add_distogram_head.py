"""Fold AlphaFold 2's distogram head into an exported bundle, as real tensors.

    python3 tools/add_distogram_head.py --model model \
        --params ~/Documents/GitHub/af-params/params_model_1_ptm.npz
    python3 tools/add_distogram_head.py --model model --check

WHAT THE HEAD IS. One linear projection of the 128-channel pair representation
to 64 distance bins, symmetrised as `logits + logits^T`, over breaks that run
2 to 22 A. AlphaFold 2 has always had it - it is what the contact map in every
AlphaFold figure is drawn from - and this repo simply never converted it,
because the bundle was built for the structure and the confidence heads.

🔴 IT LIVES IN THE SHARDS, LIKE EVERY OTHER TENSOR. It was base64 in the
manifest for a while, and before that a shard of its own; both were ways of
not rewriting published bytes. The cost was a bundle that was not the whole
model - a reader of `model/` got weights whose distogram head was somewhere
else, in a different encoding, reachable only through a special case in the
loader. The head is 33 KB against 227 MB, so it is appended to the LAST shard
and the earlier ones are untouched: only that shard and the manifest change,
and an upload transfers only what changed.

🔴 THE ORDER OF OPERATIONS IS NOT OPTIONAL, and getting it wrong breaks every
AF2 fold. The manifests are COMPILED INTO the page (src/reference/manifests/)
while the shards are fetched from a PINNED remote, so a manifest that
references bytes the pinned commit does not have makes every load ask for a
range that is not there. Upload the bundle first, re-pin the commit, and only
then regenerate the manifest module:

    python3 tools/add_distogram_head.py --model model --params ...
    hf upload USER/REPO model af2-monomer --repo-type=model
    # ...pin the new sha in src/reference/manifests/index.js
    python3 tools/write_manifest_module.py monomer
    python3 tools/check_remote_bundle.py monomer

🔴 FLOAT32, NOT QUANTISED. The rest of the bundle is int8 with per-block
scales; 33 KB is not worth a codec, and the store already reads float32
tensors - the PAE bin edges and the residue geometry tables are float32 in
these same shards.
"""
import argparse
import base64
import json
import pathlib

import numpy as np

WEIGHTS = "alphafold/alphafold_iteration/distogram_head/half_logits//weights"
BIAS = "alphafold/alphafold_iteration/distogram_head/half_logits//bias"

WEIGHTS_TENSOR = "af2DistogramHalfLogitsWeights"
BIAS_TENSOR = "af2DistogramHalfLogitsBias"

# AlphaFold 2's distogram: 64 bins with 63 breaks from 2 to 22 A. The head
# itself carries no bin edges, so they are written into the manifest here -
# the same shape the PAE breaks are stored in, so nothing downstream has to
# know a second convention.
FIRST_BREAK, LAST_BREAK, BINS = 2.0, 22.0, 64
CHANNELS = 128


def last_shard(manifest):
    """The highest-numbered shard any tensor is stored in."""
    files = {entry["file"] for entry in manifest["tensors"].values()
             if isinstance(entry, dict) and "file" in entry}
    return max(files, key=lambda name: int(name[len("weights-"):-len(".bin")]))


def read_head(root, manifest):
    """The head's two tensors, read back out of the shard they were put in."""
    section = manifest.get("distogramHead")
    if section is None:
        return None
    out = {}
    for key, name in (("weights", section.get("weights")),
                      ("bias", section.get("bias"))):
        entry = manifest["tensors"].get(name)
        if entry is None:
            return None
        data = (root / entry["file"]).read_bytes()
        count = int(np.prod(entry["shape"]))
        start = entry["byteOffset"]
        out[key] = np.frombuffer(data[start:start + count * 4], dtype="<f4")
        out[key + "_shape"] = entry["shape"]
    return out


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
        head = read_head(root, manifest)
        if head is None:
            print("FAIL: %s has no distogramHead in its tensors" % manifest_path)
            return 1
        if head["weights_shape"] != [CHANNELS, BINS] or head["bias_shape"] != [BINS]:
            print("FAIL: shapes are %s and %s" % (head["weights_shape"], head["bias_shape"]))
            return 1
        if head["weights"].size != CHANNELS * BINS or head["bias"].size != BINS:
            print("FAIL: read %d weights and %d bias"
                  % (head["weights"].size, head["bias"].size))
            return 1
        if not (np.isfinite(head["weights"]).all() and np.isfinite(head["bias"]).all()):
            print("FAIL: the head has non-finite values")
            return 1
        print("ok: distogramHead in the shards, %d weights and %d bias, |w| mean %.5f"
              % (head["weights"].size, head["bias"].size,
                 float(np.abs(head["weights"]).mean())))
        return 0

    if args.params is None:
        print("FAIL: --params is required unless --check")
        return 1
    params = np.load(args.params)
    if WEIGHTS not in params.files:
        print("FAIL: %s has no distogram head" % args.params)
        return 1
    weights = np.ascontiguousarray(params[WEIGHTS], dtype="<f4")
    bias = np.ascontiguousarray(params[BIAS], dtype="<f4")
    if weights.shape != (CHANNELS, BINS) or bias.shape != (BINS,):
        print("FAIL: unexpected shapes %s %s" % (weights.shape, bias.shape))
        return 1

    # 🔴 RE-RUNNING MUST NOT APPEND TWICE. An earlier run's tensors are dropped
    # from the manifest and the shard is truncated back to where they started,
    # so the bytes written are the same whether this is the first run or the
    # fifth - which is what makes the digests in the manifest module stable.
    existing = manifest["tensors"].get(WEIGHTS_TENSOR)
    if existing is not None:
        shard = root / existing["file"]
        with open(shard, "r+b") as handle:
            handle.truncate(existing["byteOffset"])
        manifest["tensors"].pop(WEIGHTS_TENSOR, None)
        manifest["tensors"].pop(BIAS_TENSOR, None)
    # ...and so is the base64 the head used to be carried as.
    manifest.pop("distogramHead", None)

    name = last_shard(manifest)
    shard = root / name
    offset = shard.stat().st_size
    with open(shard, "ab") as handle:
        handle.write(weights.tobytes())
        handle.write(bias.tobytes())

    manifest["tensors"][WEIGHTS_TENSOR] = {
        "file": name, "shape": [CHANNELS, BINS],
        "byteOffset": offset, "dtype": "float32",
    }
    manifest["tensors"][BIAS_TENSOR] = {
        "file": name, "shape": [BINS],
        "byteOffset": offset + weights.nbytes, "dtype": "float32",
    }
    manifest["distogramHead"] = {
        "weights": WEIGHTS_TENSOR,
        "bias": BIAS_TENSOR,
        "bins": BINS,
        "firstBreak": FIRST_BREAK,
        "lastBreak": LAST_BREAK,
        "note": "logits = half + half^T; breaks are linspace(firstBreak,"
                " lastBreak, bins - 1)",
    }
    manifest["bundle"]["tensors"] = len(manifest["tensors"])
    manifest["bundle"]["bytes"] = manifest["bundle"].get("bytes", 0)
    manifest["bundle"]["bytes"] += weights.nbytes + bias.nbytes
    manifest_path.write_text(json.dumps(manifest))

    print("wrote the head into %s at %d (+%d bytes); %s now has %d tensors"
          % (name, offset, weights.nbytes + bias.nbytes,
             manifest_path, len(manifest["tensors"])))
    print("upload the bundle and re-pin the commit BEFORE"
          " tools/write_manifest_module.py - see the note at the top")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
