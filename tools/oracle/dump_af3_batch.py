"""Dump a model's FEATURISED BATCH, so tools/gpu/fold.js can fold that model.

    cd ~/alphafold3 && PYTHONPATH=src:.:dev/oracles \
      python dump_af3_batch.py protenix2 --sequence <SEQ>

🔴 WHY A SECOND MODEL NEEDS ITS OWN. `fold.js` reads oracle-dumps/af3-6mrr.json
and its banner says "featurised by AF3, read from the dump" - which is exactly
right and exactly the problem: featurisation is PER MODEL. Feeding protenix2
AlphaFold 3's target_feat reads relRMS 9.78e-1 against its own, and the fold
then runs every stage correctly on the wrong input and returns a structure whose
consecutive CA is 8.3 A against 3.8 expected. Every stage was right; the input
was another model's.

af3-any-model's `_fold_setup(model, seq)` applies each model's own conventions -
the restype alphabet, the empty-template restype, the element index shift - so
this asks it rather than reimplementing any of them.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REFERENCE = os.path.expanduser(os.environ.get("LOCALFOLD_AF3_REFERENCE", "~/alphafold3"))
SEQUENCE = "GSMKQIEDKIEE"

# 🔴 EVERY ARRAY LEAF, NOT A LIST OF NAMES. A hand-written key list missed the
# five atom-layout GATHERS and a dozen token flags, and `fold.js` then died on
# `Cannot read properties of undefined` rather than saying which feature was
# absent. The AF3 dump this stands beside carries 59 inputs; walking the batch
# the way dump_af3_trunk.py does reproduces that set exactly, including the
# `<name>:gather_idxs` / `:gather_mask` / `:input_shape` spelling a GatherInfo
# flattens to.


def arrays(obj, prefix=""):
    """(name, value) for every array leaf of a nested batch."""
    if isinstance(obj, dict):
        for key in sorted(obj):
            yield from arrays(obj[key], f"{prefix}/{key}" if prefix else key)
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from arrays(value, f"{prefix}[{index}]")
    elif hasattr(obj, "shape") and hasattr(obj, "dtype"):
        yield prefix, obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="protenix2")
    parser.add_argument("--sequence", default=SEQUENCE)
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    for entry in (os.path.join(arguments.reference, "src"), arguments.reference,
                  os.path.join(arguments.reference, "dev", "oracles")):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    import numpy as np
    from fold_check import _fold_setup
    from alphafold3.model import feat_batch

    batch, _cfg, _dir = _fold_setup(arguments.model, arguments.sequence)
    features = feat_batch.Batch.from_data_dict(batch)
    inputs = {}
    for name, value in arrays(batch):
        array = np.asarray(value)
        if array.dtype == object or array.ndim == 0:
            continue
        inputs[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                        "data": array.astype(np.float32).ravel().tolist()}
    # ...and the atom-layout gathers, which live on the Batch rather than in the
    # raw dict, under the flattened names the AF3 dump uses.
    cross = features.atom_cross_att
    for name in ("token_atoms_to_queries", "queries_to_keys", "queries_to_token_atoms",
                 "tokens_to_queries", "tokens_to_keys", "token_atoms_to_pseudo_beta"):
        gather = getattr(cross, name, None)
        if gather is None:
            gather = getattr(getattr(features, "pseudo_beta_info", None), name, None)
        if gather is None:
            continue
        for leaf in ("gather_idxs", "gather_mask", "input_shape"):
            value = getattr(gather, leaf, None)
            if value is None:
                continue
            array = np.asarray(value)
            inputs[f"{name}:{leaf}"] = {
                "shape": list(array.shape), "dtype": str(array.dtype),
                "data": array.astype(np.float32).ravel().tolist()}

    tokens = int(np.asarray(features.token_features.mask).shape[0])
    out = arguments.out or "/tmp/af3-batch-%s.json" % arguments.model
    with open(out, "w") as handle:
        json.dump({"model": arguments.model, "sequence": arguments.sequence,
                   "tokens": tokens,
                   "numMsa": int(np.asarray(batch["msa"]).shape[0]),
                   "source": "sokrypton/alphafold3 @ af3-any-model, _fold_setup",
                   "inputs": inputs, "outputs": {}}, handle)
    print("wrote %s  tokens=%d  msa=%d  %.1f MB"
          % (out, tokens, np.asarray(batch["msa"]).shape[0],
             os.path.getsize(out) / 2 ** 20))


if __name__ == "__main__":
    raise SystemExit(main())
