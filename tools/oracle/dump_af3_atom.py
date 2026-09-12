"""Dump AF3's atom-cross-attention LAYOUT, so the four atom checkers have an oracle.

    PYTHONPATH=src:. python tools/oracle/dump_af3_atom.py --out oracle-dumps/af3-oracle-atom-f32.json

🔴 WHY THIS EXISTS RATHER THAN dump_af3_trunk.py. Four checkers -
check-af3-atom-encoder, -atom-decoder, -diffusion-head and -sampler-gpu - all
fetch `/oracle-dumps/af3-oracle-atom-f32.json`, and NOTHING in this repository
produced it: `dump_af3_trunk.py` needs ColabDesign2's runner, which is not
installed on either machine here, and the reference this port now follows is
sokrypton/alphafold3 on `af3-any-model`. So the whole atom and diffusion path -
the encoder, the decoder, the head and the sampler - had no oracle check at all,
and each checker reported a 404 rather than a mismatch.

🔴 AND IT NEEDS NO FORWARD PASS, NO WEIGHTS AND NO GPU, which is why it is a
separate and much smaller thing. Read what the checkers actually consume and it
is nine arrays, every one of them FEATURISATION output:

    ref_pos  ref_space_uid  seq_mask  pred_dense_atom_mask
    token_atoms_to_queries  queries_to_keys  queries_to_token_atoms
    tokens_to_queries  tokens_to_keys          (each an idxs/mask pair)

The weights come from the bundle through `openAf3Store()`, and the activations
the checkers compare are generated deterministically on both sides. What they
cannot generate is the LAYOUT - which atoms share a window, which share a
reference conformer, which slots are padding - because a windowed local
attention cannot be exercised on synthetic input: the windows are built from
`ref_space_uid` and the atom ordering, and a hand-built layout is regular in
ways a real one is not. Two thirds of the key slots in a real batch are empty,
and that padding is where the interesting disagreements live.

🔴 THE SEQUENCE IS TWELVE RESIDUES ON PURPOSE, and matches dump_af3_trunk.py's.
The dense atom layout is 12x24, small enough to sit in a JSON file and be read
by eye when a kernel is wrong.
"""
import argparse
import json
import os
import pathlib
import sys

# CPU only: this is featurisation, and it must not take a GPU the reference
# machine may be using for something else.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# 🔴 af3-any-model IS THE REFERENCE, AND IT NEEDS ITS REPO ROOT ON THE PATH AS
# WELL AS src/. `dev/oracles/fold_check.py` imports `converters.pdb`, which
# lives at the root, so `PYTHONPATH=src` alone fails with a bare
# ModuleNotFoundError naming `converters` and looking like a missing package.
REFERENCE = os.path.expanduser(os.environ.get("LOCALFOLD_AF3_REFERENCE", "~/alphafold3"))

SEQUENCE = "GSMKQIEDKIEE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=SEQUENCE)
    parser.add_argument("--model", default="alphafold3",
                        help="any model af3-any-model's registry knows")
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    for entry in (os.path.join(arguments.reference, "src"),
                  arguments.reference,
                  os.path.join(arguments.reference, "dev", "oracles")):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    import numpy as np
    try:
        from fold_check import _fold_setup
        from alphafold3.model import feat_batch
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"could not import af3-any-model from {arguments.reference}: {error}.\n"
            "This wants a checkout of sokrypton/alphafold3 @ af3-any-model and a\n"
            "python with `alphafold3` installed (its compiled cpp extension included,\n"
            "which the CCD loader needs). Point --reference or\n"
            "LOCALFOLD_AF3_REFERENCE at the checkout.") from error

    batch, _config, _model_dir = _fold_setup(arguments.model, arguments.sequence)
    features = feat_batch.Batch.from_data_dict(batch)
    cross = features.atom_cross_att

    inputs = {}

    def put(name, array, dtype=None):
        array = np.asarray(array if dtype is None else np.asarray(array, dtype))
        inputs[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                        "data": array.ravel().tolist()}

    # 🔴 THE GATHERS GO IN UNDER THE NAMES THE CHECKERS ASK FOR, which are
    # `<name>:gather_idxs` and `<name>:gather_mask` - the flattened argument
    # names dump_af3_trunk.py's `--capture-args` would have produced for the
    # diffusion head. Keeping that spelling is what lets the four checkers stay
    # exactly as they are.
    for name in ("token_atoms_to_queries", "queries_to_keys", "queries_to_token_atoms",
                 "tokens_to_queries", "tokens_to_keys"):
        gather = getattr(cross, name)
        put(f"{name}:gather_idxs", gather.gather_idxs, np.int32)
        put(f"{name}:gather_mask", gather.gather_mask, np.float32)

    put("ref_pos", batch["ref_pos"], np.float32)
    put("ref_space_uid", batch["ref_space_uid"], np.int32)
    put("seq_mask", features.token_features.mask, np.float32)
    put("pred_dense_atom_mask", features.predicted_structure_info.atom_mask, np.float32)

    tokens = int(np.asarray(features.token_features.mask).shape[0])
    out = pathlib.Path(arguments.out) if arguments.out else \
        pathlib.Path("oracle-dumps") / "af3-oracle-atom-f32.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": arguments.model,
        "sequence": arguments.sequence,
        "tokens": tokens,
        "source": "sokrypton/alphafold3 @ af3-any-model",
        "inputs": inputs,
    }))
    print(f"wrote {out}  tokens={tokens}  inputs={len(inputs)}  "
          f"{out.stat().st_size / 2 ** 20:.1f} MB")


if __name__ == "__main__":
    raise SystemExit(main())
