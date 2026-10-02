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
    # 🔴 THE CONVENTIONS A PLAIN PROTEIN CANNOT EXERCISE. 6MRR has no
    # ligand and no modified residue, so `symmetriseBonds`,
    # `atomized_element_names`, `atomized_backbone_bonds` and
    # `atomized_unknown_restype` are all inert on it - and a featuriser gate
    # built on that target reports them green by never reaching them.
    # `--ligand GOL` adds a ligand chain and `--ptm SEP@3` a modified residue,
    # both through `_fold_setup(chains=...)`, so every per-model convention
    # still comes from the reference rather than from this script.
    parser.add_argument("--ligand", default=None,
                        help="CCD code of a ligand chain to add, e.g. GOL; several, comma-separated, "
                             "are ONE chain of bonded components (a glycan: NAG,NAG,BMA,MAN,MAN)")
    parser.add_argument("--bonds", default=None,
                        help="bondedAtomPairs as CHAIN:RESIDUE:ATOM-CHAIN:RESIDUE:ATOM;... "
                             "(1-based residues, as AF3's job file writes them)")
    parser.add_argument("--ptm", default=None,
                        help="modified residue as CODE@POSITION, e.g. SEP@3 (1-based)")
    # a nucleic chain and its modified bases: `--kind dna --mods 5CM@5,5CM@9`
    parser.add_argument("--kind", default="protein", choices=["protein", "dna", "rna"])
    parser.add_argument("--mods", default=None,
                        help="modified bases or residues as CODE@POSITION[,...] (1-based)")
    # components the installed dictionary lacks (af3-any-model's pip install carries a minimal one,
    # without SEP or any modified base), as mmCIF text - what AF3's userCCD field takes
    parser.add_argument("--user-ccd", default=None, action="append",
                        help="a component's CCD mmCIF file to add to the dictionary (repeatable)")
    arguments = parser.parse_args()
    for entry in (os.path.join(arguments.reference, "src"), arguments.reference,
                  os.path.join(arguments.reference, "dev", "oracles")):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    import numpy as np
    if arguments.user_ccd:
        import functools
        from alphafold3.constants import decoded_ccd
        text = "\n".join(open(path).read() for path in arguments.user_ccd)
        decoded_ccd.get_ccd = functools.partial(decoded_ccd.get_ccd, user_ccd=text)
    from fold_check import _fold_setup
    from alphafold3.model import feat_batch

    chains = None
    if arguments.kind != "protein" or arguments.mods is not None:
        from alphafold3.common import folding_input
        mods = []
        for spec in (arguments.mods or "").split(","):
            if spec:
                code, _, position = spec.partition("@")
                mods.append((code.strip().upper(), int(position)))
        if arguments.kind == "protein":
            chain = folding_input.ProteinChain(id="A", sequence=arguments.sequence, ptms=mods,
                                               unpaired_msa="", paired_msa="", templates=[])
        elif arguments.kind == "dna":
            chain = folding_input.DnaChain(id="A", sequence=arguments.sequence, modifications=mods)
        else:
            chain = folding_input.RnaChain(id="A", sequence=arguments.sequence, modifications=mods,
                                           unpaired_msa="")
        chains = [chain]
        if arguments.ligand is not None:
            chains.append(folding_input.Ligand(id="B", ccd_ids=[c.strip().upper() for c in arguments.ligand.split(",")]))
    elif arguments.ligand is not None or arguments.ptm is not None:
        from alphafold3.common import folding_input
        # `ptms` is a sequence of (CODE, 1-based position) tuples and a ligand
        # is `folding_input.Ligand`, not a "LigandChain" - checked against the
        # reference's own signatures rather than guessed.
        ptms = []
        if arguments.ptm is not None:
            code, _, position = arguments.ptm.partition("@")
            ptms = [(code.strip().upper(), int(position))]
        chains = [folding_input.ProteinChain(
            id="A", sequence=arguments.sequence, ptms=ptms,
            unpaired_msa="", paired_msa="", templates=[])]
        if arguments.ligand is not None:
            chains.append(folding_input.Ligand(
                id="B", ccd_ids=[c.strip().upper() for c in arguments.ligand.split(",")]))
    bonds = None
    if arguments.bonds:
        bonds = []
        for pair in arguments.bonds.split(";"):
            ends = []
            for end in pair.split("-"):
                chain, residue, atom = end.split(":")
                ends.append((chain, int(residue), atom))
            bonds.append(tuple(ends))
    batch, _cfg, _dir = _fold_setup(arguments.model, arguments.sequence, chains=chains,
                                    bonded_atom_pairs=bonds)
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
    suffix = ""
    if arguments.ligand is not None:
        suffix += "-%s" % arguments.ligand.strip().lower().replace(",", "")
    if arguments.ptm is not None:
        suffix += "-%s" % arguments.ptm.strip().lower().replace("@", "")
    if arguments.kind != "protein":
        suffix += "-" + arguments.kind
    if arguments.mods is not None:
        suffix += "-" + arguments.mods.strip().lower().replace("@", "").replace(",", "-")
    out = arguments.out or "/tmp/af3-batch-%s%s.json" % (arguments.model, suffix)
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
