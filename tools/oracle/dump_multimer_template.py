"""Capture AF2-multimer's template embedder from JAX, with a REAL template.

    python3 tools/oracle/dump_multimer_template.py --out toy-template-jax.json
    python3 tools/oracle/dump_multimer_template.py --template 1qys-crystal.pdb:A \
      --out toy-template-jax-real.json

🔴 IT EXISTS BECAUSE TWO UNVALIDATED IMPLEMENTATIONS DISAGREED AND NEITHER WAS
AN ORACLE. `tools/oracle/template_reference.py` is a hand-written numpy
transcription of this module and nothing ever checked IT; the GPU path is
`src/multimer/template.js` and nothing ever checked that either - the reference
wrote `toy-template.json` and no JavaScript read it. Compared at last they
differ by relRMS 1.2e-1 after the first pair block, which says only that one of
them is wrong.

The input term is already settled without this: GPU and numpy agree to 2.2e-7
once the f32 bundle is used rather than the shipped int8 one. What is not
settled is the two pair blocks, and this is what settles them.

🔴 AND IT CAPTURES ARGUMENTS AS WELL AS OUTPUTS. The module reads the pair
representation the embedder built, so a checker that had only the output would
have to reproduce everything before it to feed the thing it is checking.
"""
import argparse
import json
import os
import re
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "/Users/mini/Documents/GitHub/ColabDesign2")

import numpy as np                                             # noqa: E402
import jax                                                     # noqa: E402
import haiku as hk                                             # noqa: E402

from colabdesign2 import parse_contigs                          # noqa: E402
from colabdesign2.af2 import featurize, register_losses         # noqa: E402

register_losses()
from colabdesign2.af2.runner import AF2Runner                   # noqa: E402

PARAMS = "/Users/mini/Documents/GitHub/af-params/oracle"
# atom37's order, which is what AF2 indexes template_all_atom_positions by.
ATOM37 = ("N CA C CB O CG CG1 CG2 OG OG1 SG CD CD1 CD2 ND1 ND2 OD1 OD2 SD CE"
          " CE1 CE2 CE3 NE NE1 NE2 OE1 OE2 CH2 NH1 NH2 OH CZ CZ2 CZ3 NZ OXT"
          ).split()
ONE_LETTER = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
              "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
              "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
              "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
RESTYPES = "ARNDCQEGHILKMFPSTWYV"


def arrays(obj, prefix=""):
    if isinstance(obj, dict):
        for key in sorted(obj):
            yield from arrays(obj[key], f"{prefix}/{key}" if prefix else key)
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from arrays(value, f"{prefix}[{index}]")
    elif hasattr(obj, "shape") and hasattr(obj, "dtype"):
        yield prefix, obj


def capture(pattern, argument_pattern=None):
    """Keep every array a matching module produced, and optionally its inputs.

    The activations are tracers at trace time; jax.debug.callback fires with
    concrete values at run time, which is the only way to read inside a scan.
    """
    matches = re.compile(pattern)
    takes_arguments = re.compile(argument_pattern) if argument_pattern else None
    kept, counts = {}, {}

    def record(name, value):
        kept.setdefault(name, []).append(np.asarray(value, np.float32))

    def interceptor(next_f, args, kwargs, context):
        out = next_f(*args, **kwargs)
        site = f"{context.module.module_name}/{context.method_name}"
        wanted = matches.search(site)
        if takes_arguments is not None and takes_arguments.search(site):
            wanted = True
        if context.method_name == "__init__" or not wanted:
            return out
        counts[site] = counts.get(site, 0) + 1
        if takes_arguments is not None and takes_arguments.search(site):
            for position, argument in enumerate(args):
                for leaf, value in arrays(argument):
                    name = f"{site}<{position}{'.' + leaf if leaf else ''}"
                    jax.debug.callback(lambda v, n=name: record(n, v), value)
        for leaf, value in arrays(out):
            name = f"{site}:{leaf}" if leaf else site
            jax.debug.callback(lambda v, n=name: record(n, v), value)
        return out

    return interceptor, kept


def read_template(path, chain, length):
    """A PDB chain as atom37 arrays over the query's first `length` residues.

    🔴 KEYED ON THE RESIDUE NUMBER, never on position in the atom list: a
    structure missing residues has no lines for them, and grouping by position
    closes the hole up silently.
    """
    order, atoms = [], {}
    for line in open(os.path.expanduser(path)):
        if not line.startswith("ATOM"):
            continue
        if chain and line[21] != chain:
            continue
        if line[16] not in " A":
            continue
        key = line[22:27]
        if key not in atoms:
            atoms[key] = (line[17:20].strip(), {})
            order.append(key)
        atoms[key][1].setdefault(line[12:16].strip(), (
            float(line[30:38]), float(line[38:46]), float(line[46:54])))
    order = order[:length]
    # 🔴 UNCOVERED IS THE GAP TYPE 21, NOT ALANINE. AF3 was measured doing
    # exactly this and the aatype features are read whether or not there is
    # geometry, so type 0 would contribute alanine's embedding at every pair an
    # uncovered position takes part in.
    aatype = np.full(length, 21, np.int32)
    positions = np.zeros((length, 37, 3), np.float32)
    mask = np.zeros((length, 37), np.float32)
    for index, key in enumerate(order):
        name3, found = atoms[key]
        code = ONE_LETTER.get(name3)
        aatype[index] = RESTYPES.index(code) if code in RESTYPES else 20
        for slot, atom in enumerate(ATOM37):
            if atom in found:
                positions[index, slot] = found[atom]
                mask[index, slot] = 1
    return aatype, positions, mask, len(order)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--template", default=None, metavar="PATH[:CHAIN]")
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--model", default="alphafold2_multimer_v3")
    parser.add_argument("--name", default="model_1_multimer_v3")
    parser.add_argument("--out", default="toy-template-jax.json")
    arguments = parser.parse_args()

    total = arguments.length * arguments.copies
    spec = parse_contigs(":".join([str(arguments.length)] * arguments.copies)).resolve()
    inputs = featurize(spec, chain_break=None)
    inputs["opt"] = {"weights": {}, "alpha": 2.0, "temp": 1.0, "soft": 1.0,
                     "hard": 1.0, "dropout": False, "pssm_hard": True,
                     "template": {"rm_ic": False},
                     "con": {"num": 2, "cutoff": 14.0, "binary": False,
                             "seqsep": 9, "num_pos": float("inf")}}

    covered = 0
    if arguments.template:
        path, _, chain = arguments.template.partition(":")
        aatype, positions, mask, covered = read_template(path, chain, total)
        # 🔴 WRITTEN INTO THE FEATURES THE FORWARD READS, not into a config.
        # AF2-multimer's template features are `template_aatype`,
        # `template_all_atom_positions` and `template_all_atom_mask`, with a
        # leading template axis - and its mask feature is spelled `_mask`,
        # singular, where the monomer spells it `_masks`.
        for key, value in (("template_aatype", aatype[None]),
                           ("template_all_atom_positions", positions[None]),
                           ("template_all_atom_mask", mask[None])):
            if key in inputs:
                inputs[key] = np.asarray(value, inputs[key].dtype)
            else:
                inputs[key] = np.asarray(value)
        if "template_mask" in inputs:
            inputs["template_mask"] = np.ones_like(inputs["template_mask"])
        print(f"template {path}: {covered} of {total} residues,"
              f" {int(mask.sum())} atoms")

    # 🔴 MONOMER'S TEMPLATE BRANCH IS OFF UNLESS ASKED FOR, and a run without
    # this captures NOTHING while looking like a success: the module simply
    # does not execute, so the interceptor has nothing to intercept and the
    # dump comes back with a length and no tensors. ColabDesign2 picks
    # `model_3_ptm`'s config - templates disabled - unless use_templates is
    # set, and the weight loader drops the template parameters to match.
    # Multimer has template.enabled true either way, which is why it worked
    # without this.
    runner = AF2Runner(model_type=arguments.model, data_dir=PARAMS,
                       model_names=[arguments.name], use_bfloat16=False,
                       use_templates=arguments.template is not None)
    rng = np.random.default_rng(0)
    seq = np.zeros((1, total, 20), np.float32)
    for index in range(total):
        seq[0, index, rng.integers(0, 20)] = 1.0

    interceptor, kept = capture(r"template_embedding/__call__$",
                                r"template_embedding/__call__$")
    with hk.intercept_methods(interceptor):
        out = runner.apply({"seq": jax.numpy.asarray(seq)}, inputs,
                           jax.random.PRNGKey(0))
    jax.block_until_ready(jax.tree_util.tree_leaves(out))

    payload = {"length": total, "covered": covered,
               "chains": [index * arguments.copies // total for index in range(total)]}
    for name, values in sorted(kept.items()):
        value = values[0]
        payload[name] = {"shape": list(value.shape), "data": value.ravel().tolist()}
        print(f"  {name:66s} {str(value.shape):18s}"
              f" mean {value.mean():+.4f} std {value.std():.4f}")
    if arguments.template:
        payload["template"] = {
            "aatype": aatype.tolist(),
            "positions": positions.ravel().tolist(),
            "atomMask": mask.ravel().tolist()}
    json.dump(payload, open(arguments.out, "w"))
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
