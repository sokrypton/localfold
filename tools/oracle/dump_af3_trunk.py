"""Run the AF3 trunk on a toy protein, on CPU, and dump inputs and outputs.

    python3 tools/oracle/dump_af3_trunk.py                  # full 48-block trunk
    python3 tools/oracle/dump_af3_trunk.py --blocks 1        # one pairformer block
    python3 tools/oracle/dump_af3_trunk.py --blocks 0        # embedder only

Same contract as dump_toy_multimer.py: the point is a ground truth LocalFold can
be fed EXACTLY - not "build the same features and hope", but the model input
dict itself, so any difference afterwards is in the forward pass rather than in
the featurisation. AF3 makes that mattersmore than AF2 did, because its
featurisation is a 515 MB chemical-component dictionary and a tokeniser, none of
which the browser will ever run.

🔴 THE WEIGHTS ARE OPENFOLD3'S, NOT DEEPMIND'S, AND THAT IS DELIBERATE. AF3's
own parameters carry a Prohibited Use Policy; OpenFold3's are Apache 2.0. Since
the whole point of this repository is a page anyone can load, the oracle must be
the model we can actually ship, or it would certify a graph we cannot use. The
two share an architecture - `model='openfold3'` selects the OpenFold3 dialect of
the same forward graph - so the trunk being verified here is the trunk AF3
describes.

🔴 --blocks TRUNCATES BY SLICING THE STACKED WEIGHTS, and a truncated run is a
DIFFERENT MODEL, not an approximation of the full one. hk.experimental.layer_stack
requires the parameters' leading axis to match the configured depth, so the depth
and the weights move together. This exists to localise a divergence - "our block
1 already disagrees" is a far smaller search than "our 48-block output
disagrees" - and never to produce a structure worth looking at.
"""
import argparse
import json
import os
import pathlib
import sys

os.environ["JAX_PLATFORMS"] = "cpu"

COLABDESIGN2 = os.path.expanduser("~/Documents/GitHub/ColabDesign2")
sys.path.insert(0, COLABDESIGN2)

# 🔴 THE VENDORED AF3 HAS NO COMPILED cpp EXTENSION; the installed one does, and
# the two are the same code. Without this the import dies deep inside the CCD
# loader with a bare ModuleNotFoundError that names neither package.
import alphafold3.cpp                                          # noqa: E402
import alphafold3.cpp.cif_dict                                 # noqa: E402
sys.modules.setdefault("colabdesign2.af3.alphafold3.cpp", alphafold3.cpp)
sys.modules.setdefault("colabdesign2.af3.alphafold3.cpp.cif_dict",
                       alphafold3.cpp.cif_dict)

import numpy as np                                             # noqa: E402
import jax                                                     # noqa: E402
import haiku as hk                                             # noqa: E402

from colabdesign2 import parse_contigs                          # noqa: E402
from colabdesign2.af3 import features as f3                     # noqa: E402
from colabdesign2.af3.runner import AF3Runner, make_config      # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
WEIGHTS = os.path.expanduser(
    "~/Documents/GitHub/af3_of3/weights/af3_converted_v4")
# 12 residues: the pair representation is 12x12x128, small enough to sit in a
# JSON file and to be read by eye when a kernel is wrong.
SEQUENCE = "GSMKQIEDKIEE"


def truncate_pairformer(model_params, num_layer):
    """Keep the first `num_layer` layers of the trunk pairformer stack.

    The stack's depth is a config field but its weights are one (48, ...) array
    per tensor, and layer_stack checks the two agree - so slicing the weights is
    not an optimisation, it is what makes the shorter config loadable at all.
    """
    out = {}
    for scope, leaves in model_params.items():
        if scope.endswith("__layer_stack_no_per_layer_1") or (
                "__layer_stack_no_per_layer_1/" in scope):
            out[scope] = {k: np.asarray(v)[:num_layer] for k, v in leaves.items()}
        else:
            out[scope] = leaves
    return out


def capture(pattern):
    """An interceptor that keeps the full output of every module matching `pattern`.

    🔴 THE ACTIVATIONS ARE TRACERS AT TRACE TIME AND CANNOT BE READ. The trunk's
    recycles are a fori_loop and its stacks are scans, so reading an activation
    where it is produced yields a symbol, not numbers. jax.debug.callback is the
    way through: it fires with CONCRETE values at run time, inside loops and
    scans included, so a module that runs 48 times records 48 entries under the
    same name in execution order - which is exactly the per-block granularity
    that makes a divergence findable.

    (The mechanism is ColabDesign2's tools/module_trace, which fingerprints and
    drops each array. We keep them, so this is only affordable at toy sizes.)
    """
    import re
    matches = re.compile(pattern)
    kept, counts = {}, {}

    def record(name, value):
        kept.setdefault(name, []).append(np.asarray(value, np.float32))

    def interceptor(next_f, args, kwargs, context):
        out = next_f(*args, **kwargs)
        site = f"{context.module.module_name}/{context.method_name}"
        if context.method_name == "__init__" or not matches.search(site):
            return out
        index = counts.get(site, 0)
        counts[site] = index + 1
        for leaf, value in arrays(out):
            name = f"{site}:{leaf}" if leaf else site
            jax.debug.callback(lambda v, n=name: record(n, v), value)
        return out

    return interceptor, kept


def arrays(obj, prefix=""):
    """(name, value) for every array leaf of a nested output.

    Left unconverted: inside the interceptor these are tracers, and np.asarray
    on a tracer raises. The caller converts once it holds concrete values.
    """
    if isinstance(obj, dict):
        for key in sorted(obj):
            yield from arrays(obj[key], f"{prefix}/{key}" if prefix else key)
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from arrays(value, f"{prefix}[{index}]")
    elif hasattr(obj, "shape") and hasattr(obj, "dtype"):
        yield prefix, obj


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--blocks", type=int, default=None,
                        help="trunk pairformer depth (default: all 48)")
    parser.add_argument("--sequence", default=SEQUENCE)
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--capture", default=r"evoformer/__call__$",
                        help="regex over module call sites whose full output to"
                             " keep (default: the trunk's single and pair)")
    parser.add_argument("--out", default=None,
                        help="default: af3-oracle[-<blocks>block].json at the root")
    arguments = parser.parse_args()
    # 🔴 AND NOW HIDE THE COMMAND LINE. AF3's SwiGLU goes through tokamax, which
    # reads an absl flag, and absl parses the WHOLE of sys.argv the first time
    # any flag is read - so an argument of ours it does not recognise aborts the
    # forward pass with UnrecognizedFlagError, thousands of frames deep and
    # nowhere near the parser that accepted it.
    sys.argv = sys.argv[:1]

    sequence = arguments.sequence
    spec = parse_contigs(str(len(sequence))).resolve()
    batch = f3.featurise_spec(spec, sequences={0: sequence}, msa_crop_size=8)

    config = make_config(num_recycles=0, model="openfold3", num_msa=1)
    if arguments.blocks is not None:
        config.evoformer.pairformer.num_layer = arguments.blocks
    runner = AF3Runner(model_dir=arguments.weights, cfg=config,
                       model="openfold3", diffusion="off")
    if arguments.blocks is not None:
        runner.model_params = truncate_pairformer(runner.model_params,
                                                  arguments.blocks)

    interceptor, captured = capture(arguments.capture)
    with hk.intercept_methods(interceptor):
        out = runner.predict(batch, key=jax.random.PRNGKey(0))
    jax.block_until_ready(jax.tree_util.tree_leaves(out))   # let the callbacks land

    # ...the batch as the forward saw it, dtypes preserved: aatype and the
    # gather indices are integers, and rounding them through float32 would be a
    # silent off-by-one at token 16777217 rather than an error.
    inputs = {}
    for key in sorted(batch):
        value = batch[key]
        if isinstance(value, dict) or value is None:
            continue
        array = np.asarray(value)
        if array.dtype == object or array.ndim == 0:
            continue
        inputs[key] = {"shape": list(array.shape), "dtype": str(array.dtype),
                       "data": array.ravel().tolist()}

    outputs = {}
    for name, value in arrays(out):
        array = np.asarray(value, np.float32)
        outputs[name] = {"shape": list(array.shape),
                         "data": array.ravel().tolist()}
    # ...captured sites go in the same table. A site that ran more than once
    # (every block of a stack) is written as name#0, name#1, ... in execution
    # order, so "our block 3 diverges" is a lookup rather than a bisect.
    for site in sorted(captured):
        values = captured[site]
        for index, array in enumerate(values):
            name = site if len(values) == 1 else f"{site}#{index}"
            outputs[name] = {"shape": list(array.shape),
                             "data": array.ravel().tolist()}

    blocks = 48 if arguments.blocks is None else arguments.blocks
    suffix = "" if arguments.blocks is None else f"-{blocks}block"
    path = pathlib.Path(arguments.out) if arguments.out else \
        ROOT / f"af3-oracle{suffix}.json"
    path.write_text(json.dumps({
        "model": "openfold3",
        "sequence": sequence,
        "tokens": int(np.asarray(batch["aatype"]).shape[-1]),
        "pairformerBlocks": blocks,
        "inputs": inputs,
        "outputs": outputs,
    }))

    print(f"{path.name}  {len(sequence)} residues, {blocks} pairformer blocks,"
          f" {path.stat().st_size / 1024:.0f} KiB")
    for name in sorted(outputs):
        array = np.asarray(outputs[name]["data"], np.float32)
        print(f"  {name:40s} {str(outputs[name]['shape']):18s}"
              f" mean {array.mean():+.4f} std {array.std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
