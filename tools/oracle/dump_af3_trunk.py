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

🔴 THE DEFAULT IS DEEPMIND'S AF3, WHICH THIS SITE MAY NOT SERVE. Those
parameters carry a Prohibited Use Policy and terms that forbid redistribution,
so they can verify a graph here and can never be published from it; the shipped
bundle has to be OpenFold3's Apache-2.0 weights or another open checkpoint of
the lineage (--model openfold3).

They are still the right thing to build against FIRST, because they are the
reference the others are ports of, and because the stock graph is the SIMPLER
one: `model='openfold3'` turns on four branches stock AF3 does not have - the
column-wise attention's pair-bias swap, a symmetrised bond matrix, an element
index shift, and Fourier weights read from the checkpoint. Verifying against a
port would mean carrying its divergences without knowing which were which.

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
# Each model's blob directory. The runner picks the forward graph's dialect from
# the model name, so the two must move together - a stock checkpoint read through
# the OpenFold3 graph asks for parameters it does not contain.
# Both fetched or converted from source rather than found lying around, and
# both checked against the graph's own jax.eval_shape table before use: every
# tensor present, none extra, none mis-shaped.
WEIGHTS = {
    "alphafold3": "~/af3_official_weights",
    # ...written by ColabDesign2's converter, which is the current one. The
    # alphafold3 repo ships an older convert_of3_weights.py under a different
    # name, so neither its output nor its path should be reached for here.
    "openfold3": "~/af3_converted_cd2",
}
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


def capture(pattern, argument_pattern=None):
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
    # 🔴 SOME THINGS ARE ONLY VISIBLE AS AN ARGUMENT. The denoiser's noisy
    # positions are drawn inside the sampler's scan and handed straight to the
    # diffusion head; nothing downstream is an invertible function of them, so a
    # capture that records only outputs cannot reproduce a single denoising step.
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
        index = counts.get(site, 0)
        counts[site] = index + 1
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
    parser.add_argument("--model", default="alphafold3", choices=sorted(WEIGHTS))
    parser.add_argument("--weights", default=None,
                        help="blob directory (default: the model's)")
    parser.add_argument("--diffusion", type=int, default=None, metavar="STEPS",
                        help="run the diffusion sampler for STEPS steps"
                             " (default: skip it entirely)")
    parser.add_argument("--a3m", default=None, metavar="PATH",
                        help="an A3M whose rows become the MSA, instead of the"
                             " query alone; applied to every chain")
    parser.add_argument("--paired-a3m", default=None, metavar="PATH",
                        dest="paired_a3m",
                        help="an A3M set as every chain's paired_msa, so AF3"
                             " does its own cross-chain pairing; --a3m stays"
                             " the unpaired block")
    parser.add_argument("--dna", action="append", metavar="SEQUENCE",
                        help="a DNA chain, appended after the protein chains. Repeatable.")
    parser.add_argument("--rna", action="append", metavar="SEQUENCE",
                        help="an RNA chain, appended after the protein chains. Repeatable.")
    parser.add_argument("--modification", action="append", metavar="CHAIN:CCD:POSITION",
                        help="a modified residue, as CCD:POSITION for the first chain "
                             "or CHAIN:CCD:POSITION for a numbered one, 1-based. "
                             "Repeatable. e.g. SEP:3, or 1:PTR:12")
    parser.add_argument("--ligand", action="append", metavar="CCD",
                        help="append a ligand chain by CCD code (repeatable),"
                             " e.g. --ligand ATP")
    parser.add_argument("--recycles", type=int, default=0,
                        help="trunk recycles (AF3's own default is 10);"
                             " the capture records every pass in order")
    parser.add_argument("--float32", action="store_true",
                        help="run the trunk in float32 instead of AF3's bfloat16")
    parser.add_argument("--capture-args", default=None, dest="capture_args",
                        help="regex over call sites whose ARGUMENTS to keep too,"
                             " recorded as site<0, site<1, ...")
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

    # 🔴 A COLON SEPARATES CHAINS, here and in parse_contigs. A complex is the
    # only way to exercise the relative encoding's same_chain / same_entity /
    # sym_id branches at all - on a monomer every pair is same-chain, so those
    # paths run in no check that uses one.
    sequence = arguments.sequence
    chains = [chain for chain in sequence.split(":") if chain]
    # 🔴 A LIGAND HAS NO LENGTH IN THE SPEC. Its token count comes from the
    # chemical component dictionary, which is AF3's to know, so the contig
    # carries the CCD code and resolve_ligands fills the count in afterwards
    # from the featurised batch. That is why ligands are appended as
    # `ligand:ATP` segments rather than as a number of residues.
    contig = ":".join(str(len(chain)) for chain in chains)
    for code in arguments.ligand or []:
        contig += f":ligand:{code.upper()}"
    # 🔴 AND IT MUST NOT BE RESOLVED WHEN ONE IS PRESENT. resolve() walks every
    # segment and a ligand has no length to give it, so it raises - the count is
    # AF3's to supply. A numeric protein contig is already concrete, so skipping
    # resolve() costs nothing here.
    spec = parse_contigs(contig)
    if not arguments.ligand:
        spec = spec.resolve()
    # 🔴 THE CROP MUST CLEAR THE ALIGNMENT, and num_msa must match it below.
    # AF3 pads the MSA to msa_crop_size rows and then TRUNCATES to num_msa, so a
    # crop of 8 over a 32-row alignment silently checks eight rows, and a
    # num_msa of 1 checks one - in both cases against arrays that look right.
    alignment = None
    if arguments.a3m is not None:
        alignment = pathlib.Path(arguments.a3m).read_text()
    # 🔴 THE PAIRED BLOCK IS A SEPARATE INPUT AND HAS NO OTHER WAY IN.
    # spec_to_fold_input only ever sets unpaired_msa, so a homo-oligomer
    # featurised through it has no paired rows at all - and the paired block is
    # exactly what a homo-oligomer's MSA mostly IS. Rebuilding each chain with
    # both is the only way to get ground truth for that case.
    paired = None
    if arguments.paired_a3m is not None:
        paired = pathlib.Path(arguments.paired_a3m).read_text()
        # ProteinChain is a plain __slots__ class, not a dataclass: rebuilding
        # one drops whatever the constructor does not take. The slot is set in
        # place instead, which touches exactly the one field.
        _original = f3.spec_to_fold_input

        def _with_paired(*args, **kwargs):
            fold_input = _original(*args, **kwargs)
            for chain in fold_input.chains:
                if hasattr(chain, "_paired_msa"):
                    chain._paired_msa = paired
            return fold_input

        f3.spec_to_fold_input = _with_paired

    # 🔴 MODIFICATIONS GO ON THE CHAIN OBJECT, WHICH THE SPEC CANNOT CARRY. The
    # contig says how long a chain is and spec_to_fold_input builds a plain
    # ProteinChain from it, so a PTM has no way in through that path - the
    # wrapper below sets the slot afterwards, the same trick --paired-a3m uses
    # for the same reason. ProteinChain is a __slots__ class rather than a
    # dataclass, so rebuilding one would drop whatever the constructor does not
    # take; setting the one slot touches only that field.
    # 🔴 A NUCLEIC CHAIN CANNOT COME THROUGH parse_contigs, which describes
    # protein segments by length and nothing else. It is built as a chain object
    # and featurised directly, the same path --modification takes and for the
    # same reason: featurise_spec renumbers residue_index from the contig, which
    # assumes one token per residue and knows nothing about these.
    nucleic = [("dna", seq) for seq in (arguments.dna or [])]
    nucleic += [("rna", seq) for seq in (arguments.rna or [])]
    if arguments.modification or nucleic:
        import dataclasses
        from colabdesign2.af3.alphafold3.common import folding_input as _fi
        wanted = {}
        for entry in (arguments.modification or []):
            parts = entry.split(":")
            if len(parts) == 2:
                chain_index, code, position = 0, parts[0], int(parts[1])
            elif len(parts) == 3:
                chain_index, code, position = int(parts[0]), parts[1], int(parts[2])
            else:
                raise SystemExit(f"--modification wants CCD:POSITION or CHAIN:CCD:POSITION, got {entry}")
            wanted.setdefault(chain_index, []).append((code.upper(), position))
        _before_ptms = f3.spec_to_fold_input

        def _with_ptms(*args, **kwargs):
            fold_input = _before_ptms(*args, **kwargs)
            # 🔴 REBUILT THROUGH THE CONSTRUCTOR, NOT POKED INTO THE SLOT. The
            # paired-MSA hack above sets `_paired_msa` directly and is right to:
            # that slot is read as it is written. `_ptms` is not - the
            # constructor is what turns the list into the form the rest of the
            # pipeline expects - and assigning it raw produced a batch with the
            # right TOKEN COUNT and nonsense residue numbering: 1..12 followed
            # by 4..12 for a twelve-residue chain, which looks plausible enough
            # to check against.
            rebuilt = []
            for index, chain in enumerate(fold_input.chains):
                if index in wanted and isinstance(chain, _fi.ProteinChain):
                    chain = _fi.ProteinChain(
                        id=chain.id, sequence=chain.sequence, ptms=wanted[index],
                        description=chain.description, paired_msa=chain.paired_msa,
                        unpaired_msa=chain.unpaired_msa, templates=chain.templates)
                rebuilt.append(chain)
            return dataclasses.replace(fold_input, chains=rebuilt)

        f3.spec_to_fold_input = _with_ptms

    msa_crop = 8 if alignment is None else 1 + alignment.count(">")
    if paired is not None:
        msa_crop += paired.count(">")
    # 🔴 featurise_spec RENUMBERS residue_index FROM THE CONTIG, WHICH ASSUMES
    # ONE TOKEN PER RESIDUE. With a modified residue that is false, and the
    # renumbering is silent: a twelve-residue chain with a phosphoserine comes
    # back with the right token count, the right aatype, and residue_index
    # 1..12 then 4..12 instead of 1, 2, then ten 3s and 4..12. Called through
    # featurise() the same fold_input gives the correct numbering, so that is
    # what a modified dump uses. Ligands are unaffected - their tokens are a
    # chain of their own, so contig numbering happens to land right.
    if arguments.modification or nucleic:
        captured = {}
        _before_capture = f3.spec_to_fold_input

        def _capture(*args, **kwargs):
            fold_input = _before_capture(*args, **kwargs)
            captured["fold_input"] = fold_input
            return fold_input

        f3.spec_to_fold_input = _capture
        f3.spec_to_fold_input(spec, name="design", seeds=(0,),
                              sequences=dict(enumerate(chains)))
        fold_input = captured["fold_input"]
        # 🔴 --a3m HAS TO BE APPLIED HERE TOO, AND WAS NOT. featurise_spec takes
        # the alignment as an argument; featurise() takes it off the chain, so
        # on this path every protein chain kept the query-only MSA that
        # spec_to_fold_input builds - and a dump asked for with --a3m came back
        # with padding rows, looking exactly like an alignment that had been
        # read and found empty.
        # ProteinChain is a __slots__ class, not a dataclass - the slot is set
        # in place, the way --paired-a3m does it, because rebuilding one drops
        # whatever the constructor does not take.
        if alignment is not None:
            for chain in fold_input.chains:
                if isinstance(chain, _fi.ProteinChain):
                    chain._unpaired_msa = alignment
        if nucleic:
            import string
            extra = []
            for index, (kind, seq) in enumerate(nucleic):
                ident = string.ascii_uppercase[len(fold_input.chains) + index]
                if kind == "dna":
                    extra.append(_fi.DnaChain(id=ident, sequence=seq, modifications=[]))
                else:
                    extra.append(_fi.RnaChain(id=ident, sequence=seq, modifications=[],
                                              unpaired_msa=f">query\n{seq}\n"))
            fold_input = dataclasses.replace(
                fold_input, chains=list(fold_input.chains) + extra)
        batch = f3.featurise(fold_input, msa_crop_size=msa_crop)
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
    else:
        batch = f3.featurise_spec(spec, sequences=dict(enumerate(chains)),
                                  msa=alignment, msa_crop_size=msa_crop)
    sequence = "".join(chains)

    weights = os.path.expanduser(arguments.weights
                                 or WEIGHTS[arguments.model])
    # 🔴 num_msa IS NOT THE MSA ARRAY'S HEIGHT. AF3 pads the batch to
    # msa_crop_size and then TRUNCATES to num_msa, so a dump's msa array can be
    # eight rows deep while the model read one. A checker that infers depth from
    # the array compares its own deep trunk against AF3's shallow one and
    # reports nonsense, so the number the model actually used is recorded.
    num_msa = 1 if alignment is None else msa_crop
    # 🔴 RECYCLING HAS NEVER BEEN DUMPED. This was pinned to 0, so every check
    # in tools/oracle compares a single trunk pass - and AF3's own default is
    # ten. The recycle path feeds the previous pass's pair and single back
    # through prev_embedding, which is a different code path from the first
    # pass and is exercised by nothing.
    config = make_config(num_recycles=arguments.recycles, model=arguments.model,
                         num_msa=num_msa,
                         num_diffusion_samples=1,
                         diffusion_steps=arguments.diffusion)
    # 🔴 AF3'S TRUNK COMPUTES IN BFLOAT16, and that sets the floor on what any
    # reimplementation can be checked to. bfloat16 keeps eight mantissa bits, so
    # its relative epsilon is 2^-8 = 3.9e-3 - and a block's own captured output
    # differs from its input plus its five captured deltas by 4.3e-3, which is
    # that and not a fault in either. An f32 run gives a reference the graph can
    # actually be held to; it is not what the shipped model does.
    if arguments.float32:
        config.global_config.bfloat16 = "none"
    if arguments.blocks is not None:
        config.evoformer.pairformer.num_layer = arguments.blocks
    # 🔴 THE SAMPLER IS 200 STEPS BY DEFAULT AND EACH ONE IS A FULL DENOISER
    # PASS. --diffusion 1 runs a single step, which is all a forward-pass check
    # needs and is the difference between seconds and an afternoon on a CPU.
    runner = AF3Runner(model_dir=weights, cfg=config, model=arguments.model,
                       diffusion="off" if arguments.diffusion is None else "forward")
    if arguments.blocks is not None:
        runner.model_params = truncate_pairformer(runner.model_params,
                                                  arguments.blocks)

    interceptor, captured = capture(arguments.capture, arguments.capture_args)
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
    if arguments.model != "alphafold3":
        suffix = f"-{arguments.model}{suffix}"
    if arguments.diffusion is not None:
        suffix += f"-diff{arguments.diffusion}"
    if arguments.float32:
        suffix += "-f32"
    path = pathlib.Path(arguments.out) if arguments.out else \
        ROOT / f"af3-oracle{suffix}.json"
    path.write_text(json.dumps({
        "model": arguments.model,
        "sequence": sequence,
        "modifications": arguments.modification or [],
        "dna": arguments.dna or [],
        "rna": arguments.rna or [],
        # The chains, kept separately: `sequence` is joined so a consumer can
        # index it by token, and the split is not recoverable from that.
        "chains": chains,
        "tokens": int(np.asarray(batch["aatype"]).shape[-1]),
        # 🔴 WRITTEN OUT, not merely used. A checker cannot recover this from
        # the arrays - the msa array is padded to msa_crop_size and the model
        # read only the first num_msa rows of it - so a dump without this field
        # silently invites the reader to guess, and the guess is wrong whenever
        # an alignment was supplied.
        "numMsa": int(num_msa),
        "numRecycles": int(arguments.recycles),
        # So a checker knows what to fetch: a ligand's chemistry is not
        # recoverable from the batch, and guessing it from the atom names is
        # exactly the sort of inference this file exists to avoid.
        "ligands": [code.upper() for code in (arguments.ligand or [])],
        "pairformerBlocks": blocks,
        "inputs": inputs,
        "outputs": outputs,
    }))

    print(f"{path.name}  {arguments.model}, {len(sequence)} residues,"
          f" {blocks} pairformer blocks,"
          f" {path.stat().st_size / 1024:.0f} KiB")
    for name in sorted(outputs):
        array = np.asarray(outputs[name]["data"], np.float32)
        print(f"  {name:40s} {str(outputs[name]['shape']):18s}"
              f" mean {array.mean():+.4f} std {array.std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
