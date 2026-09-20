"""Does NATIVE ESMFold2 place a modified residue, where this port does not?

    ~/venv_ef2/bin/python tools/esmc/probe-esmfold2-modified.py

🔴 THE QUESTION THIS SETTLES. docs/WEB.md records EF2-fast folding a SEP@3 with
its backbone in place and its side chain scattered - CB 2.09 A out, the
phosphate 1.4-4.1 A - while a plain CCD glycerol in the SAME fold is placed
correctly, and while more sampler steps make only the modified residue worse.
The featurisation was ruled out field by field (conformer exact, one
`refSpaceUid`, bonds present and consumed, `molType` matching the reference's
chain-type rule). What was left was "the model or this port", and nothing here
could separate them: every instrument in this repository runs the port.

This runs the REFERENCE. `esm` 3.4.1's `ESMFold2InputBuilder` takes the full
`StructurePredictionInput` - which `prepare_protein_features` is documented as
the no-modifications restriction OF - so a modified residue is expressible, and
the same checkpoint folds it.

🔴 AND IT SCORES BONDS, NOT RMSD. Two folds of a 58-mer from a single sequence
will not superpose; what is being asked is whether the CHEMISTRY of one residue
survives, which is a local question and the same one docs/AF3.md scores
everywhere else. The control is in the same structure: the unmodified residues'
own backbone bonds, which say whether this fold is any good at all before its
modified residue is read.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# The ideals the port scores against, from the CCD; see tools/gpu/bond-geometry.js.
SEP_BONDS = {("N", "CA"): 1.469, ("CA", "CB"): 1.530, ("CB", "OG"): 1.417,
             ("OG", "P"): 1.610, ("CA", "C"): 1.506, ("C", "O"): 1.231,
             ("P", "O1P"): 1.510, ("P", "O2P"): 1.510, ("P", "O3P"): 1.560}
BACKBONE = {("N", "CA"): 1.469, ("CA", "C"): 1.506, ("C", "O"): 1.231}
# A standard residue's first four heavy atoms, in the reference's own order.
BACKBONE_ORDER = ["N", "CA", "C", "O"]


def distance(a, b):
    return math.dist(a, b)


def ratios(atoms, table):
    out = []
    for (left, right), ideal in table.items():
        if left in atoms and right in atoms:
            out.append(distance(atoms[left], atoms[right]) / ideal)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence",
                        default="GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK")
    parser.add_argument("--code", default="SEP")
    parser.add_argument("--at", type=int, default=3, help="1-based, as the page counts")
    parser.add_argument("--esmfold2", default="esmfold2-fast-600m")
    parser.add_argument("--esmc", default="esmc-600m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0,
                        help="sampler steps; 0 leaves the checkpoint's own"
                             " `inference_num_steps` (15). \U0001f534 THE ARM THAT"
                             " ANSWERS 'is it just too few steps?' - the ports"
                             " run 11-15 by default, so the vendor has to be"
                             " asked at the SAME count before its 1.002 means"
                             " anything.")
    parser.add_argument("--ligand", default="",
                        help="a CCD code to fold BESIDE the modification. \U0001f534"
                             " THE DISCRIMINATOR: a ligand is atomised the same"
                             " way, one token per atom, but sits in its own"
                             " chain - so a port that misplaces both has a"
                             " general atom problem and one that misplaces only"
                             " the modification has an atomised-residue-inside-"
                             "a-polymer problem. They are different bugs.")
    parser.add_argument("--out", default=None, help="write the fold as a PDB")
    arguments = parser.parse_args()

    from esm.models.esmfold2 import EsmFold2ExperimentalModel
    from esm.models.esmfold2.processor import ESMFold2InputBuilder
    from esm.utils.structure.input_builder import (
        LigandInput, Modification, ProteinInput, StructurePredictionInput)

    builder = ESMFold2InputBuilder()
    # 🔴 ZERO-INDEXED HERE, ONE-INDEXED ON THE PAGE. `Modification.position` is
    # documented "zero-indexed" in input_builder.py; --at counts the way the
    # page's SEP@3 does, so the conversion happens once, here.
    chains_in = [ProteinInput(
        id="A", sequence=arguments.sequence,
        modifications=[Modification(position=arguments.at - 1, ccd=arguments.code)])]
    if arguments.ligand:
        chains_in.append(LigandInput(id="B", ccd=[arguments.ligand]))
    request = StructurePredictionInput(sequences=chains_in)
    features, chains = builder.prepare_input(request, seed=arguments.seed,
                                             device=arguments.device)

    # 🔴 `load_esmc=True` AND NOTHING ELSE. The tower comes either bundled with
    # the trunk or from the repo `config.esmc_id` names - there is no path
    # argument, and passing one lands in `resolve_model_dir`'s **kwargs as
    # "unexpected keyword argument 'esmc_path'".
    model = EsmFold2ExperimentalModel.from_pretrained(
        str(ROOT / arguments.esmfold2), load_esmc=True,
        device=arguments.device).eval()
    # 🔴 THE BUILDER RETURNS MORE THAN THIS FORWARD TAKES. `prepare_input` fills
    # the distogram-conditioning slots whether or not the request used them, and
    # the experimental forward refuses an unexpected keyword rather than
    # ignoring it - so the request's own keys are filtered to the signature.
    import inspect
    accepted = set(inspect.signature(model.forward).parameters)
    dropped = sorted(k for k in features if k not in accepted)
    if dropped:
        print("not passed to forward:", ", ".join(dropped))
    extra = {} if arguments.steps <= 0 else {"num_sampling_steps": arguments.steps}
    with torch.no_grad():
        output = model.forward(
            **{k: v for k, v in features.items() if k in accepted}, **extra)

    coords = output["sample_atom_coords"][0].float().cpu().numpy()

    # 🔴 THE TOKEN LAYOUT FIRST, because it is an answer on its own. This port
    # atomises a modified residue into ONE TOKEN PER ATOM; if the reference
    # keeps it as one token carrying ten atoms, that difference is the finding
    # and no bond needs measuring to see it.
    tokens = [token for chain in chains for token in chain.tokens]
    modified = [t for t in tokens if t.residue_name == arguments.code]
    # 🔴 THE RESTYPE AN ATOMISED RESIDUE'S TOKENS CARRY, which is the thing
    # sokrypton/alphafold3's 96d1958 fixed on its side: "an atomised residue is
    # UNKNOWN to esmfold2, not its parent". Printed beside a plain residue's so
    # the two are comparable at a glance.
    plain = [tok for tok in tokens if tok.residue_name not in (arguments.code,
                                                               arguments.ligand)]
    print(f"res_type: {arguments.code} tokens "
          f"{sorted({tok.res_type for tok in modified})}, "
          f"a plain {plain[0].residue_name} {plain[0].res_type}, "
          f"input_id {sorted({tok.input_id for tok in modified})}")
    print(f"tokens {len(tokens)}, {arguments.code} tokens {len(modified)}, "
          f"atom_count {[t.atom_count for t in modified]}, "
          f"mol_type {sorted({t.mol_type for t in modified})}")

    # The atom names of the modification, in the order the builder laid them
    # out - read from the reference's OWN ccd rather than assumed.
    # 🔴 THE REFERENCE'S OWN ATOM ORDER, DERIVED ITS OWN WAY. Its order is what
    # indexes `sample_atom_coords`, and the first version of this took the CCD
    # list and sliced the first ten - which keeps OXT, a LEAVING atom a
    # mid-chain residue drops, and so shifted every name after it by one. It
    # then reported OG-P at 1.447 while measuring OG against the phosphorus's
    # neighbour. That is this file's own warning, one paragraph after writing
    # it: match the labels before believing a distance.
    from esm.models.esmfold2.prepare_input import (
        get_ccd_leaving_atoms, get_ligand_ccd_atoms_with_charges)
    leaving = get_ccd_leaving_atoms(arguments.code)
    names = [a[0] for a in get_ligand_ccd_atoms_with_charges(arguments.code)
             if a[0] not in leaving]

    first = modified[0]
    start = first.atom_start
    count = sum(t.atom_count for t in modified)
    placed = coords[start:start + count]
    print(f"{arguments.code} atoms {count} at {start}..{start + count - 1}")
    if names is not None:
        print("ccd atom names:", ",".join(names[:count]))
        atoms = {names[i]: tuple(placed[i]) for i in range(min(count, len(names)))}
        got = ratios(atoms, SEP_BONDS)
        print("\nbond ratios (1.000 is the ideal):")
        for (left, right), ideal in SEP_BONDS.items():
            if left in atoms and right in atoms:
                print(f"  {left}-{right:4} {distance(atoms[left], atoms[right]):6.3f}"
                      f"  ideal {ideal:.3f}  ratio {distance(atoms[left], atoms[right]) / ideal:.3f}")
        # 🔴 THE CONTROL IS IN THE SAME STRUCTURE, which is the only way the
        # number above means anything: a fold that is bad everywhere would put
        # a bad modified residue in it for reasons that have nothing to do with
        # atomisation. Every unmodified residue's own backbone, same fold.
        control = []
        for token in tokens:
            if token.residue_name == arguments.code or token.atom_count < 4:
                continue
            here = {}
            for slot in range(token.atom_count):
                where = token.atom_start + slot
                if where < len(coords) and slot < len(BACKBONE_ORDER):
                    here[BACKBONE_ORDER[slot]] = tuple(coords[where])
            control.extend(ratios(here, BACKBONE))
        if got:
            print(f"\nNATIVE {arguments.code:4} mean ratio {sum(got) / len(got):.3f}"
                  f"  over {len(got)} bonds")
        if arguments.ligand:
            want = {("C1", "O1"): 1.430, ("C1", "C2"): 1.520, ("C2", "O2"): 1.430,
                    ("C2", "C3"): 1.520, ("C3", "O3"): 1.430}
            lig = [tok for tok in tokens if tok.residue_name == arguments.ligand]
            if lig:
                start = lig[0].atom_start
                count = sum(tok.atom_count for tok in lig)
                lnames = [a[0] for a in get_ligand_ccd_atoms_with_charges(arguments.ligand)]
                atoms_l = {lnames[i]: tuple(coords[start + i])
                           for i in range(min(count, len(lnames)))}
                scored = ratios(atoms_l, want)
                if scored:
                    print(f"NATIVE {arguments.ligand:4} mean ratio "
                          f"{sum(scored) / len(scored):.3f}  over {len(scored)} bonds")
        if control:
            print(f"NATIVE control  mean ratio {sum(control) / len(control):.3f}"
                  f"  over {len(control)} backbone bonds of {len(tokens) - len(modified)}"
                  " unmodified residues")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
