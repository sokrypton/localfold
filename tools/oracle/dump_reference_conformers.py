"""Extract AF3's reference conformers for the 21 protein components.

    python3 tools/oracle/dump_reference_conformers.py

The browser cannot run AF3's featuriser - it is a 515 MB chemical component
dictionary and a tokeniser - so the reference conformers have to ship as a
constant. This writes that constant, taken from AF3's own featuriser rather
than rebuilt from a chemistry library, so the atom ORDER, the dense-slot
layout, the four-character names, the element codes and the charges are AF3's
by construction and not by agreement.

🔴 THE CONFORMER IS SAMPLED PER INSTANCE, SO THE TABLE IS ONE SAMPLE. AF3 gives
every residue instance a fresh set of torsions around fixed bond lengths and
angles - see the commit that measured it - so no table can reproduce a dump.
Folding 6MRR with one conformer per residue type instead of AF3's own moves the
trunk's pair by relRMS 2.7e-2 and the structure by 0.01 A. The table is
admissible because the model ignores the torsions, not because it is exact.

🔴 THREE POSITIONS PER TYPE, because a chain has ends. Each type is featurised
in a 12-residue chain that carries it at tokens 0, 5 and 11 and glycine
everywhere else, so those three tokens are its N-terminal, internal and
C-terminal forms. The differences between them are what a
hand-built table gets wrong: AF3 gives the C-terminal residue an OXT that no
internal residue has.

🔴 A HOMOPOLYMER, SO THAT EVERY TYPE COMES WITH TWELVE SAMPLES OF ITSELF, which
is what makes the `rigid` list below possible. UNK is the exception: a chain of
twelve UNK is not a chain, AF3's pipeline drops it and raises "No chains in
structure", naming neither the residue nor the reason, so UNK gets glycine
filler and three samples.

WHAT `rigid` IS FOR. A table of coordinates cannot be checked against AF3 by
comparing coordinates, because the torsions are resampled. What CAN be checked
is the chemistry underneath - and rather than guess which atom pairs are rigid
from a distance cutoff (a torsion can fold two atoms to within a bond length of
each other), each type's samples say so directly: rigidity is TOPOLOGY. Bonds are read off a
conformer by distance - every covalent bond in these components is under 1.85 A
and the shortest non-bonded contact inside a residue is a 1-3 pair at about
2.2 A, so the split is unambiguous - and a pair is rigid if it is 1-2 or 1-3 in
that graph.

🔴 AND RIGIDITY IS NOT "THE SAMPLES AGREED". Measuring it that way looks more
empirical and is wrong: RDKit gave all twelve valines in a poly-valine chain
the same chi1, so N-CG1 - a 1-4 pair across a rotatable torsion - held still
across every sample and was called rigid. It then missed by 0.87 A against a
real protein. The samples VERIFY the topological claim (any 1-2 or 1-3 pair
that moves is reported below) rather than making it. Written out as index pairs into the internal atom list, so
check_af3_featurise.js can hold the shipped conformer to them.

🔴 TWELVE AND NOT THREE. The atom cross-attention builds a 128-wide key window
over the padded flat atom layout, so a chain short enough to have fewer than
128 padded atom slots indexes past the end of its own layout - an IndexError
deep in atom_layout.py that names neither the chain nor its length.
"""
import json
import os
import pathlib
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
COLABDESIGN2 = os.path.expanduser("~/Documents/GitHub/ColabDesign2")
sys.path.insert(0, COLABDESIGN2)

import alphafold3.cpp                                          # noqa: E402
import alphafold3.cpp.cif_dict                                 # noqa: E402
sys.modules.setdefault("colabdesign2.af3.alphafold3.cpp", alphafold3.cpp)
sys.modules.setdefault("colabdesign2.af3.alphafold3.cpp.cif_dict",
                       alphafold3.cpp.cif_dict)

import numpy as np                                             # noqa: E402
from colabdesign2 import parse_contigs                         # noqa: E402
from colabdesign2.af3 import features as f3                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TYPES = "ACDEFGHIKLMNPQRSTVWYX"
BOND_LENGTH = 1.85      # angstroms; every covalent bond here is shorter
RIGID_THRESHOLD = 0.25  # how far a 1-2 or 1-3 pair may move before it is reported
POSITIONS = ("nTerminal", "internal", "cTerminal")


def name_of(chars):
    return "".join(chr(int(c) + 32) for c in chars if int(c) > 0).strip()


def main():
    sys.argv = sys.argv[:1]
    table = {}
    for code in TYPES:
        length = 12
        # UNK alternates with glycine rather than filling the chain, so that it
        # still gets five internal samples to measure rigidity from.
        # 🔴 AND IT STILL HAS TO SIT AT BOTH ENDS. Sampling only the even
        # positions leaves the last token a GLYCINE, and glycine and UNK have
        # the same four atoms - so the C-terminal entry would be silently the
        # wrong component, and so would the aatype, which is why both are read
        # at a token this list names rather than at a fixed index.
        keep = lambda i: i % 2 == 0 or i == length - 1
        sequence = (code * length if code != "X"
                    else "".join(code if keep(i) else "G" for i in range(length)))
        samples = ([i for i in range(length) if keep(i)] if code == "X"
                   else list(range(length)))
        spec = parse_contigs(str(length)).resolve()
        batch = f3.featurise_spec(spec, sequences={0: sequence}, msa_crop_size=8)
        mask = np.asarray(batch["ref_mask"]).astype(bool)
        pos = np.asarray(batch["ref_pos"], np.float64)
        element = np.asarray(batch["ref_element"]).astype(int)
        charge = np.asarray(batch["ref_charge"], np.float64)
        chars = np.asarray(batch["ref_atom_name_chars"]).astype(int)
        aatype = np.asarray(batch["aatype"]).astype(int)
        entry = {"aatype": int(aatype[samples[1]])}
        internal = [t for t in samples if 0 < t < length - 1]
        for token, where in zip((samples[0], samples[1], samples[-1]), POSITIONS):
            slots = np.nonzero(mask[token])[0]
            # 🔴 THE SLOT INDEX IS PART OF THE ANSWER. The dense layout is not
            # "the real atoms packed from zero" by definition - it is whatever
            # AF3 put there - so record it rather than assume it is range(n).
            entry[where] = [
                {"slot": int(s), "name": name_of(chars[token, s]),
                 "element": int(element[token, s]),
                 "charge": round(float(charge[token, s]), 6),
                 "pos": [round(float(v), 4) for v in pos[token, s]]}
                for s in slots]
        # 🔴 RIGIDITY IS TOPOLOGY, AND THE SAMPLES ONLY VERIFY IT - see the note
        # at the top.
        slots = [a["slot"] for a in entry["internal"]]
        n = len(slots)
        spread = np.zeros((n, n))
        for t in internal:
            p = pos[t][slots]
            d = np.linalg.norm(p[:, None] - p[None], axis=-1)
            if t == internal[0]:
                first = d
            spread = np.maximum(spread, np.abs(d - first))
        bonded = (first < BOND_LENGTH) & ~np.eye(n, dtype=bool)
        rigid = bonded | (bonded @ bonded)
        entry["rigid"] = [[i, j, round(float(first[i, j]), 4)]
                          for i in range(n) for j in range(i + 1, n)
                          if rigid[i, j]]
        entry["samples"] = len(internal)
        loose = [(entry["internal"][i]["name"], entry["internal"][j]["name"],
                  round(float(spread[i, j]), 3))
                 for i, j, _ in entry["rigid"] if spread[i, j] > RIGID_THRESHOLD]
        if loose:
            print(f"  !! {code}: 1-2 or 1-3 pairs that moved across samples: {loose}")
        table[code] = entry
        names = [a["name"] for a in entry["internal"]]
        extra = [a["name"] for a in entry["cTerminal"]] 
        pairs = len(names) * (len(names) - 1) // 2
        print(f"{code} aatype {entry['aatype']:2d}  {len(names):2d} atoms"
              f"  C-term {len(extra):2d}  rigid {len(entry['rigid']):3d}/{pairs:3d}"
              f"  over {entry['samples']:2d} samples  {' '.join(names)}")

    out = ROOT / "tools" / "oracle" / "reference-conformers.json"
    out.write_text(json.dumps(table, indent=1))
    print(f"\n{out.relative_to(ROOT)}  {out.stat().st_size / 1024:.0f} KiB")

    # What actually differs between the three positions, which is the thing a
    # hand-built table would get wrong.
    for code in TYPES:
        entry = table[code]
        internal = {a["name"] for a in entry["internal"]}
        for where in ("nTerminal", "cTerminal"):
            here = {a["name"] for a in entry[where]}
            if here != internal:
                print(f"{code} {where}: +{sorted(here - internal)}"
                      f" -{sorted(internal - here)}")


if __name__ == "__main__":
    main()
