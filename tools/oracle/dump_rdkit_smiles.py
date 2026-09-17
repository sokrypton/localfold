"""RDKit's own answer for a corpus of SMILES, as the reference this port is held to.

🔴 THIS IS AN ORACLE, NOT A DEPENDENCY. RDKit does not ship to the browser and
nothing under `src/` may import anything it produces; the point is to have a
reference that is NOT this port's own CPU path, which is the habit CLAUDE.md
records as the one worth keeping. Run it once, commit nothing (the dump lands
in `oracle-dumps/`, which is ignored whole), and check against it.

    /home/ubuntu/.venv-rdkit/bin/python tools/oracle/dump_rdkit_smiles.py

What is captured per SMILES, in the order the checkers ask for it:

  formula       heavy atoms and their hydrogens, Hill order
  atoms         element, charge, aromatic flag, implicit+explicit H count
  bonds         kekulised orders over an index mapping this port can reproduce
  rings         the smallest set of smallest rings, as sorted atom lists
  stereo        CIP codes for the tetrahedral centres RDKit assigns
  conformer     one ETKDG embedding, minimised, as a reference GEOMETRY - and
                see the note on it below, because it is not a unique answer

🔴 THE CONFORMER IS NOT A TARGET TO MATCH COORDINATE BY COORDINATE. Distance
geometry is seeded, a molecule has many conformers and every one of them is a
correct answer, so comparing positions would fail for a port that is right. The
bond LENGTHS, the bond ANGLES and the ring PLANARITY are the invariants, and
`check-smiles-conformer.mjs` compares those distributions rather than points.
"""

import json
import os
import re
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 🔴 THE CORPUS IS CHOSEN FOR THE CONVENTIONS IT REACHES, not for variety. Each
# line says which one, because a corpus nobody can read the intent of stops
# being extended and starts being trimmed.
CORPUS = [
    ("methane", "C"),                                     # the degenerate case
    ("water", "O"),                                       # one heavy atom, two H
    ("ammonium", "[NH4+]"),                               # a charge moving the valence
    ("chloride", "[Cl-]"),                                # a monatomic ion
    ("magnesium", "[Mg+2]"),                              # no implicit H outside the subset
    ("ethane", "CC"),
    ("ethene", "C=C"),
    ("ethyne", "C#C"),
    ("glycerol", "OCC(O)CO"),                             # the repo's own ligand gate
    ("benzene", "c1ccccc1"),                              # aromatic carbon
    ("pyridine", "c1ccncc1"),                             # aromatic N, no H
    ("pyrrole", "c1cc[nH]c1"),                            # aromatic N, one H - the hard one
    ("furan", "c1ccoc1"),
    ("thiophene", "c1ccsc1"),
    ("imidazole", "c1cnc[nH]1"),                          # two different aromatic N
    ("naphthalene", "c1ccc2ccccc2c1"),                    # fused aromatic
    ("indole", "c1ccc2[nH]ccc2c1"),
    ("phenol", "Oc1ccccc1"),
    ("toluene", "Cc1ccccc1"),
    ("acetic_acid", "CC(=O)O"),
    ("acetate", "CC(=O)[O-]"),
    ("acetamide", "CC(N)=O"),
    ("dimethyl_sulfoxide", "CS(C)=O"),                    # sulfur valence 4
    ("methanesulfonate", "CS(=O)(=O)[O-]"),               # sulfur valence 6
    ("sulfate", "[O-]S(=O)(=O)[O-]"),
    ("phosphate", "OP(=O)(O)O"),                          # phosphorus valence 5
    ("nitro_benzene", "O=[N+]([O-])c1ccccc1"),            # a charge-separated nitro
    ("trifluoromethyl", "FC(F)(F)c1ccccc1"),
    ("bromoethane", "CCBr"),                              # a two-letter organic atom
    ("cyclohexane", "C1CCCCC1"),                          # a saturated ring
    ("cyclopropane", "C1CC1"),                            # a strained ring
    ("spiro", "C1CCC2(CC1)CCCC2"),                        # a spiro junction
    ("bicyclic", "C1CC2CCC1CC2"),                         # a bridged bicycle
    ("adamantane", "C1C2CC3CC1CC(C2)C3"),                 # three fused rings
    ("biphenyl", "c1ccc(-c2ccccc2)cc1"),                  # a rotatable aryl-aryl bond
    ("salt", "[Na+].[Cl-]"),                              # two fragments
    ("alanine", "C[C@@H](N)C(=O)O"),                      # one tetrahedral centre
    ("d_alanine", "C[C@H](N)C(=O)O"),                     # ...and its mirror
    ("butenedioic_cis", "OC(=O)/C=C\\C(=O)O"),            # cis double-bond stereo
    ("butenedioic_trans", "OC(=O)/C=C/C(=O)O"),           # ...and trans
    ("biotin", "OC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12"),   # the job fixture
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),           # fused aromatic with exocyclic =O
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("atp", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O"),
    ("porphine", "c1cc2cc3ccc(cc4ccc(cc5ccc(cc1n2)[nH]5)n4)[nH]3"),   # four fused aromatic rings, two [nH]
    ("glucose", "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ("cholesterol", "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"),  # eight stereocentres
    ("penicillin_core", "CC1(C)S[C@@H]2[C@H](NC(=O)C)C(=O)N2[C@H]1C(=O)O"),
    ("nad_frag", "NC(=O)c1ccc[n+](c1)[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O"),

    # 🔴 THE SECOND HALF, ADDED AFTER THE FIRST FIFTY-ONE ALL PASSED. A corpus
    # that agrees with the reference everywhere has stopped being evidence and
    # started being a habit: what it proves is that the cases somebody thought
    # of are right. These are the conventions the first half does NOT reach,
    # chosen by reading the parser for branches nothing exercised and by
    # reading the geometry for shapes nothing built. None of them found a
    # defect, which is worth as much as the ones that did and is why they stay.
    ("isotope", "[13CH3][13CH3]"),                        # an isotope, parsed and inert
    ("deuterium", "[2H]C([2H])([2H])O"),                  # hydrogens written as atoms
    ("atom_map", "[CH3:1][OH:2]"),                        # the :n class, parsed and inert
    ("quaternary_ammonium", "C[N+](C)(C)C"),              # four bonds on a charged N
    ("azide", "CN=[N+]=[N-]"),                            # two cumulated double bonds
    ("allene", "C=C=C"),                                  # sp carbon between two sp2
    ("nitrile", "CC#N"),                                  # a triple bond to nitrogen
    ("disulfide", "CSSC"),                                # S-S, the longest common bond
    ("phosphonate", "CP(=O)(O)O"),                        # P with a carbon substituent
    ("boronic_acid", "OB(O)c1ccccc1"),                    # boron, whose valence is 3
    ("tetrazole", "c1nnn[nH]1"),                          # four aromatic N in one ring
    ("sulfonamide", "CS(=O)(=O)N"),
    ("epoxide", "C1CO1"),                                 # a three-ring with oxygen
    ("aziridine", "C1CN1"),
    ("cubane", "C1(C2C3C14C5C2C3C45)"),                   # five rings, rank 5, all strained
    ("coronene_frag", "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61"),   # seven fused aromatics
    ("cyclopentadienide", "c1ccc[cH-]1"),                 # an aromatic carbanion
    ("hexadecane", "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"),     # 30 atoms, all rotatable
    ("macrocycle_16", "C1CCCCCCCCCCCCCCC1"),              # a ring the polygon rule skips
    ("inositol", "O[C@H]1[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@@H]1O"),   # six centres in a ring
    ("cyclosporin_frag", "CC(C)C[C@@H]1NC(=O)[C@H](C)NC(=O)[C@@H](C)N(C)C(=O)[C@H](CC(C)C)NC1=O"),
    ("erythromycin_frag", "CC[C@H]1OC(=O)[C@H](C)[C@@H](O)[C@H](C)[C@@H](O)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@H]1C"),
    # 🔴 AND ONE THAT IS BIGGER THAN ANYTHING ELSE HERE. 62 heavy atoms against
    # the previous largest of 31, which matters because the page admits 150 and
    # every cost in this port is quadratic or cubic in the count.
    ("taxol_core", "CC(=O)OC1C(=O)C2(C)C(O)CC3OCC3(OC(C)=O)C2C(OC(=O)c2ccccc2)C2(O)CC(OC(=O)C(O)C(NC(=O)c3ccccc3)c3ccccc3)C(C)=C1C2(C)C"),
]


def one(name, smiles, embed):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"name": name, "smiles": smiles, "error": "RDKit refused it"}

    # 🔴 THE HEAVY-ATOM VIEW, BECAUSE THAT IS WHAT A COMPONENT IS. RDKit keeps
    # an ISOTOPE-LABELLED hydrogen as an atom of the graph - `[2H]` survives
    # `MolFromSmiles` where a plain `[H]` is folded into a count - while still
    # reporting it as non-heavy, so `[2H]C([2H])([2H])O` has five atoms and
    # `GetNumHeavyAtoms()` of two. Both are defensible and they are not the
    # same list. AF3 tokenises one token per HEAVY atom and
    # `parseCcdComponent` drops hydrogens before anything sees them, so the
    # heavy view is the one this port has to match, and comparing against the
    # other reported deuterated methanol as five atoms with every index shifted.
    #
    # 🔴 AND IT MUST SANITISE, OR THE HYDROGENS ARE DELETED RATHER THAN
    # COUNTED. `RemoveAllHs(sanitize=False)` gives two atoms and the formula
    # CHO - the deuteriums are gone from the graph AND from the carbon's
    # count, which is not a molecule. With the default sanitisation it is CH4O
    # and the carbon carries its three, which is the same view
    # `parseCcdComponent` produces from a dictionary entry.
    mol = Chem.RemoveAllHs(mol)

    # 🔴 THE ATOM ORDER IS THE SMILES ORDER FOR BOTH SIDES. RDKit keeps the
    # input order for `MolFromSmiles`, and this port appends atoms as it reads
    # them, so index i is the same atom in both - which is what makes a bond
    # list comparable at all. Canonical ranking would break that and is not
    # asked for anywhere here.
    atoms = [{
        "symbol": a.GetSymbol().upper(),
        "charge": a.GetFormalCharge(),
        "aromatic": a.GetIsAromatic(),
        "hydrogens": a.GetTotalNumHs(includeNeighbors=True),
        "degree": a.GetDegree(),
    } for a in mol.GetAtoms()]

    # Kekulised orders, because an "aromatic" order is not a number this port
    # can place a bond length from.
    kekule = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekule, clearAromaticFlags=True)
        orders = {}
        for b in kekule.GetBonds():
            key = tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            orders[key] = int(b.GetBondTypeAsDouble())
    except Exception as error:                              # noqa: BLE001
        orders = {}
        print(f"  {name}: kekulisation failed: {error}", file=sys.stderr)

    bonds = []
    for b in mol.GetBonds():
        key = tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        bonds.append({
            "from": key[0], "to": key[1],
            "aromatic": b.GetIsAromatic(),
            "order": orders.get(key, int(b.GetBondTypeAsDouble())),
        })
    bonds.sort(key=lambda b: (b["from"], b["to"]))

    rings = [sorted(r) for r in Chem.GetSymmSSSR(mol)]
    rings.sort()

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    stereo = [{"atom": a.GetIdx(), "cip": a.GetPropsAsDict().get("_CIPCode")}
              for a in mol.GetAtoms() if a.HasProp("_CIPCode")]
    bond_stereo = [{
        "from": b.GetBeginAtomIdx(), "to": b.GetEndAtomIdx(),
        "stereo": str(b.GetStereo()),
    } for b in mol.GetBonds() if str(b.GetStereo()) != "STEREONONE"]

    record = {
        "name": name,
        "smiles": smiles,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "heavyAtoms": mol.GetNumHeavyAtoms(),
        "weight": round(Descriptors.MolWt(mol), 4),
        "atoms": atoms,
        "bonds": bonds,
        "rings": rings,
        "stereo": stereo,
        "bondStereo": bond_stereo,
        "fragments": len(Chem.GetMolFrags(mol)),
    }

    if embed:
        # 🔴 A FIXED SEED, SO THE DUMP IS THE SAME TWICE. It still is not a
        # target to match point for point; see the module note.
        withH = Chem.AddHs(mol)
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = 0xF01D
        if AllChem.EmbedMolecule(withH, parameters) == 0:
            try:
                AllChem.MMFFOptimizeMolecule(withH, maxIters=2000)
            except Exception:                               # noqa: BLE001
                pass
            heavy = Chem.RemoveHs(withH)
            conformer = heavy.GetConformer()
            record["conformer"] = [
                [round(conformer.GetAtomPosition(i).x, 4),
                 round(conformer.GetAtomPosition(i).y, 4),
                 round(conformer.GetAtomPosition(i).z, 4)]
                for i in range(heavy.GetNumAtoms())
            ]
    return record


def rewritings(name, smiles, count=40):
    """The same molecule written many different ways, with the atom map back.

    🔴 THE STRONGEST TEST HERE AND IT COSTS NOTHING TO GENERATE. A SMILES is
    one of enormous numbers of strings for the same molecule - different
    starting atom, different branch nesting, different ring-closure digits -
    and every one of them must parse to the SAME graph with the SAME
    handedness at every centre. That is a perfect invariant needing no
    reference values at all, and it exercises exactly the code that a
    hand-picked corpus reaches by accident: the ring-closure bookkeeping that
    once inverted nine of this port's twenty-nine stereocentres.

    `_smilesAtomOutputOrder` is what makes it comparable: it says which
    original atom each position in the written string is, so two graphs with
    different numbering can be held against each other.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveAllHs(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    formula = re.sub(r"[+-]\d*$", "", rdMolDescriptors.CalcMolFormula(mol))
    seen = set()
    variants = []
    for _ in range(count):
        text = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        if text in seen or Chem.MolFromSmiles(text) is None:
            continue
        seen.add(text)
        order = [int(i) for i
                 in mol.GetProp("_smilesAtomOutputOrder")[1:-1].split(",") if i != ""]
        variants.append({"smiles": text, "order": order})
    return {
        "name": name, "formula": formula,
        "heavy": mol.GetNumAtoms(), "bonds": mol.GetNumBonds(),
        "centres": sum(1 for a in mol.GetAtoms() if a.HasProp("_CIPCode")),
        "variants": variants,
    }


def main():
    embed = "--no-embed" not in sys.argv
    out = os.path.join(ROOT, "..", "oracle-dumps", "rdkit-smiles.json")
    out = os.path.abspath(os.path.join(ROOT, "oracle-dumps", "rdkit-smiles.json")) \
        if os.path.isdir(os.path.join(ROOT, "oracle-dumps")) else os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    records = []
    for name, smiles in CORPUS:
        records.append(one(name, smiles, embed))
        print(f"  {name:22s} {records[-1].get('formula', records[-1].get('error'))}")
    rewrites = [r for r in (rewritings(name, smiles) for name, smiles in CORPUS)
                if r is not None]
    print(f"\n{sum(len(r['variants']) for r in rewrites)} re-writings of "
          f"{len(rewrites)} molecules")
    with open(out, "w") as handle:
        json.dump({"rdkit": __import__("rdkit").__version__, "records": records,
                   "rewrites": rewrites}, handle)
    print(f"\n{len(records)} records -> {out}")


if __name__ == "__main__":
    main()
