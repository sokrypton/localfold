"""The native backend's worker (CUDA, or Metal on a Mac), folding for real: npm run test:native.

    python3 tools/check-native-worker.py             # every case (two go to api.colabfold.com)
    python3 tools/check-native-worker.py --offline   # without the search cases
    python3 tools/check-native-worker.py --no-page   # without the page arm (a headless Chrome, a broker)
    python3 tools/check-native-worker.py --only=6mrr  # just the cases whose name contains it
    python3 tools/check-native-worker.py --no-af3     # without AlphaFold 3 itself (whose weights need DeepMind's terms)

🔴 AND ON A MAC IT HOLDS THE METAL PORTS: python/localfold/worker.py drives metal/<port> there (NATIVE), the page names the
backend Metal, and a case may carry a Metal bar of its own (METAL_BARS) where Metal's sampling lands elsewhere on a
target - each one the Metal gate's own number (metal/af3/gate.py), never a widening for convenience.

python/localfold/worker.py over its own stdin protocol, one process for every case as the broker runs it,
each job shaped as the page sends one (AlphaFold 3 JSON from the entity rows, the rows themselves, the
form's controls), each fold scored against its deposited structure (cuda/af3/score.py) and held to a
bar, and every result held to the fields the page ingests (web/app.js, cudaPrediction): a PDB, a pLDDT a
token, a PAE of tokens^2, the token layout, the chains. test:bridge's CUDA arm is a stub and proves the
broker's routing; this is the half that proves the worker folds.

Needs the native ports built (cuda/colab_setup.sh, or each port's nvcc line) and the bundles on disk
(a missing one is fetched). The refusals are cases too: a sampler, a model or an input the CUDA backend
does not have must come back as a sentence naming it, never as some other fold.
"""
import json
import math
import os
import re
import signal
import subprocess
import sys
import time

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NATIVE = "metal" if sys.platform == "darwin" else "cuda"
NATIVE_NAME = {"metal": "Metal", "cuda": "CUDA"}[NATIVE]
# a case's bar on Metal where it differs from the A100's: Metal's sampling lands elsewhere on these targets, at the
# Metal gate's own numbers (metal/af3/gate-baseline.m2.json) and the CUDA-translated port's before it
METAL_BARS = {"protenix2 6mrr": 1.6}      # (1.561 in metal/af3/gate.py; the A100 0.499)
# a case the Metal port REFUSES by name where CUDA folds it - asserted as that refusal, so a Metal port that later folded
# it, or failed some other way, would be caught either way: never a case quietly skipped. BioEmu2's edm_fkc takes the
# potentials' gradient back through the denoiser (cuda/bioemu2's vjp.cuh, the denoiser's whole backward), which metal/bioemu2
# does not have (docs/BIOEMU2.md, "Not done yet"); its edm_gd and edm_fk fold
METAL_REFUSALS = {"bioemu2 6mrr, edm_fkc": "edm_fkc takes the potentials' gradient back through the denoiser"}
FIX = os.path.join(REPO, "tools", "fixtures")
S6 = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"
THREE = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L",
             LYS="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V", MSE="M")


def chain_sequence(pdb, chain):
    seen, out = set(), ""
    for line in open(pdb):
        if line.startswith(("ATOM", "HETATM")) and line[12:16] == " CA " and line[21] == chain and line[22:27] not in seen:
            seen.add(line[22:27])
            out += THREE.get(line[17:20], "X")
    return out


def job(sequences, ligands=(), modifications=None, seed=1):
    """AlphaFold 3 JSON as web/job-json.js writes it for these rows."""
    entries, ids = [], iter("ABCDEFGHIJ")
    for index, sequence in enumerate(sequences):
        protein = {"id": next(ids), "sequence": sequence}
        if modifications and index in modifications:
            protein["modifications"] = [{"ptmType": code, "ptmPosition": at} for code, at in modifications[index]]
        entries.append({"protein": protein})
    for code in ligands:
        # a CCD code, or {"smiles": ...} - a structure the page builds with shared/chem
        entries.append({"ligand": {"id": next(ids), **(code if isinstance(code, dict) else {"ccdCodes": [code]})}})
    return json.dumps({"name": "check", "modelSeeds": [seed], "sequences": entries, "dialect": "alphafold3", "version": 1})


BIOTIN = "OC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12"


def ligand_bonds(pdb):
    """A ligand's bonds as the fold left them: every pair of its heavy atoms under 1.95 A, and the closest pair.
    🔴 RMSD CANNOT SEE THIS - it is a protein's - and pLDDT reads high over a torn ligand: ESMFold2 at its
    eleven default steps put biotin's atoms 0.2-0.7 A apart, 18-20 "bonds" for its 17, while its fold passed."""
    atoms = [tuple(float(line[c:c + 8]) for c in (30, 38, 46)) for line in pdb.splitlines() if line.startswith("HETATM")]
    pairs = [math.dist(atoms[i], atoms[j]) for i in range(len(atoms)) for j in range(i + 1, len(atoms))]
    return sum(1 for d in pairs if d < 1.95), min(pairs) if pairs else None


def controls(**over):
    return {"model-family": "af3", "msa-mode": "none", "max-msa": "128:256", "recycles": "3", "tolerance": "0.1",
            "af3-count": "25", "random-seed": "1", **over}


def protein(sequence, template=None):
    row = {"type": "protein", "value": sequence, "copies": 1}
    if template:
        row["template"] = template
    return row


def cases(offline):
    s5 = chain_sequence(f"{FIX}/5caj-crystal.pdb", "A")
    sa, sd = chain_sequence(f"{FIX}/1brs-crystal.pdb", "A"), chain_sequence(f"{FIX}/1brs-crystal.pdb", "D")
    brs = open(f"{FIX}/1brs-crystal.pdb").read()
    caj = open(f"{FIX}/5caj-crystal.pdb").read()
    out = [
        # name, payload, (reference, chains) or None, bar (A), expected tokens or None, refusal text or None
        ("af3 6mrr", {"family": "af3", "controls": controls(), "entities": [protein(S6)], "job": job([S6])},
         (f"{FIX}/6mrr-crystal.pdb", "A"), 1.0, 68, None),
        ("af3 5caj, its crystal uploaded", {"family": "af3", "controls": controls(),
         "entities": [protein(s5, {"kind": "upload", "text": caj, "source": "A", "filename": "5caj.pdb"})],
         "job": job([s5])}, (f"{FIX}/5caj-crystal.pdb", "A"), 0.5, None, None),
        # (protenix2's 6MRR is bimodal on every sampler - a seed lands at ~0.6-0.9 A or ~1.5-1.7, Euler 200 included,
        # docs/AF3.md "The default sampler" - so its bar is a chain's, not one mode's)
        ("protenix2 6mrr", {"family": "protenix2", "controls": controls(**{"model-family": "protenix2"}),
         "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.0, 68, None),
        # (past 80 tokens: protenix2's 256-channel pair track on its fused kernels, cuda/af3/src/fused256.cuh)
        ("protenix2 5caj, its crystal uploaded", {"family": "protenix2", "controls": controls(**{"model-family": "protenix2"}),
         "entities": [protein(s5, {"kind": "upload", "text": caj, "source": "A", "filename": "5caj.pdb"})],
         "job": job([s5])}, (f"{FIX}/5caj-crystal.pdb", "A"), 0.5, None, None),
        ("opendde 6mrr", {"family": "opendde", "controls": controls(**{"model-family": "opendde"}),
         "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.0, 68, None),
        ("af3 6mrr + GOL + SEP3", {"family": "af3", "controls": controls(),
         "entities": [protein(S6), {"type": "ligand", "value": "GOL", "copies": 1}],
         "job": job([S6], ["GOL"], {0: [("SEP", 3)]})}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68 + 9 + 6, None),
        # 🔴 THE SAME JOB AGAIN ON THE SAME WARM SERVER, after other shapes and kinds of job, must come back the SAME: a
        # resident server keeps scratch, caches and graphs between jobs, and three bugs of that one class reached a
        # branch - AF2's shared scratch slot holding an emitted LayerNorm across the pair stack (argon-ptx), BioEmu2's
        # step conditioning recognised by its address (bed641ad), "All 5"'s first job (191197b3). A fresh process a
        # case (cuda/*/gate.py) cannot see any of them
        ("af3 6mrr again (the resident server, after other jobs)", {"family": "af3", "controls": controls(),
         "entities": [protein(S6)], "job": job([S6]), "samePdbAs": "af3 6mrr"}, (f"{FIX}/6mrr-crystal.pdb", "A"), 1.0, 68, None),
        # 🔴 ...AND THE SAME JOB AT ANOTHER SEED MUST NOT (the page writes its Random seed as the job's modelSeeds)
        ("af3 6mrr at seed 2 (the resident server)", {"family": "af3", "controls": controls(**{"random-seed": "2"}),
         "entities": [protein(S6)], "job": job([S6], seed=2), "otherPdbThan": "af3 6mrr"}, (f"{FIX}/6mrr-crystal.pdb", "A"), 1.0, 68, None),
        ("af2 monomer 6mrr", {"family": "monomer", "controls": controls(**{"model-family": "monomer", "af2Model": "1"}),
         "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None),
        ("af2 monomer 6mrr, stopping early at 0.5 A", {"family": "monomer",
         "controls": controls(**{"model-family": "monomer", "af2Model": "1", "tolerance": "0.5"}),
         "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None),
        ("af2 monomer 5caj, its crystal uploaded", {"family": "monomer",
         "controls": controls(**{"model-family": "monomer", "af2Model": "1", "recycles": "0"}),
         "entities": [protein(s5, {"kind": "upload", "text": caj, "source": "A", "filename": "5caj.pdb"})],
         "job": job([s5])}, (f"{FIX}/5caj-crystal.pdb", "A"), 1.0, None, None),
        ("af2 monomer 6mrr again (the resident server, after a templated 261-residue job)", {"family": "monomer",
         "controls": controls(**{"model-family": "monomer", "af2Model": "1"}), "entities": [protein(S6)], "job": job([S6]),
         "samePdbAs": "af2 monomer 6mrr"}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None),
        ("af2 multimer 1brs, both chains templated", {"family": "multimer",
         "controls": controls(**{"model-family": "multimer", "af2Model": "1"}),
         "entities": [protein(sa, {"kind": "upload", "text": brs, "source": "A", "filename": "1brs.pdb"}),
                      protein(sd, {"kind": "upload", "text": brs, "source": "D", "filename": "1brs.pdb"})],
         "job": job([sa, sd])}, (f"{FIX}/1brs-crystal.pdb", "A,D"), 1.0, len(sa) + len(sd), None),
        ("af2 multimer 1brs, an alignment per chain (an archive's)", {"family": "multimer",
         "controls": controls(**{"model-family": "multimer", "af2Model": "1", "msa-mode": "upload"}),
         "entities": [protein(sa), protein(sd)], "job": job([sa, sd]),
         "msas": {"unpaired": [f">101\n{sa}\n", f">101\n{sd}\n"], "paired": ["", ""]}},
         (f"{FIX}/1brs-crystal.pdb", "A,D"), 25.0, len(sa) + len(sd), None),
        ("esmfold2 6mrr", {"family": "ef2-fast-600m", "controls": controls(**{"model-family": "ef2"}),
         "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.0, 68, None),
        # (past 80 tokens: ESMFold2's 256-channel trunk on its fused kernels, cuda/af3/src/fused256.cuh)
        ("esmfold2 5caj", {"family": "ef2-fast-600m", "controls": controls(**{"model-family": "ef2"}),
         "entities": [protein(s5)], "job": job([s5])}, (f"{FIX}/5caj-crystal.pdb", "A"), 3.0, None, None),
        # 🔴 A SMILES LIGAND, AND ITS BONDS: biotin (three stereocentres, two fused rings) built by shared/chem,
        # through AF3 and ESMFold2 - and ESMFold2 tears any ligand at its eleven default steps, the vendor's own
        # included, so these also hold the step floor for per-atom tokens (ESMFOLD2_ATOMISED_STEPS), CCD and
        # SMILES alike. The bond check is the expected count and nothing closer than 1 A.
        ("af3 6mrr + biotin (SMILES)", {"family": "af3", "controls": controls(),
         "entities": [protein(S6), {"type": "smiles", "value": BIOTIN, "copies": 1}],
         "job": job([S6], [{"smiles": BIOTIN}]), "ligandBonds": 17}, (f"{FIX}/6mrr-crystal.pdb", "A"), 1.5, 68 + 16, None),
        ("esmfold2 6mrr + biotin (SMILES)", {"family": "ef2-fast-600m", "controls": controls(**{"model-family": "ef2"}),
         "entities": [protein(S6), {"type": "smiles", "value": BIOTIN, "copies": 1}],
         "job": job([S6], [{"smiles": BIOTIN}]), "ligandBonds": 17}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68 + 16, None),
        ("esmfold2 6mrr + GOL", {"family": "ef2-fast-600m", "controls": controls(**{"model-family": "ef2"}),
         "entities": [protein(S6), {"type": "ligand", "value": "GOL", "copies": 1}],
         "job": job([S6], ["GOL"]), "ligandBonds": 5}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68 + 6, None),
        ("esmfold2 6mrr again (the resident server, after other jobs)", {"family": "ef2-fast-600m",
         "controls": controls(**{"model-family": "ef2"}), "entities": [protein(S6)], "job": job([S6]),
         "samePdbAs": "esmfold2 6mrr"}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.0, 68, None),
        ("esmfold2 300M 6mrr", {"family": "ef2-fast-300m", "controls": controls(**{"model-family": "ef2"}),
         "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None),
        # the page's steps (web/app.js samplerPlan), which the page resolves and sends: an empty step dial is
        # the dial's preferred count there and the model's own 200 here
        ("af3 6mrr, the page's steps", {"family": "af3", "controls": controls(**{"af3-count": ""}),
         "schedule": {"steps": 25}, "entities": [protein(S6)], "job": job([S6])},
         (f"{FIX}/6mrr-crystal.pdb", "A"), 1.0, 68, None),
        ("chai1 6mrr", {"family": "chai1", "controls": controls(**{"model-family": "chai1"}),
         "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None),
        ("chai1 5caj, its crystal uploaded", {"family": "chai1", "controls": controls(**{"model-family": "chai1"}),
         "entities": [protein(s5, {"kind": "upload", "text": caj, "source": "A", "filename": "5caj.pdb"})],
         "job": job([s5])}, (f"{FIX}/5caj-crystal.pdb", "A"), 3.0, None, None),
        ("chai1 6mrr + GOL + SEP3", {"family": "chai1", "controls": controls(**{"model-family": "chai1"}),
         "entities": [protein(S6), {"type": "ligand", "value": "GOL", "copies": 1}],
         "job": job([S6], ["GOL"], {0: [("SEP", 3)]})}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68 + 9 + 6, None),
        # every family's WEIGHT WALK (cuda/featurise/af3_weights.h) read from its own bundle - no map file
        *[(f"{m} 6mrr", {"family": m, "controls": controls(**{"model-family": m}), "entities": [protein(S6)],
           "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None)
          for m in ("boltz2", "intellifold2", "rosettafold3")],
        # (Flow was removed: a job from an older page that still asks for it is refused by name)
        ("refused: the removed Flow sampler", {"family": "af3",
         "controls": controls(**{"af3-mode": "flow"}), "entities": [protein(S6)],
         "job": job([S6])}, None, None, None, "Flow was removed"),
        ("af2 monomer model 3 (a delta on model 1) 6mrr", {"family": "monomer-3",
         "controls": controls(**{"model-family": "monomer"}), "entities": [protein(S6)], "job": job([S6])},
         (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None),
        # "All 5" (the page's Model # row): the five models folded at once on one --multi server when model 1's job
        # comes, the others answered from it - each the same structure as its own job (python/localfold/worker.py,
        # MultiServer). Model 1's job carries af2Model "all", which once failed outright (191197b3)
        ("af2 All 5, model 1 (folds all five)", {"family": "monomer",
         "controls": controls(**{"model-family": "monomer", "af2Model": "all"}), "entities": [protein(S6)], "job": job([S6]),
         "samePdbAs": "af2 monomer 6mrr"}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None),
        ("af2 All 5, model 3 (answered from the sweep)", {"family": "monomer-3",
         "controls": controls(**{"model-family": "monomer", "af2Model": "all"}), "entities": [protein(S6)], "job": job([S6]),
         "samePdbAs": "af2 monomer model 3 (a delta on model 1) 6mrr"}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None),
        ("af2 multimer model 2 (a delta) 1brs, templated", {"family": "multimer-2",
         "controls": controls(**{"model-family": "multimer"}),
         "entities": [protein(sa, {"kind": "upload", "text": brs, "source": "A", "filename": "1brs.pdb"}),
                      protein(sd, {"kind": "upload", "text": brs, "source": "D", "filename": "1brs.pdb"})],
         "job": job([sa, sd])}, (f"{FIX}/1brs-crystal.pdb", "A,D"), 1.0, len(sa) + len(sd), None),
        ("refused: a template on AF2 model 4", {"family": "monomer-4", "controls": controls(**{"model-family": "monomer"}),
         "entities": [protein(s5, {"kind": "upload", "text": caj, "source": "A"})], "job": job([s5])},
         None, None, None, "no template embedder"),
        ("refused: a template on ESMFold2", {"family": "ef2-fast-600m", "controls": controls(**{"model-family": "ef2"}),
         "entities": [protein(s5, {"kind": "upload", "text": caj, "source": "A"})], "job": job([s5])},
         None, None, None, "no template"),
        ("refused: a ligand on AF2", {"family": "monomer", "controls": controls(**{"model-family": "monomer"}),
         "entities": [protein(S6), {"type": "ligand", "value": "GOL", "copies": 1}], "job": job([S6], ["GOL"])},
         None, None, None, "protein chains only"),
    ]
    # 🔴 BioEmu2: NO CONFIDENCE HEAD, SO ITS RESULT IS HELD TO WHAT THE PAGE READS INSTEAD - every sample's PDB (the page
    # lays them on the play bar and takes BioEmu2's physicality report of each), a contact map of tokens^2, the token
    # layout, and pLDDT/pTM NULL rather than a number the B-factors do not hold. A propagation starts from the frame the
    # page sends with the job (`bioemu2Frame`: here the previous case's first sample), and each step is held near IT,
    # not near the crystal: a trajectory moves. An equilibrium ensemble is held to being a chain, not to one structure.
    be2 = lambda **over: controls(**{"model-family": "bioemu2", "af3-count": "50", **over})
    j6 = job([S6])
    out += [
        ("bioemu2 6mrr, 2 samples", {"family": "bioemu2", "controls": be2(**{"bioemu2-samples": "2"}),
         "entities": [protein(S6)], "job": j6, "samples": 2}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None),
        ("bioemu2 6mrr propagated, 2 x 2 steps of 1 ns", {"family": "bioemu2",
         "controls": be2(**{"bioemu2-mode": "propagate", "bioemu2-lag": "1", "bioemu2-propagation-steps": "2",
                            "bioemu2-samples": "2"}),
         "entities": [protein(S6)], "job": j6, "samples": 4, "bioemu2Frame": "previous"},
         ("previous", "A"), 3.0, 68, None),
        ("bioemu2 6mrr equilibrium, contacts", {"family": "bioemu2",
         "controls": be2(**{"bioemu2-mode": "equilibrium", "bioemu2-samples": "2",
                            "bioemu2-contacts": open(f"{FIX}/bioemu2/contacts.json").read()}),
         "entities": [protein(S6)], "job": j6, "samples": 2}, ("chain", "A"), None, 68, None),
        # 🔴 AND A SECOND FOLD OF THE SAME LENGTH IS THE FIRST FOLD: the resident server once kept a triangle mask by its
        # address, which the denoiser's buffers overwrote, so every same-length fold after the first was garbage
        ("bioemu2 6mrr again (the resident server, warm)", {"family": "bioemu2", "controls": be2(**{"bioemu2-samples": "2"}),
         "entities": [protein(S6)], "job": j6, "samples": 2, "samePdbAs": "bioemu2 6mrr, 2 samples"},
         (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None),
        ("bioemu2 6mrr + GOL", {"family": "bioemu2", "controls": be2(), "entities": [protein(S6),
         {"type": "ligand", "value": "GOL", "copies": 1}], "job": job([S6], ["GOL"]), "samples": 1, "ligandBonds": 5},
         (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68 + 6, None),
        # 🔴 STEERED (edm_fk): every particle a sample, the potentials on the host - held to the steering's own job:
        # NAG and retinal come out of the plain sampler with wrong-handed centres and twisted stereo bonds in 9 of 10
        # samples, and none here (tools/score-bioemu2-samples.mjs measures it; this asserts the fold and its shape)
        ("bioemu2 6mrr + NAG + retinal, edm_fk", {"family": "bioemu2",
         "controls": be2(**{"bioemu2-sampler": "edm_fk", "bioemu2-samples": "3"}),
         "entities": [protein(S6)], "job": open(f"{FIX}/bioemu2/6mrr-nag-ret.json").read(), "samples": 3},
         ("chain", "A"), None, None, None),
        # ...and edm_fkc, the corrector: the potentials' gradient back through the denoiser on the device (src/vjp.cuh)
        ("bioemu2 6mrr, edm_fkc", {"family": "bioemu2",
         "controls": be2(**{"bioemu2-sampler": "edm_fkc", "bioemu2-samples": "3"}),
         "entities": [protein(S6)], "job": j6, "samples": 3}, ("chain", "A"), None, 68, None),
        ("bioemu2 refused: a steered propagation", {"family": "bioemu2",
         "controls": be2(**{"bioemu2-mode": "propagate", "bioemu2-sampler": "edm_gd"}), "entities": [protein(S6)],
         "job": j6, "bioemu2Frame": "previous"}, None, None, None, "runs unsteered"),
        ("bioemu2 refused: a template", {"family": "bioemu2", "controls": be2(),
         "entities": [protein(s5, {"kind": "upload", "text": caj, "source": "A"})], "job": job([s5])},
         None, None, None, "takes no template"),
        ("bioemu2 refused: propagating with no frame", {"family": "bioemu2",
         "controls": be2(**{"bioemu2-mode": "propagate"}), "entities": [protein(S6)], "job": j6},
         None, None, None, "propagation starts from a structure"),
        ("bioemu2 refused: propagating another input's frame", {"family": "bioemu2",
         "controls": be2(**{"bioemu2-mode": "propagate"}), "entities": [protein(s5)], "job": job([s5]),
         "bioemu2Frame": "previous"}, None, None, None, "atom(s) of this input"),
    ]
    # biohub's released ESMFold2-Fast and ESMFold2 (ESM-C 6B, local exports only: metal/README.md) where their bundles
    # are on disk under the names fetch-weights gives them - nothing downloads them, so absent is a skip, not a failure
    weights = os.environ.get("LOCALFOLD_WEIGHTS_DIR") or (     # (fetch.h's home(): no /proc/self/exe on a Mac)
        os.path.expanduser("~/.cache/localfold") if sys.platform == "darwin" else REPO)
    released = lambda *dirs: all(os.path.exists(os.path.join(weights, d, "manifest.json")) for d in dirs)
    if released("model-esmfold2-fast-f32", "model-esmc-6b-int8"):
        out.append(("esmfold2-fast released (ESM-C 6B) 6mrr", {"family": "ef2-fast",
                    "controls": controls(**{"model-family": "ef2-fast-600m", "plm-mode": "esmc-6b"}),
                    "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None))
    if released("model-esmfold2-f32", "model-esmc-6b-int8"):
        out.append(("esmfold2 released 6mrr, single sequence", {"family": "ef2",
                    "controls": controls(**{"model-family": "ef2-fast-600m", "plm-mode": "esmc-6b-full", "msa-mode": "none"}),
                    "entities": [protein(S6)], "job": job([S6])}, (f"{FIX}/6mrr-crystal.pdb", "A"), 2.5, 68, None))
        # what ESMFold2 cannot read is refused by name: its search over a complex, and one merged alignment over two
        # chains (it reads one alignment a protein chain)
        out.append(("refused: ESMFold2's search over two protein chains", {"family": "ef2",
                    "controls": controls(**{"model-family": "ef2", "msa-mode": "search"}),
                    "entities": [protein(S6), protein(S6)], "job": job([S6, S6])}, None, None, None, "takes one protein chain"))
        out.append(("refused: ESMFold2 and one merged alignment over two chains", {"family": "ef2",
                    "controls": controls(**{"model-family": "ef2", "msa-mode": "paste", "msa-text": f">101\n{S6}\n"}),
                    "entities": [protein(S6), protein(S6)], "job": job([S6, S6])}, None, None, None, "one merged alignment"))
        if not offline:
            out.append(("esmfold2 released 5caj, MSA searched", {"family": "ef2",
                        "controls": controls(**{"model-family": "ef2", "msa-mode": "search"}),
                        "entities": [protein(s5)], "job": job([s5])}, (f"{FIX}/5caj-crystal.pdb", "A"), 3.0, None, None))
    if not offline:
        out += [
            ("af3 1brs, MSA searched", {"family": "af3", "controls": controls(**{"msa-mode": "search"}),
             "entities": [protein(sa), protein(sd)], "job": job([sa, sd])}, (f"{FIX}/1brs-crystal.pdb", "A,D"), 1.5,
             len(sa) + len(sd), None),
            ("af2 monomer 5caj, MSA and template searched", {"family": "monomer",
             "controls": controls(**{"model-family": "monomer", "af2Model": "1", "msa-mode": "search"}),
             "entities": [protein(s5, {"kind": "search"})], "job": job([s5])}, (f"{FIX}/5caj-crystal.pdb", "A"), 2.5,
             None, None),
        ]
    return out


def ca_spacing(pdb_text):
    """The median distance between consecutive alpha carbons: 3.8 A in a chain."""
    ca = [tuple(float(line[c:c + 8]) for c in (30, 38, 46)) for line in pdb_text.splitlines()
          if line.startswith("ATOM") and line[12:16] == " CA "]
    gaps = sorted(math.dist(a, b) for a, b in zip(ca, ca[1:]))
    return gaps[len(gaps) // 2] if gaps else 0.0


def score(pdb_text, reference, chains):
    path = "/tmp/localfold-cuda-check.pdb"
    with open(path, "w") as handle:
        handle.write(pdb_text)
    said = subprocess.run([sys.executable, os.path.join(REPO, "cuda", "af3", "score.py"), path, reference, chains],
                          capture_output=True, text=True).stdout
    found = re.findall(r"CA RMSD ([0-9.]+) A", said)
    return float(found[0]) if found else None


def page_arm(bad):
    """The website's half: a real broker (python/localfold/server.py --native), a reader's page on it - no
    backend picker, CUDA is where its folds go - a sequence and Fold - and what the page made of the answer: the status
    line the worker wrote, a prediction in the page's own shape (its PAE a typed array of tokens^2, the
    model labelled CUDA), and no WebGPU device asked for in the reader's browser."""
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import cdp
    import urllib.request
    port, reader_port, token = 8893, 9396, "cuda-check"
    broker = subprocess.Popen([sys.executable, "python/localfold/server.py", "--port", str(port),
                               "--token", token, "--native"],
                              cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    reader = None
    try:
        deadline = time.time() + 180
        for line in broker.stdout:
            if line.startswith("BACKEND ") or time.time() > deadline:
                break
        health = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health?t={token}"))
        if "native" not in health.get("backends", []) or health.get("native") != NATIVE:
            bad.append(f"page: /health offers {health.get('backends')} / {health.get('native')} with --native")
            return
        reader, ws = cdp.launch(reader_port, "/tmp/localfold-cuda-page-reader")
        ws.call("Page.enable")
        ws.call("Runtime.enable")
        ws.call("Page.addScriptToEvaluateOnNewDocument", source="""
          window.__readerGpu = 0;
          if (navigator.gpu) {
            const ask = navigator.gpu.requestAdapter.bind(navigator.gpu);
            navigator.gpu.requestAdapter = (...a) => { window.__readerGpu += 1; return ask(...a); };
          }""")
        ws.call("Page.navigate", url=f"http://127.0.0.1:{port}/index.html?backend=native&t={token}")
        cdp.wait_for(ws, "!!window.__entityList && !!document.querySelector('.native-live')", 120, "the reader's page")
        cdp.evaluate(ws, """(() => {
          for (const key of ['alphafold3', 'openbind0', 'opendde', 'boltz2', 'protenix2', 'intellifold2', 'rosettafold3'])
            try { localStorage.setItem('localfold.modelTerms.' + key, 'accepted'); } catch (cause) {}
          return true;
        })()""")
        # ...and nothing to choose: the page offers no backend control at all
        if cdp.evaluate(ws, "!!document.querySelector('#native-status select')"):
            bad.append("page: the badge offers a backend select - CUDA is not a choice")
        # the badge's Live preview: present for CUDA, and off means the finished fold only
        live = cdp.evaluate(ws, "(() => { const l = document.querySelector('.native-live'); return l ? { shown: !l.hidden, on: l.querySelector('input').checked } : null; })()")
        if not live or not live["shown"] or not live["on"]:
            bad.append(f"page: the Live preview box is {live} - want it shown and on for CUDA")
        lineage = "protenix2" if "--no-af3" in sys.argv else "af3"
        # the native-only models' options (ESM-C 6B on the PLM row, ESMFold2 on the model row): shown exactly where the
        # server says it has them, and ESMFold2 folded through the page when it does
        served = health.get("models", [])
        shown = cdp.evaluate(ws, """[...document.querySelectorAll('option[data-native-model]')]
          .filter((o) => !o.hidden).map((o) => o.dataset.nativeModel).sort()""")
        if shown != sorted(served):
            bad.append(f"page: the native-only options shown are {shown}, where /health has {sorted(served)}")
        arms = [(lineage, 68, True), ("monomer", 68, True), ("ef2-fast-600m", 68, True), (lineage, 68, False)]
        if "ef2" in served:
            arms.insert(3, ("ef2", 68, True))
        if "ef2-fast" in served:          # (chosen as a reader does: EF2-fast on the model row, ESM-C 6B on the PLM row)
            arms.insert(3, ("ef2-fast", 68, True))
        # ...and the lineage again, streamed, after other folds: a later fold's trunk must draw its contact maps (the
        # first fold's frames used to hold the panel and drop every one)
        arms.insert(len(arms) - 1, (lineage, 68, True))
        for family, tokens, streamed in arms:
            cdp.evaluate(ws, f"""(() => {{ const b = document.querySelector('.native-live input');
              if (b.checked !== {'true' if streamed else 'false'}) b.click(); return b.checked; }})()""")
            # (ESMFold2 is one entry on the model row and the variant row picks the checkpoint, as a reader does it)
            row, plm = {"ef2-fast": ("ef2-fast-600m", "esmc-6b"), "ef2": ("ef2-fast-600m", "esmc-6b-full")}.get(
                family, (family, "esmc-600m"))
            # (a search chosen FIRST, as a reader's row might read for another model, then the model: ESMFold2 starts at
            # "ESM-C only" whatever the row read - which the rows check below sees)
            cdp.evaluate(ws, f"""(() => {{
              const g = (id) => document.getElementById(id);
              const set = (id, v) => {{ g(id).value = v; g(id).dispatchEvent(new Event('change', {{ bubbles: true }})); }};
              set('model-family', 'af3');
              if (g('msa-mode')) set('msa-mode', 'search');
              window.__entityList.set([{{ type: 'protein', value: '{S6}', copies: 1, modifications: [] }}]);
              set('model-family', '{row}');
              if (g('plm-mode')) set('plm-mode', '{plm}');
              return true;
            }})()""")
            if family in ("ef2-fast-600m", "ef2-fast", "ef2"):
                # the rows a reader sees for ESMFold2, which folds from its language model with an alignment as an
                # opt-in: the variant row and its step dial; no MSA row for the experimental checkpoint (ESM-C alone);
                # for the 6B two an MSA row starting at "ESM-C only" - though the reader's row read a search before
                rows = cdp.evaluate(ws, """(() => { const g = (id) => document.getElementById(id);
                  const m = g('msa-mode');
                  return { variant: g('plmGroup')?.hidden, msa: g('msaModeGroup')?.hidden, steps: g('af3CountGroup')?.hidden,
                           mode: m?.value, label: m?.selectedOptions?.[0]?.textContent?.trim() }; })()""")
                released = family != "ef2-fast-600m"
                want = {"variant": False, "msa": not released, "steps": False, "mode": "none", "label": "ESM-C only"}
                if rows != want:
                    bad.append(f"page: {family}'s rows read {rows} - want {want}")
            # (the gate stays offline: every arm folds without a search - ESMFold2's by its own default, the others set)
            cdp.evaluate(ws, """(() => { const m = document.getElementById('msa-mode'); if (!m) return true;
              m.value = 'none'; m.dispatchEvent(new Event('change', { bubbles: true })); return true; })()""")
            cdp.wait_for(ws, "!document.getElementById('predict').disabled", 60, "the fold button")
            shownBefore = cdp.evaluate(ws, "window.__trunkContactsShown ?? 0")
            started = time.time()
            cdp.evaluate(ws, "(document.getElementById('predict').click(), true)")
            cdp.wait_for(ws, """(() => { const t = document.getElementById('status-message').textContent;
              return / on (CUDA|Metal) .* done in |failed|refused|Error|error/.test(t); })()""", 300, f"a native fold of {family}")
            seconds = time.time() - started           # (click to the result's status line, as a reader waits)
            time.sleep(1.0)
            got = cdp.evaluate(ws, """(() => {
              const p = window.__lastPrediction();
              return { status: document.getElementById('status-message').textContent,
                       model: p?.model ?? null, family: p?.family ?? null,
                       pae: p?.confidence?.predictedAlignedError?.length ?? null,
                       paeTyped: p?.confidence?.predictedAlignedError instanceof Float32Array,
                       plddt: p?.confidence?.plddt?.length ?? null,
                       atoms: (p?.pdb ?? '').split(String.fromCharCode(10)).filter((l) => l.startsWith('ATOM')).length,
                       frames: (() => { const v = Object.values(window.py2dmol_viewers ?? {})[0]?.renderer;
                                        return v?.objectsData?.[v?.currentObjectName]?.frames?.length ?? 0; })(),
                       mapped: (() => { const v = Object.values(window.py2dmol_viewers ?? {})[0]?.renderer;
                                        const fs = v?.objectsData?.[v?.currentObjectName]?.frames ?? [];
                                        return { contact: fs.filter((f) => f.maps?.contact).length,
                                                 pae: fs.filter((f) => f.pae).length }; })(),
                       gpu: window.__readerGpu,
                       // the last frame against the one before it, as drawn (raw: no fitting) - the result lands where
                       // the frames are, or the picture jumps (web/pdb-superpose.js)
                       jump: (() => { const v = Object.values(window.py2dmol_viewers ?? {})[0]?.renderer;
                                      const fs = v?.objectsData?.[v?.currentObjectName]?.frames ?? [];
                                      if (fs.length < 2) return null;
                                      const a = fs[fs.length - 2].coords, b = fs[fs.length - 1].coords;
                                      if (!a || !b || a.length !== b.length) return 'no coords';
                                      let s = 0; for (let i = 0; i < a.length; i += 1)
                                        s += (a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2 + (a[i][2] - b[i][2]) ** 2;
                                      return Math.sqrt(s / a.length); })() };
            })()""")
            jump = f"{got['jump']:.2f} A" if isinstance(got["jump"], (int, float)) else str(got["jump"])
            print(f"page: {family:14s} live {'on ' if streamed else 'off'} {seconds:5.2f} s  {got['frames']:2d} frames {got['mapped']}  last step {jump:>7s}  {got['status'][:60]}")
            problems = []
            # every streamed fold of the AF3 lineage draws its trunk's contact maps before its first frame - the second
            # fold in a page too, where the first fold's frames used to hold the panel and drop them all
            shown = cdp.evaluate(ws, "window.__trunkContactsShown ?? 0") - shownBefore
            if streamed and family in ("af3", "protenix2", "intellifold2", "boltz2", "rosettafold3", "opendde") and shown < 1:
                problems.append("no trunk contact map was drawn")
            if got["jump"] is not None and (not isinstance(got["jump"], (int, float)) or got["jump"] > 2.0):
                problems.append(f"the result's frame lands {got['jump']} A from the last streamed one, as drawn")
            if f" on {NATIVE_NAME} " not in got["status"] or "done in" not in got["status"]:
                problems.append(f"status {got['status']!r}")
            if not (got["model"] or "").endswith(f"({NATIVE_NAME})"):
                problems.append(f"model {got['model']!r}")
            if got["pae"] != tokens * tokens or not got["paeTyped"] or got["plddt"] != tokens:
                problems.append(f"PAE {got['pae']} (typed {got['paeTyped']}), pLDDT {got['plddt']}")
            if got["atoms"] < tokens * 4:
                problems.append(f"{got['atoms']} atoms")
            if not streamed:
                if got["frames"] > 1:
                    problems.append(f"Live preview off and the viewer still holds {got['frames']} frames")
            elif family in ("af3", "protenix2", "ef2-fast-600m") and (got["frames"] < 5 or got["mapped"]["contact"] < 5):
                problems.append(f"the viewer holds {got['frames']} frames, {got['mapped']['contact']} with a contact map")
            if streamed and family == "monomer" and (got["mapped"]["pae"] < 1 or got["mapped"]["contact"] < 1):
                problems.append(f"the viewer's AF2 passes carry {got['mapped']} - no streamed PAE or contacts")
            if got["gpu"]:
                problems.append("the reader's browser asked for a WebGPU adapter")
            bad += [f"page {family}{'' if streamed else ' (live off)'}: {p}" for p in problems]
    finally:
        if reader is not None:
            reader.terminate()
        # 🔴 SIGINT, NOT SIGTERM: the broker stops its own headless Chrome on the way out of a
        # KeyboardInterrupt, and a SIGTERM skips that and orphans it on the CDP port the next run wants
        broker.send_signal(signal.SIGINT)
        try:
            broker.wait(timeout=20)
        except subprocess.TimeoutExpired:
            broker.kill()


def main():
    offline = "--offline" in sys.argv
    plan = cases(offline)
    only = next((a[7:] for a in sys.argv if a.startswith("--only=")), None)   # (a substring of the case names)
    if only is not None:
        plan = [case for case in plan if only in case[0]]
    if "--no-af3" in sys.argv:
        plan = [case for case in plan if case[1]["family"] != "af3"]
    plan = [(name, payload, reference, METAL_BARS.get(name, bar) if NATIVE == "metal" else bar, tokens, refusal)
            for name, payload, reference, bar, tokens, refusal in plan]
    if NATIVE == "metal":
        plan = [(name, payload, None, None, None, METAL_REFUSALS[name]) if name in METAL_REFUSALS else (name, payload, *rest)
                for name, payload, *rest in plan]
    worker = subprocess.Popen([sys.executable, os.path.join(REPO, "python", "localfold", "worker.py")], cwd=REPO,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/tmp/localfold-cuda-check.log", "w"),
                              text=True, bufsize=1)
    first = json.loads(worker.stdout.readline())
    bad = [] if first.get("kind") == "ready" else [f"the worker's first line was {first}, not ready"]
    previous = None      # (the last fold's PDB: a BioEmu2 propagation's starting frame)
    folded = {}
    for name, payload, reference, bar, tokens, refusal in plan:
        payload = dict(payload)
        expected_bonds = payload.pop("ligandBonds", None)
        samples = payload.pop("samples", None)
        same_as = payload.pop("samePdbAs", None)
        other_than = payload.pop("otherPdbThan", None)
        if payload.get("bioemu2Frame") == "previous":
            payload["bioemu2Frame"] = previous
        started = time.time()
        worker.stdin.write(json.dumps(payload) + "\n")
        worker.stdin.flush()
        kinds, result, fractions = [], None, []
        for line in worker.stdout:
            event = json.loads(line)
            kinds.append(event["kind"])
            if event["kind"] == "progress":
                fractions.append(event["payload"])
            if event["kind"] == "result":
                result = event["payload"]
                break
        seconds = time.time() - started
        if result is None:
            bad.append(f"{name}: the worker exited")
            break
        if refusal is not None:
            said = result.get("error", "")
            ok = refusal in said and "Error:" not in said.split(" ")[0]
            print(f"{name:44s} {'refused' if ok else 'WRONG'}: {said[:110]}")
            if not ok:
                bad.append(f"{name}: wanted a refusal naming {refusal!r}, got {result.get('error') or result.get('status')}")
            continue
        if "error" in result:
            print(f"{name:44s} FAILED: {result['error'][:200]}")
            bad.append(f"{name}: {result['error'][:200]}")
            continue
        c = result.get("confidence") or {}
        n = len(c.get("plddt") or [])
        problems = []
        if result.get("native") != NATIVE:
            problems.append(f"marked native {result.get('native')!r}, not {NATIVE!r}")
        previous = result.get("pdb")
        folded[name] = result.get("samples") or [previous]
        if same_as is not None and folded[name] != folded.get(same_as):
            problems.append(f"not the same structures as {same_as!r}")
        if other_than is not None and folded[name] == folded.get(other_than):
            problems.append(f"the same structures as {other_than!r} at another seed")
        if payload["family"] == "bioemu2":
            n = len((result.get("tokens") or {}).get("chainIds") or [])
            got = result.get("samples") or []
            if len(got) != samples:
                problems.append(f"{len(got)} samples, not {samples}")
            if c.get("plddt") is not None or c.get("meanPlddt") is not None or c.get("ptm") is not None:
                problems.append("a confidence BioEmu2 does not predict")
            if len(c.get("contactProbs") or []) != n * n:
                problems.append(f"contact map {len(c.get('contactProbs') or [])} for {n} tokens")
            if tokens is not None and n != tokens:
                problems.append(f"{n} tokens, not {tokens}")
            # (a propagation's every step near its starting frame; an ensemble's samples each a chain)
            if reference[0] == "chain":
                spacing = [ca_spacing(p) for p in got]
                worst = max(abs(v - 3.8) for v in spacing) if spacing else None
                if worst is None or worst > 0.3:
                    problems.append(f"a sample's median CA-CA {spacing} is not a chain's 3.8 A")
                bar = "CA-CA 3.8 +/- 0.3"
            else:
                against = reference
                if reference[0] == "previous":
                    with open("/tmp/localfold-cuda-check-frame.pdb", "w") as handle:
                        handle.write(payload["bioemu2Frame"])
                    against = ("/tmp/localfold-cuda-check-frame.pdb", reference[1])
                worst = max((score(p, *against) or 99) for p in got) if got else None
                if worst is None or worst > bar:
                    problems.append(f"a sample's RMSD {worst} past {bar} A")
            if expected_bonds is not None:
                count, closest = ligand_bonds(result.get("pdb") or "")
                if count != expected_bonds or closest is None or closest < 1.0:
                    problems.append(f"the ligand has {count} bonds (want {expected_bonds}), closest atoms {closest} A")
            if "status" not in kinds:
                problems.append(f"events {kinds}")
            print(f"{name:44s} worst sample RMSD {worst} A (bar {bar})  {n} tokens  {len(got)} samples  {seconds:5.1f} s"
                  f"  | {result.get('status')}")
            bad += [f"{name}: {p}" for p in problems]
            continue
        if len(c.get("predictedAlignedError") or []) != n * n:
            problems.append(f"PAE {len(c.get('predictedAlignedError') or [])} for {n} tokens")
        if len((result.get("tokens") or {}).get("chainIds") or []) != n:
            problems.append("the token layout is not one entry a token")
        if tokens is not None and n != tokens:
            problems.append(f"{n} tokens, not {tokens}")
        if not (0 < (c.get("meanPlddt") or 0) <= 100) or c.get("ptm") is None:
            problems.append(f"pLDDT {c.get('meanPlddt')} / pTM {c.get('ptm')}")
        if "status" not in kinds or kinds.count("progress") < 2:
            problems.append(f"events {kinds}")
        # what each port streams (--frames, through the binaries' AsyncTap), the bar moving with it: the
        # AF3 lineage its sampler and each trunk pass's contact map, AF2 each pass (structure + scores, and
        # contacts where the bundle carries the distogram head), ESMFold2 its trunk's contacts and sampler
        family = payload["family"].split("-")[0]
        want = {"frame": 5, "contacts": 1} if family in ("af3", "openbind0", "opendde", "boltz2", "protenix2",
                                                         "intellifold2", "rosettafold3", "chai1", "ef2") \
            else {"frame": 1, "scores": 1, **({"contacts": 1} if family == "monomer" else {})}
        for kind, least in want.items():
            if kinds.count(kind) < least:
                problems.append(f"{kinds.count(kind)} {kind} events streamed")
        if fractions != sorted(fractions):
            problems.append("the progress bar went backwards")
        if "stopping early" in name and "converged at" not in (result.get("status") or ""):
            problems.append("the page's early stop did not stop it (or did not say so)")
        if expected_bonds is not None:
            count, closest = ligand_bonds(result.get("pdb") or "")
            if count != expected_bonds or closest is None or closest < 1.0:
                problems.append(f"the ligand has {count} bonds (want {expected_bonds}), closest atoms {closest} A")
        rmsd = score(result.get("pdb") or "", *reference)
        if rmsd is None or rmsd > bar:
            problems.append(f"RMSD {rmsd} past {bar} A")
        print(f"{name:44s} RMSD {rmsd} A (bar {bar})  {n} tokens  {kinds.count('frame'):2d}f {kinds.count('contacts')}c {kinds.count('scores')}s  {seconds:5.1f} s"
              f"  | {result.get('status')}")
        bad += [f"{name}: {p}" for p in problems]
    worker.stdin.close()
    worker.wait(timeout=60)
    if "--no-page" not in sys.argv:
        page_arm(bad)
    if bad:
        print("\n" + "\n".join(bad) + "\n\nnative worker: FAILED (its log: /tmp/localfold-cuda-check.log)")
        sys.exit(1)
    print("\nnative worker: ok")


if __name__ == "__main__":
    main()
