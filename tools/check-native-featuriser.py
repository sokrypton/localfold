"""The native featurisers against the JavaScript exporters they replace: byte for byte.

    python3 tools/check-native-featuriser.py                 # the whole corpus, every family it names
    python3 tools/check-native-featuriser.py --only=6mrr-sep # cases whose name contains it
    python3 tools/check-native-featuriser.py --keep          # leave both outputs in /tmp/claude-1000/nf
    python3 tools/check-native-featuriser.py --ccd=ccd/components.cif   # the native side reads a local CCD

Each case runs the JavaScript exporter (cuda/af3/export-model.mjs --no-weights, cuda/af2/export_input.mjs,
cuda/ef2/export_input.mjs) and the native one (cuda/featurise/*-featurise) on the same arguments and holds
them to the SAME BYTES: model.idx's text, model.bin entry by entry, and the PDB records. The first entry that
differs is named with its first differing element, because "the files differ" says nothing about where.

🔴 A REFUSAL MUST BE THE SAME REFUSAL: a case either exporter refuses passes only when both refuse, in the same
words - the page's sentence is the answer a reader sees, so the native port may not say something else.

--ccd=<components.cif> hands the NATIVE featurisers wwPDB's whole dictionary (`cuda/featurise/fetch-weights ccd`) where
the JavaScript still fetches each component from the RCSB: the two must still agree byte for byte, which is what
says a local dictionary folds what the page folds.
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# `--js-float16array` only where this node lacks the type and knows the flag, as `npm test` asks: node 26 ships
# Float16Array and refuses the flag ("bad option"), which failed every JavaScript arm here
def _float16_flag():
    try:
        if subprocess.run(["node", "-e", "process.exit(typeof Float16Array === 'undefined' ? 0 : 1)"]).returncode != 0:
            return []
        known = "--js-float16array" in subprocess.run(["node", "--v8-options"], capture_output=True, text=True).stdout
        return ["--js-float16array"] if known else []
    except OSError:
        return []
NODE = ["node", *_float16_flag(), "--max-old-space-size=24000"]
WORK = "/tmp/claude-1000/nf"
FIX = os.path.join(REPO, "tools", "fixtures")
AF3_EXAMPLES = os.path.join(FIX, "af3-jobs")

SEQ_6MRR = "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK"
TEST_QUERY = open(os.path.join(FIX, "test.a3m")).read().split(">")[1].splitlines()[1].strip()
AF3_FAMILIES = ["af3", "openbind0", "opendde", "boltz2", "protenix2", "intellifold2", "rosettafold3", "chai1"]


def job(name, sequences, **extra):
    return {"name": name, "modelSeeds": [1], "dialect": "alphafold3", "version": 1, "sequences": sequences, **extra}


def protein(chain, sequence, **extra):
    return {"protein": {"id": chain, "sequence": sequence, **extra}}


def chain_sequence(name, chain):
    """A crystal's chain, one letter a residue, from its CA records"""
    three = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
             "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
             "TYR": "Y", "VAL": "V"}
    out, seen = [], set()
    for line in open(os.path.join(FIX, f"{name}-crystal.pdb")):
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain and line[22:27] not in seen:
            seen.add(line[22:27])
            out.append(three.get(line[17:20], "X"))
    return "".join(out)


def synthetic_a3m(name, query, rows, seed):
    """A deterministic alignment for query: substitutions, gaps, insertions and a duplicated row, written to WORK"""
    import random
    rng = random.Random(seed)
    letters = "ACDEFGHIKLMNPQRSTVWY"
    out = [f">query\n{query}"]
    previous = None
    for r in range(rows):
        if previous is not None and r % 7 == 3:
            out.append(f">dup{r}\n{previous}")      # (deduplication must drop it)
            continue
        row = []
        for c in query:
            x = rng.random()
            row.append("-" if x < 0.08 else rng.choice(letters) if x < 0.4 else c)
            if rng.random() < 0.03:
                row.append(rng.choice(letters).lower() * rng.randint(1, 3))
        previous = "".join(row)
        out.append(f">hit{r} desc {r}\n{previous}")
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, f"{name}.a3m")
    open(path, "w").write("\n".join(out) + "\n")
    return path


def cases():
    """(name, port, job dict or None, extra args, families)"""
    out = [
        ("6mrr", "af3", job("t", [protein("A", SEQ_6MRR)]), [], AF3_FAMILIES),
        ("6mrr-sep-gol", "af3", job("t", [protein("A", SEQ_6MRR, modifications=[{"ptmType": "SEP", "ptmPosition": 3}]),
                                          {"ligand": {"id": "B", "ccdCodes": ["GOL"]}}]), [], AF3_FAMILIES),
        ("sep-end", "af3", job("t", [protein("A", "MKTAYIAKQRS", modifications=[{"ptmType": "SEP", "ptmPosition": 11}])]),
         [], AF3_FAMILIES),
        # an unknown amino acid mid-chain and at the C-terminus (openbind0: the CCD's UNK, N CA C O CB CG)
        ("unknown-residues", "af3", job("t", [protein("A", "MKTXAYIAKQRX")]), [], AF3_FAMILIES),
        # every family's rule for declared bonds (featurise.js declaredPolymerBonds): a disulfide within a chain, an
        # adjacent pair (left to the chain by protenix), a cross-chain one, and a modified residue's bond to a ligand
        ("bond-rules", "af3", job("t", [protein("A", "CKTCYSAKC", modifications=[{"ptmType": "SEP", "ptmPosition": 6}]),
                                        protein("B", "CRSC"), {"ligand": {"id": "C", "ccdCodes": ["GOL"]}}],
                                  bondedAtomPairs=[[["A", 1, "SG"], ["A", 4, "SG"]], [["A", 1, "C"], ["A", 2, "N"]],
                                                   [["A", 9, "SG"], ["B", 1, "SG"]], [["A", 6, "OG"], ["C", 1, "C1"]]]),
         [], AF3_FAMILIES),
        # (a bond naming an atom its residue does not have is refused by name - here chai1's and openbind0's dropped
        # terminal OXT, and a mistyped name for everyone - not bonded to the residue regardless)
        # (protenix and OpenDDE drop the leaving group at a declared bond: a head-to-tail cyclic peptide's C-terminal
        # OXT, a glycan's O1 at each glycosidic C1; the rest keep both)
        ("bond-leaving-cyclic", "af3", job("t", [protein("A", "GRCKSTAG")], bondedAtomPairs=[[["A", 8, "C"], ["A", 1, "N"]]]),
         [], AF3_FAMILIES),
        ("bond-leaving-glycan", "af3", job("t", [protein("A", "MKNKTAQ"), {"ligand": {"id": "B", "ccdCodes": ["NAG", "NAG"]}}],
                                           bondedAtomPairs=[[["A", 3, "ND2"], ["B", 1, "C1"]], [["B", 1, "O4"], ["B", 2, "C1"]]]),
         [], AF3_FAMILIES),
        ("bond-missing-atom", "af3", job("t", [protein("A", "CKTCYSAKC"), {"ligand": {"id": "B", "ccdCodes": ["GOL"]}}],
                                         bondedAtomPairs=[[["A", 9, "OXT"], ["B", 1, "C1"]]]), [], AF3_FAMILIES),
        ("bond-mistyped-atom", "af3", job("t", [protein("A", "CKTCYSAKC"), {"ligand": {"id": "B", "ccdCodes": ["GOL"]}}],
                                          bondedAtomPairs=[[["A", 1, "SGX"], ["B", 1, "C1"]]]), [], AF3_FAMILIES),
        # (the unknown nucleotide, N: AF3's restype 30 and its backbone, its pseudo-beta the first held atom)
        ("unknown-nucleotide", "af3", job("t", [{"dna": {"id": "A", "sequence": "NACGNTACGTN"}},
                                                {"rna": {"id": "B", "sequence": "NACGNUACGUN"}}]), [], AF3_FAMILIES),
        ("dimer-dna", "af3", job("t", [protein(["A", "B"], "MKTAYIAKQRQISFVKSHFSRQ"),
                                       {"dna": {"id": "C", "sequence": "ACGTTGCA"}}, {"rna": {"id": "D", "sequence": "ACGUU"}}]),
         [], AF3_FAMILIES),
        ("msa-test", "af3", job("t", [protein("A", TEST_QUERY)]), ["--a3m=@test.a3m"], AF3_FAMILIES),
        # (modified nucleotides at a chain's 5' end and side by side - openbind0 drops their leaving atoms at the ends
        # too and links only two atomised residues - and a standalone glycan, whose O1 AF3's rule drops)
        ("modified-ends", "af3", job("t", [protein("A", "MKTAYIAKQR"),
                                           {"dna": {"id": "B", "sequence": "GACTGA", "modifications": [
                                               {"modificationType": "6OG", "basePosition": 1},
                                               {"modificationType": "6MA", "basePosition": 2}]}},
                                           {"rna": {"id": "C", "sequence": "GACUGA", "modifications": [
                                               {"modificationType": "2MG", "basePosition": 1}]}}]), [], AF3_FAMILIES),
        # (a protein and an RNA chain, each with its own alignment inline - boltz2 and rf3 read the protein's alone)
        ("prot-rna-inline-msa", "af3", job("t", [
            protein("A", "GWSTELEKHREELKEFLKKEG", unpairedMsa=">q\nGWSTELEKHREELKEFLKKEG\n>p1\nGWSTELEKHREELKEFLKKEA\n"
                                                         ">p2\nGWSTEL-KHREELKEFLKKEG\n>p3\nGWSTEL-KHREELKEFLKKEG\n",
                    pairedMsa=""),
            {"rna": {"id": "B", "sequence": "GGCUAGCC", "unpairedMsa": ">q\nGGCUAGCC\n>r1\nGGCUAGCA\n>r2\nGG-UAGCC\n"}},
            {"ligand": {"id": "C", "ccdCodes": ["GOL"]}}]), [], AF3_FAMILIES),
        # (two chains with paired rows inline: protenix and OpenDDE fold the paired rows into each unpaired MSA)
        ("paired-inline-msa", "af3", job("t", [
            protein("A", "PREACHINGS", pairedMsa=">query\nPREACHINGS\n>s1\n--EACHINGS\n>s2\nPREACHINGS\n",
                    unpairedMsa=">query\nPREACHINGS\n>u1\nPREA--INGS\n>u2\nPREAcCHINGS\n"),
            protein("B", "LITERALLY", pairedMsa=">query\nLITERALLY\n>s1\nLITER-LLY\n>s2\n---------\n",
                    unpairedMsa=">query\nLITERALLY\n")]), [], AF3_FAMILIES),
        # (a template holding a phosphoserine and altlocs, 5K9P chain A: every family's template policy - the parent,
        # an UNK with its N, an UNK with N/CA/C/CB, the hit dropped, the first altloc - and AF3's arginine fix)
        ("template-modified", "af3", json.load(open(os.path.join(REPO, "tools", "fixtures", "template-5k9p-sep.json"))),
         [], AF3_FAMILIES),
        ("glycan-standalone", "af3", job("t", [protein("A", "MKTAYIAKQR"), {"ligand": {"id": "B", "ccdCodes": ["NAG"]}},
                                               {"ligand": {"id": "C", "ccdCodes": ["NAG", "BMA"]}}],
                                         bondedAtomPairs=[[["C", 1, "O4"], ["C", 2, "C1"]]]), [], AF3_FAMILIES),
    ]
    caj, brs_a, brs_d = chain_sequence("5caj", "A"), chain_sequence("1brs", "A"), chain_sequence("1brs", "D")
    out += [
        ("tmpl-5caj", "af3", job("t", [protein("A", caj)]), ["--template=@F/5caj-crystal.pdb:A@0"], AF3_FAMILIES),
        # (a sequence the template does not match: the page's local alignment maps it; and a second, unrelated slot)
        ("tmpl-aligned", "af3", job("t", [protein("A", caj[:40] + caj[52:150].replace("L", "I") + caj[160:])]),
         ["--template=@F/5caj-crystal.pdb:A@0,@F/1qys-crystal.pdb:A@0"], AF3_FAMILIES),
        ("tmpl-1brs-merged", "af3", job("t", [protein("A", brs_a), protein("D", brs_d)]),
         ["--template=@F/1brs-crystal.pdb:A@0+@F/1brs-crystal.pdb:D@1"], AF3_FAMILIES),
        ("tmpl-1brs-nospan", "af3", job("t", [protein("A", brs_a), protein("D", brs_d)]),
         ["--template=@F/1brs-crystal.pdb:A@0+@F/1brs-crystal.pdb:D@1", "--no-span-chains"], AF3_FAMILIES),
        # (a modified residue and a ligand shift every token after them)
        ("tmpl-6mrr-sep-gol", "af3", job("t", [protein("A", SEQ_6MRR, modifications=[{"ptmType": "SEP", "ptmPosition": 3}]),
                                               {"ligand": {"id": "B", "ccdCodes": ["GOL"]}}]),
         ["--template=@F/6mrr-crystal.pdb:A@0"], AF3_FAMILIES),
        ("tmpl-four", "af3", job("t", [protein("A", caj)]),
         ["--template=" + ",".join(["@F/5caj-crystal.pdb:A@0"] * 4)], AF3_FAMILIES),
    ]
    # chai-1's own MSA inputs (shared/af3/featurise/chai-msa.js): a chain's uint8 profile over its deduplicated rows (600
    # rows: a column's count passes 256 and wraps), the single-sequence context of a modified protein and a DNA chain
    # with none, the duplicate rows masked - and an alignment of nothing but the query, which is none (ColabFold ebbd90d)
    wrap = open(synthetic_a3m("chai-wrap", SEQ_6MRR, 600, 3)).read()
    out += [
        ("chai-msa-complex", "af3", job("t", [protein("A", SEQ_6MRR, unpairedMsa=wrap, pairedMsa=""),
                                              protein("B", "MKTSYIAKQRQISFVK", modifications=[{"ptmType": "SEP", "ptmPosition": 4}],
                                                      unpairedMsa="", pairedMsa=""),
                                              {"dna": {"id": "C", "sequence": "ACGTTGCA"}}]), [], ["chai1"]),
        ("chai-query-only", "af3", job("t", [protein("A", SEQ_6MRR, unpairedMsa=f">q\n{SEQ_6MRR}\n", pairedMsa="")]), [],
         ["chai1"]),
    ]
    # 🔴 THE NETWORK ARM (--network): both exporters search api.colabfold.com for the same queries - the server
    # answers one query the same way twice, so the alignments, the pairing, the merge and the template hits must agree
    out += [
        ("net-search-6mrr", "af3", job("t", [protein("A", SEQ_6MRR)]), ["--search"], ["af3", "boltz2"], "network"),
        ("net-search-1brs", "af3", job("t", [protein("A", brs_a), protein("D", brs_d)]), ["--search"], ["af3", "rosettafold3"], "network"),
        ("net-search-templates", "af3", job("t", [protein("A", SEQ_6MRR)]), ["--search-templates"], ["af3", "protenix2"], "network"),
        ("net-search-chain-template", "af3", job("t", [protein("A", brs_a), protein("D", brs_d)]),
         ["--search", "--template-search-chains=1"], ["af3", "boltz2"], "network"),
    ]
    # AlphaFold 2 (cuda/af2/export_input.mjs): "families" are the bundles, monomer and multimer
    sa, sd = synthetic_a3m("brs-a", brs_a, 300, 1), synthetic_a3m("brs-d", brs_d, 200, 2)
    pa, pd = synthetic_a3m("brs-pa", brs_a, 60, 3), synthetic_a3m("brs-pd", brs_d, 60, 4)
    m6 = synthetic_a3m("6mrr", SEQ_6MRR, 900, 5)
    deep = synthetic_a3m("deep-5caj", caj, 8000, 9)      # (past 1 MiB: the JavaScript exporter's worker-thread path)
    # past AF-Multimer's crop (shared/input/chains.js MULTIMER_MSA_CROP): 1,500 paired rows a chain cut to 1,024, the
    # unpaired to what is left of 2,048, and a homodimer's 2,600 to 2,048 - the fixtures above never reach either
    ca, cd = synthetic_a3m("crop-a", brs_a, 2600, 11), synthetic_a3m("crop-d", brs_d, 2600, 12)
    cpa, cpd = synthetic_a3m("crop-pa", brs_a, 1500, 13), synthetic_a3m("crop-pd", brs_d, 1500, 14)
    c6 = synthetic_a3m("crop-6mrr", SEQ_6MRR, 2600, 15)
    # the self-MSA rule and the non-protein MSA (featurise.js dedupeSelfMsa, nativeNonproteinMsa): an alignment of the
    # query alone - one row for the families that dedupe, two for AlphaFold 3 and OpenBind-0 - and a protein's
    # alignment beside an RNA chain and a ligand (opendde and protenix2: the ligand UNK in every row and its profile,
    # the RNA's all-gap rows its query)
    query_only = os.path.join(WORK, "6mrr-query-only.a3m")
    with open(query_only, "w") as f:
        f.write(f">query\n{SEQ_6MRR}\n")
    out += [
        ("self-msa", "af3", job("t", [protein("A", SEQ_6MRR)]), [f"--a3m={query_only}"], AF3_FAMILIES),
        ("nonprotein-msa", "af3", job("t", [protein("A", SEQ_6MRR), {"rna": {"id": "B", "sequence": "GGACUUCGGUCC"}},
                                            {"ligand": {"id": "C", "ccdCodes": ["GOL"]}}]), [f"--a3m={m6}"], AF3_FAMILIES),
    ]
    AF2 = ["monomer", "multimer"]
    sample_template = {"mmcif": open(os.path.join(FIX, "bioemu2", "6mrr-sample.cif")).read(),
                       "queryIndices": list(range(2, 42)), "templateIndices": list(range(0, 40))}
    out += [
        ("af2-6mrr", "af2", job("t", [protein("A", SEQ_6MRR)]), [], AF2),
        ("af2-6mrr-msa", "af2", job("t", [protein("A", SEQ_6MRR)]), [f"--a3m={m6}", "--seed=7", "--recycles=2"], AF2),
        ("af2-6mrr-shallow", "af2", job("t", [protein("A", SEQ_6MRR)]), [f"--a3m={m6}", "--max-msa=16", "--max-extra=40"], AF2),
        ("af2-deep", "af2", job("t", [protein("A", caj)]), [f"--a3m={deep}"], ["monomer"]),
        ("af2-test59", "af2", job("t", [protein("A", TEST_QUERY)]), ["--a3m=@test.a3m", "--max-msa=508"], AF2),
        ("af2-1brs", "af2", job("t", [protein("A", brs_a), protein("D", brs_d)]), [], AF2),
        ("af2-1brs-a3ms", "af2", job("t", [protein("A", brs_a), protein("D", brs_d)]), [f"--a3m={sa},{sd}"], AF2),
        ("af2-1brs-paired", "af2", job("t", [protein("A", brs_a), protein("D", brs_d)]),
         [f"--a3m={sa},{sd}", f"--paired-a3m={pa},{pd}"], AF2),
        ("af2-homodimer", "af2", job("t", [protein(["A", "B"], SEQ_6MRR)]), [f"--a3m={m6},{m6}"], AF2),
        ("af2-1brs-crop", "af2", job("t", [protein("A", brs_a), protein("D", brs_d)]),
         [f"--a3m={ca},{cd}", f"--paired-a3m={cpa},{cpd}", "--recycles=1"], AF2),
        ("af2-homodimer-crop", "af2", job("t", [protein(["A", "B"], SEQ_6MRR)]), [f"--a3m={c6},{c6}", "--recycles=1"], AF2),
        ("af2-5caj-template", "af2", job("t", [protein("A", caj)]), ["--template=@F/5caj-crystal.pdb:A"], AF2),
        ("af2-1brs-template", "af2", job("t", [protein("A", brs_a), protein("D", brs_d)]),
         ["--template=@F/1brs-crystal.pdb:A@0+@F/1brs-crystal.pdb:D@1"], AF2),
        ("af2-ligand-refused", "af2", job("t", [protein("A", SEQ_6MRR), {"ligand": {"id": "B", "ccdCodes": ["GOL"]}}]), [], AF2),
        # a job's own inline template, aligned by its mapping; one chain of two templated, the other padded as
        # AlphaFold-Multimer's pipeline pads it (--template-pad=zero, RFD4's AF2 oracle); an alignment unmasked
        # (--msa-mask=0, the oracle's pct_seq_mask)
        ("af2-job-template", "af2", job("t", [protein("A", SEQ_6MRR, templates=[sample_template])]), [], AF2),
        ("af2-job-template-pad", "af2", job("t", [protein("A", SEQ_6MRR, templates=[sample_template]), protein("B", brs_d)]),
         ["--template-pad=zero"], ["multimer"]),
        ("af2-6mrr-unmasked", "af2", job("t", [protein("A", SEQ_6MRR)]), [f"--a3m={m6}", "--msa-mask=0"], AF2),
    ]
    # ESMFold2 (cuda/ef2/export_input.mjs)
    EF = ["ef2"]
    out += [
        ("ef2-6mrr", "ef2", job("t", [protein("A", SEQ_6MRR)]), [], EF),
        # (the unknown nucleotide: res_type 27 / 32 and its backbone at the origin, as esm's tokenize_nucleotide)
        ("ef2-unknown-nucleotide", "ef2", job("t", [protein("A", SEQ_6MRR), {"dna": {"id": "B", "sequence": "NACGNTACGTN"}},
                                                    {"rna": {"id": "C", "sequence": "NACGNUACGUN"}}]), [], EF),
        ("ef2-6mrr-sep-gol", "ef2", job("t", [protein("A", SEQ_6MRR, modifications=[{"ptmType": "SEP", "ptmPosition": 3}]),
                                                   {"ligand": {"id": "B", "ccdCodes": ["GOL"]}}]), [], EF),
        ("ef2-dimer-dna", "ef2", job("t", [protein(["A", "B"], "MKTAYIAKQRQISFVKSHFSRQ"),
                                                {"dna": {"id": "C", "sequence": "ACGTTGCA"}}, {"rna": {"id": "D", "sequence": "ACGUU"}}]),
         [], EF),
        ("ef2-biotin-smiles", "ef2", job("t", [protein("A", SEQ_6MRR),
                                                    {"ligand": {"id": "B", "smiles": "OC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12"}}]), [], EF),
        ("ef2-1brs-a3m", "ef2", job("t", [protein("A", brs_a), protein("D", brs_d)]), [f"--a3m={sa},{sd}"], EF),
    ]
    # BioEmu2 (cuda/bioemu2/export_input.mjs): its own featuriser (shared/bioemu2/featurise.js), the page's SMILES
    # templates and alignment split (shared/bioemu2/inputs.js)
    B = ["bioemu2"]
    BF = os.path.join(FIX, "bioemu2")
    out += [(f"bioemu2-{n}", "bioemu2", os.path.join(BF, f"{n}.json"), [], B)
            for n in ("6mrr", "6mrr-gol-sep", "rna-dna", "smiles", "6mrr-nag-ret", "heteromer")]
    out += [
        ("bioemu2-glycan", "bioemu2", os.path.join(FIX, "glycan-standalone.json"), [], B),
        ("bioemu2-equilibrium-contacts", "bioemu2", os.path.join(BF, "6mrr.json"),
         ["--mode=equilibrium", f"--contacts={BF}/contacts.json"], B),
        ("bioemu2-a3m-per-chain", "bioemu2", os.path.join(BF, "heteromer-nomsa.json"),
         [f"--a3m={BF}/msa/u0.a3m,{BF}/msa/u1.a3m,{BF}/msa/u2.a3m", f"--paired-a3m={BF}/msa/p0.a3m,{BF}/msa/p1.a3m,{BF}/msa/p2.a3m"], B),
        ("bioemu2-a3m-merged", "bioemu2", os.path.join(BF, "heteromer-nomsa.json"),
         [f"--a3m={BF}/msa/merged.a3m", f"--paired-a3m={BF}/msa/merged-paired.a3m"], B),
        ("bioemu2-propagate-cif", "bioemu2", os.path.join(BF, "6mrr.json"),
         [f"--propagate-from={BF}/6mrr-sample.cif", "--lag-ns=2.5", "--propagation-steps=3"], B),
        ("bioemu2-propagate-pdb", "bioemu2", os.path.join(BF, "6mrr.json"), [f"--propagate-from={BF}/6mrr-sample.pdb"], B),
        ("bioemu2-wrong-frame-refused", "bioemu2", os.path.join(BF, "6mrr-gol-sep.json"), [f"--propagate-from={BF}/6mrr-sample.pdb"], B),
    ]
    # ...and the steering potentials' tables (--steering: shared/bioemu2/steering-tables.js, the CCD's ligand half too),
    # over the stereochemistry the vendor ships (fetched once into ~/.cache/alphafold3/bioemu2)
    out += [(f"bioemu2-steering-{n}", "bioemu2", os.path.join(BF, f"{n}.json"), ["--steering"], B)
            for n in ("6mrr", "6mrr-gol-sep", "rna-dna", "smiles", "6mrr-nag-ret", "heteromer")]
    out.append(("bioemu2-steering-glycan", "bioemu2", os.path.join(FIX, "glycan-standalone.json"), ["--steering"], B))
    if os.path.isdir(AF3_EXAMPLES):
        for name in sorted(os.listdir(AF3_EXAMPLES)):
            if name.endswith(".json"):
                out.append((f"example-{name[:-5]}", "af3", os.path.join(AF3_EXAMPLES, name), [], ["af3", "boltz2", "rosettafold3"]))
                out.append((f"ef2-example-{name[:-5]}", "ef2", os.path.join(AF3_EXAMPLES, name), [], ["ef2"]))
                out.append((f"bioemu2-example-{name[:-5]}", "bioemu2", os.path.join(AF3_EXAMPLES, name), [], ["bioemu2"]))
    return out


def resolve_cases():
    """(name, request, network) for resolve-templates against cuda/resolve_templates.mjs"""
    caj = open(os.path.join(FIX, "5caj-crystal.pdb")).read()
    upload = {"kind": "upload", "text": caj, "source": "A", "filename": "5caj.pdb"}
    return [
        ("resolve-upload", {"entities": [
            {"type": "protein", "value": "MKTAYIAKQR", "copies": 2, "template": upload},
            {"type": "dna", "value": "ACGT", "copies": 1},
            {"type": "protein", "value": "MKTAYIAKQR", "copies": 1, "template": {"kind": "search"}},
            {"type": "ligand", "value": "GOL", "copies": 1},
            {"type": "protein", "value": "GGGG", "copies": 1, "template": {"kind": "pdb", "source": ""}},
            {"type": "protein", "value": "GGGG", "copies": 1, "template": {"kind": "upload", "text": caj}}]}, False),
        ("resolve-pdb", {"entities": [{"type": "protein", "value": "MKTAYIAKQR", "copies": 1,
                                       "template": {"kind": "pdb", "source": "1QYS_A"}}]}, True),
        # 🔴 AlphaFold DB ANSWERS NODE'S fetch WITH 403 (and serves curl and a browser), so the JavaScript resolver
        # cannot be this case's reference from a command line: the native side is checked on its own output
        ("resolve-afdb", {"entities": [{"type": "protein", "value": "MKTAYIAKQR", "copies": 1,
                                        "template": {"kind": "afdb", "source": "P69905"}}]}, "native-only"),
    ]


def check_resolve(name, request, network):
    base = os.path.join(WORK, name)
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base)
    path = os.path.join(base, "request.json")
    json.dump(request, open(path, "w"))
    nv = run([os.path.join(REPO, "cuda", "featurise", "resolve-templates"), path, base + "/native"])
    if network == "native-only":
        if nv[0] != 0:
            return f"native exit {nv[0]} ({refusal(nv[1])})"
        [entry] = json.loads(nv[1].strip().splitlines()[-1])
        text = open(entry["file"]).read()
        atoms = sum(line.startswith("ATOM") for line in text.splitlines())
        if entry["kind"] != "afdb" or atoms < 100 or entry["source"] != request["entities"][0]["template"]["source"]:
            return f"{entry} with {atoms} ATOM records"
        return None
    js = run(["node", os.path.join(REPO, "cuda", "resolve_templates.mjs"), path, base + "/js"])
    if js[0] != 0 or nv[0] != 0:
        if js[0] != 0 and nv[0] != 0 and refusal(js[1]) == refusal(nv[1]):
            return None
        return f"JS exit {js[0]} ({refusal(js[1])}), native exit {nv[0]} ({refusal(nv[1])})"
    a = json.loads(js[1].strip().splitlines()[-1].replace(base + "/js", "@"))
    b = json.loads(nv[1].strip().splitlines()[-1].replace(base + "/native", "@"))
    if a != b:
        return f"{a} against native {b}"
    for entry in a:
        if "file" in entry:
            fa, fb = entry["file"].replace("@", base + "/js"), entry["file"].replace("@", base + "/native")
            if open(fa, "rb").read() != open(fb, "rb").read():
                return f"{entry['file']} differs"
    return None


def read_idx(path):
    entries = []
    for line in open(path).read().splitlines():
        parts = line.split(" ")
        if parts[0] == "m":
            entries.append(("m", parts[1], " ".join(parts[2:])))
        else:
            entries.append((parts[0], parts[1], int(parts[2]), int(parts[3])))
    return entries


def compare(js_dir, native_dir):
    """None when identical, else a sentence naming the first difference."""
    for name in ("template.pdb", "pdb.template", "search.a3m", "confidence.template.json"):
        a, b = os.path.join(js_dir, name), os.path.join(native_dir, name)
        if os.path.exists(a) != os.path.exists(b):
            return f"{name}: present on one side only"
        if os.path.exists(a) and open(a, "rb").read() != open(b, "rb").read():
            la, lb = open(a).read().splitlines(), open(b).read().splitlines()
            for i, (x, y) in enumerate(zip(la, lb)):
                if x != y:
                    return f"{name} line {i + 1}: {x!r} against native {y!r}"
            return f"{name}: {len(la)} lines against native {len(lb)}"
    ia, ib = read_idx(os.path.join(js_dir, "model.idx")), read_idx(os.path.join(native_dir, "model.idx"))
    ba, bb = open(os.path.join(js_dir, "model.bin"), "rb").read(), open(os.path.join(native_dir, "model.bin"), "rb").read()
    for k, (x, y) in enumerate(zip(ia, ib)):
        if x[0] != y[0] or x[1] != y[1]:
            return f"entry {k}: {x[0]} {x[1]} against native {y[0]} {y[1]}"
        if x[0] == "m":
            if x[2] != y[2]:
                return f"{x[1]}: {x[2]} against native {y[2]}"
            continue
        if x[3] != y[3]:
            return f"{x[1]}: length {x[3]} against native {y[3]}"
        ca, cb = ba[x[2] * 4:(x[2] + x[3]) * 4], bb[y[2] * 4:(y[2] + y[3]) * 4]
        if ca != cb:
            fmt = "f" if x[0] == "t" else "i"
            va, vb = struct.unpack(f"<{x[3]}{fmt}", ca), struct.unpack(f"<{y[3]}{fmt}", cb)
            diff = [i for i in range(x[3]) if struct.pack(f"<{fmt}", va[i]) != struct.pack(f"<{fmt}", vb[i])]
            i = diff[0]
            return f"{x[1]}: {len(diff)} of {x[3]} differ, first [{i}] {va[i]!r} against native {vb[i]!r}"
    if len(ia) != len(ib):
        extra = (ia if len(ia) > len(ib) else ib)[min(len(ia), len(ib))]
        return f"{len(ia)} entries against native {len(ib)} (first unmatched: {extra[1]} on the {'JS' if len(ia) > len(ib) else 'native'} side)"
    return None


def run(cmd, cwd=REPO):
    started = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr, time.time() - started


def refusal(said):
    for line in said.splitlines():
        if line.startswith("Error: ") or "Error: " in line[:40]:
            return line.split("Error: ", 1)[1].strip()
    return said.strip().splitlines()[-1] if said.strip() else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--family", default="")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--network", action="store_true", help="only the cases that search api.colabfold.com")
    parser.add_argument("--ccd", default="", help="a local CCD the native side reads (components.cif)")
    a = parser.parse_args()
    native = os.path.join(REPO, "cuda", "featurise", "af3-featurise")
    if not os.access(native, os.X_OK):
        sys.exit(f"{native} is not built (bash cuda/build.sh featurise)")
    os.makedirs(WORK, exist_ok=True)
    ccd = [f"--ccd={os.path.abspath(a.ccd)}"] if a.ccd else []
    failed, passed, skipped = [], 0, 0
    for case in cases():
        name, port, spec, extra, families = case[:5]
        if a.only and a.only not in name:
            continue
        if (len(case) > 5 and case[5] == "network") != a.network:
            continue
        for family in families:
            if a.family and family != a.family:
                continue
            tag = f"{name}/{family}"
            base = os.path.join(WORK, name.replace("/", "_") + "-" + family)
            shutil.rmtree(base, ignore_errors=True)
            os.makedirs(base)
            if isinstance(spec, str):
                job_path = spec
            else:
                job_path = os.path.join(base, "job.json")
                json.dump(spec, open(job_path, "w"))
            args = [x.replace("@test.a3m", os.path.join(FIX, "test.a3m")).replace("@F/", FIX + "/") for x in extra]
            if port == "bioemu2":
                common = [f"--job={job_path}", *args]
                js_code, js_said, js_s = run([*NODE, os.path.join(REPO, "cuda", "bioemu2", "export_input.mjs"), base + "/js", *common])
                nv_code, nv_said, nv_s = run([native.replace("af3-featurise", "bioemu2-featurise"), base + "/native", *common, *ccd])
                js_said = js_said.replace(base + "/js", "<out>")
                nv_said = nv_said.replace(base + "/native", "<out>")
            elif port == "ef2":
                common = [f"--job={job_path}", *args]
                js_code, js_said, js_s = run([*NODE, os.path.join(REPO, "cuda", "ef2", "export_input.mjs"), base + "/js", *common])
                nv_code, nv_said, nv_s = run([native.replace("af3-featurise", "ef2-featurise"), base + "/native", *common, *ccd])
            elif port == "af2":
                bundle = os.path.join(REPO, "model" if family == "monomer" else "model-multimer")
                common = [f"--bundle={bundle}", f"--job={job_path}", *args]
                js_code, js_said, js_s = run([*NODE, os.path.join(REPO, "cuda", "af2", "export_input.mjs"), base + "/js", *common])
                nv_code, nv_said, nv_s = run([native.replace("af3-featurise", "af2-featurise"), base + "/native", *common])
            else:
                common = ["--no-weights", f"--family={family}", f"--job={job_path}", "--max-msa=512", *args]
                js_code, js_said, js_s = run([*NODE, os.path.join(REPO, "cuda", "af3", "export-model.mjs"), base + "/js", *common],
                                             cwd=os.path.join(REPO, "cuda", "af3"))
                nv_code, nv_said, nv_s = run([native, base + "/native", *common, *ccd])
            if js_code != 0 or nv_code != 0:
                if js_code != 0 and nv_code != 0 and refusal(js_said) == refusal(nv_said):
                    print(f"  ok   {tag}: both refuse - {refusal(js_said)}")
                    passed += 1
                elif js_code != 0 and nv_code != 0:
                    failed.append(tag)
                    print(f"  FAIL {tag}: both refuse, differently\n         JS:     {refusal(js_said)}\n         native: {refusal(nv_said)}")
                else:
                    failed.append(tag)
                    side, said = ("JS", js_said) if js_code != 0 else ("native", nv_said)
                    print(f"  FAIL {tag}: only {side} refuses - {refusal(said)}")
                continue
            problem = compare(base + "/js", base + "/native")
            if problem:
                failed.append(tag)
                print(f"  FAIL {tag}: {problem}")
            else:
                passed += 1
                print(f"  ok   {tag}: identical (JS {js_s:.2f} s, native {nv_s:.2f} s)")
            if not a.keep and not problem:
                shutil.rmtree(base, ignore_errors=True)
    # the SMILES chemistry on its own: tools/fixtures/smiles-corpus.txt - the 74 molecules held to RDKit and 904
    # RDKit re-writings of them, plus malformed strings - one line each, conformer coordinates included
    if not a.network and (not a.only or a.only in "chem-smiles-corpus"):
        corpus = open(os.path.join(FIX, "smiles-corpus.txt")).read()
        js = subprocess.run(["node", os.path.join(REPO, "tools", "chem-probe.mjs")], input=corpus, capture_output=True, text=True)
        nv = subprocess.run([os.path.join(REPO, "cuda", "featurise", "chem-probe")], input=corpus, capture_output=True, text=True)
        la, lb, smiles = js.stdout.splitlines(), nv.stdout.splitlines(), corpus.splitlines()
        bad = [k for k in range(max(len(la), len(lb))) if k >= len(la) or k >= len(lb) or la[k] != lb[k]]
        if bad:
            failed.append("chem-smiles-corpus")
            k = bad[0]
            print(f"  FAIL chem-smiles-corpus: {len(bad)} of {len(smiles)} differ, first {smiles[k] if k < len(smiles) else '?'}")
        else:
            passed += 1
            print(f"  ok   chem-smiles-corpus: {len(la)} of {len(smiles)} identical ({sum(l.startswith('ERR') for l in la)} refusals)")
    for name, request, network in resolve_cases():
        if (a.only and a.only not in name) or bool(network) != a.network:
            continue
        problem = check_resolve(name, request, network)
        if problem:
            failed.append(name)
            print(f"  FAIL {name}: {problem}")
        else:
            passed += 1
            print(f"  ok   {name}: " + ("the native resolver fetched it (no JavaScript reference)" if network == "native-only" else "identical"))
    print(f"\n{passed} identical, {len(failed)} differ" + (f": {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
